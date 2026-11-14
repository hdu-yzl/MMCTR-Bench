"""Legacy MMMLP/M3SRec fusion interventions with production sequence components."""

from typing import Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.layers import MultiLayerPerceptron
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.common.sequence_utils import last_valid_token
from mmctr.models.mm_models.m3srec import M3SRec
from mmctr.models.mm_models.mmmlp import _MixerModule
from .fusion_operators import SEQUENCE_FUSIONS, build_diagnostic_fusion


def projection(source, target):
    return torch.nn.Linear(source, target) if source != target else torch.nn.Identity()


def prefixed_losses(output, prefix):
    return {prefix + name: value for name, value in output.auxiliary_losses.items()}


class LegacyMixerFusion(_SequenceMultimodalModel):
    """Legacy MMMLP: modality mixers and temporal fusion mixer for local operators."""

    def __init__(self, method: str, config: Mapping, data: Mapping) -> None:
        super().__init__(config, data)
        self.structure_policy = "legacy_mmmlp_mixers"
        self.sequence_length = int(data["seq_len"])
        self.sequential = method in SEQUENCE_FUSIONS
        self.history_fusion = build_diagnostic_fusion(
            method, self.projection_dim, self.feature_names, config
        )
        self.history_projection = projection(self.history_fusion.output_dim, self.projection_dim)
        if not self.sequential:
            self.target_fusion = build_diagnostic_fusion(
                method, self.projection_dim, self.feature_names, config
            )
            self.target_projection = projection(self.target_fusion.output_dim, self.projection_dim)
            token_dim = int(config.get("token_hidden_dim", self.sequence_length * 2))
            channel_dim = int(config.get("channel_hidden_dim", self.projection_dim * 2))
            self.mixers = torch.nn.ModuleDict(
                {
                    name: _MixerModule(
                        self.sequence_length,
                        self.projection_dim,
                        int(config.get("feature_mixer_layers", config.get("mixer_layers", 2))),
                        token_dim,
                        channel_dim,
                        self.dropout,
                    )
                    for name in self.feature_names
                }
            )
            self.fusion_mixer = _MixerModule(
                self.sequence_length,
                self.projection_dim,
                int(config.get("fusion_mixer_layers", config.get("mixer_layers", 2))),
                token_dim,
                int(config.get("fusion_channel_hidden_dim", self.projection_dim * 2)),
                self.dropout,
            )
            self.target_post = torch.nn.Sequential(
                torch.nn.LayerNorm(self.projection_dim),
                torch.nn.Linear(self.projection_dim, self.projection_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(self.dropout),
            )
        width = self.projection_dim * (3 + len(self.user_feature_names))
        self.dnn = MultiLayerPerceptron(
            width, self.mlp_dims, self.dropout, batch_norm=self.batch_norm
        )
        self.output = MultiLayerPerceptron(
            self.mlp_dims[-1], [1], self.dropout, batch_norm=self.batch_norm, activation=None
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        if batch.sequence_length != self.sequence_length:
            raise ContractError("MMMLP fusion sequence length differs from configuration")
        target, history = self.project_target(batch), self.project_history(batch)
        losses = {}
        if self.sequential:
            fused = self.history_fusion(target, history, batch.history_mask)
            interest = self.history_projection(fused.fused)
            target_vector = target["id"]
            losses.update(fused.auxiliary_losses)
        else:
            mixed = {n: self.mixers[n](history[n], batch.history_mask) for n in self.feature_names}
            steps = []
            for i in range(batch.sequence_length):
                fused = self.history_fusion({n: mixed[n][:, i] for n in self.feature_names})
                steps.append(self.history_projection(fused.fused))
                losses.update(prefixed_losses(fused, "step_{}_".format(i)))
            sequence = self.fusion_mixer(torch.stack(steps, 1), batch.history_mask)
            interest = last_valid_token(sequence, batch.history_mask)
            fused = self.target_fusion(target)
            target_vector = self.target_post(self.target_projection(fused.fused))
            losses.update(prefixed_losses(fused, "target_"))
        values = [interest, target_vector, interest * target_vector, self.project_user(batch)]
        return ModelOutput(self.output(self.dnn(torch.cat(values, -1))), auxiliary_losses=losses)


class LegacyM3Fusion(M3SRec):
    """Keep M3SRec shared attention and experts; substitute its two fusion sites."""

    def __init__(self, method: str, config: Mapping, data: Mapping) -> None:
        super().__init__(config, data)
        self.structure_policy = "legacy_m3srec_shared_attention_and_experts"
        self.sequential = method in SEQUENCE_FUSIONS
        self.history_fusion = build_diagnostic_fusion(
            method, self.projection_dim, self.modal_names, config
        )
        self.history_projection = projection(self.history_fusion.output_dim, self.projection_dim)
        if self.sequential:
            del self.target_fusion
        else:
            self.target_fusion = build_diagnostic_fusion(
                method, self.projection_dim, self.modal_names, config
            )
            self.target_projection = projection(self.target_fusion.output_dim, self.projection_dim)

    def _refined_history(self, history, mask):
        sequences = self._add_embeddings([history[n] for n in self.modal_names])
        full_mask = torch.cat([~mask for _ in self.modal_names], 1)
        values = self.input_attention(torch.cat(tuple(sequences), 1), full_mask)
        refined = []
        for name, chunk in zip(self.modal_names, self._split_modalities(values, mask.shape[1])):
            for expert in self.specific_experts[name]:
                chunk = expert(chunk)
            refined.append(chunk)
        values = torch.cat(refined, 1)
        for attention, expert in zip(self.cross_attention, self.cross_experts):
            values = expert(attention(values, full_mask))
        return dict(zip(self.modal_names, self._split_modalities(values, mask.shape[1])))

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target_all = self.project_target(batch)
        target = {n: target_all[n] for n in self.modal_names}
        history = self._refined_history(self.project_history(batch), batch.history_mask)
        if self.sequential:
            fused = self.history_fusion(target, history, batch.history_mask)
            interest = self.history_projection(fused.fused)
            target_vector = target["id"]
            losses = dict(fused.auxiliary_losses)
        else:
            pooled = {n: last_valid_token(v, batch.history_mask) for n, v in history.items()}
            fused = self.history_fusion(pooled)
            interest = self.history_projection(fused.fused)
            losses = prefixed_losses(fused, "history_")
            fused = self.target_fusion(target)
            target_vector = self.target_projection(fused.fused)
            losses.update(prefixed_losses(fused, "target_"))
        values = [interest, target_vector, self.project_user(batch)]
        return ModelOutput(self.output(self.dnn(torch.cat(values, -1))), auxiliary_losses=losses)
