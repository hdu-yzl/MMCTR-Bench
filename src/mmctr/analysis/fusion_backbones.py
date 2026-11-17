"""Explicit legacy fusion-study model adaptations, separate from main-table models."""

from typing import Mapping

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.layers import MultiLayerPerceptron
from mmctr.models.common.sequence import _SequenceMultimodalModel
from .fusion_operators import SEQUENCE_FUSIONS, build_diagnostic_fusion

FORCED_WRAPPER_MODELS = frozenset({"lmf", "mtfn", "gmmf", "simcen"})


class LegacyFusionWrapper(_SequenceMultimodalModel):
    """Reproduce the old shared wrapper for four 'fusion-is-model' variants.

    This deliberately replaces the original architecture, as the AntM2C fusion
    experiment did. It must never be registered as a main-table implementation.
    """

    def __init__(
        self, model_name: str, method: str, model_config: Mapping, data_config: Mapping
    ) -> None:
        if model_name not in FORCED_WRAPPER_MODELS:
            raise ContractError("this legacy wrapper only applies to four documented models")
        super().__init__(model_config, data_config)
        self.study_model = model_name
        self.fusion_method = method
        self.structure_policy = "legacy_forced_shared_wrapper"
        self.sequence_fusion = method in SEQUENCE_FUSIONS
        self.target_fusion = build_diagnostic_fusion(
            method, self.projection_dim, self.feature_names, model_config
        )
        dimension = self.target_fusion.output_dim
        if self.sequence_fusion:
            self.fusion_projection = (
                torch.nn.Linear(dimension, self.projection_dim)
                if dimension != self.projection_dim
                else torch.nn.Identity()
            )
            predictor_dim = self.projection_dim
        else:
            self.history_fusion = build_diagnostic_fusion(
                method, self.projection_dim, self.feature_names, model_config
            )
            predictor_dim = dimension * 2
        predictor_dim += self.projection_dim * len(self.user_feature_names)
        self.dnn = MultiLayerPerceptron(
            predictor_dim, self.mlp_dims, self.dropout, batch_norm=self.batch_norm
        )
        self.output = MultiLayerPerceptron(
            self.mlp_dims[-1], [1], self.dropout, batch_norm=self.batch_norm, activation=None
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target, history = self.project_target(batch), self.project_history(batch)
        if self.sequence_fusion:
            fused = self.target_fusion(target, history, batch.history_mask)
            values = [self.fusion_projection(fused.fused), self.project_user(batch)]
            losses = dict(fused.auxiliary_losses)
        else:
            target_fused = self.target_fusion(target)
            pooled = {
                name: self.masked_pool(value, batch.history_mask) for name, value in history.items()
            }
            history_fused = self.history_fusion(pooled)
            values = [target_fused.fused, history_fused.fused, self.project_user(batch)]
            losses = {
                "target_" + name: value for name, value in target_fused.auxiliary_losses.items()
            }
            losses.update(
                {"history_" + name: value for name, value in history_fused.auxiliary_losses.items()}
            )
        logits = self.output(self.dnn(torch.cat(values, -1)))
        return ModelOutput(logits, auxiliary_losses=losses)


class LegacySimilarityFusion(_SequenceMultimodalModel):
    """Old DMF/MAKE fusion substitutions with their similarity-tier branch intact."""

    def __init__(
        self, model_name: str, method: str, model_config: Mapping, data_config: Mapping
    ) -> None:
        from mmctr.models.common.layers import DinAttention
        from mmctr.models.common.similarity import SimilarityTiers
        from mmctr.models.mm_models.dmf import _DecoupledTargetAttention

        if model_name not in {"dmf", "make"}:
            raise ContractError("similarity fusion adapter only supports DMF and MAKE")
        super().__init__(model_config, data_config)
        self.study_model, self.fusion_method = model_name, method
        self.structure_policy = "legacy_preserve_similarity_tiers"
        self.sequence_fusion = method in SEQUENCE_FUSIONS
        self.non_id = tuple(n for n in self.feature_names if n != "id")
        names = (
            self.non_id if model_name == "dmf" and not self.sequence_fusion else self.feature_names
        )
        self.study_fusion = build_diagnostic_fusion(
            method, self.projection_dim, names, model_config
        )
        fusion_dim = self.study_fusion.output_dim
        tier_count = int(model_config.get("tier_num", 10))
        self.tiers = SimilarityTiers(tier_count)
        user_dim = self.projection_dim * len(self.user_feature_names)
        if model_name == "dmf":
            self.alpha = float(model_config.get("alpha", 0.5))
            if not 0 <= self.alpha <= 1:
                raise ContractError("DMF alpha must be in [0,1]")
            self.center_mlp = MultiLayerPerceptron(
                tier_count, self.mlp_dims, self.dropout, batch_norm=self.batch_norm
            )
            if self.sequence_fusion:
                self.interest_projection = (
                    torch.nn.Linear(fusion_dim, self.mlp_dims[-1])
                    if fusion_dim != self.mlp_dims[-1]
                    else torch.nn.Identity()
                )
            else:
                attention_dim = int(model_config.get("attention_dim", 128))
                self.attention = _DecoupledTargetAttention(
                    self.projection_dim,
                    attention_dim,
                    int(model_config.get("num_buckets", 35)),
                    self.dropout,
                )
                self.enhanced_mlp = MultiLayerPerceptron(
                    attention_dim, self.mlp_dims, self.dropout, batch_norm=self.batch_norm
                )
            predictor_dim = user_dim + self.mlp_dims[-1]
        else:
            if self.sequence_fusion:
                self.interest_projection = (
                    torch.nn.Linear(fusion_dim, self.projection_dim)
                    if fusion_dim != self.projection_dim
                    else torch.nn.Identity()
                )
                predictor_dim = (
                    user_dim
                    + self.projection_dim
                    + self.projection_dim * len(self.feature_names)
                    + tier_count
                )
            else:
                self.attention = DinAttention(fusion_dim, self.mlp_dims, self.dropout)
                predictor_dim = user_dim + 2 * fusion_dim + tier_count
        self.dnn = MultiLayerPerceptron(
            predictor_dim, self.mlp_dims, self.dropout, batch_norm=self.batch_norm
        )
        self.output = MultiLayerPerceptron(
            self.mlp_dims[-1], [1], self.dropout, batch_norm=self.batch_norm, activation=None
        )

    @staticmethod
    def _similarity(target, history, names):
        return torch.nn.functional.cosine_similarity(
            torch.cat([target[n] for n in names], -1).unsqueeze(1),
            torch.cat([history[n] for n in names], -1),
            dim=-1,
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target, history = self.project_target(batch), self.project_history(batch)
        user = self.project_user(batch)
        losses = {}
        if self.sequence_fusion:
            fused = self.study_fusion(target, history, batch.history_mask)
            interest = self.interest_projection(fused.fused)
            losses.update(fused.auxiliary_losses)
            names = self.non_id if self.study_model == "dmf" else self.feature_names
            similarities = self._similarity(target, history, names)
            target_fused = torch.cat([target[n] for n in self.feature_names], -1)
        else:
            names = self.non_id if self.study_model == "dmf" else self.feature_names
            fused = self.study_fusion({n: target[n] for n in names})
            target_fused = fused.fused
            losses.update(fused.auxiliary_losses)
            # The old experiments add only target-side SimCEN auxiliary loss here.
            history_fused = torch.stack(
                [
                    self.study_fusion({n: history[n][:, i] for n in names}).fused
                    for i in range(batch.history_mask.shape[1])
                ],
                1,
            )
            local_sim = torch.nn.functional.cosine_similarity(
                target_fused.unsqueeze(1), history_fused, dim=-1
            )
            if self.study_model == "dmf":
                interest = self.enhanced_mlp(
                    self.attention(target["id"], history["id"], local_sim, batch.history_mask)
                )
                # Legacy fusion study fixes the center branch to concatenation.
                similarities = self._similarity(target, history, self.non_id)
            else:
                interest = self.attention(history_fused, batch.history_mask, target_fused)
                similarities = local_sim
        tiers = self.tiers(similarities, batch.history_mask)
        if self.study_model == "dmf":
            interest = self.alpha * interest + (1 - self.alpha) * self.center_mlp(tiers)
            values = [user, interest]
        else:
            values = [user, interest, target_fused, tiers]
        return ModelOutput(
            self.output(self.dnn(torch.cat(values, -1))),
            auxiliary_losses=losses,
            representations={"similarity_tiers": tiers},
        )


class LegacyAttentionFusion(_SequenceMultimodalModel):
    """Preserve NAML dot-product attention or EM3 content/ID contrastive learning."""

    def __init__(
        self, model_name: str, method: str, model_config: Mapping, data_config: Mapping
    ) -> None:
        from mmctr.models.common.components import AttentionPooling
        from mmctr.models.common.layers import DinAttention

        if model_name not in {"naml", "em3"}:
            raise ContractError("attention fusion adapter only supports NAML and EM3")
        super().__init__(model_config, data_config)
        self.study_model, self.fusion_method = model_name, method
        self.structure_policy = "legacy_preserve_attention_or_cic"
        self.sequence_fusion = method in SEQUENCE_FUSIONS
        self.study_fusion = build_diagnostic_fusion(
            method, self.projection_dim, self.feature_names, model_config
        )
        width = self.study_fusion.output_dim
        user_width = self.projection_dim * len(self.user_feature_names)
        self.fusion_projection = (
            torch.nn.Linear(width, self.projection_dim)
            if width != self.projection_dim
            else torch.nn.Identity()
        )
        if model_name == "naml":
            self.user_linear = torch.nn.Linear(user_width, self.projection_dim)
            if not self.sequence_fusion:
                self.attention = AttentionPooling(self.projection_dim)
        else:
            self.cic_temperature = float(model_config.get("cic_tau", 0.1))
            self.cic_weight = float(model_config.get("cic_weight", 0.1))
            if self.cic_temperature <= 0 or self.cic_weight < 0:
                raise ContractError("invalid EM3 contrastive configuration")
            if self.sequence_fusion:
                self.content_fusion = build_diagnostic_fusion(
                    "maf", self.projection_dim, self.feature_names, model_config
                )
                content_width = self.projection_dim
                interest_width = self.projection_dim
            else:
                self.attention = DinAttention(width, self.mlp_dims, self.dropout)
                content_width = interest_width = width
                # Local EM3 uses the full fused dimension, with no extra projection.
                del self.fusion_projection
            self.content_map = MultiLayerPerceptron(
                content_width, [self.projection_dim], self.dropout, batch_norm=self.batch_norm
            )
            self.dnn = MultiLayerPerceptron(
                user_width + interest_width + self.projection_dim,
                self.mlp_dims,
                self.dropout,
                batch_norm=self.batch_norm,
            )
            self.output = MultiLayerPerceptron(
                self.mlp_dims[-1], [1], self.dropout, batch_norm=self.batch_norm, activation=None
            )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        from mmctr.models.mm_models.em3 import EM3

        target, history = self.project_target(batch), self.project_history(batch)
        if self.sequence_fusion:
            fused = self.study_fusion(target, history, batch.history_mask)
            interest = self.fusion_projection(fused.fused)
            losses = dict(fused.auxiliary_losses)
            target_vector = target["id"]
            if self.study_model == "em3":
                target_vector = self.content_fusion(target).fused
        else:
            fused = self.study_fusion(target)
            losses = dict(fused.auxiliary_losses)
            target_vector = fused.fused
            sequence = torch.stack(
                [
                    self.study_fusion({n: history[n][:, i] for n in self.feature_names}).fused
                    for i in range(batch.sequence_length)
                ],
                1,
            )
            if self.study_model == "naml":
                target_vector = self.fusion_projection(target_vector)
                sequence = self.fusion_projection(sequence)
                interest = self.attention(sequence, batch.history_mask)
            else:
                interest = self.attention(sequence, batch.history_mask, target_vector)
        user = self.project_user(batch)
        if self.study_model == "naml":
            logits = (target_vector * (self.user_linear(user) + interest)).sum(-1)
        else:
            content = self.content_map(target_vector)
            losses["em3_content_item_contrastive"] = self.cic_weight * EM3._content_item_loss(
                self, content, target["id"]
            )
            logits = self.output(self.dnn(torch.cat([user, interest, content], -1)))
        return ModelOutput(logits, auxiliary_losses=losses)


def build_fusion_study_model(
    model_name, method, model_config, data_config, quantization_artifact_dir=None
):
    """Construct only explicitly implemented legacy adaptations; never fall back."""
    from .fusion_sequence_backbones import LegacyM3Fusion, LegacyMixerFusion

    if model_name in {"qarm", "psrq"}:
        from mmctr.models.common.registry import create_model_from_artifacts
        from .fusion_quantized_backbones import LegacyQuantizedFusion

        if quantization_artifact_dir is None:
            raise ContractError("quantized fusion requires frozen dataset quantizer artifacts")
        backbone = create_model_from_artifacts(
            model_name, model_config, data_config, quantization_artifact_dir
        )
        return LegacyQuantizedFusion(backbone, method, model_config)
    if model_name in FORCED_WRAPPER_MODELS:
        return LegacyFusionWrapper(model_name, method, model_config, data_config)
    if model_name in {"dmf", "make"}:
        return LegacySimilarityFusion(model_name, method, model_config, data_config)
    if model_name in {"naml", "em3"}:
        return LegacyAttentionFusion(model_name, method, model_config, data_config)
    if model_name in {"mb", "pamd"}:
        from .fusion_pooled_backbones import LegacyMBFusion, LegacyPAMDFusion

        cls = LegacyMBFusion if model_name == "mb" else LegacyPAMDFusion
        return cls(method, model_config, data_config)
    if model_name in {"marn", "diff_msin"}:
        from .fusion_expert_backbones import LegacyMARNFusion, LegacyDiffFusion

        cls = LegacyMARNFusion if model_name == "marn" else LegacyDiffFusion
        return cls(method, model_config, data_config)
    if model_name == "mmmlp":
        return LegacyMixerFusion(method, model_config, data_config)
    if model_name == "m3srec":
        return LegacyM3Fusion(method, model_config, data_config)
    raise ContractError("legacy fusion adapter is not yet implemented for " + model_name)
