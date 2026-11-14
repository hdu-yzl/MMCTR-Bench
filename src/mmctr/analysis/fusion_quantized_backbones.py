"""Legacy QARM/PSRQ fusion studies with injected production quantizers."""

import torch

from mmctr.core import ContractError, ModelOutput
from mmctr.models.common.layers import CrossNetwork, MultiLayerPerceptron
from mmctr.models.mm_models.psrq import MCCA
from mmctr.models.mm_models.qarm import QARM
from .fusion_operators import SEQUENCE_FUSIONS, build_diagnostic_fusion
from .fusion_sequence_backbones import prefixed_losses, projection


class LegacyQuantizedFusion(torch.nn.Module):
    """Keep codebooks and Cross+Deep/joint codes; replace legacy fusion sites."""

    def __init__(self, backbone, method, config):
        super().__init__()
        if not isinstance(backbone, (QARM, MCCA)):
            raise ContractError("quantized fusion requires a validated QARM or PSRQ backbone")
        self.backbone = backbone
        self.is_qarm = isinstance(backbone, QARM)
        self.structure_policy = (
            "legacy_qarm_cross_deep" if self.is_qarm else "legacy_psrq_joint_codes_mean_pooling"
        )
        self.sequential = method in SEQUENCE_FUSIONS
        dim = backbone.projection_dim
        self.history_fusion = build_diagnostic_fusion(method, dim, backbone.feature_names, config)
        self.history_projection = projection(self.history_fusion.output_dim, dim)
        if not self.sequential:
            self.target_fusion = build_diagnostic_fusion(
                method, dim, backbone.feature_names, config
            )
            self.target_projection = projection(self.target_fusion.output_dim, dim)
        # The old PSRQ fusion study uses mean pooling, not its native cross-attention.
        if not self.is_qarm:
            del backbone.history_attention
        width = dim * (2 + len(backbone.user_feature_names) + (not self.is_qarm))
        backbone.dnn = MultiLayerPerceptron(
            width, backbone.mlp_dims, backbone.dropout, batch_norm=backbone.batch_norm
        )
        if self.is_qarm:
            backbone.cross = CrossNetwork(width, backbone.cross_layers)
            backbone.combination = MultiLayerPerceptron(
                width + backbone.mlp_dims[-1],
                backbone.mlp_dims,
                backbone.dropout,
                batch_norm=backbone.batch_norm,
            )

    def _psrq_features(self, batch):
        model = self.backbone
        raw_target, raw_history = model._raw_target(batch), model._raw_history(batch)
        target_codes, joint_codes = model.quantizer.encode_items(raw_target)
        history_codes = model.quantizer.encode_history(raw_history)
        target = {"id": model.embedding(model._target_feature(batch, "id")).squeeze(1)}
        history = {"id": model.embedding(batch.history_features["id"])}
        target_presence, history_presence = {}, {"id": batch.history_mask}
        for name in model.quantized_modalities:
            target_presence[name] = model._present(raw_target[name])
            history_presence[name] = model._present(raw_history[name]) & batch.history_mask
            target[name] = model._embed_codes(name, target_codes[name]) * (
                target_presence[name].unsqueeze(-1)
            )
            history[name] = model._embed_codes(name, history_codes[name]) * (
                history_presence[name].unsqueeze(-1)
            )
        target = model.projectors(target, target_presence)
        history = model.projectors(history, history_presence)
        joint = torch.cat(
            [model.joint_embeddings[i](joint_codes[:, i]) for i in range(model.n_levels)], -1
        )
        present = torch.stack(list(target_presence.values()), -1).any(-1).unsqueeze(-1)
        joint = model.joint_projector(joint * present) * present
        return target, history, joint

    def forward(self, batch):
        model = self.backbone
        if self.is_qarm:
            target = model._project_quantized_target(batch)
            history = model._project_quantized_history(batch)
            joint = None
        else:
            target, history, joint = self._psrq_features(batch)
        if self.sequential:
            fused = self.history_fusion(target, history, batch.history_mask)
            history_vector = self.history_projection(fused.fused)
            target_vector = target["id"]
            losses = dict(fused.auxiliary_losses)
        else:
            pooled = {n: model.masked_pool(v, batch.history_mask) for n, v in history.items()}
            fused = self.history_fusion(pooled)
            history_vector = self.history_projection(fused.fused)
            losses = prefixed_losses(fused, "history_")
            fused = self.target_fusion(target)
            target_vector = self.target_projection(fused.fused)
            losses.update(prefixed_losses(fused, "target_"))
        values = [history_vector, target_vector]
        if joint is not None:
            values.append(joint)
        values.append(model.project_user(batch))
        inputs = torch.cat(values, -1)
        hidden = model.dnn(inputs)
        if self.is_qarm:
            hidden = model.combination(torch.cat([model.cross(inputs), hidden], -1))
        return ModelOutput(model.output(hidden), auxiliary_losses=losses)
