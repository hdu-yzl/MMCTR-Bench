"""Legacy MB/PAMD fusion substitutions preserving their native auxiliary branches."""

from itertools import combinations
import torch
import torch.nn.functional as F

from mmctr.core import ModelOutput
from mmctr.models.mm_models.mb import MB
from mmctr.models.mm_models.pamd import PAMD
from .fusion_operators import SEQUENCE_FUSIONS, build_diagnostic_fusion
from .fusion_sequence_backbones import prefixed_losses, projection


class LegacyMBFusion(MB):
    def __init__(self, method, config, data):
        super().__init__(config, data)
        del self.fusion
        self.structure_policy = "legacy_mb_scores_and_balance"
        self.sequential = method in SEQUENCE_FUSIONS
        names = self.feature_names if self.sequential else self.history_modal_names
        self.history_fusion = build_diagnostic_fusion(method, self.projection_dim, names, config)
        self.history_projection = projection(self.history_fusion.output_dim, self.projection_dim)
        if not self.sequential:
            self.target_fusion = build_diagnostic_fusion(
                method, self.projection_dim, self.modal_names, config
            )
            self.target_projection = projection(self.target_fusion.output_dim, self.projection_dim)

    def forward_batch(self, batch):
        target, history = self.project_target(batch), self.project_history(batch)
        raw_user, user_id, item_id, pair_id = self._raw_ids(batch)
        modal_values = {n: self.item_modal_encoders[n](target[n]) for n in self.modal_names}
        if self.sequential:
            fused = self.history_fusion(target, history, batch.history_mask)
            history_vector = self.history_projection(fused.fused)
            target_vector = torch.stack(list(modal_values.values()), 1).mean(1)
            losses = dict(fused.auxiliary_losses)
        else:
            fused = self.target_fusion(modal_values)
            target_vector = self.target_projection(fused.fused)
            losses = prefixed_losses(fused, "target_")
            values = {
                n: self.item_modal_encoders[n](self.masked_pool(history[n], batch.history_mask))
                for n in self.history_modal_names
            }
            fused = self.history_fusion(values)
            history_vector = self.history_projection(fused.fused)
            losses.update(prefixed_losses(fused, "history_"))
        id_score = (F.normalize(user_id, dim=-1) * F.normalize(item_id, dim=-1)).sum(
            -1, keepdim=True
        )
        scores = [self._modal_score(raw_user, modal_values, n) for n in self.modal_names]
        inputs = torch.cat(
            [pair_id, user_id, item_id, target_vector, history_vector, user_id * target_vector], -1
        )
        logits = self.output(self.dnn(inputs)) + id_score + sum(scores, id_score.new_zeros(()))
        losses["mb_modality_balance"] = (
            self._balance_loss(batch, raw_user, modal_values)
            if self.training and self._can_balance(modal_values)
            else logits.new_zeros(())
        )
        return ModelOutput(logits, auxiliary_losses=losses)


class LegacyPAMDFusion(PAMD):
    def __init__(self, method, config, data):
        super().__init__(config, data)
        self.structure_policy = "legacy_pamd_pair_disentanglement"
        self.sequential = method in SEQUENCE_FUSIONS
        target_names = tuple(self.target_blocks)
        self.target_fusion = build_diagnostic_fusion(
            "cat" if self.sequential else method, self.projection_dim, target_names, config
        )
        self.target_projection = projection(self.target_fusion.output_dim, self.projection_dim)
        history_names = (
            self.history_feature_names if self.sequential else tuple(self.history_blocks)
        )
        self.history_fusion = build_diagnostic_fusion(
            method, self.projection_dim, history_names, config
        )
        self.history_projection = projection(self.history_fusion.output_dim, self.projection_dim)
        if self.sequential:
            del self.history_blocks

    def _pairs(self, blocks, names, values):
        representations, losses = {}, []
        for first, second in combinations(names, 2):
            key = self._pair_key(first, second)
            fused, loss = blocks[key](values[first], values[second], values["id"])
            representations[key] = fused
            losses.append(loss)
        return representations, torch.stack(losses).mean()

    def forward_batch(self, batch):
        target, history = self.project_target(batch), self.project_history(batch)
        pairs, auxiliary = self._pairs(self.target_blocks, self.modal_names, target)
        fused = self.target_fusion(pairs)
        target_vector = self.target_projection(fused.fused)
        auxiliary = auxiliary + fused.total_auxiliary_loss()
        if self.sequential:
            fused = self.history_fusion(target, history, batch.history_mask)
        else:
            pooled = {n: self.masked_pool(v, batch.history_mask) for n, v in history.items()}
            pairs, history_loss = self._pairs(self.history_blocks, self.history_modal_names, pooled)
            auxiliary = auxiliary + history_loss
            fused = self.history_fusion(pairs)
        auxiliary = auxiliary + fused.total_auxiliary_loss()
        history_vector = self.history_projection(fused.fused)
        logits = self.output(self.dnn(torch.cat([target_vector, history_vector], -1)))
        return ModelOutput(
            logits,
            auxiliary_losses={"pamd_disentanglement_and_fusion": self.auxiliary_weight * auxiliary},
        )
