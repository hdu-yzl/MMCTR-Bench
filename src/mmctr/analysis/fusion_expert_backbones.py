"""Legacy MARN and Diff-MSIN fusion studies retaining native expert losses."""

import torch

from mmctr.core import ModelOutput
from mmctr.models.mm_models.marn import MARN
from mmctr.models.mm_models.diff_msin import Diff_MSIN, _hinge_cosine, _mean_pairwise_cosine
from .fusion_operators import SEQUENCE_FUSIONS, build_diagnostic_fusion
from .fusion_sequence_backbones import projection, prefixed_losses


class LegacyMARNFusion(MARN):
    def __init__(self, method, config, data):
        super().__init__(config, data)
        del self.private_fusion
        self.structure_policy = "legacy_marn_specific_invariant"
        self.method = method
        self.sequential = method in SEQUENCE_FUSIONS
        self.study_fusion = build_diagnostic_fusion(
            method, self.projection_dim, self.feature_names, config
        )
        self.fusion_projection = projection(self.study_fusion.output_dim, self.projection_dim)
        if self.sequential:
            del self.attention

    def forward_batch(self, batch):
        target_s, target_c = self.split(self.project_target(batch))
        history_s, history_c = self.split(self.project_history(batch))
        history_s = self._apply_mask(history_s, batch.history_mask)
        history_c = self._apply_mask(history_c, batch.history_mask)
        domain, adversarial, invariant = self.adversarial(target_c)
        losses = {
            "marn_domain_classifier": domain,
            "marn_adversarial_invariance": adversarial * self.auxiliary_weight,
            "marn_specific_classifier": self.specific_classifier(target_s) * self.auxiliary_weight,
        }
        if self.sequential:
            fused = self.study_fusion(target_s, history_s, batch.history_mask)
            interest = self.fusion_projection(fused.fused)
            if self.method == "dta":
                target_vector = invariant + interest
                # Preserve the legacy two-call DTA branch (including dropout semantics).
                interest = self.fusion_projection(
                    self.study_fusion(target_s, history_s, batch.history_mask).fused
                )
            else:
                target_vector = invariant + target_s["id"]
            losses.update(fused.auxiliary_losses)
        else:
            fused = self.study_fusion(target_s)
            target_vector = invariant + self.fusion_projection(fused.fused)
            losses.update(fused.auxiliary_losses)
            sequence = torch.stack(
                [
                    self.fusion_projection(
                        self.study_fusion({n: history_s[n][:, i] for n in self.feature_names}).fused
                    )
                    for i in range(batch.sequence_length)
                ],
                1,
            )
            sequence = sequence + self._history_invariant(history_c, batch)
            interest = self.attention(sequence, batch.history_mask, target_vector)
        values = [self.project_user(batch), interest, target_vector]
        return ModelOutput(self.output(self.dnn(torch.cat(values, -1))), auxiliary_losses=losses)


class LegacyDiffFusion(Diff_MSIN):
    def __init__(self, method, config, data):
        super().__init__(config, data)
        del self.src_fusion
        if method == "mtfn":
            from .numerical_stability import widen_batch_norm

            widen_batch_norm(self.dnn)
        self.structure_policy = "legacy_diff_msin_experts_and_two_level_gates"
        self.sequential = method in SEQUENCE_FUSIONS
        self.study_fusion = build_diagnostic_fusion(
            method, self.projection_dim, self.feature_names, config
        )
        self.fusion_projection = projection(self.study_fusion.output_dim, self.projection_dim)

    def forward_batch(self, batch):
        target, history = self.project_target(batch), self.project_history(batch)
        ts, ss, tc, sc = self._expert_outputs(target, history, batch.history_mask)
        tmean, smean = sum(tc.values()) / len(tc), sum(sc.values()) / len(sc)
        contrastive = (
            _mean_pairwise_cosine(ts, self.feature_names)
            - _mean_pairwise_cosine(tc, self.feature_names)
            + _mean_pairwise_cosine(ss, self.feature_names)
            - _mean_pairwise_cosine(sc, self.feature_names)
        ) / 2
        target_mixed, sequence_mixed = {}, {}
        for name in self.feature_names:
            gate = self.modal_gates[name](ts[name])
            target_mixed[name] = gate * ts[name] + (1 - gate) * tmean
            gate = self.modal_gates[name](ss[name])
            sequence_mixed[name] = gate * ss[name] + (1 - gate) * smean
        if self.sequential:
            # Legacy study applies sequence fusion to the already pooled one-step sequence.
            pseudo = {n: v.unsqueeze(1) for n, v in sequence_mixed.items()}
            mask = batch.history_mask.any(1, keepdim=True)
            target_out = self.study_fusion(target_mixed, pseudo, mask)
            history_out = self.study_fusion(sequence_mixed, pseudo, mask)
        else:
            target_out = self.study_fusion(target_mixed)
            history_out = self.study_fusion(sequence_mixed)
        target_synthesis = self.fusion_projection(target_out.fused)
        history_synthesis = self.fusion_projection(history_out.fused)
        synthesis = _hinge_cosine(target_synthesis, history_synthesis, batch.labels)
        gate_values = self.final_gate(target_mixed["id"]).view(
            batch.batch_size, self.gated_entry_count, self.projection_dim
        )
        names = tuple(n for n in self.feature_names if n != "id") + ("share", "synthesis")
        gates = {n: gate_values[:, i] for i, n in enumerate(names)}
        entries = [gates[n] * sequence_mixed[n] for n in self.feature_names if n != "id"]
        entries += [gates["share"] * smean, gates["synthesis"] * history_synthesis]
        crossed = self.cross_network(torch.stack(entries, 1).reshape(batch.batch_size, -1))
        mean = crossed.view(batch.batch_size, self.gated_entry_count, self.projection_dim).mean(1)
        attended = self.id_attention(sequence_mixed["id"], mean)
        logits = self.output(self.dnn(torch.cat([crossed, attended, self.project_user(batch)], -1)))
        losses = {
            "diff_msin_synthesis": synthesis * self.synthesis_weight,
            "diff_msin_contrastive": contrastive * self.contrastive_weight,
        }
        losses.update(prefixed_losses(target_out, "target_"))
        losses.update(prefixed_losses(history_out, "history_"))
        return ModelOutput(logits, auxiliary_losses=losses)
