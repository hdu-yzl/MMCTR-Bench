"""Nine paper fusion operators reusing canonical model components."""

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from mmctr.core import ContractError
from mmctr.models.common.components.fusion import FusionOutput
from mmctr.models.common.layers import MultiLayerPerceptron
from mmctr.models.common.multimodal import _build_fusion
from mmctr.models.common.similarity import SimilarityTiers
from mmctr.models.mm_models.diff_msin import _StochasticReverseFusion
from mmctr.models.mm_models.dmf import _DecoupledTargetAttention
from mmctr.models.mm_models.em3 import _FQFormer
from mmctr.models.mm_models.simcen import _MultiLevelExpert, _Segmentation

LOCAL_FUSIONS = ("cat", "maf", "lmf", "mtfn", "src", "fq-former", "simcen")
SEQUENCE_FUSIONS = ("dta", "dmf")
PAPER_FUSIONS = LOCAL_FUSIONS + SEQUENCE_FUSIONS


class DiagnosticLocalFusion(torch.nn.Module):
    """Return the legacy local fusion representation and any auxiliary loss."""

    def __init__(self, method: str, dimension: int, names: Sequence[str], config: Mapping) -> None:
        super().__init__()
        if method not in LOCAL_FUSIONS or not names:
            raise ContractError("unsupported local diagnostic fusion")
        self.method, self.names = method, tuple(names)
        self.dimension = dimension
        if method == "fq-former":
            self.layer = _FQFormer(
                dimension,
                int(config.get("query_num", 3)),
                int(config.get("fq_layer_num", 2)),
                int(config.get("fq_num_heads", 8)),
            )
            self.output_dim = self.layer.output_dim
        elif method == "src":
            self.layer = _StochasticReverseFusion(dimension, names, int(config.get("T", 10)))
            self.output_dim = dimension
        elif method == "simcen":
            hidden = tuple(config.get("simcen_hidden_units", [480, 480, 480]))
            if not hidden:
                raise ContractError("SimCEN fusion needs hidden layers")
            flat = len(names) * dimension
            self.segmentation = _Segmentation(len(names), dimension, flat)
            self.layer = _MultiLevelExpert(flat * 3, hidden, (0.0, 0.0, 0.0), (True, True, True))
            self.temperature = float(config.get("simcen_cl_temp", 0.1))
            if self.temperature <= 0:
                raise ContractError("SimCEN temperature must be positive")
            self.output_dim = hidden[-1]
        else:
            self.layer = _build_fusion(
                method,
                names,
                dimension,
                rank=int(config.get("rank", 20 if method == "mtfn" else 5)),
                output_dim=int(config.get("fusion_dim", dimension)),
            )
            self.output_dim = self.layer.output_dim

    def forward(self, features: Mapping[str, torch.Tensor]) -> FusionOutput:
        if set(features) != set(self.names):
            raise ContractError("local fusion modalities differ from configured names")
        shape = features[self.names[0]].shape
        if len(shape) != 2 or shape[-1] != self.dimension:
            raise ContractError("local diagnostic fusion requires [batch, dimension]")
        if any(v.shape != shape for v in features.values()):
            raise ContractError("local fusion shapes must agree")
        if self.method == "fq-former":
            return FusionOutput(self.layer(torch.stack([features[n] for n in self.names], 1)))
        if self.method == "simcen":
            values = torch.stack([features[n] for n in self.names], 1)
            parts = self.segmentation(values)
            ego, v1, v2 = self.layer(torch.cat(parts, -1)).chunk(3, -1)
            v1, v2 = ego + v1, ego + v2
            fused = torch.cat([ego, v1, v2], -1)
            ego, v1, v2 = [F.normalize(v, dim=-1) for v in (ego, v1, v2)]
            positive = ((ego * v1).sum(-1) + (ego * v2).sum(-1)) / 2
            log_ratio = positive / self.temperature - torch.logsumexp(
                v1 @ v2.t() / self.temperature, dim=-1
            )
            # Stable form of legacy -log(exp(log_ratio) + 1e-6).
            loss = -torch.logaddexp(log_ratio, log_ratio.new_full((), -13.815510557964274))
            return FusionOutput(fused, {"simcen_contrastive": loss.mean()})
        result = self.layer(features)
        return result if isinstance(result, FusionOutput) else FusionOutput(result)


class DiagnosticSequenceFusion(torch.nn.Module):
    """DTA or DMF fusion with explicit padding masks."""

    def __init__(self, method: str, dimension: int, names: Sequence[str], config: Mapping) -> None:
        super().__init__()
        self.method, self.names = method, tuple(names)
        self.non_id = tuple(n for n in names if n != "id")
        if method not in SEQUENCE_FUSIONS or "id" not in names or not self.non_id:
            raise ContractError("sequence diagnostic fusion needs ID and content")
        attention_dim = int(config.get("attention_dim", 128))
        dropout = float(config.get("dropout", 0.0))
        self.attention = _DecoupledTargetAttention(
            dimension,
            attention_dim,
            int(config.get("num_buckets", 35)),
            float(config.get("dta_dropout", 0.0)) if method == "dta" else dropout,
        )
        self.output_dim = attention_dim
        inner = config.get("dmf_inner_fusion", "cat") if method == "dmf" else "cat"
        self.content_fusion = _build_fusion(inner, self.non_id, dimension)
        if method == "dmf":
            hidden = tuple(config.get("dmf_mlp_dims", [1024, 512, 256]))
            tiers = int(config.get("tier_num", 10))
            self.tiers = SimilarityTiers(tiers)
            self.me = MultiLayerPerceptron(attention_dim, hidden, dropout, batch_norm=True)
            self.mc = MultiLayerPerceptron(tiers, hidden, dropout, batch_norm=True)
            self.alpha = float(config.get("alpha", 0.5))
            if not 0 <= self.alpha <= 1:
                raise ContractError("DMF alpha must be in [0,1]")
            self.output_dim = hidden[-1]

    def forward(
        self,
        target: Mapping[str, torch.Tensor],
        history: Mapping[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> FusionOutput:
        if set(target) != set(self.names) or set(history) != set(self.names):
            raise ContractError("sequence fusion modalities differ from configuration")
        if mask.dtype != torch.bool or mask.shape != history["id"].shape[:2]:
            raise ContractError("sequence fusion requires an aligned boolean history mask")
        target_content = self.content_fusion({n: target[n] for n in self.non_id}).fused
        # Separate calls retain legacy step-wise fusion/normalization behavior.
        history_content = torch.stack(
            [
                self.content_fusion({n: history[n][:, i] for n in self.non_id}).fused
                for i in range(mask.shape[1])
            ],
            1,
        )
        scores = F.cosine_similarity(target_content.unsqueeze(1), history_content, dim=-1)
        interest = self.attention(target["id"], history["id"], scores, mask)
        if self.method == "dmf":
            interest = self.alpha * self.me(interest) + (1 - self.alpha) * self.mc(
                self.tiers(scores, mask)
            )
        return FusionOutput(interest)


def build_diagnostic_fusion(method, dimension, names, config):
    cls = DiagnosticSequenceFusion if method in SEQUENCE_FUSIONS else DiagnosticLocalFusion
    return cls(method, dimension, names, config)
