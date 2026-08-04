"""Pairwise adaptive modality disentanglement model."""

import math
from itertools import combinations
from typing import Mapping, Sequence, Tuple

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.multimodal import _PooledMultimodalModel


class _PAMDDisentangleBlock(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_dim: int,
        dropout: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.common_a = self._mlp(dimension, hidden_dim, dropout, layer_norm)
        self.common_b = self._mlp(dimension, hidden_dim, dropout, layer_norm)
        self.a_to_b = self._mlp(dimension, hidden_dim, dropout, layer_norm)
        self.b_to_a = self._mlp(dimension, hidden_dim, dropout, layer_norm)
        self.query = torch.nn.Linear(dimension, dimension, bias=False)
        self.key = torch.nn.Linear(dimension, dimension, bias=False)

    @staticmethod
    def _mlp(
        dimension: int, hidden_dim: int, dropout: float, layer_norm: bool
    ) -> torch.nn.Sequential:
        layers = [
            torch.nn.Linear(dimension, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dimension),
        ]
        if layer_norm:
            layers.append(torch.nn.LayerNorm(dimension))
        return torch.nn.Sequential(*layers)

    def forward(
        self, first: torch.Tensor, second: torch.Tensor, query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        common_first = self.common_a(first)
        common_second = self.common_b(second)
        specific_first = first - common_first
        specific_second = second - common_second
        representations = torch.stack(
            [common_first, common_second, specific_first, specific_second], dim=-2
        )
        scores = torch.sum(self.query(query).unsqueeze(-2) * self.key(representations), dim=-1)
        weights = torch.softmax(scores / math.sqrt(query.shape[-1]), dim=-1)
        fused = query + torch.sum(weights.unsqueeze(-1) * representations, dim=-2)

        alignment = torch.nn.functional.mse_loss(common_first, common_second)
        orthogonal = torch.mean(
            torch.sum(
                torch.nn.functional.normalize(specific_first, dim=-1)
                * torch.nn.functional.normalize(specific_second, dim=-1),
                dim=-1,
            ).pow(2)
        )
        common_loss = torch.nn.functional.mse_loss(
            second, self.a_to_b(common_first)
        ) + torch.nn.functional.mse_loss(first, self.b_to_a(common_second))
        complete_loss = torch.nn.functional.mse_loss(
            second, self.a_to_b(first)
        ) + torch.nn.functional.mse_loss(first, self.b_to_a(second))
        specific_loss = torch.nn.functional.mse_loss(
            second, self.a_to_b(specific_first)
        ) + torch.nn.functional.mse_loss(first, self.b_to_a(specific_second))
        ranking = -torch.nn.functional.logsigmoid(complete_loss - common_loss)
        ranking = ranking - torch.nn.functional.logsigmoid(specific_loss - complete_loss)
        return fused, alignment + orthogonal + ranking


class PAMD(_PooledMultimodalModel):
    """Pairwise adaptive modality disentanglement over target and history.

    Each modality pair is decomposed into common and residual components; the
    auxiliary loss aligns common components, discourages residual correlation,
    and ranks cross-reconstruction quality.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        preferred = ("image", "text", "audio")
        self.modal_names = tuple(name for name in preferred if name in self.feature_names)
        self.history_modal_names = tuple(
            name for name in preferred if name in self.history_feature_names
        )
        if len(self.modal_names) < 2 or len(self.history_modal_names) < 2:
            raise ContractError("PAMD requires at least two target and history modalities")
        hidden_dim = int(model_config.get("pamd_hidden_dim", self.projection_dim))
        self.auxiliary_weight = float(model_config.get("pamd_aux_weight", 0.1))
        if hidden_dim <= 0 or self.auxiliary_weight < 0.0:
            raise ContractError("PAMD hidden dimension must be positive and weight non-negative")
        layer_norm = bool(model_config.get("pamd_layer_norm", True))
        self.target_blocks = self._build_blocks(self.modal_names, hidden_dim, layer_norm)
        self.history_blocks = self._build_blocks(self.history_modal_names, hidden_dim, layer_norm)
        self.dnn, self.output = self.make_predictor(self.projection_dim * 2)

    @staticmethod
    def _pair_key(first: str, second: str) -> str:
        return "{}__{}".format(first, second)

    def _build_blocks(
        self, names: Sequence[str], hidden_dim: int, layer_norm: bool
    ) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(
            {
                self._pair_key(first, second): _PAMDDisentangleBlock(
                    self.projection_dim,
                    hidden_dim,
                    self.dropout,
                    layer_norm,
                )
                for first, second in combinations(names, 2)
            }
        )

    def _run_blocks(
        self,
        blocks: torch.nn.ModuleDict,
        names: Sequence[str],
        values: Mapping[str, torch.Tensor],
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused_values = []
        losses = []
        for first, second in combinations(names, 2):
            fused, loss = blocks[self._pair_key(first, second)](
                values[first], values[second], query
            )
            fused_values.append(fused)
            losses.append(loss)
        return (
            torch.stack(fused_values, dim=1).mean(dim=1),
            torch.stack(losses).mean(),
        )

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.project_target(batch)
        history = {
            name: self.masked_pool(values, batch.history_mask)
            for name, values in self.project_history(batch).items()
        }
        target_fusion, target_loss = self._run_blocks(
            self.target_blocks, self.modal_names, target, target["id"]
        )
        history_fusion, history_loss = self._run_blocks(
            self.history_blocks,
            self.history_modal_names,
            history,
            history["id"],
        )
        logits = self.output(self.dnn(torch.cat([target_fusion, history_fusion], dim=-1)))
        return ModelOutput(
            logits,
            auxiliary_losses={
                "pamd_disentanglement": self.auxiliary_weight * (target_loss + history_loss)
            },
            representations={
                "history_fusion": history_fusion,
                "target_fusion": target_fusion,
            },
        )


__all__ = ["PAMD"]
