"""SimCEN multimodal recommendation model."""

from typing import Mapping, Sequence, Tuple

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.common.multimodal import _PooledMultimodalModel
from mmctr.models.common.layers import MultiLayerPerceptron


class _Segmentation(torch.nn.Module):
    def __init__(self, field_count: int, embedding_dim: int, flatten_dim: int) -> None:
        super().__init__()
        self.register_buffer(
            "upper_mask", torch.triu(torch.ones(field_count, field_count), 0).bool()
        )
        self.register_buffer(
            "lower_mask", torch.tril(torch.ones(field_count, field_count), 0).bool()
        )
        self.kernel_count = field_count * (field_count + 1) // 2
        self.kernel_weight = torch.nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        torch.nn.init.xavier_normal_(self.kernel_weight)
        self.project_upper = torch.nn.Linear(self.kernel_count, flatten_dim, bias=False)
        self.project_lower = torch.nn.Linear(self.kernel_count, flatten_dim, bias=False)

    def forward(
        self, feature_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kernels = torch.matmul(
            torch.matmul(feature_embeddings, self.kernel_weight),
            feature_embeddings.transpose(1, 2),
        )
        upper = torch.masked_select(kernels, self.upper_mask).view(-1, self.kernel_count)
        lower = torch.masked_select(kernels, self.lower_mask).view(-1, self.kernel_count)
        return (
            feature_embeddings.flatten(start_dim=1),
            self.project_upper(upper),
            self.project_lower(lower),
        )


class _ExpertLinear(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if output_dim % 3:
            raise ContractError("SimCEN hidden dimensions must be divisible by three")
        self.linear = torch.nn.Linear(input_dim, output_dim)
        third = output_dim // 3
        self.gate = torch.nn.Sequential(torch.nn.Linear(third, third), torch.nn.Sigmoid())
        self.gate_temperature = torch.nn.Parameter(torch.ones(third))
        self.noise = torch.nn.Parameter(torch.empty(third * 2))
        input_third = input_dim // 3
        self.residual_projection = (
            torch.nn.Identity()
            if input_third == third
            else torch.nn.Linear(input_third, third, bias=False)
        )
        torch.nn.init.uniform_(self.noise)

    def forward(
        self, values: torch.Tensor, residual: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_v1 = previous_v2 = None
        if residual:
            _, previous_v1, previous_v2 = torch.chunk(values, chunks=3, dim=-1)
        ego, view1, view2 = torch.chunk(self.linear(values), chunks=3, dim=-1)
        gate = self.gate(ego) / self.gate_temperature.clamp(min=1e-3)
        view1 = gate * view1 + view1
        view2 = gate * view2 + view2
        if previous_v1 is not None and previous_v2 is not None:
            noise1, noise2 = torch.chunk(self.noise, chunks=2, dim=-1)
            view1 = view1 + noise1 + self.residual_projection(previous_v1)
            view2 = view2 + noise2 + self.residual_projection(previous_v2)
        return ego, view1, view2


class _MultiLevelExpert(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropouts: Tuple[float, float, float],
        batch_norms: Tuple[bool, bool, bool],
    ) -> None:
        super().__init__()
        dimensions = (input_dim,) + tuple(hidden_dims)
        self.layers = torch.nn.ModuleList(
            [
                _ExpertLinear(dimensions[index], dimensions[index + 1])
                for index in range(len(hidden_dims))
            ]
        )
        self.norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for output_dim in hidden_dims:
            third = output_dim // 3
            self.norms.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.BatchNorm1d(third) if enabled else torch.nn.Identity()
                        for enabled in batch_norms
                    ]
                )
            )
            self.dropouts.append(
                torch.nn.ModuleList([torch.nn.Dropout(value) for value in dropouts])
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            experts = layer(values, residual=index > 0)
            transformed = [
                dropout(torch.relu(norm(expert)))
                for expert, norm, dropout in zip(experts, self.norms[index], self.dropouts[index])
            ]
            values = torch.cat(transformed, dim=-1)
        return values


class SimCEN(_PooledMultimodalModel):
    """Segment field interactions into ego and two contrastive expert views.

    The bilinear field matrix is split into upper/lower triangular views. The
    returned ``simcen_contrastive`` value is already scaled by ``alpha`` and is
    a scalar suitable for direct addition to the primary objective.
    """

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        hidden_dims = tuple(
            int(value) for value in model_config.get("hidden_unit", [480, 480, 480])
        )
        if not hidden_dims:
            raise ContractError("SimCEN hidden_unit cannot be empty")
        if any(value % 3 for value in hidden_dims):
            raise ContractError("SimCEN hidden dimensions must be divisible by three")
        self.temperature = float(model_config.get("cl_temperature", 0.1))
        if self.temperature <= 0:
            raise ContractError("SimCEN contrastive temperature must be positive")
        self.auxiliary_weight = float(model_config.get("alpha", 0.5))
        field_count = len(self.feature_names) + len(self.history_feature_names)
        flatten_dim = self.projection_dim * field_count
        self.segmentation = _Segmentation(field_count, self.projection_dim, flatten_dim)
        self.experts = _MultiLevelExpert(
            flatten_dim * 3,
            hidden_dims,
            (
                float(model_config.get("ego_dropout", 0.0)),
                float(model_config.get("v1_dropout", 0.0)),
                float(model_config.get("v2_dropout", 0.0)),
            ),
            (
                bool(model_config.get("ego_batch_norm", True)),
                bool(model_config.get("v1_batch_norm", True)),
                bool(model_config.get("v2_batch_norm", True)),
            ),
        )
        self.out_put = MultiLayerPerceptron(
            hidden_dims[-1],
            [1],
            self.dropout,
            batch_norm=self.batch_norm,
            activation=None,
        )

    def _contrastive_loss(
        self, ego: torch.Tensor, view1: torch.Tensor, view2: torch.Tensor
    ) -> torch.Tensor:
        ego = torch.nn.functional.normalize(ego, dim=-1)
        view1 = torch.nn.functional.normalize(view1, dim=-1)
        view2 = torch.nn.functional.normalize(view2, dim=-1)
        positive = ((ego * view1).sum(dim=-1) + (ego * view2).sum(dim=-1)) * 0.5
        all_pairs = torch.matmul(view1, view2.transpose(0, 1))
        return -(
            positive / self.temperature - torch.logsumexp(all_pairs / self.temperature, dim=-1)
        ).mean()

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.project_target(batch)
        history = {
            name: self.masked_pool(values, batch.history_mask)
            for name, values in self.project_history(batch).items()
        }
        fields = torch.stack(
            [target[name] for name in self.feature_names]
            + [history[name] for name in self.history_feature_names],
            dim=1,
        )
        ego, view1, view2 = self.segmentation(fields)
        experts = self.experts(torch.cat([ego, view1, view2], dim=-1))
        ego, view1, view2 = torch.chunk(experts, chunks=3, dim=-1)
        view1 = ego + view1
        view2 = ego + view2
        logits = self.out_put(torch.cat([ego, view1, view2], dim=-1))
        auxiliary = self.auxiliary_weight * self._contrastive_loss(ego, view1, view2)
        return ModelOutput(
            logits,
            auxiliary_losses={"simcen_contrastive": auxiliary},
            representations={"ego": ego, "view1": view1, "view2": view2},
        )


__all__ = ["SimCEN"]
