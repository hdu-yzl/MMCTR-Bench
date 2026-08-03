"""Canonical implementations of the simple multimodal model family."""

from typing import Dict, Mapping, Sequence, Tuple

import torch

from mmctr.core import Batch, ContractError, ModelOutput
from mmctr.models.base import BaseSeqModel, HistoryCapability
from mmctr.models.baselines.layers import FeatureEmbedding, MultiLayerPerceptron
from mmctr.models.components import NamedFeatureProjector, feature_presence


class _ConcatFusion(torch.nn.Module):
    def __init__(self, features: Sequence[str], dimension: int) -> None:
        super().__init__()
        self.features = tuple(features)
        self.output_dim = dimension * len(self.features)

    def forward(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([values[name] for name in self.features], dim=-1)


class _ReduceFusion(torch.nn.Module):
    def __init__(self, features: Sequence[str], dimension: int, reduction: str) -> None:
        super().__init__()
        self.features = tuple(features)
        self.output_dim = dimension
        self.reduction = reduction

    def forward(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([values[name] for name in self.features], dim=0)
        return stacked.mean(dim=0) if self.reduction == "mean" else stacked.sum(dim=0)


class _MAFFusion(torch.nn.Module):
    def __init__(self, features: Sequence[str], dimension: int) -> None:
        super().__init__()
        self.features = tuple(features)
        self.output_dim = dimension
        self.weights = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.empty(dimension, dimension)) for name in features}
        )
        self.biases = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.zeros(dimension)) for name in features}
        )
        for weight in self.weights.values():
            torch.nn.init.xavier_uniform_(weight)

    def forward(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [
                torch.tanh(values[name] @ self.weights[name] + self.biases[name])
                for name in self.features
            ],
            dim=0,
        ).sum(dim=0)


class _LowRankFusion(torch.nn.Module):
    def __init__(
        self, features: Sequence[str], dimension: int, rank: int, output_dim: int
    ) -> None:
        super().__init__()
        self.features = tuple(features)
        self.output_dim = output_dim
        self.factors = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(rank, dimension + 1, output_dim)) for _ in features]
        )
        self.fusion_weights = torch.nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = torch.nn.Parameter(torch.zeros(1, output_dim))
        for factor in self.factors:
            torch.nn.init.xavier_normal_(factor)
        torch.nn.init.xavier_normal_(self.fusion_weights)

    def forward(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        terms = []
        for name, factor in zip(self.features, self.factors):
            value = values[name]
            ones = torch.ones(*value.shape[:-1], 1, dtype=value.dtype, device=value.device)
            terms.append(torch.einsum("rdo,...d->...ro", factor, torch.cat([ones, value], dim=-1)))
        product = torch.stack(terms, dim=0).prod(dim=0)
        return (product * self.fusion_weights.unsqueeze(-1)).sum(dim=-2) + self.fusion_bias


class _MTFNFusion(torch.nn.Module):
    def __init__(self, features: Sequence[str], dimension: int, rank: int) -> None:
        super().__init__()
        self.features = tuple(features)
        self.output_dim = dimension
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.ModuleList(
                    [torch.nn.Linear(dimension, dimension, bias=False) for _ in range(rank)]
                )
                for name in features
            }
        )
        self.compress = torch.nn.Linear(dimension, dimension)
        self.rank = rank

    def forward(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rank_outputs = []
        for rank_index in range(self.rank):
            modal_values = [self.heads[name][rank_index](values[name]) for name in self.features]
            fused = modal_values[0]
            for value in modal_values[1:]:
                fused = fused * value
            rank_outputs.append(fused)
        return self.compress(torch.stack(rank_outputs, dim=0).sum(dim=0))


def _build_fusion(
    method: str,
    features: Sequence[str],
    dimension: int,
    rank: int = 5,
    output_dim: int = 16,
) -> torch.nn.Module:
    method = str(method).lower()
    if not features:
        raise ContractError("multimodal fusion requires at least one feature")
    if method == "cat":
        return _ConcatFusion(features, dimension)
    if method in {"add", "mean"}:
        return _ReduceFusion(features, dimension, method)
    if method == "maf":
        return _MAFFusion(features, dimension)
    if method == "lmf":
        return _LowRankFusion(features, dimension, rank, output_dim)
    if method == "mtfn":
        return _MTFNFusion(features, dimension, rank)
    raise ContractError(
        "simple canonical models support cat/add/mean/maf/lmf/mtfn fusion; got {!r}".format(
            method
        )
    )


class _PooledMultimodalModel(BaseSeqModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(HistoryCapability.POOLED_HISTORY)
        self.latent_dim = int(model_config.get("latent_dim", 128))
        self.projection_dim = int(model_config.get("projection_dim", 128))
        self.mlp_dims = tuple(
            int(value) for value in model_config.get("mlp_dims", [1024, 512, 256])
        )
        self.dropout = float(model_config.get("dropout", 0.5))
        self.batch_norm = bool(model_config.get("batch_norm", False))
        self.feature_names = tuple(data_config.get("use_mm_features", ("id",)))
        self.history_feature_names = tuple(
            data_config.get("use_mm_seq_features", self.feature_names)
        )
        if "id" not in self.feature_names or "id" not in self.history_feature_names:
            raise ContractError("simple multimodal models require target and history ID features")
        target_dimensions = dict(data_config.get("mm_dims", {}))
        history_dimensions = dict(data_config.get("mm_seq_dims", target_dimensions))
        target_dimensions["id"] = self.latent_dim * 2
        history_dimensions["id"] = self.latent_dim
        self.target_projectors = self._make_projectors(self.feature_names, target_dimensions)
        self.history_projectors = self._make_projectors(
            self.history_feature_names, history_dimensions
        )
        self.embedding = FeatureEmbedding(int(data_config["id_feature_num"]) + 1, self.latent_dim)

    def _make_projectors(
        self, names: Sequence[str], dimensions: Mapping[str, int]
    ) -> NamedFeatureProjector:
        missing = [name for name in names if name not in dimensions]
        if missing:
            raise ContractError("missing feature dimensions: {}".format(missing))
        return NamedFeatureProjector(
            {name: int(dimensions[name]) for name in names},
            self.projection_dim,
        )

    @staticmethod
    def _target_feature(batch: Batch, name: str) -> torch.Tensor:
        if name in batch.item_features:
            return batch.item_features[name]
        if name in batch.context_features:
            return batch.context_features[name]
        raise ContractError("target/context feature {!r} is missing".format(name))

    def project_target(self, batch: Batch) -> Dict[str, torch.Tensor]:
        try:
            target_ids = torch.cat(
                [batch.user_features["id"], batch.item_features["id"]], dim=1
            )
        except KeyError as error:
            raise ContractError("pooled multimodal models require user/item IDs") from error
        encoded = {
            "id": self.embedding(target_ids).flatten(start_dim=1)
        }
        presence = {}
        for name in self.feature_names:
            if name == "id":
                continue
            values = self._target_feature(batch, name)
            encoded[name] = values
            presence[name] = feature_presence(values)
        return self.target_projectors(encoded, presence)

    def project_history(self, batch: Batch) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        presence: Dict[str, torch.Tensor] = {}
        for name in self.history_feature_names:
            try:
                values = batch.history_features[name]
            except KeyError as error:
                raise ContractError("history feature {!r} is missing".format(name)) from error
            if name == "id":
                values = self.embedding(values)
                presence[name] = batch.history_mask
            else:
                presence[name] = feature_presence(values) & batch.history_mask
            encoded[name] = values
        return self.history_projectors(encoded, presence)

    def make_predictor(self, input_dim: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        hidden = MultiLayerPerceptron(
            input_dim,
            self.mlp_dims,
            self.dropout,
            batch_norm=self.batch_norm,
            activation="relu",
        )
        output = MultiLayerPerceptron(
            self.mlp_dims[-1],
            [1],
            self.dropout,
            batch_norm=self.batch_norm,
            activation=None,
        )
        return hidden, output


class DNN_mm(_PooledMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        method = model_config.get("modal_fusion_method", "cat")
        rank = int(model_config.get("rank", 5))
        output_dim = int(model_config.get("fusion_dim", 16))
        self.target_fusion = _build_fusion(
            method, self.feature_names, self.projection_dim, rank, output_dim
        )
        self.history_fusion = _build_fusion(
            method, self.history_feature_names, self.projection_dim, rank, output_dim
        )
        input_dim = self.target_fusion.output_dim + self.history_fusion.output_dim
        self.dnn, self.out_put = self.make_predictor(input_dim)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.target_fusion(self.project_target(batch))
        pooled = {
            name: self.masked_pool(values, batch.history_mask)
            for name, values in self.project_history(batch).items()
        }
        history = self.history_fusion(pooled)
        return ModelOutput(self.out_put(self.dnn(torch.cat([target, history], dim=-1))))


class LMF(_PooledMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        rank = int(model_config.get("rank", 5))
        output_dim = int(model_config.get("fusion_dim", 16))
        self.target_fusion = _LowRankFusion(
            self.feature_names, self.projection_dim, rank, output_dim
        )
        self.history_fusion = _LowRankFusion(
            self.history_feature_names, self.projection_dim, rank, output_dim
        )
        self.dnn, self.out_put = self.make_predictor(output_dim * 2)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.target_fusion(self.project_target(batch))
        history_tokens = self.history_fusion(self.project_history(batch))
        history = self.masked_pool(history_tokens, batch.history_mask)
        return ModelOutput(self.out_put(self.dnn(torch.cat([target, history], dim=-1))))


class MTFN(_PooledMultimodalModel):
    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(model_config, data_config)
        rank = int(model_config.get("rank", 20))
        self.target_fusion = _MTFNFusion(self.feature_names, self.projection_dim, rank)
        self.history_fusion = _MTFNFusion(
            self.history_feature_names, self.projection_dim, rank
        )
        self.dnn, self.out_put = self.make_predictor(self.projection_dim * 2)

    def forward_batch(self, batch: Batch) -> ModelOutput:
        target = self.target_fusion(self.project_target(batch))
        history_tokens = self.history_fusion(self.project_history(batch))
        history = self.masked_pool(history_tokens, batch.history_mask)
        return ModelOutput(self.out_put(self.dnn(torch.cat([target, history], dim=-1))))


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
                torch.nn.ModuleList(
                    [torch.nn.Dropout(value) for value in dropouts]
                )
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            experts = layer(values, residual=index > 0)
            transformed = [
                dropout(torch.relu(norm(expert)))
                for expert, norm, dropout in zip(
                    experts, self.norms[index], self.dropouts[index]
                )
            ]
            values = torch.cat(transformed, dim=-1)
        return values


class SimCEN(_PooledMultimodalModel):
    """Canonical SimCEN with a named, scalar contrastive auxiliary loss."""

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
            positive / self.temperature
            - torch.logsumexp(all_pairs / self.temperature, dim=-1)
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


__all__ = ["DNN_mm", "LMF", "MTFN", "SimCEN"]
