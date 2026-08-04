"""Shared projection and fusion support for pooled multimodal models."""

from typing import Dict, Mapping, Sequence, Tuple

import torch

from mmctr.core import Batch, ContractError
from mmctr.models.common.base import BaseSeqModel, HistoryCapability
from mmctr.models.common.layers import FeatureEmbedding, MultiLayerPerceptron
from mmctr.models.common.components import (
    ConcatenateFusion,
    LowRankFusion,
    MAFFusion,
    MTFNFusion,
    MeanFusion,
    ModalityFusion,
    NamedFeatureProjector,
    SumFusion,
    feature_presence,
)

_ConcatFusion = ConcatenateFusion


def _ReduceFusion(features: Sequence[str], dimension: int, reduction: str) -> ModalityFusion:
    fusion_class = MeanFusion if reduction == "mean" else SumFusion
    return fusion_class(features, dimension)


_MAFFusion = MAFFusion


_LowRankFusion = LowRankFusion


_MTFNFusion = MTFNFusion


def _build_fusion(
    method: str,
    features: Sequence[str],
    dimension: int,
    rank: int = 5,
    output_dim: int = 16,
) -> ModalityFusion:
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
        "simple canonical models support cat/add/mean/maf/lmf/mtfn fusion; got {!r}".format(method)
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
            target_ids = torch.cat([batch.user_features["id"], batch.item_features["id"]], dim=1)
        except KeyError as error:
            raise ContractError("pooled multimodal models require user/item IDs") from error
        encoded = {"id": self.embedding(target_ids).flatten(start_dim=1)}
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


__all__ = ["_LowRankFusion", "_MAFFusion", "_MTFNFusion", "_PooledMultimodalModel", "_build_fusion"]
