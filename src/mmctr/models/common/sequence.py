"""Shared named-feature encoder for sequence-token multimodal models."""

from typing import Dict, Mapping

import torch

from mmctr.core import Batch, ContractError
from mmctr.models.common.base import BaseSeqModel, HistoryCapability
from mmctr.models.common.layers import FeatureEmbedding
from mmctr.models.common.components import NamedFeatureProjector, feature_presence


class _SequenceMultimodalModel(BaseSeqModel):
    """Shared pure encoder for user, target-item, and history-token branches."""

    def __init__(self, model_config: Mapping, data_config: Mapping) -> None:
        super().__init__(HistoryCapability.SEQUENCE_TOKENS)
        self.latent_dim = int(model_config.get("latent_dim", 128))
        self.projection_dim = int(model_config.get("projection_dim", 128))
        self.dropout = float(model_config.get("dropout", 0.5))
        self.batch_norm = bool(model_config.get("batch_norm", False))
        self.mlp_dims = tuple(
            int(value) for value in model_config.get("mlp_dims", [1024, 512, 256])
        )
        self.feature_names = tuple(data_config.get("use_mm_features", ("id",)))
        self.user_feature_names = tuple(data_config.get("user_features", ("id",)))
        if self.latent_dim <= 0 or self.projection_dim <= 0:
            raise ContractError("sequence latent/projection dimensions must be positive")
        if not self.mlp_dims or any(dimension <= 0 for dimension in self.mlp_dims):
            raise ContractError("sequence model mlp_dims must contain positive dimensions")
        if not 0.0 <= self.dropout < 1.0:
            raise ContractError("sequence model dropout must be in [0, 1)")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ContractError("sequence item feature names must be unique")
        if len(set(self.user_feature_names)) != len(self.user_feature_names):
            raise ContractError("sequence user feature names must be unique")
        if "id" not in self.feature_names or "id" not in self.user_feature_names:
            raise ContractError("sequence multimodal models require item/history/user IDs")

        dimensions = dict(data_config.get("mm_seq_dims", data_config.get("mm_dims", {})))
        user_dimensions = dict(data_config.get("user_features_dim", {}))
        dimensions["id"] = self.latent_dim
        user_dimensions["id"] = self.latent_dim
        missing = [name for name in self.feature_names if name not in dimensions]
        missing.extend(name for name in self.user_feature_names if name not in user_dimensions)
        if missing:
            raise ContractError("missing sequence multimodal dimensions: {}".format(missing))
        selected_dimensions = [
            ("item.{}".format(name), dimensions[name]) for name in self.feature_names
        ]
        selected_dimensions.extend(
            ("user.{}".format(name), user_dimensions[name]) for name in self.user_feature_names
        )
        invalid = {
            name: int(dimension) for name, dimension in selected_dimensions if int(dimension) <= 0
        }
        if invalid:
            raise ContractError("sequence feature dimensions must be positive: {}".format(invalid))

        self.embedding = FeatureEmbedding(int(data_config["id_feature_num"]) + 1, self.latent_dim)
        self.projectors = NamedFeatureProjector(
            {name: int(dimensions[name]) for name in self.feature_names},
            self.projection_dim,
        )
        self.user_projectors = NamedFeatureProjector(
            {name: int(user_dimensions[name]) for name in self.user_feature_names},
            self.projection_dim,
        )

    @staticmethod
    def _target_feature(batch: Batch, name: str) -> torch.Tensor:
        if name in batch.item_features:
            return batch.item_features[name]
        if name in batch.context_features:
            return batch.context_features[name]
        raise ContractError("item/context feature {!r} is missing".format(name))

    def project_target(self, batch: Batch) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        presence: Dict[str, torch.Tensor] = {}
        for name in self.feature_names:
            values = self._target_feature(batch, name)
            if name == "id":
                values = self.embedding(values).squeeze(1)
            else:
                presence[name] = feature_presence(values)
            encoded[name] = values
        return self.projectors(encoded, presence)

    def project_history(self, batch: Batch) -> Dict[str, torch.Tensor]:
        encoded: Dict[str, torch.Tensor] = {}
        presence: Dict[str, torch.Tensor] = {}
        for name in self.feature_names:
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
        return self.projectors(encoded, presence)

    def project_user(self, batch: Batch) -> torch.Tensor:
        encoded: Dict[str, torch.Tensor] = {}
        presence: Dict[str, torch.Tensor] = {}
        for name in self.user_feature_names:
            if name in batch.user_features:
                values = batch.user_features[name]
            elif name in batch.context_features:
                values = batch.context_features[name]
            else:
                raise ContractError("user/context feature {!r} is missing".format(name))
            if name == "id":
                values = self.embedding(values).squeeze(1)
            else:
                presence[name] = feature_presence(values)
            encoded[name] = values
        projected = self.user_projectors(encoded, presence)
        return torch.cat([projected[name] for name in self.user_feature_names], dim=-1)


__all__ = ["_SequenceMultimodalModel"]
