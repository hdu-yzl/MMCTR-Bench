"""Legacy diagnostic representations through canonical production projections."""

from typing import Mapping

import torch

from mmctr.core import Batch, ContractError
from mmctr.models.common.multimodal import _PooledMultimodalModel
from mmctr.models.common.sequence import _SequenceMultimodalModel
from mmctr.models.mm_models.psrq import MCCA
from mmctr.models.mm_models.qarm import QARM


def alignment_features(model: torch.nn.Module, batch: Batch) -> Mapping[str, torch.Tensor]:
    """Use target projections, or mean pooled codebook histories for QARM/PSRQ.

    History padding follows the backbone's explicit mask convention. Unlike the
    legacy implementation, padding positions cannot enter the history mean.
    """
    if isinstance(model, QARM):
        history = model._project_quantized_history(batch)
        return {
            name: model.masked_pool(history[name], batch.history_mask)
            for name in model.feature_names
        }
    if isinstance(model, MCCA):
        raw = model._raw_history(batch)
        codes = model.quantizer.encode_history(raw)
        encoded = {"id": model.embedding(batch.history_features["id"])}
        presence = {"id": batch.history_mask}
        for name in model.quantized_modalities:
            present = model._present(raw[name]) & batch.history_mask
            encoded[name] = model._embed_codes(name, codes[name]) * present.unsqueeze(-1)
            presence[name] = present
        history = model.projectors(encoded, presence)
        return {
            name: model.masked_pool(history[name], batch.history_mask)
            for name in model.feature_names
        }
    if isinstance(model, (_PooledMultimodalModel, _SequenceMultimodalModel)):
        return model.project_target(batch)
    raise ContractError("no verified legacy alignment representation for this model")
