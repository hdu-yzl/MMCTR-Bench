"""Small deterministic fixtures that do not depend on repository datasets."""

from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def legacy_dnn_configs(checkpoint_dir: Path) -> Tuple[Dict[str, Any], ...]:
    """Return the minimal legacy configuration required to construct a CPU DNN."""
    model_config = {
        "model_name": "dnn",
        "latent_dim": 4,
        "projection_dim": 4,
        "mlp_dims": [8],
        "dropout": 0.0,
        "batch_norm": False,
    }
    train_config = {
        "seed": 2025,
        "checkpoint_dir": str(checkpoint_dir),
        "early_stop_patience": 1,
        "cuda": -1,
        "max_epochs": 1,
        "log_interval": 1,
        "optim": "adam",
        "lr": 0.001,
        "l2": 0.0,
    }
    data_config = {
        "seq_len": 3,
        "id_fields_num": 2,
        "id_feature_num": 16,
        "use_mm_features": ["id"],
        "mm_dims": {"id": 0},
        "use_mm_seq_features": ["id"],
        "mm_seq_dims": {"id": 0},
    }
    return model_config, train_config, data_config


def make_legacy_dnn_batch() -> Tuple[Dict[str, torch.Tensor], ...]:
    """Create a four-sample ID-only batch matching the current BaseModel contract."""
    features = {
        "id": torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            dtype=torch.long,
        )
    }
    history_features = {
        "id": torch.tensor(
            [[2, 3, 4], [4, 3, 2], [6, 5, 4], [8, 7, 6]],
            dtype=torch.long,
        )
    }
    labels = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float32)
    return features, history_features, labels
