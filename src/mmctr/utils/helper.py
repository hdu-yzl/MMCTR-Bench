import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_device(device_id: int) -> torch.device:
    if device_id != -1:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda:{}".format(device_id))
    return torch.device("cpu")


def setup_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


LOG_FMT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
DATE_FMT = "%m-%d %H:%M:%S"


def get_logger(
    name: str = "root",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = LOG_FMT,
    filename: Optional[str] = None,
) -> logging.Logger:
    """Return a per-name logger without attaching duplicate handlers.

    ``filename`` optionally fixes the log filename instead of deriving it from ``name``.
    """
    logger = logging.getLogger(name)

    # Reuse the configured singleton instead of duplicating console and file output.
    if logger.handlers:
        return logger

    logger.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt, datefmt=DATE_FMT))
    logger.addHandler(console)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            Path(log_dir) / (filename or f"{name}.log"), encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=DATE_FMT))
        logger.addHandler(file_handler)

    # Prevent parent handlers from emitting each record a second time.
    logger.propagate = False
    return logger
