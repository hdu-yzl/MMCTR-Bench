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
    """
    单例 logger：多次调用同名 name 不会重复加 handler；filename 可固定日志文件名。
    """
    logger = logging.getLogger(name)

    # 已实例化过直接返回
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 控制台
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

    # 禁止向上层(父 logger)传递，避免重复打印
    logger.propagate = False
    return logger
