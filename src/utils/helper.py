import torch
import random
import numpy as np
import os
import time
from contextlib import contextmanager
from typing import List, Optional
from typing import Dict, Any
import yaml
import logging
import sys
from pathlib import Path


def resolve_data_loader_class(dataset):
    from mmctr.data.registry import resolve_data_loader_class as resolve

    return resolve(dataset)


def resolve_model_class(model_name):
    from mmctr.models.registry import resolve_legacy_model_class as resolve

    return resolve(model_name)


def getOptim(network, optim, lr, l2):
    params = network.parameters()
    optim = optim.lower()
    lr, l2 = float(lr), float(l2)
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=l2)
    elif optim == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=l2)
    elif optim == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=l2)
    else:
        raise ValueError("Invalid optmizer type:{}".format(optim))


def getDevice(device_id):
    if device_id != -1:
        assert torch.cuda.is_available(), "CUDA is not available"
        return torch.device(f'cuda:{device_id}')
    else:
        return torch.device('cpu')


def setup_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_env(cuda_id: int = 0,
              num_threads: int = 8,
              enable_xla: bool = True) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(num_threads)
    if enable_xla:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"


def getDataLoader(dataset, data_config, batch_size):
    from mmctr.data.registry import create_data_loader

    return create_data_loader(dataset, data_config, batch_size)


def getModel(model_name, model_config, train_config, data_config, logger):
    from mmctr.models.registry import model_spec, resolve_legacy_model_class

    takes_logger = model_spec(model_name).metadata.get("takes_logger", True)
    arguments = (model_config, train_config, data_config)
    if takes_logger:
        arguments += (logger,)
    return resolve_legacy_model_class(model_name)(*arguments)


@contextmanager
def timer(record: Optional[List[float]] = None):
    """
    用法：
        times = []                       # 用来收集每一次耗时
        with timer(times):
            pred = model(feats)
    退出上下文后 times 里多了这一次耗时（单位秒）。
    """
    start = time.time()
    yield
    cost = time.time() - start
    if record is not None:
        record.append(cost)


def load_yaml(path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"invalid YAML syntax in {path}: {e}") from e
    if cfg is None:
        cfg = {}
    return cfg


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
