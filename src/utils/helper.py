import torch
import random
import numpy as np
import os
from data.dataloaders import Antm2cLoader, MicrolensLoader, TiktokLoader
from models.ctr_models import DNN, DNN_mm, DNN_mm_seq, DCN, DeepFM, DIN, AutoInt
from models.mm_ctr_models import LMF, Diff_MSIN, MARN, MTFN, DMF, SimCEN, NAML, MAKE, EM3, GMMF, QARM, MCCA
from models.mm_ctr_models import MB, PAMD, MMMLP, M3SRec
from models.pre_models import RQ, PSRQ
import time
from contextlib import contextmanager
from typing import List, Optional
from typing import Dict, Any
import yaml
import logging
import sys
from pathlib import Path


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
    dataset = dataset.lower()
    if dataset == "antm2c":
        return Antm2cLoader(data_config, batch_size)
    elif dataset == "microlens":
        return MicrolensLoader(data_config, batch_size)
    elif dataset == "tiktok":
        return TiktokLoader(data_config, batch_size)
    else:
        raise ValueError("Invalid dataset type:{}".format(dataset))


def getModel(model_name, model_config, train_config, data_config, logger):
    model_name = model_name.lower()
    if model_name == "dnn":
        return DNN(model_config, train_config, data_config, logger)
    elif model_name == "lmf":
        return LMF(model_config, train_config, data_config, logger)
    elif model_name == "diff_msin":
        return Diff_MSIN(model_config, train_config, data_config, logger)
    elif model_name == "marn":
        return MARN(model_config, train_config, data_config, logger)
    elif model_name == "mtfn":
        return MTFN(model_config, train_config, data_config, logger)
    elif model_name == "dmf":
        return DMF(model_config, train_config, data_config, logger)
    elif model_name == "simcen":
        return SimCEN(model_config, train_config, data_config, logger)
    elif model_name == "naml":
        return NAML(model_config, train_config, data_config, logger)
    elif model_name == "make":
        return MAKE(model_config, train_config, data_config, logger)
    elif model_name == "em3":
        return EM3(model_config, train_config, data_config, logger)
    elif model_name == "gmmf":
        return GMMF(model_config, train_config, data_config, logger)
    elif model_name == "rq":
        return RQ(model_config, train_config, data_config)
    elif model_name == "qarm":
        return QARM(model_config, train_config, data_config, logger)
    elif model_name == "psrq":
        return PSRQ(model_config, train_config, data_config)
    elif model_name == "mcca":
        return MCCA(model_config, train_config, data_config, logger)
    elif model_name == "mb":
        return MB(model_config, train_config, data_config, logger)
    elif model_name == "pamd":
        return PAMD(model_config, train_config, data_config, logger)
    elif model_name == "mmmlp":
        return MMMLP(model_config, train_config, data_config, logger)
    elif model_name == "m3srec":
        return M3SRec(model_config, train_config, data_config, logger)
    elif model_name == "dcn":
        return DCN(model_config, train_config, data_config, logger)
    elif model_name == "deepfm":
        return DeepFM(model_config, train_config, data_config, logger)
    elif model_name == "din":
        return DIN(model_config, train_config, data_config, logger)
    elif model_name == "autoint":
        return AutoInt(model_config, train_config, data_config, logger)
    elif model_name == "dnn_mm":
        return DNN_mm(model_config, train_config, data_config, logger)
    elif model_name == "dnn_mm_seq":
        return DNN_mm_seq(model_config, train_config, data_config, logger)
    else:
        raise ValueError("Invalid model type:{}".format(model_name))


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
