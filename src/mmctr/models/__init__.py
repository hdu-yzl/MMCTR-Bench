"""Lazy public model exports backed by the current legacy implementations."""

from models.base_model import BaseModel
from models.base_seq_model import BaseSeqModel
from mmctr.utils.helper import get_model, resolve_model_class


_MODEL_EXPORTS = {
    "DNN": "dnn",
    "DNN_mm": "dnn_mm",
    "DNN_mm_seq": "dnn_mm_seq",
    "DCN": "dcn",
    "DeepFM": "deepfm",
    "DIN": "din",
    "AutoInt": "autoint",
    "LMF": "lmf",
    "Diff_MSIN": "diff_msin",
    "MARN": "marn",
    "MTFN": "mtfn",
    "DMF": "dmf",
    "SimCEN": "simcen",
    "NAML": "naml",
    "MAKE": "make",
    "EM3": "em3",
    "GMMF": "gmmf",
    "QARM": "qarm",
    "MCCA": "mcca",
    "MB": "mb",
    "PAMD": "pamd",
    "MMMLP": "mmmlp",
    "M3SRec": "m3srec",
    "RQ": "rq",
    "PSRQ": "psrq",
}


def __getattr__(name):
    try:
        model_name = _MODEL_EXPORTS[name]
    except KeyError as error:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name)) from error
    model_class = resolve_model_class(model_name)
    globals()[name] = model_class
    return model_class


__all__ = ["BaseModel", "BaseSeqModel", "get_model"] + list(_MODEL_EXPORTS)
