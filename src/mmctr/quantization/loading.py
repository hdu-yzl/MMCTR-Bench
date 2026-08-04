"""Composition-layer loading and compatibility checks for quantized CTR models."""

from pathlib import Path
from typing import Dict, Mapping

from mmctr.core import ContractError

from .artifacts import psrq_artifact_path, rq_artifact_path
from .psrq import PSRQPretrainer
from .residual import ResidualQuantizer


def _effective_config(model_config: Mapping, data_config: Mapping) -> Dict:
    dataset = str(data_config.get("name", "")).lower()
    selected = model_config.get(dataset, model_config)
    if not isinstance(selected, Mapping):
        raise ContractError("dataset-specific model configuration must be a mapping")
    return dict(selected)


def _non_id_modalities(data_config: Mapping):
    return tuple(name for name in data_config.get("use_mm_features", ()) if name != "id")


def _load_qarm_dependencies(model_config: Mapping, data_config: Mapping, artifact_root: Path):
    config = _effective_config(model_config, data_config)
    dataset = str(data_config.get("name", "")).lower()
    modalities = _non_id_modalities(data_config)
    dimensions = dict(data_config.get("mm_seq_dims", data_config.get("mm_dims", {})))
    quantizers = {}
    for modality in modalities:
        quantizer = ResidualQuantizer.from_artifact(
            rq_artifact_path(artifact_root, dataset, modality)
        )
        expected = (
            int(config.get("n_levels", 3)),
            int(config.get("codebook_size", 1024)),
            int(dimensions[modality]),
        )
        actual = (
            quantizer.n_levels,
            quantizer.codebook_size,
            quantizer.dimension,
        )
        if actual != expected:
            raise ContractError(
                "QARM {!r} RQ structure {} does not match {}".format(modality, actual, expected)
            )
        metadata = quantizer.artifact_metadata
        if metadata.get("dataset", dataset) != dataset:
            raise ContractError("QARM RQ artifact dataset does not match")
        if metadata.get("modality", modality) != modality:
            raise ContractError("QARM RQ artifact modality does not match")
        quantizers[modality] = quantizer
    return {"quantizers": quantizers}


def _load_psrq_consumer_dependencies(
    model_config: Mapping, data_config: Mapping, artifact_root: Path
):
    config = _effective_config(model_config, data_config)
    dataset = str(data_config.get("name", "")).lower()
    quantizer = PSRQPretrainer.from_artifact(psrq_artifact_path(artifact_root, dataset))
    expected_modalities = _non_id_modalities(data_config)
    if quantizer.dataset_name != dataset:
        raise ContractError("PSRQ benchmark consumer artifact dataset does not match")
    if quantizer.modalities != expected_modalities:
        raise ContractError("PSRQ benchmark consumer artifact modalities do not match")
    dimensions = dict(data_config.get("mm_seq_dims", data_config.get("mm_dims", {})))
    expected_dimensions = {name: int(dimensions[name]) for name in expected_modalities}
    if quantizer.modality_dimensions != expected_dimensions:
        raise ContractError("PSRQ benchmark consumer artifact dimensions do not match")
    expected_structure = (
        int(config.get("n_levels", 3)),
        int(config.get("codebook_size", 256)),
        int(config.get("projection_dim", 128)),
        tuple(int(value) for value in config.get("psrq_dims", (256, 128))),
        float(config.get("dropout", 0.0)),
        bool(config.get("batch_norm", True)),
    )
    actual_structure = (
        quantizer.n_levels,
        quantizer.codebook_size,
        quantizer.embedding_dimension,
        quantizer.hidden_dimensions,
        quantizer.dropout,
        quantizer.batch_norm,
    )
    if actual_structure != expected_structure:
        raise ContractError(
            "PSRQ benchmark consumer structure {} does not match {}".format(
                actual_structure, expected_structure
            )
        )
    quantizer.eval()
    quantizer.requires_grad_(False)
    return {"quantizer": quantizer}


def load_model_quantization_dependencies(
    model_name: str,
    model_config: Mapping,
    data_config: Mapping,
    artifact_root,
):
    """Load validated constructor kwargs for a quantized CTR model."""

    root = Path(artifact_root).expanduser().resolve()
    name = model_name.lower()
    if name == "qarm":
        return _load_qarm_dependencies(model_config, data_config, root)
    if name == "psrq":
        return _load_psrq_consumer_dependencies(model_config, data_config, root)
    raise ContractError("model {!r} does not consume quantization artifacts".format(name))


__all__ = ["load_model_quantization_dependencies"]
