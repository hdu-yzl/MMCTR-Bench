"""Standard result ingestion and source provenance for generated figures."""

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mmctr.core import ContractError
from mmctr.experiments.runner import RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class StandardResult:
    path: Path
    run_id: str
    task_id: str
    dataset: str
    model: str
    seed: int
    data_fingerprint: str
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_standard_results(
    paths: Sequence[Path], required_metrics: Sequence[str] = ()
) -> Tuple[StandardResult, ...]:
    """Read only completed ExperimentRunner result-v1 artifacts."""

    result_paths = tuple(Path(path) for path in paths)
    required = tuple(str(name) for name in required_metrics)
    if not result_paths or len(set(result_paths)) != len(result_paths):
        raise ContractError("standard result paths must be non-empty and unique")
    if any(not name for name in required) or len(set(required)) != len(required):
        raise ContractError("required metric names must be unique and non-empty")
    results = []
    for path in result_paths:
        if not path.is_file():
            raise ContractError("standard result file does not exist: {}".format(path))
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict) or values.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ContractError("unsupported standard result schema")
        if values.get("status") != "completed":
            raise ContractError("figures can only consume completed results")
        raw_metrics = values.get("metrics")
        if not isinstance(raw_metrics, dict):
            raise ContractError("standard result metrics must be a mapping")
        try:
            metrics = {str(name): float(value) for name, value in raw_metrics.items()}
            result = StandardResult(
                path=path,
                run_id=str(values["run_id"]),
                task_id=str(values["task_id"]),
                dataset=str(values["dataset"]),
                model=str(values["model"]),
                seed=int(values["seed"]),
                data_fingerprint=str(values["data_fingerprint"]),
                metrics=metrics,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("standard result fields are invalid") from error
        missing = [name for name in required if name not in metrics]
        if missing:
            raise ContractError("standard result is missing metrics: {}".format(missing))
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ContractError("standard result metrics must be finite")
        if not all(
            (
                result.run_id,
                result.task_id,
                result.dataset,
                result.model,
                result.data_fingerprint,
            )
        ):
            raise ContractError("standard result identity fields must be non-empty")
        results.append(result)
    return tuple(results)


def save_figure_provenance(
    path: Path,
    input_paths: Sequence[Path],
    figure_config: Mapping[str, Any],
    script_version: str,
) -> Mapping[str, Any]:
    """Atomically persist the exact result hashes and config behind one figure."""

    inputs = tuple(Path(value) for value in input_paths)
    if (
        not inputs
        or len(set(inputs)) != len(inputs)
        or any(not value.is_file() for value in inputs)
    ):
        raise ContractError("figure input paths must be existing, non-empty, and unique")
    if not isinstance(figure_config, Mapping):
        raise ContractError("figure_config must be a mapping")
    if not isinstance(script_version, str) or not script_version:
        raise ContractError("script_version must be non-empty")
    payload: Dict[str, Any] = {
        "schema": "mmctr-figure-provenance-v1",
        "script_version": script_version,
        "figure_config": dict(figure_config),
        "inputs": [
            {
                "path": input_path.as_posix(),
                "bytes": input_path.stat().st_size,
                "sha256": _file_sha256(input_path),
            }
            for input_path in inputs
        ],
    }
    payload["fingerprint"] = _digest(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(destination.name), dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(destination))
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return MappingProxyType(payload)


def render_metric_figure(
    result_paths: Sequence[Path],
    output_path: Path,
    metric: str,
    kind: str = "bar",
    group_by: str = "model",
    title: Optional[str] = None,
    provenance_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Render one generic metric figure exclusively from standard run results."""

    metric_name = str(metric)
    figure_kind = str(kind).lower()
    grouping = str(group_by).lower()
    if not metric_name:
        raise ContractError("figure metric must be non-empty")
    if figure_kind not in {"bar", "line"}:
        raise ContractError("figure kind must be bar or line")
    if grouping not in {"model", "dataset", "task_id", "seed"}:
        raise ContractError("figure group_by must be model, dataset, task_id, or seed")
    if title is not None and (not isinstance(title, str) or not title):
        raise ContractError("figure title must be a non-empty string when provided")
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ContractError("figure output must use a .png, .pdf, or .svg suffix")

    results = load_standard_results(result_paths, required_metrics=(metric_name,))
    grouped_values: Dict[str, list] = defaultdict(list)
    for result in results:
        label = str(getattr(result, grouping))
        grouped_values[label].append(result.metrics[metric_name])
    groups = sorted(grouped_values)
    values = [sum(grouped_values[group]) / len(grouped_values[group]) for group in groups]
    counts = [len(grouped_values[group]) for group in groups]

    try:
        import matplotlib  # type: ignore[import-untyped]

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot  # type: ignore[import-untyped]
    except ImportError as error:
        raise ContractError("plotting requires the analysis dependency group") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure_width = max(6.0, 1.2 * len(groups))
    figure, axis = pyplot.subplots(figsize=(figure_width, 4.5))
    temporary_path: Optional[Path] = None
    try:
        positions = list(range(len(groups)))
        if figure_kind == "bar":
            axis.bar(positions, values)
        else:
            axis.plot(positions, values, marker="o")
        axis.set_xticks(positions)
        axis.set_xticklabels(groups, rotation=30, ha="right")
        axis.set_xlabel(grouping)
        axis.set_ylabel(metric_name)
        axis.set_title(title or metric_name)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        with tempfile.NamedTemporaryFile(
            prefix=".{}-".format(destination.stem),
            suffix=destination.suffix,
            dir=str(destination.parent),
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        figure.savefig(str(temporary_path), format=destination.suffix.lower().lstrip("."))
        os.replace(str(temporary_path), str(destination))
    finally:
        pyplot.close(figure)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    figure_config = {
        "metric": metric_name,
        "kind": figure_kind,
        "group_by": grouping,
        "title": title or metric_name,
        "groups": groups,
        "values": values,
        "run_counts": counts,
        "aggregation": "arithmetic_mean",
        "figure_size_inches": [figure_width, 4.5],
        "output": destination.as_posix(),
    }
    provenance_destination = (
        Path(provenance_path).expanduser().resolve()
        if provenance_path is not None
        else Path(str(destination) + ".provenance.json")
    )
    return save_figure_provenance(
        provenance_destination,
        tuple(Path(path) for path in result_paths),
        figure_config,
        script_version="standard-metric-plot-v1",
    )


__all__ = [
    "StandardResult",
    "load_standard_results",
    "render_metric_figure",
    "save_figure_provenance",
]
