"""Unified parameter, latency, throughput, and peak-memory measurement protocol."""

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch

from mmctr.core import ContractError


@dataclass(frozen=True)
class EfficiencyReport:
    device: str
    accelerator_name: Optional[str]
    torch_version: str
    cuda_version: Optional[str]
    warmup_steps: int
    measured_steps: int
    examples_per_step: int
    total_parameters: int
    trainable_parameters: int
    total_seconds: float
    latency_ms: float
    examples_per_second: float
    peak_memory_bytes: Optional[int]
    input_fingerprint: str
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "accelerator_name": self.accelerator_name,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "examples_per_step": self.examples_per_step,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "total_seconds": self.total_seconds,
            "latency_ms": self.latency_ms,
            "examples_per_second": self.examples_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "input_fingerprint": self.input_fingerprint,
            "fingerprint": self.fingerprint,
        }


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report_payload(report: EfficiencyReport) -> Dict[str, Any]:
    payload = report.to_dict()
    payload.pop("fingerprint")
    return payload


class EfficiencyProtocol:
    """Measure one already-configured step under one explicit device protocol."""

    def __init__(
        self,
        warmup_steps: int = 10,
        measured_steps: int = 100,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if isinstance(warmup_steps, bool) or int(warmup_steps) < 0:
            raise ContractError("warmup_steps must be a non-negative integer")
        if isinstance(measured_steps, bool) or int(measured_steps) <= 0:
            raise ContractError("measured_steps must be a positive integer")
        if not callable(clock):
            raise ContractError("efficiency clock must be callable")
        self.warmup_steps = int(warmup_steps)
        self.measured_steps = int(measured_steps)
        self.clock = clock

    def measure(
        self,
        step: Callable[[], Any],
        examples_per_step: int,
        parameter_source: Any,
        device: str,
        input_fingerprint: str,
    ) -> EfficiencyReport:
        if not callable(step):
            raise ContractError("efficiency step must be callable")
        if isinstance(examples_per_step, bool) or int(examples_per_step) <= 0:
            raise ContractError("examples_per_step must be a positive integer")
        if not hasattr(parameter_source, "parameters") or not callable(parameter_source.parameters):
            raise ContractError("parameter_source must expose parameters()")
        canonical_device = str(device)
        if canonical_device != "cpu" and not canonical_device.startswith("cuda"):
            raise ContractError("efficiency device must be cpu or cuda")
        if canonical_device.startswith("cuda") and not torch.cuda.is_available():
            raise ContractError("CUDA efficiency measurement requires an available CUDA device")
        if not str(input_fingerprint):
            raise ContractError("efficiency input_fingerprint must be non-empty")

        parameters = tuple(parameter_source.parameters())
        total_parameters = sum(int(parameter.numel()) for parameter in parameters)
        trainable_parameters = sum(
            int(parameter.numel()) for parameter in parameters if parameter.requires_grad
        )
        for _ in range(self.warmup_steps):
            step()

        peak_memory: Optional[int] = None
        if canonical_device.startswith("cuda"):
            torch.cuda.synchronize(canonical_device)
            torch.cuda.reset_peak_memory_stats(canonical_device)
        started = float(self.clock())
        for _ in range(self.measured_steps):
            step()
        if canonical_device.startswith("cuda"):
            torch.cuda.synchronize(canonical_device)
            peak_memory = int(torch.cuda.max_memory_allocated(canonical_device))
        elapsed = float(self.clock()) - started
        if elapsed <= 0.0:
            raise ContractError("efficiency clock produced a non-positive elapsed time")

        payload: Dict[str, Any] = {
            "device": canonical_device,
            "accelerator_name": (
                torch.cuda.get_device_name(canonical_device)
                if canonical_device.startswith("cuda")
                else None
            ),
            "torch_version": str(torch.__version__),
            "cuda_version": (str(torch.version.cuda) if torch.version.cuda is not None else None),
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "examples_per_step": int(examples_per_step),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "total_seconds": elapsed,
            "latency_ms": elapsed * 1000.0 / self.measured_steps,
            "examples_per_second": self.measured_steps * int(examples_per_step) / elapsed,
            "peak_memory_bytes": peak_memory,
            "input_fingerprint": str(input_fingerprint),
        }
        return EfficiencyReport(fingerprint=_digest(payload), **payload)


def save_efficiency_report(path: Path, report: EfficiencyReport) -> Path:
    """Atomically persist a versioned report with an outer integrity fingerprint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report_payload = report.to_dict()
    payload = {
        "schema": "mmctr-efficiency-report-v1",
        "report": report_payload,
    }
    payload["manifest_fingerprint"] = _digest(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}-".format(destination.name), dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(destination))
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return destination


def load_efficiency_report(path: Path) -> EfficiencyReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "mmctr-efficiency-report-v1":
        raise ContractError("unsupported efficiency report schema")
    manifest_fingerprint = payload.pop("manifest_fingerprint", None)
    if manifest_fingerprint != _digest(payload):
        raise ContractError("efficiency report manifest fingerprint mismatch")
    values = payload.get("report")
    if not isinstance(values, dict):
        raise ContractError("efficiency report is missing report values")
    try:
        report = EfficiencyReport(
            device=str(values["device"]),
            accelerator_name=(
                None if values["accelerator_name"] is None else str(values["accelerator_name"])
            ),
            torch_version=str(values["torch_version"]),
            cuda_version=(None if values["cuda_version"] is None else str(values["cuda_version"])),
            warmup_steps=int(values["warmup_steps"]),
            measured_steps=int(values["measured_steps"]),
            examples_per_step=int(values["examples_per_step"]),
            total_parameters=int(values["total_parameters"]),
            trainable_parameters=int(values["trainable_parameters"]),
            total_seconds=float(values["total_seconds"]),
            latency_ms=float(values["latency_ms"]),
            examples_per_second=float(values["examples_per_second"]),
            peak_memory_bytes=(
                None if values["peak_memory_bytes"] is None else int(values["peak_memory_bytes"])
            ),
            input_fingerprint=str(values["input_fingerprint"]),
            fingerprint=str(values["fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("invalid efficiency report values") from error
    if report.fingerprint != _digest(_report_payload(report)):
        raise ContractError("efficiency report fingerprint mismatch")
    if (
        report.warmup_steps < 0
        or report.measured_steps <= 0
        or report.examples_per_step <= 0
        or report.total_parameters < report.trainable_parameters
        or report.trainable_parameters < 0
        or report.total_seconds <= 0.0
        or report.latency_ms <= 0.0
        or report.examples_per_second <= 0.0
        or not report.input_fingerprint
        or not report.torch_version
        or (report.device.startswith("cuda") and not report.accelerator_name)
    ):
        raise ContractError("invalid efficiency report values")
    return report


__all__ = [
    "EfficiencyProtocol",
    "EfficiencyReport",
    "load_efficiency_report",
    "save_efficiency_report",
]
