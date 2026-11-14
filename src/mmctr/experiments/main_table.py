"""Repeat the paper table in isolated GPU processes and retain all five seeds."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from mmctr.config import (
    load_local_paths,
    load_training_config,
    load_yaml_mapping,
    resolve_dataset_config,
)
from mmctr.data import available_datasets
from mmctr.models import available_models
from mmctr.utils.run_context import _atomic_write_text


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEEDS = (2025, 2026, 2027, 2028, 2029)
METRICS = ("test_auc", "test_loss")


def _csv_values(value: str) -> List[str]:
    values = [part.strip() for part in value.split(",")]
    if not all(values) or len(set(values)) != len(values):
        raise ValueError("comma-separated values must be nonempty and unique")
    return values


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> Dict[str, str]:
    """Detect code/config changes between launches and across resumed invocations."""
    paths = list((ROOT / "src/mmctr").rglob("*.py"))
    paths += list((ROOT / "configs").rglob("*.yaml"))
    return {str(path.relative_to(ROOT)): _digest(path) for path in sorted(paths)}


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    datasets = _csv_values(args.datasets)
    registered = available_models()
    if args.models == "paper":
        models = [name for name in registered if name not in ("dnn_mm", "dnn_mm_seq")]
    else:
        models = list(registered) if args.models == "all" else _csv_values(args.models)
    if set(datasets) - set(available_datasets()) or set(models) - set(registered):
        raise ValueError("unknown dataset or model; use canonical registry names")
    seeds = [int(value) for value in _csv_values(args.seeds)]
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError("exactly five distinct seeds in [0, 2**32) are required")
    if getattr(args, "jobs_per_gpu", 1) < 1:
        raise ValueError("--jobs-per-gpu must be positive")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    train = load_training_config(ROOT / "configs/training/default.yaml")
    catalog = load_yaml_mapping(ROOT / "configs/datasets/catalog.yaml")
    local = load_local_paths(ROOT / "configs/local/paths.yaml") if args.use_local_data else None
    data = {}
    inputs = {}
    missing = []
    for dataset in datasets:
        if local is not None and dataset not in local.datasets:
            raise ValueError("local dataset path is missing: " + dataset)
        config = resolve_dataset_config(dataset, catalog[dataset], ROOT, local_paths=local)
        data[dataset] = config
        required = [Path(config["data_dir"]) / "manifest.json"]
        artifact_root = train.quantization_artifact_dir
        if "qarm" in models:
            required += [
                artifact_root / "rq" / dataset / (modality + ".npz")
                for modality in config["use_mm_features"]
                if modality != "id"
            ]
        if "psrq" in models:
            required.append(artifact_root / "psrq" / dataset / "model.npz")
        for path in required:
            if path.is_file():
                inputs[str(path)] = _digest(path)
            else:
                missing.append(str(path))
    return {
        "schema_version": 1,
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "std_ddof": 1,
        "num_threads": args.num_threads,
        "jobs_per_gpu": getattr(args, "jobs_per_gpu", 1),
        "no_checkpoints": getattr(args, "no_checkpoints", False),
        "wait_for_inputs": getattr(args, "wait_for_inputs", False),
        "use_local_data": args.use_local_data,
        "python": sys.executable,
        "versions": {name: metadata.version(name) for name in ("torch", "numpy", "PyYAML")},
        "training_defaults": train.to_dict(),
        "model_catalog": load_yaml_mapping(ROOT / "configs/models/catalog.yaml"),
        "data": data,
        "sources": source_hashes(),
        "inputs": inputs,
        "missing_inputs": missing,
    }


def new_jobs(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"dataset": dataset, "model": model, "seed": seed, "status": "pending", "attempts": []}
        for dataset in plan["datasets"]
        for model in plan["models"]
        for seed in plan["seeds"]
    ]


def aggregate(plan: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Expose partial counts/raw values, but publish statistics only for all five seeds."""
    rows = []
    for dataset in plan["datasets"]:
        for model in plan["models"]:
            group = [j for j in jobs if j["dataset"] == dataset and j["model"] == model]
            valid = [j for j in group if j["status"] == "completed"]
            complete = len(valid) == 5 and {j["seed"] for j in valid} == set(plan["seeds"])
            row = {
                "dataset": dataset,
                "model": model,
                "status": "complete" if complete else "incomplete",
                "n_success": len(valid),
                "n_failed": sum(j["status"] == "failed" for j in group),
                "std_ddof": 1,
            }
            for metric in METRICS:
                values = [float(j["metrics"][metric]) for j in valid]
                if any(not math.isfinite(value) for value in values):
                    raise ValueError("nonfinite metric in report")
                row[metric + "_mean"] = statistics.mean(values) if complete else None
                row[metric + "_std"] = statistics.stdev(values) if complete else None
                row[metric + "_mean_std"] = (
                    "{:.6f} ± {:.6f}".format(statistics.mean(values), statistics.stdev(values))
                    if complete
                    else ""
                )
                for seed in plan["seeds"]:
                    row[metric + "_seed_" + str(seed)] = next(
                        (j["metrics"][metric] for j in valid if j["seed"] == seed), None
                    )
            rows.append(row)
    return rows


def save_report(output: Path, plan: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> None:
    rows = aggregate(plan, jobs)
    _atomic_write_text(
        output / "main_table.json",
        json.dumps({"plan": plan, "table": rows, "jobs": jobs}, indent=2, allow_nan=False) + "\n",
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(output / "main_table.csv", stream.getvalue())


def command_for(job: Mapping[str, Any], plan: Mapping[str, Any], output: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "mmctr.cli",
        "train",
        "--dataset-name",
        job["dataset"],
        "--model-name",
        job["model"],
        "--seed",
        str(job["seed"]),
        "--cuda",
        "0",
        "--num-threads",
        str(plan["num_threads"]),
        "--output-root",
        str(output),
    ]
    if plan.get("no_checkpoints", False):
        command.append("--no-checkpoints")
    if plan["use_local_data"]:
        command.append("--use-local-data")
    return command


def collect_result(job: Mapping[str, Any], attempt: Mapping[str, Any]) -> Dict[str, float]:
    root = Path(attempt["output"])
    summaries = list(root.glob("training/*/*/*/summary.json"))
    if len(summaries) != 1:
        raise ValueError("expected exactly one training summary under " + str(root))
    path = summaries[0]
    run = json.loads((path.parent / "run_metadata.json").read_text())
    config = load_yaml_mapping(path.parent / "resolved_config.yaml")
    if (
        run["status"] != "completed"
        or run["seed"] != job["seed"]
        or config["train"]["seed"] != job["seed"]
        or config["model"]["model_name"] != job["model"]
        or path.parents[2].name != job["dataset"]
    ):
        raise ValueError("training result does not match the requested job")
    summary = json.loads(path.read_text())
    metrics = {name: float(summary[name]) for name in METRICS}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("training produced nonfinite metrics")
    return metrics


def hydrate_inputs(plan: Dict[str, Any]) -> None:
    """Anchor newly published atomic prerequisite artifacts before consumers start."""
    if not plan.get("wait_for_inputs", False):
        return
    for name in list(plan["missing_inputs"]):
        path = Path(name)
        if path.is_file():
            plan["inputs"][name] = _digest(path)
            plan["missing_inputs"].remove(name)


def job_ready(job: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
    """Other datasets/models can proceed while a quantizer is being fitted."""
    if not plan.get("missing_inputs"):
        return True
    dataset = job["dataset"]
    config = plan["data"][dataset]
    required = [Path(config["data_dir"]) / "manifest.json"]
    artifacts = Path(plan["training_defaults"]["quantization_artifact_dir"])
    if job["model"] == "qarm":
        required += [
            artifacts / "rq" / dataset / (name + ".npz")
            for name in config["use_mm_features"]
            if name != "id"
        ]
    if job["model"] == "psrq":
        required.append(artifacts / "psrq" / dataset / "model.npz")
    return not set(map(str, required)).intersection(plan["missing_inputs"])


def execute(plan: Dict[str, Any], jobs: List[Dict[str, Any]], output: Path, gpus: List[str]) -> int:
    """Dynamically fill GPU slots; failures do not cancel unrelated jobs."""
    pending = deque(j for j in jobs if j["status"] != "completed")
    active: Dict[str, Any] = {}
    slots = [(gpu, str(slot)) for slot in range(plan.get("jobs_per_gpu", 1)) for gpu in gpus]
    waiting_since = time.monotonic()
    try:
        while pending or active:
            hydrate_inputs(plan)
            if plan.get("missing_inputs") and time.monotonic() - waiting_since > 7200:
                raise ValueError("prerequisite artifacts still missing after two hours")
            for gpu, slot in slots:
                slot_id = gpu + ":" + slot
                if not pending or slot_id in active:
                    continue
                ready = next(
                    (i for i, candidate in enumerate(pending) if job_ready(candidate, plan)), None
                )
                if ready is None:
                    continue
                if source_hashes() != plan["sources"]:
                    raise ValueError(
                        "code/config changed during execution; use a new output directory"
                    )
                job = pending[ready]
                del pending[ready]
                attempt_dir = output / "attempts" / uuid.uuid4().hex
                attempt_dir.mkdir(parents=True)
                attempt: Dict[str, Any] = {
                    "gpu": gpu,
                    "slot": slot,
                    "output": str(attempt_dir),
                    "status": "running",
                }
                job["attempts"].append(attempt)
                job["status"] = "running"
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment["MMCTR_GPU_SLOT"] = slot
                environment["PYTHONHASHSEED"] = str(job["seed"])
                log = attempt_dir / "console.log"
                attempt["log"] = str(log)
                try:
                    with log.open("w") as stream:
                        process = subprocess.Popen(
                            command_for(job, plan, attempt_dir),
                            cwd=str(ROOT),
                            env=environment,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    attempt["pid"] = process.pid
                    active[slot_id] = (process, job, attempt)
                    print(
                        "Started {dataset}/{model} seed={seed} GPU=".format(**job) + gpu, flush=True
                    )
                except OSError:
                    job["status"] = attempt["status"] = "failed"
                    attempt["error"] = traceback.format_exc()
                save_report(output, plan, jobs)
            for slot_id, (process, job, attempt) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[slot_id]
                attempt["returncode"] = code
                try:
                    if code:
                        raise RuntimeError(
                            "training exited with code {}; see {}".format(code, attempt["log"])
                        )
                    if source_hashes() != plan["sources"]:
                        raise ValueError("code/config changed during training")
                    job["metrics"] = collect_result(job, attempt)
                    job["status"] = attempt["status"] = "completed"
                except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                    job["status"] = attempt["status"] = "failed"
                    attempt["error"] = traceback.format_exc()
                print("{status}: {dataset}/{model} seed={seed}".format(**job), flush=True)
                save_report(output, plan, jobs)
            if active or pending:
                time.sleep(0.2 if active else 1.0)
    finally:
        # SIGINT/SIGTERM must not leave expensive training subprocesses running.
        for process, job, attempt in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, job, attempt in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            job["status"] = attempt["status"] = "interrupted"
        save_report(output, plan, jobs)
    return int(any(j["status"] != "completed" for j in jobs))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="antm2c,microlens,tiktok,ebnerd")
    parser.add_argument(
        "--models", default="paper", help="paper (21), all (23), or comma-separated names"
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--gpus",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
        help="physical CUDA indices/UUIDs",
    )
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument(
        "--wait-for-inputs",
        action="store_true",
        help="run ready jobs while prerequisites are fitted (2h timeout)",
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/main_table"))
    parser.add_argument("--use-local-data", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="reuse successful seeds; retry other seeds"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print plan and missing prerequisites; no training"
    )
    args = parser.parse_args(argv)
    try:
        gpus = _csv_values(args.gpus)
        plan = build_plan(args)
        jobs = new_jobs(plan)
        print(
            "{} datasets x {} models x 5 seeds = {} runs; {} GPU slots".format(
                len(plan["datasets"]),
                len(plan["models"]),
                len(jobs),
                len(gpus) * plan.get("jobs_per_gpu", 1),
            ),
            flush=True,
        )
        if args.dry_run:
            print(json.dumps({"gpus": gpus, "plan": plan, "jobs": jobs}, indent=2))
            return 0
        if plan["missing_inputs"] and not args.wait_for_inputs:
            raise ValueError(
                "missing data/quantizer inputs:\n"
                + "\n".join(plan["missing_inputs"])
                + "\nFor missing quantizers, run scripts/pretrain_quantizers.sh "
                "--dataset NAME --gpu ID for each affected dataset."
            )
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        )
        aliases = {}
        for line in devices.splitlines():
            index, device_uuid = [value.strip() for value in line.split(",")]
            aliases[index] = aliases[device_uuid] = device_uuid
        if any(gpu not in aliases for gpu in gpus):
            raise ValueError("requested GPU is unavailable; inspect nvidia-smi")
        if len({aliases[gpu] for gpu in gpus}) != len(gpus):
            raise ValueError("GPU indices/UUIDs refer to the same physical GPU")
        output = args.output_dir.expanduser().resolve()
        existing_parent = output
        while not existing_parent.exists():
            existing_parent = existing_parent.parent
        if existing_parent.stat().st_dev != ROOT.stat().st_dev:
            raise ValueError("output directory must be on the project filesystem")
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".launcher.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = output / "main_table.json"
            if report.exists():
                if not args.resume:
                    raise ValueError("report already exists; use --resume or a new --output-dir")
                previous = json.loads(report.read_text())
                hydrate_inputs(previous["plan"])
                if previous["plan"] != plan:
                    raise ValueError(
                        "resume requires identical code, configs, data/artifacts, and seeds"
                    )
                identity_fields = ("dataset", "model", "seed")
                expected = [tuple(job[key] for key in identity_fields) for job in jobs]
                restored = previous["jobs"]
                actual = [tuple(job[key] for key in identity_fields) for job in restored]
                if actual != expected:
                    raise ValueError("saved job matrix does not match the requested plan")
                jobs = restored
                for job in jobs:
                    if job["status"] == "completed":
                        if collect_result(job, job["attempts"][-1]) != job["metrics"]:
                            raise ValueError("saved training results changed")
                    elif job["status"] == "running":
                        job["status"] = job["attempts"][-1]["status"] = "interrupted"
            save_report(output, plan, jobs)

            def interrupt(_signum, _frame):
                raise KeyboardInterrupt

            previous_handler = signal.signal(signal.SIGTERM, interrupt)
            try:
                result = execute(plan, jobs, output, gpus)
            finally:
                signal.signal(signal.SIGTERM, previous_handler)
            print("Summary: " + str(output / "main_table.csv"), flush=True)
            return result
    except KeyboardInterrupt:
        print("Interrupted; use --resume to continue.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
