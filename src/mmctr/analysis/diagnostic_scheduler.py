"""Eight-GPU resumable diagnostic queue with a main-table completion prerequisite."""

import argparse
import csv
import io
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict

from mmctr.core import ContractError
from .diagnostic_execution import baseline_auc, main_table_complete, write_json
from .benchmark_protocols import compare_auc
from mmctr.utils.run_context import _atomic_write_text


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def source_hashes(root):
    return {
        str(path.relative_to(root)): digest(path)
        for pattern in ("src/mmctr/**/*.py", "configs/**/*.yaml")
        for path in sorted(root.glob(pattern))
    }


def task_result_path(output, task):
    return output / task["family"] / task["dataset"] / task["task_id"] / "result.json"


def is_pending_protocol(task):
    return (task["family"], task["dataset"], task["model"], task["intervention"].get("method")) == (
        "fusion",
        "microlens",
        "pamd",
        "src",
    )


def collect_result(output, task, baseline):
    path = task_result_path(output, task)
    result = json.loads(path.read_text())
    if result["task"] != task or result["status"] not in {"completed", "not_evaluable"}:
        raise ContractError("worker did not produce a matching terminal diagnostic result")
    from mmctr.utils.run_context import config_fingerprint

    expected = config_fingerprint({"task": task, "baseline": baseline}, 64)
    if result["input_hash"] != expected:
        raise ContractError("worker result does not match frozen input")
    comparator = baseline_auc(task, baseline)
    if task["family"] == "cold-start" and result["status"] == "completed":
        from .cold_populations import COLD_PROTOCOL

        if task.get("cold_protocol") != COLD_PROTOCOL:
            raise ContractError("legacy cold results cannot resume under the paper protocol")
        comparator = result["summary"]["full_test_auc"]
    if result["comparison"]["baseline_test_auc"] != comparator:
        raise ContractError("worker comparator differs from main table")
    if result["status"] == "completed" and result["comparison"]["test_auc"] is None:
        raise ContractError("completed worker result is missing test AUC")
    auc = result["comparison"]["test_auc"]
    if result["comparison"] != compare_auc(auc, comparator):
        raise ContractError("worker AUC deltas do not match the recorded same-seed comparison")
    if result["status"] == "completed" and result["summary"]["test_auc"] != auc:
        raise ContractError("worker summary and comparison AUC disagree")
    return result


def save_scoreboard(output, jobs, tasks, filename="cross-dataset-diagnostics.csv"):
    fields = [
        "task_id",
        "dataset",
        "model",
        "family",
        "seed",
        "intervention",
        "status",
        "test_auc",
        "baseline_test_auc",
        "auc_delta",
        "auc_absolute_difference",
        "relative_change_percent",
        "direction",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for job in jobs:
        task = tasks[job["task_id"]]
        row = {k: task[k] for k in ("task_id", "dataset", "model", "family", "seed")}
        row.update(
            status=job["status"],
            intervention=json.dumps(task["intervention"], sort_keys=True),
            baseline_test_auc=task.get("baseline_test_auc"),
        )
        row.update(job.get("comparison", {}))
        writer.writerow(row)
    _atomic_write_text(output / filename, buffer.getvalue())
    if filename == "cross-dataset-diagnostics.csv":
        for dataset in sorted({t["dataset"] for t in tasks.values()}):
            selected = [j for j in jobs if tasks[j["task_id"]]["dataset"] == dataset]
            save_scoreboard(output, selected, tasks, dataset + "-analysis.csv")


def eligible_jobs(jobs, tasks, dataset_order):
    """Do not cross the dataset barrier until every earlier task succeeds."""
    queued = [j for j in jobs if j["status"] == "queued"]
    if not dataset_order:
        return queued
    for dataset in dataset_order:
        group = [j for j in jobs if tasks[j["task_id"]]["dataset"] == dataset]
        if any(j["status"] not in {"completed", "not_evaluable"} for j in group):
            return [j for j in queued if tasks[j["task_id"]]["dataset"] == dataset]
    return []


def run_queue(root, matrix_path, baseline_path, output, gpus, poll_seconds=15):
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "cross-dataset-diagnostics.json"
    matrix = json.loads(matrix_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    tasks = {t["task_id"]: t for t in matrix["tasks"]}
    if len(tasks) != len(matrix["tasks"]):
        raise ContractError("diagnostic matrix has duplicate task IDs")
    frozen = {
        "matrix_sha256": digest(matrix_path),
        "baseline_sha256": digest(baseline_path),
        "sources": source_hashes(root),
    }
    with (output / ".diagnostic-scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["frozen"] != frozen:
                raise ContractError("queue inputs changed; inspect before resuming")
        else:
            state = {
                "schema": "cross-dataset-diagnostic-queue-v1",
                "frozen": frozen,
                "gpus": gpus,
                "jobs": [
                    {
                        "task_id": key,
                        "status": "pending_protocol" if is_pending_protocol(t) else "queued",
                    }
                    for key, t in tasks.items()
                ],
            }
        if state["gpus"] != gpus:
            raise ContractError("GPU assignment differs from the saved queue")
        state["launcher_pid"] = os.getpid()
        children: Dict[int, subprocess.Popen] = {}
        log_root = output / "diagnostic-launcher-logs"
        log_root.mkdir(exist_ok=True)
        while True:
            for job in state["jobs"]:
                if job["status"] != "running":
                    continue
                pid = job["pid"]
                proc = children.get(pid)
                if proc is not None and proc.poll() is None:
                    continue
                if proc is None and process_identity(pid) == job["process_identity"]:
                    continue
                if proc is not None:
                    job["returncode"] = proc.returncode
                try:
                    if job.get("returncode", 0) != 0:
                        raise RuntimeError("worker exited with code " + str(job["returncode"]))
                    result = collect_result(output, tasks[job["task_id"]], baseline)
                    job.update(status=result["status"], comparison=result["comparison"])
                except Exception as error:
                    job.update(status="failed", error=repr(error))
                job["ended_at"] = time.time()
                print(job["task_id"], job["status"], flush=True)
            active_gpus = {j["gpu"] for j in state["jobs"] if j["status"] == "running"}
            queued = eligible_jobs(state["jobs"], tasks, matrix.get("dataset_order"))
            ready = main_table_complete(baseline)
            state["status"] = "running" if ready else "waiting_for_main_table"
            if ready and queued:
                if source_hashes(root) != frozen["sources"]:
                    raise ContractError("runtime sources changed; refusing new worker launches")
                for gpu in gpus:
                    if gpu in active_gpus or not queued:
                        continue
                    job = queued.pop(0)
                    log_path = log_root / (job["task_id"] + ".log")
                    command = [
                        sys.executable,
                        "-m",
                        "mmctr.analysis.diagnostic_execution",
                        "--matrix",
                        str(matrix_path),
                        "--baseline",
                        str(baseline_path),
                        "--task-id",
                        job["task_id"],
                        "--output-root",
                        str(output),
                        "--cuda",
                        "0",
                    ]
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=str(gpu),
                        PYTHONPATH=str(root / "src"),
                        OMP_NUM_THREADS="2",
                        MKL_NUM_THREADS="2",
                        OPENBLAS_NUM_THREADS="2",
                        TMPDIR=str(root / ".tmp"),
                    )
                    with log_path.open("ab") as log:
                        proc = subprocess.Popen(
                            command,
                            cwd=str(root),
                            env=env,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    children[proc.pid] = proc
                    identity = process_identity(proc.pid)
                    if identity is None:
                        proc.wait()
                        job.update(status="failed", error="worker terminated before registration")
                    else:
                        job.update(
                            status="running",
                            pid=proc.pid,
                            process_identity=identity,
                            gpu=gpu,
                            log=str(log_path),
                            started_at=time.time(),
                        )
                    # Persist every spawn before another task may start.
                    write_json(state_path, state)
                    print("started", job["task_id"], "gpu", gpu, "pid", proc.pid, flush=True)
            counts = Counter(j["status"] for j in state["jobs"])
            state["counts"] = dict(counts)
            state["updated_at"] = time.time()
            if not counts["running"] and (not counts["queued"] or not queued):
                state["status"] = (
                    "needs_attention"
                    if counts["failed"] or counts["pending_protocol"] or counts["queued"]
                    else "completed"
                )
                write_json(state_path, state)
                save_scoreboard(output, state["jobs"], tasks)
                break
            write_json(state_path, state)
            save_scoreboard(output, state["jobs"], tasks)
            time.sleep(poll_seconds)
        return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 8 or len(set(gpus)) != 8 or any(not g.isdigit() for g in gpus):
        raise ContractError("this experiment requires eight distinct GPU indices")
    root = Path(__file__).resolve().parents[3]
    state = run_queue(
        root, args.matrix.resolve(), args.baseline.resolve(), args.output_root.resolve(), gpus
    )
    print(json.dumps(state["counts"]), flush=True)


if __name__ == "__main__":
    main()
