import json

import pytest

from mmctr.analysis import diagnostic_scheduler as scheduler
from mmctr.core import ContractError


def queue_fixture(tmp_path, count=10):
    matrix = tmp_path / "matrix.json"
    baseline = tmp_path / "baseline.json"
    matrix.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "job-" + str(i),
                        "family": "alignment",
                        "dataset": "tiktok",
                        "model": "naml",
                        "seed": 2025,
                        "intervention": {"method": "cosine", "weight": 0.1},
                    }
                    for i in range(count)
                ]
            }
        )
    )
    baseline.write_text("{}")
    return matrix, baseline, tmp_path / "output"


def fake_runtime(monkeypatch, exit_code=0):
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 1000 + len(launched)
            self.returncode = None
            self.polls = 0
            self.gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            launched.append(self)

        def poll(self):
            self.polls += 1
            if self.polls >= 2:
                self.returncode = exit_code
            return self.returncode

    monkeypatch.setattr(scheduler.subprocess, "Popen", Process)
    monkeypatch.setattr(scheduler, "source_hashes", lambda root: {"fixture": "sha"})
    monkeypatch.setattr(scheduler, "process_identity", lambda pid: "start-" + str(pid))
    monkeypatch.setattr(
        scheduler,
        "collect_result",
        lambda *args: {"status": "completed", "comparison": {"test_auc": 0.8}},
    )
    return launched


def test_wait_gate_and_eight_gpu_capacity(tmp_path, monkeypatch):
    paths = queue_fixture(tmp_path, 18)
    launched = fake_runtime(monkeypatch)
    ticks = []
    monkeypatch.setattr(scheduler, "main_table_complete", lambda baseline: bool(ticks))

    def sleep(seconds):
        if not ticks:
            assert not launched
        active = [p for p in launched if p.returncode is None]
        assert len(active) <= 8
        assert len({p.gpu for p in active}) == len(active)
        ticks.append(len(active))

    monkeypatch.setattr(scheduler.time, "sleep", sleep)
    result = scheduler.run_queue(tmp_path, *paths, list(map(str, range(8))), 0)
    assert result["counts"] == {"completed": 18}
    assert len(launched) == 18 and max(ticks) == 8
    assert result["status"] == "completed"


def test_resume_adopts_live_workers_without_duplicate_spawns(tmp_path, monkeypatch):
    paths = queue_fixture(tmp_path, 8)
    launched = fake_runtime(monkeypatch)
    monkeypatch.setattr(scheduler, "main_table_complete", lambda baseline: True)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(scheduler.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        scheduler.run_queue(tmp_path, *paths, list(map(str, range(8))), 0)
    assert len(launched) == 8
    ticks = []

    def sleep(seconds):
        ticks.append(1)
        monkeypatch.setattr(scheduler, "process_identity", lambda pid: None)

    monkeypatch.setattr(scheduler.time, "sleep", sleep)
    result = scheduler.run_queue(tmp_path, *paths, list(map(str, range(8))), 0)
    assert ticks and result["counts"] == {"completed": 8}
    assert len(launched) == 8


def test_failures_are_preserved_without_automatic_retries(tmp_path, monkeypatch):
    paths = queue_fixture(tmp_path, 3)
    launched = fake_runtime(monkeypatch, exit_code=1)
    monkeypatch.setattr(scheduler, "main_table_complete", lambda baseline: True)
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: None)
    result = scheduler.run_queue(tmp_path, *paths, list(map(str, range(8))), 0)
    assert result["counts"] == {"failed": 3}
    assert result["status"] == "needs_attention"
    result = scheduler.run_queue(tmp_path, *paths, list(map(str, range(8))), 0)
    assert len(launched) == 3
    assert result["counts"] == {"failed": 3}


def test_changed_queue_inputs_cannot_resume(tmp_path, monkeypatch):
    paths = queue_fixture(tmp_path, 1)
    fake_runtime(monkeypatch)
    monkeypatch.setattr(scheduler, "main_table_complete", lambda baseline: True)
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: None)
    scheduler.run_queue(tmp_path, *paths, list(map(str, range(8))), 0)
    paths[1].write_text('{"changed":true}')
    with pytest.raises(ContractError, match="inputs changed"):
        scheduler.run_queue(tmp_path, *paths, list(map(str, range(8))), 0)


def test_dataset_barrier_waits_for_success_and_blocks_failure():
    tasks = {"a": {"dataset": "tiktok"}, "b": {"dataset": "microlens"}}
    jobs = [{"task_id": "a", "status": "running"}, {"task_id": "b", "status": "queued"}]
    order = ["tiktok", "microlens"]
    assert scheduler.eligible_jobs(jobs, tasks, order) == []
    jobs[0]["status"] = "failed"
    assert scheduler.eligible_jobs(jobs, tasks, order) == []
    jobs[0]["status"] = "completed"
    assert scheduler.eligible_jobs(jobs, tasks, order) == [jobs[1]]


def test_per_dataset_scoreboards(tmp_path):
    import csv

    tasks = {
        d: {
            "task_id": d,
            "dataset": d,
            "model": "naml",
            "family": "alignment",
            "seed": 2025,
            "intervention": {},
        }
        for d in ["tiktok", "microlens"]
    }
    jobs = [{"task_id": d, "status": "queued"} for d in tasks]
    scheduler.save_scoreboard(tmp_path, jobs, tasks)
    for d in tasks:
        with (tmp_path / (d + "-analysis.csv")).open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1 and rows[0]["dataset"] == d
