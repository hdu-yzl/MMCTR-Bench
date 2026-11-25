"""Selected methods receive independent budgets and use one parameter root."""

import subprocess
from pathlib import Path

import pytest

from mmctr.experiments import tune_models


def test_selected_pairs_have_unique_studies_and_requested_budgets():
    args = tune_models.build_parser().parse_args(
        [
            "--datasets",
            "antm2c,ebnerd",
            "--models",
            "dnn,mcca",
            "--max-trials",
            "3",
            "--max-gpu-hours",
            "2",
            "--max-hours",
            "1.5",
            "--study-prefix",
            "selection",
        ]
    )
    commands = tune_models.commands_for(args)
    assert len(commands) == 4
    assert len({c[c.index("--study-id") + 1] for c in commands}) == 4
    for c in commands:
        assert c[c.index("--model-name") + 1] in ("dnn", "psrq")
        assert c[c.index("--max-trials") + 1] == "3"
        assert c[c.index("--max-gpu-hours") + 1] == "2.0"
        assert c[c.index("--max-hours") + 1] == "1.5"
        assert Path(c[c.index("--model-config-dir") + 1]).name == "models"


def test_paper_selection_covers_all_four_datasets():
    args = tune_models.build_parser().parse_args(["--datasets", "all", "--models", "paper"])
    assert len(tune_models.commands_for(args)) == 84


@pytest.mark.parametrize(
    "arguments",
    [
        ["--models", "dnn,unknown", "--datasets", "antm2c"],
        ["--models", "psrq,mcca", "--datasets", "antm2c"],
        ["--models", "dnn", "--datasets", "missing"],
        ["--models", "dnn", "--datasets", "antm2c", "--max-trials", "0"],
    ],
)
def test_bad_batch_does_not_launch_any_search(arguments, monkeypatch):
    monkeypatch.setattr(tune_models.subprocess, "run", lambda *a, **kw: pytest.fail("launched"))
    with pytest.raises(SystemExit):
        tune_models.main(arguments)


def test_dry_run_and_first_failure_do_not_launch_later_searches(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(tune_models.subprocess, "run", run)
    args = ["--datasets", "tiktok", "--models", "dnn,din"]
    assert tune_models.main(args + ["--dry-run"]) == 0
    assert not calls
    assert tune_models.main(args) == 7
    assert len(calls) == 1
