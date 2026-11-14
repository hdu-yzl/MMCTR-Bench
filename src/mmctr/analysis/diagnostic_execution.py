"""One diagnostic job with frozen main-table settings and explicit result provenance."""

import argparse
import json
from pathlib import Path

from mmctr.core import ContractError
from mmctr.models.common.registry import resolve_model_config
from mmctr.utils.run_context import _atomic_write_text, config_fingerprint
from .benchmark_protocols import (
    LegacyRobustnessLoader,
    PaperColdStartLoader,
    compare_auc,
    reserve_oov_embeddings,
)
from .cold_populations import COLD_PROTOCOL, build_cold_populations, training_exposure

TRAIN_RUNTIME_KEYS = frozenset({"cuda", "checkpoint_dir", "log_dir", "output_root"})
FAMILIES = frozenset({"cold-start", "robustness", "alignment", "fusion"})


def write_json(path, value):
    _atomic_write_text(Path(path), json.dumps(value, indent=2, allow_nan=False) + "\n")


def baseline_auc(task, baseline):
    matches = [
        job
        for job in baseline["jobs"]
        if (job["dataset"], job["model"], job["seed"])
        == (task["dataset"], task["model"], task["seed"])
    ]
    if len(matches) != 1 or matches[0]["status"] != "completed":
        raise ContractError("comparison requires exactly one completed same-seed main-table run")
    value = float(matches[0]["metrics"]["test_auc"])
    if value != task["baseline_test_auc"]:
        raise ContractError("task baseline AUC differs from frozen same-seed main-table run")
    return value


def main_table_complete(baseline):
    report = json.loads(Path(baseline["source"]).read_text())
    jobs = report["jobs"]
    identities = {(j["dataset"], j["model"], j["seed"]) for j in jobs}
    return len(jobs) == len(identities) == 420 and all(j["status"] == "completed" for j in jobs)


def validate_task(task, baseline):
    if task["family"] not in FAMILIES or task["dataset"] not in {
        "microlens",
        "tiktok",
        "antm2c",
        "ebnerd",
    }:
        raise ContractError("unsupported diagnostic family/dataset")
    if task["dataset"] in {"antm2c", "ebnerd"} and task["family"] != "cold-start":
        raise ContractError("dataset extension only permits paper cold diagnostics")
    if task["seed"] != 2025:
        raise ContractError("diagnostic protocol requires main-table seed 2025")
    if Path(task["task_id"]).name != task["task_id"] or task["task_id"] in {"", ".", ".."}:
        raise ContractError("task ID must be a safe path component")
    baseline_auc(task, baseline)
    intervention = task["intervention"]
    expected = {
        "cold-start": {"regime", "target"},
        "robustness": {"probability", "mask_seed"},
        "alignment": {"method", "weight"},
        "fusion": {"method"},
    }[task["family"]]
    if set(intervention) != expected:
        raise ContractError("diagnostic intervention fields differ from frozen protocol")
    if task["family"] == "cold-start":
        if task.get("cold_protocol") != COLD_PROTOCOL:
            raise ContractError("cold task must be regenerated under " + COLD_PROTOCOL)
        valid = intervention["regime"] in {"zero", "low_lt5"} and intervention["target"] == "union"
    elif task["family"] == "robustness":
        valid = (
            intervention["probability"] in {0.1, 0.3, 0.5, 0.7} and intervention["mask_seed"] == 42
        )
    elif task["family"] == "alignment":
        from .legacy_alignment import METHODS

        valid = intervention["method"] in METHODS and intervention["weight"] in {0.1, 0.3, 0.5, 0.7}
    else:
        from .fusion_operators import PAPER_FUSIONS

        valid = intervention["method"] in PAPER_FUSIONS
    if not valid:
        raise ContractError("intervention is outside the confirmed experiment grid")
    if (task["family"], task["dataset"], task["model"], intervention.get("method")) == (
        "fusion",
        "microlens",
        "pamd",
        "src",
    ):
        raise ContractError("single-pair SRC protocol is awaiting user decision")


def assert_frozen_settings(trainer, task, baseline):
    data = baseline["data"][task["dataset"]]
    expected_model = dict(baseline["model_catalog"][task["model"]])
    expected_model["model_name"] = task["model"]
    expected_model = resolve_model_config(expected_model, data)
    if trainer.model_config != expected_model:
        raise ContractError("current model hyperparameters differ from the frozen main table")
    if trainer.data_config[task["dataset"]] != data:
        raise ContractError("current dataset configuration differs from the frozen main table")
    expected_train = dict(baseline["training_defaults"], seed=task["seed"])
    keys = (set(expected_train) | set(trainer.train_config)) - TRAIN_RUNTIME_KEYS
    if any(trainer.train_config.get(k) != expected_train.get(k) for k in keys):
        raise ContractError("current training hyperparameters differ from the frozen main table")


def prepare_trainer(task, baseline, output_root, cuda):
    """Compose the native Trainer, changing only the requested diagnostic intervention."""
    from mmctr.training.entrypoint import Trainer
    from mmctr.training.optimizers import build_optimizer
    from mmctr.utils.helper import setup_seed

    validate_task(task, baseline)
    masks = populations = None
    if task["family"] == "cold-start":
        exposure = training_exposure(baseline["data"][task["dataset"]]["data_dir"])
        masks, populations = build_cold_populations(
            baseline["data"][task["dataset"]]["data_dir"], **task["intervention"], exposure=exposure
        )
        if populations != task["population"]:
            raise ContractError("cold populations differ from the frozen audit")
        if not populations["test"]["auc_evaluable"]:
            return None, populations
    trainer = Trainer(
        dataset_name=task["dataset"],
        model_name=task["model"],
        cuda=cuda,
        seed=task["seed"],
        save_checkpoints=False,
        dataset_config_path=baseline.get("dataset_config_path"),
        output_root=str(
            Path(output_root) / task["family"] / task["dataset"] / task["task_id"] / "runs"
        ),
    )
    try:
        assert_frozen_settings(trainer, task, baseline)
        family, config = task["family"], task["intervention"]
        if family == "cold-start":
            assert masks is not None
            test_mask = masks["test"]
            # Keep full validation for early stopping. Attach OOV only after best restore.
            _, seen, offset = exposure

            def prepare_final_test():
                oov = reserve_oov_embeddings(trainer.model)
                trainer.final_test_loader = PaperColdStartLoader(
                    trainer.dataloader, test_mask, seen, offset, oov
                )
                trainer.final_full_test_loader = PaperColdStartLoader(
                    trainer.dataloader, None, seen, offset, oov
                )

            trainer.prepare_final_test = prepare_final_test
        elif family == "robustness":
            trainer.dataloader = LegacyRobustnessLoader(trainer.dataloader, **config)
        elif family == "alignment":
            from .alignment_features import alignment_features
            from .alignment_training import attach_alignment
            from .legacy_alignment import LegacyAlignmentObjective

            objective = LegacyAlignmentObjective(
                config["method"],
                trainer.model.feature_names,
                trainer.model.projection_dim,
                config["weight"],
            )
            attach_alignment(trainer, objective, alignment_features)
        elif family == "fusion":
            from .fusion_backbones import build_fusion_study_model

            # Initialize the intervention from the same seed, independent of unused native init.
            setup_seed(task["seed"])
            model = build_fusion_study_model(
                task["model"],
                config["method"],
                trainer.model_config,
                trainer.data_config[task["dataset"]],
                trainer.train_config["quantization_artifact_dir"],
            ).to(trainer.engine.device)
            trainer.model = trainer.engine.model = model
            trainer.engine.optimizer = build_optimizer(
                model,
                trainer.train_config["optim"],
                trainer.train_config["lr"],
                trainer.train_config["l2"],
            )
            # The legacy fusion wrappers do not retain native GMMF's GAN phases.
            trainer.engine.alternating_phases = ()
        resolved = {
            "model": trainer.model_config,
            "data": trainer.data_config[task["dataset"]],
            "train": trainer.train_config,
            "analysis": task,
            "run": trainer.run_context.runtime_config(),
        }
        trainer.run_context.write_resolved_config(resolved)
        trainer.run_context.update_metadata(
            {
                "diagnostic_task_id": task["task_id"],
                "diagnostic_family": family,
                "diagnostic_config_hash": config_fingerprint(resolved, 64),
                "baseline_test_auc": baseline_auc(task, baseline),
                "structure_policy": getattr(trainer.model, "structure_policy", "native_backbone"),
                "cold_protocol": task.get("cold_protocol"),
                "adv_protocol": "complete_discriminator_updates"
                if (family == "alignment" and config["method"] == "adversarial")
                else None,
            }
        )
        return trainer, populations
    except BaseException as error:
        trainer.run_context.finalize("failed", error=repr(error))
        raise


def _run_task_unlocked(task, baseline, output_root, cuda, verify_only=False):
    validate_task(task, baseline)
    if not verify_only and not main_table_complete(baseline):
        raise ContractError("all 420 main-table experiments must finish before diagnostics start")
    root = Path(output_root) / task["family"] / task["dataset"] / task["task_id"]
    filename = "preflight-result.json" if verify_only else "result.json"
    result_path = root / filename
    digest = config_fingerprint({"task": task, "baseline": baseline}, 64)
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("input_hash") != digest:
            raise ContractError("existing result belongs to a different frozen protocol")
        if previous["status"] in {"completed", "not_evaluable", "verified"}:
            return previous
        raise ContractError("previous attempt failed; inspect it before an explicit retry")
    result = {
        "task": task,
        "input_hash": digest,
        "status": "running",
        "comparison": compare_auc(None, baseline_auc(task, baseline)),
    }
    write_json(result_path, result)
    trainer = None
    try:
        trainer, populations = prepare_trainer(task, baseline, output_root, cuda)
        if trainer is None:
            result.update(status="not_evaluable", population=populations)
        elif verify_only:
            # One actual update plus held-out forward checks, never a formal reported AUC.
            from itertools import islice

            trainer.engine.train_epoch(islice(trainer.dataloader.iter_batches("train"), 1), 0)
            trainer.model.eval()
            import torch

            if task["family"] == "cold-start":
                assert trainer.prepare_final_test is not None
                trainer.prepare_final_test()
            with torch.no_grad():
                for split in ("val", "test"):
                    loader = (
                        (trainer.final_test_loader or trainer.dataloader)
                        if split == "test"
                        else trainer.dataloader
                    )
                    batch = next(loader.iter_batches(split)).to(trainer.engine.device)
                    if not torch.isfinite(trainer.model(batch).logits).all():
                        raise ContractError("nonfinite diagnostic verification logits")
            trainer.run_context.finalize("completed", summary={"preflight_only": True})
            result.update(status="verified", run_dir=str(trainer.run_context.root_dir))
        else:
            trainer.run()
            summary = json.loads(trainer.run_context.summary_path.read_text())
            comparator = summary.get("full_test_auc", baseline_auc(task, baseline))
            result.update(
                status="completed",
                summary=summary,
                run_dir=str(trainer.run_context.root_dir),
                comparison=compare_auc(summary["test_auc"], comparator),
            )
            if task["family"] == "cold-start":
                result["comparison_source"] = "same_frozen_weights_full_test_with_oov"
                result["main_table_reference_auc"] = baseline_auc(task, baseline)
        if trainer is not None and list(trainer.run_context.root_dir.rglob("*.pt")):
            raise ContractError("diagnostic run unexpectedly wrote a pt checkpoint")
        write_json(result_path, result)
        return result
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        write_json(result_path, result)
        if trainer is not None:
            trainer.run_context.finalize("failed", error=repr(error))
        raise


def run_task(task, baseline, output_root, cuda, verify_only=False):
    """Use a nonblocking per-job lock so duplicate invocations cannot train twice."""
    import fcntl

    validate_task(task, baseline)
    root = Path(output_root) / task["family"] / task["dataset"] / task["task_id"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_task_unlocked(task, baseline, output_root, cuda, verify_only)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    matrix = json.loads(Path(args.matrix).read_text())
    matches = [t for t in matrix["tasks"] if t["task_id"] == args.task_id]
    if len(matches) != 1:
        raise ContractError("task ID must select exactly one matrix entry")
    baseline = json.loads(Path(args.baseline).read_text())
    result = run_task(matches[0], baseline, args.output_root, args.cuda, args.verify_only)
    print(json.dumps({"task_id": args.task_id, "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
