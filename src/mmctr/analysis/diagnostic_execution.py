"""Tune and evaluate one diagnostic setting per invocation."""

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

DATASETS = ("antm2c", "ebnerd", "microlens", "tiktok")
FAMILIES = ("cold-start", "control", "robustness", "alignment", "fusion")
LEVELS = (0.1, 0.3, 0.5, 0.7)


def _setting(args):
    from mmctr.core import ContractError

    family = args.family
    if args.dataset_name in {"antm2c", "ebnerd"} and family not in {"cold-start", "control"}:
        raise ContractError("only cold-start and native control are supported for this dataset")
    intervention = {}
    if family == "cold-start":
        from .cold_populations import COLD_PROTOCOL

        intervention = {"target": "union", "regime": args.regime, "protocol": COLD_PROTOCOL}
    elif family == "robustness":
        intervention = {"probability": args.probability, "mask_seed": 42}
    elif family == "alignment":
        from .legacy_alignment import METHODS

        if args.method not in METHODS:
            raise ContractError("alignment method must be one of: " + ", ".join(sorted(METHODS)))
        intervention = {"method": args.method, "weight": args.weight}
    elif family == "fusion":
        from .fusion_operators import PAPER_FUSIONS

        if args.method not in PAPER_FUSIONS:
            raise ContractError("fusion method must be one of: " + ", ".join(sorted(PAPER_FUSIONS)))
        if (args.dataset_name, args.model_name, args.method) == ("microlens", "pamd", "src"):
            raise ContractError("SRC requires at least two inputs; MicroLens PAMD has only one")
        intervention = {"method": args.method}
    return {
        "study_id": args.study_id,
        "family": family,
        "dataset": args.dataset_name,
        "model": args.model_name,
        "seed": args.seed,
        "intervention": intervention,
    }


def _prepare_trainer(candidate, output_root, cuda, dataset_config_path):
    from mmctr.training.entrypoint import Trainer
    from mmctr.training.optimizers import build_optimizer
    from mmctr.utils.helper import setup_seed
    from .benchmark_protocols import LegacyRobustnessLoader

    task = candidate["diagnostic"]
    trainer = Trainer(
        dataset_name=task["dataset"],
        model_name=task["model"],
        cuda=cuda,
        seed=task["seed"],
        save_checkpoints=False,
        dataset_config_path=dataset_config_path,
        output_root=str(output_root),
        model_overrides=candidate.get("model"),
        training_overrides=candidate.get("training"),
    )
    try:
        family, config = task["family"], task["intervention"]
        if family == "robustness":
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
        trainer.run_context.write_resolved_config(
            {
                "model": trainer.model_config,
                "data": trainer.data_config[task["dataset"]],
                "train": trainer.train_config,
                "analysis": task,
                "run": trainer.run_context.runtime_config(),
            }
        )
        trainer.run_context.update_metadata(
            {
                "diagnostic": task,
                "structure_policy": getattr(trainer.model, "structure_policy", "native_backbone"),
                "adv_protocol": "complete_discriminator_updates"
                if (family == "alignment" and config["method"] == "adversarial")
                else None,
            }
        )
        return trainer
    except BaseException as error:
        trainer.run_context.finalize("failed", error=repr(error))
        raise


def _evaluate_cold_start(trainer, epoch, regime):
    """Audit and evaluate the test populations with one set of selected weights."""
    from mmctr.core import ContractError, RunResult
    from .benchmark_protocols import PaperColdStartLoader, compare_auc, reserve_oov_embeddings
    from .cold_populations import COLD_PROTOCOL, build_cold_populations, training_exposure

    root = trainer.data_config[trainer.dataset_name]["data_dir"]
    exposure = training_exposure(root)
    _, seen, offset = exposure
    masks, populations = build_cold_populations(
        root, "union", regime, exposure=exposure, splits=("test",)
    )
    oov_ids = reserve_oov_embeddings(trainer.model)

    def evaluate(selection):
        loader = PaperColdStartLoader(trainer.dataloader, selection, seen, offset, oov_ids)
        evaluation = trainer.engine.evaluate(loader, "test", epoch)
        if evaluation.metrics is None:
            raise ContractError("cold diagnostic test metrics were not produced")
        return evaluation.metrics

    full = evaluate(None)
    metrics = {"test_full_auc": full.auc, "test_full_loss": full.log_loss}
    population = populations["test"]
    subset_auc = None
    if population["auc_evaluable"]:
        subset = evaluate(masks["test"])
        subset_auc = subset.auc
        metrics.update(test_auc=subset.auc, test_loss=subset.log_loss)
    return RunResult(
        trainer.run_context.run_id,
        "completed",
        metrics,
        artifact_dir=trainer.run_context.root_dir,
        metadata={
            "cold_protocol": COLD_PROTOCOL,
            "subset_status": "evaluable" if population["auc_evaluable"] else "not_evaluable",
            "test_subset_population": population,
            "comparison": compare_auc(subset_auc, full.auc),
        },
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-name", choices=DATASETS, required=True)
    common.add_argument("--model-name", type=str.lower, required=True)
    common.add_argument("--dataset-config")
    common.add_argument("--candidates", type=Path, required=True)
    common.add_argument("--study-id", required=True)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--seed", type=int, default=2025)
    common.add_argument("--cuda", type=int, default=0, help="device index; use -1 for CPU")
    common.add_argument("--num-threads", type=int, default=1)
    subparsers = parser.add_subparsers(dest="family", required=True)
    families = {name: subparsers.add_parser(name, parents=[common]) for name in FAMILIES}
    families["cold-start"].add_argument("--regime", choices=("zero", "low_lt5"), required=True)
    families["robustness"].add_argument("--probability", type=float, choices=LEVELS, required=True)
    families["alignment"].add_argument("--method", type=str.lower, required=True)
    families["alignment"].add_argument("--weight", type=float, choices=LEVELS, required=True)
    families["fusion"].add_argument("--method", type=str.lower, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from mmctr.cli.app import _configure_threads

    args = _parser().parse_args(argv)
    _configure_threads(args.num_threads)

    from mmctr.config import load_dataset_catalog
    from mmctr.core import ContractError
    from mmctr.experiments.tuning import (
        ValidationOnlyTuner,
        load_tuning_candidates,
        run_validation_search,
    )
    from mmctr.training.entrypoint import PROJECT_ROOT

    setting = _setting(args)
    overrides = load_tuning_candidates(args.candidates)
    if (
        args.family in {"alignment", "fusion"}
        and args.model_name == "dmf"
        and any(
            candidate.get("model", {}).get("similarity_source") != "legacy_projected"
            for candidate in overrides
        )
    ):
        raise ContractError(
            "DMF {} requires model.similarity_source: legacy_projected in every "
            "candidate; use that same variant in the matched control".format(args.family)
        )
    candidates = [dict(candidate, diagnostic=setting) for candidate in overrides]
    catalog = load_dataset_catalog(
        args.dataset_config or PROJECT_ROOT / "configs/datasets/catalog.yaml",
        args.dataset_name,
        PROJECT_ROOT,
    )
    tuner = ValidationOnlyTuner(
        args.output_root,
        args.study_id,
        "{}-{}-{}".format(args.dataset_name, args.model_name, args.family),
        catalog[args.dataset_name]["fingerprint"],
        [args.seed],
    )
    factory = partial(
        _prepare_trainer,
        output_root=tuner.study_dir / "runs",
        cuda=args.cuda,
        dataset_config_path=args.dataset_config,
    )
    final_evaluator = (
        partial(_evaluate_cold_start, regime=args.regime) if args.family == "cold-start" else None
    )
    result = run_validation_search(tuner, candidates, factory, final_evaluator=final_evaluator)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
