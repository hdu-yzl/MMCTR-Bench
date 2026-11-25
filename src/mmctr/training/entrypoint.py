from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from mmctr.utils import helper
from mmctr.utils.run_context import create_run_context
from mmctr.config import (
    ConfigValidationError,
    load_local_paths,
    resolve_training_config,
    load_model_config,
    load_yaml_mapping,
    resolve_dataset_config,
    MODEL_TRAINING_KEYS,
)
from mmctr.data import ensure_canonical_loader, get_data_loader
from mmctr.core import RunResult
from mmctr.models.common.registry import create_model, create_model_from_artifacts, model_spec
from mmctr.training import (
    AlternatingPhase,
    CheckpointManager,
    TrainingEngine,
    build_optimizer,
    build_phased_adam,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Trainer(object):
    def __init__(
        self,
        dataset_name: str = "antm2c",
        model_name: str = "dnn",
        use_local_data: bool = False,
        cuda: Optional[int] = None,
        output_root: Optional[str] = None,
        seed: Optional[int] = None,
        save_checkpoints: bool = True,
        dataset_config_path: Optional[str] = None,
        model_overrides: Optional[Mapping] = None,
        training_overrides: Optional[Mapping] = None,
        model_config_dir: Optional[Path] = None,
        validate_model_data: bool = True,
    ):
        requested_model_name = str(model_name).lower()
        specification = model_spec(requested_model_name)
        self.model_name = specification.name
        self.dataset_name = str(dataset_name).lower()

        all_data_config = load_yaml_mapping(
            dataset_config_path or PROJECT_ROOT / "configs/datasets/catalog.yaml"
        )
        local_paths = None
        if use_local_data:
            local_paths = load_local_paths(PROJECT_ROOT / "configs/local/paths.yaml")
            if self.dataset_name not in local_paths.datasets:
                raise ConfigValidationError(
                    ["local path for dataset {!r} is missing".format(self.dataset_name)]
                )
        dataset_config = resolve_dataset_config(
            self.dataset_name,
            all_data_config[self.dataset_name],
            project_root=PROJECT_ROOT,
            local_paths=local_paths,
        )
        parameters = load_model_config(
            model_config_dir or PROJECT_ROOT / "configs/models",
            self.dataset_name,
            self.model_name,
            dataset_config["fingerprint"] if validate_model_data else None,
        )
        training_overrides = {**parameters["training"], **(training_overrides or {})}
        self.model_config = dict(parameters["model"])
        if self.model_name == "psrq":
            # A frozen pretrainer retains its own dropout configuration while the
            # downstream CTR network searches dropout independently.
            selected = self.model_config.get(self.dataset_name, self.model_config)
            quantizer_dropout = selected.get("quantizer_dropout", selected.get("dropout", 0.0))
            self.model_config.setdefault("quantizer_dropout", quantizer_dropout)
            if self.dataset_name in self.model_config:
                self.model_config[self.dataset_name] = dict(
                    selected, quantizer_dropout=quantizer_dropout
                )
        if model_overrides:
            if "model_name" in model_overrides:
                raise ValueError("model_name cannot be a tuning override")
            self.model_config.update(model_overrides)
            if specification.metadata.get("quantization_artifacts") and isinstance(
                self.model_config.get(self.dataset_name), Mapping
            ):
                # Quantized consumers select a complete dataset-scoped mapping.
                # Root tuning overrides must reach that effective configuration.
                overrides = dict(model_overrides)
                scoped = overrides.pop(self.dataset_name, {})
                self.model_config[self.dataset_name] = {
                    **self.model_config[self.dataset_name],
                    **overrides,
                    **scoped,
                }
        self.model_config["model_name"] = self.model_name
        from mmctr.models.common.registry import resolve_model_config

        self.model_config = resolve_model_config(self.model_config, dataset_config)
        self.data_config = {self.dataset_name: dataset_config}
        if set(training_overrides) - MODEL_TRAINING_KEYS:
            raise ValueError("unsupported training tuning override")
        self.train_config = resolve_training_config(training_overrides, PROJECT_ROOT).to_dict()
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("seed must be an integer in [0, 2**32)")
            self.train_config["seed"] = seed
        quantization_artifact_dir = self.train_config["quantization_artifact_dir"]
        if cuda is not None:
            self.train_config["cuda"] = cuda
        local_output_root = local_paths.output_root if local_paths else None
        resolved_output_root = output_root or local_output_root or self.train_config["output_root"]
        experiment_config = {
            "model": self.model_config,
            "data": self.data_config[self.dataset_name],
            "train": self.train_config,
        }
        self.run_context = create_run_context(
            output_root=resolved_output_root,
            experiment_name="training",
            dataset=self.dataset_name,
            model=self.model_name,
            resolved_config=experiment_config,
            repository_root=PROJECT_ROOT,
            metadata={
                "checkpoint_policy": "disk" if save_checkpoints else "memory_best_only",
                "seed": self.train_config.get("seed"),
                "parameter_selection": parameters["selection"],
                "requested_cuda": self.train_config.get("cuda"),
                "data_version": self.data_config[self.dataset_name].get("version", "unknown"),
                "data_fingerprint": self.data_config[self.dataset_name].get(
                    "fingerprint", "unknown"
                ),
            },
        )
        self.train_config["checkpoint_dir"] = str(self.run_context.checkpoints_dir)
        self.train_config["log_dir"] = str(self.run_context.root_dir)
        experiment_config["run"] = self.run_context.runtime_config()
        self.run_context.write_resolved_config(experiment_config)
        try:
            source_loader = get_data_loader(
                self.dataset_name,
                self.data_config[self.dataset_name],
                self.train_config["batch_size"],
            )
            self.logger = helper.get_logger(
                f"{self.model_name}.{self.run_context.run_id}",
                self.train_config["log_dir"],
                filename="run.log",
            )
            self.runtime_kind = "canonical"
            helper.setup_seed(self.train_config["seed"])
            self.dataloader = ensure_canonical_loader(self.dataset_name, source_loader)
            if specification.metadata.get("quantization_artifacts"):
                self.model = create_model_from_artifacts(
                    self.model_name,
                    self.model_config,
                    self.data_config[self.dataset_name],
                    quantization_artifact_dir,
                )
            else:
                self.model = create_model(
                    self.model_name,
                    self.model_config,
                    self.data_config[self.dataset_name],
                )
            device = helper.get_device(self.train_config["cuda"])
            alternating_phases: Tuple[AlternatingPhase, ...] = ()
            if specification.metadata.get("alternating_optimization") == "gmmf_gan":
                optimizer = build_phased_adam(
                    self.model.optimization_parameter_groups(),
                    {
                        "main": self.train_config["lr"],
                        "discriminator": self.model.discriminator_learning_rate,
                        "generator": self.model.generator_learning_rate,
                    },
                    self.train_config["l2"]
                    if training_overrides and "l2" in training_overrides
                    else self.model.weight_decay,
                )
                alternating_phases = (
                    AlternatingPhase(
                        "discriminator",
                        self.model.adversarial_start_epoch,
                        self.model.discriminator_loss,
                    ),
                    AlternatingPhase(
                        "generator",
                        self.model.adversarial_start_epoch,
                        self.model.generator_loss,
                    ),
                )
            else:
                optimizer = build_optimizer(
                    self.model,
                    self.train_config["optim"],
                    self.train_config["lr"],
                    self.train_config["l2"],
                )
            self.engine = TrainingEngine(
                self.model,
                optimizer,
                device,
                CheckpointManager(self.run_context.checkpoints_dir, persist=save_checkpoints),
                logger=self.logger,
                metric_writer=self.run_context.append_metrics,
                alternating_phases=alternating_phases,
            )
            self.run_context.update_metadata(
                {
                    "device": str(device),
                    "runtime_kind": self.runtime_kind,
                    "algorithm_version": getattr(self.model, "algorithm_version", "native"),
                }
            )

            self.logger.info(f"run_id={self.run_context.run_id}")
            self.logger.info(f"run_dir={self.run_context.root_dir}")
            self.logger.info(self.model_config)
        except BaseException as error:
            self.run_context.finalize("failed", error=repr(error))
            raise

    def evaluate_final(self, epoch: int) -> RunResult:
        """Evaluate selected weights once; callers must freeze HPO selection first."""
        if getattr(self, "_test_evaluated", False):
            raise RuntimeError("final test has already been evaluated for this run")
        self._test_evaluated = True
        evaluation = self.engine.evaluate(self.dataloader, "test", epoch)
        if evaluation.metrics is None:
            raise RuntimeError("test metrics were not produced")
        metrics = {"test_auc": evaluation.metrics.auc, "test_loss": evaluation.metrics.log_loss}
        return RunResult(
            self.run_context.run_id,
            "completed",
            metrics,
            artifact_dir=self.run_context.root_dir,
        )

    def run(self, evaluate_test: bool = True) -> RunResult:
        """Fit on train/validation, restore the best checkpoint, then evaluate test once.

        Test metrics are produced only after validation-based model selection has
        completed. Any failure finalizes the run as failed before it is re-raised.
        """

        try:
            fit_result = self.engine.fit(
                self.dataloader,
                self.train_config["max_epochs"],
                self.train_config["early_stop_patience"],
                self.run_context.run_id,
                self.run_context.root_dir,
            )
            summary = dict(fit_result.metrics)
            if evaluate_test:
                final = self.evaluate_final(int(fit_result.metrics["best_epoch"]))
                summary.update(final.metrics)
            else:
                validation = self.engine.evaluate(
                    self.dataloader, "val", int(fit_result.metrics["best_epoch"])
                )
                if validation.metrics is None:
                    raise RuntimeError("validation metrics were not produced")
                summary.update(
                    val_auc=validation.metrics.auc, val_log_loss=validation.metrics.log_loss
                )
            self.run_context.finalize("completed", summary=summary)
            return RunResult(
                self.run_context.run_id,
                "completed",
                summary,
                artifact_dir=self.run_context.root_dir,
            )
        except BaseException as error:
            self.run_context.finalize("failed", error=repr(error))
            raise


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="MMCTR canonical training entry point")
    parser.add_argument("--dataset_name", default="antm2c")
    parser.add_argument("--model_name", default="dnn")
    parser.add_argument("--use_local_data", action="store_true")
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    trainer = Trainer(
        dataset_name=arguments.dataset_name,
        model_name=arguments.model_name,
        use_local_data=arguments.use_local_data,
        cuda=arguments.cuda,
        output_root=arguments.output_root,
        seed=arguments.seed,
        save_checkpoints=not arguments.no_checkpoints,
    )
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
