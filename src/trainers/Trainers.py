from pathlib import Path
from typing import Optional, Sequence

from mmctr.utils import helper
from mmctr.utils.run_context import create_run_context
from mmctr.config import (
    ConfigValidationError,
    load_local_paths,
    load_training_config,
    resolve_dataset_config,
)
from mmctr.data import HistoryMode, adapt_legacy_loader
from mmctr.models.registry import create_model, create_model_from_artifacts, model_spec
from mmctr.training import CheckpointManager, TrainingEngine, build_optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Trainer(object):
    def __init__(
        self,
        dataset_name: str = "antm2c",
        model_name: str = "dnn",
        use_local_data: bool = False,
        cuda: Optional[int] = None,
        output_root: Optional[str] = None,
    ):
        requested_model_name = str(model_name).lower()
        specification = model_spec(requested_model_name)
        self.model_name = specification.name
        self.dataset_name = str(dataset_name).lower()

        model_catalog = helper.load_yaml(PROJECT_ROOT / 'config/model.yaml')
        config_name = requested_model_name
        if config_name not in model_catalog and self.model_name == 'dnn_mm_seq':
            config_name = 'dnn_seq'
        self.model_config = model_catalog[config_name]
        self.model_config['model_name'] = self.model_name
        data_config_name = 'seq_data.yaml' if self.model_config["seq_modeling"] else 'data.yaml'
        all_data_config = helper.load_yaml(PROJECT_ROOT / 'config' / data_config_name)
        local_paths = None
        if use_local_data:
            local_paths = load_local_paths(PROJECT_ROOT / 'configs/local/paths.yaml')
            if self.dataset_name not in local_paths.datasets:
                raise ConfigValidationError([
                    "local path for dataset {!r} is missing".format(self.dataset_name)
                ])
        dataset_config = resolve_dataset_config(
            self.dataset_name,
            all_data_config[self.dataset_name],
            project_root=PROJECT_ROOT,
            local_paths=local_paths,
        )
        self.data_config = {self.dataset_name: dataset_config}
        self.train_config = load_training_config(PROJECT_ROOT / 'config/train.yaml').to_dict()
        quantization_artifact_dir = self.train_config['quantization_artifact_dir']
        if cuda is not None:
            self.train_config['cuda'] = cuda
        local_output_root = local_paths.output_root if local_paths else None
        resolved_output_root = output_root or local_output_root or self.train_config['output_root']
        experiment_config = {
            'model': self.model_config,
            'data': self.data_config[self.dataset_name],
            'train': self.train_config,
        }
        self.run_context = create_run_context(
            output_root=resolved_output_root,
            experiment_name='training',
            dataset=self.dataset_name,
            model=self.model_name,
            resolved_config=experiment_config,
            repository_root=PROJECT_ROOT,
            metadata={
                'seed': self.train_config.get('seed'),
                'requested_cuda': self.train_config.get('cuda'),
                'data_version': self.data_config[self.dataset_name].get('version', 'unknown'),
                'data_fingerprint': self.data_config[self.dataset_name].get(
                    'fingerprint', 'unknown'
                ),
            },
        )
        self.train_config['checkpoint_dir'] = str(self.run_context.checkpoints_dir)
        self.train_config['log_dir'] = str(self.run_context.root_dir)
        experiment_config['run'] = self.run_context.runtime_config()
        self.run_context.write_resolved_config(experiment_config)
        try:
            legacy_data_loader = helper.getDataLoader(
                self.dataset_name,
                self.data_config[self.dataset_name],
                self.train_config["batch_size"],
            )
            self.logger = helper.get_logger(
                f"{self.model_name}.{self.run_context.run_id}",
                self.train_config['log_dir'],
                filename='run.log',
            )
            if specification.module.startswith('mmctr.models.'):
                self.runtime_kind = 'canonical'
                helper.setup_seed(self.train_config['seed'])
                history_mode = (
                    HistoryMode.POOLED_COMPAT
                    if specification.metadata.get('history') == 'pooled'
                    else HistoryMode.SEQUENCE_TOKENS
                )
                self.dataloader = adapt_legacy_loader(
                    self.dataset_name,
                    legacy_data_loader,
                    self.data_config[self.dataset_name],
                    history_mode=history_mode,
                )
                if specification.metadata.get('quantization_artifacts'):
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
                device = helper.getDevice(self.train_config['cuda'])
                optimizer = build_optimizer(
                    self.model,
                    self.train_config['optim'],
                    self.train_config['lr'],
                    self.train_config['l2'],
                )
                self.engine = TrainingEngine(
                    self.model,
                    optimizer,
                    device,
                    CheckpointManager(self.run_context.checkpoints_dir),
                    logger=self.logger,
                    metric_writer=self.run_context.append_metrics,
                )
            else:
                self.runtime_kind = 'legacy'
                self.dataloader = legacy_data_loader
                self.model = helper.getModel(
                    self.model_name,
                    self.model_config,
                    self.train_config,
                    self.data_config[self.dataset_name],
                    self.logger,
                )
                device = self.model.device
            self.run_context.update_metadata(
                {'device': str(device), 'runtime_kind': self.runtime_kind}
            )

            self.logger.info(f"run_id={self.run_context.run_id}")
            self.logger.info(f"run_dir={self.run_context.root_dir}")
            self.logger.info(self.model_config)
        except BaseException as error:
            self.run_context.finalize('failed', error=repr(error))
            raise

    def run(self):
        try:
            if self.runtime_kind == 'canonical':
                fit_result = self.engine.fit(
                    self.dataloader,
                    self.train_config['max_epochs'],
                    self.train_config['early_stop_patience'],
                    self.run_context.run_id,
                    self.run_context.root_dir,
                )
                test_result = self.engine.evaluate(
                    self.dataloader,
                    'test',
                    int(fit_result.metrics['best_epoch']),
                )
                if test_result.metrics is None:
                    raise RuntimeError('test metrics were not produced')
                test_auc = test_result.metrics.auc
                test_loss = test_result.metrics.log_loss
                summary = dict(fit_result.metrics)
                summary.update({'test_auc': test_auc, 'test_loss': test_loss})
            else:
                best_val = self.model.fit(self.dataloader)
                test_auc, test_loss = self.model.evalate(self.dataloader, 'test')
                summary = {
                    'best_val_auc': float(best_val),
                    'test_auc': float(test_auc),
                    'test_loss': float(test_loss),
                }
            self.logger.info(f'test_auc={test_auc:.6f}, test_loss={test_loss:.6f}')
            self.run_context.finalize('completed', summary=summary)
        except BaseException as error:
            self.run_context.finalize('failed', error=repr(error))
            raise


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="MMCTR legacy training adapter")
    parser.add_argument("--dataset_name", default="antm2c")
    parser.add_argument("--model_name", default="dnn")
    parser.add_argument("--use_local_data", action="store_true")
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--output_root", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    trainer = Trainer(
        dataset_name=arguments.dataset_name,
        model_name=arguments.model_name,
        use_local_data=arguments.use_local_data,
        cuda=arguments.cuda,
        output_root=arguments.output_root,
    )
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
