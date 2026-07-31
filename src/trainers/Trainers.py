import os
num_threads = "24"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import argparse
from pathlib import Path

from mmctr.utils import helper
from mmctr.utils.run_context import create_run_context
from mmctr.config import (
    ConfigValidationError,
    load_local_paths,
    load_training_config,
    resolve_dataset_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="MMCTR-Trainer")
parser.add_argument("--dataset_name", type=str, help="specify dataset", default="antm2c")
parser.add_argument("--model_name", type=str, help="specify model", default="dnn")
parser.add_argument("--use_local_data", type=int, help="data", default=0)
parser.add_argument("--cuda", type=int, help="override cuda id in config/train.yaml", default=None)
parser.add_argument("--output_root", type=str, help="isolated run output root", default=None)
args = parser.parse_args()


class Trainer(object):
    def __init__(self):
        self.model_name = str(args.model_name).lower()
        self.dataset_name = str(args.dataset_name).lower()

        self.model_config = helper.load_yaml(PROJECT_ROOT / 'config/model.yaml')[self.model_name]
        self.model_config['model_name'] = self.model_name
        data_config_name = 'seq_data.yaml' if self.model_config["seq_modeling"] else 'data.yaml'
        all_data_config = helper.load_yaml(PROJECT_ROOT / 'config' / data_config_name)
        local_paths = None
        if args.use_local_data:
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
        if args.cuda is not None:
            self.train_config['cuda'] = args.cuda
        local_output_root = local_paths.output_root if local_paths else None
        output_root = args.output_root or local_output_root or self.train_config['output_root']
        experiment_config = {
            'model': self.model_config,
            'data': self.data_config[self.dataset_name],
            'train': self.train_config,
        }
        self.run_context = create_run_context(
            output_root=output_root,
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
            self.dataloader = helper.getDataLoader(
                self.dataset_name,
                self.data_config[self.dataset_name],
                self.train_config["batch_size"],
            )
            self.logger = helper.get_logger(
                f"{self.model_name}.{self.run_context.run_id}",
                self.train_config['log_dir'],
                filename='run.log',
            )
            self.model = helper.getModel(
                self.model_name,
                self.model_config,
                self.train_config,
                self.data_config[self.dataset_name],
                self.logger,
            )
            self.run_context.update_metadata({'device': str(self.model.device)})

            self.logger.info(f"run_id={self.run_context.run_id}")
            self.logger.info(f"run_dir={self.run_context.root_dir}")
            self.logger.info(self.model_config)
        except BaseException as error:
            self.run_context.finalize('failed', error=repr(error))
            raise

    def run(self):
        try:
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


if __name__ == "__main__":
    '''
    python trainers.py --model_name=dnn --dataset_name=Tiktok
    '''
    trainer = Trainer()
    trainer.run()
