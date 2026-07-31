import os
num_threads = "24"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import argparse
from utils import helper
from utils.run_context import create_run_context

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

        self.model_config = helper.load_yaml(f'config/model.yaml')[self.model_name]
        self.model_config['model_name'] = self.model_name
        if self.model_config["seq_modeling"]:
            if args.use_local_data:
                self.data_config = helper.load_yaml('config/local_seq_data.yaml')
            else:
                self.data_config = helper.load_yaml('config/seq_data.yaml')
        else:
            if args.use_local_data:
                self.data_config = helper.load_yaml('config/local_data.yaml')
            else:
                self.data_config = helper.load_yaml('config/data.yaml')
        self.train_config = helper.load_yaml('config/train.yaml')
        if args.cuda is not None:
            self.train_config['cuda'] = args.cuda
        output_root = args.output_root or self.train_config.get('output_root', 'outputs')
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
            repository_root='.',
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
