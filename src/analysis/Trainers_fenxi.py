import os
num_threads = "24"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import argparse
from pathlib import Path

from mmctr.config import load_dataset_catalog
from mmctr.utils import helper


PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="MMCTR-Trainer")
parser.add_argument("--dataset_name", type=str, help="specify dataset", default="antm2c")
parser.add_argument("--model_name", type=str, help="specify model", default="dnn")
parser.add_argument("--use_local_data", type=int, help="data", default=0)
parser.add_argument("--data_dir", type=str, help="override data_dir in config", default=None)
parser.add_argument("--cuda", type=int, help="override cuda device id", default=None)
args = parser.parse_args()


class Trainer(object):
    def __init__(self):
        self.model_name = str(args.model_name).lower()
        self.dataset_name = str(args.dataset_name).lower()

        self.model_config = helper.load_yaml(
            PROJECT_ROOT / 'config/best_param.yaml'
        )[self.model_name][self.dataset_name]
        self.model_config['model_name'] = self.model_name
        data_config_name = (
            'seq_data.yaml' if self.model_config["seq_modeling"] else 'data.yaml'
        )
        local_environment = None
        if args.data_dir is not None:
            environment_key = "MMCTR_{}_DATA_DIR".format(self.dataset_name.upper())
            local_environment = {environment_key: args.data_dir}
        self.data_config = load_dataset_catalog(
            PROJECT_ROOT / 'config' / data_config_name,
            self.dataset_name,
            project_root=PROJECT_ROOT,
            use_local_data=bool(args.use_local_data or args.data_dir),
            environ=local_environment,
        )
        self.train_config = helper.load_yaml(PROJECT_ROOT / 'config/train.yaml')
        self.train_config['lr'] = self.model_config['lr']
        self.train_config['l2'] = self.model_config['l2']
        if args.cuda is not None:
            self.train_config['cuda'] = args.cuda
        self.dataloader = helper.getDataLoader(
            self.dataset_name,
            self.data_config[self.dataset_name],
            self.train_config["batch_size"],
        )
        self.logger = helper.get_logger(self.model_name, self.train_config['log_dir'])
        self.model = helper.getModel(self.model_name, self.model_config, self.train_config,
                                     self.data_config[self.dataset_name], self.logger)

        self.logger.info(self.model_config)

    def run(self):
        best_val = self.model.fit(self.dataloader)
        test_auc, test_loss = self.model.evalate(self.dataloader, 'test')
        self.logger.info(f'test_auc={test_auc:.6f}, test_loss={test_loss:.6f}')


if __name__ == "__main__":
    '''
    python trainers.py --model_name=dnn --dataset_name=Tiktok
    '''
    trainer = Trainer()
    trainer.run()
