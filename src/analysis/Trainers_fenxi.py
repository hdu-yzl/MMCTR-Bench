import os
num_threads = "24"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import argparse
from utils import helper

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

        self.model_config = helper.load_yaml(f'config/best_param.yaml')[self.model_name][self.dataset_name]
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
        self.train_config['lr'] = self.model_config['lr']
        self.train_config['l2'] = self.model_config['l2']
        if args.cuda is not None:
            self.train_config['cuda'] = args.cuda
        if args.data_dir is not None:
            self.data_config[self.dataset_name]['data_dir'] = args.data_dir
        self.dataloader = helper.getDataLoader(self.dataset_name, self.data_config[self.dataset_name],
                                               self.train_config["batch_size"])
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
