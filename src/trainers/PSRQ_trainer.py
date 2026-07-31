import os
num_threads = "16"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import argparse
from utils import helper

from models.pre_models.PSRQ import PSRQ_Premodel
import numpy as np

parser = argparse.ArgumentParser(description="MMCTR-Trainer")
parser.add_argument("--dataset_name", type=str, help="specify dataset", default="antm2c")
args = parser.parse_args()

class Trainer(object):
    def __init__(self):
        self.dataset_name = str(args.dataset_name).lower()
        self.model_name = 'psrq'
        self.model_config = helper.load_yaml(f'config/model.yaml')

        self.data_config = helper.load_yaml('config/seq_data.yaml')

        self.train_config = helper.load_yaml('config/train.yaml')
        self.dataloader = helper.getDataLoader(self.dataset_name, self.data_config[self.dataset_name],
                                               self.train_config["batch_size"])

        # 优先使用数据集特定配置，缺省时回退到顶层配置（与 mcca 推荐模型保持一致）
        psrq_config = self.model_config[self.model_name]
        psrq_config = psrq_config.get(self.dataset_name, psrq_config)
        self.model = PSRQ_Premodel(psrq_config, self.train_config, self.data_config[self.dataset_name])

    def run(self):
        self.model.fit(self.dataloader)
        self.model.save()

if __name__ == "__main__":
    '''
    python trainers.py --model_name=dnn --dataset_name=Tiktok
    '''
    trainer = Trainer()
    trainer.run()
