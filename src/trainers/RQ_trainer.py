import os
num_threads = "16"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import argparse
from utils import helper

from models.pre_models.RQ import ResidualQuantizer
import numpy as np

parser = argparse.ArgumentParser(description="MMCTR-Trainer")
parser.add_argument("--dataset_name", type=str, help="specify dataset", default="antm2c")
args = parser.parse_args()

class Trainer(object):
    def __init__(self):
        self.dataset_name = str(args.dataset_name).lower()
        self.model_name = 'rq'
        self.model_config = helper.load_yaml(f'config/model.yaml')

        self.data_config = helper.load_yaml('config/seq_data.yaml')

        self.train_config = helper.load_yaml('config/train.yaml')
        self.dataloader = helper.getDataLoader(self.dataset_name, self.data_config[self.dataset_name],
                                               self.train_config["batch_size"])

    def run(self):
        epsilon = 1e-8
        # 优先使用数据集特定配置，缺省时回退到顶层配置（与 qarm 推荐模型保持一致）
        rq_config = self.model_config[self.model_name]
        rq_config = rq_config.get(self.dataset_name, rq_config)
        mm_modals = self.dataloader.get_multi_modal()
        for k in mm_modals.keys():
            mm_modal = mm_modals[k] / np.maximum(np.linalg.norm(mm_modals[k], axis=1, keepdims=True), epsilon)
            rq = ResidualQuantizer(rq_config,
                                   self.train_config,
                                   self.data_config[self.dataset_name])

            rq.fit(mm_modal, verbose=True)
            # 量化预模型统一保存到 checkpoint_dir，按 数据集_模态 命名，避免各数据集冲突
            save_dir = f"{self.train_config['checkpoint_dir']}/{args.dataset_name}_{k}_rq.npz"
            rq.save(save_dir)

if __name__ == "__main__":
    '''
    python trainers.py --model_name=dnn --dataset_name=Tiktok
    '''
    trainer = Trainer()
    trainer.run()
