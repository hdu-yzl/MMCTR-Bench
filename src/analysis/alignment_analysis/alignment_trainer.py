"""
对齐分析训练脚本
为各个模型添加模态对齐损失，测试不同对齐方法和lambda_weight的效果
"""
import os
import sys
import argparse
import torch

# 设置线程数
num_threads = "12"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

# 将 src 加入路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import helper
from models.layers import alignment


# 所有支持的对齐方法
ALIGNMENT_METHODS = {
    'kl': alignment.KLDivergenceAlignment,
    'contrastive': alignment.ContrastiveAlignment,
    'cosine': alignment.CosineAlignment,
    'mmd': alignment.MMDAlignment,
    'adversarial': alignment.AdversarialAlignment,
}


class AlignmentModelWrapper(torch.nn.Module):
    """
    对齐损失包装器，为任意模型添加对齐损失
    通过继承nn.Module并重写compute_loss方法来注入对齐损失
    """
    def __init__(self, model, alignment_method, lambda_weight, mm_features, projection_dim):
        super().__init__()
        self.model = model
        self.lambda_weight = lambda_weight
        self.alignment_method = alignment_method
        
        # 创建对齐模块
        if alignment_method == 'none' or lambda_weight == 0:
            self.alignment_module = None
        elif alignment_method == 'kl':
            self.alignment_module = alignment.KLDivergenceAlignment(
                mm_features=mm_features,
                lambda_weight=lambda_weight
            )
        elif alignment_method == 'contrastive':
            self.alignment_module = alignment.ContrastiveAlignment(
                mm_features=mm_features,
                lambda_weight=lambda_weight
            )
        elif alignment_method == 'cosine':
            self.alignment_module = alignment.CosineAlignment(
                mm_features=mm_features,
                lambda_weight=lambda_weight
            )
        elif alignment_method == 'mmd':
            self.alignment_module = alignment.MMDAlignment(
                mm_features=mm_features,
                lambda_weight=lambda_weight
            )
        elif alignment_method == 'adversarial':
            self.alignment_module = alignment.AdversarialAlignment(
                input_dim=projection_dim,
                mm_features=mm_features,
                lambda_weight=lambda_weight
            )
        else:
            raise ValueError(f"Unknown alignment method: {alignment_method}")
        
        if self.alignment_module is not None:
            self.alignment_module = self.alignment_module.to(model.device)
        
        # 保存原始的_predict_batch方法
        self.original_predict_batch = model._predict_batch
        # 替换为我们的版本
        model._predict_batch = self._predict_batch_with_alignment
    
    def _predict_batch_with_alignment(self, batch):
        """带对齐损失的batch预测"""
        # 调用原始的_predict_batch
        out, label = self.original_predict_batch(batch)
        
        # 如果没有对齐模块，直接返回
        if self.alignment_module is None:
            return out, label
        
        # 获取投影后的特征用于对齐
        if hasattr(self.model, 'get_alignment_feats'):
            # 离散编码类模型（QARM / MCCA）：获取码本中最近的向量 →
            # 多层码本嵌入查表 → 投影 → 序列 mean pooling，得到每模态多模态表征
            # 后再与 ID 模态对齐（mm_projector 期望码本嵌入维度，不能直接喂原始特征）
            _, feats, feats_seq, _ = batch
            feats = {k: v.to(self.model.device) for k, v in feats.items()}
            feats_seq = {k: v.to(self.model.device) for k, v in feats_seq.items()}
            feats_p = self.model.get_alignment_feats(feats, feats_seq)
        else:
            is_seq_model = hasattr(self.model, 'seq_modeling') and self.model.seq_modeling

            if is_seq_model:
                user_feats, feats, feats_seq, _ = batch
                feats = {k: v.to(self.model.device) for k, v in feats.items()}
                id_emb = self.model.embedding(feats['id'])
                # 仅去掉物品维 (B, 1, D) -> (B, D)，避免 B=1 时 batch 维被压缩
                if id_emb.dim() == 3 and id_emb.size(1) == 1:
                    id_emb = id_emb.squeeze(1)
                feats['id'] = id_emb
            else:
                feats, feats_seq, _ = batch
                feats = {k: v.to(self.model.device) for k, v in feats.items()}
                feats['id'] = self.model.embedding(feats['id']).view(-1, self.model.id_dim)

            feats_p = {k: self.model.mm_projector[k](feats[k]) for k in self.model.mm_features}
        
        # 计算对齐损失
        if self.alignment_method == 'adversarial':
            align_loss = self.alignment_module.get_generator_loss(feats_p)
        else:
            align_loss = self.alignment_module(feats_p)
        
        # 将对齐损失添加到au_loss中
        if 'au_loss' in out:
            out['au_loss'] = out['au_loss'] + align_loss
        else:
            out['au_loss'] = align_loss
        
        return out, label
    
    def __getattr__(self, name):
        """代理所有其他属性到原始模型"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class Trainer:
    def __init__(self, args):
        self.model_name = str(args.model_name).lower()
        self.dataset_name = str(args.dataset_name).lower()
        self.alignment_method = str(args.alignment_method).lower()
        self.lambda_weight = float(args.lambda_weight)
        
        # 加载配置
        self.model_config = helper.load_yaml('config/best_param.yaml')[self.model_name][self.dataset_name]
        
        # 修改模型名称以区分不同的对齐实验
        self.model_config['model_name'] = (
            f"{self.model_name}_align_{self.alignment_method}_"
            f"lambda{self.lambda_weight:.2f}_{self.dataset_name}"
        )
        
        # 加载数据配置
        if self.model_config.get("seq_modeling", False):
            if args.use_local_data:
                self.data_config = helper.load_yaml('config/local_seq_data.yaml')
            else:
                self.data_config = helper.load_yaml('config/seq_data.yaml')
        else:
            if args.use_local_data:
                self.data_config = helper.load_yaml('config/local_data.yaml')
            else:
                self.data_config = helper.load_yaml('config/data.yaml')
        
        # 加载训练配置
        self.train_config = helper.load_yaml('config/train.yaml')
        self.train_config['lr'] = self.model_config.get('lr', self.train_config.get('lr', 1e-3))
        self.train_config['l2'] = self.model_config.get('l2', self.train_config.get('l2', 1e-6))
        
        if args.cuda is not None:
            self.train_config['cuda'] = args.cuda
        
        # 创建数据加载器
        self.dataloader = helper.getDataLoader(
            self.dataset_name,
            self.data_config[self.dataset_name],
            self.train_config["batch_size"]
        )
        
        # 创建日志
        log_name = f"{self.model_config['model_name']}_alignment_{self.lambda_weight}"
        self.logger = helper.get_logger(log_name, self.train_config['log_dir'])
        
        # 创建模型
        self.model = helper.getModel(
            self.model_name,
            self.model_config,
            self.train_config,
            self.data_config[self.dataset_name],
            self.logger
        )
        
        self.logger.info(f"模型配置: {self.model_config}")
        self.logger.info(f"对齐方法: {self.alignment_method}, lambda_weight: {self.lambda_weight}")
    
    def run(self):
        # 创建对齐包装器
        mm_features = self.data_config[self.dataset_name].get('use_mm_features', ['id', 'text', 'image'])
        projection_dim = self.model_config.get('projection_dim', 128)
        
        wrapper = AlignmentModelWrapper(
            self.model,
            self.alignment_method,
            self.lambda_weight,
            mm_features,
            projection_dim
        )
        
        # 训练（使用模型自己的fit方法）
        best_val = self.model.fit(self.dataloader)
        
        # 测试
        test_auc, test_loss = self.model.evalate(self.dataloader, 'test')
        self.logger.info(
            f'最终结果 - Val AUC: {best_val:.6f}, '
            f'Test AUC: {test_auc:.6f}, Test Loss: {test_loss:.6f}'
        )
        
        return {
            'model': self.model_name,
            'dataset': self.dataset_name,
            'alignment': self.alignment_method,
            'lambda': self.lambda_weight,
            'val_auc': best_val,
            'test_auc': test_auc,
            'test_loss': test_loss
        }


def main():
    parser = argparse.ArgumentParser(description="对齐分析训练器")
    parser.add_argument("--model_name", type=str, required=True,
                       help="模型名称")
    parser.add_argument("--dataset_name", type=str, default="antm2c",
                       help="数据集名称")
    parser.add_argument("--alignment_method", type=str, default="none",
                       choices=['none', 'kl', 'contrastive', 'cosine', 'mmd', 'adversarial'],
                       help="对齐方法")
    parser.add_argument("--lambda_weight", type=float, default=0.0,
                       help="对齐损失权重")
    parser.add_argument("--use_local_data", type=int, default=0,
                       help="是否使用本地数据")
    parser.add_argument("--cuda", type=int, default=None,
                       help="GPU设备ID")
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    result = trainer.run()
    
    print(f"\n{'='*60}")
    print(f"实验完成:")
    print(f"  模型: {result['model']}")
    print(f"  数据集: {result['dataset']}")
    print(f"  对齐方法: {result['alignment']}")
    print(f"  Lambda: {result['lambda']}")
    print(f"  Val AUC: {result['val_auc']:.6f}")
    print(f"  Test AUC: {result['test_auc']:.6f}")
    print(f"  Test Loss: {result['test_loss']:.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()