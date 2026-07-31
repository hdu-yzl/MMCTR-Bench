"""
分析各种模态融合方法在 DNN backbone 下的 CTR 预估效果

用法 (从 Benchmark 根目录运行):
    python src/analysis/fusion_analysis.py --dataset_name tiktok
    python src/analysis/fusion_analysis.py --dataset_name antm2c --use_local_data 1
    python src/analysis/fusion_analysis.py --dataset_name tiktok --fusions cat add mean maf lmf
    python src/analysis/fusion_analysis.py --dataset_name tiktok --cuda 0 --max_epochs 3

支持的融合方法: cat, maf, lmf, mtfn, src, fq-former
(dta 的 forward 接口与 DNN_mm 不兼容，故不参与对比)
"""
import os
import sys

num_threads = "24"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

# 将 src 和根目录加入搜索路径，使 import 正常工作
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
for p in (SRC_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import csv
import time
from datetime import datetime

import numpy as np
import torch

from mmctr.utils import helper
from models.ctr_models.dnn import DNN_mm
from models.layers.modal_fusion import _FUSION_MAP, FQFormer


class FQFormerFusion(torch.nn.Module):
    """将 FQFormer 包装为与 DNN_mm 兼容的 dict->Tensor 融合层

    将各模态特征堆叠为序列 (B, M, D)，经 FQFormer 自注意力后
    取 learnable query 位置的输出展平为 (B, query_num * D)
    """

    def __init__(self, dim, mm_features=('id', 'text', 'image'),
                 query_num=3, layer_num=2, num_heads=8,
                 attn_drop=0., hidden_drop=0.):
        super().__init__()
        self.mm_features = list(mm_features)
        self.fqformer = FQFormer(
            dim=dim, query_num=query_num, layer_num=layer_num,
            num_heads=num_heads, attn_drop=attn_drop, hidden_drop=hidden_drop,
        )

    def getDim(self):
        return self.fqformer.getDim()

    def forward(self, mm_feats: dict):
        # 将各模态特征 (B, D) 堆叠为 (B, M, D)
        x = torch.stack([mm_feats[m] for m in self.mm_features], dim=1)
        out = self.fqformer(x)  # (B, query_num + M, D)
        query_num = self.fqformer.query_num
        queries_out = out[:, :query_num, :]  # (B, query_num, D)
        return queries_out.reshape(queries_out.size(0), -1)  # (B, query_num * D)


# 注册到全局映射表，使 DNN_mm 中 get_fusion_layer 可直接使用
_FUSION_MAP['fq-former-fusion'] = FQFormerFusion

# 与 DNN_mm 的 forward(dict -> Tensor) 接口兼容的融合方法
COMPATIBLE_FUSIONS = ['cat', 'maf', 'lmf', 'mtfn', 'src', 'fq-former-fusion']


def parse_args():
    parser = argparse.ArgumentParser(description="模态融合方法对比分析 (DNN backbone)")
    parser.add_argument("--dataset_name", type=str, default="tiktok",
                        choices=["tiktok", "antm2c", "microlens"])
    parser.add_argument("--use_local_data", type=int, default=0,
                        help="是否使用本地离线数据 (0/1)")
    parser.add_argument("--fusions", nargs="+", default=COMPATIBLE_FUSIONS,
                        help="要测试的融合方法列表，可选: " + str(COMPATIBLE_FUSIONS))
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="覆盖 train.yaml 中的 max_epochs")
    parser.add_argument("--cuda", type=int, default=None,
                        help="覆盖 train.yaml 中的 cuda 设备编号, -1 表示 CPU")
    parser.add_argument("--seed", type=int, default=2025,
                        help="随机种子，保证公平对比")
    return parser.parse_args()


def build_model_config(fusion_method):
    """构建 DNN_mm 所需的 model_config，仅变更融合方法"""
    return {
        'model_name': f'dnn_mm_{fusion_method}',
        'latent_dim': 128,
        'projection_dim': 128,
        'mlp_dims': [1024, 512, 256],
        'dropout': 0.0,
        'batch_norm': True,
        'seq_pooling_method': 'mean',
        'modal_fusion_method': fusion_method,
        'seq_modeling': False,
    }


def run_single_fusion(fusion_method, dataloader, data_config, train_config, logger, seed):
    """训练并测试单个融合方法，返回结果字典"""
    # 每轮重置随机种子，确保公平对比
    helper.setup_seed(seed)

    model_config = build_model_config(fusion_method)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"开始测试融合方法: {fusion_method.upper()}")
    logger.info(f"{'=' * 60}")

    model = DNN_mm(model_config, train_config, data_config, logger)

    start_time = time.time()
    best_val_auc = model.fit(dataloader)
    train_time = time.time() - start_time

    test_auc, test_loss = model.evalate(dataloader, 'test')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion_params = (sum(p.numel() for p in model.modal_fusion.parameters())
                     + sum(p.numel() for p in model.seq_modal_fusion.parameters()))

    result = {
        'fusion': fusion_method,
        'val_auc': best_val_auc,
        'test_auc': test_auc,
        'test_loss': test_loss,
        'train_time_s': round(train_time, 1),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'fusion_params': fusion_params,
    }

    logger.info(
        f"[{fusion_method}] val_auc={best_val_auc:.6f}  test_auc={test_auc:.6f}  "
        f"test_loss={test_loss:.6f}  time={train_time:.1f}s  "
        f"params={trainable_params:,}  fusion_params={fusion_params:,}"
    )

    del model
    torch.cuda.empty_cache()

    return result


def print_comparison_table(results, logger):
    """按 Test AUC 降序打印对比表格"""
    sorted_results = sorted(results, key=lambda x: x['test_auc'], reverse=True)

    logger.info("\n" + "=" * 100)
    logger.info("模态融合方法对比结果 (按 Test AUC 降序)")
    logger.info("=" * 100)

    header = (f"{'排名':<4} {'融合方法':<10} {'Val AUC':>10} {'Test AUC':>10} "
              f"{'Test Loss':>10} {'时间(s)':>8} {'总参数':>14} {'融合层参数':>14}")
    logger.info(header)
    logger.info("-" * 100)

    for rank, r in enumerate(sorted_results, 1):
        line = (f"{rank:<4} {r['fusion']:<10} {r['val_auc']:>10.6f} {r['test_auc']:>10.6f} "
                f"{r['test_loss']:>10.6f} {r['train_time_s']:>8.1f} "
                f"{r['trainable_params']:>14,} {r['fusion_params']:>14,}")
        logger.info(line)

    logger.info("-" * 100)
    best, worst = sorted_results[0], sorted_results[-1]
    logger.info(f"最佳: {best['fusion']}  (Test AUC = {best['test_auc']:.6f})")
    logger.info(f"最差: {worst['fusion']}  (Test AUC = {worst['test_auc']:.6f})")
    logger.info(f"AUC 差距: {best['test_auc'] - worst['test_auc']:.6f}")


def save_results_csv(results, save_path):
    """保存结果到 CSV 文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()
    dataset_name = args.dataset_name.lower()

    # ---- 加载配置 ----
    data_config_all = helper.load_yaml('config/data.yaml')
    train_config = helper.load_yaml('config/train.yaml')
    data_config = data_config_all[dataset_name]
    data_config['use_mm_features'] = ['id', 'text', 'image']  # 固定使用这三种模态特征进行融合对比
    data_config['use_mm_seq_features'] = ['id', 'text', 'image']  # 与 mm_features 保持一致，避免 cat 融合时维度不匹配

    if args.use_local_data:
        data_config['using_local_data'] = True
    if args.max_epochs is not None:
        train_config['max_epochs'] = args.max_epochs
    if args.cuda is not None:
        train_config['cuda'] = args.cuda

    # ---- 日志 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"fusion_analysis_{dataset_name}_{timestamp}"
    logger = helper.get_logger(log_name, train_config['log_dir'])

    logger.info(f"数据集: {dataset_name}")
    logger.info(f"待测试融合方法: {args.fusions}")
    logger.info(f"训练配置: {train_config}")
    logger.info(f"数据配置: {data_config}")

    # ---- 创建 dataloader（复用同一份，避免重复加载大型 npy 文件）----
    dataloader = helper.getDataLoader(dataset_name, data_config, train_config["batch_size"])

    # ---- 逐个测试融合方法 ----
    all_results = []
    for fusion in args.fusions:
        if fusion not in _FUSION_MAP:
            logger.warning(f"跳过未知融合方法: {fusion}")
            continue
        if fusion not in COMPATIBLE_FUSIONS:
            logger.warning(f"跳过不兼容的融合方法: {fusion} (forward 接口不匹配 DNN_mm)")
            continue

        try:
            result = run_single_fusion(
                fusion_method=fusion,
                dataloader=dataloader,
                data_config=data_config,
                train_config=train_config,
                logger=logger,
                seed=args.seed,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"融合方法 {fusion} 运行失败: {e}", exc_info=True)

    # ---- 汇总输出 ----
    if not all_results:
        logger.error("所有融合方法均运行失败，无结果可展示")
        return

    print_comparison_table(all_results, logger)

    # 保存 CSV
    csv_path = os.path.join(train_config['log_dir'],
                            f"fusion_analysis_{dataset_name}_{timestamp}.csv")
    save_results_csv(all_results, csv_path)
    logger.info(f"CSV 结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()
