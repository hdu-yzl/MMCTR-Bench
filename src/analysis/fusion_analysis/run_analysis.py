"""
模态融合方法 × 模型架构 对比分析

对每个 MM-CTR 模型分别测试不同融合方法，分析融合策略对不同模型架构的影响。

用法 (从 Benchmark 根目录运行):
    python src/analysis/fusion_analysis/run_analysis.py --dataset_name tiktok
    python src/analysis/fusion_analysis/run_analysis.py --dataset_name antm2c --models NAML MARN
    python src/analysis/fusion_analysis/run_analysis.py --dataset_name tiktok --fusions cat maf lmf
    python src/analysis/fusion_analysis/run_analysis.py --dataset_name tiktok --cuda 0 --max_epochs 3

支持的融合方法:
  本地融合: maf, cat, lmf, src, mtfn, fq-former, simcen
  序列感知融合: dta, gmmf, dmf
"""
import os
import sys

num_threads = "24"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
for p in (SRC_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import copy
import csv
import time
from datetime import datetime

import torch

from mmctr.utils import helper

# ── 需要适配融合方法的模型（有额外操作 + 原始硬编码融合） ──
from analysis.fusion_analysis.Diff_MSIN import Diff_MSIN
from analysis.fusion_analysis.EM3 import EM3
from analysis.fusion_analysis.MARN import MARN
from analysis.fusion_analysis.NAML import NAML
from analysis.fusion_analysis.M3SRec import M3SRec
from analysis.fusion_analysis.MB import MB
from analysis.fusion_analysis.MMMLP import MMMLP
from analysis.fusion_analysis.PAMD import PAMD

# ── 依赖预训练码本（RQ / PSRQ）的离散编码类模型 ──
from analysis.fusion_analysis.QARM import QARM
from analysis.fusion_analysis.MCCA import MCCA

# ── 已可配置融合方法的模型 ──
from analysis.fusion_analysis.dnn import DNN_mm, DNN_mm_seq
from analysis.fusion_analysis.DMF import DMF
from analysis.fusion_analysis.MAKE import MAKE

# ── 强制适配的 “融合即模型” 模型（忽略原始结构，统一为 BaseSeqModel + 可配置融合）──
from analysis.fusion_analysis.LMF import LMF
from analysis.fusion_analysis.MTFN import MTFN
from analysis.fusion_analysis.GMMF import GMMF
from analysis.fusion_analysis.SimCEN import SimCEN

# 融合方法
from analysis.fusion_analysis._fusion_helper import (
    LOCAL_FUSIONS, SEQ_FUSIONS, ALL_FUSIONS, normalize_fusion_method,
)
SUPPORTED_FUSIONS = sorted(ALL_FUSIONS)

# 所有模型均已强制适配为支持所有融合方法（含序列感知融合）
# BaseModel 模型在序列感知融合时自动跳过 pooling、直接对原始 3D 序列融合
_LOCAL_FUSION_ONLY = set()  # 不再有兼容性限制

# 可分析的模型注册表（模型名 → (模型类, 基类类型)）
# base_type: 'base' = BaseModel (forward(feats, feats_seq))
#            'seq'  = BaseSeqModel (forward(user_feats, feats, feats_seq))
#            'seq_label' = BaseSeqModel + label (forward(user_feats, feats, feats_seq, label))
MODEL_REGISTRY = {
    # ── 已可配置融合方法的模型 ──
    'DMF':         (DMF,          'seq'),
    'MAKE':        (MAKE,         'seq'),
    # ── 已适配的多模态 CTR 模型 ──
    'M3SRec':      (M3SRec,       'seq'),
    'MB':          (MB,           'base'),
    'MMMLP':       (MMMLP,        'seq'),
    'PAMD':        (PAMD,         'base'),
    # ── 强制适配的 “融合即模型” 模型 ──
    'LMF':         (LMF,          'seq'),
    'MTFN':        (MTFN,         'seq'),
    'GMMF':        (GMMF,         'seq'),
    'SimCEN':      (SimCEN,       'seq'),
    # ── 依赖预训练码本的离散编码类模型（需先训练好 RQ / PSRQ 码本）──
    'QARM':        (QARM,         'seq'),
    'MCCA':        (MCCA,         'seq'),
}

# 这些模型依赖 config/model.yaml 中的专属参数（codebook_size / n_levels /
# psrq_dims 等）以匹配预训练码本，需从 model.yaml 加载并合并到融合配置中。
SPECIAL_CONFIG_MODELS = {'QARM', 'MCCA'}

# 模型名大小写不敏感查找表（如用户传入 'mcca' 可解析到注册键 'MCCA'）
_MODEL_KEY_LOOKUP = {k.lower(): k for k in MODEL_REGISTRY}


def resolve_model_name(name):
    """将任意大小写的模型名解析为注册表中的规范键，未注册则返回 None。"""
    return _MODEL_KEY_LOOKUP.get(str(name).lower())
'''
其他已实现但默认不开启的模型（按需启用）：
    'DNN_mm':      (DNN_mm,       'base'),
    'DNN_mm_seq':  (DNN_mm_seq,   'seq'),
    'Diff_MSIN':   (Diff_MSIN,    'seq_label'),
    'EM3':         (EM3,          'seq'),
    'MARN':        (MARN,         'seq'),
    'NAML':        (NAML,         'seq'),
'''

def is_fusion_compatible(model_name, fusion_method):
    fusion_method = normalize_fusion_method(fusion_method)
    return not (model_name in _LOCAL_FUSION_ONLY and fusion_method in SEQ_FUSIONS)


def parse_args():
    parser = argparse.ArgumentParser(description="模态融合方法 × 模型架构 对比分析")
    parser.add_argument("--dataset_name", type=str, default="tiktok",
                        choices=["tiktok", "antm2c", "microlens"])
    parser.add_argument("--use_local_data", type=int, default=0,
                        help="是否使用本地离线数据 (0/1)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                        help=f"要测试的模型列表，可选: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--fusions", nargs="+", default=SUPPORTED_FUSIONS,
                        help=f"要测试的融合方法列表，可选: {SUPPORTED_FUSIONS}")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def build_model_config(model_name, fusion_method, base_config=None):
    """Build model config and inject a normalized fusion method."""
    fusion_method = normalize_fusion_method(fusion_method)
    config = {
        'model_name': f'{model_name}_{fusion_method}',
        'latent_dim': 128,
        'projection_dim': 128,
        'mlp_dims': [1024, 512, 256],
        'dropout': 0.0,
        'batch_norm': True,
        'seq_pooling_method': 'mean',
        'modal_fusion_method': fusion_method,
    }
    if base_config:
        config.update(base_config)
        config['modal_fusion_method'] = fusion_method
        config['model_name'] = f'{model_name}_{fusion_method}'
    return config


def run_single(model_name, fusion_method, dataloader, data_config, train_config,
               logger, seed, model_yaml=None, best_param=None, dataset_name=None):
    """训练并测试单个 (模型, 融合方法) 组合"""
    helper.setup_seed(seed)
    fusion_method = normalize_fusion_method(fusion_method)
    if not is_fusion_compatible(model_name, fusion_method):
        raise ValueError(
            f"Incompatible combination: {model_name} + {fusion_method}. "
            f"{model_name} only supports local fusion methods: {sorted(LOCAL_FUSIONS)}")

    model_cls, base_type = MODEL_REGISTRY[model_name]
    # 离散编码类模型需合并专属参数（含码本 codebook_size / n_levels）以匹配预训练码本。
    # 优先从 best_param.yaml 的最优参数加载，缺失时回退 model.yaml。
    base_config = None
    if model_name in SPECIAL_CONFIG_MODELS:
        ds_key = (dataset_name or data_config.get('name', '')).lower()
        bp_model = (best_param or {}).get(model_name.lower(), {}) or {}
        if bp_model.get(ds_key) is not None:
            base_config = copy.deepcopy(bp_model[ds_key])
            logger.info(f"[{model_name}] 从 best_param.yaml 加载码本参数: "
                        f"codebook_size={base_config.get('codebook_size')}, "
                        f"n_levels={base_config.get('n_levels')}")
        elif model_yaml is not None:
            logger.info(f"[{model_name}] best_param.yaml 无 [{model_name.lower()}][{ds_key}]，"
                        f"回退 model.yaml")
            base_config = copy.deepcopy(model_yaml.get(model_name.lower(), {}))
    model_config = build_model_config(model_name, fusion_method, base_config)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"模型: {model_name}  融合方法: {fusion_method.upper()}")
    logger.info(f"{'=' * 60}")

    model = model_cls(model_config, train_config, data_config, logger)

    start_time = time.time()
    best_val_auc = model.fit(dataloader)
    train_time = time.time() - start_time

    test_auc, test_loss = model.evalate(dataloader, 'test')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        'model': model_name,
        'fusion': fusion_method,
        'val_auc': best_val_auc,
        'test_auc': test_auc,
        'test_loss': test_loss,
        'train_time_s': round(train_time, 1),
        'total_params': total_params,
        'trainable_params': trainable_params,
    }

    logger.info(
        f"[{model_name}+{fusion_method}] val_auc={best_val_auc:.6f}  "
        f"test_auc={test_auc:.6f}  test_loss={test_loss:.6f}  "
        f"time={train_time:.1f}s  params={trainable_params:,}"
    )

    del model
    torch.cuda.empty_cache()
    return result


def print_comparison_table(results, logger):
    """按模型分组，打印对比表格"""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r['model']].append(r)

    logger.info("\n" + "=" * 110)
    logger.info("模态融合方法 × 模型架构 对比结果")
    logger.info("=" * 110)

    header = (f"{'模型':<14} {'融合方法':<10} {'Val AUC':>10} {'Test AUC':>10} "
              f"{'Test Loss':>10} {'时间(s)':>8} {'可训练参数':>14}")
    logger.info(header)
    logger.info("-" * 110)

    for model_name in sorted(grouped.keys()):
        model_results = sorted(grouped[model_name], key=lambda x: x['test_auc'], reverse=True)
        for r in model_results:
            line = (f"{r['model']:<14} {r['fusion']:<10} {r['val_auc']:>10.6f} "
                    f"{r['test_auc']:>10.6f} {r['test_loss']:>10.6f} "
                    f"{r['train_time_s']:>8.1f} {r['trainable_params']:>14,}")
            logger.info(line)
        logger.info("-" * 110)


def save_results_csv(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def main():
    args = parse_args()
    dataset_name = args.dataset_name.lower()

    # ── 加载配置 ──
    train_config = helper.load_yaml('config/train.yaml')
    # base 模型用 data.yaml；seq 模型用 seq_data.yaml（与主训练流程一致），
    # 这样 QARM/MCCA 的 RQ/PSRQ 预模型维度与训练时（seq_data.yaml）保持一致。
    data_yaml_all = helper.load_yaml('config/data.yaml')
    seq_yaml_all = helper.load_yaml('config/seq_data.yaml')
    # 离散编码类模型（QARM / MCCA）的专属超参；model.yaml 作为回退
    model_yaml = helper.load_yaml('config/model.yaml')
    # best_param.yaml 保存各模型/数据集的最优参数（含码本参数），优先加载
    best_param = helper.load_yaml('config/best_param.yaml')

    if args.max_epochs is not None:
        train_config['max_epochs'] = args.max_epochs
    if args.cuda is not None:
        train_config['cuda'] = args.cuda

    # ── 日志 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"fusion_model_analysis_{dataset_name}_{timestamp}"
    logger = helper.get_logger(log_name, train_config['log_dir'])

    logger.info(f"数据集: {dataset_name}")
    logger.info(f"待测试模型: {args.models}")
    logger.info(f"待测试融合方法: {args.fusions}")

    # ── dataloader（按 base/seq 类型选择数据配置，懒加载并缓存）──
    _loader_cache = {}

    def get_loader_and_config(base_type):
        is_seq = base_type in ('seq', 'seq_label')
        key = 'seq' if is_seq else 'base'
        if key not in _loader_cache:
            dc = copy.deepcopy((seq_yaml_all if is_seq else data_yaml_all)[dataset_name])
            if args.use_local_data:
                dc['using_local_data'] = True
            dl = helper.getDataLoader(dataset_name, dc, train_config["batch_size"])
            _loader_cache[key] = (dl, dc)
        return _loader_cache[key]

    # ── 逐模型逐融合方法测试 ──
    all_results = []
    for model_name in args.models:
        canonical_name = resolve_model_name(model_name)
        if canonical_name is None:
            logger.warning(f"跳过未知模型: {model_name}")
            continue
        model_name = canonical_name  # 统一使用注册表规范键
        base_type = MODEL_REGISTRY[model_name][1]
        dataloader, data_config = get_loader_and_config(base_type)
        for fusion in args.fusions:
            try:
                fusion = normalize_fusion_method(fusion)
            except ValueError:
                logger.warning(f"跳过不支持的融合方法: {fusion}")
                continue
            # 兼容性检查：部分模型不支持序列感知融合
            if not is_fusion_compatible(model_name, fusion):
                logger.info(f"跳过不兼容组合: {model_name} + {fusion}（该模型仅支持本地融合）")
                continue
            try:
                result = run_single(
                    model_name=model_name,
                    fusion_method=fusion,
                    dataloader=dataloader,
                    data_config=data_config,
                    train_config=train_config,
                    logger=logger,
                    seed=args.seed,
                    model_yaml=model_yaml,
                    best_param=best_param,
                    dataset_name=dataset_name,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"[{model_name}+{fusion}] 运行失败: {e}", exc_info=True)

    if not all_results:
        logger.error("所有组合均运行失败")
        return

    print_comparison_table(all_results, logger)

    csv_path = os.path.join(train_config['log_dir'],
                            f"fusion_model_analysis_{dataset_name}_{timestamp}.csv")
    save_results_csv(all_results, csv_path)
    logger.info(f"CSV 结果已保存到: {csv_path}")


if __name__ == "__main__":
    main()
