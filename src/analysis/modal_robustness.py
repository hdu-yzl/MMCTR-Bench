"""
模态鲁棒性分析实验（训练 + 测试均施加掩码，多 GPU 并行）
==================
每个 (model, dataset, drop_ratio) 组合独立训练 + 评估：
    - 训练阶段：每个 batch 随机生成掩码，将 drop_ratio 比例样本的非 ID 模态置零
      （不同 epoch 掩码不同，相当于数据增强）。
    - 测试阶段：确定性种子生成掩码，保证跨模型比较公平。

多 GPU 并行：将任务均匀分配到指定 GPU（默认 0/1/2/3），每张卡顺序执行
分配到的任务，不同 GPU 之间并行。每个任务有独立日志文件，不会重叠。

最终为每个数据集输出一张结果大表（AUC / ΔAUC / Logloss × drop_ratio）。

用法:
    python src/analysis/modal_robustness.py
    python src/analysis/modal_robustness.py --models dnn_mm lmf marn --datasets antm2c
    python src/analysis/modal_robustness.py --drop_ratios 0.0 0.2 0.5 1.0
    python src/analysis/modal_robustness.py --gpu_ids 0 1 2 3
    python src/analysis/modal_robustness.py --use_local_data 1
"""

import os
import sys
import csv
import logging
import argparse
import traceback
import numpy as np
import torch
import torch.multiprocessing as mp
from datetime import datetime
from sklearn import metrics

# 将 src/ 加入 Python 路径（文件位于 src/analysis/，向上一层即 src/）
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils import helper  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# CPU 利用率限制
# ─────────────────────────────────────────────────────────────────────────────
def limit_cpu(max_cpu_pct: float = 0.40, num_workers: int = 4):
    """
    限制当前进程的 CPU 线程数，使所有 worker 总 CPU 利用率不超过 max_cpu_pct。
    按 (总核数 × max_cpu_pct / num_workers) 计算每个 worker 可用线程数，最少 1。
    同时设置 PyTorch / TF / OpenMP / MKL 等线程池上限。
    """
    total_cores = os.cpu_count() or 8
    threads_per_worker = max(1, int(total_cores * max_cpu_pct / num_workers))

    # PyTorch
    torch.set_num_threads(threads_per_worker)
    torch.set_num_interop_threads(max(1, threads_per_worker // 2))

    # 环境变量（影响 OpenMP / MKL / NumExpr / TF 等）
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads_per_worker)

    # TensorFlow（数据加载用到 tf.data）
    try:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(threads_per_worker)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, threads_per_worker // 2))
    except Exception:
        pass

    return threads_per_worker


# ─────────────────────────────────────────────────────────────────────────────
# 默认实验配置
# ─────────────────────────────────────────────────────────────────────────────
ALL_MODELS = [
    "dnn_mm",
    "lmf",
    "marn",
    "mtfn",
    "naml",
    "make",
    "em3",
    "simcen",
    "gmmf",
    "dmf",
]

ALL_DATASETS = ["antm2c", "microlens"]

# 测试的缺失比例梯度
DROP_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]

# 用于生成可复现掩码的全局基底种子（所有模型统一使用，不可修改）
MASK_BASE_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 日志工具
# ─────────────────────────────────────────────────────────────────────────────
def setup_main_logger(log_dir: str) -> logging.Logger:
    """主进程汇总日志。"""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"modal_robustness_summary_{ts}.log")

    logger = logging.getLogger("robustness_main")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def setup_task_logger(log_dir: str, model_name: str, dataset_name: str,
                      drop_ratio: float, gpu_id: int) -> logging.Logger:
    """每个 (model, dataset, drop_ratio) 任务的独立日志，文件名含 GPU 编号。"""
    os.makedirs(log_dir, exist_ok=True)
    tag = f"rob_{model_name}_{dataset_name}_d{drop_ratio:.1f}_gpu{gpu_id}"
    log_file = os.path.join(log_dir, f"{tag}.log")

    logger = logging.getLogger(tag)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# 核心：批次级模态缺失
# ─────────────────────────────────────────────────────────────────────────────
def _zero_non_id(feat_dict: dict, mask: torch.Tensor) -> dict:
    """
    将 feat_dict 中所有非 'id' 键对应张量的 mask 位置（样本维度）置零。
    操作在 .clone() 后的副本上进行，不修改原始张量。
    """
    result = {}
    for k, v in feat_dict.items():
        if k != "id" and mask.any():
            v = v.clone()
            v[mask] = 0.0
        result[k] = v
    return result


def apply_modal_drop(batch: tuple, is_seq_model: bool,
                     drop_ratio: float, batch_idx: int) -> tuple:
    """
    确定性模态缺失（测试用）。
    seed = MASK_BASE_SEED * 1_000_003 + batch_idx，跨模型一致。
    """
    if drop_ratio <= 0.0:
        return batch

    if is_seq_model:
        user_feats, feats, feats_seq, label = batch
    else:
        feats, feats_seq, label = batch

    batch_size = label.shape[0]
    rng = np.random.RandomState(MASK_BASE_SEED * 1_000_003 + batch_idx)
    rand_vals = rng.random(batch_size)
    mask_np = rand_vals < drop_ratio
    mask_t = torch.from_numpy(mask_np)

    if not mask_t.any():
        return batch

    feats     = _zero_non_id(feats, mask_t)
    feats_seq = _zero_non_id(feats_seq, mask_t)

    if is_seq_model:
        user_feats = _zero_non_id(user_feats, mask_t)
        return user_feats, feats, feats_seq, label
    else:
        return feats, feats_seq, label


def apply_train_modal_drop(batch: tuple, is_seq_model: bool,
                           drop_ratio: float,
                           epoch_idx: int, batch_idx: int) -> tuple:
    """
    确定性模态缺失（训练用）。
    seed = MASK_BASE_SEED * 1_000_003 + epoch_idx * 100_003 + batch_idx
    保证同一 (epoch, batch_idx) 下所有模型掩码一致；
    不同 epoch 掩码不同（相当于数据增强）。
    """
    if drop_ratio <= 0.0:
        return batch

    if is_seq_model:
        user_feats, feats, feats_seq, label = batch
    else:
        feats, feats_seq, label = batch

    batch_size = label.shape[0]
    rng = np.random.RandomState(MASK_BASE_SEED * 1_000_003 + epoch_idx * 100_003 + batch_idx)
    rand_vals = rng.random(batch_size)
    mask_np = rand_vals < drop_ratio
    mask_t = torch.from_numpy(mask_np)

    if not mask_t.any():
        return batch

    feats     = _zero_non_id(feats, mask_t)
    feats_seq = _zero_non_id(feats_seq, mask_t)

    if is_seq_model:
        user_feats = _zero_non_id(user_feats, mask_t)
        return user_feats, feats, feats_seq, label
    else:
        return feats, feats_seq, label


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载器包装：训练时自动施加模态缺失
# ─────────────────────────────────────────────────────────────────────────────
class MaskedDataLoaderWrapper:
    """包装 DataLoader，在训练阶段对每个 batch 施加确定性模态掩码。

    epoch 计数器自动递增：每次调用 get_data/get_data_seq("train") 时
    epoch +1，保证不同 epoch 掩码不同、同 epoch 内跨模型掩码一致。
    """

    def __init__(self, dataloader, drop_ratio: float):
        self._dl = dataloader
        self._drop_ratio = drop_ratio
        self._train_epoch = -1  # 首次调用后变为 0

    def get_data(self, data_type):
        if data_type == "train":
            self._train_epoch += 1
        for batch_idx, batch in enumerate(self._dl.get_data(data_type)):
            if data_type == "train" and self._drop_ratio > 0:
                batch = apply_train_modal_drop(
                    batch, False, self._drop_ratio,
                    self._train_epoch, batch_idx)
            yield batch

    def get_data_seq(self, data_type):
        if data_type == "train":
            self._train_epoch += 1
        for batch_idx, batch in enumerate(self._dl.get_data_seq(data_type)):
            if data_type == "train" and self._drop_ratio > 0:
                batch = apply_train_modal_drop(
                    batch, True, self._drop_ratio,
                    self._train_epoch, batch_idx)
            yield batch

    def __getattr__(self, name):
        return getattr(self._dl, name)


# ─────────────────────────────────────────────────────────────────────────────
# 带模态缺失的推理评估
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_with_drop(model, data_loader, drop_ratio: float):
    """在测试集上带确定性模态缺失的推理，返回 (AUC, Logloss)。"""
    model.eval()
    is_seq = model.seq_modeling
    device = model.device
    preds, trues = [], []

    data_iter = data_loader.get_data_seq("test") if is_seq else data_loader.get_data("test")

    for batch_idx, batch in enumerate(data_iter):
        batch = apply_modal_drop(batch, is_seq, drop_ratio, batch_idx)

        if is_seq:
            user_feats, feats, feats_seq, label = batch
            user_feats = {k: v.to(device) for k, v in user_feats.items()}
            feats      = {k: v.to(device) for k, v in feats.items()}
            feats_seq  = {k: v.to(device) for k, v in feats_seq.items()}
            label      = label.to(device)
            out = model(user_feats, feats, feats_seq)
        else:
            feats, feats_seq, label = batch
            feats     = {k: v.to(device) for k, v in feats.items()}
            feats_seq = {k: v.to(device) for k, v in feats_seq.items()}
            label     = label.to(device)
            out = model(feats, feats_seq)

        preds.append(out["pred"].sigmoid().detach().cpu().numpy())
        trues.append(label.detach().cpu().numpy())

    y_pred = np.concatenate(preds).astype("float64")
    y_true = np.concatenate(trues).astype("float64")
    auc     = metrics.roc_auc_score(y_true, y_pred)
    logloss = metrics.log_loss(y_true, y_pred)
    return auc, logloss


# ─────────────────────────────────────────────────────────────────────────────
# 单个 (model, dataset, drop_ratio) 任务
# ─────────────────────────────────────────────────────────────────────────────
def run_single_experiment(model_name: str, dataset_name: str,
                          drop_ratio: float, gpu_id: int,
                          use_local_data: bool, log_dir: str) -> dict:
    """
    训练（带模态掩码）+ 测试（确定性模态掩码），返回结果字典。
    """
    logger = setup_task_logger(log_dir, model_name, dataset_name, drop_ratio, gpu_id)
    logger.info(f"[{model_name} | {dataset_name} | drop={drop_ratio:.2f} | GPU={gpu_id}] 开始")

    # ── 配置加载 ──────────────────────────────────────────────
    try:
        model_config = helper.load_yaml("config/best_param.yaml")[model_name][dataset_name]
    except KeyError:
        raise KeyError(
            f"best_param.yaml 中未找到 [{model_name}][{dataset_name}] 配置，"
            "请先完成该组合的超参数调优。"
        )
    # 加后缀区分不同 drop_ratio 和数据集，避免 checkpoint 冲突
    model_config["model_name"] = f"{model_name}_rob_{dataset_name}_d{drop_ratio:.1f}"
    is_seq = model_config.get("seq_modeling", False)

    if is_seq:
        data_cfg_file = "config/local_seq_data.yaml" if use_local_data else "config/seq_data.yaml"
    else:
        data_cfg_file = "config/local_data.yaml" if use_local_data else "config/data.yaml"

    data_config  = helper.load_yaml(data_cfg_file)
    train_config = helper.load_yaml("config/train.yaml")
    train_config["lr"] = model_config.get("lr", train_config.get("lr", 1e-3))
    train_config["l2"] = model_config.get("l2", train_config.get("l2", 1e-6))
    train_config["cuda"] = gpu_id
    data_config[dataset_name]['use_mm_features'] = ['id', 'text', 'image']
    data_config[dataset_name]['use_mm_seq_features'] = ['id', 'text', 'image']

    # ── 数据加载 ──────────────────────────────────────────────
    dataloader = helper.getDataLoader(
        dataset_name, data_config[dataset_name], train_config["batch_size"]
    )
    masked_dl = MaskedDataLoaderWrapper(dataloader, drop_ratio)

    # ── 模型构建 ──────────────────────────────────────────────
    model_logger = helper.get_logger(
        f"{model_name}_{dataset_name}_rob_d{drop_ratio:.1f}_gpu{gpu_id}",
        train_config["log_dir"]
    )
    model = helper.getModel(
        model_name, model_config, train_config,
        data_config[dataset_name], model_logger
    )

    # ── 训练（带掩码的 DataLoader）────────────────────────────
    logger.info(f"开始训练 (train drop_ratio={drop_ratio:.2f}) ...")
    best_val_auc = model.fit(masked_dl)
    logger.info(f"训练完成，最优 Val AUC={best_val_auc:.6f}")

    # ── 测试（确定性掩码）──────────────────────────────────────
    auc, logloss = evaluate_with_drop(model, dataloader, drop_ratio)
    logger.info(f"测试结果: AUC={auc:.6f}, Logloss={logloss:.6f}")

    del model
    torch.cuda.empty_cache()
    return {"auc": auc, "logloss": logloss, "val_auc": best_val_auc}


# ─────────────────────────────────────────────────────────────────────────────
# GPU Worker（子进程入口）
# ─────────────────────────────────────────────────────────────────────────────
def gpu_worker(gpu_id: int, tasks: list, use_local_data: bool,
               log_dir: str, result_queue, num_workers: int = 4):
    """单个 GPU 上顺序执行分配到的所有任务。"""
    # spawn 模式下需要确保 src/ 在 path 中
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # 限制本 worker 的 CPU 线程数，使总 CPU 不超过 40%
    thr = limit_cpu(max_cpu_pct=0.40, num_workers=num_workers)
    print(f"[GPU {gpu_id}] CPU 线程限制: {thr} threads/worker")

    for model_name, dataset_name, drop_ratio in tasks:
        try:
            res = run_single_experiment(
                model_name, dataset_name, drop_ratio,
                gpu_id, use_local_data, log_dir
            )
            result_queue.put((model_name, dataset_name, drop_ratio, res))
        except Exception:
            tb = traceback.format_exc()
            print(f"[GPU {gpu_id}] 失败: {model_name}|{dataset_name}|d={drop_ratio:.2f}\n{tb}")
            result_queue.put((model_name, dataset_name, drop_ratio, None))


# ─────────────────────────────────────────────────────────────────────────────
# 结果输出
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(all_results: dict, drop_ratios: list, output_path: str):
    """将结果保存为 CSV 文件，格式：model, dataset, drop_ratio, auc, logloss, delta_auc"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dataset", "drop_ratio", "auc", "logloss", "delta_auc"])
        for (model, dataset), ratio_dict in all_results.items():
            base_auc = ratio_dict.get(0.0, {}).get("auc", None)
            for ratio in drop_ratios:
                if ratio not in ratio_dict:
                    continue
                r = ratio_dict[ratio]
                delta = (r["auc"] - base_auc) if base_auc is not None else ""
                writer.writerow([
                    model, dataset, f"{ratio:.2f}",
                    f"{r['auc']:.6f}",
                    f"{r['logloss']:.6f}",
                    f"{delta:+.6f}" if isinstance(delta, float) else "",
                ])
    print(f"\n结果已保存至: {output_path}")


def print_dataset_table(dataset: str, all_results: dict, models: list,
                        drop_ratios: list, logger: logging.Logger):
    """为单个数据集输出结果大表：行 = 模型，列 = drop_ratio。"""
    col_w = 12
    header_cols = "  ".join(f"d={r:.1f}".center(col_w) for r in drop_ratios)

    # ── AUC 表 ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 100)
    logger.info(f"【{dataset}】模态鲁棒性 —— AUC (↑越高越好)")
    logger.info("=" * 100)
    logger.info(f"{'模型':<16} {header_cols}")
    logger.info("-" * 100)
    for model in models:
        key = (model, dataset)
        if key not in all_results:
            continue
        res = all_results[key]
        vals = "  ".join(
            f"{res[r]['auc']:.6f}".center(col_w) if r in res else "N/A".center(col_w)
            for r in drop_ratios
        )
        logger.info(f"{model:<16} {vals}")

    # ── ΔAUC 表 ─────────────────────────────────────────────
    logger.info("")
    logger.info(f"【{dataset}】ΔAUC (相对 drop=0.0 的变化量，负数 = 退化)")
    logger.info("-" * 100)
    logger.info(f"{'模型':<16} {header_cols}")
    logger.info("-" * 100)
    for model in models:
        key = (model, dataset)
        if key not in all_results:
            continue
        res = all_results[key]
        base = res.get(0.0, {}).get("auc", None)
        vals = "  ".join(
            (f"{res[r]['auc'] - base:+.6f}".center(col_w)
             if (r in res and base is not None) else "N/A".center(col_w))
            for r in drop_ratios
        )
        logger.info(f"{model:<16} {vals}")

    # ── Logloss 表 ────────────────────────────────────────────
    logger.info("")
    logger.info(f"【{dataset}】Logloss (↓越低越好)")
    logger.info("-" * 100)
    logger.info(f"{'模型':<16} {header_cols}")
    logger.info("-" * 100)
    for model in models:
        key = (model, dataset)
        if key not in all_results:
            continue
        res = all_results[key]
        vals = "  ".join(
            f"{res[r]['logloss']:.6f}".center(col_w) if r in res else "N/A".center(col_w)
            for r in drop_ratios
        )
        logger.info(f"{model:<16} {vals}")

    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="模态鲁棒性分析：训练 + 测试均随机置零非 ID 模态，多 GPU 并行"
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="要评估的模型列表，默认全部")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help="要评估的数据集列表，默认全部")
    parser.add_argument("--drop_ratios", nargs="+", type=float, default=DROP_RATIOS,
                        help="模态缺失比例列表，取值范围 [0,1]")
    parser.add_argument("--use_local_data", type=int, default=0,
                        help="是否使用本地数据路径（0/1）")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1, 2, 3],
                        help="可用 GPU 编号列表，默认 0 1 2 3")
    parser.add_argument("--log_dir", type=str, default="experiments/logs",
                        help="日志保存目录")
    parser.add_argument("--output_csv", type=str,
                        default="experiments/results/modal_robustness.csv",
                        help="结果 CSV 保存路径")
    args = parser.parse_args()

    num_gpus = len(args.gpu_ids)
    main_logger = setup_main_logger(args.log_dir)

    # ── 构建任务列表 ──────────────────────────────────────────
    all_tasks = []
    for dataset in args.datasets:
        for model in args.models:
            for ratio in args.drop_ratios:
                all_tasks.append((model, dataset, ratio))

    # ── 轮询分配任务到各 GPU ──────────────────────────────────
    gpu_tasks = {gid: [] for gid in args.gpu_ids}
    for i, task in enumerate(all_tasks):
        gid = args.gpu_ids[i % num_gpus]
        gpu_tasks[gid].append(task)

    main_logger.info("=" * 60)
    main_logger.info("模态鲁棒性分析实验启动（训练 + 测试均施加掩码）")
    main_logger.info(f"  测试掩码种子  : {MASK_BASE_SEED}  (不可变，保证跨模型一致性)")
    main_logger.info(f"  缺失比例梯度  : {args.drop_ratios}")
    main_logger.info(f"  模型列表      : {args.models}")
    main_logger.info(f"  数据集列表    : {args.datasets}")
    main_logger.info(f"  GPU 列表      : {args.gpu_ids}")
    main_logger.info(f"  共 {len(all_tasks)} 个任务")
    for gid in args.gpu_ids:
        main_logger.info(f"    GPU {gid}: {len(gpu_tasks[gid])} 个任务")
    main_logger.info("=" * 60)

    # ── 限制主进程 CPU + 多 GPU 并行执行 ─────────────────────
    active_gpus = [gid for gid in args.gpu_ids if gpu_tasks[gid]]
    n_workers = len(active_gpus)
    main_thr = limit_cpu(max_cpu_pct=0.40, num_workers=max(n_workers, 1))
    main_logger.info(f"  CPU 线程限制  : {main_thr} threads/worker × {n_workers} workers "
                     f"(总核数 {os.cpu_count()}, 上限 40%)")

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    processes = []
    for gid in active_gpus:
        p = mp.Process(
            target=gpu_worker,
            args=(gid, gpu_tasks[gid], bool(args.use_local_data),
                  args.log_dir, result_queue, n_workers),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # ── 收集结果 ──────────────────────────────────────────────
    all_results = {}   # {(model, dataset): {ratio: {"auc":..., "logloss":...}}}
    failed = []
    while not result_queue.empty():
        model, dataset, ratio, res = result_queue.get()
        if res is not None:
            all_results.setdefault((model, dataset), {})[ratio] = res
        else:
            failed.append(f"{model}+{dataset}+d{ratio:.1f}")

    # ── 按数据集输出汇总大表 ──────────────────────────────────
    for dataset in args.datasets:
        print_dataset_table(dataset, all_results, args.models, args.drop_ratios, main_logger)

    if all_results:
        save_csv(all_results, args.drop_ratios, args.output_csv)

    main_logger.info(f"全部完成: 成功 {len(all_tasks) - len(failed)}/{len(all_tasks)}")
    if failed:
        main_logger.warning(f"失败任务: {failed}")


if __name__ == "__main__":
    main()
