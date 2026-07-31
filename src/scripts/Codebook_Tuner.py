"""
码本(codebook)层数与大小调参脚本（QARM / MCCA 专用，多 GPU 并行版）

调参逻辑:
- 固定 lr / l2 等其他超参数（从 best_param.yaml 读取，保持一致）。
- 只网格搜索两个量化相关参数:
    codebook_size ∈ {256, 512, 1024}
    n_levels      ∈ {2, 3, 4}
- 关键耦合: 码本参数同时决定「量化预模型」和「推荐模型」的结构，二者必须一致。
  因此每个组合都需要:
    1) 用该组合重新训练对应的量化预模型 (QARM->RQ, MCCA->PSRQ) 并保存;
    2) 创建并训练推荐模型（会加载上一步的预模型），在验证集上评估 AUC。
- 对每个数据集选出验证集 AUC 最优的 (codebook_size, n_levels)，写回 config/best_param.yaml。
- 最后用最优组合在标准 checkpoint 目录重训一次，固化预模型与推荐模型权重。

多 GPU 并行:
- 每个 (model, dataset, codebook_size, n_levels) 试验是一个独立 job，分配到一块 GPU 上运行。
- 为避免并行写冲突，搜索阶段每个 job 使用「独立隔离的 checkpoint 子目录」
  （通过覆盖 train_config['checkpoint_dir'] 实现），互不影响；结束后自动清理。
- 所有进程的日志通过 QueueListener 汇聚到「同一个日志文件」（多进程安全）。

用法:
    python src/scripts/Codebook_Tuner.py --model_name all --dataset_name all --gpus 0-7
    python src/scripts/Codebook_Tuner.py --model_name qarm --dataset_name antm2c --gpus 0,1,2
"""

import os
import math
import psutil

# 限制每个进程的线程数，避免多进程并行时 CPU 过度争抢。
# 父进程会在确定并行进程数后，通过环境变量 CODEBOOK_THREADS 把「每进程线程数」
# 传给 spawn 出来的子进程；子进程在导入 numpy/BLAS 之前读取该值，确保线程限制真正生效。
_total_cpu = psutil.cpu_count(logical=True) or 1
_env_threads = os.environ.get("CODEBOOK_THREADS")
_per_proc = int(_env_threads) if _env_threads else max(1, _total_cpu // 8)
num_threads = str(_per_proc)
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import shutil
import logging
import logging.handlers
import argparse
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from mmctr.utils import helper
from mmctr.utils.tuning_protocol import SelectionMetrics, evaluate_for_selection, is_better
from models.pre_models.RQ import ResidualQuantizer
from models.pre_models.PSRQ import PSRQ_Premodel


# ----------------------------- 常量 -----------------------------
CODEBOOK_SIZES = [256, 512, 1024]
N_LEVELS = [2, 3, 4]
ALL_DATASETS = ["antm2c", "microlens", "tiktok"]
ALL_MODELS = ["qarm", "mcca"]
BEST_PARAM_PATH = "config/best_param.yaml"

# 每个 worker 进程内的全局状态（由 _pool_init 设置）
_LOG_QUEUE = None
_MY_GPU = None


# ----------------------------- 日志（多进程汇聚到单文件）-----------------------------
def _make_queue_logger(name):
    """返回一个把日志发送到全局队列的 logger（worker 进程内使用）。"""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.handlers.QueueHandler(_LOG_QUEUE))
    logger.propagate = False
    return logger


def _pool_init(log_queue, gpu_queue):
    """worker 进程初始化：绑定日志队列，从 GPU 队列领取本进程固定使用的 GPU，
    并按 CODEBOOK_THREADS 限制本进程的 CPU 线程数。"""
    global _LOG_QUEUE, _MY_GPU
    _LOG_QUEUE = log_queue
    _MY_GPU = gpu_queue.get()
    # 限制 CPU 线程（_per_proc 已在模块顶部依据 CODEBOOK_THREADS 计算）
    try:
        torch.set_num_threads(_per_proc)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# ----------------------------- best_param.yaml 更新 -----------------------------
def update_best_param(model_name, dataset_name, updates, path=BEST_PARAM_PATH):
    """更新 best_param.yaml 中 [model][dataset] 的指定键，尽量保留注释/格式。"""
    try:
        from ruamel.yaml import YAML
        yaml_rt = YAML()
        yaml_rt.preserve_quotes = True
        yaml_rt.indent(mapping=4, sequence=4, offset=2)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml_rt.load(f)
        if data.get(model_name) is None:
            data[model_name] = {}
        if data[model_name].get(dataset_name) is None:
            data[model_name][dataset_name] = {}
        for k, v in updates.items():
            data[model_name][dataset_name][k] = v
        with open(path, "w", encoding="utf-8") as f:
            yaml_rt.dump(data, f)
    except ImportError:
        import yaml as pyyaml
        with open(path, "r", encoding="utf-8") as f:
            data = pyyaml.safe_load(f) or {}
        data.setdefault(model_name, {})
        if data[model_name].get(dataset_name) is None:
            data[model_name][dataset_name] = {}
        for k, v in updates.items():
            data[model_name][dataset_name][k] = v
        with open(path, "w", encoding="utf-8") as f:
            pyyaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ----------------------------- 量化预模型训练 -----------------------------
def train_rq_premodel(dataloader, model_cfg_yaml, base_model_cfg, train_config, data_config_ds,
                      codebook_size, n_levels, logger):
    """按指定码本参数训练 RQ 预模型（逐模态），保存到 train_config['checkpoint_dir']。

    RQ 保存的是 numpy 码本，加载时按码本恢复，QARM 仅依赖 codebook_size / n_levels
    与之匹配（n_init/max_iter 等只影响训练过程，不影响加载结果）。
    """
    epsilon = 1e-8
    rq_cfg = deepcopy(model_cfg_yaml.get("rq", {}))
    rq_cfg["codebook_size"] = codebook_size
    rq_cfg["n_levels"] = n_levels

    dataset = data_config_ds["name"]
    ckpt_dir = train_config["checkpoint_dir"]

    mm_modals = dataloader.get_multi_modal()
    for k in mm_modals.keys():
        mm_modal = mm_modals[k] / np.maximum(
            np.linalg.norm(mm_modals[k], axis=1, keepdims=True), epsilon)
        rq = ResidualQuantizer(deepcopy(rq_cfg), train_config, data_config_ds)
        rq.fit(mm_modal, verbose=False)
        rq.save(f"{ckpt_dir}/{dataset}_{k}_rq.npz")
    logger.info(f"[RQ] 预模型已保存: codebook_size={codebook_size}, n_levels={n_levels}, dir={ckpt_dir}")


def train_psrq_premodel(dataloader, model_cfg_yaml, base_model_cfg, train_config, data_config_ds,
                        codebook_size, n_levels, logger):
    """按指定码本参数训练 PSRQ 预模型，保存到 train_config['checkpoint_dir']。

    关键: PSRQ 通过 state_dict 加载，结构必须与 MCCA 内部构造的 PSRQ 完全一致。
    MCCA 用推荐模型配置(base_model_cfg)构造其 PSRQ，故这里的结构参数(psrq_dims/
    latent_dim/projection_dim/dropout/batch_norm)同样取自 base_model_cfg。
    """
    psrq_cfg = deepcopy(model_cfg_yaml.get("psrq", {}))  # 提供 mu / quant_loss_weight 等默认
    for key in ["latent_dim", "projection_dim", "psrq_dims", "dropout", "batch_norm"]:
        if key in base_model_cfg:
            psrq_cfg[key] = base_model_cfg[key]
    psrq_cfg["codebook_size"] = codebook_size
    psrq_cfg["n_levels"] = n_levels

    # PSRQ 的 KMeans 初始化要求 batch_size >= codebook_size
    pre_train_config = deepcopy(train_config)
    pre_train_config["batch_size"] = max(int(train_config["batch_size"]), int(codebook_size))

    model = PSRQ_Premodel(psrq_cfg, pre_train_config, data_config_ds)
    model.fit(dataloader)
    model.save()
    logger.info(f"[PSRQ] 预模型已保存: codebook_size={codebook_size}, n_levels={n_levels}, "
                f"premodel_batch_size={pre_train_config['batch_size']}, "
                f"dir={pre_train_config['checkpoint_dir']}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_premodel(model_name, *args, **kwargs):
    if model_name == "qarm":
        train_rq_premodel(*args, **kwargs)
    elif model_name == "mcca":
        train_psrq_premodel(*args, **kwargs)
    else:
        raise ValueError(f"不支持的模型: {model_name}")


# ----------------------------- 单组合：训练预模型 + 推荐模型 -----------------------------
def _load_base_cfg(model_name, dataset_name, model_cfg_yaml, best_param, logger):
    """获取推荐模型基础配置（固定 lr/l2 等），优先 best_param，缺失回退 model.yaml。"""
    base = best_param.get(model_name, {}).get(dataset_name)
    if base is None:
        logger.info(f"best_param.yaml 无 [{model_name}][{dataset_name}]，回退 model.yaml")
        mm = model_cfg_yaml.get(model_name, {})
        base = mm.get(dataset_name, mm)
    return deepcopy(base)


def run_single_trial(model_name, dataset_name, codebook_size, n_levels, ckpt_dir, gpu, logger):
    """在指定 GPU、指定隔离目录下训练模型，并返回 validation 选优指标。"""
    model_cfg_yaml = helper.load_yaml("config/model.yaml")
    best_param = helper.load_yaml(BEST_PARAM_PATH)
    train_config = helper.load_yaml("config/train.yaml")
    data_config_ds = helper.load_yaml("config/seq_data.yaml")[dataset_name]

    base_model_cfg = _load_base_cfg(model_name, dataset_name, model_cfg_yaml, best_param, logger)

    # 固定 lr/l2，设置 GPU 与隔离的 checkpoint 目录
    train_config["cuda"] = gpu
    train_config["checkpoint_dir"] = ckpt_dir
    if "lr" in base_model_cfg:
        train_config["lr"] = base_model_cfg["lr"]
    if "l2" in base_model_cfg:
        train_config["l2"] = base_model_cfg["l2"]
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    dataloader = helper.getDataLoader(dataset_name, data_config_ds, train_config["batch_size"])

    # 1) 训练量化预模型（结构与本组合一致）
    train_premodel(model_name, dataloader, model_cfg_yaml, base_model_cfg, train_config,
                   data_config_ds, codebook_size, n_levels, logger)

    # 2) 推荐模型：固定其他超参，只覆盖码本参数
    rec_model_config = deepcopy(base_model_cfg)
    rec_model_config["model_name"] = model_name
    rec_model_config["codebook_size"] = codebook_size
    rec_model_config["n_levels"] = n_levels

    model = helper.getModel(model_name, rec_model_config, train_config, data_config_ds, logger)
    model.fit(dataloader)
    val_metrics = evaluate_for_selection(model, dataloader)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_metrics


# ----------------------------- worker -----------------------------
def worker(job):
    """job: dict(model, dataset, C, L, mode, ckpt_dir)。返回结果 dict。"""
    logger = _make_queue_logger(f"trial.{job['model']}.{job['dataset']}")
    gpu = _MY_GPU
    tag = (f"{job['model']}/{job['dataset']} codebook_size={job['C']} "
           f"n_levels={job['L']} [{job['mode']}] gpu={gpu}")
    result = {"model": job["model"], "dataset": job["dataset"],
              "C": job["C"], "L": job["L"], "mode": job["mode"], "ok": False}
    try:
        logger.info(f"START {tag}")
        val_metrics = run_single_trial(
            job["model"], job["dataset"], job["C"], job["L"],
            job["ckpt_dir"], gpu, logger,
        )
        result.update({
            "ok": True,
            "val_auc": val_metrics.auc,
            "val_loss": val_metrics.loss,
        })
        logger.info(
            f"DONE  {tag} -> Validation AUC={val_metrics.auc:.6f}, "
            f"Validation Loss={val_metrics.loss:.6f}"
        )
    except Exception as e:
        logger.error(f"FAIL  {tag}: {repr(e)}")
    finally:
        # 搜索阶段的隔离目录用完即删，避免占用磁盘
        if job["mode"] == "search":
            shutil.rmtree(job["ckpt_dir"], ignore_errors=True)
    return result


# ----------------------------- 主流程 -----------------------------
def parse_gpus(s):
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser(description="QARM/MCCA 码本调参（多GPU并行）")
    parser.add_argument("--model_name", type=str, default="all", help="qarm / mcca / all")
    parser.add_argument("--dataset_name", type=str, default="all",
                        help="antm2c / microlens / tiktok / all")
    parser.add_argument("--gpus", type=str, default="0-7", help="如 0-7 或 0,1,2")
    parser.add_argument("--cpu_fraction", type=float, default=0.5,
                        help="所有进程合计 CPU 占用上限比例（默认 0.5 即总核数的 50%）")
    args = parser.parse_args()

    models = ALL_MODELS if args.model_name.lower() == "all" else [args.model_name.lower()]
    datasets = ALL_DATASETS if args.dataset_name.lower() == "all" else [args.dataset_name.lower()]
    models = [m for m in models if m in ALL_MODELS]
    gpus = parse_gpus(args.gpus)
    n_workers = max(1, len(gpus))

    # CPU 限制：所有进程合计不超过 cpu_fraction * 总核数，平摊到每个进程
    total_threads = max(1, int(args.cpu_fraction * _total_cpu))
    per_proc_threads = max(1, total_threads // n_workers)
    # 通过环境变量传给 spawn 子进程（子进程在导入 numpy/BLAS 前读取并设置线程数）
    os.environ["CODEBOOK_THREADS"] = str(per_proc_threads)
    for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[_v] = str(per_proc_threads)

    # 标准 checkpoint 目录 & 隔离临时目录
    base_ckpt = helper.load_yaml("config/train.yaml").get("checkpoint_dir", "experiments/checkpoints/")
    tmp_root = str(Path(base_ckpt) / "codebook_tune_tmp")
    Path(tmp_root).mkdir(parents=True, exist_ok=True)

    # 单一日志文件
    log_dir = helper.load_yaml("config/train.yaml").get("log_dir", "experiments/logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    single_log = str(Path(log_dir) / f"codebook_tune_{ts}.log")

    # spawn 启动（CUDA + 多进程必须）
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    log_queue = manager.Queue()
    gpu_queue = manager.Queue()
    for g in gpus:
        gpu_queue.put(g)

    # 日志监听器：把队列里所有进程的日志写入同一个文件 + 控制台
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(single_log, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    listener = logging.handlers.QueueListener(log_queue, fh, ch, respect_handler_level=True)
    listener.start()

    main_logger = logging.getLogger("codebook_main")
    main_logger.handlers.clear()
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(logging.handlers.QueueHandler(log_queue))
    main_logger.propagate = False

    main_logger.info(f"===== 码本调参启动 =====")
    main_logger.info(f"模型: {models}, 数据集: {datasets}, GPU: {gpus}, 并行进程: {n_workers}")
    main_logger.info(f"CPU 限制: 合计上限 {args.cpu_fraction*100:.0f}% ({total_threads}/{_total_cpu} 线程), "
                     f"每进程 {per_proc_threads} 线程")
    main_logger.info(f"搜索空间 codebook_size={CODEBOOK_SIZES}, n_levels={N_LEVELS}")
    main_logger.info(f"统一日志文件: {single_log}")

    # 搜索 jobs（每个组合一个隔离目录）
    search_jobs = []
    for m in models:
        for d in datasets:
            for C in CODEBOOK_SIZES:
                for L in N_LEVELS:
                    ckpt_dir = str(Path(tmp_root) / f"{m}_{d}_c{C}_l{L}")
                    search_jobs.append({"model": m, "dataset": d, "C": C, "L": L,
                                        "mode": "search", "ckpt_dir": ckpt_dir})

    main_logger.info(f"搜索试验总数: {len(search_jobs)}")

    pool = mp.Pool(n_workers, initializer=_pool_init, initargs=(log_queue, gpu_queue))

    best = {}  # (model,dataset) -> best result dict
    done = 0
    for res in pool.imap_unordered(worker, search_jobs):
        done += 1
        if res.get("ok"):
            g = (res["model"], res["dataset"])
            incumbent = None
            if g in best:
                incumbent = SelectionMetrics(
                    auc=best[g]["val_auc"],
                    loss=best[g]["val_loss"],
                )
            candidate = SelectionMetrics(auc=res["val_auc"], loss=res["val_loss"])
            if is_better(candidate, incumbent):
                best[g] = res
                main_logger.info(f"[{done}/{len(search_jobs)}] 新最优 {g[0]}/{g[1]}: "
                                 f"codebook_size={res['C']}, n_levels={res['L']}, "
                                 f"Validation AUC={res['val_auc']:.6f}")
        else:
            main_logger.info(f"[{done}/{len(search_jobs)}] 试验失败已跳过")

    # 写回 best_param.yaml
    for (m, d), b in best.items():
        update_best_param(m, d, {"codebook_size": int(b["C"]), "n_levels": int(b["L"])})
        main_logger.info(f"已写入 best_param.yaml: [{m}][{d}] "
                         f"codebook_size={b['C']}, n_levels={b['L']} "
                         f"(Validation AUC={b['val_auc']:.6f})")

    # 固化阶段：用最优组合在标准目录重训（不同 (model,dataset) 路径不冲突，可并行）
    persist_jobs = [{"model": m, "dataset": d, "C": b["C"], "L": b["L"],
                     "mode": "persist", "ckpt_dir": base_ckpt}
                    for (m, d), b in best.items()]
    if persist_jobs:
        main_logger.info(f"固化最优组合（重训保存至 {base_ckpt}）: {len(persist_jobs)} 个")
        for res in pool.imap_unordered(worker, persist_jobs):
            status = "成功" if res.get("ok") else "失败"
            main_logger.info(f"固化 {res['model']}/{res['dataset']} "
                             f"codebook_size={res['C']}, n_levels={res['L']}: {status}")

    pool.close()
    pool.join()

    # 清理临时目录
    shutil.rmtree(tmp_root, ignore_errors=True)

    # 汇总
    main_logger.info("===== 调参完成，各 (模型,数据集) 最优码本参数 =====")
    for (m, d), b in sorted(best.items()):
        main_logger.info(
            f"  {m}/{d}: codebook_size={b['C']}, n_levels={b['L']}, "
            f"Validation AUC={b['val_auc']:.6f}"
        )
    main_logger.info(f"统一日志: {single_log}")

    listener.stop()
    print(f"\n调参完成。统一日志: {single_log}")


if __name__ == "__main__":
    main()
