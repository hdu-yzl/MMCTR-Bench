"""
批量运行所有模型在 Zeroshot 冷启动数据集上的脚本
用法: python src/analysis/cold_start/run_all_zeroshot.py
      python src/analysis/cold_start/run_all_zeroshot.py --models dnn_mm lmf --data_dir /path/to/zeroshot
      python src/analysis/cold_start/run_all_zeroshot.py --use_local_data 1
      python src/analysis/cold_start/run_all_zeroshot.py --gpus 0 1 2 3
"""
import os
import sys
import argparse
import subprocess
import time
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

ALL_MODELS = [
    "qarm",
    "mcca",
]

ALL_DATASETS = ["antm2c"]

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "Trainers_fenxi.py")

# 每个子进程的 CPU 线程数上限，控制 CPU 占用率不超过 30%
CPU_THREADS_PER_PROCESS = "4"


def setup_summary_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_all_zeroshot_{timestamp}.log")

    logger = logging.getLogger("run_all_zeroshot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_one(model: str, dataset: str, use_local_data: int, data_dir: str,
            python_exe: str, gpu_id: int) -> dict:
    env = os.environ.copy()
    # 限制 CPU 线程数，控制 CPU 占用率
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = CPU_THREADS_PER_PROCESS
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        python_exe, SCRIPT_PATH,
        "--model_name", model,
        "--dataset_name", dataset,
        "--use_local_data", str(use_local_data),
        "--data_dir", data_dir,
        "--cuda", "0",  # CUDA_VISIBLE_DEVICES 已映射，使用设备 0
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env)
    elapsed = time.time() - start
    return {
        "model": model,
        "dataset": dataset,
        "gpu_id": gpu_id,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description="批量运行所有模型 - Zeroshot 冷启动（多GPU并行）")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="要运行的模型列表，默认全部")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help="要运行的数据集列表，默认全部")
    parser.add_argument("--use_local_data", type=int, default=0,
                        help="是否使用本地数据路径（0/1）")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Zeroshot 冷启动数据目录")
    parser.add_argument("--log_dir", type=str, default="experiments/log_zeroshot",
                        help="汇总日志保存目录")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python 解释器路径，默认使用当前环境")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3],
                        help="可用 GPU 列表，默认 0 1 2 3")
    args = parser.parse_args()

    logger = setup_summary_logger(args.log_dir)

    # 构建所有任务列表
    tasks = []
    for model in args.models:
        for dataset in args.datasets:
            tasks.append((model, dataset))

    total = len(tasks)
    num_gpus = len(args.gpus)
    logger.info(f"[Zeroshot 冷启动] 数据目录={args.data_dir}")
    logger.info(f"共 {total} 个任务：{len(args.models)} 个模型 × {len(args.datasets)} 个数据集")
    logger.info(f"模型: {args.models}")
    logger.info(f"数据集: {args.datasets}")
    logger.info(f"使用 GPU: {args.gpus}，并行度={num_gpus}，每进程CPU线程={CPU_THREADS_PER_PROCESS}")
    logger.info("=" * 60)

    results = []
    finished = 0

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        future_to_task = {}
        for i, (model, dataset) in enumerate(tasks):
            gpu_id = args.gpus[i % num_gpus]
            future = executor.submit(
                run_one, model, dataset, args.use_local_data,
                args.data_dir, args.python, gpu_id,
            )
            future_to_task[future] = (model, dataset, gpu_id, i + 1)

        for future in as_completed(future_to_task):
            model, dataset, gpu_id, idx = future_to_task[future]
            res = future.result()
            results.append(res)
            finished += 1

            status = "SUCCESS" if res["returncode"] == 0 else f"FAILED (code={res['returncode']})"
            logger.info(f"[{finished}/{total}] 完成 model={model} dataset={dataset} "
                        f"gpu={gpu_id} {status} 耗时 {res['elapsed']:.1f}s")

            if res["returncode"] != 0:
                logger.error(f"  stderr: {res['stderr'][-1000:].strip()}")
            else:
                last_lines = res["stdout"].strip().splitlines()
                for line in reversed(last_lines):
                    if "test_auc" in line or "test_loss" in line:
                        logger.info(f"  结果: {line.strip()}")
                        break

    # 打印汇总表格
    results.sort(key=lambda r: (r["model"], r["dataset"]))
    logger.info("")
    logger.info("=" * 60)
    logger.info("汇总结果 (Zeroshot 冷启动)")
    logger.info("=" * 60)
    header = f"{'模型':<14} {'数据集':<12} {'GPU':<5} {'状态':<10} {'耗时(s)':>8}"
    logger.info(header)
    logger.info("-" * 55)
    failed = []
    for r in results:
        status = "OK" if r["returncode"] == 0 else "FAIL"
        logger.info(f"{r['model']:<14} {r['dataset']:<12} {r['gpu_id']:<5} {status:<10} {r['elapsed']:>8.1f}")
        if r["returncode"] != 0:
            failed.append(f"{r['model']} + {r['dataset']}")

    logger.info("")
    logger.info(f"成功: {total - len(failed)}/{total}")
    if failed:
        logger.warning(f"失败任务: {failed}")


if __name__ == "__main__":
    main()
