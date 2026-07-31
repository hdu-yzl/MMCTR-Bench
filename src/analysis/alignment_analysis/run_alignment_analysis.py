"""
对齐分析批量运行脚本
在多个GPU上并行运行所有模型、所有对齐方法、所有lambda_weight的组合
"""
import os
import sys
import argparse
import subprocess
import time
import logging
import csv
import torch.multiprocessing as mp
from datetime import datetime
from collections import defaultdict

# 设置线程数
num_threads = "12"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
# 所有模型
ALL_MODELS = [
    "diff_msin",
    "mb",
    "pamd",
    "mmmlp",
    "m3srec",
    "qarm",
    "mcca",
]

# 所有对齐方法
ALIGNMENT_METHODS = ['none', 'kl', 'contrastive', 'cosine', 'mmd', 'adversarial']

# Lambda权重范围：0, 0.1, 0.2, 0.3, 0.4, 0.5
LAMBDA_WEIGHTS = [0.0, 0.1, 0.3, 0.5,0.7]

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "alignment_trainer.py")


def setup_logger(log_dir: str) -> logging.Logger:
    """设置主日志"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"alignment_analysis_{timestamp}.log")
    
    logger = logging.getLogger("alignment_analysis")
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


def run_single_experiment(model: str, dataset: str, alignment: str, 
                         lambda_weight: float, cuda: int, 
                         use_local_data: int, python_exe: str) -> dict:
    """运行单个实验"""
    cmd = [
        python_exe, SCRIPT_PATH,
        "--model_name", model,
        "--dataset_name", dataset,
        "--alignment_method", alignment,
        "--lambda_weight", str(lambda_weight),
        "--use_local_data", str(use_local_data),
        "--cuda", str(cuda),
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, 
                          encoding="utf-8", errors="replace")
    elapsed = time.time() - start
    
    # 解析输出获取结果
    test_auc = None
    test_loss = None
    val_auc = None
    
    if result.returncode == 0:
        lines = result.stdout.strip().splitlines()
        for line in lines:
            if "Test AUC:" in line:
                try:
                    test_auc = float(line.split("Test AUC:")[1].split()[0])
                except:
                    pass
            if "Test Loss:" in line:
                try:
                    test_loss = float(line.split("Test Loss:")[1].split()[0])
                except:
                    pass
            if "Val AUC:" in line and val_auc is None:
                try:
                    val_auc = float(line.split("Val AUC:")[1].split()[0])
                except:
                    pass
    
    return {
        "model": model,
        "dataset": dataset,
        "alignment": alignment,
        "lambda": lambda_weight,
        "cuda": cuda,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "test_loss": test_loss,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def gpu_worker(gpu_id: int, tasks: list, use_local_data: int, 
               python_exe: str, result_queue):
    """GPU工作进程"""
    for model, dataset, alignment, lambda_weight in tasks:
        try:
            result = run_single_experiment(
                model, dataset, alignment, lambda_weight,
                gpu_id, use_local_data, python_exe
            )
            result_queue.put(result)
        except Exception as e:
            print(f"[GPU {gpu_id}] 任务失败: {model}|{alignment}|λ={lambda_weight}: {e}")
            result_queue.put({
                "model": model,
                "dataset": dataset,
                "alignment": alignment,
                "lambda": lambda_weight,
                "cuda": gpu_id,
                "returncode": -1,
                "elapsed": 0,
                "val_auc": None,
                "test_auc": None,
                "test_loss": None,
            })


def save_results_csv(results: list, output_path: str):
    """保存结果到CSV"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['model', 'dataset', 'alignment', 'lambda', 'cuda', 
                     'val_auc', 'test_auc', 'test_loss', 'elapsed', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'model': r['model'],
                'dataset': r['dataset'],
                'alignment': r['alignment'],
                'lambda': f"{r['lambda']:.2f}",
                'cuda': r['cuda'],
                'val_auc': f"{r['val_auc']:.6f}" if r['val_auc'] is not None else "N/A",
                'test_auc': f"{r['test_auc']:.6f}" if r['test_auc'] is not None else "N/A",
                'test_loss': f"{r['test_loss']:.6f}" if r['test_loss'] is not None else "N/A",
                'elapsed': f"{r['elapsed']:.1f}",
                'status': 'OK' if r['returncode'] == 0 else 'FAIL',
            })


def print_summary_table(results: list, logger: logging.Logger):
    """打印汇总表格"""
    # 按模型和对齐方法分组
    grouped = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if r['returncode'] == 0 and r['test_auc'] is not None:
            key = (r['model'], r['alignment'])
            grouped[key][r['lambda']].append(r['test_auc'])
    
    logger.info("\n" + "="*100)
    logger.info("对齐分析结果汇总 (Test AUC)")
    logger.info("="*100)
    
    # 按模型分组打印
    for model in ALL_MODELS:
        logger.info(f"\n模型: {model.upper()}")
        logger.info("-"*100)
        
        # 表头
        header = f"{'对齐方法':<15}"
        for lw in LAMBDA_WEIGHTS:
            header += f"λ={lw:.1f}".center(12)
        logger.info(header)
        logger.info("-"*100)
        
        # 每个对齐方法一行
        for alignment in ALIGNMENT_METHODS:
            key = (model, alignment)
            if key not in grouped:
                continue
            
            row = f"{alignment:<15}"
            for lw in LAMBDA_WEIGHTS:
                if lw in grouped[key] and grouped[key][lw]:
                    auc = grouped[key][lw][0]  # 取第一个结果
                    row += f"{auc:.6f}".center(12)
                else:
                    row += "N/A".center(12)
            logger.info(row)
    
    logger.info("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="对齐分析批量运行")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                       help="要测试的模型列表")
    parser.add_argument("--datasets", nargs="+", default=["antm2c"],
                       help="要测试的数据集列表")
    parser.add_argument("--alignments", nargs="+", default=ALIGNMENT_METHODS,
                       help="要测试的对齐方法列表")
    parser.add_argument("--lambda_weights", nargs="+", type=float, default=LAMBDA_WEIGHTS,
                       help="要测试的lambda权重列表")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1, 2, 3,4 ,5,6,7],
                       help="可用的GPU ID列表")
    parser.add_argument("--use_local_data", type=int, default=0,
                       help="是否使用本地数据")
    parser.add_argument("--log_dir", type=str, default="experiments/logs",
                       help="日志目录")
    parser.add_argument("--output_csv", type=str, 
                       default="experiments/results/alignment_analysis.csv",
                       help="结果CSV保存路径")
    parser.add_argument("--python", type=str, default=sys.executable,
                       help="Python解释器路径")
    
    args = parser.parse_args()
    
    logger = setup_logger(args.log_dir)
    
    # 构建任务列表
    all_tasks = []
    for dataset in args.datasets:
        for model in args.models:
            for alignment in args.alignments:
                for lambda_weight in args.lambda_weights:
                    # none对齐方法只需要运行lambda=0的情况
                    if alignment == 'none' and lambda_weight != 0.0:
                        continue
                    all_tasks.append((model, dataset, alignment, lambda_weight))
    
    num_gpus = len(args.gpu_ids)
    total_tasks = len(all_tasks)
    
    logger.info("="*60)
    logger.info("对齐分析实验启动")
    logger.info(f"  模型数量: {len(args.models)}")
    logger.info(f"  数据集: {args.datasets}")
    logger.info(f"  对齐方法: {args.alignments}")
    logger.info(f"  Lambda权重: {args.lambda_weights}")
    logger.info(f"  GPU列表: {args.gpu_ids}")
    logger.info(f"  总任务数: {total_tasks}")
    logger.info("="*60)
    
    # 将任务分配到各GPU
    gpu_tasks = {gid: [] for gid in args.gpu_ids}
    for i, task in enumerate(all_tasks):
        gid = args.gpu_ids[i % num_gpus]
        gpu_tasks[gid].append(task)
    
    for gid in args.gpu_ids:
        logger.info(f"  GPU {gid}: {len(gpu_tasks[gid])} 个任务")
    
    # 多进程并行执行
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    
    processes = []
    for gid in args.gpu_ids:
        if gpu_tasks[gid]:
            p = mp.Process(
                target=gpu_worker,
                args=(gid, gpu_tasks[gid], args.use_local_data, 
                     args.python, result_queue)
            )
            processes.append(p)
            p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # 保存结果
    save_results_csv(results, args.output_csv)
    logger.info(f"\n结果已保存到: {args.output_csv}")
    
    # 打印汇总
    print_summary_table(results, logger)
    
    # 统计
    success = sum(1 for r in results if r['returncode'] == 0)
    failed = len(results) - success
    
    logger.info(f"\n任务完成统计:")
    logger.info(f"  成功: {success}/{len(results)}")
    logger.info(f"  失败: {failed}/{len(results)}")
    
    if failed > 0:
        logger.info(f"\n失败任务:")
        for r in results:
            if r['returncode'] != 0:
                logger.info(f"  {r['model']} | {r['alignment']} | λ={r['lambda']:.2f}")


if __name__ == "__main__":
    main()
