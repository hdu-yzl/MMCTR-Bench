"""
MMCTR 模型超参数调优脚本

功能:
- 自动调优 CTR/MMCTR 模型的超参数（学习率、隐藏层维度等）
- 追踪多个指标：AUC、Loss
- 保存 AUC 最优的参数
- 生成调优报告和可视化结果
"""

import os
import sys
import math
import yaml
import random
import logging
import itertools
import psutil
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple, Any

# 设置线程数（限制为总逻辑核心数的 30%，至少 1 个）
_total_cpu = psutil.cpu_count(logical=True) or 1
_max_cpu = max(1, math.floor(_total_cpu * 0.30))
num_threads = str(_max_cpu)
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

# 通过 CPU 亲和性将进程可用核心数硬限制为 30%
try:
    _proc = psutil.Process(os.getpid())
    _all_cpus = list(range(_total_cpu))
    _proc.cpu_affinity(_all_cpus[:_max_cpu])
except (AttributeError, NotImplementedError, psutil.AccessDenied):
    pass  # 部分系统/权限下不支持亲和性设置，忽略

import argparse
import torch
torch.set_num_threads(_max_cpu)         # 限制 PyTorch 线程数
torch.set_num_interop_threads(_max_cpu) # 限制跨算子并行线程数
from mmctr.utils import helper
from mmctr.utils.tuning_protocol import evaluate_for_selection, is_better


def dict_to_list_of_tuples(d):
    """将嵌套字典转为 (key_path, value_list) 的列表，用于展开笛卡尔积"""
    result = []
    for k, v in d.items():
        if isinstance(v, dict):
            sub = dict_to_list_of_tuples(v)
            for sub_k, sub_v in sub:
                result.append((f"{k}.{sub_k}", sub_v))
        else:
            result.append((k, v))
    return result


def set_nested_value(d, key_path, value):
    """根据 'a.b.c' 形式的 key_path 设置嵌套字典中的值"""
    keys = key_path.split('.')
    current = d
    for k in keys[:-1]:
        current = current[k]
    current[keys[-1]] = value


def generate_combinations(param_dict):
    """生成所有参数组合（笛卡尔积）"""
    flat_params = dict_to_list_of_tuples(param_dict)
    keys, values = zip(*flat_params)
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def setup_logger(model_name: str, dataset_name: str, log_dir: str) -> logging.Logger:
    """设置日志记录器（与 Pre_Tuner 统一格式）"""
    logger = logging.getLogger(f'{model_name}_tuner')
    logger.setLevel(logging.INFO)

    # 清除旧的 handlers
    logger.handlers.clear()

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # 文件 handler
    log_file = log_dir_path / f'{model_name}_{dataset_name}_tuner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def print_summary(results: List[Dict], best_result: Dict):
    """打印调优总结"""
    print(f"\n{'='*80}")
    print(f"调优完成总结")
    print(f"{'='*80}\n")

    print(f"总试验数: {len(results)}")

    if best_result:
        print(f"\n🏆 最优结果 (试验 {best_result['best_trial']}):")
        print(f"   Validation AUC:  {best_result['val_auc']:.6f}")
        print(f"   Validation Loss: {best_result['val_loss']:.6f}")
        print(f"   参数:   {best_result['params']}")
    else:
        print("\n⚠️ 无有效的调优结果")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="MMCTR-Tuner")
    parser.add_argument("--dataset_name", type=str, default="antm2c")
    parser.add_argument("--model_name", type=str, default="dnn")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--use_local_data", type=int, default=0)
    parser.add_argument("--max_trials", type=int, default=30)
    args = parser.parse_args()

    model_name = args.model_name.lower()
    dataset_name = args.dataset_name.lower()

    # 加载原始配置
    base_model_config = helper.load_yaml('config/model.yaml')[model_name]
    base_model_config['model_name'] = model_name

    if base_model_config.get("seq_modeling", False):
        data_config_path = 'config/local_seq_data.yaml' if args.use_local_data else 'config/seq_data.yaml'
    else:
        data_config_path = 'config/local_data.yaml' if args.use_local_data else 'config/data.yaml'
    data_config = helper.load_yaml(data_config_path)
    train_config = helper.load_yaml('config/train.yaml')
    train_config['cuda'] = args.cuda

    tuner_config = helper.load_yaml('config/Tuner.yaml')
    train_tune = tuner_config.get('train', {})
    model_tune = tuner_config.get('model', {}).get(model_name, {})

    all_tune_params = {}
    if train_tune:
        all_tune_params['train'] = train_tune
    if model_tune:
        all_tune_params['model'] = model_tune

    # 初始化日志
    log_dir = train_config.get('log_dir', 'experiments/logs')
    logger = setup_logger(model_name, dataset_name, log_dir)
    logger.info(f"MMCTR Tuner 初始化完成")
    logger.info(f"模型: {model_name}, 数据集: {dataset_name}")
    logger.info(f"CPU 限制: {_max_cpu}/{_total_cpu} 逻辑核 (~30%), 线程数: {num_threads}")

    if not all_tune_params:
        logger.error("未在 Tuner.yaml 中找到可调参数，退出")
        return

    combinations = list(generate_combinations(all_tune_params))
    if not combinations:
        logger.error("未生成任何超参数组合，请检查 Tuner.yaml 配置")
        return

    random.shuffle(combinations)
    max_trials = min(args.max_trials, len(combinations))

    best_metrics = None
    best_params = None
    best_trial = -1
    results = []

    logger.info(f"总组合数: {len(combinations)}, 将运行 {max_trials} 个试验")
    print(f"\n{'='*80}")
    print(f"开始调优 {model_name.upper()} 模型")
    print(f"数据集: {dataset_name}")
    print(f"总试验数: {max_trials}")
    print(f"{'='*80}\n")

    for trial_idx, combo in enumerate(combinations[:max_trials]):
        print(f"\n{'─'*80}")
        print(f"试验 {trial_idx + 1}/{max_trials}")
        print(f"参数: {combo}")
        print(f"{'─'*80}")

        logger.info(f"试验 {trial_idx + 1}: {combo}")

        current_train_config = deepcopy(train_config)
        current_model_config = deepcopy(base_model_config)

        for key, val in combo.items():
            if key.startswith('train.'):
                real_key = key[len('train.'):]
                set_nested_value(current_train_config, real_key, val)
            elif key.startswith('model.'):
                real_key = key[len('model.'):]
                set_nested_value(current_model_config, real_key, val)

        # 初始化数据加载器和模型
        dataloader = helper.getDataLoader(
            dataset_name,
            data_config[dataset_name],
            current_train_config["batch_size"]
        )
        model_logger = helper.get_logger(f"{model_name}_tune", current_train_config['log_dir'])
        model = helper.getModel(
            model_name,
            current_model_config,
            current_train_config,
            data_config[dataset_name],
            model_logger
        )

        try:
            model.fit(dataloader)
            val_metrics = evaluate_for_selection(model, dataloader)

            # 记录结果
            result = {
                'trial': trial_idx + 1,
                'params': combo,
                'val_auc': val_metrics.auc,
                'val_loss': val_metrics.loss,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

            # 打印结果
            print(f"\n✓ 试验 {trial_idx + 1} 完成:")
            print(f"  Validation AUC:  {val_metrics.auc:.6f}")
            print(f"  Validation Loss: {val_metrics.loss:.6f}")

            logger.info(
                f"试验 {trial_idx + 1} 完成 - Validation AUC: {val_metrics.auc:.6f}, "
                f"Validation Loss: {val_metrics.loss:.6f}"
            )

            if is_better(val_metrics, best_metrics):
                best_metrics = val_metrics
                best_params = combo
                best_trial = trial_idx + 1
                print(
                    f"  🎉 新最优 Validation AUC: "
                    f"{best_metrics.auc:.6f} (试验 {best_trial})"
                )
                logger.info(
                    f"新最优 Validation AUC: "
                    f"{best_metrics.auc:.6f} (试验 {best_trial})"
                )

        except Exception as e:
            logger.error(f"试验 {trial_idx + 1} 失败: {e}")
            print(f"\n✗ 试验 {trial_idx + 1} 失败: {e}")
            continue

    # 保存结果
    best_result = None
    if best_params is not None and best_metrics is not None:
        output_file = "config/best_params.yaml"
        best_result = {
            'model': model_name,
            'dataset': dataset_name,
            'best_trial': best_trial,
            'selection_split': 'val',
            'val_auc': best_metrics.auc,
            'val_loss': best_metrics.loss,
            'params': best_params
        }
        with open(output_file, 'a', encoding='utf-8') as f:
            yaml.dump(best_result, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"最优参数已保存到: {output_file}")
        print(f"\n✅ 最优参数已保存到 {output_file}")

    # 打印总结
    print_summary(results, best_result)


if __name__ == "__main__":
    """
    使用示例:

    # DNN 模型调优
    python src/scripts/Tuner.py --model_name dnn --dataset_name antm2c --max_trials 20

    # DeepFM 模型调优
    python src/scripts/Tuner.py --model_name deepfm --dataset_name antm2c --max_trials 20

    # 使用 GPU 1
    python src/scripts/Tuner.py --model_name dnn --cuda 1 --max_trials 30
    """
    main()
