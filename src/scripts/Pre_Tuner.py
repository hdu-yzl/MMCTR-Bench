"""
RQ 和 PSRQ 预训练模型超参数调优脚本

功能:
- 自动调优 RQ/PSRQ 的超参数（n_levels, codebook_size, 学习率等）
- 追踪多个指标：重建损失、量化损失、坍缺率
- 保存损失最小和坍缺率最小的最优参数
- 生成调优报告和可视化结果
"""

import os
import sys
import math
import yaml
import json
import random
import logging
import numpy as np
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
from utils import helper
from models.pre_models.RQ import ResidualQuantizer
from models.pre_models.PSRQ import PSRQ_Premodel


class QuantizationMetrics:
    """量化模型的指标追踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.recon_loss = 0.0
        self.quant_loss = 0.0
        self.total_loss = 0.0
        self.collapse_rate = {}  # 每层的坍缺率
        self.codebook_usage = {}  # 每层的码本使用率
        self.num_batches = 0
    
    def compute_collapse_rate(self, model) -> Dict[str, float]:
        """
        计算码本坍缺率（未被使用的码本向量比例）
        
        对于 VectorQuantizer，坍缺率 = 1 - (使用的码本数 / 总码本数)
        """
        collapse_rates = {}
        
        if isinstance(model, PSRQ_Premodel):
            # PSRQ 多层量化
            for lvl, vq_layer in enumerate(model.joint_psrq.psrq.vq_layers):
                n_e = vq_layer.n_e
                embedding = vq_layer.embedding.weight.data
                
                # 计算码本向量的 L2 范数，识别零向量或接近零的向量
                norms = torch.norm(embedding, dim=1)
                # 假设规范化后的向量范数应该接近 1，低于阈值的认为是未使用
                used_codebook = (norms > 0.5).sum().item()
                collapse_rate = 1.0 - (used_codebook / n_e)
                collapse_rates[f'level_{lvl}'] = collapse_rate
        
        elif isinstance(model, ResidualQuantizer):
            # RQ 多层量化
            if hasattr(model, 'codebooks') and model.codebooks:
                for lvl, codebook in enumerate(model.codebooks):
                    n_e = model.codebook_size
                    norms = np.linalg.norm(codebook, axis=1)
                    used_codebook = (norms > 0.5).sum()
                    collapse_rate = 1.0 - float(used_codebook) / n_e
                    collapse_rates[f'level_{lvl}'] = collapse_rate
        
        self.collapse_rate = collapse_rates
        mean_collapse = np.mean(list(collapse_rates.values()))
        return {'mean': mean_collapse, **collapse_rates}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'recon_loss': float(self.recon_loss),
            'quant_loss': float(self.quant_loss),
            'total_loss': float(self.total_loss),
            'collapse_rate': {k: float(v) for k, v in self.collapse_rate.items()},
            'num_batches': self.num_batches
        }


class RQPSRQTuner:
    """RQ 和 PSRQ 模型调优器"""
    
    def __init__(self, 
                 model_type: str = 'psrq',
                 dataset_name: str = 'antm2c',
                 modality: str = 'text',
                 use_local_data: int = 0,
                 cuda: int = 0,
                 max_trials: int = 20,
                 verbose: int = 1):
        """
        Args:
            model_type: 'rq' 或 'psrq'
            dataset_name: 数据集名称
            modality: 模态（text 或 image）
            use_local_data: 是否使用本地数据
            cuda: GPU 编号
            max_trials: 最大试验次数
            verbose: 打印详度
        """
        self.model_type = model_type.lower()
        self.dataset_name = dataset_name.lower()
        self.modality = modality.lower()
        self.use_local_data = use_local_data
        self.cuda = cuda
        self.max_trials = max_trials
        self.verbose = verbose
        
        # 加载配置
        self.model_config = helper.load_yaml('config/model.yaml')
        self.data_config = helper.load_yaml('config/seq_data.yaml')
        self.train_config = helper.load_yaml('config/train.yaml')
        self.tuner_config = helper.load_yaml('config/Tuner.yaml')
        
        # 设置设备
        self.train_config['cuda'] = cuda
        
        # 获取数据加载器
        self.dataloader = helper.getDataLoader(
            self.dataset_name,
            self.data_config[self.dataset_name],
            self.train_config["batch_size"]
        )
        
        # 初始化日志
        log_dir = Path(self.train_config.get('log_dir', 'experiments/logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger(log_dir)
        
        # 调优结果存储
        self.results = []
        self.best_loss_result = None
        self.best_collapse_result = None
        self.best_combined_result = None
        
        self.logger.info(f"RQPSRQTuner 初始化完成")
        self.logger.info(f"模型类型: {self.model_type}, 数据集: {self.dataset_name}, 模态: {self.modality}")
        self.logger.info(f"CPU 限制: {_max_cpu}/{_total_cpu} 逻辑核 (~30%), 线程数: {num_threads}")
    
    def _setup_logger(self, log_dir: Path) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f'{self.model_type}_tuner')
        logger.setLevel(logging.INFO)
        
        # 清除旧的 handlers
        logger.handlers.clear()
        
        # 文件 handler
        log_file = log_dir / f'{self.model_type}_{self.dataset_name}_tuner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
    
    def _dict_to_list_of_tuples(self, d: Dict) -> List[Tuple[str, List]]:
        """将嵌套字典转为 (key_path, value_list) 的列表，用于展开笛卡尔积"""
        result = []
        for k, v in d.items():
            if isinstance(v, dict):
                sub = self._dict_to_list_of_tuples(v)
                for sub_k, sub_v in sub:
                    result.append((f"{k}.{sub_k}", sub_v))
            elif isinstance(v, list):
                result.append((k, v))
            else:
                result.append((k, [v]))
        return result
    
    def _set_nested_value(self, d: Dict, key_path: str, value: Any):
        """根据 'a.b.c' 形式的 key_path 设置嵌套字典中的值"""
        keys = key_path.split('.')
        current = d
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """从 Tuner.yaml 生成超参数组合（支持 train 和 model 参数）"""
        tuner_config = self.tuner_config
        
        # 获取 train 和 model 部分的调参配置
        all_tune_params = {}
        
        train_tune = tuner_config.get('train', {})
        if train_tune:
            all_tune_params['train'] = train_tune
        
        model_tune = tuner_config.get('model', {}).get(self.model_type, {})
        if model_tune:
            all_tune_params['model'] = model_tune
        
        if not all_tune_params:
            self.logger.warning(f"未在 Tuner.yaml 中找到 {self.model_type} 的调参配置")
            return []
        
        # 展开嵌套字典为 (key_path, value_list) 对
        flat_params = self._dict_to_list_of_tuples(all_tune_params)
        
        if not flat_params:
            return []
        
        # 生成笛卡尔积
        keys, values_lists = zip(*flat_params)
        combinations = []
        for combo in itertools.product(*values_lists):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _train_rq(self, mm_modals: Dict[str, np.ndarray], params: Dict[str, Any]) -> Tuple[QuantizationMetrics, Dict[str, Any], Dict[str, float]]:
        """
        训练 RQ 模型（对所有模态分别量化，求平均指标）
        
        Args:
            mm_modals: 所有模态数据的字典，如 {'text': np.ndarray, 'image': np.ndarray, ...}
            params: 超参数组合
        
        Returns:
            metrics: 所有模态的平均性能指标
            models: 每个模态对应的训练后模型字典
            collapse_rates: 平均坍缺率
        """
        epsilon = 1e-8
        
        # 创建配置副本
        train_config = deepcopy(self.train_config)
        model_config = deepcopy(self.model_config[self.model_type])
        
        # 应用调参参数
        for key, val in params.items():
            if key.startswith('train.'):
                real_key = key[len('train.'):]
                self._set_nested_value(train_config, real_key, val)
            elif key.startswith('model.'):
                real_key = key[len('model.'):]
                self._set_nested_value(model_config, real_key, val)
            else:
                # 默认当作 model 参数
                self._set_nested_value(model_config, key, val)
        
        # 对每个模态分别训练 RQ 并收集指标
        all_recon_losses = []
        all_collapse_means = []
        all_collapse_details = {}  # 每个模态的详细坍缺率
        models = {}
        
        for modality_name, mm_modal in mm_modals.items():
            # 数据归一化
            mm_modal = mm_modal / np.maximum(np.linalg.norm(mm_modal, axis=1, keepdims=True), epsilon)
            
            # 创建和训练模型
            rq_model = ResidualQuantizer(
                deepcopy(model_config),
                deepcopy(train_config),
                self.data_config[self.dataset_name]
            )
            
            rq_model.fit(mm_modal, verbose=False)
            models[modality_name] = rq_model
            
            # 获取重建损失
            indices, reconstructed = rq_model.encode(mm_modal)
            recon_loss = float(np.mean((mm_modal - reconstructed) ** 2))
            all_recon_losses.append(recon_loss)
            
            # 计算该模态的坍缺率
            modal_metrics = QuantizationMetrics()
            with torch.no_grad():
                modal_collapse = modal_metrics.compute_collapse_rate(rq_model)
            all_collapse_means.append(modal_collapse['mean'])
            all_collapse_details[modality_name] = modal_collapse
            
            self.logger.info(f"  模态 [{modality_name}] - 重建损失: {recon_loss:.6f}, 坍缺率: {modal_collapse['mean']:.4f}")
        
        # 计算所有模态的平均指标
        metrics = QuantizationMetrics()
        metrics.recon_loss = float(np.mean(all_recon_losses))
        metrics.total_loss = metrics.recon_loss
        
        avg_collapse_mean = float(np.mean(all_collapse_means))
        collapse_rates = {
            'mean': avg_collapse_mean,
            **{f'{mod}_{k}': v for mod, detail in all_collapse_details.items() for k, v in detail.items()}
        }
        
        return metrics, models, collapse_rates
    
    def _train_psrq(self, dataloader, params: Dict[str, Any]) -> Tuple[QuantizationMetrics, Any]:
        """
        训练 PSRQ 模型
        
        Returns:
            metrics: 性能指标
            model: 训练后的模型
        """
        # 创建配置副本
        train_config = deepcopy(self.train_config)
        model_config = deepcopy(self.model_config[self.model_type])
        
        # 应用调参参数
        for key, val in params.items():
            if key.startswith('train.'):
                real_key = key[len('train.'):]
                self._set_nested_value(train_config, real_key, val)
            elif key.startswith('model.'):
                real_key = key[len('model.'):]
                self._set_nested_value(model_config, real_key, val)
            else:
                # 默认当作 model 参数
                self._set_nested_value(model_config, key, val)
        
        model_config['model_name'] = self.model_type
        
        # 创建模型
        psrq_model = PSRQ_Premodel(
            model_config,
            train_config,
            self.data_config[self.dataset_name]
        )
        
        # 训练
        psrq_model.fit(dataloader)
        
        # 计算指标
        metrics = QuantizationMetrics()
        
        # 获取多模态数据
        mm_modals = dataloader.get_multi_modal()
        
        # 计算每个模态的损失
        total_recon_loss = 0.0
        total_quant_loss = 0.0
        modality_count = 0
        
        with torch.no_grad():
            for modality_name, mm_modal in mm_modals.items():
                mm_modal_tensor = torch.from_numpy(mm_modal).to(psrq_model.device).float()
                
                # 获取索引
                indices = psrq_model.mm_psrq[modality_name].get_indices(mm_modal_tensor)
                
                # 重建
                x_e = psrq_model.mm_psrq[modality_name].encoder(mm_modal_tensor)
                x_q, quant_loss, _ = psrq_model.mm_psrq[modality_name].psrq(x_e)
                reconstructed = psrq_model.mm_psrq[modality_name].decoder(x_q)
                
                recon_loss = torch.nn.functional.mse_loss(reconstructed, mm_modal_tensor)
                total_recon_loss += recon_loss.item()
                total_quant_loss += quant_loss.item()
                modality_count += 1
        
        metrics.recon_loss = total_recon_loss / modality_count if modality_count > 0 else 0.0
        metrics.quant_loss = total_quant_loss / modality_count if modality_count > 0 else 0.0
        metrics.total_loss = metrics.recon_loss + metrics.quant_loss
        
        # 计算坍缺率
        with torch.no_grad():
            collapse_rates = metrics.compute_collapse_rate(psrq_model)
        
        return metrics, psrq_model, collapse_rates
    
    def run(self):
        """执行调优"""
        # 生成超参数组合
        combinations = self._generate_param_combinations()
        
        if not combinations:
            self.logger.error("未生成任何超参数组合，请检查 Tuner.yaml 配置")
            return
        
        # 随机打乱以增加多样性
        random.shuffle(combinations)
        max_trials = min(self.max_trials, len(combinations))
        
        self.logger.info(f"总组合数: {len(combinations)}, 将运行 {max_trials} 个试验")
        print(f"\n{'='*80}")
        print(f"开始调优 {self.model_type.upper()} 模型")
        print(f"数据集: {self.dataset_name}, 模态: {self.modality}")
        print(f"总试验数: {max_trials}")
        print(f"{'='*80}\n")
        
        # 获取数据
        if self.model_type == 'rq':
            mm_modals = self.dataloader.get_multi_modal()
            
            if not mm_modals:
                self.logger.error(f"未找到任何模态数据")
                return
            
            modality_names = list(mm_modals.keys())
            self.logger.info(f"RQ 将对所有模态进行量化: {modality_names}")
            print(f"\n📦 数据集包含 {len(modality_names)} 个模态: {modality_names}")
        
        # 运行试验
        for trial_idx, params in enumerate(combinations[:max_trials]):
            print(f"\n{'─'*80}")
            print(f"试验 {trial_idx + 1}/{max_trials}")
            print(f"参数: {params}")
            print(f"{'─'*80}")
            
            self.logger.info(f"试验 {trial_idx + 1}: {params}")
            
            try:
                if self.model_type == 'rq':
                    metrics, model, collapse_rates = self._train_rq(mm_modals, params)
                elif self.model_type == 'psrq':
                    metrics, model, collapse_rates = self._train_psrq(self.dataloader, params)
                else:
                    raise ValueError(f"未知的模型类型: {self.model_type}")
                
                # 记录结果
                result = {
                    'trial': trial_idx + 1,
                    'params': params,
                    'metrics': metrics.to_dict(),
                    'collapse_rates': {k: float(v) for k, v in collapse_rates.items()},
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                
                # 打印结果
                print(f"\n✓ 试验 {trial_idx + 1} 完成:")
                print(f"  重建损失: {metrics.recon_loss:.6f}")
                print(f"  量化损失: {metrics.quant_loss:.6f}")
                print(f"  总损失:   {metrics.total_loss:.6f}")
                print(f"  平均坍缺率: {collapse_rates['mean']:.4f}")
                for level, cr in collapse_rates.items():
                    if level != 'mean':
                        print(f"    {level}: {cr:.4f}")
                
                # 更新最优结果
                self._update_best_results(result, metrics, collapse_rates)
                
                self.logger.info(f"试验 {trial_idx + 1} 完成 - "
                               f"损失: {metrics.total_loss:.6f}, "
                               f"坍缺率: {collapse_rates['mean']:.4f}")
                
            except Exception as e:
                self.logger.error(f"试验 {trial_idx + 1} 失败: {e}")
                print(f"\n✗ 试验 {trial_idx + 1} 失败: {e}")
                continue
        
        # 保存结果
        self._save_results()
        
        # 打印总结
        self._print_summary()
    
    def _update_best_results(self, result: Dict, metrics: QuantizationMetrics, collapse_rates: Dict):
        """更新最优结果"""
        # 更新最小损失结果
        if self.best_loss_result is None or metrics.total_loss < self.best_loss_result['metrics']['total_loss']:
            self.best_loss_result = result
        
        # 更新最小坍缺率结果
        if self.best_collapse_result is None or collapse_rates['mean'] < self.best_collapse_result['collapse_rates']['mean']:
            self.best_collapse_result = result
        
        # 更新综合最优结果（加权）
        # 权重: 损失 60%, 坍缺率 40%
        combined_score = 0.6 * (metrics.total_loss / 10.0) + 0.4 * collapse_rates['mean']
        if self.best_combined_result is None:
            self.best_combined_result = (result, combined_score)
        elif combined_score < self.best_combined_result[1]:
            self.best_combined_result = (result, combined_score)
    
    def _save_results(self):
        """保存调优结果（与 Tuner.py 格式统一）"""
        # 保存到 config/best_params.yaml（追加模式）
        output_file = "config/best_params.yaml"
        
        if self.best_combined_result:
            result = self.best_combined_result[0]
            best_params_entry = {
                'model': self.model_type,
                'dataset': self.dataset_name,
                'modality': self.modality,
                'best_trial': result['trial'],
                'total_loss': float(result['metrics']['total_loss']),
                'recon_loss': float(result['metrics']['recon_loss']),
                'mean_collapse_rate': float(result['collapse_rates']['mean']),
                'params': result['params']
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                yaml.dump(best_params_entry, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            self.logger.info(f"最优参数已保存到: {output_file}")
            print(f"\n✅ 最优参数已保存到 {output_file}")
        else:
            self.logger.warning("无有效的调优结果")
            print("\n⚠️ 无有效的调优结果")
    
    def _print_summary(self):
        """打印调优总结"""
        print(f"\n{'='*80}")
        print(f"调优完成总结")
        print(f"{'='*80}\n")
        
        print(f"总试验数: {len(self.results)}")
        
        if self.best_combined_result:
            result = self.best_combined_result[0]
            print(f"\n🏆 综合最优结果 (试验 {result['trial']}):")
            print(f"   总损失: {result['metrics']['total_loss']:.6f}")
            print(f"   重建损失: {result['metrics']['recon_loss']:.6f}")
            print(f"   量化损失: {result['metrics']['quant_loss']:.6f}")
            print(f"   平均坍缺率: {result['collapse_rates']['mean']:.4f}")
            print(f"   综合得分: {self.best_combined_result[1]:.6f}")
            print(f"   参数: {result['params']}")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="RQ/PSRQ 模型调优脚本")
    parser.add_argument("--model_type", type=str, choices=['rq', 'psrq'], 
                       default='psrq', help="模型类型")
    parser.add_argument("--dataset_name", type=str, default="antm2c", 
                       help="数据集名称")
    parser.add_argument("--modality", type=str, choices=['text', 'image','audio'], 
                       default='text', help="模态类型（仅 RQ 使用）")
    parser.add_argument("--cuda", type=int, default=0, 
                       help="GPU 编号")
    parser.add_argument("--max_trials", type=int, default=20, 
                       help="最大试验次数")
    parser.add_argument("--verbose", type=int, default=1, 
                       help="打印详度")
    
    args = parser.parse_args()
    
    # 创建调优器
    tuner = RQPSRQTuner(
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        modality=args.modality,
        cuda=args.cuda,
        max_trials=args.max_trials,
        verbose=args.verbose
    )
    
    # 运行调优
    tuner.run()


if __name__ == "__main__":
    """
    使用示例:
    
    # PSRQ 调优
    python src/scripts/RQ_PSRQ_Tuner.py --model_type psrq --dataset_name antm2c --max_trials 20
    
    # RQ 调优
    python src/scripts/RQ_PSRQ_Tuner.py --model_type rq --dataset_name antm2c --modality text --max_trials 20
    
    # 使用 GPU 1
    python src/scripts/RQ_PSRQ_Tuner.py --model_type psrq --cuda 1 --max_trials 30
    """
    main()
