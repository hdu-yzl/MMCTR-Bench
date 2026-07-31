# 模态对齐分析实验

本目录包含对所有多模态CTR模型进行模态对齐分析的完整实验框架。

## 概述

该实验框架测试5种不同的模态对齐方法对11个多模态CTR模型性能的影响：

### 支持的模型
1. `dnn_mm` - 基础多模态DNN
2. `lmf` - Low-rank Multimodal Fusion
3. `marn` - Modality-Aware Representation Network
4. `mtfn` - Multimodal Tensor Fusion Network
5. `naml` - Neural Attentive Multi-view Learning
6. `make` - Multimodal Attention Knowledge Embedding
7. `em3` - Enhanced Multimodal Modeling
8. `simcen` - Similarity-based Contrastive Enhancement Network
9. `gmmf` - Gated Multimodal Fusion
10. `dmf` - Dynamic Multimodal Fusion
11. `diff_msin` - Diffusion-enhanced Multimodal Sequential Interest Network

### 对齐方法
1. `none` - 无对齐（基线）
2. `kl` - KL散度对齐
3. `contrastive` - 对比学习对齐
4. `cosine` - 余弦相似度对齐
5. `mmd` - 最大均值差异对齐
6. `adversarial` - 对抗对齐

### Lambda权重
测试6个不同的权重值：`0.0, 0.1, 0.2, 0.3, 0.4, 0.5`

## 文件说明

- `alignment_trainer.py` - 单个实验的训练脚本
- `run_alignment_analysis.py` - 批量运行所有实验的主脚本（支持多GPU并行）
- `summarize_results.py` - 结果汇总和分析脚本
- `README.md` - 本文档

## 使用方法

### 1. 运行单个实验

```bash
python src/analysis/alignment_analysis/alignment_trainer.py \
    --model_name dnn_mm \
    --dataset_name antm2c \
    --alignment_method kl \
    --lambda_weight 0.1 \
    --cuda 0
```

参数说明：
- `--model_name`: 模型名称
- `--dataset_name`: 数据集名称（默认：antm2c）
- `--alignment_method`: 对齐方法（none/kl/contrastive/cosine/mmd/adversarial）
- `--lambda_weight`: 对齐损失权重（0.0-0.5）
- `--cuda`: GPU设备ID
- `--use_local_data`: 是否使用本地数据（0/1）

### 2. 批量运行所有实验（推荐）

在4个GPU上并行运行所有模型、所有对齐方法、所有lambda权重的组合：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 2 3 \
    --datasets antm2c \
    --models dnn_mm lmf diff_msin marn mtfn naml make em3 simcen gmmf dmf \
    --alignments none kl contrastive cosine mmd adversarial \
    --lambda_weights 0.0 0.1 0.2 0.3 0.4 0.5
```

参数说明：
- `--gpu_ids`: 可用的GPU ID列表（默认：0 1 2 3）
- `--datasets`: 数据集列表（默认：antm2c）
- `--models`: 模型列表（默认：所有11个模型）
- `--alignments`: 对齐方法列表（默认：所有6种方法）
- `--lambda_weights`: lambda权重列表（默认：0.0到0.5的6个值）
- `--use_local_data`: 是否使用本地数据（0/1）
- `--log_dir`: 日志保存目录（默认：experiments/logs）
- `--output_csv`: 结果CSV保存路径（默认：experiments/results/alignment_analysis.csv）

### 3. 结果汇总

实验完成后，使用汇总脚本生成详细报告：

```bash
python src/analysis/alignment_analysis/summarize_results.py \
    --csv experiments/results/alignment_analysis.csv \
    --output experiments/results/alignment_summary.txt
```

## 实验设计

### 对齐损失的作用

对齐损失旨在将非ID模态（如text、image）的特征表示与ID模态对齐，使得：
1. 不同模态的语义信息更加一致
2. 提高模型对模态缺失的鲁棒性
3. 改善多模态融合的效果

### 实验流程

1. **训练阶段**：在原有的CTR损失基础上，添加对齐损失
   ```
   总损失 = CTR损失 + 辅助损失 + λ × 对齐损失
   ```

2. **评估阶段**：在验证集和测试集上评估AUC和Logloss

3. **对比分析**：
   - 不同对齐方法的效果对比
   - 不同lambda权重的影响
   - 不同模型对对齐方法的敏感度

## 预期结果

实验将生成以下输出：

1. **CSV结果文件** (`experiments/results/alignment_analysis.csv`)
   - 包含所有实验的详细结果
   - 字段：model, dataset, alignment, lambda, cuda, val_auc, test_auc, test_loss, elapsed, status

2. **汇总报告** (`experiments/results/alignment_summary.txt`)
   - 每个模型的最佳配置
   - 对齐方法效果对比
   - Lambda权重影响分析
   - 模型×对齐方法矩阵

3. **日志文件** (`experiments/logs/`)
   - 主日志：`alignment_analysis_YYYYMMDD_HHMMSS.log`
   - 每个实验的独立日志

## 注意事项

1. **计算资源**：
   - 总实验数 = 模型数 × 对齐方法数 × lambda权重数
   - 默认配置：11 × 6 × 6 = 396个实验（none方法只运行lambda=0，实际约341个）
   - 建议使用4个GPU并行，预计总时间：数小时到数天（取决于数据集大小）

2. **存储空间**：
   - 每个实验会保存模型checkpoint
   - 确保有足够的磁盘空间（建议至少50GB）

3. **日志管理**：
   - 每个实验有独立的日志文件
   - 日志文件会占用一定空间，可定期清理旧日志

4. **中断恢复**：
   - 如果实验中断，可以通过修改`--models`、`--alignments`或`--lambda_weights`参数来只运行未完成的部分
   - CSV文件支持追加，但建议备份已有结果

## 示例：快速测试

如果想快速测试框架是否正常工作，可以只运行少量实验：

```bash
# 只测试2个模型、2种对齐方法、2个lambda值
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 \
    --models dnn_mm lmf \
    --alignments none kl \
    --lambda_weights 0.0 0.1
```

## 问题排查

1. **CUDA out of memory**：
   - 减少batch_size（在config/train.yaml中修改）
   - 减少并行GPU数量

2. **实验失败**：
   - 检查对应的日志文件
   - 确认模型配置在config/best_param.yaml中存在
   - 确认数据文件路径正确

3. **结果异常**：
   - 检查是否有NaN或Inf值
   - 确认lambda权重设置合理（过大可能导致训练不稳定）

## 引用

如果使用本实验框架，请引用相关的对齐方法论文和MMCTR Benchmark。
