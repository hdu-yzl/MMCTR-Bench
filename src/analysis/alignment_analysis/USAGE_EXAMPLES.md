# 使用示例

本文档提供对齐分析实验的详细使用示例。

## 快速开始

### 方式1: 使用快速启动脚本（推荐）

**Linux/Mac:**
```bash
# 使用默认配置（antm2c数据集，TFRecord数据）
bash src/analysis/alignment_analysis/quick_start.sh

# 指定数据集
bash src/analysis/alignment_analysis/quick_start.sh antm2c

# 使用本地数据
bash src/analysis/alignment_analysis/quick_start.sh antm2c 1
```

**Windows:**
```cmd
REM 使用默认配置
src\analysis\alignment_analysis\quick_start.bat

REM 指定数据集
src\analysis\alignment_analysis\quick_start.bat antm2c

REM 使用本地数据
src\analysis\alignment_analysis\quick_start.bat antm2c 1
```

### 方式2: 直接运行Python脚本

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 2 3 \
    --datasets antm2c \
    --use_local_data 0
```

## 详细示例

### 示例1: 测试单个模型的所有对齐方法

只测试`dnn_mm`模型：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 \
    --models dnn_mm \
    --alignments none kl contrastive cosine mmd adversarial \
    --lambda_weights 0.0 0.1 0.2 0.3 0.4 0.5
```

### 示例2: 测试特定的对齐方法

只测试KL散度和对比学习对齐：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 \
    --models dnn_mm lmf marn \
    --alignments kl contrastive \
    --lambda_weights 0.1 0.2 0.3
```

### 示例3: 快速验证（小规模测试）

用于验证代码是否正常工作：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 \
    --models dnn_mm \
    --alignments none kl \
    --lambda_weights 0.0 0.1
```

这将只运行2个实验（none-0.0 和 kl-0.1），大约需要几分钟到几十分钟。

### 示例4: 单GPU运行

如果只有一个GPU可用：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 \
    --models dnn_mm lmf marn mtfn naml \
    --alignments none kl contrastive
```

### 示例5: 运行单个实验

如果需要精确控制单个实验：

```bash
python src/analysis/alignment_analysis/alignment_trainer.py \
    --model_name dnn_mm \
    --dataset_name antm2c \
    --alignment_method kl \
    --lambda_weight 0.1 \
    --cuda 0 \
    --use_local_data 0
```

### 示例6: 不同数据集

在microlens数据集上运行：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 2 3 \
    --datasets microlens \
    --models dnn_mm lmf marn
```

### 示例7: 自定义lambda权重范围

测试更细粒度的lambda值：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 \
    --models dnn_mm lmf \
    --alignments kl contrastive \
    --lambda_weights 0.0 0.05 0.1 0.15 0.2 0.25 0.3
```

## 结果分析示例

### 查看CSV结果

```bash
# 使用pandas查看
python -c "import pandas as pd; df = pd.read_csv('experiments/results/alignment_analysis.csv'); print(df.head(20))"

# 或使用Excel/LibreOffice打开CSV文件
```

### 生成汇总报告

```bash
python src/analysis/alignment_analysis/summarize_results.py \
    --csv experiments/results/alignment_analysis.csv \
    --output experiments/results/alignment_summary.txt
```

### 查看特定模型的结果

```bash
python -c "
import pandas as pd
df = pd.read_csv('experiments/results/alignment_analysis.csv')
model_df = df[df['model'] == 'dnn_mm'].sort_values('test_auc', ascending=False)
print(model_df[['alignment', 'lambda', 'test_auc', 'test_loss']].head(10))
"
```

### 对比不同对齐方法

```bash
python -c "
import pandas as pd
df = pd.read_csv('experiments/results/alignment_analysis.csv')
df = df[df['status'] == 'OK']
summary = df.groupby('alignment')['test_auc'].agg(['mean', 'std', 'min', 'max'])
print(summary)
"
```

## 常见使用场景

### 场景1: 论文实验 - 完整对比

运行所有模型、所有对齐方法的完整实验：

```bash
bash src/analysis/alignment_analysis/quick_start.sh antm2c 0
```

预计时间：数小时到1-2天（取决于硬件）

### 场景2: 快速原型 - 验证想法

只测试几个代表性模型和方法：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 \
    --models dnn_mm lmf naml \
    --alignments none kl contrastive \
    --lambda_weights 0.0 0.1 0.2
```

预计时间：几小时

### 场景3: 调优 - 寻找最佳lambda

针对特定模型和对齐方法，测试更多lambda值：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 \
    --models dnn_mm \
    --alignments kl \
    --lambda_weights 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
```

### 场景4: 消融实验 - 单一变量

固定其他条件，只改变对齐方法：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 2 3 \
    --models dnn_mm \
    --alignments none kl contrastive cosine mmd adversarial \
    --lambda_weights 0.2
```

## 高级用法

### 自定义训练配置

修改`config/train.yaml`中的参数，例如：
- `max_epochs`: 最大训练轮数
- `batch_size`: 批次大小
- `early_stop_patience`: 早停耐心值

### 自定义模型配置

修改`config/best_param.yaml`中特定模型的超参数。

### 并行策略优化

如果有更多GPU，可以增加并行度：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 2 3 4 5 6 7 \
    --models dnn_mm lmf marn mtfn naml make em3 simcen gmmf dmf
```

任务会自动均匀分配到所有GPU。

### 断点续传

如果实验中断，可以通过排除已完成的实验来继续：

1. 查看CSV文件，确定哪些实验已完成
2. 修改`--models`、`--alignments`或`--lambda_weights`参数，只运行未完成的部分

例如，如果dnn_mm和lmf已完成：

```bash
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 2 3 \
    --models marn mtfn naml make em3 simcen gmmf dmf \
    --alignments none kl contrastive cosine mmd adversarial \
    --lambda_weights 0.0 0.1 0.2 0.3 0.4 0.5
```

## 故障排除

### 问题1: CUDA out of memory

**解决方案：**
```bash
# 减少并行GPU数量
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1  # 只用2个GPU

# 或在config/train.yaml中减小batch_size
```

### 问题2: 某个模型一直失败

**解决方案：**
```bash
# 跳过该模型，先运行其他模型
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --models dnn_mm lmf marn mtfn naml make em3 simcen gmmf  # 排除dmf
```

### 问题3: 训练不收敛

**可能原因：**
- lambda_weight过大
- 学习率不合适

**解决方案：**
- 减小lambda_weight范围
- 调整config/best_param.yaml中的学习率

## 性能优化建议

1. **使用SSD存储数据**：可显著提升数据加载速度
2. **增加batch_size**：如果GPU内存允许，增大batch_size可提升训练速度
3. **使用混合精度训练**：在代码中启用AMP可加速训练
4. **预加载数据**：使用`use_local_data=1`预先将数据加载到内存

## 结果解读

### AUC提升的意义

- **+0.001 ~ +0.005**: 小幅提升，可能有统计意义
- **+0.005 ~ +0.01**: 中等提升，通常有实际价值
- **+0.01以上**: 显著提升，值得深入研究

### Lambda权重的选择

- **0.0**: 基线，无对齐损失
- **0.1 ~ 0.2**: 通常是较好的起点
- **0.3 ~ 0.5**: 较大的权重，可能导致训练不稳定

### 对齐方法的特点

- **KL散度**: 适合分布对齐，计算高效
- **对比学习**: 适合语义对齐，效果通常较好
- **余弦相似度**: 简单直接，适合快速验证
- **MMD**: 理论基础扎实，适合分布匹配
- **对抗对齐**: 效果可能最好，但训练较复杂

## 引用和参考

如果使用本框架进行研究，请引用：
- MMCTR Benchmark
- 相关的对齐方法论文

更多信息请参考`README.md`。