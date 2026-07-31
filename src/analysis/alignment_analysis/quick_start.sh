#!/bin/bash
# 对齐分析快速启动脚本

echo "=========================================="
echo "模态对齐分析实验 - 快速启动"
echo "=========================================="
echo ""

# 检查参数
DATASET=${1:-antm2c}
USE_LOCAL=${2:-0}

echo "配置信息:"
echo "  数据集: $DATASET"
echo "  使用本地数据: $USE_LOCAL"
echo "  GPU: 0, 1, 2, 3"
echo ""

# 创建结果目录
mkdir -p experiments/results
mkdir -p experiments/logs

echo "开始运行实验..."
echo "预计总实验数: ~341个 (11模型 × 6对齐方法 × 6lambda值，去除重复)"
echo "预计时间: 数小时到数天（取决于数据集大小和GPU性能）"
echo ""

# 运行主脚本
python src/analysis/alignment_analysis/run_alignment_analysis.py \
    --gpu_ids 0 1 2 3 \
    --datasets $DATASET \
    --models mtfn naml make em3 simcen gmmf dmf diff_msin \
    --alignments none kl contrastive cosine mmd adversarial \
    --lambda_weights 0.0 0.1 0.2 0.3 0.4 0.5 \
    --use_local_data $USE_LOCAL \
    --log_dir experiments/logs \
    --output_csv experiments/results/alignment_analysis_${DATASET}.csv

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "实验完成！"
    echo "=========================================="
    echo ""
    echo "生成结果汇总..."
    
    # 生成汇总报告
    python src/analysis/alignment_analysis/summarize_results.py \
        --csv experiments/results/alignment_analysis_${DATASET}.csv \
        --output experiments/results/alignment_summary_${DATASET}.txt
    
    echo ""
    echo "结果文件:"
    echo "  CSV: experiments/results/alignment_analysis_${DATASET}.csv"
    echo "  报告: experiments/results/alignment_summary_${DATASET}.txt"
    echo "  日志: experiments/logs/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "实验失败，请检查日志文件"
    echo "=========================================="
    exit 1
fi
