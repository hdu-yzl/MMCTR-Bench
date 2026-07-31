"""
对齐分析结果汇总脚本
读取CSV结果文件，生成详细的分析报告和可视化图表
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

def load_results(csv_path: str) -> pd.DataFrame:
    """加载结果CSV"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"结果文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # 过滤失败的实验
    df = df[df['status'] == 'OK'].copy()
    
    # 转换数据类型
    df['lambda'] = df['lambda'].astype(float)
    df['test_auc'] = df['test_auc'].astype(float)
    df['test_loss'] = df['test_loss'].astype(float)
    
    return df


def print_best_configs(df: pd.DataFrame):
    """打印每个模型的最佳配置"""
    print("\n" + "="*80)
    print("每个模型的最佳对齐配置 (按Test AUC排序)")
    print("="*80)
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        # 找到最佳配置
        best_row = model_df.loc[model_df['test_auc'].idxmax()]
        baseline_row = model_df[model_df['alignment'] == 'none']
        
        if len(baseline_row) > 0:
            baseline_auc = baseline_row.iloc[0]['test_auc']
            improvement = best_row['test_auc'] - baseline_auc
            improvement_pct = (improvement / baseline_auc) * 100
        else:
            baseline_auc = None
            improvement = None
            improvement_pct = None
        
        print(f"\n模型: {model.upper()}")
        print(f"  最佳配置: {best_row['alignment']} (λ={best_row['lambda']:.2f})")
        print(f"  Test AUC: {best_row['test_auc']:.6f}")
        print(f"  Test Loss: {best_row['test_loss']:.6f}")
        
        if baseline_auc is not None:
            print(f"  基线 (none): {baseline_auc:.6f}")
            print(f"  提升: {improvement:+.6f} ({improvement_pct:+.2f}%)")


def print_alignment_comparison(df: pd.DataFrame):
    """打印不同对齐方法的对比"""
    print("\n" + "="*80)
    print("对齐方法效果对比 (所有模型平均)")
    print("="*80)
    
    # 按对齐方法分组
    alignment_stats = df.groupby('alignment').agg({
        'test_auc': ['mean', 'std', 'min', 'max'],
        'test_loss': ['mean', 'std']
    }).round(6)
    
    print(f"\n{'对齐方法':<15} {'平均AUC':<12} {'标准差':<12} {'最小AUC':<12} {'最大AUC':<12}")
    print("-"*80)
    
    for alignment in alignment_stats.index:
        mean_auc = alignment_stats.loc[alignment, ('test_auc', 'mean')]
        std_auc = alignment_stats.loc[alignment, ('test_auc', 'std')]
        min_auc = alignment_stats.loc[alignment, ('test_auc', 'min')]
        max_auc = alignment_stats.loc[alignment, ('test_auc', 'max')]
        
        print(f"{alignment:<15} {mean_auc:<12.6f} {std_auc:<12.6f} {min_auc:<12.6f} {max_auc:<12.6f}")


def print_lambda_analysis(df: pd.DataFrame):
    """分析lambda权重的影响"""
    print("\n" + "="*80)
    print("Lambda权重影响分析")
    print("="*80)
    
    # 排除none方法
    df_with_align = df[df['alignment'] != 'none'].copy()
    
    if len(df_with_align) == 0:
        print("没有对齐方法的结果")
        return
    
    # 按lambda分组
    lambda_stats = df_with_align.groupby('lambda').agg({
        'test_auc': ['mean', 'std', 'count']
    }).round(6)
    
    print(f"\n{'Lambda':<10} {'平均AUC':<12} {'标准差':<12} {'实验数':<10}")
    print("-"*50)
    
    for lw in sorted(lambda_stats.index):
        mean_auc = lambda_stats.loc[lw, ('test_auc', 'mean')]
        std_auc = lambda_stats.loc[lw, ('test_auc', 'std')]
        count = int(lambda_stats.loc[lw, ('test_auc', 'count')])
        
        print(f"{lw:<10.2f} {mean_auc:<12.6f} {std_auc:<12.6f} {count:<10}")


def print_model_alignment_matrix(df: pd.DataFrame):
    """打印模型×对齐方法的矩阵"""
    print("\n" + "="*100)
    print("模型 × 对齐方法 最佳AUC矩阵")
    print("="*100)
    
    # 对每个(模型, 对齐方法)组合，找到最佳lambda的AUC
    pivot_data = []
    
    for model in sorted(df['model'].unique()):
        row_data = {'model': model}
        for alignment in sorted(df['alignment'].unique()):
            subset = df[(df['model'] == model) & (df['alignment'] == alignment)]
            if len(subset) > 0:
                best_auc = subset['test_auc'].max()
                row_data[alignment] = best_auc
            else:
                row_data[alignment] = None
        pivot_data.append(row_data)
    
    pivot_df = pd.DataFrame(pivot_data)
    pivot_df = pivot_df.set_index('model')
    
    # 打印表格
    alignments = sorted(df['alignment'].unique())
    header = f"{'模型':<12}"
    for align in alignments:
        header += f"{align:<12}"
    print(header)
    print("-"*100)
    
    for model in pivot_df.index:
        row = f"{model:<12}"
        for align in alignments:
            val = pivot_df.loc[model, align]
            if pd.notna(val):
                row += f"{val:<12.6f}"
            else:
                row += f"{'N/A':<12}"
        print(row)


def generate_summary_report(csv_path: str, output_path: str = None):
    """生成完整的汇总报告"""
    df = load_results(csv_path)
    
    print("\n" + "="*80)
    print("对齐分析结果汇总报告")
    print("="*80)
    print(f"结果文件: {csv_path}")
    print(f"总实验数: {len(df)}")
    print(f"模型数: {df['model'].nunique()}")
    print(f"对齐方法数: {df['alignment'].nunique()}")
    print(f"Lambda权重数: {df['lambda'].nunique()}")
    
    # 各种分析
    print_best_configs(df)
    print_alignment_comparison(df)
    print_lambda_analysis(df)
    print_model_alignment_matrix(df)
    
    # 保存详细报告
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("对齐分析详细报告\n")
            f.write("="*80 + "\n\n")
            
            # 每个模型的详细结果
            for model in sorted(df['model'].unique()):
                model_df = df[df['model'] == model].sort_values('test_auc', ascending=False)
                
                f.write(f"\n模型: {model.upper()}\n")
                f.write("-"*80 + "\n")
                f.write(f"{'排名':<6} {'对齐方法':<15} {'Lambda':<10} {'Test AUC':<12} {'Test Loss':<12}\n")
                f.write("-"*80 + "\n")
                
                for rank, (idx, row) in enumerate(model_df.iterrows(), 1):
                    f.write(f"{rank:<6} {row['alignment']:<15} {row['lambda']:<10.2f} "
                           f"{row['test_auc']:<12.6f} {row['test_loss']:<12.6f}\n")
        
        print(f"\n详细报告已保存到: {output_path}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="对齐分析结果汇总")
    parser.add_argument("--csv", type=str, 
                       default="experiments/results/alignment_analysis.csv",
                       help="结果CSV文件路径")
    parser.add_argument("--output", type=str,
                       default="experiments/results/alignment_summary.txt",
                       help="输出报告路径")
    
    args = parser.parse_args()
    
    generate_summary_report(args.csv, args.output)


if __name__ == "__main__":
    main()