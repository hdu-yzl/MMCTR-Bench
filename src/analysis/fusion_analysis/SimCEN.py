"""
SimCEN 模型 —— 融合分析版本（强制适配）
原始 SimCEN 模型核心是 Segmentation + MLEP（三路 ego/v1/v2 拆分）+ InfoNCE 对比损失，
融合即模型本身。本文件忽略原始结构，将 SimCEN 视为一个标准 BaseSeqModel 包装：
默认融合方法为 'simcen'，但可被任意配置覆盖。
"""
from analysis.fusion_analysis._fusion_wrapper import FusionWrapperBase


class SimCEN(FusionWrapperBase):
    _DEFAULT_FUSION = 'simcen'
