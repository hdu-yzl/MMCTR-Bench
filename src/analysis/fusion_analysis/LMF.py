"""
LMF 模型 —— 融合分析版本（强制适配）
原始 LMF 模型本身就是 “LMF 融合 + MLP”，融合即模型。本文件忽略原始结构，
将 LMF 视为一个标准 BaseSeqModel 包装：默认融合方法为 'lmf'，但可被任意配置覆盖。
"""
from analysis.fusion_analysis._fusion_wrapper import FusionWrapperBase


class LMF(FusionWrapperBase):
    _DEFAULT_FUSION = 'lmf'
