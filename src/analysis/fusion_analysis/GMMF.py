"""
GMMF 模型 —— 融合分析版本（强制适配）
原始 GMMF 模型核心是 DSN（自编码器 + CGAN + 自动差分）+ 模态兴趣门控，
融合即模型本身。本文件忽略原始结构（包括 GAN 训练循环），
将 GMMF 视为一个标准 BaseSeqModel 包装：默认融合方法为 'gmmf'，但可被任意配置覆盖。
"""
from analysis.fusion_analysis._fusion_wrapper import FusionWrapperBase


class GMMF(FusionWrapperBase):
    _DEFAULT_FUSION = 'gmmf'
