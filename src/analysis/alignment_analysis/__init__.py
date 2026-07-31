"""
模态对齐分析模块

提供对多模态CTR模型进行模态对齐分析的完整框架
"""

__version__ = "1.0.0"
__author__ = "MMCTR Benchmark Team"

from .alignment_trainer import Trainer, AlignmentModelWrapper

__all__ = [
    'Trainer',
    'AlignmentModelWrapper',
]