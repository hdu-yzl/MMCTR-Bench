"""
融合分析模块 —— 将各 MM-CTR 模型适配可配置的融合方法，用于对比实验。

支持的融合方法（排除 add 和 mean）:
  本地融合: maf, cat, lmf, src, mtfn, fq-former, simcen
  序列感知融合: dta, gmmf, dmf

设计目标：所有已注册模型 × 所有融合方法 均可执行（强制兼容）。
  - BaseSeqModel 模型：原生支持本地与序列感知融合
  - BaseModel 模型：序列感知融合时跳过 pooling，直接对原始 3D 序列融合

模型分类:
  - 需要适配（有独立于融合的操作 + 原始硬编码融合）:
      Diff_MSIN, EM3, MARN, NAML, M3SRec, MB, MMMLP, PAMD
  - 已可配置融合方法的模型:
      DNN_mm, DNN_mm_seq, DMF, MAKE
  - 强制适配（融合即模型 → 包装为标准 BaseSeqModel + 可配置融合）:
      LMF, MTFN, GMMF, SimCEN
  - 暂未接入（依赖预训练 RQ/PSRQ codebooks）:
      QARM, MCCA
"""

from analysis.fusion_analysis._fusion_helper import LOCAL_FUSIONS, SEQ_FUSIONS, ALL_FUSIONS

SUPPORTED_FUSIONS = sorted(ALL_FUSIONS)

# 需要适配融合方法的模型（有额外操作 + 原始硬编码融合）
ADAPTED_MODELS = ['Diff_MSIN', 'EM3', 'MARN', 'NAML',
                  'M3SRec', 'MB', 'MMMLP', 'PAMD']

# 已可配置融合方法的模型
CONFIGURABLE_MODELS = ['DNN_mm', 'DNN_mm_seq', 'DMF', 'MAKE']

# 强制适配的 "融合即模型" 模型（统一为 BaseSeqModel + 可配置融合包装）
FORCE_ADAPTED_MODELS = ['LMF', 'MTFN', 'GMMF', 'SimCEN']

# 暂未接入的模型（依赖预训练 RQ/PSRQ codebooks）
PRETRAINED_DEPENDENT_MODELS = ['QARM', 'MCCA']
