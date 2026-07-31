# 模态融合方法 × 模型架构 分析文档

## 1. 概述

本模块将各 MM-CTR 模型复制到 `fusion_analysis/` 目录，并对使用硬编码融合方法的模型进行适配，
使其支持可配置的融合策略，便于系统性地分析不同融合方法对不同模型架构的影响。

### 支持的融合方法（排除 add / mean）

#### 本地融合方法（应用于单个物品特征）

| 融合方法 | 类名 | 输出维度 | 特点 |
|---------|------|---------|------|
| **maf** | MAF | dim | 模态注意力融合，逐模态线性变换 + 非线性激活后求和 |
| **cat** | CAT | dim × M | 直接拼接，保留所有模态信息，维度随模态数线性增长 |
| **lmf** | LMF | output_dim (可配置) | 低秩多模态融合，通过秩分解近似张量积 |
| **src** | SRCModule | dim | 随机逆向跨模态扩散融合，多步扩散-去噪过程 |
| **mtfn** | MTFNFusion | dim | 多秩张量融合网络，多组线性 head 逐元素乘积后求和 |
| **fq-former** | FQFormer | dim × Q | 可学习 query 自注意力融合，Q 为 query 数量 |
| **simcen** | SimCENFusion | hidden_units[-1] | Segmentation + MLEP 三路拆分网络，附带对比损失 |

#### 序列感知融合方法（需要序列上下文）

| 融合方法 | 类名 | 输出维度 | 特点 |
|---------|------|---------|------|
| **dta** | DecoupledTargetAttention | attention_dim | 解耦目标注意力，基于多模态相似度加权ID序列 |
| **gmmf** | GMMFFusion | dim × M | DSN(自编码器+CGAN+门控) 融合，附带重建/GAN辅助损失 |
| **dmf** | DMFFusion | mlp_dims[-1] | DTA + SimTier 双路径加权融合 |

> M = 模态数量，dim = projection_dim，Q = query_num

**注意**：所有已注册模型均已强制兼容序列感知融合 —— BaseSeqModel 模型原生支持；
BaseModel 模型（MB / PAMD / DNN_mm）在序列感知融合时会自动跳过 pooling，
直接对原始 3D 序列特征应用序列感知融合。

---

## 2. 模型分类

### 2.1 已适配融合方法的模型（有额外操作 + 原始硬编码融合）

这些模型除了融合外有独特的建模操作，原始代码硬编码了特定融合方法，
本版本通过 `model_config['modal_fusion_method']` 使其可配置。

| 模型 | 基类 | 原始融合 | 核心额外操作 | 适配方式 |
|------|-----|---------|------------|---------|
| **Diff_MSIN** | BaseSeqModel | src | 专有/共享专家、门控、CrossNetwork、HingeCosineLoss | 替换 `src_fusion` 为可配置融合层 + 投影对齐 |
| **EM3** | BaseSeqModel | fq-former | DIN 池化、CIC 对比学习 | FQ-Former 特殊路径 + 通用 dict 融合路径分支 |
| **MARN** | BaseSeqModel | maf | ModalitySplit、DDMA(双判别器)、GRL | 替换 `maf` 为可配置融合层 + 逐步序列融合 |
| **NAML** | BaseSeqModel | maf | UserEncoder 注意力编码 | 替换 `maf` 为可配置融合层 + 投影对齐至点积维度 |
| **M3SRec** | BaseSeqModel | _ModalAttentionFusion | 位置/模态嵌入、模态专有 MoE、跨模态自注意力+MoE | 替换 user/target `_ModalAttentionFusion` 为可配置融合层；序列感知融合时跳过最后有效时间步聚合 |
| **MB** | BaseModel | _MBModalityFusion | 双 ID 编码、模态特定 user encoder、模态打分、模态平衡对抗损失（PGD） | 本地融合：替换 target/seq 模态融合；**序列感知融合：跳过 pooling，直接对原始 3D 序列融合** |
| **MMMLP** | BaseSeqModel | cat (channel-wise) | 每模态 MLP-Mixer + Fusion Mixer | 每时间步可配置融合 + 跨时间 Fusion Mixer；序列感知融合时跳过 Fusion Mixer |
| **PAMD** | BaseModel | mean over pairs | 模态对解耦（共有/独有）+ pair-attention + aux loss | 本地融合：保留 pair 解耦 + 配置融合；**序列感知融合：目标侧保留 pair 解耦（pair 融合用 cat），序列侧跳过 pair 解耦直接序列感知融合** |

**适配关键设计：**
- **fusion_projector**：当融合输出维度 ≠ `projection_dim` 时（如 `cat` 或 `lmf`），
  自动插入线性投影层将输出映射回 `projection_dim`，确保与下游操作维度兼容。
- **逐时间步融合**：对序列特征逐时间步应用本地融合函数，而非直接传入 3D 张量，
  确保兼容仅支持 2D 输入的融合方法（如 LMF、SRC）。
- **BaseModel 模型的序列感知融合强制兼容**：MB / PAMD / DNN_mm 在选择 DTA/GMMF/DMF 时
  会跳过 pooling，使用 `feats_seq_p` 原始 3D 序列直接调用 `fuse_seq_*` 系列函数。

### 2.2 已可配置融合方法的模型

| 模型 | 基类 | 说明 |
|------|-----|------|
| **DNN_mm** | BaseModel | 基础多模态 DNN，支持本地融合 + 序列感知融合（跳过 pooling 路径） |
| **DNN_mm_seq** | BaseSeqModel | 序列版多模态 DNN，全融合方法支持 |
| **DMF** | BaseSeqModel | 本地融合：原始 DTA + SimTier 双路径；**序列感知融合：用配置 fusion 替代 DTA + me_mlp 路径，保留 SimTier（基于固定 cat 相似度）** |
| **MAKE** | BaseSeqModel | DIN + SimTier，全融合方法支持 |

### 2.3 强制适配的"融合即模型"模型

原始模型本身就是 “某种融合 + MLP”，融合即模型核心。本框架强制忽略原始结构，
将其视为标准 BaseSeqModel 包装：默认融合方法保持原模型同名，但可被任意覆盖。

| 模型 | 默认融合 | 实现 |
|------|---------|------|
| **LMF** | lmf | 继承通用 `FusionWrapperBase`（`_fusion_wrapper.py`） |
| **MTFN** | mtfn | 同上 |
| **GMMF** | gmmf | 同上（忽略原 GAN/DSN 训练循环） |
| **SimCEN** | simcen | 同上（忽略原 Segmentation+MLEP 结构） |

### 2.4 离散编码（码本量化）类模型

这类模型先将连续模态特征量化为离散码字，再查表得到码本嵌入，原始融合策略为 `cat`。
本框架已将其接入，使融合方法可配置。**前提**：需先训练好对应的预训练码本
（QARM 依赖 `{checkpoint_dir}/{dataset}_{m}_rq.npz`；MCCA 依赖
`{checkpoint_dir}/{dataset}_psrq.pt`），其专属超参（`codebook_size` / `n_levels` /
`psrq_dims` 等）会从 `config/model.yaml` 自动加载合并，以匹配预训练码本。

| 模型 | 基类 | 原始融合 | 适配方式 |
|------|-----|---------|---------|
| **QARM** | BaseSeqModel | cat | RQ 编码取码本最近向量 → 多层码本嵌入查表 → 序列侧 mean pooling → 可配置融合（目标侧 + 序列侧）→ 与原始 Cross+Deep（comb）结构融合输出 |
| **MCCA** | BaseSeqModel | cat | PSRQ 编码取码本最近向量（含 joint 联合码本）→ 多层码本嵌入查表 → 序列侧 mean pooling → 可配置融合 → 与原始 joint 联合码本表征拼接后送入 DNN |

> 适配思路：**获取码本中最近的向量 → mean pooling 得到多模态表征 → 与原结构融合**，
> 与其他模型共用 `_fusion_helper` 的本地 / 序列感知融合接口。

---

## 3. 文件结构

```
src/analysis/fusion_analysis/
├── __init__.py              # 模块说明与常量定义
├── _fusion_helper.py        # 融合方法统一初始化与前向接口工具
├── _fusion_wrapper.py       # 强制适配的通用 BaseSeqModel 融合包装基类
├── run_analysis.py          # 主运行脚本（模型 × 融合方法 遍历测试）
├── README.md                # 本文档
│
├── dnn.py                   # DNN / DNN_mm（强制兼容序列感知融合）/ DNN_mm_seq
├── DMF.py                   # DMF（强制兼容序列感知融合）
├── MAKE.py                  # MAKE
│
├── Diff_MSIN.py             # Diff_MSIN（已适配：src → 可配置）
├── EM3.py                   # EM3（已适配：fq-former → 可配置）
├── MARN.py                  # MARN（已适配：maf → 可配置）
├── NAML.py                  # NAML（已适配：maf → 可配置）
├── M3SRec.py                # M3SRec（已适配：_ModalAttentionFusion → 可配置）
├── MB.py                    # MB（已适配 + 强制兼容序列感知融合）
├── MMMLP.py                 # MMMLP（已适配：cat → 可配置 + Fusion Mixer 保留）
├── PAMD.py                  # PAMD（已适配 + 强制兼容序列感知融合）
│
├── LMF.py                   # 强制适配（继承 FusionWrapperBase）
├── MTFN.py                  # 强制适配
├── GMMF.py                  # 强制适配
├── SimCEN.py                # 强制适配
│
├── QARM.py                  # 离散编码类（RQ 码本 → mean pool → 可配置融合 + Cross/Deep）
└── MCCA.py                  # 离散编码类（PSRQ 码本 → mean pool → 可配置融合 + joint）
```

---

## 4. 使用方式

### 4.1 完整分析（所有模型 × 所有融合方法）

```bash
cd Benchmark
python src/analysis/fusion_analysis/run_analysis.py --dataset_name tiktok
```

### 4.2 指定模型和融合方法

```bash
# 只测试 MARN 和 NAML，使用 maf 和 cat
python src/analysis/fusion_analysis/run_analysis.py \
    --dataset_name tiktok \
    --models MARN NAML \
    --fusions maf cat

# 只测试 Diff_MSIN 的所有融合方法
python src/analysis/fusion_analysis/run_analysis.py \
    --dataset_name antm2c \
    --models Diff_MSIN
```

### 4.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_name` | tiktok | 数据集：tiktok / antm2c / microlens |
| `--models` | 所有已注册模型 | 要测试的模型列表 |
| `--fusions` | 所有支持的融合方法 | 要测试的融合方法列表 |
| `--use_local_data` | 0 | 是否使用本地离线数据 |
| `--max_epochs` | 配置文件值 | 覆盖最大训练轮数 |
| `--cuda` | 配置文件值 | GPU 设备编号，-1 为 CPU |
| `--seed` | 2025 | 随机种子 |

---

## 5. 输出

运行结束后会输出：

1. **控制台/日志**：按模型分组的对比表格（Val AUC、Test AUC、训练时间、参数量等）
2. **CSV 文件**：保存在 `experiments/logs/` 目录，命名格式：
   `fusion_model_analysis_{dataset}_{timestamp}.csv`

### 输出示例

```
══════════════════════════════════════════════════════════════
模态融合方法 × 模型架构 对比结果
══════════════════════════════════════════════════════════════
模型           融合方法     Val AUC   Test AUC  Test Loss   时间(s)       可训练参数
──────────────────────────────────────────────────────────────
Diff_MSIN      maf          0.7842    0.7821    0.4523     120.3       2,345,678
Diff_MSIN      cat          0.7810    0.7798    0.4567     118.7       2,567,890
Diff_MSIN      src          0.7856    0.7835    0.4501     145.2       2,445,678
──────────────────────────────────────────────────────────────
MARN           maf          0.7765    0.7742    0.4612     95.4        1,890,432
MARN           cat          0.7730    0.7715    0.4645     93.2        2,012,544
──────────────────────────────────────────────────────────────
```

---

## 6. 扩展

### 添加新融合方法

1. 在 `models/layers/modal_fusion.py` 中实现新融合类，确保：
   - 接受 `(dim, mm_features)` 构造参数
   - 提供 `getDim()` 方法返回输出维度
   - `forward(mm_feats: dict)` 接受模态特征字典
2. 在 `_FUSION_MAP` 中注册
3. 在 `run_analysis.py` 的 `SUPPORTED_FUSIONS` 中添加

### 添加新模型

1. 在 `fusion_analysis/` 目录中创建模型文件
2. 确保 `__init__` 读取 `model_config['modal_fusion_method']`
3. 使用 `modal_fusion.get_fusion_layer()` 创建融合层
4. 通过 `fusion_projector` 处理维度不匹配
5. 在 `run_analysis.py` 的 `MODEL_REGISTRY` 中注册
