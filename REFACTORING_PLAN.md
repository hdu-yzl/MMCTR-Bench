# MMCTR Benchmark 全局改造方案与协作规范

> 文档状态：`ACTIVE`  
> 当前版本：`v1.20`
> 最近更新：`2026-08-04`
> 适用范围：本仓库内的代码、配置、数据处理、训练、调参、评估、分析、文档与产物管理  
> 目标读者：维护者、研究人员，以及参与本项目改造的所有 agent

## 0. 文档定位与执行规则

本文档是本次工程化改造的单一事实源（Single Source of Truth）。所有 agent 在修改仓库前必须完整阅读本文档，并遵循以下顺序：

1. 查看“当前基线”和“不可违反的红线”。
2. 在“改造任务总表”中确认任务编号、依赖、状态和文件范围。
3. 将任务标记为 `IN_PROGRESS`，登记负责人、预计修改文件和验证方式。
4. 完成最小范围修改，执行对应质量门禁。
5. 补充验证证据，将任务更新为 `REVIEW` 或 `DONE`。
6. 若发现计划外问题，先登记新任务或决策记录，不顺手扩大改造范围。

本文档可以更新，但必须同时更新版本、日期、进度表和变更记录。涉及目录架构、公共接口、配置层级、指标口径或实验协议的变更，必须先记录决策再实施。

状态统一使用：

| 状态 | 含义 |
|---|---|
| `TODO` | 尚未开始，前置条件可能尚未满足 |
| `IN_PROGRESS` | 已有唯一负责人，正在实施 |
| `BLOCKED` | 存在明确阻塞，必须写明原因和解除条件 |
| `REVIEW` | 实现完成，等待验证或审查 |
| `DONE` | 验收条件全部满足并已有证据 |
| `CANCELLED` | 经维护者确认不再实施，必须写明原因 |

---

## 1. 当前仓库基线

### 1.1 已确认的规模

本次盘点基于 `2026-07-31` 工作区：

| 项目 | 当前值 |
|---|---:|
| `src/` 下 Python 文件 | 117 |
| Python 物理代码行 | 约 16,679 |
| Python 3.12 AST 解析失败 | 0 |
| 源码树内 `.pyc` | 105 |
| 源码树内 PDF/PNG | 19 |
| `mm_ctr_models` 与 `fusion_analysis` 同名实现 | 17 |
| 数据集适配 | AntM2C、MicroLens、TikTok |
| 普通/多模态/序列/量化推荐模型 | 多类，共 20+ 注册名称 |
| 自动化测试目录 | 0 |
| 根 README 有效内容 | 0 |
| 可用 Git 工作树 | 当前未检测到 |

静态解析通过只表示语法可解析，不表示依赖可安装、模块可导入、数据可读取、训练可运行或结果可复现。

### 1.2 当前有效模块

- `src/data/processors/`：三个数据集的预处理脚本。
- `src/data/dataloaders/`：基于 TensorFlow TFRecord、NumPy 和 PyTorch 的数据适配。
- `src/models/ctr_models/`：DNN、DCN、DeepFM、DIN、AutoInt 等 CTR 基线。
- `src/models/mm_ctr_models/`：多模态 CTR 模型。
- `src/models/pre_models/`：RQ、PSRQ 等量化预模型。
- `src/models/layers/`：通用层、融合层、对齐层、池化和损失。
- `src/trainers/`：训练入口及预模型训练入口。
- `src/scripts/`：调参和批量运行脚本。
- `src/analysis/`：冷启动、模态鲁棒性、对齐、融合和绘图分析。

### 1.3 已确认的主要问题

以下问题不是风格偏好，而是会影响可运行性、可维护性或 benchmark 可信度的事实。

| 优先级 | 问题 | 当前证据 | 影响 |
|---|---|---|---|
| P0 | 调参使用 test AUC 选择最优参数 | `src/scripts/Tuner.py`、`Codebook_Tuner.py` | 测试集泄漏，最终结果不再是无偏评估 |
| P0 | checkpoint 只按 `model_name.pt` 命名 | `BaseModel`、`BaseSeqModel` | 跨数据集、种子或并行任务互相覆盖，可能加载错误权重 |
| P0 | Linux 服务器环境快照尚未整理成面向开源用户的可移植安装说明 | `bm_env.yml` 含服务器绝对 prefix、底层平台包和镜像地址 | 服务器环境可以复现实验，但外部用户无法直接照搬安装 |
| P0 | `use_local_data=1` 会引用不存在的配置文件 | 多个 Trainer/Tuner 引用 `local_data.yaml`、`local_seq_data.yaml` | 明确运行分支不可用 |
| P0 | AntM2C 默认路径硬编码为特定服务器路径 | `config/data.yaml`、`seq_data.yaml` | 外部 Linux 服务器和开源用户无法通过配置直接迁移 |
| P1 | 根 `README.md`、`requrements.txt` 为空，且依赖文件名拼写错误 | 仓库根目录 | 无法完成标准安装和上手 |
| P1 | `setup.py` 发布名为 `mmctr`，实际安装为 `models`、`data`、`utils` 等顶层包 | `setup.py` 与绝对导入 | 易发生包名冲突，入口依赖工作目录和 `sys.path` |
| P1 | `BaseModel` 与 `BaseSeqModel` 重复训练、评估、保存代码 | 两个基类 | 修复需要双改，行为容易漂移 |
| P1 | 模型构造函数负责移设备、建优化器、日志和参数统计 | 多数模型 `__init__` | 模型无法作为纯模块独立测试、导出和组合 |
| P1 | 训练入口在模块导入阶段解析 CLI 并修改全局线程环境 | `src/trainers/Trainers.py` | 不可安全导入，测试和复用困难 |
| P1 | 模型和数据集注册是长 `if/elif` | `src/utils/helper.py` | 扩展点分散，名称与配置易不一致 |
| P1 | `dnn_seq` 配置与 `dnn_mm_seq` 注册名称不一致 | `config/model.yaml`、`helper.py` | 配置无法按名称稳定实例化 |
| P1 | `best_param.yaml` 与 `best_params.yaml` 并存且结构不同（`CFG-001` 已修复） | `config/` 与多个 tuner | 历史文件已迁出可执行配置目录，新 legacy tuner 输出进入 ignored `outputs/tuning/` |
| P1 | 生产模型与融合分析目录存在 17 个同名实现 | 两个模型目录 | 修复无法自动同步，实验实现可能偏离主实现 |
| P1 | forward 会原地修改输入字典，部分模型使用无维度约束的 `squeeze()` | 基类派生模型 | 输入复用和 batch size=1 时存在形状风险 |
| P1 | AntM2C 将六组文本语义拼成 4608 维字段，loader 再按固定位置切出 item 文本 | `6.to_tf.py`、`Dtfloader_Antm2c.py` | 数据语义靠切片位置维持，item 特征被重复存储且难以扩展 |
| P1 | AntM2C 历史序列通过全量逐行 `apply` 和 `items.index` 构建 | `2.user_seq.py`、`3.append_user_seq.py` | 复杂度高，重复 item 事件可能定位到错误位置 |
| P1 | AntM2C 特征提取重复加载模型、图像逐张推理，序列化循环对所有 split 仍使用 train embedding 变量 | `4.ex_text.py`、`5.ex_image.py`、`6.to_tf.py` | 预处理耗时、内存与 split 正确性存在风险 |
| P1 | AntM2C 同一 4608 维字段在非序列 loader 中整体作为 item text，在序列 loader 中又拆成 user/context 与 item text | `Dtfloader_Antm2c.py` 两个读取分支 | 同一数据字段语义不一致，模型比较不公平 |
| P1 | AntM2C writer、配置路径和 `train_shuffle` 查找约定不统一 | `6.to_tf.py` 与 loader/config | 新生成数据不一定能被训练入口直接发现 |
| P1 | 模型、数据处理、分析、绘图脚本普遍依赖当前工作目录 | 多处相对路径和 `sys.path.insert` | 从其他目录或安装后执行不稳定 |
| P2 | `.pyc`、`egg-info`、图表和论文产物混入源码树 | `src/` | 噪声、体积和版本审查成本增加 |
| P2 | 缺少测试、CI、格式化、lint、类型检查和覆盖率基线 | 仓库级 | 改造无法获得快速回归反馈 |
| P2 | 多处裸 `except`、直接 `print`、硬编码 GPU/线程数 | 分析、处理和脚本代码 | 错误可能被吞掉，运行行为不透明 |
| P2 | `evalate`、`multimodel`、文件命名和大小写不规范 | 多处公共接口 | 增加理解成本并固化错误 API |

### 1.4 当前可保留的优点

- 已经形成数据、模型、训练、分析的大体分层。
- 模型统一返回包含 `pred` 的结果字典，并有辅助损失约定的雏形。
- 数据集和模型配置大部分已外置到 YAML。
- 已包含冷启动、缺失模态、对齐、融合、效率等较完整的研究分析维度。
- 固定随机种子、early stop、日志、checkpoint 和 AUC/LogLoss 已有基础实现。

改造应尽量保留这些研究资产，并通过接口收敛代替“大重写”。

---

## 2. 改造目标与非目标

### 2.1 总体目标

改造后的仓库应满足：

1. **可安装**：从任意工作目录均可通过统一包名导入和运行。
2. **可复现**：相同代码、配置、数据版本和种子能够复现实验。
3. **可比较**：所有模型遵循一致的数据、训练预算、指标和结果协议。
4. **可扩展**：新增模型、融合方法或数据集不需要修改长条件分支。
5. **可测试**：不依赖真实大数据或 GPU 即可完成单元和最小 smoke test。
6. **可追踪**：每次运行有唯一 run ID、解析后配置、环境和数据指纹。
7. **可协作**：多个 agent 能按任务边界并行工作，不互相覆盖或重复造轮子。
8. **可发布**：具备完整 README、依赖元数据、许可证/引用说明和版本策略。

本项目按四个大阶段推进，后文所有任务和进度均归入这四个阶段：

1. **第一阶段：开源发布与工程基线**——完成仓库治理、Linux 安装复现、文档、测试、科研红线和运行隔离。
2. **第二阶段：数据处理与模型主干整改**——重点重做 AntM2C 数据链路；多模态模型统一只保留 `BaseSeqModel`，原 `BaseModel` 模型通过历史序列 pooling 接入统一基类。
3. **第三阶段：模型公共组件优化**——统一投影、mask、pooling、fusion 与维度适配，使不同分支、不同模态可独立选择 pooling/fusion 策略。
4. **第四阶段：实验分析体系改造**——统一实验 runner、扰动/分析协议和结果 schema，分析代码复用正式模型，不再复制实现。

### 2.2 非目标

- 第一阶段不追求重写全部算法。
- 不在缺少回归证据时“优化”论文模型公式或默认超参数。
- 不把真实数据、个人路径、密钥或大 checkpoint 纳入源码版本控制。
- 不以代码风格改造为理由同时改变数据切分、训练预算或指标口径。
- 不承诺所有历史脚本永久兼容；废弃接口必须有迁移说明和过渡期。

---

## 3. 不可违反的改造红线

### 3.1 科研与评估红线

1. 训练集用于拟合；验证集用于 early stop、模型选择和超参数搜索；测试集只用于配置冻结后的最终评估。
2. 调参 objective 中禁止访问 test loader、test AUC 或 test loss。
3. 不得用测试集结果回写“最佳参数”。现有相关逻辑必须作为 P0 修复。
4. 数据切分算法、过滤规则、负采样、ID 映射或特征提取发生变化时，必须生成新数据版本并重新建立基线。
5. 不得静默改变论文模型公式、损失权重、输入模态、padding/mask 或指标实现。
6. 任何声称“性能不变”的重构，都必须有相同配置和种子的回归证据。
7. 多种子报告至少保存每个 seed 的独立结果，汇总使用 `mean ± std`，不得只保留最优 seed。

### 3.2 工程红线

1. 禁止不同运行共享同一个 checkpoint、日志或结果文件。
2. 禁止在公共模块导入阶段解析 CLI、启动任务、读大文件或修改全局进程状态。
3. 禁止业务代码依赖 `sys.path.insert` 解决包导入问题。
4. 禁止新增个人绝对路径、固定 GPU 编号、固定机器线程数或环境前缀。
5. 禁止裸 `except:`；必须捕获可预期异常，或记录堆栈后重新抛出。
6. 禁止模型 forward 原地修改调用者传入的 batch/feature 字典。
7. 禁止将数据、checkpoint、运行日志、缓存和临时调参目录提交到源码目录。
8. 禁止未经登记的大范围移动、删除、自动格式化或批量重命名。

### 3.3 协作红线

1. 一个任务同时只能有一个负责人。
2. agent 不得覆盖、回滚或清理不属于自己任务的已有改动。
3. 公共高冲突文件（包元数据、注册表、公共配置、本文档）修改前必须登记。
4. 任务验收前不得将状态标记为 `DONE`。
5. 验证失败时先确认解释器、配置、数据条件，再修改代码。

---

## 4. Linux 服务器定位与执行规范

### 4.1 环境权威边界

自 2026-07-31 起，本项目后续修改与验证统一在 Linux GPU 服务器完成：

| 环境 | 定位 | 可以作为验收依据 | 不可以据此下结论 |
|---|---|---|---|
| Linux GPU 服务器 | **实验运行与依赖兼容的权威环境** | 安装、真实数据训练、CUDA、性能、正式回归和论文结果 | 不把服务器个人路径直接发布给用户 |

Windows 只保留为早期改造记录中的历史环境，不再承担后续编辑、静态检查或任务验收；
已有 Windows 证据无需重跑，也不能替代新的 Linux 验证。

`bm_env.yml` 是现有 Linux 服务器环境的有价值快照，不再标记为“错误环境”或仅因本地 Python 3.12 不同而废弃。开源整改需要在保留其复现实验价值的同时，去掉绝对 `prefix` 等机器专属信息，并补充更适合外部安装的环境文件和说明。

### 4.2 Linux 服务器规范

- 正式训练、全量测试、模型性能回归和依赖兼容性都必须在 Linux 服务器完成。
- 每次正式验证记录实际解释器绝对路径、CUDA/驱动、GPU、PyTorch/TensorFlow 和关键依赖版本。
- 当前 `bm_env.yml` 在新环境方案验证通过前保留为 server snapshot，不随意升级或重解依赖。
- 发布版建议同时提供：去机器路径的 `environment.yml`、必要的精确锁定/快照、以及 README 中的服务器复现步骤。
- `.sh` 批量实验脚本是 Linux 一等入口，但必须调用可配置的解释器、使用唯一输出目录并正确传播退出码。
- 开源配置不得写死服务器个人绝对路径；服务器路径通过未提交的 local config 或环境变量注入。

Linux 文档中的 Python 命令必须指向服务器环境的实际绝对解释器，例如：

```bash
<SERVER_ENV>/bin/python -m pytest
<SERVER_ENV>/bin/python -m mmctr.cli train --config configs/experiments/example.yaml
```

`<SERVER_ENV>` 是文档占位符；执行记录中必须替换为服务器上的真实绝对路径。

#### 4.2.1 VS Code/Codex 服务器接续备注

- 维护者指定服务器继续使用既有环境 `conda activate bm`；不得自动新建替代环境。激活后先记录 `command -v python` 返回的绝对解释器路径。
- 任何新增包、模型、数据、下载缓存、构建依赖和临时产物必须写入当前项目所在的同一磁盘/文件系统。优先使用项目根下 ignored `.cache/`、`.tmp/`、`downloads/`，并按需设置 `PIP_CACHE_DIR`、`CONDA_PKGS_DIRS`、`HF_HOME`、`TORCH_HOME`、`TMPDIR`。
- `<ORIGINAL_REFERENCE_ROOT>` 是改造开始前的最原始版本，仅作为数据位置、历史配置和旧运行约定的**只读参考**；真实路径只进入 ignored private evidence，不得在该目录执行修改、删除或迁移，也不得将其中真实数据和机器专属配置提交到 Git。
- 接续改造的 Codex 必须先读根 `AGENTS.md` 和本文档。以上内容是维护者提供的交接事实，不代表 Linux/CUDA 门禁已经通过；实际服务器验证证据仍需回填 `ENV-001`。

### 4.3 服务器唯一验证规范

- 后续代码修改、静态检查、单元/集成测试、构建、CUDA、真实数据和性能验证全部在本服务器执行。
- Python、pip、pytest、Ruff、mypy 和 build 命令统一使用 `bm` 的绝对解释器
  `<BM_PYTHON>`；绝对路径只记录在 ignored private evidence，不得用 Windows 结果补齐门禁。
- 维护者已授权在既有 `bm` 中按需安装依赖和解决依赖冲突；每次变更仍须记录原因、版本、
  resolver 影响和 smoke 结果，且不得创建替代 Conda/venv 环境。

---

## 5. 目标目录与架构

目标形态如下。迁移应分阶段完成，不要求一次性搬完：

```text
Benchmark/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ CONTRIBUTING.md
├─ REFACTORING_PLAN.md
├─ configs/
│  ├─ datasets/
│  │  ├─ antm2c.yaml
│  │  ├─ microlens.yaml
│  │  └─ tiktok.yaml
│  ├─ models/
│  ├─ training/
│  ├─ experiments/
│  └─ local/
│     └─ paths.example.yaml
├─ src/
│  └─ mmctr/
│     ├─ cli/
│     ├─ config/
│     ├─ data/
│     │  ├─ datasets/
│     │  ├─ preprocessing/
│     │  ├─ schemas.py
│     │  └─ registry.py
│     ├─ models/
│     │  ├─ baselines/
│     │  ├─ multimodal/
│     │  ├─ quantization/
│     │  ├─ components/
│     │  │  ├─ projection.py
│     │  │  ├─ pooling.py
│     │  │  ├─ fusion.py
│     │  │  └─ pipeline.py
│     │  ├─ layers/
│     │  ├─ outputs.py
│     │  └─ registry.py
│     ├─ training/
│     │  ├─ engine.py
│     │  ├─ checkpointing.py
│     │  └─ callbacks.py
│     ├─ evaluation/
│     │  ├─ metrics.py
│     │  └─ evaluator.py
│     ├─ experiments/
│     │  ├─ runner.py
│     │  ├─ tuning.py
│     │  └─ tracking.py
│     ├─ analysis/
│     │  ├─ protocols/
│     │  ├─ aggregation.py
│     │  └─ plotting.py
│     └─ utils/
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ smoke/
│  ├─ regression/
│  └─ fixtures/
├─ scripts/
│  ├─ preprocess/
│  └─ maintenance/
├─ docs/
│  ├─ architecture.md
│  ├─ data.md
│  ├─ models.md
│  ├─ experiments.md
│  └─ migration.md
├─ data/
│  ├─ README.md
│  ├─ raw/
│  └─ processed/
├─ outputs/                 # 默认忽略，不提交
└─ reports/
   └─ figures/              # 仅保留明确需要版本化的最终图
```

### 5.1 模块职责

| 模块 | 只负责 | 不负责 |
|---|---|---|
| `config` | 加载、合并、校验、序列化配置 | 构建模型和执行训练 |
| `data` | 数据读取、batch 组装、mask、schema | 模型选择和指标汇总 |
| `models` | 张量到张量的前向与模型损失项 | CLI、日志路径、optimizer、checkpoint |
| `training` | 训练循环、优化器、调度器、early stop、checkpoint | 数据预处理和论文绘图 |
| `evaluation` | 指标计算和预测汇总 | 超参数搜索策略 |
| `experiments` | 组合配置、运行编排、追踪与结果落盘 | 复制模型实现 |
| `analysis`/`reports` | 读取标准结果并分析/绘图 | 维护第二套训练模型代码 |
| `cli` | 参数解析和调用应用服务 | 在 import 时执行副作用 |

### 5.2 统一数据契约

目标 batch 至少包含以下语义，具体实现可使用冻结 dataclass：

```text
Batch
├─ user_features:  Mapping[str, Tensor]
├─ item_features:  Mapping[str, Tensor]
├─ history_features: Mapping[str, Tensor]
├─ history_mask:   BoolTensor[B, L]
├─ labels:         FloatTensor[B]
└─ metadata:       sample_id / domain / optional fields
```

约束：

- label 和 logits 的目标形状统一为 `[B]`；不依赖隐式广播。
- ID 使用 `torch.long`，连续特征使用明确的浮点 dtype，mask 使用 `torch.bool`。
- padding ID、最小 item ID、ID offset 必须由数据配置或 manifest 定义，不在 loader 内硬编码。
- loader 负责产生 mask；模型不得通过“向量是否全零”猜测 padding，除非模型论文明确如此且有测试。
- TensorFlow 仅能存在于 TFRecord 读取适配层；核心训练接口只接收 PyTorch tensor。

### 5.3 AntM2C 数据链路目标

AntM2C 是第二阶段数据整改的首要对象。当前 4608 维 `text_features` 实际按固定顺序打包了：

```text
service + query + bill + item_entity_names + item_title + log_time
```

现有 loader 依赖 `[6, 768]` reshape 和固定下标切出 `item_title`，这是必须移除的隐式协议。目标数据布局按语义拆分：

```text
interactions/{split}
├─ event_id / user_id / item_id / timestamp / label / scene
├─ history_item_ids[L]
└─ interaction_or_user_context
   ├─ service_text
   ├─ query_text
   ├─ bill_text
   └─ time_context

item_feature_store
├─ item_index / original_item_id
├─ title_text
├─ entity_text
└─ image
```

字段归属必须先按数据集语义确认；例如 `item_entity_names` 属于 item 还是交互上下文，不能继续靠它在拼接数组中的位置决定。确认后写入 schema 和数据文档。

硬性验收条件：

- interaction 记录不再重复写入可由 `item_id` 查询的 item title/image/entity 特征。
- target item 和 history item 都通过同一 `item_index -> item_feature_store` 映射直接 gather。
- loader 不出现 `reshape([6, 768])`、`text_full[4]` 或重新拼接其余切片的逻辑。
- 每个 split 使用自己的交互级 embedding，禁止 val/test 复用或索引 train embedding 变量。
- history 构造基于稳定 `event_id + timestamp` 单次顺序扫描，不再使用逐行 `items.index(item_id)`。
- BERT/CLIP 每个 worker 只加载一次；文本和图像都是真正的 batch 推理，支持断点续跑。
- item 特征去重提取；缺失图片、空文本和异常样本写入 manifest，不以裸异常静默跳过。
- ID 映射、padding、OOV、split 边界和是否允许静态 item metadata 跨 split 共享都写入数据协议。
- 序列化路径、文件名和 loader 查找规则一致；生成后自动校验样本数、维度、hash 和抽样内容。

具体输出格式（继续使用 TFRecord，或改为 Parquet/Arrow + NumPy/memmap 特征表）应在真实 Linux 服务器上做吞吐与空间基准后决定；无论选择哪种格式，都必须保持上述语义分离。

### 5.4 单一多模态基类目标

多模态模型主干最终只保留 `BaseSeqModel`。`BaseModel` 进入弃用流程，所有当前继承 `BaseModel` 的模型迁移到统一输入契约。

这不意味着所有模型都必须改成 DIN/Transformer 一类的序列架构：

- 原本使用 `BaseModel` 的模型，在公共输入阶段先对每个历史模态执行 pooling，将 `[B, L, D]` 变为 `[B, D]`，再按原模型结构进行融合和预测。
- 原本使用 `BaseSeqModel` 且需要逐位置交互的模型，保留 `[B, L, D]` 与 `history_mask`，由模型或 sequence-aware 公共组件处理。
- 两类模型共享同一个 `Batch`、投影、mask、设备和输出协议，只通过显式 capability/config 区分 `pooled_history` 与 `sequence_tokens`。
- pooling 是可配置公共组件，不得在各模型内重复硬编码 `mean`。
- 基类只提供通用编码和契约，不负责 optimizer、训练循环、日志和文件写入。

当前 `BaseModel` 子类至少包括 DNN/DNN-MM、DCN、DeepFM、AutoInt、LMF、MTFN、SimCEN、MB、PAMD；迁移时逐个建立固定 fixture 回归，确认“先 pool 再进入原结构”没有改变非序列模型语义。

### 5.5 可组合模态处理管线

统一管线的逻辑顺序为：

```text
raw features
  -> modality-specific projection
  -> padding/missing mask
  -> modality-specific sequence pooling（按模型需要）
  -> modality selection
  -> configurable fusion
  -> optional dimension adapter
  -> model-specific head
```

目标配置应支持按分支、按模态选择策略，例如：

```yaml
modal_pipeline:
  target:
    modalities: [id, image, text]
    fusion: {name: cat}
  history:
    topology: pool_then_fuse
    modalities: [id, image, text]
    pooling:
      id: {name: din}
      image: {name: mean}
      text: {name: attention}
    fusion: {name: maf}
```

“可自由选择”不等于允许运行时盲目组合。每个 pooling/fusion 组件必须声明输入 rank、mask 支持、是否需要 target、最少模态数、输出维度和辅助损失；配置加载阶段拒绝不兼容组合。

### 5.6 统一模型输出契约

目标输出使用 typed `ModelOutput`，至少包含：

```text
ModelOutput
├─ logits: Tensor[B]
├─ auxiliary_losses: Mapping[str, scalar Tensor]
└─ representations: Optional[Mapping[str, Tensor]]
```

约束：

- 模型不在构造函数中创建 optimizer、写日志、保存文件或选择设备。
- 模型不原地修改输入。
- 每个辅助损失必须有稳定名称、标量形状、权重来源和单元测试。
- `forward` 只返回 logits 与训练所需附加量，不计算 sklearn 指标。
- 模型注册名称使用稳定的全小写 snake_case；Python 类名使用 PascalCase。

### 5.7 统一运行产物

每个实验必须写入独立目录：

```text
outputs/{experiment_name}/{dataset}/{model}/{run_id}/
├─ resolved_config.yaml
├─ run_metadata.json
├─ metrics.jsonl
├─ summary.json
├─ checkpoints/
│  ├─ best.pt
│  └─ last.pt                # 可选
└─ run.log
```

`run_id` 至少应包含时间戳和配置短哈希；并行运行不得共享目录。`run_metadata.json` 至少记录：代码版本、Python/框架版本、OS、设备、数据版本/指纹、seed、启动命令、开始结束时间和状态。

---

## 6. 配置治理规范

### 6.1 配置分层

配置按以下顺序合并，后者覆盖前者：

```text
训练默认值 < 数据集配置 < 模型配置 < 实验配置 < CLI 显式覆盖
```

解析后必须生成 `resolved_config.yaml`，并对其计算稳定哈希。代码只消费解析并校验后的配置对象，不在运行中随意修改原始 dict。

### 6.2 配置要求

- 配置键统一 snake_case，弃用 `multimodel_num` 等拼写错误或语义不清名称。
- 同一参数只有一个权威来源；不得同时维护 `best_param.yaml` 和 `best_params.yaml`。
- 数据集特定参数放到明确的 dataset override 中，不复制整段公共参数。
- 路径使用 `pathlib.Path`，相对路径相对于项目根或配置文件解析，规则必须唯一。
- 个人数据路径放在被忽略的 `configs/local/paths.yaml`；仓库只提交 example。
- YAML 加载后必须校验必填项、类型、范围、未知键和跨字段约束。
- 禁止自动把 test 指标写回可执行配置。
- “最优参数”必须附带产生它的 experiment ID、验证指标、seed 集合和数据版本。

### 6.3 配置命名建议

- `configs/datasets/{dataset}.yaml`
- `configs/models/{model}.yaml`
- `configs/training/default.yaml`
- `configs/experiments/{experiment}.yaml`
- `configs/local/paths.example.yaml`
- 生成的搜索结果放 `outputs/...`，经审查后再人工提升为正式模型配置。

---

## 7. 代码规范

### 7.1 Python 基线

- 首个开源版本以 Linux 服务器快照中的 Python 3.8.5 为实验兼容基线；最终 `requires-python` 范围由 Linux 安装与 CI 实测决定。
- Python 语法与依赖范围以 Linux `bm` 的 Python 3.8 实测为准，不使用 3.9+ 专属语法。
- 源码编码：UTF-8。
- 新文件换行：LF；避免无意义地改写历史文件换行。
- 行宽：100。
- 格式化和 lint：Ruff；配置统一放在 `pyproject.toml`。
- 测试：pytest。
- 新公共核心模块必须提供类型标注；mypy 先覆盖新核心模块，再逐步扩大。

### 7.2 命名

| 对象 | 规范 | 示例 |
|---|---|---|
| 文件/模块 | snake_case | `base_model.py`, `tiktok_loader.py` |
| 类 | PascalCase | `ExperimentRunner` |
| 函数/变量 | snake_case | `evaluate_model` |
| 常量 | UPPER_SNAKE_CASE | `DEFAULT_SEED` |
| CLI 选项 | kebab-case 优先 | `--dataset-name` |
| 配置键 | snake_case | `batch_size` |

历史名称的改造必须有兼容映射或迁移说明。`evalate` 应迁移为 `evaluate`；旧名称可短期发出 deprecation warning，不允许永久双实现。

### 7.3 导入与依赖方向

- 统一从 `mmctr...` 绝对导入。
- 禁止 `from utils import ...`、`from models import ...` 等易冲突顶层导入。
- 禁止通配符导入。
- 禁止循环依赖；底层模块不能导入 CLI 或 experiment runner。
- 可选重依赖应在明确适配层延迟导入，并提供清晰错误信息。

建议依赖方向：

```text
cli -> experiments/analysis -> training/evaluation -> models + data -> config/utils
```

### 7.4 函数与类

- 单一职责，避免继续扩大超长脚本。
- 公共函数必须标注参数、返回值和异常语义。
- docstring 解释契约、形状、单位和非显然约束，不复述代码。
- 不使用可变默认参数。
- 不依赖全局解析后的 `args`。
- 不用 `assert` 验证用户输入；使用带上下文的异常。
- 不用无维度的 `squeeze()` 处理 batch tensor；显式指定维度。
- 配置和输入映射默认视为只读，需要修改时先复制。

### 7.5 日志与异常

- 库代码使用 `logging`，CLI 决定输出格式；临时 `print` 不进入核心模块。
- 日志必须包含 run ID、model、dataset、seed，便于并行排错。
- 捕获异常时记录任务上下文和 traceback；不能只打印一句后继续制造不完整结果。
- 任务失败必须在结构化结果中标记 `FAILED`，进程返回非零退出码。
- 不记录密钥、个人路径、隐私数据或完整样本内容。

### 7.6 路径和文件写入

- 使用 `pathlib.Path`，入口启动时一次性解析项目根和输出根。
- 所有写入位置必须由配置或 runner 提供，模型本身不拼接路径。
- 原子写入关键 YAML/JSON：先写临时文件，校验成功后替换。
- checkpoint 文件名和目录必须包含运行隔离维度。
- CSV/JSON schema 必须版本化，禁止不同脚本向同名文件追加不同结构。

---

## 8. Benchmark 科研规范

### 8.1 实验协议

每个正式结果必须明确：

- 数据集名称、版本、切分方法和 manifest/hash。
- 模型名称、实现版本、论文来源和必要偏差说明。
- 输入模态及特征提取器版本。
- 随机种子集合。
- batch size、optimizer、学习率、正则、最大 epoch 和 early-stop 规则。
- 超参数搜索空间、预算、选择指标和搜索种子。
- 硬件和关键软件版本。
- 参数量、训练时间和推理时间的测量协议。

### 8.2 模型选择

- early stopping 只看验证集。
- 超参数比较只看验证集主指标；并列时按预先声明的次指标或复杂度规则处理。
- 搜索完成后冻结配置，再运行测试集。
- test 结果不可参与下一轮配置选择；如发生，必须声明该结果已污染并重新建立盲测集。

### 8.3 指标

- CTR 默认主指标：ROC-AUC；次指标：LogLoss。新增指标需说明方向和计算实现。
- 对单类别 split、NaN/Inf、空 batch、极端概率和 shape 不一致进行显式检查。
- 指标输入统一使用概率或 logits，接口中必须写清，禁止调用端自行猜测。
- 多种子汇总保留原始结果，并报告均值、标准差和有效运行数。
- 效率比较必须固定 batch size、设备和计时区间；CUDA 计时前后同步，并记录 warm-up。

### 8.4 公平比较

- 同类模型尽量共享数据处理、切分、训练预算、early stop 和评估实现。
- 论文特定训练策略必须显式配置，不能隐藏在模型 forward 或特殊 Trainer 中。
- 预训练模型和外部特征的额外成本必须单独说明。
- 模态缺失、冷启动、few-shot/zero-shot 数据必须验证集合定义和交集约束。
- 分析代码只能消费标准运行产物；不得通过复制模型文件形成“分析版模型”。

---

## 9. 数据、依赖与产物规范

### 9.1 数据

- `data/raw/` 和 `data/processed/` 默认忽略，只保留 `data/README.md`、manifest 和小型合成 fixture。
- 每个数据集提供来源、许可、下载方式、目录结构、字段说明和预处理命令。
- 预处理步骤必须可重复执行，输入输出均有 schema 和统计校验。
- 处理后 manifest 至少记录文件、大小、hash、样本数、正例率、用户/物品数、模态维度和生成配置。
- 不提交隐私数据；日志中不得输出可识别用户或完整原始文本。

### 9.2 依赖

- 使用 `pyproject.toml` 作为包与工具配置的权威入口。
- 核心、数据处理、分析、开发依赖应拆分为明确 optional groups。
- Linux 服务器是唯一兼容性验证环境；后续不再维护 Windows 验证门禁。
- 保留 `bm_env.yml` 作为现有成功运行环境快照；发布时去掉机器 `prefix`，并区分“可移植安装文件”和“精确服务器快照”。
- 不同时维护空 `requrements.txt` 和另一套隐式依赖。
- 深度学习框架/CUDA 版本调整必须单独任务处理，并记录兼容矩阵和 smoke 结果。

### 9.3 Git 与生成物

在维护者确认后建立或接管 Git 基线。`.gitignore` 至少覆盖：

- `__pycache__/`、`*.py[cod]`
- `*.egg-info/`、`build/`、`dist/`
- `.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`
- 本地 IDE/系统文件
- `data/raw/**`、`data/processed/**`（保留说明/manifest）
- `outputs/**`、checkpoint、运行日志、调参临时目录
- 本地路径配置和环境秘密

论文最终图若需要版本化，放入 `reports/figures/`，并保留生成脚本与数据来源。中间图和缓存不放 `src/`。

---

## 10. 测试策略与质量门禁

### 10.1 测试层级

| 层级 | 范围 | 默认设备 | 目标 |
|---|---|---|---|
| Unit | config、registry、metrics、layers、shape、mask | CPU | 秒级反馈 |
| Integration | loader → model → loss → optimizer 单步 | CPU | 验证模块契约 |
| Smoke | 每个模型家族至少一个模型，小型合成数据 | CPU，GPU 可选 | 验证入口可运行 |
| Regression | 重构前后固定 fixture 的 logits/loss/metric | CPU | 防止语义漂移 |
| Full benchmark | 真实数据、多 seed、GPU | 显式触发 | 论文级验证 |

默认测试不得依赖真实数据、网络、个人目录或 GPU。

### 10.2 最低测试矩阵

- 三个 dataset adapter 的 schema/shape 测试。
- 迁移期间为 `BaseModel` 兼容路径和 `BaseSeqModel` 各保留一个训练单步；第二阶段结束后只保留统一 `BaseSeqModel` 路径。
- 所有 fusion layer 的输出 shape、反向传播和异常输入测试。
- batch size=1、全 padding history、缺失模态、单类别标签等边界测试。
- model/dataset registry 名称唯一性和配置可构建测试。
- checkpoint 隔离、save/load 一致性和 resume 测试。
- 调参 objective 无权访问 test split 的防回归测试。
- 每个正式模型至少有 constructor + forward smoke test。

### 10.3 分阶段门禁

Linux 服务器是完整门禁执行环境，以下占位符必须替换为实际绝对解释器路径：

```bash
<SERVER_ENV>/bin/python -m ruff format --check .
<SERVER_ENV>/bin/python -m ruff check .
<SERVER_ENV>/bin/python -m pytest tests/unit tests/smoke
<SERVER_ENV>/bin/python -m mypy src/mmctr
<SERVER_ENV>/bin/python -m pytest --cov=mmctr --cov-report=term-missing
<SERVER_ENV>/bin/python -m build
```

工具尚未安装时，不得伪造通过状态；应将任务标为 `BLOCKED` 或记录“未执行及原因”。安装依赖需遵循环境授权规则。

### 10.4 Definition of Done

一个任务只有同时满足以下条件才能标记 `DONE`：

- 实现范围与任务描述一致，没有未登记的扩张。
- 代码、配置、文档和迁移说明同步更新。
- 对应自动化测试已新增或更新。
- 约定门禁通过，或明确记录无法执行的外部原因并经维护者接受。
- 没有写入个人路径、真实数据、缓存或共享 checkpoint。
- 若改变公共接口，已有兼容策略或清晰迁移说明。
- 若影响实验结果，已提供基线对比和科研协议说明。
- 进度表包含负责人、状态、验证证据和完成日期。

---

## 11. 四大阶段改造路线图

### 第一阶段：开源发布与工程基线（S1）

目标：先把现有研究代码整理为可公开、可安装、可追踪、可验证的工程基线，为后续算法级整改提供保护网。

主要工作：

- 建立 Git 基线、`.gitignore` 和生成物边界，清理源码树内缓存、egg-info 与中间图表。
- 完成 README、LICENSE、CITATION、CONTRIBUTING、数据许可/下载说明和模型来源表。
- 保留 Linux server snapshot，同时提供无绝对 prefix 的开源环境说明和 Linux 安装验证。
- 建立 `pyproject.toml`、`src/mmctr` 包命名空间、统一 CLI 和配置加载入口。
- 消除依赖当前工作目录、个人绝对路径、import-time argparse 和长 `sys.path` 注入。
- 建立合成数据 fixture、静态检查、CPU unit/smoke，以及 Linux 服务器回归命令。
- 先修复或隔离 checkpoint 覆盖、run 目录冲突、test-set 调参泄漏等 P0 问题。
- 建立重构前行为基线；无法在当前数据/依赖下验证的模型明确标记，不伪造通过。

第一阶段不大改模型公式和数据语义，只建立安全边界。退出条件：外部用户可按 README 在 Linux 创建环境、安装包、列出模型/数据集并运行合成 smoke；正式实验不会互相覆盖或使用 test 调参。

### 第二阶段：数据处理与模型主干整改（S2）

目标：重建统一数据语义和模型输入主干，重点解决 AntM2C 预处理与双基类分裂。

#### S2-A：数据处理整改

通用目标：

- 三个数据集统一 `Batch`、mask、dtype、padding、ID 映射和 manifest 契约。
- 预处理改为可配置、幂等、可断点续跑的 CLI stage；每一步有输入/输出 schema 和统计校验。
- interaction feature、user/context feature、item feature 分层存储，避免可查询 item 特征按交互重复写入。
- loader 只做读取、索引和 batch 组装，不再承担固定下标切片、隐式语义拆包或特征提取。

AntM2C 专项目标：

1. 为每条交互生成稳定 `event_id`，一次按用户/时间扫描构造历史，替代 `progress_apply + items.index`。
2. 明确 service/query/bill/item_entity/title/log_time 的语义归属，拆成具名字段。
3. item title、item entity、image 建立去重 item feature store；target/history 都按 `item_index` 直接 gather。
4. 删除 4608 维拼接和 loader `[6,768]` 切片协议，item 模态提取不再需要切片。
5. BERT/CLIP 模型单次加载、真正 batch 推理；提供进度、断点、缺失样本清单和输出 hash。
6. train/val/test 使用正确的 split 级交互 embedding；修复当前统一引用 train embedding 的风险。
7. 在 Linux 服务器比较 TFRecord 与候选 Arrow/Parquet + NumPy/memmap 方案的吞吐、空间和 CPU/GPU 等待时间后再定最终格式。
8. 输出 manifest 和抽样审计，验证时间切分、历史无未来事件、ID/OOV、样本数、正例率和模态维度。

#### S2-B：模型主干整改

- 多模态模型只保留一个 `BaseSeqModel`；`BaseModel` 标记弃用并在本阶段结束前移除正式依赖。
- 统一基类接收 `user_features/item_features/history_features/history_mask`。
- 原 `BaseModel` 子类走 `pooled_history` 路径：公共层先对每个序列模态 pooling，再传入原有模型主体。
- 真正序列模型走 `sequence_tokens` 路径：保留时间维和 mask，不被提前 pooling。
- 本阶段先使用与旧实现等价的兼容 pooling/fusion preset，保证迁移可回归；灵活组合能力放到第三阶段。
- optimizer、device、训练循环、early stop、checkpoint、evaluate 从模型基类移到统一 training engine。
- 按“基础 CTR → 简单多模态 → 复杂序列/辅助损失 → 新模型 → 量化模型”顺序迁移。

第二阶段退出条件：三个数据集产生统一 batch；AntM2C loader 无固定文本切片；所有正式模型均通过统一 `BaseSeqModel` 输入主干，正式代码不再依赖 `BaseModel`；原 `BaseModel` 模型经 pooling 后的固定 fixture 行为与迁移前一致。

### 第三阶段：模型公共组件与自由组合（S3）

目标：把各模型重复实现的投影、pooling、fusion、mask 和维度适配收敛为可注册、可校验、可按模态配置的公共组件。

主要工作：

- 统一 pooling 接口：`forward(sequence, mask, target=None) -> pooled`，收敛 mean/max/sum/DIN/attention/cross-attention 等实现。
- 统一 fusion 接口：接收具名模态映射，显式返回输出维度和辅助损失，收敛 cat/add/mean/MAF/LMF/MTFN/FQ-Former/DTA/GMMF/DMF 等实现。
- 为组件建立 registry 和 capability 元数据：输入 rank、是否 sequence-aware、mask/target 要求、模态数限制、输出维度。
- target、history、user 三个分支可分别选择参与模态和 fusion；history 的每个模态可分别选择 pooling。
- 支持明确的 `pool_then_fuse`、`fuse_then_pool`、`sequence_fusion` 拓扑；不在模型里手写顺序分支。
- 统一 projection、missing-modality mask、dimension adapter 和 auxiliary-loss 汇总。
- 每个模型只保留模型特有结构，并提供一个默认 pipeline preset 复现原论文/原实现。
- 建立组件组合矩阵测试，配置阶段拒绝不兼容组合，避免“可配置但运行时才炸”。

第三阶段退出条件：新增 pooling/fusion 不需要修改现有模型；同一模型可仅通过配置更换不同模态的 pooling 与各分支 fusion；默认 preset 通过回归测试。

### 第四阶段：实验分析体系改造（S4）

目标：让所有分析实验共享正式模型、统一 runner 和可信结果协议，分析代码只表达“实验变量”，不再复制训练实现。

主要工作：

- 建立统一 experiment/analysis runner，负责任务矩阵、Linux GPU 调度、seed、失败重试、resume 和唯一输出目录。
- fusion analysis 使用第三阶段 pipeline 配置，不保留 17 个同名模型副本。
- alignment analysis 通过训练 hook/auxiliary loss 配置注入，不维护特殊 Trainer 和模型复制。
- modal robustness 合并旧/new 脚本，用标准 batch transform 描述模态缺失、随机丢弃和比例。
- cold-start/few-shot/zero-shot 使用版本化 split protocol，并自动验证用户/物品集合约束。
- efficiency analysis 统一参数量、显存、warm-up、CUDA synchronize、训练/推理计时与硬件记录。
- tuner 只根据 validation 选择配置，完整保存 trial；配置冻结后才评估 test。
- 结果统一写 JSONL/JSON schema；aggregation 负责多 seed `mean ± std` 和失败数，plotting 只读取标准结果。
- 移除绘图脚本中的硬编码实验数字；最终图保存生成配置、输入结果 hash 和脚本版本。

第四阶段退出条件：冷启动、鲁棒性、对齐、融合和效率实验都可由配置启动；分析目录不包含第二套生产模型/Trainer；同一结果可追溯到代码、数据、配置、seed 和服务器环境。

---

## 12. 多 agent 协作流程

### 12.1 领取任务

每个 agent 开始前在任务表中填写：

- `Owner`：agent 标识。
- `Status`：改为 `IN_PROGRESS`。
- `Files`：预计修改的文件或目录。
- `Evidence`：计划执行的测试/对比。

任务依赖未完成时不得抢跑公共接口。可以先做只读调研，并把结果登记在 Notes。

### 12.2 文件冲突控制

- `REFACTORING_PLAN.md`、`pyproject.toml`、registry、公共 schema 和根配置属于高冲突文件。
- 同时改造时按“S1 开源基线 → S2 数据/模型主干 → S3 公共组件 → S4 实验分析”的依赖顺序合并。
- 单个 agent 尽量只迁移一个模型家族或一个横切模块。
- 批量重命名必须单独任务执行，且在其他依赖任务暂停时进行。

### 12.3 交付说明模板

每个任务完成时在 Notes 或交付消息中提供：

```text
Task ID:
Summary:
Changed files:
Behavior changes:
Compatibility/migration:
Validation commands:
Validation results:
Metric comparison (if applicable):
Remaining risks:
Follow-up tasks:
```

### 12.4 阻塞处理

`BLOCKED` 必须说明：

- 已尝试的安全检查。
- 精确错误或缺失条件。
- 是否与解释器、依赖、数据、GPU、权限或接口决策有关。
- 解除阻塞所需的维护者决策或外部资源。

不得通过切换未授权 Python 环境、伪造数据或跳过关键测试绕开阻塞。

---

## 13. 改造任务总表

### 13.1 里程碑总览

| Milestone | 状态 | 当前进度说明 | 退出证据 |
|---|---|---|---|
| S1 开源发布与工程基线 | `DONE` | 包元数据、Linux `bm` 依赖/CUDA 基线、`mmctr` 公共命名空间、统一 CLI、严格 training/本地路径配置层、主要公开文档、合成 CPU/TFRecord smoke、首个数值回归基线、tuner 科研红线修复、主训练运行目录隔离、分阶段 Linux QA 门禁、首次 GitHub clean-runner CI 及 Apache-2.0 软件许可均已完成 | Linux 安装、公开文档、P0 修复、smoke baseline |
| S2 数据处理与模型主干 | `DONE` | 统一契约、三个真实 canonical-v1、AntM2C 11,190,985 events/173.23 GB store、23 个 CTR 模型、legacy 物理收敛及 TikTok 正式量化工件均完成；三份原始 AntM2C 事件、图片归档与 BERT/ChineseCLIP checkpoint 已做全源 hash，30K raw history replay、真实 item store、V100 encoder/restartable extraction 均通过 | 无切片数据链路、统一 Batch、单一 BaseSeqModel；真实 raw replay、encoder 实测和 Linux 门禁 |
| S3 模型公共组件 | `DONE` | projection/mask/dimension adapter、六类 pooling/fusion API/registry，以及按 target/history/user 分支配置和三种 history topology 的 modal pipeline 已通过 Linux 门禁；23 个正式模型已有完整 preset 覆盖决策，其中 13 个严格等价公共 preset 完成数值回归，10 个按 ADR-032 保留具名 paper-private 默认边界 | pooling/fusion 可按模态配置且默认 preset 回归通过 |
| S4 实验分析体系 | `DONE` | experiment matrix/device queue/失败隔离/resume、validation-only tuner、cold-start、fusion、alignment、robustness、efficiency 与 plotting 均已收敛到 canonical 配置/runner/result 协议；17 个融合模型副本、第二套 analysis Trainer/GPU subprocess/4608 TFRecord 脚本和硬编码绘图脚本已物理删除 | 无模型复制、统一 runner/result schema、五类分析可配置运行 |

> 当前结论：S1–S4 均已完成。AntM2C 原始事件/history/item/BERT/ChineseCLIP 链路已有真实 hash 与 V100 replay 证据，现有全量 canonical-v1 继续作为正式训练版本；MicroLens/TikTok 的 manifest source hashes 已逐文件与只读参考匹配，并明确其模态输入是上游预提取特征。23 个正式模型、量化工件、五类分析及 13 个严格等价 preset 均通过 canonical 门禁；其余 10 个模型按 ADR-032 保留具名 paper-private 默认边界。顶层 `data/trainers/utils` 与旧 analysis Python 运行代码已物理删除，wheel 只发布 `mmctr` 运行时。

### 13.2 可领取任务

| ID | Priority | Task | Depends on | Status | Owner | Files | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| GOV-000 | P0 | 仓库盘点并建立全局改造方案 | - | `DONE` | Codex | `REFACTORING_PLAN.md` | 117 files、16,679 LOC、AST 0 failures；2026-07-31 |
| GOV-001 | P0 | 建立/确认 Git 基线，保护现有用户改动 | GOV-000 | `DONE` | Codex (`/root`) | repo metadata、`REFACTORING_PLAN.md` | 改造前 267 个文件已纳入本地可恢复快照；公开分支从远程初始提交生成干净历史；2026-07-31 |
| GOV-002 | P0 | 添加 `.gitignore`，治理 pyc/egg-info/outputs | GOV-001 | `DONE` | Codex (`/root`) | `.gitignore`、generated files、`REFACTORING_PLAN.md` | 移除 105 个 `.pyc`、9 个 `__pycache__` 和 1 个 `egg-info`；本地 `.vscode/settings.json` 保留但取消跟踪；19 个论文图表保留；`git ls-files -ci --exclude-standard` 为 0；2026-07-31 |
| PUB-001 | P0 | 公开推送前净化个人路径与历史生成物，并接入目标远程 | GOV-002 | `DONE` | Codex (`/root`) | hard-coded path files、Git refs/history、`REFACTORING_PLAN.md` | 远程 `origin/main` 非强制前进至 `222591b`；当前树个人路径/常见密钥扫描均为 0；公开历史 pyc/egg-info/IDE 对象为 0；6 个改动脚本 AST 与 4 个 YAML 静态检查通过；原本地历史保留于 `local/refactor-bootstrap`；2026-07-31 |
| ENV-001 | P0 | 审计并保留 Linux server snapshot，生成可移植环境说明 | GOV-001 | `DONE` | Codex (`/root`) | `bm_env.yml`、`environment.yml`、`pyproject.toml`、`docs/environment.md`、`AGENTS.md`、`tests/smoke/test_tensorflow_tfrecord.py`、`REFACTORING_PLAN.md` | 原始 snapshot 保留，现有 `bm` 未新建环境；解释器 `<BM_PYTHON>`（3.8.20，绝对路径见 ignored private evidence）。依赖对齐为 TensorFlow/Keras 2.12.1/2.12.0、h5py 3.10.0、typing-extensions 4.5.0、JAX/JAXlib 0.4.13 等，保留 NumPy 1.23.5 与 PyTorch 1.13.1+cu117；`pip check` 无冲突，JAX CPU 与三 loader import/TFRecord round-trip 通过。非沙箱实测驱动 535.183.01、CUDA 12.2、8×V100-32GB，PyTorch 前反向/同步及 TensorFlow GPU matmul 均通过。追加 CI 合约测试后的最终 Ruff 34 files、mypy 14 files、strict unit+smoke 25 passed、全量 35 passed/83.80%；隔离 wheel 137 files/268965 bytes/SHA-256 `c279ede0278cdd3871aefdaf08b3040fac4ddd90b4b849e913ce7e18388d3413`；2026-07-31 |
| ENV-002 | P0 | 建立 `pyproject.toml`、依赖分组和 Python 兼容范围 | ENV-001 | `DONE` | Codex (`/root`) | `pyproject.toml`、`setup.py`、`requrements.txt`、`docs/environment.md`、`REFACTORING_PLAN.md` | 新增 PEP 517/621 元数据与 6 个 optional groups，声明 Python `>=3.8,<3.9`，删除空的错误拼写依赖文件；Windows 3.12.11：TOML、118 AST、20 legacy packages、`setup.py --name` 通过；常规 wheel 正确拒绝 3.12，`--ignore-requires-python --no-deps --no-build-isolation` 构建成功，wheel 120 文件且无缓存；临时产物已清理；Linux 门禁按 ADR-010 延期；2026-07-31 |
| OSS-001 | P0 | README/Quick Start、LICENSE、CITATION、CONTRIBUTING | ENV-002 | `DONE` | Codex (`/root`) | `README.md`、`CONTRIBUTING.md`、`CITATION.cff`、`LICENSE`、`pyproject.toml`、`MANIFEST.in`、metadata tests、`REFACTORING_PLAN.md` | 维护者明确选择 Apache-2.0；标准许可证全文、PEP 621 license file、OSI classifier、CFF SPDX 标识和代码/三方数据许可边界已落地。TDD 由缺少 LICENSE 的 1 failure 转为 5 passed；Linux pip/Ruff/mypy、194 unit+smoke、208 full/86.78%、build 与仓库外安装通过，wheel/sdist 均恰含一份 LICENSE，安装元数据含 Apache classifier/text；2026-08-04。 |
| OSS-002 | P0 | 数据来源、许可、下载、目录与模型引用清单 | OSS-001 | `DONE` | Codex (`/root`) | `data/README.md`、`docs/references.md`、`tests/unit/test_open_source_metadata.py`、README、`REFACTORING_PLAN.md` | 三套数据均记录原提供方/论文/获取方式/许可核验状态/本地 provenance 要求，未核验授权时明确禁止镜像；23 个正式 registry 全部映射为 paper adaptation、fusion adaptation 或 benchmark variant，RQ/PSRQ 引用边界另列。TDD 首轮 2 failures（引用文档/README 链接缺失），实现后 2 passed；不以论文或代码许可证推定数据再分发；2026-08-04。 |
| SCI-001 | P0 | 隔离并修复现有 tuner 的 test-set 泄漏 | TEST-001 | `DONE` | Codex (`/root`) | `src/scripts/Tuner.py`、`src/scripts/Codebook_Tuner.py`、`src/utils/tuning_protocol.py`、`tests/`、`REFACTORING_PLAN.md` | 两处搜索流程均改用共享 validation-only evaluator，结果字段改为 `val_auc`/`val_loss`，保留严格更高 AUC 胜出的旧比较规则；三个 tuner 的 test 指标泄漏扫描为 0；Windows 指定解释器完成 `src`/`tests` compileall，最终 `unittest discover` 8 tests/4.563s 全通过；测试后 13 个缓存目录已清理；Linux 验证按 ADR-010 延期；2026-07-31 |
| RUN-001 | P0 | 设计唯一 run ID 和隔离输出/checkpoint | GOV-001 | `DONE` | Codex (`/root`) | `src/utils/run_context.py`、`src/trainers/Trainers.py`、`src/utils/helper.py`、`config/train.yaml`、`docs/run-layout.md`、`tests/`、`REFACTORING_PLAN.md` | run ID 含 UTC 微秒时间戳、10 位配置哈希和 8 位随机熵；原子创建独立 resolved config/metadata/metrics/checkpoint/log/summary 路径，主训练初始化或运行失败均落 `failed`；32 路相同配置同微秒并发目录全部唯一，精确碰撞和路径穿越均拒绝；Windows 指定解释器完成 `src`/`tests` compileall，最终 `unittest discover` 14 tests/10.194s 全通过；25 个缓存目录已清理；tuning/analysis 全面接入留给 `TUNE-001`/`EXP-001`，Linux 验证按 ADR-010 延期；2026-07-31 |
| TEST-001 | P0 | 建立 pytest、合成 batch 和首个 CPU smoke | ENV-002 | `DONE` | Codex (`/root`) | `tests/`、`pyproject.toml`、`README.md`、`REFACTORING_PLAN.md` | 建立 pytest 可收集的 unittest 结构、确定性 ID batch、pooling unit 与 legacy registry DNN CPU smoke；Windows 3.12.11 未安装 pytest，未修改环境；`unittest discover` 4 tests/5.392s 全通过，覆盖 forward/loss/backward/optimizer；首次直接导入暴露 `BaseModel ↔ utils.helper` 循环，改按现有 registry 入口验证并登记给 `PKG-001`；12 个测试缓存目录已清理；Linux pytest 按 ADR-010 延期；2026-07-31 |
| BASE-001 | P0 | 保存重构前可获得的行为/指标基线 | TEST-001 | `DONE` | Codex (`/root`) | `tests/baselines/`、`tests/regression/`、`REFACTORING_PLAN.md` | `legacy_dnn_id_cpu_v1` 保存 schema、registry 入口、完整配置/seed/输入、Windows/Python/Torch/NumPy/sklearn 版本、4 logits、loss 与 205 参数量；容差 `1e-6`；Windows `unittest discover` 5 tests/5.263s 全通过；13 个缓存目录已清理；明确为合成行为而非论文指标，Linux 正式基线按 ADR-010 延期；2026-07-31 |
| PKG-001 | P1 | 创建 `src/mmctr` 包并迁移导入 | ENV-002, TEST-001 | `DONE` | Codex (`/root`) | `src/mmctr/`、legacy import bridge、`src/utils/helper.py`、core imports、docs、`tests/`、`REFACTORING_PLAN.md` | 新增 `mmctr.models`/`mmctr.data`/`mmctr.utils` 公共入口；helper 改为 model/data 调用时加载，直接 model-first 与 helper-first 导入顺序均通过且 DNN 类身份一致；15 个核心 helper 调用方迁至 `mmctr.utils`，legacy 直接 helper import 为 0；Windows 指定解释器完成 `src`/`tests` compileall，`unittest discover` 16 tests/21.870s 全通过；仓库外 cwd 临时安装 wheel 成功，`mmctr` 路径来自目标目录，wheel 260175 bytes、SHA-256 `5410f853c3fcb52076b3210615741e6c3c6dc895ed082b11ef9d6f67de46c524`；临时目录、2 个构建目录和 29 个缓存目录已清理；legacy 顶层包作为显式兼容桥保留，物理迁移随模型/数据任务推进；Linux 验证按 ADR-010 延期；2026-07-31 |
| PKG-002 | P0 | 物理收敛剩余顶层 runtime 包和旧数据脚本 | PKG-001, DATA-002, MODEL-BASE-002 | `DONE` | Codex (`/root`) | `src/mmctr/{training,quantization,utils,data}/`、deleted `src/{data,trainers,utils}/`、deleted legacy analysis Python、tests/docs/build、`REFACTORING_PLAN.md` | TDD 入口/目录契约由 2 failures 转 3 passed；Trainer、RQ/PSRQ、helper/run-context/tuning 全部迁入 `mmctr`，21 个旧 data loader/processor、8 个顶层 trainer/utils 文件及最后一个旧 analysis Python 已删除，19 个历史最终图保留。Ruff 143 files、mypy 87 files、197 unit+smoke、212 full/84.57%、仓库外 canonical entrypoints 全通过；93-member wheel 无 `data/trainers/utils/analysis` 顶层 runtime；2026-08-04。 |
| DIR-001 | P0 | 收敛公开目录、配置根、数据骨架与生成物边界 | PKG-002, CFG-002, PLOT-001 | `DONE` | Codex (`/root`) | `configs/`、`data/`、`outputs/`、`reports/`、deleted `config/`/`experiments/`/root packaging shims、code/tests/docs/build、`REFACTORING_PLAN.md` | 删除 `config/` 及未消费的 tuner/参数/重复 sequence 配置，建立唯一 dataset/model/training catalog；三数据集均有 tracked raw/processed placement README，mutable artifact 统一进 ignored `outputs/`，19 个最终图迁入 `reports/figures/`，删除根/src 空 package、legacy `setup.py` 与浅 utility。目录契约首轮 8 failures 后转 26 passed；最终 Linux pip/Ruff/mypy、198 unit+smoke、213 full/84.90% 及定向 24 passed；wheel 为 92 members/186,939 bytes/SHA-256 `cf8a216c3ddd957a9d250515ce43fca98fadf909620a04ed39a9504ebf665870`，sdist 为 238 members/242,442 bytes/SHA-256 `b08ac1ca0c5c7aea4b387ee8a1af98e97e2f81727c5726c9f52e58c8b1363901`，成员审计无数据 payload/checkpoint/output，仓库外安装与 Trainer/RQ/PSRQ help、23/3/2 registry 通过；2026-08-04。 |
| CFG-001 | P1 | 配置分层、typed schema 和校验 | PKG-001 | `DONE` | Codex (`/root`) | `src/mmctr/config/`、`src/trainers/Trainers.py`、`config/`、`docs/configuration.md`、`docs/legacy_tuning_history.yaml`、`tests/`、`REFACTORING_PLAN.md` | 新增 frozen `TrainingConfig`、唯一键/顶层 mapping YAML loader、项目根发现、无副作用递归 layer merge；严格覆盖必填、未知、类型、范围、optimizer 和 patience 跨字段约束，主训练通过显式 `to_dict()` 兼容边界消费且配置文件路径不依赖 cwd；`best_params.yaml` 历史 test 记录迁出 `config/`，新 legacy 输出写入 ignored `outputs/tuning/`，`best_param.yaml` 成为唯一 tracked 参数快照；6 个可执行 YAML 唯一键检查通过；Windows 指定解释器完成 `src`/`tests` compileall，最终 `unittest discover` 21 tests/20.310s 全通过，30 个缓存目录已清理；算法专属 model/data typed schema 随对应重构任务推进，tuning provenance 随 `TUNE-001` 完成；Linux 验证按 ADR-010 延期；2026-07-31 |
| CFG-002 | P0 | 消除缺失 local config 和服务器个人绝对路径 | CFG-001 | `DONE` | Codex (`/root`) | `src/mmctr/config/paths.py`、training/tuning/analysis callers、`configs/local/paths.example.yaml`、`.gitignore`、`docs/local-paths.md`、`tests/`、`REFACTORING_PLAN.md` | 新增 frozen `LocalPaths`、selected dataset catalog resolver、绝对路径/存在性/未知数据集校验；ignored `paths.yaml` < 环境变量 < 显式 legacy CLI override，canonical 路径按项目根解析且不修改输入；主训练、普通 tuner、alignment、两版 modal robustness 和 analysis trainer 全部移除缺失 local YAML；`src/` 中 `local_data.yaml`/`local_seq_data.yaml` 引用为 0，公开 config/example 个人路径扫描为 0；真实 `paths.yaml` 命中 ignore，example 不被忽略；Windows 指定解释器完成 `src`/`tests` compileall，最终 `unittest discover` 28 tests/19.096s 全通过，30 个缓存目录已清理；Linux local-path smoke 按 ADR-010 延期；2026-07-31 |
| CLI-001 | P1 | 建立统一 CLI，移除 import-time argparse | PKG-001, CFG-001 | `DONE` | Codex (`/root`) | `src/mmctr/cli/`、`src/trainers/Trainers.py`、`src/mmctr/models/`、`src/mmctr/data/`、`pyproject.toml`、`docs/cli.md`、README、`tests/`、`REFACTORING_PLAN.md` | 新增 console/module CLI 与 train/validate-config/list-models/list-datasets 子命令；CLI/catalog import 不加载 torch/TensorFlow，train 设置 5 组线程环境后才 lazy import runtime；主 Trainer 改为显式参数构造，import 不解析宿主 argv 且移除硬编码 24 线程副作用；Windows 指定解释器完成 `src`/`tests` compileall，`unittest discover` 33 tests/26.696s 全通过；仓库外临时 wheel 的 module help、dependency-light import 和 `mmctr = mmctr.cli:main` entry point 均通过，wheel 268941 bytes、SHA-256 `f03e6bf71e6bab2ebda0ff735deccad8b4ed39eba36b5a9a11a5710faf2342a8`；临时目录、2 个构建目录和 32 个缓存目录已清理；真实 train Linux/CUDA gate 按 ADR-010 延期；2026-07-31 |
| QA-001 | P1 | Ruff/pytest/mypy/coverage 分阶段门禁 | ENV-002, PKG-001 | `DONE` | Codex (`/root`) | `pyproject.toml`、`src/mmctr/`、`tests/`、`docs/quality-gates.md`、README、`REFACTORING_PLAN.md` | 已建立明确阶段边界：Ruff 约束 `src/mmctr + tests`，mypy 检查 14 个公共源文件但不递归 legacy bridge，coverage 下限 80%。最终 Linux `<BM_PYTHON>`：Ruff format 34 files 与 lint 通过，mypy 1.10.1 的 14 files 通过，unit+smoke 25 passed/7.48s，全量 pytest 35 passed/17.84s、coverage 83.80%，隔离 sdist+wheel build 通过；wheel 137 files/268965 bytes/SHA-256 `c279ede0278cdd3871aefdaf08b3040fac4ddd90b4b849e913ce7e18388d3413`。全仓 legacy 基线仍为 160 lint 问题/116 待格式化文件并已文档化；依据 ADR-016 仅以 Linux 验收；2026-07-31 |
| QA-002 | P0 | 拉取 v0.73 后在 Linux `bm` 执行全量接续门禁并修复回归 | POOL-001 | `DONE` | Codex (`/root`) | `src/mmctr/`、`tests/`、`REFACTORING_PLAN.md` | 修复 31 个格式差异、4 个 lint、23 个 mypy 问题和 AntM2C 只读 memmap tensor 安全警告；统一 DIN 兼容签名与公共 pooling 合约，量化配置转换改为明确校验。Linux `<BM_PYTHON>`：pip check、Ruff format/lint、mypy 58 files 通过；unit+smoke 115 passed，全量 125 passed/86.99%；sdist+wheel 构建通过，wheel 350184 bytes/SHA-256 `422f1e4f8b93e847c9676ac6613766a21903de7c51781c83b0e6c848d0a89059`；2026-08-03 |
| CI-001 | P1 | 建立 Linux CPU CI | QA-001 | `DONE` | Codex (`/root`) | `.github/workflows/linux-ci.yml`、`tests/unit/test_ci_workflow.py`、`docs/quality-gates.md`、`REFACTORING_PLAN.md` | 以固定 Ubuntu 22.04/Python 3.8 + PyTorch 1.13.1 CPU 执行 pip check、Ruff、mypy、unit/smoke、全量 coverage 和隔离 build，缓存/临时目录全部位于 workspace；TDD 合约测试先因 workflow 缺失为 RED，随后发现并阻止无效 YAML 单行命令，最终 YAML 解析和 runner/action/Python/命令/缓存约束通过。Linux `bm` 复核 Ruff 34 files、mypy 14 files、unit+smoke 25 passed、全量 35 passed/83.80% 及隔离 build；提交 `66db7d7` 的首次 clean GitHub Actions run `30642128515` 全部成功；2026-07-31 |
| CORE-001 | P0 | 定义统一 Batch/ModelOutput/RunResult | PKG-001, CFG-001 | `DONE` | Codex (`/root`) | `src/mmctr/core/`、`src/mmctr/__init__.py`、`tests/unit/test_core_schemas.py`、`docs/architecture.md`、`REFACTORING_PLAN.md` | 不可变核心契约、shape/dtype/batch-size、设备迁移和 legacy adapter 完成；专项测试及 `QA-002` Linux 133 项全量回归/87.11% 覆盖率通过；2026-08-03 |
| DATA-001 | P0 | 统一三个数据集 loader contract 与 manifest | CORE-001 | `DONE` | Codex (`/root`) | `src/mmctr/data/`、legacy loader compatibility boundary、`tests/unit/test_data_contracts.py`、`docs/data.md`、`REFACTORING_PLAN.md` | 统一 split/history capability、版本化 manifest 和 `CanonicalDataLoader` 完成，三类 legacy adapter contract 及 Linux 全量门禁通过；真实 AntM2C 字段/存储审计仍属后续 `ANT-*`；2026-08-03 |
| ANT-001 | P0 | 审计 AntM2C 六类文本字段语义与 feature ownership | DATA-001 | `DONE` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/`、`docs/data/antm2c.md`、`tests/unit/test_antm2c_schema.py`、真实分层 benchmark、`REFACTORING_PLAN.md` | 已固化 service/query/bill/item_entity/log_time 为 interaction context，title/image 为 item feature；6,144 条跨 split/source-shard 真实样本覆盖 2,820 item，其中 247 item 的 entity embedding 有多个版本、单 item 最多 19 个，故 item_entity 明确拒绝提升为 item ownership。稳定 event_id、原脚本午夜切分、padding/ID offset、split embedding 与未来泄漏协议已登记，schema/audit 测试及 Linux 门禁通过；2026-08-03 |
| ANT-002 | P0 | 用 event_id + 单次时序扫描重写历史构造 | ANT-001 | `DONE` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/{history,raw}.py`、raw tests/data docs、ignored validation report、`REFACTORING_PLAN.md` | 三份 raw part 全文件 SHA-256 已锚定；每 part 前 10K 的真实 replay 共 30,000 events、20,337 users、6,972 items，train/val/test 22,739/1,967/5,294，稳定 source-file/row event ID、午夜 cutoffs、严格早期正反馈、重复事件和最长 5 history 均通过泄漏审计；2026-08-04。 |
| ANT-003 | P0 | 建立 item index 和去重 item feature store | ANT-001 | `DONE` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/{item_store,raw,encoders}.py`、tests/data docs、ignored validation report、`REFACTORING_PLAN.md` | 真实 30-event tracer 形成 25 个首次出现连续 item index；V100 BERT 编码 25 个去重 title，缺失 0，target `(30,768)` 与 history `(30,5,768)` 从同一 store gather，finite 且 padding row 全零；全量 canonical item hashes/gather 继续由 ANT-007 守护；2026-08-04。 |
| ANT-004 | P1 | 重写文本/图像 batch 提取、单次模型加载和断点续跑 | ANT-003 | `DONE` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/{extraction,encoders}.py`、tests/data docs、ignored validation report、`REFACTORING_PLAN.md` | BERT/ChineseCLIP checkpoint aggregate SHA-256 为 `d14cb738...42b53`/`17173f19...96105`；GPU0 的 8-row bill replay 对旧 array 最大/均值差 `8.389354e-6`/`2.527670e-7`，GPU1 从 24.8GB tar 流式读取真实 640×640 PNG 并产出 finite `(1,512)`。16-row/1-image extraction 分别 2/1 shards，完成后 resume 均未重载 encoder；8×V100 可见；2026-08-04。 |
| ANT-005 | P0 | 重写 split 序列化和 loader，删除 4608 拼接/固定切片 | ANT-002, ANT-003, ANT-004 | `DONE` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/{array_store,canonical}.py`、`src/mmctr/core/schemas.py`、dataset registry/config、tests/data docs、真实 canonical-v1、`REFACTORING_PLAN.md` | 具名 interaction context 与共享 item store 已用于 11,190,985-event 正式数据；磁盘无 4608 packed view，loader 仅按具名五段动态提供 3840 维兼容 context，target/history 从同一 title/image store gather。正式 registry/config 切换完成；三 split 真实 `dnn_mm_seq` 均完成有限 BCE forward/backward/Adam step，全量逐数组验证通过；2026-08-04。 |
| ANT-006 | P1 | Linux 基准比较 TFRecord 与候选分层格式 | ANT-005 | `DONE` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/benchmark.py`、`tests/unit/test_antm2c_benchmark.py`、当前项目 `data/processed/antm2c/benchmark-v1/`、`docs/data/antm2c.md`、`docs/data/antm2c-format-benchmark.json`、`REFACTORING_PLAN.md` | 真实只读来源的权威 split TFRecord 为 231,139,326,681 bytes，共享 item store 434,524,416 bytes；当前项目生成 6,144 条跨 source-shard/domain 分层样本（628 MiB），扫描时显式剔除并记录 train 4/val 10 条 title-store 冲突，image 冲突 0。batch 256、每格式完整 warm-up、5 次全量 median：legacy 11,135 samples/s、named arrays 67,803 samples/s（6.09×），交互体积 126,903,807→95,408,256 bytes（-24.82%），semantic checksum 完全一致；全量外推约 174.21 GB、节约 57.36 GB。当前进程无 CUDA 且 loader 不含 H2D，GPU wait 明确记为 unavailable；真实 `dnn_mm_seq` 前反向通过。Linux `<BM_PYTHON>` Ruff 117 files、mypy 71 files、unit+smoke 163 passed、全量 173 passed/87.53%；no-isolation sdist+wheel 通过，wheel 376 KiB/SHA-256 `ff312cd7bef65eb6a32c529cbacc3e3819ff08a9d28591c9437460f81ef5049f`；2026-08-03 |
| ANT-007 | P0 | 生成全量 AntM2C canonical-v1 并切换正式 registry/config | ANT-006 | `DONE` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/canonical.py`、dataset registry/config、当前项目 `data/processed/antm2c/canonical-v1/`、数据文档、integration tests、`REFACTORING_PLAN.md` | 28/28 源文件原子/可恢复转换完成；train/val/test 为 8,536,558/721,500/1,932,927 events、1,059/91/239 shards，总计 15,284 files/173,233,549,382 bytes，保留并审计 title 冲突 16,348/4,029/1,068、image 冲突均 0。最终 fingerprint `ece3afcc876853caadd4e277421c938a59426e2e003577b1fd02b90e2d3b2aca` 同时锚定 item store 与 aggregate shard metadata；逐文件 SHA-256/bytes/shape/dtype/count 全验 711.558s 通过。三 split CPU 训练 step、V100 256-event forward/backward/Adam 及效率报告通过；Ruff 137 files、mypy 82 files、unit+smoke 179 passed、全量 189 passed/87.02%、sdist+wheel 通过，wheel 403,642 bytes/SHA-256 `01ede8a6f6dfe5e1a22d13c96bbc78cd13a59d915b4dc04e73fc3ee96b3a4d53`；无效 7.6GB skip-policy staging 已删除且不可恢复；2026-08-04。 |
| DATA-002 | P1 | 将 MicroLens/TikTok 对齐统一数据契约 | DATA-001, ANT-005 | `DONE` | Codex (`/root`) | `src/mmctr/data/datasets/arrays.py`、`src/mmctr/data/datasets/microlens/`、`src/mmctr/data/datasets/tiktok/`、dataset registry/config、数据文档、`tests/unit/test_microlens_dataset.py`、`tests/unit/test_tiktok_dataset.py`、真实 `data/processed/{microlens,tiktok}/canonical-v1/`、`REFACTORING_PLAN.md` | 已以可 memory-map 的具名 NumPy interaction arrays + 去重 item feature store 对齐 canonical `Batch`，loader 保留真实 manifest 而非降级为 legacy adapter。MicroLens 3,600,000 交互按 seed 42 确定性 8:1:1 输出 2,880,000/360,000/360,000 样本，manifest `1e28692d5b7f5c722c6621f78533a1974eef9f9b64256a6a3e51e8a27889221f`，产物 834 MiB；TikTok 严守官方 split 和因果 prefix history，以 1:5 正负比输出 357,246/18,306/36,780 样本，manifest `a908bb25f28f899bf2cf21f81483ae67d2106adea582e0e5576a2e9cb1ff05bc`，产物 50 MiB，未继承 legacy 合并三 split 后随机切分的数据泄漏协议。合成 round-trip、mask/ID/manifest/泄漏守卫通过；两套真实 loader 均以 `dnn_mm_seq` 完成有限 BCE+aux loss 前向、反向与 Adam step，loss 分别为 0.692235/0.678381；Linux `<BM_PYTHON>` Ruff 115 files、mypy 70 files、unit+smoke 160 passed、全量 170 passed/87.20%；no-isolation sdist+wheel 构建通过，wheel 369 KiB/SHA-256 `fe3a7939b5f1957141879655d783c75f23c8d57dd6aa8dc06a3e2bda4af9109b`；2026-08-03 |
| TRAIN-001 | P0 | 统一 training engine、evaluate、early stop、optimizer | CORE-001, RUN-001 | `DONE` | Codex (`/root`) | `src/mmctr/training/`、`src/mmctr/evaluation/`、`src/trainers/Trainers.py`、`src/mmctr/utils/run_context.py`、`tests/unit/test_training_engine.py`、`docs/training.md`、`REFACTORING_PLAN.md` | canonical engine、validation-only early stop、optimizer、AUC/LogLoss、原子 checkpoint、完整 resume 状态与 run metrics 完成；单步/save-load/resume/test 隔离及 Linux 全量门禁通过；2026-08-03 |
| TRAIN-002 | P0 | 建立具名参数组的交替多优化阶段协议 | TRAIN-001 | `DONE` | Codex (`/root`) | `src/mmctr/training/optimizers.py`、`src/mmctr/training/engine.py`、training public exports、Trainer composition、`tests/unit/test_training_engine.py`、`docs/training.md`、`REFACTORING_PLAN.md` | `PhasedAdam` 以互斥具名参数组保存 main/discriminator/generator 的独立 Adam state，`step_phase()` 只更新目标组；engine 显式消费 phase/start_epoch/objective 并保持 main→discriminator→generator。更新隔离（含 closure）、`epoch >= N` 日程、checkpoint state 往返、普通单优化器回归及真实 canonical GMMF 训练测试均通过；全量 Linux 门禁 166 passed/87.29%；2026-08-03 |
| MODEL-BASE-001 | P0 | 建立单一 BaseSeqModel 与 pooled/token 两种历史能力 | CORE-001, TRAIN-001 | `DONE` | Codex (`/root`) | `src/mmctr/models/base.py`、model public exports/registry、`tests/unit/test_model_base.py`、`docs/models.md`、`REFACTORING_PLAN.md` | 纯 `Batch -> ModelOutput` 基类、显式 history capability 与 mask-aware pooling 已成为唯一模型基类；过渡 adapter 已在 `MODEL-BASE-002` 完成后删除。Linux 最终全量 202 passed/85%；2026-08-04。 |
| MODEL-BASE-002 | P0 | 迁移并弃用 BaseModel 子类，先 pool 后复用原主体 | MODEL-BASE-001, DATA-001 | `DONE` | Codex (`/root`) | `src/mmctr/models/`、deleted `src/models/`/`src/scripts/`、model registry/helper/Trainer、regression/smoke/architecture tests、model/migration docs、package manifest、`REFACTORING_PLAN.md` | 23 个正式 CTR 模型均只继承 canonical `BaseSeqModel`；物理删除 37 个 legacy model Python 文件、双旧基类、adapter、3 个 tuner、2 个无效 shell runner、2 个旧 cold-start 统计脚本及 registry/helper 兼容解析元数据。DNN canonical 路径精确复现冻结 logits/loss/205 参数，GMMF 固定 seed fixture 覆盖 logits/reconstruction/discriminator/generator/1,051 参数；架构测试先 3 failures 后通过。Linux Ruff 135 files、mypy 82 files、分片全集 202 passed/85%、sdist+wheel 通过；wheel 199,967 bytes/SHA-256 `6aae71c781d3de24c2910d4af34b7a21b99b14bdb7fc7cbb22d14fb0036e296f`，清单无 `models/`、`scripts/` 或 `mmctr/models/compat.py`；2026-08-04。 |
| REG-001 | P1 | 模型/数据集 registry（fusion registry 留到 S3） | PKG-001, CFG-001 | `DONE` | Codex (`/root`) | `src/mmctr/core/registry.py`、`src/mmctr/{models,data,quantization}/registry.py`、public exports、tests、真实 manifests、`REFACTORING_PLAN.md` | 稳定 snake_case、`dnn_seq -> dnn_mm_seq` alias、capability metadata 与 lazy import 完成；23 个 CTR、2 个量化预模型、3 个数据集分别只有一份正式注册表并全部解析到 canonical 实现。legacy module/symbol 元数据及解析器已删除，唯一性、dependency-light import、三套真实 manifest 和最终 202 项全量门禁通过；2026-08-04。 |
| MOD-001 | P1 | 迁移基础 CTR 模型 | MODEL-BASE-002, REG-001 | `DONE` | Codex (`/root`) | `src/mmctr/models/baselines/`、model registry、`src/trainers/Trainers.py`、tests/docs、`REFACTORING_PLAN.md` | DNN/DCN/DeepFM/AutoInt/DIN 为纯 `Batch -> ModelOutput`；pooled 模型显式 masked mean，DIN 保留 token/mask attention，batch size 1 不降秩。DNN 以相同 seed 精确复现迁移前冻结 logits/loss/205 参数；旧类和 helper 构造入口已删除，Linux 最终 202 passed/85%；2026-08-04。 |
| MOD-002 | P1 | 迁移简单多模态模型 | MOD-001 | `DONE` | Codex (`/root`) | `src/mmctr/models/multimodal.py`、`src/mmctr/models/sequence.py`、model registry、tests/docs、`REFACTORING_PLAN.md` | DNN-MM、DNN-MM-Seq、LMF、MTFN 为纯 `Batch -> ModelOutput`，具名 target/history/context 投影、显式 padding mask和稳定 batch rank；公共 fusion/preset 等价回归和 Linux 最终 202 passed/85%，旧实现已删除；2026-08-04。 |
| MOD-003 | P1 | 迁移复杂序列/辅助损失模型 | MOD-002 | `DONE` | Codex (`/root`) | `src/mmctr/models/{multimodal,sequence,advanced_sequence,gmmf}.py`、registry、canonical complex/GMMF tests、docs、`REFACTORING_PLAN.md` | SimCEN/NAML/MAKE/DMF/MARN/EM3/Diff-MSIN/GMMF 均为纯 canonical；GMMF 保留 autoencoder/CGAN/auto-difference/user gate/cosine pooling、重建损失与交替日程。固定 seed fixture 以 `atol=1e-7` 冻结 logits/reconstruction/discriminator/generator 和 1,051 参数，forward/backward/all-padding/参数组/真实 engine 日程及 Linux 最终 202 passed/85%，旧实现已删除；2026-08-04。 |
| MOD-004 | P1 | 迁移 MB/PAMD/MMMLP/M3SRec | MOD-002, MODEL-BASE-002 | `DONE` | Codex (`/root`) | `src/mmctr/models/specialized.py`、model registry、tests/docs、`REFACTORING_PLAN.md` | MB/PAMD/MMMLP/M3SRec 均为纯 canonical 并保留各自 scoring/PGD、decomposition、Mixer、MoE/attention 主体；masked history、left/all padding、batch-size-1、context/Batch 不变性和配置守卫纳入 Linux 最终 202 passed/85%。registry 仅保留 canonical capability 元数据，旧实现已删除；2026-08-04。 |
| MOD-005 | P1 | 迁移 RQ/PSRQ/QARM/MCCA | MOD-004 | `DONE` | Codex (`/root`) | `src/mmctr/quantization/`、`src/mmctr/models/quantized.py`、registries/Trainer/config/quantization trainers、tests/docs、ignored real artifacts/report、`REFACTORING_PLAN.md` | RQ/PSRQ 是独立 canonical 预训练组件，pickle-free 原子 NPZ 严格校验并注入纯 QARM/MCCA。TikTok fingerprint `a908...05bc` 的完整 6,711-row item store 按正式配置生成 image/text/audio 三个 `3×1024` RQ 工件（SHA-256 `84863e...5300`/`a3d854...b98`/`ac4d4e...68ca`）及 V100 五 epoch `3×256` PSRQ（final loss 8.485107，9,787,530 bytes，SHA-256 `561348...3336`）；严格重载后 QARM/MCCA 对真实 32-event batch 完成 forward/backward/Adam step，loss 0.730354/0.849905、logits finite。修复只读 memmap `torch.from_numpy` 告警，TDD RED import 后专项 25 passed，Ruff/mypy 通过；本地证据 `outputs/quantization/tiktok-real-validation.json` ignored；2026-08-04。 |
| COMP-001 | P1 | 统一 projection、mask、dimension adapter | MOD-005 | `DONE` | Codex (`/root`) | `src/mmctr/models/components/`、canonical model backbones、`tests/unit/test_model_components.py`、`docs/model-components.md`、模型文档、`REFACTORING_PLAN.md` | 严格具名 projection、dimension adapter、sequence/presence/masked-softmax 与 state-key 兼容迁移完成；dtype/rank/dimension/缺失/all-padding/backward/state-key 及 Linux 全量门禁通过；2026-08-03 |
| POOL-001 | P1 | 统一 pooling API、registry 与 capability | COMP-001 | `DONE` | Codex (`/root`) | `src/mmctr/models/components/pooling.py`、`src/mmctr/models/components/pooling_registry.py`、public exports、canonical compatibility layers、`tests/unit/test_pooling_components.py`、组件文档、`REFACTORING_PLAN.md` | mean/sum/max/attention/DIN/cross-attention 统一签名、6 名称 lazy registry/capability 和 state-key 兼容委托完成；shape/mask/all-padding/target/head/dtype/dimension/backward/registry/state-key 及 Linux 全量门禁通过；2026-08-03 |
| FUSE-001 | P1 | 统一 fusion API、registry、output_dim 与 aux loss | COMP-001 | `DONE` | Codex (`/root`) | `src/mmctr/models/components/fusion.py`、`fusion_registry.py`、canonical fusion compatibility、`tests/unit/test_fusion_components.py`、组件文档、`REFACTORING_PLAN.md` | 完成 concatenate/sum/mean/MAF/LMF/MTFN 的具名模态、rank/dtype/device/dimension/presence、output_dim、类型 `FusionOutput` 与 scalar auxiliary-loss 合约；6 名称 lazy registry 及 cat/add/average alias；DNN-MM/DNN-MM-Seq/NAML/DMF/MARN/LMF/MTFN 等价委托且学习参数 state key 不变。Linux Ruff/mypy 60 files、133 tests/87.11%、sdist+wheel 通过，wheel 353394 bytes/SHA-256 `dccc853b357afff7f9f26de62f47b366e1ec92df8cfa10473555edc58f994ade`；2026-08-03 |
| PIPE-001 | P1 | 实现按分支/模态配置的 pipeline 与三种 topology | POOL-001, FUSE-001 | `DONE` | Codex (`/root`) | `src/mmctr/models/components/pipeline.py`、component exports、`tests/unit/test_modal_pipeline.py`、`docs/model-components.md`、`REFACTORING_PLAN.md` | 冻结 typed config/严格 YAML-like 解析、target/history/user branch set、`feature_fusion` 与三种 history topology、projection/presence/mask/target/pooling/fusion/adapter 组合和带 presence/aux-loss 的 typed output 完成；专项 8 passed，Ruff 102 files、mypy 61 files、unit+smoke 131 passed、全量 141 passed/86.94%；无隔离网络失败后以既有 `bm` 依赖完成 no-isolation sdist+wheel，wheel 357657 bytes/SHA-256 `8ed96027d707604e111d77ee058362c6ba75554e859ce5f6d06e1d455c76ba79`；2026-08-03 |
| PIPE-002 | P1 | 为全部模型建立默认兼容 preset | PIPE-001 | `DONE` | Codex (`/root`) | `src/mmctr/models/presets.py`、model public exports、`tests/unit/test_model_pipeline_presets.py`、`docs/models.md`、`REFACTORING_PLAN.md` | 23 个正式 registry 名称及 `dnn_seq` alias 均有显式覆盖决策；DNN/DCN/DeepFM/AutoInt/DNN-MM/DNN-MM-Seq/LMF/MTFN/NAML/MAKE/SimCEN/QARM/DMF 共 13 个 preset 可执行，并以固定权重在 target/history/user branch 上按 `atol=1e-7` 回归且保持参数共享。其余 10 个按 ADR-032 返回具名 model-specific 默认决策并拒绝用近似公共组件 build，避免改变 DIN Dice padding、FQ-Former/SRC/DTA/MoE/量化/GAN 等论文公式。专项累计 23 passed，既有 Linux 全量 166 passed/87.29% 及构建证据有效；2026-08-03 |
| EXP-001 | P1 | 统一 Linux experiment runner、GPU 调度和结果 schema | TRAIN-001, RUN-001, PIPE-002 | `DONE` | Codex (`/root`) | `src/mmctr/experiments/`、`tests/unit/test_experiment_runner.py`、`docs/experiments.md`、`REFACTORING_PLAN.md` | 冻结 task identity 含 dataset/model/seed/resolved config/data fingerprint；显式 device queue 保证每设备最多一个 worker，任务使用唯一 RunContext，异常写 failed RunResult 而不取消 sibling。原子 matrix/result schema 支持只复用 completed task、failed task 新目录重试且保持调用顺序；补齐 RunContext 创建异常归还 device 的防死锁回归；3 项专项、更新前 unit+smoke 173 passed，Ruff/mypy 通过；2026-08-03。 |
| TUNE-001 | P0 | 建立只按 validation 选择的正式 tuner | EXP-001, SCI-001 | `DONE` | Codex (`/root`) | `src/mmctr/experiments/tuning.py`、`tests/unit/test_formal_tuning.py`、`docs/experiments.md`、`REFACTORING_PLAN.md` | `ValidationOnlyTuner` 保存全部 trial（含失败）并拒绝任何 `test*` trial metric；严格较高 `val_auc` 选优，`val_log_loss` 同步留证。原子 frozen selection 记录 experiment/run/validation/seeds/data fingerprint/完整配置且不写 tracked YAML；仅该选择可创建 final-test task，`test_*` 结果另文件保存。专项通过，更新后 unit+smoke 167 passed，Ruff/mypy 通过；2026-08-03。 |
| ANL-FUS-001 | P1 | fusion analysis 删除 17 个平行模型实现 | EXP-001 | `DONE` | Codex (`/root`) | `src/mmctr/analysis/fusion.py`、`src/mmctr/cli/app.py`、`configs/experiments/fusion.example.yaml`、tests/docs、legacy fusion analysis、`REFACTORING_PLAN.md` | 严格 YAML、`plan-fusion-study`、原子带指纹矩阵已取代旧入口，17 个平行模型已物理删除。Linux Ruff 138 files、mypy 83 files、分片全集 201 passed/86.75%、sdist+wheel 构建通过；2026-08-04。 |
| ANL-ALI-001 | P1 | alignment 改为 hook/aux-loss protocol | EXP-001 | `DONE` | Codex (`/root`) | `src/mmctr/analysis/alignment.py`、`src/mmctr/experiments/matrix.py`、`src/mmctr/cli/app.py`、`configs/experiments/alignment.example.yaml`、tests/docs、`REFACTORING_PLAN.md` | method/weight/module path/seed/data fingerprint 任务矩阵已取代旧 wrapper/专用 Trainer/GPU subprocess/CSV 目录。Linux Ruff 138 files、mypy 83 files、分片全集 201 passed/86.75%、sdist+wheel 通过；2026-08-04。 |
| ANL-ROB-001 | P1 | 合并 modal robustness 新旧实现 | EXP-001 | `DONE` | Codex (`/root`) | `src/mmctr/analysis/robustness.py`、`src/mmctr/experiments/matrix.py`、`src/mmctr/cli/app.py`、`configs/experiments/robustness.example.yaml`、tests/docs、`REFACTORING_PLAN.md` | drop probability/modalities/splits/seed/data fingerprint 生产任务已取代最后一个 500+ 行 legacy 训练/GPU/CSV 入口。Linux Ruff 138 files、mypy 83 files、分片全集 201 passed/86.75%、sdist+wheel 通过；2026-08-04。 |
| ANL-COLD-001 | P1 | 规范 cold/few/zero-shot split 与校验 | EXP-001 | `DONE` | Codex (`/root`) | `src/mmctr/analysis/cold_start.py`、`src/mmctr/cli/app.py`、cold-start config/tests/docs、legacy analysis Trainer/scripts、`REFACTORING_PLAN.md` | `plan-cold-start-study` 强制加载并校验版本化 split audit，任务身份锚定 protocol fingerprint、audit 文件 SHA-256、绝对 manifest 路径、seed 与 data fingerprint；第二套 `Trainers_fenxi.py`、analysis batch runner 和 4 个 4608 TFRecord 构建/运行脚本已物理删除。Linux Ruff 138 files、mypy 83 files、分片全集 204 passed/85%、sdist+wheel 通过；wheel 303,851 bytes/SHA-256 `632623437c2e97c88729b7feea3c07596051b12c65669576a4aa18a9a5ecbecb`；2026-08-04。 |
| ANL-EFF-001 | P1 | 统一效率、显存、参数量和 CUDA 计时协议 | EXP-001 | `DONE` | Codex (`/root`) | `src/mmctr/analysis/efficiency.py`、`tests/unit/test_efficiency_protocol.py`、`docs/experiments.md`、`outputs/analysis/efficiency/antm2c-dnn-mm-seq-cuda0.json`（ignored）、`REFACTORING_PLAN.md` | 统一 warm-up/measurement step、总/可训练参数、CPU/CUDA 同步计时、吞吐、CUDA peak allocated memory、GPU/PyTorch/CUDA runtime；输入 fingerprint 和双重完整性 fingerprint 的版本化 report 原子落盘。2 项 CPU 协议/篡改专项、Ruff/mypy 通过；最终 AntM2C fingerprint 的真实 256-event `dnn_mm_seq` V100 inference：10 warm-up + 50 measured、2.7244 ms/step、93,965.27 events/s、40,358,400 peak bytes、5,087,009 参数，report fingerprint `abf85644ef9d70a30d13596075a8ebfc7735dc6731b4af44a0579f20f5bd7142`；2026-08-04。 |
| PLOT-001 | P2 | 绘图只读取标准结果，去掉硬编码数据 | ANL-FUS-001, ANL-ALI-001, ANL-ROB-001, ANL-COLD-001, ANL-EFF-001 | `DONE` | Codex (`/root`) | `src/mmctr/analysis/plotting.py`、`src/mmctr/cli/app.py`、`tests/unit/test_plotting_protocol.py`、docs、legacy plot scripts、`REFACTORING_PLAN.md` | `plot-results`/`render_metric_figure` 只消费 completed result-v1，按 model/dataset/task/seed 聚合，原子写 PNG/PDF/SVG 并强制 provenance；11 个嵌入实验数字脚本已删除，19 个历史最终图保留。Linux Ruff/mypy、201 passed/86.75%、build 通过；2026-08-04。 |
| REL-001 | P2 | 四阶段完成后的开源 release checklist | CI-001, PLOT-001 | `IN_PROGRESS` | Codex (`/root`) | `MANIFEST.in`、`docs/release-checklist.md`、package/release metadata tests、build/install evidence、`REFACTORING_PLAN.md` | S1–S4、Apache-2.0、本地工程演练及最终目录/发行包审计均完成；维护者已批准版本/public tree/changelog 并明确要求上传最新版。当前最终本地证据与 `DIR-001` 一致，正在创建并推送 clean source commit、等待该 commit 的 GitHub CI；签名 tag 与 artifact publication 不在本次源码上传授权范围内，继续作为独立维护者动作；2026-08-04。 |

### 13.3 首轮建议执行顺序

在没有更多业务优先级输入时，按以下顺序推进：

1. `GOV-001` → `GOV-002`
2. `ENV-001` → `ENV-002` → `OSS-001`
3. `TEST-001` → `BASE-001`，并行处理 `RUN-001` 与 `SCI-001`
4. `PKG-001` → `CFG-001` → `CFG-002` → `CLI-001` → `CI-001`
5. 第一阶段验收后进入 `CORE-001` → `DATA-001` → `ANT-001`
6. AntM2C 按 `ANT-002/003` → `ANT-004` → `ANT-005` 推进，再开始批量模型迁移

在 `TEST-001` 和 `BASE-001` 完成前，不进行全仓库包迁移或模型批量重写。

---

## 14. 决策记录

重大决策在此登记；更完整内容可迁入 `docs/adr/`。

| ADR | Date | Decision | Status | Rationale |
|---|---|---|---|---|
| ADR-001 | 2026-07-31 | 采用 `src/mmctr` 单一公共包命名空间；首轮以显式兼容桥保留 legacy 顶层包，后续按任务逐模块迁入 | `ACCEPTED` | 先提供稳定公共入口并消除循环导入，避免一次性搬迁全部研究代码造成不可审计的大范围回归 |
| ADR-002 | 2026-07-31 | 分析通过配置/adapter 复用生产模型，不复制模型文件 | `ACCEPTED` | 防止 17 组平行实现漂移 |
| ADR-003 | 2026-07-31 | 模型与训练 engine 解耦 | `ACCEPTED` | 支持纯 forward 测试、统一训练和实验协议 |
| ADR-004 | 2026-07-31 | 正式调参只使用 validation，test 仅冻结后评估 | `ACCEPTED` | benchmark 科研诚信底线 |
| ADR-005 | 2026-07-31 | 每次运行使用 UTC 微秒时间戳、解析配置短哈希和随机熵组成的 run ID，并以原子目录创建隔离产物 | `ACCEPTED` | 时间与配置可读可追踪，随机熵与 `exist_ok=False` 防止同配置并发覆盖 |
| ADR-006 | 2026-07-31 | Linux 服务器是依赖、训练和性能验收的权威环境 | `ACCEPTED` | 后续由 ADR-016 进一步收敛为唯一修改与验证环境 |
| ADR-007 | 2026-07-31 | 多模态模型主干只保留 BaseSeqModel | `ACCEPTED` | 原 BaseModel 模型通过序列 pooling 接入统一契约 |
| ADR-008 | 2026-07-31 | AntM2C item 特征独立存储并按 item index gather | `ACCEPTED` | 删除 4608 打包和 item 固定切片协议 |
| ADR-009 | 2026-07-31 | pooling/fusion 按分支与模态配置，并做 capability 校验 | `ACCEPTED` | 支持可组合实验且避免无效组合 |
| ADR-010 | 2026-07-31 | `ENV-001` 保持 `REVIEW`，Linux 实测延期为发布门禁；允许继续本地静态可验证任务 | `SUPERSEDED` | 历史延期决定；ADR-016 已恢复服务器唯一验证，Linux/CUDA 与依赖门禁现已实际通过 |
| ADR-011 | 2026-07-31 | 分发名使用 `mmctr-bench`，目标导入命名空间仍为 `mmctr`；迁移前暂时发现 legacy `src` 包 | `ACCEPTED` | 区分 PyPI 分发名与 Python 包名，并使 `ENV-002` 可在 `PKG-001` 前建立可构建元数据 |
| ADR-012 | 2026-07-31 | 配置相对路径统一相对包含 `pyproject.toml` 的项目根解析；training 配置先以 frozen dataclass 严格校验，模型/数据算法字段在对应任务中逐步 typed 化 | `ACCEPTED` | 消除 cwd 差异并立即保护运行关键字段，同时避免在未建立模型回归前一次性重写全部论文专属参数 |
| ADR-013 | 2026-07-31 | 机器专属数据路径只允许来自 ignored `configs/local/paths.yaml`、`MMCTR_*_DATA_DIR` 环境变量或显式 CLI override；优先级为本地文件 < 环境变量 < CLI，tracked 配置只保留相对路径和空 example | `ACCEPTED` | 同一公开配置可跨机器复用，真实服务器路径不进入 Git，缺失 override 时给出明确错误而非引用不存在文件 |
| ADR-014 | 2026-07-31 | 公共命令统一为 `mmctr` / `python -m mmctr.cli` 子命令；CLI 模块保持 dependency-light，train 命令完成参数/线程设置后才导入训练 runtime | `ACCEPTED` | `--help`、配置检查和列表命令可安全导入运行，避免 argparse/torch/TensorFlow 在模块导入阶段产生副作用 |
| ADR-015 | 2026-07-31 | 服务器接续使用既有 Conda `bm` 环境；下载、缓存和临时产物与当前项目同盘；原始 `<ORIGINAL_REFERENCE_ROOT>` 只读参考 | `ACCEPTED` | 遵循维护者提供的服务器资源边界，避免占用其他磁盘或误改原始版本，同时为数据位置和历史约定保留可核对的 ignored private evidence |
| ADR-016 | 2026-07-31 | 后续修改、静态检查、依赖/CUDA 和任务验收全部在当前 Linux 服务器完成，不再使用 Windows；允许在既有 `bm` 中按需安装并调整冲突依赖 | `ACCEPTED` | 维护者明确指定服务器为唯一工作面并授权环境修复，消除双环境复核造成的任务悬置，同时仍要求记录精确版本、影响与真实运行证据 |
| ADR-017 | 2026-07-31 | `bm` 的 TFRecord 适配栈对齐 TensorFlow `>=2.12.1,<2.13`、h5py `>=3.8,<3.11`、JAX/JAXlib `0.4.13` 与 typing-extensions `>=4.4,<4.6`；保留 NumPy 1.23.5、PyTorch 1.13.1+cu117 和 CUDA 主栈，并让 dev/tuning 的传递依赖选择兼容版本 | `ACCEPTED` | TensorFlow 2.4 与当前 NumPy/typing/Keras 明确冲突，旧 h5py 2.10 另有可复现 NumPy C-ABI warning；2.12.1 支持 Python 3.8、NumPy 1.22–1.24.3 且包含已知安全修复，但官方要求 typing-extensions `<4.6` 且其 Python 3.8 JAX 传递解析未自动安装 jaxlib，因此显式配对 JAX/JAXlib，只联动更新 TF/h5py 并回退冲突的工具/ORM/backport，不降级数值或训练框架 |
| ADR-018 | 2026-08-03 | 在 `mmctr.core` 建立 `Batch`、`ModelOutput`、`RunResult` 单一公共契约；核心对象使用冻结 dataclass、只读映射视图和构造时校验，legacy tuple/dict 仅通过显式 adapter 进入核心边界 | `ACCEPTED` | 先固定跨 data/model/training/experiments 的 shape、dtype、mask、辅助损失与运行结果语义，避免后续迁移继续扩散隐式 tuple 顺序和 `pred`/`au_loss` 拼写；显式兼容适配允许按模型逐步迁移而不一次性改变论文实现 |
| ADR-019 | 2026-08-03 | 数据公共入口使用 `CanonicalDataLoader` 适配现有三类 loader，统一为 `iter_batches(split, history_mode)` 与版本化 `DatasetManifest`；数据集专属读取/索引暂留 adapter 内，AntM2C 字段拆分由后续 `ANT-*` 实施 | `ACCEPTED` | 先让 training/model 只消费统一 `Batch`，同时避免在字段 ownership 尚未审计前静默改变 TFRecord 语义；manifest 固定 padding/OOV/ID offset、特征维度和 split 统计边界 |
| ADR-020 | 2026-08-03 | 模型和数据集使用独立的正式 registry，注册项只保存规范名称、lazy import spec、alias 与 capability metadata；fusion/pooling registry 仍按 S3 任务边界后置 | `ACCEPTED` | CLI 列表与配置校验无需加载 torch/TensorFlow，构造时才解析实现；唯一规范名称消除 helper 长分支和 `dnn_seq`/`dnn_mm_seq` 漂移，同时不抢跑第三阶段组件协议 |
| ADR-021 | 2026-08-03 | 训练生命周期从模型类移入 `TrainingEngine`；engine 只消费 canonical loader/Batch/ModelOutput，early stop 仅看 validation AUC，checkpoint manager 原子管理 `best.pt`/`last.pt` 并保存 optimizer/epoch/metric | `ACCEPTED` | 消除 BaseModel/BaseSeqModel 重复训练和跨 run checkpoint 覆盖；模型恢复为纯张量模块，测试集评估保持为配置冻结后的显式独立步骤，resume 所需状态由统一 payload 管理 |
| ADR-022 | 2026-08-03 | 正式模型只继承 `mmctr.models.BaseSeqModel` 并声明 `pooled_history` 或 `sequence_tokens` capability；legacy 双基类及其旧 forward 签名通过只复制输入的 `LegacyModelAdapter` 过渡，不进入新 engine 核心 | `ACCEPTED` | 将迁移风险限制在适配边界，立即保证 canonical forward 不修改 `Batch`，同时允许按模型 fixture 逐步替换旧实现；公共基类不再持有 train config、optimizer、device、logger 或 checkpoint 路径 |
| ADR-023 | 2026-08-03 | AntM2C 的 service/query/bill/log_time 属于交互上下文，item_title/image 属于 item；item_entity_names 在全量 `item_id -> value` 一致性审计前标为 item candidate，不得直接去重；event_id 使用数据版本内稳定的 source-shard/source-row 身份，历史按 `(user_id, timestamp, event_id)` 单次扫描 | `ACCEPTED` | 字段名与现有提取/序列化代码足以确认明确归属，但没有真实数据时不能假设 item_entity_names 对同一 item 永远不变；稳定事件身份和并列时间排序是消除 `items.index` 重复 item 错位与未来泄漏的前提 |
| ADR-024 | 2026-08-03 | `Batch` 增加独立只读 `context_features`；AntM2C serializer 以具名 interaction/item arrays 为语义边界，`named-npy-candidate-v1` 只作为 ANT-006 基准候选而非最终格式 | `ACCEPTED` | interaction context 不能伪装成 item feature，也不能继续依赖 4608 拼接偏移；先稳定字段 ownership 和 loader 契约可并行推进模型，最终存储仍必须由服务器空间/吞吐实测决定 |
| ADR-025 | 2026-08-03 | 简单多模态迁移在模型模块内保留私有 cat/add/mean/MAF/LMF/MTFN 运算，正式 registry 指向 canonical 模型，旧 helper 指向冻结实现；公共 fusion registry 仍留给 FUSE-001 | `ACCEPTED` | DNN-MM/LMF/MTFN 可先脱离训练型基类，同时不抢跑尚需 capability/output_dim/aux-loss 统一设计的第三阶段公共组件；双入口为服务器数值回归保留证据 |
| ADR-026 | 2026-08-03 | `MOD-004` 的四个单优化器模型可在 GMMF multi-optimizer 协议完成前独立迁移；任务依赖收窄为已稳定的 `MOD-002`/`MODEL-BASE-002` 公共契约，GMMF 继续留在 `MOD-003` 且不得改用单优化器近似 | `ACCEPTED` | MB/PAMD/MMMLP/M3SRec 仅依赖既有 Batch、projection、mask、ModelOutput 和单优化器 engine，不消费 GMMF 私有 GAN 日程；串行阻塞会降低可交付迁移量，而把 GMMF 强行并入普通 forward 会改变科研训练协议 |
| ADR-027 | 2026-08-03 | RQ/PSRQ 从 CTR `BaseSeqModel`/model registry 分离为 `mmctr.quantization` 预训练组件和独立 quantizer registry；量化工件使用带版本/结构元数据的无 pickle NPZ，QARM/MCCA 构造仅接收已加载量化器，路径解析由 registry/Trainer 组合层负责 | `ACCEPTED` | RQ/PSRQ 产出离散表示而非点击 logits，强塞入 `Batch -> ModelOutput` 或 `list-models` 会混淆训练目标；旧模型构造期文件 I/O 破坏纯模型、隔离 run 与可测试性，裸 codebook/state 又无法验证层数、维度、模态和消费者配置是否一致 |
| ADR-028 | 2026-08-03 | `mmctr.models.components` 提供严格的具名浮点投影、维度适配、序列 mask 与缺失模态 mask；投影只处理已编码连续张量，ID embedding 和模型特有变换仍由模型负责；公共组件拒绝未知/缺失字段、错误 rank/dtype/末维且不修改输入 | `ACCEPTED` | 把 ID 查表、离散量化和所有模型公式塞入单一 projector 会形成不稳定的万能抽象；在张量已编码边界统一线性维度与 mask 可消除重复实现，同时保持论文特有编码器和现有参数共享语义，配置错误也能在靠近组件入口的位置失败 |
| ADR-029 | 2026-08-03 | pooling 公共接口统一为 `forward(sequence, mask, target=None) -> [B,D]`；registry metadata 声明 input/output rank、mask/target 要求和 output-dimension 规则；mean/sum/max/attention/DIN/cross-attention 作为独立注册组件，现有私有层只在公式与 state key 等价时委托公共实现 | `ACCEPTED` | 单一签名允许 pipeline 在配置阶段校验 target-aware 组合；把所有历史 attention 强行替换为一个算法会改变论文公式和 checkpoint，因此公共组件先覆盖明确算法族，兼容层仅做等价委托，其余模型在默认 preset 回归建立后再切换 |
| ADR-030 | 2026-08-03 | fusion 公共边界接收精确具名模态映射和可选显式 presence，返回包含 fused tensor 与具名 scalar auxiliary losses 的类型结果；每个组件声明模态数、rank、output_dim 和 aux-loss capability，cat/add/mean/MAF/LMF/MTFN 先作为等价算法族注册 | `ACCEPTED` | 类型结果避免 pipeline 猜测某个 fusion 是否产生辅助损失；精确具名和显式 presence 防止配置遗漏或输入顺序改变语义，又不强行将论文私有 fusion 在无回归时合并 |
| ADR-031 | 2026-08-03 | history modal pipeline 使用冻结 typed config，在构造阶段解析 pooling/fusion registry capability；`pool_then_fuse` 为每模态独立 pooling 后 fusion，`fuse_then_pool` 先做 token fusion 后单一 pooling，`sequence_fusion` 保留 token rank；target-aware pooling 显式接收同名 target mapping，维度适配只作用于最终表示 | `ACCEPTED` | 单一 `values/presence/sequence_mask/targets` 公共入口可覆盖三种拓扑并保持组件可替换；在模型 forward 前拒绝缺失 target、错误 pooling 数量和不支持 rank，避免配置看似有效却到训练中途失败，同时不让 pipeline 猜测 Batch 字段 ownership |
| ADR-032 | 2026-08-03 | 默认 pipeline preset 按模型规范名由 resolved model/data config 生成；只对 projection/pooling/fusion 公式与现有 state 语义等价的模型产出可执行 preset，paper-private FQ-Former/SRC/DTA/MoE/量化等路径显式保留为 model-specific，直到对应组件注册和数值回归齐备 | `ACCEPTED` | “默认 preset”必须是可验证的旧行为描述，不能用相近的 mean/cat 等公共组件冒充论文私有公式；当前 13 个 state-compatible 路径已回归，QARM 覆盖 ID/多层量化模态分支，DMF 仅抽取 DTA/SimTier 前的等价模态中心，DIN 因 Dice 对 padding projection bias 的观察产生约 `1e-3` 差异而保持私有，其余模型也保留明确依赖，不污染 benchmark 语义 |
| ADR-033 | 2026-08-03 | GMMF 的主/判别器/生成器使用互斥具名参数组与逐阶段 step 的单一 phased Adam；engine 只消费显式 `AlternatingPhase(name,start_epoch,objective)`，checkpoint 保存一个完整 optimizer state | `ACCEPTED` | 对互斥参数，Adam 状态逐参数维护，清空全部梯度后只 step 当前组与三个独立 Adam 的更新等价；单一 state 消除遗漏判别器/生成器 resume 的风险，显式阶段避免 engine 按模型名分支，同时严格保留每 batch main→discriminator→generator 和 `epoch >= N` 日程 |
| ADR-034 | 2026-08-03 | MicroLens/TikTok 的 canonical-v1 使用版本化具名 NumPy arrays：interaction split 与去重 item feature store 分离，loader 直接产出 `Batch`；MicroLens 对既有交互作确定性 8:1:1 新版本切分，TikTok 以原始官方 split 和因果 prefix history 生成 CTR 样本，不继承 legacy 合并三 split 后随机切分的泄漏协议 | `ACCEPTED` | 两数据集规模和定长 dense 模态适合可 memory-map 的数组格式，可复用 AntM2C 候选格式的具名语义而无需 TensorFlow 进入训练路径；TikTok legacy 预处理将 train/val/test 用户序列合并后再切分，会污染正式评估，修复必须发布新数据版本并在 manifest 记录负采样与 ID 映射，而不是静默覆盖旧 TFRecord |
| ADR-035 | 2026-08-03 | AntM2C 格式基准按源 shard 分层扫描真实 TFRecord，剔除并显式统计 packed target 与共享 item store 不一致的事件后，为两格式写入同一固定事件集合；格式固有 item store 与 interaction 增量体积分开报告，两条读取路径均计入解析、item gather、PyTorch batch 组装和完整样本消费，GPU 等待仅在显式 H2D/CUDA synchronize 协议下单列 | `ACCEPTED` | 直接比较 413 GiB legacy 目录与小样本候选会把重复 train 副本、共享 item store 和样本规模混为一谈；跨 shard 固定事件集合避免 val/test 只采 domain 0，相同 batch size、semantic checksum 与端到端 batch 语义形成公平 CPU I/O 对比。当前 loader 是 CPU 数据边界，未执行 H2D 时把差值称为 GPU wait 会产生虚假性能结论 |
| ADR-036 | 2026-08-03 | `item_entity_names/entity_text` 从 item candidate 定为 interaction context；service/query/bill/entity/time 以五个具名数组保存，旧模型所需 3840 维 user `text` 仅由 loader 提供虚拟拼接视图，模型按配置名从 context 显式消费 | `ACCEPTED` | 真实分层样本中 247/2,820 item 有多个 entity 版本，单 item 最多 19 个，已足以否定 item ownership；虚拟视图保持 legacy user-text 数值顺序但不在磁盘重复写入，也不把 interaction context 伪装成静态 user metadata |
| ADR-037 | 2026-08-03 | AntM2C 全量 canonical 选择分层 named arrays + 单一共享 item feature store，不继续以 packed TFRecord 作为正式核心训练格式；全量写入必须分 shard、可恢复并保留 source/输出 hash 与冲突审计 | `ACCEPTED` | 相同 6,144 条真实事件和完整 Batch checksum 下，named interaction 体积减少 24.82%、warm-cache CPU batch 吞吐提升 6.09×，且删除 TensorFlow/固定 `[6,768]` 切片训练依赖；全量约 174.21 GB，单文件预分配或不可恢复的一次性转换风险过高，故由 `ANT-007` 完成流式分 shard 发布 |
| ADR-038 | 2026-08-03 | AntM2C `sharded-named-npy-v1` 以源 split/file/row 组成稳定 event ID；每个输出 shard 使用独立临时目录并在逐数组 SHA-256 完成后原子发布，恢复状态只承认已发布 shard，完成全部源文件后才生成最终 manifest 并将 staging 原子提升为 canonical-v1；packed target 与共享 store 冲突时保留交互并以共享 item store 为权威，只累计冲突审计 | `ACCEPTED` | 大体量五组交互上下文不能安全预分配为单文件，也不能在中断后从头转换；按源文件保留逻辑 record hash、按输出数组保留 hash，使恢复、篡改检测和冲突审计都有明确边界。packed target 是待删除的重复字段，因其冲突丢弃完整事件会无必要改变 split 样本集合；canonical target/history 始终从同一共享 title/image store gather。 |
| ADR-039 | 2026-08-04 | AntM2C manifest 必须直接锚定共享 item feature SHA-256，同时锚定 aggregate shard metadata；layout 只描述物理路径且其 hash 必须与 manifest 一致 | `ACCEPTED` | 若 item hash 只存在于可同时修改的 layout，攻击者或磁盘错误同时改变 array/layout 时 manifest fingerprint 不变；把 item digests 纳入 manifest 后，item 与 interaction 两类数据均由数据版本 fingerprint 封闭，专项覆盖同时篡改 item+layout 的拒绝路径。 |
| ADR-040 | 2026-08-04 | 23 个正式 registry 全部 canonical 后，不再发布 `models` legacy 包、`BaseModel`、legacy adapter/解析器或依赖旧构造签名的 tuner；历史数值基线保留为固定输入/权重/期望值的 canonical 回归 fixture，不为测试继续携带第二套生产实现 | `ACCEPTED` | 同一 wheel 同时发布 canonical 与旧模型会继续产生双修、错误入口和包名冲突；以自包含 fixture 守住迁移数值证据，比让测试动态导入整套旧模型更能封闭正式运行边界。正式 validation-only tuner、统一 Trainer 和 canonical RQ/PSRQ 入口已经覆盖有效工作流。 |
| ADR-041 | 2026-08-04 | 数据集获取与软件发布分离：仓库只发布处理代码、schema 和指纹，不打包或提供三方数据镜像；没有原提供方明确数据许可证时，不以论文许可、上游代码许可证或公开下载链接推定再分发授权 | `ACCEPTED` | MicroLens 原作者明确禁止私自修改后提供二次下载，AntM2C 与当前 TikTok 处理版均未核验到独立、机器可读的数据许可证。逐数据集记录来源、引用和未知项，并要求用户自行遵守提供方条款，可避免软件 LICENSE 选择意外扩张到数据内容。 |
| ADR-042 | 2026-08-04 | 仓库自有源代码、配置和文档采用 Apache License 2.0（SPDX `Apache-2.0`）；三方数据、原始媒体、派生数组与 checkpoint 不因软件许可证获得再分发授权，未知权利人时不创建 `NOTICE` 声明 | `ACCEPTED` | 维护者已明确选型；Apache-2.0 提供明确的版权与专利许可并要求分发时保留许可证。将数据/权重边界写入 README、贡献指南与发布门禁，可避免把软件授权错误扩张到 AntM2C、MicroLens、TikTok 或 encoder 权重。 |
| ADR-043 | 2026-08-04 | AntM2C 原始事件、图片归档与 encoder 保持维护者提供的外部存储只读，真实 tracer/extraction/report 写项目盘；raw history/item/encoder 边界以跨 split 30K event、真实 title/image 和 checkpoint/全源 hash 验收，现有 11.19M canonical-v1 不因验证而静默覆盖 | `ACCEPTED` | 全量重新编码约五组 interaction 文本及 84.9K 图片，成本和新浮点/依赖版本会形成新的数据版本；先用真实上游输入关闭实现与生命周期实证，再保留已逐 hash 验证的全量 canonical-v1，兼顾可追溯性、磁盘安全和版本语义。若未来全量 raw 重编码，必须发布 canonical-v2。 |
| ADR-044 | 2026-08-04 | installable runtime 只允许 `mmctr` 顶层包；剩余 `data/trainers/utils` 代码迁入 canonical namespace 后物理删除；曾暂准 `src/analysis` 只保留无 Python 代码的 19 个历史最终图 | `SUPERSEDED` | runtime 收敛决定仍有效；历史最终图的暂存位置已由 ADR-045 进一步收敛到 `reports/figures/`，因此 `src/` 现在只承载可安装源码。wheel member audit 继续强制无四个 legacy 顶层包。 |
| ADR-045 | 2026-08-04 | 公开配置只保留 `configs/`：dataset/model catalog 与 training default 各有唯一文件，experiment/local 配置同属该树；mutable checkpoint/log/quantization/run 统一进入 ignored `outputs/`，19 个版本化最终图从 `src` 迁入 `reports/figures/`；数据 payload 继续忽略但 `data/raw|processed/{dataset}/README.md` 形成 tracked filesystem interface；删除空根 `__init__.py` 和 PEP 517 已不需要的 `setup.py` | `ACCEPTED` | `config`/`configs`、`experiments`/`outputs` 与 source/figure 双位置使调用方泄漏路径知识，模块 shallow 且 deletion test 表明删除可集中复杂度。单一配置根、单一 mutable output seam、明确 raw 下载目录和 source-only `src` 提升 locality/leverage；`pyproject.toml` 已是唯一 build interface。 |

---

## 15. 变更记录

| Version | Date | Author | Change |
|---|---|---|---|
| v1.20 | 2026-08-04 | Codex | 完成 `DIR-001` 并推进 `REL-001`：删除 `config/`、根/`src` 空 package、legacy `setup.py` 和未消费 tuner 配置，建立唯一 `configs/` catalog、三数据集 raw/processed 下载说明、单一 ignored `outputs/` 与 `reports/figures/`。最终 Linux pip/Ruff/mypy、198 unit+smoke、213 full/84.90% 及定向 24 passed；构建/成员审计/仓库外安装通过，wheel 186,939 bytes/92 members/SHA `cf8a216c...5870`，sdist 242,442 bytes/238 members/SHA `b08ac1ca...3901`；维护者已批准公开树并要求上传，继续完成 release commit 与 GitHub CI。 |
| v1.19 | 2026-08-04 | Codex | 完成 S2、`ANT-002/003/004` 与 `PKG-002`：三份 82.95GB raw events、24.83GB 图片包和两套 checkpoint 均有全文件/aggregate hash；30K event history replay、25-title shared item store、GPU0 BERT 旧 embedding 对照、GPU1 真实 tar image ChineseCLIP 及 restartable extraction 均通过。MicroLens/TikTok 的 8 个 source hashes 与只读参考完全一致，并澄清为上游预提取模态输入。TDD 入口契约由 2 failures 转 3 passed；删除 21 个旧 data 文件、8 个顶层 trainer/utils、最后一个旧 analysis Python 和过期 dist/audit，保留 19 个历史图。Linux Ruff 143、mypy 87、197 unit+smoke、212 full/84.57%、build/仓库外 canonical entrypoints 全通过；wheel 188,342 bytes/93 members/SHA `f5c9d343...bddf9e`，sdist 241,473 bytes/225 members/SHA `045b62f1...5f2f9e`。 |
| v1.18 | 2026-08-04 | Codex | 重新打开 S2 与 `ANT-002/003/004`：维护者给出外部只读 AntM2C 三份原始事件分片、图片归档及 BERT/CLIP checkpoint，纠正此前只搜索原始参考树形成的错误 prerequisite 结论。外部源保持只读、所有新产物写项目盘；已确认 8×V100-32GB 可用，按 TDD 开始真实 raw→history/item/encoder tracer replay。同时确认 MicroLens canonical 源为交互/预提取 BERT+CLIP parquet，TikTok canonical 源为官方 split JSON/预提取三模态 arrays，二者均非原始媒体重编码。 |
| v1.17 | 2026-08-04 | Codex | 完成 `OSS-001` 与 S1：按维护者决定加入标准 Apache-2.0 `LICENSE`，同步 PEP 621/OSI/CFF/README/CONTRIBUTING/manifest，并明确三方数据与 checkpoint 不在软件授权内。TDD 由 1 failure 转 5 passed；Linux pip/Ruff/mypy、194 unit+smoke、208 full/86.78%、build、LICENSE 成员审计与仓库外 metadata/CLI 全通过。最终 wheel 208,382 bytes/118 members/SHA-256 `fda91c49...2d3e3e`，sdist 251,320 bytes/257 members/SHA-256 `e3dfe1de...fed177`。同时澄清现有 AntM2C canonical-v1 是从当时已有的 28 个派生 TFRecord 与两份 item feature arrays 迁移，原始上游仍待恢复。 |
| v1.16 | 2026-08-04 | Codex | 完成 `REL-001` 工程演练并转 `BLOCKED`：TDD 发现 sdist 缺 citation/贡献/数据许可/引用文档，新增 `MANIFEST.in` 后契约 4 passed；最终 Linux pip/Ruff/mypy、193 unit+smoke、207 full/86.78%、build 与仓库外 wheel CLI 全通过。最终 117-member wheel 为 200,252 bytes/SHA-256 `857cb719...5ef31c`，256-member sdist 为 241,842 bytes/SHA-256 `bb0fa8a9...87260a2`，禁带成员/敏感路径 0；计划本身不进入 sdist，避免工件哈希自引用。解除条件为维护者选择 LICENSE、恢复 AntM2C 原始输入关闭 S2，并批准 clean release commit/tag。 |
| v1.15 | 2026-08-04 | Codex | 完成 `OSS-002`：三套数据来源/获取/许可状态和不再分发规则落地，23 个 registry 名称全部进入论文适配/融合适配/内部变体引用矩阵；TDD 文档契约由 2 failures 转 2 passed。继续领取 `REL-001`，准备最终 Linux wheel 仓库外安装与发布审计。 |
| v1.14 | 2026-08-04 | Codex | 领取 `OSS-002` 并接受 ADR-041：开始补齐三套数据来源/获取/许可边界与 23 个正式模型引用映射；仓库只发布处理代码、schema 和指纹，任何未核验的数据许可不由论文或代码仓库许可证替代。 |
| v1.13 | 2026-08-04 | Codex | 穷尽 `ANT-002/003/004` 真实前置审计：权威只读参考与第二副本的 AntM2C raw 目录均为 0 文件，同盘搜索 raw parts/CSV/source TFRecord/title/image 输入均为 0；canonical-v1 不含 timestamp/original item/raw content，不能冒充原始历史/index/encoder 重放。三项及 S2 改为带精确恢复条件的 `BLOCKED`，ignored 审计保存于 `outputs/data/antm2c-raw-prerequisite-audit.json`。 |
| v1.12 | 2026-08-04 | Codex | 完成 `MOD-005`：以 TikTok 完整真实 item store 生成正式 RQ 三模态与 V100 PSRQ 工件，严格重载后 QARM/MCCA 均完成真实 32-event forward/backward/Adam step；保存 ignored 带哈希报告。真实运行发现并以 TDD 修复 PSRQ 对只读 memmap 的 `torch.from_numpy` 告警，专项 25 passed、Ruff/mypy/diff check 通过。 |
| v1.11 | 2026-08-04 | Codex | 完成 `MODEL-BASE-002` 及 `MOD-001..004`：以 TDD 物理删除 37 个 legacy 模型文件、双基类/adapter、旧 tuner/runner/统计脚本和 registry/helper 兼容解析；DNN 原冻结数值精确保持，GMMF 改为固定 canonical 公式 fixture。Linux Ruff 135 files、mypy 82 files、分片全集 202 passed/85%、sdist+wheel 与 diff check 通过；wheel 199,967 bytes/SHA-256 `6aae71c781d3de24c2910d4af34b7a21b99b14bdb7fc7cbb22d14fb0036e296f` 且无 `models/`、`scripts/`、compat 模块。 |
| v1.10 | 2026-08-04 | Codex | 领取 `MODEL-BASE-002` 最终物理收敛并接受 ADR-040：禁止 wheel/正式源码继续包含 `models` legacy 包、双基类/adapter/解析元数据和旧 tuner；DNN/GMMF 数值基线迁为自包含 canonical fixture 后再删除旧实现，完成前保持 `IN_PROGRESS`。 |
| v1.09 | 2026-08-04 | Codex | 完成 `ANL-COLD-001` 与 S4：cold-start 配置任务强制绑定已校验 split audit，物理删除最后第二套 analysis Trainer、批跑及 4608 TFRecord cold/few/zero-shot 脚本。Linux Ruff 138 files、mypy 83 files、分片全集 204 passed/85%、sdist+wheel 和 `git diff --check` 通过；wheel 303,851 bytes/SHA-256 `632623437c2e97c88729b7feea3c07596051b12c65669576a4aa18a9a5ecbecb`。 |
| v1.08 | 2026-08-04 | Codex | 完成 `ANL-FUS/ALI/ROB/PLOT`：四类正式 config/CLI/带指纹产物已取代平行模型、专用 Trainer 与硬编码图表；Linux Ruff 138 files、mypy 83 files、分片全集 201 passed/86.75%、wheel 304,766 bytes/SHA-256 `5748a5571afccd38fef11ebde58822e54e39f6dc3e9c05d52a6c4e5ebd8335d7` 通过。重新打开 `ANL-COLD-001` 以删除最后第二套 analysis Trainer/4608 TFRecord 脚本。 |
| v1.07 | 2026-08-04 | Codex | `ANL-ALI-001` 实现转 REVIEW：新增 alignment config/CLI/带指纹任务矩阵，删除旧 wrapper/专用 Trainer/GPU subprocess/CSV 目录，相关 20 passed；继续领取 `PLOT-001` 通用标准结果绘图入口与硬编码脚本收敛。 |
| v1.06 | 2026-08-04 | Codex | `ANL-ROB-001` 实现转 REVIEW：新增正式 robustness config/CLI/带指纹任务矩阵，删除最后一个 legacy 训练/GPU/CSV 入口，相关 15 passed 且 Ruff/mypy 通过；继续领取 `ANL-ALI-001` 物理收敛。 |
| v1.05 | 2026-08-04 | Codex | `ANL-FUS-001` 实现转 REVIEW：新增 canonical YAML/CLI/带指纹任务矩阵，物理删除 17 个 fusion 平行模型目录和旧入口，专项 4 passed、Ruff/mypy 通过；继续领取 `ANL-ROB-001` 的最后 legacy CLI 收敛。 |
| v1.04 | 2026-08-04 | Codex | 继续领取 `ANL-FUS-001`：只读引用审计确认 17 个 fusion 模型副本仅在旧目录内自引用；计划以 TDD 增加 canonical 配置/CLI 任务矩阵导出，随后物理删除平行实现和旧入口并执行 Linux 门禁。 |
| v1.03 | 2026-08-04 | Codex | 完成 `REG-001`：23 CTR、2 quantizer、3 dataset 的唯一 lazy registry 均解析正式 canonical 实现，三套真实 manifest 可构造且最终 189 项门禁通过；legacy 文件物理删除继续由独立 MODEL-BASE/analysis 任务跟踪。 |
| v1.02 | 2026-08-04 | Codex | 完成 `ANT-007`/`ANT-005`：28 源文件发布 11,190,985 events、1,389 shards、173.23GB，manifest 同时锚定 item/shard hash；全量 15,284 文件逐 SHA/bytes/shape/dtype/count 验证 711.558s 通过，三 split CPU 与 V100 训练 step、真实效率报告通过。最终 Ruff/mypy、179 unit+smoke、189 full/87.02%、build 全通过；删除已被替代的无效 7.6GB staging。接受 ADR-039。 |
| v1.01 | 2026-08-04 | Codex | `ANL-FUS-001` 转 REVIEW：新增仅复用 4 个可配置正式模型和 fusion registry 的 ExperimentTask matrix，固化 config/seed/data fingerprint/canonical fusion；paper-private 近似替换被拒绝，生产模型前反向与边界专项 2 passed。legacy 17 个平行实现待 CLI 替换后删除。 |
| v1.00 | 2026-08-04 | Codex | 完成 `ANL-EFF-001`：效率 report 增补 accelerator/PyTorch/CUDA runtime；最终 AntM2C fingerprint 的真实 V100 256-event inference 经 10 warm-up + 50 measured 得 2.7244 ms/step、93,965.27 events/s、40,358,400 peak bytes，版本化 report 完整性加载通过。 |
| v0.99 | 2026-08-03 | Codex | `PLOT-001` 转 REVIEW：新增 completed result-v1/finite metric 标准 reader 与包含配置、脚本版本、逐输入 SHA-256 的原子 figure provenance；专项、Ruff/mypy 通过，legacy 硬编码绘图脚本迁移待续。 |
| v0.98 | 2026-08-03 | Codex | 按已完成的 PIPE-002 与 ADR-032 退出证据将 S3 转 DONE：13 个严格等价公共 preset 已数值回归，10 个具名 paper-private 默认边界是科研正确性决策而非未完成近似组件化。 |
| v0.97 | 2026-08-03 | Codex | 删除重复的 `modal_robustness_new.py`，robustness legacy 双实现收敛为单入口；修复 ExperimentRunner 在 RunContext 创建失败时未归还 device、可能阻塞后续任务的问题并新增回归，专项 3 passed、Ruff/mypy 通过。 |
| v0.96 | 2026-08-03 | Codex | `ANL-ALI-001` 转 REVIEW：新增临时具名 activation hook 与 canonical cosine/MSE auxiliary-loss 协议，不改 logits、模型/optimizer 所有权；2 项专项、Ruff/mypy 通过。legacy alignment 专用 Trainer/批量入口物理收敛待续。 |
| v0.95 | 2026-08-03 | Codex | `ANL-EFF-001` 转 REVIEW：统一 warm-up、参数量、CPU/CUDA 同步计时、吞吐与 peak allocated memory，版本化完整性 report 落盘；2 项专项、Ruff/mypy 通过。待 AntM2C 转换完成后以真实 batch 生成 Linux GPU report。 |
| v0.94 | 2026-08-03 | Codex | 完成 `ANL-COLD-001`：user/item cold/zero/few-shot 的 event 集不相交与 target 暴露约束落地；完整分区 fingerprint、原子版本化 audit manifest 及篡改拒绝专项 2 passed，Ruff/mypy 通过。继续推进 AntM2C 全量转换、校验和真实模型门禁。 |
| v0.93 | 2026-08-03 | Codex | `ANL-ROB-001` 转 REVIEW：完成确定性、非修改、target/history 一致的 canonical modality dropout 与 split wrapper，2 项专项及 Ruff/mypy 通过；legacy 双入口物理合并仍待续。继续领取 `ANL-COLD-001` 的集合约束与 manifest audit。 |
| v0.92 | 2026-08-03 | Codex | 完成 `TUNE-001`：trial test-metric 污染守卫、validation-only 严格 AUC 选优、完整 provenance frozen selection 和冻结后独立 final-test 结果边界通过；unit+smoke 167 passed，Ruff/mypy 通过。继续领取 `ANL-ROB-001`。 |
| v0.91 | 2026-08-03 | Codex | 完成 `EXP-001`：新增不可变任务身份、显式 device queue、唯一 run context、任务级失败隔离、completed-only resume 和版本化原子 matrix/result；2 项专项及 unit+smoke 166 passed，Ruff/mypy 通过。S4 转 `IN_PROGRESS`，继续领取 `TUNE-001`。 |
| v0.90 | 2026-08-03 | Codex | 按 ADR-032 的科研边界完成 `PIPE-002`：13 个严格等价公共 preset 可执行，10 个论文私有默认路径均具名且拒绝近似替换，23 项专项回归通过；由此领取 `EXP-001`，计划建立任务矩阵、设备分配、失败/resume 和版本化结果 schema。 |
| v0.89 | 2026-08-03 | Codex | 领取 `ANT-007` 并接受 ADR-038：以 `sharded-named-npy-v1` 建立源文件/行稳定事件身份、原子 shard、逐数组 hash、跨进程恢复和最终原子发布。首个真实源文件试跑确认 packed/shared title 冲突约 0.19%；修正正式协议为保留交互、共享 store 胜出并记录冲突，错误策略的半成品已移到项目 `.tmp` 作为可恢复证据；转换完成并通过三 split 真实模型单步前不更新正式 config fingerprint。 |
| v0.88 | 2026-08-03 | Codex | 完成 `ANT-006` 并据真实证据完成 `ANT-001`：新增可复核 benchmark API、跨 source-shard/domain 分层取样、packed/shared-item 冲突审计、legacy→canonical Batch 等价 checksum 和 tracked JSON 报告；6,144 条样本中 title 冲突 train 4/val 10、image 0，entity 有冲突的 item 为 247/2,820，故 entity 明确定为 interaction context。named arrays 交互体积 -24.82%、warm-cache CPU 吞吐 6.09×，接受 ADR-036/037 并登记全量 `ANT-007`。loader 以五段具名 context 动态组合 3840 维 user text，真实 `dnn_mm_seq` 前反向/Adam step loss 0.686935。Linux `bm` Ruff 117 files、mypy 71 files、unit+smoke 163 passed、全量 173 passed/87.53%；no-isolation wheel 376 KiB/SHA-256 `ff312cd7bef65eb6a32c529cbacc3e3819ff08a9d28591c9437460f81ef5049f` |
| v0.87 | 2026-08-03 | Codex | 领取 `ANT-006` 并接受 ADR-035：从只读真实 AntM2C TFRecord 固定前缀构造当前项目内 `benchmark-v1` 候选，分别报告共享 item store、interaction 增量体积和全库外推；以相同样本、batch size 和端到端解析/gather/PyTorch 组装协议比较 TFRecord 与 named arrays，TDD 覆盖语义等价、确定性报告及无位置切片的新 loader；GPU wait 仅在显式 H2D 同步测量时登记，Linux `bm` 结果待回填 |
| v0.86 | 2026-08-03 | Codex | 完成 `DATA-002`：新增可 memory-map 的 canonical-v1 具名数组 store、MicroLens/TikTok 幂等原子预处理与直接 `Batch` loader，正式 registry/config 已切换，Trainer 只对 legacy source 适配并保留 canonical manifest。MicroLens 真实 3,600,000 交互输出 2,880,000/360,000/360,000（834 MiB，manifest `1e28692d5b7f5c722c6621f78533a1974eef9f9b64256a6a3e51e8a27889221f`）；TikTok 使用官方 split、因果历史和 1:5 负采样输出 357,246/18,306/36,780（50 MiB，manifest `a908bb25f28f899bf2cf21f81483ae67d2106adea582e0e5576a2e9cb1ff05bc`），不再继承 legacy split 泄漏。两套真实 loader→`dnn_mm_seq` 均完成有限前反向/Adam step；Linux `bm` Ruff 115 files、mypy 70 files、unit+smoke 160 passed、全量 170 passed/87.20%；no-isolation sdist+wheel 通过，wheel 369 KiB/SHA-256 `fe3a7939b5f1957141879655d783c75f23c8d57dd6aa8dc06a3e2bda4af9109b` |
| v0.85 | 2026-08-03 | Codex | 领取 `DATA-002` 并接受 ADR-034：优先把 MicroLens/TikTok 处理为当前项目内 canonical-v1 具名数组与去重 item feature store；MicroLens 显式版本化确定性 8:1:1 切分，TikTok 改用官方 split + 因果 prefix history，隔离 legacy 合并三 split 后随机切分的数据泄漏。计划按 TDD 完成合成 round-trip、manifest/ID/mask、泄漏守卫和两数据集 loader→canonical 模型前反向，再对原始只读目录中的真实数据执行到当前项目 `data/processed/`；Linux `bm` 门禁待回填 |
| v0.84 | 2026-08-03 | Codex | 完成 `TRAIN-002` 并将 `MOD-003`/`MODEL-BASE-002` 转入 `REVIEW`：GMMF 迁为第 23 个 canonical CTR 模型，保留 DSN/CGAN/auto-difference/user gate/cosine pooling 与重建目标；新增 checkpoint-compatible `PhasedAdam` 和显式 `AlternatingPhase`，严格保持 main→discriminator→generator 及 `epoch >= N`，并封闭 closure 绕过参数组隔离的边界。冻结 legacy 公式在 logits/reconstruction/discriminator/generator 上均以 `atol=1e-7` 回归，DMF 成为第 13 个可执行 preset。Linux `bm` Ruff 106 files、mypy 63 files、unit+smoke 156 passed、全量 166 passed/87.29%、no-isolation sdist+wheel 通过，wheel 366001 bytes/SHA-256 `c53b57024576dd36c80f75ee723c1ca23d1c2a7c8943bb72984fc6754387dcb4`；正式 registry 已全部 canonical，legacy 物理移除和 10 个 paper-private preset 留待后续 |
| v0.83 | 2026-08-03 | Codex | 领取 `TRAIN-002` 并接受 ADR-033：为 GMMF 登记 checkpoint-compatible phased Adam 与显式交替阶段协议；保持互斥 main/discriminator/generator 参数组、每 batch 固定顺序及 `epoch >= N` 启动条件，计划补齐参数更新隔离、checkpoint resume 和普通 optimizer 不变性测试；Linux `bm` 门禁待回填 |
| v0.82 | 2026-08-03 | Codex | 继续 `PIPE-002`：领取 DMF 默认 pipeline 切片；仅抽取 DTA/相似度分层之前可严格等价的非 ID target/history 模态中心与 user 分支，保持 projector/fusion 共享，并计划以固定权重 `atol=1e-7` 数值回归；不把 DTA 私有 head 近似成公共组件，Linux `bm` 门禁待回填 |
| v0.81 | 2026-08-03 | Codex | 继续 `PIPE-002`：QARM 默认 preset 进入可执行覆盖，按 ID 与多层量化模态的真实输出维度配置 target/history/user 分支并完成固定权重 `atol=1e-7` 数值回归；当前 12 个 executable、11 个 paper-private。专项 14 passed，Ruff 104 files、mypy 62 files、unit+smoke 145 passed、全量 155 passed/87.15%、sdist+wheel 通过，wheel SHA-256 `bf76ec90a6ec217da5f6263e3198689a7d203051537dca253bb472464a04e260`；任务保持 `IN_PROGRESS` |
| v0.80 | 2026-08-03 | Codex | 继续 `PIPE-002`：为全部 23 个正式模型建立可执行/model-specific 覆盖决策，11 个公共等价 preset 完成参数共享与 `atol=1e-7` branch 数值回归；DIN 探针发现旧 Dice/padding bias 约 `1e-3` 差异后按科研红线保留私有。专项 13 passed，Ruff 104 files、mypy 62 files、unit+smoke 144 passed、全量 154 passed/87.13%、sdist+wheel 通过，wheel SHA-256 `1b6465bf460f5693fdf5d3e6e372379d9d97de36b66601d6d04f303e2f108ada`；12 个 paper-private 路径待续，任务保持 `IN_PROGRESS` |
| v0.79 | 2026-08-03 | Codex | 领取 `PIPE-002` 并接受 ADR-032：首批为 DNN-MM/DNN-MM-Seq/LMF/MTFN/NAML/MAKE 生成可执行默认 branch preset，以固定权重对比现有 branch 与 modal pipeline；paper-private 模型只登记缺口，不做公式近似替换 |
| v0.78 | 2026-08-03 | Codex | 完成 `PIPE-001`：新增 target/history/user branch set、严格 resolved mapping、`feature_fusion` 及三种 history topology，统一 projection、presence/mask、target-aware pooling、fusion、adapter 和 typed output；8 项专项、Ruff 102 files、mypy 61 files、unit+smoke 131 passed、全量 141 passed/86.94%、sdist+wheel 通过，wheel SHA-256 `8ed96027d707604e111d77ee058362c6ba75554e859ce5f6d06e1d455c76ba79` |
| v0.77 | 2026-08-03 | Codex | 领取 `PIPE-001` 并接受 ADR-031：以冻结 typed config 组合现有 projection/pooling/fusion/adapter，计划实现 `pool_then_fuse`、`fuse_then_pool`、`sequence_fusion`，在构造阶段校验 capability 和 target 依赖，并以逐条 TDD 覆盖具名输入、mask/presence、all-padding、辅助损失和 backward；Linux `bm` 门禁待回填 |
| v0.76 | 2026-08-03 | Codex | 完成 `FUSE-001`：新增六类具名 fusion、类型 `FusionOutput`、presence/output_dim/aux-loss capability 与 lazy registry，并以 state-key 兼容方式收敛简单 canonical 模型的私有实现；Linux Ruff/mypy 60 files、133 tests/87.11%、sdist+wheel 全通过，wheel SHA-256 `dccc853b357afff7f9f26de62f47b366e1ec92df8cfa10473555edc58f994ade`。同据服务器证据完成 `CORE-001`、`DATA-001`、`TRAIN-001`、`MODEL-BASE-001`、`COMP-001`、`POOL-001` |
| v0.75 | 2026-08-03 | Codex | 完成 `QA-002`：修复拉取后 Ruff/mypy 全部 RED 项、DIN 签名偏差、量化配置校验和 AntM2C 只读 tensor 警告；Linux `bm` 的 pip check、Ruff、mypy 58 files、125 tests/86.99%、sdist+wheel 全通过，wheel SHA-256 `422f1e4f8b93e847c9676ac6613766a21903de7c51781c83b0e6c848d0a89059`。领取 `FUSE-001` 并接受 ADR-030，开始建立类型 fusion 结果、capability 和 lazy registry |
| v0.74 | 2026-08-03 | Codex | 服务器快进拉取至 `d00ac83`，保留拉取前未提交改动于可恢复 stash；领取 `QA-002`，Linux `bm` 首轮 unit+smoke 115 passed，同时记录 Ruff format 31 files、lint 4 项、mypy 23 项和 AntM2C 只读 memmap tensor 警告的 RED 基线，开始修复并执行完整 Linux 门禁 |
| v0.73 | 2026-08-03 | Codex | `POOL-001` 实现转 `REVIEW`：新增统一 SequencePooling/PoolingCapability 与 mean/sum/max/attention/DIN/cross-attention 六类实现、lazy registry 和 `average` alias；reduce、DinAttention、NAML attention 在公式/state key 等价前提下委托公共组件，其他论文私有 attention 保留；补齐 registry/capability、shape/mask/all-padding/target/head/dtype/dimension/backward/state-key 测试及文档。静态计数为 6 类/6 specs，改造 Python 100 列、冲突、裸 squeeze、直接 history-mask 乘法和 projector ModuleDict 重复均为 0，`git diff --check` 通过；Python/Linux 门禁未运行 |
| v0.72 | 2026-08-03 | Codex | 领取 `POOL-001` 并接受 ADR-029：统一 pooling 签名及 capability，计划实现 mean/sum/max/attention/DIN/cross-attention lazy registry，只有公式/state key 等价的现有 helper 才委托公共实现；补齐 shape/mask/all-padding/backward/registry 测试，非服务器环境不执行 Python 门禁 |
| v0.71 | 2026-08-03 | Codex | `COMP-001` 实现转 `REVIEW`：新增严格具名投影、identity/linear 维度适配、序列/缺失 mask 与 all-padding masked-softmax；迁移基础、pooled/sequence/advanced/specialized/quantized canonical 公共路径，非 ID 零向量在 bias 后归零且 history 与 padding mask 求交；Named projector 继承 ModuleDict 保持原 state key，补齐 dtype/rank/dimension/缺失/backward/state-key 合约测试和组件文档；Python/Linux 门禁未运行 |
| v0.70 | 2026-08-03 | Codex | 领取 `COMP-001` 并接受 ADR-028：建立只处理已编码连续张量的具名 projection、严格 dimension adapter、sequence/missing mask 公共边界；计划迁移 pooled/sequence canonical 共享主干，保持 ID embedding、量化编码和论文特有变换在模型侧，并补齐 dtype/rank/dimension/缺失/不变性测试；非服务器环境不执行 Python 门禁 |
| v0.69 | 2026-08-03 | Codex | `MOD-005` 实现转 `REVIEW`：新增独立 lazy quantizer registry 与 `list-quantizers`，RQ/PSRQ 迁为纯预训练组件；建立版本化、逐数组校验、原子且无 pickle 的 NPZ 工件，独立 `quantization_artifact_dir` 与组合层结构/数据集/模态校验；QARM/MCCA 迁为第 21/22 个 canonical CTR 模型，显式注入 RQ/PSRQ、mask padding/零模态并固定 MCCA 量化器 eval。重写两预训练入口并补齐工件往返/篡改/encode-decode/预训练及 CTR 前反向/边界/registry/CLI/config 测试；`git diff --cached --check`、改造 Python 100 列、冲突/裸 squeeze/固定切片/构造期加载文本审计通过；23 CTR + 2 quantizer 注册名均唯一；Python/Linux 门禁未运行，旧 Codebook_Tuner 明确保留为 legacy 回归边界 |
| v0.68 | 2026-08-03 | Codex | 领取 `MOD-005` 并接受 ADR-027：RQ/PSRQ 建立独立量化预训练边界，计划以版本化无 pickle NPZ 固化结构/模态/状态；QARM/MCCA 迁为纯 canonical CTR 模型，通过 registry/Trainer 组合层注入已加载工件，消除构造期路径 I/O；同步改造量化训练入口、配置、文档和 save/load/mask/smoke 合约测试；非服务器环境不执行 Python 门禁 |
| v0.67 | 2026-08-03 | Codex | `MOD-004` 实现转 `REVIEW`：MB/PAMD/MMMLP/M3SRec 整组迁入 specialized canonical 模块，正式模型数增至 20；保留四模型主体与 MB/PAMD 具名辅助损失，统一显式 mask、稳定 batch rank、context fallback 和 left/all-padding 语义；补齐合约/前反向/配置守卫/legacy metadata 测试。`git diff --cached --check`、3 个改造 Python 文件 100 列、冲突/裸 squeeze/固定切片/全局 anomaly 副作用文本检查均通过；registry 为 25 个名称加 1 个 alias、无重复、20 个 canonical 模型；Python/Linux 门禁未运行 |
| v0.66 | 2026-08-03 | Codex | 领取 `MOD-004` 并接受 ADR-026：整组迁移 MB/PAMD/MMMLP/M3SRec，计划保留各自模态平衡、pairwise disentanglement、Mixer、shared-attention/MoE 主体并统一 canonical mask/output；GMMF 的 multi-optimizer 缺口继续独立登记，不再串行阻塞不依赖该协议的四模型迁移；非服务器环境不执行 Python 门禁 |
| v0.65 | 2026-08-03 | Codex | 完成 EM3/Diff-MSIN canonical 代码迁移：正式模型数增至 16；新增 advanced-sequence 模块，保留 FQ-Former/CIC、SRC、specific/shared experts、两级门控、CrossNetwork 及标签 hinge/cosine 目标，历史 DIN 全部显式使用 mask，辅助目标拆为具名 scalar；补齐 forward/backward、all-padding、batch-size-1、context fallback/Batch 不变性、配置守卫与 legacy metadata 测试。`git diff --cached --check`、3 个改造 Python 文件 100 列、冲突/裸 squeeze/固定切片/全局 anomaly 副作用文本检查均通过；registry 为 25 个名称加 1 个 alias、无重复、16 个 canonical 模型；Python/Linux 门禁未运行；GMMF 等待 multi-optimizer engine 协议，`MOD-003` 保持 `IN_PROGRESS` |
| v0.64 | 2026-08-03 | Codex | 继续 `MOD-003`：登记 EM3/Diff-MSIN canonical 迁移，新增独立 advanced-sequence 模块范围；EM3 保留 FQ-Former/CIC，Diff-MSIN 保留 SRC、specific/shared experts、双层门控、cross network 与标签辅助损失。GMMF 因三优化器/交替 GAN 日程依赖暂缓，避免 forward-only 迁移改变训练协议；非服务器环境不执行 Python 门禁 |
| v0.63 | 2026-08-03 | Codex | 完成 DMF/MARN canonical 迁移：正式模型数增至 14；DMF 的 DTA/SimTier 和 MARN 的 DIN 均显式使用 history mask，MARN 三项旧辅助损失拆成具名 scalar 且保留 `lambda0` 权重；补齐 forward/backward、all-padding、batch-size-1、context fallback/Batch 不变性、配置守卫和 legacy metadata 测试。`git diff --check`、3 个改造 Python 文件 100 列、冲突/裸 squeeze/固定切片/全局 anomaly 副作用文本检查均通过；Python/Linux 门禁未运行，`MOD-003` 保持 `IN_PROGRESS` |
| v0.62 | 2026-08-03 | Codex | 继续 `MOD-003`：登记 DMF/MARN canonical 迁移范围；DMF 保留非 ID 模态中心、DTA 与 SimTier 双分支，MARN 保留 specific/invariant、梯度反转判别和原辅助损失权重，同时统一显式 history mask、纯 `Batch -> ModelOutput`、具名辅助损失与 legacy 回归元数据；非服务器环境不执行 Python 门禁 |
| v0.61 | 2026-08-03 | Codex | 完成本批非服务器静态审计：`git diff --check` 通过，6 个新增/改造 Python 文件 100 列超限 0、冲突标记 0、正式 registry 25 个 canonical 名称加 1 个 alias 无重复、canonical 模型模块计数 12、模型目录裸 `.squeeze()` 与固定 `text_full[...]` 切片均为 0；清理未使用导入并补充正维度/dropout 配置守卫及 context fallback/Batch 不变性测试；未执行 Python/pytest/Linux/CUDA 门禁，`MOD-003` 保持 `IN_PROGRESS` |
| v0.60 | 2026-08-03 | Codex | NAML/MAKE 与 DNN-MM-Seq 共享序列编码边界完成：正式 registry 新增 2 个 canonical 模型，NAML attention、MAKE DIN/相似度 tier 全部显式使用 history mask，补齐 all-padding/batch-size-1/forward-backward/legacy metadata 测试；仅静态审查，Python/Linux 门禁未运行 |
| v0.59 | 2026-08-03 | Codex | 继续领取 `MOD-003`：登记 DNN-MM-Seq 公共序列编码边界整理和 NAML/MAKE canonical 迁移范围；计划补齐 masked attention/DIN、相似度分层、稳定 batch rank、registry/legacy bridge 与合成契约测试，非服务器环境不执行 Python 门禁 |
| v0.58 | 2026-08-03 | Codex | 收尾强化 canonical/compat 边界：checkpoint resume 恢复完整 early-stop 状态并校验 run/patience；legacy adapter 显式合并 interaction context 且拒绝同名冲突；对应测试已写未运行 |
| v0.57 | 2026-08-03 | Codex | 领取 `MOD-003` 并迁移首个复杂模型 SimCEN：分割/多层专家纯模块化，contrastive loss/representations 进入 `ModelOutput`，正式 registry 接入；测试已写未运行，其余复杂模型待续 |
| v0.56 | 2026-08-03 | Codex | `MOD-002` 实现转 `REVIEW`：DNN-MM/DNN-MM-Seq/LMF/MTFN 迁入纯 canonical 模块，正式 registry/Trainer 接入，私有 fusion 只覆盖迁移范围，forward/backward 测试已写未运行；接受 ADR-025 |
| v0.55 | 2026-08-03 | Codex | `ANT-005` 实现转 `REVIEW`：新增一等 `Batch.context_features`、具名 InteractionTable/候选 array store/memory-map canonical loader，删除新路径中的 4608 固定切片依赖；真实三 split 审计与格式基准延后；接受 ADR-024 |
| v0.54 | 2026-08-03 | Codex | `ANT-004` 实现转 `REVIEW`：新增单次 encoder 加载、batch shard 原子写入、source fingerprint/checksum resume 与 missing manifest；测试已写未运行，真实模型吞吐门禁延后 |
| v0.53 | 2026-08-03 | Codex | 回填当前非服务器静态审计：`git diff --check` 通过，新增/改造范围 100 列超限 0、冲突标记 0、25 模型/3 数据集 registry 均无重复、三个 dependency-light `__init__` 无直接 torch import；未运行任何 Python/pytest/Linux/CUDA 门禁，`MODEL-BASE-002` 保持 `IN_PROGRESS` |
| v0.52 | 2026-08-03 | Codex | `ANT-002/003` 实现转 `REVIEW`：新增稳定事件单次历史扫描、重复 item/未来泄漏守卫、连续 item index、去重具名 feature store、缺失 audit 及单元测试；仅文本静态检查，真实数据门禁延后 |
| v0.51 | 2026-08-03 | Codex | `ANT-001` 实现转 `REVIEW`：具名字段 ownership/schema、candidate 一致性审计函数和协议文档已写，真实数据确认延后；领取 `ANT-002/003` 纯历史扫描与 item store 实现 |
| v0.50 | 2026-08-03 | Codex | 领取 `ANT-001` 并接受 ADR-023：登记六类文本/图像字段 ownership、item_entity_names 待全量一致性审计边界、稳定 event_id/时间排序与 split embedding 协议 |
| v0.49 | 2026-08-03 | Codex | 完成首批 canonical CTR 主干并将 `MOD-001` 转 `REVIEW`：DNN/DCN/DeepFM/AutoInt/DIN、正式 registry 和主 Trainer 已接入统一 Batch/engine；修复 core/data lazy export 以保持 CLI 不加载 torch；仅完成文本静态检查，服务器门禁延后 |
| v0.48 | 2026-08-03 | Codex | `MODEL-BASE-001` 实现转 `REVIEW`：新增纯统一基类、history capability、mask-aware helper、legacy 输入隔离适配和弃用标记；测试已写未运行；领取 `MODEL-BASE-002` 首批基础 CTR 迁移 |
| v0.47 | 2026-08-03 | Codex | `TRAIN-001` 核心实现转 `REVIEW`：新增统一 engine/evaluator/optimizer/early-stop/checkpoint/resume 与指标写入，测试已写未运行；领取 `MODEL-BASE-001` 并接受 ADR-022 |
| v0.46 | 2026-08-03 | Codex | `REG-001` 实现转 `REVIEW`：建立 25 模型/3 数据集唯一 lazy registry、alias/capability 元数据及 helper 兼容委托；测试已写未运行；领取 `TRAIN-001` 并接受 ADR-021 |
| v0.45 | 2026-08-03 | Codex | `DATA-001` 实现转 `REVIEW`：新增统一 loader protocol、legacy adapter、版本化 manifest、数据文档和三 adapter 合约测试；未改变 AntM2C 数据语义且门禁延后；领取 `REG-001` 并接受 ADR-020 |
| v0.44 | 2026-08-03 | Codex | `CORE-001` 实现转 `REVIEW`：新增统一核心 dataclass、严格契约、设备迁移、legacy adapter、架构文档和单元测试；因当前非服务器环境未执行门禁；领取 `DATA-001` 并接受 ADR-019 |
| v0.43 | 2026-08-03 | Codex | 领取 `CORE-001` 并接受 ADR-018：登记统一 `Batch`/`ModelOutput`/`RunResult`、严格契约校验、设备迁移、legacy 显式适配及非服务器环境暂缓门禁的实施边界 |
| v0.42 | 2026-07-31 | Codex | 完成 `CI-001`：将 origin 切换为已认证 SSH 并快进推送 `66db7d7`；首次 GitHub Actions Linux CPU quality run `30642128515` 在 clean Ubuntu runner 上成功，远端提交 SHA 与本地一致 |
| v0.41 | 2026-07-31 | Codex | `CI-001` 实现转 `REVIEW`：新增固定 Ubuntu 22.04/Python 3.8 CPU workflow 和 YAML 合约测试，采用当前 actions v6，项目本地化全部缓存/临时目录；本地 Ruff/mypy、25 unit+smoke、35 tests/83.80% 与 137-file wheel 隔离构建通过，首次远程 clean-runner 运行仍待推送后取得 |
| v0.40 | 2026-07-31 | Codex | 领取 `CI-001`：登记 Linux-only Ubuntu/Python 3.8 CPU workflow、项目本地缓存、CPU PyTorch 安装、阶段质量门禁与 workflow 合约测试；远程 runner 通过前不标 `DONE` |
| v0.39 | 2026-07-31 | Codex | 完成 `ENV-001`：在保留原始 snapshot、NumPy/PyTorch/CUDA 主栈的前提下对齐 TensorFlow/TFRecord、h5py、typing 与传递依赖；`pip check`、JAX/loader/TFRecord、PyTorch/TensorFlow 8×V100、34 tests/83.80% 和隔离构建全部通过；同步更新 Linux-only 文档和最终包哈希 |
| v0.38 | 2026-07-31 | Codex | 接受 ADR-017 并进入 `ENV-001` 依赖修复：收窄 TensorFlow/Python 3.8 兼容范围，登记 typing-extensions 与 dev 解析边界，新增可选依赖感知的真实 TFRecord round-trip smoke；安装与全门禁结果待回填 |
| v0.37 | 2026-07-31 | Codex | 接受 ADR-016：后续改造与验收收敛到当前 Linux 服务器并授权修复 `bm` 依赖；据既有 Linux 门禁将 `QA-001` 转 `DONE`，恢复 `ENV-001` 为 `IN_PROGRESS`，记录 8×V100、驱动/CUDA 和 PyTorch GPU 前反向实证及 TensorFlow 依赖 RED 基线 |
| v0.36 | 2026-07-31 | Codex | `QA-001` 实现转 `REVIEW`：固化公共命名空间/测试的阶段 Ruff、mypy、pytest 与 80% coverage 门禁，修复公共配置类型问题，新增质量门禁文档；Linux 全量 33 tests、83.80% coverage、类型/格式/lint/build 均通过，等待 Windows 静态复核 |
| v0.35 | 2026-07-31 | Codex | 记录 `QA-001` dev 工具精确版本和 Linux 初始门禁：pytest 33 项与 coverage/build 通过；将 Ruff 的 160 个 legacy 问题和 mypy 的 6 个公共配置类型问题分层，扩展任务文件范围以实施最小修复 |
| v0.34 | 2026-07-31 | Codex | 维护者明确授权在 `bm` 中按需安装并要求临时文件/数据留在项目磁盘；解除 `QA-001` 阻塞、恢复为 `IN_PROGRESS`，继续 dev 工具安装与 Linux 门禁基线 |
| v0.33 | 2026-07-31 | Codex | 将 `QA-001` 标为 `BLOCKED`：共享 `bm` 缺少 dev 门禁工具，安装请求因尚无维护者明确授权被拒绝且未改变环境；记录版本范围、影响和解除条件 |
| v0.32 | 2026-07-31 | Codex | 领取 `QA-001`：登记 Linux `bm` 上 pytest/Ruff/mypy/coverage 分阶段门禁范围、现有 33 项回归基线和 dev 工具安装授权边界 |
| v0.31 | 2026-07-31 | Codex | 回填首次 Linux `bm` 服务器实证：记录绝对解释器、OS、框架/CPU smoke、33 项回归通过，以及 `pip check` 冲突、开发工具缺失和 CUDA 会话不可用的精确缺口；`ENV-001` 保持 `REVIEW` |
| v0.30 | 2026-07-31 | Codex | 固化服务器 VS Code/Codex 交接：新增根 `AGENTS.md`，记录 `bm` 环境、项目同盘下载/缓存、原始版本只读参考规则；接受 ADR-015，`ENV-001` 保持 `REVIEW` |
| v0.29 | 2026-07-31 | Codex | 完成 `CLI-001`：新增 import-safe console/module CLI 与四个子命令，显式参数化主 Trainer，移除 import-time argparse/硬编码线程，并通过仓库外 wheel 入口验证 |
| v0.28 | 2026-07-31 | Codex | 领取 `CLI-001`；接受 ADR-014，登记统一 console/module CLI、lazy train runtime、显式 Trainer 参数化及 help/import/list/config tests |
| v0.27 | 2026-07-31 | Codex | 完成 `CFG-002`：新增 ignored local-path/environment/CLI 注入与严格解析，迁移全部缺失 local YAML 调用方，提交无真实路径 example 和回归测试 |
| v0.26 | 2026-07-31 | Codex | 领取 `CFG-002`；接受 ADR-013，登记 ignored local paths、环境变量优先级、无真实路径 example、canonical 路径解析与主训练接入范围 |
| v0.25 | 2026-07-31 | Codex | 完成 `CFG-001`：建立 frozen training schema、严格唯一键 YAML/分层合并/项目根路径规则，接入主训练并消除双 best-params tracked 配置 |
| v0.24 | 2026-07-31 | Codex | 领取 `CFG-001`；接受 ADR-012，登记 frozen training schema、唯一键 YAML、严格校验、项目根路径规则、分层合并与双 best-params 配置治理 |
| v0.23 | 2026-07-31 | Codex | 完成 `PKG-001`：建立 `mmctr` 公共命名空间和 legacy 兼容桥，以 lazy helper 解开循环导入，迁移核心 imports，并通过仓库外 wheel 导入验证 |
| v0.22 | 2026-07-31 | Codex | 领取 `PKG-001`；接受 ADR-001，登记 lazy import 解环、`mmctr` 公共命名空间、legacy 兼容桥、核心导入迁移和仓库外 wheel 导入验证 |
| v0.21 | 2026-07-31 | Codex | 完成 `RUN-001`：新增唯一 run context、原子产物/生命周期元数据、主训练入口隔离、文档及 32 路并发防冲突回归测试 |
| v0.20 | 2026-07-31 | Codex | 领取 `RUN-001`；接受 ADR-005，登记唯一 run ID、原子目录/元数据写入、主训练入口接入和并发隔离测试范围 |
| v0.19 | 2026-07-31 | Codex | 完成 `SCI-001`：legacy 普通/码本 tuner 改为 validation-only 选优，新增共享选择协议和运行时/AST 防回归测试 |
| v0.18 | 2026-07-31 | Codex | 领取 `SCI-001`；记录两处 legacy tuner 的 test-set 泄漏、validation-only 修复范围与自动化守卫，执行既有 ADR-004 |
| v0.17 | 2026-07-31 | Codex | 完成 `BASE-001`：冻结 legacy DNN 的完整合成 CPU 数值/环境基线并新增 logits、loss 与参数量回归测试 |
| v0.16 | 2026-07-31 | Codex | 领取 `BASE-001`；登记 legacy DNN 合成 CPU 行为基线的配置、输入、版本、数值证据和跨环境边界 |
| v0.15 | 2026-07-31 | Codex | 完成 `TEST-001`：新增合成 batch、pooling unit 和 legacy DNN CPU smoke；记录并隔离现有直接导入循环，未安装缺失的 pytest |
| v0.14 | 2026-07-31 | Codex | 领取 `TEST-001`；登记无真实数据/网络/GPU的合成 batch、pooling unit、DNN CPU smoke 与 pytest 缺失时的 unittest 本地验证方式 |
| v0.13 | 2026-07-31 | Codex | 完成 `OSS-001` 中 README、CONTRIBUTING 与 CITATION；因许可证类型尚未经维护者决定，将任务标记为 `BLOCKED` 并记录解除条件 |
| v0.12 | 2026-07-31 | Codex | 领取 `OSS-001`；登记 README/贡献/引用文档范围，并将许可证类型列为必须由维护者决定的边界 |
| v0.11 | 2026-07-31 | Codex | 完成 `ENV-002`：建立 `pyproject.toml` 权威元数据与依赖分组、收敛 legacy setup shim、删除空的错误拼写依赖文件并通过本地 wheel 检查 |
| v0.10 | 2026-07-31 | Codex | 记录维护者延期 Linux 验证的决定；新增 ADR-010/011；领取 `ENV-002` 并登记依赖分组、兼容范围和本地验收方式 |
| v0.9 | 2026-07-31 | Codex | 完成 `PUB-001`：净化个人路径与服务器 prefix，基于远程初始提交发布无生成物历史的干净快照 |
| v0.8 | 2026-07-31 | Codex | 新增并领取 `PUB-001`：响应维护者指定 GitHub 远程，登记公开推送前的个人路径与历史生成物净化策略 |
| v0.7 | 2026-07-31 | Codex | `ENV-001` 实现转 `REVIEW`：保留服务器快照，新增可移植 bootstrap 与审计文档；明确 Linux 验证缺口 |
| v0.6 | 2026-07-31 | Codex | 领取 `ENV-001`；登记 Linux 快照保留边界、可移植环境产物和双环境验证范围 |
| v0.5 | 2026-07-31 | Codex | 完成 `GOV-002`：新增 `.gitignore`，清理可恢复的 Python/打包缓存，保留研究图表和本地 IDE 配置 |
| v0.4 | 2026-07-31 | Codex | 完成 `GOV-001`：建立可恢复的改造前本地基线；领取 `GOV-002` 并登记生成物治理边界 |
| v0.3 | 2026-07-31 | Codex | 领取 `GOV-001`；登记 Git 基线的文件范围、保护边界和验证计划 |
| v0.2 | 2026-07-31 | Codex | 明确 Linux 为权威实验环境；按四大阶段重排；加入 AntM2C 无切片数据方案、单一 BaseSeqModel、按模态 pooling/fusion 与分析体系整改 |
| v0.1 | 2026-07-31 | Codex | 首次仓库盘点；建立目标架构、强制规范、科研协议、路线图和 agent 进度机制 |
