# MMCTR Benchmark 全局改造方案与协作规范

> 文档状态：`ACTIVE`  
> 当前版本：`v0.61`
> 最近更新：`2026-08-03`
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
- `/home/star/Disk3/hkl/Benchmark` 是改造开始前的最原始版本，仅作为数据位置、历史配置和旧运行约定的**只读参考**；不得在该目录执行修改、删除或迁移，也不得将其中真实数据和机器专属配置提交到 Git。
- 接续改造的 Codex 必须先读根 `AGENTS.md` 和本文档。以上内容是维护者提供的交接事实，不代表 Linux/CUDA 门禁已经通过；实际服务器验证证据仍需回填 `ENV-001`。

### 4.3 服务器唯一验证规范

- 后续代码修改、静态检查、单元/集成测试、构建、CUDA、真实数据和性能验证全部在本服务器执行。
- Python、pip、pytest、Ruff、mypy 和 build 命令统一使用 `bm` 的绝对解释器
  `/home/star/Disk3/hkl/envs/bm/bin/python`；不得用 Windows 结果补齐门禁。
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
| S1 开源发布与工程基线 | `IN_PROGRESS` | 包元数据、Linux `bm` 依赖/CUDA 基线、`mmctr` 公共命名空间、统一 CLI、严格 training/本地路径配置层、主要公开文档、合成 CPU/TFRecord smoke、首个数值回归基线、tuner 科研红线修复、主训练运行目录隔离、分阶段 Linux QA 门禁及首次 GitHub clean-runner CI 已完成；`OSS-001` 等待许可证 | Linux 安装、公开文档、P0 修复、smoke baseline |
| S2 数据处理与模型主干 | `IN_PROGRESS` | 统一 Batch/ModelOutput/RunResult、dataset manifest/adapter、model/data registry、training engine、单一公共 BaseSeqModel、AntM2C 无切片候选链路与 12 个 canonical 模型已实现并等待服务器门禁；其余复杂/量化模型迁移和真实数据格式基准尚未完成 | 无切片数据链路、统一 Batch、单一 BaseSeqModel |
| S3 模型公共组件 | `TODO` | 未开始 | pooling/fusion 可按模态配置且默认 preset 回归通过 |
| S4 实验分析体系 | `TODO` | 未开始 | 无模型复制、统一 runner/result schema、五类分析可配置运行 |

> 当前结论：S1 工程基线已基本落地；S2 的核心契约、统一 loader/registry/training/base-model 主干、AntM2C 具名候选链路及 12 个 canonical 模型代码已进入 `REVIEW`/`IN_PROGRESS`。当前非服务器工作区未执行 Python 门禁，因此不得把这些实现写成已验收；AntM2C 最终格式基准和其余复杂/量化模型仍未完成。

### 13.2 可领取任务

| ID | Priority | Task | Depends on | Status | Owner | Files | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| GOV-000 | P0 | 仓库盘点并建立全局改造方案 | - | `DONE` | Codex | `REFACTORING_PLAN.md` | 117 files、16,679 LOC、AST 0 failures；2026-07-31 |
| GOV-001 | P0 | 建立/确认 Git 基线，保护现有用户改动 | GOV-000 | `DONE` | Codex (`/root`) | repo metadata、`REFACTORING_PLAN.md` | 改造前 267 个文件已纳入本地可恢复快照；公开分支从远程初始提交生成干净历史；2026-07-31 |
| GOV-002 | P0 | 添加 `.gitignore`，治理 pyc/egg-info/outputs | GOV-001 | `DONE` | Codex (`/root`) | `.gitignore`、generated files、`REFACTORING_PLAN.md` | 移除 105 个 `.pyc`、9 个 `__pycache__` 和 1 个 `egg-info`；本地 `.vscode/settings.json` 保留但取消跟踪；19 个论文图表保留；`git ls-files -ci --exclude-standard` 为 0；2026-07-31 |
| PUB-001 | P0 | 公开推送前净化个人路径与历史生成物，并接入目标远程 | GOV-002 | `DONE` | Codex (`/root`) | hard-coded path files、Git refs/history、`REFACTORING_PLAN.md` | 远程 `origin/main` 非强制前进至 `222591b`；当前树个人路径/常见密钥扫描均为 0；公开历史 pyc/egg-info/IDE 对象为 0；6 个改动脚本 AST 与 4 个 YAML 静态检查通过；原本地历史保留于 `local/refactor-bootstrap`；2026-07-31 |
| ENV-001 | P0 | 审计并保留 Linux server snapshot，生成可移植环境说明 | GOV-001 | `DONE` | Codex (`/root`) | `bm_env.yml`、`environment.yml`、`pyproject.toml`、`docs/environment.md`、`AGENTS.md`、`tests/smoke/test_tensorflow_tfrecord.py`、`REFACTORING_PLAN.md` | 原始 snapshot 保留，现有 `bm` 未新建环境；解释器 `/home/star/Disk3/hkl/envs/bm/bin/python`（3.8.20）。依赖对齐为 TensorFlow/Keras 2.12.1/2.12.0、h5py 3.10.0、typing-extensions 4.5.0、JAX/JAXlib 0.4.13 等，保留 NumPy 1.23.5 与 PyTorch 1.13.1+cu117；`pip check` 无冲突，JAX CPU 与三 loader import/TFRecord round-trip 通过。非沙箱实测驱动 535.183.01、CUDA 12.2、8×V100-32GB，PyTorch 前反向/同步及 TensorFlow GPU matmul 均通过。追加 CI 合约测试后的最终 Ruff 34 files、mypy 14 files、strict unit+smoke 25 passed、全量 35 passed/83.80%；隔离 wheel 137 files/268965 bytes/SHA-256 `c279ede0278cdd3871aefdaf08b3040fac4ddd90b4b849e913ce7e18388d3413`；2026-07-31 |
| ENV-002 | P0 | 建立 `pyproject.toml`、依赖分组和 Python 兼容范围 | ENV-001 | `DONE` | Codex (`/root`) | `pyproject.toml`、`setup.py`、`requrements.txt`、`docs/environment.md`、`REFACTORING_PLAN.md` | 新增 PEP 517/621 元数据与 6 个 optional groups，声明 Python `>=3.8,<3.9`，删除空的错误拼写依赖文件；Windows 3.12.11：TOML、118 AST、20 legacy packages、`setup.py --name` 通过；常规 wheel 正确拒绝 3.12，`--ignore-requires-python --no-deps --no-build-isolation` 构建成功，wheel 120 文件且无缓存；临时产物已清理；Linux 门禁按 ADR-010 延期；2026-07-31 |
| OSS-001 | P0 | README/Quick Start、LICENSE、CITATION、CONTRIBUTING | ENV-002 | `BLOCKED` | Codex (`/root`) | `README.md`、`CONTRIBUTING.md`、`CITATION.cff`、`LICENSE`、`REFACTORING_PLAN.md` | README、贡献指南和实体引用已完成；CFF YAML、2 个文档的本地相对链接、diff/path 扫描通过；未创建 `LICENSE`。阻塞条件：维护者需明确选择许可证类型；外部 Linux 合成 smoke 证据由 `TEST-001` 建立后回填 |
| OSS-002 | P0 | 数据来源、许可、下载、目录与模型引用清单 | OSS-001 | `TODO` | - | `data/README.md`, docs | 无私有数据/路径，第三方许可可追踪 |
| SCI-001 | P0 | 隔离并修复现有 tuner 的 test-set 泄漏 | TEST-001 | `DONE` | Codex (`/root`) | `src/scripts/Tuner.py`、`src/scripts/Codebook_Tuner.py`、`src/utils/tuning_protocol.py`、`tests/`、`REFACTORING_PLAN.md` | 两处搜索流程均改用共享 validation-only evaluator，结果字段改为 `val_auc`/`val_loss`，保留严格更高 AUC 胜出的旧比较规则；三个 tuner 的 test 指标泄漏扫描为 0；Windows 指定解释器完成 `src`/`tests` compileall，最终 `unittest discover` 8 tests/4.563s 全通过；测试后 13 个缓存目录已清理；Linux 验证按 ADR-010 延期；2026-07-31 |
| RUN-001 | P0 | 设计唯一 run ID 和隔离输出/checkpoint | GOV-001 | `DONE` | Codex (`/root`) | `src/utils/run_context.py`、`src/trainers/Trainers.py`、`src/utils/helper.py`、`config/train.yaml`、`docs/run-layout.md`、`tests/`、`REFACTORING_PLAN.md` | run ID 含 UTC 微秒时间戳、10 位配置哈希和 8 位随机熵；原子创建独立 resolved config/metadata/metrics/checkpoint/log/summary 路径，主训练初始化或运行失败均落 `failed`；32 路相同配置同微秒并发目录全部唯一，精确碰撞和路径穿越均拒绝；Windows 指定解释器完成 `src`/`tests` compileall，最终 `unittest discover` 14 tests/10.194s 全通过；25 个缓存目录已清理；tuning/analysis 全面接入留给 `TUNE-001`/`EXP-001`，Linux 验证按 ADR-010 延期；2026-07-31 |
| TEST-001 | P0 | 建立 pytest、合成 batch 和首个 CPU smoke | ENV-002 | `DONE` | Codex (`/root`) | `tests/`、`pyproject.toml`、`README.md`、`REFACTORING_PLAN.md` | 建立 pytest 可收集的 unittest 结构、确定性 ID batch、pooling unit 与 legacy registry DNN CPU smoke；Windows 3.12.11 未安装 pytest，未修改环境；`unittest discover` 4 tests/5.392s 全通过，覆盖 forward/loss/backward/optimizer；首次直接导入暴露 `BaseModel ↔ utils.helper` 循环，改按现有 registry 入口验证并登记给 `PKG-001`；12 个测试缓存目录已清理；Linux pytest 按 ADR-010 延期；2026-07-31 |
| BASE-001 | P0 | 保存重构前可获得的行为/指标基线 | TEST-001 | `DONE` | Codex (`/root`) | `tests/baselines/`、`tests/regression/`、`REFACTORING_PLAN.md` | `legacy_dnn_id_cpu_v1` 保存 schema、registry 入口、完整配置/seed/输入、Windows/Python/Torch/NumPy/sklearn 版本、4 logits、loss 与 205 参数量；容差 `1e-6`；Windows `unittest discover` 5 tests/5.263s 全通过；13 个缓存目录已清理；明确为合成行为而非论文指标，Linux 正式基线按 ADR-010 延期；2026-07-31 |
| PKG-001 | P1 | 创建 `src/mmctr` 包并迁移导入 | ENV-002, TEST-001 | `DONE` | Codex (`/root`) | `src/mmctr/`、legacy import bridge、`src/utils/helper.py`、core imports、docs、`tests/`、`REFACTORING_PLAN.md` | 新增 `mmctr.models`/`mmctr.data`/`mmctr.utils` 公共入口；helper 改为 model/data 调用时加载，直接 model-first 与 helper-first 导入顺序均通过且 DNN 类身份一致；15 个核心 helper 调用方迁至 `mmctr.utils`，legacy 直接 helper import 为 0；Windows 指定解释器完成 `src`/`tests` compileall，`unittest discover` 16 tests/21.870s 全通过；仓库外 cwd 临时安装 wheel 成功，`mmctr` 路径来自目标目录，wheel 260175 bytes、SHA-256 `5410f853c3fcb52076b3210615741e6c3c6dc895ed082b11ef9d6f67de46c524`；临时目录、2 个构建目录和 29 个缓存目录已清理；legacy 顶层包作为显式兼容桥保留，物理迁移随模型/数据任务推进；Linux 验证按 ADR-010 延期；2026-07-31 |
| CFG-001 | P1 | 配置分层、typed schema 和校验 | PKG-001 | `DONE` | Codex (`/root`) | `src/mmctr/config/`、`src/trainers/Trainers.py`、`config/`、`docs/configuration.md`、`docs/legacy_tuning_history.yaml`、`tests/`、`REFACTORING_PLAN.md` | 新增 frozen `TrainingConfig`、唯一键/顶层 mapping YAML loader、项目根发现、无副作用递归 layer merge；严格覆盖必填、未知、类型、范围、optimizer 和 patience 跨字段约束，主训练通过显式 `to_dict()` 兼容边界消费且配置文件路径不依赖 cwd；`best_params.yaml` 历史 test 记录迁出 `config/`，新 legacy 输出写入 ignored `outputs/tuning/`，`best_param.yaml` 成为唯一 tracked 参数快照；6 个可执行 YAML 唯一键检查通过；Windows 指定解释器完成 `src`/`tests` compileall，最终 `unittest discover` 21 tests/20.310s 全通过，30 个缓存目录已清理；算法专属 model/data typed schema 随对应重构任务推进，tuning provenance 随 `TUNE-001` 完成；Linux 验证按 ADR-010 延期；2026-07-31 |
| CFG-002 | P0 | 消除缺失 local config 和服务器个人绝对路径 | CFG-001 | `DONE` | Codex (`/root`) | `src/mmctr/config/paths.py`、training/tuning/analysis callers、`configs/local/paths.example.yaml`、`.gitignore`、`docs/local-paths.md`、`tests/`、`REFACTORING_PLAN.md` | 新增 frozen `LocalPaths`、selected dataset catalog resolver、绝对路径/存在性/未知数据集校验；ignored `paths.yaml` < 环境变量 < 显式 legacy CLI override，canonical 路径按项目根解析且不修改输入；主训练、普通 tuner、alignment、两版 modal robustness 和 analysis trainer 全部移除缺失 local YAML；`src/` 中 `local_data.yaml`/`local_seq_data.yaml` 引用为 0，公开 config/example 个人路径扫描为 0；真实 `paths.yaml` 命中 ignore，example 不被忽略；Windows 指定解释器完成 `src`/`tests` compileall，最终 `unittest discover` 28 tests/19.096s 全通过，30 个缓存目录已清理；Linux local-path smoke 按 ADR-010 延期；2026-07-31 |
| CLI-001 | P1 | 建立统一 CLI，移除 import-time argparse | PKG-001, CFG-001 | `DONE` | Codex (`/root`) | `src/mmctr/cli/`、`src/trainers/Trainers.py`、`src/mmctr/models/`、`src/mmctr/data/`、`pyproject.toml`、`docs/cli.md`、README、`tests/`、`REFACTORING_PLAN.md` | 新增 console/module CLI 与 train/validate-config/list-models/list-datasets 子命令；CLI/catalog import 不加载 torch/TensorFlow，train 设置 5 组线程环境后才 lazy import runtime；主 Trainer 改为显式参数构造，import 不解析宿主 argv 且移除硬编码 24 线程副作用；Windows 指定解释器完成 `src`/`tests` compileall，`unittest discover` 33 tests/26.696s 全通过；仓库外临时 wheel 的 module help、dependency-light import 和 `mmctr = mmctr.cli:main` entry point 均通过，wheel 268941 bytes、SHA-256 `f03e6bf71e6bab2ebda0ff735deccad8b4ed39eba36b5a9a11a5710faf2342a8`；临时目录、2 个构建目录和 32 个缓存目录已清理；真实 train Linux/CUDA gate 按 ADR-010 延期；2026-07-31 |
| QA-001 | P1 | Ruff/pytest/mypy/coverage 分阶段门禁 | ENV-002, PKG-001 | `DONE` | Codex (`/root`) | `pyproject.toml`、`src/mmctr/`、`tests/`、`docs/quality-gates.md`、README、`REFACTORING_PLAN.md` | 已建立明确阶段边界：Ruff 约束 `src/mmctr + tests`，mypy 检查 14 个公共源文件但不递归 legacy bridge，coverage 下限 80%。最终 Linux `/home/star/Disk3/hkl/envs/bm/bin/python`：Ruff format 34 files 与 lint 通过，mypy 1.10.1 的 14 files 通过，unit+smoke 25 passed/7.48s，全量 pytest 35 passed/17.84s、coverage 83.80%，隔离 sdist+wheel build 通过；wheel 137 files/268965 bytes/SHA-256 `c279ede0278cdd3871aefdaf08b3040fac4ddd90b4b849e913ce7e18388d3413`。全仓 legacy 基线仍为 160 lint 问题/116 待格式化文件并已文档化；依据 ADR-016 仅以 Linux 验收；2026-07-31 |
| CI-001 | P1 | 建立 Linux CPU CI | QA-001 | `DONE` | Codex (`/root`) | `.github/workflows/linux-ci.yml`、`tests/unit/test_ci_workflow.py`、`docs/quality-gates.md`、`REFACTORING_PLAN.md` | 以固定 Ubuntu 22.04/Python 3.8 + PyTorch 1.13.1 CPU 执行 pip check、Ruff、mypy、unit/smoke、全量 coverage 和隔离 build，缓存/临时目录全部位于 workspace；TDD 合约测试先因 workflow 缺失为 RED，随后发现并阻止无效 YAML 单行命令，最终 YAML 解析和 runner/action/Python/命令/缓存约束通过。Linux `bm` 复核 Ruff 34 files、mypy 14 files、unit+smoke 25 passed、全量 35 passed/83.80% 及隔离 build；提交 `66db7d7` 的首次 clean GitHub Actions run `30642128515` 全部成功；2026-07-31 |
| CORE-001 | P0 | 定义统一 Batch/ModelOutput/RunResult | PKG-001, CFG-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/core/`、`src/mmctr/__init__.py`、`tests/unit/test_core_schemas.py`、`docs/architecture.md`、`REFACTORING_PLAN.md` | 已建立不可变顶层核心契约、严格 shape/dtype/batch-size 校验、设备迁移与 legacy tuple/dict 显式适配；代码与单元测试已写入，按维护者本轮指示未在非服务器环境执行 Python/Linux 门禁，待服务器验证后转 `DONE`；2026-08-03 |
| DATA-001 | P0 | 统一三个数据集 loader contract 与 manifest | CORE-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/data/`、legacy loader compatibility boundary、`tests/unit/test_data_contracts.py`、`docs/data.md`、`REFACTORING_PLAN.md` | 已建立统一 split/history capability、版本化 manifest 和 `CanonicalDataLoader`，三类 legacy loader 可显式转换为 `Batch`，并分离兼容路径中的组合 user/item ID；未改变 TFRecord 与 AntM2C 4608 字段语义；代码与三 adapter 合约测试已写入，真实数据/Linux 门禁按本轮指示延后；2026-08-03 |
| ANT-001 | P0 | 审计 AntM2C 六类文本字段语义与 feature ownership | DATA-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/`、`docs/data/antm2c.md`、`tests/unit/test_antm2c_schema.py`、`REFACTORING_PLAN.md` | 已固化 service/query/bill/log_time 为 interaction context，title/image 为 item feature，item_entity_names 为待全量一致性审计的 item candidate；已登记稳定 event_id、原脚本精确午夜切分、padding/ID offset、split embedding 和未来泄漏协议。schema/audit 测试已写；真实数据一致性与切分意图确认延后，故不转 `DONE`；2026-08-03 |
| ANT-002 | P0 | 用 event_id + 单次时序扫描重写历史构造 | ANT-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/history.py`、`tests/unit/test_antm2c_history.py`、`docs/data/antm2c.md`、`REFACTORING_PLAN.md` | 已实现稳定 `(user_id,timestamp,event_id)` 排序后每用户单次扫描、只纳入严格更早正反馈、左 padding、重复 item 事件保留、event_id 唯一性和顺序审计；泄漏/重复事件测试已写但未运行，真实全量复杂度与 split 抽样门禁延后；2026-08-03 |
| ANT-003 | P0 | 建立 item index 和去重 item feature store | ANT-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/item_store.py`、`tests/unit/test_antm2c_item_store.py`、`docs/data/antm2c.md`、`REFACTORING_PLAN.md` | 已实现按稳定事件流首次出现顺序分配的连续 item index、padding 0/OOV 规则、具名 float feature table、target/history 同表 gather、缺失零填充与 audit；一致性/维度/范围校验和测试已写，真实数据构建/hash/规模门禁延后；2026-08-03 |
| ANT-004 | P1 | 重写文本/图像 batch 提取、单次模型加载和断点续跑 | ANT-003 | `REVIEW` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/extraction.py`、`tests/unit/test_antm2c_extraction.py`、数据文档、`REFACTORING_PLAN.md` | 已实现单次 encoder factory、缺失值零填充、具名 key、分批原子 `.npy`/JSON 写入、连续 shard resume、source fingerprint/shape/finiteness/checksum 校验；测试已写但未运行，真实文本/图像模型、吞吐和故障恢复门禁留待服务器，故不转 `DONE`；2026-08-03 |
| ANT-005 | P0 | 重写 split 序列化和 loader，删除 4608 拼接/固定切片 | ANT-002, ANT-003, ANT-004 | `REVIEW` | Codex (`/root`) | `src/mmctr/data/datasets/antm2c/array_store.py`、`src/mmctr/core/schemas.py`、`tests/unit/test_antm2c_array_store.py`、数据文档、`REFACTORING_PLAN.md` | 已实现具名 InteractionTable、独立 context/item arrays、item store target/history gather、padding-safe ID offset、memory-map loader 与 canonical `Batch.context_features`；`named-npy-candidate-v1` 仅为 ANT-006 候选且无固定切片，round-trip 测试已写未运行，三 split 真实抽样审计延后，故不转 `DONE`；2026-08-03 |
| ANT-006 | P1 | Linux 基准比较 TFRecord 与候选分层格式 | ANT-005 | `TODO` | - | benchmark/docs | 空间、吞吐、CPU/GPU wait 报告 |
| DATA-002 | P1 | 将 MicroLens/TikTok 对齐统一数据契约 | DATA-001, ANT-005 | `TODO` | - | data/datasets | contract + smoke tests |
| TRAIN-001 | P0 | 统一 training engine、evaluate、early stop、optimizer | CORE-001, RUN-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/training/`、`src/mmctr/evaluation/`、`src/trainers/Trainers.py`、`src/mmctr/utils/run_context.py`、`tests/unit/test_training_engine.py`、`docs/training.md`、`REFACTORING_PLAN.md` | 已建立只消费 `Batch -> ModelOutput` 的 engine、validation-only early stop、显式 optimizer、严格 AUC/LogLoss、原子 `best.pt`/`last.pt` checkpoint、resume 与 run metrics writer；resume 同步恢复 best/best_epoch/bad_epochs/patience 并校验 run ID，避免中断后重置选择状态；主 Trainer 对正式 registry 中已迁移模型自动走 canonical engine，其余模型保留 legacy 路由；单步/save-load/resume/test 隔离测试已写入，Python/Linux 门禁按本轮指示延后；2026-08-03 |
| MODEL-BASE-001 | P0 | 建立单一 BaseSeqModel 与 pooled/token 两种历史能力 | CORE-001, TRAIN-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/models/base.py`、`src/mmctr/models/compat.py`、model public exports/registry、`tests/unit/test_model_base.py`、`docs/models.md`、`REFACTORING_PLAN.md` | 已建立纯 `Batch -> ModelOutput` 单一基类、显式 history capability、mask-aware mean/sum/max/softmax helper、不修改输入映射的 legacy adapter，并将旧双基类标记弃用；pooled/all-padding/adapter 测试已写，Linux 门禁延后；2026-08-03 |
| MODEL-BASE-002 | P0 | 迁移并弃用 BaseModel 子类，先 pool 后复用原主体 | MODEL-BASE-001, DATA-001 | `IN_PROGRESS` | Codex (`/root`) | `src/mmctr/models/baselines/`、`src/mmctr/models/multimodal.py`、`src/mmctr/models/sequence.py`、model registry/legacy bridge、legacy subclasses、regression tests、`docs/models.md`、`REFACTORING_PLAN.md` | 已迁移 DNN/DCN/DeepFM/AutoInt/DIN、DNN-MM/DNN-MM-Seq/LMF/MTFN、SimCEN、NAML、MAKE 共 12 个纯 canonical 模型；正式 registry 指向新实现，helper 历史入口继续指向冻结 legacy 类。物理 legacy 双基类文件仍保留供回归，其余复杂/量化模型尚待迁移，完成前不删除兼容层；2026-08-03 |
| REG-001 | P1 | 模型/数据集 registry（fusion registry 留到 S3） | PKG-001, CFG-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/core/registry.py`、`src/mmctr/models/registry.py`、`src/mmctr/data/registry.py`、public exports、legacy helper bridge、`tests/unit/test_registries.py`、`REFACTORING_PLAN.md` | 已建立稳定 snake_case 名称、`dnn_seq -> dnn_mm_seq` alias、capability metadata 与 lazy import；25 模型/3 数据集只有一份正式注册表，helper 改为兼容委托，fusion/pooling 未抢跑；唯一性/无重依赖导入测试已写入，构造与 Linux 门禁按本轮指示延后；2026-08-03 |
| MOD-001 | P1 | 迁移基础 CTR 模型 | MODEL-BASE-002, REG-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/models/baselines/`、model registry/legacy helper、`src/trainers/Trainers.py`、`tests/unit/test_canonical_baselines.py`、`tests/regression/`、`docs/models.md`、`REFACTORING_PLAN.md` | DNN/DCN/DeepFM/AutoInt/DIN 已迁为纯 `Batch -> ModelOutput`，pooled 模型显式 mask mean，DIN 保留 token 并用 mask attention，batch size 1 不降秩；正式 registry/Trainer 走新实现，helper 保留冻结 legacy 数值入口；五模型 forward/backward 测试已写但未运行，最终验收依赖 `MODEL-BASE-002` 完成和 Linux 回归；2026-08-03 |
| MOD-002 | P1 | 迁移简单多模态模型 | MOD-001 | `REVIEW` | Codex (`/root`) | `src/mmctr/models/multimodal.py`、`src/mmctr/models/sequence.py`、model registry/legacy bridge、`tests/unit/test_canonical_multimodal.py`、`tests/unit/test_canonical_sequence.py`、`docs/models.md`、`REFACTORING_PLAN.md` | DNN-MM、DNN-MM-Seq、LMF、MTFN 已迁为纯 `Batch -> ModelOutput`，具名 target/history/context 投影、显式 padding mask、稳定 batch rank；DNN-MM-Seq 后续整理到共享 sequence-token 编码边界，模型私有 cat/add/mean/MAF/LMF/MTFN 仅覆盖迁移所需能力，不提前建立公共 fusion registry；正式 registry/Trainer 走 canonical，helper 仍提供 legacy 回归入口；forward/backward 测试已写未运行，数值回归与 Linux 门禁延后；2026-08-03 |
| MOD-003 | P1 | 迁移复杂序列/辅助损失模型 | MOD-002 | `IN_PROGRESS` | Codex (`/root`) | `src/mmctr/models/multimodal.py`、`src/mmctr/models/sequence.py`、model registry/legacy bridge、`tests/unit/test_canonical_multimodal.py`、`tests/unit/test_canonical_sequence.py`、`docs/models.md`、`REFACTORING_PLAN.md` | 已迁移 SimCEN、NAML、MAKE，并将 DNN-MM-Seq 整理到共享纯序列编码边界；NAML 使用 mask-aware learned attention，MAKE 使用 mask-aware DIN/相似度 tier，三路投影不修改 Batch、batch size 1 不降秩、all-padding 输出有限；正式 registry 指向 canonical，helper 保留 legacy 回归入口。forward/backward/registry/边界测试已写未运行；Diff-MSIN/MARN/DMF/EM3/GMMF 等尚未迁移，故保持 `IN_PROGRESS`；2026-08-03 |
| MOD-004 | P1 | 迁移 MB/PAMD/MMMLP/M3SRec | MOD-003 | `TODO` | - | multimodal | model-specific regression |
| MOD-005 | P1 | 迁移 RQ/PSRQ/QARM/MCCA | MOD-004 | `TODO` | - | quantization | codebook save/load + smoke |
| COMP-001 | P1 | 统一 projection、mask、dimension adapter | MOD-005 | `TODO` | - | model components | dtype/rank/dimension tests |
| POOL-001 | P1 | 统一 pooling API、registry 与 capability | COMP-001 | `TODO` | - | pooling components | 全 pooling shape/mask/backward |
| FUSE-001 | P1 | 统一 fusion API、registry、output_dim 与 aux loss | COMP-001 | `TODO` | - | fusion components | 全 fusion 组合边界测试 |
| PIPE-001 | P1 | 实现按分支/模态配置的 pipeline 与三种 topology | POOL-001, FUSE-001 | `TODO` | - | modal pipeline/config | compatible/incompatible config tests |
| PIPE-002 | P1 | 为全部模型建立默认兼容 preset | PIPE-001 | `TODO` | - | model configs | 默认 preset 回归与旧实现一致 |
| EXP-001 | P1 | 统一 Linux experiment runner、GPU 调度和结果 schema | TRAIN-001, RUN-001, PIPE-002 | `TODO` | - | experiments | task matrix/resume/failure tests |
| TUNE-001 | P0 | 建立只按 validation 选择的正式 tuner | EXP-001, SCI-001 | `TODO` | - | experiments/tuning | trial 历史与冻结后 test 测试；被提升的参数必须记录 experiment ID、validation 指标、seeds 和数据版本，不直接写 tracked config |
| ANL-FUS-001 | P1 | fusion analysis 删除 17 个平行模型实现 | EXP-001 | `TODO` | - | analysis/fusion | pipeline 配置复用正式模型 |
| ANL-ALI-001 | P1 | alignment 改为 hook/aux-loss protocol | EXP-001 | `TODO` | - | analysis/alignment | 无特殊模型/Trainer 复制 |
| ANL-ROB-001 | P1 | 合并 modal robustness 新旧实现 | EXP-001 | `TODO` | - | analysis/robustness | batch transform + seed tests |
| ANL-COLD-001 | P1 | 规范 cold/few/zero-shot split 与校验 | EXP-001 | `TODO` | - | analysis/cold_start | 集合约束与 manifest tests |
| ANL-EFF-001 | P1 | 统一效率、显存、参数量和 CUDA 计时协议 | EXP-001 | `TODO` | - | analysis/efficiency | Linux GPU protocol report |
| PLOT-001 | P2 | 绘图只读取标准结果，去掉硬编码数据 | ANL-FUS-001, ANL-ALI-001, ANL-ROB-001, ANL-COLD-001, ANL-EFF-001 | `TODO` | - | analysis/plotting | figure provenance/hash |
| REL-001 | P2 | 四阶段完成后的开源 release checklist | CI-001, PLOT-001 | `TODO` | - | root/docs | clean clone Linux reproduction |

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
| ADR-015 | 2026-07-31 | 服务器接续使用既有 Conda `bm` 环境；下载、缓存和临时产物与当前项目同盘；原始 `/home/star/Disk3/hkl/Benchmark` 只读参考 | `ACCEPTED` | 遵循维护者提供的服务器资源边界，避免占用其他磁盘或误改原始版本，同时为数据位置和历史约定保留可核对证据 |
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

---

## 15. 变更记录

| Version | Date | Author | Change |
|---|---|---|---|
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
