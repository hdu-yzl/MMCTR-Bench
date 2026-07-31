# MMCTR Benchmark 全局改造方案与协作规范

> 文档状态：`ACTIVE`  
> 当前版本：`v0.30`
> 最近更新：`2026-07-31`  
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

## 4. 双环境定位与执行规范

### 4.1 环境权威边界

本项目存在两个用途不同的环境，不能混为一谈：

| 环境 | 定位 | 可以作为验收依据 | 不可以据此下结论 |
|---|---|---|---|
| Linux GPU 服务器 | **实验运行与依赖兼容的权威环境** | 安装、真实数据训练、CUDA、性能、正式回归和论文结果 | 不把服务器个人路径直接发布给用户 |
| Windows 本地工作区 | 编辑、只读检查和轻量静态验证 | UTF-8、AST/语法、Markdown、无需重依赖的纯单元检查 | 不能因本地缺少 TensorFlow/CUDA 就修改服务器依赖或判定实验不可运行 |

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

### 4.3 Windows 本地验证规范

当前共享工作区的本地解释器固定为：

- Python：`d:\anaconda\envs\py312\python.exe`
- 环境目录：`d:\anaconda\envs\py312`

在本地执行任何 Python 相关命令时必须使用上述绝对路径。禁止直接使用 `python`、`py`、`pip`、`pytest` 等未限定环境的命令，也不得自动创建新环境。

```powershell
& 'd:\anaconda\envs\py312\python.exe' --version
& 'd:\anaconda\envs\py312\python.exe' -c "import ast; print('syntax-only validation')"
```

本地检查失败时先确认解释器路径。依赖安装或升级会改变共享环境，必须由维护者明确授权；agent 不得为了让本地检查通过而擅自改变 Linux 实验依赖。

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
- Windows Python 3.12 只作为补充静态解析环境，不得反向迫使服务器代码使用 3.9+ / 3.10+ / 3.12 专属语法。
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
- 以 Linux 服务器为兼容性验证环境；Windows/Python 3.12 只做不依赖完整实验栈的补充检查。
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

Windows 本地最低要求是 UTF-8、Markdown 和 Python AST/语法检查；只有本地已具备相同依赖时才追加 unit test，不能替代 Linux 验收。

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
| S1 开源发布与工程基线 | `IN_PROGRESS` | 包元数据、`mmctr` 公共命名空间、统一 CLI、严格 training/本地路径配置层、主要公开文档、合成 CPU smoke、首个数值回归基线、tuner 科研红线修复和主训练运行目录隔离已完成；`OSS-001` 等待许可证 | Linux 安装、公开文档、P0 修复、smoke baseline |
| S2 数据处理与模型主干 | `TODO` | 已完成 AntM2C 只读问题定位，尚未实施 | 无切片数据链路、统一 Batch、单一 BaseSeqModel |
| S3 模型公共组件 | `TODO` | 未开始 | pooling/fusion 可按模态配置且默认 preset 回归通过 |
| S4 实验分析体系 | `TODO` | 未开始 | 无模型复制、统一 runner/result schema、五类分析可配置运行 |

> 当前结论：治理、包元数据、公开文档主体、合成 smoke 与首个行为基线已落地；模型和数据主干改造尚未开始。不得把“已写方案”计作模型或训练链路完成度。

### 13.2 可领取任务

| ID | Priority | Task | Depends on | Status | Owner | Files | Evidence / Notes |
|---|---|---|---|---|---|---|---|
| GOV-000 | P0 | 仓库盘点并建立全局改造方案 | - | `DONE` | Codex | `REFACTORING_PLAN.md` | 117 files、16,679 LOC、AST 0 failures；2026-07-31 |
| GOV-001 | P0 | 建立/确认 Git 基线，保护现有用户改动 | GOV-000 | `DONE` | Codex (`/root`) | repo metadata、`REFACTORING_PLAN.md` | 改造前 267 个文件已纳入本地可恢复快照；公开分支从远程初始提交生成干净历史；2026-07-31 |
| GOV-002 | P0 | 添加 `.gitignore`，治理 pyc/egg-info/outputs | GOV-001 | `DONE` | Codex (`/root`) | `.gitignore`、generated files、`REFACTORING_PLAN.md` | 移除 105 个 `.pyc`、9 个 `__pycache__` 和 1 个 `egg-info`；本地 `.vscode/settings.json` 保留但取消跟踪；19 个论文图表保留；`git ls-files -ci --exclude-standard` 为 0；2026-07-31 |
| PUB-001 | P0 | 公开推送前净化个人路径与历史生成物，并接入目标远程 | GOV-002 | `DONE` | Codex (`/root`) | hard-coded path files、Git refs/history、`REFACTORING_PLAN.md` | 远程 `origin/main` 非强制前进至 `222591b`；当前树个人路径/常见密钥扫描均为 0；公开历史 pyc/egg-info/IDE 对象为 0；6 个改动脚本 AST 与 4 个 YAML 静态检查通过；原本地历史保留于 `local/refactor-bootstrap`；2026-07-31 |
| ENV-001 | P0 | 审计并保留 Linux server snapshot，生成可移植环境说明 | GOV-001 | `REVIEW` | Codex (`/root`) | `bm_env.yml`、`environment.yml`、`docs/environment.md`、`AGENTS.md`、`REFACTORING_PLAN.md` | 服务器包清单保留，机器专属 prefix 在 `PUB-001` 中移除；Windows `d:\anaconda\envs\py312\python.exe` 3.12.11 完成 UTF-8/YAML/无 prefix/无镜像 URL 静态检查；发现 TF/Keras/NumPy 与 CUDA 组合风险；维护者于 2026-07-31 提供服务器交接：使用既有 `bm` 环境、下载/缓存/临时产物与项目同盘、原始 `/home/star/Disk3/hkl/Benchmark` 仅作只读数据/历史路径参考；实际解释器、框架版本和 CPU/GPU smoke 仍待服务器回填，状态保持 `REVIEW` |
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
| QA-001 | P1 | Ruff/pytest/mypy/coverage 分阶段门禁 | ENV-002, PKG-001 | `TODO` | - | pyproject/tests | Linux 门禁报告，Windows 静态报告 |
| CI-001 | P1 | Linux CPU CI；Windows 仅可选静态检查 | QA-001 | `TODO` | - | CI config | clean Linux runner 通过 |
| CORE-001 | P0 | 定义统一 Batch/ModelOutput/RunResult | PKG-001, CFG-001 | `TODO` | - | core schemas | shape/dtype/mask/type unit tests |
| DATA-001 | P0 | 统一三个数据集 loader contract 与 manifest | CORE-001 | `TODO` | - | data/datasets | 三 adapter contract tests |
| ANT-001 | P0 | 审计 AntM2C 六类文本字段语义与 feature ownership | DATA-001 | `TODO` | - | antm2c schema/docs | 字段归属、shape、split 协议经确认 |
| ANT-002 | P0 | 用 event_id + 单次时序扫描重写历史构造 | ANT-001 | `TODO` | - | antm2c preprocessing | 无未来泄漏、重复 item 和复杂度测试 |
| ANT-003 | P0 | 建立 item index 和去重 item feature store | ANT-001 | `TODO` | - | antm2c preprocessing | target/history 同表 gather 一致性 |
| ANT-004 | P1 | 重写文本/图像 batch 提取、单次模型加载和断点续跑 | ANT-003 | `TODO` | - | antm2c extractors | 吞吐报告、resume、missing manifest |
| ANT-005 | P0 | 重写 split 序列化和 loader，删除 4608 拼接/固定切片 | ANT-002, ANT-003, ANT-004 | `TODO` | - | antm2c serializer/loader | 无 `text_full[4]`；三 split 抽样审计 |
| ANT-006 | P1 | Linux 基准比较 TFRecord 与候选分层格式 | ANT-005 | `TODO` | - | benchmark/docs | 空间、吞吐、CPU/GPU wait 报告 |
| DATA-002 | P1 | 将 MicroLens/TikTok 对齐统一数据契约 | DATA-001, ANT-005 | `TODO` | - | data/datasets | contract + smoke tests |
| TRAIN-001 | P0 | 统一 training engine、evaluate、early stop、optimizer | CORE-001, RUN-001 | `TODO` | - | training/evaluation | 单步训练、save/load、resume；复用 `RUN-001` run context，并把 legacy 模型 checkpoint basename 收敛为 `best.pt`/`last.pt` |
| MODEL-BASE-001 | P0 | 建立单一 BaseSeqModel 与 pooled/token 两种历史能力 | CORE-001, TRAIN-001 | `TODO` | - | model base | pooled_history/sequence_tokens tests |
| MODEL-BASE-002 | P0 | 迁移并弃用 BaseModel 子类，先 pool 后复用原主体 | MODEL-BASE-001, DATA-001 | `TODO` | - | baselines/multimodal | 每模型迁移前后 fixture 回归 |
| REG-001 | P1 | 模型/数据集 registry（fusion registry 留到 S3） | PKG-001, CFG-001 | `TODO` | - | registries | 唯一名称与全构造测试 |
| MOD-001 | P1 | 迁移基础 CTR 模型 | MODEL-BASE-002, REG-001 | `TODO` | - | baselines | 每模型 forward/backward |
| MOD-002 | P1 | 迁移简单多模态模型 | MOD-001 | `TODO` | - | multimodal | regression + smoke |
| MOD-003 | P1 | 迁移复杂序列/辅助损失模型 | MOD-002 | `TODO` | - | multimodal | mask/aux loss + regression |
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
| ADR-006 | 2026-07-31 | Linux 服务器是依赖、训练和性能验收的权威环境 | `ACCEPTED` | Windows 本地仅用于编辑和静态验证 |
| ADR-007 | 2026-07-31 | 多模态模型主干只保留 BaseSeqModel | `ACCEPTED` | 原 BaseModel 模型通过序列 pooling 接入统一契约 |
| ADR-008 | 2026-07-31 | AntM2C item 特征独立存储并按 item index gather | `ACCEPTED` | 删除 4608 打包和 item 固定切片协议 |
| ADR-009 | 2026-07-31 | pooling/fusion 按分支与模态配置，并做 capability 校验 | `ACCEPTED` | 支持可组合实验且避免无效组合 |
| ADR-010 | 2026-07-31 | `ENV-001` 保持 `REVIEW`，Linux 实测延期为发布门禁；允许继续本地静态可验证任务 | `ACCEPTED` | 维护者明确要求服务器环境暂不验证并继续改造；不得据此宣称 Linux/CUDA 已通过 |
| ADR-011 | 2026-07-31 | 分发名使用 `mmctr-bench`，目标导入命名空间仍为 `mmctr`；迁移前暂时发现 legacy `src` 包 | `ACCEPTED` | 区分 PyPI 分发名与 Python 包名，并使 `ENV-002` 可在 `PKG-001` 前建立可构建元数据 |
| ADR-012 | 2026-07-31 | 配置相对路径统一相对包含 `pyproject.toml` 的项目根解析；training 配置先以 frozen dataclass 严格校验，模型/数据算法字段在对应任务中逐步 typed 化 | `ACCEPTED` | 消除 cwd 差异并立即保护运行关键字段，同时避免在未建立模型回归前一次性重写全部论文专属参数 |
| ADR-013 | 2026-07-31 | 机器专属数据路径只允许来自 ignored `configs/local/paths.yaml`、`MMCTR_*_DATA_DIR` 环境变量或显式 CLI override；优先级为本地文件 < 环境变量 < CLI，tracked 配置只保留相对路径和空 example | `ACCEPTED` | 同一公开配置可跨机器复用，真实服务器路径不进入 Git，缺失 override 时给出明确错误而非引用不存在文件 |
| ADR-014 | 2026-07-31 | 公共命令统一为 `mmctr` / `python -m mmctr.cli` 子命令；CLI 模块保持 dependency-light，train 命令完成参数/线程设置后才导入训练 runtime | `ACCEPTED` | `--help`、配置检查和列表命令可安全导入运行，避免 argparse/torch/TensorFlow 在模块导入阶段产生副作用 |
| ADR-015 | 2026-07-31 | 服务器接续使用既有 Conda `bm` 环境；下载、缓存和临时产物与当前项目同盘；原始 `/home/star/Disk3/hkl/Benchmark` 只读参考 | `ACCEPTED` | 遵循维护者提供的服务器资源边界，避免占用其他磁盘或误改原始版本，同时为数据位置和历史约定保留可核对证据 |

---

## 15. 变更记录

| Version | Date | Author | Change |
|---|---|---|---|
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
