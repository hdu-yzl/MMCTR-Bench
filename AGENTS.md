# Repository instructions for Codex

本文件适用于整个仓库。开始修改前，先完整阅读 `REFACTORING_PLAN.md`，并按其中的任务状态、决策记录和变更记录要求登记进度。

## 运行环境

先确认当前操作系统和主机，再选择对应环境；不要把 Windows 本地验证结果当作 Linux 服务器验收结果。

### Linux 实验服务器

- 使用维护者已经准备好的 Conda 环境：`conda activate bm`。
- 不要自动创建新的 Conda/venv 环境，也不要擅自升级或重解整个环境。
- 激活后先用 `command -v python` 取得并记录解释器绝对路径；后续 Python、pip、pytest 等命令必须通过该绝对解释器执行。
- 安装新依赖会改变实验环境，先说明原因、版本和影响，并取得维护者授权。

### Windows 本地工作区

- Python 解释器固定为 `d:\anaconda\envs\py312\python.exe`。
- 所有 Python 相关命令都必须使用上述绝对路径，禁止直接使用未限定环境的 `python`、`py`、`pip` 或 `pytest`。
- 不要自动创建新的 venv/Conda 环境。检查失败时，先确认解释器路径是否正确。

## 下载、缓存和临时文件

- 任何新增包、模型、数据、构建依赖、下载缓存和临时产物都必须放在**当前项目所在的同一磁盘/文件系统**，不要写入系统盘或其他 home 磁盘。
- 在服务器下载前，以当前克隆的项目根目录为基准配置仓库本地目录，例如 `.cache/pip`、`.cache/conda-pkgs`、`.cache/huggingface`、`.cache/torch`、`.tmp` 和 `downloads`。
- 可相应设置 `PIP_CACHE_DIR`、`CONDA_PKGS_DIRS`、`HF_HOME`、`TORCH_HOME` 和 `TMPDIR`。不要改写 `HOME`、`CODEX_HOME` 等全局目录变量。
- 上述目录是本地运行产物，不得提交到 Git；真实数据和 checkpoint 同样不得进入源码历史。

## 原始版本只读参考

- `/home/star/Disk3/hkl/Benchmark` 是项目开始改造前的最原始版本。
- 遇到数据位置、历史配置、旧脚本约定、checkpoint 或运行方式不明确时，优先到该目录核对。
- 该目录只作为只读证据源：不得在其中编辑、删除、移动、格式化或执行迁移操作，也不得把它当作当前改造工作区。
- 不要把原始目录中的真实数据、凭据或机器专属配置提交到 Git。需要接入数据路径时，使用 ignored `configs/local/paths.yaml`、环境变量或显式 CLI override。

## 服务器接续与进度记录

- 在新的服务器克隆中先检查 `git status`、当前分支和远程同步状态，再继续修改。
- Linux 服务器是依赖、真实数据、CUDA、训练和性能验证的权威环境。记录实际 Python 路径、Conda 环境、OS、GPU/驱动、CUDA、框架版本和命令结果。
- 每次改造都要同步更新 `REFACTORING_PLAN.md` 的版本、任务状态/证据、必要的 ADR 和变更记录；没有实际运行的检查不得写成已通过。
