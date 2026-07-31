# Environment and reproducibility

MMCTR currently has two environment artifacts with deliberately different roles:

- `bm_env.yml` is the preserved Linux experiment-server snapshot. It records the environment
  that was exported from the original machine and must remain available for forensic
  reproduction.
- `environment.yml` is a machine-independent bootstrap file. It creates the Python toolchain
  without embedding a server path, a private mirror, or Linux platform packages.
- `pyproject.toml` is the package and dependency-group authority. `setup.py` remains only as a
  compatibility shim for legacy build frontends; no separate requirements file is maintained.

The bootstrap file is not yet a claim that the complete training stack is reproducible. Linux is
the authority for installation, CUDA, full imports, training, and performance checks; Windows is
used only for editing and lightweight static validation.

## Snapshot audit

The checked-in `bm_env.yml` contains a full Conda export from a Linux server. The audit found:

| Item | Snapshot value | Portability treatment |
|---|---|---|
| Environment prefix | The original server export contained a machine-specific absolute `prefix` | Removed from the published snapshot and excluded from `environment.yml` |
| Environment name | `bm` | Renamed to `mmctr` for the portable bootstrap |
| Python | `3.8.5` | Preserved as the initial experiment compatibility baseline |
| PyTorch stack | `torch 1.13.1+cu117`, `torchvision 0.14.1+cu117`, `torchaudio 0.13.1+cu117` | Must be validated with the server driver and CUDA runtime before publication |
| TensorFlow | `tensorflow 2.4.0` | Must be validated separately from the PyTorch/CUDA stack |
| Numeric stack | `numpy 1.23.5`, `pandas 2.0.3`, `scipy 1.10.1`, `scikit-learn 1.3.2` | Exact installed state is retained only in the snapshot pending compatibility tests |
| Data/search stack | `pyarrow 17.0.0`, `lmdb 1.3.0`, `optuna 3.6.1` | Candidate runtime dependencies for `ENV-002` |
| Model stack | `transformers 4.46.3`, `sentence-transformers 3.2.1`, `cn-clip 1.5.1`, `faiss-gpu 1.7.2` | Treat as optional heavy groups and validate wheel/channel availability |
| Channels | Five Tsinghua mirror URLs, including duplicates and the retired `free` channel | Replaced by the named `conda-forge` channel in the bootstrap |
| Platform packages | Linux libc, compiler runtime, ncurses, readline, and related build pins | Excluded; the target Linux solver selects platform packages |

The snapshot also contains potentially incompatible combinations, notably TensorFlow 2.4.0 with
Keras 2.15.0 and a newer NumPy installation. This may represent a working but incrementally
mutated environment. Do not convert the complete package list into published dependency pins
until imports and a minimal training smoke test succeed on the Linux server.

## Package metadata and dependency groups

The initial package compatibility range is deliberately conservative: Python `>=3.8,<3.9`.
Python 3.8.5 is the only experiment interpreter represented by the server snapshot. Windows
Python 3.12 remains valid for AST, UTF-8, TOML, and other static checks, but is not currently a
supported installation target. The range may be widened only after Linux and CI validation.

Core NumPy and YAML dependencies are installed by default. Optional dependencies are grouped by
responsibility:

| Extra | Scope |
|---|---|
| `training` | PyTorch training and scikit-learn metrics |
| `data` | TFRecord/TensorFlow adapters, tabular processing, images, Arrow, and LMDB |
| `multimodal` | Transformer, sentence-transformer, and image-model feature extraction |
| `analysis` | Plotting, scientific interpolation, tables, and process metrics |
| `tuning` | Optuna, process metrics, and round-trip YAML editing |
| `dev` | Build, pytest, coverage, Ruff, mypy, and YAML type stubs |

These constraints describe compatible version bands; they do not choose a CUDA build. Install
the server-approved PyTorch/CUDA combination using its official channel before running the full
editable installation. The exact snapshot remains in `bm_env.yml` for comparison.

## Linux bootstrap

Replace both placeholders with real absolute paths in an execution record:

```bash
<CONDA_EXE> env create --prefix <SERVER_ENV> --file environment.yml
<SERVER_ENV>/bin/python --version
<SERVER_ENV>/bin/python -m pip --version
```

The expected full installation shape is:

```bash
<SERVER_ENV>/bin/python -m pip install --editable '.[training,data,multimodal,analysis,tuning]'
<SERVER_ENV>/bin/python -m pip check
<SERVER_ENV>/bin/python -c "import torch, tensorflow; print(torch.__version__, tensorflow.__version__)"
<SERVER_ENV>/bin/python -m pytest tests/unit tests/smoke
<SERVER_ENV>/bin/python -m build
```

These commands are a protocol, not completed evidence. Record the actual interpreter path, OS,
GPU, driver, CUDA, PyTorch, TensorFlow, and command exit codes when they are run.

## Maintainer server handoff

The next refactoring stage will continue in VS Code on the Linux experiment server. The
maintainer supplied the following operational facts on 2026-07-31; they are handoff inputs, not
completed validation evidence:

- Activate the existing environment with `conda activate bm`. Do not create a replacement
  environment automatically. After activation, record the absolute path reported by
  `command -v python` and use that interpreter for Python-related commands.
- Keep every new package download, package cache, model/data download, build dependency, and
  temporary artifact on the same filesystem as the current project clone. Repository-local
  `.cache/`, `.tmp/`, and `downloads/` directories are ignored by Git for this purpose.
- `/home/star/Disk3/hkl/Benchmark` is the original pre-refactoring project. Use it as a read-only
  reference when locating data or reconstructing historical path/configuration conventions.
  Never modify that directory or copy its private data and machine-specific configuration into
  tracked files.

Before downloading anything on the server, run the equivalent of the following from the new
clone's root (do not assume that the clone itself has a fixed absolute path):

```bash
conda activate bm
MMCTR_PROJECT_ROOT="$(pwd -P)"
export PIP_CACHE_DIR="$MMCTR_PROJECT_ROOT/.cache/pip"
export CONDA_PKGS_DIRS="$MMCTR_PROJECT_ROOT/.cache/conda-pkgs"
export HF_HOME="$MMCTR_PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$MMCTR_PROJECT_ROOT/.cache/torch"
export TMPDIR="$MMCTR_PROJECT_ROOT/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
MMCTR_PYTHON="$(command -v python)"
"$MMCTR_PYTHON" --version
```

Once `command -v python` has resolved the `bm` interpreter, use `"$MMCTR_PYTHON"` in
recorded validation commands. Actual data paths belong in ignored `configs/local/paths.yaml`,
environment variables, or explicit CLI overrides—not in published configuration.

## Windows validation boundary

Local Python commands must use the configured interpreter explicitly:

```powershell
& 'd:\anaconda\envs\py312\python.exe' --version
& 'd:\anaconda\envs\py312\python.exe' -c "import yaml; yaml.safe_load(open('environment.yml', encoding='utf-8'))"
& 'd:\anaconda\envs\py312\python.exe' -m pip wheel --no-deps --no-build-isolation --wheel-dir <TEMP_DIR> .
```

A successful Windows YAML parse confirms only that the file is readable and structurally valid.
It does not replace Linux dependency resolution or GPU validation.

## Server validation record

Complete this table on the Linux experiment server before marking `ENV-001` as `DONE`:

| Check | Result |
|---|---|
| Conda executable absolute path | Pending |
| Python executable absolute path | Pending |
| OS/kernel | Pending |
| GPU and driver | Pending |
| CUDA runtime | Pending |
| Bootstrap environment creation | Pending |
| PyTorch/TensorFlow import versions | Pending |
| `pip check` | Pending |
| CPU smoke test | Pending |
| GPU smoke test | Pending |
