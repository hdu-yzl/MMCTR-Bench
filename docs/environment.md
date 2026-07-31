# Environment and reproducibility

MMCTR currently has two environment artifacts with deliberately different roles:

- `bm_env.yml` is the preserved Linux experiment-server snapshot. It records the environment
  that was exported from the original machine and must remain available for forensic
  reproduction.
- `environment.yml` is a machine-independent bootstrap file. It creates the Python toolchain
  without embedding a server path, a private mirror, or Linux platform packages.
- `pyproject.toml` is the package and dependency-group authority. `setup.py` remains only as a
  compatibility shim for legacy build frontends; no separate requirements file is maintained.

The bootstrap file is not by itself a claim that the complete training stack is reproducible.
The maintainer-selected Linux server is the sole environment for subsequent editing, dependency
resolution, CUDA, tests, training, and performance checks.

## Snapshot audit

The checked-in `bm_env.yml` contains a full Conda export from a Linux server. The audit found:

| Item | Snapshot value | Portability treatment |
|---|---|---|
| Environment prefix | The original server export contained a machine-specific absolute `prefix` | Removed from the published snapshot and excluded from `environment.yml` |
| Environment name | `bm` | Renamed to `mmctr` for the portable bootstrap |
| Python | `3.8.5` | Preserved as the initial experiment compatibility baseline |
| PyTorch stack | `torch 1.13.1+cu117`, `torchvision 0.14.1+cu117`, `torchaudio 0.13.1+cu117` | Retained and validated with the server driver and CUDA runtime |
| TensorFlow | `tensorflow 2.4.0` | Historical snapshot only; the working `bm` environment is aligned to TensorFlow 2.12.1 |
| Numeric stack | `numpy 1.23.5`, `pandas 2.0.3`, `scipy 1.10.1`, `scikit-learn 1.3.2` | Exact installed state is retained only in the snapshot pending compatibility tests |
| Data/search stack | `pyarrow 17.0.0`, `lmdb 1.3.0`, `optuna 3.6.1` | Candidate runtime dependencies for `ENV-002` |
| Model stack | `transformers 4.46.3`, `sentence-transformers 3.2.1`, `cn-clip 1.5.1`, `faiss-gpu 1.7.2` | Treat as optional heavy groups and validate wheel/channel availability |
| Channels | Five Tsinghua mirror URLs, including duplicates and the retired `free` channel | Replaced by the named `conda-forge` channel in the bootstrap |
| Platform packages | Linux libc, compiler runtime, ncurses, readline, and related build pins | Excluded; the target Linux solver selects platform packages |

The snapshot contains incompatible combinations, notably TensorFlow 2.4.0 with Keras 2.15.0 and
newer NumPy/typing-extensions installations. It remains unchanged as forensic evidence; it is not
the current install prescription. The validated working versions and resolver changes are
recorded below and constrained in `pyproject.toml`.

## Package metadata and dependency groups

The initial package compatibility range is deliberately conservative: Python `>=3.8,<3.9`.
Python 3.8.5 is the snapshot baseline and Python 3.8.20 is the validated `bm` interpreter. The
range may be widened only after Linux CI validation.

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

The data extra pins the Python 3.8-compatible TFRecord stack: TensorFlow 2.12.1, h5py 3.8–3.10,
typing-extensions 4.4–4.5, and paired JAX/JAXlib 0.4.13 on Linux x86-64. The JAX pair exists only
because it is a TensorFlow dependency and uses its CPU backend; MMCTR GPU training remains on
PyTorch. These constraints do not choose a PyTorch CUDA build. The exact historical snapshot
remains in `bm_env.yml` for comparison.

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

These commands remain the clean-bootstrap protocol. The existing `bm` execution evidence is
recorded below with its actual interpreter, OS, GPU, driver, CUDA, framework versions, and results.

## Maintainer server handoff

Refactoring continues in VS Code on the Linux experiment server. The maintainer supplied and the
current session confirmed the following operating rules on 2026-07-31:

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

## Linux-only validation boundary

All subsequent editing and validation runs on this Linux server with the absolute `bm`
interpreter. Windows results in the historical task log are retained as provenance but are not
rerun and do not satisfy any current gate.

## Server validation record

The Linux validation and dependency alignment were completed on 2026-07-31. GPU checks were run
outside the filesystem/network sandbox so that the process could access the real NVIDIA devices.
`ENV-001` is `DONE`; no bootstrap environment was created and the PyTorch/NumPy/CUDA main stack
was not changed.

| Check | Result |
|---|---|
| Conda executable absolute path | `/root/anaconda3/bin/conda` |
| Active environment | Existing maintainer-provided `bm`; no bootstrap environment was created |
| Python executable absolute path | `/home/star/Disk3/hkl/envs/bm/bin/python` (Python 3.8.20) |
| OS/kernel | Ubuntu 20.04.6 LTS; Linux 5.15.0-139-generic; host `ESC8000-G4` |
| GPU and driver | `nvidia-smi` reports driver 535.183.01, CUDA 12.2 compatibility, and 8× Tesla V100-SXM2-32GB |
| Framework builds | PyTorch 1.13.1+cu117; TensorFlow 2.12.1 built with CUDA support; NumPy remains 1.23.5 |
| PyTorch GPU smoke | Enumerated all 8 GPUs; 256×256 matmul, backward, and synchronize passed with finite gradients |
| TensorFlow GPU smoke | Enumerated and created all 8 GPU devices; 256×256 matmul on `/GPU:0` passed |
| TensorFlow data smoke | TFRecord write/read/parse round trip passed; AntM2C, MicroLens, and TikTok loader modules import with TensorFlow 2.12.1 |
| Dependency consistency | `pip check`: `No broken requirements found`; JAX/JAXlib 0.4.13 import and CPU tensor smoke passed |
| CPU regression | Unit+smoke: 25 passed with `RuntimeWarning` treated as errors; complete pytest: 35 passed; coverage 83.80% |
| Static/build gates | Ruff: 34 scoped files formatted and lint-clean; mypy 1.10.1: 14 files clean; isolated sdist+wheel build passed |

The resolver-aligned packages are:

- TensorFlow 2.4.0 → 2.12.1; Keras 2.15.0 → 2.12.0; estimator 2.4.0 → 2.12.0;
  tensorboard 2.11.2 → 2.12.3.
- h5py 2.10.0 → 3.10.0, removing a reproducible NumPy C-ABI warning.
- typing-extensions 4.13.2 → 4.5.0; mypy 1.14.1 → 1.10.1; SQLAlchemy 2.0.31 →
  2.0.20; exceptiongroup 1.3.1 → 1.1.3.
- JAX/JAXlib 0.4.13 were paired explicitly because the archived Python 3.8 JAX metadata did not
  install jaxlib automatically. TensorFlow also added/updated its normal transitive packages such
  as flatbuffers, grpcio, libclang, ml-dtypes, tensorflow-io-gcs-filesystem, and absl-py.

TensorRT remains absent and TensorFlow emits an informational TF-TRT warning; MMCTR does not use
TensorRT, so this is not a failed gate.
