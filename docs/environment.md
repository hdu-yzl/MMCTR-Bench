# Environment and reproducibility

MMCTR currently has two environment artifacts with deliberately different roles:

- `bm_env.yml` is the preserved Linux experiment-server snapshot. It records the environment
  that was exported from the original machine and must remain available for forensic
  reproduction.
- `environment.yml` is a machine-independent bootstrap file. It creates the Python toolchain
  without embedding a server path, a private mirror, or Linux platform packages.
- `pyproject.toml` is the sole package/build and dependency-group authority. Modern PEP 517
  frontends are required; no `setup.py` or separate requirements file is maintained.

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
| Data/search stack | `pyarrow 17.0.0`, `lmdb 1.3.0`, `optuna 3.6.1` | Install through the matching optional dependency groups |
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

For an existing experiment environment, activate `bm` and use the absolute interpreter returned
by `command -v python`. Keep data, caches, temporary artifacts, and validation logs on the project
filesystem. Store machine-specific paths in ignored `configs/local/paths.yaml` or environment
variables.

## Reference runtime

The reference Linux stack uses Python 3.8.20, PyTorch 1.13.1+cu117 (CUDA 11.7 build),
TensorFlow 2.12.1, NumPy 1.23.5, scikit-learn 1.3.2, and JAX/JAXlib 0.4.13. GPU experiments
use Tesla V100-SXM2-32GB hardware with an NVIDIA driver compatible with the selected CUDA build.
Choose a PyTorch build appropriate for the target server before installing optional dependencies.

TensorRT is optional and is not required by MMCTR-Bench. Retain the actual interpreter,
framework versions, driver, device, dependency check, and command outcomes in local run metadata.
See [quality gates](quality-gates.md) for validation commands.
