# MMCTR-Bench

[![Linux CI](https://github.com/hdu-yzl/MMCTR-Bench/actions/workflows/linux-ci.yml/badge.svg)](https://github.com/hdu-yzl/MMCTR-Bench/actions/workflows/linux-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](pyproject.toml)

MMCTR-Bench is a reproducible benchmark for click-through-rate prediction with ID, sequential,
text, image, and audio features. It provides one canonical Python package, strict dataset and
training contracts, isolated experiment outputs, and one-click scripts for single-model and
multi-GPU training.

This project was refactored with the assistance of AI.

## What is included

- Three dataset adapters: **AntM2C**, **MicroLens**, and **TikTok**.
- The paper evaluation covers five ID-only baselines (`dnn`, `deepfm`, `din`, `autoint`, and
  `dcn`) and 16 multimodal models organized into four paradigms:
  - **TMIE**: `mmmlp`, `diff_msin`, and `dmf`;
  - **CSAQ**: `make`, `m3srec`, `em3`, `psrq`, and `qarm`;
  - **GFFI**: `naml`, `mb`, `lmf`, `simcen`, and `mtfn`;
  - **RDRR**: `pamd`, `gmmf`, and `marn`.
- The software registry additionally exposes `dnn_mm` and `dnn_mm_seq` as reference variants; they
  are not additional paper-benchmarked models. The canonical `psrq` entry implements the paper's
  PSRQ pretraining and downstream MCCA consumer; `mcca` remains only a backward-compatible alias.
  RQ analogously supplies pretrained artifacts to `qarm`.
- Canonical training, validation-only selection, experiment planning, robustness/alignment/
  cold-start analysis, plotting, and versioned result protocols.
- Apache-2.0 licensed repository code and documentation. Third-party datasets and checkpoints are
  not redistributed or relicensed.

The public runtime namespace is only `mmctr`; the former duplicate model, trainer, processor, and
analysis trees have been removed.

## Installation

Linux and Python 3.8 are the validated environment. Select the server-compatible PyTorch/CUDA
build first; the optional dependencies intentionally do not choose a GPU build for you.

```bash
git clone https://github.com/hdu-yzl/MMCTR-Bench.git
cd MMCTR-Bench

conda env create -n bm -f environment.yml
conda activate bm
python -m pip install --editable '.[training,data,multimodal,analysis,tuning]'
python -m pip check

# Keep the interpreter explicit for every launcher.
export MMCTR_PYTHON="$(command -v python)"
"$MMCTR_PYTHON" -m mmctr.cli --version
```

Maintainers of the validated experiment server should activate the existing `bm` environment; do
not create or replace it. See [docs/environment.md](docs/environment.md) for the tested dependency
matrix.

## Prepare the data

MMCTR-Bench does not download or publish third-party data. Obtain each dataset from its provider,
review its terms, record local provenance and SHA-256 values, and place the inputs under:

```text
data/raw/antm2c/
data/raw/microlens/
data/raw/tiktok/
```

The exact required files are documented in:

- [data/raw/antm2c/README.md](data/raw/antm2c/README.md): event shards, image archive, BERT and
  ChineseCLIP checkpoints;
- [data/raw/microlens/README.md](data/raw/microlens/README.md): interaction and item-feature
  Parquet files;
- [data/raw/tiktok/README.md](data/raw/tiktok/README.md): official JSON splits and upstream text,
  image, and audio arrays.

Canonical processors write ignored payloads to:

```text
data/processed/antm2c/canonical-v1/
data/processed/microlens/canonical-v1/
data/processed/tiktok/canonical-v1/
```

Tracked manifests describe the data version, schema, split statistics, fingerprints, and source
hashes without publishing the payload. For data stored elsewhere, copy
`configs/local/paths.example.yaml` to ignored `configs/local/paths.yaml` and use the
`--use-local-data` launcher option. See [data/README.md](data/README.md) and
[docs/data.md](docs/data.md) for acquisition and format details.

## One-click training

All launchers resolve the repository root themselves, validate dataset/model names, and delegate
to the canonical `mmctr` entry points. Use `--dry-run` first to inspect a command without loading
data or starting training.

### Train one model

```bash
scripts/train_model.sh \
  --dataset antm2c \
  --model dnn_mm_seq \
  --gpu 0 \
  --dry-run

scripts/train_model.sh --dataset antm2c --model dnn_mm_seq --gpu 0
```

Useful overrides:

```bash
scripts/train_model.sh \
  --dataset microlens \
  --model deepfm \
  --gpu 1 \
  --num-threads 8 \
  --output-root outputs
```

Run `scripts/train_model.sh --help` for every option. Training hyperparameters come from
`configs/training/default.yaml`; model and dataset definitions come from the two catalog files
under `configs/models/` and `configs/datasets/`.

### Train all standard models on multiple GPUs

```bash
# Preview 21 commands and their GPU assignment.
scripts/train_all_models.sh --dataset antm2c --gpus 0,1,2,3,4,5,6,7 --dry-run

# One sequential worker per GPU; a worker takes its next model after the previous one finishes.
scripts/train_all_models.sh --dataset antm2c --gpus 0,1,2,3,4,5,6,7
```

The default batch intentionally excludes `qarm` and `psrq` because they require pretrained RQ and
PSRQ artifacts, respectively. The canonical `psrq` model uses PSRQ-pretrained semantic codes in its
internal downstream MCCA consumer. Train a selected subset with:

```bash
scripts/train_all_models.sh \
  --dataset tiktok \
  --gpus 0,1 \
  --models dnn,dcn,deepfm,din,autoint
```

Each GPU has at most one worker, duplicate GPU indices are rejected, and any failed worker makes
the launcher return a non-zero exit code.

### Pretrain quantizers and include QARM/PSRQ

```bash
scripts/pretrain_quantizers.sh --dataset tiktok --gpu 0 --dry-run
scripts/pretrain_quantizers.sh --dataset tiktok --gpu 0

scripts/train_all_models.sh \
  --dataset tiktok \
  --gpus 0,1,2,3 \
  --include-quantized
```

RQ artifacts are written to `outputs/quantization/artifacts/rq/<dataset>/`; PSRQ is written to
`outputs/quantization/artifacts/psrq/<dataset>/`. Their lifecycle and compatibility checks are
described in [docs/quantization.md](docs/quantization.md).

## Runs and results

Every invocation creates an isolated directory:

```text
outputs/training/<dataset>/<model>/<run_id>/
├── resolved_config.yaml
├── run_metadata.json
├── metrics.jsonl
├── run.log
├── summary.json
└── checkpoints/
    ├── best.pt
    └── last.pt
```

The run ID contains a timestamp, configuration hash, and random entropy. Runs never share logs or
checkpoints. Validation controls early stopping and model selection; test evaluation occurs only
after the selected checkpoint is frozen. See [docs/run-layout.md](docs/run-layout.md) and
[docs/training.md](docs/training.md).

## CLI

The scripts are convenience wrappers around the public CLI:

```bash
"$MMCTR_PYTHON" -m mmctr.cli --help
"$MMCTR_PYTHON" -m mmctr.cli list-models
"$MMCTR_PYTHON" -m mmctr.cli list-datasets
"$MMCTR_PYTHON" -m mmctr.cli list-quantizers
"$MMCTR_PYTHON" -m mmctr.cli validate-config --config configs/training/default.yaml
"$MMCTR_PYTHON" -m mmctr.cli train --dataset-name antm2c --model-name dnn --cuda 0
```

Analysis matrix and plotting commands are documented in [docs/cli.md](docs/cli.md) and
[docs/experiments.md](docs/experiments.md).

## Repository layout

```text
configs/          Dataset/model catalogs, training defaults, experiment examples, local template
data/raw/         Provider inputs supplied by the user; payloads are ignored
data/processed/   Versioned canonical stores; payloads are ignored, small manifests are tracked
scripts/          One-click single-model, multi-GPU, and quantization launchers
src/mmctr/        The only runtime package: data, models, training, analysis, CLI, and utilities
outputs/          Ignored runs, checkpoints, logs, quantization artifacts, figures, and results
tests/            Unit, integration, regression, and smoke gates
docs/             Architecture, protocols, configuration, data, and release documentation
```

## Development and validation

```bash
"$MMCTR_PYTHON" -m ruff format --check src/mmctr tests
"$MMCTR_PYTHON" -m ruff check src/mmctr tests
"$MMCTR_PYTHON" -m mypy src/mmctr
"$MMCTR_PYTHON" -m pytest -q tests/unit tests/smoke
"$MMCTR_PYTHON" -m pytest --cov=mmctr --cov-report=term-missing --cov-fail-under=80
"$MMCTR_PYTHON" -m build --no-isolation
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing public contracts. The exact Linux gates
and release audit are in [docs/quality-gates.md](docs/quality-gates.md) and
[docs/release-checklist.md](docs/release-checklist.md).

## Citation and references

Use [CITATION.cff](CITATION.cff) when citing MMCTR-Bench. Dataset provenance and the mapping for
all 23 software registry entries are documented in [docs/references.md](docs/references.md), including
21 paper evaluation entries (five ID-only and 16 multimodal) plus the two DNN multimodal reference
variants.

## License

Repository-owned source code, configuration, and documentation are licensed under the
[Apache License 2.0](LICENSE). Third-party datasets and checkpoints are not covered by this
software license; the same exclusion applies to third-party media and derived arrays. None of
these payloads are redistributed by this repository.
