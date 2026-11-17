# MMCTR-Bench

[![Linux CI](https://github.com/hdu-yzl/MMCTR-Bench/actions/workflows/linux-ci.yml/badge.svg)](https://github.com/hdu-yzl/MMCTR-Bench/actions/workflows/linux-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](pyproject.toml)

MMCTR-Bench is a reproducible benchmark for click-through-rate prediction with ID, sequential,
text, image, and audio features. It provides one canonical Python package, strict dataset and
training contracts, isolated experiment outputs, and one-click scripts for single-model and
multi-GPU training.

This project was refactored with the assistance of AI.

Current TikTok canonical-v1 contains **357,246 / 18,306 / 36,780** train/validation/test
CTR samples (**412,332 total**, 9,308 users, 6,710 target items across splits, positive:negative
ratio 1:5). See [measured per-split counts](docs/data.md#tiktok-canonical-v1) and
[benchmark protocol](docs/benchmark-protocol.md).

## What is included

- Four dataset adapters: **AntM2C**, **MicroLens**, **TikTok**, and **EB-NeRD**.
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

The public runtime namespace is `mmctr`; data processing, models, training, and analysis share
its canonical interfaces.

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
data/raw/ebnerd/
```

The exact required files are documented in:

- [data/raw/antm2c/README.md](data/raw/antm2c/README.md): event shards, image archive, BERT and
  ChineseCLIP checkpoints;
- [data/raw/microlens/README.md](data/raw/microlens/README.md): interaction and item-feature
  Parquet files;
- [data/raw/tiktok/README.md](data/raw/tiktok/README.md): official JSON splits and upstream text,
  image, and audio arrays;
- [data/raw/ebnerd/README.md](data/raw/ebnerd/README.md): EB-NeRD behaviors, histories, article metadata,
  and upstream text/image embeddings.

Canonical processors write ignored payloads to:

```text
data/processed/antm2c/canonical-v1/
data/processed/microlens/canonical-v1/
data/processed/tiktok/canonical-v1/
data/processed/ebnerd/canonical-v4/
```

Tracked manifests describe the data version, schema, split statistics, fingerprints, and source
hashes without publishing the payload. For data stored elsewhere, copy
`configs/local/paths.example.yaml` to ignored `configs/local/paths.yaml` and use the
`--use-local-data` launcher option. See [data/README.md](data/README.md) and
[docs/data.md](docs/data.md) for acquisition and format details.

### EB-NeRD

EB-NeRD small is available through the canonical loader and dataset catalog.
See [processing, training and analysis](docs/data/ebnerd.md).

EB-NeRD canonical-v4 uses 56 additional categorical/bucketed ID fields in the existing
ID embedding branches, with the same latent width as user/item IDs and no residual predictor.
See [the EB-NeRD ID schema](docs/data/ebnerd.md#canonical-v4-categorical-and-bucketed-id-fields-current).

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

### Main-table mean and standard deviation on eight GPUs

Activate the existing `bm` environment and record its absolute interpreter:

```bash
conda activate bm
command -v python
export MMCTR_PYTHON="$(command -v python)"
scripts/run_main_table.sh --dry-run
scripts/run_main_table.sh --gpus 0,1,2,3,4,5,6,7
# After an interruption/failure, reuse completed seeds and retry the others:
scripts/run_main_table.sh --gpus 0,1,2,3,4,5,6,7 --resume
```

Defaults: AntM2C, MicroLens, TikTok and EB-NeRD (canonical-v4, 10% test impressions, 58 ID fields); the **21 paper models** (including QARM and PSRQ,
excluding the `dnn_mm`/`dnn_mm_seq` reference variants); seeds **2025–2029**. This creates
**420 training runs**, with one independent process per GPU and dynamic assignment when a
GPU becomes free. The current model/dataset catalogs and training defaults are reused,
including dataset-specific model settings. Each run uses validation-based checkpoint selection
and evaluates test once; the launcher changes only the seed and runtime placement.

`outputs/main_table/main_table.csv` contains one row per dataset/model, test AUC and LogLoss
(`test_loss`) mean, **sample standard deviation (`ddof=1`)**, formatted `mean ± std`, success/failure
counts and each seed's raw scores. `main_table.json` additionally preserves configuration,
source/data-manifest/quantizer hashes, every attempt, errors and artifact paths. Both are updated
atomically after each completion. Incomplete rows retain raw scores and counts but leave mean/std
blank until all five seeds succeed. Failed jobs do not cancel unrelated jobs, and any failure
returns a nonzero exit status. SIGINT/SIGTERM stops active training children.

Resume requires matching code, configuration, input fingerprints, seed set and interpreter;
completed results are checked before reuse. Keep code, data and pretrained artifacts fixed during
the experiment. A new configuration requires a new `--output-dir`. A lock prevents two launchers
from writing the same report. Each attempt retains its own logs and resolved config.
Use `--no-checkpoints` to retain validation-selected best weights only in CPU memory and
write no `.pt` files. Test evaluation still uses the best validation epoch. Interrupted seeds
restart from the beginning in this mode; completed metrics remain reusable.
Use `--jobs-per-gpu 2 --num-threads 4` for two independent workers per GPU when measured
GPU memory and CPU capacity permit. Concurrency timings are not isolated efficiency benchmarks.

Use `--datasets tiktok`, `--models dnn,din`, `--models all` (23 registry models),
`--seeds 42,43,44,45,46`, `--num-threads 4`, `--output-dir outputs/main_table_v2`, or
`--use-local-data` as needed. GPU identifiers are physical indices or full GPU UUIDs;
by default the launcher honors `CUDA_VISIBLE_DEVICES`, otherwise it uses indices 0–7.
The launcher sets repository-local cache/temporary directories and uses `MMCTR_PYTHON`.
`--dry-run` prints the full plan and missing prerequisites without training or writing reports.

QARM/PSRQ reuse fixed pretrained artifacts across seeds. Missing artifacts are reported before
any training starts by default. With `--wait-for-inputs`, ready jobs proceed while
missing artifacts are generated separately; consumers wait for atomic publication and SHA-256
registration (two-hour timeout). Generate them for each affected dataset using the existing entry point:

```bash
scripts/pretrain_quantizers.sh --dataset microlens --gpu 0
```

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

<!-- MAIN_TABLE_SNAPSHOT_START -->
## Reference results

The following tables provide **reference results after dataset restructuring**. They cover four datasets, 21 models and five seeds (2025–2029), totaling 420 runs. Values are arithmetic mean ± sample standard deviation (`ddof=1`), recomputed from all per-seed job results. AUC ↑; LogLoss ↓.

AntM2C / MicroLens / TikTok use canonical-v1; EB-NeRD uses canonical-v4 (history length 5, ID width 16, 10% of local-test impressions). See the [benchmark protocol](docs/benchmark-protocol.md) for dataset and evaluation details.

### AntM2C

| Model | AUC ↑ (mean ± std) | LogLoss ↓ (mean ± std) |
|---|---:|---:|
| autoint | 0.876872 ± 0.001141 | 0.430796 ± 0.003057 |
| dcn | 0.877555 ± 0.000716 | 0.427266 ± 0.000812 |
| deepfm | 0.877390 ± 0.000762 | 0.426978 ± 0.001678 |
| diff_msin | 0.864590 ± 0.012512 | 0.475081 ± 0.040836 |
| din | 0.876841 ± 0.000934 | 0.429837 ± 0.002637 |
| dmf | 0.877028 ± 0.000929 | 0.424963 ± 0.003704 |
| dnn | 0.877122 ± 0.000610 | 0.427930 ± 0.001512 |
| em3 | 0.877249 ± 0.004876 | 0.441474 ± 0.012958 |
| gmmf | 0.880366 ± 0.001668 | 0.425214 ± 0.004552 |
| lmf | 0.889545 ± 0.001488 | 0.403807 ± 0.003053 |
| m3srec | 0.885317 ± 0.002248 | 0.416456 ± 0.004843 |
| make | 0.893544 ± 0.003157 | 0.399262 ± 0.005901 |
| marn | 0.846796 ± 0.030815 | 0.474980 ± 0.044725 |
| mb | 0.891174 ± 0.000940 | 0.401789 ± 0.001052 |
| mmmlp | 0.892262 ± 0.000122 | 0.406994 ± 0.003104 |
| mtfn | 0.887786 ± 0.000465 | 0.407792 ± 0.000890 |
| naml | 0.875942 ± 0.001390 | 0.431518 ± 0.003567 |
| pamd | 0.889503 ± 0.000635 | 0.404988 ± 0.001851 |
| psrq | 0.883453 ± 0.000650 | 0.424437 ± 0.005117 |
| qarm | 0.890959 ± 0.001526 | 0.407300 ± 0.004566 |
| simcen | 0.882321 ± 0.001147 | 0.421007 ± 0.003018 |

### MicroLens

| Model | AUC ↑ (mean ± std) | LogLoss ↓ (mean ± std) |
|---|---:|---:|
| autoint | 0.957427 ± 0.000175 | 0.202241 ± 0.000123 |
| dcn | 0.957632 ± 0.000145 | 0.201845 ± 0.000443 |
| deepfm | 0.958371 ± 0.000142 | 0.200462 ± 0.000382 |
| diff_msin | 0.957071 ± 0.001849 | 0.229109 ± 0.024270 |
| din | 0.957340 ± 0.000273 | 0.203440 ± 0.000341 |
| dmf | 0.852822 ± 0.026015 | 0.390653 ± 0.029701 |
| dnn | 0.957561 ± 0.000156 | 0.201857 ± 0.000479 |
| em3 | 0.968902 ± 0.007069 | 0.177373 ± 0.016146 |
| gmmf | 0.958775 ± 0.000708 | 0.201636 ± 0.001412 |
| lmf | 0.971991 ± 0.000587 | 0.169853 ± 0.001581 |
| m3srec | 0.964057 ± 0.003375 | 0.238818 ± 0.074210 |
| make | 0.973433 ± 0.000100 | 0.166115 ± 0.000449 |
| marn | 0.952199 ± 0.013224 | 0.212854 ± 0.023797 |
| mb | 0.970448 ± 0.000479 | 0.172986 ± 0.000641 |
| mmmlp | 0.970966 ± 0.001302 | 0.174439 ± 0.004818 |
| mtfn | 0.971252 ± 0.000293 | 0.171994 ± 0.000863 |
| naml | 0.972434 ± 0.000564 | 0.169860 ± 0.001692 |
| pamd | 0.973717 ± 0.000473 | 0.165708 ± 0.001379 |
| psrq | 0.908864 ± 0.005230 | 0.335569 ± 0.091888 |
| qarm | 0.968736 ± 0.000315 | 0.178645 ± 0.000708 |
| simcen | 0.957187 ± 0.001847 | 0.207295 ± 0.007351 |

### TikTok

| Model | AUC ↑ (mean ± std) | LogLoss ↓ (mean ± std) |
|---|---:|---:|
| autoint | 0.769759 ± 0.005939 | 0.469214 ± 0.015520 |
| dcn | 0.762052 ± 0.006234 | 0.494021 ± 0.038708 |
| deepfm | 0.770466 ± 0.003499 | 0.441353 ± 0.072469 |
| diff_msin | 0.753061 ± 0.012195 | 0.421233 ± 0.035093 |
| din | 0.768122 ± 0.004027 | 0.466679 ± 0.004939 |
| dmf | 0.559722 ± 0.014569 | 0.511758 ± 0.044927 |
| dnn | 0.762481 ± 0.006416 | 0.472025 ± 0.008558 |
| em3 | 0.769581 ± 0.004616 | 0.461108 ± 0.025540 |
| gmmf | 0.743747 ± 0.007753 | 0.432577 ± 0.037012 |
| lmf | 0.746793 ± 0.003495 | 0.457212 ± 0.005753 |
| m3srec | 0.762954 ± 0.002835 | 0.455193 ± 0.016295 |
| make | 0.760972 ± 0.002415 | 0.473985 ± 0.004822 |
| marn | 0.729187 ± 0.032554 | 0.442987 ± 0.033977 |
| mb | 0.766587 ± 0.002944 | 0.375061 ± 0.003088 |
| mmmlp | 0.791964 ± 0.007825 | 0.447259 ± 0.063048 |
| mtfn | 0.748527 ± 0.004596 | 0.450759 ± 0.001832 |
| naml | 0.776620 ± 0.000750 | 0.401439 ± 0.001136 |
| pamd | 0.766851 ± 0.010744 | 0.419870 ± 0.042921 |
| psrq | 0.575582 ± 0.005923 | 0.542868 ± 0.006599 |
| qarm | 0.775227 ± 0.001804 | 0.446220 ± 0.007287 |
| simcen | 0.652603 ± 0.098887 | 0.442521 ± 0.061320 |

### EB-NeRD

| Model | AUC ↑ (mean ± std) | LogLoss ↓ (mean ± std) |
|---|---:|---:|
| autoint | 0.769991 ± 0.002017 | 0.256667 ± 0.001036 |
| dcn | 0.770415 ± 0.000956 | 0.257339 ± 0.000456 |
| deepfm | 0.768070 ± 0.000971 | 0.259772 ± 0.000560 |
| diff_msin | 0.764184 ± 0.001955 | 0.264270 ± 0.002573 |
| din | 0.769162 ± 0.001312 | 0.260265 ± 0.001310 |
| dmf | 0.763105 ± 0.002671 | 0.265416 ± 0.001857 |
| dnn | 0.768546 ± 0.000994 | 0.261218 ± 0.001325 |
| em3 | 0.767380 ± 0.001923 | 0.261757 ± 0.003041 |
| gmmf | 0.766106 ± 0.003429 | 0.261111 ± 0.002891 |
| lmf | 0.763650 ± 0.001388 | 0.265818 ± 0.002480 |
| m3srec | 0.769934 ± 0.000500 | 0.261598 ± 0.000401 |
| make | 0.767937 ± 0.001755 | 0.268071 ± 0.002520 |
| marn | 0.760019 ± 0.002831 | 0.278603 ± 0.040688 |
| mb | 0.770734 ± 0.001286 | 0.256948 ± 0.002155 |
| mmmlp | 0.769615 ± 0.001338 | 0.266131 ± 0.004908 |
| mtfn | 0.712496 ± 0.001293 | 0.266070 ± 0.000822 |
| naml | 0.758331 ± 0.003007 | 0.264334 ± 0.004025 |
| pamd | 0.767653 ± 0.000939 | 0.259673 ± 0.002901 |
| psrq | 0.766218 ± 0.002213 | 0.261485 ± 0.001365 |
| qarm | 0.770242 ± 0.000760 | 0.258044 ± 0.001552 |
| simcen | 0.759884 ± 0.001956 | 0.260031 ± 0.001611 |

Source: `outputs/main_table_ebnerd/main_table.json`. Regenerate the tables with `"$MMCTR_PYTHON" scripts/update_readme_main_table.py`.
<!-- MAIN_TABLE_SNAPSHOT_END -->

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

See the [analysis experiment guide](docs/analysis/cross-dataset-diagnostics.md) for
single-setting diagnostics and how to tune their controls. Matrix planning and plotting
commands are documented in [docs/cli.md](docs/cli.md) and [docs/experiments.md](docs/experiments.md).

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
