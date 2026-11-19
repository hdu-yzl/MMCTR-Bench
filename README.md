# MMCTR-Bench

[![Linux CI](https://github.com/hdu-yzl/MMCTR-Bench/actions/workflows/linux-ci.yml/badge.svg)](https://github.com/hdu-yzl/MMCTR-Bench/actions/workflows/linux-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](pyproject.toml)

MMCTR-Bench compares click-through-rate models that use IDs, interaction histories, text,
images, and audio. It provides a common training and evaluation workflow for AntM2C, MicroLens,
TikTok, and EB-NeRD. The benchmark covers five ID-only baselines and 16 multimodal models across
TMIE, CSAQ, GFFI, and RDRR, plus two reference variants in the software.

This project was refactored with the assistance of AI.

- [Datasets](docs/data.md): sources, splits, and preparation.
- [Methods](docs/models.md): what each model does and its reference.
- [Experiments](docs/benchmark-protocol.md): tuning and evaluation.

## Installation

Use Linux, Python 3.8, and a PyTorch build compatible with your device. Activate an existing
project environment if one is available. For a fresh setup:

```bash
conda env create -n bm -f environment.yml
conda activate bm
export MMCTR_PYTHON="$(command -v python)"
"$MMCTR_PYTHON" -m pip install --editable '.[training,data]'
"$MMCTR_PYTHON" -m mmctr.cli --help
```

Install `.[multimodal]` if you need to extract AntM2C embeddings from local encoder checkpoints.

## Prepare the data

Follow the [dataset guide](docs/data.md) and [source file layouts](data/README.md).
The default input directories are `data/raw/antm2c`, `data/raw/microlens`,
`data/raw/tiktok`, and `data/raw/ebnerd`. For data stored elsewhere, copy
[the local paths template](configs/local/paths.example.yaml) to ignored
`configs/local/paths.yaml`, set absolute paths, and run with `--use-local-data`.

## One-click training

Run from the repository root. Add `--dry-run` to preview model commands and prerequisites.

```bash
scripts/train_model.sh --dataset tiktok --model dmf --gpu 0
scripts/train_all_models.sh --dataset tiktok --gpus 0,1
scripts/pretrain_quantizers.sh --dataset tiktok --gpu 0
scripts/train_all_models.sh --dataset tiktok --gpus 0,1 --include-quantized
```

QARM and PSRQ require their dataset's pretrained quantizer artifacts. The standard batch skips
them unless you pass `--include-quantized`. Dataset and model settings come from the catalogs
under `configs/`; shared run settings come from `configs/training/default.yaml`.

```bash
bash scripts/run_main_table.sh --dry-run
bash scripts/run_main_table.sh --gpus 0,1,2,3,4,5,6,7
```

The main-table launcher runs 21 models on four datasets with seeds 2025–2029. Use `--resume`
with the same settings to reuse completed runs. Reports go under `outputs/main_table/`; ordinary
training creates isolated runs under `outputs/training/`. See the
[experiment guide](docs/benchmark-protocol.md) for manual tuning and test evaluation.

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

## Diagnostic experiments

Pending organization; diagnostic experiments are not available in this release.

## Citation and contributions

Use [CITATION.cff](CITATION.cff) to cite MMCTR-Bench. See
[CONTRIBUTING.md](CONTRIBUTING.md) for development and validation.

## License

Repository code and documentation use the [Apache-2.0 license](LICENSE).
Third-party datasets and checkpoints are not covered by this software license or redistributed here.
