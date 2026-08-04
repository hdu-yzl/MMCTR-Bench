# MMCTR-Bench

MMCTR-Bench is a research benchmark for click-through-rate recommendation with multimodal and
sequential features. The repository includes conventional CTR baselines, multimodal fusion
models, quantization-oriented models, dataset adapters, tuning scripts, and analysis workflows.

> **Project status:** release candidate. The public package, CLI, canonical model/data/training
> internals, CI, Linux quality gates, and Apache-2.0 software licensing are complete. See
> [REFACTORING_PLAN.md](REFACTORING_PLAN.md) for authoritative status and validation evidence.

## Scope

Current dataset adapters:

- AntM2C
- MicroLens
- TikTok

The registry exposes 23 canonical CTR models, including DNN, DCN, DeepFM, DIN, AutoInt, LMF,
Diff-MSIN, MARN, MTFN, DMF, SimCEN, NAML, MAKE, EM3, GMMF, QARM, MCCA, MB, PAMD, MMMLP, and
M3SRec. RQ and PSRQ are separate quantization pretraining components.

The public package namespace is `mmctr`; model classes, factories, and training components are
published only through that namespace. Import migration details are documented in
[docs/migration.md](docs/migration.md). Typed data/model contracts and shared analysis protocols
are the only formal runtime paths.

Strict training configuration validation and layer precedence are documented in
[docs/configuration.md](docs/configuration.md).
RQ/PSRQ artifact layout and the QARM/MCCA composition boundary are documented in
[docs/quantization.md](docs/quantization.md).
Machine-specific dataset and output paths are configured as described in
[docs/local-paths.md](docs/local-paths.md); real server paths remain untracked.

## Requirements

- Linux is the authority for dependency resolution, CUDA, training, performance, and formal
  regression results.
- Python 3.8.5 is the initial experiment compatibility baseline. Package metadata currently
  declares Python `>=3.8,<3.9` until Linux and CI validation justify a wider range.
- All subsequent editing, static checks, tests, dependency validation, and CUDA work run on the
  Linux server in the maintainer-provided `bm` environment.
- Real datasets, checkpoints, local path files, and experiment outputs are not included.

Environment roles and known framework compatibility risks are documented in
[docs/environment.md](docs/environment.md).

## Quick start

The commands below describe the current Linux bootstrap. Replace placeholders with actual
absolute paths and record them in validation evidence.

```bash
git clone https://github.com/hdu-yzl/MMCTR-Bench.git
cd MMCTR-Bench

<CONDA_EXE> env create --prefix <SERVER_ENV> --file environment.yml
<SERVER_ENV>/bin/python -m pip install --editable \
  '.[training,data,multimodal,analysis,tuning]'
<SERVER_ENV>/bin/python -m pip check
<SERVER_ENV>/bin/python -c "import mmctr; print(mmctr.__version__)"
```

Install the server-approved PyTorch/CUDA build from its official channel before the editable
installation. The dependency groups do not select a GPU build automatically.

A public CLI surface is now available for help, registry listing, strict config validation, and an
isolated canonical training entry point:

```bash
<SERVER_ENV>/bin/python -m mmctr.cli --help
<SERVER_ENV>/bin/python -m mmctr.cli list-models
<SERVER_ENV>/bin/python -m mmctr.cli list-quantizers
<SERVER_ENV>/bin/python -m mmctr.cli validate-config \
  --config configs/training/default.yaml
```

See [docs/cli.md](docs/cli.md) and [docs/run-layout.md](docs/run-layout.md). Linux framework, real
canonical data, CPU, and CUDA gates pass. A dependency-light test baseline covers pooling behavior
and an ID-only DNN CPU forward/loss/backward step with synthetic tensors:

```bash
<SERVER_ENV>/bin/python -m pytest tests/unit tests/smoke
```

These CPU checks do not use real data or by themselves prove that full CUDA training is
reproducible; GPU and real-data gates are run explicitly on the same Linux server.

## Data layout

Repository configs use these portable defaults:

```text
data/raw/
├─ antm2c/       # user downloads from the provider
├─ microlens/    # user downloads from the provider
└─ tiktok/       # user downloads from an authorized upstream source

data/processed/
├─ antm2c/
├─ microlens/
└─ tiktok/
```

Private data stays outside version control. Exact user-supplied filenames are documented in the
tracked README inside each `data/raw/<dataset>/` directory. Dataset sources, acquisition
boundaries, current license findings, local placement, and provenance requirements are documented
in [data/README.md](data/README.md); schemas and manifests are documented in
[docs/data.md](docs/data.md). Do not infer permission to redistribute a dataset merely because an
adapter exists in this repository.

## Research protocol

The benchmark follows these non-negotiable rules:

- training data is used for fitting;
- validation data is used for early stopping, model selection, and hyperparameter search;
- test data is evaluated only after configuration freeze;
- test metrics must never be written back as best-parameter selections;
- every seed keeps an independent result, with aggregate reporting as mean and standard
  deviation;
- runs must not share checkpoint, log, or result paths.

The full protocol and refactor red lines are defined in
[REFACTORING_PLAN.md](REFACTORING_PLAN.md).

## Repository layout

```text
configs/          Dataset/model catalogs, training defaults, experiments, and local-path example
data/raw/         User-downloaded inputs; only placement README files are tracked
data/processed/   Generated canonical datasets; payload arrays are ignored
src/mmctr/        Canonical core, data, models, training, analysis, and utilities
reports/figures/  Versioned final research figures
outputs/          Ignored runs, checkpoints, logs, quantization artifacts, and analyses
docs/             Architecture and usage documentation
```

## Development

Read [CONTRIBUTING.md](CONTRIBUTING.md) and claim a task in
[REFACTORING_PLAN.md](REFACTORING_PLAN.md) before modifying shared interfaces. Local Python
commands in this workspace must use the configured absolute interpreter path; server validation
records must likewise use the real absolute server interpreter.

The staged Linux lint, formatting, typing, test, coverage, and package-build commands are
documented in [docs/quality-gates.md](docs/quality-gates.md). The final source/privacy,
distribution-rehearsal, and publication gates are in
[docs/release-checklist.md](docs/release-checklist.md).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). The dataset papers and all 23 model
registry entries are mapped in [docs/references.md](docs/references.md), including explicit labels
for benchmark-only variants and adapted methods.

## License

Repository-owned source code, configuration, and documentation are licensed under the
[Apache License 2.0](LICENSE) (`Apache-2.0`). Third-party datasets and checkpoints are not covered
by this software license and are not redistributed by the repository; review their providers'
terms before obtaining, using, or sharing them. See [data/README.md](data/README.md) for the
dataset-specific boundaries.
