# MMCTR-Bench

MMCTR-Bench is a research benchmark for click-through-rate recommendation with multimodal and
sequential features. The repository includes conventional CTR baselines, multimodal fusion
models, quantization-oriented models, dataset adapters, tuning scripts, and analysis workflows.

> **Project status:** active engineering refactor. The historical research implementation is
> available, but the unified package, CLI, synthetic smoke suite, and reproducible Linux release
> gate are still being built. See [REFACTORING_PLAN.md](REFACTORING_PLAN.md) for the authoritative
> task status and validation evidence.

## Scope

Current dataset adapters:

- AntM2C
- MicroLens
- TikTok

Current registered model names include DNN, DCN, DeepFM, DIN, AutoInt, LMF, Diff-MSIN, MARN,
MTFN, DMF, SimCEN, NAML, MAKE, EM3, GMMF, QARM, MCCA, MB, PAMD, MMMLP, M3SRec, RQ, and PSRQ.

The public package namespace is `mmctr`; the current bridge exposes models, data factories, and
utilities there while retaining legacy top-level packages for existing research scripts. Import
migration details and current boundaries are documented in
[docs/migration.md](docs/migration.md). Typed data/model contracts and shared analysis protocols
are still introduced incrementally.

## Requirements

- Linux is the authority for dependency resolution, CUDA, training, performance, and formal
  regression results.
- Python 3.8.5 is the initial experiment compatibility baseline. Package metadata currently
  declares Python `>=3.8,<3.9` until Linux and CI validation justify a wider range.
- Windows Python 3.12 is used only for UTF-8, AST, TOML, Markdown, and lightweight build checks.
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

There is not yet a supported public full-training command: the legacy trainer parses arguments
and changes process-wide thread settings during import. Its output is now isolated by run ID, but
that does not remove the remaining CLI/environment limitations. The current directory contract is
documented in [docs/run-layout.md](docs/run-layout.md). A dependency-light test baseline now covers
pooling behavior and an ID-only DNN CPU forward/loss/backward step with synthetic tensors:

```bash
<SERVER_ENV>/bin/python -m pytest tests/unit tests/smoke
```

When pytest is unavailable in the configured Windows static-check environment, the same
`unittest.TestCase` suite can be run without installing packages:

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath 'src').Path
& 'd:\anaconda\envs\py312\python.exe' -m unittest discover -s tests -p 'test_*.py' -v
```

These checks do not use real data or prove that full Linux/CUDA training is reproducible.

## Data layout

Repository configs use these portable defaults:

```text
data/processed/
├─ antm2c/
├─ microlens/
└─ tiktok/
```

Private data stays outside version control. Dataset sources, licenses, download instructions,
field schemas, and manifests are scheduled under `OSS-002`. Do not infer permission to
redistribute a dataset merely because an adapter exists in this repository.

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
config/        Legacy dataset, model, training, and tuning configuration
src/data/      Dataset adapters and preprocessing scripts
src/models/    CTR, multimodal, quantization, and shared layer implementations
src/trainers/  Legacy training entry points
src/scripts/   Legacy tuning and batch scripts
src/analysis/  Cold-start, robustness, alignment, fusion, efficiency, and plotting analysis
docs/          Environment and future architecture/usage documentation
```

## Development

Read [CONTRIBUTING.md](CONTRIBUTING.md) and claim a task in
[REFACTORING_PLAN.md](REFACTORING_PLAN.md) before modifying shared interfaces. Local Python
commands in this workspace must use the configured absolute interpreter path; server validation
records must likewise use the real absolute server interpreter.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Model- and dataset-specific paper
citations are not yet complete and are tracked by `OSS-002`.

## License

A project license has not yet been selected by the maintainer. The `LICENSE` file will be added
only after that explicit decision; `OSS-001` remains incomplete until then.
