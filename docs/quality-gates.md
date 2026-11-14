# Quality gates

Linux is the authoritative environment for MMCTR quality gates. The gate protects the public
`src/mmctr` namespace, the complete test suite, and the installable package boundary.

## Current scope

- Ruff lint and formatting include `src/mmctr/**/*.py` and `tests/**/*.py`.
- Pytest collects every test below `tests/`.
- Coverage measures the public `mmctr` package and requires at least 80% statement coverage.
- Mypy checks every file under `src/mmctr`; `follow_imports = "skip"` keeps optional framework
  packages outside the typed core boundary.
- Package build creates both an sdist and a wheel in an isolated build environment.

The former `src/models`, `src/scripts`, `src/data`, `src/trainers`, and `src/utils` runtime trees
are no longer shipped. Their canonical implementations and entry points live entirely below
`src/mmctr`.

## Linux commands

Activate the maintainer-provided environment, resolve its absolute interpreter, and keep caches
and temporary artifacts on the project filesystem:

```bash
conda activate bm
MMCTR_PROJECT_ROOT="$(pwd -P)"
MMCTR_PYTHON="$(command -v python)"
export PIP_CACHE_DIR="$MMCTR_PROJECT_ROOT/.cache/pip"
export RUFF_CACHE_DIR="$MMCTR_PROJECT_ROOT/.cache/ruff"
export MYPY_CACHE_DIR="$MMCTR_PROJECT_ROOT/.cache/mypy"
export COVERAGE_FILE="$MMCTR_PROJECT_ROOT/.cache/coverage/.coverage"
export PYTHONPYCACHEPREFIX="$MMCTR_PROJECT_ROOT/.tmp/pycache"
export TMPDIR="$MMCTR_PROJECT_ROOT/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$RUFF_CACHE_DIR" "$MYPY_CACHE_DIR" \
  "$(dirname "$COVERAGE_FILE")" "$PYTHONPYCACHEPREFIX" "$TMPDIR"

"$MMCTR_PYTHON" -m ruff format --check .
"$MMCTR_PYTHON" -m ruff check .
"$MMCTR_PYTHON" -m pytest tests/unit tests/smoke
"$MMCTR_PYTHON" -m mypy src/mmctr
"$MMCTR_PYTHON" -m pytest --cov=mmctr --cov-report=term-missing
"$MMCTR_PYTHON" -m build --outdir "$TMPDIR/qa-dist" .
```

Run plain `"$MMCTR_PYTHON" -m pytest` as the complete CPU regression gate. GPU, real-data, and
performance checks remain separate explicit server gates. Runtime requirements are documented in [environment.md](environment.md).

## Linux CPU CI

The workflow at [`.github/workflows/linux-ci.yml`](../.github/workflows/linux-ci.yml) runs the
same staged gates on a fixed Ubuntu 22.04 and Python 3.8 CPU runner. It installs the CPU build of
PyTorch 1.13.1 and the `training,ci,dev` extras, verifies the resolved environment with `pip check`,
then runs Ruff, mypy, unit/smoke tests, the complete coverage gate, and an isolated package build.
All caches and temporary files are rooted below the GitHub workspace.

The `ci` extra includes the optional image, plotting, tabular, Arrow, and TensorFlow packages
needed by collected tests. The workflow contract test verifies the runner, actions, Python
version, dependency groups, project-local caches, and commands.

Record execution logs and coverage/build reports locally. Public source contains reusable
procedures; validation records and generated distributions remain under ignored output paths.
See the [release checklist](release-checklist.md) before source or artifact publication.
