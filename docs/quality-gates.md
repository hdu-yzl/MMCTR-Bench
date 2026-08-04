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
performance checks remain separate explicit server gates. The known TensorFlow dependency and
CUDA-session findings are recorded in [environment.md](environment.md).

## Linux CPU CI

The workflow at [`.github/workflows/linux-ci.yml`](../.github/workflows/linux-ci.yml) runs the
same staged gates on a fixed Ubuntu 22.04 and Python 3.8 CPU runner. It installs the CPU build of
PyTorch 1.13.1 and the `training,dev` extras, verifies the resolved environment with `pip check`,
then runs Ruff, mypy, unit/smoke tests, the complete coverage gate, and an isolated package build.
All caches and temporary files are rooted below the GitHub workspace.

TensorFlow remains in the separate `data` extra, so the optional TFRecord smoke is skipped on the
clean CPU CI job and is covered by the authoritative Linux `bm` environment instead. A repository
contract test parses the workflow as YAML and locks its runner, actions, Python version, cache
boundary, and commands. The first clean remote run for commit `66db7d7` completed successfully as
[GitHub Actions run 30642128515](https://github.com/hdu-yzl/MMCTR-Bench/actions/runs/30642128515),
so `CI-001` is `DONE`.

## Linux baseline: 2026-07-31

The following results were produced with `<BM_PYTHON>` resolved from `conda activate bm` on Ubuntu
20.04.6 LTS:

| Gate | Result |
|---|---|
| Ruff format | 34 scoped files already formatted |
| Ruff lint | Passed for the configured `src/mmctr + tests` scope |
| Unit + smoke | 25 passed in 7.48 seconds, including CI workflow and TFRecord round trips; strict RuntimeWarning rerun also passed |
| Complete pytest + coverage | 35 passed in 17.84 seconds; 83.80% coverage, above the 80% gate |
| Mypy | Mypy 1.10.1 found no issues in 14 `src/mmctr` source files |
| Isolated package build | sdist and wheel built successfully; wheel contains 137 files |

The final wheel is 268,965 bytes with SHA-256
`c279ede0278cdd3871aefdaf08b3040fac4ddd90b4b849e913ce7e18388d3413`. The final sdist is
175,986 bytes with SHA-256
`4c1aa88017e7f8cab04812fd9fecbd5676984099d054bf56945fa63e60ce9ed3`. Both artifacts were
written below ignored project-local `.tmp/` storage.

The maintainer selected this Linux server as the sole environment for all subsequent changes and
validation. The recorded Linux evidence therefore closes `QA-001` as `DONE`; no Windows rerun is
required or inferred.

## Current release rehearsal: 2026-08-04

Using the absolute `bm` interpreter recorded in the server evidence (Python 3.8.20), the cumulative
refactor passes `pip check`, Ruff format for 143 files, Ruff lint, mypy for 86 source files, 200
unit/smoke tests, and 220 complete tests with 84.90% coverage. Bash syntax, repository-external
launcher dry-runs, and the no-isolation source/wheel build also pass. Artifact hashes and remaining
non-engineering blockers are recorded in [`release-checklist.md`](release-checklist.md).
