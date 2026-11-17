# Contributing to MMCTR-Bench

Keep changes scoped and document their effect on data splits, features, model behavior, and
evaluation. Fit on training data, select on validation, and evaluate test after freezing the
configuration and weights. Keep per-seed results with their settings and data fingerprint.

Use the [project environment](README.md#installation) on Linux. From the repository root, run
the checks relevant to your change:

```bash
MMCTR_PYTHON="$(command -v python)"
export TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR"
"$MMCTR_PYTHON" -m ruff format --check .
"$MMCTR_PYTHON" -m ruff check .
"$MMCTR_PYTHON" -m mypy src/mmctr
"$MMCTR_PYTHON" -m pytest --cov=mmctr --cov-report=term-missing
"$MMCTR_PYTHON" -m build --outdir "$TMPDIR/dist"
```

Keep data, checkpoints, results, credentials, and machine-specific settings out of Git.
For a distribution change, check the sdist and wheel contents as well as the staged diff.
Contributions to the repository are covered by [Apache-2.0](LICENSE); third-party assets
retain their own terms.
