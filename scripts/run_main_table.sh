#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${MMCTR_PYTHON:-$(command -v python)}"
[[ "$python_bin" = /* && -x "$python_bin" ]] || {
  echo 'Activate bm or set MMCTR_PYTHON to an absolute interpreter path.' >&2
  exit 2
}
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PIP_CACHE_DIR="$project_root/.cache/pip"
export CONDA_PKGS_DIRS="$project_root/.cache/conda-pkgs"
export HF_HOME="$project_root/.cache/huggingface"
export TORCH_HOME="$project_root/.cache/torch"
export TMPDIR="$project_root/.tmp"
mkdir -p "$TMPDIR"
cd "$project_root"
exec "$python_bin" -m mmctr.experiments.main_table "$@"
