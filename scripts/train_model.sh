#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_model.sh [options]

Train one canonical MMCTR model.

Options:
  --dataset NAME       Dataset registry name (default: antm2c)
  --model NAME         Model registry name (default: dnn)
  --gpu ID             CUDA device index (default: 0)
  --python PATH        Python interpreter (default: MMCTR_PYTHON or active environment)
  --output-root PATH   Run output root (default: outputs)
  --num-threads N      CPU thread count passed to mmctr (default: 8)
  --use-local-data     Read ignored configs/local/paths.yaml
  --dry-run            Validate and print the command without training
  -h, --help           Show this help
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

contains_line() {
  local expected="$1"
  shift
  local value
  for value in "$@"; do
    [[ "$value" == "$expected" ]] && return 0
  done
  return 1
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="antm2c"
model="dnn"
gpu="0"
python_bin="${MMCTR_PYTHON:-}"
output_root="${MMCTR_OUTPUT_ROOT:-outputs}"
num_threads="${MMCTR_NUM_THREADS:-8}"
use_local_data="${MMCTR_USE_LOCAL_DATA:-0}"
dry_run=0

while (($#)); do
  case "$1" in
    --dataset)
      (($# >= 2)) || die "--dataset requires a value"
      dataset="$2"
      shift 2
      ;;
    --model)
      (($# >= 2)) || die "--model requires a value"
      model="$2"
      shift 2
      ;;
    --gpu)
      (($# >= 2)) || die "--gpu requires a value"
      gpu="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a value"
      python_bin="$2"
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a value"
      output_root="$2"
      shift 2
      ;;
    --num-threads)
      (($# >= 2)) || die "--num-threads requires a value"
      num_threads="$2"
      shift 2
      ;;
    --use-local-data)
      use_local_data=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ "$gpu" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
[[ "$num_threads" =~ ^[1-9][0-9]*$ ]] || die "--num-threads must be a positive integer"

if [[ -z "$python_bin" ]]; then
  python_bin="$(command -v python || true)"
elif [[ "$python_bin" != */* ]]; then
  python_bin="$(command -v "$python_bin" || true)"
fi
[[ -n "$python_bin" && -x "$python_bin" ]] || die \
  "Python was not found; activate bm or pass --python /absolute/path/to/python"

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
dataset_output="$("$python_bin" -m mmctr.cli list-datasets)"
model_output="$("$python_bin" -m mmctr.cli list-models)"
mapfile -t datasets <<< "$dataset_output"
mapfile -t models <<< "$model_output"
contains_line "$dataset" "${datasets[@]}" || die "unknown dataset: $dataset"
contains_line "$model" "${models[@]}" || die "unknown model: $model"

command=(
  "$python_bin" -m mmctr.cli train
  --dataset-name "$dataset"
  --model-name "$model"
  --cuda "$gpu"
  --output-root "$output_root"
  --num-threads "$num_threads"
)
if [[ "$use_local_data" == "1" ]]; then
  command+=(--use-local-data)
fi

cd "$project_root"
if ((dry_run)); then
  printf 'Dry run:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

printf 'Starting %s/%s on cuda:%s\n' "$dataset" "$model" "$gpu"
# Replace the launcher so signals and the training process exit status reach the caller unchanged.
exec "${command[@]}"
