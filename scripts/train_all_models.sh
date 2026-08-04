#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_all_models.sh [options]

Train canonical MMCTR models with at most one worker per GPU.

Options:
  --dataset NAME          Dataset registry name (default: antm2c)
  --gpus IDS              Comma-separated CUDA indices (default: 0)
  --models NAMES          Comma-separated model names (default: all)
  --include-quantized     Include QARM and PSRQ; pretrain their artifacts first
  --python PATH           Python interpreter (default: MMCTR_PYTHON or active environment)
  --output-root PATH      Run output root (default: outputs)
  --num-threads N         CPU threads per training worker (default: 8)
  --use-local-data        Read ignored configs/local/paths.yaml
  --dry-run               Validate and print commands without training
  -h, --help              Show this help
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
single_script="$project_root/scripts/train_model.sh"
dataset="antm2c"
gpu_csv="0"
model_csv="all"
include_quantized=0
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
    --gpus)
      (($# >= 2)) || die "--gpus requires a value"
      gpu_csv="$2"
      shift 2
      ;;
    --models)
      (($# >= 2)) || die "--models requires a value"
      model_csv="$2"
      shift 2
      ;;
    --include-quantized)
      include_quantized=1
      shift
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

[[ "$num_threads" =~ ^[1-9][0-9]*$ ]] || die "--num-threads must be a positive integer"
IFS=',' read -r -a gpus <<< "$gpu_csv"
((${#gpus[@]} > 0)) || die "--gpus must contain at least one CUDA index"
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid CUDA index in --gpus: $gpu"
  [[ -z "${seen_gpus[$gpu]+present}" ]] || die "duplicate CUDA index in --gpus: $gpu"
  seen_gpus["$gpu"]=1
done

if [[ -z "$python_bin" ]]; then
  python_bin="$(command -v python || true)"
elif [[ "$python_bin" != */* ]]; then
  python_bin="$(command -v "$python_bin" || true)"
fi
[[ -n "$python_bin" && -x "$python_bin" ]] || die \
  "Python was not found; activate bm or pass --python /absolute/path/to/python"

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$model_csv" == "all" ]]; then
  model_output="$("$python_bin" -m mmctr.cli list-models)"
  mapfile -t models <<< "$model_output"
  if ((include_quantized == 0)); then
    filtered_models=()
    for model in "${models[@]}"; do
      [[ "$model" == "qarm" || "$model" == "psrq" ]] || filtered_models+=("$model")
    done
    models=("${filtered_models[@]}")
  fi
else
  IFS=',' read -r -a models <<< "$model_csv"
  ((${#models[@]} > 0)) || die "--models must contain at least one model"
  if ((include_quantized == 0)); then
    for model in "${models[@]}"; do
      if [[ "$model" == "qarm" || "$model" == "psrq" ]]; then
        die "$model requires --include-quantized and pretrained RQ/PSRQ artifacts"
      fi
    done
  fi
fi

build_arguments() {
  local model="$1"
  local gpu="$2"
  model_arguments=(
    --dataset "$dataset"
    --model "$model"
    --gpu "$gpu"
    --python "$python_bin"
    --output-root "$output_root"
    --num-threads "$num_threads"
  )
  if [[ "$use_local_data" == "1" ]]; then
    model_arguments+=(--use-local-data)
  fi
  if ((dry_run)); then
    model_arguments+=(--dry-run)
  fi
}

if ((dry_run)); then
  for index in "${!models[@]}"; do
    gpu="${gpus[index % ${#gpus[@]}]}"
    build_arguments "${models[index]}" "$gpu"
    "$single_script" "${model_arguments[@]}"
  done
  exit 0
fi

printf 'Training %d model(s) on dataset %s across GPU(s): %s\n' \
  "${#models[@]}" "$dataset" "$gpu_csv"

worker() {
  local worker_index="$1"
  local gpu="$2"
  local index
  for ((index = worker_index; index < ${#models[@]}; index += ${#gpus[@]})); do
    build_arguments "${models[index]}" "$gpu"
    "$single_script" "${model_arguments[@]}"
  done
}

pids=()
for worker_index in "${!gpus[@]}"; do
  worker "$worker_index" "${gpus[worker_index]}" &
  pids+=("$!")
done

terminate_workers() {
  ((${#pids[@]} == 0)) || kill "${pids[@]}" 2>/dev/null || true
}
trap terminate_workers INT TERM

# Reap every child, but preserve any worker failure as the launcher exit status.
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
