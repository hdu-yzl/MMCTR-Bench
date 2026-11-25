#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_all_models.sh [options]

Train all canonical MMCTR models, including QARM and PSRQ, with one worker per GPU.
Required quantizers are checked and pretrained automatically before CTR training.

Options:
  --dataset NAME          Dataset registry name (default: antm2c)
  --gpus IDS              Comma-separated CUDA indices (default: 0)
  --models NAMES          Comma-separated model names (default: all)
  --python PATH           Python interpreter (default: MMCTR_PYTHON or active environment)
  --output-root PATH      Run output root (default: outputs)
  --num-threads N         CPU threads per training worker (default: 8)
  --use-local-data        Read ignored configs/local/paths.yaml
  --dry-run               Validate and print commands without training
  -h, --help              Show this help
EOF
}

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
common_options+=" --output-root"
gpu_csv="0"
model_csv="all"
while (($#)); do
  case "$1" in
    --gpus|--models)
      (($# >= 2)) || die "$1 requires a value"
      if [[ "$1" == --gpus ]]; then gpu_csv="$2"; else model_csv="$2"; fi
      shift 2 ;;
    *) common_option "$@"; shift "$consumed" ;;
  esac
done
IFS=',' read -r -a gpus <<< "$gpu_csv"
[[ -n "$gpu_csv" && "$gpu_csv" != *, ]] || die "--gpus must contain CUDA indices"
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid CUDA index in --gpus: $gpu"
  gpu=$((10#$gpu))
  [[ -z "${seen_gpus[$gpu]+present}" ]] || die "duplicate CUDA index in --gpus: $gpu"
  seen_gpus["$gpu"]=1
done
setup_runtime
if [[ "$model_csv" == all ]]; then
  model_output="$("$python_bin" -m mmctr.cli list-models)"
  mapfile -t models <<< "$model_output"
else
  [[ -n "$model_csv" && "$model_csv" != *, ]] || die "--models must contain model names"
  IFS=',' read -r -a models <<< "$model_csv"
fi
# Validate the entire selection once, before starting any training worker.
prepare_command=("$python_bin" -m mmctr.quantization.ensure --dataset-name "$dataset"
  --models "$(IFS=','; echo "${models[*]}")" --cuda "${gpus[0]}" --num-threads "$num_threads")
[[ "$use_local_data" != 1 ]] || prepare_command+=(--use-local-data)
((dry_run == 0)) || prepare_command+=(--dry-run)
"${prepare_command[@]}"
if ((dry_run)); then
  for index in "${!models[@]}"; do
    build_train_command "${models[index]}" "${gpus[index % ${#gpus[@]}]}"
    print_command "${command[@]}"
  done
  exit 0
fi
printf 'Training %d model(s) on dataset %s across GPU(s): %s\n' "${#models[@]}" "$dataset" "$gpu_csv"
worker() {
  local index child=""
  trap '[[ -z "$child" ]] || kill "$child" 2>/dev/null; exit 130' INT TERM
  for ((index = $1; index < ${#models[@]}; index += ${#gpus[@]})); do
    build_train_command "${models[index]}" "$2"
    "${command[@]}" &
    child=$!
    wait "$child" || return $?
    child=""
  done
}
pids=()
for index in "${!gpus[@]}"; do
  worker "$index" "${gpus[index]}" &
  pids+=("$!")
done
terminate_workers() {
  kill "${pids[@]}" 2>/dev/null || true
  wait || true
  exit 130
}
trap terminate_workers INT TERM
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
