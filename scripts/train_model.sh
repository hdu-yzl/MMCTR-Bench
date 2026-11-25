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

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
common_options+=" --gpu --output-root"
model="dnn"
while (($#)); do
  case "$1" in
    --model) (($# >= 2)) || die "--model requires a value"; model="$2"; shift 2 ;;
    *) common_option "$@"; shift "$consumed" ;;
  esac
done
setup_runtime
validate_name dataset "$dataset"
validate_name model "$model"
build_train_command "$model" "$gpu"
if ((dry_run)); then print_command "${command[@]}"; exit 0; fi
printf 'Starting %s/%s on cuda:%s\n' "$dataset" "$model" "$gpu"
exec "${command[@]}"
