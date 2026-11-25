#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/pretrain_quantizers.sh [options]

Fit the RQ and PSRQ artifacts required by QARM and the PSRQ benchmark consumer.

Options:
  --dataset NAME       Dataset registry name (default: antm2c)
  --gpu ID             CUDA device used by PSRQ (default: 0)
  --python PATH        Python interpreter (default: MMCTR_PYTHON or active environment)
  --num-threads N      CPU threads per pretrainer (default: 8)
  --use-local-data     Read ignored configs/local/paths.yaml
  --dry-run            Validate and print commands without fitting artifacts
  -h, --help           Show this help
EOF
}

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
common_options+=" --gpu"
while (($#)); do common_option "$@"; shift "$consumed"; done
setup_runtime
validate_name dataset "$dataset"
for quantizer in rq psrq; do
  command=("$python_bin" -m "mmctr.quantization.${quantizer}_entrypoint" --dataset-name "$dataset")
  [[ "$quantizer" != psrq ]] || command+=(--cuda "$gpu")
  [[ "$use_local_data" != 1 ]] || command+=(--use-local-data)
  if ((dry_run)); then
    print_command "${command[@]}"
  else
    printf 'Fitting %s artifacts for %s\n' "$quantizer" "$dataset"
    "${command[@]}"
  fi
done
