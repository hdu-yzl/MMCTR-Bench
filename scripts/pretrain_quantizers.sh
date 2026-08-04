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
  --use-local-data     Read ignored configs/local/paths.yaml
  --dry-run            Validate and print commands without fitting artifacts
  -h, --help           Show this help
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="antm2c"
gpu="0"
python_bin="${MMCTR_PYTHON:-}"
use_local_data="${MMCTR_USE_LOCAL_DATA:-0}"
dry_run=0

while (($#)); do
  case "$1" in
    --dataset)
      (($# >= 2)) || die "--dataset requires a value"
      dataset="$2"
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
if [[ -z "$python_bin" ]]; then
  python_bin="$(command -v python || true)"
elif [[ "$python_bin" != */* ]]; then
  python_bin="$(command -v "$python_bin" || true)"
fi
[[ -n "$python_bin" && -x "$python_bin" ]] || die \
  "Python was not found; activate bm or pass --python /absolute/path/to/python"

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
dataset_output="$("$python_bin" -m mmctr.cli list-datasets)"
mapfile -t datasets <<< "$dataset_output"
known_dataset=0
for candidate in "${datasets[@]}"; do
  [[ "$candidate" == "$dataset" ]] && known_dataset=1
done
((known_dataset == 1)) || die "unknown dataset: $dataset"

rq_command=(
  "$python_bin" -m mmctr.quantization.rq_entrypoint
  --dataset-name "$dataset"
)
psrq_command=(
  "$python_bin" -m mmctr.quantization.psrq_entrypoint
  --dataset-name "$dataset"
  --cuda "$gpu"
)
if [[ "$use_local_data" == "1" ]]; then
  rq_command+=(--use-local-data)
  psrq_command+=(--use-local-data)
fi

cd "$project_root"
if ((dry_run)); then
  printf 'Dry run:'
  printf ' %q' "${rq_command[@]}"
  printf '\nDry run:'
  printf ' %q' "${psrq_command[@]}"
  printf '\n'
  exit 0
fi

# Keep the two required artifact stages ordered; set -e prevents a partial success report.
printf 'Fitting RQ artifacts for %s\n' "$dataset"
"${rq_command[@]}"
printf 'Fitting PSRQ artifact for %s on cuda:%s\n' "$dataset" "$gpu"
"${psrq_command[@]}"
