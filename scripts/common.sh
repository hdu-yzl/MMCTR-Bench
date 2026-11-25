# Shared setup for executable launchers. Source this file; do not execute it.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="antm2c"
gpu="0"
python_bin="${MMCTR_PYTHON:-}"
output_root="${MMCTR_OUTPUT_ROOT:-outputs}"
num_threads="${MMCTR_NUM_THREADS:-8}"
use_local_data="${MMCTR_USE_LOCAL_DATA:-0}"
dry_run=0
common_options="--dataset --python --num-threads --use-local-data --dry-run -h --help"

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

# Parse options shared by launchers; callers handle their own model/GPU-list options.
common_option() {
  [[ " $common_options " == *" $1 "* ]] || die "unknown option: $1"
  consumed=1
  case "$1" in
    --dataset|--gpu|--python|--output-root|--num-threads)
      (($# >= 2)) || die "$1 requires a value"
      case "$1" in
        --dataset) dataset="$2" ;;
        --gpu) gpu="$2" ;;
        --python) python_bin="$2" ;;
        --output-root) output_root="$2" ;;
        --num-threads) num_threads="$2" ;;
      esac
      consumed=2 ;;
    --use-local-data) use_local_data=1 ;;
    --dry-run) dry_run=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
}

setup_runtime() {
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
  [[ "$num_threads" =~ ^[1-9][0-9]*$ ]] || die "--num-threads must be a positive integer"
  if [[ -z "$python_bin" ]]; then
    python_bin="$(command -v python || true)"
  elif [[ "$python_bin" != */* ]]; then
    python_bin="$(command -v "$python_bin" || true)"
  fi
  [[ -n "$python_bin" && -x "$python_bin" ]] || die \
    "Python was not found; activate bm or pass --python /absolute/path/to/python"
  # Resolve a relative interpreter before changing the working directory.
  python_bin="$(cd -- "$(dirname -- "$python_bin")" && pwd)/$(basename -- "$python_bin")"
  export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
  export TMPDIR="$project_root/.tmp"
  export OMP_NUM_THREADS="$num_threads" MKL_NUM_THREADS="$num_threads"
  export OPENBLAS_NUM_THREADS="$num_threads" NUMEXPR_NUM_THREADS="$num_threads"
  mkdir -p "$TMPDIR"
  cd "$project_root"
}

validate_name() {
  local kind="$1" value="$2" choices
  choices="$("$python_bin" -m mmctr.cli "list-${kind}s")"
  [[ $'\n'"$choices"$'\n' == *$'\n'"$value"$'\n'* ]] || die "unknown $kind: $value"
}

print_command() { printf 'Dry run:'; printf ' %q' "$@"; printf '\n'; }

build_train_command() {
  command=("$python_bin" -m mmctr.cli train --dataset-name "$dataset"
    --model-name "$1" --cuda "$2" --output-root "$output_root" --num-threads "$num_threads")
  [[ "$use_local_data" != 1 ]] || command+=(--use-local-data)
}
