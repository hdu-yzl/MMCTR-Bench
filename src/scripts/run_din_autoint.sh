#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"
RESULT_FILE="${RESULT_FILE:-$ROOT_DIR/experiments/din_autoint_results.log}"
USE_LOCAL_DATA="${USE_LOCAL_DATA:-0}"
# 使用的 GPU（可通过环境变量 CUDA 覆盖，默认单卡 0）
CUDA="${CUDA:-0}"

MODELS=(din autoint)
DATASETS=(antm2c microlens tiktok)

mkdir -p "$(dirname "$RESULT_FILE")"
: > "$RESULT_FILE"

run_task() {
    local model="$1"
    local dataset="$2"

    {
        echo "================================================================"
        echo "START $(date '+%Y-%m-%d %H:%M:%S') cuda=${CUDA} model=${model} dataset=${dataset}"
        echo "CMD: CUDA_VISIBLE_DEVICES=${CUDA} ${PYTHON_EXE} src/trainers/Trainers.py --model_name ${model} --dataset_name ${dataset} --cuda 0 --use_local_data ${USE_LOCAL_DATA}"
        echo "----------------------------------------------------------------"
    } >> "$RESULT_FILE"

    (
        cd "$ROOT_DIR" || exit 1
        CUDA_VISIBLE_DEVICES="$CUDA" "$PYTHON_EXE" src/trainers/Trainers.py \
            --model_name "$model" \
            --dataset_name "$dataset" \
            --cuda 0 \
            --use_local_data "$USE_LOCAL_DATA"
    ) >> "$RESULT_FILE" 2>&1

    local status=$?
    {
        echo "----------------------------------------------------------------"
        echo "END $(date '+%Y-%m-%d %H:%M:%S') cuda=${CUDA} model=${model} dataset=${dataset} status=${status}"
        echo
    } >> "$RESULT_FILE"

    return "$status"
}

echo "Result file: $RESULT_FILE"
echo "Python: $PYTHON_EXE"
echo "Use local data: $USE_LOCAL_DATA"
echo "CUDA: $CUDA"
"$PYTHON_EXE" --version

exit_code=0
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo ">>> Running model=${model} dataset=${dataset}"
        run_task "$model" "$dataset" || exit_code=1
    done
done

echo "All tasks finished. Result file: $RESULT_FILE"
exit "$exit_code"
