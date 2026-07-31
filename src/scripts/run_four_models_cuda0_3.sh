#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python}"
RESULT_FILE="${RESULT_FILE:-$ROOT_DIR/experiments/four_models_cuda0_3_results.log}"
USE_LOCAL_DATA="${USE_LOCAL_DATA:-0}"
LOCK_DIR="$RESULT_FILE.lock"

MODELS=(mb pamd mmmlp m3srec)
DATASETS=(antm2c microlens tiktok)
GPUS=(4 5 6 7)

mkdir -p "$(dirname "$RESULT_FILE")"
: > "$RESULT_FILE"

append_result() {
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        sleep 0.2
    done
    cat "$1" >> "$RESULT_FILE"
    rmdir "$LOCK_DIR"
}

run_task() {
    local cuda="$1"
    local model="$2"
    local dataset="$3"
    local tmp_log
    tmp_log="$(mktemp)"

    {
        echo "================================================================"
        echo "START $(date '+%Y-%m-%d %H:%M:%S') cuda=${cuda} model=${model} dataset=${dataset}"
        echo "CMD: CUDA_VISIBLE_DEVICES=${cuda} ${PYTHON_EXE} src/trainers/Trainers.py --model_name ${model} --dataset_name ${dataset} --cuda 0 --use_local_data ${USE_LOCAL_DATA}"
        echo "----------------------------------------------------------------"
    } > "$tmp_log"

    (
        cd "$ROOT_DIR" || exit 1
        CUDA_VISIBLE_DEVICES="$cuda" "$PYTHON_EXE" src/trainers/Trainers.py \
            --model_name "$model" \
            --dataset_name "$dataset" \
            --cuda 0 \
            --use_local_data "$USE_LOCAL_DATA"
    ) >> "$tmp_log" 2>&1

    local status=$?
    {
        echo "----------------------------------------------------------------"
        echo "END $(date '+%Y-%m-%d %H:%M:%S') cuda=${cuda} model=${model} dataset=${dataset} status=${status}"
        echo
    } >> "$tmp_log"

    append_result "$tmp_log"
    rm -f "$tmp_log"
    return "$status"
}

worker() {
    local worker_id="$1"
    local cuda="${GPUS[$worker_id]}"
    local task_id=0
    local failed=0

    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            if (( task_id % ${#GPUS[@]} == worker_id )); then
                run_task "$cuda" "$model" "$dataset" || failed=1
            fi
            task_id=$((task_id + 1))
        done
    done

    return "$failed"
}

echo "Result file: $RESULT_FILE"
echo "Python: $PYTHON_EXE"
echo "Use local data: $USE_LOCAL_DATA"
"$PYTHON_EXE" --version

pids=()
for worker_id in "${!GPUS[@]}"; do
    worker "$worker_id" &
    pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
    wait "$pid" || exit_code=1
done

echo "All tasks finished. Result file: $RESULT_FILE"
exit "$exit_code"
