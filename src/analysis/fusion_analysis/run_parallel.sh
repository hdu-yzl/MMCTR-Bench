#!/usr/bin/env bash
# ============================================================================
# 融合分析 —— 并行启动脚本
#
# 将 (模型 × 融合方法) 任务集合按 round-robin 分配到指定的 CUDA 设备并行执行。
# 每个 CUDA 设备作为一个 worker，串行处理被分配到它的所有任务，避免 OOM。
#
# 用法:
#   bash src/analysis/fusion_analysis/run_parallel.sh \
#       [--dataset DATASET] \
#       [--models "MODEL1 MODEL2 ..."] \
#       [--fusions "FUSION1 FUSION2 ..."] \
#       [--gpus "GPU1 GPU2 ..."] \
#       [--max_epochs N] \
#       [--use_local_data 0|1] \
#       [--seed N] \
#       [--python PYTHON_EXE]
#
# 示例:
#   # 用 4 张 GPU 跑所有模型 × 所有融合方法（默认）
#   bash src/analysis/fusion_analysis/run_parallel.sh --gpus "0 1 2 3"
#
#   # 指定模型、融合方法和 GPU
#   bash src/analysis/fusion_analysis/run_parallel.sh \
#       --models "MB PAMD M3SRec MMMLP" \
#       --fusions "maf cat dta gmmf" \
#       --gpus "0 1"
#
#   # 单 GPU 串行跑（仅在 cuda 4 上，用 antm2c 数据集）
#   bash src/analysis/fusion_analysis/run_parallel.sh \
#       --dataset antm2c --gpus "4"
# ============================================================================
set -u

# ── 默认值 ──
DATASET="antm2c"
MODELS_DEFAULT="qarm mcca"
FUSIONS_DEFAULT="maf cat lmf src mtfn fq-former simcen dta gmmf dmf"
GPUS_DEFAULT="0 1 2 3 4 5 6 7"
MAX_EPOCHS=""
USE_LOCAL_DATA="0"
SEED="2025"
PYTHON_EXE="${PYTHON_EXE:-python}"

MODELS_ARG=""
FUSIONS_ARG=""
GPUS_ARG=""

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|--dataset_name)
            DATASET="$2"; shift 2 ;;
        --models)
            MODELS_ARG="$2"; shift 2 ;;
        --fusions)
            FUSIONS_ARG="$2"; shift 2 ;;
        --gpus|--cuda)
            GPUS_ARG="$2"; shift 2 ;;
        --max_epochs)
            MAX_EPOCHS="$2"; shift 2 ;;
        --use_local_data)
            USE_LOCAL_DATA="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --python)
            PYTHON_EXE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *)
            echo "[WARN] 未知参数: $1"; shift ;;
    esac
done

MODELS_STR="${MODELS_ARG:-$MODELS_DEFAULT}"
FUSIONS_STR="${FUSIONS_ARG:-$FUSIONS_DEFAULT}"
GPUS_STR="${GPUS_ARG:-$GPUS_DEFAULT}"

# 拆分为数组
read -r -a MODELS <<< "$MODELS_STR"
read -r -a FUSIONS <<< "$FUSIONS_STR"
read -r -a GPUS <<< "$GPUS_STR"

if [[ ${#MODELS[@]} -eq 0 || ${#FUSIONS[@]} -eq 0 || ${#GPUS[@]} -eq 0 ]]; then
    echo "[ERROR] models/fusions/gpus 不能为空" >&2
    exit 1
fi

# ── 定位项目根目录 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_ROOT="$ROOT_DIR/experiments/logs/fusion_parallel_${DATASET}_${TIMESTAMP}"
SUMMARY_FILE="$LOG_ROOT/summary.log"
LOCK_DIR="$LOG_ROOT/.summary.lock"
mkdir -p "$LOG_ROOT"

# ── 打印配置 ──
{
    echo "============================================================"
    echo "融合分析并行执行  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo "数据集     : $DATASET"
    echo "模型       : ${MODELS[*]}  (共 ${#MODELS[@]} 个)"
    echo "融合方法   : ${FUSIONS[*]}  (共 ${#FUSIONS[@]} 个)"
    echo "GPU        : ${GPUS[*]}  (共 ${#GPUS[@]} 张)"
    echo "总任务数   : $(( ${#MODELS[@]} * ${#FUSIONS[@]} ))"
    echo "max_epochs : ${MAX_EPOCHS:-<config 默认>}"
    echo "本地数据   : $USE_LOCAL_DATA"
    echo "随机种子   : $SEED"
    echo "Python     : $PYTHON_EXE  ($($PYTHON_EXE --version 2>&1))"
    echo "日志目录   : $LOG_ROOT"
    echo "============================================================"
} | tee "$SUMMARY_FILE"

# ── 记录摘要（带文件锁，避免并发写损坏） ──
append_summary() {
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        sleep 0.1
    done
    cat "$1" >> "$SUMMARY_FILE"
    rmdir "$LOCK_DIR"
}

# ── 单个任务执行 ──
run_task() {
    local cuda="$1"
    local model="$2"
    local fusion="$3"
    local task_log="$LOG_ROOT/${model}__${fusion}__cuda${cuda}.log"
    local tmp_summary
    tmp_summary="$(mktemp)"

    {
        echo "------------------------------------------------------------"
        echo "START $(date '+%Y-%m-%d %H:%M:%S') cuda=${cuda} model=${model} fusion=${fusion}"
        echo "LOG  : $task_log"
    } > "$tmp_summary"

    local extra_args=()
    if [[ -n "$MAX_EPOCHS" ]]; then
        extra_args+=(--max_epochs "$MAX_EPOCHS")
    fi

    local cmd_str="CUDA_VISIBLE_DEVICES=${cuda} ${PYTHON_EXE} src/analysis/fusion_analysis/run_analysis.py \
        --dataset_name ${DATASET} --models ${model} --fusions ${fusion} \
        --cuda 0 --seed ${SEED} --use_local_data ${USE_LOCAL_DATA} ${extra_args[*]}"
    echo "CMD  : $cmd_str" >> "$tmp_summary"

    (
        cd "$ROOT_DIR" || exit 1
        CUDA_VISIBLE_DEVICES="$cuda" "$PYTHON_EXE" src/analysis/fusion_analysis/run_analysis.py \
            --dataset_name "$DATASET" \
            --models "$model" \
            --fusions "$fusion" \
            --cuda 0 \
            --seed "$SEED" \
            --use_local_data "$USE_LOCAL_DATA" \
            "${extra_args[@]}"
    ) > "$task_log" 2>&1
    local status=$?

    {
        echo "END   $(date '+%Y-%m-%d %H:%M:%S') cuda=${cuda} model=${model} fusion=${fusion} status=${status}"
    } >> "$tmp_summary"

    append_summary "$tmp_summary"
    rm -f "$tmp_summary"
    return "$status"
}

# ── Worker：一个 GPU 一个 worker，按 round-robin 取任务 ──
# 任务展开为线性列表 (model_i, fusion_j)，task_id = i * |F| + j
worker() {
    local worker_id="$1"
    local cuda="${GPUS[$worker_id]}"
    local nworkers="${#GPUS[@]}"
    local task_id=0
    local failed=0

    for model in "${MODELS[@]}"; do
        for fusion in "${FUSIONS[@]}"; do
            if (( task_id % nworkers == worker_id )); then
                run_task "$cuda" "$model" "$fusion" || failed=1
            fi
            task_id=$((task_id + 1))
        done
    done

    return "$failed"
}

# ── 启动并行 worker ──
pids=()
for worker_id in "${!GPUS[@]}"; do
    worker "$worker_id" &
    pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
    wait "$pid" || exit_code=1
done

{
    echo "============================================================"
    echo "全部任务结束  $(date '+%Y-%m-%d %H:%M:%S')  exit_code=$exit_code"
    echo "汇总日志: $SUMMARY_FILE"
    echo "单任务日志目录: $LOG_ROOT"
    echo "============================================================"
} | tee -a "$SUMMARY_FILE"

exit "$exit_code"
