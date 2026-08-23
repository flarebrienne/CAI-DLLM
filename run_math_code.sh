#!/bin/bash
# =============================================================================
# run_math_code.sh
#
# Evaluates all baselines on MathQA (500) and MBPP (257):
#   nocache, dualcache, esdllm, caidllm, caidllm_enhanced
#
# Usage:
#   bash run_math_code.sh 0          # both models
#   bash run_math_code.sh 0 llada    # LLaDA only (~1.5 hours)
#   bash run_math_code.sh 0 dream    # Dream only (~1.5 hours)
#   bash run_math_code.sh 0 both mathqa   # MathQA only
#   bash run_math_code.sh 0 both mbpp     # MBPP only
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
TASKS=${3:-"mathqa,mbpp"}
LOG_DIR="log_results/math_code"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    local LOG="$LOG_DIR/${MODEL}_math_code_${DATE}.log"

    echo "================================================"
    echo " Math & Code — ${MODEL}"
    echo " Tasks: $TASKS"
    echo " GPU=$GPU  DATE=$(date)"
    echo "================================================"

    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval_math_code.py \
        --model ${MODEL} \
        --tasks $TASKS \
        --methods nocache,dualcache,esdllm,caidllm,caidllm_enhanced \
        --n_mathqa 500 \
        --n_mbpp 257 \
        --output_dir $LOG_DIR \
        2>&1 | tee $LOG

    echo ""
    echo "================================================"
    echo " Done: ${MODEL}. Log: $LOG"
    echo "================================================"
}

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA-Instruct"
fi

if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream-Instruct"
fi
