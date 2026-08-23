#!/bin/bash
# =============================================================================
# run_poem_completion.sh
#
# Evaluates poem completion task (forward + reversal) on LLaDA and Dream
# Methods: nocache, dualcache, esdllm, caidllm, caidllm_no_cg
# Also prints reference AR model scores from LLaDA paper
#
# Usage:
#   bash run_poem_completion.sh 0          # both models
#   bash run_poem_completion.sh 0 llada    # LLaDA only
#   bash run_poem_completion.sh 0 dream    # Dream only
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/poem"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    local LOG="$LOG_DIR/${MODEL}_poem_${DATE}.log"

    echo "================================================"
    echo " Poem Completion — ${MODEL}"
    echo " GPU=$GPU  DATE=$(date)"
    echo "================================================"

    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval_poem_completion.py \
        --model ${MODEL} \
        --methods nocache,dualcache,esdllm,caidllm,caidllm_no_cg \
        --n_samples 496 \
        --batch_size 8 \
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

echo ""
echo "================================================"
echo " Reference AR Models (LLaDA Paper)"
echo "================================================"
echo "  GPT-4o (2024-08-06):    Forward=82.7%  Reversal=34.3%"
echo "  Qwen2.5-7B-Instruct:    Forward=75.9%  Reversal=38.0%"
echo "  LLaDA-8B (paper ref):   Forward=51.8%  Reversal=45.6%"
echo "================================================"
