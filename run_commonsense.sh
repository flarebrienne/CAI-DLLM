#!/bin/bash
# =============================================================================
# run_commonsense.sh
#
# Evaluates nocache, dualcache, esdllm, caidllm on:
#   WinoGrande (500), PIQA (500), OpenBookQA (500)
#
# Usage:
#   bash run_commonsense.sh 0          # both models
#   bash run_commonsense.sh 0 llada    # LLaDA only (~1.5 hours)
#   bash run_commonsense.sh 0 dream    # Dream only (~1.5 hours)
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/commonsense"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    local LOG="$LOG_DIR/${MODEL}_commonsense_${DATE}.log"

    echo "================================================"
    echo " Commonsense Benchmarks — ${MODEL}"
    echo " GPU=$GPU  DATE=$(date)"
    echo " Tasks: WinoGrande, PIQA, OpenBookQA (500 each)"
    echo "================================================"

    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval_commonsense.py \
        --model ${MODEL} \
        --methods nocache,dualcache,esdllm,caidllm \
        --n_samples 500 \
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
