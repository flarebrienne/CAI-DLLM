#!/bin/bash
# =============================================================================
# run_piqa_enhanced.sh
#
# Runs PIQA with CAI-dLLM + All Enhancements (Ideas 1+3+4+5)
# for comparison with baseline methods
#
# Usage:
#   bash run_piqa_enhanced.sh 0          # both models
#   bash run_piqa_enhanced.sh 0 llada    # LLaDA only
#   bash run_piqa_enhanced.sh 0 dream    # Dream only
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/commonsense"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    local BLOCK_FREQ=$2
    local LOG="$LOG_DIR/${MODEL}_piqa_enhanced_${DATE}.log"

    echo "================================================"
    echo " PIQA + All Enhancements — ${MODEL}"
    echo " GPU=$GPU  DATE=$(date)"
    echo "================================================"

    # Step 1: Run baselines (nocache, dualcache, esdllm, caidllm)
    echo ""; echo "[1/2] Running baselines (nocache, dualcache, esdllm, caidllm)..."
    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval_commonsense.py \
        --model ${MODEL} \
        --methods nocache,dualcache,esdllm,caidllm \
        --n_samples 500 \
        --tasks piqa \
        --output_dir $LOG_DIR \
        2>&1 | tee $LOG

    # Step 2: Run CAI + All Enhancements (Ideas 1+3+4+5)
    echo ""; echo "[2/2] Running CAI + All Enhancements (Ideas 1+3+4+5)..."
    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval_commonsense_enhanced.py \
        --model ${MODEL} \
        --block_update_freq ${BLOCK_FREQ} \
        --n_samples 500 \
        --tasks piqa \
        --output_dir $LOG_DIR \
        2>&1 | tee -a $LOG

    echo ""
    echo "================================================"
    echo " Done: ${MODEL} PIQA. Log: $LOG"
    echo "================================================"
}

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA-Instruct" 4
fi

if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream-Instruct" 8
fi
