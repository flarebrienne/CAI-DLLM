#!/bin/bash
# run_openbookqa.sh — OpenBookQA 500 samples, all methods, both models
# Usage: bash run_openbookqa.sh 0 llada | dream | both

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/commonsense"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    echo ""; echo "=== OpenBookQA — ${MODEL} ==="; echo ""
    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval_commonsense.py \
        --model ${MODEL} \
        --methods nocache,dualcache,esdllm,caidllm \
        --n_samples 500 \
        --tasks openbookqa \
        --output_dir $LOG_DIR \
        2>&1 | tee $LOG_DIR/${MODEL}_openbookqa_${DATE}.log
}

[ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ] && run_model "LLaDA-Instruct"
[ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ] && run_model "Dream-Instruct"
