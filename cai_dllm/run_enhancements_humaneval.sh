#!/bin/bash
# =============================================================================
# run_enhancements_humaneval.sh
#
# Runs 4 enhancement evaluations on HumanEval for LLaDA and/or Dream:
#   1. Idea 3 — Soft Belief Propagation
#   2. Idea 4 — Oracle Step Budget
#   3. Idea 5 — Prompt KV Prefix Cache
#   4. All three combined
#
# Usage:
#   bash run_enhancements_humaneval.sh 0          # both models
#   bash run_enhancements_humaneval.sh 0 llada    # LLaDA only
#   bash run_enhancements_humaneval.sh 0 dream    # Dream only
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/enhancements"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    local BLOCK_FREQ=$2
    local LOG="$LOG_DIR/${MODEL}_humaneval_enhancements_${DATE}.log"

    echo "================================================"
    echo " Novel Enhancements — ${MODEL} HumanEval"
    echo " GPU=$GPU  DATE=$(date)"
    echo " Log: $LOG"
    echo "================================================"

    rm -rf lm_cache/

    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py \
        --model ${MODEL} \
        --task humaneval \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq ${BLOCK_FREQ} \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        --use_cai \
        --cai_mode apd_per_block \
        --all_enhancements \
        2>&1 | tee $LOG

    echo ""
    echo "================================================"
    echo " Done: ${MODEL} HumanEval. Log: $LOG"
    echo "================================================"

    echo ""
    echo "=== QUICK SUMMARY: ${MODEL} HumanEval ==="
    grep -E "RUNNING:|pass@1|Total generation time" $LOG | \
        awk '/RUNNING:/{name=$0} /Total generation/{time=$4} /pass@1/{
            acc=$6; printf "%-45s | pass@1=%-8s | time=%s\n", name, acc, time}'
}

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA-Instruct" 4
fi

if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream-Instruct" 8
fi