#!/bin/bash
# =============================================================================
# run_enhancements_dream_gsm8k.sh
#
# Runs 4 enhancement evaluations on Dream-7B-Instruct GSM8K:
#   1. Idea 3 — Soft Belief Propagation
#   2. Idea 4 — Oracle Step Budget
#   3. Idea 5 — Prompt KV Prefix Cache
#   4. All three combined
#
# Usage:
#   bash run_enhancements_dream_gsm8k.sh 0
# =============================================================================

GPU=${1:-0}
LOG_DIR="log_results/enhancements"
mkdir -p $LOG_DIR
DATE=$(date +%s)
LOG="$LOG_DIR/dream_all_enhancements_${DATE}.log"

echo "================================================"
echo " Novel Enhancements — Dream-7B GSM8K"
echo " GPU=$GPU  DATE=$(date)"
echo " Log: $LOG"
echo "================================================"

rm -rf lm_cache/

CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py \
    --model Dream-Instruct \
    --task gsm8k \
    --esdllm_mode HiddenState \
    --alpha 0.5 \
    --prompt_update_freq 64 \
    --block_update_freq 8 \
    --proportions 1 0.5 0.25 \
    --positions 0 0.125 0.25 \
    --use_cai \
    --cai_mode apd_per_block \
    --all_enhancements \
    2>&1 | tee $LOG

echo ""
echo "================================================"
echo " All 4 runs complete. Full log: $LOG"
echo "================================================"

echo ""
echo "=== QUICK SUMMARY ==="
grep -E "RUNNING:|flexible-extract|Total generation time" $LOG | \
    awk '/RUNNING:/{name=$0} /Total generation/{time=$4} /flexible-extract/{
        acc=$8; printf "%-45s | acc=%-8s | time=%s\n", name, acc, time}'