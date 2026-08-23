#!/bin/bash
# =============================================================================
# run_enhancements_gsm8k.sh
#
# Runs 5 enhancement evaluations in one command:
#   1. Idea 3 — Soft Belief Propagation
#   2. Idea 4 — Oracle Step Budget
#   3. Idea 5 — Prompt KV Prefix Cache
#   4. Idea 1 — Adaptive Block Sizing       ← NEW
#   5. All four combined (1+3+4+5)
#
# Usage:
#   bash run_enhancements_gsm8k.sh 0           # LLaDA
#   bash run_enhancements_gsm8k.sh 0 dream     # Dream
# =============================================================================

GPU=${1:-0}
MODEL=${2:-"llada"}
LOG_DIR="log_results/enhancements"
mkdir -p $LOG_DIR
DATE=$(date +%s)

if [ "$MODEL" == "dream" ]; then
    MODEL_NAME="Dream-Instruct"
    BLOCK_FREQ=8
else
    MODEL_NAME="LLaDA-Instruct"
    BLOCK_FREQ=4
fi

LOG="$LOG_DIR/${MODEL}_gsm8k_all_enhancements_${DATE}.log"

echo "================================================"
echo " Novel Enhancements — ${MODEL_NAME} GSM8K"
echo " GPU=$GPU  DATE=$(date)"
echo " Log: $LOG"
echo "================================================"

rm -rf lm_cache/

CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py \
    --model ${MODEL_NAME} \
    --task gsm8k \
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
echo " All 5 runs complete. Full log: $LOG"
echo "================================================"

echo ""
echo "=== QUICK SUMMARY ==="
grep -E "RUNNING:|flexible-extract|Total generation time" $LOG | \
    awk '/RUNNING:/{name=$0} /Total generation/{time=$4} /flexible-extract/{
        acc=$8; printf "%-50s | acc=%-8s | time=%s\n", name, acc, time}'
