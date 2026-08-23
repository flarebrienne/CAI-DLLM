#!/bin/bash
# =============================================================================
# run_abs_dream_gsm8k.sh
#
# Runs 2 evaluations on Dream-7B-Instruct GSM8K:
#   1. CAI-dLLM + Idea 1 (Adaptive Block Sizing) only
#   2. CAI-dLLM + All Combined (Ideas 1+3+4+5)
#
# Usage:
#   bash run_abs_dream_gsm8k.sh 0
# =============================================================================

GPU=${1:-0}
LOG_DIR="log_results/enhancements"
mkdir -p $LOG_DIR
DATE=$(date +%s)

echo "================================================"
echo " Adaptive Block Sizing — Dream-7B GSM8K"
echo " GPU=$GPU  DATE=$(date)"
echo "================================================"

BASE_ARGS="--model Dream-Instruct --task gsm8k \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 8 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
    --use_cai --cai_mode apd_per_block"

# ── Run 1: Idea 1 only ───────────────────────────────────────────────────────
echo ""
echo "[1/2] CAI-dLLM + Idea 1 (Adaptive Block Sizing only)..."
rm -rf lm_cache/
LOG1="$LOG_DIR/dream_gsm8k_abs_only_${DATE}.log"
CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py $BASE_ARGS \
    --use_adaptive_block \
    2>&1 | tee $LOG1
echo "[1/2] Done."

# ── Run 2: All Combined ───────────────────────────────────────────────────────
echo ""
echo "[2/2] CAI-dLLM + All Combined (Ideas 1+3+4+5)..."
rm -rf lm_cache/
LOG2="$LOG_DIR/dream_gsm8k_all_combined_${DATE}.log"
CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py $BASE_ARGS \
    --use_adaptive_block --use_soft_belief \
    --use_oracle_budget --use_prompt_kvcache \
    2>&1 | tee $LOG2
echo "[2/2] Done."

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo " RESULTS"
echo "================================================"
for LOG in $LOG1 $LOG2; do
    label=$(grep "RUNNING\|use_adaptive" $LOG 2>/dev/null | head -1 || echo "$LOG")
    time=$(grep "Total generation time" $LOG | awk '{print $4}')
    acc=$(grep "flexible-extract" $LOG | tail -1 | awk '{print $8}')
    req=$(grep "Request count" $LOG | awk '{print $3}')
    eff=$(grep "Effective tokens" $LOG | awk '{print $5}')
    tps=$(echo "$req $eff $time" | awk '{printf "%.1f", ($1*$2)/$3}' 2>/dev/null)
    echo "  time=${time}s  tps=${tps}  acc=${acc}  [$(basename $LOG)]"
done
echo "================================================"