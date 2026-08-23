#!/bin/bash
# =============================================================================
# run_llada_nocache_baseline.sh  —  LLaDA-Instruct nocache baseline + full table
# =============================================================================
# Runs the vanilla baseline (no ES-dLLM, no CAI) and the full CAI ablation
# to complete the comparison table for the paper.
#
# Usage:
#   bash run_llada_nocache_baseline.sh 0    # GPU 0
# =============================================================================

GPU=${1:-0}
LOG_DIR="log_results/llada_cai"
mkdir -p $LOG_DIR
DATE=$(date +%s)

echo "============================================================"
echo " LLaDA-Instruct Full Ablation on GPU $GPU"
echo "============================================================"

# ── 1. Nocache baseline (vanilla LLaDA, no acceleration) ─────────────────────
echo ""
echo "[1/3] Running: nocache baseline (vanilla LLaDA)..."
CUDA_VISIBLE_DEVICES=$GPU python eval.py \
    --model LLaDA-Instruct \
    --task gsm8k \
    --esdllm_mode nocache \
    2>&1 | tee $LOG_DIR/llada_gsm8k_nocache_${DATE}.log

echo "[1/3] Done."

# ── 2. ES-dLLM baseline (already have this from earlier runs, but re-run cleanly)
echo ""
echo "[2/3] Running: ES-dLLM baseline (HiddenState)..."
CUDA_VISIBLE_DEVICES=$GPU python eval.py \
    --model LLaDA-Instruct \
    --task gsm8k \
    --esdllm_mode HiddenState \
    --alpha 0.5 \
    --prompt_update_freq 64 \
    --block_update_freq 16 \
    --proportions 1 0.5 0.25 \
    --positions 0 0.125 0.25 \
    2>&1 | tee $LOG_DIR/llada_gsm8k_esdllm_${DATE}.log

echo "[2/3] Done."

# ── 3. CAI best config (apd_per_block + confidence gating) ───────────────────
echo ""
echo "[3/3] Running: CAI-dLLM best config..."
rm -rf lm_cache/
CUDA_VISIBLE_DEVICES=$GPU python eval.py \
    --model LLaDA-Instruct \
    --task gsm8k \
    --esdllm_mode HiddenState \
    --alpha 0.5 \
    --prompt_update_freq 64 \
    --block_update_freq 16 \
    --proportions 1 0.5 0.25 \
    --positions 0 0.125 0.25 \
    --use_cai \
    --cai_mode apd_per_block \
    2>&1 | tee $LOG_DIR/llada_gsm8k_cai_best_${DATE}.log

echo "[3/3] Done."

echo ""
echo "============================================================"
echo " RESULTS SUMMARY"
echo "============================================================"
for log in $LOG_DIR/*_${DATE}.log; do
    name=$(basename $log .log)
    time=$(grep "Total generation time" $log | awk '{print $4}')
    acc=$(grep "flexible-extract" $log | awk '{print $8}')
    echo "  $name  |  time=${time}s  |  acc=${acc}"
done
