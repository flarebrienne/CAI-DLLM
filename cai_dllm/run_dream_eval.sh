#!/bin/bash
# =============================================================================
# run_dream_eval.sh  —  CAI-dLLM evaluation on Dream-Instruct
# =============================================================================
# Usage:
#   bash run_dream_eval.sh 0          # GPU 0
#   bash run_dream_eval.sh 1          # GPU 1
# =============================================================================

GPU=${1:-0}
LOG_DIR="log_results/dream_cai"
mkdir -p $LOG_DIR
DATE=$(date +%s)

echo "============================================================"
echo " CAI-dLLM Dream-Instruct Evaluation on GPU $GPU"
echo "============================================================"

# ── 1. Nocache baseline (no ES-dLLM, no CAI) ─────────────────────────────────
echo ""
echo "[1/4] Running: nocache baseline (vanilla Dream)..."
CUDA_VISIBLE_DEVICES=$GPU python eval.py \
    --model Dream-Instruct \
    --task gsm8k \
    --esdllm_mode nocache \
    2>&1 | tee $LOG_DIR/dream_gsm8k_nocache_${DATE}.log

echo "[1/4] Done."

# ── 2. ES-dLLM baseline (HiddenState, no CAI) ────────────────────────────────
echo ""
echo "[2/4] Running: ES-dLLM baseline (HiddenState)..."
CUDA_VISIBLE_DEVICES=$GPU python eval.py \
    --model Dream-Instruct \
    --task gsm8k \
    --esdllm_mode HiddenState \
    --alpha 0.5 \
    --prompt_update_freq 64 \
    --block_update_freq 8 \
    --proportions 1 0.5 0.25 \
    --positions 0 0.125 0.25 \
    2>&1 | tee $LOG_DIR/dream_gsm8k_esdllm_${DATE}.log

echo "[2/4] Done."

# ── 3. CAI APD only (§4.1) ────────────────────────────────────────────────────
echo ""
echo "[3/4] Running: CAI-dLLM apd_only..."
rm -rf lm_cache/
CUDA_VISIBLE_DEVICES=$GPU python eval.py \
    --model Dream-Instruct \
    --task gsm8k \
    --esdllm_mode HiddenState \
    --alpha 0.5 \
    --prompt_update_freq 64 \
    --block_update_freq 8 \
    --proportions 1 0.5 0.25 \
    --positions 0 0.125 0.25 \
    --use_cai \
    --cai_mode apd_only \
    2>&1 | tee $LOG_DIR/dream_gsm8k_cai_apd_${DATE}.log

echo "[3/4] Done."

# ── 4. CAI full (§4.1 + §4.2 + confidence gating) ────────────────────────────
echo ""
echo "[4/4] Running: CAI-dLLM apd_per_block + confidence gating..."
rm -rf lm_cache/
CUDA_VISIBLE_DEVICES=$GPU python eval.py \
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
    2>&1 | tee $LOG_DIR/dream_gsm8k_cai_full_${DATE}.log

echo "[4/4] Done."

echo ""
echo "============================================================"
echo " All runs complete. Logs saved to $LOG_DIR/"
echo "============================================================"

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "=== RESULTS SUMMARY ==="
for log in $LOG_DIR/*_${DATE}.log; do
    name=$(basename $log .log)
    time=$(grep "Total generation time" $log | awk '{print $4}')
    acc=$(grep "flexible-extract" $log | awk '{print $8}')
    echo "  $name  |  time=${time}s  |  acc=${acc}"
done
