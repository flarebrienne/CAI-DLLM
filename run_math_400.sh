#!/bin/bash
# =============================================================================
# run_math_400.sh — MinervaMAth 400 samples, all 5 methods
#
# Usage:
#   bash run_math_400.sh 0          # both models
#   bash run_math_400.sh 0 llada    # LLaDA only
#   bash run_math_400.sh 0 dream    # Dream only
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/math"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    local BLOCK_FREQ=$2

    echo "================================================"
    echo " MinervaMAth 400 samples — ${MODEL} — 5 methods"
    echo " GPU=$GPU  DATE=$(date)"
    echo "================================================"

    BASE="--model ${MODEL} --task minerva_math --limit 400 \
          --esdllm_mode HiddenState --alpha 0.5 \
          --prompt_update_freq 64 --block_update_freq ${BLOCK_FREQ} \
          --proportions 1 0.5 0.25 --positions 0 0.125 0.25"

    # 1. Nocache
    echo ""; echo "[1/5] nocache..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model ${MODEL} --task minerva_math \
        --esdllm_mode nocache --limit 400 \
        2>&1 | tee $LOG_DIR/${MODEL}_nocache_400_${DATE}.log
    echo "[1/5] Done."

    # 2. DualCache
    echo ""; echo "[2/5] dualcache..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model ${MODEL} --task minerva_math \
        --esdllm_mode DualCache --limit 400 \
        2>&1 | tee $LOG_DIR/${MODEL}_dualcache_400_${DATE}.log
    echo "[2/5] Done."

    # 3. ES-dLLM
    echo ""; echo "[3/5] esdllm..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        $BASE \
        2>&1 | tee $LOG_DIR/${MODEL}_esdllm_400_${DATE}.log
    echo "[3/5] Done."

    # 4. CAI-dLLM
    echo ""; echo "[4/5] caidllm..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        $BASE --use_cai --cai_mode apd_per_block \
        2>&1 | tee $LOG_DIR/${MODEL}_caidllm_400_${DATE}.log
    echo "[4/5] Done."

    # 5. CAI-dLLM + All Enhancements
    echo ""; echo "[5/5] caidllm_enhanced..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py \
        $BASE --use_cai --cai_mode apd_per_block \
        --all_enhancements \
        2>&1 | tee $LOG_DIR/${MODEL}_caidllm_enhanced_400_${DATE}.log
    echo "[5/5] Done."

    # Summary
    echo ""
    echo "=== ${MODEL} MinervaMAth (400 samples) ==="
    for METHOD in nocache dualcache esdllm caidllm caidllm_enhanced; do
        LOG="$LOG_DIR/${MODEL}_${METHOD}_400_${DATE}.log"
        time=$(grep "Total generation time" $LOG | tail -1 | awk '{print $4}')
        acc=$(grep "exact_match\|flexible" $LOG | tail -1 | awk '{print $8}')
        req=$(grep "Request count" $LOG | tail -1 | awk '{print $3}')
        eff=$(grep "Effective tokens" $LOG | tail -1 | awk '{print $5}')
        tps=$(echo "$req $eff $time" | awk '{printf "%.1f", ($1*$2)/$3}' 2>/dev/null)
        printf "  %-22s | acc=%-8s | time=%-8s | tps=%s\n" \
            "$METHOD" "$acc" "${time}s" "$tps"
    done
}

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA-Instruct" 16
fi

if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream-Instruct" 8
fi