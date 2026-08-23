#!/bin/bash
# =============================================================================
# run_mbpp.sh  —  MBPP evaluation using lm-eval (paper-accurate pass@1)
#
# Methods: dualcache, esdllm, caidllm, caidllm_enhanced
# (nocache already done: LLaDA=40%, use paper ref for Dream=60.4%)
#
# Usage:
#   bash run_mbpp.sh 0          # both models
#   bash run_mbpp.sh 0 llada    # LLaDA only (~6 hours)
#   bash run_mbpp.sh 0 dream    # Dream only (~6 hours)
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/mbpp"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    local BLOCK_FREQ=$2

    echo "================================================"
    echo " MBPP — ${MODEL} — 4 methods"
    echo " GPU=$GPU  DATE=$(date)"
    echo "================================================"

    BASE="--model ${MODEL} --task mbpp \
          --esdllm_mode HiddenState --alpha 0.5 \
          --prompt_update_freq 64 --block_update_freq ${BLOCK_FREQ} \
          --proportions 1 0.5 0.25 --positions 0 0.125 0.25"

    # 1. DualCache
    echo ""; echo "[1/4] dualcache..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model ${MODEL} --task mbpp --esdllm_mode DualCache \
        2>&1 | tee $LOG_DIR/${MODEL}_dualcache_${DATE}.log
    echo "[1/4] Done."

    # 2. ES-dLLM
    echo ""; echo "[2/4] esdllm..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        $BASE \
        2>&1 | tee $LOG_DIR/${MODEL}_esdllm_${DATE}.log
    echo "[2/4] Done."

    # 3. CAI-dLLM
    echo ""; echo "[3/4] caidllm..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        $BASE --use_cai --cai_mode apd_per_block \
        2>&1 | tee $LOG_DIR/${MODEL}_caidllm_${DATE}.log
    echo "[3/4] Done."

    # 4. CAI-dLLM + All Enhancements (Ideas 1+3+4+5)
    echo ""; echo "[4/4] caidllm_enhanced..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py \
        $BASE --use_cai --cai_mode apd_per_block \
        --all_enhancements \
        2>&1 | tee $LOG_DIR/${MODEL}_caidllm_enhanced_${DATE}.log
    echo "[4/4] Done."

    # Summary
    echo ""
    echo "=== ${MODEL} MBPP Results ==="
    for METHOD in dualcache esdllm caidllm caidllm_enhanced; do
        LOG="$LOG_DIR/${MODEL}_${METHOD}_${DATE}.log"
        time=$(grep "Total generation time" $LOG | awk '{print $4}')
        acc=$(grep "pass_at_1" $LOG | tail -1 | awk '{print $8}')
        req=$(grep "Request count" $LOG | awk '{print $3}')
        eff=$(grep "Effective tokens" $LOG | awk '{print $5}')
        tps=$(echo "$req $eff $time" | awk '{printf "%.1f", ($1*$2)/$3}' 2>/dev/null)
        printf "  %-20s | pass@1=%-6s | time=%-8s | tps=%s\n" \
            "$METHOD" "$acc" "${time}s" "$tps"
    done
}

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA-Instruct" 16
fi

if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream-Instruct" 8
fi