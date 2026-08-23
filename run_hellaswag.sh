#!/bin/bash
# =============================================================================
# run_hellaswag.sh  —  HellaSwag evaluation (500 samples)
#
# Compares: nocache, dualcache, esdllm, caidllm
# Models:   LLaDA-Instruct, Dream-Instruct
#
# Usage:
#   bash run_hellaswag.sh 0          # both models
#   bash run_hellaswag.sh 0 llada    # LLaDA only
#   bash run_hellaswag.sh 0 dream    # Dream only
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/hellaswag"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    local BLOCK_FREQ=$2

    echo "================================================"
    echo " HellaSwag — ${MODEL} — 4 methods"
    echo " GPU=$GPU  DATE=$(date)"
    echo "================================================"

    BASE="--model ${MODEL} --task hellaswag --limit 500 \
          --esdllm_mode HiddenState --alpha 0.5 \
          --prompt_update_freq 64 --block_update_freq ${BLOCK_FREQ} \
          --proportions 1 0.5 0.25 --positions 0 0.125 0.25"

    # 1. Nocache
    echo ""; echo "[1/4] nocache..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        $BASE --esdllm_mode nocache \
        2>&1 | tee $LOG_DIR/${MODEL}_nocache_${DATE}.log
    echo "[1/4] Done."

    # 2. DualCache
    echo ""; echo "[2/4] dualcache..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        $BASE --esdllm_mode DualCache \
        2>&1 | tee $LOG_DIR/${MODEL}_dualcache_${DATE}.log
    echo "[2/4] Done."

    # 3. ES-dLLM
    echo ""; echo "[3/4] esdllm..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        $BASE --esdllm_mode HiddenState \
        2>&1 | tee $LOG_DIR/${MODEL}_esdllm_${DATE}.log
    echo "[3/4] Done."

    # 4. CAI-dLLM
    echo ""; echo "[4/4] caidllm..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        $BASE --esdllm_mode HiddenState \
        --use_cai --cai_mode apd_per_block \
        2>&1 | tee $LOG_DIR/${MODEL}_caidllm_${DATE}.log
    echo "[4/4] Done."

    # Summary
    echo ""
    echo "=== ${MODEL} HellaSwag Results ==="
    for METHOD in nocache dualcache esdllm caidllm; do
        LOG="$LOG_DIR/${MODEL}_${METHOD}_${DATE}.log"
        time=$(grep "Total generation time" $LOG | awk '{print $4}')
        acc=$(grep "acc_norm\|acc," $LOG | grep -v "stderr" | tail -1 | awk '{print $NF}')
        req=$(grep "Request count" $LOG | awk '{print $3}')
        eff=$(grep "Effective tokens" $LOG | awk '{print $5}')
        tps=$(echo "$req $eff $time" | awk '{printf "%.1f", ($1*$2)/$3}' 2>/dev/null)
        printf "  %-12s | acc=%-8s | time=%-8s | tps=%s\n" \
            "$METHOD" "$acc" "${time}s" "$tps"
    done
}

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA-Instruct" 4
fi

if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream-Instruct" 8
fi