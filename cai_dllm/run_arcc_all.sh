#!/bin/bash
# =============================================================================
# run_arcc_all.sh  —  ARC-C evaluation for LLaDA and Dream
# Fills table: Benchmark | Method | Accuracy | Time | Tok/s | Requests | Speedup
#
# Usage:
#   bash run_arcc_all.sh 0            # both models on GPU 0
#   bash run_arcc_all.sh 0 llada      # LLaDA only
#   bash run_arcc_all.sh 0 dream      # Dream only
# =============================================================================

GPU=${1:-0}
MODEL_FILTER=${2:-"both"}
LOG_DIR="log_results/arcc_cai"
mkdir -p $LOG_DIR
DATE=$(date +%s)
RESULTS_FILE="$LOG_DIR/summary_${DATE}.txt"

echo "================================================" | tee $RESULTS_FILE
echo " ARC-C CAI-dLLM Evaluation"                       | tee -a $RESULTS_FILE
echo " GPU=$GPU  DATE=$(date)"                           | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE

# ─── extract metrics from log ────────────────────────────────────────────────
extract_metrics() {
    local log=$1
    local time=$(grep "Total generation time" $log | awk '{print $4}' | tr -d 's')
    local requests=$(grep "Request count" $log | awk '{print $3}')
    local eff_tokens=$(grep "Effective tokens per request" $log | awk '{print $5}')
    local accuracy=$(grep -oP '\d+\.\d+' <(grep "acc_norm\|acc " $log | tail -1) | head -1)
    local tps=""
    if [ -n "$time" ] && [ -n "$requests" ] && [ -n "$eff_tokens" ]; then
        tps=$(echo "$requests $eff_tokens $time" | awk '{printf "%.1f", ($1 * $2) / $3}')
    fi
    echo "$time|$requests|$eff_tokens|$accuracy|$tps"
}

print_row() {
    local model=$1; local task=$2; local method=$3
    local log=$4;   local baseline_time=$5
    if [ ! -f "$log" ]; then
        echo "  [$model | $task | $method] log not found" | tee -a $RESULTS_FILE
        return
    fi
    IFS='|' read -r time requests eff_tokens accuracy tps <<< "$(extract_metrics $log)"
    local speedup=""
    if [ -n "$baseline_time" ] && [ -n "$time" ]; then
        speedup=$(echo "$baseline_time $time" | awk '{printf "%.2f", $1 / $2}')
    fi
    printf "  %-8s | %-22s | acc=%-8s | time=%-8s | tps=%-8s | req=%-6s | speedup=%sx\n" \
        "$task" "$method" "$accuracy" "${time}s" "$tps" "$requests" "$speedup" \
        | tee -a $RESULTS_FILE
}

# =============================================================================
# LLaDA-8B-Instruct — ARC-C
# Note: ARC-C is a multiple-choice task (likelihood-based), not generation.
# It uses arc_challenge in lm-eval.
# =============================================================================
run_llada() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " LLaDA-8B-Instruct — ARC-C"
    echo "──────────────────────────────────────────────────"

    # ── 1. Nocache vanilla (LLaDA baseline) ──────────────────────────────
    echo ""
    echo "[LLaDA 1/4] Nocache vanilla (LLaDA baseline)..."
    rm -rf lm_cache/
    LOG_LLADA_NOCACHE=$LOG_DIR/llada_arcc_nocache_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task arc_challenge \
        --esdllm_mode nocache \
        2>&1 | tee $LOG_LLADA_NOCACHE
    LLADA_BASE_TIME=$(grep "Total generation time" $LOG_LLADA_NOCACHE | awk '{print $4}' | tr -d 's')
    echo "[LLaDA 1/4] Done. Time=${LLADA_BASE_TIME}s"

    # ── 2. DualCache ──────────────────────────────────────────────────────
    echo ""
    echo "[LLaDA 2/4] DualCache..."
    rm -rf lm_cache/
    LOG_LLADA_DC=$LOG_DIR/llada_arcc_dualcache_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task arc_challenge \
        --esdllm_mode DualCache \
        2>&1 | tee $LOG_LLADA_DC
    echo "[LLaDA 2/4] Done."

    # ── 3. ES-dLLM ───────────────────────────────────────────────────────
    echo ""
    echo "[LLaDA 3/4] ES-dLLM (HiddenState)..."
    rm -rf lm_cache/
    LOG_LLADA_ES=$LOG_DIR/llada_arcc_esdllm_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task arc_challenge \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 16 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        2>&1 | tee $LOG_LLADA_ES
    echo "[LLaDA 3/4] Done."

    # ── 4. CAI-dLLM (best config) ─────────────────────────────────────────
    echo ""
    echo "[LLaDA 4/4] CAI-dLLM (apd_per_block + confidence gating)..."
    rm -rf lm_cache/
    LOG_LLADA_CAI=$LOG_DIR/llada_arcc_cai_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task arc_challenge \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 16 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        --use_cai \
        --cai_mode apd_per_block \
        2>&1 | tee $LOG_LLADA_CAI
    echo "[LLaDA 4/4] Done."

    echo ""
    echo "=== LLaDA ARC-C Results ===" | tee -a $RESULTS_FILE
    print_row "LLaDA" "ARC-C" "Nocache (vanilla)" $LOG_LLADA_NOCACHE ""
    print_row "LLaDA" "ARC-C" "DualCache"          $LOG_LLADA_DC      $LLADA_BASE_TIME
    print_row "LLaDA" "ARC-C" "ES-dLLM"            $LOG_LLADA_ES      $LLADA_BASE_TIME
    print_row "LLaDA" "ARC-C" "CAI-dLLM"           $LOG_LLADA_CAI     $LLADA_BASE_TIME
}

# =============================================================================
# Dream-7B-Instruct — ARC-C
# =============================================================================
run_dream() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " Dream-7B-Instruct — ARC-C"
    echo "──────────────────────────────────────────────────"

    # ── 1. Nocache vanilla (Dream baseline) ──────────────────────────────
    echo ""
    echo "[Dream 1/4] Nocache vanilla (Dream baseline)..."
    rm -rf lm_cache/
    LOG_DREAM_NOCACHE=$LOG_DIR/dream_arcc_nocache_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task arc_challenge \
        --esdllm_mode nocache \
        2>&1 | tee $LOG_DREAM_NOCACHE
    DREAM_BASE_TIME=$(grep "Total generation time" $LOG_DREAM_NOCACHE | awk '{print $4}' | tr -d 's')
    echo "[Dream 1/4] Done. Time=${DREAM_BASE_TIME}s"

    # ── 2. DualCache ──────────────────────────────────────────────────────
    echo ""
    echo "[Dream 2/4] DualCache..."
    rm -rf lm_cache/
    LOG_DREAM_DC=$LOG_DIR/dream_arcc_dualcache_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task arc_challenge \
        --esdllm_mode DualCache \
        2>&1 | tee $LOG_DREAM_DC
    echo "[Dream 2/4] Done."

    # ── 3. ES-dLLM ───────────────────────────────────────────────────────
    echo ""
    echo "[Dream 3/4] ES-dLLM (HiddenState)..."
    rm -rf lm_cache/
    LOG_DREAM_ES=$LOG_DIR/dream_arcc_esdllm_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task arc_challenge \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 8 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        2>&1 | tee $LOG_DREAM_ES
    echo "[Dream 3/4] Done."

    # ── 4. CAI-dLLM (best config) ─────────────────────────────────────────
    echo ""
    echo "[Dream 4/4] CAI-dLLM (apd_per_block + confidence gating)..."
    rm -rf lm_cache/
    LOG_DREAM_CAI=$LOG_DIR/dream_arcc_cai_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task arc_challenge \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 8 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        --use_cai \
        --cai_mode apd_per_block \
        2>&1 | tee $LOG_DREAM_CAI
    echo "[Dream 4/4] Done."

    echo ""
    echo "=== Dream ARC-C Results ===" | tee -a $RESULTS_FILE
    print_row "Dream" "ARC-C" "Nocache (vanilla)" $LOG_DREAM_NOCACHE ""
    print_row "Dream" "ARC-C" "DualCache"          $LOG_DREAM_DC      $DREAM_BASE_TIME
    print_row "Dream" "ARC-C" "ES-dLLM"            $LOG_DREAM_ES      $DREAM_BASE_TIME
    print_row "Dream" "ARC-C" "CAI-dLLM"           $LOG_DREAM_CAI     $DREAM_BASE_TIME
}

# =============================================================================
# Run
# =============================================================================
if [ "$MODEL_FILTER" == "llada" ]; then
    run_llada
elif [ "$MODEL_FILTER" == "dream" ]; then
    run_dream
else
    run_llada
    run_dream
fi

echo ""
echo "================================================" | tee -a $RESULTS_FILE
echo " All done. Summary: $RESULTS_FILE"               | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE
cat $RESULTS_FILE