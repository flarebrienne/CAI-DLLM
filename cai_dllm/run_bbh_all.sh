#!/bin/bash
# =============================================================================
# run_bbh_all.sh  —  BBH evaluation for LLaDA and Dream
# Fills table: Benchmark | Method | Accuracy | Time | Tok/s | Requests | Speedup
#
# Usage:
#   bash run_bbh_all.sh 0            # both models on GPU 0
#   bash run_bbh_all.sh 0 llada      # LLaDA only
#   bash run_bbh_all.sh 0 dream      # Dream only
# =============================================================================

GPU=${1:-0}
MODEL_FILTER=${2:-"both"}
LOG_DIR="log_results/bbh_cai"
mkdir -p $LOG_DIR
DATE=$(date +%s)
RESULTS_FILE="$LOG_DIR/summary_${DATE}.txt"

echo "================================================" | tee $RESULTS_FILE
echo " BBH CAI-dLLM Evaluation"                         | tee -a $RESULTS_FILE
echo " GPU=$GPU  DATE=$(date)"                           | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE

# ─── extract metrics ─────────────────────────────────────────────────────────
extract_metrics() {
    local log=$1
    local time=$(grep "Total generation time" $log | awk '{print $4}' | tr -d 's')
    local requests=$(grep "Request count" $log | awk '{print $3}')
    local eff_tokens=$(grep "Effective tokens per request" $log | awk '{print $5}')
    local accuracy=$(grep -oP '\d+\.\d+' <(grep "exact_match\|acc" $log | tail -2 | head -1) | head -1)
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
    printf "  %-8s | %-26s | acc=%-8s | time=%-8s | tps=%-8s | req=%-6s | speedup=%sx\n" \
        "$task" "$method" "$accuracy" "${time}s" "$tps" "$requests" "$speedup" \
        | tee -a $RESULTS_FILE
}

# =============================================================================
# LLaDA-8B-Instruct — BBH
# =============================================================================
run_llada() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " LLaDA-8B-Instruct — BBH"
    echo "──────────────────────────────────────────────────"

    # ── 1. ES-dLLM baseline ───────────────────────────────────────────────
    echo ""
    echo "[LLaDA 1/4] ES-dLLM baseline..."
    rm -rf lm_cache/
    LOG_LLADA_BASE=$LOG_DIR/llada_bbh_esdllm_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task bbh \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 4 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        2>&1 | tee $LOG_LLADA_BASE
    LLADA_BASE_TIME=$(grep "Total generation time" $LOG_LLADA_BASE | awk '{print $4}' | tr -d 's')
    echo "[LLaDA 1/4] Done. Time=${LLADA_BASE_TIME}s"

    # ── 2. CAI APD only ───────────────────────────────────────────────────
    echo ""
    echo "[LLaDA 2/4] CAI-APD only..."
    rm -rf lm_cache/
    LOG_LLADA_APD=$LOG_DIR/llada_bbh_cai_apd_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task bbh \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 4 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        --use_cai \
        --cai_mode apd_only \
        2>&1 | tee $LOG_LLADA_APD
    echo "[LLaDA 2/4] Done."

    # ── 3. CAI per-block ──────────────────────────────────────────────────
    echo ""
    echo "[LLaDA 3/4] CAI per-block..."
    rm -rf lm_cache/
    LOG_LLADA_PB=$LOG_DIR/llada_bbh_cai_perblock_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task bbh \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 4 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        --use_cai \
        --cai_mode apd_per_block \
        2>&1 | tee $LOG_LLADA_PB
    echo "[LLaDA 3/4] Done."

    # ── 4. CAI confidence gating (all) ────────────────────────────────────
    echo ""
    echo "[LLaDA 4/4] CAI confidence gating (all)..."
    rm -rf lm_cache/
    LOG_LLADA_CG=$LOG_DIR/llada_bbh_cai_full_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task bbh \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 4 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        --use_cai \
        --cai_mode apd_per_block \
        2>&1 | tee $LOG_LLADA_CG
    echo "[LLaDA 4/4] Done."

    echo ""
    echo "=== LLaDA BBH Results ===" | tee -a $RESULTS_FILE
    print_row "LLaDA" "BBH" "ES-dLLM (baseline)"    $LOG_LLADA_BASE ""
    print_row "LLaDA" "BBH" "CAI-APD only"           $LOG_LLADA_APD  $LLADA_BASE_TIME
    print_row "LLaDA" "BBH" "CAI per-block"          $LOG_LLADA_PB   $LLADA_BASE_TIME
    print_row "LLaDA" "BBH" "CAI confidence gating"  $LOG_LLADA_CG   $LLADA_BASE_TIME
}

# =============================================================================
# Dream-7B-Instruct — BBH
# =============================================================================
run_dream() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " Dream-7B-Instruct — BBH"
    echo "──────────────────────────────────────────────────"

    # ── 1. DualCache baseline ─────────────────────────────────────────────
    echo ""
    echo "[Dream 1/4] DualCache baseline..."
    rm -rf lm_cache/
    LOG_DREAM_BASE=$LOG_DIR/dream_bbh_dualcache_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task bbh \
        --esdllm_mode DualCache \
        2>&1 | tee $LOG_DREAM_BASE
    DREAM_BASE_TIME=$(grep "Total generation time" $LOG_DREAM_BASE | awk '{print $4}' | tr -d 's')
    echo "[Dream 1/4] Done. Time=${DREAM_BASE_TIME}s"

    # ── 2. CAI APD only ───────────────────────────────────────────────────
    echo ""
    echo "[Dream 2/4] CAI-APD only..."
    rm -rf lm_cache/
    LOG_DREAM_APD=$LOG_DIR/dream_bbh_cai_apd_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task bbh \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 8 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        --use_cai \
        --cai_mode apd_only \
        2>&1 | tee $LOG_DREAM_APD
    echo "[Dream 2/4] Done."

    # ── 3. CAI per-block ──────────────────────────────────────────────────
    echo ""
    echo "[Dream 3/4] CAI per-block..."
    rm -rf lm_cache/
    LOG_DREAM_PB=$LOG_DIR/dream_bbh_cai_perblock_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task bbh \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 8 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        --use_cai \
        --cai_mode apd_per_block \
        2>&1 | tee $LOG_DREAM_PB
    echo "[Dream 3/4] Done."

    # ── 4. CAI confidence gating (all) ────────────────────────────────────
    echo ""
    echo "[Dream 4/4] CAI confidence gating (all)..."
    rm -rf lm_cache/
    LOG_DREAM_CG=$LOG_DIR/dream_bbh_cai_full_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task bbh \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 8 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        --use_cai \
        --cai_mode apd_per_block \
        2>&1 | tee $LOG_DREAM_CG
    echo "[Dream 4/4] Done."

    echo ""
    echo "=== Dream BBH Results ===" | tee -a $RESULTS_FILE
    print_row "Dream" "BBH" "DualCache (baseline)"   $LOG_DREAM_BASE ""
    print_row "Dream" "BBH" "CAI-APD only"           $LOG_DREAM_APD  $DREAM_BASE_TIME
    print_row "Dream" "BBH" "CAI per-block"          $LOG_DREAM_PB   $DREAM_BASE_TIME
    print_row "Dream" "BBH" "CAI confidence gating"  $LOG_DREAM_CG   $DREAM_BASE_TIME
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