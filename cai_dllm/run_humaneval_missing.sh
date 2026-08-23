#!/bin/bash
# =============================================================================
# run_humaneval_missing.sh  —  Fill missing HumanEval baseline rows
#
# Already measured (skip these):
#   LLaDA ES-dLLM  : 35.98% / 379s
#   LLaDA CAI-dLLM : 35.98% / 192s
#   Dream DualCache: 45.12% / 347s
#   Dream CAI-dLLM : 46.34% / 136s
#
# Still needed:
#   LLaDA Nocache    [~2-3 hours]
#   LLaDA DualCache  [~20 min]
#   Dream Nocache    [~2-3 hours]
#   Dream ES-dLLM    [~20 min]
#
# Usage:
#   bash run_humaneval_missing.sh 0          # all missing runs
#   bash run_humaneval_missing.sh 0 llada    # LLaDA missing only
#   bash run_humaneval_missing.sh 0 dream    # Dream missing only
#   bash run_humaneval_missing.sh 0 fast     # skip nocache (only DualCache + ES-dLLM, ~40 min total)
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"all"}
LOG_DIR="log_results/humaneval_missing"
mkdir -p $LOG_DIR
DATE=$(date +%s)
RESULTS_FILE="$LOG_DIR/summary_${DATE}.txt"

echo "================================================" | tee $RESULTS_FILE
echo " HumanEval Missing Baselines"                      | tee -a $RESULTS_FILE
echo " GPU=$GPU  FILTER=$FILTER  DATE=$(date)"           | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE

# ─── extract metrics ─────────────────────────────────────────────────────────
extract_metrics() {
    local log=$1
    local time=$(grep "Total generation time" $log | awk '{print $4}' | tr -d 's')
    local requests=$(grep "Request count" $log | awk '{print $3}')
    local eff_tokens=$(grep "Effective tokens per request" $log | awk '{print $5}')
    local accuracy=$(grep -oP '\d+\.\d+' <(grep "pass@1" $log | tail -1) | head -1)
    local tps=""
    if [ -n "$time" ] && [ -n "$requests" ] && [ -n "$eff_tokens" ]; then
        tps=$(echo "$requests $eff_tokens $time" | awk '{printf "%.1f", ($1 * $2) / $3}')
    fi
    echo "$time|$requests|$eff_tokens|$accuracy|$tps"
}

print_row() {
    local model=$1; local method=$2; local log=$3; local baseline_time=$4
    if [ ! -f "$log" ]; then
        echo "  [$model | $method] log not found" | tee -a $RESULTS_FILE
        return
    fi
    IFS='|' read -r time requests eff_tokens accuracy tps <<< "$(extract_metrics $log)"
    local speedup=""
    if [ -n "$baseline_time" ] && [ -n "$time" ]; then
        speedup=$(echo "$baseline_time $time" | awk '{printf "%.2f", $1 / $2}')
    fi
    printf "  %-10s | %-22s | acc=%-8s | time=%-8s | tps=%-8s | req=%-5s | speedup=%sx\n" \
        "$model" "$method" "$accuracy" "${time}s" "$tps" "$requests" "$speedup" \
        | tee -a $RESULTS_FILE
}

# =============================================================================
# LLaDA missing runs
# =============================================================================
run_llada_missing() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " LLaDA-8B-Instruct — HumanEval missing baselines"
    echo "──────────────────────────────────────────────────"

    LOG_LLADA_NOCACHE=$LOG_DIR/llada_he_nocache_${DATE}.log
    LOG_LLADA_DC=$LOG_DIR/llada_he_dualcache_${DATE}.log

    # ── Nocache (skip if FILTER=fast) ────────────────────────────────────
    if [ "$FILTER" != "fast" ]; then
        echo ""
        echo "[LLaDA 1/2] Nocache vanilla — WARNING: ~2-3 hours..."
        rm -rf lm_cache/
        CUDA_VISIBLE_DEVICES=$GPU python eval.py \
            --model LLaDA-Instruct \
            --task humaneval \
            --esdllm_mode nocache \
            2>&1 | tee $LOG_LLADA_NOCACHE
        LLADA_NOCACHE_TIME=$(grep "Total generation time" $LOG_LLADA_NOCACHE | awk '{print $4}' | tr -d 's')
        echo "[LLaDA 1/2] Done. Time=${LLADA_NOCACHE_TIME}s"
    else
        echo "[LLaDA 1/2] Skipping nocache (fast mode)"
        LLADA_NOCACHE_TIME=""
    fi

    # ── DualCache ─────────────────────────────────────────────────────────
    echo ""
    echo "[LLaDA 2/2] DualCache..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task humaneval \
        --esdllm_mode DualCache \
        2>&1 | tee $LOG_LLADA_DC
    echo "[LLaDA 2/2] Done."

    echo ""
    echo "=== LLaDA HumanEval New Results ===" | tee -a $RESULTS_FILE
    [ "$FILTER" != "fast" ] && print_row "LLaDA" "Nocache (vanilla)" $LOG_LLADA_NOCACHE ""
    print_row "LLaDA" "DualCache" $LOG_LLADA_DC "$LLADA_NOCACHE_TIME"
}

# =============================================================================
# Dream missing runs
# =============================================================================
run_dream_missing() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " Dream-7B-Instruct — HumanEval missing baselines"
    echo "──────────────────────────────────────────────────"

    LOG_DREAM_NOCACHE=$LOG_DIR/dream_he_nocache_${DATE}.log
    LOG_DREAM_ES=$LOG_DIR/dream_he_esdllm_${DATE}.log

    # ── Nocache (skip if FILTER=fast) ────────────────────────────────────
    if [ "$FILTER" != "fast" ]; then
        echo ""
        echo "[Dream 1/2] Nocache vanilla — WARNING: ~2-3 hours..."
        rm -rf lm_cache/
        CUDA_VISIBLE_DEVICES=$GPU python eval.py \
            --model Dream-Instruct \
            --task humaneval \
            --esdllm_mode nocache \
            2>&1 | tee $LOG_DREAM_NOCACHE
        DREAM_NOCACHE_TIME=$(grep "Total generation time" $LOG_DREAM_NOCACHE | awk '{print $4}' | tr -d 's')
        echo "[Dream 1/2] Done. Time=${DREAM_NOCACHE_TIME}s"
    else
        echo "[Dream 1/2] Skipping nocache (fast mode)"
        DREAM_NOCACHE_TIME=""
    fi

    # ── ES-dLLM ───────────────────────────────────────────────────────────
    echo ""
    echo "[Dream 2/2] ES-dLLM (HiddenState)..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task humaneval \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 8 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        2>&1 | tee $LOG_DREAM_ES
    echo "[Dream 2/2] Done."

    echo ""
    echo "=== Dream HumanEval New Results ===" | tee -a $RESULTS_FILE
    [ "$FILTER" != "fast" ] && print_row "Dream" "Nocache (vanilla)" $LOG_DREAM_NOCACHE ""
    print_row "Dream" "ES-dLLM" $LOG_DREAM_ES "$DREAM_NOCACHE_TIME"
}

# =============================================================================
# Run based on filter
# =============================================================================
if [ "$FILTER" == "llada" ]; then
    run_llada_missing
elif [ "$FILTER" == "dream" ]; then
    run_dream_missing
else
    run_llada_missing
    run_dream_missing
fi

# ── Print full combined summary ───────────────────────────────────────────────
echo ""
echo "================================================"          | tee -a $RESULTS_FILE
echo " FULL HumanEval Summary (all runs combined)"               | tee -a $RESULTS_FILE
echo "================================================"          | tee -a $RESULTS_FILE
echo " LLaDA HumanEval:"                                         | tee -a $RESULTS_FILE
echo "  Nocache    : FROM THIS RUN (see above)"                  | tee -a $RESULTS_FILE
echo "  DualCache  : FROM THIS RUN (see above)"                  | tee -a $RESULTS_FILE
echo "  ES-dLLM    : acc=35.98% | time=379s  | tps=139.8 | req=164 | speedup vs nocache=TBD" | tee -a $RESULTS_FILE
echo "  CAI-dLLM   : acc=35.98% | time=192s  | tps=435.3 | req=164 | speedup vs nocache=TBD" | tee -a $RESULTS_FILE
echo ""                                                           | tee -a $RESULTS_FILE
echo " Dream HumanEval:"                                         | tee -a $RESULTS_FILE
echo "  Nocache    : FROM THIS RUN (see above)"                  | tee -a $RESULTS_FILE
echo "  DualCache  : acc=45.12% | time=347s  | tps=72.5  | req=164 | speedup vs nocache=TBD" | tee -a $RESULTS_FILE
echo "  ES-dLLM    : FROM THIS RUN (see above)"                  | tee -a $RESULTS_FILE
echo "  CAI-dLLM   : acc=46.34% | time=136s  | tps=614.6 | req=164 | speedup vs nocache=TBD" | tee -a $RESULTS_FILE
echo "================================================"          | tee -a $RESULTS_FILE