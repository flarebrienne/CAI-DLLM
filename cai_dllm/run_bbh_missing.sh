#!/bin/bash
# =============================================================================
# run_bbh_missing.sh  —  Fill missing BBH rows for Dream and LLaDA
#
# Already measured (skip these):
#   LLaDA ES-dLLM  : 54.11% / 6979s
#   LLaDA CAI-dLLM : 51.33% / 3355s
#
# Still needed:
#   LLaDA Nocache    [~8-10 hours — WARNING]
#   LLaDA DualCache  [~2-3 hours]
#   Dream Nocache    [~8-10 hours — WARNING]
#   Dream DualCache  [~2-3 hours]
#   Dream ES-dLLM    [~2-3 hours]
#   Dream CAI-dLLM   [~1.5 hours]
#
# Usage:
#   bash run_bbh_missing.sh 0          # all missing runs (~20+ hours)
#   bash run_bbh_missing.sh 0 fast     # skip nocache (~8-9 hours total)
#   bash run_bbh_missing.sh 0 llada    # LLaDA missing only
#   bash run_bbh_missing.sh 0 dream    # Dream missing only
#   bash run_bbh_missing.sh 0 dream_fast  # Dream only, skip nocache
#   bash run_bbh_missing.sh 0 llada_fast  # LLaDA only, skip nocache
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"all"}
LOG_DIR="log_results/bbh_missing"
mkdir -p $LOG_DIR
DATE=$(date +%s)
RESULTS_FILE="$LOG_DIR/summary_${DATE}.txt"

echo "================================================" | tee $RESULTS_FILE
echo " BBH Missing Baselines"                            | tee -a $RESULTS_FILE
echo " GPU=$GPU  FILTER=$FILTER  DATE=$(date)"           | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE

# ─── extract metrics ─────────────────────────────────────────────────────────
extract_metrics() {
    local log=$1
    local time=$(grep "Total generation time" $log | awk '{print $4}' | tr -d 's')
    local requests=$(grep "Request count" $log | awk '{print $3}')
    local eff_tokens=$(grep "Effective tokens per request" $log | awk '{print $5}')
    local accuracy=$(grep "^|bbh " $log | grep -oP '\d+\.\d+' | head -1)
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
    printf "  %-10s | %-22s | acc=%-8s | time=%-8s | tps=%-8s | req=%-6s | speedup=%sx\n" \
        "$model" "$method" "$accuracy" "${time}s" "$tps" "$requests" "$speedup" \
        | tee -a $RESULTS_FILE
}

skip_nocache() {
    [[ "$FILTER" == *"fast"* ]]
}

# =============================================================================
# LLaDA missing runs
# =============================================================================
run_llada_missing() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " LLaDA-8B-Instruct — BBH missing baselines"
    echo "──────────────────────────────────────────────────"

    LOG_LLADA_NOCACHE=$LOG_DIR/llada_bbh_nocache_${DATE}.log
    LOG_LLADA_DC=$LOG_DIR/llada_bbh_dualcache_${DATE}.log

    # ── Nocache ───────────────────────────────────────────────────────────
    if ! skip_nocache; then
        echo ""
        echo "[LLaDA 1/2] Nocache vanilla — WARNING: ~8-10 hours..."
        rm -rf lm_cache/
        CUDA_VISIBLE_DEVICES=$GPU python eval.py \
            --model LLaDA-Instruct \
            --task bbh \
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
        --task bbh \
        --esdllm_mode DualCache \
        2>&1 | tee $LOG_LLADA_DC
    echo "[LLaDA 2/2] Done."

    echo ""
    echo "=== LLaDA BBH New Results ===" | tee -a $RESULTS_FILE
    ! skip_nocache && print_row "LLaDA" "Nocache (vanilla)" $LOG_LLADA_NOCACHE ""
    print_row "LLaDA" "DualCache" $LOG_LLADA_DC "$LLADA_NOCACHE_TIME"
}

# =============================================================================
# Dream all runs (nothing measured yet)
# =============================================================================
run_dream_all() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " Dream-7B-Instruct — BBH all runs"
    echo "──────────────────────────────────────────────────"

    LOG_DREAM_NOCACHE=$LOG_DIR/dream_bbh_nocache_${DATE}.log
    LOG_DREAM_DC=$LOG_DIR/dream_bbh_dualcache_${DATE}.log
    LOG_DREAM_ES=$LOG_DIR/dream_bbh_esdllm_${DATE}.log
    LOG_DREAM_CAI=$LOG_DIR/dream_bbh_cai_${DATE}.log

    # ── Nocache ───────────────────────────────────────────────────────────
    if ! skip_nocache; then
        echo ""
        echo "[Dream 1/4] Nocache vanilla — WARNING: ~8-10 hours..."
        rm -rf lm_cache/
        CUDA_VISIBLE_DEVICES=$GPU python eval.py \
            --model Dream-Instruct \
            --task bbh \
            --esdllm_mode nocache \
            2>&1 | tee $LOG_DREAM_NOCACHE
        DREAM_NOCACHE_TIME=$(grep "Total generation time" $LOG_DREAM_NOCACHE | awk '{print $4}' | tr -d 's')
        echo "[Dream 1/4] Done. Time=${DREAM_NOCACHE_TIME}s"
    else
        echo "[Dream 1/4] Skipping nocache (fast mode)"
        DREAM_NOCACHE_TIME=""
    fi

    # ── DualCache ─────────────────────────────────────────────────────────
    echo ""
    echo "[Dream 2/4] DualCache..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task bbh \
        --esdllm_mode DualCache \
        2>&1 | tee $LOG_DREAM_DC
    echo "[Dream 2/4] Done."

    # ── ES-dLLM ───────────────────────────────────────────────────────────
    echo ""
    echo "[Dream 3/4] ES-dLLM (HiddenState)..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task bbh \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 8 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        2>&1 | tee $LOG_DREAM_ES
    echo "[Dream 3/4] Done."

    # ── CAI-dLLM ──────────────────────────────────────────────────────────
    echo ""
    echo "[Dream 4/4] CAI-dLLM (apd_per_block + confidence gating)..."
    rm -rf lm_cache/
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
        2>&1 | tee $LOG_DREAM_CAI
    echo "[Dream 4/4] Done."

    echo ""
    echo "=== Dream BBH Results ===" | tee -a $RESULTS_FILE
    ! skip_nocache && print_row "Dream" "Nocache (vanilla)" $LOG_DREAM_NOCACHE ""
    print_row "Dream" "DualCache" $LOG_DREAM_DC "$DREAM_NOCACHE_TIME"
    print_row "Dream" "ES-dLLM"   $LOG_DREAM_ES "$DREAM_NOCACHE_TIME"
    print_row "Dream" "CAI-dLLM"  $LOG_DREAM_CAI "$DREAM_NOCACHE_TIME"
}

# =============================================================================
# Run based on filter
# =============================================================================
case "$FILTER" in
    llada|llada_fast) run_llada_missing ;;
    dream|dream_fast) run_dream_all ;;
    *) run_llada_missing; run_dream_all ;;
esac

# ── Full summary ──────────────────────────────────────────────────────────────
echo ""
echo "================================================"        | tee -a $RESULTS_FILE
echo " FULL BBH Summary (all runs combined)"                   | tee -a $RESULTS_FILE
echo "================================================"        | tee -a $RESULTS_FILE
echo " LLaDA BBH:"                                             | tee -a $RESULTS_FILE
echo "  Nocache   : FROM THIS RUN (see above)"                 | tee -a $RESULTS_FILE
echo "  DualCache : FROM THIS RUN (see above)"                 | tee -a $RESULTS_FILE
echo "  ES-dLLM   : acc=54.11% | time=6979s | tps=168.8 | req=6511 | speedup vs nocache=TBD" | tee -a $RESULTS_FILE
echo "  CAI-dLLM  : acc=51.33% | time=3355s | tps=496.0 | req=6511 | speedup vs nocache=TBD" | tee -a $RESULTS_FILE
echo ""                                                         | tee -a $RESULTS_FILE
echo " Dream BBH:"                                             | tee -a $RESULTS_FILE
echo "  All 4 methods: FROM THIS RUN (see above)"             | tee -a $RESULTS_FILE
echo "================================================"        | tee -a $RESULTS_FILE
cat $RESULTS_FILE