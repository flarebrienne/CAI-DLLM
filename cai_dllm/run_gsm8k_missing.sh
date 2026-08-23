#!/bin/bash
# =============================================================================
# run_gsm8k_missing.sh  —  Fill missing LLaDA GSM8K baseline rows
# Runs LLaDA Nocache and DualCache which were not run before.
# Dream GSM8K is already complete from previous runs.
#
# Usage:
#   bash run_gsm8k_missing.sh 0
# =============================================================================

GPU=${1:-0}
LOG_DIR="log_results/gsm8k_baselines"
mkdir -p $LOG_DIR
DATE=$(date +%s)
RESULTS_FILE="$LOG_DIR/summary_${DATE}.txt"

echo "================================================" | tee $RESULTS_FILE
echo " GSM8K Missing Baselines"                         | tee -a $RESULTS_FILE
echo " GPU=$GPU  DATE=$(date)"                           | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE

# ─── extract metrics ─────────────────────────────────────────────────────────
extract_metrics() {
    local log=$1
    local time=$(grep "Total generation time" $log | awk '{print $4}' | tr -d 's')
    local requests=$(grep "Request count" $log | awk '{print $3}')
    local eff_tokens=$(grep "Effective tokens per request" $log | awk '{print $5}')
    local accuracy=$(grep -oP '\d+\.\d+' <(grep "flexible-extract" $log | tail -1) | head -1)
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
    printf "  %-12s | %-22s | acc=%-8s | time=%-8s | tps=%-8s | req=%-6s | speedup=%sx\n" \
        "$model" "$method" "$accuracy" "${time}s" "$tps" "$requests" "$speedup" \
        | tee -a $RESULTS_FILE
}

# ── LLaDA Nocache (vanilla) ───────────────────────────────────────────────────
# WARNING: This will take ~4-5 hours (same as Dream nocache)
# Skip this if you want to estimate from Dream's nocache time (~16000s)
echo ""
echo "[1/2] LLaDA Nocache (vanilla) — WARNING: ~4-5 hours..."
echo "      Press Ctrl+C within 10s to skip and only run DualCache"
sleep 10
rm -rf lm_cache/
LOG_LLADA_NOCACHE=$LOG_DIR/llada_gsm8k_nocache_${DATE}.log
CUDA_VISIBLE_DEVICES=$GPU python eval.py \
    --model LLaDA-Instruct \
    --task gsm8k \
    --esdllm_mode nocache \
    2>&1 | tee $LOG_LLADA_NOCACHE
LLADA_NOCACHE_TIME=$(grep "Total generation time" $LOG_LLADA_NOCACHE | awk '{print $4}' | tr -d 's')
echo "[1/2] Done. Time=${LLADA_NOCACHE_TIME}s"

# ── LLaDA DualCache ───────────────────────────────────────────────────────────
echo ""
echo "[2/2] LLaDA DualCache..."
rm -rf lm_cache/
LOG_LLADA_DC=$LOG_DIR/llada_gsm8k_dualcache_${DATE}.log
CUDA_VISIBLE_DEVICES=$GPU python eval.py \
    --model LLaDA-Instruct \
    --task gsm8k \
    --esdllm_mode DualCache \
    2>&1 | tee $LOG_LLADA_DC
echo "[2/2] Done."

echo ""
echo "=== Results ===" | tee -a $RESULTS_FILE
print_row "LLaDA" "Nocache (vanilla)" $LOG_LLADA_NOCACHE ""
print_row "LLaDA" "DualCache"         $LOG_LLADA_DC      $LLADA_NOCACHE_TIME

echo ""
echo "================================================"         | tee -a $RESULTS_FILE
echo " Previously measured (from earlier runs):"                | tee -a $RESULTS_FILE
echo "  LLaDA  | ES-dLLM   | acc=77.10% | time=1728s | tps=175.3 | req=1319 | speedup vs nocache=TBD" | tee -a $RESULTS_FILE
echo "  LLaDA  | CAI-dLLM  | acc=77.41% | time=1006s | tps=335.1 | req=1319 | speedup vs nocache=TBD" | tee -a $RESULTS_FILE
echo "================================================"         | tee -a $RESULTS_FILE
echo " Dream GSM8K (all previously measured):"                  | tee -a $RESULTS_FILE
echo "  Dream  | Nocache   | acc=79.68% | time=16204s | tps=11.6  | req=1319 | speedup=1.00x"  | tee -a $RESULTS_FILE
echo "  Dream  | DualCache | acc=78.47% | time=1749s  | tps=103.2 | req=1319 | speedup=9.3x"   | tee -a $RESULTS_FILE
echo "  Dream  | ES-dLLM   | acc=77.48% | time=1811s  | tps=97.9  | req=1319 | speedup=8.9x"   | tee -a $RESULTS_FILE
echo "  Dream  | CAI-dLLM  | acc=75.51% | time=881s   | tps=383.0 | req=1319 | speedup=18.4x"  | tee -a $RESULTS_FILE
echo "================================================"         | tee -a $RESULTS_FILE
cat $RESULTS_FILE