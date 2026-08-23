#!/bin/bash
# =============================================================================
# run_error_budget_gsm8k.sh  —  Error Budget ablation on LLaDA-8B GSM8K
# Tests 5 methods: nocache, dualcache, esdllm, caidllm, caidllm+error_budget
# Usage: bash run_error_budget_gsm8k.sh 0
# =============================================================================

GPU=${1:-0}
LOG_DIR="log_results/error_budget"
mkdir -p $LOG_DIR
DATE=$(date +%s)
SUMMARY="$LOG_DIR/summary_${DATE}.txt"

echo "================================================" | tee $SUMMARY
echo " Error Budget Ablation — LLaDA-8B GSM8K"        | tee -a $SUMMARY
echo " GPU=$GPU  DATE=$(date)"                          | tee -a $SUMMARY
echo "================================================" | tee -a $SUMMARY

extract_metrics() {
    local log=$1
    local time=$(grep "Total generation time" $log | awk '{print $4}' | tr -d 's')
    local req=$(grep "Request count" $log | awk '{print $3}')
    local eff=$(grep "Effective tokens per request" $log | awk '{print $5}')
    local acc=$(grep -oP '\d+\.\d+' <(grep "flexible-extract" $log | tail -1) | head -1)
    local tps=""
    if [ -n "$time" ] && [ -n "$req" ] && [ -n "$eff" ]; then
        tps=$(echo "$req $eff $time" | awk '{printf "%.1f", ($1*$2)/$3}')
    fi
    echo "$time|$req|$eff|$acc|$tps"
}

print_row() {
    local method=$1; local log=$2; local baseline_time=$3
    [ ! -f "$log" ] && echo "  $method — log not found" | tee -a $SUMMARY && return
    IFS='|' read -r time req eff acc tps <<< "$(extract_metrics $log)"
    local spd=""
    [ -n "$baseline_time" ] && [ -n "$time" ] && \
        spd=$(echo "$baseline_time $time" | awk '{printf "%.2f", $1/$2}')
    printf "  %-28s | acc=%-8s | time=%-8s | tps=%-8s | speedup=%sx\n" \
        "$method" "$acc" "${time}s" "$tps" "$spd" | tee -a $SUMMARY
}

# 1. Nocache
echo ""; echo "[1/5] nocache..."
rm -rf lm_cache/
CUDA_VISIBLE_DEVICES=$GPU python eval2.py \
    --model LLaDA-Instruct --task gsm8k --esdllm_mode nocache \
    2>&1 | tee $LOG_DIR/llada_gsm8k_nocache_${DATE}.log
NOCACHE_TIME=$(grep "Total generation time" $LOG_DIR/llada_gsm8k_nocache_${DATE}.log | awk '{print $4}' | tr -d 's')
echo "[1/5] Done. time=${NOCACHE_TIME}s"

# 2. DualCache
echo ""; echo "[2/5] dualcache..."
rm -rf lm_cache/
CUDA_VISIBLE_DEVICES=$GPU python eval2.py \
    --model LLaDA-Instruct --task gsm8k --esdllm_mode DualCache \
    2>&1 | tee $LOG_DIR/llada_gsm8k_dualcache_${DATE}.log
echo "[2/5] Done."

# 3. ES-dLLM
echo ""; echo "[3/5] esdllm..."
rm -rf lm_cache/
CUDA_VISIBLE_DEVICES=$GPU python eval2.py \
    --model LLaDA-Instruct --task gsm8k \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 16 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
    2>&1 | tee $LOG_DIR/llada_gsm8k_esdllm_${DATE}.log
echo "[3/5] Done."

# 4. CAI-dLLM (no error budget)
echo ""; echo "[4/5] caidllm..."
rm -rf lm_cache/
CUDA_VISIBLE_DEVICES=$GPU python eval2.py \
    --model LLaDA-Instruct --task gsm8k \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 16 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
    --use_cai --cai_mode apd_per_block \
    2>&1 | tee $LOG_DIR/llada_gsm8k_cai_${DATE}.log
echo "[4/5] Done."

# 5. CAI-dLLM + Error Budget
echo ""; echo "[5/5] caidllm + error budget..."
rm -rf lm_cache/
CUDA_VISIBLE_DEVICES=$GPU python eval2.py \
    --model LLaDA-Instruct --task gsm8k \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 16 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
    --use_cai --cai_mode apd_per_block \
    --use_error_budget \
    2>&1 | tee $LOG_DIR/llada_gsm8k_cai_eb_${DATE}.log
echo "[5/5] Done."

echo ""; echo "=== RESULTS ===" | tee -a $SUMMARY
print_row "Nocache (vanilla)"        $LOG_DIR/llada_gsm8k_nocache_${DATE}.log   ""
print_row "DualCache"                $LOG_DIR/llada_gsm8k_dualcache_${DATE}.log $NOCACHE_TIME
print_row "ES-dLLM"                  $LOG_DIR/llada_gsm8k_esdllm_${DATE}.log    $NOCACHE_TIME
print_row "CAI-dLLM"                 $LOG_DIR/llada_gsm8k_cai_${DATE}.log       $NOCACHE_TIME
print_row "CAI-dLLM + Error Budget"  $LOG_DIR/llada_gsm8k_cai_eb_${DATE}.log    $NOCACHE_TIME

echo "" && echo "Done. Summary: $SUMMARY" | tee -a $SUMMARY