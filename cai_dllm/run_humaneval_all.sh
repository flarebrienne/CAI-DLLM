#!/bin/bash
# =============================================================================
# run_humaneval_all.sh  —  HumanEval evaluation for all CAI modes
# Fills the table: Benchmark | Method | Accuracy | Time | Tok/s | Requests | Speedup
#
# Usage:
#   bash run_humaneval_all.sh 0            # run all on GPU 0
#   bash run_humaneval_all.sh 0 llada      # llada only
#   bash run_humaneval_all.sh 0 dream      # dream only
# =============================================================================

GPU=${1:-0}
MODEL_FILTER=${2:-"both"}   # "llada", "dream", or "both"
LOG_DIR="log_results/humaneval_cai"
mkdir -p $LOG_DIR
DATE=$(date +%s)
RESULTS_FILE="$LOG_DIR/summary_${DATE}.txt"

echo "================================================" | tee $RESULTS_FILE
echo " HumanEval CAI-dLLM Full Evaluation"            | tee -a $RESULTS_FILE
echo " GPU=$GPU  DATE=$(date)"                         | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE

# ─── helper: extract metrics from a log file ─────────────────────────────────
extract_metrics() {
    local log=$1
    local time=$(grep "Total generation time" $log | awk '{print $4}' | tr -d 's')
    local requests=$(grep "Request count" $log | awk '{print $3}')
    local eff_tokens=$(grep "Effective tokens per request" $log | awk '{print $5}')
    local accuracy=$(grep "exact_match" $log | grep "pass" | awk '{print $8}' | head -1)
    # fallback: flexible-extract for humaneval
    if [ -z "$accuracy" ]; then
        accuracy=$(grep "exact_match\|pass@1\|pass_at_1" $log | awk '{print $8}' | head -1)
    fi
    # tokens per second
    local tps=""
    if [ -n "$time" ] && [ -n "$requests" ] && [ -n "$eff_tokens" ] && [ "$time" != "" ]; then
        tps=$(echo "$requests $eff_tokens $time" | awk '{printf "%.1f", ($1 * $2) / $3}')
    fi
    echo "$time|$requests|$eff_tokens|$accuracy|$tps"
}

# ─── helper: print one result row ────────────────────────────────────────────
print_row() {
    local model=$1
    local task=$2
    local method=$3
    local log=$4
    local baseline_time=$5

    if [ ! -f "$log" ]; then
        echo "  [$model | $task | $method] — log not found: $log" | tee -a $RESULTS_FILE
        return
    fi

    IFS='|' read -r time requests eff_tokens accuracy tps <<< "$(extract_metrics $log)"
    local speedup=""
    if [ -n "$baseline_time" ] && [ -n "$time" ] && [ "$baseline_time" != "" ] && [ "$time" != "" ]; then
        speedup=$(echo "$baseline_time $time" | awk '{printf "%.2f", $1 / $2}')
    fi

    printf "  %-12s | %-28s | acc=%-8s | time=%-8s | tps=%-8s | req=%-6s | speedup=%sx\n" \
        "$task" "$method" "$accuracy" "${time}s" "$tps" "$requests" "$speedup" \
        | tee -a $RESULTS_FILE
}

# =============================================================================
# LLaDA-8B-Instruct
# =============================================================================
run_llada() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " LLaDA-8B-Instruct — HumanEval"
    echo "──────────────────────────────────────────────────"

    # ── Baseline: ES-dLLM (used as speedup reference, same as GSM8K runs) ──
    echo ""
    echo "[LLaDA 1/4] ES-dLLM baseline (HiddenState)..."
    LOG_LLADA_BASE=$LOG_DIR/llada_humaneval_esdllm_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task humaneval \
        --esdllm_mode HiddenState \
        --alpha 0.5 \
        --prompt_update_freq 64 \
        --block_update_freq 4 \
        --proportions 1 0.5 0.25 \
        --positions 0 0.125 0.25 \
        2>&1 | tee $LOG_LLADA_BASE
    LLADA_BASE_TIME=$(grep "Total generation time" $LOG_LLADA_BASE | awk '{print $4}' | tr -d 's')
    echo "[LLaDA 1/4] Done. Time=${LLADA_BASE_TIME}s"

    # ── CAI APD only ──────────────────────────────────────────────────────
    echo ""
    echo "[LLaDA 2/4] CAI-APD only..."
    rm -rf lm_cache/
    LOG_LLADA_APD=$LOG_DIR/llada_humaneval_cai_apd_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task humaneval \
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

    # ── CAI per-block ─────────────────────────────────────────────────────
    echo ""
    echo "[LLaDA 3/4] CAI per-block..."
    rm -rf lm_cache/
    LOG_LLADA_PB=$LOG_DIR/llada_humaneval_cai_perblock_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task humaneval \
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

    # ── CAI confidence gating (all) ───────────────────────────────────────
    echo ""
    echo "[LLaDA 4/4] CAI confidence gating (all)..."
    rm -rf lm_cache/
    LOG_LLADA_CG=$LOG_DIR/llada_humaneval_cai_full_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model LLaDA-Instruct \
        --task humaneval \
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
    echo "=== LLaDA HumanEval Results ==="               | tee -a $RESULTS_FILE
    print_row "LLaDA" "HumanEval" "ES-dLLM (baseline)"   $LOG_LLADA_BASE ""
    print_row "LLaDA" "HumanEval" "CAI-APD only"         $LOG_LLADA_APD  $LLADA_BASE_TIME
    print_row "LLaDA" "HumanEval" "CAI per-block"        $LOG_LLADA_PB   $LLADA_BASE_TIME
    print_row "LLaDA" "HumanEval" "CAI confidence gating" $LOG_LLADA_CG  $LLADA_BASE_TIME
}

# =============================================================================
# Dream-7B-Instruct
# =============================================================================
run_dream() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " Dream-7B-Instruct — HumanEval"
    echo "──────────────────────────────────────────────────"

    # ── Baseline: DualCache ───────────────────────────────────────────────
    echo ""
    echo "[Dream 1/4] DualCache baseline..."
    LOG_DREAM_BASE=$LOG_DIR/dream_humaneval_dualcache_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task humaneval \
        --esdllm_mode DualCache \
        2>&1 | tee $LOG_DREAM_BASE
    DREAM_BASE_TIME=$(grep "Total generation time" $LOG_DREAM_BASE | awk '{print $4}' | tr -d 's')
    echo "[Dream 1/4] Done. Time=${DREAM_BASE_TIME}s"

    # ── CAI APD only ──────────────────────────────────────────────────────
    echo ""
    echo "[Dream 2/4] CAI-APD only..."
    rm -rf lm_cache/
    LOG_DREAM_APD=$LOG_DIR/dream_humaneval_cai_apd_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task humaneval \
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

    # ── CAI per-block ─────────────────────────────────────────────────────
    echo ""
    echo "[Dream 3/4] CAI per-block..."
    rm -rf lm_cache/
    LOG_DREAM_PB=$LOG_DIR/dream_humaneval_cai_perblock_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task humaneval \
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

    # ── CAI confidence gating (all) ───────────────────────────────────────
    echo ""
    echo "[Dream 4/4] CAI confidence gating (all)..."
    rm -rf lm_cache/
    LOG_DREAM_CG=$LOG_DIR/dream_humaneval_cai_full_${DATE}.log
    CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model Dream-Instruct \
        --task humaneval \
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
    echo "=== Dream HumanEval Results ==="               | tee -a $RESULTS_FILE
    print_row "Dream" "HumanEval" "DualCache (baseline)"  $LOG_DREAM_BASE ""
    print_row "Dream" "HumanEval" "CAI-APD only"          $LOG_DREAM_APD  $DREAM_BASE_TIME
    print_row "Dream" "HumanEval" "CAI per-block"         $LOG_DREAM_PB   $DREAM_BASE_TIME
    print_row "Dream" "HumanEval" "CAI confidence gating" $LOG_DREAM_CG   $DREAM_BASE_TIME
}

# =============================================================================
# Run based on filter
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
echo " All runs complete."                              | tee -a $RESULTS_FILE
echo " Full summary saved to: $RESULTS_FILE"           | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE
cat $RESULTS_FILE