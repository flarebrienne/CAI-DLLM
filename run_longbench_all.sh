#!/bin/bash
# =============================================================================
# run_longbench_all.sh  —  LongBench HotpotQA for all methods + both models
#
# Usage:
#   bash run_longbench_all.sh 0           # all 8 runs on GPU 0
#   bash run_longbench_all.sh 0 llada     # LLaDA only (4 runs)
#   bash run_longbench_all.sh 0 dream     # Dream only (4 runs)
#   bash run_longbench_all.sh 0 fast      # skip nocache (6 runs, ~2 hours)
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"all"}
LOG_DIR="log_results/longbench"
mkdir -p $LOG_DIR
DATE=$(date +%s)
SUMMARY="$LOG_DIR/summary_${DATE}.txt"

echo "================================================" | tee $SUMMARY
echo " LongBench HotpotQA — All Methods"               | tee -a $SUMMARY
echo " GPU=$GPU  FILTER=$FILTER  DATE=$(date)"          | tee -a $SUMMARY
echo "================================================" | tee -a $SUMMARY

run_eval() {
    local model=$1
    local method=$2
    echo ""
    echo "── $model | $method ─────────────────────────────"
    CUDA_VISIBLE_DEVICES=$GPU python eval_longbench_hotpotqa.py \
        --model $model \
        --method $method \
        --n_samples 200 \
        --batch_size 4 \
        --max_input_len 2048 \
        --gen_length 64 \
        --output_dir $LOG_DIR \
        2>&1 | tee $LOG_DIR/${model}_${method}_${DATE}.log

    # Extract and print summary line
    f1=$(grep "F1 Score" $LOG_DIR/${model}_${method}_${DATE}.log | awk '{print $3}')
    em=$(grep "Exact Match" $LOG_DIR/${model}_${method}_${DATE}.log | awk '{print $3}')
    t=$(grep "Total time" $LOG_DIR/${model}_${method}_${DATE}.log | awk '{print $3}')
    tps=$(grep "Tokens/sec" $LOG_DIR/${model}_${method}_${DATE}.log | awk '{print $2}')
    printf "  %-18s | %-10s | F1=%-8s | EM=%-8s | time=%-8s | tps=%s\n" \
        "$model" "$method" "$f1" "$em" "${t}s" "$tps" | tee -a $SUMMARY
}

skip_nocache() { [[ "$FILTER" == "fast" ]]; }
run_llada()    { [[ "$FILTER" == "all" || "$FILTER" == "llada" || "$FILTER" == "fast" ]]; }
run_dream()    { [[ "$FILTER" == "all" || "$FILTER" == "dream" || "$FILTER" == "fast" ]]; }

# ── LLaDA runs ────────────────────────────────────────────────────────────────
if run_llada; then
    skip_nocache || run_eval "LLaDA-Instruct" "nocache"
    run_eval "LLaDA-Instruct" "dualcache"
    run_eval "LLaDA-Instruct" "esdllm"
    run_eval "LLaDA-Instruct" "caidllm"
fi

# ── Dream runs ────────────────────────────────────────────────────────────────
if run_dream; then
    skip_nocache || run_eval "Dream-Instruct" "nocache"
    run_eval "Dream-Instruct" "dualcache"
    run_eval "Dream-Instruct" "esdllm"
    run_eval "Dream-Instruct" "caidllm"
fi

echo ""
echo "================================================" | tee -a $SUMMARY
echo " Done. Summary saved to: $SUMMARY"               | tee -a $SUMMARY
echo "================================================" | tee -a $SUMMARY
cat $SUMMARY
