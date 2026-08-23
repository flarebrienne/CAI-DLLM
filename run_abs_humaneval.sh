#!/bin/bash
# =============================================================================
# run_abs_humaneval.sh
#
# Runs 2 evaluations per model on HumanEval:
#   1. CAI-dLLM + Idea 1 (Adaptive Block Sizing) only
#   2. CAI-dLLM + All Combined (Ideas 1+3+4+5)
#
# Usage:
#   bash run_abs_humaneval.sh 0          # both models
#   bash run_abs_humaneval.sh 0 llada    # LLaDA only
#   bash run_abs_humaneval.sh 0 dream    # Dream only
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/enhancements"
mkdir -p $LOG_DIR
DATE=$(date +%s)

run_model() {
    local MODEL=$1
    local BLOCK_FREQ=$2
    local LOG1="$LOG_DIR/${MODEL}_humaneval_abs_only_${DATE}.log"
    local LOG2="$LOG_DIR/${MODEL}_humaneval_all_combined_${DATE}.log"

    echo ""
    echo "================================================"
    echo " ABS HumanEval — ${MODEL}"
    echo " GPU=$GPU  DATE=$(date)"
    echo "================================================"

    BASE="--model ${MODEL} --task humaneval \
        --esdllm_mode HiddenState --alpha 0.5 \
        --prompt_update_freq 64 --block_update_freq ${BLOCK_FREQ} \
        --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
        --use_cai --cai_mode apd_per_block"

    # Run 1: ABS only
    echo ""; echo "[1/2] ABS only..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py $BASE \
        --use_adaptive_block \
        2>&1 | tee $LOG1
    echo "[1/2] Done."

    # Run 2: All combined
    echo ""; echo "[2/2] All combined (Ideas 1+3+4+5)..."
    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py $BASE \
        --use_adaptive_block --use_soft_belief \
        --use_oracle_budget --use_prompt_kvcache \
        2>&1 | tee $LOG2
    echo "[2/2] Done."

    # Print summary
    echo ""
    echo "=== ${MODEL} HumanEval Summary ==="
    for LOG in $LOG1 $LOG2; do
        time=$(grep "Total generation time" $LOG | awk '{print $4}')
        acc=$(grep "pass@1" $LOG | tail -1 | awk '{print $6}')
        req=$(grep "Request count" $LOG | awk '{print $3}')
        eff=$(grep "Effective tokens" $LOG | awk '{print $5}')
        tps=$(echo "$req $eff $time" | awk '{printf "%.1f", ($1*$2)/$3}' 2>/dev/null)
        label=$(basename $LOG | sed "s/_${DATE}.log//")
        printf "  %-45s | pass@1=%-6s | time=%-8s | tps=%s\n" \
            "$label" "$acc" "${time}s" "$tps"
    done
}

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA-Instruct" 4
fi

if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream-Instruct" 8
fi

echo ""
echo "================================================"
echo " All done."
echo "================================================"