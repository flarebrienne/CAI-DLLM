#!/bin/bash
# =============================================================================
# run_longctx_all.sh  —  Long Context LongBench (Exact Paper Protocol)
#
# Table 6: 5 methods × 6 tasks aggregated, max_input=4000, block=32, gen=512
# Figure 5: 4 methods × 4 context lengths × 6 tasks, TPS + peak memory
#
# ALL methods use IDENTICAL settings: block=32, gen=512, max_input=4000
# OOM recorded explicitly, not silently dropped
#
# Usage:
#   bash run_longctx_all.sh 0 llada    (~6 hours)
#   bash run_longctx_all.sh 0 dream    (~6 hours)
#   bash run_longctx_all.sh 0 both     (~12 hours)
# =============================================================================

GPU=${1:-0}
MODEL_FILTER=${2:-"both"}
LOG_DIR="log_results/longctx"
mkdir -p $LOG_DIR
DATE=$(date +%s)
SUMMARY="$LOG_DIR/summary_${DATE}.txt"

EVAL="python cai_dllm/eval_longbench_longctx.py"

# Fixed protocol args — IDENTICAL for every method (fairness)
PROTO="--n_samples 200 --batch_size 2 \
       --block_length 32 --gen_length 512 \
       --max_input_len 4000 --tasks all \
       --output_dir $LOG_DIR"

# Scaling uses same protocol but fewer samples and varies max_input_len
PROTO_SCALE="--n_samples 100 --batch_size 2 \
             --block_length 32 --gen_length 512 \
             --tasks all --output_dir $LOG_DIR"

echo "================================================" | tee $SUMMARY
echo " Long Context LongBench — Paper Protocol"        | tee -a $SUMMARY
echo " GPU=$GPU  FILTER=$MODEL_FILTER  DATE=$(date)"   | tee -a $SUMMARY
echo " Table 6: block=32 | gen=512 | max_input=4000"   | tee -a $SUMMARY
echo " Figure 5: ctx=1k/2k/3k/4k, all 6 tasks"        | tee -a $SUMMARY
echo " Paper ref: LLaDA≈34.55  Dream≈38.62"            | tee -a $SUMMARY
echo "================================================" | tee -a $SUMMARY

# ── Extract from JSON ─────────────────────────────────────────────────────────
get_json() {
    local fname=$1; local key=$2
    python -c "import json; d=json.load(open('$fname')); print(d.get('$key','N/A'))" 2>/dev/null
}

print_table6_row() {
    local model=$1; local method=$2
    local fname="${LOG_DIR}/${model//-/_}_${method}_longbench_ctx4000.json"
    [ ! -f "$fname" ] && \
        printf "  %-20s | %-20s | NOT FOUND\n" "$model" "$method" | tee -a $SUMMARY && return
    local score=$(get_json $fname longbench_score)
    local tps=$(get_json $fname avg_tps)
    local mem=$(get_json $fname peak_memory_mb)
    local oom=$(get_json $fname total_oom)
    printf "  %-20s | %-20s | score=%-6s | tps=%-7s | mem=%-6s MB | OOM=%s\n" \
        "$model" "$method" "${score}%" "$tps" "$mem" "$oom" | tee -a $SUMMARY
}

print_scaling_row() {
    local model=$1; local method=$2; local ctx=$3
    local fname="${LOG_DIR}/${model//-/_}_${method}_longbench_ctx${ctx}.json"
    [ ! -f "$fname" ] && \
        printf "    %-12s ctx=%-5s | NOT FOUND\n" "$method" "${ctx}tok" | tee -a $SUMMARY && return
    local tps=$(get_json $fname avg_tps)
    local mem=$(get_json $fname peak_memory_mb)
    local oom=$(get_json $fname total_oom)
    printf "    %-12s | ctx=%-5s | tps=%-7s | mem=%-6s MB | OOM=%s\n" \
        "$method" "${ctx}tok" "$tps" "$mem" "$oom" | tee -a $SUMMARY
}

# =============================================================================
# TABLE 6 — Quality at ctx=4000, all 6 tasks aggregated
# All 5 methods with IDENTICAL protocol
# =============================================================================
run_table6() {
    local MODEL=$1
    echo ""
    echo "══════════════════════════════════════════════"
    echo " TABLE 6 — $MODEL — 5 methods × 6 tasks"
    echo " All use: block=32, gen=512, max_input=4000"
    echo "══════════════════════════════════════════════"

    for METHOD in nocache dualcache esdllm caidllm caidllm_enhanced; do
        echo ""; echo "── $METHOD ..."
        CUDA_VISIBLE_DEVICES=$GPU $EVAL \
            --model $MODEL --method $METHOD \
            $PROTO \
            2>&1 | tee $LOG_DIR/${MODEL//-/_}_${METHOD}_table6_${DATE}.log
    done

    echo ""
    echo "=== TABLE 6 — $MODEL ===" | tee -a $SUMMARY
    echo "  (Paper: Base=nocache, LLaDA≈34.55, Dream≈38.62)" | tee -a $SUMMARY
    for METHOD in nocache dualcache esdllm caidllm caidllm_enhanced; do
        print_table6_row "$MODEL" "$METHOD"
    done
}

# =============================================================================
# FIGURE 5 — Scaling efficiency curves
# ALL 4 methods at ctx = 1000, 2000, 3000, 4000
# ALL 6 tasks aggregated (same as Table 6, just shorter samples)
# Records TPS + peak memory at each ctx length
# OOM recorded explicitly
# =============================================================================
run_figure5() {
    local MODEL=$1
    echo ""
    echo "══════════════════════════════════════════════"
    echo " FIGURE 5 — $MODEL — Context Scaling"
    echo " 4 methods × 4 ctx lengths × 6 tasks"
    echo "══════════════════════════════════════════════"

    for CTX in 1000 2000 3000 4000; do
        echo ""; echo "── ctx=$CTX ──"
        for METHOD in nocache dualcache esdllm caidllm; do
            echo "  $METHOD ctx=$CTX..."
            CUDA_VISIBLE_DEVICES=$GPU $EVAL \
                --model $MODEL --method $METHOD \
                $PROTO_SCALE --max_input_len $CTX \
                2>&1 | tee $LOG_DIR/${MODEL//-/_}_${METHOD}_ctx${CTX}_${DATE}.log
        done
    done

    echo ""
    echo "=== FIGURE 5 — $MODEL Context Scaling ===" | tee -a $SUMMARY
    for CTX in 1000 2000 3000 4000; do
        echo "  ctx=${CTX}tok:" | tee -a $SUMMARY
        for METHOD in nocache dualcache esdllm caidllm; do
            print_scaling_row "$MODEL" "$METHOD" "$CTX"
        done
    done
}

# =============================================================================
# Run
# =============================================================================
if [ "$MODEL_FILTER" == "llada" ] || [ "$MODEL_FILTER" == "both" ]; then
    run_table6  "LLaDA-Instruct"
    run_figure5 "LLaDA-Instruct"
fi

if [ "$MODEL_FILTER" == "dream" ] || [ "$MODEL_FILTER" == "both" ]; then
    run_table6  "Dream-Instruct"
    run_figure5 "Dream-Instruct"
fi

echo ""
echo "================================================" | tee -a $SUMMARY
echo " Done. Summary → $SUMMARY"                       | tee -a $SUMMARY
echo "================================================" | tee -a $SUMMARY
cat $SUMMARY
