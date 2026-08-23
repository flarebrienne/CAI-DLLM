#!/bin/bash
# =============================================================================
# run_memory_humaneval.sh  —  Memory cost measurement during HumanEval
# Measures peak GPU memory (MB) for each method alongside accuracy & speed
#
# Usage:
#   bash run_memory_humaneval.sh 0          # both models
#   bash run_memory_humaneval.sh 0 llada    # LLaDA only
#   bash run_memory_humaneval.sh 0 dream    # Dream only
# =============================================================================

GPU=${1:-0}
MODEL_FILTER=${2:-"both"}
LOG_DIR="log_results/memory_humaneval"
mkdir -p $LOG_DIR
DATE=$(date +%s)
RESULTS_FILE="$LOG_DIR/summary_${DATE}.txt"

echo "================================================" | tee $RESULTS_FILE
echo " HumanEval Memory + Performance Profiling"        | tee -a $RESULTS_FILE
echo " GPU=$GPU  DATE=$(date)"                           | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE

# ─── Memory monitor: samples GPU memory every 0.5s, writes peak to file ───────
# Usage: start_memory_monitor <output_file>
start_memory_monitor() {
    local mem_file=$1
    rm -f $mem_file
    (
        peak=0
        while true; do
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                  -i $GPU 2>/dev/null | tr -d ' ')
            if [ -n "$mem" ] && [ "$mem" -gt "$peak" ] 2>/dev/null; then
                peak=$mem
            fi
            echo $peak > $mem_file
            sleep 0.5
        done
    ) &
    echo $!   # return PID
}

stop_memory_monitor() {
    local pid=$1
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
}

read_peak_memory() {
    local mem_file=$1
    cat $mem_file 2>/dev/null || echo "N/A"
}

# ─── Run one eval + memory profile ───────────────────────────────────────────
run_with_memory() {
    local model=$1
    local method=$2
    local log=$3
    shift 3
    local eval_args="$@"

    local mem_file="${log%.log}_mem.txt"

    echo ""
    echo "  Running: $model | $method"

    # baseline memory before loading (model not yet loaded between runs
    # since eval.py loads fresh each time)
    local pid=$(start_memory_monitor $mem_file)
    sleep 1   # let monitor stabilize

    rm -rf lm_cache/
    CUDA_VISIBLE_DEVICES=$GPU python eval.py $eval_args \
        2>&1 | tee $log

    stop_memory_monitor $pid

    local peak_mem=$(read_peak_memory $mem_file)
    local time=$(grep "Total generation time" $log | awk '{print $4}' | tr -d 's')
    local requests=$(grep "Request count" $log | awk '{print $3}')
    local eff_tokens=$(grep "Effective tokens per request" $log | awk '{print $5}')
    local accuracy=$(grep -oP '\d+\.\d+' <(grep "pass@1" $log | tail -1) | head -1)
    local tps=""
    if [ -n "$time" ] && [ -n "$requests" ] && [ -n "$eff_tokens" ]; then
        tps=$(echo "$requests $eff_tokens $time" | awk '{printf "%.1f", ($1 * $2) / $3}')
    fi

    printf "  %-10s | %-22s | acc=%-7s | time=%-7s | tps=%-7s | mem=%-6s MB\n" \
        "$model" "$method" "$accuracy" "${time}s" "$tps" "$peak_mem" \
        | tee -a $RESULTS_FILE

    # store memory for speedup calc
    eval "MEM_${model}_${method//[-. ]/_}=$peak_mem"
    eval "TIME_${model}_${method//[-. ]/_}=$time"
}

# =============================================================================
# LLaDA HumanEval
# =============================================================================
run_llada() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " LLaDA-8B-Instruct — HumanEval Memory Profiling"
    echo "──────────────────────────────────────────────────"
    echo "" | tee -a $RESULTS_FILE
    echo "=== LLaDA HumanEval ===" | tee -a $RESULTS_FILE

    # 1. Nocache (vanilla)
    run_with_memory "LLaDA" "Nocache" \
        $LOG_DIR/llada_he_nocache_${DATE}.log \
        --model LLaDA-Instruct --task humaneval --esdllm_mode nocache

    # 2. DualCache
    run_with_memory "LLaDA" "DualCache" \
        $LOG_DIR/llada_he_dualcache_${DATE}.log \
        --model LLaDA-Instruct --task humaneval --esdllm_mode DualCache

    # 3. ES-dLLM
    run_with_memory "LLaDA" "ES-dLLM" \
        $LOG_DIR/llada_he_esdllm_${DATE}.log \
        --model LLaDA-Instruct --task humaneval \
        --esdllm_mode HiddenState --alpha 0.5 \
        --prompt_update_freq 64 --block_update_freq 4 \
        --proportions 1 0.5 0.25 --positions 0 0.125 0.25

    # 4. CAI-dLLM
    run_with_memory "LLaDA" "CAI-dLLM" \
        $LOG_DIR/llada_he_cai_${DATE}.log \
        --model LLaDA-Instruct --task humaneval \
        --esdllm_mode HiddenState --alpha 0.5 \
        --prompt_update_freq 64 --block_update_freq 4 \
        --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
        --use_cai --cai_mode apd_per_block
}

# =============================================================================
# Dream HumanEval
# =============================================================================
run_dream() {
    echo ""
    echo "──────────────────────────────────────────────────"
    echo " Dream-7B-Instruct — HumanEval Memory Profiling"
    echo "──────────────────────────────────────────────────"
    echo "" | tee -a $RESULTS_FILE
    echo "=== Dream HumanEval ===" | tee -a $RESULTS_FILE

    # 1. Nocache (vanilla)
    run_with_memory "Dream" "Nocache" \
        $LOG_DIR/dream_he_nocache_${DATE}.log \
        --model Dream-Instruct --task humaneval --esdllm_mode nocache

    # 2. DualCache
    run_with_memory "Dream" "DualCache" \
        $LOG_DIR/dream_he_dualcache_${DATE}.log \
        --model Dream-Instruct --task humaneval --esdllm_mode DualCache

    # 3. ES-dLLM
    run_with_memory "Dream" "ES-dLLM" \
        $LOG_DIR/dream_he_esdllm_${DATE}.log \
        --model Dream-Instruct --task humaneval \
        --esdllm_mode HiddenState --alpha 0.5 \
        --prompt_update_freq 64 --block_update_freq 8 \
        --proportions 1 0.5 0.25 --positions 0 0.125 0.25

    # 4. CAI-dLLM
    run_with_memory "Dream" "CAI-dLLM" \
        $LOG_DIR/dream_he_cai_${DATE}.log \
        --model Dream-Instruct --task humaneval \
        --esdllm_mode HiddenState --alpha 0.5 \
        --prompt_update_freq 64 --block_update_freq 8 \
        --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
        --use_cai --cai_mode apd_per_block
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
echo " All done. Full summary: $RESULTS_FILE"          | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE

# ─── Also print GPU info for the paper ───────────────────────────────────────
echo ""
echo "=== GPU Info ===" | tee -a $RESULTS_FILE
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i $GPU \
    | tee -a $RESULTS_FILE