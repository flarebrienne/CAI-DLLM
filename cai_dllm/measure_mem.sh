# Quick memory check — run ONCE for each method, measure peak memory
# Takes ~5 min per method (just loads model + runs a few batches)

GPU=0
LOG_DIR="log_results/memory_humaneval"
mkdir -p $LOG_DIR

measure_mem() {
    local label=$1
    shift
    # Sample memory every 2s during the run
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        -i $GPU -l 2 > $LOG_DIR/${label}_mem.txt &
    MEM_PID=$!
    CUDA_VISIBLE_DEVICES=$GPU python eval.py "$@" \
        > $LOG_DIR/${label}.log 2>&1
    kill $MEM_PID 2>/dev/null
    PEAK=$(sort -n $LOG_DIR/${label}_mem.txt | tail -1)
    echo "$label: peak memory = ${PEAK} MB"
}

# Run each method — skip nocache (too slow)
rm -rf lm_cache/
measure_mem "llada_dualcache" --model LLaDA-Instruct --task humaneval --esdllm_mode DualCache

rm -rf lm_cache/
measure_mem "llada_esdllm" --model LLaDA-Instruct --task humaneval \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 4 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25

rm -rf lm_cache/
measure_mem "llada_cai" --model LLaDA-Instruct --task humaneval \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 4 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
    --use_cai --cai_mode apd_per_block