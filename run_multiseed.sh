#!/bin/bash
GPU=${1:-0}
FILTER=${2:-"both"}
TASK=${3:-"piqa"}
SEEDS=(42 123 456)
LOG_DIR="log_results/multiseed"
mkdir -p $LOG_DIR

run_seed() {
    local MODEL=$1
    local MODEL_FLAG=$2
    local SEED=$3
    local N_SAMPLES=$4
    local TASK_FLAG=$5

    echo "  [Seed $SEED] Running ${MODEL} ${TASK_FLAG}..."
    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval_commonsense.py \
        --model ${MODEL_FLAG} \
        --methods nocache,dualcache,esdllm,caidllm \
        --n_samples ${N_SAMPLES} \
        --tasks ${TASK_FLAG} \
        --output_dir $LOG_DIR \
        2>&1 | tee $LOG_DIR/${MODEL}_${TASK_FLAG}_seed${SEED}.log
    echo "  [Seed $SEED] Done."
}

run_model() {
    local MODEL=$1
    local MODEL_FLAG=$2
    local N_SAMPLES=$3

    echo ""
    echo "================================================"
    echo " Multi-seed ${TASK} — ${MODEL}"
    echo "================================================"

    for SEED in "${SEEDS[@]}"; do
        run_seed "$MODEL" "$MODEL_FLAG" "$SEED" "$N_SAMPLES" "$TASK"
    done

    echo ""
    echo "=== ${MODEL} ${TASK} — Results across 3 seeds ==="
    python3 - << PYEOF
import glob, re, numpy as np

methods = ['nocache','dualcache','esdllm','caidllm']
results = {m: [] for m in methods}

for seed in [42, 123, 456]:
    log = f"$LOG_DIR/${MODEL}_${TASK}_seed{seed}.log"
    try:
        with open(log) as f:
            content = f.read()
        for method in methods:
            # Find accuracy line for this method
            pattern = rf'{method}\s+[\d.]+%\s+([\d.]+)%'
            matches = re.findall(pattern, content)
            if matches:
                results[method].append(float(matches[0]))
    except: pass

print(f"  {'Method':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'─'*55}")
for method, accs in results.items():
    if accs:
        print(f"  {method:<20} {np.mean(accs):>7.2f}% "
              f"{np.std(accs):>7.2f}% "
              f"{min(accs):>7.2f}% "
              f"{max(accs):>7.2f}%")
    else:
        print(f"  {method:<20} {'no data':>8}")
PYEOF
}

N=500
[ "$TASK" == "piqa" ] && N=1838

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA" "LLaDA-Instruct" $N
fi
if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream" "Dream-Instruct" $N
fi
