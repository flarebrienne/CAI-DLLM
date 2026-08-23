#!/bin/bash
GPU=${1:-0}
SEEDS=(42 123 456)
LOG_DIR="log_results/multiseed"
mkdir -p $LOG_DIR

for MODEL_FLAG in "LLaDA-Instruct" "Dream-Instruct"; do
    SHORT=${MODEL_FLAG%-Instruct}
    echo "================================================"
    echo " Multi-seed MathQA — ${MODEL_FLAG}"
    echo "================================================"

    for SEED in "${SEEDS[@]}"; do
        echo ""; echo "  [Seed $SEED] ${MODEL_FLAG} mathqa..."
        CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval_math_code.py \
            --model ${MODEL_FLAG} \
            --methods nocache,dualcache,esdllm,caidllm \
            --tasks mathqa \
            --n_mathqa 500 \
            --output_dir $LOG_DIR \
            2>&1 | tee $LOG_DIR/${SHORT}_mathqa_seed${SEED}.log
        echo "  [Seed $SEED] Done."
    done

    echo ""
    echo "=== ${SHORT} MathQA — Results across 3 seeds ==="
    python3 - << PYEOF
import re, numpy as np

methods = ['nocache','dualcache','esdllm','caidllm']
results = {m: [] for m in methods}

for seed in [42, 123, 456]:
    log = f"$LOG_DIR/${SHORT}_mathqa_seed{seed}.log"
    try:
        with open(log) as f:
            content = f.read()
        for method in methods:
            # Match: "  caidllm                      33.60%      163.0"
            pattern = rf'{method}\s+([\d.]+)%'
            matches = re.findall(pattern, content)
            if matches:
                results[method].append(float(matches[-1]))
    except Exception as e:
        print(f"  Error: {e}")

print(f"  {'Method':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"  {'─'*55}")
for method, accs in results.items():
    if accs:
        print(f"  {method:<20} {np.mean(accs):>7.2f}% "
              f"{np.std(accs):>7.2f}% "
              f"{min(accs):>7.2f}% "
              f"{max(accs):>7.2f}%")
    else:
        print(f"  {method:<20}  no data")
PYEOF
done
