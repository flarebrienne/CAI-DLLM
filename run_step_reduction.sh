#!/bin/bash
# run_step_reduction.sh
# Measures average denoising steps per request for ES-dLLM vs CAI-dLLM
#
# Usage:
#   bash run_step_reduction.sh 0 llada
#   bash run_step_reduction.sh 0 dream
#   bash run_step_reduction.sh 0 both

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/step_reduction"
mkdir -p $LOG_DIR

run_model() {
    local MODEL=$1
    echo ""
    echo "================================================"
    echo " Step Reduction — ${MODEL}"
    echo "================================================"

    CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/measure_step_reduction.py \
        --model ${MODEL} \
        --tasks gsm8k,humaneval,bbh \
        --n_samples 10 \
        --output_dir $LOG_DIR \
        2>&1 | tee $LOG_DIR/${MODEL}_steps.log

    echo "Done: ${MODEL}"
}

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA-Instruct"
fi
if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream-Instruct"
fi

echo ""
echo "=== Combined Step Reduction Summary ==="
python3 - << 'EOF'
import json, glob

for model in ["LLaDA_Instruct", "Dream_Instruct"]:
    f = f"log_results/step_reduction/{model.replace('_','-')}_step_reduction.json"
    try:
        data = json.load(open(f))
        print(f"\n{data['model']}:")
        print(f"  {'Task':<15} {'ES-dLLM':>10} {'CAI-dLLM':>10} {'Reduction':>12}")
        print(f"  {'─'*50}")
        for task, r in data['results'].items():
            print(f"  {task:<15} {r['es_steps']:>10} "
                  f"{r['cai_steps']:>10} "
                  f"{r['reduction_pct']:>11.1f}%")
    except:
        pass
EOF
