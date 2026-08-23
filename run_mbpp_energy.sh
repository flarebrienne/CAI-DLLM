#!/bin/bash
# =============================================================================
# run_mbpp_energy.sh
# Measures GPU energy consumption for MBPP evaluation
# Methods: nocache, dualcache, esdllm, caidllm, caidllm_no_cg
#
# Usage:
#   bash run_mbpp_energy.sh 0 llada   # LLaDA only
#   bash run_mbpp_energy.sh 0 dream   # Dream only
#   bash run_mbpp_energy.sh 0 both    # Both models
# =============================================================================

GPU=${1:-0}
FILTER=${2:-"both"}
LOG_DIR="log_results/energy"
mkdir -p $LOG_DIR
DATE=$(date +%s)

measure_energy() {
    local METHOD=$1
    local MODEL=$2
    local CMD=$3
    local LOG_PREFIX="$LOG_DIR/${MODEL}_mbpp_${METHOD}_${DATE}"
    local POWER_LOG="${LOG_PREFIX}_power.csv"
    local RUN_LOG="${LOG_PREFIX}_run.log"

    echo "  [Energy] Starting power monitor..."
    nvidia-smi --query-gpu=timestamp,power.draw \
        --format=csv,noheader --loop-ms=500 -i $GPU > $POWER_LOG &
    MONITOR_PID=$!
    START_TIME=$(date +%s%3N)

    eval "$CMD" > $RUN_LOG 2>&1

    END_TIME=$(date +%s%3N)
    DURATION_MS=$((END_TIME - START_TIME))
    kill $MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null

    python3 - << PYEOF
import json

power_log = "$POWER_LOG"
duration_s = $DURATION_MS / 1000.0
readings = []
with open(power_log, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                readings.append(float(parts[1].strip().replace(' W','')))
            except: pass

if readings:
    avg_w     = sum(readings)/len(readings)
    energy_j  = avg_w * duration_s
    energy_wh = energy_j / 3600
    print(f"  Method:     $METHOD")
    print(f"  Model:      $MODEL")
    print(f"  Duration:   {duration_s:.1f}s")
    print(f"  Avg Power:  {avg_w:.1f} W")
    print(f"  Peak Power: {max(readings):.1f} W")
    print(f"  Energy:     {energy_j:.1f} J")
    print(f"  Energy:     {energy_wh:.4f} Wh")
    print(f"  Samples:    {len(readings)}")
    result = {
        "method": "$METHOD", "model": "$MODEL", "task": "mbpp",
        "duration_s": duration_s, "avg_power_w": avg_w,
        "peak_power_w": max(readings),
        "energy_j": energy_j, "energy_wh": energy_wh,
        "n_samples": len(readings),
    }
    with open("${LOG_PREFIX}_energy.json","w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: ${LOG_PREFIX}_energy.json")
else:
    print("  No power readings captured")
PYEOF
}

run_model() {
    local MODEL=$1
    local BLOCK_FREQ=$2
    local MODEL_FLAG=$3

    echo ""
    echo "================================================"
    echo " MBPP Energy — ${MODEL}"
    echo " GPU=$GPU  DATE=$(date)"
    echo "================================================"

    BASE_ARGS="--model ${MODEL_FLAG} --task mbpp \
        --esdllm_mode HiddenState --alpha 0.5 \
        --prompt_update_freq 64 --block_update_freq ${BLOCK_FREQ} \
        --proportions 1 0.5 0.25 --positions 0 0.125 0.25"

    CAI_ARGS="$BASE_ARGS --use_cai --cai_mode apd_per_block"

    # 1. Nocache
    echo ""; echo "[1/5] nocache..."
    rm -rf lm_cache/
    measure_energy "nocache" "${MODEL}" \
        "CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model ${MODEL_FLAG} --task mbpp --esdllm_mode nocache"
    echo "[1/5] Done."

    # 2. DualCache
    echo ""; echo "[2/5] dualcache..."
    rm -rf lm_cache/
    measure_energy "dualcache" "${MODEL}" \
        "CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model ${MODEL_FLAG} --task mbpp --esdllm_mode DualCache"
    echo "[2/5] Done."

    # 3. ES-dLLM
    echo ""; echo "[3/5] esdllm..."
    rm -rf lm_cache/
    measure_energy "esdllm" "${MODEL}" \
        "CUDA_VISIBLE_DEVICES=$GPU python eval.py $BASE_ARGS"
    echo "[3/5] Done."

    # 4. CAI-dLLM
    echo ""; echo "[4/5] caidllm..."
    rm -rf lm_cache/
    measure_energy "caidllm" "${MODEL}" \
        "CUDA_VISIBLE_DEVICES=$GPU python eval.py $CAI_ARGS"
    echo "[4/5] Done."

    # 5. CAI-dLLM no confidence gating
    echo ""; echo "[5/5] caidllm_no_cg..."
    rm -rf lm_cache/
    measure_energy "caidllm_no_cg" "${MODEL}" \
        "CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        $CAI_ARGS --no_confidence_gating"
    echo "[5/5] Done."

    # Summary
    echo ""
    echo "=== ${MODEL} MBPP Energy Summary ==="
    python3 - << PYEOF
import json, glob

pattern = "log_results/energy/${MODEL}_mbpp_*_${DATE}_energy.json"
files = sorted(glob.glob(pattern))
if not files:
    print("  No results found")
else:
    results = []
    for f in files:
        with open(f) as fp: results.append(json.load(fp))
    nocache_j = next((r['energy_j'] for r in results if r['method']=='nocache'), None)
    print(f"  {'Method':<20} {'Duration':>10} {'Avg W':>8} {'Energy J':>12} {'Energy Wh':>10} {'vs Nocache':>12} {'Saved':>8}")
    print(f"  {'─'*80}")
    for r in sorted(results, key=lambda x: x['energy_j'], reverse=True):
        rel  = f"{r['energy_j']/nocache_j:.3f}x" if nocache_j else "—"
        saved = f"{(1-r['energy_j']/nocache_j)*100:.1f}%" if nocache_j else "—"
        print(f"  {r['method']:<20} {r['duration_s']:>9.1f}s "
              f"{r['avg_power_w']:>7.1f}W "
              f"{r['energy_j']:>11.1f}J "
              f"{r['energy_wh']:>9.4f}Wh "
              f"{rel:>12} "
              f"{saved:>8}")
PYEOF
}

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA" 4 "LLaDA-Instruct"
fi

if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream" 8 "Dream-Instruct"
fi

echo ""
echo "All MBPP energy measurements complete."
echo "Results saved in: $LOG_DIR"