#!/bin/bash
# =============================================================================
# run_bbh_energy.sh
# Measures GPU energy for BBH evaluation (nocache skipped — estimated)
# Methods: dualcache, esdllm, caidllm, caidllm_no_cg
#
# Usage:
#   bash run_bbh_energy.sh 0 llada
#   bash run_bbh_energy.sh 0 dream
#   bash run_bbh_energy.sh 0 both
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
    local LOG_PREFIX="$LOG_DIR/${MODEL}_bbh_${METHOD}_${DATE}"
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
        "method": "$METHOD", "model": "$MODEL", "task": "bbh",
        "duration_s": duration_s, "avg_power_w": avg_w,
        "peak_power_w": max(readings),
        "energy_j": energy_j, "energy_wh": energy_wh,
        "n_samples": len(readings),
        "estimated": False,
    }
    with open("${LOG_PREFIX}_energy.json","w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: ${LOG_PREFIX}_energy.json")
else:
    print("  No power readings captured")
PYEOF
}

save_nocache_estimate() {
    local MODEL=$1
    local DURATION_S=$2
    local LOG_PREFIX="$LOG_DIR/${MODEL}_bbh_nocache_${DATE}"

    python3 - << PYEOF
import json
duration_s = $DURATION_S
avg_power_w = 680.0  # estimated from HumanEval/GSM8K measurements
energy_j  = avg_power_w * duration_s
energy_wh = energy_j / 3600
print(f"  Method:     nocache (estimated)")
print(f"  Model:      $MODEL")
print(f"  Duration:   {duration_s:.0f}s ({duration_s/3600:.1f} hours)")
print(f"  Avg Power:  {avg_power_w:.1f} W (estimated from prior runs)")
print(f"  Energy:     {energy_j:.1f} J")
print(f"  Energy:     {energy_wh:.2f} Wh")
result = {
    "method": "nocache", "model": "$MODEL", "task": "bbh",
    "duration_s": duration_s, "avg_power_w": avg_power_w,
    "peak_power_w": avg_power_w,
    "energy_j": energy_j, "energy_wh": energy_wh,
    "n_samples": 0, "estimated": True,
    "note": "Estimated from speedup ratio and prior power measurements"
}
with open("${LOG_PREFIX}_energy.json","w") as f:
    json.dump(result, f, indent=2)
print(f"  Saved: ${LOG_PREFIX}_energy.json")
PYEOF
}

run_model() {
    local MODEL=$1
    local BLOCK_FREQ=$2
    local MODEL_FLAG=$3
    local NOCACHE_DURATION=$4   # estimated duration in seconds

    echo ""
    echo "================================================"
    echo " BBH Energy — ${MODEL}"
    echo " GPU=$GPU  DATE=$(date)"
    echo "================================================"

    BASE_ARGS="--model ${MODEL_FLAG} --task bbh \
        --esdllm_mode HiddenState --alpha 0.5 \
        --prompt_update_freq 64 --block_update_freq ${BLOCK_FREQ} \
        --proportions 1 0.5 0.25 --positions 0 0.125 0.25"

    CAI_ARGS="$BASE_ARGS --use_cai --cai_mode apd_per_block"

    # 0. Save nocache estimate (skip actual run)
    echo ""; echo "[0/4] nocache (estimated, skipping actual run)..."
    save_nocache_estimate "${MODEL}" "${NOCACHE_DURATION}"
    echo "[0/4] Done."

    # 1. DualCache
    echo ""; echo "[1/4] dualcache..."
    rm -rf lm_cache/
    measure_energy "dualcache" "${MODEL}" \
        "CUDA_VISIBLE_DEVICES=$GPU python eval.py \
        --model ${MODEL_FLAG} --task bbh --esdllm_mode DualCache"
    echo "[1/4] Done."

    # 2. ES-dLLM
    echo ""; echo "[2/4] esdllm..."
    rm -rf lm_cache/
    measure_energy "esdllm" "${MODEL}" \
        "CUDA_VISIBLE_DEVICES=$GPU python eval.py $BASE_ARGS"
    echo "[2/4] Done."

    # 3. CAI-dLLM
    echo ""; echo "[3/4] caidllm..."
    rm -rf lm_cache/
    measure_energy "caidllm" "${MODEL}" \
        "CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py $CAI_ARGS"
    echo "[3/4] Done."

    # 4. CAI-dLLM no confidence gating
    echo ""; echo "[4/4] caidllm_no_cg..."
    rm -rf lm_cache/
    measure_energy "caidllm_no_cg" "${MODEL}" \
        "CUDA_VISIBLE_DEVICES=$GPU python cai_dllm/eval3.py \
        $CAI_ARGS --no_confidence_gating"
    echo "[4/4] Done."

    # Summary
    echo ""
    echo "=== ${MODEL} BBH Energy Summary ==="
    python3 - << PYEOF
import json, glob

pattern = "log_results/energy/${MODEL}_bbh_*_${DATE}_energy.json"
files = sorted(glob.glob(pattern))
if not files:
    print("  No results found")
else:
    results = []
    for f in files:
        with open(f) as fp: results.append(json.load(fp))
    nocache_j = next((r['energy_j'] for r in results if r['method']=='nocache'), None)
    print(f"  {'Method':<22} {'Duration':>10} {'Avg W':>8} {'Energy J':>14} {'Energy Wh':>11} {'vs Nocache':>12} {'Saved':>8}")
    print(f"  {'─'*85}")
    for r in sorted(results, key=lambda x: x['energy_j'], reverse=True):
        rel   = f"{r['energy_j']/nocache_j:.4f}x" if nocache_j else "—"
        saved = f"{(1-r['energy_j']/nocache_j)*100:.1f}%" if nocache_j else "—"
        est   = " *" if r.get('estimated') else ""
        print(f"  {r['method']+est:<22} {r['duration_s']:>9.1f}s "
              f"{r['avg_power_w']:>7.1f}W "
              f"{r['energy_j']:>13.1f}J "
              f"{r['energy_wh']:>10.2f}Wh "
              f"{rel:>12} "
              f"{saved:>8}")
    print(f"  * estimated from speedup ratio and prior power measurements")
PYEOF
}

# LLaDA nocache estimated: 150483s (44.84x speedup over caidllm 3356s)
# Dream nocache estimated: 66999s  (23.0x speedup over caidllm 2913s)

if [ "$FILTER" == "llada" ] || [ "$FILTER" == "both" ]; then
    run_model "LLaDA" 4 "LLaDA-Instruct" 150483
fi

if [ "$FILTER" == "dream" ] || [ "$FILTER" == "both" ]; then
    run_model "Dream" 8 "Dream-Instruct" 66999
fi

echo ""
echo "All BBH energy measurements complete."
echo "Results saved in: $LOG_DIR"