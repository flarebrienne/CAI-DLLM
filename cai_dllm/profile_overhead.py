#!/usr/bin/env python3
"""
profile_overhead.py
Measures the overhead of CAI-dLLM controller vs standard parallel decoding.
Runs N iterations with controller ON vs OFF and reports overhead percentage.

Usage:
  CUDA_VISIBLE_DEVICES=0 python cai_dllm/profile_overhead.py \
      --model LLaDA-Instruct --n_iters 100 --gen_length 256
"""

import os, sys, time, argparse, statistics
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from transformers import AutoTokenizer, AutoModel
from generate import batch_generate
from cai_generate3 import cai_batch_generate, CAIConfig

MODEL_PATHS = {
    "LLaDA-Instruct": "GSAI-ML/LLaDA-8B-Instruct",
    "Dream-Instruct":  "Dream-org/Dream-v0-Instruct-7B",
}
MODEL_TYPE = {"LLaDA-Instruct": "llada", "Dream-Instruct": "Dream"}

# Fixed prompt for profiling
FIXED_PROMPT = (
    "Janet sells 16 duck eggs a day. She eats 3 for breakfast every morning "
    "and bakes muffins for her friends every day with 4. She sells the "
    "remainder for $2 per egg. How much does she make every day? Answer:"
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--n_iters",    type=int, default=100)
    p.add_argument("--warmup",     type=int, default=10,
                   help="Warmup iterations before timing")
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length",type=int, default=64)
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()

def make_gen_kwargs(model_type, gen_length, block_length, block_freq, parallel=True):
    return {
        "gen_length": gen_length, "block_length": block_length,
        "temperature": 0.0, "cfg_scale": 0.0, "use_kvcache": True,
        "parallel_mode": parallel, "token_per_step": 1, "threshold": 0.9 if parallel else None,
        "print_log": False, "record_time": False, "statistics": False,
        "delay_eos_generation": True, "sparse_kv": 1.0, "delay_step": -1,
        "top_p": 0.95, "top_k": 50,
        "block_update_freq": block_freq, "prompt_update_freq": 64,
        "ESdLLM_mode": None, "importance_score_alpha": 0.5,
        "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
    }

def run_iteration(model, input_ids, attn_mask, gen_kwargs, cai_cfg=None):
    """Run one decode loop and return wall-clock time."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        if cai_cfg is not None:
            cai_batch_generate(model, input_ids, attn_mask,
                               generation_kwargs=gen_kwargs,
                               cai_config=cai_cfg)
        else:
            batch_generate(model, input_ids, attn_mask, gen_kwargs)
    torch.cuda.synchronize()
    return time.perf_counter() - t0

def main():
    args = parse_args()
    model_type = MODEL_TYPE[args.model]
    block_freq  = 16 if model_type == "llada" else 8

    print(f"\n{'='*60}", flush=True)
    print(f" CAI-dLLM Overhead Profiling", flush=True)
    print(f" Model:      {args.model}", flush=True)
    print(f" Iterations: {args.n_iters} (+ {args.warmup} warmup)", flush=True)
    print(f" Gen length: {args.gen_length}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load model
    print("Loading model...", flush=True)
    model_path = MODEL_PATHS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        padding_side="left" if model_type == "Dream" else "right")
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda")
    if model_type == "llada":
        from models.hook_model import transform_llada_model
        transform_llada_model(model)
    else:
        from models.hook_model import transform_dream_model
        transform_dream_model(model)
    model.eval()
    print("Model ready.\n", flush=True)

    # Prepare fixed input
    prompts = [FIXED_PROMPT] * args.batch_size
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
    input_ids = enc["input_ids"].cuda()
    attn_mask  = enc["attention_mask"].float().cuda()

    # Gen kwargs
    gen_kwargs_off = make_gen_kwargs(model_type, args.gen_length,
                                     args.block_length, block_freq,
                                     parallel=True)
    gen_kwargs_on  = make_gen_kwargs(model_type, args.gen_length,
                                     args.block_length, block_freq,
                                     parallel=True)

    # CAI config
    cai_cfg = CAIConfig(
        use_apd=True, use_per_block=True,
        use_layer_adaptive=False, use_confidence_gating=True,
    )

    # ── Warmup ────────────────────────────────────────────────
    print(f"Warming up ({args.warmup} iters)...", flush=True)
    for i in range(args.warmup):
        run_iteration(model, input_ids, attn_mask, gen_kwargs_off, None)
        run_iteration(model, input_ids, attn_mask, gen_kwargs_on, cai_cfg)
        if (i+1) % 5 == 0:
            print(f"  Warmup {i+1}/{args.warmup}", flush=True)

    # ── Controller OFF (standard parallel decoding) ───────────
    print(f"\nProfiling controller OFF ({args.n_iters} iters)...", flush=True)
    times_off = []
    for i in range(args.n_iters):
        t = run_iteration(model, input_ids, attn_mask, gen_kwargs_off, None)
        times_off.append(t)
        if (i+1) % 20 == 0:
            print(f"  OFF [{i+1}/{args.n_iters}] "
                  f"avg={statistics.mean(times_off)*1000:.1f}ms", flush=True)

    # ── Controller ON (CAI-dLLM) ──────────────────────────────
    print(f"\nProfiling controller ON ({args.n_iters} iters)...", flush=True)
    times_on = []
    for i in range(args.n_iters):
        t = run_iteration(model, input_ids, attn_mask, gen_kwargs_on, cai_cfg)
        times_on.append(t)
        if (i+1) % 20 == 0:
            print(f"  ON  [{i+1}/{args.n_iters}] "
                  f"avg={statistics.mean(times_on)*1000:.1f}ms", flush=True)

    # ── Results ───────────────────────────────────────────────
    mean_off = statistics.mean(times_off)
    mean_on  = statistics.mean(times_on)
    std_off  = statistics.stdev(times_off)
    std_on   = statistics.stdev(times_on)
    overhead_pct = (mean_on - mean_off) / mean_off * 100

    print(f"\n{'='*60}", flush=True)
    print(f" PROFILING RESULTS — {args.model}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Controller OFF: {mean_off*1000:.2f} ± {std_off*1000:.2f} ms/iter", flush=True)
    print(f"  Controller ON:  {mean_on*1000:.2f} ± {std_on*1000:.2f} ms/iter", flush=True)
    print(f"  Overhead:       {overhead_pct:+.2f}%", flush=True)
    print(f"  Absolute delta: {(mean_on-mean_off)*1000:.2f} ms/iter", flush=True)
    print(f"{'='*60}", flush=True)

    if overhead_pct < 2.0:
        print(f"\n  Overhead < 2% — negligible, confirms CAI controller", flush=True)
        print(f"  adds no meaningful latency per denoising step.", flush=True)
    else:
        print(f"\n  Overhead = {overhead_pct:.1f}% — worth investigating.", flush=True)

    # Save
    import json
    os.makedirs("log_results/profiling", exist_ok=True)
    result = {
        "model": args.model, "n_iters": args.n_iters,
        "gen_length": args.gen_length, "block_length": args.block_length,
        "mean_off_ms": mean_off*1000, "std_off_ms": std_off*1000,
        "mean_on_ms":  mean_on*1000,  "std_on_ms":  std_on*1000,
        "overhead_pct": overhead_pct,
        "times_off_ms": [t*1000 for t in times_off],
        "times_on_ms":  [t*1000 for t in times_on],
    }
    fname = f"log_results/profiling/{args.model.replace('-','_')}_overhead.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {fname}", flush=True)

if __name__ == "__main__":
    main()
