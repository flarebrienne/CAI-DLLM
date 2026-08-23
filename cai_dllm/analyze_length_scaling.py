#!/usr/bin/env python3
"""
analyze_length_scaling.py
Shows how CAI-dLLM speedup scales with generation length.
Compares nocache, dualcache, esdllm, caidllm across gen_lengths.

Runtime: ~20-30 minutes on H200
Output: length_scaling.json + printed table

Usage:
  CUDA_VISIBLE_DEVICES=0 python cai_dllm/analyze_length_scaling.py \
      --model LLaDA-Instruct --n_samples 10 \
      --lengths 64,128,256,512
"""

import os, sys, time, json, argparse
import torch
import numpy as np

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
EOS_IDS    = {"llada": 126081, "Dream": 151643}

# Fixed prompt for all experiments
PROMPT = (
    "Janet sells 16 duck eggs a day. She eats 3 for breakfast every morning "
    "and bakes muffins for her friends every day with 4. She sells the "
    "remainder for $2 per egg. How much does she make every day? Answer:"
)

def get_gen_kwargs(method, model_type, gen_length, block_freq):
    block_length = min(64, gen_length)
    base = {
        "gen_length": gen_length,
        "block_length": block_length,
        "temperature": 0.0, "cfg_scale": 0.0,
        "use_kvcache": True, "parallel_mode": False,
        "token_per_step": 1, "threshold": None,
        "print_log": False, "record_time": False, "statistics": False,
        "delay_eos_generation": True, "sparse_kv": 1.0, "delay_step": -1,
        "top_p": 0.95, "top_k": 50,
        "ESdLLM_mode": None, "importance_score_alpha": 0.5,
        "proportion_steps": None, "block_update_freq": None,
        "prompt_update_freq": None,
    }
    if method == "nocache":
        base["use_kvcache"] = False
    elif method == "dualcache":
        base["use_kvcache"] = True
    elif method == "esdllm":
        base.update({
            "ESdLLM_mode": "HiddenState",
            "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
            "block_update_freq": block_freq,
            "prompt_update_freq": 64,
        })
    elif method == "caidllm":
        base.update({
            "parallel_mode": True,
            "ESdLLM_mode": None,
            "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
            "block_update_freq": block_freq,
            "prompt_update_freq": 64,
        })
    return base

def measure_tps(model, tokenizer, prompt, gen_kwargs,
                method, cai_cfg, eos_id, n_samples):
    """Measure tokens per second for a given method and gen_length."""
    enc = tokenizer([prompt], return_tensors="pt",
                    truncation=True, max_length=256)
    input_ids = enc["input_ids"].cuda()
    attn_mask  = enc["attention_mask"].float().cuda()
    gen_length = gen_kwargs["gen_length"]

    times = []
    for i in range(n_samples):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if method == "caidllm":
                cai_batch_generate(model, input_ids, attn_mask,
                                   generation_kwargs=gen_kwargs,
                                   cai_config=cai_cfg)
            else:
                batch_generate(model, input_ids, attn_mask, gen_kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_time = np.mean(times)
    tps = gen_length / avg_time
    return round(tps, 1), round(avg_time * 1000, 1)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     required=True,
                   choices=list(MODEL_PATHS.keys()))
    p.add_argument("--lengths",   type=str,
                   default="64,128,256,512",
                   help="Comma-separated generation lengths")
    p.add_argument("--n_samples", type=int, default=10,
                   help="Samples per configuration")
    p.add_argument("--warmup",    type=int, default=3)
    p.add_argument("--methods",   type=str,
                   default="nocache,dualcache,esdllm,caidllm")
    p.add_argument("--output_dir",type=str,
                   default="log_results/length_scaling")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_type = MODEL_TYPE[args.model]
    eos_id     = EOS_IDS[model_type]
    block_freq = 16 if model_type == "llada" else 8
    lengths    = [int(l) for l in args.lengths.split(",")]
    methods    = args.methods.split(",")

    print(f"\n{'='*65}", flush=True)
    print(f" Length Scaling Analysis — {args.model}", flush=True)
    print(f" Lengths: {lengths}", flush=True)
    print(f" Methods: {methods}", flush=True)
    print(f" Samples: {args.n_samples} (+ {args.warmup} warmup)", flush=True)
    print(f"{'='*65}\n", flush=True)

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

    cai_cfg = CAIConfig(
        use_apd=True, use_per_block=True,
        use_layer_adaptive=False,
        use_confidence_gating=(model_type == "llada"),
    )

    # Warmup
    print(f"Warming up ({args.warmup} iters)...", flush=True)
    wkwargs = get_gen_kwargs("caidllm", model_type, 64, block_freq)
    for _ in range(args.warmup):
        with torch.no_grad():
            enc = tokenizer([PROMPT], return_tensors="pt",
                            truncation=True, max_length=256)
            cai_batch_generate(
                model, enc["input_ids"].cuda(),
                enc["attention_mask"].float().cuda(),
                generation_kwargs=wkwargs, cai_config=cai_cfg)
    print("Warmup done.\n", flush=True)

    # Main experiment
    all_results = {}

    for gen_length in lengths:
        print(f"\n── gen_length={gen_length} ──────────────────────",
              flush=True)
        all_results[gen_length] = {}

        for method in methods:
            gen_kwargs = get_gen_kwargs(
                method, model_type, gen_length, block_freq)
            cfg = cai_cfg if method == "caidllm" else None

            tps, ms = measure_tps(
                model, tokenizer, PROMPT, gen_kwargs,
                method, cfg, eos_id, args.n_samples)

            all_results[gen_length][method] = {
                "tps": tps, "ms_per_iter": ms
            }
            print(f"  {method:<12} tps={tps:>7.1f}  ms={ms:>8.1f}",
                  flush=True)

    # Compute speedup vs nocache at each length
    print(f"\n{'='*65}", flush=True)
    print(f" SPEEDUP vs NOCACHE — {args.model}", flush=True)
    print(f"{'='*65}", flush=True)

    header = f"  {'Method':<12}"
    for gl in lengths:
        header += f" {gl:>8}"
    print(header, flush=True)
    print(f"  {'─'*55}", flush=True)

    for method in methods:
        row = f"  {method:<12}"
        for gl in lengths:
            nc_tps  = all_results[gl]["nocache"]["tps"]
            m_tps   = all_results[gl][method]["tps"]
            speedup = m_tps / nc_tps
            row += f" {speedup:>7.2f}x"
        print(row, flush=True)

    # TPS table
    print(f"\n{'='*65}", flush=True)
    print(f" TPS TABLE — {args.model}", flush=True)
    print(f"{'='*65}", flush=True)
    header = f"  {'Method':<12}"
    for gl in lengths:
        header += f" {gl:>8}"
    print(header, flush=True)
    print(f"  {'─'*55}", flush=True)
    for method in methods:
        row = f"  {method:<12}"
        for gl in lengths:
            row += f" {all_results[gl][method]['tps']:>8.1f}"
        print(row, flush=True)

    # Key finding: CAI-dLLM speedup trend
    print(f"\n  CAI-dLLM speedup trend:", flush=True)
    for gl in lengths:
        nc  = all_results[gl]["nocache"]["tps"]
        cai = all_results[gl]["caidllm"]["tps"]
        print(f"    gen={gl:>4}: {cai/nc:.2f}x speedup "
              f"({cai:.1f} vs {nc:.1f} TPS)", flush=True)

    print(f"{'='*65}\n", flush=True)

    # Save
    fname = os.path.join(
        args.output_dir,
        f"{args.model.replace('-','_')}_length_scaling.json")
    with open(fname, "w") as f:
        json.dump({
            "model": args.model,
            "lengths": lengths,
            "methods": methods,
            "results": {str(k): v for k,v in all_results.items()},
        }, f, indent=2)
    print(f"Saved → {fname}", flush=True)

if __name__ == "__main__":
    main()