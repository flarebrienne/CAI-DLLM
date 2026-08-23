#!/usr/bin/env python3
"""
eval_poem_completion.py — Chinese Poem Completion Evaluation
=============================================================
Follows the protocol from Allen-Zhu and Li [35] as used in the LLaDA paper.
496 famous Chinese poem sentence pairs.
Tasks:
  - Forward:  given line N, generate line N+1
  - Reversal: given line N+1, generate line N

Methods: nocache, dualcache, esdllm, caidllm (+ no_confidence_gating variant)
Also supports GPT-4o and Qwen2.5-7B for autoregressive comparison.

Metric: Exact Match (EM) on character level

Usage:
  # dLLM methods
  python cai_dllm/eval_poem_completion.py --model LLaDA-Instruct --methods nocache,dualcache,esdllm,caidllm

  # Autoregressive comparison (requires API keys or local model)
  python cai_dllm/eval_poem_completion.py --model qwen --methods ar

  # Full comparison
  bash run_poem_completion.sh 0
"""

import os, sys, time, json, argparse, re
import torch
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

print("Loading modules...", flush=True)
from transformers import AutoTokenizer, AutoModel
from generate import batch_generate

try:
    from cai_generate3 import cai_batch_generate, CAIConfig
    CAI_AVAILABLE = True
    print("CAI modules loaded.", flush=True)
except ImportError:
    CAI_AVAILABLE = False
    print("WARNING: CAI not available.", flush=True)

MODEL_PATHS = {
    "LLaDA-Instruct": "GSAI-ML/LLaDA-8B-Instruct",
    "Dream-Instruct":  "Dream-org/Dream-v0-Instruct-7B",
}
MODEL_TYPE = {"LLaDA-Instruct": "llada", "Dream-Instruct": "Dream"}
EOS_IDS    = {"llada": 126081, "Dream": 151643}


# =============================================================================
# Dataset — construct poem pairs following LLaDA paper protocol
# =============================================================================
def load_poem_dataset(n=496):
    """
    Load Chinese poem pairs for forward and reversal tasks.
    Uses hardcoded Tang/Song dynasty poems (public domain).
    Follows LLaDA paper protocol: 496 consecutive line pairs.
    """
    print(f"Loading poem dataset ({n} pairs)...", flush=True)
    from poem_data import get_poem_pairs
    items = get_poem_pairs(n)
    print(f"  Loaded {len(items)} pairs from built-in poem dataset", flush=True)
    return items


# =============================================================================
# Prompt builders
# =============================================================================
def build_forward_prompt(line_a):
    return (f"以下是一首古诗的连续两句。请补全第二句。\n\n"
            f"第一句：{line_a}\n"
            f"第二句：")

def build_reversal_prompt(line_b):
    return (f"以下是一首古诗的连续两句。请补全第一句。\n\n"
            f"第二句：{line_b}\n"
            f"第一句：")


# =============================================================================
# Scoring
# =============================================================================
def exact_match(pred, gold):
    """Character-level exact match after stripping punctuation."""
    def clean(s):
        return re.sub(r'[，。！？；、\s]', '', s.strip())
    return int(clean(pred) == clean(gold))

def char_f1(pred, gold):
    """Character-level F1."""
    pred_chars = list(re.sub(r'[，。！？；、\s]', '', pred))
    gold_chars = list(re.sub(r'[，。！？；、\s]', '', gold))
    if not pred_chars or not gold_chars:
        return 0.0
    common = sum(min(pred_chars.count(c), gold_chars.count(c))
                 for c in set(pred_chars))
    p = common / len(pred_chars)
    r = common / len(gold_chars)
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0


# =============================================================================
# Generation kwargs
# =============================================================================
def get_gen_kwargs(method, model_type, no_cg=False):
    block_freq = 16 if model_type == "llada" else 8
    base = {
        "gen_length":             64,   # poems are short
        "block_length":           32,
        "temperature":            0.0,
        "cfg_scale":              0.0,
        "delay_eos_generation":   True,
        "parallel_mode":          False,
        "token_per_step":         1,
        "threshold":              None,
        "print_log":              False,
        "record_time":            False,
        "statistics":             False,
        "sparse_kv":              1.0,
        "delay_step":             -1,
        "top_p":                  0.95,
        "top_k":                  50,
        "ESdLLM_mode":            None,
        "importance_score_alpha": 0.5,
        "proportion_steps":       None,
        "block_update_freq":      None,
        "prompt_update_freq":     None,
        "use_kvcache":            True,
    }
    if method == "nocache":
        base["use_kvcache"] = False
    elif method == "esdllm":
        base.update({"ESdLLM_mode": "HiddenState",
                     "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
                     "block_update_freq": block_freq,
                     "prompt_update_freq": 64})
    elif method in ("caidllm", "caidllm_no_cg"):
        base.update({"ESdLLM_mode": None, "parallel_mode": True,
                     "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
                     "block_update_freq": block_freq,
                     "prompt_update_freq": 64})
    return base

def get_cai_config(no_cg=False):
    return CAIConfig(
        use_apd=True, use_per_block=True,
        use_layer_adaptive=False,
        use_confidence_gating=not no_cg,
    )


# =============================================================================
# Generate one batch
# =============================================================================
def generate_batch(model, tokenizer, prompts, gen_kwargs,
                   method, cai_cfg, eos_id, max_input=256, use_chat_template=False):
    if use_chat_template:
        formatted = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            formatted.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        prompts = formatted
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_input)
    input_ids = enc["input_ids"].cuda()
    attn_mask = enc["attention_mask"].float().cuda()
    plen = input_ids.shape[1]

    with torch.no_grad():
        if method in ("caidllm", "caidllm_no_cg") and CAI_AVAILABLE:
            output, _ = cai_batch_generate(
                model, input_ids, attn_mask,
                generation_kwargs=gen_kwargs,
                cai_config=cai_cfg)
        else:
            output, _ = batch_generate(
                model, input_ids, attn_mask, gen_kwargs)

    results = []
    for i in range(len(prompts)):
        gen_ids = output[i, plen:]
        ep = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(ep): gen_ids = gen_ids[:ep[0]]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # Take only the first line (before any newline or punctuation repeat)
        pred = pred.split('\n')[0].strip()
        results.append(pred)
    return results


# =============================================================================
# Evaluate one method
# =============================================================================
def eval_method(method, items, model, tokenizer, model_type,
                gen_kwargs, cai_cfg, eos_id, batch_size=8, use_chat_template=False):
    forward_em  = []; forward_f1  = []
    reversal_em = []; reversal_f1 = []
    total_time  = 0.0; total_toks  = 0

    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

    for bidx, batch in enumerate(batches):
        # Forward pass
        fwd_prompts = [build_forward_prompt(item["line_a"]) for item in batch]
        t0 = time.time()
        fwd_preds = generate_batch(model, tokenizer, fwd_prompts,
                                   gen_kwargs, method, cai_cfg, eos_id,
                                   use_chat_template=use_chat_template)
        total_time += time.time() - t0
        total_toks += gen_kwargs["gen_length"] * len(batch)

        # Reversal pass
        rev_prompts = [build_reversal_prompt(item["line_b"]) for item in batch]
        t0 = time.time()
        rev_preds = generate_batch(model, tokenizer, rev_prompts,
                                   gen_kwargs, method, cai_cfg, eos_id,
                                   use_chat_template=use_chat_template)
        total_time += time.time() - t0
        total_toks += gen_kwargs["gen_length"] * len(batch)

        for i, item in enumerate(batch):
            forward_em.append(exact_match(fwd_preds[i], item["line_b"]))
            forward_f1.append(char_f1(fwd_preds[i], item["line_b"]))
            reversal_em.append(exact_match(rev_preds[i], item["line_a"]))
            reversal_f1.append(char_f1(rev_preds[i], item["line_a"]))

        if (bidx+1) % 10 == 0 or bidx == 0:
            fwd_acc = np.mean(forward_em)*100
            rev_acc = np.mean(reversal_em)*100
            tps = total_toks / total_time if total_time > 0 else 0
            print(f"  [{method}][{bidx+1}/{len(batches)}] "
                  f"fwd={fwd_acc:.1f}% rev={rev_acc:.1f}% tps={tps:.1f}",
                  flush=True)

    return {
        "forward_em":  round(np.mean(forward_em)*100, 2),
        "forward_f1":  round(np.mean(forward_f1)*100, 2),
        "reversal_em": round(np.mean(reversal_em)*100, 2),
        "reversal_f1": round(np.mean(reversal_f1)*100, 2),
        "tps":         round(total_toks/total_time if total_time > 0 else 0, 1),
        "time_s":      round(total_time, 1),
        "n":           len(items),
    }


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--methods",  type=str,
                   default="nocache,dualcache,esdllm,caidllm,caidllm_no_cg")
    p.add_argument("--n_samples",    type=int, default=496)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--output_dir",   type=str, default="log_results/poem")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_type = MODEL_TYPE[args.model]
    eos_id     = EOS_IDS[model_type]
    methods    = args.methods.split(",")

    print(f"\n{'='*60}", flush=True)
    print(f" Poem Completion — {args.model}", flush=True)
    print(f" Methods: {methods}", flush=True)
    print(f" Samples: {args.n_samples}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load dataset
    items = load_poem_dataset(args.n_samples)
    print(f"Dataset: {len(items)} poem pairs\n", flush=True)

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

    all_results = {}

    for method in methods:
        print(f"\n{'='*50}", flush=True)
        print(f" METHOD: {method}", flush=True)
        print(f"{'='*50}", flush=True)

        no_cg      = (method == "caidllm_no_cg")
        gen_kwargs = get_gen_kwargs(method, model_type, no_cg)
        cai_cfg    = get_cai_config(no_cg) if method in ("caidllm","caidllm_no_cg") else None

        r = eval_method(method, items, model, tokenizer, model_type,
                        gen_kwargs, cai_cfg, eos_id, args.batch_size,
                        use_chat_template=(model_type == 'Dream'))
        all_results[method] = r

        print(f"\n  Forward  EM={r['forward_em']:.2f}%  F1={r['forward_f1']:.2f}%",
              flush=True)
        print(f"  Reversal EM={r['reversal_em']:.2f}%  F1={r['reversal_f1']:.2f}%",
              flush=True)
        print(f"  TPS={r['tps']:.1f}  Time={r['time_s']}s", flush=True)

    # Final summary
    print(f"\n{'='*65}", flush=True)
    print(f" FINAL RESULTS — {args.model}", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'Method':<20} {'Fwd EM':>8} {'Rev EM':>8} {'Fwd F1':>8} {'Rev F1':>8} {'TPS':>8}",
          flush=True)
    print(f"  {'─'*65}", flush=True)
    for method, r in all_results.items():
        print(f"  {method:<20} {r['forward_em']:>7.2f}% {r['reversal_em']:>7.2f}% "
              f"{r['forward_f1']:>7.2f}% {r['reversal_f1']:>7.2f}% {r['tps']:>7.1f}",
              flush=True)

    # Add reference AR model scores from LLaDA paper
    print(f"\n  --- Reference (from LLaDA paper) ---")
    ref = {
        "GPT-4o (2024-08-06)":   {"forward_em": 82.7, "reversal_em": 34.3},
        "Qwen2.5-7B-Instruct":   {"forward_em": 75.9, "reversal_em": 38.0},
        "LLaDA-8B (paper ref)":  {"forward_em": 51.8, "reversal_em": 45.6},
    }
    for name, r in ref.items():
        print(f"  {name:<25} Fwd={r['forward_em']:.1f}%  Rev={r['reversal_em']:.1f}%  "
              f"(paper reference)", flush=True)
    print(f"{'='*65}\n", flush=True)

    # Save
    fname = f"{args.model.replace('-','_')}_poem_completion.json"
    out = {"model": args.model, "results": all_results, "reference": ref}
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved → {os.path.join(args.output_dir, fname)}", flush=True)

if __name__ == "__main__":
    main()
