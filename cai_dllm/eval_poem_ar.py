#!/usr/bin/env python3
"""
eval_poem_ar.py — Autoregressive Model Evaluation for Poem Completion
=====================================================================
Tests GPT-4o and Qwen2.5-7B-Instruct on Chinese poem completion task.
Follows the exact protocol from the LLaDA paper.

Usage:
  # GPT-4o (requires OPENAI_API_KEY)
  python cai_dllm/eval_poem_ar.py --model gpt4o --n_samples 496

  # Qwen2.5-7B local
  python cai_dllm/eval_poem_ar.py --model qwen --n_samples 496

  # Both
  python cai_dllm/eval_poem_ar.py --model all --n_samples 496
"""

import os, sys, json, time, re, argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from poem_data import get_poem_pairs

# =============================================================================
# Prompt builders
# =============================================================================
def build_forward_prompt(line_a):
    return (f"以下是一首古诗的连续两句。请只输出第二句，不要输出任何其他内容。\n\n"
            f"第一句：{line_a}\n"
            f"第二句：")

def build_reversal_prompt(line_b):
    return (f"以下是一首古诗的连续两句。请只输出第一句，不要输出任何其他内容。\n\n"
            f"第二句：{line_b}\n"
            f"第一句：")

# =============================================================================
# Scoring
# =============================================================================
def clean(s):
    return re.sub(r'[，。！？；、\s第一句第二句：]', '', s.strip())

def exact_match(pred, gold):
    return int(clean(pred) == clean(gold))

def char_f1(pred, gold):
    pred_c = list(clean(pred)); gold_c = list(clean(gold))
    if not pred_c or not gold_c: return 0.0
    common = sum(min(pred_c.count(c), gold_c.count(c)) for c in set(pred_c))
    p = common/len(pred_c); r = common/len(gold_c)
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

# =============================================================================
# GPT-4o evaluation
# =============================================================================
def eval_gpt4o(items, n_samples, api_key=None):
    try:
        from openai import OpenAI
    except ImportError:
        print("Install openai: pip install openai --break-system-packages")
        return None

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Set OPENAI_API_KEY environment variable")
        return None

    client = OpenAI(api_key=key)
    forward_em = []; reversal_em = []
    forward_f1 = []; reversal_f1 = []

    print(f"\n[GPT-4o] Evaluating {n_samples} pairs...", flush=True)

    for i, item in enumerate(items[:n_samples]):
        # Forward
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": build_forward_prompt(item["line_a"])}],
                max_tokens=32, temperature=0.0)
            fwd_pred = resp.choices[0].message.content.strip().split('\n')[0]
        except Exception as e:
            print(f"  GPT error: {e}"); fwd_pred = ""

        # Reversal
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": build_reversal_prompt(item["line_b"])}],
                max_tokens=32, temperature=0.0)
            rev_pred = resp.choices[0].message.content.strip().split('\n')[0]
        except Exception as e:
            print(f"  GPT error: {e}"); rev_pred = ""

        forward_em.append(exact_match(fwd_pred, item["line_b"]))
        reversal_em.append(exact_match(rev_pred, item["line_a"]))
        forward_f1.append(char_f1(fwd_pred, item["line_b"]))
        reversal_f1.append(char_f1(rev_pred, item["line_a"]))

        if (i+1) % 50 == 0 or i == 0:
            print(f"  [GPT-4o][{i+1}/{n_samples}] "
                  f"fwd={np.mean(forward_em)*100:.1f}% "
                  f"rev={np.mean(reversal_em)*100:.1f}%", flush=True)

        time.sleep(0.5)  # rate limit

    return {
        "forward_em":  round(np.mean(forward_em)*100, 2),
        "reversal_em": round(np.mean(reversal_em)*100, 2),
        "forward_f1":  round(np.mean(forward_f1)*100, 2),
        "reversal_f1": round(np.mean(reversal_f1)*100, 2),
    }

# =============================================================================
# Qwen2.5-7B local evaluation
# =============================================================================
def eval_qwen(items, n_samples, model_path="Qwen/Qwen2.5-7B-Instruct"):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("transformers not available"); return None

    print(f"\n[Qwen] Loading {model_path}...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        print("[Qwen] Model loaded.", flush=True)
    except Exception as e:
        print(f"Failed to load Qwen: {e}"); return None

    forward_em = []; reversal_em = []
    forward_f1 = []; reversal_f1 = []
    total_time = 0.0

    for i, item in enumerate(items[:n_samples]):
        for task, prompt, gold in [
            ("fwd", build_forward_prompt(item["line_a"]), item["line_b"]),
            ("rev", build_reversal_prompt(item["line_b"]), item["line_a"]),
        ]:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer([text], return_tensors="pt").to("cuda")
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=32,
                    do_sample=False, temperature=None, top_p=None)
            total_time += time.time() - t0
            pred = tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:],
                skip_special_tokens=True).strip().split('\n')[0]

            if task == "fwd":
                forward_em.append(exact_match(pred, gold))
                forward_f1.append(char_f1(pred, gold))
            else:
                reversal_em.append(exact_match(pred, gold))
                reversal_f1.append(char_f1(pred, gold))

        if (i+1) % 50 == 0 or i == 0:
            print(f"  [Qwen][{i+1}/{n_samples}] "
                  f"fwd={np.mean(forward_em)*100:.1f}% "
                  f"rev={np.mean(reversal_em)*100:.1f}%", flush=True)

    return {
        "forward_em":  round(np.mean(forward_em)*100, 2),
        "reversal_em": round(np.mean(reversal_em)*100, 2),
        "forward_f1":  round(np.mean(forward_f1)*100, 2),
        "reversal_f1": round(np.mean(reversal_f1)*100, 2),
        "time_s":      round(total_time, 1),
    }

# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["gpt4o", "qwen", "all"],
                   help="Which AR model to evaluate")
    p.add_argument("--n_samples", type=int, default=496)
    p.add_argument("--qwen_path", type=str,
                   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--openai_key", type=str, default=None,
                   help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--output_dir", type=str, default="log_results/poem")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    items = get_poem_pairs(args.n_samples)
    print(f"Dataset: {len(items)} poem pairs\n", flush=True)

    results = {}

    if args.model in ("gpt4o", "all"):
        r = eval_gpt4o(items, args.n_samples, args.openai_key)
        if r:
            results["GPT-4o-2024-08-06"] = r
            print(f"\n[GPT-4o] Fwd={r['forward_em']:.2f}% "
                  f"Rev={r['reversal_em']:.2f}%", flush=True)

    if args.model in ("qwen", "all"):
        r = eval_qwen(items, args.n_samples, args.qwen_path)
        if r:
            results["Qwen2.5-7B-Instruct"] = r
            print(f"\n[Qwen] Fwd={r['forward_em']:.2f}% "
                  f"Rev={r['reversal_em']:.2f}%", flush=True)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print(f" AR Model Results — Poem Completion ({args.n_samples} pairs)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Model':<25} {'Fwd EM':>8} {'Rev EM':>8} {'Fwd F1':>8} {'Rev F1':>8}",
          flush=True)
    print(f"  {'─'*60}", flush=True)
    for name, r in results.items():
        print(f"  {name:<25} {r['forward_em']:>7.2f}% "
              f"{r['reversal_em']:>7.2f}% "
              f"{r['forward_f1']:>7.2f}% "
              f"{r['reversal_f1']:>7.2f}%", flush=True)

    # Paper reference
    print(f"\n  --- Paper Reference ---")
    print(f"  {'GPT-4o (paper)':<25} {'82.7%':>8} {'34.3%':>8}")
    print(f"  {'Qwen2.5-7B (paper)':<25} {'75.9%':>8} {'38.0%':>8}")
    print(f"{'='*60}\n", flush=True)

    fname = f"ar_models_poem_{args.n_samples}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved → {os.path.join(args.output_dir, fname)}", flush=True)

if __name__ == "__main__":
    main()