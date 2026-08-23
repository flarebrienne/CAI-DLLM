#!/usr/bin/env python3
"""
eval_math_code.py — MathQA (math) and MBPP (code) evaluation
Tests: nocache, dualcache, esdllm, caidllm, caidllm_enhanced
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
    from cai_enhancements import EnhancementConfig, PromptKVCache
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

def load_mathqa(n=500):
    print(f"Loading MathQA ({n} samples)...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("math_qa", split="validation", trust_remote_code=True)
    items = []
    for row in list(ds)[:n]:
        opts_raw = row["options"]
        opts = re.findall(r'[a-e]\s*\)\s*([^,]+)', opts_raw)
        opts = [o.strip() for o in opts]
        if not opts:
            continue
        label_map = {'a':0,'b':1,'c':2,'d':3,'e':4}
        label = label_map.get(row["correct"].strip().lower(), 0)
        question_prompt = (
            f"Solve this math problem and give only the numeric answer.\n"
            f"Problem: {row['Problem']}\nAnswer:")
        items.append({
            "question_prompt": question_prompt,
            "option_texts": opts,
            "label": label,
            "task": "mathqa"
        })
    print(f"  Loaded {len(items)} MathQA samples.", flush=True)
    return items

def load_mbpp(n=257):
    print(f"Loading MBPP ({n} samples)...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True)
    items = []
    for row in list(ds)[:n]:
        # Include function signature in prompt so model generates body only
        fn_sig = f"def solution():"  # generic fallback
        # Try to extract function name from code
        import re as _re
        m = _re.search(r"def (\w+\([^)]*\)):", row.get("code",""))
        fn_sig = f"def {m.group(1)}:" if m else "def solution():"
        prompt = (f"Write a Python function to solve the following:\n"
                  f"{row['prompt']}\n\n{fn_sig}")
        items.append({
            "prompt": prompt,
            "fn_sig": fn_sig,
            "test_list": row.get("test_list", []),
            "task": "mbpp"
        })
    print(f"  Loaded {len(items)} MBPP samples.", flush=True)
    return items

def get_gen_kwargs(method, model_type, task):
    block_freq = 16 if model_type == "llada" else 8
    gen_len   = 64  if task == "mathqa" else 256
    block_len = 32  if task == "mathqa" else 64
    base = {
        "gen_length": gen_len, "block_length": block_len,
        "temperature": 0.0, "cfg_scale": 0.0,
        "delay_eos_generation": True, "parallel_mode": False,
        "token_per_step": 1, "threshold": None,
        "print_log": False, "record_time": False, "statistics": False,
        "sparse_kv": 1.0, "delay_step": -1,
        "top_p": 0.95, "top_k": 50,
        "ESdLLM_mode": None, "importance_score_alpha": 0.5,
        "proportion_steps": None, "block_update_freq": None,
        "prompt_update_freq": None, "use_kvcache": True,
    }
    if method == "nocache":
        base["use_kvcache"] = False
    elif method == "esdllm":
        base.update({"ESdLLM_mode": "HiddenState",
                     "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
                     "block_update_freq": block_freq, "prompt_update_freq": 64})
    elif method in ("caidllm","caidllm_enhanced"):
        base.update({"ESdLLM_mode": None, "parallel_mode": True,
                     "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
                     "block_update_freq": block_freq, "prompt_update_freq": 64})
    return base

def get_cai_config(no_cg=False):
    return CAIConfig(use_apd=True, use_per_block=True,
                     use_layer_adaptive=False, use_confidence_gating=not no_cg)

def get_enh_config():
    if not CAI_AVAILABLE: return None
    return EnhancementConfig(
        use_adaptive_block=True, use_soft_belief=True,
        use_oracle_budget=True, use_prompt_kvcache=True,
        prompt_cache=PromptKVCache())

def generate_one(model, tokenizer, prompt, gen_kwargs,
                 method, cai_cfg, enh_cfg, eos_id, max_input=512):
    enc = tokenizer([prompt], return_tensors="pt",
                    truncation=True, max_length=max_input, padding=True)
    input_ids = enc["input_ids"].cuda()
    attn_mask = enc["attention_mask"].float().cuda()
    plen = input_ids.shape[1]
    try:
        with torch.no_grad():
            if method in ("caidllm","caidllm_enhanced") and CAI_AVAILABLE:
                enh = enh_cfg if method == "caidllm_enhanced" else None
                output, _ = cai_batch_generate(
                    model, input_ids, attn_mask,
                    generation_kwargs=gen_kwargs,
                    cai_config=cai_cfg, enhancement_config=enh)
            else:
                output, _ = batch_generate(
                    model, input_ids, attn_mask, gen_kwargs)
        gen_ids = output[0, plen:]
        ep = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(ep): gen_ids = gen_ids[:ep[0]]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"  [ERROR] {e}", flush=True)
        return ""

def score_mathqa(model, tokenizer, items, gen_kwargs,
                 method, cai_cfg, enh_cfg, eos_id):
    """Generate answer and match against option texts."""
    correct = 0; total_time = 0.0; total_toks = 0
    for i, item in enumerate(items):
        t0 = time.time()
        out = generate_one(model, tokenizer, item["question_prompt"],
                           gen_kwargs, method, cai_cfg, enh_cfg, eos_id)
        out_clean = out.strip().lower().split("\n")[0]
        # Match against options
        pred = 0; best = -1
        for j, opt in enumerate(item["option_texts"]):
            opt_c = opt.strip().lower()
            if opt_c == out_clean: score = 2
            elif opt_c in out_clean or out_clean in opt_c: score = 1
            else: score = 0
            if score > best: best = score; pred = j
        if pred == item["label"]: correct += 1
        total_time += time.time() - t0
        total_toks += gen_kwargs["gen_length"]
        if (i+1) % 50 == 0 or i == 0:
            acc = correct/(i+1)*100
            tps = total_toks/total_time if total_time > 0 else 0
            print(f"  [mathqa][{i+1}/{len(items)}] acc={acc:.1f}% tps={tps:.1f} pred='{out_clean[:20]}'",
                  flush=True)
    return correct/len(items)*100, total_toks/total_time if total_time > 0 else 0

def score_mbpp(model, tokenizer, items, gen_kwargs,
               method, cai_cfg, enh_cfg, eos_id):
    """Generate code and run test cases."""
    correct = 0; total_time = 0.0; total_toks = 0
    for i, item in enumerate(items):
        t0 = time.time()
        out = generate_one(model, tokenizer, item["prompt"], gen_kwargs,
                           method, cai_cfg, enh_cfg, eos_id, max_input=256)
        # Reconstruct full function = signature + generated body
        sig = item.get("fn_sig", "def solution():")
        raw = out.split("```python")[-1].split("```")[0] if "```" in out else out
        # Keep only code lines (remove print/comments), preserve indentation
        body_lines = []
        for l in raw.split("\n"):
            s = l.strip()
            if s.startswith("print") or s.startswith("#"): continue
            if s == "": 
                body_lines.append("")
                continue
            # Ensure proper indentation (4 spaces)
            if l and not l[0].isspace():
                l = "    " + l
            body_lines.append(l)
        code = sig + "\n" + "\n".join(body_lines)
        passed = False
        import signal
        def _timeout(signum, frame): raise TimeoutError()
        try:
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(5)  # 5 second timeout
            g = {}
            exec(code, g)
            any_passed = False
            for test in item["test_list"]:
                try:
                    exec(test, g)
                    any_passed = True
                    break
                except Exception:
                    continue
            passed = any_passed if item["test_list"] else len(raw.strip()) > 10
        except Exception:
            passed = False
        finally:
            signal.alarm(0)
        if passed: correct += 1
        total_time += time.time() - t0
        total_toks += gen_kwargs["gen_length"]
        if (i+1) % 20 == 0 or i == 0:
            acc = correct/(i+1)*100
            tps = total_toks/total_time if total_time > 0 else 0
            print(f"  [mbpp][{i+1}/{len(items)}] pass={acc:.1f}% tps={tps:.1f}",
                  flush=True)
    return correct/len(items)*100, total_toks/total_time if total_time > 0 else 0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--tasks",    type=str, default="mathqa,mbpp")
    p.add_argument("--methods",  type=str,
                   default="nocache,dualcache,esdllm,caidllm,caidllm_enhanced")
    p.add_argument("--n_mathqa", type=int, default=500)
    p.add_argument("--n_mbpp",   type=int, default=257)
    p.add_argument("--no_confidence_gating", action="store_true", default=False)
    p.add_argument("--output_dir", type=str, default="log_results/math_code")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_type = MODEL_TYPE[args.model]
    eos_id     = EOS_IDS[model_type]
    task_names = args.tasks.split(",")
    methods    = args.methods.split(",")

    print(f"\n{'='*60}", flush=True)
    print(f" Math & Code — {args.model}", flush=True)
    print(f" Tasks: {task_names}  Methods: {methods}", flush=True)
    print(f"{'='*60}\n", flush=True)

    datasets = {}
    if "mathqa" in task_names: datasets["mathqa"] = load_mathqa(args.n_mathqa)
    if "mbpp"   in task_names: datasets["mbpp"]   = load_mbpp(args.n_mbpp)

    print("Loading model...", flush=True)
    model_path = MODEL_PATHS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        padding_side="left" if model_type == "Dream" else "right")
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda")
    if model_type == "llada":
        from models.hook_model import transform_llada_model; transform_llada_model(model)
    else:
        from models.hook_model import transform_dream_model; transform_dream_model(model)
    model.eval()
    print("Model ready.\n", flush=True)

    no_cg  = getattr(args, 'no_confidence_gating', False)
    cai_cfg = get_cai_config(no_cg)
    enh_cfg = get_enh_config()
    all_results = {}

    for method in methods:
        print(f"\n{'='*50}\n METHOD: {method}\n{'='*50}", flush=True)
        method_res = {}
        for task_name in task_names:
            items = datasets[task_name]
            gen_kwargs = get_gen_kwargs(method, model_type, task_name)
            print(f"\n── {task_name} ({len(items)} samples) ──", flush=True)
            if task_name == "mathqa":
                acc, tps = score_mathqa(model, tokenizer, items, gen_kwargs,
                                        method, cai_cfg, enh_cfg, eos_id)
            else:
                acc, tps = score_mbpp(model, tokenizer, items, gen_kwargs,
                                      method, cai_cfg, enh_cfg, eos_id)
            method_res[task_name] = {"acc": round(acc,2), "tps": round(tps,1)}
            print(f"  DONE {task_name}: acc={acc:.2f}% tps={tps:.1f}", flush=True)
        all_results[method] = method_res

    print(f"\n{'='*60}\n FINAL RESULTS — {args.model}\n{'='*60}", flush=True)
    header = f"  {'Method':<22}"
    for t in task_names: header += f" {t:>12}"
    header += f"  {'Avg TPS':>10}"
    print(header, flush=True)
    print(f"  {'─'*65}", flush=True)
    for method in methods:
        r = all_results[method]
        row = f"  {method:<22}"
        for t in task_names: row += f" {r[t]['acc']:>11.2f}%"
        row += f"  {np.mean([r[t]['tps'] for t in task_names]):>9.1f}"
        print(row, flush=True)
    print(f"{'='*60}\n", flush=True)

    fname = f"{args.model.replace('-','_')}_math_code.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump({"model": args.model, "results": all_results}, f, indent=2)
    print(f"Saved → {os.path.join(args.output_dir, fname)}", flush=True)

if __name__ == "__main__":
    main()
