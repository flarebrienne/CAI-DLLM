#!/usr/bin/env python3
"""
eval_commonsense.py — Commonsense Benchmarks (WinoGrande, PIQA, OpenBookQA)
Evaluates nocache, dualcache, esdllm, caidllm on 500 samples per task.
Usage:
  python cai_dllm/eval_commonsense.py --model LLaDA-Instruct --tasks winogrande
  python cai_dllm/eval_commonsense.py --model Dream-Instruct --tasks piqa
"""
import os, sys, time, json, argparse
import torch
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

print("Importing generate...", flush=True)
from generate import batch_generate
print("Importing transformers...", flush=True)
from transformers import AutoTokenizer, AutoModel
print("Imports done.", flush=True)

try:
    from cai_generate3 import cai_batch_generate, CAIConfig
    CAI_AVAILABLE = True
    print("CAI modules loaded (cai_generate3).", flush=True)
except ImportError:
    try:
        from cai_generate import cai_batch_generate, CAIConfig
        CAI_AVAILABLE = True
        print("CAI modules loaded (cai_generate).", flush=True)
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
# Dataset loaders
# =============================================================================
def load_winogrande(n=500):
    print(f"Loading winogrande ({n} samples)...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("winogrande", "winogrande_xl",
                      split="validation", trust_remote_code=True)
    items = []
    for row in list(ds)[:n]:
        items.append({
            "prompt_a": row["sentence"].replace("_", row["option1"]),
            "prompt_b": row["sentence"].replace("_", row["option2"]),
            "label": 0 if row["answer"] == "1" else 1,
            "n_choices": 2
        })
    print(f"  Loaded {len(items)} winogrande samples.", flush=True)
    return items

def load_piqa(n=500):
    print(f"Loading piqa ({n} samples)...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    items = []
    for row in list(ds)[:n]:
        items.append({
            "prompt_a": f"Goal: {row['goal']}\nSolution: {row['sol1']}",
            "prompt_b": f"Goal: {row['goal']}\nSolution: {row['sol2']}",
            "label": row["label"],
            "n_choices": 2
        })
    print(f"  Loaded {len(items)} piqa samples.", flush=True)
    return items

def load_openbookqa(n=500):
    print(f"Loading openbookqa ({n} samples)...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("openbookqa", "main",
                      split="validation", trust_remote_code=True)
    items = []
    for row in list(ds)[:n]:
        label = ord(row["answerKey"]) - ord("A")
        prompts = [f"Question: {row['question_stem']}\nAnswer: {t}"
                   for t in row["choices"]["text"]]
        items.append({"prompts": prompts, "label": label, "n_choices": 4})
    print(f"  Loaded {len(items)} openbookqa samples.", flush=True)
    return items

# =============================================================================
# Generation kwargs
# =============================================================================
def get_gen_kwargs(method, model_type):
    block_freq = 16 if model_type == "llada" else 8
    base = {
        "gen_length": 32, "block_length": 32,
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
                     "block_update_freq": block_freq,
                     "prompt_update_freq": 64})
    elif method == "caidllm":
        base.update({"ESdLLM_mode": None, "parallel_mode": True,
                     "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
                     "block_update_freq": block_freq,
                     "prompt_update_freq": 64})
    return base

def get_cai_config(no_cg=False):
    return CAIConfig(use_apd=True, use_per_block=True,
                     use_layer_adaptive=False, use_confidence_gating=not no_cg)

# =============================================================================
# Score one item — generate for each option, pick longest/highest confidence
# =============================================================================
def score_item(model, tokenizer, prompts, gen_kwargs, method, cai_cfg, eos_id):
    scores = []
    for prompt in prompts:
        enc = tokenizer([prompt], return_tensors="pt", truncation=True,
                        max_length=256, padding=True)
        input_ids = enc["input_ids"].cuda()
        attn_mask = enc["attention_mask"].float().cuda()
        plen = input_ids.shape[1]
        try:
            with torch.no_grad():
                if method == "caidllm" and CAI_AVAILABLE:
                    output, _ = cai_batch_generate(
                        model, input_ids, attn_mask,
                        generation_kwargs=gen_kwargs, cai_config=cai_cfg)
                else:
                    output, _ = batch_generate(
                        model, input_ids, attn_mask, gen_kwargs)
            gen_ids = output[0, plen:]
            ep = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(ep): gen_ids = gen_ids[:ep[0]]
            scores.append(len(gen_ids))
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            scores.append(0)
    return int(np.argmax(scores))

# =============================================================================
# Evaluate one method on one task
# =============================================================================
def eval_method_task(task_name, items, model, tokenizer, model_type,
                     method, gen_kwargs, cai_cfg, eos_id):
    correct = 0
    total_time = 0.0
    total_toks = 0
    print(f"\n  [{method}] {task_name} — {len(items)} samples", flush=True)

    for i, item in enumerate(items):
        prompts = item.get("prompts", [item["prompt_a"], item["prompt_b"]])
        t0 = time.time()
        pred = score_item(model, tokenizer, prompts, gen_kwargs,
                          method, cai_cfg, eos_id)
        elapsed = time.time() - t0
        total_time += elapsed
        total_toks += gen_kwargs["gen_length"] * len(prompts)

        if pred == item["label"]:
            correct += 1

        if (i+1) % 50 == 0 or i == 0:
            acc = correct / (i+1) * 100
            tps = total_toks / total_time if total_time > 0 else 0
            print(f"  [{method}][{task_name}] [{i+1}/{len(items)}] "
                  f"acc={acc:.1f}%  tps={tps:.1f}  last={elapsed:.1f}s",
                  flush=True)

    acc = correct / len(items) * 100
    tps = total_toks / total_time if total_time > 0 else 0
    print(f"  DONE [{method}][{task_name}]: acc={acc:.2f}%  tps={tps:.1f}",
          flush=True)
    return acc, tps

# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--methods", type=str, default="nocache,dualcache,esdllm,caidllm")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--tasks",   type=str, default="winogrande,piqa,openbookqa")
    p.add_argument("--no_confidence_gating", action="store_true", default=False)
    p.add_argument("--output_dir", type=str, default="log_results/commonsense")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_type = MODEL_TYPE[args.model]
    eos_id     = EOS_IDS[model_type]
    methods    = args.methods.split(",")
    task_names = args.tasks.split(",")

    print(f"\n{'='*60}", flush=True)
    print(f" Commonsense — {args.model}", flush=True)
    print(f" Methods: {methods}", flush=True)
    print(f" Tasks:   {task_names}", flush=True)
    print(f" Samples: {args.n_samples}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load datasets
    loaders = {"winogrande": load_winogrande,
               "piqa": load_piqa,
               "openbookqa": load_openbookqa}
    datasets = {t: loaders[t](args.n_samples) for t in task_names}

    # Load model once
    print("\nLoading model...", flush=True)
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
        gen_kwargs = get_gen_kwargs(method, model_type)
        no_cg = getattr(args, 'no_confidence_gating', False)
        cai_cfg    = get_cai_config(no_cg) if method == "caidllm" else None
        method_res = {}

        for task_name in task_names:
            acc, tps = eval_method_task(
                task_name, datasets[task_name], model, tokenizer,
                model_type, method, gen_kwargs, cai_cfg, eos_id)
            method_res[task_name] = {"acc": round(acc,2), "tps": round(tps,1)}

        avg = np.mean([v["acc"] for v in method_res.values()])
        method_res["avg_acc"] = round(avg, 2)
        all_results[method] = method_res
        print(f"\n  [{method}] avg_acc={avg:.2f}%", flush=True)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print(f" RESULTS — {args.model}", flush=True)
    print(f"{'='*60}", flush=True)
    header = f"  {'Method':<15}"
    for t in task_names: header += f" {t:>12}"
    header += f" {'Avg':>8} {'TPS':>8}"
    print(header, flush=True)
    print(f"  {'─'*70}", flush=True)
    for method in methods:
        r = all_results[method]
        row = f"  {method:<15}"
        for t in task_names:
            row += f" {r[t]['acc']:>11.2f}%"
        row += f" {r['avg_acc']:>7.2f}%"
        avg_tps = np.mean([r[t]['tps'] for t in task_names])
        row += f" {avg_tps:>7.1f}"
        print(row, flush=True)
    print(f"{'='*60}\n", flush=True)

    fname = f"{args.model.replace('-','_')}_{'_'.join(task_names)}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump({"model": args.model, "results": all_results}, f, indent=2)
    print(f"Saved → {os.path.join(args.output_dir, fname)}", flush=True)

if __name__ == "__main__":
    main()
