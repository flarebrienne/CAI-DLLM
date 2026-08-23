#!/usr/bin/env python3
"""
eval_commonsense_enhanced.py
CAI-dLLM + All Enhancements (Ideas 1+3+4+5) on commonsense tasks.
Usage:
  python cai_dllm/eval_commonsense_enhanced.py \
      --model LLaDA-Instruct --tasks piqa --n_samples 500
"""
import os, sys, time, json, argparse
import torch
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from transformers import AutoTokenizer, AutoModel
from cai_generate3 import cai_batch_generate, CAIConfig
from cai_enhancements import EnhancementConfig, PromptKVCache, AdaptiveBlockSizer

MODEL_PATHS = {
    "LLaDA-Instruct": "GSAI-ML/LLaDA-8B-Instruct",
    "Dream-Instruct":  "Dream-org/Dream-v0-Instruct-7B",
}
MODEL_TYPE = {"LLaDA-Instruct": "llada", "Dream-Instruct": "Dream"}
EOS_IDS    = {"llada": 126081, "Dream": 151643}

# =============================================================================
# Dataset loaders
# =============================================================================
def load_winogrande(n):
    from datasets import load_dataset
    ds = load_dataset("winogrande", "winogrande_xl",
                      split="validation", trust_remote_code=True)
    items = []
    for row in list(ds)[:n]:
        items.append({
            "prompt_a": row["sentence"].replace("_", row["option1"]),
            "prompt_b": row["sentence"].replace("_", row["option2"]),
            "label": 0 if row["answer"] == "1" else 1,
        })
    return items

def load_piqa(n):
    from datasets import load_dataset
    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    items = []
    for row in list(ds)[:n]:
        items.append({
            "prompt_a": f"Goal: {row['goal']}\nSolution: {row['sol1']}",
            "prompt_b": f"Goal: {row['goal']}\nSolution: {row['sol2']}",
            "label": row["label"],
        })
    return items

def load_openbookqa(n):
    from datasets import load_dataset
    ds = load_dataset("openbookqa", "main",
                      split="validation", trust_remote_code=True)
    items = []
    for row in list(ds)[:n]:
        label = ord(row["answerKey"]) - ord("A")
        prompts = [f"Question: {row['question_stem']}\nAnswer: {t}"
                   for t in row["choices"]["text"]]
        items.append({"prompts": prompts, "label": label})
    return items

# =============================================================================
# Generation kwargs for CAI-dLLM
# =============================================================================
def get_gen_kwargs(model_type, block_freq):
    return {
        "gen_length": 32, "block_length": 32,
        "temperature": 0.0, "cfg_scale": 0.0,
        "delay_eos_generation": True, "parallel_mode": True,
        "token_per_step": 1, "threshold": None,
        "print_log": False, "record_time": False, "statistics": False,
        "sparse_kv": 1.0, "delay_step": -1,
        "top_p": 0.95, "top_k": 50,
        "ESdLLM_mode": None, "importance_score_alpha": 0.5,
        "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
        "block_update_freq": block_freq,
        "prompt_update_freq": 64,
        "use_kvcache": True,
    }

# =============================================================================
# Score one item
# =============================================================================
def score_item(model, tokenizer, prompts, gen_kwargs,
               cai_cfg, enh_cfg, eos_id):
    scores = []
    for prompt in prompts:
        enc = tokenizer([prompt], return_tensors="pt",
                        truncation=True, max_length=256, padding=True)
        input_ids = enc["input_ids"].cuda()
        attn_mask = enc["attention_mask"].float().cuda()
        plen = input_ids.shape[1]
        try:
            with torch.no_grad():
                output, _ = cai_batch_generate(
                    model, input_ids, attn_mask,
                    generation_kwargs=gen_kwargs,
                    cai_config=cai_cfg,
                    enhancement_config=enh_cfg)
            gen_ids = output[0, plen:]
            ep = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(ep): gen_ids = gen_ids[:ep[0]]
            scores.append(len(gen_ids))
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            scores.append(0)
    return int(np.argmax(scores))

# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--tasks",    type=str, default="piqa")
    p.add_argument("--n_samples",type=int, default=500)
    p.add_argument("--block_update_freq", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="log_results/commonsense")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_type = MODEL_TYPE[args.model]
    eos_id     = EOS_IDS[model_type]
    task_names = args.tasks.split(",")
    block_freq = args.block_update_freq or (16 if model_type == "llada" else 8)

    print(f"\n{'='*60}", flush=True)
    print(f" CAI-dLLM + All Enhancements (Ideas 1+3+4+5)", flush=True)
    print(f" Model: {args.model}  Tasks: {task_names}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load datasets
    loaders = {"winogrande": load_winogrande,
               "piqa": load_piqa,
               "openbookqa": load_openbookqa}
    datasets = {t: loaders[t](args.n_samples) for t in task_names}

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

    # Build configs — All Enhancements (1+3+4+5)
    gen_kwargs = get_gen_kwargs(model_type, block_freq)
    cai_cfg    = CAIConfig(use_apd=True, use_per_block=True,
                           use_layer_adaptive=False, use_confidence_gating=True)
    enh_cfg    = EnhancementConfig(
        use_adaptive_block=True,
        use_soft_belief=True,
        use_oracle_budget=True,
        use_prompt_kvcache=True,
        prompt_cache=PromptKVCache(),
    )

    all_results = {}

    for task_name in task_names:
        items = datasets[task_name]
        correct = 0
        total_time = 0.0
        total_toks = 0
        print(f"\n── {task_name} ({len(items)} samples) ──", flush=True)

        for i, item in enumerate(items):
            prompts = item.get("prompts", [item["prompt_a"], item["prompt_b"]])
            t0 = time.time()
            pred = score_item(model, tokenizer, prompts, gen_kwargs,
                              cai_cfg, enh_cfg, eos_id)
            elapsed = time.time() - t0
            total_time += elapsed
            total_toks += gen_kwargs["gen_length"] * len(prompts)
            if pred == item["label"]:
                correct += 1

            if (i+1) % 50 == 0 or i == 0:
                acc = correct / (i+1) * 100
                tps = total_toks / total_time if total_time > 0 else 0
                print(f"  [{task_name}] [{i+1}/{len(items)}] "
                      f"acc={acc:.1f}%  tps={tps:.1f}  last={elapsed:.1f}s",
                      flush=True)

        acc = correct / len(items) * 100
        tps = total_toks / total_time if total_time > 0 else 0
        all_results[task_name] = {"acc": round(acc,2), "tps": round(tps,1)}
        print(f"  DONE {task_name}: acc={acc:.2f}%  tps={tps:.1f}", flush=True)

    # Summary
    avg_acc = np.mean([v["acc"] for v in all_results.values()])
    avg_tps = np.mean([v["tps"] for v in all_results.values()])
    print(f"\n{'='*60}", flush=True)
    print(f" RESULTS — {args.model} | CAI + All (1+3+4+5)", flush=True)
    print(f"{'='*60}", flush=True)
    for t, v in all_results.items():
        print(f"  {t:20s}: acc={v['acc']:.2f}%  tps={v['tps']:.1f}", flush=True)
    print(f"  {'avg':20s}: acc={avg_acc:.2f}%  tps={avg_tps:.1f}", flush=True)
    print(f"{'='*60}\n", flush=True)

    fname = f"{args.model.replace('-','_')}_{'_'.join(task_names)}_enhanced.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump({"model": args.model, "method": "caidllm_all_enhancements",
                   "results": all_results}, f, indent=2)
    print(f"Saved → {os.path.join(args.output_dir, fname)}", flush=True)

if __name__ == "__main__":
    main()
