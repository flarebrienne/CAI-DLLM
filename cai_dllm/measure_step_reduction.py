#!/usr/bin/env python3
"""
measure_step_reduction.py
Measures average denoising steps per request for ES-dLLM vs CAI-dLLM
on GSM8K, HumanEval, BBH for both LLaDA and Dream.

Approach: hook into the generation loop to count actual steps taken.
"""

import os, sys, time, json, argparse
import torch
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from transformers import AutoTokenizer, AutoModel

MODEL_PATHS = {
    "LLaDA-Instruct": "GSAI-ML/LLaDA-8B-Instruct",
    "Dream-Instruct":  "Dream-org/Dream-v0-Instruct-7B",
}
MODEL_TYPE = {"LLaDA-Instruct": "llada", "Dream-Instruct": "Dream"}
EOS_IDS    = {"llada": 126081, "Dream": 151643}

TASK_SETTINGS = {
    "gsm8k":     {"gen_length": 256, "block_length": 64, "max_input": 256},
    "humaneval": {"gen_length": 512, "block_length": 64, "max_input": 512},
    "bbh":       {"gen_length": 256, "block_length": 64, "max_input": 256},
}

# Sample prompts per task for step measurement (10 samples each)
SAMPLE_PROMPTS = {
    "gsm8k": [
        "Janet sells 16 duck eggs a day. She eats 3 for breakfast every morning. Answer:",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. Answer:",
        "Josh decides to try flipping a house. He buys a house for $80,000. Answer:",
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. Answer:",
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. Answer:",
        "Kylar went to the store to buy glasses for his new apartment. Answer:",
        "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. Answer:",
        "Carla is downloading a 200 GB file. She gets 30% of it downloaded in the first hour. Answer:",
        "John writes 20 pages a day. How long will it take him to write 3 books? Answer:",
        "A car is driving at 60 km/h. How far does it travel in 2.5 hours? Answer:",
    ],
    "humaneval": [
        "def has_close_elements(numbers, threshold):\n    \"\"\" Check if list has close elements \"\"\"\n",
        "def separate_paren_groups(paren_string):\n    \"\"\" Separate parenthesis groups \"\"\"\n",
        "def truncate_number(number):\n    \"\"\" Return decimal part of number \"\"\"\n",
        "def below_zero(operations):\n    \"\"\" Check if balance goes below zero \"\"\"\n",
        "def mean_absolute_deviation(numbers):\n    \"\"\" Calculate mean absolute deviation \"\"\"\n",
        "def intersperse(numbers, delimiter):\n    \"\"\" Intersperse delimiter between elements \"\"\"\n",
        "def parse_nested_parens(paren_string):\n    \"\"\" Parse nested parentheses depth \"\"\"\n",
        "def filter_by_substring(strings, substring):\n    \"\"\" Filter strings by substring \"\"\"\n",
        "def sum_product(numbers):\n    \"\"\" Return sum and product of numbers \"\"\"\n",
        "def rolling_max(numbers):\n    \"\"\" Return running maximum of list \"\"\"\n",
    ],
    "bbh": [
        "Q: Which of the following is a humorous edit of a movie title?\nA:",
        "Q: Is the following sentence grammatically correct?\nA:",
        "Q: What is the next letter in the sequence: A, C, E, G?\nA:",
        "Q: Solve the logic puzzle: All cats are animals. Some animals are dogs. Therefore?\nA:",
        "Q: What comes next in the pattern: 2, 4, 8, 16?\nA:",
        "Q: Translate to formal English: gonna go to the store\nA:",
        "Q: Is this statement true or false: All prime numbers are odd.\nA:",
        "Q: What is the opposite of 'begin'?\nA:",
        "Q: Sort these words alphabetically: banana, apple, cherry\nA:",
        "Q: Complete the analogy: Hot is to Cold as Day is to?\nA:",
    ],
}

def count_steps_esdllm(model, tokenizer, prompts, gen_length,
                        block_length, model_type, block_freq):
    """Count steps for ES-dLLM — always uses full budget."""
    n_blocks = gen_length // block_length
    steps_per_block = block_length  # fixed budget
    total_steps = n_blocks * steps_per_block
    # ES-dLLM always uses full denoising budget per block
    return total_steps, total_steps

def count_steps_caidllm(model, tokenizer, prompts, gen_length,
                         block_length, model_type, block_freq,
                         no_cg=False):
    """Count actual steps used by CAI-dLLM via the results dict."""
    from cai_generate3 import cai_batch_generate, CAIConfig

    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512)
    input_ids = enc["input_ids"].cuda()
    attn_mask = enc["attention_mask"].float().cuda()

    gen_kwargs = {
        "gen_length": gen_length, "block_length": block_length,
        "temperature": 0.0, "cfg_scale": 0.0, "use_kvcache": True,
        "parallel_mode": True, "token_per_step": 1, "threshold": None,
        "print_log": False, "record_time": False, "statistics": True,
        "delay_eos_generation": True, "sparse_kv": 1.0, "delay_step": -1,
        "top_p": 0.95, "top_k": 50,
        "block_update_freq": block_freq, "prompt_update_freq": 64,
        "ESdLLM_mode": None, "importance_score_alpha": 0.5,
        "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
    }

    cai_cfg = CAIConfig(
        use_apd=True, use_per_block=True,
        use_layer_adaptive=False,
        use_confidence_gating=not no_cg,
    )

    with torch.no_grad():
        output, results = cai_batch_generate(
            model, input_ids, attn_mask,
            generation_kwargs=gen_kwargs,
            cai_config=cai_cfg)

    # Sum block steps from results
    block_steps = results.get("block_steps", [])
    total_steps = sum(block_steps) if block_steps else gen_length
    max_possible = gen_length  # full budget
    return total_steps, max_possible

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--tasks",   type=str, default="gsm8k,humaneval,bbh")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="log_results/step_reduction")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_type = MODEL_TYPE[args.model]
    task_names = args.tasks.split(",")
    block_freq = 16 if model_type == "llada" else 8

    print(f"\n{'='*60}", flush=True)
    print(f" Step Reduction Measurement — {args.model}", flush=True)
    print(f" Tasks: {task_names}", flush=True)
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

    all_results = {}

    for task in task_names:
        settings = TASK_SETTINGS[task]
        gen_len   = settings["gen_length"]
        block_len = settings["block_length"]
        prompts   = SAMPLE_PROMPTS[task][:args.n_samples]

        print(f"\n── {task} (gen={gen_len}, block={block_len}) ──", flush=True)

        # ES-dLLM — fixed budget
        es_steps = gen_len  # always full budget
        print(f"  ES-dLLM:        {es_steps} steps (fixed budget)", flush=True)

        # CAI-dLLM
        try:
            cai_steps, max_steps = count_steps_caidllm(
                model, tokenizer, prompts, gen_len, block_len,
                model_type, block_freq, no_cg=False)
            cai_avg = cai_steps / (gen_len // block_len)
            print(f"  CAI-dLLM:       {cai_steps} steps avg "
                  f"({cai_avg:.1f}/block) "
                  f"[reduction: {(1-cai_steps/es_steps)*100:.1f}%]",
                  flush=True)
        except Exception as e:
            print(f"  CAI-dLLM error: {e}", flush=True)
            cai_steps = -1

        # CAI-dLLM no conf gating
        try:
            cai_nocg_steps, _ = count_steps_caidllm(
                model, tokenizer, prompts, gen_len, block_len,
                model_type, block_freq, no_cg=True)
            print(f"  CAI no-CG:      {cai_nocg_steps} steps avg "
                  f"[reduction: {(1-cai_nocg_steps/es_steps)*100:.1f}%]",
                  flush=True)
        except Exception as e:
            print(f"  CAI no-CG error: {e}", flush=True)
            cai_nocg_steps = -1

        all_results[task] = {
            "es_steps":       es_steps,
            "cai_steps":      cai_steps,
            "cai_nocg_steps": cai_nocg_steps,
            "max_steps":      gen_len,
            "reduction_pct":  round((1 - cai_steps/es_steps)*100, 1) if cai_steps > 0 else -1,
        }

    # Final summary table
    print(f"\n{'='*65}", flush=True)
    print(f" STEP REDUCTION SUMMARY — {args.model}", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'Task/Model':<25} {'ES-dLLM':>10} {'CAI-dLLM':>10} {'Reduction':>10}", flush=True)
    print(f"  {'─'*55}", flush=True)
    for task, r in all_results.items():
        label = f"{task}/{args.model.split('-')[0]}"
        print(f"  {label:<25} {r['es_steps']:>10} "
              f"{r['cai_steps']:>10} "
              f"{r['reduction_pct']:>9.1f}%", flush=True)
    print(f"{'='*65}\n", flush=True)

    # Save
    fname = f"{args.model.replace('-','_')}_step_reduction.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump({"model": args.model, "results": all_results}, f, indent=2)
    print(f"Saved → {os.path.join(args.output_dir, fname)}", flush=True)

if __name__ == "__main__":
    main()
