"""
eval_longbench_hotpotqa.py  —  LongBench HotpotQA evaluation
=============================================================
Standalone script — does NOT use lm-eval harness.
Directly calls batch_generate / cai_batch_generate.

Usage:
  python eval_longbench_hotpotqa.py --model LLaDA-Instruct --method nocache
  python eval_longbench_hotpotqa.py --model LLaDA-Instruct --method dualcache
  python eval_longbench_hotpotqa.py --model LLaDA-Instruct --method esdllm
  python eval_longbench_hotpotqa.py --model LLaDA-Instruct --method caidllm
  python eval_longbench_hotpotqa.py --model Dream-Instruct --method caidllm
  python eval_longbench_hotpotqa.py --model Dream-Instruct --method nocache --n_samples 50

Models:  LLaDA-Instruct | Dream-Instruct
Methods: nocache | dualcache | esdllm | caidllm
"""

import os, sys, time, json, re, string, argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# ── ES-dLLM imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import batch_generate
from models.hook_model import transform_llada_model, transform_dream_model

# ── CAI-dLLM imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cai_dllm"))
from cai_generate import cai_batch_generate, CAIConfig


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True,
                        choices=["LLaDA-Instruct", "Dream-Instruct"])
    parser.add_argument("--method",    required=True,
                        choices=["nocache", "dualcache", "esdllm", "caidllm"])
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of HotpotQA samples to evaluate (max 1000)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_len", type=int, default=2048,
                        help="Max prompt length in tokens (truncated if longer)")
    parser.add_argument("--gen_length",   type=int, default=64,
                        help="Max new tokens to generate per answer")
    parser.add_argument("--output_dir",   type=str, default="log_results/longbench")
    return parser.parse_args()


# =============================================================================
# Model paths
# =============================================================================
MODEL_PATHS = {
    "LLaDA-Instruct": "GSAI-ML/LLaDA-8B-Instruct",
    "Dream-Instruct":  "Dream-org/Dream-v0-Instruct-7B",
}

MODEL_TYPE = {
    "LLaDA-Instruct": "llada",
    "Dream-Instruct":  "Dream",
}

MASK_IDS = {"llada": 126336, "Dream": 151666}
EOS_IDS  = {"llada": 126081, "Dream": 151643}
TOKEN_OFFSETS = {"llada": 0, "Dream": 1}


# =============================================================================
# F1 evaluation (standard QA metric)
# =============================================================================
def normalize_answer(s):
    """Lower, remove punctuation, articles, extra whitespace."""
    def remove_articles(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def white_space_fix(t): return ' '.join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return ''.join(c for c in t if c not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_f1(prediction, ground_truths):
    """Return max F1 over all ground truth answers."""
    max_f1 = 0.0
    pred_tokens = get_tokens(prediction)
    for gt in ground_truths:
        gt_tokens = get_tokens(gt)
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens) if pred_tokens else 0
        recall    = num_same / len(gt_tokens)    if gt_tokens   else 0
        f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, f1)
    return max_f1

def compute_em(prediction, ground_truths):
    """Exact match — 1 if prediction matches any ground truth."""
    pred_norm = normalize_answer(prediction)
    return max(int(pred_norm == normalize_answer(gt)) for gt in ground_truths)


# =============================================================================
# Prompt builder for HotpotQA
# =============================================================================
PROMPT_TEMPLATE = (
    "Read the following passages and answer the question with a short phrase.\n\n"
    "Passages:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

def build_prompt(item, tokenizer, max_input_len):
    """Build tokenized prompt, truncating context if needed."""
    context = item["context"]
    question = item["input"]

    # Rough truncation: trim context to fit max_input_len
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > max_input_len:
        # Trim context proportionally
        overhead  = len(tokenizer.encode(
            PROMPT_TEMPLATE.format(context="", question=question),
            add_special_tokens=False))
        budget    = max_input_len - overhead - 10
        ctx_tokens = tokenizer.encode(context, add_special_tokens=False)[:budget]
        context   = tokenizer.decode(ctx_tokens, skip_special_tokens=True)
        prompt    = PROMPT_TEMPLATE.format(context=context, question=question)

    return prompt


# =============================================================================
# Generation kwargs per method
# =============================================================================
def get_gen_kwargs(method, model_type, gen_length, block_length=64):
    base = {
        "gen_length":        gen_length,
        "block_length":      min(block_length, gen_length),
        "temperature":       0.0,
        "cfg_scale":         0.0,
        "delay_eos_generation": True,
        "parallel_mode":     False,
        "token_per_step":    1,
        "threshold":         None,
        "print_log":         False,
        "record_time":       False,
        "statistics":        False,
        "sparse_kv":         1.0,
        "delay_step":        -1,
        "top_p":             0.95,
        "top_k":             50,
        "ESdLLM_mode":       None,
        "importance_score_alpha": 0.5,
        "proportion_steps":  None,
        "block_update_freq": None,
        "prompt_update_freq": None,
        "model_type":        model_type,
    }

    if method == "nocache":
        base["use_kvcache"] = False

    elif method == "dualcache":
        base["use_kvcache"] = True

    elif method == "esdllm":
        base["use_kvcache"]           = True
        base["ESdLLM_mode"]           = "HiddenState"
        base["importance_score_alpha"] = 0.5
        base["proportion_steps"]      = [(1.0, 0.0), (0.5, 0.125), (0.25, 0.25)]
        base["block_update_freq"]     = 16 if model_type == "llada" else 8
        base["prompt_update_freq"]    = 64

    elif method == "caidllm":
        base["use_kvcache"]           = True
        base["ESdLLM_mode"]           = None
        base["parallel_mode"]         = True
        base["importance_score_alpha"] = 0.5
        base["proportion_steps"]      = [(1.0, 0.0), (0.5, 0.125), (0.25, 0.25)]
        base["block_update_freq"]     = 16 if model_type == "llada" else 8
        base["prompt_update_freq"]    = 64

    return base


def get_cai_config():
    return CAIConfig(
        use_apd=True,
        use_per_block=True,
        use_layer_adaptive=False,
        use_confidence_gating=True,
    )


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = MODEL_PATHS[args.model]
    model_type = MODEL_TYPE[args.model]
    mask_id    = MASK_IDS[model_type]
    eos_id     = EOS_IDS[model_type]
    tok_offset = TOKEN_OFFSETS[model_type]

    print(f"\n{'='*60}")
    print(f" LongBench HotpotQA Evaluation")
    print(f" Model:  {args.model}  ({model_path})")
    print(f" Method: {args.method}")
    print(f" Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading LongBench HotpotQA ...")
    dataset = load_dataset("THUDM/LongBench", "hotpotqa", split="test", trust_remote_code=True)
    samples = list(dataset)[:args.n_samples]
    print(f"Loaded {len(samples)} samples "
          f"(avg context length: {np.mean([s['length'] for s in samples]):.0f} words)")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        padding_side="left" if model_type == "Dream" else "right"
    )
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if model_type == "llada":
        transform_llada_model(model)
    else:
        from models.hook_model import transform_dream_model
        transform_dream_model(model)
    model.eval()
    print("Model loaded.")

    # ── Build generation kwargs ───────────────────────────────────────────────
    gen_kwargs  = get_gen_kwargs(args.method, model_type,
                                 args.gen_length, args.gen_length)
    cai_config  = get_cai_config() if args.method == "caidllm" else None

    # ── Evaluate ─────────────────────────────────────────────────────────────
    all_f1, all_em = [], []
    total_time = 0.0
    total_tokens = 0
    results = []

    batches = [samples[i:i+args.batch_size]
               for i in range(0, len(samples), args.batch_size)]

    for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating")):
        # Build prompts
        prompts = [build_prompt(item, tokenizer, args.max_input_len)
                   for item in batch]

        # Tokenize with padding
        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_input_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to("cuda")
        attn_mask = enc["attention_mask"].float().to("cuda")
        prompt_len = input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            if args.method == "caidllm":
                output, _ = cai_batch_generate(
                    model, input_ids, attn_mask,
                    generation_kwargs=gen_kwargs,
                    cai_config=cai_config,
                )
            else:
                output, _ = batch_generate(
                    model, input_ids, attn_mask,
                    generation_kwargs=gen_kwargs,
                )
        elapsed = time.time() - t0
        total_time += elapsed

        # Decode generated answers (only new tokens)
        for i, item in enumerate(batch):
            gen_ids = output[i, prompt_len:]
            # Stop at EOS
            eos_positions = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gen_ids = gen_ids[:eos_positions[0]]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Extract first line / short answer
            pred = pred.split("\n")[0].strip()

            ground_truths = item["answers"]
            f1 = compute_f1(pred, ground_truths)
            em = compute_em(pred, ground_truths)
            all_f1.append(f1)
            all_em.append(em)
            total_tokens += len(gen_ids)

            results.append({
                "question":       item["input"],
                "ground_truths":  ground_truths,
                "prediction":     pred,
                "f1":             f1,
                "em":             em,
                "context_length": item["length"],
            })

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx+1}/{len(batches)}] "
                  f"F1={np.mean(all_f1)*100:.1f}%  "
                  f"EM={np.mean(all_em)*100:.1f}%  "
                  f"time={total_time:.0f}s")

    # ── Final metrics ─────────────────────────────────────────────────────────
    mean_f1 = np.mean(all_f1) * 100
    mean_em = np.mean(all_em) * 100
    tps     = total_tokens / total_time if total_time > 0 else 0

    print(f"\n{'='*60}")
    print(f" RESULTS — {args.model} | {args.method}")
    print(f"{'='*60}")
    print(f"  Samples:      {len(samples)}")
    print(f"  F1 Score:     {mean_f1:.2f}%")
    print(f"  Exact Match:  {mean_em:.2f}%")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"  Tokens/sec:   {tps:.1f}")
    print(f"{'='*60}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    out_name = f"{args.model.replace('-','_')}_{args.method}_hotpotqa.json"
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w") as f:
        json.dump({
            "model":       args.model,
            "method":      args.method,
            "n_samples":   len(samples),
            "f1":          mean_f1,
            "em":          mean_em,
            "total_time":  total_time,
            "tokens_per_sec": tps,
            "results":     results,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
