"""
eval_longbench_longctx.py  —  Long Context LongBench Evaluation
================================================================
Follows the paper protocol EXACTLY:
  - LongBench benchmark (6 English tasks, averaged)
  - Input truncated to 4000 tokens
  - block_length = 32
  - decoding_steps = 512  (gen_length)
  - generation_length = 512
  - Reports: aggregated LongBench score, TPS, peak GPU memory
  - Supports context length scaling for efficiency curves
  - Records OOM explicitly

Paper Table 6 reference scores (Base / nocache):
  LLaDA-8B-Instruct    : 34.55
  Dream-v0-7B-Instruct : 38.62

Tasks aggregated (LongBench English subset):
  hotpotqa, narrativeqa, qasper, multifieldqa_en, gov_report, qmsum

Usage:
  python cai_dllm/eval_longbench_longctx.py --model LLaDA-Instruct --method nocache
  python cai_dllm/eval_longbench_longctx.py --model Dream-Instruct  --method caidllm
"""

import os, sys, time, json, re, string, argparse, traceback
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from generate import batch_generate

try:
    from cai_generate3 import cai_batch_generate, CAIConfig
    from cai_enhancements import EnhancementConfig, PromptKVCache
    CAI_AVAILABLE = True
except ImportError:
    try:
        from cai_generate import cai_batch_generate, CAIConfig
        CAI_AVAILABLE = True
        EnhancementConfig = None; PromptKVCache = None
    except ImportError:
        CAI_AVAILABLE = False

MODEL_PATHS = {
    "LLaDA-Instruct": "GSAI-ML/LLaDA-8B-Instruct",
    "Dream-Instruct":  "Dream-org/Dream-v0-Instruct-7B",
}
MODEL_TYPE = {"LLaDA-Instruct": "llada", "Dream-Instruct": "Dream"}
MASK_IDS   = {"llada": 126336, "Dream": 151666}
EOS_IDS    = {"llada": 126081, "Dream": 151643}

TASKS = [
    {"name": "hotpotqa",        "metric": "f1"},
    {"name": "narrativeqa",     "metric": "f1"},
    {"name": "qasper",          "metric": "f1"},
    {"name": "multifieldqa_en", "metric": "f1"},
    {"name": "gov_report",      "metric": "rouge_l"},
    {"name": "qmsum",           "metric": "rouge_l"},
]

QA_PROMPT   = "Read the following passages carefully and answer the question with a short phrase.\n\nPassages:\n{context}\n\nQuestion: {question}\n\nAnswer:"
SUMM_PROMPT = "Please summarize the following document in a few sentences.\n\nDocument:\n{context}\n\nSummary:"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         required=True, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--method",        required=True,
                   choices=["nocache","dualcache","esdllm","caidllm","caidllm_enhanced"])
    p.add_argument("--n_samples",     type=int, default=200)
    p.add_argument("--batch_size",    type=int, default=2)
    p.add_argument("--max_input_len", type=int, default=4000)
    p.add_argument("--gen_length",    type=int, default=512)
    p.add_argument("--block_length",  type=int, default=32)
    p.add_argument("--output_dir",    type=str, default="log_results/longctx")
    p.add_argument("--tasks",         type=str, default="all")
    return p.parse_args()

def normalize(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def f1_score(pred, gts):
    pred_toks = normalize(pred).split()
    best = 0.0
    for gt in (gts if isinstance(gts, list) else [gts]):
        gt_toks = normalize(gt).split()
        common  = Counter(pred_toks) & Counter(gt_toks)
        n = sum(common.values())
        if not n: continue
        p = n/len(pred_toks) if pred_toks else 0
        r = n/len(gt_toks)   if gt_toks   else 0
        best = max(best, 2*p*r/(p+r))
    return best * 100

def rouge_l_score(pred, ref):
    p_toks = pred.lower().split()
    r_toks = (ref[0] if isinstance(ref, list) else ref).lower().split()
    if not p_toks or not r_toks: return 0.0
    m, n = len(r_toks), len(p_toks)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = (dp[i-1][j-1]+1 if r_toks[i-1]==p_toks[j-1]
                        else max(dp[i-1][j], dp[i][j-1]))
    lcs = dp[m][n]
    pr  = lcs/n if n else 0
    rc  = lcs/m if m else 0
    return 2*pr*rc/(pr+rc)*100 if (pr+rc) else 0.0

def build_prompt(item, task_name, tokenizer, max_input_len):
    context  = item.get("context", "")
    question = item.get("input", item.get("question", ""))
    is_summ  = task_name in ("gov_report", "qmsum")
    template = SUMM_PROMPT if is_summ else QA_PROMPT
    full = template.format(context=context) if is_summ else template.format(context=context, question=question)
    toks = tokenizer.encode(full, add_special_tokens=False)
    if len(toks) <= max_input_len:
        return full
    oh_text = template.format(context="") if is_summ else template.format(context="", question=question)
    oh = len(tokenizer.encode(oh_text, add_special_tokens=False))
    budget   = max(0, max_input_len - oh - 20)
    ctx_toks = tokenizer.encode(context, add_special_tokens=False)[:budget]
    context  = tokenizer.decode(ctx_toks, skip_special_tokens=True)
    return template.format(context=context) if is_summ else template.format(context=context, question=question)

def get_gen_kwargs(method, model_type, gen_length, block_length):
    block_freq = 16 if model_type == "llada" else 8
    base = {
        "gen_length": gen_length, "block_length": block_length,
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
    elif method in ("caidllm", "caidllm_enhanced"):
        base.update({"ESdLLM_mode": None, "parallel_mode": True,
                     "proportion_steps": [(1.0,0.0),(0.5,0.125),(0.25,0.25)],
                     "block_update_freq": block_freq, "prompt_update_freq": 64})
    return base

def get_cai_config():
    return CAIConfig(use_apd=True, use_per_block=True,
                     use_layer_adaptive=False, use_confidence_gating=True)

def get_enh_config(method):
    if method != "caidllm_enhanced" or not CAI_AVAILABLE or EnhancementConfig is None:
        return None
    return EnhancementConfig(use_soft_belief=False, use_oracle_budget=True,
                             use_prompt_kvcache=True, prompt_cache=PromptKVCache())

def eval_task(task, model, tokenizer, model_type, args, gen_kwargs, cai_cfg, enh_cfg):
    name   = task["name"]
    metric = task["metric"]
    eos_id = EOS_IDS[model_type]

    dataset = load_dataset("THUDM/LongBench", name, split="test", trust_remote_code=True)
    samples = list(dataset)[:args.n_samples]
    scores  = []; total_time = 0.0; total_toks = 0
    peak_mem = 0.0; oom = 0

    batches = [samples[i:i+args.batch_size] for i in range(0, len(samples), args.batch_size)]
    for bidx, batch in enumerate(batches):
        prompts = [build_prompt(item, name, tokenizer, args.max_input_len) for item in batch]
        enc = tokenizer(prompts, padding=True, truncation=True,
                        max_length=args.max_input_len, return_tensors="pt")
        input_ids = enc["input_ids"].cuda()
        attn_mask = enc["attention_mask"].float().cuda()
        plen = input_ids.shape[1]
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with torch.no_grad():
                if args.method in ("caidllm","caidllm_enhanced") and CAI_AVAILABLE:
                    output, _ = cai_batch_generate(model, input_ids, attn_mask,
                                                   generation_kwargs=gen_kwargs,
                                                   cai_config=cai_cfg,
                                                   enhancement_config=enh_cfg)
                else:
                    output, _ = batch_generate(model, input_ids, attn_mask, gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            oom += len(batch)
            print(f"  [OOM] {name} batch={bidx} ctx={plen}")
            torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"  [ERROR] {name} batch={bidx}: {e}"); traceback.print_exc(); continue

        total_time += time.time() - t0
        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated()/(1024**2))

        for i, item in enumerate(batch):
            gen_ids = output[i, plen:]
            ep = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(ep): gen_ids = gen_ids[:ep[0]]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split("\n")[0].strip()
            gts  = item.get("answers", item.get("answer", ""))
            if isinstance(gts, str): gts = [gts]
            scores.append(f1_score(pred, gts) if metric == "f1" else rouge_l_score(pred, gts))
            total_toks += len(gen_ids)

        if (bidx+1) % 20 == 0:
            print(f"  [{name}] [{bidx+1}/{len(batches)}] "
                  f"score={np.mean(scores):.1f}% tps={total_toks/max(total_time,1):.1f}")

    mean = np.mean(scores) if scores else 0.0
    tps  = total_toks / total_time if total_time > 0 else 0.0
    return mean, tps, peak_mem, oom, total_time, len(scores)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_type = MODEL_TYPE[args.model]
    model_path = MODEL_PATHS[args.model]

    run_tasks = TASKS if args.tasks == "all" else \
                [t for t in TASKS if t["name"] in args.tasks.split(",")]

    print(f"\n{'='*65}")
    print(f" Long Context LongBench — Paper Protocol")
    print(f" Model: {args.model}  Method: {args.method}")
    print(f" Tasks: {[t['name'] for t in run_tasks]}")
    print(f" Protocol: max_input={args.max_input_len} | block={args.block_length} | gen={args.gen_length}")
    print(f" Samples/task: {args.n_samples}")
    print(f"{'='*65}\n")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                   padding_side="left" if model_type == "Dream" else "right")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="cuda")
    if model_type == "llada":
        from models.hook_model import transform_llada_model; transform_llada_model(model)
    else:
        from models.hook_model import transform_dream_model; transform_dream_model(model)
    model.eval()
    print("Model loaded.\n")

    gen_kwargs = get_gen_kwargs(args.method, model_type, args.gen_length, args.block_length)
    cai_cfg    = get_cai_config() if args.method in ("caidllm","caidllm_enhanced") else None
    enh_cfg    = get_enh_config(args.method)

    task_results = {}; all_scores = []; tps_list = []; peak_mem = 0.0; total_oom = 0

    for task in run_tasks:
        print(f"\n── {task['name']} {'─'*40}")
        score, tps, mem, oom, t, n = eval_task(
            task, model, tokenizer, model_type, args, gen_kwargs, cai_cfg, enh_cfg)
        task_results[task["name"]] = {"score": round(score,2), "tps": round(tps,1),
                                       "mem_mb": round(mem), "oom": oom, "n": n}
        all_scores.append(score); tps_list.append(tps)
        peak_mem = max(peak_mem, mem); total_oom += oom
        print(f"  {task['name']}: score={score:.2f}%  tps={tps:.1f}  mem={mem:.0f}MB  oom={oom}")

    agg  = np.mean(all_scores) if all_scores else 0.0
    atps = np.mean(tps_list)   if tps_list   else 0.0

    print(f"\n{'='*65}")
    print(f" AGGREGATED — {args.model} | {args.method}")
    print(f"{'='*65}")
    print(f"  LongBench Score : {agg:.2f}%   (paper ref: LLaDA≈34.55, Dream≈38.62)")
    print(f"  Avg TPS         : {atps:.1f}")
    print(f"  Peak GPU Memory : {peak_mem:.0f} MB")
    print(f"  Total OOM       : {total_oom}")
    for k, v in task_results.items():
        print(f"  {k:25s}: {v['score']:.2f}%")
    print(f"{'='*65}\n")

    out = {"model": args.model, "method": args.method,
           "longbench_score": round(agg,2), "avg_tps": round(atps,1),
           "peak_memory_mb": round(peak_mem), "total_oom": total_oom,
           "max_input_len": args.max_input_len, "block_length": args.block_length,
           "gen_length": args.gen_length, "n_samples": args.n_samples,
           "per_task": task_results}
    fname = f"{args.model.replace('-','_')}_{args.method}_longbench_ctx{args.max_input_len}.json"
    out_path = os.path.join(args.output_dir, fname)
    with open(out_path, "w") as f: json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    main()
