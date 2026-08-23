'''
This file is inspired by the code from https://github.com/ML-GSAI/LLaDA
'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cai_dllm"))

import accelerate
import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate, setup_parser
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from generate import batch_generate, print_statistics
from models.hook_model import transform_llada_model, transform_dream_model
import time
import argparse

# CAI imports — always from cai_generate3 which has all features
try:
    from cai_generate3 import cai_batch_generate, CAIConfig
    CAI_AVAILABLE = True
except ImportError:
    try:
        from cai_generate import cai_batch_generate, CAIConfig
        CAI_AVAILABLE = True
    except ImportError:
        CAI_AVAILABLE = False


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(self, model_path='', mask_id=126336, max_length=4096,
                 batch_size=32, device="cuda", **kwargs):
        super().__init__()
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        if self.model.config.model_type == 'llada':
            transform_llada_model(self.model)
        elif self.model.config.model_type == 'Dream':
            transform_dream_model(self.model)

        if self.accelerator is not None:
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.device = (torch.device("cuda") if torch.cuda.is_available()
                           else torch.device("cpu"))
            self.model = self.model.to(self.device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        tokenizer_kwargs = {}
        if self.model.config.model_type == 'Dream':
            tokenizer_kwargs['padding_side'] = 'left'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, **tokenizer_kwargs)
        self.batch_size = int(batch_size)
        self.sampling_eps = 0.
        self.max_length = max_length
        self.cai_config = kwargs.pop('cai_config', None)
        self.generation_kwargs = kwargs
        self.cfg = kwargs.get('cfg_scale', 0.0)

    @property
    def rank(self): return self._rank
    @property
    def world_size(self): return self._world_size
    @property
    def tokenizer_name(self): return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt)

    def loglikelihood(self, requests): raise NotImplementedError
    def loglikelihood_rolling(self, requests): raise NotImplementedError

    def generate_until(self, requests):
        out = []
        total_time = 0
        request_cnt = 0
        questions = [req.args[0] for req in requests]
        untils = [req.args[1]["until"] for req in requests]

        for start in tqdm(range(0, len(questions), self.batch_size), desc="Generating..."):
            batch_questions = questions[start:start + self.batch_size]
            batch_untils = untils[start:start + self.batch_size]
            enc = self.tokenizer(batch_questions, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            prompts = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            start_time = time.time()
            request_cnt += prompts.shape[0]

            if self.cai_config is not None and CAI_AVAILABLE:
                generated_answers, _ = cai_batch_generate(
                    self.model, prompts, attention_mask,
                    generation_kwargs=self.generation_kwargs,
                    cai_config=self.cai_config)
            else:
                generated_answers, _ = batch_generate(
                    self.model, prompts, attention_mask, self.generation_kwargs)

            total_time += time.time() - start_time

            for generated_answer, prompt, stop_tokens in zip(
                    generated_answers, prompts, batch_untils):
                generated_answer = self.tokenizer.decode(
                    generated_answer[prompt.shape[0]:], skip_special_tokens=False)
                for stop_seq in stop_tokens + ["```"]:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]
                generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
                generated_answer = self.tokenizer.decode(
                    generated_answer_ids, skip_special_tokens=True)
                out.append(generated_answer)

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        print(f"Total generation time: {total_time:.2f} seconds")
        print(f"Request count: {request_cnt}")
        print_statistics()
        return out


def evaluate(model, task, batch_size, **kwargs):
    model_map = {
        "GSAI-ML/LLaDA-8B-Base":       (False,),
        "GSAI-ML/LLaDA-8B-Instruct":   (True,),
        "Dream-org/Dream-v0-Base-7B":   (False,),
        "Dream-org/Dream-v0-Instruct-7B": (True,),
    }
    if model not in model_map:
        raise ValueError(f"Not supported model: {model}")
    is_instruct = model_map[model][0]
    model_path = model

    confirm_run_unsafe_code = False
    limit = None

    task_map = {
        "bbh":          ("bbh", False),
        "gsm8k":        ("gsm8k", False),
        "minerva_math": ("minerva_math", False),
        "humaneval":    ("humaneval", True),
        "hellaswag":    ("hellaswag", False),
        "mbpp":         ("mbpp", True),
        "mmlu_pro":     ("mmlu_pro", False),
    }
    if task not in task_map:
        raise ValueError(f"Not supported task: {task}")
    tasks, confirm_run_unsafe_code = task_map[task]

    if confirm_run_unsafe_code:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    set_seed(123)
    args = setup_parser().parse_args([])
    args.__dict__.update({
        "tasks": tasks,
        "model": "llada_dist",
        "model_args": {"model_path": model_path, **kwargs},
        "batch_size": batch_size,
        "limit": limit,
        "cache_requests": True,
        "apply_chat_template": is_instruct,
        "fewshot_as_multiturn": is_instruct,
        "confirm_run_unsafe_code": confirm_run_unsafe_code,
    })
    print(args)
    cli_evaluate(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--esdllm_mode", required=True)
    parser.add_argument("--prompt_update_freq", type=int, default=None)
    parser.add_argument("--block_update_freq", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--proportions", nargs='*', type=float, default=[])
    parser.add_argument("--positions",   nargs='*', type=float, default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_confidence_gating", action="store_true", default=False)
    parser.add_argument("--apd_theta_end", type=float, default=0.70)
    parser.add_argument("--apd_theta_start", type=float, default=0.90)
    parser.add_argument("--apd_warmup_frac", type=float, default=0.15)
    parser.add_argument("--use_soft_grind", action="store_true", default=False,
                        help="Soft grinding phase: only kills truly wasteful steps")
    parser.add_argument("--use_cai", action="store_true", default=False)
    parser.add_argument("--cai_mode", default="full",
                        choices=["apd_only", "apd_per_block", "full"])
    args = parser.parse_args()

    assert len(args.proportions) == len(args.positions)
    proportion_steps = [(pos, pr) for pos, pr in zip(args.proportions, args.positions)] \
                       if args.proportions else None

    model_info = {
        "LLaDA-Instruct": ("GSAI-ML/LLaDA-8B-Instruct",   "LLaDA"),
        "LLaDA-Base":     ("GSAI-ML/LLaDA-8B-Base",        "LLaDA"),
        "Dream-Base":     ("Dream-org/Dream-v0-Base-7B",    "Dream"),
        "Dream-Instruct": ("Dream-org/Dream-v0-Instruct-7B","Dream"),
    }
    if args.model not in model_info:
        raise ValueError(f"Not supported model: {args.model}")
    model_name, model_type = model_info[args.model]

    task_settings = {
        "bbh":          (8, 256, 64),
        "gsm8k":        (8, 256, 64),
        "hellaswag":    (8, 256, 64),
        "mbpp":         (8, 256, 64),
        "minerva_math": (8, 256, 256),
        "humaneval":    (8, 512, 64),
        "mmlu_pro":     (8, 256, 256),
    }
    if args.task not in task_settings:
        raise ValueError(f"Not supported task: {args.task}")
    batch_size, gen_length, block_length = task_settings[args.task]

    mode_list = args.esdllm_mode.split("_")
    token_per_step = 1
    threshold = None
    parallel_mode = False
    sparse_kv = 1.0
    delay_step = -1

    for mode in mode_list[1:]:
        if mode == "p":
            threshold = 0.9; parallel_mode = True
        elif mode == "s":
            sparse_kv = 0.5; delay_step = 1
        else:
            raise ValueError(f"Not supported ES-dLLM mode: {mode}")
    args.esdllm_mode = mode_list[0]

    if args.esdllm_mode == "DualCache":
        ESdLLM_mode = None; use_kvcache = True
    elif args.esdllm_mode == "nocache":
        ESdLLM_mode = None; use_kvcache = False
    elif args.esdllm_mode in ["HiddenState", "Key", "Value", "Query"]:
        ESdLLM_mode = args.esdllm_mode; use_kvcache = True
        n_layers = 32 if model_type == "LLaDA" else 28
        decode_proportions = [
            [p for p, s in proportion_steps if i >= s * n_layers][-1]
            for i in range(n_layers)
        ] if proportion_steps else [1.0] * n_layers
        print(f"FLOPs proportions: "
              f"{(sum(decode_proportions)+1-decode_proportions[-1])/len(decode_proportions):.2f}")
    else:
        raise ValueError(f"Not supported ES-dLLM mode: {args.esdllm_mode}")

    generation_kwargs = {
        "gen_length": gen_length, "block_length": block_length,
        "temperature": 0.0, "cfg_scale": 0.0,
        "use_kvcache": use_kvcache, "parallel_mode": parallel_mode,
        "token_per_step": token_per_step, "threshold": threshold,
        "print_log": False, "record_time": False, "statistics": False,
        "delay_eos_generation": True, "sparse_kv": sparse_kv,
        "delay_step": delay_step, "top_p": 0.95, "top_k": 50,
        "block_update_freq": args.block_update_freq,
        "prompt_update_freq": args.prompt_update_freq,
        "ESdLLM_mode": ESdLLM_mode,
        "importance_score_alpha": args.alpha,
        "proportion_steps": proportion_steps,
    }

    if args.use_cai and CAI_AVAILABLE:
        generation_kwargs['parallel_mode'] = True
        generation_kwargs['ESdLLM_mode'] = None
        generation_kwargs['cai_config'] = CAIConfig(
            use_apd=True,
            use_per_block=(args.cai_mode in ('apd_per_block', 'full')),
            use_layer_adaptive=(args.cai_mode == 'full'),
            use_confidence_gating=not args.no_confidence_gating,
            apd_theta_end=args.apd_theta_end,
            apd_theta_start=args.apd_theta_start,
            apd_warmup_frac=args.apd_warmup_frac,
        )

    # Inject --limit into sys.argv for lm-eval
    if args.limit is not None:
        if "--limit" not in sys.argv and "-L" not in sys.argv:
            sys.argv.extend(["--limit", str(float(args.limit))])

    evaluate(model_name, args.task, batch_size, **generation_kwargs)
    accelerate.PartialState().destroy_process_group()