"""
smoke_test.py  —  CAI-dLLM quick smoke test
Run from ES-dLLM root:  srun python cai_dllm/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))   # cai_dllm/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # ES-dLLM root

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.hook_model import transform_llada_model
from cai_generate import cai_batch_generate, CAIConfig

# ── model ─────────────────────────────────────────────────────────────
model_path = "GSAI-ML/LLaDA-8B-Instruct"
print(f"Loading {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="cuda"
)
transform_llada_model(model)
model.eval()

# ── prompt ────────────────────────────────────────────────────────────
prompt = (
    "Natalia sold clips to 48 of her friends in April and then sold "
    "half as many clips in May. How many clips did Natalia sell altogether?"
)
msgs = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
mask = inputs["attention_mask"].float()

# ── generate ──────────────────────────────────────────────────────────
cfg = CAIConfig(
    use_apd            = True,
    use_per_block      = True,
    use_layer_adaptive = False,   # enable after smoke-test passes
    print_log          = True,
)

gen_kwargs = {
    "gen_length"            : 256,
    "block_length"          : 64,
    "model_type"            : "llada",
    "use_kvcache"           : True,
    "ESdLLM_mode"           : "HiddenState",
    "importance_score_alpha": 0.5,
    "proportion_steps"      : [(1, 0), (0.5, 0.125), (0.25, 0.25)],
    "block_update_freq"     : 64,
    "prompt_update_freq"    : 64,
    "eos_token"             : tokenizer.eos_token_id,
}

output, results = cai_batch_generate(
    model, input_ids, mask,
    generation_kwargs=gen_kwargs,
    cai_config=cfg,
)

answer = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
print("\n=== ANSWER ===")
print(answer)
print("\n=== TIMING ===")
print(results["time_info"])
print("Block steps:", results["block_steps"])
