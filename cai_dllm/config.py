"""
config.py  ─  CAI-dLLM  Master Configuration
=============================================
Extends ES-dLLM's config.py with all CAI-dLLM parameters.
Import this instead of (or alongside) the original config.py.
"""

# ─── ES-dLLM base settings (unchanged) ──────────────────────────────────────
record_time        = False
statistics         = False
model_name         = "LLaDA-Instruct"
gen_length         = 256
token_per_step     = 1
block_length       = 64
delay_eos_generation = True

sparse_kv          = 1.0        # 1.0 = disabled; set <1.0 to enable global sparse KV
delay_step         = 1

use_kvcache        = True
block_update_freq  = 64
prompt_update_freq = block_length
ESdLLM_mode        = "HiddenState"   # "Key" | "Value" | "Query" | "HiddenState" | None
importance_score_alpha = 0.5
proportion_steps   = [(1, 0), (0.5, 0.125), (0.25, 0.25)]

# ─── §4.1  APD Schedule ──────────────────────────────────────────────────────
USE_APD            = True        # enable Adaptive Parallel Decoding
APD_THETA_START    = 0.90        # conservative threshold (warmup phase)
APD_THETA_END      = 0.70        # aggressive  threshold (end of block)
APD_WARMUP_FRAC    = 0.15        # fraction of steps held at θ_start (≈10 of 64)
APD_SHAPE          = "cosine"    # "cosine" | "linear"

# Parallel mode must be True for APD to work
parallel_mode      = True
threshold          = APD_THETA_START   # fallback if APD disabled

# ─── §4.2  Per-Block Schedule ────────────────────────────────────────────────
USE_PER_BLOCK      = True        # use block-position-specific θ_end values

# θ_end per block (index 0 = first block, index 3 = last block)
# Derived from Experiment 5: Block 4 is 37% faster than Block 1
PER_BLOCK_THETA_ENDS = [
    0.70,   # Block 0: conservative  (27 steps avg, bottleneck)
    0.60,   # Block 1: moderate      (25 steps avg)
    0.50,   # Block 2: aggressive    (20 steps avg)
    0.40,   # Block 3: very aggressive (17 steps avg)
]

# Online adaptation of per-block schedules (lightweight, optional)
USE_ADAPTIVE_UPDATE = False
ADAPT_RATE          = 0.05       # θ_end nudge per mismatch step

# ─── §4.3  Layer-Adaptive Compute ────────────────────────────────────────────
USE_LAYER_ADAPTIVE  = True       # enable layer-adaptive forward

# Tier boundaries (layer index, 0-indexed for LLaDA-8B with 32 layers)
EARLY_LAYER_RANGE   = (0,  9)    # layers 0–8  : 2–3% drift
MIDDLE_LAYER_RANGE  = (9,  17)   # layers 9–16 : 3–6% drift
DEEP_LAYER_RANGE    = (17, 32)   # layers 17–31: 6–14% drift

# Step stride for each tier (1 = every step, 2 = every other step)
EARLY_STEP_STRIDE   = 2          # early layers: skip on odd steps
MIDDLE_STEP_STRIDE  = 1
DEEP_STEP_STRIDE    = 1

# KV retention ratios per tier (from Experiment 2)
KV_RETENTION_EARLY  = 0.60       # layers 0,4 (broadest): 60%
KV_RETENTION_L16    = 0.25       # layer 16 (sparsest): 25%
KV_RETENTION_MIDDLE = 0.65       # other middle layers: 65%
KV_RETENTION_DEEP   = 1.00       # deep layers: no pruning

# ─── Logging ─────────────────────────────────────────────────────────────────
PRINT_LOG           = False
PRINT_SCHEDULE      = False      # log θ(t) values
