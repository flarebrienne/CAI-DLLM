"""
cai_generate.py  ─  CAI-dLLM  Integrated Generation Loop
=========================================================
Drop-in replacement for ES-dLLM's `batch_generate` in generate.py.

Integrates §4.1, §4.2, and §4.3 on top of the ES-dLLM base:

  §4.1  APD schedule     → apd_schedule.py
  §4.2  Per-block sched  → per_block_schedule.py
  §4.3  Layer-adaptive   → layer_adaptive.py

The function signature is backward-compatible with ES-dLLM's
`batch_generate` (same positional/keyword arguments), with new
keyword arguments added at the end for CAI-dLLM features.

Quick-start
-----------
    from cai_generate import cai_batch_generate, CAIConfig
    from per_block_schedule import PerBlockScheduleTable
    from layer_adaptive import LayerTierTable

    cfg = CAIConfig(
        use_apd            = True,
        use_per_block      = True,
        use_layer_adaptive = True,
    )
    output, results = cai_batch_generate(model, input_ids, mask,
                                         generation_kwargs={...},
                                         cai_config=cfg)
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# ── ES-dLLM imports (unchanged) ──────────────────────────────────────────────
from generate import (
    prepare_data_caches,
    add_gumbel_noise,
    sample_tokens_LLaDA,
    sample_tokens_Dream,
    statistics_per_step,
    statistics_final,
    total_effective_tokens,
    request_cnt,
    total_importance_time_global,
    total_time_global,
)
import early_skipping

# ── CAI-dLLM imports ──────────────────────────────────────────────────────────
from apd_schedule import (
    APDScheduleConfig,
    APDScheduler,
    apd_sample_tokens,
    probe_step0_confidence,
    classify_token_difficulty,
)
from per_block_schedule import (
    PerBlockScheduleTable,
    BlockConvergenceTracker,
    AdaptiveScheduleUpdater,
    get_block_threshold,
)
from layer_adaptive import (
    LayerTierTable,
    LayerAdaptiveForward,
    estimate_flop_savings,
)
from confidence_gating import (
    TokenBudgetManager,
    PositionAwareScheduler,
    GrindingPhaseDetector,
    confidence_gated_sample,
)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  CAI configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CAIConfig:
    """
    Master configuration for CAI-dLLM.  All new features are opt-in
    so the caller can enable them incrementally.
    """

    # ── §4.1  APD ────────────────────────────────────────────────────────
    use_apd:          bool  = True
    apd_theta_start:  float = 0.90
    apd_theta_end:    float = 0.70
    apd_warmup_frac:  float = 0.15
    apd_shape:        str   = "cosine"   # "cosine" | "linear"

    # ── §4.2  Per-block schedule ─────────────────────────────────────────
    use_per_block:    bool  = True
    # theta_ends[i] overrides apd_theta_end for block i
    # If None, uses the default table from §4.2
    per_block_theta_ends: Optional[List[float]] = None

    # Online adaptation of per-block schedules (lightweight bandit)
    use_adaptive_update: bool  = False
    adapt_rate:          float = 0.05

    # ── §4.3  Layer-adaptive compute ─────────────────────────────────────
    use_layer_adaptive:  bool  = True
    # If None, uses LayerTierTable.for_llada_8b()
    layer_tier_table:    Optional[LayerTierTable] = None

    # ── §4.4  Confidence gating (Exp 2 + 3) ─────────────────────────────
    use_confidence_gating: bool = True

    # §4.5 error budget
    # ── §4.5  Error budget controller ────────────────────────────────────
    use_error_budget:   bool  = False
    error_budget_max:   float = 0.05
    error_budget_decay: float = 0.90

    # ── Logging / debug ──────────────────────────────────────────────────
    print_log:      bool = False
    print_schedule: bool = False   # log θ(t) for each step

    def build_schedule_table(self, num_blocks: int) -> PerBlockScheduleTable:
        if self.per_block_theta_ends is not None:
            return PerBlockScheduleTable.from_theta_ends(
                self.per_block_theta_ends[:num_blocks],
                theta_start=self.apd_theta_start,
                warmup_frac=self.apd_warmup_frac,
            )
        if self.use_per_block:
            return PerBlockScheduleTable.default(num_blocks)
        # Single schedule for all blocks
        single_cfg = APDScheduleConfig(
            theta_start=self.apd_theta_start,
            theta_end=self.apd_theta_end,
            warmup_frac=self.apd_warmup_frac,
            shape=self.apd_shape,
        )
        return PerBlockScheduleTable(configs=[single_cfg] * num_blocks)

    def build_tier_table(self) -> Optional[LayerTierTable]:
        if not self.use_layer_adaptive:
            return None
        if self.layer_tier_table is not None:
            return self.layer_tier_table
        return LayerTierTable.for_llada_8b()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Main generation function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def cai_batch_generate(
    model,
    input_ids:  torch.Tensor,
    mask:       torch.Tensor,
    generation_kwargs: dict = {},
    cai_config: CAIConfig   = None,
) -> Tuple[torch.Tensor, dict]:
    """
    CAI-dLLM integrated generation.

    Parameters
    ----------
    model            : LLaDA / Dream model (hooked with ES-dLLM patches)
    input_ids        : (B, prompt_len) int64 tensor
    mask             : (B, prompt_len) attention mask (1 = real, 0 = pad)
    generation_kwargs: same as ES-dLLM batch_generate (see generate.py)
    cai_config       : CAIConfig instance (defaults to all features enabled)

    Returns
    -------
    x       : (B, prompt_len + gen_length) generated token ids
    results : dict with timing, statistics, per-block convergence info
    """
    if cai_config is None:
        cai_config = CAIConfig()

    # ── unpack generation kwargs (mirrors ES-dLLM batch_generate) ────────
    gen_length       = generation_kwargs.get("gen_length",       256)
    block_length     = generation_kwargs.get("block_length",      64)
    token_per_step   = generation_kwargs.get("token_per_step",     1)
    temperature      = generation_kwargs.get("temperature",       0.0)
    remasking        = generation_kwargs.get("remasking", "low_confidence")
    eos_token        = generation_kwargs.get("eos_token",          -1)
    use_kvcache      = generation_kwargs.get("use_kvcache",       True)
    cfg_scale        = generation_kwargs.get("cfg_scale",          0.0)
    sparse_kv        = generation_kwargs.get("sparse_kv",         1.0)
    delay_step       = generation_kwargs.get("delay_step",          1)
    ESdLLM_mode      = generation_kwargs.get("ESdLLM_mode",       None)
    importance_score_alpha = generation_kwargs.get("importance_score_alpha", 0.5)
    proportion_steps = generation_kwargs.get("proportion_steps",   [(1, 0)])
    block_update_freq= generation_kwargs.get("block_update_freq",  64)
    prompt_update_freq=generation_kwargs.get("prompt_update_freq", block_length)
    # token_offset is set per model type in the metadata block below (0=LLaDA, 1=Dream)
    do_statistics    = generation_kwargs.get("do_statistics",      False)
    record_time      = generation_kwargs.get("record_time",        False)
    model_type       = generation_kwargs.get("model_type",      "llada")

    # ── model metadata ────────────────────────────────────────────────────
    # Names taken directly from ES-dLLM batch_generate (generate.py L278-296)
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    device     = model.device

    if model.config.model_type == "llada":
        n_layers     = model.config.n_layers
        n_heads      = model.config.n_heads
        n_kv_heads   = model.config.n_kv_heads
        hidden_dim   = model.config.d_model
        mask_id      = 126336
        eos_token    = generation_kwargs.get("eos_token", 126081)
        token_offset = 0
    elif model.config.model_type == "Dream":
        n_layers     = model.config.num_hidden_layers
        n_heads      = model.config.num_attention_heads
        n_kv_heads   = model.config.num_key_value_heads
        hidden_dim   = model.config.hidden_size
        mask_id      = 151666
        eos_token    = generation_kwargs.get("eos_token", 151643)
        token_offset = 1
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")

    max_length = prompt_len + gen_length + token_offset

    # ── build CAI-dLLM components ─────────────────────────────────────────
    num_blocks      = gen_length // block_length
    steps_per_block = block_length   # one step per masked token slot (parallel mode)

    schedule_table = cai_config.build_schedule_table(num_blocks)
    tier_table     = cai_config.build_tier_table()
    la_forward     = LayerAdaptiveForward(tier_table) if tier_table is not None else None
    conv_tracker   = BlockConvergenceTracker(num_blocks) if cai_config.use_apd else None
    use_cg = cai_config.use_apd and cai_config.use_confidence_gating
    sched_updater: Optional[AdaptiveScheduleUpdater] = None
    if cai_config.use_per_block and cai_config.use_adaptive_update:
        sched_updater = AdaptiveScheduleUpdater(
            schedule_table, adapt_rate=cai_config.adapt_rate
        )

    if cai_config.print_schedule:
        print(schedule_table)
        if tier_table:
            tier_table.print_summary()

    # ── token buffer and attention mask ──────────────────────────────────
    # Mirror original batch_generate exactly: pad x to max_length + token_offset
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_id).to(device)
    prompt_index = (x != mask_id)

    if mask is not None and torch.any(mask == 0.0):
        mask = F.pad(mask, (0, gen_length + token_offset), value=1.0)
        real_position = torch.cumsum(mask, dim=-1).to(device) - 1
        attention_mask = torch.logical_and(
            mask.unsqueeze(1).unsqueeze(-2),
            mask.unsqueeze(1).unsqueeze(-1)
        ).to(device)
    else:
        mask = torch.ones((batch_size, max_length), dtype=torch.bfloat16, device=device)
        attention_mask = torch.ones(
            (batch_size, 1, max_length, max_length), dtype=torch.bool, device=device
        )
        real_position = torch.arange(max_length, dtype=torch.long, device=device).expand(batch_size, -1)
    mask = mask.to(device)

    # ── ES-dLLM proportion / skip-layers setup ────────────────────────────
    decode_proportions = None
    if ESdLLM_mode is not None:
        decode_proportions = [
            [p for p, s in proportion_steps if i >= s * n_layers][-1]
            for i in range(n_layers)
        ]
        skip_layers = [decode_proportions[i] != decode_proportions[i - 1]
                       for i in range(1, n_layers)]
        skip_layers = [decode_proportions[0] != 1] + skip_layers
    else:
        skip_layers = [False] * n_layers

    # ── data caches ───────────────────────────────────────────────────────
    data_caches, all_confidence, confidence_coef, absdiff_coef, constant_coef = \
        prepare_data_caches(
            kv_shape=(batch_size, n_kv_heads, max_length, hidden_dim // n_heads),
            real_position=real_position,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            device=device,
            use_cfg_scale=(cfg_scale > 0.),
            ESdLLM_mode=ESdLLM_mode,
            importance_score_alpha=importance_score_alpha,
            skip_layers=skip_layers,
            statistics=do_statistics,
            record_time=record_time,
        )

    decoding_confidence = torch.zeros((batch_size, max_length), dtype=torch.bfloat16, device=device)
    decoding_confidence[:, :prompt_len] = 1.0

    # ── results dict ──────────────────────────────────────────────────────
    results = {
        "confidence": [], "confidence_diff": [],
        "key_similarity": [], "key_absdiff": [], "key_absdiff_layersim": [],
        "value_similarity": [], "value_absdiff": [], "value_absdiff_layersim": [],
        "query_similarity": [], "query_absdiff": [], "query_absdiff_layersim": [],
        "hidden_state_similarity": [], "hidden_state_absdiff": [],
        "hidden_state_absdiff_layersim": [],
        "predicted_token": [], "token_change_cnt": [],
        # CAI-dLLM extras
        "block_steps": [],           # steps taken per block
        "block_token_difficulty": [], # step-0 confidence per block
        "schedule_thresholds": [],   # θ(t) values logged per block
    }
    hidden_state_absdiff_all = torch.zeros((n_layers, 0), device=device)
    key_absdiff_all   = torch.zeros((n_layers, 0), device=device)
    value_absdiff_all = torch.zeros((n_layers, 0), device=device)
    query_absdiff_all = torch.zeros((n_layers, 0), device=device)
    confidence_diff_all = torch.zeros((0,), device=device)

    st_time   = time.time()
    step_cnt  = 0
    last_decode_cnt    = 0
    last_decode_tokens = None
    # Always defined so ES-dLLM forward_start logic never hits NameError
    leftmost_decode_pos = 0

    use_eb = cai_config.use_apd and cai_config.use_error_budget
    error_budget = 0.0
    eb_max = cai_config.error_budget_max
    eb_decay = cai_config.error_budget_decay
    recovery_steps_taken = 0
    recovery_steps_taken = 0

    # ─────────────────────────────────────────────────────────────────────
    # OUTER LOOP: blocks
    # ─────────────────────────────────────────────────────────────────────
    for num_block in range(num_blocks):
        start_index = prompt_len + num_block * block_length - token_offset
        end_index   = prompt_len + (num_block + 1) * block_length - token_offset

        # ── get scheduler for this block (§4.2) ─────────────────────────
        if sched_updater is not None:
            block_scheduler = sched_updater.get_scheduler(num_block)
        else:
            block_scheduler = schedule_table.get_scheduler(num_block)

        block_thresholds_log = []   # for results dict

        # ── per-block step-0 confidence probe (§3.3) ────────────────────
        step0_confidence = None

        # ── §4.4 per-block gating objects ────────────────────────────────
        budget_mgr     = TokenBudgetManager(block_len=block_length) if use_cg else None
        grind_detector = GrindingPhaseDetector() if use_cg else None
        pos_scheduler  = None

        # ─────────────────────────────────────────────────────────────────
        # INNER LOOP: denoising steps within a block
        # ─────────────────────────────────────────────────────────────────
        for step in range(block_length):

            # ── determine forward range (same as ES-dLLM) ─────────────
            if ESdLLM_mode is not None:
                if step_cnt == 0:
                    forward_start, forward_end = 0, max_length
                    all_update = True
                else:
                    forward_start = start_index if step != 0 else leftmost_decode_pos
                    if step_cnt % prompt_update_freq == 0:
                        forward_start = 0
                    forward_end  = end_index
                    all_update = (
                        (step_cnt % block_update_freq == 0)
                        or (step_cnt % prompt_update_freq == 0)
                        or (step == delay_step and sparse_kv != 1.0)
                    )
            else:
                if step == 0 or not use_kvcache:
                    forward_start, forward_end = 0, max_length
                else:
                    forward_start, forward_end = start_index, end_index
                all_update = True

            # ── ES-dLLM token budget per layer ────────────────────────
            if not all_update and decode_proportions is not None:
                for bi in range(n_layers):
                    data_caches[bi]["ES_token_cnt"] = max(
                        int(decode_proportions[bi] * (forward_end - forward_start)),
                        min(last_decode_cnt + 2, end_index - start_index),
                    )
            else:
                for bi in range(n_layers):
                    data_caches[bi]["ES_token_cnt"] = None

            # ── sparse KV flags ───────────────────────────────────────
            for bi in range(n_layers):
                data_caches[bi]["inference_start_index"] = start_index
                data_caches[bi]["inference_end_index"]   = end_index
                if step == 0:
                    data_caches[bi]["sparse_key"]   = None
                    data_caches[bi]["sparse_value"] = None
                    data_caches[bi]["valid_index"]  = mask
                if sparse_kv != 1.0 and step == delay_step:
                    data_caches[bi]["record_sparse_kv"]  = True
                    data_caches[bi]["sparse_kv_ratio"]   = sparse_kv
                else:
                    data_caches[bi]["record_sparse_kv"]  = False

            # ── mask index ────────────────────────────────────────────
            if token_offset != 0:
                mask_index = (x[:, :-token_offset] == mask_id).clone()
                mask_index[:, :-token_offset] = mask_index[:, token_offset:].clone()
            else:
                mask_index = (x == mask_id)
            mask_index[:, :start_index] = False
            mask_index[:, end_index:]   = False

            if mask_index.sum() == 0:
                step_cnt += block_length - step
                break

            # ── ES-dLLM constant coefficient ──────────────────────────
            if not all_update and constant_coef is not None:
                constant_coef[:, :] = 0.0
                for i in range(batch_size):
                    nzero = mask_index[i].nonzero()
                    if nzero.shape[0] > 0:
                        leftmost_mask_index = nzero[0].item()
                        constant_coef[i, leftmost_mask_index] = 5.0
                    # last_decode_tokens[i] is a python list of int positions
                    if last_decode_tokens is not None and importance_score_alpha < 1.0:
                        for pos_idx in last_decode_tokens[i]:
                            constant_coef[i, pos_idx] = 3.0

            # ── §4.5 error budget recovery pass ──────────────────────
                    print(f"  [EB] recovery pass at step {step} block {num_block}")

            # ── forward pass ─────────────────────────────────────────
            pos = torch.arange(
                forward_start, forward_end, dtype=torch.long, device=device
            ).expand(batch_size, -1)
            input_x = torch.gather(x, 1, pos)
            cur_mask = torch.gather(mask, 1, pos)

            logits = model(
                input_x,
                attention_mask=cur_mask,
                position_ids=pos,
                data_caches=data_caches,
                use_cache=use_kvcache,
            ).logits

            if not all_update:
                pos = data_caches[n_layers - 1]["position_ids"]

            # truncate to block range
            if forward_start == 0 and not do_statistics:
                logits = logits[:, start_index:end_index]
                pos    = pos[:, start_index:end_index]

            # ── step-0 confidence probe (§3.3) ────────────────────────
            if step == 0 and cai_config.use_apd:
                step0_confidence = probe_step0_confidence(logits, mask_index, pos)
                if do_statistics:
                    results["block_token_difficulty"].append(
                        classify_token_difficulty(step0_confidence).cpu()
                    )
                if use_cg and budget_mgr is not None:
                    budget_mgr.init_from_step0(step0_confidence, mask_index, pos)
                    pos_scheduler = PositionAwareScheduler(block_scheduler)

            # ── §4.1  APD threshold ───────────────────────────────────
            if cai_config.use_apd:
                theta = block_scheduler.get_threshold(step, block_length)
            else:
                theta = generation_kwargs.get("threshold", 0.9)

            block_thresholds_log.append(theta)

            # §4.5 error budget recovery
            if use_eb and step > 0 and error_budget > cai_config.error_budget_max:
                rec_pos = torch.arange(0, max_length, dtype=torch.long, device=device).expand(batch_size, -1)
                logits = model(
                    torch.gather(x, 1, rec_pos),
                    attention_mask=torch.gather(mask, 1, rec_pos),
                    position_ids=rec_pos,
                    data_caches=data_caches,
                    use_cache=use_kvcache,
                ).logits[:, start_index:end_index]
                error_budget = 0.0
                recovery_steps_taken += 1

            # ── token sampling ────────────────────────────────────────
            if cai_config.use_apd:
                if use_cg and pos_scheduler is not None:
                    x0, x0_p, select_index = confidence_gated_sample(
                        logits         = logits,
                        pos            = pos,
                        mask_index     = mask_index,
                        all_confidence = all_confidence,
                        step           = step,
                        total_steps    = block_length,
                        block_start    = start_index,
                        budget_mgr     = budget_mgr,
                        pos_scheduler  = pos_scheduler,
                        grind_detector = grind_detector,
                        eos_token      = eos_token,
                        temperature    = temperature,
                        exist_eos      = (x[:, prompt_len:] == eos_token).any(dim=-1)
                                         if eos_token != -1 else None,
                    )
                else:
                    x0, x0_p, select_index = apd_sample_tokens(
                        logits      = logits,
                        pos         = pos,
                        mask_index  = mask_index,
                        all_confidence = all_confidence,
                        threshold   = theta,
                        eos_token   = eos_token,
                        temperature = temperature,
                        remasking   = remasking,
                        exist_eos   = (x[:, prompt_len:] == eos_token).any(dim=-1)
                                      if eos_token != -1 else None,
                    )
            else:
                # fall back to ES-dLLM sampler
                if model_type == "llada":
                    x0, x0_p, select_index = sample_tokens_LLaDA(
                        logits, pos, mask_index, all_confidence,
                        decode_token_cnt=token_per_step,
                        threshold=theta, temperature=temperature,
                        eos_token=eos_token,
                        exist_eos=(x[:, prompt_len:] == eos_token).any(dim=-1),
                        parallel_mode=True,
                    )
                else:
                    x0, x0_p, select_index = sample_tokens_Dream(
                        logits, pos, mask_index, all_confidence,
                        decode_token_cnt=token_per_step,
                        threshold=theta, temperature=temperature,
                        eos_token=eos_token,
                        exist_eos=(x[:, prompt_len:] == eos_token).any(dim=-1),
                        parallel_mode=True,
                    )

            # ── commit tokens ─────────────────────────────────────────
            last_decode_cnt = max(idx.shape[0] for idx in select_index)
            for i in range(batch_size):
                x[i, pos[i][select_index[i]] + token_offset] = x0[i, select_index[i]]
                decoding_confidence[i, pos[i][select_index[i]]] = x0_p[i, select_index[i]].to(decoding_confidence.dtype)

            # reset confidence for re-masked tokens
            last_decode_tokens_list = [pos[i, select_index[i]] for i in range(batch_size)]
            for i in range(batch_size):
                all_confidence[i, last_decode_tokens_list[i]] = 0.0

            # leftmost position committed this step (used by ES-dLLM forward_start)
            _mins = [t.min().item() if t.numel() > 0 else 10000 for t in last_decode_tokens_list]
            leftmost_decode_pos = min(_mins)
            last_decode_tokens = [t.tolist() for t in last_decode_tokens_list]

            # ── §4.5 update error budget ─────────────────────────────
            if use_eb:
                still_masked = mask_index.float()
                n_masked = still_masked.sum().clamp(min=1.0)
                avg_unc = ((1.0 - all_confidence.abs()) * still_masked).sum() / n_masked
                error_budget = eb_decay * error_budget + (1.0 - eb_decay) * avg_unc.item()

            # ── block convergence tracking (§4.2) ─────────────────────
            if conv_tracker is not None:
                n_unmasked = sum(idx.shape[0] for idx in select_index)
                conv_tracker.record(num_block, step, n_unmasked)
                if use_cg and grind_detector is not None:
                    remaining = mask_index.sum().item()
                    grind_detector.record_step(n_unmasked, remaining, step)

            if do_statistics:
                (hidden_state_absdiff_all, key_absdiff_all,
                 value_absdiff_all, query_absdiff_all,
                 confidence_diff_all) = statistics_per_step(
                    results, data_caches, x0, x0_p, step, n_layers,
                    mask_index, hidden_state_absdiff_all, key_absdiff_all,
                    value_absdiff_all, query_absdiff_all, confidence_diff_all
                )

            if cai_config.print_log:
                print(f"Step {step+1}/{block_length} Block {num_block+1}/{num_blocks} "
                      f"θ={theta:.3f} committed={last_decode_cnt}")

            step_cnt += 1

        # ── post-block: record schedule trace ────────────────────────────
        results["block_steps"].append(
            conv_tracker._stats[num_block].steps_taken if conv_tracker else step_cnt
        )
        results["schedule_thresholds"].append(block_thresholds_log)

        # ── online schedule adaptation (§4.2, optional) ─────────────────
        if sched_updater is not None and num_block + 1 < num_blocks:
            actual_steps = conv_tracker._stats[num_block].steps_taken if conv_tracker else step_cnt
            sched_updater.update_after_block(
                finished_block_idx=num_block,
                actual_steps=actual_steps,
                next_block_idx=num_block + 1,
            )

    # ─────────────────────────────────────────────────────────────────────
    # Finalize
    # ─────────────────────────────────────────────────────────────────────
    if do_statistics:
        statistics_final(results, n_layers,
                         hidden_state_absdiff_all, key_absdiff_all,
                         value_absdiff_all, query_absdiff_all,
                         confidence_diff_all)

    torch.cuda.synchronize()
    ed_time = time.time()
    results["time_info"] = f"Total time: {ed_time - st_time:.2f}s | Steps: {step_cnt}"

    if conv_tracker is not None and cai_config.print_log:
        conv_tracker.print_summary()

    if token_offset != 0:
        x = x[:, :-token_offset]

    # ── log FLOP savings if tier_table was used ───────────────────────────
    if tier_table is not None:
        flop_info = estimate_flop_savings(
            tier_table,
            n_steps   = step_cnt,
            block_len = block_length,
            d_model   = hidden_dim,
            seq_len   = max_length,
        )
        results["flop_savings"] = flop_info
        if cai_config.print_log:
            print(f"Estimated FLOP reduction: {flop_info['reduction_factor']:.2f}×")

    if do_statistics and recovery_steps_taken > 0:
        results["error_budget_recovery_steps"] = recovery_steps_taken

    # Update generate.py global counters so print_statistics() works
    import generate as _gen
    _gen.request_cnt += batch_size
    _gen.total_effective_tokens += int((x[:, prompt_len:] != mask_id).sum().item())

    return x, results
