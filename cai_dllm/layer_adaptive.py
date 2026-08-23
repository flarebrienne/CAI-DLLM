"""
layer_adaptive.py  ─  CAI-dLLM  §4.3  Layer-Adaptive Compute
=============================================================
From Experiments 1 & 2:

  Layer Drift (Exp 1):
    • Early   layers  0– 8  :  2– 3% avg drift  → can skip every-other step
    • Middle  layers  9–16  :  3– 6% avg drift  → run every step, prune KV
    • Deep    layers 17–31  :  6–14% avg drift  → always full precision

  Attention Sparsity (Exp 2):
    • Layer 16 : 70% of attention on top-10 tokens → keep only 20-30 % KV
    • Layers 0,4: broad (130-216 effective tokens)  → keep 50-70 % KV

Three orthogonal axes (§4.3 paragraph 2):
  Axis A  –  Precision tier    : float16 / bfloat16 / int8 for early layers
  Axis B  –  Step-skipping     : compute early layers every other step
  Axis C  –  KV retention      : sparse attention via top-k KV selection

Combined these multiply:  1.2 × 1.2 × 1.3 ≈ 1.87× on top of base APD.

The module provides:
  1. LayerTierConfig    – maps each layer to its tier + hyper-params
  2. LayerSkipManager   – decides whether a layer should run this step
  3. AdaptiveKVPruner   – sparse KV retention (replaces record_sparse_kvcache)
  4. LayerAdaptiveForward – drop-in wrapper around LLaDALlamaBlock_forward
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Constants  (from Experiment 1 + 2 measurements on LLaDA-8B)
# ─────────────────────────────────────────────────────────────────────────────

EARLY_LAYERS  = list(range(0,  9))    # layers 0–8   : 2–3 % drift
MIDDLE_LAYERS = list(range(9,  17))   # layers 9–16  : 3–6 % drift
DEEP_LAYERS   = list(range(17, 32))   # layers 17–31 : 6–14% drift

# Sparse KV retention ratios from Exp 2
KV_RETENTION_LAYER16  = 0.25    # layer 16: 20-30 % of KV entries
KV_RETENTION_EARLY    = 0.60    # layers 0-4: 50-70 %
KV_RETENTION_DEFAULT  = 1.00    # all other layers: no pruning


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Layer tier config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerTierConfig:
    """
    Specifies the compute policy for every transformer layer.

    Fields
    ------
    tier            : 0=EARLY, 1=MIDDLE, 2=DEEP
    step_stride     : run this layer every `step_stride` denoising steps
                      (1 = every step, 2 = every other step)
    kv_retention    : fraction of KV entries to keep (1.0 = full attention)
    precision       : "bf16" | "int8"   (int8 not yet wired; placeholder)
    sparse_layer    : True if kv_retention < 1.0
    """
    layer_idx:    int
    tier:         int     = 2        # default DEEP (safe)
    step_stride:  int     = 1        # default: run every step
    kv_retention: float   = 1.00     # default: no pruning
    precision:    str     = "bf16"   # "bf16" | "int8"

    @property
    def sparse_layer(self) -> bool:
        return self.kv_retention < 1.0

    @property
    def tier_name(self) -> str:
        return ["EARLY", "MIDDLE", "DEEP"][self.tier]


class LayerTierTable:
    """
    Full table of LayerTierConfig objects for all N layers of the model.

    Construction
    ------------
    Use `LayerTierTable.for_llada_8b()` for the default LLaDA-8B setup
    that matches Experiments 1 & 2.  Or build with `from_drift_data()`
    if you have your own drift measurements.
    """

    def __init__(self, configs: List[LayerTierConfig]):
        self.configs = configs
        self._by_idx: Dict[int, LayerTierConfig] = {c.layer_idx: c for c in configs}

    # ── factory: default LLaDA-8B config ─────────────────────────────────
    @classmethod
    def for_llada_8b(cls) -> "LayerTierTable":
        """
        Default config derived from Experiments 1 & 2 on LLaDA-8B (32 layers).

        EARLY  (0–8)  : step_stride=2, kv_retention=0.60 (layers 0,4 per Exp 2)
        MIDDLE (9–16) : step_stride=1, kv_retention=0.25 for layer 16, else 0.70
        DEEP  (17–31) : step_stride=1, kv_retention=1.00
        """
        configs = []
        for l in range(32):
            if l in EARLY_LAYERS:
                # Layers 0 & 4 are broadest (Exp 2) → slightly more KV retention
                kv = KV_RETENTION_EARLY if l in (0, 4) else 0.55
                configs.append(LayerTierConfig(
                    layer_idx=l, tier=0,
                    step_stride=2, kv_retention=kv, precision="bf16"
                ))
            elif l in MIDDLE_LAYERS:
                kv = KV_RETENTION_LAYER16 if l == 16 else 0.65
                configs.append(LayerTierConfig(
                    layer_idx=l, tier=1,
                    step_stride=1, kv_retention=kv, precision="bf16"
                ))
            else:  # DEEP
                configs.append(LayerTierConfig(
                    layer_idx=l, tier=2,
                    step_stride=1, kv_retention=KV_RETENTION_DEFAULT, precision="bf16"
                ))
        return cls(configs)

    @classmethod
    def from_drift_data(
        cls,
        avg_drifts: Sequence[float],   # one float per layer
        n_layers: int,
        early_thresh:  float = 0.04,   # avg drift < 4 % → EARLY tier
        middle_thresh: float = 0.08,   # avg drift < 8 % → MIDDLE tier
    ) -> "LayerTierTable":
        """
        Build a tier table from measured per-layer drift values.
        Useful when fine-tuning the tiers to a new model or dataset.
        """
        configs = []
        for l, drift in enumerate(avg_drifts[:n_layers]):
            if drift < early_thresh:
                configs.append(LayerTierConfig(
                    layer_idx=l, tier=0,
                    step_stride=2, kv_retention=0.60, precision="bf16"
                ))
            elif drift < middle_thresh:
                configs.append(LayerTierConfig(
                    layer_idx=l, tier=1,
                    step_stride=1, kv_retention=0.65, precision="bf16"
                ))
            else:
                configs.append(LayerTierConfig(
                    layer_idx=l, tier=2,
                    step_stride=1, kv_retention=1.00, precision="bf16"
                ))
        return cls(configs)

    # ── accessors ─────────────────────────────────────────────────────────
    def __getitem__(self, layer_idx: int) -> LayerTierConfig:
        return self._by_idx[layer_idx]

    def __len__(self) -> int:
        return len(self.configs)

    def flop_proportion(self) -> float:
        """
        Estimate fraction of FLOPs used vs. a full run.
        Each layer contributes (1/step_stride) × kv_retention of its share.
        """
        n = len(self.configs)
        total = sum(
            (1.0 / c.step_stride) * c.kv_retention
            for c in self.configs
        )
        return total / n

    def print_summary(self):
        print(f"LayerTierTable ({len(self.configs)} layers)")
        print(f"  Estimated FLOP proportion: {self.flop_proportion():.2f}×")
        print("  ─" * 25)
        for c in self.configs:
            print(f"  L{c.layer_idx:02d}  {c.tier_name:6s}  "
                  f"stride={c.step_stride}  kv={c.kv_retention:.2f}  {c.precision}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Layer skip manager  (Axis B: step-skipping)
# ─────────────────────────────────────────────────────────────────────────────

class LayerSkipManager:
    """
    Decides, for each (layer, global_step), whether to run a full forward
    pass or reuse the previous layer output.

    For EARLY layers with step_stride = 2:
      • Even steps  : compute normally, cache the output
      • Odd  steps  : skip computation, return cached output

    The cache is stored in data_cache['layer_skip_cache'] (a tensor of
    shape [B, active_tokens, d_model]).
    """

    def __init__(self, tier_table: LayerTierTable):
        self.tier_table = tier_table

    def should_skip(self, layer_idx: int, global_step: int) -> bool:
        """
        Returns True if this layer should be skipped on this step.
        A skipped layer reuses its cached hidden state from the last run.
        """
        cfg = self.tier_table[layer_idx]
        if cfg.step_stride == 1:
            return False
        # stride=2: skip on odd steps
        return (global_step % cfg.step_stride) != 0

    def get_cached_output(
        self,
        data_cache: dict,
        hidden_state: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return (hidden_state, position_ids) from cache if available,
        otherwise return the current hidden state unchanged.

        The cache is populated by `store_output`.
        """
        cached = data_cache.get("layer_skip_cache", None)
        if cached is None:
            # First ever call – no cache yet, return current state
            return hidden_state, position_ids
        return cached["hidden_state"], cached["position_ids"]

    def store_output(
        self,
        data_cache: dict,
        hidden_state: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ):
        """Cache the layer output for potential reuse next step."""
        data_cache["layer_skip_cache"] = {
            "hidden_state": hidden_state.detach(),
            "position_ids": position_ids.detach() if position_ids is not None else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Adaptive KV pruner  (Axis C: sparse attention)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveKVPruner:
    """
    Layer-specific sparse KV retention.  Replaces / extends the
    `record_sparse_kvcache` function in early_skipping.py.

    Strategy (from Experiment 2):
      • Use mean query attention scores to rank outside-block KV entries.
      • Keep the top-k_retention fraction.
      • Apply 3-token max-pool smoothing before ranking (from ES-dLLM).

    This is applied once per block at `delay_step` (same as ES-dLLM's
    sparse kv trigger) but with a layer-specific retention ratio.
    """

    def __init__(self, tier_table: LayerTierTable, kernel_size: int = 3):
        self.tier_table  = tier_table
        self.kernel_size = kernel_size

    def should_apply(self, layer_idx: int, data_cache: dict) -> bool:
        """True if this layer uses sparse attention and hasn't been pruned yet."""
        cfg = self.tier_table[layer_idx]
        return cfg.sparse_layer and data_cache.get("record_sparse_kv", False)

    @torch.no_grad()
    def prune(
        self,
        data_cache: dict,
        layer_idx:  int,
        query:      torch.Tensor,    # (B, n_heads, active_tokens, head_dim)
        key:        torch.Tensor,    # (B, n_kv_heads, full_seq_len, head_dim)
        value:      torch.Tensor,    # (B, n_kv_heads, full_seq_len, head_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune the outside-block KV cache to `kv_retention` fraction.

        Returns (pruned_key, pruned_value) with dimensions:
          [B, n_kv_heads, kept_tokens + block_tokens, head_dim]

        The current block's KV entries are always kept in full (they are
        needed for intra-block attention correctness).
        """
        cfg = self.tier_table[layer_idx]
        start = data_cache.get("inference_start_index", 0)
        end   = data_cache.get("inference_end_index",   key.shape[2])

        # ── split block vs outside-block KV ──────────────────────────────
        key_outside = torch.cat([key[:, :, :start, :], key[:, :, end:, :]], dim=2)
        val_outside = torch.cat([value[:, :, :start, :], value[:, :, end:, :]], dim=2)
        key_block   = key[:, :, start:end, :]
        val_block   = value[:, :, start:end, :]

        n_outside = key_outside.shape[2]
        if n_outside == 0:
            return key, value    # nothing to prune

        keep_n = max(1, int(n_outside * cfg.kv_retention))

        # ── score outside-block tokens by mean query attention ────────────
        n_kv_heads = key_outside.shape[1]
        n_q_heads  = query.shape[1]
        if n_kv_heads != n_q_heads:
            # GQA: expand KV heads to match query heads
            k_for_score = key_outside.repeat_interleave(n_q_heads // n_kv_heads, dim=1)
        else:
            k_for_score = key_outside

        avg_q = query.mean(dim=-2)                             # (B, n_q_heads, head_dim)
        scores = torch.matmul(
            avg_q.unsqueeze(-2),                               # (B, nh, 1, hd)
            k_for_score.transpose(-2, -1)                      # (B, nh, hd, n_out)
        ).squeeze(-2)                                          # (B, nh, n_out)
        importance = scores.mean(dim=1)                        # (B, n_out)

        # max-pool smoothing (ES-dLLM trick – prevents rank instability)
        importance = F.max_pool1d(
            importance.unsqueeze(1),
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        ).squeeze(1)                                           # (B, n_out)

        # ── select top-k indices ──────────────────────────────────────────
        _, keep_idx = torch.topk(importance, k=keep_n, dim=-1)  # (B, keep_n)
        keep_idx_sorted, _ = keep_idx.sort(dim=-1)

        # gather pruned KV
        pruned_key = key_outside.take_along_dim(
            keep_idx_sorted[:, None, :, None].expand(-1, key_outside.shape[1], -1, key_outside.shape[3]), dim=2
        )
        pruned_val = val_outside.take_along_dim(
            keep_idx_sorted[:, None, :, None].expand(-1, val_outside.shape[1], -1, val_outside.shape[3]), dim=2
        )

        # ── reassemble: [pruned_outside | current_block] ─────────────────
        final_key = torch.cat([pruned_key, key_block], dim=2)
        final_val = torch.cat([pruned_val, val_block], dim=2)

        return final_key, final_val


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Layer-adaptive forward wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LayerAdaptiveForward:
    """
    Drop-in orchestrator that wraps each transformer block's forward method
    and applies the three compute-reduction axes:

      A. Precision tier   – placeholder; actual quantisation not yet wired
      B. Step-skipping    – skip early layers on odd denoising steps
      C. Sparse attention – per-layer KV pruning via AdaptiveKVPruner

    Usage in LLaDAModel_forward (replacing the plain block() call):

        la_fwd = LayerAdaptiveForward(tier_table)

        for block_idx, block in enumerate(self.transformer.blocks):
            layer_past = data_caches[block_idx]
            x, position_ids, attention_mask = la_fwd(
                block, block_idx, x,
                position_ids=position_ids,
                attention_bias=attention_mask,
                data_cache=layer_past,
                use_cache=use_cache,
                global_step=global_step,      # ← pass from outer loop
            )
    """

    def __init__(self, tier_table: LayerTierTable):
        self.tier_table  = tier_table
        self.skip_mgr    = LayerSkipManager(tier_table)
        self.kv_pruner   = AdaptiveKVPruner(tier_table)

    # ------------------------------------------------------------------
    def __call__(
        self,
        block,                           # LLaDALlamaBlock instance
        layer_idx:     int,
        x:             torch.Tensor,     # (B, active_tokens, d_model)
        position_ids:  Optional[torch.Tensor],
        attention_bias: Optional[torch.Tensor],
        data_cache:    dict,
        use_cache:     bool,
        global_step:   int,              # monotonically increasing denoising step
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (x, position_ids, attention_bias) – same signature as
        LLaDALlamaBlock_forward.
        """
        cfg = self.tier_table[layer_idx]

        # ── Axis B: step-skipping ─────────────────────────────────────────
        if self.skip_mgr.should_skip(layer_idx, global_step):
            # Return cached output from last computed step
            x_cached, pos_cached = self.skip_mgr.get_cached_output(
                data_cache, x, position_ids
            )
            # x_cached may have a different token count if ES-dLLM also
            # pruned tokens; we fall back to the raw x if shapes mismatch
            if x_cached.shape == x.shape:
                return x_cached, position_ids, attention_bias
            # shape mismatch (token count changed) – fall through to full forward
            # (safety: never skip a layer when the active token set has changed)

        # ── Axis C: sparse KV (inject into data_cache before forward) ────
        # The actual KV pruning happens inside LLaDABlock_attention →
        # update_fetch_kvcache.  We set `record_sparse_kv` and `sparse_kv_ratio`
        # in data_cache so the existing early_skipping.record_sparse_kvcache
        # uses our layer-specific ratio instead of a global one.
        original_ratio = data_cache.get("sparse_kv_ratio", 1.0)
        original_record = data_cache.get("record_sparse_kv", False)

        if self.kv_pruner.should_apply(layer_idx, data_cache):
            data_cache["sparse_kv_ratio"] = cfg.kv_retention
            # flag remains True; early_skipping.record_sparse_kvcache will fire

        # ── actual forward pass ──────────────────────────────────────────
        x_out, pos_out, attn_out = block(
            x,
            position_ids=position_ids,
            attention_bias=attention_bias,
            data_cache=data_cache,
            use_cache=use_cache,
        )

        # ── restore data_cache fields ────────────────────────────────────
        data_cache["sparse_kv_ratio"] = original_ratio
        data_cache["record_sparse_kv"] = original_record

        # ── Axis B: store output for future skipping ──────────────────────
        if cfg.step_stride > 1:
            self.skip_mgr.store_output(data_cache, x_out, pos_out)

        return x_out, pos_out, attn_out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Utility: flop budget estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_flop_savings(
    tier_table: LayerTierTable,
    n_steps:    int,
    block_len:  int,
    d_model:    int,
    seq_len:    int,
) -> dict:
    """
    Rough FLOPs estimate comparing full vs. layer-adaptive inference.

    All numbers are in units of "full-block full-step attention FLOPs".

    Returns a dict with:
      full_flops       : baseline
      adaptive_flops   : estimated under LayerAdaptiveForward
      reduction_factor : full / adaptive
    """
    n_layers = len(tier_table)

    # per-layer cost per step ∝ seq_len × block_len × d_model  (attention)
    # step-skipping halves the cost; KV pruning scales quadratically but we
    # approximate linearly (conservative)
    full_cost = n_layers * n_steps * seq_len * block_len * d_model

    adaptive_cost = 0
    for cfg in tier_table.configs:
        effective_steps = math.ceil(n_steps / cfg.step_stride)
        effective_kv    = cfg.kv_retention
        adaptive_cost  += effective_steps * seq_len * block_len * effective_kv * d_model

    return {
        "full_flops":       full_cost,
        "adaptive_flops":   adaptive_cost,
        "reduction_factor": full_cost / max(adaptive_cost, 1),
    }
