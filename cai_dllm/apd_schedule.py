"""
apd_schedule.py  ─  CAI-dLLM  §4.1  Adaptive Parallel Decoding
================================================================
Implements the cosine-warmup threshold schedule described in the
paper, plus all sampling helpers that consume it.

Key symbols (matching the paper):
  θ_start  : initial (conservative) unmasking confidence threshold
  θ_end    : final  (aggressive)    unmasking confidence threshold
  w        : warmup fraction  (default 0.15, i.e. first 15 % of steps)
  T        : total denoising steps for one block
  t        : current step index within the block  (0 … T-1)

Schedule formula (§4.1):
  During warmup  (t < w·T):
      θ(t) = θ_start

  After warmup:
      θ(t) = θ_end + (θ_start − θ_end) · ½ [1 + cos(π · (t−wT)/(T−wT))]

This matches the empirical finding that the first ~10 steps are the
critical "discovery phase" where wrong early commitments cascade.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Schedule dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APDScheduleConfig:
    """
    All hyper-parameters for a single APD threshold schedule.
    One instance is shared across all denoising steps of a block.
    For per-block adaptation (§4.2) you create one APDScheduleConfig
    per block with different θ_end values.
    """
    theta_start:  float = 0.90   # conservative threshold at step 0
    theta_end:    float = 0.70   # aggressive  threshold at final step
    warmup_frac:  float = 0.15   # fraction of steps held at θ_start
    shape:        str   = "cosine"  # "cosine" | "linear"

    # ── derived (set by APDScheduler.__post_init__) ────────────────────────
    _total_steps: int  = field(default=0, init=False, repr=False)

    def bind(self, total_steps: int) -> "APDScheduleConfig":
        """Attach a step count so get_threshold(t) works correctly."""
        self._total_steps = total_steps
        return self

    # ── Pareto-optimal presets from Experiment 4 ──────────────────────────
    @classmethod
    def quality(cls) -> "APDScheduleConfig":
        """100 % accuracy, 5.07× speedup (default production mode)."""
        return cls(theta_start=0.90, theta_end=0.70, warmup_frac=0.15, shape="cosine")

    @classmethod
    def balanced(cls) -> "APDScheduleConfig":
        """80 % accuracy, 6.05× speedup (interactive chat)."""
        return cls(theta_start=0.50, theta_end=0.50, warmup_frac=0.00, shape="cosine")

    @classmethod
    def speed(cls) -> "APDScheduleConfig":
        """60 % accuracy, 7.92× speedup (brainstorming)."""
        return cls(theta_start=0.90, theta_end=0.70, warmup_frac=0.00, shape="cosine")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Scheduler  –  converts (step_index, total_steps) → θ(t)
# ─────────────────────────────────────────────────────────────────────────────

class APDScheduler:
    """
    Stateless schedule calculator.  Call get_threshold(t, T) at every
    denoising step to retrieve the current threshold.

    Example
    -------
    >>> cfg = APDScheduleConfig.quality()
    >>> sched = APDScheduler(cfg)
    >>> for t in range(64):
    ...     theta = sched.get_threshold(t, total_steps=64)
    """

    def __init__(self, config: APDScheduleConfig):
        self.cfg = config

    # ------------------------------------------------------------------
    def get_threshold(self, t: int, total_steps: int) -> float:
        """
        Return θ(t) for step t out of total_steps.

        Parameters
        ----------
        t            : current step (0-indexed, 0 … total_steps-1)
        total_steps  : T in the paper
        """
        cfg = self.cfg
        warmup_end = cfg.warmup_frac * total_steps          # w·T

        # ── Phase 1: warmup ────────────────────────────────────────────
        if t < warmup_end:
            return cfg.theta_start

        # ── Phase 2: decay ─────────────────────────────────────────────
        decay_range = total_steps - warmup_end              # T − w·T
        if decay_range <= 0:                                # edge case: w = 1
            return cfg.theta_end

        progress = (t - warmup_end) / decay_range           # 0 → 1

        if cfg.shape == "cosine":
            # half-cosine: starts at θ_start, ends at θ_end
            cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cfg.theta_end + (cfg.theta_start - cfg.theta_end) * cos_val
        elif cfg.shape == "linear":
            return cfg.theta_start + progress * (cfg.theta_end - cfg.theta_start)
        else:
            raise ValueError(f"Unknown schedule shape: {cfg.shape!r}")

    # ------------------------------------------------------------------
    def get_all_thresholds(self, total_steps: int) -> List[float]:
        """Precompute all T thresholds for a block (useful for logging)."""
        return [self.get_threshold(t, total_steps) for t in range(total_steps)]

    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"APDScheduler(θ_start={self.cfg.theta_start}, "
                f"θ_end={self.cfg.theta_end}, w={self.cfg.warmup_frac}, "
                f"shape={self.cfg.shape!r})")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Token sampling with APD threshold   (replaces sample_tokens_LLaDA)
# ─────────────────────────────────────────────────────────────────────────────

def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max trick for categorical sampling (unchanged from ES-dLLM)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def apd_sample_tokens(
    logits:       torch.Tensor,          # (B, block_len, vocab)
    pos:          torch.Tensor,          # (B, block_len) – absolute token positions
    mask_index:   torch.Tensor,          # (B, full_seq_len) – True where still masked
    all_confidence: torch.Tensor,        # (B, full_seq_len) – running confidence scores
    threshold:    float,                 # θ(t) from APDScheduler
    eos_token:    int    = -1,
    temperature:  float  = 0.0,
    remasking:    str    = "low_confidence",
    exist_eos:    Optional[torch.Tensor] = None,  # (B,) bool
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    Parallel (threshold-based) token selection for one denoising step.

    Returns
    -------
    x0            : (B, block_len)  – predicted tokens for all positions
    x0_p          : (B, block_len)  – confidence scores for all positions
    selected_idx  : list[Tensor]    – per-sample indices (within pos) to commit
    """
    B = logits.shape[0]

    # ── predict tokens ────────────────────────────────────────────────────
    logits_with_noise = _add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)                # (B, block_len)

    # ── compute confidence ────────────────────────────────────────────────
    if remasking == "low_confidence":
        p = F.softmax(logits.float(), dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)   # (B, block_len)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise ValueError(f"Unknown remasking strategy: {remasking!r}")

    confidence_with_unmask = x0_p.clone()                       # returned for stats

    # ── suppress EOS before a real EOS exists ────────────────────────────
    if eos_token != -1 and exist_eos is not None:
        # zero out eos confidence if eos not yet generated AND this is the last
        # still-masked position in the block (mirrors ES-dLLM logic)
        last_mask_in_block = mask_index.gather(
            -1, torch.clamp(pos - 1, min=0)
        )
        eos_suppress = (
            (x0 == eos_token)
            & (~exist_eos.unsqueeze(1).expand_as(x0))
            & last_mask_in_block
        )
        x0_p = x0_p.masked_fill(eos_suppress, 0.0)

    # ── zero out already-unmasked positions ──────────────────────────────
    still_masked = mask_index.gather(-1, pos)                    # (B, block_len)
    x0_p = x0_p.masked_fill(~still_masked, 0.0)

    # ── update global confidence tensor ──────────────────────────────────
    all_confidence.scatter_(1, pos, x0_p.to(all_confidence.dtype))

    # ── apply APD threshold ───────────────────────────────────────────────
    # Always commit at least 1 token (the most confident masked one)
    commit_mask = x0_p > threshold                               # (B, block_len)
    # guarantee at least one commit per sample
    top1 = torch.topk(x0_p, k=1, dim=-1).indices
    commit_mask.scatter_(-1, top1, True)
    # never commit already-unmasked tokens
    commit_mask = commit_mask & (x0_p > 0.0)

    selected_idx = [commit_mask[i].nonzero().squeeze(1) for i in range(B)]

    return x0, confidence_with_unmask, selected_idx


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Step-0 confidence probe  (§3.3 – free difficulty signal)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_step0_confidence(
    logits: torch.Tensor,          # (B, block_len, vocab)
    mask_index: torch.Tensor,      # (B, full_seq_len)
    pos: torch.Tensor,             # (B, block_len)
) -> torch.Tensor:
    """
    Extract per-token confidence at step 0 (no-cost difficulty signal).

    From Experiment 3: tokens with step-0 confidence > 0.50 almost
    always lock in by step 5; tokens < 0.25 need 30-50+ more steps.

    Returns
    -------
    confidence0 : (B, block_len) – softmax probability of argmax prediction,
                  zeroed for already-unmasked positions.
    """
    p = F.softmax(logits.float(), dim=-1)
    max_p, _ = p.max(dim=-1)                                    # (B, block_len)
    still_masked = mask_index.gather(-1, pos)
    return max_p.masked_fill(~still_masked, 0.0)


@torch.no_grad()
def classify_token_difficulty(
    confidence0: torch.Tensor,     # (B, block_len) from probe_step0_confidence
    easy_thresh:  float = 0.50,
    hard_thresh:  float = 0.25,
) -> torch.Tensor:
    """
    Returns a (B, block_len) integer tensor:
        0 = EASY   (locks by ~step 5)
        1 = MEDIUM (locks by ~step 30)
        2 = HARD   (needs 30-50+ steps)
    """
    tier = torch.full_like(confidence0, fill_value=1, dtype=torch.long)
    tier[confidence0 > easy_thresh] = 0
    tier[confidence0 < hard_thresh] = 2
    return tier
