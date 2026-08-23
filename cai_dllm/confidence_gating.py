"""
confidence_gating.py  ─  CAI-dLLM  §4.4  Confidence-Gated Early Exit
=====================================================================
Directly exploits Experiment 3 findings to eliminate the grinding phase.

KEY INSIGHT (Exp 3, Finding 3):
  - Tokens with step-0 confidence > 0.50 → locked by step 5   (EASY)
  - Tokens with step-0 confidence 0.22-0.41 → locked by step 30 (MID)
  - Tokens with step-0 confidence < 0.25 → still unsure at step 50 (HARD)

  Step-0 confidence is a FREE signal — no extra computation needed.

KEY INSIGHT (Exp 3, Finding 2):
  - Positions 0-31 (first half of block): avg lock step 12-14   (EASY)
  - Positions 32-47 (late-mid):          avg lock step 29.8     (MEDIUM)
  - Positions 48-63 (end):               avg lock step 44.3     (HARD)

  Position within block is a second free signal.

KEY INSIGHT (Exp 3, Finding 4 — the grinding phase):
  - Steps 0-10:  3.1% tokens decided per step  (HIGH efficiency)
  - Steps 11-30: 0.8% tokens decided per step  (MODERATE)
  - Steps 31-50: 1.2% tokens decided per step  (LOW — the grind)
  - Steps 51-64: 2.1% tokens decided per step  (MODERATE recovery)

  The grinding phase (steps 31-50) is the primary waste target.

SOLUTION — Two complementary mechanisms:

  A) ConfidenceGatedAPD:
     At step 0, classify every token as EASY/MID/HARD using confidence.
     Assign each token its own maximum step budget:
       EASY  (conf > 0.50) → max 8 steps   (skip the grind entirely)
       MID   (conf 0.25-0.50) → max 32 steps
       HARD  (conf < 0.25) → max 64 steps  (full budget)
     Once a token's budget is exhausted, force-commit it.

  B) PositionAwareThreshold:
     Override APD cosine θ(t) with a position-aware version:
     Early positions (0-31) use more aggressive θ (lower threshold).
     Late positions  (32-63) use more conservative θ (higher threshold).
     This matches the empirical lock-in distribution from Exp 3.

  C) GrindingPhaseDetector:
     Track tokens/step efficiency in real time.
     When efficiency drops below a threshold for N consecutive steps,
     force-commit all remaining tokens above a fallback confidence.
     This directly cuts the grinding phase short.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Constants from Experiment 3
# ─────────────────────────────────────────────────────────────────────────────

EASY_CONF_THRESH  = 0.50   # tokens above this at step 0 → EASY
HARD_CONF_THRESH  = 0.25   # tokens below this at step 0 → HARD

EASY_MAX_STEPS    = 8      # EASY tokens don't need more than 8 steps
MID_MAX_STEPS     = 32     # MID tokens: skip the grind phase
HARD_MAX_STEPS    = 64     # HARD tokens: full budget

EASY_POS_BOUNDARY = 32     # positions 0-31 → easier (Exp 3 Finding 2)
MID_POS_BOUNDARY  = 48     # positions 32-47 → medium

GRIND_EFFICIENCY_THRESH = 1.5   # tokens/step below this → grinding
GRIND_CONSECUTIVE_STEPS = 4     # steps below threshold before triggering exit
GRIND_FALLBACK_CONF     = 0.40  # force-commit tokens above this during grind exit


# ─────────────────────────────────────────────────────────────────────────────
# A)  Per-token step budget (Exp 3, Finding 3)
# ─────────────────────────────────────────────────────────────────────────────

class TokenBudgetManager:
    """
    Assigns each token position a maximum step budget based on its
    step-0 confidence and block position.

    Once a token's budget is exhausted, it is force-committed at the
    next step regardless of the APD threshold — eliminating the grinding
    phase for easy tokens.

    Usage:
        mgr = TokenBudgetManager(block_len=64)
        mgr.init_from_step0(confidence0, mask_index, pos)

        for step in range(block_len):
            force_commit_mask = mgr.get_force_commit_mask(step, pos, mask_index)
            # merge force_commit_mask with APD's normal commit decisions
            mgr.record_committed(pos, committed_mask)
    """

    def __init__(self, block_len: int = 64):
        self.block_len  = block_len
        self.budgets:   Optional[torch.Tensor] = None   # (B, block_len) int
        self.committed: Optional[torch.Tensor] = None   # (B, block_len) bool
        self.token_tiers: Optional[torch.Tensor] = None # (B, block_len) 0/1/2

    # ------------------------------------------------------------------
    def init_from_step0(
        self,
        confidence0: torch.Tensor,   # (B, block_len)  from probe_step0_confidence
        mask_index:  torch.Tensor,   # (B, full_seq)   True = still masked
        pos:         torch.Tensor,   # (B, block_len)  absolute positions
    ):
        """Call once at step=0 after the first forward pass."""
        B, L = confidence0.shape
        device = confidence0.device

        # ── confidence-based tier ────────────────────────────────────────
        tiers = torch.full((B, L), fill_value=1, dtype=torch.long, device=device)
        tiers[confidence0 > EASY_CONF_THRESH] = 0   # EASY
        tiers[confidence0 < HARD_CONF_THRESH] = 2   # HARD

        # ── position-based tier override (Exp 3 Finding 2) ───────────────
        # Block-relative position (0-63)
        block_start = pos[:, 0].unsqueeze(1)              # (B, 1)
        rel_pos = pos - block_start                       # (B, block_len)

        # Upgrade EASY→MID for late positions, and MID→HARD for very late
        late_mid  = (rel_pos >= EASY_POS_BOUNDARY) & (rel_pos < MID_POS_BOUNDARY)
        late_hard = rel_pos >= MID_POS_BOUNDARY
        tiers = torch.where(late_mid  & (tiers == 0), torch.ones_like(tiers),  tiers)
        tiers = torch.where(late_hard & (tiers <= 1), torch.full_like(tiers,2), tiers)

        # ── assign budgets ────────────────────────────────────────────────
        budgets = torch.full((B, L), fill_value=HARD_MAX_STEPS,
                             dtype=torch.long, device=device)
        budgets[tiers == 0] = EASY_MAX_STEPS
        budgets[tiers == 1] = MID_MAX_STEPS

        # Already-unmasked tokens get budget=0 (never force-commit)
        still_masked = mask_index.gather(-1, pos)
        budgets[~still_masked] = 0

        self.budgets    = budgets
        self.tiers      = tiers
        self.committed  = ~still_masked   # (B, block_len) starts False for masked

    # ------------------------------------------------------------------
    def get_force_commit_mask(
        self,
        step:       int,
        pos:        torch.Tensor,      # (B, active_len)
        mask_index: torch.Tensor,      # (B, full_seq)
    ) -> torch.Tensor:
        """
        Returns (B, active_len) bool — True for tokens whose budget is
        exhausted and should be force-committed this step.
        """
        if self.budgets is None:
            return torch.zeros(pos.shape, dtype=torch.bool, device=pos.device)

        # Map absolute pos → block-relative index
        block_start = pos[:, 0].unsqueeze(1)
        rel_pos = (pos - block_start).clamp(0, self.block_len - 1)

        # Budget exhausted = step >= budget AND still masked
        budget_here = self.budgets.gather(1, rel_pos)          # (B, active_len)
        still_masked = mask_index.gather(-1, pos)              # (B, active_len)
        return (step >= budget_here) & still_masked & (budget_here > 0)

    # ------------------------------------------------------------------
    def record_committed(
        self,
        pos:            torch.Tensor,   # (B, active_len)
        committed_mask: torch.Tensor,   # (B, active_len) bool — newly committed
    ):
        """Mark tokens as committed so they are excluded from future force-commits."""
        if self.committed is None:
            return
        block_start = pos[:, 0].unsqueeze(1)
        rel_pos = (pos - block_start).clamp(0, self.block_len - 1)
        self.committed.scatter_(1, rel_pos, committed_mask)

    # ------------------------------------------------------------------
    def tier_summary(self) -> dict:
        if self.tiers is None:
            return {}
        t = self.tiers
        return {
            "easy":  (t == 0).sum().item(),
            "mid":   (t == 1).sum().item(),
            "hard":  (t == 2).sum().item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# B)  Position-aware threshold override (Exp 3, Finding 2)
# ─────────────────────────────────────────────────────────────────────────────

class PositionAwareScheduler:
    """
    Wraps APDScheduler and overrides θ(t) based on block-relative position.

    For early positions (0-31): use a lower (more aggressive) threshold
    since they statistically lock in by step 12-14.

    For late positions (32-63): use a higher (more conservative) threshold
    since they need up to step 44 on average.

    The per-position threshold is a multiplicative scaling of the base θ(t):
      θ_pos(t, p) = θ(t) × position_scale(p)

    where position_scale(p) < 1 for early positions (more aggressive)
    and position_scale(p) > 1 for late positions (more conservative),
    clipped to [0.20, 0.98].
    """

    # Scale factors derived from Exp 3 lock-in ratios
    # Early (0-15): avg_lock=12.7, late (48-63): avg_lock=44.3 → ratio 3.5×
    EARLY_SCALE  = 0.80   # more aggressive for positions 0-15
    MID_SCALE    = 0.90   # positions 16-31
    LATE_MID_SCALE = 1.05 # positions 32-47
    LATE_SCALE   = 1.10   # positions 48-63 (more conservative)

    def __init__(self, base_scheduler):
        self.base = base_scheduler   # APDScheduler instance

    def get_position_scales(
        self,
        pos:         torch.Tensor,   # (B, active_len) absolute positions
        block_start: int,
    ) -> torch.Tensor:
        """Returns (B, active_len) float scale factors."""
        rel = pos - block_start                          # (B, active_len)
        scales = torch.full_like(pos, fill_value=1.0, dtype=torch.float32)
        scales[rel < 16]                    = self.EARLY_SCALE
        scales[(rel >= 16) & (rel < 32)]    = self.MID_SCALE
        scales[(rel >= 32) & (rel < 48)]    = self.LATE_MID_SCALE
        scales[rel >= 48]                   = self.LATE_SCALE
        return scales

    def get_thresholds(
        self,
        step:        int,
        total_steps: int,
        pos:         torch.Tensor,   # (B, active_len)
        block_start: int,
    ) -> torch.Tensor:
        """
        Returns (B, active_len) per-position thresholds for this step.
        """
        base_theta = self.base.get_threshold(step, total_steps)
        scales     = self.get_position_scales(pos, block_start).to(pos.device)
        thresholds = (base_theta * scales).clamp(0.20, 0.98)
        return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# C)  Grinding phase detector (Exp 3, Finding 4)
# ─────────────────────────────────────────────────────────────────────────────

class GrindingPhaseDetector:
    """
    Monitors tokens-per-step efficiency in real time and triggers a
    forced exit from the grinding phase when progress stalls.

    From Exp 3 Finding 4:
      - The grinding phase (steps 31-50) has efficiency ~1.0 tokens/step
      - The discovery phase (steps 0-10) has efficiency ~3.1 tokens/step
      - We can detect the grind by watching for sustained low efficiency

    When triggered, all remaining masked tokens with confidence above
    GRIND_FALLBACK_CONF are force-committed immediately.
    """

    def __init__(
        self,
        efficiency_thresh:  float = GRIND_EFFICIENCY_THRESH,
        consecutive_steps:  int   = GRIND_CONSECUTIVE_STEPS,
        fallback_conf:      float = GRIND_FALLBACK_CONF,
        min_step:           int   = 15,   # don't trigger before step 15
    ):
        self.efficiency_thresh = efficiency_thresh
        self.consecutive_steps = consecutive_steps
        self.fallback_conf     = fallback_conf
        self.min_step          = min_step

        self._low_eff_count = 0
        self._triggered     = False
        self._history: List[float] = []

    def record_step(self, n_committed: int, n_remaining: int, step: int):
        """Call after each denoising step."""
        efficiency = float(n_committed)
        self._history.append(efficiency)

        if step < self.min_step:
            self._low_eff_count = 0
            return

        if efficiency < self.efficiency_thresh:
            self._low_eff_count += 1
        else:
            self._low_eff_count = 0

    def should_force_exit(self) -> bool:
        """True when grinding phase has been detected and not yet handled."""
        return (
            not self._triggered
            and self._low_eff_count >= self.consecutive_steps
        )

    def trigger(self):
        """Mark as triggered so we don't fire twice."""
        self._triggered = True

    @torch.no_grad()
    def get_force_exit_mask(
        self,
        confidence: torch.Tensor,   # (B, block_len) current confidence scores
        mask_index: torch.Tensor,   # (B, full_seq)
        pos:        torch.Tensor,   # (B, block_len)
    ) -> torch.Tensor:
        """
        Returns (B, block_len) bool — tokens to force-commit during grind exit.
        Only commits tokens that are still masked AND above fallback confidence.
        """
        still_masked = mask_index.gather(-1, pos)
        above_fallback = confidence > self.fallback_conf
        return still_masked & above_fallback

    def reset(self):
        """Call at the start of each new block."""
        self._low_eff_count = 0
        self._triggered     = False
        self._history       = []


# ─────────────────────────────────────────────────────────────────────────────
# D)  Integrated sampler  (replaces apd_sample_tokens in the generation loop)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def confidence_gated_sample(
    logits:          torch.Tensor,        # (B, active_len, vocab)
    pos:             torch.Tensor,        # (B, active_len)
    mask_index:      torch.Tensor,        # (B, full_seq)
    all_confidence:  torch.Tensor,        # (B, full_seq)  running scores
    step:            int,
    total_steps:     int,
    block_start:     int,
    budget_mgr:      TokenBudgetManager,
    pos_scheduler:   PositionAwareScheduler,
    grind_detector:  GrindingPhaseDetector,
    eos_token:       int   = -1,
    temperature:     float = 0.0,
    exist_eos:       Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Drop-in replacement for apd_sample_tokens that adds:
      - Per-position threshold scaling (§B)
      - Per-token budget enforcement (§A)
      - Grinding phase force-exit (§C)

    Returns same signature as apd_sample_tokens:
      (x0, x0_p, selected_idx)
    """
    B = logits.shape[0]

    # ── predict tokens ────────────────────────────────────────────────────
    if temperature > 0:
        logits_noise = logits.to(torch.float64)
        noise = torch.rand_like(logits_noise)
        gumbel = (-torch.log(noise)) ** temperature
        x0 = torch.argmax(logits_noise.exp() / gumbel, dim=-1)
    else:
        x0 = torch.argmax(logits, dim=-1)

    p = F.softmax(logits.float(), dim=-1)
    x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)    # (B, active_len)
    confidence_with_unmask = x0_p.clone()

    # ── suppress EOS ─────────────────────────────────────────────────────
    if eos_token != -1 and exist_eos is not None:
        last_mask = mask_index.gather(-1, torch.clamp(pos - 1, min=0))
        eos_suppress = (x0 == eos_token) & (~exist_eos.unsqueeze(1).expand_as(x0)) & last_mask
        x0_p = x0_p.masked_fill(eos_suppress, 0.0)

    # ── zero already-unmasked ─────────────────────────────────────────────
    still_masked = mask_index.gather(-1, pos)
    x0_p = x0_p.masked_fill(~still_masked, 0.0)
    all_confidence.scatter_(1, pos, x0_p.to(all_confidence.dtype))

    # ── B: position-aware thresholds ─────────────────────────────────────
    thresholds = pos_scheduler.get_thresholds(step, total_steps, pos, block_start)
    commit_mask = x0_p > thresholds                              # (B, active_len)

    # ── A: budget-exhausted force-commit ──────────────────────────────────
    force_budget = budget_mgr.get_force_commit_mask(step, pos, mask_index)
    commit_mask  = commit_mask | force_budget

    # ── C: grinding phase force-exit ──────────────────────────────────────
    if grind_detector.should_force_exit():
        force_grind = grind_detector.get_force_exit_mask(x0_p, mask_index, pos)
        commit_mask = commit_mask | force_grind
        grind_detector.trigger()

    # ── always commit at least 1 token ───────────────────────────────────
    top1 = torch.topk(x0_p, k=1, dim=-1).indices
    commit_mask.scatter_(-1, top1, True)
    commit_mask = commit_mask & (x0_p > 0.0)

    selected_idx = [commit_mask[i].nonzero().squeeze(1) for i in range(B)]

    # ── update budget manager ─────────────────────────────────────────────
    budget_mgr.record_committed(pos, commit_mask)

    return x0, confidence_with_unmask, selected_idx


# ─────────────────────────────────────────────────────────────────────────────
# E)  Layer 16 attention FLOPs reduction (Exp 2, Finding 1)
# ─────────────────────────────────────────────────────────────────────────────

class Layer16SparseMask:
    """
    Exploits Exp 2 Finding 1: Layer 16 has 70% attention on just 10 tokens.

    Instead of full O(n²) attention, we use a fixed-sparsity mask that
    keeps only the top-k attended positions. This reduces attention FLOPs
    directly (not just memory bandwidth), giving compute savings on H200.

    The mask is built once from the first forward pass's attention scores
    and reused for subsequent steps where attention patterns are stable
    (Exp 1 confirmed early layers drift only 2-3% between steps).

    Integration: inject into data_cache['layer16_sparse_mask'] before
    the model forward pass. The LLaDA attention function checks for this
    and applies it when present.

    Expected FLOP reduction at layer 16:
      Full attention: n² operations
      Sparse (keep top 10 of n): 10n operations
      For n=256: 256² → 10×256 = 93.8% reduction at this layer
    """

    LAYER_IDX    = 16
    KEEP_FRACTION = 0.15   # keep top 15% of KV positions (Exp 2: 20-30% recommended)

    def __init__(self, keep_fraction: float = KEEP_FRACTION):
        self.keep_fraction = keep_fraction
        self._mask: Optional[torch.Tensor] = None   # (B, 1, n, n) bool
        self._built = False

    def build_from_attention(
        self,
        attn_weights: torch.Tensor,   # (B, n_heads, seq_len, seq_len)
        seq_len: int,
    ):
        """
        Build sparse mask from measured attention weights.
        Keep top-k positions per query position based on mean head attention.
        """
        keep_k = max(1, int(seq_len * self.keep_fraction))

        # Average across heads → (B, seq_len, seq_len)
        mean_attn = attn_weights.float().mean(dim=1)

        # Top-k per query row
        _, top_indices = torch.topk(mean_attn, k=keep_k, dim=-1)  # (B, seq_len, k)
        mask = torch.zeros_like(mean_attn, dtype=torch.bool)
        mask.scatter_(-1, top_indices, True)

        self._mask  = mask.unsqueeze(1)   # (B, 1, seq_len, seq_len)
        self._built = True

    @property
    def mask(self) -> Optional[torch.Tensor]:
        return self._mask

    @property
    def built(self) -> bool:
        return self._built

    def reset(self):
        self._mask  = None
        self._built = False
