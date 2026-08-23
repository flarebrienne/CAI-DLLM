"""
per_block_schedule.py  ─  CAI-dLLM  §4.2  Per-Block Adaptive Schedules
=======================================================================
From Experiment 5 (Block-Level Denoising Behavior):

  Block 1  →  27 steps average  (bottleneck, needs protection)
  Block 2  →  25 steps average  (-7 %  faster than Block 1)
  Block 3  →  20 steps average  (-27 %, high confidence bursts)
  Block 4  →  17 steps average  (-37 %, near-EOS predictable content)

Key insight: Every block runs the *same* APD cosine schedule today.
Using block-position-aware schedules reduces total steps from ~90 → ~75
(~17 % improvement) with no quality loss because:
  • Block 1 provides left-context that later blocks exploit.
  • Later blocks contain more predictable content (conclusions, EOS).

This module provides:
  1. PerBlockScheduleTable  – maps block_idx → APDScheduleConfig
  2. BlockConvergenceTracker – tracks live per-block statistics
  3. AdaptiveScheduleUpdater – optionally updates schedules online
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from apd_schedule import APDScheduleConfig, APDScheduler


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Per-block schedule table  (static presets from Experiment 5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerBlockScheduleTable:
    """
    Holds one APDScheduleConfig per generation block.

    Default values from §4.2 paper table:
      Block 0: Cosine 0.90 → 0.70  (conservative  – 27 steps avg)
      Block 1: Cosine 0.90 → 0.60  (moderate      – 25 steps avg)
      Block 2: Cosine 0.90 → 0.50  (aggressive    – 20 steps avg)
      Block 3: Cosine 0.90 → 0.40  (very aggressive – 17 steps avg)

    All blocks share θ_start = 0.90 and warmup_frac = 0.15 from the
    single-block ablation (Experiment 4).
    """

    configs: List[APDScheduleConfig] = field(default_factory=list)

    # ── class-level factory ───────────────────────────────────────────────
    @classmethod
    def default(cls, num_blocks: int = 4) -> "PerBlockScheduleTable":
        """
        Create the default per-block schedule from §4.2.
        Works for any num_blocks; blocks beyond index 3 reuse the
        most-aggressive config (θ_end = 0.40).
        """
        # θ_end values from the paper (indexed 0 … 3)
        theta_ends = [0.70, 0.60, 0.50, 0.40]

        configs = []
        for b in range(num_blocks):
            t_end = theta_ends[min(b, len(theta_ends) - 1)]
            configs.append(APDScheduleConfig(
                theta_start = 0.90,
                theta_end   = t_end,
                warmup_frac = 0.15,
                shape       = "cosine",
            ))
        return cls(configs=configs)

    @classmethod
    def quality_only(cls, num_blocks: int = 4) -> "PerBlockScheduleTable":
        """All blocks use the conservative Quality config (100 % acc baseline)."""
        return cls(configs=[APDScheduleConfig.quality()] * num_blocks)

    @classmethod
    def from_theta_ends(
        cls,
        theta_ends: Sequence[float],
        theta_start: float = 0.90,
        warmup_frac: float = 0.15,
    ) -> "PerBlockScheduleTable":
        """Build a table from a list of θ_end values, one per block."""
        configs = [
            APDScheduleConfig(
                theta_start = theta_start,
                theta_end   = t_end,
                warmup_frac = warmup_frac,
                shape       = "cosine",
            )
            for t_end in theta_ends
        ]
        return cls(configs=configs)

    # ── accessors ─────────────────────────────────────────────────────────
    def get_scheduler(self, block_idx: int) -> APDScheduler:
        """Return the APDScheduler for the given block index."""
        if block_idx >= len(self.configs):
            # fall back to the most aggressive config
            cfg = self.configs[-1]
        else:
            cfg = self.configs[block_idx]
        return APDScheduler(cfg)

    def get_threshold(self, block_idx: int, step: int, total_steps: int) -> float:
        """Convenience: get θ(t) for a given block and step."""
        return self.get_scheduler(block_idx).get_threshold(step, total_steps)

    def __len__(self) -> int:
        return len(self.configs)

    def __repr__(self) -> str:
        lines = ["PerBlockScheduleTable:"]
        for i, cfg in enumerate(self.configs):
            lines.append(
                f"  Block {i}: θ_start={cfg.theta_start:.2f}, "
                f"θ_end={cfg.theta_end:.2f}, w={cfg.warmup_frac:.2f}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Block convergence tracker  (live stats from Experiment 5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BlockStats:
    """Running statistics for a single block."""
    block_idx:       int
    steps_taken:     int   = 0
    tokens_per_step: List[float] = field(default_factory=list)
    max_burst:       int   = 0       # largest single-step unmask count
    total_unmasked:  int   = 0

    @property
    def avg_tokens_per_step(self) -> float:
        if not self.tokens_per_step:
            return 0.0
        return sum(self.tokens_per_step) / len(self.tokens_per_step)

    def record_step(self, n_unmasked: int):
        self.steps_taken    += 1
        self.total_unmasked += n_unmasked
        self.tokens_per_step.append(float(n_unmasked))
        self.max_burst = max(self.max_burst, n_unmasked)

    def __repr__(self) -> str:
        return (f"BlockStats(idx={self.block_idx}, steps={self.steps_taken}, "
                f"avg_tok/step={self.avg_tokens_per_step:.1f}, "
                f"max_burst={self.max_burst})")


class BlockConvergenceTracker:
    """
    Tracks per-block denoising statistics in real time.

    Usage inside the generation loop:
        tracker = BlockConvergenceTracker(num_blocks=4)
        for block_idx in range(num_blocks):
            for step in range(steps_per_block):
                n_unmasked = sum(len(si) for si in selected_idx)
                tracker.record(block_idx, step, n_unmasked)
            if tracker.is_converged(block_idx, steps_per_block):
                break  # early exit for this block
    """

    def __init__(
        self,
        num_blocks: int,
        early_exit_frac: float = 0.95,   # fraction of block_len considered converged
        min_steps: int = 3,              # always run at least this many steps
    ):
        self.num_blocks      = num_blocks
        self.early_exit_frac = early_exit_frac
        self.min_steps       = min_steps
        self._stats: List[BlockStats] = [BlockStats(block_idx=b) for b in range(num_blocks)]

    # ------------------------------------------------------------------
    def record(self, block_idx: int, step: int, n_unmasked: int):
        """Call once per denoising step with the count of committed tokens."""
        self._stats[block_idx].record_step(n_unmasked)

    # ------------------------------------------------------------------
    def is_converged(
        self,
        block_idx: int,
        block_len: int,
        remaining_masked: int,
    ) -> bool:
        """
        Returns True when it is safe to stop denoising this block early.

        Criteria (from §4.2 + Experiment 5 analysis):
          • At least min_steps steps have been taken, AND
          • remaining_masked / block_len < (1 - early_exit_frac)
            (i.e. > early_exit_frac of the block is already committed)
        """
        stats = self._stats[block_idx]
        if stats.steps_taken < self.min_steps:
            return False
        threshold = (1.0 - self.early_exit_frac) * block_len
        return remaining_masked <= threshold

    # ------------------------------------------------------------------
    def summary(self) -> Dict[int, BlockStats]:
        return {s.block_idx: s for s in self._stats}

    def print_summary(self):
        print("Block Convergence Summary")
        print("─" * 50)
        for s in self._stats:
            print(s)
        # Replicate the Exp-5 finding: show speedup vs Block 0
        if self._stats[0].steps_taken > 0:
            base = self._stats[0].steps_taken
            for s in self._stats:
                pct = 100.0 * (base - s.steps_taken) / base if base > 0 else 0.0
                print(f"  Block {s.block_idx}: {s.steps_taken} steps  "
                      f"({'-' if pct >= 0 else '+'}{abs(pct):.0f}% vs Block 0)")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Adaptive schedule updater  (online, optional)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveScheduleUpdater:
    """
    Optionally refines per-block θ_end values after seeing live convergence.

    After each block completes, it compares actual steps_taken against the
    expected steps (from historical data) and nudges θ_end for future blocks
    in the same generation if they follow a similar pattern.

    This implements the lightweight "continuous adaptation" aspect from §4.2:
      • If a block converged much faster than expected → next block can be
        more aggressive (lower θ_end).
      • If a block was slower than expected → pull back (raise θ_end).

    Note: this only adjusts remaining *future* blocks in the same request;
    it does NOT update the global PerBlockScheduleTable (that would require
    a bandit/meta-learning framework, out of scope for §4.2).
    """

    # Expected steps per block from Experiment 5 (used as reference)
    EXPECTED_STEPS = {0: 27.3, 1: 25.3, 2: 20.0, 3: 17.3}

    def __init__(
        self,
        table: PerBlockScheduleTable,
        adapt_rate: float = 0.05,    # how much θ_end shifts per mismatch step
        theta_end_min: float = 0.30,
        theta_end_max: float = 0.80,
    ):
        # Deep-copy configs so we don't mutate the global table
        import copy
        self._live_configs = [copy.deepcopy(cfg) for cfg in table.configs]
        self.adapt_rate    = adapt_rate
        self.theta_end_min = theta_end_min
        self.theta_end_max = theta_end_max

    # ------------------------------------------------------------------
    def update_after_block(
        self,
        finished_block_idx: int,
        actual_steps: int,
        next_block_idx: int,
    ):
        """
        Adjust θ_end for next_block based on how fast the finished block was.

        Parameters
        ----------
        finished_block_idx : index of the block that just finished
        actual_steps       : how many denoising steps it actually took
        next_block_idx     : index of the block to potentially adjust
        """
        if next_block_idx >= len(self._live_configs):
            return

        expected = self.EXPECTED_STEPS.get(finished_block_idx, actual_steps)
        delta = actual_steps - expected                    # > 0 → slower than expected

        # Slower than expected → be more conservative (raise θ_end = less aggressive)
        # Faster than expected → be more aggressive   (lower θ_end)
        # delta > 0 → adjustment > 0 → θ_end rises (more conservative)
        sign = 1.0 if delta > 0 else -1.0
        adjustment = sign * self.adapt_rate

        cfg = self._live_configs[next_block_idx]
        new_end = cfg.theta_end + adjustment
        cfg.theta_end = max(self.theta_end_min, min(self.theta_end_max, new_end))

    # ------------------------------------------------------------------
    def get_scheduler(self, block_idx: int) -> APDScheduler:
        """Return an APDScheduler using the (possibly adapted) config."""
        if block_idx >= len(self._live_configs):
            cfg = self._live_configs[-1]
        else:
            cfg = self._live_configs[block_idx]
        return APDScheduler(cfg)

    def get_threshold(self, block_idx: int, step: int, total_steps: int) -> float:
        return self.get_scheduler(block_idx).get_threshold(step, total_steps)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Integration helper  (used inside batch_generate)
# ─────────────────────────────────────────────────────────────────────────────

def get_block_threshold(
    block_idx:   int,
    step:        int,
    total_steps: int,
    schedule_table: PerBlockScheduleTable | AdaptiveScheduleUpdater,
) -> float:
    """
    Single call-site for fetching θ(t) inside the generation loop.

    Works with both PerBlockScheduleTable (static) and
    AdaptiveScheduleUpdater (live-adjusted).
    """
    return schedule_table.get_threshold(block_idx, step, total_steps)
