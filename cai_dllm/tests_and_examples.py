"""
tests_and_examples.py  ─  CAI-dLLM  Unit Tests & Quick-Start
=============================================================
Run with:  python tests_and_examples.py
No GPU required for these tests (all run on CPU with tiny tensors).
"""

import math
import torch
import sys
import os

# Make sure the cai_dllm directory is on the path when running from root
sys.path.insert(0, os.path.dirname(__file__))

from apd_schedule import (
    APDScheduleConfig, APDScheduler,
    apd_sample_tokens, probe_step0_confidence, classify_token_difficulty
)
from per_block_schedule import (
    PerBlockScheduleTable, BlockConvergenceTracker, AdaptiveScheduleUpdater,
    get_block_threshold
)
from layer_adaptive import (
    LayerTierConfig, LayerTierTable, LayerSkipManager,
    AdaptiveKVPruner, estimate_flop_savings,
    EARLY_LAYERS, MIDDLE_LAYERS, DEEP_LAYERS
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def check(name, cond):
    status = PASS if cond else FAIL
    print(f"  {status}  {name}")
    if not cond:
        raise AssertionError(f"Test failed: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# §4.1  APD Schedule Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_apd_schedule():
    print("\n── §4.1  APD Schedule ──────────────────────────────────────────")

    cfg = APDScheduleConfig.quality()
    sched = APDScheduler(cfg)
    T = 64

    # Warmup: first 15% of steps should return θ_start exactly
    warmup_end = int(0.15 * T)   # = 9
    for t in range(warmup_end):
        theta = sched.get_threshold(t, T)
        check(f"Warmup step {t}: θ={theta:.4f} == {cfg.theta_start}", 
              abs(theta - cfg.theta_start) < 1e-6)

    # After warmup: should be strictly between θ_end and θ_start
    for t in range(warmup_end, T):
        theta = sched.get_threshold(t, T)
        check(f"Step {t}: θ in [{cfg.theta_end:.2f}, {cfg.theta_start:.2f}]",
              cfg.theta_end - 1e-6 <= theta <= cfg.theta_start + 1e-6)

    # Monotonically decreasing after warmup
    thetas = [sched.get_threshold(t, T) for t in range(warmup_end, T)]
    diffs = [thetas[i+1] - thetas[i] for i in range(len(thetas)-1)]
    check("Schedule is monotonically non-increasing after warmup",
          all(d <= 1e-9 for d in diffs))

    # Final step ≈ θ_end
    theta_last = sched.get_threshold(T-1, T)
    check(f"Final θ ≈ θ_end ({cfg.theta_end})", abs(theta_last - cfg.theta_end) < 0.01)

    # Linear schedule
    lin_cfg = APDScheduleConfig(theta_start=0.90, theta_end=0.50,
                                warmup_frac=0.0, shape="linear")
    lin_sched = APDScheduler(lin_cfg)
    mid_theta = lin_sched.get_threshold(T//2, T)
    check("Linear midpoint ≈ 0.70", abs(mid_theta - 0.70) < 0.01)

    # Presets
    for name, preset in [("quality", APDScheduleConfig.quality()),
                          ("balanced", APDScheduleConfig.balanced()),
                          ("speed", APDScheduleConfig.speed())]:
        APDScheduler(preset)   # should not raise
        check(f"Preset '{name}' constructs cleanly", True)

    print("  apd_sample_tokens smoke test …")
    B, block_len, vocab = 2, 8, 100
    logits = torch.randn(B, block_len, vocab)
    pos    = torch.arange(block_len).expand(B, -1)
    mask_index = torch.ones(B, block_len, dtype=torch.bool)
    all_conf   = torch.zeros(B, block_len)
    x0, x0_p, sel_idx = apd_sample_tokens(
        logits, pos, mask_index, all_conf, threshold=0.0
    )
    check("apd_sample_tokens returns correct x0 shape", x0.shape == (B, block_len))
    check("At threshold=0 all tokens selected", all(len(s) > 0 for s in sel_idx))

    print("  probe_step0_confidence smoke test …")
    conf0 = probe_step0_confidence(logits, mask_index, pos)
    check("probe shape", conf0.shape == (B, block_len))
    check("conf0 in [0, 1]", conf0.min() >= 0.0 and conf0.max() <= 1.0)

    tiers = classify_token_difficulty(conf0)
    check("tier tensor shape", tiers.shape == (B, block_len))
    check("tier values in {0,1,2}", set(tiers.unique().tolist()).issubset({0, 1, 2}))


# ─────────────────────────────────────────────────────────────────────────────
# §4.2  Per-Block Schedule Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_per_block_schedule():
    print("\n── §4.2  Per-Block Schedule ────────────────────────────────────")

    table = PerBlockScheduleTable.default(4)
    check("Table has 4 configs", len(table) == 4)

    # Each block should be more aggressive (lower θ_end) than the last
    ends = [table.configs[b].theta_end for b in range(4)]
    check("θ_end decreases across blocks", all(ends[i] > ends[i+1] for i in range(3)))

    # Block 0 should be the most conservative
    sched0 = table.get_scheduler(0)
    sched3 = table.get_scheduler(3)
    T = 64
    # After warmup, block 3 should be more aggressive
    theta0_mid = sched0.get_threshold(40, T)
    theta3_mid = sched3.get_threshold(40, T)
    check("Block 3 more aggressive than Block 0 at mid-decay",
          theta3_mid < theta0_mid)

    # quality_only table: all identical
    qt = PerBlockScheduleTable.quality_only(4)
    ends_qt = [qt.configs[b].theta_end for b in range(4)]
    check("quality_only: all θ_end equal", len(set(ends_qt)) == 1)

    # from_theta_ends
    custom = PerBlockScheduleTable.from_theta_ends([0.65, 0.55, 0.45, 0.35])
    check("Custom table θ_end[0]=0.65", abs(custom.configs[0].theta_end - 0.65) < 1e-6)

    # BlockConvergenceTracker
    tracker = BlockConvergenceTracker(num_blocks=4, early_exit_frac=0.95, min_steps=3)
    for s in range(2):
        tracker.record(0, s, 1)
    check("Not converged after 2 steps (< min_steps=3)",
          not tracker.is_converged(0, 64, 0))
    tracker.record(0, 2, 1)
    check("Converged after 3 steps with 0 remaining",
          tracker.is_converged(0, 64, 0))
    check("Not converged with many remaining",
          not tracker.is_converged(0, 64, 10))

    # AdaptiveScheduleUpdater
    updater = AdaptiveScheduleUpdater(table, adapt_rate=0.05)
    orig_end = updater._live_configs[1].theta_end
    # Block 0 was slower than expected → block 1 should be more conservative
    updater.update_after_block(0, actual_steps=35, next_block_idx=1)  # expected ~27
    new_end = updater._live_configs[1].theta_end
    check("Slower block → next block more conservative (θ_end rises)",
          new_end >= orig_end)

    # get_block_threshold helper
    theta = get_block_threshold(0, 10, 64, table)
    check("get_block_threshold returns float in [0.3, 1.0]",
          isinstance(theta, float) and 0.3 <= theta <= 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# §4.3  Layer-Adaptive Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_layer_adaptive():
    print("\n── §4.3  Layer-Adaptive Compute ────────────────────────────────")

    table = LayerTierTable.for_llada_8b()
    check("Table has 32 layers", len(table) == 32)

    # Tier assignments
    for l in EARLY_LAYERS:
        check(f"Layer {l} is EARLY tier", table[l].tier == 0)
    for l in MIDDLE_LAYERS:
        check(f"Layer {l} is MIDDLE tier", table[l].tier == 1)
    for l in DEEP_LAYERS[:3]:   # spot-check first 3
        check(f"Layer {l} is DEEP tier", table[l].tier == 2)

    # Step strides
    check("Early layers have stride=2", table[0].step_stride == 2)
    check("Deep layers have stride=1",  table[20].step_stride == 1)

    # Layer 16 sparse
    check("Layer 16 kv_retention=0.25", abs(table[16].kv_retention - 0.25) < 1e-6)
    check("Layer 16 is sparse_layer",   table[16].sparse_layer)
    check("Layer 20 is not sparse",     not table[20].sparse_layer)

    # from_drift_data
    drifts = [0.02 + 0.01 * l for l in range(32)]   # increasing drift
    table2 = LayerTierTable.from_drift_data(drifts, n_layers=32)
    check("from_drift_data early layers EARLY", table2[0].tier == 0)

    # FLOP proportion is less than 1.0
    fp = table.flop_proportion()
    check(f"FLOP proportion < 1.0 (got {fp:.3f})", fp < 1.0)
    check(f"FLOP proportion > 0.3 (got {fp:.3f})", fp > 0.3)

    # LayerSkipManager
    skip_mgr = LayerSkipManager(table)
    check("EARLY layer, step=0: do NOT skip",  not skip_mgr.should_skip(0, 0))
    check("EARLY layer, step=1: skip",         skip_mgr.should_skip(0, 1))
    check("EARLY layer, step=2: do NOT skip",  not skip_mgr.should_skip(0, 2))
    check("DEEP  layer, step=1: do NOT skip",  not skip_mgr.should_skip(20, 1))

    # Cache store / retrieve
    data_cache = {}
    dummy_hidden = torch.randn(2, 10, 64)
    dummy_pos    = torch.arange(10).expand(2, -1)
    skip_mgr.store_output(data_cache, dummy_hidden, dummy_pos)
    h_cached, p_cached = skip_mgr.get_cached_output(data_cache, dummy_hidden, dummy_pos)
    check("Cached hidden state matches stored",
          torch.allclose(h_cached, dummy_hidden))

    # AdaptiveKVPruner
    pruner = AdaptiveKVPruner(table)
    B, n_kv_h, seq_len, hd = 1, 4, 32, 16
    n_q_h = 8   # GQA
    key   = torch.randn(B, n_kv_h, seq_len, hd)
    value = torch.randn(B, n_kv_h, seq_len, hd)
    query = torch.randn(B, n_q_h, 8, hd)      # 8 active tokens
    dc = {
        "inference_start_index": 10,
        "inference_end_index":   18,
        "record_sparse_kv":      True,
    }
    pk, pv = pruner.prune(dc, layer_idx=16, query=query, key=key, value=value)
    outside_total = 10 + (seq_len - 18)        # 10 + 14 = 24
    expected_kept = max(1, int(outside_total * 0.25))   # 6
    block_size    = 18 - 10                    # 8
    expected_total = expected_kept + block_size  # 14
    check(f"Pruned KV seq_len = {pk.shape[2]} (expected {expected_total})",
          pk.shape[2] == expected_total)
    check("Pruned key and value same shape", pk.shape == pv.shape)

    # estimate_flop_savings
    savings = estimate_flop_savings(table, n_steps=64, block_len=64,
                                    d_model=4096, seq_len=512)
    check("reduction_factor > 1.0", savings["reduction_factor"] > 1.0)
    print(f"  Estimated FLOP reduction: {savings['reduction_factor']:.2f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Quick-start: show how to construct a CAIConfig and run
# ─────────────────────────────────────────────────────────────────────────────

def quickstart_demo():
    print("\n── Quick-start demo (no model required) ───────────────────────")
    try:
        from cai_generate import CAIConfig
    except ImportError as e:
        print(f"  Skipping (cai_generate imports ES-dLLM which needs the repo): {e}")
        return

    # Quality mode  (100% accuracy, 5.07×)
    cfg_quality = CAIConfig(
        use_apd=True, apd_theta_start=0.90, apd_theta_end=0.70,
        apd_warmup_frac=0.15,
        use_per_block=True,
        use_layer_adaptive=True,
    )

    # Speed mode  (60% accuracy, 7.92×)
    cfg_speed = CAIConfig(
        use_apd=True, apd_theta_start=0.90, apd_theta_end=0.70,
        apd_warmup_frac=0.00,
        use_per_block=False,
        use_layer_adaptive=True,
    )

    for name, cfg in [("quality", cfg_quality), ("speed", cfg_speed)]:
        table = cfg.build_schedule_table(4)
        tier  = cfg.build_tier_table()
        check(f"{name} config builds schedule table", table is not None)
        check(f"{name} config builds tier table",     tier is not None)

    print("\n  Example usage:")
    print("    from cai_generate import cai_batch_generate, CAIConfig")
    print("    cfg = CAIConfig()   # defaults: quality mode, all features on")
    print("    output, results = cai_batch_generate(model, input_ids, mask,")
    print("                                         generation_kwargs={...},")
    print("                                         cai_config=cfg)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" CAI-dLLM Unit Tests")
    print("=" * 60)

    test_apd_schedule()
    test_per_block_schedule()
    test_layer_adaptive()
    quickstart_demo()

    print("\n" + "=" * 60)
    print(" All tests passed ✓")
    print("=" * 60)
