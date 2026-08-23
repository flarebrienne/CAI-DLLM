# CAI-dLLM: Convergence-Aware Inference for Diffusion Language Models

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **CAI-dLLM**, a training-free inference acceleration method for masked diffusion language models. CAI-dLLM uses step-zero confidence signals to adaptively schedule token commitment, achieving **13–45× speedup** and **92–96% energy reduction** over vanilla inference with competitive or improved accuracy.

> **Paper:** CAI-dLLM: Convergence-Aware Inference for Diffusion Language Models  
> **Authors:** Farhana, Sabiha  
> **Venue:** EMNLP 2025 (ARR submission)

---

## Overview

Diffusion language models (dLLMs) generate text by iteratively unmasking tokens over multiple denoising steps. Each step requires a full forward pass, making inference slow. CAI-dLLM addresses this by:

1. **Adaptive Parallel Decoding (APD)** — cosine warmup threshold schedule that starts conservative and relaxes over time
2. **Per-Block Adaptive Schedules** — descending end thresholds across blocks exploiting empirical difficulty gradient
3. **Confidence-Gated Token Budgets** — per-token step budgets based on step-zero confidence, with grinding-phase detection
4. **Model-Adaptive Gating** — confidence gating ON for LLaDA, OFF for Dream (model-specific calibration)

---

## Results Summary

### LLaDA-8B-Instruct

| Benchmark | Nocache | DualCache | ES-dLLM | **CAI-dLLM** | Speedup |
|-----------|---------|-----------|---------|-------------|---------|
| GSM8K | 76.27% | 77.18% | 77.10% | **77.41%** | **18.2×** |
| HumanEval | 37.20% | 35.98% | 35.98% | 35.98% | **11.2×** |
| BBH | 56.75% | 53.29% | 54.51% | 52.33% | **44.8×** |
| MBPP | 40.00% | 38.20% | 38.00% | 37.00% | **28.5×** |

### Dream-7B-Instruct

| Benchmark | Nocache | DualCache | ES-dLLM | **CAI-dLLM** | Speedup |
|-----------|---------|-----------|---------|-------------|---------|
| GSM8K | 79.68% | 78.47% | 77.48% | 77.26% | **18.7×** |
| HumanEval | 46.95% | 45.12% | 41.46% | **48.17%** | **13.1×** |
| BBH | 61.48% | 58.27% | 57.84% | 56.95% | **23.0×** |
| MBPP | 57.40% | 55.40% | 56.40% | 56.60% | **20.0×** |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/ES-dLLM.git
cd ES-dLLM

# Create conda environment
conda create -n esdllm python=3.10
conda activate esdllm

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
pip install lm-eval[math]
pip install antlr4-python3-runtime==4.11
```

---

## Models

CAI-dLLM supports two diffusion language models:

| Model | HuggingFace ID |
|-------|---------------|
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` |
| Dream-7B-Instruct | `Dream-org/Dream-v0-Instruct-7B` |

Models are downloaded automatically on first run.

---

## Project Structure

```
ES-dLLM/
├── cai_dllm/
│   ├── cai_generate3.py          # Core CAI generation loop
│   ├── cai_enhancements.py       # Enhancement modules (Ideas 1,3,4,5)
│   ├── confidence_gating.py      # Token budget manager, grinding detector
│   ├── eval3.py                  # Evaluation with all enhancements
│   ├── eval_commonsense.py       # WinoGrande, PIQA evaluation
│   ├── eval_math_code.py         # MathQA, MBPP evaluation
│   ├── eval_longbench_longctx.py # LongBench evaluation
│   ├── eval_poem_completion.py   # Chinese poem completion
│   ├── eval_poem_ar.py           # AR model comparison (GPT-4o, Qwen)
│   ├── analyze_length_scaling.py # Throughput vs sequence length
│   ├── measure_step_reduction.py # Denoising step analysis
│   └── profile_overhead.py       # Controller overhead profiling
├── eval.py                       # Main evaluation script (lm-eval)
├── generate.py                   # Base generation (ES-dLLM)
├── models/
│   ├── hook_model.py             # Model transformation hooks
│   ├── llada_func.py             # LLaDA forward pass
│   └── dream_func.py             # Dream forward pass
├── run_gsm8k_energy.sh           # GSM8K energy measurement
├── run_humaneval_energy.sh       # HumanEval energy measurement
├── run_mbpp_energy.sh            # MBPP energy measurement
├── run_bbh_energy.sh             # BBH energy measurement
├── run_multiseed.sh              # Multi-seed evaluation
└── log_results/                  # All experimental results
    ├── energy/                   # GPU energy measurements
    ├── multiseed/                # Multi-seed results
    ├── step_reduction/           # Step count analysis
    └── profiling/                # Overhead profiling
```

---

## Quick Start

### Run CAI-dLLM on GSM8K

```bash
# LLaDA-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model LLaDA-Instruct --task gsm8k \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 4 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
    --use_cai --cai_mode apd_per_block

# Dream-7B-Instruct (with model-adaptive no confidence gating)
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model Dream-Instruct --task gsm8k \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 8 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
    --use_cai --cai_mode apd_per_block \
    --no_confidence_gating
```

### Run on HumanEval

```bash
HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=0 python cai_dllm/eval3.py \
    --model Dream-Instruct --task humaneval \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 8 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25 \
    --use_cai --cai_mode apd_per_block \
    --no_confidence_gating
```

### Run Baselines

```bash
# Nocache
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model LLaDA-Instruct --task gsm8k --esdllm_mode nocache

# DualCache
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model LLaDA-Instruct --task gsm8k --esdllm_mode DualCache

# ES-dLLM
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model LLaDA-Instruct --task gsm8k \
    --esdllm_mode HiddenState --alpha 0.5 \
    --prompt_update_freq 64 --block_update_freq 4 \
    --proportions 1 0.5 0.25 --positions 0 0.125 0.25
```

---

## Supported Benchmarks

| Benchmark | Script | Samples | Metric |
|-----------|--------|---------|--------|
| GSM8K | `eval.py --task gsm8k` | 1,319 | Exact Match |
| HumanEval | `eval3.py --task humaneval` | 164 | pass@1 |
| BBH | `eval.py --task bbh` | 6,511 | Exact Match |
| MBPP | `eval.py --task mbpp` | 257 | pass@1 |
| MathQA | `eval_math_code.py` | 500 | Exact Match |
| PIQA | `eval_commonsense.py` | 1,838 | Accuracy |
| WinoGrande | `eval_commonsense.py` | 500 | Accuracy |
| LongBench | `eval_longbench_longctx.py` | 100/task | F1/ROUGE-L |
| Poem Completion | `eval_poem_completion.py` | 496 | Exact Match |

---

## Energy Measurement

```bash
# Measure GPU energy for GSM8K (all methods)
bash run_gsm8k_energy.sh 0 both

# Measure GPU energy for HumanEval
bash run_humaneval_energy.sh 0 both

# Measure GPU energy for MBPP
bash run_mbpp_energy.sh 0 both
```

Energy is measured by sampling GPU power every 500ms via `nvidia-smi` and integrating over wall-clock time.

---

## Analysis Experiments

### Throughput Scaling with Sequence Length

```bash
CUDA_VISIBLE_DEVICES=0 python cai_dllm/analyze_length_scaling.py \
    --model LLaDA-Instruct --n_samples 20 \
    --lengths 64,128,256,512
```

### Denoising Step Reduction

```bash
CUDA_VISIBLE_DEVICES=0 python cai_dllm/measure_step_reduction.py \
    --model LLaDA-Instruct --tasks gsm8k,humaneval
```

### Controller Overhead Profiling

```bash
CUDA_VISIBLE_DEVICES=0 python cai_dllm/profile_overhead.py \
    --model LLaDA-Instruct --n_iters 100
```

### Chinese Poem Completion

```bash
# dLLM methods
CUDA_VISIBLE_DEVICES=0 python cai_dllm/eval_poem_completion.py \
    --model LLaDA-Instruct \
    --methods nocache,dualcache,esdllm,caidllm \
    --n_samples 496

# Qwen2.5-7B comparison
CUDA_VISIBLE_DEVICES=0 python cai_dllm/eval_poem_ar.py \
    --model qwen --n_samples 496
```

---

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name (`LLaDA-Instruct` or `Dream-Instruct`) | required |
| `--task` | Benchmark task | required |
| `--use_cai` | Enable CAI-dLLM | False |
| `--cai_mode` | CAI mode (`apd_only`, `apd_per_block`, `full`) | `full` |
| `--no_confidence_gating` | Disable confidence gating (recommended for Dream) | False |
| `--block_update_freq` | KV cache update frequency (16 for LLaDA, 8 for Dream) | None |
| `--apd_theta_start` | APD start threshold | 0.90 |
| `--apd_theta_end` | APD end threshold | 0.70 |
| `--apd_warmup_frac` | APD warmup fraction | 0.15 |
| `--use_soft_grind` | Soft grinding phase detector | False |
| `--limit` | Limit evaluation samples | None |

---

## Model-Adaptive Configuration

CAI-dLLM uses different configurations per model based on confidence calibration:

```python
# LLaDA-8B: confidence gating ON
CAIConfig(
    use_apd=True,
    use_per_block=True,
    use_confidence_gating=True,   # ON for LLaDA
    apd_theta_start=0.90,
    apd_theta_end=0.70,
    apd_warmup_frac=0.15,
)

# Dream-7B: confidence gating OFF
CAIConfig(
    use_apd=True,
    use_per_block=True,
    use_confidence_gating=False,  # OFF for Dream
    apd_theta_start=0.90,
    apd_theta_end=0.70,
    apd_warmup_frac=0.15,
)
```

---

## Compute Budget

Total inference compute across all benchmarks, models, and methods:

| Method | GPU-Hours | % of Total |
|--------|-----------|------------|
| No Cache | 77.38 h | 77.9% |
| ES-dLLM | 8.19 h | 8.2% |
| DualCache | 8.07 h | 8.1% |
| **CAI-dLLM** | **3.89 h** | **3.9%** |
| CAI-dLLM (no CG) | 1.85 h | 1.9% |
| **Total** | **99.38 h** | 100% |

CAI-dLLM reduces inference compute by **95.0%** compared to nocache, saving 73.5 GPU-hours across the full evaluation suite. All experiments run on a single NVIDIA H200 GPU (141 GB).

---

## Citation

```bibtex
@inproceedings{farhana2025caidllm,
  title     = {CAI-dLLM: Convergence-Aware Inference for Diffusion Language Models},
  author    = {Farhana and Sabiha},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025}
}
```

---

## Acknowledgements

This work builds on the [ES-dLLM](https://github.com/ML-GSAI/ES-dLLM) codebase. We thank the authors of [LLaDA](https://github.com/ML-GSAI/LLaDA) and [Dream](https://github.com/Dream-org/Dream) for releasing their models.
