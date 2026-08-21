# LLM Post-Training: DPO vs GRPO Alignment Comparison

End-to-end LLM post-training pipeline that compares offline (DPO) and online (GRPO) alignment methods on **Qwen2.5-0.5B**. Includes GPT-2 124M pre-training from scratch as an introductory exercise.

## Project Structure

```
├── gpt2/                          # Part 1: GPT-2 pre-training from scratch
│   ├── train_gpt2.py              # Full GPT-2 124M training (DDP-ready)
│   ├── fineweb.py                 # FineWeb-Edu 10B data download & tokenization
│   ├── hellaswag.py               # HellaSwag evaluation benchmark
│   ├── chatgpt.ipynb              # Character-level bigram → self-attention walkthrough
│   └── results.ipynb              # Training curves & HellaSwag scores
│
├── sft.py                         # Part 2: Supervised Fine-Tuning on Qwen2.5-0.5B
├── reward_model.py                # Reward Model (Bradley-Terry) training
├── dpo.py                         # DPO training with LoRA
├── grpo.py                        # GRPO training with LoRA + RM
├── grpo_smoke.py                  # GRPO 5-step smoke test
│
├── notebooks/                     # Step-by-step notebooks with saved outputs
│   ├── SFT.ipynb                  # SFT training walkthrough
│   ├── Reward_Model.ipynb         # RM training walkthrough
│   ├── Pair_Generation.ipynb      # DPO pair generation via RM scoring
│   ├── DPO.ipynb                  # DPO training walkthrough
│   └── GRPO.ipynb                 # GRPO training walkthrough
│
├── experiments/
│   ├── Comparison.ipynb           # Full DPO vs GRPO comparison
│   └── LLM-critic.ipynb           # LLM-as-judge evaluation of RM quality
│
├── eval_prompts.json              # 200 held-out eval prompts
├── dpo_pairs.json                 # 2644 preference pairs (RM-scored)
├── comparison_results.json        # Aggregated comparison metrics
├── comparison_judge_results.json  # LLM judge pairwise verdicts
└── comparison_rm_scores.json      # RM scores for all model variants
```

## Pipeline

### Part 1: GPT-2 Pre-Training

GPT-2 124M trained from scratch on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (10B tokens subset). Follows [Karpathy's reproducing GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) approach with DDP multi-GPU support. Evaluated on HellaSwag — achieved comparable results to the OpenAI checkpoint in ~4x fewer steps.

### Part 2: Post-Training Pipeline

```
Qwen2.5-0.5B (base)
       │
       ▼
   ┌───────┐     OpenHermes 2.5 (7K samples)
   │  SFT  │◄────────────────────────────────
   └───┬───┘
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
   ┌───────┐     UltraFeedback (20K)    ┌──────────┐
   │  RM   │◄───────────────────────    │  Pair    │  8 responses/prompt
   └───┬───┘                            │  Gen     │  RM best/worst → pair
       │                                └────┬─────┘
       │                                     │
       │                                     ▼
       │                               ┌──────────┐
       │                               │   DPO    │  LoRA, β=0.1
       │                               │  ×3 seeds│
       │                               └──────────┘
       ▼
   ┌──────────┐
   │   GRPO   │  LoRA, 4 generations/prompt
   │  ×1 seed │  RM as reward function
   └──────────┘
```

**SFT** — Full fine-tuning on [OpenHermes 2.5](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) (7000 samples), 1 epoch, effective batch size 4.

**Reward Model** — `AutoModelForSequenceClassification` initialized from the SFT checkpoint, trained on [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) (20K pairs). Eval accuracy: **67.3%**.

**Preference Pair Generation** — For each of 3000 prompts, generate 8 responses from the SFT model, score with the RM, keep best/worst as chosen/rejected pairs. Filtered by reward std > 0.15. Result: **2644 pairs** in 1.03 GPU-hours.

**DPO** — Direct Preference Optimization with LoRA (r=16, α=32) on the SFT checkpoint. Precomputed reference log probs. 3 seeds for variance estimation.

**GRPO** — Group Relative Policy Optimization. LoRA on SFT checkpoint, RM as the reward function. 4 generations per prompt, group-normalized rewards.

## Results

### Reward Model Scores (200 eval prompts)

| Method | RM Reward | Response Length (words) | Trigram Uniqueness |
|--------|-----------|------------------------|--------------------|
| SFT | +0.417 | 52.0 | 0.896 |
| DPO (3 seeds) | +0.540 ± 0.023 | 77.5 ± 0.8 | 0.878 |
| GRPO (1 seed) | +0.482 | 78.4 | 0.884 |

### LLM Judge (Claude, DPO vs GRPO, 200 prompts)

| | Count | % |
|---|---|---|
| DPO wins | 49 | 24.5% |
| GRPO wins | 49 | 24.5% |
| Ties | 64 | 32.0% |
| Inconsistent | 38 | 19.0% |

Position bias: +0.041 (minimal).

### Key Findings

1. Both DPO and GRPO significantly outperform the SFT baseline by RM reward
2. DPO achieves higher RM reward, but the independent LLM judge shows a tie — suggesting DPO optimizes the metric more than actual quality
3. GRPO is more reward-efficient: same quality by LLM judge at lower RM score, less reward over-optimization
4. Both methods increase response length and slightly reduce diversity compared to SFT

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers trl peft datasets wandb anthropic python-dotenv
```

### Training

```bash
# SFT
python sft.py --seed 42

# Reward Model
python reward_model.py --seed 42

# DPO (run pair generation notebook first)
python dpo.py --seed 42

# GRPO
python grpo.py --seed 42

# GRPO smoke test (5 steps, no wandb)
python grpo_smoke.py
```

## Hardware

All experiments run on a single **NVIDIA GeForce RTX 5070 Ti** (16 GB VRAM).

## Acknowledgments

- GPT-2 pre-training code based on [Andrej Karpathy's build-nanogpt](https://github.com/karpathy/build-nanogpt)
- Post-training pipeline built with [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft)
