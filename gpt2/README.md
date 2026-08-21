# GPT-2 124M Pre-Training from Scratch

Reproducing GPT-2 (124M parameters) trained from scratch on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 10B tokens subset, following [Karpathy's approach](https://www.youtube.com/watch?v=l8pRSuU81PU).

## Architecture

Standard GPT-2 config: 12 layers, 12 heads, 768 embedding dim, 1024 context length, ~124M parameters.

Key implementation details:
- Weight tying between token embeddings and LM head
- GELU activation (tanh approximation)
- Flash Attention via `scaled_dot_product_attention`
- `1/√(2N)` residual scaling on projection layers
- Cosine LR schedule with linear warmup

## Training

- **Data**: FineWeb-Edu 10B, tokenized with `tiktoken` GPT-2 encoding, sharded into 100M-token `.npy` files
- **Batch size**: 524,288 tokens (B=16, T=1024, gradient accumulation)
- **Optimizer**: AdamW (fused), lr=6e-4, weight decay=0.1, β=(0.9, 0.95)
- **Steps**: 5000 (with warmup ~187 steps)
- **Precision**: bfloat16 mixed precision
- **DDP**: Multi-GPU ready via `DistributedDataParallel`

## Results

Achieved results comparable to the OpenAI GPT-2 124M checkpoint in ~4x fewer training steps:

- **Validation loss**: converges near the OpenAI checkpoint baseline (3.29)
- **HellaSwag accuracy**: approaches the OpenAI reference (~29.5% acc_norm)

See [results.ipynb](results.ipynb) for training curves.

## Files

| File | Description |
|------|-------------|
| `train_gpt2.py` | Full training script with DDP, eval, HellaSwag, generation |
| `fineweb.py` | Downloads FineWeb-Edu and tokenizes into shards |
| `hellaswag.py` | HellaSwag benchmark download and evaluation |
| `chatgpt.ipynb` | Step-by-step walkthrough: char-level bigram → self-attention |
| `results.ipynb` | Training loss / HellaSwag plots |

## Usage

```bash
# 1. Download and tokenize data
python fineweb.py

# 2. Train (single GPU)
python train_gpt2.py

# 2. Train (multi-GPU with DDP)
torchrun --nproc_per_node=N train_gpt2.py
```
