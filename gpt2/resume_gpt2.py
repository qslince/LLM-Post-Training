"""Resume GPT-2 pretraining from a checkpoint."""

import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from hellaswag import iterate_examples, render_example


# ── Model (copied from train_gpt2.py) ──────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate,
                                 fused=use_fused, betas=(0.9, 0.95), eps=1e-8)


# ── Data ────────────────────────────────────────────────────────────────────

def load_tokens(filename):
    return torch.tensor(np.load(filename), dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        data_root = 'edu_fineweb10B'
        shards = sorted(s for s in os.listdir(data_root) if split in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no {split} shards found"
        self.reset()

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.cur_pos: self.cur_pos + B * T + 1]
        self.cur_pos += B * T * self.num_processes
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        if self.cur_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.cur_pos = self.B * self.T * self.process_rank
            self.tokens = load_tokens(self.shards[self.current_shard])
        return x, y

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.cur_pos = self.B * self.T * self.process_rank


# ── Eval helpers ────────────────────────────────────────────────────────────

def get_most_likely_row(tokens, mask, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    shift_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1), reduction='none',
    ).view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    avg_loss = (shift_losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()


def run_hellaswag(raw_model, device, device_type):
    num_correct = num_total = 0
    for i, example in enumerate(iterate_examples('val')):
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = raw_model(tokens)
            pred = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct += int(pred == label)
    acc = num_correct / num_total
    print(f"HellaSwag accuracy: {num_correct}/{num_total}={acc:.4f}")
    return acc


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="log/model_04999.pt")
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--total_batch_size", type=int, default=2**19)
    p.add_argument("--max_lr", type=float, default=6e-4)
    p.add_argument("--warmup_frac", type=float, default=0.0375)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    torch.set_float32_matmul_precision('high')

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    start_step = ckpt['step'] + 1
    print(f"Resuming from step {start_step} (val_loss={ckpt['val_loss']:.4f})")

    config = ckpt['config']
    model = GPT(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    raw_model = model
    model = torch.compile(model)

    # Training setup
    B, T = args.B, args.T
    assert args.total_batch_size % (B * T) == 0
    grad_accum_steps = args.total_batch_size // (B * T)
    print(f"total batch size: {args.total_batch_size}")
    print(f"gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split='val')

    min_lr = args.max_lr * 0.1
    warmup_steps = int(args.max_steps * args.warmup_frac)

    def get_lr(it):
        if it < warmup_steps:
            return args.max_lr * (it + 1) / warmup_steps
        if it > args.max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (args.max_steps - warmup_steps)
        coeff = 0.5 * (1. + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.max_lr - min_lr)

    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=args.max_lr, device=device,
    )
    enc = tiktoken.get_encoding('gpt2')

    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log.txt')

    print(f"Training steps {start_step} → {args.max_steps}")

    for step in range(start_step, args.max_steps):
        t0 = time.time()
        last_step = (step == args.max_steps - 1)

        # Validation
        if step % 100 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    val_loss_accum += loss.detach() / val_loss_steps
            print(f"step {step} | val_loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # HellaSwag
        if (step % 1000 == 0 or last_step) and step > 0:
            acc = run_hellaswag(raw_model, device, device_type)
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc:.4f}\n")

        # Sample generation
        if step % 500 == 0 or last_step:
            model.eval()
            xgen = enc.encode("Hello, I'm a language model,")
            xgen = torch.tensor(xgen, dtype=torch.long, device=device).unsqueeze(0).repeat(4, 1)
            rng = torch.Generator(device=device)
            rng.manual_seed(42 + step)
            while xgen.size(1) < 64:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(xgen)
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=rng)
                    xgen = torch.cat((xgen, torch.gather(topk_indices, -1, ix)), dim=1)
            for i in range(4):
                print(f"sample {i}: {enc.decode(xgen[i].tolist())}")

        # Save checkpoint
        if (step % 1000 == 0 or last_step) and step > 0:
            torch.save({
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': val_loss_accum.item(),
            }, os.path.join(log_dir, f"model_{step:05d}.pt"))

        # Training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if 'cuda' in device:
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_per_sec = B * T * grad_accum_steps / dt
        print(f"step {step} | loss: {loss_accum.item():.4f} | lr {lr:.4e} | norm: {norm:.4f} | dt:{dt:.2f}s | tok/sec:{tokens_per_sec:.0f}")
        with open(log_file, 'a') as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

    print("Done.")


if __name__ == "__main__":
    main()
