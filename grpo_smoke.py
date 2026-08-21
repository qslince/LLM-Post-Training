"""GRPO smoke test — 5 training steps to verify the pipeline works."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

sft_path = "./checkpoints/sft/sft-seed42/final"
rm_path = "./checkpoints/rm/final"

print("Loading policy model...")
model = AutoModelForCausalLM.from_pretrained(sft_path, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(sft_path)
tokenizer.padding_side = "left"
tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
model.resize_token_embeddings(len(tokenizer))

print("Loading reward model...")
rm_model = AutoModelForSequenceClassification.from_pretrained(
    rm_path, num_labels=1, dtype=torch.bfloat16
)
rm_tokenizer = AutoTokenizer.from_pretrained(rm_path)

print("Preparing dataset...")
with open("dpo_pairs.json") as f:
    data = json.load(f)

rows = [{"prompt": [{"role": "user", "content": p["prompt"]}]} for p in data["pairs"][:20]]
dataset = Dataset.from_list(rows)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

cfg = GRPOConfig(
    output_dir="./checkpoints/grpo/smoke",
    num_generations=4,
    generation_batch_size=4,
    max_completion_length=64,
    temperature=0.9,
    scale_rewards="group",
    learning_rate=5e-6,
    num_train_epochs=1,
    max_steps=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    seed=42,
)

print("Creating trainer...")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=rm_model,
    args=cfg,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_processing_classes=[rm_tokenizer],
    peft_config=lora_config,
)

print("Training (5 steps)...")
trainer.train()
print("Smoke test PASSED")
