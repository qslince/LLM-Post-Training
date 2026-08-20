import argparse
import json
import os

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser(description="SFT training for alignment comparison")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dataset_size", type=int, default=7000)
    p.add_argument("--eval_size", type=int, default=200)
    p.add_argument("--output_dir", default="checkpoints/sft")
    p.add_argument("--num_gen_prompts", type=int, default=20,
                   help="Number of prompts for quick generation check (use 200 for full run)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=2048)
    return p.parse_args()


def setup_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def prepare_datasets(dataset_size, eval_size, seed):
    dataset = load_dataset(
        "HuggingFaceTB/smoltalk2", "SFT",
        split="OpenHermes_2.5_no_think",
    ).select(range(dataset_size)).shuffle(seed=seed)

    val_dataset = dataset.select(range(eval_size))
    train_dataset = dataset.select(range(eval_size, dataset_size))
    return train_dataset, val_dataset


def save_eval_prompts(val_dataset, path="eval_prompts.json"):
    if os.path.exists(path):
        print(f"Eval prompts already exist at {path}, skipping creation")
        return
    eval_prompts = [
        [{"role": "user", "content": ex["messages"][0]["content"]}]
        for ex in val_dataset
    ]
    with open(path, "w") as f:
        json.dump(eval_prompts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(eval_prompts)} eval prompts to {path}")


def train(model, tokenizer, train_dataset, val_dataset, args, output_dir, run_name):
    config = SFTConfig(
        output_dir=output_dir,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        num_train_epochs=args.epochs,
        max_steps=-1,
        warmup_steps=50,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        push_to_hub=False,
        report_to=["wandb"],
        run_name=run_name,
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=config,
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")

    return trainer


def generate_eval(model, tokenizer, num_prompts, output_dir):
    with open("eval_prompts.json") as f:
        eval_prompts = json.load(f)

    model.eval()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id, im_end_id]
    results = []

    for i, messages in enumerate(eval_prompts[:num_prompts]):
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        results.append({"prompt": messages[0]["content"], "response": response})
        print(f"[{i+1}/{num_prompts}] {messages[0]['content'][:80]}")
        print(f"  -> {response[:200]}\n")

    out_path = os.path.join(output_dir, "generations.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} generations to {out_path}")


def main():
    args = parse_args()
    run_name = f"sft-seed{args.seed}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    wandb.init(project="mini-llm", name=run_name)

    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    train_dataset, val_dataset = prepare_datasets(args.dataset_size, args.eval_size, args.seed)
    save_eval_prompts(val_dataset)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    train(model, tokenizer, train_dataset, val_dataset, args, output_dir, run_name)
    generate_eval(model, tokenizer, args.num_gen_prompts, output_dir)

    wandb.finish()
    print(f"Done. Checkpoint: {output_dir}/final")


if __name__ == "__main__":
    main()
