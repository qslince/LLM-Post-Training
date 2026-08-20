import argparse
import json
import os

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Reward model training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft_checkpoint", default="checkpoints/sft/sft-seed42/final")
    p.add_argument("--dataset_size", type=int, default=20000)
    p.add_argument("--eval_size", type=int, default=1000)
    p.add_argument("--output_dir", default="checkpoints/rm")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_length", type=int, default=1024)
    return p.parse_args()


def setup_model_and_tokenizer(sft_checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(
        sft_checkpoint, num_labels=1, dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_datasets(tokenizer, dataset_size, eval_size, seed):
    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", split="train_prefs",
    ).select(range(dataset_size)).shuffle(seed=seed)

    val = dataset.select(range(eval_size))
    train = dataset.select(range(eval_size, dataset_size))

    def format_pair(example):
        chosen = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
        rejected = tokenizer.apply_chat_template(example["rejected"], tokenize=False)
        return {"prompt": example["prompt"], "chosen": chosen, "rejected": rejected}

    train = train.map(format_pair)
    val = val.map(format_pair)
    return train, val


def train_rm(model, tokenizer, train_dataset, val_dataset, args, output_dir, run_name):
    config = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        center_rewards_coefficient=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=["wandb"],
        run_name=run_name,
        warmup_steps=50,
    )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=config,
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics = trainer.evaluate()
    print(f"Eval accuracy: {metrics.get('eval_accuracy', 'N/A')}")
    print(f"Eval loss: {metrics.get('eval_loss', 'N/A')}")

    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model saved to {final_dir}")

    return trainer


def main():
    args = parse_args()
    run_name = f"rm-seed{args.seed}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    wandb.init(project="mini-llm", name=run_name)

    model, tokenizer = setup_model_and_tokenizer(args.sft_checkpoint)
    train_dataset, val_dataset = prepare_datasets(
        tokenizer, args.dataset_size, args.eval_size, args.seed,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    train_rm(model, tokenizer, train_dataset, val_dataset, args, output_dir, run_name)

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
