import argparse
import json
import os

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="DPO training for alignment comparison")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft_checkpoint", default="checkpoints/sft/sft-seed42/final")
    p.add_argument("--pairs_path", default="dpo_pairs.json")
    p.add_argument("--output_dir", default="checkpoints/dpo")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval_size", type=int, default=200)
    p.add_argument("--num_gen_prompts", type=int, default=20)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    return p.parse_args()


def load_pairs(pairs_path, tokenizer, eval_size, seed):
    with open(pairs_path) as f:
        data = json.load(f)

    gpu_hours_data = data["metadata"]["gpu_hours"]
    print(f"Loaded {len(data['pairs'])} pairs (data generation cost: {gpu_hours_data:.2f} GPU-hours)")

    rows = []
    for pair in data["pairs"]:
        prompt_msgs = [{"role": "user", "content": pair["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
        rows.append({
            "prompt": prompt_text,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        })

    ds = Dataset.from_list(rows).shuffle(seed=seed)
    val = ds.select(range(eval_size))
    train = ds.select(range(eval_size, len(ds)))
    return train, val, gpu_hours_data


def setup_model_and_tokenizer(sft_checkpoint):
    model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint, dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def generate_eval(sft_checkpoint, adapter_dir, tokenizer, num_prompts, output_dir):
    eval_prompts_path = "eval_prompts.json"
    if not os.path.exists(eval_prompts_path):
        print(f"No {eval_prompts_path} found, skipping generation")
        return

    with open(eval_prompts_path) as f:
        eval_prompts = json.load(f)

    base_model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint, dtype=torch.bfloat16,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = merged_model.merge_and_unload()
    merged_model = merged_model.to("cuda")
    merged_model.eval()

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id, im_end_id]
    results = []

    for i, messages in enumerate(eval_prompts[:num_prompts]):
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(merged_model.device)

        with torch.no_grad():
            outputs = merged_model.generate(
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

    merged_dir = os.path.join(output_dir, "merged")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    out_path = os.path.join(output_dir, "generations.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} generations to {out_path}")
    print(f"Merged model saved to {merged_dir}")


def main():
    args = parse_args()
    run_name = f"dpo-seed{args.seed}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    wandb.init(project="mini-llm", name=run_name)

    model, tokenizer = setup_model_and_tokenizer(args.sft_checkpoint)
    train_dataset, val_dataset, gpu_hours_data = load_pairs(
        args.pairs_path, tokenizer, args.eval_size, args.seed,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Beta: {args.beta}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    config = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=True,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        beta=args.beta,
        precompute_ref_log_probs=True,
        warmup_steps=50,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=100,
        report_to=["wandb"],
        run_name=run_name,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        args=config,
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adapters saved to {final_dir}")

    metrics = trainer.evaluate()
    metrics["gpu_hours_data_generation"] = gpu_hours_data
    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Eval metrics: {json.dumps(metrics, indent=2)}")

    del model, trainer
    torch.cuda.empty_cache()

    generate_eval(args.sft_checkpoint, final_dir, tokenizer, args.num_gen_prompts, output_dir)

    wandb.finish()
    print(f"Done. Adapters: {final_dir}, Merged: {output_dir}/merged")
    print(f"Total DPO budget = {gpu_hours_data:.2f}h (data) + training time")


if __name__ == "__main__":
    main()
