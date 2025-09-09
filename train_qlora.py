#!/usr/bin/env python3
import argparse, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--output_dir", default="outputs/mistral-7b-instruct-qlora")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 4-bit quantization
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token  # keep it simple; SFTTrainer can also set pad/eos via SFTConfig

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,   # use the transformers kwarg (not `dtype`)
    )

    # QLoRA config
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Load dataset (ChatML-style JSONL with "messages")
    ds = load_dataset("json", data_files=args.train_file, split="train")

    def format_example(example):
        # Use the tokenizer's chat template so special tokens & roles are right
        return tok.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    # TRL SFTConfig (replaces many TrainingArguments for SFTTrainer)
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        fp16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        report_to=None,                      # "none" -> None
        max_length=args.max_seq_len,
        packing=True,                        # token packing now lives in SFTConfig
        dataset_text_field="text",           # will be created from formatting_func
        gradient_checkpointing=True,         # memory saver
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        peft_config=lora_cfg,
        formatting_func=format_example,
        processing_class=tok,
    )

    # Enable gradient checkpointing to save VRAM
    trainer.model.config.use_cache = False

    trainer.train()

    # Save LoRA adapter + tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
