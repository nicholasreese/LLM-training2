#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--context", default="")
    ap.add_argument("--context_file", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    context = args.context
    if args.context_file:
        with open(args.context_file, 'r') as f:
            context = f.read()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16
    )

    # Load LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    prompt = f"Answer the question using ONLY the given context.\n\nContext:\n{context}\n\nQuestion: {args.question}"
    messages = [{"role":"user","content":prompt}]
    if hasattr(tok, "apply_chat_template"):
        text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text_in = f"<s>[INST] {prompt} [/INST]"

    gen = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")
    out = gen(text_in, max_new_tokens=args.max_new_tokens, do_sample=False,
              eos_token_id=tok.eos_token_id)[0]["generated_text"]
    print("\n=== Answer ===\n")
    print(out)

if __name__ == "__main__":
    main()