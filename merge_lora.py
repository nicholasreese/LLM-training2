# save as merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, sys, os

base_model = sys.argv[1]            # e.g. mistralai/Mistral-7B-Instruct-v0.3
adapter_dir = sys.argv[2]           # e.g. outputs/mistral7b-md-qlora
out_dir    = sys.argv[3]            # e.g. outputs/mistral7b-merged

tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu")
merged = PeftModel.from_pretrained(base, adapter_dir)
merged = merged.merge_and_unload()  # apply LoRA into base

os.makedirs(out_dir, exist_ok=True)
merged.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("Merged model saved to:", out_dir)
