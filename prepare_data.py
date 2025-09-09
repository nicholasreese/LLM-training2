#!/usr/bin/env python3
import argparse, os, json, re, glob
from pathlib import Path

def read_markdown_files(root):
    for p in Path(root).rglob("*.md"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                yield str(p), text
        except Exception:
            pass

def chunk_text(text, chunk_chars=1500, overlap=200):
    # Respect paragraph boundaries when possible
    paras = re.split(r"\n\s*\n", text)
    buf, chunks = "", []
    for para in paras:
        if len(buf) + len(para) + 2 <= chunk_chars:
            buf += (("\n\n" if buf else "") + para)
        else:
            if buf:
                chunks.append(buf)
            # start new buffer possibly with overlap from previous
            if chunks and overlap > 0:
                tail = chunks[-1][-overlap:]
                buf = tail + "\n\n" + para
            else:
                buf = para
            while len(buf) > chunk_chars:
                chunks.append(buf[:chunk_chars])
                buf = (buf[chunk_chars - overlap:] if overlap > 0 else buf[chunk_chars:])
    if buf.strip():
        chunks.append(buf)
    return chunks

PROMPT = """You are creating high-quality training data.
From the CONTEXT below, write {n} diverse question/answer pairs (JSON lines).
- Focus on concrete facts contained in the context.
- DO NOT invent information not in the context.
- Keep answers concise but complete.
Output ONLY JSON Lines with fields: "question", "answer".

CONTEXT:
{context}
"""

def build_pipeline(model_name, load_8bit=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    import torch
    
    print(f"Loading model: {model_name}")
    print("Setting up quantization...")
    quant = None
    if load_8bit:
        quant = BitsAndBytesConfig(load_in_8bit=True)

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    print("Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None
    )
    
    print("Creating pipeline...")
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto"
    )

def extract_json_lines(s):
    """Improved JSON extraction with better error handling"""
    results = []
    print(f"    Raw generated text: {s[:300]}...")
    
    # Try to find JSON objects in the text
    lines = s.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Clean up common prefixes
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        
        # Look for JSON-like patterns
        if "question" in line.lower() and "answer" in line.lower():
            try:
                # Try to extract JSON from the line
                obj = json.loads(line)
                if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                    results.append(obj)
                    print(f"    Found valid JSON: {obj}")
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    # Add missing quotes around keys if needed
                    fixed_line = re.sub(r'(\w+):', r'"\1":', line)
                    obj = json.loads(fixed_line)
                    if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                        results.append(obj)
                        print(f"    Found fixed JSON: {obj}")
                except:
                    continue
    
    # If no JSON found, try to create a simple Q&A from the text
    if not results and "question" in s.lower():
        print("    No valid JSON found, attempting to extract Q&A manually...")
        # Simple extraction - look for question and answer patterns
        question_match = re.search(r'question["\']?\s*:\s*["\']?([^"\']+)["\']?', s, re.IGNORECASE)
        answer_match = re.search(r'answer["\']?\s*:\s*["\']?([^"\']+)["\']?', s, re.IGNORECASE)
        
        if question_match and answer_match:
            results.append({
                "question": question_match.group(1).strip(),
                "answer": answer_match.group(1).strip()
            })
            print(f"    Extracted Q&A manually: {results[-1]}")
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model", default="gpt2")  # Use smaller, faster model
    ap.add_argument("--chunk_chars", type=int, default=500)  # Smaller chunks
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--pairs_per_chunk", type=int, default=1)  # Fewer pairs per chunk
    ap.add_argument("--max_new_tokens", type=int, default=150)  # Shorter responses
    ap.add_argument("--max_files", type=int, default=1)  # Limit files for testing
    args = ap.parse_args()

    print(f"Starting data preparation...")
    print(f"Model: {args.model}")
    print(f"Max files to process: {args.max_files}")
    
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    gen = build_pipeline(args.model, load_8bit=False)  # No quantization for small model

    processed_files = 0
    total_chunks = 0
    total_qa_pairs = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for path, text in read_markdown_files(args.md_dir):
            if processed_files >= args.max_files:
                print(f"Reached max files limit ({args.max_files}), stopping...")
                break
                
            processed_files += 1
            print(f"\nProcessing file {processed_files}: {path}")
            print(f"File size: {len(text)} characters")
            
            chunks = chunk_text(text, args.chunk_chars, args.overlap)
            print(f"Created {len(chunks)} chunks")
            total_chunks += len(chunks)
            
            for i, ch in enumerate(chunks[:3]):  # Limit to first 3 chunks for testing
                print(f"\n  Processing chunk {i+1}/{min(3, len(chunks))} ({len(ch)} chars)")
                
                # Truncate context if too long to prevent sequence overflow
                max_context_length = 300  # Conservative limit
                if len(ch) > max_context_length:
                    ch = ch[:max_context_length] + "..."
                    print(f"    Truncated context to {len(ch)} chars")
                
                prompt = PROMPT.format(n=args.pairs_per_chunk, context=ch)
                
                # Simple prompt for GPT-2 (no chat template needed)
                inp = prompt
                
                try:
                    print(f"    Generating response...")
                    out_text = gen(
                        inp,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        eos_token_id=gen.tokenizer.eos_token_id,
                        pad_token_id=gen.tokenizer.eos_token_id
                    )[0]["generated_text"]
                    
                    # Remove the input prompt from the generated text
                    if inp in out_text:
                        out_text = out_text[len(inp):].strip()
                    
                    print(f"    Generated {len(out_text)} characters")

                    qa_items = extract_json_lines(out_text)
                    print(f"    Extracted {len(qa_items)} QA pairs")
                    
                    for qa in qa_items:
                        total_qa_pairs += 1
                        record = {
                            "messages": [
                                {"role": "user",
                                 "content": f"Question: {qa['question']}\n\nUse only this context:\n{ch}"},
                                {"role": "assistant", "content": qa["answer"]}
                            ]
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        print(f"    Wrote QA pair {total_qa_pairs}")
                        
                except Exception as e:
                    print(f"    Error processing chunk: {e}")
                    continue

    print(f"\n=== Summary ===")
    print(f"Processed files: {processed_files}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total QA pairs generated: {total_qa_pairs}")
    print(f"Output saved to: {args.out_jsonl}")

if __name__ == "__main__":
    main()
