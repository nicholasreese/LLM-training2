#!/usr/bin/env python3
import argparse, os, json, re
from pathlib import Path

def read_markdown_files(root):
    for p in Path(root).rglob("*.md"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                yield str(p), text
        except Exception:
            pass

def chunk_text(text, chunk_chars=500, overlap=100):
    paras = re.split(r"\n\s*\n", text)
    buf, chunks = "", []
    for para in paras:
        if len(buf) + len(para) + 2 <= chunk_chars:
            buf += (("\n\n" if buf else "") + para)
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf.strip():
        chunks.append(buf)
    return chunks

def extract_key_facts(text):
    """Extract key facts from text using improved patterns"""
    facts = []
    
    # Look for titles/headings (improved patterns)
    titles = re.findall(r'^#+\s*(.+)$', text, re.MULTILINE)
    for title in titles:
        clean_title = title.strip()
        if len(clean_title) > 3 and not clean_title.lower().startswith(('table of', 'list of', 'status')):
            facts.append(f"Title: {clean_title}")
    
    # Look for dates (improved patterns)
    date_patterns = [
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b'
    ]
    for pattern in date_patterns:
        dates = re.findall(pattern, text, re.IGNORECASE)
        for date in dates[:2]:  # Limit to 2 dates
            facts.append(f"Date: {date}")
    
    # Look for years (more specific)
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    unique_years = list(set(years))[:3]  # Get unique years, max 3
    for year in unique_years:
        facts.append(f"Year: {year}")
    
    # Look for definitions or key terms (improved)
    definition_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+([^.]{10,100})',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+means\s+([^.]{10,100})',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+refers\s+to\s+([^.]{10,100})'
    ]
    for pattern in definition_patterns:
        definitions = re.findall(pattern, text)
        for term, definition in definitions[:2]:  # Limit to 2 per pattern
            clean_term = term.strip()
            clean_def = definition.strip()
            if len(clean_term) > 2 and len(clean_def) > 10:
                facts.append(f"{clean_term}: {clean_def}")
    
    # Look for lists (improved)
    list_items = re.findall(r'^\s*[-*•]\s+(.+)$', text, re.MULTILINE)
    for item in list_items[:3]:  # Limit to first 3 items
        clean_item = item.strip()
        if len(clean_item) > 10 and not clean_item.lower().startswith(('page', 'section')):
            facts.append(f"List item: {clean_item}")
    
    # Look for specific legal/regulatory terms
    legal_terms = re.findall(r'\b(?:Act|Section|Chapter|Part|Article|Clause|Subsection)\s+\d+[A-Za-z]?\b', text)
    for term in legal_terms[:2]:  # Limit to 2 legal terms
        facts.append(f"Legal reference: {term}")
    
    # Look for numbers/statistics
    numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:percent|%|dollars?|\$|years?|months?|days?)\b', text, re.IGNORECASE)
    for num in numbers[:2]:  # Limit to 2 numbers
        facts.append(f"Statistic: {num}")
    
    return facts[:8]  # Return max 8 facts

def generate_qa_pairs(facts, context):
    """Generate diverse Q&A pairs from facts"""
    qa_pairs = []
    
    for fact in facts:
        if "Title:" in fact:
            title = fact.replace("Title:", "").strip()
            qa_pairs.append({
                "question": f"What is the title of this document?",
                "answer": f"The title is '{title}'."
            })
        
        elif "Date:" in fact:
            date = fact.replace("Date:", "").strip()
            qa_pairs.append({
                "question": f"What date is mentioned in this document?",
                "answer": f"The document mentions the date {date}."
            })
        
        elif "Year:" in fact:
            year = fact.replace("Year:", "").strip()
            qa_pairs.append({
                "question": f"What year is referenced in this document?",
                "answer": f"The document references the year {year}."
            })
        
        elif "Legal reference:" in fact:
            ref = fact.replace("Legal reference:", "").strip()
            qa_pairs.append({
                "question": f"What legal reference is mentioned?",
                "answer": f"The document mentions {ref}."
            })
        
        elif "Statistic:" in fact:
            stat = fact.replace("Statistic:", "").strip()
            qa_pairs.append({
                "question": f"What statistic or number is mentioned?",
                "answer": f"The document mentions {stat}."
            })
        
        elif ":" in fact and len(fact.split(":")) == 2:
            term, definition = fact.split(":", 1)
            term = term.strip()
            definition = definition.strip()
            
            # Generate different question types based on the term
            if any(word in term.lower() for word in ['act', 'law', 'regulation']):
                qa_pairs.append({
                    "question": f"What is the {term}?",
                    "answer": f"The {term} is {definition}"
                })
            elif any(word in term.lower() for word in ['section', 'chapter', 'part']):
                qa_pairs.append({
                    "question": f"What does {term} cover?",
                    "answer": f"{term} covers {definition}"
                })
            else:
                qa_pairs.append({
                    "question": f"What is {term}?",
                    "answer": f"{term} is {definition}"
                })
        
        elif "List item:" in fact:
            item = fact.replace("List item:", "").strip()
            qa_pairs.append({
                "question": f"What is one item mentioned in this document?",
                "answer": f"One item mentioned is: {item}"
            })
    
    return qa_pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--chunk_chars", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--pairs_per_chunk", type=int, default=2)
    ap.add_argument("--max_files", type=int, default=None, help="Limit number of files to process (None = all files)")
    args = ap.parse_args()

    print(f"Starting template-based data preparation...")
    if args.max_files:
        print(f"Max files to process: {args.max_files}")
    else:
        print("Processing all files")
    
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    processed_files = 0
    total_chunks = 0
    total_qa_pairs = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for path, text in read_markdown_files(args.md_dir):
            if args.max_files and processed_files >= args.max_files:
                print(f"Reached max files limit ({args.max_files}), stopping...")
                break
                
            processed_files += 1
            print(f"\nProcessing file {processed_files}: {path}")
            print(f"File size: {len(text)} characters")
            
            chunks = chunk_text(text, args.chunk_chars, args.overlap)
            print(f"Created {len(chunks)} chunks")
            total_chunks += len(chunks)
            
            for i, ch in enumerate(chunks[:10]):  # Process first 10 chunks
                print(f"\n  Processing chunk {i+1}/{min(10, len(chunks))} ({len(ch)} chars)")
                
                # Extract facts from this chunk
                facts = extract_key_facts(ch)
                print(f"    Extracted {len(facts)} facts: {facts[:2]}...")
                
                # Generate Q&A pairs from facts
                qa_pairs = generate_qa_pairs(facts, ch)
                print(f"    Generated {len(qa_pairs)} QA pairs")
                
                for qa in qa_pairs[:args.pairs_per_chunk]:  # Limit pairs per chunk
                    total_qa_pairs += 1
                    record = {
                        "messages": [
                            {"role": "user",
                             "content": f"Question: {qa['question']}\n\nUse only this context:\n{ch[:200]}..."},
                            {"role": "assistant", "content": qa["answer"]}
                        ]
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"    Wrote QA pair {total_qa_pairs}: {qa['question'][:50]}...")

    print(f"\n=== Summary ===")
    print(f"Processed files: {processed_files}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total QA pairs generated: {total_qa_pairs}")
    print(f"Output saved to: {args.out_jsonl}")

if __name__ == "__main__":
    main()
