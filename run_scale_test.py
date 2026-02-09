#!/usr/bin/env python3
"""
run_scale_test.py
=================
Scout-KV Phase 5: Scale & ONNX Optimization

Tests Scout-KV on a FULL BOOK (~100k tokens):
1. Exports Scout model to ONNX for fast inference
2. Downloads "The Adventures of Sherlock Holmes" from Project Gutenberg
3. Indexes all ~800+ blocks using Llama-3 embedding layer
4. Runs ONNX inference to find relevant blocks
5. Generates answer using only Top-5 blocks

"Needle in a Haystack" Query:
    Question: "What was the speckled band?"
    Answer: A swamp adder (snake)
"""

import sys
import os
import time
import gc
import re
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import requests
from tqdm import tqdm

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# =============================================================================
# Scout Model Definition (Must Match Phase 3)
# =============================================================================

class ScoutModel(nn.Module):
    """Scout MLP for block relevance prediction."""
    def __init__(self, input_dim=8192, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


# =============================================================================
# Step 1: ONNX Export
# =============================================================================

def export_to_onnx():
    """Export Scout model to ONNX format."""
    print("\n" + "="*70)
    print("STEP 1: ONNX Export")
    print("="*70)
    
    # Load PyTorch model
    model = ScoutModel(input_dim=8192, hidden_dim=1024, dropout=0.3)
    model.load_state_dict(torch.load("scout_v1.pth", weights_only=True))
    model.eval()
    
    # Create dummy input (batch of 1 combined embedding)
    dummy_input = torch.randn(1, 8192)
    
    # Export to ONNX
    onnx_path = "scout_v1.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    print(f"✓ ONNX model exported: {onnx_path}")
    print(f"  File size: {Path(onnx_path).stat().st_size / 1024:.1f} KB")
    
    # Verify ONNX
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")
    
    return onnx_path


# =============================================================================
# Step 2: Download & Process Book
# =============================================================================

def download_sherlock_holmes():
    """Download The Adventures of Sherlock Holmes from Project Gutenberg."""
    print("\n" + "="*70)
    print("STEP 2: Download Book")
    print("="*70)
    
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    cache_file = Path("sherlock_holmes.txt")
    
    if cache_file.exists():
        print(f"✓ Using cached book: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"Downloading from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
        
        # Cache for future runs
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✓ Downloaded and cached: {cache_file}")
    
    # Clean Gutenberg headers/footers
    # Find start: "*** START OF THE PROJECT GUTENBERG EBOOK ***"
    # Find end: "*** END OF THE PROJECT GUTENBERG EBOOK ***"
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1:
        # Find the end of the start line
        start_idx = text.find("\n", start_idx) + 1
    else:
        start_idx = 0
        
    if end_idx != -1:
        text = text[start_idx:end_idx]
    else:
        text = text[start_idx:]
    
    # Basic cleanup
    text = text.strip()
    
    print(f"  Book length: {len(text):,} characters")
    return text


def chunk_book(text, tokenizer, block_size=128):
    """Chunk the entire book into blocks."""
    print("\nChunking book into blocks...")
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    num_blocks = (len(tokens) + block_size - 1) // block_size
    
    blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, len(tokens))
        block_tokens = tokens[start:end]
        block_text = tokenizer.decode(block_tokens, skip_special_tokens=True)
        blocks.append({
            "block_id": i,
            "text": block_text,
            "tokens": block_tokens
        })
    
    print(f"✓ Created {len(blocks)} blocks ({len(tokens):,} total tokens)")
    return blocks, tokens


def compute_block_embeddings(model, tokenizer, blocks, batch_size=8):
    """Compute embeddings for all blocks using Llama's embedding layer."""
    print("\nComputing block embeddings...")
    
    start_time = time.time()
    embeddings = []
    
    for i in tqdm(range(0, len(blocks), batch_size), desc="Indexing"):
        batch = blocks[i:i+batch_size]
        batch_embeddings = []
        
        for block in batch:
            tokens = torch.tensor([block["tokens"]]).to(model.device)
            
            with torch.no_grad():
                emb = model.model.embed_tokens(tokens)  # [1, seq, 4096]
                pooled = emb[0].mean(dim=0).cpu().float()  # [4096]
                batch_embeddings.append(pooled)
            
            del tokens, emb
        
        embeddings.extend(batch_embeddings)
        
        # Periodic cleanup
        if (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
    
    indexing_time = time.time() - start_time
    print(f"✓ Indexed {len(embeddings)} blocks in {indexing_time:.2f}s")
    
    return embeddings, indexing_time


# =============================================================================
# Step 3: ONNX Inference
# =============================================================================

def run_onnx_inference(onnx_path, query_embedding, block_embeddings, top_k=5):
    """Run ONNX inference to score all blocks."""
    import onnxruntime as ort
    
    # Create ONNX session
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Prepare inputs: [num_blocks, 8192]
    num_blocks = len(block_embeddings)
    
    # Create combined embeddings for all blocks
    query_np = query_embedding.numpy()  # [4096]
    
    combined = []
    for block_emb in block_embeddings:
        concat = np.concatenate([query_np, block_emb.numpy()], axis=0)  # [8192]
        combined.append(concat)
    
    combined = np.stack(combined, axis=0).astype(np.float32)  # [num_blocks, 8192]
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {"input": combined})
    onnx_time = time.time() - start_time
    
    scores = outputs[0]  # [num_blocks]
    
    # Get top-k
    indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_block_ids = [idx for idx, _ in indexed_scores[:top_k]]
    top_block_ids.sort()  # Keep reading order
    
    return top_block_ids, onnx_time, scores


# =============================================================================
# Step 4: Answer Generation
# =============================================================================

def load_llama_model():
    """Load Llama-3-8B-Instruct with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"\nLoading Reader: {model_name}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.eval()
    
    print(f"✓ Reader loaded ({torch.cuda.memory_allocated() / 1024**3:.2f} GB)")
    return model, tokenizer


def format_prompt(question, context):
    """Format prompt for Llama-3-Instruct."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based on the provided context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}

Please answer the question based on the context above. Be concise.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def generate_answer(model, tokenizer, prompt, max_new_tokens=100):
    """Generate answer using Llama-3."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    del inputs, outputs
    torch.cuda.empty_cache()
    
    return answer.strip()


def get_question_embedding(model, tokenizer, question):
    """Get embedding for the question text."""
    tokens = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt")
    tokens = tokens.to(model.device)
    
    with torch.no_grad():
        emb = model.model.embed_tokens(tokens)
        pooled = emb[0].mean(dim=0).cpu().float()
    
    del tokens, emb
    return pooled


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("Scout-KV Phase 5: Scale & ONNX Optimization")
    print("="*70)
    
    # Step 1: Export to ONNX
    onnx_path = export_to_onnx()
    
    # Step 2: Load book
    book_text = download_sherlock_holmes()
    
    # Load Llama for embeddings
    model, tokenizer = load_llama_model()
    
    # Chunk book
    blocks, all_tokens = chunk_book(book_text, tokenizer, block_size=128)
    
    # Compute block embeddings
    block_embeddings, indexing_time = compute_block_embeddings(model, tokenizer, blocks)
    
    # Step 3: Run "Needle in a Haystack" query
    print("\n" + "="*70)
    print("STEP 3: Needle Query")
    print("="*70)
    
    QUESTION = "What was the speckled band?"
    print(f"Question: {QUESTION}")
    print(f"Expected Answer: A swamp adder (snake)")
    
    # Get question embedding
    query_embedding = get_question_embedding(model, tokenizer, QUESTION)
    
    # Run ONNX inference
    top_block_ids, onnx_time, all_scores = run_onnx_inference(
        onnx_path, query_embedding, block_embeddings, top_k=5
    )
    
    print(f"\n✓ ONNX Inference Time: {onnx_time*1000:.2f} ms")
    print(f"  Top-5 Block IDs: {top_block_ids}")
    
    # Show what's in the selected blocks (preview)
    print("\nSelected Block Previews:")
    for bid in top_block_ids:
        preview = blocks[bid]["text"][:80].replace("\n", " ")
        print(f"  Block {bid}: \"{preview}...\"")
    
    # Step 4: Generate answers
    print("\n" + "="*70)
    print("STEP 4: Answer Generation")
    print("="*70)
    
    # Scout answer (top-5 blocks)
    scout_context = "\n\n".join(blocks[bid]["text"] for bid in top_block_ids)
    scout_prompt = format_prompt(QUESTION, scout_context)
    scout_answer = generate_answer(model, tokenizer, scout_prompt)
    
    # Baseline answer (first-5 blocks)
    baseline_context = "\n\n".join(blocks[i]["text"] for i in range(5))
    baseline_prompt = format_prompt(QUESTION, baseline_context)
    baseline_answer = generate_answer(model, tokenizer, baseline_prompt)
    
    # Step 5: Benchmarking Summary
    print("\n" + "="*70)
    print("BENCHMARKING SUMMARY")
    print("="*70)
    
    print(f"Total Blocks:        {len(blocks)}")
    print(f"Total Tokens:        {len(all_tokens):,}")
    print(f"Indexing Time:       {indexing_time:.2f}s")
    print(f"ONNX Inference Time: {onnx_time*1000:.2f} ms")
    
    print(f"\n{'='*70}")
    print("ANSWER COMPARISON")
    print(f"{'='*70}")
    print(f"Question:       {QUESTION}")
    print(f"Expected:       A swamp adder (snake)")
    print(f"\nScout Answer:   {scout_answer}")
    print(f"Scout Blocks:   {top_block_ids}")
    print(f"\nBaseline Answer: {baseline_answer}")
    print(f"Baseline Blocks: [0, 1, 2, 3, 4]")
    
    # Final verdict
    print(f"\n{'='*70}")
    if "adder" in scout_answer.lower() or "snake" in scout_answer.lower():
        print("✓ SCOUT FOUND THE NEEDLE!")
    else:
        print("⚠ Scout may need tuning (check context)")
    
    if "adder" in baseline_answer.lower() or "snake" in baseline_answer.lower():
        print("✓ Baseline also found it (unlikely)")
    else:
        print("✗ Baseline failed (as expected - needle is buried)")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
