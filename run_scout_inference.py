#!/usr/bin/env python3
"""
run_scout_inference.py
======================
Scout-KV Phase 4: End-to-End Inference Demo

Demonstrates the Scout-KV system on a REAL query:
1. Loads Llama-3 (Reader) and Scout model
2. Indexes document into blocks
3. Uses Scout to predict top-5 relevant blocks
4. Generates answer using only those blocks
5. Compares with baseline (first-5 blocks)

This proves the Scout system works end-to-end.
"""

import json
import sys
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# =============================================================================
# Scout Model Definition (Must Match Phase 3)
# =============================================================================

class ScoutModel(nn.Module):
    """
    Lightweight Scout model for predicting block relevance.
    Architecture matches train_scout.py exactly.
    """
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
# Model Loading
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


def load_scout_model():
    """Load the trained Scout model."""
    print("\nLoading Scout model...")
    
    scout = ScoutModel(input_dim=8192, hidden_dim=1024, dropout=0.3)
    scout.load_state_dict(torch.load("scout_v1.pth", weights_only=True))
    scout = scout.cuda()
    scout.eval()
    
    print(f"✓ Scout loaded (8.4M params)")
    return scout


# =============================================================================
# Block Processing
# =============================================================================

def chunk_document(text, tokenizer, block_size=128):
    """
    Chunk document into blocks of block_size tokens.
    Returns: List of (block_text, token_ids)
    """
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
    
    return blocks


def get_embeddings(model, tokenizer, texts, batch_size=16):
    """
    Get embeddings for a list of texts using the LLM's embedding layer.
    Returns: List of [4096] tensors
    """
    embeddings = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        tokens = tokens.to(model.device)
        
        with torch.no_grad():
            # Use embedding layer directly
            emb = model.model.embed_tokens(tokens)  # [1, seq_len, 4096]
            # Mean pool
            pooled = emb[0].mean(dim=0)  # [4096]
            embeddings.append(pooled.cpu().float())
        
        del tokens, emb
    
    return embeddings


def format_prompt(question, context):
    """Format prompt for Llama-3-Instruct."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based on the provided context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}

Please answer the question based on the context above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def generate_answer(model, tokenizer, prompt, max_new_tokens=100):
    """Generate answer using Llama-3."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    del inputs, outputs
    torch.cuda.empty_cache()
    
    return answer.strip()


# =============================================================================
# Answer Functions
# =============================================================================

def answer_with_scout(model, tokenizer, scout, question, document, block_size=128, top_k=5):
    """
    Answer using Scout-based block selection.
    
    Steps:
    A. Chunk document into blocks and compute embeddings
    B. Compute question embedding, run Scout, get top-k blocks
    C. Construct paged context from top-k blocks
    D. Generate answer
    
    Returns: (answer, selected_block_ids, scout_time)
    """
    # Step A: Chunk and embed document blocks
    blocks = chunk_document(document, tokenizer, block_size)
    block_texts = [b["text"] for b in blocks]
    block_embeddings = get_embeddings(model, tokenizer, block_texts)
    
    # Step B: Compute question embedding and run Scout
    start_time = time.time()
    
    # Embed the question (IMPORTANT: just the question, not mean of blocks)
    question_embedding = get_embeddings(model, tokenizer, [question])[0]
    
    # Run Scout on all blocks
    scores = []
    with torch.no_grad():
        for block_emb in block_embeddings:
            # Concatenate [query, block]
            combined = torch.cat([question_embedding, block_emb], dim=0).unsqueeze(0).cuda()
            logit = scout(combined)
            score = torch.sigmoid(logit).item()
            scores.append(score)
    
    scout_time = time.time() - start_time
    
    # Get top-k block IDs
    indexed_scores = [(i, s) for i, s in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_block_ids = [idx for idx, _ in indexed_scores[:top_k]]
    top_block_ids.sort()  # Keep in order for coherent reading
    
    # Step C: Construct paged context
    paged_context = "\n\n".join(blocks[bid]["text"] for bid in top_block_ids)
    
    # Step D: Generate answer
    prompt = format_prompt(question, paged_context)
    answer = generate_answer(model, tokenizer, prompt)
    
    return answer, top_block_ids, scout_time


def answer_linear(model, tokenizer, question, document, block_size=128, num_blocks=5):
    """
    Baseline: Answer using the FIRST num_blocks blocks.
    This simulates naive truncation.
    """
    blocks = chunk_document(document, tokenizer, block_size)
    
    # Take first num_blocks
    first_blocks = blocks[:num_blocks]
    linear_context = "\n\n".join(b["text"] for b in first_blocks)
    
    prompt = format_prompt(question, linear_context)
    answer = generate_answer(model, tokenizer, prompt)
    
    return answer


# =============================================================================
# Main Demo
# =============================================================================

def main():
    print("=" * 70)
    print("Scout-KV Phase 4: End-to-End Inference Demo")
    print("=" * 70)
    
    # Load models
    model, tokenizer = load_llama_model()
    scout = load_scout_model()
    
    # Load a NEW sample (not in training set)
    # Training used samples 0-49, so we use sample 60
    from datasets import load_dataset
    print("\nLoading test sample from NarrativeQA...")
    
    dataset = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    
    # Skip to sample 60
    TARGET_INDEX = 60
    sample = None
    for i, s in enumerate(dataset):
        if i == TARGET_INDEX:
            sample = s
            break
    
    if sample is None:
        print("Error: Could not load sample")
        return
    
    # Extract fields
    question = sample["question"]["text"]
    document = sample["document"]["text"]
    answers = [a["text"] for a in sample["answers"]]
    ground_truth = answers[0] if answers else "N/A"
    
    # Truncate document to manageable size
    MAX_TOKENS = 2048
    doc_tokens = tokenizer.encode(document, add_special_tokens=False)
    if len(doc_tokens) > MAX_TOKENS:
        doc_tokens = doc_tokens[:MAX_TOKENS]
        document = tokenizer.decode(doc_tokens, skip_special_tokens=True)
    
    print(f"\n{'='*70}")
    print("TEST SAMPLE")
    print(f"{'='*70}")
    print(f"Document Length: {len(doc_tokens)} tokens")
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    
    # Run Scout-based answer
    print(f"\n{'='*70}")
    print("SCOUT-BASED ANSWER")
    print(f"{'='*70}")
    
    scout_answer, selected_blocks, scout_time = answer_with_scout(
        model, tokenizer, scout, question, document
    )
    
    print(f"Selected Blocks: {selected_blocks}")
    print(f"Scout Time: {scout_time*1000:.2f} ms")
    print(f"Answer: {scout_answer}")
    
    # Run baseline answer
    print(f"\n{'='*70}")
    print("BASELINE ANSWER (First 5 Blocks)")
    print(f"{'='*70}")
    
    baseline_answer = answer_linear(model, tokenizer, question, document)
    print(f"Answer: {baseline_answer}")
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Question:      {question}")
    print(f"Ground Truth:  {ground_truth}")
    print(f"Scout Answer:  {scout_answer[:100]}...")
    print(f"Base Answer:   {baseline_answer[:100]}...")
    print(f"Scout Blocks:  {selected_blocks}")
    print(f"Scout Latency: {scout_time*1000:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
