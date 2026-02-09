#!/usr/bin/env python3
"""
build_index.py
==============
Scout-KV Phase 2: Training Data Generation

This script generates the training dataset for the Scout model.
It aligns the oracle attention scores (from Phase 1) with the
embedding vectors of the corresponding text blocks.

Process:
1. Load oracle_data.json
2. Load NarrativeQA dataset (streaming)
3. Load Llama-3-8B Embedding Layer (4-bit or FP16)
4. For each sample:
   - Re-tokenize context into 128-token blocks
   - Encode tokens -> Embeddings [128, 4096]
   - Mean Pool -> Block Vector [4096]
   - Get corresponding Oracle Score
   - Save {vector, score}
5. Save training_dataset.pt

Hardware:
- 16GB VRAM (loads full model structure but only uses embedding)
"""

import json
import sys
import gc
import os
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Env vars
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def check_cuda():
    if not torch.cuda.is_available():
        print("Error: CUDA required.")
        sys.exit(1)
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

def load_embedding_model():
    """
    Load Llama-3-8B-Instruct.
    We need the full model for bitsandbytes to load correctly,
    but we will essentially just use the embedding layer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"\nLoading model: {model_name}")
    
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
    
    return model, tokenizer

def get_block_embeddings(model, tokenizer, text, block_size=128):
    """
    Tokenize text and generate embeddings for each block.
    Returns list of tensors [4096].
    """
    # 1. Tokenize (no special tokens to keep strict blocking)
    # We use format_prompt manually or just raw text?
    # Phase 1 used `format_prompt`. We must match Phase 1 input exactly!
    # Validation Phase 1: format_prompt(question, context).
    # BUT Phase 1 extracted attention from the full prompt.
    # The Oracle Scores correspond to blocks of the FULL PROMPT (System + Context + Question).
    # Wait. Phase 1 `extract_attention_scores` tokenizes `prompt`.
    # `prompt = format_prompt(question, context)`.
    # `inputs = tokenizer(prompt, ...)`
    # `num_blocks = (seq_length + block_size - 1) // block_size`
    # So we MUST reproduce the exact same token sequence.
    
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt") # Shape [1, seq]
    
    # Check if length matches what we expect?
    # We don't have exact length in oracle_data.json, only block count.
    
    seq_length = tokens.shape[1]
    num_blocks = (seq_length + block_size - 1) // block_size
    
    block_vectors = []
    
    # We can process in batches or loop?
    # Given we only need embedding layer, we can process all.
    # Embedding layer is cheap.
    
    # Move tokens to GPU
    tokens = tokens.to(model.device)
    
    with torch.no_grad():
        # Get embeddings: [1, seq_len, 4096]
        # model.model.embed_tokens is the embedding layer
        embeddings = model.model.embed_tokens(tokens)
    
    # Now pool into blocks
    # Embeddings shape: [1, seq_len, 4096]
    data = embeddings[0] # [seq_len, 4096]
    
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, seq_length)
        
        # Slice: [chunk_len, 4096]
        chunk = data[start:end]
        
        # Mean Pool: [4096]
        pooled = chunk.mean(dim=0)
        
        # Move to CPU immediately
        block_vectors.append(pooled.cpu())
    
    del tokens, embeddings, data
    return block_vectors

def format_prompt(question: str, context: str) -> str:
    """Same format as Phase 1."""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based on the provided context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}

Please answer the question based on the context above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

def main():
    print("="*60)
    print("Scout-KV Phase 2: Training Data Generation")
    print("="*60)
    
    check_cuda()
    
    # Load Oracle Data
    oracle_file = Path("oracle_data.json")
    if not oracle_file.exists():
        print("Error: oracle_data.json not found.")
        sys.exit(1)
        
    with open(oracle_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    samples = data["samples"]
    print(f"Loaded {len(samples)} oracle samples.")
    
    # Build lookup for alignment (Question -> Sample)
    # Using question text as key because ID is missing
    oracle_lookup = {s["question"]: s for s in samples}
    
    # Load Dataset
    from datasets import load_dataset
    print("Loading NarrativeQA...")
    dataset = load_dataset("deepmind/narrativeqa", split="test", streaming=True)
    
    # Load Model
    model, tokenizer = load_embedding_model()
    
    training_data = []
    
    # Parameters MUST match Phase 1
    BLOCK_SIZE = 128
    MIN_TOKENS = 1000
    MAX_TOKENS = 2048
    
    print("\nGenerating features...")
    matched_count = 0
    
    # Iterate through streaming dataset
    # We stop when we find all samples or verify we checked enough
    # Since oracle_data.json has 50 samples, we should find them quickly.
    
    for i, ds_sample in enumerate(dataset):
        if matched_count >= len(samples):
            print("Found all oracle samples.")
            break
            
        # Parse sample
        try:
            context = ds_sample["document"]["text"]
            question = ds_sample["question"]["text"]
        except:
            continue
            
        # Check if in oracle lookup
        if question not in oracle_lookup:
            continue
            
        # Found a match!
        oracle_sample = oracle_lookup[question]
        
        # Validate Length Constraints (Must Match Phase 1 Logic)
        # Note: Phase 1 truncated inputs > MAX_TOKENS.
        # We must apply SAME truncation and formatting to get SAME blocks.
        
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        
        # Phase 1 Logic:
        # if len < min: skip (But oracle data passed this check)
        # if len > max: truncate
        
        if len(context_tokens) > MAX_TOKENS:
            context_tokens = context_tokens[:MAX_TOKENS]
            # Decode back to string to use format_prompt?
            # Phase 1: context = tokenizer.decode(context_tokens, skip_special_tokens=True)
            # This might change the text slightly!
            # We must replicate exactly.
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)
            
        prompt = format_prompt(question, context)
        
        # Generate Embeddings
        # Returns list of Tensors
        block_embeddings = get_block_embeddings(model, tokenizer, prompt, BLOCK_SIZE)
        
        # Align with Oracle Scores
        # oracle["all_block_scores"] is a dict "0": 0.1, "1": 0.2
        oracle_scores = oracle_sample["all_block_scores"]
        
        # Check integrity
        if len(block_embeddings) != len(oracle_scores):
            # This can happen if tokenization differs slightly or off-by-one
            # We skip to be safe, or truncate/pad?
            # Let's print warning and skip
            print(f"Warning: Block mismatch for sample '{question[:30]}...'")
            print(f"  Embeddings: {len(block_embeddings)}, Oracle: {len(oracle_scores)}")
            # Try to recover? Usually if length differs, it's safer to skip.
            continue
            
        # Pack data
        # Structure: List of {embedding, score, sample_id, block_id}
        for block_idx in range(len(block_embeddings)):
            vec = block_embeddings[block_idx] # Tensor
            score = oracle_scores.get(str(block_idx), 0.0) # Float
            
            training_data.append({
                "embedding": vec, # [4096]
                "label": float(score),
                "question": question,
                "block_id": block_idx
            })
            
        matched_count += 1
        print(f"Processed match {matched_count}/{len(samples)}")
        
        # GC
        if matched_count % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Save
    out_file = "training_dataset.pt"
    print(f"\nSaving {len(training_data)} training vectors to {out_file}...")
    torch.save(training_data, out_file)
    
    # Verification
    print("\nVerification:")
    if len(training_data) > 0:
        ex = training_data[0]
        print(f"  Shape: {ex['embedding'].shape}")
        print(f"  Label Type: {type(ex['label'])}")
    print("Done.")

if __name__ == "__main__":
    main()
