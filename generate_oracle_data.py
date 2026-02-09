#!/usr/bin/env python3
"""
generate_oracle_data.py
=======================
Scout-KV Phase 1: Validation Script

This script validates the "Sparsity Hypothesis" by extracting ground-truth
attention data from Llama-3-8B-Instruct on NarrativeQA long documents.

Hardware Requirements:
- NVIDIA GPU with 16GB+ VRAM (uses 4-bit quantization)
- 32GB+ System RAM recommended
- CUDA-enabled environment

Author: SVScout Research Team
"""

import json
import sys
import gc
from pathlib import Path
from typing import Optional

import torch


def check_cuda_availability() -> None:
    """Check if CUDA is available. Exit immediately if not."""
    if not torch.cuda.is_available():
        print("=" * 60)
        print("ERROR: CUDA is not available!")
        print("=" * 60)
        print("\nThis script requires an NVIDIA GPU with CUDA support.")
        print("Please ensure you have:")
        print("  1. An NVIDIA GPU installed")
        print("  2. CUDA drivers installed")
        print("  3. PyTorch with CUDA support (torch.cuda.is_available() == True)")
        print(f"\nCurrent PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        sys.exit(1)
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")


def load_model_and_tokenizer():
    """
    Load Llama-3-8B-Instruct with 4-bit quantization.
    
    CRITICAL: Uses bitsandbytes for 4-bit quantization to fit in 16GB VRAM.
    Uses eager attention to extract raw attention weights.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    print(f"\nLoading model: {model_name}")
    print("Using 4-bit quantization (bitsandbytes)...")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Reduces memory further
        bnb_4bit_quant_type="nf4"  # Normal float 4-bit
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    # CRITICAL: attn_implementation="eager" is required to extract attention weights
    # FlashAttention and SDPA often hide the raw attention matrices
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="eager",  # REQUIRED for attention extraction
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Set to evaluation mode
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  - Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer


def load_narrativeqa_dataset():
    """Load NarrativeQA dataset (test split)."""
    from datasets import load_dataset
    
    print("\nLoading NarrativeQA dataset (test split)...")
    
    # Load the dataset
    dataset = load_dataset("deepmind/narrativeqa", split="test")
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    return dataset


def format_prompt(question: str, context: str) -> str:
    """
    Format the prompt for Llama-3-Instruct.
    Uses the official Llama-3 chat template.
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions based on the provided context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}

Please answer the question based on the context above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def extract_attention_scores(
    model,
    tokenizer,
    prompt: str,
    block_size: int = 512
) -> tuple[list[float], list[int], int]:
    """
    Extract attention scores for the last token across all context blocks.
    
    Args:
        model: The loaded LLM
        tokenizer: The tokenizer
        prompt: The formatted prompt string
        block_size: Size of each block in tokens (default: 512)
    
    Returns:
        all_block_scores: List of attention scores for each block
        top_5_block_ids: IDs of the top 5 blocks
        num_blocks: Total number of blocks
    """
    # Tokenize the input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192  # Safety limit
    ).to(model.device)
    
    seq_length = inputs["input_ids"].shape[1]
    
    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )
    
    # Get attention from the last layer
    # Shape: (batch_size, num_heads, seq_length, seq_length)
    last_layer_attention = outputs.attentions[-1]
    
    # Get attention weights for the last token (the one that would generate the answer)
    # Shape: (batch_size, num_heads, seq_length)
    last_token_attention = last_layer_attention[:, :, -1, :]
    
    # Average across all heads
    # Shape: (batch_size, seq_length)
    avg_attention = last_token_attention.mean(dim=1)
    
    # Move to CPU and convert to numpy
    attention_scores = avg_attention[0].cpu().float().numpy()
    
    # Calculate number of blocks
    num_blocks = (seq_length + block_size - 1) // block_size
    
    # Sum attention scores for each block
    all_block_scores = []
    for block_id in range(num_blocks):
        start_idx = block_id * block_size
        end_idx = min((block_id + 1) * block_size, seq_length)
        block_score = float(attention_scores[start_idx:end_idx].sum())
        all_block_scores.append(block_score)
    
    # Normalize block scores (so they sum to 1)
    total_score = sum(all_block_scores)
    if total_score > 0:
        all_block_scores = [score / total_score for score in all_block_scores]
    
    # Get top 5 blocks
    indexed_scores = [(i, score) for i, score in enumerate(all_block_scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_5_block_ids = [idx for idx, _ in indexed_scores[:5]]
    
    # Clean up to prevent OOM
    del outputs, last_layer_attention, last_token_attention, avg_attention
    torch.cuda.empty_cache()
    
    return all_block_scores, top_5_block_ids, num_blocks


def process_sample(
    model,
    tokenizer,
    sample: dict,
    min_tokens: int = 4000,
    max_tokens: int = 8000,
    block_size: int = 512
) -> Optional[dict]:
    """
    Process a single NarrativeQA sample.
    
    Returns None if the document is too short.
    """
    # Extract fields from NarrativeQA
    # NarrativeQA structure: document.text, question.text, answers[].text
    try:
        context = sample["document"]["text"]
        question = sample["question"]["text"]
        answers = [a["text"] for a in sample["answers"]]
        answer = answers[0] if answers else ""
    except (KeyError, TypeError) as e:
        print(f"  Warning: Could not extract fields: {e}")
        return None
    
    # Check document length
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    
    if len(context_tokens) < min_tokens:
        return None  # Too short
    
    # Truncate if necessary
    if len(context_tokens) > max_tokens:
        context_tokens = context_tokens[:max_tokens]
        context = tokenizer.decode(context_tokens, skip_special_tokens=True)
    
    # Format the prompt
    prompt = format_prompt(question, context)
    
    # Extract attention scores
    try:
        all_block_scores, top_5_block_ids, num_blocks = extract_attention_scores(
            model, tokenizer, prompt, block_size
        )
    except Exception as e:
        print(f"  Warning: Attention extraction failed: {e}")
        return None
    
    # Build result
    result = {
        "question": question,
        "answer": answer,
        "top_5_block_ids": top_5_block_ids,
        "all_block_scores": {str(i): score for i, score in enumerate(all_block_scores)},
        "num_blocks": num_blocks,
        "context_tokens": len(context_tokens)
    }
    
    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Scout-KV Phase 1: Oracle Data Generation")
    print("=" * 60)
    
    # Configuration
    NUM_SAMPLES = 50
    MIN_TOKENS = 4000
    MAX_TOKENS = 8000
    BLOCK_SIZE = 512
    OUTPUT_FILE = Path("oracle_data.json")
    
    print(f"\nConfiguration:")
    print(f"  - Target samples: {NUM_SAMPLES}")
    print(f"  - Min tokens: {MIN_TOKENS}")
    print(f"  - Max tokens: {MAX_TOKENS}")
    print(f"  - Block size: {BLOCK_SIZE}")
    print(f"  - Output file: {OUTPUT_FILE}")
    
    # Step 1: Check CUDA
    check_cuda_availability()
    
    # Step 2: Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Step 3: Load dataset
    dataset = load_narrativeqa_dataset()
    
    # Step 4: Process samples
    print(f"\nProcessing samples...")
    results = []
    samples_processed = 0
    samples_skipped = 0
    
    for i, sample in enumerate(dataset):
        if len(results) >= NUM_SAMPLES:
            break
        
        print(f"\n[{i+1}] Processing sample {i}...", end=" ")
        
        result = process_sample(
            model, tokenizer, sample,
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS,
            block_size=BLOCK_SIZE
        )
        
        if result is None:
            print("SKIPPED (too short or error)")
            samples_skipped += 1
            continue
        
        results.append(result)
        samples_processed += 1
        
        print(f"OK ({result['context_tokens']} tokens, {result['num_blocks']} blocks)")
        print(f"  Question: {result['question'][:80]}...")
        print(f"  Top 5 blocks: {result['top_5_block_ids']}")
        
        # Periodic garbage collection
        if samples_processed % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Step 5: Save results
    print(f"\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  - Samples processed: {samples_processed}")
    print(f"  - Samples skipped: {samples_skipped}")
    print(f"=" * 60)
    
    # Save to JSON
    output_data = {
        "metadata": {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "quantization": "4-bit (bitsandbytes)",
            "block_size": BLOCK_SIZE,
            "min_tokens": MIN_TOKENS,
            "max_tokens": MAX_TOKENS,
            "num_samples": len(results)
        },
        "samples": results
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    
    # Print summary statistics
    if results:
        all_top_blocks = [r["top_5_block_ids"][0] for r in results]
        avg_top_block = sum(all_top_blocks) / len(all_top_blocks)
        
        print(f"\nQuick Statistics:")
        print(f"  - Average top block position: {avg_top_block:.1f}")
        
        # Check sparsity hypothesis
        # If most attention goes to a few blocks, the hypothesis holds
        avg_top5_attention = 0
        for r in results:
            top5_scores = sum(
                r["all_block_scores"][str(bid)] 
                for bid in r["top_5_block_ids"]
            )
            avg_top5_attention += top5_scores
        avg_top5_attention /= len(results)
        
        print(f"  - Average attention in top-5 blocks: {avg_top5_attention*100:.1f}%")
        
        if avg_top5_attention > 0.5:
            print(f"\n✓ SPARSITY HYPOTHESIS SUPPORTED!")
            print(f"  Top 5 blocks receive {avg_top5_attention*100:.1f}% of attention")
        else:
            print(f"\n⚠ Sparsity hypothesis needs further investigation")
            print(f"  Top 5 blocks only receive {avg_top5_attention*100:.1f}% of attention")


if __name__ == "__main__":
    main()
