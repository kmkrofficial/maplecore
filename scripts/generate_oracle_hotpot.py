#!/usr/bin/env python3
"""
Generate Oracle Data (HotpotQA)
===============================

Generates ground-truth attention labels for HotpotQA using Llama-3-8B-Instruct.
This script complements `generate_oracle.py` (NarrativeQA) to create a diverse
multi-domain training set.

Method:
1. Load HotpotQA (distractor) dataset via HuggingFace.
2. Flatten the list of paragraphs into a single context string.
3. Feed to Llama-3 (4-bit).
4. Extract attention weights from the last layer.
5. Identify top-5 "heavy hitter" blocks.
6. Save to `data/oracle_hotpotqa.json`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.config import DATA_DIR
from maplecore.indexer import MapleIndexer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_CONTEXT_TOKENS = 1024  # Reduced from 1536 to save VRAM
OUTPUT_FILE = DATA_DIR / "oracle_hotpotqa.json"
CHECKPOINT_FILE = DATA_DIR / "oracle_hotpot_checkpoint.json"

def get_hotpot_samples(split="train", max_samples=None):
    """Load and format HotpotQA samples."""
    logger.info(f"Loading HotpotQA ({split})...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    
    samples = []
    count = 0
    
    for item in dataset:
        if max_samples and count >= max_samples:
            break
            
        # HotpotQA structure:
        # context: {'title': ['t1', 't2'], 'sentences': [['s1', 's2'], ['s3']]}
        # question: str
        # answer: str
        
        titles = item['context']['title']
        sentences_list = item['context']['sentences']
        
        # Flatten into one document
        full_text = ""
        for title, sentences in zip(titles, sentences_list):
            full_text += f"\n\nTitle: {title}\n"
            full_text += " ".join(sentences)
            
        samples.append({
            "id": item['id'],
            "question": item['question'],
            "text": full_text.strip(),
            "answer": item['answer']
        })
        count += 1
        
    logger.info(f"Loaded {len(samples)} samples.")
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=50, help="Number of samples to process")
    args = parser.parse_args()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    samples = get_hotpot_samples("train", args.max_samples)
    
    # 2. Load Model (4-bit)
    logger.info(f"Loading {MODEL_ID} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager", # Required for extracting weights
        torch_dtype=torch.float16,
    )
    model.eval()
    
    # 3. Process Samples
    indexer = MapleIndexer()
    indexer.chunk_size = 500 # Valid if property exists, otherwise it relies on config
    oracle_data = []
    
    # Resume if checkpoint exists
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            oracle_data = json.load(f)
            logger.info(f"Resumed from checkpoint: {len(oracle_data)} samples.")
    
    start_idx = len(oracle_data)
    
    logger.info("Generating oracle attention labels...")
    for i in tqdm(range(start_idx, len(samples))):
        sample = samples[i]
        
        # Chunk text
        blocks = indexer.chunk_text(sample["text"])
        if not blocks:
            continue
            
        # Build Prompt trace
        # We need to map tokens back to block IDs
        prompt_text = ""
        block_ranges = [] # (block_id, start_char, end_char) relative to prompt
        
        # System prompt
        system_msg = "You are a helpful AI assistant. Answer the user's question based on the context.\n\nContext:\n"
        prompt_text += system_msg
        
        current_pos = len(prompt_text)
        
        # Add blocks to prompt
        valid_blocks = []
        for block in blocks:
            # Simple format: [ID] Text
            block_str = f"[{block.id}] {block.text}\n"
            
            # Check token limit roughly (4 chars ~ 1 token)
            if (len(prompt_text) + len(block_str)) / 4 > MAX_CONTEXT_TOKENS:
                break
                
            prompt_text += block_str
            end_pos = current_pos + len(block_str)
            
            # We track valid blocks that made it into context
            valid_blocks.append(block)
            block_ranges.append((block.id, current_pos, end_pos))
            current_pos = end_pos
            
        prompt_text += f"\nQuestion: {sample['question']}\nAnswer:"
        
        # Tokenize (fast)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids
        
        # Check explicit token length
        if input_ids.shape[1] > MAX_CONTEXT_TOKENS:
            input_ids = input_ids[:, :MAX_CONTEXT_TOKENS]
            
        # Forward Pass
        with torch.no_grad():
            outputs = model(
                input_ids, 
                output_attentions=True,
                use_cache=False
            )
            
        # Extract Attention
        # Shape: (batch, heads, seq_len, seq_len)
        # We take the last layer, average across heads
        last_layer_attn = outputs.attentions[-1][0].mean(dim=0).cpu() # Move to CPU
        
        # Cleanup VRAM immediately
        del outputs
        torch.cuda.empty_cache()
        
        # We care about the attention of the LAST token (the one generating the answer)
        # towards all previous tokens.
        last_token_attn = last_layer_attn[-1, :] # (seq_len,)
        
        # Map tokens to blocks
        # We need token offsets. offset_mapping is supported by FastTokenizer
        token_offsets = tokenizer(prompt_text, return_offsets_mapping=True).offset_mapping
        # Truncate offsets to match input_ids length
        token_offsets = token_offsets[:input_ids.shape[1]]
        
        block_scores = {b.id: 0.0 for b in valid_blocks}
        
        for token_idx, score in enumerate(last_token_attn):
            start_char, end_char = token_offsets[token_idx]
            if start_char == end_char: continue # Special token
            
            # Find which block this token belongs to
            # This is O(N_blocks), could be optimized but N is small (10-50)
            for bid, b_start, b_end in block_ranges:
                if start_char >= b_start and end_char <= b_end:
                    block_scores[bid] += score.item()
                    break
        
        # Top-5 Blocks
        sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_ids = [bid for bid, score in sorted_blocks[:5]]
        
        # Save Sample
        # Store block texts for the trainer to re-embed
        processed_blocks = [{"id": b.id, "text": b.text} for b in valid_blocks]
        
        oracle_data.append({
            "id": sample["id"],
            "question": sample["question"],
            "blocks": processed_blocks,
            "labels": top_k_ids, # Target block IDs
            "domain": "hotpot_qa"
        })
        
        # Checkpoint
        if i % 10 == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(oracle_data, f)
                
    # Final Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(oracle_data, f, indent=2)
        
    logger.info(f"Saved {len(oracle_data)} samples to {OUTPUT_FILE}")
    
    # Cleanup checkpointer
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

if __name__ == "__main__":
    main()
