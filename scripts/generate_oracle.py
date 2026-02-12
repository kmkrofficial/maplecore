#!/usr/bin/env python3
"""
MAPLE Oracle Generator
=======================
Generate ground-truth attention labels by running Llama-3-8B-Instruct
over NarrativeQA and recording which text blocks receive the highest
attention from the model.

Requirements:
    pip install maplecore[benchmarks] bitsandbytes accelerate

Usage:
    python scripts/generate_oracle.py                # full run
    python scripts/generate_oracle.py --max-samples 5  # smoke test
    python scripts/generate_oracle.py --resume         # resume from checkpoint
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore import MapleIndexer
from benchmarks.data_loader import get_narrative_qa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_CONTEXT_TOKENS = 1536      # keeps VRAM under 12 GB with attention matrices
CHUNK_SIZE = 500               # characters per block
TOP_K = 5                      # oracle top-k blocks
CHECKPOINT_EVERY = 25          # save checkpoint every N samples

DATA_DIR = Path("data")
CHECKPOINT_PATH = DATA_DIR / "oracle_checkpoint.json"
OUTPUT_PATH = DATA_DIR / "oracle_data.json"


# ===================================================================
# Model Loading
# ===================================================================

def load_model_4bit(model_id: str = MODEL_ID):
    """
    Load a causal LM in 4-bit NF4 quantization.

    Uses bitsandbytes for quantization and eager attention so we can
    extract attention weight matrices.

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading {model_id} in 4-bit (NF4)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",   # required for attention weights
        torch_dtype=torch.float16,
    )
    model.eval()

    vram_mb = torch.cuda.memory_allocated() / 1024**2
    logger.info(f"Model loaded. VRAM used: {vram_mb:.0f} MB")

    return model, tokenizer


# ===================================================================
# Prompt Building & Token-to-Block Mapping
# ===================================================================

def build_prompt_with_mapping(
    blocks: list,
    question: str,
    tokenizer,
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    """
    Build a tokenized prompt and return token->block mapping.

    Prompt structure:
        [SYSTEM] Context: [block_0][block_1]... Question: {q} Answer:

    Returns:
        input_ids: tensor of shape [1, seq_len]
        block_ranges: list of (block_id, start_token_idx, end_token_idx)
    """
    system = (
        "Read the following context carefully and answer the question.\n\n"
        "Context:\n"
    )
    question_part = f"\n\nQuestion: {question}\nAnswer:"

    system_ids = tokenizer.encode(system, add_special_tokens=True)
    question_ids = tokenizer.encode(question_part, add_special_tokens=False)

    # Token budget for context blocks
    budget = max_tokens - len(system_ids) - len(question_ids) - 4  # margin

    block_ranges = []
    context_ids = []
    separator = tokenizer.encode("\n", add_special_tokens=False)

    for block in blocks:
        block_ids = tokenizer.encode(block.text, add_special_tokens=False)
        needed = len(block_ids) + len(separator)

        if len(context_ids) + needed > budget:
            break

        start = len(system_ids) + len(context_ids)
        context_ids.extend(block_ids)
        end = len(system_ids) + len(context_ids)

        block_ranges.append((block.id, start, end))
        context_ids.extend(separator)

    all_ids = system_ids + context_ids + question_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long)

    return input_ids, block_ranges


# ===================================================================
# Attention Extraction
# ===================================================================

def extract_block_attention(
    model,
    input_ids: torch.Tensor,
    block_ranges: List[Tuple[int, int, int]],
) -> Dict[int, float]:
    """
    Run a forward pass and aggregate last-layer attention per block.

    Looks at the LAST token's attention distribution (the position
    that would generate the answer) and sums attention scores across
    all tokens belonging to each block.

    Returns:
        Dict mapping block_id -> aggregated attention score
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            use_cache=False,
        )

    # Last layer attention: (1, num_heads, seq_len, seq_len)
    last_attn = outputs.attentions[-1]

    # Average across heads, take last token's attention to all positions
    # Shape: [seq_len]
    last_token_attn = last_attn[0, :, -1, :].mean(dim=0).float().cpu()

    # Aggressively free GPU memory
    del outputs, last_attn
    torch.cuda.empty_cache()

    # Aggregate attention per block
    block_scores = {}
    for block_id, start, end in block_ranges:
        score = last_token_attn[start:end].sum().item()
        block_scores[block_id] = score

    return block_scores


# ===================================================================
# Checkpointing
# ===================================================================

def save_checkpoint(samples: list, processed_questions: set, path: Path):
    """Save progress to a checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "samples": samples,
        "processed_questions": list(processed_questions),
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    logger.info(f"Checkpoint saved: {len(samples)} samples -> {path}")


def load_checkpoint(path: Path) -> Tuple[list, set]:
    """Load from a checkpoint file if it exists."""
    if not path.exists():
        return [], set()

    with open(path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    samples = checkpoint["samples"]
    processed = set(checkpoint["processed_questions"])
    logger.info(f"Resumed from checkpoint: {len(samples)} samples")
    return samples, processed


# ===================================================================
# Main
# ===================================================================

def run(max_samples: int = 500, resume: bool = False):
    """Run the oracle generation pipeline."""
    print("=" * 70)
    print("MAPLE Oracle Generator")
    print(f"  Model:       {MODEL_ID} (4-bit NF4)")
    print(f"  Max Context: {MAX_CONTEXT_TOKENS} tokens")
    print(f"  Chunk Size:  {CHUNK_SIZE} chars")
    print(f"  Top-K:       {TOP_K}")
    print(f"  Max Samples: {max_samples}")
    print("=" * 70)

    # ---- Resume / Fresh start ----
    if resume:
        oracle_samples, processed_questions = load_checkpoint(CHECKPOINT_PATH)
    else:
        oracle_samples, processed_questions = [], set()

    # ---- Load model ----
    model, tokenizer = load_model_4bit()

    # ---- Load dataset ----
    logger.info("Loading NarrativeQA train set...")
    dataset = get_narrative_qa("train")

    # ---- Indexer (text chunking only -- no BGE model loaded) ----
    indexer = MapleIndexer.__new__(MapleIndexer)
    indexer.device = "cpu"
    indexer.batch_size = 32
    indexer._model = None

    # ---- Process samples ----
    total_target = min(max_samples, len(dataset))
    start_time = time.time()
    skipped = 0

    for i, sample in enumerate(dataset):
        if len(oracle_samples) >= total_target:
            break

        try:
            doc_text = sample["document"]["text"]
            question = sample["question"]["text"]
        except (KeyError, TypeError):
            skipped += 1
            continue

        # Skip already processed
        if question in processed_questions:
            continue

        # Skip very short documents
        if len(doc_text) < CHUNK_SIZE * 2:
            skipped += 1
            continue

        # Chunk
        blocks = indexer.chunk_text(doc_text, chunk_size=CHUNK_SIZE)
        if len(blocks) < TOP_K:
            skipped += 1
            continue

        # Build prompt with block-to-token mapping
        input_ids, block_ranges = build_prompt_with_mapping(
            blocks, question, tokenizer, MAX_CONTEXT_TOKENS
        )

        if len(block_ranges) < TOP_K:
            skipped += 1
            continue

        # Extract attention
        try:
            block_scores = extract_block_attention(model, input_ids, block_ranges)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM on sample {i}, skipping (seq_len={input_ids.shape[1]})")
            torch.cuda.empty_cache()
            gc.collect()
            skipped += 1
            continue

        # Rank blocks by attention
        sorted_blocks = sorted(
            block_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_k_ids = [bid for bid, _ in sorted_blocks[:TOP_K]]

        # Store all block texts for training (needed for embedding generation)
        blocks_in_context = [bid for bid, _, _ in block_ranges]
        all_block_texts = {
            b.id: b.text for b in blocks if b.id in blocks_in_context
        }

        oracle_samples.append({
            "question": question,
            "top_5_block_ids": top_k_ids,
            "attention_scores": {str(bid): round(score, 6) for bid, score in sorted_blocks},
            "all_block_texts": all_block_texts,
            "num_blocks_in_context": len(block_ranges),
        })
        processed_questions.add(question)

        # Progress
        elapsed = time.time() - start_time
        rate = len(oracle_samples) / elapsed if elapsed > 0 else 0
        logger.info(
            f"[{len(oracle_samples):>4d}/{total_target}] "
            f"q=\"{question[:50]}...\" "
            f"top_blocks={top_k_ids} "
            f"({rate:.1f} samples/s)"
        )

        # Checkpoint
        if len(oracle_samples) % CHECKPOINT_EVERY == 0:
            save_checkpoint(oracle_samples, processed_questions, CHECKPOINT_PATH)

        # Memory management
        gc.collect()

    # ---- Save final output ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "model": MODEL_ID,
            "quantization": "4-bit NF4",
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "chunk_size": CHUNK_SIZE,
            "top_k": TOP_K,
            "num_samples": len(oracle_samples),
            "timestamp": datetime.now().isoformat(),
        },
        "samples": oracle_samples,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Oracle generation complete!")
    print(f"  Samples:  {len(oracle_samples)}")
    print(f"  Skipped:  {skipped}")
    print(f"  Time:     {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output:   {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPLE Oracle Generator")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum number of samples to process")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    run(max_samples=args.max_samples, resume=args.resume)
