#!/usr/bin/env python3
"""
MAPLE Oracle Generator — HotpotQA (distractor)
==============================================
Same teacher, prompt and attention aggregation as ``generate_oracle.py``
(see ``scripts/oracle_common.py``), applied to HotpotQA. The 10 paragraphs
(2 gold + 8 distractors) are flattened into one document and chunked into
500-char blocks.

Because HotpotQA annotates supporting *sentences*, each sample also stores
``gold_block_ids`` — the blocks overlapping a supporting sentence — which lets
us measure how well the attention oracle agrees with human evidence.

Usage:
    python scripts/generate_oracle_hotpot.py --split train --max-samples 500
    python scripts/generate_oracle_hotpot.py --split validation --max-samples 200
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore.indexer import MapleIndexer
from scripts.oracle_common import (
    DEFAULT_TEACHER,
    LastRowAttentionRecorder,
    block_attention,
    build_prompt_with_mapping,
    load_teacher,
    make_metadata,
    overlapping_block_ids,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
TOP_K = 5
CHECKPOINT_EVERY = 25
OUT_DIR = Path("data/oracle")


def flatten_hotpot(item: dict):
    """
    Flatten HotpotQA context into one string.

    Returns:
        text, list of (start, end) char spans of supporting sentences,
        list of (start, end) char spans of supporting paragraphs
    """
    support = {}
    for title, sent_id in zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]):
        support.setdefault(title, set()).add(sent_id)

    parts, pos = [], 0
    sent_spans, para_spans = [], []
    for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
        header = f"Title: {title}\n"
        para_start = pos
        parts.append(header)
        pos += len(header)
        for j, sent in enumerate(sentences):
            piece = sent if j == 0 else " " + sent
            if title in support and j in support[title]:
                lead = 0 if j == 0 else 1
                sent_spans.append((pos + lead, pos + len(piece)))
            parts.append(piece)
            pos += len(piece)
        if title in support:
            para_spans.append((para_start, pos))
        parts.append("\n\n")
        pos += 2
    return "".join(parts), sent_spans, para_spans


def run(split: str = "train", max_samples: int = 500, max_tokens: int = 1536,
        teacher: str = DEFAULT_TEACHER, resume: bool = False, output: Path | None = None) -> Path:
    from datasets import load_dataset

    output = output or OUT_DIR / f"hotpotqa_{split}.json"
    print("=" * 70)
    print("MAPLE Oracle Generator (HotpotQA distractor)")
    print(f"  Teacher:     {teacher}")
    print(f"  Split:       {split}")
    print(f"  Max context: {max_tokens} tokens | chunk {CHUNK_SIZE} chars | top-{TOP_K}")
    print(f"  Max samples: {max_samples}")
    print(f"  Output:      {output}")
    print("=" * 70)

    samples, done = [], set()
    if resume and output.exists():
        with open(output, "r", encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        done = {s["id"] for s in samples}

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    model, tokenizer = load_teacher(teacher)
    recorder = LastRowAttentionRecorder(model)
    chunker = MapleIndexer(device="cpu")
    skipped = Counter()
    start_time = time.time()

    def save(final: bool = False):
        meta = make_metadata("hotpotqa_distractor", teacher, max_tokens, CHUNK_SIZE, TOP_K,
                             len(samples), split, skipped=dict(skipped), complete=final)
        save_json({"metadata": meta, "samples": samples}, output)

    for item in dataset:
        if len(samples) >= max_samples:
            break
        if item["id"] in done:
            continue

        text, sent_spans, para_spans = flatten_hotpot(item)
        blocks = chunker.chunk_text(text, chunk_size=CHUNK_SIZE)
        input_ids, block_ranges = build_prompt_with_mapping(blocks, item["question"], tokenizer, max_tokens)
        if len(block_ranges) < TOP_K + 1:
            skipped["too_few_blocks"] += 1
            continue

        try:
            last_scores, all_scores = block_attention(model, recorder, input_ids, block_ranges)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            skipped["oom"] += 1
            continue

        ranked = sorted(last_scores.items(), key=lambda x: x[1], reverse=True)
        in_context_ids = {bid for bid, _, _ in block_ranges}
        in_context = [b for b in blocks if b.id in in_context_ids]
        gold = overlapping_block_ids(in_context, sent_spans)

        samples.append({
            "id": item["id"],
            "domain": "hotpot_qa",
            "type": item.get("type"),
            "level": item.get("level"),
            "question": item["question"],
            "answers": [item["answer"]],
            "blocks": [{"id": b.id, "text": b.text} for b in in_context],
            "top_5_block_ids": [bid for bid, _ in ranked[:TOP_K]],
            "attention_scores": {str(k): round(v, 6) for k, v in ranked},
            "attention_scores_all_layers": {str(k): round(v, 6) for k, v in all_scores.items()},
            "gold_block_ids": gold,
            "gold_paragraph_block_ids": overlapping_block_ids(in_context, para_spans),
            "num_blocks_in_context": len(block_ranges),
            "num_blocks_total": len(blocks),
            "seq_len": int(input_ids.shape[1]),
        })

        rate = len(samples) / max(time.time() - start_time, 1e-6)
        logger.info(f"[{len(samples):>4d}/{max_samples}] blocks={len(block_ranges)} "
                    f"top={samples[-1]['top_5_block_ids']} gold={gold} ({rate:.2f}/s)")
        if len(samples) % CHECKPOINT_EVERY == 0:
            save()

    recorder.remove()
    save(final=True)
    print(f"\nDone: {len(samples)} samples in {time.time() - start_time:.0f}s; skipped={dict(skipped)}")
    print(f"Output: {output}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MAPLE Oracle Generator (HotpotQA)")
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--teacher", default=DEFAULT_TEACHER)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    run(a.split, a.max_samples, a.max_tokens, a.teacher, a.resume, a.output)
