#!/usr/bin/env python3
"""
MAPLE Oracle Generator — NarrativeQA
====================================
Label which 500-char blocks a teacher LLM attends to when answering
NarrativeQA questions. See ``scripts/oracle_common.py`` for the label
definition and ``docs/DATA.md`` for the output schema.

Requirements:
    pip install -e ".[benchmarks,training]"   # + HF login for gated Llama-3

Usage:
    python scripts/generate_oracle.py --split train --max-samples 500
    python scripts/generate_oracle.py --split test  --max-samples 200
    python scripts/generate_oracle.py --max-samples 5           # smoke test
    python scripts/generate_oracle.py --resume                  # continue
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maplecore.indexer import MapleIndexer
from scripts.oracle_common import (
    DEFAULT_TEACHER,
    OracleScorer,
    answer_block_ids,
    build_prompt_with_mapping,
    load_teacher,
    make_metadata,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
TOP_K = 5
CHECKPOINT_EVERY = 25
OUT_DIR = Path("data/oracle")


def iter_narrativeqa(split: str):
    """Load NarrativeQA (cached by scripts/download_data.py)."""
    from datasets import load_dataset
    return load_dataset("deepmind/narrativeqa", split=split)


def run(split: str = "train", max_samples: int = 500, max_tokens: int = 1536,
        max_per_doc: int = 5, teacher: str = DEFAULT_TEACHER, resume: bool = False,
        output: Path | None = None, attn_impl: str = "fast", batch_size: int = 8) -> Path:
    output = output or OUT_DIR / f"narrativeqa_{split}.json"
    print("=" * 70)
    print("MAPLE Oracle Generator (NarrativeQA)")
    print(f"  Teacher:      {teacher}")
    print(f"  Split:        {split}")
    print(f"  Max context:  {max_tokens} tokens | chunk {CHUNK_SIZE} chars | top-{TOP_K}")
    print(f"  Max samples:  {max_samples} (<= {max_per_doc} per document)")
    print(f"  Attention:    {attn_impl} (batch {batch_size if attn_impl == 'fast' else 1})")
    print(f"  Output:       {output}")
    print("=" * 70)

    samples, done = [], set()
    if resume and output.exists():
        with open(output, "r", encoding="utf-8") as f:
            samples = json.load(f)["samples"]
        done = {s["id"] for s in samples}
        logger.info(f"Resuming with {len(samples)} samples")

    model, tokenizer = load_teacher(teacher, attn_impl=attn_impl)
    scorer = OracleScorer(model, tokenizer, attn_impl)
    if attn_impl == "eager":
        batch_size = 1
    chunker = MapleIndexer(device="cpu")  # chunking only; encoder never loaded

    per_doc = Counter(s.get("document_id") for s in samples)
    skipped = Counter()
    start_time = time.time()
    new_count = 0
    pending = []  # (record without scores, input_ids, block_ranges)
    last_save = len(samples)

    def save(final: bool = False):
        meta = make_metadata(
            "narrativeqa", teacher, max_tokens, CHUNK_SIZE, TOP_K, len(samples), split,
            max_per_doc=max_per_doc, skipped=dict(skipped), complete=final,
            attn_impl=attn_impl, batch_size=batch_size,
        )
        save_json({"metadata": meta, "samples": samples}, output)

    def flush():
        nonlocal new_count, last_save
        if not pending:
            return
        results = scorer.score([(ids, ranges) for _, ids, ranges in pending])
        for (rec, ids, ranges), res in zip(pending, results):
            if res is None:
                skipped["oom"] += 1
                per_doc[rec["document_id"]] -= 1
                continue
            last_scores, all_scores = res
            ranked = sorted(last_scores.items(), key=lambda x: x[1], reverse=True)
            rec["top_5_block_ids"] = [bid for bid, _ in ranked[:TOP_K]]
            rec["attention_scores"] = {str(k): round(v, 6) for k, v in ranked}
            rec["attention_scores_all_layers"] = {str(k): round(v, 6) for k, v in all_scores.items()}
            rec["attn_impl"] = attn_impl
            samples.append(rec)
            new_count += 1
        pending.clear()
        rate = new_count / max(time.time() - start_time, 1e-6)
        s = samples[-1]
        logger.info(f"[{len(samples):>4d}/{max_samples}] doc={s['document_id'][:8]} "
                    f"blocks={s['num_blocks_in_context']} top={s['top_5_block_ids']} ({rate:.2f}/s)")
        if len(samples) - last_save >= CHECKPOINT_EVERY:
            save()
            last_save = len(samples)

    for i, row in enumerate(iter_narrativeqa(split)):
        if len(samples) + len(pending) >= max_samples:
            break
        try:
            doc_id = row["document"]["id"]
            doc_text = row["document"]["text"]
            question = row["question"]["text"]
            answers = [a["text"] for a in row["answers"]]
        except (KeyError, TypeError):
            skipped["bad_row"] += 1
            continue

        sample_id = f"nqa-{split}-{i}"
        if sample_id in done:
            continue
        if per_doc[doc_id] >= max_per_doc:
            skipped["doc_quota"] += 1
            continue
        if len(doc_text) < CHUNK_SIZE * 2:
            skipped["short_doc"] += 1
            continue

        blocks = chunker.chunk_text(doc_text, chunk_size=CHUNK_SIZE)
        input_ids, block_ranges = build_prompt_with_mapping(blocks, question, tokenizer, max_tokens)
        if len(block_ranges) < TOP_K * 2:
            skipped["too_few_blocks"] += 1
            continue

        in_context_ids = {bid for bid, _, _ in block_ranges}
        in_context = [b for b in blocks if b.id in in_context_ids]
        pending.append(({
            "id": sample_id,
            "document_id": doc_id,
            "domain": "narrative_qa",
            "question": question,
            "answers": answers,
            "blocks": [{"id": b.id, "text": b.text} for b in in_context],
            # Heuristic: NarrativeQA answers are abstractive, so this is often empty.
            "gold_block_ids": answer_block_ids(in_context, answers),
            "num_blocks_in_context": len(block_ranges),
            "num_blocks_total": len(blocks),
            "seq_len": int(input_ids.shape[1]),
        }, input_ids, block_ranges))
        per_doc[doc_id] += 1
        if len(pending) >= batch_size:
            flush()

    flush()
    scorer.close()
    save(final=True)
    elapsed = time.time() - start_time
    print(f"\nDone: {new_count} new samples ({len(samples)} total) in {elapsed:.0f}s "
          f"({new_count / max(elapsed, 1e-6):.2f}/s); skipped={dict(skipped)}")
    print(f"Output: {output}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MAPLE Oracle Generator (NarrativeQA)")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=1536,
                   help="Teacher context budget (1536 reproduces v0.2.0)")
    p.add_argument("--max-per-doc", type=int, default=5,
                   help="Cap questions per story so samples are not dominated by a few books")
    p.add_argument("--teacher", default=DEFAULT_TEACHER)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--attn-impl", choices=["fast", "eager"], default="fast",
                   help="fast = SDPA + last-row capture, batched; eager = v0.2.0-style reference")
    p.add_argument("--batch-size", type=int, default=8, help="Prompts per forward pass (fast path)")
    a = p.parse_args()
    run(a.split, a.max_samples, a.max_tokens, a.max_per_doc, a.teacher, a.resume, a.output,
        a.attn_impl, a.batch_size)
