#!/usr/bin/env python3
"""
Benchmark 07: Long-context evidence retrieval
=============================================

The setting MAPLE is pitched for: find the evidence for a question inside a
*large* pool of blocks. Ground truth is human evidence, not teacher attention,
so baselines run without any oracle data.

HotpotQA (distractor, validation)
    Each question's own paragraphs (2 gold + 8 distractors) are inserted
    contiguously at a random position into a pool of paragraphs drawn from
    *other* questions, for pool sizes POOL_SIZES (in blocks).
    gold = blocks overlapping a supporting sentence.

NarrativeQA (test)
    Whole stories (hundreds to thousands of 500-char blocks).
    gold = blocks containing a normalised answer string (only questions where
    at least one block matches are kept — reported as coverage).

Methods: BM25, dense cosine, MAPLE linear / hierarchical / adaptive.
Metrics: evidence_recall@k, all_evidence@k, answer_in@k, evidence_mrr@10,
         search latency (ms) and blocks scored per query.

Output: benchmarks/results/long_context.json

Usage:
    python benchmarks/07_long_context.py --max-questions 300
    python benchmarks/07_long_context.py --datasets hotpotqa --pool-sizes 10 100 1000 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics as M
from benchmarks.config import BGE_MODEL_NAME, MODEL_PATH, RESULTS_DIR
from maplecore import MapleIndexer, MapleScanner, load_model
from maplecore.indexer import Block, Index
from maplecore.utils import _cosine_similarity
from scripts.generate_oracle_hotpot import flatten_hotpot
from scripts.oracle_common import answer_block_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

K_VALUES = [1, 5, 10, 20]
_CHUNKER = MapleIndexer(device="cpu")  # chunking only; never loads an encoder
POOL_SIZES = [10, 100, 1000, 5000]
CHUNK = 500


# ----------------------------------------------------------------- rankers

class Rankers:
    def __init__(self, indexer: MapleIndexer, model: Optional[torch.nn.Module], chapter_size: int = 100):
        self.indexer = indexer
        self.scanner = MapleScanner(model, device=indexer.device, chapter_size=chapter_size) if model else None

    def rank_all(self, question: str, index: Index, texts: List[str], k_max: int,
                 bm25=None) -> Dict[str, dict]:
        q = self.indexer.encode_query(question)
        out = {}

        if bm25 is not None:
            t = time.perf_counter()
            scores = bm25.get_scores(question.lower().split())
            order = np.argsort(-scores, kind="stable")[:k_max].tolist()
            out["bm25"] = {"ranking": order, "ms": (time.perf_counter() - t) * 1000, "scored": len(texts)}

        t = time.perf_counter()
        sims = _cosine_similarity(q.to(index.embeddings.device), index.embeddings)
        order = torch.topk(sims, min(k_max, len(sims))).indices.tolist()
        out["dense"] = {"ranking": order, "ms": (time.perf_counter() - t) * 1000, "scored": len(texts)}

        if self.scanner:
            for strat in ["linear", "hierarchical", "adaptive"]:
                r = self.scanner.search(q, index, k=k_max, strategy=strat)
                out[f"maple_{strat}"] = {
                    "ranking": r.block_ids, "ms": r.latency_ms,
                    "scored": r.metadata.get("blocks_scored", r.metadata.get("blocks_scanned", len(texts))),
                }
        return out


def _score(ranking: List[int], gold: set, texts: List[str], answers: List[str]) -> Dict[str, Optional[float]]:
    out = {}
    for k in K_VALUES:
        out[f"evidence_recall@{k}"] = M.recall_at_k(ranking, gold, k)
        out[f"all_evidence@{k}"] = M.hit_all_at_k(ranking, gold, k)
        out[f"answer_in@{k}"] = M.answer_in_context(" ".join(texts[i] for i in ranking[:k]), answers, min_len=3)
    out["evidence_mrr@10"] = M.mrr_at_k(ranking, gold, 10)
    return out


# ---------------------------------------------------------------- hotpotqa

def _paragraph_blocks(item: dict) -> List[dict]:
    """Chunk each paragraph of a HotpotQA item separately; mark gold blocks."""
    text, sent_spans, _ = flatten_hotpot(item)
    blocks = []
    # Split flattened text back into paragraphs on the blank-line separator
    pos = 0
    for para in text.split("\n\n"):
        if not para.strip():
            pos += len(para) + 2
            continue
        for b in _CHUNKER.chunk_text(para, CHUNK):
            s, e = pos + b.start_char, pos + b.end_char
            gold = any(s < se and ss < e for ss, se in sent_spans)
            blocks.append({"text": b.text, "gold": gold})
        pos += len(para) + 2
    return blocks


def eval_hotpot(rankers: Rankers, max_questions: int, pool_sizes: Sequence[int], seed: int) -> dict:
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    eval_ids = order[:max_questions]
    pool_ids = order[max_questions:max_questions + max(pool_sizes) // 3 + 200]

    logger.info("HotpotQA: building distractor pool...")
    pool_blocks = []
    for i in pool_ids:
        pool_blocks.extend(b["text"] for b in _paragraph_blocks(ds[i]))
        if len(pool_blocks) >= max(pool_sizes):
            break
    pool_emb = rankers.indexer.encode_texts(pool_blocks, show_progress=True)
    logger.info(f"  pool: {len(pool_blocks)} blocks")

    results = {}
    for size in pool_sizes:
        per_method = defaultdict(lambda: defaultdict(list))
        for n, qi in enumerate(eval_ids):
            item = ds[qi]
            own = _paragraph_blocks(item)
            own_emb = rankers.indexer.encode_texts([b["text"] for b in own])
            n_distr = max(0, size - len(own))
            distr_idx = rng.sample(range(len(pool_blocks)), min(n_distr, len(pool_blocks)))
            insert_at = rng.randint(0, len(distr_idx))

            texts = [pool_blocks[i] for i in distr_idx[:insert_at]] + [b["text"] for b in own] + \
                    [pool_blocks[i] for i in distr_idx[insert_at:]]
            emb = torch.cat([pool_emb[distr_idx[:insert_at]], own_emb, pool_emb[distr_idx[insert_at:]]]) \
                if distr_idx else own_emb
            gold = {insert_at + j for j, b in enumerate(own) if b["gold"]}
            index = Index(blocks=[Block(i, t) for i, t in enumerate(texts)], embeddings=emb, chunk_size=CHUNK)
            bm25 = BM25Okapi([t.lower().split() for t in texts])

            for method, r in rankers.rank_all(item["question"], index, texts, max(K_VALUES), bm25).items():
                for metric, v in _score(r["ranking"], gold, texts, [item["answer"]]).items():
                    per_method[method][metric].append(v)
                per_method[method]["latency_ms"].append(r["ms"])
                per_method[method]["blocks_scored"].append(r["scored"])
            if (n + 1) % 100 == 0:
                logger.info(f"  pool={size}: {n + 1}/{len(eval_ids)}")

        results[str(size)] = _summarize(per_method)
        _print(f"HotpotQA pool={size} blocks", results[str(size)])
    return {"pool_sizes": list(pool_sizes), "num_questions": len(eval_ids), "results": results}


# ------------------------------------------------------------- narrativeqa

def eval_narrativeqa(rankers: Rankers, max_questions: int, seed: int, max_per_doc: int = 5) -> dict:
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi

    ds = load_dataset("deepmind/narrativeqa", split="test")
    per_method = defaultdict(lambda: defaultdict(list))
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    doc_cache = {}
    seen_per_doc = defaultdict(int)
    kept = examined = 0

    for row in ds:
        if kept >= max_questions:
            break
        doc_id = row["document"]["id"]
        if seen_per_doc[doc_id] >= max_per_doc:
            continue
        examined += 1
        if doc_id not in doc_cache:
            doc_cache.clear()  # rows are grouped by document; keep one in memory
            blocks = rankers.indexer.chunk_text(row["document"]["text"], CHUNK)
            texts = [b.text for b in blocks]
            emb = rankers.indexer.encode_texts(texts)
            doc_cache[doc_id] = (blocks, texts, Index(blocks=blocks, embeddings=emb, chunk_size=CHUNK),
                                 BM25Okapi([t.lower().split() for t in texts]))
        blocks, texts, index, bm25 = doc_cache[doc_id]
        answers = [a["text"] for a in row["answers"]]
        gold = set(answer_block_ids(blocks, answers, min_len=4))
        if not gold:
            continue
        seen_per_doc[doc_id] += 1
        kept += 1
        bucket = "<500" if len(blocks) < 500 else "500-1000" if len(blocks) < 1000 else "1000-2000" if len(blocks) < 2000 else ">=2000"

        for method, r in rankers.rank_all(row["question"]["text"], index, texts, max(K_VALUES), bm25).items():
            for metric, v in _score(r["ranking"], gold, texts, answers).items():
                per_method[method][metric].append(v)
                buckets[bucket][method][metric].append(v)
            per_method[method]["latency_ms"].append(r["ms"])
            per_method[method]["blocks_scored"].append(r["scored"])
            per_method[method]["pool_blocks"].append(len(blocks))
        if kept % 100 == 0:
            logger.info(f"  NarrativeQA: {kept}/{max_questions}")

    res = {"num_questions": kept, "questions_examined": examined,
           "gold_coverage": kept / max(examined, 1),
           "results": _summarize(per_method),
           "by_pool_size": {b: _summarize(d) for b, d in buckets.items()}}
    _print("NarrativeQA (full stories)", res["results"])
    print(f"  answer-span gold available for {res['gold_coverage'] * 100:.0f}% of examined questions")
    return res


# ------------------------------------------------------------------ output

def _summarize(per_method) -> dict:
    out = {}
    for method, d in per_method.items():
        out[method] = {}
        for metric, vals in d.items():
            if metric in ("latency_ms", "blocks_scored", "pool_blocks"):
                arr = np.array(vals, dtype=float)
                out[method][metric] = {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
                                       "p95": float(np.percentile(arr, 95))}
            else:
                out[method][metric] = M.summarize(vals)
    return out


def _print(title: str, res: dict) -> None:
    cols = ["evidence_recall@5", "all_evidence@10", "answer_in@5", "evidence_mrr@10"]
    print(f"\n--- {title} ---")
    print(f"  {'method':<20}" + "".join(f"{c:>19}" for c in cols) + f"{'p50 ms':>10}{'scored':>9}")
    for method, d in res.items():
        row = "".join(f"{d[c]['mean'] * 100:>18.1f}%" if d.get(c) and d[c]["n"] else f"{'n/a':>19}" for c in cols)
        print(f"  {method:<20}{row}{d['latency_ms']['p50']:>10.2f}{d['blocks_scored']['mean']:>9.0f}")


def run(model_path: Optional[Path] = None, embedding_model: str = BGE_MODEL_NAME,
        datasets: Sequence[str] = ("hotpotqa", "narrativeqa"), max_questions: int = 300,
        pool_sizes: Sequence[int] = POOL_SIZES, seed: int = 0, device: Optional[str] = None) -> dict:
    model_path = Path(model_path or MODEL_PATH)
    indexer = MapleIndexer(model_name=embedding_model, device=device)
    model = load_model(model_path, device=indexer.device) if model_path.exists() else None
    if model is None:
        logger.warning(f"No MAPLE model at {model_path}; running baselines only.")
    rankers = Rankers(indexer, model)

    out = {"config": {"model_path": str(model_path) if model else None, "embedding_model": embedding_model,
                      "max_questions": max_questions, "pool_sizes": list(pool_sizes), "seed": seed,
                      "chunk_size": CHUNK, "k_values": K_VALUES}}
    if "hotpotqa" in datasets:
        out["hotpotqa"] = eval_hotpot(rankers, max_questions, pool_sizes, seed)
    if "narrativeqa" in datasets:
        out["narrativeqa"] = eval_narrativeqa(rankers, max_questions, seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if model else "_baselines"
    path = RESULTS_DIR / f"long_context{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Long-context evidence retrieval")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--embedding-model", default=BGE_MODEL_NAME)
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "narrativeqa"], choices=["hotpotqa", "narrativeqa"])
    p.add_argument("--max-questions", type=int, default=300)
    p.add_argument("--pool-sizes", type=int, nargs="+", default=POOL_SIZES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    from scripts.runlog import flatten_metrics, record_outputs, start_run
    with start_run("bench", "07_long_context", vars(a)) as _run:
        _run.log_metrics(flatten_metrics(run(a.model, a.embedding_model, a.datasets, a.max_questions, a.pool_sizes, a.seed, a.device)))
        record_outputs(_run, RESULTS_DIR / 'long_context.json', RESULTS_DIR / 'long_context_baselines.json')
