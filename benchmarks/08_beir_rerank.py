#!/usr/bin/env python3
"""
Benchmark 08: BEIR top-100 re-ranking (the protocol of ICR and QRHead)
=====================================================================

BM25 retrieves the top ``first_stage_k`` (default 100) documents per query from the *whole* BEIR
corpus; every ranker re-orders those candidates; metric = nDCG@10 against the graded test qrels.
Re-ranking BM25's top-100 is the standard used by ICR (Chen et al., ICLR 2025, Table 1) and, with the
top-200, by QRHead (Zhang et al., EMNLP 2025), so the numbers can be put next to theirs.

Sanity rule: our BM25 nDCG@10 must be within +/-2.0 of the published BM25 values (SciFact 67.9, NFCorpus 32.2,
FiQA 23.6, TREC-COVID 59.5, from ICR Table 1). Otherwise ``protocol_mismatch`` is set and the numbers must
not be compared with papers.

Usage:
    python benchmarks/08_beir_rerank.py --rankers bm25 dense hybrid cross_encoder maple:main --model main=models/x.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics as M
from benchmarks.config import BGE_MODEL_NAME, RESULTS_DIR
from benchmarks.rankers import build_rankers, fit_rankers

logger = logging.getLogger(__name__)

DATASETS = ("scifact", "nfcorpus", "fiqa", "trec-covid")
# BM25 (flat, BEIR leaderboard) nDCG@10 x100 as reported in ICR Table 1
PUBLISHED_BM25 = {"scifact": 67.9, "nfcorpus": 32.2, "fiqa": 23.6, "trec-covid": 59.5}
BM25_TOLERANCE = 2.0
DEFAULT_RANKERS = ("bm25", "dense", "hybrid", "cross_encoder", "maple:main")


def bm25_sanity(ours: Dict[str, float]) -> Dict[str, dict]:
    """Compare our BM25 nDCG@10 (fractions 0..1) with the published values; unknown datasets are skipped."""
    out = {}
    for name, value in ours.items():
        if name in PUBLISHED_BM25:
            pct = round(value * 100, 1)
            out[name] = {"ours": pct, "published": PUBLISHED_BM25[name],
                         "ok": abs(pct - PUBLISHED_BM25[name]) <= BM25_TOLERANCE}
    return out


def _doc_text(doc: dict) -> str:
    return f"{doc.get('title', '')} {doc['text']}".strip()


def bm25_first_stage(corpus_ids: Sequence[str], corpus_texts: Sequence[str], queries: Sequence[str], k: int):
    """Top-k corpus indices per query with bm25s (Lucene BM25, English stop-words)."""
    import bm25s

    k = min(k, len(corpus_ids))
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(list(corpus_texts), stopwords="en", show_progress=False), show_progress=False)
    docs, _ = retriever.retrieve(bm25s.tokenize(list(queries), stopwords="en", show_progress=False),
                                 k=k, show_progress=False)
    return docs


def evaluate_rerank(corpus: Dict[str, dict], queries: Dict[str, str], qrels: Dict[str, Dict[str, int]],
                    rankers: Sequence, first_stage_k: int = 100, max_queries: Optional[int] = None,
                    seed: int = 0, indexer=None) -> dict:
    """nDCG@10 of each ranker re-ordering BM25's top-``first_stage_k``; also reports BM25's own order.

    ``indexer`` (a MapleIndexer) supplies query / candidate embeddings for dense-style rankers;
    without it rankers receive ``None`` embeddings (fine for lexical ones).
    """
    qids = sorted(q for q in queries if q in qrels and any(g > 0 for g in qrels[q].values()))
    if max_queries and len(qids) > max_queries:
        random.Random(seed).shuffle(qids)
        qids = sorted(qids[:max_queries])
    corpus_ids = list(corpus)
    corpus_texts = [_doc_text(corpus[d]) for d in corpus_ids]
    first = bm25_first_stage(corpus_ids, corpus_texts, [queries[q] for q in qids], first_stage_k)
    fit_rankers(rankers, corpus_texts)  # BM25 / hybrid re-rank with corpus-level IDF, not top-k-local IDF

    scores: Dict[str, List[Optional[float]]] = {"_first_stage:bm25": []}
    for r in rankers:
        scores[r.name] = []
    for qi, qid in enumerate(qids):
        cand = [int(i) for i in first[qi]]
        texts = [corpus_texts[i] for i in cand]
        doc_ids = [corpus_ids[i] for i in cand]
        scores["_first_stage:bm25"].append(M.ndcg_at_k(doc_ids, qrels[qid], 10))
        q_emb = block_embs = None
        if indexer is not None:
            q_emb, block_embs = indexer.encode_query(queries[qid]), indexer.encode_texts(texts)
        for r in rankers:
            order = r.rank(queries[qid], texts, q_emb, block_embs)
            scores[r.name].append(M.ndcg_at_k([doc_ids[i] for i in order], qrels[qid], 10))

    out = {name: {"ndcg@10": M.summarize(vals)} for name, vals in scores.items() if not name.startswith("_first_stage")}
    out["_first_stage"] = {"bm25": {"ndcg@10": M.summarize(scores["_first_stage:bm25"])}}
    return out


def load_beir(name: str) -> tuple[Dict[str, dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load corpus, queries and test qrels of a BeIR/<name> dataset from the Hugging Face hub (cached)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def fetch(repo, path):
        return hf_hub_download(repo, path, repo_type="dataset")

    corpus_df = pd.read_parquet(fetch(f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet"))
    queries_df = pd.read_parquet(fetch(f"BeIR/{name}", "queries/queries-00000-of-00001.parquet"))
    qrels_df = pd.read_csv(fetch(f"BeIR/{name}-qrels", "test.tsv"), sep="\t")
    corpus = {str(r["_id"]): {"title": r.get("title") or "", "text": r["text"]} for _, r in corpus_df.iterrows()}
    queries = {str(r["_id"]): r["text"] for _, r in queries_df.iterrows()}
    qrels: Dict[str, Dict[str, int]] = {}
    for _, r in qrels_df.iterrows():
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    return corpus, queries, qrels


def run(datasets: Sequence[str] = DATASETS, first_stage_k: int = 100, rankers: Sequence[str] = DEFAULT_RANKERS,
        max_queries: Optional[int] = None, model_paths: Optional[Dict[str, Path]] = None, seed: int = 0,
        embedding_model: str = BGE_MODEL_NAME, device: Optional[str] = None) -> dict:
    from maplecore import MapleIndexer

    ranker_objs = build_rankers(list(rankers), model_paths or {})
    indexer = MapleIndexer(model_name=embedding_model, device=device)
    results = {"config": {"datasets": list(datasets), "first_stage_k": first_stage_k, "rankers": list(rankers),
                          "max_queries": max_queries, "seed": seed, "embedding_model": embedding_model},
               "datasets": {}}
    bm25_scores = {}
    for name in datasets:
        logger.info(f"BEIR {name}: loading")
        corpus, queries, qrels = load_beir(name)
        res = evaluate_rerank(corpus, queries, qrels, ranker_objs, first_stage_k, max_queries, seed, indexer)
        res["_size"] = {"docs": len(corpus), "queries_evaluated": res["_first_stage"]["bm25"]["ndcg@10"]["n"]}
        results["datasets"][name] = res
        bm25_scores[name] = res["_first_stage"]["bm25"]["ndcg@10"]["mean"]
        print(f"\n--- {name} ({res['_size']['queries_evaluated']} queries, {len(corpus):,} docs) ---")
        print(f"  {'bm25 (first stage)':<22}{res['_first_stage']['bm25']['ndcg@10']['mean'] * 100:6.1f}")
        for r in ranker_objs:
            print(f"  {r.name:<22}{res[r.name]['ndcg@10']['mean'] * 100:6.1f}")

    results["bm25_sanity"] = bm25_sanity(bm25_scores)
    results["protocol_mismatch"] = (first_stage_k != 100 or max_queries is not None
                                    or any(not v["ok"] for v in results["bm25_sanity"].values()))
    macro = {}
    for r in ranker_objs:
        vals = [results["datasets"][d][r.name]["ndcg@10"]["mean"] for d in datasets]
        macro[r.name] = float(np.mean(vals))
    results["macro_ndcg@10"] = macro
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "beir_rerank.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBM25 sanity: {results['bm25_sanity']}  protocol_mismatch={results['protocol_mismatch']}")
    print(f"Saved -> {out}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="BEIR top-k re-ranking")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--first-stage-k", type=int, default=100)
    ap.add_argument("--rankers", nargs="+", default=list(DEFAULT_RANKERS))
    ap.add_argument("--model", action="append", default=[], metavar="KEY=PATH",
                    help="MAPLE checkpoint for ranker 'maple:KEY' (repeatable)")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedding-model", default=BGE_MODEL_NAME)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    paths = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in a.model}
    from scripts.runlog import flatten_metrics, record_outputs, start_run
    with start_run("bench", "08_beir", vars(a)) as _run:
        _res = run(a.datasets, a.first_stage_k, a.rankers, a.max_queries, paths, a.seed, a.embedding_model, a.device)
        _run.log_metrics(flatten_metrics(_res))
        record_outputs(_run, RESULTS_DIR / "beir_rerank.json")
