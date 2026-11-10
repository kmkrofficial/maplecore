#!/usr/bin/env python3
"""
Benchmark 09: multi-hop passage re-ranking (the protocol of ICR, Table 2/3)
==========================================================================

Data: HippoRAG's released sets (``osunlp/HippoRAG_2``): 1,000 questions each from MuSiQue, 2WikiMultiHopQA and
HotpotQA with a shared corpus per dataset (6k-12k passages; the union of the questions' paragraphs).
A first stage retrieves the top-20 passages from the whole corpus, every ranker re-orders those 20, and
we report Recall@2, Recall@5 and All-Recall@5 over the supporting passages (ICR, Chen et al., ICLR 2025).

First stage: ColBERTv2 in the paper. Here ``dense`` (BGE cosine) by default, or ColBERTv2 candidate lists produced
elsewhere (``--first-stage-file``, e.g. from WSL with ragatouille). The protocol counts as reproduced
(``protocol_ok``) only with ColBERTv2 AND a first-stage Recall@5 within +/-3.0 of ICR's ColBERTv2 row
(MuSiQue 49.2, 2Wiki 68.2, HotpotQA 79.3).

Usage:
    python benchmarks/09_multihop_rerank.py --rankers bm25 dense hybrid cross_encoder maple:main --model main=models/x.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics as M
from benchmarks.config import BGE_MODEL_NAME, RESULTS_DIR
from benchmarks.rankers import build_rankers, fit_rankers

logger = logging.getLogger(__name__)

DATASETS = {"musique": "musique", "2wiki": "2wikimultihopqa", "hotpotqa": "hotpotqa"}  # short key -> file stem
# ColBERTv2 Recall@5 x100 from ICR Table 2 (the first-stage row)
PUBLISHED_FIRST_STAGE_R5 = {"musique": 49.2, "2wiki": 68.2, "hotpotqa": 79.3}
FIRST_STAGE_TOLERANCE = 3.0
DEFAULT_RANKERS = ("bm25", "dense", "hybrid", "cross_encoder", "maple:main")


def build_corpus(questions: Sequence[dict]) -> List[dict]:
    """Union of all paragraphs of the questions, deduplicated by (title, text)."""
    seen, corpus = set(), []
    for q in questions:
        for p in q["paragraphs"]:
            key = (p["title"], p["text"])
            if key not in seen:
                seen.add(key)
                corpus.append({"title": p["title"], "text": p["text"]})
    return corpus


def score_reranked(gold: set, ranking: Sequence) -> Dict[str, float]:
    """Recall@2, Recall@5 and All-Recall@5 of a ranking over the gold (supporting) passages."""
    return {"recall@2": M.recall_at_k(ranking, gold, 2), "recall@5": M.recall_at_k(ranking, gold, 5),
            "all_recall@5": M.hit_all_at_k(ranking, gold, 5)}


def first_stage_sanity(first_stage_recall5: Dict[str, float], used: str) -> Dict[str, dict]:
    """Compare first-stage Recall@5 (fractions) with ICR's ColBERTv2 row; only ColBERTv2 can reproduce the protocol."""
    out = {}
    for name, value in first_stage_recall5.items():
        if name in PUBLISHED_FIRST_STAGE_R5:
            pct = round(value * 100, 1)
            out[name] = {"ours": pct, "published": PUBLISHED_FIRST_STAGE_R5[name], "first_stage": used,
                         "ok": used == "colbertv2" and abs(pct - PUBLISHED_FIRST_STAGE_R5[name]) <= FIRST_STAGE_TOLERANCE}
    return out


def sample_questions(questions: Sequence[dict], n: int, seed: int = 0) -> List[dict]:
    """Seeded subsample (keeps the original order)."""
    if n >= len(questions):
        return list(questions)
    idx = sorted(random.Random(seed).sample(range(len(questions)), n))
    return [questions[i] for i in idx]


def load_hipporag(stem: str):
    """Questions with gold corpus indices, and the corpus, from osunlp/HippoRAG_2."""
    from huggingface_hub import hf_hub_download

    def get(name):
        return json.loads(Path(hf_hub_download("osunlp/HippoRAG_2", name, repo_type="dataset")).read_text(encoding="utf-8"))

    raw, corpus = get(f"{stem}.json"), get(f"{stem}_corpus.json")
    by_title = {}
    by_key = {}
    for i, p in enumerate(corpus):
        by_title.setdefault(p["title"], i)
        by_key[(p["title"], p["text"])] = i
    questions = []
    for q in raw:
        if stem == "musique":
            gold = {by_key[(p["title"], p["paragraph_text"])] for p in q["paragraphs"] if p["is_supporting"]}
            qid = q["id"]
        else:  # hotpotqa, 2wikimultihopqa: supporting facts name the supporting paragraphs by title
            gold = {by_title[t] for t, _ in q["supporting_facts"]}
            qid = q["_id"]
        questions.append({"id": qid, "question": q["question"], "answer": q["answer"], "gold": gold})
    return questions, corpus


def evaluate(questions: Sequence[dict], corpus: Sequence[dict], rankers: Sequence, indexer,
             first_stage_k: int = 20, first_stage_file: Optional[Dict[str, List[int]]] = None) -> dict:
    texts = [f"{p['title']}\n{p['text']}" for p in corpus]
    fit_rankers(rankers, texts)  # BM25 / hybrid use corpus-level IDF
    emb = indexer.encode_texts(texts, show_progress=True)
    emb_n = torch.nn.functional.normalize(emb.float(), dim=1)
    per: Dict[str, Dict[str, list]] = {"first_stage": {}}
    for r in rankers:
        per[r.name] = {}
    for n, q in enumerate(questions):
        q_emb = indexer.encode_query(q["question"])
        if first_stage_file is not None:
            cand = [int(i) for i in first_stage_file[q["id"]]][:first_stage_k]
        else:
            sims = emb_n @ torch.nn.functional.normalize(q_emb.float().to(emb_n.device), dim=0)
            cand = torch.topk(sims, first_stage_k).indices.tolist()
        cand_texts = [texts[i] for i in cand]
        cand_emb = emb[cand]
        rows = {"first_stage": cand}
        for r in rankers:
            rows[r.name] = [cand[i] for i in r.rank(q["question"], cand_texts, q_emb, cand_emb)]
        for name, ranking in rows.items():
            for metric, v in score_reranked(q["gold"], ranking).items():
                per[name].setdefault(metric, []).append(v)
        if (n + 1) % 250 == 0:
            logger.info(f"  {n + 1}/{len(questions)}")
    return {name: {metric: M.summarize(vals) for metric, vals in d.items()} for name, d in per.items()}


def run(datasets: Sequence[str] = tuple(DATASETS), n_queries: int = 1000, first_stage: str = "dense",
        first_stage_file: Optional[Path] = None, first_stage_k: int = 20, rankers: Sequence[str] = DEFAULT_RANKERS,
        model_paths: Optional[Dict[str, Path]] = None, seed: int = 0, embedding_model: str = BGE_MODEL_NAME,
        device: Optional[str] = None) -> dict:
    from maplecore import MapleIndexer

    used = "colbertv2" if first_stage_file else first_stage
    fs_lists = json.loads(Path(first_stage_file).read_text(encoding="utf-8")) if first_stage_file else None
    ranker_objs = build_rankers(list(rankers), model_paths or {})
    indexer = MapleIndexer(model_name=embedding_model, device=device)
    results = {"config": {"datasets": list(datasets), "n_queries": n_queries, "first_stage": used,
                          "first_stage_k": first_stage_k, "rankers": list(rankers), "seed": seed,
                          "embedding_model": embedding_model}, "datasets": {}}
    first_r5 = {}
    for key in datasets:
        logger.info(f"{key}: loading")
        questions, corpus = load_hipporag(DATASETS[key])
        questions = sample_questions(questions, n_queries, seed)
        fs = None if fs_lists is None else fs_lists[key]
        res = evaluate(questions, corpus, ranker_objs, indexer, first_stage_k, fs)
        res["_size"] = {"questions": len(questions), "corpus": len(corpus)}
        results["datasets"][key] = res
        first_r5[key] = res["first_stage"]["recall@5"]["mean"]
        print(f"\n--- {key} ({len(questions)} questions, {len(corpus):,} passages, first stage: {used}) ---")
        print(f"  {'method':<22}{'R@2':>8}{'R@5':>8}{'All-R@5':>9}")
        for name in ["first_stage"] + [r.name for r in ranker_objs]:
            s = res[name]
            print(f"  {name:<22}{s['recall@2']['mean'] * 100:8.1f}{s['recall@5']['mean'] * 100:8.1f}{s['all_recall@5']['mean'] * 100:9.1f}")
    results["first_stage_sanity"] = first_stage_sanity(first_r5, used)
    results["protocol_ok"] = all(v["ok"] for v in results["first_stage_sanity"].values()) and n_queries >= 1000
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "multihop_rerank.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nfirst-stage sanity: {results['first_stage_sanity']}\nprotocol_ok={results['protocol_ok']}\nSaved -> {out}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="ICR multi-hop re-ranking protocol")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--first-stage", default="dense", choices=["dense"])
    ap.add_argument("--first-stage-file", type=Path, default=None,
                    help="JSON {dataset: {question id: [corpus indices]}} of ColBERTv2 top-k candidates")
    ap.add_argument("--first-stage-k", type=int, default=20)
    ap.add_argument("--rankers", nargs="+", default=list(DEFAULT_RANKERS))
    ap.add_argument("--model", action="append", default=[], metavar="KEY=PATH")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedding-model", default=BGE_MODEL_NAME)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    paths = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in a.model}
    from scripts.runlog import flatten_metrics, record_outputs, start_run
    with start_run("bench", "09_multihop", vars(a)) as _run:
        _res = run(a.datasets, a.n_queries, a.first_stage, a.first_stage_file, a.first_stage_k, a.rankers, paths,
                   a.seed, a.embedding_model, a.device)
        _run.log_metrics(flatten_metrics(_res))
        record_outputs(_run, RESULTS_DIR / "multihop_rerank.json")
