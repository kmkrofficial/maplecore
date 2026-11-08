#!/usr/bin/env python3
"""
Benchmark 10: reader-fixed end-to-end QA (LongBench + NarrativeQA full story)
============================================================================

One reader (Llama-3-8B-Instruct, 4-bit, greedy) answers questions from context chosen by different methods, so
differences come from retrieval alone.

* ``longbench_hotpotqa`` / ``longbench_narrativeqa``: official LongBench v1 prompts and generation lengths
  (benchmarks/longbench_prompts.json), metric = LongBench qa_f1. ``full`` = the whole context, middle-truncated to
  7,500 tokens (the "Original" row of InfLLM's table); other methods split the context into 500-character blocks,
  rank them, and keep top blocks up to ``budget_tokens`` (restored to document order).
* ``narrativeqa_fullstory``: NarrativeQA test questions over the *whole story* (100-word chunks, as in Izacard & Grave
  2021 and RAPTOR 2024), ROUGE-L / BLEU-1 / BLEU-4 / METEOR.

Every generated answer is saved with the retrieved block ids (``run.save_items``) for later analysis.
Sanity: the ``full`` LongBench F1 must be within +/-3 of InfLLM's Llama-3-8B-Instruct "Original" (HotpotQA 47.38,
NarrativeQA 19.85), else ``protocol_ok`` is False.

Usage:
    python benchmarks/10_e2e_longbench.py --tasks longbench_hotpotqa --rankers dense --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics as M
from benchmarks.config import BGE_MODEL_NAME, RESULTS_DIR
from benchmarks.rankers import build_rankers

logger = logging.getLogger(__name__)

PROMPTS_FILE = Path(__file__).with_name("longbench_prompts.json")
LONGBENCH_MAX_TOKENS = 7500
PUBLISHED_ORIGINAL_F1 = {"longbench_hotpotqa": 47.38, "longbench_narrativeqa": 19.85}  # InfLLM Table 5, Llama-3-8B
ORIGINAL_TOLERANCE = 3.0
TASKS = ("longbench_hotpotqa", "longbench_narrativeqa", "narrativeqa_fullstory")
DEFAULT_RANKERS = ("bm25", "dense", "hybrid", "cross_encoder", "maple:main")


# ------------------------------------------------------------------ helpers (unit-tested)

def middle_truncate(ids: Sequence[int], max_len: int) -> List[int]:
    """LongBench truncation: keep the first and last ``max_len // 2`` tokens."""
    ids = list(ids)
    if len(ids) <= max_len:
        return ids
    half = max_len // 2
    return ids[:half] + ids[-half:]


def budget_context(blocks: Sequence[str], ranking: Sequence[int], budget_tokens: int,
                   count_tokens: Callable[[str], int]) -> str:
    """Take blocks in ranked order until the token budget is full (the best block is always kept), then join in
    document order."""
    chosen, used = [], 0
    for i in ranking:
        n = count_tokens(blocks[i])
        if chosen and used + n > budget_tokens:
            break
        chosen.append(i)
        used += n
    return "\n".join(blocks[i] for i in sorted(chosen))


def chunk_words(text: str, n: int = 100) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def nqa_generation_metrics(preds: Sequence[str], refs: Sequence[Sequence[str]]) -> Dict[str, Optional[float]]:
    """ROUGE-L (F), BLEU-1, BLEU-4 (corpus, smoothed) and METEOR, all x100. METEOR is None without NLTK's WordNet."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = [max(scorer.score(r, p)["rougeL"].fmeasure for r in rs) for p, rs in zip(preds, refs)]
    hyp = [p.lower().split() for p in preds]
    ref = [[r.lower().split() for r in rs] for rs in refs]
    smooth = SmoothingFunction().method1
    out: Dict[str, Optional[float]] = {
        "rouge_l": 100 * float(np.mean(rouge)),
        "bleu_1": 100 * corpus_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "bleu_4": 100 * corpus_bleu(ref, hyp, weights=(0.25,) * 4, smoothing_function=smooth),
    }
    try:
        from nltk.translate.meteor_score import meteor_score
        out["meteor"] = 100 * float(np.mean([meteor_score(r, h) for r, h in zip(ref, hyp)]))
    except LookupError:
        out["meteor"] = None
    return out


# ------------------------------------------------------------------------- data

def load_longbench(task: str, limit: Optional[int] = None) -> List[dict]:
    from huggingface_hub import hf_hub_download

    stem = task.replace("longbench_", "")
    with zipfile.ZipFile(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")) as z:
        rows = [json.loads(line) for line in z.read(f"data/{stem}.jsonl").decode("utf-8").splitlines() if line.strip()]
    items = [{"id": f"{stem}-{i}", "question": r["input"], "context": r["context"], "answers": r["answers"],
              "doc": f"{stem}-{i}"} for i, r in enumerate(rows)]
    return items[:limit] if limit else items


def load_narrativeqa_fullstory(n: int, seed: int) -> List[dict]:
    from datasets import load_dataset

    ds = load_dataset("deepmind/narrativeqa", split="test")
    idx = sorted(random.Random(seed).sample(range(len(ds)), min(n, len(ds))))
    out = []
    for i in idx:
        r = ds[i]
        out.append({"id": f"nqa-test-{i}", "question": r["question"]["text"], "context": r["document"]["text"],
                    "answers": [a["text"] for a in r["answers"]], "doc": r["document"]["id"]})
    return out


# ----------------------------------------------------------------------- reader

class Reader:
    """Llama-3-8B-Instruct (4-bit), greedy decoding, chat template, LongBench middle truncation."""

    def __init__(self, model_id: Optional[str] = None):
        from scripts.oracle_common import DEFAULT_TEACHER, load_teacher

        self.model, self.tok = load_teacher(model_id or DEFAULT_TEACHER, attn_impl="fast")
        self.tok.padding_side = "left"
        self.eos = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|eot_id|>")]

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def _chat(self, prompt: str, max_len: int) -> str:
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) > max_len:
            ids = middle_truncate(ids, max_len)
            prompt = self.tok.decode(ids, skip_special_tokens=True)
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    def generate(self, prompts: Sequence[str], max_new_tokens: int, batch_size: int = 4,
                 max_len: int = LONGBENCH_MAX_TOKENS) -> List[str]:
        import torch

        texts = [self._chat(p, max_len) for p in prompts]
        lens = [len(self.tok.encode(t, add_special_tokens=False)) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: -lens[i])
        out = [""] * len(texts)
        s = 0
        while s < len(order):
            # keep the (batch x L x L x heads) attention working set around 3 GB: long prompts run one at a time
            b = max(1, min(batch_size, int(3e9 // (64 * lens[order[s]] ** 2))))
            idx = order[s:s + b]
            s += b
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          eos_token_id=self.eos, pad_token_id=self.tok.eos_token_id)
            for j, i in enumerate(idx):
                out[i] = self.tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return out


# -------------------------------------------------------------------------- run

def _prompt_for(task: str, prompts: dict, question: str, context: str) -> str:
    if task.startswith("longbench_"):
        return prompts["dataset2prompt"][task.replace("longbench_", "")].format(input=question, context=context)
    return ("You are given excerpts of a story and a question. Answer the question as concisely as you can, "
            "using a single phrase if possible. Do not provide any explanation.\n\n"
            f"Story excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:")


def run(tasks: Sequence[str] = TASKS, rankers: Sequence[str] = DEFAULT_RANKERS, budget_tokens: int = 2000,
        n_narrativeqa: int = 1000, limit: Optional[int] = None, seed: int = 0,
        model_paths: Optional[Dict[str, Path]] = None, reader_model: Optional[str] = None,
        embedding_model: str = BGE_MODEL_NAME, device: Optional[str] = None, batch_size: int = 4) -> dict:
    from maplecore import MapleIndexer
    from scripts.runlog import current_run

    prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    ranker_objs = build_rankers(list(rankers), model_paths or {})
    indexer = MapleIndexer(model_name=embedding_model, device=device)
    reader = Reader(reader_model)
    run_ctx = current_run()
    results = {"config": {"tasks": list(tasks), "rankers": list(rankers), "budget_tokens": budget_tokens,
                          "seed": seed, "reader": reader_model or "meta-llama/Meta-Llama-3-8B-Instruct"}, "tasks": {}}

    for task in tasks:
        if task.startswith("longbench_"):
            items = load_longbench(task, limit)
            chunker = lambda t: indexer.chunk_text(t, 500)  # noqa: E731
            blocks_of = lambda ctx: [b.text for b in chunker(ctx)]  # noqa: E731
            max_new = prompts["dataset2maxlen"][task.replace("longbench_", "")]
        else:
            items = load_narrativeqa_fullstory(limit or n_narrativeqa, seed)
            blocks_of = lambda ctx: chunk_words(ctx, 100)  # noqa: E731
            max_new = 64
        logger.info(f"{task}: {len(items)} questions")

        method_prompts: Dict[str, List[str]] = {}
        method_meta: Dict[str, List[dict]] = {}
        if task.startswith("longbench_"):
            method_prompts["full"] = [_prompt_for(task, prompts, it["question"], it["context"]) for it in items]
            method_meta["full"] = [{"context_tokens": None, "blocks": None} for _ in items]
        emb_cache: Dict[str, tuple] = {}
        for it in items:
            if it["doc"] not in emb_cache:
                blocks = blocks_of(it["context"])
                emb_cache = {it["doc"]: (blocks, indexer.encode_texts(blocks))}  # one document at a time (items grouped by doc)
            it["_blocks"], it["_emb"] = emb_cache[it["doc"]]
            it["_q"] = indexer.encode_query(it["question"])
        for r in ranker_objs:
            method_prompts[r.name], method_meta[r.name] = [], []
            for it in items:
                ranking = r.rank(it["question"], it["_blocks"], it["_q"], it["_emb"])
                ctx = budget_context(it["_blocks"], ranking, budget_tokens, reader.count_tokens)
                method_prompts[r.name].append(_prompt_for(task, prompts, it["question"], ctx))
                method_meta[r.name].append({"context_tokens": reader.count_tokens(ctx), "blocks": ranking[:20],
                                            "retrieved_has_answer": M.answer_in_context(ctx, it["answers"], min_len=3)})

        task_res = {}
        for method, plist in method_prompts.items():
            preds = reader.generate(plist, max_new, batch_size)
            if task.startswith("longbench_"):
                f1 = [M.token_f1(p, it["answers"]) for p, it in zip(preds, items)]
                metric = {"f1": M.summarize(f1)}
            else:
                metric = {k: v for k, v in nqa_generation_metrics(preds, [it["answers"] for it in items]).items()}
            has = [m.get("retrieved_has_answer") for m in method_meta[method] if m.get("retrieved_has_answer") is not None]
            toks = [m["context_tokens"] for m in method_meta[method] if m["context_tokens"] is not None]
            task_res[method] = {"metrics": metric, "answer_in_context": float(np.mean(has)) if has else None,
                                "mean_context_tokens": float(np.mean(toks)) if toks else None}
            print(f"  {task:<24}{method:<24}{json.dumps({k: (round(v['mean'] * 100, 1) if isinstance(v, dict) else v) for k, v in metric.items()})}")
            if run_ctx is not None:
                run_ctx.save_items(f"{task}__{method.replace(':', '_')}", (
                    {"id": it["id"], "method": method, "prediction": p, "references": it["answers"], **meta}
                    for it, p, meta in zip(items, preds, method_meta[method])))
        results["tasks"][task] = task_res

    sanity = {}
    for task, published in PUBLISHED_ORIGINAL_F1.items():
        if task in results["tasks"] and "full" in results["tasks"][task]:
            ours = results["tasks"][task]["full"]["metrics"]["f1"]["mean"] * 100
            sanity[task] = {"ours": round(ours, 1), "published": published, "ok": abs(ours - published) <= ORIGINAL_TOLERANCE}
    results["original_sanity"] = sanity
    results["protocol_ok"] = bool(sanity) and all(v["ok"] for v in sanity.values())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "e2e_reader_fixed.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nfull-context sanity vs InfLLM 'Original': {sanity}  protocol_ok={results['protocol_ok']}\nSaved -> {out}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Reader-fixed end-to-end QA benchmark")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--rankers", nargs="+", default=list(DEFAULT_RANKERS))
    ap.add_argument("--model", action="append", default=[], metavar="KEY=PATH")
    ap.add_argument("--budget-tokens", type=int, default=2000)
    ap.add_argument("--n-narrativeqa", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="Questions per task (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reader-model", default=None)
    ap.add_argument("--embedding-model", default=BGE_MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    paths = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in a.model}
    from scripts.runlog import record_outputs, start_run
    with start_run("bench", "10_e2e", vars(a)) as _run:
        run(a.tasks, a.rankers, a.budget_tokens, a.n_narrativeqa, a.limit, a.seed, paths, a.reader_model,
            a.embedding_model, a.device, a.batch_size)
        record_outputs(_run, RESULTS_DIR / "e2e_reader_fixed.json")
