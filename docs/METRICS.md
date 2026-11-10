# MAPLE — Evaluation metrics

MAPLE is an **experiment**. This page fixes, *before* any results are
collected, which numbers are needed to support or reject its hypothesis, how
each is computed, and which script produces it. Implementations live in
[`benchmarks/metrics.py`](../benchmarks/metrics.py) (unit-tested in
`tests/test_metrics.py`).

## 1. The hypothesis being tested

> A tiny MLP trained to imitate a teacher LLM's attention over context blocks
> selects context for long-document QA **better than standard retrievers
> (BM25, dense embeddings) at equal context budget**, and does so **fast
> enough** to scan very long documents.

That splits into four questions, each needing its own metrics:

| # | Question | Section |
|---|---|---|
| Q1 | Is teacher attention a good label at all? (does the teacher attend to the evidence?) | §3 |
| Q2 | Does MAPLE imitate the teacher? | §4 |
| Q3 | Does MAPLE find the *evidence* better than baselines — especially as documents get long? | §5, §6 |
| Q4 | Is it fast / cheap enough, and how much does approximate (hierarchical/adaptive) search lose? | §7 |
| Q5 | Does better retrieval turn into better answers? | §8 |

If Q1 fails (teacher attention is no better than dense retrieval at finding
evidence), a positive Q2 result is not interesting. Report Q1 first.

## 2. Conventions

* **Block**: 500 characters, no overlap (the oracle's chunking) unless stated.
* **Ranking**: list of block ids, best first. **k**: blocks returned.
* **Oracle set O**: the teacher's top-5 blocks (last-layer, head-mean attention
  of the final prompt token, summed per block).
* **Gold set G**: dataset evidence —
  * HotpotQA: blocks overlapping a supporting sentence (span-based).
  * NarrativeQA: blocks containing a normalised answer string. Answers are
    often abstractive, so G exists only for a subset; **always report coverage**
    (fraction of questions with non-empty G).
* **Uncertainty**: every mean is reported with a 95% percentile bootstrap CI
  (2,000 resamples over questions). Method comparisons use a paired bootstrap
  p-value over the same questions. Training-dependent numbers are reported as
  mean ± std over **3 seeds**.
* **Baselines** (always run alongside MAPLE): `random`, `lead` (document
  order), `bm25`, `dense` (cosine with the *same* encoder MAPLE uses), and the
  teacher's own attention (`oracle_last`, `oracle_all_layers`) as a reference.

## 3. Q1 — Label validity

Run on held-out oracle files (`benchmarks/01_recall_enhanced.py`), method
`oracle_last` / `oracle_all_layers` scored against **G**:

| Metric | Definition |
|---|---|
| `evidence_recall@5` of `oracle_last` | \|top-5 attention ∩ G\| / \|G\| |
| … of `oracle_all_layers` | same, attention averaged over all layers |
| … of `dense`, `bm25` | the bar the oracle must clear |
| Attention mass on context | fraction of last-token attention landing on context blocks (vs prompt / BOS sink) — the sum of a sample's saved `attention_scores` |

**Decision rule:** if `oracle_last` is not better than `dense` on
evidence_recall@5 (paired p < 0.05), switch the label (e.g. all-layer mean,
answer-gradient attribution, or gold evidence directly) before scaling up.

## 4. Q2 — Imitation quality (held-out oracle sets)

`benchmarks/01_recall_enhanced.py`, scored against **O**:

| Metric | Definition | Notes |
|---|---|---|
| `oracle_recall@k` (k=1,3,5,10) | \|top-k ∩ O\| / min(\|O\|, k) | v0.2.0's "Recall@K"; capped so a perfect ranking scores 1 |
| `oracle_ndcg@10` | nDCG with teacher attention mass as graded relevance | uses the full attention distribution, not just top-5 |
| Train-time `val_recall@5` | same as oracle_recall@5 on the trainer's validation split | diagnostic only — never reported as a result |

Always state the **candidate pool size** next to these numbers (≈10–12 blocks
for the 1,536-token oracle). On a 12-block pool, random already scores ≈ 42%
oracle_recall@5.

## 5. Q3a — Evidence retrieval on oracle-sized pools

Same script, scored against **G**:

| Metric | Definition |
|---|---|
| `evidence_recall@k` | \|top-k ∩ G\| / \|G\| |
| `all_evidence@k` | 1 if G ⊆ top-k (multi-hop needs *both* HotpotQA facts) |
| `evidence_mrr@10` | 1 / rank of the first gold block (0 if not in top-10) |
| `answer_in@k` | 1 if a normalised gold answer occurs in the concatenated top-k text |

## 6. Q3b — Long context (the claim that matters)

`benchmarks/07_long_context.py`, `benchmarks/03_needle_robust.py`,
`benchmarks/05_ablation_chunking.py`.

| Setting | Pool | Metrics |
|---|---|---|
| HotpotQA + injected distractor paragraphs | 10 → 100 → 1,000 → 5,000 blocks | evidence_recall@{1,5,10,20}, all_evidence@k, answer_in@k, MRR@10 — **plotted vs pool size** |
| NarrativeQA full stories (test split) | natural, 100s–1,000s of blocks | same, bucketed by story length; plus gold coverage |
| Needle-in-a-haystack, held-out templates | 10k → 500k chars | success@5 per (length × depth) cell, ≥10 trials/cell, CI |
| Chunk-size ablation at fixed context budget | 2,500 chars | evidence_recall@budget, all_evidence@budget, answer_in@budget vs chunk size |

Known limitation: random distractor paragraphs from other HotpotQA questions
are topically unrelated, so they are *easy* (dense recall barely drops with
pool size). A harder variant should draw distractors by BM25 similarity to the
question (HotpotQA "fullwiki"-style); report both once implemented.

## 7. Q4 — Efficiency and approximation cost

| Metric | Script | Definition |
|---|---|---|
| Search latency p50 / p95 | 02, 07 | ms per query, search only, per strategy, vs index size (10k → 1M blocks), real embeddings |
| End-to-end query latency | 07 | query encoding + search |
| Indexing throughput | 02 | blocks / s for encoding |
| Blocks scored per query | 07 | work done by linear / hierarchical / adaptive |
| **Approximation gap** | 07 | evidence_recall@k(adaptive or hierarchical) − evidence_recall@k(linear) — what the speed-up costs |
| Adaptive fallback rate | 07 | fraction of queries taking the wide path |
| Peak RAM / VRAM | profiler | during indexing, search, training |
| Model size / params | — | MapleNet checkpoint |
| Token reduction | 04 | full-document tokens ÷ (top-k tokens + query) |

Dollar costs in benchmark 04 use **illustrative** prices; only the token
reduction factor is a result.

## 8. Q5 — Downstream answers

`benchmarks/06_e2e_generation.py` (needs `GEMINI_API_KEY`). Contexts are
padded to `--haystack-chars` (default 50,000) so retrieval is actually
exercised.

| Metric | Definition |
|---|---|
| Exact match | SQuAD-normalised string equality with any reference |
| Token F1 | max SQuAD token-F1 over references |
| LLM-judge accuracy | separate judge model answers YES/NO; report the judge model name |
| `answer_in_context` | retrieved chunks contain a gold answer — separates retrieval failures from reading failures |
| Context tokens sent | per query, per method |
| Hops | agentic mode only |

Compare at the **same context budget**: full context (where it fits),
BM25-top-k, dense-top-k, MAPLE-top-k.

## 8b. Published-protocol benchmarks (08-11)

These follow a paper's protocol so our numbers can be placed next to published
ones. Each reproduces a published *baseline* first; if that check fails the
run is flagged and no comparison with the paper is made.

| Benchmark | Protocol | Metric | Sanity check (must pass to compare) |
|---|---|---|---|
| 08 BEIR re-rank | SciFact, NFCorpus, FiQA, TREC-COVID test queries; BM25 (bm25s, English stop-words) retrieves the top 100 from the full corpus; each method re-orders those 100 | nDCG@10 | our first-stage BM25 within ±2.0 of the published BM25 per dataset (ICR, ICLR 2025, Table 1) |
| 09 multi-hop re-rank | 1,000 seeded questions each from HippoRAG's released MuSiQue / 2Wiki / HotpotQA sets; corpus = union of their paragraphs; first stage takes the top 20; methods re-order the 20 | Recall@2, Recall@5, All-Recall@5 | first-stage Recall@5 within ±3.0 of ColBERTv2 in ICR Table 2 (a dense first stage never passes) |
| 10 reader-fixed QA | LongBench HotpotQA / NarrativeQA and NarrativeQA full stories; one fixed reader (Llama-3-8B-Instruct, greedy) gets each method's top blocks up to a 2,000-token budget, restored to document order | F1 (LongBench); ROUGE-L, BLEU-1/4, METEOR (NarrativeQA) | full-context (middle-truncated to 7,500 tokens, as LongBench does for 8k-context models) F1 within ±3.0 of InfLLM's "Original" Llama-3-8B row |
| 11 efficiency | every method on pools of 20-100k real 500-character blocks, GPU and CPU | p50 / p95 ms (query encoding + ranking), queries/s, peak VRAM/RAM | none (machine-specific; report the hardware) |

BM25 used as a *re-ranker* is fitted on the whole corpus (`BM25Ranker.fit`),
not on the candidate list: IDF computed over a top-k list is nearly flat,
because every candidate matches the query.

**Latency targets for the `rerank` strategy** (search only, query already
encoded, real embeddings): ≤ 20 ms p50 at 1M blocks on a GPU, ≤ 50 ms p50 at
100k blocks on a CPU.

## 9. What to report (minimum table set)

1. Q1 table: evidence_recall@5 of oracle_last / oracle_all_layers vs dense / BM25 (both datasets).
2. Q2 table: oracle_recall@{1,5}, oracle_ndcg@10 — MAPLE vs baselines, with pool size.
3. Q3 figure: evidence_recall@5 vs pool size (HotpotQA), and NarrativeQA by story-length bucket.
4. Needle heatmaps per method.
5. Latency vs index size per strategy, with the approximation gap.
6. E2E: EM / F1 / judge / answer_in_context per method at a fixed budget.

Every table states: dataset split, number of questions, candidate pool size,
encoder, MAPLE checkpoint (+ training seed), and the git commit.

## 10. Numbers that must not be reused

The v0.2.0 figures (">95% recall", "71% Recall@5", "98% Recall@1", "98%
SQuAD", ">90% needle") were produced by code with known bugs — unreachable
trailing blocks in hierarchical search, mislabelled training block ids, needle
templates shared between training and test, padding that hid the search bug,
and QA contexts too short to exercise retrieval — and cannot be reproduced from
this repository. They are kept only in `docs/archive/` for history.
