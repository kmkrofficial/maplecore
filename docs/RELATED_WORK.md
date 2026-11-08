# MAPLE — Related work

Compiled 2026-11-08 from a web search. Venue/year from the papers' own pages;
numbers quoted are the authors' claims unless stated otherwise.

MAPLE's recipe: label context blocks by a decoder LLM's attention while it
answers a question → train a tiny MLP on frozen sentence embeddings to predict
those labels → use the MLP to pick context for long documents, before the LLM
sees them. Every ingredient has close prior work; the combination (external,
model-agnostic, decoder self-attention labels, tiny scorer over frozen
embeddings, long-document block selection) is what would be new.

## 1. Attention of a reader model as supervision for a retriever (closest prior work)

| Paper | What it does | Relation to MAPLE |
|---|---|---|
| **Izacard & Grave, *Distilling Knowledge from Reader to Retriever for QA*** (ICLR 2021) [arXiv 2012.04584](https://arxiv.org/abs/2012.04584) | Uses a Fusion-in-Decoder reader's **cross-attention** over retrieved passages as synthetic labels to train a dense retriever; reader and retriever are trained in alternation. **KL divergence** between normalised scores worked best. | Same core idea, years earlier. Differences: they fine-tune the full bi-encoder (initialised from a retriever), use a soft-distribution KL loss, and work on passage retrieval, not within-document blocks. MAPLE must cite and compare to this. |
| **Atlas** (Izacard et al., JMLR 2023) [arXiv 2208.03299](https://arxiv.org/abs/2208.03299) | Compares retriever-training losses incl. **attention distillation (ADist)** and **perplexity distillation (PDist)**. | Shows attention is one of several comparable LM-derived signals; perplexity-based labels are a direct alternative to MAPLE's attention labels. |
| **REPLUG LSR** (Shi et al., NAACL 2024) [arXiv 2301.12652](https://arxiv.org/abs/2301.12652) | Trains a retriever so that retrieved documents lower a **black-box** LM's perplexity. | Label from LM likelihood rather than attention; works with closed models (attention does not). |

## 2. Attention as a zero-shot relevance signal (no student model)

| Paper | Finding | Implication for MAPLE |
|---|---|---|
| **Retrieval Head** (Wu et al., ICLR 2025 oral) [arXiv 2404.15574](https://arxiv.org/abs/2404.15574) | Fewer than 5% of heads do long-context retrieval; masking them breaks retrieval. | Averaging **all** heads (MAPLE's label) dilutes the signal with ~95% non-retrieval heads. |
| **QRHead** (Zhang et al., EMNLP 2025) [arXiv 2506.09944](https://arxiv.org/abs/2506.09944) | Selects query-focused heads with a few examples; summing their attention per chunk retrieves better than strong dense retrievers and re-ranks better than RankGPT. | Strongest evidence that attention *can* beat dense retrieval — but only from **selected heads**. A natural label upgrade and a baseline MAPLE should include. |
| **ICR — *Attention in LLMs Yields Efficient Zero-Shot Re-Rankers*** (Chen et al., ICLR 2025) [arXiv 2410.02642](https://arxiv.org/abs/2410.02642) | Re-ranks with query-induced attention change; **calibrates with a content-free query** to remove intrinsic bias; beats RankGPT with >60% lower latency. | Calibration directly targets the attention-sink / position bias MAPLE observes (only ~10–12% of attention lands on context). |
| **AttentionRAG** (Fang et al., 2025) [arXiv 2503.10720](https://arxiv.org/abs/2503.10720) | Prunes RAG context using **intermediate-layer** attention from a single answer-hint token; up to 6.3× compression. | Again: middle layers, not the last layer, carry the useful relevance signal. |

## 3. Learned predictors of attention for efficient long-context inference

These predict *which blocks/tokens the model will attend to* — MAPLE's goal —
but operate **inside** the LLM on its own queries/keys (KV-cache selection or
sparse attention), so the LLM must still process (or cache) the whole context.

| Paper | Mechanism |
|---|---|
| **SeerAttention** (Gao et al., NeurIPS 2025) [arXiv 2410.13276](https://arxiv.org/abs/2410.13276) | Learnable gate on pooled Q/K predicts block-level attention; **distilled from the model's pooled attention map** (≈40 A100-h for Llama-3.1-8B). Conceptually the closest "learn to predict attended blocks" method. |
| **AttentionPredictor** (Yang et al., NeurIPS 2025) [arXiv 2502.04077](https://arxiv.org/abs/2502.04077) | Lightweight CNN forecasts next-step attention to prefetch KV cache; 13× compression claimed. |
| **Quest** (Tang et al., ICML 2024) [arXiv 2406.10774](https://arxiv.org/abs/2406.10774) | Per-page min/max key bounds estimate page criticality for the current query. |
| **InfLLM** (Xiao et al., NeurIPS 2024) [arXiv 2402.04617](https://arxiv.org/abs/2402.04617) | Training-free block memory; each block represented by a few representative tokens; blocks retrieved per step (cf. MAPLE's chapter embeddings). |
| **Landmark Attention** (Mohtashami & Jaggi, NeurIPS 2023) [arXiv 2305.16300](https://arxiv.org/abs/2305.16300) | A landmark token per block; attention itself learns block selection. |
| **NSA** (DeepSeek, 2025) [arXiv 2502.11089](https://arxiv.org/abs/2502.11089) · **MoBA** (Moonshot, 2025) [arXiv 2502.13189](https://arxiv.org/abs/2502.13189) | Natively trained block-sparse attention with learned/derived block importance. |
| **COBS** (2026) [arXiv 2607.09052](https://arxiv.org/abs/2607.09052) | Frames block selection as estimating per-block attention mass from a compact summary; an exact-mass oracle essentially matches dense attention. |

## 4. Long-document retrieval structure and context pruning

| Paper | Relation |
|---|---|
| **RAPTOR** (Sarthi et al., ICLR 2024) [arXiv 2401.18059](https://arxiv.org/abs/2401.18059) | Tree of recursive summaries for long-document retrieval — a hierarchical alternative to MAPLE's mean-pooled chapters. |
| **Provence** (Chirkova et al., 2025) [arXiv 2501.16214](https://arxiv.org/abs/2501.16214) | Trained context pruner unified with re-ranking — a strong learned baseline for "which context to keep". |

## 5. What this means for MAPLE

**Where MAPLE sits.** Unlike section 3, MAPLE runs *outside* the LLM on cheap
sentence embeddings, so it can choose context before the LLM reads anything
(a RAG-style setting, compatible with closed LLMs at inference). Unlike section
1, it uses decoder self-attention labels and a tiny scorer instead of
fine-tuning a retriever. That niche is real, but the prior work predicts the
problems seen in our first runs:

| Our observation (held-out, budget 1,536, seed 42) | Explained by |
|---|---|
| Last-layer head-mean attention finds HotpotQA evidence *worse* than dense (74.7% vs 82.8% evidence recall@5) | Retrieval Head, QRHead: only a few heads retrieve; head-mean dilutes them |
| All-layer mean beats dense (85.9%) | AttentionRAG, QRHead: useful signal is in intermediate layers / specific heads |
| Only ~10–12% of final-token attention lands on context | ICR: intrinsic attention bias needs calibration |
| MAPLE imitates the teacher well (74.9% vs 54.6% oracle recall) but trails dense on evidence (63.7%) | Izacard & Grave fine-tune a retriever initialised from a good one, with a KL loss; an MLP on concatenated embeddings starts from scratch with no similarity prior |

**Changes the literature suggests (in priority order):**
1. **Better label** — all-layer mean (already in the data), QRHead-style selected heads, and ICR-style calibration with a content-free query.
2. **Soft target + KL loss** over each question's blocks (Izacard & Grave) instead of BCE on a binary top-5.
3. **Similarity prior in the student** — interaction features (q⊙b, |q−b|), a learned residual on top of cosine, or fine-tuning the BGE encoder itself.
4. **Stronger baselines** — QRHead/ICR (attention without a student), a cross-encoder re-ranker, and a Izacard & Grave–style distilled dense retriever.

If MAPLE still cannot beat dense retrieval after (1)–(3), the publishable
finding is a careful negative result: a tiny external scorer distilled from
decoder attention does not outperform a good embedding model, and attention's
value as a retrieval signal depends on head/layer selection.
