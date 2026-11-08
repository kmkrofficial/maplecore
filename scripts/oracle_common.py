"""
Shared oracle-generation utilities
==================================

Used by ``generate_oracle.py`` (NarrativeQA) and ``generate_oracle_hotpot.py``
(HotpotQA) so both datasets are labelled with the *same* teacher, prompt and
attention aggregation, and saved in the *same* schema.

Oracle label definition (unchanged from v0.2.0):
    Teacher LLM reads ``[prompt][block_0]\\n[block_1]\\n...[question] Answer:``.
    For the final token, take the attention distribution of the LAST layer,
    average over heads, and sum it over each block's tokens. The top-5 blocks
    are the positive labels.

Additionally recorded (new, for validating the label — see docs/METRICS.md):
    * ``attention_scores_all_layers``: same aggregation, averaged over ALL layers
    * ``gold_block_ids``: blocks that contain the dataset's evidence
      (HotpotQA supporting sentences / NarrativeQA answer strings)

Two interchangeable ways to read the final token's attention:

* ``attn_impl="fast"`` (default) - the teacher runs on PyTorch SDPA kernels
  (never materialising ``[heads, L, L]``). ``LastRowCapture`` wraps the SDPA
  attention function and, from the post-RoPE query/key it receives, computes
  only ``softmax(q_last . K^T . scale [+ mask])`` per layer. Samples are batched
  with left padding (padded keys masked; RoPE is relative so positions are
  shift-invariant) and the LM head is skipped.
* ``attn_impl="eager"`` - the v0.2.0-style reference: eager attention plus a
  forward hook keeping the last row of the full weight matrix. Batch size 1.

``tests/test_oracle_fast.py`` checks the two agree on a random tiny Llama, and
``docs/DATA.md`` records the agreement measured on the real teacher.
"""

from __future__ import annotations

import json
import logging
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

DEFAULT_TEACHER = "meta-llama/Meta-Llama-3-8B-Instruct"
SCHEMA_VERSION = 2
PROMPT_PREFIX = "Read the following context carefully and answer the question.\n\nContext:\n"


# ===================================================================
# Teacher model
# ===================================================================

def load_teacher(model_id: str = DEFAULT_TEACHER, load_in_4bit: bool = True, attn_impl: str = "fast"):
    """Load the teacher. ``attn_impl``: "fast" (SDPA + LastRowCapture) or "eager"."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if attn_impl not in ("fast", "eager"):
        raise ValueError("attn_impl must be 'fast' or 'eager'")
    kwargs = dict(attn_implementation="sdpa" if attn_impl == "fast" else "eager", dtype=torch.float16)
    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif not torch.cuda.is_available():
        kwargs["dtype"] = torch.float32

    logger.info(f"Loading teacher {model_id} (4-bit={load_in_4bit and torch.cuda.is_available()}, attn={attn_impl})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if torch.cuda.is_available() and "device_map" not in kwargs:
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


class LastRowAttentionRecorder:
    """
    Forward hooks that keep only the final query row of each layer's attention.

    Works with HF attention modules that return ``(attn_output, attn_weights, ...)``
    when ``attn_implementation="eager"``.
    """

    def __init__(self, model):
        self.rows: List[torch.Tensor] = []
        self._handles = []
        for name, module in model.named_modules():
            if name.endswith("self_attn"):
                self._handles.append(module.register_forward_hook(self._hook))
        if not self._handles:
            raise RuntimeError("No '*.self_attn' modules found to hook.")

    def _hook(self, module, args, output):
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if weights is None:
            raise RuntimeError(
                "Attention module returned no weights. Load the model with "
                "attn_implementation='eager'."
            )
        # weights: [batch, heads, q_len, k_len] -> mean over heads of last row
        self.rows.append(weights[0, :, -1, :].float().mean(dim=0).cpu())

    def reset(self):
        self.rows = []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class LastRowCapture:
    """
    Context manager: wrap the SDPA attention function so every layer also
    records the head-averaged attention row of the LAST query position.

    ``rows`` gets one ``[batch, key_len]`` float32 CPU tensor per layer, or with
    ``keep_heads=True`` one ``[batch, heads, key_len]`` tensor (no head averaging).
    """

    def __init__(self, keep_heads: bool = False):
        self.rows: List[torch.Tensor] = []
        self.keep_heads = keep_heads
        self._orig = None

    def _wrapper(self, module, query, key, value, attention_mask, *args, **kwargs):
        out = self._orig(module, query, key, value, attention_mask, *args, **kwargs)
        with torch.no_grad():
            q = query[:, :, -1:, :].float()                       # [B, Hq, 1, d]
            k = key.float()                                        # [B, Hkv, L, d]
            groups = q.shape[1] // k.shape[1]
            if groups > 1:
                k = k.repeat_interleave(groups, dim=1)             # same as HF repeat_kv
            scaling = kwargs.get("scaling")
            scale = scaling if scaling is not None else q.shape[-1] ** -0.5
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, H, 1, L]
            softcap = kwargs.get("softcap")
            if softcap is not None:
                scores = torch.tanh(scores / softcap) * softcap
            if attention_mask is not None:
                m = attention_mask[:, :, -1:, : k.shape[-2]]
                if m.dtype == torch.bool:
                    scores = scores.masked_fill(~m, float("-inf"))
                else:
                    scores = scores + m.float()
            probs = torch.softmax(scores, dim=-1)[:, :, 0, :]  # [B, H, L]
            self.rows.append((probs if self.keep_heads else probs.mean(dim=1)).cpu())
        return out

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self._registry = ALL_ATTENTION_FUNCTIONS
        self._orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS["sdpa"] = self._wrapper
        self.rows = []
        return self

    def __exit__(self, *exc):
        self._registry["sdpa"] = self._orig
        return False


def _aggregate(last: torch.Tensor, mean_all: torch.Tensor,
               block_ranges: List[Tuple[int, int, int]]) -> Tuple[Dict[int, float], Dict[int, float]]:
    last_scores, all_scores = {}, {}
    for bid, s, e in block_ranges:
        last_scores[bid] = float(last[s:e].sum())
        all_scores[bid] = float(mean_all[s:e].sum())
    return last_scores, all_scores


@torch.no_grad()
def score_batch(
    model,
    items: Sequence[Tuple[torch.Tensor, List[Tuple[int, int, int]]]],
    pad_token_id: int,
    keep_heads: bool = False,
) -> List[tuple]:
    """
    Fast path: score several prompts in one left-padded forward pass.

    Args:
        items: (input_ids [1, L_i], block_ranges) pairs from build_prompt_with_mapping.
        keep_heads: also return the per-layer, per-head attention mass of every block.
    Returns:
        One (last-layer scores, all-layer-mean scores) pair per item; with ``keep_heads`` a triple whose
        third entry is a float16 tensor ``[layers, heads, n_blocks]`` (blocks in ``block_ranges`` order).
    """
    device = next(model.parameters()).device
    seqs = [ids.reshape(-1) for ids, _ in items]
    # +1: always keep at least one masked pad column. With an all-ones mask HF drops
    # the mask and SDPA takes its is_causal path, which on Windows fell back to a
    # kernel ~25x slower at 4k tokens (0.03 vs 0.73 samples/s measured).
    max_len = max(len(x) for x in seqs) + 1
    input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    pads = []
    for i, x in enumerate(seqs):
        pad = max_len - len(x)
        input_ids[i, pad:] = x
        mask[i, pad:] = 1
        pads.append(pad)
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)

    base = getattr(model, "model", model)  # skip the LM head: only attention is needed
    with LastRowCapture(keep_heads=keep_heads) as cap:
        base(input_ids=input_ids.to(device), attention_mask=mask.to(device),
             position_ids=position_ids.to(device), use_cache=False)
    if not cap.rows:
        raise RuntimeError("No attention rows captured; is the teacher using attn_implementation='sdpa'?")

    stacked = torch.stack(cap.rows)  # [layers, B, L]  or  [layers, B, H, L] with keep_heads
    if not keep_heads:
        last, mean_all = stacked[-1], stacked.mean(dim=0)
        return [_aggregate(last[i, p:], mean_all[i, p:], ranges)
                for i, ((_, ranges), p) in enumerate(zip(items, pads))]
    last, mean_all = stacked[-1].mean(dim=1), stacked.mean(dim=(0, 2))  # head-mean of last layer; layer+head mean
    out = []
    for i, ((_, ranges), p) in enumerate(zip(items, pads)):
        last_scores, all_scores = _aggregate(last[i, p:], mean_all[i, p:], ranges)
        heads = torch.stack([stacked[:, i, :, p + s:p + e].sum(dim=-1) for _, s, e in ranges], dim=-1)
        out.append((last_scores, all_scores, heads.half()))
    return out


def score_batch_safe(model, items, pad_token_id, keep_heads: bool = False):
    """score_batch that halves the batch on CUDA OOM; None for items that still OOM alone."""
    try:
        return score_batch(model, items, pad_token_id, keep_heads=keep_heads)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(items) == 1:
            return [None]
        mid = len(items) // 2
        return (score_batch_safe(model, items[:mid], pad_token_id, keep_heads)
                + score_batch_safe(model, items[mid:], pad_token_id, keep_heads))


class OracleScorer:
    """Uniform interface over the fast (batched) and eager (reference) paths."""

    def __init__(self, model, tokenizer, attn_impl: str = "fast", keep_heads: bool = False):
        if keep_heads and attn_impl != "fast":
            raise ValueError("per-head attention mass (--save-heads) requires --attn-impl fast")
        self.model, self.attn_impl, self.keep_heads = model, attn_impl, keep_heads
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.recorder = LastRowAttentionRecorder(model) if attn_impl == "eager" else None

    def score(self, items):
        """items: list of (input_ids, block_ranges). Returns a list of score pairs (or None on OOM)."""
        if self.attn_impl == "fast":
            return score_batch_safe(self.model, items, self.pad_id, self.keep_heads)
        out = []
        for ids, ranges in items:
            try:
                out.append(block_attention(self.model, self.recorder, ids, ranges))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                out.append(None)
        return out

    def close(self):
        if self.recorder:
            self.recorder.remove()


class HeadStore:
    """Per-sample, per-head attention mass saved next to an oracle JSON (``<stem>.heads.pt``).

    ``mass[id]``: float16 ``[layers, heads, n_blocks]``; ``block_order[id]``: the block ids of its columns
    (identical to the sample's ``blocks`` order). Contains only tensors / lists / dicts / strings, so it loads
    with ``weights_only=True``.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.block_order: Dict[str, List[int]] = {}
        self.mass: Dict[str, torch.Tensor] = {}

    def load(self) -> "HeadStore":
        if self.path.exists():
            blob = torch.load(self.path, map_location="cpu", weights_only=True)
            self.block_order, self.mass = blob["block_order"], blob["mass"]
        return self

    def add(self, sample_id: str, block_ids: Sequence[int], mass: torch.Tensor) -> None:
        if mass.shape[-1] != len(block_ids):
            raise ValueError(f"head mass for {sample_id} has {mass.shape[-1]} columns but {len(block_ids)} blocks")
        self.block_order[str(sample_id)] = [int(b) for b in block_ids]
        self.mass[str(sample_id)] = mass

    def align_to(self, sample_ids: Sequence[str]) -> None:
        """Drop entries for samples not in ``sample_ids`` (an interrupted run); fail if any sample lacks heads."""
        keep = {str(i) for i in sample_ids}
        missing = [i for i in sample_ids if str(i) not in self.mass]
        if missing:
            raise ValueError(f"{self.path.name} has no heads for {len(missing)} resumed samples "
                             f"(e.g. {missing[0]}); regenerate this oracle file with --save-heads")
        self.block_order = {k: v for k, v in self.block_order.items() if k in keep}
        self.mass = {k: v for k, v in self.mass.items() if k in keep}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save({"block_order": self.block_order, "mass": self.mass}, tmp)
        tmp.replace(self.path)


# ===================================================================
# Prompt construction
# ===================================================================

def build_prompt_with_mapping(
    blocks: Sequence,
    question: str,
    tokenizer,
    max_tokens: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    """
    Tokenise ``[prefix][block_0]\\n[block_1]\\n...[question]`` block by block.

    Returns:
        input_ids [1, seq_len], and (block_id, start_tok, end_tok) for each
        block that fit in ``max_tokens``.
    """
    question_part = f"\n\nQuestion: {question}\nAnswer:"
    system_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=True)
    question_ids = tokenizer.encode(question_part, add_special_tokens=False)
    separator = tokenizer.encode("\n", add_special_tokens=False)

    budget = max_tokens - len(system_ids) - len(question_ids) - 4
    block_ranges = []
    context_ids: List[int] = []
    for block in blocks:
        block_ids = tokenizer.encode(block.text, add_special_tokens=False)
        if len(context_ids) + len(block_ids) + len(separator) > budget:
            break
        start = len(system_ids) + len(context_ids)
        context_ids.extend(block_ids)
        block_ranges.append((block.id, start, len(system_ids) + len(context_ids)))
        context_ids.extend(separator)

    input_ids = torch.tensor([system_ids + context_ids + question_ids], dtype=torch.long)
    return input_ids, block_ranges


@torch.no_grad()
def block_attention(
    model,
    recorder: LastRowAttentionRecorder,
    input_ids: torch.Tensor,
    block_ranges: List[Tuple[int, int, int]],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Forward pass; return per-block attention mass for (last layer, all-layer mean).
    """
    device = next(model.parameters()).device
    recorder.reset()
    model(input_ids.to(device), use_cache=False)
    if not recorder.rows:
        raise RuntimeError("No attention rows recorded.")
    last = recorder.rows[-1]
    mean_all = torch.stack(recorder.rows).mean(dim=0)
    recorder.reset()

    last_scores, all_scores = {}, {}
    for bid, s, e in block_ranges:
        last_scores[bid] = float(last[s:e].sum())
        all_scores[bid] = float(mean_all[s:e].sum())
    return last_scores, all_scores


# ===================================================================
# Evidence helpers
# ===================================================================

def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation/articles/whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_block_ids(blocks: Iterable, answers: Sequence[str], min_len: int = 3) -> List[int]:
    """Blocks whose normalised text contains any normalised answer string."""
    norm_answers = [normalize_answer(a) for a in answers]
    norm_answers = [a for a in norm_answers if len(a) >= min_len]
    hits = []
    for b in blocks:
        nb = normalize_answer(b.text)
        if any(a in nb for a in norm_answers):
            hits.append(b.id)
    return hits


def overlapping_block_ids(blocks: Iterable, spans: Sequence[Tuple[int, int]]) -> List[int]:
    """Blocks whose [start_char, end_char) overlaps any of the given char spans."""
    hits = []
    for b in blocks:
        if any(b.start_char < e and s < b.end_char for s, e in spans):
            hits.append(b.id)
    return hits


# ===================================================================
# I/O
# ===================================================================

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(path)


def make_metadata(dataset: str, teacher: str, max_tokens: int, chunk_size: int, top_k: int,
                  num_samples: int, split: str, **extra) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "teacher_model": teacher,
        "quantization": "4-bit NF4" if torch.cuda.is_available() else "none (cpu)",
        "label": "last-layer, head-mean attention of final token, summed per block",
        "max_context_tokens": max_tokens,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat(),
        **extra,
    }


def load_oracle_samples(path: Path) -> List[dict]:
    """
    Load oracle samples in either schema and normalise them to schema v2:

        {"id", "question", "domain", "blocks": [{"id", "text"}],
         "top_5_block_ids": [...], "gold_block_ids": [...] (may be empty), ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw["samples"] if isinstance(raw, dict) else raw

    out = []
    for i, s in enumerate(samples):
        s = dict(s)
        if "blocks" not in s and "all_block_texts" in s:  # v1 NarrativeQA
            s["blocks"] = [{"id": int(k), "text": v} for k, v in s["all_block_texts"].items()]
        if "top_5_block_ids" not in s and "labels" in s:  # v1 HotpotQA
            s["top_5_block_ids"] = s["labels"]
        s["top_5_block_ids"] = [int(x) for x in s["top_5_block_ids"]]
        s.setdefault("gold_block_ids", [])
        s.setdefault("id", f"{path.stem}-{i}")
        s.setdefault("domain", "unknown")
        out.append(s)
    return out
