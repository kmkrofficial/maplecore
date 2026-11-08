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

Memory: instead of ``output_attentions=True`` (which keeps every layer's full
``[heads, L, L]`` matrix alive), a forward hook on each attention module keeps
only the last query row. This lets the context budget go well beyond 1,536
tokens on a 16 GB GPU.
"""

from __future__ import annotations

import json
import logging
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

DEFAULT_TEACHER = "meta-llama/Meta-Llama-3-8B-Instruct"
SCHEMA_VERSION = 2
PROMPT_PREFIX = "Read the following context carefully and answer the question.\n\nContext:\n"


# ===================================================================
# Teacher model
# ===================================================================

def load_teacher(model_id: str = DEFAULT_TEACHER, load_in_4bit: bool = True):
    """Load a causal LM with eager attention (needed to read attention weights)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = dict(attn_implementation="eager", dtype=torch.float16)
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

    logger.info(f"Loading teacher {model_id} (4-bit={load_in_4bit and torch.cuda.is_available()})")
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
