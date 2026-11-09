import json
from pathlib import Path

import pytest
import torch

REPO_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "huggingface" / "hub"
BASE = "BAAI/bge-small-en-v1.5"


def _cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(BASE, "config.json", cache_dir=str(REPO_CACHE)), str)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cached(), reason="BGE-small is not in the repo HF cache")

OTHER = ["Bread is baked from flour and water.", "The train leaves at noon from platform four.",
         "A violin has four strings.", "Rain fell steadily over the harbour.", "The committee approved the budget."]


def _write_oracle_texts(tmp_path: Path, n: int) -> Path:
    """Every sample's target puts ~all mass on block 0 (an unrelated 'quartz' sentence)."""
    samples = []
    for i in range(n):
        blocks = ["Quartz crystals grow slowly inside granite caves."] + OTHER
        mass = {str(b): (0.95 if b == 0 else 0.01) for b in range(len(blocks))}
        samples.append({"id": f"s{i}", "domain": "hotpot_qa", "question": f"What do you know about topic number {i}?",
                        "blocks": [{"id": b, "text": t} for b, t in enumerate(blocks)],
                        "top_5_block_ids": [0], "gold_block_ids": [0],
                        "attention_scores": mass, "attention_scores_all_layers": mass})
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps({"metadata": {"schema_version": 2, "split": "train"}, "samples": samples}), encoding="utf-8")
    return path


def _mean_rank_of_block0(model, oracle: Path) -> float:
    from maplecore.indexer import query_prefix_for
    samples = json.loads(oracle.read_text(encoding="utf-8"))["samples"]
    prefix = query_prefix_for(BASE)
    ranks = []
    for s in samples:
        q = model.encode(prefix + s["question"], convert_to_tensor=True)
        b = model.encode([x["text"] for x in s["blocks"]], convert_to_tensor=True)
        order = torch.argsort(torch.nn.functional.cosine_similarity(q.unsqueeze(0), b), descending=True).tolist()
        ranks.append(order.index(0))
    return sum(ranks) / len(ranks)


def test_distill_moves_toward_target(tmp_path):
    from sentence_transformers import SentenceTransformer
    from scripts.train_biencoder import distill_biencoder

    oracle = _write_oracle_texts(tmp_path, 16)
    before = _mean_rank_of_block0(SentenceTransformer(BASE, device="cpu", cache_folder=str(REPO_CACHE)), oracle)
    out = distill_biencoder(BASE, [oracle], tmp_path / "enc", epochs=3, max_samples=16, device="cpu",
                            batch_questions=4, cache_folder=str(REPO_CACHE))
    after = _mean_rank_of_block0(SentenceTransformer(str(tmp_path / "enc"), device="cpu"), oracle)
    assert after < before and (tmp_path / "enc").exists()
    assert out["epochs"] == 3 and out["loss"][-1] < out["loss"][0]


def test_distill_rejects_degenerate_only_data(tmp_path):
    from scripts.train_biencoder import distill_biencoder

    oracle = _write_oracle_texts(tmp_path, 2)
    raw = json.loads(oracle.read_text(encoding="utf-8"))
    for s in raw["samples"]:
        s["attention_scores_all_layers"] = {k: 0.0 for k in s["attention_scores_all_layers"]}
    oracle.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="usable"):
        distill_biencoder(BASE, [oracle], tmp_path / "enc", epochs=1, device="cpu", cache_folder=str(REPO_CACHE))


def _toy_batch():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    blocks = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)  # question 0 owns 3 blocks, question 1 owns 2
    targets = [torch.tensor([0.7, 0.2, 0.1]), torch.tensor([0.4, 0.6])]
    return q, blocks, [3, 2], targets


def test_listwise_batch_within_document_scores_only_own_blocks():
    from scripts.train_biencoder import listwise_batch

    q, blocks, sizes, targets = _toy_batch()
    scores, target, mask = listwise_batch(q, blocks, sizes, targets, cross_negatives=False)
    assert scores.shape == (2, 3) and mask.tolist() == [[True, True, True], [True, True, False]]
    assert torch.allclose(scores[1, :2], blocks[3:5] @ q[1])
    assert torch.allclose(target[0], targets[0]) and target[1, 2] == 0


def test_listwise_batch_cross_negatives_scores_every_block_with_zero_target_outside_own_span():
    from scripts.train_biencoder import listwise_batch

    q, blocks, sizes, targets = _toy_batch()
    scores, target, mask = listwise_batch(q, blocks, sizes, targets, cross_negatives=True)
    assert scores.shape == (2, 5) and bool(mask.all())
    assert torch.allclose(scores, q @ blocks.T)
    assert torch.allclose(target[0], torch.tensor([0.7, 0.2, 0.1, 0.0, 0.0]))
    assert torch.allclose(target[1], torch.tensor([0.0, 0.0, 0.0, 0.4, 0.6]))


def test_distill_with_cross_negatives_records_the_setting(tmp_path):
    from scripts.train_biencoder import distill_biencoder

    oracle = _write_oracle_texts(tmp_path, 8)
    out = distill_biencoder(BASE, [oracle], tmp_path / "enc", epochs=1, max_samples=8, device="cpu",
                            batch_questions=4, cache_folder=str(REPO_CACHE), cross_negatives=True)
    assert out["cross_negatives"] is True and (tmp_path / "enc").exists()
