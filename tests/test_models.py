import pytest
import torch

from maplecore import MapleNet, MapleResidualNet, MapleScanner, load_model, save_model
from maplecore.models import export_to_onnx
from maplecore.onnx_runner import MapleONNXRunner

from conftest import make_index


def test_untrained_residual_ranks_like_cosine():
    m = MapleResidualNet(dim=16).eval()
    q, b = torch.randn(16), torch.randn(50, 16)
    logits = m(torch.cat([q.expand(50, -1), b], 1))
    cos = torch.nn.functional.cosine_similarity(q.unsqueeze(0), b)
    assert torch.equal(torch.argsort(logits, descending=True), torch.argsort(cos, descending=True))


def test_residual_shape_and_config():
    m = MapleResidualNet(dim=16, hidden_dim=32)
    assert m(torch.randn(5, 32)).shape == (5,)
    assert m.dim == 16 and m.input_dim == 32
    assert m.config == {"dim": 16, "hidden_dim": 32, "dropout": 0.1, "init_scale": 20.0}


def test_residual_learns_beyond_cosine():
    """The MLP branch receives gradient even though its last layer starts at zero."""
    m = MapleResidualNet(dim=8, hidden_dim=16)
    x = torch.randn(64, 16)
    m(x).sum().backward()
    assert m.mlp[-1].weight.grad.abs().sum() > 0 and m.scale.grad is not None


def test_save_load_roundtrip(tmp_path):
    m = MapleResidualNet(dim=16, hidden_dim=32).eval()
    save_model(m, tmp_path / "m.pt")
    m2 = load_model(tmp_path / "m.pt", device="cpu")
    x = torch.randn(4, 32)
    assert isinstance(m2, MapleResidualNet) and torch.allclose(m(x), m2(x))
    assert not m2.training


def test_maplenet_roundtrips_through_registry(tmp_path):
    m = MapleNet(input_dim=32, hidden_dim=8).eval()
    save_model(m, tmp_path / "n.pt")
    m2 = load_model(tmp_path / "n.pt", device="cpu")
    x = torch.randn(3, 32)
    assert isinstance(m2, MapleNet) and torch.allclose(m(x), m2(x))


def test_load_model_reads_legacy_maplenet(tmp_path):
    MapleNet(input_dim=32, hidden_dim=8).save(tmp_path / "old.pth")
    assert isinstance(load_model(tmp_path / "old.pth", device="cpu"), MapleNet)


def test_load_model_dim_mismatch_raises(tmp_path):
    save_model(MapleResidualNet(dim=16), tmp_path / "m.pt")
    with pytest.raises(ValueError, match="input_dim"):
        load_model(tmp_path / "m.pt", device="cpu", expected_input_dim=768)


def test_unknown_class_in_checkpoint_raises(tmp_path):
    torch.save({"format": "maple-v2", "class": "Nope", "config": {}, "state_dict": {}}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="Nope"):
        load_model(tmp_path / "x.pt", device="cpu")


def test_residual_works_with_scanner_and_onnx(tmp_path):
    m = MapleResidualNet(dim=16)
    res = MapleScanner(m, device="cpu").search(torch.randn(16), make_index(300), k=5, strategy="linear")
    assert res.num_results == 5
    pytest.importorskip("onnxruntime")
    m.eval()
    export_to_onnx(m, tmp_path / "m.onnx")
    x = torch.randn(3, 32)
    onnx_scores = MapleONNXRunner(tmp_path / "m.onnx").predict(x.numpy())
    assert onnx_scores.shape == (3,)
    assert torch.allclose(torch.sigmoid(m(x)), torch.from_numpy(onnx_scores), atol=1e-4)
