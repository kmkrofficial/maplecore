import hashlib
import json


def test_build_manifest_hashes_every_file_and_skips_missing_roots(tmp_path):
    from scripts.evidence_manifest import build_manifest

    (tmp_path / "res" / "sub").mkdir(parents=True)
    (tmp_path / "res" / "a.json").write_text('{"x": 1}', encoding="utf-8")
    (tmp_path / "res" / "sub" / "b.txt").write_bytes(b"hello")
    rows = build_manifest([tmp_path / "res", tmp_path / "missing"], base=tmp_path)
    assert [r["path"] for r in rows] == ["res/a.json", "res/sub/b.txt"]
    assert rows[1]["bytes"] == 5 and rows[1]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert all("modified" in r for r in rows)


def test_write_manifest_records_git_head_and_file_count(tmp_path):
    from scripts.evidence_manifest import write_manifest

    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "a.json").write_text("{}", encoding="utf-8")
    out = write_manifest([tmp_path / "res"], tmp_path / "manifest.json", base=tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["n_files"] == 1 == len(data["files"]) and "git_head" in data and out == data
