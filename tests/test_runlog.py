import hashlib
import json
import logging
import sys

import pytest

from scripts.runlog import current_run, flatten_metrics, import_legacy, start_run, write_model_card


def test_run_dir_contents(tmp_path):
    with start_run("bench", "t", {"a": 1}, root=tmp_path) as run:
        logging.getLogger("x").info("hello")
        print("printed")
        run.log_metrics({"loss": 0.5}, step=1)
        run.save_items("per_q", [{"id": "q1", "ranking": [3, 1]}])
    d = run.dir
    names = {p.name for p in d.iterdir()}
    assert {"config.json", "run.log", "metrics.jsonl", "summary.json", "hardware.json"} <= names
    log = (d / "run.log").read_text(encoding="utf-8")
    assert "hello" in log and "printed" in log
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg["a"] == 1 and {"commit", "dirty"} <= set(cfg["git"]) and "torch" in cfg["env"]
    assert any(p.lower().startswith("torch==") for p in cfg["env"]["packages"])  # venv without pip still lists packages
    row = json.loads((d / "items" / "per_q.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["ranking"] == [3, 1]
    last = json.loads((tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["status"] == "completed" and last["id"] == run.id


def test_failed_run_keeps_log_and_status(tmp_path):
    with pytest.raises(RuntimeError):
        with start_run("train", "boom", {}, root=tmp_path) as run:
            logging.getLogger("x").info("before crash")
            raise RuntimeError("oom")
    s = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert s["status"] == "failed" and "RuntimeError: oom" in s["traceback"]
    assert "before crash" in (run.dir / "run.log").read_text(encoding="utf-8")


def test_keyboard_interrupt_recorded_as_failed(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with start_run("train", "ctrlc", {}, root=tmp_path) as run:
            raise KeyboardInterrupt()
    s = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert s["status"] == "failed" and s["traceback"].startswith("KeyboardInterrupt")


def test_streams_and_handlers_restored(tmp_path):
    out, err = sys.stdout, sys.stderr
    n_handlers = len(logging.getLogger().handlers)
    with start_run("bench", "restore", {}, root=tmp_path):
        pass
    assert sys.stdout is out and sys.stderr is err
    assert len(logging.getLogger().handlers) == n_handlers


def test_artifact_checksum(tmp_path):
    f = tmp_path / "m.pt"
    f.write_bytes(b"abc")
    with start_run("train", "a", {}, root=tmp_path) as run:
        run.add_artifact(f, role="checkpoint")
    (a,) = json.loads((run.dir / "artifacts.json").read_text(encoding="utf-8"))
    assert a["sha256"] == hashlib.sha256(b"abc").hexdigest() and a["role"] == "checkpoint"
    assert a["bytes"] == 3
    assert (run.dir / "artifacts" / "m.pt").read_bytes() == b"abc"  # small files are copied


def test_model_card(tmp_path):
    f = tmp_path / "m.pt"
    f.write_bytes(b"x")
    with start_run("train", "c", {"seed": 42}, root=tmp_path) as run:
        card = write_model_card(f, run, {"val_evidence_recall@5": 0.8})
    c = json.loads(card.read_text(encoding="utf-8"))
    assert c["run_id"] == run.id and c["config"]["seed"] == 42
    assert c["metrics"]["val_evidence_recall@5"] == 0.8
    assert c["sha256"] == hashlib.sha256(b"x").hexdigest()


def test_import_legacy_collects_logs_results_and_checksums(tmp_path):
    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True)
    (repo / "logs" / "a.log").write_text("log a", encoding="utf-8")
    (repo / "benchmarks" / "results" / "b1536_s42").mkdir(parents=True)
    (repo / "benchmarks" / "results" / "b1536_s42" / "r.json").write_text("{}", encoding="utf-8")
    (repo / "models").mkdir()
    (repo / "models" / "m.pth").write_bytes(b"w")
    (repo / "models" / "m.train.json").write_text("{}", encoding="utf-8")
    (repo / "data" / "oracle").mkdir(parents=True)
    (repo / "data" / "oracle" / "o.json").write_text("{}", encoding="utf-8")

    out = import_legacy(root=tmp_path / "runs", repo=repo)

    assert (out / "logs" / "a.log").read_text(encoding="utf-8") == "log a"
    assert (out / "benchmarks" / "results" / "b1536_s42" / "r.json").exists()
    assert (out / "models" / "m.train.json").exists()
    arts = {a["path"]: a for a in json.loads((out / "artifacts.json").read_text(encoding="utf-8"))}
    assert arts["models/m.pth"]["sha256"] == hashlib.sha256(b"w").hexdigest()
    assert "data/oracle/o.json" in arts
    last = json.loads((tmp_path / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["kind"] == "legacy"


def test_same_second_runs_get_distinct_dirs(tmp_path):
    with start_run("bench", "dup", {}, root=tmp_path) as a:
        pass
    # force a collision by reusing the first run's id prefix
    with start_run("bench", "dup", {}, root=tmp_path) as b:
        pass
    assert a.dir != b.dir and a.dir.exists() and b.dir.exists()


def test_flatten_metrics_keeps_only_summary_means():
    tree = {"datasets": {"HotpotQA": {"summary": {"dense": {"recall@5": {"mean": 0.8, "ci_low": 0.7, "n": 10}}},
                                       "num_samples": 500}}, "config": {"seed": 1}, "ranking": [1, 2]}
    flat = flatten_metrics(tree)
    assert flat == {"datasets/HotpotQA/summary/dense/recall@5": 0.8}


def test_current_run_is_active_only_inside_context(tmp_path):
    assert current_run() is None
    with start_run("bench", "cur", {}, root=tmp_path) as run:
        assert current_run() is run
    assert current_run() is None
