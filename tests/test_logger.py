"""etot_core.logger.RunLogger.snapshot_corpus. Offline."""

import json

import pytest

from etot_core.logger import RunLogger


def motif_corpus(d):
    """The shape Motif's intake writes: manifest entries with name and words, <name>.txt and <name>.jsonl, and other
    files beside them that a snapshot does not copy."""
    d.mkdir()
    manifest = [{"file": "Alice.docx", "name": "alice", "turns": 4, "words": 40},
                {"file": "Bob.docx", "name": "bob", "turns": 2, "words": 25},
                {"file": "Cara.docx", "name": "cara", "turns": 3, "words": 30}]
    (d / "manifest.json").write_text(json.dumps(manifest))
    for m in manifest:
        (d / f"{m['name']}.txt").write_text(f"{m['name']} text")
        (d / f"{m['name']}.jsonl").write_text(json.dumps({"turn": 1}) + "\n")
        (d / f"{m['name']}.json").write_text("{}")          # not part of a snapshot
    return manifest


# ── 0.1.0 behaviour, pinned: Motif depends on it ───────────────────────────────────────────────────

def test_motif_shape_snapshot(tmp_path):
    src = tmp_path / "processed"
    manifest = motif_corpus(src)
    log = RunLogger(root=tmp_path / "runs", condition="C")
    dst = log.snapshot_corpus(src, ["alice", "bob"])
    assert dst == log.dir / "corpus"
    assert sorted(p.name for p in dst.iterdir()) == ["alice.jsonl", "alice.txt", "bob.jsonl", "bob.txt", "manifest.json"]
    assert json.loads((dst / "manifest.json").read_text()) == manifest[:2]
    assert (dst / "manifest.json").read_text() == json.dumps(manifest[:2], indent=2, ensure_ascii=False)
    expected = {"source": str(src.resolve()), "transcripts": ["alice", "bob"], "words": 65}
    assert log.meta["corpus"] == expected
    assert json.loads((log.dir / "meta.json").read_text())["corpus"] == expected


def test_no_names_snapshots_every_entry(tmp_path):
    src = tmp_path / "processed"
    motif_corpus(src)
    log = RunLogger(root=tmp_path / "runs")
    log.snapshot_corpus(src)
    assert log.meta["corpus"]["transcripts"] == ["alice", "bob", "cara"] and log.meta["corpus"]["words"] == 95


def test_redacted_run_records_the_manifest_and_copies_nothing(tmp_path):
    src = tmp_path / "processed"
    motif_corpus(src)
    log = RunLogger(root=tmp_path / "runs", redact=True)
    assert log.snapshot_corpus(src, ["bob"]) is None
    assert not (log.dir / "corpus").exists()
    assert log.meta["corpus"] == {"source": str(src.resolve()), "transcripts": ["bob"], "words": 25}


def test_a_missing_file_is_an_error(tmp_path):
    src = tmp_path / "processed"
    motif_corpus(src)
    (src / "bob.jsonl").unlink()
    log = RunLogger(root=tmp_path / "runs")
    with pytest.raises(FileNotFoundError):
        log.snapshot_corpus(src, ["bob"])


# ── 0.2.0: entries that declare their files; words optional ────────────────────────────────────────

def declared_corpus(d):
    """A consumer that is not Motif: entries name their own files and carry no word count."""
    d.mkdir()
    manifest = [{"name": "library", "kind": "library", "nodes": 387, "files": ["library.jsonl", "library.txt"]},
                {"name": "frames", "kind": "frames", "nodes": 186, "files": ["frames.jsonl"]}]
    (d / "manifest.json").write_text(json.dumps(manifest))
    for f in ("library.jsonl", "library.txt", "frames.jsonl", "frames.txt"):
        (d / f).write_text(f)
    return manifest


def test_declared_files_are_copied_exactly(tmp_path):
    src = tmp_path / "processed"
    manifest = declared_corpus(src)
    log = RunLogger(root=tmp_path / "runs")
    dst = log.snapshot_corpus(src, ["library", "frames"])
    assert sorted(p.name for p in dst.iterdir()) == ["frames.jsonl", "library.jsonl", "library.txt", "manifest.json"]
    assert json.loads((dst / "manifest.json").read_text()) == manifest


def test_words_are_omitted_unless_every_entry_has_them(tmp_path):
    src = tmp_path / "processed"
    declared_corpus(src)
    log = RunLogger(root=tmp_path / "runs")
    log.snapshot_corpus(src)
    assert log.meta["corpus"] == {"source": str(src.resolve()), "transcripts": ["library", "frames"]}


def test_declared_and_default_entries_mix(tmp_path):
    src = tmp_path / "processed"
    motif_corpus(src)
    manifest = json.loads((src / "manifest.json").read_text())
    manifest.append({"name": "extra", "files": ["extra.jsonl"]})
    (src / "manifest.json").write_text(json.dumps(manifest))
    (src / "extra.jsonl").write_text("{}\n")
    log = RunLogger(root=tmp_path / "runs")
    dst = log.snapshot_corpus(src, ["alice", "extra"])
    assert sorted(p.name for p in dst.iterdir()) == ["alice.jsonl", "alice.txt", "extra.jsonl", "manifest.json"]
    assert "words" not in log.meta["corpus"]


@pytest.mark.parametrize("bad", [["../escape.jsonl"], ["sub/dir.jsonl"], ["/abs.jsonl"], [""], [".."], "one.jsonl", [3]])
def test_declared_files_must_be_plain_names_and_nothing_is_written_first(tmp_path, bad):
    src = tmp_path / "processed"
    src.mkdir()
    (src / "manifest.json").write_text(json.dumps([{"name": "x", "files": bad}]))
    log = RunLogger(root=tmp_path / "runs")
    with pytest.raises(ValueError, match="plain file names"):
        log.snapshot_corpus(src)
    assert "corpus" not in log.meta and not (log.dir / "corpus").exists()


def test_a_missing_declared_file_is_an_error(tmp_path):
    src = tmp_path / "processed"
    declared_corpus(src)
    (src / "frames.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        RunLogger(root=tmp_path / "runs").snapshot_corpus(src, ["frames"])
