"""
Run logger. One directory per run:

    runs/<run_id>/
        meta.json              config, condition, corpus, start/end, totals
        calls/NNN_label.json   every prompt + response + usage
        iterations/NN.json     loop state after each iteration
        corpus/                snapshot of the processed transcripts the run cited
        output.json            final result
        output.md              human-readable report

Nothing is ever overwritten; failed and bad runs are kept — they are
evidence for the case study.

Progress lines go through `emit`. The default writes to stderr, never stdout:
an MCP server on stdio transport owns stdout for the protocol, and one stray
print corrupts the session. Surfaces pass their own emit (a notification, a UI).

`redact=True` (hosted mode) keeps token counts and timings but stores prompt
and response bodies as length + SHA-256, and does not snapshot the corpus.
Transcript content never lands on disk on a machine the user does not own.
"""

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _digest(s: str) -> dict:
    return {"chars": len(s or ""), "sha256": hashlib.sha256((s or "").encode("utf-8")).hexdigest()}


class RunLogger:
    def __init__(self, root: str | Path = "runs", condition: str = "C",
                 tag: str = "", config: dict | None = None,
                 emit: Callable[[str], None] | None = None, redact: bool = False):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{ts}-{condition}{('-' + tag) if tag else ''}"
        self.dir = Path(root) / self.run_id
        (self.dir / "calls").mkdir(parents=True, exist_ok=True)
        (self.dir / "iterations").mkdir(exist_ok=True)
        self.emit = emit or _stderr
        self.redact = redact
        self.t0 = time.time()
        self.n_calls = 0
        self.totals = {"in_tok": 0, "out_tok": 0, "cost": 0.0, "calls": 0}
        self.meta = {
            "run_id": self.run_id,
            "condition": condition,
            "tag": tag,
            "started": datetime.now(timezone.utc).isoformat(),
            "redacted": redact,
            "config": config or {},
        }
        self._write("meta.json", self.meta)

    def _write(self, rel: str, obj):
        with open(self.dir / rel, "w", encoding="utf-8") as fh:
            if isinstance(obj, str):
                fh.write(obj)
            else:
                json.dump(obj, fh, indent=2, ensure_ascii=False)

    def record_call(self, label: str, system: str, user: str, result: dict):
        self.n_calls += 1
        self.totals["calls"] += 1
        self.totals["in_tok"] += result["in_tok"]
        self.totals["out_tok"] += result["out_tok"]
        self.totals["cost"] += result["cost"]
        if self.redact:
            # Prompts carry the transcripts and responses quote them; both become digests.
            # Usage, timing, and stop reason stay: that is the metering record.
            rec = {"label": label, "system": _digest(system), "user": _digest(user),
                   **{k: v for k, v in result.items() if k not in ("text", "data")},
                   "text": _digest(result.get("text") or "")}
        else:
            rec = {"label": label, "system": system, "user": user, **result}
        self._write(f"calls/{self.n_calls:03d}_{label}.json", rec)
        self.emit(f"[{label}] {result['model']}  {result['in_tok']}→{result['out_tok']} tok  "
                  f"{result['seconds']}s  ${result['cost']:.4f}")

    def record_iteration(self, n: int, state: dict):
        stage = state.get("stage", "state")
        self._write(f"iterations/{n:02d}_{stage}.json", state)

    def note(self, msg: str):
        self.emit(f"· {msg}")
        with open(self.dir / "notes.txt", "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    def snapshot_corpus(self, processed_dir: str | Path, names: list[str] | None = None) -> Path | None:
        """Copy the processed transcripts this run uses into runs/<id>/corpus/ so
        receipts and re-critiques resolve against exactly what the run saw.
        Skipped in redact mode; the manifest is still recorded in meta."""
        src = Path(processed_dir)
        manifest = json.load(open(src / "manifest.json", encoding="utf-8"))
        if names:
            manifest = [m for m in manifest if m["name"] in names]
        self.meta["corpus"] = {"source": str(src.resolve()), "transcripts": [m["name"] for m in manifest],
                               "words": sum(m["words"] for m in manifest)}
        self._write("meta.json", self.meta)
        if self.redact:
            return None
        dst = self.dir / "corpus"
        dst.mkdir(exist_ok=True)
        json.dump(manifest, open(dst / "manifest.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        for m in manifest:
            for ext in (".txt", ".jsonl"):
                shutil.copy(src / f"{m['name']}{ext}", dst / f"{m['name']}{ext}")
        return dst

    def finish(self, output: dict, markdown: str | None = None,
               stop_reason: str = "", iterations: int = 0):
        self.meta.update({
            "finished": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": round(time.time() - self.t0, 1),
            "iterations": iterations,
            "stop_reason": stop_reason,
            **self.totals,
            "cost": round(self.totals["cost"], 4),
        })
        self._write("meta.json", self.meta)
        self._write("output.json", output)
        if markdown:
            self._write("output.md", markdown)
        self.emit(f"run {self.run_id}: {iterations} iteration(s), stop={stop_reason}, "
                  f"{self.meta['wall_seconds']}s, {self.totals['in_tok']:,}→{self.totals['out_tok']:,} tok, "
                  f"${self.meta['cost']}")
        return self.dir
