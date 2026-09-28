"""Redact local machine paths from frozen evidence snapshots (evidence/<name>/).

Snapshots captured before src/guardrails/pii.py::strip_local_paths existed can contain absolute
paths (e.g. an index location or a library stack trace). This rewrites only those path strings,
in place, via the same strip_local_paths() used by every logger: record order, line numbers, run
ids, trace ids and span ids are unchanged, so citations into the snapshot still resolve. Each
snapshot's manifest.json records the redaction (file, old and new sha256) for transparency.

Run: python -m scripts.redact_local_paths evidence/pre_fix evidence/governance
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR
from src.guardrails.pii import strip_local_paths

TEXT = {".jsonl", ".json", ".log", ".csv", ".md", ".txt"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def redact_file(p: Path) -> bool:
    if p.suffix in TEXT:
        old = p.read_text(encoding="utf-8")
        new = "\n".join(strip_local_paths(line) for line in old.split("\n"))
        if new != old:
            p.write_text(new, encoding="utf-8")
            return True
        return False
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        changed = False
        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                new = df[col].map(lambda v: strip_local_paths(v) if isinstance(v, str) else v)
                if not new.equals(df[col]):
                    df[col], changed = new, True
        if changed:
            df.to_parquet(p, index=False)
        return changed
    return False


def main(dirs: list[str]) -> int:
    for d in dirs:
        base = (ROOT_DIR / d).resolve()
        manifest_path = base / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        changes = []
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.name != "manifest.json":
                before = sha(p)
                if redact_file(p):
                    rel = str(p.relative_to(base))
                    changes.append({"file": rel, "sha256_before": before, "sha256_after": sha(p)})
                    if rel in manifest.get("files", {}):
                        manifest["files"][rel]["sha256"] = changes[-1]["sha256_after"]
        if changes:
            manifest.setdefault("redactions", []).append({
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tool": "scripts/redact_local_paths.py (src/guardrails/pii.py::strip_local_paths)",
                "what": "local machine path strings only; no records, ids or line numbers changed",
                "files": changes})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{d}: {len(changes)} file(s) redacted")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
