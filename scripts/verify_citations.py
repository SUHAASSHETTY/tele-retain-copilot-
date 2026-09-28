"""Citation-resolves check for docs/*.md. Exits 1 if any citation does not resolve.

Checked citation types (anything in a doc that points at evidence):
  run_id      UUIDs                      -> must appear in a committed artifact (logs/, traces/, reports/, evidence/)
  trace_id    32-hex ids                 -> must be a trace id in a committed span parquet
  span_id     16-hex ids                 -> must be a span id in a committed span parquet
  log line    `path/file.jsonl:L12`      -> file exists, has that line, and (if the doc gives `run_id` on the
                                             same bullet/line) the record carries that run_id
  file path   `src/...`, `scripts/...`, `logs/...`, `reports/...`, `traces/...`, `evidence/...`, `docs/...`,
              `data/...`, `tests/...`, `mcp_server/...`  -> must exist
  policy ref  POL-XXX-NNN §a.b           -> must be a clause in data/policy_corpus
  code symbol `src/x.py::name` or `src/x.py::Class.method` -> file defines it (def / class / assignment)
  code line   `src/x.py:L12`             -> file has that line
  doc anchor  [text](other.md#heading)   -> the target doc has that heading

Run: python -m scripts.verify_citations [docs/file.md ...]
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR
from src.guardrails.output_guard import known_citations

UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
TRACE_RE = re.compile(r"(?<![0-9a-f-])[0-9a-f]{32}(?![0-9a-f-])")
SPAN_RE = re.compile(r"(?<![0-9a-f-])[0-9a-f]{16}(?![0-9a-f-])")
LINE_RE = re.compile(r"`?((?:logs|reports|evidence|traces)/[\w./-]+\.(?:jsonl|log|json|csv|md)):L(\d+)`?")
PATH_RE = re.compile(r"`((?:src|scripts|logs|reports|traces|evidence|docs|data|tests|mcp_server)/[\w./*-]+)`")
POLICY_RE = re.compile(r"POL-[A-Z]{3}-\d{3} §\d+\.\d+")
ANCHOR_RE = re.compile(r"\]\(([\w./-]*\.md)?#([\w-]+)\)")
CODE_ROOTS = r"(?:src|scripts|mcp_server|tests)"


def slug(heading: str) -> str:
    """GitHub-style heading anchor."""
    h = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", h))
SYMBOL_RE = re.compile(rf"({CODE_ROOTS}/[\w./-]+\.py)::([A-Za-z_][\w.]*)")
CODE_LINE_RE = re.compile(rf"({CODE_ROOTS}/[\w./-]+\.py):L(\d+)")
EVIDENCE_DIRS = ("logs", "traces", "reports", "evidence")


@lru_cache(maxsize=1)
def artifact_text() -> str:
    parts = []
    for d in EVIDENCE_DIRS:
        for f in sorted((ROOT_DIR / d).rglob("*")):
            if f.is_file() and f.suffix in {".jsonl", ".json", ".log", ".csv", ".md", ".txt"}:
                parts.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


@lru_cache(maxsize=1)
def span_ids() -> tuple[set[str], set[str], set[str]]:
    spans, traces, runs = set(), set(), set()
    for d in ("traces", "evidence"):
        for f in (ROOT_DIR / d).rglob("*.parquet"):
            cols = pd.read_parquet(f).columns
            use = [c for c in ("context.span_id", "context.trace_id", "attributes.copilot.run_id") if c in cols]
            df = pd.read_parquet(f, columns=use)
            spans |= set(df.get("context.span_id", pd.Series(dtype=str)).dropna())
            traces |= set(df.get("context.trace_id", pd.Series(dtype=str)).dropna())
            runs |= set(df.get("attributes.copilot.run_id", pd.Series(dtype=str)).dropna())
    return spans, traces, runs


def content_mismatch(file: str, cited: str, doc_line: str) -> str | None:
    """A line reference must point at what the doc says it is, not just at a line that exists."""
    if file.endswith(".jsonl"):
        try:
            rec = json.loads(cited)
        except json.JSONDecodeError:
            return None
        key = rec.get("action") or rec.get("tool_name") or rec.get("name")
        if key and key not in doc_line:
            return f"record is '{key}', which the citing line does not mention"
        return None
    values = [v for v in re.findall(r'"([^"]{3,})"', cited) if not v.endswith(("_id", "name", "status", "reason"))]
    if values and not any(v in doc_line for v in values):
        return f"cited line ({cited.strip()[:60]}) does not match the citing text"
    return None


def check_doc(path: Path) -> tuple[int, list[str]]:
    text = path.read_text()
    errors: list[str] = []
    checked = 0
    spans, traces, span_runs = span_ids()
    corpus = artifact_text()

    for rid in sorted(set(UUID_RE.findall(text))):
        checked += 1
        if rid not in corpus and rid not in span_runs:
            errors.append(f"run_id {rid} not found in any committed artifact")
    no_uuid = UUID_RE.sub("", text)
    for tid in sorted(set(TRACE_RE.findall(no_uuid))):
        checked += 1
        if tid not in traces:
            errors.append(f"trace_id {tid} not found in any span parquet")
    for sid in sorted(set(SPAN_RE.findall(TRACE_RE.sub("", no_uuid)))):
        checked += 1
        if sid not in spans:
            errors.append(f"span_id {sid} not found in any span parquet")

    for line in text.splitlines():
        for file, n in LINE_RE.findall(line):
            checked += 1
            f = ROOT_DIR / file
            if not f.exists():
                errors.append(f"{file}:L{n}: file does not exist")
                continue
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            if int(n) < 1 or int(n) > len(lines):
                errors.append(f"{file}:L{n}: file has only {len(lines)} lines")
                continue
            cited = lines[int(n) - 1]
            for rid in UUID_RE.findall(line):
                if rid in corpus and rid not in cited and file.endswith(".jsonl"):
                    errors.append(f"{file}:L{n}: record does not carry run_id {rid}")
            problem = content_mismatch(file, cited, line)
            if problem:
                errors.append(f"{file}:L{n}: {problem}")

    for p in sorted(set(PATH_RE.findall(text))):
        checked += 1
        if "*" in p:
            if not list(ROOT_DIR.glob(p)):
                errors.append(f"path glob `{p}` matches nothing")
        elif not (ROOT_DIR / p.split(":L")[0]).exists():
            errors.append(f"path `{p}` does not exist")

    for file, symbol in sorted(set(SYMBOL_RE.findall(text))):
        checked += 1
        f = ROOT_DIR / file
        if not f.exists():
            errors.append(f"{file}::{symbol}: file does not exist")
            continue
        src = f.read_text()
        for part in symbol.split("."):
            if not re.search(rf"^\s*(async\s+def|def|class)\s+{re.escape(part)}\b|^\s*{re.escape(part)}\s*[:=]", src, re.M):
                errors.append(f"{file}::{symbol}: '{part}' is not defined in that file")
                break

    for file, n in sorted(set(CODE_LINE_RE.findall(text))):
        checked += 1
        f = ROOT_DIR / file
        if not f.exists() or int(n) > len(f.read_text().splitlines()):
            errors.append(f"{file}:L{n}: line does not exist")

    for target, anchor in sorted(set(ANCHOR_RE.findall(text))):
        checked += 1
        doc = path.parent / target if target else path
        if not doc.exists():
            errors.append(f"link {target}#{anchor}: document does not exist")
        elif anchor not in {slug(h) for h in re.findall(r"^#+\s+(.+)$", doc.read_text(), re.M)}:
            errors.append(f"link {target}#{anchor}: no such heading")

    for ref in sorted(set(POLICY_RE.findall(text))):
        checked += 1
        if ref not in known_citations():
            errors.append(f"policy clause {ref} does not exist in data/policy_corpus")
    return checked, errors


def main(argv: list[str]) -> int:
    docs = [Path(a) for a in argv] or [*sorted((ROOT_DIR / "docs").glob("*.md")), ROOT_DIR / "README.md"]
    total, failed = 0, 0
    for d in docs:
        d = d if d.is_absolute() else ROOT_DIR / d
        checked, errors = check_doc(d)
        total += checked
        status = "PASS" if not errors else "FAIL"
        print(f"[{status}] {d.relative_to(ROOT_DIR)}: {checked} citation(s) checked, {len(errors)} unresolved")
        for e in errors:
            print(f"    - {e}")
        failed += bool(errors)
    if not docs:
        print("no docs to check")
    print(f"{'PASS' if not failed else 'FAIL'}: {total} citation(s) across {len(docs)} doc(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
