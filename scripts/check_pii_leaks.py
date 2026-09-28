"""Grep-style PII leak check over committed evidence. Exits 1 on any finding.

Scans logs/, traces/ and reports/ (text files and Parquet) for:
- every synthetic customer_id, account_number, phone, email and full name in data/synthetic/telecom.db
- generic unmasked shapes: CUST-######, ACC-########, +1-555-01##, emails, Luhn-valid card numbers

Run: python -m scripts.check_pii_leaks [--paths logs traces reports]
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

from src.config import ROOT_DIR, SYNTHETIC_DIR

TEXT_SUFFIXES = {".jsonl", ".json", ".log", ".csv", ".txt", ".md"}
SHAPES = {
    "customer_id": re.compile(r"\bCUST-\d{6}\b"),
    "account_number": re.compile(r"\bACC-\d{8}\b"),
    "synthetic_phone": re.compile(r"\+1-555-01\d{2}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # not preceded by a digit or '.': fractional digits of floats (66.95700000000001) are not card numbers
    "card_number": re.compile(r"(?<![\d.])\b(?:\d[ -]?){12,18}\d\b"),
}


def luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return len(digits) >= 13 and total % 10 == 0


def known_identifiers() -> dict[str, str]:
    con = sqlite3.connect(SYNTHETIC_DIR / "telecom.db")
    ids: dict[str, str] = {}
    for cid, acc, phone, email, first, last in con.execute(
            "SELECT customer_id, account_number, phone, email, first_name, last_name FROM customers"):
        ids.update({cid: "customer_id", acc: "account_number", phone: "phone", email: "email",
                    f"{first} {last}": "full_name"})
    return ids


def file_text(path: Path) -> str | None:
    if path.suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).astype(str).to_csv(index=False)
    return None


def scan(paths: list[Path]) -> list[dict]:
    ids = known_identifiers()
    findings = []
    for root in paths:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for f in files:
            text = file_text(f)
            if text is None:
                continue
            rel = f.relative_to(ROOT_DIR)
            for value, kind in ids.items():
                if value in text:
                    findings.append({"file": str(rel), "kind": f"known_{kind}", "sample": value[:4] + "..."})
            for kind, rx in SHAPES.items():
                for m in rx.finditer(text):
                    hit = m.group(0)
                    if kind == "card_number" and not luhn_ok(hit):
                        continue
                    if kind == "email" and hit.split("@")[0].endswith("***"):
                        continue
                    findings.append({"file": str(rel), "kind": f"shape_{kind}", "sample": hit[:4] + "..."})
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="*", default=["logs", "traces", "reports", "evidence"])
    args = ap.parse_args()
    paths = [ROOT_DIR / p for p in args.paths if (ROOT_DIR / p).exists()]
    findings = scan(paths)
    scanned = sum(1 for p in paths for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file())
    if findings:
        by_file: dict[str, int] = {}
        for f in findings:
            by_file[f["file"]] = by_file.get(f["file"], 0) + 1
        print(f"FAIL: {len(findings)} plaintext identifier(s) in {len(by_file)} file(s)")
        for file, n in sorted(by_file.items()):
            kinds = sorted({x["kind"] for x in findings if x["file"] == file})
            print(f"  {file}: {n} ({', '.join(kinds)})")
        return 1
    print(f"PASS: no plaintext synthetic identifiers in {scanned} file(s) under "
          f"{', '.join(p.name for p in paths)} ({len(known_identifiers())} known identifiers + shape patterns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
