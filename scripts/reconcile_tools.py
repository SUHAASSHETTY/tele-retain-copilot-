"""Reconcile tool names across code, logs and traces. Exits 1 on any mismatch.

Registry (from code):
  - MCP tools: FastMCP.list_tools() on mcp_server/server.py
  - local tools: policy_rag (src/tools/rag_tool.py), manage_memory (LangMem, src/memory/long_term.py)
Checks, both directions:
  1. every tool_name in logs/tool_calls.jsonl is a registered tool
  2. every TOOL span name in traces/phoenix_spans.parquet is a registered tool
  3. every tool called by agent code (`mcp.call("...")`) is a registered MCP tool
  4. every tool in logs/mcp_transcript.jsonl is a registered MCP tool
  5. every registered tool appears in the tool log AND in the exported spans
Writes reports/tool_reconciliation.json.

Run: python -m scripts.reconcile_tools
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone

import pandas as pd

from src.config import MCP_TRANSCRIPT_LOG, PHOENIX_SPANS_PATH, REPORTS_DIR, ROOT_DIR, TOOL_CALLS_LOG

CALL_RE = re.compile(r"""\.call\(\s*["']([a-z_]+)["']""")


async def registry() -> dict[str, str]:
    from mcp_server.server import mcp
    from src.tools.rag_tool import policy_rag_tool

    tools = {t.name: "mcp" for t in await mcp.list_tools()}
    tools[policy_rag_tool.name] = "local"
    tools["manage_memory"] = "langmem"  # create_manage_memory_tool default name, used in src/memory/long_term.py
    return tools


def code_calls() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted((ROOT_DIR / "src").rglob("*.py")):
        for name in CALL_RE.findall(path.read_text()):
            found.setdefault(name, []).append(str(path.relative_to(ROOT_DIR)))
    return found


def main() -> int:
    reg = asyncio.run(registry())
    logged = {json.loads(l)["tool_name"] for l in TOOL_CALLS_LOG.read_text().splitlines() if l.strip()}
    transcript = {json.loads(l)["name"] for l in MCP_TRANSCRIPT_LOG.read_text().splitlines()
                  if l.strip() and json.loads(l)["method"] == "tools/call"}
    spans: set[str] = set()
    if PHOENIX_SPANS_PATH.exists():
        df = pd.read_parquet(PHOENIX_SPANS_PATH, columns=["name", "span_kind"])
        spans = set(df[df["span_kind"] == "TOOL"]["name"])
    calls = code_calls()
    mcp_tools = {n for n, src in reg.items() if src == "mcp"}

    checks = {
        "logged_tools_not_in_code": sorted(logged - reg.keys()),
        "span_tools_not_in_code": sorted(spans - reg.keys()),
        "code_calls_not_registered": sorted(set(calls) - mcp_tools),
        "transcript_tools_not_in_code": sorted(transcript - mcp_tools),
        "registered_tools_never_logged": sorted(reg.keys() - logged),
        "registered_tools_without_spans": sorted(reg.keys() - spans) if spans else ["(no traces exported)"],
    }
    ok = not any(checks.values())
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "generator": "scripts/reconcile_tools.py", "status": "PASS" if ok else "FAIL",
              "registry": reg, "code_calls": calls, "logged_tools": sorted(logged),
              "transcript_tools": sorted(transcript), "span_tools": sorted(spans), "checks": checks}
    out = REPORTS_DIR / "tool_reconciliation.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"registry ({len(reg)}): {reg}")
    print(f"logged: {sorted(logged)}\nspans:  {sorted(spans)}\ncode calls: {sorted(calls)}")
    for name, missing in checks.items():
        print(f"  [{'PASS' if not missing else 'FAIL'}] {name}: {missing or '-'}")
    print(f"{'PASS' if ok else 'FAIL'} -> {out.relative_to(ROOT_DIR)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
