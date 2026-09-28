"""Export Phoenix spans to traces/phoenix_spans.parquet and summarize them.

Reads from the Phoenix app on PHOENIX_PORT; if none is running, launches the in-process app over the
persisted PHOENIX_WORKING_DIR (where `python -m src.cli run` stored the spans). Columns holding
dicts/lists are serialised to JSON and every string is masked again before writing.

Run: python -m scripts.export_traces
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter

import pandas as pd

from src.config import PHOENIX_SPANS_PATH, ROOT_DIR, settings
from src.guardrails.pii import mask
from src.observability import tracing

KIND_CLASS = {"LLM": "thinking", "AGENT": "acting", "CHAIN": "acting", "GUARDRAIL": "acting",
              "TOOL": "tool", "RETRIEVER": "tool"}


def fetch_spans(settle_s: float = 20.0) -> pd.DataFrame:
    """Fetch the project's spans; Phoenix ingests in the background, so wait until the count settles."""
    from phoenix.client import Client

    tracing.launch_phoenix()
    client = Client(base_url=tracing.phoenix_url())
    df, last, waited = None, -1, 0.0
    while waited <= settle_s:
        try:
            df = client.spans.get_spans_dataframe(project_identifier=settings.phoenix_project_name,
                                                  limit=200_000, timeout=180)
        except Exception:  # project not created yet
            df = None
        n = 0 if df is None else len(df)
        if n and n == last:
            break
        last = n
        time.sleep(2.0)
        waited += 2.0
    return df


def _cell(v):
    if isinstance(v, (dict, list, tuple)):
        return mask(json.dumps(v, default=str, ensure_ascii=False))
    if hasattr(v, "tolist") and not isinstance(v, str):  # numpy arrays
        return mask(json.dumps(v.tolist(), default=str))
    if isinstance(v, str):
        return mask(v)
    return v


def to_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index() if "context.span_id" not in df.columns else df.copy()
    out = _expand_dict_columns(out, ("attributes.copilot", "attributes.metadata"))
    out["copilot_agent"] = _agent_column(out)
    out["span_class"] = out["span_kind"].fillna("UNKNOWN").map(lambda k: KIND_CLASS.get(k, "other"))
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(_cell)
    PHOENIX_SPANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PHOENIX_SPANS_PATH, index=False)
    return out


def _as_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.startswith("{"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return None


def _expand_dict_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Phoenix nests dotted attributes (e.g. copilot.run_id) as dicts; give each key its own column."""
    for col in cols:
        if col not in df.columns:
            continue
        dicts = df[col].map(_as_dict)
        keys = sorted({k for d in dicts.dropna() for k in d})
        for k in keys:
            name = f"{col}.{k}"
            if name not in df.columns:
                df[name] = dicts.map(lambda d, k=k: d.get(k) if d else None)
    return df


def _agent_column(df: pd.DataFrame) -> pd.Series:
    """Acting agent: the run-context agent stamped on the span, else the LangGraph node."""
    agent = df.get("attributes.copilot.agent", pd.Series([None] * len(df), index=df.index))
    node = df.get("attributes.metadata.langgraph_node", pd.Series([None] * len(df), index=df.index))
    return agent.where(agent.notna() & (agent != ""), node)


def summarize(df: pd.DataFrame) -> dict:
    kind = df["span_kind"].fillna("UNKNOWN")
    agents = df["copilot_agent"] if "copilot_agent" in df.columns else _agent_column(df)
    latency = (pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])).dt.total_seconds() * 1000
    tools = df[kind == "TOOL"]["name"].value_counts().to_dict()
    runs = df.get("attributes.copilot.run_id")
    return {
        "spans": len(df), "traces": df["context.trace_id"].nunique(),
        "runs": int(runs.dropna().nunique()) if runs is not None else 0,
        "by_kind": kind.value_counts().to_dict(),
        "by_class": kind.map(lambda k: KIND_CLASS.get(k, "other")).value_counts().to_dict(),
        "by_agent": agents.fillna("(graph/root)").value_counts().to_dict(),
        "tool_spans_by_name": tools,
        "latency_ms_p50_by_kind": latency.groupby(kind).median().round(1).to_dict(),
        "spans_with_latency": int(latency.notna().sum()),
    }


def export_spans(path=None) -> dict:
    global PHOENIX_SPANS_PATH
    if path is not None:
        PHOENIX_SPANS_PATH = path
    df = fetch_spans()
    if df is None or df.empty:
        print("no spans found in Phoenix project", settings.phoenix_project_name)
        return {}
    to_parquet(df)
    back = pd.read_parquet(PHOENIX_SPANS_PATH)  # confirm by loading what was written
    s = summarize(back)
    print(f"{PHOENIX_SPANS_PATH.relative_to(ROOT_DIR)}: {s['spans']} spans, {s['traces']} traces, {s['runs']} runs")
    print(f"  by kind:  {s['by_kind']}")
    print(f"  by class: {s['by_class']}")
    print(f"  by agent: {s['by_agent']}")
    print(f"  tool spans: {s['tool_spans_by_name']}")
    print(f"  p50 latency ms by kind: {s['latency_ms_p50_by_kind']}")
    return s


if __name__ == "__main__":
    sys.exit(0 if export_spans() else 1)
