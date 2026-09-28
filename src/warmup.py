"""Process warm-up: load heavy local models before the first customer turn.

Without this the first turn of every process paid for loading the MiniLM embedder, opening the Chroma
index and loading Presidio/spaCy (~13 s), which showed up as a latency outlier on the first
copilot.turn span (docs/failure-analysis.md, F3). Called by the CLI, eval and red-team runners
before any contact is processed, outside any turn span.
"""

from __future__ import annotations

import time


def warm_up() -> dict:
    timings = {}
    t0 = time.perf_counter()
    from src.tools.rag_tool import embed, get_collection

    embed(["warm-up"])
    get_collection()
    timings["embeddings_and_index_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    t1 = time.perf_counter()
    from src.guardrails.pii import detect_pii

    detect_pii("warm-up CUST-000000")
    timings["presidio_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    return timings
