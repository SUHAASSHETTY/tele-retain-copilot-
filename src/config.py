"""Central configuration, loaded from environment variables (and `.env` via python-dotenv).

Importing this module never requires a live API key, so graph-logic tests can run offline.
Code paths that actually call Gemini must use `require_google_api_key()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)

# Third-party telemetry off by default: nothing leaves the machine except Gemini API calls.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb

# --- Repository paths (scored layout; do not rename) --------------------------
DATA_DIR = ROOT_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
POLICY_CORPUS_DIR = DATA_DIR / "policy_corpus"
RUNTIME_DIR = DATA_DIR / "runtime"  # SQLite checkpoints, memory store, vector index (gitignored)
SAMPLE_CONTACTS_PATH = DATA_DIR / "sample_contacts.jsonl"
GOLDEN_SET_PATH = DATA_DIR / "golden_set.jsonl"

LOGS_DIR = ROOT_DIR / "logs"
TOOL_CALLS_LOG = LOGS_DIR / "tool_calls.jsonl"
AGENT_ACTIONS_LOG = LOGS_DIR / "agent_actions.jsonl"
MCP_TRANSCRIPT_LOG = LOGS_DIR / "mcp_transcript.jsonl"
MEMORY_TEST_LOG = LOGS_DIR / "memory_test.log"

TRACES_DIR = ROOT_DIR / "traces"
PHOENIX_SPANS_PATH = TRACES_DIR / "phoenix_spans.parquet"

REPORTS_DIR = ROOT_DIR / "reports"
GOLDEN_SIGNALS_PATH = REPORTS_DIR / "golden_signals.json"
DASHBOARD_PNG_PATH = REPORTS_DIR / "dashboard.png"
DASHBOARD_CSV_PATH = REPORTS_DIR / "dashboard_data.csv"
EVAL_REPORT_PATH = REPORTS_DIR / "eval_report.json"

DOCS_DIR = ROOT_DIR / "docs"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


@dataclass(frozen=True)
class Settings:
    # Gemini (the only LLM provider)
    google_api_key: str | None
    gemini_model: str
    gemini_judge_model: str
    gemini_temperature: float
    gemini_input_price_per_mtok: float   # USD per 1M input tokens, for cost estimates
    gemini_output_price_per_mtok: float  # USD per 1M output tokens (incl. thinking)

    # Business rules
    offer_approval_threshold: float  # USD value of a credit/discount above which a human must approve

    # Resilience / loop guards
    llm_timeout_s: float
    tool_timeout_s: float
    max_retries: int
    graph_recursion_limit: int
    rag_max_rewrites: int

    # Retrieval / memory
    embedding_model: str

    # Observability
    phoenix_project_name: str
    phoenix_collector_endpoint: str
    phoenix_port: int
    phoenix_working_dir: Path


def load_settings() -> Settings:
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY") or None,
        gemini_model=_env_str("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_judge_model=_env_str("GEMINI_JUDGE_MODEL", _env_str("GEMINI_MODEL", "gemini-2.5-flash")),
        gemini_temperature=_env_float("GEMINI_TEMPERATURE", 0.0),
        gemini_input_price_per_mtok=_env_float("GEMINI_INPUT_PRICE_PER_MTOK", 0.30),
        gemini_output_price_per_mtok=_env_float("GEMINI_OUTPUT_PRICE_PER_MTOK", 2.50),
        offer_approval_threshold=_env_float("OFFER_APPROVAL_THRESHOLD", 50.0),
        llm_timeout_s=_env_float("LLM_TIMEOUT_S", 60.0),
        tool_timeout_s=_env_float("TOOL_TIMEOUT_S", 20.0),
        max_retries=_env_int("MAX_RETRIES", 3),
        graph_recursion_limit=_env_int("GRAPH_RECURSION_LIMIT", 25),
        rag_max_rewrites=_env_int("RAG_MAX_REWRITES", 2),
        embedding_model=_env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        phoenix_project_name=_env_str("PHOENIX_PROJECT_NAME", "retention-copilot"),
        phoenix_collector_endpoint=_env_str(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
        ),
        phoenix_port=_env_int("PHOENIX_PORT", 6006),
        phoenix_working_dir=ROOT_DIR / _env_str("PHOENIX_WORKING_DIR", ".phoenix"),
    )


settings = load_settings()


def require_google_api_key() -> str:
    """Return the Gemini API key or fail with an actionable message."""
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your Gemini API key."
        )
    return settings.google_api_key
