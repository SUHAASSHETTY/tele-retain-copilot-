"""Supply-chain / provider policy: Gemini is the only LLM provider.

openai / anthropic packages arrive transitively (langmem, deepeval, arize-phoenix), so the policy is
enforced where we control it: our requirements and our code never reference them, and the eval judge
is a Gemini model.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.config import ROOT_DIR

FORBIDDEN_IMPORT = re.compile(r"^\s*(from|import)\s+(openai|anthropic|langchain_openai|langchain_anthropic)\b", re.M)
FORBIDDEN_REQ = re.compile(r"^(openai|anthropic|langchain-openai|langchain-anthropic)\b", re.M | re.I)
CODE_DIRS = ("src", "scripts", "mcp_server", "tests")


def test_no_forbidden_provider_imports_in_our_code():
    offenders = [str(p.relative_to(ROOT_DIR)) for d in CODE_DIRS for p in (ROOT_DIR / d).rglob("*.py")
                 if FORBIDDEN_IMPORT.search(p.read_text())]
    assert offenders == []


def test_requirements_pin_no_other_llm_provider():
    assert not FORBIDDEN_REQ.search((ROOT_DIR / "requirements.txt").read_text())


def test_requirements_are_pinned():
    lines = [l.split("#")[0].strip() for l in (ROOT_DIR / "requirements.txt").read_text().splitlines()]
    unpinned = [l for l in lines if l and "==" not in l and " @ " not in l]
    assert unpinned == []


def test_eval_judge_is_gemini():
    src = (ROOT_DIR / "scripts" / "run_eval.py").read_text()
    assert "GeminiModel(" in src and 'eval_mode="llm"' in src
    assert "isinstance(metric.model, CachedGeminiJudge)" in src


def test_default_models_are_gemini():
    from src.config import settings
    assert settings.gemini_model.startswith("gemini-") and settings.gemini_judge_model.startswith("gemini-")
