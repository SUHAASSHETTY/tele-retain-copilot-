"""Loop / cascade guards: runaway loops end in a graceful escalation, never an exception.

1. max_steps: a stub intent worker that never makes progress keeps the supervisor looping; the
   step guard in `allowed_next` stops it after max_steps and routes to `escalate`.
2. recursion_limit: with the step guard disabled, LangGraph's recursion_limit fires; run_turn
   catches GraphRecursionError and returns an escalation (audited as loop_limit).
3. RAG retry loop: a judge that always grades evidence as weak is capped at RAG_MAX_REWRITES.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import src.graph as graph_mod
from src.config import settings
from src.tools.rag_tool import Draft, Grade, Rewrite, run_policy_rag


class FakeMCP:
    """Stands in for the MCP BoundSession; only the escalation tool is ever reached."""

    def __init__(self):
        self.calls = []
        self.owner = self

    def issue_session_token(self, customer_id, session_id):
        return f"{session_id}:{customer_id}:test"

    async def call(self, name, **args):
        self.calls.append(name)
        if name == "create_escalation_ticket":
            return {"ok": True, "ticket_id": "ESC-TEST0001", "queue": "care-tier2", "priority": args.get("priority")}
        raise AssertionError(f"unexpected tool {name}")


def _stuck_graph(monkeypatch):
    calls = {"n": 0}

    async def stuck_intent_agent(state, runtime):
        calls["n"] += 1
        return {}  # never classifies: the supervisor keeps sending work back here

    monkeypatch.setattr(graph_mod, "intent_agent", stuck_intent_agent)
    return graph_mod.build_graph(checkpointer=InMemorySaver()), calls


async def _approve(_):
    return {"approved": False}


CONTACT = {"contact_id": "LOOP-1", "customer_id": "CUST-000635", "session_id": "SES-LOOP", "turns": ["help"]}


@pytest.mark.asyncio
async def test_max_steps_stops_runaway_loop_with_escalation(monkeypatch, isolated_logs):
    app, calls = _stuck_graph(monkeypatch)
    mcp = FakeMCP()
    result = await graph_mod.run_contact(app, mcp, CONTACT, approve=_approve, use_llm=False, max_steps=4)
    state = result["turns"][-1]
    assert calls["n"] == 3                                    # steps 1-3 dispatch the stuck worker
    assert state["step_count"] == 4                           # step 4 = max_steps: guard routes to escalate
    assert state["next_worker"] == "escalate"                 # node decision and routing edge agree
    assert state["resolution"]["outcome"] == "escalate"
    assert "step limit" in state["resolution"]["reason"]
    assert mcp.calls == ["create_escalation_ticket"]
    assert "ESC-TEST0001" in state["final_response"]
    actions = [json.loads(l)["action"] for l in (isolated_logs / "agent_actions.jsonl").read_text().splitlines()]
    assert "escalation" in actions and "contact_resolved" in actions


@pytest.mark.asyncio
async def test_recursion_limit_is_caught_and_escalates(monkeypatch, isolated_logs):
    app, calls = _stuck_graph(monkeypatch)
    monkeypatch.setattr(graph_mod, "settings", dataclasses.replace(settings, graph_recursion_limit=12))
    result = await graph_mod.run_contact(app, FakeMCP(), CONTACT, approve=_approve, use_llm=False,
                                         max_steps=1000)          # step guard effectively off
    state = result["turns"][-1]
    assert state["loop_guard"] == "recursion_limit"
    assert state["resolution"]["outcome"] == "escalate"
    assert calls["n"] < 12
    records = [json.loads(l) for l in (isolated_logs / "agent_actions.jsonl").read_text().splitlines()]
    assert any(r["action"] == "loop_limit" and r["decision"] == "escalated" for r in records)


class AlwaysWeakJudge:
    name = "custom"

    def __init__(self):
        self.rewrites = 0

    async def grade(self, question, hits):
        return Grade(relevant_citations=[], sufficient=False, reason="never enough")

    async def rewrite(self, question, tried, hits):
        self.rewrites += 1
        return Rewrite(query=f"{question} attempt {self.rewrites}")

    async def answer(self, question, hits):  # pragma: no cover - never reached
        return Draft(answer="", citations=[])


@pytest.mark.asyncio
async def test_rag_retry_loop_is_bounded():
    judge = AlwaysWeakJudge()
    out = await run_policy_rag("When does a credit need approval?", judge=judge)
    assert judge.rewrites == settings.rag_max_rewrites
    assert out.attempts == settings.rag_max_rewrites + 1
    assert out.status == "no_relevant_policy" and out.citations == []
