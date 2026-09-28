# Output-risk classification

Every resolution is tiered by `src/guardrails/output_risk.py::classify_output_risk`. It runs in the
graph's last node (`src/guardrails/nodes.py::output_guard`), and the tier, gate and whether the gate was
satisfied are written to the audit trail (`contact_resolved` records, `details.risk_tier`) and to the
turn's Phoenix span (`copilot.output_risk_tier`). Tiers are computed from structured state (outcome,
offer proposal, approval, guard flags), never from the drafted text.

| Tier | What it covers | Examples | Gate before release |
|---|---|---|---|
| **Low** | Informational answers that change nothing | Bill explanation, plan/allowance facts, policy lookup, clarifying question, out-of-scope decline | Output guard only (`src/guardrails/output_guard.py::check_output`): PII masking, unknown-citation stripping |
| **Medium** | Within-policy changes or commitments at or below the auto-approval threshold | Plan change recommendation, complaint logged with SLA, duplicate-charge credit, non-monetary data boost | Deterministic policy check `mcp_server/server.py::evaluate_offer` plus audit record; output guard blocks any figure above the checked value |
| **High** | Money above the threshold, fee waivers, anything during a cancellation, refusals and human hand-offs | Retention discount above the threshold, fee waiver, cancellation retention offer, injection / cross-customer refusal, fraud / safety / repeat-complaint escalation | Above-threshold offers and waivers: LangGraph interrupt for human approval (`src/agents/human_loop.py::human_approval`); prohibited requests: refusal (`src/agents/resolution_agent.py::decide_outcome`); fraud / safety / regulator / repeat complaint: escalation ticket (`src/agents/human_loop.py::escalate`). The output guard blocks a pending offer described as applied. |

**How high is gated.**
- Routing sends an `offer_pending_approval` outcome to the approval interrupt
  (`src/agents/resolution_agent.py::route_after_resolution`). The graph cannot reach the output guard
  without a human decision.
- `gate_satisfied` is false until that decision exists. Tested by
  `tests/test_guardrails.py::test_high_risk_offer_gate_requires_human_decision` and
  `tests/test_routing.py::test_over_threshold_offer_routes_to_human_approval`.
- The threshold is `src/policy_limits.py::APPROVAL_THRESHOLD` (POL-RET-003 §1.2); discounts above 25% are
  refused outright (POL-RET-002 §2.3).

## Real samples (from the audit log)
Records are taken verbatim from `evidence/governance/logs/agent_actions.jsonl`, a frozen copy of
`logs/agent_actions.jsonl` from a full `python -m scripts.regenerate_evidence` run
(`evidence/governance/manifest.json`). Identifiers and amounts are already masked in the log.

**Low:** CT-0001 bill explanation, contact_resolved `evidence/governance/logs/agent_actions.jsonl:L6` (risk_tier low, gate output_guard).

**Medium:** CT-0002 duplicate-charge credit within threshold, contact_resolved `evidence/governance/logs/agent_actions.jsonl:L11` (risk_tier medium);
its policy check is offer_proposed `evidence/governance/logs/agent_actions.jsonl:L8` (within_limits).

**High, refused:** CT-0009 prompt injection, guardrail_block `evidence/governance/logs/agent_actions.jsonl:L39` then contact_resolved `evidence/governance/logs/agent_actions.jsonl:L42` (refuse, risk_tier high).

**High, human approval (CT-0006, run 2d2e42f7-dcab-4b04-8b89-e0245e346065, trace 6cdbce448460572aa626435c68fe0333):**
the customer asked for half off. The four records below are verbatim.
- offer_blocked `evidence/governance/logs/agent_actions.jsonl:L26`: 50% refused with POL-RET-002 §2.3 / POL-RET-004 §1.1.
- approval_requested `evidence/governance/logs/agent_actions.jsonl:L29`: 20% for 6 months exceeds the threshold (request_approval span 2bf346b0e0b47c60).
- approval_granted `evidence/governance/logs/agent_actions.jsonl:L30` (human_approval span 119414006dd8c37b).
- contact_resolved `evidence/governance/logs/agent_actions.jsonl:L32`: risk_tier high, gate human_approval_interrupt, gate_satisfied true.

```json
{"actor": "retention_offer_agent", "actor_type": "agent", "action": "offer_blocked", "tool": "check_offer_eligibility", "decision": "blocked", "reason": "Discounts above 25% are prohibited.", "policy_ref": ["POL-RET-002 §2.3", "POL-RET-004 §1.1"], "details": {"source": "customer_request", "offer_type": "discount_pct", "value": 50.0, "months": 6}}
{"actor": "human_approval", "actor_type": "agent", "action": "approval_requested", "tool": "interrupt", "decision": "pending", "reason": "Offer value exceeds the $**.** auto-approval threshold.", "policy_ref": ["POL-RET-002 §1.3", "POL-RET-002 §2.2", "POL-RET-003 §1.2"]}
{"actor": "system:cli-auto-approve", "actor_type": "system", "action": "approval_granted", "tool": "interrupt", "decision": "granted", "reason": "non-interactive run", "policy_ref": "POL-RET-003 §1.2"}
{"actor": "output_guard", "actor_type": "system", "action": "contact_resolved", "decision": "offer", "details": {"risk_tier": "high", "risk_gate": "human_approval_interrupt", "gate_satisfied": true}}
```
(The four records are shortened here for readability: timestamps, run_id, session_id and the repeated
details are omitted. The full lines are at the cited line numbers.)

**Note on the approver.** In reproducible runs the approval comes from the scripted stand-in
`system:cli-auto-approve` (the `--auto-approve` flag of `src/cli.py`) and is recorded with `actor_type`
system, not as a person. With interactive approval (`src/cli.py::approver`), the same record carries
`human:<approver>` and `actor_type` human. The same gate also rejects: red-team runs record approval_denied
from `system:redteam-auto-reject`, e.g. `evidence/governance/logs/agent_actions.jsonl:L98` (approval_denied).
