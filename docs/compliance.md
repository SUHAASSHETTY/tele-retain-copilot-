# Compliance mapping

**Status and caveats (read first).**
- This is a hackathon prototype running on synthetic data only. No real personal data is processed and the
  system is not deployed.
- The mapping records which obligations would plausibly apply to a production deployment of this design,
  and what the project already provides as evidence.
- This is an engineering self-assessment, **not legal advice**. Applicability must be confirmed by counsel
  for the actual deployer, jurisdiction and use.

**Applicability summary.**
- **EU AI Act (Regulation (EU) 2024/1689).** A customer-service and retention assistant is not among the
  Annex III high-risk use cases as we read them: it does no creditworthiness assessment, and it does not
  price life or health insurance. The most plausible category is a limited-risk system that interacts with
  natural persons, which carries the Art. 50 transparency duty. General-purpose AI model obligations
  (Art. 53) fall on the model provider (Google), not on this deployer.
- **NIST AI RMF 1.0.** Voluntary; used as the organising framework for the risk register (`docs/risk-register.md`).
- **India DPDP Act 2023.** Would apply if the copilot processed digital personal data of Data Principals in
  India. With synthetic data it does not apply today; the rows below show readiness and gaps. We reference
  sections of the Act as enacted and have not assessed the implementing Rules.

| Framework | Obligation / expectation | How addressed | Evidence artifact | Status |
|---|---|---|---|---|
| EU AI Act | Art. 50(1): tell people they are interacting with an AI system | Chat mode shows a disclosure before the conversation; replies offer a human | `src/cli.py::AI_DISCLOSURE`, `src/cli.py::cmd_chat` | Partial: CLI only; no production UI |
| EU AI Act | Art. 5(1)(a)-(b): no manipulative or exploitative techniques that distort behaviour | Retention policy forbids offers conditional on withdrawing complaints and forbids obstructing cancellation; one retention attempt only | `data/policy_corpus/POL-RET-004.md` (POL-RET-004 §1.4), `data/policy_corpus/POL-CAN-001.md` (POL-CAN-001 §1.2, POL-CAN-001 §1.3) | Design intent; not audited for manipulation |
| EU AI Act | Art. 6 / Annex III: high-risk classification check | Self-assessed as not high-risk (see summary); no creditworthiness or insurance pricing | this document | Needs legal confirmation |
| EU AI Act | Art. 4: AI literacy of staff | Not addressed in code; guidance for agents is limited to the model card | `docs/model-card.md` | Gap (organisational) |
| EU AI Act | Human oversight (Art. 14 is for high-risk systems; applied voluntarily) | Offers above threshold stop at a human approval interrupt; decisions audited with actor type | `src/agents/human_loop.py::human_approval`, approval_granted `evidence/governance/logs/agent_actions.jsonl:L30` | Implemented (voluntary) |
| NIST AI RMF | GOVERN: policies, accountability, roles | Written policy corpus incl. AI-use policy; owners per risk; audit trail with actor / actor_type | `data/policy_corpus/POL-GOV-001.md`, `docs/risk-register.md`, `src/audit/audit_middleware.py::audit` | Implemented (prototype roles) |
| NIST AI RMF | MAP: context, intended use, limitations | Model / system card with intended users, out-of-scope uses and limitations | `docs/model-card.md` | Implemented |
| NIST AI RMF | MEASURE: test and evaluate, incl. security | Golden-set eval, red team, routing / loop / contract tests, golden signals from traces | `scripts/run_eval.py`, `scripts/run_redteam.py`, `tests/test_routing.py`, `scripts/golden_signals.py` | Partial: LLM-judge metrics not run (F5) |
| NIST AI RMF | MANAGE: prioritise, respond, recover | Risk register with residual risk; failure analysis with root cause and verified fixes; escalation to humans on failure | `docs/risk-register.md`, `docs/failure-analysis.md`, `src/agents/human_loop.py::escalate` | Implemented |
| DPDP Act 2023 | s.4-s.6: lawful purpose, notice and consent | Not applicable to synthetic data; no consent or notice flow exists | n/a | Gap before real data |
| DPDP Act 2023 | s.6(1) data limited to what is necessary (minimisation) | Tools return masked identifiers; workers see only scoped context | `mcp_server/server.py::get_account`, `src/context/isolate.py::view` | Implemented |
| DPDP Act 2023 | s.8(3): completeness and accuracy of data used for decisions | Decisions use the system of record through tools, never model memory; policy checks are deterministic | `mcp_server/server.py::evaluate_offer` | Implemented (synthetic source) |
| DPDP Act 2023 | s.8(5): reasonable security safeguards to prevent breach | Masking in every log and trace, access control on every tool, PII scan on every regeneration, no secrets in the project (only `.env.example` placeholders) | `src/guardrails/pii.py::mask_obj`, `mcp_server/server.py::_authorized_call`, `scripts/check_pii_leaks.py`, `.env.example` | Implemented for the prototype |
| DPDP Act 2023 | s.8(6): intimate personal data breaches to the Board and affected persons | No breach-notification process | n/a | Gap |
| DPDP Act 2023 | s.8(7) / s.12: erase data when the purpose is served or on request | Per-customer erasure of long-term memory; runtime stores are separate local files, excluded from the project package | `src/memory/long_term.py::LongTermMemory.forget_customer` | Partial: no retention schedule or checkpoint purge job |
| DPDP Act 2023 | s.13: grievance redressal | Complaints and escalations create tickets routed to human queues | `src/agents/human_loop.py::escalate`, `mcp_server/server.py::create_escalation_ticket` | Partial: no published grievance officer |

**What would change for a real deployment.**
- Legal classification review.
- A DPIA or privacy impact assessment.
- Consent and notice flows (DPDP s.5-s.6).
- Breach response procedures (s.8(6)).
- A retention and erasure schedule covering checkpoints and traces.
- Staff AI-literacy training (AI Act Art. 4).
- A production-grade disclosure in the customer-facing UI.
