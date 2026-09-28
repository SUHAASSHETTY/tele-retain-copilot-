# Risk register: Customer Service & Retention Copilot (BC-AAIE-HACK-17)

Scope: the synthetic-data prototype in this repository. Categories use the OWASP Top 10 for LLM
Applications (2025) ids and the NIST AI RMF 1.0 core functions (GOVERN / MAP / MEASURE / MANAGE).
Likelihood and impact are the team's qualitative judgement for a production deployment of this
design (H / M / L). Residual risk is after the cited controls.

**How to read the citations.**
- Controls are cited as `file::symbol`; `scripts/verify_citations.py` checks that each one exists.
- Runtime evidence comes from the frozen snapshot `evidence/governance/` (manifest
  `evidence/governance/manifest.json`). That snapshot is a full `python -m scripts.regenerate_evidence`
  run in deterministic mode: rules/templates, no Gemini calls.
- Failure write-ups are in `docs/failure-analysis.md`.

Owners are roles, because this is a prototype with no staffed organisation:

| Owner | Area |
|---|---|
| **AI Eng** | Copilot engineering lead |
| **Sec** | Security |
| **DPO** | Data protection officer |
| **RetOps** | Retention operations (approvers) |
| **Platform** | Infrastructure / cost |

## Summary

| ID | Risk | Category (OWASP / NIST) | L | I | Residual | Owner |
|---|---|---|---|---|---|---|
| R01 | Prompt injection via customer text | LLM01 / MANAGE, MEASURE | H | H | Medium | Sec |
| R02 | PII / billing data leakage in answers, logs or traces | LLM02 / MANAGE, GOVERN | M | H | Low | DPO |
| R03 | Excessive agency: unauthorised discounts or credits | LLM06 / MANAGE | M | H | Low | RetOps |
| R04 | Hallucinated or misquoted policy | LLM09 / MEASURE | M | M | Medium | AI Eng |
| R05 | Cross-customer data access | LLM02, LLM06 / MANAGE | M | H | Low | Sec |
| R06 | Cost runaway (token / request consumption) | LLM10 / MANAGE | M | M | Medium | Platform |
| R07 | Loop / cascade runaway in the agent graph | LLM10 / MANAGE, MEASURE | L | M | Low | AI Eng |
| R08 | Model outage, quota exhaustion or deprecation | n/a (availability) / MANAGE | H | M | Medium | Platform |
| R09 | Supply chain: dependencies and provider drift | LLM03 / GOVERN, MAP | M | M | Medium | Sec |
| R10 | Over-reliance on copilot output by agents or customers | LLM09 / GOVERN, MAP | M | M | Medium | RetOps |
| R11 | Long-term memory poisoning | LLM04 / MANAGE | M | M | Low | AI Eng |
| R12 | System prompt / policy internals leakage | LLM07 / MANAGE | L | L | Low | Sec |
| R13 | Observability egress (trace data, tool phoning home) | LLM02 / GOVERN | L | M | Low | DPO |

## Details

### R01: Prompt injection via customer text
- **Likelihood / impact.** H / H: customers can type anything, and a successful override could grant
  offers or change behaviour.
- **Mitigations → controls.**
  - Customer text is only ever passed to a model as labelled data, in a separate message from the
    instructions: `src/context/quarantine.py::untrusted_prompt`, `src/context/quarantine.py::quarantine`.
  - Injection patterns (including paraphrases and approval-bypass requests) are detected and the turn is
    blocked before any worker runs: `src/context/quarantine.py::detect_injection`,
    `src/guardrails/input_guard.py::check_input`, wired as the first graph node in `src/guardrails/nodes.py::input_guard`.
  - Even if detection misses, offers are decided by a deterministic policy check, not by the model:
    `mcp_server/server.py::evaluate_offer`.
  - Tests: `tests/test_guardrails.py::test_input_guard_blocks`,
    `tests/test_guardrails.py::test_paraphrased_overrides_are_blocked`,
    `tests/test_context.py::test_customer_text_is_data_not_instructions`.
- **Evidence.**
  - Blocked injection: guardrail_block record `evidence/governance/logs/agent_actions.jsonl:L39`.
  - That turn's refusal: contact_resolved record `evidence/governance/logs/agent_actions.jsonl:L42`.
  - Trace a4191b48a3d28f8a0674513d5b98e28d, input guardrail span 0a03c85d3e82f822.
  - Red team: `evidence/governance/reports/redteam_results.json` (14 attacks, 0 harmful outcomes, detection 0.929).
- **Residual: Medium.** Detection is pattern-based. A novel paraphrase can still get through the input guard
  (see `docs/failure-analysis.md` F2), with harm limited by the policy check and the approval gate.

### R02: PII / billing data leakage in answers, logs or traces
- **Likelihood / impact.** M / H.
- **Mitigations → controls.**
  - Tools return masked identifiers only: `mcp_server/server.py::get_account`.
  - Card numbers are redacted before state is checkpointed: `src/graph.py::redact_at_ingest`.
  - The output guard masks any PII found by Presidio (custom recognizers for CUST-, ACC- and synthetic
    phones) and regexes: `src/guardrails/output_guard.py::check_output`,
    `src/guardrails/pii.py::presidio_analyzer`.
  - Every logger masks through `src/guardrails/pii.py::mask_obj`: tool log
    `src/tools/logging_middleware.py::log_tool_call`, audit `src/audit/audit_middleware.py::audit`,
    MCP transcript `src/tools/mcp_client.py::TranscriptWriter`.
  - Spans are masked before export: `src/observability/tracing.py::MaskingSpanExporter`.
  - A leak scan runs in every evidence regeneration: `scripts/check_pii_leaks.py`.
- **Evidence.**
  - `evidence/governance/reports/regenerate_summary.json`: pii step PASS.
  - A real leak that the scan caught, and its fix: `docs/failure-analysis.md` F4.
  - Test: `tests/test_guardrails.py::test_output_guard_masks_pii_with_presidio`.
- **Residual: Low.** The customer's own bill amounts are deliberately shown to the verified account holder;
  amounts are masked in logs and traces only.

### R03: Excessive agency (unauthorised discounts or credits)
- **Likelihood / impact.** M / H.
- **Mitigations → controls.**
  - The offer agent can only propose what the eligibility tool allows:
    `src/agents/retention_offer_agent.py::retention_offer_agent`, `mcp_server/server.py::evaluate_offer`,
    limits in `src/policy_limits.py::APPROVAL_THRESHOLD`.
  - Offers above the threshold pause the graph for a human: `src/agents/human_loop.py::human_approval`
    (LangGraph `interrupt`), routed by `src/agents/resolution_agent.py::route_after_resolution`.
  - The output guard blocks replies stating a larger discount or credit than was checked, or calling a
    pending offer applied: `src/guardrails/output_guard.py::check_output`.
- **Evidence.** CT-0006, run 2d2e42f7-dcab-4b04-8b89-e0245e346065:
  - The customer's 50% demand was blocked: offer_blocked `evidence/governance/logs/agent_actions.jsonl:L26`.
  - 20% for 6 months went to approval: approval_requested `evidence/governance/logs/agent_actions.jsonl:L29`.
  - The decision was recorded as approval_granted `evidence/governance/logs/agent_actions.jsonl:L30`, attributed to
    `system:cli-auto-approve` (the non-interactive stand-in used for reproducible runs, not a person).
  - Trace 6cdbce448460572aa626435c68fe0333, human_approval span 119414006dd8c37b.
  - Tests: `tests/test_routing.py::test_over_threshold_offer_routes_to_human_approval`,
    `tests/test_tool_contracts.py::test_over_limit_offers_are_blocked_with_policy_ref`.
- **Residual: Low.** In reproducible runs approvals are scripted; production must use the interactive
  approver (`src/cli.py::approver`).

### R04: Hallucinated or misquoted policy
- **Likelihood / impact.** M / M.
- **Mitigations → controls.**
  - Agentic RAG grades retrieved clauses and drops any citation not in graded evidence:
    `src/tools/rag_tool.py::build_rag_graph`.
  - Resolution drafting may only cite a whitelist of allowed clauses; otherwise the template is used:
    `src/agents/resolution_agent.py::allowed_citations`.
  - The output guard strips clause IDs that do not exist in the corpus: `src/guardrails/output_guard.py::known_citations`.
  - Outcomes are decided by rules, not the model: `src/agents/resolution_agent.py::decide_outcome`.
- **Evidence.**
  - `evidence/governance/reports/eval_report.json`: citation_validity 1.0, policy_ref_recall 1.0 on 18 golden cases.
- **Residual: Medium.** LLM-as-judge coverage is partial: hallucination was judged on 2 of 18 cases
  (both pass) and faithfulness was not run (`docs/failure-analysis.md` F5). The deterministic metrics come
  from template answers.

### R05: Cross-customer data access
- **Likelihood / impact.** M / H.
- **Mitigations → controls.**
  - An HMAC session token binds each tool call to the authenticated caller:
    `mcp_server/auth.py::issue_token`, `mcp_server/auth.py::verify_token`.
  - Server-side account-holder check on every customer tool: `mcp_server/server.py::_authorized_call`.
  - The token is injected by the client and hidden from the model, which cannot supply or forge it:
    `src/tools/mcp_client.py::AuthTranscriptInterceptor`, `src/tools/mcp_client.py::_prepare_tools`.
  - The input guard also blocks explicit requests for another customer's data: `src/guardrails/input_guard.py::check_input`.
- **Evidence.**
  - data_access_refused `evidence/governance/logs/agent_actions.jsonl:L1` (AUTHZ_DENIED, run 72c31696-bd13-40b3-92e3-a24e88395724).
  - CT-0010: guardrail_block `evidence/governance/logs/agent_actions.jsonl:L43`, trace 3f24fc8bcaa8d2addd9b8402f6c8fe90,
    input guardrail span aa66c3cea49ceb74.
  - Test: `tests/test_tool_contracts.py::test_tool_error_paths_are_structured`.
- **Residual: Low.** Upstream authentication of the caller is simulated: the contact's `customer_id` stands in
  for a real login (`src/cli.py` docstring).

### R06: Cost runaway
- **Likelihood / impact.** M / M.
- **Mitigations → controls.**
  - A shared client-side rate limiter for every Gemini call: `src/llm.py::RATE_LIMITER` (`GEMINI_RPM`).
  - Bounded steps per turn: `src/agents/supervisor.py::allowed_next`.
  - Bounded RAG rewrites: `src/tools/rag_tool.py::build_rag_graph` (`RAG_MAX_REWRITES`).
  - Timeouts and bounded retries: `src/resilience.py::resilient_call`.
  - Preflight disables Gemini when it is unusable, avoiding failed paid calls: `src/llm.py::llm_preflight`.
  - Eval judge responses are cached: `scripts/run_eval.py::CachedGeminiJudge`.
  - Cost is measured from traces: `scripts/golden_signals.py`, prices dated in `src/observability/cost.py::PRICE_TABLE`.
- **Evidence.** `evidence/governance/reports/golden_signals.json`: 0 LLM calls in the deterministic snapshot,
  so the cost of a Gemini-driven run is not yet measured.
- **Residual: Medium.** There is no per-run token budget and no max-output-token cap.

### R07: Loop / cascade runaway
- **Likelihood / impact.** L / M.
- **Mitigations → controls.**
  - A step limit routes to escalation: `src/agents/supervisor.py::allowed_next`.
  - The recursion limit is caught and escalated: `src/graph.py::_run_turn`.
  - The RAG loop is capped and stops on a repeated query: `src/tools/rag_tool.py::build_rag_graph`.
- **Evidence.**
  - Tests: `tests/test_loops.py::test_max_steps_stops_runaway_loop_with_escalation`,
    `tests/test_loops.py::test_recursion_limit_is_caught_and_escalates`, `tests/test_loops.py::test_rag_retry_loop_is_bounded`.
  - No loop_limit records in the snapshot: `evidence/governance/failure_candidates.json`.
- **Residual: Low.**

### R08: Model outage, quota exhaustion or deprecation
- **Likelihood / impact.** H / M. Observed: `gemini-2.5-flash` returned 404 for this key, and the free tier
  allows 20 requests per day per model.
- **Mitigations → controls.**
  - Transient errors (429/503/DNS) are retried with backoff: `src/resilience.py::is_transient`.
  - Every model-backed step falls back to deterministic rules or templates, and the fallback is labelled
    in the engine log: `src/agents/intent_agent.py::classify_rules`,
    `src/agents/resolution_agent.py::template_message`, `src/tools/rag_tool.py::HeuristicJudge`.
  - Preflight check: `src/llm.py::llm_preflight`.
  - The eval records the judge as not run instead of scoring: `scripts/run_eval.py::judge_preflight`.
- **Evidence.**
  - `evidence/governance/reports/eval_report.json`: judge status not_run.
  - `docs/failure-analysis.md` F5.
- **Residual: Medium.** Quality metrics that depend on the model are unavailable during an outage.

### R09: Supply chain
- **Likelihood / impact.** M / M.
- **Mitigations → controls.**
  - All versions are pinned and resolved together, with the conflict pins explained: `requirements.txt`.
  - Tests enforce that Gemini is the only provider (no OpenAI/Anthropic imports or requirements; the judge
    is Gemini): `tests/test_provider_policy.py::test_no_forbidden_provider_imports_in_our_code`,
    `tests/test_provider_policy.py::test_requirements_are_pinned`, `tests/test_provider_policy.py::test_eval_judge_is_gemini`.
  - Third-party telemetry is disabled by default: `src/config.py`.
- **Residual: Medium.** `openai` / `anthropic` packages are installed transitively (langmem, deepeval,
  arize-phoenix) even though unused; there is no automated vulnerability scan.

### R10: Over-reliance on copilot output
- **Likelihood / impact.** M / M.
- **Mitigations → controls.**
  - Every policy statement cites its clause, and cited clause IDs are validated by the output guard
    (`src/guardrails/output_guard.py::check_output`).
  - High-risk outputs are tiered and gated: `src/guardrails/output_risk.py::classify_output_risk`.
  - Ambiguity triggers one clarifying question, then escalation: `src/agents/human_loop.py::clarify`.
  - Customers are told they are talking to an AI and can ask for a human: `src/cli.py::AI_DISCLOSURE`.
  - Limitations are documented in `docs/model-card.md`.
- **Residual: Medium.** Agent training and an operating procedure for reviewing drafts are outside this repository.

### R11: Long-term memory poisoning
- **Likelihood / impact.** M / M.
- **Mitigations → controls.**
  - Turns blocked by the input guard are never mined or written to memory: `src/context/summarization.py::memory_writer`.
  - Memories are namespaced per customer: `src/memory/long_term.py::namespace`.
  - Stored facts are passed to the model as quarantined data only: `src/agents/resolution_agent.py::resolution_agent`.
- **Evidence.**
  - CT-0009 memory_write skipped `evidence/governance/logs/agent_actions.jsonl:L41`.
  - Red-team RT-14 in `evidence/governance/reports/redteam_results.json`.
  - Test: `tests/test_memory_persistence.py::test_cross_session_recall_after_process_restart` (the control customer recalls nothing).
- **Residual: Low.**

### R12: System prompt / policy internals leakage
- **Likelihood / impact.** L / L. The policies are not secret; prompts contain no credentials.
- **Mitigations → controls.**
  - Prompt-extraction patterns are blocked: `src/context/quarantine.py::detect_injection`.
  - The red-team harm check looks for prompt text in replies: `scripts/run_redteam.py::judge`.
- **Evidence.** RT-04 blocked in `evidence/governance/reports/redteam_results.json`.
- **Residual: Low.**

### R13: Observability egress
- **Likelihood / impact.** L / M.
- **Mitigations → controls.**
  - Phoenix runs locally in-process: `src/observability/tracing.py::launch_phoenix`.
  - Spans are masked before export: `src/observability/tracing.py::MaskingSpanExporter`.
  - Third-party telemetry opt-outs are set in `src/config.py`.
- **Residual: Low.** Phoenix 19.17 contacts an external documentation service at startup (observed in its log
  output). No copilot data is sent, but a production deployment should block outbound traffic from the
  observability host.
