"""Demo of the streaming API: starts uvicorn in-process, calls the endpoints over HTTP and writes every
Server-Sent Event it receives to logs/api_demo.log (already masked by the API).

Scenarios: health check; a billing question (streamed to resolution); an over-threshold retention
offer (stream pauses at `approval_required`, then POST /approval resumes it); a prompt injection
(blocked by the input guard); a request for a session owned by another customer (403).
The API's tool/audit logs go to logs/api_demo/ so the main evidence logs are not mixed.

Run: python -m scripts.api_demo
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

import httpx
import uvicorn

from src.audit import audit_middleware
from src.config import LOGS_DIR, SAMPLE_CONTACTS_PATH
from src.guardrails.pii import mask, mask_obj
from src.tools import logging_middleware

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
LOG = LOGS_DIR / "api_demo.log"
LOG_DIR = LOGS_DIR / "api_demo"


def log(line: str) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(mask(line, amounts=True) + "\n")
    print(mask(line, amounts=True))


async def read_sse(resp: httpx.Response) -> list[dict]:
    events, event = [], None
    async for line in resp.aiter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
            events.append({"event": event, "data": data})
            # the approver / account holder may see amounts in the stream; the log may not
            log(f"    <- {event}: {json.dumps(mask_obj(data, amounts=True), ensure_ascii=False)[:300]}")
    return events


async def scenarios() -> bool:
    contacts = {json.loads(l)["scenario"]: json.loads(l) for l in SAMPLE_CONTACTS_PATH.read_text().splitlines()}
    ok = True
    async with httpx.AsyncClient(base_url=BASE, timeout=120) as client:
        r = await client.get("/health")
        log(f"GET /health -> {r.status_code} {r.json()}")

        c = contacts["billing_query"]
        log(f"\n[1] POST /v1/contacts/stream  customer={c['customer_id']}  message={c['turns'][0]!r}")
        async with client.stream("POST", "/v1/contacts/stream", headers={"X-Customer-Id": c["customer_id"]},
                                 json={"message": c["turns"][0], "session_id": "API-DEMO-1"}) as resp:
            ev = await read_sse(resp)
        ok &= ev[-1]["event"] == "resolution" and ev[-1]["data"]["outcome"] == "resolve"

        c = contacts["cancellation_at_risk_needs_approval"]
        log(f"\n[2] POST /v1/contacts/stream  customer={c['customer_id']}  message={c['turns'][0]!r}")
        async with client.stream("POST", "/v1/contacts/stream", headers={"X-Customer-Id": c["customer_id"]},
                                 json={"message": c["turns"][0], "session_id": "API-DEMO-2"}) as resp:
            ev = await read_sse(resp)
        ok &= ev[-1]["event"] == "approval_required"
        log("    POST /v1/sessions/API-DEMO-2/approval  {approved: true, approver: api-demo-auto-approve}")
        async with client.stream("POST", "/v1/sessions/API-DEMO-2/approval",
                                 json={"approved": True, "approver": "api-demo-auto-approve"}) as resp:
            ev = await read_sse(resp)
        ok &= ev[-1]["event"] == "resolution" and ev[-1]["data"]["outcome"] == "offer"

        c = contacts["prompt_injection"]
        log(f"\n[3] POST /v1/contacts/stream  customer={c['customer_id']}  message={c['turns'][0]!r}")
        async with client.stream("POST", "/v1/contacts/stream", headers={"X-Customer-Id": c["customer_id"]},
                                 json={"message": c["turns"][0], "session_id": "API-DEMO-3"}) as resp:
            ev = await read_sse(resp)
        ok &= ev[-1]["data"]["outcome"] == "refuse"

        other = contacts["billing_query"]["customer_id"]
        log(f"\n[4] POST /v1/contacts/stream  customer={other}  session_id=API-DEMO-2 (owned by another customer)")
        r = await client.post("/v1/contacts/stream", headers={"X-Customer-Id": other},
                              json={"message": "What's on this bill?", "session_id": "API-DEMO-2"})
        log(f"    <- {r.status_code} {r.json()}")
        ok &= r.status_code == 403

        log("\n[5] Browser UI: GET / , /v1/samples , /v1/account")
        page = await client.get("/")
        samples = (await client.get("/v1/samples")).json()
        acct = await client.get("/v1/account", headers={"X-Customer-Id": samples[0]["customer_id"]})
        log(f"    <- / {page.status_code} ({len(page.text)} bytes, title present: {'Retention Copilot' in page.text}); "
            f"samples {len(samples)}; account {acct.status_code}: {acct.json()['lines'][0]}")
        ok &= page.status_code == 200 and len(samples) > 0 and acct.status_code == 200
    ok &= await browser_walkthrough()
    return ok


async def browser_walkthrough() -> bool:
    """Drive the web UI in headless Chromium: sample -> send -> approval card -> approve -> reply + citations."""
    from playwright.async_api import async_playwright

    from src.config import REPORTS_DIR
    log("\n[6] Browser walkthrough (headless Chromium): 'half off' scenario with team-lead approval")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 1000})
        await page.goto(BASE + "/")
        await page.wait_for_function("document.querySelectorAll('#sample option').length > 1")
        idx = await page.evaluate("[...document.querySelectorAll('#sample option')].findIndex(o => o.textContent.startsWith('cancellation at risk needs approval')) - 1")
        await page.select_option("#sample", str(idx))
        await page.wait_for_function("document.querySelector('#account pre').textContent.startsWith('Customer')")
        await page.click("#send")
        await page.wait_for_selector(".approval button.ok", timeout=60_000)
        log("    approval card shown: " + (await page.inner_text(".approval h3")))
        await page.click(".approval button.ok")
        await page.wait_for_selector(".msg.bot .cites", timeout=60_000)
        reply = await page.inner_text(".msg.bot")
        log("    final reply shown with citations: " + " ".join(reply.split())[:200])
        shot = REPORTS_DIR / "web_ui.png"
        await page.screenshot(path=str(shot), full_page=True)
        log(f"    screenshot -> reports/{shot.name}")
        await browser.close()
    return "approved" in reply.lower() and "POL-RET-003" in reply


async def main() -> int:
    LOG.write_text("")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging_middleware.set_log_path(LOG_DIR / "tool_calls.jsonl")
    audit_middleware.set_log_path(LOG_DIR / "agent_actions.jsonl")
    log(f"api_demo.log generated by scripts/api_demo.py at {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    server = uvicorn.Server(uvicorn.Config("src.api.app:app", host="127.0.0.1", port=PORT, log_level="warning"))
    task = asyncio.create_task(server.serve())
    for _ in range(600):
        if server.started:
            break
        await asyncio.sleep(0.1)
    try:
        ok = await scenarios()
    finally:
        server.should_exit = True
        await task
    log(f"\nRESULT: {'PASS' if ok else 'FAIL'} (all scenarios behaved as expected)" if ok else "\nRESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
