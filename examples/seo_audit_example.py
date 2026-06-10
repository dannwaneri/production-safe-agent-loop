"""End-to-end demo of the production-safe agent loop pattern.

Crawls a single URL with requests + BeautifulSoup, then runs the agent
loop against the extracted SEO signals — under a strict circuit breaker.

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-...
    python examples/seo_audit_example.py https://example.com
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_loop import AgentLoop
from circuit_breaker import CircuitBreaker
from ledger import Ledger
from spec_writer import SpecWriter


@dataclass(frozen=True)
class CrawlResult:
    url: str
    status_code: int
    title: str
    meta_description: str
    h1_count: int
    link_count: int


def crawl(url: str, timeout: int = 10) -> CrawlResult:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "production-safe-agent-loop/1.0 (tutorial demo)"},
    )
    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = (
        meta_tag.get("content", "") if meta_tag is not None else ""
    )

    return CrawlResult(
        url=url,
        status_code=response.status_code,
        title=title,
        meta_description=meta_description,
        h1_count=len(soup.find_all("h1")),
        link_count=len(soup.find_all("a")),
    )


def format_task(c: CrawlResult) -> str:
    return (
        f"Analyze the following SEO signals from {c.url}:\n\n"
        f"- HTTP status: {c.status_code}\n"
        f"- Title ({len(c.title)} chars): {c.title!r}\n"
        f"- Meta description ({len(c.meta_description)} chars): {c.meta_description!r}\n"
        f"- H1 count: {c.h1_count}\n"
        f"- Link count: {c.link_count}\n\n"
        "Identify the top three SEO issues and how to fix them. "
        "Reply with a numbered list, then say DONE."
    )


def print_ledger(ledger: Ledger, session_id: str) -> None:
    print("\n=== LEDGER ===")
    rows = ledger.get_session(session_id)
    print(f"{len(rows)} row(s) for session {session_id}")
    for r in rows:
        breach = f" breach={r.breach_reason}" if r.breach_reason else ""
        verdict = "PASS" if r.pass_fail else "FAIL"
        print(
            f"  turn={r.turn_count} origin={r.state_origin} "
            f"tokens={r.token_delta} ms={r.execution_time_ms} {verdict}{breach}"
        )


def main(target_url: str) -> int:
    print(f"Production-safe agent loop demo — target: {target_url}\n")

    print("Step 1: capture spec (3 questions)")
    spec = SpecWriter(db_path="spec.db").run()
    print(f"Spec captured: session_id={spec.session_id}\n")

    print(f"Step 2: crawl {target_url}")
    try:
        crawl_result = crawl(target_url)
    except requests.RequestException as exc:
        print(f"Crawl failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"  status={crawl_result.status_code} "
        f"title={crawl_result.title!r}\n"
    )

    print("Step 3: configure circuit breaker (turn=5, token=15000) + ledger")
    breaker = CircuitBreaker(turn_limit=5, token_limit=15000)
    ledger = Ledger(db_path="ledger.db")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set; skipping live LLM call.\n"
            "Set it and rerun to see the full loop.",
            file=sys.stderr,
        )
        return 1

    print("Step 4: run agent loop")
    from anthropic import Anthropic

    client = Anthropic()
    loop = AgentLoop(
        spec=spec,
        circuit_breaker=breaker,
        ledger=ledger,
        client=client,
    )
    result = loop.run(format_task(crawl_result))

    print(
        f"\nLoopResult: success={result.success} turns={result.turns} "
        f"tokens={result.total_tokens} breach={result.breach_reason}"
    )
    print_ledger(ledger, result.session_id)
    return 0 if result.success else 2


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    sys.exit(main(url))
