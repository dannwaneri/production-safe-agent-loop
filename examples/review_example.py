"""End-to-end demo of the review surface.

Runs the SEO audit (same as seo_audit_example), then loads the resulting
session into the review surface and prompts a human reviewer to attest
or reject.

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-...
    python examples/review_example.py https://example.com
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_loop import AgentLoop
from circuit_breaker import CircuitBreaker
from ledger import Ledger
from review_surface import ReviewSurface
from spec_writer import SpecWriter

from examples.seo_audit_example import crawl, format_task


def main(target_url: str) -> int:
    print(f"Review surface demo — target: {target_url}\n")

    print("Step 1: capture spec (3 questions)")
    spec = SpecWriter(db_path="spec.db").run()
    print(f"Spec captured: session_id={spec.session_id}\n")

    print(f"Step 2: crawl {target_url}")
    try:
        crawl_result = crawl(target_url)
    except Exception as exc:
        print(f"Crawl failed: {exc}", file=sys.stderr)
        return 1
    print(f"  status={crawl_result.status_code}\n")

    print("Step 3: run agent loop")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; cannot run live LLM.", file=sys.stderr)
        return 1
    from anthropic import Anthropic

    breaker = CircuitBreaker(turn_limit=5, token_limit=15000)
    ledger = Ledger(db_path="ledger.db")
    loop = AgentLoop(spec=spec, circuit_breaker=breaker, ledger=ledger, client=Anthropic())
    result = loop.run(format_task(crawl_result))
    print(f"  loop result: success={result.success} turns={result.turns} "
          f"tokens={result.total_tokens}\n")

    print("Step 4: load review surface")
    surface = ReviewSurface(spec_db_path="spec.db", ledger_db_path="ledger.db")
    print(surface.render(result.session_id))

    print("\nStep 5: attestation")
    decision = input("Attest this session? [y/N]: ").strip().lower()
    if decision != "y":
        print("Session not attested. Exiting.")
        return 0

    reviewer = input("Reviewer name (or identifier): ").strip() or "anonymous"
    notes = input("Notes (optional, press enter to skip): ").strip()

    attestation = surface.attest(
        session_id=result.session_id,
        reviewer=reviewer,
        notes=notes,
    )
    print(
        f"\nAttested.\n"
        f"  attestation_id: {attestation.id}\n"
        f"  reviewer:       {attestation.reviewer}\n"
        f"  attested_at:    {attestation.attested_at}\n"
        f"  frame_hash:     {attestation.frame_hash}"
    )
    return 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    sys.exit(main(url))
