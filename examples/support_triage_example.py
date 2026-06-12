"""Customer support ticket triage agent demo.

Demonstrates the five-primitive pattern against a support triage task:
spec -> circuit breaker -> ledger -> agent loop -> review surface ->
human attestation. Uses a hardcoded sample batch of tickets so the
example runs end-to-end without a real support inbox.

Run:
    export ANTHROPIC_API_KEY=sk-...
    python examples/support_triage_example.py
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


SAMPLE_TICKETS = [
    {
        "id": "T-1041",
        "subject": "Cannot reset password — link expired",
        "body": "I clicked the reset link three times, it says expired every time. Please help, I have a meeting in 20 minutes.",
        "customer_tier": "free",
    },
    {
        "id": "T-1042",
        "subject": "URGENT: production billing endpoint returning 500",
        "body": "Our /v1/billing/invoices endpoint started 500ing 14 minutes ago. We are an enterprise customer. Please escalate.",
        "customer_tier": "enterprise",
    },
    {
        "id": "T-1043",
        "subject": "feature request: dark mode for the dashboard",
        "body": "Would love a dark mode toggle. Bright UI hurts at night. Not urgent.",
        "customer_tier": "pro",
    },
    {
        "id": "T-1044",
        "subject": "Charged twice for last month's subscription",
        "body": "I see two identical charges on my card from May 28. Need a refund for one.",
        "customer_tier": "pro",
    },
    {
        "id": "T-1045",
        "subject": "Where's the API documentation?",
        "body": "I can't find docs for the search API. Is there a public link?",
        "customer_tier": "free",
    },
]


SUGGESTED_SPEC_ANSWERS = {
    "what_it_does": (
        "Triages a batch of support tickets: assigns priority "
        "(P0/P1/P2/P3), routes to a queue (billing/auth/ops/docs/feature), "
        "and flags any ticket needing same-hour escalation."
    ),
    "what_it_does_not": (
        "Does not reply to customers. Does not change ticket state in any "
        "downstream system. Does not handle batches larger than 20 tickets."
    ),
    "done_looks_like": (
        "All tickets in batch classified by priority, high-priority items "
        "flagged for escalation, routing assigned per ticket, batch size "
        "limit not exceeded."
    ),
}


def format_task(tickets: list[dict]) -> str:
    lines = [
        f"Triage the following {len(tickets)} support tickets. For each one, "
        "output: priority (P0/P1/P2/P3), queue (billing/auth/ops/docs/feature), "
        "and ESCALATE if same-hour intervention is needed.\n",
    ]
    for t in tickets:
        lines.append(
            f"\n--- {t['id']} | tier={t['customer_tier']} ---\n"
            f"Subject: {t['subject']}\n"
            f"Body: {t['body']}"
        )
    lines.append("\n\nWhen finished, end your response with the literal token DONE.")
    return "".join(lines)


def attest_or_skip(surface: ReviewSurface, session_id: str) -> int:
    decision = input("\nAttest this triage batch? [y/N]: ").strip().lower()
    if decision != "y":
        print("Session not attested. Routing NOT executed.")
        return 0
    reviewer = input("Reviewer name: ").strip() or "anonymous"
    notes = input("Notes (optional): ").strip()
    attestation = surface.attest(
        session_id=session_id, reviewer=reviewer, notes=notes,
    )
    print(
        f"\nAttested. frame_hash={attestation.frame_hash}\n"
        "Routing decisions cleared for execution."
    )
    return 0


def main() -> int:
    print("Support triage agent demo\n")
    print("Suggested spec answers for this task:")
    for k, v in SUGGESTED_SPEC_ANSWERS.items():
        print(f"  {k}: {v}")
    print()

    spec = SpecWriter(db_path="spec.db").run()
    breaker = CircuitBreaker(turn_limit=5, token_limit=15000)
    ledger = Ledger(db_path="ledger.db")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; cannot run live LLM.", file=sys.stderr)
        return 1
    from anthropic import Anthropic

    loop = AgentLoop(
        spec=spec, circuit_breaker=breaker, ledger=ledger, client=Anthropic(),
    )
    result = loop.run(format_task(SAMPLE_TICKETS))
    print(
        f"\nLoop done: success={result.success} turns={result.turns} "
        f"tokens={result.total_tokens} breach={result.breach_reason}\n"
    )

    surface = ReviewSurface(spec_db_path="spec.db", ledger_db_path="ledger.db")
    print(surface.render(result.session_id))
    return attest_or_skip(surface, result.session_id)


if __name__ == "__main__":
    sys.exit(main())
