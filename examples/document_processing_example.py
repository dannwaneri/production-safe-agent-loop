"""Document data extraction agent demo.

Demonstrates the five-primitive pattern against an extraction task:
spec -> circuit breaker -> ledger -> agent loop -> review surface ->
human attestation. Uses hardcoded sample document text so the example
runs end-to-end without an external document store.

Run:
    export ANTHROPIC_API_KEY=sk-...
    python examples/document_processing_example.py
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


SCHEMA_V1 = {
    "vendor_name": "string",
    "invoice_number": "string",
    "invoice_date": "ISO 8601 date",
    "due_date": "ISO 8601 date",
    "total_amount": "decimal",
    "currency": "ISO 4217 code",
    "line_items": "list[{description, qty, unit_price}]",
    "tax_amount": "decimal | null",
    "payment_terms": "string | null",
}


SAMPLE_DOCUMENTS = [
    """\
ACME WIDGETS LTD.
14 Foundry Lane, Sheffield S1 2HE

INVOICE  #INV-2026-04832
Issued: 2026-05-14    Due: 2026-06-13

Bill to:
Cobalt Operations, 88 Riverside, Manchester

Description                       Qty    Unit     Line
Industrial bracket (Type B)        12    18.50   222.00
Mounting plate, stainless           4    34.00   136.00
Delivery (next-day)                 1    25.00    25.00
                                        -----------
                                  Subtotal       383.00
                                  VAT (20%)       76.60
                                  TOTAL GBP      459.60

Payment terms: Net 30. Late payment subject to 3% monthly interest.
""",
    """\
LUMA STUDIOS INC.
Receipt of Services

To: Helix Robotics
Invoice no: LUMA-2026-Q2-117
Date: 06/02/2026
Due:  07/02/2026

Brand identity package - 1 @ $4,200.00
Web revision sprint  - 1 @ $1,800.00

Subtotal:   $6,000.00
Tax:        (exempt)
Amount due: $6,000.00 USD

Terms: due on receipt
""",
    """\
INVOICE
North Sea Logistics
ID: NSL/26/SEP/0094

Period: September 2026
Charges:
  Container handling x 18    540.00 EUR
  Customs filing fee          75.00 EUR
  Demurrage (2 days)         180.00 EUR

Total: 795.00 EUR
Due:   Within 14 days of receipt
""",
]


SUGGESTED_SPEC_ANSWERS = {
    "what_it_does": (
        "Extracts structured fields from invoice-like documents into "
        "schema v1: vendor_name, invoice_number, dates, totals, currency, "
        "line items, tax, payment terms."
    ),
    "what_it_does_not": (
        "Does not OCR scanned images. Does not classify document types. "
        "Does not write to any downstream system."
    ),
    "done_looks_like": (
        "All documents processed, structured data extracted per schema v1, "
        "fields with null rate above 5% flagged, batch complete."
    ),
}


def format_task(documents: list[str]) -> str:
    schema_lines = "\n".join(f"  {k}: {v}" for k, v in SCHEMA_V1.items())
    parts = [
        "Extract the following fields from each document into schema v1:",
        f"\n{schema_lines}\n",
        "Return one JSON object per document. After all documents, list any "
        "fields with null rate above 5% across the batch.",
    ]
    for i, doc in enumerate(documents, start=1):
        parts.append(f"\n--- DOCUMENT {i} ---\n{doc}")
    parts.append("\n\nWhen finished, end your response with the literal token DONE.")
    return "".join(parts)


def attest_or_skip(surface: ReviewSurface, session_id: str) -> int:
    decision = input("\nAttest this extraction batch? [y/N]: ").strip().lower()
    if decision != "y":
        print("Session not attested. Downstream systems will NOT receive data.")
        return 0
    reviewer = input("Reviewer name: ").strip() or "anonymous"
    notes = input("Notes (optional): ").strip()
    attestation = surface.attest(
        session_id=session_id, reviewer=reviewer, notes=notes,
    )
    print(
        f"\nAttested. frame_hash={attestation.frame_hash}\n"
        "Extraction batch cleared for downstream ingestion."
    )
    return 0


def main() -> int:
    print("Document extraction agent demo\n")
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
    result = loop.run(format_task(SAMPLE_DOCUMENTS))
    print(
        f"\nLoop done: success={result.success} turns={result.turns} "
        f"tokens={result.total_tokens} breach={result.breach_reason}\n"
    )

    surface = ReviewSurface(spec_db_path="spec.db", ledger_db_path="ledger.db")
    print(surface.render(result.session_id))
    return attest_or_skip(surface, result.session_id)


if __name__ == "__main__":
    sys.exit(main())
