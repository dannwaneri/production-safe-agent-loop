"""PR code-review agent demo.

Demonstrates the five-primitive pattern against a code review task:
spec -> circuit breaker -> ledger -> agent loop -> review surface ->
human attestation. Uses a hardcoded sample diff so the example runs
end-to-end without a real PR.

Run:
    export ANTHROPIC_API_KEY=sk-...
    python examples/coding_review_example.py
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


SAMPLE_DIFF = """\
diff --git a/auth/login.py b/auth/login.py
index 1a2b3c4..5d6e7f8 100644
--- a/auth/login.py
+++ b/auth/login.py
@@ -1,9 +1,18 @@
 import hashlib
+import os
+import requests

-DB_PASSWORD = "changeme"
+DB_PASSWORD = "Hunter2!"  # TODO rotate

 def authenticate(username, password):
-    return check_user(username, password)
+    payload = {"u": username, "p": password}
+    r = requests.get("http://auth.internal/login", params=payload)
+    return r.json()["ok"]

+def reset_password(user_id, new_password):
+    sql = f"UPDATE users SET pw='{new_password}' WHERE id={user_id}"
+    db.execute(sql)
+    return True
"""


SUGGESTED_SPEC_ANSWERS = {
    "what_it_does": (
        "Reviews a Python diff for P0 security violations: hardcoded "
        "secrets, SQL injection, unsafe HTTP, missing input validation."
    ),
    "what_it_does_not": (
        "Does not enforce style, type hints, docstrings, or non-security "
        "lint rules. Does not auto-fix the diff."
    ),
    "done_looks_like": (
        "All files in diff reviewed, P0 violations flagged with file:line, "
        "no security patterns left unsurfaced, diff under 500 lines."
    ),
}


def format_task(diff: str) -> str:
    return (
        "Review the following diff. Flag every P0 security issue with "
        "file:line and a one-line fix. When complete, end your response "
        "with the literal token DONE.\n\n"
        f"```diff\n{diff}\n```"
    )


def attest_or_skip(surface: ReviewSurface, session_id: str) -> int:
    decision = input("\nAttest this review session? [y/N]: ").strip().lower()
    if decision != "y":
        print("Session not attested. Code review NOT cleared for merge.")
        return 0
    reviewer = input("Reviewer name: ").strip() or "anonymous"
    notes = input("Notes (optional): ").strip()
    attestation = surface.attest(
        session_id=session_id, reviewer=reviewer, notes=notes,
    )
    print(
        f"\nAttested. frame_hash={attestation.frame_hash}\n"
        "Review cleared. Downstream merge gate may proceed."
    )
    return 0


def main() -> int:
    print("Code review agent demo\n")
    print("Suggested spec answers for this task (copy/paste if you want):")
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
    result = loop.run(format_task(SAMPLE_DIFF))
    print(
        f"\nLoop done: success={result.success} turns={result.turns} "
        f"tokens={result.total_tokens} breach={result.breach_reason}\n"
    )

    surface = ReviewSurface(spec_db_path="spec.db", ledger_db_path="ledger.db")
    print(surface.render(result.session_id))
    return attest_or_skip(surface, result.session_id)


if __name__ == "__main__":
    sys.exit(main())
