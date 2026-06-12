# Production-Safe Agent Loop

A minimal Python library that turns a `while True: call_LLM()` sketch
into something you can deploy without burning $47,000.

Reference repo for the freeCodeCamp tutorial
**"How to Build a Production-Safe Agent Loop: From Exit Conditions to
Audit Trails."**

---

## Why this exists

Production agent loops have a documented track record of running away:

- A 4-agent LangChain loop ran 11 days and cost **$47,000** (Nov 2025).
- A Claude Code recursion loop burned **$16,000–$50,000 in 5 hours** (Jul 2025).
- Gartner pegs the token multiplier at **5–30x**; production benchmarks
  show **70x**.
- **Fable 5** (released 2026-06-09) doubles Opus 4.8's token price.
- Gartner: **40% of agentic projects scrapped by 2027** due to economic failure.
- FinOps Foundation 2026: **73% of enterprises** report AI costs exceeded
  projections.

The pattern this repo teaches: **four small primitives — a spec writer,
a circuit breaker, an audit ledger, and a loop that respects both —
catch most of those failure modes before they ship.**

---

## The five primitives

| Module | Role |
|---|---|
| `spec_writer.py` | Forces three answers before the loop is allowed to run. |
| `circuit_breaker.py` | Hard ceilings on turns and tokens. Trips with a checkpoint. |
| `ledger.py` | Append-only SQLite audit trail. One row per turn. |
| `agent_loop.py` | The loop that ties them together. |
| `review_surface.py` | Assembles a five-element review frame for human attestation. |

Each module is independently importable and testable.

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ required. Verified on Python 3.14.

### LLM client

`AgentLoop` is parameterised on an `LLMClient` protocol — it accepts
anything that looks like the Anthropic SDK shape:

```python
class LLMClient(Protocol):
    messages: MessagesEndpoint  # .create(model, max_tokens, system, messages)
```

The default install ships with `anthropic` and `examples/` uses it
directly. To plug in a different provider (OpenAI, Gemini, Ollama, local
inference), write a ~20-line adapter that exposes that shape and pass it
to `AgentLoop`. The `FakeClient` in `tests/test_agent_loop.py` is the
canonical example of a non-Anthropic client satisfying the protocol —
it's a pure Python class with no SDK dependency, and the full test suite
exercises it.

---

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-...
python examples/seo_audit_example.py https://example.com
```

The example:

1. Prompts you for a 3-question spec.
2. Crawls the URL with `requests` + `BeautifulSoup`.
3. Runs the agent loop against the extracted SEO signals.
4. Prints the ledger on completion (or breach).

---

## Anatomy of a turn

Every turn the loop does, in order:

1. `circuit_breaker.check(turn, accumulated_tokens)` — raises if either ceiling is exceeded.
2. `client.messages.create(...)` — the actual LLM call.
3. `ledger.write(...)` — one row, append-only.
4. If `stop_reason == "end_turn"`, return; otherwise loop.

If the circuit breaker raises mid-loop, the loop catches it, writes a
breach row to the ledger, and returns
`LoopResult(success=False, breach_reason=...)`.

---

## Defaults

- `turn_limit=5`, `token_limit=15000`
- `model="claude-sonnet-4-6"`, `max_tokens=1024`

These match a tight tutorial demo, not your production budget. Tune
them at instantiation.

---

## Component reference

### `SpecWriter`

```python
from spec_writer import SpecWriter

spec = SpecWriter(db_path="spec.db").run()
```

Forces three answers (`what_it_does`, `what_it_does_not`,
`done_looks_like`) before returning a `SpecResult` with a fresh
`session_id`. Stored to SQLite. `load(session_id)` retrieves later.

For testing, `SpecWriter` accepts injectable `input_fn` and `output_fn`
callables — no `monkeypatch` required.

### `CircuitBreaker`

```python
from circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(turn_limit=5, token_limit=15000)
breaker.check(turn_count, accumulated_tokens)  # raises on breach
```

`check()` must be called **before** every LLM call. On breach it
prints a checkpoint message to stdout and raises `CircuitBreakerError`
with `reason` ∈ `{"turn_ceiling", "token_ceiling"}`.

Boundary is strict `>`: `turn_count == turn_limit` is allowed;
`turn_count == turn_limit + 1` trips.

### `Ledger`

```python
from ledger import Ledger

ledger = Ledger(db_path="ledger.db")
ledger.write(
    session_id=...,
    turn_count=...,
    state_origin="llm",
    input_str=...,
    token_delta=...,
    execution_time_ms=...,
    pass_fail=True,
    breach_reason=None,
)
ledger.get_session(session_id)  # list[LedgerRow]
ledger.get_all()                # list[LedgerRow]
```

Append-only. No updates, no deletes. `input_hash` is stored as
SHA-256 of the input string — the original text never persists, so
PII does not enter the audit trail.

### `AgentLoop`

```python
from agent_loop import AgentLoop

loop = AgentLoop(spec, breaker, ledger, client)
result = loop.run(task)
# LoopResult(success, turns, total_tokens, session_id, breach_reason)
```

`client` is anything that satisfies the `LLMClient` protocol — the real
`anthropic.Anthropic()` client, an adapter wrapping OpenAI / Gemini /
Ollama, or a test double. See the **LLM client** section under Install
for the protocol shape.

`LoopResult.session_id` is inherited from `spec.session_id` so the
ledger rows tie back to the spec without a join.

### `ReviewSurface`

```python
from review_surface import ReviewSurface

surface = ReviewSurface(spec_db_path="spec.db", ledger_db_path="ledger.db")
print(surface.render(session_id))                 # human-readable frame
attestation = surface.attest(session_id, "alice", notes="LGTM")
# AttestationResult(id, session_id, reviewer, attested_at, notes, frame_hash)
```

Reads a completed session out of the spec + ledger databases and
assembles a five-element frame for human review:

1. **Original promise** — the three SpecWriter answers.
2. **Acceptance criteria** — the `done_looks_like` benchmark.
3. **Diff** — first input hash, final state, turns, tokens, breach flag.
4. **Evidence** — every ledger row, formatted.
5. **Unresolved assumptions** — derived from any breach rows or
   `pass_fail=False` rows.

`attest()` writes a row to a new append-only `attestations` table and
returns an `AttestationResult` carrying a `frame_hash` — a SHA-256 over a
canonical serialization of the frame data (excluding the load timestamp,
so two reviewers attesting the same session get the same hash).

---

## Database schema

Two tables across two SQLite files (separate by default).

### `spec.db` → `spec`

| col | type |
|---|---|
| id | INTEGER PRIMARY KEY AUTOINCREMENT |
| session_id | TEXT NOT NULL |
| what_it_does | TEXT NOT NULL |
| what_it_does_not | TEXT NOT NULL |
| done_looks_like | TEXT NOT NULL |
| created_at | TEXT NOT NULL (ISO 8601, UTC) |

Index: `idx_spec_session` on `session_id`.

### `ledger.db` → `ledger`

| col | type |
|---|---|
| id | INTEGER PRIMARY KEY AUTOINCREMENT |
| session_id | TEXT NOT NULL |
| turn_count | INTEGER NOT NULL |
| state_origin | TEXT NOT NULL |
| input_hash | TEXT NOT NULL (SHA-256) |
| token_delta | INTEGER NOT NULL |
| execution_time_ms | INTEGER NOT NULL |
| pass_fail | INTEGER NOT NULL (1=pass, 0=fail) |
| breach_reason | TEXT (NULL unless circuit breaker fired) |
| created_at | TEXT NOT NULL (ISO 8601, UTC) |

Index: `idx_ledger_session` on `session_id`.

### `ledger.db` → `attestations`

Created the first time `ReviewSurface` is instantiated against the
ledger database.

| col | type |
|---|---|
| id | INTEGER PRIMARY KEY AUTOINCREMENT |
| session_id | TEXT NOT NULL |
| reviewer | TEXT NOT NULL |
| attested_at | TEXT NOT NULL (ISO 8601, UTC) |
| notes | TEXT |
| frame_hash | TEXT NOT NULL (SHA-256 of canonical frame) |

Index: `idx_attestation_session` on `session_id`. Append-only — no
updates, no deletes.

---

## Running tests

```bash
python -m pytest tests/
```

With coverage:

```bash
python -m coverage run --source=circuit_breaker,ledger,spec_writer,agent_loop -m pytest tests/
python -m coverage report -m
```

**Current status: 80 tests, 100% coverage on all five core modules.**

Tests run without network access and without an Anthropic key —
the loop is exercised against a `FakeClient` test double that mimics
the Anthropic SDK's shape.

---

## Repo layout

```
production-safe-agent-loop/
├── README.md
├── requirements.txt
├── spec_writer.py
├── circuit_breaker.py
├── ledger.py
├── agent_loop.py
├── review_surface.py
├── examples/
│   ├── seo_audit_example.py
│   └── review_example.py
└── tests/
    ├── test_spec_writer.py
    ├── test_circuit_breaker.py
    ├── test_ledger.py
    ├── test_agent_loop.py
    ├── test_review_surface.py
    └── test_seo_audit_example.py
```

---

## Design notes

A few choices the tutorial expands on:

- **Result objects are frozen dataclasses** (`SpecResult`, `LedgerRow`,
  `LoopResult`). The ledger is append-only by spec; immutable rows make
  that visible at the Python type level instead of relying on convention.

- **Timestamps are tz-aware** (`datetime.now(timezone.utc).isoformat()`).
  `datetime.utcnow()` was deprecated in Python 3.12 and is a footgun
  in any system that crosses timezones.

- **`pass_fail` is `bool` at the API edge, `INTEGER 1/0` in storage.**
  Clean Python ergonomics; canonical SQL types on disk.

- **Session-id indexes** on both `spec` and `ledger` tables. The
  primary lookup path is "give me one run's history" — index it.

- **Input strings are hashed, not stored.** PII never reaches the
  ledger; the hash is enough to detect identical inputs across runs.

- **The circuit breaker raises an exception, not a return code.**
  Forces callers to handle it (or crash) — silent breach is impossible.

- **`SpecWriter` injects `input_fn`/`output_fn`** rather than reading
  `stdin` directly. The interactive CLI flow is unit-testable with no
  stdin monkey-patching.

---

## Background reading

- *The Loop Is Not the Product* — argues agent loops burn money because
  nobody defines exit conditions before deploying.
- *You Didn't Build a Workflow. You Built a While Loop With Vibes.* —
  argues token burn is a requirements failure, not an architecture failure.

---

## License

MIT.
