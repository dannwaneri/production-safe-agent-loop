import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger import Ledger
from review_surface import (
    AttestationResult,
    DiffResult,
    ReviewFrame,
    ReviewSurface,
    SessionNotFoundError,
)
from spec_writer import SpecWriter


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


@pytest.fixture
def db_paths(tmp_path):
    return str(tmp_path / "spec.db"), str(tmp_path / "ledger.db")


def _make_spec(spec_db_path, answers=("does X", "not Y", "Z looks like done")):
    sw = SpecWriter(
        db_path=spec_db_path,
        input_fn=scripted(list(answers)),
        output_fn=lambda _: None,
    )
    return sw.run()


def _populate_ledger(ledger_db_path, session_id, rows):
    led = Ledger(db_path=ledger_db_path)
    for r in rows:
        led.write(
            session_id=session_id,
            turn_count=r["turn"],
            state_origin=r["state"],
            input_str=r.get("input", "task"),
            token_delta=r["tokens"],
            execution_time_ms=r["ms"],
            pass_fail=r["pass"],
            breach_reason=r.get("breach"),
        )


def _clean_run(db_paths):
    spec_db, ledger_db = db_paths
    spec = _make_spec(spec_db)
    _populate_ledger(ledger_db, spec.session_id, [
        {"turn": 1, "state": "llm", "tokens": 100, "ms": 50, "pass": True},
        {"turn": 2, "state": "llm", "tokens": 120, "ms": 60, "pass": True},
        {"turn": 3, "state": "llm", "tokens": 90, "ms": 45, "pass": True},
    ])
    return spec


def _breached_run(db_paths):
    spec_db, ledger_db = db_paths
    spec = _make_spec(spec_db)
    _populate_ledger(ledger_db, spec.session_id, [
        {"turn": 1, "state": "llm", "tokens": 200, "ms": 50, "pass": True},
        {"turn": 2, "state": "llm", "tokens": 220, "ms": 55, "pass": True},
        {"turn": 3, "state": "circuit_breaker", "tokens": 0, "ms": 0,
         "pass": False, "breach": "turn_ceiling"},
    ])
    return spec


def test_load_returns_review_frame(db_paths):
    spec = _clean_run(db_paths)
    surface = ReviewSurface(*db_paths)
    frame = surface.load(spec.session_id)
    assert isinstance(frame, ReviewFrame)
    assert frame.session_id == spec.session_id
    assert frame.original_promise == spec
    assert frame.acceptance_criteria == spec.done_looks_like
    assert isinstance(frame.diff, DiffResult)
    assert frame.diff.turns_completed == 3
    assert frame.diff.total_tokens == 310
    assert frame.diff.breached is False
    assert frame.diff.final_output == "llm"
    assert len(frame.evidence) == 3
    assert frame.unresolved_assumptions == ()
    assert frame.created_at


def test_render_output_contains_all_five_elements(db_paths):
    spec = _clean_run(db_paths)
    surface = ReviewSurface(*db_paths)
    output = surface.render(spec.session_id)
    assert "REVIEW SURFACE" in output
    assert "[1] ORIGINAL PROMISE" in output
    assert "[2] ACCEPTANCE CRITERIA" in output
    assert "[3] DIFF" in output
    assert "[4] EVIDENCE" in output
    assert "[5] UNRESOLVED ASSUMPTIONS" in output
    assert spec.what_it_does in output
    assert spec.what_it_does_not in output
    assert spec.done_looks_like in output
    assert "Turns completed:     3" in output
    assert "Total tokens:        310" in output
    assert "No unresolved assumptions detected." in output


def test_render_includes_breach_in_unresolved_for_breached_run(db_paths):
    spec = _breached_run(db_paths)
    surface = ReviewSurface(*db_paths)
    output = surface.render(spec.session_id)
    assert "turn_ceiling" in output
    assert "No unresolved assumptions detected." not in output


def test_attest_writes_attestation_row(db_paths):
    spec = _clean_run(db_paths)
    surface = ReviewSurface(*db_paths)
    attestation = surface.attest(spec.session_id, reviewer="alice", notes="LGTM")
    assert isinstance(attestation, AttestationResult)
    assert attestation.session_id == spec.session_id
    assert attestation.reviewer == "alice"
    assert attestation.notes == "LGTM"
    assert len(attestation.frame_hash) == 64
    _, ledger_db = db_paths
    with sqlite3.connect(ledger_db) as conn:
        row = conn.execute(
            "SELECT session_id, reviewer, notes, frame_hash FROM attestations "
            "WHERE id = ?",
            (attestation.id,),
        ).fetchone()
    assert row == (spec.session_id, "alice", "LGTM", attestation.frame_hash)


def test_attest_frame_hash_is_deterministic(db_paths):
    spec = _clean_run(db_paths)
    surface = ReviewSurface(*db_paths)
    a1 = surface.attest(spec.session_id, reviewer="alice")
    a2 = surface.attest(spec.session_id, reviewer="bob", notes="different note")
    assert a1.frame_hash == a2.frame_hash


def test_attest_frame_hash_differs_across_sessions(db_paths):
    spec_a = _clean_run(db_paths)
    _, ledger_db = db_paths
    # populate a second session with different content
    spec_b = _make_spec(db_paths[0], answers=("alt does", "alt not", "alt done"))
    _populate_ledger(ledger_db, spec_b.session_id, [
        {"turn": 1, "state": "llm", "tokens": 50, "ms": 10, "pass": True},
    ])
    surface = ReviewSurface(*db_paths)
    a = surface.attest(spec_a.session_id, reviewer="r")
    b = surface.attest(spec_b.session_id, reviewer="r")
    assert a.frame_hash != b.frame_hash


def test_load_raises_on_missing_session(db_paths):
    surface = ReviewSurface(*db_paths)
    with pytest.raises(SessionNotFoundError):
        surface.load("nonexistent-session-id")


def test_load_raises_when_only_spec_exists(db_paths):
    spec = _make_spec(db_paths[0])
    surface = ReviewSurface(*db_paths)
    with pytest.raises(SessionNotFoundError):
        surface.load(spec.session_id)


def test_load_raises_when_only_ledger_exists(db_paths):
    _, ledger_db = db_paths
    _populate_ledger(ledger_db, "orphan-session", [
        {"turn": 1, "state": "llm", "tokens": 10, "ms": 5, "pass": True},
    ])
    surface = ReviewSurface(*db_paths)
    with pytest.raises(SessionNotFoundError):
        surface.load("orphan-session")


def test_unresolved_assumptions_populated_from_breach_rows(db_paths):
    spec = _breached_run(db_paths)
    surface = ReviewSurface(*db_paths)
    frame = surface.load(spec.session_id)
    assert len(frame.unresolved_assumptions) == 1
    assert "turn_ceiling" in frame.unresolved_assumptions[0]
    assert frame.diff.breached is True


def test_unresolved_assumptions_empty_when_no_breaches(db_paths):
    spec = _clean_run(db_paths)
    surface = ReviewSurface(*db_paths)
    frame = surface.load(spec.session_id)
    assert frame.unresolved_assumptions == ()
    assert frame.diff.breached is False


def test_unresolved_assumptions_from_pass_fail_false_without_breach(db_paths):
    spec_db, ledger_db = db_paths
    spec = _make_spec(spec_db)
    _populate_ledger(ledger_db, spec.session_id, [
        {"turn": 1, "state": "llm", "tokens": 100, "ms": 50, "pass": True},
        {"turn": 2, "state": "llm", "tokens": 90, "ms": 40, "pass": False},
    ])
    surface = ReviewSurface(*db_paths)
    frame = surface.load(spec.session_id)
    assert len(frame.unresolved_assumptions) == 1
    assert "pass_fail=False" in frame.unresolved_assumptions[0]
    assert frame.diff.breached is True


def test_get_attestations_returns_history(db_paths):
    spec = _clean_run(db_paths)
    surface = ReviewSurface(*db_paths)
    surface.attest(spec.session_id, reviewer="alice", notes="first review")
    surface.attest(spec.session_id, reviewer="bob", notes="second review")
    attestations = surface.get_attestations(spec.session_id)
    assert len(attestations) == 2
    assert attestations[0].reviewer == "alice"
    assert attestations[1].reviewer == "bob"


def test_get_attestations_empty_for_unattested_session(db_paths):
    spec = _clean_run(db_paths)
    surface = ReviewSurface(*db_paths)
    assert surface.get_attestations(spec.session_id) == []


def test_attestations_table_created_idempotently(db_paths):
    ReviewSurface(*db_paths)
    ReviewSurface(*db_paths)


def test_diff_handles_session_with_only_breach_rows(db_paths):
    spec_db, ledger_db = db_paths
    spec = _make_spec(spec_db)
    _populate_ledger(ledger_db, spec.session_id, [
        {"turn": 1, "state": "circuit_breaker", "tokens": 0, "ms": 0,
         "pass": False, "breach": "token_ceiling"},
    ])
    surface = ReviewSurface(*db_paths)
    frame = surface.load(spec.session_id)
    assert frame.diff.turns_completed == 0
    assert frame.diff.final_output == "circuit_breaker"
    assert frame.diff.breached is True


def test_attest_hashes_unresolved_assumptions_into_frame_hash(db_paths):
    spec = _breached_run(db_paths)
    surface = ReviewSurface(*db_paths)
    attestation = surface.attest(spec.session_id, reviewer="r")
    assert len(attestation.frame_hash) == 64
    # Hash must change if the unresolved assumptions change. We assert by
    # comparing against a session whose ledger has no breach.
    spec2 = _make_spec(db_paths[0], answers=("a", "b", "c"))
    _populate_ledger(db_paths[1], spec2.session_id, [
        {"turn": 1, "state": "llm", "tokens": 50, "ms": 10, "pass": True},
    ])
    attestation2 = surface.attest(spec2.session_id, reviewer="r")
    assert attestation.frame_hash != attestation2.frame_hash
