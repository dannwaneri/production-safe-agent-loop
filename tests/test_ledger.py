import hashlib
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ledger import Ledger, LedgerRow


@pytest.fixture
def ledger(tmp_path):
    return Ledger(db_path=str(tmp_path / "test.db"))


def test_init_creates_schema(tmp_path):
    db_path = tmp_path / "fresh.db"
    Ledger(db_path=str(db_path))
    with sqlite3.connect(str(db_path)) as conn:
        cols = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(ledger)").fetchall()
        }
    expected = {
        "id": "INTEGER",
        "session_id": "TEXT",
        "turn_count": "INTEGER",
        "state_origin": "TEXT",
        "input_hash": "TEXT",
        "token_delta": "INTEGER",
        "execution_time_ms": "INTEGER",
        "pass_fail": "INTEGER",
        "breach_reason": "TEXT",
        "created_at": "TEXT",
    }
    assert cols == expected


def test_init_is_idempotent(tmp_path):
    db_path = str(tmp_path / "idem.db")
    Ledger(db_path=db_path)
    Ledger(db_path=db_path)


def test_write_returns_row_id_and_persists(ledger):
    row_id = ledger.write(
        session_id="s1",
        turn_count=1,
        state_origin="planner",
        input_str="hello world",
        token_delta=120,
        execution_time_ms=350,
        pass_fail=True,
    )
    assert row_id == 1
    rows = ledger.get_all()
    assert len(rows) == 1
    r = rows[0]
    assert r.session_id == "s1"
    assert r.turn_count == 1
    assert r.state_origin == "planner"
    assert r.token_delta == 120
    assert r.execution_time_ms == 350
    assert r.pass_fail is True
    assert r.breach_reason is None
    assert r.created_at  # ISO 8601 timestamp


def test_input_hash_is_sha256_of_input_str(ledger):
    payload = "the quick brown fox"
    ledger.write(
        session_id="s1",
        turn_count=1,
        state_origin="x",
        input_str=payload,
        token_delta=10,
        execution_time_ms=5,
        pass_fail=True,
    )
    row = ledger.get_all()[0]
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert row.input_hash == expected
    assert len(row.input_hash) == 64


def test_pass_fail_stored_as_zero_or_one(ledger, tmp_path):
    ledger.write(
        session_id="s",
        turn_count=1,
        state_origin="x",
        input_str="a",
        token_delta=0,
        execution_time_ms=0,
        pass_fail=False,
    )
    with sqlite3.connect(ledger.db_path) as conn:
        raw = conn.execute("SELECT pass_fail FROM ledger").fetchone()[0]
    assert raw == 0
    assert isinstance(raw, int)
    rows = ledger.get_all()
    assert rows[0].pass_fail is False


def test_breach_reason_stored_when_provided(ledger):
    ledger.write(
        session_id="s",
        turn_count=6,
        state_origin="executor",
        input_str="x",
        token_delta=0,
        execution_time_ms=12,
        pass_fail=False,
        breach_reason="turn_ceiling",
    )
    row = ledger.get_all()[0]
    assert row.breach_reason == "turn_ceiling"


def test_get_session_filters_by_session_id(ledger):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    for i in range(3):
        ledger.write(a, i, "p", f"in-{i}", 10, 5, True)
    for i in range(2):
        ledger.write(b, i, "p", f"in-{i}", 10, 5, True)

    rows_a = ledger.get_session(a)
    rows_b = ledger.get_session(b)
    assert len(rows_a) == 3
    assert len(rows_b) == 2
    assert all(r.session_id == a for r in rows_a)
    assert all(r.session_id == b for r in rows_b)


def test_two_sessions_have_independent_rows(ledger):
    a, b = "session-a", "session-b"
    ledger.write(a, 1, "p", "data", 100, 10, True)
    ledger.write(b, 1, "p", "data", 200, 20, False)
    ra = ledger.get_session(a)
    rb = ledger.get_session(b)
    assert len(ra) == 1 and len(rb) == 1
    assert ra[0].token_delta == 100
    assert rb[0].token_delta == 200
    assert ra[0].pass_fail is True
    assert rb[0].pass_fail is False


def test_get_all_returns_rows_in_insertion_order(ledger):
    for i in range(5):
        ledger.write("s", i, "p", str(i), 1, 1, True)
    rows = ledger.get_all()
    assert [r.turn_count for r in rows] == [0, 1, 2, 3, 4]
    assert [r.id for r in rows] == [1, 2, 3, 4, 5]


def test_get_session_empty_when_no_match(ledger):
    ledger.write("real", 1, "p", "x", 0, 0, True)
    assert ledger.get_session("missing") == []


def test_get_all_empty_on_fresh_db(ledger):
    assert ledger.get_all() == []


def test_ledger_row_is_frozen_dataclass():
    row = LedgerRow(
        id=1,
        session_id="s",
        turn_count=1,
        state_origin="p",
        input_hash="h",
        token_delta=0,
        execution_time_ms=0,
        pass_fail=True,
        breach_reason=None,
        created_at="2026-06-10T00:00:00+00:00",
    )
    with pytest.raises(Exception):
        row.turn_count = 99  # type: ignore[misc]


def test_created_at_is_iso8601_with_timezone(ledger):
    from datetime import datetime
    ledger.write("s", 1, "p", "x", 0, 0, True)
    row = ledger.get_all()[0]
    parsed = datetime.fromisoformat(row.created_at)
    assert parsed.tzinfo is not None
