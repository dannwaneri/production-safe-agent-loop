import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spec_writer import SpecResult, SpecWriter


def scripted_input(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def collecting_output():
    captured = []
    return captured, captured.append


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "spec.db")


def test_init_creates_schema(db_path):
    SpecWriter(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        cols = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(spec)").fetchall()
        }
    assert cols == {
        "id": "INTEGER",
        "session_id": "TEXT",
        "what_it_does": "TEXT",
        "what_it_does_not": "TEXT",
        "done_looks_like": "TEXT",
        "created_at": "TEXT",
    }


def test_init_is_idempotent(db_path):
    SpecWriter(db_path=db_path)
    SpecWriter(db_path=db_path)


def test_run_returns_spec_with_all_three_answers(db_path):
    sw = SpecWriter(
        db_path=db_path,
        input_fn=scripted_input(["crawl one URL", "no JS rendering", "report printed to stdout"]),
        output_fn=lambda _: None,
    )
    result = sw.run()
    assert result.what_it_does == "crawl one URL"
    assert result.what_it_does_not == "no JS rendering"
    assert result.done_looks_like == "report printed to stdout"


def test_run_generates_uuid4_session_id(db_path):
    sw = SpecWriter(
        db_path=db_path,
        input_fn=scripted_input(["a", "b", "c"]),
        output_fn=lambda _: None,
    )
    result = sw.run()
    parsed = uuid.UUID(result.session_id)
    assert parsed.version == 4


def test_two_runs_have_distinct_session_ids(db_path):
    answers = ["a", "b", "c", "d", "e", "f"]
    sw = SpecWriter(
        db_path=db_path,
        input_fn=scripted_input(answers),
        output_fn=lambda _: None,
    )
    r1 = sw.run()
    r2 = sw.run()
    assert r1.session_id != r2.session_id


def test_run_stores_to_sqlite(db_path):
    sw = SpecWriter(
        db_path=db_path,
        input_fn=scripted_input(["does X", "not Y", "Z works"]),
        output_fn=lambda _: None,
    )
    result = sw.run()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT session_id, what_it_does, what_it_does_not, done_looks_like "
            "FROM spec WHERE session_id = ?",
            (result.session_id,),
        ).fetchone()
    assert row == (result.session_id, "does X", "not Y", "Z works")


def test_run_blocks_on_empty_answer_then_continues(db_path):
    captured, output_fn = collecting_output()
    sw = SpecWriter(
        db_path=db_path,
        input_fn=scripted_input(["", "real what", "real not", "real done"]),
        output_fn=output_fn,
    )
    result = sw.run()
    assert result.what_it_does == "real what"
    assert any("missing answer" in m for m in captured)
    assert any("What does this do?" in m for m in captured)


def test_run_blocks_on_whitespace_only_answer(db_path):
    captured, output_fn = collecting_output()
    sw = SpecWriter(
        db_path=db_path,
        input_fn=scripted_input(["   ", "\t\n", "real what", "real not", "real done"]),
        output_fn=output_fn,
    )
    result = sw.run()
    assert result.what_it_does == "real what"
    assert sum(1 for m in captured if "missing answer" in m) == 2


def test_run_blocks_each_question_independently(db_path):
    captured, output_fn = collecting_output()
    sw = SpecWriter(
        db_path=db_path,
        input_fn=scripted_input(["q1", "", "q2", "", "q3"]),
        output_fn=output_fn,
    )
    sw.run()
    msgs = [m for m in captured if "missing answer" in m]
    assert len(msgs) == 2
    assert any("NOT do" in m for m in msgs)
    assert any("done look like" in m for m in msgs)


def test_load_retrieves_stored_result(db_path):
    sw = SpecWriter(
        db_path=db_path,
        input_fn=scripted_input(["x", "y", "z"]),
        output_fn=lambda _: None,
    )
    written = sw.run()
    loaded = sw.load(written.session_id)
    assert loaded == written


def test_load_returns_none_for_unknown_session(db_path):
    sw = SpecWriter(db_path=db_path)
    assert sw.load("nonexistent-session") is None


def test_spec_result_is_frozen():
    r = SpecResult(
        what_it_does="a",
        what_it_does_not="b",
        done_looks_like="c",
        session_id="sid",
    )
    with pytest.raises(Exception):
        r.what_it_does = "mutated"  # type: ignore[misc]


def test_load_after_separate_writer_instance(db_path):
    sw1 = SpecWriter(
        db_path=db_path,
        input_fn=scripted_input(["does it", "not this", "ship it"]),
        output_fn=lambda _: None,
    )
    written = sw1.run()
    sw2 = SpecWriter(db_path=db_path)
    loaded = sw2.load(written.session_id)
    assert loaded == written


def test_default_input_output_callables():
    sw = SpecWriter(db_path=":memory:")
    assert sw.input_fn is __builtins__["input"] if isinstance(__builtins__, dict) else __builtins__.input
    assert sw.output_fn is print
