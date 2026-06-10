import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_loop import AgentLoop, LoopResult
from circuit_breaker import CircuitBreaker
from ledger import Ledger
from spec_writer import SpecResult


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeBlock:
    text: str
    type: str = "text"


@dataclass
class FakeResponse:
    usage: FakeUsage
    content: list
    stop_reason: str = "end_turn"


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self._responses:
            raise AssertionError("FakeClient: ran out of scripted responses")
        return self._responses.pop(0)


def make_response(input_tokens=10, output_tokens=20, text="ok", stop_reason="end_turn"):
    return FakeResponse(
        usage=FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        content=[FakeBlock(text=text)],
        stop_reason=stop_reason,
    )


@pytest.fixture
def spec():
    return SpecResult(
        what_it_does="audit one URL",
        what_it_does_not="render JS",
        done_looks_like="report printed",
        session_id="spec-sid-1",
    )


@pytest.fixture
def ledger(tmp_path):
    return Ledger(db_path=str(tmp_path / "ledger.db"))


@pytest.fixture
def breaker():
    return CircuitBreaker(turn_limit=5, token_limit=15000)


def test_completes_on_first_turn_with_end_turn(spec, breaker, ledger):
    client = FakeClient([make_response(50, 50, "done", "end_turn")])
    loop = AgentLoop(spec, breaker, ledger, client)
    result = loop.run("do the thing")
    assert result == LoopResult(
        success=True,
        turns=1,
        total_tokens=100,
        session_id="spec-sid-1",
        breach_reason=None,
    )
    assert len(client.calls) == 1


def test_continues_until_end_turn(spec, breaker, ledger):
    client = FakeClient([
        make_response(10, 20, "thinking", "tool_use"),
        make_response(15, 25, "still thinking", "tool_use"),
        make_response(5, 15, "done", "end_turn"),
    ])
    loop = AgentLoop(spec, breaker, ledger, client)
    result = loop.run("multi-turn task")
    assert result.success is True
    assert result.turns == 3
    assert result.total_tokens == 10 + 20 + 15 + 25 + 5 + 15
    assert len(client.calls) == 3


def test_ledger_receives_one_row_per_turn(spec, breaker, ledger):
    client = FakeClient([
        make_response(10, 10, "a", "tool_use"),
        make_response(20, 20, "b", "end_turn"),
    ])
    loop = AgentLoop(spec, breaker, ledger, client)
    loop.run("two turn task")
    rows = ledger.get_session("spec-sid-1")
    assert len(rows) == 2
    assert [r.turn_count for r in rows] == [1, 2]
    assert all(r.state_origin == "llm" for r in rows)
    assert all(r.pass_fail is True for r in rows)
    assert [r.token_delta for r in rows] == [20, 40]


def test_turn_ceiling_breach_returns_failed_result(spec, ledger):
    breaker = CircuitBreaker(turn_limit=2, token_limit=15000)
    client = FakeClient([
        make_response(10, 10, "x", "tool_use"),
        make_response(10, 10, "x", "tool_use"),
        make_response(10, 10, "x", "tool_use"),
    ])
    loop = AgentLoop(spec, breaker, ledger, client)
    result = loop.run("will breach turn")
    assert result.success is False
    assert result.breach_reason == "turn_ceiling"
    assert result.turns == 3
    assert result.session_id == "spec-sid-1"
    assert len(client.calls) == 2


def test_token_ceiling_breach_returns_failed_result(spec, ledger):
    breaker = CircuitBreaker(turn_limit=10, token_limit=100)
    client = FakeClient([
        make_response(50, 60, "x", "tool_use"),
        make_response(10, 10, "x", "tool_use"),
    ])
    loop = AgentLoop(spec, breaker, ledger, client)
    result = loop.run("will breach tokens")
    assert result.success is False
    assert result.breach_reason == "token_ceiling"
    assert result.total_tokens == 110
    assert len(client.calls) == 1


def test_breach_is_logged_to_ledger(spec, ledger):
    breaker = CircuitBreaker(turn_limit=1, token_limit=15000)
    client = FakeClient([
        make_response(5, 5, "x", "tool_use"),
        make_response(5, 5, "x", "tool_use"),
    ])
    loop = AgentLoop(spec, breaker, ledger, client)
    loop.run("breach me")
    rows = ledger.get_session("spec-sid-1")
    last = rows[-1]
    assert last.state_origin == "circuit_breaker"
    assert last.breach_reason == "turn_ceiling"
    assert last.pass_fail is False


def test_session_id_matches_spec(spec, breaker, ledger):
    client = FakeClient([make_response()])
    loop = AgentLoop(spec, breaker, ledger, client)
    result = loop.run("task")
    assert result.session_id == spec.session_id


def test_system_prompt_contains_spec_content(spec, breaker, ledger):
    client = FakeClient([make_response()])
    loop = AgentLoop(spec, breaker, ledger, client)
    loop.run("task")
    sys_prompt = client.calls[0]["system"]
    assert spec.what_it_does in sys_prompt
    assert spec.what_it_does_not in sys_prompt
    assert spec.done_looks_like in sys_prompt


def test_model_and_max_tokens_passed_to_client(spec, breaker, ledger):
    client = FakeClient([make_response()])
    loop = AgentLoop(spec, breaker, ledger, client, model="claude-opus-4-8", max_tokens=2048)
    loop.run("task")
    assert client.calls[0]["model"] == "claude-opus-4-8"
    assert client.calls[0]["max_tokens"] == 2048


def test_default_model_and_max_tokens(spec, breaker, ledger):
    client = FakeClient([make_response()])
    loop = AgentLoop(spec, breaker, ledger, client)
    loop.run("task")
    assert client.calls[0]["model"] == "claude-sonnet-4-6"
    assert client.calls[0]["max_tokens"] == 1024


def test_loop_result_is_frozen():
    r = LoopResult(success=True, turns=1, total_tokens=0, session_id="s")
    with pytest.raises(Exception):
        r.turns = 99  # type: ignore[misc]


def test_text_extracted_from_response_content(spec, breaker, ledger):
    client = FakeClient([make_response(text="hello world")])
    loop = AgentLoop(spec, breaker, ledger, client)
    loop.run("task")
    assert client.calls[0]["messages"] == [{"role": "user", "content": "task"}]


def test_empty_content_handled_gracefully(spec, breaker, ledger):
    response = FakeResponse(
        usage=FakeUsage(input_tokens=5, output_tokens=5),
        content=[],
        stop_reason="end_turn",
    )
    client = FakeClient([response])
    loop = AgentLoop(spec, breaker, ledger, client)
    result = loop.run("task")
    assert result.success is True


def test_block_without_text_attribute_returns_empty(spec, breaker, ledger):
    @dataclass
    class WeirdBlock:
        type: str = "image"
    response = FakeResponse(
        usage=FakeUsage(input_tokens=5, output_tokens=5),
        content=[WeirdBlock()],
        stop_reason="end_turn",
    )
    client = FakeClient([response])
    loop = AgentLoop(spec, breaker, ledger, client)
    result = loop.run("task")
    assert result.success is True


def test_messages_grow_across_turns(spec, breaker, ledger):
    client = FakeClient([
        make_response(text="first", stop_reason="tool_use"),
        make_response(text="second", stop_reason="end_turn"),
    ])
    loop = AgentLoop(spec, breaker, ledger, client)
    loop.run("starter")
    msgs_first = client.calls[0]["messages"]
    msgs_second = client.calls[1]["messages"]
    assert len(msgs_first) == 1
    assert len(msgs_second) == 3
    assert msgs_second[0] == {"role": "user", "content": "starter"}
    assert msgs_second[1] == {"role": "assistant", "content": "first"}
    assert msgs_second[2] == {"role": "user", "content": "continue"}


def test_response_without_stop_reason_treated_as_end_turn(spec, breaker, ledger):
    @dataclass
    class MinimalResponse:
        usage: FakeUsage
        content: list
    response = MinimalResponse(
        usage=FakeUsage(input_tokens=5, output_tokens=5),
        content=[FakeBlock(text="ok")],
    )
    client = FakeClient([response])
    loop = AgentLoop(spec, breaker, ledger, client)
    result = loop.run("task")
    assert result.success is True
    assert result.turns == 1
