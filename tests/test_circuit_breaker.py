import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from circuit_breaker import CircuitBreaker, CircuitBreakerError


def test_defaults():
    cb = CircuitBreaker()
    assert cb.turn_limit == 5
    assert cb.token_limit == 15000


def test_custom_limits():
    cb = CircuitBreaker(turn_limit=10, token_limit=50000)
    assert cb.turn_limit == 10
    assert cb.token_limit == 50000


def test_check_under_both_limits_does_not_raise():
    cb = CircuitBreaker(turn_limit=5, token_limit=15000)
    cb.check(turn_count=3, accumulated_tokens=5000)


def test_check_at_turn_limit_does_not_raise():
    cb = CircuitBreaker(turn_limit=5, token_limit=15000)
    cb.check(turn_count=5, accumulated_tokens=14999)


def test_check_at_token_limit_does_not_raise():
    cb = CircuitBreaker(turn_limit=5, token_limit=15000)
    cb.check(turn_count=4, accumulated_tokens=15000)


def test_turn_ceiling_trips_on_turn_six(capsys):
    cb = CircuitBreaker(turn_limit=5, token_limit=15000)
    with pytest.raises(CircuitBreakerError) as exc:
        cb.check(turn_count=6, accumulated_tokens=1000)
    assert exc.value.reason == "turn_ceiling"
    assert exc.value.turn_count == 6
    assert exc.value.accumulated_tokens == 1000
    captured = capsys.readouterr()
    assert "CIRCUIT BREAKER CHECKPOINT" in captured.out
    assert "turn_ceiling" in captured.out


def test_token_ceiling_trips_when_tokens_exceed_limit(capsys):
    cb = CircuitBreaker(turn_limit=5, token_limit=15000)
    with pytest.raises(CircuitBreakerError) as exc:
        cb.check(turn_count=2, accumulated_tokens=15001)
    assert exc.value.reason == "token_ceiling"
    assert exc.value.turn_count == 2
    assert exc.value.accumulated_tokens == 15001
    captured = capsys.readouterr()
    assert "token_ceiling" in captured.out


def test_token_ceiling_independent_of_turn_count():
    cb = CircuitBreaker(turn_limit=5, token_limit=15000)
    with pytest.raises(CircuitBreakerError) as exc:
        cb.check(turn_count=1, accumulated_tokens=20000)
    assert exc.value.reason == "token_ceiling"


def test_turn_breach_fires_before_token_breach_when_both_exceeded():
    cb = CircuitBreaker(turn_limit=5, token_limit=15000)
    with pytest.raises(CircuitBreakerError) as exc:
        cb.check(turn_count=6, accumulated_tokens=20000)
    assert exc.value.reason == "turn_ceiling"


def test_error_carries_state_on_raise():
    err = CircuitBreakerError(
        reason="token_ceiling",
        turn_count=4,
        accumulated_tokens=15500,
    )
    assert err.reason == "token_ceiling"
    assert err.turn_count == 4
    assert err.accumulated_tokens == 15500
    assert "token_ceiling" in str(err)
    assert "turn=4" in str(err)
    assert "tokens=15500" in str(err)


def test_error_is_exception_subclass():
    err = CircuitBreakerError(reason="turn_ceiling", turn_count=6, accumulated_tokens=0)
    assert isinstance(err, Exception)


def test_checkpoint_message_contains_limits(capsys):
    cb = CircuitBreaker(turn_limit=3, token_limit=1000)
    with pytest.raises(CircuitBreakerError):
        cb.check(turn_count=4, accumulated_tokens=500)
    captured = capsys.readouterr()
    assert "limit 3" in captured.out
    assert "limit 1000" in captured.out
