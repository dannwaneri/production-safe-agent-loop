from dataclasses import dataclass


@dataclass
class CircuitBreakerError(Exception):
    reason: str
    turn_count: int
    accumulated_tokens: int

    def __post_init__(self) -> None:
        super().__init__(
            f"circuit breaker tripped: {self.reason} "
            f"(turn={self.turn_count}, tokens={self.accumulated_tokens})"
        )


class CircuitBreaker:
    def __init__(self, turn_limit: int = 5, token_limit: int = 15000) -> None:
        self.turn_limit = turn_limit
        self.token_limit = token_limit

    def check(self, turn_count: int, accumulated_tokens: int) -> None:
        if turn_count > self.turn_limit:
            self._trip("turn_ceiling", turn_count, accumulated_tokens)
        if accumulated_tokens > self.token_limit:
            self._trip("token_ceiling", turn_count, accumulated_tokens)

    def _trip(self, reason: str, turn_count: int, accumulated_tokens: int) -> None:
        print(
            "\n=== CIRCUIT BREAKER CHECKPOINT ===\n"
            f"reason         : {reason}\n"
            f"turn_count     : {turn_count} / limit {self.turn_limit}\n"
            f"tokens_used    : {accumulated_tokens} / limit {self.token_limit}\n"
            "action         : halt loop, surface to human reviewer\n"
            "=================================="
        )
        raise CircuitBreakerError(
            reason=reason,
            turn_count=turn_count,
            accumulated_tokens=accumulated_tokens,
        )
