import time
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from circuit_breaker import CircuitBreaker, CircuitBreakerError
from ledger import Ledger
from spec_writer import SpecResult


@runtime_checkable
class MessagesEndpoint(Protocol):
    def create(self, *, model: str, max_tokens: int, system: str, messages: list) -> object: ...


@runtime_checkable
class LLMClient(Protocol):
    messages: MessagesEndpoint


@dataclass(frozen=True)
class LoopResult:
    success: bool
    turns: int
    total_tokens: int
    session_id: str
    breach_reason: Optional[str] = None


class AgentLoop:
    def __init__(
        self,
        spec: SpecResult,
        circuit_breaker: CircuitBreaker,
        ledger: Ledger,
        client: LLMClient,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
    ) -> None:
        self.spec = spec
        self.circuit_breaker = circuit_breaker
        self.ledger = ledger
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def run(self, task: str) -> LoopResult:
        session_id = self.spec.session_id
        messages: list[dict] = [{"role": "user", "content": task}]
        turn = 0
        total_tokens = 0

        try:
            while True:
                turn += 1
                self.circuit_breaker.check(turn, total_tokens)

                started = time.perf_counter()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self._system_prompt(),
                    messages=messages,
                )
                elapsed_ms = int((time.perf_counter() - started) * 1000)

                turn_tokens = (
                    getattr(response.usage, "input_tokens", 0)
                    + getattr(response.usage, "output_tokens", 0)
                )
                total_tokens += turn_tokens

                text = self._text_from(response)
                messages.append({"role": "assistant", "content": text})

                self.ledger.write(
                    session_id=session_id,
                    turn_count=turn,
                    state_origin="llm",
                    input_str=task,
                    token_delta=turn_tokens,
                    execution_time_ms=elapsed_ms,
                    pass_fail=True,
                )

                if getattr(response, "stop_reason", "end_turn") == "end_turn":
                    return LoopResult(
                        success=True,
                        turns=turn,
                        total_tokens=total_tokens,
                        session_id=session_id,
                    )

                messages.append({"role": "user", "content": "continue"})

        except CircuitBreakerError as err:
            self.ledger.write(
                session_id=session_id,
                turn_count=turn,
                state_origin="circuit_breaker",
                input_str=task,
                token_delta=0,
                execution_time_ms=0,
                pass_fail=False,
                breach_reason=err.reason,
            )
            return LoopResult(
                success=False,
                turns=turn,
                total_tokens=total_tokens,
                session_id=session_id,
                breach_reason=err.reason,
            )

    def _system_prompt(self) -> str:
        return (
            "You are an agent working on a tightly-scoped task.\n\n"
            f"What this does: {self.spec.what_it_does}\n"
            f"What this does NOT do: {self.spec.what_it_does_not}\n"
            f"Done looks like: {self.spec.done_looks_like}\n"
        )

    @staticmethod
    def _text_from(response) -> str:
        content = getattr(response, "content", None)
        if not content:
            return ""
        block = content[0]
        return getattr(block, "text", "") or ""
