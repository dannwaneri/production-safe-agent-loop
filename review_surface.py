import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ledger import Ledger, LedgerRow
from spec_writer import SpecResult, SpecWriter


ATTESTATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS attestations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    attested_at TEXT NOT NULL,
    notes TEXT,
    frame_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attestation_session ON attestations(session_id);
"""


class SessionNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class DiffResult:
    first_input: str
    final_output: str
    turns_completed: int
    total_tokens: int
    breached: bool


@dataclass(frozen=True)
class ReviewFrame:
    session_id: str
    original_promise: SpecResult
    acceptance_criteria: str
    diff: DiffResult
    evidence: tuple
    unresolved_assumptions: tuple
    created_at: str


@dataclass(frozen=True)
class AttestationResult:
    id: int
    session_id: str
    reviewer: str
    attested_at: str
    notes: str
    frame_hash: str


class ReviewSurface:
    def __init__(
        self,
        spec_db_path: str = "spec.db",
        ledger_db_path: str = "ledger.db",
    ) -> None:
        self.spec_db_path = spec_db_path
        self.ledger_db_path = ledger_db_path
        self._spec_writer = SpecWriter(
            db_path=spec_db_path,
            input_fn=lambda _: "",
            output_fn=lambda _: None,
        )
        self._ledger = Ledger(db_path=ledger_db_path)
        with sqlite3.connect(self.ledger_db_path) as conn:
            conn.executescript(ATTESTATION_SCHEMA)

    def load(self, session_id: str) -> ReviewFrame:
        spec = self._spec_writer.load(session_id)
        rows = self._ledger.get_session(session_id)
        if spec is None and not rows:
            raise SessionNotFoundError(
                f"no spec or ledger rows found for session_id={session_id!r}"
            )
        if spec is None:
            raise SessionNotFoundError(
                f"ledger rows exist but spec missing for session_id={session_id!r}"
            )
        if not rows:
            raise SessionNotFoundError(
                f"spec exists but no ledger rows for session_id={session_id!r}"
            )

        diff = self._build_diff(rows)
        unresolved = self._unresolved_assumptions(rows)
        return ReviewFrame(
            session_id=session_id,
            original_promise=spec,
            acceptance_criteria=spec.done_looks_like,
            diff=diff,
            evidence=tuple(rows),
            unresolved_assumptions=tuple(unresolved),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def render(self, session_id: str) -> str:
        frame = self.load(session_id)
        lines = []
        bar = "=" * 64
        lines.append(bar)
        lines.append("REVIEW SURFACE")
        lines.append(f"session_id: {frame.session_id}")
        lines.append(f"loaded_at:  {frame.created_at}")
        lines.append(bar)
        lines.append("")
        lines.append("[1] ORIGINAL PROMISE")
        lines.append(f"    What this does:      {frame.original_promise.what_it_does}")
        lines.append(f"    What this does NOT:  {frame.original_promise.what_it_does_not}")
        lines.append(f"    Done looks like:     {frame.original_promise.done_looks_like}")
        lines.append("")
        lines.append("[2] ACCEPTANCE CRITERIA")
        lines.append(f"    {frame.acceptance_criteria}")
        lines.append("")
        lines.append("[3] DIFF")
        lines.append(f"    First input hash:    {frame.diff.first_input}")
        lines.append(f"    Final state origin:  {frame.diff.final_output}")
        lines.append(f"    Turns completed:     {frame.diff.turns_completed}")
        lines.append(f"    Total tokens:        {frame.diff.total_tokens}")
        lines.append(f"    Breached:            {frame.diff.breached}")
        lines.append("")
        lines.append("[4] EVIDENCE")
        for row in frame.evidence:
            verdict = "PASS" if row.pass_fail else "FAIL"
            breach = f" breach={row.breach_reason}" if row.breach_reason else ""
            lines.append(
                f"    turn={row.turn_count:<3} state={row.state_origin:<16} "
                f"tokens={row.token_delta:<6} ms={row.execution_time_ms:<6} "
                f"{verdict}{breach}"
            )
        lines.append("")
        lines.append("[5] UNRESOLVED ASSUMPTIONS")
        if not frame.unresolved_assumptions:
            lines.append("    No unresolved assumptions detected.")
        else:
            for note in frame.unresolved_assumptions:
                lines.append(f"    - {note}")
        lines.append("")
        lines.append(bar)
        return "\n".join(lines)

    def attest(
        self,
        session_id: str,
        reviewer: str,
        notes: str = "",
    ) -> AttestationResult:
        frame = self.load(session_id)
        frame_hash = self._frame_hash(frame)
        attested_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.ledger_db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO attestations (
                    session_id, reviewer, attested_at, notes, frame_hash
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, reviewer, attested_at, notes, frame_hash),
            )
            row_id = cur.lastrowid
        return AttestationResult(
            id=row_id,
            session_id=session_id,
            reviewer=reviewer,
            attested_at=attested_at,
            notes=notes,
            frame_hash=frame_hash,
        )

    def get_attestations(self, session_id: str) -> list[AttestationResult]:
        with sqlite3.connect(self.ledger_db_path) as conn:
            rows = conn.execute(
                "SELECT id, session_id, reviewer, attested_at, notes, frame_hash "
                "FROM attestations WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [
            AttestationResult(
                id=r[0],
                session_id=r[1],
                reviewer=r[2],
                attested_at=r[3],
                notes=r[4] or "",
                frame_hash=r[5],
            )
            for r in rows
        ]

    @staticmethod
    def _build_diff(rows: list[LedgerRow]) -> DiffResult:
        llm_rows = [r for r in rows if r.state_origin == "llm"]
        total_tokens = sum(r.token_delta for r in rows)
        breached = any(
            r.state_origin == "circuit_breaker" or not r.pass_fail for r in rows
        )
        if llm_rows:
            final_output = llm_rows[-1].state_origin
        else:
            final_output = rows[-1].state_origin
        return DiffResult(
            first_input=rows[0].input_hash,
            final_output=final_output,
            turns_completed=len(llm_rows),
            total_tokens=total_tokens,
            breached=breached,
        )

    @staticmethod
    def _unresolved_assumptions(rows: list[LedgerRow]) -> list[str]:
        notes: list[str] = []
        for r in rows:
            if r.breach_reason:
                notes.append(
                    f"turn {r.turn_count}: circuit_breaker fired "
                    f"({r.breach_reason})"
                )
            elif not r.pass_fail:
                notes.append(f"turn {r.turn_count}: pass_fail=False")
        return notes

    @staticmethod
    def _frame_hash(frame: ReviewFrame) -> str:
        parts: list[str] = [
            f"session_id:{frame.session_id}",
            f"what_it_does:{frame.original_promise.what_it_does}",
            f"what_it_does_not:{frame.original_promise.what_it_does_not}",
            f"done_looks_like:{frame.original_promise.done_looks_like}",
            f"acceptance_criteria:{frame.acceptance_criteria}",
            f"diff.first_input:{frame.diff.first_input}",
            f"diff.final_output:{frame.diff.final_output}",
            f"diff.turns_completed:{frame.diff.turns_completed}",
            f"diff.total_tokens:{frame.diff.total_tokens}",
            f"diff.breached:{frame.diff.breached}",
        ]
        for row in frame.evidence:
            parts.append(
                f"row:{row.turn_count}|{row.state_origin}|{row.input_hash}|"
                f"{row.token_delta}|{int(row.pass_fail)}|{row.breach_reason or ''}"
            )
        for note in frame.unresolved_assumptions:
            parts.append(f"assumption:{note}")
        canonical = "\n".join(parts)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
