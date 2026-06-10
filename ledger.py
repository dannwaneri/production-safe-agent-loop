import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_count INTEGER NOT NULL,
    state_origin TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    token_delta INTEGER NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    pass_fail INTEGER NOT NULL,
    breach_reason TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_session ON ledger(session_id);
"""


@dataclass(frozen=True)
class LedgerRow:
    id: int
    session_id: str
    turn_count: int
    state_origin: str
    input_hash: str
    token_delta: int
    execution_time_ms: int
    pass_fail: bool
    breach_reason: Optional[str]
    created_at: str


class Ledger:
    def __init__(self, db_path: str = "ledger.db") -> None:
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def write(
        self,
        session_id: str,
        turn_count: int,
        state_origin: str,
        input_str: str,
        token_delta: int,
        execution_time_ms: int,
        pass_fail: bool,
        breach_reason: Optional[str] = None,
    ) -> int:
        input_hash = hashlib.sha256(input_str.encode("utf-8")).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO ledger (
                    session_id, turn_count, state_origin, input_hash,
                    token_delta, execution_time_ms, pass_fail,
                    breach_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_count,
                    state_origin,
                    input_hash,
                    token_delta,
                    execution_time_ms,
                    int(bool(pass_fail)),
                    breach_reason,
                    created_at,
                ),
            )
            return cur.lastrowid

    def get_session(self, session_id: str) -> list[LedgerRow]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, session_id, turn_count, state_origin, input_hash, "
                "token_delta, execution_time_ms, pass_fail, breach_reason, created_at "
                "FROM ledger WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def get_all(self) -> list[LedgerRow]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, session_id, turn_count, state_origin, input_hash, "
                "token_delta, execution_time_ms, pass_fail, breach_reason, created_at "
                "FROM ledger ORDER BY id ASC"
            ).fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(r: tuple) -> LedgerRow:
        return LedgerRow(
            id=r[0],
            session_id=r[1],
            turn_count=r[2],
            state_origin=r[3],
            input_hash=r[4],
            token_delta=r[5],
            execution_time_ms=r[6],
            pass_fail=bool(r[7]),
            breach_reason=r[8],
            created_at=r[9],
        )
