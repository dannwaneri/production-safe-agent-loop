import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS spec (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    what_it_does TEXT NOT NULL,
    what_it_does_not TEXT NOT NULL,
    done_looks_like TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spec_session ON spec(session_id);
"""


QUESTIONS = (
    ("what_it_does", "What does this do?"),
    ("what_it_does_not", "What does this NOT do?"),
    ("done_looks_like", "What does done look like in one sentence?"),
)


@dataclass(frozen=True)
class SpecResult:
    what_it_does: str
    what_it_does_not: str
    done_looks_like: str
    session_id: str


class SpecWriter:
    def __init__(
        self,
        db_path: str = "spec.db",
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self.db_path = db_path
        self.input_fn = input_fn
        self.output_fn = output_fn
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def run(self) -> SpecResult:
        answers: dict[str, str] = {}
        for field, prompt in QUESTIONS:
            while True:
                raw = self.input_fn(f"{prompt} ")
                ans = raw.strip() if raw is not None else ""
                if ans:
                    answers[field] = ans
                    break
                self.output_fn(
                    f"[missing answer for: {prompt}] "
                    "spec cannot proceed until all three questions are answered."
                )
        result = SpecResult(
            what_it_does=answers["what_it_does"],
            what_it_does_not=answers["what_it_does_not"],
            done_looks_like=answers["done_looks_like"],
            session_id=str(uuid.uuid4()),
        )
        self._save(result)
        return result

    def _save(self, result: SpecResult) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO spec (
                    session_id, what_it_does, what_it_does_not,
                    done_looks_like, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.session_id,
                    result.what_it_does,
                    result.what_it_does_not,
                    result.done_looks_like,
                    created_at,
                ),
            )

    def load(self, session_id: str) -> Optional[SpecResult]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT what_it_does, what_it_does_not, done_looks_like, session_id "
                "FROM spec WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SpecResult(
            what_it_does=row[0],
            what_it_does_not=row[1],
            done_looks_like=row[2],
            session_id=row[3],
        )
