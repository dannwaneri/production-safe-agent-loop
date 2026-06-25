"""Independent verification layer for the production-safe agent loop.

Addresses the structural gap in ReviewSurface: the same agent process
that makes decisions also writes the ledger rows that get assembled into
the review frame, so the "evidence" element is self-curated. This module
cross-checks the agent's self-reported ledger against Anthropic's
billing-side usage records — a genuinely independent signal served from
a different motive than the agent's self-favorable reporting.

What it catches: ghost calls (billing has records the ledger does not),
missing records (ledger has rows billing doesn't), model substitution
(ledger claims one model, billing shows another), token-count drift
beyond infra-noise tolerance, and per-actor accumulation patterns.

What it does NOT catch: pure content-level fabrication with otherwise-
honest metadata. Stated limitation — do not oversell as full content
audit.

Ships in shadow mode first (log-only, no blocking). Real thresholds get
calibrated from real shadow-mode data; defaults here are conservative.
"""
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Protocol, runtime_checkable


SCHEMA = """
CREATE TABLE IF NOT EXISTS fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    bucket_start TEXT NOT NULL,
    model_id TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    tool_call_count INTEGER NOT NULL,
    api_key_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fp_bucket ON fingerprints(bucket_start);
CREATE INDEX IF NOT EXISTS idx_fp_actor ON fingerprints(api_key_id, workspace_id);
CREATE INDEX IF NOT EXISTS idx_fp_session ON fingerprints(session_id);

CREATE TABLE IF NOT EXISTS diff_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bucket_start TEXT NOT NULL,
    bucket_end TEXT NOT NULL,
    model_id TEXT NOT NULL,
    api_key_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    ledger_call_count INTEGER NOT NULL,
    usage_records_present INTEGER NOT NULL,
    drift_findings_json TEXT NOT NULL,
    severity TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dr_bucket ON diff_reports(bucket_start);
CREATE INDEX IF NOT EXISTS idx_dr_actor ON diff_reports(api_key_id);
CREATE INDEX IF NOT EXISTS idx_dr_severity ON diff_reports(severity);
"""


@runtime_checkable
class UsageRecordsClient(Protocol):
    def get_messages_usage_report(
        self,
        *,
        starting_at: str,
        ending_at: Optional[str] = None,
        bucket_width: str = "1m",
        group_by: Optional[list] = None,
        models: Optional[list] = None,
        api_key_ids: Optional[list] = None,
        workspace_ids: Optional[list] = None,
        page: Optional[str] = None,
    ) -> dict: ...


@dataclass(frozen=True)
class Fingerprint:
    id: int
    timestamp_utc: str
    bucket_start: str
    model_id: str
    tokens_in: int
    tokens_out: int
    tool_call_count: int
    api_key_id: str
    workspace_id: str
    session_id: str
    turn_count: int


@dataclass(frozen=True)
class BucketAggregate:
    bucket_start: str
    bucket_end: str
    model_id: str
    api_key_id: str
    workspace_id: str
    call_count: int
    tokens_in: int
    tokens_out: int
    tool_call_count: int


@dataclass(frozen=True)
class UsageRecord:
    starting_at: str
    ending_at: str
    model: Optional[str]
    api_key_id: Optional[str]
    workspace_id: Optional[str]
    uncached_input_tokens: int
    cache_read_input_tokens: int
    cache_creation_5m_input_tokens: int
    cache_creation_1h_input_tokens: int
    output_tokens: int
    server_tool_use_count: int

    @property
    def total_tokens(self) -> int:
        return (
            self.uncached_input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_5m_input_tokens
            + self.cache_creation_1h_input_tokens
            + self.output_tokens
        )


@dataclass(frozen=True)
class DriftFinding:
    drift_class: str
    severity: str
    details: str

    def to_dict(self) -> dict:
        return {
            "drift_class": self.drift_class,
            "severity": self.severity,
            "details": self.details,
        }


@dataclass(frozen=True)
class DiffResult:
    id: int
    bucket_start: str
    bucket_end: str
    model_id: str
    api_key_id: str
    workspace_id: str
    ledger_call_count: int
    usage_records_present: bool
    drift_findings: tuple
    severity: str
    created_at: str


@dataclass(frozen=True)
class ThresholdConfig:
    model_id_mismatch_tolerance: int = 0
    token_mismatch_percent_infra: float = 0.05
    token_mismatch_percent_adversarial: float = 0.20
    ghost_call_count_threshold: int = 1
    actor_window_minutes: int = 15
    actor_drift_threshold: int = 3


class DriftAlert(Exception):
    def __init__(self, result: DiffResult) -> None:
        super().__init__(
            f"drift detected: {result.severity} at {result.bucket_start} "
            f"(model={result.model_id}, actor={result.api_key_id})"
        )
        self.result = result


class AttestationFingerprint:
    def __init__(
        self,
        ledger_db_path: str,
        usage_client: UsageRecordsClient,
        api_key_id: str,
        workspace_id: str,
        threshold_config: Optional[ThresholdConfig] = None,
        shadow_mode: bool = True,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.ledger_db_path = ledger_db_path
        self.usage_client = usage_client
        self.api_key_id = api_key_id
        self.workspace_id = workspace_id
        self.config = threshold_config or ThresholdConfig()
        self.shadow_mode = shadow_mode
        self._clock = clock
        with sqlite3.connect(self.ledger_db_path) as conn:
            conn.executescript(SCHEMA)

    def record_fingerprint(
        self,
        *,
        session_id: str,
        turn_count: int,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        tool_call_count: int = 0,
        timestamp: Optional[datetime] = None,
    ) -> int:
        ts = timestamp or self._clock()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bucket = ts.replace(second=0, microsecond=0)
        with sqlite3.connect(self.ledger_db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO fingerprints (
                    timestamp_utc, bucket_start, model_id, tokens_in, tokens_out,
                    tool_call_count, api_key_id, workspace_id, session_id,
                    turn_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts.isoformat(), bucket.isoformat(), model_id, tokens_in,
                    tokens_out, tool_call_count, self.api_key_id,
                    self.workspace_id, session_id, turn_count,
                    self._clock().isoformat(),
                ),
            )
            return cur.lastrowid

    def get_fingerprints_in_window(
        self, starting_at: datetime, ending_at: datetime
    ) -> list:
        with sqlite3.connect(self.ledger_db_path) as conn:
            rows = conn.execute(
                "SELECT id, timestamp_utc, bucket_start, model_id, tokens_in, "
                "tokens_out, tool_call_count, api_key_id, workspace_id, "
                "session_id, turn_count FROM fingerprints "
                "WHERE timestamp_utc >= ? AND timestamp_utc < ? "
                "ORDER BY timestamp_utc ASC",
                (self._iso(starting_at), self._iso(ending_at)),
            ).fetchall()
        return [
            Fingerprint(
                id=r[0], timestamp_utc=r[1], bucket_start=r[2], model_id=r[3],
                tokens_in=r[4], tokens_out=r[5], tool_call_count=r[6],
                api_key_id=r[7], workspace_id=r[8], session_id=r[9],
                turn_count=r[10],
            )
            for r in rows
        ]

    def aggregate_buckets(self, fingerprints: list) -> list:
        # Retry-aware aggregation: SDK retries of the same logical call share
        # (session_id, turn_count). Collapse them to one entry, keeping the
        # final (largest-token) attempt — that's what billing records.
        deduped: dict = {}
        for fp in fingerprints:
            key = (fp.bucket_start, fp.session_id, fp.turn_count)
            existing = deduped.get(key)
            total = fp.tokens_in + fp.tokens_out
            if existing is None or total > (existing.tokens_in + existing.tokens_out):
                deduped[key] = fp

        groups: dict = {}
        for fp in deduped.values():
            key = (fp.bucket_start, fp.model_id, fp.api_key_id, fp.workspace_id)
            groups.setdefault(key, []).append(fp)

        aggregates = []
        for (bucket_start, model_id, api_key_id, workspace_id), items in groups.items():
            bucket_end = (
                datetime.fromisoformat(bucket_start) + timedelta(minutes=1)
            ).isoformat()
            aggregates.append(BucketAggregate(
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                model_id=model_id,
                api_key_id=api_key_id,
                workspace_id=workspace_id,
                call_count=len(items),
                tokens_in=sum(i.tokens_in for i in items),
                tokens_out=sum(i.tokens_out for i in items),
                tool_call_count=sum(i.tool_call_count for i in items),
            ))
        return aggregates

    def fetch_usage_window(
        self,
        starting_at: datetime,
        ending_at: datetime,
        bucket_width: str = "1m",
    ) -> list:
        response = self.usage_client.get_messages_usage_report(
            starting_at=self._iso(starting_at),
            ending_at=self._iso(ending_at),
            bucket_width=bucket_width,
            group_by=["model", "api_key_id", "workspace_id"],
            api_key_ids=[self.api_key_id],
            workspace_ids=[self.workspace_id],
        )
        return self._normalize_usage_records(response)

    @staticmethod
    def _normalize_usage_records(response: dict) -> list:
        records = []
        for bucket in response.get("data", []):
            for result in bucket.get("results", []):
                cache_create = result.get("cache_creation") or {}
                server_tool = result.get("server_tool_use") or {}
                records.append(UsageRecord(
                    starting_at=bucket["starting_at"],
                    ending_at=bucket["ending_at"],
                    model=result.get("model"),
                    api_key_id=result.get("api_key_id"),
                    workspace_id=result.get("workspace_id"),
                    uncached_input_tokens=result.get("uncached_input_tokens", 0),
                    cache_read_input_tokens=result.get("cache_read_input_tokens", 0),
                    cache_creation_5m_input_tokens=cache_create.get(
                        "ephemeral_5m_input_tokens", 0
                    ),
                    cache_creation_1h_input_tokens=cache_create.get(
                        "ephemeral_1h_input_tokens", 0
                    ),
                    output_tokens=result.get("output_tokens", 0),
                    server_tool_use_count=server_tool.get("web_search_requests", 0),
                ))
        return records

    def diff_window(
        self, starting_at: datetime, ending_at: datetime
    ) -> list:
        fps = self.get_fingerprints_in_window(starting_at, ending_at)
        ledger_buckets = self.aggregate_buckets(fps)
        ledger_by_key = {
            (b.bucket_start, b.model_id, b.api_key_id, b.workspace_id): b
            for b in ledger_buckets
        }
        usage_records = self.fetch_usage_window(starting_at, ending_at)
        usage_by_key = {
            (r.starting_at, r.model or "", r.api_key_id or "", r.workspace_id or ""): r
            for r in usage_records
        }

        all_keys = set(ledger_by_key) | set(usage_by_key)
        results = []
        for key in sorted(all_keys):
            bucket_start, model_id, api_key_id, workspace_id = key
            ledger_agg = ledger_by_key.get(key)
            usage_rec = usage_by_key.get(key)
            findings = self._diff_one(ledger_agg, usage_rec)
            severity = self._highest_severity(findings)
            bucket_end = (
                datetime.fromisoformat(bucket_start) + timedelta(minutes=1)
            ).isoformat()
            result = self._record_diff(
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                model_id=model_id,
                api_key_id=api_key_id,
                workspace_id=workspace_id,
                ledger_call_count=ledger_agg.call_count if ledger_agg else 0,
                usage_records_present=usage_rec is not None,
                findings=findings,
                severity=severity,
            )
            results.append(result)

        # Zero-tolerance cross-key model_substitution check. Fires when
        # ledger and billing claim DIFFERENT models for the same
        # (bucket, actor). Independent of the per-key missing_record /
        # ghost_call signals — those alert too, but their severity could
        # plausibly be softened in future for ingestion-lag handling.
        # This check enforces the zero-tolerance contract directly,
        # regardless of what happens to the decomposed signals.
        for sub in self._detect_cross_key_model_substitutions(
            ledger_by_key, usage_by_key,
        ):
            bucket_start, api_key_id, workspace_id, ledger_model, billing_model = sub
            bucket_end = (
                datetime.fromisoformat(bucket_start) + timedelta(minutes=1)
            ).isoformat()
            results.append(self._record_diff(
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                model_id=f"{ledger_model}|{billing_model}",
                api_key_id=api_key_id,
                workspace_id=workspace_id,
                ledger_call_count=0,
                usage_records_present=True,
                findings=[DriftFinding(
                    drift_class="model_substitution",
                    severity="alert",
                    details=(
                        f"Ledger claims model={ledger_model!r} but "
                        f"billing claims model={billing_model!r} for "
                        f"(bucket={bucket_start}, actor={api_key_id})"
                    ),
                )],
                severity="alert",
            ))

        return self._apply_actor_accumulation(results)

    @staticmethod
    def _detect_cross_key_model_substitutions(
        ledger_by_key: dict, usage_by_key: dict,
    ) -> list:
        ledger_models: dict = defaultdict(set)
        usage_models: dict = defaultdict(set)
        for (bucket, model, ak, ws) in ledger_by_key:
            ledger_models[(bucket, ak, ws)].add(model)
        for (bucket, model, ak, ws) in usage_by_key:
            usage_models[(bucket, ak, ws)].add(model)

        substitutions = []
        common = set(ledger_models) & set(usage_models)
        for actor_bucket in sorted(common):
            bucket, ak, ws = actor_bucket
            ledger_only = ledger_models[actor_bucket] - usage_models[actor_bucket]
            billing_only = usage_models[actor_bucket] - ledger_models[actor_bucket]
            for lm in sorted(ledger_only):
                for bm in sorted(billing_only):
                    substitutions.append((bucket, ak, ws, lm, bm))
        return substitutions

    def diff_recent_buckets(
        self,
        lookback_minutes: int = 15,
        lag_buffer_minutes: int = 6,
    ) -> list:
        # Anthropic's usage data has a ~5-minute ingestion lag. Back off so
        # we don't false-positive missing-record on buckets that just
        # haven't been indexed yet.
        now = self._clock()
        ending_at = now - timedelta(minutes=lag_buffer_minutes)
        starting_at = ending_at - timedelta(minutes=lookback_minutes)
        return self.diff_window(starting_at, ending_at)

    def _diff_one(
        self,
        ledger: Optional[BucketAggregate],
        usage: Optional[UsageRecord],
    ) -> list:
        findings = []

        if ledger and usage is None:
            findings.append(DriftFinding(
                drift_class="missing_record",
                severity="alert",
                details=(
                    f"Ledger has {ledger.call_count} call(s) but no API "
                    "usage record for this (bucket, actor, model)"
                ),
            ))
            return findings

        if ledger is None and usage is not None:
            findings.append(DriftFinding(
                drift_class="ghost_call",
                severity="alert",
                details=(
                    "API usage records present but ledger has 0 calls "
                    "for this (bucket, actor, model)"
                ),
            ))
            return findings

        if ledger is None or usage is None:
            return findings

        if usage.model and ledger.model_id != usage.model:
            findings.append(DriftFinding(
                drift_class="model_substitution",
                severity="alert",
                details=(
                    f"Ledger model={ledger.model_id!r} but "
                    f"billing model={usage.model!r}"
                ),
            ))

        ledger_total = ledger.tokens_in + ledger.tokens_out
        usage_total = usage.total_tokens
        if ledger_total > 0 or usage_total > 0:
            denom = max(ledger_total, usage_total, 1)
            drift_ratio = abs(ledger_total - usage_total) / denom
            if drift_ratio > self.config.token_mismatch_percent_adversarial:
                findings.append(DriftFinding(
                    drift_class="token_mismatch",
                    severity="alert",
                    details=(
                        f"Token drift {drift_ratio:.1%} exceeds adversarial "
                        f"threshold (ledger={ledger_total}, billing={usage_total})"
                    ),
                ))
            elif drift_ratio > self.config.token_mismatch_percent_infra:
                findings.append(DriftFinding(
                    drift_class="token_mismatch",
                    severity="warn",
                    details=(
                        f"Token drift {drift_ratio:.1%} above infra-noise "
                        f"tolerance (ledger={ledger_total}, billing={usage_total})"
                    ),
                ))
        return findings

    @staticmethod
    def _highest_severity(findings: list) -> str:
        if not findings:
            return "info"
        order = {"alert": 3, "warn": 2, "info": 1}
        return max(findings, key=lambda f: order.get(f.severity, 0)).severity

    def _record_diff(
        self,
        *,
        bucket_start: str,
        bucket_end: str,
        model_id: str,
        api_key_id: str,
        workspace_id: str,
        ledger_call_count: int,
        usage_records_present: bool,
        findings: list,
        severity: str,
    ) -> DiffResult:
        created_at = self._clock().isoformat()
        findings_json = json.dumps(
            [f.to_dict() for f in findings], sort_keys=True
        )
        with sqlite3.connect(self.ledger_db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO diff_reports (
                    bucket_start, bucket_end, model_id, api_key_id,
                    workspace_id, ledger_call_count, usage_records_present,
                    drift_findings_json, severity, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bucket_start, bucket_end, model_id, api_key_id,
                    workspace_id, ledger_call_count,
                    int(usage_records_present), findings_json, severity,
                    created_at,
                ),
            )
            row_id = cur.lastrowid
        result = DiffResult(
            id=row_id,
            bucket_start=bucket_start,
            bucket_end=bucket_end,
            model_id=model_id,
            api_key_id=api_key_id,
            workspace_id=workspace_id,
            ledger_call_count=ledger_call_count,
            usage_records_present=usage_records_present,
            drift_findings=tuple(findings),
            severity=severity,
            created_at=created_at,
        )
        if not self.shadow_mode and severity == "alert":
            raise DriftAlert(result)
        return result

    def get_actor_drift_history(
        self,
        api_key_id: str,
        window_minutes: Optional[int] = None,
    ) -> list:
        window = window_minutes or self.config.actor_window_minutes
        cutoff = (self._clock() - timedelta(minutes=window)).isoformat()
        with sqlite3.connect(self.ledger_db_path) as conn:
            rows = conn.execute(
                "SELECT id, bucket_start, bucket_end, model_id, api_key_id, "
                "workspace_id, ledger_call_count, usage_records_present, "
                "drift_findings_json, severity, created_at FROM diff_reports "
                "WHERE api_key_id = ? AND created_at >= ? "
                "ORDER BY created_at DESC",
                (api_key_id, cutoff),
            ).fetchall()
        return [self._row_to_diff(r) for r in rows]

    @staticmethod
    def _row_to_diff(r: tuple) -> DiffResult:
        findings = tuple(
            DriftFinding(
                drift_class=f["drift_class"],
                severity=f["severity"],
                details=f["details"],
            )
            for f in json.loads(r[8])
        )
        return DiffResult(
            id=r[0], bucket_start=r[1], bucket_end=r[2], model_id=r[3],
            api_key_id=r[4], workspace_id=r[5], ledger_call_count=r[6],
            usage_records_present=bool(r[7]), drift_findings=findings,
            severity=r[9], created_at=r[10],
        )

    def _apply_actor_accumulation(self, results: list) -> list:
        # Per-actor accumulation: a single warn is noise; warn + history of
        # recent drift on the same actor is a pattern. Elevate to alert.
        # Original warn rows stay in diff_reports as immutable audit; the
        # elevation writes a NEW row so the audit trail records the moment
        # accumulation tripped.
        elevated = []
        for result in results:
            if (
                result.severity != "warn"
                or not result.drift_findings
                or self.config.actor_drift_threshold <= 0
            ):
                elevated.append(result)
                continue
            history = self.get_actor_drift_history(result.api_key_id)
            drift_count = sum(1 for h in history if h.drift_findings)
            if drift_count >= self.config.actor_drift_threshold:
                elevated_findings = [
                    DriftFinding(
                        drift_class=f.drift_class,
                        severity="alert",
                        details=(
                            f"{f.details} [ACCUMULATED: actor has "
                            f"{drift_count} recent drift events]"
                        ),
                    )
                    for f in result.drift_findings
                ]
                elevated.append(self._record_diff(
                    bucket_start=result.bucket_start,
                    bucket_end=result.bucket_end,
                    model_id=result.model_id,
                    api_key_id=result.api_key_id,
                    workspace_id=result.workspace_id,
                    ledger_call_count=result.ledger_call_count,
                    usage_records_present=result.usage_records_present,
                    findings=elevated_findings,
                    severity="alert",
                ))
            else:
                elevated.append(result)
        return elevated

    @staticmethod
    def _iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
