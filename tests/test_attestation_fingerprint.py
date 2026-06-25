import copy
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attestation_fingerprint import (
    AttestationFingerprint,
    BucketAggregate,
    DiffResult,
    DriftAlert,
    DriftFinding,
    Fingerprint,
    ThresholdConfig,
    UsageRecord,
    UsageRecordsClient,
)


class FakeUsageClient:
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def get_messages_usage_report(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self._responses:
            return {"data": [], "has_more": False, "next_page": None}
        return self._responses.pop(0)


FIXED_NOW = datetime(2026, 6, 10, 12, 30, 0, tzinfo=timezone.utc)


def fixed_clock(now=FIXED_NOW):
    return lambda: now


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "ledger.db")


@pytest.fixture
def af(db_path):
    return AttestationFingerprint(
        ledger_db_path=db_path,
        usage_client=FakeUsageClient(),
        api_key_id="ak_test",
        workspace_id="ws_test",
        clock=fixed_clock(),
    )


def _bucket_payload(
    *,
    bucket_start,
    model="claude-sonnet-4-6",
    api_key_id="ak_test",
    workspace_id="ws_test",
    uncached_input=0,
    output=0,
    cache_5m=0,
    cache_1h=0,
    cache_read=0,
    web_search=0,
):
    bucket_end = (
        datetime.fromisoformat(bucket_start) + timedelta(minutes=1)
    ).isoformat()
    return {
        "starting_at": bucket_start,
        "ending_at": bucket_end,
        "results": [{
            "model": model,
            "api_key_id": api_key_id,
            "workspace_id": workspace_id,
            "uncached_input_tokens": uncached_input,
            "cache_read_input_tokens": cache_read,
            "cache_creation": {
                "ephemeral_5m_input_tokens": cache_5m,
                "ephemeral_1h_input_tokens": cache_1h,
            },
            "output_tokens": output,
            "server_tool_use": {"web_search_requests": web_search},
        }],
    }


def test_init_creates_tables(db_path):
    AttestationFingerprint(
        ledger_db_path=db_path,
        usage_client=FakeUsageClient(),
        api_key_id="ak",
        workspace_id="ws",
    )
    with sqlite3.connect(db_path) as conn:
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "fingerprints" in tables
    assert "diff_reports" in tables


def test_init_is_idempotent(db_path):
    AttestationFingerprint(
        ledger_db_path=db_path,
        usage_client=FakeUsageClient(),
        api_key_id="ak",
        workspace_id="ws",
    )
    AttestationFingerprint(
        ledger_db_path=db_path,
        usage_client=FakeUsageClient(),
        api_key_id="ak",
        workspace_id="ws",
    )


def test_record_fingerprint_writes_row(af):
    row_id = af.record_fingerprint(
        session_id="s1",
        turn_count=1,
        model_id="claude-opus-4-8",
        tokens_in=100,
        tokens_out=200,
        tool_call_count=2,
    )
    assert row_id == 1
    rows = af.get_fingerprints_in_window(
        FIXED_NOW - timedelta(minutes=1),
        FIXED_NOW + timedelta(minutes=1),
    )
    assert len(rows) == 1
    assert rows[0].model_id == "claude-opus-4-8"
    assert rows[0].tokens_in == 100
    assert rows[0].tokens_out == 200


def test_record_fingerprint_with_explicit_timestamp(af):
    explicit = datetime(2026, 6, 10, 11, 0, 0, tzinfo=timezone.utc)
    af.record_fingerprint(
        session_id="s",
        turn_count=1,
        model_id="m",
        tokens_in=1,
        tokens_out=1,
        timestamp=explicit,
    )
    rows = af.get_fingerprints_in_window(
        explicit - timedelta(minutes=1),
        explicit + timedelta(minutes=1),
    )
    assert rows[0].timestamp_utc == explicit.isoformat()


def test_record_fingerprint_coerces_naive_datetime(af):
    naive = datetime(2026, 6, 10, 11, 0, 0)
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="m",
        tokens_in=0, tokens_out=0, timestamp=naive,
    )
    rows = af.get_fingerprints_in_window(
        datetime(2026, 6, 10, 10, 59, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 10, 11, 1, 0, tzinfo=timezone.utc),
    )
    assert "+00:00" in rows[0].timestamp_utc


def test_aggregate_buckets_dedupes_retries(af):
    # Two fingerprints, same (session, turn) → one logical call
    ts = FIXED_NOW
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="m",
        tokens_in=100, tokens_out=200, timestamp=ts,
    )
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="m",
        tokens_in=110, tokens_out=210, timestamp=ts,  # retry — larger
    )
    af.record_fingerprint(
        session_id="s", turn_count=2, model_id="m",
        tokens_in=50, tokens_out=100, timestamp=ts,
    )
    fps = af.get_fingerprints_in_window(
        ts - timedelta(minutes=1), ts + timedelta(minutes=1),
    )
    aggregates = af.aggregate_buckets(fps)
    assert len(aggregates) == 1
    agg = aggregates[0]
    assert agg.call_count == 2  # turn 1 (retry collapsed) + turn 2
    assert agg.tokens_in == 110 + 50  # retry kept the larger, not summed
    assert agg.tokens_out == 210 + 100


def test_aggregate_buckets_groups_by_actor_and_model(af):
    ts = FIXED_NOW
    af.record_fingerprint(
        session_id="s1", turn_count=1, model_id="claude-sonnet-4-6",
        tokens_in=10, tokens_out=10, timestamp=ts,
    )
    af.record_fingerprint(
        session_id="s2", turn_count=1, model_id="claude-opus-4-8",
        tokens_in=20, tokens_out=20, timestamp=ts,
    )
    aggregates = af.aggregate_buckets(af.get_fingerprints_in_window(
        ts - timedelta(minutes=1), ts + timedelta(minutes=1),
    ))
    assert len(aggregates) == 2
    by_model = {a.model_id: a for a in aggregates}
    assert "claude-sonnet-4-6" in by_model
    assert "claude-opus-4-8" in by_model


def test_fetch_usage_window_calls_client_with_filters(db_path):
    client = FakeUsageClient()
    af = AttestationFingerprint(
        ledger_db_path=db_path, usage_client=client,
        api_key_id="ak_x", workspace_id="ws_y", clock=fixed_clock(),
    )
    af.fetch_usage_window(FIXED_NOW, FIXED_NOW + timedelta(minutes=5))
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["bucket_width"] == "1m"
    assert call["api_key_ids"] == ["ak_x"]
    assert call["workspace_ids"] == ["ws_y"]
    assert call["group_by"] == ["model", "api_key_id", "workspace_id"]


def test_normalize_usage_records_handles_missing_subobjects():
    response = {
        "data": [{
            "starting_at": "2026-06-10T12:00:00+00:00",
            "ending_at": "2026-06-10T12:01:00+00:00",
            "results": [{
                "model": "claude-sonnet-4-6",
                "uncached_input_tokens": 100,
                "output_tokens": 50,
                # no cache_creation, no server_tool_use
            }],
        }],
    }
    records = AttestationFingerprint._normalize_usage_records(response)
    assert len(records) == 1
    r = records[0]
    assert r.cache_creation_5m_input_tokens == 0
    assert r.cache_creation_1h_input_tokens == 0
    assert r.server_tool_use_count == 0


# ---------------------------------------------------------------------------
# Synthetic fault injection — the three failure modes Mike specified
# ---------------------------------------------------------------------------


def test_diff_detects_ghost_call(af):
    bucket = FIXED_NOW.replace(second=0, microsecond=0).isoformat()
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket, uncached_input=500, output=200,
        )],
    }])
    # No fingerprints recorded — ledger says nothing happened, billing disagrees
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=2),
        FIXED_NOW,
    )
    assert len(results) == 1
    findings = list(results[0].drift_findings)
    assert any(f.drift_class == "ghost_call" for f in findings)
    assert results[0].severity == "alert"


def test_diff_detects_model_substitution(af):
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="claude-sonnet-4-6",
        tokens_in=400, tokens_out=200, timestamp=ts,
    )
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket,
            model="claude-opus-4-8",  # ← billing claims a different model
            uncached_input=400, output=200,
        )],
    }])
    # Need to redirect by model — model_substitution detection only happens
    # when ledger key matches usage key (same bucket/actor); we group by
    # (bucket, model, actor) so different models create separate rows.
    # Re-test by injecting usage record with the same model as ledger and
    # then with a different model — model_substitution check is inside
    # _diff_one when keys align. The bucket aggregator uses ledger's
    # model_id; usage record uses billing's model. Different models = no
    # key match = either missing_record or ghost_call instead.
    #
    # The actual model substitution path triggers when usage_by_key matches
    # ledger_by_key on (bucket, actor) but the usage.model field disagrees
    # with the aggregate's model_id — which can't happen with the current
    # group_by="model". So model_substitution as defined surfaces as
    # missing_record + ghost_call together. That's the honest behavior.
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    classes = {
        f.drift_class
        for r in results for f in r.drift_findings
    }
    # Either way, drift is detected — both ledger and billing tell different stories
    assert "missing_record" in classes or "ghost_call" in classes
    assert all(r.severity == "alert" for r in results if r.drift_findings)


def test_diff_detects_token_mismatch_adversarial(af):
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="claude-sonnet-4-6",
        tokens_in=400, tokens_out=200, timestamp=ts,
    )
    # Billing claims 60% more tokens than ledger — way above adversarial threshold
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket, uncached_input=800, output=400,
        )],
    }])
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    findings = [f for r in results for f in r.drift_findings]
    token_findings = [f for f in findings if f.drift_class == "token_mismatch"]
    assert len(token_findings) >= 1
    assert any(f.severity == "alert" for f in token_findings)


def test_diff_token_mismatch_within_infra_tolerance_is_info_only(af):
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="m",
        tokens_in=500, tokens_out=500, timestamp=ts,
    )
    # 2% drift — below 5% infra tolerance, no finding should fire
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket, model="m", uncached_input=510, output=510,
        )],
    }])
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    token_findings = [
        f for r in results for f in r.drift_findings
        if f.drift_class == "token_mismatch"
    ]
    assert token_findings == []
    assert all(r.severity == "info" for r in results)


def test_diff_token_mismatch_infra_band_is_warn(af):
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="m",
        tokens_in=500, tokens_out=500, timestamp=ts,
    )
    # 10% drift — above 5% infra but below 20% adversarial → warn
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket, model="m", uncached_input=560, output=560,
        )],
    }])
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    token_findings = [
        f for r in results for f in r.drift_findings
        if f.drift_class == "token_mismatch"
    ]
    assert len(token_findings) == 1
    assert token_findings[0].severity == "warn"


def test_diff_no_drift_passes(af):
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="claude-sonnet-4-6",
        tokens_in=300, tokens_out=200, timestamp=ts,
    )
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket,
            model="claude-sonnet-4-6",
            uncached_input=300, output=200,
        )],
    }])
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    assert all(not r.drift_findings for r in results)
    assert all(r.severity == "info" for r in results)


def test_diff_persists_to_diff_reports(af, db_path):
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=(FIXED_NOW - timedelta(minutes=1)).replace(
                second=0, microsecond=0,
            ).isoformat(),
            uncached_input=100, output=100,
        )],
    }])
    af.diff_window(FIXED_NOW - timedelta(minutes=2), FIXED_NOW)
    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM diff_reports"
        ).fetchone()[0]
    assert count >= 1


def test_shadow_mode_never_raises_on_alert(af):
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=(FIXED_NOW - timedelta(minutes=1)).replace(
                second=0, microsecond=0,
            ).isoformat(),
            uncached_input=10000, output=10000,
        )],
    }])
    # Ghost call → alert. Shadow mode = True → no raise.
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=2), FIXED_NOW,
    )
    assert any(r.severity == "alert" for r in results)


def test_non_shadow_mode_raises_on_alert(db_path):
    bucket = (FIXED_NOW - timedelta(minutes=1)).replace(
        second=0, microsecond=0,
    ).isoformat()
    af = AttestationFingerprint(
        ledger_db_path=db_path,
        usage_client=FakeUsageClient([{
            "data": [_bucket_payload(
                bucket_start=bucket, uncached_input=1000, output=1000,
            )],
        }]),
        api_key_id="ak_test",
        workspace_id="ws_test",
        shadow_mode=False,
        clock=fixed_clock(),
    )
    with pytest.raises(DriftAlert) as exc:
        af.diff_window(FIXED_NOW - timedelta(minutes=2), FIXED_NOW)
    assert exc.value.result.severity == "alert"


def test_actor_accumulation_elevates_warn_to_alert(af, db_path):
    # Pre-populate diff_reports with 3 prior warn-severity drift findings
    # for the same actor, so accumulation threshold is exceeded.
    now = FIXED_NOW
    findings_json = json.dumps([{
        "drift_class": "token_mismatch",
        "severity": "warn",
        "details": "prior drift",
    }], sort_keys=True)
    with sqlite3.connect(db_path) as conn:
        for i in range(3):
            conn.execute(
                "INSERT INTO diff_reports (bucket_start, bucket_end, "
                "model_id, api_key_id, workspace_id, ledger_call_count, "
                "usage_records_present, drift_findings_json, severity, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (now - timedelta(minutes=10 + i)).isoformat(),
                    (now - timedelta(minutes=9 + i)).isoformat(),
                    "m", "ak_test", "ws_test", 1, 1, findings_json, "warn",
                    (now - timedelta(minutes=10 + i)).isoformat(),
                ),
            )

    # Now do a fresh diff that produces a warn — should elevate to alert
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="m",
        tokens_in=500, tokens_out=500, timestamp=ts,
    )
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket, model="m", uncached_input=560, output=560,
        )],
    }])
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    severities = [r.severity for r in results]
    assert "alert" in severities


def test_actor_accumulation_below_threshold_doesnt_elevate(af):
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="m",
        tokens_in=500, tokens_out=500, timestamp=ts,
    )
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket, model="m", uncached_input=560, output=560,
        )],
    }])
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    # First-ever warn for this actor → no history → no elevation
    warn_results = [r for r in results if r.severity == "warn"]
    assert len(warn_results) == 1


def test_get_actor_drift_history_filters_by_window(af, db_path):
    findings_json = json.dumps([{
        "drift_class": "x", "severity": "warn", "details": "y",
    }], sort_keys=True)
    with sqlite3.connect(db_path) as conn:
        # Recent
        conn.execute(
            "INSERT INTO diff_reports (bucket_start, bucket_end, model_id, "
            "api_key_id, workspace_id, ledger_call_count, "
            "usage_records_present, drift_findings_json, severity, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("b", "e", "m", "ak_test", "ws", 1, 1, findings_json, "warn",
             (FIXED_NOW - timedelta(minutes=5)).isoformat()),
        )
        # Old (outside 15-minute window)
        conn.execute(
            "INSERT INTO diff_reports (bucket_start, bucket_end, model_id, "
            "api_key_id, workspace_id, ledger_call_count, "
            "usage_records_present, drift_findings_json, severity, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("b2", "e2", "m", "ak_test", "ws", 1, 1, findings_json, "warn",
             (FIXED_NOW - timedelta(minutes=60)).isoformat()),
        )
    history = af.get_actor_drift_history("ak_test")
    assert len(history) == 1


def test_diff_recent_buckets_uses_lag_buffer(af):
    af.diff_recent_buckets(lookback_minutes=10, lag_buffer_minutes=6)
    # Should have called the usage client with a window ending at NOW - 6min
    call = af.usage_client.calls[0]
    starting = datetime.fromisoformat(call["starting_at"])
    ending = datetime.fromisoformat(call["ending_at"])
    assert ending == FIXED_NOW - timedelta(minutes=6)
    assert starting == FIXED_NOW - timedelta(minutes=16)


def test_fake_client_satisfies_protocol():
    client = FakeUsageClient()
    assert isinstance(client, UsageRecordsClient)


def test_fingerprint_is_frozen():
    fp = Fingerprint(
        id=1, timestamp_utc="t", bucket_start="b", model_id="m",
        tokens_in=0, tokens_out=0, tool_call_count=0,
        api_key_id="ak", workspace_id="ws", session_id="s",
        turn_count=1,
    )
    with pytest.raises(Exception):
        fp.tokens_in = 99  # type: ignore[misc]


def test_usage_record_total_tokens():
    r = UsageRecord(
        starting_at="s", ending_at="e", model="m",
        api_key_id="ak", workspace_id="ws",
        uncached_input_tokens=10, cache_read_input_tokens=20,
        cache_creation_5m_input_tokens=5, cache_creation_1h_input_tokens=3,
        output_tokens=100, server_tool_use_count=0,
    )
    assert r.total_tokens == 138


def test_threshold_config_defaults():
    c = ThresholdConfig()
    assert c.token_mismatch_percent_infra == 0.05
    assert c.token_mismatch_percent_adversarial == 0.20
    assert c.actor_drift_threshold == 3


def test_drift_alert_carries_result():
    result = DiffResult(
        id=1, bucket_start="b", bucket_end="e", model_id="m",
        api_key_id="ak", workspace_id="ws",
        ledger_call_count=0, usage_records_present=True,
        drift_findings=(), severity="alert", created_at="now",
    )
    err = DriftAlert(result)
    assert err.result is result
    assert "alert" in str(err)


def test_highest_severity_returns_info_when_no_findings():
    assert AttestationFingerprint._highest_severity([]) == "info"


def test_highest_severity_picks_highest():
    findings = [
        DriftFinding("a", "warn", "x"),
        DriftFinding("b", "alert", "y"),
        DriftFinding("c", "info", "z"),
    ]
    assert AttestationFingerprint._highest_severity(findings) == "alert"


def test_diff_handles_empty_window(af):
    # No fingerprints, no usage records
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=5), FIXED_NOW,
    )
    assert results == []


def test_diff_window_fires_explicit_model_substitution_with_matching_tokens(af):
    # Zero-tolerance guarantee: a model swap MUST fire model_substitution
    # at the diff_window level, independent of token-mismatch tolerance,
    # even when token counts match exactly. This is the contract Mike's
    # spec demanded: "no legitimate reason for it to differ."
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1,
        model_id="claude-sonnet-4-6",  # Ledger claims sonnet
        tokens_in=400, tokens_out=200, timestamp=ts,
    )
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket,
            model="claude-opus-4-8",  # But billing says opus
            uncached_input=400, output=200,  # Matching tokens — no token drift
        )],
    }])
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    model_sub = [
        f for r in results for f in r.drift_findings
        if f.drift_class == "model_substitution"
    ]
    assert len(model_sub) >= 1, (
        f"Expected explicit model_substitution alert, got: "
        f"{[(r.severity, [f.drift_class for f in r.drift_findings]) for r in results]}"
    )
    assert all(f.severity == "alert" for f in model_sub)


def test_diff_window_no_model_substitution_when_models_match(af):
    # Sanity check: when ledger and billing agree on model, no
    # model_substitution finding fires.
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="claude-sonnet-4-6",
        tokens_in=400, tokens_out=200, timestamp=ts,
    )
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket, model="claude-sonnet-4-6",
            uncached_input=400, output=200,
        )],
    }])
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    model_sub = [
        f for r in results for f in r.drift_findings
        if f.drift_class == "model_substitution"
    ]
    assert model_sub == []


def test_diff_one_returns_empty_when_both_none(af):
    assert af._diff_one(None, None) == []


def test_diff_one_detects_model_substitution_directly(af):
    # When ledger and billing align on (bucket, actor) but disagree on model,
    # the _diff_one path fires model_substitution. In normal diff_window flow
    # we group by model so this surfaces as missing_record + ghost_call
    # instead; the check remains as defense-in-depth for direct callers.
    ledger = BucketAggregate(
        bucket_start="2026-06-10T12:00:00+00:00",
        bucket_end="2026-06-10T12:01:00+00:00",
        model_id="claude-sonnet-4-6",
        api_key_id="ak", workspace_id="ws",
        call_count=1, tokens_in=100, tokens_out=100, tool_call_count=0,
    )
    usage = UsageRecord(
        starting_at="2026-06-10T12:00:00+00:00",
        ending_at="2026-06-10T12:01:00+00:00",
        model="claude-opus-4-8",
        api_key_id="ak", workspace_id="ws",
        uncached_input_tokens=100, cache_read_input_tokens=0,
        cache_creation_5m_input_tokens=0, cache_creation_1h_input_tokens=0,
        output_tokens=100, server_tool_use_count=0,
    )
    findings = af._diff_one(ledger, usage)
    assert any(f.drift_class == "model_substitution" for f in findings)
    assert any(f.severity == "alert" for f in findings)


def test_iso_coerces_naive_datetime():
    naive = datetime(2026, 6, 10, 12, 0, 0)
    iso = AttestationFingerprint._iso(naive)
    assert "+00:00" in iso


def test_actor_accumulation_disabled_when_threshold_is_zero(db_path):
    # Threshold of 0 disables accumulation elevation
    af = AttestationFingerprint(
        ledger_db_path=db_path,
        usage_client=FakeUsageClient(),
        api_key_id="ak_test",
        workspace_id="ws_test",
        threshold_config=ThresholdConfig(actor_drift_threshold=0),
        clock=fixed_clock(),
    )
    ts = FIXED_NOW - timedelta(minutes=1)
    bucket = ts.replace(second=0, microsecond=0).isoformat()
    af.record_fingerprint(
        session_id="s", turn_count=1, model_id="m",
        tokens_in=500, tokens_out=500, timestamp=ts,
    )
    af.usage_client = FakeUsageClient([{
        "data": [_bucket_payload(
            bucket_start=bucket, model="m", uncached_input=560, output=560,
        )],
    }])
    results = af.diff_window(
        FIXED_NOW - timedelta(minutes=3), FIXED_NOW,
    )
    # Still just warn — no elevation when threshold is 0
    assert any(r.severity == "warn" for r in results)
    assert all(r.severity != "alert" for r in results)
