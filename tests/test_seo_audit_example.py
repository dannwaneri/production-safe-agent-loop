import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from examples import seo_audit_example as ex
from ledger import Ledger


SAMPLE_HTML = """
<html>
  <head>
    <title>  Example Domain  </title>
    <meta name="description" content="A demo page">
  </head>
  <body>
    <h1>Example</h1>
    <h1>Another</h1>
    <a href="/a">A</a>
    <a href="/b">B</a>
    <a href="/c">C</a>
  </body>
</html>
"""


def _mock_response(text=SAMPLE_HTML, status_code=200):
    m = MagicMock()
    m.text = text
    m.status_code = status_code
    return m


def test_crawl_extracts_seo_signals():
    with patch.object(ex.requests, "get", return_value=_mock_response()) as mocked:
        result = ex.crawl("https://example.com")
    assert result.url == "https://example.com"
    assert result.status_code == 200
    assert result.title == "Example Domain"
    assert result.meta_description == "A demo page"
    assert result.h1_count == 2
    assert result.link_count == 3
    assert mocked.called
    kwargs = mocked.call_args.kwargs
    assert kwargs["timeout"] == 10
    assert "User-Agent" in kwargs["headers"]


def test_crawl_handles_missing_title_and_meta():
    html = "<html><body><p>no head tags</p></body></html>"
    with patch.object(ex.requests, "get", return_value=_mock_response(text=html)):
        result = ex.crawl("https://x.test")
    assert result.title == ""
    assert result.meta_description == ""
    assert result.h1_count == 0
    assert result.link_count == 0


def test_crawl_handles_meta_without_content_attr():
    html = '<html><head><meta name="description"></head></html>'
    with patch.object(ex.requests, "get", return_value=_mock_response(text=html)):
        result = ex.crawl("https://x.test")
    assert result.meta_description == ""


def test_format_task_contains_signals():
    c = ex.CrawlResult(
        url="https://x.test",
        status_code=200,
        title="Hi",
        meta_description="desc",
        h1_count=2,
        link_count=7,
    )
    out = ex.format_task(c)
    assert "https://x.test" in out
    assert "status: 200" in out
    assert "'Hi'" in out
    assert "'desc'" in out
    assert "H1 count: 2" in out
    assert "Link count: 7" in out


def test_print_ledger_outputs_rows(capsys, tmp_path):
    ledger = Ledger(db_path=str(tmp_path / "led.db"))
    ledger.write(
        session_id="s1",
        turn_count=1,
        state_origin="llm",
        input_str="task",
        token_delta=42,
        execution_time_ms=100,
        pass_fail=True,
    )
    ledger.write(
        session_id="s1",
        turn_count=2,
        state_origin="circuit_breaker",
        input_str="task",
        token_delta=0,
        execution_time_ms=0,
        pass_fail=False,
        breach_reason="turn_ceiling",
    )
    ex.print_ledger(ledger, "s1")
    out = capsys.readouterr().out
    assert "2 row(s)" in out
    assert "turn=1" in out
    assert "tokens=42" in out
    assert "PASS" in out
    assert "FAIL" in out
    assert "breach=turn_ceiling" in out


def test_main_exits_when_no_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from spec_writer import SpecResult
    monkeypatch.setattr(
        ex.SpecWriter,
        "run",
        lambda self: SpecResult(
            what_it_does="a", what_it_does_not="b",
            done_looks_like="c", session_id="sid",
        ),
    )
    monkeypatch.setattr(
        ex, "crawl",
        lambda url, timeout=10: ex.CrawlResult(
            url=url, status_code=200, title="t",
            meta_description="m", h1_count=1, link_count=1,
        ),
    )
    rc = ex.main("https://example.com")
    assert rc == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err


def test_main_exits_when_crawl_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    from spec_writer import SpecResult
    monkeypatch.setattr(
        ex.SpecWriter,
        "run",
        lambda self: SpecResult(
            what_it_does="a", what_it_does_not="b", done_looks_like="c", session_id="sid"
        ),
    )

    def boom(url, timeout=10):
        raise ex.requests.RequestException("network down")

    monkeypatch.setattr(ex, "crawl", boom)
    rc = ex.main("https://example.com")
    assert rc == 1
    err = capsys.readouterr().err
    assert "Crawl failed" in err
