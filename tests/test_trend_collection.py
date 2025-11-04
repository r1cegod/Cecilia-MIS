from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import pytest

import trend_collection as tc
import main as pipeline_main


class StubClient:
    def __init__(self, frames: Dict[str, tc.TrendFrame]) -> None:
        self.frames = {key.lower(): value for key, value in frames.items()}
        self.calls = []

    def interest_over_time(self, keyword: str, *, geo: str, timeframe: str) -> tc.TrendFrame:
        self.calls.append((keyword, geo, timeframe))
        try:
            return self.frames[keyword.lower()]
        except KeyError as exc:  # pragma: no cover - defensive
            raise tc.TrendCollectionError(f"Missing data for {keyword}") from exc


def test_collect_keyword_computes_metrics() -> None:
    frame = tc.TrendFrame(columns=["ai tutor"], data={"ai tutor": [10, 40, 60]})
    stub = StubClient({"ai tutor": frame})
    collector = tc.TrendCollector(geo="US", days=90, client=stub)

    rows = collector.collect_keywords(["ai tutor"])

    assert len(rows) == 1
    row = rows[0]
    assert row["keyword"] == "ai tutor"
    assert row["geo"] == "US"
    assert row["days"] == 3
    assert pytest.approx(row["avg_volume"], 0.001) == pytest.approx((10 + 40 + 60) / 3, 0.001)
    assert pytest.approx(row["last_value"], 0.001) == 60
    assert pytest.approx(row["growth_pct"], 0.001) == 500.0
    assert stub.calls[0][2] == "today 3-m"


def test_collect_keywords_deduplicates_case_insensitive() -> None:
    frame = tc.TrendFrame(columns=["ai tutor"], data={"ai tutor": [10, 20]})
    stub = StubClient({"ai tutor": frame})
    collector = tc.TrendCollector(geo="US", days=14, client=stub)

    rows = collector.collect_keywords(["AI Tutor", "ai tutor", " ai tutor "])

    assert len(rows) == 1
    assert stub.calls == [("AI Tutor", "US", "now 14-d")]


def test_merge_keywords_and_file_loading(tmp_path: Path) -> None:
    keyword_file = tmp_path / "keywords.txt"
    keyword_file.write_text("# comment\nai tutor\nstudent planner\n\nAI tutor\n")

    file_keywords = tc.load_keywords_from_file(str(keyword_file))
    merged = tc.merge_keywords(["creator economy"], file_keywords, ["Student Planner"])

    assert merged == ["creator economy", "ai tutor", "student planner"]


def test_timeframe_errors() -> None:
    with pytest.raises(ValueError):
        tc._derive_timeframe(0)


def test_collect_keywords_no_input() -> None:
    stub = StubClient({})
    collector = tc.TrendCollector(geo="US", days=7, client=stub)
    with pytest.raises(ValueError):
        collector.collect_keyword("")


def test_write_trends_preserves_base_column_order(tmp_path: Path) -> None:
    rows = [
        {
            "keyword": "ai tutor",
            "geo": "US",
            "days": 90,
            "avg_volume": 50,
            "growth_pct": 75,
            "last_value": 80,
            "extra_metric": 123,
        }
    ]

    output = pipeline_main._write_trends(rows, tmp_path)

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

    expected_prefix = ["keyword", "geo", "days", "avg_volume", "growth_pct", "last_value"]
    assert header[: len(expected_prefix)] == expected_prefix
    assert header[-1] == "extra_metric"
