from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import lewd_scoring
import main as pipeline_main


REQUIRED_COLUMNS = ["keyword", "geo", "days", "avg_volume", "growth_pct", "last_value"]


def _write_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_score_row_with_maps():
    cfg = lewd_scoring.load_config(None)
    row = {
        "keyword": "ai therapy",
        "geo": "US",
        "days": 90,
        "avg_volume": 70,
        "growth_pct": 90,
        "last_value": 100,
    }
    pain_map = {"ai therapy": 12}
    buyer_map = {"ai therapy": "enterprise"}

    result = lewd_scoring.score_row(row, cfg, pain_map, buyer_map)
    assert pytest.approx(result.large, 0.001) == 25.0
    assert pytest.approx(result.early, 0.001) == 25.0
    assert pytest.approx(result.who_pays, 0.001) == 25.0
    assert pytest.approx(result.desperate, 0.001) == 22.0
    assert result.total == 97


def test_score_row_heuristics_when_no_pain_data():
    cfg = lewd_scoring.load_config(None)
    row = {
        "keyword": "urgent passport renewal",
        "geo": "US",
        "days": 90,
        "avg_volume": 20,
        "growth_pct": 50,
        "last_value": 10,
    }
    result = lewd_scoring.score_row(row, cfg, pain_map={}, buyer_map={})
    assert pytest.approx(result.large, 0.001) == 6.25
    assert pytest.approx(result.early, 0.001) == 15.625
    assert pytest.approx(result.who_pays, 0.001) == 8.0
    assert result.desperate == 10.0
    assert result.total == 40


def test_desperate_scoring_uses_bins(tmp_path: Path):
    cfg = lewd_scoring.load_config(None)
    desperate_cfg = cfg["thresholds"]["desperate"]
    assert lewd_scoring._score_desperate("keyword", 0, desperate_cfg) == 4.0
    assert lewd_scoring._score_desperate("keyword", 5, desperate_cfg) == 17.0
    assert lewd_scoring._score_desperate("keyword", 100, desperate_cfg) == 25.0


def test_score_file_creates_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    rows = [
        {
            "keyword": "ai tutor",
            "geo": "US",
            "days": 90,
            "avg_volume": 65,
            "growth_pct": 82,
            "last_value": 70,
        },
        {
            "keyword": "broken laptop",
            "geo": "US",
            "days": 90,
            "avg_volume": 12,
            "growth_pct": 10,
            "last_value": 20,
        },
    ]
    input_path = tmp_path / "trends_20250101.csv"
    _write_rows(input_path, rows)

    output_path = lewd_scoring.score_file(input_path, top=2, out_dir=tmp_path)
    assert output_path.exists()

    captured = capsys.readouterr().out
    assert "lewd_total_0_100" in captured
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        scored_rows = list(csv.DictReader(handle))
    assert scored_rows
    assert len(scored_rows) == 2
    assert "lewd_total_0_100" in scored_rows[0]


def test_score_file_respects_custom_config(tmp_path: Path):
    custom_cfg = tmp_path / "custom.yml"
    custom_cfg.write_text(
        '{"weights": {"large": 0.1, "early": 0.6, "who_pays": 0.2, "desperate": 0.1}}'
    )

    rows = [
        {
            "keyword": "student loan relief",
            "geo": "US",
            "days": 90,
            "avg_volume": 25,
            "growth_pct": 85,
            "last_value": 30,
        }
    ]
    input_path = tmp_path / "trends_20250102.csv"
    _write_rows(input_path, rows)

    output_path = lewd_scoring.score_file(
        input_path,
        config_path=str(custom_cfg),
        top=1,
        out_dir=tmp_path,
    )
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        scored_rows = list(csv.DictReader(handle))
    total = int(scored_rows[0]["lewd_total_0_100"])
    assert total == 73


def test_load_optional_csv(tmp_path: Path):
    pain_csv = tmp_path / "pain.csv"
    pain_csv.write_text("keyword,pain_mentions\nAI tutor,12\n ,  \nBroken sink,\n")
    mapping = lewd_scoring.load_optional_csv(str(pain_csv))
    assert mapping["ai tutor"] == 12
    assert "" not in mapping
    assert "broken sink" not in mapping


def test_load_optional_csv_skips_blank_entries(tmp_path: Path):
    buyers_csv = tmp_path / "buyers.csv"
    buyers_csv.write_text(
        "keyword,buyer\nCreator economy, enterprise\n,student\nNeed passport, \n"
    )

    mapping = lewd_scoring.load_optional_csv(str(buyers_csv))
    assert mapping == {"creator economy": "enterprise"}


def test_main_autoscore_creates_scored_file(tmp_path: Path):
    rows = [
        {
            "keyword": "creator economy",
            "geo": "US",
            "days": 90,
            "avg_volume": 45,
            "growth_pct": 60,
            "last_value": 50,
        },
        {
            "keyword": "need faster shipping",
            "geo": "US",
            "days": 90,
            "avg_volume": 15,
            "growth_pct": 20,
            "last_value": 30,
        },
    ]
    input_path = tmp_path / "input.csv"
    _write_rows(input_path, rows)

    output_path = pipeline_main.main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path),
            "--autoscore",
            "--score-top",
            "1",
        ]
    )

    assert output_path.exists()
    scored_files = list(tmp_path.glob("trends_scored_*.csv"))
    assert len(scored_files) == 1
    with scored_files[0].open("r", encoding="utf-8", newline="") as handle:
        scored_rows = list(csv.DictReader(handle))
    assert scored_rows
    assert "lewd_total_0_100" in scored_rows[0]


def test_ranking_order_stable(tmp_path: Path):
    rows = [
        {
            "keyword": "enterprise ai",
            "geo": "US",
            "days": 90,
            "avg_volume": 80,
            "growth_pct": 85,
            "last_value": 90,
        },
        {
            "keyword": "student planner",
            "geo": "US",
            "days": 90,
            "avg_volume": 25,
            "growth_pct": 30,
            "last_value": 25,
        },
        {
            "keyword": "urgent help desk",
            "geo": "US",
            "days": 90,
            "avg_volume": 35,
            "growth_pct": 55,
            "last_value": 40,
        },
    ]
    input_path = tmp_path / "trends_20250103.csv"
    _write_rows(input_path, rows)

    output_path = lewd_scoring.score_file(input_path, top=3, out_dir=tmp_path)
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = list(csv.DictReader(handle))
    totals = [int(row["lewd_total_0_100"]) for row in sorted(reader, key=lambda r: int(r["lewd_total_0_100"]), reverse=True)]
    assert totals == sorted(totals, reverse=True)
    assert totals[0] >= totals[-1]
