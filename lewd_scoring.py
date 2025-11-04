"""NT-LEWD scoring utilities and CLI."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG = {
    "weights": {
        "large": 0.25,
        "early": 0.25,
        "who_pays": 0.25,
        "desperate": 0.25,
    },
    "thresholds": {
        "large": {"low": 10, "mid": 30, "high": 60},
        "early": {"low": 0, "mid": 40, "high": 80},
        "who_pays": {
            "map": {
                "student": 10,
                "job_seeker": 15,
                "creator": 18,
                "freelancer": 20,
                "SME": 23,
                "enterprise": 25,
            },
            "unknown_default": 8,
        },
        "desperate": {
            "bins": [
                {"max": 0, "score": 4},
                {"max": 3, "score": 10},
                {"max": 10, "score": 17},
                {"max": 30, "score": 22},
                {"max": 1_000_000_000, "score": 25},
            ],
            "heuristics_keywords": [
                "need",
                "urgent",
                "stuck",
                "broken",
                "fail",
            ],
        },
    },
}


class LewdScoringError(RuntimeError):
    """Raised for expected runtime errors in the scoring workflow."""


@dataclass
class ScoreResult:
    """Represents the scoring output for a single trend row."""

    row: Mapping[str, object]
    large: float
    early: float
    who_pays: float
    desperate: float
    total: int

    def to_record(self) -> Dict[str, object]:
        record = dict(self.row)
        record.update(
            {
                "large_0_25": round(self.large, 2),
                "early_0_25": round(self.early, 2),
                "who_pays_0_25": round(self.who_pays, 2),
                "desperate_0_25": round(self.desperate, 2),
                "lewd_total_0_100": int(self.total),
            }
        )
        return record


def load_config(path: Optional[str]) -> Dict[str, object]:
    """Load configuration YAML if available, otherwise return defaults."""

    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))

    config_path = Path(path)
    if not config_path.exists():
        raise LewdScoringError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()

    if yaml is not None:
        loaded = yaml.safe_load(raw_text) or {}
    else:
        try:
            loaded = json.loads(raw_text or "{}")
        except json.JSONDecodeError as exc:
            raise LewdScoringError("Config file is not valid JSON/YAML") from exc

    if not isinstance(loaded, Mapping):
        raise LewdScoringError("Config file must contain a mapping at the top level")

    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged_weights = dict(merged["weights"])
    merged_thresholds = dict(merged["thresholds"])

    merged_weights.update(loaded.get("weights", {}))

    thresholds = loaded.get("thresholds", {})
    for key, value in thresholds.items():
        if key in merged_thresholds and isinstance(value, Mapping):
            merged_thresholds[key] = {**merged_thresholds[key], **value}
        else:
            merged_thresholds[key] = value

    merged["weights"] = merged_weights
    merged["thresholds"] = merged_thresholds
    return merged


def load_trends(path: str | Path) -> List[Dict[str, object]]:
    """Load trend rows from a CSV file into a list of dictionaries."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise LewdScoringError(f"Trend CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"keyword", "geo", "days", "avg_volume", "growth_pct", "last_value"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise LewdScoringError(f"Trend CSV missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    return rows


def load_optional_csv(path: Optional[str]) -> Dict[str, object]:
    """Load optional CSVs (pain signals or buyer maps)."""

    if path is None:
        return {}

    csv_path = Path(path)
    if not csv_path.exists():
        raise LewdScoringError(f"Optional CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "keyword" not in reader.fieldnames:
            raise LewdScoringError(f"Optional CSV missing 'keyword' column: {csv_path}")
        other_columns = [c for c in reader.fieldnames if c != "keyword"]
        if len(other_columns) != 1:
            raise LewdScoringError("Optional CSV must have exactly one additional column besides 'keyword'")
        value_column = other_columns[0]
        mapping = {}
        for record in reader:
            raw_keyword = record.get("keyword", "")
            if raw_keyword is None:
                continue
            keyword = str(raw_keyword).strip()
            if not keyword:
                continue

            value = record.get(value_column)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                else:
                    value = ""

            key_lower = keyword.lower()
            if value in (None, ""):
                # Skip entries lacking usable enrichment values.
                continue
            mapping[key_lower] = value
    return mapping


def _piecewise_scale(value: float, thresholds: Mapping[str, float]) -> float:
    """Apply a piecewise scale returning a value between 0 and 25."""

    low = thresholds.get("low", 0.0)
    mid = thresholds.get("mid", low)
    high = thresholds.get("high", mid)

    if value <= low:
        return 0.0
    if value >= high:
        return 25.0
    if value <= mid:
        span = max(mid - low, 1e-9)
        return (value - low) / span * 12.5
    span = max(high - mid, 1e-9)
    return 12.5 + (value - mid) / span * 12.5


def _score_desperate(
    keyword: str,
    pain_mentions: Optional[float],
    desperate_cfg: Mapping[str, object],
) -> float:
    """Score desperation either from pain mentions or keyword heuristics."""

    bins = desperate_cfg.get("bins", [])
    heuristics = [str(h).lower() for h in desperate_cfg.get("heuristics_keywords", [])]

    if pain_mentions is not None and not math.isnan(float(pain_mentions)):
        for bucket in bins:
            if pain_mentions <= bucket.get("max", float("inf")):
                return float(bucket.get("score", 0))
        return float(bins[-1]["score"]) if bins else 0.0

    score = 7.0
    keyword_lower = keyword.lower()
    for token in heuristics:
        if token and token in keyword_lower:
            score += 3
    return min(score, 25.0)


def score_row(
    row: Mapping[str, object],
    cfg: Mapping[str, object],
    pain_map: Mapping[str, object],
    buyer_map: Mapping[str, object],
) -> ScoreResult:
    """Score a single trend row."""

    thresholds = cfg["thresholds"]
    weights = cfg["weights"]

    avg_volume = float(row.get("avg_volume", 0) or 0)
    growth_pct = float(row.get("growth_pct", 0) or 0)
    keyword = str(row.get("keyword", "")).strip()

    large_score = _piecewise_scale(avg_volume, thresholds.get("large", {}))
    early_score = _piecewise_scale(growth_pct, thresholds.get("early", {}))

    buyer_lookup = {str(k).strip().lower(): v for k, v in thresholds.get("who_pays", {}).get("map", {}).items()}
    buyer = buyer_map.get(keyword.lower()) if keyword else None
    buyer = buyer if buyer not in (None, "", float("nan")) else row.get("buyer")
    buyer_score = buyer_lookup.get(str(buyer).strip().lower(), thresholds.get("who_pays", {}).get("unknown_default", 8))
    buyer_score = float(buyer_score)

    pain_mentions = None
    if keyword:
        pain_raw = pain_map.get(keyword.lower())
        if pain_raw is not None and pain_raw != "":
            try:
                pain_mentions = float(pain_raw)
            except (TypeError, ValueError):
                pain_mentions = None

    desperate_score = _score_desperate(keyword, pain_mentions, thresholds.get("desperate", {}))

    total_weight = sum(float(w) for w in weights.values()) or 1.0
    weighted_sum = (
        large_score * float(weights.get("large", 0))
        + early_score * float(weights.get("early", 0))
        + buyer_score * float(weights.get("who_pays", 0))
        + desperate_score * float(weights.get("desperate", 0))
    )
    normalized = weighted_sum / total_weight
    total_score = int(round(normalized * 4))

    return ScoreResult(
        row=row,
        large=large_score,
        early=early_score,
        who_pays=buyer_score,
        desperate=desperate_score,
        total=total_score,
    )




def _format_table(headers: List[str], rows: List[List[object]]) -> str:
    if not rows:
        return ""
    formatted_rows: List[List[str]] = []
    for row in rows:
        formatted_rows.append([
            f"{value:.2f}" if isinstance(value, float) else str(value)
            for value in row
        ])
    widths = [len(str(header)) for header in headers]
    for row in formatted_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    separator = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    data_lines = [
        " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        for row in formatted_rows
    ]
    return "\n".join([header_line, separator, *data_lines])


def save_scored(rows: Iterable[Mapping[str, object]], out_path: str | Path) -> None:
    """Persist scored rows to disk as CSV."""

    row_list = [dict(row) for row in rows]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in row_list:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(out_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def _resolve_output_path(input_path: Path, out_dir: Path) -> Path:
    match = re.search(r"trends_(\d{8})", input_path.stem)
    if match:
        date_part = match.group(1)
    else:
        date_part = dt.datetime.utcnow().strftime("%Y%m%d")
    return out_dir / f"trends_scored_{date_part}.csv"


def score_file(
    input_path: str | Path,
    config_path: Optional[str] = None,
    pain_path: Optional[str] = None,
    buyers_path: Optional[str] = None,
    top: int = 10,
    out_dir: str | Path | None = None,
) -> Path:
    """Run the scoring workflow for a single CSV file."""

    cfg = load_config(config_path)
    rows = load_trends(input_path)
    pain_map = load_optional_csv(pain_path)
    buyer_map = load_optional_csv(buyers_path)

    scored = [
        score_row(row, cfg=cfg, pain_map=pain_map, buyer_map=buyer_map).to_record()
        for row in rows
    ]

    out_dir_path = Path(out_dir) if out_dir else Path("out")
    out_path = _resolve_output_path(Path(input_path), out_dir_path)
    save_scored(scored, out_path)

    top_n = max(1, int(top)) if scored else 0
    if top_n and scored:
        ranked = sorted(scored, key=lambda r: r["lewd_total_0_100"], reverse=True)[:top_n]
        preview_headers = [
            "keyword",
            "large_0_25",
            "early_0_25",
            "who_pays_0_25",
            "desperate_0_25",
            "lewd_total_0_100",
        ]
        table_rows = [[row.get(header, "") for header in preview_headers] for row in ranked]
        rendered = _format_table(preview_headers, table_rows)
        if rendered:
            print(rendered)

    print(f"Saved scored trends to {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NT-LEWD scoring CLI")
    parser.add_argument("--input", required=True, help="Path to trends CSV (out/trends_YYYYMMDD.csv)")
    parser.add_argument("--config", default=None, help="Optional config YAML path")
    parser.add_argument("--pain", default=None, help="Optional pain signals CSV path")
    parser.add_argument("--buyers", default=None, help="Optional who pays CSV path")
    parser.add_argument("--top", default=10, type=int, help="Preview top N rows")
    parser.add_argument(
        "--out-dir",
        default="out",
        help="Directory to place scored CSV (defaults to 'out')",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Path:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return score_file(
        input_path=args.input,
        config_path=args.config,
        pain_path=args.pain,
        buyers_path=args.buyers,
        top=args.top,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
