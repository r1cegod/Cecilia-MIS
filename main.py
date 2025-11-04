"""Entry point for Cecilia-MIS trend extraction pipeline."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import csv

import lewd_scoring
from trend_collection import (
    TrendCollector,
    TrendCollectionError,
    load_keywords_from_file,
    merge_keywords,
)


DEFAULT_OUTPUT_DIR = Path("out")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cecilia-MIS trend collector")
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV containing precomputed trend observations",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        help="Keywords to collect via Google Trends (mutually exclusive with --input)",
    )
    parser.add_argument(
        "--keywords-file",
        default=None,
        help="Path to a newline-delimited keyword list for Google Trends collection",
    )
    parser.add_argument(
        "--geo",
        default="US",
        help="Geo code for Google Trends collection (defaults to US)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Window size (in days) for Google Trends collection",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write trend CSV outputs",
    )
    parser.add_argument(
        "--autoscore",
        action="store_true",
        help="Automatically run NT-LEWD scoring after CSV export",
    )
    parser.add_argument(
        "--score-config",
        default=None,
        help="Optional scoring configuration YAML used when --autoscore is supplied",
    )
    parser.add_argument(
        "--score-pain",
        default=None,
        help="Optional pain signals CSV used when --autoscore is supplied",
    )
    parser.add_argument(
        "--score-buyers",
        default=None,
        help="Optional buyer map CSV used when --autoscore is supplied",
    )
    parser.add_argument(
        "--score-top",
        type=int,
        default=10,
        help="Number of rows to preview during auto scoring",
    )
    return parser


def _load_input(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input trends CSV not found: {path}")
    rows = lewd_scoring.load_trends(path)
    return rows


def _collect_keywords(args: argparse.Namespace) -> List[Dict[str, object]]:
    inline_keywords = args.keywords or []
    file_keywords: Iterable[str] = []
    if args.keywords_file:
        file_keywords = load_keywords_from_file(args.keywords_file)
    keywords = merge_keywords(inline_keywords, file_keywords)
    if not keywords:
        raise TrendCollectionError("No keywords supplied for Google Trends collection")

    collector = TrendCollector(geo=args.geo, days=args.days)
    return collector.collect_keywords(keywords)


def _write_trends(rows: List[Dict[str, object]], out_dir: Path) -> Path:
    if not rows:
        raise ValueError("No trend rows available to write")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d")
    output_path = out_dir / f"trends_{timestamp}.csv"
    base_order = ["keyword", "geo", "days", "avg_volume", "growth_pct", "last_value"]
    fieldnames: List[str] = [
        key for key in base_order if any(key in row for row in rows)
    ]
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def _autoscore(output_path: Path, args: argparse.Namespace) -> Path:
    return lewd_scoring.score_file(
        input_path=output_path,
        config_path=args.score_config,
        pain_path=args.score_pain,
        buyers_path=args.score_buyers,
        top=args.score_top,
        out_dir=args.output_dir,
    )


def main(argv: Optional[List[str]] = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)

    if args.input and (args.keywords or args.keywords_file):
        parser.error("--input cannot be combined with --keywords/--keywords-file")

    if args.input:
        input_path = Path(args.input)
        rows = _load_input(input_path)
    else:
        rows = _collect_keywords(args)

    output_path = _write_trends(rows, out_dir)

    if args.autoscore:
        _autoscore(output_path, args)

    return output_path


if __name__ == "__main__":
    main()
