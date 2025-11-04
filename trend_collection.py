"""Google Trends collection utilities for Cecilia-MIS."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

try:  # pragma: no cover - exercised in production but mocked in tests
    from pytrends.request import TrendReq
except ModuleNotFoundError:  # pragma: no cover - dependency validated in tests
    TrendReq = None  # type: ignore[misc]

try:  # pragma: no cover - optional dependency used in production
    import pandas as _pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - tests rely on lightweight stubs
    _pd = None


class TrendCollectionError(RuntimeError):
    """Raised when collecting Google Trends data fails."""


def _derive_timeframe(days: int) -> str:
    """Convert the requested day window into a pytrends timeframe string."""

    if days <= 0:
        raise ValueError("days must be positive")
    if days <= 7:
        return f"now {days}-d"
    if days <= 30:
        # 30d window provides daily granularity without clipping
        return f"now {days}-d"
    if days <= 90:
        return "today 3-m"
    if days <= 180:
        return "today 6-m"
    if days <= 365:
        return "today 12-m"
    years = math.ceil(days / 365)
    return f"today {years}-y"


@dataclass(slots=True)
class TrendFrame:
    """Minimal representation of a trends response."""

    columns: List[str]
    data: Dict[str, List[float]]

    def first_numeric_column(self) -> tuple[str, List[float]]:
        for column in self.columns:
            if column.lower() == "ispartial":
                continue
            if column in self.data:
                return column, self.data[column]
        raise TrendCollectionError("No numeric column found in trends response")


@dataclass(slots=True)
class TrendRecord:
    """A normalized snapshot of a keyword's search interest."""

    keyword: str
    geo: str
    days: int
    avg_volume: float
    growth_pct: float
    last_value: float

    def to_row(self) -> Mapping[str, object]:
        return {
            "keyword": self.keyword,
            "geo": self.geo,
            "days": self.days,
            "avg_volume": round(self.avg_volume, 4),
            "growth_pct": round(self.growth_pct, 4),
            "last_value": round(self.last_value, 4),
        }


class GoogleTrendsClient:
    """Thin wrapper around ``pytrends`` with retry and dependency injection."""

    def __init__(
        self,
        *,
        hl: str = "en-US",
        tz: int = 360,
        retries: int = 3,
        backoff: float = 2.0,
        request_args: Optional[Mapping[str, object]] = None,
        trend_req: Optional[TrendReq] = None,
    ) -> None:
        if retries <= 0:
            raise ValueError("retries must be positive")
        if backoff <= 0:
            raise ValueError("backoff must be positive")

        self.retries = retries
        self.backoff = backoff

        if trend_req is not None:
            self._trend_req = trend_req
        else:
            if TrendReq is None:  # pragma: no cover - dependency guard
                raise TrendCollectionError("pytrends is required to collect Google Trends data")
            self._trend_req = TrendReq(hl=hl, tz=tz, requests_args=request_args)

    def interest_over_time(self, keyword: str, *, geo: str, timeframe: str) -> TrendFrame:
        """Fetch interest-over-time data for a keyword with retries."""

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                self._trend_req.build_payload([keyword], geo=geo, timeframe=timeframe)
                df = self._trend_req.interest_over_time()
                frame = _to_trend_frame(df)
                if not frame.data:
                    raise TrendCollectionError(f"No data returned for keyword '{keyword}'")
                return frame
            except Exception as exc:  # pragma: no cover - network layer
                last_error = exc
                if attempt == self.retries:
                    break
                sleep_for = self.backoff ** (attempt - 1)
                time.sleep(sleep_for)
        raise TrendCollectionError(f"Failed to retrieve data for '{keyword}'") from last_error


def _to_trend_frame(raw: object) -> TrendFrame:
    if isinstance(raw, TrendFrame):
        return raw

    if raw is None:
        raise TrendCollectionError("No data returned from Google Trends")

    if _pd is not None and isinstance(raw, _pd.DataFrame):  # type: ignore[attr-defined]
        columns = list(raw.columns)
        data = raw.to_dict(orient="list")  # type: ignore[call-arg]
        return TrendFrame(columns=columns, data={col: list(data[col]) for col in columns})

    if isinstance(raw, Mapping):
        columns = list(raw.keys())
        data = {col: list(raw[col]) for col in columns}
        return TrendFrame(columns=columns, data=data)

    if hasattr(raw, "columns") and hasattr(raw, "__getitem__"):
        columns = list(raw.columns)  # type: ignore[attr-defined]
        data = {col: list(raw[col]) for col in columns}  # type: ignore[index]
        return TrendFrame(columns=columns, data=data)

    raise TrendCollectionError("Unsupported trends response type")


def _filter_numeric(values: Sequence[object]) -> List[float]:
    numeric: List[float] = []
    for value in values:
        try:
            if value is None:
                continue
            as_float = float(value)
            if math.isnan(as_float):
                continue
            numeric.append(as_float)
        except (TypeError, ValueError):
            continue
    return numeric


def _summarize_keyword(
    keyword: str,
    geo: str,
    days: int,
    frame: TrendFrame,
) -> TrendRecord:
    column, values = frame.first_numeric_column()
    numeric = _filter_numeric(values)
    if not numeric:
        raise TrendCollectionError(f"No numeric data available for '{keyword}'")

    avg_volume = float(sum(numeric) / len(numeric))
    last_value = float(numeric[-1])
    first_value = float(numeric[0])

    if math.isclose(first_value, 0.0, abs_tol=1e-9):
        growth_pct = 100.0 if last_value > 0 else 0.0
    else:
        growth_pct = ((last_value - first_value) / first_value) * 100.0

    return TrendRecord(
        keyword=keyword,
        geo=geo,
        days=len(numeric),
        avg_volume=avg_volume,
        growth_pct=growth_pct,
        last_value=last_value,
    )


class TrendCollector:
    """Collect and normalize Google Trends metrics for a keyword batch."""

    def __init__(
        self,
        *,
        geo: str = "US",
        days: int = 90,
        client: Optional[GoogleTrendsClient] = None,
    ) -> None:
        if not geo:
            raise ValueError("geo must be provided")
        if days <= 0:
            raise ValueError("days must be positive")

        self.geo = geo
        self.days = days
        self.client = client or GoogleTrendsClient()

    def collect_keyword(self, keyword: str) -> TrendRecord:
        """Collect a single keyword, returning a :class:`TrendRecord`."""

        keyword_clean = keyword.strip()
        if not keyword_clean:
            raise ValueError("keyword cannot be blank")

        timeframe = _derive_timeframe(self.days)
        df = self.client.interest_over_time(keyword_clean, geo=self.geo, timeframe=timeframe)
        return _summarize_keyword(keyword_clean, self.geo, self.days, df)

    def collect_keywords(self, keywords: Sequence[str]) -> List[Mapping[str, object]]:
        """Collect a batch of keywords returning CSV-ready dictionaries."""

        records: List[Mapping[str, object]] = []
        seen: set[str] = set()
        for keyword in keywords:
            normalized = keyword.strip()
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            record = self.collect_keyword(normalized)
            records.append(record.to_row())
        return records


def load_keywords_from_file(path: str) -> List[str]:
    """Load keywords (one per line) ignoring comments and blanks."""

    keywords: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            keywords.append(token)
    return keywords


def merge_keywords(*groups: Iterable[str]) -> List[str]:
    """Merge keyword iterables preserving order and uniqueness (case insensitive)."""

    merged: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for keyword in group:
            normalized = keyword.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged

