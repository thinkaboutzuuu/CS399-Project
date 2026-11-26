"""Query contracts and annual percentage-rate evidence for the trend agents."""
import csv
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Query:
    metric: str
    start: int
    end: int
    kind: Literal["trend", "net change"]


@dataclass(frozen=True)
class Observation:
    year: int
    rate: Decimal


@dataclass(frozen=True)
class Evidence:
    scope: Literal["endpoints", "annual"]
    points: tuple[Observation, ...]
    change: Decimal | None  # percentage points, end minus start
    annual_pattern: str | None


class RateTable:
    def __init__(self, path: Path):
        self.series: dict[str, dict[int, Decimal]] = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            if rows.fieldnames != ["metric", "year", "rate"]:
                raise ValueError("CSV header must be metric,year,rate")
            for row in rows:
                try:
                    metric = (row["metric"] or "").strip().casefold()
                    year = int(row["year"])
                    rate = Decimal(row["rate"])
                    if (None in row or not metric or not 1000 <= year <= 9999
                            or not rate.is_finite() or not 0 <= rate <= 100):
                        raise ValueError
                except (TypeError, InvalidOperation, ValueError):
                    raise ValueError(f"Invalid metric, year, or percentage on line {rows.line_num}") from None
                series = self.series.setdefault(metric, {})
                if year in series:
                    raise ValueError(f"Duplicate observation for {metric}, {year}")
                series[year] = rate
        if not self.series:
            raise ValueError("The CSV needs at least one observation.")

    def lookup(self, query: Query, annual: bool) -> Evidence:
        years = range(query.start, query.end + 1) if annual else (query.start, query.end)
        rates = self.series[query.metric]
        points = tuple(Observation(y, rates[y]) for y in years if y in rates)
        change = None
        if query.start in rates and query.end in rates:
            change = rates[query.end] - rates[query.start]
        pattern = None
        if query.kind == "trend" and len(points) == query.end - query.start + 1:
            pattern = annual_pattern(points)
        return Evidence("annual" if annual else "endpoints", points, change, pattern)


def make_query(metric: str, start: int, end: int, kind: str, metrics: tuple[str, ...]) -> Query:
    if metric not in metrics or kind not in ("trend", "net change"):
        raise ValueError("Choose one available metric and either trend or net change.")
    if type(start) is not int or type(end) is not int or not 1000 <= start < end <= 9999:
        raise ValueError("Specify two different four-digit years in chronological order.")
    return Query(metric, start, end, kind)


def parse_question(question: str, metrics: tuple[str, ...]) -> Query:
    names = "|".join(re.escape(metric) for metric in metrics)
    request = " ".join(question.casefold().split()).rstrip(".?!")
    match = re.fullmatch(
        rf"(?:(?:show|give me|what is|what was) (?:the )?)?"
        rf"(?:(?P<metric>{names}) (?P<kind>trend|net change)|"
        rf"(?P<which>trend|net change) (?:in|for) (?:the )?(?P<name>{names})) "
        r"(?:from (?P<start>\d{4}) to (?P<end>\d{4})|"
        r"between (?P<first>\d{4}) and (?P<last>\d{4}))", request,
    )
    if not match:
        raise ValueError(
            f"Try 'Show the {metrics[0]} trend from 2021 to 2024' or replace "
            f"'trend' with 'net change'. Available metrics: {', '.join(metrics)}."
        )
    return make_query(match["metric"] or match["name"], int(match["start"] or match["first"]),
                      int(match["end"] or match["last"]), match["kind"] or match["which"], metrics)


@dataclass(frozen=True)
class Feedback:
    verdict: Literal["accept", "request_annual", "insufficient"]
    reason: str


def annual_pattern(points: tuple[Observation, ...]) -> str:
    pairs = tuple(zip(points, points[1:]))
    rising = any(b.rate > a.rate for a, b in pairs)
    falling = any(b.rate < a.rate for a, b in pairs)
    if rising and falling:
        return "changed direction"
    if not (rising or falling):
        return "was flat"
    direction = "rose" if rising else "fell"
    if any(b.rate == a.rate for a, b in pairs):
        return f"{direction} with a flat interval"
    return f"{direction} every year"


def evaluate(query: Query, evidence: Evidence, can_retry: bool) -> Feedback:
    points = evidence.points
    if not points or points[0].year != query.start or points[-1].year != query.end:
        return Feedback("insufficient", "The data is missing a requested endpoint.")
    if evidence.change != points[-1].rate - points[0].rate:
        return Feedback("insufficient", "The reported change disagrees with the cited rates.")
    if query.kind == "trend" and tuple(p.year for p in points) != tuple(range(query.start, query.end + 1)):
        if evidence.scope == "endpoints" and can_retry:
            return Feedback("request_annual", "Endpoints establish net change. Retrieve every year to check the path.")
        if evidence.scope == "annual":
            return Feedback("insufficient", "The annual series has missing years within the requested interval.")
        return Feedback("insufficient", "The query budget ended before the annual series could be checked.")
    if query.kind == "trend" and evidence.annual_pattern != annual_pattern(points):
        return Feedback("insufficient", "The reported annual pattern disagrees with the cited rates.")
    return Feedback("accept", "The cited observations cover the request and support the endpoint change.")


def render(query: Query, evidence: Evidence) -> str:
    points = evidence.points
    delta = evidence.change
    summary = (f"{query.metric.capitalize()}, {query.start}–{query.end}: "
               f"{points[0].rate}% to {points[-1].rate}% ({delta:+f} percentage points).")
    if evidence.annual_pattern:
        summary += f" The annual rate {evidence.annual_pattern}."
    return summary + "\nEvidence: " + ", ".join(f"{p.year} {p.rate}%" for p in points)
