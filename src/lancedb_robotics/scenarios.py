"""Deterministic scenario windowing over ingested robot runs."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA


class ScenarioError(Exception):
    """Raised when scenario windows cannot be created."""


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ns|us|ms|s|m)\s*$")
_DURATION_UNITS = {
    "ns": Decimal("1"),
    "us": Decimal("1000"),
    "ms": Decimal("1000000"),
    "s": Decimal("1000000000"),
    "m": Decimal("60000000000"),
}


@dataclass(frozen=True)
class ScenarioWindowReport:
    """Summary of one deterministic windowing transform."""

    lake_uri: str
    transform_id: str
    window_ns: int
    topics: tuple[str, ...] = ()
    include_partial: bool = True
    runs_considered: int = 0
    rows_added: int = 0
    rows_replaced: int = 0
    windows_by_run: dict[str, int] = field(default_factory=dict)

    @property
    def topic_label(self) -> str:
        return ", ".join(self.topics) if self.topics else "all topics"


def parse_duration_ns(value: str) -> int:
    """Parse CLI durations such as ``5s``, ``100ms``, or ``250000ns``."""
    match = _DURATION_RE.match(value)
    if not match:
        raise ScenarioError("window must be a positive duration like 5s, 100ms, or 250000ns")
    amount_raw, unit = match.groups()
    try:
        amount = Decimal(amount_raw)
    except InvalidOperation as exc:
        raise ScenarioError(f"invalid window duration {value!r}") from exc
    ns = amount * _DURATION_UNITS[unit]
    if ns <= 0:
        raise ScenarioError("window duration must be positive")
    integral = ns.to_integral_value()
    if ns != integral:
        raise ScenarioError("window duration must resolve to a whole number of nanoseconds")
    return int(integral)


def _stable_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _normalize_topics(topics: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(sorted({topic for topic in topics if topic}))


def _window_bounds(
    start_time_ns: int, end_time_ns: int, window_ns: int, *, include_partial: bool
) -> list[tuple[int, int, bool]]:
    if end_time_ns < start_time_ns:
        raise ScenarioError("run end_time_ns is earlier than start_time_ns")
    if end_time_ns == start_time_ns:
        return [(start_time_ns, end_time_ns, True)] if include_partial else []

    bounds: list[tuple[int, int, bool]] = []
    start = start_time_ns
    while start < end_time_ns:
        planned_end = start + window_ns
        if planned_end <= end_time_ns:
            bounds.append((start, planned_end, False))
            start = planned_end
            continue
        if include_partial:
            bounds.append((start, end_time_ns, True))
        break
    return bounds


def _observations_in_window(
    observations: list[dict], start_time_ns: int, end_time_ns: int, *, is_final: bool
) -> list[dict]:
    rows = []
    for row in observations:
        timestamp = row["timestamp_ns"]
        if timestamp < start_time_ns:
            continue
        if timestamp < end_time_ns or (is_final and timestamp <= end_time_ns):
            rows.append(row)
    return rows


def _scenario_id(
    *,
    run_id: str,
    start_time_ns: int,
    end_time_ns: int,
    window_ns: int,
    topics: tuple[str, ...],
) -> str:
    digest = _stable_digest(
        {
            "run_id": run_id,
            "start_time_ns": start_time_ns,
            "end_time_ns": end_time_ns,
            "window_ns": window_ns,
            "topics": topics,
        }
    )
    return f"scn-{digest}"


def create_scenario_windows(
    lake: Lake,
    *,
    window_ns: int,
    topics: tuple[str, ...] | list[str] = (),
    include_partial: bool = True,
    created_by: str = "lancedb-robotics",
) -> ScenarioWindowReport:
    """Write fixed-duration scenario rows for ingested runs.

    Boundaries are anchored to each run's recorded start/end time. A window is
    materialized only when at least one matching observation falls inside it.
    Window membership is half-open, except the final materialized window
    includes observations exactly at the run end timestamp.
    """
    if window_ns <= 0:
        raise ScenarioError("window duration must be positive")

    topic_filters = _normalize_topics(topics)
    transform_payload = {
        "kind": "scenario-windowing",
        "window_ns": window_ns,
        "topics": topic_filters,
        "include_partial": include_partial,
    }
    transform_id = f"tfm-scenarios-{_stable_digest(transform_payload)}"

    runs = sorted(lake.table("runs").to_arrow().to_pylist(), key=lambda row: row["run_id"])
    observations_by_run: dict[str, list[dict]] = {}
    for row in lake.table("observations").to_arrow().to_pylist():
        if topic_filters and row["topic"] not in topic_filters:
            continue
        observations_by_run.setdefault(row["run_id"], []).append(row)
    for rows in observations_by_run.values():
        rows.sort(key=lambda row: (row["timestamp_ns"], row["raw_sequence"], row["topic"]))

    scenario_rows = []
    windows_by_run: dict[str, int] = {}
    for run in runs:
        run_id = run["run_id"]
        run_observations = observations_by_run.get(run_id, [])
        if not run_observations:
            continue

        bounds = _window_bounds(
            run["start_time_ns"],
            run["end_time_ns"],
            window_ns,
            include_partial=include_partial,
        )
        for start_time_ns, end_time_ns, is_partial in bounds:
            is_final = end_time_ns == run["end_time_ns"]
            window_observations = _observations_in_window(
                run_observations, start_time_ns, end_time_ns, is_final=is_final
            )
            if not window_observations:
                continue

            row_topics = tuple(sorted({row["topic"] for row in window_observations}))
            row = {
                "scenario_id": _scenario_id(
                    run_id=run_id,
                    start_time_ns=start_time_ns,
                    end_time_ns=end_time_ns,
                    window_ns=window_ns,
                    topics=topic_filters or row_topics,
                ),
                "run_id": run_id,
                "start_time_ns": start_time_ns,
                "end_time_ns": end_time_ns,
                "window_ns": window_ns,
                "is_partial": is_partial,
                "topics": list(row_topics),
                "observation_ids": [row["observation_id"] for row in window_observations],
                "observation_count": len(window_observations),
                "scenario_type": "fixed-window",
                "source": "scenario-windowing",
                "coverage_tags": [
                    "window:fixed",
                    f"window_ns:{window_ns}",
                    f"partial:{str(is_partial).lower()}",
                ],
                "transform_id": transform_id,
                "created_at": datetime.now(UTC),
            }
            scenario_rows.append(row)
            windows_by_run[run_id] = windows_by_run.get(run_id, 0) + 1

    scenarios_table = lake.table("scenarios")
    existing_ids = {
        row["scenario_id"]
        for row in scenarios_table.to_arrow().to_pylist()
        if row["transform_id"] == transform_id
    }
    rows_to_add = [row for row in scenario_rows if row["scenario_id"] not in existing_ids]
    rows_replaced = 0
    if rows_to_add:
        scenarios_table.add(pa.Table.from_pylist(rows_to_add, schema=scenarios_table.schema))

    now = datetime.now(UTC)
    transform_row = {
        "transform_id": transform_id,
        "kind": "scenario-windowing",
        "input_uris": [],
        "input_table_versions": [],
        "output_tables": ["scenarios"],
        "params": json.dumps(
            {
                **transform_payload,
                "scenario_ids": sorted(row["scenario_id"] for row in scenario_rows),
                "rows_added": len(rows_to_add),
                "rows_reused": len(scenario_rows) - len(rows_to_add),
            },
            sort_keys=True,
        ),
        "status": "completed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms_table = lake.table("transform_runs")
    transforms_table.delete(f"transform_id = '{transform_id}'")
    transforms_table.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): scenario windowing records its execution
    # + scenario row-set without requiring a later refresh_graph().
    emit_transform_lineage(lake, transform_row)

    return ScenarioWindowReport(
        lake_uri=lake.uri,
        transform_id=transform_id,
        window_ns=window_ns,
        topics=topic_filters,
        include_partial=include_partial,
        runs_considered=len(runs),
        rows_added=len(rows_to_add),
        rows_replaced=rows_replaced,
        windows_by_run=dict(sorted(windows_by_run.items())),
    )
