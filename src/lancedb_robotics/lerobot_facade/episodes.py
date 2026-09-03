"""Episode/frame-index segmentation over a materialized alignment.

``aligned_ticks`` has no episode concept -- it is keyed by ``run_id`` +
``tick_index`` (see ``schemas/__init__.py``'s ``ALIGNED_TICKS_SCHEMA``), one
row per synchronized instant across the alignment's declared streams. This
module derives LeRobot-style ``episode_index``/``frame_index`` ordering by
bucketing an alignment's ticks against the ``episodes``/``scenarios`` time
ranges for the same run -- the same "physical episode preferred, else
scenario window" pattern ``dataset_export.py`` already uses for raw
observations (``_physical_episodes``/``_episodes``/``_task_description``),
adapted here for aligned ticks read live (no snapshot/version pinning).
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lancedb_robotics.lake import Lake


@dataclass(frozen=True)
class FacadeEpisode:
    """One LeRobot-shaped episode: an ordered sequence of aligned ticks."""

    episode_index: int
    episode_id: str
    task: str
    tick_indices: tuple[int, ...]


def build_episode_index(
    lake: Lake,
    job: dict[str, Any],
    *,
    episode_ids: Sequence[str] | None = None,
    allowed_tick_indices: Sequence[int] | None = None,
) -> tuple[FacadeEpisode, ...]:
    """Resolve ``job``'s aligned ticks into ordered, task-labeled episodes.

    ``allowed_tick_indices`` restricts the tick catalog to a caller-supplied
    set (e.g. the quality-filtered ``tick_plan.tick_indices`` of a wrapping
    :class:`~lancedb_robotics.training.AlignedFrameTrainingDataset`) -- pass
    ``None`` to consider every materialized tick for this alignment.
    """
    recipe = json.loads(job["recipe"] or "{}")
    run_id = recipe.get("run_id")
    if not run_id:
        raise ValueError(
            f"alignment {job['alignment_id']!r} has no run_id in its recipe; "
            "the LeRobot facade requires a single-run alignment"
        )

    allowed = (
        set(int(t) for t in allowed_tick_indices) if allowed_tick_indices is not None else None
    )
    tick_timestamps, tick_indices = _tick_catalog(lake, str(job["alignment_id"]), allowed=allowed)

    physical_rows = _physical_episode_rows(lake, run_id)
    if physical_rows:
        selected = physical_rows
        if episode_ids is not None:
            wanted = set(episode_ids)
            selected = [row for row in selected if row["episode_id"] in wanted]
        selected.sort(key=lambda row: (int(row.get("episode_index") or 0), row["episode_id"]))
        bounds = [
            (row["episode_id"], int(row["from_timestamp_ns"]), int(row["to_timestamp_ns"]))
            for row in selected
        ]
        task_by_id = {row["episode_id"]: row.get("task_id") for row in selected}
    else:
        selected = _scenario_rows(lake, run_id)
        if episode_ids is not None:
            wanted = set(episode_ids)
            selected = [row for row in selected if row["scenario_id"] in wanted]
        selected.sort(key=lambda row: (int(row["start_time_ns"]), row["scenario_id"]))
        bounds = [
            (row["scenario_id"], int(row["start_time_ns"]), int(row["end_time_ns"]))
            for row in selected
        ]
        task_by_id = {
            row["scenario_id"]: row.get("summary") or row.get("scenario_type") for row in selected
        }

    run_row = _run_row(lake, run_id)
    fallback_task = run_row.get("task_id") or "unknown task"

    episodes: list[FacadeEpisode] = []
    for index, (episode_id, start_ns, end_ns) in enumerate(bounds):
        lo = bisect.bisect_left(tick_timestamps, start_ns)
        hi = bisect.bisect_right(tick_timestamps, end_ns)
        episodes.append(
            FacadeEpisode(
                episode_index=index,
                episode_id=episode_id,
                task=task_by_id.get(episode_id) or fallback_task,
                tick_indices=tuple(tick_indices[lo:hi]),
            )
        )
    return tuple(episodes)


def _tick_catalog(
    lake: Lake,
    alignment_id: str,
    *,
    allowed: set[int] | None,
) -> tuple[list[int], list[int]]:
    """Return ``(timestamps, tick_indices)``, both ascending by tick_index."""
    table = lake.table("aligned_ticks")
    rows = (
        table.search()
        .select(["tick_index", "timestamp_ns"])
        .where(f"alignment_id = {_sql_literal(alignment_id)}")
        .to_arrow()
        .to_pylist()
    )
    if allowed is not None:
        rows = [row for row in rows if int(row["tick_index"]) in allowed]
    rows.sort(key=lambda row: int(row["tick_index"]))
    return [int(row["timestamp_ns"]) for row in rows], [int(row["tick_index"]) for row in rows]


def _physical_episode_rows(lake: Lake, run_id: str) -> list[dict[str, Any]]:
    return (
        lake.table("episodes")
        .search()
        .where(f"run_id = {_sql_literal(run_id)}")
        .to_arrow()
        .to_pylist()
    )


def _scenario_rows(lake: Lake, run_id: str) -> list[dict[str, Any]]:
    return (
        lake.table("scenarios")
        .search()
        .where(f"run_id = {_sql_literal(run_id)}")
        .to_arrow()
        .to_pylist()
    )


def _run_row(lake: Lake, run_id: str) -> dict[str, Any]:
    rows = (
        lake.table("runs").search().where(f"run_id = {_sql_literal(run_id)}").to_arrow().to_pylist()
    )
    return rows[0] if rows else {}


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"
