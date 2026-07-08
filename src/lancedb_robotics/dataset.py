"""Dataset snapshots: freeze curated scenario results into a reproducible slice.

A snapshot (backlog 0009 / Feature Set 7) turns a selection of ``scenarios``
rows — chosen explicitly or via a recorded search — into one
``dataset_snapshots`` row that pins:

- the selected ``scenario_id``s and the filters/search spec that chose them
  (``query_spec``),
- a deterministic train/val/test split, assigned by run or scenario id so the
  same inputs always yield the same partition (``split``),
- the source table versions, so the slice can be re-read reproducibly without
  repacking the corpus (``table_versions``),
- transform lineage (a ``kind="dataset-snapshot"`` ``transform_runs`` row).

Everything is a stable function of the inputs: the ``dataset_id`` is a digest of
name + selection + split config + table versions, so re-creating an identical
snapshot replaces it in place rather than duplicating.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.lineage_hooks import (
    attach_lineage_context_to_params,
    begin_lineage_execution,
)
from lancedb_robotics.schemas import DATASET_SNAPSHOTS_SCHEMA, TRANSFORM_RUNS_SCHEMA

SPLIT_BY_RUN = "run"
SPLIT_BY_SCENARIO = "scenario"
SPLIT_BYS = (SPLIT_BY_RUN, SPLIT_BY_SCENARIO)

DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}

_SOURCE_TABLES = (
    "scenarios",
    "episodes",
    "videos",
    "video_encodings",
    "runs",
    "observations",
    "curation_views",
    "curation_view_membership_chunks",
    "curation_memberships",
    "curation_materializations",
)


class DatasetError(Exception):
    """Raised when a dataset snapshot cannot be created."""


@dataclass(frozen=True)
class SnapshotManifest:
    """Reproducible description of one dataset snapshot."""

    lake_uri: str
    dataset_id: str
    name: str
    tag: str
    transform_id: str
    scenario_ids: tuple[str, ...]
    split_by: str
    split_ratios: dict[str, float]
    split_counts: dict[str, int]
    split_assignments: dict[str, str]
    table_versions: tuple[tuple[str, int], ...]
    source: dict
    balance_report: dict | None = None
    coverage_report: dict | None = None


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _assign_split(key: str, ratios: dict[str, float]) -> str:
    """Deterministically map a split key into a split bucket by hash."""
    bucket = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10_000 / 10_000
    cumulative = 0.0
    for split, ratio in ratios.items():
        cumulative += ratio
        if bucket < cumulative:
            return split
    return next(reversed(ratios))  # rounding guard: last split absorbs the tail


def create_snapshot(
    lake: Lake,
    *,
    name: str,
    scenario_ids: list[str],
    source: dict | None = None,
    split_by: str = SPLIT_BY_RUN,
    split_ratios: dict[str, float] | None = None,
    tag: str | None = None,
    balance_report: dict | None = None,
    coverage_report: dict | None = None,
    created_by: str = "lancedb-robotics",
    lineage_context: object | None = None,
) -> SnapshotManifest:
    """Freeze ``scenario_ids`` into a reproducible ``dataset_snapshots`` row."""
    if split_by not in SPLIT_BYS:
        raise DatasetError(f"unknown split_by {split_by!r}; expected one of {', '.join(SPLIT_BYS)}")
    ratios = split_ratios or DEFAULT_SPLIT_RATIOS

    scenarios = {row["scenario_id"]: row for row in lake.table("scenarios").to_arrow().to_pylist()}
    requested = list(dict.fromkeys(scenario_ids))  # dedupe, preserve order
    if not requested:
        raise DatasetError("no scenarios selected for the snapshot")
    unknown = [sid for sid in requested if sid not in scenarios]
    if unknown:
        raise DatasetError(f"unknown scenario ids: {unknown}")
    selected = sorted(requested)

    assignments = {
        sid: _assign_split(scenarios[sid]["run_id"] if split_by == SPLIT_BY_RUN else sid, ratios)
        for sid in selected
    }
    counts = {split: 0 for split in ratios}
    for split in assignments.values():
        counts[split] += 1

    table_versions = [
        {"table": table, "version": int(lake.table(table).version), "tag": ""}
        for table in _SOURCE_TABLES
    ]
    source = source or {"kind": "explicit"}
    tag = tag or name
    handle = begin_lineage_execution(
        lineage_context,
        operation="dataset-snapshot",
        params={"name": name, "tag": tag, "scenario_count": len(selected)},
    )
    context = handle.finish(status="completed")

    dataset_id = "ds-" + _digest(
        {
            "name": name,
            "scenario_ids": selected,
            "split_by": split_by,
            "split_ratios": ratios,
            "table_versions": table_versions,
        }
    )
    transform_id = "tfm-snapshot-" + dataset_id.removeprefix("ds-")

    query_spec = {
        "source": source,
        "scenario_ids": selected,
        "split_by": split_by,
        "split_ratios": ratios,
    }
    split_payload = {
        "by": split_by,
        "ratios": ratios,
        "counts": counts,
        "assignments": assignments,
    }
    now = datetime.now(UTC)

    snapshots = lake.table("dataset_snapshots")
    snapshots.delete(f"dataset_id = '{dataset_id}'")
    snapshots.add(
        pa.Table.from_pylist(
            [
                {
                    "dataset_id": dataset_id,
                    "name": name,
                    "kind": "scenario-snapshot",
                    "query_spec": json.dumps(query_spec, sort_keys=True),
                    "table_versions": table_versions,
                    "tag": tag,
                    "split": json.dumps(split_payload, sort_keys=True),
                    "balance_report": (
                        json.dumps(balance_report, sort_keys=True)
                        if balance_report is not None
                        else None
                    ),
                    "coverage_report": (
                        json.dumps(coverage_report, sort_keys=True)
                        if coverage_report is not None
                        else None
                    ),
                    "created_by": created_by,
                    "transform_id": transform_id,
                    "created_at": now,
                }
            ],
            schema=DATASET_SNAPSHOTS_SCHEMA,
        )
    )

    transform_row = {
        "transform_id": transform_id,
        "kind": "dataset-snapshot",
        "input_uris": [],
        "input_table_versions": table_versions,
        "output_tables": ["dataset_snapshots"],
        "params": json.dumps(
            attach_lineage_context_to_params(
                {
                    "dataset_id": dataset_id,
                    "name": name,
                    "query_spec": query_spec,
                    "split": split_payload,
                    "balance_report": balance_report,
                    "coverage_report": coverage_report,
                },
                context,
            ),
            sort_keys=True,
        ),
        "status": "completed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit the snapshot's lineage slice inline so provenance is present without a
    # later refresh_graph() (backlog 0098). Best-effort: a projection error must
    # not fail snapshot creation.
    emit_transform_lineage(lake, transform_row)

    return SnapshotManifest(
        lake_uri=lake.uri,
        dataset_id=dataset_id,
        name=name,
        tag=tag,
        transform_id=transform_id,
        scenario_ids=tuple(selected),
        split_by=split_by,
        split_ratios=ratios,
        split_counts=counts,
        split_assignments=assignments,
        table_versions=tuple((tv["table"], tv["version"]) for tv in table_versions),
        source=source,
        balance_report=balance_report,
        coverage_report=coverage_report,
    )
