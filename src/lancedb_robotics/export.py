"""Export selected scenario clips back into robotics tools.

The boundary the lakehouse is built to respect (backlog 0011 / Feature Set 8):
it selects, traces, and projects slices back into MCAP/Foxglove/Rerun-style
workflows rather than replacing them. :func:`export_snapshot` reads a dataset
snapshot's scenario windows and emits an export manifest — row IDs, time
windows, source URIs, lossiness/capability metadata, and optional replay links
— then writes one lossless MCAP clip per scenario when the source adapter
supports reversible slicing.

When clip export is unavailable (the adapter cannot slice, or the raw source is
no longer reachable) the manifest still describes the plan and records a clear,
per-clip skip reason, so the selection and its lineage survive even when the
bytes cannot be re-emitted here.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from lancedb_robotics.adapters import get_adapter
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.lineage_hooks import (
    attach_lineage_context_to_params,
    begin_lineage_execution,
    lineage_context_digest,
)
from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA

DEFAULT_FORMAT = "mcap"
MANIFEST_FILENAME = "export_manifest.json"

STATUS_EXPORTED = "exported"
STATUS_SKIPPED = "skipped"
STATUS_PLANNED = "planned"

LOSSINESS_LOSSLESS = "lossless-slice"
LOSSINESS_PLAN_ONLY = "plan-only"


class ExportError(Exception):
    """Raised when an export cannot be produced."""


@dataclass(frozen=True)
class ClipExport:
    """One scenario's clip plan/result with lineage and lossiness metadata."""

    scenario_id: str
    run_id: str
    start_time_ns: int
    end_time_ns: int
    topics: tuple[str, ...]
    observation_ids: tuple[str, ...]
    source_uri: str | None
    out_path: str | None
    status: str
    lossiness: str
    message_count: int | None = None
    reason: str | None = None
    external_links: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "topics": list(self.topics),
            "observation_ids": list(self.observation_ids),
            "source_uri": self.source_uri,
            "out_path": self.out_path,
            "status": self.status,
            "lossiness": self.lossiness,
            "message_count": self.message_count,
            "reason": self.reason,
            "external_links": self.external_links,
        }


@dataclass(frozen=True)
class ExportManifest:
    """Manifest for one snapshot export: per-clip plans/results + counts."""

    lake_uri: str
    dataset_id: str
    snapshot_name: str
    format: str
    out_dir: str
    transform_id: str
    clips: list[ClipExport]
    lineage_context: dict = field(default_factory=dict)

    @property
    def exported(self) -> int:
        return sum(1 for clip in self.clips if clip.status == STATUS_EXPORTED)

    @property
    def skipped(self) -> int:
        return sum(1 for clip in self.clips if clip.status == STATUS_SKIPPED)

    @property
    def planned(self) -> int:
        return sum(1 for clip in self.clips if clip.status == STATUS_PLANNED)

    def to_dict(self) -> dict:
        payload = {
            "lake_uri": self.lake_uri,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "format": self.format,
            "out_dir": self.out_dir,
            "transform_id": self.transform_id,
            "exported": self.exported,
            "skipped": self.skipped,
            "planned": self.planned,
            "clips": [clip.to_dict() for clip in self.clips],
        }
        if self.lineage_context:
            payload["lineage_context"] = dict(self.lineage_context)
        return payload


def _latest_snapshot_row(lake: Lake, name: str) -> dict:
    rows = [
        row for row in lake.table("dataset_snapshots").to_arrow().to_pylist() if row["name"] == name
    ]
    if not rows:
        raise ExportError(f"no snapshot named {name!r} in {lake.uri}")
    return max(rows, key=lambda row: (row["created_at"], row["dataset_id"]))


def _external_links(clip_path: Path) -> list[dict]:
    """Illustrative replay-tool hints for opening the clip locally."""
    return [
        {"tool": "foxglove", "kind": "open-file", "target": str(clip_path)},
        {"tool": "rerun", "kind": "open-file", "target": str(clip_path)},
    ]


def _slice_window(
    source_uri: str,
    *,
    start_time_ns: int,
    end_time_ns: int,
    out_path: Path,
    topics: tuple[str, ...],
) -> dict:
    """Slice ``[start, end]`` from a source into one clip — file or split recording.

    A single file goes through the adapter's proven single-file slicer; a split
    recording (directory) merges the in-window messages across its ordered shards
    into one clip, so a window that spans a shard boundary still yields a single
    correct file (backlog 0019).
    """
    from lancedb_robotics.recordings import export_window, resolve_shards

    plan = resolve_shards(source_uri)
    if plan.is_split:
        return export_window(
            plan.paths,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            out_path=out_path,
            topics=topics,
        )
    return get_adapter("mcap").export(
        plan.paths[0],
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        out_path=out_path,
        topics=topics,
    )


def export_snapshot(
    lake: Lake,
    snapshot_name: str,
    *,
    out_dir: str | Path,
    fmt: str = DEFAULT_FORMAT,
    plan_only: bool = False,
    adapter_name: str = "mcap",
    created_by: str = "lancedb-robotics",
    lineage_context: object | None = None,
) -> ExportManifest:
    """Plan and (unless ``plan_only``) write per-scenario clips for a snapshot."""
    if fmt != DEFAULT_FORMAT:
        raise ExportError(
            f"unsupported export format {fmt!r}; only {DEFAULT_FORMAT!r} is supported"
        )

    row = _latest_snapshot_row(lake, snapshot_name)
    dataset_id = row["dataset_id"]
    scenario_ids = sorted(json.loads(row["query_spec"]).get("scenario_ids", []))

    scenarios = {
        scenario["scenario_id"]: scenario
        for scenario in lake.table("scenarios").to_arrow().to_pylist()
    }
    source_uris = {
        run["run_id"]: run["raw_uri"] for run in lake.table("runs").to_arrow().to_pylist()
    }

    adapter = get_adapter(adapter_name)
    can_slice = "export" in adapter.info.capabilities
    handle = begin_lineage_execution(
        lineage_context,
        operation="export",
        params={
            "snapshot_name": snapshot_name,
            "dataset_id": dataset_id,
            "format": fmt,
            "plan_only": plan_only,
        },
    )
    context = handle.finish(status="completed")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips: list[ClipExport] = []
    for scenario_id in scenario_ids:
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            continue
        run_id = scenario["run_id"]
        topics = tuple(scenario["topics"] or ())
        observation_ids = tuple(scenario["observation_ids"] or ())
        source_uri = source_uris.get(run_id)
        clip_path = out_dir / f"{scenario_id}.{fmt}"
        base = {
            "scenario_id": scenario_id,
            "run_id": run_id,
            "start_time_ns": scenario["start_time_ns"],
            "end_time_ns": scenario["end_time_ns"],
            "topics": topics,
            "observation_ids": observation_ids,
            "source_uri": source_uri,
            "external_links": _external_links(clip_path),
        }

        if plan_only:
            clips.append(
                ClipExport(
                    **base,
                    out_path=str(clip_path),
                    status=STATUS_PLANNED,
                    lossiness=LOSSINESS_PLAN_ONLY,
                    reason="plan-only: no clip written",
                )
            )
            continue
        if not can_slice:
            clips.append(
                ClipExport(
                    **base,
                    out_path=None,
                    status=STATUS_SKIPPED,
                    lossiness=LOSSINESS_PLAN_ONLY,
                    reason=f"adapter {adapter_name!r} does not support reversible slicing",
                )
            )
            continue
        # A run's raw_uri may be a single file or a split-recording directory
        # (backlog 0019); both are reachable sources to slice from.
        if not source_uri or not Path(source_uri).exists():
            clips.append(
                ClipExport(
                    **base,
                    out_path=None,
                    status=STATUS_SKIPPED,
                    lossiness=LOSSINESS_PLAN_ONLY,
                    reason=f"source not reachable: {source_uri}",
                )
            )
            continue

        result = _slice_window(
            source_uri,
            start_time_ns=scenario["start_time_ns"],
            end_time_ns=scenario["end_time_ns"],
            out_path=clip_path,
            topics=topics,
        )
        clips.append(
            ClipExport(
                **base,
                out_path=result["out_path"],
                status=STATUS_EXPORTED,
                lossiness=LOSSINESS_LOSSLESS,
                message_count=result["message_count"],
            )
        )

    suffix = lineage_context_digest(context)
    transform_id = f"tfm-export-{dataset_id.removeprefix('ds-')}"
    if suffix:
        transform_id = f"{transform_id}-{suffix}"
    manifest = ExportManifest(
        lake_uri=lake.uri,
        dataset_id=dataset_id,
        snapshot_name=row["name"],
        format=fmt,
        out_dir=str(out_dir),
        transform_id=transform_id,
        clips=clips,
        lineage_context=context.to_dict() if context else {},
    )

    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2)
    )
    _record_transform(lake, manifest, created_by=created_by)
    return manifest


def _record_transform(lake: Lake, manifest: ExportManifest, *, created_by: str) -> None:
    now = datetime.now(UTC)
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{manifest.transform_id}'")
    transform_row = {
        "transform_id": manifest.transform_id,
        "kind": "export",
        "input_uris": [clip.source_uri for clip in manifest.clips if clip.source_uri],
        "input_table_versions": [],
        "output_tables": [],
        "params": json.dumps(
            attach_lineage_context_to_params(
                {
                    "dataset_id": manifest.dataset_id,
                    "snapshot_name": manifest.snapshot_name,
                    "format": manifest.format,
                    "out_dir": manifest.out_dir,
                    "exported": manifest.exported,
                    "skipped": manifest.skipped,
                    "planned": manifest.planned,
                    "scenario_ids": [clip.scenario_id for clip in manifest.clips],
                },
                manifest.lineage_context,
            ),
            sort_keys=True,
        ),
        "status": "completed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): export transform without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
