"""Deterministic source evidence packs for lineage traces.

Evidence packs are intentionally metadata-first. A plan manifest lists source
coordinates, table/version pins, row ids, writeback rows, payload/video/blob
references, and transform provenance without reading blob bytes. Materialized
packs fetch selected blobs into a directory only when the caller asks for them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lancedb_robotics.blob import (
    ATTACHMENT_DATA_COLUMN,
    PAYLOAD_BLOB_COLUMN,
    fetch_blobs,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.video import VIDEO_ENCODING_BLOB_COLUMN

if TYPE_CHECKING:
    from lancedb_robotics.lineage import LineageGraph, RegressionTrace


EVIDENCE_PACK_SCHEMA = "lancedb-robotics/evidence-pack/v1"

#: Evidence-pack manifest schema versions that downstream consumers (replay
#: bundles, the durable catalog, materialization) can load. New manifest
#: versions are appended here so old manifests stay loadable (backlog 0108).
SUPPORTED_EVIDENCE_PACK_SCHEMAS: tuple[str, ...] = (EVIDENCE_PACK_SCHEMA,)


def is_supported_evidence_schema(schema: Any) -> bool:
    """True if ``schema`` is an evidence-pack manifest version this code loads."""
    return schema in SUPPORTED_EVIDENCE_PACK_SCHEMAS


_BLOB_COLUMNS = {
    "observations": {PAYLOAD_BLOB_COLUMN},
    "attachments": {ATTACHMENT_DATA_COLUMN},
    "video_encodings": {VIDEO_ENCODING_BLOB_COLUMN},
}
_ID_COLUMNS = {
    "runs": "run_id",
    "episodes": "episode_id",
    "observations": "observation_id",
    "videos": "video_id",
    "video_encodings": "encoding_id",
    "attachments": "attachment_id",
    "events": "event_id",
    "scenarios": "scenario_id",
    "dataset_snapshots": "dataset_id",
    "training_runs": "training_run_id",
    "model_artifacts": "model_artifact_id",
    "evaluation_runs": "eval_run_id",
    "labels": "label_id",
    "model_outputs": "model_output_id",
    "feedback": "feedback_id",
    "alignment_jobs": "alignment_id",
    "aligned_frames": "aligned_frame_id",
    "aligned_ticks": "aligned_tick_id",
    "transform_runs": "transform_id",
}
_ROW_TABLES = {
    "runs",
    "episodes",
    "observations",
    "videos",
    "video_encodings",
    "attachments",
    "events",
    "scenarios",
    "dataset_snapshots",
    "training_runs",
    "model_artifacts",
    "evaluation_runs",
    "labels",
    "model_outputs",
    "feedback",
    "alignment_jobs",
    "aligned_frames",
    "aligned_ticks",
    "transform_runs",
}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class EvidencePackError(Exception):
    """Raised when an evidence pack cannot be planned or materialized."""


@dataclass(frozen=True)
class EvidencePackReport:
    """JSON-ready report returned by evidence-pack APIs."""

    lake_uri: str
    subject: str
    mode: str
    manifest_digest: str
    bytes_copied: int
    files: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    output_dir: str | None = None
    manifest_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "subject": self.subject,
            "mode": self.mode,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "manifest_digest": self.manifest_digest,
            "bytes_copied": self.bytes_copied,
            "files": list(self.files),
            "manifest": self.manifest,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


def evidence_pack_from_trace(
    lake: Lake,
    trace: RegressionTrace,
    *,
    output_dir: str | Path | None = None,
    materialize: bool = False,
    include_payloads: bool = False,
    include_attachments: bool = False,
    include_video: bool = False,
    redaction_policy: Any | None = None,
) -> EvidencePackReport:
    """Build a deterministic evidence pack from a checkpoint regression trace."""

    table_versions = _table_versions_by_name(trace.table_versions)
    row_ids = _row_ids_from_trace(trace)
    rows = _collect_rows(lake, row_ids, table_versions)
    writeback = _related_writeback_rows(lake, row_ids)
    _merge_rows(rows, writeback)
    media_refs = _media_refs(lake, row_ids, table_versions)

    manifest = {
        "schema_version": EVIDENCE_PACK_SCHEMA,
        "lake_uri": lake.uri,
        "mode": _mode(materialize),
        "subject": {
            "type": "checkpoint-trace",
            "model_run_id": trace.model_run_id,
            "where": trace.where,
            "dataset_id": trace.dataset_snapshot.get("dataset_id"),
        },
        "dataset_snapshot": _json_ready(trace.dataset_snapshot),
        "table_versions": _version_rows(table_versions),
        "row_ids": _sorted_row_ids(row_ids),
        "source_coordinates": _source_coordinates_from_trace(trace),
        "rows": _sorted_rows_map(rows),
        "model_outputs": _json_ready(list(trace.model_outputs)),
        "model_artifacts": _json_ready(list(trace.model_artifacts)),
        "training_run": _json_ready(trace.training_run),
        "lineage_executions": [],
        "transform_runs": _sorted_rows(list(trace.transform_runs), "transform_id"),
        "payload_refs": _payload_refs(rows.get("observations", ()), table_versions),
        "attachment_refs": media_refs["attachments"],
        "video_refs": media_refs["videos"],
        "video_encoding_refs": media_refs["video_encodings"],
        "materialized_files": [],
        "verification": {},
    }
    return _finalize_pack(
        lake,
        manifest,
        output_dir=output_dir,
        materialize=materialize,
        include_payloads=include_payloads,
        include_attachments=include_attachments,
        include_video=include_video,
        redaction_policy=redaction_policy,
    )


def evidence_pack_from_graph(
    lake: Lake,
    graph: LineageGraph,
    *,
    output_dir: str | Path | None = None,
    materialize: bool = False,
    include_payloads: bool = False,
    include_attachments: bool = False,
    include_video: bool = False,
    redaction_policy: Any | None = None,
) -> EvidencePackReport:
    """Build a deterministic evidence pack from a canonical lineage graph."""

    table_versions = _table_versions_from_graph(graph)
    row_ids = _row_ids_from_graph(graph)
    rows = _collect_rows(lake, row_ids, table_versions)
    writeback = _related_writeback_rows(lake, row_ids)
    _merge_rows(rows, writeback)
    media_refs = _media_refs(lake, row_ids, table_versions)

    manifest = {
        "schema_version": EVIDENCE_PACK_SCHEMA,
        "lake_uri": lake.uri,
        "mode": _mode(materialize),
        "subject": {
            "type": "lineage-graph",
            "handle": graph.resolved_handle or graph.root_artifact_id,
            "direction": graph.direction,
            "root_artifact_ids": list(graph.root_artifact_ids or (graph.root_artifact_id,)),
        },
        "dataset_snapshot": _dataset_snapshot_from_rows(rows),
        "table_versions": _version_rows(table_versions),
        "row_ids": _sorted_row_ids(row_ids),
        "source_coordinates": _source_coordinates_from_graph(graph),
        "rows": _sorted_rows_map(rows),
        "model_outputs": _sorted_rows(rows.get("model_outputs", ()), "model_output_id"),
        "model_artifacts": _sorted_rows(rows.get("model_artifacts", ()), "model_artifact_id"),
        "training_run": _first_or_none(_sorted_rows(rows.get("training_runs", ()), "training_run_id")),
        "lineage_graph": graph.as_dict(),
        "lineage_executions": _sorted_rows(list(graph.executions), "execution_id"),
        "transform_runs": _sorted_rows(rows.get("transform_runs", ()), "transform_id"),
        "payload_refs": _payload_refs(rows.get("observations", ()), table_versions),
        "attachment_refs": media_refs["attachments"],
        "video_refs": media_refs["videos"],
        "video_encoding_refs": media_refs["video_encodings"],
        "materialized_files": [],
        "verification": {},
    }
    return _finalize_pack(
        lake,
        manifest,
        output_dir=output_dir,
        materialize=materialize,
        include_payloads=include_payloads,
        include_attachments=include_attachments,
        include_video=include_video,
        redaction_policy=redaction_policy,
    )


def _finalize_pack(
    lake: Lake,
    manifest: dict[str, Any],
    *,
    output_dir: str | Path | None,
    materialize: bool,
    include_payloads: bool,
    include_attachments: bool,
    include_video: bool,
    redaction_policy: Any | None = None,
) -> EvidencePackReport:
    files: list[dict[str, Any]] = []
    bytes_copied = 0
    destination: Path | None = None

    # Apply redaction before hashing / materializing so denied context and
    # environment keys never reach the pack digest or the written manifest.
    if redaction_policy is not None:
        manifest = redaction_policy.redact_manifest(manifest)

    if materialize:
        if output_dir is None:
            raise EvidencePackError("output_dir is required when materialize=True")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        files.extend(
            _materialize_blobs(
                lake,
                manifest,
                destination,
                include_payloads=include_payloads,
                include_attachments=include_attachments,
                include_video=include_video,
            )
        )
        bytes_copied = sum(int(item["bytes"]) for item in files)

    manifest["materialized_files"] = files
    manifest["verification"] = _verification(manifest, files)
    digest = _stable_sha256(manifest)

    manifest_path = None
    if destination is not None:
        manifest_path = str(destination / "manifest.json")
        (destination / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
        )

    return EvidencePackReport(
        lake_uri=lake.uri,
        subject=str(manifest["subject"].get("handle") or manifest["subject"].get("model_run_id")),
        mode=_mode(materialize),
        output_dir=str(destination) if destination is not None else None,
        manifest_path=manifest_path,
        manifest_digest=digest,
        bytes_copied=bytes_copied,
        files=tuple(files),
        manifest=_json_ready(manifest),
    )


def _materialize_blobs(
    lake: Lake,
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    include_payloads: bool,
    include_attachments: bool,
    include_video: bool,
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if include_payloads:
        files.extend(_materialize_payloads(lake, manifest, output_dir))
    if include_attachments:
        files.extend(_materialize_attachments(lake, manifest, output_dir))
    if include_video:
        files.extend(_materialize_video_encodings(lake, manifest, output_dir))
    return sorted(files, key=lambda item: (item["kind"], item["path"]))


def _materialize_payloads(
    lake: Lake,
    manifest: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    refs = manifest.get("payload_refs") or []
    ids = [ref["row_id"] for ref in refs]
    table_version = _table_version(manifest, "observations")
    blobs = _fetch_blobs_as_of(
        lake,
        "observations",
        PAYLOAD_BLOB_COLUMN,
        ids,
        id_column="observation_id",
        table_version=table_version,
    )
    files = []
    for row_id in sorted(blobs):
        payload = blobs[row_id]
        if not payload:
            continue
        path = output_dir / "payloads" / f"{_safe_name(row_id)}.bin"
        files.append(_write_materialized_file(path, payload, kind="payload", table="observations", row_id=row_id))
    return files


def _materialize_attachments(
    lake: Lake,
    manifest: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    refs = manifest.get("attachment_refs") or []
    ids = [ref["attachment_id"] for ref in refs]
    table_version = _table_version(manifest, "attachments")
    blobs = _fetch_blobs_as_of(
        lake,
        "attachments",
        ATTACHMENT_DATA_COLUMN,
        ids,
        id_column="attachment_id",
        table_version=table_version,
    )
    files = []
    for row_id in sorted(blobs):
        payload = blobs[row_id]
        if not payload:
            continue
        path = output_dir / "attachments" / f"{_safe_name(row_id)}.bin"
        files.append(_write_materialized_file(path, payload, kind="attachment", table="attachments", row_id=row_id))
    return files


def _materialize_video_encodings(
    lake: Lake,
    manifest: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    refs = manifest.get("video_encoding_refs") or []
    ids = [ref["encoding_id"] for ref in refs]
    table_version = _table_version(manifest, "video_encodings")
    blobs = _fetch_blobs_as_of(
        lake,
        "video_encodings",
        VIDEO_ENCODING_BLOB_COLUMN,
        ids,
        id_column="encoding_id",
        table_version=table_version,
    )
    files = []
    for row_id in sorted(blobs):
        payload = blobs[row_id]
        if not payload:
            continue
        path = output_dir / "video_encodings" / f"{_safe_name(row_id)}.bin"
        files.append(
            _write_materialized_file(
                path,
                payload,
                kind="video-encoding",
                table="video_encodings",
                row_id=row_id,
            )
        )
    return files


def _write_materialized_file(
    path: Path,
    payload: bytes,
    *,
    kind: str,
    table: str,
    row_id: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "kind": kind,
        "table": table,
        "row_id": row_id,
        "path": str(path.relative_to(path.parents[1])),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fetch_blobs_as_of(
    lake: Lake,
    table_name: str,
    column: str,
    ids: Iterable[str],
    *,
    id_column: str,
    table_version: int | None,
) -> dict[str, bytes]:
    table = lake.table(table_name)
    if table_version is not None:
        table.checkout(int(table_version))
    try:
        return fetch_blobs(table, column, ids, id_column=id_column)
    finally:
        if table_version is not None:
            table.checkout_latest()


def _row_ids_from_trace(trace: RegressionTrace) -> dict[str, set[str]]:
    rows: dict[str, set[str]] = defaultdict(set)
    snapshot_id = trace.dataset_snapshot.get("dataset_id")
    if snapshot_id:
        rows["dataset_snapshots"].add(str(snapshot_id))
    if trace.training_run and trace.training_run.get("training_run_id"):
        rows["training_runs"].add(str(trace.training_run["training_run_id"]))
    for row in trace.model_artifacts:
        if row.get("model_artifact_id"):
            rows["model_artifacts"].add(str(row["model_artifact_id"]))
    for row in trace.model_outputs:
        if row.get("model_output_id"):
            rows["model_outputs"].add(str(row["model_output_id"]))
    for row in trace.rows:
        for table_name, key in (
            ("runs", "run_id"),
            ("observations", "observation_id"),
            ("scenarios", "scenario_id"),
        ):
            if row.get(key):
                rows[table_name].add(str(row[key]))
    for row in trace.transform_runs:
        if row.get("transform_id"):
            rows["transform_runs"].add(str(row["transform_id"]))
    return rows


def _row_ids_from_graph(graph: LineageGraph) -> dict[str, set[str]]:
    rows: dict[str, set[str]] = defaultdict(set)
    for artifact in graph.artifacts:
        table_name = artifact.get("row_grain") or artifact.get("table_name")
        if table_name not in _ROW_TABLES:
            continue
        for row_id in artifact.get("row_ids") or []:
            if row_id:
                rows[str(table_name)].add(str(row_id))
    for execution in graph.executions:
        transform_id = execution.get("transform_id")
        if transform_id:
            rows["transform_runs"].add(str(transform_id))
        execution_id = str(execution.get("execution_id") or "")
        if execution_id.startswith("lancedb-robotics:execution:"):
            rows["transform_runs"].add(execution_id.rsplit(":", 1)[-1])
    return rows


def _collect_rows(
    lake: Lake,
    row_ids: Mapping[str, set[str]],
    table_versions: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for table_name, ids in sorted(row_ids.items()):
        id_column = _ID_COLUMNS.get(table_name)
        if not id_column or not ids:
            continue
        table_rows = _rows_as_of(lake, table_name, table_versions.get(table_name))
        selected = [row for row in table_rows if str(row.get(id_column)) in ids]
        if selected:
            rows[table_name] = _sorted_rows(selected, id_column)
    return rows


def _related_writeback_rows(
    lake: Lake,
    row_ids: Mapping[str, set[str]],
) -> dict[str, list[dict[str, Any]]]:
    observation_ids = row_ids.get("observations", set())
    scenario_ids = row_ids.get("scenarios", set())
    run_ids = row_ids.get("runs", set())
    model_output_ids = row_ids.get("model_outputs", set())
    label_ids = row_ids.get("labels", set())

    rows: dict[str, list[dict[str, Any]]] = {}
    labels = [
        row
        for row in _rows_as_of(lake, "labels", None)
        if _matches_any(row, observation_ids=observation_ids, scenario_ids=scenario_ids, run_ids=run_ids)
        or str(row.get("label_id")) in label_ids
    ]
    if labels:
        rows["labels"] = _sorted_rows(labels, "label_id")
        label_ids = label_ids | {str(row["label_id"]) for row in labels if row.get("label_id")}

    model_outputs = [
        row
        for row in _rows_as_of(lake, "model_outputs", None)
        if _matches_any(row, observation_ids=observation_ids, scenario_ids=scenario_ids, run_ids=run_ids)
        or str(row.get("model_output_id")) in model_output_ids
    ]
    if model_outputs:
        rows["model_outputs"] = _sorted_rows(model_outputs, "model_output_id")
        model_output_ids = model_output_ids | {
            str(row["model_output_id"]) for row in model_outputs if row.get("model_output_id")
        }

    feedback = [
        row
        for row in _rows_as_of(lake, "feedback", None)
        if _matches_any(row, observation_ids=observation_ids, scenario_ids=scenario_ids, run_ids=run_ids)
        or str(row.get("model_output_id")) in model_output_ids
        or str(row.get("label_id")) in label_ids
        or str(row.get("feedback_id")) in row_ids.get("feedback", set())
    ]
    if feedback:
        rows["feedback"] = _sorted_rows(feedback, "feedback_id")
    return rows


def _matches_any(
    row: dict[str, Any],
    *,
    observation_ids: set[str],
    scenario_ids: set[str],
    run_ids: set[str],
) -> bool:
    return (
        (row.get("observation_id") is not None and str(row["observation_id"]) in observation_ids)
        or (row.get("scenario_id") is not None and str(row["scenario_id"]) in scenario_ids)
        or (row.get("run_id") is not None and str(row["run_id"]) in run_ids)
    )


def _merge_rows(target: dict[str, list[dict[str, Any]]], extra: Mapping[str, list[dict[str, Any]]]) -> None:
    for table_name, incoming in extra.items():
        id_column = _ID_COLUMNS.get(table_name)
        if not id_column:
            continue
        merged = {str(row[id_column]): row for row in target.get(table_name, []) if row.get(id_column)}
        merged.update({str(row[id_column]): row for row in incoming if row.get(id_column)})
        target[table_name] = _sorted_rows(merged.values(), id_column)


def _media_refs(
    lake: Lake,
    row_ids: Mapping[str, set[str]],
    table_versions: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    run_ids = row_ids.get("runs", set())
    observation_ids = row_ids.get("observations", set())
    episode_ids = row_ids.get("episodes", set())

    attachments = [
        _attachment_ref(row, table_versions.get("attachments"))
        for row in _rows_as_of(lake, "attachments", table_versions.get("attachments"))
        if str(row.get("run_id")) in run_ids
    ]

    videos = [
        row
        for row in _rows_as_of(lake, "videos", table_versions.get("videos"))
        if str(row.get("run_id")) in run_ids
        or str(row.get("episode_id")) in episode_ids
        or observation_ids.intersection(str(value) for value in row.get("observation_ids") or [])
    ]
    video_ids = {str(row["video_id"]) for row in videos if row.get("video_id")}
    encodings = [
        row
        for row in _rows_as_of(lake, "video_encodings", table_versions.get("video_encodings"))
        if str(row.get("video_id")) in video_ids
    ]
    return {
        "attachments": sorted(attachments, key=lambda row: row["attachment_id"]),
        "videos": [_video_ref(row, table_versions.get("videos")) for row in _sorted_rows(videos, "video_id")],
        "video_encodings": [
            _video_encoding_ref(row, table_versions.get("video_encodings"))
            for row in _sorted_rows(encodings, "encoding_id")
        ],
    }


def _attachment_ref(row: dict[str, Any], table_version: int | None) -> dict[str, Any]:
    return {
        "table": "attachments",
        "table_version": table_version,
        "attachment_id": row["attachment_id"],
        "run_id": row.get("run_id"),
        "name": row.get("name"),
        "media_type": row.get("media_type"),
        "size": row.get("size"),
        "sha256": row.get("sha256"),
        "column": ATTACHMENT_DATA_COLUMN,
    }


def _video_ref(row: dict[str, Any], table_version: int | None) -> dict[str, Any]:
    return {
        "table": "videos",
        "table_version": table_version,
        "video_id": row["video_id"],
        "run_id": row.get("run_id"),
        "episode_id": row.get("episode_id"),
        "camera_key": row.get("camera_key"),
        "topic": row.get("topic"),
        "codec": row.get("codec"),
        "uri": row.get("uri"),
        "raw_uri": row.get("raw_uri"),
        "observation_ids": list(row.get("observation_ids") or []),
    }


def _video_encoding_ref(row: dict[str, Any], table_version: int | None) -> dict[str, Any]:
    return {
        "table": "video_encodings",
        "table_version": table_version,
        "encoding_id": row["encoding_id"],
        "video_id": row.get("video_id"),
        "codec": row.get("codec"),
        "gop_size": row.get("gop_size"),
        "keyframe_map_ref": row.get("keyframe_map_ref"),
        "encoded_size_bytes": row.get("encoded_size_bytes"),
        "column": VIDEO_ENCODING_BLOB_COLUMN,
    }


def _payload_refs(
    observations: Iterable[dict[str, Any]],
    table_versions: Mapping[str, int],
) -> list[dict[str, Any]]:
    refs = []
    for row in observations:
        refs.append(
            {
                "table": "observations",
                "table_version": table_versions.get("observations"),
                "id_column": "observation_id",
                "row_id": row["observation_id"],
                "column": PAYLOAD_BLOB_COLUMN,
                "raw_uri": row.get("raw_uri"),
                "raw_channel": row.get("raw_channel") or row.get("topic"),
                "raw_sequence": row.get("raw_sequence"),
                "raw_log_time_ns": row.get("raw_log_time_ns"),
                "decode_status": row.get("decode_status"),
                "message_encoding": row.get("message_encoding"),
                "schema_encoding": row.get("schema_encoding"),
            }
        )
    return sorted(refs, key=lambda row: row["row_id"])


def _source_coordinates_from_trace(trace: RegressionTrace) -> list[dict[str, Any]]:
    coords = []
    for coord in trace.source_logs:
        row = coord.as_dict()
        row["coordinate_hash"] = _stable_sha256(row)
        coords.append(row)
    return sorted(coords, key=lambda row: (row["uri"], row["channel"], row.get("offset") or -1))


def _source_coordinates_from_graph(graph: LineageGraph) -> list[dict[str, Any]]:
    observation_by_artifact = {
        row["artifact_id"]: row
        for row in graph.artifacts
        if row.get("row_grain") == "observations" or row.get("table_name") == "observations"
    }
    source_rows = {
        row["artifact_id"]: row
        for row in graph.artifacts
        if row.get("kind") == "source" and row.get("source_uri")
    }
    coords_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source_id, source in source_rows.items():
        metadata = _metadata_map(source)
        linked_observation_ids = []
        for edge in graph.edges:
            if edge.get("from_artifact_id") != source_id:
                continue
            observation = observation_by_artifact.get(edge.get("to_artifact_id"))
            if observation:
                linked_observation_ids.extend(observation.get("row_ids") or [])
        row = {
            "uri": source.get("source_uri"),
            "channel": metadata.get("channel"),
            "offset": _maybe_int(metadata.get("offset")),
            "log_time_ns": _maybe_int(metadata.get("log_time_ns")),
            "source_id": source.get("source_id"),
            "observation_ids": sorted({str(value) for value in linked_observation_ids if value}),
        }
        row["coordinate_hash"] = _stable_sha256(row)
        key = (row["uri"], row["channel"], row["offset"], row["log_time_ns"], tuple(row["observation_ids"]))
        coords_by_key[key] = row
    return [
        coords_by_key[key]
        for key in sorted(
            coords_by_key,
            key=lambda item: tuple("" if value is None else str(value) for value in item),
        )
    ]


def _rows_as_of(
    lake: Lake,
    table_name: str,
    version: int | None,
) -> list[dict[str, Any]]:
    table = lake.table(table_name)
    if version is not None:
        table.checkout(int(version))
    try:
        columns = [
            name
            for name in table.schema.names
            if name not in _BLOB_COLUMNS.get(table_name, set())
        ]
        return table.to_lance().to_table(columns=columns).to_pylist()
    finally:
        if version is not None:
            table.checkout_latest()


def _table_versions_by_name(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    versions: dict[str, int] = {}
    for row in rows:
        table = row.get("table")
        version = row.get("version")
        if table and version is not None:
            versions[str(table)] = int(version)
    return versions


def _table_versions_from_graph(graph: LineageGraph) -> dict[str, int]:
    versions: dict[str, int] = {}
    for artifact in graph.artifacts:
        table = artifact.get("table_name")
        version = artifact.get("table_version")
        if table and version is not None:
            versions.setdefault(str(table), int(version))
    return versions


def _version_rows(table_versions: Mapping[str, int]) -> list[dict[str, Any]]:
    return [
        {"table": table, "version": int(version), "tag": ""}
        for table, version in sorted(table_versions.items())
    ]


def _table_version(manifest: dict[str, Any], table_name: str) -> int | None:
    for row in manifest.get("table_versions") or []:
        if row.get("table") == table_name and row.get("version") is not None:
            return int(row["version"])
    return None


def _dataset_snapshot_from_rows(rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    snapshots = _sorted_rows(rows.get("dataset_snapshots", ()), "dataset_id")
    return snapshots[0] if snapshots else None


def _verification(manifest: dict[str, Any], files: list[dict[str, Any]]) -> dict[str, Any]:
    source_hashes = [row["coordinate_hash"] for row in manifest.get("source_coordinates") or []]
    return {
        "source_coordinate_hashes": source_hashes,
        "materialized_file_hashes": [
            {"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]}
            for row in files
        ],
    }


def _sorted_row_ids(row_ids: Mapping[str, set[str]]) -> dict[str, list[str]]:
    return {table: sorted(ids) for table, ids in sorted(row_ids.items()) if ids}


def _sorted_rows_map(rows: Mapping[str, Iterable[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for table_name, table_rows in sorted(rows.items()):
        id_column = _ID_COLUMNS.get(table_name, "")
        result[table_name] = _sorted_rows(table_rows, id_column)
    return _json_ready(result)


def _sorted_rows(rows: Iterable[dict[str, Any]], id_column: str) -> list[dict[str, Any]]:
    return sorted(
        (_json_ready(dict(row)) for row in rows),
        key=lambda row: str(row.get(id_column) or row),
    )


def _metadata_map(row: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("key")): "" if item.get("value") is None else str(item.get("value"))
        for item in row.get("metadata") or []
        if isinstance(item, dict) and item.get("key") is not None
    }


def _maybe_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_or_none(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_name(value: str) -> str:
    return _SAFE_NAME_RE.sub("_", str(value)).strip("._") or "item"


def _mode(materialize: bool) -> str:
    return "materialized" if materialize else "plan"
