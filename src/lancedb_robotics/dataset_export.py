"""Dataset snapshot exports to training-native projection layouts.

Backlog 0026 is a static projection path: a version-pinned
``dataset_snapshots`` row is materialized into LeRobot- or RLDS-shaped files,
with a manifest that keeps the exported bytes traceable to the Lance source of
truth. The writer is intentionally dependency-light. Native ``lerobot``/RLDS
packages are detected lazily and reported in the manifest, so environments that
do not install those large extras still get a clear availability signal instead
of an import-time crash.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import math
import re
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lancedb_robotics.blob import PAYLOAD_BLOB_COLUMN, fetch_blobs
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.materialization import (
    ProjectionAccounting,
    metadata_bytes_written,
    payload_size,
)
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, TRANSFORM_RUNS_SCHEMA
from lancedb_robotics.video import VIDEO_ENCODING_BLOB_COLUMN

LEROBOT_FORMAT = "lerobot"
RLDS_FORMAT = "rlds"
WEBDATASET_FORMAT = "webdataset"
DATASET_EXPORT_FORMATS = (LEROBOT_FORMAT, RLDS_FORMAT, WEBDATASET_FORMAT)

LEROBOT_FORMAT_VERSION = "lerobot-v3.0"
RLDS_FORMAT_VERSION = "rlds-tfds-style-v0"
WEBDATASET_FORMAT_VERSION = "webdataset-tar-v0"

DATASET_EXPORT_MANIFEST_FILENAME = "dataset_export_manifest.json"
DEFAULT_WEBDATASET_SHARD_SIZE = 1000
DEFAULT_WEBDATASET_COMPRESSION = "none"
WEBDATASET_COMPRESSIONS = (DEFAULT_WEBDATASET_COMPRESSION, "gzip")

_CHUNK = "chunk-000"
_FILE = "file-000"


class DatasetExportError(Exception):
    """Raised when a dataset snapshot cannot be exported."""


@dataclass(frozen=True)
class DatasetExportManifest:
    """Lineage and reproducibility manifest for one materialized export."""

    lake_uri: str
    dataset_id: str
    snapshot_name: str
    format: str
    format_version: str
    out_dir: str
    transform_id: str
    table_versions: tuple[dict[str, Any], ...]
    feature_spec: dict[str, Any]
    content_hash: str
    episode_count: int
    step_count: int
    data_files: tuple[str, ...]
    native_loader: dict[str, Any]
    lossy_mapping: tuple[str, ...]
    accounting: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "format": self.format,
            "format_version": self.format_version,
            "out_dir": self.out_dir,
            "transform_id": self.transform_id,
            "table_versions": list(self.table_versions),
            "feature_spec": self.feature_spec,
            "content_hash": self.content_hash,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "data_files": list(self.data_files),
            "native_loader": self.native_loader,
            "lossy_mapping": list(self.lossy_mapping),
            "accounting": dict(self.accounting),
            "manifest": DATASET_EXPORT_MANIFEST_FILENAME,
        }


@dataclass(frozen=True)
class _SnapshotContext:
    row: dict[str, Any]
    dataset_id: str
    snapshot_name: str
    scenario_ids: tuple[str, ...]
    split_assignments: dict[str, str]
    table_versions: tuple[dict[str, Any], ...]
    scenarios: dict[str, dict[str, Any]]
    episodes: dict[str, dict[str, Any]]
    observations: dict[str, dict[str, Any]]
    videos: dict[str, dict[str, Any]]
    video_encodings: dict[str, dict[str, Any]]
    runs: dict[str, dict[str, Any]]
    payload_blobs: dict[str, bytes]
    video_encoding_blobs: dict[str, bytes]


@dataclass(frozen=True)
class _Episode:
    episode_id: str
    index: int
    scenario: dict[str, Any]
    physical_episode: dict[str, Any] | None
    run: dict[str, Any]
    observations: tuple[dict[str, Any], ...]
    split: str
    task_index: int
    task: str


def export_dataset_snapshot(
    lake: Lake,
    snapshot_name: str,
    *,
    out_dir: str | Path,
    fmt: str,
    require_native: bool = False,
    shard_size: int = DEFAULT_WEBDATASET_SHARD_SIZE,
    compression: str = DEFAULT_WEBDATASET_COMPRESSION,
    created_by: str = "lancedb-robotics",
    record_materialization: bool = True,
) -> DatasetExportManifest:
    """Export ``snapshot_name`` into ``fmt`` and return its manifest.

    ``fmt`` is one of ``"lerobot"``, ``"rlds"``, or ``"webdataset"``. The
    exported layout is a deterministic projection of the snapshot's pinned table
    versions: re-running the export for the same snapshot and format/options
    yields the same ``content_hash`` even when the destination directory differs.
    """
    if fmt not in DATASET_EXPORT_FORMATS:
        raise DatasetExportError(
            f"unsupported dataset export format {fmt!r}; choose from "
            f"{', '.join(DATASET_EXPORT_FORMATS)}"
        )

    native_loader = native_loader_status(fmt)
    if require_native and not native_loader["available"]:
        missing = ", ".join(native_loader["missing"])
        raise DatasetExportError(
            f"{fmt} native loader unavailable: missing {missing}; "
            f"install {native_loader['install']}"
        )

    context = _snapshot_context(lake, snapshot_name)
    episodes = _episodes(context)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if fmt == LEROBOT_FORMAT:
        format_version = LEROBOT_FORMAT_VERSION
        files, feature_spec, lossy_mapping, step_count = _write_lerobot(out_path, context, episodes)
    elif fmt == RLDS_FORMAT:
        format_version = RLDS_FORMAT_VERSION
        files, feature_spec, lossy_mapping, step_count = _write_rlds(out_path, context, episodes)
    else:
        shard_size = _normalize_webdataset_shard_size(shard_size)
        compression = _normalize_webdataset_compression(compression)
        format_version = WEBDATASET_FORMAT_VERSION
        files, feature_spec, lossy_mapping, step_count = _write_webdataset(
            out_path,
            context,
            episodes,
            shard_size=shard_size,
            compression=compression,
        )

    content_hash = _content_hash(out_path, files)
    transform_id = f"tfm-dataset-export-{fmt}-{context.dataset_id.removeprefix('ds-')}"
    manifest = DatasetExportManifest(
        lake_uri=lake.uri,
        dataset_id=context.dataset_id,
        snapshot_name=context.snapshot_name,
        format=fmt,
        format_version=format_version,
        out_dir=str(out_path),
        transform_id=transform_id,
        table_versions=context.table_versions,
        feature_spec=feature_spec,
        content_hash=content_hash,
        episode_count=len(episodes),
        step_count=step_count,
        data_files=tuple(sorted(files)),
        native_loader=native_loader,
        lossy_mapping=tuple(lossy_mapping),
    )

    output_paths = tuple(
        str(out_path / rel)
        for rel in (*manifest.data_files, DATASET_EXPORT_MANIFEST_FILENAME)
    )
    manifest = _manifest_with_accounting(
        manifest,
        context,
        episodes,
        metadata_bytes=0,
    )
    for _ in range(3):
        _write_json(out_path, DATASET_EXPORT_MANIFEST_FILENAME, manifest.to_dict())
        observed_metadata_bytes = metadata_bytes_written(
            output_paths,
            payload_bytes_copied=manifest.accounting["payload_bytes_copied"],
        )
        if observed_metadata_bytes == manifest.accounting["metadata_bytes_written"]:
            break
        manifest = _manifest_with_accounting(
            manifest,
            context,
            episodes,
            metadata_bytes=observed_metadata_bytes,
        )
    _write_json(out_path, DATASET_EXPORT_MANIFEST_FILENAME, manifest.to_dict())
    _record_transform(lake, manifest, created_by=created_by)
    if record_materialization:
        _record_materialization_accounting(lake, manifest, created_by=created_by)
    return manifest


def _manifest_with_accounting(
    manifest: DatasetExportManifest,
    context: _SnapshotContext,
    episodes: tuple[_Episode, ...],
    *,
    metadata_bytes: int,
) -> DatasetExportManifest:
    selected_observations = tuple(obs for episode in episodes for obs in episode.observations)
    payload_bytes_referenced = _payload_bytes_for_observations(context, selected_observations)
    payload_bytes_copied = _payload_bytes_for_observations(
        context,
        selected_observations,
        camera_only=True,
    )
    accounting = ProjectionAccounting(
        logical_row_count=manifest.step_count,
        selected_scenario_count=len(context.scenario_ids),
        selected_observation_count=len(selected_observations),
        payload_bytes_referenced=payload_bytes_referenced,
        payload_bytes_copied=payload_bytes_copied,
        metadata_bytes_written=int(metadata_bytes),
        target_format=manifest.format,
        target_path=manifest.out_dir,
        projection_transform_id=manifest.transform_id,
        source_snapshot_id=manifest.dataset_id,
        source_snapshot_name=manifest.snapshot_name,
        source_table_versions=manifest.table_versions,
        mode="export",
        payload_copy_policy="materialized-copy" if payload_bytes_copied else "metadata-only",
        dry_run=False,
        payload_bytes_planned=payload_bytes_copied,
    ).to_dict()
    return replace(manifest, accounting=accounting)


def _payload_bytes_for_observations(
    context: _SnapshotContext,
    observations: tuple[dict[str, Any], ...],
    *,
    camera_only: bool = False,
) -> int:
    total = 0
    for obs in observations:
        if camera_only and not _is_camera_observation(obs):
            continue
        observation_id = str(obs.get("observation_id") or "")
        if observation_id in context.payload_blobs:
            total += len(context.payload_blobs.get(observation_id) or b"")
        else:
            total += payload_size(obs.get("payload_blob"))
    return total


def _record_materialization_accounting(
    lake: Lake,
    manifest: DatasetExportManifest,
    *,
    created_by: str,
) -> None:
    accounting = ProjectionAccounting.from_dict(manifest.accounting)
    lake.curate.materialization_report(
        manifest.snapshot_name,
        target_format=manifest.format,
        output_uri=manifest.out_dir,
        mode="export",
        copied_payload_bytes=accounting.payload_bytes_copied,
        metadata_bytes_written=accounting.metadata_bytes_written,
        planned_payload_bytes=accounting.payload_bytes_planned,
        projection_transform_id=manifest.transform_id,
        created_by=created_by,
    )


def native_loader_status(fmt: str) -> dict[str, Any]:
    """Return lazy optional-dependency status for a native dataset target."""
    if fmt == LEROBOT_FORMAT:
        modules = ("lerobot",)
        install = "lancedb-robotics[lerobot]"
    elif fmt == RLDS_FORMAT:
        modules = ("rlds", "tensorflow", "tensorflow_datasets", "reverb")
        install = "lancedb-robotics[rlds]"
    elif fmt == WEBDATASET_FORMAT:
        modules = ("webdataset",)
        install = "lancedb-robotics[webdataset]"
    else:
        raise DatasetExportError(
            f"unsupported dataset export format {fmt!r}; choose from "
            f"{', '.join(DATASET_EXPORT_FORMATS)}"
        )

    missing = [module for module in modules if not _module_available(module)]
    return {
        "available": not missing,
        "modules": list(modules),
        "missing": missing,
        "install": install,
    }


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _latest_snapshot_row(lake: Lake, name: str) -> dict[str, Any]:
    rows = [
        row for row in lake.table("dataset_snapshots").to_arrow().to_pylist() if row["name"] == name
    ]
    if not rows:
        raise DatasetExportError(f"no snapshot named {name!r} in {lake.uri}")
    return max(rows, key=lambda row: (row["created_at"], row["dataset_id"]))


def _table_rows_as_of(lake: Lake, table_name: str, version: int | None) -> list[dict[str, Any]]:
    table = lake.table(table_name)
    if version is not None:
        table.checkout(int(version))
    try:
        return table.to_arrow().to_pylist()
    finally:
        if version is not None:
            table.checkout_latest()


def _scan_table_rows_as_of(
    lake: Lake,
    table_name: str,
    version: int | None,
    *,
    columns: Sequence[str],
    where: str | None = None,
    batch_size: int = 4096,
) -> list[dict[str, Any]]:
    """Projected, streamed read of ``table_name`` as of ``version``.

    The bounded-memory counterpart to :func:`_table_rows_as_of`: it selects only
    ``columns`` -- so a Lance blob-encoded column (``observations.payload_blob``,
    ``video_encodings.data``) is never read unless explicitly named -- and streams
    record batches instead of materializing the whole table into Arrow + Python at
    once. Used by the native training context to avoid the BUG-06 full-corpus,
    all-columns observation read.
    """
    table = lake.table(table_name)
    if version is not None:
        table.checkout(int(version))
    try:
        query = table.search().select(list(columns))
        if where:
            query = query.where(where)
        rows: list[dict[str, Any]] = []
        for batch in query.to_batches(batch_size=batch_size):
            rows.extend(batch.to_pylist())
        return rows
    finally:
        if version is not None:
            table.checkout_latest()


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in_predicate(column: str, values: Sequence[str]) -> str:
    unique = tuple(dict.fromkeys(str(value) for value in values if str(value)))
    if not unique:
        return "FALSE"
    if len(unique) == 1:
        return f"{column} = {_sql_literal(unique[0])}"
    return f"{column} IN ({', '.join(_sql_literal(value) for value in unique)})"


def _observation_metadata_columns() -> list[str]:
    """Every ``observations`` column except the ``payload_blob`` blob column."""
    return [field.name for field in OBSERVATIONS_SCHEMA if field.name != PAYLOAD_BLOB_COLUMN]


def _observation_in_any_scenario_window(
    observation: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> bool:
    run_id = observation.get("run_id")
    timestamp_ns = observation.get("timestamp_ns")
    if run_id is None or timestamp_ns is None:
        return False
    ts = int(timestamp_ns)
    return any(
        scenario.get("run_id") == run_id
        and int(scenario["start_time_ns"]) <= ts <= int(scenario["end_time_ns"])
        for scenario in scenarios
    )


#: Lance virtual row-id column surfaced by ``with_row_id`` scans (see ``blob.py``).
_ROW_ID = "_rowid"
#: Row ids per ``take_row_ids`` call -- random access is cheap, this just caps the
#: Arrow batch the take materializes at once.
_TAKE_BATCH_ROWS = 16_384


def _rows_by_id_take(
    table: Any,
    *,
    columns: Sequence[str],
    id_column: str,
    id_values: Sequence[Any],
    batch_size: int = _TAKE_BATCH_ROWS,
) -> list[dict[str, Any]]:
    """Fetch rows whose ``id_column`` is one of ``id_values`` by Lance random access.

    The bounded alternative to an ``id_column IN (<ids>)`` scan: one projected
    ``id_column`` scan (``with_row_id``) maps the logical ids to Lance row ids, then
    ``take_row_ids`` materializes only the referenced rows, projected to ``columns``.
    This is the BUG-06 fix -- a snapshot pinning ~178K observation ids otherwise
    builds a multi-MB ``observation_id IN (...)`` predicate that blows up the query
    planner natively (a giant ``IN`` is the same whole-corpus anti-pattern as an
    unprojected ``to_pylist``). Work is proportional to the referenced set, not the
    corpus, and a blob column projects as its cheap ``{position, size}`` descriptor
    (no bytes) -- exactly like ``blob.py``'s by-id blob fetch.

    ``table`` is a lancedb ``Table`` already checked out at the desired version; row
    ids are version-relative, so the id-scan and the take must see the same version.
    """
    wanted = list(
        dict.fromkeys(str(value) for value in id_values if value is not None and str(value))
    )
    if not wanted:
        return []
    select_columns = list(dict.fromkeys([*columns, id_column]))
    index = table.search().select([id_column]).with_row_id(True).to_arrow()
    rowid_by_id = dict(
        zip(index[id_column].to_pylist(), index[_ROW_ID].to_pylist(), strict=True)
    )
    row_ids = [rowid_by_id[value] for value in wanted if value in rowid_by_id]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(row_ids), batch_size):
        chunk = row_ids[start : start + batch_size]
        if not chunk:
            continue
        batch = table.take_row_ids(chunk).select(select_columns).to_arrow()
        rows.extend(batch.to_pylist())
    return rows


def _scoped_observation_rows(
    lake: Lake,
    version: int | None,
    scenario_rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Read only the observations the snapshot's scenarios reference (BUG-06).

    Scenarios that pin explicit ``observation_ids`` are fetched by Lance random
    access (``_rows_by_id_take`` -- never an ``observation_id IN (<ids>)`` predicate,
    which blows up the planner for the ~178K-id snapshots real curation produces).
    Scenarios that window a run are fetched by ``run_id`` (a small ``IN`` over the
    handful of runs) and filtered to the scenario's time window while streaming.
    Both project ``columns`` (a blob column comes through only as its cheap
    ``{position, size}`` descriptor), so the context holds only the snapshot's
    observation metadata -- not the whole corpus, and no blob bytes.
    """
    scan_columns = list(
        dict.fromkeys([*columns, "observation_id", "run_id", "timestamp_ns"])
    )
    by_id: dict[str, dict[str, Any]] = {}

    requested_ids = sorted(
        {
            str(observation_id)
            for scenario in scenario_rows
            for observation_id in (scenario.get("observation_ids") or [])
            if observation_id
        }
    )
    windowed_scenarios = [
        scenario for scenario in scenario_rows if not (scenario.get("observation_ids") or [])
    ]

    table = lake.table("observations")
    if version is not None:
        table.checkout(int(version))
    try:
        if requested_ids:
            for row in _rows_by_id_take(
                table,
                columns=scan_columns,
                id_column="observation_id",
                id_values=requested_ids,
            ):
                by_id[row["observation_id"]] = row

        if windowed_scenarios:
            run_ids = sorted({str(scenario["run_id"]) for scenario in windowed_scenarios})
            query = table.search().select(scan_columns).where(
                _sql_in_predicate("run_id", run_ids)
            )
            for batch in query.to_batches(batch_size=4096):
                for row in batch.to_pylist():
                    if _observation_in_any_scenario_window(row, windowed_scenarios):
                        by_id[row["observation_id"]] = row
    finally:
        if version is not None:
            table.checkout_latest()

    return list(by_id.values())


def _payload_blobs_as_of(
    lake: Lake,
    version: int | None,
    observation_ids: list[str],
) -> dict[str, bytes]:
    if not observation_ids:
        return {}
    table = lake.table("observations")
    if version is not None:
        table.checkout(int(version))
    try:
        return fetch_blobs(
            table,
            PAYLOAD_BLOB_COLUMN,
            observation_ids,
            id_column="observation_id",
        )
    finally:
        if version is not None:
            table.checkout_latest()


def _video_encoding_blobs_as_of(
    lake: Lake,
    version: int | None,
    encoding_ids: list[str],
) -> dict[str, bytes]:
    if not encoding_ids:
        return {}
    table = lake.table("video_encodings")
    if version is not None:
        table.checkout(int(version))
    try:
        return fetch_blobs(
            table,
            VIDEO_ENCODING_BLOB_COLUMN,
            encoding_ids,
            id_column="encoding_id",
        )
    finally:
        if version is not None:
            table.checkout_latest()


def _snapshot_context(
    lake: Lake,
    snapshot_name: str,
    *,
    include_payload_blobs: bool = True,
    include_video_encoding_blobs: bool = True,
    observation_columns: Sequence[str] | None = None,
    scope_observations_to_scenarios: bool = False,
) -> _SnapshotContext:
    """Build a snapshot context from the pinned table versions.

    By default (the export path) every table is read whole. The native training
    context opts into bounded reads via ``observation_columns`` (project only these
    observation columns; ``None`` means all non-blob columns) and
    ``scope_observations_to_scenarios`` (read only the observations the snapshot's
    scenarios reference, streamed, not the whole corpus eagerly materialized to
    Python) -- this is the BUG-06 fix. Lance blob columns are scanned as cheap
    ``{position, size}`` descriptors regardless, so projecting ``payload_blob``
    reads no blob bytes.
    """
    row = _latest_snapshot_row(lake, snapshot_name)
    query_spec = json.loads(row["query_spec"] or "{}")
    split_payload = json.loads(row["split"] or "{}")
    versions = {tv["table"]: int(tv["version"]) for tv in row["table_versions"]}

    scenario_ids = tuple(sorted(query_spec.get("scenario_ids", [])))
    scenario_rows = _table_rows_as_of(lake, "scenarios", versions.get("scenarios"))
    episode_rows = (
        _table_rows_as_of(lake, "episodes", versions["episodes"])
        if "episodes" in versions
        else []
    )
    video_rows = _table_rows_as_of(lake, "videos", versions.get("videos"))
    video_encoding_rows = (
        _table_rows_as_of(lake, "video_encodings", versions["video_encodings"])
        if "video_encodings" in versions
        else []
    )
    if scope_observations_to_scenarios:
        selected_ids = set(scenario_ids)
        selected_scenarios = [
            scenario for scenario in scenario_rows if scenario.get("scenario_id") in selected_ids
        ]
        observation_rows = _scoped_observation_rows(
            lake,
            versions.get("observations"),
            selected_scenarios,
            columns=observation_columns or _observation_metadata_columns(),
        )
    elif observation_columns is not None:
        observation_rows = _scan_table_rows_as_of(
            lake, "observations", versions.get("observations"), columns=observation_columns
        )
    else:
        observation_rows = _table_rows_as_of(lake, "observations", versions.get("observations"))
    run_rows = _table_rows_as_of(lake, "runs", versions.get("runs"))

    scenarios = {scenario["scenario_id"]: scenario for scenario in scenario_rows}
    episodes = {episode["episode_id"]: episode for episode in episode_rows}
    videos = {video["video_id"]: video for video in video_rows}
    video_encodings = {row["encoding_id"]: row for row in video_encoding_rows}
    observations = {obs["observation_id"]: obs for obs in observation_rows}
    runs = {run["run_id"]: run for run in run_rows}

    missing_scenarios = [sid for sid in scenario_ids if sid not in scenarios]
    if missing_scenarios:
        raise DatasetExportError(
            f"snapshot {snapshot_name!r} references scenarios not present at the "
            f"pinned version: {missing_scenarios}"
        )

    camera_ids = [
        obs["observation_id"] for obs in observations.values() if _is_camera_observation(obs)
    ]
    payload_blobs = (
        _payload_blobs_as_of(lake, versions.get("observations"), camera_ids)
        if include_payload_blobs
        else {}
    )
    video_encoding_blobs = _video_encoding_blobs_as_of(
        lake,
        versions.get("video_encodings"),
        list(video_encodings),
    ) if include_video_encoding_blobs else {}

    return _SnapshotContext(
        row=row,
        dataset_id=row["dataset_id"],
        snapshot_name=row["name"],
        scenario_ids=scenario_ids,
        split_assignments=dict(split_payload.get("assignments", {})),
        table_versions=tuple(
            {
                "table": tv["table"],
                "version": int(tv["version"]),
                "tag": tv.get("tag") or "",
            }
            for tv in row["table_versions"]
        ),
        scenarios=scenarios,
        episodes=episodes,
        observations=observations,
        videos=videos,
        video_encodings=video_encodings,
        runs=runs,
        payload_blobs=payload_blobs,
        video_encoding_blobs=video_encoding_blobs,
    )


def _episodes(context: _SnapshotContext) -> tuple[_Episode, ...]:
    physical = _physical_episodes(context)
    if physical:
        return physical

    task_by_description: dict[str, int] = {}
    prepared: list[tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...], str]] = []

    for scenario_id in context.scenario_ids:
        scenario = context.scenarios[scenario_id]
        run = context.runs.get(scenario["run_id"], {})
        observations = _observations_for_scenario(context, scenario)
        task = _task_description(scenario, run)
        prepared.append((scenario, run, observations, task))
        task_by_description.setdefault(task, 0)

    for index, task in enumerate(sorted(task_by_description)):
        task_by_description[task] = index

    return tuple(
        _Episode(
            episode_id=scenario["scenario_id"],
            index=index,
            scenario=scenario,
            physical_episode=None,
            run=run,
            observations=observations,
            split=context.split_assignments.get(scenario["scenario_id"], "train"),
            task_index=task_by_description[task],
            task=task,
        )
        for index, (scenario, run, observations, task) in enumerate(prepared)
    )


def _physical_episodes(context: _SnapshotContext) -> tuple[_Episode, ...]:
    if not context.episodes:
        return ()

    selected: dict[str, str] = {}
    for scenario_id in context.scenario_ids:
        scenario = context.scenarios[scenario_id]
        for obs in _observations_for_scenario(context, scenario):
            episode_id = obs.get("episode_id")
            if episode_id and episode_id in context.episodes:
                selected.setdefault(episode_id, scenario_id)

    if not selected:
        return ()

    task_by_description: dict[str, int] = {}
    prepared: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...], str, str]
    ] = []
    for episode_id in sorted(
        selected,
        key=lambda eid: (
            int(context.episodes[eid].get("episode_index") or 0),
            context.episodes[eid]["episode_id"],
        ),
    ):
        physical = context.episodes[episode_id]
        source_scenario = context.scenarios[selected[episode_id]]
        run = context.runs.get(physical["run_id"], {})
        scenario_like = {
            **source_scenario,
            "run_id": physical["run_id"],
            "start_time_ns": physical["from_timestamp_ns"],
            "end_time_ns": physical["to_timestamp_ns"],
            "summary": source_scenario.get("summary") or physical.get("task_id"),
            "scenario_type": "episode",
        }
        observations = _observations_for_physical_episode(context, physical)
        task = _task_description(scenario_like, run, physical)
        prepared.append((physical, scenario_like, run, observations, task, selected[episode_id]))
        task_by_description.setdefault(task, 0)

    for index, task in enumerate(sorted(task_by_description)):
        task_by_description[task] = index

    return tuple(
        _Episode(
            episode_id=physical["episode_id"],
            index=index,
            scenario=scenario,
            physical_episode=physical,
            run=run,
            observations=observations,
            split=context.split_assignments.get(source_scenario_id, "train"),
            task_index=task_by_description[task],
            task=task,
        )
        for index, (physical, scenario, run, observations, task, source_scenario_id) in enumerate(
            prepared
        )
    )


def _observations_for_scenario(
    context: _SnapshotContext,
    scenario: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    requested = list(scenario.get("observation_ids") or [])
    if requested:
        missing = [obs_id for obs_id in requested if obs_id not in context.observations]
        if missing:
            raise DatasetExportError(
                f"scenario {scenario['scenario_id']} references observations not "
                f"present at the pinned version: {missing}"
            )
        rows = [context.observations[obs_id] for obs_id in requested]
    else:
        rows = [
            obs
            for obs in context.observations.values()
            if obs["run_id"] == scenario["run_id"]
            and scenario["start_time_ns"] <= obs["timestamp_ns"] <= scenario["end_time_ns"]
        ]
    return tuple(
        sorted(
            rows,
            key=lambda obs: (
                int(obs["timestamp_ns"]),
                str(obs.get("topic") or ""),
                str(obs["observation_id"]),
            ),
        )
    )


def _observations_for_physical_episode(
    context: _SnapshotContext,
    episode: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    rows = [
        obs for obs in context.observations.values() if obs.get("episode_id") == episode["episode_id"]
    ]
    if not rows:
        rows = [
            obs
            for obs in context.observations.values()
            if obs["run_id"] == episode["run_id"]
            and episode["from_timestamp_ns"] <= obs["timestamp_ns"] <= episode["to_timestamp_ns"]
        ]
    return tuple(sorted(rows, key=_observation_sort_key))


def _write_lerobot(
    root: Path,
    context: _SnapshotContext,
    episodes: tuple[_Episode, ...],
) -> tuple[list[str], dict[str, Any], list[str], int]:
    files: list[str] = []
    frame_count = sum(len(episode.observations) for episode in episodes)
    camera_topics = _camera_topics(episodes)
    feature_spec, lossy_mapping = _feature_spec(
        context,
        episodes,
        camera_topics,
        format=LEROBOT_FORMAT,
        step_count=frame_count,
    )

    tasks = _task_rows(episodes)
    files.append(_write_lerobot_tasks(root, tasks))
    files.append(_write_jsonl(root, "meta/tasks.jsonl", tasks))
    files.append(
        _write_json(
            root,
            "meta/info.json",
            _lerobot_info(context, episodes, feature_spec, frame_count, len(tasks)),
        )
    )
    files.append(_write_json(root, "meta/stats.json", _stats(episodes)))
    episode_rows = _lerobot_episode_metadata_rows(episodes, camera_topics)
    files.append(
        _write_parquet(
            root,
            f"meta/episodes/{_CHUNK}/{_FILE}.parquet",
            episode_rows,
            _lerobot_episode_metadata_schema(camera_topics),
        )
    )
    files.append(_write_jsonl(root, "meta/episodes.jsonl", _episode_rows(episodes)))

    frame_rows: list[dict[str, Any]] = []
    for episode in episodes:
        rows, image_files = _lerobot_frame_rows(root, context, episode, camera_topics)
        frame_rows.extend(rows)
        files.extend(image_files)
    files.append(
        _write_parquet(
            root,
            f"data/{_CHUNK}/{_FILE}.parquet",
            frame_rows,
            _lerobot_schema(camera_topics),
        )
    )

    return files, feature_spec, lossy_mapping, frame_count


def _write_rlds(
    root: Path,
    context: _SnapshotContext,
    episodes: tuple[_Episode, ...],
) -> tuple[list[str], dict[str, Any], list[str], int]:
    files: list[str] = []
    step_count = sum(len(episode.observations) for episode in episodes)
    camera_topics = _camera_topics(episodes)
    feature_spec, lossy_mapping = _feature_spec(
        context,
        episodes,
        camera_topics,
        format=RLDS_FORMAT,
        step_count=step_count,
    )

    files.append(
        _write_json(
            root,
            "dataset_info.json",
            {
                "dataset_id": context.dataset_id,
                "format": RLDS_FORMAT,
                "format_version": RLDS_FORMAT_VERSION,
                "description": "RLDS-style episode/step projection from a LanceDB Robotics snapshot.",
                "features": feature_spec["features"],
                "lossy_mapping": lossy_mapping,
                "total_episodes": len(episodes),
                "total_steps": step_count,
            },
        )
    )
    files.append(_write_json(root, "features.json", feature_spec))
    files.append(_write_jsonl(root, "episodes.jsonl", _episode_rows(episodes)))

    for episode in episodes:
        rows, image_files = _rlds_step_rows(root, context, episode, camera_topics)
        files.extend(image_files)
        rel = f"episodes/episode_{episode.index:06d}/steps.parquet"
        schema = _rlds_schema(camera_topics)
        files.append(_write_parquet(root, rel, rows, schema))

    return files, feature_spec, lossy_mapping, step_count


def _write_webdataset(
    root: Path,
    context: _SnapshotContext,
    episodes: tuple[_Episode, ...],
    *,
    shard_size: int,
    compression: str,
) -> tuple[list[str], dict[str, Any], list[str], int]:
    files: list[str] = []
    step_count = sum(len(episode.observations) for episode in episodes)
    camera_topics = _camera_topics(episodes)
    feature_spec, lossy_mapping = _feature_spec(
        context,
        episodes,
        camera_topics,
        format=WEBDATASET_FORMAT,
        step_count=step_count,
        shard_size=shard_size,
        compression=compression,
    )

    samples = _webdataset_sample_rows(context, episodes, camera_topics)
    shard_plan = _webdataset_shard_plan(
        context,
        samples,
        shard_size=shard_size,
        compression=compression,
    )
    files.append(
        _write_json(
            root,
            "webdataset_plan.json",
            {
                "format": WEBDATASET_FORMAT,
                "format_version": WEBDATASET_FORMAT_VERSION,
                "dataset_id": context.dataset_id,
                "snapshot_name": context.snapshot_name,
                "shard_size": shard_size,
                "compression": compression,
                "sample_schema": _webdataset_sample_schema(camera_topics),
                "estimated_bytes": sum(item["estimated_bytes"] for item in shard_plan),
                "shards": shard_plan,
            },
        )
    )

    for shard in shard_plan:
        shard_samples = samples[shard["sample_from"] : shard["sample_to"]]
        files.append(
            _write_webdataset_shard(
                root,
                shard["path"],
                shard_samples,
                compression=compression,
            )
        )

    return files, feature_spec, lossy_mapping, step_count


def _lerobot_frame_rows(
    root: Path,
    context: _SnapshotContext,
    episode: _Episode,
    camera_topics: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    image_files: list[str] = []
    for frame_index, obs in enumerate(episode.observations):
        camera_paths, written = _camera_paths(root, context, episode, frame_index, obs)
        image_files.extend(written)
        row = {
            "index": _global_index(episode, frame_index),
            "episode_index": episode.index,
            "frame_index": frame_index,
            "timestamp": _relative_seconds(obs["timestamp_ns"], episode.scenario["start_time_ns"]),
            "timestamp_ns": int(obs["timestamp_ns"]),
            "task_index": episode.task_index,
            "observation.state": _vector(obs.get("state_vector")),
            "action": _vector(obs.get("action_vector")),
            "task": episode.task,
            "language_instruction": obs.get("caption") or episode.task,
            "observation_id": obs["observation_id"],
            "episode_id": episode.episode_id,
            "scenario_id": episode.scenario["scenario_id"],
            "run_id": episode.scenario["run_id"],
            "topic": obs.get("topic"),
            "modality": obs.get("modality"),
            "caption": obs.get("caption") or episode.task,
        }
        for camera_key in camera_topics:
            path = camera_paths.get(camera_key)
            row[f"observation.images.{camera_key}"] = (
                {"bytes": None, "path": path} if path else None
            )
        rows.append(row)
    return rows, image_files


def _rlds_step_rows(
    root: Path,
    context: _SnapshotContext,
    episode: _Episode,
    camera_topics: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    image_files: list[str] = []
    last_index = len(episode.observations) - 1
    for step_index, obs in enumerate(episode.observations):
        camera_paths, written = _camera_paths(root, context, episode, step_index, obs)
        image_files.extend(written)
        is_last = step_index == last_index
        row = {
            "episode_id": episode.episode_id,
            "episode_index": episode.index,
            "step_index": step_index,
            "timestamp_ns": int(obs["timestamp_ns"]),
            "relative_time_s": _relative_seconds(
                obs["timestamp_ns"],
                episode.scenario["start_time_ns"],
            ),
            "observation.state": _vector(obs.get("state_vector")),
            "observation.caption": obs.get("caption") or episode.task,
            "observation.payload_json": obs.get("payload_json"),
            "observation.sensor_id": obs.get("sensor_id"),
            "action": _vector(obs.get("action_vector")),
            "reward": 0.0,
            "discount": 0.0 if is_last else 1.0,
            "is_first": step_index == 0,
            "is_last": is_last,
            "is_terminal": is_last,
            "observation_id": obs["observation_id"],
            "scenario_id": episode.scenario["scenario_id"],
            "run_id": episode.scenario["run_id"],
            "split": episode.split,
            "task_index": episode.task_index,
            "task": episode.task,
            "topic": obs.get("topic"),
            "modality": obs.get("modality"),
            "lineage.raw_uri": obs.get("raw_uri"),
            "lineage.raw_channel": obs.get("raw_channel"),
            "lineage.raw_sequence": obs.get("raw_sequence"),
            "lineage.table_versions_json": json.dumps(
                list(context.table_versions),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for camera_key in camera_topics:
            row[f"observation.image.{camera_key}"] = camera_paths.get(camera_key)
        rows.append(row)
    return rows, image_files


def _webdataset_sample_rows(
    context: _SnapshotContext,
    episodes: tuple[_Episode, ...],
    camera_topics: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for frame_index, obs in enumerate(episode.observations):
            key = _webdataset_sample_key(episode, frame_index, obs)
            payload = (
                context.payload_blobs.get(obs["observation_id"], b"")
                if _is_camera_observation(obs)
                else b""
            )
            media_ext = _webdataset_media_extension(obs) if payload else None
            metadata = _webdataset_metadata(
                context,
                episode,
                frame_index,
                obs,
                key=key,
                media_ext=media_ext,
                payload_size=len(payload),
            )
            rows.append(
                {
                    "key": key,
                    "json": json.dumps(
                        metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode(),
                    "caption": (metadata["caption"] or metadata["task"]).encode(),
                    "media_ext": media_ext,
                    "media": payload,
                    "estimated_bytes": _webdataset_estimated_sample_bytes(
                        metadata,
                        payload,
                    ),
                }
            )
    return rows


def _webdataset_metadata(
    context: _SnapshotContext,
    episode: _Episode,
    frame_index: int,
    obs: dict[str, Any],
    *,
    key: str,
    media_ext: str | None,
    payload_size: int,
) -> dict[str, Any]:
    caption = obs.get("caption") or episode.task
    return {
        "__key__": key,
        "sample_id": obs["observation_id"],
        "observation_id": obs["observation_id"],
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario["scenario_id"],
        "run_id": episode.scenario["run_id"],
        "split": episode.split,
        "episode_index": episode.index,
        "frame_index": frame_index,
        "timestamp_ns": int(obs["timestamp_ns"]),
        "relative_time_s": _relative_seconds(
            obs["timestamp_ns"],
            episode.scenario["start_time_ns"],
        ),
        "sensor_id": obs.get("sensor_id"),
        "topic": obs.get("topic"),
        "modality": obs.get("modality"),
        "state_vector": _vector(obs.get("state_vector")),
        "action_vector": _vector(obs.get("action_vector")),
        "caption": caption,
        "task": episode.task,
        "task_index": episode.task_index,
        "media": {
            "key": media_ext,
            "camera_key": _camera_key(obs) if _is_camera_observation(obs) else None,
            "source": "observations.payload_blob" if media_ext else None,
            "payload_bytes": payload_size,
        },
        "lineage": {
            "dataset_id": context.dataset_id,
            "snapshot_name": context.snapshot_name,
            "table_versions": list(context.table_versions),
            "raw_uri": obs.get("raw_uri"),
            "raw_channel": obs.get("raw_channel"),
            "raw_sequence": obs.get("raw_sequence"),
        },
    }


def _webdataset_sample_key(
    episode: _Episode,
    frame_index: int,
    obs: dict[str, Any],
) -> str:
    return _webdataset_sample_key_from_parts(
        episode_index=episode.index,
        frame_index=frame_index,
        observation_id=obs["observation_id"],
    )


def _webdataset_sample_key_from_parts(
    *,
    episode_index: int,
    frame_index: int,
    observation_id: str,
) -> str:
    return (
        f"episode-{int(episode_index):06d}/"
        f"frame-{int(frame_index):06d}-{_safe_key_component(observation_id)}"
    )


def _safe_key_component(value: Any) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return key or "sample"


def _webdataset_media_extension(obs: dict[str, Any]) -> str:
    encoding = str(
        obs.get("message_encoding")
        or obs.get("schema_encoding")
        or obs.get("modality")
        or ""
    ).lower()
    if "jpeg" in encoding or "jpg" in encoding:
        return "jpg"
    if "png" in encoding:
        return "png"
    if "webp" in encoding:
        return "webp"
    return "bin"


def _webdataset_estimated_sample_bytes(metadata: dict[str, Any], payload: bytes) -> int:
    return (
        len(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        + len((metadata["caption"] or metadata["task"]).encode())
        + len(payload or b"")
    )


def _webdataset_shard_plan(
    context: _SnapshotContext,
    samples: list[dict[str, Any]],
    *,
    shard_size: int,
    compression: str,
) -> list[dict[str, Any]]:
    suffix = ".tar.gz" if compression == "gzip" else ".tar"
    plan: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(samples), shard_size)):
        stop = min(start + shard_size, len(samples))
        shard_samples = samples[start:stop]
        first_key = shard_samples[0]["key"] if shard_samples else ""
        last_key = shard_samples[-1]["key"] if shard_samples else ""
        plan.append(
            {
                "path": f"shards/shard-{shard_index:06d}{suffix}",
                "shard_index": shard_index,
                "sample_from": start,
                "sample_to": stop,
                "sample_count": stop - start,
                "first_key": first_key,
                "last_key": last_key,
                "estimated_bytes": sum(int(sample["estimated_bytes"]) for sample in shard_samples),
                "compression": compression,
                "dataset_id": context.dataset_id,
            }
        )
    return plan


def _webdataset_sample_schema(camera_topics: dict[str, str]) -> dict[str, Any]:
    media_keys = [
        {"key": camera_key, "topic": topic, "extensions": ["jpg", "png", "webp", "bin"]}
        for camera_key, topic in camera_topics.items()
    ]
    return {
        "__key__": "deterministic episode/frame/observation key",
        "json": {
            "sample_id": "observation_id",
            "state_vector": "float32 list",
            "action_vector": "float32 list",
            "caption": "string",
            "task": "string",
            "lineage": "dataset snapshot/table versions/raw source",
        },
        "txt": "caption/task text",
        "media": media_keys,
    }


def _write_webdataset_shard(
    root: Path,
    rel: str,
    samples: list[dict[str, Any]],
    *,
    compression: str,
) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "gzip":
        with path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tar:
                    _write_webdataset_tar_members(tar, samples)
    else:
        with tarfile.open(path, mode="w") as tar:
            _write_webdataset_tar_members(tar, samples)
    return rel


def _write_webdataset_tar_members(
    tar: tarfile.TarFile,
    samples: list[dict[str, Any]],
) -> None:
    for sample in samples:
        key = sample["key"]
        _add_tar_bytes(tar, f"{key}.json", sample["json"])
        _add_tar_bytes(tar, f"{key}.txt", sample["caption"])
        if sample["media_ext"] and sample["media"]:
            _add_tar_bytes(tar, f"{key}.{sample['media_ext']}", sample["media"])


def _add_tar_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def _normalize_webdataset_shard_size(value: int) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise DatasetExportError("webdataset shard_size must be positive")
    return parsed


def _normalize_webdataset_compression(value: str) -> str:
    normalized = str(value).lower()
    if normalized == "gz":
        normalized = "gzip"
    if normalized not in WEBDATASET_COMPRESSIONS:
        raise DatasetExportError(
            f"unsupported webdataset compression {value!r}; choose from "
            f"{', '.join(WEBDATASET_COMPRESSIONS)}"
        )
    return normalized


def _camera_paths(
    root: Path,
    context: _SnapshotContext,
    episode: _Episode,
    frame_index: int,
    obs: dict[str, Any],
) -> tuple[dict[str, str | None], list[str]]:
    if not _is_camera_observation(obs):
        return {}, []

    payload = context.payload_blobs.get(obs["observation_id"], b"")
    camera_key = _camera_key(obs)
    if not payload:
        return {camera_key: None}, []

    rel = (
        f"images/{camera_key}/episode_{episode.index:06d}/"
        f"frame_{frame_index:06d}.bin"
    )
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {camera_key: rel}, [rel]


def _lerobot_schema(camera_topics: dict[str, str]) -> pa.Schema:
    fields = [
        pa.field("index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("timestamp", pa.float32()),
        pa.field("timestamp_ns", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("observation.state", pa.list_(pa.float32())),
        pa.field("action", pa.list_(pa.float32())),
        pa.field("task", pa.string()),
        pa.field("language_instruction", pa.string()),
        pa.field("observation_id", pa.string()),
        pa.field("episode_id", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("modality", pa.string()),
        pa.field("caption", pa.string()),
    ]
    image_ref = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    fields.extend(pa.field(f"observation.images.{key}", image_ref) for key in camera_topics)
    return pa.schema(fields)


def _lerobot_info(
    context: _SnapshotContext,
    episodes: tuple[_Episode, ...],
    feature_spec: dict[str, Any],
    frame_count: int,
    task_count: int,
) -> dict[str, Any]:
    fps = _infer_fps(episodes)
    return {
        "codebase_version": LEROBOT_FORMAT_VERSION.removeprefix("lerobot-"),
        "dataset_id": context.dataset_id,
        "format": LEROBOT_FORMAT,
        "fps": int(round(fps or 1)),
        "features": _lerobot_format_features(feature_spec),
        "total_episodes": len(episodes),
        "total_frames": frame_count,
        "total_tasks": task_count,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "splits": {"train": f"0:{len(episodes)}"},
        "robot_type": None,
        "source_lake_dataset_id": context.dataset_id,
    }


def _lerobot_format_features(feature_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = feature_spec["features"]
    state_shape = list(features["observation.state"]["shape"])
    action_shape = list(features["action"]["shape"])
    projected: dict[str, dict[str, Any]] = {
        "index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "timestamp_ns": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "observation.state": {"dtype": "float32", "shape": state_shape},
        "action": {"dtype": "float32", "shape": action_shape},
        "task": {"dtype": "string", "shape": [1]},
        "language_instruction": {"dtype": "string", "shape": [1]},
        "observation_id": {"dtype": "string", "shape": [1]},
        "episode_id": {"dtype": "string", "shape": [1]},
        "scenario_id": {"dtype": "string", "shape": [1]},
        "run_id": {"dtype": "string", "shape": [1]},
        "topic": {"dtype": "string", "shape": [1]},
        "modality": {"dtype": "string", "shape": [1]},
        "caption": {"dtype": "string", "shape": [1]},
    }
    for camera in features.get("cameras", ()):
        projected[f"observation.images.{camera['key']}"] = {
            "dtype": "image",
            "shape": [3, 0, 0],
            "source": camera.get("source", "observations.payload_blob"),
            "topic": camera.get("topic", ""),
        }
    return projected


def _lerobot_episode_metadata_rows(
    episodes: tuple[_Episode, ...],
    camera_topics: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for episode in episodes:
        length = len(episode.observations)
        row: dict[str, Any] = {
            "episode_index": episode.index,
            "tasks": [episode.task],
            "length": length,
            "dataset_from_index": offset,
            "dataset_to_index": offset + length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "episode_id": episode.episode_id,
            "scenario_id": episode.scenario["scenario_id"],
            "run_id": episode.scenario["run_id"],
            "split": episode.split,
        }
        for camera_key in camera_topics:
            row[f"images/{camera_key}/from_frame"] = offset
            row[f"images/{camera_key}/to_frame"] = offset + length
        rows.append(row)
        offset += length
    return rows


def _lerobot_episode_metadata_schema(camera_topics: dict[str, str]) -> pa.Schema:
    fields = [
        pa.field("episode_index", pa.int64()),
        pa.field("tasks", pa.list_(pa.string())),
        pa.field("length", pa.int64()),
        pa.field("dataset_from_index", pa.int64()),
        pa.field("dataset_to_index", pa.int64()),
        pa.field("data/chunk_index", pa.int64()),
        pa.field("data/file_index", pa.int64()),
        pa.field("meta/episodes/chunk_index", pa.int64()),
        pa.field("meta/episodes/file_index", pa.int64()),
        pa.field("episode_id", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("split", pa.string()),
    ]
    for camera_key in camera_topics:
        fields.append(pa.field(f"images/{camera_key}/from_frame", pa.int64()))
        fields.append(pa.field(f"images/{camera_key}/to_frame", pa.int64()))
    return pa.schema(fields)


def _rlds_schema(camera_topics: dict[str, str]) -> pa.Schema:
    fields = [
        pa.field("episode_id", pa.string()),
        pa.field("episode_index", pa.int64()),
        pa.field("step_index", pa.int64()),
        pa.field("timestamp_ns", pa.int64()),
        pa.field("relative_time_s", pa.float32()),
        pa.field("observation.state", pa.list_(pa.float32())),
        pa.field("observation.caption", pa.string()),
        pa.field("observation.payload_json", pa.string()),
        pa.field("observation.sensor_id", pa.string()),
        pa.field("action", pa.list_(pa.float32())),
        pa.field("reward", pa.float32()),
        pa.field("discount", pa.float32()),
        pa.field("is_first", pa.bool_()),
        pa.field("is_last", pa.bool_()),
        pa.field("is_terminal", pa.bool_()),
        pa.field("observation_id", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("split", pa.string()),
        pa.field("task_index", pa.int64()),
        pa.field("task", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("modality", pa.string()),
        pa.field("lineage.raw_uri", pa.string()),
        pa.field("lineage.raw_channel", pa.string()),
        pa.field("lineage.raw_sequence", pa.int64()),
        pa.field("lineage.table_versions_json", pa.string()),
    ]
    fields.extend(pa.field(f"observation.image.{key}", pa.string()) for key in camera_topics)
    return pa.schema(fields)


def _feature_spec(
    context: _SnapshotContext,
    episodes: tuple[_Episode, ...],
    camera_topics: dict[str, str],
    *,
    format: str,
    step_count: int,
    shard_size: int | None = None,
    compression: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    state_shape = _vector_shape(obs.get("state_vector") for episode in episodes for obs in episode.observations)
    action_shape = _vector_shape(
        obs.get("action_vector") for episode in episodes for obs in episode.observations
    )
    camera_features = [
        {
            "key": camera_key,
            "topic": topic,
            "dtype": "path",
            "source": "observations.payload_blob",
        }
        for camera_key, topic in camera_topics.items()
    ]
    has_physical_episodes = any(episode.physical_episode is not None for episode in episodes)
    lossy_mapping = []
    if not has_physical_episodes:
        lossy_mapping.append(
            "scenario windows are exported as episodes when first-class episode rows are absent"
        )
    if any(not obs.get("state_vector") for episode in episodes for obs in episode.observations):
        lossy_mapping.append("some observations have no state_vector; exported state is an empty vector")
    if any(not obs.get("action_vector") for episode in episodes for obs in episode.observations):
        lossy_mapping.append("some observations have no action_vector; exported action is an empty vector")
    if camera_topics and any(
        _is_camera_observation(obs)
        and not context.payload_blobs.get(obs["observation_id"], b"")
        for episode in episodes
        for obs in episode.observations
    ):
        lossy_mapping.append(
            "some camera observations have no payload_blob bytes; exported camera path is null"
        )

    features = {
        "episode": {
            "source": "episodes" if has_physical_episodes else "scenarios",
            "count": len(episodes),
        },
        "step": {"source": "observations", "count": step_count},
        "observation.state": {
            "source": "observations.state_vector",
            "dtype": "float32",
            "shape": state_shape,
        },
        "action": {
            "source": "observations.action_vector",
            "dtype": "float32",
            "shape": action_shape,
        },
        "language": {
            "source": "observations.caption | scenarios.summary | runs.task_id",
            "dtype": "string",
        },
        "cameras": camera_features,
    }
    if format == RLDS_FORMAT:
        features["rlds"] = {
            "episode_key": "episode_metadata",
            "steps_key": "steps",
            "step_fields": [
                "observation",
                "action",
                "reward",
                "discount",
                "is_first",
                "is_last",
                "is_terminal",
                "metadata",
            ],
            "observation_keys": ["state", "caption", "payload_json", "image"],
            "metadata_keys": [
                "observation_id",
                "episode_id",
                "scenario_id",
                "run_id",
                "split",
                "episode_index",
                "frame_index",
                "timestamp_ns",
                "relative_time_s",
                "sensor_id",
                "topic",
                "modality",
                "raw_uri",
                "raw_channel",
                "raw_sequence",
            ],
        }
        features["reward"] = {
            "source": "constant",
            "dtype": "float32",
            "value": 0.0,
        }
        features["discount"] = {
            "source": "step terminal flag",
            "dtype": "float32",
            "non_terminal": 1.0,
            "terminal": 0.0,
        }
        lossy_mapping.extend(
            [
                "RLDS reward is synthesized as 0.0 because canonical observations do not store reward",
                "RLDS discount is synthesized from terminal step position",
                "robotics fields outside RLDS observation/action/reward/discount are preserved in metadata and lineage",
            ]
        )
    spec: dict[str, Any] = {
        "format": format,
        "snapshot_dataset_id": context.dataset_id,
        "features": features,
        "table_versions": list(context.table_versions),
    }
    if format == WEBDATASET_FORMAT:
        normalized_shard_size = _normalize_webdataset_shard_size(
            shard_size or DEFAULT_WEBDATASET_SHARD_SIZE
        )
        normalized_compression = _normalize_webdataset_compression(
            compression or DEFAULT_WEBDATASET_COMPRESSION
        )
        samples = _webdataset_sample_rows(context, episodes, camera_topics)
        shards = _webdataset_shard_plan(
            context,
            samples,
            shard_size=normalized_shard_size,
            compression=normalized_compression,
        )
        spec["dry_run"] = {
            "shard_count": len(shards),
            "estimated_bytes": sum(item["estimated_bytes"] for item in shards),
            "sample_count": len(samples),
            "shard_size": normalized_shard_size,
            "compression": normalized_compression,
            "sample_schema": _webdataset_sample_schema(camera_topics),
            "shards": shards,
        }
    return spec, lossy_mapping


def _camera_topics(episodes: tuple[_Episode, ...]) -> dict[str, str]:
    topics: dict[str, str] = {}
    for episode in episodes:
        for obs in episode.observations:
            if _is_camera_observation(obs):
                topics.setdefault(_camera_key(obs), obs.get("topic") or obs.get("sensor_id") or "camera")
    return dict(sorted(topics.items()))


def _task_rows(episodes: tuple[_Episode, ...]) -> list[dict[str, Any]]:
    tasks = {episode.task_index: episode.task for episode in episodes}
    return [{"task_index": index, "task": tasks[index]} for index in sorted(tasks)]


def _episode_rows(episodes: tuple[_Episode, ...]) -> list[dict[str, Any]]:
    return [
        {
            "episode_index": episode.index,
            "episode_id": episode.episode_id,
            "scenario_id": episode.scenario["scenario_id"],
            "run_id": episode.scenario["run_id"],
            "split": episode.split,
            "length": len(episode.observations),
            "task_index": episode.task_index,
            "task": episode.task,
            "start_time_ns": episode.scenario["start_time_ns"],
            "end_time_ns": episode.scenario["end_time_ns"],
        }
        for episode in episodes
    ]


def _stats(episodes: tuple[_Episode, ...]) -> dict[str, Any]:
    return {
        "observation.state": _numeric_stats(
            _vector(obs.get("state_vector")) for episode in episodes for obs in episode.observations
        ),
        "action": _numeric_stats(
            _vector(obs.get("action_vector")) for episode in episodes for obs in episode.observations
        ),
    }


def _numeric_stats(vectors: Any) -> dict[str, Any]:
    flat = [value for vector in vectors for value in vector]
    if not flat:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(flat) / len(flat)
    variance = sum((value - mean) ** 2 for value in flat) / len(flat)
    return {
        "count": len(flat),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(flat),
        "max": max(flat),
    }


def _vector_shape(vectors: Any) -> list[int | str]:
    dims = {len(_vector(vector)) for vector in vectors if _vector(vector)}
    if len(dims) == 1:
        return [next(iter(dims))]
    if dims:
        return ["variable"]
    return [0]


def _vector(value: Any) -> list[float]:
    if value is None:
        return []
    return [float(item) for item in value]


def _infer_fps(episodes: tuple[_Episode, ...]) -> float | None:
    deltas: list[int] = []
    for episode in episodes:
        timestamps = [int(obs["timestamp_ns"]) for obs in episode.observations]
        deltas.extend(b - a for a, b in zip(timestamps, timestamps[1:], strict=False) if b > a)
    if not deltas:
        return None
    median_delta_ns = sorted(deltas)[len(deltas) // 2]
    return 1_000_000_000 / median_delta_ns


def _global_index(episode: _Episode, frame_index: int) -> int:
    # Stable within this projection. Episode files carry their own frame rows, so
    # the wide gap keeps frame indices unique without scanning previous episodes.
    return episode.index * 1_000_000 + frame_index


def _relative_seconds(timestamp_ns: int, start_time_ns: int) -> float:
    return (int(timestamp_ns) - int(start_time_ns)) / 1_000_000_000


def _observation_sort_key(obs: dict[str, Any]) -> tuple[Any, ...]:
    frame_index = obs.get("frame_index")
    return (
        frame_index if frame_index is not None else 10**18,
        int(obs["timestamp_ns"]),
        str(obs.get("topic") or ""),
        str(obs["observation_id"]),
    )


def _task_description(
    scenario: dict[str, Any],
    run: dict[str, Any],
    physical_episode: dict[str, Any] | None = None,
) -> str:
    return (
        (physical_episode or {}).get("task_id")
        or scenario.get("summary")
        or run.get("task_id")
        or scenario.get("scenario_type")
        or "unknown task"
    )


def _is_camera_observation(obs: dict[str, Any]) -> bool:
    modality = str(obs.get("modality") or "").lower()
    topic = str(obs.get("topic") or "").lower()
    sensor = str(obs.get("sensor_id") or "").lower()
    return modality in {"image", "camera", "video"} or "camera" in topic or "camera" in sensor


def _camera_key(obs: dict[str, Any]) -> str:
    base = str(obs.get("topic") or obs.get("sensor_id") or "camera")
    key = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    return key or "camera"


def _write_json(root: Path, rel: str, payload: dict[str, Any]) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
    )
    return rel


def _write_jsonl(root: Path, rel: str, rows: list[dict[str, Any]]) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    return rel


def _write_lerobot_tasks(root: Path, rows: list[dict[str, Any]]) -> str:
    rel = "meta/tasks.parquet"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError:
        _write_parquet(
            root,
            rel,
            rows,
            pa.schema(
                [
                    pa.field("task_index", pa.int64()),
                    pa.field("task", pa.string()),
                ]
            ),
        )
        return rel

    frame = pd.DataFrame(rows).set_index("task")
    frame.to_parquet(path)
    return rel


def _write_parquet(root: Path, rel: str, rows: list[dict[str, Any]], schema: pa.Schema) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression=None)
    return rel


def _content_hash(root: Path, files: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update((root / rel).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _record_transform(
    lake: Lake,
    manifest: DatasetExportManifest,
    *,
    created_by: str,
) -> None:
    now = datetime.now(UTC)
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{manifest.transform_id}'")
    transform_row = {
        "transform_id": manifest.transform_id,
        "kind": "dataset-export",
        "source_id": manifest.dataset_id,
        "input_uris": [f"{lake.uri}#dataset_snapshots/{manifest.dataset_id}"],
        "input_table_versions": list(manifest.table_versions),
        "output_tables": [],
        "params": json.dumps(
            {
                "dataset_id": manifest.dataset_id,
                "snapshot_name": manifest.snapshot_name,
                "format": manifest.format,
                "format_version": manifest.format_version,
                "out_dir": manifest.out_dir,
                "content_hash": manifest.content_hash,
                "episode_count": manifest.episode_count,
                "step_count": manifest.step_count,
                "data_files": list(manifest.data_files),
                "native_loader": manifest.native_loader,
                "lossy_mapping": list(manifest.lossy_mapping),
                "accounting": dict(manifest.accounting),
            },
            sort_keys=True,
        ),
        "status": "completed",
        "error": "",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): dataset-export transform without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
