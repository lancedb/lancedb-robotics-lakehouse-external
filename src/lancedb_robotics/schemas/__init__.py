"""Canonical Arrow schemas for the LanceDB robotics lake (v0 contract).

Column sets follow the SDK's "Core Table Schemas" design, trimmed to the
current substrate: IDs, timestamps, raw URI/provenance, source-of-record
writeback rows, and lineage fields. Deliberately deferred to later feature
sets:

- payload blob/bytes columns on ``observations`` (ingest, feature set 3)
- embedding vector columns (dimension is unknown until an embedder is chosen;
  added via ``add_columns`` by the enrichment feature set)
- ``simulation_runs`` (later simulation epic; this task keeps it a documented
  stub rather than forcing early schema churn)

Structural rules: no struct nested inside struct (Lance does not round-trip
nested-struct nullity), free-form metadata as ``list<struct<key,value>>``,
JSON-encoded strings for tool-specific specs and reports.

Each schema carries its name and version in Arrow schema metadata, which Lance
persists, so a reopened lake can report versions without a side table.
"""

import pyarrow as pa

SCHEMA_METADATA_TABLE_KEY = "lancedb_robotics.table"
SCHEMA_METADATA_VERSION_KEY = "lancedb_robotics.schema_version"

# Lance blob-encoding marker (decision 0024, backlog 0035). A column whose field
# metadata carries this key is stored as a blob sidecar: a scan that does not
# project it reads no blob bytes, and bytes are fetchable lazily by row id
# (``take_blobs`` / ``BlobFile``). The Table-level schema still reports the column
# as ``large_binary``; reads return it as a ``struct<position, size>`` lazy ref
# until the bytes are explicitly fetched. In-place ``Table.update`` is not
# supported on a table that has a blob column, which is why annotation moves to
# the additive/versioned model (see ``lancedb_robotics.quality``).
BLOB_ENCODING_KEY = "lance-encoding:blob"
BLOB_ENCODING_METADATA = {BLOB_ENCODING_KEY: "true"}


def _blob(field: pa.Field) -> pa.Field:
    """Tag ``field`` as a Lance blob-encoded column (decision 0024 / backlog 0035)."""
    return field.with_metadata(BLOB_ENCODING_METADATA)


def _kv() -> pa.DataType:
    """Free-form metadata: list<struct<key,value>> keeps unknown keys queryable."""
    return pa.list_(pa.struct([("key", pa.string()), ("value", pa.string())]))


def _table_versions() -> pa.DataType:
    """Per-table Lance version/tag pins for reproducible snapshots."""
    return pa.list_(
        pa.struct([("table", pa.string()), ("version", pa.int64()), ("tag", pa.string())])
    )


def _jsonb() -> pa.DataType:
    """LanceDB JSONB-compatible dynamic maps backed by Arrow JSON."""
    return pa.json_(pa.string())


def _schema(table: str, version: str, fields: list[pa.Field]) -> pa.Schema:
    return pa.schema(
        fields,
        metadata={
            SCHEMA_METADATA_TABLE_KEY: table,
            SCHEMA_METADATA_VERSION_KEY: version,
        },
    )


_CREATED_AT = pa.field("created_at", pa.timestamp("us", tz="UTC"))


INTEGRATION_SOURCES_SCHEMA = _schema(
    "integration_sources",
    "1",
    [
        pa.field("source_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("display_name", pa.string()),
        pa.field("uri", pa.string()),
        pa.field("auth_ref", pa.string()),
        pa.field("metadata", _kv()),
        _CREATED_AT,
    ],
)

RUNS_SCHEMA = _schema(
    "runs",
    "1",
    [
        pa.field("run_id", pa.string()),
        pa.field("run_kind", pa.string()),
        pa.field("source", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("raw_uri", pa.string()),
        pa.field("robot_id", pa.string()),
        pa.field("site_id", pa.string()),
        pa.field("task_id", pa.string()),
        pa.field("start_time_ns", pa.int64()),
        pa.field("end_time_ns", pa.int64()),
        pa.field("duration_ns", pa.int64()),
        pa.field("software_version", pa.string()),
        pa.field("hardware_version", pa.string()),
        pa.field("calibration_version", pa.string()),
        pa.field("model_version", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("quality_flags", pa.list_(pa.string())),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

OBSERVATIONS_SCHEMA = _schema(
    "observations",
    # v2 (backlog 0014): payload columns added. Decoding MCAP message bytes into
    # observations is a breaking change; lakes created at v1 re-ingest (the same
    # re-ingest story as the content-addressed-id change, decision 0023).
    # v3 (backlog 0035, decision 0024): payload_blob becomes Lance blob-encoded.
    # The encoding is a storage-format change, so v2 lakes re-ingest as well.
    # v4 (backlog 0029): observations become the frame grain by carrying stable
    # episode/frame indices plus denormalized hot-path training/filter scalars.
    "4",
    [
        pa.field("observation_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("episode_id", pa.string()),
        pa.field("episode_index", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("timestamp_ns", pa.int64()),
        pa.field("sensor_id", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("modality", pa.string()),
        # Denormalized frame-grain scalars (decision 0025 / backlog 0029). The
        # structural episode/run rows remain the provenance source of record,
        # but hot curation/training filters stay single-table predicates.
        pa.field("robot_id", pa.string()),
        pa.field("site_id", pa.string()),
        pa.field("task_id", pa.string()),
        pa.field("software_version", pa.string()),
        pa.field("outcome", pa.string()),
        # Flat raw_ref: enough provenance to recover original bytes.
        pa.field("raw_uri", pa.string()),
        pa.field("raw_channel", pa.string()),
        pa.field("raw_log_time_ns", pa.int64()),
        pa.field("raw_sequence", pa.int64()),
        # Decoded payload (backlog 0014). payload_json is the decoded message as
        # canonical JSON (NULL when undecodable); payload_blob carries large
        # binary message bytes hoisted out so re-decode/export need not reopen
        # the source file (NULL for scalar messages). message_encoding /
        # schema_encoding are enough to re-decode. decode_status is one of
        # decoded | raw | failed; decode_error explains raw/failed outcomes.
        #
        # payload_blob is Lance blob-encoded (decision 0024: Lance is the index
        # *and* the fast-access layer). A scan that does not project it reads no
        # blob bytes, and a row's bytes are fetchable lazily by id (take_blobs /
        # BlobFile -- see lancedb_robotics.blob). The cost of blob encoding is
        # that in-place Table.update no longer works on this table, so quality and
        # other annotations are written additively/versioned (lancedb_robotics.
        # quality), never via row UPDATE. Camera/video pixels are NOT duplicated
        # here (decision 0025): they live once in the codec-aware video column
        # (backlog 0030) and are referenced by (episode_index, frame_index);
        # payload_blob is the blob home for non-video heavy payloads
        # (lidar/pointcloud, standalone compressed images).
        pa.field("payload_json", pa.string()),
        _blob(pa.field("payload_blob", pa.large_binary())),
        pa.field("message_encoding", pa.string()),
        pa.field("schema_encoding", pa.string()),
        pa.field("decode_status", pa.string()),
        pa.field("decode_error", pa.string()),
        pa.field("state_vector", pa.list_(pa.float32())),
        pa.field("action_vector", pa.list_(pa.float32())),
        pa.field("caption", pa.string()),
        pa.field("quality_flags", pa.list_(pa.string())),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

EPISODES_SCHEMA = _schema(
    "episodes",
    "1",
    [
        pa.field("episode_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("episode_index", pa.int64()),
        pa.field("from_timestamp_ns", pa.int64()),
        pa.field("to_timestamp_ns", pa.int64()),
        pa.field("boundary_source", pa.string()),
        pa.field("outcome", pa.string()),
        pa.field("frame_count", pa.int64()),
        # Camera/video bytes are not duplicated here; these are lazy frame/blob
        # handles until backlog 0030 adds codec-aware video storage.
        pa.field("camera_blobs", pa.list_(pa.string())),
        pa.field("task_id", pa.string()),
        # Optional semantic handle; provider-specific vector dimensions are
        # still free to add named embedding columns later.
        pa.field("embedding", pa.list_(pa.float32())),
        # JSON provenance for marker ids, query predicates, and source ranges.
        pa.field("provenance", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

VIDEOS_SCHEMA = _schema(
    "videos",
    "1",
    [
        pa.field("video_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("episode_id", pa.string()),
        pa.field("episode_index", pa.int64()),
        pa.field("camera_key", pa.string()),
        pa.field("sensor_id", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("from_timestamp_ns", pa.int64()),
        pa.field("to_timestamp_ns", pa.int64()),
        pa.field("frame_count", pa.int64()),
        pa.field("observation_ids", pa.list_(pa.string())),
        pa.field("raw_uri", pa.string()),
        pa.field("codec", pa.string()),
        pa.field("uri", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

VIDEO_ENCODINGS_SCHEMA = _schema(
    "video_encodings",
    "1",
    [
        pa.field("encoding_id", pa.string()),
        pa.field("video_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("episode_id", pa.string()),
        pa.field("episode_index", pa.int64()),
        pa.field("camera_key", pa.string()),
        pa.field("codec", pa.string()),
        pa.field("gop_size", pa.int64()),
        pa.field("resolution", pa.string()),
        pa.field("fps", pa.float64()),
        pa.field("frame_count", pa.int64()),
        pa.field("keyframe_map_ref", pa.string()),
        pa.field("keyframe_map_json", pa.string()),
        pa.field("nvdec_compatible", pa.bool_()),
        pa.field("source_size_bytes", pa.int64()),
        pa.field("encoded_size_bytes", pa.int64()),
        # Codec-aware video bytes live in Lance as a blob-encoded column. The
        # keyframe map above resolves frame -> GOP byte range, then callers fetch
        # and decode only the enclosing GOP rather than scanning a whole clip.
        _blob(pa.field("data", pa.large_binary())),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

KEYFRAME_MAP_ARTIFACTS_SCHEMA = _schema(
    "keyframe_map_artifacts",
    "1",
    [
        pa.field("artifact_id", pa.string()),
        pa.field("keyframe_map_ref", pa.string()),
        pa.field("content_sha256", pa.string()),
        pa.field("json_size_bytes", pa.int64()),
        pa.field("frame_count", pa.int64()),
        pa.field("gop_count", pa.int64()),
        pa.field("source_video_fingerprint", pa.string()),
        pa.field("inspection_id", pa.string()),
        pa.field("source_uri", pa.string()),
        pa.field("source_path", pa.string()),
        pa.field("encoding_id", pa.string()),
        pa.field("video_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("episode_id", pa.string()),
        pa.field("episode_index", pa.int64()),
        pa.field("camera_key", pa.string()),
        pa.field("keyframe_map_json", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

ATTACHMENTS_SCHEMA = _schema(
    "attachments",
    # Backlog 0016: MCAP attachment records (embedded files travelling with the
    # log — calibration, intrinsics, mission/config blobs, thumbnails). One row
    # per attachment, linked to its run by ``run_id``. The bytes live in the lake
    # (``data``), not a sidecar, so calibration/run context travels with the log
    # into the lakehouse and ``export`` can round-trip it later.
    # v2 (backlog 0035, decision 0024): ``data`` becomes Lance blob-encoded, the
    # same storage-format change as observations.payload_blob; v1 lakes re-ingest.
    "2",
    [
        pa.field("attachment_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("media_type", pa.string()),
        pa.field("size", pa.int64()),
        # Content hash of the attachment bytes: a stable handle, and the seam an
        # export round-trip verifies the recovered bytes against.
        pa.field("sha256", pa.string()),
        pa.field("log_time_ns", pa.int64()),
        pa.field("create_time_ns", pa.int64()),
        # Bytes are Lance blob-encoded (decision 0024 / backlog 0035), like
        # observations.payload_blob (see OBSERVATIONS_SCHEMA): a metadata scan
        # reads no attachment bytes, and a row's bytes are fetched lazily by id
        # (take_blobs / BlobFile -- see lancedb_robotics.blob). Blob encoding
        # rules out in-place Table.update on this table, which attachments never
        # needed -- they are written once at ingest and read back by fetch.
        _blob(pa.field("data", pa.large_binary())),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

EVENTS_SCHEMA = _schema(
    "events",
    "1",
    [
        pa.field("event_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("timestamp_ns", pa.int64()),
        pa.field("start_time_ns", pa.int64()),
        pa.field("end_time_ns", pa.int64()),
        pa.field("event_type", pa.string()),
        pa.field("severity", pa.string()),
        pa.field("source", pa.string()),
        pa.field("notes", pa.string()),
        pa.field("linked_incident_id", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

SCENARIOS_SCHEMA = _schema(
    "scenarios",
    "3",
    [
        pa.field("scenario_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("start_time_ns", pa.int64()),
        pa.field("end_time_ns", pa.int64()),
        pa.field("window_ns", pa.int64()),
        pa.field("is_partial", pa.bool_()),
        pa.field("topics", pa.list_(pa.string())),
        pa.field("observation_ids", pa.list_(pa.string())),
        pa.field("observation_count", pa.int64()),
        pa.field("scenario_type", pa.string()),
        pa.field("trigger_event_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("parent_scenario_id", pa.string()),
        pa.field("coverage_tags", pa.list_(pa.string())),
        # Semantic handle written by the enrichment feature set (backlog 0007).
        # NULL until a caption provider runs. Embedding vectors are added as a
        # separate column at enrich time via ``add_columns`` (their dimension
        # is owned by the chosen embedder, not the canonical contract).
        pa.field("summary", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

DATASET_SNAPSHOTS_SCHEMA = _schema(
    "dataset_snapshots",
    "1",
    [
        pa.field("dataset_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("query_spec", pa.string()),
        pa.field("table_versions", _table_versions()),
        pa.field("tag", pa.string()),
        pa.field("split", pa.string()),
        pa.field("balance_report", pa.string()),
        pa.field("coverage_report", pa.string()),
        pa.field("created_by", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

TRAINING_RUNS_SCHEMA = _schema(
    "training_runs",
    "1",
    [
        pa.field("training_run_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("snapshot_name", pa.string()),
        pa.field("snapshot_tag", pa.string()),
        pa.field("table_versions", _table_versions()),
        pa.field("row_plan_id", pa.string()),
        pa.field("epoch_plan_id", pa.string()),
        pa.field("projection_manifest_ids", pa.list_(pa.string())),
        pa.field("code_ref", pa.string()),
        pa.field("package_versions_json", pa.string()),
        pa.field("environment_json", pa.string()),
        pa.field("hardware_json", pa.string()),
        pa.field("runtime_json", pa.string()),
        pa.field("hyperparameters_json", pa.string()),
        pa.field("random_seeds_json", pa.string()),
        pa.field("split_policy_json", pa.string()),
        pa.field("external_refs", _kv()),
        pa.field("status", pa.string()),
        pa.field("manifest_digest", pa.string()),
        pa.field("transform_id", pa.string()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

MODEL_ARTIFACTS_SCHEMA = _schema(
    "model_artifacts",
    "1",
    [
        pa.field("model_artifact_id", pa.string()),
        pa.field("training_run_id", pa.string()),
        pa.field("artifact_uri", pa.string()),
        pa.field("checksum", pa.string()),
        pa.field("aliases", pa.list_(pa.string())),
        pa.field("framework", pa.string()),
        pa.field("epoch", pa.int64()),
        pa.field("step", pa.int64()),
        pa.field("metrics_json", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("external_refs", _kv()),
        pa.field("manifest_digest", pa.string()),
        pa.field("transform_id", pa.string()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

EVALUATION_RUNS_SCHEMA = _schema(
    "evaluation_runs",
    "1",
    [
        pa.field("eval_run_id", pa.string()),
        pa.field("model_artifact_id", pa.string()),
        pa.field("training_run_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("snapshot_name", pa.string()),
        pa.field("snapshot_tag", pa.string()),
        pa.field("table_versions", _table_versions()),
        pa.field("metrics_json", pa.string()),
        pa.field("slice_metrics_json", pa.string()),
        pa.field("failure_outputs_json", pa.string()),
        pa.field("code_ref", pa.string()),
        pa.field("package_versions_json", pa.string()),
        pa.field("environment_json", pa.string()),
        pa.field("hardware_json", pa.string()),
        pa.field("runtime_json", pa.string()),
        pa.field("external_refs", _kv()),
        pa.field("status", pa.string()),
        pa.field("manifest_digest", pa.string()),
        pa.field("transform_id", pa.string()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

EVALUATION_RUN_METRICS_SCHEMA = _schema(
    "evaluation_run_metrics",
    # v1 (backlog 0100): scalar-indexable materialized surface over the aggregate
    # and slice metrics that 0062 stores as ``evaluation_runs.metrics_json`` /
    # ``slice_metrics_json``. The source of record stays those JSON columns; this
    # table promotes each numeric metric into one row keyed by (eval_run_id,
    # scope, metric_key) so a metric-key lookup ("night/rain.success_rate < 0.8")
    # pushes down to an indexed predicate instead of parsing every eval manifest
    # in Python. Rows are deterministic per source metric and fully rebuildable
    # from ``evaluation_runs`` (``run_manifests.sync_evaluation_run_metrics`` /
    # ``train sync-metrics``); the eval write path also emits them inline (0098
    # inline-emission model) so the materialized path is available by default.
    "1",
    [
        pa.field("metric_row_id", pa.string()),
        pa.field("eval_run_id", pa.string()),
        pa.field("model_artifact_id", pa.string()),
        pa.field("training_run_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("snapshot_name", pa.string()),
        pa.field("snapshot_tag", pa.string()),
        # scope: "aggregate" for top-level metrics, "slice" for slice_metrics.
        pa.field("scope", pa.string()),
        # slice_label is "" for aggregate rows, else the slice key (e.g. "night/rain").
        pa.field("slice_label", pa.string()),
        pa.field("metric", pa.string()),
        # metric_key: metric for aggregate rows, "<slice_label>.<metric>" for slices;
        # the single high-cardinality handle a lookup filters on.
        pa.field("metric_key", pa.string()),
        # score is the numeric value; value_json preserves non-numeric raw values.
        pa.field("score", pa.float64()),
        pa.field("value_json", pa.string()),
        pa.field("status", pa.string()),
        pa.field("eval_created_at", pa.timestamp("us", tz="UTC")),
        pa.field("transform_id", pa.string()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

CURATION_VIEWS_SCHEMA = _schema(
    "curation_views",
    "2",
    [
        pa.field("view_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("owner", pa.string()),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("description", pa.string()),
        pa.field("source_kind", pa.string()),
        pa.field("scope", pa.string()),
        pa.field("query_spec", pa.string()),
        pa.field("scenario_ids", pa.list_(pa.string())),
        pa.field("table_versions", _table_versions()),
        pa.field("parent_transform_ids", pa.list_(pa.string())),
        pa.field("status", pa.string()),
        pa.field("created_by", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

CURATION_VIEW_MEMBERSHIP_CHUNKS_SCHEMA = _schema(
    "curation_view_membership_chunks",
    "1",
    [
        pa.field("chunk_id", pa.string()),
        pa.field("view_id", pa.string()),
        pa.field("chunk_index", pa.int64()),
        pa.field("start_ordinal", pa.int64()),
        pa.field("end_ordinal", pa.int64()),
        pa.field("scenario_ids", pa.list_(pa.string())),
        pa.field("scenario_count", pa.int64()),
        pa.field("chunk_digest", pa.string()),
        pa.field("created_by", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

CURATION_MEMBERSHIPS_SCHEMA = _schema(
    "curation_memberships",
    "2",
    [
        pa.field("membership_id", pa.string()),
        pa.field("view_id", pa.string()),
        pa.field("target_grain", pa.string()),
        pa.field("target_id", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("decision", pa.string()),
        pa.field("reason_code", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("note", pa.string()),
        pa.field("reviewer", pa.string()),
        pa.field("queue", pa.string()),
        pa.field("priority", pa.int64()),
        pa.field("score", pa.float64()),
        pa.field("metadata", _kv()),
        pa.field("source", pa.string()),
        pa.field("supersedes_membership_id", pa.string()),
        pa.field("created_by", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

CURATION_REVIEW_QUEUES_SCHEMA = _schema(
    "curation_review_queues",
    "1",
    [
        pa.field("queue_item_id", pa.string()),
        pa.field("queue_id", pa.string()),
        pa.field("queue_name", pa.string()),
        pa.field("target_grain", pa.string()),
        pa.field("target_id", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("source_operation", pa.string()),
        pa.field("source_ref", pa.string()),
        pa.field("priority", pa.int64()),
        pa.field("priority_score", pa.float64()),
        pa.field("priority_reason", pa.string()),
        pa.field("assignee", pa.string()),
        pa.field("status", pa.string()),
        pa.field("export_uri", pa.string()),
        pa.field("external_task_id", pa.string()),
        pa.field("external_url", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("table_versions", _table_versions()),
        pa.field("source_transform_ids", pa.list_(pa.string())),
        pa.field("created_by", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

CURATION_MATERIALIZATIONS_SCHEMA = _schema(
    "curation_materializations",
    "1",
    [
        pa.field("materialization_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("snapshot_name", pa.string()),
        pa.field("target_format", pa.string()),
        pa.field("output_uri", pa.string()),
        pa.field("mode", pa.string()),
        pa.field("selected_scenario_count", pa.int64()),
        pa.field("selected_observation_count", pa.int64()),
        pa.field("total_payload_bytes", pa.int64()),
        pa.field("copied_payload_bytes", pa.int64()),
        pa.field("logical_reference_bytes", pa.int64()),
        pa.field("metadata_bytes_written", pa.int64()),
        pa.field("copy_ratio", pa.float64()),
        pa.field("source_table_versions", _table_versions()),
        pa.field("report_json", pa.string()),
        pa.field("projection_transform_id", pa.string()),
        pa.field("created_by", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

CURATION_COMPARISONS_SCHEMA = _schema(
    "curation_comparisons",
    # v2 (backlog 0093): comparison reports become a first-class catalog with a
    # reload-by-id/alias surface and a safe-delete retention lifecycle. Added
    # ``pair_alias`` (stable snapshot-pair handle for latest resolution),
    # ``state`` (active|archived|pruned), report-body digest/byte accounting, and
    # retention metadata. Pruning clears ``report_json`` only; snapshot ids,
    # table versions, transform id, counts, and the body digest survive so
    # lineage and snapshot evidence remain queryable. Lakes created at v1 add the
    # new columns via ``Lake.init`` (same re-init story as 0057's table add).
    "2",
    [
        pa.field("comparison_id", pa.string()),
        pa.field("pair_alias", pa.string()),
        pa.field("state", pa.string()),
        pa.field("left_dataset_id", pa.string()),
        pa.field("right_dataset_id", pa.string()),
        pa.field("left_snapshot_name", pa.string()),
        pa.field("right_snapshot_name", pa.string()),
        pa.field("metrics", pa.list_(pa.string())),
        pa.field("dimensions", pa.list_(pa.string())),
        pa.field("added_scenario_count", pa.int64()),
        pa.field("removed_scenario_count", pa.int64()),
        pa.field("shared_scenario_count", pa.int64()),
        pa.field("report_json", pa.string()),
        pa.field("report_sha1", pa.string()),
        pa.field("report_bytes", pa.int64()),
        pa.field("retention_policy_json", pa.string()),
        pa.field("archived_at", pa.timestamp("us", tz="UTC")),
        pa.field("pruned_at", pa.timestamp("us", tz="UTC")),
        pa.field("table_versions", _table_versions()),
        pa.field("created_by", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

EVAL_METRIC_CATALOG_SCHEMA = _schema(
    "eval_metric_catalog",
    # v1 (backlog 0095): scalar-indexable catalog over the feedback-loop eval
    # metric rows that 0059 stores in ``model_outputs``. The source of record
    # stays ``model_outputs.output_json``; this table promotes the metric
    # identity (snapshot, training/eval run, model version, metric, slice) and
    # scores into dedicated columns so lookups push down instead of parsing
    # JSON per row, and carries the lifecycle state (active|superseded|pruned)
    # that must not be written onto the audit rows themselves. Rows are
    # deterministic per ``model_output_id`` and fully rebuildable from
    # ``model_outputs`` (``curate sync-eval-metrics``). Pruning deletes the
    # superseded source row only; the catalog row survives as audit metadata.
    "1",
    [
        pa.field("model_output_id", pa.string()),
        pa.field("series_key", pa.string()),
        pa.field("state", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("snapshot_name", pa.string()),
        pa.field("snapshot_tag", pa.string()),
        pa.field("training_run_id", pa.string()),
        pa.field("model_version", pa.string()),
        pa.field("evaluation_run_id", pa.string()),
        pa.field("metric", pa.string()),
        pa.field("output_type", pa.string()),
        pa.field("slice_label", pa.string()),
        pa.field("slice_values", _kv()),
        pa.field("score", pa.float64()),
        pa.field("baseline_score", pa.float64()),
        pa.field("improvement", pa.float64()),
        pa.field("higher_is_better", pa.bool_()),
        pa.field("regressed", pa.bool_()),
        pa.field("regression_threshold", pa.float64()),
        pa.field("scenario_count", pa.int64()),
        pa.field("superseded_by", pa.string()),
        pa.field("superseded_at", pa.timestamp("us", tz="UTC")),
        pa.field("pruned_at", pa.timestamp("us", tz="UTC")),
        pa.field("retention_policy_json", pa.string()),
        pa.field("table_versions", _table_versions()),
        pa.field("created_by", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

DISTRIBUTION_CATALOG_SCHEMA = _schema(
    "distribution_catalog",
    "1",
    [
        pa.field("catalog_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("name", pa.string()),
        pa.field("spec_id", pa.string()),
        pa.field("report_id", pa.string()),
        pa.field("comparison_id", pa.string()),
        pa.field("finding_id", pa.string()),
        pa.field("source_kind", pa.string()),
        pa.field("source_name", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("view_id", pa.string()),
        pa.field("source_digest", pa.string()),
        pa.field("summary_json", pa.string()),
        pa.field("body_json", pa.string()),
        pa.field("body_sha1", pa.string()),
        pa.field("body_bytes", pa.int64()),
        pa.field("body_compacted", pa.bool_()),
        pa.field("retention_policy_json", pa.string()),
        pa.field("expires_at", pa.timestamp("us", tz="UTC")),
        pa.field("compacted_at", pa.timestamp("us", tz="UTC")),
        pa.field("table_versions", _table_versions()),
        pa.field("source_transform_ids", pa.list_(pa.string())),
        pa.field("created_by", pa.string()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

LABELS_SCHEMA = _schema(
    "labels",
    "1",
    [
        pa.field("label_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("observation_id", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("event_id", pa.string()),
        pa.field("label_type", pa.string()),
        # Denormalized scalar used for hot-path filters/training. The full
        # annotation payload stays in label_value for audit/tool round-trips.
        pa.field("label", pa.string()),
        pa.field("label_value", pa.string()),
        pa.field("label_spec", pa.string()),
        pa.field("source", pa.string()),
        pa.field("reviewer", pa.string()),
        pa.field("confidence", pa.float32()),
        pa.field("status", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

MODEL_OUTPUTS_SCHEMA = _schema(
    "model_outputs",
    "1",
    [
        pa.field("model_output_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("observation_id", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("model_version", pa.string()),
        pa.field("output_type", pa.string()),
        # Denormalized scalar prediction + score live on the target grain; the
        # full provider-specific output is source-of-record JSON here.
        pa.field("prediction", pa.string()),
        pa.field("output_json", pa.string()),
        pa.field("score", pa.float32()),
        pa.field("producer_run_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

FEEDBACK_SCHEMA = _schema(
    "feedback",
    "1",
    [
        pa.field("feedback_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("observation_id", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("event_id", pa.string()),
        pa.field("label_id", pa.string()),
        pa.field("model_output_id", pa.string()),
        pa.field("feedback_type", pa.string()),
        pa.field("severity", pa.string()),
        pa.field("linked_incident_id", pa.string()),
        pa.field("notes", pa.string()),
        pa.field("source", pa.string()),
        pa.field("status", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

ALIGNMENT_JOBS_SCHEMA = _schema(
    "alignment_jobs",
    "1",
    [
        pa.field("alignment_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("input_tables", pa.list_(pa.string())),
        pa.field("input_versions", _table_versions()),
        pa.field("streams", pa.list_(pa.string())),
        pa.field("clock", pa.string()),
        pa.field("rate_hz", pa.float64()),
        pa.field("tolerance_ms", pa.float64()),
        # JSON-encoded recipe and quality summary keep the alignment contract
        # inspectable while the aligned rows remain a virtual view by default.
        pa.field("recipe", pa.string()),
        pa.field("output_table", pa.string()),
        pa.field("quality_summary", pa.string()),
        pa.field("quality_flags", pa.list_(pa.string())),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

ALIGNED_FRAMES_SCHEMA = _schema(
    "aligned_frames",
    "1",
    [
        pa.field("aligned_frame_id", pa.string()),
        pa.field("alignment_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("tick_index", pa.int64()),
        pa.field("timestamp_ns", pa.int64()),
        pa.field("stream", pa.string()),
        pa.field("status", pa.string()),
        pa.field("interpolation", pa.string()),
        pa.field("observation_id", pa.string()),
        pa.field("source_observation_ids", pa.list_(pa.string())),
        # Row ids are version-relative, so every row is only meaningful with the
        # alignment job's pinned `observations` table version.
        pa.field("source_row_ids", pa.list_(pa.int64())),
        pa.field("source_timestamp_ns", pa.int64()),
        pa.field("source_time_ns", pa.int64()),
        pa.field("receive_time_ns", pa.int64()),
        pa.field("latency_ns", pa.int64()),
        pa.field("error_ns", pa.int64()),
        pa.field("absolute_error_ns", pa.int64()),
        pa.field("confidence", pa.float64()),
        pa.field("value_json", pa.string()),
        pa.field("quality_flags", pa.list_(pa.string())),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

ALIGNED_TICKS_SCHEMA = _schema(
    "aligned_ticks",
    "1",
    [
        pa.field("aligned_tick_id", pa.string()),
        pa.field("alignment_id", pa.string()),
        pa.field("alignment_name", pa.string()),
        pa.field("recipe_digest", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("tick_index", pa.int64()),
        pa.field("timestamp_ns", pa.int64()),
        pa.field("available_streams", pa.list_(pa.string())),
        pa.field("missing_streams", pa.list_(pa.string())),
        pa.field("interpolated_streams", pa.list_(pa.string())),
        pa.field("out_of_tolerance_streams", pa.list_(pa.string())),
        pa.field("has_missing", pa.bool_()),
        pa.field("has_out_of_tolerance", pa.bool_()),
        pa.field("min_confidence", pa.float64()),
        pa.field("quality_flags", pa.list_(pa.string())),
        # Dynamic, recipe-dependent stream maps live in JSONB columns. Common
        # predicates graduate to typed top-level columns through additive schema
        # evolution rather than JSON-path assumptions.
        pa.field("stream_detail_json", _jsonb()),
        pa.field("masks_json", _jsonb()),
        pa.field("stream_values_json", _jsonb()),
        pa.field("lineage_json", _jsonb()),
        pa.field("transform_id", pa.string()),
        _CREATED_AT,
    ],
)

TRANSFORM_RUNS_SCHEMA = _schema(
    "transform_runs",
    "1",
    [
        pa.field("transform_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("input_uris", pa.list_(pa.string())),
        pa.field("input_table_versions", _table_versions()),
        pa.field("output_tables", pa.list_(pa.string())),
        pa.field("params", pa.string()),
        pa.field("status", pa.string()),
        pa.field("error", pa.string()),
        pa.field("started_at", pa.timestamp("us", tz="UTC")),
        pa.field("finished_at", pa.timestamp("us", tz="UTC")),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

LEROBOT_INGEST_CHECKPOINTS_SCHEMA = _schema(
    "lerobot_ingest_checkpoints",
    "1",
    [
        pa.field("checkpoint_id", pa.string()),
        pa.field("job_id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("transform_id", pa.string()),
        pa.field("source_uri", pa.string()),
        pa.field("source_ref", pa.string()),
        pa.field("hf_repo_id", pa.string()),
        pa.field("requested_revision", pa.string()),
        pa.field("resolved_revision", pa.string()),
        pa.field("hf_cache_path", pa.string()),
        pa.field("hf_download_json", pa.string()),
        pa.field("source_identity_json", pa.string()),
        pa.field("status", pa.string()),
        pa.field("phase", pa.string()),
        pa.field("claim_owner", pa.string()),
        pa.field("claim_token", pa.string()),
        pa.field("checkpoint_index", pa.int64()),
        pa.field("data_file", pa.string()),
        pa.field("row_group", pa.int64()),
        pa.field("batch_index", pa.int64()),
        pa.field("rows_seen", pa.int64()),
        pa.field("observations_written", pa.int64()),
        pa.field("episodes_written", pa.int64()),
        pa.field("scenarios_written", pa.int64()),
        pa.field("videos_written", pa.int64()),
        pa.field("video_encodings_written", pa.int64()),
        pa.field("rows_skipped_existing", pa.int64()),
        pa.field("bytes_scanned", pa.int64()),
        pa.field("last_observation_id", pa.string()),
        pa.field("progress_json", pa.string()),
        pa.field("error", pa.string()),
        pa.field("started_at", pa.timestamp("us", tz="UTC")),
        pa.field("updated_at", pa.timestamp("us", tz="UTC")),
        pa.field("finished_at", pa.timestamp("us", tz="UTC")),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

LEROBOT_CHECKPOINT_HOLDS_SCHEMA = _schema(
    "lerobot_checkpoint_holds",
    "1",
    [
        pa.field("hold_id", pa.string()),
        pa.field("selector_json", pa.string()),
        pa.field("checkpoint_ids", pa.list_(pa.string())),
        pa.field("job_id", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("hf_repo_id", pa.string()),
        pa.field("requested_revision", pa.string()),
        pa.field("resolved_revision", pa.string()),
        pa.field("statuses", pa.list_(pa.string())),
        pa.field("updated_after", pa.timestamp("us", tz="UTC")),
        pa.field("updated_before", pa.timestamp("us", tz="UTC")),
        pa.field("retain_until", pa.timestamp("us", tz="UTC")),
        pa.field("legal_hold", pa.bool_()),
        pa.field("audit_hold", pa.bool_()),
        pa.field("promotion_hold", pa.bool_()),
        pa.field("owner", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("active", pa.bool_()),
        pa.field("released_at", pa.timestamp("us", tz="UTC")),
        pa.field("released_by", pa.string()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

LINEAGE_ARTIFACTS_SCHEMA = _schema(
    "lineage_artifacts",
    "1",
    [
        pa.field("artifact_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("name", pa.string()),
        pa.field("table_name", pa.string()),
        pa.field("table_version", pa.int64()),
        pa.field("table_tag", pa.string()),
        pa.field("row_grain", pa.string()),
        pa.field("row_ids", pa.list_(pa.string())),
        pa.field("source_uri", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("digest", pa.string()),
        pa.field("producer_execution_id", pa.string()),
        pa.field("metadata", _kv()),
        _CREATED_AT,
    ],
)

LINEAGE_EXECUTIONS_SCHEMA = _schema(
    "lineage_executions",
    "1",
    [
        pa.field("execution_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("name", pa.string()),
        pa.field("transform_id", pa.string()),
        pa.field("status", pa.string()),
        pa.field("params_json", pa.string()),
        pa.field("code_ref", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("environment_json", pa.string()),
        pa.field("input_artifact_ids", pa.list_(pa.string())),
        pa.field("output_artifact_ids", pa.list_(pa.string())),
        pa.field("input_table_versions", _table_versions()),
        pa.field("output_table_versions", _table_versions()),
        pa.field("started_at", pa.timestamp("us", tz="UTC")),
        pa.field("finished_at", pa.timestamp("us", tz="UTC")),
        pa.field("created_by", pa.string()),
        pa.field("metadata", _kv()),
        _CREATED_AT,
    ],
)

LINEAGE_EDGES_SCHEMA = _schema(
    "lineage_edges",
    "1",
    [
        pa.field("edge_id", pa.string()),
        pa.field("edge_type", pa.string()),
        pa.field("from_artifact_id", pa.string()),
        pa.field("to_artifact_id", pa.string()),
        pa.field("execution_id", pa.string()),
        pa.field("metadata", _kv()),
        _CREATED_AT,
    ],
)

LINEAGE_DELIVERY_ATTEMPTS_SCHEMA = _schema(
    "lineage_delivery_attempts",
    "1",
    [
        pa.field("attempt_id", pa.string()),
        pa.field("backend", pa.string()),
        pa.field("target", pa.string()),
        pa.field("payload_kind", pa.string()),
        pa.field("payload_digest", pa.string()),
        pa.field("payload_count", pa.int64()),
        pa.field("mode", pa.string()),
        pa.field("status", pa.string()),
        pa.field("remote_response_ids", pa.list_(pa.string())),
        pa.field("error", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

# Durable lineage-audit report catalog (backlog 0112). One idempotent row per
# content-addressed audit report, keyed by ``report_digest`` (== ``report_id``).
# The report body is stored inline so cleanup/release gates can require and reload
# a recent passed audit without re-running graph traversal. Summary count columns
# keep listing/filtering cheap while ``report_json`` preserves the full finding
# payload and validator diagnostics for drill-down/export.
LINEAGE_AUDIT_REPORTS_SCHEMA = _schema(
    "lineage_audit_reports",
    "1",
    [
        pa.field("report_id", pa.string()),
        pa.field("report_digest", pa.string()),
        pa.field("catalog_schema_version", pa.string()),
        pa.field("report_schema_version", pa.string()),
        pa.field("lake_uri", pa.string()),
        pa.field("subject", pa.string()),
        pa.field("root_artifact_ids", pa.list_(pa.string())),
        pa.field("status", pa.string()),
        pa.field("artifact_count", pa.int64()),
        pa.field("edge_count", pa.int64()),
        pa.field("finding_count", pa.int64()),
        pa.field("unresolved_reference_count", pa.int64()),
        pa.field("missing_source_count", pa.int64()),
        pa.field("missing_table_version_count", pa.int64()),
        pa.field("stale_external_link_count", pa.int64()),
        pa.field("retained_version_count", pa.int64()),
        pa.field("retention_hold_count", pa.int64()),
        pa.field("cleanup_candidate_count", pa.int64()),
        pa.field("validator_statuses_json", pa.string()),
        pa.field("report_json", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
        pa.field("updated_at", pa.timestamp("us", tz="UTC")),
    ],
)

# Durable evidence-pack catalog (backlog 0108). One idempotent row per evidence
# pack, keyed by ``manifest_digest`` (== ``pack_id``). The full metadata-first v1
# manifest is stored inline so a pack can be reloaded by digest or subject handle
# without re-tracing the lineage graph. Retention/protection and redaction
# metadata is queryable and survives catalog reloads.
EVIDENCE_PACKS_SCHEMA = _schema(
    "evidence_packs",
    "1",
    [
        pa.field("pack_id", pa.string()),
        pa.field("manifest_digest", pa.string()),
        pa.field("catalog_schema_version", pa.string()),
        pa.field("manifest_schema_version", pa.string()),
        pa.field("subject_kind", pa.string()),
        pa.field("subject_handle", pa.string()),
        pa.field("lake_uri", pa.string()),
        pa.field("mode", pa.string()),
        pa.field("materialization_status", pa.string()),
        pa.field("output_uri", pa.string()),
        pa.field("bytes_total", pa.int64()),
        pa.field("file_count", pa.int64()),
        pa.field("row_id_count", pa.int64()),
        pa.field("source_coordinate_hashes", pa.list_(pa.string())),
        pa.field("table_version_pins", _table_versions()),
        pa.field("retention_policy", pa.string()),
        pa.field("protected", pa.bool_()),
        pa.field("expires_at", pa.timestamp("us", tz="UTC")),
        pa.field("redaction_policy", pa.string()),
        pa.field("redacted", pa.bool_()),
        pa.field("sensitive_sources", pa.list_(pa.string())),
        pa.field("manifest_json", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
        pa.field("updated_at", pa.timestamp("us", tz="UTC")),
    ],
)

# Append-only audit log for evidence-pack lifecycle events (backlog 0108):
# creation, materialization (full/partial), retention changes, expiry, and
# redaction/sensitive-source denials. Excluded from the lineage graph projection.
EVIDENCE_PACK_EVENTS_SCHEMA = _schema(
    "evidence_pack_events",
    "1",
    [
        pa.field("event_id", pa.string()),
        pa.field("pack_id", pa.string()),
        pa.field("manifest_digest", pa.string()),
        pa.field("event_type", pa.string()),
        pa.field("subject_kind", pa.string()),
        pa.field("subject_handle", pa.string()),
        pa.field("mode", pa.string()),
        pa.field("status", pa.string()),
        pa.field("bytes_total", pa.int64()),
        pa.field("file_count", pa.int64()),
        pa.field("output_uri", pa.string()),
        pa.field("detail", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

# Durable rebuild-plan catalog (backlog 0109). One idempotent row per rebuild
# plan, keyed by ``plan_digest`` (== ``plan_id``), which is content-addressed over
# the plan's roots, reason, severity, and ordered actions -- and independent of
# any invalidation timestamp -- so the same plan re-records to the same row. The
# ordered action list is stored inline (``plan_json``) so a plan reloads and
# exports for orchestrators without re-tracing the lineage graph. Lifecycle
# columns (status/revision/approver/timestamps/note) carry generic approval
# metadata; ``revision`` is an optimistic-concurrency counter that lets status
# updates reject stale writes.
REBUILD_PLANS_SCHEMA = _schema(
    "rebuild_plans",
    "1",
    [
        pa.field("plan_id", pa.string()),
        pa.field("plan_digest", pa.string()),
        pa.field("catalog_schema_version", pa.string()),
        pa.field("plan_schema_version", pa.string()),
        pa.field("lake_uri", pa.string()),
        pa.field("invalidation_id", pa.string()),
        pa.field("root_artifact_ids", pa.list_(pa.string())),
        pa.field("action_count", pa.int64()),
        pa.field("reason", pa.string()),
        pa.field("severity", pa.string()),
        pa.field("status", pa.string()),
        pa.field("revision", pa.int64()),
        pa.field("actor", pa.string()),
        pa.field("approver", pa.string()),
        pa.field("approved_at", pa.timestamp("us", tz="UTC")),
        pa.field("dispatched_at", pa.timestamp("us", tz="UTC")),
        pa.field("completed_at", pa.timestamp("us", tz="UTC")),
        pa.field("note", pa.string()),
        pa.field("plan_json", pa.string()),
        pa.field("table_version_pins", _table_versions()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
        pa.field("updated_at", pa.timestamp("us", tz="UTC")),
    ],
)

# Append-only audit log for rebuild-plan lifecycle events (backlog 0109):
# recording, status transitions (with from/to status and the post-transition
# revision), and orchestrator dispatch exports. Survives status churn and is
# excluded from the lineage graph projection.
REBUILD_PLAN_EVENTS_SCHEMA = _schema(
    "rebuild_plan_events",
    "1",
    [
        pa.field("event_id", pa.string()),
        pa.field("plan_id", pa.string()),
        pa.field("plan_digest", pa.string()),
        pa.field("event_type", pa.string()),
        pa.field("from_status", pa.string()),
        pa.field("to_status", pa.string()),
        pa.field("revision", pa.int64()),
        pa.field("actor", pa.string()),
        pa.field("approver", pa.string()),
        pa.field("detail", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

# Durable retention-policy catalog (backlog 0111). One idempotent row per policy,
# keyed by ``policy_id`` (== ``policy_digest``), content-addressed over the policy
# name/version/scope selectors/rules/owner/reason template -- so re-recording the
# same policy definition maps to the same row and never resets its lifecycle. The
# full policy definition is stored inline (``policy_json``) so a policy reloads,
# applies, and projects without a separate definition store. Rule fields
# (retain_until/retain_for_days/legal_hold/audit_hold/promotion_hold/owner) are
# surfaced as columns for filter pushdown; ``status``/``revision`` carry a generic
# approval-style lifecycle with an optimistic-concurrency counter. Policies expand
# to explicit 0067 artifact holds on apply -- they never store per-artifact state
# here -- and organization-specific authorization stays out of OSS core.
RETENTION_POLICIES_SCHEMA = _schema(
    "retention_policies",
    "1",
    [
        pa.field("policy_id", pa.string()),
        pa.field("policy_digest", pa.string()),
        pa.field("catalog_schema_version", pa.string()),
        pa.field("policy_schema_version", pa.string()),
        pa.field("lake_uri", pa.string()),
        pa.field("name", pa.string()),
        pa.field("version", pa.string()),
        pa.field("scope_summary", pa.string()),
        pa.field("retain_until", pa.timestamp("us", tz="UTC")),
        pa.field("retain_for_days", pa.int64()),
        pa.field("legal_hold", pa.bool_()),
        pa.field("audit_hold", pa.bool_()),
        pa.field("promotion_hold", pa.bool_()),
        pa.field("owner", pa.string()),
        pa.field("status", pa.string()),
        pa.field("revision", pa.int64()),
        pa.field("actor", pa.string()),
        pa.field("approver", pa.string()),
        pa.field("activated_at", pa.timestamp("us", tz="UTC")),
        pa.field("archived_at", pa.timestamp("us", tz="UTC")),
        pa.field("note", pa.string()),
        pa.field("policy_json", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
        pa.field("updated_at", pa.timestamp("us", tz="UTC")),
    ],
)

# Append-only audit log for retention-policy lifecycle (backlog 0111): recording,
# status transitions (with from/to status + post-transition revision), and
# hold application/release/expiry-notification events (with the affected artifact
# count). Survives status churn and policy archival, and is excluded from the
# lineage graph projection.
RETENTION_POLICY_EVENTS_SCHEMA = _schema(
    "retention_policy_events",
    "1",
    [
        pa.field("event_id", pa.string()),
        pa.field("policy_id", pa.string()),
        pa.field("policy_digest", pa.string()),
        pa.field("event_type", pa.string()),
        pa.field("from_status", pa.string()),
        pa.field("to_status", pa.string()),
        pa.field("revision", pa.int64()),
        pa.field("artifact_count", pa.int64()),
        pa.field("actor", pa.string()),
        pa.field("approver", pa.string()),
        pa.field("detail", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

# Durable external-context catalog (backlog 0114). One idempotent row per external
# run/job context observed on a canonical execution or transform, keyed by
# ``context_id`` (a content digest over the source row + external handle fields) so
# re-running the backfill maps to the same row instead of duplicating. External
# handles (``provider`` / ``external_run_id`` / ``external_job_id`` / ``code_ref`` /
# ``environment_digest``) are surfaced as columns for filter pushdown; the resolved
# canonical anchors (``execution_id`` / ``artifact_ids`` / ``transform_id``) keep
# Lance IDs authoritative. ``context_json`` stores the (optionally redaction-applied)
# context payload for reload without re-tracing. Retention/redaction governance is
# carried inline and the table is excluded from the lineage graph projection.
EXTERNAL_CONTEXTS_SCHEMA = _schema(
    "external_contexts",
    "1",
    [
        pa.field("context_id", pa.string()),
        pa.field("catalog_schema_version", pa.string()),
        pa.field("context_schema_version", pa.string()),
        pa.field("lake_uri", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("external_run_id", pa.string()),
        pa.field("external_job_id", pa.string()),
        pa.field("external_parent_run_id", pa.string()),
        pa.field("external_url", pa.string()),
        pa.field("code_ref", pa.string()),
        pa.field("environment_digest", pa.string()),
        pa.field("artifact_uris", pa.list_(pa.string())),
        pa.field("source_table", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("execution_id", pa.string()),
        pa.field("artifact_ids", pa.list_(pa.string())),
        pa.field("transform_id", pa.string()),
        pa.field("status", pa.string()),
        pa.field("redacted", pa.bool_()),
        pa.field("redaction_policy", pa.string()),
        pa.field("context_json", pa.string()),
        pa.field("retention_policy", pa.string()),
        pa.field("protected", pa.bool_()),
        pa.field("expires_at", pa.timestamp("us", tz="UTC")),
        pa.field("legal_hold", pa.bool_()),
        pa.field("audit_hold", pa.bool_()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
        pa.field("updated_at", pa.timestamp("us", tz="UTC")),
    ],
)

# Append-only audit log for external-context lifecycle (backlog 0114): backfill /
# manual recording, retention updates, redaction, and expiry. Survives row
# expiry and is excluded from the lineage graph projection.
EXTERNAL_CONTEXT_EVENTS_SCHEMA = _schema(
    "external_context_events",
    "1",
    [
        pa.field("event_id", pa.string()),
        pa.field("context_id", pa.string()),
        pa.field("event_type", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("external_run_id", pa.string()),
        pa.field("detail", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

# Durable Enterprise training loader/backend report catalog (backlog 0115). One
# idempotent row per emitted report, keyed by ``report_id`` (== "trpt-<digest>",
# content-addressed over the report + backend payload). Persists the in-memory
# 0069/0073 reports as queryable run history: compare cold/warm epochs, audit
# fallbacks, and reload the full report without re-running the loader. Scalar
# columns carry the queryable dimensions (snapshot, run, backend kind, fallback
# reason/transition, cache policy, prewarm); ``report_json``/``backend_json``
# carry the full payloads for point reload. Aggregatable counters (cache
# hits/misses, bytes read) are materialized so cross-worker/epoch sums are a
# bounded column projection, never a payload scan. Sibling to the other training
# manifests: single table, idempotent replace, reference-based retention.
TRAINING_REPORTS_SCHEMA = _schema(
    "training_reports",
    "1",
    [
        pa.field("report_id", pa.string()),
        pa.field("report_digest", pa.string()),
        pa.field("catalog_schema_version", pa.string()),
        pa.field("report_schema_version", pa.string()),
        pa.field("lake_uri", pa.string()),
        pa.field("loader_kind", pa.string()),
        # Run / snapshot / plan identity (queryable).
        pa.field("training_run_id", pa.string()),
        pa.field("model_run_id", pa.string()),
        pa.field("model_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("snapshot_name", pa.string()),
        pa.field("alignment_id", pa.string()),
        pa.field("alignment_name", pa.string()),
        pa.field("table_versions", _table_versions()),
        pa.field("row_plan_id", pa.string()),
        pa.field("tick_plan_id", pa.string()),
        pa.field("epoch_plan_id", pa.string()),
        pa.field("epoch", pa.int64()),
        pa.field("worker_id", pa.int64()),
        pa.field("num_workers", pa.int64()),
        # Backend / capability identity (queryable).
        pa.field("requested_backend", pa.string()),
        pa.field("resolved_backend", pa.string()),
        pa.field("connection_kind", pa.string()),
        pa.field("execution_mode", pa.string()),
        pa.field("cache_policy", pa.string()),
        pa.field("prewarm_requested", pa.bool_()),
        pa.field("prewarm_status", pa.string()),
        # Fallback (queryable by reason and backend transition).
        pa.field("fallback", pa.bool_()),
        pa.field("fallback_reason", pa.string()),
        pa.field("fallback_from_backend", pa.string()),
        pa.field("fallback_to_backend", pa.string()),
        # Aggregatable telemetry counters (materialized for bounded sums).
        pa.field("cache_hits", pa.int64()),
        pa.field("cache_misses", pa.int64()),
        pa.field("bytes_read", pa.int64()),
        pa.field("rows_hydrated", pa.int64()),
        pa.field("pe_fanout", pa.int64()),
        # Full payloads for point reload without re-running the loader.
        pa.field("report_json", pa.string()),
        pa.field("backend_json", pa.string()),
        pa.field("status", pa.string()),
        pa.field("metadata", _kv()),
        pa.field("created_by", pa.string()),
        _CREATED_AT,
    ],
)

# Creation order: referenced-before-referencing reads naturally in listings.
TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "integration_sources": INTEGRATION_SOURCES_SCHEMA,
    "runs": RUNS_SCHEMA,
    "episodes": EPISODES_SCHEMA,
    "observations": OBSERVATIONS_SCHEMA,
    "videos": VIDEOS_SCHEMA,
    "video_encodings": VIDEO_ENCODINGS_SCHEMA,
    "keyframe_map_artifacts": KEYFRAME_MAP_ARTIFACTS_SCHEMA,
    "attachments": ATTACHMENTS_SCHEMA,
    "events": EVENTS_SCHEMA,
    "scenarios": SCENARIOS_SCHEMA,
    "dataset_snapshots": DATASET_SNAPSHOTS_SCHEMA,
    "training_runs": TRAINING_RUNS_SCHEMA,
    "model_artifacts": MODEL_ARTIFACTS_SCHEMA,
    "evaluation_runs": EVALUATION_RUNS_SCHEMA,
    "evaluation_run_metrics": EVALUATION_RUN_METRICS_SCHEMA,
    "training_reports": TRAINING_REPORTS_SCHEMA,
    "curation_views": CURATION_VIEWS_SCHEMA,
    "curation_view_membership_chunks": CURATION_VIEW_MEMBERSHIP_CHUNKS_SCHEMA,
    "curation_memberships": CURATION_MEMBERSHIPS_SCHEMA,
    "curation_review_queues": CURATION_REVIEW_QUEUES_SCHEMA,
    "curation_materializations": CURATION_MATERIALIZATIONS_SCHEMA,
    "curation_comparisons": CURATION_COMPARISONS_SCHEMA,
    "distribution_catalog": DISTRIBUTION_CATALOG_SCHEMA,
    "labels": LABELS_SCHEMA,
    "model_outputs": MODEL_OUTPUTS_SCHEMA,
    "eval_metric_catalog": EVAL_METRIC_CATALOG_SCHEMA,
    "feedback": FEEDBACK_SCHEMA,
    "alignment_jobs": ALIGNMENT_JOBS_SCHEMA,
    "aligned_frames": ALIGNED_FRAMES_SCHEMA,
    "aligned_ticks": ALIGNED_TICKS_SCHEMA,
    "transform_runs": TRANSFORM_RUNS_SCHEMA,
    "lerobot_ingest_checkpoints": LEROBOT_INGEST_CHECKPOINTS_SCHEMA,
    "lerobot_checkpoint_holds": LEROBOT_CHECKPOINT_HOLDS_SCHEMA,
    "lineage_artifacts": LINEAGE_ARTIFACTS_SCHEMA,
    "lineage_executions": LINEAGE_EXECUTIONS_SCHEMA,
    "lineage_edges": LINEAGE_EDGES_SCHEMA,
    "lineage_delivery_attempts": LINEAGE_DELIVERY_ATTEMPTS_SCHEMA,
    "lineage_audit_reports": LINEAGE_AUDIT_REPORTS_SCHEMA,
    "evidence_packs": EVIDENCE_PACKS_SCHEMA,
    "evidence_pack_events": EVIDENCE_PACK_EVENTS_SCHEMA,
    "rebuild_plans": REBUILD_PLANS_SCHEMA,
    "rebuild_plan_events": REBUILD_PLAN_EVENTS_SCHEMA,
    "retention_policies": RETENTION_POLICIES_SCHEMA,
    "retention_policy_events": RETENTION_POLICY_EVENTS_SCHEMA,
    "external_contexts": EXTERNAL_CONTEXTS_SCHEMA,
    "external_context_events": EXTERNAL_CONTEXT_EVENTS_SCHEMA,
}

CANONICAL_TABLES: tuple[str, ...] = tuple(TABLE_SCHEMAS)

SCHEMA_VERSIONS: dict[str, str] = {
    name: schema.metadata[SCHEMA_METADATA_VERSION_KEY.encode()].decode()
    for name, schema in TABLE_SCHEMAS.items()
}
