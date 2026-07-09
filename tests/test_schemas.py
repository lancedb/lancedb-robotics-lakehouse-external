"""Schema contract tests for the canonical lake tables (backlog 0002).

These pin the v0 contract: every canonical table exists in the manifest,
carries schema-version metadata, and has the required ID, timestamp,
raw-provenance, and lineage fields.
"""

import pyarrow as pa
import pytest

from lancedb_robotics.schemas import (
    CANONICAL_TABLES,
    SCHEMA_METADATA_TABLE_KEY,
    SCHEMA_METADATA_VERSION_KEY,
    SCHEMA_VERSIONS,
    TABLE_SCHEMAS,
)

# The canonical substrate tables from the PRD MVP. This list is the contract:
# changing it is a product decision, not a refactor.
EXPECTED_TABLES = [
    "integration_sources",
    "runs",
    "episodes",
    "observations",
    "videos",
    "video_encodings",
    "keyframe_map_artifacts",
    "keyframe_map_artifact_referrers",
    "attachments",
    "events",
    "scenarios",
    "dataset_snapshots",
    "training_runs",
    "model_artifacts",
    "evaluation_runs",
    "evaluation_run_metrics",
    "training_reports",
    "curation_views",
    "curation_view_membership_chunks",
    "curation_memberships",
    "curation_review_queues",
    "curation_materializations",
    "curation_comparisons",
    "distribution_catalog",
    "labels",
    "model_outputs",
    "eval_metric_catalog",
    "feedback",
    "alignment_jobs",
    "aligned_frames",
    "aligned_ticks",
    "transform_runs",
    "lerobot_ingest_checkpoints",
    "lerobot_checkpoint_holds",
    "lineage_artifacts",
    "lineage_executions",
    "lineage_edges",
    "lineage_delivery_attempts",
    "lineage_audit_reports",
    "evidence_packs",
    "evidence_pack_events",
    "rebuild_plans",
    "rebuild_plan_events",
    "retention_policies",
    "retention_policy_events",
    "external_contexts",
    "external_context_events",
]

# Primary-key column per table.
ID_COLUMNS = {
    "integration_sources": "source_id",
    "runs": "run_id",
    "episodes": "episode_id",
    "observations": "observation_id",
    "videos": "video_id",
    "video_encodings": "encoding_id",
    "keyframe_map_artifacts": "artifact_id",
    "keyframe_map_artifact_referrers": "referrer_id",
    "attachments": "attachment_id",
    "events": "event_id",
    "scenarios": "scenario_id",
    "dataset_snapshots": "dataset_id",
    "training_runs": "training_run_id",
    "model_artifacts": "model_artifact_id",
    "evaluation_runs": "eval_run_id",
    "evaluation_run_metrics": "metric_row_id",
    "training_reports": "report_id",
    "curation_views": "view_id",
    "curation_view_membership_chunks": "chunk_id",
    "curation_memberships": "membership_id",
    "curation_review_queues": "queue_item_id",
    "curation_materializations": "materialization_id",
    "curation_comparisons": "comparison_id",
    "distribution_catalog": "catalog_id",
    "labels": "label_id",
    "model_outputs": "model_output_id",
    "eval_metric_catalog": "model_output_id",
    "feedback": "feedback_id",
    "alignment_jobs": "alignment_id",
    "aligned_frames": "aligned_frame_id",
    "aligned_ticks": "aligned_tick_id",
    "transform_runs": "transform_id",
    "lerobot_ingest_checkpoints": "checkpoint_id",
    "lerobot_checkpoint_holds": "hold_id",
    "lineage_artifacts": "artifact_id",
    "lineage_executions": "execution_id",
    "lineage_edges": "edge_id",
    "lineage_delivery_attempts": "attempt_id",
    "lineage_audit_reports": "report_id",
    "evidence_packs": "pack_id",
    "evidence_pack_events": "event_id",
    "rebuild_plans": "plan_id",
    "rebuild_plan_events": "event_id",
    "retention_policies": "policy_id",
    "retention_policy_events": "event_id",
    "external_contexts": "context_id",
    "external_context_events": "event_id",
}

# Tables whose rows are produced by a transform and must point back to it.
LINEAGE_TABLES = [
    "runs",
    "episodes",
    "observations",
    "videos",
    "video_encodings",
    "keyframe_map_artifacts",
    "keyframe_map_artifact_referrers",
    "attachments",
    "events",
    "scenarios",
    "dataset_snapshots",
    "training_runs",
    "model_artifacts",
    "evaluation_runs",
    "curation_views",
    "curation_view_membership_chunks",
    "curation_memberships",
    "curation_review_queues",
    "curation_materializations",
    "curation_comparisons",
    "distribution_catalog",
    "labels",
    "model_outputs",
    "feedback",
    "alignment_jobs",
    "aligned_frames",
    "aligned_ticks",
    "lerobot_ingest_checkpoints",
]


def test_canonical_table_list_matches_contract():
    assert list(CANONICAL_TABLES) == EXPECTED_TABLES
    assert set(TABLE_SCHEMAS) == set(EXPECTED_TABLES)
    assert set(SCHEMA_VERSIONS) == set(EXPECTED_TABLES)


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_schema_is_arrow_schema(table):
    assert isinstance(TABLE_SCHEMAS[table], pa.Schema)


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_schema_carries_version_metadata(table):
    metadata = TABLE_SCHEMAS[table].metadata
    assert metadata[SCHEMA_METADATA_TABLE_KEY.encode()] == table.encode()
    version = metadata[SCHEMA_METADATA_VERSION_KEY.encode()].decode()
    assert version == SCHEMA_VERSIONS[table]


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_id_column_exists(table):
    schema = TABLE_SCHEMAS[table]
    id_col = ID_COLUMNS[table]
    assert schema.field(id_col).type == pa.string()


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_created_at_timestamp_exists(table):
    field = TABLE_SCHEMAS[table].field("created_at")
    assert pa.types.is_timestamp(field.type)


@pytest.mark.parametrize("table", LINEAGE_TABLES)
def test_lineage_field_exists(table):
    assert TABLE_SCHEMAS[table].field("transform_id").type == pa.string()


def test_runs_raw_provenance_fields():
    schema = TABLE_SCHEMAS["runs"]
    assert schema.field("raw_uri").type == pa.string()
    assert schema.field("source_id").type == pa.string()
    assert schema.field("start_time_ns").type == pa.int64()
    assert schema.field("end_time_ns").type == pa.int64()


def test_observations_raw_provenance_fields():
    schema = TABLE_SCHEMAS["observations"]
    assert schema.field("run_id").type == pa.string()
    assert schema.field("timestamp_ns").type == pa.int64()
    # Flat raw_ref fields: source URI + channel/topic + time/sequence identity.
    assert schema.field("raw_uri").type == pa.string()
    assert schema.field("raw_channel").type == pa.string()
    assert schema.field("raw_log_time_ns").type == pa.int64()
    assert schema.field("raw_sequence").type == pa.int64()


def test_observations_is_schema_v4():
    # Payload columns (backlog 0014) bumped v1 -> v2; blob-encoding payload_blob
    # (backlog 0035 / decision 0024) is a storage-format change: v2 -> v3;
    # stable episode/frame indices and denormalized frame-grain scalars
    # (backlog 0029 / decision 0025) bump v3 -> v4.
    assert SCHEMA_VERSIONS["observations"] == "4"


def test_observations_episode_frame_fields():
    schema = TABLE_SCHEMAS["observations"]
    assert schema.field("episode_id").type == pa.string()
    assert schema.field("episode_index").type == pa.int64()
    assert schema.field("frame_index").type == pa.int64()
    assert schema.field("robot_id").type == pa.string()
    assert schema.field("site_id").type == pa.string()
    assert schema.field("task_id").type == pa.string()
    assert schema.field("software_version").type == pa.string()
    assert schema.field("outcome").type == pa.string()


def test_observations_payload_columns():
    schema = TABLE_SCHEMAS["observations"]
    assert schema.field("payload_json").type == pa.string()
    # payload_blob stays large_binary at the type level, but is blob-encoded via
    # field metadata (decision 0024): a metadata scan reads no blob bytes.
    assert schema.field("payload_blob").type == pa.large_binary()
    assert schema.field("payload_blob").metadata == {b"lance-encoding:blob": b"true"}
    assert schema.field("message_encoding").type == pa.string()
    assert schema.field("schema_encoding").type == pa.string()
    assert schema.field("decode_status").type == pa.string()
    assert schema.field("decode_error").type == pa.string()


def test_attachments_manifest_fields():
    # Backlog 0016: manifest (name/media_type/size/sha256) + recoverable bytes,
    # linked to the run. Bytes are Lance blob-encoded (backlog 0035 / decision
    # 0024), matching the observations.payload_blob precedent.
    schema = TABLE_SCHEMAS["attachments"]
    assert schema.field("run_id").type == pa.string()
    assert schema.field("name").type == pa.string()
    assert schema.field("media_type").type == pa.string()
    assert schema.field("size").type == pa.int64()
    assert schema.field("sha256").type == pa.string()
    assert schema.field("log_time_ns").type == pa.int64()
    assert schema.field("create_time_ns").type == pa.int64()
    assert schema.field("data").type == pa.large_binary()
    assert schema.field("data").metadata == {b"lance-encoding:blob": b"true"}


def test_episodes_first_class_fields():
    schema = TABLE_SCHEMAS["episodes"]
    assert schema.field("run_id").type == pa.string()
    assert schema.field("episode_index").type == pa.int64()
    assert schema.field("from_timestamp_ns").type == pa.int64()
    assert schema.field("to_timestamp_ns").type == pa.int64()
    assert schema.field("boundary_source").type == pa.string()
    assert schema.field("outcome").type == pa.string()
    assert schema.field("frame_count").type == pa.int64()
    assert pa.types.is_list(schema.field("camera_blobs").type)
    assert schema.field("task_id").type == pa.string()
    assert pa.types.is_list(schema.field("embedding").type)
    assert schema.field("provenance").type == pa.string()


def test_videos_frame_handle_fields():
    schema = TABLE_SCHEMAS["videos"]
    assert schema.field("run_id").type == pa.string()
    assert schema.field("episode_id").type == pa.string()
    assert schema.field("episode_index").type == pa.int64()
    assert schema.field("camera_key").type == pa.string()
    assert schema.field("sensor_id").type == pa.string()
    assert schema.field("topic").type == pa.string()
    assert schema.field("frame_count").type == pa.int64()
    assert pa.types.is_list(schema.field("observation_ids").type)
    assert schema.field("codec").type == pa.string()


def test_video_encodings_codec_aware_fields():
    schema = TABLE_SCHEMAS["video_encodings"]
    assert schema.field("video_id").type == pa.string()
    assert schema.field("episode_id").type == pa.string()
    assert schema.field("episode_index").type == pa.int64()
    assert schema.field("camera_key").type == pa.string()
    assert schema.field("codec").type == pa.string()
    assert schema.field("gop_size").type == pa.int64()
    assert schema.field("resolution").type == pa.string()
    assert schema.field("fps").type == pa.float64()
    assert schema.field("frame_count").type == pa.int64()
    assert schema.field("keyframe_map_ref").type == pa.string()
    assert schema.field("keyframe_map_json").type == pa.string()
    assert schema.field("nvdec_compatible").type == pa.bool_()
    assert schema.field("source_size_bytes").type == pa.int64()
    assert schema.field("encoded_size_bytes").type == pa.int64()
    assert schema.field("data").type == pa.large_binary()
    assert schema.field("data").metadata == {b"lance-encoding:blob": b"true"}


def test_keyframe_map_artifact_referrer_fields():
    schema = TABLE_SCHEMAS["keyframe_map_artifact_referrers"]
    assert schema.field("artifact_id").type == pa.string()
    assert schema.field("keyframe_map_ref").type == pa.string()
    assert schema.field("content_sha256").type == pa.string()
    assert schema.field("referrer_kind").type == pa.string()
    assert schema.field("referrer_table").type == pa.string()
    assert schema.field("referrer_table_version").type == pa.int64()
    assert schema.field("source_video_fingerprint").type == pa.string()
    assert schema.field("inspection_id").type == pa.string()
    assert schema.field("encoding_id").type == pa.string()
    assert schema.field("video_id").type == pa.string()
    assert schema.field("run_id").type == pa.string()
    assert schema.field("episode_id").type == pa.string()
    assert schema.field("episode_index").type == pa.int64()
    assert schema.field("camera_key").type == pa.string()


def test_alignment_jobs_fields():
    schema = TABLE_SCHEMAS["alignment_jobs"]
    assert schema.field("name").type == pa.string()
    assert pa.types.is_list(schema.field("input_tables").type)
    assert pa.types.is_list(schema.field("input_versions").type)
    assert pa.types.is_list(schema.field("streams").type)
    assert schema.field("clock").type == pa.string()
    assert schema.field("rate_hz").type == pa.float64()
    assert schema.field("tolerance_ms").type == pa.float64()
    assert schema.field("recipe").type == pa.string()
    assert schema.field("output_table").type == pa.string()
    assert schema.field("quality_summary").type == pa.string()
    assert pa.types.is_list(schema.field("quality_flags").type)


def test_aligned_frames_fields():
    schema = TABLE_SCHEMAS["aligned_frames"]
    assert schema.field("alignment_id").type == pa.string()
    assert schema.field("run_id").type == pa.string()
    assert schema.field("tick_index").type == pa.int64()
    assert schema.field("timestamp_ns").type == pa.int64()
    assert schema.field("stream").type == pa.string()
    assert schema.field("status").type == pa.string()
    assert schema.field("interpolation").type == pa.string()
    assert schema.field("observation_id").type == pa.string()
    assert pa.types.is_list(schema.field("source_observation_ids").type)
    assert pa.types.is_list(schema.field("source_row_ids").type)
    assert schema.field("source_time_ns").type == pa.int64()
    assert schema.field("absolute_error_ns").type == pa.int64()
    assert schema.field("confidence").type == pa.float64()
    assert schema.field("value_json").type == pa.string()
    assert pa.types.is_list(schema.field("quality_flags").type)


def test_aligned_ticks_fields():
    schema = TABLE_SCHEMAS["aligned_ticks"]
    assert schema.field("aligned_tick_id").type == pa.string()
    assert schema.field("alignment_id").type == pa.string()
    assert schema.field("alignment_name").type == pa.string()
    assert schema.field("recipe_digest").type == pa.string()
    assert schema.field("run_id").type == pa.string()
    assert schema.field("tick_index").type == pa.int64()
    assert schema.field("timestamp_ns").type == pa.int64()
    assert pa.types.is_list(schema.field("available_streams").type)
    assert pa.types.is_list(schema.field("missing_streams").type)
    assert pa.types.is_list(schema.field("interpolated_streams").type)
    assert pa.types.is_list(schema.field("out_of_tolerance_streams").type)
    assert schema.field("has_missing").type == pa.bool_()
    assert schema.field("has_out_of_tolerance").type == pa.bool_()
    assert schema.field("min_confidence").type == pa.float64()
    assert pa.types.is_list(schema.field("quality_flags").type)
    assert schema.field("stream_detail_json").type == pa.json_(pa.string())
    assert schema.field("masks_json").type == pa.json_(pa.string())
    assert schema.field("stream_values_json").type == pa.json_(pa.string())
    assert schema.field("lineage_json").type == pa.json_(pa.string())


def test_attachments_is_schema_v2():
    # Blob-encoding data (backlog 0035 / decision 0024) is a storage-format
    # change: v1 -> v2.
    assert SCHEMA_VERSIONS["attachments"] == "2"


def test_events_window_fields():
    schema = TABLE_SCHEMAS["events"]
    for col in ("timestamp_ns", "start_time_ns", "end_time_ns"):
        assert schema.field(col).type == pa.int64()


def test_scenarios_window_and_parentage():
    schema = TABLE_SCHEMAS["scenarios"]
    assert schema.field("start_time_ns").type == pa.int64()
    assert schema.field("end_time_ns").type == pa.int64()
    assert schema.field("window_ns").type == pa.int64()
    assert schema.field("is_partial").type == pa.bool_()
    assert pa.types.is_list(schema.field("topics").type)
    assert pa.types.is_list(schema.field("observation_ids").type)
    assert schema.field("observation_count").type == pa.int64()
    assert schema.field("trigger_event_id").type == pa.string()
    assert schema.field("parent_scenario_id").type == pa.string()
    # Enrichment writes summary text here (backlog 0007); embedding vectors are
    # added as a separate column at enrich time, not part of this contract.
    assert schema.field("summary").type == pa.string()


def test_dataset_snapshots_reproducibility_fields():
    schema = TABLE_SCHEMAS["dataset_snapshots"]
    assert schema.field("query_spec").type == pa.string()
    table_versions = schema.field("table_versions").type
    assert pa.types.is_list(table_versions)
    struct = table_versions.value_type
    assert struct.field("table").type == pa.string()
    assert struct.field("version").type == pa.int64()
    assert struct.field("tag").type == pa.string()


def test_training_run_manifest_fields():
    schema = TABLE_SCHEMAS["training_runs"]
    assert schema.field("dataset_id").type == pa.string()
    assert schema.field("snapshot_name").type == pa.string()
    assert schema.field("snapshot_tag").type == pa.string()
    assert pa.types.is_list(schema.field("table_versions").type)
    assert schema.field("row_plan_id").type == pa.string()
    assert schema.field("epoch_plan_id").type == pa.string()
    assert pa.types.is_list(schema.field("projection_manifest_ids").type)
    assert schema.field("code_ref").type == pa.string()
    assert schema.field("package_versions_json").type == pa.string()
    assert schema.field("environment_json").type == pa.string()
    assert schema.field("hardware_json").type == pa.string()
    assert schema.field("runtime_json").type == pa.string()
    assert schema.field("hyperparameters_json").type == pa.string()
    assert schema.field("random_seeds_json").type == pa.string()
    assert schema.field("split_policy_json").type == pa.string()
    assert pa.types.is_list(schema.field("external_refs").type)
    assert schema.field("manifest_digest").type == pa.string()
    assert schema.field("created_by").type == pa.string()


def test_model_artifact_manifest_fields():
    schema = TABLE_SCHEMAS["model_artifacts"]
    assert schema.field("training_run_id").type == pa.string()
    assert schema.field("artifact_uri").type == pa.string()
    assert schema.field("checksum").type == pa.string()
    assert pa.types.is_list(schema.field("aliases").type)
    assert schema.field("framework").type == pa.string()
    assert schema.field("epoch").type == pa.int64()
    assert schema.field("step").type == pa.int64()
    assert schema.field("metrics_json").type == pa.string()
    assert pa.types.is_list(schema.field("metadata").type)
    assert pa.types.is_list(schema.field("external_refs").type)
    assert schema.field("manifest_digest").type == pa.string()
    assert schema.field("created_by").type == pa.string()


def test_evaluation_run_manifest_fields():
    schema = TABLE_SCHEMAS["evaluation_runs"]
    assert schema.field("model_artifact_id").type == pa.string()
    assert schema.field("training_run_id").type == pa.string()
    assert schema.field("dataset_id").type == pa.string()
    assert schema.field("snapshot_name").type == pa.string()
    assert schema.field("snapshot_tag").type == pa.string()
    assert pa.types.is_list(schema.field("table_versions").type)
    assert schema.field("metrics_json").type == pa.string()
    assert schema.field("slice_metrics_json").type == pa.string()
    assert schema.field("failure_outputs_json").type == pa.string()
    assert schema.field("code_ref").type == pa.string()
    assert schema.field("package_versions_json").type == pa.string()
    assert schema.field("environment_json").type == pa.string()
    assert schema.field("hardware_json").type == pa.string()
    assert schema.field("runtime_json").type == pa.string()
    assert pa.types.is_list(schema.field("external_refs").type)
    assert schema.field("manifest_digest").type == pa.string()
    assert schema.field("created_by").type == pa.string()


def test_curation_view_fields():
    schema = TABLE_SCHEMAS["curation_views"]
    assert schema.field("name").type == pa.string()
    assert schema.field("owner").type == pa.string()
    assert pa.types.is_list(schema.field("tags").type)
    assert schema.field("source_kind").type == pa.string()
    assert schema.field("scope").type == pa.string()
    assert schema.field("query_spec").type == pa.string()
    assert pa.types.is_list(schema.field("scenario_ids").type)
    assert pa.types.is_list(schema.field("table_versions").type)
    assert pa.types.is_list(schema.field("parent_transform_ids").type)
    assert schema.field("status").type == pa.string()


def test_curation_view_membership_chunk_fields():
    schema = TABLE_SCHEMAS["curation_view_membership_chunks"]
    assert schema.field("view_id").type == pa.string()
    assert schema.field("chunk_index").type == pa.int64()
    assert schema.field("start_ordinal").type == pa.int64()
    assert schema.field("end_ordinal").type == pa.int64()
    assert pa.types.is_list(schema.field("scenario_ids").type)
    assert schema.field("scenario_count").type == pa.int64()
    assert schema.field("chunk_digest").type == pa.string()
    assert schema.field("created_by").type == pa.string()


def test_curation_membership_fields():
    schema = TABLE_SCHEMAS["curation_memberships"]
    assert schema.field("view_id").type == pa.string()
    assert schema.field("target_grain").type == pa.string()
    assert schema.field("target_id").type == pa.string()
    assert schema.field("scenario_id").type == pa.string()
    assert schema.field("decision").type == pa.string()
    assert schema.field("reason_code").type == pa.string()
    assert schema.field("reason").type == pa.string()
    assert schema.field("note").type == pa.string()
    assert schema.field("reviewer").type == pa.string()
    assert schema.field("queue").type == pa.string()
    assert schema.field("priority").type == pa.int64()
    assert schema.field("score").type == pa.float64()
    assert pa.types.is_list(schema.field("metadata").type)
    assert schema.field("supersedes_membership_id").type == pa.string()
    assert schema.field("created_by").type == pa.string()


def test_curation_review_queue_fields():
    schema = TABLE_SCHEMAS["curation_review_queues"]
    assert schema.field("queue_id").type == pa.string()
    assert schema.field("queue_name").type == pa.string()
    assert schema.field("target_grain").type == pa.string()
    assert schema.field("target_id").type == pa.string()
    assert schema.field("scenario_id").type == pa.string()
    assert schema.field("source_operation").type == pa.string()
    assert schema.field("source_ref").type == pa.string()
    assert schema.field("priority").type == pa.int64()
    assert schema.field("priority_score").type == pa.float64()
    assert schema.field("priority_reason").type == pa.string()
    assert schema.field("assignee").type == pa.string()
    assert schema.field("status").type == pa.string()
    assert schema.field("export_uri").type == pa.string()
    assert schema.field("external_task_id").type == pa.string()
    assert schema.field("external_url").type == pa.string()
    assert pa.types.is_list(schema.field("metadata").type)
    assert pa.types.is_list(schema.field("table_versions").type)
    assert pa.types.is_list(schema.field("source_transform_ids").type)
    assert schema.field("created_by").type == pa.string()


def test_curation_materialization_fields():
    schema = TABLE_SCHEMAS["curation_materializations"]
    assert schema.field("dataset_id").type == pa.string()
    assert schema.field("snapshot_name").type == pa.string()
    assert schema.field("target_format").type == pa.string()
    assert schema.field("output_uri").type == pa.string()
    assert schema.field("selected_scenario_count").type == pa.int64()
    assert schema.field("selected_observation_count").type == pa.int64()
    assert schema.field("total_payload_bytes").type == pa.int64()
    assert schema.field("copied_payload_bytes").type == pa.int64()
    assert schema.field("logical_reference_bytes").type == pa.int64()
    assert schema.field("copy_ratio").type == pa.float64()
    assert schema.field("report_json").type == pa.string()


def test_curation_comparison_fields():
    schema = TABLE_SCHEMAS["curation_comparisons"]
    assert schema.field("left_dataset_id").type == pa.string()
    assert schema.field("right_dataset_id").type == pa.string()
    assert schema.field("left_snapshot_name").type == pa.string()
    assert schema.field("right_snapshot_name").type == pa.string()
    assert pa.types.is_list(schema.field("metrics").type)
    assert pa.types.is_list(schema.field("dimensions").type)
    assert schema.field("added_scenario_count").type == pa.int64()
    assert schema.field("removed_scenario_count").type == pa.int64()
    assert schema.field("shared_scenario_count").type == pa.int64()
    assert schema.field("report_json").type == pa.string()
    assert pa.types.is_list(schema.field("table_versions").type)
    # Backlog 0093: catalog / retention lifecycle columns.
    assert schema.field("pair_alias").type == pa.string()
    assert schema.field("state").type == pa.string()
    assert schema.field("report_sha1").type == pa.string()
    assert schema.field("report_bytes").type == pa.int64()
    assert schema.field("retention_policy_json").type == pa.string()
    assert pa.types.is_timestamp(schema.field("archived_at").type)
    assert pa.types.is_timestamp(schema.field("pruned_at").type)
    assert schema.metadata[SCHEMA_METADATA_VERSION_KEY.encode()] == b"2"


def test_distribution_catalog_fields():
    schema = TABLE_SCHEMAS["distribution_catalog"]
    assert schema.field("kind").type == pa.string()
    assert schema.field("name").type == pa.string()
    assert schema.field("spec_id").type == pa.string()
    assert schema.field("report_id").type == pa.string()
    assert schema.field("comparison_id").type == pa.string()
    assert schema.field("finding_id").type == pa.string()
    assert schema.field("source_kind").type == pa.string()
    assert schema.field("source_name").type == pa.string()
    assert schema.field("source_id").type == pa.string()
    assert schema.field("dataset_id").type == pa.string()
    assert schema.field("view_id").type == pa.string()
    assert schema.field("summary_json").type == pa.string()
    assert schema.field("body_json").type == pa.string()
    assert schema.field("body_sha1").type == pa.string()
    assert schema.field("body_bytes").type == pa.int64()
    assert schema.field("body_compacted").type == pa.bool_()
    assert schema.field("retention_policy_json").type == pa.string()
    assert pa.types.is_timestamp(schema.field("expires_at").type)
    assert pa.types.is_timestamp(schema.field("compacted_at").type)
    assert pa.types.is_list(schema.field("table_versions").type)
    assert pa.types.is_list(schema.field("source_transform_ids").type)


def test_closed_loop_label_fields():
    schema = TABLE_SCHEMAS["labels"]
    assert schema.field("run_id").type == pa.string()
    assert schema.field("observation_id").type == pa.string()
    assert schema.field("scenario_id").type == pa.string()
    assert schema.field("event_id").type == pa.string()
    assert schema.field("label_type").type == pa.string()
    assert schema.field("label").type == pa.string()
    assert schema.field("label_value").type == pa.string()
    assert schema.field("label_spec").type == pa.string()
    assert schema.field("source").type == pa.string()
    assert schema.field("reviewer").type == pa.string()
    assert schema.field("confidence").type == pa.float32()
    assert schema.field("status").type == pa.string()
    assert pa.types.is_list(schema.field("metadata").type)


def test_closed_loop_model_output_fields():
    schema = TABLE_SCHEMAS["model_outputs"]
    assert schema.field("run_id").type == pa.string()
    assert schema.field("observation_id").type == pa.string()
    assert schema.field("scenario_id").type == pa.string()
    assert schema.field("dataset_id").type == pa.string()
    assert schema.field("model_version").type == pa.string()
    assert schema.field("output_type").type == pa.string()
    assert schema.field("prediction").type == pa.string()
    assert schema.field("output_json").type == pa.string()
    assert schema.field("score").type == pa.float32()
    assert schema.field("producer_run_id").type == pa.string()
    assert schema.field("source").type == pa.string()
    assert pa.types.is_list(schema.field("metadata").type)


def test_closed_loop_feedback_fields():
    schema = TABLE_SCHEMAS["feedback"]
    assert schema.field("run_id").type == pa.string()
    assert schema.field("observation_id").type == pa.string()
    assert schema.field("scenario_id").type == pa.string()
    assert schema.field("event_id").type == pa.string()
    assert schema.field("label_id").type == pa.string()
    assert schema.field("model_output_id").type == pa.string()
    assert schema.field("feedback_type").type == pa.string()
    assert schema.field("severity").type == pa.string()
    assert schema.field("linked_incident_id").type == pa.string()
    assert schema.field("notes").type == pa.string()
    assert schema.field("source").type == pa.string()
    assert schema.field("status").type == pa.string()
    assert pa.types.is_list(schema.field("metadata").type)


def test_simulation_runs_remains_deferred_stub():
    assert "simulation_runs" not in TABLE_SCHEMAS


def test_transform_runs_lineage_fields():
    schema = TABLE_SCHEMAS["transform_runs"]
    assert schema.field("kind").type == pa.string()
    assert pa.types.is_list(schema.field("input_uris").type)
    assert pa.types.is_list(schema.field("output_tables").type)
    assert schema.field("params").type == pa.string()
    assert schema.field("status").type == pa.string()


def test_lineage_artifact_graph_fields():
    schema = TABLE_SCHEMAS["lineage_artifacts"]
    assert schema.field("kind").type == pa.string()
    assert schema.field("name").type == pa.string()
    assert schema.field("table_name").type == pa.string()
    assert schema.field("table_version").type == pa.int64()
    assert schema.field("table_tag").type == pa.string()
    assert schema.field("row_grain").type == pa.string()
    assert pa.types.is_list(schema.field("row_ids").type)
    assert schema.field("source_uri").type == pa.string()
    assert schema.field("source_id").type == pa.string()
    assert schema.field("digest").type == pa.string()
    assert schema.field("producer_execution_id").type == pa.string()
    assert pa.types.is_list(schema.field("metadata").type)


def test_lineage_execution_graph_fields():
    schema = TABLE_SCHEMAS["lineage_executions"]
    assert schema.field("kind").type == pa.string()
    assert schema.field("name").type == pa.string()
    assert schema.field("transform_id").type == pa.string()
    assert schema.field("status").type == pa.string()
    assert schema.field("params_json").type == pa.string()
    assert schema.field("code_ref").type == pa.string()
    assert schema.field("provider").type == pa.string()
    assert schema.field("environment_json").type == pa.string()
    assert pa.types.is_list(schema.field("input_artifact_ids").type)
    assert pa.types.is_list(schema.field("output_artifact_ids").type)
    assert pa.types.is_list(schema.field("input_table_versions").type)
    assert pa.types.is_list(schema.field("output_table_versions").type)
    assert pa.types.is_timestamp(schema.field("started_at").type)
    assert pa.types.is_timestamp(schema.field("finished_at").type)
    assert schema.field("created_by").type == pa.string()
    assert pa.types.is_list(schema.field("metadata").type)


def test_lineage_edge_graph_fields():
    schema = TABLE_SCHEMAS["lineage_edges"]
    assert schema.field("edge_type").type == pa.string()
    assert schema.field("from_artifact_id").type == pa.string()
    assert schema.field("to_artifact_id").type == pa.string()
    assert schema.field("execution_id").type == pa.string()
    assert pa.types.is_list(schema.field("metadata").type)


def test_lineage_audit_report_catalog_fields():
    schema = TABLE_SCHEMAS["lineage_audit_reports"]
    assert schema.field("report_digest").type == pa.string()
    assert schema.field("catalog_schema_version").type == pa.string()
    assert schema.field("report_schema_version").type == pa.string()
    assert schema.field("status").type == pa.string()
    assert pa.types.is_list(schema.field("root_artifact_ids").type)
    assert schema.field("finding_count").type == pa.int64()
    assert schema.field("missing_source_count").type == pa.int64()
    assert schema.field("validator_statuses_json").type == pa.string()
    assert schema.field("report_json").type == pa.string()
    assert pa.types.is_list(schema.field("metadata").type)
    assert schema.field("created_by").type == pa.string()
    assert pa.types.is_timestamp(schema.field("updated_at").type)


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_no_struct_inside_struct(table):
    """Nested struct-in-struct nullity does not round-trip through Lance;
    the v0 contract keeps payload fields flat or list<struct-of-scalars>."""

    def check(dtype, depth):
        if pa.types.is_struct(dtype):
            assert depth == 0, f"struct nested inside struct in {table}"
            for field in dtype:
                check(field.type, depth + 1)
        elif pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
            check(dtype.value_type, depth)

    for field in TABLE_SCHEMAS[table]:
        check(field.type, 0)
