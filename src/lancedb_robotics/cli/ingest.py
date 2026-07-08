"""`lancedb-robotics ingest` subcommands."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import typer

from lancedb_robotics.cli.lineage_context import echo_emitted_lineage
from lancedb_robotics.lerobot_object_store_validation import (
    DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
)

ingest_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_PATH_ARGUMENT = typer.Argument(..., help="Path or object-store URI to the source recording.")
_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_BATCH_SIZE_OPTION = typer.Option(
    1024,
    "--batch-size",
    help="Observation rows flushed to the lake per batch (drop to ~100 for image/lidar-heavy logs).",
)
_VALIDATE_CRCS_OPTION = typer.Option(
    True,
    "--validate-crcs/--no-validate-crcs",
    help="Validate chunk CRCs while reading (disable on the hot path for trusted data).",
)
_COMPACT_OPTION = typer.Option(
    True,
    "--compact/--no-compact",
    help="Compact the observations grain to a healthy fragment size at end of ingest "
    "(default on; avoids the per-fragment scan tax from per-batch writes).",
)
_PRUNE_VERSIONS_OPTION = typer.Option(
    True,
    "--prune-versions/--no-prune-versions",
    help="Snapshot-safely prune the per-flush version churn after ingest (default on). "
    "Versions pinned by a live snapshot/lineage are never removed.",
)
_RETAIN_VERSIONS_OPTION = typer.Option(
    2,
    "--retain-versions",
    help="Recent versions to keep when pruning (in addition to pinned versions).",
)
_INDEX_PREDICATES_OPTION = typer.Option(
    True,
    "--index-predicates/--no-index-predicates",
    help="Build the observations scalar predicate indexes (run_id/topic/...) at end of "
    "ingest (default on; --no-index-predicates for bulk backfill, then `lake maintain`).",
)
_AUTH_REF_OPTION = typer.Option(
    None,
    "--auth-ref",
    help="Backward-compatible credential reference alias for both lake and source planes.",
)
_REMOTE_AUTH_REF_OPTION = typer.Option(
    None,
    "--remote-auth-ref",
    help="LanceDB Enterprise API credential reference for --lake db://...",
)
_STORAGE_AUTH_REF_OPTION = typer.Option(
    None,
    "--storage-auth-ref",
    help="Lake object-store credential reference.",
)
_SOURCE_AUTH_REF_OPTION = typer.Option(
    None,
    "--source-auth-ref",
    help="Raw source object credential reference; this logical ref is recorded on integration_sources.",
)
_STORAGE_OPTION = typer.Option(
    None,
    "--storage-option",
    help="Lake storage client option as key=value; repeat for endpoint_url, region, etc.",
)
_SOURCE_STORAGE_OPTION = typer.Option(
    None,
    "--source-storage-option",
    help="Raw source storage option as key=value; defaults to --storage-option when omitted.",
)
_REGION_OPTION = typer.Option(None, "--region", help="LanceDB Enterprise region.")
_HOST_OVERRIDE_OPTION = typer.Option(
    None,
    "--host-override",
    help="LanceDB Enterprise private endpoint override.",
)
_LEROBOT_CLAIM_WATCHDOG_JSON_OUT_OPTION = typer.Option(
    None,
    "--out-json",
    help="Write the watchdog report JSON to this path.",
)
_LEROBOT_CLAIM_WATCHDOG_MARKDOWN_OUT_OPTION = typer.Option(
    None,
    "--out-markdown",
    help="Write a Markdown watchdog report to this path.",
)
_LEROBOT_CLAIM_CHAOS_CHECKPOINT_ROWS_JSON_OPTION = typer.Option(
    None,
    "--checkpoint-rows-json",
    help="JSON file containing checkpoint rows, or an object with checkpoint_rows/rows.",
)
_LEROBOT_EXPECT_LATEST_CHECKPOINT_ID_OPTION = typer.Option(
    None,
    "--expected-latest-checkpoint-id",
    "--expect-latest-checkpoint-id",
    help="Require the latest LeRobot checkpoint id to still match this observed value before claiming/recovery.",
)
_LEROBOT_EXPECT_LATEST_CLAIM_TOKEN_OPTION = typer.Option(
    None,
    "--expected-latest-claim-token",
    "--expect-claim-token",
    help="Require the latest LeRobot claim token to still match this observed value before claiming/recovery.",
)
_LEROBOT_EXPECT_CHECKPOINT_INDEX_OPTION = typer.Option(
    None,
    "--expected-checkpoint-index",
    "--expect-checkpoint-index",
    help="Require the latest LeRobot checkpoint index to still match this observed value before claiming/recovery.",
)
_LEROBOT_MEDIA_INSPECTION_TIMEOUT_OPTION = typer.Option(
    None,
    "--media-inspection-timeout-seconds",
    help="Per-video LeRobot MP4 metadata inspection timeout; <=0 disables the wall-clock guard.",
)
_LEROBOT_MEDIA_INSPECTION_WORKERS_OPTION = typer.Option(
    None,
    "--media-inspection-workers",
    min=1,
    help="Maximum concurrent LeRobot media-inspection workers.",
)
_LEROBOT_MEDIA_INSPECTION_RETRIES_OPTION = typer.Option(
    0,
    "--media-inspection-retries",
    help="Retries per LeRobot video after transient MP4 metadata read failures.",
)
_LEROBOT_MEDIA_INSPECTION_RETRY_BACKOFF_OPTION = typer.Option(
    0.0,
    "--media-inspection-retry-backoff-seconds",
    help="Sleep seconds between LeRobot media-inspection retry attempts.",
)
_LEROBOT_MEDIA_INSPECTION_RETRY_POLICY_OPTION = typer.Option(
    "fixed",
    "--media-inspection-retry-policy",
    help="LeRobot media-inspection retry policy: fixed or exponential-jitter.",
)
_LEROBOT_MEDIA_INSPECTION_EXECUTION_MODE_OPTION = typer.Option(
    "thread",
    "--media-inspection-execution-mode",
    help="LeRobot media-inspection execution mode: thread or process.",
)
_LEROBOT_SOURCE_MANIFEST_CACHE_OPTION = typer.Option(
    None,
    "--source-manifest-cache",
    help="JSON cache path for object-store LeRobot source listings; omit to list directly.",
)
_LEROBOT_SOURCE_VALIDATION_POLICY_OPTION = typer.Option(
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
    "--source-validation-policy",
    help=(
        "LeRobot object-store source validation policy: metadata-only, "
        "sampled-validation, or strict-content-hash."
    ),
)
_LEROBOT_SOURCE_VALIDATION_SAMPLE_COUNT_OPTION = typer.Option(
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
    "--source-validation-sample-count",
    min=0,
    help="Objects to sample for LeRobot sampled-validation.",
)
_LEROBOT_SOURCE_VALIDATION_SAMPLE_BYTES_OPTION = typer.Option(
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
    "--source-validation-sample-bytes",
    min=1,
    help="Prefix bytes to hash from each sampled LeRobot object-store object.",
)
_LEROBOT_SOURCE_VALIDATION_STRICT_MAX_BYTES_OPTION = typer.Option(
    DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    "--strict-content-hash-max-bytes",
    min=0,
    help="Maximum total bytes strict-content-hash may read from a LeRobot object-store source.",
)
_LEROBOT_KEYFRAME_MAP_INLINE_THRESHOLD_BYTES_OPTION = typer.Option(
    65536,
    "--keyframe-map-inline-threshold-bytes",
    min=0,
    help="Keep LeRobot keyframe maps inline up to this JSON byte size; larger maps are cataloged.",
)
_LEROBOT_KEYFRAME_MAP_INLINE_THRESHOLD_FRAMES_OPTION = typer.Option(
    4096,
    "--keyframe-map-inline-threshold-frames",
    min=0,
    help="Keep LeRobot keyframe maps inline up to this frame count; larger maps are cataloged.",
)
_LEROBOT_RETENTION_STATUS_OPTION = typer.Option(
    None,
    "--status",
    help="Terminal job status eligible for retention; repeat for abandoned/completed/failed/skipped.",
)
_LEROBOT_CHECKPOINT_HOLD_STATUS_OPTION = typer.Option(
    None,
    "--status",
    help="Checkpoint status selector; repeat to match multiple statuses.",
)


@ingest_app.command("mcap")
def ingest_mcap_command(
    path: str = _PATH_ARGUMENT,
    lake: str = _LAKE_OPTION,
    batch_size: int = _BATCH_SIZE_OPTION,
    validate_crcs: bool = _VALIDATE_CRCS_OPTION,
    compact: bool = _COMPACT_OPTION,
    prune_versions: bool = _PRUNE_VERSIONS_OPTION,
    retain_versions: int = _RETAIN_VERSIONS_OPTION,
    index_predicates: bool = _INDEX_PREDICATES_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    source_auth_ref: str | None = _SOURCE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    source_storage_option: list[str] | None = _SOURCE_STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Register an MCAP file as a source and ingest it into canonical lake rows."""
    from lancedb_robotics.adapters import AdapterError
    from lancedb_robotics.ingest import ingest_mcap
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.storage import parse_storage_option_pairs

    try:
        storage_options = parse_storage_option_pairs(storage_option)
        source_storage_options = (
            parse_storage_option_pairs(source_storage_option)
            if source_storage_option is not None
            else storage_options
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        opened = Lake.open(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_options=storage_options,
            region=region,
            host_override=host_override,
        )
    except LakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    resolved_source_auth_ref = source_auth_ref or auth_ref
    try:
        report = ingest_mcap(
            opened,
            path,
            batch_size=batch_size,
            validate_crcs=validate_crcs,
            compact=compact,
            prune_versions=prune_versions,
            retain_versions=retain_versions,
            index_predicates=index_predicates,
            auth_ref=resolved_source_auth_ref,
            storage_options=source_storage_options,
        )
    except AdapterError as exc:
        # CodecUnavailableError (a missing compression codec) lands here too, with
        # an actionable "install <codec>" message.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_report(report, opened)


@ingest_app.command("rosbag")
def ingest_rosbag_command(
    path: str = _PATH_ARGUMENT,
    lake: str = _LAKE_OPTION,
    batch_size: int = _BATCH_SIZE_OPTION,
    compact: bool = _COMPACT_OPTION,
    prune_versions: bool = _PRUNE_VERSIONS_OPTION,
    retain_versions: int = _RETAIN_VERSIONS_OPTION,
    index_predicates: bool = _INDEX_PREDICATES_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    source_auth_ref: str | None = _SOURCE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    source_storage_option: list[str] | None = _SOURCE_STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Register a ROS1 `.bag` or ROS2 sqlite `.db3` source and ingest it."""
    from lancedb_robotics.adapters import AdapterError
    from lancedb_robotics.ingest import ingest_rosbag
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.storage import parse_storage_option_pairs

    try:
        storage_options = parse_storage_option_pairs(storage_option)
        source_storage_options = (
            parse_storage_option_pairs(source_storage_option)
            if source_storage_option is not None
            else None
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        opened = Lake.open(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_options=storage_options,
            region=region,
            host_override=host_override,
        )
    except LakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if source_storage_options:
        typer.echo(
            "error: --source-storage-option is not supported for rosbag ingest yet", err=True
        )
        raise typer.Exit(code=2)

    try:
        report = ingest_rosbag(
            opened,
            path,
            batch_size=batch_size,
            compact=compact,
            prune_versions=prune_versions,
            retain_versions=retain_versions,
            index_predicates=index_predicates,
            auth_ref=source_auth_ref or auth_ref,
        )
    except AdapterError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_report(report, opened)


@ingest_app.command("lerobot")
def ingest_lerobot_command(
    path: str = typer.Argument(
        ...,
        help="Local/object-store LeRobot dataset directory or HF Hub repo id.",
    ),
    lake: str = _LAKE_OPTION,
    batch_size: int = _BATCH_SIZE_OPTION,
    claim_lease_seconds: float = typer.Option(
        21600.0,
        "--claim-lease-seconds",
        help="Seconds before a LeRobot running claim is considered stale without a heartbeat.",
    ),
    claim_heartbeat_seconds: float = typer.Option(
        300.0,
        "--claim-heartbeat-seconds",
        help="Expected seconds between durable LeRobot claim heartbeats.",
    ),
    compact: bool = _COMPACT_OPTION,
    prune_versions: bool = _PRUNE_VERSIONS_OPTION,
    retain_versions: int = _RETAIN_VERSIONS_OPTION,
    index_predicates: bool = _INDEX_PREDICATES_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    source_auth_ref: str | None = _SOURCE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    source_storage_option: list[str] | None = _SOURCE_STORAGE_OPTION,
    source_manifest_cache: Path | None = _LEROBOT_SOURCE_MANIFEST_CACHE_OPTION,
    source_validation_policy: str = _LEROBOT_SOURCE_VALIDATION_POLICY_OPTION,
    source_validation_sample_count: int = _LEROBOT_SOURCE_VALIDATION_SAMPLE_COUNT_OPTION,
    source_validation_sample_bytes: int = _LEROBOT_SOURCE_VALIDATION_SAMPLE_BYTES_OPTION,
    strict_content_hash_max_bytes: int = _LEROBOT_SOURCE_VALIDATION_STRICT_MAX_BYTES_OPTION,
    media_inspection_timeout_seconds: float | None = _LEROBOT_MEDIA_INSPECTION_TIMEOUT_OPTION,
    media_inspection_workers: int | None = _LEROBOT_MEDIA_INSPECTION_WORKERS_OPTION,
    media_inspection_retries: int = _LEROBOT_MEDIA_INSPECTION_RETRIES_OPTION,
    media_inspection_retry_backoff_seconds: float = _LEROBOT_MEDIA_INSPECTION_RETRY_BACKOFF_OPTION,
    media_inspection_retry_policy: str = _LEROBOT_MEDIA_INSPECTION_RETRY_POLICY_OPTION,
    media_inspection_execution_mode: str = _LEROBOT_MEDIA_INSPECTION_EXECUTION_MODE_OPTION,
    keyframe_map_inline_threshold_bytes: int = _LEROBOT_KEYFRAME_MAP_INLINE_THRESHOLD_BYTES_OPTION,
    keyframe_map_inline_threshold_frames: int = _LEROBOT_KEYFRAME_MAP_INLINE_THRESHOLD_FRAMES_OPTION,
    expected_latest_checkpoint_id: str | None = _LEROBOT_EXPECT_LATEST_CHECKPOINT_ID_OPTION,
    expected_latest_claim_token: str | None = _LEROBOT_EXPECT_LATEST_CLAIM_TOKEN_OPTION,
    expected_checkpoint_index: int | None = _LEROBOT_EXPECT_CHECKPOINT_INDEX_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Ingest a LeRobot dataset into canonical episodes, observations, and videos."""
    from lancedb_robotics.adapters import AdapterError
    from lancedb_robotics.ingest import ingest_lerobot
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.storage import parse_storage_option_pairs

    try:
        storage_options = parse_storage_option_pairs(storage_option)
        source_storage_options = (
            parse_storage_option_pairs(source_storage_option)
            if source_storage_option is not None
            else storage_options
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        opened = Lake.open(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_options=storage_options,
            region=region,
            host_override=host_override,
        )
    except LakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        report = ingest_lerobot(
            opened,
            path,
            batch_size=batch_size,
            compact=compact,
            prune_versions=prune_versions,
            retain_versions=retain_versions,
            index_predicates=index_predicates,
            auth_ref=source_auth_ref or auth_ref,
            storage_options=source_storage_options,
            source_manifest_cache=source_manifest_cache,
            object_store_validation_policy=source_validation_policy,
            object_store_validation_sample_count=source_validation_sample_count,
            object_store_validation_sample_bytes=source_validation_sample_bytes,
            object_store_validation_strict_max_bytes=strict_content_hash_max_bytes,
            media_inspection_workers=media_inspection_workers,
            media_inspection_timeout_seconds=media_inspection_timeout_seconds,
            media_inspection_retries=media_inspection_retries,
            media_inspection_retry_backoff_seconds=media_inspection_retry_backoff_seconds,
            media_inspection_retry_policy=media_inspection_retry_policy,
            media_inspection_execution_mode=media_inspection_execution_mode,
            keyframe_map_inline_threshold_bytes=keyframe_map_inline_threshold_bytes,
            keyframe_map_inline_threshold_frames=keyframe_map_inline_threshold_frames,
            claim_lease_timeout=timedelta(seconds=claim_lease_seconds),
            claim_heartbeat_interval=timedelta(seconds=claim_heartbeat_seconds),
            expected_latest_checkpoint_id=expected_latest_checkpoint_id,
            expected_latest_claim_token=expected_latest_claim_token,
            expected_checkpoint_index=expected_checkpoint_index,
        )
    except AdapterError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _print_report(report, opened)


@ingest_app.command("lerobot-jobs")
def ingest_lerobot_jobs_command(
    lake: str = _LAKE_OPTION,
    status: str | None = typer.Option(None, "--status", help="Filter by latest job status."),
    source_id: str | None = typer.Option(None, "--source-id", help="Filter by source id."),
    limit: int = typer.Option(50, "--limit", help="Maximum recent jobs to list."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """List latest durable LeRobot ingest jobs."""
    from lancedb_robotics.ingest import list_lerobot_ingest_jobs

    opened = _open_lake_for_ingest_cli(
        lake,
        auth_ref=auth_ref,
        remote_auth_ref=remote_auth_ref,
        storage_auth_ref=storage_auth_ref,
        storage_option=storage_option,
        region=region,
        host_override=host_override,
    )
    jobs = list_lerobot_ingest_jobs(opened, status=status, source_id=source_id, limit=limit)
    parsed_format = _parse_lerobot_job_format(output_format)
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(jobs), indent=2, sort_keys=True))
        return
    if not jobs:
        typer.echo("no LeRobot ingest jobs")
        return
    for job in jobs:
        _print_lerobot_job_summary(job)


@ingest_app.command("lerobot-job")
def ingest_lerobot_job_command(
    job_id: str = typer.Argument(..., help="LeRobot ingest job id."),
    lake: str = _LAKE_OPTION,
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Show one durable LeRobot ingest job and its checkpoint history."""
    from lancedb_robotics.ingest import get_lerobot_ingest_job

    opened = _open_lake_for_ingest_cli(
        lake,
        auth_ref=auth_ref,
        remote_auth_ref=remote_auth_ref,
        storage_auth_ref=storage_auth_ref,
        storage_option=storage_option,
        region=region,
        host_override=host_override,
    )
    parsed_format = _parse_lerobot_job_format(output_format)
    try:
        job = get_lerobot_ingest_job(opened, job_id)
    except KeyError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(job), indent=2, sort_keys=True))
        return
    _print_lerobot_job_summary(job)
    history = job.get("history") or []
    typer.echo("checkpoints:")
    for checkpoint in history:
        typer.echo(
            "  "
            f"{checkpoint.get('checkpoint_index')} "
            f"{checkpoint.get('status')} "
            f"{checkpoint.get('phase')} "
            f"rows={checkpoint.get('rows_seen')} "
            f"obs={checkpoint.get('observations_written')} "
            f"updated={checkpoint.get('updated_at')}"
        )


@ingest_app.command("lerobot-media-inspection-timeout-plan")
def ingest_lerobot_media_inspection_timeout_plan_command(
    lake: str | None = typer.Option(
        None,
        "--lake",
        help="Path or object-store URI to the lake; optional with --checkpoint-rows-json.",
    ),
    checkpoint_rows_json: Path | None = _LEROBOT_CLAIM_CHAOS_CHECKPOINT_ROWS_JSON_OPTION,
    job_id: str | None = typer.Option(None, "--job-id", help="Filter by LeRobot job id."),
    source_id: str | None = typer.Option(None, "--source-id", help="Filter by source id."),
    source_uri: str | None = typer.Option(None, "--source-uri", help="Filter by source URI."),
    storage_tier: str | None = typer.Option(
        None,
        "--storage-tier",
        help="Group/filter storage tier label, for example local, object-store, or huggingface.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Group/filter storage provider label, for example local, s3, gcs, or huggingface.",
    ),
    min_timeout_seconds: float = typer.Option(
        1.0,
        "--min-timeout-seconds",
        help="Minimum recommended media-inspection timeout.",
    ),
    max_timeout_seconds: float = typer.Option(
        600.0,
        "--max-timeout-seconds",
        help="Maximum recommended media-inspection timeout.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Recommend LeRobot media-inspection timeout and retry settings from telemetry."""
    from lancedb_robotics.ingest import recommend_lerobot_media_inspection_timeouts

    parsed_format = _parse_lerobot_job_format(output_format)
    try:
        checkpoint_rows = _load_lerobot_checkpoint_rows_json(checkpoint_rows_json)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if lake is None and checkpoint_rows is None:
        typer.echo("error: --lake or --checkpoint-rows-json is required", err=True)
        raise typer.Exit(code=2)
    opened = None
    if lake is not None:
        opened = _open_lake_for_ingest_cli(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_option=storage_option,
            region=region,
            host_override=host_override,
        )
    try:
        report = recommend_lerobot_media_inspection_timeouts(
            opened,
            checkpoint_rows=checkpoint_rows,
            job_id=job_id,
            source_id=source_id,
            source_uri=source_uri,
            storage_tier=storage_tier,
            provider=provider,
            min_timeout_seconds=min_timeout_seconds,
            max_timeout_seconds=max_timeout_seconds,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(report), indent=2, sort_keys=True))
        return
    _print_lerobot_media_inspection_timeout_plan(report)


@ingest_app.command("lerobot-claim-watchdog")
def ingest_lerobot_claim_watchdog_command(
    lake: str = _LAKE_OPTION,
    source_id: str | None = typer.Option(
        None, "--source-id", help="Only report claims for one source id."
    ),
    stale_after_seconds: float = typer.Option(
        21600.0,
        "--stale-after-seconds",
        help="Fallback stale timeout when a running checkpoint has no claim_expires_at.",
    ),
    recovery_action: str = typer.Option(
        "abandon",
        "--recovery-action",
        help="Suggested recovery action in emitted commands: abandon or steal.",
    ),
    new_owner: str | None = typer.Option(
        None,
        "--new-owner",
        help="Owner to include in suggested recovery commands.",
    ),
    created_by: str = typer.Option(
        "lancedb-robotics",
        "--created-by",
        help="Actor to include in suggested recovery commands when non-default.",
    ),
    fail_on_stale: bool = typer.Option(
        False,
        "--fail-on-stale",
        help="Exit 3 when stale claims are found after writing the report.",
    ),
    json_out: Path | None = _LEROBOT_CLAIM_WATCHDOG_JSON_OUT_OPTION,
    markdown_out: Path | None = _LEROBOT_CLAIM_WATCHDOG_MARKDOWN_OUT_OPTION,
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Dry-run stale LeRobot ingest claim watchdog and recovery plan."""
    from lancedb_robotics.ingest import watch_lerobot_ingest_claims

    opened = _open_lake_for_ingest_cli(
        lake,
        auth_ref=auth_ref,
        remote_auth_ref=remote_auth_ref,
        storage_auth_ref=storage_auth_ref,
        storage_option=storage_option,
        region=region,
        host_override=host_override,
    )
    try:
        report = watch_lerobot_ingest_claims(
            opened,
            source_id=source_id,
            stale_after=timedelta(seconds=stale_after_seconds),
            recovery_action=recovery_action,
            new_owner=new_owner,
            created_by=created_by,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    parsed_format = _parse_lerobot_job_format(output_format)
    payload = report.to_params()
    json_payload = json.dumps(_json_ready(payload), indent=2, sort_keys=True)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json_payload + "\n", encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(
            _lerobot_claim_watchdog_markdown(_json_ready(payload)), encoding="utf-8"
        )

    if parsed_format == "json":
        typer.echo(json_payload)
    else:
        _print_lerobot_claim_watchdog_report(_json_ready(payload))
    if fail_on_stale and report.has_stale:
        raise typer.Exit(code=3)


@ingest_app.command("lerobot-claim-recovery-simulate")
def ingest_lerobot_claim_recovery_simulate_command(
    lake: str | None = typer.Option(
        None,
        "--lake",
        help="Path or object-store URI to an existing lake; omit when using checkpoint rows or synthetic inputs.",
    ),
    checkpoint_rows_json: Path | None = _LEROBOT_CLAIM_CHAOS_CHECKPOINT_ROWS_JSON_OPTION,
    scenario: str = typer.Option(
        "auto",
        "--scenario",
        help="Recommendation target: auto, local-smoke, ci, mid-corpus, full-public-corpus, audit-window.",
    ),
    source_id: str | None = typer.Option(None, "--source-id", help="Simulate one source id."),
    recovery_action: str = typer.Option(
        "abandon",
        "--recovery-action",
        help="Recovery action in generated samples: abandon or steal.",
    ),
    new_owner: str | None = typer.Option(
        None,
        "--new-owner",
        help="Owner to include in generated sample recovery commands.",
    ),
    stale_after_seconds: float = typer.Option(
        21600.0,
        "--stale-after-seconds",
        help="Claim lease/stale timeout to simulate.",
    ),
    heartbeat_interval_seconds: float = typer.Option(
        300.0,
        "--heartbeat-interval-seconds",
        help="Claim heartbeat interval to simulate.",
    ),
    source_size_frames: int = typer.Option(
        4096,
        "--source-size-frames",
        help="Frames in the modeled source used for crash-point duplicate checks.",
    ),
    episode_count: int | None = typer.Option(
        None,
        "--episode-count",
        help="Episode count in the modeled source; defaults to the frame count.",
    ),
    camera_count: int = typer.Option(
        1,
        "--camera-count",
        help="Camera streams per episode in the modeled source.",
    ),
    batch_size: int = typer.Option(
        1024,
        "--batch-size",
        help="Frame batch size used for checkpoint-row growth simulation.",
    ),
    retry_owner_count: int = typer.Option(
        2,
        "--retry-owner-count",
        help="Operators/workers racing to recover the same stale latest checkpoint.",
    ),
    remote_latency_ms: float = typer.Option(
        0.0,
        "--remote-latency-ms",
        help="Per-operation remote/object-store latency budget to include in latency estimates.",
    ),
    synthetic_sources: int | None = typer.Option(
        None,
        "--synthetic-sources",
        help="Synthetic source count; use instead of --lake or --checkpoint-rows-json.",
    ),
    synthetic_completed_jobs_per_source: int = typer.Option(
        0,
        "--synthetic-completed-jobs-per-source",
        help="Synthetic completed terminal jobs per source.",
    ),
    synthetic_failed_jobs_per_source: int = typer.Option(
        0,
        "--synthetic-failed-jobs-per-source",
        help="Synthetic failed terminal jobs per source.",
    ),
    synthetic_running_jobs_per_source: int = typer.Option(
        1,
        "--synthetic-running-jobs-per-source",
        help="Synthetic running jobs per source.",
    ),
    synthetic_checkpoints_per_job: int = typer.Option(
        4,
        "--synthetic-checkpoints-per-job",
        help="Synthetic checkpoint rows per job before recovery/retention.",
    ),
    synthetic_terminal_age_days: float = typer.Option(
        90.0,
        "--synthetic-terminal-age-days",
        help="Synthetic age of terminal jobs for retention projections.",
    ),
    synthetic_stale_running_fraction: float = typer.Option(
        0.25,
        "--synthetic-stale-running-fraction",
        help="Fraction of selected synthetic running jobs that are stale.",
    ),
    synthetic_missing_lease_fraction: float = typer.Option(
        0.0,
        "--synthetic-missing-lease-fraction",
        help="Fraction of selected synthetic running jobs whose lease metadata is missing.",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Deterministic seed used to choose bounded synthetic sample jobs.",
    ),
    output_format: str = typer.Option("json", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Simulate LeRobot stale-claim recovery crash points and scale risk."""
    from lancedb_robotics.ingest import simulate_lerobot_claim_recovery_chaos

    opened = None
    if lake is not None:
        opened = _open_lake_for_ingest_cli(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_option=storage_option,
            region=region,
            host_override=host_override,
        )
    try:
        rows = _load_lerobot_checkpoint_rows_json(checkpoint_rows_json)
        report = simulate_lerobot_claim_recovery_chaos(
            opened,
            checkpoint_rows=rows,
            scenario=scenario,
            source_id=source_id,
            recovery_action=recovery_action,
            new_owner=new_owner,
            stale_after=timedelta(seconds=stale_after_seconds),
            claim_heartbeat_interval=timedelta(seconds=heartbeat_interval_seconds),
            source_size_frames=source_size_frames,
            episode_count=episode_count,
            camera_count=camera_count,
            batch_size=batch_size,
            retry_owner_count=retry_owner_count,
            remote_latency_ms=remote_latency_ms,
            synthetic_sources=synthetic_sources,
            synthetic_completed_jobs_per_source=synthetic_completed_jobs_per_source,
            synthetic_failed_jobs_per_source=synthetic_failed_jobs_per_source,
            synthetic_running_jobs_per_source=synthetic_running_jobs_per_source,
            synthetic_checkpoints_per_job=synthetic_checkpoints_per_job,
            synthetic_terminal_age_days=synthetic_terminal_age_days,
            synthetic_stale_running_fraction=synthetic_stale_running_fraction,
            synthetic_missing_lease_fraction=synthetic_missing_lease_fraction,
            seed=seed,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    parsed_format = _parse_lerobot_job_format(output_format)
    payload = report.to_params()
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return
    _print_lerobot_claim_recovery_chaos_report(payload)


@ingest_app.command("lerobot-claim-recover")
def ingest_lerobot_claim_recover_command(
    job_id: str = typer.Argument(..., help="LeRobot ingest job id with a stale running claim."),
    lake: str = _LAKE_OPTION,
    action: str = typer.Option(
        "abandon",
        "--action",
        help="Recovery action: abandon or steal.",
    ),
    new_owner: str | None = typer.Option(
        None,
        "--new-owner",
        help="Owner recorded on the recovery checkpoint; defaults to --created-by.",
    ),
    stale_after_seconds: float = typer.Option(
        21600.0,
        "--stale-after-seconds",
        help="Fallback stale timeout when the running checkpoint has no claim_expires_at.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Record recovery even when the current claim lease has not expired.",
    ),
    created_by: str = typer.Option(
        "lancedb-robotics",
        "--created-by",
        help="Actor recorded as creating the recovery checkpoint.",
    ),
    expected_latest_checkpoint_id: str | None = _LEROBOT_EXPECT_LATEST_CHECKPOINT_ID_OPTION,
    expected_latest_claim_token: str | None = _LEROBOT_EXPECT_LATEST_CLAIM_TOKEN_OPTION,
    expected_checkpoint_index: int | None = _LEROBOT_EXPECT_CHECKPOINT_INDEX_OPTION,
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Abandon or steal a stale durable LeRobot ingest claim."""
    from lancedb_robotics.adapters import AdapterError
    from lancedb_robotics.ingest import LeRobotClaimPreconditionError, recover_lerobot_ingest_claim

    opened = _open_lake_for_ingest_cli(
        lake,
        auth_ref=auth_ref,
        remote_auth_ref=remote_auth_ref,
        storage_auth_ref=storage_auth_ref,
        storage_option=storage_option,
        region=region,
        host_override=host_override,
    )
    parsed_format = _parse_lerobot_job_format(output_format)
    try:
        report = recover_lerobot_ingest_claim(
            opened,
            job_id,
            action=action,
            new_owner=new_owner,
            stale_after=timedelta(seconds=stale_after_seconds),
            force=force,
            created_by=created_by,
            expected_latest_checkpoint_id=expected_latest_checkpoint_id,
            expected_latest_claim_token=expected_latest_claim_token,
            expected_checkpoint_index=expected_checkpoint_index,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except LeRobotClaimPreconditionError as exc:
        if parsed_format == "json":
            typer.echo(json.dumps(_json_ready(exc.to_params()), indent=2, sort_keys=True))
        else:
            typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (AdapterError, KeyError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = report.to_params()
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return
    _print_lerobot_claim_recovery_report(payload)


@ingest_app.command("lerobot-checkpoint-hold")
def ingest_lerobot_checkpoint_hold_command(
    lake: str = _LAKE_OPTION,
    checkpoint_id: str | None = typer.Option(
        None, "--checkpoint-id", help="Hold one checkpoint row id."
    ),
    job_id: str | None = typer.Option(
        None, "--job-id", help="Hold checkpoints for one ingest job."
    ),
    source_id: str | None = typer.Option(
        None, "--source-id", help="Hold checkpoints for one source id."
    ),
    hf_repo_id: str | None = typer.Option(
        None, "--hf-repo-id", help="Hold checkpoints for one HF repo id."
    ),
    requested_revision: str | None = typer.Option(
        None,
        "--requested-revision",
        help="Hold checkpoints whose requested HF revision matches this value.",
    ),
    resolved_revision: str | None = typer.Option(
        None,
        "--resolved-revision",
        help="Hold checkpoints whose resolved HF revision matches this value.",
    ),
    status: list[str] | None = _LEROBOT_CHECKPOINT_HOLD_STATUS_OPTION,
    updated_after: str | None = typer.Option(
        None,
        "--updated-after",
        help="Hold checkpoints updated at or after this ISO-8601 timestamp.",
    ),
    updated_before: str | None = typer.Option(
        None,
        "--updated-before",
        help="Hold checkpoints updated at or before this ISO-8601 timestamp.",
    ),
    retain_until: str | None = typer.Option(
        None,
        "--retain-until",
        help="Release-safe hold expiry timestamp; legal/audit/promotion holds are indefinite.",
    ),
    legal_hold: bool = typer.Option(
        False, "--legal-hold/--no-legal-hold", help="Mark as a legal hold."
    ),
    audit_hold: bool = typer.Option(
        True, "--audit-hold/--no-audit-hold", help="Mark as an audit hold."
    ),
    promotion_hold: bool = typer.Option(
        False,
        "--promotion-hold/--no-promotion-hold",
        help="Mark as a promotion/release hold.",
    ),
    owner: str | None = typer.Option(None, "--owner", help="Governance owner for the hold."),
    reason: str | None = typer.Option(None, "--reason", help="Human-readable hold reason."),
    created_by: str = typer.Option(
        "lancedb-robotics", "--created-by", help="Actor recorded on the hold row."
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Create a first-class hold over matching LeRobot checkpoint rows."""
    from lancedb_robotics.ingest import hold_lerobot_checkpoints

    opened = _open_lake_for_ingest_cli(
        lake,
        auth_ref=auth_ref,
        remote_auth_ref=remote_auth_ref,
        storage_auth_ref=storage_auth_ref,
        storage_option=storage_option,
        region=region,
        host_override=host_override,
    )
    try:
        report = hold_lerobot_checkpoints(
            opened,
            checkpoint_id=checkpoint_id,
            job_id=job_id,
            source_id=source_id,
            hf_repo_id=hf_repo_id,
            requested_revision=requested_revision,
            resolved_revision=resolved_revision,
            status=tuple(status) if status else None,
            updated_after=updated_after,
            updated_before=updated_before,
            retain_until=retain_until,
            legal_hold=legal_hold,
            audit_hold=audit_hold,
            promotion_hold=promotion_hold,
            owner=owner,
            reason=reason,
            created_by=created_by,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    parsed_format = _parse_lerobot_job_format(output_format)
    payload = report.to_params()
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return
    _print_lerobot_checkpoint_hold_report(payload)


@ingest_app.command("lerobot-checkpoint-release-hold")
def ingest_lerobot_checkpoint_release_hold_command(
    hold_id: str = typer.Argument(..., help="LeRobot checkpoint hold id to release."),
    lake: str = _LAKE_OPTION,
    released_by: str = typer.Option(
        "lancedb-robotics",
        "--released-by",
        help="Actor recorded on the released hold row.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Release a first-class LeRobot checkpoint hold."""
    from lancedb_robotics.ingest import release_lerobot_checkpoint_hold

    opened = _open_lake_for_ingest_cli(
        lake,
        auth_ref=auth_ref,
        remote_auth_ref=remote_auth_ref,
        storage_auth_ref=storage_auth_ref,
        storage_option=storage_option,
        region=region,
        host_override=host_override,
    )
    try:
        report = release_lerobot_checkpoint_hold(
            opened,
            hold_id,
            released_by=released_by,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except KeyError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    parsed_format = _parse_lerobot_job_format(output_format)
    payload = report.to_params()
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return
    _print_lerobot_checkpoint_hold_report(payload)


@ingest_app.command("lerobot-checkpoint-retention")
def ingest_lerobot_checkpoint_retention_command(
    lake: str = _LAKE_OPTION,
    apply_changes: bool = typer.Option(
        False,
        "--apply/--dry-run",
        help="Apply retention changes; default is a dry run that only reports planned deletions.",
    ),
    older_than_days: float = typer.Option(
        30.0,
        "--older-than-days",
        help="Only summarize terminal jobs older than this many days; use -1 to ignore age.",
    ),
    source_id: str | None = typer.Option(
        None, "--source-id", help="Limit retention to one source id."
    ),
    status: list[str] | None = _LEROBOT_RETENTION_STATUS_OPTION,
    retain_completed_per_source: int = typer.Option(
        10,
        "--retain-completed-per-source",
        help="Keep this many completed job histories fully expanded per source.",
    ),
    retain_failed_per_source: int = typer.Option(
        10,
        "--retain-failed-per-source",
        help="Keep this many failed job histories fully expanded per source.",
    ),
    compact: bool = typer.Option(
        True,
        "--compact/--no-compact",
        help="Compact the checkpoint table after applying row retention.",
    ),
    cleanup_older_than_days: float = typer.Option(
        7.0,
        "--cleanup-older-than-days",
        help="Clean old checkpoint-table versions older than this many days after apply; use -1 to skip.",
    ),
    retain_versions: int | None = typer.Option(
        2,
        "--retain-versions",
        help="Recent checkpoint-table versions to keep during cleanup.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Dry-run or apply retention for durable LeRobot ingest checkpoints."""
    from lancedb_robotics.ingest import apply_lerobot_checkpoint_retention

    opened = _open_lake_for_ingest_cli(
        lake,
        auth_ref=auth_ref,
        remote_auth_ref=remote_auth_ref,
        storage_auth_ref=storage_auth_ref,
        storage_option=storage_option,
        region=region,
        host_override=host_override,
    )
    try:
        report = apply_lerobot_checkpoint_retention(
            opened,
            older_than=None if older_than_days < 0 else timedelta(days=older_than_days),
            statuses=tuple(status) if status else None,
            source_id=source_id,
            retain_completed_per_source=retain_completed_per_source,
            retain_failed_per_source=retain_failed_per_source,
            dry_run=not apply_changes,
            compact=compact,
            cleanup_older_than=(
                None if cleanup_older_than_days < 0 else timedelta(days=cleanup_older_than_days)
            ),
            retain_versions=retain_versions,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    parsed_format = _parse_lerobot_job_format(output_format)
    payload = report.to_params()
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return
    _print_lerobot_checkpoint_retention_report(payload)


@ingest_app.command("lerobot-checkpoint-retention-plan")
def ingest_lerobot_checkpoint_retention_plan_command(
    lake: str | None = typer.Option(
        None,
        "--lake",
        help="Path or object-store URI to an existing lake; omit when using synthetic inputs.",
    ),
    scenario: str = typer.Option(
        "auto",
        "--scenario",
        help="Recommendation target: auto, local-smoke, ci, mid-corpus, full-public-corpus, audit-window.",
    ),
    source_id: str | None = typer.Option(
        None, "--source-id", help="Plan retention for one source id."
    ),
    status: list[str] | None = _LEROBOT_RETENTION_STATUS_OPTION,
    synthetic_sources: int | None = typer.Option(
        None,
        "--synthetic-sources",
        help="Synthetic source count; use instead of --lake for pre-backfill planning.",
    ),
    synthetic_completed_jobs_per_source: int = typer.Option(
        0,
        "--synthetic-completed-jobs-per-source",
        help="Synthetic completed terminal jobs per source.",
    ),
    synthetic_failed_jobs_per_source: int = typer.Option(
        0,
        "--synthetic-failed-jobs-per-source",
        help="Synthetic failed terminal jobs per source.",
    ),
    synthetic_running_jobs_per_source: int = typer.Option(
        0,
        "--synthetic-running-jobs-per-source",
        help="Synthetic active/running jobs per source.",
    ),
    synthetic_checkpoints_per_job: int = typer.Option(
        4,
        "--synthetic-checkpoints-per-job",
        help="Synthetic checkpoint rows per job before retention.",
    ),
    synthetic_terminal_age_days: float = typer.Option(
        90.0,
        "--synthetic-terminal-age-days",
        help="Synthetic age of terminal jobs for age-threshold comparison.",
    ),
    output_format: str = typer.Option("json", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Plan LeRobot checkpoint retention scale and policy recommendations."""
    from lancedb_robotics.ingest import plan_lerobot_checkpoint_retention_scale

    opened = None
    if lake is not None:
        opened = _open_lake_for_ingest_cli(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_option=storage_option,
            region=region,
            host_override=host_override,
        )
    try:
        report = plan_lerobot_checkpoint_retention_scale(
            opened,
            scenario=scenario,
            source_id=source_id,
            statuses=tuple(status) if status else None,
            synthetic_sources=synthetic_sources,
            synthetic_completed_jobs_per_source=synthetic_completed_jobs_per_source,
            synthetic_failed_jobs_per_source=synthetic_failed_jobs_per_source,
            synthetic_running_jobs_per_source=synthetic_running_jobs_per_source,
            synthetic_checkpoints_per_job=synthetic_checkpoints_per_job,
            synthetic_terminal_age_days=synthetic_terminal_age_days,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    parsed_format = _parse_lerobot_job_format(output_format)
    payload = report.to_params()
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return
    _print_lerobot_checkpoint_retention_plan_report(payload)


@ingest_app.command("lerobot-checkpoint-retention-schedule")
def ingest_lerobot_checkpoint_retention_schedule_command(
    lake: str = _LAKE_OPTION,
    config_json: str | None = typer.Option(
        None,
        "--config-json",
        help="JSON schedule config; CLI flags override omitted values.",
    ),
    schedule_id: str | None = typer.Option(
        None,
        "--schedule-id",
        help="Stable scheduler/report id.",
    ),
    every_minutes: float | None = typer.Option(
        None,
        "--every-minutes",
        help="Nominal cadence for next_run_after; use -1 for ad hoc/no next run.",
    ),
    apply_changes: bool = typer.Option(
        False,
        "--apply/--dry-run",
        help="Apply retention changes; default/config dry-run emits only telemetry.",
    ),
    older_than_days: float | None = typer.Option(
        None,
        "--older-than-days",
        help="Only summarize terminal jobs older than this many days; use -1 to ignore age.",
    ),
    source_id: str | None = typer.Option(
        None, "--source-id", help="Limit retention to one source id."
    ),
    status: list[str] | None = _LEROBOT_RETENTION_STATUS_OPTION,
    retain_completed_per_source: int | None = typer.Option(
        None,
        "--retain-completed-per-source",
        help="Keep this many completed job histories fully expanded per source.",
    ),
    retain_failed_per_source: int | None = typer.Option(
        None,
        "--retain-failed-per-source",
        help="Keep this many failed job histories fully expanded per source.",
    ),
    compact: bool = typer.Option(
        True,
        "--compact/--no-compact",
        help="Compact the checkpoint table after applying row retention.",
    ),
    cleanup_older_than_days: float | None = typer.Option(
        None,
        "--cleanup-older-than-days",
        help="Clean old checkpoint-table versions after apply; use -1 to skip.",
    ),
    retain_versions: int | None = typer.Option(
        None,
        "--retain-versions",
        help="Recent checkpoint-table versions to keep during cleanup.",
    ),
    max_rows: int | None = typer.Option(
        None,
        "--max-rows",
        help="Warn when post-run checkpoint rows exceed this count.",
    ),
    max_rows_per_source: int | None = typer.Option(
        None,
        "--max-rows-per-source",
        help="Warn when any source exceeds this post-run checkpoint row count.",
    ),
    max_version_delta: int | None = typer.Option(
        None,
        "--max-version-delta",
        help="Warn when checkpoint table version growth exceeds this count.",
    ),
    output_format: str = typer.Option("json", "--format", help="Output format: text or json."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
) -> None:
    """Run one scheduler-friendly LeRobot checkpoint retention pass."""
    from lancedb_robotics.ingest import run_lerobot_checkpoint_retention_schedule

    try:
        config = _parse_lerobot_schedule_config(config_json)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    opened = _open_lake_for_ingest_cli(
        lake,
        auth_ref=auth_ref,
        remote_auth_ref=remote_auth_ref,
        storage_auth_ref=storage_auth_ref,
        storage_option=storage_option,
        region=region,
        host_override=host_override,
    )
    try:
        report = run_lerobot_checkpoint_retention_schedule(
            opened,
            schedule_id=str(
                schedule_id or config.get("schedule_id") or "lerobot-checkpoint-retention"
            ),
            interval=_schedule_minutes(
                every_minutes,
                config,
                "every_minutes",
                default=1440.0,
            ),
            older_than=_schedule_days(
                older_than_days,
                config,
                "older_than_days",
                default=30.0,
            ),
            statuses=tuple(status or config.get("statuses") or ()) or None,
            source_id=source_id or config.get("source_id"),
            retain_completed_per_source=_schedule_int(
                retain_completed_per_source,
                config,
                "retain_completed_per_source",
                default=10,
            ),
            retain_failed_per_source=_schedule_int(
                retain_failed_per_source,
                config,
                "retain_failed_per_source",
                default=10,
            ),
            dry_run=_schedule_dry_run(apply_changes, config),
            compact=bool(config.get("compact", compact)),
            cleanup_older_than=_schedule_days(
                cleanup_older_than_days,
                config,
                "cleanup_older_than_days",
                default=7.0,
            ),
            retain_versions=_schedule_int(
                retain_versions,
                config,
                "retain_versions",
                default=2,
                allow_none=True,
            ),
            max_rows=_schedule_int(max_rows, config, "max_rows", default=None, allow_none=True),
            max_rows_per_source=_schedule_int(
                max_rows_per_source,
                config,
                "max_rows_per_source",
                default=None,
                allow_none=True,
            ),
            max_version_delta=_schedule_int(
                max_version_delta,
                config,
                "max_version_delta",
                default=None,
                allow_none=True,
            ),
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    parsed_format = _parse_lerobot_job_format(output_format)
    payload = report.to_params()
    if parsed_format == "json":
        typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return
    _print_lerobot_checkpoint_retention_schedule_report(payload)


def _print_report(report, opened=None) -> None:
    typer.echo(f"lake: {report.lake_uri}")
    typer.echo(
        f"source: {report.source.source_id} "
        f"({'registered' if report.source.created else 'already registered'})"
    )
    run_state = "already ingested" if report.already_ingested else "ingested"
    typer.echo(f"run: {report.run_id} ({run_state})")
    typer.echo(
        f"  {report.message_count} messages, {report.duration_ns / 1e9:.1f}s "
        f"({report.start_time_ns} .. {report.end_time_ns})"
    )
    if report.quarantined:
        typer.echo(
            f"  QUARANTINED: integrity '{report.integrity_status}' "
            f"(recovered {report.recovered_count} message(s); {report.integrity_reason})"
        )
    if report.ingest_job_id:
        typer.echo(f"  ingest job: {report.ingest_job_id}")
    if report.decode_by_status:
        decode = ", ".join(f"{k} {v}" for k, v in report.decode_by_status.items())
        encodings = ", ".join(f"{k} {v}" for k, v in report.decode_by_encoding.items())
        typer.echo(f"  decode: {decode}")
        typer.echo(f"  encodings: {encodings}")
    if report.compaction:
        c = report.compaction
        line = (
            f"observations layout: {c['fragments_before']} -> {c['fragments_after']} fragments, "
            f"v{c['version_before']} -> v{c['version_after']}"
        )
        cleanup = c.get("cleanup") or {}
        if cleanup.get("old_versions"):
            line += f" (pruned {cleanup['old_versions']} version(s))"
        typer.echo(line)
    typer.echo("rows added:")
    for table, count in report.rows_added.items():
        typer.echo(f"  {table} +{count}")
    typer.echo("observations by topic:")
    for topic, count in report.observations_by_topic.items():
        typer.echo(f"  {topic}\t{count}")
    if opened is not None:
        echo_emitted_lineage(opened, report.transform_id)


def _open_lake_for_ingest_cli(
    lake: str,
    *,
    auth_ref: str | None,
    remote_auth_ref: str | None,
    storage_auth_ref: str | None,
    storage_option: list[str] | None,
    region: str | None,
    host_override: str | None,
):
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.storage import parse_storage_option_pairs

    try:
        storage_options = parse_storage_option_pairs(storage_option)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        return Lake.open(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_options=storage_options,
            region=region,
            host_override=host_override,
        )
    except LakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _parse_lerobot_job_format(output_format: str) -> str:
    parsed = output_format.strip().lower()
    if parsed not in {"text", "json"}:
        typer.echo("error: --format must be one of: text, json", err=True)
        raise typer.Exit(code=2)
    return parsed


def _load_lerobot_checkpoint_rows_json(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read --checkpoint-rows-json: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --checkpoint-rows-json: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("checkpoint_rows", payload.get("rows"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("--checkpoint-rows-json must contain a list of checkpoint row objects")
    return [dict(item) for item in payload]


def _parse_lerobot_schedule_config(config_json: str | None) -> dict:
    if not config_json:
        return {}
    try:
        payload = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --config-json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--config-json must be a JSON object")
    statuses = payload.get("statuses")
    if statuses is not None:
        if not isinstance(statuses, list) or not all(isinstance(item, str) for item in statuses):
            raise ValueError("config statuses must be a list of strings")
    return payload


def _schedule_days(
    cli_value: float | None,
    config: dict,
    key: str,
    *,
    default: float | None,
) -> timedelta | None:
    value = cli_value if cli_value is not None else config.get(key, default)
    if value is None:
        return None
    numeric = float(value)
    if numeric < 0:
        return None
    return timedelta(days=numeric)


def _schedule_minutes(
    cli_value: float | None,
    config: dict,
    key: str,
    *,
    default: float | None,
) -> timedelta | None:
    value = cli_value if cli_value is not None else config.get(key, default)
    if value is None:
        return None
    numeric = float(value)
    if numeric < 0:
        return None
    return timedelta(minutes=numeric)


def _schedule_int(
    cli_value: int | None,
    config: dict,
    key: str,
    *,
    default: int | None,
    allow_none: bool = False,
) -> int | None:
    value = cli_value if cli_value is not None else config.get(key, default)
    if value is None and allow_none:
        return None
    if value is None:
        raise ValueError(f"{key} must not be null")
    return int(value)


def _schedule_dry_run(apply_changes: bool, config: dict) -> bool:
    if apply_changes:
        return False
    if "dry_run" in config:
        return bool(config["dry_run"])
    return True


def _print_lerobot_job_summary(job: dict) -> None:
    typer.echo(
        f"{job.get('job_id')} {job.get('status')} {job.get('phase')} "
        f"rows={job.get('rows_seen')} obs={job.get('observations_written')} "
        f"updated={job.get('updated_at')}"
    )
    if job.get("hf_repo_id"):
        typer.echo(
            f"  hf: {job.get('hf_repo_id')} "
            f"requested={job.get('requested_revision')} resolved={job.get('resolved_revision')}"
        )
    if job.get("error"):
        typer.echo(f"  error: {job.get('error')}")


def _print_lerobot_media_inspection_timeout_plan(report: dict) -> None:
    telemetry = report.get("telemetry") or {}
    recommendation = report.get("recommendation") or {}
    durations = telemetry.get("duration_ms") or {}
    source_counts = report.get("source_counts") or {}
    typer.echo(f"lake: {report.get('lake_uri') or 'offline'}")
    typer.echo(
        "lerobot media-inspection timeout plan: "
        f"reports={source_counts.get('reports')} "
        f"checkpoints={source_counts.get('checkpoints')} "
        f"transforms={source_counts.get('completed_transforms')} "
        f"videos={telemetry.get('reported_video_count')} "
        f"timeouts={telemetry.get('total_timeouts')} "
        f"retries={telemetry.get('total_retries')}"
    )
    typer.echo(
        "duration_ms: "
        f"count={durations.get('count')} "
        f"p95={durations.get('p95')} "
        f"p99={durations.get('p99')} "
        f"max={durations.get('max')}"
    )
    typer.echo(
        "recommended: "
        f"status={recommendation.get('status')} "
        f"timeout_s={recommendation.get('timeout_seconds')} "
        f"retries={recommendation.get('retry_count')} "
        f"backoff_s={recommendation.get('retry_backoff_seconds')}"
    )
    apply_args = recommendation.get("apply_args") or []
    if apply_args:
        typer.echo("apply args: " + " ".join(str(value) for value in apply_args))
    flags = recommendation.get("flags") or []
    if flags:
        typer.echo("flags: " + ", ".join(str(flag) for flag in flags))
    for group in report.get("groups") or []:
        selector = group.get("selector") or {}
        group_recommendation = group.get("recommendation") or {}
        group_telemetry = group.get("telemetry") or {}
        typer.echo(
            "  "
            f"{selector.get('storage_tier')}/{selector.get('provider')}/"
            f"{selector.get('corpus_size_tier')}: "
            f"reports={group.get('sample_count')} "
            f"videos={group_telemetry.get('reported_video_count')} "
            f"timeout_s={group_recommendation.get('timeout_seconds')} "
            f"retries={group_recommendation.get('retry_count')}"
        )


def _print_lerobot_checkpoint_retention_report(report: dict) -> None:
    mode = "dry-run" if report.get("dry_run") else "applied"
    typer.echo(f"lake: {report.get('lake_uri')}")
    typer.echo(
        f"lerobot checkpoint retention: {mode}; "
        f"rows {report.get('rows_before')}->{report.get('rows_after')} "
        f"(deleted {report.get('rows_deleted')})"
    )
    typer.echo(
        f"versions: v{report.get('version_before')}->v{report.get('version_after')}; "
        f"fragments {report.get('fragments_before')}->{report.get('fragments_after')}"
    )
    typer.echo(
        f"jobs: seen {report.get('jobs_seen')}, "
        f"compacted {report.get('jobs_compacted')}, protected {report.get('jobs_protected')}"
    )
    compacted = [job for job in report.get("jobs") or [] if job.get("rows_deleted")]
    if compacted:
        typer.echo("compacted jobs:")
        for job in compacted:
            typer.echo(
                "  "
                f"{job.get('job_id')} {job.get('status')} "
                f"rows {job.get('rows_before')}->{job.get('rows_after')} "
                f"terminal={job.get('terminal_checkpoint_id')}"
            )


def _print_lerobot_checkpoint_retention_plan_report(report: dict) -> None:
    typer.echo(
        f"lerobot checkpoint retention plan: {report.get('mode')} scenario={report.get('scenario')}"
    )
    typer.echo(f"recommended policy: {report.get('recommended_policy')}")
    for policy in report.get("policies") or ():
        typer.echo(
            "  "
            f"{policy.get('name')}: rows {policy.get('rows_before')}->{policy.get('rows_after')} "
            f"(deleted {policy.get('rows_deleted')}), "
            f"jobs compacted={policy.get('jobs_compacted')} protected={policy.get('jobs_protected')}, "
            f"version +{policy.get('estimated_version_delta')}"
        )


def _print_lerobot_checkpoint_retention_schedule_report(report: dict) -> None:
    telemetry = report.get("telemetry") or {}
    typer.echo(f"lake: {report.get('lake_uri')}")
    typer.echo(
        f"lerobot checkpoint retention schedule: {report.get('schedule_id')} "
        f"dry_run={report.get('dry_run')} next={report.get('next_run_after')}"
    )
    typer.echo(
        f"rows {telemetry.get('rows_before')}->{telemetry.get('rows_after')} "
        f"(deleted {telemetry.get('rows_deleted')}); "
        f"jobs compacted={telemetry.get('jobs_compacted')} "
        f"protected={telemetry.get('jobs_protected')}"
    )
    for alert in report.get("alerts") or ():
        source = f" source={alert.get('source_id')}" if alert.get("source_id") else ""
        typer.echo(
            f"alert[{alert.get('level')}]: {alert.get('metric')}{source} "
            f"actual={alert.get('actual')} threshold={alert.get('threshold')}"
        )


def _print_lerobot_checkpoint_hold_report(report: dict) -> None:
    selector = report.get("selector") or {}
    checkpoint_ids = report.get("checkpoint_ids") or []
    typer.echo(f"lake: {report.get('lake_uri')}")
    typer.echo(
        f"lerobot checkpoint hold: {report.get('action')} {report.get('hold_id')} "
        f"active={report.get('active')} checkpoints={len(checkpoint_ids)}"
    )
    if selector:
        typer.echo("selector: " + json.dumps(selector, sort_keys=True))
    if report.get("reason"):
        typer.echo(f"reason: {report.get('reason')}")
    if report.get("released_at"):
        typer.echo(f"released: {report.get('released_at')} by {report.get('released_by')}")


def _print_lerobot_claim_recovery_report(report: dict) -> None:
    typer.echo(f"lake: {report.get('lake_uri')}")
    typer.echo(
        f"lerobot claim recovery: {report.get('action')} -> {report.get('status')} "
        f"({report.get('phase')}); stale={report.get('stale')} force={report.get('force')}"
    )
    typer.echo(f"job: {report.get('job_id')}")
    typer.echo(
        f"previous: checkpoint={report.get('previous_checkpoint_id')} "
        f"owner={report.get('previous_owner')} token={report.get('previous_token')} "
        f"expires={report.get('previous_expires_at')}"
    )
    typer.echo(
        f"recovery: checkpoint={report.get('recovery_checkpoint_id')} "
        f"owner={report.get('new_owner')} token={report.get('new_token')} "
        f"at={report.get('recovered_at')}"
    )


def _print_lerobot_claim_watchdog_report(report: dict) -> None:
    typer.echo(f"lake: {report.get('lake_uri')}")
    typer.echo(
        "lerobot claim watchdog: "
        f"stale={report.get('stale_count')} live={report.get('live_count')} "
        f"inactive={report.get('inactive_count')} total={report.get('total_jobs')}"
    )
    for finding in report.get("stale_claims") or ():
        typer.echo(
            "  stale "
            f"{finding.get('job_id')} owner={finding.get('claim_owner')} "
            f"reason={finding.get('stale_reason')} expires={finding.get('claim_expires_at')} "
            f"rows={finding.get('rows_seen')} obs={finding.get('observations_written')}"
        )
        if finding.get("suggested_recovery_command"):
            typer.echo(f"    recover: {finding.get('suggested_recovery_command')}")
    for finding in report.get("live_claims") or ():
        typer.echo(
            "  live "
            f"{finding.get('job_id')} owner={finding.get('claim_owner')} "
            f"expires={finding.get('claim_expires_at')} "
            f"seconds_until_stale={finding.get('seconds_until_stale')}"
        )


def _print_lerobot_claim_recovery_chaos_report(report: dict) -> None:
    workload = report.get("workload") or {}
    watchdog = report.get("watchdog") or {}
    recovery = report.get("recovery") or {}
    recommendations = report.get("recommendations") or {}
    typer.echo(
        "lerobot claim recovery simulation: "
        f"{report.get('mode')} scenario={report.get('scenario')} passed={report.get('passed')}"
    )
    typer.echo(
        f"jobs={workload.get('jobs_seen')} checkpoint_rows={workload.get('checkpoint_rows')} "
        f"stale={watchdog.get('stale_count')} live={watchdog.get('live_count')} "
        f"inactive={watchdog.get('inactive_count')}"
    )
    typer.echo(
        f"recovery: accepted={recovery.get('accepted_recoveries')} "
        f"cas_conflicts={recovery.get('cas_conflicts')} "
        f"latency_s={recovery.get('estimated_recovery_latency_seconds')}"
    )
    typer.echo(
        f"recommended profile={recommendations.get('profile')} "
        f"lease_s={recommendations.get('lease_timeout_seconds')} "
        f"heartbeat_s={recommendations.get('heartbeat_interval_seconds')}"
    )
    for crash in report.get("crash_points") or ():
        typer.echo(
            "  "
            f"{crash.get('crash_point')}: checkpoints "
            f"{crash.get('checkpoint_rows_before')}->{crash.get('checkpoint_rows_after')} "
            f"obs_skip={crash.get('rows_skipped_existing')} "
            f"duplicates={sum((crash.get('duplicate_rows') or {}).values())}"
        )
    for warning in report.get("warnings") or ():
        typer.echo(
            f"warning[{warning.get('level')}]: {warning.get('metric')} "
            f"actual={warning.get('actual')} threshold={warning.get('threshold')}"
        )


def _lerobot_claim_watchdog_markdown(report: dict) -> str:
    lines = [
        "# LeRobot Claim Watchdog",
        "",
        f"- Lake: `{report.get('lake_uri')}`",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Stale claims: {report.get('stale_count')}",
        f"- Live claims: {report.get('live_count')}",
        f"- Inactive jobs: {report.get('inactive_count')}",
        "",
    ]
    stale_claims = report.get("stale_claims") or []
    if stale_claims:
        lines.extend(["## Stale Claims", ""])
        for finding in stale_claims:
            lines.extend(
                [
                    f"### `{finding.get('job_id')}`",
                    "",
                    f"- Source: `{finding.get('source_id')}`",
                    f"- Owner: `{finding.get('claim_owner')}`",
                    f"- Reason: `{finding.get('stale_reason')}`",
                    f"- Last heartbeat: `{finding.get('last_heartbeat_at')}`",
                    f"- Expires: `{finding.get('claim_expires_at')}`",
                    f"- Rows seen: {finding.get('rows_seen')}",
                    f"- Observations written: {finding.get('observations_written')}",
                    "",
                    "```bash",
                    str(finding.get("suggested_recovery_command") or ""),
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(["## Stale Claims", "", "No stale running claims detected.", ""])
    live_claims = report.get("live_claims") or []
    if live_claims:
        lines.extend(["## Live Claims", ""])
        for finding in live_claims:
            lines.append(
                f"- `{finding.get('job_id')}` owner `{finding.get('claim_owner')}` "
                f"expires `{finding.get('claim_expires_at')}`"
            )
        lines.append("")
    inactive_jobs = report.get("inactive_jobs") or []
    if inactive_jobs:
        lines.extend(["## Inactive Jobs", ""])
        for finding in inactive_jobs:
            lines.append(
                f"- `{finding.get('job_id')}` latest status `{finding.get('status')}` "
                f"phase `{finding.get('phase')}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _json_ready(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value
