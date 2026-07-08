"""`lancedb-robotics inspect` subcommands."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import typer

from lancedb_robotics.lerobot_object_store_validation import (
    DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
)

inspect_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_PATH_ARGUMENT = typer.Argument(
    ..., help="Path or object-store URI to an MCAP file, split directory, or metadata.yaml."
)
_FORMAT_OPTION = typer.Option("json", "--format", help="Output format: json or text.")
_AUTH_REF_OPTION = typer.Option(None, "--auth-ref", help="Raw source object credential reference.")
_STORAGE_OPTION = typer.Option(
    None,
    "--storage-option",
    help="Raw source storage client option as key=value; repeat for endpoint_url, region, etc.",
)
_MEDIA_INSPECTION_TIMEOUT_OPTION = typer.Option(
    None,
    "--media-inspection-timeout-seconds",
    help="Per-video LeRobot MP4 metadata inspection timeout; <=0 disables the wall-clock guard.",
)
_MEDIA_INSPECTION_WORKERS_OPTION = typer.Option(
    None,
    "--media-inspection-workers",
    min=1,
    help="Maximum concurrent LeRobot media-inspection workers.",
)
_MEDIA_INSPECTION_RETRIES_OPTION = typer.Option(
    0,
    "--media-inspection-retries",
    help="Retries per LeRobot video after transient MP4 metadata read failures.",
)
_MEDIA_INSPECTION_RETRY_BACKOFF_OPTION = typer.Option(
    0.0,
    "--media-inspection-retry-backoff-seconds",
    help="Sleep seconds between LeRobot media-inspection retry attempts.",
)
_MEDIA_INSPECTION_EXECUTION_MODE_OPTION = typer.Option(
    "thread",
    "--media-inspection-execution-mode",
    help="LeRobot media-inspection execution mode: thread or process.",
)
_SOURCE_MANIFEST_CACHE_OPTION = typer.Option(
    None,
    "--source-manifest-cache",
    help="JSON cache path for object-store LeRobot source listings; omit to list directly.",
)
_SOURCE_VALIDATION_POLICY_OPTION = typer.Option(
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
    "--source-validation-policy",
    help=(
        "LeRobot object-store source validation policy: metadata-only, "
        "sampled-validation, or strict-content-hash."
    ),
)
_SOURCE_VALIDATION_SAMPLE_COUNT_OPTION = typer.Option(
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
    "--source-validation-sample-count",
    min=0,
    help="Objects to sample for LeRobot sampled-validation.",
)
_SOURCE_VALIDATION_SAMPLE_BYTES_OPTION = typer.Option(
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
    "--source-validation-sample-bytes",
    min=1,
    help="Prefix bytes to hash from each sampled LeRobot object-store object.",
)
_SOURCE_VALIDATION_STRICT_MAX_BYTES_OPTION = typer.Option(
    DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    "--strict-content-hash-max-bytes",
    min=0,
    help="Maximum total bytes strict-content-hash may read from a LeRobot object-store source.",
)
_LEROBOT_OBJECT_STORE_ROOT_OPTION = typer.Option(
    None,
    "--root",
    help="LeRobot object-store root URI to probe; repeat for S3/GCS/Azure prefixes.",
)
_LEROBOT_OBJECT_STORE_SCHEME_OPTION = typer.Option(
    None,
    "--scheme",
    help="Object-store scheme to include when no root is supplied; repeat for s3/gs/gcs/az/abfs/abfss.",
)
_LEROBOT_OBJECT_STORE_FORMAT_OPTION = typer.Option(
    "text", "--format", help="Output format: json, text, or markdown."
)
_LEROBOT_OBJECT_STORE_PROVIDER_DEFAULT_OPTION = typer.Option(
    True,
    "--provider-default/--no-provider-default",
    help="Also probe the provider's default credential chain without explicit options/auth_ref.",
)
_LEROBOT_OBJECT_STORE_INSPECT_VIDEOS_OPTION = typer.Option(
    False,
    "--inspect-videos/--no-inspect-videos",
    help="Include LeRobot MP4 media metadata inspection in the adapter probe.",
)
_LEROBOT_OBJECT_STORE_STRICT_OPTION = typer.Option(
    False,
    "--strict",
    help="Exit 3 when any probed conformance case fails.",
)


@inspect_app.command("mcap")
def inspect_mcap(path: str = _PATH_ARGUMENT, format: str = _FORMAT_OPTION) -> None:
    """Describe an MCAP file or split recording (topics, counts, range) without ingesting."""
    from lancedb_robotics.adapters import AdapterError
    from lancedb_robotics.recordings import inspect_source

    if format not in ("json", "text"):
        typer.echo(f"error: unknown format {format!r}; expected json or text", err=True)
        raise typer.Exit(code=1)

    try:
        report = inspect_source(path)
    except AdapterError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_report(report, format=format)


@inspect_app.command("rosbag")
def inspect_rosbag(path: str = _PATH_ARGUMENT, format: str = _FORMAT_OPTION) -> None:
    """Describe a ROS1 `.bag` or ROS2 sqlite `.db3` source without ingesting."""
    from lancedb_robotics.adapters import AdapterError, get_adapter

    if format not in ("json", "text"):
        typer.echo(f"error: unknown format {format!r}; expected json or text", err=True)
        raise typer.Exit(code=1)

    try:
        report = get_adapter("rosbag").inspect(path)
    except AdapterError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _print_report(report, format=format)


@inspect_app.command("lerobot")
def inspect_lerobot(
    path: str = typer.Argument(..., help="Local/object-store LeRobot dataset directory or HF Hub repo id."),
    format: str = _FORMAT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    source_manifest_cache: Path | None = _SOURCE_MANIFEST_CACHE_OPTION,
    source_validation_policy: str = _SOURCE_VALIDATION_POLICY_OPTION,
    source_validation_sample_count: int = _SOURCE_VALIDATION_SAMPLE_COUNT_OPTION,
    source_validation_sample_bytes: int = _SOURCE_VALIDATION_SAMPLE_BYTES_OPTION,
    strict_content_hash_max_bytes: int = _SOURCE_VALIDATION_STRICT_MAX_BYTES_OPTION,
    media_inspection_timeout_seconds: float | None = _MEDIA_INSPECTION_TIMEOUT_OPTION,
    media_inspection_workers: int | None = _MEDIA_INSPECTION_WORKERS_OPTION,
    media_inspection_retries: int = _MEDIA_INSPECTION_RETRIES_OPTION,
    media_inspection_retry_backoff_seconds: float = _MEDIA_INSPECTION_RETRY_BACKOFF_OPTION,
    media_inspection_execution_mode: str = _MEDIA_INSPECTION_EXECUTION_MODE_OPTION,
) -> None:
    """Describe a LeRobot dataset without ingesting it."""
    from lancedb_robotics.adapters import AdapterError, get_adapter
    from lancedb_robotics.storage import parse_storage_option_pairs

    if format not in ("json", "text"):
        typer.echo(f"error: unknown format {format!r}; expected json or text", err=True)
        raise typer.Exit(code=1)

    try:
        storage_options = parse_storage_option_pairs(storage_option)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        report = get_adapter("lerobot").inspect(
            path,
            storage_options=storage_options,
            auth_ref=auth_ref,
            media_inspection_workers=media_inspection_workers,
            media_inspection_timeout_seconds=media_inspection_timeout_seconds,
            media_inspection_retries=media_inspection_retries,
            media_inspection_retry_backoff_seconds=media_inspection_retry_backoff_seconds,
            media_inspection_execution_mode=media_inspection_execution_mode,
            source_manifest_cache=source_manifest_cache,
            object_store_validation_policy=source_validation_policy,
            object_store_validation_sample_count=source_validation_sample_count,
            object_store_validation_sample_bytes=source_validation_sample_bytes,
            object_store_validation_strict_max_bytes=strict_content_hash_max_bytes,
        )
    except AdapterError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if format == "text":
        _print_lerobot_report(report)
    else:
        _print_report(report, format=format)


@inspect_app.command("lerobot-object-store-conformance")
def inspect_lerobot_object_store_conformance(
    root: list[str] | None = _LEROBOT_OBJECT_STORE_ROOT_OPTION,
    scheme: list[str] | None = _LEROBOT_OBJECT_STORE_SCHEME_OPTION,
    format: str = _LEROBOT_OBJECT_STORE_FORMAT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    provider_default: bool = _LEROBOT_OBJECT_STORE_PROVIDER_DEFAULT_OPTION,
    inspect_videos: bool = _LEROBOT_OBJECT_STORE_INSPECT_VIDEOS_OPTION,
    strict: bool = _LEROBOT_OBJECT_STORE_STRICT_OPTION,
) -> None:
    """Probe LeRobot object-store auth, listing, metadata, and ingest preflight behavior."""
    from lancedb_robotics.lerobot_object_store_conformance import (
        lerobot_object_store_conformance,
    )
    from lancedb_robotics.storage import parse_storage_option_pairs

    if format not in ("json", "text", "markdown"):
        typer.echo(f"error: unknown format {format!r}; expected json, text, or markdown", err=True)
        raise typer.Exit(code=1)

    try:
        storage_options = (
            parse_storage_option_pairs(storage_option) if storage_option is not None else None
        )
        report = lerobot_object_store_conformance(
            roots=root,
            schemes=scheme,
            storage_options=storage_options,
            auth_ref=auth_ref,
            include_provider_default=provider_default,
            inspect_videos=inspect_videos,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    payload = report.to_params()
    if format == "json":
        typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
    elif format == "markdown":
        typer.echo(_lerobot_object_store_conformance_markdown(payload))
    else:
        _print_lerobot_object_store_conformance_report(payload)
    if strict and report.failed_count:
        raise typer.Exit(code=3)


def _print_report(report: dict, *, format: str) -> None:
    if format == "json":
        typer.echo(json.dumps(report, indent=2, sort_keys=False))
        return
    duration_s = report["duration_ns"] / 1e9
    typer.echo(f"{report['path']}")
    if report.get("is_split"):
        typer.echo(f"  split recording: {report['shard_count']} shards")
    if report.get("adapter") == "rosbag":
        typer.echo(f"  ROS bag: {report['profile']} / {report.get('storage_identifier')}")
    typer.echo(
        f"  {report['message_count']} messages, {report['channel_count']} topics, "
        f"{duration_s:.1f}s ({report['start_time_ns']} .. {report['end_time_ns']})"
    )
    for topic in report["topics"]:
        decode = "decodable" if topic["can_decode"] else "no decoder"
        typer.echo(
            f"  {topic['topic']}\t{topic['message_encoding']}\t{topic['schema_name']}\t"
            f"{topic['message_count']} msgs\t{decode}"
        )
    # MCAP's other first-class records (backlog 0016), shown only when present.
    for record in report.get("metadata", []):
        typer.echo(f"  metadata: {record['name']}\t{', '.join(record['keys'])}")
    for attachment in report.get("attachments", []):
        typer.echo(
            f"  attachment: {attachment['name']}\t{attachment['media_type']}\t"
            f"{attachment['size']} bytes"
        )
    # Split-recording inventory (backlog 0019), shown only for a directory source.
    for shard in report.get("shards", []):
        if shard.get("readable", True):
            typer.echo(
                f"  shard {shard['index']}: {shard['name']}\t{shard['message_count']} msgs\t"
                f"({shard['start_time_ns']} .. {shard['end_time_ns']})"
            )
        else:
            typer.echo(
                f"  shard {shard['index']}: {shard['name']}\tUNREADABLE ({shard.get('error')})"
            )
    for gap in report.get("gaps", []):
        typer.echo(
            f"  {gap['kind']} between shard {gap['after_shard']} and {gap['before_shard']}: "
            f"{gap['delta_ns']} ns"
        )


def _print_lerobot_report(report: dict) -> None:
    typer.echo(f"{report['path']}")
    typer.echo(
        f"  LeRobot {report['codebase_version']}: {report['episode_count']} episodes, "
        f"{report['frame_count']} frames, fps={report.get('fps')}"
    )
    if report.get("robot_type"):
        typer.echo(f"  robot: {report['robot_type']}")
    cameras = ", ".join(report.get("camera_keys") or []) or "<none>"
    typer.echo(f"  cameras: {cameras}")
    native = report.get("native_loader") or {}
    if not native.get("available", False):
        missing = ", ".join(native.get("missing") or [])
        typer.echo(f"  native loader unavailable: missing {missing}; install {native.get('install')}")
    validation = report.get("object_store_validation") or (
        report.get("source_identity") or {}
    ).get("object_store_validation")
    if validation:
        typer.echo(
            "  object-store validation: "
            f"{validation.get('policy')} ({validation.get('assurance')}), "
            f"samples={validation.get('sample_count')}, "
            f"hashed_bytes={validation.get('hashed_bytes')}"
        )
        for warning in validation.get("warnings") or ():
            typer.echo(f"    {warning.get('severity', 'warning')} {warning.get('code')}: {warning.get('message')}")
    for task in report.get("tasks", []):
        typer.echo(f"  task {task['task_index']}: {task['task']}")
    for episode in report.get("episodes", []):
        tasks = ", ".join(episode.get("tasks") or [])
        typer.echo(
            f"  episode {episode['episode_index']}: {episode['length']} frames"
            + (f"\t{tasks}" if tasks else "")
        )
    for video in report.get("video_files", []):
        keyframes = video.get("keyframe_map") or []
        resolution = video.get("resolution") or "unknown"
        fps = video.get("fps")
        frame_count = video.get("frame_count")
        typer.echo(
            f"  video episode {video['episode_index']} {video['camera_key']}: "
            f"{video.get('codec') or 'unknown'} {resolution}, "
            f"fps={fps if fps is not None else 'unknown'}, "
            f"frames={frame_count if frame_count is not None else 'unknown'}, "
            f"keyframes={len(keyframes)}"
        )
        for diagnostic in video.get("diagnostics") or []:
            typer.echo(
                f"    {diagnostic['severity']} {diagnostic['code']}: {diagnostic['message']}"
            )


def _print_lerobot_object_store_conformance_report(report: dict) -> None:
    summary = report.get("summary") or {}
    typer.echo(
        "lerobot object-store conformance: "
        f"{report.get('overall_status')} "
        f"passed={summary.get('passed')} failed={summary.get('failed')} "
        f"skipped={summary.get('skipped')}"
    )
    for case in report.get("cases") or ():
        ops = case.get("operations") or ()
        passed_ops = sum(1 for operation in ops if operation.get("status") == "passed")
        failed_ops = [operation for operation in ops if operation.get("status") == "failed"]
        metadata = ", ".join(case.get("metadata_fields") or ()) or "<none>"
        root = case.get("root_uri") or f"{case.get('scheme')}://<not supplied>"
        typer.echo(
            f"  {case.get('scheme')} {case.get('auth_mode')} {case.get('status')}: "
            f"{root} ops={passed_ops}/{len(ops)} metadata={metadata}"
        )
        for operation in failed_ops[:2]:
            typer.echo(
                f"    failed {operation.get('name')}: "
                f"{operation.get('error_class')}: {operation.get('message')}"
            )
        for recommendation in case.get("recommendations") or ():
            typer.echo(f"    recommendation: {recommendation}")


def _lerobot_object_store_conformance_markdown(report: dict) -> str:
    lines = [
        "# LeRobot Object-Store Conformance",
        "",
        f"Overall status: **{report.get('overall_status')}**",
        "",
        "| scheme | auth mode | status | root | metadata | failures |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in report.get("cases") or ():
        failures = [
            f"{operation.get('name')} ({operation.get('error_class')})"
            for operation in case.get("operations") or ()
            if operation.get("status") == "failed"
        ]
        metadata = ", ".join(case.get("metadata_fields") or ()) or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("scheme") or ""),
                    str(case.get("auth_mode") or ""),
                    str(case.get("status") or ""),
                    str(case.get("root_uri") or ""),
                    metadata,
                    ", ".join(failures),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Credential Resolution")
    lines.append("")
    for item in report.get("credential_resolution_order") or ():
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
