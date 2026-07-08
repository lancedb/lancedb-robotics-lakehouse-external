"""`lancedb-robotics video` subcommands."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer

video_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_VIDEO_ID_OPTION = typer.Option(None, "--video-id", help="Encode one videos.video_id.")
_EPISODE_ID_OPTION = typer.Option(None, "--episode-id", help="Restrict to one episode_id.")
_CAMERA_KEY_OPTION = typer.Option(None, "--camera-key", help="Restrict to one camera key.")
_GOP_SIZE_OPTION = typer.Option(2, "--gop-size", min=1, help="Frames per GOP/keyframe interval.")
_RESOLUTION_OPTION = typer.Option("unknown", "--resolution", help="Resolution label, e.g. 640x480.")
_NVDEC_OPTION = typer.Option(False, "--nvdec-compatible", help="Mark encoding as NVDEC-compatible.")
_CONFORM_SOURCE_SAMPLES_OPTION = typer.Option(
    None,
    "--samples",
    help="JSON or JSONL source-frame sample specs. Omit to use default lake samples.",
)
_CONFORM_SOURCE_DECODER_OPTION = typer.Option(
    "auto",
    "--decoder",
    help="Decoder backend: auto, pyav, pillow-mjpeg, or a registered backend name.",
)
_CONFORM_SOURCE_FORMAT_OPTION = typer.Option(
    "text",
    "--format",
    help="Output format: text, json, or jsonl.",
)
_FAIL_ON_MISMATCH_OPTION = typer.Option(
    False,
    "--fail-on-mismatch",
    help="Exit non-zero when decoded frames mismatch or decoder errors occur.",
)
_INSTALL_HINT_OPTION = typer.Option(
    True,
    "--install-hint/--no-install-hint",
    help="Show decoder install hints in text output when samples are skipped.",
)


@video_app.command("encode")
def encode(
    lake: str = _LAKE_OPTION,
    video_id: str | None = _VIDEO_ID_OPTION,
    episode_id: str | None = _EPISODE_ID_OPTION,
    camera_key: str | None = _CAMERA_KEY_OPTION,
    gop_size: int = _GOP_SIZE_OPTION,
    resolution: str = _RESOLUTION_OPTION,
    nvdec_compatible: bool = _NVDEC_OPTION,
) -> None:
    """Encode camera videos with GOP/keyframe metadata."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.video import VideoError

    try:
        opened = Lake.open(lake)
        report = opened.video.encode(
            video_id=video_id,
            episode_id=episode_id,
            camera_key=camera_key,
            gop_size=gop_size,
            resolution=resolution,
            nvdec_compatible=nvdec_compatible,
        )
    except (LakeError, VideoError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"lake: {report.lake_uri}")
    typer.echo(f"encodings: {report.encodings_written}")
    typer.echo(f"codec: {report.codec}")
    typer.echo(f"gop size: {report.gop_size}")
    typer.echo(f"transform: {report.transform_id}")
    for encoding_id in report.encoding_ids:
        typer.echo(f"  {encoding_id}")


@video_app.command("inspect")
def inspect(
    lake: str = _LAKE_OPTION,
    video_id: str | None = _VIDEO_ID_OPTION,
) -> None:
    """Inspect codec-aware video encoding rows."""
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        rows = opened.video.encodings(video_id=video_id)
    except LakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    from lancedb_robotics.keyframe_maps import KeyframeMapError, keyframe_map_entries_for_encoding

    typer.echo(f"lake: {lake}")
    typer.echo(f"encodings: {len(rows)}")
    for row in rows:
        try:
            keyframes = keyframe_map_entries_for_encoding(opened, row)
        except KeyframeMapError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"encoding: {row['encoding_id']}")
        typer.echo(f"  video: {row['video_id']}")
        typer.echo(f"  episode: {row['episode_id']}")
        typer.echo(f"  camera: {row['camera_key']}")
        typer.echo(f"  codec: {row['codec']}")
        typer.echo(f"  gop size: {row['gop_size']}")
        typer.echo(f"  resolution: {row['resolution']}")
        typer.echo(f"  fps: {row['fps']}")
        typer.echo(f"  keyframes: {len(keyframes)}")
        typer.echo(f"  encoded bytes: {row['encoded_size_bytes']}")


@video_app.command("conform-source")
def conform_source(
    lake: str = _LAKE_OPTION,
    samples: Path | None = _CONFORM_SOURCE_SAMPLES_OPTION,
    decoder: str = _CONFORM_SOURCE_DECODER_OPTION,
    fail_on_mismatch: bool = _FAIL_ON_MISMATCH_OPTION,
    output_format: str = _CONFORM_SOURCE_FORMAT_OPTION,
    install_hint: bool = _INSTALL_HINT_OPTION,
) -> None:
    """Run decoded source-MP4 frame conformance checks."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.video import VideoError

    output_format = output_format.lower()
    if output_format not in {"text", "json", "jsonl"}:
        typer.echo("error: --format must be text, json, or jsonl", err=True)
        raise typer.Exit(code=1)
    try:
        sample_rows = _load_conform_source_samples(samples)
        opened = Lake.open(lake)
        report = opened.video.conform_source(
            sample_rows,
            decoder=decoder,
            fail_on_mismatch=False,
            created_by="lancedb-robotics-cli",
        )
    except (LakeError, OSError, ValueError, VideoError, json.JSONDecodeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _emit_conform_source_report(report.to_dict(), output_format, install_hint=install_hint)

    if fail_on_mismatch and report.status in {"failed", "error"}:
        raise typer.Exit(code=1)


def _load_conform_source_samples(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if path.suffix.lower() == ".jsonl":
        return _load_conform_source_jsonl(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping) and "samples" in payload:
        payload = payload["samples"]
    elif isinstance(payload, Mapping):
        return [dict(payload)]
    if not isinstance(payload, list):
        raise ValueError("source conformance sample JSON must be a sample object, list, or object with samples")
    return [_coerce_sample_mapping(item, path=str(path)) for item in payload]


def _load_conform_source_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(_coerce_sample_mapping(json.loads(stripped), path=f"{path}:{line_number}"))
    return rows


def _coerce_sample_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"source conformance sample in {path} must be a JSON object")
    return dict(value)


def _emit_conform_source_report(
    payload: dict[str, Any],
    output_format: str,
    *,
    install_hint: bool,
) -> None:
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if output_format == "jsonl":
        for result in payload["results"]:
            typer.echo(json.dumps(result, sort_keys=True))
        return

    counts = payload["status_counts"]
    typer.echo(
        "source conformance: "
        f"status={payload['status']} "
        f"samples={len(payload['results'])} "
        f"passed={counts.get('passed', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"error={counts.get('error', 0)} "
        f"skipped={counts.get('skipped', 0)}"
    )
    typer.echo(f"decoder: {payload['decoder']}")
    typer.echo(f"transform: {payload['transform_id']}")
    for backend, version in sorted(payload["backend_versions"].items()):
        typer.echo(f"backend: {backend} {version}")
    for codec, count in sorted(payload["codec_counts"].items()):
        typer.echo(f"codec: {codec} frames={count}")
    for result in payload["results"]:
        detail = result.get("reason") or result.get("seek_strategy") or "-"
        typer.echo(
            "  "
            f"{result['status']}: "
            f"episode={result['episode_index']} "
            f"frame={result['frame_index']} "
            f"camera={result['camera_key']} "
            f"codec={result['codec']} "
            f"decoder={result.get('decoder_backend') or result.get('decoder_requested')} "
            f"frames_decoded={result.get('frames_decoded', 0)} "
            f"detail={detail}"
        )
    if install_hint:
        for hint in _conform_source_install_hints(payload["results"]):
            typer.echo(f"install hint: {hint}")


def _conform_source_install_hints(results: list[dict[str, Any]]) -> list[str]:
    hints = {
        str(result["error"])
        for result in results
        if result.get("reason") == "decoder-unavailable" and result.get("error")
    }
    return sorted(hints)
