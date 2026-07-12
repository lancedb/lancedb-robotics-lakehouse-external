"""Replayable bundle exporters driven by an evidence-pack manifest (backlog 0107).

Backlog 0065 makes evidence packs deterministic and source-coordinate complete,
but stops at metadata manifests plus selected Lance blob materialization. This
module turns a pack manifest into bundles a human can open directly:

- ``mcap`` slice bundles reconstructed from a pack's ``source_coordinates`` (one
  deterministic clip per source URI, restricted to the referenced channels and
  the ``[min, max]`` log-time window), so the referenced messages round-trip
  into any MCAP-aware, Foxglove, or Rerun workflow.
- codec-aware video clip/GOP byte extraction from a pack's
  ``video_encoding_refs``, sliced by the stored keyframe map, with per-object
  byte counts and SHA-256 hashes.

Determinism is preserved from the parent pack: stable output paths, a byte count
and SHA-256 for every emitted object, the source table-version pins, and the
parent evidence-pack manifest digest are all recorded in the bundle manifest,
whose own digest is stable for unchanged inputs. Large media stays opt-in and is
bounded by user-supplied ``max_bytes`` / ``max_files`` limits, enforced before
any partial success is reported: a missing source or a broken limit fails with
an :class:`~lancedb_robotics.evidence.EvidencePackError` and leaves no partial
bundle behind.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from lancedb_robotics.evidence import (
    SUPPORTED_EVIDENCE_PACK_SCHEMAS,
    EvidencePackError,
    _json_ready,
    _safe_name,
    _stable_sha256,
    is_supported_evidence_schema,
)
from lancedb_robotics.keyframe_maps import KeyframeMapError, keyframe_map_entries_for_encoding
from lancedb_robotics.lake import Lake
from lancedb_robotics.video import VIDEO_ENCODING_BLOB_COLUMN

REPLAY_BUNDLE_SCHEMA = "lancedb-robotics/replay-bundle/v1"

DEFAULT_VIEWER_FORMATS = ("foxglove", "rerun")
_MCAP_EXT = "mcap"
_VIDEO_CLIP_EXT = "lrbgop"


@dataclass(frozen=True)
class ReplayBundleReport:
    """JSON-ready report returned by the replay-bundle exporters."""

    lake_uri: str
    subject: str
    output_dir: str
    manifest_path: str
    parent_evidence_pack_digest: str
    bundle_digest: str
    bytes_written: int
    file_count: int
    files: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "subject": self.subject,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "parent_evidence_pack_digest": self.parent_evidence_pack_digest,
            "bundle_digest": self.bundle_digest,
            "bytes_written": self.bytes_written,
            "file_count": self.file_count,
            "files": list(self.files),
            "manifest": self.manifest,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


def build_replay_bundle(
    lake: Lake,
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path,
    include_mcap: bool = True,
    include_video: bool = False,
    include_gops: bool = True,
    viewer_formats: Sequence[str] = DEFAULT_VIEWER_FORMATS,
    max_bytes: int | None = None,
    max_files: int | None = None,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> ReplayBundleReport:
    """Export a deterministic replay bundle from an evidence-pack manifest.

    ``manifest`` is a v1 evidence-pack manifest (``EvidencePackReport.manifest``
    or a reloaded ``manifest.json``). ``include_mcap`` reconstructs one MCAP
    slice per source URI from the pack's ``source_coordinates``; ``include_video``
    extracts codec-aware clip/GOP bytes from ``video_encoding_refs``.

    Nothing is written until the whole bundle is planned and validated. A missing
    source, an unresolvable slice window, or a busted ``max_bytes`` / ``max_files``
    limit raises :class:`EvidencePackError` and removes any partial output rather
    than reporting a half-written bundle.
    """

    schema = manifest.get("schema_version")
    if not is_supported_evidence_schema(schema):
        raise EvidencePackError(
            f"unsupported evidence-pack schema {schema!r}; "
            f"expected one of {SUPPORTED_EVIDENCE_PACK_SCHEMAS}"
        )
    if not include_mcap and not include_video:
        raise EvidencePackError("nothing to export: enable include_mcap and/or include_video")

    parent_digest = _stable_sha256(manifest)
    destination = Path(output_dir)

    # Plan first so a bad source or limit fails before any byte is written.
    mcap_plan = _plan_mcap_slices(manifest) if include_mcap else []
    video_plan = (
        _plan_video_clips(lake, manifest, include_gops=include_gops) if include_video else []
    )
    if not mcap_plan and not video_plan:
        raise EvidencePackError(
            "evidence pack has no source coordinates or video encoding refs to export"
        )

    planned_file_count = len(mcap_plan) + sum(1 + len(item["gops"]) for item in video_plan)
    _enforce_file_limit(planned_file_count, max_files)
    _enforce_byte_limit(sum(item["planned_bytes"] for item in video_plan), max_bytes)

    created_root = not destination.exists()
    files: list[dict[str, Any]] = []
    mcap_entries: list[dict[str, Any]] = []
    video_entries: list[dict[str, Any]] = []
    try:
        destination.mkdir(parents=True, exist_ok=True)
        for plan in mcap_plan:
            entry, file_record = _write_mcap_slice(
                plan,
                destination,
                viewer_formats=viewer_formats,
                storage_options=storage_options,
                auth_ref=auth_ref,
            )
            mcap_entries.append(entry)
            files.append(file_record)
            _enforce_byte_limit(_total_bytes(files), max_bytes)
        for plan in video_plan:
            entry, file_records = _write_video_clip(plan, destination)
            video_entries.append(entry)
            files.extend(file_records)
            _enforce_byte_limit(_total_bytes(files), max_bytes)
    except Exception:
        _cleanup(destination, created_root=created_root)
        raise

    files = sorted(files, key=lambda item: (item["kind"], item["path"]))
    bundle_manifest = _bundle_manifest(
        manifest,
        parent_digest=parent_digest,
        mcap_slices=mcap_entries,
        video_clips=video_entries,
        files=files,
        limits={"max_bytes": max_bytes, "max_files": max_files},
    )
    bundle_manifest["bundle_digest"] = _stable_sha256(bundle_manifest)

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(bundle_manifest), indent=2, sort_keys=True) + "\n"
    )

    subject = manifest.get("subject") or {}
    return ReplayBundleReport(
        lake_uri=str(manifest.get("lake_uri") or lake.uri),
        subject=str(subject.get("handle") or subject.get("model_run_id") or ""),
        output_dir=str(destination),
        manifest_path=str(manifest_path),
        parent_evidence_pack_digest=parent_digest,
        bundle_digest=bundle_manifest["bundle_digest"],
        bytes_written=_total_bytes(files),
        file_count=len(files),
        files=tuple(files),
        manifest=_json_ready(bundle_manifest),
    )


# --- MCAP slice planning / writing -----------------------------------------


def _plan_mcap_slices(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Group source coordinates + payload refs into one slice plan per URI."""

    channels: dict[str, set[str]] = defaultdict(set)
    times: dict[str, list[int]] = defaultdict(list)
    observation_ids: dict[str, set[str]] = defaultdict(set)

    for coord in manifest.get("source_coordinates") or []:
        uri = coord.get("uri")
        if not uri:
            continue
        channel = coord.get("channel")
        if channel:
            channels[uri].add(str(channel))
        log_time = coord.get("log_time_ns")
        if log_time is not None:
            times[uri].append(int(log_time))
        for observation_id in coord.get("observation_ids") or []:
            if observation_id:
                observation_ids[uri].add(str(observation_id))

    for ref in manifest.get("payload_refs") or []:
        uri = ref.get("raw_uri")
        if not uri:
            continue
        channel = ref.get("raw_channel")
        if channel:
            channels[uri].add(str(channel))
        log_time = ref.get("raw_log_time_ns")
        if log_time is not None:
            times[uri].append(int(log_time))
        row_id = ref.get("row_id")
        if row_id:
            observation_ids[uri].add(str(row_id))

    plans: list[dict[str, Any]] = []
    for uri in sorted(set(channels) | set(times)):
        window = times.get(uri) or []
        if not window:
            raise EvidencePackError(
                f"cannot plan a deterministic mcap slice for {uri!r}: "
                "no source log-time coordinates are recorded in the evidence pack"
            )
        if _is_local_uri(uri) and not _local_source_exists(uri):
            raise EvidencePackError(f"missing source bytes for mcap slice: {uri}")
        plans.append(
            {
                "uri": uri,
                "channels": sorted(channels.get(uri, set())),
                "start_time_ns": min(window),
                "end_time_ns": max(window),
                "observation_ids": sorted(observation_ids.get(uri, set())),
            }
        )
    return plans


def _write_mcap_slice(
    plan: Mapping[str, Any],
    output_dir: Path,
    *,
    viewer_formats: Sequence[str],
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from lancedb_robotics.adapters import AdapterError

    uri = plan["uri"]
    channels = tuple(plan["channels"])
    out_path = output_dir / "mcap" / _slice_basename(uri)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = _slice_source(
            uri,
            start_time_ns=plan["start_time_ns"],
            end_time_ns=plan["end_time_ns"],
            out_path=out_path,
            topics=channels,
            storage_options=storage_options,
            auth_ref=auth_ref,
        )
    except AdapterError as exc:
        raise EvidencePackError(f"failed to slice mcap for {uri}: {exc}") from exc

    payload = out_path.read_bytes()
    rel_path = str(out_path.relative_to(output_dir))
    file_record = {
        "kind": "mcap-slice",
        "uri": uri,
        "path": rel_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    entry = {
        "uri": uri,
        "path": rel_path,
        "channels": list(channels),
        "start_time_ns": plan["start_time_ns"],
        "end_time_ns": plan["end_time_ns"],
        "message_count": result["message_count"],
        "written_topics": list(result["topics"]),
        "observation_ids": list(plan["observation_ids"]),
        "bytes": file_record["bytes"],
        "sha256": file_record["sha256"],
        "external_links": _viewer_links(rel_path, viewer_formats),
    }
    return entry, file_record


def _slice_source(
    uri: str,
    *,
    start_time_ns: int,
    end_time_ns: int,
    out_path: Path,
    topics: tuple[str, ...],
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, Any]:
    """Slice ``[start, end]`` from a local (single/split) or object-store source."""

    from lancedb_robotics.adapters import get_adapter

    if _is_local_uri(uri):
        from lancedb_robotics.recordings import export_window, resolve_shards

        plan = resolve_shards(_local_path(uri))
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
    return get_adapter("mcap").export(
        uri,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        out_path=out_path,
        topics=topics,
        storage_options=dict(storage_options) if storage_options else None,
        auth_ref=auth_ref,
    )


# --- Video clip / GOP planning + writing -----------------------------------


def _plan_video_clips(
    lake: Lake,
    manifest: Mapping[str, Any],
    *,
    include_gops: bool,
) -> list[dict[str, Any]]:
    refs = manifest.get("video_encoding_refs") or []
    if not refs:
        return []
    table_version = _table_version(manifest, "video_encodings")
    artifact_table_version = _table_version(manifest, "keyframe_map_artifacts")
    encoding_ids = [ref["encoding_id"] for ref in refs if ref.get("encoding_id")]
    rows = _encoding_rows(lake, encoding_ids, table_version)
    blobs = _fetch_encoding_blobs(lake, encoding_ids, table_version)

    plans: list[dict[str, Any]] = []
    for ref in sorted(refs, key=lambda item: str(item.get("encoding_id"))):
        encoding_id = ref.get("encoding_id")
        row = rows.get(str(encoding_id))
        blob = blobs.get(str(encoding_id))
        if row is None or not blob:
            raise EvidencePackError(f"missing source bytes for video encoding {encoding_id!r}")
        ref_artifact_version = _keyframe_map_artifact_version(ref, artifact_table_version)
        try:
            keyframe_map = keyframe_map_entries_for_encoding(
                lake,
                row,
                artifact_table_version=ref_artifact_version,
            )
        except (KeyframeMapError, json.JSONDecodeError) as exc:
            raise EvidencePackError(
                f"cannot resolve keyframe map for video encoding {encoding_id!r}: {exc}"
            ) from exc
        gops = _plan_gops(keyframe_map, blob, encoding_id) if include_gops else []
        plans.append(
            {
                "encoding_id": str(encoding_id),
                "video_id": row.get("video_id"),
                "codec": row.get("codec"),
                "gop_size": row.get("gop_size"),
                "frame_count": row.get("frame_count"),
                "table_version": table_version,
                "keyframe_map_ref": row.get("keyframe_map_ref"),
                "keyframe_map_artifact_table_version": ref_artifact_version,
                "blob": blob,
                "gops": gops,
                "planned_bytes": len(blob) + sum(len(item["bytes"]) for item in gops),
            }
        )
    return plans


def _plan_gops(
    keyframe_map: list[dict[str, Any]],
    blob: bytes,
    encoding_id: Any,
) -> list[dict[str, Any]]:
    gops: list[dict[str, Any]] = []
    for entry in sorted(keyframe_map, key=lambda item: int(item.get("gop_index", 0))):
        start = int(entry["byte_start"])
        end = int(entry["byte_end"])
        if start < 0 or end > len(blob) or start > end:
            raise EvidencePackError(
                f"video encoding {encoding_id!r} has a corrupt keyframe map: "
                f"gop byte range [{start}, {end}] outside blob of {len(blob)} bytes"
            )
        gops.append(
            {
                "gop_index": int(entry.get("gop_index", 0)),
                "first_frame_index": int(entry.get("first_frame_index", 0)),
                "last_frame_index": int(entry.get("last_frame_index", 0)),
                "frame_count": int(entry.get("frame_count", 0)),
                "byte_start": start,
                "byte_end": end,
                "bytes": blob[start:end],
            }
        )
    return gops


def _write_video_clip(
    plan: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    encoding_id = plan["encoding_id"]
    clip_dir = output_dir / "video" / _safe_name(encoding_id)
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_blob = plan["blob"]
    clip_path = clip_dir / f"clip.{_VIDEO_CLIP_EXT}"
    clip_record = _write_file(
        clip_path,
        clip_blob,
        output_dir,
        kind="video-clip",
        extra={"encoding_id": encoding_id, "codec": plan["codec"]},
    )

    file_records = [clip_record]
    gop_entries: list[dict[str, Any]] = []
    for gop in plan["gops"]:
        gop_path = clip_dir / f"gop-{gop['gop_index']:04d}.bin"
        gop_record = _write_file(
            gop_path,
            gop["bytes"],
            output_dir,
            kind="video-gop",
            extra={"encoding_id": encoding_id, "gop_index": gop["gop_index"]},
        )
        file_records.append(gop_record)
        gop_entries.append(
            {
                "gop_index": gop["gop_index"],
                "first_frame_index": gop["first_frame_index"],
                "last_frame_index": gop["last_frame_index"],
                "frame_count": gop["frame_count"],
                "byte_start": gop["byte_start"],
                "byte_end": gop["byte_end"],
                "path": gop_record["path"],
                "bytes": gop_record["bytes"],
                "sha256": gop_record["sha256"],
            }
        )

    entry = {
        "encoding_id": encoding_id,
        "video_id": plan["video_id"],
        "codec": plan["codec"],
        "gop_size": plan["gop_size"],
        "frame_count": plan["frame_count"],
        "table_version": plan["table_version"],
        "column": VIDEO_ENCODING_BLOB_COLUMN,
        "keyframe_map_ref": plan.get("keyframe_map_ref"),
        "keyframe_map_artifact_table_version": plan.get("keyframe_map_artifact_table_version"),
        "clip": {
            "path": clip_record["path"],
            "bytes": clip_record["bytes"],
            "sha256": clip_record["sha256"],
        },
        "gops": gop_entries,
    }
    return entry, file_records


def _encoding_rows(
    lake: Lake,
    encoding_ids: Sequence[str],
    table_version: int | None,
) -> dict[str, dict[str, Any]]:
    if not encoding_ids:
        return {}
    wanted = set(encoding_ids)
    table = lake.table("video_encodings")
    if table_version is not None:
        table.checkout(int(table_version))
    try:
        columns = [name for name in table.schema.names if name != VIDEO_ENCODING_BLOB_COLUMN]
        rows = table.to_lance().to_table(columns=columns).to_pylist()
    finally:
        if table_version is not None:
            table.checkout_latest()
    return {str(row["encoding_id"]): row for row in rows if str(row.get("encoding_id")) in wanted}


def _fetch_encoding_blobs(
    lake: Lake,
    encoding_ids: Sequence[str],
    table_version: int | None,
) -> dict[str, bytes]:
    from lancedb_robotics.blob import fetch_blobs

    if not encoding_ids:
        return {}
    table = lake.table("video_encodings")
    if table_version is not None:
        table.checkout(int(table_version))
    try:
        return fetch_blobs(
            table,
            VIDEO_ENCODING_BLOB_COLUMN,
            encoding_ids,
            id_column="encoding_id",
            connection_spec=lake.connection_spec,
        )
    finally:
        if table_version is not None:
            table.checkout_latest()


# --- Manifest assembly + shared helpers ------------------------------------


def _bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    parent_digest: str,
    mcap_slices: list[dict[str, Any]],
    video_clips: list[dict[str, Any]],
    files: list[dict[str, Any]],
    limits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REPLAY_BUNDLE_SCHEMA,
        "lake_uri": manifest.get("lake_uri"),
        "subject": _json_ready(manifest.get("subject") or {}),
        "parent_evidence_pack": {
            "schema_version": manifest.get("schema_version"),
            "manifest_digest": parent_digest,
        },
        "source_table_versions": _json_ready(manifest.get("table_versions") or []),
        "mcap_slices": _json_ready(mcap_slices),
        "video_clips": _json_ready(video_clips),
        "files": _json_ready(files),
        "limits": limits,
        "totals": {"files": len(files), "bytes": _total_bytes(files)},
    }


def _viewer_links(target: str, viewer_formats: Sequence[str]) -> list[dict[str, str]]:
    """Illustrative bundle-relative replay-tool hints (Foxglove/Rerun/…)."""

    return [
        {"tool": tool, "kind": "open-file", "target": target}
        for tool in sorted({str(fmt) for fmt in viewer_formats})
    ]


def _write_file(
    path: Path,
    payload: bytes,
    output_dir: Path,
    *,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    record = {
        "kind": kind,
        "path": str(path.relative_to(output_dir)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if extra:
        record.update(dict(extra))
    return record


def _table_version(manifest: Mapping[str, Any], table_name: str) -> int | None:
    for row in manifest.get("table_versions") or []:
        if row.get("table") == table_name and row.get("version") is not None:
            return int(row["version"])
    return None


def _keyframe_map_artifact_version(
    ref: Mapping[str, Any],
    manifest_table_version: int | None,
) -> int | None:
    value = ref.get("keyframe_map_artifact_table_version")
    if value is not None:
        return int(value)
    if ref.get("keyframe_map_storage") == "artifact":
        return manifest_table_version
    return None


def _slice_basename(uri: str) -> str:
    parsed = urlparse(uri)
    stem = Path(parsed.path or uri).name or "source"
    digest = hashlib.sha256(uri.encode()).hexdigest()[:12]
    return f"{_safe_name(stem)}-{digest}.{_MCAP_EXT}"


def _is_local_uri(uri: str) -> bool:
    scheme = urlparse(uri).scheme
    return scheme in ("", "file") or (len(scheme) == 1)  # Windows drive letters


def _local_path(uri: str) -> Path:
    from urllib.parse import unquote

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _local_source_exists(uri: str) -> bool:
    return _local_path(uri).exists()


def _total_bytes(files: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(item["bytes"]) for item in files)


def _enforce_file_limit(planned: int, max_files: int | None) -> None:
    if max_files is not None and planned > max_files:
        raise EvidencePackError(
            f"replay bundle would emit {planned} files, exceeding max_files={max_files}"
        )


def _enforce_byte_limit(planned: int, max_bytes: int | None) -> None:
    if max_bytes is not None and planned > max_bytes:
        raise EvidencePackError(
            f"replay bundle would write {planned} bytes, exceeding max_bytes={max_bytes}"
        )


def _cleanup(destination: Path, *, created_root: bool) -> None:
    """Remove partial bundle output so a failed export leaves nothing behind."""

    if created_root:
        shutil.rmtree(destination, ignore_errors=True)
        return
    for sub in ("mcap", "video"):
        shutil.rmtree(destination / sub, ignore_errors=True)
    manifest = destination / "manifest.json"
    if manifest.exists():
        manifest.unlink()
