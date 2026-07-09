import hashlib
import io
import json
import struct
import subprocess
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from lancedb_robotics._mp4 import Mp4MetadataError, inspect_mp4_video, read_mp4_frame_sample
from lancedb_robotics.adapters import AdapterError, get_adapter, list_adapters
from lancedb_robotics.cli import app
from lancedb_robotics.dataset import SPLIT_BY_SCENARIO, create_snapshot
from lancedb_robotics.dataset_export import export_dataset_snapshot
from lancedb_robotics.ingest import (
    LeRobotClaimPreconditionError,
    _lerobot_observation_rows,
    apply_lerobot_checkpoint_retention,
    get_lerobot_ingest_job,
    hold_lerobot_checkpoints,
    ingest_lerobot,
    list_lerobot_ingest_jobs,
    plan_lerobot_checkpoint_retention_scale,
    recommend_lerobot_media_inspection_timeouts,
    recover_lerobot_ingest_claim,
    release_lerobot_checkpoint_hold,
    run_lerobot_checkpoint_retention_schedule,
    simulate_lerobot_claim_recovery_chaos,
    watch_lerobot_ingest_claims,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import (
    LEROBOT_INGEST_CHECKPOINTS_SCHEMA,
    LINEAGE_ARTIFACTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
)
from lancedb_robotics.storage import StorageConfigError

runner = CliRunner()


def _install_fake_lerobot_object_store(
    monkeypatch,
    *,
    remote_root: str,
    local_root: Path,
    expected_options: dict | None = None,
):
    expected_options = expected_options or {}
    parsed = remote_root.removeprefix("s3://").rstrip("/")
    opened: list[str] = []
    listed: list[str] = []
    stated: list[str] = []

    def to_local(uri_or_path: str) -> Path:
        value = str(uri_or_path)
        if value.startswith("s3://"):
            value = value.removeprefix("s3://")
        rel = value.removeprefix(parsed).lstrip("/")
        return local_root / rel

    def assert_options(kwargs: dict) -> None:
        for key, value in expected_options.items():
            assert kwargs.get(key) == value

    class FakeFs:
        def glob(self, pattern: str):
            listed.append(pattern)
            rel_pattern = pattern.removeprefix(parsed).lstrip("/")
            return [
                f"{parsed}/{path.relative_to(local_root).as_posix()}"
                for path in local_root.glob(rel_pattern)
            ]

        def info(self, path: str):
            stated.append(path)
            local = to_local(path)
            if not local.exists():
                raise FileNotFoundError(path)
            stat = local.stat()
            rel = local.relative_to(local_root).as_posix()
            return {
                "name": path,
                "type": "directory" if local.is_dir() else "file",
                "size": int(stat.st_size),
                "etag": f"etag-{rel}",
                "version_id": f"version-{rel}",
                "generation": f"generation-{rel}",
                "last_modified": "2026-01-15T12:00:00+00:00",
            }

    def fake_open(uri: str, mode: str = "rb", **kwargs):
        assert_options(kwargs)
        opened.append(uri)
        return to_local(uri).open(mode)

    def fake_url_to_fs(uri: str, **kwargs):
        assert_options(kwargs)
        return FakeFs(), uri.removeprefix("s3://").rstrip("/")

    fake_fsspec = SimpleNamespace(
        open=fake_open,
        core=SimpleNamespace(url_to_fs=fake_url_to_fs),
    )
    monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)
    return {"opened": opened, "listed": listed, "stated": stated}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_parquet(
    path: Path,
    rows: list[dict],
    schema: pa.Schema | None = None,
    *,
    row_group_size: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = (
        pa.Table.from_pylist(rows, schema=schema)
        if schema is not None
        else pa.Table.from_pylist(rows)
    )
    pq.write_table(table, path, compression=None, row_group_size=row_group_size)


def _mp4_box(name: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), name) + payload


def _mp4_full_box(name: bytes, payload: bytes, *, version: int = 0, flags: int = 0) -> bytes:
    return _mp4_box(name, bytes([version]) + flags.to_bytes(3, "big") + payload)


def _fixed_16_16(value: int) -> int:
    return int(value) << 16


def _tiny_mp4(
    samples: list[bytes],
    *,
    keyframes: list[int],
    width: int = 64,
    height: int = 48,
    fps: int = 10,
) -> bytes:
    timescale = 1000
    sample_delta = timescale // fps
    ftyp = _mp4_box(b"ftyp", b"isom\x00\x00\x00\x01isomavc1")
    first_sample_offset = len(ftyp) + 8
    moov = _tiny_mp4_moov(
        samples,
        keyframes=keyframes,
        first_sample_offset=first_sample_offset,
        width=width,
        height=height,
        timescale=timescale,
        sample_delta=sample_delta,
    )
    return ftyp + _mp4_box(b"mdat", b"".join(samples)) + moov


def _tiny_mp4_moov(
    samples: list[bytes],
    *,
    keyframes: list[int],
    first_sample_offset: int,
    width: int,
    height: int,
    timescale: int,
    sample_delta: int,
) -> bytes:
    frame_count = len(samples)
    duration = frame_count * sample_delta
    tkhd = _mp4_full_box(
        b"tkhd",
        b"\x00" * 76 + struct.pack(">II", _fixed_16_16(width), _fixed_16_16(height)),
        flags=7,
    )
    mdhd = _mp4_full_box(
        b"mdhd", struct.pack(">IIIIH2s", 0, 0, timescale, duration, 0, b"\x00\x00")
    )
    hdlr = _mp4_full_box(b"hdlr", b"\x00" * 4 + b"vide" + b"\x00" * 12 + b"VideoHandler\x00")
    stbl = _mp4_box(
        b"stbl",
        _tiny_mp4_stsd(width, height)
        + _mp4_full_box(b"stts", struct.pack(">III", 1, frame_count, sample_delta))
        + _mp4_full_box(
            b"stss",
            struct.pack(">I", len(keyframes))
            + b"".join(struct.pack(">I", index + 1) for index in keyframes),
        )
        + _mp4_full_box(b"stsc", struct.pack(">IIII", 1, 1, frame_count, 1))
        + _mp4_full_box(
            b"stsz",
            struct.pack(">II", 0, frame_count)
            + b"".join(struct.pack(">I", len(sample)) for sample in samples),
        )
        + _mp4_full_box(b"stco", struct.pack(">II", 1, first_sample_offset)),
    )
    return _mp4_box(
        b"moov", _mp4_box(b"trak", tkhd + _mp4_box(b"mdia", mdhd + hdlr + _mp4_box(b"minf", stbl)))
    )


def _tiny_mp4_stsd(width: int, height: int) -> bytes:
    visual_sample_entry = (
        b"\x00" * 6
        + struct.pack(">H", 1)
        + b"\x00" * 16
        + struct.pack(">HH", width, height)
        + struct.pack(">II", 0x00480000, 0x00480000)
        + b"\x00" * 4
        + struct.pack(">H", 1)
        + b"\x00" * 32
        + b"\x00\x18\xff\xff"
        + _mp4_box(b"avcC", b"\x01\x64\x00\x1f")
    )
    entry = struct.pack(">I4s", 8 + len(visual_sample_entry), b"avc1") + visual_sample_entry
    return _mp4_full_box(b"stsd", struct.pack(">I", 1) + entry)


def _write_tiny_mp4(path: Path, samples: list[bytes], *, keyframes: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_tiny_mp4(samples, keyframes=keyframes))


def _lerobot_fixture(root: Path, *, version: str) -> Path:
    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
        "observation.images.front": {"dtype": "video", "shape": [3, 0, 0]},
        "vendor.force": {"dtype": "float32", "shape": [1]},
    }
    info = {
        "codebase_version": version,
        "fps": 10,
        "robot_type": "aloha",
        "features": features,
        "total_episodes": 2,
        "total_frames": 3,
        "total_tasks": 2,
        "chunks_size": 1000,
        "data_path": (
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            if version == "v3.0"
            else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    _write_json(root / "meta/info.json", info)
    _write_json(root / "meta/stats.json", {"observation.state": {"count": 6}})
    _write_jsonl(
        root / "meta/tasks.jsonl",
        [
            {"task_index": 0, "task": "pick cube"},
            {"task_index": 1, "task": "place cube"},
        ],
    )
    episode_rows = [
        {
            "episode_index": 0,
            "episode_id": "lerobot-ep-0",
            "scenario_id": "source-scn-0",
            "tasks": ["pick cube"],
            "length": 2,
            "dataset_from_index": 0,
            "dataset_to_index": 2,
            "split": "train",
        },
        {
            "episode_index": 1,
            "episode_id": "lerobot-ep-1",
            "scenario_id": "source-scn-1",
            "tasks": ["place cube"],
            "length": 1,
            "dataset_from_index": 2,
            "dataset_to_index": 3,
            "split": "train",
        },
    ]
    if version == "v3.0":
        _write_parquet(root / "meta/episodes/chunk-000/file-000.parquet", episode_rows)
    else:
        _write_jsonl(root / "meta/episodes.jsonl", episode_rows)

    rows = [
        {
            "index": 0,
            "episode_index": 0,
            "frame_index": 0,
            "timestamp": 0.0,
            "task_index": 0,
            "task": "pick cube",
            "observation.state": [1.0, 2.0],
            "action": [0.1, 0.2],
            "observation.images.front": "videos/chunk-000/observation.images.front/episode_000000.mp4",
            "vendor.force": 3.5,
        },
        {
            "index": 1,
            "episode_index": 0,
            "frame_index": 1,
            "timestamp": 0.1,
            "task_index": 0,
            "task": "pick cube",
            "observation.state": [1.5, 2.5],
            "action": [0.3, 0.4],
            "observation.images.front": "videos/chunk-000/observation.images.front/episode_000000.mp4",
            "vendor.force": 4.5,
        },
        {
            "index": 2,
            "episode_index": 1,
            "frame_index": 0,
            "timestamp": 0.0,
            "task_index": 1,
            "task": "place cube",
            "observation.state": [5.0, 6.0],
            "action": [0.7, 0.8],
            "observation.images.front": "videos/chunk-000/observation.images.front/episode_000001.mp4",
            "vendor.force": 5.5,
        },
    ]
    if version == "v3.0":
        _write_parquet(root / "data/chunk-000/file-000.parquet", rows)
    else:
        _write_parquet(root / "data/chunk-000/episode_000000.parquet", rows[:2])
        _write_parquet(root / "data/chunk-000/episode_000001.parquet", rows[2:])

    for episode_index in (0, 1):
        path = root / f"videos/chunk-000/observation.images.front/episode_{episode_index:06d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-mp4-" + str(episode_index).encode())
    return root


def _streaming_lerobot_fixture(root: Path) -> Path:
    source = _lerobot_fixture(root, version="v3.0")
    info = json.loads((source / "meta/info.json").read_text())
    info["chunks_size"] = 2
    info["total_episodes"] = 4
    info["total_frames"] = 4
    _write_json(source / "meta/info.json", info)

    episode_rows = [
        {
            "episode_index": index,
            "episode_id": f"lerobot-ep-{index}",
            "scenario_id": f"source-scn-{index}",
            "tasks": ["pick cube" if index % 2 == 0 else "place cube"],
            "length": 1,
            "dataset_from_index": index,
            "dataset_to_index": index + 1,
            "split": "train",
        }
        for index in range(4)
    ]
    _write_parquet(source / "meta/episodes/chunk-000/file-000.parquet", episode_rows)

    rows = [
        {
            "index": index,
            "episode_index": index,
            "frame_index": 0,
            "timestamp": 0.0,
            "task_index": index % 2,
            "task": "pick cube" if index % 2 == 0 else "place cube",
            "observation.state": [float(index), float(index) + 0.5],
            "action": [float(index) / 10.0, float(index) / 10.0 + 0.05],
            "observation.images.front": (
                f"videos/chunk-{index // 2:03d}/observation.images.front/episode_{index:06d}.mp4"
            ),
            "vendor.force": float(index),
        }
        for index in range(4)
    ]
    _write_parquet(source / "data/chunk-000/file-000.parquet", rows[:2], row_group_size=1)
    _write_parquet(source / "data/chunk-001/file-000.parquet", rows[2:], row_group_size=1)
    for episode_index in range(4):
        path = source / (
            f"videos/chunk-{episode_index // 2:03d}/"
            f"observation.images.front/episode_{episode_index:06d}.mp4"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-streaming-mp4-" + str(episode_index).encode())
    return source


def _lerobot_ingest_params(lake: Lake, *, status: str = "completed") -> dict:
    rows = lake.table("transform_runs").to_arrow().to_pylist()
    matches = []
    for row in rows:
        params = json.loads(row["params"] or "{}")
        if (
            row["kind"] == "ingest"
            and row["status"] == status
            and params.get("adapter") == "lerobot"
        ):
            matches.append(params)
    assert len(matches) == 1
    return matches[0]


def _durable_lerobot_checkpoint_rows(lake: Lake) -> list[dict]:
    return sorted(
        lake.table("lerobot_ingest_checkpoints").to_arrow().to_pylist(),
        key=lambda row: (row["checkpoint_index"], row["checkpoint_id"]),
    )


def _seed_lerobot_running_claim(
    lake: Lake,
    source: Path,
    *,
    updated_at: datetime,
    claim_expires_at: datetime | None = None,
    job_id: str | None = None,
    source_id: str | None = None,
    claim_owner: str = "worker-a",
    claim_token: str = "claim-token",
    include_claim: bool = True,
    claim_lease_seconds: float | None = 300.0,
    rows_seen: int = 0,
    observations_written: int = 0,
) -> str:
    dataset = get_adapter("lerobot").dataset(source, include_frames=False)
    digest = dataset.source.digest
    job_id = job_id or f"lerobot-ingest-{digest}"
    source_id = source_id or f"src-{digest}"
    progress = {
        "status": "running",
        "rows_seen": rows_seen,
        "rows_written": {
            "observations": observations_written,
            "episodes": 0,
            "scenarios": 0,
            "videos": 0,
            "video_encodings": 0,
        },
        "rows_skipped_existing": 0,
        "bytes_scanned": 0,
        "last_observation_id": None,
        "last_checkpoint": None,
        "source_identity": {},
    }
    if include_claim:
        claim = {
            "owner": claim_owner,
            "token": claim_token,
            "generation": 1,
            "active": True,
            "heartbeat_interval_seconds": 60.0,
            "heartbeat_count": 1,
            "last_heartbeat_at": updated_at.isoformat(),
            "claim_expires_at": claim_expires_at.isoformat() if claim_expires_at else None,
        }
        if claim_lease_seconds is not None:
            claim["lease_seconds"] = claim_lease_seconds
        progress["claim"] = claim
    row = {
        "checkpoint_id": f"{job_id}:00000000",
        "job_id": job_id,
        "source_id": source_id,
        "run_id": f"run-{digest}",
        "transform_id": f"tfm-{digest}-ingest",
        "source_uri": dataset.source.uri,
        "source_ref": dataset.source.uri,
        "hf_repo_id": None,
        "requested_revision": None,
        "resolved_revision": None,
        "hf_cache_path": None,
        "hf_download_json": "{}",
        "source_identity_json": "{}",
        "status": "running",
        "phase": "claimed",
        "claim_owner": claim_owner,
        "claim_token": claim_token,
        "checkpoint_index": 0,
        "data_file": None,
        "row_group": None,
        "batch_index": None,
        "rows_seen": rows_seen,
        "observations_written": observations_written,
        "episodes_written": 0,
        "scenarios_written": 0,
        "videos_written": 0,
        "video_encodings_written": 0,
        "rows_skipped_existing": 0,
        "bytes_scanned": 0,
        "last_observation_id": None,
        "progress_json": json.dumps(progress, sort_keys=True),
        "error": None,
        "started_at": updated_at - timedelta(minutes=5),
        "updated_at": updated_at,
        "finished_at": None,
        "created_by": claim_owner,
        "created_at": updated_at,
    }
    lake.table("lerobot_ingest_checkpoints").add(
        pa.Table.from_pylist([row], schema=LEROBOT_INGEST_CHECKPOINTS_SCHEMA)
    )
    return job_id


def _add_lerobot_checkpoint_rows(
    lake: Lake,
    *,
    job_id: str,
    source_id: str,
    updated_at: datetime,
    phases: list[tuple[str, str]],
    hf_repo_id: str = "robotics/corpus",
    requested_revision: str | None = "main",
    resolved_revision: str | None = "a" * 40,
    media_inspection: dict | None = None,
    error: str | None = None,
) -> None:
    rows = []
    for index, (phase, status) in enumerate(phases):
        terminal = status in {"abandoned", "completed", "failed", "skipped"}
        progress = {
            "status": status,
            "rows_seen": 100 + index,
            "rows_written": {
                "observations": 90 + index,
                "episodes": 4 if terminal else 0,
                "scenarios": 4 if terminal else 0,
                "videos": 2 if terminal else 0,
                "video_encodings": 2 if terminal else 0,
            },
            "rows_skipped_existing": 3,
            "bytes_scanned": 2048 + index,
            "last_observation_id": f"{job_id}:obs:{index}",
            "last_checkpoint": {
                "data_file": "data/chunk-001/file-000.parquet",
                "row_group": 1,
                "batch_index": index,
                "rows_written": {"observations": 90 + index},
            },
            "source_identity": {
                "kind": "hf-revision",
                "revision": resolved_revision,
                "manifest_fingerprint": f"manifest-{job_id}",
            },
            "media_inspection": dict(
                media_inspection
                or {
                    "status_counts": {"completed": 2},
                    "videos": [
                        {
                            "path": "videos/chunk-000/observation.images.front/episode_000000.mp4",
                            "inspection_status": "completed",
                            "inspection_fingerprint": f"video-{job_id}",
                        }
                    ],
                }
            ),
        }
        if error:
            progress["error"] = error
        ledger = {
            "repo_id": hf_repo_id,
            "repo_type": "dataset",
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "cache_path": f"/tmp/hf-cache/{job_id}",
            "source_ref": f"hf://{hf_repo_id}@{resolved_revision}",
            "manifest_fingerprints": [
                {"path": "data/chunk-001/file-000.parquet", "fingerprint": f"fp-{job_id}"}
            ],
        }
        rows.append(
            {
                "checkpoint_id": f"{job_id}:{index:08d}",
                "job_id": job_id,
                "source_id": source_id,
                "run_id": f"run-{job_id}",
                "transform_id": f"tfm-{job_id}-ingest",
                "source_uri": f"hf://{hf_repo_id}",
                "source_ref": ledger["source_ref"],
                "hf_repo_id": hf_repo_id,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
                "hf_cache_path": ledger["cache_path"],
                "hf_download_json": json.dumps(ledger, sort_keys=True),
                "source_identity_json": json.dumps(progress["source_identity"], sort_keys=True),
                "status": status,
                "phase": phase,
                "claim_owner": "worker-a",
                "claim_token": f"{job_id}:worker-a",
                "checkpoint_index": index,
                "data_file": progress["last_checkpoint"]["data_file"],
                "row_group": progress["last_checkpoint"]["row_group"],
                "batch_index": progress["last_checkpoint"]["batch_index"],
                "rows_seen": progress["rows_seen"],
                "observations_written": progress["rows_written"]["observations"],
                "episodes_written": progress["rows_written"]["episodes"],
                "scenarios_written": progress["rows_written"]["scenarios"],
                "videos_written": progress["rows_written"]["videos"],
                "video_encodings_written": progress["rows_written"]["video_encodings"],
                "rows_skipped_existing": progress["rows_skipped_existing"],
                "bytes_scanned": progress["bytes_scanned"],
                "last_observation_id": progress["last_observation_id"],
                "progress_json": json.dumps(progress, sort_keys=True),
                "error": error if terminal else None,
                "started_at": updated_at - timedelta(minutes=5),
                "updated_at": updated_at,
                "finished_at": updated_at if terminal else None,
                "created_by": "worker-a",
                "created_at": updated_at,
            }
        )
    lake.table("lerobot_ingest_checkpoints").add(
        pa.Table.from_pylist(rows, schema=LEROBOT_INGEST_CHECKPOINTS_SCHEMA)
    )


def _require_media_inspection(payload: dict) -> dict:
    assert "media_inspection" in payload
    media_inspection = payload["media_inspection"]
    assert isinstance(media_inspection, dict)
    return media_inspection


def _require_decoded_frame_conformance(payload: dict) -> dict:
    assert "decoded_frame_conformance" in payload
    conformance = payload["decoded_frame_conformance"]
    assert isinstance(conformance, dict)
    return conformance


def _decoded_frame_conformance_options(*, backend: str, samples: list[dict]) -> dict:
    return {
        "enabled": True,
        "backend": backend,
        "samples": samples,
    }


class _FakeDecodedFrameBackend:
    name = "fake-decoder"
    version = "0.1-test"
    supported_codecs = ("h264",)

    def __init__(self, frames: dict[tuple[str, int, int], bytes]) -> None:
        self.frames = frames
        self.requests: list[dict] = []

    def decode_frame(self, *args, **kwargs) -> dict:
        if args:
            kwargs.setdefault("uri", str(args[0]))
        camera_key = str(kwargs["camera_key"])
        episode_index = int(kwargs["episode_index"])
        frame_index = int(kwargs["frame_index"])
        codec = str(kwargs.get("codec") or "unknown")
        uri = str(kwargs.get("uri") or kwargs.get("path") or "")
        keyframe_map = kwargs.get("keyframe_map")
        self.requests.append(
            {
                "camera_key": camera_key,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "codec": codec,
                "uri": uri,
                "keyframe_map": keyframe_map,
            }
        )
        pixels = self.frames[(camera_key, episode_index, frame_index)]
        return {
            "pixels": pixels,
            "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
            "dtype": "uint8",
            "shape": [1, len(pixels), 1],
        }

    def decode_lerobot_frame(self, *args, **kwargs) -> dict:
        return self.decode_frame(*args, **kwargs)

    def frame_at(self, *args, **kwargs) -> dict:
        return self.decode_frame(*args, **kwargs)


def _install_fake_decoded_frame_backend(monkeypatch, backend: _FakeDecodedFrameBackend) -> None:
    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module
    import lancedb_robotics.ingest as ingest_module

    def resolve_backend(name: str = "auto"):
        assert name in {"auto", backend.name}
        return backend

    for module in (lerobot_module, ingest_module):
        monkeypatch.setattr(
            module, "_resolve_decoded_frame_decoder_backend", resolve_backend, raising=False
        )
        monkeypatch.setattr(
            module, "_resolve_lerobot_decoded_frame_decoder", resolve_backend, raising=False
        )


def _install_missing_decoded_frame_backend(monkeypatch) -> None:
    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module
    import lancedb_robotics.ingest as ingest_module

    def resolve_backend(name: str = "auto"):
        assert name == "missing-decoder"
        return None

    def backend_status(name: str = "auto") -> dict:
        assert name == "missing-decoder"
        return {
            "requested": name,
            "available": False,
            "status": "unsupported",
            "reason": "decoder backend missing; install lancedb-robotics[media]",
        }

    for module in (lerobot_module, ingest_module):
        monkeypatch.setattr(
            module, "_resolve_decoded_frame_decoder_backend", resolve_backend, raising=False
        )
        monkeypatch.setattr(
            module, "_resolve_lerobot_decoded_frame_decoder", resolve_backend, raising=False
        )
        monkeypatch.setattr(
            module, "_decoded_frame_decoder_backend_status", backend_status, raising=False
        )


def test_lerobot_adapter_registers_and_declares_capabilities():
    adapter = get_adapter("lerobot")

    assert "lerobot" in [info.name for info in list_adapters()]
    assert adapter.info.name == "lerobot"
    assert adapter.info.format == "lerobot"
    assert {"inspect", "ingest"} <= set(adapter.info.capabilities)
    assert callable(adapter.inspect)
    assert callable(adapter.ingest)


def test_inspect_lerobot_reports_version_frames_tasks_and_cameras(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot", version="v3.0")

    report = get_adapter("lerobot").inspect(source)

    assert report["codebase_version"] == "v3.0"
    assert report["episode_count"] == 2
    assert report["frame_count"] == 3
    assert report["fps"] == 10
    assert report["camera_keys"] == ["front"]
    assert [task["task"] for task in report["tasks"]] == ["pick cube", "place cube"]
    assert len(report["video_files"]) == 2
    assert report["native_loader"]["install"] == "lancedb-robotics[lerobot]"


def test_lerobot_video_metadata_keyframe_map_and_seek_conformance(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot-video", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    report = get_adapter("lerobot").inspect(source)

    videos = sorted(report["video_files"], key=lambda row: row["episode_index"])
    assert report["diagnostics"] == []
    assert videos[0]["codec"] == "h264"
    assert videos[0]["codec_tag"] == "avc1"
    assert videos[0]["resolution"] == "64x48"
    assert videos[0]["fps"] == pytest.approx(10.0)
    assert videos[0]["frame_count"] == 2
    assert videos[0]["gop_size"] == 2
    keyframes = videos[0]["keyframe_map"]
    assert len(keyframes) == 1
    assert keyframes[0]["first_frame_index"] == 0
    assert keyframes[0]["last_frame_index"] == 1
    assert [frame["frame_index"] for frame in keyframes[0]["frames"]] == [0, 1]
    assert read_mp4_frame_sample(first_video, keyframes, 1) == b"front-frame-1"

    lake = Lake.init(tmp_path / "lake")
    report = ingest_lerobot(
        lake, source, compact=False, prune_versions=False, index_predicates=False
    )

    assert report.rows_added["observations"] == 3
    encodings = sorted(
        lake.table("video_encodings").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    assert encodings[0]["codec"] == "h264"
    assert encodings[0]["resolution"] == "64x48"
    assert encodings[0]["fps"] == pytest.approx(10.0)
    assert encodings[0]["frame_count"] == 2
    assert encodings[0]["gop_size"] == 2
    assert encodings[0]["keyframe_map_ref"].startswith("sha256:")
    assert encodings[0]["data"] is None
    assert encodings[0]["source_size_bytes"] == encodings[0]["encoded_size_bytes"]
    assert (
        read_mp4_frame_sample(
            first_video,
            encodings[0]["keyframe_map_json"],
            1,
        )
        == b"front-frame-1"
    )
    decoded = lake.video.seek(encodings[0]["episode_id"], 1, camera_key="front")
    assert decoded.episode_index == 0
    assert decoded.frame_index == 1
    assert decoded.camera_key == "front"
    assert decoded.frame == b"front-frame-1"
    assert _lerobot_ingest_params(lake)["video_diagnostics"] == []


def test_ingest_lerobot_offloads_large_keyframe_maps_and_seek_resolves(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot-keyframe-offload", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    samples = [b"front-frame-0", b"front-frame-1"]
    _write_tiny_mp4(first_video, samples, keyframes=[0])
    _write_tiny_mp4(second_video, samples, keyframes=[0])

    lake_path = tmp_path / "lake"
    lake = Lake.init(lake_path)
    report = ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        keyframe_map_inline_threshold_bytes=1,
    )

    assert report.rows_added["keyframe_map_artifacts"] == 1
    assert report.rows_added["keyframe_map_artifact_referrers"] == 2
    encodings = sorted(
        lake.table("video_encodings").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    assert len({row["keyframe_map_ref"] for row in encodings}) == 1
    assert all(row["keyframe_map_ref"].startswith("sha256:") for row in encodings)
    assert all(row["keyframe_map_json"] is None for row in encodings)

    artifacts = lake.table("keyframe_map_artifacts").to_arrow().to_pylist()
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["artifact_id"].startswith("kfmap-")
    assert artifact["keyframe_map_ref"] == encodings[0]["keyframe_map_ref"]
    assert artifact["json_size_bytes"] > 1
    assert artifact["frame_count"] == 2
    assert read_mp4_frame_sample(first_video, artifact["keyframe_map_json"], 1) == b"front-frame-1"

    referrers = sorted(
        lake.table("keyframe_map_artifact_referrers").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    assert len(referrers) == 2
    assert len({row["referrer_id"] for row in referrers}) == 2
    assert all(row["artifact_id"] == artifact["artifact_id"] for row in referrers)
    assert [row["encoding_id"] for row in referrers] == [row["encoding_id"] for row in encodings]
    assert [row["video_id"] for row in referrers] == [row["video_id"] for row in encodings]
    assert all(row["referrer_table"] == "video_encodings" for row in referrers)
    assert all(row["referrer_table_version"] >= 1 for row in referrers)
    assert [row["camera_key"] for row in referrers] == ["front", "front"]

    api_referrers = lake.video.keyframe_map_referrers(artifact["artifact_id"])
    assert [row["referrer_id"] for row in api_referrers] == [
        row["referrer_id"] for row in referrers
    ]
    cli = runner.invoke(
        app,
        [
            "video",
            "keyframe-map-referrers",
            "--lake",
            str(lake_path),
            "--artifact",
            artifact["artifact_id"],
            "--format",
            "json",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert [row["referrer_id"] for row in json.loads(cli.output)["referrers"]] == [
        row["referrer_id"] for row in referrers
    ]

    decoded = lake.video.seek(encodings[0]["episode_id"], 1, camera_key="front")
    assert decoded.frame == b"front-frame-1"
    assert decoded.decoder == "mp4-sample"

    params = _lerobot_ingest_params(lake)
    assert params["keyframe_map_artifacts"]["offloaded_video_count"] == 2
    assert params["keyframe_map_artifacts"]["artifact_count"] == 1
    media = _require_media_inspection(params)
    assert media["keyframe_map_offloaded_video_count"] == 2
    assert all("keyframe_map" not in row for row in media["videos"])
    assert all(
        row["keyframe_map_artifact_id"] == artifact["artifact_id"] for row in media["videos"]
    )
    for checkpoint in _durable_lerobot_checkpoint_rows(lake):
        progress = json.loads(checkpoint["progress_json"])
        media_progress = _require_media_inspection(progress)
        assert all(not row.get("keyframe_map") for row in media_progress["videos"])

    second = ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        keyframe_map_inline_threshold_bytes=1,
    )
    assert second.already_ingested is True
    assert lake.table("keyframe_map_artifacts").count_rows() == 1
    assert lake.table("keyframe_map_artifact_referrers").count_rows() == 2


def test_lerobot_object_store_inspect_preserves_uris_and_etag_fingerprints(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-s3", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])
    local_report = get_adapter("lerobot").inspect(source)

    remote_root = "s3://robotics-raw/lerobot-s3"
    calls = _install_fake_lerobot_object_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        expected_options={"anon": True},
    )

    report = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        auth_ref="raw-bucket",
    )

    assert report["path"] == remote_root
    assert report["source_identity"]["kind"] == "object-store-metadata"
    assert report["frame_count"] == 3
    assert report["data_files"] == ["data/chunk-000/file-000.parquet"]
    videos = sorted(report["video_files"], key=lambda row: row["episode_index"])
    local_videos = sorted(local_report["video_files"], key=lambda row: row["episode_index"])
    assert [video["keyframe_map"] for video in videos] == [
        video["keyframe_map"] for video in local_videos
    ]
    assert (
        videos[0]["uri"]
        == f"{remote_root}/videos/chunk-000/observation.images.front/episode_000000.mp4"
    )
    assert videos[0]["object_metadata"]["etag"].startswith("etag-videos/")
    assert videos[0]["object_metadata"]["version_id"].startswith("version-videos/")
    assert videos[0]["object_metadata"]["generation"].startswith("generation-videos/")
    expected_fingerprint_payload = {
        "uri": videos[0]["uri"],
        "size": videos[0]["size"],
        "metadata": videos[0]["object_metadata"],
        "expected_frame_count": videos[0]["expected_frame_count"],
    }
    assert (
        videos[0]["inspection_fingerprint"]
        == "sha256:"
        + hashlib.sha256(
            json.dumps(expected_fingerprint_payload, sort_keys=True).encode()
        ).hexdigest()
    )
    assert any(uri.startswith(remote_root + "/videos/") for uri in calls["opened"])
    assert any(path.endswith("meta/info.json") for path in calls["stated"])


def test_ingest_lerobot_object_store_root_preserves_video_uris_and_keyframes(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-s3-ingest", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    local_lake = Lake.init(tmp_path / "local-lake")
    local = ingest_lerobot(
        local_lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    remote_root = "s3://robotics-raw/lerobot-s3-ingest"
    _install_fake_lerobot_object_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        expected_options={"anon": True},
    )
    remote_lake = Lake.init(tmp_path / "remote-lake")
    remote = ingest_lerobot(
        remote_lake,
        remote_root,
        storage_options={"anon": True},
        auth_ref="raw-bucket",
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert remote.rows_added["observations"] == local.rows_added["observations"] == 3
    assert remote.rows_added["videos"] == local.rows_added["videos"] == 2
    assert remote.rows_added["video_encodings"] == local.rows_added["video_encodings"] == 2
    sources = remote_lake.table("integration_sources").to_arrow().to_pylist()
    assert sources[0]["uri"] == remote_root
    assert sources[0]["auth_ref"] == "raw-bucket"
    videos = sorted(
        remote_lake.table("videos").to_arrow().to_pylist(),
        key=lambda row: row["episode_index"],
    )
    assert all(row["uri"].startswith(remote_root + "/videos/") for row in videos)
    assert all(row["raw_uri"] == row["uri"] for row in videos)
    local_keyframes = [
        json.loads(row["keyframe_map_json"])
        for row in sorted(
            local_lake.table("video_encodings").to_arrow().to_pylist(),
            key=lambda item: item["episode_index"],
        )
    ]
    remote_keyframes = [
        json.loads(row["keyframe_map_json"])
        for row in sorted(
            remote_lake.table("video_encodings").to_arrow().to_pylist(),
            key=lambda item: item["episode_index"],
        )
    ]
    assert remote_keyframes == local_keyframes
    params = _lerobot_ingest_params(remote_lake)
    assert params["source_identity"]["kind"] == "object-store-metadata"
    assert params["video_files"][0]["uri"].startswith(remote_root + "/videos/")


def test_lake_video_conform_source_records_decoded_source_report(tmp_path, monkeypatch):
    import lancedb_robotics.video as video_module

    source = _lerobot_fixture(tmp_path / "lerobot-video-conformance", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])
    expected_pixels = b"\x01\x02\x03\x04"
    requests: list[dict] = []

    def fake_source_decoder(**kwargs):
        requests.append(dict(kwargs))
        return {
            "frame": expected_pixels,
            "backend": "fake-source-decoder",
            "version": "0.2-test",
            "seek_strategy": "nearest-keyframe",
            "frames_decoded": 2,
        }

    monkeypatch.setitem(
        video_module._SOURCE_MP4_DECODER_BACKENDS,
        "fake-source-decoder",
        fake_source_decoder,
    )
    lake = Lake.init(tmp_path / "lake")
    ingest_lerobot(lake, source, compact=False, prune_versions=False, index_predicates=False)

    report = lake.video.conform_source(
        [
            {
                "camera_key": "front",
                "episode_index": 0,
                "frame_index": 1,
                "expected_sha256": hashlib.sha256(expected_pixels).hexdigest(),
            }
        ],
        decoder="fake-source-decoder",
    )

    assert report.status == "passed"
    assert report.status_counts == {"passed": 1}
    assert report.codec_counts == {"h264": 1}
    assert report.backend_versions == {"fake-source-decoder": "0.2-test"}
    assert report.results[0]["frame_index"] == 1
    assert report.results[0]["seek_strategy"] == "nearest-keyframe"
    assert report.results[0]["frames_decoded"] == 2
    assert requests[0]["entry"]["keyframe_frame_index"] == 0
    assert requests[0]["frame_entry"]["frame_index"] == 1
    rows = lake.table("transform_runs").to_arrow().to_pylist()
    conformance_rows = [row for row in rows if row["kind"] == "video-conformance"]
    assert len(conformance_rows) == 1
    params = json.loads(conformance_rows[0]["params"])
    assert params["status"] == "passed"
    assert params["decoder"] == "fake-source-decoder"


def test_video_conform_source_cli_emits_json_report(tmp_path, monkeypatch):
    import lancedb_robotics.video as video_module

    source = _lerobot_fixture(tmp_path / "lerobot-cli-conform-source", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])
    expected_pixels = b"\x09\x08\x07\x06"

    def fake_source_decoder(**kwargs):
        return {
            "frame": expected_pixels,
            "backend": "fake-source-decoder",
            "version": "0.3-cli",
            "seek_strategy": "nearest-keyframe",
            "frames_decoded": 2,
            "seek_frame_index": int(kwargs["entry"]["keyframe_frame_index"]),
        }

    monkeypatch.setitem(
        video_module._SOURCE_MP4_DECODER_BACKENDS,
        "fake-source-decoder",
        fake_source_decoder,
    )
    lake = Lake.init(tmp_path / "lake")
    ingest_lerobot(lake, source, compact=False, prune_versions=False, index_predicates=False)
    samples = tmp_path / "samples.json"
    _write_json(
        samples,
        {
            "samples": [
                {
                    "camera_key": "front",
                    "episode_index": 0,
                    "frame_index": 1,
                    "expected_sha256": hashlib.sha256(expected_pixels).hexdigest(),
                }
            ]
        },
    )

    result = runner.invoke(
        app,
        [
            "video",
            "conform-source",
            "--lake",
            str(lake.uri),
            "--samples",
            str(samples),
            "--decoder",
            "fake-source-decoder",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "passed"
    assert payload["decoder"] == "fake-source-decoder"
    assert payload["backend_versions"] == {"fake-source-decoder": "0.3-cli"}
    assert payload["status_counts"] == {"passed": 1}
    assert payload["results"][0]["seek_strategy"] == "nearest-keyframe"
    assert payload["results"][0]["seek_frame_index"] == 0


def test_video_conform_source_cli_jsonl_fails_on_mismatch(tmp_path, monkeypatch):
    import lancedb_robotics.video as video_module

    source = _lerobot_fixture(tmp_path / "lerobot-cli-conform-source-mismatch", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    def fake_source_decoder(**kwargs):
        return {
            "frame": b"actual-cli-pixels",
            "backend": "fake-source-decoder",
            "version": "0.3-cli",
            "seek_strategy": "nearest-keyframe",
            "frames_decoded": 1,
        }

    monkeypatch.setitem(
        video_module._SOURCE_MP4_DECODER_BACKENDS,
        "fake-source-decoder",
        fake_source_decoder,
    )
    lake = Lake.init(tmp_path / "lake")
    ingest_lerobot(lake, source, compact=False, prune_versions=False, index_predicates=False)
    samples = tmp_path / "samples.jsonl"
    _write_jsonl(
        samples,
        [
            {
                "camera_key": "front",
                "episode_index": 0,
                "frame_index": 1,
                "expected_sha256": hashlib.sha256(b"expected-cli-pixels").hexdigest(),
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "video",
            "conform-source",
            "--lake",
            str(lake.uri),
            "--samples",
            str(samples),
            "--decoder",
            "fake-source-decoder",
            "--format",
            "jsonl",
            "--fail-on-mismatch",
        ],
    )

    assert result.exit_code == 1
    rows = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["reason"] == "decoded-frame-mismatch"
    assert rows[0]["decoder_backend"] == "fake-source-decoder"


def test_video_conform_source_cli_missing_decoder_reports_install_hint(tmp_path, monkeypatch):
    import lancedb_robotics.video as video_module

    source = _lerobot_fixture(
        tmp_path / "lerobot-cli-conform-source-missing-decoder", version="v3.0"
    )
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])
    lake = Lake.init(tmp_path / "lake")
    ingest_lerobot(lake, source, compact=False, prune_versions=False, index_predicates=False)
    samples = tmp_path / "samples.json"
    _write_json(
        samples,
        {
            "camera_key": "front",
            "episode_index": 0,
            "frame_index": 1,
            "expected_sha256": "0" * 64,
        },
    )
    original_find_spec = video_module.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "av":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(video_module.importlib.util, "find_spec", fake_find_spec)

    result = runner.invoke(
        app,
        [
            "video",
            "conform-source",
            "--lake",
            str(lake.uri),
            "--samples",
            str(samples),
            "--decoder",
            "pyav",
            "--format",
            "text",
        ],
    )

    assert result.exit_code == 0
    assert "source conformance: status=skipped" in result.output
    assert "detail=decoder-unavailable" in result.output
    assert "install hint:" in result.output
    assert "video-decode" in result.output


def test_lerobot_decoded_frame_conformance_skips_when_decoder_backend_missing(
    tmp_path, monkeypatch
):
    source = _lerobot_fixture(tmp_path / "lerobot-decoded-missing", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])
    _install_missing_decoded_frame_backend(monkeypatch)
    lake = Lake.init(tmp_path / "lake")

    report = ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        decoded_frame_conformance=_decoded_frame_conformance_options(
            backend="missing-decoder",
            samples=[
                {
                    "camera_key": "front",
                    "episode_index": 0,
                    "frame_index": 1,
                    "expected_pixel_sha256": hashlib.sha256(b"unused").hexdigest(),
                }
            ],
        ),
    )

    assert report.rows_added["observations"] == 3
    assert lake.table("observations").count_rows() == 3
    conformance = _require_decoded_frame_conformance(_lerobot_ingest_params(lake))
    assert conformance["enabled"] is True
    assert conformance["status"] in {"skipped", "unsupported"}
    assert conformance["backend"]["requested"] == "missing-decoder"
    assert conformance["backend"]["available"] is False
    assert conformance["frames_checked"] == 0
    assert conformance["failures"] == []
    assert "install" in conformance["reason"]


def test_lerobot_decoded_frame_conformance_verifies_episode_frame_with_fake_backend(
    tmp_path, monkeypatch
):
    source = _lerobot_fixture(tmp_path / "lerobot-decoded-fake", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])
    expected_pixels = b"\x10\x20\x30"
    backend = _FakeDecodedFrameBackend({("front", 0, 1): expected_pixels})
    _install_fake_decoded_frame_backend(monkeypatch, backend)
    lake = Lake.init(tmp_path / "lake")

    ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        decoded_frame_conformance=_decoded_frame_conformance_options(
            backend="fake-decoder",
            samples=[
                {
                    "camera_key": "front",
                    "episode_index": 0,
                    "frame_index": 1,
                    "expected_pixel_sha256": hashlib.sha256(expected_pixels).hexdigest(),
                }
            ],
        ),
    )

    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request["camera_key"] == "front"
    assert request["episode_index"] == 0
    assert request["frame_index"] == 1
    assert request["codec"] == "h264"
    assert request["uri"].endswith("episode_000000.mp4")
    assert request["keyframe_map"]
    conformance = _require_decoded_frame_conformance(_lerobot_ingest_params(lake))
    assert conformance["status"] == "passed"
    assert conformance["frames_checked"] == 1
    assert conformance["backend"]["name"] == "fake-decoder"
    assert conformance["backend"]["version"] == "0.1-test"
    assert conformance["failures"] == []
    assert conformance["codec_coverage"]["h264"]["supported"] is True
    assert conformance["codec_coverage"]["h264"]["frames_checked"] == 1


def test_lerobot_decoded_frame_conformance_report_records_backend_codec_failures(
    tmp_path, monkeypatch
):
    source = _lerobot_fixture(tmp_path / "lerobot-decoded-failure-report", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])
    backend = _FakeDecodedFrameBackend({("front", 1, 0): b"actual-decoded-pixels"})
    _install_fake_decoded_frame_backend(monkeypatch, backend)
    lake = Lake.init(tmp_path / "lake")

    report = ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        decoded_frame_conformance=_decoded_frame_conformance_options(
            backend="fake-decoder",
            samples=[
                {
                    "camera_key": "front",
                    "episode_index": 1,
                    "frame_index": 0,
                    "expected_pixel_sha256": hashlib.sha256(b"expected-pixels").hexdigest(),
                }
            ],
        ),
    )

    assert report.rows_added["observations"] == 3
    conformance = _require_decoded_frame_conformance(_lerobot_ingest_params(lake))
    assert conformance["status"] == "failed"
    assert conformance["backend"] == {
        "name": "fake-decoder",
        "version": "0.1-test",
        "requested": "fake-decoder",
        "available": True,
    }
    assert conformance["codec_coverage"]["h264"]["frames_checked"] == 1
    assert conformance["codec_coverage"]["h264"]["failures"] == 1
    assert len(conformance["failures"]) == 1
    failure = conformance["failures"][0]
    assert failure["camera_key"] == "front"
    assert failure["episode_index"] == 1
    assert failure["frame_index"] == 0
    assert failure["codec"] == "h264"
    assert failure["backend"] == "fake-decoder"
    assert failure["reason"] == "pixel-sha256-mismatch"


def test_lerobot_base_import_does_not_import_pyav_opencv_or_ffmpeg():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {"av", "cv2", "ffmpeg"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise AssertionError(f"unexpected eager media decoder import: {fullname}")
                return None

        sys.meta_path.insert(0, Blocker())

        import lancedb_robotics.adapters.lerobot_adapter
        import lancedb_robotics.ingest
        from lancedb_robotics.adapters import get_adapter

        assert get_adapter("lerobot").availability()["install"] == "lancedb-robotics[lerobot]"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_generated_multi_gop_mp4_fixture_maps_keyframe_ranges(tmp_path):
    path = tmp_path / "multi-gop.mp4"
    samples = [f"frame-{index}".encode() for index in range(5)]
    _write_tiny_mp4(path, samples, keyframes=[0, 2, 4])

    metadata = inspect_mp4_video(path)

    assert metadata.codec == "h264"
    assert metadata.frame_count == 5
    assert metadata.gop_size == 2
    assert [entry["keyframe_frame_index"] for entry in metadata.keyframe_map] == [0, 2, 4]
    assert [
        (entry["first_frame_index"], entry["last_frame_index"]) for entry in metadata.keyframe_map
    ] == [
        (0, 1),
        (2, 3),
        (4, 4),
    ]
    assert read_mp4_frame_sample(path, metadata.keyframe_map, 3) == b"frame-3"


def test_lerobot_missing_and_corrupt_video_diagnostics_preserve_frames(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot-video-diagnostics", version="v3.0")
    missing = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    missing.unlink()

    report = get_adapter("lerobot").inspect(source)

    codes = sorted(diagnostic["code"] for diagnostic in report["diagnostics"])
    assert codes == ["corrupt-video", "missing-video"]
    assert report["frame_count"] == 3

    lake = Lake.init(tmp_path / "lake")
    ingest_report = ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert ingest_report.rows_added["observations"] == 3
    assert lake.table("observations").count_rows() == 3
    encodings = lake.table("video_encodings").to_arrow().to_pylist()
    assert len(encodings) == 2
    assert all(row["data"] is None for row in encodings)
    assert all(json.loads(row["keyframe_map_json"]) == [] for row in encodings)
    params = _lerobot_ingest_params(lake)
    assert sorted(diagnostic["code"] for diagnostic in params["video_diagnostics"]) == [
        "corrupt-video",
        "missing-video",
    ]


def test_lerobot_mp4_media_inspector_reports_bounded_local_reads(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot-media-inspection", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"a" * 64_000, b"b" * 64_000], keyframes=[0])
    _write_tiny_mp4(second_video, [b"c" * 64_000], keyframes=[0])

    report = get_adapter("lerobot").inspect(source)

    media = _require_media_inspection(report)
    assert media["mode"] == "bounded-metadata-inspection"
    assert media["video_count"] == 2
    assert media["status_counts"] == {"completed": 2}
    assert media["codec_counts"] == {"h264": 2}
    assert media["diagnostic_counts"] == {}
    videos = sorted(media["videos"], key=lambda row: row["episode_index"])
    assert [video["path"] for video in videos] == [
        "videos/chunk-000/observation.images.front/episode_000000.mp4",
        "videos/chunk-000/observation.images.front/episode_000001.mp4",
    ]
    assert [video["frame_count"] for video in videos] == [2, 1]
    assert all(video["inspection_status"] == "completed" for video in videos)
    assert all(video["inspection_reused"] is False for video in videos)
    assert all(video["diagnostics"] == [] for video in videos)
    for video in videos:
        assert 0 < video["inspection_bytes_read"] < video["size"]
        assert video["inspection_bytes_read"] < 8192
        assert video["inspection_duration_ms"] >= 0


def test_lerobot_media_inspection_retries_transient_read_failures(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-media-retry-success", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    real_inspect = lerobot_module.inspect_mp4_video
    attempts: dict[str, int] = {}

    def flaky_inspect(path, *, storage_options=None, auth_ref=None):
        key = str(path)
        attempts[key] = attempts.get(key, 0) + 1
        if attempts[key] == 1:
            raise Mp4MetadataError("transient ranged read failed")
        return real_inspect(path, storage_options=storage_options, auth_ref=auth_ref)

    monkeypatch.setattr(lerobot_module, "inspect_mp4_video", flaky_inspect)

    report = get_adapter("lerobot").inspect(source, media_inspection_retries=1)

    media = _require_media_inspection(report)
    assert media["status_counts"] == {"completed": 2}
    assert media["diagnostic_counts"] == {}
    assert media["retry_policy"] == "fixed"
    assert media["retry_backoff_seconds"] == 0.0
    assert media["retry_class_counts"] == {"retryable-transient": 2}
    assert media["total_attempts"] == 4
    assert media["total_retries"] == 2
    videos = sorted(media["videos"], key=lambda row: row["episode_index"])
    assert all(video["inspection_attempts"] == 2 for video in videos)
    assert all(video["inspection_retries"] == 1 for video in videos)
    assert all(video["inspection_error"] is None for video in videos)
    assert all(
        video["inspection_attempt_errors"][0]["error_class"] == "Mp4MetadataError"
        for video in videos
    )
    assert all(
        video["inspection_attempt_errors"][0]["retry_class"] == "retryable-transient"
        for video in videos
    )
    assert all(video["inspection_attempt_errors"][0]["retry_policy"] == "fixed" for video in videos)
    assert all(video["inspection_attempt_errors"][0]["retryable"] is True for video in videos)


def test_lerobot_media_inspection_retry_exhaustion_records_attempts(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-media-retry-exhausted", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    def failing_inspect(path, *, storage_options=None, auth_ref=None):
        raise Mp4MetadataError("object-store read exhausted")

    monkeypatch.setattr(lerobot_module, "inspect_mp4_video", failing_inspect)

    report = get_adapter("lerobot").inspect(source, media_inspection_retries=2)

    media = _require_media_inspection(report)
    assert media["status_counts"] == {"failed": 2}
    assert media["diagnostic_counts"] == {"retryable-transient": 2}
    assert media["retry_class_counts"] == {"retryable-transient": 6}
    assert media["total_attempts"] == 6
    assert media["total_retries"] == 4
    videos = sorted(media["videos"], key=lambda row: row["episode_index"])
    assert all(video["inspection_attempts"] == 3 for video in videos)
    assert all(video["inspection_retries"] == 2 for video in videos)
    assert all(video["inspection_error_class"] == "Mp4MetadataError" for video in videos)
    assert all(video["inspection_retry_class"] == "retryable-transient" for video in videos)
    assert all(video["inspection_retryable"] is True for video in videos)
    assert all(len(video["inspection_attempt_errors"]) == 3 for video in videos)
    assert all(video["diagnostics"][0]["attempts"] == 3 for video in videos)
    assert all(video["diagnostics"][0]["retry_class"] == "retryable-transient" for video in videos)


def test_lerobot_media_inspection_does_not_retry_corrupt_media(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-media-corrupt-no-retry", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    def corrupt_inspect(path, *, storage_options=None, auth_ref=None):
        raise Mp4MetadataError("missing ftyp box")

    monkeypatch.setattr(lerobot_module, "inspect_mp4_video", corrupt_inspect)

    report = get_adapter("lerobot").inspect(source, media_inspection_retries=3)

    media = _require_media_inspection(report)
    assert media["status_counts"] == {"failed": 2}
    assert media["diagnostic_counts"] == {"corrupt-video": 2}
    assert media["retry_class_counts"] == {"corrupt-media": 2}
    assert media["total_attempts"] == 2
    assert media["total_retries"] == 0
    videos = sorted(media["videos"], key=lambda row: row["episode_index"])
    assert all(video["inspection_attempts"] == 1 for video in videos)
    assert all(video["inspection_retry_class"] == "corrupt-media" for video in videos)
    assert all(video["inspection_retryable"] is False for video in videos)
    assert all(
        video["inspection_attempt_errors"][0]["retry_class"] == "corrupt-media" for video in videos
    )


def test_lerobot_media_inspection_does_not_retry_auth_or_missing_objects(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-media-auth-missing", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    def classified_inspect(path, *, storage_options=None, auth_ref=None):
        if str(path).endswith("episode_000000.mp4"):
            raise StorageConfigError("Forbidden credentials for object")
        raise FileNotFoundError("NoSuchKey: missing object")

    monkeypatch.setattr(lerobot_module, "inspect_mp4_video", classified_inspect)

    report = get_adapter("lerobot").inspect(source, media_inspection_retries=3)

    media = _require_media_inspection(report)
    assert media["status_counts"] == {"failed": 2}
    assert media["diagnostic_counts"] == {"auth-config": 1, "missing-object": 1}
    assert media["retry_class_counts"] == {"auth-config": 1, "missing-object": 1}
    assert media["total_attempts"] == 2
    assert media["total_retries"] == 0
    videos = sorted(media["videos"], key=lambda row: row["episode_index"])
    assert videos[0]["inspection_retry_class"] == "auth-config"
    assert videos[1]["inspection_retry_class"] == "missing-object"
    assert all(video["inspection_retryable"] is False for video in videos)


def test_lerobot_media_inspection_retries_throttle_with_exponential_jitter(
    tmp_path,
    monkeypatch,
):
    source = _lerobot_fixture(tmp_path / "lerobot-media-throttle-jitter", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    real_inspect = lerobot_module.inspect_mp4_video
    attempts: dict[str, int] = {}
    sleeps: list[float] = []

    def throttled_inspect(path, *, storage_options=None, auth_ref=None):
        key = str(path)
        attempts[key] = attempts.get(key, 0) + 1
        if key.endswith("episode_000000.mp4") and attempts[key] <= 2:
            raise OSError("SlowDown: TooManyRequests 429")
        return real_inspect(path, storage_options=storage_options, auth_ref=auth_ref)

    monkeypatch.setattr(lerobot_module, "inspect_mp4_video", throttled_inspect)
    monkeypatch.setattr(lerobot_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(lerobot_module.random, "uniform", lambda low, high: high / 2)

    report = get_adapter("lerobot").inspect(
        source,
        media_inspection_workers=1,
        media_inspection_retries=2,
        media_inspection_retry_backoff_seconds=0.5,
        media_inspection_retry_policy="exponential-jitter",
    )

    media = _require_media_inspection(report)
    assert media["status_counts"] == {"completed": 2}
    assert media["retry_policy"] == "exponential-jitter"
    assert media["retry_backoff_seconds"] == 0.5
    assert media["retry_class_counts"] == {"throttle-backoff": 2}
    assert sleeps == pytest.approx([1.25, 2.25])
    videos = sorted(media["videos"], key=lambda row: row["episode_index"])
    assert videos[0]["inspection_attempts"] == 3
    assert videos[0]["inspection_retry_class"] == "throttle-backoff"
    assert [
        error["retry_delay_seconds"] for error in videos[0]["inspection_attempt_errors"]
    ] == pytest.approx([1.25, 2.25])
    assert videos[1]["inspection_attempts"] == 1


def test_ingest_lerobot_media_inspection_timeout_does_not_abort_frames(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-media-timeout", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    real_inspect = lerobot_module.inspect_mp4_video

    def slow_inspect(path, *, storage_options=None, auth_ref=None):
        time.sleep(0.2)
        return real_inspect(path, storage_options=storage_options, auth_ref=auth_ref)

    monkeypatch.setattr(lerobot_module, "inspect_mp4_video", slow_inspect)
    lake = Lake.init(tmp_path / "lake")

    report = ingest_lerobot(
        lake,
        source,
        media_inspection_timeout_seconds=0.01,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert report.rows_added["observations"] == 3
    params = _lerobot_ingest_params(lake)
    media = _require_media_inspection(params)
    assert media["status_counts"] == {"timeout": 2}
    assert media["diagnostic_counts"] == {"timeout-video": 2}
    assert media["total_timeouts"] == 2
    assert media["timeout_seconds"] == pytest.approx(0.01)
    assert media["retry_count"] == 0
    videos = sorted(media["videos"], key=lambda row: row["episode_index"])
    assert all(video["inspection_status"] == "timeout" for video in videos)
    assert all(video["inspection_error_class"] == "TimeoutError" for video in videos)
    assert all(video["diagnostics"][0]["code"] == "timeout-video" for video in videos)
    job = get_lerobot_ingest_job(lake, report.ingest_job_id)
    checkpoint_media = _require_media_inspection(job["history"][-1]["progress"])
    assert checkpoint_media["total_timeouts"] == 2


def test_lerobot_process_media_inspection_completes_tiny_mp4s(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot-media-process-success", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    report = get_adapter("lerobot").inspect(
        source,
        media_inspection_workers=1,
        media_inspection_timeout_seconds=5.0,
        media_inspection_execution_mode="process",
    )

    media = _require_media_inspection(report)
    assert media["execution_mode"] == "process"
    assert media["max_workers"] == 1
    assert media["killed_worker_count"] == 0
    assert media["status_counts"] == {"completed": 2}
    assert all(video["inspection_execution"] == "process" for video in media["videos"])
    assert all(video["inspection_worker_killed"] is False for video in media["videos"])


def test_ingest_lerobot_process_media_inspection_timeout_kills_workers(tmp_path, monkeypatch):
    from conftest import require_start_method

    require_start_method("fork")
    monkeypatch.setenv("LANCEDB_ROBOTICS_LEROBOT_MEDIA_INSPECTION_START_METHOD", "fork")
    source = _lerobot_fixture(tmp_path / "lerobot-media-process-timeout", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    def wedged_inspect(path, *, storage_options=None, auth_ref=None):
        time.sleep(30)
        raise AssertionError("process-isolated worker should be killed before returning")

    monkeypatch.setattr(lerobot_module, "inspect_mp4_video", wedged_inspect)
    lake = Lake.init(tmp_path / "lake")
    started = time.perf_counter()

    report = ingest_lerobot(
        lake,
        source,
        media_inspection_timeout_seconds=0.05,
        media_inspection_execution_mode="process",
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert time.perf_counter() - started < 2.0
    assert report.rows_added["observations"] == 3
    params = _lerobot_ingest_params(lake)
    media = _require_media_inspection(params)
    assert media["execution_mode"] == "process"
    assert media["status_counts"] == {"timeout": 2}
    assert media["diagnostic_counts"] == {"timeout-video": 2}
    assert media["total_timeouts"] == 2
    assert media["killed_worker_count"] == 2
    videos = sorted(media["videos"], key=lambda row: row["episode_index"])
    assert all(video["inspection_status"] == "timeout" for video in videos)
    assert all(video["inspection_execution"] == "process" for video in videos)
    assert all(video["inspection_worker_killed"] is True for video in videos)
    assert all(video["diagnostics"][0]["worker_killed"] is True for video in videos)
    job = get_lerobot_ingest_job(lake, report.ingest_job_id)
    checkpoint_media = _require_media_inspection(job["history"][-1]["progress"])
    assert checkpoint_media["execution_mode"] == "process"
    assert checkpoint_media["killed_worker_count"] == 2


def test_lerobot_mp4_inspector_supports_object_store_style_seekable_streams(tmp_path, monkeypatch):
    local = tmp_path / "local.mp4"
    payload = _tiny_mp4([b"remote-frame-0", b"remote-frame-1"], keyframes=[0])
    local.write_bytes(payload)
    expected = inspect_mp4_video(local)

    import lancedb_robotics._mp4 as mp4_module

    @contextmanager
    def fake_open_binary_uri(uri, *, storage_options=None, auth_ref=None):
        assert uri == "s3://robotics-lakehouse/videos/episode_000000.mp4"
        assert storage_options == {"anon": True}
        assert auth_ref == "test-auth"
        yield io.BytesIO(payload)

    monkeypatch.setattr(mp4_module, "open_binary_uri", fake_open_binary_uri)

    inspected = mp4_module.inspect_mp4_video(
        "s3://robotics-lakehouse/videos/episode_000000.mp4",
        storage_options={"anon": True},
        auth_ref="test-auth",
    )
    assert inspected.keyframe_map == expected.keyframe_map
    assert inspected.bytes_read == expected.bytes_read
    assert inspected.bytes_read < len(payload)
    assert (
        mp4_module.read_mp4_frame_sample(
            "s3://robotics-lakehouse/videos/episode_000000.mp4",
            inspected.keyframe_map,
            1,
            storage_options={"anon": True},
            auth_ref="test-auth",
        )
        == b"remote-frame-1"
    )


def test_lerobot_source_identity_uses_video_stats_without_hashing_payload(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-video-stat-identity", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    _write_tiny_mp4(first_video, [b"a" * 64_000, b"b" * 64_000], keyframes=[0])

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    original_checksum = lerobot_module.file_checksum
    hashed_paths: list[str] = []

    def checksum_without_video_payloads(path):
        rel = Path(path).resolve().relative_to(source.resolve()).as_posix()
        assert not rel.startswith("videos/")
        hashed_paths.append(rel)
        return original_checksum(path)

    monkeypatch.setattr(lerobot_module, "file_checksum", checksum_without_video_payloads)

    report = get_adapter("lerobot").inspect(source)

    assert report["source_identity"]["kind"] == "content-sha256-media-stat"
    assert hashed_paths
    assert all(not path.startswith("videos/") for path in hashed_paths)
    media = _require_media_inspection(report)
    assert media["status_counts"]["completed"] >= 1


def test_ingest_lerobot_persists_media_inspection_and_keeps_partial_failures_local(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-media-partial")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    missing_video = source / "videos/chunk-001/observation.images.front/episode_000002.mp4"
    corrupt_video = source / "videos/chunk-001/observation.images.front/episode_000003.mp4"
    _write_tiny_mp4(first_video, [b"front-0"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-1"], keyframes=[0])
    missing_video.unlink()
    assert corrupt_video.read_bytes().startswith(b"fake-streaming-mp4-")

    lake = Lake.init(tmp_path / "lake")
    ingest_report = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert ingest_report.rows_added["observations"] == 4
    assert lake.table("observations").count_rows() == 4
    params = _lerobot_ingest_params(lake)
    media = _require_media_inspection(params)
    assert media["video_count"] == 4
    assert media["status_counts"] == {"completed": 2, "failed": 1, "missing": 1}
    assert media["codec_counts"]["h264"] == 2
    assert media["diagnostic_counts"] == {
        "corrupt-video": 1,
        "missing-video": 1,
    }
    videos = sorted(media["videos"], key=lambda row: row["episode_index"])
    assert [video["inspection_status"] for video in videos] == [
        "completed",
        "completed",
        "missing",
        "failed",
    ]
    assert [
        video["diagnostics"][0]["code"] if video["diagnostics"] else None for video in videos
    ] == [
        None,
        None,
        "missing-video",
        "corrupt-video",
    ]

    job = get_lerobot_ingest_job(lake, ingest_report.ingest_job_id)
    checkpoint_media = _require_media_inspection(job["history"][-1]["progress"])
    assert checkpoint_media["status_counts"] == media["status_counts"]
    assert checkpoint_media["diagnostic_counts"] == media["diagnostic_counts"]


def test_ingest_lerobot_retry_reuses_completed_media_inspection_results(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-media-retry", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])
    lake = Lake.init(tmp_path / "lake")

    first = ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    assert first.rows_added["observations"] == 3
    media = _require_media_inspection(_lerobot_ingest_params(lake))
    assert media["status_counts"] == {"completed": 2}

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    def fail_completed_media_reinspection(path):
        raise AssertionError(f"completed media inspection should be reused for {path}")

    monkeypatch.setattr(lerobot_module, "inspect_mp4_video", fail_completed_media_reinspection)

    second = ingest_lerobot(
        lake,
        source,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert second.already_ingested is True
    assert second.rows_added["observations"] == 0
    assert lake.table("observations").count_rows() == 3
    job = get_lerobot_ingest_job(lake, second.ingest_job_id)
    retry_media = _require_media_inspection(job["history"][-1]["progress"])
    assert retry_media["status_counts"] == {"completed": 2}
    assert retry_media["reused_count"] == 2
    assert all(video["inspection_reused"] is True for video in retry_media["videos"])


def test_lerobot_iter_frame_batches_yields_data_file_row_group_batches(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-streaming")
    adapter = get_adapter("lerobot")

    batches = list(adapter.iter_frame_batches(source, batch_size=1))

    assert [
        (batch.data_file, batch.row_group, batch.batch_index, batch.row_count) for batch in batches
    ] == [
        ("data/chunk-000/file-000.parquet", 0, 0, 1),
        ("data/chunk-000/file-000.parquet", 1, 0, 1),
        ("data/chunk-001/file-000.parquet", 0, 0, 1),
        ("data/chunk-001/file-000.parquet", 1, 0, 1),
    ]
    assert [batch.rows[0]["_global_index"] for batch in batches] == [0, 1, 2, 3]
    assert [batch.rows[0]["_source_parquet"] for batch in batches] == [
        "data/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
        "data/chunk-001/file-000.parquet",
        "data/chunk-001/file-000.parquet",
    ]
    assert all(batch.bytes_scanned > 0 for batch in batches)


def test_lerobot_hf_revision_source_identity_avoids_local_content_hash(tmp_path, monkeypatch):
    source = _streaming_lerobot_fixture(tmp_path / "hf-cache")
    adapter = get_adapter("lerobot")
    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    def fail_local_hash(*_args, **_kwargs):
        raise AssertionError("HF revision identity should avoid local content hashing")

    monkeypatch.setattr(
        lerobot_module, "_download_hf_dataset", lambda repo_id, revision=None: source
    )
    monkeypatch.setattr(lerobot_module, "_combined_checksum", fail_local_hash)

    resolved = adapter.source("robotics/corpus@abc123")
    report = adapter.inspect("robotics/corpus@abc123")

    assert resolved.repo_id == "robotics/corpus"
    assert resolved.revision == "abc123"
    assert resolved.identity_kind == "hf-revision"
    assert resolved.checksum == "hf:robotics/corpus@abc123"
    assert report["source_identity"]["kind"] == "hf-revision"
    assert report["source_identity"]["revision"] == "abc123"


def test_ingest_lerobot_uses_frame_batches_without_materializing_frames(tmp_path, monkeypatch):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-streaming-ingest")
    lake = Lake.init(tmp_path / "lake")
    real_adapter = get_adapter("lerobot")

    class StreamingOnlyAdapter:
        info = real_adapter.info

        def __init__(self) -> None:
            self.dataset_include_frames: list[bool] = []
            self.iter_batch_sizes: list[int] = []

        def inspect(self, source_arg, **kwargs):
            return real_adapter.inspect(source_arg, **kwargs)

        def dataset(self, source_arg, *, include_frames=True):
            self.dataset_include_frames.append(include_frames)
            if include_frames:
                pytest.fail("ingest must stream LeRobot frames without requesting dataset.frames")
            return real_adapter.dataset(source_arg, include_frames=False)

        def iter_frame_batches(self, source_arg, *, batch_size=1024):
            self.iter_batch_sizes.append(batch_size)
            yield from real_adapter.iter_frame_batches(source_arg, batch_size=batch_size)

    streaming_adapter = StreamingOnlyAdapter()
    import lancedb_robotics.ingest as ingest_module

    monkeypatch.setattr(
        ingest_module,
        "get_adapter",
        lambda name: streaming_adapter if name == "lerobot" else real_adapter,
    )

    report = ingest_module.ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert report.rows_added["observations"] == 4
    assert streaming_adapter.dataset_include_frames
    assert all(
        include_frames is False for include_frames in streaming_adapter.dataset_include_frames
    )
    assert 1 in streaming_adapter.iter_batch_sizes


def test_ingest_lerobot_v21_maps_canonical_rows_and_video_references(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot-v21", version="v2.1")
    lake = Lake.init(tmp_path / "lake")

    report = ingest_lerobot(
        lake, source, compact=False, prune_versions=False, index_predicates=False
    )

    assert report.already_ingested is False
    assert report.rows_added["episodes"] == 2
    assert report.rows_added["observations"] == 3
    assert report.rows_added["scenarios"] == 2
    assert report.rows_added["videos"] == 2
    assert report.rows_added["video_encodings"] == 2

    episodes = sorted(
        lake.table("episodes").to_arrow().to_pylist(), key=lambda row: row["episode_index"]
    )
    assert [row["task_id"] for row in episodes] == ["pick cube", "place cube"]
    assert [row["frame_count"] for row in episodes] == [2, 1]
    assert episodes[0]["boundary_source"] == "lerobot-authored"

    observations = sorted(
        lake.table("observations").to_arrow().to_pylist(),
        key=lambda row: (row["episode_index"], row["frame_index"]),
    )
    assert observations[0]["state_vector"] == pytest.approx([1.0, 2.0])
    assert observations[0]["action_vector"] == pytest.approx([0.1, 0.2])
    assert observations[0]["task_id"] == "pick cube"
    payload = json.loads(observations[0]["payload_json"])
    assert payload["unmapped"]["vendor.force"] == 3.5
    assert payload["images"]["front"].endswith("episode_000000.mp4")

    videos = lake.table("videos").to_arrow().to_pylist()
    assert videos[0]["uri"].endswith(".mp4")
    assert videos[0]["codec"] == "mp4"
    assert videos[0]["raw_uri"] == videos[0]["uri"]

    encodings = lake.table("video_encodings").to_arrow().to_pylist()
    assert json.loads(encodings[0]["keyframe_map_json"]) == []
    assert encodings[0]["data"] is None
    params = _lerobot_ingest_params(lake)
    assert any(diagnostic["code"] == "corrupt-video" for diagnostic in params["video_diagnostics"])


def test_ingest_lerobot_records_streaming_progress_checkpoints_and_fingerprints(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-progress")
    lake = Lake.init(tmp_path / "lake")
    inspect_report = get_adapter("lerobot").inspect(source)

    report = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    params = _lerobot_ingest_params(lake)
    source_identity = params.get("source_identity") or params.get("source_fingerprint")
    assert source_identity == inspect_report["source_identity"]
    progress = params["progress"]
    assert progress["rows_written"]["observations"] == report.rows_added["observations"] == 4
    assert progress["rows_written"]["episodes"] == report.rows_added["episodes"] == 4
    assert progress["last_checkpoint"]["data_file"] == "data/chunk-001/file-000.parquet"
    assert progress["last_checkpoint"]["row_group"] == 1
    assert progress["last_checkpoint"]["batch_index"] == 0
    assert progress["last_checkpoint"]["rows_written"]["observations"] == 4

    data_files = progress["data_files"]
    assert [item["path"] for item in data_files] == [
        "data/chunk-000/file-000.parquet",
        "data/chunk-001/file-000.parquet",
    ]
    assert [group["row_group"] for item in data_files for group in item["row_groups"]] == [
        0,
        1,
        0,
        1,
    ]
    assert [
        batch["rows_written"]["observations"]
        for item in data_files
        for group in item["row_groups"]
        for batch in group["batches"]
    ] == [1, 1, 1, 1]
    assert all(item["fingerprint"]["value"] for item in data_files)


def test_ingest_lerobot_writes_listable_durable_job_checkpoints(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-durable-checkpoints")
    lake = Lake.init(tmp_path / "lake")

    report = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    checkpoints = _durable_lerobot_checkpoint_rows(lake)
    assert checkpoints, "LeRobot ingest progress must be durable outside transform_runs.params"
    terminal = max(
        checkpoints,
        key=lambda row: (
            row["observations_written"],
            row["episodes_written"],
            row["checkpoint_index"],
        ),
    )
    assert report.ingest_job_id == terminal["job_id"]
    assert checkpoints[0]["phase"] == "claimed"
    assert terminal["status"] == "completed"
    assert terminal["phase"] in {"completed", "finalized"}
    assert terminal["run_id"] == report.run_id
    assert terminal["transform_id"] == report.transform_id
    assert terminal["observations_written"] == report.rows_added["observations"] == 4
    assert terminal["episodes_written"] == report.rows_added["episodes"] == 4

    claimed_progress = json.loads(checkpoints[0]["progress_json"])
    assert claimed_progress["claim"]["active"] is True
    assert claimed_progress["claim"]["claim_expires_at"]
    assert claimed_progress["claim"]["heartbeat_count"] == 1
    progress = json.loads(terminal["progress_json"])
    assert progress["claim"]["active"] is False
    assert progress["claim"]["claim_expires_at"] is None
    assert progress["claim"]["heartbeat_count"] >= 2
    assert progress["last_checkpoint"]["data_file"] == "data/chunk-001/file-000.parquet"
    assert progress["last_checkpoint"]["rows_written"]["observations"] == 4
    transform_rows = lake.table("transform_runs").to_arrow().to_pylist()
    ingest_transform = next(
        row for row in transform_rows if row["transform_id"] == report.transform_id
    )
    assert "lerobot_ingest_checkpoints" in ingest_transform["output_tables"]

    jobs = list_lerobot_ingest_jobs(lake, limit=5)
    assert [job["job_id"] for job in jobs] == [report.ingest_job_id]
    detail = get_lerobot_ingest_job(lake, report.ingest_job_id)
    assert [entry["checkpoint_id"] for entry in detail["history"]] == [
        row["checkpoint_id"] for row in checkpoints
    ]


def test_lerobot_checkpoint_retention_summarizes_old_terminal_histories(tmp_path):
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    old = now - timedelta(days=90)
    recent = now - timedelta(days=1)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-old-completed",
        source_id="src-a",
        updated_at=old,
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("metadata-ready", "running"),
            ("finalized", "completed"),
        ],
    )
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-recent-failed",
        source_id="src-a",
        updated_at=recent,
        phases=[("claimed", "running"), ("frame-batch", "failed")],
        error="recent failure",
    )
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-old-running",
        source_id="src-a",
        updated_at=old,
        phases=[("claimed", "running"), ("frame-batch", "running")],
    )
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-old-failed",
        source_id="src-a",
        updated_at=old,
        phases=[("claimed", "running"), ("frame-batch", "running"), ("frame-batch", "failed")],
        error="old failure",
    )

    dry_run = apply_lerobot_checkpoint_retention(
        lake,
        older_than=timedelta(days=30),
        retain_completed_per_source=0,
        retain_failed_per_source=1,
        dry_run=True,
        compact=False,
        cleanup_older_than=None,
        retain_versions=None,
        now=now,
    )
    assert dry_run.rows_before == 11
    assert dry_run.rows_after == 6
    assert dry_run.rows_deleted == 5
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 11

    report = apply_lerobot_checkpoint_retention(
        lake,
        older_than=timedelta(days=30),
        retain_completed_per_source=0,
        retain_failed_per_source=1,
        dry_run=False,
        compact=False,
        cleanup_older_than=None,
        retain_versions=None,
        now=now,
    )
    assert report.rows_before == 11
    assert report.rows_after == 6
    assert report.rows_deleted == 5
    assert report.jobs_compacted == 2

    completed = get_lerobot_ingest_job(lake, "job-old-completed")
    assert [row["phase"] for row in completed["history"]] == ["finalized"]
    assert completed["status"] == "completed"
    assert completed["requested_revision"] == "main"
    assert completed["resolved_revision"] == "a" * 40
    assert (
        completed["hf_download"]["manifest_fingerprints"][0]["fingerprint"]
        == "fp-job-old-completed"
    )
    assert completed["source_identity"]["manifest_fingerprint"] == "manifest-job-old-completed"
    assert completed["progress"]["media_inspection"]["status_counts"] == {"completed": 2}
    assert completed["rows_written"]["observations"] == completed["observations_written"]

    failed = get_lerobot_ingest_job(lake, "job-old-failed")
    assert [row["phase"] for row in failed["history"]] == ["frame-batch"]
    assert failed["status"] == "failed"
    assert failed["error"] == "old failure"
    assert failed["progress"]["error"] == "old failure"

    assert len(get_lerobot_ingest_job(lake, "job-recent-failed")["history"]) == 2
    assert len(get_lerobot_ingest_job(lake, "job-old-running")["history"]) == 2


def test_lerobot_checkpoint_retention_preserves_lineage_held_checkpoints(tmp_path):
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    old = now - timedelta(days=90)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-held",
        source_id="src-held",
        updated_at=old,
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )
    lake.table("lineage_artifacts").add(
        pa.Table.from_pylist(
            [
                {
                    "artifact_id": "artifact-held-checkpoint",
                    "kind": "table-row",
                    "name": "held LeRobot checkpoint",
                    "table_name": "lerobot_ingest_checkpoints",
                    "table_version": int(lake.table("lerobot_ingest_checkpoints").version),
                    "table_tag": "",
                    "row_grain": "checkpoint",
                    "row_ids": ["job-held:00000001"],
                    "source_uri": "",
                    "source_id": "src-held",
                    "digest": "",
                    "producer_execution_id": "",
                    "metadata": [
                        {"key": "audit_hold", "value": "true"},
                        {"key": "reason", "value": "operator audit"},
                    ],
                    "created_at": now,
                }
            ],
            schema=LINEAGE_ARTIFACTS_SCHEMA,
        )
    )

    report = apply_lerobot_checkpoint_retention(
        lake,
        older_than=timedelta(days=30),
        retain_completed_per_source=0,
        retain_failed_per_source=0,
        dry_run=False,
        compact=False,
        cleanup_older_than=None,
        retain_versions=None,
        now=now,
    )

    assert report.rows_deleted == 0
    assert report.jobs[0].reason == "retention-hold"
    assert report.jobs[0].hold_ids == ("artifact-held-checkpoint",)
    assert report.jobs[0].hold_reasons == ("operator audit",)
    assert report.protected_checkpoint_ids == ("job-held:00000001",)
    assert len(get_lerobot_ingest_job(lake, "job-held")["history"]) == 3


def test_lerobot_checkpoint_catalog_hold_apply_release_and_retention(tmp_path):
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-catalog-held",
        source_id="src-held",
        updated_at=now - timedelta(days=90),
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )

    hold = hold_lerobot_checkpoints(
        lake,
        job_id="job-catalog-held",
        legal_hold=True,
        audit_hold=False,
        owner="governance",
        reason="legal discovery",
        created_by="test",
        now=now,
    )
    assert hold.active is True
    assert hold.checkpoint_ids == (
        "job-catalog-held:00000000",
        "job-catalog-held:00000001",
        "job-catalog-held:00000002",
    )

    dry_run = apply_lerobot_checkpoint_retention(
        lake,
        older_than=timedelta(days=30),
        retain_completed_per_source=0,
        retain_failed_per_source=0,
        dry_run=True,
        compact=False,
        cleanup_older_than=None,
        retain_versions=None,
        now=now,
    )
    assert dry_run.rows_deleted == 0
    assert dry_run.jobs[0].reason == "retention-hold"
    assert dry_run.jobs[0].hold_ids == (hold.hold_id,)
    assert dry_run.jobs[0].hold_reasons == ("legal discovery",)

    released = release_lerobot_checkpoint_hold(
        lake,
        hold.hold_id,
        released_by="governance",
        now=now + timedelta(hours=1),
    )
    assert released.active is False
    assert released.released_by == "governance"

    applied = apply_lerobot_checkpoint_retention(
        lake,
        older_than=timedelta(days=30),
        retain_completed_per_source=0,
        retain_failed_per_source=0,
        dry_run=False,
        compact=False,
        cleanup_older_than=None,
        retain_versions=None,
        now=now + timedelta(hours=2),
    )
    assert applied.rows_deleted == 2
    assert get_lerobot_ingest_job(lake, "job-catalog-held")["history"][0]["phase"] == "finalized"


def test_lerobot_checkpoint_catalog_hold_selects_source_repo_status_and_window(tmp_path):
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    old = now - timedelta(days=90)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-window-held",
        source_id="src-window",
        hf_repo_id="robotics/held",
        requested_revision="main",
        resolved_revision="b" * 40,
        updated_at=old,
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-window-other",
        source_id="src-window",
        hf_repo_id="robotics/other",
        requested_revision="main",
        resolved_revision="c" * 40,
        updated_at=old,
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )

    hold = hold_lerobot_checkpoints(
        lake,
        source_id="src-window",
        hf_repo_id="robotics/held",
        requested_revision="main",
        resolved_revision="b" * 40,
        status=("completed",),
        updated_after=old - timedelta(minutes=1),
        updated_before=old + timedelta(minutes=1),
        reason="audit window",
        owner="audit",
        now=now,
    )

    assert hold.selector["source_id"] == "src-window"
    assert hold.selector["statuses"] == ["completed"]
    assert hold.checkpoint_ids == ("job-window-held:00000002",)

    report = apply_lerobot_checkpoint_retention(
        lake,
        older_than=timedelta(days=30),
        retain_completed_per_source=0,
        retain_failed_per_source=0,
        dry_run=True,
        compact=False,
        cleanup_older_than=None,
        retain_versions=None,
        now=now,
    )
    by_job = {job.job_id: job for job in report.jobs}
    assert by_job["job-window-held"].reason == "retention-hold"
    assert by_job["job-window-held"].hold_reasons == ("audit window",)
    assert by_job["job-window-other"].reason == "terminal-summary"
    assert by_job["job-window-other"].rows_deleted == 2


def test_lake_maintain_runs_lerobot_checkpoint_retention(tmp_path):
    from lancedb_robotics.maintenance import maintain_lake

    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-maintain-old",
        source_id="src-maintain",
        updated_at=now - timedelta(days=90),
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )

    report = maintain_lake(
        lake,
        tables=("lerobot_ingest_checkpoints",),
        compact=False,
        refresh_indexes=False,
        protect_lineage=False,
        cleanup_older_than=None,
        retain_versions=None,
        lerobot_checkpoint_retention_older_than=timedelta(days=30),
        lerobot_checkpoint_retain_completed_per_source=0,
        lerobot_checkpoint_retain_failed_per_source=0,
        created_by="test",
    )

    assert report.lerobot_checkpoint_retention is not None
    assert report.lerobot_checkpoint_retention["rows_deleted"] == 2
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 1
    maintenance_row = max(
        lake.table("transform_runs").to_arrow().to_pylist(),
        key=lambda row: row["created_at"],
    )
    params = json.loads(maintenance_row["params"])
    assert params["lerobot_checkpoint_retention"]["jobs_compacted"] == 1


def test_lerobot_checkpoint_retention_schedule_emits_threshold_telemetry(tmp_path):
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-scheduled-old",
        source_id="src-scheduled",
        updated_at=now - timedelta(days=90),
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )

    report = run_lerobot_checkpoint_retention_schedule(
        lake,
        schedule_id="nightly-checkpoint-retention",
        interval=timedelta(hours=6),
        older_than=timedelta(days=30),
        retain_completed_per_source=0,
        retain_failed_per_source=0,
        dry_run=True,
        max_rows=0,
        max_rows_per_source=0,
        max_version_delta=0,
        now=now,
    )

    assert report.schedule_id == "nightly-checkpoint-retention"
    assert report.dry_run is True
    assert report.next_run_after == now + timedelta(hours=6)
    assert report.retention.rows_deleted == 2
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 3
    assert report.telemetry["rows_before"] == 3
    assert report.telemetry["rows_after"] == 1
    assert report.telemetry["rows_deleted"] == 2
    assert report.telemetry["version_delta"] == 0
    assert report.telemetry["reason_counts"] == {"terminal-summary": 1}
    assert report.telemetry["per_source"]["src-scheduled"]["rows_after"] == 1
    assert {alert["metric"] for alert in report.alerts} == {
        "rows_after",
        "rows_after_per_source",
    }


def test_lerobot_checkpoint_retention_plan_uses_observed_rows_without_mutation(tmp_path):
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-plan-old-a",
        source_id="src-plan",
        updated_at=now - timedelta(days=90),
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-plan-old-b",
        source_id="src-plan",
        updated_at=now - timedelta(days=89),
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )

    report = plan_lerobot_checkpoint_retention_scale(
        lake,
        scenario="ci",
        source_id="src-plan",
        now=now,
    )

    assert report.mode == "observed"
    assert report.recommended_policy == "ci-disposable"
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 6
    by_name = {policy.name: policy for policy in report.policies}
    ci = by_name["ci-disposable"]
    assert ci.rows_before == 6
    assert ci.rows_after == 2
    assert ci.rows_deleted == 4
    assert ci.estimated_version_delta == 1
    assert ci.jobs_compacted == 2
    assert ci.protected_jobs_by_reason == {"terminal-summary": 2}
    audit = by_name["audit-window"]
    assert audit.rows_after == 6
    assert audit.jobs_protected == 2


def test_lerobot_checkpoint_retention_plan_accepts_checkpoint_rows_without_lake(tmp_path):
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-row-plan-held",
        source_id="src-row-plan",
        updated_at=now - timedelta(days=90),
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )
    rows = lake.table("lerobot_ingest_checkpoints").to_arrow().to_pylist()

    report = plan_lerobot_checkpoint_retention_scale(
        checkpoint_rows=rows,
        hold_details_by_checkpoint={
            "job-row-plan-held:00000001": ({"hold_id": "hold-row-plan", "reason": "audit sample"},)
        },
        scenario="ci",
        now=now,
    )

    assert report.mode == "checkpoint-rows"
    assert report.lake_uri is None
    by_name = {policy.name: policy for policy in report.policies}
    ci = by_name["ci-disposable"]
    assert ci.rows_deleted == 0
    assert ci.hold_protected_jobs == 1
    assert ci.protected_jobs_by_reason == {"retention-hold": 1}


def test_lerobot_checkpoint_retention_plan_projects_synthetic_scale():
    report = plan_lerobot_checkpoint_retention_scale(
        scenario="full-public-corpus",
        synthetic_sources=2,
        synthetic_completed_jobs_per_source=25,
        synthetic_failed_jobs_per_source=5,
        synthetic_running_jobs_per_source=1,
        synthetic_checkpoints_per_job=4,
        synthetic_terminal_age_days=90,
    )

    assert report.mode == "synthetic"
    assert report.synthetic == {
        "sources": 2,
        "completed_jobs_per_source": 25,
        "failed_jobs_per_source": 5,
        "running_jobs_per_source": 1,
        "checkpoints_per_job": 4,
        "terminal_age_days": 90.0,
    }
    assert report.recommended_policy == "full-public-corpus"
    by_name = {policy.name: policy for policy in report.policies}
    full = by_name["full-public-corpus"]
    assert full.rows_before == 248
    assert full.jobs_compacted == 30
    assert full.jobs_protected == 32
    assert full.rows_deleted == 90
    assert full.rows_after == 158
    assert full.protected_jobs_by_reason == {
        "active": 2,
        "minimum-history": 30,
        "terminal-summary": 30,
    }
    assert by_name["audit-window"].rows_deleted == 0

    filtered = plan_lerobot_checkpoint_retention_scale(
        scenario="ci",
        source_id="src-0",
        synthetic_sources=2,
        synthetic_completed_jobs_per_source=3,
        synthetic_failed_jobs_per_source=0,
        synthetic_running_jobs_per_source=1,
        synthetic_checkpoints_per_job=4,
        synthetic_terminal_age_days=90,
    )
    filtered_ci = {policy.name: policy for policy in filtered.policies}["ci-disposable"]
    assert filtered_ci.rows_before == 32
    assert filtered_ci.rows_deleted == 9
    assert filtered_ci.jobs_compacted == 3
    assert filtered_ci.jobs_protected == 5
    assert filtered_ci.protected_jobs_by_reason == {
        "active": 1,
        "source-filter": 4,
        "terminal-summary": 3,
    }


def test_lerobot_claim_recovery_chaos_rehearsal_detects_broken_cas(monkeypatch):
    """The rehearsal must actually exercise the real CAS code, not model it.

    Directly closes the original audit finding: the old simulator hardcoded
    duplicate_rows to zero and never called the real recovery/CAS path, so
    it could not have caught this even if the guard were completely
    missing. Here we monkeypatch the CAS guard itself to a no-op (every
    racer believes it won), and confirm checkpoint_duplicate_rows/`passed`
    actually detect the resulting duplication. If the rehearsal didn't call
    the real code, this monkeypatch would have no effect and the assertion
    below would wrongly fail to catch it -- proving the wiring is real.
    """
    import lancedb_robotics.ingest as ingest_module

    monkeypatch.setattr(
        ingest_module,
        "_lerobot_claim_cas_supersede",
        lambda *args, **kwargs: None,  # every racer believes it won the race
    )

    report = simulate_lerobot_claim_recovery_chaos(
        synthetic_sources=1,
        synthetic_running_jobs_per_source=1,
        synthetic_stale_running_fraction=1.0,
        retry_owner_count=4,
        seed=1,
    )

    recovery_points = [c for c in report.crash_points if c.recovery_required]
    assert recovery_points
    assert all(c.checkpoint_duplicate_rows == 3 for c in recovery_points)
    assert all(c.accepted_recoveries == 4 for c in recovery_points)
    assert all(c.cas_conflicts == 0 for c in recovery_points)
    assert report.passed is False


def test_lerobot_claim_recovery_simulator_is_deterministic_and_checks_duplicates():
    now = datetime(2026, 1, 15, 12, tzinfo=UTC)

    first = simulate_lerobot_claim_recovery_chaos(
        scenario="ci",
        synthetic_sources=2,
        synthetic_completed_jobs_per_source=1,
        synthetic_failed_jobs_per_source=1,
        synthetic_running_jobs_per_source=4,
        synthetic_checkpoints_per_job=6,
        synthetic_stale_running_fraction=0.5,
        synthetic_missing_lease_fraction=0.25,
        source_size_frames=10,
        episode_count=5,
        camera_count=2,
        batch_size=3,
        retry_owner_count=3,
        remote_latency_ms=25,
        seed=7,
        now=now,
    )
    second = simulate_lerobot_claim_recovery_chaos(
        scenario="ci",
        synthetic_sources=2,
        synthetic_completed_jobs_per_source=1,
        synthetic_failed_jobs_per_source=1,
        synthetic_running_jobs_per_source=4,
        synthetic_checkpoints_per_job=6,
        synthetic_stale_running_fraction=0.5,
        synthetic_missing_lease_fraction=0.25,
        source_size_frames=10,
        episode_count=5,
        camera_count=2,
        batch_size=3,
        retry_owner_count=3,
        remote_latency_ms=25,
        seed=7,
        now=now,
    )

    assert first.to_params() == second.to_params()
    payload = first.to_params()
    assert payload["mode"] == "synthetic"
    assert payload["passed"] is True
    assert payload["watchdog"]["stale_count"] == 4
    assert payload["watchdog"]["missing_lease_count"] == 2
    assert payload["watchdog"]["live_count"] == 4
    assert payload["recovery"]["accepted_recoveries"] == 4
    assert payload["recovery"]["cas_conflicts"] == 8
    assert payload["recommendations"]["profile"] == "ci"
    assert payload["duplicate_protection"]["expected_final_rows"]["videos"] == 10
    by_crash = {crash["crash_point"]: crash for crash in payload["crash_points"]}
    assert by_crash["frame-batch"]["rows_skipped_existing"] == 6
    assert by_crash["frame-batch"]["observations_written_after_retry"] == 4
    assert by_crash["metadata-ready"]["rows_skipped_existing"] == 10
    assert all(sum(crash["duplicate_rows"].values()) == 0 for crash in payload["crash_points"])
    assert payload["retention_plan"]["mode"] == "synthetic"


def test_lerobot_claim_recovery_simulator_observed_lake_does_not_mutate(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-claim-chaos")
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, 12, tzinfo=UTC)
    _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-chaos-stale",
        source_id="src-chaos",
        updated_at=now - timedelta(hours=2),
        claim_expires_at=now - timedelta(minutes=30),
        claim_owner="worker-stale",
        claim_token="token-stale",
        rows_seen=7,
        observations_written=5,
    )
    _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-chaos-live",
        source_id="src-chaos",
        updated_at=now - timedelta(minutes=1),
        claim_expires_at=now + timedelta(minutes=10),
        claim_owner="worker-live",
        claim_token="token-live",
    )
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-chaos-complete",
        source_id="src-chaos",
        updated_at=now - timedelta(days=30),
        phases=[("claimed", "running"), ("finalized", "completed")],
    )
    table = lake.table("lerobot_ingest_checkpoints")
    rows_before = table.count_rows()
    version_before = table.version

    report = simulate_lerobot_claim_recovery_chaos(
        lake,
        source_id="src-chaos",
        source_size_frames=4,
        batch_size=2,
        stale_after=timedelta(minutes=15),
        now=now,
    )

    assert table.count_rows() == rows_before
    assert table.version == version_before
    payload = report.to_params()
    assert payload["mode"] == "observed"
    assert payload["watchdog"]["stale_count"] == 1
    assert payload["watchdog"]["live_count"] == 1
    assert payload["watchdog"]["inactive_count"] == 1
    assert payload["watchdog"]["stale_reasons"] == {"expired-claim": 1}
    assert payload["recovery"]["accepted_recoveries"] == 1
    assert payload["crash_points"][-1]["crash_point"] == "metadata-ready"
    assert payload["crash_points"][-1]["observations_written_after_retry"] == 0


def test_ingest_lerobot_claims_same_source_once_and_records_retry(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-claim-retry")
    lake = Lake.init(tmp_path / "lake")

    first = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        ingest_job_id="job-lerobot-claim-retry",
        claim_owner="worker-a",
    )
    second = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        ingest_job_id="job-lerobot-claim-retry",
        claim_owner="worker-b",
    )

    assert first.ingest_job_id == second.ingest_job_id == "job-lerobot-claim-retry"
    assert first.already_ingested is False
    assert second.already_ingested is True
    assert second.rows_added["observations"] == 0
    assert lake.table("observations").count_rows() == 4

    history = get_lerobot_ingest_job(lake, "job-lerobot-claim-retry")["history"]
    claimed = [entry for entry in history if entry["phase"] == "claimed"]
    assert [entry["claim_owner"] for entry in claimed] == ["worker-a"]
    assert any(
        entry["claim_owner"] == "worker-b"
        and entry["phase"] in {"deduped", "skipped-duplicate", "reused-completed"}
        for entry in history
    )


def test_ingest_lerobot_reports_running_claim_without_writing_rows(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-running-claim")
    lake = Lake.init(tmp_path / "lake")
    now = datetime.now(UTC)
    job_id = _seed_lerobot_running_claim(
        lake,
        source,
        updated_at=now,
        claim_expires_at=now + timedelta(minutes=5),
    )

    with pytest.raises(AdapterError, match="already running") as excinfo:
        ingest_lerobot(
            lake,
            source,
            batch_size=1,
            compact=False,
            prune_versions=False,
            index_predicates=False,
        )

    message = str(excinfo.value)
    assert job_id in message
    assert "lease expires at" in message
    assert lake.table("observations").count_rows() == 0


def test_lerobot_claim_watchdog_reports_live_stale_missing_and_inactive_without_mutation(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-claim-watchdog")
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, 12, tzinfo=UTC)
    _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-watchdog-live",
        source_id="src-watchdog",
        updated_at=now - timedelta(minutes=1),
        claim_expires_at=now + timedelta(minutes=5),
        claim_owner="worker-live",
        claim_token="token-live",
    )
    _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-watchdog-stale",
        source_id="src-watchdog",
        updated_at=now - timedelta(hours=2),
        claim_expires_at=now - timedelta(minutes=30),
        claim_owner="worker-stale",
        claim_token="token-stale",
        rows_seen=42,
        observations_written=40,
    )
    _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-watchdog-missing-lease",
        source_id="src-watchdog",
        updated_at=now - timedelta(hours=2),
        include_claim=False,
        claim_owner="worker-missing",
        claim_token="token-missing",
        rows_seen=3,
        observations_written=2,
    )
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-watchdog-abandoned",
        source_id="src-watchdog",
        updated_at=now - timedelta(hours=1),
        phases=[
            ("claimed", "running"),
            ("claim-abandoned", "abandoned"),
        ],
    )
    rows_before = lake.table("lerobot_ingest_checkpoints").count_rows()

    report = watch_lerobot_ingest_claims(
        lake,
        source_id="src-watchdog",
        stale_after=timedelta(minutes=30),
        recovery_action="steal",
        new_owner="operator",
        now=now,
    )

    assert lake.table("lerobot_ingest_checkpoints").count_rows() == rows_before
    assert report.stale_count == 2
    assert report.live_count == 1
    assert report.inactive_count == 1
    by_stale_job = {finding.job_id: finding for finding in report.stale_claims}
    assert set(by_stale_job) == {"job-watchdog-stale", "job-watchdog-missing-lease"}
    stale = by_stale_job["job-watchdog-stale"]
    assert stale.claim_owner == "worker-stale"
    assert stale.claim_token == "token-stale"
    assert stale.stale_reason == "expired-claim"
    assert stale.lease_state == "stale"
    assert stale.rows_seen == 42
    assert stale.observations_written == 40
    assert "--action steal" in stale.suggested_recovery_command
    assert "--new-owner operator" in stale.suggested_recovery_command
    missing = by_stale_job["job-watchdog-missing-lease"]
    assert missing.stale_reason == "missing-lease"
    assert missing.lease_state == "missing-lease"
    assert missing.expiration_source == "missing-lease"
    assert missing.claim_expires_at == now - timedelta(minutes=90)
    live = report.live_claims[0]
    assert live.job_id == "job-watchdog-live"
    assert live.stale is False
    assert live.lease_state == "live"
    assert live.seconds_until_stale == 300.0
    inactive = report.inactive_jobs[0]
    assert inactive.job_id == "job-watchdog-abandoned"
    assert inactive.status == "abandoned"
    assert inactive.stale is False
    assert inactive.suggested_recovery_command is None


def test_lerobot_stale_claim_recovery_resumes_without_duplicate_rows(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-stale-claim-recovery")
    lake = Lake.init(tmp_path / "lake")
    adapter = get_adapter("lerobot")
    dataset = adapter.dataset(source, include_frames=True)
    run_id = f"run-{dataset.source.digest}"
    partial_rows, _episode_state = _lerobot_observation_rows(
        dataset,
        run_id=run_id,
        transform_id=f"tfm-{dataset.source.digest}-ingest",
        created_at=datetime.now(UTC),
    )
    lake.table("observations").add(
        pa.Table.from_pylist(partial_rows[:2], schema=OBSERVATIONS_SCHEMA)
    )

    recovery_time = datetime(2026, 1, 15, 12, tzinfo=UTC)
    job_id = _seed_lerobot_running_claim(
        lake,
        source,
        updated_at=recovery_time - timedelta(hours=2),
        claim_expires_at=recovery_time - timedelta(minutes=30),
        rows_seen=2,
        observations_written=2,
    )

    with pytest.raises(AdapterError, match="already running") as excinfo:
        ingest_lerobot(
            lake,
            source,
            batch_size=1,
            compact=False,
            prune_versions=False,
            index_predicates=False,
        )
    assert "lerobot-claim-recover" in str(excinfo.value)

    recovery = recover_lerobot_ingest_claim(
        lake,
        job_id,
        action="steal",
        new_owner="worker-b",
        stale_after=timedelta(minutes=15),
        now=recovery_time,
    )
    assert recovery.action == "steal"
    assert recovery.status == "abandoned"
    assert recovery.phase == "claim-stolen"
    assert recovery.previous_owner == "worker-a"
    assert recovery.new_owner == "worker-b"

    recovered_history = get_lerobot_ingest_job(lake, job_id)["history"]
    recovery_row = recovered_history[-1]
    recovery_progress = recovery_row["progress"]
    assert recovery_row["status"] == "abandoned"
    assert recovery_progress["claim"]["active"] is False
    assert recovery_progress["claim_recovery"]["previous_token"] == "claim-token"
    assert recovery_progress["claim_recovery"]["new_token"] == recovery.new_token

    report = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
        claim_owner="worker-b",
    )

    assert report.ingest_job_id == job_id
    assert report.rows_added["observations"] == 2
    assert lake.table("observations").count_rows() == 4
    key_columns = {
        "observations": "observation_id",
        "episodes": "episode_id",
        "videos": "video_id",
        "events": "event_id",
        "runs": "run_id",
        "transform_runs": "transform_id",
    }
    for table_name, key_column in key_columns.items():
        rows = lake.table(table_name).to_arrow().to_pylist()
        assert len({row[key_column] for row in rows}) == len(rows), table_name

    history = get_lerobot_ingest_job(lake, job_id)["history"]
    assert [entry["phase"] for entry in history[:3]] == ["claimed", "claim-stolen", "claimed"]
    assert history[2]["claim_owner"] == "worker-b"


def test_lerobot_claim_recovery_cas_rejects_loser_after_first_transition(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-recovery-cas")
    lake = Lake.init(tmp_path / "lake")
    recovery_time = datetime(2026, 1, 15, 12, tzinfo=UTC)
    job_id = _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-recovery-cas",
        updated_at=recovery_time - timedelta(hours=2),
        claim_expires_at=recovery_time - timedelta(minutes=30),
        claim_token="token-recovery-cas",
    )
    observed = get_lerobot_ingest_job(lake, job_id)

    accepted = recover_lerobot_ingest_claim(
        lake,
        job_id,
        action="abandon",
        new_owner="operator-a",
        stale_after=timedelta(minutes=15),
        expected_latest_checkpoint_id=observed["checkpoint_id"],
        expected_latest_claim_token=observed["claim_token"],
        expected_checkpoint_index=observed["checkpoint_index"],
        now=recovery_time,
    )

    with pytest.raises(LeRobotClaimPreconditionError) as excinfo:
        recover_lerobot_ingest_claim(
            lake,
            job_id,
            action="steal",
            new_owner="operator-b",
            stale_after=timedelta(minutes=15),
            expected_latest_checkpoint_id=observed["checkpoint_id"],
            expected_latest_claim_token=observed["claim_token"],
            expected_checkpoint_index=observed["checkpoint_index"],
            now=recovery_time + timedelta(seconds=1),
        )

    message = str(excinfo.value)
    assert "claim precondition failed" in message
    assert observed["checkpoint_id"] in message
    assert accepted.recovery_checkpoint_id in message
    assert "status='abandoned'" in message
    history = get_lerobot_ingest_job(lake, job_id)["history"]
    assert [row["phase"] for row in history] == ["claimed", "claim-abandoned"]


def test_ingest_lerobot_claim_cas_rejects_stale_retry_observation(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-ingest-cas")
    lake = Lake.init(tmp_path / "lake")
    recovery_time = datetime(2026, 1, 15, 12, tzinfo=UTC)
    job_id = _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-ingest-cas",
        updated_at=recovery_time - timedelta(hours=2),
        claim_expires_at=recovery_time - timedelta(minutes=30),
        claim_token="token-ingest-cas",
    )
    observed = get_lerobot_ingest_job(lake, job_id)
    recovered = recover_lerobot_ingest_claim(
        lake,
        job_id,
        action="abandon",
        new_owner="operator",
        stale_after=timedelta(minutes=15),
        expected_latest_checkpoint_id=observed["checkpoint_id"],
        expected_latest_claim_token=observed["claim_token"],
        expected_checkpoint_index=observed["checkpoint_index"],
        now=recovery_time,
    )
    rows_after_recovery = lake.table("lerobot_ingest_checkpoints").count_rows()

    with pytest.raises(LeRobotClaimPreconditionError) as excinfo:
        ingest_lerobot(
            lake,
            source,
            batch_size=1,
            compact=False,
            prune_versions=False,
            index_predicates=False,
            ingest_job_id=job_id,
            expected_latest_checkpoint_id=observed["checkpoint_id"],
            expected_latest_claim_token=observed["claim_token"],
            expected_checkpoint_index=observed["checkpoint_index"],
        )

    message = str(excinfo.value)
    assert recovered.recovery_checkpoint_id in message
    assert "claim precondition failed" in message
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == rows_after_recovery


def test_lerobot_claim_recovery_cas_true_concurrent_race_has_single_winner(tmp_path, monkeypatch):
    """Two operators race to recover the SAME claim with no explicit expected_*.

    This is the scenario backlog 0379 names directly: "two operators or
    automation loops... observe the same stale latest row and attempt
    recovery." Neither caller supplies expected_latest_* (there is nothing
    stale to compare against -- both are acting on a fresh, correct read),
    so this exercises the CAS-supersede guard itself, not the older
    explicit-precondition check covered above. A ``threading.Barrier``
    forces both callers' underlying ``table.update(...)`` calls to execute
    at the same instant rather than merely close together in wall-clock
    time, verified separately to reliably yield exactly one winner (see the
    0379 decision record).
    """
    import lancedb_robotics.ingest as ingest_module

    source = _streaming_lerobot_fixture(tmp_path / "lerobot-recovery-race")
    lake = Lake.init(tmp_path / "lake")
    recovery_time = datetime(2026, 1, 15, 12, tzinfo=UTC)
    job_id = _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-recovery-race",
        updated_at=recovery_time - timedelta(hours=2),
        claim_expires_at=recovery_time - timedelta(minutes=30),
        claim_token="token-recovery-race",
    )

    real_supersede = ingest_module._lerobot_claim_cas_supersede
    barrier = threading.Barrier(2)

    def synchronized_supersede(*args, **kwargs):
        barrier.wait()
        return real_supersede(*args, **kwargs)

    monkeypatch.setattr(ingest_module, "_lerobot_claim_cas_supersede", synchronized_supersede)

    results: dict[str, tuple[str, object]] = {}

    def racer(owner: str) -> None:
        try:
            report = recover_lerobot_ingest_claim(
                lake,
                job_id,
                action="steal",
                new_owner=owner,
                stale_after=timedelta(minutes=15),
                now=recovery_time,
            )
            results[owner] = ("ok", report)
        except LeRobotClaimPreconditionError as exc:
            results[owner] = ("error", exc)

    threads = [
        threading.Thread(target=racer, args=(owner,)) for owner in ("operator-a", "operator-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    outcomes = [status for status, _ in results.values()]
    assert outcomes.count("ok") == 1, results
    assert outcomes.count("error") == 1, results

    # The loser's diagnostic names the winner's actual checkpoint, not a
    # generic failure -- matching the backlog's "explain which newer
    # checkpoint won the race" requirement.
    winner_owner = next(owner for owner, (status, _) in results.items() if status == "ok")
    loser_owner = next(owner for owner, (status, _) in results.items() if status == "error")
    winning_report = results[winner_owner][1]
    losing_error = results[loser_owner][1]
    assert winning_report.recovery_checkpoint_id in str(losing_error)

    # Exactly one recovery checkpoint landed -- not two, not zero.
    history = get_lerobot_ingest_job(lake, job_id)["history"]
    recovery_rows = [row for row in history if row["phase"] in ("claim-abandoned", "claim-stolen")]
    assert len(recovery_rows) == 1
    assert recovery_rows[0]["claim_owner"] == winner_owner


def test_lerobot_claim_cas_supersede_rejects_second_writer_deterministically(tmp_path):
    """Direct, repeated stress of the CAS primitive itself (not the full recovery flow).

    ``_lerobot_claim_cas_supersede`` is the load-bearing guard behind both the
    recovery and initial-claim paths. Run the actual race many times (not
    once) so a test that only got lucky on one interleaving would still be
    caught -- every trial must produce exactly one winner and zero duplicate
    rows, never both-succeed (the failure mode a plain ``.add()`` or an
    insert-only ``merge_insert`` both have, verified directly in the 0379
    decision record).
    """
    from lancedb_robotics.ingest import LeRobotClaimPreconditionError as CasError
    from lancedb_robotics.ingest import _lerobot_claim_cas_supersede

    def racer(
        owner: str, lake: Lake, job_id: str, barrier: threading.Barrier, outcomes: dict[str, str]
    ) -> None:
        barrier.wait()
        try:
            _lerobot_claim_cas_supersede(
                lake,
                job_id=job_id,
                prior_checkpoint_id=f"{job_id}:00000000",
                prior_claim_token="prior-token",
                new_checkpoint_id=f"{job_id}:{owner}",
                operation="recover",
            )
            outcomes[owner] = "ok"
        except CasError:
            outcomes[owner] = "error"

    for trial in range(20):
        lake = Lake.init(tmp_path / f"lake-cas-stress-{trial}")
        job_id = f"job-cas-stress-{trial}"
        row = {
            "checkpoint_id": f"{job_id}:00000000",
            "job_id": job_id,
            "claim_token": "prior-token",
            "checkpoint_index": 0,
            "status": "running",
            "phase": "claimed",
            "claim_owner": "seed",
            "created_by": "seed",
            "started_at": datetime(2026, 1, 1, tzinfo=UTC),
            "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        }
        lake.table("lerobot_ingest_checkpoints").add(
            pa.Table.from_pylist([row], schema=LEROBOT_INGEST_CHECKPOINTS_SCHEMA)
        )

        barrier = threading.Barrier(2)
        outcomes: dict[str, str] = {}
        threads = [
            threading.Thread(target=racer, args=(o, lake, job_id, barrier, outcomes))
            for o in ("A", "B")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(outcomes.values()) == ["error", "ok"], (trial, outcomes)
        winning_rows = [
            r
            for r in lake.table("lerobot_ingest_checkpoints").to_arrow().to_pylist()
            if r["checkpoint_id"] == f"{job_id}:00000000"
        ]
        assert len(winning_rows) == 1
        assert winning_rows[0]["superseded_by_checkpoint_id"] in {
            f"job-cas-stress-{trial}:A",
            f"job-cas-stress-{trial}:B",
        }


def test_ingest_lerobot_failed_checkpoint_can_resume_without_duplicates(tmp_path, monkeypatch):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-failed-resume")
    lake = Lake.init(tmp_path / "lake")
    real_adapter = get_adapter("lerobot")

    class FailingAfterFirstBatchAdapter:
        info = real_adapter.info

        def inspect(self, source_arg, **kwargs):
            return real_adapter.inspect(source_arg, **kwargs)

        def dataset(self, source_arg, *, include_frames=True):
            return real_adapter.dataset(source_arg, include_frames=include_frames)

        def iter_frame_batches(self, source_arg, *, batch_size=1024):
            batches = iter(real_adapter.iter_frame_batches(source_arg, batch_size=batch_size))
            yield next(batches)
            raise AdapterError("synthetic stream interruption")

    import lancedb_robotics.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "get_adapter", lambda name: FailingAfterFirstBatchAdapter())
    with pytest.raises(AdapterError, match="synthetic stream interruption"):
        ingest_module.ingest_lerobot(
            lake,
            source,
            batch_size=1,
            compact=False,
            prune_versions=False,
            index_predicates=False,
        )

    failed = list_lerobot_ingest_jobs(lake)
    assert failed[0]["status"] == "failed"
    assert failed[0]["rows_written"]["observations"] == 1
    assert lake.table("observations").count_rows() == 1
    assert lake.table("transform_runs").count_rows() == 0

    monkeypatch.setattr(ingest_module, "get_adapter", lambda name: real_adapter)
    report = ingest_module.ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    observation_ids = [
        row["observation_id"] for row in lake.table("observations").to_arrow().to_pylist()
    ]
    assert report.already_ingested is False
    assert report.rows_added["observations"] == 3
    assert lake.table("observations").count_rows() == 4
    assert len(set(observation_ids)) == 4
    completed = list_lerobot_ingest_jobs(lake, status="completed")
    assert [job["job_id"] for job in completed] == [report.ingest_job_id]
    assert list_lerobot_ingest_jobs(lake, status="failed") == []


def test_ingest_lerobot_hf_download_ledger_records_requested_resolved_revision_and_cache(
    tmp_path, monkeypatch
):
    resolved_revision = "a" * 40
    cache_root = _streaming_lerobot_fixture(tmp_path / resolved_revision)
    lake = Lake.init(tmp_path / "lake")
    calls = []

    import lancedb_robotics.adapters.lerobot_adapter as lerobot_module

    def fake_download(repo_id: str, *, revision: str | None = None) -> Path:
        calls.append({"repo_id": repo_id, "revision": revision})
        return cache_root

    monkeypatch.setattr(lerobot_module, "_download_hf_dataset", fake_download)

    report = ingest_lerobot(
        lake,
        "robotics/corpus@main",
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert calls
    assert calls[0] == {"repo_id": "robotics/corpus", "revision": "main"}
    checkpoints = _durable_lerobot_checkpoint_rows(lake)
    terminal = max(
        checkpoints,
        key=lambda row: (row["observations_written"], row["checkpoint_index"]),
    )
    assert terminal["job_id"] == report.ingest_job_id
    assert terminal["hf_repo_id"] == "robotics/corpus"
    assert terminal["requested_revision"] == "main"
    assert terminal["resolved_revision"] == resolved_revision
    assert Path(terminal["hf_cache_path"]) == cache_root.resolve()
    ledger = json.loads(terminal["hf_download_json"])
    assert ledger["repo_id"] == "robotics/corpus"
    assert ledger["repo_type"] == "dataset"
    assert ledger["requested_revision"] == "main"
    assert ledger["resolved_revision"] == resolved_revision
    assert ledger["cache_path"] == str(cache_root.resolve())
    assert ledger["source_ref"] == f"hf://robotics/corpus@{resolved_revision}"
    assert {item["path"] for item in ledger["manifest_fingerprints"]} == {
        "data/chunk-000/file-000.parquet",
        "data/chunk-001/file-000.parquet",
    }
    source_identity = json.loads(terminal["source_identity_json"])
    assert source_identity["kind"] == "hf-revision"
    assert source_identity["revision"] == resolved_revision


def test_lerobot_ingest_job_api_lists_and_gets_without_observation_scan(tmp_path, monkeypatch):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-api-no-observation-scan")
    lake = Lake.init(tmp_path / "lake")
    report = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    original_table = lake.table

    def table_without_observations(name: str):
        if name == "observations":
            raise AssertionError("job list/get must read the durable checkpoint catalog")
        return original_table(name)

    monkeypatch.setattr(lake, "table", table_without_observations)

    jobs = list_lerobot_ingest_jobs(lake, limit=5)
    assert [job["job_id"] for job in jobs] == [report.ingest_job_id]
    assert jobs[0]["progress"]["rows_written"]["observations"] == 4
    detail = get_lerobot_ingest_job(lake, report.ingest_job_id)
    assert detail["run_id"] == report.run_id
    assert detail["history"][-1]["phase"] in {"completed", "finalized"}


def test_ingest_lerobot_completed_rerun_is_no_dup_for_streaming_fixture(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-rerun")
    lake = Lake.init(tmp_path / "lake")

    first = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )
    observation_ids = sorted(
        row["observation_id"] for row in lake.table("observations").to_arrow().to_pylist()
    )
    second = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert first.already_ingested is False
    assert second.already_ingested is True
    assert second.rows_added["observations"] == 0
    assert (
        sorted(row["observation_id"] for row in lake.table("observations").to_arrow().to_pylist())
        == observation_ids
    )
    assert lake.table("observations").count_rows() == 4


def test_ingest_lerobot_resumes_partial_observation_rows_without_duplicates(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-partial-resume")
    lake = Lake.init(tmp_path / "lake")
    adapter = get_adapter("lerobot")
    dataset = adapter.dataset(source, include_frames=True)
    run_id = f"run-{dataset.source.digest}"
    partial_rows, _episode_state = _lerobot_observation_rows(
        dataset,
        run_id=run_id,
        transform_id=f"tfm-{dataset.source.digest}-ingest",
        created_at=datetime.now(UTC),
    )
    lake.table("observations").add(
        pa.Table.from_pylist(partial_rows[:2], schema=OBSERVATIONS_SCHEMA)
    )

    report = ingest_lerobot(
        lake,
        source,
        batch_size=1,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    observation_ids = [
        row["observation_id"] for row in lake.table("observations").to_arrow().to_pylist()
    ]
    params = _lerobot_ingest_params(lake)
    progress = params["progress"]
    assert report.already_ingested is False
    assert report.rows_added["observations"] == 2
    assert lake.table("observations").count_rows() == 4
    assert len(set(observation_ids)) == 4
    assert progress["existing_rows_before"] == 2
    assert progress["rows_seen"] == 4
    assert progress["rows_written"]["observations"] == 2
    assert progress["rows_skipped_existing"] == 2


def test_ingest_lerobot_v30_round_trips_state_action_task_to_export(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot-v30", version="v3.0")
    lake = Lake.init(tmp_path / "lake")
    ingest_lerobot(lake, source, compact=False, prune_versions=False, index_predicates=False)
    scenario_ids = sorted(
        row["scenario_id"] for row in lake.table("scenarios").to_arrow().to_pylist()
    )
    create_snapshot(
        lake,
        name="lerobot-roundtrip",
        scenario_ids=scenario_ids,
        split_by=SPLIT_BY_SCENARIO,
    )

    out = tmp_path / "roundtrip"
    manifest = export_dataset_snapshot(lake, "lerobot-roundtrip", out_dir=out, fmt="lerobot")

    assert manifest.format == "lerobot"
    assert manifest.episode_count == 2
    assert manifest.step_count == 3
    info = json.loads((out / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    rows = pq.read_table(out / "data/chunk-000/file-000.parquet").to_pylist()
    assert rows[0]["observation.state"] == pytest.approx([1.0, 2.0])
    assert rows[0]["action"] == pytest.approx([0.1, 0.2])
    assert rows[0]["task"] == "pick cube"


def test_unknown_lerobot_codebase_version_is_refused(tmp_path):
    source = _lerobot_fixture(tmp_path / "unknown", version="v3.0")
    info = json.loads((source / "meta/info.json").read_text())
    info["codebase_version"] = "v9.9"
    _write_json(source / "meta/info.json", info)

    with pytest.raises(AdapterError, match="unsupported LeRobot codebase_version"):
        get_adapter("lerobot").inspect(source)


def test_cli_inspect_and_ingest_lerobot(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot-cli", version="v3.0")
    lake_path = tmp_path / "lake"
    Lake.init(lake_path)

    inspected = runner.invoke(app, ["inspect", "lerobot", str(source), "--format", "text"])
    assert inspected.exit_code == 0, inspected.output
    assert "LeRobot v3.0" in inspected.output
    assert "cameras: front" in inspected.output

    ingested = runner.invoke(
        app,
        [
            "ingest",
            "lerobot",
            str(source),
            "--lake",
            str(lake_path),
            "--no-compact",
            "--no-prune-versions",
            "--no-index-predicates",
        ],
    )
    assert ingested.exit_code == 0, ingested.output
    assert "observations +3" in ingested.output
    assert "episodes +2" in ingested.output


def test_cli_inspect_lerobot_process_media_inspection_options(tmp_path):
    source = _lerobot_fixture(tmp_path / "lerobot-cli-process-media", version="v3.0")
    first_video = source / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    second_video = source / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    _write_tiny_mp4(first_video, [b"front-frame-0", b"front-frame-1"], keyframes=[0])
    _write_tiny_mp4(second_video, [b"front-frame-2"], keyframes=[0])

    inspected = runner.invoke(
        app,
        [
            "inspect",
            "lerobot",
            str(source),
            "--media-inspection-workers",
            "1",
            "--media-inspection-timeout-seconds",
            "5",
            "--media-inspection-retries",
            "1",
            "--media-inspection-retry-backoff-seconds",
            "0.5",
            "--media-inspection-retry-policy",
            "exponential-jitter",
            "--media-inspection-execution-mode",
            "process",
            "--format",
            "json",
        ],
    )

    assert inspected.exit_code == 0, inspected.output
    payload = json.loads(inspected.output)
    media = _require_media_inspection(payload)
    assert media["execution_mode"] == "process"
    assert media["retry_count"] == 1
    assert media["retry_backoff_seconds"] == 0.5
    assert media["retry_policy"] == "exponential-jitter"
    assert media["max_workers"] == 1
    assert media["status_counts"] == {"completed": 2}


def test_cli_ingest_lerobot_object_store_source_options(tmp_path, monkeypatch):
    source = _lerobot_fixture(tmp_path / "lerobot-cli-s3", version="v3.0")
    remote_root = "s3://robotics-raw/lerobot-cli-s3"
    _install_fake_lerobot_object_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        expected_options={"anon": "true"},
    )
    lake_path = tmp_path / "lake"
    Lake.init(lake_path)

    inspected = runner.invoke(
        app,
        [
            "inspect",
            "lerobot",
            remote_root,
            "--storage-option",
            "anon=true",
            "--auth-ref",
            "raw-bucket",
            "--format",
            "json",
        ],
    )
    assert inspected.exit_code == 0, inspected.output
    inspect_payload = json.loads(inspected.output)
    assert inspect_payload["source_identity"]["kind"] == "object-store-metadata"
    assert inspect_payload["video_files"][0]["uri"].startswith(remote_root + "/videos/")

    ingested = runner.invoke(
        app,
        [
            "ingest",
            "lerobot",
            remote_root,
            "--lake",
            str(lake_path),
            "--source-storage-option",
            "anon=true",
            "--source-auth-ref",
            "raw-bucket",
            "--no-compact",
            "--no-prune-versions",
            "--no-index-predicates",
        ],
    )

    assert ingested.exit_code == 0, ingested.output
    assert "observations +3" in ingested.output
    lake = Lake.open(lake_path)
    sources = lake.table("integration_sources").to_arrow().to_pylist()
    assert sources[0]["uri"] == remote_root
    assert sources[0]["auth_ref"] == "raw-bucket"
    videos = lake.table("videos").to_arrow().to_pylist()
    assert all(row["raw_uri"].startswith(remote_root + "/videos/") for row in videos)


def test_cli_lerobot_claim_recover_records_auditable_checkpoint(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-cli-claim-recovery")
    lake_path = tmp_path / "lake"
    lake = Lake.init(lake_path)
    now = datetime(2026, 1, 15, 12, tzinfo=UTC)
    job_id = _seed_lerobot_running_claim(
        lake,
        source,
        updated_at=now - timedelta(hours=2),
        claim_expires_at=now - timedelta(minutes=30),
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-claim-recover",
            job_id,
            "--lake",
            str(lake_path),
            "--action",
            "abandon",
            "--new-owner",
            "operator",
            "--stale-after-seconds",
            "900",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["job_id"] == job_id
    assert payload["action"] == "abandon"
    assert payload["status"] == "abandoned"
    assert payload["previous_owner"] == "worker-a"
    assert payload["new_owner"] == "operator"
    detail = get_lerobot_ingest_job(lake, job_id)
    assert detail["status"] == "abandoned"
    assert detail["history"][-1]["progress"]["claim_recovery"]["new_owner"] == "operator"


def test_cli_lerobot_claim_recover_rejects_failed_cas_precondition(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-cli-claim-cas")
    lake_path = tmp_path / "lake"
    lake = Lake.init(lake_path)
    now = datetime(2026, 1, 15, 12, tzinfo=UTC)
    job_id = _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-cli-claim-cas",
        updated_at=now - timedelta(hours=2),
        claim_expires_at=now - timedelta(minutes=30),
        claim_token="token-cli-cas",
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-claim-recover",
            job_id,
            "--lake",
            str(lake_path),
            "--expected-latest-checkpoint-id",
            "not-the-latest",
            "--expected-latest-claim-token",
            "token-cli-cas",
            "--expected-checkpoint-index",
            "0",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "lerobot_claim_precondition_failed"
    assert payload["operation"] == "recover"
    assert payload["job_id"] == job_id
    assert payload["expected_latest_checkpoint_id"] == "not-the-latest"
    assert payload["expected_latest_claim_token"] == "token-cli-cas"
    assert payload["actual_latest_checkpoint_id"] == f"{job_id}:00000000"
    assert payload["actual_latest_claim_token"] == "token-cli-cas"
    assert payload["actual_latest_status"] == "running"
    assert "claim precondition failed" in payload["message"]
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 1


def test_cli_lerobot_claim_watchdog_writes_reports_and_can_fail_on_stale(tmp_path):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-cli-claim-watchdog")
    lake_path = tmp_path / "lake"
    lake = Lake.init(lake_path)
    now = datetime(2026, 1, 15, 12, tzinfo=UTC)
    job_id = _seed_lerobot_running_claim(
        lake,
        source,
        job_id="job-cli-watchdog-stale",
        source_id="src-cli-watchdog",
        updated_at=now - timedelta(hours=2),
        claim_expires_at=now - timedelta(minutes=30),
        claim_owner="worker-cli",
        claim_token="token-cli",
        rows_seen=7,
        observations_written=5,
    )
    json_out = tmp_path / "reports" / "stale-claims.json"
    markdown_out = tmp_path / "reports" / "stale-claims.md"

    result = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-claim-watchdog",
            "--lake",
            str(lake_path),
            "--source-id",
            "src-cli-watchdog",
            "--recovery-action",
            "abandon",
            "--new-owner",
            "operator",
            "--stale-after-seconds",
            "900",
            "--out-json",
            str(json_out),
            "--out-markdown",
            str(markdown_out),
            "--fail-on-stale",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 3, result.output
    payload = json.loads(result.output)
    assert payload["stale_count"] == 1
    assert payload["live_count"] == 0
    assert payload["has_stale"] is True
    finding = payload["stale_claims"][0]
    assert finding["job_id"] == job_id
    assert finding["claim_owner"] == "worker-cli"
    assert finding["rows_seen"] == 7
    assert "lerobot-claim-recover" in finding["suggested_recovery_command"]
    assert "--new-owner operator" in finding["suggested_recovery_command"]
    assert (
        "--expected-latest-checkpoint-id job-cli-watchdog-stale:00000000"
        in finding["suggested_recovery_command"]
    )
    assert "--expected-latest-claim-token token-cli" in finding["suggested_recovery_command"]
    assert "--expected-checkpoint-index 0" in finding["suggested_recovery_command"]
    assert json.loads(json_out.read_text())["stale_claims"][0]["job_id"] == job_id
    markdown = markdown_out.read_text()
    assert "LeRobot Claim Watchdog" in markdown
    assert job_id in markdown
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 1


def test_cli_lerobot_checkpoint_retention_dry_run_and_apply(tmp_path):
    lake_path = tmp_path / "lake"
    lake = Lake.init(lake_path)
    now = datetime(2026, 1, 15, tzinfo=UTC)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-cli-old",
        source_id="src-cli",
        updated_at=now - timedelta(days=90),
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )

    dry_run = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-checkpoint-retention",
            "--lake",
            str(lake_path),
            "--older-than-days",
            "30",
            "--retain-completed-per-source",
            "0",
            "--retain-failed-per-source",
            "0",
            "--format",
            "json",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    dry_payload = json.loads(dry_run.output)
    assert dry_payload["dry_run"] is True
    assert dry_payload["rows_deleted"] == 2
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 3

    applied = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-checkpoint-retention",
            "--lake",
            str(lake_path),
            "--older-than-days",
            "30",
            "--retain-completed-per-source",
            "0",
            "--retain-failed-per-source",
            "0",
            "--apply",
            "--no-compact",
            "--cleanup-older-than-days",
            "-1",
            "--format",
            "json",
        ],
    )
    assert applied.exit_code == 0, applied.output
    payload = json.loads(applied.output)
    assert payload["dry_run"] is False
    assert payload["rows_deleted"] == 2
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 1


def test_cli_lerobot_checkpoint_retention_schedule_config_json(tmp_path):
    lake_path = tmp_path / "lake"
    lake = Lake.init(lake_path)
    now = datetime(2026, 1, 15, tzinfo=UTC)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-cli-schedule-old",
        source_id="src-cli-schedule",
        updated_at=now - timedelta(days=90),
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )
    config = {
        "schedule_id": "cron-nightly",
        "every_minutes": 60,
        "older_than_days": 30,
        "retain_completed_per_source": 0,
        "retain_failed_per_source": 0,
        "dry_run": True,
        "max_rows": 0,
        "max_rows_per_source": 0,
    }

    result = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-checkpoint-retention-schedule",
            "--lake",
            str(lake_path),
            "--config-json",
            json.dumps(config),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schedule_id"] == "cron-nightly"
    assert payload["dry_run"] is True
    assert payload["interval_seconds"] == 3600
    assert payload["retention"]["rows_deleted"] == 2
    assert payload["telemetry"]["per_source"]["src-cli-schedule"]["rows_after"] == 1
    assert {alert["metric"] for alert in payload["alerts"]} == {
        "rows_after",
        "rows_after_per_source",
    }
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 3

    applied = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-checkpoint-retention-schedule",
            "--lake",
            str(lake_path),
            "--config-json",
            json.dumps(config),
            "--apply",
            "--no-compact",
            "--cleanup-older-than-days",
            "-1",
            "--format",
            "json",
        ],
    )
    assert applied.exit_code == 0, applied.output
    applied_payload = json.loads(applied.output)
    assert applied_payload["dry_run"] is False
    assert applied_payload["retention"]["rows_deleted"] == 2
    assert lake.table("lerobot_ingest_checkpoints").count_rows() == 1


def test_cli_lerobot_checkpoint_retention_plan_synthetic_json():
    result = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-checkpoint-retention-plan",
            "--scenario",
            "mid-corpus",
            "--synthetic-sources",
            "2",
            "--synthetic-completed-jobs-per-source",
            "10",
            "--synthetic-failed-jobs-per-source",
            "2",
            "--synthetic-running-jobs-per-source",
            "1",
            "--synthetic-checkpoints-per-job",
            "4",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "synthetic"
    assert payload["scenario"] == "mid-corpus"
    assert payload["recommended_policy"] == "mid-corpus"
    by_name = {policy["name"]: policy for policy in payload["policies"]}
    assert by_name["mid-corpus"]["rows_deleted"] == 30
    assert by_name["audit-window"]["rows_deleted"] == 0


def test_cli_lerobot_claim_recovery_simulate_synthetic_json():
    result = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-claim-recovery-simulate",
            "--scenario",
            "ci",
            "--synthetic-sources",
            "1",
            "--synthetic-completed-jobs-per-source",
            "1",
            "--synthetic-running-jobs-per-source",
            "2",
            "--synthetic-checkpoints-per-job",
            "5",
            "--synthetic-stale-running-fraction",
            "1",
            "--synthetic-missing-lease-fraction",
            "0.5",
            "--source-size-frames",
            "5",
            "--episode-count",
            "3",
            "--camera-count",
            "2",
            "--batch-size",
            "2",
            "--retry-owner-count",
            "2",
            "--seed",
            "11",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "synthetic"
    assert payload["passed"] is True
    assert payload["watchdog"]["stale_count"] == 2
    assert payload["watchdog"]["missing_lease_count"] == 1
    assert payload["recovery"]["cas_conflicts"] == 2
    assert payload["duplicate_protection"]["expected_final_rows"]["videos"] == 6
    assert payload["retention_plan"]["recommended_policy"] == "ci-disposable"
    assert payload["watchdog"]["sample_recovery_commands"]
    assert payload["crash_points"][0]["crash_point"] == "before-claim"


def test_cli_lerobot_checkpoint_hold_create_and_release(tmp_path):
    lake_path = tmp_path / "lake"
    lake = Lake.init(lake_path)
    now = datetime(2026, 1, 15, tzinfo=UTC)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-cli-hold",
        source_id="src-cli",
        updated_at=now - timedelta(days=90),
        phases=[
            ("claimed", "running"),
            ("frame-batch", "running"),
            ("finalized", "completed"),
        ],
    )

    held = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-checkpoint-hold",
            "--lake",
            str(lake_path),
            "--job-id",
            "job-cli-hold",
            "--legal-hold",
            "--no-audit-hold",
            "--owner",
            "legal",
            "--reason",
            "litigation hold",
            "--format",
            "json",
        ],
    )
    assert held.exit_code == 0, held.output
    hold_payload = json.loads(held.output)
    assert hold_payload["active"] is True
    assert hold_payload["reason"] == "litigation hold"
    assert hold_payload["checkpoint_ids"] == [
        "job-cli-hold:00000000",
        "job-cli-hold:00000001",
        "job-cli-hold:00000002",
    ]

    dry_run = apply_lerobot_checkpoint_retention(
        lake,
        older_than=timedelta(days=30),
        retain_completed_per_source=0,
        retain_failed_per_source=0,
        dry_run=True,
        compact=False,
        cleanup_older_than=None,
        retain_versions=None,
        now=now,
    )
    assert dry_run.rows_deleted == 0

    released = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-checkpoint-release-hold",
            hold_payload["hold_id"],
            "--lake",
            str(lake_path),
            "--released-by",
            "legal",
            "--format",
            "json",
        ],
    )
    assert released.exit_code == 0, released.output
    release_payload = json.loads(released.output)
    assert release_payload["active"] is False
    assert release_payload["released_by"] == "legal"

    after_release = apply_lerobot_checkpoint_retention(
        lake,
        older_than=timedelta(days=30),
        retain_completed_per_source=0,
        retain_failed_per_source=0,
        dry_run=True,
        compact=False,
        cleanup_older_than=None,
        retain_versions=None,
        now=now,
    )
    assert after_release.rows_deleted == 2


def test_cli_lerobot_ingest_jobs_list_and_get_without_observation_scan(tmp_path, monkeypatch):
    source = _streaming_lerobot_fixture(tmp_path / "lerobot-cli-jobs")
    lake_path = tmp_path / "lake"
    Lake.init(lake_path)
    ingested = runner.invoke(
        app,
        [
            "ingest",
            "lerobot",
            str(source),
            "--lake",
            str(lake_path),
            "--batch-size",
            "1",
            "--no-compact",
            "--no-prune-versions",
            "--no-index-predicates",
        ],
    )
    assert ingested.exit_code == 0, ingested.output

    original_table = Lake.table

    def table_without_observations(self, name: str):
        if name == "observations":
            raise AssertionError("CLI job list/get must read the durable checkpoint catalog")
        return original_table(self, name)

    monkeypatch.setattr(Lake, "table", table_without_observations)

    listed = runner.invoke(
        app,
        ["ingest", "lerobot-jobs", "--lake", str(lake_path), "--format", "json"],
    )
    assert listed.exit_code == 0, listed.output
    jobs = json.loads(listed.output)
    assert len(jobs) == 1
    assert jobs[0]["adapter"] == "lerobot"
    assert jobs[0]["progress"]["rows_written"]["observations"] == 4

    shown = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-job",
            jobs[0]["job_id"],
            "--lake",
            str(lake_path),
            "--format",
            "json",
        ],
    )
    assert shown.exit_code == 0, shown.output
    detail = json.loads(shown.output)
    assert detail["job_id"] == jobs[0]["job_id"]
    assert detail["history"][-1]["phase"] in {"completed", "finalized"}


def test_lerobot_media_inspection_timeout_recommendations_use_checkpoint_and_transform_telemetry(
    tmp_path,
):
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    checkpoint_media = {
        "video_count": 3,
        "timeout_seconds": 10.0,
        "retry_count": 1,
        "retry_backoff_seconds": 0.25,
        "execution_mode": "thread",
        "total_attempts": 5,
        "total_retries": 2,
        "total_timeouts": 1,
        "status_counts": {"completed": 2, "timeout": 1},
        "videos": [
            {
                "path": "videos/front/episode_000000.mp4",
                "inspection_status": "completed",
                "inspection_duration_ms": 4_000.0,
                "inspection_attempts": 1,
                "inspection_retries": 0,
                "inspection_timeouts": 0,
                "inspection_fingerprint": "fp-front-0",
            },
            {
                "path": "videos/front/episode_000001.mp4",
                "inspection_status": "timeout",
                "inspection_duration_ms": 10_000.0,
                "inspection_attempts": 3,
                "inspection_retries": 2,
                "inspection_timeouts": 1,
                "inspection_fingerprint": "fp-front-1",
            },
        ],
    }
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-timeout",
        source_id="src-timeout",
        updated_at=now,
        phases=[("media-inspection", "running"), ("finalized", "completed")],
        media_inspection=checkpoint_media,
    )
    transform_media = {
        "video_count": 2,
        "timeout_seconds": 20.0,
        "retry_count": 0,
        "execution_mode": "process",
        "total_attempts": 2,
        "total_retries": 0,
        "total_timeouts": 0,
        "status_counts": {"completed": 2},
        "videos": [
            {
                "path": "videos/wrist/episode_000000.mp4",
                "inspection_status": "completed",
                "inspection_duration_ms": 6_000.0,
                "inspection_fingerprint": "fp-wrist-0",
            },
            {
                "path": "videos/wrist/episode_000001.mp4",
                "inspection_status": "completed",
                "inspection_duration_ms": 500.0,
                "inspection_reused": True,
                "inspection_fingerprint": "fp-wrist-1",
            },
        ],
    }
    lake.table("transform_runs").add(
        pa.Table.from_pylist(
            [
                {
                    "transform_id": "tfm-transform-media",
                    "kind": "ingest",
                    "source_id": "src-transform",
                    "input_uris": ["s3://robotics-corpus/lerobot"],
                    "output_tables": [],
                    "params": json.dumps(
                        {
                            "adapter": "lerobot",
                            "run_id": "run-transform-media",
                            "media_inspection": transform_media,
                        },
                        sort_keys=True,
                    ),
                    "status": "completed",
                    "started_at": now - timedelta(minutes=2),
                    "finished_at": now,
                    "created_by": "test",
                    "created_at": now,
                }
            ],
            schema=TRANSFORM_RUNS_SCHEMA,
        )
    )

    report = recommend_lerobot_media_inspection_timeouts(
        lake,
        max_timeout_seconds=60.0,
    )

    assert report["source_counts"] == {
        "reports": 2,
        "checkpoints": 1,
        "completed_transforms": 1,
    }
    telemetry = report["telemetry"]
    assert telemetry["total_timeouts"] == 1
    assert telemetry["total_retries"] == 2
    assert telemetry["duration_ms"]["count"] == 3
    assert telemetry["duration_reused_excluded_count"] == 1
    assert telemetry["observed_timeout_seconds"] == [10.0, 20.0]
    assert report["recommendation"]["timeout_seconds"] == 40
    assert report["recommendation"]["retry_count"] == 2
    assert report["recommendation"]["basis"]["total_timeouts"] == 1
    assert report["recommendation"]["basis"]["timeout_video_count"] == 1
    assert "timeout-policy-too-aggressive" in report["recommendation"]["flags"]
    assert {
        (
            group["selector"]["storage_tier"],
            group["selector"]["provider"],
        )
        for group in report["groups"]
    } == {("huggingface", "huggingface"), ("object-store", "s3")}

    s3_report = recommend_lerobot_media_inspection_timeouts(lake, provider="s3")
    assert s3_report["source_counts"]["reports"] == 1
    assert s3_report["groups"][0]["selector"]["storage_tier"] == "object-store"


def test_cli_lerobot_media_inspection_timeout_plan_accepts_checkpoint_rows_json(tmp_path):
    lake = Lake.init(tmp_path / "lake")
    now = datetime(2026, 1, 15, tzinfo=UTC)
    _add_lerobot_checkpoint_rows(
        lake,
        job_id="job-timeout",
        source_id="src-timeout",
        updated_at=now,
        phases=[("media-inspection", "running"), ("finalized", "completed")],
        media_inspection={
            "video_count": 2,
            "timeout_seconds": 10.0,
            "retry_count": 1,
            "total_attempts": 4,
            "total_retries": 1,
            "total_timeouts": 1,
            "status_counts": {"completed": 1, "timeout": 1},
            "videos": [
                {
                    "path": "videos/front/episode_000000.mp4",
                    "inspection_status": "completed",
                    "inspection_duration_ms": 3_000.0,
                    "inspection_fingerprint": "fp-front-0",
                },
                {
                    "path": "videos/front/episode_000001.mp4",
                    "inspection_status": "timeout",
                    "inspection_duration_ms": 10_000.0,
                    "inspection_retries": 1,
                    "inspection_timeouts": 1,
                    "inspection_fingerprint": "fp-front-1",
                },
            ],
        },
    )
    rows_path = tmp_path / "checkpoint-rows.json"
    rows_path.write_text(
        json.dumps({"checkpoint_rows": _durable_lerobot_checkpoint_rows(lake)}, default=str),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            "lerobot-media-inspection-timeout-plan",
            "--checkpoint-rows-json",
            str(rows_path),
            "--max-timeout-seconds",
            "60",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["lake_uri"] is None
    assert payload["source_counts"]["checkpoints"] == 1
    assert payload["telemetry"]["total_timeouts"] == 1
    assert payload["recommendation"]["timeout_seconds"] == 20
    assert payload["recommendation"]["apply_args"][:2] == [
        "--media-inspection-timeout-seconds",
        "20",
    ]
