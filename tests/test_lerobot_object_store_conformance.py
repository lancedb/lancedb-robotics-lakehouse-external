import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.lerobot_object_store_conformance import (
    lerobot_object_store_conformance,
)

runner = CliRunner()


def test_lerobot_object_store_conformance_reports_s3_and_gcs_metadata(
    tmp_path, monkeypatch
):
    s3_source = _mini_lerobot_fixture(tmp_path / "s3-source")
    gs_source = _mini_lerobot_fixture(tmp_path / "gcs-source")
    s3_root = "s3://robotics-raw/lerobot-s3"
    gs_root = "gs://robotics-raw/lerobot-gcs"
    calls = _install_fake_lerobot_object_stores(
        monkeypatch,
        {
            s3_root: (s3_source, "s3"),
            gs_root: (gs_source, "gcs"),
        },
    )
    monkeypatch.setenv(
        "LANCEDB_ROBOTICS_AUTH_RAW_BUCKET_STORAGE_OPTIONS_JSON",
        '{"token": "env-secret", "project": "robotics"}',
    )

    report = lerobot_object_store_conformance(
        roots=[s3_root, gs_root],
        storage_options={"anon": "true", "secret": "top-secret"},
        auth_ref="raw-bucket",
        include_provider_default=False,
    )
    payload = report.to_params()

    assert payload["summary"] == {"passed": 4, "failed": 0, "skipped": 0, "total": 4}
    assert "top-secret" not in json.dumps(payload)
    assert "env-secret" not in json.dumps(payload)
    s3_explicit = _case(payload, "s3", "explicit-options")
    gs_explicit = _case(payload, "gs", "explicit-options")
    assert {"etag", "version_id", "size", "last_modified"}.issubset(
        set(s3_explicit["metadata_fields"])
    )
    assert "generation" in gs_explicit["metadata_fields"]
    assert _operation(s3_explicit, "lerobot_inspect")["details"]["source_identity_kind"] == (
        "object-store-metadata"
    )
    assert _operation(s3_explicit, "lerobot_ingest_stream_preflight")["details"][
        "row_count"
    ] == 1
    assert any(call["kwargs"].get("secret") == "top-secret" for call in calls["opened"])
    assert any(call["kwargs"].get("token") == "env-secret" for call in calls["opened"])


def test_cli_lerobot_object_store_conformance_json(tmp_path, monkeypatch):
    source = _mini_lerobot_fixture(tmp_path / "cli-source")
    remote_root = "s3://robotics-raw/lerobot-cli-conformance"
    _install_fake_lerobot_object_stores(monkeypatch, {remote_root: (source, "s3")})

    result = runner.invoke(
        app,
        [
            "inspect",
            "lerobot-object-store-conformance",
            "--root",
            remote_root,
            "--storage-option",
            "anon=true",
            "--no-provider-default",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"].endswith("provider-conformance/v1")
    assert payload["summary"] == {"failed": 0, "passed": 1, "skipped": 1, "total": 2}
    assert _case(payload, "s3", "explicit-options")["status"] == "passed"
    assert _case(payload, "s3", "auth-ref-env")["status"] == "skipped"


def test_lerobot_object_store_conformance_reports_missing_dependency(monkeypatch):
    def fake_open(*args, **kwargs):
        raise ModuleNotFoundError("s3fs")

    def fake_url_to_fs(*args, **kwargs):
        raise ModuleNotFoundError("s3fs")

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        SimpleNamespace(open=fake_open, core=SimpleNamespace(url_to_fs=fake_url_to_fs)),
    )

    report = lerobot_object_store_conformance(
        roots=["s3://robotics-raw/missing-sdk"],
        storage_options={},
        include_provider_default=False,
    )
    payload = report.to_params()

    failed = _case(payload, "s3", "explicit-options")
    assert failed["status"] == "failed"
    assert any(
        operation["error_class"] == "StorageConfigError" and "install s3fs" in operation["message"]
        for operation in failed["operations"]
    )


def test_lerobot_object_store_conformance_reports_unauthorized_prefix(monkeypatch):
    def fake_open(*args, **kwargs):
        raise PermissionError("access denied")

    class FakeFs:
        def glob(self, pattern):
            raise PermissionError("access denied")

        def info(self, path):
            raise PermissionError("access denied")

    def fake_url_to_fs(*args, **kwargs):
        return FakeFs(), "robotics-raw/denied"

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        SimpleNamespace(open=fake_open, core=SimpleNamespace(url_to_fs=fake_url_to_fs)),
    )

    report = lerobot_object_store_conformance(
        roots=["s3://robotics-raw/denied"],
        storage_options={},
        include_provider_default=False,
    )
    payload = report.to_params()

    failed = _case(payload, "s3", "explicit-options")
    assert failed["status"] == "failed"
    assert any(
        operation["error_class"] == "StorageConfigError"
        and "cannot stat s3://robotics-raw/denied/meta/info.json" in operation["message"]
        for operation in failed["operations"]
    )


def _case(payload: dict, scheme: str, auth_mode: str) -> dict:
    return next(
        case
        for case in payload["cases"]
        if case["scheme"] == scheme and case["auth_mode"] == auth_mode
    )


def _operation(case: dict, name: str) -> dict:
    return next(operation for operation in case["operations"] if operation["name"] == name)


def _mini_lerobot_fixture(root: Path) -> Path:
    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
        "observation.images.front": {"dtype": "video", "shape": [3, 0, 0]},
    }
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "fps": 10,
            "robot_type": "aloha",
            "features": features,
            "total_episodes": 1,
            "total_frames": 2,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
        },
    )
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "pick cube"}])
    _write_parquet(
        root / "meta/episodes/chunk-000/file-000.parquet",
        [
            {
                "episode_index": 0,
                "episode_id": "episode-000000",
                "scenario_id": "lerobot-episode-000000",
                "tasks": ["pick cube"],
                "length": 2,
                "dataset_from_index": 0,
                "dataset_to_index": 2,
                "split": "train",
            }
        ],
    )
    _write_parquet(
        root / "data/chunk-000/file-000.parquet",
        [
            {
                "index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
                "task_index": 0,
                "task": "pick cube",
                "observation.state": [1.0, 2.0],
                "action": [0.1, 0.2],
                "observation.images.front": (
                    "videos/chunk-000/observation.images.front/episode_000000.mp4"
                ),
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
                "observation.images.front": (
                    "videos/chunk-000/observation.images.front/episode_000000.mp4"
                ),
            },
        ],
    )
    video = root / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake-mp4")
    return root


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression=None)


def _install_fake_lerobot_object_stores(
    monkeypatch,
    providers: dict[str, tuple[Path, str]],
) -> dict[str, list[dict]]:
    calls: dict[str, list[dict]] = {"opened": [], "listed": [], "stated": []}
    normalized = {
        remote_root.rstrip("/"): (local_root, metadata_style)
        for remote_root, (local_root, metadata_style) in providers.items()
    }

    def match(value: str) -> tuple[str, Path, str, str]:
        text = str(value)
        for remote_root, (local_root, metadata_style) in sorted(
            normalized.items(), key=lambda item: len(item[0]), reverse=True
        ):
            scheme, remote_path = remote_root.split("://", 1)
            candidate = text.removeprefix(f"{scheme}://")
            if candidate == remote_path or candidate.startswith(remote_path + "/"):
                rel = candidate.removeprefix(remote_path).lstrip("/")
                return remote_path, local_root, metadata_style, rel
        raise FileNotFoundError(value)

    def to_local(value: str) -> Path:
        _, local_root, _, rel = match(value)
        return local_root / rel

    class FakeFs:
        def glob(self, pattern: str):
            calls["listed"].append({"pattern": pattern})
            remote_path, local_root, _, _ = match(pattern.split("*", 1)[0].rstrip("/"))
            rel_pattern = pattern.removeprefix(remote_path).lstrip("/")
            return [
                f"{remote_path}/{path.relative_to(local_root).as_posix()}"
                for path in local_root.glob(rel_pattern)
            ]

        def info(self, path: str):
            calls["stated"].append({"path": path})
            remote_path, local_root, metadata_style, rel = match(path)
            local = local_root / rel
            if not local.exists():
                raise FileNotFoundError(path)
            stat = local.stat()
            info = {
                "name": f"{remote_path}/{rel}",
                "type": "directory" if local.is_dir() else "file",
            }
            if metadata_style == "s3":
                info.update(
                    {
                        "Size": int(stat.st_size),
                        "ETag": f"etag-{rel}",
                        "VersionId": f"version-{rel}",
                        "LastModified": "2026-01-15T12:00:00+00:00",
                    }
                )
            else:
                info.update(
                    {
                        "size": int(stat.st_size),
                        "etag": f"gcs-etag-{rel}",
                        "Generation": f"generation-{rel}",
                        "LastModified": "2026-01-15T12:00:00+00:00",
                    }
                )
            return info

    def fake_open(uri: str, mode: str = "rb", **kwargs):
        calls["opened"].append({"uri": uri, "kwargs": dict(kwargs)})
        return to_local(uri).open(mode)

    def fake_url_to_fs(uri: str, **kwargs):
        scheme, path = str(uri).split("://", 1)
        assert scheme in {"s3", "gs"}
        return FakeFs(), path.rstrip("/")

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        SimpleNamespace(open=fake_open, core=SimpleNamespace(url_to_fs=fake_url_to_fs)),
    )
    return calls
