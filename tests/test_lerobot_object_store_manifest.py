import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from lancedb_robotics.adapters import get_adapter
from lancedb_robotics.cli import app
from lancedb_robotics.ingest import ingest_lerobot
from lancedb_robotics.lake import Lake

runner = CliRunner()


def test_lerobot_object_store_manifest_cache_reuses_paginated_listing_for_ingest(
    tmp_path, monkeypatch
):
    source = _multi_file_lerobot_fixture(tmp_path / "lerobot-paged", data_file_count=5)
    remote_root = "s3://robotics-raw/lerobot-paged"
    calls, state = _install_paginated_lerobot_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        page_size=3,
    )
    cache_path = tmp_path / "manifest-cache.json"

    inspect_report = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        auth_ref="raw-bucket",
        source_manifest_cache=cache_path,
    )

    assert inspect_report["frame_count"] == 5
    assert sorted(inspect_report["data_files"]) == [
        f"data/chunk-{index:03d}/file-000.parquet" for index in range(5)
    ]
    manifest = inspect_report["object_store_manifest"]
    assert manifest["metrics"]["cache_status"] == "miss"
    assert manifest["metrics"]["listed_object_count"] >= 9
    assert manifest["metrics"]["list_calls"] == calls["find_pages"]
    first_listing_calls = calls["find_pages"]
    assert first_listing_calls > 1
    assert cache_path.exists()

    lake = Lake.init(tmp_path / "lake")
    ingest_report = ingest_lerobot(
        lake,
        remote_root,
        storage_options={"anon": True},
        auth_ref="raw-bucket",
        source_manifest_cache=cache_path,
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert ingest_report.rows_added["observations"] == 5
    assert calls["find_pages"] == first_listing_calls
    assert calls["glob"] == 0
    assert calls["info"] > 0
    assert state["metadata_generation"] == 0


def test_lerobot_object_store_manifest_cache_invalidates_on_metadata_change(
    tmp_path, monkeypatch
):
    source = _multi_file_lerobot_fixture(tmp_path / "lerobot-invalidates", data_file_count=3)
    remote_root = "s3://robotics-raw/lerobot-invalidates"
    calls, state = _install_paginated_lerobot_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        page_size=4,
    )
    cache_path = tmp_path / "manifest-cache.json"

    first = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        source_manifest_cache=cache_path,
    )
    first_listing_calls = calls["find_pages"]
    assert first["object_store_manifest"]["metrics"]["cache_status"] == "miss"

    second = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        source_manifest_cache=cache_path,
    )
    assert calls["find_pages"] == first_listing_calls
    assert second["object_store_manifest"]["metrics"]["cache_status"] == "hit"

    state["metadata_generation"] += 1
    third = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        source_manifest_cache=cache_path,
    )

    assert calls["find_pages"] > first_listing_calls
    assert third["object_store_manifest"]["metrics"]["cache_status"] == "invalidated"
    assert third["frame_count"] == 3


def test_cli_lerobot_manifest_cache_for_object_store_inspect(tmp_path, monkeypatch):
    source = _multi_file_lerobot_fixture(tmp_path / "lerobot-cli-cache", data_file_count=2)
    remote_root = "s3://robotics-raw/lerobot-cli-cache"
    _install_paginated_lerobot_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        page_size=2,
    )
    cache_path = tmp_path / "cli-manifest-cache.json"

    result = runner.invoke(
        app,
        [
            "inspect",
            "lerobot",
            remote_root,
            "--storage-option",
            "anon=true",
            "--source-manifest-cache",
            str(cache_path),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["object_store_manifest"]["metrics"]["cache_status"] == "miss"
    assert payload["object_store_manifest"]["object_count"] >= 6
    assert cache_path.exists()


def test_lerobot_object_store_metadata_only_preserves_default_identity(
    tmp_path, monkeypatch
):
    source = _multi_file_lerobot_fixture(tmp_path / "lerobot-default-validation", data_file_count=2)
    remote_root = "s3://robotics-raw/lerobot-default-validation"
    _install_paginated_lerobot_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        page_size=3,
    )

    default = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        inspect_videos=False,
    )
    explicit = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        inspect_videos=False,
        object_store_validation_policy="metadata-only",
    )

    assert default["source_identity"]["kind"] == "object-store-metadata"
    assert default["source_identity"]["checksum"] == explicit["source_identity"]["checksum"]
    assert default["source_identity"]["digest"] == explicit["source_identity"]["digest"]
    validation = default["object_store_validation"]
    assert validation["policy"] == "metadata-only"
    assert validation["hashed_bytes"] == 0
    assert validation["warnings"] == []


def test_lerobot_object_store_sampled_validation_detects_content_drift(
    tmp_path, monkeypatch
):
    source = _multi_file_lerobot_fixture(tmp_path / "lerobot-sampled-validation", data_file_count=2)
    remote_root = "s3://robotics-raw/lerobot-sampled-validation"
    calls, _ = _install_paginated_lerobot_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        page_size=3,
    )
    cache_path = tmp_path / "manifest-cache.json"

    first = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        inspect_videos=False,
        source_manifest_cache=cache_path,
        object_store_validation_policy="sampled-validation",
        object_store_validation_sample_count=100,
        object_store_validation_sample_bytes=64,
    )
    first_listing_calls = calls["find_pages"]
    first_validation = first["object_store_validation"]
    assert first["source_identity"]["kind"] == "object-store-sampled-validation"
    assert first_validation["sample_count"] == first_validation["object_count"]
    assert first_validation["evidence_digest"]
    video_rel = "videos/chunk-000/observation.images.front/episode_000000.mp4"
    assert video_rel in {sample["relative_path"] for sample in first_validation["samples"]}

    (source / video_rel).write_bytes(b"FAKE-mp4")
    second = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        inspect_videos=False,
        source_manifest_cache=cache_path,
        object_store_validation_policy="sampled-validation",
        object_store_validation_sample_count=100,
        object_store_validation_sample_bytes=64,
    )

    assert calls["find_pages"] == first_listing_calls
    assert second["object_store_manifest"]["metrics"]["cache_status"] == "hit"
    assert second["object_store_validation"]["evidence_digest"] != first_validation["evidence_digest"]
    assert second["source_identity"]["digest"] != first["source_identity"]["digest"]


def test_lerobot_object_store_strict_content_hash_records_full_object_evidence(
    tmp_path, monkeypatch
):
    source = _multi_file_lerobot_fixture(tmp_path / "lerobot-strict-validation", data_file_count=1)
    remote_root = "s3://robotics-raw/lerobot-strict-validation"
    _install_paginated_lerobot_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        page_size=4,
    )

    report = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        inspect_videos=False,
        object_store_validation_policy="strict-content-hash",
        object_store_validation_strict_max_bytes=1_000_000,
    )

    validation = report["object_store_validation"]
    assert report["source_identity"]["kind"] == "object-store-strict-content-hash"
    assert validation["policy"] == "strict-content-hash"
    assert validation["assurance"] == "full-content"
    assert validation["sample_count"] == validation["object_count"]
    assert validation["hashed_bytes"] == sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    assert {sample["mode"] for sample in validation["samples"]} == {"full-object"}


def test_lerobot_object_store_validation_warns_for_weak_provider_metadata(
    tmp_path, monkeypatch
):
    source = _multi_file_lerobot_fixture(tmp_path / "lerobot-weak-metadata", data_file_count=1)
    remote_root = "s3://robotics-raw/lerobot-weak-metadata"
    _install_paginated_lerobot_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        page_size=4,
        weak_metadata=True,
    )

    report = get_adapter("lerobot").inspect(
        remote_root,
        storage_options={"anon": True},
        inspect_videos=False,
    )

    warnings = report["object_store_validation"]["warnings"]
    assert warnings
    assert warnings[0]["code"] == "weak-provider-metadata"
    assert warnings[0]["object_count"] == report["object_store_validation"]["object_count"]


def test_cli_lerobot_sampled_validation_for_object_store_inspect(tmp_path, monkeypatch):
    source = _multi_file_lerobot_fixture(tmp_path / "lerobot-cli-validation", data_file_count=1)
    remote_root = "s3://robotics-raw/lerobot-cli-validation"
    _install_paginated_lerobot_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        page_size=4,
        weak_metadata=True,
    )

    result = runner.invoke(
        app,
        [
            "inspect",
            "lerobot",
            remote_root,
            "--storage-option",
            "anon=true",
            "--source-validation-policy",
            "sampled-validation",
            "--source-validation-sample-count",
            "2",
            "--source-validation-sample-bytes",
            "16",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    validation = payload["object_store_validation"]
    assert validation["policy"] == "sampled-validation"
    assert validation["sample_count"] == 2
    assert validation["hashed_bytes"] <= 32
    assert validation["warnings"][0]["code"] == "weak-provider-metadata"


def test_ingest_lerobot_records_object_store_validation_provenance(
    tmp_path, monkeypatch
):
    source = _multi_file_lerobot_fixture(tmp_path / "lerobot-ingest-validation", data_file_count=2)
    remote_root = "s3://robotics-raw/lerobot-ingest-validation"
    _install_paginated_lerobot_store(
        monkeypatch,
        remote_root=remote_root,
        local_root=source,
        page_size=3,
    )
    lake = Lake.init(tmp_path / "lake")

    report = ingest_lerobot(
        lake,
        remote_root,
        storage_options={"anon": True},
        compact=False,
        prune_versions=False,
        index_predicates=False,
        object_store_validation_policy="sampled-validation",
        object_store_validation_sample_count=100,
        object_store_validation_sample_bytes=64,
    )

    assert report.rows_added["observations"] == 2
    source_row = lake.table("integration_sources").to_arrow().to_pylist()[0]
    source_metadata = {item["key"]: item["value"] for item in source_row["metadata"]}
    assert source_metadata["source_identity.validation_policy"] == "sampled-validation"
    assert source_metadata["source_identity.assurance"] == "sampled-content"
    checkpoint = lake.table("lerobot_ingest_checkpoints").to_arrow().to_pylist()[-1]
    checkpoint_identity = json.loads(checkpoint["source_identity_json"])
    assert checkpoint_identity["object_store_validation"]["policy"] == "sampled-validation"
    transform_rows = lake.table("transform_runs").to_arrow().to_pylist()
    ingest_params = json.loads(
        next(row for row in transform_rows if row["kind"] == "ingest")["params"]
    )
    assert ingest_params["source_identity"]["object_store_validation"]["policy"] == (
        "sampled-validation"
    )


def _multi_file_lerobot_fixture(root: Path, *, data_file_count: int) -> Path:
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
            "total_frames": data_file_count,
            "total_tasks": 1,
            "chunks_size": 1,
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
                "length": data_file_count,
                "dataset_from_index": 0,
                "dataset_to_index": data_file_count,
                "split": "train",
            }
        ],
    )
    video = root / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake-mp4")
    for index in range(data_file_count):
        _write_parquet(
            root / f"data/chunk-{index:03d}/file-000.parquet",
            [
                {
                    "index": index,
                    "episode_index": 0,
                    "frame_index": index,
                    "timestamp": index / 10.0,
                    "task_index": 0,
                    "task": "pick cube",
                    "observation.state": [float(index), float(index + 1)],
                    "action": [0.1, 0.2],
                    "observation.images.front": (
                        "videos/chunk-000/observation.images.front/episode_000000.mp4"
                    ),
                }
            ],
        )
    return root


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression=None)


def _install_paginated_lerobot_store(
    monkeypatch,
    *,
    remote_root: str,
    local_root: Path,
    page_size: int,
    weak_metadata: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    calls = {"find_pages": 0, "glob": 0, "info": 0, "open": 0}
    state = {"metadata_generation": 0}
    parsed = remote_root.removeprefix("s3://").rstrip("/")

    def to_local(uri_or_path: str) -> Path:
        value = str(uri_or_path)
        if value.startswith("s3://"):
            value = value.removeprefix("s3://")
        rel = value.removeprefix(parsed).lstrip("/")
        return local_root / rel

    def provider_path(path: Path) -> str:
        return f"{parsed}/{path.relative_to(local_root).as_posix()}"

    class FakeFs:
        def find_pages(self, base_path: str, *, detail: bool = True):
            assert detail is True
            paths = sorted(path for path in local_root.rglob("*") if path.is_file())
            for offset in range(0, len(paths), page_size):
                calls["find_pages"] += 1
                yield {
                    provider_path(path): self.info(provider_path(path))
                    for path in paths[offset : offset + page_size]
                }

        def glob(self, pattern: str):
            calls["glob"] += 1
            rel_pattern = pattern.removeprefix(parsed).lstrip("/")
            return [
                provider_path(path)
                for path in sorted(local_root.glob(rel_pattern))
                if path.is_file()
            ]

        def info(self, path: str):
            calls["info"] += 1
            local = to_local(path)
            if not local.exists():
                raise FileNotFoundError(path)
            stat = local.stat()
            rel = local.relative_to(local_root).as_posix()
            payload = {
                "name": path,
                "type": "file",
                "size": int(stat.st_size),
                "last_modified": "2026-01-15T12:00:00+00:00",
            }
            if not weak_metadata:
                payload.update(
                    {
                        "etag": f"etag-{rel}-{state['metadata_generation']}",
                        "version_id": f"version-{rel}-{state['metadata_generation']}",
                    }
                )
            return payload

    def fake_open(uri: str, mode: str = "rb", **kwargs):
        assert kwargs.get("anon") in {True, "true"}
        calls["open"] += 1
        return to_local(uri).open(mode)

    def fake_url_to_fs(uri: str, **kwargs):
        assert kwargs.get("anon") in {True, "true"}
        return FakeFs(), str(uri).removeprefix("s3://").rstrip("/")

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        SimpleNamespace(open=fake_open, core=SimpleNamespace(url_to_fs=fake_url_to_fs)),
    )
    return calls, state
