"""Object-store projection accounting reconciliation (backlog 0144)."""

import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pyarrow as pa
import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.dataset_export import (
    DATASET_EXPORT_MANIFEST_FILENAME,
    export_dataset_snapshot,
)
from lancedb_robotics.export_reconciliation import (
    ExportReconciliationError,
    ExportTarget,
    classify_relative_path,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.projections import (
    PROJECTION_MANIFEST_FILENAME,
    export_projection,
)
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA, SCENARIOS_SCHEMA

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
_CAMERA_FRAME_BYTES = b"\xff\xd8\xff" + b"\x2a" * 3000


def _write_camera_mcap(path, frame: bytes) -> None:
    """A minimal real MCAP with one cbor `/camera/front` message.

    The LeRobot exporter now re-decodes real camera bytes from each
    observation's original source (`payload_blob` is the still-wire-encoded
    envelope, not a bare image — see `lancedb_robotics.adapters.decoders`),
    so a fixture claiming a camera payload needs a real, resolvable
    `raw_uri` to point at. cbor is self-describing (no schema definition
    needed to decode), so this is a real, exercisable decode without a full
    ROS message-definition fixture. The frame is large enough to cross
    `DEFAULT_BLOB_THRESHOLD` so the decoder hoists it as a blob field rather
    than base64-inlining it, matching how a real compressed image behaves.
    """
    import cbor2
    from mcap.writer import CompressionType, Writer

    with open(path, "wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start(profile="", library="lancedb-robotics-test")
        schema_id = writer.register_schema(
            name="test.CompressedImage",
            encoding="cbor",
            data=cbor2.dumps({"fields": ["format", "data"]}),
        )
        channel_id = writer.register_channel(
            topic="/camera/front", message_encoding="cbor", schema_id=schema_id
        )
        writer.add_message(
            channel_id=channel_id,
            log_time=1_000,
            publish_time=1_000,
            data=cbor2.dumps({"format": "jpeg", "data": frame}),
        )
        writer.finish()


def _snapshot_lake(path):
    """A minimal lake with one camera payload observation and a snapshot."""
    mcap_path = path.parent / "camera.mcap"
    _write_camera_mcap(mcap_path, _CAMERA_FRAME_BYTES)
    raw_uri = str(mcap_path)
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-x",
                    "run_kind": "demo",
                    "source": "synthetic",
                    "source_id": "src-x",
                    "raw_uri": "memory://run-x",
                    "robot_id": "robot-1",
                    "site_id": "lab",
                    "task_id": "pick the cube",
                    "start_time_ns": 1_000,
                    "end_time_ns": 2_000,
                    "duration_ns": 1_000,
                    "software_version": "test",
                    "hardware_version": "test",
                    "calibration_version": "test",
                    "model_version": "",
                    "metadata": [],
                    "quality_flags": [],
                    "transform_id": "tfm-source",
                    "created_at": now,
                }
            ],
            schema=RUNS_SCHEMA,
        )
    )
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                {
                    "observation_id": "obs-camera-0",
                    "run_id": "run-x",
                    "timestamp_ns": 1_000,
                    "sensor_id": "camera_front",
                    "topic": "/camera/front",
                    "modality": "image",
                    "raw_uri": raw_uri,
                    "raw_channel": "/camera/front",
                    "raw_log_time_ns": 1_000,
                    "raw_sequence": 0,
                    "payload_json": None,
                    "payload_blob": b"frame-bytes-0",
                    "message_encoding": "jpeg",
                    "schema_encoding": "jpeg",
                    "decode_status": "decoded",
                    "decode_error": "",
                    "state_vector": [1.0, 2.0],
                    "action_vector": [0.25, -0.5],
                    "caption": "reach toward the cube",
                    "quality_flags": [],
                    "transform_id": "tfm-ingest",
                    "created_at": now,
                },
                {
                    "observation_id": "obs-state-1",
                    "run_id": "run-x",
                    "timestamp_ns": 2_000,
                    "sensor_id": "joint_state",
                    "topic": "/joint_states",
                    "modality": "state",
                    "raw_uri": "memory://run-x",
                    "raw_channel": "/joint_states",
                    "raw_log_time_ns": 2_000,
                    "raw_sequence": 1,
                    "payload_json": "{\"joint\":1}",
                    "payload_blob": None,
                    "message_encoding": "json",
                    "schema_encoding": "json",
                    "decode_status": "decoded",
                    "decode_error": "",
                    "state_vector": [1.5, 2.5],
                    "action_vector": [0.5, -0.25],
                    "caption": "close the gripper",
                    "quality_flags": [],
                    "transform_id": "tfm-ingest",
                    "created_at": now,
                },
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-a",
                    "run_id": "run-x",
                    "start_time_ns": 1_000,
                    "end_time_ns": 1_000,
                    "window_ns": 0,
                    "is_partial": False,
                    "topics": ["/camera/front"],
                    "observation_ids": ["obs-camera-0"],
                    "observation_count": 1,
                    "scenario_type": "demo",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["camera"],
                    "summary": "pick the cube",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
                {
                    "scenario_id": "scn-b",
                    "run_id": "run-x",
                    "start_time_ns": 2_000,
                    "end_time_ns": 2_000,
                    "window_ns": 0,
                    "is_partial": False,
                    "topics": ["/joint_states"],
                    "observation_ids": ["obs-state-1"],
                    "observation_count": 1,
                    "scenario_type": "demo",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["state"],
                    "summary": "pick the cube",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    create_snapshot(
        lake,
        name="demo-v1",
        scenario_ids=["scn-a", "scn-b"],
        split_by="scenario",
    )
    return lake


@pytest.fixture
def lake(tmp_path):
    return _snapshot_lake(tmp_path / "robot.lance")


# --------------------------------------------------------------------------- #
# Fake object store (fsspec seam)
# --------------------------------------------------------------------------- #
class _FakeWriteStream:
    def __init__(self, store, key):
        self._store = store
        self._key = key
        self._buf = bytearray()

    def write(self, data):
        self._buf += bytes(data)
        return len(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._store._commit(self._key, bytes(self._buf))
        return False


class _FakeReadStream:
    def __init__(self, data):
        self._io = io.BytesIO(data)

    def __enter__(self):
        return self._io

    def __exit__(self, *exc):
        self._io.close()
        return False


class _FakeFs:
    def __init__(self, store):
        self._store = store

    def info(self, path):
        key = self._store._key(path)
        if key not in self._store.objects:
            raise FileNotFoundError(path)
        data = self._store.objects[key]
        return {
            "name": path,
            "size": len(data),
            "type": "file",
            "ETag": self._store.etags[key],
            "VersionId": "v1",
        }

    def exists(self, path):
        return self._store._key(path) in self._store.objects


class FakeObjectStore:
    """In-memory object store mirroring the fsspec surface the code uses."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.writes: list[str] = []
        self.drop_suffix: str | None = None
        self.truncate_suffix: str | None = None

    def _key(self, uri):
        text = str(uri)
        if "://" in text:
            text = text.split("://", 1)[1]
        return text.strip("/")

    def _commit(self, key, data):
        if self.drop_suffix and key.endswith(self.drop_suffix):
            return
        if self.truncate_suffix and key.endswith(self.truncate_suffix):
            data = data[:-1]
        self.objects[key] = data
        self.etags[key] = '"' + hashlib.md5(data).hexdigest() + '"'
        self.writes.append(key)

    def open(self, uri, mode="rb", **kwargs):
        key = self._key(uri)
        if "w" in mode:
            return _FakeWriteStream(self, key)
        if key not in self.objects:
            raise FileNotFoundError(uri)
        return _FakeReadStream(self.objects[key])

    def url_to_fs(self, uri, **kwargs):
        return _FakeFs(self), self._key(uri)

    def has(self, suffix):
        return any(key.endswith(suffix) for key in self.objects)


def _install_fake_store(monkeypatch):
    store = FakeObjectStore()
    fake = SimpleNamespace(
        open=store.open,
        core=SimpleNamespace(url_to_fs=store.url_to_fs),
    )
    monkeypatch.setitem(sys.modules, "fsspec", fake)
    return store


def _materialization_rows(lake, snapshot_name="demo-v1"):
    return [
        row
        for row in lake.table("curation_materializations").to_arrow().to_pylist()
        if row["snapshot_name"] == snapshot_name
    ]


# --------------------------------------------------------------------------- #
# Unit tests: classification + ExportTarget
# --------------------------------------------------------------------------- #
def test_classify_relative_path_separates_payload_metadata_and_containers():
    assert classify_relative_path("data/chunk-000/file-000.parquet") == (
        "metadata",
        "parquet",
        "none",
    )
    assert classify_relative_path("meta/info.json") == ("metadata", "json", "none")
    assert classify_relative_path("meta/episodes.jsonl") == ("metadata", "jsonl", "none")
    assert classify_relative_path("images/cam/episode_000000/frame_000000.bin") == (
        "payload",
        "image-bytes",
        "none",
    )
    assert classify_relative_path("shards/shard-000000.tar") == ("mixed", "tar", "none")
    assert classify_relative_path("shards/shard-000000.tar.gz") == (
        "mixed",
        "tar",
        "gzip",
    )


def test_export_target_local_is_credential_free_and_publish_is_noop(tmp_path):
    target = ExportTarget(tmp_path / "out")
    assert target.is_remote is False
    assert target.backend == "local"
    (target.local_root / "a.json").write_text("{}")
    assert target.publish(["a.json"]) == 0
    objects = target.build_objects(["a.json"])
    report = target.reconcile(objects)
    assert report.status == "passed"
    assert report.checked is True
    assert report.object_count == 1


def test_export_target_reconcile_flags_missing_and_mismatch(tmp_path):
    target = ExportTarget(tmp_path / "out")
    (target.local_root / "a.json").write_text("{}")
    objects = target.build_objects(["a.json"])
    # Delete the file to simulate a missing object after publish.
    (target.local_root / "a.json").unlink()
    with pytest.raises(ExportReconciliationError, match="missing"):
        target.reconcile(objects)


# --------------------------------------------------------------------------- #
# Local export path: reconciliation is recorded and passes
# --------------------------------------------------------------------------- #
def test_local_export_records_passing_reconciliation(lake, tmp_path):
    manifest = export_dataset_snapshot(
        lake, "demo-v1", out_dir=tmp_path / "lerobot", fmt="lerobot"
    )
    recon = manifest.reconciliation
    assert recon["backend"] == "local"
    assert recon["status"] == "passed"
    assert recon["checked"] is True
    assert recon["object_count"] >= 1
    # Every object carries a size, a provider fingerprint, and a
    # payload/metadata classification.
    classes = {obj["classification"] for obj in recon["objects"]}
    assert "payload" in classes  # the camera .bin
    assert all(obj["content_length"] >= 0 for obj in recon["objects"])
    assert all(obj.get("provider_fingerprint") for obj in recon["objects"])
    # The persisted curation row carries the reconciliation *summary* only --
    # never the exhaustive per-object array (BUG-02 oversized-cell shape).
    row = _materialization_rows(lake)[0]
    report = json.loads(row["report_json"])
    assert report["reconciliation_status"] == "passed"
    assert report["reconciliation"]["backend"] == "local"
    assert report["reconciliation"]["object_count"] >= 1
    assert "objects" not in report["reconciliation"]


# --------------------------------------------------------------------------- #
# Object-store export path
# --------------------------------------------------------------------------- #
def test_direct_export_dataset_snapshot_to_object_store(lake, monkeypatch):
    store = _install_fake_store(monkeypatch)
    manifest = export_dataset_snapshot(
        lake,
        "demo-v1",
        out_dir="s3://exports/direct",
        fmt="lerobot",
    )
    assert manifest.out_dir == "s3://exports/direct"
    assert manifest.reconciliation["backend"] == "s3"
    assert manifest.reconciliation["status"] == "passed"
    assert store.has(DATASET_EXPORT_MANIFEST_FILENAME)
    # The direct API records the materialization row with the reconciliation.
    row = _materialization_rows(lake)[0]
    report = json.loads(row["report_json"])
    assert report["reconciliation"]["backend"] == "s3"


def test_object_store_export_publishes_and_reconciles(lake, monkeypatch):
    store = _install_fake_store(monkeypatch)
    manifest = export_projection(
        lake,
        "demo-v1",
        fmt="lerobot",
        out_dir="s3://exports/demo",
        storage_options={"key": "k", "secret": "s"},
    )
    recon = manifest.reconciliation
    assert recon["backend"] == "s3"
    assert recon["status"] == "passed"
    assert recon["destination"] == "s3://exports/demo"
    assert manifest.accounting["target_path"] == "s3://exports/demo"
    # Data files + both manifests landed in the object store.
    assert store.has(DATASET_EXPORT_MANIFEST_FILENAME)
    assert store.has(PROJECTION_MANIFEST_FILENAME)
    assert store.has(".bin")  # camera payload
    # Provider fingerprints are recorded per object for later mutation detection.
    assert all(obj.get("provider_fingerprint") for obj in recon["objects"])


def test_local_and_object_store_exports_have_consistent_accounting(tmp_path, monkeypatch):
    local_lake = _snapshot_lake(tmp_path / "local.lance")
    local = export_projection(
        local_lake, "demo-v1", fmt="lerobot", out_dir=tmp_path / "local-out"
    )

    remote_lake = _snapshot_lake(tmp_path / "remote.lance")
    _install_fake_store(monkeypatch)
    remote = export_projection(
        remote_lake, "demo-v1", fmt="lerobot", out_dir="s3://exports/demo"
    )

    # The copy accounting (what bytes were copied vs referenced) is identical:
    # it is derived from the snapshot data, not from where the objects landed.
    for key in (
        "payload_bytes_referenced",
        "payload_bytes_copied",
        "logical_reference_bytes",
        "copy_ratio",
    ):
        assert local.accounting[key] == remote.accounting[key], key
    assert local.content_hashes["dataset"] == remote.content_hashes["dataset"]
    # The reconciled data-file byte totals are identical too -- same content,
    # different destination. (metadata_bytes_written differs only because the
    # manifests embed destination-specific object URIs.)
    for key in (
        "total_object_bytes",
        "payload_object_bytes",
        "metadata_object_bytes",
        "mixed_object_bytes",
        "object_count",
    ):
        assert local.reconciliation[key] == remote.reconciliation[key], key
    assert local.reconciliation["backend"] == "local"
    assert remote.reconciliation["backend"] == "s3"


def test_object_store_export_reconciliation_fails_on_missing_object(lake, monkeypatch):
    store = _install_fake_store(monkeypatch)
    store.drop_suffix = "meta/info.json"
    with pytest.raises(ExportReconciliationError, match="missing"):
        export_projection(lake, "demo-v1", fmt="lerobot", out_dir="s3://exports/demo")


def test_object_store_export_reconciliation_fails_on_size_mismatch(lake, monkeypatch):
    store = _install_fake_store(monkeypatch)
    store.truncate_suffix = "meta/info.json"
    with pytest.raises(ExportReconciliationError, match="size-mismatch"):
        export_projection(lake, "demo-v1", fmt="lerobot", out_dir="s3://exports/demo")


def test_object_store_export_is_idempotent_on_retry(lake, monkeypatch):
    _install_fake_store(monkeypatch)
    first = export_projection(lake, "demo-v1", fmt="lerobot", out_dir="s3://exports/demo")
    second = export_projection(lake, "demo-v1", fmt="lerobot", out_dir="s3://exports/demo")

    assert first.content_hashes["dataset"] == second.content_hashes["dataset"]
    assert first.reconciliation["status"] == "passed"
    assert second.reconciliation["status"] == "passed"
    # Re-running the same export to the same destination yields exactly one row.
    rows = _materialization_rows(lake)
    assert len({row["materialization_id"] for row in rows}) == 1


def test_export_object_bound_fails_fast_before_publishing(lake, monkeypatch):
    from lancedb_robotics import export_reconciliation

    store = _install_fake_store(monkeypatch)
    monkeypatch.setattr(export_reconciliation, "MAX_RECONCILED_OBJECTS", 1)
    with pytest.raises(ExportReconciliationError, match="exceeding the reconciliation bound"):
        export_projection(lake, "demo-v1", fmt="lerobot", out_dir="s3://exports/demo")
    # Fail-fast: the bound tripped before any object was uploaded.
    assert store.writes == []


def test_webdataset_object_store_export_classifies_shards_as_mixed(lake, monkeypatch):
    store = _install_fake_store(monkeypatch)
    manifest = export_projection(
        lake,
        "demo-v1",
        fmt="webdataset",
        out_dir="s3://exports/wds",
        shard_size=1,
    )
    assert manifest.reconciliation["status"] == "passed"
    containers = {obj["container"] for obj in manifest.reconciliation["objects"]}
    assert "tar" in containers
    assert store.has(".tar")


# --------------------------------------------------------------------------- #
# Manifest-only plan stays zero-copy and credential-free
# --------------------------------------------------------------------------- #
def test_projection_plan_never_touches_object_store(lake, monkeypatch):
    def _explode(*args, **kwargs):
        raise AssertionError("plan mode must not resolve an object-store destination")

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        SimpleNamespace(open=_explode, core=SimpleNamespace(url_to_fs=_explode)),
    )
    manifest = lake.projections.plan("lerobot", "demo-v1")
    assert manifest.accounting["payload_bytes_copied"] == 0
    assert not manifest.reconciliation
