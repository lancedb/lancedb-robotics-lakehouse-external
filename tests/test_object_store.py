"""Object-store lake/source behavior (backlog 0023)."""

import sys
from types import SimpleNamespace

import lancedb
import pytest

from lancedb_robotics.adapters import AdapterError, get_adapter
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.storage import lancedb_storage_options


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


def _rows(lake, table):
    return lake.table(table).to_arrow().to_pylist()


def test_auth_ref_env_options_resolve_without_persisting_secrets(monkeypatch):
    monkeypatch.setenv(
        "LANCEDB_ROBOTICS_AUTH_LAB_MINIO_STORAGE_OPTIONS_JSON",
        '{"endpoint_url": "http://localhost:9000", "key": "access", "secret": "hidden"}',
    )

    options = lancedb_storage_options(
        "s3://robotics-lake/demo",
        auth_ref="lab-minio",
        storage_options={"region": "us-east-1"},
    )

    assert options == {
        "endpoint_url": "http://localhost:9000",
        "key": "access",
        "secret": "hidden",
        "region": "us-east-1",
    }


def test_ingest_mcap_from_s3_uri_preserves_low_copy_raw_uri(
    tmp_path, sample_mcap, monkeypatch
):
    remote_uri = "s3://robotics-raw/sample.mcap"
    opened = []

    def fake_open(uri, mode="rb", **kwargs):
        assert uri == remote_uri
        assert mode == "rb"
        assert kwargs["endpoint_url"] == "http://localhost:9000"
        assert kwargs["key"] == "access"
        assert kwargs["secret"] == "hidden"
        assert kwargs["client_kwargs"]["region_name"] == "us-east-1"
        opened.append(uri)
        return sample_mcap.open(mode)

    monkeypatch.setitem(sys.modules, "fsspec", SimpleNamespace(open=fake_open))

    local_lake = Lake.init(tmp_path / "local.lance")
    remote_lake = Lake.init(tmp_path / "remote.lance")

    local = ingest_mcap(local_lake, sample_mcap)
    remote = ingest_mcap(
        remote_lake,
        remote_uri,
        auth_ref="lab-minio",
        storage_options={
            "endpoint_url": "http://localhost:9000",
            "region": "us-east-1",
            "key": "access",
            "secret": "hidden",
        },
    )

    assert opened
    assert remote.run_id == local.run_id
    assert remote.message_count == local.message_count
    assert remote.observations_by_topic == local.observations_by_topic
    assert remote.source.uri == remote_uri
    assert remote.source.auth_ref == "lab-minio"

    run = _rows(remote_lake, "runs")[0]
    assert run["raw_uri"] == remote_uri
    assert run["source_id"] == remote.source.source_id

    observations = _rows(remote_lake, "observations")
    assert observations
    assert {row["raw_uri"] for row in observations} == {remote_uri}

    local_observations = _rows(local_lake, "observations")
    assert _stable_observations(observations) == _stable_observations(local_observations)

    source = _rows(remote_lake, "integration_sources")[0]
    assert source["uri"] == remote_uri
    assert source["auth_ref"] == "lab-minio"
    metadata = {entry["key"]: entry["value"] for entry in source["metadata"]}
    assert metadata["adapter"] == "mcap"
    assert "hidden" not in str(source)
    assert "secret" not in metadata


class _TrackingTable:
    def __init__(self, table, name, added):
        self._table = table
        self._name = name
        self._added = added

    def add(self, data, *args, **kwargs):
        self._added.append(self._name)
        return self._table.add(data, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._table, name)


class _TrackingDb:
    def __init__(self, db):
        self._db = db
        self.added = []

    def list_tables(self):
        return self._db.list_tables()

    def create_table(self, name, *args, **kwargs):
        return self._db.create_table(name, *args, **kwargs)

    def open_table(self, name):
        return _TrackingTable(self._db.open_table(name), name, self.added)


def test_mixed_s3_ingest_into_db_remote_lake_keeps_auth_refs_separate(
    tmp_path, sample_mcap, monkeypatch
):
    remote_uri = "s3://robotics-raw/sample.mcap"
    opened_sources = []

    def fake_open(uri, mode="rb", **kwargs):
        assert uri == remote_uri
        assert mode == "rb"
        assert kwargs["endpoint_url"] == "http://localhost:9000"
        assert kwargs["key"] == "source-access"
        assert kwargs["secret"] == "source-hidden"
        assert kwargs["client_kwargs"]["region_name"] == "us-east-1"
        opened_sources.append(uri)
        return sample_mcap.open(mode)

    backing_db = lancedb.connect(tmp_path / "enterprise-backed.lance")
    tracking_db = _TrackingDb(backing_db)

    def fake_connect(uri, **kwargs):
        assert uri == "db://robotics"
        assert kwargs == {"api_key": "ldb-secret", "region": "us-east-1"}
        return tracking_db

    monkeypatch.setitem(sys.modules, "fsspec", SimpleNamespace(open=fake_open))
    monkeypatch.setenv("LANCEDB_ROBOTICS_AUTH_ENTERPRISE_PROD_REMOTE_API_KEY", "ldb-secret")
    monkeypatch.setattr("lancedb_robotics.lake.lancedb.connect", fake_connect)

    lake = Lake.init("db://robotics", remote_auth_ref="enterprise-prod")
    report = ingest_mcap(
        lake,
        remote_uri,
        auth_ref="raw-prod",
        storage_options={
            "endpoint_url": "http://localhost:9000",
            "region": "us-east-1",
            "key": "source-access",
            "secret": "source-hidden",
        },
    )

    assert opened_sources
    assert set(opened_sources) == {remote_uri}
    assert "observations" in tracking_db.added
    assert "integration_sources" in tracking_db.added
    assert report.source.uri == remote_uri
    assert report.source.auth_ref == "raw-prod"

    source = _rows(lake, "integration_sources")[0]
    assert source["auth_ref"] == "raw-prod"
    assert source["uri"] == remote_uri
    assert "ldb-secret" not in str(source)
    assert "source-hidden" not in str(source)

    run = _rows(lake, "runs")[0]
    assert run["raw_uri"] == remote_uri


def test_remote_mcap_missing_sdk_is_actionable(monkeypatch):
    def fake_open(uri, mode="rb", **kwargs):
        raise ImportError("No module named 's3fs'")

    monkeypatch.setitem(sys.modules, "fsspec", SimpleNamespace(open=fake_open))

    with pytest.raises(AdapterError, match="install s3fs"):
        get_adapter("mcap").inspect("s3://robotics-raw/sample.mcap")


def test_remote_mcap_bad_credentials_are_actionable(monkeypatch):
    def fake_open(uri, mode="rb", **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setitem(sys.modules, "fsspec", SimpleNamespace(open=fake_open))

    with pytest.raises(AdapterError, match="cannot read s3://robotics-raw/sample.mcap"):
        get_adapter("mcap").inspect("s3://robotics-raw/sample.mcap")


def _stable_observations(rows):
    stable = []
    for row in rows:
        copy = dict(row)
        copy.pop("created_at", None)
        copy["raw_uri"] = "<raw>"
        stable.append(copy)
    return sorted(stable, key=lambda row: row["observation_id"])
