"""Lake open/create API tests (backlog 0002)."""

from dataclasses import dataclass

import lancedb
import pytest

from lancedb_robotics.lake import Lake, LakeError
from lancedb_robotics.schemas import CANONICAL_TABLES, SCHEMA_VERSIONS, TABLE_SCHEMAS


@pytest.fixture
def lake_path(tmp_path):
    return tmp_path / "robot.lance"


def test_init_creates_all_canonical_tables(lake_path):
    lake = Lake.init(lake_path)
    assert lake.table_names() == list(CANONICAL_TABLES)


def test_init_creates_empty_tables(lake_path):
    lake = Lake.init(lake_path)
    for name in lake.table_names():
        assert lake.table(name).count_rows() == 0


def test_init_is_idempotent(lake_path):
    Lake.init(lake_path)
    lake = Lake.init(lake_path)  # second init must not fail or duplicate
    assert lake.table_names() == list(CANONICAL_TABLES)
    for name in lake.table_names():
        assert lake.table(name).count_rows() == 0


def test_init_preserves_existing_rows(lake_path):
    lake = Lake.init(lake_path)
    lake.table("integration_sources").add(
        [{"source_id": "src-1", "kind": "mcap", "uri": "s3://bucket/log.mcap"}]
    )
    lake = Lake.init(lake_path)
    assert lake.table("integration_sources").count_rows() == 1


def test_init_adds_post_v0_tables_to_existing_lake(lake_path):
    db = lancedb.connect(lake_path)
    for name, schema in TABLE_SCHEMAS.items():
        if name not in {
            "labels",
            "model_outputs",
            "feedback",
            "alignment_jobs",
            "aligned_frames",
            "aligned_ticks",
            "curation_comparisons",
            "distribution_catalog",
            "training_runs",
            "model_artifacts",
            "evaluation_runs",
            "lineage_delivery_attempts",
            "lineage_audit_reports",
            "keyframe_map_artifact_referrers",
        }:
            db.create_table(name, schema=schema, exist_ok=True)

    lake = Lake.init(lake_path)

    assert lake.table_names() == list(CANONICAL_TABLES)
    assert lake.table("labels").count_rows() == 0
    assert lake.table("model_outputs").count_rows() == 0
    assert lake.table("feedback").count_rows() == 0
    assert lake.table("alignment_jobs").count_rows() == 0
    assert lake.table("aligned_frames").count_rows() == 0
    assert lake.table("aligned_ticks").count_rows() == 0
    assert lake.table("curation_comparisons").count_rows() == 0
    assert lake.table("distribution_catalog").count_rows() == 0
    assert lake.table("training_runs").count_rows() == 0
    assert lake.table("model_artifacts").count_rows() == 0
    assert lake.table("evaluation_runs").count_rows() == 0
    assert lake.table("lineage_delivery_attempts").count_rows() == 0
    assert lake.table("lineage_audit_reports").count_rows() == 0
    assert lake.table("keyframe_map_artifact_referrers").count_rows() == 0


def test_schema_versions_inspectable_after_reopen(lake_path):
    Lake.init(lake_path)
    lake = Lake.open(lake_path)
    assert lake.schema_versions() == SCHEMA_VERSIONS


def test_open_missing_lake_raises(lake_path):
    with pytest.raises(LakeError):
        Lake.open(lake_path)


def test_open_does_not_create(lake_path):
    with pytest.raises(LakeError):
        Lake.open(lake_path)
    assert not lake_path.exists()


@dataclass
class _ListTables:
    tables: list[str]


class _RemoteDb:
    def __init__(self, tables=()):
        self.tables = list(tables)

    def list_tables(self):
        return _ListTables(self.tables)

    def create_table(self, name, *, schema, exist_ok):
        if name not in self.tables:
            self.tables.append(name)


def test_open_accepts_s3_uri_without_local_path_gate(monkeypatch):
    seen = {}

    def fake_connect(uri, **kwargs):
        seen["uri"] = uri
        seen["kwargs"] = kwargs
        return _RemoteDb(CANONICAL_TABLES)

    monkeypatch.setattr("lancedb_robotics.lake.lancedb.connect", fake_connect)

    lake = Lake.open(
        "s3://robotics-lake/demo",
        storage_options={"endpoint_url": "http://localhost:9000", "region": "us-east-1"},
    )

    assert lake.uri == "s3://robotics-lake/demo"
    assert lake.table_names() == list(CANONICAL_TABLES)
    assert seen == {
        "uri": "s3://robotics-lake/demo",
        "kwargs": {
            "storage_options": {
                "endpoint_url": "http://localhost:9000",
                "region": "us-east-1",
            }
        },
    }


def test_open_accepts_db_uri_without_local_path_gate(monkeypatch):
    monkeypatch.setenv("LANCEDB_ROBOTICS_AUTH_ENTERPRISE_PROD_REMOTE_API_KEY", "ldb-secret")
    seen = {}

    def fake_connect(uri, **kwargs):
        seen["uri"] = uri
        seen["kwargs"] = kwargs
        return _RemoteDb(CANONICAL_TABLES)

    monkeypatch.setattr("lancedb_robotics.lake.lancedb.connect", fake_connect)

    lake = Lake.open(
        "db://robotics",
        remote_auth_ref="enterprise-prod",
        region="us-west-2",
        host_override="https://phalanx.acme.internal",
        client_config={"retry_config": {"retries": 3}},
    )

    assert lake.uri == "db://robotics"
    assert lake.table_names() == list(CANONICAL_TABLES)
    assert seen == {
        "uri": "db://robotics",
        "kwargs": {
            "api_key": "ldb-secret",
            "region": "us-west-2",
            "host_override": "https://phalanx.acme.internal",
            "client_config": {"retry_config": {"retries": 3}},
        },
    }


@pytest.mark.parametrize("uri", ["lancedb://robotics", "phalanx://robotics"])
def test_open_rejects_unsupported_remote_schemes(uri):
    with pytest.raises(LakeError, match="unsupported lake URI scheme"):
        Lake.open(uri)


def test_open_builds_rest_namespace_connection(monkeypatch):
    seen = {}

    def fake_connect(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return _RemoteDb(CANONICAL_TABLES)

    monkeypatch.setattr("lancedb_robotics.lake.lancedb.connect", fake_connect)

    lake = Lake.open(
        namespace_client_impl="rest",
        namespace_client_properties={
            "uri": "https://phalanx.acme.internal",
            "header.x-lancedb-database": "acme",
            "header.x-lancedb-database-prefix": "robotics",
        },
        namespace_auth_ref="phalanx-prod",
        namespace_client_pushdown_operations=["QueryTable", "CreateTable"],
    )

    assert lake.uri == "namespace://rest/https://phalanx.acme.internal"
    assert seen == {
        "args": (),
        "kwargs": {
            "namespace_client_impl": "rest",
            "namespace_client_properties": {
                "uri": "https://phalanx.acme.internal",
                "header.x-lancedb-database": "acme",
                "header.x-lancedb-database-prefix": "robotics",
                "dynamic_context_provider.impl": (
                    "lancedb_robotics.connections.RuntimeNamespaceAuthProvider"
                ),
                "dynamic_context_provider.auth_ref": "phalanx-prod",
            },
            "namespace_client_pushdown_operations": ["QueryTable", "CreateTable"],
        },
    }


def test_init_accepts_s3_uri_and_creates_remote_tables(monkeypatch):
    db = _RemoteDb()

    def fake_connect(uri, **kwargs):
        assert uri == "s3://robotics-lake/demo"
        assert kwargs == {"storage_options": {"endpoint_url": "http://localhost:9000"}}
        return db

    monkeypatch.setattr("lancedb_robotics.lake.lancedb.connect", fake_connect)

    lake = Lake.init(
        "s3://robotics-lake/demo",
        storage_options={"endpoint_url": "http://localhost:9000"},
    )

    assert lake.table_names() == list(CANONICAL_TABLES)


def test_remote_lake_connection_errors_are_actionable(monkeypatch):
    def fake_connect(uri, **kwargs):
        raise ImportError("No module named 's3fs'")

    monkeypatch.setattr("lancedb_robotics.lake.lancedb.connect", fake_connect)

    with pytest.raises(LakeError, match="object-store dependency"):
        Lake.open("s3://robotics-lake/demo")
