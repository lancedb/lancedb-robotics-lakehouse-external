"""CLI tests for `lancedb-robotics lake` (backlog 0002)."""

from types import SimpleNamespace

from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.schemas import CANONICAL_TABLES, SCHEMA_METADATA_VERSION_KEY, SCHEMA_VERSIONS

runner = CliRunner()


class _ListTables:
    tables = list(CANONICAL_TABLES)


class _Table:
    def __init__(self, name):
        self.schema = SimpleNamespace(
            metadata={SCHEMA_METADATA_VERSION_KEY.encode(): SCHEMA_VERSIONS[name].encode()}
        )

    def count_rows(self):
        return 0


class _RemoteDb:
    def list_tables(self):
        return _ListTables()

    def open_table(self, name):
        return _Table(name)


def test_lake_init_creates_tables_without_ingesting(tmp_path):
    lake_path = tmp_path / "robot.lance"
    result = runner.invoke(app, ["lake", "init", "--lake", str(lake_path)])
    assert result.exit_code == 0

    from lancedb_robotics.lake import Lake

    lake = Lake.open(lake_path)
    assert lake.table_names() == list(CANONICAL_TABLES)
    for name in lake.table_names():
        assert lake.table(name).count_rows() == 0


def test_lake_init_reports_each_table(tmp_path):
    lake_path = tmp_path / "robot.lance"
    result = runner.invoke(app, ["lake", "init", "--lake", str(lake_path)])
    assert result.exit_code == 0
    for name in CANONICAL_TABLES:
        assert name in result.output


def test_lake_init_twice_is_idempotent(tmp_path):
    lake_path = tmp_path / "robot.lance"
    first = runner.invoke(app, ["lake", "init", "--lake", str(lake_path)])
    second = runner.invoke(app, ["lake", "init", "--lake", str(lake_path)])
    assert first.exit_code == 0
    assert second.exit_code == 0


def test_lake_tables_lists_tables_and_schema_versions(tmp_path):
    lake_path = tmp_path / "robot.lance"
    runner.invoke(app, ["lake", "init", "--lake", str(lake_path)])
    result = runner.invoke(app, ["lake", "tables", "--lake", str(lake_path)])
    assert result.exit_code == 0
    for name in CANONICAL_TABLES:
        assert name in result.output
    assert "v1" in result.output


def test_lake_tables_on_missing_lake_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["lake", "tables", "--lake", str(tmp_path / "nope.lance")])
    assert result.exit_code != 0


def test_lake_tables_accepts_enterprise_remote_options(monkeypatch):
    monkeypatch.setenv("LANCEDB_ROBOTICS_AUTH_ENTERPRISE_PROD_REMOTE_API_KEY", "ldb-secret")
    seen = {}

    def fake_connect(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return _RemoteDb()

    monkeypatch.setattr("lancedb_robotics.lake.lancedb.connect", fake_connect)

    result = runner.invoke(
        app,
        [
            "lake",
            "tables",
            "--lake",
            "db://robotics",
            "--remote-auth-ref",
            "enterprise-prod",
            "--region",
            "us-west-2",
            "--host-override",
            "https://phalanx.acme.internal",
            "--client-config-json",
            '{"retry_config": {"retries": 3}}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {
        "args": ("db://robotics",),
        "kwargs": {
            "api_key": "ldb-secret",
            "region": "us-west-2",
            "host_override": "https://phalanx.acme.internal",
            "client_config": {"retry_config": {"retries": 3}},
        },
    }


def test_lake_tables_accepts_rest_namespace_options(monkeypatch):
    seen = {}

    def fake_connect(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return _RemoteDb()

    monkeypatch.setattr("lancedb_robotics.lake.lancedb.connect", fake_connect)

    result = runner.invoke(
        app,
        [
            "lake",
            "tables",
            "--namespace-impl",
            "rest",
            "--namespace-uri",
            "https://phalanx.acme.internal",
            "--namespace-database",
            "acme",
            "--namespace-prefix",
            "robotics",
            "--namespace-auth-ref",
            "phalanx-prod",
            "--namespace-pushdown",
            "QueryTable",
            "--namespace-pushdown",
            "CreateTable",
        ],
    )

    assert result.exit_code == 0, result.output
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
