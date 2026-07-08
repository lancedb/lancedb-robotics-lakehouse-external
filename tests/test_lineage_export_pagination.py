"""Bounded external lineage export pagination + bulk URN catalog (backlog 0105)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from lancedb_robotics import lineage_integrations
from lancedb_robotics.cli import app
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage_integrations import (
    LineageExportFilters,
    LineageIntegrationError,
    artifact_id_from_external_urn,
)
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.schemas import (
    LINEAGE_ARTIFACTS_SCHEMA,
    LINEAGE_EDGES_SCHEMA,
    LINEAGE_EXECUTIONS_SCHEMA,
)

runner = CliRunner()


def _pipeline_lake(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    scenarios = sorted(
        lake.table("scenarios").to_arrow().to_pylist(),
        key=lambda row: (row["start_time_ns"], row["scenario_id"]),
    )
    snapshot = create_snapshot(
        lake,
        name="ol-demo",
        tag="pagination-demo",
        scenario_ids=[row["scenario_id"] for row in scenarios],
    )
    training = lake.training.record_run(
        "ol-demo",
        code_ref="git:trainer",
        hyperparameters={"lr": 0.001},
    )
    return lake, snapshot, training


def _synthetic_graph(tmp_path, *, count: int, kinds: tuple[str, ...] = ("dataset-snapshot",)):
    """Write ``count`` artifacts (+ a chain of edges/executions) directly.

    Bypasses the pipeline so tests can control graph size without ingesting.
    Callers must export with ``refresh=False`` so the synthetic rows survive.
    """

    lake = Lake.init(tmp_path / "synthetic.lance")
    base = datetime(2026, 1, 1, tzinfo=UTC)
    artifacts = []
    executions = []
    edges = []
    for index in range(count):
        artifact_id = f"lancedb-robotics:synthetic:{index:05d}"
        kind = kinds[index % len(kinds)]
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "kind": kind,
                "name": f"synthetic-{index:05d}",
                "table_name": "scenarios" if index % 2 == 0 else "training_runs",
                "table_version": index % 4,
                "table_tag": None,
                "row_grain": None,
                "row_ids": [],
                "source_uri": None,
                "source_id": None,
                "digest": f"sha256:{index:064d}",
                "producer_execution_id": None,
                "metadata": [],
                "created_at": base + timedelta(minutes=index),
            }
        )
        if index > 0:
            executions.append(
                {
                    "execution_id": f"lancedb-robotics:execution:{index:05d}",
                    "kind": "training-run" if index % 3 == 0 else "dataset-snapshot",
                    "name": f"exec-{index:05d}",
                    "transform_id": None,
                    "status": "completed",
                    "params_json": None,
                    "code_ref": None,
                    "provider": None,
                    "environment_json": None,
                    "input_artifact_ids": [f"lancedb-robotics:synthetic:{index - 1:05d}"],
                    "output_artifact_ids": [artifact_id],
                    "input_table_versions": [],
                    "output_table_versions": [],
                    "started_at": base + timedelta(minutes=index),
                    "finished_at": base + timedelta(minutes=index, seconds=30),
                    "created_by": None,
                    "metadata": [],
                    "created_at": base + timedelta(minutes=index),
                }
            )
            edges.append(
                {
                    "edge_id": f"lancedb-robotics:edge:{index:05d}",
                    "edge_type": "derived-from",
                    "from_artifact_id": f"lancedb-robotics:synthetic:{index - 1:05d}",
                    "to_artifact_id": artifact_id,
                    "execution_id": f"lancedb-robotics:execution:{index:05d}",
                    "metadata": [],
                    "created_at": base + timedelta(minutes=index),
                }
            )
    lake.table("lineage_artifacts").add(
        pa.Table.from_pylist(artifacts, schema=LINEAGE_ARTIFACTS_SCHEMA)
    )
    if executions:
        lake.table("lineage_executions").add(
            pa.Table.from_pylist(executions, schema=LINEAGE_EXECUTIONS_SCHEMA)
        )
    if edges:
        lake.table("lineage_edges").add(
            pa.Table.from_pylist(edges, schema=LINEAGE_EDGES_SCHEMA)
        )
    return lake


def test_urn_catalog_pages_cover_all_artifacts_without_duplicates(tmp_path):
    lake = _synthetic_graph(tmp_path, count=57)

    seen: list[str] = []
    token = None
    pages = 0
    while True:
        page = lake.lineage.export_artifact_urns(
            page_size=10,
            page_token=token,
            refresh=False,
        )
        pages += 1
        assert page.record_count <= 10
        seen.extend(record["artifact_id"] for record in page.records)
        if page.next_page_token is None:
            assert page.truncated is False
            break
        assert page.truncated is True
        token = page.next_page_token

    assert pages == 6  # ceil(57 / 10)
    assert len(seen) == 57
    assert len(set(seen)) == 57  # no duplicates
    assert seen == sorted(seen)  # deterministic, sorted by canonical id
    assert all(page_total == 57 for page_total in (57,))  # sanity


def test_urn_catalog_record_has_required_fields(tmp_path):
    lake = _synthetic_graph(tmp_path, count=3)

    page = lake.lineage.export_artifact_urns(backend="datahub", refresh=False)

    assert page.matched_total == 3
    record = page.records[0]
    assert set(record) >= {
        "artifact_id",
        "artifact_urn",
        "kind",
        "table_name",
        "table_version",
        "digest",
    }
    assert record["artifact_urn"].startswith("urn:li:dataset:")
    assert artifact_id_from_external_urn(record["artifact_urn"]) == record["artifact_id"]


def test_page_token_rejected_when_filters_change(tmp_path):
    lake = _synthetic_graph(
        tmp_path, count=20, kinds=("dataset-snapshot", "training-run")
    )

    page = lake.lineage.export_artifact_urns(
        page_size=5,
        artifact_kind="dataset-snapshot",
        refresh=False,
    )
    assert page.next_page_token is not None

    # Resuming with a different filter must reject the continuation handle.
    with pytest.raises(LineageIntegrationError, match="does not match this query"):
        lake.lineage.export_artifact_urns(
            page_size=5,
            page_token=page.next_page_token,
            artifact_kind="training-run",
            refresh=False,
        )

    # Same filter resumes cleanly.
    resumed = lake.lineage.export_artifact_urns(
        page_size=5,
        page_token=page.next_page_token,
        artifact_kind="dataset-snapshot",
        refresh=False,
    )
    assert resumed.records
    assert all(record["kind"] == "dataset-snapshot" for record in resumed.records)


def test_artifact_kind_filter_narrows_catalog(tmp_path):
    lake = _synthetic_graph(
        tmp_path, count=20, kinds=("dataset-snapshot", "training-run")
    )

    page = lake.lineage.export_artifact_urns(
        artifact_kind="training-run", refresh=False
    )

    assert page.matched_total == 10
    assert all(record["kind"] == "training-run" for record in page.records)


def test_export_stays_bounded_in_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(lineage_integrations, "_EXPORT_SCAN_BATCH_SIZE", 16)
    lake = _synthetic_graph(tmp_path, count=400)

    page = lake.lineage.export_artifact_urns(page_size=10, refresh=False)

    assert page.matched_total == 400
    assert page.scanned_rows == 400
    # Peak rows held in memory must be independent of table size: at most one
    # batch plus the bounded selection heap (page_size + 1).
    assert page.peak_retained_rows <= 16 + 10 + 1
    assert page.peak_retained_rows < 400


def test_openlineage_page_filters_by_execution_kind_and_paginates(tmp_path):
    lake = _synthetic_graph(tmp_path, count=30)

    seen = []
    token = None
    while True:
        page = lake.lineage.export_openlineage_page(
            page_size=7,
            page_token=token,
            execution_kind="training-run",
            refresh=False,
        )
        assert page.payload_kind == "openlineage-run-event"
        for event in page.records:
            facet = event["run"]["facets"]["lancedb_robotics_execution"]
            assert facet["kind"] == "training-run"
            seen.append(facet["execution_id"])
        if page.next_page_token is None:
            break
        token = page.next_page_token

    assert seen == sorted(seen)
    assert len(seen) == len(set(seen))
    assert seen  # at least one training-run execution


def test_datahub_page_paginates_edges(tmp_path):
    lake = _synthetic_graph(tmp_path, count=25)

    seen = []
    token = None
    while True:
        page = lake.lineage.export_datahub_page(
            page_size=8, page_token=token, refresh=False
        )
        assert page.payload_kind == "datahub-upstream-lineage-edge"
        for edge in page.records:
            assert edge["upstreamUrn"].startswith("urn:li:dataset:")
            seen.append(edge["lineage"]["edge_id"])
        if page.next_page_token is None:
            break
        token = page.next_page_token

    assert len(seen) == 24  # count-1 chained edges
    assert len(seen) == len(set(seen))


def test_iter_ndjson_streams_every_record_with_summary(tmp_path):
    lake = _synthetic_graph(tmp_path, count=13)

    records = list(
        lake.lineage.iter_artifact_urn_ndjson(
            page_size=4, refresh=False, include_summary=True
        )
    )

    summary = records[-1]
    assert summary["type"] == "lineage-export-summary"
    assert summary["record_count"] == 13
    catalog = records[:-1]
    assert len(catalog) == 13
    assert [r["artifact_id"] for r in catalog] == sorted(
        r["artifact_id"] for r in catalog
    )


# --- CLI coverage -----------------------------------------------------------


def test_cli_export_openlineage_ndjson_is_line_delimited(tmp_path, fixtures_dir):
    lake, _snapshot, _training = _pipeline_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "lineage",
            "export-openlineage",
            "--lake",
            str(lake.uri),
            "--format",
            "ndjson",
            "--page-size",
            "2",
            "--summary",
        ],
    )

    assert result.exit_code == 0, result.output
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[-1]["type"] == "lineage-export-summary"
    for record in parsed[:-1]:
        assert record["eventType"] in {"COMPLETE", "START", "FAIL", "ABORT"}


def test_cli_export_urns_paginates_json(tmp_path, fixtures_dir):
    lake, _snapshot, _training = _pipeline_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "lineage",
            "export-urns",
            "--lake",
            str(lake.uri),
            "--backend",
            "datahub",
            "--page-size",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["payload_kind"] == "artifact-urn-catalog"
    assert payload["backend"] == "datahub"
    assert payload["record_count"] <= 1
    assert "next_page_token" in payload
    record = payload["records"][0]
    assert record["artifact_urn"].startswith("urn:li:dataset:")


def test_cli_export_datahub_ndjson(tmp_path, fixtures_dir):
    lake, _snapshot, _training = _pipeline_lake(tmp_path, fixtures_dir)

    result = runner.invoke(
        app,
        [
            "lineage",
            "export-datahub",
            "--lake",
            str(lake.uri),
            "--format",
            "ndjson",
            "--page-size",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines
    for line in lines:
        record = json.loads(line)
        assert record["type"] == "DataHubUpstreamLineageEdge"


def test_filters_build_normalizes_created_and_table_versions():
    filters = LineageExportFilters.build(
        created_after="2026-01-01T00:00:00Z",
        table_versions=["scenarios=3", "training_runs=1"],
    )
    assert filters.created_after == datetime(2026, 1, 1, tzinfo=UTC)
    assert filters.table_version_map() == {"scenarios": 3, "training_runs": 1}
