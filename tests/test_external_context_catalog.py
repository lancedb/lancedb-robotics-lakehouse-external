"""Queryable external-context catalog, indexing, and redaction tests (backlog 0114)."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.evidence import _finalize_pack
from lancedb_robotics.external_context_catalog import (
    CATALOG_TABLE,
    ExternalContextError,
    backfill_external_contexts,
    find_external_context,
    record_external_context,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.redaction import ContextRedactionPolicy

runner = CliRunner()


def _lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


def _record_execution(lake, *, execution_id, provider, run_id, job_id="", **ctx):
    context = {"provider": provider, "external_run_id": run_id, "external_job_id": job_id}
    context.update(ctx)
    lake.lineage.record_execution(
        kind="training-run",
        execution_id=execution_id,
        provider=provider,
        transform_id=f"transform-{execution_id}",
        params={"lineage_context": {k: v for k, v in context.items() if v}},
        output_artifact_ids=[f"lancedb-robotics:model:{execution_id}"],
        status="completed",
    )


# --- Acceptance: distinct namespaced records --------------------------------


def test_two_providers_same_run_id_resolve_to_distinct_records(tmp_path):
    lake = _lake(tmp_path)
    _record_execution(lake, execution_id="exec-a", provider="wandb", run_id="RUN-1", job_id="jobA")
    _record_execution(lake, execution_id="exec-b", provider="mlflow", run_id="RUN-1", job_id="jobB")

    backfill_external_contexts(lake)
    page = find_external_context(lake, external_run_id="RUN-1")

    assert page.as_dict()["count"] == 2
    by_provider = {c.provider: c for c in page.contexts}
    assert set(by_provider) == {"wandb", "mlflow"}
    # Namespaced: distinct catalog ids and distinct resolved canonical executions.
    assert by_provider["wandb"].context_id != by_provider["mlflow"].context_id
    assert by_provider["wandb"].execution_id == "exec-a"
    assert by_provider["mlflow"].execution_id == "exec-b"
    # Provider + run filter narrows to exactly one.
    only = find_external_context(lake, provider="wandb", external_run_id="RUN-1")
    assert only.as_dict()["count"] == 1
    assert only.contexts[0].execution_id == "exec-a"


def test_resolves_to_canonical_execution_and_artifacts(tmp_path):
    lake = _lake(tmp_path)
    _record_execution(
        lake,
        execution_id="exec-a",
        provider="wandb",
        run_id="RUN-9",
        code_ref="git:abc123",
        artifact_refs=[{"provider": "wandb", "artifact_uri": "wandb://team/model:v1"}],
    )
    backfill_external_contexts(lake)

    page = find_external_context(lake, provider="wandb")
    entry = page.contexts[0]
    assert entry.execution_id == "exec-a"
    assert entry.artifact_ids == ("lancedb-robotics:model:exec-a",)
    assert entry.transform_id == "transform-exec-a"
    assert entry.code_ref == "git:abc123"
    # External artifact URI lookup resolves back to the same canonical row.
    by_uri = find_external_context(lake, artifact_uri="wandb://team/model:v1")
    assert by_uri.as_dict()["count"] == 1
    assert by_uri.contexts[0].execution_id == "exec-a"


# --- Acceptance: idempotent, bounded backfill -------------------------------


def test_backfill_is_idempotent_and_bounded(tmp_path):
    lake = _lake(tmp_path)
    for i in range(5):
        _record_execution(lake, execution_id=f"exec-{i}", provider="wandb", run_id=f"RUN-{i}")

    first = backfill_external_contexts(lake, batch_size=2)
    assert first.recorded == 5
    assert first.updated == 0
    assert first.batches >= 3  # 5 rows / batch_size 2 -> bounded batches
    total_rows = lake.table(CATALOG_TABLE).to_lance().to_table().num_rows
    assert total_rows == 5

    # Re-running produces no duplicate rows.
    second = backfill_external_contexts(lake, batch_size=2)
    assert second.recorded == 0
    assert second.updated == 5
    assert lake.table(CATALOG_TABLE).to_lance().to_table().num_rows == 5


def test_backfill_skips_rows_without_external_handles(tmp_path):
    lake = _lake(tmp_path)
    # No lineage_context on this execution -> nothing to index.
    lake.lineage.record_execution(kind="row-projection", execution_id="plain", transform_id="t-plain")
    _record_execution(lake, execution_id="exec-a", provider="wandb", run_id="RUN-1")

    report = backfill_external_contexts(lake)
    assert report.recorded == 1
    assert report.skipped >= 1


def test_backfill_reads_transform_runs_source(tmp_path):
    lake = _lake(tmp_path)
    import pyarrow as pa

    from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA

    params = json.dumps({"lineage_context": {"provider": "airflow", "external_run_id": "dag-run-7", "external_job_id": "dag-x"}})
    row = {
        "transform_id": "tr-airflow",
        "kind": "training-run",
        "source_id": "",
        "input_uris": [],
        "input_table_versions": [],
        "output_tables": [],
        "params": params,
        "status": "completed",
        "error": "",
        "started_at": None,
        "finished_at": None,
        "created_by": "",
        "created_at": datetime.now(UTC),
    }
    lake.table("transform_runs").add(pa.Table.from_pylist([row], schema=TRANSFORM_RUNS_SCHEMA))

    report = backfill_external_contexts(lake, sources=("transform_runs",))
    assert report.recorded == 1
    page = find_external_context(lake, provider="airflow")
    assert page.contexts[0].external_job_id == "dag-x"
    assert page.contexts[0].source_table == "transform_runs"


# --- Acceptance: deterministic paging ---------------------------------------


def test_find_pages_results_deterministically(tmp_path):
    lake = _lake(tmp_path)
    for i in range(6):
        _record_execution(lake, execution_id=f"exec-{i}", provider="wandb", run_id=f"RUN-{i}")
    backfill_external_contexts(lake)

    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = find_external_context(lake, provider="wandb", page_size=2, cursor=cursor)
        seen.extend(c.context_id for c in page.contexts)
        pages += 1
        cursor = page.next_cursor
        if cursor is None:
            break
    assert pages == 3
    assert len(seen) == 6
    assert len(set(seen)) == 6  # no repeats across pages

    # Re-paging yields the identical ordering (deterministic).
    again: list[str] = []
    cursor = None
    while True:
        page = find_external_context(lake, provider="wandb", page_size=2, cursor=cursor)
        again.extend(c.context_id for c in page.contexts)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert again == seen


# --- Acceptance: redaction before evidence-pack materialization -------------


def test_redacted_environment_keys_absent_from_evidence_pack_output(tmp_path):
    lake = _lake(tmp_path)
    manifest = {
        "schema_version": "lancedb-robotics/evidence-pack/v1",
        "lake_uri": lake.uri,
        "mode": "plan",
        "subject": {"type": "lineage-graph", "handle": "model:m1"},
        "table_versions": [],
        "row_ids": {},
        "source_coordinates": [],
        "rows": {
            "training_runs": [
                {
                    "training_run_id": "tr1",
                    "environment_json": json.dumps({"python": "3.14", "wandb_api_key": "super-secret"}),
                    "external_refs": [
                        {"key": "external_run_id", "value": "run-1"},
                        {"key": "hf_token", "value": "leaked-token"},
                    ],
                }
            ],
        },
        "model_outputs": [],
        "model_artifacts": [],
        "training_run": None,
        "lineage_executions": [],
        "transform_runs": [
            {"transform_id": "t1", "params": json.dumps({"lineage_context": {"password": "hunter2"}})}
        ],
        "payload_refs": [],
        "attachment_refs": [],
        "video_refs": [],
        "video_encoding_refs": [],
        "materialized_files": [],
        "verification": {},
    }

    plain = _finalize_pack(
        lake, json.loads(json.dumps(manifest)), output_dir=None, materialize=False,
        include_payloads=False, include_attachments=False, include_video=False,
    )
    policy = ContextRedactionPolicy(name="strict")
    redacted = _finalize_pack(
        lake, json.loads(json.dumps(manifest)), output_dir=str(tmp_path / "pack"), materialize=True,
        include_payloads=False, include_attachments=False, include_video=False,
        redaction_policy=policy,
    )

    written = (tmp_path / "pack" / "manifest.json").read_text()
    for secret in ("super-secret", "leaked-token", "hunter2"):
        assert secret not in written
        assert secret not in json.dumps(redacted.manifest)
    # Non-sensitive data survives, and redaction changes the pack digest.
    assert "run-1" in written
    assert redacted.manifest_digest != plain.manifest_digest


def test_record_external_context_stores_redacted_context(tmp_path):
    lake = _lake(tmp_path)
    entry = record_external_context(
        lake,
        provider="wandb",
        external_run_id="run-1",
        context={"provider": "wandb", "external_run_id": "run-1", "api_key": "secret"},
        redaction_policy=ContextRedactionPolicy(name="strict"),
    )
    assert entry.redacted is True
    assert "api_key" not in entry.context
    stored = lake.table(CATALOG_TABLE).to_lance().to_table().to_pylist()[0]
    assert "secret" not in stored["context_json"]


# --- Retention / governance -------------------------------------------------


def test_retention_holds_gate_expiry(tmp_path):
    lake = _lake(tmp_path)
    _record_execution(lake, execution_id="exec-a", provider="wandb", run_id="RUN-1")
    backfill_external_contexts(lake)
    cid = find_external_context(lake, provider="wandb").contexts[0].context_id

    lake.lineage.set_external_context_retention(cid, legal_hold=True)
    with pytest.raises(ExternalContextError):
        lake.lineage.expire_external_context(cid)

    plan = lake.lineage.external_context_retention_plan()
    assert plan["held_count"] == 1
    assert plan["expirable_count"] == 0

    result = lake.lineage.expire_external_context(cid, force=True)
    assert result["expired"] is True and result["forced"] is True
    assert lake.table(CATALOG_TABLE).to_lance().to_table().num_rows == 0


def test_expire_respects_expires_at(tmp_path):
    lake = _lake(tmp_path)
    _record_execution(lake, execution_id="exec-a", provider="wandb", run_id="RUN-1")
    backfill_external_contexts(lake)
    cid = find_external_context(lake, provider="wandb").contexts[0].context_id

    lake.lineage.set_external_context_retention(cid, expires_at=datetime.now(UTC) - timedelta(days=1))
    plan = lake.lineage.external_context_retention_plan()
    assert plan["expirable_count"] == 1
    assert plan["expirable"][0]["reason"] == "expired"


def test_events_are_appended_for_backfill_and_retention(tmp_path):
    lake = _lake(tmp_path)
    _record_execution(lake, execution_id="exec-a", provider="wandb", run_id="RUN-1")
    backfill_external_contexts(lake)
    cid = find_external_context(lake, provider="wandb").contexts[0].context_id
    lake.lineage.set_external_context_retention(cid, protected=True)

    events = lake.lineage.external_context_events()
    types = {e["event_type"] for e in events}
    assert "backfilled" in types
    assert "retention-updated" in types


# --- Backwards compatibility ------------------------------------------------


def test_lineage_context_json_readable_without_catalog(tmp_path):
    """Lakes that never build the catalog still read lineage_context from params."""
    lake = _lake(tmp_path)
    _record_execution(lake, execution_id="exec-a", provider="wandb", run_id="RUN-1")
    row = lake.table("lineage_executions").to_lance().to_table(columns=["params_json"]).to_pylist()[0]
    assert json.loads(row["params_json"])["lineage_context"]["provider"] == "wandb"
    # No external_contexts rows were created implicitly.
    assert lake.table(CATALOG_TABLE).to_lance().to_table().num_rows == 0


# --- CLI --------------------------------------------------------------------


def test_cli_backfill_find_and_events(tmp_path):
    lake = _lake(tmp_path)
    _record_execution(lake, execution_id="exec-a", provider="wandb", run_id="RUN-1", job_id="jobA")
    _record_execution(lake, execution_id="exec-b", provider="mlflow", run_id="RUN-1", job_id="jobB")
    uri = str(tmp_path / "robot.lance")

    backfill = runner.invoke(app, ["lineage", "backfill-external-context", "--lake", uri])
    assert backfill.exit_code == 0, backfill.output
    assert json.loads(backfill.output)["recorded"] == 2

    found = runner.invoke(
        app, ["lineage", "find-external-context", "--lake", uri, "--run-id", "RUN-1", "--page-size", "1"]
    )
    assert found.exit_code == 0, found.output
    payload = json.loads(found.output)
    assert payload["count"] == 1
    assert payload["next_cursor"] is not None

    events = runner.invoke(app, ["lineage", "external-context-events", "--lake", uri])
    assert events.exit_code == 0, events.output
    assert any(e["event_type"] == "backfilled" for e in json.loads(events.output))


def test_cli_find_requires_valid_cursor(tmp_path):
    lake = _lake(tmp_path)
    _record_execution(lake, execution_id="exec-a", provider="wandb", run_id="RUN-1")
    uri = str(tmp_path / "robot.lance")
    runner.invoke(app, ["lineage", "backfill-external-context", "--lake", uri])
    bad = runner.invoke(
        app, ["lineage", "find-external-context", "--lake", uri, "--cursor", "not-base64!!"]
    )
    assert bad.exit_code == 1
    assert "invalid cursor" in bad.output
