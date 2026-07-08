"""Evidence-pack catalog, retention, and scalable materialization tests (0108)."""

import copy
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.evidence import EvidencePackError, is_supported_evidence_schema
from lancedb_robotics.evidence_catalog import (
    CATALOG_SCHEMA_VERSION,
    CATALOG_TABLE,
    EVENTS_TABLE,
    evidence_pack_events,
    evidence_retention_plan,
    expire_evidence_pack,
    list_evidence_packs,
    load_evidence_pack,
    materialize_evidence_pack,
    plan_materialization,
    record_evidence_pack,
    set_evidence_retention,
)
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.storage import join_uri, uri_exists
from lancedb_robotics.writeback import ingest_model_outputs

runner = CliRunner()


def _lake(tmp_path, fixtures_dir, fixture_name="records.mcap"):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / fixture_name)
    create_scenario_windows(lake, window_ns=100_000_000)
    scenarios = sorted(
        lake.table("scenarios").to_arrow().to_pylist(),
        key=lambda row: (row["start_time_ns"], row["scenario_id"]),
    )
    manifest = create_snapshot(
        lake,
        name="demo-v1",
        tag="training-demo",
        scenario_ids=[row["scenario_id"] for row in scenarios],
    )
    ingest_model_outputs(
        lake,
        {
            "model_output_id": "out-regression",
            "observation_id": scenarios[0]["observation_ids"][0],
            "scenario_id": scenarios[0]["scenario_id"],
            "dataset_id": manifest.dataset_id,
            "model_version": "policy@abc123",
            "prediction": "regressed",
            "score": 0.12,
            "producer_run_id": "checkpoint-abc123",
        },
        source="trainer",
    )
    return lake, manifest


def _pack(lake, **kwargs):
    return lake.lineage.trace_checkpoint("checkpoint-abc123", limit=1).evidence_pack(**kwargs)


def _data_versions(lake):
    return {
        name: (int(lake.table(name).version), lake.table(name).count_rows())
        for name in lake.table_names()
        if name not in {CATALOG_TABLE, EVENTS_TABLE}
    }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --- record / load ----------------------------------------------------------


def test_record_and_reload_by_digest_and_subject(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)

    entry = record_evidence_pack(lake, pack, metadata={"owner": "qa"})
    assert entry.pack_id == pack.manifest_digest
    assert entry.catalog_schema_version == CATALOG_SCHEMA_VERSION
    assert entry.manifest_schema_version == "lancedb-robotics/evidence-pack/v1"
    assert entry.subject_handle == "checkpoint-abc123"
    assert entry.materialization_status == "planned"
    assert entry.metadata == {"owner": "qa"}
    assert entry.source_coordinate_hashes  # source-coordinate hashes recorded

    by_digest, manifest = load_evidence_pack(lake, digest=pack.manifest_digest)
    assert by_digest.pack_id == entry.pack_id
    assert manifest == pack.manifest  # stored manifest round-trips exactly

    by_subject, _ = load_evidence_pack(lake, subject="checkpoint-abc123")
    assert by_subject.pack_id == entry.pack_id


def test_record_is_idempotent_by_digest(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)

    first = record_evidence_pack(lake, pack, protected=True)
    second = record_evidence_pack(lake, pack, protected=True)

    assert first.pack_id == second.pack_id
    assert first.created_at == second.created_at  # created_at preserved on re-record
    assert lake.table(CATALOG_TABLE).count_rows() == 1


def test_load_missing_and_ambiguous_args_raise(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    with pytest.raises(EvidencePackError):
        load_evidence_pack(lake, digest="does-not-exist")
    with pytest.raises(EvidencePackError):
        load_evidence_pack(lake)
    with pytest.raises(EvidencePackError):
        load_evidence_pack(lake, digest="a", subject="b")


def test_record_does_not_mutate_canonical_data_tables(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    before = _data_versions(lake)

    record_evidence_pack(lake, pack)

    assert _data_versions(lake) == before


# --- list + pagination ------------------------------------------------------


def test_list_evidence_packs_paginates_and_filters(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    base = _pack(lake).manifest
    digests = []
    for i in range(5):
        manifest = copy.deepcopy(base)
        manifest["subject"] = dict(manifest["subject"])
        manifest["subject"]["model_run_id"] = f"ckpt-{i}"
        digests.append(record_evidence_pack(lake, manifest).pack_id)

    page1 = list_evidence_packs(lake, page_size=2)
    assert len(page1.packs) == 2
    assert page1.next_cursor is not None

    page2 = list_evidence_packs(lake, page_size=2, cursor=page1.next_cursor)
    page3 = list_evidence_packs(lake, page_size=2, cursor=page2.next_cursor)
    assert page3.next_cursor is None

    seen = {e.pack_id for e in (*page1.packs, *page2.packs, *page3.packs)}
    assert seen == set(digests)

    filtered = list_evidence_packs(lake, subject_handle="ckpt-3")
    assert [e.subject_handle for e in filtered.packs] == ["ckpt-3"]


# --- retention --------------------------------------------------------------


def test_retention_metadata_survives_catalog_reload(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    entry = record_evidence_pack(lake, pack, protected=True, retention_policy="legal-hold")

    reopened = Lake.open(tmp_path / "robot.lance")
    loaded, _ = load_evidence_pack(reopened, digest=entry.pack_id)
    assert loaded.protected is True
    assert loaded.retention_policy == "legal-hold"


def test_set_retention_and_expiry_plan(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    entry = record_evidence_pack(lake, pack)

    updated = set_evidence_retention(lake, entry.pack_id, protected=True, retention_policy="hold")
    assert updated.protected is True
    assert updated.retention_policy == "hold"

    plan = evidence_retention_plan(lake, older_than=timedelta(days=0))
    assert plan["expirable_count"] == 0
    assert any(row["pack_id"] == entry.pack_id for row in plan["protected"])

    set_evidence_retention(lake, entry.pack_id, protected=False)
    plan2 = evidence_retention_plan(lake, older_than=timedelta(days=0))
    assert any(row["pack_id"] == entry.pack_id for row in plan2["expirable"])


def test_expire_is_force_gated_by_protection(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    entry = record_evidence_pack(lake, pack, protected=True)

    with pytest.raises(EvidencePackError):
        expire_evidence_pack(lake, entry.pack_id)
    assert lake.table(CATALOG_TABLE).count_rows() == 1

    result = expire_evidence_pack(lake, entry.pack_id, force=True)
    assert result["expired"] is True
    assert lake.table(CATALOG_TABLE).count_rows() == 0
    # Audit trail of the pack survives its expiry.
    events = evidence_pack_events(lake, pack_id=entry.pack_id)
    assert {"created", "expired"} <= {e["event_type"] for e in events}


# --- materialization --------------------------------------------------------


def test_materialization_is_chunked_idempotent_and_deterministic(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    out = tmp_path / "mat"

    first = materialize_evidence_pack(
        lake, pack, output_dir=str(out), include_payloads=True, include_attachments=True, chunk_size=1
    )
    assert first.status == "materialized"
    assert first.copied_count == first.file_count > 0
    assert first.chunk_count == first.file_count  # chunk_size=1 -> one chunk per object
    for record in first.files:
        payload = (out / record["path"]).read_bytes()
        assert record["bytes"] == len(payload)
        assert record["sha256"] == _sha256(payload)

    second = materialize_evidence_pack(
        lake, pack, output_dir=str(out), include_payloads=True, include_attachments=True, chunk_size=1
    )
    assert second.copied_count == 0
    assert second.skipped_count == first.file_count
    assert [f["sha256"] for f in first.files] == [f["sha256"] for f in second.files]

    # Catalog row flips to materialized with the byte/file totals.
    entry, _ = load_evidence_pack(lake, digest=pack.manifest_digest)
    assert entry.materialization_status == "materialized"
    assert entry.bytes_total == first.bytes_total
    assert entry.file_count == first.file_count


def test_materialization_resumes_after_a_missing_object(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    out = tmp_path / "mat"

    first = materialize_evidence_pack(lake, pack, output_dir=str(out), include_attachments=True)
    assert first.copied_count >= 2

    # Simulate an interrupted copy: one completed object goes missing on disk.
    dropped = first.files[0]
    (out / dropped["path"]).unlink()

    resumed = materialize_evidence_pack(lake, pack, output_dir=str(out), include_attachments=True)
    assert resumed.copied_count == 1  # only the missing object is re-copied
    assert resumed.skipped_count == first.file_count - 1
    assert [f["sha256"] for f in first.files] == [f["sha256"] for f in resumed.files]


def test_materialization_enforces_file_limit_before_writing(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    out = tmp_path / "mat"

    with pytest.raises(EvidencePackError, match="max_files"):
        materialize_evidence_pack(
            lake, pack, output_dir=str(out), include_attachments=True, max_files=1
        )
    assert not out.exists()  # nothing written on a failed plan


def test_materialization_enforces_byte_limit_before_writing(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    out = tmp_path / "mat"

    with pytest.raises(EvidencePackError, match="max_bytes"):
        materialize_evidence_pack(
            lake, pack, output_dir=str(out), include_attachments=True, max_bytes=1
        )
    assert not out.exists()


def test_plan_materialization_chunks_and_totals(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)

    plan = plan_materialization(lake, pack, include_attachments=True, chunk_size=1)
    assert plan["object_count"] >= 2
    assert plan["chunk_count"] == plan["object_count"]
    assert plan["known_bytes"] > 0  # attachment sizes known from the manifest


def test_materialize_via_output_uri_writes_manifest(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    dest = str(tmp_path / "uri-dest")

    report = materialize_evidence_pack(
        lake, pack, output_uri=dest, include_attachments=True
    )
    assert uri_exists(join_uri(dest, "manifest.json"))
    written = json.loads((Path(dest) / "manifest.json").read_text())
    assert written["mode"] == "materialized"
    assert written["verification"]["materialized_file_hashes"]
    assert report.output_uri == dest


# --- privacy / governance ---------------------------------------------------


def test_sensitive_source_deny_and_flag(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)

    with pytest.raises(EvidencePackError, match="sensitive"):
        record_evidence_pack(
            lake, pack, sensitive_source_patterns=["*records.mcap*"], on_sensitive="deny"
        )
    denials = evidence_pack_events(lake, event_type="redaction-denied")
    assert denials and denials[0]["status"] == "denied"

    flagged = record_evidence_pack(
        lake, pack, sensitive_source_patterns=["*records.mcap*"], on_sensitive="flag"
    )
    assert flagged.redacted is True
    assert any("records.mcap" in src for src in flagged.sensitive_sources)


def test_audit_events_recorded_for_lifecycle(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    record_evidence_pack(lake, pack)
    materialize_evidence_pack(lake, pack, output_dir=str(tmp_path / "m"), include_attachments=True)

    events = evidence_pack_events(lake, pack_id=pack.manifest_digest)
    assert [e["event_type"] for e in events][:1] == ["created"]
    assert "materialized" in {e["event_type"] for e in events}


# --- schema versioning ------------------------------------------------------


def test_unsupported_manifest_schema_is_rejected(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    manifest = copy.deepcopy(_pack(lake).manifest)
    manifest["schema_version"] = "lancedb-robotics/evidence-pack/v99"

    assert is_supported_evidence_schema("lancedb-robotics/evidence-pack/v1")
    assert not is_supported_evidence_schema("lancedb-robotics/evidence-pack/v99")
    with pytest.raises(EvidencePackError, match="unsupported"):
        record_evidence_pack(lake, manifest)


def test_v1_manifests_remain_loadable(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    pack = _pack(lake)
    entry = record_evidence_pack(lake, pack)
    _, manifest = load_evidence_pack(lake, digest=entry.pack_id)
    assert manifest["schema_version"] == "lancedb-robotics/evidence-pack/v1"


# --- CLI --------------------------------------------------------------------


def test_cli_evidence_catalog_roundtrip(tmp_path, fixtures_dir):
    lake, _ = _lake(tmp_path, fixtures_dir)
    lake_uri = str(tmp_path / "robot.lance")

    def run(*args):
        result = runner.invoke(app, list(args))
        assert result.exit_code == 0, (args, result.output)
        return json.loads(result.output)

    recorded = run(
        "lineage", "record-evidence", "checkpoint-abc123",
        "--lake", lake_uri, "--checkpoint", "--limit", "1", "--protected",
    )
    digest = recorded["pack_id"]
    assert recorded["protected"] is True

    listed = run("lineage", "list-evidence", "--lake", lake_uri)
    assert listed["count"] == 1

    shown = run("lineage", "show-evidence", "--lake", lake_uri, "--digest", digest, "--include-manifest")
    assert shown["manifest"]["schema_version"] == "lancedb-robotics/evidence-pack/v1"

    mat = run(
        "lineage", "materialize-evidence", "--lake", lake_uri, "--digest", digest,
        "--include-attachments", "--output-dir", str(tmp_path / "cli-mat"), "--chunk-size", "1",
    )
    assert mat["status"] == "materialized"
    assert mat["copied_count"] == mat["file_count"]

    refused = runner.invoke(app, ["lineage", "expire-evidence", "--lake", lake_uri, "--digest", digest])
    assert refused.exit_code == 1  # protected

    forced = run("lineage", "expire-evidence", "--lake", lake_uri, "--digest", digest, "--force")
    assert forced["expired"] is True
