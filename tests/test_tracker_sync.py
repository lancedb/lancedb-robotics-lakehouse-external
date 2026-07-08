"""External experiment-tracker manifest sync tests (backlog 0101)."""

import json

import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import training_run_artifact_id
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.tracker_sync import (
    TrackerSyncError,
    drift_report,
    export_manifest_bundle,
    import_manifest_bundle,
)


def _bundle(code_ref: str = "git:v1") -> dict:
    return {
        "source": "generic",
        "training_runs": [
            {
                "external_run_id": "run-1",
                "dataset_id": "ds-ext-1",
                "snapshot_name": "ext-snap",
                "snapshot_tag": "v1",
                "table_versions": [{"table": "observations", "version": 2, "tag": ""}],
                "code_ref": code_ref,
                "hyperparameters": {"lr": 0.001, "batch_size": 8},
                "environment": {"image": "trainer@sha256:abc"},
                "status": "completed",
                "external_refs": {"team": "perception"},
            }
        ],
        "model_artifacts": [
            {
                "external_artifact_id": "ckpt-1",
                "external_run_id": "run-1",
                "artifact_uri": "s3://models/policy.ckpt",
                "checksum": "sha256:ckpt",
                "framework": "torch",
                "epoch": 2,
                "step": 128,
                "metrics": {"train_loss": 0.1},
            }
        ],
        "evaluation_runs": [
            {
                "external_eval_id": "eval-1",
                "external_artifact_id": "ckpt-1",
                "dataset_id": "ds-ext-1",
                "snapshot_name": "ext-snap",
                "metrics": {"success_rate": 0.8},
                "slice_metrics": {"night": {"success_rate": 0.6}},
                "code_ref": "git:eval",
            }
        ],
    }


def _counts(lake) -> dict:
    return {
        table: lake.table(table).count_rows()
        for table in ("training_runs", "model_artifacts", "evaluation_runs")
    }


def _manifest_lake(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
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
    return lake, manifest


def test_generic_json_import_records_run_checkpoint_eval_and_is_idempotent(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")

    report = import_manifest_bundle(lake, _bundle(), source="generic")

    assert report.created == 3
    assert _counts(lake) == {"training_runs": 1, "model_artifacts": 1, "evaluation_runs": 1}

    training = lake.table("training_runs").to_arrow().to_pylist()[0]
    model = lake.table("model_artifacts").to_arrow().to_pylist()[0]
    evaluation = lake.table("evaluation_runs").to_arrow().to_pylist()[0]

    # Parent links resolve through the derived canonical ids.
    assert model["training_run_id"] == training["training_run_id"]
    assert evaluation["model_artifact_id"] == model["model_artifact_id"]
    assert evaluation["training_run_id"] == training["training_run_id"]

    # External idempotency key is recorded as a tracker reference.
    refs = {item["key"]: item["value"] for item in training["external_refs"]}
    assert refs["generic_run_id"] == "run-1"
    assert refs["tracker_source"] == "generic"
    assert json.loads(training["hyperparameters_json"])["lr"] == 0.001

    # Materialized eval metric surface (0100) is refreshed on eval import.
    metrics = lake.eval.metrics(metric="success_rate")
    assert metrics.materialized
    assert any(row["score"] == 0.8 for row in metrics.rows)

    # Re-import is idempotent: same ids, no duplicate rows.
    second = import_manifest_bundle(lake, _bundle(), source="generic")
    assert second.unchanged == 3
    assert second.created == 0
    assert _counts(lake) == {"training_runs": 1, "model_artifacts": 1, "evaluation_runs": 1}


def test_missing_mlflow_extra_raises_install_hint(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    with pytest.raises(TrackerSyncError, match=r"mlflow.*install"):
        # No bundle => live fetch => requires the mlflow client (absent in CI).
        import_manifest_bundle(lake, source="mlflow")


def test_labelling_a_bundle_as_mlflow_still_requires_the_client(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    with pytest.raises(TrackerSyncError, match="mlflow"):
        import_manifest_bundle(lake, _bundle(), source="mlflow")


def test_drift_report_flags_changed_run_and_leaves_rows_unchanged(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    import_manifest_bundle(lake, _bundle("git:v1"), source="generic")

    report = drift_report(lake, _bundle("git:v2"), source="generic")

    assert report.has_drift
    conflicts = report.conflicts
    assert len(conflicts) == 1
    assert conflicts[0].kind == "training-run"
    assert conflicts[0].drift is True
    assert conflicts[0].previous_digest is not None

    # Read-only: the canonical row is untouched.
    training = lake.table("training_runs").to_arrow().to_pylist()[0]
    assert training["code_ref"] == "git:v1"


def test_dry_run_import_plans_update_without_writing(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    import_manifest_bundle(lake, _bundle("git:v1"), source="generic")

    report = import_manifest_bundle(lake, _bundle("git:v2"), source="generic", dry_run=True)

    assert report.dry_run is True
    assert report.updated == 1
    assert report.transform_id is None
    training = lake.table("training_runs").to_arrow().to_pylist()[0]
    assert training["code_ref"] == "git:v1"


def test_conflict_external_wins_overwrites(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    import_manifest_bundle(lake, _bundle("git:v1"), source="generic")
    report = import_manifest_bundle(
        lake, _bundle("git:v2"), source="generic", conflict="external-wins"
    )
    assert report.updated == 1
    assert lake.table("training_runs").count_rows() == 1
    training = lake.table("training_runs").to_arrow().to_pylist()[0]
    assert training["code_ref"] == "git:v2"


def test_conflict_lake_wins_skips(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    import_manifest_bundle(lake, _bundle("git:v1"), source="generic")
    report = import_manifest_bundle(
        lake, _bundle("git:v2"), source="generic", conflict="lake-wins"
    )
    assert report.skipped == 1
    assert report.updated == 0
    training = lake.table("training_runs").to_arrow().to_pylist()[0]
    assert training["code_ref"] == "git:v1"


def test_conflict_append_superseding_adds_new_row(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    import_manifest_bundle(lake, _bundle("git:v1"), source="generic")
    report = import_manifest_bundle(
        lake, _bundle("git:v2"), source="generic", conflict="append-superseding"
    )
    assert report.superseded == 1
    rows = lake.table("training_runs").to_arrow().to_pylist()
    assert len(rows) == 2
    code_refs = {row["code_ref"] for row in rows}
    assert code_refs == {"git:v1", "git:v2"}
    superseding = next(row for row in rows if row["code_ref"] == "git:v2")
    refs = {item["key"]: item["value"] for item in superseding["external_refs"]}
    assert refs["tracker_supersedes"]

    # Re-importing the same superseding content is idempotent (no third row).
    again = import_manifest_bundle(
        lake, _bundle("git:v2"), source="generic", conflict="append-superseding"
    )
    assert again.superseded == 0
    assert lake.table("training_runs").count_rows() == 2


def test_import_records_sync_transform_and_lineage(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    report = import_manifest_bundle(lake, _bundle(), source="generic")

    transforms = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    ]
    assert len(transforms) == 1
    assert transforms[0]["kind"] == "tracker-import"
    assert transforms[0]["status"] == "completed"
    params = json.loads(transforms[0]["params"])
    assert params["source"] == "generic"

    training_id = lake.table("training_runs").to_arrow().to_pylist()[0]["training_run_id"]
    lake.lineage.refresh_graph()
    artifact_ids = {row["artifact_id"] for row in lake.table("lineage_artifacts").to_arrow().to_pylist()}
    assert training_run_artifact_id(training_id) in artifact_ids


def test_export_payload_contains_table_versions_checksum_and_digest(tmp_path, fixtures_dir):
    lake, snapshot = _manifest_lake(tmp_path, fixtures_dir)
    training = lake.training.record_run("demo-v1", code_ref="git:native")
    checkpoint = lake.training.record_checkpoint(
        training_run_id=training.training_run_id,
        model_artifact_id="policy-ckpt",
        artifact_uri="s3://models/policy.ckpt",
        checksum="sha256:native-ckpt",
    )
    lake.eval.record_run(
        "demo-v1",
        model_artifact_id=checkpoint.model_artifact_id,
        metrics={"success_rate": 0.9},
    )

    report = export_manifest_bundle(lake, source="generic")
    bundle = report.bundle

    exported_run = bundle["training_runs"][0]
    assert exported_run["dataset_id"] == snapshot.dataset_id
    assert exported_run["table_versions"]  # pinned versions present
    assert exported_run["manifest_digest"] == training.manifest_digest
    assert exported_run["code_ref"] == "git:native"

    exported_ckpt = bundle["model_artifacts"][0]
    assert exported_ckpt["checksum"] == "sha256:native-ckpt"
    assert exported_ckpt["manifest_digest"] == checkpoint.manifest_digest

    assert bundle["evaluation_runs"][0]["metrics"] == {"success_rate": 0.9}


def test_export_writes_json_file_and_records_transform(tmp_path, fixtures_dir):
    lake, _snapshot = _manifest_lake(tmp_path, fixtures_dir)
    lake.training.record_run("demo-v1", code_ref="git:native")

    out = tmp_path / "bundle.json"
    report = export_manifest_bundle(lake, source="generic", out_path=out)

    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk["training_runs"][0]["code_ref"] == "git:native"
    assert report.transform_id is not None
    kinds = {row["kind"] for row in lake.table("transform_runs").to_arrow().to_pylist()}
    assert "tracker-export" in kinds


def test_export_round_trips_ids_and_content_into_a_fresh_lake(tmp_path, fixtures_dir):
    src, _snapshot = _manifest_lake(tmp_path / "src", fixtures_dir)
    training = src.training.record_run("demo-v1", code_ref="git:native")
    checkpoint = src.training.record_checkpoint(
        training_run_id=training.training_run_id,
        model_artifact_id="policy-ckpt",
        artifact_uri="s3://models/policy.ckpt",
        checksum="sha256:native-ckpt",
    )
    src.eval.record_run(
        "demo-v1",
        model_artifact_id=checkpoint.model_artifact_id,
        metrics={"success_rate": 0.9},
    )
    bundle = export_manifest_bundle(src, source="generic").bundle

    dest = Lake.init(tmp_path / "dest.lance")
    imported = import_manifest_bundle(dest, bundle, source="generic")
    assert imported.created == 3

    # The explicit ids in the exported bundle preserve identity across lakes.
    dest_run = dest.table("training_runs").to_arrow().to_pylist()[0]
    dest_model = dest.table("model_artifacts").to_arrow().to_pylist()[0]
    assert dest_run["training_run_id"] == training.training_run_id
    assert dest_run["code_ref"] == "git:native"
    assert dest_model["model_artifact_id"] == checkpoint.model_artifact_id
    assert dest_model["checksum"] == "sha256:native-ckpt"
    assert dest_model["training_run_id"] == training.training_run_id

    # Re-import into the same fresh lake stays idempotent.
    again = import_manifest_bundle(dest, bundle, source="generic")
    assert again.unchanged == 3
    assert dest.table("training_runs").count_rows() == 1


def test_unknown_conflict_policy_is_rejected(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    with pytest.raises(TrackerSyncError, match="conflict policy"):
        import_manifest_bundle(lake, _bundle(), source="generic", conflict="nonsense")


def test_bundle_entry_missing_external_id_is_rejected(tmp_path):
    lake = Lake.init(tmp_path / "robot.lance")
    bad = {"training_runs": [{"code_ref": "git:v1"}]}
    with pytest.raises(TrackerSyncError, match="external id"):
        import_manifest_bundle(lake, bad, source="generic")
