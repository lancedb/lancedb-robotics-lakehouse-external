"""Distribution gap analysis and stratified balance reports (backlog 0055)."""

import json
from datetime import timedelta

import pyarrow as pa
import pytest
from test_curate import NOW, _build_curation_lake, _scenario, _snapshot_row
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.distributions import DistributionError
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import CURATION_MEMBERSHIPS_SCHEMA, LABELS_SCHEMA

runner = CliRunner()


def _distribution_spec(lake: Lake):
    return lake.distributions.define(
        name="pick-coverage",
        dimensions=["site_id", "object_category"],
        min_count_per_slice=1,
    )


def _training_snapshot(lake: Lake, name: str = "train-v1"):
    return lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name=name,
        split_by="scenario",
    )


def _deployment_manifest():
    return {
        "kind": "external-manifest",
        "name": "deployment-window",
        "dimensions": ["site_id", "object_category"],
        "slices": {
            "site_id=site-b|object_category=box": 4,
        },
        "table_versions": [{"table": "deployment", "version": 42, "tag": "last-30-days"}],
    }


def test_distribution_measure_counts_percentages_and_stats(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    spec = _distribution_spec(lake)

    report = lake.distributions.measure(spec, source=lake.curate.workbench())

    assert report.slice_counts == {
        "site_id=site-a|object_category=box": 1,
        "site_id=site-a|object_category=cup": 2,
        "site_id=site-b|object_category=box": 2,
        "site_id=site-b|object_category=cup": 1,
    }
    cup = report.slice("site_id=site-a|object_category=cup")
    assert cup is not None
    assert cup.percentage == pytest.approx(2 / 6)
    assert cup.quality_stats["quality_score_mean"] == pytest.approx(0.95)
    assert cup.label_completeness["completeness"] == 0.0
    assert {table for table, _ in report.table_versions} >= {"scenarios", "runs"}

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    )
    params = json.loads(transform["params"])
    assert transform["kind"] == "distribution-report"
    assert params["slice_counts"] == report.slice_counts
    assert {tv["table"] for tv in params["input_table_versions"]} >= {"scenarios", "runs"}


def test_distribution_compare_surfaces_missing_deployment_slice(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _training_snapshot(lake)
    spec = _distribution_spec(lake)

    train_report = lake.distributions.measure(spec, source={"kind": "snapshot", "name": "train-v1"})
    deployment_report = lake.distributions.measure(spec, source=_deployment_manifest())
    comparison = lake.distributions.compare(observed=train_report, target=deployment_report)

    missing = [finding for finding in comparison.gap_findings if finding.kind == "missing"]
    assert missing
    assert missing[0].label == "site_id=site-b|object_category=box"
    assert missing[0].needed_count == 2
    assert comparison.summary["missing_count"] >= 1

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == comparison.transform_id
    )
    assert transform["kind"] == "distribution-comparison"


def test_curate_from_gaps_selects_candidates_only_from_gap_slices(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _training_snapshot(lake)
    spec = _distribution_spec(lake)
    train_report = lake.distributions.measure(spec, source={"kind": "snapshot", "name": "train-v1"})
    deployment_report = lake.distributions.measure(spec, source=_deployment_manifest())
    comparison = lake.distributions.compare(observed=train_report, target=deployment_report)

    selection = lake.curate.from_gaps(comparison)
    manifest = selection.snapshot(name="gap-closure-v1", split_by="scenario")

    assert set(selection.scenario_ids) == {"scn-site-b-box", "scn-site-b-box-extra"}
    assert all(scenario_id not in train_report.scenario_ids for scenario_id in selection.scenario_ids)
    assert selection.report["selected_by_gap"]["site_id=site-b|object_category=box"] == [
        "scn-site-b-box",
        "scn-site-b-box-extra",
    ]
    snapshot = _snapshot_row(lake, "gap-closure-v1")
    assert manifest.scenario_ids == tuple(sorted(selection.scenario_ids))
    assert json.loads(snapshot["balance_report"])["operation"] == "from-gaps"


def test_distribution_report_uses_snapshot_versions_after_live_append(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    snapshot = _training_snapshot(lake)
    spec = _distribution_spec(lake)

    report = lake.distributions.measure(spec, source={"kind": "snapshot", "name": "train-v1"})
    before_counts = report.slice_counts
    pinned_versions = dict(report.table_versions)

    scenarios = lake.table("scenarios")
    scenarios.add(
        pa.Table.from_pylist(
            [
                _scenario(
                    "scn-late-site-b-box",
                    run_id="run-b",
                    start_time_ns=90,
                    object_category="box",
                    embedding=[0.0, 0.0, 0.8, 0.2],
                )
            ],
            schema=scenarios.schema,
        )
    )
    repeated = lake.distributions.measure(spec, source={"kind": "snapshot", "name": "train-v1"})

    assert report.table_versions == snapshot.table_versions
    assert repeated.slice_counts == before_counts
    assert dict(repeated.table_versions) == pinned_versions
    assert pinned_versions["scenarios"] < int(lake.table("scenarios").version)


def test_stratified_sample_accepts_distribution_spec(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    spec = _distribution_spec(lake)

    sampled = lake.curate.workbench().stratified_sample(spec=spec)

    assert sampled.report["distribution_spec"]["spec_id"] == spec.spec_id
    assert set(sampled.report["slice_counts"].values()) == {1}
    assert len(sampled.scenario_ids) == 4


def test_distributions_cli_measure_snapshot(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    _training_snapshot(lake)

    result = runner.invoke(
        app,
        [
            "gaps",
            "measure",
            "--lake",
            str(lake_path),
            "--snapshot",
            "train-v1",
            "--dimension",
            "site_id",
            "--dimension",
            "object_category",
            "--min-count-per-slice",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "report: dist-report-" in result.output
    assert "source: dataset-snapshot" in result.output
    assert "slice: site_id=site-a|object_category=box count=1" in result.output


def test_distribution_catalog_lists_fetches_and_resolves_latest_report(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    spec = _distribution_spec(lake)

    first = lake.distributions.measure(spec, name="daily-balance")
    scenarios = lake.table("scenarios")
    scenarios.add(
        pa.Table.from_pylist(
            [
                _scenario(
                    "scn-later-site-b-cup",
                    run_id="run-b",
                    start_time_ns=90,
                    object_category="cup",
                    embedding=[0.0, 0.8, 0.2, 0.0],
                )
            ],
            schema=scenarios.schema,
        )
    )
    second = lake.distributions.measure(spec, name="daily-balance")

    entries = lake.distributions.list_reports(
        name="daily-balance",
        spec_id=spec.spec_id,
        source_kind="scope",
    )
    assert [entry.report_id for entry in entries] == [second.report_id, first.report_id]
    assert entries[0].summary["total_count"] == second.total_count
    assert entries[0].transform_id == second.transform_id

    fetched = lake.distributions.get_report(first.report_id)
    assert fetched.slice_counts == first.slice_counts
    assert fetched.table_versions == first.table_versions
    latest = lake.distributions.latest_report(name="daily-balance", spec_id=spec.spec_id)
    assert latest.report_id == second.report_id

    transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == second.transform_id
    )
    assert "distribution_catalog" in transform["output_tables"]


def test_distribution_catalog_retention_compacts_body_but_keeps_audit_metadata(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    spec = _distribution_spec(lake)
    first = lake.distributions.measure(spec, name="retained-balance")
    scenarios = lake.table("scenarios")
    scenarios.add(
        pa.Table.from_pylist(
            [
                _scenario(
                    "scn-retention-late",
                    run_id="run-b",
                    start_time_ns=95,
                    object_category="box",
                    embedding=[0.0, 0.1, 0.7, 0.2],
                )
            ],
            schema=scenarios.schema,
        )
    )
    second = lake.distributions.measure(spec, name="retained-balance")

    result = lake.distributions.compact_reports(
        older_than=timedelta(seconds=0),
        retain_latest_per_name=1,
    )

    assert result.compacted_count == 1
    assert result.body_bytes_after < result.body_bytes_before
    first_entry = lake.distributions.list_catalog(report_id=first.report_id)[0]
    assert first_entry.body_compacted is True
    assert first_entry.body is None
    assert first_entry.body_sha1
    assert first_entry.table_versions == first.table_versions
    assert first_entry.transform_id == first.transform_id
    second_entry = lake.distributions.list_catalog(report_id=second.report_id)[0]
    assert second_entry.body_compacted is False
    with pytest.raises(DistributionError, match="body was compacted"):
        lake.distributions.get_report(first.report_id)
    assert lake.distributions.latest_report(name="retained-balance").report_id == second.report_id


def test_distribution_catalog_records_comparisons_and_gap_findings(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _training_snapshot(lake)
    spec = _distribution_spec(lake)
    train_report = lake.distributions.measure(spec, source={"kind": "snapshot", "name": "train-v1"})
    deployment_report = lake.distributions.measure(spec, source=_deployment_manifest())
    comparison = lake.distributions.compare(observed=train_report, target=deployment_report)

    entries = lake.distributions.list_catalog(comparison_id=comparison.comparison_id)

    assert {entry.kind for entry in entries} >= {"comparison", "gap-finding"}
    persisted = lake.distributions.get_comparison(comparison.comparison_id)
    assert persisted["comparison_id"] == comparison.comparison_id
    finding_entries = [entry for entry in entries if entry.kind == "gap-finding"]
    assert finding_entries
    assert finding_entries[0].summary["kind"] in {"missing", "underrepresented", "overrepresented"}


def test_distributions_cli_lists_and_fetches_latest_catalog_report(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    spec = _distribution_spec(lake)
    report = lake.distributions.measure(spec, name="cli-balance")

    listed = runner.invoke(
        app,
        [
            "gaps",
            "list",
            "--lake",
            str(lake_path),
            "--name",
            "cli-balance",
            "--format",
            "json",
        ],
    )

    assert listed.exit_code == 0, listed.output
    rows = json.loads(listed.output)
    assert rows[0]["report_id"] == report.report_id
    assert rows[0]["summary"]["slice_counts"] == report.slice_counts

    latest = runner.invoke(
        app,
        [
            "gaps",
            "latest",
            "--lake",
            str(lake_path),
            "--name",
            "cli-balance",
            "--format",
            "json",
        ],
    )

    assert latest.exit_code == 0, latest.output
    payload = json.loads(latest.output)
    assert payload["report_id"] == report.report_id
    assert payload["slice_counts"] == report.slice_counts


def test_distribution_measure_buckets_high_cardinality_with_reason(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    scenarios = lake.table("scenarios")
    scenarios.add(
        pa.Table.from_pylist(
            [
                _scenario(
                    f"scn-rare-{index}",
                    run_id="run-a",
                    start_time_ns=100 + index,
                    object_category=f"rare-{index}",
                    embedding=[0.1, 0.2, 0.3, 0.4],
                )
                for index in range(5)
            ],
            schema=scenarios.schema,
        )
    )
    spec = lake.distributions.define(name="object-coverage", dimensions=["object_category"])

    with pytest.raises(DistributionError, match="exceeding max_slice_count=3"):
        lake.distributions.measure(spec, max_slice_count=3)

    report = lake.distributions.measure(
        spec,
        max_slice_count=3,
        overflow="bucket",
        top_k_overflow=2,
        max_scenario_ids_per_slice=1,
    )

    assert len(report.slices) == 3
    assert report.slice_counts["__bucket__=overflow"] == 5
    assert "exceeding max_slice_count=3" in report.execution["cardinality"]["reason"]
    action = report.execution["cardinality"]["actions"][0]
    assert action["action"] == "bucket-overflow"
    assert action["top_overflow_slices"] == [
        {"label": "object_category=rare-0", "count": 1},
        {"label": "object_category=rare-1", "count": 1},
    ]
    overflow = report.slice("__bucket__=overflow")
    assert overflow is not None
    assert len(overflow.scenario_ids) == 1
    assert overflow.quality_stats["scenario_ids_omitted"] == 4

    fetched = lake.distributions.get_report(report.report_id)
    assert fetched.execution["cardinality"]["slice_count"] == 3


def test_distribution_measure_buckets_rare_slices_and_explicit_bins(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    scenarios = lake.table("scenarios")
    scenarios.add(
        pa.Table.from_pylist(
            [
                _scenario(
                    "scn-red-bolt",
                    run_id="run-a",
                    start_time_ns=101,
                    object_category="red-bolt",
                    embedding=[0.1, 0.2, 0.3, 0.4],
                ),
                _scenario(
                    "scn-blue-bolt",
                    run_id="run-a",
                    start_time_ns=102,
                    object_category="blue-bolt",
                    embedding=[0.1, 0.2, 0.3, 0.4],
                ),
            ],
            schema=scenarios.schema,
        )
    )
    spec = lake.distributions.define(name="object-coverage", dimensions=["object_category"])

    report = lake.distributions.measure(
        spec,
        rare_slice_min_count=2,
        slice_bins={
            "object_category=red-bolt": "object_category=bolt",
            "object_category=blue-bolt": "object_category=bolt",
        },
    )

    assert report.slice_counts["object_category=bolt"] == 2
    assert "__bucket__=rare" not in report.slice_counts
    assert report.execution["cardinality"]["actions"] == []


def test_distribution_related_summaries_filter_to_selected_source_rows(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.table("labels").add(
        pa.Table.from_pylist(
            [
                {
                    "label_id": "lbl-anchor",
                    "run_id": "run-a",
                    "observation_id": "",
                    "scenario_id": "scn-anchor",
                    "event_id": "",
                    "label_type": "object",
                    "label": "cup",
                    "label_value": "{}",
                    "label_spec": "fixture",
                    "source": "human",
                    "reviewer": "qa",
                    "confidence": 1.0,
                    "status": "accepted",
                    "metadata": [],
                    "transform_id": "tfm-label-anchor",
                    "created_at": NOW,
                },
                {
                    "label_id": "lbl-unrelated",
                    "run_id": "run-b",
                    "observation_id": "",
                    "scenario_id": "scn-site-b-box",
                    "event_id": "",
                    "label_type": "object",
                    "label": "box",
                    "label_value": "{}",
                    "label_spec": "fixture",
                    "source": "human",
                    "reviewer": "qa",
                    "confidence": 1.0,
                    "status": "accepted",
                    "metadata": [],
                    "transform_id": "tfm-label-unrelated",
                    "created_at": NOW,
                },
            ],
            schema=LABELS_SCHEMA,
        )
    )
    lake.table("curation_memberships").add(
        pa.Table.from_pylist(
            [
                {
                    "membership_id": "m-anchor",
                    "view_id": "dedup",
                    "target_grain": "scenario",
                    "target_id": "scn-anchor",
                    "scenario_id": "scn-anchor",
                    "decision": "exclude",
                    "reason_code": "duplicate",
                    "reason": "near duplicate",
                    "note": "",
                    "reviewer": "",
                    "queue": "",
                    "priority": 0,
                    "score": 0.99,
                    "metadata": [],
                    "source": "dedup",
                    "supersedes_membership_id": "",
                    "created_by": "fixture",
                    "transform_id": "tfm-dedup-anchor",
                    "created_at": NOW,
                },
                {
                    "membership_id": "m-unrelated",
                    "view_id": "dedup",
                    "target_grain": "scenario",
                    "target_id": "scn-site-b-box",
                    "scenario_id": "scn-site-b-box",
                    "decision": "exclude",
                    "reason_code": "duplicate",
                    "reason": "near duplicate",
                    "note": "",
                    "reviewer": "",
                    "queue": "",
                    "priority": 0,
                    "score": 0.99,
                    "metadata": [],
                    "source": "dedup",
                    "supersedes_membership_id": "",
                    "created_by": "fixture",
                    "transform_id": "tfm-dedup-unrelated",
                    "created_at": NOW,
                },
            ],
            schema=CURATION_MEMBERSHIPS_SCHEMA,
        )
    )
    spec = lake.distributions.define(
        name="selected-object-coverage",
        dimensions=["object_category"],
    )

    report = lake.distributions.measure(
        spec,
        source={"kind": "scenarios", "scenario_ids": ["scn-anchor"]},
    )

    cup = report.slice("object_category=cup")
    assert cup is not None
    assert cup.label_completeness["labeled_count"] == 1
    assert cup.duplicate_pressure["duplicate_decision_count"] == 1
    assert report.execution["related_scan"]["labels"]["rows"] == 1
    assert report.execution["related_scan"]["curation_memberships"]["rows"] == 1
    assert report.execution["related_scan"]["labels"]["filter"] == (
        "selected-scenario-or-observation-ids"
    )


def test_distributions_cli_measure_exposes_scalable_controls(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    scenarios = lake.table("scenarios")
    scenarios.add(
        pa.Table.from_pylist(
            [
                _scenario(
                    f"scn-cli-rare-{index}",
                    run_id="run-a",
                    start_time_ns=200 + index,
                    object_category=f"cli-rare-{index}",
                    embedding=[0.1, 0.2, 0.3, 0.4],
                )
                for index in range(3)
            ],
            schema=scenarios.schema,
        )
    )

    result = runner.invoke(
        app,
        [
            "gaps",
            "measure",
            "--lake",
            str(lake_path),
            "--dimension",
            "object_category",
            "--max-slice-count",
            "3",
            "--overflow",
            "bucket",
            "--max-scenario-ids-per-slice",
            "1",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["execution"]["cardinality"]["slice_count"] == 3
    assert payload["slice_counts"]["__bucket__=overflow"] == 3
