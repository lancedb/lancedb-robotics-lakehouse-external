"""CLI tests for `lancedb-robotics bench run`."""

import json
from types import SimpleNamespace

from test_benchmark import benchmark_lake
from typer.testing import CliRunner

from lancedb_robotics.cli import app

runner = CliRunner()


def test_bench_run_writes_report_and_prints_skips(tmp_path):
    lake_path = tmp_path / "robot.lance"
    benchmark_lake(lake_path)
    report_path = tmp_path / "bench-report.json"
    artifact_dir = tmp_path / "bench-artifacts"

    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--lake",
            str(lake_path),
            "--snapshot",
            "bench-v1",
            "--formats",
            "lance,lerobot-default,deeplake",
            "--sample-limit",
            "2",
            "--random-access-samples",
            "2",
            "--random-frame-samples",
            "2",
            "--frames-per-clip",
            "3",
            "--query-limit",
            "1",
            "--source-dataset-id",
            "lerobot/droid_100",
            "--source-dataset-revision",
            "abc123",
            "--source-dataset-uri",
            "hf://lerobot/droid_100",
            "--source-size-tier",
            "droid-100",
            "--storage-tier",
            "hf-cache",
            "--lerobot-native-source-mode",
            "projection",
            "--lerobot-native-cache-mode",
            "cache-only",
            "--lerobot-native-episode-limit",
            "2",
            "--out",
            str(report_path),
            "--artifacts",
            str(artifact_dir),
        ],
    )

    assert result.exit_code == 0
    assert "format lance: completed" in result.output
    assert "source dataset: lerobot/droid_100 tier=droid-100 storage=hf-cache" in result.output
    assert "random frames:" in result.output
    assert "format lerobot-default: completed" in result.output
    assert "format deeplake: skipped" in result.output
    assert f"report: {report_path}" in result.output

    report = json.loads(report_path.read_text())
    assert report["formats"]["lance"]["status"] == "completed"
    assert report["dataset"]["source_corpus"]["dataset_id"] == "lerobot/droid_100"
    assert report["dataset"]["source_corpus"]["revision"] == "abc123"
    assert report["storage_tiers"]["active"] == "hf-cache"
    assert report["params"]["lerobot_native_source_mode"] == "projection"
    assert report["params"]["lerobot_native_cache_mode"] == "cache-only"
    assert report["params"]["lerobot_native_episode_limit"] == 2
    assert {
        tier["tier"]
        for tier in report["storage_tiers"]["tiers"]
        if tier["status"] == "skipped"
    } == {"local", "object-store"}
    assert report["comparison_table"][0]["format"] == "lance"
    assert report["formats"]["lance"]["metrics"]["random_frame_sampling"]["details"][
        "frames_per_clip"
    ] == 3
    assert report["formats"]["lerobot-default"]["status"] == "completed"
    assert report["formats"]["deeplake"]["status"] == "skipped"
    assert report["formats"]["deeplake"]["skip_reason"]
    assert report["formats"]["deeplake"]["notes"]
    assert (artifact_dir / "lerobot-default" / "dataset_export_manifest.json").exists()


def test_bench_run_enterprise_fixture_writes_phase_report(tmp_path):
    lake_path = tmp_path / "robot.lance"
    benchmark_lake(lake_path)
    report_path = tmp_path / "enterprise-report.json"

    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--lake",
            str(lake_path),
            "--snapshot",
            "bench-v1",
            "--formats",
            "enterprise-lance",
            "--sample-limit",
            "2",
            "--random-access-samples",
            "2",
            "--enterprise-fixture-uri",
            "db://robotics-benchmark",
            "--out",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "format enterprise-lance: completed" in result.output
    assert "cold cache:" in result.output
    assert "warm second epoch:" in result.output

    report = json.loads(report_path.read_text())
    enterprise = report["formats"]["enterprise-lance"]
    assert enterprise["status"] == "completed"
    assert enterprise["remote_endpoint"]["profile"] == "fake-local-db"
    assert enterprise["phases"]["cold_cache"]["cache"]["misses"] > 0
    assert enterprise["metrics"]["subset_filter_change"]["details"][
        "materialized_bytes_written"
    ] == 0


def test_bench_run_enterprise_live_without_live_lake_exits_with_diagnostic(tmp_path):
    lake_path = tmp_path / "robot.lance"
    benchmark_lake(lake_path)

    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--lake",
            str(lake_path),
            "--snapshot",
            "bench-v1",
            "--formats",
            "enterprise-lance",
            "--enterprise-live",
        ],
    )

    assert result.exit_code == 1
    assert "error:" in result.output
    assert "enterprise_live=True requires" in result.output
    assert "connection_kind='local_path'" in result.output


def test_bench_run_enterprise_live_and_fixture_together_exits_with_diagnostic(tmp_path):
    lake_path = tmp_path / "robot.lance"
    benchmark_lake(lake_path)

    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--lake",
            str(lake_path),
            "--snapshot",
            "bench-v1",
            "--formats",
            "enterprise-lance",
            "--enterprise-live",
            "--enterprise-fixture-uri",
            "db://robotics-benchmark",
        ],
    )

    assert result.exit_code == 1
    assert "cannot be combined with enterprise_fixture_uri" in result.output


def test_bench_run_lerobot_native_skip_is_reported(tmp_path, monkeypatch):
    import lancedb_robotics.benchmark as benchmark

    monkeypatch.setattr(
        benchmark,
        "_lerobot_native_dependency_status",
        lambda: {
            "available": False,
            "modules": ["lerobot", "torch"],
            "missing": ["lerobot", "torchcodec", "av"],
            "install": "install native stack",
            "versions": {},
            "decode_backend": {
                "selected": None,
                "available": False,
                "missing": ["torchcodec", "av"],
            },
        },
    )
    lake_path = tmp_path / "robot.lance"
    benchmark_lake(lake_path)
    report_path = tmp_path / "native-report.json"

    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--lake",
            str(lake_path),
            "--snapshot",
            "bench-v1",
            "--formats",
            "lerobot-native",
            "--out",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "format lerobot-native: skipped" in result.output
    assert "reason: optional LeRobot native dependency stack" in result.output
    report = json.loads(report_path.read_text())
    native = report["formats"]["lerobot-native"]
    assert native["status"] == "skipped"
    assert native["metrics"]["dataloader_throughput"]["status"] == "skipped"
    assert native["native_loader"]["dependency_status"]["missing"] == [
        "lerobot",
        "torchcodec",
        "av",
    ]


def test_bench_prepare_lerobot_writes_descriptor_without_network(tmp_path, monkeypatch):
    import lancedb_robotics.benchmark as benchmark

    lake_path = tmp_path / "robot.lance"
    lake = benchmark_lake(lake_path)
    report_path = tmp_path / "prepare-report.json"

    def fake_ingest(lake_arg, source, **kwargs):
        assert lake_arg.uri == lake.uri
        assert source == "lerobot/droid_100@abc123"
        assert kwargs["created_by"] == "lancedb-robotics-bench"
        return SimpleNamespace(run_id="run-bench", rows_added={"observations": 4})

    monkeypatch.setattr(benchmark, "ingest_lerobot", fake_ingest)

    result = runner.invoke(
        app,
        [
            "bench",
            "prepare-lerobot",
            "--lake",
            str(lake_path),
            "--revision",
            "abc123",
            "--snapshot",
            "droid-100-smoke",
            "--out",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "prepared: lerobot/droid_100" in result.output
    assert "snapshot: droid-100-smoke" in result.output
    assert "scenarios: 2" in result.output

    report = json.loads(report_path.read_text())
    assert report["dataset_id"] == "lerobot/droid_100"
    assert report["source_uri"] == "hf://lerobot/droid_100"
    assert report["source_ref"] == "lerobot/droid_100@abc123"
    assert report["snapshot_name"] == "droid-100-smoke"
    assert report["scenario_count"] == 2


def test_bench_run_public_lerobot_retains_artifacts_and_rebuilds_dashboard(
    tmp_path, monkeypatch
):
    import lancedb_robotics.benchmark as benchmark

    lake_path = tmp_path / "robot.lance"
    lake = benchmark_lake(lake_path)
    artifact_root = tmp_path / "public-benchmarks"

    def fake_prepare(lake_arg, source, **kwargs):
        assert lake_arg.uri == lake.uri
        assert source == "lerobot/droid_100"
        assert kwargs["revision"] == "abc123"
        return {
            "status": "completed",
            "prepared_at": "2026-01-01T00:00:00Z",
            "lake_uri": lake.uri,
            "dataset_id": "lerobot/droid_100",
            "source_uri": "hf://lerobot/droid_100",
            "source_ref": "lerobot/droid_100@abc123",
            "revision": "abc123",
            "size_tier": "droid-100",
            "storage_tier": "hf-cache",
            "source_size_bytes": None,
            "size_note": "DROID-100 smoke tier",
            "availability": {"status": "skipped", "reason": "test", "notes": ["offline"]},
            "run_id": "run-bench",
            "snapshot_name": "bench-v1",
            "snapshot_dataset_id": "dataset-bench-v1",
            "scenario_count": 2,
            "rows_added": {"observations": 4},
        }

    monkeypatch.setattr(benchmark, "prepare_lerobot_benchmark_dataset", fake_prepare)
    monkeypatch.setenv("GITHUB_SHA", "abc1234")

    result = runner.invoke(
        app,
        [
            "bench",
            "run-public-lerobot",
            "--lake",
            str(lake_path),
            "--artifact-root",
            str(artifact_root),
            "--revision",
            "abc123",
            "--report-id",
            "droid-100-ci",
            "--snapshot",
            "bench-v1",
            "--formats",
            "lance,deeplake,enterprise-lance",
            "--sample-limit",
            "2",
            "--random-access-samples",
            "2",
            "--random-frame-samples",
            "2",
            "--frames-per-clip",
            "3",
            "--query-limit",
            "1",
            "--storage-tier",
            "hf-cache",
            "--lerobot-native-source-mode",
            "source",
            "--lerobot-native-cache-mode",
            "download",
            "--lerobot-native-episode-limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "public benchmark: droid-100-ci" in result.output
    assert f"run: {artifact_root / 'runs/droid-100-ci'}" in result.output
    assert f"manifest: {artifact_root / 'runs/droid-100-ci/artifact-manifest.json'}" in result.output
    assert f"dashboard: {artifact_root / 'dashboard.md'}" in result.output
    assert "format lance: completed" in result.output
    assert "format deeplake: skipped" in result.output
    assert "format enterprise-lance: skipped" in result.output
    manifest_path = artifact_root / "runs/droid-100-ci/artifact-manifest.json"
    prepare_path = artifact_root / "runs/droid-100-ci/reports/prepare.json"
    report_path = artifact_root / "runs/droid-100-ci/reports/benchmark.json"
    index_path = artifact_root / "index.json"
    dashboard_path = artifact_root / "dashboard.md"
    assert manifest_path.exists()
    assert prepare_path.exists()
    assert report_path.exists()
    assert index_path.exists()
    assert dashboard_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["report_id"] == "droid-100-ci"
    assert manifest["commit"] == "abc1234"
    assert manifest["dataset"]["dataset_id"] == "lerobot/droid_100"
    assert manifest["dataset"]["revision"] == "abc123"
    assert manifest["dataset"]["source_uri"] == "hf://lerobot/droid_100"
    assert manifest["dataset"]["storage_tier"] == "hf-cache"
    assert manifest["capacity"]["status"] == "allowed"
    assert manifest["format_statuses"]["lance"] == "completed"
    assert manifest["format_statuses"]["deeplake"] == "skipped"
    assert manifest["format_statuses"]["enterprise-lance"] == "skipped"
    assert manifest["skipped"]["deeplake"]
    assert manifest["skipped"]["enterprise-lance"]
    assert "reports/prepare.json" in manifest["artifact_files"]
    assert "reports/benchmark.json" in manifest["artifact_files"]
    benchmark_report = json.loads(report_path.read_text())
    assert benchmark_report["params"]["lerobot_native_source_mode"] == "source"
    assert benchmark_report["params"]["lerobot_native_cache_mode"] == "download"
    assert benchmark_report["params"]["lerobot_native_episode_limit"] == 1

    index = json.loads(index_path.read_text())
    assert index["latest_report_id"] == "droid-100-ci"
    row = index["runs"][0]
    assert row["commit"] == "abc1234"
    assert row["revision"] == "abc123"
    assert row["storage_tier"] == "hf-cache"
    assert row["capacity_status"] == "allowed"
    assert row["artifact_manifest"] == "runs/droid-100-ci/artifact-manifest.json"

    dashboard_result = runner.invoke(
        app,
        [
            "bench",
            "public-dashboard",
            "--artifact-root",
            str(artifact_root),
        ],
    )

    assert dashboard_result.exit_code == 0, dashboard_result.output
    assert "runs: 1" in dashboard_result.output
    assert "latest: droid-100-ci" in dashboard_result.output
    dashboard = dashboard_path.read_text()
    assert "droid-100-ci" in dashboard
    assert "abc1234" in dashboard
    assert "abc123" in dashboard
    assert "hf-cache" in dashboard
    assert "enterprise-lance:skipped" in dashboard

    publish_root = tmp_path / "published-public-benchmarks"
    publish_result = runner.invoke(
        app,
        [
            "bench",
            "publish-public-lerobot",
            "--artifact-root",
            str(artifact_root),
            "--destination",
            str(publish_root),
            "--report-id",
            "droid-100-ci",
            "--retain-latest",
            "1",
        ],
    )

    assert publish_result.exit_code == 0, publish_result.output
    assert "publication: published" in publish_result.output
    assert f"destination: {publish_root}" in publish_result.output
    assert "reports: droid-100-ci" in publish_result.output
    assert "retention: public-benchmark-history" in publish_result.output
    assert (publish_root / "runs/droid-100-ci/artifact-manifest.json").exists()
    published_manifest = json.loads(
        (publish_root / "runs/droid-100-ci/artifact-manifest.json").read_text()
    )
    assert published_manifest["artifact_checksums"]["reports/benchmark.json"]["sha256"]
    assert published_manifest["retention"]["protected"] is True


def test_bench_run_public_lerobot_capacity_dry_run_skips_before_prepare(
    tmp_path, monkeypatch
):
    import lancedb_robotics.benchmark as benchmark

    lake_path = tmp_path / "robot.lance"
    benchmark_lake(lake_path)
    artifact_root = tmp_path / "public-benchmarks"

    def fail_prepare(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("capacity dry-run should not prepare the source")

    monkeypatch.setattr(benchmark, "prepare_lerobot_benchmark_dataset", fail_prepare)
    monkeypatch.setenv("GITHUB_SHA", "capacity123")

    result = runner.invoke(
        app,
        [
            "bench",
            "run-public-lerobot",
            "--lake",
            str(lake_path),
            "--artifact-root",
            str(artifact_root),
            "--revision",
            "abc123",
            "--report-id",
            "mid-dry-run",
            "--formats",
            "lance,lerobot-default",
            "--size-tier",
            "mid",
            "--storage-tier",
            "hf-cache",
            "--capacity-dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "public benchmark: mid-dry-run" in result.output
    assert "status: dry-run" in result.output
    assert "capacity status: skipped" in result.output
    assert "capacity reason:" in result.output
    assert "capacity tier droid-100: allowed" in result.output
    assert "capacity tier mid selected: skipped" in result.output
    assert "prepare:" not in result.output
    assert "\nbenchmark:" not in result.output
    assert f"capacity: {artifact_root / 'runs/mid-dry-run/reports/capacity.json'}" in result.output

    manifest_path = artifact_root / "runs/mid-dry-run/artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "dry-run"
    assert manifest["capacity"]["dry_run"] is True
    assert manifest["capacity"]["status"] == "skipped"
    assert manifest["format_statuses"]["lance"] == "skipped"
    assert not (artifact_root / "runs/mid-dry-run/reports/prepare.json").exists()


def test_bench_validate_public_lerobot_claims(tmp_path, monkeypatch):
    import lancedb_robotics.benchmark as benchmark

    lake_path = tmp_path / "robot.lance"
    lake = benchmark_lake(lake_path)
    artifact_root = tmp_path / "public-benchmarks"

    def fake_prepare(lake_arg, source, **kwargs):
        assert lake_arg.uri == lake.uri
        assert source == "lerobot/droid_100"
        return {
            "status": "completed",
            "prepared_at": "2026-01-01T00:00:00Z",
            "lake_uri": lake.uri,
            "dataset_id": "lerobot/droid_100",
            "source_uri": "hf://lerobot/droid_100",
            "source_ref": "lerobot/droid_100@abc123",
            "revision": "abc123",
            "size_tier": "droid-100",
            "storage_tier": "hf-cache",
            "source_size_bytes": None,
            "size_note": "DROID-100 smoke tier",
            "availability": {"status": "skipped", "reason": "test", "notes": ["offline"]},
            "run_id": "run-bench",
            "snapshot_name": kwargs["snapshot_name"],
            "snapshot_dataset_id": "dataset-bench-v1",
            "scenario_count": 2,
            "rows_added": {"observations": 4},
        }

    monkeypatch.setattr(benchmark, "prepare_lerobot_benchmark_dataset", fake_prepare)
    monkeypatch.setenv("GITHUB_SHA", "claim1234")
    run_result = runner.invoke(
        app,
        [
            "bench",
            "run-public-lerobot",
            "--lake",
            str(lake_path),
            "--artifact-root",
            str(artifact_root),
            "--revision",
            "abc123",
            "--report-id",
            "droid-100-claim",
            "--snapshot",
            "bench-v1",
            "--formats",
            "lance,deeplake",
            "--sample-limit",
            "2",
            "--random-access-samples",
            "2",
            "--random-frame-samples",
            "2",
            "--frames-per-clip",
            "3",
            "--query-limit",
            "1",
            "--storage-tier",
            "hf-cache",
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    claims_path = tmp_path / "claims.json"
    claims_path.write_text(
        json.dumps(
            {
                "schema_version": benchmark.PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION,
                "claims": [
                    {
                        "id": "docs-lance-throughput",
                        "report_id": "droid-100-claim",
                        "commit": "claim1234",
                        "dataset_revision": "abc123",
                        "storage_tier": "hf-cache",
                        "format": "lance",
                        "metric": "dataloader_throughput",
                    }
                ],
            }
        )
    )
    report_path = tmp_path / "validation.json"
    result = runner.invoke(
        app,
        [
            "bench",
            "validate-public-lerobot",
            "--artifact-root",
            str(artifact_root),
            "--claims",
            str(claims_path),
            "--out",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "validation: passed" in result.output
    assert "reports: 1" in result.output
    assert "claims: 1" in result.output
    saved = json.loads(report_path.read_text())
    assert saved["status"] == "passed"

    claims_path.write_text(
        json.dumps(
            {
                "schema_version": benchmark.PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION,
                "claims": [
                    {
                        "id": "bad-deeplake-throughput",
                        "report_id": "droid-100-claim",
                        "commit": "claim1234",
                        "dataset_revision": "abc123",
                        "storage_tier": "hf-cache",
                        "format": "deeplake",
                        "metric": "dataloader_throughput",
                    }
                ],
            }
        )
    )
    failed = runner.invoke(
        app,
        [
            "bench",
            "validate-public-lerobot",
            "--artifact-root",
            str(artifact_root),
            "--claims",
            str(claims_path),
        ],
    )

    assert failed.exit_code == 1
    assert "validation: failed" in failed.output
    assert "claim-format-status-mismatch" in failed.output


def test_bench_run_unknown_snapshot_exits_one(tmp_path):
    lake_path = tmp_path / "robot.lance"
    benchmark_lake(lake_path)

    result = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--lake",
            str(lake_path),
            "--snapshot",
            "ghost",
            "--formats",
            "lance",
        ],
    )

    assert result.exit_code == 1
    assert "no snapshot named" in result.output
