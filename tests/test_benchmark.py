"""Reproducible benchmark harness tests (backlog 0034)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from lancedb_robotics.benchmark import (
    BENCHMARK_METRICS,
    DEEPLAKE_FORMAT,
    DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
    DEFAULT_LEROBOT_BENCHMARK_SOURCE_URI,
    ENTERPRISE_LANCE_FORMAT,
    FAKE_LOCAL_DB_PROFILE,
    ICEBERG_FORMAT,
    LANCE_FORMAT,
    LEROBOT_DEFAULT_FORMAT,
    LEROBOT_NATIVE_FORMAT,
    LIVE_DB_PROFILE,
    LIVE_NAMESPACE_PROFILE,
    METRIC_METADATA_SCAN_LATENCY,
    METRIC_RANDOM_FRAME_SAMPLING,
    METRIC_ROW_HYDRATION_LATENCY,
    METRIC_SHUFFLED_EPOCH_THROUGHPUT,
    METRIC_SUBSET_FILTER_CHANGE,
    PARQUET_FORMAT,
    PAYLOAD_PLACEMENT_INLINE,
    PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION,
    WEBDATASET_FORMAT,
    BenchmarkError,
    compare_enterprise_benchmark_results,
    plan_public_lerobot_benchmark_capacity,
    prepare_lerobot_benchmark_dataset,
    publish_public_lerobot_benchmark,
    run_benchmark_suite,
    run_public_lerobot_benchmark,
    validate_public_lerobot_benchmark_claims,
    write_benchmark_report,
)
from lancedb_robotics.connections import LakeCapabilities, LakeConnectionSpec
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA, SCENARIOS_SCHEMA
from lancedb_robotics.training_report_schema import scan_report_secrets


def benchmark_lake(path):
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-bench",
                    "run_kind": "demo",
                    "source": "synthetic",
                    "source_id": "src-bench",
                    "raw_uri": "memory://run-bench",
                    "robot_id": "robot-bench",
                    "site_id": "lab",
                    "task_id": "push the block",
                    "start_time_ns": 0,
                    "end_time_ns": 3_000_000_000,
                    "duration_ns": 3_000_000_000,
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
    observations = []
    for index in range(4):
        observations.append(
            {
                "observation_id": f"obs-bench-{index}",
                "run_id": "run-bench",
                "timestamp_ns": index * 1_000_000_000,
                "sensor_id": "camera_front",
                "topic": "/camera/front",
                "modality": "image",
                "raw_uri": "memory://run-bench",
                "raw_channel": "/camera/front",
                "raw_log_time_ns": index * 1_000_000_000,
                "raw_sequence": index,
                "payload_json": None,
                "payload_blob": f"frame-{index}".encode(),
                "message_encoding": "jpeg",
                "schema_encoding": "jpeg",
                "decode_status": "decoded",
                "decode_error": "",
                "state_vector": [float(index), float(index) + 0.5],
                "action_vector": [10.0 + index, -10.0 - index],
                "caption": f"frame {index}",
                "quality_flags": [],
                "transform_id": "tfm-ingest",
                "created_at": now,
            }
        )
    lake.table("observations").add(pa.Table.from_pylist(observations, schema=OBSERVATIONS_SCHEMA))
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-bench-a",
                    "run_id": "run-bench",
                    "start_time_ns": 0,
                    "end_time_ns": 1_000_000_000,
                    "window_ns": 1_000_000_000,
                    "is_partial": False,
                    "topics": ["/camera/front"],
                    "observation_ids": ["obs-bench-0", "obs-bench-1"],
                    "observation_count": 2,
                    "scenario_type": "demo",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["camera"],
                    "summary": "push the block start",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
                {
                    "scenario_id": "scn-bench-b",
                    "run_id": "run-bench",
                    "start_time_ns": 2_000_000_000,
                    "end_time_ns": 3_000_000_000,
                    "window_ns": 1_000_000_000,
                    "is_partial": False,
                    "topics": ["/camera/front"],
                    "observation_ids": ["obs-bench-2", "obs-bench-3"],
                    "observation_count": 2,
                    "scenario_type": "demo",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["camera"],
                    "summary": "push the block finish",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                },
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    create_snapshot(
        lake,
        name="bench-v1",
        scenario_ids=["scn-bench-a", "scn-bench-b"],
        split_by="scenario",
    )
    return lake


@pytest.fixture
def lake(tmp_path):
    return benchmark_lake(tmp_path / "robot.lance")


def _apply_live_enterprise_connection(
    lake,
    *,
    kind="lancedb_remote_db",
    uri="db://robotics-live",
    host_override="https://phalanx.acme.internal",
    region="us-west-2",
    api_key="super-secret-live-api-key",
    server_side_query=True,
    blob_fetch_remote=True,
):
    """Simulate a lake already opened against a live Enterprise endpoint.

    Mirrors the connection-spec-swap trick ``_fake_enterprise_lake`` and
    ``enterprise_conformance._apply_case`` use: the lake keeps reading its real
    local Lance tables, but the benchmark harness sees a live ``db://`` or
    namespace connection report -- exactly what backlog 0125's live-mode
    preflight and profile labeling need to prove, without a real remote server.
    """
    lake.connection_spec = LakeConnectionSpec(
        kind=kind,
        uri=uri,
        display_uri=uri,
        lancedb_connect_kwargs={
            "region": region,
            "host_override": host_override,
            "api_key": api_key,
        },
        auth_refs={"remote": "enterprise-prod"},
        direct_object_io_allowed=False,
        capabilities=LakeCapabilities(
            server_side_query=server_side_query,
            direct_object_io=False,
            namespace_resolution=kind != "lancedb_remote_db",
            geneva_worker_specs=False,
            blob_fetch_remote=blob_fetch_remote,
        ),
    )
    # Lake.__init__ normally derives `.capabilities` from the connection spec it
    # was opened with; swapping `.connection_spec` on an already-open lake must
    # update `.capabilities` too, or the real capability resolver
    # (training._lake_capabilities_dict) keeps reading the stale local-lake value.
    lake.capabilities = lake.connection_spec.capabilities
    return lake


def test_benchmark_runs_lance_metrics_and_lerobot_comparison(lake, tmp_path):
    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LANCE_FORMAT, LEROBOT_DEFAULT_FORMAT, WEBDATASET_FORMAT],
        output_dir=tmp_path / "artifacts",
        sample_limit=2,
        random_access_samples=3,
        random_frame_samples=2,
        frames_per_clip=3,
        seed=7,
        query_limit=1,
        source_dataset_id=DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
        source_dataset_revision="abc123",
        source_dataset_uri=DEFAULT_LEROBOT_BENCHMARK_SOURCE_URI,
        source_dataset_size_tier="droid-100",
        storage_tier="hf-cache",
    )

    assert report["dataset"]["snapshot_name"] == "bench-v1"
    assert report["dataset"]["scenario_count"] == 2
    source = report["dataset"]["source_corpus"]
    assert source["dataset_id"] == "lerobot/droid_100"
    assert source["revision"] == "abc123"
    assert source["source_uri"] == "hf://lerobot/droid_100"
    assert source["size_tier"] == "droid-100"
    assert source["storage_tier"] == "hf-cache"
    assert source["public_default_dataset_id"] == "lerobot/droid_100"
    assert report["storage_tiers"]["active"] == "hf-cache"
    skipped_tiers = {
        tier["tier"]: tier
        for tier in report["storage_tiers"]["tiers"]
        if tier["status"] == "skipped"
    }
    assert "local" in skipped_tiers
    assert "object-store" in skipped_tiers
    assert report["params"]["seed"] == 7
    assert report["params"]["random_frame_samples"] == 2
    assert report["params"]["frames_per_clip"] == 3
    assert report["hardware"]["cpu_count"]
    assert report["format_versions"][LANCE_FORMAT]["lancedb"]
    assert report["methodology"]["deterministic"] is True
    table_by_format = {row["format"]: row for row in report["comparison_table"]}
    assert table_by_format[LANCE_FORMAT][METRIC_RANDOM_FRAME_SAMPLING] > 0
    assert table_by_format[LEROBOT_DEFAULT_FORMAT]["dataloader_throughput"] > 0

    lance = report["formats"][LANCE_FORMAT]
    assert lance["status"] == "completed"
    assert set(lance["metrics"]) == set(BENCHMARK_METRICS)
    assert lance["metrics"]["dataloader_throughput"]["details"]["samples"] == 2
    assert lance["metrics"][METRIC_RANDOM_FRAME_SAMPLING]["details"]["clips"] == 2
    assert lance["metrics"][METRIC_RANDOM_FRAME_SAMPLING]["details"]["frames_per_clip"] == 3
    assert (
        lance["metrics"]["query_to_dataset_curation"]["details"]["selected_scenarios"] == 1
    )
    assert "gpu_utilization_pct" in lance["metrics"]["dataloader_throughput"]["details"]

    lerobot = report["formats"][LEROBOT_DEFAULT_FORMAT]
    assert lerobot["status"] == "completed"
    assert set(lerobot["metrics"]) == set(BENCHMARK_METRICS)
    assert lerobot["metrics"][METRIC_RANDOM_FRAME_SAMPLING]["status"] == "completed"
    assert lerobot["metrics"][METRIC_RANDOM_FRAME_SAMPLING]["details"]["frames"] == 6
    assert lerobot["projection_manifest"]["format"] == "lerobot"
    assert lerobot["projection_manifest"]["step_count"] == 4

    webdataset = report["formats"][WEBDATASET_FORMAT]
    assert webdataset["status"] == "completed"
    assert set(webdataset["metrics"]) == set(BENCHMARK_METRICS)
    assert webdataset["metrics"][METRIC_RANDOM_FRAME_SAMPLING]["details"]["clips"] == 2
    assert webdataset["projection_manifest"]["format"] == "webdataset"
    assert webdataset["projection_manifest"]["step_count"] == 4
    assert webdataset["projection_manifest"]["native_loader"]["modules"] == ["webdataset"]
    assert (
        webdataset["metrics"]["storage_footprint"]["details"][
            "materialized_bytes_written"
        ]
        > 0
    )


def test_parquet_baseline_materializes_same_sample_set_and_splits_scan_from_hydration(
    lake, tmp_path
):
    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LANCE_FORMAT, WEBDATASET_FORMAT, PARQUET_FORMAT],
        output_dir=tmp_path / "artifacts",
        sample_limit=2,
        random_access_samples=3,
        random_frame_samples=2,
        frames_per_clip=3,
        seed=7,
        query_limit=1,
    )

    parquet = report["formats"][PARQUET_FORMAT]
    assert parquet["status"] == "completed"
    assert parquet["payload_placement"] == PAYLOAD_PLACEMENT_INLINE

    # Same logical sample set as the Lance and WebDataset arms: the snapshot's four
    # observations across its two scenarios.
    assert parquet["table_manifest"]["rows"] == 4
    assert report["formats"][WEBDATASET_FORMAT]["projection_manifest"]["step_count"] == 4

    metrics = parquet["metrics"]
    # Standard comparison metrics are present so parquet appears in the table.
    assert set(BENCHMARK_METRICS).issubset(metrics)
    # Analytics-specific metrics separate metadata scan from payload hydration.
    for key in (
        METRIC_METADATA_SCAN_LATENCY,
        METRIC_ROW_HYDRATION_LATENCY,
        METRIC_SHUFFLED_EPOCH_THROUGHPUT,
        METRIC_SUBSET_FILTER_CHANGE,
    ):
        assert key in metrics

    scan = metrics[METRIC_METADATA_SCAN_LATENCY]
    assert scan["status"] == "completed"
    assert scan["details"]["read_payload"] is False
    assert scan["details"]["num_row_groups"] >= 1

    hydration = metrics[METRIC_ROW_HYDRATION_LATENCY]
    assert hydration["status"] == "completed"
    assert hydration["details"]["payload_bytes_materialized"] > 0

    storage = metrics["storage_footprint"]
    assert storage["details"]["materialized_bytes_written"] > 0
    assert storage["details"]["payload_placement"] == PAYLOAD_PLACEMENT_INLINE

    table_by_format = {row["format"]: row for row in report["comparison_table"]}
    assert table_by_format[PARQUET_FORMAT]["dataloader_throughput"] >= 0
    assert report["format_versions"][PARQUET_FORMAT]["layout"] == "parquet-analytics-table-v0"
    assert (artifact_dir_table := tmp_path / "artifacts" / "parquet" / "table.parquet").exists()
    assert artifact_dir_table.stat().st_size > 0


def test_parquet_filter_change_records_rewrite_cost(lake, tmp_path):
    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[PARQUET_FORMAT],
        output_dir=tmp_path / "artifacts",
        sample_limit=2,
    )
    filter_change = report["formats"][PARQUET_FORMAT]["metrics"][METRIC_SUBSET_FILTER_CHANGE]
    assert filter_change["status"] == "completed"
    details = filter_change["details"]
    assert details["requires_full_rewrite"] is True
    assert details["rewritten_bytes"] > 0
    assert details["materialized_bytes_written"] == details["rewritten_bytes"]
    assert details["selected_rows"] == 2
    assert "row plan" in details["contrast"]
    # The rewrite produced a distinct materialized artifact.
    assert (tmp_path / "artifacts" / "parquet" / "table-filtered.parquet").exists()


def test_iceberg_baseline_skipped_when_dependency_absent(lake, tmp_path, monkeypatch):
    import lancedb_robotics.benchmark as benchmark

    # Force the optional dependency to look absent regardless of the environment.
    monkeypatch.setattr(
        benchmark,
        "_module_available",
        lambda module: False if module == benchmark.ICEBERG_MODULE else True,
    )
    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[ICEBERG_FORMAT],
        output_dir=tmp_path / "artifacts",
    )
    iceberg = report["formats"][ICEBERG_FORMAT]
    assert iceberg["status"] == "skipped"
    assert "pyiceberg" in iceberg["skip_reason"]
    assert iceberg["iceberg"]["available"] is False
    assert iceberg["iceberg"]["install"]
    # Every standard metric is explicitly skipped so the report cannot be read as
    # Iceberg coverage.
    assert set(iceberg["metrics"]) == set(BENCHMARK_METRICS)
    assert all(
        metric["status"] == "skipped" for metric in iceberg["metrics"].values()
    )
    table_by_format = {row["format"]: row for row in report["comparison_table"]}
    assert table_by_format[ICEBERG_FORMAT]["status"] == "skipped"


def test_iceberg_baseline_skips_without_catalog_when_ephemeral(lake, monkeypatch):
    import lancedb_robotics.benchmark as benchmark

    # pyiceberg present but no catalog configured and no artifact dir to root a
    # local catalog -> structured skip rather than a misleading empty run.
    monkeypatch.setattr(benchmark, "_module_available", lambda module: True)
    monkeypatch.delenv(benchmark.ICEBERG_CATALOG_URI_ENV, raising=False)
    report = run_benchmark_suite(lake, "bench-v1", formats=[ICEBERG_FORMAT])
    iceberg = report["formats"][ICEBERG_FORMAT]
    assert iceberg["status"] == "skipped"
    assert "catalog" in iceberg["skip_reason"]


def test_enterprise_remote_benchmark_reports_cache_phases_and_filter_change(
    lake,
    tmp_path,
):
    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[ENTERPRISE_LANCE_FORMAT, WEBDATASET_FORMAT],
        output_dir=tmp_path / "artifacts",
        sample_limit=2,
        random_access_samples=2,
        seed=11,
        query_limit=1,
        enterprise_fixture_uri="db://robotics-benchmark",
    )

    enterprise = report["formats"][ENTERPRISE_LANCE_FORMAT]
    assert enterprise["status"] == "completed"
    assert enterprise["format"] == ENTERPRISE_LANCE_FORMAT
    assert enterprise["remote_endpoint"]["uri"] == "db://robotics-benchmark"
    assert enterprise["remote_endpoint"]["profile"] == "fake-local-db"
    assert enterprise["dataset_manifest"]["access_pattern"] == "enterprise-remote-snapshot"

    assert set(enterprise["phases"]) == {
        "cold_cache",
        "prewarmed",
        "warm_second_epoch",
    }
    cold = enterprise["phases"]["cold_cache"]
    prewarmed = enterprise["phases"]["prewarmed"]
    warm = enterprise["phases"]["warm_second_epoch"]
    assert cold["cache"]["misses"] > 0
    assert cold["cache"]["hits"] == 0
    assert prewarmed["prewarm"]["status"] == "complete"
    assert prewarmed["cache"]["hits"] > cold["cache"]["hits"]
    assert warm["cache"]["misses"] == 0
    assert warm["metrics"]["shuffled_epoch_throughput"]["details"]["samples"] == 2
    assert warm["loader_report"]["remote_execution"]["resolved_backend"] == "enterprise"

    metrics = enterprise["metrics"]
    assert metrics["query_to_first_batch_latency"]["status"] == "completed"
    assert metrics["shuffled_epoch_throughput"]["status"] == "completed"
    assert metrics["random_access_latency"]["details"]["samples"] == 2
    assert metrics["subset_filter_change"]["details"]["base_row_plan_id"]
    assert metrics["subset_filter_change"]["details"]["filtered_row_plan_id"]
    assert metrics["subset_filter_change"]["details"]["materialized_bytes_written"] == 0
    assert metrics["storage_footprint"]["details"]["materialized_bytes_written"] == 0
    assert metrics["storage_footprint"]["details"]["payload_bytes_hydrated"] > 0

    webdataset = report["formats"][WEBDATASET_FORMAT]
    assert webdataset["metrics"]["storage_footprint"]["details"][
        "materialized_bytes_written"
    ] > 0


def test_live_enterprise_benchmark_records_live_profile_without_credentials(lake):
    _apply_live_enterprise_connection(lake)

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[ENTERPRISE_LANCE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        seed=11,
        enterprise_live=True,
    )

    enterprise = report["formats"][ENTERPRISE_LANCE_FORMAT]
    assert enterprise["status"] == "completed"
    endpoint = enterprise["remote_endpoint"]
    assert endpoint["profile"] == LIVE_DB_PROFILE
    assert endpoint["confidence"] == "production-calibrated"
    assert endpoint["region"] == "us-west-2"
    assert endpoint["host_override"] == "https://phalanx.acme.internal"
    assert endpoint["software_versions"]["lancedb-robotics"]
    assert {check["name"] for check in endpoint["capability_checks"]} >= {
        "remote_scan",
        "plan_executor_cache_metrics",
        "page_cache_prewarm",
    }
    assert endpoint["degraded_capabilities"] == []
    assert enterprise["confidence"] == "production-calibrated"
    assert enterprise["degraded_phases"] == []

    # Credential-shaped endpoint data must never reach the benchmark report,
    # even though the live connection's connect kwargs carry a real-looking key.
    serialized = json.dumps(report)
    assert "super-secret-live-api-key" not in serialized
    assert "api_key" not in json.dumps(endpoint)
    assert scan_report_secrets(report) == []


def test_live_enterprise_benchmark_namespace_connection_labels_profile(lake):
    _apply_live_enterprise_connection(
        lake,
        kind="rest_namespace_lancedb",
        uri="namespace://robotics-live",
        host_override=None,
    )

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[ENTERPRISE_LANCE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        enterprise_live=True,
    )

    endpoint = report["formats"][ENTERPRISE_LANCE_FORMAT]["remote_endpoint"]
    assert endpoint["profile"] == LIVE_NAMESPACE_PROFILE
    assert endpoint["confidence"] == "production-calibrated"


def test_live_enterprise_benchmark_missing_cache_metrics_marks_phases_degraded(lake):
    _apply_live_enterprise_connection(lake)
    lake.enterprise_training_capabilities = {"plan_executor_cache_metrics": False}

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[ENTERPRISE_LANCE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        enterprise_live=True,
    )

    enterprise = report["formats"][ENTERPRISE_LANCE_FORMAT]
    assert enterprise["status"] == "completed"
    assert "plan_executor_cache_metrics" in enterprise["remote_endpoint"]["degraded_capabilities"]
    assert set(enterprise["degraded_phases"]) == {"cold_cache", "prewarmed", "warm_second_epoch"}
    cold = enterprise["phases"]["cold_cache"]
    assert cold["status"] == "degraded"
    assert cold["cache"]["hits"] is None
    assert cold["cache"]["misses"] is None
    assert "unavailable, not zero" in cold["degraded_reasons"][0]
    storage = enterprise["metrics"]["storage_footprint"]["details"]
    assert storage["cache_hits"] is None
    assert storage["cache_misses"] is None


def test_live_enterprise_benchmark_missing_prewarm_marks_prewarmed_phase_degraded(lake):
    _apply_live_enterprise_connection(lake)
    lake.enterprise_training_capabilities = {"page_cache_prewarm": False}

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[ENTERPRISE_LANCE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        enterprise_live=True,
    )

    enterprise = report["formats"][ENTERPRISE_LANCE_FORMAT]
    assert "page_cache_prewarm" in enterprise["remote_endpoint"]["degraded_capabilities"]
    assert enterprise["phases"]["prewarmed"]["status"] == "degraded"
    assert "rejected or degraded" in enterprise["phases"]["prewarmed"]["degraded_reasons"][0]


def test_live_enterprise_benchmark_without_server_side_query_is_skipped(lake):
    _apply_live_enterprise_connection(lake, server_side_query=False)

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[ENTERPRISE_LANCE_FORMAT],
        enterprise_live=True,
    )

    enterprise = report["formats"][ENTERPRISE_LANCE_FORMAT]
    assert enterprise["status"] == "skipped"
    assert "remote_scan" in enterprise["skip_reason"]


def test_enterprise_live_without_live_connection_raises_helpful_diagnostic(lake):
    with pytest.raises(BenchmarkError, match="enterprise_live=True requires"):
        run_benchmark_suite(
            lake,
            "bench-v1",
            formats=[ENTERPRISE_LANCE_FORMAT],
            enterprise_live=True,
        )


def test_enterprise_live_cannot_combine_with_fixture(lake):
    with pytest.raises(BenchmarkError, match="cannot be combined with enterprise_fixture_uri"):
        run_benchmark_suite(
            lake,
            "bench-v1",
            formats=[ENTERPRISE_LANCE_FORMAT],
            enterprise_fixture_uri="db://robotics-benchmark",
            enterprise_live=True,
        )


def test_compare_enterprise_benchmark_results_labels_confidence(lake):
    fake_report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[ENTERPRISE_LANCE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        seed=5,
        enterprise_fixture_uri="db://robotics-benchmark",
    )

    live_lake = benchmark_lake(Path(lake.uri).parent / "robot-live.lance")
    _apply_live_enterprise_connection(live_lake)
    live_report = run_benchmark_suite(
        live_lake,
        "bench-v1",
        formats=[ENTERPRISE_LANCE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        seed=5,
        enterprise_live=True,
    )

    calibration = compare_enterprise_benchmark_results(fake_report, live_report)
    assert calibration["fake_local"]["profile"] == FAKE_LOCAL_DB_PROFILE
    assert calibration["fake_local"]["confidence"] == "sdk-contract-only"
    assert calibration["live"]["profile"] == LIVE_DB_PROFILE
    assert calibration["live"]["confidence"] == "production-calibrated"
    assert set(calibration["metrics"]) == {
        "query_to_first_batch_latency",
        "shuffled_epoch_throughput",
        "random_access_latency",
    }
    for metric in calibration["metrics"].values():
        assert "absolute_delta" in metric
        assert metric["fake_local_value"] is not None
        assert metric["live_value"] is not None
    assert any("sdk-contract-only" in note for note in calibration["notes"])


def test_compare_enterprise_benchmark_results_rejects_two_fake_reports(lake):
    fake_report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[ENTERPRISE_LANCE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        enterprise_fixture_uri="db://robotics-benchmark",
    )

    with pytest.raises(BenchmarkError, match="live_report must use a live Enterprise profile"):
        compare_enterprise_benchmark_results(fake_report, fake_report)


def test_absent_optional_formats_are_visible(lake, monkeypatch):
    import lancedb_robotics.benchmark as benchmark
    import lancedb_robotics.dataset_export as dataset_export

    monkeypatch.setattr(benchmark, "_module_available", lambda module: False)
    monkeypatch.setattr(dataset_export, "_module_available", lambda module: False)
    monkeypatch.setattr(
        benchmark,
        "_lerobot_native_dependency_status",
        lambda: {
            "available": False,
            "modules": ["lerobot", "torch"],
            "missing": ["lerobot", "torch", "torchcodec", "av"],
            "install": "install native stack",
            "versions": {},
            "decode_backend": {
                "selected": None,
                "available": False,
                "missing": ["torchcodec", "av"],
            },
        },
    )

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=["webdataset", "deeplake", LEROBOT_NATIVE_FORMAT],
    )

    assert report["formats"]["webdataset"]["status"] == "completed"
    assert report["formats"]["webdataset"]["projection_manifest"]["native_loader"][
        "missing"
    ] == ["webdataset"]
    assert report["formats"]["deeplake"]["status"] == "skipped"
    assert "not installed" in report["formats"]["deeplake"]["skip_reason"]
    native = report["formats"][LEROBOT_NATIVE_FORMAT]
    assert native["status"] == "skipped"
    assert set(native["metrics"]) == set(BENCHMARK_METRICS)
    assert native["metrics"]["dataloader_throughput"]["status"] == "skipped"
    assert "lerobot" in native["native_loader"]["dependency_status"]["missing"]
    assert "official LeRobot loader coverage" in native["notes"][1]


def _install_fake_lerobot_native_stack(monkeypatch):
    import lancedb_robotics.benchmark as benchmark

    opened: dict[str, object] = {}

    class FakeLeRobotDataset:
        repo_id = "lerobot/droid_100"
        root = ""
        revision = "abc123"
        num_frames = 4
        num_episodes = 2
        fps = 30
        features = {
            "observation.state": {"shape": [2]},
            "action": {"shape": [2]},
        }
        meta = SimpleNamespace(camera_keys=["front"], features=features, fps=30)

        def __init__(self, repo_id, **kwargs):
            opened["repo_id"] = repo_id
            opened["kwargs"] = dict(kwargs)

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {
                "observation.state": [float(index), float(index) + 0.5],
                "action": [10.0 + index, -10.0 - index],
                "frame": f"frame-{index}".encode(),
            }

    class FakeDataLoader:
        def __init__(self, dataset, *, batch_size, shuffle, num_workers):
            assert batch_size == 1
            assert shuffle is False
            assert num_workers == 0
            self._dataset = dataset

        def __iter__(self):
            for index in range(len(self._dataset)):
                yield self._dataset[index]

    monkeypatch.setattr(
        benchmark,
        "_lerobot_native_dependency_status",
        lambda: {
            "available": True,
            "modules": ["lerobot", "torch"],
            "missing": [],
            "install": "installed",
            "versions": {
                "lerobot": "0.4.0",
                "torch": "2.9.0",
                "torchcodec": None,
                "av": "15.1.0",
            },
            "decode_backend": {
                "selected": "pyav",
                "available": True,
                "missing": [],
            },
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_load_lerobot_native_components",
        lambda: (
            benchmark._LeRobotNativeComponents(
                dataset_cls=FakeLeRobotDataset,
                dataloader_cls=FakeDataLoader,
            ),
            None,
        ),
    )
    return opened


def test_lerobot_native_format_uses_official_loader_when_available(
    lake,
    monkeypatch,
):
    opened = _install_fake_lerobot_native_stack(monkeypatch)

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LEROBOT_NATIVE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        random_frame_samples=2,
        frames_per_clip=2,
        seed=7,
        source_dataset_id="lerobot/droid_100",
        source_dataset_revision="abc123",
        source_dataset_uri="hf://lerobot/droid_100",
    )

    assert opened["repo_id"] == "lerobot/droid_100"
    assert opened["kwargs"]["revision"] == "abc123"
    assert opened["kwargs"]["download_videos"] is True
    assert opened["kwargs"]["video_backend"] == "pyav"
    assert report["format_versions"][LEROBOT_NATIVE_FORMAT]["official_api"] == (
        "lerobot.datasets.LeRobotDataset"
    )
    native = report["formats"][LEROBOT_NATIVE_FORMAT]
    assert native["status"] == "completed"
    assert native["format"] == LEROBOT_NATIVE_FORMAT
    assert set(native["metrics"]) == set(BENCHMARK_METRICS)
    assert native["metrics"]["dataloader_throughput"]["details"]["samples"] == 2
    assert native["metrics"]["random_access_latency"]["details"]["samples"] == 2
    assert native["metrics"][METRIC_RANDOM_FRAME_SAMPLING]["details"]["frames"] == 4
    assert native["native_loader"]["official_api"] == "lerobot.datasets.LeRobotDataset"
    assert native["native_loader"]["dependency_status"]["versions"]["lerobot"] == "0.4.0"
    assert native["native_loader"]["dependency_status"]["decode_backend"]["selected"] == "pyav"
    assert native["native_loader"]["source"]["kind"] == "source-corpus"
    assert native["native_loader"]["source"]["revision"] == "abc123"
    assert native["native_loader"]["source"]["resolver"]["source_mode"] == "auto"
    assert native["native_loader"]["source"]["preflight"]["implicit_hub_download"] is True
    assert native["projection_manifest"] is None


def test_lerobot_native_source_resolver_applies_episode_limit(lake, monkeypatch):
    opened = _install_fake_lerobot_native_stack(monkeypatch)

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LEROBOT_NATIVE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        random_frame_samples=2,
        frames_per_clip=2,
        source_dataset_id="lerobot/droid_100",
        source_dataset_revision="abc123",
        source_dataset_uri="hf://lerobot/droid_100",
        lerobot_native_source_mode="source",
        lerobot_native_cache_mode="download",
        lerobot_native_episode_limit=1,
    )

    assert opened["kwargs"]["episodes"] == [0]
    native = report["formats"][LEROBOT_NATIVE_FORMAT]
    source = native["native_loader"]["source"]
    assert native["status"] == "completed"
    assert source["resolver"]["source_mode"] == "source"
    assert source["resolver"]["cache_mode"] == "download"
    assert source["episode_filter"] == {
        "mode": "first-n",
        "episode_limit": 1,
        "episodes": [0],
        "applied_to_loader": True,
    }
    assert source["preflight"]["sample_budgets"]["random_frame_samples"] == 2


def test_lerobot_native_cache_only_refuses_implicit_hf_download(lake, monkeypatch):
    _install_fake_lerobot_native_stack(monkeypatch)

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LEROBOT_NATIVE_FORMAT],
        source_dataset_id="lerobot/droid_100",
        source_dataset_revision="abc123",
        source_dataset_uri="hf://lerobot/droid_100",
        lerobot_native_source_mode="source",
        lerobot_native_cache_mode="cache-only",
    )

    native = report["formats"][LEROBOT_NATIVE_FORMAT]
    source = native["native_loader"]["source"]
    assert native["status"] == "skipped"
    assert "cache-only mode requires" in native["skip_reason"]
    assert source["kind"] == "source-corpus"
    assert source["status"] == "skipped"
    assert source["resolver"]["cache_only"] is True
    assert source["preflight"]["implicit_hub_download"] is True


def test_lerobot_native_source_mode_skips_without_source_descriptor(lake, monkeypatch):
    _install_fake_lerobot_native_stack(monkeypatch)

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LEROBOT_NATIVE_FORMAT],
        lerobot_native_source_mode="source",
    )

    native = report["formats"][LEROBOT_NATIVE_FORMAT]
    source = native["native_loader"]["source"]
    assert native["status"] == "skipped"
    assert "projection fallback is disabled" in native["skip_reason"]
    assert source["kind"] == "unresolved-source"
    assert source["resolver"]["source_mode"] == "source"
    assert source["resolver"]["projection_fallback_enabled"] is False


def test_lerobot_native_projection_mode_forces_projection_fallback(
    lake,
    monkeypatch,
    tmp_path,
):
    opened = _install_fake_lerobot_native_stack(monkeypatch)

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LEROBOT_NATIVE_FORMAT],
        output_dir=tmp_path / "artifacts",
        source_dataset_id="lerobot/droid_100",
        source_dataset_revision="abc123",
        source_dataset_uri="hf://lerobot/droid_100",
        lerobot_native_source_mode="projection",
        lerobot_native_cache_mode="cache-only",
        lerobot_native_episode_limit=1,
    )

    native = report["formats"][LEROBOT_NATIVE_FORMAT]
    source = native["native_loader"]["source"]
    assert native["status"] == "completed"
    assert source["kind"] == "projection"
    assert source["resolver"]["source_mode"] == "projection"
    assert source["download_videos"] is False
    assert source["episode_filter"]["episodes"] == [0]
    assert opened["kwargs"]["root"] == source["root"]
    assert opened["kwargs"]["download_videos"] is False
    assert opened["kwargs"]["episodes"] == [0]
    assert native["projection_manifest"]["format"] == "lerobot"


def test_lerobot_native_cache_only_accepts_prepared_local_root(
    lake,
    monkeypatch,
    tmp_path,
):
    opened = _install_fake_lerobot_native_stack(monkeypatch)
    prepared = tmp_path / "prepared-lerobot-cache"
    prepared.mkdir()

    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LEROBOT_NATIVE_FORMAT],
        source_dataset_id="lerobot/droid_100",
        source_dataset_revision="abc123",
        source_dataset_uri=str(prepared),
        lerobot_native_source_mode="source",
        lerobot_native_cache_mode="cache-only",
    )

    native = report["formats"][LEROBOT_NATIVE_FORMAT]
    source = native["native_loader"]["source"]
    assert native["status"] == "completed"
    assert source["kind"] == "source-corpus"
    assert source["root"] == str(prepared)
    assert source["download_videos"] is False
    assert source["resolver"]["cache_mode"] == "cache-only"
    assert source["preflight"]["implicit_hub_download"] is False
    assert opened["kwargs"]["root"] == str(prepared)
    assert opened["kwargs"]["download_videos"] is False


def test_lerobot_v3_alias_selects_native_arm(lake, monkeypatch):
    import lancedb_robotics.benchmark as benchmark

    monkeypatch.setattr(
        benchmark,
        "_lerobot_native_dependency_status",
        lambda: {
            "available": False,
            "modules": ["lerobot", "torch"],
            "missing": ["lerobot"],
            "install": "install native stack",
            "versions": {},
            "decode_backend": {
                "selected": None,
                "available": False,
                "missing": ["torchcodec", "av"],
            },
        },
    )

    report = run_benchmark_suite(lake, "bench-v1", formats=["lerobot", "lerobot-v3"])

    assert report["params"]["formats"] == [LEROBOT_DEFAULT_FORMAT, LEROBOT_NATIVE_FORMAT]
    assert report["formats"][LEROBOT_DEFAULT_FORMAT]["status"] == "completed"
    assert report["formats"][LEROBOT_NATIVE_FORMAT]["status"] == "skipped"


def test_prepare_lerobot_benchmark_dataset_records_public_descriptor(lake, monkeypatch):
    import lancedb_robotics.benchmark as benchmark

    def fake_ingest(lake_arg, source, **kwargs):
        assert lake_arg is lake
        assert source == "lerobot/droid_100@abc123"
        assert kwargs["created_by"] == "lancedb-robotics-bench"
        return SimpleNamespace(
            run_id="run-bench",
            rows_added={
                "runs": 1,
                "episodes": 2,
                "observations": 4,
                "scenarios": 2,
                "videos": 0,
                "video_encodings": 0,
            },
        )

    monkeypatch.setattr(benchmark, "ingest_lerobot", fake_ingest)

    report = prepare_lerobot_benchmark_dataset(
        lake,
        "lerobot/droid_100",
        revision="abc123",
        snapshot_name="droid-100-smoke",
        size_tier="droid-100",
        storage_tier="hf-cache",
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert report["status"] == "completed"
    assert report["dataset_id"] == "lerobot/droid_100"
    assert report["source_uri"] == "hf://lerobot/droid_100"
    assert report["source_ref"] == "lerobot/droid_100@abc123"
    assert report["revision"] == "abc123"
    assert report["size_tier"] == "droid-100"
    assert report["storage_tier"] == "hf-cache"
    assert report["source_size_bytes"] is None
    assert "DROID-100" in report["size_note"]
    availability = report["availability"]
    assert availability["status"] in {"available", "skipped"}
    assert availability["notes"]
    if availability["status"] == "skipped":
        assert availability["reason"]
    assert report["snapshot_name"] == "droid-100-smoke"
    assert report["scenario_count"] == 2
    assert report["rows_added"]["observations"] == 4


def test_public_lerobot_capacity_plan_gates_mid_full_tiers():
    mid = plan_public_lerobot_benchmark_capacity(size_tier="mid", storage_tier="hf-cache")

    assert mid["status"] == "skipped"
    assert mid["selected_tier"] == "mid"
    assert any("capacity-max-source-bytes" in reason for reason in mid["skip_reasons"])
    tiers = {tier["tier"]: tier for tier in mid["tiers"]}
    assert tiers["droid-100"]["status"] == "allowed"
    assert tiers["full"]["status"] == "skipped"

    allowed = plan_public_lerobot_benchmark_capacity(
        size_tier="mid",
        storage_tier="object-store",
        max_source_bytes=600_000_000_000,
        max_artifact_bytes=30_000_000_000,
        time_budget_seconds=20_000,
        require_gpu=True,
        gpu_available=True,
        require_object_store=True,
        publication_destination="s3://robotics-benchmarks/lerobot",
    )

    assert allowed["status"] == "allowed"
    assert allowed["budgets"]["require_gpu"] is True
    assert allowed["budgets"]["gpu_available"] is True
    assert allowed["budgets"]["publication_destination_ready"]["ready"] is True
    assert allowed["requested_samples"]["decoded_frame_probes"] == 64


def test_enterprise_remote_benchmark_skip_is_explicit_without_remote_fixture(lake):
    report = run_benchmark_suite(lake, "bench-v1", formats=[ENTERPRISE_LANCE_FORMAT])

    enterprise = report["formats"][ENTERPRISE_LANCE_FORMAT]
    assert enterprise["status"] == "skipped"
    assert enterprise["format"] == ENTERPRISE_LANCE_FORMAT
    assert "enterprise_fixture_uri" in enterprise["skip_reason"]


def test_benchmark_report_writes_json(lake, tmp_path):
    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LANCE_FORMAT],
        output_dir=tmp_path / "artifacts",
        sample_limit=2,
        random_access_samples=2,
    )
    path = write_benchmark_report(report, tmp_path / "report.json")

    saved = json.loads(path.read_text())
    assert saved["schema_version"] == report["schema_version"]
    assert saved["formats"][LANCE_FORMAT]["status"] == "completed"


def test_public_lerobot_benchmark_retains_artifacts_and_dashboard(
    lake,
    tmp_path,
    monkeypatch,
):
    import lancedb_robotics.benchmark as benchmark

    def fake_prepare(lake_arg, source, **kwargs):
        assert lake_arg is lake
        assert source == "lerobot/droid_100"
        assert kwargs["revision"] == "abc123"
        assert kwargs["snapshot_name"] == "bench-v1"
        assert kwargs["compact"] is False
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
    artifact_root = tmp_path / "public-benchmarks"

    result = run_public_lerobot_benchmark(
        lake,
        artifact_root=artifact_root,
        source="lerobot/droid_100",
        revision="abc123",
        report_id="droid-100-ci",
        snapshot_name="bench-v1",
        formats=[LANCE_FORMAT, DEEPLAKE_FORMAT, ENTERPRISE_LANCE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        random_frame_samples=2,
        frames_per_clip=3,
        query_limit=1,
        size_tier="droid-100",
        storage_tier="hf-cache",
        compact=False,
        prune_versions=False,
        index_predicates=False,
    )

    assert result["status"] == "completed"
    assert result["report_id"] == "droid-100-ci"
    assert result["run_dir"] == str(artifact_root / "runs" / "droid-100-ci")
    paths = result["paths"]
    for key in ("prepare_report", "benchmark_report", "artifact_manifest", "log"):
        assert Path(paths[key]).exists()
    assert Path(paths["dashboard"]).exists()
    assert Path(paths["index"]).exists()
    assert Path(paths["prepare_report"]).relative_to(artifact_root) == Path(
        "runs/droid-100-ci/reports/prepare.json"
    )
    assert Path(paths["benchmark_report"]).relative_to(artifact_root) == Path(
        "runs/droid-100-ci/reports/benchmark.json"
    )

    manifest = json.loads(Path(paths["artifact_manifest"]).read_text())
    assert manifest["schema_version"] == "lancedb-robotics-public-lerobot-benchmark-v0"
    assert manifest["report_id"] == "droid-100-ci"
    assert manifest["commit"] == "abc1234"
    assert manifest["dataset"]["dataset_id"] == "lerobot/droid_100"
    assert manifest["dataset"]["revision"] == "abc123"
    assert manifest["dataset"]["source_uri"] == "hf://lerobot/droid_100"
    assert manifest["dataset"]["size_tier"] == "droid-100"
    assert manifest["dataset"]["storage_tier"] == "hf-cache"
    assert manifest["hardware"]["cpu_count"]
    assert manifest["storage_tiers"]["active"] == "hf-cache"
    skipped_tiers = {
        tier["tier"]: tier
        for tier in manifest["storage_tiers"]["tiers"]
        if tier["status"] == "skipped"
    }
    assert skipped_tiers["local"]["skip_reason"]
    assert skipped_tiers["object-store"]["skip_reason"]
    assert manifest["format_statuses"][LANCE_FORMAT] == "completed"
    assert manifest["format_statuses"][DEEPLAKE_FORMAT] == "skipped"
    assert manifest["format_statuses"][ENTERPRISE_LANCE_FORMAT] == "skipped"
    assert DEEPLAKE_FORMAT in manifest["skipped"]
    assert ENTERPRISE_LANCE_FORMAT in manifest["skipped"]
    assert "reports/prepare.json" in manifest["artifact_files"]
    assert "reports/benchmark.json" in manifest["artifact_files"]
    assert "logs/run.log" in manifest["artifact_files"]
    assert manifest["history"]["index_path"] == paths["index"]
    assert manifest["history"]["dashboard_path"] == paths["dashboard"]
    assert manifest["history"]["latest_report_id"] == "droid-100-ci"
    assert manifest["capacity"]["status"] == "allowed"
    assert manifest["capacity"]["selected_tier"] == "droid-100"
    assert "reports/capacity.json" in manifest["artifact_files"]

    index = json.loads(Path(paths["index"]).read_text())
    assert index["schema_version"] == "lancedb-robotics-public-lerobot-benchmark-v0"
    assert index["run_count"] == 1
    assert index["latest_report_id"] == "droid-100-ci"
    row = index["runs"][0]
    assert row["report_id"] == "droid-100-ci"
    assert row["commit"] == "abc1234"
    assert row["revision"] == "abc123"
    assert row["storage_tier"] == "hf-cache"
    assert row["artifact_manifest"] == "runs/droid-100-ci/artifact-manifest.json"
    assert row["prepare_report"] == "runs/droid-100-ci/reports/prepare.json"
    assert row["benchmark_report"] == "runs/droid-100-ci/reports/benchmark.json"
    assert row["capacity_report"] == "runs/droid-100-ci/reports/capacity.json"
    assert row["capacity_status"] == "allowed"
    assert row["format_statuses"][LANCE_FORMAT] == "completed"
    assert row["format_statuses"][DEEPLAKE_FORMAT] == "skipped"
    assert row["format_statuses"][ENTERPRISE_LANCE_FORMAT] == "skipped"
    assert row["skipped"][DEEPLAKE_FORMAT]
    assert row["skipped"][ENTERPRISE_LANCE_FORMAT]
    assert row["metrics"]["lance_throughput"] is not None

    dashboard = Path(paths["dashboard"]).read_text()
    assert "Public LeRobot Benchmark History" in dashboard
    assert "droid-100-ci" in dashboard
    assert "abc1234" in dashboard
    assert "abc123" in dashboard
    assert "hf-cache" in dashboard
    assert "lance:completed" in dashboard
    assert "deeplake:skipped" in dashboard
    assert "enterprise-lance:skipped" in dashboard

    published_root = tmp_path / "published-public-benchmarks"
    publication = publish_public_lerobot_benchmark(
        artifact_root,
        destination=published_root,
        report_id="droid-100-ci",
        retain_latest=1,
    )

    assert publication["status"] == "published"
    assert publication["backend"] == "filesystem"
    assert publication["report_ids"] == ["droid-100-ci"]
    assert "runs/droid-100-ci/reports/prepare.json" in publication["files"]["planned_or_written"]
    assert "index.json" in publication["files"]["planned_or_written"]
    published_manifest_path = published_root / "runs/droid-100-ci/artifact-manifest.json"
    assert published_manifest_path.exists()
    published_manifest = json.loads(published_manifest_path.read_text())
    assert published_manifest["artifact_checksums"]["reports/prepare.json"]["sha256"]
    assert published_manifest["artifact_checksums"]["reports/benchmark.json"]["size_bytes"] > 0
    assert published_manifest["retention"]["class"] == "public-benchmark-history"
    assert published_manifest["retention"]["protected"] is True
    assert published_manifest["retention"]["reason"] == "within-latest-1-window"
    target = published_manifest["publication"]["targets"][0]
    assert target["destination"] == str(published_root)
    assert target["status"] == "published"
    published_index = json.loads((published_root / "index.json").read_text())
    assert published_index["runs"][0]["retention_protected"] is True
    assert published_index["runs"][0]["publication"]["latest_destination"] == str(published_root)

    repeated = publish_public_lerobot_benchmark(
        artifact_root,
        destination=published_root,
        report_id="droid-100-ci",
        retain_latest=1,
    )
    assert "runs/droid-100-ci/reports/prepare.json" in repeated["files"]["unchanged"]

    (published_root / "runs/droid-100-ci/reports/benchmark.json").write_text("tampered\n")
    with pytest.raises(BenchmarkError, match="immutable"):
        publish_public_lerobot_benchmark(
            artifact_root,
            destination=published_root,
            report_id="droid-100-ci",
            retain_latest=1,
        )


def test_public_lerobot_capacity_skip_retains_manifest_without_prepare(
    lake,
    tmp_path,
    monkeypatch,
):
    import lancedb_robotics.benchmark as benchmark

    def fail_prepare(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("capacity skip should not prepare the source")

    monkeypatch.setattr(benchmark, "prepare_lerobot_benchmark_dataset", fail_prepare)
    monkeypatch.setenv("GITHUB_SHA", "capacity123")
    artifact_root = tmp_path / "public-benchmarks"

    result = run_public_lerobot_benchmark(
        lake,
        artifact_root=artifact_root,
        source="lerobot/droid_100",
        revision="abc123",
        report_id="mid-capacity-skip",
        snapshot_name="bench-v1",
        formats=[LANCE_FORMAT, LEROBOT_DEFAULT_FORMAT],
        size_tier="mid",
        storage_tier="hf-cache",
    )

    assert result["status"] == "skipped"
    assert result["capacity"]["status"] == "skipped"
    assert "prepare_report" not in result["paths"]
    assert Path(result["paths"]["capacity_report"]).exists()
    assert not (artifact_root / "runs/mid-capacity-skip/reports/prepare.json").exists()

    manifest = json.loads(Path(result["paths"]["artifact_manifest"]).read_text())
    assert manifest["status"] == "skipped"
    assert manifest["skip_reason"]
    assert manifest["dataset"]["size_tier"] == "mid"
    assert manifest["capacity"]["status"] == "skipped"
    assert manifest["format_statuses"][LANCE_FORMAT] == "skipped"
    assert manifest["skipped"][LANCE_FORMAT] == manifest["skip_reason"]
    assert "reports/capacity.json" in manifest["artifact_files"]

    index = json.loads(Path(result["paths"]["index"]).read_text())
    row = index["runs"][0]
    assert row["status"] == "skipped"
    assert row["capacity_status"] == "skipped"
    assert row["capacity_skip_reasons"]
    assert row["capacity_report"] == "runs/mid-capacity-skip/reports/capacity.json"

    dashboard = Path(result["paths"]["dashboard"]).read_text()
    assert "mid-capacity-skip" in dashboard
    assert "skipped" in dashboard
    assert "capacity-max-source-bytes" in dashboard


def test_public_lerobot_claim_validator_accepts_valid_claim(
    lake,
    tmp_path,
    monkeypatch,
):
    import lancedb_robotics.benchmark as benchmark

    def fake_prepare(lake_arg, source, **kwargs):
        assert lake_arg is lake
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
    artifact_root = tmp_path / "public-benchmarks"
    run_public_lerobot_benchmark(
        lake,
        artifact_root=artifact_root,
        source="lerobot/droid_100",
        revision="abc123",
        report_id="droid-100-claim",
        snapshot_name="bench-v1",
        formats=[LANCE_FORMAT, DEEPLAKE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        random_frame_samples=2,
        frames_per_clip=3,
        query_limit=1,
        size_tier="droid-100",
        storage_tier="hf-cache",
    )
    claims = {
        "schema_version": PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION,
        "claims": [
            {
                "id": "readme-lance-throughput",
                "report_id": "droid-100-claim",
                "commit": "claim1234",
                "dataset_revision": "abc123",
                "storage_tier": "hf-cache",
                "format": LANCE_FORMAT,
                "metric": "dataloader_throughput",
            }
        ],
    }

    report = validate_public_lerobot_benchmark_claims(artifact_root, claims=claims)

    assert report["status"] == "passed"
    assert report["manifest_count"] == 1
    assert report["claim_count"] == 1
    assert report["claims"][0]["status"] == "passed"
    assert report["diagnostics"] == []

    manifest_path = artifact_root / "runs/droid-100-claim/artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "stale-public-lerobot-schema"
    manifest_path.write_text(json.dumps(manifest))
    stale = validate_public_lerobot_benchmark_claims(artifact_root, claims=claims)
    codes = {diagnostic["code"] for diagnostic in stale["diagnostics"]}
    assert stale["status"] == "failed"
    assert "manifest-schema-version" in codes


def test_public_lerobot_claim_validator_rejects_skipped_format_claim(
    lake,
    tmp_path,
    monkeypatch,
):
    import lancedb_robotics.benchmark as benchmark

    def fake_prepare(lake_arg, source, **kwargs):
        assert lake_arg is lake
        return {
            "status": "completed",
            "prepared_at": "2026-01-01T00:00:00Z",
            "lake_uri": lake.uri,
            "dataset_id": str(source),
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
    artifact_root = tmp_path / "public-benchmarks"
    run_public_lerobot_benchmark(
        lake,
        artifact_root=artifact_root,
        source="lerobot/droid_100",
        revision="abc123",
        report_id="droid-100-claim",
        snapshot_name="bench-v1",
        formats=[LANCE_FORMAT, DEEPLAKE_FORMAT],
        sample_limit=2,
        random_access_samples=2,
        random_frame_samples=2,
        frames_per_clip=3,
        query_limit=1,
        size_tier="droid-100",
        storage_tier="hf-cache",
    )
    claims = {
        "schema_version": PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION,
        "claims": [
            {
                "id": "bad-deeplake-throughput",
                "report_id": "droid-100-claim",
                "commit": "claim1234",
                "dataset_revision": "abc123",
                "storage_tier": "hf-cache",
                "format": DEEPLAKE_FORMAT,
                "metric": "dataloader_throughput",
            }
        ],
    }

    report = validate_public_lerobot_benchmark_claims(artifact_root, claims=claims)

    assert report["status"] == "failed"
    assert report["claims"][0]["status"] == "failed"
    codes = {diagnostic["code"] for diagnostic in report["diagnostics"]}
    assert "claim-format-status-mismatch" in codes
    assert "claim-format-not-measured" in codes


def test_public_lerobot_benchmark_requires_pinned_hf_revision(lake, tmp_path):
    with pytest.raises(BenchmarkError, match="pinned source revision"):
        run_public_lerobot_benchmark(
            lake,
            artifact_root=tmp_path / "public-benchmarks",
            source="lerobot/droid_100",
            report_id="missing-revision",
            snapshot_name="bench-v1",
            formats=[LANCE_FORMAT],
            sample_limit=2,
            random_access_samples=2,
        )
