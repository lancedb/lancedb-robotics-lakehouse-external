"""Live plan-executor client + cache metrics tests (backlog 0119).

Exercises the concrete plan-executor client behind the 0071 hydration executor
boundary: request envelopes with pinned version + manifest e-tag, real Sophon
``x-cache-*`` cache metrics aggregated by plan executor and request, request-id
recording, typed remote-hydration diagnostics, the metadata-only guardrail, and
local-vs-live equivalence.
"""

from datetime import UTC, datetime

import pyarrow as pa
import pytest

from lancedb_robotics.blob import PAYLOAD_BLOB_COLUMN
from lancedb_robotics.connections import LakeCapabilities, LakeConnectionSpec
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.enterprise_conformance import FakeQueryNodeClient
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA, SCENARIOS_SCHEMA
from lancedb_robotics.training import (
    MetadataOnlyViolationError,
    QueryNodeRequest,
    RemoteQueryNodeClient,
    RemoteQueryNodeError,
    iter_training_batches,
)


def _training_lake(path, *, frame_count=3):
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    end_time_ns = (frame_count - 1) * 1_000_000_000
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-native-training",
                    "run_kind": "demo",
                    "source": "synthetic",
                    "source_id": "src-native-training",
                    "raw_uri": "memory://run-native-training",
                    "robot_id": "robot-1",
                    "site_id": "lab",
                    "task_id": "pick the cube",
                    "start_time_ns": 0,
                    "end_time_ns": end_time_ns,
                    "duration_ns": end_time_ns,
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
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                {
                    "observation_id": f"obs-camera-{index}",
                    "run_id": "run-native-training",
                    "episode_id": "ep-native-training",
                    "episode_index": 0,
                    "frame_index": index,
                    "timestamp_ns": index * 1_000_000_000,
                    "sensor_id": "camera-front",
                    "topic": "/camera/front",
                    "modality": "image",
                    "raw_uri": f"memory://frame-{index}",
                    "raw_channel": "front",
                    "raw_sequence": index,
                    "state_vector": [float(index), float(index) + 0.5],
                    "action_vector": [float(index) * 11.0, float(index) * -11.0],
                    "reward": 0.0,
                    "done": index == frame_count - 1,
                    "payload_blob": f"frame-{index}".encode(),
                    "payload_encoding": "raw",
                    "payload_size": len(f"frame-{index}".encode()),
                    "metadata": [],
                    "quality_flags": [],
                    "transform_id": "tfm-obs",
                    "created_at": now,
                }
                for index in range(frame_count)
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-native-training",
                    "run_id": "run-native-training",
                    "episode_id": "ep-native-training",
                    "episode_index": 0,
                    "name": "native-training",
                    "start_time_ns": 0,
                    "end_time_ns": end_time_ns,
                    "window_ns": end_time_ns,
                    "is_partial": False,
                    "topics": ["/camera/front"],
                    "observation_ids": [f"obs-camera-{index}" for index in range(frame_count)],
                    "observation_count": frame_count,
                    "scenario_type": "demo",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": ["camera"],
                    "summary": "pick the cube",
                    "transform_id": "tfm-scenario",
                    "created_at": now,
                }
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    create_snapshot(
        lake,
        name="demo-v1",
        scenario_ids=["scn-native-training"],
        split_by="scenario",
    )
    return lake


def _mark_enterprise_lake(lake, *, uri="db://robotics"):
    lake.connection_spec = LakeConnectionSpec(
        kind="lancedb_remote_db",
        uri=uri,
        display_uri=uri,
        lancedb_connect_kwargs={
            "api_key": "secret-api-key",
            "region": "us-west-2",
            "host_override": "https://phalanx.acme.internal",
        },
        auth_refs={"remote": "enterprise-prod"},
        capabilities=LakeCapabilities(
            server_side_query=True,
            direct_object_io=False,
            blob_fetch_remote=True,
        ),
        direct_object_io_allowed=False,
    )
    lake.capabilities = lake.connection_spec.capabilities
    lake.uri = uri
    return lake


@pytest.fixture
def lake(tmp_path):
    return _training_lake(tmp_path / "robot.lance")


def _hydrate_all(dataset):
    """Drive a single coalesced batch through the hydration executor."""
    return dataset.__getitems__(list(range(len(dataset))))


def test_live_take_reports_x_cache_headers_aggregated_by_pe_and_request(lake):
    _mark_enterprise_lake(lake)
    lake.query_node_client = FakeQueryNodeClient(
        lake, plan_executors=("pe-a", "pe-b"), hit_pct=75
    )
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
    )
    samples = _hydrate_all(dataset)

    assert [s["payload"] for s in samples] == [b"frame-0", b"frame-1", b"frame-2"]
    metrics = dataset.manifest.backend["metrics"]
    # The fake returns x-cache-hits / x-cache-misses headers; the loader parses
    # and aggregates them.
    assert metrics["cache_hits"] + metrics["cache_misses"] == 3
    assert metrics["cache_hits"] > 0
    assert metrics["live_hydration_requests"] > 0
    assert metrics["pe_fanout"] == 2
    # aggregated by plan executor
    per_pe = metrics["cache_by_plan_executor"]
    assert set(per_pe) == {"pe-a", "pe-b"}
    assert sum(v["hits"] for v in per_pe.values()) == metrics["cache_hits"]
    # aggregated by request
    per_request = metrics["cache_by_request"]
    assert per_request
    assert sum(v["hits"] for v in per_request.values()) == metrics["cache_hits"]
    assert all(str(rid).startswith("req-") for rid in metrics["request_ids"])


def test_present_manifest_etag_is_carried_into_every_request_envelope(lake):
    _mark_enterprise_lake(lake)
    lake.manifest_etags = {"observations": "etag-obs-42"}
    lake.query_node_client = FakeQueryNodeClient(lake)
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
    )
    _hydrate_all(dataset)

    metrics = dataset.manifest.backend["metrics"]
    take_ops = [op for op in metrics["operations"] if op["operation"] == "remote_take"]
    assert take_ops
    assert all(op["manifest_etag"] == "etag-obs-42" for op in take_ops)
    assert metrics["manifest_etags"]["observations"] == "etag-obs-42"


def test_missing_manifest_etag_is_allowed(lake):
    _mark_enterprise_lake(lake)
    # No lake.manifest_etags attached: envelopes simply omit the e-tag.
    lake.query_node_client = FakeQueryNodeClient(lake)
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
    )
    samples = _hydrate_all(dataset)

    assert [s["payload"] for s in samples] == [b"frame-0", b"frame-1", b"frame-2"]
    metrics = dataset.manifest.backend["metrics"]
    take_ops = [op for op in metrics["operations"] if op["operation"] == "remote_take"]
    assert take_ops
    assert all(op["manifest_etag"] is None for op in take_ops)
    assert metrics["manifest_etags"] == {}


def test_remote_take_failure_raises_typed_diagnostic(lake):
    _mark_enterprise_lake(lake)
    lake.query_node_client = FakeQueryNodeClient(
        lake, fail_operations={"remote_take"}
    )
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
    )
    with pytest.raises(RemoteQueryNodeError) as excinfo:
        _hydrate_all(dataset)
    error = excinfo.value
    assert error.operation == "remote_take"
    assert error.table == "observations"
    assert error.request_id is not None
    assert error.remediation
    assert "fallback" in str(error).lower()


def test_remote_filtered_read_error_includes_context_and_guidance(lake):
    _mark_enterprise_lake(lake)
    lake.query_node_client = FakeQueryNodeClient(
        lake, fail_operations={"remote_filtered_read"}
    )
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
    )
    with pytest.raises(RemoteQueryNodeError) as excinfo:
        dataset._hydration_executor.filtered_read(
            "observations",
            columns=["observation_id"],
            where_sql="observation_id = 'obs-camera-0'",
        )
    error = excinfo.value
    assert error.operation == "remote_filtered_read"
    assert error.table == "observations"
    assert error.request_id is not None
    message = str(error)
    assert "observations" in message
    assert "Remediation" in message


def test_metadata_only_live_client_refuses_payload_columns(lake):
    _mark_enterprise_lake(lake)
    lake.query_node_client = FakeQueryNodeClient(lake)
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        media="metadata",
        backend="enterprise",
    )
    executor = dataset._hydration_executor
    with pytest.raises(MetadataOnlyViolationError):
        executor.take_blobs("observations", PAYLOAD_BLOB_COLUMN, [0])
    with pytest.raises(MetadataOnlyViolationError):
        executor.take_rows(
            "observations", [0], columns=["observation_id", PAYLOAD_BLOB_COLUMN]
        )
    with pytest.raises(MetadataOnlyViolationError):
        executor.filtered_read(
            "video_encodings",
            columns=["video_encodings.data"],
            where_sql="",
        )


def test_local_and_live_plan_executor_produce_equivalent_payloads_and_lineage(lake):
    local = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="local",
    )
    local_payloads = [
        b for batch in iter_training_batches(local, batch_size=8) for b in batch["payload"]
    ]
    local_versions = list(local.manifest.table_versions)

    _mark_enterprise_lake(lake)
    lake.manifest_etags = {"observations": "etag-obs-7"}
    lake.query_node_client = FakeQueryNodeClient(lake)
    live = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
    )
    live_payloads = [
        b for batch in iter_training_batches(live, batch_size=8) for b in batch["payload"]
    ]

    assert live_payloads == local_payloads
    assert list(live.manifest.table_versions) == local_versions


def test_backend_report_marks_live_client_attached(lake):
    _mark_enterprise_lake(lake)
    without = lake.training.dataset(
        "demo-v1", columns=["observation_id"], backend="enterprise"
    )
    assert without.manifest.backend["plan_executor"]["available"] is True
    assert without.manifest.backend["plan_executor"]["live_client_attached"] is False
    assert without.manifest.backend["plan_executor"]["integration_status"] == (
        "capability-reported"
    )

    lake.query_node_client = FakeQueryNodeClient(lake)
    attached = lake.training.dataset(
        "demo-v1", columns=["observation_id"], backend="enterprise"
    )
    assert attached.manifest.backend["plan_executor"]["live_client_attached"] is True
    assert attached.manifest.backend["plan_executor"]["integration_status"] == (
        "live-client-attached"
    )


def test_remote_plan_executor_client_reference_impl_echoes_metadata(lake):
    _mark_enterprise_lake(lake)
    client = RemoteQueryNodeClient(
        lake, metrics_source={"remote_take": {"x-cache-hits": 2, "x-cache-misses": 1}}
    )
    request = QueryNodeRequest(
        operation="remote_take",
        table="observations",
        version=None,
        columns=("observation_id",),
        coalescing_window="native-training-batch",
        row_ids=(0, 1, 2),
        manifest_etag="etag-obs-9",
    )
    response = client.execute(request)
    assert [row["observation_id"] for row in response.rows] == [
        "obs-camera-0",
        "obs-camera-1",
        "obs-camera-2",
    ]
    assert response.manifest_etag == "etag-obs-9"
    assert response.request_id.startswith("pe-remote_take-")
    assert response.cache_metrics == {"x-cache-hits": 2, "x-cache-misses": 1}


def test_plan_executor_conformance_run_passes(lake):
    _mark_enterprise_lake(lake)
    report = lake.training.query_node_conformance("demo-v1", strict=True)
    assert report.ok()
    assert set(report.checks) == {
        "local_live_equivalence",
        "cache_metrics_by_pe_and_request",
        "request_envelope_carries_etag",
        "request_ids_recorded",
        "remote_failure_is_typed",
        "metadata_only_guardrail",
    }
    assert report.metrics["cache_hits"] > 0


def test_backlog_0345_deprecated_plan_executor_names_still_work(lake):
    """Back-compat: the pre-0345 plan-executor SDK names alias the query-node ones.

    Old class imports resolve to the new classes, the old ``lake.plan_executor_client``
    hook is still honored by the loader (dual-read), and the old
    ``lake.training.plan_executor_conformance`` method still runs.
    """
    import lancedb_robotics.enterprise_conformance as ec
    import lancedb_robotics.training as t

    # Class + function aliases resolve to the query-node objects.
    assert t.PlanExecutorRequest is t.QueryNodeRequest
    assert t.PlanExecutorResponse is t.QueryNodeResponse
    assert t.PlanExecutorClient is t.QueryNodeClient
    assert t.RemotePlanExecutorClient is t.RemoteQueryNodeClient
    assert t.RemotePlanExecutorError is t.RemoteQueryNodeError
    assert t.PlanExecutorUnavailableError is t.QueryNodeUnavailableError
    assert ec.FakePlanExecutorClient is ec.FakeQueryNodeClient
    assert ec.PlanExecutorConformanceReport is ec.QueryNodeConformanceReport
    assert ec.run_plan_executor_conformance is ec.run_query_node_conformance

    # The deprecated lake hook name is still honored by the loader.
    _mark_enterprise_lake(lake)
    lake.plan_executor_client = ec.FakePlanExecutorClient(
        lake, plan_executors=("pe-a",), hit_pct=100
    )
    dataset = lake.training.dataset(
        "demo-v1", columns=["observation_id", "payload"], media="bytes", backend="enterprise"
    )
    samples = _hydrate_all(dataset)
    assert [s["payload"] for s in samples] == [b"frame-0", b"frame-1", b"frame-2"]

    # The deprecated conformance method alias still runs.
    report = lake.training.plan_executor_conformance("demo-v1", strict=True)
    assert report.ok()
