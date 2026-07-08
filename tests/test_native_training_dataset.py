"""Lance-native training dataset tests."""

import json
import tracemalloc
from datetime import UTC, datetime

import pyarrow as pa
import pytest
from conftest import require_start_method, require_torch_loader

import lancedb_robotics.training as training_mod
from lancedb_robotics.blob import PAYLOAD_BLOB_COLUMN
from lancedb_robotics.connections import LakeCapabilities, LakeConnectionSpec
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA, SCENARIOS_SCHEMA
from lancedb_robotics.training import (
    MissingEnterpriseAuthError,
    PrewarmUnavailableError,
    StaleTableVersionError,
    TrainingError,
    TrainingMediaHandle,
    UnsupportedRemoteOperationError,
    collate_torch_training_samples,
    iter_training_batches,
    to_torch_dataloader,
    to_torch_map_dataset,
    training_dataset,
)
from lancedb_robotics.training_prewarm_jobs import InMemoryPrewarmJobRunStore


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
    observations = []
    for index in range(frame_count):
        timestamp_ns = index * 1_000_000_000
        observations.append(
            {
                "observation_id": f"obs-camera-{index}",
                "run_id": "run-native-training",
                "timestamp_ns": timestamp_ns,
                "sensor_id": "camera_front",
                "topic": "/camera/front",
                "modality": "image",
                "task_id": "pick_place",
                "raw_uri": "memory://run-native-training",
                "raw_channel": "/camera/front",
                "raw_log_time_ns": timestamp_ns,
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
                    "scenario_id": "scn-native-training",
                    "run_id": "run-native-training",
                    "start_time_ns": 0,
                    "end_time_ns": end_time_ns,
                    "window_ns": end_time_ns,
                    "is_partial": False,
                    "topics": ["/camera/front"],
                    "observation_ids": [
                        f"obs-camera-{index}" for index in range(frame_count)
                    ],
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


def _large_training_lake(path, *, total_frames, referenced_frames):
    """A lake with more observations than the snapshot references, each carrying a
    non-trivial ``payload_json`` string and a ``payload_blob``.

    Used by the BUG-06 regression test: the old whole-corpus, all-columns context
    read would materialize every row (incl. ``payload_json``) into Python; the fix
    must read only the snapshot's referenced rows, projected to the columns the
    request needs.
    """
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    end_time_ns = (total_frames - 1) * 1_000_000
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-big",
                    "run_kind": "demo",
                    "source": "synthetic",
                    "source_id": "src-big",
                    "raw_uri": "memory://run-big",
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
    # ~0.5 KB of payload_json per row: cheap on disk, but multi-GB in Python if the
    # whole corpus were materialized with all columns (the BUG-06 regression).
    heavy_json = json.dumps({"blob": "x" * 480})
    observations = [
        {
            "observation_id": f"obs-{index}",
            "run_id": "run-big",
            "timestamp_ns": index * 1_000_000,
            "sensor_id": "camera_front",
            "topic": "/camera/front",
            "modality": "image",
            "task_id": "pick_place",
            "raw_uri": "memory://run-big",
            "raw_channel": "/camera/front",
            "raw_log_time_ns": index * 1_000_000,
            "raw_sequence": index,
            "payload_json": heavy_json,
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
        for index in range(total_frames)
    ]
    lake.table("observations").add(pa.Table.from_pylist(observations, schema=OBSERVATIONS_SCHEMA))
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-big",
                    "run_id": "run-big",
                    "start_time_ns": 0,
                    "end_time_ns": (referenced_frames - 1) * 1_000_000,
                    "window_ns": end_time_ns,
                    "is_partial": False,
                    "topics": ["/camera/front"],
                    "observation_ids": [f"obs-{index}" for index in range(referenced_frames)],
                    "observation_count": referenced_frames,
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
    create_snapshot(lake, name="big-v1", scenario_ids=["scn-big"], split_by="scenario")
    return lake


def test_training_dataset_construction_is_scoped_projected_and_bounded(tmp_path):
    # BUG-06: building + iterating a dataset over a large scenario snapshot must
    # read only the referenced observations, projected to the requested columns,
    # and stay bounded in memory -- not materialize the whole corpus.
    total_frames, referenced_frames = 120_000, 100_000
    lake = _large_training_lake(
        tmp_path / "big.lance",
        total_frames=total_frames,
        referenced_frames=referenced_frames,
    )

    tracemalloc.start()
    try:
        dataset = lake.training.dataset("big-v1", columns=["scenario_id"])
        iterator = iter_training_batches(dataset, batch_size=4)
        batches = [next(iterator) for _ in range(2)]
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Scope: the context holds only the snapshot's referenced observations, not the
    # 120k-row lake total.
    assert len(dataset._context.observations) == referenced_frames
    assert dataset.total_frames == referenced_frames

    # Projection: a scenario_id-only request never materializes the heavy
    # payload_json column...
    sample_obs = next(iter(dataset._context.observations.values()))
    assert "payload_json" not in sample_obs
    assert "state_vector" not in sample_obs
    # ...while payload_blob is present only as its cheap {position, size} descriptor
    # (the scan reads no blob bytes), keeping payload accounting accurate.
    assert isinstance(sample_obs.get("payload_blob"), dict)
    assert dataset.manifest.accounting["payload_bytes_referenced"] > 0

    # Iteration yields real batches.
    assert len(batches) == 2
    assert all(batch["scenario_id"] for batch in batches)

    # Bounded memory: the projected metadata for 100k rows is well under the
    # multi-GB Python blowup the whole-corpus read produced. tracemalloc tracks the
    # Python-object amplification that was the regression (the Arrow buffers are
    # C++ and bounded by the 4096-row batch).
    assert peak < 1_000_000_000, f"peak Python allocation {peak} bytes exceeds the 1 GB bound"


def test_scoped_observation_read_takes_by_row_id_not_in_predicate(tmp_path, monkeypatch):
    # BUG-06 (round 3): an id-pinned snapshot must resolve its observations by
    # random-access take_row_ids, never an `observation_id IN (...)` predicate -- a
    # ~178K-id predicate blows up the query planner natively. This is a structural
    # guard: a revert to the IN predicate fails it regardless of fixture size (the
    # native blowup is invisible to the tracemalloc bound in the test above).
    from lancedb_robotics import dataset_export

    lake = _large_training_lake(
        tmp_path / "take.lance", total_frames=2_000, referenced_frames=1_500
    )

    in_predicate_columns: list[str] = []
    real_sql_in = dataset_export._sql_in_predicate

    def spy_sql_in(column, values):
        in_predicate_columns.append(column)
        return real_sql_in(column, values)

    take_calls: list[dict] = []
    real_take = dataset_export._rows_by_id_take

    def spy_take(table, *, columns, id_column, id_values, **kwargs):
        ids = list(id_values)
        take_calls.append({"id_column": id_column, "count": len(ids)})
        return real_take(
            table, columns=columns, id_column=id_column, id_values=ids, **kwargs
        )

    monkeypatch.setattr(dataset_export, "_sql_in_predicate", spy_sql_in)
    monkeypatch.setattr(dataset_export, "_rows_by_id_take", spy_take)

    dataset = lake.training.dataset("big-v1", columns=["scenario_id", "caption"])

    # Correctness: exactly the snapshot's referenced observations, no more.
    assert len(dataset._context.observations) == 1_500
    assert dataset.total_frames == 1_500
    # The pinned ids were resolved by take_row_ids over observation_id...
    assert any(
        call["id_column"] == "observation_id" and call["count"] == 1_500
        for call in take_calls
    ), f"scoped read did not take observations by row id: {take_calls}"
    # ...and never by an observation_id IN (...) predicate (the BUG-06 blowup).
    assert "observation_id" not in in_predicate_columns


@pytest.fixture
def lake(tmp_path):
    return _training_lake(tmp_path / "robot.lance")


def _mark_enterprise_lake(
    lake,
    *,
    uri="db://robotics",
    server_side_query=True,
    blob_fetch_remote=True,
    direct_object_io=False,
):
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
            server_side_query=server_side_query,
            direct_object_io=direct_object_io,
            blob_fetch_remote=blob_fetch_remote,
        ),
        direct_object_io_allowed=direct_object_io,
    )
    lake.capabilities = lake.connection_spec.capabilities
    lake.uri = uri
    return lake


def test_lake_training_dataset_reads_native_samples_and_manifest(lake):
    dataset = lake.training.dataset("demo-v1")

    assert not hasattr(lake, "facades")
    assert len(dataset) == 3
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3
    assert dataset.total_frames == 3
    assert dataset.episode_data_index == {"from": [0], "to": [3]}

    sample = dataset[1]
    assert sample["observation_id"] == "obs-camera-1"
    assert sample["scenario_id"] == "scn-native-training"
    assert sample["run_id"] == "run-native-training"
    assert sample["relative_time_s"] == 1.0
    assert sample["state_vector"] == [1.0, 1.5]
    assert sample["action_vector"] == [11.0, -11.0]
    assert sample["payload_size"] is None
    assert sample["_media"]["policy"] == "metadata"
    assert sample["_media"]["fields"]["payload_size"]["materialized"] is False
    assert dataset.manifest.access_pattern == "lance-native-snapshot"
    assert dataset.manifest.table_versions
    assert dataset.manifest.row_plan_id == dataset.row_plan.plan_id
    assert dataset.manifest.epoch_plan_id == dataset.epoch_plan.plan_id
    assert dataset.manifest.media_policy == "metadata"
    assert dataset.manifest.to_dict()["media"]["cache"]["policy"] == "none"
    assert dataset.manifest.accounting["target_format"] == "lance-native-training"
    assert dataset.manifest.accounting["payload_bytes_referenced"] == sum(
        len(f"frame-{index}".encode()) for index in range(3)
    )
    assert dataset.manifest.accounting["payload_bytes_copied"] == 0
    assert dataset.manifest.accounting["payload_copy_policy"] == "logical-reference"
    backend = dataset.manifest.to_dict()["backend"]
    assert backend["requested_backend"] == "auto"
    assert backend["resolved_backend"] == "local"
    assert backend["execution_mode"] == "local-lance-native"
    assert backend["metrics"]["rows_planned"] == 3
    assert dataset.manifest.to_dict()["epoch_backend"]["kind"] == "python"
    assert dataset.manifest.to_dict()["epoch_backend"]["execution_mode"] == (
        "python-snapshot-order"
    )


def test_training_epoch_backend_capability_reports_lancedb_permutation(lake):
    capability = lake.training.epoch_backend_capability()

    assert capability["python"]["supported"] is True
    assert "direct_lance" in capability
    assert isinstance(capability["direct_lance"]["supported"], bool)
    assert capability["lancedb_permutation"]["supported"] is True
    assert capability["lancedb_permutation"]["execution_mode"] == (
        "lancedb-permutation-table"
    )


def test_training_dataset_records_lancedb_permutation_epoch_backend(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=17,
        epoch=2,
    )

    backend = dataset.epoch_plan.backend.to_dict()
    assert backend["kind"] == "lancedb_permutation"
    assert backend["execution_mode"] == "lancedb-permutation-table"
    assert backend["row_plan_id"] == dataset.row_plan.plan_id
    assert backend["epoch_plan_id"] == dataset.epoch_plan.plan_id
    assert backend["snapshot_id"] == dataset.manifest.dataset_id
    assert backend["permutation_table"].startswith("__lancedb_robotics_epoch_perm_")
    assert backend["permutation_ref"] == f"lancedb://{backend['permutation_table']}"

    manifest = dataset.manifest.to_dict()
    assert manifest["epoch_backend"] == backend
    assert manifest["loader_report"]["plans"]["epoch_backend"] == backend
    assert backend["permutation_table"] in set(lake._db.list_tables().tables)

    rows = lake._db.open_table(backend["permutation_table"]).to_arrow().to_pylist()
    id_by_row_id = {
        int(row_id): frame_id
        for row_id, frame_id in zip(dataset.row_plan.row_ids, dataset.row_plan.frame_ids, strict=True)
    }
    assert [id_by_row_id[int(row["row_id"])] for row in rows] == [
        dataset.row_plan.frame_ids[index] for index in dataset.epoch_plan.global_order
    ]


def test_lancedb_permutation_backend_matches_python_epoch_order(
    lake,
    monkeypatch,
):
    native = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=29,
        epoch=1,
    )

    monkeypatch.setattr(training_mod, "_lancedb_permutation_module", lambda: None)
    python = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=29,
        epoch=1,
    )

    assert native.epoch_plan.backend.kind == "lancedb_permutation"
    assert python.epoch_plan.backend.kind == "python"
    assert "lancedb.permutation is not importable" in python.epoch_plan.backend.reason
    assert [row["observation_id"] for row in native] == [
        row["observation_id"] for row in python
    ]
    assert python.manifest.to_dict()["epoch_backend"]["kind"] == "python"


def test_training_dataset_rejects_enterprise_backend_for_local_lake(lake):
    with pytest.raises(TrainingError, match="requires a db:// or namespace-backed lake"):
        lake.training.dataset("demo-v1", backend="enterprise")


def test_training_dataset_records_explicit_enterprise_fallback(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        allow_fallback=True,
    )

    report = dataset.manifest.to_dict()["backend"]
    assert report["requested_backend"] == "enterprise"
    assert report["resolved_backend"] == "local"
    assert report["fallback"] == {
        "from": "enterprise",
        "to": "local",
        "reason": "lake connection kind 'local_path' is not Enterprise remote",
    }
    assert dataset.manifest.access_pattern == "lance-native-snapshot"


def test_training_dataset_reports_mocked_enterprise_backend(lake):
    _mark_enterprise_lake(lake)

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
        prewarm=True,
    )
    report = dataset.manifest.to_dict()["backend"]

    assert dataset.manifest.access_pattern == "enterprise-remote-snapshot"
    assert dataset.manifest.accounting["target_format"] == "lance-native-enterprise-training"
    assert report["resolved_backend"] == "enterprise"
    assert report["connection_kind"] == "lancedb_remote_db"
    assert report["display_uri"] == "db://robotics"
    assert report["request_routing"] == {
        "mode": "host-override",
        "http_endpoint": "https://phalanx.acme.internal",
        "host_override": "https://phalanx.acme.internal",
        "region": "us-west-2",
        "all_requests_use_host_override": True,
    }
    assert report["plan_executor"]["available"] is True
    assert report["cache"]["policy"] == "epoch"
    assert report["cache"]["scope"] == "plan-executor"
    assert report["cache"]["prewarm_requested"] is True
    assert report["cache"]["prewarm_executed"] is False
    assert report["cache"]["prewarm_status"] == "planned"
    assert report["metrics"]["row_plan_id"] == dataset.row_plan.plan_id
    assert report["metrics"]["rows_planned"] == 3

    batch = next(iter_training_batches(dataset, batch_size=2))
    assert batch["_lineage"]["backend"]["resolved_backend"] == "enterprise"
    assert batch["_lineage"]["backend"]["connection_kind"] == "lancedb_remote_db"

    loader_config = dataset.loader_config()
    assert loader_config["connection"]["remote"] == {
        "remote_auth_ref": "enterprise-prod",
        "region": "us-west-2",
        "host_override": "https://phalanx.acme.internal",
    }
    assert "secret-api-key" not in str(loader_config)


def test_enterprise_epoch_cache_policy_submits_prewarm_request_and_status(lake):
    _mark_enterprise_lake(lake)
    requests = []

    def prewarm(request):
        requests.append(request)
        return {
            "status": "active",
            "completed_executors": 0,
            "failed_executors": 0,
            "pe_fanout": 2,
        }

    def status(**kwargs):
        return {
            "status": "complete",
            "prewarm_id": kwargs["prewarm_id"],
            "completed_executors": 2,
            "failed_executors": 0,
            "cache_hits": 4,
            "cache_misses": 0,
            "warm_bytes": 128,
            "duration_ms": 12.5,
        }

    lake.page_cache_prewarm = prewarm
    lake.page_cache_prewarm_status = status

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "caption"],
        backend="enterprise",
        cache_policy="epoch",
        prewarm_options={"wait": True},
    )

    assert len(requests) == 1
    request = requests[0]
    assert request["prewarm_id"].startswith("prewarm-")
    assert request["policy"] == "epoch"
    assert request["row_plan_id"] == dataset.row_plan.plan_id
    assert request["epoch_plan_id"] == dataset.epoch_plan.plan_id
    assert request["worker"]["sample_count"] == 3
    table = request["tables"][0]
    assert table["table"] == "observations"
    assert table["version"] is not None
    assert table["projected_columns"] == ["observation_id", "caption"]
    assert sum(item["count"] for item in table["row_id_ranges"]) == 3

    cache = dataset.manifest.backend["cache"]
    assert cache["prewarm_executed"] is True
    assert cache["prewarm_status"] == "complete"
    assert cache["prewarm_id"] == request["prewarm_id"]
    assert dataset.prewarm_status()["status"] == "complete"
    metrics = dataset.manifest.backend["metrics"]
    assert metrics["prewarm_requests"] == 1
    assert metrics["prewarm_completed_executors"] == 2
    assert metrics["prewarm_failed_executors"] == 0
    assert metrics["prewarm_cache_hits"] == 4
    assert metrics["prewarm_warm_bytes"] == 128


def test_enterprise_none_cache_policy_never_submits_prewarm(lake):
    _mark_enterprise_lake(lake)
    requests = []
    lake.page_cache_prewarm = requests

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="none",
        prewarm=True,
    )

    assert requests == []
    cache = dataset.manifest.backend["cache"]
    assert cache["prewarm_requested"] is False
    assert cache["prewarm_executed"] is False
    assert cache["prewarm_status"] == "not-requested"
    assert cache["prewarm_requests"] == []
    assert any("prewarm=True is ignored" in warning for warning in dataset.manifest.backend["warnings"])


def test_enterprise_prewarm_excludes_payload_columns_until_opted_in(lake):
    _mark_enterprise_lake(lake)
    requests = []
    lake.page_cache_prewarm = requests

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
        cache_policy="epoch",
    )

    request = requests[0]
    assert PAYLOAD_BLOB_COLUMN not in request["projected_columns"]
    assert request["projected_columns"] == ["observation_id"]
    assert request["excluded_columns"] == [
        {
            "column": "payload",
            "reason": "heavy media prewarm requires include_heavy=True",
        }
    ]
    assert dataset.manifest.backend["cache"]["prewarm_executed"] is True

    opted_in_requests = []
    lake.page_cache_prewarm = opted_in_requests
    opted_in = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
        cache_policy="epoch",
        prewarm_options={"include_heavy": True},
    )

    assert PAYLOAD_BLOB_COLUMN in opted_in_requests[0]["projected_columns"]
    assert opted_in.manifest.backend["cache"]["prewarm_executed"] is True


def test_enterprise_prewarm_limits_skip_runaway_requests(lake):
    _mark_enterprise_lake(lake)
    requests = []
    lake.page_cache_prewarm = requests

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
        prewarm_options={"max_rows": 1},
    )

    assert requests == []
    cache = dataset.manifest.backend["cache"]
    assert cache["prewarm_executed"] is False
    assert cache["prewarm_status"] == "skipped"
    assert "exceeds max_rows" in cache["prewarm_status_detail"]["reason"]


def test_enterprise_prewarm_fake_warm_cache_reports_fewer_misses(tmp_path):
    cold_lake = _mark_enterprise_lake(_training_lake(tmp_path / "cold.lance"))
    cold_lake.query_node_cache_telemetry = {"remote_take": {"hits": 0, "misses": 3}}
    cold = cold_lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
        cache_policy="lazy",
    )
    next(iter_training_batches(cold, batch_size=3))

    warm_lake = _mark_enterprise_lake(_training_lake(tmp_path / "warm.lance"))

    def prewarm(request):
        warm_lake.query_node_cache_telemetry = {
            "remote_take": {"hits": request["row_count"], "misses": 0}
        }
        return {"status": "complete", "completed_executors": 1, "failed_executors": 0}

    warm_lake.page_cache_prewarm = prewarm
    warm = warm_lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
        cache_policy="epoch",
    )
    next(iter_training_batches(warm, batch_size=3))

    assert cold.manifest.backend["metrics"]["cache_misses"] == 3
    assert warm.manifest.backend["metrics"]["cache_misses"] == 0


# --------------------------------------------------------------------------- #
# Backlog 0121: durable, deduplicated prewarm JobRun lifecycle                 #
# --------------------------------------------------------------------------- #


def test_enterprise_prewarm_jobrun_dedups_across_workers(lake):
    _mark_enterprise_lake(lake)
    lake.prewarm_job_store = InMemoryPrewarmJobRunStore()
    calls = []

    def prewarm(request):
        calls.append(request)
        return {"status": "complete", "completed_executors": 2, "failed_executors": 0}

    lake.page_cache_prewarm = prewarm

    worker0 = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "caption"],
        backend="enterprise",
        cache_policy="epoch",
        worker_id=0,
        num_workers=2,
    )
    worker1 = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "caption"],
        backend="enterprise",
        cache_policy="epoch",
        worker_id=1,
        num_workers=2,
    )

    prewarm_id = worker0.manifest.backend["cache"]["prewarm_id"]
    # One durable JobRun, one submitted remote operation, two status reads.
    assert worker1.manifest.backend["cache"]["prewarm_id"] == prewarm_id
    assert len(calls) == 1
    jobs = lake.training.prewarm_jobs()
    assert len(jobs) == 1
    run = lake.training.prewarm_job(prewarm_id)
    assert run["attach_count"] == 2
    assert set(run["workers"]) == {"worker-0/2", "worker-1/2"}
    assert run["status"] == "complete"
    assert worker0.prewarm_status()["status"] == "complete"
    assert worker1.prewarm_status()["status"] == "complete"
    # The status envelope surfaces terminal lifecycle detail (AC2).
    status = lake.training.prewarm_job_status(prewarm_id)
    assert status["job_run_id"] == prewarm_id
    assert status["completed_executors"] == 2
    assert status["retry_count"] == 0
    assert status["completed_at"] is not None


def test_enterprise_prewarm_jobrun_failed_raises_under_fail_fast(lake):
    _mark_enterprise_lake(lake)
    lake.prewarm_job_store = InMemoryPrewarmJobRunStore()

    def prewarm(request):
        raise RuntimeError("plan executor unreachable")

    lake.page_cache_prewarm = prewarm

    with pytest.raises(PrewarmUnavailableError):
        lake.training.dataset(
            "demo-v1",
            columns=["observation_id"],
            backend="enterprise",
            cache_policy="epoch",
            prewarm_options={"on_error": "raise"},
        )
    # The failed JobRun is still durably recorded with its terminal reason.
    jobs = lake.training.prewarm_jobs(status="failed")
    assert len(jobs) == 1
    assert "plan executor unreachable" in jobs[0]["terminal_reason"]


def test_enterprise_prewarm_jobrun_failed_warns_and_proceeds(lake):
    _mark_enterprise_lake(lake)
    lake.prewarm_job_store = InMemoryPrewarmJobRunStore()

    def prewarm(request):
        raise RuntimeError("plan executor unreachable")

    lake.page_cache_prewarm = prewarm

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
        prewarm_options={"on_error": "warn"},
    )
    assert dataset.prewarm_status()["status"] == "failed"
    assert any(
        "did not complete" in warning for warning in dataset.manifest.backend["warnings"]
    )


def test_enterprise_prewarm_jobrun_reuses_warm_until_ttl_then_resubmits(lake):
    _mark_enterprise_lake(lake)
    lake.prewarm_job_store = InMemoryPrewarmJobRunStore()
    clock = {"t": datetime(2026, 7, 6, tzinfo=UTC)}
    lake.prewarm_clock = lambda: clock["t"]
    lake.prewarm_job_ttl_s = 60.0
    calls = []

    def prewarm(request):
        calls.append(request)
        return {"status": "complete", "completed_executors": 1}

    lake.page_cache_prewarm = prewarm

    def open_epoch():
        return lake.training.dataset(
            "demo-v1",
            columns=["observation_id"],
            backend="enterprise",
            cache_policy="epoch",
        )

    first = open_epoch()
    prewarm_id = first.manifest.backend["cache"]["prewarm_id"]
    # Reopening within TTL reuses the warm JobRun -- no second remote submit.
    second = open_epoch()
    assert second.manifest.backend["cache"]["prewarm_id"] == prewarm_id
    assert len(calls) == 1
    assert lake.training.prewarm_job(prewarm_id)["status"] == "complete"

    # Past the TTL the JobRun expires and a fresh open re-submits (retry_count++).
    clock["t"] = datetime(2026, 7, 6, 1, tzinfo=UTC)
    expired = lake.training.expire_prewarm_jobs()
    assert any(job["prewarm_id"] == prewarm_id for job in expired)
    third = open_epoch()
    assert len(calls) == 2
    assert third.manifest.backend["cache"]["prewarm_id"] == prewarm_id
    assert lake.training.prewarm_job(prewarm_id)["retry_count"] == 1


def test_enterprise_prewarm_jobrun_canceled_is_not_warm(lake):
    _mark_enterprise_lake(lake)
    lake.prewarm_job_store = InMemoryPrewarmJobRunStore()
    calls = []

    def prewarm(request):
        calls.append(request)
        return {"status": "active", "pe_fanout": 2}

    lake.page_cache_prewarm = prewarm

    first = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
    )
    prewarm_id = first.manifest.backend["cache"]["prewarm_id"]
    canceled = lake.training.cancel_prewarm_job(prewarm_id, reason="no longer needed")
    assert canceled["status"] == "canceled"

    # A canceled JobRun is not warm: reopening re-submits rather than reusing it.
    lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
    )
    assert len(calls) == 2
    assert lake.training.prewarm_job(prewarm_id)["status"] == "active"


def test_enterprise_prewarm_jobrun_config_and_record_are_secret_free(lake):
    _mark_enterprise_lake(lake)
    lake.prewarm_job_store = InMemoryPrewarmJobRunStore()
    lake.page_cache_prewarm = lambda request: {"status": "complete"}

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
    )
    prewarm_id = dataset.manifest.backend["cache"]["prewarm_id"]

    config_json = json.dumps(dataset.loader_config())
    assert "secret-api-key" not in config_json
    assert "api_key" not in config_json

    record = lake.training.prewarm_job(prewarm_id)
    record_json = json.dumps(record)
    assert "secret-api-key" not in record_json
    assert "api_key" not in record_json
    # Only non-secret routing survives (host override / region), never credentials.
    assert record["routing"]["mode"] == "host-override"
    assert record["routing"]["host_override"] == "https://phalanx.acme.internal"


def test_enterprise_prewarm_jobrun_durable_store_survives_reopen(tmp_path):
    path = tmp_path / "durable.lance"
    lake = _mark_enterprise_lake(_training_lake(path))
    lake.prewarm_jobs_durable = True
    calls = []

    def prewarm(request):
        calls.append(request)
        return {"status": "complete", "completed_executors": 1}

    lake.page_cache_prewarm = prewarm

    lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
        worker_id=0,
        num_workers=2,
    )
    lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
        worker_id=1,
        num_workers=2,
    )
    assert len(calls) == 1

    # A fresh Lake over the same durable table sees the JobRun (cross-process).
    reopened = _mark_enterprise_lake(_training_lake(path))
    jobs = reopened.training.prewarm_jobs()
    assert len(jobs) == 1
    assert jobs[0]["attach_count"] == 2
    assert jobs[0]["status"] == "complete"


def test_training_loader_config_reopens_enterprise_workers_with_host_override(
    lake, monkeypatch
):
    _mark_enterprise_lake(lake)
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
    )
    opened: dict[str, object] = {}

    def fake_open(uri, **kwargs):
        opened["uri"] = uri
        opened["kwargs"] = kwargs
        return lake

    monkeypatch.setattr(training_mod.Lake, "open", fake_open)

    worker_dataset = training_mod._dataset_from_loader_config(
        dataset.loader_config(),
        worker_id=1,
        num_workers=2,
    )

    assert opened == {
        "uri": str(lake.uri),
        "kwargs": {
            "remote_auth_ref": "enterprise-prod",
            "region": "us-west-2",
            "host_override": "https://phalanx.acme.internal",
            "client_config": None,
        },
    }
    assert worker_dataset.epoch_plan.worker_id == 1
    assert worker_dataset.epoch_plan.num_workers == 2
    assert worker_dataset.manifest.backend["request_routing"]["http_endpoint"] == (
        "https://phalanx.acme.internal"
    )


def test_enterprise_training_setup_uses_remote_scans_without_to_arrow(lake, monkeypatch):
    _mark_enterprise_lake(lake)

    def fail_snapshot_context(*args, **kwargs):
        raise AssertionError("enterprise training should not use local snapshot export context")

    class NoToArrowTable:
        def __init__(self, table):
            self._table = table

        def to_arrow(self, *args, **kwargs):
            raise AssertionError("enterprise training setup should not call table.to_arrow()")

        def __getattr__(self, name):
            return getattr(self._table, name)

    open_table = lake.table
    monkeypatch.setattr(training_mod, "_snapshot_context", fail_snapshot_context)
    monkeypatch.setattr(lake, "table", lambda name: NoToArrowTable(open_table(name)))

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "run_id"],
        backend="enterprise",
    )

    assert [sample["observation_id"] for sample in dataset] == [
        "obs-camera-0",
        "obs-camera-1",
        "obs-camera-2",
    ]
    assert dataset.manifest.access_pattern == "enterprise-remote-snapshot"
    assert dataset.manifest.backend["display_uri"] == "db://robotics"


def test_enterprise_and_local_paths_emit_equivalent_sample_ids(lake):
    local = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=31,
        epoch=1,
    )
    _mark_enterprise_lake(lake)
    remote = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=31,
        epoch=1,
        backend="enterprise",
    )

    assert [sample["observation_id"] for sample in remote] == [
        sample["observation_id"] for sample in local
    ]
    assert remote.row_plan.frame_ids == local.row_plan.frame_ids
    assert remote.manifest.backend["resolved_backend"] == "enterprise"


def test_enterprise_batch_hydration_coalesces_duplicate_row_ids(lake):
    _mark_enterprise_lake(lake)
    lake.query_node_cache_telemetry = {
        "remote_take": {
            "per_addr": {
                "pe-a": {"hits": 5, "misses": 1},
                "pe-b": {"hits": 2, "misses": 0},
            }
        }
    }
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
    )

    samples = dataset.__getitems__([0, 0, 1])

    assert [sample["observation_id"] for sample in samples] == [
        "obs-camera-0",
        "obs-camera-0",
        "obs-camera-1",
    ]
    assert [sample["payload"] for sample in samples] == [
        b"frame-0",
        b"frame-0",
        b"frame-1",
    ]
    metrics = dataset.manifest.backend["metrics"]
    assert metrics["remote_take_requests"] == 1
    assert metrics["row_ids_requested"] == 3
    assert metrics["row_ids_unique"] == 2
    assert metrics["row_ids_coalesced"] == 1
    assert metrics["bytes_read"] == len(b"frame-0") + len(b"frame-1")
    assert metrics["cache_hits"] == 7
    assert metrics["cache_misses"] == 1
    assert metrics["pe_fanout"] == 2
    assert metrics["operations"][0]["columns"] == ["payload_blob"]
    assert metrics["operations"][0]["coalescing_window"] == "native-training-batch"


def test_enterprise_loader_report_is_json_serializable_and_redacted(lake):
    _mark_enterprise_lake(lake)
    lake.query_node_cache_telemetry = {
        "remote_take": {
            "per_addr": {
                "pe-a": {"hits": 4, "misses": 1},
                "pe-b": {"hits": 1, "misses": 0},
            }
        }
    }
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
        cache_policy="lazy",
        shuffle=True,
        shuffle_seed=17,
        epoch=2,
        worker_id=0,
        num_workers=2,
    )

    dataset.__getitems__([0, 0, 1])
    report = dataset.loader_report(
        training_run_id="train-run-1",
        model_run_id="wandb-run-7",
        extra={
            "authorization_header": "Bearer secret-token",
            "storage_secret_key": "super-secret",
            "tracker": "wandb",
        },
    ).to_dict()

    encoded = json.dumps(report, sort_keys=True)
    assert report["kind"] == "lancedb-robotics/training-loader-report/v1"
    assert report["loader"] == {
        "kind": "native-training",
        "access_pattern": "enterprise-remote-snapshot",
    }
    assert report["snapshot"] == {
        "id": dataset.manifest.dataset_id,
        "name": "demo-v1",
    }
    assert report["plans"]["row_plan_id"] == dataset.row_plan.plan_id
    assert report["plans"]["epoch_plan_id"] == dataset.epoch_plan.plan_id
    assert report["plans"]["worker"] == {
        "id": 0,
        "num_workers": 2,
        "resume_from": 0,
    }
    assert report["policies"]["media"]["policy"] == "bytes"
    assert report["policies"]["enterprise_cache"]["policy"] == "lazy"
    assert report["remote_execution"]["plan_executor"]["remote_take"] is True
    assert report["metrics"]["operations_by_type"]["remote_take"] == 1
    assert report["metrics"]["cache"]["hits"] == 5
    assert report["metrics"]["cache"]["misses"] == 1
    assert report["metrics"]["cache"]["by_worker"]["0/2"] == {
        "hits": 5,
        "misses": 1,
    }
    assert report["metrics"]["cache"]["by_epoch"]["2"] == {"hits": 5, "misses": 1}
    assert report["metrics"]["cache"]["by_batch"]["native-training-batch"] == {
        "hits": 5,
        "misses": 1,
    }
    assert report["run"] == {
        "training_run_id": "train-run-1",
        "model_run_id": "wandb-run-7",
        "tracker": "wandb",
        "authorization_header": "<redacted>",
        "storage_secret_key": "<redacted>",
    }
    assert "secret-token" not in encoded
    assert "super-secret" not in encoded
    assert dataset.manifest.to_dict()["loader_report"]["kind"] == report["kind"]


def test_training_loader_report_records_explicit_fallback(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        allow_fallback=True,
    )

    report = dataset.loader_report().to_dict()

    assert report["remote_execution"]["resolved_backend"] == "local"
    assert report["fallback_events"] == [
        {
            "from": "enterprise",
            "to": "local",
            "reason": "lake connection kind 'local_path' is not Enterprise remote",
        }
    ]
    assert "server_side_query" in report["disabled_capabilities"]


def test_enterprise_metadata_only_hydration_does_not_read_payload_columns(
    lake,
    monkeypatch,
):
    _mark_enterprise_lake(lake)

    def fail_take_blobs(*args, **kwargs):
        raise AssertionError("metadata-only Enterprise samples must not hydrate payload bytes")

    monkeypatch.setattr(
        training_mod._QueryNodeHydrationExecutor,
        "take_blobs",
        fail_take_blobs,
    )

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="metadata",
        backend="enterprise",
    )
    samples = dataset.__getitems__([0, 1])

    assert [sample["observation_id"] for sample in samples] == [
        "obs-camera-0",
        "obs-camera-1",
    ]
    assert all(isinstance(sample["payload"], TrainingMediaHandle) for sample in samples)
    assert dataset.manifest.backend["metrics"].get("hydration_requests", 0) in {0, None}


def test_enterprise_and_local_paths_hydrate_equivalent_payloads_and_lineage(lake):
    local = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
    )
    _mark_enterprise_lake(lake)
    remote = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
    )

    local_batch = next(iter_training_batches(local, batch_size=2))
    remote_batch = next(iter_training_batches(remote, batch_size=2))

    assert remote_batch["observation_id"] == local_batch["observation_id"]
    assert remote_batch["payload"] == local_batch["payload"]
    assert remote_batch["_lineage"]["frame_ids"] == local_batch["_lineage"]["frame_ids"]
    assert remote_batch["_lineage"]["row_ids"] == local_batch["_lineage"]["row_ids"]
    assert remote_batch["_lineage"]["backend"]["resolved_backend"] == "enterprise"
    assert remote_batch["_lineage"]["backend"]["metrics"]["remote_take_requests"] == 1


def test_enterprise_worker_partitions_cover_large_epoch_once(tmp_path):
    lake = _mark_enterprise_lake(
        _training_lake(tmp_path / "robot-large.lance", frame_count=103)
    )
    base = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
        backend="enterprise",
    )
    workers = [
        lake.training.dataset(
            "demo-v1",
            columns=["observation_id"],
            shuffle=True,
            shuffle_seed=23,
            epoch=2,
            worker_id=worker_id,
            num_workers=4,
            backend="enterprise",
        )
        for worker_id in range(4)
    ]

    expected = [sample["observation_id"] for sample in base]
    merged = []
    for offset in range(max(len(worker) for worker in workers)):
        for worker in workers:
            if offset < len(worker):
                merged.append(worker[offset]["observation_id"])

    assert merged == expected
    assert len(merged) == len(set(merged)) == 103
    assert workers[2].manifest.to_dict()["worker"] == {
        "id": 2,
        "num_workers": 4,
        "resume_from": 0,
    }

    resumed_workers = [
        lake.training.dataset(
            "demo-v1",
            columns=["observation_id"],
            shuffle=True,
            shuffle_seed=23,
            epoch=2,
            worker_id=worker_id,
            num_workers=4,
            resume_from=5,
            backend="enterprise",
        )
        for worker_id in range(4)
    ]
    resumed_merged = []
    for offset in range(max(len(worker) for worker in resumed_workers)):
        for worker in resumed_workers:
            if offset < len(worker):
                resumed_merged.append(worker[offset]["observation_id"])

    assert resumed_merged == expected[5:]
    assert len(resumed_merged) == len(set(resumed_merged)) == 98


@pytest.mark.torch_loader
def test_enterprise_torch_iterable_dataloader_multi_worker(
    lake, monkeypatch
):
    require_torch_loader()
    # Fork inherits the parent address space, so the in-process monkeypatched
    # Enterprise lake below survives into the workers (a mocked db:// endpoint
    # cannot be reopened by a fresh spawn process). Windows has no fork; that
    # combination is documented and skipped, not silently ignored.
    require_start_method("fork")

    _mark_enterprise_lake(lake)
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
        backend="enterprise",
    )
    monkeypatch.setattr(
        training_mod,
        "_open_lake_from_loader_config",
        lambda config: lake,
    )
    loader = to_torch_dataloader(
        dataset,
        batch_size=1,
        num_workers=2,
        adapter="iterable",
        multiprocessing_context="fork",
    )

    seen = []
    worker_ids = set()
    for batch in loader:
        seen.extend(batch["observation_id"])
        worker_ids.add(batch["_lineage"]["worker"]["id"])
        assert batch["_lineage"]["backend"]["display_uri"] == "db://robotics"

    expected = [sample["observation_id"] for sample in dataset]
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == len(expected)
    assert worker_ids == {0, 1}


def test_enterprise_backend_missing_remote_capability_has_guidance(lake):
    _mark_enterprise_lake(lake, server_side_query=False)

    with pytest.raises(TrainingError, match="remote query-node planning"):
        lake.training.dataset(
            "demo-v1",
            columns=["observation_id"],
            backend="enterprise",
        )


def test_enterprise_missing_remote_take_fail_policy_blocks_before_iteration(lake):
    _mark_enterprise_lake(lake)
    lake.enterprise_training_capabilities = {"remote_take": False}

    with pytest.raises(UnsupportedRemoteOperationError, match="remote_take"):
        lake.training.dataset(
            "demo-v1",
            columns=["observation_id", "payload"],
            media="bytes",
            backend="enterprise",
            fallback="fail",
        )


def test_enterprise_warn_policy_downgrades_missing_prewarm_to_lazy_cache(lake):
    _mark_enterprise_lake(lake)
    lake.enterprise_training_capabilities = {
        "page_cache_prewarm": False,
        "page_cache_status": False,
    }
    requests = []
    lake.page_cache_prewarm = requests

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        backend="enterprise",
        cache_policy="epoch",
        fallback="warn",
    )

    backend = dataset.manifest.backend
    assert requests == []
    assert backend["resolved_backend"] == "enterprise"
    assert backend["cache"]["policy"] == "lazy"
    assert backend["cache"]["requested_policy"] == "epoch"
    assert backend["cache"]["prewarm_requested"] is False
    assert backend["fallback_events"][0]["to"] == "lazy-cache"
    assert backend["fallback_events"][0]["missing_capabilities"] == [
        "page_cache_prewarm",
        "page_cache_status",
    ]
    assert dataset.loader_report().to_dict()["fallback_events"][0]["to"] == "lazy-cache"


def test_enterprise_warn_policy_rejects_unreported_payload_materialization(lake):
    _mark_enterprise_lake(lake)
    lake.enterprise_training_capabilities = {"remote_take": False}

    with pytest.raises(UnsupportedRemoteOperationError, match="fallback='warn'"):
        lake.training.dataset(
            "demo-v1",
            columns=["observation_id", "payload"],
            media="bytes",
            backend="enterprise",
            fallback="warn",
        )


def test_enterprise_direct_fallback_records_authorized_data_plane(lake):
    _mark_enterprise_lake(lake, direct_object_io=True)
    lake.enterprise_training_capabilities = {"remote_take": False}

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
        backend="enterprise",
        fallback="direct",
    )

    assert dataset.manifest.backend["resolved_backend"] == "local"
    assert dataset.manifest.backend["fallback"]["to"] == "direct-data-plane"
    assert dataset.manifest.backend["fallback"]["preserves_versions"] is True
    assert dataset[0]["payload"] == b"frame-0"


def test_enterprise_missing_auth_has_typed_remediation(lake):
    _mark_enterprise_lake(lake)
    lake.connection_spec = LakeConnectionSpec(
        kind="lancedb_remote_db",
        uri="db://robotics",
        display_uri="db://robotics",
        lancedb_connect_kwargs={"region": "us-west-2"},
        capabilities=LakeCapabilities(server_side_query=True, blob_fetch_remote=True),
    )
    lake.capabilities = lake.connection_spec.capabilities

    with pytest.raises(MissingEnterpriseAuthError, match="remote_auth_ref"):
        lake.training.dataset(
            "demo-v1",
            columns=["observation_id"],
            backend="enterprise",
        )


def test_enterprise_stale_table_version_error_includes_snapshot_and_version(
    lake,
    monkeypatch,
):
    _mark_enterprise_lake(lake)
    open_table = lake.table

    class StaleScenariosTable:
        def __init__(self, table):
            self._table = table

        def checkout(self, version):
            raise RuntimeError(f"version {version} was compacted away")

        def __getattr__(self, name):
            return getattr(self._table, name)

    def table(name):
        opened = open_table(name)
        return StaleScenariosTable(opened) if name == "scenarios" else opened

    monkeypatch.setattr(lake, "table", table)

    with pytest.raises(StaleTableVersionError) as exc_info:
        lake.training.dataset(
            "demo-v1",
            columns=["observation_id"],
            backend="enterprise",
        )

    message = str(exc_info.value)
    assert "snapshot ds-" in message
    assert "requested version" in message
    assert "recreate the training snapshot" in message


def test_torch_iterable_missing_dependency_has_install_guidance(lake, monkeypatch):
    dataset = lake.training.dataset("demo-v1", columns=["observation_id"])
    monkeypatch.setattr(training_mod, "torch_available", lambda: False)

    with pytest.raises(TrainingError, match=r"lancedb-robotics\[torch\]"):
        dataset.torch(iterable=True)


def test_training_dataset_metadata_policy_returns_lazy_handles_without_blob_reads(
    lake, monkeypatch
):
    def fail_payload_preload(*args, **kwargs):
        raise AssertionError("metadata training context should not preload payload blobs")

    def fail_fetch_blob(*args, **kwargs):
        raise AssertionError("metadata samples should not fetch blob bytes")

    monkeypatch.setattr(
        "lancedb_robotics.dataset_export._payload_blobs_as_of",
        fail_payload_preload,
    )
    monkeypatch.setattr(training_mod, "fetch_blob", fail_fetch_blob)

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="metadata",
    )
    sample = dataset[0]

    assert sample["observation_id"] == "obs-camera-0"
    assert isinstance(sample["payload"], TrainingMediaHandle)
    assert sample["payload"].observation_id == "obs-camera-0"
    assert sample["_media"]["fields"]["payload"]["materialized"] is False
    assert dataset.row_plan.materialization_policies["payload"].startswith("metadata:")


def test_training_dataset_bytes_policy_reads_only_selected_payload(lake, monkeypatch):
    calls: list[str] = []
    original = training_mod.fetch_blob

    def counted_fetch_blob(handle, blob_column, row_id, *, id_column):
        calls.append(row_id)
        return original(handle, blob_column, row_id, id_column=id_column)

    monkeypatch.setattr(training_mod, "fetch_blob", counted_fetch_blob)

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "payload"],
        media="bytes",
    )
    sample = dataset[2]

    assert sample["observation_id"] == "obs-camera-2"
    assert sample["payload"] == b"frame-2"
    assert calls == ["obs-camera-2"]
    assert sample["_media"]["fields"]["payload"]["bytes_read"] == len(b"frame-2")


def test_training_dataset_epoch_cache_reuses_payload_reads(lake, monkeypatch):
    calls: list[str] = []
    original = training_mod.fetch_blob

    def counted_fetch_blob(handle, blob_column, row_id, *, id_column):
        calls.append(row_id)
        return original(handle, blob_column, row_id, id_column=id_column)

    monkeypatch.setattr(training_mod, "fetch_blob", counted_fetch_blob)

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["payload", "payload_size"],
        media="bytes",
        media_cache="epoch",
    )
    first = dataset[1]
    second = dataset[1]

    assert first["payload"] == second["payload"] == b"frame-1"
    assert first["payload_size"] == second["payload_size"] == len(b"frame-1")
    assert calls == ["obs-camera-1"]
    assert first["_media"]["fields"]["payload_size"]["cache_hit"] is True
    assert second["_media"]["fields"]["payload"]["cache_hit"] is True


def test_training_dataset_decoded_media_missing_dependencies_are_targeted(
    lake, monkeypatch
):
    original_find_spec = training_mod.importlib.util.find_spec

    def missing_pillow(module):
        if module == "PIL":
            return None
        return original_find_spec(module)

    monkeypatch.setattr(training_mod.importlib.util, "find_spec", missing_pillow)

    metadata = lake.training.dataset("demo-v1", columns=["payload"], media="metadata")
    assert isinstance(metadata[0]["payload"], TrainingMediaHandle)

    bytes_dataset = lake.training.dataset("demo-v1", columns=["payload"], media="bytes")
    assert bytes_dataset[0]["payload"] == b"frame-0"

    decoded = lake.training.dataset("demo-v1", columns=["payload"], media="array")
    with pytest.raises(TrainingError, match="media='array'.*PIL"):
        decoded[0]


def test_training_dataset_projects_filters_and_shuffles(lake):
    first = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "topic"],
        filters={"topic": "/camera/front"},
        shuffle=True,
        shuffle_seed=7,
    )
    second = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "topic"],
        filters={"topic": "/camera/front"},
        shuffle=True,
        shuffle_seed=7,
    )

    assert first.manifest.filters == {"topic": "/camera/front"}
    assert first.manifest.shuffle is True
    assert first.manifest.shuffle_seed == 7
    assert [row["observation_id"] for row in first] == [
        row["observation_id"] for row in second
    ]
    assert all(set(row) == {"observation_id", "topic"} for row in first)


def test_training_dataset_builds_row_plan_with_lance_pushdown(lake):
    split = lake.training.dataset("demo-v1", columns=["split"])[0]["split"]

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "topic", "modality", "task"],
        filters={"split": split, "task": "pick_place", "modality": "image"},
    )

    assert [row["observation_id"] for row in dataset] == [
        "obs-camera-0",
        "obs-camera-1",
        "obs-camera-2",
    ]
    assert dataset.row_plan.frame_ids == (
        "obs-camera-0",
        "obs-camera-1",
        "obs-camera-2",
    )
    assert all(isinstance(row_id, int) for row_id in dataset.row_plan.row_ids)
    scan = dataset.row_plan.scan
    assert scan["table"] == "observations"
    assert {"observation_id", "topic", "modality", "task_id"} <= set(scan["columns"])
    assert "modality = 'image'" in scan["filter_predicate"]
    assert "task_id = 'pick_place'" in scan["filter_predicate"]
    assert f"split = '{split}'" in scan["logical_predicates"]
    assert dataset.manifest.to_dict()["row_plan_id"] == dataset.row_plan.plan_id


def test_training_dataset_epoch_plans_are_deterministic_and_resumable(lake):
    first = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=17,
        epoch=0,
    )
    second = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=17,
        epoch=0,
    )
    next_epoch = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=17,
        epoch=1,
    )

    first_ids = [row["observation_id"] for row in first]
    assert first_ids == [row["observation_id"] for row in second]
    assert first.epoch_plan.plan_id == second.epoch_plan.plan_id
    assert first.epoch_plan.backend.kind == "lancedb_permutation"
    assert second.epoch_plan.backend.permutation_table == first.epoch_plan.backend.permutation_table
    assert first_ids != [row["observation_id"] for row in next_epoch]
    assert first[0] == list(first)[0]

    resumed = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=17,
        epoch=0,
        resume_from=1,
    )
    assert [row["observation_id"] for row in resumed] == first_ids[1:]
    assert resumed.epoch_plan.sample_indices == first.epoch_plan.sample_indices[1:]
    assert resumed.epoch_plan.backend.kind == "lancedb_permutation"


def test_training_dataset_worker_partitions_cover_epoch_once(lake):
    base = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
    )
    workers = [
        lake.training.dataset(
            "demo-v1",
            columns=["observation_id"],
            shuffle=True,
            shuffle_seed=23,
            epoch=2,
            worker_id=worker_id,
            num_workers=2,
        )
        for worker_id in range(2)
    ]

    base_ids = [row["observation_id"] for row in base]
    worker_ids = [[row["observation_id"] for row in worker] for worker in workers]
    flattened = [item for partition in worker_ids for item in partition]
    assert sorted(flattened) == sorted(base_ids)
    assert len(flattened) == len(set(flattened)) == len(base_ids)
    assert all(
        worker.epoch_plan.backend.kind == "lancedb_permutation" for worker in workers
    )
    assert workers[0].manifest.to_dict()["worker"] == {
        "id": 0,
        "num_workers": 2,
        "resume_from": 0,
    }


def test_training_dataset_time_windows_use_canonical_fields(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "timestamp_ns"],
        time_windows={"state_vector": [-1.0, 0.0, 1.0]},
    )

    sample = dataset[1]
    window = sample["windows"]["state_vector"]
    assert [item["timestamp_ns"] for item in window] == [0, 1_000_000_000, 2_000_000_000]
    assert [item["value"] for item in window] == [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]
    assert [item["frame_index"] for item in window] == [0, 1, 2]


def test_training_dataset_function_matches_lake_namespace(lake):
    direct = training_dataset(lake, "demo-v1", columns=["observation_id"])
    via_lake = lake.training.dataset("demo-v1", columns=["observation_id"])

    assert [row["observation_id"] for row in direct] == [
        row["observation_id"] for row in via_lake
    ]


def test_training_dataset_rejects_unknown_projection_filter_and_window(lake):
    with pytest.raises(TrainingError, match="unknown training columns"):
        lake.training.dataset("demo-v1", columns=["observation.images.camera_front"])

    with pytest.raises(TrainingError, match="unknown training filters"):
        lake.training.dataset("demo-v1", filters={"observation.images.camera_front": "x"})

    with pytest.raises(TrainingError, match="unknown time window columns"):
        lake.training.dataset("demo-v1", time_windows={"observation.state": [0.0]})


def test_training_batch_iterator_collates_schema_lineage_media_and_windows(lake):
    dataset = lake.training.dataset(
        "demo-v1",
        columns=[
            "observation_id",
            "timestamp_ns",
            "state_vector",
            "action_vector",
            "payload",
            "raw_uri",
            "raw_channel",
            "raw_sequence",
        ],
        time_windows={"state_vector": [-1.0, 0.0, 1.0]},
        media="metadata",
    )

    batches = list(iter_training_batches(dataset, batch_size=2))

    assert len(batches) == 2
    batch = batches[0]
    assert batch["_schema"]["columns"] == [
        "observation_id",
        "timestamp_ns",
        "state_vector",
        "state_vector_mask",
        "action_vector",
        "action_vector_mask",
        "payload",
        "raw_uri",
        "raw_channel",
        "raw_sequence",
        "windows",
    ]
    assert batch["observation_id"] == ["obs-camera-0", "obs-camera-1"]
    assert batch["state_vector"] == [[0.0, 0.5], [1.0, 1.5]]
    assert batch["state_vector_mask"] == [[True, True], [True, True]]
    assert batch["action_vector"] == [[10.0, -10.0], [11.0, -11.0]]
    assert isinstance(batch["payload"][0], TrainingMediaHandle)
    assert batch["_media"]["policy"] == "metadata"
    assert len(batch["_media"]["fields"]["payload"]) == 2

    window = batch["windows"]["state_vector"]
    assert window["delta_s"] == [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
    assert window["value"] == [
        [[0.0, 0.5], [0.0, 0.5], [1.0, 1.5]],
        [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]],
    ]
    assert window["mask"] == [[True, True, True], [True, True, True]]
    assert batch["_schema"]["windows"]["state_vector"]["mask_column"] == "mask"

    assert batch["_lineage"]["dataset_id"] == dataset.manifest.dataset_id
    assert batch["_lineage"]["snapshot_name"] == "demo-v1"
    assert batch["_lineage"]["row_plan_id"] == dataset.row_plan.plan_id
    assert batch["_lineage"]["epoch_plan_id"] == dataset.epoch_plan.plan_id
    assert batch["_lineage"]["frame_ids"] == ["obs-camera-0", "obs-camera-1"]
    assert batch["_lineage"]["row_ids"] == list(dataset.row_plan.row_ids[:2])
    assert batch["_lineage"]["source"] == [
        {
            "run_id": "run-native-training",
            "raw_uri": "memory://run-native-training",
            "raw_channel": "/camera/front",
            "raw_sequence": 0,
        },
        {
            "run_id": "run-native-training",
            "raw_uri": "memory://run-native-training",
            "raw_channel": "/camera/front",
            "raw_sequence": 1,
        },
    ]


def test_training_batch_iterator_is_torch_free_and_can_drop_last(lake, monkeypatch):
    original_find_spec = training_mod.importlib.util.find_spec

    def missing_torch(module):
        if module == "torch":
            return None
        return original_find_spec(module)

    monkeypatch.setattr(training_mod.importlib.util, "find_spec", missing_torch)
    dataset = lake.training.dataset("demo-v1", columns=["observation_id"])

    batches = list(iter_training_batches(dataset, batch_size=2, drop_last=True))

    assert [batch["observation_id"] for batch in batches] == [["obs-camera-0", "obs-camera-1"]]
    with pytest.raises(TrainingError, match=r"lancedb-robotics\[torch\]"):
        to_torch_map_dataset(dataset)


@pytest.mark.torch_loader
def test_torch_map_dataset_matches_native_samples_and_collates(lake):
    require_torch_loader()

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "timestamp_ns", "state_vector", "action_vector"],
    )
    torch_dataset = to_torch_map_dataset(dataset)

    assert len(torch_dataset) == len(dataset)
    assert torch_dataset[1]["observation_id"] == dataset[1]["observation_id"]
    assert torch_dataset[1]["_lineage"]["frame_id"] == "obs-camera-1"

    batch = collate_torch_training_samples([torch_dataset[0], torch_dataset[1]])

    import torch

    assert batch["observation_id"] == ["obs-camera-0", "obs-camera-1"]
    assert torch.equal(batch["timestamp_ns"], torch.tensor([0, 1_000_000_000]))
    assert tuple(batch["state_vector"].shape) == (2, 2)
    assert tuple(batch["action_vector"].shape) == (2, 2)
    assert torch.equal(batch["state_vector_mask"], torch.ones((2, 2), dtype=torch.bool))
    assert batch["_lineage"]["frame_ids"] == ["obs-camera-0", "obs-camera-1"]


@pytest.mark.torch_loader
def test_torch_iterable_dataloader_multi_worker_covers_epoch_once(lake):
    require_torch_loader()

    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
    )
    require_start_method("fork")
    loader = to_torch_dataloader(
        dataset,
        batch_size=1,
        num_workers=2,
        adapter="iterable",
        multiprocessing_context="fork",
    )

    seen = []
    worker_ids = set()
    for batch in loader:
        seen.extend(batch["observation_id"])
        worker_ids.add(batch["_lineage"]["worker"]["id"])

    expected = [row["observation_id"] for row in dataset]
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == len(expected)
    assert worker_ids == {0, 1}
