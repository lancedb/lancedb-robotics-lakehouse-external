"""Sub-frame temporal alignment engine tests (backlog 0031)."""

import json
from datetime import UTC, datetime

import pyarrow as pa
from typer.testing import CliRunner

from lancedb_robotics.align import SUB_FRAME_BOUND_NS
from lancedb_robotics.cli import app
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA

runner = CliRunner()


def _alignment_lake(path):
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-align",
                    "run_kind": "teleop",
                    "source": "synthetic",
                    "source_id": "src-align",
                    "raw_uri": "memory://align",
                    "robot_id": "robot-arm-1",
                    "site_id": "lab-a",
                    "task_id": "insert peg",
                    "start_time_ns": 0,
                    "end_time_ns": 100_000_000,
                    "duration_ns": 100_000_000,
                    "software_version": "sw-1",
                    "hardware_version": "hw-1",
                    "calibration_version": "cal-1",
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
    rows = []

    def add_obs(
        index: int,
        topic: str,
        timestamp_ns: int,
        *,
        raw_log_time_ns: int | None = None,
        state_vector: list[float] | None = None,
        action_vector: list[float] | None = None,
    ) -> None:
        rows.append(
            {
                "observation_id": f"obs-{index}",
                "run_id": "run-align",
                "episode_id": None,
                "episode_index": None,
                "frame_index": None,
                "timestamp_ns": timestamp_ns,
                "sensor_id": topic.strip("/").replace("/", "_"),
                "topic": topic,
                "modality": "state",
                "robot_id": None,
                "site_id": None,
                "task_id": None,
                "software_version": None,
                "outcome": None,
                "raw_uri": "memory://align",
                "raw_channel": topic,
                "raw_log_time_ns": raw_log_time_ns if raw_log_time_ns is not None else timestamp_ns,
                "raw_sequence": index,
                "payload_json": None,
                "payload_blob": None,
                "message_encoding": "json",
                "schema_encoding": "json",
                "decode_status": "decoded",
                "decode_error": "",
                "state_vector": state_vector,
                "action_vector": action_vector,
                "caption": "",
                "quality_flags": [],
                "transform_id": "tfm-ingest",
                "created_at": now,
            }
        )

    add_obs(0, "/joint_states", 0, state_vector=[0.0])
    add_obs(1, "/joint_states", 100_000_000, state_vector=[100.0])
    add_obs(2, "/action", 0, raw_log_time_ns=5_000_000, action_vector=[0.0])
    add_obs(3, "/action", 50_000_000, raw_log_time_ns=55_000_000, action_vector=[50.0])
    add_obs(4, "/action", 100_000_000, raw_log_time_ns=105_000_000, action_vector=[100.0])

    for offset, timestamp_ns in enumerate([0, 1_000_000, 2_000_000, 4_000_000, 5_000_000], start=5):
        # The missing 3 ms sample should be detected as a dropped frame. The
        # receive-minus-hardware latency also drifts enough to trip drift flags.
        drift_ns = (offset - 5) * 600_000
        add_obs(
            offset,
            "/force_torque",
            timestamp_ns,
            raw_log_time_ns=timestamp_ns + drift_ns,
            state_vector=[timestamp_ns / 1_000_000],
        )

    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    return lake


def _dense_alignment_lake(path, *, samples: int = 21, step_ns: int = 10_000_000):
    lake = Lake.init(path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    end_time_ns = (samples - 1) * step_ns
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-long",
                    "run_kind": "teleop",
                    "source": "synthetic",
                    "source_id": "src-long",
                    "raw_uri": "memory://long",
                    "robot_id": "robot-arm-1",
                    "site_id": "lab-a",
                    "task_id": "insert peg",
                    "start_time_ns": 0,
                    "end_time_ns": end_time_ns,
                    "duration_ns": end_time_ns,
                    "software_version": "sw-1",
                    "hardware_version": "hw-1",
                    "calibration_version": "cal-1",
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
    rows = []
    for index in range(samples):
        timestamp_ns = index * step_ns
        rows.append(
            {
                "observation_id": f"joint-{index}",
                "run_id": "run-long",
                "episode_id": None,
                "episode_index": None,
                "frame_index": None,
                "timestamp_ns": timestamp_ns,
                "sensor_id": "joint_states",
                "topic": "/joint_states",
                "modality": "state",
                "robot_id": None,
                "site_id": None,
                "task_id": None,
                "software_version": None,
                "outcome": None,
                "raw_uri": "memory://long",
                "raw_channel": "/joint_states",
                "raw_log_time_ns": timestamp_ns,
                "raw_sequence": index * 2,
                "payload_json": None,
                "payload_blob": None,
                "message_encoding": "json",
                "schema_encoding": "json",
                "decode_status": "decoded",
                "decode_error": "",
                "state_vector": [float(index)],
                "action_vector": None,
                "caption": "",
                "quality_flags": [],
                "transform_id": "tfm-ingest",
                "created_at": now,
            }
        )
        rows.append(
            {
                "observation_id": f"action-{index}",
                "run_id": "run-long",
                "episode_id": None,
                "episode_index": None,
                "frame_index": None,
                "timestamp_ns": timestamp_ns,
                "sensor_id": "action",
                "topic": "/action",
                "modality": "action",
                "robot_id": None,
                "site_id": None,
                "task_id": None,
                "software_version": None,
                "outcome": None,
                "raw_uri": "memory://long",
                "raw_channel": "/action",
                "raw_log_time_ns": timestamp_ns,
                "raw_sequence": index * 2 + 1,
                "payload_json": None,
                "payload_blob": None,
                "message_encoding": "json",
                "schema_encoding": "json",
                "decode_status": "decoded",
                "decode_error": "",
                "state_vector": None,
                "action_vector": [float(index)],
                "caption": "",
                "quality_flags": [],
                "transform_id": "tfm-ingest",
                "created_at": now,
            }
        )
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    return lake


def _materialized_alignment_rows(lake, alignment_id):
    return sorted(
        [
            row
            for row in lake.table("aligned_frames").to_arrow().to_pylist()
            if row["alignment_id"] == alignment_id
        ],
        key=lambda row: (row["tick_index"], row["stream"]),
    )


def test_create_view_resamples_to_requested_cadence_and_interpolates(tmp_path):
    lake = _alignment_lake(tmp_path / "robot.lance")

    view = lake.align.create_view(
        "policy_train_20hz",
        run_id="run-align",
        rate_hz=20.0,
        streams=["/joint_states", "/action"],
        tolerance_ms=60.0,
        interpolation={"/joint_states": "linear", "/action": "previous"},
    )

    assert [row["timestamp_ns"] for row in view.rows] == [0, 50_000_000, 100_000_000]
    assert view.rows[1]["streams"]["/joint_states"]["value"] == [50.0]
    assert view.rows[1]["streams"]["/joint_states"]["interpolation"] == "linear"
    assert view.rows[1]["streams"]["/action"]["observation_id"] == "obs-3"
    assert view.rows[1]["streams"]["/action"]["value"] == [50.0]

    materialized = sorted(
        lake.table("aligned_frames").to_arrow().to_pylist(),
        key=lambda row: (row["tick_index"], row["stream"]),
    )
    assert len(materialized) == 6
    assert materialized[0]["alignment_id"] == view.alignment_id
    assert materialized[0]["source_row_ids"]
    joint_mid = next(
        row
        for row in materialized
        if row["tick_index"] == 1 and row["stream"] == "/joint_states"
    )
    assert json.loads(joint_mid["value_json"]) == [50.0]


def test_known_receive_time_offset_aligns_within_sub_frame_bound(tmp_path):
    lake = _alignment_lake(tmp_path / "robot.lance")

    view = lake.align.create_view(
        "known_offset",
        run_id="run-align",
        clock="receive_time_ns",
        rate_hz=20.0,
        streams=["/joint_states", "/action"],
        tolerance_ms=60.0,
        interpolation={"/joint_states": "linear", "/action": "nearest"},
        latency_ns={"/action": 5_000_000},
    )

    assert view.quality_summary["sub_frame_bound_ns"] == SUB_FRAME_BOUND_NS
    assert view.quality_summary["max_abs_error_ns"] <= view.quality_summary["sub_frame_bound_ns"]
    assert view.quality_summary["execution"] == "streaming-row-id-plan"
    assert view.quality_summary["source_rows_read"] == 5
    assert view.quality_summary["segment_count"] == 1
    assert view.quality_summary["output_rows_written"] == 3
    assert view.quality_summary["compatibility_frame_rows_written"] == 6
    for row in view.rows:
        assert row["streams"]["/action"]["absolute_error_ns"] == 0
        assert row["streams"]["/action"]["latency_ns"] == 5_000_000
    assert not view.quality_flags


def test_clock_drift_and_dropped_frames_surface_quality_flags(tmp_path):
    lake = _alignment_lake(tmp_path / "robot.lance")

    view = lake.align.create_view(
        "force_quality",
        run_id="run-align",
        rate_hz=1000.0,
        streams=["/force_torque"],
        tolerance_ms=0.25,
    )

    stats = view.quality_summary["streams"]["/force_torque"]
    assert stats["dropped_frame_count"] == 1
    assert stats["clock_drift_detected"] is True
    assert "alignment:force-torque:dropped-frames" in view.quality_flags
    assert "alignment:force-torque:clock-drift" in view.quality_flags
    assert "alignment:force-torque:tolerance-exceeded" in view.quality_flags


def test_alignment_recipe_and_quality_lineage_are_deterministic(tmp_path):
    lake = _alignment_lake(tmp_path / "robot.lance")

    first = lake.align.create_view(
        "deterministic",
        run_id="run-align",
        rate_hz=20.0,
        streams=["/joint_states", "/action"],
        tolerance_ms=60.0,
        interpolation={"/joint_states": "linear", "/action": "nearest"},
    )
    second = lake.align.create_view(
        "deterministic",
        run_id="run-align",
        rate_hz=20.0,
        streams=["/joint_states", "/action"],
        tolerance_ms=60.0,
        interpolation={"/joint_states": "linear", "/action": "nearest"},
    )

    assert second.alignment_id == first.alignment_id
    assert second.transform_id == first.transform_id

    jobs = [
        row
        for row in lake.table("alignment_jobs").to_arrow().to_pylist()
        if row["alignment_id"] == first.alignment_id
    ]
    assert len(jobs) == 1
    assert jobs[0]["streams"] == ["/joint_states", "/action"]
    assert jobs[0]["output_table"] == "aligned_ticks"
    recipe = json.loads(jobs[0]["recipe"])
    assert recipe["interpolation"]["/joint_states"] == "linear"
    assert recipe["execution"] == "streaming-row-id-plan"
    assert recipe["materialize"] is True
    summary = json.loads(jobs[0]["quality_summary"])
    assert summary["alignment_id"] == first.alignment_id
    assert summary["execution"] == "streaming-row-id-plan"
    assert summary["segment_count"] == 1
    assert summary["output_rows_written"] == 3
    assert summary["compatibility_frame_rows_written"] == 6

    transforms = [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == first.transform_id
    ]
    assert len(transforms) == 1
    assert transforms[0]["kind"] == "alignment-view"
    assert transforms[0]["output_tables"] == [
        "alignment_jobs",
        "aligned_ticks",
        "aligned_frames",
    ]

    materialized = [
        row
        for row in lake.table("aligned_frames").to_arrow().to_pylist()
        if row["alignment_id"] == first.alignment_id
    ]
    assert len(materialized) == 6
    ticks = [
        row
        for row in lake.table("aligned_ticks").to_arrow().to_pylist()
        if row["alignment_id"] == first.alignment_id
    ]
    assert len(ticks) == 3


def test_alignment_uses_query_node_safe_row_id_plan(tmp_path, monkeypatch):
    lake = _alignment_lake(tmp_path / "robot.lance")
    real_table = lake.table
    calls = []

    class _QuerySpy:
        def __init__(self, query):
            self._query = query

        def select(self, *args, **kwargs):
            calls.append("metadata.select")
            return _QuerySpy(self._query.select(*args, **kwargs))

        def where(self, *args, **kwargs):
            calls.append("metadata.where")
            return _QuerySpy(self._query.where(*args, **kwargs))

        def with_row_id(self, *args, **kwargs):
            calls.append("metadata.with_row_id")
            return _QuerySpy(self._query.with_row_id(*args, **kwargs))

        def order_by(self, *args, **kwargs):
            calls.append("metadata.order_by")
            return _QuerySpy(self._query.order_by(*args, **kwargs))

        def to_batches(self, *args, **kwargs):
            calls.append("metadata.to_batches")
            return self._query.to_batches(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._query, name)

    class _TakeSpy:
        def __init__(self, query):
            self._query = query

        def select(self, *args, **kwargs):
            calls.append("hydrate.select")
            return _TakeSpy(self._query.select(*args, **kwargs))

        def to_batches(self, *args, **kwargs):
            calls.append("hydrate.to_batches")
            return self._query.to_batches(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._query, name)

    class _NoObservationToArrow:
        def __init__(self, table):
            self._table = table

        def search(self, *args, **kwargs):
            calls.append("metadata.search")
            return _QuerySpy(self._table.search(*args, **kwargs))

        def take_row_ids(self, *args, **kwargs):
            calls.append("hydrate.take_row_ids")
            return _TakeSpy(self._table.take_row_ids(*args, **kwargs))

        def to_arrow(self):  # pragma: no cover - this should not be called
            raise AssertionError("alignment should not eagerly load observations")

        def to_lance(self):  # pragma: no cover - this should not be called
            raise AssertionError("alignment should use LanceDB query APIs, not to_lance")

        def __getattr__(self, name):
            return getattr(self._table, name)

    def table(name: str):
        opened = real_table(name)
        if name == "observations":
            return _NoObservationToArrow(opened)
        return opened

    monkeypatch.setattr(lake, "table", table)

    view = lake.align.create_view(
        "row_id_plan",
        run_id="run-align",
        rate_hz=20.0,
        streams=["/joint_states", "/action"],
        tolerance_ms=60.0,
        interpolation={"/joint_states": "linear", "/action": "nearest"},
    )

    assert view.output_table == "aligned_ticks"
    assert view.rows[1]["streams"]["/joint_states"]["value"] == [50.0]
    assert "metadata.search" in calls
    assert "metadata.with_row_id" in calls
    assert "metadata.to_batches" in calls
    assert "hydrate.take_row_ids" in calls
    assert "hydrate.to_batches" in calls


def test_streaming_materialization_is_segment_bounded_and_records_metrics(tmp_path):
    lake = _dense_alignment_lake(tmp_path / "robot.lance")

    view = lake.align.create_view(
        "long_streaming",
        run_id="run-long",
        rate_hz=100.0,
        streams=["/joint_states", "/action"],
        tolerance_ms=1.0,
        interpolation={"/joint_states": "linear", "/action": "nearest"},
        batch_size=4,
    )

    rows = _materialized_alignment_rows(lake, view.alignment_id)
    assert len(rows) == 42
    assert view.quality_summary["rows"] == 21
    assert len(view.rows) == 4
    assert view.quality_summary["preview_rows"] == 4
    assert view.quality_summary["segment_count"] == 6
    assert view.quality_summary["output_rows_written"] == 21
    assert view.quality_summary["compatibility_frame_rows_written"] == 42
    assert view.quality_summary["source_rows_hydrated"] == 42
    assert view.quality_summary["max_in_memory_segment_size"] == 4
    assert view.quality_summary["metadata_rows_scanned"] >= 42


def test_streaming_materialization_matches_virtual_alignment_rows(tmp_path):
    lake = _alignment_lake(tmp_path / "robot.lance")

    reference = lake.align.window(
        name="reference",
        run_id="run-align",
        rate_hz=20.0,
        streams=["/joint_states", "/action"],
        tolerance_ms=60.0,
        interpolation={"/joint_states": "linear", "/action": "nearest"},
        batch_size=1,
    )
    view = lake.align.create_view(
        "streaming_equivalence",
        run_id="run-align",
        rate_hz=20.0,
        streams=["/joint_states", "/action"],
        tolerance_ms=60.0,
        interpolation={"/joint_states": "linear", "/action": "nearest"},
        batch_size=1,
    )

    expected = {}
    for row in reference.rows:
        for stream, result in row["streams"].items():
            expected[(row["index"], stream)] = {
                "status": result["status"],
                "observation_id": result.get("observation_id"),
                "source_observation_ids": list(result.get("source_observation_ids") or []),
                "source_row_ids": [int(row_id) for row_id in result.get("source_row_ids") or []],
                "value_json": json.dumps(result.get("value"), sort_keys=True)
                if result.get("value") is not None
                else None,
                "quality_flags": list(result.get("quality_flags") or []),
            }
    actual = {
        (row["tick_index"], row["stream"]): {
            "status": row["status"],
            "observation_id": row["observation_id"],
            "source_observation_ids": list(row["source_observation_ids"] or []),
            "source_row_ids": [int(row_id) for row_id in row["source_row_ids"] or []],
            "value_json": row["value_json"],
            "quality_flags": list(row["quality_flags"] or []),
        }
        for row in _materialized_alignment_rows(lake, view.alignment_id)
    }
    assert actual == expected


def test_streaming_segment_boundary_linear_interpolation_uses_neighbor_segments(tmp_path):
    lake = _alignment_lake(tmp_path / "robot.lance")

    view = lake.align.create_view(
        "boundary_linear",
        run_id="run-align",
        rate_hz=20.0,
        streams=["/joint_states"],
        tolerance_ms=60.0,
        interpolation={"/joint_states": "linear"},
        batch_size=1,
    )

    rows = _materialized_alignment_rows(lake, view.alignment_id)
    middle = next(row for row in rows if row["tick_index"] == 1 and row["stream"] == "/joint_states")
    assert json.loads(middle["value_json"]) == [50.0]
    assert middle["source_observation_ids"] == ["obs-0", "obs-1"]
    assert view.quality_summary["segment_count"] == 3


def test_align_create_cli_reports_view_and_quality(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _alignment_lake(lake_path)

    result = runner.invoke(
        app,
        [
            "align",
            "create",
            "cli_alignment",
            "--lake",
            str(lake_path),
            "--run-id",
            "run-align",
            "--rate-hz",
            "20",
            "--stream",
            "/joint_states",
            "--stream",
            "/action",
            "--tolerance-ms",
            "60",
            "--interpolation",
            "/joint_states=linear",
            "--latency",
            "/action=5ms",
            "--clock",
            "receive_time_ns",
        ],
    )

    assert result.exit_code == 0
    assert "view: cli_alignment" in result.output
    assert "rows: 3" in result.output
    assert "quality flags: none" in result.output
