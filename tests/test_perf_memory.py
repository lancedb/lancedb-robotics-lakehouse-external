"""Opt-in memory smoke test: streaming ingest stays bounded on a large file (backlog 0017).

This is NOT part of the default suite -- it needs a multi-GB corpus file that the
repo does not ship. Point it at one and run explicitly::

    LANCEDB_ROBOTICS_PERF_MCAP=data/didi/.../2.mcap \\
        uv run --no-sync pytest tests/test_perf_memory.py -s

It ingests with a small ``batch_size`` and asserts that peak Python heap usage
stays far below the file size -- i.e. observations stream to the lake instead of
materializing the whole table in memory. ``LANCEDB_ROBOTICS_PERF_PEAK_MB`` (default
1024) sets the ceiling.
"""

import os
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import RUNS_SCHEMA, SCENARIOS_SCHEMA

_CORPUS_ENV = "LANCEDB_ROBOTICS_PERF_MCAP"
_PEAK_ENV = "LANCEDB_ROBOTICS_PERF_PEAK_MB"
_DISTRIBUTION_ROWS_ENV = "LANCEDB_ROBOTICS_PERF_DISTRIBUTION_ROWS"
_DISTRIBUTION_PEAK_ENV = "LANCEDB_ROBOTICS_PERF_DISTRIBUTION_PEAK_MB"


@pytest.fixture
def perf_mcap() -> Path:
    raw = os.environ.get(_CORPUS_ENV)
    if not raw:
        pytest.skip(f"set {_CORPUS_ENV}=<path to a multi-GB .mcap> to run the memory smoke test")
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"{_CORPUS_ENV} does not point to a file: {path}")
    return path


def test_large_file_ingests_with_bounded_memory(perf_mcap, tmp_path):
    file_mb = perf_mcap.stat().st_size / (1024 * 1024)
    peak_ceiling_mb = float(os.environ.get(_PEAK_ENV, "1024"))
    lake = Lake.init(tmp_path / "perf.lance")

    tracemalloc.start()
    # Small batch + CRC validation off: the streaming hot path for a trusted,
    # image/lidar-heavy multi-GB log.
    report = ingest_mcap(lake, perf_mcap, batch_size=100, validate_crcs=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    print(
        f"\nfile={file_mb:.0f} MB messages={report.message_count} "
        f"peak_heap={peak_mb:.0f} MB ceiling={peak_ceiling_mb:.0f} MB"
    )
    assert report.message_count > 0
    assert lake.table("observations").count_rows() == report.message_count
    # Bounded: peak heap is a small fraction of the file, never proportional to it.
    assert peak_mb < peak_ceiling_mb


def test_large_distribution_report_streams_with_bounded_memory(tmp_path):
    raw_count = os.environ.get(_DISTRIBUTION_ROWS_ENV)
    if not raw_count:
        pytest.skip(
            f"set {_DISTRIBUTION_ROWS_ENV}=1000000 to run the distribution memory smoke test"
        )
    row_count = int(raw_count)
    if row_count <= 0:
        pytest.skip(f"{_DISTRIBUTION_ROWS_ENV} must be positive")
    peak_ceiling_mb = float(os.environ.get(_DISTRIBUTION_PEAK_ENV, "512"))
    lake = Lake.init(tmp_path / "distribution-perf.lance")
    now = datetime(2026, 6, 21, tzinfo=UTC)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-perf-a",
                    "run_kind": "teleop",
                    "source": "synthetic",
                    "source_id": "perf-a",
                    "raw_uri": "memory://perf-a",
                    "robot_id": "arm-a",
                    "site_id": "site-a",
                    "task_id": "pick",
                    "start_time_ns": 0,
                    "end_time_ns": row_count,
                    "duration_ns": row_count,
                    "software_version": "",
                    "hardware_version": "",
                    "calibration_version": "",
                    "model_version": "",
                    "metadata": [],
                    "quality_flags": [],
                    "transform_id": "tfm-perf",
                    "created_at": now,
                },
                {
                    "run_id": "run-perf-b",
                    "run_kind": "teleop",
                    "source": "synthetic",
                    "source_id": "perf-b",
                    "raw_uri": "memory://perf-b",
                    "robot_id": "arm-b",
                    "site_id": "site-b",
                    "task_id": "place",
                    "start_time_ns": 0,
                    "end_time_ns": row_count,
                    "duration_ns": row_count,
                    "software_version": "",
                    "hardware_version": "",
                    "calibration_version": "",
                    "model_version": "",
                    "metadata": [],
                    "quality_flags": [],
                    "transform_id": "tfm-perf",
                    "created_at": now,
                },
            ],
            schema=RUNS_SCHEMA,
        )
    )
    scenarios = lake.table("scenarios")
    batch_size = 10_000
    for start in range(0, row_count, batch_size):
        rows = []
        for index in range(start, min(start + batch_size, row_count)):
            run_id = "run-perf-a" if index % 2 == 0 else "run-perf-b"
            rows.append(
                {
                    "scenario_id": f"scn-perf-{index:08d}",
                    "run_id": run_id,
                    "start_time_ns": index,
                    "end_time_ns": index + 1,
                    "window_ns": 1,
                    "is_partial": False,
                    "topics": ["/camera/front"],
                    "observation_ids": [f"obs-perf-{index:08d}"],
                    "observation_count": 1,
                    "scenario_type": "episode",
                    "trigger_event_id": "",
                    "source": "synthetic",
                    "parent_scenario_id": "",
                    "coverage_tags": [f"object_category:object-{index % 20}"],
                    "summary": "",
                    "transform_id": "tfm-perf",
                    "created_at": now,
                }
            )
        scenarios.add(pa.Table.from_pylist(rows, schema=SCENARIOS_SCHEMA))

    spec = lake.distributions.define(
        name="perf-distribution",
        dimensions=["site_id", "task_id", "object_category"],
    )
    tracemalloc.start()
    report = lake.distributions.measure(
        spec,
        batch_size=2048,
        max_slice_count=50,
        overflow="bucket",
        max_scenario_ids_per_slice=0,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    print(
        f"\nrows={row_count} slices={len(report.slices)} "
        f"scenario_batches={report.execution['scan']['scenario_batches']} "
        f"peak_heap={peak_mb:.0f} MB ceiling={peak_ceiling_mb:.0f} MB"
    )
    assert report.total_count == row_count
    assert report.execution["scan"]["scenario_batches"] > 1
    assert all(len(item.scenario_ids) == 0 for item in report.slices)
    assert peak_mb < peak_ceiling_mb
