"""Curation / mining workbench tests (backlog 0032)."""

import hashlib
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from lancedb_robotics import curate as curate_mod
from lancedb_robotics.cli import app
from lancedb_robotics.curate import CurationError, CurationScope
from lancedb_robotics.indexing import ScalarIndexResult
from lancedb_robotics.lake import Lake
from lancedb_robotics.review_connectors import ReviewConnectorResult, ReviewConnectorTask
from lancedb_robotics.schemas import (
    ALIGNED_FRAMES_SCHEMA,
    CURATION_COMPARISONS_SCHEMA,
    CURATION_REVIEW_QUEUES_SCHEMA,
    DATASET_SNAPSHOTS_SCHEMA,
    EVENTS_SCHEMA,
    LABELS_SCHEMA,
    MODEL_OUTPUTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RUNS_SCHEMA,
)
from lancedb_robotics.training import load_snapshot_preview

runner = CliRunner()

NOW = datetime(2026, 6, 16, tzinfo=UTC)


def _unit(vector):
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def _mean_similarity(lake: Lake, scenario_ids: tuple[str, ...]) -> float:
    rows = {
        row["scenario_id"]: row
        for row in lake.table("scenarios").to_arrow().to_pylist()
        if row["scenario_id"] in set(scenario_ids)
    }
    if len(scenario_ids) < 2:
        return 0.0
    total = 0.0
    count = 0
    for index, left_id in enumerate(scenario_ids):
        for right_id in scenario_ids[index + 1:]:
            total += _cosine(rows[left_id]["embedding"], rows[right_id]["embedding"])
            count += 1
    return total / count


def _scenario(
    scenario_id: str,
    *,
    run_id: str,
    start_time_ns: int,
    object_category: str,
    embedding: list[float],
) -> dict:
    observation_id = f"obs-{scenario_id}"
    return {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "start_time_ns": start_time_ns,
        "end_time_ns": start_time_ns + 10,
        "window_ns": 10,
        "is_partial": False,
        "topics": ["/camera/front"],
        "observation_ids": [observation_id],
        "observation_count": 1,
        "scenario_type": "episode",
        "source": "fixture",
        "coverage_tags": [f"object_category:{object_category}", "quality_score:0.95"],
        "summary": scenario_id,
        "transform_id": "tfm-fixture",
        "created_at": NOW,
        "embedding": _unit(embedding),
    }


def _observation(scenario: dict, *, payload_blob: bytes | None = None) -> dict:
    observation_id = scenario["observation_ids"][0]
    return {
        "observation_id": observation_id,
        "run_id": scenario["run_id"],
        "episode_id": "",
        "episode_index": 0,
        "frame_index": 0,
        "timestamp_ns": scenario["start_time_ns"],
        "sensor_id": "camera_front",
        "topic": "/camera/front",
        "modality": "image",
        "robot_id": "",
        "site_id": "",
        "task_id": "",
        "software_version": "",
        "outcome": "",
        "raw_uri": f"memory://{scenario['run_id']}",
        "raw_channel": "/camera/front",
        "raw_log_time_ns": scenario["start_time_ns"],
        "raw_sequence": 0,
        "payload_json": None,
        "payload_blob": payload_blob,
        "message_encoding": "jpeg",
        "schema_encoding": "jpeg",
        "decode_status": "decoded",
        "decode_error": "",
        "state_vector": [],
        "action_vector": [],
        "caption": scenario["summary"],
        "quality_flags": [],
        "transform_id": "tfm-fixture",
        "created_at": NOW,
    }


def _aligned_frame(
    aligned_frame_id: str,
    *,
    run_id: str,
    timestamp_ns: int,
    observation_id: str,
    tick_index: int,
) -> dict:
    return {
        "aligned_frame_id": aligned_frame_id,
        "alignment_id": "align-fixture",
        "run_id": run_id,
        "tick_index": tick_index,
        "timestamp_ns": timestamp_ns,
        "stream": "/camera/front",
        "status": "aligned",
        "interpolation": "nearest",
        "observation_id": observation_id,
        "source_observation_ids": [observation_id],
        "source_row_ids": [tick_index],
        "source_timestamp_ns": timestamp_ns,
        "source_time_ns": timestamp_ns,
        "receive_time_ns": timestamp_ns,
        "latency_ns": 0,
        "error_ns": 0,
        "absolute_error_ns": 0,
        "confidence": 1.0,
        "value_json": "{}",
        "quality_flags": [],
        "transform_id": "tfm-align-fixture",
        "created_at": NOW,
    }


def _build_curation_lake(path) -> Lake:
    lake = Lake.init(path)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-a",
                    "run_kind": "teleop",
                    "raw_uri": "memory://run-a",
                    "robot_id": "arm-a",
                    "site_id": "site-a",
                    "task_id": "pick",
                    "start_time_ns": 0,
                    "end_time_ns": 100,
                    "duration_ns": 100,
                    "quality_flags": [],
                    "created_at": NOW,
                },
                {
                    "run_id": "run-b",
                    "run_kind": "teleop",
                    "raw_uri": "memory://run-b",
                    "robot_id": "arm-b",
                    "site_id": "site-b",
                    "task_id": "pick",
                    "start_time_ns": 0,
                    "end_time_ns": 100,
                    "duration_ns": 100,
                    "quality_flags": [],
                    "created_at": NOW,
                },
            ],
            schema=RUNS_SCHEMA,
        )
    )

    scenarios = lake.table("scenarios")
    scenarios.add_columns(pa.schema([pa.field("embedding", pa.list_(pa.float32(), 4))]))
    scenario_rows = [
        _scenario(
            "scn-anchor",
            run_id="run-a",
            start_time_ns=0,
            object_category="cup",
            embedding=[1.0, 0.0, 0.0, 0.0],
        ),
        _scenario(
            "scn-duplicate",
            run_id="run-a",
            start_time_ns=10,
            object_category="cup",
            embedding=[0.99995, 0.01, 0.0, 0.0],
        ),
        _scenario(
            "scn-neighbor",
            run_id="run-a",
            start_time_ns=20,
            object_category="box",
            embedding=[0.97, 0.24, 0.0, 0.0],
        ),
        _scenario(
            "scn-site-b-cup",
            run_id="run-b",
            start_time_ns=0,
            object_category="cup",
            embedding=[0.0, 1.0, 0.0, 0.0],
        ),
        _scenario(
            "scn-site-b-box",
            run_id="run-b",
            start_time_ns=10,
            object_category="box",
            embedding=[0.0, 0.0, 1.0, 0.0],
        ),
        _scenario(
            "scn-site-b-box-extra",
            run_id="run-b",
            start_time_ns=20,
            object_category="box",
            embedding=[0.0, 0.1, 0.9, 0.0],
        ),
    ]
    scenarios.add(
        pa.Table.from_pylist(
            scenario_rows,
            schema=scenarios.schema,
        )
    )
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                _observation(row, payload_blob=f"payload-{row['scenario_id']}".encode())
                for row in scenario_rows
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    lake.table("events").add(
        pa.Table.from_pylist(
            [
                {
                    "event_id": "evt-failure",
                    "run_id": "run-a",
                    "timestamp_ns": 5,
                    "start_time_ns": 5,
                    "end_time_ns": 6,
                    "event_type": "failure",
                    "severity": "high",
                    "source": "fixture",
                    "notes": "seeded failure",
                    "linked_incident_id": "incident-1",
                    "transform_id": "tfm-fixture",
                    "created_at": NOW,
                }
            ],
            schema=EVENTS_SCHEMA,
        )
    )
    return lake


def _add_synthetic_review_queue(lake: Lake, *, count: int = 30) -> str:
    scenario_ids = [
        row["scenario_id"]
        for row in sorted(
            lake.table("scenarios").to_arrow().to_pylist(),
            key=lambda row: row["scenario_id"],
        )
    ]
    queue_id = "queue-large-review"
    table_versions = [
        {"table": "scenarios", "version": int(lake.table("scenarios").version), "tag": ""},
        {
            "table": "curation_review_queues",
            "version": int(lake.table("curation_review_queues").version),
            "tag": "",
        },
    ]
    rows = []
    for index in range(count):
        scenario_id = scenario_ids[index % len(scenario_ids)]
        target_id = f"target-{index:03d}"
        rows.append(
            {
                "queue_item_id": f"qitem-large-{index:03d}",
                "queue_id": queue_id,
                "queue_name": "large-review",
                "target_grain": "scenario",
                "target_id": target_id,
                "scenario_id": scenario_id,
                "source_operation": "active-learning" if index % 3 else "failure-mining",
                "source_ref": json.dumps({"synthetic_index": index}, sort_keys=True),
                "priority": index // 2,
                "priority_score": float(count - index),
                "priority_reason": "synthetic-scale-test",
                "assignee": "qa-a" if index % 2 == 0 else "qa-b",
                "status": "open" if index % 2 == 0 else "assigned",
                "export_uri": "",
                "external_task_id": "",
                "external_url": "",
                "metadata": [],
                "table_versions": table_versions,
                "source_transform_ids": ["tfm-synthetic-review-source"],
                "created_by": "test",
                "transform_id": "tfm-synthetic-review-queue",
                "created_at": NOW,
            }
        )
    lake.table("curation_review_queues").add(
        pa.Table.from_pylist(rows, schema=CURATION_REVIEW_QUEUES_SCHEMA)
    )
    return queue_id


def _transform(lake: Lake, transform_id: str) -> dict:
    return next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == transform_id
    )


def _snapshot_row(lake: Lake, name: str) -> dict:
    return next(
        row for row in lake.table("dataset_snapshots").to_arrow().to_pylist() if row["name"] == name
    )


def _metadata(row: dict) -> dict:
    return {item["key"]: json.loads(item["value"]) for item in row["metadata"]}


class _RecordingReviewConnector:
    tool = "label-studio"

    def __init__(
        self,
        *,
        fail_queue_item_ids: set[str] | None = None,
        statuses: dict[str, str] | None = None,
        outcomes: dict[str, dict] | None = None,
    ) -> None:
        self.fail_queue_item_ids = fail_queue_item_ids or set()
        self.statuses = statuses or {}
        self.outcomes = outcomes or {}
        self.upsert_calls: list[tuple[ReviewConnectorTask, ...]] = []
        self.status_calls: list[tuple[ReviewConnectorTask, ...]] = []
        self.import_calls: list[tuple[ReviewConnectorTask, ...]] = []
        self.created_by_key: dict[str, tuple[str, str]] = {}

    def upsert_tasks(
        self,
        tasks: tuple[ReviewConnectorTask, ...],
        *,
        project_id: str,
    ) -> tuple[ReviewConnectorResult, ...]:
        self.upsert_calls.append(tuple(tasks))
        results = []
        for task in tasks:
            if task.queue_item_id in self.fail_queue_item_ids:
                results.append(
                    ReviewConnectorResult(
                        queue_item_id=task.queue_item_id,
                        idempotency_key=task.idempotency_key,
                        status="failed",
                        error="injected connector failure",
                    )
                )
                continue
            external = self.created_by_key.get(task.idempotency_key)
            if external is None:
                index = len(self.created_by_key) + 1
                external = (
                    f"ls-task-{index}",
                    f"https://label-studio.local/tasks/{index}",
                )
                self.created_by_key[task.idempotency_key] = external
                status = "exported"
            else:
                status = "already-present"
            results.append(
                ReviewConnectorResult(
                    queue_item_id=task.queue_item_id,
                    idempotency_key=task.idempotency_key,
                    status=status,
                    external_task_id=external[0],
                    external_url=external[1],
                )
            )
        return tuple(results)

    def sync_task_status(
        self,
        tasks: tuple[ReviewConnectorTask, ...],
        *,
        project_id: str,
    ) -> tuple[ReviewConnectorResult, ...]:
        self.status_calls.append(tuple(tasks))
        return tuple(
            ReviewConnectorResult(
                queue_item_id=task.queue_item_id,
                idempotency_key=task.idempotency_key,
                status=self.statuses.get(task.queue_item_id, "completed"),
                external_task_id=f"synced-{task.queue_item_id}",
            )
            for task in tasks
        )

    def import_outcomes(
        self,
        tasks: tuple[ReviewConnectorTask, ...],
        *,
        project_id: str,
    ) -> tuple[dict, ...]:
        self.import_calls.append(tuple(tasks))
        rows = []
        for task in tasks:
            outcome = self.outcomes.get(task.queue_item_id)
            if outcome:
                rows.append({**outcome, "queue_item_id": task.queue_item_id})
        return tuple(rows)


def _transform_params(row: dict) -> dict:
    return json.loads(row["params"] or "{}")


def _transform_rows_by_kind(lake: Lake, kind: str) -> list[dict]:
    return [
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == kind
    ]


def _add_dense_cluster(lake: Lake, *, prefix: str = "dense", count: int = 5) -> tuple[str, ...]:
    scenarios = lake.table("scenarios")
    ids = tuple(f"scn-{prefix}-{index}" for index in range(count))
    scenarios.add(
        pa.Table.from_pylist(
            [
                _scenario(
                    scenario_id,
                    run_id="run-a",
                    start_time_ns=1_000 + index,
                    object_category=prefix,
                    embedding=[1.0, index / 10_000, 0.0, 0.0],
                )
                for index, scenario_id in enumerate(ids)
            ],
            schema=scenarios.schema,
        )
    )
    return ids


def test_dedup_collapses_planted_near_duplicate_and_records_lineage(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    deduped = lake.curate.workbench().dedup(near_duplicate_threshold=0.999)

    assert "scn-anchor" in deduped.scenario_ids
    assert "scn-duplicate" not in deduped.scenario_ids
    assert "scn-neighbor" in deduped.scenario_ids
    assert deduped.report["dropped_scenario_ids"] == ["scn-duplicate"]

    transform = _transform(lake, deduped.transform_id)
    assert transform["kind"] == "curation-dedup"
    params = json.loads(transform["params"])
    assert params["operation"] == "dedup"
    assert params["near_duplicate_threshold"] == 0.999
    assert params["kept_scenario_ids"] == list(deduped.scenario_ids)


def test_semantic_dedup_plan_persists_auditable_membership_decisions(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    workbench = lake.curate.workbench()
    plan = workbench.plan_dedup(near_duplicate_threshold=0.999)
    deduped = workbench.dedup(near_duplicate_threshold=0.999, view_name="auto-dedup")

    assert plan.representative_ids == deduped.scenario_ids
    assert plan.dropped_scenario_ids == ("scn-duplicate",)
    assert plan.report["search_strategy"] == "exact-small"
    assert plan.report["planner"]["all_pairs_scanned"] is True
    group = next(group for group in plan.groups if group["dropped"])
    assert group["representative"] == "scn-anchor"
    assert group["dropped"] == ["scn-duplicate"]

    view = next(
        row for row in lake.table("curation_views").to_arrow().to_pylist() if row["name"] == "auto-dedup"
    )
    decisions = [
        row
        for row in lake.table("curation_memberships").to_arrow().to_pylist()
        if row["view_id"] == view["view_id"]
    ]
    assert {(row["scenario_id"], row["decision"]) for row in decisions} == {
        ("scn-anchor", "include"),
        ("scn-duplicate", "exclude"),
    }
    duplicate = next(row for row in decisions if row["scenario_id"] == "scn-duplicate")
    metadata = {item["key"]: json.loads(item["value"]) for item in duplicate["metadata"]}
    assert duplicate["source"] == "dedup"
    assert duplicate["reason_code"] == "semantic-duplicate"
    assert metadata["representative_id"] == "scn-anchor"
    assert metadata["dedup_transform_id"] == deduped.transform_id


def test_representative_policy_prefers_higher_quality_or_labeled_examples(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    scenarios = lake.table("scenarios")
    scenarios.add(
        pa.Table.from_pylist(
            [
                _scenario(
                    "scn-low-quality-early",
                    run_id="run-a",
                    start_time_ns=200,
                    object_category="bolt",
                    embedding=[0.8, 0.2, 0.0, 0.0],
                )
                | {"coverage_tags": ["object_category:bolt", "quality_score:0.20"]},
                _scenario(
                    "scn-high-quality-late",
                    run_id="run-a",
                    start_time_ns=210,
                    object_category="bolt",
                    embedding=[0.8001, 0.1999, 0.0, 0.0],
                )
                | {"coverage_tags": ["object_category:bolt", "quality_score:0.99"]},
            ],
            schema=scenarios.schema,
        )
    )
    lake.table("labels").add(
        pa.Table.from_pylist(
            [
                {
                    "label_id": "lbl-high-quality-late",
                    "run_id": "run-a",
                    "observation_id": "",
                    "scenario_id": "scn-high-quality-late",
                    "event_id": "",
                    "label_type": "object",
                    "label": "bolt",
                    "label_value": "{}",
                    "label_spec": "fixture",
                    "source": "human",
                    "reviewer": "qa",
                    "confidence": 1.0,
                    "status": "accepted",
                    "metadata": [],
                    "transform_id": "tfm-label",
                    "created_at": NOW,
                }
            ],
            schema=LABELS_SCHEMA,
        )
    )

    plan = lake.curate.workbench(
        scope=["scn-low-quality-early", "scn-high-quality-late"]
    ).plan_dedup(
        near_duplicate_threshold=0.999,
        representative_policy=("labels", "quality", "earliest"),
    )

    assert plan.representative_ids == ("scn-high-quality-late",)
    assert plan.dropped_scenario_ids == ("scn-low-quality-early",)
    score = plan.groups[0]["scores"][0]
    assert score["scenario_id"] == "scn-low-quality-early"
    assert score["role"] == "duplicate"


def test_large_unindexed_dedup_plan_refuses_accidental_all_pairs_scan(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    scenarios = lake.table("scenarios")
    scenarios.add(
        pa.Table.from_pylist(
            [
                _scenario(
                    f"scn-scale-{index:03d}",
                    run_id="run-a",
                    start_time_ns=1_000 + index,
                    object_category="scale",
                    embedding=[1.0, index / 1000, 0.0, 0.0],
                )
                for index in range(20)
            ],
            schema=scenarios.schema,
        )
    )

    with pytest.raises(CurationError, match="requires a persistent vector index"):
        lake.curate.workbench().plan_dedup(index_min_rows=10)


def test_distributed_dedup_resumes_shards_and_keeps_memberships_idempotent(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    workbench = lake.curate.workbench()

    with pytest.raises(CurationError, match="rerun with job_id='job-resume'"):
        workbench.plan_distributed_dedup(
            job_id="job-resume",
            shard_by=("run_id",),
            near_duplicate_threshold=0.999,
            max_shards=1,
        )

    shard_rows = _transform_rows_by_kind(lake, "curation-distributed-dedup-shard")
    assert len(shard_rows) == 1
    assert not lake.table("curation_memberships").to_arrow().to_pylist()

    deduped = workbench.distributed_dedup(
        job_id="job-resume",
        shard_by=("run_id",),
        near_duplicate_threshold=0.999,
        view_name="distributed-dedup",
    )
    first_memberships = lake.table("curation_memberships").to_arrow().to_pylist()
    first_ids = [row["membership_id"] for row in first_memberships]

    rerun = workbench.distributed_dedup(
        job_id="job-resume",
        shard_by=("run_id",),
        near_duplicate_threshold=0.999,
        view_name="distributed-dedup",
    )
    second_memberships = lake.table("curation_memberships").to_arrow().to_pylist()
    second_ids = [row["membership_id"] for row in second_memberships]
    final_shard_rows = _transform_rows_by_kind(lake, "curation-distributed-dedup-shard")
    final_params = [_transform_params(row) for row in final_shard_rows]

    assert deduped.scenario_ids == rerun.scenario_ids
    assert first_ids == second_ids
    assert len(second_ids) == len(set(second_ids))
    assert len(final_shard_rows) == 2
    assert {params["job_id"] for params in final_params} == {"job-resume"}
    assert all(params["timings"]["duration_ms"] >= 0 for params in final_params)
    assert all(params["index_query"] for params in final_params)
    assert all(
        any(version["table"] == "scenarios" for version in row["input_table_versions"])
        for row in final_shard_rows
    )


def test_distributed_dedup_adaptively_expands_dense_indexed_clusters(tmp_path, monkeypatch):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    dense_ids = _add_dense_cluster(lake)
    monkeypatch.setattr(curate_mod, "has_vector_index", lambda table, column: True)
    monkeypatch.setattr(curate_mod, "vector_index_columns", lambda table: {"embedding"})

    plan = lake.curate.workbench(scope=dense_ids).plan_distributed_dedup(
        job_id="job-dense",
        near_duplicate_threshold=0.99999,
        neighbor_limit=1,
        max_neighbor_limit=4,
        require_index=True,
        recall_audit_sample_size=20,
    )

    duplicate_group = next(group for group in plan.groups if len(group["members"]) == len(dense_ids))
    shard = plan.report["shards"][0]
    assert set(duplicate_group["members"]) == set(dense_ids)
    assert plan.report["planner"]["neighbor_expansions"] > 0
    assert shard["final_neighbor_limit"] == 4
    assert shard["recall_audit"]["estimated_recall"] == 1.0


def test_distributed_dedup_recall_audit_reports_missed_edges(tmp_path, monkeypatch):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    dense_ids = _add_dense_cluster(lake, prefix="audit", count=5)
    monkeypatch.setattr(curate_mod, "has_vector_index", lambda table, column: True)
    monkeypatch.setattr(curate_mod, "vector_index_columns", lambda table: {"embedding"})

    plan = lake.curate.workbench(scope=dense_ids).plan_distributed_dedup(
        job_id="job-audit",
        near_duplicate_threshold=0.99999,
        neighbor_limit=1,
        max_neighbor_limit=1,
        adaptive_neighbor_limit=False,
        require_index=True,
        recall_audit_sample_size=20,
    )

    audit = plan.report["recall_audit"]
    shard_params = _transform_params(
        _transform_rows_by_kind(lake, "curation-distributed-dedup-shard")[0]
    )
    assert audit["sampled_pairs"] == 10
    assert audit["estimated_recall"] < 1.0
    assert audit["missed_edge_count"] > 0
    assert shard_params["recall_audit"]["missed_edges"]


def test_diversity_sample_reduces_redundancy_and_records_balance_report(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    sampled = lake.curate.workbench().diversity_sample(
        limit=3,
        by=["site_id", "object_category"],
        min_per_slice=1,
        duplicate_threshold=0.999,
    )
    manifest = sampled.snapshot(name="diverse-v1", split_by="scenario")
    naive_first_three = ("scn-anchor", "scn-duplicate", "scn-neighbor")

    assert len(sampled.scenario_ids) == 3
    assert "scn-duplicate" not in sampled.scenario_ids
    assert sampled.report["mean_pairwise_similarity"] < _mean_similarity(lake, naive_first_three)
    assert all(count <= 1 for count in sampled.report["duplicate_group_usage"].values())

    row = _snapshot_row(lake, "diverse-v1")
    balance = json.loads(row["balance_report"])
    assert manifest.scenario_ids == tuple(sorted(sampled.scenario_ids))
    assert balance["operation"] == "diversity-sample"
    assert balance["duplicate_plan_transform_id"]


def test_diversity_optimizer_satisfies_multidimensional_constraints(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    optimized = lake.curate.workbench().optimize_diversity(
        limit=4,
        constraint_spec={
            "minimum": {
                "site_id": {"site-a": 1, "site-b": 1},
                "object_category": {"cup": 2, "box": 1},
            },
            "maximum": {
                "site_id": {"site-a": 2},
                "object_category": {"cup": 2},
            },
            "weights": {
                "site_id": {"site-b": 2.0},
                "object_category": {"box": 1.5},
            },
            "quality": {"column": "quality_score", "weight": 1.0},
            "label_completeness": {"weight": 1.0},
        },
        duplicate_threshold=0.999,
        max_per_duplicate_group=1,
    )
    manifest = optimized.snapshot(name="optimized-diverse-v1", split_by="scenario")

    rows = {row["scenario_id"]: row for row in lake.table("scenarios").to_arrow().to_pylist()}
    runs = {row["run_id"]: row for row in lake.table("runs").to_arrow().to_pylist()}
    site_counts = Counter(runs[rows[scenario_id]["run_id"]]["site_id"] for scenario_id in optimized.scenario_ids)
    object_counts = Counter(
        next(
            tag.split(":", 1)[1]
            for tag in rows[scenario_id]["coverage_tags"]
            if tag.startswith("object_category:")
        )
        for scenario_id in optimized.scenario_ids
    )

    assert len(optimized.scenario_ids) == 4
    assert site_counts["site-a"] >= 1
    assert site_counts["site-b"] >= 1
    assert site_counts["site-a"] <= 2
    assert object_counts["cup"] == 2
    assert object_counts["box"] >= 1
    assert all(item["status"] == "satisfied" for item in optimized.report["constraints"])

    row = _snapshot_row(lake, "optimized-diverse-v1")
    coverage = json.loads(row["coverage_report"])
    balance = json.loads(row["balance_report"])
    assert manifest.coverage_report == coverage
    assert coverage["operation"] == "diversity-optimization"
    assert coverage["constraint_spec"]["minimum"]["object_category=cup"] == 2
    assert balance["greedy_baseline"]["operation"] == "diversity-sample"


def test_diversity_optimizer_reports_infeasible_constraints(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    optimized = lake.curate.workbench().optimize_diversity(
        limit=3,
        constraint_spec={
            "minimum": {
                "site_id": {"site-b": 4},
                "object_category": {"cone": 1},
            },
        },
        duplicate_threshold=0.999,
        max_per_duplicate_group=1,
    )

    assert 0 < len(optimized.scenario_ids) <= 3
    violated = {
        item["label"]: item
        for item in optimized.report["constraints"]
        if item["status"] == "violated"
    }
    assert violated["site_id=site-b"]["feasible_count"] == 3
    assert violated["object_category=cone"]["feasible_count"] == 0
    assert optimized.report["constraint_summary"]["violated"] == 2


def test_diversity_optimizer_caps_duplicate_group_and_prefers_quality_labels(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    scenarios = lake.table("scenarios")
    high_quality_duplicate = _scenario(
        "scn-high-quality-duplicate",
        run_id="run-a",
        start_time_ns=30,
        object_category="cup",
        embedding=[0.99999, 0.003, 0.0, 0.0],
    ) | {"coverage_tags": ["object_category:cup", "quality_score:0.99"]}
    scenarios.add(pa.Table.from_pylist([high_quality_duplicate], schema=scenarios.schema))
    lake.table("labels").add(
        pa.Table.from_pylist(
            [
                {
                    "label_id": "lbl-high-quality-duplicate",
                    "run_id": "run-a",
                    "observation_id": "",
                    "scenario_id": "scn-high-quality-duplicate",
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
                    "transform_id": "tfm-label-high-quality-duplicate",
                    "created_at": NOW,
                }
            ],
            schema=LABELS_SCHEMA,
        )
    )

    optimized = lake.curate.workbench(
        scope=["scn-anchor", "scn-duplicate", "scn-high-quality-duplicate", "scn-site-b-cup"]
    ).optimize_diversity(
        limit=2,
        constraint_spec={
            "minimum": {"object_category": {"cup": 2}},
            "quality": {"column": "quality_score", "weight": 2.0},
            "label_completeness": {"weight": 2.0},
        },
        duplicate_threshold=0.999,
        max_per_duplicate_group=1,
    )

    assert "scn-high-quality-duplicate" in optimized.scenario_ids
    assert "scn-site-b-cup" in optimized.scenario_ids
    assert "scn-anchor" not in optimized.scenario_ids
    assert "scn-duplicate" not in optimized.scenario_ids
    assert all(count <= 1 for count in optimized.report["duplicate_group_usage"].values())


def test_stratified_sample_hits_requested_per_slice_counts(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    sampled = lake.curate.workbench().stratified_sample(
        by=["site_id", "object_category"], per_slice=1
    )

    rows = {row["scenario_id"]: row for row in lake.table("scenarios").to_arrow().to_pylist()}
    run_site = {row["run_id"]: row["site_id"] for row in lake.table("runs").to_arrow().to_pylist()}
    slices = Counter()
    for scenario_id in sampled.scenario_ids:
        row = rows[scenario_id]
        object_category = next(
            tag.split(":", 1)[1]
            for tag in row["coverage_tags"]
            if tag.startswith("object_category:")
        )
        slices[(run_site[row["run_id"]], object_category)] += 1

    assert slices == {
        ("site-a", "cup"): 1,
        ("site-a", "box"): 1,
        ("site-b", "cup"): 1,
        ("site-b", "box"): 1,
    }
    assert set(sampled.report["slice_counts"].values()) == {1}


def test_mine_failures_returns_seed_neighbors_ahead_of_unrelated_rows(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    mined = lake.curate.workbench().mine_failures(seed_event="evt-failure", limit=3)

    assert mined.scenario_ids == ("scn-anchor", "scn-duplicate", "scn-neighbor")
    scores = mined.report["neighbors"]
    assert [row["scenario_id"] for row in scores] == list(mined.scenario_ids)
    assert scores[0]["similarity"] >= scores[1]["similarity"] >= scores[2]["similarity"]


def test_failure_review_queue_ranks_neighbors_and_exports_lineage_manifest(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    queue = lake.curate.workbench().failure_review_queue(
        "eval-regression-review",
        seed_event="evt-failure",
        limit=3,
        assignee="qa-team",
    )

    assert queue.scenario_ids == ("scn-anchor", "scn-duplicate", "scn-neighbor")
    assert queue.target_ids == queue.scenario_ids
    rows = queue.rows()
    assert [row["target_id"] for row in rows] == list(queue.scenario_ids)
    assert [row["priority"] for row in rows] == [1, 2, 3]
    assert rows[0]["priority_score"] >= rows[1]["priority_score"] >= rows[2]["priority_score"]
    assert rows[0]["source_operation"] == "failure-mining"
    assert rows[0]["assignee"] == "qa-team"
    assert rows[0]["status"] == "open"
    source_ref = json.loads(rows[0]["source_ref"])
    assert source_ref["seed"]["event_id"] == "evt-failure"

    transform = _transform(lake, queue.transform_id)
    assert transform["kind"] == "curation-review-queue"
    assert "curation_review_queues" in transform["output_tables"]

    manifest = queue.export_manifest(
        tool="label-studio",
        output_uri="s3://review/eval-regression-review.json",
    )

    assert manifest["kind"] == "curation-review-queue-export"
    assert manifest["queue_id"] == queue.queue_id
    assert manifest["queue_transform_id"] == queue.transform_id
    assert queue.transform_id in manifest["source_transform_ids"]
    assert {row["scenario_id"] for row in manifest["items"]} == set(queue.scenario_ids)
    assert {row["table"] for row in manifest["source_table_versions"]} >= {
        "scenarios",
        "events",
        "curation_review_queues",
    }
    assert manifest["export_transform_id"].startswith("tfm-curate-review_queue_export-")
    accounting = manifest["projection_accounting"]
    assert accounting["target_format"] == "labeling-manifest:label-studio"
    assert accounting["logical_row_count"] == len(queue.scenario_ids)
    assert accounting["payload_bytes_referenced"] > 0
    assert accounting["payload_bytes_copied"] == 0
    assert accounting["projection_transform_id"] == manifest["export_transform_id"]


def test_review_queue_pages_summary_and_paged_export_are_stable(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _add_synthetic_review_queue(lake, count=30)

    queue = lake.curate.queue("large-review")
    first = queue.page(limit=7)
    second = queue.page(limit=7, cursor=first.next_cursor)
    first_keys = [(row["priority"], row["target_id"], row["queue_item_id"]) for row in first.rows]
    second_keys = [
        (row["priority"], row["target_id"], row["queue_item_id"]) for row in second.rows
    ]

    assert queue.item_count == 30
    assert queue.item_ids == ()
    assert first.has_more is True
    assert len(first.rows) == 7
    assert len(second.rows) == 7
    assert first_keys == sorted(first_keys)
    assert second_keys == sorted(second_keys)
    assert set(first.item_ids).isdisjoint(second.item_ids)
    assert max(first_keys) < min(second_keys)

    summary = queue.summary(batch_size=5)

    assert summary["item_count"] == 30
    assert summary["counts_by_status"] == {"assigned": 15, "open": 15}
    assert summary["counts_by_assignee"] == {"qa-a": 15, "qa-b": 15}
    assert summary["counts_by_source_operation"] == {
        "active-learning": 20,
        "failure-mining": 10,
    }
    assert summary["counts_by_priority_band"]["1"] == 4
    assert summary["counts_by_priority_band"]["11-50"] == 8

    manifest = queue.export_manifest(tool="label-studio", limit=5)

    assert len(manifest["items"]) == 5
    assert manifest["total_item_count"] == 30
    assert manifest["page"]["has_more"] is True
    assert manifest["page"]["next_cursor"]
    assert manifest["projection_accounting"]["logical_row_count"] == 5


def test_review_connector_export_uses_deterministic_payloads_and_idempotency(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    queue = lake.curate.workbench().failure_review_queue(
        "connector-review",
        seed_event="evt-failure",
        limit=3,
    )

    plan = queue.export_to_connector(
        tool="label-studio",
        project_id="project-a",
        dry_run=True,
    )
    connector = _RecordingReviewConnector()
    report = queue.export_to_connector(
        connector,
        tool="label-studio",
        project_id="project-a",
        output_uri="label-studio://project-a",
    )

    assert plan.counts_by_status == {"queued": 3}
    assert report.counts_by_status == {"exported": 3}
    assert [task.idempotency_key for task in connector.upsert_calls[0]] == [
        result.idempotency_key for result in plan.results
    ]
    first_task = connector.upsert_calls[0][0]
    assert first_task.idempotency_key.startswith("review-task-")
    assert first_task.payload["queue_id"] == queue.queue_id
    assert first_task.payload["queue_item_id"] == first_task.queue_item_id
    assert queue.transform_id in first_task.payload["source_transform_ids"]
    assert first_task.payload["source_ref"]["seed"]["event_id"] == "evt-failure"

    rows = lake.curate.queue(queue.queue_id).rows()
    assert {row["status"] for row in rows} == {"exported"}
    assert all(row["external_task_id"].startswith("ls-task-") for row in rows)
    connector_metadata = _metadata(rows[0])["review_connectors"]["label-studio:project-a"]
    assert connector_metadata["status"] == "exported"
    assert connector_metadata["transform_id"] == report.transform_id
    transform = _transform(lake, report.transform_id)
    assert transform["kind"] == "curation-review-queue-connector-export"


def test_review_connector_reexport_reports_already_present_without_duplicate_tasks(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    queue = lake.curate.workbench().failure_review_queue(
        "connector-review",
        seed_event="evt-failure",
        limit=3,
    )
    connector = _RecordingReviewConnector()

    queue.export_to_connector(connector, tool="label-studio", project_id="project-a")
    second = queue.export_to_connector(connector, tool="label-studio", project_id="project-a")

    assert second.counts_by_status == {"already-present": 3}
    assert len(connector.upsert_calls) == 1
    rows = lake.curate.queue(queue.queue_id).rows()
    assert len({row["external_task_id"] for row in rows}) == 3


def test_review_connector_partial_failure_is_retryable(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    queue = lake.curate.workbench().failure_review_queue(
        "connector-review",
        seed_event="evt-failure",
        limit=3,
    )
    failed_item_id = queue.item_ids[1]
    failing = _RecordingReviewConnector(fail_queue_item_ids={failed_item_id})

    first = queue.export_to_connector(failing, tool="label-studio", project_id="project-a")

    assert first.counts_by_status == {"exported": 2, "failed": 1}
    rows_after_first = {row["queue_item_id"]: row for row in lake.curate.queue(queue.queue_id).rows()}
    assert rows_after_first[failed_item_id]["external_task_id"] == ""
    assert _metadata(rows_after_first[failed_item_id])["review_connectors"]["label-studio:project-a"][
        "error"
    ] == "injected connector failure"

    retry = _RecordingReviewConnector()
    second = queue.export_to_connector(retry, tool="label-studio", project_id="project-a")

    assert second.counts_by_status == {"already-present": 2, "exported": 1}
    assert len(retry.upsert_calls) == 1
    assert [task.queue_item_id for task in retry.upsert_calls[0]] == [failed_item_id]
    rows_after_retry = lake.curate.queue(queue.queue_id).rows()
    assert all(row["external_task_id"] for row in rows_after_retry)


def test_review_connector_status_sync_updates_queue_rows(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    queue = lake.curate.workbench().failure_review_queue(
        "connector-review",
        seed_event="evt-failure",
        limit=3,
    )
    connector = _RecordingReviewConnector()
    queue.export_to_connector(connector, tool="label-studio", project_id="project-a")
    statuses = {
        queue.item_ids[0]: "completed",
        queue.item_ids[1]: "reviewing",
        queue.item_ids[2]: "canceled",
    }
    connector.statuses = statuses

    report = queue.sync_connector_status(connector, tool="label-studio", project_id="project-a")

    assert report.counts_by_status == {"completed": 1, "exported": 1, "skipped": 1}
    row_statuses = {
        row["queue_item_id"]: row["status"] for row in lake.curate.queue(queue.queue_id).rows()
    }
    assert row_statuses == {
        queue.item_ids[0]: "completed",
        queue.item_ids[1]: "exported",
        queue.item_ids[2]: "skipped",
    }
    assert _transform(lake, report.transform_id)["kind"] == (
        "curation-review-queue-connector-status-sync"
    )


def test_review_connector_outcome_import_matches_local_writeback_semantics(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    queue = lake.curate.workbench().failure_review_queue(
        "reviewed-failures",
        seed_event="evt-failure",
        limit=3,
    )
    connector = _RecordingReviewConnector()
    queue.export_to_connector(connector, tool="label-studio", project_id="project-a")
    connector.outcomes = {
        queue.item_ids[0]: {
            "decision": "include",
            "reason_code": "confirmed-failure",
            "label_type": "failure_review",
            "label": "true_positive",
            "feedback_type": "failure-review",
            "severity": "high",
            "notes": "confirmed by qa",
            "reviewer": "qa",
            "status": "accepted",
        },
        queue.item_ids[1]: {
            "decision": "exclude",
            "reason_code": "duplicate",
            "reviewer": "qa",
        },
        queue.item_ids[2]: {
            "decision": "exclude",
            "reason_code": "not-the-same-failure",
            "reviewer": "qa",
        },
    }

    report = queue.import_connector_outcomes(
        connector,
        tool="label-studio",
        project_id="project-a",
        created_by="qa-import",
    )
    curated = lake.curate.workbench(scope="reviewed-failures", apply_decisions=True)

    assert report.outcome_count == 3
    assert report.outcome_report is not None
    assert report.outcome_report.label_ids
    assert report.outcome_report.feedback_ids
    assert curated.scenario_ids == ("scn-anchor",)
    memberships = [
        row
        for row in lake.table("curation_memberships").to_arrow().to_pylist()
        if row["queue"] == "reviewed-failures"
    ]
    assert {row["scenario_id"]: row["decision"] for row in memberships} == {
        "scn-anchor": "include",
        "scn-duplicate": "exclude",
        "scn-neighbor": "exclude",
    }
    assert _transform(lake, report.transform_id)["kind"] == (
        "curation-review-queue-connector-outcome-import"
    )


def test_active_learning_queue_selects_low_confidence_and_caps_duplicate_cluster(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.table("model_outputs").add(
        pa.Table.from_pylist(
            [
                {
                    "model_output_id": "out-anchor",
                    "run_id": "run-a",
                    "observation_id": "",
                    "scenario_id": "scn-anchor",
                    "dataset_id": "",
                    "model_version": "policy-v1",
                    "output_type": "classification",
                    "prediction": "maybe-cup",
                    "output_json": "{}",
                    "score": 0.10,
                    "producer_run_id": "eval-v1",
                    "source": "fixture",
                    "metadata": [],
                    "transform_id": "tfm-output-anchor",
                    "created_at": NOW,
                },
                {
                    "model_output_id": "out-duplicate",
                    "run_id": "run-a",
                    "observation_id": "",
                    "scenario_id": "scn-duplicate",
                    "dataset_id": "",
                    "model_version": "policy-v1",
                    "output_type": "classification",
                    "prediction": "maybe-cup",
                    "output_json": "{}",
                    "score": 0.05,
                    "producer_run_id": "eval-v1",
                    "source": "fixture",
                    "metadata": [],
                    "transform_id": "tfm-output-duplicate",
                    "created_at": NOW,
                },
                {
                    "model_output_id": "out-neighbor",
                    "run_id": "run-a",
                    "observation_id": "",
                    "scenario_id": "scn-neighbor",
                    "dataset_id": "",
                    "model_version": "policy-v1",
                    "output_type": "classification",
                    "prediction": "uncertain-box",
                    "output_json": "{}",
                    "score": 0.20,
                    "producer_run_id": "eval-v1",
                    "source": "fixture",
                    "metadata": [],
                    "transform_id": "tfm-output-neighbor",
                    "created_at": NOW,
                },
                {
                    "model_output_id": "out-site-b-cup",
                    "run_id": "run-b",
                    "observation_id": "",
                    "scenario_id": "scn-site-b-cup",
                    "dataset_id": "",
                    "model_version": "policy-v1",
                    "output_type": "classification",
                    "prediction": "uncertain-cup",
                    "output_json": "{}",
                    "score": 0.30,
                    "producer_run_id": "eval-v1",
                    "source": "fixture",
                    "metadata": [],
                    "transform_id": "tfm-output-site-b-cup",
                    "created_at": NOW,
                },
            ],
            schema=MODEL_OUTPUTS_SCHEMA,
        )
    )

    queue = lake.curate.workbench().active_learning_queue(
        "uncertain-policy-v1",
        limit=3,
        model_version="policy-v1",
        output_type="classification",
        duplicate_threshold=0.999,
        max_per_duplicate_group=1,
    )

    assert queue.scenario_ids == ("scn-duplicate", "scn-neighbor", "scn-site-b-cup")
    assert "scn-anchor" not in queue.scenario_ids
    rows = queue.rows()
    assert [row["priority_score"] for row in rows] == pytest.approx([0.95, 0.80, 0.70])
    assert all("low-confidence" in row["priority_reason"] for row in rows)

    active_transform = _transform(lake, queue.source_transform_ids[-1])
    params = json.loads(active_transform["params"])
    assert params["operation"] == "active-learning"
    assert params["duplicate_group_usage"]
    assert all(count <= 1 for count in params["duplicate_group_usage"].values())
    # The default queue records the 0056 uncertainty scorer + confidence calibration.
    assert params["scorer"]["name"] == "uncertainty"
    assert params["calibration"]["mode"] == "confidence"


def _scorer_kv_metadata(values: dict) -> list[dict]:
    return [{"key": key, "value": json.dumps(value)} for key, value in values.items()]


def _scorer_model_output(
    model_output_id: str,
    scenario_id: str,
    *,
    score=None,
    metadata: dict | None = None,
    model_version: str = "policy-v1",
    output_type: str = "classification",
    prediction: str = "pred",
    run_id: str = "run-a",
) -> dict:
    return {
        "model_output_id": model_output_id,
        "run_id": run_id,
        "observation_id": "",
        "scenario_id": scenario_id,
        "dataset_id": "",
        "model_version": model_version,
        "output_type": output_type,
        "prediction": prediction,
        "output_json": "{}",
        "score": score,
        "producer_run_id": "eval-v1",
        "source": "fixture",
        "metadata": _scorer_kv_metadata(metadata or {}),
        "transform_id": f"tfm-{model_output_id}",
        "created_at": NOW,
    }


def _add_scorer_model_outputs(lake: Lake, rows: list[dict]) -> None:
    lake.table("model_outputs").add(pa.Table.from_pylist(rows, schema=MODEL_OUTPUTS_SCHEMA))


def test_active_learning_custom_scorer_ranks_and_records_scorer_metadata(tmp_path):
    from lancedb_robotics.scoring import (
        ActiveLearningScorer,
        ScorerResult,
        register_scorer,
        unregister_scorer,
    )

    lake = _build_curation_lake(tmp_path / "robot.lance")
    _add_scorer_model_outputs(
        lake,
        [
            _scorer_model_output("out-a", "scn-anchor", score=0.5, metadata={"risk": 0.2}),
            _scorer_model_output("out-n", "scn-neighbor", score=0.5, metadata={"risk": 0.9}),
            _scorer_model_output("out-c", "scn-site-b-cup", score=0.5, metadata={"risk": 0.6}),
        ],
    )

    class _RiskScorer(ActiveLearningScorer):
        name = "incident-risk"
        version = "2"

        def params(self):
            return {"field": "risk"}

        def score(self, candidate):
            best = None
            for output in candidate.outputs:
                value = output.metadata.get("risk")
                if value is None:
                    continue
                if best is None or float(value) > best[0]:
                    best = (float(value), output.model_output_id)
            if best is None:
                return None
            return ScorerResult(
                score=best[0], reason=f"risk={best[0]:.3f}", metric="risk", model_output_id=best[1]
            )

    queue = lake.curate.workbench().active_learning_queue(
        "incident-risk-queue", limit=3, scorer=_RiskScorer(), max_per_duplicate_group=0
    )
    # Planted risk values drive the ranking, not model confidence.
    assert queue.scenario_ids == ("scn-neighbor", "scn-site-b-cup", "scn-anchor")
    rows = {row["scenario_id"]: row for row in queue.rows()}
    assert rows["scn-neighbor"]["priority_score"] == pytest.approx(0.9)
    source_ref = json.loads(rows["scn-neighbor"]["source_ref"])
    assert source_ref["scorer"] == "incident-risk"
    assert source_ref["scorer_version"] == "2"
    assert source_ref["metric"] == "risk"

    params = json.loads(_transform(lake, queue.source_transform_ids[-1])["params"])
    assert params["scorer"] == {"name": "incident-risk", "version": "2", "params": {"field": "risk"}}

    # The same scorer is usable by registered name without changing queue code.
    register_scorer("incident-risk", _RiskScorer, overwrite=True)
    try:
        by_name = lake.curate.workbench().active_learning_queue(
            "incident-risk-by-name", limit=3, scorer="incident-risk", max_per_duplicate_group=0
        )
        assert by_name.scenario_ids == ("scn-neighbor", "scn-site-b-cup", "scn-anchor")
    finally:
        unregister_scorer("incident-risk")


def test_active_learning_calibration_mode_changes_priority_scores(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _add_scorer_model_outputs(
        lake,
        [
            _scorer_model_output("out-a", "scn-anchor", score=0.10),
            _scorer_model_output("out-n", "scn-neighbor", score=0.40),
            _scorer_model_output("out-c", "scn-site-b-cup", score=0.80),
        ],
    )

    confidence_queue = lake.curate.workbench().active_learning_queue(
        "al-confidence", limit=3, scorer="confidence-margin", max_per_duplicate_group=0
    )
    loss_queue = lake.curate.workbench().active_learning_queue(
        "al-loss", limit=3, scorer="confidence-margin", calibration="loss", max_per_duplicate_group=0
    )

    # Confidence: low score => high priority. Loss: high score => high priority.
    assert confidence_queue.scenario_ids == ("scn-anchor", "scn-neighbor", "scn-site-b-cup")
    assert loss_queue.scenario_ids == ("scn-site-b-cup", "scn-neighbor", "scn-anchor")
    confidence_scores = {row["scenario_id"]: row["priority_score"] for row in confidence_queue.rows()}
    loss_scores = {row["scenario_id"]: row["priority_score"] for row in loss_queue.rows()}
    assert confidence_scores["scn-anchor"] == pytest.approx(0.90)
    assert loss_scores["scn-anchor"] == pytest.approx(0.10)

    confidence_params = json.loads(_transform(lake, confidence_queue.source_transform_ids[-1])["params"])
    loss_params = json.loads(_transform(lake, loss_queue.source_transform_ids[-1])["params"])
    assert confidence_params["calibration"]["mode"] == "confidence"
    assert loss_params["calibration"]["mode"] == "loss"


def test_active_learning_ensemble_disagreement_consumes_multiple_outputs(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _add_scorer_model_outputs(
        lake,
        [
            _scorer_model_output("out-a1", "scn-anchor", metadata={"member_score": 0.1}),
            _scorer_model_output("out-a2", "scn-anchor", metadata={"member_score": 0.5}),
            _scorer_model_output("out-a3", "scn-anchor", metadata={"member_score": 0.9}),
            _scorer_model_output("out-n1", "scn-neighbor", metadata={"member_score": 0.50}),
            _scorer_model_output("out-n2", "scn-neighbor", metadata={"member_score": 0.52}),
            _scorer_model_output("out-n3", "scn-neighbor", metadata={"member_score": 0.55}),
        ],
    )

    queue = lake.curate.workbench().active_learning_queue(
        "ensemble-disagreement", limit=2, scorer="ensemble-disagreement", max_per_duplicate_group=0
    )
    assert queue.scenario_ids == ("scn-anchor", "scn-neighbor")
    rows = {row["scenario_id"]: row for row in queue.rows()}
    # Disagreement = spread of member_score across the scenario's model_outputs rows.
    assert rows["scn-anchor"]["priority_score"] == pytest.approx(0.8)
    assert rows["scn-neighbor"]["priority_score"] == pytest.approx(0.05)
    assert "members=3" in rows["scn-anchor"]["priority_reason"]


def test_active_learning_benchmark_reports_baseline_comparisons(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _add_scorer_model_outputs(
        lake,
        [
            _scorer_model_output("out-a", "scn-anchor", score=0.10),
            _scorer_model_output("out-n", "scn-neighbor", score=0.40),
            _scorer_model_output("out-c", "scn-site-b-cup", score=0.80),
            _scorer_model_output("out-b", "scn-site-b-box", score=0.60),
        ],
    )

    report = lake.curate.workbench().evaluate_active_learning_selection(
        limit=3, scorer="confidence-margin", seed=7
    )
    assert set(report["strategies"]) == {"scored", "random", "low-confidence", "diversity-only"}
    assert report["scorer"]["name"] == "confidence-margin"
    for metrics in report["strategies"].values():
        assert {"mean_scored_priority", "overlap_with_scored", "jaccard_with_scored"} <= set(metrics)
    assert report["strategies"]["scored"]["jaccard_with_scored"] == pytest.approx(1.0)
    # confidence-margin and an explicit low-confidence baseline agree here.
    assert report["strategies"]["low-confidence"]["jaccard_with_scored"] == pytest.approx(1.0)
    assert report["transform_id"]
    assert _transform(lake, report["transform_id"])["kind"] == "curation-active-learning-benchmark"

    # Determinism: identical inputs/seed reproduce the random baseline selection.
    repeat = lake.curate.workbench().evaluate_active_learning_selection(
        limit=3, scorer="confidence-margin", seed=7, record=False
    )
    assert (
        repeat["strategies"]["random"]["scenario_ids"]
        == report["strategies"]["random"]["scenario_ids"]
    )


def test_active_learning_distribution_gap_boost_scorer(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    # No model outputs; the gap-boost scorer fires purely on slice deficits.
    queue = lake.curate.workbench().active_learning_queue(
        "gap-boost",
        limit=10,
        scorer="distribution-gap-boost",
        gap_by=("site_id",),
        gap_min_per_slice=7,
        max_per_duplicate_group=0,
    )
    rows = {row["scenario_id"]: row for row in queue.rows()}
    # Every slice is under the (deliberately high) target, so each scenario is
    # boosted and its reason carries the slice label + remaining deficit.
    assert rows
    assert all("distribution-gap" in row["priority_reason"] for row in rows.values())
    assert all("needed=" in row["priority_reason"] for row in rows.values())
    params = json.loads(_transform(lake, queue.source_transform_ids[-1])["params"])
    assert params["gap_by"] == ["site_id"]
    assert params["scorer"]["name"] == "distribution-gap-boost"


def test_cli_active_learning_scorer_and_benchmark(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    _add_scorer_model_outputs(
        lake,
        [
            _scorer_model_output("out-a", "scn-anchor", score=0.10),
            _scorer_model_output("out-n", "scn-neighbor", score=0.40),
            _scorer_model_output("out-c", "scn-site-b-cup", score=0.80),
        ],
    )

    al_result = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "active-learning",
            "--lake",
            str(lake_path),
            "--queue",
            "cli-loss-queue",
            "--scorer",
            "confidence-margin",
            "--calibration",
            "loss",
            "--max-per-duplicate-group",
            "0",
            "--limit",
            "3",
        ],
    )
    assert al_result.exit_code == 0, al_result.output
    assert "queue: cli-loss-queue" in al_result.output

    benchmark_result = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "benchmark",
            "--lake",
            str(lake_path),
            "--scorer",
            "confidence-margin",
            "--seed",
            "3",
            "--limit",
            "2",
        ],
    )
    assert benchmark_result.exit_code == 0, benchmark_result.output
    payload = json.loads(benchmark_result.output)
    assert payload["operation"] == "active-learning-benchmark"
    assert set(payload["strategies"]) == {"scored", "random", "low-confidence", "diversity-only"}


def test_review_outcomes_write_labels_feedback_and_membership_for_snapshot(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    queue = lake.curate.workbench().failure_review_queue(
        "reviewed-failures",
        seed_event="evt-failure",
        limit=3,
    )

    report = queue.import_outcomes(
        [
            {
                "scenario_id": "scn-anchor",
                "decision": "include",
                "reason_code": "confirmed-failure",
                "label_type": "failure_review",
                "label": "true_positive",
                "feedback_type": "failure-review",
                "severity": "high",
                "notes": "confirmed by qa",
                "reviewer": "qa",
                "status": "accepted",
            },
            {
                "scenario_id": "scn-duplicate",
                "decision": "exclude",
                "reason_code": "duplicate",
                "reviewer": "qa",
            },
            {
                "scenario_id": "scn-neighbor",
                "decision": "exclude",
                "reason_code": "not-the-same-failure",
                "reviewer": "qa",
            },
        ],
        created_by="qa-import",
    )
    curated = lake.curate.workbench(scope="reviewed-failures", apply_decisions=True)
    manifest = curated.snapshot(name="reviewed-failure-positives", split_by="scenario")

    assert report.outcome_count == 3
    assert report.decision_transform_ids
    assert report.label_ids
    assert report.feedback_ids
    assert curated.scenario_ids == ("scn-anchor",)
    assert manifest.scenario_ids == ("scn-anchor",)

    label = next(row for row in lake.table("labels").to_arrow().to_pylist() if row["label_id"] in report.label_ids)
    feedback = next(
        row for row in lake.table("feedback").to_arrow().to_pylist()
        if row["feedback_id"] in report.feedback_ids
    )
    memberships = [
        row
        for row in lake.table("curation_memberships").to_arrow().to_pylist()
        if row["queue"] == "reviewed-failures"
    ]

    assert label["scenario_id"] == "scn-anchor"
    assert label["label_type"] == "failure_review"
    assert label["label"] == "true_positive"
    assert feedback["scenario_id"] == "scn-anchor"
    assert feedback["feedback_type"] == "failure-review"
    assert feedback["severity"] == "high"
    assert {row["scenario_id"]: row["decision"] for row in memberships} == {
        "scn-anchor": "include",
        "scn-duplicate": "exclude",
        "scn-neighbor": "exclude",
    }


def test_review_outcome_import_reports_bad_row_and_keeps_committed_audit(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    queue = lake.curate.workbench().failure_review_queue(
        "reviewed-failures",
        seed_event="evt-failure",
        limit=3,
    )

    with pytest.raises(CurationError) as excinfo:
        queue.import_outcomes(
            (
                row
                for row in [
                    {
                        "scenario_id": "scn-anchor",
                        "decision": "include",
                        "reason_code": "confirmed-failure",
                        "label_type": "failure_review",
                        "label": "true_positive",
                        "reviewer": "qa",
                    },
                    {
                        "scenario_id": "scn-missing",
                        "decision": "exclude",
                        "reason_code": "not-in-queue",
                        "reviewer": "qa",
                    },
                ]
            ),
            created_by="qa-import",
            writeback_batch_size=1,
        )

    assert "review outcome row 1" in str(excinfo.value)
    assert "scn-missing" in str(excinfo.value)
    labels = lake.table("labels").to_arrow().to_pylist()
    memberships = lake.table("curation_memberships").to_arrow().to_pylist()

    assert any(
        row["scenario_id"] == "scn-anchor" and row["label_type"] == "failure_review"
        for row in labels
    )
    committed = [
        row
        for row in memberships
        if row["queue"] == "reviewed-failures" and row["scenario_id"] == "scn-anchor"
    ]
    assert committed
    metadata = {item["key"]: json.loads(item["value"]) for item in committed[0]["metadata"]}
    assert metadata["review_outcome_row_index"] == 0


def test_branch_and_snapshot_are_version_pinned_without_copying_payload_rows(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    before_observations = lake.table("observations").count_rows()

    workbench = lake.curate.workbench(scope=lake.scope(site_id="site-a"))
    deduped = workbench.dedup(near_duplicate_threshold=0.999)
    branch_a = deduped.branch("candidate-a")
    branch_a.snapshot(tag="candidate-a-frozen")
    branch_b = deduped.stratified_sample(by=["object_category"], per_slice=1).branch(
        "candidate-b"
    )

    assert lake.table("observations").count_rows() == before_observations
    assert branch_a.manifest.name == "candidate-a"
    assert branch_b.manifest.name == "candidate-b"

    row_a = _snapshot_row(lake, "candidate-a")
    row_b = _snapshot_row(lake, "candidate-b")
    assert row_a["tag"] == "candidate-a-frozen"
    assert row_b["tag"] == "candidate-b"
    assert {tv["table"] for tv in row_a["table_versions"]} >= {"scenarios", "runs"}
    spec = json.loads(row_a["query_spec"])
    assert spec["source"]["kind"] == "curation-workbench"
    assert deduped.transform_id in spec["source"]["operation_transform_ids"]


def test_saved_view_reopens_pinned_order_after_live_table_mutation(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    workbench = lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"])
    view = workbench.save_view(
        "pinned-review",
        owner="qa-team",
        tags=("edge-case", "training"),
        description="stable review candidates",
    )
    pinned_versions = dict(view.table_versions)

    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                _scenario(
                    "scn-late-arrival",
                    run_id="run-b",
                    start_time_ns=99,
                    object_category="cup",
                    embedding=[0.0, 0.0, 0.0, 1.0],
                )
            ],
            schema=lake.table("scenarios").schema,
        )
    )

    reopened = lake.curate.workbench(scope="pinned-review")
    row = next(
        row
        for row in lake.table("curation_views").to_arrow().to_pylist()
        if row["view_id"] == view.view_id
    )

    assert reopened.scenario_ids == ("scn-anchor", "scn-neighbor")
    assert "scn-late-arrival" not in reopened.scenario_ids
    assert pinned_versions["scenarios"] < int(lake.table("scenarios").version)
    assert row["owner"] == "qa-team"
    assert row["tags"] == ["edge-case", "training"]
    assert reopened.report["table_versions"] == [
        {"table": table, "version": version, "tag": ""}
        for table, version in view.table_versions
    ]


def test_large_saved_view_uses_chunked_membership_storage(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    selected = (
        "scn-site-b-box-extra",
        "scn-anchor",
        "scn-neighbor",
        "scn-site-b-cup",
        "scn-duplicate",
    )

    workbench = lake.curate.workbench(scope=selected)
    expected_order = workbench.scenario_ids

    view = workbench.save_view(
        "chunked-review",
        inline_scenario_limit=2,
        membership_chunk_size=2,
    )
    reopened = lake.curate.workbench(scope="chunked-review")
    view_row = next(
        row
        for row in lake.table("curation_views").to_arrow().to_pylist()
        if row["view_id"] == view.view_id
    )
    chunks = sorted(
        [
            row
            for row in lake.table("curation_view_membership_chunks").to_arrow().to_pylist()
            if row["view_id"] == view.view_id
        ],
        key=lambda row: row["start_ordinal"],
    )
    query_spec = json.loads(view_row["query_spec"])

    assert view.scenario_ids == expected_order
    assert view.membership_storage == "chunked"
    assert view_row["scenario_ids"] == []
    assert query_spec["scenario_ids"] == []
    assert query_spec["membership_storage"]["kind"] == "chunked"
    assert query_spec["membership_storage"]["chunk_count"] == 3
    assert [row["scenario_ids"] for row in chunks] == [
        list(expected_order[:2]),
        list(expected_order[2:4]),
        list(expected_order[4:]),
    ]
    assert reopened.scenario_ids == expected_order
    assert reopened.report["membership_storage"]["kind"] == "chunked"
    assert {entry["table"] for entry in reopened.report["predicate_indexes"]} >= {
        "curation_memberships",
        "curation_view_membership_chunks",
    }


def test_existing_inline_saved_view_rows_continue_to_reopen(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    view = lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).save_view(
        "inline-review",
        inline_scenario_limit=100,
    )
    row = next(
        row
        for row in lake.table("curation_views").to_arrow().to_pylist()
        if row["view_id"] == view.view_id
    )

    reopened = lake.curate.workbench(scope="inline-review")

    assert row["scenario_ids"] == ["scn-anchor", "scn-neighbor"]
    assert reopened.scenario_ids == ("scn-anchor", "scn-neighbor")
    assert reopened.report["membership_storage"]["kind"] == "inline"


def test_chunked_view_snapshot_records_membership_and_skipped_index_status(
    tmp_path, monkeypatch
):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    def fake_describe_curation_predicate_indexes(lake, *, include_view_chunks=True):
        assert include_view_chunks is True
        return (
            ScalarIndexResult(
                table="curation_memberships",
                column="view_id",
                status="skipped",
                reason="test backend does not expose create_scalar_index",
            ),
            ScalarIndexResult(
                table="curation_view_membership_chunks",
                column="view_id",
                status="skipped",
                reason="test backend does not expose create_scalar_index",
            ),
        )

    monkeypatch.setattr(
        curate_mod,
        "describe_curation_predicate_indexes",
        fake_describe_curation_predicate_indexes,
    )
    workbench = lake.curate.workbench(scope=["scn-neighbor", "scn-anchor", "scn-site-b-cup"])
    expected_order = workbench.scenario_ids
    workbench.save_view(
        "chunked-snapshot-source",
        inline_scenario_limit=1,
        membership_chunk_size=2,
    )
    selection = lake.curate.workbench(scope="chunked-snapshot-source")
    manifest = selection.snapshot(name="chunked-snapshot", split_by="scenario")
    snapshot = _snapshot_row(lake, "chunked-snapshot")
    query_spec = json.loads(snapshot["query_spec"])
    source_report = query_spec["source"]["report"]

    assert selection.scenario_ids == expected_order
    assert manifest.scenario_ids == tuple(sorted(selection.scenario_ids))
    assert source_report["membership_storage"]["kind"] == "chunked"
    assert {entry["status"] for entry in source_report["predicate_indexes"]} == {"skipped"}
    assert "curation_view_membership_chunks" in {
        item["table"] for item in snapshot["table_versions"]
    }


def test_include_exclude_decisions_override_automatic_selection(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    workbench = lake.curate.workbench()
    workbench.save_view("review")
    automatic = workbench.dedup(near_duplicate_threshold=0.999)
    workbench.record_decisions(
        view_name="review",
        decision="include",
        scenario_ids=["scn-duplicate"],
        reason_code="dedup-override",
        note="keep a borderline duplicate for regression coverage",
        source="human",
    )
    workbench.record_decisions(
        view_name="review",
        decision="exclude",
        scenario_ids=["scn-neighbor"],
        reason_code="false-positive",
        note="nearby but not the desired behavior",
        source="human",
    )

    curated = automatic.apply_decisions(view_name="review")
    manifest = curated.snapshot(name="reviewed-with-overrides", split_by="scenario")

    assert "scn-duplicate" in curated.scenario_ids
    assert "scn-neighbor" not in curated.scenario_ids
    assert curated.report["included_by_decision"] == ["scn-duplicate"]
    assert curated.report["removed"] == {"scn-neighbor": "exclude"}
    assert manifest.scenario_ids == tuple(sorted(curated.scenario_ids))


def test_superseded_decisions_preserve_audit_rows_and_resolve_latest(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    workbench = lake.curate.workbench()
    workbench.save_view("supersession")
    first = workbench.record_decisions(
        view_name="supersession",
        decision="exclude",
        scenario_ids=["scn-neighbor"],
        reason_code="initial-review",
        note="hold out until reviewed",
    )
    second = workbench.record_decisions(
        view_name="supersession",
        decision="include",
        scenario_ids=["scn-neighbor"],
        reason_code="review-complete",
        note="cleared for training",
    )

    rows = [
        row
        for row in lake.table("curation_memberships").to_arrow().to_pylist()
        if row["scenario_id"] == "scn-neighbor"
    ]
    curated = workbench.apply_decisions(view_name="supersession")

    assert len(rows) == 2
    assert {row["membership_id"] for row in rows} == {
        first.membership_ids[0],
        second.membership_ids[0],
    }
    latest = max(rows, key=lambda row: (row["created_at"], row["membership_id"]))
    assert latest["decision"] == "include"
    assert latest["supersedes_membership_id"] == first.membership_ids[0]
    assert "scn-neighbor" in curated.scenario_ids


def test_as_of_membership_resolver_replays_latest_decision_by_timestamp(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    workbench = lake.curate.workbench()
    workbench.save_view("audit-replay")
    workbench.record_decisions(
        view_name="audit-replay",
        decision="include",
        scenario_ids=["scn-neighbor"],
        reason_code="initial-include",
    )
    time.sleep(0.01)
    workbench.record_decisions(
        view_name="audit-replay",
        decision="exclude",
        scenario_ids=["scn-neighbor"],
        reason_code="safety-review",
    )
    time.sleep(0.01)
    workbench.record_decisions(
        view_name="audit-replay",
        decision="include",
        scenario_ids=["scn-neighbor"],
        reason_code="review-cleared",
    )

    rows = [
        row
        for row in lake.table("curation_memberships").to_arrow().to_pylist()
        if row["scenario_id"] == "scn-neighbor"
    ]
    rows = sorted(rows, key=lambda row: (row["created_at"], row["membership_id"]))

    first = lake.curate.resolve_membership(
        view_name="audit-replay",
        target_grain="scenario",
        target_ids=["scn-neighbor"],
        as_of=rows[0]["created_at"],
    )
    second = lake.curate.resolve_membership(
        view_name="audit-replay",
        target_grain="scenario",
        target_ids=["scn-neighbor"],
        as_of=rows[1]["created_at"],
    )
    final = lake.curate.resolve_membership(
        view_name="audit-replay",
        target_grain="scenario",
        target_ids=["scn-neighbor"],
        superseded_policy="history",
    )

    assert [row["decision"] for row in first.latest_decisions] == ["include"]
    assert [row["decision"] for row in second.latest_decisions] == ["exclude"]
    assert [row["decision"] for row in final.latest_decisions] == ["include"]
    assert [row["decision"] for row in final.membership_history] == [
        "include",
        "exclude",
        "include",
    ]
    assert final.report["superseded_count"] == 2


def test_snapshot_membership_trace_explains_decisions_and_supersession(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    workbench = lake.curate.workbench()
    automatic = workbench.dedup(near_duplicate_threshold=0.999, view_name="dedup-audit")
    override = workbench.record_decisions(
        view_name="dedup-audit",
        decision="include",
        scenario_ids=["scn-duplicate"],
        reason_code="regression-holdout",
        note="keep this near duplicate for a regression trace",
        source="human",
    )
    curated = automatic.apply_decisions(view_name="dedup-audit")
    manifest = curated.snapshot(name="dedup-audit-snapshot", split_by="scenario")

    trace = lake.curate.trace_membership("dedup-audit-snapshot", "scn-duplicate")

    assert manifest.name == trace.snapshot_name
    assert trace.final_result == "include"
    assert trace.included_in_snapshot is True
    assert [row["decision"] for row in trace.report["membership_history"]] == [
        "exclude",
        "include",
    ]
    assert trace.report["membership_history"][0]["source"] == "dedup"
    assert trace.report["membership_history"][1]["source"] == "human"
    assert override.membership_ids[0] in trace.report["supersession_chain"]
    assert {"curation-dedup", "curation-apply-decisions"} <= {
        row["kind"] for row in trace.report["transforms"]
    }
    assert trace.report["table_version_validation"]["status"] == "passed"


def test_snapshot_membership_trace_reports_unreadable_pinned_versions(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    workbench = lake.curate.workbench()
    workbench.save_view("bad-pins")
    workbench.record_decisions(
        view_name="bad-pins",
        decision="exclude",
        scenario_ids=["scn-neighbor"],
    )
    curated = workbench.apply_decisions(view_name="bad-pins")
    curated.snapshot(name="bad-pins-snapshot", split_by="scenario")

    snapshot = _snapshot_row(lake, "bad-pins-snapshot")
    mutated = dict(snapshot)
    mutated["table_versions"] = [
        {
            **item,
            "version": 999_999 if item["table"] == "curation_memberships" else item["version"],
        }
        for item in snapshot["table_versions"]
    ]
    snapshots = lake.table("dataset_snapshots")
    snapshots.delete(f"dataset_id = '{snapshot['dataset_id']}'")
    snapshots.add(pa.Table.from_pylist([mutated], schema=DATASET_SNAPSHOTS_SCHEMA))

    with pytest.raises(CurationError, match="curation_memberships@999999"):
        lake.curate.trace_membership("bad-pins-snapshot", "scn-neighbor")


def test_observation_grain_label_decision_records_context_without_changing_snapshot(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    workbench = lake.curate.workbench(scope=["scn-anchor"])
    view = workbench.save_view("row-level-review")
    decisions = workbench.record_decisions(
        view_name="row-level-review",
        target_grain="observation",
        target_ids=["obs-scn-anchor"],
        decision="label",
        reason_code="needs-box-label",
        note="route this camera frame to the box-labeling ontology",
        reviewer="qa",
        source="active-learning",
        created_by="curation-bot",
    )
    curated = lake.curate.workbench(scope="row-level-review", apply_decisions=True)

    row = next(
        row
        for row in lake.table("curation_memberships").to_arrow().to_pylist()
        if row["membership_id"] == decisions.membership_ids[0]
    )
    assert view.view_id == decisions.view.view_id
    assert decisions.target_grain == "observation"
    assert decisions.target_ids == ("obs-scn-anchor",)
    assert row["target_grain"] == "observation"
    assert row["target_id"] == "obs-scn-anchor"
    assert row["scenario_id"] == "scn-anchor"
    assert row["decision"] == "label"
    assert row["reason_code"] == "needs-box-label"
    assert row["note"] == "route this camera frame to the box-labeling ontology"
    assert row["source"] == "active-learning"
    assert row["created_by"] == "curation-bot"
    assert curated.scenario_ids == ("scn-anchor",)


def test_flywheel_path_records_decisions_and_freezes_training_snapshot(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    before_observations = lake.table("observations").count_rows()

    workbench = lake.curate.workbench()
    view = workbench.save_view("teleop-review", description="manual review queue")
    decisions = workbench.record_decisions(
        view_name="teleop-review",
        decision="exclude",
        scenario_ids=["scn-neighbor"],
        reason="ambiguous camera obstruction",
        reviewer="qa",
        queue="labeling",
        priority=5,
    )
    curated = (
        workbench
        .filter_quality(min_score=0.9)
        .dedup(near_duplicate_threshold=0.999)
        .distribution_gap(
            by=["site_id", "object_category"],
            min_per_slice=2,
            required_slices=["site_id=site-c|object_category=cup"],
        )
        .apply_decisions(view_name="teleop-review")
    )
    manifest = curated.snapshot(name="flywheel-v1", split_by="scenario")

    assert lake.table("observations").count_rows() == before_observations
    assert manifest.name == "flywheel-v1"
    assert view.view_id == decisions.view.view_id
    assert "scn-duplicate" not in curated.scenario_ids
    assert "scn-neighbor" not in curated.scenario_ids
    assert curated.report["removed"] == {"scn-neighbor": "exclude"}
    gap_transform = next(
        row
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == "curation-distribution-gap-analysis"
    )
    gap_params = json.loads(gap_transform["params"])
    assert "site_id=site-c|object_category=cup" in gap_params["gaps"]

    snapshot = _snapshot_row(lake, "flywheel-v1")
    spec = json.loads(snapshot["query_spec"])
    pinned_tables = {tv["table"] for tv in snapshot["table_versions"]}
    assert {"curation_views", "curation_memberships"} <= pinned_tables
    assert decisions.transform_id in spec["source"]["operation_transform_ids"]
    assert curated.transform_id in spec["source"]["operation_transform_ids"]
    assert json.loads(snapshot["coverage_report"])["operation"] == "apply-decisions"

    preview = load_snapshot_preview(
        lake,
        "flywheel-v1",
        columns=["scenario_id", "split"],
        batch_size=10,
    )
    assert preview.total_scenarios == len(curated.scenario_ids)
    assert {sample["scenario_id"] for sample in preview.samples} == set(curated.scenario_ids)


def test_compare_and_materialization_report_account_for_logical_references(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    lake.curate.workbench().dedup(near_duplicate_threshold=0.999, view_name="auto-dedup")
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a",
        split_by="scenario",
    )
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b",
        split_by="scenario",
    )
    row_a = _snapshot_row(lake, "candidate-a")
    row_b = _snapshot_row(lake, "candidate-b")
    report = lake.curate.materialization_report(
        "candidate-b",
        target_format="webdataset",
        output_uri="s3://exports/candidate-b",
        copied_payload_bytes=0,
        metadata_bytes_written=128,
    )
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
                    "label_id": "lbl-site-b-cup",
                    "run_id": "run-b",
                    "observation_id": "",
                    "scenario_id": "scn-site-b-cup",
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
                    "transform_id": "tfm-label-site-b-cup",
                    "created_at": NOW,
                },
            ],
            schema=LABELS_SCHEMA,
        )
    )
    lake.table("model_outputs").add(
        pa.Table.from_pylist(
            [
                {
                    "model_output_id": "eval-a",
                    "run_id": "",
                    "observation_id": "",
                    "scenario_id": "",
                    "dataset_id": row_a["dataset_id"],
                    "model_version": "policy-v1",
                    "output_type": "eval_success",
                    "prediction": "success_rate",
                    "output_json": "{}",
                    "score": 0.60,
                    "producer_run_id": "train-a",
                    "source": "fixture",
                    "metadata": [],
                    "transform_id": "tfm-eval-a",
                    "created_at": NOW,
                },
                {
                    "model_output_id": "eval-b",
                    "run_id": "",
                    "observation_id": "",
                    "scenario_id": "",
                    "dataset_id": row_b["dataset_id"],
                    "model_version": "policy-v1",
                    "output_type": "eval_success",
                    "prediction": "success_rate",
                    "output_json": "{}",
                    "score": 0.80,
                    "producer_run_id": "train-b",
                    "source": "fixture",
                    "metadata": [],
                    "transform_id": "tfm-eval-b",
                    "created_at": NOW,
                },
            ],
            schema=MODEL_OUTPUTS_SCHEMA,
        )
    )

    comparison = lake.curate.compare("candidate-a", "candidate-b", by=["site_id"])

    assert comparison.comparison_id.startswith("cmp-")
    assert comparison.report["shared_count"] == 1
    assert comparison.report["left_only"] == ["scn-neighbor"]
    assert comparison.report["right_only"] == ["scn-site-b-cup"]
    assert comparison.report["membership"]["removed_scenario_ids"] == ["scn-neighbor"]
    assert comparison.report["membership"]["added_scenario_ids"] == ["scn-site-b-cup"]
    assert comparison.report["left_slices"] == {"site_id=site-a": 2}
    assert comparison.report["right_slices"] == {"site_id=site-a": 1, "site_id=site-b": 1}
    assert comparison.report["coverage"]["deltas"]["site_id=site-b"]["delta_count"] == 1
    assert comparison.report["duplicate_pressure"]["left"]["duplicate_group_count"] == 1
    assert comparison.report["quality"]["left"]["quality_score_mean"] == pytest.approx(0.95)
    assert comparison.report["label_completeness"]["left"]["completeness"] == pytest.approx(0.5)
    assert comparison.report["label_completeness"]["right"]["completeness"] == pytest.approx(1.0)
    assert comparison.report["payload"]["right"]["total_payload_bytes"] > 0
    assert comparison.report["materialization"]["right"]["materialization_count"] == 1
    assert comparison.report["training_eval"]["right"]["training_run_ids"] == ["train-b"]
    assert comparison.report["left_metrics"]["eval_success"]["avg_score"] == pytest.approx(0.60)
    assert comparison.report["right_metrics"]["eval_success"]["avg_score"] == pytest.approx(0.80)

    persisted = next(
        row
        for row in lake.table("curation_comparisons").to_arrow().to_pylist()
        if row["comparison_id"] == comparison.comparison_id
    )
    persisted_report = json.loads(persisted["report_json"])
    assert persisted["left_dataset_id"] == row_a["dataset_id"]
    assert persisted["right_dataset_id"] == row_b["dataset_id"]
    assert persisted["added_scenario_count"] == 1
    assert persisted["removed_scenario_count"] == 1
    assert persisted_report["left_snapshot"]["table_versions"] == comparison.report["left_snapshot"]["table_versions"]
    transform = _transform(lake, comparison.transform_id)
    assert "curation_comparisons" in transform["output_tables"]

    diff = lake.curate.diff_snapshots("candidate-a", "candidate-b")
    assert diff.report["metrics"] == ["membership"]
    assert diff.report["membership"]["shared_scenario_ids"] == ["scn-anchor"]

    assert report.copied_payload_bytes == 0
    assert report.total_payload_bytes > 0
    assert report.logical_reference_bytes > 0
    assert report.metadata_bytes_written == 128
    row = next(
        row
        for row in lake.table("curation_materializations").to_arrow().to_pylist()
        if row["materialization_id"] == report.materialization_id
    )
    assert row["target_format"] == "webdataset"
    assert row["copy_ratio"] == 0.0
    assert row["metadata_bytes_written"] == 128
    assert json.loads(row["report_json"])["accounting"]["metadata_bytes_written"] == 128


def test_feedback_from_eval_links_metrics_to_snapshot_and_branch_comparison(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a",
        split_by="scenario",
    )
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b",
        split_by="scenario",
    )
    row_a = _snapshot_row(lake, "candidate-a")
    row_b = _snapshot_row(lake, "candidate-b")

    left_feedback = lake.curate.feedback_from_eval(
        "candidate-a",
        training_run_id="train-a",
        model_version="policy-v2",
        evaluation_run_id="eval-a",
        metrics=[
            {
                "metric": "success_rate",
                "output_type": "eval_success",
                "slice": "site_id=site-a|object_category=cup",
                "score": 0.80,
                "baseline_score": 0.75,
                "scenario_ids": ["scn-anchor"],
            }
        ],
    )
    right_feedback = lake.curate.feedback_from_eval(
        "candidate-b",
        training_run_id="train-b",
        model_version="policy-v2",
        evaluation_run_id="eval-b",
        metrics=[
            {
                "metric": "success_rate",
                "output_type": "eval_success",
                "slice": "site_id=site-b|object_category=cup",
                "score": 0.45,
                "baseline_score": 0.70,
                "scenario_ids": ["scn-site-b-cup"],
            }
        ],
        regression_threshold=0.05,
    )

    assert left_feedback.dataset_id == row_a["dataset_id"]
    assert right_feedback.dataset_id == row_b["dataset_id"]
    assert len(right_feedback.regressions) == 1
    assert right_feedback.regressions[0]["slice"] == "site_id=site-b|object_category=cup"

    metric_row = next(
        row
        for row in lake.table("model_outputs").to_arrow().to_pylist()
        if row["model_output_id"] == right_feedback.metric_output_ids[0]
    )
    metric_metadata = _metadata(metric_row)
    assert metric_row["dataset_id"] == row_b["dataset_id"]
    assert metric_row["producer_run_id"] == "train-b"
    assert metric_metadata["snapshot_name"] == "candidate-b"
    assert metric_metadata["evaluation_run_id"] == "eval-b"
    assert metric_metadata["regressed"] is True

    transform = _transform(lake, right_feedback.transform_id)
    params = json.loads(transform["params"])
    assert params["dataset_id"] == row_b["dataset_id"]
    assert row_b["transform_id"] in params["prior_operation_transform_ids"]
    assert "model_outputs" in transform["output_tables"]

    comparison = lake.curate.compare("candidate-a", "candidate-b", metrics=["training-eval"])
    right_eval = comparison.report["training_eval"]["right"]
    slice_metrics = right_eval["slice_metrics"]["site_id=site-b|object_category=cup"]
    assert right_eval["training_run_ids"] == ["train-b"]
    assert right_eval["regression_count"] == 1
    assert slice_metrics["success_rate"]["avg_score"] == pytest.approx(0.45)
    assert comparison.report["training_eval"]["delta"]["output_types"]["eval_success"][
        "avg_score"
    ] == pytest.approx(-0.35)


def test_next_candidates_from_eval_regression_is_idempotent_review_queue(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b",
        split_by="scenario",
    )
    feedback = lake.curate.feedback_from_eval(
        "candidate-b",
        training_run_id="train-b",
        model_version="policy-v2",
        evaluation_run_id="eval-b",
        metrics=[
            {
                "metric": "success_rate",
                "slice": "site_id=site-b|object_category=cup",
                "score": 0.45,
                "baseline_score": 0.70,
                "scenario_ids": ["scn-site-b-cup"],
            }
        ],
    )

    first = lake.curate.next_candidates(
        from_regressions=feedback,
        queue_name="eval-b-regressions",
        limit_per_regression=2,
    )
    queue_rows_after_first = lake.table("curation_review_queues").count_rows()
    second = lake.curate.next_candidates(
        from_regressions=feedback,
        queue_name="eval-b-regressions",
        limit_per_regression=2,
    )

    assert first.queue is not None
    assert second.queue is not None
    assert first.queue.queue_id == second.queue.queue_id
    assert lake.table("curation_review_queues").count_rows() == queue_rows_after_first
    assert "scn-site-b-cup" in first.queue.scenario_ids
    assert first.selection.report["regression_count"] == 1
    assert first.selection.transform_id == second.selection.transform_id
    source_ref = json.loads(first.queue.rows()[0]["source_ref"])
    assert source_ref["kind"] == "eval-regression"
    assert source_ref["evaluation_run_id"] == "eval-b"


def test_promote_snapshot_records_rejection_reason_and_source_metrics(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b",
        split_by="scenario",
    )
    snapshot = _snapshot_row(lake, "candidate-b")
    feedback = lake.curate.feedback_from_eval(
        "candidate-b",
        training_run_id="train-b",
        model_version="policy-v2",
        evaluation_run_id="eval-b",
        metrics=[
            {
                "metric": "success_rate",
                "slice": "site_id=site-b|object_category=cup",
                "score": 0.45,
                "baseline_score": 0.70,
                "scenario_ids": ["scn-site-b-cup"],
            }
        ],
    )

    decision = lake.curate.promote_snapshot(
        "candidate-b",
        decision="reject",
        reason="regressed transparent-cup pickups at site-b",
        evaluation_run_id="eval-b",
        training_run_id="train-b",
        model_version="policy-v2",
        metrics=feedback.regressions,
        reviewer="qa",
    )

    assert decision.dataset_id == snapshot["dataset_id"]
    assert decision.decision == "reject"
    row = next(
        row
        for row in lake.table("curation_memberships").to_arrow().to_pylist()
        if row["membership_id"] in decision.membership_ids
    )
    metadata = _metadata(row)
    assert row["target_grain"] == "snapshot-row"
    assert row["target_id"] == snapshot["dataset_id"]
    assert row["decision"] == "reject"
    assert row["reason"] == "regressed transparent-cup pickups at site-b"
    assert metadata["snapshot_name"] == "candidate-b"
    assert metadata["evaluation_run_id"] == "eval-b"
    assert metadata["metrics"][0]["slice"] == "site_id=site-b|object_category=cup"
    transform = _transform(lake, decision.transform_id)
    assert snapshot["transform_id"] in json.loads(transform["params"])[
        "prior_operation_transform_ids"
    ]


def test_curate_cli_feedback_loop_commands(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b",
        split_by="scenario",
    )
    metrics_path = tmp_path / "eval-metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": [
                    {
                        "metric": "success_rate",
                        "slice": "site_id=site-b|object_category=cup",
                        "score": 0.45,
                        "baseline_score": 0.70,
                        "scenario_ids": ["scn-site-b-cup"],
                    }
                ]
            }
        )
    )

    feedback_result = runner.invoke(
        app,
        [
            "curate",
            "feedback-from-eval",
            "--lake",
            str(lake_path),
            "--snapshot",
            "candidate-b",
            "--input",
            str(metrics_path),
            "--training-run",
            "train-b",
            "--model-version",
            "policy-v2",
            "--evaluation-run",
            "eval-b",
            "--json",
        ],
    )
    assert feedback_result.exit_code == 0, feedback_result.output
    feedback_payload = json.loads(feedback_result.output)
    assert feedback_payload["regression_count"] == 1

    regressions_path = tmp_path / "regressions.json"
    regressions_path.write_text(json.dumps(feedback_payload))
    candidates_result = runner.invoke(
        app,
        [
            "curate",
            "next-candidates",
            "--lake",
            str(lake_path),
            "--input",
            str(regressions_path),
            "--queue",
            "eval-b-regressions",
            "--limit",
            "2",
        ],
    )
    assert candidates_result.exit_code == 0, candidates_result.output
    assert "queue: eval-b-regressions" in candidates_result.output

    promote_result = runner.invoke(
        app,
        [
            "curate",
            "promote-snapshot",
            "--lake",
            str(lake_path),
            "--snapshot",
            "candidate-b",
            "--decision",
            "reject",
            "--reason",
            "regressed transparent-cup pickups at site-b",
            "--input",
            str(regressions_path),
        ],
    )
    assert promote_result.exit_code == 0, promote_result.output
    assert "decision: reject" in promote_result.output

    reopened = Lake.open(lake_path)
    assert reopened.table("model_outputs").count_rows() == 1
    assert reopened.table("curation_review_queues").count_rows() == 2
    snapshot = _snapshot_row(reopened, "candidate-b")
    decision = next(
        row
        for row in reopened.table("curation_memberships").to_arrow().to_pylist()
        if row["target_id"] == snapshot["dataset_id"]
    )
    assert decision["decision"] == "reject"


def test_curate_cli_compare_can_emit_json_report(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a",
        split_by="scenario",
    )
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b",
        split_by="scenario",
    )

    result = runner.invoke(
        app,
        [
            "curate",
            "compare",
            "--lake",
            str(lake_path),
            "--left",
            "candidate-a",
            "--right",
            "candidate-b",
            "--by",
            "site_id",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["operation"] == "compare-branches"
    assert payload["membership"]["added_scenario_ids"] == ["scn-site-b-cup"]
    assert payload["coverage"]["deltas"]["site_id=site-b"]["delta_count"] == 1
    assert payload["comparison_id"].startswith("cmp-")


def test_curate_cli_stratified_sample_creates_snapshot(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _build_curation_lake(lake_path)

    result = runner.invoke(
        app,
        [
            "curate",
            "stratified-sample",
            "--lake",
            str(lake_path),
            "--by",
            "site_id",
            "--by",
            "object_category",
            "--per-slice",
            "1",
            "--snapshot",
            "balanced-v1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "operation: stratified-sample" in result.output
    assert "snapshot: balanced-v1" in result.output
    assert "scenarios: 4" in result.output

    lake = Lake.open(lake_path)
    row = _snapshot_row(lake, "balanced-v1")
    spec = json.loads(row["query_spec"])
    assert spec["source"]["operation"] == "stratified-sample"


def test_curate_cli_dedup_plan_and_apply_commands(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _build_curation_lake(lake_path)

    plan_result = runner.invoke(
        app,
        [
            "curate",
            "dedup",
            "plan",
            "--lake",
            str(lake_path),
            "--threshold",
            "0.999",
        ],
    )
    assert plan_result.exit_code == 0, plan_result.output
    assert "operation: dedup-plan" in plan_result.output
    assert "dropped: 1" in plan_result.output

    apply_result = runner.invoke(
        app,
        [
            "curate",
            "dedup",
            "apply",
            "--lake",
            str(lake_path),
            "--threshold",
            "0.999",
            "--snapshot",
            "deduped-v1",
        ],
    )
    assert apply_result.exit_code == 0, apply_result.output
    assert "operation: dedup" in apply_result.output
    assert "snapshot: deduped-v1" in apply_result.output

    lake = Lake.open(lake_path)
    row = _snapshot_row(lake, "deduped-v1")
    spec = json.loads(row["query_spec"])
    assert "scn-duplicate" not in spec["scenario_ids"]


def test_curate_cli_distributed_dedup_plan_reports_job_and_recall(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _build_curation_lake(lake_path)

    result = runner.invoke(
        app,
        [
            "curate",
            "dedup",
            "distributed-plan",
            "--lake",
            str(lake_path),
            "--threshold",
            "0.999",
            "--shard-by",
            "run_id",
            "--job-id",
            "cli-dedup",
            "--recall-audit-sample-size",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "operation: distributed-dedup-plan" in result.output
    assert "job: cli-dedup" in result.output
    assert "recall_estimate:" in result.output


def test_curate_cli_diversity_sample_creates_snapshot(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _build_curation_lake(lake_path)

    result = runner.invoke(
        app,
        [
            "curate",
            "diversity",
            "sample",
            "--lake",
            str(lake_path),
            "--limit",
            "3",
            "--by",
            "site_id",
            "--by",
            "object_category",
            "--min-per-slice",
            "1",
            "--threshold",
            "0.999",
            "--snapshot",
            "diverse-cli-v1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "operation: diversity-sample" in result.output
    assert "snapshot: diverse-cli-v1" in result.output

    lake = Lake.open(lake_path)
    row = _snapshot_row(lake, "diverse-cli-v1")
    balance = json.loads(row["balance_report"])
    assert balance["operation"] == "diversity-sample"


def test_curate_cli_diversity_optimize_creates_constraint_snapshot(tmp_path):
    lake_path = tmp_path / "robot.lance"
    constraint_path = tmp_path / "constraints.json"
    _build_curation_lake(lake_path)
    constraint_path.write_text(
        json.dumps(
            {
                "minimum": {
                    "site_id": {"site-a": 1, "site-b": 1},
                    "object_category": {"cup": 2},
                },
                "maximum": {"object_category": {"cup": 2}},
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "curate",
            "diversity",
            "optimize",
            "--lake",
            str(lake_path),
            "--limit",
            "4",
            "--constraint",
            str(constraint_path),
            "--threshold",
            "0.999",
            "--snapshot",
            "optimized-cli-v1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "operation: diversity-optimization" in result.output
    assert "snapshot: optimized-cli-v1" in result.output

    lake = Lake.open(lake_path)
    row = _snapshot_row(lake, "optimized-cli-v1")
    coverage = json.loads(row["coverage_report"])
    assert coverage["operation"] == "diversity-optimization"
    assert coverage["constraint_spec"]["maximum"]["object_category=cup"] == 2


def test_curate_cli_review_queue_failure_and_export(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _build_curation_lake(lake_path)

    create_result = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "failure",
            "--lake",
            str(lake_path),
            "--queue",
            "cli-failure-review",
            "--seed-event",
            "evt-failure",
            "--limit",
            "2",
        ],
    )

    assert create_result.exit_code == 0, create_result.output
    assert "queue: cli-failure-review" in create_result.output
    assert "source: failure-mining" in create_result.output
    assert "items: 2" in create_result.output

    export_result = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "export",
            "--lake",
            str(lake_path),
            "--queue",
            "cli-failure-review",
            "--tool",
            "label-studio",
            "--page-limit",
            "1",
        ],
    )

    assert export_result.exit_code == 0, export_result.output
    manifest = json.loads(export_result.output)
    assert manifest["kind"] == "curation-review-queue-export"
    assert manifest["queue_name"] == "cli-failure-review"
    assert len(manifest["items"]) == 1
    assert manifest["total_item_count"] == 2
    assert manifest["page"]["has_more"] is True
    assert manifest["page"]["next_cursor"]
    assert manifest["queue_transform_id"] in manifest["source_transform_ids"]
    assert {row["table"] for row in manifest["source_table_versions"]} >= {
        "scenarios",
        "curation_review_queues",
    }

    summary_result = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "summary",
            "--lake",
            str(lake_path),
            "--queue",
            "cli-failure-review",
            "--json",
        ],
    )

    assert summary_result.exit_code == 0, summary_result.output
    summary = json.loads(summary_result.output)
    assert summary["item_count"] == 2
    assert summary["counts_by_status"] == {"open": 2}

    index_result = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "index-predicates",
            "--lake",
            str(lake_path),
            "--status-only",
            "--json",
        ],
    )

    assert index_result.exit_code == 0, index_result.output
    indexes = json.loads(index_result.output)
    assert {entry["column"] for entry in indexes} >= {"queue_id", "target_id", "status"}


def test_curate_cli_review_queue_connector_lifecycle(tmp_path):
    lake_path = tmp_path / "robot.lance"
    state_path = tmp_path / "label-studio-state.json"
    _build_curation_lake(lake_path)
    create_result = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "failure",
            "--lake",
            str(lake_path),
            "--queue",
            "cli-connector-review",
            "--seed-event",
            "evt-failure",
            "--limit",
            "2",
        ],
    )
    assert create_result.exit_code == 0, create_result.output

    dry_run = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "connector-export",
            "--lake",
            str(lake_path),
            "--queue",
            "cli-connector-review",
            "--tool",
            "label-studio",
            "--project",
            "project-a",
            "--dry-run",
            "--page-limit",
            "2",
        ],
    )

    assert dry_run.exit_code == 0, dry_run.output
    dry_payload = json.loads(dry_run.output)
    assert dry_payload["counts_by_status"] == {"queued": 2}
    assert len(dry_payload["report"]["idempotency_keys"]) == 2

    export = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "connector-export",
            "--lake",
            str(lake_path),
            "--queue",
            "cli-connector-review",
            "--tool",
            "json-file",
            "--project",
            "project-a",
            "--connector-state",
            str(state_path),
        ],
    )

    assert export.exit_code == 0, export.output
    export_payload = json.loads(export.output)
    assert export_payload["counts_by_status"] == {"exported": 2}
    state = json.loads(state_path.read_text())
    tasks = state["projects"]["project-a"]["tasks"]
    for index, stored in enumerate(tasks.values()):
        stored["status"] = "completed"
        stored["outcome"] = {
            "decision": "include" if index == 0 else "exclude",
            "reason_code": "cli-reviewed",
            "label_type": "failure_review" if index == 0 else "",
            "label": "true_positive" if index == 0 else "",
            "reviewer": "qa",
        }
    state_path.write_text(json.dumps(state, sort_keys=True))

    sync = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "sync-status",
            "--lake",
            str(lake_path),
            "--queue",
            "cli-connector-review",
            "--tool",
            "json-file",
            "--project",
            "project-a",
            "--connector-state",
            str(state_path),
        ],
    )

    assert sync.exit_code == 0, sync.output
    assert json.loads(sync.output)["counts_by_status"] == {"completed": 2}

    imported = runner.invoke(
        app,
        [
            "curate",
            "review-queue",
            "connector-import",
            "--lake",
            str(lake_path),
            "--queue",
            "cli-connector-review",
            "--tool",
            "json-file",
            "--project",
            "project-a",
            "--connector-state",
            str(state_path),
        ],
    )

    assert imported.exit_code == 0, imported.output
    import_payload = json.loads(imported.output)
    assert import_payload["outcome_count"] == 2
    assert import_payload["label_ids"]


def test_curate_cli_saves_decisions_and_composes_snapshot(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _build_curation_lake(lake_path)

    save_result = runner.invoke(
        app,
        [
            "curate",
            "save-view",
            "--lake",
            str(lake_path),
            "--view",
            "review-v1",
            "--scenario-id",
            "scn-anchor",
            "--scenario-id",
            "scn-neighbor",
        ],
    )
    assert save_result.exit_code == 0, save_result.output
    assert "view: review-v1" in save_result.output

    decide_result = runner.invoke(
        app,
        [
            "curate",
            "decide",
            "--lake",
            str(lake_path),
            "--view",
            "review-v1",
            "--decision",
            "exclude",
            "--scenario-id",
            "scn-neighbor",
            "--reason",
            "hold out of training",
        ],
    )
    assert decide_result.exit_code == 0, decide_result.output
    assert "decision: exclude" in decide_result.output

    compose_result = runner.invoke(
        app,
        [
            "curate",
            "compose-snapshot",
            "--lake",
            str(lake_path),
            "--view",
            "review-v1",
            "--snapshot",
            "reviewed-training",
        ],
    )
    assert compose_result.exit_code == 0, compose_result.output
    assert "operation: apply-decisions" in compose_result.output
    assert "scenarios: 1" in compose_result.output

    lake = Lake.open(lake_path)
    row = _snapshot_row(lake, "reviewed-training")
    spec = json.loads(row["query_spec"])
    assert spec["scenario_ids"] == ["scn-anchor"]

    history_result = runner.invoke(
        app,
        [
            "curate",
            "membership-history",
            "--lake",
            str(lake_path),
            "--view",
            "review-v1",
            "--snapshot",
            "reviewed-training",
            "--scenario-id",
            "scn-neighbor",
            "--superseded-policy",
            "history",
            "--json",
        ],
    )
    assert history_result.exit_code == 0, history_result.output
    history = json.loads(history_result.output)
    assert history["latest_decisions"][0]["decision"] == "exclude"
    assert history["membership_history"][0]["reason"] == "hold out of training"

    trace_result = runner.invoke(
        app,
        [
            "curate",
            "trace-membership",
            "--lake",
            str(lake_path),
            "--snapshot",
            "reviewed-training",
            "--scenario-id",
            "scn-neighbor",
            "--json",
        ],
    )
    assert trace_result.exit_code == 0, trace_result.output
    trace = json.loads(trace_result.output)
    assert trace["final_result"] == "exclude"
    assert trace["included_in_snapshot"] is False
    assert trace["membership_history"][0]["decision"] == "exclude"


def test_compile_row_plan_observation_exclude_is_row_precise(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    selection = lake.curate.workbench(
        scope=CurationScope(scenario_ids=("scn-anchor", "scn-neighbor"))
    )
    selection.save_view("row-review")
    selection.record_decisions(
        view_name="row-review",
        decision="exclude",
        target_grain="observation",
        target_ids=["obs-scn-neighbor"],
        reason="bad camera exposure",
    )

    plan = lake.curate.compile_row_plan(view_name="row-review", target_grain="observation")

    assert plan.target_ids == ("obs-scn-anchor",)
    assert plan.report["rejected"]["obs-scn-neighbor"]["reason"] == "row-exclude"
    assert plan.report["copied_payload_bytes"] == 0
    assert {row["table"] for row in plan.report["source_table_versions"]} >= {
        "observations",
        "curation_memberships",
    }

    # Backward compatibility: row-grain decisions do not change scenario snapshots.
    scenario_selection = selection.apply_decisions(view_name="row-review")
    assert scenario_selection.scenario_ids == ("scn-anchor", "scn-neighbor")


def test_compile_row_plan_aligned_frame_include_is_exact_and_freezable(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.table("aligned_frames").add(
        pa.Table.from_pylist(
            [
                _aligned_frame(
                    "af-anchor",
                    run_id="run-a",
                    timestamp_ns=0,
                    observation_id="obs-scn-anchor",
                    tick_index=0,
                ),
                _aligned_frame(
                    "af-neighbor",
                    run_id="run-a",
                    timestamp_ns=20,
                    observation_id="obs-scn-neighbor",
                    tick_index=1,
                ),
            ],
            schema=ALIGNED_FRAMES_SCHEMA,
        )
    )
    selection = lake.curate.workbench(
        scope=CurationScope(scenario_ids=("scn-anchor", "scn-neighbor"))
    )
    selection.save_view("aligned-review")
    selection.record_decisions(
        view_name="aligned-review",
        decision="include",
        target_grain="aligned-frame",
        target_ids=["af-anchor"],
        reason="single good policy tick",
    )

    plan = lake.curate.compile_row_plan(
        view_name="aligned-review",
        target_grain="aligned-frame",
        freeze=True,
    )

    assert plan.target_ids == ("af-anchor",)
    assert plan.lance_row_ids[0] is not None
    assert plan.frozen is True
    assert plan.artifact_id == f"lancedb-robotics:curation-row-plan:{plan.plan_id}"
    artifact = next(
        row
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
        if row["artifact_id"] == plan.artifact_id
    )
    assert artifact["kind"] == "curation-row-plan"
    assert artifact["row_grain"] == "aligned-frame"
    assert artifact["row_ids"] == ["af-anchor"]
    artifact_metadata = {item["key"]: item["value"] for item in artifact["metadata"]}
    assert artifact_metadata["copied_payload_bytes"] == "0"


def test_compile_row_plan_reports_scenario_exclude_row_include_conflict(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-anchor",)))
    selection.save_view("conflict-review")
    selection.record_decisions(
        view_name="conflict-review",
        decision="exclude",
        target_grain="scenario",
        scenario_ids=["scn-anchor"],
        reason="exclude full scenario",
    )
    time.sleep(0.001)
    selection.record_decisions(
        view_name="conflict-review",
        decision="include",
        target_grain="observation",
        target_ids=["obs-scn-anchor"],
        reason="override one clean frame",
    )

    plan = lake.curate.compile_row_plan(
        view_name="conflict-review",
        target_grain="observation",
    )

    assert plan.target_ids == ("obs-scn-anchor",)
    assert plan.report["base_policy"] == "row-include-decisions"
    assert plan.report["conflicts"][0]["scenario_decision"] == "exclude"
    assert plan.report["conflicts"][0]["row_decision"] == "include"
    assert plan.report["conflicts"][0]["resolution"] == "row-include"


def test_curate_cli_compile_row_plan_json(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-anchor",)))
    selection.save_view("cli-row-review")
    selection.record_decisions(
        view_name="cli-row-review",
        decision="label",
        target_grain="observation",
        target_ids=["obs-scn-anchor"],
        reason="send to fine label pass",
        queue="fine-labels",
    )

    result = runner.invoke(
        app,
        [
            "curate",
            "compile-row-plan",
            "--lake",
            str(lake_path),
            "--view",
            "cli-row-review",
            "--target-grain",
            "observation",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["target_ids"] == ["obs-scn-anchor"]
    assert payload["report"]["label_intents"][0]["decision"] == "label"
    assert payload["report"]["label_intents"][0]["queue"] == "fine-labels"


# ---------------------------------------------------------------------------
# Backlog 0093: comparison report catalog, retention, and reload.
# ---------------------------------------------------------------------------


def _seed_comparison(
    lake: Lake,
    *,
    comparison_id: str,
    left_name: str,
    right_name: str,
    created_at: datetime,
    left_dataset_id: str = "ds-left",
    right_dataset_id: str = "ds-right",
    metrics=("membership",),
    state: str = "active",
    transform_id: str | None = None,
    report: dict | None = None,
) -> dict:
    """Write a comparison catalog row with a controlled created_at/state."""
    transform_id = transform_id or f"tfm-{comparison_id}"
    report = report if report is not None else {
        "operation": "compare-branches",
        "comparison_id": comparison_id,
        "left": left_name,
        "right": right_name,
        "metrics": list(metrics),
        "shared_count": 0,
        "left_only": [],
        "right_only": [],
        "transform_id": transform_id,
    }
    report_json = json.dumps(report, sort_keys=True)
    row = {
        "comparison_id": comparison_id,
        "pair_alias": f"{left_name}..{right_name}",
        "state": state,
        "left_dataset_id": left_dataset_id,
        "right_dataset_id": right_dataset_id,
        "left_snapshot_name": left_name,
        "right_snapshot_name": right_name,
        "metrics": list(metrics),
        "dimensions": [],
        "added_scenario_count": 0,
        "removed_scenario_count": 0,
        "shared_scenario_count": 0,
        "report_json": report_json,
        "report_sha1": hashlib.sha1(report_json.encode()).hexdigest(),
        "report_bytes": len(report_json.encode()),
        "retention_policy_json": "",
        "archived_at": None,
        "pruned_at": None,
        "table_versions": [{"table": "scenarios", "version": 1, "tag": ""}],
        "created_by": "lancedb-robotics",
        "transform_id": transform_id,
        "created_at": created_at,
    }
    lake.table("curation_comparisons").add(
        pa.Table.from_pylist([row], schema=CURATION_COMPARISONS_SCHEMA)
    )
    return row


def _comparison_rows(lake: Lake) -> dict[str, dict]:
    return {
        row["comparison_id"]: row
        for row in lake.table("curation_comparisons").to_arrow().to_pylist()
    }


def test_compare_persists_active_catalog_row_with_digest(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a", split_by="scenario"
    )
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b", split_by="scenario"
    )

    comparison = lake.curate.compare("candidate-a", "candidate-b")

    (entry,) = lake.curate.list_comparisons(left="candidate-a", right="candidate-b")
    assert entry.comparison_id == comparison.comparison_id
    assert entry.state == "active"
    assert entry.pair_alias == "candidate-a..candidate-b"
    assert entry.report_available is True
    assert entry.report_bytes > 0
    assert entry.report_sha1
    assert entry.table_versions  # snapshot evidence captured


def test_list_comparisons_filters_by_pair_metric_and_orders_newest_first(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    base = datetime(2026, 6, 1, tzinfo=UTC)
    _seed_comparison(
        lake, comparison_id="cmp-ab-1", left_name="snap-a", right_name="snap-b",
        created_at=base, metrics=("membership",),
    )
    _seed_comparison(
        lake, comparison_id="cmp-ab-2", left_name="snap-a", right_name="snap-b",
        created_at=base + timedelta(days=2), metrics=("coverage",),
    )
    _seed_comparison(
        lake, comparison_id="cmp-ac-1", left_name="snap-a", right_name="snap-c",
        created_at=base + timedelta(days=1), metrics=("membership",),
    )

    pair = lake.curate.list_comparisons(left="snap-a", right="snap-b")
    assert [entry.comparison_id for entry in pair] == ["cmp-ab-2", "cmp-ab-1"]

    either_side = lake.curate.list_comparisons(snapshot="snap-c")
    assert [entry.comparison_id for entry in either_side] == ["cmp-ac-1"]

    by_metric = lake.curate.list_comparisons(metric="coverage")
    assert [entry.comparison_id for entry in by_metric] == ["cmp-ab-2"]

    windowed = lake.curate.list_comparisons(since=base + timedelta(days=1, hours=1))
    assert [entry.comparison_id for entry in windowed] == ["cmp-ab-2"]


def test_comparison_reload_by_id_and_alias_returns_persisted_report(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a", split_by="scenario"
    )
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b", split_by="scenario"
    )
    first = lake.curate.compare("candidate-a", "candidate-b", metrics=["membership"])
    latest = lake.curate.compare(
        "candidate-a", "candidate-b", metrics=["coverage"], by=["site_id"]
    )

    reloaded = lake.curate.comparison(first.comparison_id)
    assert reloaded.comparison_id == first.comparison_id
    assert reloaded.report == first.report  # same JSON shape as curate compare --json

    by_alias = lake.curate.comparison("candidate-a..candidate-b")
    assert by_alias.comparison_id == latest.comparison_id

    by_name = lake.curate.comparison("candidate-a")
    assert by_name.comparison_id == latest.comparison_id

    with pytest.raises(CurationError):
        lake.curate.comparison("cmp-does-not-exist")


def test_prune_comparisons_archives_then_prunes_and_preserves_lineage(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    now = datetime.now(UTC)
    transforms_before = lake.table("transform_runs").count_rows()
    old = _seed_comparison(
        lake, comparison_id="cmp-old", left_name="snap-a", right_name="snap-b",
        created_at=now - timedelta(days=40),
    )
    _seed_comparison(
        lake, comparison_id="cmp-mid", left_name="snap-a", right_name="snap-b",
        created_at=now - timedelta(days=20),
    )
    _seed_comparison(
        lake, comparison_id="cmp-new", left_name="snap-a", right_name="snap-b",
        created_at=now - timedelta(days=1),
    )
    _seed_comparison(
        lake, comparison_id="cmp-other", left_name="snap-a", right_name="snap-c",
        created_at=now - timedelta(days=99),
    )

    report = lake.curate.prune_comparisons(
        retain_latest=1, older_than=timedelta(days=30)
    )

    assert report.pruned_comparison_ids == ("cmp-old",)
    assert report.archived_comparison_ids == ("cmp-mid",)
    assert set(report.retained_comparison_ids) == {"cmp-new", "cmp-other"}
    assert report.body_bytes_after == report.body_bytes_before - old["report_bytes"]
    assert report.transform_id

    rows = _comparison_rows(lake)
    # Pruned: body cleared, but audit metadata survives.
    assert rows["cmp-old"]["state"] == "pruned"
    assert rows["cmp-old"]["report_json"] == ""
    assert rows["cmp-old"]["report_sha1"] == old["report_sha1"]
    assert rows["cmp-old"]["left_dataset_id"] == old["left_dataset_id"]
    assert rows["cmp-old"]["transform_id"] == old["transform_id"]
    assert rows["cmp-old"]["table_versions"]
    assert rows["cmp-old"]["pruned_at"] is not None
    # Archived: superseded but body retained and reloadable.
    assert rows["cmp-mid"]["state"] == "archived"
    assert rows["cmp-mid"]["report_json"]
    assert rows["cmp-mid"]["archived_at"] is not None
    # Newest per pair stays active.
    assert rows["cmp-new"]["state"] == "active"
    assert rows["cmp-other"]["state"] == "active"

    # Retention recorded exactly one audit transform; no lineage rows deleted.
    transforms_after = lake.table("transform_runs").count_rows()
    assert transforms_after == transforms_before + 1
    audit = _transform(lake, report.transform_id)
    assert audit["kind"] == "curation-comparison-retention"
    assert "curation_comparisons" in audit["output_tables"]

    # Reload semantics after retention.
    assert lake.curate.comparison("cmp-mid").comparison_id == "cmp-mid"
    with pytest.raises(CurationError):
        lake.curate.comparison("cmp-old")


def test_prune_comparisons_dry_run_does_not_mutate(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    now = datetime.now(UTC)
    _seed_comparison(
        lake, comparison_id="cmp-old", left_name="snap-a", right_name="snap-b",
        created_at=now - timedelta(days=40),
    )
    _seed_comparison(
        lake, comparison_id="cmp-new", left_name="snap-a", right_name="snap-b",
        created_at=now - timedelta(days=1),
    )

    report = lake.curate.prune_comparisons(
        retain_latest=1, older_than=timedelta(days=30), dry_run=True
    )

    assert report.dry_run is True
    assert report.pruned_comparison_ids == ("cmp-old",)
    assert report.transform_id == ""
    rows = _comparison_rows(lake)
    assert rows["cmp-old"]["state"] == "active"
    assert rows["cmp-old"]["report_json"]


def test_curate_cli_comparisons_list_show_and_prune(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _build_curation_lake(lake_path)
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a", split_by="scenario"
    )
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b", split_by="scenario"
    )
    comparison = lake.curate.compare("candidate-a", "candidate-b")

    listed = runner.invoke(
        app,
        ["curate", "comparisons", "--lake", str(lake_path), "--json"],
    )
    assert listed.exit_code == 0, listed.output
    payload = json.loads(listed.output)
    assert payload[0]["comparison_id"] == comparison.comparison_id
    assert payload[0]["state"] == "active"

    shown = runner.invoke(
        app,
        ["curate", "comparison", comparison.comparison_id, "--lake", str(lake_path), "--json"],
    )
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output)["comparison_id"] == comparison.comparison_id

    pruned = runner.invoke(
        app,
        ["curate", "prune-comparisons", "--lake", str(lake_path), "--retain-latest", "1"],
    )
    assert pruned.exit_code == 0, pruned.output
    assert "archived: 0" in pruned.output
    assert "pruned: 0" in pruned.output

    # CLI text listing distinguishes lifecycle state.
    text_list = runner.invoke(app, ["curate", "comparisons", "--lake", str(lake_path)])
    assert text_list.exit_code == 0, text_list.output
    assert "state=active" in text_list.output


# ---------------------------------------------------------------------------
# Backlog 0094: scalable comparison execution + metric plugins.
# ---------------------------------------------------------------------------

from lancedb_robotics.comparison_plugins import (  # noqa: E402
    ComparisonMetricPlugin,
    clear_comparison_plugins,
    register_comparison_plugin,
)


def _build_scale_curation_lake(
    path,
    *,
    total: int = 500,
    left_slice: tuple[int, int] = (0, 300),
    right_slice: tuple[int, int] = (200, 500),
) -> tuple[Lake, list[str], list[str]]:
    """Seed a lake with many scenarios + two overlapping snapshot branches."""
    lake = Lake.init(path)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-a",
                    "run_kind": "teleop",
                    "raw_uri": "memory://run-a",
                    "robot_id": "arm-a",
                    "site_id": "site-a",
                    "task_id": "pick",
                    "start_time_ns": 0,
                    "end_time_ns": 10_000,
                    "duration_ns": 10_000,
                    "quality_flags": [],
                    "created_at": NOW,
                },
                {
                    "run_id": "run-b",
                    "run_kind": "teleop",
                    "raw_uri": "memory://run-b",
                    "robot_id": "arm-b",
                    "site_id": "site-b",
                    "task_id": "pick",
                    "start_time_ns": 0,
                    "end_time_ns": 10_000,
                    "duration_ns": 10_000,
                    "quality_flags": ["blurry"],
                    "created_at": NOW,
                },
            ],
            schema=RUNS_SCHEMA,
        )
    )
    scenarios = lake.table("scenarios")
    scenarios.add_columns(pa.schema([pa.field("embedding", pa.list_(pa.float32(), 4))]))
    ids = [f"scn-{index:04d}" for index in range(total)]
    scenario_rows = [
        _scenario(
            scenario_id,
            run_id="run-a" if index % 2 == 0 else "run-b",
            start_time_ns=index * 10,
            object_category="cup" if index % 3 == 0 else "box",
            embedding=[1.0, float(index % 5), 0.0, 0.0],
        )
        for index, scenario_id in enumerate(ids)
    ]
    scenarios.add(pa.Table.from_pylist(scenario_rows, schema=scenarios.schema))
    # Label every other scenario (scenario-keyed) for label-completeness coverage.
    label_rows = [
        {
            "label_id": f"lbl-{scenario_id}",
            "run_id": scenario_rows[index]["run_id"],
            "observation_id": "",
            "scenario_id": scenario_id,
            "event_id": "",
            "label_type": "object",
            "label": "bolt",
            "label_value": "{}",
            "label_spec": "fixture",
            "source": "human",
            "reviewer": "qa",
            "confidence": 1.0,
            "status": "accepted",
            "metadata": [],
            "transform_id": "tfm-label",
            "created_at": NOW,
        }
        for index, scenario_id in enumerate(ids)
        if index % 2 == 0
    ]
    lake.table("labels").add(pa.Table.from_pylist(label_rows, schema=LABELS_SCHEMA))
    left_ids = ids[left_slice[0] : left_slice[1]]
    right_ids = ids[right_slice[0] : right_slice[1]]
    lake.curate.workbench(scope=left_ids).snapshot(name="branch-left", split_by="scenario")
    lake.curate.workbench(scope=right_ids).snapshot(name="branch-right", split_by="scenario")
    return lake, left_ids, right_ids


def test_compare_large_snapshots_uses_bounded_streaming_memory(tmp_path):
    # AC: comparing two large snapshots uses bounded memory and avoids full
    # Python row materialization for supported metrics.
    lake, left_ids, right_ids = _build_scale_curation_lake(tmp_path / "robot.lance")
    batch_size = 64

    comparison = lake.curate.compare(
        "branch-left",
        "branch-right",
        by=["site_id"],
        metrics=["membership", "coverage", "quality", "label-completeness"],
        batch_size=batch_size,
    )
    execution = comparison.report["execution"]

    # Peak Python row residency never exceeds the configured batch ceiling, even
    # though far more rows are scanned in total.
    assert execution["peak_batch_rows"] <= batch_size
    assert execution["total_scanned_rows"] >= len(set(left_ids) | set(right_ids))
    # No supported-metric table fell back to a full materialized scan.
    assert execution["materialized_tables"] == []
    assert "scenarios" in execution["streamed_tables"]
    assert execution["bounded"] is True
    # Counts and deltas stay complete regardless of streaming.
    assert comparison.report["left_count"] == len(left_ids)
    assert comparison.report["right_count"] == len(right_ids)
    assert comparison.report["shared_count"] == len(set(left_ids) & set(right_ids))


def test_plan_comparison_changes_with_metric_selection_and_flags_executor(tmp_path):
    # AC: metric planning reports required tables, versions, estimated rows, and
    # whether each metric runs locally or needs an external executor.
    lake, _left, _right = _build_scale_curation_lake(tmp_path / "robot.lance")

    membership_only = lake.curate.plan_comparison(
        "branch-left", "branch-right", metrics=["membership"]
    )
    assert membership_only.metrics == ("membership",)
    (membership_entry,) = membership_only.entries
    assert membership_entry.execution == "local"
    assert membership_entry.required_tables == ()

    richer = lake.curate.plan_comparison(
        "branch-left",
        "branch-right",
        by=["site_id"],
        metrics=["coverage", "label-completeness"],
    )
    # Metric plan output changes when optional metrics are selected.
    assert {entry.metric for entry in richer.entries} == {"coverage", "label-completeness"}
    coverage_entry = next(entry for entry in richer.entries if entry.metric == "coverage")
    assert "scenarios" in coverage_entry.required_tables
    assert coverage_entry.estimated_rows > 0
    assert coverage_entry.table_versions  # current source versions captured
    assert richer.estimated_scan_rows >= coverage_entry.estimated_rows
    assert richer.requires_external_executor is False

    # A tiny local budget pushes scenario-scale metrics to an external executor.
    budgeted = lake.curate.plan_comparison(
        "branch-left", "branch-right", metrics=["coverage"], local_row_budget=10
    )
    coverage_budgeted = next(entry for entry in budgeted.entries if entry.metric == "coverage")
    assert coverage_budgeted.execution == "external"
    assert budgeted.requires_external_executor is True


def test_compare_membership_preview_is_capped_with_paging_handles(tmp_path):
    # AC: long added/removed/shared id lists are emitted through a bounded
    # preview plus deterministic reload/paging handles.
    lake, left_ids, right_ids = _build_scale_curation_lake(tmp_path / "robot.lance")
    expected_added = sorted(set(right_ids) - set(left_ids))
    expected_removed = sorted(set(left_ids) - set(right_ids))

    comparison = lake.curate.compare(
        "branch-left", "branch-right", metrics=["membership"], preview_limit=100
    )
    preview = comparison.report["membership_preview"]
    # Counts are complete; previews are capped.
    assert preview["added"]["count"] == len(expected_added)
    assert preview["removed"]["count"] == len(expected_removed)
    assert len(preview["added"]["preview"]) == 100
    assert preview["added"]["truncated"] is True
    assert preview["added"]["next_page_token"] == "added:100"
    # Persisted report body does not inline the full id lists.
    assert len(comparison.report["right_only"]) == 100
    assert comparison.report["membership"]["added_scenario_ids_truncated"] is True

    # Deterministic paging reloads the full sorted list across pages.
    page = lake.curate.comparison_membership(
        comparison.comparison_id, field="added", offset=0, limit=100
    )
    assert page.total == len(expected_added)
    assert list(page.scenario_ids) == expected_added[:100]
    assert page.next_page_token == "added:100"
    next_page = lake.curate.comparison_membership(
        comparison.comparison_id, page_token=page.next_page_token
    )
    assert list(next_page.scenario_ids) == expected_added[100:200]
    # Pages reconstruct exactly the membership diff.
    assert list(page.scenario_ids) + list(next_page.scenario_ids) == expected_added


def test_comparison_metric_plugin_adds_section_and_lineage(tmp_path):
    # AC: a custom metric plugin can add a report section and transform lineage
    # without modifying curate.py.
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a", split_by="scenario"
    )
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b", split_by="scenario"
    )

    class _BenchmarkPlugin(ComparisonMetricPlugin):
        name = "benchmark-outcome"

        def required_tables(self, ctx):
            return ("model_outputs",)

        def estimate_rows(self, ctx):
            return 7

        def compute(self, ctx):
            # Plugins can use bounded streaming + see snapshot evidence.
            scanned = sum(len(batch) for batch in ctx.stream("model_outputs"))
            return {
                "left_dataset_id": ctx.left_dataset_id,
                "right_dataset_id": ctx.right_dataset_id,
                "added_count": ctx.membership["added_count"],
                "model_output_rows_seen": scanned,
                "delta_pass_rate": 0.25,
                "transform_id": "tfm-benchmark-ext",
            }

        def lineage_transform_ids(self, section):
            return (section["transform_id"],)

    plugin = _BenchmarkPlugin()
    comparison = lake.curate.compare(
        "candidate-a", "candidate-b", metrics=["membership"], plugins=[plugin]
    )

    section = comparison.report["plugins"]["benchmark-outcome"]
    assert section["delta_pass_rate"] == 0.25
    assert section["added_count"] == comparison.report["membership"]["added_count"]
    assert comparison.report["plugin_metrics"] == ["benchmark-outcome"]
    # The plugin's transform id threads into comparison lineage.
    params = _transform_params(_transform(lake, comparison.transform_id))
    assert "tfm-benchmark-ext" in params["prior_operation_transform_ids"]

    # The plugin shows up in the plan, scoped to its declared tables/estimate.
    plan = lake.curate.plan_comparison(
        "candidate-a", "candidate-b", metrics=["membership"], plugins=[plugin]
    )
    plugin_entry = next(entry for entry in plan.entries if entry.metric == "benchmark-outcome")
    assert plugin_entry.kind == "plugin"
    assert plugin_entry.required_tables == ("model_outputs",)
    assert plugin_entry.estimated_rows == 7
    assert plan.plugin_metrics == ("benchmark-outcome",)

    # Plugins can also be selected by registered name.
    clear_comparison_plugins()
    try:
        register_comparison_plugin(plugin)
        by_name = lake.curate.compare(
            "candidate-a", "candidate-b", metrics=["membership"], plugins=["benchmark-outcome"]
        )
        assert "benchmark-outcome" in by_name.report["plugins"]
    finally:
        clear_comparison_plugins()


def test_comparison_staleness_fires_after_source_table_advances(tmp_path):
    # AC: persisted reports can be marked stale when source table versions advance.
    lake = _build_curation_lake(tmp_path / "robot.lance")
    lake.curate.workbench(scope=["scn-anchor", "scn-neighbor"]).snapshot(
        name="candidate-a", split_by="scenario"
    )
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b", split_by="scenario"
    )
    comparison = lake.curate.compare(
        "candidate-a", "candidate-b", metrics=["label-completeness"]
    )

    fresh = lake.curate.comparison_staleness(comparison.comparison_id)
    assert fresh.stale is False
    assert fresh.advanced_tables == ()

    # Append a relevant source row, advancing the labels table version.
    lake.table("labels").add(
        pa.Table.from_pylist(
            [
                {
                    "label_id": "lbl-new",
                    "run_id": "run-a",
                    "observation_id": "",
                    "scenario_id": "scn-anchor",
                    "event_id": "",
                    "label_type": "object",
                    "label": "bolt",
                    "label_value": "{}",
                    "label_spec": "fixture",
                    "source": "human",
                    "reviewer": "qa",
                    "confidence": 1.0,
                    "status": "accepted",
                    "metadata": [],
                    "transform_id": "tfm-label",
                    "created_at": NOW,
                }
            ],
            schema=LABELS_SCHEMA,
        )
    )

    stale = lake.curate.comparison_staleness(comparison.comparison_id)
    assert stale.stale is True
    advanced_tables = {item["table"] for item in stale.advanced_tables}
    assert "labels" in advanced_tables


def test_curate_cli_compare_plan_members_and_staleness(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake, left_ids, right_ids = _build_scale_curation_lake(lake_path)

    plan = runner.invoke(
        app,
        [
            "curate", "compare-plan", "--lake", str(lake_path),
            "--left", "branch-left", "--right", "branch-right",
            "--metric", "coverage", "--by", "site_id", "--json",
        ],
    )
    assert plan.exit_code == 0, plan.output
    plan_payload = json.loads(plan.output)
    assert plan_payload["operation"] == "compare-branches-plan"
    assert any(entry["metric"] == "coverage" for entry in plan_payload["entries"])

    comparison = lake.curate.compare("branch-left", "branch-right", metrics=["membership"])

    members = runner.invoke(
        app,
        [
            "curate", "comparison-members", comparison.comparison_id,
            "--lake", str(lake_path), "--field", "added", "--limit", "50", "--json",
        ],
    )
    assert members.exit_code == 0, members.output
    members_payload = json.loads(members.output)
    assert members_payload["total"] == len(set(right_ids) - set(left_ids))
    assert len(members_payload["scenario_ids"]) == 50
    assert members_payload["next_page_token"] == "added:50"

    staleness = runner.invoke(
        app,
        [
            "curate", "comparison-staleness", comparison.comparison_id,
            "--lake", str(lake_path), "--json",
        ],
    )
    assert staleness.exit_code == 0, staleness.output
    assert json.loads(staleness.output)["stale"] is False
