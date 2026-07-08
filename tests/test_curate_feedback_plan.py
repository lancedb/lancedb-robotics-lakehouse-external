"""Scalable feedback candidate generation (backlog 0096).

The 0059 feedback loop turned eval regressions into review queues with an
unconditional in-memory all-pairs cosine scan and artifact identities that
churned with table versions. 0096 adds a replayable ``FeedbackCandidatePlan``
(explain), routed candidate search that rides a persistent LanceDB vector
index when one exists, idempotent plan-keyed artifact writes, bounded
deterministic candidate previews, and a dry-run apply mode. Fixtures follow
the ``test_curate.py`` / ``test_search_routing.py`` patterns: tiny lakes with
deterministic unit-vector embeddings, and a 300-row lake (above
``MIN_INDEX_ROWS``) when a real ANN index is needed.
"""

import hashlib
import json
import math
from datetime import UTC, datetime

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from lancedb_robotics import curate as curate_mod
from lancedb_robotics.cli import app
from lancedb_robotics.curate import CurationError, FeedbackCandidatePlan
from lancedb_robotics.indexing import build_vector_index, has_vector_index
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import RUNS_SCHEMA

runner = CliRunner()

NOW = datetime(2026, 7, 2, tzinfo=UTC)

_DIM = 8


def _unit(vector):
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _seeded_vector(seed: str, dimension: int = _DIM) -> list[float]:
    """Deterministic distinct unit vector per seed (test_search_routing pattern)."""
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        for offset in range(0, len(digest), 4):
            if len(values) >= dimension:
                break
            word = int.from_bytes(digest[offset : offset + 4], "big")
            values.append(word / 0xFFFFFFFF * 2.0 - 1.0)
        counter += 1
    return _unit(values)


def _run_row(run_id: str, *, site_id: str) -> dict:
    return {
        "run_id": run_id,
        "run_kind": "teleop",
        "raw_uri": f"memory://{run_id}",
        "robot_id": "arm-a",
        "site_id": site_id,
        "task_id": "pick",
        "start_time_ns": 0,
        "end_time_ns": 1_000_000,
        "duration_ns": 1_000_000,
        "quality_flags": [],
        "created_at": NOW,
    }


def _scenario(
    scenario_id: str,
    *,
    run_id: str,
    start_time_ns: int,
    object_category: str,
    embedding: list[float],
) -> dict:
    return {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "start_time_ns": start_time_ns,
        "end_time_ns": start_time_ns + 10,
        "window_ns": 10,
        "is_partial": False,
        "topics": ["/camera/front"],
        "observation_ids": [f"obs-{scenario_id}"],
        "observation_count": 1,
        "scenario_type": "episode",
        "source": "fixture",
        "coverage_tags": [f"object_category:{object_category}", "quality_score:0.95"],
        "summary": scenario_id,
        "transform_id": "tfm-fixture",
        "created_at": NOW,
        "embedding": _unit(embedding),
    }


def _small_lake(path) -> Lake:
    """Six scenarios across two runs; no vector index (exact fallback territory)."""
    lake = Lake.init(path)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [_run_row("run-a", site_id="site-a"), _run_row("run-b", site_id="site-b")],
            schema=RUNS_SCHEMA,
        )
    )
    scenarios = lake.table("scenarios")
    scenarios.add_columns(pa.schema([pa.field("embedding", pa.list_(pa.float32(), _DIM))]))
    rows = [
        _scenario(
            "scn-anchor",
            run_id="run-a",
            start_time_ns=0,
            object_category="cup",
            embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        _scenario(
            "scn-duplicate",
            run_id="run-a",
            start_time_ns=10,
            object_category="cup",
            embedding=[0.99995, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        _scenario(
            "scn-neighbor",
            run_id="run-a",
            start_time_ns=20,
            object_category="box",
            embedding=[0.97, 0.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        _scenario(
            "scn-site-b-cup",
            run_id="run-b",
            start_time_ns=0,
            object_category="cup",
            embedding=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        _scenario(
            "scn-site-b-box",
            run_id="run-b",
            start_time_ns=10,
            object_category="box",
            embedding=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        _scenario(
            "scn-site-b-box-extra",
            run_id="run-b",
            start_time_ns=20,
            object_category="box",
            embedding=[0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ]
    scenarios.add(pa.Table.from_pylist(rows, schema=scenarios.schema))
    return lake


def _indexed_lake(path, *, n: int = 300) -> Lake:
    """A lake above ``MIN_INDEX_ROWS`` with a real persistent ANN index."""
    lake = Lake.init(path)
    lake.table("runs").add(
        pa.Table.from_pylist([_run_row("run-synthetic", site_id="site-a")], schema=RUNS_SCHEMA)
    )
    scenarios = lake.table("scenarios")
    scenarios.add_columns(pa.schema([pa.field("embedding", pa.list_(pa.float32(), _DIM))]))
    rows = [
        _scenario(
            f"scn-{index:04d}",
            run_id="run-synthetic",
            start_time_ns=index * 1000,
            object_category="cup",
            embedding=_seeded_vector(f"scn-{index:04d}"),
        )
        for index in range(n)
    ]
    scenarios.add(pa.Table.from_pylist(rows, schema=scenarios.schema))
    result = build_vector_index(lake, table="scenarios", column="embedding")
    assert result.status == "built", result
    assert has_vector_index(lake.table("scenarios"), "embedding")
    return lake


def _regression(scenario_id: str = "scn-anchor") -> dict:
    return {
        "metric": "success_rate",
        "slice": "site_id=site-a|object_category=cup",
        "score": 0.45,
        "baseline_score": 0.70,
        "regressed": True,
        "scenario_ids": [scenario_id],
    }


def _feedback_for_small_lake(lake: Lake):
    lake.curate.workbench(scope=["scn-anchor", "scn-site-b-cup"]).snapshot(
        name="candidate-b",
        split_by="scenario",
    )
    return lake.curate.feedback_from_eval(
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


# --- 1. indexed candidate search avoids the full Python all-pairs scan ---


def test_indexed_route_avoids_exact_all_pairs_scan(tmp_path, monkeypatch):
    lake = _indexed_lake(tmp_path / "robot.lance")
    regression = _regression("scn-0001")

    exact_calls: list[str] = []
    original = curate_mod._nearest_scenarios

    def spy(*args, **kwargs):
        exact_calls.append(kwargs.get("seed_scenario", ""))
        return original(*args, **kwargs)

    monkeypatch.setattr(curate_mod, "_nearest_scenarios", spy)

    indexed = lake.curate.next_candidates(
        from_regressions=regression,
        queue_name="q-indexed",
        limit_per_regression=5,
    )
    assert exact_calls == [], "route=auto with an index must not run the exact all-pairs scan"
    assert indexed.plan is not None
    assert indexed.plan.effective_route == "ann"
    assert "scn-0001" in indexed.selection.scenario_ids
    assert len(indexed.selection.scenario_ids) == 5

    exact = lake.curate.next_candidates(
        from_regressions=regression,
        queue_name="q-exact",
        limit_per_regression=5,
        route="exact",
    )
    assert exact_calls, "route=exact must use the exact scan helper"
    assert exact.plan.effective_route == "exact"
    # IVF_FLAT with nprobes >= partitions is exhaustive, so both routes agree.
    assert set(indexed.selection.scenario_ids) == set(exact.selection.scenario_ids)


def test_plan_records_met_index_requirement_on_indexed_lake(tmp_path):
    lake = _indexed_lake(tmp_path / "robot.lance")
    plan = lake.curate.plan_next_candidates(
        from_regressions=_regression("scn-0001"),
        limit_per_regression=5,
    )
    assert isinstance(plan, FeedbackCandidatePlan)
    vector_reqs = [req for req in plan.index_requirements if req["kind"] == "vector"]
    assert vector_reqs and vector_reqs[0]["met"] is True
    assert plan.runnable is True
    assert plan.effective_route == "ann"
    stage_names = [stage["stage"] for stage in plan.stages]
    assert stage_names == ["failure-mining", "gap-analysis", "dedup-diversity"]
    assert plan.to_dict()["plan_id"] == plan.plan_id


# --- 2. missing required indexes produce an actionable plan error ---


def test_missing_index_over_large_pool_is_actionable_plan_error(tmp_path, monkeypatch):
    lake = _small_lake(tmp_path / "robot.lance")
    monkeypatch.setattr(curate_mod, "_FEEDBACK_CANDIDATE_EXACT_SCAN_LIMIT", 3)
    regression = _regression("scn-anchor")

    plan = lake.curate.plan_next_candidates(from_regressions=regression, limit_per_regression=2)
    unmet = [req for req in plan.index_requirements if req["required"] and not req["met"]]
    assert unmet, "plan must list the unmet vector index requirement"
    assert unmet[0]["table"] == "scenarios"
    assert unmet[0]["column"] == "embedding"
    assert "scenarios index --column embedding" in unmet[0]["remedy"]
    assert plan.runnable is False
    assert plan.candidate_digest == ""

    with pytest.raises(CurationError) as excinfo:
        lake.curate.next_candidates(from_regressions=regression, limit_per_regression=2)
    message = str(excinfo.value)
    assert "vector index" in message
    assert "scenarios index --column embedding" in message
    assert lake.table("curation_review_queues").count_rows() == 0

    # An explicit exact route stays allowed as an operator override.
    exact = lake.curate.next_candidates(
        from_regressions=regression,
        queue_name="q-exact-override",
        limit_per_regression=2,
        route="exact",
    )
    assert exact.plan.effective_route == "exact"


def test_ann_route_without_index_errors(tmp_path):
    lake = _small_lake(tmp_path / "robot.lance")
    with pytest.raises(CurationError, match="no vector index"):
        lake.curate.next_candidates(
            from_regressions=_regression("scn-anchor"),
            limit_per_regression=2,
            route="ann",
        )


# --- 3. repeat execution produces the same queue id and row count ---


def test_repeat_execution_is_idempotent_across_all_artifacts(tmp_path):
    lake = _small_lake(tmp_path / "robot.lance")
    feedback = _feedback_for_small_lake(lake)
    kwargs = dict(
        from_regressions=feedback,
        queue_name="eval-b-regressions",
        view_name="eval-b-view",
        snapshot_name="eval-b-next",
        snapshot_tag="round-1",
        limit_per_regression=2,
    )

    first = lake.curate.next_candidates(**kwargs)
    queue_rows = lake.table("curation_review_queues").count_rows()
    view_rows = [
        row
        for row in lake.table("curation_views").to_arrow().to_pylist()
        if row["name"] == "eval-b-view"
    ]
    snapshot_rows = [
        row
        for row in lake.table("dataset_snapshots").to_arrow().to_pylist()
        if row["name"] == "eval-b-next"
    ]
    assert len(view_rows) == 1
    assert len(snapshot_rows) == 1

    second = lake.curate.next_candidates(**kwargs)

    assert first.queue is not None and second.queue is not None
    assert first.queue.queue_id == second.queue.queue_id
    assert first.plan.plan_id == second.plan.plan_id
    assert lake.table("curation_review_queues").count_rows() == queue_rows
    assert second.view is not None and second.view.view_id == first.view.view_id
    assert second.snapshot is not None
    assert second.snapshot.dataset_id == first.snapshot.dataset_id
    assert (
        len(
            [
                row
                for row in lake.table("dataset_snapshots").to_arrow().to_pylist()
                if row["name"] == "eval-b-next"
            ]
        )
        == 1
    ), "re-running the same call must not duplicate the candidate snapshot"
    assert (
        len(
            [
                row
                for row in lake.table("curation_views").to_arrow().to_pylist()
                if row["name"] == "eval-b-view"
            ]
        )
        == 1
    ), "re-running the same call must not duplicate the saved view"
    assert second.report["artifact_writes"] == {
        "queue": "unchanged",
        "view": "unchanged",
        "snapshot": "unchanged",
    }


# --- 4. dry-run and apply share candidate digest inputs ---


def test_dry_run_shares_candidate_digest_inputs_and_writes_nothing(tmp_path):
    lake = _small_lake(tmp_path / "robot.lance")
    feedback = _feedback_for_small_lake(lake)
    watched = (
        "curation_review_queues",
        "curation_views",
        "dataset_snapshots",
        "curation_memberships",
        "transform_runs",
    )
    before = {table: lake.table(table).count_rows() for table in watched}
    kwargs = dict(
        from_regressions=feedback,
        queue_name="eval-b-regressions",
        view_name="eval-b-view",
        snapshot_name="eval-b-next",
        limit_per_regression=2,
    )

    dry = lake.curate.next_candidates(dry_run=True, **kwargs)

    assert dry.dry_run is True
    assert dry.queue is None and dry.view is None and dry.snapshot is None
    assert dry.report["dry_run"] is True
    after = {table: lake.table(table).count_rows() for table in watched}
    assert after == before, "dry run must write zero rows"
    assert not any(
        row["transform_id"] == dry.transform_id
        for row in lake.table("transform_runs").to_arrow().to_pylist()
    ), "dry run must not record the feedback-loop transform"

    apply = lake.curate.next_candidates(**kwargs)

    assert dry.plan.plan_id == apply.plan.plan_id
    assert dry.report["candidate_digest"] == apply.report["candidate_digest"]
    assert dry.report["candidate_digest"]
    assert dry.transform_id == apply.transform_id
    assert dry.selection.scenario_ids == apply.selection.scenario_ids
    expected_queue = dry.plan.expected_artifacts["queue"]
    assert expected_queue["queue_id"] == apply.queue.queue_id
    assert apply.report["artifact_writes"]["queue"] == "written"


# --- 5. preview caps do not alter complete candidate counts ---


def test_preview_caps_do_not_alter_complete_counts(tmp_path):
    lake = _small_lake(tmp_path / "robot.lance")
    regression = _regression("scn-anchor")

    plan_default = lake.curate.plan_next_candidates(
        from_regressions=regression, limit_per_regression=4
    )
    plan_capped = lake.curate.plan_next_candidates(
        from_regressions=regression, limit_per_regression=4, preview_limit=2
    )

    assert plan_default.total_candidate_count == 4
    assert plan_capped.total_candidate_count == plan_default.total_candidate_count
    assert plan_capped.candidate_counts == plan_default.candidate_counts
    assert plan_capped.plan_id == plan_default.plan_id
    assert len(plan_capped.preview) == 2
    assert len(plan_default.preview) == 4


def test_preview_paging_is_deterministic_across_reloads(tmp_path):
    lake = _small_lake(tmp_path / "robot.lance")
    regression = _regression("scn-anchor")
    plan = lake.curate.plan_next_candidates(from_regressions=regression, limit_per_regression=4)

    page_one = lake.curate.preview_candidates(plan, limit=2)
    assert page_one.total_count == 4
    assert len(page_one.rows) == 2
    assert page_one.has_more is True
    assert page_one.next_cursor

    page_two = lake.curate.preview_candidates(plan, cursor=page_one.next_cursor, limit=2)
    assert len(page_two.rows) == 2
    assert page_two.has_more is False
    first_ids = [row["scenario_id"] for row in page_one.rows]
    second_ids = [row["scenario_id"] for row in page_two.rows]
    assert not set(first_ids) & set(second_ids)
    assert [row["ordinal"] for row in page_one.rows] == [0, 1]
    assert [row["ordinal"] for row in page_two.rows] == [2, 3]

    # A reload (re-plan) yields the identical plan and the identical pages.
    replanned = lake.curate.plan_next_candidates(
        from_regressions=regression, limit_per_regression=4
    )
    assert replanned.plan_id == plan.plan_id
    again = lake.curate.preview_candidates(replanned, limit=2)
    assert again.rows == page_one.rows
    assert again.next_cursor == page_one.next_cursor


# --- CLI: --explain / --dry-run / routing flags ---


def test_cli_next_candidates_explain_and_dry_run_write_nothing(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _small_lake(lake_path)
    feedback = _feedback_for_small_lake(lake)
    regressions_path = tmp_path / "regressions.json"
    regressions_path.write_text(json.dumps({"regressions": list(feedback.regressions)}))
    queue_rows_before = lake.table("curation_review_queues").count_rows()

    explain = runner.invoke(
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
            "--explain",
            "--json",
        ],
    )
    assert explain.exit_code == 0, explain.output
    plan_payload = json.loads(explain.output)
    assert plan_payload["plan_id"].startswith("fplan-")
    assert plan_payload["expected_artifacts"]["queue"]["name"] == "eval-b-regressions"
    assert [stage["stage"] for stage in plan_payload["stages"]] == [
        "failure-mining",
        "gap-analysis",
        "dedup-diversity",
    ]

    dry = runner.invoke(
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
            "--route",
            "auto",
            "--preview-limit",
            "2",
            "--dry-run",
            "--json",
        ],
    )
    assert dry.exit_code == 0, dry.output
    dry_payload = json.loads(dry.output)
    assert dry_payload["dry_run"] is True
    assert dry_payload["plan"]["plan_id"] == plan_payload["plan_id"]

    reopened = Lake.open(lake_path)
    assert reopened.table("curation_review_queues").count_rows() == queue_rows_before

    apply = runner.invoke(
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
    assert apply.exit_code == 0, apply.output
    assert "queue: eval-b-regressions" in apply.output
    reopened = Lake.open(lake_path)
    assert reopened.table("curation_review_queues").count_rows() > queue_rows_before
