"""Scalable curation row-plan catalog and chunked targets (backlog 0146).

The tests here pin the *shape* of the scale risk rather than today's row count
(SKILLS.md §3): the candidate scan's projection, the chunked write's commit
bounding, the target pager's windowed reads, the compile report's boundedness,
and the retention protections. Where a bound is the whole point of a fix, the
test asserts the mechanism (which columns were projected, which windows were
read) and not only that the answer is right.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest
from test_curate import _build_curation_lake
from typer.testing import CliRunner

from lancedb_robotics import curate as curate_module
from lancedb_robotics import curation_row_plans as rp
from lancedb_robotics.cli import app
from lancedb_robotics.curate import CurationScope
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import (
    CANONICAL_TABLES,
    OBSERVATIONS_SCHEMA,
    SCENARIOS_SCHEMA,
)

runner = CliRunner()


# --------------------------------------------------------------------------- #
# Fixtures / builders.
# --------------------------------------------------------------------------- #


def _wide_observation_lake(path, *, count: int) -> Lake:
    """A lake whose single scenario references ``count`` observations.

    Each observation carries a non-empty ``payload_blob`` so a test that asserts
    the candidate scan never projects the blob column is measuring something real
    (``payload_blob`` is Lance blob-encoded -- decision 0024). The blob is kept
    small on purpose: no assertion depends on its size, and the default suite
    should not need meaningful disk.
    """
    lake = _build_curation_lake(path)
    now = datetime.now(UTC)
    blob = b"x" * 256
    rows = [
        {
            "observation_id": f"obs-wide-{index:06d}",
            "run_id": "run-a",
            "episode_id": "",
            "episode_index": 0,
            "frame_index": index,
            "timestamp_ns": 10_000 + index,
            "sensor_id": "cam-0",
            "topic": "/cam0/image",
            "modality": "image",
            "robot_id": "arm-a",
            "site_id": "site-a",
            "task_id": "pick",
            "software_version": "",
            "outcome": "",
            "raw_uri": "memory://run-a",
            "raw_channel": "/cam0/image",
            "raw_log_time_ns": 10_000 + index,
            "raw_sequence": index,
            "payload_json": "",
            "payload_blob": blob,
            "message_encoding": "raw",
            "schema_encoding": "",
            "decode_status": "ok",
            "decode_error": "",
            "state_vector": [],
            "action_vector": [],
            "caption": "",
            "quality_flags": [],
            "transform_id": "tfm-seed",
            "created_at": now,
        }
        for index in range(count)
    ]
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-wide",
                    "run_id": "run-a",
                    "start_time_ns": 0,
                    "end_time_ns": 10_000 + count + 1,
                    "window_ns": 10_000 + count + 1,
                    "is_partial": False,
                    "topics": ["/cam0/image"],
                    "observation_ids": [row["observation_id"] for row in rows],
                    "observation_count": count,
                    "scenario_type": "window",
                    "trigger_event_id": "",
                    "source": "test",
                    "parent_scenario_id": "",
                    "coverage_tags": [],
                    "summary": "wide scenario",
                    "transform_id": "tfm-seed",
                    "created_at": now,
                }
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    return lake


def _compile_wide_plan(lake, *, freeze: bool = False):
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-wide",)))
    selection.save_view("wide-review")
    return lake.curate.compile_row_plan(
        view_name="wide-review", target_grain="observation", freeze=freeze
    )


def _small_plan(lake, *, freeze: bool = False):
    selection = lake.curate.workbench(
        scope=CurationScope(scenario_ids=("scn-anchor", "scn-neighbor"))
    )
    selection.save_view("row-review")
    return lake.curate.compile_row_plan(
        view_name="row-review", target_grain="observation", freeze=freeze
    )


# --------------------------------------------------------------------------- #
# Registration / structural guards.
# --------------------------------------------------------------------------- #


def test_tables_are_canonical_and_indexed():
    from lancedb_robotics.indexing import PREDICATE_INDEX_COLUMNS_BY_TABLE

    assert "curation_row_plans" in CANONICAL_TABLES
    assert "curation_row_plan_chunks" in CANONICAL_TABLES
    assert "curation_row_plans" in PREDICATE_INDEX_COLUMNS_BY_TABLE
    assert "curation_row_plan_chunks" in PREDICATE_INDEX_COLUMNS_BY_TABLE
    # The chunk table's paging predicate is ``plan_id = ? AND start_ordinal
    # BETWEEN ...``; both columns need an index or a deep page becomes a scan.
    chunk_columns = dict(PREDICATE_INDEX_COLUMNS_BY_TABLE["curation_row_plan_chunks"])
    assert chunk_columns["plan_id"] == "BTREE"
    assert chunk_columns["start_ordinal"] == "BTREE"


def test_lake_maintain_builds_the_new_tables_indexes(tmp_path, monkeypatch):
    """The new tables must be wired into ``lake maintain``, not indexed only inline.

    Index coverage does not auto-extend to new fragments (BUG-15), so a catalog that
    only gets indexes at write time degrades as it grows.
    """
    from lancedb_robotics.indexing import build_predicate_indexes_for_table

    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    _compile_wide_plan(lake)

    for table in ("curation_row_plans", "curation_row_plan_chunks"):
        results = build_predicate_indexes_for_table(lake, table)
        assert results, f"lake maintain would build no indexes for {table}"
        built = {result.column for result in results}
        assert "plan_id" in built
    # And the chunk pager's second predicate column is covered.
    chunk_columns = {
        result.column for result in build_predicate_indexes_for_table(
            lake, "curation_row_plan_chunks"
        )
    }
    assert "start_ordinal" in chunk_columns


def test_candidate_projection_never_includes_a_blob_column():
    """The candidate scan must never project a blob column (decision 0024).

    0084 read each grain table with a bare ``search()`` -- projecting every
    column, ``payload_blob`` included -- and kept whole rows in the candidate
    list. This guardrail fails if a future edit widens the projection.
    """
    blob_columns = {"payload_blob", "state_blob", "video_blob", "frame_blob"}
    for table, columns in curate_module._ROW_PLAN_CANDIDATE_COLUMNS.items():
        assert not (set(columns) & blob_columns), f"{table} projects a blob column"
    # And the observation projection is exactly the fields the builder reads.
    assert set(curate_module._ROW_PLAN_CANDIDATE_COLUMNS["observations"]) == {
        "observation_id",
        "run_id",
        "timestamp_ns",
        "raw_log_time_ns",
    }


def test_candidate_rows_do_not_retain_the_source_row(tmp_path):
    """A candidate keeps ids and a sort key -- never the source row.

    Retaining the row made candidate memory scale with row *width* (blob column
    included) rather than candidate count.
    """
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=8)
    candidates = curate_module._row_plan_candidates(
        lake, target_grain="observation", scenario_ids=("scn-wide",), source_snapshot_name=None
    )
    assert candidates
    for candidate in candidates:
        assert "row" not in candidate
        assert set(candidate) == {
            "target_grain",
            "table",
            "target_id",
            "scenario_id",
            "scenario_ids",
            "row_id",
            "sort_key",
        }


def test_candidate_scan_projects_and_streams(tmp_path, monkeypatch):
    """The scan must push a projection down, not select every column."""
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=8)
    selected_columns: list[list[str]] = []
    original = curate_module._stream_rows_with_row_id

    def _spy(lake_arg, table_name, columns):
        selected_columns.append(list(columns))
        return original(lake_arg, table_name, columns)

    monkeypatch.setattr(curate_module, "_stream_rows_with_row_id", _spy)
    curate_module._row_plan_candidates(
        lake, target_grain="observation", scenario_ids=("scn-wide",), source_snapshot_name=None
    )
    assert selected_columns, "candidate builder did not go through the projected scan"
    for columns in selected_columns:
        assert "payload_blob" not in columns


# --------------------------------------------------------------------------- #
# Inline plans stay byte-compatible with 0084.
# --------------------------------------------------------------------------- #


def test_small_plan_stays_inline_and_is_catalogued(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    plan = _small_plan(lake)

    assert plan.storage_kind == "inline"
    assert plan.materialized is True
    assert plan.target_count == len(plan.target_ids)
    # No chunk rows for an inline plan at all.
    assert lake.table("curation_row_plan_chunks").count_rows() == 0

    entry = lake.curate.row_plan(plan.plan_id)
    assert entry.plan_id == plan.plan_id
    assert entry.storage_kind == "inline"
    assert entry.target_count == plan.target_count
    assert entry.state == rp.STATE_ACTIVE
    assert entry.target_ids_digest == rp.target_ids_digest(plan.target_ids)

    targets = lake.curate.row_plan_targets(plan.plan_id)
    assert [target.target_id for target in targets.iter_targets()] == list(plan.target_ids)
    assert [target.ordinal for target in targets.iter_targets()] == list(
        range(len(plan.target_ids))
    )


def test_inline_frozen_artifact_keeps_its_row_ids(tmp_path):
    """Below the threshold the compact lineage handle is unchanged from 0084."""
    lake = _build_curation_lake(tmp_path / "robot.lance")
    plan = _small_plan(lake, freeze=True)

    assert plan.frozen is True
    artifact = next(
        row
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
        if row["artifact_id"] == plan.artifact_id
    )
    assert artifact["kind"] == "curation-row-plan"
    assert list(artifact["row_ids"]) == sorted(plan.target_ids)
    metadata = {item["key"]: item["value"] for item in artifact["metadata"]}
    assert metadata["row_plan_storage"] == "inline"
    assert metadata["row_ids_inline"] == "true"
    assert metadata["row_plan_id"] == plan.plan_id


# --------------------------------------------------------------------------- #
# Chunked plans: the scale shape.
# --------------------------------------------------------------------------- #


def test_large_plan_chunks_targets_and_pages_in_stable_order(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 16)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=137)
    plan = _compile_wide_plan(lake)

    assert plan.storage_kind == "chunked"
    assert plan.target_count == 137
    assert plan.storage["chunk_count"] == 18  # ceil(137 / 8)
    assert lake.table("curation_row_plan_chunks").count_rows() == 18

    entry = lake.curate.row_plan(plan.plan_id)
    assert entry.chunk_count == 18
    assert entry.chunk_size == 8
    assert entry.target_count == 137

    handle = lake.curate.row_plan_targets(plan.plan_id)
    streamed = list(handle.iter_targets())
    assert len(streamed) == 137
    assert [target.ordinal for target in streamed] == list(range(137))
    # Compile order is scenario order then observation ordinal.
    assert streamed[0].target_id == "obs-wide-000000"
    assert streamed[-1].target_id == "obs-wide-000136"
    # Row ids survive the round trip so a consumer can ``take_row_ids``.
    assert all(target.lance_row_id is not None for target in streamed)
    assert rp.target_ids_digest(t.target_id for t in streamed) == entry.target_ids_digest


def test_chunked_paging_is_resumable_and_covers_every_target(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 16)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=100)
    plan = _compile_wide_plan(lake)

    handle = lake.curate.row_plan_targets(plan.plan_id)
    seen: list[str] = []
    pages = 0
    cursor = None
    while True:
        page = handle.page(page_size=17, cursor=cursor)
        pages += 1
        assert len(page.targets) <= 17
        assert page.start_ordinal == len(seen)
        seen.extend(target.target_id for target in page.targets)
        if not page.has_more:
            assert page.next_cursor is None
            break
        cursor = page.next_cursor
    assert pages == 6  # ceil(100 / 17)
    assert len(seen) == 100
    assert seen == sorted(seen)
    assert len(set(seen)) == 100


def test_deep_page_reads_only_its_own_window(tmp_path, monkeypatch):
    """Paging from a late ordinal must not read the chunks before it.

    This is the bound the whole chunked layout exists for: without the windowed
    ordinal predicate a deep page re-scans the plan's prefix every time.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 8)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 4)
    monkeypatch.setattr(rp, "_CHUNK_SCAN_BATCH", 2)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=64)
    plan = _compile_wide_plan(lake)

    windows: list[str] = []
    original = rp._stream_rows

    def _spy(lake_arg, table, **kwargs):
        if table == "curation_row_plan_chunks" and kwargs.get("where_sql"):
            windows.append(kwargs["where_sql"])
        return original(lake_arg, table, **kwargs)

    monkeypatch.setattr(rp, "_stream_rows", _spy)
    handle = lake.curate.row_plan_targets(plan.plan_id)
    page = handle.page(page_size=4, cursor=rp.encode_target_cursor(48))

    assert [target.ordinal for target in page.targets] == [48, 49, 50, 51]
    assert windows, "the pager did not push an ordinal window into the scan"
    # Every window read starts at or past the requested ordinal's chunk.
    for where_sql in windows:
        assert "start_ordinal >= 48" in where_sql or "start_ordinal >= 5" in where_sql
    assert not any("start_ordinal >= 0 " in where_sql for where_sql in windows)


def test_above_soft_limit_the_result_is_not_materialized(tmp_path, monkeypatch):
    """Past the soft limit the dataclass stops holding the id list."""
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    monkeypatch.setattr(rp, "MATERIALIZE_SOFT_LIMIT", 10)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan = _compile_wide_plan(lake)

    assert plan.target_count == 40
    assert plan.materialized is False
    assert plan.target_ids == ()
    assert plan.lance_row_ids == ()
    assert plan.report["targets_materialized"] is False
    assert plan.report["selected_target_ids"] == []
    # ... but the targets are still fully readable, in order, via the handle.
    assert len(list(plan.targets().iter_targets())) == 40


def test_chunked_frozen_artifact_points_at_chunk_storage(tmp_path, monkeypatch):
    """Above the threshold the artifact carries a pointer, not an id list."""
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan = _compile_wide_plan(lake, freeze=True)

    artifact = next(
        row
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
        if row["artifact_id"] == plan.artifact_id
    )
    assert list(artifact["row_ids"]) == []
    metadata = {item["key"]: item["value"] for item in artifact["metadata"]}
    assert metadata["row_plan_storage"] == "chunked"
    assert metadata["row_ids_inline"] == "false"
    assert metadata["row_plan_chunk_table"] == "curation_row_plan_chunks"
    assert metadata["row_plan_chunk_count"] == "5"
    assert metadata["row_plan_target_count"] == "40"
    # The pointer is enough to reopen the full membership.
    reopened = lake.curate.row_plan_targets(metadata["row_plan_id"])
    assert reopened.target_count == 40
    assert len(list(reopened.iter_targets())) == 40


def test_chunk_writes_are_bounded_commits(tmp_path, monkeypatch):
    """A large plan is a sequence of bounded commits, never one oversized write."""
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 4)
    monkeypatch.setattr(rp, "_CHUNK_WRITE_BATCH", 3)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)

    commit_sizes: list[int] = []
    original = rp._merge_insert_with_retry

    def _spy(table, key_column, data, *, update_matched):
        if key_column == "chunk_id":
            commit_sizes.append(data.num_rows)
        return original(table, key_column, data, update_matched=update_matched)

    monkeypatch.setattr(rp, "_merge_insert_with_retry", _spy)
    plan = _compile_wide_plan(lake)

    assert plan.storage["chunk_count"] == 10
    assert commit_sizes, "chunks were not committed through the bounded writer"
    assert max(commit_sizes) <= 3
    assert sum(commit_sizes) == 10


def test_recompiling_the_same_plan_converges_without_duplicating(tmp_path, monkeypatch):
    """Content-addressed chunks make a retried write idempotent (BUG-04)."""
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    first = _compile_wide_plan(lake)
    chunk_rows = lake.table("curation_row_plan_chunks").count_rows()
    plan_rows = lake.table("curation_row_plans").count_rows()

    second = lake.curate.compile_row_plan(
        view_name="wide-review", target_grain="observation"
    )

    assert second.plan_id == first.plan_id
    assert lake.table("curation_row_plan_chunks").count_rows() == chunk_rows
    assert lake.table("curation_row_plans").count_rows() == plan_rows


def test_plan_id_is_computed_without_the_id_list(tmp_path, monkeypatch):
    """The v2 plan payload carries a digest + count, never the ordered id list."""
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=20)
    plan = _compile_wide_plan(lake)

    assert plan.report["schema_version"] == "lancedb-robotics/curation-row-plan/v2"
    assert "target_ids" not in {
        key for key in plan.report if key in {"target_ids"}
    } or plan.report.get("target_ids") is None
    assert plan.report["target_count"] == 20
    assert plan.report["target_ids_digest"] == rp.target_ids_digest(
        target.target_id for target in plan.targets().iter_targets()
    )


# --------------------------------------------------------------------------- #
# Report / summary boundedness.
# --------------------------------------------------------------------------- #


def test_compile_report_and_transform_params_stay_bounded(tmp_path, monkeypatch):
    """The report is serialized into ``transform_runs.params``; it must not grow.

    0084 embedded ``selected_rows`` (one dict per target), ``selected_target_ids``,
    ``lance_row_ids``, and the full ``rejected`` map -- so the transform row grew
    with the plan (BUG-02's oversized-commit shape).
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    monkeypatch.setattr(rp, "MATERIALIZE_SOFT_LIMIT", 8)
    monkeypatch.setattr(rp, "SAMPLE_LIMIT", 5)

    sizes = {}
    for count in (40, 200):
        lake = _wide_observation_lake(tmp_path / f"robot-{count}.lance", count=count)
        plan = _compile_wide_plan(lake)
        assert plan.report["selected_count"] == count
        assert len(plan.report["selected_rows"]) == 5
        assert plan.report["selected_rows_truncated"] is True
        assert plan.report["selected_target_ids"] == []
        params = next(
            row["params"]
            for row in lake.table("transform_runs").to_arrow().to_pylist()
            if row["transform_id"] == plan.transform_id
        )
        sizes[count] = len(params)
        # The persisted params must not contain the plan's target ids.
        decoded = json.loads(params)
        assert decoded["selected_target_ids"] == []
        assert decoded["target_count"] == count

    # 5x the targets must not grow the transform row meaningfully.
    assert sizes[200] < sizes[40] * 1.2


def test_report_stays_bounded_at_default_limits_as_decisions_grow(tmp_path):
    """At *default* limits, more targets AND more decisions must not grow params.

    ``selected_target_ids``/``lance_row_ids`` are gated on the inline threshold, and
    the membership-side diagnostics (``latest_membership_ids``,
    ``superseded_membership_ids``, ``supersession_chains``) are sampled -- those
    three grow with the branch's decision history, not the target count, and were
    the remaining unbounded lists.
    """
    # Both counts exceed the default INLINE_TARGET_LIMIT, so both plans are chunked
    # and the ids belong in the chunk table rather than the report.
    assert rp.INLINE_TARGET_LIMIT < 1200
    sizes = {}
    for count, decisions in ((1200, 0), (2400, 250)):
        lake = _wide_observation_lake(tmp_path / f"robot-{count}.lance", count=count)
        selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-wide",)))
        selection.save_view("wide-review")
        if decisions:
            # ``exclude`` (not ``promote``) so the plan stays large: a promote
            # decision switches the compiler to include-mode and would select only
            # the promoted rows, shrinking the plan back under the inline limit.
            selection.record_decisions(
                view_name="wide-review",
                decision="exclude",
                target_grain="observation",
                target_ids=[f"obs-wide-{index:06d}" for index in range(decisions)],
                reason="excluded batch",
            )
        plan = lake.curate.compile_row_plan(
            view_name="wide-review", target_grain="observation"
        )
        assert plan.storage_kind == "chunked"
        params = next(
            row["params"]
            for row in lake.table("transform_runs").to_arrow().to_pylist()
            if row["transform_id"] == plan.transform_id
        )
        decoded = json.loads(params)
        sizes[(count, decisions)] = len(params)
        assert decoded["selected_target_ids"] == []
        assert decoded["lance_row_ids"] == []
        for key in (
            "selected_rows",
            "latest_membership_ids",
            "superseded_membership_ids",
            "supersession_chains",
        ):
            assert len(decoded[key]) <= rp.SAMPLE_LIMIT, key
        # Exact counts survive alongside the samples.
        assert decoded["latest_membership_id_count"] == decisions
        # And the identity payload carries a digest, never the id list.
        assert "superseded_membership_ids_digest" in decoded
        assert isinstance(decoded["superseded_membership_count"], int)

    small = sizes[(1200, 0)]
    large = sizes[(2400, 250)]
    assert large < small * 3, f"params grew {small} -> {large} with targets/decisions"


def test_summary_samples_are_capped_and_counts_are_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "SAMPLE_LIMIT", 3)
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=20)
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-wide",)))
    selection.save_view("wide-review")
    selection.record_decisions(
        view_name="wide-review",
        decision="exclude",
        target_grain="observation",
        target_ids=[f"obs-wide-{index:06d}" for index in range(10)],
        reason="bad exposure",
    )
    plan = lake.curate.compile_row_plan(
        view_name="wide-review", target_grain="observation"
    )

    assert plan.report["rejected_count"] == 10
    assert len(plan.report["rejected"]) == 3
    assert plan.report["rejected_truncated"] is True
    assert plan.report["reject_reason_counts"] == {"row-exclude": 10}

    summary = lake.curate.row_plan_summary(plan.plan_id)
    assert summary["rejected_count"] == 10
    assert summary["summary"]["rejected_truncated"] is True
    assert len(summary["summary"]["rejected"]) == 3
    assert summary["summary"]["reject_reason_counts"] == {"row-exclude": 10}


# --------------------------------------------------------------------------- #
# Listing.
# --------------------------------------------------------------------------- #


def test_row_plan_listing_is_paged_filtered_and_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=12)
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-wide",)))
    selection.save_view("wide-review")
    plan_ids = []
    for index in range(3):
        selection.record_decisions(
            view_name="wide-review",
            decision="exclude",
            target_grain="observation",
            target_ids=[f"obs-wide-{index:06d}"],
            reason=f"pass-{index}",
        )
        plan_ids.append(
            lake.curate.compile_row_plan(
                view_name="wide-review", target_grain="observation"
            ).plan_id
        )

    seen = []
    cursor = None
    pages = 0
    while True:
        page = lake.curate.row_plans(page_size=2, cursor=cursor)
        pages += 1
        seen.extend(record.plan_id for record in page.records)
        if not page.has_more:
            break
        cursor = page.next_cursor
    assert pages >= 2
    assert set(plan_ids) <= set(seen)
    assert len(seen) == len(set(seen))

    # Filters narrow to indexed columns.
    filtered = lake.curate.row_plans(target_grain="observation", storage_kind="chunked")
    assert filtered.records
    assert all(record.target_grain == "observation" for record in filtered.records)
    assert all(record.storage_kind == "chunked" for record in filtered.records)
    assert not lake.curate.row_plans(target_grain="episode").records


def test_listing_never_reads_the_summary_body(tmp_path, monkeypatch):
    """The listing projection must stay JSON-free (SKILLS.md §2)."""
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=12)
    _compile_wide_plan(lake)

    assert "summary_json" not in rp.PLAN_SUMMARY_COLUMNS
    projected: list[list[str]] = []
    original = rp._stream_rows

    def _spy(lake_arg, table, **kwargs):
        if table == "curation_row_plans":
            projected.append(list(kwargs.get("columns") or ()))
        return original(lake_arg, table, **kwargs)

    monkeypatch.setattr(rp, "_stream_rows", _spy)
    # Force the heap fallback path so the projection is observable here too.
    monkeypatch.setattr(rp, "_read_plans_ordered", lambda *a, **k: None)
    with pytest.warns(RuntimeWarning):
        page = rp.list_row_plans(lake, page_size=5)
    assert page.records
    assert projected
    for columns in projected:
        assert "summary_json" not in columns


# --------------------------------------------------------------------------- #
# Validation / crash convergence.
# --------------------------------------------------------------------------- #


def test_heap_fallback_returns_every_plan_exactly_once(tmp_path, monkeypatch):
    """The bounded-heap fallback must not drop rows.

    Keeping the ``page_size+1`` smallest keys requires evicting the current
    *largest* on overflow. A min-heap rooted at the smallest evicts the row that
    belongs on page one, and that plan is then returned by no page at all.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=12)
    plan_ids = _compile_series(lake, passes=5)
    assert len(set(plan_ids)) == 5

    # Force the heap path for every page.
    monkeypatch.setattr(rp, "_read_plans_ordered", lambda *a, **k: None)
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        with pytest.warns(RuntimeWarning, match="bounded top-page heap"):
            page = rp.list_row_plans(lake, page_size=2, cursor=cursor)
        seen.extend(record.plan_id for record in page.records)
        if not page.has_more:
            break
        cursor = page.next_cursor

    assert len(seen) == len(set(seen)), "the heap fallback duplicated a plan"
    assert set(seen) == set(plan_ids), f"missing: {sorted(set(plan_ids) - set(seen))}"
    # And the pages are in ascending key order overall.
    keys = [lake.curate.row_plan(plan_id).created_at for plan_id in seen]
    assert keys == sorted(keys)


def test_retuning_chunk_size_yields_a_new_plan_id(tmp_path, monkeypatch):
    """The physical chunk width is part of the plan's identity.

    Without it, recompiling after retuning the width reuses the same ``plan_id``,
    and the insert-only chunk write leaves two chunkings coexisting -- ordinals
    stop lining up and the plan can never be read again, while the compile still
    reports success.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    first = _compile_wide_plan(lake)
    assert first.storage["chunk_count"] == 5

    monkeypatch.setattr(rp, "CHUNK_SIZE", 4)
    second = lake.curate.compile_row_plan(
        view_name="wide-review", target_grain="observation"
    )

    assert second.plan_id != first.plan_id
    assert second.storage["chunk_count"] == 10
    # Both plans remain independently readable at their own width.
    assert len(list(lake.curate.row_plan_targets(first.plan_id).iter_targets())) == 40
    assert len(list(lake.curate.row_plan_targets(second.plan_id).iter_targets())) == 40
    assert lake.curate.validate_row_plans().ok is True


def test_stale_row_ids_after_compaction_are_reported_not_silently_served(
    tmp_path, monkeypatch
):
    """Lance row ids are fragment addresses; compaction remaps them.

    A plan whose target table has moved past the pinned version must say its stored
    row ids are untrustworthy rather than hand a consumer addresses that fail deep
    inside Lance. The target ids stay valid either way.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan = _compile_wide_plan(lake)

    handle = lake.curate.row_plan_targets(plan.plan_id)
    assert handle.row_ids_trustworthy is True
    assert lake.curate.validate_row_plans().ok is True

    from lancedb_robotics.maintenance import maintain_lake

    maintain_lake(
        lake,
        tables=["observations"],
        compact=True,
        refresh_indexes=False,
        refresh_lineage=False,
        cleanup_older_than=None,
    )

    stale = lake.curate.row_plan_targets(plan.plan_id)
    assert stale.row_ids_trustworthy is False
    assert stale.pinned_table_version != stale.current_table_version
    assert stale.to_dict()["row_ids_trustworthy"] is False
    # Target ids are unaffected and still complete + ordered.
    assert len(list(stale.iter_targets())) == 40

    report = lake.curate.validate_row_plans(plan_id=plan.plan_id)
    assert "row-ids-possibly-stale" in {issue.code for issue in report.issues}


def test_a_plain_append_does_not_mark_row_ids_stale(tmp_path, monkeypatch):
    """An append bumps the table version but remaps nothing.

    Keying staleness on the version number would flag every plan in any actively
    ingesting lake — `validate-row-plans` would never return ok again, and consumers
    would be pushed off the `take_row_ids` fast path BUG-06 round 2 exists to
    provide. The signal has to be the fragment set, which only compaction rewrites.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan = _compile_wide_plan(lake)
    pinned_version = lake.curate.row_plan_targets(plan.plan_id).pinned_table_version

    # Append more observations: version moves, fragments are only extended.
    now = datetime.now(UTC)
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                {
                    **{field.name: None for field in OBSERVATIONS_SCHEMA},
                    "observation_id": f"obs-later-{index:04d}",
                    "run_id": "run-a",
                    "episode_index": 0,
                    "frame_index": index,
                    "timestamp_ns": 900_000 + index,
                    "raw_log_time_ns": 900_000 + index,
                    "raw_sequence": index,
                    "topic": "/cam0/image",
                    "payload_blob": b"y" * 64,
                    "state_vector": [],
                    "action_vector": [],
                    "quality_flags": [],
                    "created_at": now,
                }
                for index in range(5)
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )

    handle = lake.curate.row_plan_targets(plan.plan_id)
    assert handle.current_table_version != pinned_version, "the append should move the version"
    assert handle.row_ids_trustworthy is True
    assert lake.curate.validate_row_plans(plan_id=plan.plan_id).ok is True
    assert len(list(handle.iter_targets())) == 40


def test_validation_passes_on_a_healthy_chunked_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan = _compile_wide_plan(lake)

    report = lake.curate.validate_row_plans()
    assert report.ok is True
    assert plan.plan_id in report.scanned_plan_ids


def test_validation_catches_a_missing_chunk_and_the_reader_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan = _compile_wide_plan(lake)

    lake.table("curation_row_plan_chunks").delete(
        f"plan_id = '{plan.plan_id}' AND chunk_index = 2"
    )

    report = lake.curate.validate_row_plans(plan_id=plan.plan_id)
    assert report.ok is False
    codes = {issue.code for issue in report.issues}
    assert "target-count-mismatch" in codes

    # A short read is a loud failure, never a silently truncated plan.
    with pytest.raises(rp.RowPlanCatalogError):
        list(lake.curate.row_plan_targets(plan.plan_id).iter_targets())


def test_orphan_chunks_from_an_interrupted_write_are_reported_and_reclaimed(
    tmp_path, monkeypatch
):
    """Chunks-first / header-last means a crash leaves orphans, not a bad header."""
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)

    boom = RuntimeError("crash before the header publish")

    def _explode(*args, **kwargs):
        raise boom

    monkeypatch.setattr(curate_module._row_plan_catalog, "publish_row_plan", _explode)
    with pytest.raises(RuntimeError):
        _compile_wide_plan(lake)
    monkeypatch.undo()

    assert lake.table("curation_row_plan_chunks").count_rows() > 0
    assert lake.table("curation_row_plans").count_rows() == 0

    report = lake.curate.validate_row_plans()
    assert report.orphan_chunk_plan_ids

    # A retry converges onto the same content-addressed chunks and publishes.
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    plan = lake.curate.compile_row_plan(
        view_name="wide-review", target_grain="observation"
    )
    assert plan.storage_kind == "chunked"
    assert lake.curate.validate_row_plans().ok is True
    assert len(list(plan.targets().iter_targets())) == 40


def test_compact_reclaims_only_orphans(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan = _compile_wide_plan(lake)
    healthy = lake.table("curation_row_plan_chunks").count_rows()

    # Fabricate an orphan chunk with no header.
    from lancedb_robotics.schemas import CURATION_ROW_PLAN_CHUNKS_SCHEMA

    lake.table("curation_row_plan_chunks").add(
        pa.Table.from_pylist(
            [
                {
                    "chunk_id": "rowplanchunk-orphan",
                    "plan_id": "curation-rowplan-ghost",
                    "chunk_index": 0,
                    "start_ordinal": 0,
                    "end_ordinal": 0,
                    "target_ids": ["ghost"],
                    "lance_row_ids": [0],
                    "target_count": 1,
                    "chunk_digest": "deadbeef",
                    "created_by": "test",
                    "transform_id": "",
                    "created_at": datetime.now(UTC),
                }
            ],
            schema=CURATION_ROW_PLAN_CHUNKS_SCHEMA,
        )
    )

    # A *fresh* orphan is indistinguishable from a compile that has written its
    # chunks and not yet published its header, so the grace window protects it.
    fresh = lake.curate.compact_row_plans(dry_run=True)
    assert fresh["orphan_plan_ids"] == []
    assert fresh["skipped_in_flight_plan_ids"] == ["curation-rowplan-ghost"]
    assert lake.curate.compact_row_plans()["chunks_deleted"] == 0
    assert lake.table("curation_row_plan_chunks").count_rows() == healthy + 1

    # Past the grace window the same rows are reclaimable.
    preview = lake.curate.compact_row_plans(dry_run=True, grace=timedelta(seconds=0))
    assert preview["orphan_plan_ids"] == ["curation-rowplan-ghost"]
    assert preview["chunks_deleted"] == 0

    applied = lake.curate.compact_row_plans(grace=timedelta(seconds=0))
    assert applied["chunks_deleted"] == 1
    assert lake.table("curation_row_plan_chunks").count_rows() == healthy
    assert len(list(lake.curate.row_plan_targets(plan.plan_id).iter_targets())) == 40


def test_lake_maintain_reclaims_row_plan_orphans(tmp_path, monkeypatch):
    """The write path deliberately produces orphans, so maintenance must collect them.

    Leaving it to an operator remembering ``curate compact-row-plans`` means they
    accumulate forever.
    """
    from lancedb_robotics.maintenance import maintain_lake
    from lancedb_robotics.schemas import CURATION_ROW_PLAN_CHUNKS_SCHEMA

    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    # Zero grace so the fabricated orphan is immediately eligible.
    monkeypatch.setattr(rp, "ORPHAN_GRACE", timedelta(seconds=0))
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan = _compile_wide_plan(lake)
    healthy = lake.table("curation_row_plan_chunks").count_rows()

    lake.table("curation_row_plan_chunks").add(
        pa.Table.from_pylist(
            [
                {
                    "chunk_id": "rowplanchunk-maint-orphan",
                    "plan_id": "curation-rowplan-maint-ghost",
                    "chunk_index": 0,
                    "start_ordinal": 0,
                    "end_ordinal": 0,
                    "target_ids": ["ghost"],
                    "lance_row_ids": [0],
                    "target_count": 1,
                    "chunk_digest": "deadbeef",
                    "created_by": "test",
                    "transform_id": "",
                    "created_at": datetime.now(UTC),
                }
            ],
            schema=CURATION_ROW_PLAN_CHUNKS_SCHEMA,
        )
    )

    report = maintain_lake(
        lake,
        tables=["curation_row_plans", "curation_row_plan_chunks"],
        compact=False,
        refresh_indexes=False,
        refresh_lineage=False,
        cleanup_older_than=None,
        compact_curation_chunks=True,
    )
    assert report.curation_row_plan_chunks is not None
    assert report.curation_row_plan_chunks["compaction"]["chunks_deleted"] == 1
    assert lake.table("curation_row_plan_chunks").count_rows() == healthy
    assert len(list(lake.curate.row_plan_targets(plan.plan_id).iter_targets())) == 40


def test_compaction_never_reclaims_an_in_flight_compiles_chunks(tmp_path, monkeypatch):
    """The chunks-first window must not be mistaken for garbage (backlog 0258 shape).

    A compile that has written 5M targets' chunks but not yet published its header
    looks exactly like an orphan. Reclaiming inside the grace window would delete
    rows a live writer is about to publish a header for, and the compile would then
    report success on an unreadable plan.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)

    # Simulate the in-flight window: chunks land, header publish is interrupted.
    monkeypatch.setattr(
        curate_module._row_plan_catalog,
        "publish_row_plan",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(RuntimeError):
        _compile_wide_plan(lake)
    monkeypatch.undo()
    in_flight_chunks = lake.table("curation_row_plan_chunks").count_rows()
    assert in_flight_chunks > 0

    report = lake.curate.compact_row_plans()
    assert report["chunks_deleted"] == 0
    assert report["skipped_in_flight_count"] == 1
    assert lake.table("curation_row_plan_chunks").count_rows() == in_flight_chunks


# --------------------------------------------------------------------------- #
# Retention.
# --------------------------------------------------------------------------- #


def _compile_series(lake, *, passes: int, freeze_last: bool = False) -> list[str]:
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-wide",)))
    selection.save_view("wide-review")
    plan_ids = []
    for index in range(passes):
        selection.record_decisions(
            view_name="wide-review",
            decision="exclude",
            target_grain="observation",
            target_ids=[f"obs-wide-{index:06d}"],
            reason=f"pass-{index}",
        )
        plan_ids.append(
            lake.curate.compile_row_plan(
                view_name="wide-review",
                target_grain="observation",
                freeze=freeze_last and index == passes - 1,
            ).plan_id
        )
    return plan_ids


def test_retention_supersedes_then_prunes_only_old_unfrozen_plans(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan_ids = _compile_series(lake, passes=3)
    counts = {plan_id: lake.curate.row_plan(plan_id).target_count for plan_id in plan_ids}
    digests = {
        plan_id: lake.curate.row_plan(plan_id).target_ids_digest for plan_id in plan_ids
    }

    # No cutoff: soft-retire only, nothing deleted.
    soft = lake.curate.prune_row_plans(retain_latest=1)
    assert soft.pruned_count == 0
    assert soft.superseded_count == 2
    assert lake.table("curation_row_plan_chunks").count_rows() > 0

    # With a cutoff in the future, the superseded plans prune.
    hard = lake.curate.prune_row_plans(retain_latest=1, older_than=timedelta(seconds=-60))
    assert hard.pruned_count == 2
    assert set(hard.pruned_plan_ids) == set(plan_ids[:2])
    assert hard.chunks_deleted > 0

    # Promoted columns survive as audit evidence; the body and chunks do not.
    pruned = lake.curate.row_plan(plan_ids[0])
    assert pruned.state == rp.STATE_PRUNED
    assert pruned.target_count == counts[plan_ids[0]]
    assert pruned.target_ids_digest == digests[plan_ids[0]]
    assert pruned.summary_available is False
    with pytest.raises(rp.RowPlanCatalogError):
        lake.curate.row_plan_targets(plan_ids[0])

    # The retained plan is untouched and still fully readable.
    kept = lake.curate.row_plan(plan_ids[-1])
    assert kept.state == rp.STATE_ACTIVE
    assert (
        len(list(lake.curate.row_plan_targets(plan_ids[-1]).iter_targets()))
        == counts[plan_ids[-1]]
    )


def test_retention_refuses_to_prune_a_frozen_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-wide",)))
    selection.save_view("wide-review")
    frozen = lake.curate.compile_row_plan(
        view_name="wide-review", target_grain="observation", freeze=True
    )
    selection.record_decisions(
        view_name="wide-review",
        decision="exclude",
        target_grain="observation",
        target_ids=["obs-wide-000000"],
        reason="later pass",
    )
    lake.curate.compile_row_plan(view_name="wide-review", target_grain="observation")

    report = lake.curate.prune_row_plans(
        retain_latest=1, older_than=timedelta(seconds=-60)
    )
    assert frozen.plan_id not in report.pruned_plan_ids
    assert frozen.plan_id in report.protected_plan_ids
    # Frozen evidence stays readable through its chunked storage.
    assert len(list(lake.curate.row_plan_targets(frozen.plan_id).iter_targets())) == 40


def test_retention_refuses_to_prune_a_plan_a_training_run_references(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan_ids = _compile_series(lake, passes=3)
    pinned = plan_ids[0]

    from lancedb_robotics.schemas import TRAINING_RUNS_SCHEMA

    row = {name: None for name in TRAINING_RUNS_SCHEMA.names}
    row.update(
        {
            "training_run_id": "trn-pinned",
            "dataset_id": "ds-1",
            "snapshot_name": "snap-1",
            "snapshot_tag": "",
            "table_versions": [],
            "row_plan_id": pinned,
            "epoch_plan_id": "",
            "projection_manifest_ids": [],
            "created_at": datetime.now(UTC),
        }
    )
    lake.table("training_runs").add(
        pa.Table.from_pylist([row], schema=TRAINING_RUNS_SCHEMA)
    )

    assert rp.referenced_plan_ids(lake, plan_ids) == {pinned}
    pinned_chunks = lake.table("curation_row_plan_chunks").count_rows(
        f"plan_id = '{pinned}'"
    )
    assert pinned_chunks > 0

    # A dry run must preview the protection, or it cannot warn about it.
    preview = lake.curate.prune_row_plans(
        retain_latest=1, older_than=timedelta(seconds=-60), dry_run=True
    )
    assert pinned not in preview.pruned_plan_ids
    assert pinned in preview.protected_plan_ids

    report = lake.curate.prune_row_plans(
        retain_latest=1, older_than=timedelta(seconds=-60)
    )
    assert pinned not in report.pruned_plan_ids
    assert pinned in report.protected_plan_ids
    # The load-bearing assertion: "protected" must mean the membership SURVIVES.
    # Reporting protection while deleting the chunks is the failure mode here.
    assert (
        lake.table("curation_row_plan_chunks").count_rows(f"plan_id = '{pinned}'")
        == pinned_chunks
    )
    assert lake.curate.row_plan(pinned).state != rp.STATE_PRUNED
    assert len(list(lake.curate.row_plan_targets(pinned).iter_targets())) == counts_of(
        lake, pinned
    )


def counts_of(lake, plan_id: str) -> int:
    return lake.curate.row_plan(plan_id).target_count


def test_soft_retire_preserves_version_pinning_and_the_summary_body(tmp_path, monkeypatch):
    """A state flip must not null the columns the retention scan didn't read.

    ``merge_insert(...).when_matched_update_all()`` writes every column of the row
    it is handed, so rewriting a narrow scan row would drop ``table_versions`` --
    the plan's version pinning, and the thing that makes it reproducible -- on the
    path documented as "nothing is deleted".
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=12)
    plan_ids = _compile_series(lake, passes=2)
    oldest = plan_ids[0]

    before = next(
        row
        for row in lake.table("curation_row_plans").to_arrow().to_pylist()
        if row["plan_id"] == oldest
    )
    assert before["table_versions"]
    assert before["summary_json"]

    report = lake.curate.prune_row_plans(retain_latest=1)
    assert report.pruned_count == 0
    assert oldest in report.superseded_plan_ids

    after = next(
        row
        for row in lake.table("curation_row_plans").to_arrow().to_pylist()
        if row["plan_id"] == oldest
    )
    assert after["state"] == rp.STATE_SUPERSEDED
    assert after["table_versions"] == before["table_versions"]
    assert after["summary_json"] == before["summary_json"]
    assert after["summary_available"] is True
    # And the summary still round-trips through the public reader.
    assert lake.curate.row_plan_summary(oldest)["summary_available"] is True


def test_retention_retains_the_newest_plan_on_the_unordered_fallback(tmp_path, monkeypatch):
    """The Python-sort fallback must produce the same order as the engine.

    Sorting ascending there would invert ``retain_latest``: the freshest plan would
    be pruned and the oldest kept.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan_ids = _compile_series(lake, passes=3)  # oldest -> newest

    # Force the fallback by making the engine ordering unavailable.
    import lancedb.query as lq

    monkeypatch.delattr(lq, "ColumnOrdering", raising=False)
    with pytest.warns(RuntimeWarning, match="could not push series ordering"):
        report = lake.curate.prune_row_plans(
            retain_latest=1, older_than=timedelta(seconds=-60)
        )

    newest = plan_ids[-1]
    oldest = plan_ids[0]
    assert newest not in report.pruned_plan_ids
    assert oldest in report.pruned_plan_ids
    assert lake.curate.row_plan(newest).state == rp.STATE_ACTIVE
    assert lake.curate.row_plan(oldest).state == rp.STATE_PRUNED


def test_retention_never_restarts_its_scan_and_double_counts_rank(tmp_path, monkeypatch):
    """A mid-stream read error must fail the pass, not re-read from the top.

    Retention is rank-sensitive: a re-yielded row double-counts ``series_seen``, the
    rank shifts, and the *newest* (rank 0) plan gets pruned while the report claims
    success. The scan must be at-most-once.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan_ids = _compile_series(lake, passes=3)  # oldest -> newest
    newest = plan_ids[-1]

    # A read that fails partway must propagate, never silently restart.
    real_iter = rp._iter_plans_for_retention

    def _failing(lake_arg, *, where_sql, stats):
        for index, row in enumerate(real_iter(lake_arg, where_sql=where_sql, stats=stats)):
            if index == 1:
                raise RuntimeError("transient read error")
            yield row

    monkeypatch.setattr(rp, "_iter_plans_for_retention", _failing)
    with pytest.raises(RuntimeError):
        lake.curate.prune_row_plans(retain_latest=1, older_than=timedelta(seconds=-60))
    monkeypatch.undo()

    # The newest plan must not have been pruned by a partial pass.
    assert lake.curate.row_plan(newest).state == rp.STATE_ACTIVE
    assert len(list(lake.curate.row_plan_targets(newest).iter_targets())) > 0

def test_stream_rows_is_at_most_once(tmp_path, monkeypatch):
    """``_stream_rows`` must never deliver a row twice.

    Its materialized fallback exists for a backend that cannot build or start the
    streaming query. If it also fired *after* rows had been yielded it would re-read
    the table from the top, and every order- or rank-sensitive consumer would
    silently get duplicates: retention's series rank shifts (pruning the wrong
    plan), the listing heap double-counts ``qualifying``, and the chunk reader
    duplicates a target ordinal. So a mid-stream error propagates instead.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=12)
    _compile_series(lake, passes=3)

    # Fails on the SECOND batch, i.e. after rows have already been handed out.
    real_table = type(lake).table

    def _table(self, name):
        handle = real_table(self, name)
        if name != "curation_row_plans":
            return handle

        class _FailsMidStream:
            def __getattr__(self, item):
                return getattr(handle, item)

            def search(self):
                query = handle.search()

                class _Q:
                    def __getattr__(self, item):
                        inner = getattr(query, item)
                        if item in {"select", "where", "order_by"}:
                            return lambda *a, **k: (inner(*a, **k), _Q())[1]
                        return inner

                    def to_batches(self, **kwargs):
                        def _gen():
                            for index, batch in enumerate(query.to_batches(**kwargs)):
                                if index >= 1:
                                    raise RuntimeError("transient mid-stream read error")
                                yield batch

                        return _gen()

                return _Q()

        return _FailsMidStream()

    monkeypatch.setattr(type(lake), "table", _table)
    # One row per batch so the failure lands after a successful yield.
    with pytest.raises(RuntimeError, match="transient mid-stream read error"):
        list(rp._stream_rows(lake, "curation_row_plans", columns=("plan_id",), batch_size=1))


def test_supersession_chains_do_not_rebuild_their_index_per_target(tmp_path, monkeypatch):
    """Chain walking must not be O(decisions) per chain (BUG-01 shape).

    ``_supersession_chain`` rebuilt its whole id index on every call, so a branch
    where every target was re-decided cost O(decisions^2) — and the work was then
    thrown away by the sampling.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "SAMPLE_LIMIT", 5)

    index_builds: list[int] = []
    real_index = curate_module._supersession_index

    def _spy(rows):
        index_builds.append(len(rows))
        return real_index(rows)

    monkeypatch.setattr(curate_module, "_supersession_index", _spy)

    lake = _wide_observation_lake(tmp_path / "robot.lance", count=60)
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-wide",)))
    selection.save_view("wide-review")
    targets = [f"obs-wide-{index:06d}" for index in range(40)]
    # Decide every target twice so each latest decision supersedes an earlier one.
    for reason in ("first", "second"):
        selection.record_decisions(
            view_name="wide-review",
            decision="promote",
            target_grain="observation",
            target_ids=targets,
            reason=reason,
        )
    plan = lake.curate.compile_row_plan(
        view_name="wide-review", target_grain="observation"
    )

    assert plan.report["supersession_chain_count"] == 40
    assert len(plan.report["supersession_chains"]) == 5
    assert plan.report["supersession_chains_truncated"] is True
    # The index is built ONCE per compile, not once per chain.
    assert len(index_builds) == 1, f"index rebuilt {len(index_builds)} times"
    # And each chain is depth-capped.
    for chain in plan.report["supersession_chains"]:
        assert len(chain["chain"]) <= curate_module._ROW_PLAN_MAX_CHAIN_DEPTH


def test_membership_transform_ids_are_capped_everywhere_they_are_persisted(
    tmp_path, monkeypatch
):
    """This list grows with review batches, so it must be capped like the id lists.

    It was copied verbatim into the compile report (hence
    ``transform_runs.params``) and into the frozen artifact metadata.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "SAMPLE_LIMIT", 3)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-wide",)))
    selection.save_view("wide-review")
    for index in range(8):
        selection.record_decisions(
            view_name="wide-review",
            decision="exclude",
            target_grain="observation",
            target_ids=[f"obs-wide-{index:06d}"],
            reason=f"batch-{index}",
        )
    plan = lake.curate.compile_row_plan(
        view_name="wide-review", target_grain="observation", freeze=True
    )

    assert plan.report["membership_transform_count"] == 8
    assert len(plan.report["membership_transform_ids"]) == 3
    assert plan.report["membership_transform_ids_truncated"] is True
    # The identity payload carries a digest, not the list.
    assert "membership_transform_ids_digest" in plan.report
    assert "membership_transform_ids" not in {
        key for key in plan.report if key == "membership_transform_ids_full"
    }

    artifact = next(
        row
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
        if row["artifact_id"] == plan.artifact_id
    )
    metadata = {item["key"]: item["value"] for item in artifact["metadata"]}
    assert metadata["membership_transform_count"] == "8"
    assert metadata["membership_transform_ids_truncated"] == "true"
    # The persisted params carry the capped list too.
    params = json.loads(
        next(
            row["params"]
            for row in lake.table("transform_runs").to_arrow().to_pylist()
            if row["transform_id"] == plan.transform_id
        )
    )
    assert len(params["membership_transform_ids"]) == 3


def test_an_unverifiable_chunk_delete_fails_closed(tmp_path, monkeypatch):
    """An unverifiable delete is not a verified one.

    Reading a failed post-delete count as "all rows gone" would mark the plan pruned
    with its chunks possibly intact — the exact stranding the verification exists to
    prevent.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan_ids = _compile_series(lake, passes=2)
    victim = plan_ids[0]

    real_table = type(lake).table
    calls = {"n": 0}

    def _table(self, name):
        handle = real_table(self, name)
        if name == "curation_row_plan_chunks":
            class _UncountableAfterDelete:
                def __getattr__(self, item):
                    return getattr(handle, item)

                def count_rows(self, *a, **k):
                    calls["n"] += 1
                    if calls["n"] > 1:  # the post-delete verification read
                        raise RuntimeError("count unavailable")
                    return handle.count_rows(*a, **k)

                def delete(self, *a, **k):
                    return None  # pretend the delete succeeded, delete nothing

            return _UncountableAfterDelete()
        return handle

    monkeypatch.setattr(type(lake), "table", _table)
    report = lake.curate.prune_row_plans(
        retain_latest=1, older_than=timedelta(seconds=-60)
    )
    monkeypatch.undo()

    assert victim not in report.pruned_plan_ids
    assert victim in report.delete_failed_plan_ids
    assert lake.curate.row_plan(victim).state != rp.STATE_PRUNED
    assert list(lake.curate.row_plan_targets(victim).iter_targets())


def test_a_failed_chunk_delete_leaves_the_header_reclaimable(tmp_path, monkeypatch):
    """Never flip a header to ``pruned`` when its chunks are still there.

    A pruned header is skipped by the retention scan, by validation, and by orphan
    compaction (its header exists) -- so the rows would be stranded forever and the
    reported ``chunks_deleted`` would be fiction.
    """
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan_ids = _compile_series(lake, passes=2)
    victim = plan_ids[0]

    real_table = type(lake).table

    def _table(self, name):
        handle = real_table(self, name)
        if name == "curation_row_plan_chunks":
            class _NoDelete:
                def __getattr__(self, item):
                    return getattr(handle, item)

                def delete(self, *a, **k):
                    raise RuntimeError("delete unavailable")

            return _NoDelete()
        return handle

    monkeypatch.setattr(type(lake), "table", _table)
    report = lake.curate.prune_row_plans(
        retain_latest=1, older_than=timedelta(seconds=-60)
    )
    monkeypatch.undo()

    assert victim not in report.pruned_plan_ids
    assert report.chunks_deleted == 0
    assert lake.curate.row_plan(victim).state != rp.STATE_PRUNED
    # Still readable, and a later pass can retry it.
    assert list(lake.curate.row_plan_targets(victim).iter_targets())
    retry = lake.curate.prune_row_plans(
        retain_latest=1, older_than=timedelta(seconds=-60)
    )
    assert victim in retry.pruned_plan_ids
    assert retry.chunks_deleted > 0


def test_retention_reads_only_promoted_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=12)
    _compile_series(lake, passes=2)

    assert "summary_json" not in rp._RETENTION_COLUMNS
    report = lake.curate.prune_row_plans(retain_latest=1, dry_run=True)
    assert report.dry_run is True
    assert report.scanned_count >= 2
    # A dry run writes nothing.
    assert all(
        entry.state == rp.STATE_ACTIVE for entry in lake.curate.row_plans().records
    )


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def test_cli_row_plan_surfaces(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    uri = str(tmp_path / "robot.lance")
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    plan = _compile_wide_plan(lake)

    listed = runner.invoke(app, ["curate", "row-plans", "--lake", uri, "--json"])
    assert listed.exit_code == 0, listed.output
    # Pages stream as JSONL so `--all` never holds the catalog in memory.
    lines = [json.loads(line) for line in listed.stdout.strip().splitlines()]
    assert lines[0]["lake"] == uri
    assert lines[1]["records"][0]["plan_id"] == plan.plan_id
    assert lines[-1]["page_count"] == 1

    shown = runner.invoke(
        app, ["curate", "row-plan", "--lake", uri, "--plan", plan.plan_id, "--json"]
    )
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.stdout)["target_count"] == 40

    paged = runner.invoke(
        app,
        [
            "curate",
            "row-plan-targets",
            "--lake",
            uri,
            "--plan",
            plan.plan_id,
            "--page-size",
            "9",
            "--all",
            "--json",
        ],
    )
    assert paged.exit_code == 0, paged.output
    target_lines = [json.loads(line) for line in paged.stdout.strip().splitlines()]
    assert target_lines[0]["plan"]["target_count"] == 40
    pages = [line for line in target_lines if "targets" in line]
    assert sum(page["returned"] for page in pages) == 40
    assert pages[-1]["has_more"] is False
    assert target_lines[-1]["page_count"] == 5  # ceil(40 / 9)
    # Every target appears exactly once, in ordinal order, across the streamed pages.
    ordinals = [target["ordinal"] for page in pages for target in page["targets"]]
    assert ordinals == list(range(40))

    validated = runner.invoke(
        app, ["curate", "validate-row-plans", "--lake", uri, "--json"]
    )
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.stdout)["ok"] is True

    pruned = runner.invoke(
        app, ["curate", "prune-row-plans", "--lake", uri, "--dry-run", "--json"]
    )
    assert pruned.exit_code == 0, pruned.output
    assert json.loads(pruned.stdout)["dry_run"] is True

    compacted = runner.invoke(
        app, ["curate", "compact-row-plans", "--lake", uri, "--json"]
    )
    assert compacted.exit_code == 0, compacted.output
    assert json.loads(compacted.stdout)["orphan_plan_count"] == 0


def test_cli_compile_row_plan_reports_exact_counts_not_sample_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "INLINE_TARGET_LIMIT", 4)
    monkeypatch.setattr(rp, "CHUNK_SIZE", 8)
    monkeypatch.setattr(rp, "MATERIALIZE_SOFT_LIMIT", 8)
    uri = str(tmp_path / "robot.lance")
    lake = _wide_observation_lake(tmp_path / "robot.lance", count=40)
    selection = lake.curate.workbench(scope=CurationScope(scenario_ids=("scn-wide",)))
    selection.save_view("wide-review")

    result = runner.invoke(
        app,
        ["curate", "compile-row-plan", "--lake", uri, "--view", "wide-review"],
    )
    assert result.exit_code == 0, result.output
    # ``rows`` must be the real target count even though the ids are not
    # materialized on the returned object.
    assert "rows: 40" in result.stdout
    assert "storage: chunked" in result.stdout
