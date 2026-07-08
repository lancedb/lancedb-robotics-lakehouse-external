"""Handle-resolution catalog and ambiguity diagnostics (backlog 0102).

``lake.lineage.resolve(handle)`` is the read-only diagnostic surface layered over
the v1 resolver: it reports whether a handle resolves exactly, is ambiguous across
several distinct entities, is only known canonically (needs a graph refresh), or is
unknown, and it does so over bounded/indexed reads rather than whole-table scans.
"""

import pytest

from lancedb_robotics import lineage as lineage_mod
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows


def _snapshot_lake(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    scenarios = sorted(
        lake.table("scenarios").to_arrow().to_pylist(),
        key=lambda row: (row["start_time_ns"], row["scenario_id"]),
    )
    return lake, [row["scenario_id"] for row in scenarios]


def test_source_uri_resolves_to_multiple_coordinate_roots(tmp_path, fixtures_dir):
    """A source URI fans out to many source coordinates: resolved, multi-root."""
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    lake.lineage.refresh_graph()
    source_uri = str((fixtures_dir / "sample.mcap").resolve())

    resolution = lake.lineage.resolve(source_uri)

    assert resolution.status == "resolved"
    assert resolution.multi_root is True
    assert resolution.root_count > 1
    assert {candidate.kind for candidate in resolution.candidates} == {"source"}
    # Every root carries the source URI it matched on.
    assert all(candidate.source_uri == source_uri for candidate in resolution.candidates)
    # The narrative's ready-to-run impact command is emitted with the resolved kind.
    assert any(
        command.startswith("lancedb-robotics lineage impact") and "--kind source" in command
        for command in resolution.suggested_commands
    )
    assert resolution.graph_fresh is True


def test_two_snapshots_sharing_a_name_report_ambiguity(tmp_path, fixtures_dir):
    """Two distinct snapshots sharing a name resolve to an ambiguity, not a silent pick."""
    lake, scenario_ids = _snapshot_lake(tmp_path, fixtures_dir)
    first = create_snapshot(lake, name="shared", tag="shared", scenario_ids=scenario_ids)
    second = create_snapshot(lake, name="shared", tag="shared", scenario_ids=scenario_ids[:1])
    assert first.dataset_id != second.dataset_id
    lake.lineage.refresh_graph()

    resolution = lake.lineage.resolve("shared")

    assert resolution.status == "ambiguous"
    assert resolution.root_count == 2
    assert set(resolution.artifact_ids) == {
        lineage_mod.snapshot_artifact_id(first.dataset_id),
        lineage_mod.snapshot_artifact_id(second.dataset_id),
    }
    assert {candidate.kind for candidate in resolution.candidates} == {"dataset-snapshot"}
    # Even when kind/version cannot disambiguate, an exact-artifact-id path is offered.
    artifact_id_hints = [
        hint for hint in resolution.disambiguation_hints if hint["flag"] == "artifact-id"
    ]
    assert artifact_id_hints and set(artifact_id_hints[0]["values"]) == set(resolution.artifact_ids)
    assert all(
        candidate.artifact_id in resolution.artifact_ids for candidate in resolution.candidates
    )


def test_exact_artifact_id_disambiguates_the_ambiguous_handle(tmp_path, fixtures_dir):
    lake, scenario_ids = _snapshot_lake(tmp_path, fixtures_dir)
    first = create_snapshot(lake, name="shared", tag="shared", scenario_ids=scenario_ids)
    create_snapshot(lake, name="shared", tag="shared", scenario_ids=scenario_ids[:1])
    lake.lineage.refresh_graph()

    resolution = lake.lineage.resolve(lineage_mod.snapshot_artifact_id(first.dataset_id))

    assert resolution.status == "resolved"
    assert resolution.root_count == 1
    assert resolution.artifact_ids == (lineage_mod.snapshot_artifact_id(first.dataset_id),)


def test_row_appended_after_refresh_reports_stale_graph(tmp_path, fixtures_dir):
    """A run appended after the last refresh is known canonically but stale in the graph."""
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    lake.lineage.refresh_graph()
    # Append a second recording -> new run row + bumped source-table versions, no refresh.
    ingest_mcap(lake, fixtures_dir / "records.mcap")
    new_run_id = lake.table("runs").to_arrow().to_pylist()[-1]["run_id"]

    resolution = lake.lineage.resolve(new_run_id, kind="run")

    assert resolution.status == "stale"
    assert resolution.graph_fresh is False
    assert "runs" in resolution.stale_tables
    assert resolution.pending_refresh_artifact_ids  # a refresh would materialize it
    assert any("refresh" in command for command in resolution.suggested_commands)
    assert resolution.message and "refresh_graph" in resolution.message


def test_unknown_handle_is_reported_not_raised(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    lake.lineage.refresh_graph()

    resolution = lake.lineage.resolve("no-such-handle-anywhere")

    assert resolution.status == "unknown"
    assert resolution.root_count == 0
    assert resolution.candidates == ()


def test_unsupported_kind_is_reported(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")

    resolution = lake.lineage.resolve("anything", kind="banana")

    assert resolution.status == "unsupported-kind"
    assert resolution.message and "banana" in resolution.message


def test_empty_handle_raises(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    with pytest.raises(lineage_mod.LineageError):
        lake.lineage.resolve("   ")


def test_resolution_is_read_only(tmp_path, fixtures_dir):
    """resolve() never mutates the lineage graph (no implicit refresh/record)."""
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    lake.lineage.refresh_graph()
    source_uri = str((fixtures_dir / "sample.mcap").resolve())
    versions_before = {
        table: lake.table(table).version
        for table in ("lineage_artifacts", "lineage_edges", "lineage_executions")
    }

    lake.lineage.resolve(source_uri)
    lake.lineage.resolve("unknown-handle")
    lake.lineage.resolve("anything", kind="banana")

    versions_after = {
        table: lake.table(table).version
        for table in ("lineage_artifacts", "lineage_edges", "lineage_executions")
    }
    assert versions_before == versions_after


def test_source_uri_resolution_is_bounded(tmp_path, monkeypatch):
    """Resolving one URI over many unrelated artifacts reads only matching rows.

    A large synthetic graph carries many source URIs (each with several coordinates)
    plus many non-source artifacts. Resolving a single URI must materialize only the
    rows that match, never the whole ``lineage_artifacts`` table, and must never leak
    an unrelated kind into the candidates.
    """
    lake = Lake.init(tmp_path / "robot.lance")
    n_uris, coords_per_uri = 40, 4
    for u in range(n_uris):
        uri = f"s3://bucket/run-{u:03d}.mcap"
        for c in range(coords_per_uri):
            lake.lineage.record_artifact(
                artifact_id=f"lancedb-robotics:source:{u:03d}:{c}",
                kind="source",
                source_uri=uri,
                metadata={"channel": f"/topic-{c}", "offset": str(c)},
            )
    # Unrelated artifacts a naive full scan would touch.
    for r in range(200):
        lake.lineage.record_artifact(
            artifact_id=f"lancedb-robotics:row:obs-{r}",
            kind="row",
            table_name="observations",
            row_grain="observations",
            row_ids=[f"obs-{r}"],
        )
    total_artifacts = lake.table("lineage_artifacts").count_rows()
    target = "s3://bucket/run-017.mcap"

    rows_read = {"n": 0}
    original_query = lineage_mod._query_table_by_column
    original_fetch = lineage_mod._fetch_rows_by_id_in

    def spy_query(*args, **kwargs):
        rows = original_query(*args, **kwargs)
        rows_read["n"] += len(rows)
        return rows

    def spy_fetch(*args, **kwargs):
        rows = original_fetch(*args, **kwargs)
        rows_read["n"] += len(rows)
        return rows

    # Guard: the full whole-table load must not be used on the resolve path.
    def forbidden_scan(self, *args, **kwargs):  # pragma: no cover - only fires on regression
        raise AssertionError("resolve() must not load the whole lineage_artifacts table")

    monkeypatch.setattr(lineage_mod, "_query_table_by_column", spy_query)
    monkeypatch.setattr(lineage_mod, "_fetch_rows_by_id_in", spy_fetch)

    resolution = lake.lineage.resolve(target)

    assert resolution.status == "resolved"
    assert resolution.root_count == coords_per_uri
    assert {candidate.kind for candidate in resolution.candidates} == {"source"}
    assert all(candidate.source_uri == target for candidate in resolution.candidates)
    # Bounded: rows materialized are proportional to the matched coordinates, far below
    # the full table (matched on source_uri and name -> ~2x coords, plus tiny probes).
    assert rows_read["n"] < total_artifacts
    assert rows_read["n"] <= coords_per_uri * 4
