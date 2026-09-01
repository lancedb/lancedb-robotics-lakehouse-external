"""Backlog 0143: curation replay retention protection and version-read conformance.

Covers the four acceptance criteria:
- maintenance protects (tags) every curation version an active snapshot pins;
- replay fails with actionable diagnostics when a pinned version is pruned;
- chunked saved-view replay reads chunk rows at the snapshot-pinned version;
- backend conformance records supported / capability-gated / unavailable.
"""

import json

import pyarrow as pa
import pytest
from test_curate import _build_curation_lake, _snapshot_row
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.connections import LakeCapabilities, LakeConnectionSpec
from lancedb_robotics.curate import CurationError
from lancedb_robotics.curation_replay_retention import (
    CURATION_REPLAY_TABLES,
    PIN_PROTECTED,
    PIN_PRUNED,
    PIN_UNPROTECTED,
    REPLAY_CAPABILITY_GATED,
    REPLAY_SUPPORTED,
    REPLAY_UNAVAILABLE,
    curation_replay_conformance,
    curation_replay_readiness,
)
from lancedb_robotics.maintenance import _PIN_TAG_PREFIX as MAINT_PIN_TAG_PREFIX
from lancedb_robotics.maintenance import maintain_lake
from lancedb_robotics.schemas import DATASET_SNAPSHOTS_SCHEMA

runner = CliRunner()


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _chunked_view_snapshot(lake, *, view_name="chunked-view", snapshot="chunked-snap"):
    """Save a chunked saved view and freeze a snapshot pinning it."""
    selection = lake.curate.workbench()
    selection.save_view(view_name, inline_scenario_limit=2, membership_chunk_size=2)
    expected = tuple(selection.scenario_ids)
    curated = selection.apply_decisions(view_name=view_name)
    curated.snapshot(name=snapshot, split_by="scenario")
    return view_name, snapshot, expected


def _spec_lake(spec):
    class _SpecLake:
        connection_spec = spec
        uri = "mem://conformance"

    return _SpecLake()


def _remote_db_spec(**control_plane):
    return LakeConnectionSpec(
        kind="lancedb_remote_db",
        uri="db://robotics",
        display_uri="db://robotics",
        capabilities=LakeCapabilities(
            server_side_query=True, blob_fetch_remote=True, **control_plane
        ),
    )


def _managed_namespace_spec():
    return LakeConnectionSpec(
        kind="rest_namespace_lancedb",
        uri=None,
        display_uri="namespace://robotics",
        capabilities=LakeCapabilities(
            direct_object_io=True,
            namespace_resolution=True,
            namespace_managed_versioning=True,
            table_versioning=True,
        ),
    )


# --------------------------------------------------------------------------- #
# AC4 -- backend version-read conformance
# --------------------------------------------------------------------------- #


def test_pin_tag_prefix_matches_maintenance():
    # The readiness check recognises a version as protected by the managed pin
    # tag maintenance writes; if the prefixes drift, protection silently stops
    # being detected. This guardrail fails the moment they diverge.
    from lancedb_robotics.curation_replay_retention import _PIN_TAG_PREFIX

    assert _PIN_TAG_PREFIX == MAINT_PIN_TAG_PREFIX


def test_conformance_local_and_object_store_supported(tmp_path):
    from lancedb_robotics.connections import resolve_lake_connection

    local = curation_replay_conformance(_spec_lake(resolve_lake_connection("/tmp/l.lance")))
    assert local.status == REPLAY_SUPPORTED
    assert local.advertised is True

    obj = curation_replay_conformance(_spec_lake(resolve_lake_connection("s3://b/l.lance")))
    assert obj.status == REPLAY_SUPPORTED


def test_conformance_remote_db_is_capability_gated():
    conformance = curation_replay_conformance(_spec_lake(_remote_db_spec()))
    assert conformance.status == REPLAY_CAPABILITY_GATED
    assert conformance.advertised is False
    assert conformance.fallbacks  # names a fallback plane
    assert "object_store_lancedb_oss" in conformance.fallbacks
    assert conformance.suggested_action


def test_conformance_remote_db_supported_when_versioning_advertised():
    conformance = curation_replay_conformance(_spec_lake(_remote_db_spec(table_versioning=True)))
    assert conformance.status == REPLAY_SUPPORTED


def test_conformance_namespace_managed_versioning_is_unavailable():
    conformance = curation_replay_conformance(_spec_lake(_managed_namespace_spec()))
    assert conformance.status == REPLAY_UNAVAILABLE
    assert conformance.namespace_managed_versioning is True
    assert conformance.suggested_action


def test_conformance_unclassified_backend_supported():
    assert curation_replay_conformance(_spec_lake(None)).status == REPLAY_SUPPORTED


def test_readiness_not_falsely_ready_when_versioning_advertised_but_no_direct_io(tmp_path):
    # A db:// lake that advertises table_versioning is conformance-"supported",
    # but without direct object IO the pins cannot be inspected on disk. The
    # verdict must NOT be a bare ready=True -- that would claim a guarantee we
    # never checked (SKILLS.md no-silent-degrade). Regression for the scale-review
    # HIGH finding.
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _chunked_view_snapshot(lake)
    lake.connection_spec = _remote_db_spec(table_versioning=True)
    lake.capabilities = lake.connection_spec.capabilities

    report = curation_replay_readiness(lake)
    assert report.backend.status == REPLAY_SUPPORTED  # capability says supported
    assert report.ready is False  # ...but we could not verify the pins
    assert report.status == "backend-gated"
    assert report.pins and all(pin.status == "backend-gated" for pin in report.pins)
    assert report.suggested_actions


def test_scoped_readiness_picks_latest_snapshot_row_by_created_at(tmp_path):
    # Two rows share a snapshot name; created_at order and dataset_id order
    # disagree. Scoped readiness must follow `_latest_snapshot_row` (latest by
    # created_at), not lexical dataset_id. Regression for the scale-review MEDIUM
    # finding (created_at was projected out).
    from datetime import UTC, datetime

    lake = _build_curation_lake(tmp_path / "robot.lance")
    snapshots = lake.table("dataset_snapshots")

    def _row(dataset_id, created_at, memberships_version):
        return {
            "dataset_id": dataset_id,
            "name": "dup-name",
            "kind": "snapshot",
            "query_spec": "{}",
            "table_versions": [
                {"table": "curation_memberships", "version": memberships_version, "tag": ""}
            ],
            "tag": "",
            "split": "",
            "balance_report": "",
            "coverage_report": "",
            "created_by": "test",
            "transform_id": "",
            "created_at": created_at,
        }

    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    later = datetime(2026, 6, 1, tzinfo=UTC)
    # dataset_id lexical order (zzz > aaa) is the OPPOSITE of time order.
    snapshots.add(
        pa.Table.from_pylist(
            [
                _row("zzz-older", earlier, 111),
                _row("aaa-newer", later, 222),
            ],
            schema=DATASET_SNAPSHOTS_SCHEMA,
        )
    )

    report = curation_replay_readiness(lake, snapshot_name="dup-name")
    versions = {pin.version for pin in report.pins if pin.table == "curation_memberships"}
    assert 222 in versions  # latest-by-created_at row's pin
    assert 111 not in versions  # not the lexically-larger dataset_id row


def test_readiness_backend_gated_does_not_touch_dataset(tmp_path):
    # On a capability-gated backend the on-disk state cannot be inspected; the
    # readiness report is backend-gated and never drops to LanceDataset access.
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _chunked_view_snapshot(lake)
    lake.connection_spec = _remote_db_spec()
    lake.capabilities = lake.connection_spec.capabilities

    report = curation_replay_readiness(lake)
    assert report.backend.status == REPLAY_CAPABILITY_GATED
    assert report.ready is False
    assert report.status == "backend-gated"
    assert report.pins  # pins listed
    assert all(pin.status == "backend-gated" for pin in report.pins)
    assert report.suggested_actions


# --------------------------------------------------------------------------- #
# AC1 -- maintenance protects the pinned curation versions
# --------------------------------------------------------------------------- #


def test_maintenance_protects_and_readiness_confirms(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _view, snapshot, _expected = _chunked_view_snapshot(lake)

    # Advance the curation tables *after* the snapshot so the pinned versions are
    # no longer the current version and are not yet tagged -> at risk.
    workbench = lake.curate.workbench()
    workbench.save_view("later-view")
    workbench.record_decisions(
        view_name="later-view", decision="exclude", scenario_ids=["scn-neighbor"]
    )

    before = curation_replay_readiness(lake)
    assert before.backend.status == REPLAY_SUPPORTED
    assert not before.ready
    assert before.status == "at-risk"
    assert any(pin.status == PIN_UNPROTECTED for pin in before.pins)
    assert any("lake maintain" in action for action in before.suggested_actions)

    # Maintenance tags every snapshot-pinned version (including curation tables).
    result = maintain_lake(lake, cleanup_older_than=None)
    assert result.curation_replay_retention is not None
    assert result.curation_replay_retention["ready"] is True

    after = curation_replay_readiness(lake)
    assert after.ready is True
    assert after.status == "ready"
    assert all(pin.status == PIN_PROTECTED for pin in after.pins)

    # The pinned curation versions carry the managed pin tag on their tables.
    pinned = {
        (entry["table"], int(entry["version"]))
        for entry in _snapshot_row(lake, snapshot)["table_versions"]
        if entry["table"] in CURATION_REPLAY_TABLES
    }
    assert pinned  # snapshot recorded curation-table pins
    for table, version in pinned:
        tags = set(lake.table(table).to_lance().tags.list())
        assert f"{MAINT_PIN_TAG_PREFIX}{version}" in tags


def test_maintenance_report_absent_when_no_curation_tables_selected(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _chunked_view_snapshot(lake)
    report = maintain_lake(lake, tables=("scenarios",), cleanup_older_than=None)
    assert report.curation_replay_retention is None


# --------------------------------------------------------------------------- #
# AC2 -- replay fails with actionable diagnostics when a pin is pruned
# --------------------------------------------------------------------------- #


def test_pruned_pin_is_flagged_and_replay_fails_with_guidance(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _view, snapshot, _expected = _chunked_view_snapshot(lake)

    # Rewrite the snapshot to pin a curation_memberships version that never
    # existed -- the operational equivalent of a pruned replay pin.
    row = _snapshot_row(lake, snapshot)
    mutated = dict(row)
    mutated["table_versions"] = [
        {
            **entry,
            "version": 999_999
            if entry["table"] == "curation_memberships"
            else entry["version"],
        }
        for entry in row["table_versions"]
    ]
    snapshots = lake.table("dataset_snapshots")
    snapshots.delete(f"dataset_id = '{row['dataset_id']}'")
    snapshots.add(pa.Table.from_pylist([mutated], schema=DATASET_SNAPSHOTS_SCHEMA))

    readiness = curation_replay_readiness(lake, snapshot_name=snapshot)
    pruned = [pin for pin in readiness.pins if pin.status == PIN_PRUNED]
    assert any(pin.table == "curation_memberships" and pin.version == 999_999 for pin in pruned)
    assert readiness.ready is False
    assert any("pruned" in action or "non-replayable" in action for action in readiness.suggested_actions)

    # Replay itself fails with an actionable, remediation-bearing error.
    with pytest.raises(CurationError) as excinfo:
        lake.curate.resolve_membership(snapshot_name=snapshot, target_ids=["scn-neighbor"])
    message = str(excinfo.value)
    assert "curation_memberships@999999" in message
    assert "lake maintain" in message


# --------------------------------------------------------------------------- #
# AC3 -- chunked saved-view replay reads chunk rows at the pinned version
# --------------------------------------------------------------------------- #


def test_chunked_replay_reads_pinned_chunk_version_not_live(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    view_name, snapshot, expected = _chunked_view_snapshot(lake)

    # Baseline: at the snapshot the chunked view resolves to its full membership.
    view_id = next(
        row["view_id"]
        for row in lake.table("curation_views").to_arrow().to_pylist()
        if row["name"] == view_name
    )
    baseline = lake.curate.resolve_membership(view_name=view_name, snapshot_name=snapshot)
    assert baseline.view is not None
    assert baseline.view.membership_storage == "chunked"
    assert tuple(baseline.view.scenario_ids) == expected

    pinned_chunk_version = next(
        entry["version"]
        for entry in baseline.report["read_table_versions"]
        if entry["table"] == "curation_view_membership_chunks"
    )

    # Mutate the LIVE chunk table: drop this view's chunk rows entirely. The
    # pinned version still holds them; the current version no longer does.
    chunk_table = lake.table("curation_view_membership_chunks")
    chunk_table.delete(f"view_id = '{view_id}'")
    assert not [
        r
        for r in chunk_table.to_arrow().to_pylist()
        if r["view_id"] == view_id
    ]

    # Replay still resolves the pinned membership (reads chunks at the pinned
    # version, not the emptied live table). Without the 0143 fix this raised
    # "expected N chunked scenarios, found 0".
    replayed = lake.curate.resolve_membership(view_name=view_name, snapshot_name=snapshot)
    assert tuple(replayed.view.scenario_ids) == expected
    assert replayed.view.membership_count == len(expected)
    # Mechanism: the audit envelope records the pinned chunk version was read.
    assert any(
        entry["table"] == "curation_view_membership_chunks"
        and entry["version"] == pinned_chunk_version
        for entry in replayed.report["read_table_versions"]
    )


# --------------------------------------------------------------------------- #
# trace-membership replay-readiness hints + CLI
# --------------------------------------------------------------------------- #


def test_trace_membership_carries_replay_readiness(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")

    workbench = lake.curate.workbench()
    workbench.dedup(near_duplicate_threshold=0.999, view_name="dedup-audit")
    curated = workbench.apply_decisions(view_name="dedup-audit")
    curated.snapshot(name="trace-snap", split_by="scenario")

    trace = lake.curate.trace_membership("trace-snap", "scn-duplicate")
    readiness = trace.report["replay_readiness"]
    assert readiness["backend"]["status"] == REPLAY_SUPPORTED
    assert readiness["snapshots_checked"] == 1
    assert readiness["schema_version"] == "lancedb-robotics/curation-replay-readiness/v1"


def test_cli_replay_readiness(tmp_path):
    lake = _build_curation_lake(tmp_path / "robot.lance")
    _chunked_view_snapshot(lake)
    lake_path = str(tmp_path / "robot.lance")

    result = runner.invoke(
        app, ["curate", "replay-readiness", "--lake", lake_path, "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"]["status"] == REPLAY_SUPPORTED
    assert payload["schema_version"] == "lancedb-robotics/curation-replay-readiness/v1"
    assert "pins" in payload
