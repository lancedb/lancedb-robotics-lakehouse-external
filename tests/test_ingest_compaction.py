"""Ingest-time compaction + version retention (backlog 0180 / BUG-14).

Each streaming ``_flush`` is one Lance fragment + one version, so a freshly
ingested ``observations`` grain is born at ~``batch_size`` rows/fragment -- 3-4
orders of magnitude below Lance's healthy ~1M-row target, which imposes a ~79x
per-row scan tax on every later read. Ingest now compacts the grain up to a
healthy fragment size and snapshot-safely prunes the per-flush version churn, on
by default. These tests pin that behavior and the invariant that it never changes
which rows land or their contents (compaction is a pure-layout operation).
"""

import json

import pytest

from lancedb_robotics.ingest import _finalize_ingest, ingest_mcap
from lancedb_robotics.lake import Lake

_PROJECTION = (
    "observation_id",
    "topic",
    "raw_sequence",
    "timestamp_ns",
    "modality",
    "payload_json",
    "state_vector",
    "action_vector",
    "created_at",
)


@pytest.fixture
def sample_mcap(fixtures_dir):
    return fixtures_dir / "sample.mcap"


def _frag_count(lake):
    return len(lake.table("observations").to_lance().get_fragments())


def _version(lake):
    return int(lake.table("observations").version)


def test_compact_off_leaves_one_fragment_per_flush(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "off.lance")
    report = ingest_mcap(
        lake, sample_mcap, batch_size=1, compact=False, prune_versions=False, index_predicates=False
    )
    # batch_size=1 -> one `table.add` (one fragment) per row; nothing compacts/indexes.
    assert _frag_count(lake) == report.message_count == 5
    assert report.compaction is None


def test_default_ingest_compacts_grain_to_few_fragments(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "on.lance")
    report = ingest_mcap(lake, sample_mcap, batch_size=1)  # compact defaults on
    # 5 born-fragments collapse to a single healthy fragment.
    assert _frag_count(lake) == 1
    assert lake.table("observations").count_rows() == 5
    assert report.compaction is not None
    assert report.compaction["fragments_before"] == 5
    assert report.compaction["fragments_after"] == 1


def test_compaction_is_pure_layout_rows_unchanged(tmp_path, sample_mcap):
    """Compacting a lake must not change any observation row (incl. created_at)."""
    lake = Lake.init(tmp_path / "l.lance")
    ingest_mcap(
        lake, sample_mcap, batch_size=1, compact=False, prune_versions=False, index_predicates=False
    )
    before = sorted(
        lake.table("observations").to_arrow().to_pylist(), key=lambda r: r["observation_id"]
    )
    assert _frag_count(lake) == 5
    _finalize_ingest(lake, compact=True, prune_versions=True, retain_versions=2, created_by="t")
    reopened = Lake.open(tmp_path / "l.lance")
    after = sorted(
        reopened.table("observations").to_arrow().to_pylist(), key=lambda r: r["observation_id"]
    )
    assert _frag_count(reopened) == 1
    assert before == after  # byte-identical, including created_at


def test_batch_size_independent_of_compaction(tmp_path, sample_mcap):
    """With compaction on, batch size still never changes which rows land."""

    def rows(name, batch_size):
        lake = Lake.init(tmp_path / name)
        ingest_mcap(lake, sample_mcap, batch_size=batch_size)
        out = sorted(
            lake.table("observations").to_arrow().to_pylist(), key=lambda r: r["observation_id"]
        )
        # created_at differs across independent ingests; project it out.
        return [{k: r[k] for k in _PROJECTION if k != "created_at"} for r in out]

    assert rows("a.lance", 1) == rows("b.lance", 1000)


def test_default_ingest_bounds_version_churn(tmp_path, sample_mcap):
    """Per-flush versions are pruned; a compact-off ingest keeps them all."""
    off = Lake.init(tmp_path / "off.lance")
    ingest_mcap(
        off, sample_mcap, batch_size=1, compact=False, prune_versions=False, index_predicates=False
    )

    on = Lake.init(tmp_path / "on.lance")
    ingest_mcap(on, sample_mcap, batch_size=1)  # compact + prune + index default on

    # The unpruned lake accumulates a version per flush; the pruned lake is bounded.
    assert len(on.table("observations").to_lance().versions()) < len(
        off.table("observations").to_lance().versions()
    )


def test_compaction_records_maintenance_transform(tmp_path, sample_mcap):
    lake = Lake.init(tmp_path / "l.lance")
    ingest_mcap(lake, sample_mcap, batch_size=1)
    maint = [
        t
        for t in lake.table("transform_runs").to_arrow().to_pylist()
        if t["kind"] == "maintenance"
    ]
    assert len(maint) == 1
    params = json.loads(maint[0]["params"])
    obs = params["tables"]["observations"]
    assert obs["fragments_before"] == 5
    assert obs["fragments_after"] == 1


def test_compaction_stays_incremental_across_runs(tmp_path, sample_mcap, fixtures_dir):
    """A second ingest compacts only its own fragments; prior runs stay compact."""
    lake = Lake.init(tmp_path / "l.lance")
    ingest_mcap(lake, sample_mcap, batch_size=1)
    assert _frag_count(lake) == 1
    ingest_mcap(lake, fixtures_dir / "records.mcap", batch_size=1)
    # Both runs' rows remain, and the table stays at a small (healthy) fragment count.
    assert _frag_count(lake) <= 2
