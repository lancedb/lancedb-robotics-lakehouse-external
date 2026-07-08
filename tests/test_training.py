"""Snapshot preview loader tests (backlog 0010)."""

import pytest

from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.enrich import enrich_scenarios
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.training import (
    TrainingError,
    load_snapshot_preview,
    to_torch_dataset,
    torch_available,
)


def _snapshot_lake(path, fixtures_dir, *, name="demo-v1"):
    lake = Lake.init(path)
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=50_000_000)
    enrich_scenarios(lake)
    ids = sorted(r["scenario_id"] for r in lake.table("scenarios").to_arrow().to_pylist())
    create_snapshot(lake, name=name, scenario_ids=ids)
    return lake


@pytest.fixture
def lake(tmp_path, fixtures_dir):
    return _snapshot_lake(tmp_path / "robot.lance", fixtures_dir)


# --- deterministic preview --------------------------------------------------


def test_preview_returns_samples_for_the_snapshot(lake):
    preview = load_snapshot_preview(lake, "demo-v1")

    assert preview.name == "demo-v1"
    assert preview.dataset_id.startswith("ds-")
    assert preview.total_scenarios == 4
    assert preview.samples  # non-empty batch
    sample = preview.samples[0]
    assert sample["scenario_id"].startswith("scn-")
    assert sample["split"] in {"train", "val", "test"}


def test_preview_is_deterministic_for_fixed_inputs(tmp_path, fixtures_dir):
    lake_a = _snapshot_lake(tmp_path / "a.lance", fixtures_dir)
    lake_b = _snapshot_lake(tmp_path / "b.lance", fixtures_dir)

    samples_a = load_snapshot_preview(lake_a, "demo-v1").samples
    samples_b = load_snapshot_preview(lake_b, "demo-v1").samples
    assert samples_a == samples_b


def test_preview_batch_size_caps_samples(lake):
    preview = load_snapshot_preview(lake, "demo-v1", batch_size=2)
    assert len(preview.samples) == 2
    # Stable, sorted-by-id batch.
    ids = [s["scenario_id"] for s in preview.samples]
    assert ids == sorted(ids)


def test_preview_column_projection(lake):
    preview = load_snapshot_preview(lake, "demo-v1", columns=["scenario_id", "split"])
    assert preview.columns == ("scenario_id", "split")
    for sample in preview.samples:
        assert set(sample) == {"scenario_id", "split"}


def test_preview_includes_embedding_vector_by_default(lake):
    preview = load_snapshot_preview(lake, "demo-v1")
    assert "embedding" in preview.columns
    assert len(preview.samples[0]["embedding"]) == 16


def test_preview_reads_as_of_pinned_versions_not_new_rows(lake):
    before_tables = lake.table_names()
    before_count = lake.table("scenarios").count_rows()

    load_snapshot_preview(lake, "demo-v1")

    # Preview only reads; it must not create a new shard layout or rows.
    assert lake.table_names() == before_tables
    assert lake.table("scenarios").count_rows() == before_count


def test_preview_unknown_snapshot_raises(lake):
    with pytest.raises(TrainingError):
        load_snapshot_preview(lake, "no-such-snapshot")


def test_preview_unknown_column_raises(lake):
    with pytest.raises(TrainingError):
        load_snapshot_preview(lake, "demo-v1", columns=["not_a_field"])


# --- optional torch dependency ----------------------------------------------


def test_torch_available_reflects_environment():
    import importlib.util

    assert torch_available() == (importlib.util.find_spec("torch") is not None)


def test_to_torch_dataset_messages_when_torch_missing(lake):
    preview = load_snapshot_preview(lake, "demo-v1")
    if torch_available():
        pytest.skip("torch is installed; the missing-dependency path does not apply")
    with pytest.raises(TrainingError, match="torch"):
        to_torch_dataset(preview)
