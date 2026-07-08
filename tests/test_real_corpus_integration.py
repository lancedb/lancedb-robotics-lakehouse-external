"""Integration tests over the **real** MCAP corpora under ``data/``.

The rest of the suite runs on tiny, deterministic, checked-in fixtures (sliced
from these corpora). This module instead exercises the full lakehouse spine on
the genuine, large, gitignored robot logs that live under ``data/`` at the repo
root, so the real decode, compression, mixed-encoding, idempotency, and
end-to-end pipeline paths are proven on actual bytes rather than slices:

- ``data/demo/demo.mcap`` — ros1msg, zstd (~59 MB).
- ``data/nuScenes/NuScenes-v1.0-mini-scene-*.mcap`` — **mixed** json + protobuf +
  ros1 encodings in one file, lz4 (~435–686 MB).
- ``data/didi/...`` — ros1msg, zstd (up to multiple GB).

The corpora are gitignored and not present in CI, so **every test skips cleanly
when its file is absent** (the suite stays green without the data) and runs
locally where ``data/`` is present. Bytes are large but message *counts* are
modest (image/lidar-heavy), so inspect and the demo pipeline are fast; the one
full-file nuScenes ingest is marked ``slow`` and the multi-GB didi logs are only
ever inspected here (streaming-memory scale is owned by ``test_perf_memory.py``).

Markers (registered in ``pyproject.toml``)::

    pytest -m realcorpus                  # every real-corpus test
    pytest -m "realcorpus and not slow"   # fast subset (no full-file ingest)
"""

import os
import shutil
from pathlib import Path

import pytest
from mcap.reader import make_reader

from lancedb_robotics.adapters import get_adapter
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.enrich import enrich_scenarios
from lancedb_robotics.export import export_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.quality import (
    QUARANTINED_FLAG,
    RequiredTopic,
    ValidationProfile,
    apply_quality_results,
    validate_lake,
)
from lancedb_robotics.scenarios import create_scenario_windows, parse_duration_ns
from lancedb_robotics.search import last_search, record_search, search_scenarios
from lancedb_robotics.training import load_snapshot_preview

pytestmark = pytest.mark.realcorpus

# A clean full-file ingest is only allowed below this size, so a stray multi-GB
# didi log can never be fully ingested by accident here (that scale belongs to
# the opt-in memory smoke). The smallest nuScenes scene (~435 MB) sits under it.
MAX_FULL_INGEST_BYTES = 800 * 1024 * 1024


# --- corpus discovery: locate data/ by walking up, skip when absent ---------


def _find_data_root() -> Path | None:
    """Locate the gitignored ``data/`` corpus by walking up from this file.

    Inside a nested git worktree the corpus lives at the real repo root, not in
    the worktree, so search upward rather than assuming a fixed depth — same
    approach as the fixture ``make_*`` scripts.
    """
    for base in Path(__file__).resolve().parents:
        candidate = base / "data"
        if candidate.is_dir() and any(candidate.rglob("*.mcap")):
            return candidate
    return None


DATA_ROOT = _find_data_root()


def _require_root() -> Path:
    if DATA_ROOT is None:
        pytest.skip("no real MCAP corpus under data/ on this machine")
    return DATA_ROOT


def _demo_file() -> Path:
    path = _require_root() / "demo" / "demo.mcap"
    if not path.is_file():
        pytest.skip(f"demo corpus not present: {path}")
    return path


def _smallest_mcap_under(*parts: str) -> Path:
    """The smallest ``*.mcap`` under ``data/<parts...>`` (deterministic, cheapest)."""
    root = _require_root().joinpath(*parts)
    files = (
        sorted(root.rglob("*.mcap"), key=lambda p: (p.stat().st_size, str(p)))
        if root.is_dir()
        else []
    )
    if not files:
        pytest.skip(f"no mcap files under {root}")
    return files[0]


def _nuscenes_file() -> Path:
    return _smallest_mcap_under("nuScenes")


def _didi_file() -> Path:
    # The Training-Release-1 set is uniformly ros1msg/zstd; pick its smallest.
    return _smallest_mcap_under("didi", "Didi-Training-Release-1")


# --- shared helpers ---------------------------------------------------------


def _encodings(report: dict) -> set[tuple[str | None, str | None]]:
    return {(t["message_encoding"], t["schema_encoding"]) for t in report["topics"]}


def _chunk_compressions(report: dict) -> set[str]:
    return {chunk["compression"] for chunk in report["chunks"]}


def _read_messages(path: Path) -> list:
    with Path(path).open("rb") as handle:
        return list(make_reader(handle).iter_messages())


def _slice_to(adapter, source: Path, out: Path, *, window_ns: int, compression: str | None) -> dict:
    """Cut ``[start, start+window]`` from ``source`` into ``out`` (no payload decode).

    Slicing copies raw message bytes, so it is cheap even on a large source and
    keeps the source's real encodings + chunk codec — a bounded, genuine sample.
    """
    info = adapter.inspect(source)
    start = info["start_time_ns"]
    return adapter.export(
        source,
        start_time_ns=start,
        end_time_ns=start + window_ns,
        out_path=out,
        compression=compression,
    )


def _assert_ingest_self_consistent(lake: Lake, report) -> None:
    """Invariants that must hold for any clean single-file ingest, on any corpus."""
    # The streamed observation rows are exactly the messages the report counted.
    assert report.message_count > 0
    assert lake.table("observations").count_rows() == report.message_count
    assert sum(report.observations_by_topic.values()) == report.message_count
    assert sum(report.decode_by_status.values()) == report.message_count
    assert sum(report.decode_by_encoding.values()) == report.message_count
    # Exactly one run, content-addressed id, and a non-negative span.
    assert lake.table("runs").count_rows() == 1
    assert report.run_id.startswith("run-")
    assert report.end_time_ns >= report.start_time_ns
    assert report.rows_added["observations"] == report.message_count


def _full_pipeline(
    lake: Lake,
    *,
    scenario_window: str,
    out_dir: Path,
    query: str = "observations",
):
    """Drive scenarios → enrich → search → snapshot → export over an ingested lake.

    Returns ``(windowing, enrichment, results, snapshot, manifest)`` so callers can
    assert on each stage. Mirrors the demo showcase spine but parameterized so it
    runs identically on a demo log and on a sliced nuScenes window.
    """
    window_ns = parse_duration_ns(scenario_window)
    windowing = create_scenario_windows(lake, window_ns=window_ns)
    assert windowing.rows_added > 0, "real run produced no scenario windows"

    enrichment = enrich_scenarios(lake)
    assert enrichment.scenarios_enriched == windowing.rows_added
    assert enrichment.embedding_dimension == 16

    results = search_scenarios(lake, mode="hybrid", query=query, limit=5)
    assert results, "hybrid search returned no hits over enriched real scenarios"
    assert all(r.relevance_score is not None for r in results)
    # Every hit links back to the raw source log.
    assert all(r.source_uri for r in results)

    record_search(
        lake,
        mode="hybrid",
        query=query,
        where=None,
        limit=5,
        scenario_ids=[r.scenario_id for r in results],
    )
    spec = last_search(lake)
    assert spec and spec["scenario_ids"]

    snapshot = create_snapshot(
        lake,
        name="real-it-v1",
        scenario_ids=spec["scenario_ids"],
        source={"kind": "search", "mode": "hybrid", "query": query},
    )
    assert sum(snapshot.split_counts.values()) == len(snapshot.scenario_ids)

    manifest = export_snapshot(lake, "real-it-v1", out_dir=str(out_dir))
    return windowing, enrichment, results, snapshot, manifest


# --- demo corpus: full end-to-end spine on real ros1 / zstd bytes -----------


def test_demo_full_pipeline_end_to_end(tmp_path):
    """init → ingest → quality → scenarios → enrich → search → snapshot → export,
    on the real 59 MB ros1msg/zstd demo log."""
    source = _demo_file()
    lake = Lake.init(tmp_path / "demo.lance")

    report = ingest_mcap(lake, source, batch_size=256)
    _assert_ingest_self_consistent(lake, report)
    # Real shape: a finalized, fully-decodable ros1 log (no mixed encodings).
    assert report.integrity_status == "complete"
    assert not report.quarantined
    assert set(report.decode_by_encoding) == {"ros1"}
    assert report.decode_by_status == {"decoded": report.message_count}
    # Stable demo topics survive ingest as observation channels.
    assert {"/diagnostics", "/velodyne_points"} <= set(report.observations_by_topic)

    # Quality gate on a real log. The built-in "demo" profile targets the
    # synthetic sample fixture (/imu, /camera/front); the real demo.mcap carries
    # different topics, so validate against a profile that matches its actual
    # shape. require_monotonic is off because a genuine multi-publisher log has
    # repeated per-topic log_times (a real property, not a defect).
    profile = ValidationProfile(
        name="demo-real",
        required_topics=(RequiredTopic("/velodyne_points", 1), RequiredTopic("/diagnostics", 1)),
        decodable_topics=("/diagnostics",),
        require_monotonic=False,
    )
    reports = validate_lake(lake, profile, run_id=report.run_id)
    apply_quality_results(lake, reports, profile)
    assert reports[0].passed, [r.to_dict() for r in reports]
    flags = lake.table("runs").to_arrow().to_pylist()[0]["quality_flags"] or []
    assert QUARANTINED_FLAG not in flags
    assert f"quality:{profile.name}:passed" in flags

    # The rest of the spine.
    clips_dir = tmp_path / "clips"
    windowing, _enrich, _results, snapshot, manifest = _full_pipeline(
        lake, scenario_window="1s", out_dir=clips_dir
    )

    # End-to-end lineage: one transform row per pipeline stage.
    kinds = {row["kind"] for row in lake.table("transform_runs").to_arrow().to_pylist()}
    assert {
        "inspect",
        "ingest",
        "quality",
        "scenario-windowing",
        "enrichment",
        "search",
        "dataset-snapshot",
        "export",
    } <= kinds

    # Training preview reads the snapshot's pinned versions deterministically.
    preview = load_snapshot_preview(lake, "real-it-v1")
    assert preview.total_scenarios == len(snapshot.scenario_ids)
    assert preview.samples and all(len(s["embedding"]) == 16 for s in preview.samples)

    # Exported clips are real, re-readable MCAP carrying the window's messages.
    assert manifest.exported == len(manifest.clips) > 0
    assert manifest.skipped == 0
    clip_files = sorted(clips_dir.glob("*.mcap"))
    assert len(clip_files) == manifest.exported
    assert _read_messages(clip_files[0]), "exported clip has no messages"


def test_demo_cli_inspect_text(tmp_path):
    """The CLI text inspector renders a real log without crashing (smoke)."""
    from typer.testing import CliRunner

    from lancedb_robotics.cli import app

    source = _demo_file()
    result = CliRunner().invoke(app, ["inspect", "mcap", str(source), "--format", "text"])
    assert result.exit_code == 0, result.output
    # Real ros1 demo facts surface in the rendered report.
    assert "/velodyne_points" in result.output
    assert "ros1" in result.output


# --- nuScenes corpus: the mixed-encoding + lz4 story ------------------------


def test_nuscenes_inspect_mixed_encodings_and_lz4(tmp_path):
    """A single nuScenes scene mixes json + protobuf + ros1 under lz4 chunks, and
    carries the log-level ``scene-info`` metadata record — all visible to inspect."""
    source = _nuscenes_file()
    report = get_adapter("mcap").inspect(source)

    assert report["indexed"] is True
    assert report["message_count"] > 0
    # The defining property: three distinct message encodings in ONE file.
    encodings = _encodings(report)
    assert ("json", "jsonschema") in encodings
    assert ("protobuf", "protobuf") in encodings
    assert ("ros1", "ros1msg") in encodings
    # Real nuScenes chunks are lz4-compressed.
    assert _chunk_compressions(report) == {"lz4"}
    # MCAP's other first-class record type: a single log-level metadata block.
    metadata_names = {m["name"] for m in report["metadata"]}
    assert "scene-info" in metadata_names


def test_nuscenes_window_pipeline_decodes_all_encodings(tmp_path):
    """Slice a 1 s window of real nuScenes lz4 bytes and run the full pipeline.

    Bounds the heavy json/protobuf/ros1 decode to the window's ~1.7k messages
    while proving every encoding decodes end-to-end and the spine completes on a
    genuine mixed-encoding source."""
    source = _nuscenes_file()
    adapter = get_adapter("mcap")
    slice_path = tmp_path / "nuscenes_window.mcap"
    sliced = _slice_to(
        adapter, source, slice_path, window_ns=parse_duration_ns("1s"), compression="lz4"
    )
    assert sliced["message_count"] > 0
    # The slice keeps the source's mixed encodings + lz4 codec.
    assert _chunk_compressions(adapter.inspect(slice_path)) == {"lz4"}

    lake = Lake.init(tmp_path / "nuscenes.lance")
    report = ingest_mcap(lake, slice_path, batch_size=256)
    _assert_ingest_self_consistent(lake, report)
    assert report.integrity_status == "complete"
    # All three real encodings decoded from genuine bytes, none dropped.
    assert {"json", "protobuf", "ros1"} <= set(report.decode_by_encoding)
    assert report.decode_by_status.get("decoded", 0) > 0
    assert "failed" not in report.decode_by_status

    windowing, _e, _r, snapshot, manifest = _full_pipeline(
        lake, scenario_window="250ms", out_dir=tmp_path / "clips"
    )
    assert windowing.runs_considered == 1
    # The window spanned multiple sensor topics; the export round-trips them.
    assert manifest.exported == len(manifest.clips) > 0
    clip_files = sorted((tmp_path / "clips").glob("*.mcap"))
    assert _read_messages(clip_files[0])


@pytest.mark.slow
def test_nuscenes_full_ingest_scale_and_metadata_record(tmp_path):
    """Ingest a whole nuScenes scene (hundreds of MB) and assert real-scale facts:
    the mixed encodings all decode, and the ``scene-info`` metadata record merges
    into ``runs.metadata`` namespaced by record name (backlog 0016)."""
    source = _nuscenes_file()
    if source.stat().st_size > MAX_FULL_INGEST_BYTES:
        pytest.skip(f"{source.name} exceeds the full-ingest cap; covered by the window test")

    lake = Lake.init(tmp_path / "nuscenes_full.lance")
    report = ingest_mcap(lake, source, batch_size=128, validate_crcs=False)
    _assert_ingest_self_consistent(lake, report)
    assert report.integrity_status == "complete"
    # At real scale the file decodes into all three encodings, none failed.
    assert {"json", "protobuf", "ros1"} <= set(report.decode_by_encoding)
    assert "failed" not in report.decode_by_status

    # The log-level scene-info metadata record is merged onto the run, prefixed
    # by the record name (e.g. ``scene-info.location``).
    run = lake.table("runs").to_arrow().to_pylist()[0]
    metadata = {entry["key"]: entry["value"] for entry in run["metadata"]}
    scene_keys = [key for key in metadata if key.startswith("scene-info.")]
    assert scene_keys, f"scene-info metadata not merged into run; keys={sorted(metadata)}"


# --- didi corpus: a second, larger ros1 / zstd source -----------------------


def test_didi_inspect_ros1_zstd_shape(tmp_path):
    """The smallest didi training log is a finalized ros1msg/zstd recording whose
    streams are all decodable — a second independent ros1/zstd corpus."""
    source = _didi_file()
    report = get_adapter("mcap").inspect(source)

    assert report["profile"] == "ros1"
    assert report["message_count"] > 0
    assert report["indexed"] is True
    assert _encodings(report) == {("ros1", "ros1msg")}
    assert _chunk_compressions(report) == {"zstd"}
    assert all(topic["can_decode"] for topic in report["topics"])


# --- cross-cutting robustness over every discovered corpus shape ------------


def _representative_files() -> list:
    """One file per corpus shape present on this machine, for parametrized checks.

    Resolved at import time so absent corpora drop out of the parametrization
    rather than erroring; if none are present every case is skipped.
    """
    if DATA_ROOT is None:
        return []
    candidates = {
        "demo": DATA_ROOT / "demo" / "demo.mcap",
        "nuscenes": _first_or_none("nuScenes"),
        "didi": _first_or_none("didi", "Didi-Training-Release-1"),
    }
    return [pytest.param(path, id=name) for name, path in candidates.items() if path]


def _first_or_none(*parts: str) -> Path | None:
    root = DATA_ROOT.joinpath(*parts) if DATA_ROOT else None
    if not root or not root.is_dir():
        return None
    files = sorted(root.rglob("*.mcap"), key=lambda p: (p.stat().st_size, str(p)))
    return files[0] if files else None


@pytest.mark.parametrize("source", _representative_files())
def test_inspect_is_self_consistent(source):
    """Inspect succeeds on every real corpus shape and its stats agree internally."""
    report = get_adapter("mcap").inspect(source)
    assert report["message_count"] > 0
    assert report["duration_ns"] == report["end_time_ns"] - report["start_time_ns"]
    assert report["end_time_ns"] >= report["start_time_ns"]
    # One row per topic, not per channel: a topic multiplexed across channels
    # (ROS /rosout, /diagnostics) must not appear — or double-count — twice.
    names = [t["topic"] for t in report["topics"]]
    assert len(names) == len(set(names)), f"duplicate topic rows: {names}"
    # Per-topic counts reconstruct the headline message count exactly.
    assert sum(t["message_count"] for t in report["topics"]) == report["message_count"]
    # Every topic reports a message encoding; there are at least as many
    # channels as distinct topics (more when a topic has several publishers).
    assert all(t["message_encoding"] for t in report["topics"])
    assert report["channel_count"] >= len(report["topics"])


def test_ingest_is_idempotent_by_content(tmp_path):
    """Re-ingesting the same demo bytes is a content-addressed no-op, not a dup."""
    source = _demo_file()
    lake = Lake.init(tmp_path / "idem.lance")

    first = ingest_mcap(lake, source, batch_size=256)
    assert not first.already_ingested
    obs_after_first = lake.table("observations").count_rows()

    second = ingest_mcap(lake, source, batch_size=256)
    assert second.already_ingested
    assert second.run_id == first.run_id
    # No new rows: one run, same observation count.
    assert lake.table("runs").count_rows() == 1
    assert lake.table("observations").count_rows() == obs_after_first
    assert second.rows_added["transform_runs"] == 1
    assert all(
        count == 0 for table, count in second.rows_added.items() if table != "transform_runs"
    )


def test_run_id_is_content_addressed_and_path_independent(tmp_path):
    """The same bytes ingested from a different path yield the same run/source id.

    Run identity is keyed on file content, not the absolute path (backlog 0013),
    so a relocated copy is the same run."""
    source = _demo_file()
    relocated = tmp_path / "relocated" / "renamed-demo.mcap"
    relocated.parent.mkdir(parents=True)
    # Hardlink when possible (no byte copy); fall back to a real copy across devices.
    try:
        os.link(source, relocated)
    except OSError:
        shutil.copy2(source, relocated)

    lake_a = Lake.init(tmp_path / "a.lance")
    lake_b = Lake.init(tmp_path / "b.lance")
    report_a = ingest_mcap(lake_a, source, batch_size=256)
    report_b = ingest_mcap(lake_b, relocated, batch_size=256)

    assert report_a.run_id == report_b.run_id
    assert report_a.source.source_id == report_b.source.source_id
    assert report_a.message_count == report_b.message_count
    # raw_uri provenance still records the actual path each was read from.
    assert report_a.source.uri != report_b.source.uri
