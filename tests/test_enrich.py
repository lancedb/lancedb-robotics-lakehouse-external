"""Caption/embedding provider and scenario-enrichment tests (backlog 0007)."""

import json
import math
import time

import pyarrow as pa
import pytest

from lancedb_robotics.captions import (
    ImageStatisticsCaptionProvider,
    resolve_caption_provider,
)
from lancedb_robotics.enrich import (
    CaptionProvider,
    DemoCaptionProvider,
    DemoEmbeddingProvider,
    EmbeddingProvider,
    EnrichmentError,
    ScenarioContext,
    enrich_scenarios,
)
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA, SCENARIOS_SCHEMA


def _ctx(**overrides) -> ScenarioContext:
    base = dict(
        scenario_id="scn-0001",
        run_id="run-abc",
        start_time_ns=0,
        end_time_ns=100_000_000,
        window_ns=100_000_000,
        is_partial=False,
        topics=("/camera/front", "/imu"),
        observation_count=3,
    )
    base.update(overrides)
    return ScenarioContext(**base)


# --- provider contract + determinism (no lake required) ---------------------


def test_demo_caption_is_deterministic_and_descriptive():
    provider = DemoCaptionProvider()
    ctx = _ctx()
    caption = provider.caption(ctx)

    assert caption == (
        "3 observations on run-abc across /camera/front, /imu spanning 100 ms (full window)"
    )
    # Same input always yields the same caption (fresh provider, too).
    assert DemoCaptionProvider().caption(ctx) == caption


def test_demo_caption_marks_partial_windows():
    caption = DemoCaptionProvider().caption(_ctx(is_partial=True, end_time_ns=50_000_000))
    assert "partial window" in caption
    assert "50 ms" in caption


def test_demo_embedding_is_deterministic_unit_vector():
    provider = DemoEmbeddingProvider()
    ctx = _ctx()
    vector = provider.embed(ctx)

    assert len(vector) == provider.dimension
    assert all(isinstance(v, float) for v in vector)
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, abs_tol=1e-6)
    # Deterministic across calls and across fresh instances.
    assert provider.embed(ctx) == vector
    assert DemoEmbeddingProvider().embed(ctx) == vector


def test_demo_embedding_varies_with_scenario_content():
    provider = DemoEmbeddingProvider()
    assert provider.embed(_ctx()) != provider.embed(_ctx(observation_count=4))
    assert provider.embed(_ctx()) != provider.embed(_ctx(topics=("/imu",)))


def test_demo_embedding_dimension_is_configurable():
    provider = DemoEmbeddingProvider(dimension=8)
    assert provider.dimension == 8
    assert len(provider.embed(_ctx())) == 8


def test_demo_providers_satisfy_the_contracts():
    assert isinstance(DemoCaptionProvider(), CaptionProvider)
    assert isinstance(DemoEmbeddingProvider(), EmbeddingProvider)


def test_scenario_context_descriptor_is_stable_and_ordered():
    descriptor = _ctx().descriptor()
    # Stable JSON keyed on the windowing fields, independent of dict order.
    assert json.loads(descriptor) == {
        "run_id": "run-abc",
        "start_time_ns": 0,
        "end_time_ns": 100_000_000,
        "window_ns": 100_000_000,
        "is_partial": False,
        "topics": ["/camera/front", "/imu"],
        "observation_count": 3,
    }


# --- enrichment over a real fixture lake ------------------------------------


@pytest.fixture
def windowed_lake(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    return lake


def _scenarios(lake):
    return sorted(lake.table("scenarios").to_arrow().to_pylist(), key=lambda r: r["scenario_id"])


def test_enrich_writes_summary_and_embedding_onto_scenario_rows(windowed_lake):
    report = enrich_scenarios(windowed_lake)

    assert report.scenarios_enriched == 2
    assert report.embedding_dimension == 16
    rows = _scenarios(windowed_lake)
    for row in rows:
        assert row["summary"]  # non-empty text
        assert len(row["embedding"]) == 16


def test_enrich_outputs_are_deterministic_for_fixtures(windowed_lake, tmp_path, fixtures_dir):
    enrich_scenarios(windowed_lake)
    first = {
        r["scenario_id"]: (r["summary"], tuple(r["embedding"])) for r in _scenarios(windowed_lake)
    }

    # A second, independently built lake must produce identical handles.
    other = Lake.init(tmp_path / "robot2.lance")
    ingest_mcap(other, fixtures_dir / "sample.mcap")
    create_scenario_windows(other, window_ns=100_000_000)
    enrich_scenarios(other)
    second = {r["scenario_id"]: (r["summary"], tuple(r["embedding"])) for r in _scenarios(other)}

    assert first == second


def test_enrich_preserves_windowing_lineage_on_rows(windowed_lake):
    windowing_transform = _scenarios(windowed_lake)[0]["transform_id"]
    enrich_scenarios(windowed_lake)

    for row in _scenarios(windowed_lake):
        # Enrichment does not erase the row's windowing provenance.
        assert row["transform_id"] == windowing_transform
        assert row["observation_ids"]
        assert row["run_id"]


def test_enrich_records_transform_lineage_from_windows_to_rows(windowed_lake):
    windowing_transform = _scenarios(windowed_lake)[0]["transform_id"]
    report = enrich_scenarios(windowed_lake)

    transform = next(
        row
        for row in windowed_lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == report.transform_id
    )
    assert transform["kind"] == "enrichment"
    assert transform["output_tables"] == ["scenarios"]
    params = json.loads(transform["params"])
    assert params["caption_provider"] == "demo-template-v1"
    assert params["embedding_provider"] == "demo-hash-v1"
    assert params["embedding_dimension"] == 16
    assert windowing_transform in params["source_transform_ids"]
    assert sorted(params["scenario_ids"]) == [r["scenario_id"] for r in _scenarios(windowed_lake)]


def test_enrich_rerun_is_idempotent(windowed_lake):
    first = enrich_scenarios(windowed_lake)
    second = enrich_scenarios(windowed_lake)

    assert second.transform_id == first.transform_id
    assert windowed_lake.table("scenarios").count_rows() == 2
    matching = [
        row
        for row in windowed_lake.table("transform_runs").to_arrow().to_pylist()
        if row["transform_id"] == first.transform_id
    ]
    assert len(matching) == 1


def test_enrich_captions_only_skips_the_embedding_column(windowed_lake):
    report = enrich_scenarios(windowed_lake, embedding_provider=None)

    assert report.embedding_provider is None
    assert report.embedding_dimension is None
    rows = _scenarios(windowed_lake)
    assert all(row["summary"] for row in rows)
    assert "embedding" not in windowed_lake.table("scenarios").schema.names


def test_enrich_without_scenarios_raises(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")  # ingested but no windows yet
    with pytest.raises(EnrichmentError):
        enrich_scenarios(lake)


def test_enrich_accepts_custom_model_backed_providers(windowed_lake):
    """The contracts must leave room for real model-backed implementations."""

    class StubVLMCaptionProvider(CaptionProvider):
        name = "stub-vlm"

        def caption(self, ctx: ScenarioContext) -> str:
            return f"vlm:{ctx.scenario_id}"

    class StubModelEmbeddingProvider(EmbeddingProvider):
        name = "stub-model"
        dimension = 4

        def embed(self, ctx: ScenarioContext) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0]

    report = enrich_scenarios(
        windowed_lake,
        caption_provider=StubVLMCaptionProvider(),
        embedding_provider=StubModelEmbeddingProvider(),
    )

    assert report.caption_provider == "stub-vlm"
    assert report.embedding_provider == "stub-model"
    rows = _scenarios(windowed_lake)
    assert all(row["summary"].startswith("vlm:") for row in rows)
    assert all(len(row["embedding"]) == 4 for row in rows)


# --- real caption-provider path (backlog 0022) -----------------------------


def _striped_bright_image() -> bytes:
    return bytes(70 if i % 2 == 0 else 255 for i in range(4096))


def _camera_lake(tmp_path):
    lake = Lake.init(tmp_path / "camera.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": "run-camera", "run_kind": "drive", "raw_uri": "/camera.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    observations = [
        {
            "observation_id": "obs-camera-1",
            "run_id": "run-camera",
            "timestamp_ns": 0,
            "topic": "/camera/front",
            "modality": "image",
            "payload_blob": _striped_bright_image(),
            "decode_status": "decoded",
        },
        {
            "observation_id": "obs-imu-1",
            "run_id": "run-camera",
            "timestamp_ns": 1,
            "topic": "/imu",
            "modality": "imu",
            "payload_blob": None,
            "decode_status": "decoded",
        },
    ]
    lake.table("observations").add(pa.Table.from_pylist(observations, schema=OBSERVATIONS_SCHEMA))
    scenarios = [
        {
            "scenario_id": "scn-camera",
            "run_id": "run-camera",
            "start_time_ns": 0,
            "end_time_ns": 10,
            "window_ns": 10,
            "is_partial": False,
            "topics": ["/camera/front", "/imu"],
            "observation_ids": ["obs-camera-1", "obs-imu-1"],
            "observation_count": 2,
        }
    ]
    lake.table("scenarios").add(pa.Table.from_pylist(scenarios, schema=SCENARIOS_SCHEMA))
    return lake


def test_image_statistics_caption_provider_describes_camera_payload(tmp_path):
    lake = _camera_lake(tmp_path)

    report = enrich_scenarios(
        lake,
        caption_provider=ImageStatisticsCaptionProvider(),
        embedding_provider=None,
        fts_index=False,
    )

    assert report.caption_provider == "image-statistics-v1"
    row = lake.table("scenarios").to_arrow().to_pylist()[0]
    assert "bright high-contrast striped scene" in row["summary"]
    assert not row["summary"].startswith("2 observations on")
    transform = next(
        r
        for r in lake.table("transform_runs").to_arrow().to_pylist()
        if r["transform_id"] == report.transform_id
    )
    assert json.loads(transform["params"])["caption_provider"] == "image-statistics-v1"


def test_caption_provider_without_required_env_falls_back_to_demo(monkeypatch):
    monkeypatch.delenv("LANCEDB_ROBOTICS_CAPTION_API_KEY", raising=False)
    monkeypatch.delenv("LANCEDB_ROBOTICS_CAPTION_ENDPOINT", raising=False)

    provider, notice = resolve_caption_provider("vlm-api")

    assert isinstance(provider, DemoCaptionProvider)
    assert notice and "vlm-api" in notice and "falling back" in notice


# --- perf: enrich must stay linear in referenced observations (BUG-01) -------


def test_enrich_over_100k_observations_stays_linear(tmp_path):
    """Regression guard for BUG-01.

    ``_camera_observations_by_scenario`` used to rebuild ``set(all_ids)`` once
    per observation row, making the scan O(observations x referenced_ids); it did
    not finish within 600 s on a 350k-observation lake. Build a 100k-observation
    lake and assert a full enrich completes well under a generous bound. The
    fixed (linear) path is sub-second here; the quadratic regression projects to
    tens of minutes.
    """
    n_obs = 100_000
    obs_ids = [f"obs-{i:06d}" for i in range(n_obs)]

    lake = Lake.init(tmp_path / "scale.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": "run-scale", "run_kind": "drive", "raw_uri": "/scale.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                {
                    "observation_id": oid,
                    "run_id": "run-scale",
                    "timestamp_ns": i,
                    "topic": "/camera/front",
                    "modality": "image",
                    "decode_status": "decoded",
                }
                for i, oid in enumerate(obs_ids)
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    # A few scenarios referencing a few thousand observations -- enough that the
    # old per-row ``set(all_ids)`` rebuild would dominate the run.
    per_scenario = 1_000
    scenarios = [
        {
            "scenario_id": f"scn-{s:04d}",
            "run_id": "run-scale",
            "start_time_ns": s * per_scenario,
            "end_time_ns": (s + 1) * per_scenario,
            "window_ns": per_scenario,
            "is_partial": False,
            "topics": ["/camera/front"],
            "observation_ids": obs_ids[s * per_scenario : (s + 1) * per_scenario],
            "observation_count": per_scenario,
        }
        for s in range(3)
    ]
    lake.table("scenarios").add(pa.Table.from_pylist(scenarios, schema=SCENARIOS_SCHEMA))

    start = time.perf_counter()
    report = enrich_scenarios(lake, fts_index=False)
    elapsed = time.perf_counter() - start

    assert report.scenarios_enriched == 3
    rows = _scenarios(lake)
    assert all(row["summary"] for row in rows)
    assert all(len(row["embedding"]) == 16 for row in rows)
    # Generous bound: the linear path finishes well under a second; the quadratic
    # regression would take tens of minutes.
    assert elapsed < 30, (
        f"enrich over {n_obs} observations took {elapsed:.1f}s (BUG-01 regression?)"
    )


# --- dense embed_text seam: embed scene semantics, not topic boilerplate (BUG-07) --


def _cos(a, b) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def test_build_embed_text_drops_structural_boilerplate():
    from lancedb_robotics.enrich import _build_embed_text

    description = "Night, after rain, many peds"
    caption = "8830 observations on run-x across /CAM_BACK/... spanning 5000 ms"
    # A structural caption (the demo default) is excluded: embed_text is the
    # description alone, so the topic boilerplate never reaches the vector space.
    assert _build_embed_text(description, caption, caption_is_semantic=False) == description
    # A scene-semantic caption (e.g. a VLM) is folded in alongside the description.
    assert (
        _build_embed_text(description, caption, caption_is_semantic=True)
        == f"{description} — {caption}"
    )
    # No description + structural caption -> None, so the embedder falls back.
    assert _build_embed_text(None, caption, caption_is_semantic=False) is None
    # No description + semantic caption -> just the caption.
    assert _build_embed_text(None, caption, caption_is_semantic=True) == caption


def test_enrich_embeds_dense_description_not_summary_boilerplate(tmp_path):
    """The stored embedding is built from the dense description, not the boilerplate.

    BUG-07: ``summary`` keeps the structural caption for display/FTS, but the
    embedder must see description-forward text so vectors rank by scene content.
    """
    from lancedb_robotics.embeddings import HashedTextEmbeddingProvider

    lake = _scene_metadata_lake(tmp_path)
    provider = HashedTextEmbeddingProvider(dimension=128)
    enrich_scenarios(lake, embedding_provider=provider, fts_index=False)

    row = _scenarios(lake)[0]
    stored = list(row["embedding"])
    description, sep, _caption = row["summary"].partition(" — ")
    assert sep  # the summary still folds description — structural caption for display
    # The vector matches embedding the dense description, not the full summary.
    assert _cos(stored, provider.embed_text(description)) > 0.999
    assert _cos(stored, provider.embed_text(description)) > _cos(
        stored, provider.embed_text(row["summary"])
    )


# --- descriptive run-metadata fold makes summaries searchable ---------------


_SCENE_DESCRIPTION = "Night, after rain, many peds, jaywalker, truck, scooter"


def _scene_metadata_lake(tmp_path, *, description=_SCENE_DESCRIPTION):
    """A minimal lake whose single run carries nuScenes-style scene metadata."""
    lake = Lake.init(tmp_path / "scene.lance")
    metadata = [
        {"key": "profile", "value": ""},
        {"key": "library", "value": "nuscenes2mcap"},
        {"key": "scene-info.location", "value": "singapore-hollandvillage"},
        {"key": "integrity.status", "value": "complete"},
    ]
    if description:
        metadata.append({"key": "scene-info.description", "value": description})
    lake.table("runs").add(
        pa.Table.from_pylist(
            [
                {
                    "run_id": "run-scene",
                    "run_kind": "drive",
                    "raw_uri": "/scene.mcap",
                    "metadata": metadata,
                }
            ],
            schema=RUNS_SCHEMA,
        )
    )
    lake.table("observations").add(
        pa.Table.from_pylist(
            [
                {
                    "observation_id": "obs-1",
                    "run_id": "run-scene",
                    "timestamp_ns": 0,
                    "topic": "/lidar",
                    "modality": "pointcloud",
                    "decode_status": "decoded",
                }
            ],
            schema=OBSERVATIONS_SCHEMA,
        )
    )
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-scene",
                    "run_id": "run-scene",
                    "start_time_ns": 0,
                    "end_time_ns": 10_000_000,
                    "window_ns": 10_000_000,
                    "is_partial": False,
                    "topics": ["/lidar"],
                    "observation_ids": ["obs-1"],
                    "observation_count": 1,
                }
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )
    return lake


def test_enrich_folds_scene_description_into_summary(tmp_path):
    lake = _scene_metadata_lake(tmp_path)

    enrich_scenarios(lake, fts_index=False)

    summary = _scenarios(lake)[0]["summary"]
    # Scene semantics lead the summary; the structural caption is still present.
    assert summary.startswith(_SCENE_DESCRIPTION)
    assert "singapore-hollandvillage" in summary
    assert "1 observations on run-scene" in summary
    # Structural/operational metadata is never folded into the searchable text.
    assert "nuscenes2mcap" not in summary
    assert "complete" not in summary


def test_enrich_leaves_summary_unchanged_without_descriptive_metadata(windowed_lake):
    enrich_scenarios(windowed_lake, embedding_provider=None, fts_index=False)

    for row in _scenarios(windowed_lake):
        # The fixture run carries only structural metadata (profile/library/
        # integrity), so nothing is folded and the summary is exactly the caption.
        ctx = ScenarioContext.from_row(row)
        assert row["summary"] == DemoCaptionProvider().caption(ctx)


def test_enrich_makes_scene_semantics_full_text_searchable(tmp_path):
    from lancedb_robotics.search import TEXT, search_scenarios

    lake = _scene_metadata_lake(tmp_path)
    enrich_scenarios(lake)  # builds the persistent FTS index over summary

    # The default offline path now finds the scene by what its source described,
    # even though the structural caption never mentions night/rain/peds.
    results = search_scenarios(lake, mode=TEXT, query="night rain pedestrians")
    assert [r.scenario_id for r in results] == ["scn-scene"]


def test_enrich_refolds_summaries_that_predate_the_fold(tmp_path):
    lake = _scene_metadata_lake(tmp_path)
    enrich_scenarios(lake, fts_index=False)

    scenarios = lake.table("scenarios")
    row = scenarios.to_arrow().to_pylist()[0]
    assert _SCENE_DESCRIPTION in row["summary"]  # folded on the first run

    # Simulate a summary written before the fold existed (structural only), while
    # the enrichment transform still records the scenario as already done.
    row["summary"] = "1 observations on run-scene (structural only)"
    schema = scenarios.schema
    scenarios.delete("scenario_id = 'scn-scene'")
    scenarios.add(pa.Table.from_pylist([row], schema=schema))

    # A re-run re-folds the scene text in instead of skipping it as up-to-date.
    report = enrich_scenarios(lake, fts_index=False)

    assert report.scenarios_enriched == 1
    # Re-open the table so the read sees enrich's write, not the stale handle.
    refolded = lake.table("scenarios").to_arrow().to_pylist()[0]["summary"]
    assert refolded.startswith(_SCENE_DESCRIPTION)


# --- concurrent-writer convergence + partial-write safety net (BUG-04) -------


def _commit_conflict() -> RuntimeError:
    """A stand-in for Lance's optimistic-concurrency commit-conflict error."""
    return RuntimeError(
        "lance error: Retryable commit conflict for version 3: This Merge transaction "
        "was preempted by concurrent transaction Merge at version 3"
    )


def test_is_retryable_commit_conflict_classifier():
    from lancedb_robotics.enrich import _is_retryable_commit_conflict

    assert _is_retryable_commit_conflict(_commit_conflict())
    assert not _is_retryable_commit_conflict(RuntimeError("disk on fire"))
    assert not _is_retryable_commit_conflict(EnrichmentError("no scenarios to enrich"))


def test_enrich_retries_on_commit_conflict_and_converges(windowed_lake, monkeypatch):
    """A preempted writer re-reads the latest version and re-runs, not fails.

    Lance arbitrates concurrent enrich writers at commit time and raises a
    retryable commit conflict for the loser; enrich is idempotent, so it retries
    and converges instead of erroring or nulling the table (BUG-04).
    """
    import lancedb_robotics.enrich as enrich_mod

    real = enrich_mod._enrich_once
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _commit_conflict()  # first attempt loses the race
        return real(*args, **kwargs)

    monkeypatch.setattr(enrich_mod, "_enrich_once", flaky)
    report = enrich_scenarios(windowed_lake, fts_index=False)

    assert calls["n"] == 2  # retried once, then committed
    assert report.scenarios_enriched == 2
    rows = _scenarios(windowed_lake)
    assert all(row["summary"] for row in rows)
    assert all(len(row["embedding"]) == 16 for row in rows)


def test_enrich_does_not_retry_non_conflict_errors(windowed_lake, monkeypatch):
    """Any error that is not a commit conflict propagates immediately."""
    import lancedb_robotics.enrich as enrich_mod

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(enrich_mod, "_enrich_once", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        enrich_scenarios(windowed_lake, fts_index=False)


def test_enrich_gives_up_after_sustained_commit_contention(windowed_lake, monkeypatch):
    """Bounded retries: sustained conflicts surface a clear EnrichmentError."""
    import lancedb_robotics.enrich as enrich_mod

    attempts = {"n": 0}

    def always_conflict(*args, **kwargs):
        attempts["n"] += 1
        raise _commit_conflict()

    monkeypatch.setattr(enrich_mod, "_enrich_once", always_conflict)
    with pytest.raises(EnrichmentError, match="commit races"):
        enrich_scenarios(windowed_lake, fts_index=False)
    assert attempts["n"] == enrich_mod._ENRICH_COMMIT_RETRIES + 1


def test_enrich_validation_flags_a_partially_written_summary(windowed_lake):
    """The post-write check refuses to report success over a nulled summary."""
    from lancedb_robotics.enrich import _assert_fully_enriched

    enrich_scenarios(windowed_lake, embedding_provider=None, fts_index=False)

    # Simulate a corrupt/partial write: blank one row's summary back to null.
    scenarios = windowed_lake.table("scenarios")
    rows = scenarios.to_arrow().to_pylist()
    rows[0]["summary"] = None
    schema = scenarios.schema
    scenarios.delete(f"scenario_id = '{rows[0]['scenario_id']}'")
    scenarios.add(pa.Table.from_pylist([rows[0]], schema=schema))

    with pytest.raises(EnrichmentError, match="null"):
        _assert_fully_enriched(windowed_lake, "embedding", require_embedding=False)


def test_enrich_upsert_updates_in_place_without_duplicating_rows(windowed_lake):
    """The atomic merge_insert write updates rows in place (no row growth)."""
    before = windowed_lake.table("scenarios").count_rows()

    enrich_scenarios(windowed_lake, fts_index=False)
    after_first = windowed_lake.table("scenarios").count_rows()

    # Re-enrich after invalidating the stored summaries: rows are upserted in
    # place, never deleted-then-appended into duplicates.
    scenarios = windowed_lake.table("scenarios")
    rows = scenarios.to_arrow().to_pylist()
    schema = scenarios.schema
    for row in rows:
        row["summary"] = None
    scenarios.delete("scenario_id IS NOT NULL")
    scenarios.add(pa.Table.from_pylist(rows, schema=schema))
    enrich_scenarios(windowed_lake, fts_index=False)

    assert after_first == before
    assert windowed_lake.table("scenarios").count_rows() == before
    assert all(row["summary"] for row in _scenarios(windowed_lake))


# --- embedding-column targeting + in-place dimension migration (BUG-08) ------


def _list_size(lake, column: str) -> int:
    return lake.table("scenarios").schema.field(column).type.list_size


def test_enrich_writes_into_a_named_embedding_column(windowed_lake):
    """A non-default ``embedding_column`` writes its own vector column."""
    report = enrich_scenarios(
        windowed_lake,
        embedding_provider=DemoEmbeddingProvider(dimension=32),
        embedding_column="embedding_alt",
        fts_index=False,
    )

    assert report.embedding_column == "embedding_alt"
    assert "embedding_alt" in windowed_lake.table("scenarios").schema.names
    assert all(len(row["embedding_alt"]) == 32 for row in _scenarios(windowed_lake))


def test_enrich_multiple_embedding_columns_coexist(windowed_lake):
    """Two providers/dims enrich side by side without clobbering each other (BUG-08)."""
    enrich_scenarios(
        windowed_lake,
        embedding_provider=DemoEmbeddingProvider(dimension=16),
        embedding_column="embedding",
        fts_index=False,
    )
    enrich_scenarios(
        windowed_lake,
        embedding_provider=DemoEmbeddingProvider(dimension=32),
        embedding_column="embedding_alt",
        fts_index=False,
    )

    names = windowed_lake.table("scenarios").schema.names
    assert "embedding" in names and "embedding_alt" in names
    for row in _scenarios(windowed_lake):
        assert len(row["embedding"]) == 16
        assert len(row["embedding_alt"]) == 32
    # Each (provider, dim, column) gets its own enrichment lineage row.
    columns = {
        json.loads(r["params"])["embedding_column"]
        for r in windowed_lake.table("transform_runs").to_arrow().to_pylist()
        if r["kind"] == "enrichment"
    }
    assert {"embedding", "embedding_alt"} <= columns


def test_enrich_dimension_change_without_replace_raises_signposted_error(windowed_lake):
    """A dim change on the same column fails fast, pointing at both escape hatches."""
    enrich_scenarios(
        windowed_lake, embedding_provider=DemoEmbeddingProvider(dimension=16), fts_index=False
    )
    with pytest.raises(EnrichmentError) as excinfo:
        enrich_scenarios(
            windowed_lake, embedding_provider=DemoEmbeddingProvider(dimension=32), fts_index=False
        )
    message = str(excinfo.value)
    assert "already exists" in message
    assert "--embedding-column" in message
    assert "--replace-embedding" in message
    # The failed attempt left the original column intact (no silent data loss).
    assert _list_size(windowed_lake, "embedding") == 16
    assert all(len(row["embedding"]) == 16 for row in _scenarios(windowed_lake))


def test_enrich_replace_embedding_migrates_dimension_in_place(windowed_lake):
    """``replace_embedding`` drops+recreates the column and re-embeds at the new dim."""
    enrich_scenarios(
        windowed_lake, embedding_provider=DemoEmbeddingProvider(dimension=16), fts_index=False
    )
    assert _list_size(windowed_lake, "embedding") == 16

    report = enrich_scenarios(
        windowed_lake,
        embedding_provider=DemoEmbeddingProvider(dimension=32),
        embedding_column="embedding",
        replace_embedding=True,
        fts_index=False,
    )

    assert report.embedding_dimension == 32
    rows = _scenarios(windowed_lake)
    assert rows  # the table is not nulled/emptied by the migration
    assert all(len(row["embedding"]) == 32 for row in rows)
    assert _list_size(windowed_lake, "embedding") == 32


def test_enrich_replace_embedding_on_fresh_column_just_adds_it(windowed_lake):
    """``replace_embedding`` on a lake with no embedding column is a plain add."""
    report = enrich_scenarios(
        windowed_lake,
        embedding_provider=DemoEmbeddingProvider(dimension=16),
        replace_embedding=True,
        fts_index=False,
    )

    assert report.embedding_dimension == 16
    assert all(len(row["embedding"]) == 16 for row in _scenarios(windowed_lake))
