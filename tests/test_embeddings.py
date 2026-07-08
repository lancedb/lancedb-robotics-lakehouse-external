"""Real, pluggable embedding providers + observation image embeddings (backlog 0021).

Covers acceptance criteria 1, 2, 4, and the provider/version half of 5:

1. a real text provider shares one space for ``embed``/``embed_text`` and is
   selectable on ``scenarios enrich``;
2. camera observations carry real image embeddings; near-duplicate frames are
   near neighbors and an unrelated frame is farther;
4. the model-backed extra is absent in CI, so resolving ``clip`` /
   ``sentence-transformers`` degrades to the demo provider with a notice -- never
   a crash;
5. the enrichment lineage records the provider name *and version*.

The index half of criterion 3/5 lives in ``test_indexing.py``.
"""

import math

import pyarrow as pa
import pytest

from lancedb_robotics import embeddings as emb
from lancedb_robotics.embeddings import (
    DEFAULT_IMAGE_EMBEDDING_COLUMN,
    PROVIDER_REGISTRY,
    HashedImageEmbeddingProvider,
    HashedTextEmbeddingProvider,
    embed_observations,
    embedding_extra_available,
    resolve_embedding_provider,
)
from lancedb_robotics.enrich import (
    EmbeddingProvider,
    EnrichmentError,
    ScenarioContext,
    enrich_scenarios,
)
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _ctx(summary: str | None = None, **overrides) -> ScenarioContext:
    base = dict(
        scenario_id="scn-0001",
        run_id="run-abc",
        start_time_ns=0,
        end_time_ns=100_000_000,
        window_ns=100_000_000,
        is_partial=False,
        topics=("/camera/front", "/imu"),
        observation_count=3,
        summary=summary,
    )
    base.update(overrides)
    return ScenarioContext(**base)


# --- hashed-text provider: a real, dependency-free text contract (AC1) ------


def test_hashed_text_is_a_provider_and_unit_normed():
    provider = HashedTextEmbeddingProvider(dimension=64)
    assert isinstance(provider, EmbeddingProvider)
    vector = provider.embed_text("pedestrian cut-in at dusk")
    assert len(vector) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, abs_tol=1e-6)


def test_hashed_text_shares_one_space_for_scenario_and_query():
    # AC1: embed(caption) and embed_text(same text) land in the SAME space, so a
    # query is a near neighbor of a caption about the same content.
    provider = HashedTextEmbeddingProvider(dimension=64)
    text = "pedestrian cut-in at dusk near the crosswalk"
    assert math.isclose(_cos(provider.embed(_ctx(summary=text)), provider.embed_text(text)), 1.0,
                        abs_tol=1e-6)


def test_hashed_text_embeds_dense_embed_text_not_boilerplate_summary():
    # BUG-07: the embedder must embed the dense, description-forward ``embed_text``,
    # not the display/FTS ``summary`` whose ~90% topic boilerplate collapses vectors.
    provider = HashedTextEmbeddingProvider(dimension=128)
    dense = "night after rain many pedestrians jaywalker scooter"
    boilerplate = (
        "8830 observations on run-x across /CAM_BACK/annotations, "
        "/CAM_BACK/camera_info, /CAM_BACK/image_rect_compressed spanning 5000 ms"
    )
    ctx = _ctx(summary=f"{dense} — {boilerplate}", embed_text=dense)
    # embed() follows embed_text, so it equals embedding the dense text alone...
    assert _cos(provider.embed(ctx), provider.embed_text(dense)) > 0.999
    # ...and is more like the dense text than like the boilerplate-laden summary.
    assert _cos(provider.embed(ctx), provider.embed_text(dense)) > _cos(
        provider.embed(ctx), provider.embed_text(ctx.summary)
    )


def test_hashed_text_embed_falls_back_to_summary_without_embed_text():
    # Description-less corpora carry no embed_text; the embedder must fall back to
    # the summary (then topics) exactly as before, so those lakes are unchanged.
    provider = HashedTextEmbeddingProvider(dimension=64)
    ctx = _ctx(summary="some structural caption text", embed_text=None)
    assert provider.embed(ctx) == provider.embed_text("some structural caption text")
    topics_only = _ctx(summary=None, embed_text=None, topics=("/imu", "/camera"))
    assert provider.embed(topics_only) == provider.embed_text("/imu /camera")


def test_hashed_text_ranks_similar_content_closer():
    provider = HashedTextEmbeddingProvider(dimension=128)
    anchor = provider.embed_text("pedestrian cut-in at dusk near the crosswalk")
    similar = provider.embed_text("pedestrian crossing at dusk near a crosswalk")
    unrelated = provider.embed_text("highway merge in bright daylight sunshine")
    assert _cos(anchor, similar) > _cos(anchor, unrelated)


def test_hashed_text_is_deterministic_across_instances():
    a = HashedTextEmbeddingProvider(dimension=32).embed_text("imu observations on run-abc")
    b = HashedTextEmbeddingProvider(dimension=32).embed_text("imu observations on run-abc")
    assert a == b


# --- hashed-image provider: near-duplicate frames are near neighbors (AC2) --


def _gradient_image(reversed_: bool = False, perturb: int = 0) -> bytes:
    out = bytearray(4096)
    for i in range(4096):
        value = (i // 64) * 4 % 256
        out[i] = (255 - value) if reversed_ else value
    for k in range(0, 4096, 200):  # a few perturbed bytes -> tiny mean shift
        out[k] = (out[k] + perturb) % 256
    return bytes(out)


def test_hashed_image_near_duplicate_is_nearer_than_unrelated():
    provider = HashedImageEmbeddingProvider(dimension=64)
    anchor = provider.embed_image(_gradient_image())
    near_dup = provider.embed_image(_gradient_image(perturb=20))
    unrelated = provider.embed_image(_gradient_image(reversed_=True))
    assert _cos(anchor, near_dup) > _cos(anchor, unrelated)
    assert _cos(anchor, near_dup) > 0.99  # a barely-perturbed frame is essentially identical


def test_hashed_image_is_image_only():
    provider = HashedImageEmbeddingProvider(dimension=16)
    assert math.isclose(math.sqrt(sum(v * v for v in provider.embed_image(b"abcd" * 100))), 1.0,
                        abs_tol=1e-6)
    with pytest.raises(NotImplementedError):
        provider.embed(_ctx())


# --- registry + degrade-safe resolution (AC4) -------------------------------


def test_resolve_demo_and_hashed_have_no_notice():
    demo, notice = resolve_embedding_provider("demo", dimension=16)
    assert demo.name == "demo-hash-v1" and demo.dimension == 16 and notice is None
    hashed, notice = resolve_embedding_provider("hashed-text", dimension=48)
    assert hashed.name == "hashed-text-v1" and hashed.dimension == 48 and notice is None


@pytest.mark.parametrize("name", ["clip", "sentence-transformers"])
def test_resolve_model_provider_without_extra_falls_back_to_demo(name):
    # The 'embeddings' extra is intentionally absent in CI: resolving a
    # model-backed provider must degrade to demo with a notice, never crash.
    # On a dev box where the extra IS installed (BUG-12) the fallback path does not
    # apply -- resolving builds the real model -- so skip rather than assert/false.
    assert PROVIDER_REGISTRY[name].requires_extra
    if embedding_extra_available(name):
        pytest.skip(f"{name} extra is installed; the missing-extra fallback path does not apply")
    provider, notice = resolve_embedding_provider(name, dimension=16)
    assert provider.name == "demo-hash-v1"
    assert notice and name in notice and "embeddings" in notice


@pytest.mark.parametrize("name", ["clip", "sentence-transformers"])
def test_resolve_model_provider_with_extra_builds_real_provider(name):
    # The complement of the fallback test: where the extra is present, resolving
    # must build the real model-backed provider with no fallback notice.
    if not embedding_extra_available(name):
        pytest.skip(f"{name} extra is not installed; the model-backed path is unavailable")
    provider, notice = resolve_embedding_provider(name, dimension=16)
    assert provider.name != "demo-hash-v1"
    assert notice is None


def test_resolve_unknown_provider_raises():
    with pytest.raises(EnrichmentError):
        resolve_embedding_provider("not-a-provider")


# --- BUG-11: strict mode (fallback=None) fails instead of silent demo fallback ---


def test_strict_resolution_raises_instead_of_demo_fallback(monkeypatch):
    """``fallback=None`` (the CLI ``--strict`` path) re-raises rather than degrading.

    Deterministic regardless of which extras are installed: a registry entry is
    forced to look unavailable, so both the safe-degrade default and the strict
    re-raise are exercised here, not just in whatever the host happens to have.
    """

    def boom(dimension=None):
        raise emb.EmbeddingExtraMissing(
            "embedding provider needs the optional dependency 'x'; "
            "install it with `pip install 'lancedb-robotics[embeddings]'`"
        )

    monkeypatch.setitem(
        emb.PROVIDER_REGISTRY,
        "sentence-transformers",
        emb.ProviderInfo(boom, requires_extra=True, modality="text"),
    )
    # default fallback degrades to demo with a notice (the safe, fail-open default)
    provider, notice = resolve_embedding_provider("sentence-transformers", dimension=16)
    assert provider.name == "demo-hash-v1" and notice
    # strict (fallback=None) re-raises instead of quietly using a demo embedder
    with pytest.raises(emb.EmbeddingExtraMissing):
        resolve_embedding_provider("sentence-transformers", dimension=16, fallback=None)


def test_strict_caption_resolution_raises_instead_of_demo_fallback():
    from lancedb_robotics.captions import (
        CaptionProviderUnavailable,
        resolve_caption_provider,
    )

    # vlm-api needs an endpoint that is absent in tests -> unavailable, deterministically.
    provider, notice = resolve_caption_provider("vlm-api")
    assert provider.name == "demo-template-v1" and notice  # safe default degrades
    with pytest.raises(CaptionProviderUnavailable):
        resolve_caption_provider("vlm-api", fallback=None)


# --- BUG-09: a present-but-broken model import degrades, never crashes -------


def test_import_extra_absent_extra_gives_install_hint():
    # The original failure mode: the dependency is simply not installed.
    with pytest.raises(emb.EmbeddingExtraMissing) as excinfo:
        emb._import_extra("lancedb_robotics_no_such_pkg", provider="sentence-transformers")
    message = str(excinfo.value)
    assert "lancedb_robotics_no_such_pkg" in message
    assert "lancedb-robotics[embeddings]" in message


def test_import_extra_wraps_broken_native_import(monkeypatch):
    # BUG-09: sentence-transformers 5.x is installed but eagerly imports
    # torchcodec, whose native dylib won't load -> OSError. _import_extra must
    # map that to EmbeddingExtraMissing (so resolve degrades) with an actionable
    # hint, and preserve the original error as __cause__ for debuggability.
    def boom(name):
        raise OSError("Could not load libtorchcodec_core4.dylib")

    monkeypatch.setattr(emb.importlib, "import_module", boom)
    with pytest.raises(emb.EmbeddingExtraMissing) as excinfo:
        emb._import_extra("json", provider="sentence-transformers", hint=emb._SENTENCE_TRANSFORMERS_HINT)
    message = str(excinfo.value)
    assert "failed to import" in message
    assert "OSError" in message
    assert "torchcodec" in message  # carried in via the hint
    assert isinstance(excinfo.value.__cause__, OSError)


def test_resolve_degrades_when_model_import_is_broken(monkeypatch):
    # End to end: a broken model import on the real provider path must degrade to
    # demo with a notice, exactly like an absent extra -- never propagate the
    # OSError out of resolve_embedding_provider (backlog 0021 AC4).
    real_find_spec = emb.importlib.util.find_spec
    real_import_module = emb.importlib.import_module

    def fake_find_spec(name, *args, **kwargs):
        if name == "sentence_transformers":
            return object()  # pretend it is installed
        return real_find_spec(name, *args, **kwargs)

    def fake_import_module(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise OSError("Could not load libtorchcodec_core4.dylib")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(emb.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(emb.importlib, "import_module", fake_import_module)

    provider, notice = resolve_embedding_provider("sentence-transformers", dimension=16)
    assert provider.name == "demo-hash-v1"
    assert notice and "sentence-transformers" in notice
    assert "failed to import" in notice and "torchcodec" in notice


# --- enrich with a real (content-based) provider records name + version -----


@pytest.fixture
def windowed_lake(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    return lake


def test_enrich_with_hashed_text_provider_is_selectable_and_versioned(windowed_lake):
    import json

    provider = HashedTextEmbeddingProvider(dimension=32)
    report = enrich_scenarios(windowed_lake, embedding_provider=provider)

    assert report.embedding_provider == "hashed-text-v1"
    assert report.embedding_provider_version == "1"
    assert report.embedding_dimension == 32
    rows = windowed_lake.table("scenarios").to_arrow().to_pylist()
    assert all(len(r["embedding"]) == 32 for r in rows)

    transform = next(
        r
        for r in windowed_lake.table("transform_runs").to_arrow().to_pylist()
        if r["transform_id"] == report.transform_id
    )
    params = json.loads(transform["params"])
    assert params["embedding_provider"] == "hashed-text-v1"
    assert params["embedding_provider_version"] == "1"  # AC5: name + version recorded


# --- observation image embeddings written additively onto a new column ------

_RUN_ID = "run-img"


def _image_obs_lake(tmp_path):
    """A lake whose camera observations carry controlled image bytes in payload_blob."""
    lake = Lake.init(tmp_path / "img.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": _RUN_ID, "run_kind": "drive", "raw_uri": "/src.mcap"}], schema=RUNS_SCHEMA
        )
    )
    rows = [
        {  # anchor frame
            "observation_id": f"{_RUN_ID}:/cam:000000",
            "run_id": _RUN_ID,
            "topic": "/cam",
            "modality": "image",
            "payload_blob": _gradient_image(),
            "decode_status": "decoded",
            "raw_sequence": 0,
        },
        {  # near-duplicate of the anchor
            "observation_id": f"{_RUN_ID}:/cam:000001",
            "run_id": _RUN_ID,
            "topic": "/cam",
            "modality": "image",
            "payload_blob": _gradient_image(perturb=20),
            "decode_status": "decoded",
            "raw_sequence": 1,
        },
        {  # an unrelated frame
            "observation_id": f"{_RUN_ID}:/cam:000002",
            "run_id": _RUN_ID,
            "topic": "/cam",
            "modality": "image",
            "payload_blob": _gradient_image(reversed_=True),
            "decode_status": "decoded",
            "raw_sequence": 2,
        },
        {  # a non-image row: must end up NULL in the image-embedding column
            "observation_id": f"{_RUN_ID}:/imu:000000",
            "run_id": _RUN_ID,
            "topic": "/imu",
            "modality": "imu",
            "payload_blob": None,
            "decode_status": "decoded",
            "raw_sequence": 3,
        },
    ]
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    return lake


def test_embed_observations_writes_image_vectors_and_ranks_near_duplicates(tmp_path):
    lake = _image_obs_lake(tmp_path)
    provider = HashedImageEmbeddingProvider(dimension=64)

    report = embed_observations(lake, provider)

    assert report.column == DEFAULT_IMAGE_EMBEDDING_COLUMN
    assert report.observations_embedded == 3  # the three image rows
    assert report.observations_skipped == 0
    assert report.embedding_provider == "hashed-image-v1"
    assert report.embedding_provider_version == "1"

    rows = {
        r["observation_id"]: r
        for r in lake.table("observations").to_arrow().to_pylist()
    }
    anchor = rows[f"{_RUN_ID}:/cam:000000"][DEFAULT_IMAGE_EMBEDDING_COLUMN]
    near = rows[f"{_RUN_ID}:/cam:000001"][DEFAULT_IMAGE_EMBEDDING_COLUMN]
    far = rows[f"{_RUN_ID}:/cam:000002"][DEFAULT_IMAGE_EMBEDDING_COLUMN]
    # The non-image row carries no vector in the image-embedding column.
    assert rows[f"{_RUN_ID}:/imu:000000"][DEFAULT_IMAGE_EMBEDDING_COLUMN] is None
    # Real visual similarity: near-duplicate frames are nearer neighbors (AC2).
    assert _cos(anchor, near) > _cos(anchor, far)


def test_embed_observations_records_lineage(tmp_path):
    import json

    lake = _image_obs_lake(tmp_path)
    report = embed_observations(lake, HashedImageEmbeddingProvider(dimension=32))

    transform = next(
        r
        for r in lake.table("transform_runs").to_arrow().to_pylist()
        if r["transform_id"] == report.transform_id
    )
    assert transform["kind"] == "embedding"
    assert transform["output_tables"] == ["observations"]
    params = json.loads(transform["params"])
    assert params["embedding_provider"] == "hashed-image-v1"
    assert params["embedding_provider_version"] == "1"
    assert params["column"] == DEFAULT_IMAGE_EMBEDDING_COLUMN
    assert params["observations_embedded"] == 3
