"""Prototype pluggable embedding pipeline (backlog 0189).

Parity gate: ``lake.embeddings.embed(EmbeddingSpec)`` must reproduce the legacy
``embed_observations`` result byte-for-byte (column values, dimension, count, and
the ``transform_runs`` lineage key). Plus generality checks: a text-column spec
runs through the *same* engine, and the SDK exposes providers/decoders.
"""

from __future__ import annotations

import pyarrow as pa

from lancedb_robotics.embeddings import (
    DEFAULT_IMAGE_EMBEDDING_COLUMN,
    EmbeddingSpec,
    HashedImageEmbeddingProvider,
    HashedTextEmbeddingProvider,
    Source,
    embed_observations,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA


def _seed(path, *, n: int = 40, with_caption: bool = False) -> Lake:
    lake = Lake.init(path)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": "run-img", "run_kind": "drive", "raw_uri": "/s.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    rows = []
    for i in range(n):
        row = {
            "observation_id": f"run-img:/cam:{i:06d}",
            "run_id": "run-img",
            "topic": "/cam",
            "modality": "image",
            "payload_blob": bytes((i + j) % 256 for j in range(256)),
            "decode_status": "decoded",
            "raw_sequence": i,
        }
        if with_caption:
            row["caption"] = f"a scene number {i % 5} with a robot arm"
        rows.append(row)
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    return lake


def _col(lake: Lake, column: str):
    return lake.table("observations").to_lance().to_table(columns=[column])[column].to_pylist()


def test_embed_spec_matches_embed_observations(tmp_path):
    """lake.embeddings.embed(EmbeddingSpec(...)) == embed_observations(...) byte-for-byte."""
    legacy = _seed(tmp_path / "legacy.lance")
    report_legacy = embed_observations(
        legacy, HashedImageEmbeddingProvider(dimension=16),
        checkpoint_file=str(tmp_path / "ckpt_legacy"),
    )

    sdk = _seed(tmp_path / "sdk.lance")
    report_sdk = sdk.embeddings.embed(
        EmbeddingSpec(
            provider=HashedImageEmbeddingProvider(dimension=16),
            target_column=DEFAULT_IMAGE_EMBEDDING_COLUMN,
        ),
        checkpoint_file=str(tmp_path / "ckpt_sdk"),
    )

    assert _col(sdk, DEFAULT_IMAGE_EMBEDDING_COLUMN) == _col(
        legacy, DEFAULT_IMAGE_EMBEDDING_COLUMN
    )
    assert report_sdk.embedding_dimension == report_legacy.embedding_dimension == 16
    assert report_sdk.observations_embedded == report_legacy.observations_embedded == 40
    # Identical pipeline identity -> identical lineage key.
    assert report_sdk.transform_id == report_legacy.transform_id


def test_embed_spec_text_column_through_same_engine(tmp_path):
    """A text spec (decode='text', no modality filter) runs through the same embed()."""
    lake = _seed(tmp_path / "text.lance", with_caption=True)
    report = lake.embeddings.embed(
        EmbeddingSpec(
            provider=HashedTextEmbeddingProvider(dimension=24),
            target_column="emb_caption",
            source=Source(
                input="caption", decoder="text", modality_column=None, modalities=()
            ),
        ),
        checkpoint_file=str(tmp_path / "ckpt_text"),
    )

    vecs = _col(lake, "emb_caption")
    assert report.embedding_dimension == 24
    assert len(vecs) == 40
    assert all(v is not None and len(v) == 24 for v in vecs)  # every captioned row embedded


def test_providers_and_decoders_listing(tmp_path):
    lake = _seed(tmp_path / "list.lance")
    providers = lake.embeddings.providers()
    assert providers["clip"]["modality"] == "text+image"
    assert providers["hashed-text"]["available"] is True  # dependency-free, always available
    assert lake.embeddings.decoders()["ros_image"] == "pil"


def test_runtime_register_custom_provider(tmp_path):
    """A provider registered at runtime embeds a column with no edit to embeddings.py."""
    from lancedb_robotics.enrich import EmbeddingProvider

    class ConstProvider(EmbeddingProvider):
        name = "const-test"
        version = "1"
        dimension = 8

        def embed(self, ctx):  # unused image-path provider
            raise NotImplementedError

        def embed_image(self, image: bytes):
            return [1.0] + [0.0] * 7

    lake = _seed(tmp_path / "custom.lance", n=20)
    lake.embeddings.register(
        "const-test", lambda dimension=None: ConstProvider(), modality="image"
    )
    report = lake.embeddings.embed(
        EmbeddingSpec(provider="const-test", target_column="emb_const"),
        checkpoint_file=str(tmp_path / "ckpt_const"),
    )
    vecs = _col(lake, "emb_const")
    assert report.embedding_dimension == 8
    assert any(v == [1.0] + [0.0] * 7 for v in vecs if v is not None)


def test_embed_batch_override_is_used_by_engine(tmp_path):
    """A provider overriding embed_batch drives the engine (no method-name sniffing)."""
    from lancedb_robotics.enrich import EmbeddingProvider

    class BatchProvider(EmbeddingProvider):
        name = "batch-test"
        version = "1"
        dimension = 4

        def embed(self, ctx):
            raise NotImplementedError

        # No embed_image/embed_pil_images/embed_images -> engine picks kind="image"
        # and calls embed_batch, which this provider serves in one shot.
        def embed_batch(self, inputs, *, kind):
            assert kind == "image"  # raw bytes, no batched image method advertised
            return [[float(len(b) % 7), 0.0, 0.0, 0.0] if b else None for b in inputs]

    lake = _seed(tmp_path / "batch.lance", n=20)
    lake.embeddings.register(
        "batch-test", lambda dimension=None: BatchProvider(), modality="image"
    )
    report = lake.embeddings.embed(
        EmbeddingSpec(provider="batch-test", target_column="emb_batch"),
        checkpoint_file=str(tmp_path / "ckpt_batch"),
    )
    vecs = _col(lake, "emb_batch")
    assert report.embedding_dimension == 4
    assert any(v is not None and len(v) == 4 for v in vecs)


def test_entry_point_provider_discovery(monkeypatch, tmp_path):
    """A provider advertised via an installed entry point resolves by name."""
    from lancedb_robotics import embeddings as emb

    class _FakeEP:
        name = "plugin-demo"

        def load(self):
            return emb.ProviderInfo(
                factory=lambda dimension=None: emb.HashedTextEmbeddingProvider(
                    dimension=dimension or 8
                ),
                requires_extra=False,
                modality="text",
            )

    monkeypatch.setattr(emb.metadata, "entry_points", lambda group=None: [_FakeEP()])
    monkeypatch.setattr(emb, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.delitem(emb.PROVIDER_REGISTRY, "plugin-demo", raising=False)
    try:
        provider, notice = emb.resolve_embedding_provider("plugin-demo", dimension=8)
        assert notice is None
        assert provider.dimension == 8
        assert "plugin-demo" in emb.PROVIDER_REGISTRY
    finally:
        emb.PROVIDER_REGISTRY.pop("plugin-demo", None)  # keep the global registry clean


def test_entry_point_broken_plugin_is_skipped(monkeypatch):
    """A plugin that fails to load never breaks resolution of a real provider."""
    from lancedb_robotics import embeddings as emb

    class _BadEP:
        name = "broken-plugin"

        def load(self):
            raise RuntimeError("plugin blew up")

    monkeypatch.setattr(emb.metadata, "entry_points", lambda group=None: [_BadEP()])
    monkeypatch.setattr(emb, "_ENTRY_POINTS_LOADED", False)
    # A broken plugin is swallowed; a built-in still resolves.
    provider, notice = emb.resolve_embedding_provider("hashed-text", dimension=8)
    assert provider.dimension == 8
    assert "broken-plugin" not in emb.PROVIDER_REGISTRY
