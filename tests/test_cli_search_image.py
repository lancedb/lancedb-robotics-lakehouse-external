"""CLI tests for `search image` (backlog 0187 -- the F3 text->image search half).

The pixel column is built with the dependency-free ``hashed-image`` provider, and
the query side is exercised through providers registered for the test whose
``embed_text`` is pinned to known image vectors -- so "the planted frame ranks
top-1" and "--exact equals brute-force ground truth" are assertable in CI without
CLIP weights, while going through exactly the code path CLIP uses.
"""

import math
import re

import pyarrow as pa
from typer.testing import CliRunner

from lancedb_robotics import embeddings as emb
from lancedb_robotics.cli import app
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA

runner = CliRunner()

_RUN_ID = "run-image-search"
_DIM = 32


def _frame_bytes(i: int) -> bytes:
    # One hot 16-byte chunk per frame: pooled by the 32-bucket hashed-image
    # provider this yields (near-)orthogonal vectors for distinct frames and
    # identical vectors for identical frames, so ranking and near-duplicate
    # collapse are deterministic. (Arbitrary byte patterns pool into the positive
    # orthant where everything sits above the 0.98 near-duplicate cosine.)
    return bytes(255 if (j // 16) == (i % 32) else (i // 32) for j in range(512))


def _seed_lake(path, *, frames: int, duplicates_of: int | None = None) -> Lake:
    lake = Lake.init(path)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": _RUN_ID, "run_kind": "drive", "raw_uri": "/logs/search.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    rows = [
        {
            "observation_id": f"{_RUN_ID}:/cam:{i:06d}",
            "run_id": _RUN_ID,
            "topic": "/cam",
            "modality": "image",
            "payload_blob": _frame_bytes(duplicates_of if duplicates_of is not None and i >= frames else i),
            "decode_status": "decoded",
            "raw_sequence": i,
            "timestamp_ns": 1_000_000 * i,
        }
        for i in range(frames + (2 if duplicates_of is not None else 0))
    ]
    rows.append(
        {  # a lidar row: must never surface in image search results
            "observation_id": f"{_RUN_ID}:/lidar:000000",
            "run_id": _RUN_ID,
            "topic": "/lidar",
            "modality": "lidar",
            "payload_blob": b"\x00" * 64,
            "decode_status": "decoded",
            "raw_sequence": frames + 10,
            "timestamp_ns": 0,
        }
    )
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    return lake


def _embed_column(lake_path, *, checkpoint, index: bool = False) -> None:
    args = [
        "embed", "observations", "--lake", str(lake_path),
        "--provider", "hashed-image", "--dimension", str(_DIM),
        "--checkpoint-file", str(checkpoint),
    ]
    if not index:
        args.append("--no-index")
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output


def _register_query_provider(name: str, vector: list[float]) -> None:
    """A provider whose text tower returns ``vector`` for any query -- the planted
    stand-in for CLIP's shared text+image space."""

    from lancedb_robotics.enrich import EmbeddingProvider

    class _Planted(EmbeddingProvider):
        dimension = len(vector)
        version = "1"

        def __init__(self):
            self.name = name

        def embed(self, ctx):
            raise NotImplementedError

        def embed_text(self, text: str) -> list[float]:
            return list(vector)

    emb.PROVIDER_REGISTRY[name] = emb.ProviderInfo(
        factory=lambda dimension=None: _Planted(), requires_extra=False, modality="text+image"
    )


def _ranked_ids(output: str) -> list[str]:
    return re.findall(r"^\d+\. (\S+)$", output, flags=re.MULTILINE)


def _stored_vectors(lake: Lake) -> dict[str, list[float] | None]:
    table = (
        lake.table("observations")
        .to_lance()
        .to_table(columns=["observation_id", "emb_image"], filter="modality = 'image'")
    )
    return dict(
        zip(table["observation_id"].to_pylist(), table["emb_image"].to_pylist(), strict=True)
    )


def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def test_search_image_planted_frame_ranks_top1(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _seed_lake(lake_path, frames=6)
    _embed_column(lake_path, checkpoint=tmp_path / "e.ckpt")

    planted_id = f"{_RUN_ID}:/cam:000003"
    planted_vector = _stored_vectors(lake)[planted_id]
    _register_query_provider("planted-query", planted_vector)
    try:
        result = runner.invoke(
            app,
            ["search", "image", "a planted scene", "--lake", str(lake_path),
             "--provider", "planted-query", "--limit", "3"],
        )
    finally:
        emb.PROVIDER_REGISTRY.pop("planted-query", None)

    assert result.exit_code == 0, result.output
    ranked = _ranked_ids(result.output)
    assert ranked and ranked[0] == planted_id
    assert "route: auto" in result.output
    assert "/lidar" not in result.output  # modality prefilter holds
    assert "source: /logs/search.mcap" in result.output


def test_search_image_exact_matches_brute_force_ground_truth(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _seed_lake(lake_path, frames=8)
    _embed_column(lake_path, checkpoint=tmp_path / "e.ckpt")

    query_vector = _stored_vectors(lake)[f"{_RUN_ID}:/cam:000005"]
    vectors = _stored_vectors(lake)
    truth = sorted(
        (oid for oid, v in vectors.items() if v is not None),
        key=lambda oid: (_l2(query_vector, vectors[oid]), oid),
    )[:5]

    _register_query_provider("truth-query", query_vector)
    try:
        result = runner.invoke(
            app,
            ["search", "image", "ground truth", "--lake", str(lake_path),
             "--provider", "truth-query", "--limit", "5", "--exact", "--no-diversify"],
        )
    finally:
        emb.PROVIDER_REGISTRY.pop("truth-query", None)

    assert result.exit_code == 0, result.output
    assert "route: exact" in result.output
    assert _ranked_ids(result.output) == truth


def test_search_image_indexed_default_matches_exact(tmp_path):
    """0186 correct-by-default: the scale-aware IVF_FLAT index returns the same
    top-k as --exact with no --refine-factor tuning required."""
    lake_path = tmp_path / "robot.lance"
    lake = _seed_lake(lake_path, frames=300)
    _embed_column(lake_path, checkpoint=tmp_path / "e.ckpt", index=True)

    query_vector = _stored_vectors(lake)[f"{_RUN_ID}:/cam:000042"]
    _register_query_provider("flat-query", query_vector)
    try:
        indexed = runner.invoke(
            app,
            ["search", "image", "q", "--lake", str(lake_path),
             "--provider", "flat-query", "--limit", "5", "--ann", "--no-diversify"],
        )
        exact = runner.invoke(
            app,
            ["search", "image", "q", "--lake", str(lake_path),
             "--provider", "flat-query", "--limit", "5", "--exact", "--no-diversify"],
        )
    finally:
        emb.PROVIDER_REGISTRY.pop("flat-query", None)

    assert indexed.exit_code == 0, indexed.output
    assert exact.exit_code == 0, exact.output
    assert "route: ann" in indexed.output
    assert _ranked_ids(indexed.output) == _ranked_ids(exact.output)


def test_search_image_diversify_collapses_duplicate_frames(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _seed_lake(lake_path, frames=5, duplicates_of=2)  # 2 extra clones of frame 2
    _embed_column(lake_path, checkpoint=tmp_path / "e.ckpt")

    target = f"{_RUN_ID}:/cam:000002"
    clones = {target, f"{_RUN_ID}:/cam:000005", f"{_RUN_ID}:/cam:000006"}
    query_vector = _stored_vectors(lake)[target]
    _register_query_provider("dup-query", query_vector)
    try:
        collapsed = runner.invoke(
            app,
            ["search", "image", "q", "--lake", str(lake_path),
             "--provider", "dup-query", "--limit", "4"],
        )
        every = runner.invoke(
            app,
            ["search", "image", "q", "--lake", str(lake_path),
             "--provider", "dup-query", "--limit", "4", "--no-diversify"],
        )
    finally:
        emb.PROVIDER_REGISTRY.pop("dup-query", None)

    assert collapsed.exit_code == 0, collapsed.output
    assert every.exit_code == 0, every.output
    # Diversified: exactly one representative of the identical-frame cluster.
    assert len(clones & set(_ranked_ids(collapsed.output))) == 1
    # Undiversified: the clones flood the top-k.
    assert clones <= set(_ranked_ids(every.output))


def test_search_image_ann_requires_index(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _seed_lake(lake_path, frames=4)
    _embed_column(lake_path, checkpoint=tmp_path / "e.ckpt")  # --no-index

    query_vector = _stored_vectors(lake)[f"{_RUN_ID}:/cam:000000"]
    _register_query_provider("ann-query", query_vector)
    try:
        result = runner.invoke(
            app,
            ["search", "image", "q", "--lake", str(lake_path),
             "--provider", "ann-query", "--ann"],
        )
    finally:
        emb.PROVIDER_REGISTRY.pop("ann-query", None)

    assert result.exit_code == 1
    assert "no vector index" in result.output


def test_search_image_exact_and_ann_are_mutually_exclusive(tmp_path):
    result = runner.invoke(
        app,
        ["search", "image", "q", "--lake", str(tmp_path / "x.lance"),
         "--exact", "--ann"],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_search_image_without_embedding_column_signposts_embed(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _seed_lake(lake_path, frames=2)

    result = runner.invoke(
        app,
        ["search", "image", "q", "--lake", str(lake_path), "--provider", "hashed-text"],
    )

    assert result.exit_code == 1
    assert "embed observations" in result.output  # actionable signpost


def test_search_image_text_only_provider_fails_clearly(tmp_path):
    lake_path = tmp_path / "robot.lance"
    _seed_lake(lake_path, frames=3)
    _embed_column(lake_path, checkpoint=tmp_path / "e.ckpt")

    result = runner.invoke(
        app,
        ["search", "image", "q", "--lake", str(lake_path), "--provider", "hashed-image"],
    )

    assert result.exit_code == 1
    assert "cannot embed a text query" in result.output


def test_search_image_where_filter_composes_with_modality(tmp_path):
    lake_path = tmp_path / "robot.lance"
    lake = _seed_lake(lake_path, frames=4)
    _embed_column(lake_path, checkpoint=tmp_path / "e.ckpt")

    query_vector = _stored_vectors(lake)[f"{_RUN_ID}:/cam:000000"]
    _register_query_provider("where-query", query_vector)
    try:
        result = runner.invoke(
            app,
            ["search", "image", "q", "--lake", str(lake_path),
             "--provider", "where-query", "--where", "run_id = 'no-such-run'"],
        )
    finally:
        emb.PROVIDER_REGISTRY.pop("where-query", None)

    assert result.exit_code == 0, result.output
    assert "results: 0" in result.output
