"""Persistent ANN vector index + index-aware search (backlog 0021, AC3/AC5).

AC3: after indexing a corpus, vector search uses the ANN index (the query plan
shows ``ANNSubIndex``, not a flat full scan) and returns the same top-k as exact
brute force -- and stays deterministic. AC5 (index half): the enrichment lineage
records the index params, including the ``skipped`` reason below the training
floor.

IVF_PQ needs >= 256 rows to train, so the index tests build a synthetic
300-scenario lake with distinct unit vectors; the small fixture covers the
below-floor skip path.
"""

import hashlib
import json

import pyarrow as pa
import pytest

from lancedb_robotics.embeddings import HashedImageEmbeddingProvider
from lancedb_robotics.enrich import EmbeddingProvider, ScenarioContext, enrich_scenarios
from lancedb_robotics.indexing import (
    ALIGNED_FRAME_PREDICATE_INDEX_COLUMNS,
    ALIGNED_TICK_PREDICATE_INDEX_COLUMNS,
    CURATION_REVIEW_QUEUE_PREDICATE_INDEX_COLUMNS,
    MIN_INDEX_ROWS,
    SUPPORTED_VECTOR_INDEX_TYPES,
    IndexingError,
    IndexSpec,
    build_aligned_training_predicate_indexes,
    build_curation_predicate_indexes,
    build_fts_index,
    build_review_queue_predicate_indexes,
    build_scalar_index,
    build_scalar_indexes,
    build_vector_index,
    describe_curation_predicate_indexes,
    describe_review_queue_predicate_indexes,
    has_fts_index,
    has_scalar_index,
    has_vector_index,
)
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.scenarios import create_scenario_windows
from lancedb_robotics.schemas import SCENARIOS_SCHEMA
from lancedb_robotics.search import VECTOR, search_scenarios


class _SeededVectorProvider(EmbeddingProvider):
    """Deterministic distinct unit vectors per seed -- a controllable stand-in.

    Distinct vectors per ``scenario_id`` make brute-force top-k unambiguous (no
    ties), so an exact match against the index path is a clean correctness check.
    """

    name = "seeded-test"

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def embed(self, ctx: ScenarioContext) -> list[float]:
        return self._vector(ctx.scenario_id)

    def embed_text(self, text: str) -> list[float]:
        return self._vector(f"query::{text}")

    def _vector(self, seed: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
            for offset in range(0, len(digest), 4):
                if len(values) >= self.dimension:
                    break
                word = int.from_bytes(digest[offset : offset + 4], "big")
                values.append(word / 0xFFFFFFFF * 2.0 - 1.0)
            counter += 1
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class _FakeSchema:
    def __init__(self, names):
        self.names = list(names)


class _FakeIndex:
    def __init__(self, index_type, columns):
        self.index_type = index_type
        self.columns = list(columns)


class _FakeScalarTable:
    def __init__(self, names, *, indices=(), error=None, expose_create=True):
        self.name = "fake"
        self.schema = _FakeSchema(names)
        self.indices = list(indices)
        self.error = error
        self.calls = []
        if expose_create:
            self.create_scalar_index = self._create_scalar_index

    def count_rows(self):
        return 42

    def list_indices(self):
        return list(self.indices)

    def _create_scalar_index(self, column, *, replace=True, index_type="BTREE"):
        self.calls.append((column, replace, index_type))
        if self.error is not None:
            raise self.error
        self.indices = [
            index
            for index in self.indices
            if column not in getattr(index, "columns", ())
        ]
        self.indices.append(_FakeIndex("BTree", [column]))


class _FakeLake:
    def __init__(self, tables):
        self.tables = dict(tables)

    def table(self, name):
        return self.tables[name]


def _seeded_scenario_rows(start: int, stop: int) -> list[dict]:
    return [
        {
            "scenario_id": f"scn-{i:04d}",
            "run_id": "run-synthetic",
            "start_time_ns": i * 1000,
            "end_time_ns": i * 1000 + 500,
            "window_ns": 500,
            "is_partial": False,
            "topics": ["/imu"],
            "observation_ids": [f"obs-{i}"],
            "observation_count": 1,
        }
        for i in range(start, stop)
    ]


def _seeded_scenario_lake(path, *, n: int = 300, auto_index: bool):
    """A lake of ``n`` synthetic scenarios with distinct seeded vectors.

    ``n >= MIN_INDEX_ROWS`` puts it above the IVF_PQ training floor. ``auto_index``
    controls whether ``enrich`` auto-builds the ANN index (backlog 0183).
    """
    lake = Lake.init(path)
    lake.table("scenarios").add(
        pa.Table.from_pylist(_seeded_scenario_rows(0, n), schema=SCENARIOS_SCHEMA)
    )
    enrich_scenarios(
        lake, embedding_provider=_SeededVectorProvider(dimension=16), auto_index=auto_index
    )
    return lake


@pytest.fixture
def big_lake(tmp_path):
    """A 300-scenario lake (above the IVF_PQ floor); left un-indexed for explicit-build tests."""
    return _seeded_scenario_lake(tmp_path / "big.lance", auto_index=False)


@pytest.fixture
def small_lake(tmp_path, fixtures_dir):
    lake = Lake.init(tmp_path / "small.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=50_000_000)
    enrich_scenarios(lake)
    return lake


# --- build / skip / detect --------------------------------------------------


def test_below_floor_skips_index_and_stays_searchable(small_lake):
    result = build_vector_index(small_lake, table="scenarios", column="embedding")

    assert result.status == "skipped"
    assert str(MIN_INDEX_ROWS) in result.reason
    assert not has_vector_index(small_lake.table("scenarios"), "embedding")
    # Brute-force vector search still works without an index.
    hits = search_scenarios(small_lake, mode=VECTOR, query="imu")
    assert hits


def test_build_creates_a_detectable_index(big_lake):
    assert not has_vector_index(big_lake.table("scenarios"), "embedding")
    result = build_vector_index(big_lake, table="scenarios", column="embedding")

    assert result.status == "built"
    # Scale-aware default (backlog 0186): 300 vectors is far below the
    # PQ-meaningful floor, so the default type is IVF_FLAT (no PQ recall footgun)
    # and the downgrade is recorded, never silent.
    assert result.index_type == "IVF_FLAT"
    assert result.num_rows == 300
    assert result.num_partitions and result.num_sub_vectors is None
    assert result.reason and "scale-aware default" in result.reason
    assert has_vector_index(big_lake.table("scenarios"), "embedding")


def test_default_index_type_is_ivf_pq_at_pq_meaningful_scale(big_lake, monkeypatch):
    """Above the floor the default stays IVF_PQ (floor lowered to fixture size)."""
    from lancedb_robotics import indexing as indexing_module

    monkeypatch.setattr(indexing_module, "PQ_MEANINGFUL_ROWS", 300)
    result = build_vector_index(big_lake, table="scenarios", column="embedding")

    assert result.status == "built"
    assert result.index_type == "IVF_PQ"
    assert result.reason is None  # the at-scale default is not a downgrade


def test_explicit_index_type_wins_over_scale_default(big_lake):
    """An explicit ivf_pq below the floor is honored (never overridden)."""
    result = build_vector_index(
        big_lake, table="scenarios", column="embedding",
        spec=IndexSpec(index_type="ivf_pq"),
    )

    assert result.status == "built"
    assert result.index_type == "IVF_PQ"
    assert result.reason is None


def test_scale_decision_counts_vectors_not_table_rows(tmp_path, monkeypatch):
    """A mostly-NULL vector column (the observation image case) keys the type
    decision on non-null vectors: 300 table rows but only 150 vectors stays
    IVF_FLAT even with the floor at 200."""
    from lancedb_robotics import indexing as indexing_module

    lake = _seeded_scenario_lake(tmp_path / "sparse.lance", auto_index=False)
    table = lake.table("scenarios")
    table.update(where="start_time_ns >= 150000", values={"embedding": None})
    assert table.count_rows("embedding IS NOT NULL") == 150

    monkeypatch.setattr(indexing_module, "PQ_MEANINGFUL_ROWS", 200)
    result = build_vector_index(lake, table="scenarios", column="embedding")

    assert result.status == "built"
    assert result.index_type == "IVF_FLAT"  # 150 vectors < 200, despite 300 rows
    assert result.reason and "150 vectors" in result.reason


def test_enrich_auto_index_uses_scale_aware_default(tmp_path):
    """The enrich auto-index path inherits the scale-aware default (0186 AC)."""
    lake = Lake.init(tmp_path / "auto-flat.lance")
    lake.table("scenarios").add(
        pa.Table.from_pylist(_seeded_scenario_rows(0, 300), schema=SCENARIOS_SCHEMA)
    )
    report = enrich_scenarios(
        lake, embedding_provider=_SeededVectorProvider(dimension=16), auto_index=True
    )

    assert report.index and report.index["index_type"] == "IVF_FLAT"
    assert "scale-aware default" in (report.index.get("reason") or "")


def test_build_fts_index_creates_a_detectable_index(small_lake):
    result = build_fts_index(small_lake, table="scenarios", column="summary")

    assert result.status == "built"
    assert result.index_type == "FTS"
    assert result.num_rows == small_lake.table("scenarios").count_rows()
    assert has_fts_index(small_lake.table("scenarios"), "summary")
    assert not has_vector_index(small_lake.table("scenarios"), "summary")


def test_scalar_index_helper_builds_aligned_tick_predicate_columns():
    tick_table = _FakeScalarTable(ALIGNED_TICK_PREDICATE_INDEX_COLUMNS)
    frame_table = _FakeScalarTable(ALIGNED_FRAME_PREDICATE_INDEX_COLUMNS)
    lake = _FakeLake({"aligned_ticks": tick_table, "aligned_frames": frame_table})

    results = build_aligned_training_predicate_indexes(lake)

    assert [call[0] for call in tick_table.calls] == list(
        ALIGNED_TICK_PREDICATE_INDEX_COLUMNS
    )
    assert [call[0] for call in frame_table.calls] == list(
        ALIGNED_FRAME_PREDICATE_INDEX_COLUMNS
    )
    assert {result.status for result in results} == {"built"}


def test_curation_predicate_index_helper_requests_hot_columns():
    membership_columns = [
        "view_id",
        "target_grain",
        "target_id",
        "scenario_id",
        "decision",
        "queue",
        "created_at",
    ]
    chunk_columns = ["view_id", "chunk_index", "start_ordinal", "end_ordinal"]
    memberships = _FakeScalarTable(membership_columns)
    chunks = _FakeScalarTable(chunk_columns)
    lake = _FakeLake(
        {
            "curation_memberships": memberships,
            "curation_view_membership_chunks": chunks,
        }
    )

    results = build_curation_predicate_indexes(lake)

    assert [call[0] for call in memberships.calls] == membership_columns
    assert [call[0] for call in chunks.calls] == chunk_columns
    assert {result.status for result in results} == {"built"}


def test_curation_predicate_index_status_skips_unsupported_backend():
    membership_columns = [
        "view_id",
        "target_grain",
        "target_id",
        "scenario_id",
        "decision",
        "queue",
        "created_at",
    ]
    chunk_columns = ["view_id", "chunk_index", "start_ordinal", "end_ordinal"]
    memberships = _FakeScalarTable(membership_columns, expose_create=False)
    chunks = _FakeScalarTable(chunk_columns, expose_create=False)
    lake = _FakeLake(
        {
            "curation_memberships": memberships,
            "curation_view_membership_chunks": chunks,
        }
    )

    results = describe_curation_predicate_indexes(lake)

    assert {result.status for result in results} == {"skipped"}
    assert all("create_scalar_index" in str(result.reason) for result in results)


def test_review_queue_predicate_index_helper_requests_hot_columns():
    table = _FakeScalarTable(CURATION_REVIEW_QUEUE_PREDICATE_INDEX_COLUMNS)
    lake = _FakeLake({"curation_review_queues": table})

    results = build_review_queue_predicate_indexes(lake)

    assert [call[0] for call in table.calls] == list(
        CURATION_REVIEW_QUEUE_PREDICATE_INDEX_COLUMNS
    )
    assert {result.status for result in results} == {"built"}


def test_review_queue_predicate_index_status_skips_unsupported_backend():
    table = _FakeScalarTable(
        CURATION_REVIEW_QUEUE_PREDICATE_INDEX_COLUMNS,
        expose_create=False,
    )
    lake = _FakeLake({"curation_review_queues": table})

    results = describe_review_queue_predicate_indexes(lake)

    assert {result.status for result in results} == {"skipped"}
    assert all("create_scalar_index" in str(result.reason) for result in results)


def test_scalar_index_helper_detects_already_present_without_rebuild():
    table = _FakeScalarTable(
        ["alignment_id"],
        indices=[_FakeIndex("BTree", ["alignment_id"])],
    )
    lake = _FakeLake({"aligned_ticks": table})

    result = build_scalar_index(lake, table="aligned_ticks", column="alignment_id")

    assert result.status == "already_present"
    assert table.calls == []
    assert has_scalar_index(table, "alignment_id")


def test_scalar_index_helper_skips_unsupported_backend():
    table = _FakeScalarTable(["alignment_id"], expose_create=False)
    lake = _FakeLake({"aligned_ticks": table})

    results = build_scalar_indexes(lake, table="aligned_ticks", columns=["alignment_id"])

    assert results[0].status == "skipped"
    assert "create_scalar_index" in results[0].reason


def test_scalar_index_helper_reports_failed_build():
    table = _FakeScalarTable(["alignment_id"], error=RuntimeError("catalog denied"))
    lake = _FakeLake({"aligned_ticks": table})

    result = build_scalar_index(lake, table="aligned_ticks", column="alignment_id")

    assert result.status == "failed"
    assert "catalog denied" in result.reason


# --- index-aware search: uses the index, matches brute force, deterministic --


def test_indexed_vector_search_matches_bruteforce_topk(big_lake):
    provider = _SeededVectorProvider(dimension=16)
    query = "a rare scenario to find"
    query_vector = provider.embed_text(query)
    k = 5

    # Exact brute-force reference, computed independently of the engine.
    stored = [
        (r["scenario_id"], list(r["embedding"]))
        for r in big_lake.table("scenarios").to_arrow().to_pylist()
    ]
    reference = [
        sid
        for sid, _ in sorted(stored, key=lambda sv: (-_cos(query_vector, sv[1]), sv[0]))[:k]
    ]

    build_vector_index(big_lake, table="scenarios", column="embedding")
    results = search_scenarios(
        big_lake, mode=VECTOR, query=query, embedding_provider=provider, limit=k
    )

    assert [r.scenario_id for r in results] == reference


def test_indexed_search_uses_the_ann_index_not_a_full_scan(big_lake):
    build_vector_index(big_lake, table="scenarios", column="embedding")
    table = big_lake.table("scenarios")
    query_vector = _SeededVectorProvider(dimension=16).embed_text("find")

    # Brute force scans every row (no ANN sub-index); the indexed path hits it.
    brute_plan = (
        table.search(query_vector, query_type="vector").bypass_vector_index().limit(5).explain_plan()
    )
    indexed_plan = (
        table.search(query_vector, query_type="vector").nprobes(64).limit(5).explain_plan()
    )
    assert "ANNSubIndex" not in brute_plan
    assert "ANNSubIndex" in indexed_plan


def test_indexed_vector_search_is_deterministic(big_lake):
    provider = _SeededVectorProvider(dimension=16)
    build_vector_index(big_lake, table="scenarios", column="embedding")
    first = [r.scenario_id for r in search_scenarios(
        big_lake, mode=VECTOR, query="imu", embedding_provider=provider)]
    second = [r.scenario_id for r in search_scenarios(
        big_lake, mode=VECTOR, query="imu", embedding_provider=provider)]
    assert first == second


# --- AC5: enrich-time index params are recorded in the enrichment lineage ---


def test_enrich_records_index_params_even_when_skipped(small_lake):
    report = enrich_scenarios(small_lake, index=IndexSpec())

    assert report.index is not None
    assert report.index["status"] == "skipped"
    transform = next(
        r
        for r in small_lake.table("transform_runs").to_arrow().to_pylist()
        if r["transform_id"] == report.transform_id
    )
    params = json.loads(transform["params"])
    assert params["index"]["status"] == "skipped"
    assert params["index"]["column"] == "embedding"


def test_observation_index_builds_over_a_new_column(tmp_path):
    # An image-embedding column on observations gets its own independent index
    # (decision 0025) once the corpus is large enough.
    from lancedb_robotics.embeddings import DEFAULT_IMAGE_EMBEDDING_COLUMN, embed_observations
    from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA

    lake = Lake.init(tmp_path / "obs.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": "run-img", "run_kind": "drive", "raw_uri": "/s.mcap"}], schema=RUNS_SCHEMA
        )
    )
    rows = [
        {
            "observation_id": f"run-img:/cam:{i:06d}",
            "run_id": "run-img",
            "topic": "/cam",
            "modality": "image",
            "payload_blob": bytes((i + j) % 256 for j in range(256)),
            "decode_status": "decoded",
            "raw_sequence": i,
        }
        for i in range(MIN_INDEX_ROWS + 20)
    ]
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))

    report = embed_observations(
        lake, HashedImageEmbeddingProvider(dimension=16), index=IndexSpec()
    )

    assert report.index is not None and report.index["status"] == "built"
    assert has_vector_index(lake.table("observations"), DEFAULT_IMAGE_EMBEDDING_COLUMN)


# --- replace-embedding rebuilds the dropped ANN index at the new dim (BUG-08) -


def test_replace_embedding_rebuilds_vector_index_at_new_dimension(big_lake):
    """A ``replace_embedding`` dim migration restores the index Lance drops on the column.

    Dropping the column drops its IVF_PQ index (verified: no dangling index), so a
    naive replace would silently leave search on a full brute-force scan. The
    replace path captures the pre-drop index and rebuilds it at the new dimension;
    the FTS index over ``summary`` is a different column and must survive untouched.
    """
    build_fts_index(big_lake, table="scenarios", column="summary")
    build_vector_index(big_lake, table="scenarios", column="embedding")
    table = big_lake.table("scenarios")
    assert has_vector_index(table, "embedding")
    assert has_fts_index(table, "summary")

    report = enrich_scenarios(
        big_lake,
        embedding_provider=_SeededVectorProvider(dimension=32),
        embedding_column="embedding",
        replace_embedding=True,
        fts_index=False,
    )

    assert report.embedding_dimension == 32
    assert report.index is not None and report.index["status"] == "built"
    assert report.index["dimension"] == 32

    table = big_lake.table("scenarios")
    assert table.schema.field("embedding").type.list_size == 32
    assert has_vector_index(table, "embedding")  # rebuilt at the new dim, not lost
    assert has_fts_index(table, "summary")  # the embedding swap leaves FTS untouched
    # Vector search runs against the migrated, re-indexed column.
    assert search_scenarios(big_lake, mode=VECTOR, query="imu", embedding_column="embedding")


def test_replace_embedding_without_prior_index_does_not_build_one(big_lake):
    """If the column had no ANN index, a replace re-embeds but does not restore one.

    Isolates the restore-on-replace logic from auto-indexing (``auto_index=False``):
    a replace only re-creates an index that *existed* before, it does not invent one.
    """
    assert not has_vector_index(big_lake.table("scenarios"), "embedding")

    report = enrich_scenarios(
        big_lake,
        embedding_provider=_SeededVectorProvider(dimension=32),
        embedding_column="embedding",
        replace_embedding=True,
        auto_index=False,
        fts_index=False,
    )

    assert report.embedding_dimension == 32
    assert report.index is None  # nothing to restore, and none requested
    assert not has_vector_index(big_lake.table("scenarios"), "embedding")


# --- auto-build the ANN index at scale (backlog 0183 / round-4 §5) -----------
# The IVF_PQ path was never exercised on a real lake because every spike lake had
# only ~40 scenarios (< MIN_INDEX_ROWS), so the index never built and search stayed
# brute-force. enrich now auto-builds the index once there are enough scenarios to
# train IVF_PQ, so the ANN path engages by default at scale.


def test_enrich_auto_builds_ann_index_at_scale(tmp_path):
    lake = _seeded_scenario_lake(tmp_path / "auto.lance", auto_index=True)
    assert lake.table("scenarios").count_rows() >= MIN_INDEX_ROWS
    # no explicit build_vector_index call -- enrich built it because it could.
    assert has_vector_index(lake.table("scenarios"), "embedding")
    hits = search_scenarios(
        lake, mode=VECTOR, query="anything", embedding_provider=_SeededVectorProvider(dimension=16)
    )
    assert hits  # the index serves queries


def test_enrich_auto_index_can_be_disabled_at_scale(tmp_path):
    lake = _seeded_scenario_lake(tmp_path / "noauto.lance", auto_index=False)
    assert lake.table("scenarios").count_rows() >= MIN_INDEX_ROWS
    assert not has_vector_index(lake.table("scenarios"), "embedding")  # opted out


def test_enrich_does_not_auto_build_below_floor(small_lake):
    # The small fixture is below MIN_INDEX_ROWS, so the default enrich must skip the
    # index and leave search exact brute-force (the correct small-scale behavior).
    assert small_lake.table("scenarios").count_rows() < MIN_INDEX_ROWS
    assert not has_vector_index(small_lake.table("scenarios"), "embedding")
    assert search_scenarios(small_lake, mode=VECTOR, query="imu")  # still searchable


def test_auto_index_covers_appended_scenarios(tmp_path):
    """A re-enrich after appending scenarios re-indexes so the new rows are findable."""
    lake = _seeded_scenario_lake(tmp_path / "grow.lance", n=300, auto_index=True)
    provider = _SeededVectorProvider(dimension=16)
    lake.table("scenarios").add(
        pa.Table.from_pylist(_seeded_scenario_rows(300, 400), schema=SCENARIOS_SCHEMA)
    )
    enrich_scenarios(lake, embedding_provider=provider, auto_index=True)
    table = lake.table("scenarios")
    assert table.count_rows() == 400
    assert has_vector_index(table, "embedding")
    # an appended scenario is indexed: an exact-vector search returns it top-1.
    appended_id = "scn-0350"
    query_vector = provider._vector(appended_id)  # the exact vector enrich stored for it
    top = table.search(query_vector, vector_column_name="embedding").limit(1).to_list()
    assert top and top[0]["scenario_id"] == appended_id


# --- configurable vector index type + per-type params ------------------------
# build_vector_index hardcoded IVF_PQ; IndexSpec.index_type now selects any of
# SUPPORTED_VECTOR_INDEX_TYPES and threads the params each family needs
# (num_sub_vectors/num_bits for PQ, m/ef_construction for HNSW). Search already
# rides the IVF family via nprobes/refine_factor, so a non-PQ index must serve
# queries the same way.


@pytest.mark.parametrize("index_type", SUPPORTED_VECTOR_INDEX_TYPES)
def test_build_supports_each_vector_index_type(big_lake, index_type):
    result = build_vector_index(
        big_lake, table="scenarios", column="embedding", spec=IndexSpec(index_type=index_type)
    )

    assert result.status == "built"
    assert result.index_type == index_type
    assert result.num_partitions  # every IVF type partitions
    # Sub-vectors are a PQ-only concept; non-PQ types must not carry one.
    assert (result.num_sub_vectors is not None) == index_type.endswith("PQ")
    assert has_vector_index(big_lake.table("scenarios"), "embedding")


def test_build_rejects_unsupported_index_type(big_lake):
    with pytest.raises(IndexingError, match="unsupported vector index type"):
        build_vector_index(
            big_lake, table="scenarios", column="embedding", spec=IndexSpec(index_type="bogus")
        )


def test_index_type_is_case_insensitive(big_lake):
    result = build_vector_index(
        big_lake, table="scenarios", column="embedding", spec=IndexSpec(index_type="ivf_flat")
    )

    assert result.status == "built"
    assert result.index_type == "IVF_FLAT"


def test_hnsw_index_records_graph_params(big_lake):
    result = build_vector_index(
        big_lake,
        table="scenarios",
        column="embedding",
        spec=IndexSpec(index_type="IVF_HNSW_SQ", m=16, ef_construction=120),
    )

    assert result.status == "built"
    assert result.index_type == "IVF_HNSW_SQ"
    assert result.m == 16
    assert result.ef_construction == 120
    assert result.num_sub_vectors is None  # SQ quantizer, not PQ
    assert has_vector_index(big_lake.table("scenarios"), "embedding")


def test_pq_index_records_num_bits(big_lake):
    result = build_vector_index(
        big_lake,
        table="scenarios",
        column="embedding",
        spec=IndexSpec(index_type="IVF_PQ", num_bits=8),
    )

    assert result.status == "built"
    assert result.num_bits == 8
    assert result.num_sub_vectors  # PQ still auto-tunes sub-vectors


def test_below_floor_skip_reports_requested_index_type(small_lake):
    # Validation + the requested type are reported even when the row count is below
    # the training floor, so the skip reason stays honest about what was requested.
    result = build_vector_index(
        small_lake,
        table="scenarios",
        column="embedding",
        spec=IndexSpec(index_type="IVF_HNSW_SQ"),
    )

    assert result.status == "skipped"
    assert result.index_type == "IVF_HNSW_SQ"
    assert "IVF_HNSW_SQ" in result.reason


def test_search_works_over_non_pq_index(big_lake):
    provider = _SeededVectorProvider(dimension=16)
    build_vector_index(
        big_lake,
        table="scenarios",
        column="embedding",
        spec=IndexSpec(index_type="IVF_HNSW_SQ"),
    )
    assert has_vector_index(big_lake.table("scenarios"), "embedding")

    hits = search_scenarios(
        big_lake, mode=VECTOR, query="imu", embedding_provider=provider, limit=5
    )
    again = search_scenarios(
        big_lake, mode=VECTOR, query="imu", embedding_provider=provider, limit=5
    )
    assert hits
    assert [h.scenario_id for h in hits] == [h.scenario_id for h in again]


def test_enrich_records_chosen_index_type_in_lineage(big_lake):
    report = enrich_scenarios(
        big_lake,
        embedding_provider=_SeededVectorProvider(dimension=16),
        index=IndexSpec(index_type="IVF_FLAT"),
        auto_index=False,
    )

    assert report.index is not None
    assert report.index["status"] == "built"
    assert report.index["index_type"] == "IVF_FLAT"
    assert report.index["num_sub_vectors"] is None
    transform = next(
        r
        for r in big_lake.table("transform_runs").to_arrow().to_pylist()
        if r["transform_id"] == report.transform_id
    )
    params = json.loads(transform["params"])
    assert params["index"]["index_type"] == "IVF_FLAT"


def test_cli_index_builds_requested_type(big_lake):
    from typer.testing import CliRunner

    from lancedb_robotics.cli import app

    result = CliRunner().invoke(
        app,
        ["scenarios", "index", "--lake", big_lake.uri, "--index-type", "ivf_flat"],
    )

    assert result.exit_code == 0, result.output
    assert "IVF_FLAT" in result.output
    assert has_vector_index(big_lake.table("scenarios"), "embedding")
