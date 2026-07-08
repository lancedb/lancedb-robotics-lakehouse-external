"""Baseline search over enriched scenario rows.

One query surface (backlog 0008 / Feature Set 6) over the canonical lake: the
same ``scenarios`` rows are queried by scalar metadata, full-text over the
``summary`` caption, vector similarity over the demo ``embedding``, and a hybrid
of text + vector. Search is LanceDB-native — FTS index, vector search, and a
reciprocal-rank-fusion (RRF) reranker — so the demo dogfoods the unified
substrate instead of stitching together sidecar indexes.

Every result exposes IDs, the scenario time window, transparent score
components (``text_score``/BM25, ``vector_distance``, RRF ``relevance_score``),
and a source link back to the raw log. Ranking is made deterministic for
fixtures by re-sorting engine output with an explicit ``scenario_id`` tie-break,
so equal scores never reorder between runs.

Backlog 0187 adds :func:`search_observations`: text->image similarity over an
*observation*-level embedding column (per-camera-frame vectors created by
``embed observations``), with explicit query routing (auto/exact/ann) carrying
the 0185 ANN controls, bounded projected reads, and the same near-duplicate
diversification at frame grain.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa
from lancedb.rerankers import RRFReranker

from lancedb_robotics.enrich import (
    DEFAULT_EMBEDDING_COLUMN,
    DemoEmbeddingProvider,
    EmbeddingProvider,
)
from lancedb_robotics.indexing import has_fts_index, has_vector_index, is_fts_index_stale
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA

SCALAR = "scalar"
TEXT = "text"
VECTOR = "vector"
HYBRID = "hybrid"
MODES = (SCALAR, TEXT, VECTOR, HYBRID)

# Query routing for vector/hybrid search (introduced for observation image search
# in backlog 0187, extended to scenario search in 0185): ``auto`` uses the ANN
# index when one exists else brute-force, ``exact`` bypasses any index
# (correct-by-construction), ``ann`` requires one and errors when absent.
ROUTE_AUTO = "auto"
ROUTE_EXACT = "exact"
ROUTE_ANN = "ann"
ROUTES = (ROUTE_AUTO, ROUTE_EXACT, ROUTE_ANN)
#: Back-compat alias from the 0187 increment, before routing covered scenarios too.
OBSERVATION_ROUTES = ROUTES

SEARCH_TRANSFORM_KIND = "search"

_SUMMARY_COLUMN = "summary"

# When a persistent ANN index exists on the embedding column, the vector path
# hits the index instead of scanning every row. We probe enough partitions to
# keep recall high and re-rank with exact distances (``refine_factor``), then
# over-fetch a small bounded set so the deterministic ``scenario_id`` tie-break
# re-sort still has candidates to order. None of this fetches the whole table --
# that unconditional full scan is the thing 0021 drops on the indexed path.
_DEFAULT_NPROBES = 64
_DEFAULT_REFINE_FACTOR = 50
_VECTOR_INDEX_OVERFETCH = 64
_TEXT_INDEX_OVERFETCH = 64

# Vector/hybrid results are diversified so a scene's near-identical window-scenarios
# don't flood the top-k (BUG-07): once the embedded text is description-forward, a
# scene's windows share one description and embed to (nearly) the same vector. We
# collapse candidates whose embeddings are within this cosine of an already-kept
# result, keeping the best-ranked representative. Matches the curation dedup
# convention (``curate.py`` near_duplicate_threshold) so "distinct scenes" means the
# same thing at query time and at curation time.
_NEAR_DUPLICATE_COSINE = 0.98


class SearchError(Exception):
    """Raised when a search cannot be run as requested."""


@dataclass(frozen=True)
class SearchResult:
    """One ranked scenario hit with transparent score components."""

    scenario_id: str
    run_id: str
    start_time_ns: int
    end_time_ns: int
    topics: tuple[str, ...]
    summary: str | None
    source_uri: str | None
    text_score: float | None = None
    vector_distance: float | None = None
    relevance_score: float | None = None


@dataclass(frozen=True)
class ObservationSearchResult:
    """One ranked observation (camera-frame) hit from text->image search (0187)."""

    observation_id: str
    run_id: str
    topic: str | None
    timestamp_ns: int | None
    modality: str | None
    source_uri: str | None
    vector_distance: float | None = None


def _source_uris(lake: Lake) -> dict[str, str]:
    """Map run_id -> raw_uri so results can link back to the source log."""
    return {row["run_id"]: row["raw_uri"] for row in lake.table("runs").to_arrow().to_pylist()}


def _embedding_dimension(table, column: str) -> int | None:
    if column not in table.schema.names:
        return None
    return table.schema.field(column).type.list_size


def embedding_column_dimension(lake: Lake, column: str = DEFAULT_EMBEDDING_COLUMN) -> int | None:
    """The fixed-size-list dimension of a scenarios embedding column, or ``None``.

    Lets the CLI resolve a *query*-embedding provider at the column's dimension, so
    a free-text query is embedded in the same space the column was built in (the
    dependency-free demo/hashed providers take a dimension; the model-backed ones
    own their own and ignore it).
    """
    return _embedding_dimension(lake.table("scenarios"), column)


def observation_embedding_dimension(lake: Lake, column: str) -> int | None:
    """The fixed-size-list dimension of an observations embedding column, or ``None``.

    The observation-grain counterpart of :func:`embedding_column_dimension`, so the
    image-search CLI can resolve its query-embedding provider at the dimension the
    column was built with (backlog 0187).
    """
    return _embedding_dimension(lake.table("observations"), column)


def _to_result(row: dict, source_uris: dict[str, str]) -> SearchResult:
    return SearchResult(
        scenario_id=row["scenario_id"],
        run_id=row["run_id"],
        start_time_ns=row["start_time_ns"],
        end_time_ns=row["end_time_ns"],
        topics=tuple(row.get("topics") or ()),
        summary=row.get(_SUMMARY_COLUMN),
        source_uri=source_uris.get(row["run_id"]),
        text_score=row.get("_score"),
        vector_distance=row.get("_distance"),
        relevance_score=row.get("_relevance_score"),
    )


def _require_fresh_fts_index(lake: Lake, table, column: str) -> None:
    if not has_fts_index(table, column):
        raise SearchError(
            f"no persistent FTS index on scenarios.{column}; "
            "run `lancedb-robotics scenarios enrich --fts-index` or "
            "`lancedb-robotics scenarios index-fts` first"
        )
    if is_fts_index_stale(lake, table="scenarios", column=column):
        raise SearchError(
            f"persistent FTS index on scenarios.{column} is stale; "
            "run `lancedb-robotics scenarios index-fts` to refresh it"
        )


def search_scenarios(
    lake: Lake,
    *,
    mode: str,
    query: str | None = None,
    where: str | None = None,
    limit: int = 10,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
    nprobes: int | None = None,
    refine_factor: int | None = None,
    route: str = ROUTE_AUTO,
    diversify: bool = True,
) -> list[SearchResult]:
    """Search scenario rows by ``mode`` and return deterministically ranked hits.

    Modes: ``scalar`` (metadata filter only), ``text`` (FTS over summaries),
    ``vector`` (similarity to the query embedding), and ``hybrid`` (text + vector
    fused by RRF). ``text``/``vector``/``hybrid`` require a ``query``;
    ``vector``/``hybrid`` require an enriched ``embedding`` column.

    The text/hybrid path requires a managed persistent FTS index over
    ``summary``; it is built at enrich/index time and never rebuilt on the query
    hot path. The vector/hybrid path follows ``route`` (backlog 0185, same
    contract as :func:`search_observations`): ``auto`` (default) uses a persistent
    ANN index when one exists on ``embedding_column`` -- probing it with
    ``nprobes``/``refine_factor`` and over-fetching a bounded candidate set --
    and stays exact brute force over the full set otherwise; ``exact`` bypasses
    any index so results are correct by construction; ``ann`` requires the index
    and errors when absent. For the PQ index family a non-trivial
    ``refine_factor`` is what recovers recall. Either way results are re-sorted
    with an explicit ``scenario_id`` tie-break, so ranking is deterministic.

    For ``vector``/``hybrid``, ``diversify`` (default on) collapses candidates whose
    embeddings are near-duplicates (cosine >= ``_NEAR_DUPLICATE_COSINE``) of an
    already-kept hit, so a scene's near-identical window-scenarios don't flood the
    top-k (BUG-07) -- the top-k spans distinct scenes. It is a no-op when candidate
    vectors are genuinely distinct (e.g. the structural demo embedder), so it never
    over-collapses. Pass ``diversify=False`` to get every matching window (e.g. to
    seed a snapshot from all windows of the matched scenes).
    """
    if mode not in MODES:
        raise SearchError(f"unknown search mode {mode!r}; expected one of {', '.join(MODES)}")
    if route not in ROUTES:
        raise SearchError(f"unknown route {route!r}; expected one of {', '.join(ROUTES)}")
    if mode != SCALAR and not (query and query.strip()):
        raise SearchError(f"{mode} search requires a query string")

    table = lake.table("scenarios")
    total = table.count_rows()
    if total == 0:
        raise SearchError(f"no scenarios to search in {lake.uri}; run `scenarios create` first")
    source_uris = _source_uris(lake)
    if mode in (TEXT, HYBRID):
        _require_fresh_fts_index(lake, table, _SUMMARY_COLUMN)

    query_vector = None
    has_index = False
    use_index = False
    if mode in (VECTOR, HYBRID):
        dimension = _embedding_dimension(table, embedding_column)
        if dimension is None:
            raise SearchError(
                f"no {embedding_column!r} column in {lake.uri}; "
                "run `scenarios enrich` to add embeddings first"
            )
        provider = embedding_provider or DemoEmbeddingProvider(dimension=dimension)
        query_vector = provider.embed_text(query)
        has_index = has_vector_index(table, embedding_column)
        if route == ROUTE_ANN and not has_index:
            raise SearchError(
                f"--ann requested but scenarios.{embedding_column} has no vector index; "
                "build one with `scenarios index` (or enrich --index) or drop --ann"
            )
        use_index = has_index and route != ROUTE_EXACT

    # With an ANN index, over-fetch a bounded candidate set (not the whole table)
    # and re-rank exactly; without one, fall back to the exact full-set scan so
    # small/un-indexed lakes stay deterministic.
    if mode == TEXT:
        fetch_k = max(limit, _TEXT_INDEX_OVERFETCH)
    elif mode == HYBRID:
        fetch_k = max(limit, _VECTOR_INDEX_OVERFETCH if use_index else _TEXT_INDEX_OVERFETCH)
    else:
        fetch_k = max(limit, _VECTOR_INDEX_OVERFETCH) if use_index else total

    if mode == SCALAR:
        builder = table.search()
    elif mode == TEXT:
        builder = table.search(query, query_type="fts")
    elif mode == VECTOR:
        # Name the vector column explicitly: a lake enriched with more than one
        # embedding column (e.g. text MiniLM + CLIP) is ambiguous otherwise, and
        # LanceDB only disambiguates the plain vector query by dimension.
        builder = table.search(
            query_vector, vector_column_name=embedding_column, query_type="vector"
        )
        if use_index:
            builder = builder.nprobes(nprobes or _DEFAULT_NPROBES).refine_factor(
                refine_factor or _DEFAULT_REFINE_FACTOR
            )
        elif has_index:  # route == "exact" with an index present: force brute force
            builder = builder.bypass_vector_index()
    else:  # HYBRID
        # Hybrid does not infer the vector column from the query dimension, so it
        # must be named explicitly once the lake carries multiple embedding columns.
        builder = (
            table.search(query_type="hybrid", vector_column_name=embedding_column)
            .text(query)
            .vector(query_vector)
            .rerank(RRFReranker(return_score="all"))
        )
        if use_index:
            builder = builder.nprobes(nprobes or _DEFAULT_NPROBES).refine_factor(
                refine_factor or _DEFAULT_REFINE_FACTOR
            )
        elif has_index:  # route == "exact": bypass the index on the vector leg too
            builder = builder.bypass_vector_index()

    if where:
        builder = builder.where(where)
    rows = builder.limit(fetch_k).to_list()

    # Keep the candidate embeddings alongside the results so near-duplicate scene
    # windows can be collapsed below (BUG-07) without a second fetch.
    vectors = {row["scenario_id"]: row.get(embedding_column) for row in rows}
    results = [_to_result(row, source_uris) for row in rows]
    results.sort(key=_sort_key(mode))
    if diversify and mode in (VECTOR, HYBRID):
        return _collapse_near_duplicates(results, vectors, _NEAR_DUPLICATE_COSINE, limit)
    return results[:limit]


#: Frame-identity columns projected by observation search. The payload (blob)
#: column is deliberately never fetched -- results carry identity + score, and the
#: caller pulls pixels separately if it wants them (bounded-reads discipline).
_OBSERVATION_IDENTITY_COLUMNS = (
    "observation_id",
    "run_id",
    "topic",
    "timestamp_ns",
    "modality",
)


def _quote_sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def search_observations(
    lake: Lake,
    *,
    query: str,
    embedding_provider: EmbeddingProvider,
    column: str,
    modalities: tuple[str, ...] = ("image",),
    where: str | None = None,
    limit: int = 10,
    nprobes: int | None = None,
    refine_factor: int | None = None,
    route: str = ROUTE_AUTO,
    diversify: bool = True,
) -> list[ObservationSearchResult]:
    """Text->image similarity search over an observation embedding column (0187).

    Embeds ``query`` with ``embedding_provider`` (its text tower must share the
    space the ``column`` vectors were built in -- CLIP for pixels) and returns
    frame identity + distance for the nearest camera frames. ``route`` carries the
    0185 controls: ``auto`` (default) rides the ANN index when one exists and
    stays brute-force otherwise; ``exact`` bypasses any index so results are
    correct by construction; ``ann`` requires the index and errors when absent.
    ``nprobes``/``refine_factor`` tune the indexed path -- for the PQ family a
    non-trivial ``refine_factor`` is what recovers recall (the 0185 spike measured
    0/10 recall@10 without it at 57k rows).

    Reads are bounded: only identity columns (+ the vector when diversifying) are
    projected, a bounded candidate set is fetched (never the whole table), and the
    payload blob column is never touched. ``modalities`` prefilters to camera rows
    (rows outside it hold NULL vectors by construction); ``where`` ANDs a caller
    filter on top. ``diversify`` collapses near-duplicate frames (cosine >=
    ``_NEAR_DUPLICATE_COSINE``) so the top-k spans distinct moments.
    """
    if not (query and query.strip()):
        raise SearchError("image search requires a query string")
    if route not in OBSERVATION_ROUTES:
        raise SearchError(
            f"unknown route {route!r}; expected one of {', '.join(OBSERVATION_ROUTES)}"
        )
    table = lake.table("observations")
    if table.count_rows() == 0:
        raise SearchError(f"no observations to search in {lake.uri}; ingest a run first")
    dimension = _embedding_dimension(table, column)
    if dimension is None:
        raise SearchError(
            f"no {column!r} vector column on observations in {lake.uri}; "
            "run `lancedb-robotics embed observations` to create it first"
        )
    try:
        query_vector = embedding_provider.embed_text(query)
    except NotImplementedError as exc:
        raise SearchError(
            f"embedding provider {embedding_provider.name!r} cannot embed a text query; "
            "image search needs a text+image provider (e.g. clip) whose text tower "
            "shares the column's space"
        ) from exc
    if len(query_vector) != dimension:
        raise SearchError(
            f"query embedding dimension {len(query_vector)} does not match "
            f"observations.{column} (dimension {dimension}); pass the provider the "
            "column was built with"
        )

    has_index = has_vector_index(table, column)
    if route == ROUTE_ANN and not has_index:
        raise SearchError(
            f"--ann requested but observations.{column} has no vector index; "
            "build one with `embed observations --index` or drop --ann"
        )
    use_index = has_index and route != ROUTE_EXACT

    builder = table.search(query_vector, vector_column_name=column, query_type="vector")
    if use_index:
        builder = builder.nprobes(nprobes or _DEFAULT_NPROBES).refine_factor(
            refine_factor or _DEFAULT_REFINE_FACTOR
        )
    elif has_index:  # route == "exact" with an index present: force brute force
        builder = builder.bypass_vector_index()

    filters = []
    if modalities:
        quoted = ", ".join(_quote_sql_literal(m) for m in modalities)
        filters.append(f"modality IN ({quoted})")
    if where:
        filters.append(f"({where})")
    if filters:
        builder = builder.where(" AND ".join(filters), prefilter=True)

    identity = [c for c in _OBSERVATION_IDENTITY_COLUMNS if c in table.schema.names]
    projection = identity + ([column] if diversify else [])
    # Bounded candidate fetch: over-fetch enough for the deterministic re-sort and
    # the near-duplicate collapse to have candidates, never the whole table (the
    # observations grain is the largest table in any real lake).
    fetch_k = max(limit, _VECTOR_INDEX_OVERFETCH)
    rows = builder.select(projection).limit(fetch_k).to_list()

    source_uris = _source_uris(lake)
    vectors = {row["observation_id"]: row.get(column) for row in rows}
    results = [
        ObservationSearchResult(
            observation_id=row["observation_id"],
            run_id=row["run_id"],
            topic=row.get("topic"),
            timestamp_ns=row.get("timestamp_ns"),
            modality=row.get("modality"),
            source_uri=source_uris.get(row["run_id"]),
            vector_distance=row.get("_distance"),
        )
        for row in rows
    ]
    results.sort(
        key=lambda r: (
            r.vector_distance if r.vector_distance is not None else float("inf"),
            r.observation_id,
        )
    )
    if diversify:
        return _collapse_near_duplicates(
            results, vectors, _NEAR_DUPLICATE_COSINE, limit, key=lambda r: r.observation_id
        )
    return results[:limit]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _collapse_near_duplicates(
    results: list,
    vectors: dict[str, list[float] | None],
    threshold: float,
    limit: int,
    key=lambda r: r.scenario_id,
) -> list:
    """Keep the best-ranked representative of each near-duplicate vector cluster.

    Walks the already-ranked results and drops any hit whose embedding is within
    ``threshold`` cosine of one already kept -- so a scene's near-identical
    window-scenarios (or a camera's near-identical consecutive frames, keyed by
    ``key``) collapse to a single top hit and the top-k spans distinct scenes /
    moments (BUG-07, 0187). A hit with no stored vector is never treated as a
    duplicate, so the path degrades to "return everything" rather than dropping
    rows. O(kept x candidates) over a bounded candidate set; stops once ``limit``
    distinct hits are kept.
    """
    kept: list = []
    kept_vectors: list[list[float]] = []
    for result in results:
        vector = vectors.get(key(result))
        if vector is not None and any(
            _cosine(vector, other) >= threshold for other in kept_vectors
        ):
            continue
        kept.append(result)
        if vector is not None:
            kept_vectors.append(vector)
        if len(kept) >= limit:
            break
    return kept


def _sort_key(mode: str):
    if mode == TEXT:
        return lambda r: (-(r.text_score or 0.0), r.scenario_id)
    if mode == VECTOR:
        return lambda r: (
            r.vector_distance if r.vector_distance is not None else float("inf"),
            r.scenario_id,
        )
    if mode == HYBRID:
        return lambda r: (-(r.relevance_score or 0.0), r.scenario_id)
    return lambda r: (r.start_time_ns, r.scenario_id)


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _search_transforms(lake: Lake) -> list[tuple[dict, dict]]:
    """Recorded search rows as ``(row, parsed_params)`` pairs."""
    return [
        (row, json.loads(row["params"]) if row["params"] else {})
        for row in lake.table("transform_runs").to_arrow().to_pylist()
        if row["kind"] == SEARCH_TRANSFORM_KIND
    ]


def record_search(
    lake: Lake,
    *,
    mode: str,
    query: str | None,
    where: str | None,
    limit: int,
    scenario_ids: list[str],
    created_by: str = "lancedb-robotics",
) -> str:
    """Record a search and its result as a ``kind="search"`` transform row.

    Lets ``dataset snapshot create --from-search last`` resolve the most recent
    selection deterministically. A monotonic ``sequence`` (max existing + 1)
    orders searches independent of wall-clock, and one row is kept per distinct
    query (re-running the same query refreshes its sequence to become "last").
    """
    existing = _search_transforms(lake)
    sequence = max((params.get("sequence", 0) for _, params in existing), default=0) + 1
    spec = {
        "mode": mode,
        "query": query,
        "where": where,
        "limit": limit,
        "scenario_ids": list(scenario_ids),
        "sequence": sequence,
    }
    transform_id = "tfm-search-" + _digest(
        {"mode": mode, "query": query, "where": where, "limit": limit}
    )
    now = datetime.now(UTC)
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    transforms.add(
        pa.Table.from_pylist(
            [
                {
                    "transform_id": transform_id,
                    "kind": SEARCH_TRANSFORM_KIND,
                    "input_uris": [],
                    "input_table_versions": [],
                    "output_tables": [],
                    "params": json.dumps(spec, sort_keys=True),
                    "status": "completed",
                    "started_at": now,
                    "finished_at": now,
                    "created_by": created_by,
                    "created_at": now,
                }
            ],
            schema=TRANSFORM_RUNS_SCHEMA,
        )
    )
    return transform_id


def last_search(lake: Lake) -> dict | None:
    """Parsed spec of the most recently recorded search, or ``None``."""
    recorded = _search_transforms(lake)
    if not recorded:
        return None
    row, params = max(recorded, key=lambda rp: (rp[1].get("sequence", 0), rp[0]["transform_id"]))
    return params
