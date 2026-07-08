"""`lancedb-robotics search` subcommands (scalar / text / vector / hybrid / image).

``image`` (backlog 0187) is the observation-grain command: CLIP-style text->image
retrieval over a per-frame embedding column created by ``embed observations``,
returning frame identity + distance. ``image``, ``vector``, and ``hybrid`` all
carry the query-time ANN controls (backlog 0185): explicit routing
(``--exact``/``--ann``) and the tuning knobs (``--nprobes``/``--refine-factor``).
"""

import typer

search_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_WHERE_OPTION = typer.Option(None, "--where", help="SQL filter over scenario columns.")
_LIMIT_OPTION = typer.Option(10, "--limit", help="Maximum number of results to return.")
_RECORD_OPTION = typer.Option(
    True,
    "--record/--no-record",
    help="Record this search so `dataset snapshot create --from-search last` can use it.",
)
_PROVIDER_OPTION = typer.Option(
    "demo",
    "--provider",
    help=(
        "Embedding provider used to embed the QUERY -- must match the provider the "
        "embedding column was enriched with so query and column share a space "
        "(e.g. sentence-transformers, clip, hashed-text, demo)."
    ),
)
_DIVERSIFY_OPTION = typer.Option(
    True,
    "--diversify/--no-diversify",
    help=(
        "Collapse a scene's near-duplicate window-scenarios so the top-k spans "
        "distinct scenes. Use --no-diversify to return every matching window."
    ),
)
_COLUMN_OPTION = typer.Option(
    "embedding",
    "--column",
    help=(
        "Embedding column to search (default 'embedding'). Match it to the column "
        "the lake was enriched into, and pass the same --provider that built it so "
        "query and column vectors share one space."
    ),
)
_STRICT_OPTION = typer.Option(
    False,
    "--strict/--no-strict",
    help="Fail instead of embedding the query with a demo/hash stand-in when the "
    "requested --provider is unavailable (a mismatched query space ranks by noise).",
)
_QUERY_ARG = typer.Argument(..., help="Query text.")
# Query-time ANN controls (backlog 0185), shared by vector, hybrid, and image search.
_NPROBES_OPTION = typer.Option(
    None,
    "--nprobes",
    help="IVF partitions probed on the indexed path (default 64). More partitions "
    "widen coverage at some latency cost.",
)
_REFINE_FACTOR_OPTION = typer.Option(
    None,
    "--refine-factor",
    help="Re-rank the top limit*refine candidates with exact distances (default 50). "
    "For PQ-family indexes a non-trivial value is what recovers recall -- the "
    "default IVF_PQ query returned 0/10 recall at 57k rows without it.",
)
_EXACT_OPTION = typer.Option(
    False,
    "--exact",
    help="Bypass any ANN index and score every candidate exactly "
    "(correct-by-construction; fine at small scale).",
)
_ANN_OPTION = typer.Option(
    False,
    "--ann",
    help="Require the ANN index; error if the column has none. Default (neither "
    "flag) auto-routes: index when present, else brute-force.",
)


def _run(
    mode: str,
    lake: str,
    *,
    query: str | None,
    where: str | None,
    limit: int,
    record: bool = True,
    provider: str = "demo",
    diversify: bool = True,
    column: str = "embedding",
    strict: bool = False,
    nprobes: int | None = None,
    refine_factor: int | None = None,
    exact: bool = False,
    ann: bool = False,
) -> None:
    from lancedb_robotics.enrich import EnrichmentError
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.search import (
        ROUTE_ANN,
        ROUTE_AUTO,
        ROUTE_EXACT,
        SearchError,
        embedding_column_dimension,
        record_search,
        search_scenarios,
    )

    if exact and ann:
        typer.echo("error: --exact and --ann are mutually exclusive", err=True)
        raise typer.Exit(code=1)
    route = ROUTE_EXACT if exact else ROUTE_ANN if ann else ROUTE_AUTO

    try:
        opened = Lake.open(lake)
        # For vector/hybrid, embed the query with the same provider the column was
        # built with, so query and column vectors share one space (a demo-hashed
        # query against a model-built column ranks by noise). The dimension is read
        # from the *selected* column so a named column (e.g. embedding_minilm_384)
        # embeds the query in its own space, not the default column's.
        embedding_provider = None
        if mode in ("vector", "hybrid"):
            from lancedb_robotics.embeddings import resolve_embedding_provider

            dimension = embedding_column_dimension(opened, column)
            # --strict (fallback=None): refuse to embed the query with a demo/hash
            # stand-in when the requested provider is unavailable -- a mismatched
            # query space ranks by noise, so fail loud instead.
            embedding_provider, notice = resolve_embedding_provider(
                provider, dimension=dimension, **({"fallback": None} if strict else {})
            )
            if notice:
                typer.echo(f"notice: {notice}", err=True)
        results = search_scenarios(
            opened,
            mode=mode,
            query=query,
            where=where,
            limit=limit,
            embedding_provider=embedding_provider,
            embedding_column=column,
            nprobes=nprobes,
            refine_factor=refine_factor,
            route=route,
            diversify=diversify,
        )
        if record:
            record_search(
                opened,
                mode=mode,
                query=query,
                where=where,
                limit=limit,
                scenario_ids=[r.scenario_id for r in results],
            )
    except (LakeError, SearchError, EnrichmentError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"mode: {mode}")
    if query:
        typer.echo(f'query: "{query}"')
    if where:
        typer.echo(f"where: {where}")
    typer.echo(f"results: {len(results)}")
    for rank, result in enumerate(results, start=1):
        typer.echo(
            f"{rank}. {result.scenario_id}  {result.run_id}  "
            f"[{result.start_time_ns}..{result.end_time_ns} ns]"
        )
        typer.echo(f"     topics: {', '.join(result.topics) if result.topics else 'none'}")
        if result.summary:
            typer.echo(f"     summary: {result.summary}")
        scores = []
        if result.text_score is not None:
            scores.append(f"text={result.text_score:.4f}")
        if result.vector_distance is not None:
            scores.append(f"vector_distance={result.vector_distance:.4f}")
        if result.relevance_score is not None:
            scores.append(f"relevance={result.relevance_score:.4f}")
        if scores:
            typer.echo(f"     scores: {' '.join(scores)}")
        typer.echo(f"     source: {result.source_uri or 'unknown'}")


@search_app.command("scalar")
def scalar(
    lake: str = _LAKE_OPTION,
    where: str = _WHERE_OPTION,
    limit: int = _LIMIT_OPTION,
    record: bool = _RECORD_OPTION,
) -> None:
    """Filter scenarios by scalar metadata (no scoring)."""
    _run("scalar", lake, query=None, where=where, limit=limit, record=record)


@search_app.command("text")
def text(
    query: str = _QUERY_ARG,
    lake: str = _LAKE_OPTION,
    where: str = _WHERE_OPTION,
    limit: int = _LIMIT_OPTION,
    record: bool = _RECORD_OPTION,
) -> None:
    """Full-text search over scenario summaries (BM25)."""
    _run("text", lake, query=query, where=where, limit=limit, record=record)


@search_app.command("vector")
def vector(
    query: str = _QUERY_ARG,
    lake: str = _LAKE_OPTION,
    where: str = _WHERE_OPTION,
    limit: int = _LIMIT_OPTION,
    record: bool = _RECORD_OPTION,
    provider: str = _PROVIDER_OPTION,
    diversify: bool = _DIVERSIFY_OPTION,
    column: str = _COLUMN_OPTION,
    strict: bool = _STRICT_OPTION,
    nprobes: int | None = _NPROBES_OPTION,
    refine_factor: int | None = _REFINE_FACTOR_OPTION,
    exact: bool = _EXACT_OPTION,
    ann: bool = _ANN_OPTION,
) -> None:
    """Vector similarity search over scenario embeddings."""
    _run(
        "vector", lake, query=query, where=where, limit=limit,
        record=record, provider=provider, diversify=diversify, column=column, strict=strict,
        nprobes=nprobes, refine_factor=refine_factor, exact=exact, ann=ann,
    )


_IMAGE_COLUMN_OPTION = typer.Option(
    "emb_image",
    "--column",
    help=(
        "Observation embedding column to search (default 'emb_image'; e.g. "
        "embedding_clip_image). Pass the provider the column was built with so "
        "query and frame vectors share one space."
    ),
)
_IMAGE_PROVIDER_OPTION = typer.Option(
    "clip",
    "--provider",
    help=(
        "Provider whose TEXT tower embeds the query (default clip -- the shared "
        "text+image space). It must match the provider that built --column; a "
        "mismatched space ranks by noise."
    ),
)
_IMAGE_WHERE_OPTION = typer.Option(
    None, "--where", help="Extra SQL filter over observation columns (ANDed with --modality)."
)
_IMAGE_MODALITY_OPTION = typer.Option(
    None,
    "--modality",
    help="Observation modality to search; repeat for multiple. Default: image.",
)
_IMAGE_DIVERSIFY_OPTION = typer.Option(
    True,
    "--diversify/--no-diversify",
    help="Collapse near-duplicate frames (consecutive camera frames embed almost "
    "identically) so the top-k spans distinct moments. --no-diversify returns "
    "every matching frame.",
)


@search_app.command("image")
def image(
    query: str = _QUERY_ARG,
    lake: str = _LAKE_OPTION,
    column: str = _IMAGE_COLUMN_OPTION,
    provider: str = _IMAGE_PROVIDER_OPTION,
    where: str = _IMAGE_WHERE_OPTION,
    modality: list[str] | None = _IMAGE_MODALITY_OPTION,
    limit: int = _LIMIT_OPTION,
    nprobes: int | None = _NPROBES_OPTION,
    refine_factor: int | None = _REFINE_FACTOR_OPTION,
    exact: bool = _EXACT_OPTION,
    ann: bool = _ANN_OPTION,
    diversify: bool = _IMAGE_DIVERSIFY_OPTION,
    strict: bool = _STRICT_OPTION,
) -> None:
    """Find camera frames matching a text description (text->image similarity)."""
    from lancedb_robotics.enrich import EnrichmentError
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.search import (
        ROUTE_ANN,
        ROUTE_AUTO,
        ROUTE_EXACT,
        SearchError,
        observation_embedding_dimension,
        search_observations,
    )

    if exact and ann:
        typer.echo("error: --exact and --ann are mutually exclusive", err=True)
        raise typer.Exit(code=1)
    route = ROUTE_EXACT if exact else ROUTE_ANN if ann else ROUTE_AUTO

    try:
        opened = Lake.open(lake)
        from lancedb_robotics.embeddings import resolve_embedding_provider

        # Embed the query at the column's dimension with the provider that built
        # the column, so query and frame vectors share one space. --strict refuses
        # the demo/hash stand-in fallback (a mismatched space ranks by noise).
        dimension = observation_embedding_dimension(opened, column)
        embedding_provider, notice = resolve_embedding_provider(
            provider, dimension=dimension, **({"fallback": None} if strict else {})
        )
        if notice:
            typer.echo(f"notice: {notice}", err=True)
        results = search_observations(
            opened,
            query=query,
            embedding_provider=embedding_provider,
            column=column,
            modalities=tuple(modality) if modality else ("image",),
            where=where,
            limit=limit,
            nprobes=nprobes,
            refine_factor=refine_factor,
            route=route,
            diversify=diversify,
        )
    except (LakeError, SearchError, EnrichmentError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"lake: {opened.uri}")
    typer.echo(f'query: "{query}"')
    typer.echo(f"column: {column}")
    typer.echo(f"route: {route}")
    if where:
        typer.echo(f"where: {where}")
    typer.echo(f"results: {len(results)}")
    for rank, result in enumerate(results, start=1):
        typer.echo(f"{rank}. {result.observation_id}")
        typer.echo(
            f"     run: {result.run_id}  topic: {result.topic or 'unknown'}  "
            f"t={result.timestamp_ns}"
        )
        if result.vector_distance is not None:
            typer.echo(f"     distance: {result.vector_distance:.4f}")
        typer.echo(f"     source: {result.source_uri or 'unknown'}")


@search_app.command("hybrid")
def hybrid(
    query: str = _QUERY_ARG,
    lake: str = _LAKE_OPTION,
    where: str = _WHERE_OPTION,
    limit: int = _LIMIT_OPTION,
    record: bool = _RECORD_OPTION,
    provider: str = _PROVIDER_OPTION,
    diversify: bool = _DIVERSIFY_OPTION,
    column: str = _COLUMN_OPTION,
    strict: bool = _STRICT_OPTION,
    nprobes: int | None = _NPROBES_OPTION,
    refine_factor: int | None = _REFINE_FACTOR_OPTION,
    exact: bool = _EXACT_OPTION,
    ann: bool = _ANN_OPTION,
) -> None:
    """Hybrid text + vector search fused by reciprocal rank fusion."""
    _run(
        "hybrid", lake, query=query, where=where, limit=limit,
        record=record, provider=provider, diversify=diversify, column=column, strict=strict,
        nprobes=nprobes, refine_factor=refine_factor, exact=exact, ann=ann,
    )
