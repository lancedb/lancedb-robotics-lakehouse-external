"""`lancedb-robotics scenarios` subcommands."""

import typer

from lancedb_robotics.cli.lineage_context import echo_emitted_lineage

scenarios_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_WINDOW_OPTION = typer.Option(..., "--window", help="Fixed window duration, for example 5s.")
_TOPIC_OPTION = typer.Option(
    None,
    "--topic",
    "-t",
    help="Topic to include; repeat for multiple exact topic filters.",
)
_INCLUDE_PARTIAL_OPTION = typer.Option(
    True,
    "--include-partial/--drop-partial",
    help="Whether to materialize a final shorter-than-window segment.",
)
_EMBED_OPTION = typer.Option(
    True,
    "--embed/--no-embed",
    help="Also write embedding vectors, or write summary text only.",
)
_DIMENSION_OPTION = typer.Option(
    16, "--dimension", help="Embedding dimension (honored by the demo/hashed providers)."
)
_EMBEDDING_COLUMN_OPTION = typer.Option(
    "embedding",
    "--embedding-column",
    help=(
        "Scenario column to write embeddings into (default 'embedding'). Name a "
        "second column (e.g. embedding_minilm_384) to keep multiple embedding "
        "spaces side by side instead of overwriting the default one."
    ),
)
_REPLACE_EMBEDDING_OPTION = typer.Option(
    False,
    "--replace-embedding/--no-replace-embedding",
    help=(
        "Drop and rebuild the target embedding column from scratch -- required to "
        "switch to a different-dimension provider in place. Any ANN index on the "
        "column is rebuilt at the new dimension; the FTS index over summaries is "
        "untouched. Without this, a dimension change fails fast."
    ),
)
_PROVIDER_OPTION = typer.Option(
    "demo",
    "--provider",
    help=(
        "Embedding provider. Semantic vector/hybrid search needs a model-backed "
        "provider: sentence-transformers / clip (the 'embeddings' extra). "
        "demo (default, deterministic structural hash) and hashed-text "
        "(caption-content, dependency-free) are deterministic offline stand-ins. "
        "A model provider whose extra is absent falls back to demo with a warning; "
        "pass --strict to error instead."
    ),
)
_CAPTION_PROVIDER_OPTION = typer.Option(
    "demo",
    "--caption-provider",
    help=(
        "Caption provider: demo (default), image-stats (camera payload statistics), "
        "or vlm-api (uses LANCEDB_ROBOTICS_CAPTION_ENDPOINT/API_KEY; falls back to "
        "demo with a warning unless --strict is set)."
    ),
)
_BUILD_INDEX_OPTION = typer.Option(
    False,
    "--index/--no-index",
    help="Force a persistent ANN index over the embedding column now (backlog 0021).",
)
_AUTO_INDEX_OPTION = typer.Option(
    True,
    "--auto-index/--no-auto-index",
    help="Auto-build the ANN index once there are enough scenarios to train IVF_PQ "
    "(default on; no-op below the floor, so search stays exact at small scale).",
)
_BUILD_FTS_INDEX_OPTION = typer.Option(
    True,
    "--fts-index/--no-fts-index",
    help="Build or refresh the persistent FTS index over scenario summaries.",
)
_INDEX_COLUMN_OPTION = typer.Option(
    "embedding", "--column", help="Embedding column to index."
)
_FTS_INDEX_COLUMN_OPTION = typer.Option(
    "summary", "--column", help="Text column to index for full-text search."
)
_INDEX_METRIC_OPTION = typer.Option(
    "cosine", "--metric", help="Distance metric for the ANN index (cosine|l2|dot)."
)
_INDEX_TYPE_OPTION = typer.Option(
    None,
    "--index-type",
    help=(
        "Vector index type: ivf_pq, ivf_flat, ivf_sq, ivf_hnsw_flat, ivf_hnsw_sq, "
        "ivf_hnsw_pq. Default is scale-aware (backlog 0186): ivf_flat below the "
        "PQ-meaningful floor (~65k vectors -- PQ there only costs recall and needs "
        "a large --refine-factor at query time), ivf_pq at scale. An explicit "
        "value always wins. The per-type tuning flags below only apply to the "
        "family that has them (num-sub-vectors/num-bits for *_pq, m/"
        "ef-construction for ivf_hnsw_*)."
    ),
)
_INDEX_NUM_PARTITIONS_OPTION = typer.Option(
    None,
    "--num-partitions",
    help="IVF partition count (all index types). Auto-tuned to the corpus size when unset.",
)
_INDEX_NUM_SUB_VECTORS_OPTION = typer.Option(
    None,
    "--num-sub-vectors",
    help="PQ sub-vector count (ivf_pq/ivf_hnsw_pq only). Must divide the dimension; auto when unset.",
)
_INDEX_NUM_BITS_OPTION = typer.Option(
    None,
    "--num-bits",
    help="PQ bits per sub-vector (ivf_pq/ivf_hnsw_pq only). Engine default when unset.",
)
_INDEX_HNSW_M_OPTION = typer.Option(
    None,
    "--m",
    help="HNSW graph connectivity m (ivf_hnsw_* only). Engine default when unset.",
)
_INDEX_EF_CONSTRUCTION_OPTION = typer.Option(
    None,
    "--ef-construction",
    help="HNSW build-time candidate list size (ivf_hnsw_* only). Engine default when unset.",
)


def _index_spec_from_cli(
    *,
    index_type: str | None,
    metric: str,
    num_partitions: int | None,
    num_sub_vectors: int | None,
    num_bits: int | None,
    m: int | None,
    ef_construction: int | None,
):
    """Build an ``IndexSpec`` from the shared CLI index-tuning flags."""
    from lancedb_robotics.indexing import IndexSpec

    return IndexSpec(
        index_type=index_type,
        metric=metric,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        num_bits=num_bits,
        m=m,
        ef_construction=ef_construction,
    )


def _index_params_overridden(spec) -> bool:
    """True when the user tuned any vector-index flag away from the defaults.

    Lets ``enrich`` treat e.g. ``--index-type ivf_hnsw_sq`` as a build request
    without a separate ``--index`` flag (the build still skips below the training
    floor, so it stays a no-op on small lakes). ``index_type=None`` is the
    scale-aware *default*, not an override.
    """
    from lancedb_robotics.indexing import DEFAULT_METRIC

    return (
        spec.index_type is not None
        or spec.metric != DEFAULT_METRIC
        or spec.num_partitions is not None
        or spec.num_sub_vectors is not None
        or spec.num_bits is not None
        or spec.m is not None
        or spec.ef_construction is not None
    )


def _echo_index(index: dict) -> None:
    """Print an index build/skip outcome in a stable, human-readable form."""
    if index["status"] != "built":
        typer.echo(f"vector index: skipped ({index['reason']})")
        return
    parts = [f"{index['num_partitions']} partitions"]
    if index.get("num_sub_vectors"):
        sub = f"{index['num_sub_vectors']} sub-vectors"
        if index.get("num_bits"):
            sub += f" x {index['num_bits']} bits"
        parts.append(sub)
    if index.get("m"):
        parts.append(f"m={index['m']}")
    if index.get("ef_construction"):
        parts.append(f"ef_construction={index['ef_construction']}")
    typer.echo(
        f"vector index: built ({index['index_type']} {index['metric']}, "
        f"{' / '.join(parts)} over {index['num_rows']} rows)"
    )
    if index.get("reason"):
        # The scale-aware default downgraded the type -- say so, never silently.
        typer.echo(f"index note: {index['reason']}")


def _echo_fts_index(index: dict) -> None:
    typer.echo(
        f"fts index: {index['status']} ({index['index_type']} over "
        f"{index['table']}.{index['column']}, {index['num_rows']} rows)"
    )


@scenarios_app.command("create")
def create(
    lake: str = _LAKE_OPTION,
    window: str = _WINDOW_OPTION,
    topics: list[str] | None = _TOPIC_OPTION,
    include_partial: bool = _INCLUDE_PARTIAL_OPTION,
) -> None:
    """Create deterministic scenario windows over ingested runs."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.scenarios import (
        ScenarioError,
        create_scenario_windows,
        parse_duration_ns,
    )

    try:
        opened = Lake.open(lake)
        window_ns = parse_duration_ns(window)
        report = create_scenario_windows(
            opened,
            window_ns=window_ns,
            topics=topics or [],
            include_partial=include_partial,
        )
    except (LakeError, ScenarioError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    partial_label = "included" if include_partial else "dropped"
    typer.echo(f"lake: {report.lake_uri}")
    typer.echo(f"window: {window} ({report.window_ns} ns)")
    typer.echo(f"topics: {report.topic_label}")
    typer.echo(f"partial final window: {partial_label}")
    typer.echo(f"runs: {report.runs_considered}")
    typer.echo(f"scenarios: {report.rows_added} created")
    if report.rows_replaced:
        typer.echo(f"replaced: {report.rows_replaced}")
    for run_id, count in report.windows_by_run.items():
        typer.echo(f"  {run_id}\t{count} windows")
    echo_emitted_lineage(opened, report.transform_id)


_STRICT_OPTION = typer.Option(
    False,
    "--strict/--no-strict",
    help=(
        "Fail instead of degrading to the demo provider when a requested real "
        "caption/embedding provider is unavailable (for example, its extra is not "
        "installed or its endpoint is not configured)."
    ),
)


@scenarios_app.command("enrich")
def enrich(
    lake: str = _LAKE_OPTION,
    provider: str = _PROVIDER_OPTION,
    caption_provider: str = _CAPTION_PROVIDER_OPTION,
    embed: bool = _EMBED_OPTION,
    dimension: int = _DIMENSION_OPTION,
    embedding_column: str = _EMBEDDING_COLUMN_OPTION,
    replace_embedding: bool = _REPLACE_EMBEDDING_OPTION,
    build_index: bool = _BUILD_INDEX_OPTION,
    auto_index: bool = _AUTO_INDEX_OPTION,
    index_type: str = _INDEX_TYPE_OPTION,
    metric: str = _INDEX_METRIC_OPTION,
    num_partitions: int | None = _INDEX_NUM_PARTITIONS_OPTION,
    num_sub_vectors: int | None = _INDEX_NUM_SUB_VECTORS_OPTION,
    num_bits: int | None = _INDEX_NUM_BITS_OPTION,
    m: int | None = _INDEX_HNSW_M_OPTION,
    ef_construction: int | None = _INDEX_EF_CONSTRUCTION_OPTION,
    build_fts_index: bool = _BUILD_FTS_INDEX_OPTION,
    strict: bool = _STRICT_OPTION,
) -> None:
    """Attach captions and embeddings to scenario rows.

    Semantic search depends on the embedding provider. demo and hashed-text are
    deterministic offline stand-ins; sentence-transformers and clip (the
    'embeddings' extra) produce model-backed vectors. When a requested real
    provider is unavailable, the command warns on stderr and falls back to demo;
    pass --strict to fail instead.
    """
    from lancedb_robotics.captions import resolve_caption_provider
    from lancedb_robotics.embeddings import resolve_embedding_provider
    from lancedb_robotics.enrich import EnrichmentError, enrich_scenarios
    from lancedb_robotics.indexing import IndexingError
    from lancedb_robotics.lake import Lake, LakeError

    # --strict maps to fallback=None: a missing real provider raises (caught below)
    # instead of degrading to demo, so a run never quietly writes fake captions/vectors.
    strict_kwargs = {"fallback": None} if strict else {}
    warnings: list[str] = []
    try:
        opened = Lake.open(lake)
        resolved_caption_provider, notice = resolve_caption_provider(
            caption_provider, **strict_kwargs
        )
        if notice:
            warnings.append(notice)
        if embed:
            embedding_provider, notice = resolve_embedding_provider(
                provider, dimension=dimension, **strict_kwargs
            )
            if notice:
                warnings.append(notice)
        else:
            embedding_provider = None
        # Build the spec from the shared index flags. A spec is passed (forcing a
        # build, which still skips below the training floor) when --index is set or
        # any index flag is tuned away from its default; otherwise the auto-index
        # path owns the build decision with default IVF_PQ params.
        spec = _index_spec_from_cli(
            index_type=index_type,
            metric=metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            num_bits=num_bits,
            m=m,
            ef_construction=ef_construction,
        )
        index_spec = (
            spec if (embed and (build_index or _index_params_overridden(spec))) else None
        )
        report = enrich_scenarios(
            opened,
            caption_provider=resolved_caption_provider,
            embedding_provider=embedding_provider,
            embedding_column=embedding_column,
            replace_embedding=replace_embedding,
            index=index_spec,
            auto_index=auto_index,
            fts_index=build_fts_index,
        )
    except (LakeError, EnrichmentError, IndexingError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)
    typer.echo(f"lake: {report.lake_uri}")
    typer.echo(f"caption provider: {report.caption_provider}")
    if report.embedding_provider:
        typer.echo(
            f"embedding provider: {report.embedding_provider} (dim {report.embedding_dimension})"
        )
        typer.echo(f"embedding column: {report.embedding_column}")
    else:
        typer.echo("embedding provider: none")
    typer.echo(f"scenarios enriched: {report.scenarios_enriched}")
    typer.echo(f"transform: {report.transform_id}")
    if report.index:
        _echo_index(report.index)
    if report.fts_index:
        _echo_fts_index(report.fts_index)


@scenarios_app.command("index")
def index(
    lake: str = _LAKE_OPTION,
    column: str = _INDEX_COLUMN_OPTION,
    metric: str = _INDEX_METRIC_OPTION,
    index_type: str = _INDEX_TYPE_OPTION,
    num_partitions: int | None = _INDEX_NUM_PARTITIONS_OPTION,
    num_sub_vectors: int | None = _INDEX_NUM_SUB_VECTORS_OPTION,
    num_bits: int | None = _INDEX_NUM_BITS_OPTION,
    m: int | None = _INDEX_HNSW_M_OPTION,
    ef_construction: int | None = _INDEX_EF_CONSTRUCTION_OPTION,
) -> None:
    """Build a persistent ANN vector index over a scenario embedding column."""
    from lancedb_robotics.indexing import (
        IndexingError,
        build_vector_index,
        record_index_transform,
    )
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        spec = _index_spec_from_cli(
            index_type=index_type,
            metric=metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            num_bits=num_bits,
            m=m,
            ef_construction=ef_construction,
        )
        result = build_vector_index(opened, table="scenarios", column=column, spec=spec)
        record_index_transform(opened, result)
    except (LakeError, IndexingError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"table: {result.table}")
    typer.echo(f"column: {result.column}")
    _echo_index(result.to_params())


@scenarios_app.command("index-fts")
def index_fts(
    lake: str = _LAKE_OPTION,
    column: str = _FTS_INDEX_COLUMN_OPTION,
) -> None:
    """Build or refresh the persistent FTS index over scenario summaries."""
    from lancedb_robotics.indexing import (
        IndexingError,
        build_fts_index,
        record_fts_index_transform,
    )
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        result = build_fts_index(opened, table="scenarios", column=column)
        record_fts_index_transform(opened, result)
    except (LakeError, IndexingError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"table: {result.table}")
    typer.echo(f"column: {result.column}")
    _echo_fts_index(result.to_params())
