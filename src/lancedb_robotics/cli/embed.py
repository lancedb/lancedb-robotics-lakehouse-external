"""`lancedb-robotics embed` subcommands (observation-level embedding columns, 0187).

Closes finding F3 of the EC2/S3 vertical slice: creating per-frame (pixel) image
embeddings previously required the throwaway spike driver
``robotics-spike/scripts/clip_image_embed.py``. ``embed observations`` wraps the
streaming/resumable :func:`~lancedb_robotics.embeddings.embed_observations` seam
-- blob-safe additive column write, bounded memory, checkpoint resume -- and
builds the column's ANN index with a scale-aware type default (backlog 0186
policy: ``IVF_FLAT`` below the PQ-meaningful floor, so small corpora never
inherit the silent-0-recall ``IVF_PQ`` behavior).
"""

import typer

embed_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_PROVIDER_OPTION = typer.Option(
    "hashed-image",
    "--provider",
    help=(
        "Embedding provider for the frame payloads: hashed-image (default, "
        "dependency-free byte pooling -- near-duplicate frames land near each "
        "other) or clip (real text+image space, needs the 'embeddings' extra; "
        "required for text->image `search image`). A model-backed provider whose "
        "extra is missing degrades to hashed-image with a notice unless --strict."
    ),
)
_COLUMN_OPTION = typer.Option(
    "emb_image",
    "--column",
    help=(
        "Observation column to write vectors into (default 'emb_image'). Name it "
        "per model (e.g. embedding_clip_image) to keep multiple embedding spaces "
        "side by side; re-embedding with a new model is a new column, not a rewrite."
    ),
)
_MODALITY_OPTION = typer.Option(
    None,
    "--modality",
    help="Observation modality to embed; repeat for multiple. Default: image. "
    "Rows outside the selected modalities hold NULL in the vector column.",
)
_DIMENSION_OPTION = typer.Option(
    None,
    "--dimension",
    help="Embedding dimension (honored by the dependency-free hashed providers; "
    "model-backed providers own their dimension and ignore this).",
)
_IMAGE_BATCH_SIZE_OPTION = typer.Option(
    256,
    "--image-batch-size",
    help="Rows decoded + embedded per batch; peak memory scales with this, not the corpus.",
)
_CHECKPOINT_OPTION = typer.Option(
    None,
    "--checkpoint-file",
    help=(
        "Checkpoint path for resumable runs: re-running an *interrupted* run with "
        "the same file skips already-embedded batches instead of recomputing "
        "(a completed run re-executes cleanly). Auto-derived under the system "
        "temp dir when unset; the command echoes the path."
    ),
)
_BUILD_INDEX_OPTION = typer.Option(
    True,
    "--index/--no-index",
    help="Build a persistent ANN index over the new column after embedding "
    "(skipped below the training floor). --no-index leaves search brute-force.",
)
_INDEX_TYPE_OPTION = typer.Option(
    None,
    "--index-type",
    help=(
        "Vector index type: ivf_pq, ivf_flat, ivf_sq, ivf_hnsw_flat, ivf_hnsw_sq, "
        "ivf_hnsw_pq. Default is scale-aware: ivf_flat below the PQ-meaningful "
        "floor (~65k vectors -- PQ there only costs recall and needs a large "
        "--refine-factor at query time), ivf_pq at scale. An explicit value "
        "always wins."
    ),
)
_INDEX_METRIC_OPTION = typer.Option(
    "cosine", "--metric", help="Distance metric for the ANN index (cosine|l2|dot)."
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
_STRICT_OPTION = typer.Option(
    False,
    "--strict/--no-strict",
    help="Fail instead of degrading to the hashed-image stand-in when the requested "
    "model-backed --provider is unavailable (a stand-in column cannot serve "
    "text->image search).",
)


@embed_app.command("observations")
def observations(
    lake: str = _LAKE_OPTION,
    provider: str = _PROVIDER_OPTION,
    column: str = _COLUMN_OPTION,
    modality: list[str] | None = _MODALITY_OPTION,
    dimension: int | None = _DIMENSION_OPTION,
    image_batch_size: int = _IMAGE_BATCH_SIZE_OPTION,
    checkpoint_file: str | None = _CHECKPOINT_OPTION,
    build_index: bool = _BUILD_INDEX_OPTION,
    index_type: str | None = _INDEX_TYPE_OPTION,
    metric: str = _INDEX_METRIC_OPTION,
    num_partitions: int | None = _INDEX_NUM_PARTITIONS_OPTION,
    num_sub_vectors: int | None = _INDEX_NUM_SUB_VECTORS_OPTION,
    num_bits: int | None = _INDEX_NUM_BITS_OPTION,
    m: int | None = _INDEX_HNSW_M_OPTION,
    ef_construction: int | None = _INDEX_EF_CONSTRUCTION_OPTION,
    strict: bool = _STRICT_OPTION,
) -> None:
    """Embed camera-frame payloads into a per-observation vector column."""
    import os
    import tempfile

    from lancedb_robotics.cli.scenarios import _echo_index, _index_spec_from_cli
    from lancedb_robotics.embeddings import embed_observations, resolve_embedding_provider
    from lancedb_robotics.enrich import EnrichmentError
    from lancedb_robotics.indexing import IndexingError
    from lancedb_robotics.lake import Lake, LakeError

    modalities = tuple(modality) if modality else ("image",)
    try:
        opened = Lake.open(lake)
        # --strict maps to fallback=None (fail loud); otherwise a missing model
        # extra degrades to the image-capable hashed-image stand-in with a notice
        # (never the text-only demo provider).
        embedding_provider, notice = resolve_embedding_provider(
            provider,
            dimension=dimension,
            fallback=None if strict else "hashed-image",
        )
        if notice:
            typer.echo(f"notice: {notice}", err=True)

        index_spec = None
        if build_index:
            # index_type=None means the scale-aware default (0186): the library
            # resolves it after embedding, over the count of vectors actually
            # written, and records the choice in the result echoed below.
            index_spec = _index_spec_from_cli(
                index_type=index_type,
                metric=metric,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                num_bits=num_bits,
                m=m,
                ef_construction=ef_construction,
            )

        report = embed_observations(
            opened,
            embedding_provider,
            column=column,
            modalities=modalities,
            index=index_spec,
            image_batch_size=image_batch_size,
            checkpoint_file=checkpoint_file,
        )
    except (LakeError, EnrichmentError, IndexingError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"lake: {report.lake_uri}")
    typer.echo(
        f"provider: {report.embedding_provider} (dim {report.embedding_dimension})"
    )
    typer.echo(f"column: {report.column}")
    typer.echo(f"modalities: {', '.join(report.modalities)}")
    typer.echo(f"observations embedded: {report.observations_embedded}")
    typer.echo(f"observations skipped: {report.observations_skipped}")
    typer.echo(f"transform: {report.transform_id}")
    resumable = checkpoint_file or os.path.join(
        tempfile.gettempdir(), f"lr-{report.transform_id}.ckpt"
    )
    typer.echo(f"checkpoint: {resumable}")
    if report.index:
        _echo_index(report.index)
    else:
        typer.echo("vector index: none (--no-index)")
