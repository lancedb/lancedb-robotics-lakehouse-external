"""Caption and embedding enrichment for scenario windows.

The baseline search demo needs semantic handles (summary text and a similarity
vector) on scenario rows, but the substrate must not depend on a production
perception stack. This module defines two provider contracts and a pair of
deterministic, local demo implementations:

- :class:`CaptionProvider` turns a :class:`ScenarioContext` into summary text.
- :class:`EmbeddingProvider` turns the same context into a fixed-length vector.

:func:`enrich_scenarios` reads the windowed ``scenarios`` rows, asks the
providers for a summary and (optionally) an embedding, writes them back onto
the rows, and records one ``transform_runs`` lineage row tracing the source
windowing transform(s) to the enriched scenario rows.

Determinism is the contract: the demo providers derive everything from a
stable canonical descriptor of the scenario (no clocks, no ``hash()``
randomization, no model weights), so fixtures and tests are reproducible. Real
model-backed providers can replace the demo ones by satisfying the same
contracts.

Summary text lives in the canonical ``scenarios.summary`` column. Embedding
vectors are added as a separate ``embedding`` column via Lance ``add_columns``
at enrich time, because the vector dimension is owned by the chosen embedder
rather than the canonical schema contract.
"""

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pyarrow as pa

from lancedb_robotics.blob import PAYLOAD_BLOB_COLUMN, fetch_blobs
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA

DEFAULT_EMBEDDING_COLUMN = "embedding"
DEFAULT_EMBEDDING_DIMENSION = 16

# Upsert rows in bounded batches, mirroring the lineage graph writer: a single
# multi-megabyte fragment can overflow the Lance mini-block encoder and panic
# (BUG-02). Scenario counts are small, so this is effectively one batch.
_ENRICH_WRITE_BATCH_ROWS = 16_384

# Concurrent enrich writers are arbitrated by Lance's optimistic-concurrency
# commit: the loser of a race gets a *retryable* commit conflict. Enrich is
# idempotent, so we re-read the latest committed version and re-run a bounded
# number of times, converging instead of failing or corrupting (BUG-04). No
# external lock -- multi-writer / object-store safety is delegated to Lance's
# commit machinery, where it belongs.
_ENRICH_COMMIT_RETRIES = 8

# Sentinel so callers can pass ``embedding_provider=None`` to disable embeddings
# while the default (a demo embedder) stays distinct from that explicit choice.
_USE_DEMO_EMBEDDER = object()


class EnrichmentError(Exception):
    """Raised when scenario enrichment cannot proceed."""


@dataclass(frozen=True)
class CameraObservation:
    """Camera observation bytes available to content-derived caption providers."""

    observation_id: str
    topic: str
    payload_blob: bytes | None = None
    payload_json: str | None = None


@dataclass(frozen=True)
class ScenarioContext:
    """The deterministic input a provider sees for one scenario window.

    Built purely from a scenario row's structured fields so that captions and
    embeddings are a stable function of the window itself.
    """

    scenario_id: str
    run_id: str
    start_time_ns: int
    end_time_ns: int
    window_ns: int
    is_partial: bool
    topics: tuple[str, ...]
    observation_count: int
    # The freshly computed caption for this window, threaded in by
    # :func:`enrich_scenarios` so a content-based text embedder (backlog 0021)
    # can embed what the scenario *says*, in the same space its ``embed_text``
    # queries land. ``None`` when no caption is available yet. Intentionally
    # excluded from :meth:`descriptor` so the structural demo embedder's output
    # is unchanged (fixtures/snapshots stay green).
    summary: str | None = None
    # The semantically *dense* text actually fed to a content/model text embedder,
    # threaded in by :func:`enrich_scenarios`. The display/FTS ``summary`` carries
    # the structural caption (topic list, counts) so BM25 can match structural
    # queries, but feeding that ~90%-boilerplate string to a sentence/CLIP model
    # collapses every scenario to nearly the same vector (BUG-07). ``embed_text``
    # is description-forward and drops the boilerplate; embedders prefer it over
    # ``summary``. ``None`` when no dense text exists, so the embedder falls back to
    # the summary/topics (description-less corpora are unchanged). Excluded from
    # :meth:`descriptor` so the structural demo embedder's output is unchanged.
    embed_text: str | None = None
    # Optional camera observations for providers that caption from decoded image
    # payloads. Kept out of ``descriptor`` so deterministic demo embeddings stay
    # structural and existing fixtures remain stable.
    camera_observations: tuple[CameraObservation, ...] = ()

    @property
    def duration_ns(self) -> int:
        return self.end_time_ns - self.start_time_ns

    @classmethod
    def from_row(
        cls,
        row: dict,
        *,
        camera_observations: tuple[CameraObservation, ...] = (),
    ) -> "ScenarioContext":
        return cls(
            scenario_id=row["scenario_id"],
            run_id=row["run_id"],
            start_time_ns=row["start_time_ns"],
            end_time_ns=row["end_time_ns"],
            window_ns=row["window_ns"],
            is_partial=bool(row["is_partial"]),
            topics=tuple(row["topics"] or ()),
            observation_count=row["observation_count"],
            summary=row.get("summary"),
            camera_observations=tuple(camera_observations),
        )

    def descriptor(self) -> str:
        """Stable JSON descriptor used to seed deterministic providers."""
        return json.dumps(
            {
                "run_id": self.run_id,
                "start_time_ns": self.start_time_ns,
                "end_time_ns": self.end_time_ns,
                "window_ns": self.window_ns,
                "is_partial": self.is_partial,
                "topics": list(self.topics),
                "observation_count": self.observation_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


class CaptionProvider(ABC):
    """Contract for turning a scenario window into summary text."""

    name: str
    uses_camera_payloads: bool = False
    # Whether :meth:`caption` emits scene-*semantic* free text (a VLM description)
    # versus structural boilerplate (the demo template's topic list + counts). When
    # True, :func:`enrich_scenarios` folds the caption into the dense ``embed_text``
    # alongside the scene description; when False, only the description seeds the
    # embedding, so the topic boilerplate never reaches the vector space (BUG-07).
    caption_is_semantic: bool = False

    @abstractmethod
    def caption(self, ctx: ScenarioContext) -> str:
        """Return a human-readable summary for ``ctx``."""


class EmbeddingProvider(ABC):
    """Contract for turning content into a similarity vector.

    A provider implements whatever subset of the three spaces it supports
    (backlog 0021 / decision 0025 -- multiple pluggable embedding columns, each
    produced by its own provider, each independently indexed):

    - :meth:`embed` -- a scenario window (its caption + structure) -> vector.
    - :meth:`embed_text` -- a free-text query -> vector, in the **same space** as
      :meth:`embed` so query/scenario similarity is meaningful.
    - :meth:`embed_image` -- decoded image bytes (a camera ``observation``'s
      payload) -> vector, for real visual-similarity search.

    ``embed`` stays abstract (every scenario/text provider must answer it); an
    image-only provider satisfies the ABC by overriding ``embed`` to raise.
    ``embed_text``/``embed_image`` are concrete and default to "unsupported" so a
    provider only declares the spaces it actually serves.
    """

    name: str
    # Provider/model version, recorded alongside ``name`` in the enrichment
    # lineage so a snapshot is reproducible against the exact embedder used
    # (backlog 0021 AC5). The demo providers already version via their name
    # (``demo-hash-v1``); real model providers also set this from the weights.
    version: str = "1"
    dimension: int

    @abstractmethod
    def embed(self, ctx: ScenarioContext) -> list[float]:
        """Return a length-``dimension`` vector of floats for ``ctx``."""

    def embed_text(self, text: str) -> list[float]:
        """Return a length-``dimension`` query vector for free-text ``text``.

        Concrete (not abstract) so existing scenario-only providers keep
        working; search-capable providers override it. The vector must live in
        the same space as :meth:`embed` for similarity to be meaningful.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support text-query embedding")

    def embed_image(self, image: bytes) -> list[float]:
        """Return a length-``dimension`` vector for the decoded ``image`` bytes.

        Concrete (not abstract) so text-only providers keep working; image/CLIP
        providers override it. The vector must live in the same space the
        column's neighbors are compared in (per-column space, decision 0025).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support image embedding")

    def embed_batch(self, inputs: list, *, kind: str) -> list[list[float] | None]:
        """Embed a batch of already-prepared ``inputs`` in one call, keyed by ``kind``.

        The one batched entry point the embedding pipeline calls regardless of
        modality (backlog 0189), so the engine never sniffs provider method names.
        ``kind`` names what the engine prepared:

        - ``"pil"`` -- decoded PIL images -> :meth:`embed_pil_images`;
        - ``"image_bytes"`` -- raw encoded bytes -> :meth:`embed_images`;
        - ``"text"`` -- strings -> :meth:`embed_text` per row;
        - ``"image"`` -- raw bytes -> :meth:`embed_image` per row (no-batch fallback).

        The engine only requests ``"pil"``/``"image_bytes"`` from providers that
        expose those batched methods. ``None`` inputs and per-row failures map to
        ``None`` so results align back to row ids. A provider with a faster native
        batch path overrides this method once instead of the four-way dispatch.
        """
        if kind == "pil":
            return self.embed_pil_images(inputs)  # type: ignore[attr-defined]
        if kind == "image_bytes":
            return self.embed_images(inputs)  # type: ignore[attr-defined]
        out: list[list[float] | None] = []
        if kind == "text":
            for value in inputs:
                try:
                    out.append(self.embed_text(value) if value else None)
                except Exception:  # noqa: BLE001 - unembeddable row -> None, never fatal
                    out.append(None)
            return out
        for value in inputs:  # kind == "image": per-item bytes, the no-batch fallback
            try:
                out.append(self.embed_image(value) if value else None)
            except Exception:  # noqa: BLE001 - undecodable frame -> None, never fatal
                out.append(None)
        return out


class DemoCaptionProvider(CaptionProvider):
    """Deterministic, template-based caption provider for the baseline demo."""

    name = "demo-template-v1"

    def caption(self, ctx: ScenarioContext) -> str:
        topic_label = ", ".join(ctx.topics) if ctx.topics else "no topics"
        duration_ms = ctx.duration_ns / 1_000_000
        kind = "partial window" if ctx.is_partial else "full window"
        return (
            f"{ctx.observation_count} observations on {ctx.run_id} "
            f"across {topic_label} spanning {duration_ms:g} ms ({kind})"
        )


class DemoEmbeddingProvider(EmbeddingProvider):
    """Deterministic, local embedding provider for the baseline demo.

    Hashes the scenario descriptor into a fixed-length L2-normalized vector.
    Uses :mod:`hashlib` (not the salted built-in ``hash()``) so output is
    stable across processes and machines.
    """

    name = "demo-hash-v1"

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        if dimension <= 0:
            raise EnrichmentError("embedding dimension must be positive")
        self.dimension = dimension

    def embed(self, ctx: ScenarioContext) -> list[float]:
        return self._embed_seed(ctx.descriptor().encode())

    def embed_text(self, text: str) -> list[float]:
        # Query text is embedded by the same hash so query and scenario vectors
        # share a space. Demo vectors are structural, not semantic; a real model
        # provider plugs in here for meaningful text/scenario alignment.
        return self._embed_seed(text.encode())

    def _embed_seed(self, seed: bytes) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for offset in range(0, len(digest), 4):
                if len(values) >= self.dimension:
                    break
                word = int.from_bytes(digest[offset : offset + 4], "big")
                values.append(word / 0xFFFFFFFF * 2.0 - 1.0)  # map to [-1, 1]
            counter += 1
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


@dataclass(frozen=True)
class EnrichmentReport:
    """Summary of one scenario-enrichment transform."""

    lake_uri: str
    transform_id: str
    caption_provider: str
    scenarios_enriched: int
    source_transform_ids: tuple[str, ...]
    embedding_provider: str | None = None
    embedding_provider_version: str | None = None
    embedding_dimension: int | None = None
    embedding_column: str | None = None
    # Set when an ANN vector index was requested for the embedding column at
    # enrich time (backlog 0021). ``status`` is ``built`` or ``skipped``; the
    # full dict is also folded into the enrichment ``transform_runs`` params.
    index: dict | None = None
    # Set when the persistent FTS index over ``scenarios.summary`` was built or
    # refreshed at enrich time (backlog 0022).
    fts_index: dict | None = None


def _stable_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _enrichment_transform_params(lake: Lake, transform_id: str) -> dict | None:
    for row in lake.table("transform_runs").to_arrow().to_pylist():
        if row["transform_id"] != transform_id or row["kind"] != "enrichment":
            continue
        try:
            return json.loads(row["params"]) if row["params"] else {}
        except json.JSONDecodeError:
            return {}
    return None


def _ensure_embedding_column(
    table, column: str, dimension: int, *, replace: bool = False
) -> bool:
    """Ensure ``column`` exists as ``fixed_size_list<float>[dimension]``.

    Returns ``True`` when an existing column was dropped and recreated (so the
    caller knows its ANN index is gone and the rows are now null, to be
    re-embedded), ``False`` when the column was added fresh or already matched.

    ``replace=True`` drops and recreates the column from scratch -- the only way to
    switch to a *different* dimension in place, and also a clean rebuild for a
    same-dimension provider swap. Lance drops the column's vector index along with
    the column (no dangling index); the FTS index over ``summary`` is a different
    column and is untouched. Without ``replace``, a dimension change is a loud,
    signposted error rather than silent data loss (BUG-08).
    """
    vector_type = pa.list_(pa.float32(), dimension)
    if column in table.schema.names:
        existing = table.schema.field(column).type
        if existing == vector_type and not replace:
            return False
        if existing != vector_type and not replace:
            raise EnrichmentError(
                f"column {column!r} already exists as {existing}, "
                f"incompatible with embedding dimension {dimension}; pass a different "
                f"--embedding-column to write a second column, or --replace-embedding "
                f"to drop and rebuild it in place (also rebuilds the column's ANN index)"
            )
        # replace=True: drop the old column (Lance drops its vector index too) and
        # re-add at the requested dimension. The rows are now null on this column,
        # so the caller re-embeds every row into the fresh column.
        table.drop_columns([column])
        table.add_columns(pa.schema([pa.field(column, vector_type)]))
        return True
    table.add_columns(pa.schema([pa.field(column, vector_type)]))
    return False


# Scenario fan-in for the camera-payload fetch. Enrich streams ``rows_to_enrich``
# in chunks of this many scenarios so the image-blob fetch (image-stats captions /
# CLIP embedding) holds at most one batch's camera frames in memory at a time,
# instead of every frame in the corpus. The all-at-once fetch pinned ~45 GB on the
# 58-run didi lake (57k frames) and stalled ~30 min before any embedding ran.
_ENRICH_CAMERA_BATCH = 32


def _camera_observations_by_scenario(
    lake: Lake,
    rows: list[dict],
    *,
    include_payloads: bool,
) -> dict[str, tuple[CameraObservation, ...]]:
    """Batch-load camera observation metadata/payloads for scenario contexts."""
    wanted_ids: dict[str, list[str]] = {}
    all_ids: list[str] = []
    for row in rows:
        ids = [str(oid) for oid in (row.get("observation_ids") or [])]
        wanted_ids[row["scenario_id"]] = ids
        all_ids.extend(ids)
    if not all_ids:
        return {row["scenario_id"]: () for row in rows}

    # Build the membership set once. Constructing it inside the comprehension's
    # ``if`` rebuilds it for every observation row, making the scan below
    # O(rows x ids) -- it stalls for tens of minutes on real corpora (BUG-01).
    wanted = set(all_ids)

    observations = lake.table("observations")
    # Push the image-modality predicate into the Lance scan so non-camera
    # observations are never materialized, and project ``payload_json`` only when
    # a content-based caption provider will actually read it (the demo/hashed-text
    # default leaves it out, keeping the scan metadata-only instead of pulling
    # every decoded payload into memory -- ~1.6 GB RSS on a 350k-obs lake).
    columns = ["observation_id", "topic", "modality"]
    if include_payloads:
        columns.append("payload_json")
    meta = observations.to_lance().to_table(columns=columns, filter="modality = 'image'")
    image_rows = {
        row["observation_id"]: row
        for row in meta.to_pylist()
        if row["observation_id"] in wanted
    }
    blobs = (
        fetch_blobs(
            observations,
            PAYLOAD_BLOB_COLUMN,
            image_rows,
            id_column="observation_id",
        )
        if include_payloads
        else {}
    )

    by_scenario: dict[str, tuple[CameraObservation, ...]] = {}
    for scenario_id, ids in wanted_ids.items():
        frames: list[CameraObservation] = []
        for oid in ids:
            row = image_rows.get(oid)
            if row is None:
                continue
            frames.append(
                CameraObservation(
                    observation_id=oid,
                    topic=row["topic"],
                    payload_blob=blobs.get(oid),
                    payload_json=row.get("payload_json"),
                )
            )
        by_scenario[scenario_id] = tuple(frames)
    return by_scenario


# Run-metadata keys whose values are structural/operational bookkeeping (build
# profile, source library, integrity verdict, schema inventory, ...) rather than
# scene semantics. The descriptive-metadata fold skips them so the search index
# is not polluted; everything else on ``runs.metadata`` -- scene descriptions,
# location, operator notes, event labels -- is free-text worth searching.
_NON_DESCRIPTIVE_RUN_METADATA_KEYS = frozenset(
    {"profile", "library", "storage_identifier", "checksum", "adapter", "kind"}
)
_NON_DESCRIPTIVE_RUN_METADATA_PREFIXES = ("integrity.", "recording.", "schema:", "file:")


def _is_descriptive_run_metadata_key(key: str) -> bool:
    """True when a ``runs.metadata`` key carries searchable free-text."""
    return key not in _NON_DESCRIPTIVE_RUN_METADATA_KEYS and not key.startswith(
        _NON_DESCRIPTIVE_RUN_METADATA_PREFIXES
    )


def _run_descriptions(lake: Lake) -> dict[str, str]:
    """Map ``run_id`` -> descriptive free-text gathered from ``runs.metadata``.

    Rich, queryable text (nuScenes ``scene-info.description``/``location``,
    operator notes, event labels) is captured at ingest on ``runs.metadata``, but
    the structural caption providers never see it -- so full-text/vector/hybrid
    search over ``scenarios.summary`` matches nothing scene-level on the default
    offline path. :func:`enrich_scenarios` folds this text into every window's
    summary (independent of the caption provider) so even the dependency-free
    path is semantically searchable. Values are ordered by key and de-duplicated
    so the fold is deterministic and reproducible.
    """
    descriptions: dict[str, str] = {}
    for row in lake.table("runs").to_arrow().to_pylist():
        seen: set[str] = set()
        parts: list[str] = []
        for entry in sorted(row.get("metadata") or [], key=lambda kv: kv.get("key") or ""):
            key = entry.get("key") or ""
            value = (entry.get("value") or "").strip()
            if value and value not in seen and _is_descriptive_run_metadata_key(key):
                seen.add(value)
                parts.append(value)
        if parts:
            descriptions[row["run_id"]] = ", ".join(parts)
    return descriptions


def _fold_run_context(caption: str, description: str | None) -> str:
    """Prefix the provider caption with descriptive run context for search.

    Leads with the scene description so search hits are scannable (the rare
    behavior that matched a query is the first thing shown); falls back to the
    bare caption when a run carries no descriptive metadata, leaving fixtures
    without scene text unchanged.
    """
    if not description:
        return caption
    return f"{description} — {caption}"


def _build_embed_text(
    description: str | None, caption: str, *, caption_is_semantic: bool
) -> str | None:
    """Assemble the dense text fed to a text/CLIP embedder for vector search (BUG-07).

    The display/FTS ``summary`` deliberately keeps the structural caption (topic
    list, observation counts, duration) so BM25 can still match structural queries
    like "camera". But a sentence/CLIP model mean-pools tokens, so a string that is
    ~90% identical per-scene topic boilerplate collapses every scenario to nearly
    the same vector. The embedded text is therefore *description-forward*: the scene
    description (run metadata), plus the caption **only when the provider emits
    scene-semantic text** (e.g. a VLM caption), never the structural topic
    boilerplate. Returns ``None`` when no dense text exists, so the embedder falls
    back to the summary/topics exactly as before -- description-less corpora (and
    every existing fixture) are unchanged.
    """
    parts: list[str] = []
    if description:
        parts.append(description)
    if caption_is_semantic and caption:
        parts.append(caption)
    return " — ".join(parts) if parts else None


def _summary_missing_run_context(row: dict, run_descriptions: dict[str, str]) -> bool:
    """True when a row's stored summary predates the descriptive-metadata fold.

    Lets a re-run on an already-enriched lake re-fold the scene text into stale
    summaries without a full rebuild, instead of skipping them as up-to-date.
    """
    description = run_descriptions.get(row["run_id"])
    return bool(description) and description not in (row.get("summary") or "")


def _is_retryable_commit_conflict(exc: BaseException) -> bool:
    """True when ``exc`` is a Lance optimistic-concurrency commit conflict.

    Lance arbitrates concurrent writers at commit time and raises a *retryable*
    commit conflict for the writer that was preempted (e.g. "Retryable commit
    conflict for version N: This Merge transaction was preempted by concurrent
    transaction Merge"). Because enrich is idempotent, re-reading the latest
    version and re-running converges, so these are retried rather than surfaced.
    """
    return "commit conflict" in str(exc).lower()


def _upsert_rows(
    lake: Lake, table_name: str, key_column: str, rows: list[dict], schema: pa.Schema
) -> None:
    """Atomically upsert ``rows`` onto ``table_name`` keyed on ``key_column``.

    Uses ``merge_insert`` rather than delete-by-id + add (mirrors lineage's
    ``_replace_rows``). The old two-commit sequence left a window in which a crash
    or a racing writer could drop rows after the delete but before the add (and
    could double-insert under concurrency); merge_insert updates matched rows in a
    single commit, so a failure leaves the prior committed version intact (BUG-04).
    """
    if not rows:
        return
    table = lake.table(table_name)
    for start in range(0, len(rows), _ENRICH_WRITE_BATCH_ROWS):
        batch = rows[start : start + _ENRICH_WRITE_BATCH_ROWS]
        data = pa.Table.from_pylist(batch, schema=schema)
        (
            table.merge_insert(key_column)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(data)
        )


def _assert_fully_enriched(
    lake: Lake, embedding_column: str, *, require_embedding: bool
) -> None:
    """Fail loudly if any scenario summary (or embedding) is still null.

    A successful enrich must leave the lake fully populated, never half-written.
    This post-write check is the safety net behind the ``[BUG-04]`` partial-write
    gate: if a write silently dropped values, surface it as an error rather than
    returning a "completed" report over a nulled table.
    """
    columns = ["summary"]
    if require_embedding:
        columns.append(embedding_column)
    materialized = lake.table("scenarios").to_lance().to_table(columns=columns)
    summary_nulls = materialized.column("summary").null_count
    if summary_nulls:
        raise EnrichmentError(
            f"enrich left {summary_nulls} scenario summary value(s) null in {lake.uri}; "
            "refusing to report success over a partially written table"
        )
    if require_embedding:
        embedding_nulls = materialized.column(embedding_column).null_count
        if embedding_nulls:
            raise EnrichmentError(
                f"enrich left {embedding_nulls} scenario {embedding_column!r} value(s) null "
                f"in {lake.uri}; refusing to report success over a partially written table"
            )


def _enrich_once(
    lake: Lake,
    *,
    caption_provider: CaptionProvider,
    embedding_provider: EmbeddingProvider | None,
    embedding_column: str,
    replace_embedding: bool,
    index: "object | None",
    auto_index: bool,
    fts_index: bool,
    created_by: str,
) -> EnrichmentReport:
    """One enrichment attempt: read the latest version, compute, atomically upsert.

    :func:`enrich_scenarios` wraps this in a bounded retry loop, so a Lance commit
    conflict raised by a concurrent writer is retried against the freshly
    committed version rather than failing or corrupting (BUG-04). Every write here
    is an atomic, idempotent ``merge_insert``, so re-running converges.
    """
    scenarios_table = lake.table("scenarios")
    rows = sorted(scenarios_table.to_arrow().to_pylist(), key=lambda row: row["scenario_id"])
    if not rows:
        raise EnrichmentError(f"no scenarios to enrich in {lake.uri}; run `scenarios create` first")

    # When a replace drops+recreates the column, its ANN index is dropped with it.
    # Remember whether one existed so we can rebuild it at the new dimension below,
    # rather than silently leaving search to fall back to a full brute-force scan.
    restore_index = False
    if embedding_provider is not None:
        from lancedb_robotics.indexing import has_vector_index

        had_index = has_vector_index(scenarios_table, embedding_column)
        replaced = _ensure_embedding_column(
            scenarios_table,
            embedding_column,
            embedding_provider.dimension,
            replace=replace_embedding,
        )
        restore_index = replaced and had_index
        # Re-read so every row carries the (possibly new) embedding key.
        rows = sorted(scenarios_table.to_arrow().to_pylist(), key=lambda row: row["scenario_id"])

    schema = scenarios_table.schema
    source_transform_ids = tuple(
        sorted({row["transform_id"] for row in rows if row["transform_id"]})
    )

    # transform_key (and thus the transform_id digest) deliberately omits the
    # provider version and index params -- the provider *name* already carries
    # its version (e.g. ``demo-hash-v1``), and the index is a downstream property
    # of the same enrichment. Keeping the key stable keeps the demo lineage id
    # (and the showcase snapshot) unchanged.
    transform_key = {
        "kind": "enrichment",
        "caption_provider": caption_provider.name,
        "embedding_provider": embedding_provider.name if embedding_provider else None,
        "embedding_dimension": embedding_provider.dimension if embedding_provider else None,
        "embedding_column": embedding_column if embedding_provider else None,
    }
    transform_id = f"tfm-enrich-{_stable_digest(transform_key)}"

    existing_transform = _enrichment_transform_params(lake, transform_id)
    previously_enriched = (
        set(existing_transform.get("scenario_ids", [])) if existing_transform else set()
    )

    # Descriptive run metadata (scene description/location, operator notes, event
    # labels) is folded into each window's summary so the search index carries
    # scene semantics, not just structural tokens.
    run_descriptions = _run_descriptions(lake)

    rows_to_enrich = [
        row
        for row in rows
        if (
            row["scenario_id"] not in previously_enriched
            or not row.get("summary")
            or (embedding_provider is not None and row.get(embedding_column) is None)
            or _summary_missing_run_context(row, run_descriptions)
        )
    ]
    # Stream camera payloads in scenario batches so a corpus with tens of thousands
    # of camera frames never materializes every frame's bytes at once. Only the blob
    # fetch (image-stats captions / CLIP embedding) is memory-heavy; bounding it to a
    # batch keeps peak RSS flat and interleaves the S3 blob reads with caption/embed
    # work instead of one up-front stall. The in-place ``row`` mutations accumulate
    # across batches and are upserted once below, so batching changes only the read
    # cadence, not the write.
    include_camera_payloads = bool(getattr(caption_provider, "uses_camera_payloads", False))
    caption_is_semantic = bool(getattr(caption_provider, "caption_is_semantic", False))
    for start in range(0, len(rows_to_enrich), _ENRICH_CAMERA_BATCH):
        batch = rows_to_enrich[start : start + _ENRICH_CAMERA_BATCH]
        camera_by_scenario = _camera_observations_by_scenario(
            lake, batch, include_payloads=include_camera_payloads
        )
        for row in batch:
            ctx = ScenarioContext.from_row(
                row, camera_observations=camera_by_scenario.get(row["scenario_id"], ())
            )
            caption = caption_provider.caption(ctx)
            description = run_descriptions.get(row["run_id"])
            summary = _fold_run_context(caption, description)
            row["summary"] = summary
            if embedding_provider is not None:
                # Embed description-forward dense text, not the structural boilerplate
                # carried in ``summary`` for display/FTS (BUG-07). The demo embedder
                # ignores both (it hashes ``descriptor()``), so its vectors are unchanged.
                embed_text = _build_embed_text(
                    description, caption, caption_is_semantic=caption_is_semantic
                )
                row[embedding_column] = embedding_provider.embed(
                    replace(ctx, summary=summary, embed_text=embed_text)
                )
        # Release this batch's camera bytes before the next batch's fetch.
        camera_by_scenario = None

    # Atomic in-place upsert (single commit per batch) rather than delete-by-id +
    # add: a crash or contention leaves the prior committed version intact instead
    # of a half-deleted, nulled table, and a concurrent writer's preemption surfaces
    # as a retryable commit conflict (handled by enrich_scenarios) (BUG-04).
    _upsert_rows(lake, "scenarios", "scenario_id", rows_to_enrich, schema)

    # Fail loudly if the write left any summary/embedding null, rather than
    # returning a "completed" report over a partially written table.
    _assert_fully_enriched(
        lake, embedding_column, require_embedding=embedding_provider is not None
    )

    # Build the persistent ANN index over the freshly written column when:
    #   - one was explicitly requested (``index=IndexSpec()``), or
    #   - a replace dropped an index that existed before (preserve indexed status at
    #     the new dimension after a --replace-embedding migration), or
    #   - ``auto_index`` and the column now has enough rows to train IVF_PQ.
    # The third case makes the ANN path engage *by default once it can* (backlog
    # 0183 / round-4 §5): below ``MIN_INDEX_ROWS`` ``build_vector_index`` correctly
    # skips and search stays exact brute-force, so on small lakes this is a no-op;
    # at scale it stops silently leaving search on a full scan. Imported lazily so
    # enrich has no import-time dependency on indexing.
    index_params: dict | None = None
    if embedding_provider is not None:
        from lancedb_robotics.indexing import MIN_INDEX_ROWS, IndexSpec, build_vector_index

        want_index = (
            index is not None
            or restore_index
            or (auto_index and len(rows) >= MIN_INDEX_ROWS)
        )
        if want_index:
            spec = index if isinstance(index, IndexSpec) else IndexSpec()
            index_params = build_vector_index(
                lake, table="scenarios", column=embedding_column, spec=spec
            ).to_params()

    fts_index_params: dict | None = None
    if fts_index:
        from lancedb_robotics.indexing import build_fts_index

        fts_index_params = build_fts_index(lake, table="scenarios", column="summary").to_params()

    scenario_ids = [row["scenario_id"] for row in rows]
    now = datetime.now(UTC)
    # Upsert the enrichment lineage row by transform_id (atomic, idempotent)
    # rather than delete + add, so concurrent writers can't double-insert it.
    transform_row = {
        "transform_id": transform_id,
        "kind": "enrichment",
        "input_uris": [],
        "input_table_versions": [],
        "output_tables": ["scenarios"],
        "params": json.dumps(
            {
                **transform_key,
                "embedding_provider_version": (
                    embedding_provider.version if embedding_provider else None
                ),
                "index": index_params,
                "fts_index": fts_index_params,
                "source_transform_ids": list(source_transform_ids),
                "scenario_ids": scenario_ids,
                "scenarios_enriched": len(rows_to_enrich),
            },
            sort_keys=True,
        ),
        "status": "completed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    _upsert_rows(
        lake,
        "transform_runs",
        "transform_id",
        [transform_row],
        TRANSFORM_RUNS_SCHEMA,
    )
    # Emit lineage inline (backlog 0098): the enrichment slice without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)

    return EnrichmentReport(
        lake_uri=lake.uri,
        transform_id=transform_id,
        caption_provider=caption_provider.name,
        scenarios_enriched=len(rows_to_enrich),
        source_transform_ids=source_transform_ids,
        embedding_provider=embedding_provider.name if embedding_provider else None,
        embedding_provider_version=embedding_provider.version if embedding_provider else None,
        embedding_dimension=embedding_provider.dimension if embedding_provider else None,
        embedding_column=embedding_column if embedding_provider else None,
        index=index_params,
        fts_index=fts_index_params,
    )


def enrich_scenarios(
    lake: Lake,
    *,
    caption_provider: CaptionProvider | None = None,
    embedding_provider: EmbeddingProvider | None = _USE_DEMO_EMBEDDER,
    embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
    replace_embedding: bool = False,
    index: "object | None" = None,
    auto_index: bool = True,
    fts_index: bool = True,
    created_by: str = "lancedb-robotics",
) -> EnrichmentReport:
    """Write summary text and optional embedding vectors onto scenario rows.

    By default both a demo caption provider and a demo embedding provider run.
    Pass ``embedding_provider=None`` to enrich captions only (no vector column
    is added, and any existing embeddings are preserved). The write is
    idempotent: re-running with the same providers replaces the enrichment
    transform row and rewrites the same scenario rows in place.

    Descriptive run metadata captured at ingest (nuScenes
    ``scene-info.description``/``location``, operator notes, event labels on
    ``runs.metadata``) is folded into each window's summary -- independent of the
    caption provider -- so full-text/vector/hybrid search over
    ``scenarios.summary`` matches scene semantics, not just structural tokens, on
    the default offline path. A re-run re-folds the scene text into summaries that
    predate the fold, so an already-enriched lake becomes searchable without a
    full rebuild.

    The folded summary is threaded into the embedding provider (via
    ``ScenarioContext.summary``) so a content-based text embedder embeds what the
    scenario *says*, in the same space its query vectors land. The structural
    demo embedder ignores it, so its output -- and the fixtures/snapshots built
    on it -- are unchanged.

    ``embedding_column`` names the vector column to write (default ``embedding``).
    Pass a distinct name to keep multiple embedding spaces side by side (e.g. a
    16-dim demo column and a 384-dim model column), each with its own lineage row.
    ``replace_embedding`` drops and rebuilds that column from scratch -- the only
    way to switch to a *different dimension* in place (without it, a dimension
    change is a loud, signposted error). A replace re-embeds every row and, if the
    column had an ANN index, rebuilds it at the new dimension; the FTS index over
    ``summary`` is unaffected (BUG-08).

    Pass ``index`` (a ``lancedb_robotics.indexing.IndexSpec``) to build a
    persistent ANN index over ``embedding_column`` right after the embeddings are
    written (backlog 0021). By default enrichment also builds/refreshes the
    persistent FTS index over ``summary`` (backlog 0022) so text/hybrid search
    can reuse it without a hot-path rebuild. Resolved index params are recorded
    in the enrichment ``transform_runs`` row for reproducibility.
    """
    caption_provider = caption_provider or DemoCaptionProvider()
    if embedding_provider is _USE_DEMO_EMBEDDER:
        embedding_provider = DemoEmbeddingProvider()

    # No external lock: Lance arbitrates concurrent writers at commit time. The
    # writer that loses a race gets a retryable commit conflict; because each
    # attempt re-reads the latest committed version and every write is an
    # idempotent merge_insert, re-running converges on the fully enriched lake
    # instead of failing or nulling it (BUG-04).
    last_conflict: Exception | None = None
    for _attempt in range(_ENRICH_COMMIT_RETRIES + 1):
        try:
            return _enrich_once(
                lake,
                caption_provider=caption_provider,
                embedding_provider=embedding_provider,
                embedding_column=embedding_column,
                replace_embedding=replace_embedding,
                index=index,
                auto_index=auto_index,
                fts_index=fts_index,
                created_by=created_by,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is a commit conflict
            if not _is_retryable_commit_conflict(exc):
                raise
            last_conflict = exc
    raise EnrichmentError(
        f"enrich of {lake.uri} lost {_ENRICH_COMMIT_RETRIES} consecutive commit races to "
        "concurrent writers; the lake is under sustained write contention -- retry later"
    ) from last_conflict
