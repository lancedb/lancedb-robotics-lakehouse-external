"""Published, version-pinned LeRobot views over a lake (backlogs 0490/0491).

A *published view* is the deliberate act by which lakehouse data is exposed to
LeRobot clients: one row in the canonical ``lerobot_views`` table holding the
view's full definition (mapping, quality policy, episode selection) plus the
Lance version of every canonical table at publish time, and one row per derived
``meta/`` file (``info.json``, chunked episode index parquet, ``stats.json``,
the companion source manifest) in ``lerobot_view_files``. A client pointed at
``root="s3://acme/lake.lance"`` materializes those files through the same Lance
connection it already has -- no hand-distributed side channel (prior art:
``lancedb/lerobot-lancedb``'s ``meta.lance`` transport).

Version pinning happens at the :meth:`Lake.table` seam: :class:`PinnedLake`
shares the parent connection and checks out each canonical table at the version
recorded at publish time, so *every* facade read path (``alignment_jobs``,
``aligned_ticks``, ``episodes``/``scenarios``/``runs``, ``videos``/
``video_encodings``, ``observations``, blob fetches) reads the exact state the
manifest was derived from. Two opens of the same published view return
byte-identical frames no matter how far the live lake has advanced; a pruned or
missing pinned version raises :class:`StaleViewVersionError` -- reads never
silently fall back to latest (SKILLS.md: typed error, never silent drift).

Write discipline (SKILLS.md BUG-04 rule): file rows first, header row last, all
writes bounded ``merge_insert`` upserts keyed by content-addressed ids, with a
bounded retry on retryable commit conflicts -- concurrent identical publishes
converge, a crash mid-publish leaves no discoverable header, and re-publishing
an unchanged definition over an unchanged lake is a no-op upsert.

Concurrency note: Lance arbitrates truly simultaneous same-key ``merge_insert``
inserts as non-conflicting appends, so N concurrent publishers of the same view
can land N byte-identical rows per key. Because every row is content-addressed
(the key embeds the view digest and the content is deterministic), duplicates
are *benign*: every read path deduplicates by key, and the publish
postcondition counts distinct keys — semantic convergence, verified. Physical
dedup is ``view_lifecycle.compact_view_catalog`` (backlog 0507), wired into
``lake maintain``.

Resolve/listing scale (backlog 0507): ``get_view(repo_id=...)`` — the default
client resolve behind every ``LeRobotDataset(root=<lake>)`` open — resolves
through the ``lerobot_view_latest`` pointer table (one point read per open,
maintained newest-wins by publish), falling back to a backend-ordered top-1
read and only then to the loudly-guarded bounded scan. ``list_view_pages``
pages the catalog with a stable descending ``(created_at, view_id)`` keyset
cursor; the unpaged ``list_views`` keeps its loud over-read guard.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake, LakeError
from lancedb_robotics.schemas import (
    LEROBOT_VIEW_FILES_SCHEMA,
    LEROBOT_VIEW_LATEST_SCHEMA,
    LEROBOT_VIEWS_SCHEMA,
)

from ._reader_core import SOURCE_MANIFEST_FILENAME
from .manifest import derive_manifest, write_manifest_files
from .mapping import CanonicalVectorMapping
from .reader import LiveLeRobotFacade

VIEWS_TABLE = "lerobot_views"
VIEW_FILES_TABLE = "lerobot_view_files"
VIEW_LATEST_TABLE = "lerobot_view_latest"
SOURCE_MANIFEST_PATH = f"meta/{SOURCE_MANIFEST_FILENAME}"

VIEW_CACHE_ENV = "LANCEDB_ROBOTICS_VIEW_CACHE"

_MERGE_INSERT_ATTEMPTS = 5
# Bounded merge_insert commits (SKILLS.md BUG-02: never one oversized one-shot
# write): a batch flushes at either bound, so commit size is capped in *bytes*,
# not just file count.
_FILE_WRITE_BATCH = 16
_FILE_WRITE_BATCH_BYTES = 32 * 1024 * 1024
# Caller-provided explicit episode selections are inlined into the header's
# definition (and its digest) only up to this bound (0146 discipline); larger
# selections are digested + counted, with the full resolved list living in the
# source-manifest file row.
_DEFINITION_EPISODE_IDS_INLINE_LIMIT = 1024
# list/get read guard: a catalog scan larger than this raises loudly instead of
# silently degrading into an unbounded read (views accrue one row per deliberate
# publish, so hitting this means the caller should page with list_view_pages).
_MAX_CATALOG_SCAN_ROWS = 10_000
# Bounds for the keyset-paged listing (backlog 0507).
_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 1_000
_PAGE_SCAN_BATCH = 1_024

# Bounded columns projected by listings: deliberately excludes definition_json
# and stats_sampling_json (unbounded-ish strings a catalog listing never
# needs — SKILLS.md: project only what you need). get_view point-reads the full
# row for one exact view_id.
_LIST_COLUMNS = [
    "view_id",
    "repo_id",
    "alignment_id",
    "fps",
    "robot_type",
    "total_frames",
    "total_episodes",
    "file_count",
    "files_bytes",
    "created_by",
    "created_at",
]
_FULL_COLUMNS = [*_LIST_COLUMNS, "definition_json", "table_versions", "stats_sampling_json"]


class ViewError(Exception):
    """Base error for published-view operations."""


class ViewPublishError(ViewError):
    """Raised when a view cannot be published or fails post-write validation."""


class ViewNotFoundError(ViewError):
    """Raised when no published view matches the requested repo_id/view_id."""


class StaleViewVersionError(ViewError):
    """A pinned table version can no longer be served -- never silently skew.

    Raised when a table pinned by a published view is missing from the pin map
    (created after publish) or its pinned Lance version cannot be checked out
    (pruned by retention/compaction, or the backend does not support version
    checkout). The remedy is to re-publish the view against the current lake.
    """

    def __init__(self, table: str, version: int | None, reason: str) -> None:
        version_text = "unpinned" if version is None else f"version {version}"
        super().__init__(
            f"published view cannot serve table {table!r} at {version_text}: {reason}. "
            "Re-publish the view (`lake` has advanced past what this view pinned) or "
            "open the live facade directly if you do not need reproducibility."
        )
        self.table = table
        self.version = version
        self.reason = reason


class PinnedLake(Lake):
    """A :class:`Lake` whose ``table()`` serves version-pinned handles.

    Shares the parent lake's connection; every canonical-table open is checked
    out at the version recorded in ``table_versions``. Because each
    ``Lake.table`` call returns a fresh handle, the checkout is scoped to that
    handle and never mutates other readers of the same connection.
    """

    def __init__(self, base: Lake, table_versions: Mapping[str, int]) -> None:
        super().__init__(base._db, base.uri, connection_spec=base.connection_spec)
        self._pinned_versions = {name: int(version) for name, version in table_versions.items()}

    @property
    def pinned_versions(self) -> dict[str, int]:
        return dict(self._pinned_versions)

    def table(self, name: str):
        table = super().table(name)
        version = self._pinned_versions.get(name)
        if version is None:
            raise StaleViewVersionError(
                name,
                None,
                "this view's manifest records no pinned version for it (the table "
                "did not exist when the view was published)",
            )
        try:
            table.checkout(version)
        except Exception as exc:  # noqa: BLE001 - surface every checkout failure typed.
            raise StaleViewVersionError(name, version, f"checkout failed: {exc}") from exc
        # Marker read by blob._to_dataset: the namespace-direct hydration route
        # opens its own lance dataset (it never calls to_lance() on this
        # handle), so the pin must travel alongside the handle or that route
        # would silently read the live table version.
        table._lancedb_robotics_pinned_version = version
        return table


# The view catalog itself is never pinned: the facade never reads it, and
# publishing writes to it -- including it would make every publish change the
# very digest that is supposed to be idempotent over an unchanged lake.
_UNPINNED_TABLES = frozenset({VIEWS_TABLE, VIEW_FILES_TABLE, VIEW_LATEST_TABLE})


def capture_table_versions(lake: Lake) -> list[dict[str, Any]]:
    """Current Lance version of every canonical data table present in ``lake``.

    Same row shape as ``dataset_snapshots``' pins (``training.py``'s
    ``_current_table_versions``): ``{"table", "version", "tag"}``. One
    ``.version`` metadata read per table -- no row data is touched. The view
    catalog tables themselves are excluded (see ``_UNPINNED_TABLES``).
    """
    return [
        {"table": name, "version": int(lake.table(name).version), "tag": ""}
        for name in lake.table_names()
        if name not in _UNPINNED_TABLES
    ]


def _versions_map(table_versions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {str(item["table"]): int(item["version"]) for item in table_versions}


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _view_digest(definition: Mapping[str, Any], table_versions: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256(
        _canonical_json({"definition": definition, "table_versions": list(table_versions)}).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"lrv-{digest[:16]}"


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _is_retryable_commit_conflict(exc: BaseException) -> bool:
    """Match Lance's retryable optimistic-concurrency conflict, and only that.

    Mirrors ``enrich._is_retryable_commit_conflict`` (the BUG-04 reference
    implementation) exactly: a broader match would retry — and then mask —
    genuinely non-retryable errors that merely mention concurrency.
    """
    return "commit conflict" in str(exc).lower()


def _merge_insert_with_retry(
    table: Any, key_column: str, data: pa.Table, *, update_condition: str | None = None
) -> None:
    """Single-commit upsert (BUG-04 shape) with bounded commit-conflict retry.

    ``update_condition`` (a ``target.``/``source.``-qualified SQL expression)
    makes the matched-row update conditional -- the pointer table's newest-wins
    CAS -- while inserts of missing keys always land.
    """
    if data.num_rows == 0:
        return
    last_error: BaseException | None = None
    for _ in range(_MERGE_INSERT_ATTEMPTS):
        builder = table.merge_insert(key_column).when_matched_update_all(
            where=update_condition
        )
        try:
            builder.when_not_matched_insert_all().execute(data)
            return
        except Exception as exc:  # noqa: BLE001 - retry only the retryable conflict.
            if not _is_retryable_commit_conflict(exc):
                raise
            last_error = exc
    raise ViewPublishError(
        f"merge_insert on {key_column!r} kept hitting commit conflicts "
        f"after {_MERGE_INSERT_ATTEMPTS} attempts: {last_error}"
    )


@dataclass(frozen=True)
class PublishedView:
    """Result of one :func:`publish_view` call."""

    view_id: str
    repo_id: str
    alignment_id: str
    total_frames: int
    total_episodes: int
    file_count: int
    files_bytes: int
    table_versions: tuple[dict[str, Any], ...]
    created_at: datetime
    #: False when the ``lerobot_view_latest`` pointer upsert could not land.
    #: The publish itself is complete; any pre-existing pointer row is
    #: best-effort removed so repo_id resolves fall back to a real catalog
    #: read, and ``lake maintain`` (compact_view_catalog) reconciles a pointer
    #: a crash or failed removal left stale.
    latest_pointer_updated: bool = True


def publish_view(
    lake: Lake,
    *,
    repo_id: str,
    fps: int,
    mapping: CanonicalVectorMapping,
    alignment: str | None = None,
    alignment_id: str | None = None,
    name: str | None = None,
    episode_ids: Sequence[str] | None = None,
    statuses: Sequence[str] | str | None = None,
    min_confidence: float | None = None,
    require_streams: bool | Sequence[str] = True,
    robot_type: str | None = None,
    created_by: str | None = None,
    include_stats: bool = True,
    stats_options: Mapping[str, Any] | None = None,
) -> PublishedView:
    """Publish one version-pinned LeRobot view of this lake.

    Captures every canonical table's Lance version *first*, then builds the
    facade over a :class:`PinnedLake` at exactly those versions -- so the
    derived manifest (episode boundaries, frame counts, normalization
    statistics) can never drift from what a later pinned open serves, even if
    a concurrent writer advances the lake mid-publish.

    Idempotent by construction: ``view_id`` is a content digest over the
    definition plus the pinned versions, and both catalog writes are
    ``merge_insert`` upserts, so republishing an unchanged definition over an
    unchanged lake converges on the same row; after the lake advances, the same
    definition publishes a *new* view id and previously published views stay
    reproducible (backlog 0491 acceptance).
    """
    table_versions = capture_table_versions(lake)
    pinned = PinnedLake(lake, _versions_map(table_versions))
    facade = LiveLeRobotFacade(
        pinned,
        alignment=alignment,
        alignment_id=alignment_id,
        name=name,
        mapping=mapping,
        episode_ids=episode_ids,
        statuses=statuses,
        min_confidence=min_confidence,
        require_streams=require_streams,
    )
    resolved_alignment_id = str(facade._dataset.alignment_id)
    normalized_statuses = (
        list(statuses)
        if isinstance(statuses, Sequence) and not isinstance(statuses, str)
        else statuses
    )
    # The header definition records *caller intent*, bounded: an explicit
    # episode selection is inlined only up to a small limit and digested+counted
    # beyond it; the resolved list (bounded separately) lives in the
    # source-manifest file row, never in a header cell a listing projects.
    if episode_ids is None:
        episode_selection: dict[str, Any] = {"kind": "all"}
    else:
        selected = [str(episode_id) for episode_id in episode_ids]
        from .manifest import EPISODE_IDS_INLINE_LIMIT

        if len(selected) > EPISODE_IDS_INLINE_LIMIT:
            raise ViewPublishError(
                f"explicit episode_ids selection has {len(selected)} entries, over the "
                f"{EPISODE_IDS_INLINE_LIMIT} bound; publish the whole alignment (episode_ids="
                "None) with a quality policy instead of an id list this large"
            )
        episode_selection = {
            "kind": "explicit",
            "count": len(selected),
            "ids_sha256": hashlib.sha256(_canonical_json(selected).encode("utf-8")).hexdigest(),
        }
        if len(selected) <= _DEFINITION_EPISODE_IDS_INLINE_LIMIT:
            episode_selection["ids"] = selected
    definition = {
        "repo_id": repo_id,
        "alignment_id": resolved_alignment_id,
        "mapping": {
            "state_streams": list(mapping.state_streams),
            "action_streams": list(mapping.action_streams),
            "camera_streams": list(mapping.camera_streams),
        },
        "statuses": normalized_statuses,
        "min_confidence": min_confidence,
        "require_streams": (
            list(require_streams)
            if isinstance(require_streams, Sequence) and not isinstance(require_streams, str)
            else require_streams
        ),
        "episode_selection": episode_selection,
        "total_episodes": len(facade._episodes),
        "fps": int(fps),
        "robot_type": robot_type,
        "include_stats": bool(include_stats),
    }
    view_id = _view_digest(definition, table_versions)

    derived = derive_manifest(
        facade,
        repo_id=repo_id,
        fps=fps,
        statuses=statuses,
        min_confidence=min_confidence,
        require_streams=require_streams,
        robot_type=robot_type,
        lake_uri=None,  # the client's own root is injected at materialization time
        view_id=view_id,
        table_versions=table_versions,
        include_stats=include_stats,
        stats_options=stats_options,
    )

    now = datetime.now(UTC)
    # Files first, header last: a reader that resolves views through the header
    # can never observe a view whose files are still being written (0141 shape).
    # The derive stream is consumed one file at a time and flushed in commits
    # bounded by BOTH file count and bytes (BUG-02: no oversized one-shot
    # write), so publish-side peak memory is one write batch, not the whole
    # derived meta/ tree.
    files_table = lake.table(VIEW_FILES_TABLE)
    pending: list[dict[str, Any]] = []
    pending_bytes = 0
    file_count = 0
    files_bytes = 0

    def _flush_pending() -> None:
        nonlocal pending, pending_bytes
        if pending:
            batch = pa.Table.from_pylist(pending, schema=LEROBOT_VIEW_FILES_SCHEMA)
            _merge_insert_with_retry(files_table, "file_id", batch)
            pending, pending_bytes = [], 0

    for path, content in derived.files:
        pending.append(
            {
                "file_id": f"{view_id}/{path}",
                "view_id": view_id,
                "path": path,
                "content": content,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "created_at": now,
            }
        )
        pending_bytes += len(content)
        file_count += 1
        files_bytes += len(content)
        if len(pending) >= _FILE_WRITE_BATCH or pending_bytes >= _FILE_WRITE_BATCH_BYTES:
            _flush_pending()
    _flush_pending()

    header = {
        "view_id": view_id,
        "repo_id": repo_id,
        "alignment_id": resolved_alignment_id,
        "definition_json": _canonical_json(definition),
        "table_versions": table_versions,
        "fps": int(fps),
        "robot_type": robot_type,
        "total_frames": derived.total_frames,
        "total_episodes": derived.total_episodes,
        "stats_sampling_json": (
            _canonical_json(derived.stats_sampling) if derived.stats_sampling else None
        ),
        "file_count": file_count,
        "files_bytes": files_bytes,
        "created_by": created_by,
        "created_at": now,
    }
    _merge_insert_with_retry(
        lake.table(VIEWS_TABLE),
        "view_id",
        pa.Table.from_pylist([header], schema=LEROBOT_VIEWS_SCHEMA),
    )

    # Postcondition (BUG-04 rule): never report success without checking the
    # write actually landed complete. Distinct keys, because concurrent
    # identical publishers can land benign byte-identical duplicates (see the
    # module docstring); the projection is one small string column, bounded by
    # the loud over-read guard below.
    persisted_rows = (
        lake.table(VIEW_FILES_TABLE)
        .search()
        .select(["file_id"])
        .where(f"view_id = {_sql_literal(view_id)}")
        .limit(max(file_count * 8, 64) + 1)
        .to_arrow()
        .to_pylist()
    )
    persisted_files = len({row["file_id"] for row in persisted_rows})
    if persisted_files != file_count or len(persisted_rows) > max(file_count * 8, 64):
        raise ViewPublishError(
            f"view {view_id!r} published {file_count} files but {persisted_files} "
            f"distinct ({len(persisted_rows)} physical) rows are readable back; "
            "not reporting success"
        )
    pointer_updated = _update_latest_pointer(
        lake, repo_id=repo_id, view_id=view_id, view_created_at=now
    )
    return PublishedView(
        view_id=view_id,
        repo_id=repo_id,
        alignment_id=resolved_alignment_id,
        total_frames=derived.total_frames,
        total_episodes=derived.total_episodes,
        file_count=file_count,
        files_bytes=files_bytes,
        table_versions=tuple(table_versions),
        created_at=now,
        latest_pointer_updated=pointer_updated,
    )


# ── Latest-view pointer (backlog 0507) ───────────────────────────────────────


def _ensure_latest_table(lake: Lake) -> Any:
    """Open ``lerobot_view_latest``, creating it on lakes initialized before 0507.

    ``lake init`` creates the table for new/upgraded lakes; publish creates it
    on demand elsewhere so older lakes gain the pointer without a manual
    upgrade. Creation failures propagate to the caller, which degrades loudly.
    """
    try:
        return lake.table(VIEW_LATEST_TABLE)
    except Exception:  # noqa: BLE001 - missing table; try to create it below.
        lake._db.create_table(VIEW_LATEST_TABLE, schema=LEROBOT_VIEW_LATEST_SCHEMA, exist_ok=True)
        return lake.table(VIEW_LATEST_TABLE)


# Newest-wins CAS condition for the pointer upsert. The view_id tie-break keeps
# the pointer consistent with every other "newest" comparison in this module
# ((created_at, view_id) tuples) when two distinct views land in the same
# microsecond.
_POINTER_UPDATE_CONDITION = (
    "target.view_created_at < source.view_created_at OR "
    "(target.view_created_at = source.view_created_at AND target.view_id < source.view_id)"
)


def _cas_update_latest_pointer(table: Any, row: Mapping[str, Any]) -> bool:
    """Predicate-gated CAS pointer update for backends without conditional merge.

    The 0258 shape: read the observed pointer, then ``update`` gated on exactly
    that observed ``(view_created_at, view_id)`` -- a racing writer changes the
    predicate and the loser updates zero rows. Returns True when the pointer
    ends at least as new as ``row`` (ours, or a newer concurrent winner --
    both are correct outcomes); raises after bounded attempts.
    """
    ours = (row["view_created_at"], str(row["view_id"]))
    for _ in range(_MERGE_INSERT_ATTEMPTS):
        observed = (
            table.search()
            .select(["repo_id", "view_id", "view_created_at"])
            .where(f"repo_id = {_sql_literal(row['repo_id'])}")
            .limit(257)
            .to_arrow()
            .to_pylist()
        )
        if not observed:
            # Insert-only merge: a concurrent insert makes this a no-op and the
            # next iteration observes the winner.
            data = pa.Table.from_pylist([dict(row)], schema=LEROBOT_VIEW_LATEST_SCHEMA)
            table.merge_insert("repo_id").when_not_matched_insert_all().execute(data)
            continue
        best = max(observed, key=lambda r: (r["view_created_at"], str(r["view_id"])))
        if (best["view_created_at"], str(best["view_id"])) >= ours:
            return True
        ts = _timestamp_literal(best["view_created_at"])
        updated = table.update(
            where=(
                f"repo_id = {_sql_literal(row['repo_id'])} "
                f"AND view_created_at = {ts} "
                f"AND view_id = {_sql_literal(best['view_id'])}"
            ),
            values={
                "view_id": str(row["view_id"]),
                "view_created_at": row["view_created_at"],
                "created_at": row["created_at"],
            },
        )
        del updated  # rows-affected shape differs per backend; re-read decides.
    raise ViewPublishError(
        f"pointer CAS update for repo_id {row['repo_id']!r} kept losing after "
        f"{_MERGE_INSERT_ATTEMPTS} attempts"
    )


def _update_latest_pointer(
    lake: Lake, *, repo_id: str, view_id: str, view_created_at: datetime
) -> bool:
    """Upsert the ``repo_id -> newest view`` pointer; True when it landed.

    Newest-wins CAS: the matched-row update is conditional on the pointed-at
    view being newer (``_POINTER_UPDATE_CONDITION``), so a late-arriving older
    publish can never regress the pointer (last-writer-wins would). If the
    backend rejects the conditional merge form, a predicate-gated
    ``Table.update`` CAS loop (the 0258 shape) is used instead -- never a
    blind last-writer-wins.

    On persistent failure the publish is NOT failed (the view landed and its
    postconditions passed; failing here would force a full republish to fix an
    accelerator row). Instead any pre-existing pointer row -- which would
    otherwise keep serving the *previous* view as "latest" with no signal --
    is best-effort deleted so ``get_view`` genuinely falls back, and a warning
    says which state the caller is in. A crash between the catalog write and
    this update leaves a stale pointer that ``compact_view_catalog`` (run by
    ``lake maintain``) reconciles.
    """
    row = {
        "repo_id": repo_id,
        "view_id": view_id,
        "view_created_at": view_created_at,
        "created_at": datetime.now(UTC),
    }
    failure: BaseException | None = None
    table = None
    try:
        table = _ensure_latest_table(lake)
        data = pa.Table.from_pylist([row], schema=LEROBOT_VIEW_LATEST_SCHEMA)
        try:
            _merge_insert_with_retry(
                table, "repo_id", data, update_condition=_POINTER_UPDATE_CONDITION
            )
            return True
        except ViewPublishError:
            raise
        except Exception:  # noqa: BLE001 - conditional form rejected by backend.
            warnings.warn(
                f"backend rejected the conditional newest-wins update on "
                f"{VIEW_LATEST_TABLE!r}; using a predicate-gated CAS update for "
                f"repo_id {repo_id!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            if _cas_update_latest_pointer(table, row):
                return True
    except Exception as exc:  # noqa: BLE001 - degrade loudly, never fail the publish.
        failure = exc
    # The pointer could not be brought up to date. Remove any existing pointer
    # rows so resolves fall back to a real read instead of silently serving the
    # previous view as "latest" (scale-review H1).
    removed = False
    if table is not None:
        try:
            table.delete(f"repo_id = {_sql_literal(repo_id)}")
            removed = True
        except Exception:  # noqa: BLE001 - removal is best-effort too.
            removed = False
    warnings.warn(
        f"publish could not update the {VIEW_LATEST_TABLE!r} pointer for "
        f"repo_id {repo_id!r}: {failure}; "
        + (
            "the stale pointer was removed, so get_view(repo_id=...) falls back "
            "to a catalog read until a publish lands the pointer"
            if removed
            else "an existing pointer row may keep resolving to the previous "
            "view until `lake maintain` (compact_view_catalog) reconciles it"
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    return False


def _latest_pointer_view_id(lake: Lake, repo_id: str) -> str | None:
    """The pointed-at newest ``view_id`` for ``repo_id``, or None on any miss.

    Bounded point read; physical duplicate pointer rows (the same benign
    merge_insert append race as the other catalog tables) are deduplicated by
    the newest ``(view_created_at, view_id, created_at)``.
    """
    try:
        rows = (
            lake.table(VIEW_LATEST_TABLE)
            .search()
            .select(["repo_id", "view_id", "view_created_at", "created_at"])
            .where(f"repo_id = {_sql_literal(repo_id)}")
            .limit(257)
            .to_arrow()
            .to_pylist()
        )
    except Exception:  # noqa: BLE001 - pointer table absent on pre-0507 lakes.
        return None
    if not rows or len(rows) > 256:
        # Zero rows: no pointer yet. Over the bound: a max over a truncated
        # read could pick a stale pointer -- fall back to a real resolve and
        # let compaction collapse the duplicates.
        return None
    best = max(rows, key=lambda r: (r["view_created_at"], r["view_id"], r["created_at"]))
    return str(best["view_id"])


def _header_rows(
    lake: Lake, where: str | None, *, columns: list[str] | None = None
) -> list[dict[str, Any]]:
    try:
        table = lake.table(VIEWS_TABLE)
    except LakeError as exc:
        raise ViewNotFoundError(
            f"lake at {lake.uri} has no {VIEWS_TABLE!r} table; run `lake init` to "
            "upgrade it, then publish a view"
        ) from exc
    query = table.search().select(columns or _LIST_COLUMNS)
    if where:
        query = query.where(where)
    rows = query.limit(_MAX_CATALOG_SCAN_ROWS + 1).to_arrow().to_pylist()
    if len(rows) > _MAX_CATALOG_SCAN_ROWS:
        raise ViewError(
            f"more than {_MAX_CATALOG_SCAN_ROWS} published views match; narrow the "
            "filter or page with list_view_pages (keyset paging; "
            "`train view list --page-size/--cursor`)"
        )
    # Concurrent same-view publishers can land byte-identical duplicate rows
    # (see module docstring); reads deduplicate by view_id.
    return list({row["view_id"]: row for row in rows}.values())


def list_views(
    lake: Lake, *, repo_id: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    """Published views, newest first — bounded listing columns only.

    Convenience listing behind the loud over-read guard: past
    ``_MAX_CATALOG_SCAN_ROWS`` matching rows it raises instead of degrading.
    Use :func:`list_view_pages` to walk a catalog of any size.
    """
    where = f"repo_id = {_sql_literal(repo_id)}" if repo_id else None
    rows = _header_rows(lake, where)
    rows.sort(key=lambda row: (row["created_at"], row["view_id"]), reverse=True)
    return rows[: max(0, int(limit))]


# ── Keyset-paged listing (backlog 0507) ──────────────────────────────────────


@dataclass(frozen=True)
class ViewsPage:
    """One newest-first page of published-view headers.

    ``next_cursor`` is an opaque resume token (empty when the listing is
    exhausted); pass it back to :func:`list_view_pages` for the next page. The
    cursor is a stable ``(created_at, view_id)`` keyset position: views
    published *after* a page was read never shift later pages, and every page
    read is bounded regardless of catalog size.
    """

    rows: tuple[dict[str, Any], ...]
    next_cursor: str


def _encode_view_cursor(row: Mapping[str, Any]) -> str:
    if row["created_at"] is None:
        raise ViewError(
            f"cannot page past view {row['view_id']!r}: its created_at is null "
            "(written outside publish_view); repair the row or list unpaged"
        )
    payload = {
        "created_at": row["created_at"].astimezone(UTC).isoformat(),
        "view_id": str(row["view_id"]),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return encoded.decode("ascii")


def _decode_view_cursor(cursor: str | None) -> tuple[datetime, str] | None:
    if not cursor:
        return None
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(str(cursor).encode("ascii")).decode("utf-8")
        )
        return (
            datetime.fromisoformat(payload["created_at"]).astimezone(UTC),
            str(payload["view_id"]),
        )
    except Exception as exc:
        raise ViewError(
            "invalid view listing cursor; pass a next_cursor returned by "
            "list_view_pages"
        ) from exc


def _timestamp_literal(value: datetime) -> str:
    # Typed literal, not a quoted string: Lance compares strings to timestamp
    # columns lexically otherwise (0146 gotcha).
    return "timestamp '" + value.astimezone(UTC).isoformat() + "'"


def _view_page_where(repo_id: str | None, cursor: tuple[datetime, str] | None) -> str | None:
    clauses: list[str] = []
    if repo_id:
        clauses.append(f"repo_id = {_sql_literal(repo_id)}")
    if cursor is not None:
        created_at, view_id = cursor
        ts = _timestamp_literal(created_at)
        clauses.append(
            f"(created_at < {ts} OR "
            f"(created_at = {ts} AND view_id < {_sql_literal(view_id)}))"
        )
    if not clauses:
        return None
    return " AND ".join(clauses)


def _view_page_key(row: Mapping[str, Any]) -> tuple[tuple[bool, datetime], str]:
    # Null-safe: a row stamped outside publish_view sorts as oldest instead of
    # crashing tuple comparison (encoding a cursor AT such a row still raises).
    created = row["created_at"]
    if created is None:
        created = datetime.min.replace(tzinfo=UTC)
        return ((False, created), str(row["view_id"]))
    return ((True, created), str(row["view_id"]))


def _list_view_page_ordered(
    lake: Lake,
    *,
    where: str | None,
    cursor: tuple[datetime, str] | None,
    page_size: int,
) -> list[dict[str, Any]] | None:
    """Descending ordered page read with early break; None when unorderable."""
    try:
        from lancedb.query import ColumnOrdering

        query = lake.table(VIEWS_TABLE).search().select(_LIST_COLUMNS)
        if where:
            query = query.where(where)
        query = query.order_by(
            [
                ColumnOrdering(column_name="created_at", ascending=False),
                ColumnOrdering(column_name="view_id", ascending=False),
            ]
        )
        batches = query.to_batches(batch_size=_PAGE_SCAN_BATCH)
    except Exception:  # noqa: BLE001 - ordering unavailable; caller falls back.
        return None
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    want = page_size + 1
    try:
        for batch in batches:
            for row in batch.to_pylist():
                view_id = str(row["view_id"])
                # Dedup physical duplicates: the bounded seen-set catches copies
                # inside this page, the cursor-id check catches copies of the
                # page-boundary row. A copy of an *earlier* page's row whose
                # created_at differs enough to fall past the cursor can still
                # resurface once -- duplicates land microseconds apart, so that
                # window is a page cut inside one race; compaction removes the
                # cause entirely.
                if view_id in seen or (cursor is not None and view_id == cursor[1]):
                    continue
                seen.add(view_id)
                collected.append(row)
                if len(collected) >= want:
                    return collected
    except Exception:  # noqa: BLE001 - ordering rejected lazily; caller falls back.
        return None
    return collected


def _list_view_page_heap(
    lake: Lake,
    *,
    where: str | None,
    cursor: tuple[datetime, str] | None,
    page_size: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Bounded top-(page_size+1) heap over the scoped scan (order fallback).

    O(page_size) client memory, O(scope) rows scanned for this one page. Rows
    are streamed with the keyset predicate pushed down, so already-listed pages
    are excluded server-side. The second return value is True when any scanned
    row fell outside the kept set -- duplicates can crowd the heap and shorten
    a page below ``page_size``, so "more pages exist" must be tracked from the
    scan itself, never inferred from the deduplicated page length (that would
    silently truncate the listing).
    """
    import heapq

    query = lake.table(VIEWS_TABLE).search().select(_LIST_COLUMNS)
    if where:
        query = query.where(where)
    want = page_size + 1
    # Min-heap of the largest `want` keys seen; heap entries carry the sort key
    # ascending so the smallest of the kept set is evicted first.
    heap: list[tuple[tuple[datetime, str], int, dict[str, Any]]] = []
    tie = 0
    overflow = False
    for batch in query.to_batches(batch_size=_PAGE_SCAN_BATCH):
        for row in batch.to_pylist():
            if cursor is not None and str(row["view_id"]) == cursor[1]:
                continue
            key = _view_page_key(row)
            tie += 1
            if len(heap) < want:
                heapq.heappush(heap, (key, tie, row))
            elif key > heap[0][0]:
                heapq.heapreplace(heap, (key, tie, row))
                overflow = True
            else:
                overflow = True
    rows = [entry[2] for entry in heap]
    rows.sort(key=_view_page_key, reverse=True)
    # Deduplicate physical duplicate rows by view_id, newest kept.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        view_id = str(row["view_id"])
        if view_id in seen:
            continue
        seen.add(view_id)
        deduped.append(row)
    return deduped, overflow


def list_view_pages(
    lake: Lake,
    *,
    repo_id: str | None = None,
    page_size: int = _DEFAULT_PAGE_SIZE,
    cursor: str | None = None,
) -> ViewsPage:
    """One bounded, resumable newest-first page of published views.

    Deterministic descending ``(created_at, view_id)`` order with an opaque
    keyset cursor -- the scalable listing surface for catalogs past the
    :func:`list_views` guard (0142/0145 precedents). The primary path pushes
    the scope + keyset predicate and a descending ``order_by`` into Lance and
    stops after ``page_size + 1`` distinct views; when the backend cannot
    order the scan it falls back to a bounded top-k heap with a warning
    (deterministic, O(page_size) client memory, O(scope) rows scanned per
    page -- never silently unbounded).
    """
    if page_size < 1 or page_size > _MAX_PAGE_SIZE:
        raise ViewError(f"page_size must be between 1 and {_MAX_PAGE_SIZE}")
    # Fail loudly on a missing catalog table before paging (same typed error
    # and `lake init` remedy as every other listing path).
    try:
        lake.table(VIEWS_TABLE)
    except LakeError as exc:
        raise ViewNotFoundError(
            f"lake at {lake.uri} has no {VIEWS_TABLE!r} table; run `lake init` to "
            "upgrade it, then publish a view"
        ) from exc
    decoded = _decode_view_cursor(cursor)
    where = _view_page_where(repo_id, decoded)
    rows = _list_view_page_ordered(
        lake, where=where, cursor=decoded, page_size=page_size
    )
    if rows is not None:
        has_more = len(rows) > page_size
    else:
        warnings.warn(
            "view-catalog paging could not order the scan in the backend; using "
            "a bounded heap over a scoped scan (deterministic, O(page_size) "
            "client memory, but O(scope) rows scanned per page).",
            RuntimeWarning,
            stacklevel=2,
        )
        rows, overflow = _list_view_page_heap(
            lake, where=where, cursor=decoded, page_size=page_size
        )
        has_more = overflow or len(rows) > page_size
    page_rows = rows[:page_size]
    next_cursor = _encode_view_cursor(page_rows[-1]) if has_more and page_rows else ""
    return ViewsPage(rows=tuple(page_rows), next_cursor=next_cursor)


def _newest_view_id_ordered(lake: Lake, repo_id: str) -> str | None:
    """Newest ``view_id`` for ``repo_id`` via a backend-ordered top-1 read.

    Pushes the ``repo_id`` predicate plus a descending ``(created_at, view_id)``
    ``order_by`` into Lance and reads only the first row -- bounded regardless
    of how many views the repo_id has. Returns None when the backend cannot
    order the scan (the caller falls back to the guarded bounded scan); raises
    :class:`ViewNotFoundError` when the ordered read worked and found nothing.
    """
    try:
        from lancedb.query import ColumnOrdering

        query = (
            lake.table(VIEWS_TABLE)
            .search()
            .select(["view_id", "created_at"])
            .where(f"repo_id = {_sql_literal(repo_id)}")
            .order_by(
                [
                    ColumnOrdering(column_name="created_at", ascending=False),
                    ColumnOrdering(column_name="view_id", ascending=False),
                ]
            )
        )
        batches = query.to_batches(batch_size=8)
        for batch in batches:
            rows = batch.to_pylist()
            if rows:
                return str(rows[0]["view_id"])
        raise ViewNotFoundError(
            f"no published view named {repo_id!r}; publish one with "
            "`lancedb-robotics train view publish`"
        )
    except ViewNotFoundError:
        raise
    except Exception:  # noqa: BLE001 - ordering unavailable; caller falls back.
        return None


def _resolve_latest_view_id(lake: Lake, repo_id: str) -> str:
    """Newest ``view_id`` for ``repo_id``: pointer, then ordered top-1, then scan.

    Chain (each step bounded, each degradation explicit):

    1. ``lerobot_view_latest`` point read -- O(1) per open, every backend.
    2. Backend-ordered descending top-1 over ``lerobot_views`` -- pre-0507
       lakes without a pointer row.
    3. Today's loudly-guarded bounded scan -- backends that cannot order a
       scan; warns because it materializes up to the guard bound (SKILLS.md:
       never take the costlier path silently).

    A pointer naming a header that no longer resolves is treated as a miss
    (warned), not trusted -- reads never serve a dangling pointer.
    """
    pointed = _latest_pointer_view_id(lake, repo_id)
    if pointed is not None:
        if _header_rows(lake, f"view_id = {_sql_literal(pointed)}"):
            return pointed
        warnings.warn(
            f"{VIEW_LATEST_TABLE!r} points repo_id {repo_id!r} at view "
            f"{pointed!r} which has no header row; ignoring the stale pointer "
            "(re-publish the view to repair it)",
            RuntimeWarning,
            stacklevel=2,
        )
    ordered = _newest_view_id_ordered(lake, repo_id)
    if ordered is not None:
        return ordered
    warnings.warn(
        "the backend could not order the view-catalog scan; resolving the "
        f"newest view for repo_id {repo_id!r} via the guarded bounded scan",
        RuntimeWarning,
        stacklevel=2,
    )
    rows = _header_rows(lake, f"repo_id = {_sql_literal(repo_id)}")
    if not rows:
        raise ViewNotFoundError(
            f"no published view named {repo_id!r}; publish one with "
            "`lancedb-robotics train view publish`"
        )
    return str(max(rows, key=lambda row: (row["created_at"], row["view_id"]))["view_id"])


def get_view(
    lake: Lake, *, repo_id: str | None = None, view_id: str | None = None
) -> dict[str, Any]:
    """One published view header (full row): exact ``view_id``, or newest for ``repo_id``.

    The ``repo_id`` resolve goes through the ``lerobot_view_latest`` pointer
    (one point read; ordered top-1 and guarded-scan fallbacks -- see
    :func:`_resolve_latest_view_id`), then point-reads that one full row — the
    unbounded-ish ``definition_json`` never rides a multi-row scan.
    """
    if not view_id:
        if not repo_id:
            raise ViewError("get_view needs repo_id or view_id")
        view_id = _resolve_latest_view_id(lake, repo_id)
    rows = _header_rows(
        lake, f"view_id = {_sql_literal(view_id)}", columns=_FULL_COLUMNS
    )
    if not rows:
        raise ViewNotFoundError(f"no published view with view_id {view_id!r}")
    return rows[0]


def _read_view_file(lake: Lake, view_id: str, path: str) -> bytes:
    """Point-read one derived file's content (projection: ``content`` only)."""
    rows = (
        lake.table(VIEW_FILES_TABLE)
        .search()
        .select(["content"])
        .where(f"file_id = {_sql_literal(f'{view_id}/{path}')}")
        .limit(1)
        .to_arrow()
        .to_pylist()
    )
    if not rows:
        raise ViewError(
            f"view {view_id!r} has no stored file {path!r}; the catalog is "
            "inconsistent — re-publish the view"
        )
    return rows[0]["content"]


def open_published_facade(lake: Lake, view: Mapping[str, Any]) -> LiveLeRobotFacade:
    """Open ``view``'s facade over a :class:`PinnedLake` at its recorded versions.

    Reopens from the view's stored source-manifest *file* (one indexed point
    read), the same record the registered dataset reader uses — the header's
    ``definition_json`` records caller intent for audit/digest purposes and
    deliberately does not inline unbounded resolved episode lists.
    """
    from ._reader_core import SourceManifest

    view_id = str(view["view_id"])
    manifest = SourceManifest(json.loads(_read_view_file(lake, view_id, SOURCE_MANIFEST_PATH)))
    pinned = PinnedLake(lake, _versions_map(manifest.table_versions))
    return LiveLeRobotFacade(
        pinned,
        alignment_id=manifest.alignment_id,
        mapping=manifest.mapping,
        episode_ids=manifest.episode_ids,
        statuses=manifest.statuses,
        min_confidence=manifest.min_confidence,
        require_streams=manifest.require_streams,
    )


# ── Materialization: lake rows -> a local meta/ directory ────────────────────


def view_cache_root() -> Path:
    override = os.environ.get(VIEW_CACHE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "lancedb-robotics" / "lerobot-views"


def _cache_dir_name(lake_uri: str, view_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", lake_uri).strip("-")[:80]
    uri_digest = hashlib.sha256(lake_uri.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}-{uri_digest}@{view_id}"


def materialize_view(
    lake: Lake,
    view: Mapping[str, Any],
    dest: Path | str,
    *,
    lake_uri: str | None = None,
    force: bool = False,
) -> Path:
    """Materialize ``view``'s files from the lake into local directory ``dest``.

    Concurrency-safe the lerobot-lancedb way: files are written into a
    pid+uuid-suffixed temp directory next to ``dest`` and atomically renamed
    into place, so N concurrent DataLoader workers converge on one complete
    cache directory and none ever reads a half-written ``meta/``. An existing
    ``dest`` is trusted as a completed materialization and returned untouched
    unless ``force``.

    File contents are fetched one row at a time (projection: ``content`` only,
    keyed by ``file_id``), so peak memory is bounded by the largest single
    derived file, not the view's total size. Every file's sha256 and size are
    verified against the catalog row before the rename; ``lake_uri`` (when
    given -- normally the exact ``root`` the client passed) is injected into
    the materialized source manifest *after* verification, since the published
    row deliberately stores no publish-time URI for the client to mistrust.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    view_id = str(view["view_id"])
    file_count = int(view["file_count"])
    files_table = lake.table(VIEW_FILES_TABLE)
    raw_listing = (
        files_table.search()
        .select(["file_id", "path", "sha256", "size_bytes"])
        .where(f"view_id = {_sql_literal(view_id)}")
        .limit(max(file_count * 8, 64) + 1)
        .to_arrow()
        .to_pylist()
    )
    # Deduplicate by file_id: concurrent identical publishers can land benign
    # byte-identical duplicate rows (see module docstring).
    listing = list({row["file_id"]: row for row in raw_listing}.values())
    if len(listing) != file_count or len(raw_listing) > max(file_count * 8, 64):
        raise ViewError(
            f"view {view_id!r} declares {view['file_count']} files but "
            f"{len(listing)} distinct ({len(raw_listing)} physical) rows are "
            "readable; the catalog is inconsistent — re-publish"
        )

    temp = dest.parent / f".{dest.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    temp.mkdir(parents=True, exist_ok=False)
    try:
        source_manifest_target: Path | None = None
        for entry in listing:
            content_rows = (
                files_table.search()
                .select(["content"])
                .where(f"file_id = {_sql_literal(entry['file_id'])}")
                .limit(1)
                .to_arrow()
                .to_pylist()
            )
            if not content_rows:
                raise ViewError(f"file row {entry['file_id']!r} vanished mid-materialization")
            content = content_rows[0]["content"]
            if len(content) != int(entry["size_bytes"]) or (
                hashlib.sha256(content).hexdigest() != entry["sha256"]
            ):
                raise ViewError(
                    f"content verification failed for {entry['path']!r} of view "
                    f"{view_id!r}; refusing to materialize a corrupt manifest"
                )
            written = write_manifest_files(temp, [(entry["path"], content)])
            if entry["path"] == SOURCE_MANIFEST_PATH:
                source_manifest_target = written / entry["path"]
        if lake_uri is not None and source_manifest_target is not None:
            payload = json.loads(source_manifest_target.read_text())
            payload["lake_uri"] = lake_uri
            source_manifest_target.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temp, dest)
        except OSError:
            # A concurrent worker renamed its own complete copy first: theirs wins.
            if dest.exists():
                shutil.rmtree(temp, ignore_errors=True)
            else:
                raise
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return dest


def _normalize_lake_uri(lake_uri: str) -> str:
    """Map upstream's local-URI form to what ``Lake.open`` accepts.

    lerobot's ``is_remote_uri`` treats any ``scheme://`` root as remote —
    ``file://…`` included — and only such roots reach the per-format probe at
    all. This repo's connection resolver deliberately rejects ``file://`` in
    favor of plain paths, so a client-side ``root="file:///path/robot.lance"``
    is translated to the plain local path here; every other scheme passes
    through untouched.
    """
    if lake_uri.startswith("file://"):
        from urllib.parse import unquote, urlparse

        return unquote(urlparse(lake_uri).path)
    return lake_uri


def resolve_view_root(
    repo_id: str | None,
    lake_uri: str,
    *,
    revision: str | None = None,
    force_cache_sync: bool = False,
    lake_open_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve ``root=<lake uri>`` to a local, cached, materialized view directory.

    ``revision`` selects an exact ``view_id``; otherwise the newest view
    published under ``repo_id`` wins. The cache key includes the view id, so a
    re-published view lands in a *new* directory and previously resolved roots
    stay valid and reproducible.
    """
    normalized = _normalize_lake_uri(lake_uri)
    lake = Lake.open(normalized, **dict(lake_open_kwargs or {}))
    view = get_view(lake, repo_id=repo_id, view_id=revision)
    cache_dir = view_cache_root() / _cache_dir_name(lake_uri, str(view["view_id"]))
    return materialize_view(lake, view, cache_dir, lake_uri=normalized, force=force_cache_sync)
