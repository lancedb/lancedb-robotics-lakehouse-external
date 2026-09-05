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
postcondition counts distinct keys — semantic convergence, verified, with
physical dedup left to the catalog-lifecycle follow-on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake, LakeError
from lancedb_robotics.schemas import LEROBOT_VIEW_FILES_SCHEMA, LEROBOT_VIEWS_SCHEMA

from ._reader_core import SOURCE_MANIFEST_FILENAME
from .manifest import derive_manifest, write_manifest_files
from .mapping import CanonicalVectorMapping
from .reader import LiveLeRobotFacade

VIEWS_TABLE = "lerobot_views"
VIEW_FILES_TABLE = "lerobot_view_files"
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
# publish, so hitting this means the catalog needs the keyset-paging follow-on).
_MAX_CATALOG_SCAN_ROWS = 10_000

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
_UNPINNED_TABLES = frozenset({VIEWS_TABLE, VIEW_FILES_TABLE})


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


def _merge_insert_with_retry(table: Any, key_column: str, data: pa.Table) -> None:
    """Single-commit upsert (BUG-04 shape) with bounded commit-conflict retry."""
    if data.num_rows == 0:
        return
    last_error: BaseException | None = None
    for _ in range(_MERGE_INSERT_ATTEMPTS):
        builder = table.merge_insert(key_column).when_matched_update_all()
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
    )


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
            "filter (this catalog needs keyset paging before growing further)"
        )
    # Concurrent same-view publishers can land byte-identical duplicate rows
    # (see module docstring); reads deduplicate by view_id.
    return list({row["view_id"]: row for row in rows}.values())


def list_views(
    lake: Lake, *, repo_id: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    """Published views, newest first — bounded listing columns only."""
    where = f"repo_id = {_sql_literal(repo_id)}" if repo_id else None
    rows = _header_rows(lake, where)
    rows.sort(key=lambda row: (row["created_at"], row["view_id"]), reverse=True)
    return rows[: max(0, int(limit))]


def get_view(
    lake: Lake, *, repo_id: str | None = None, view_id: str | None = None
) -> dict[str, Any]:
    """One published view header (full row): exact ``view_id``, or newest for ``repo_id``.

    The ``repo_id`` resolve scans bounded listing columns to pick the newest
    view, then point-reads that one full row — the unbounded-ish
    ``definition_json`` never rides a multi-row scan.
    """
    if not view_id:
        if not repo_id:
            raise ViewError("get_view needs repo_id or view_id")
        rows = _header_rows(lake, f"repo_id = {_sql_literal(repo_id)}")
        if not rows:
            raise ViewNotFoundError(
                f"no published view named {repo_id!r}; publish one with "
                "`lancedb-robotics train view publish`"
            )
        view_id = max(rows, key=lambda row: (row["created_at"], row["view_id"]))["view_id"]
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
