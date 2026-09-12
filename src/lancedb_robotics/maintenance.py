"""Lake maintenance: compaction, index refresh, and snapshot-safe retention."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa

from lancedb_robotics.capability_gates import MAINTENANCE, require_lake_capability
from lancedb_robotics.indexing import (
    build_fts_index,
    build_predicate_indexes_for_table,
    build_scalar_index,
    fts_index_columns,
    scalar_index_columns,
    vector_index_columns,
)
from lancedb_robotics.lake import Lake, LakeError
from lancedb_robotics.lineage import (
    LineageError,
    emit_transform_lineage,
    lineage_retention_pin_details,
    merge_retention_pin_details,
    retention_pin_rows,
    snapshot_retention_pin_details,
)
from lancedb_robotics.pylance_execution import require_namespace_write_supported
from lancedb_robotics.scalar_index_jobs import (
    reconcile_scalar_index_jobs as _reconcile_scalar_index_jobs,
)
from lancedb_robotics.scalar_index_jobs import (
    request_aligned_predicate_index_jobs as _request_aligned_predicate_index_jobs,
)
from lancedb_robotics.schemas import CANONICAL_TABLES, TRANSFORM_RUNS_SCHEMA

MAINTENANCE_TRANSFORM_KIND = "maintenance"
_PIN_TAG_PREFIX = "lbr-snapshot-pin-v"


class MaintenanceError(Exception):
    """Raised when a lake maintenance operation cannot be completed."""


@dataclass(frozen=True)
class TableMaintenanceReport:
    """Maintenance outcome for one table."""

    table: str
    version_before: int
    version_after: int
    fragments_before: int
    fragments_after: int
    fragments_removed: int = 0
    fragments_added: int = 0
    files_removed: int = 0
    files_added: int = 0
    indexes_refreshed: tuple[dict[str, Any], ...] = ()
    pinned_versions: tuple[int, ...] = ()
    pinned_tags: tuple[str, ...] = ()
    lineage_pinned_versions: tuple[int, ...] = ()
    lineage_pin_reasons: tuple[dict[str, Any], ...] = ()
    retention_reasons: tuple[dict[str, Any], ...] = ()
    retention_hold_versions: tuple[int, ...] = ()
    cleanup_candidate_versions: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    cleanup: dict[str, int] | None = None

    def to_params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaintenanceReport:
    """Summary of one lake maintenance run."""

    lake_uri: str
    transform_id: str
    tables: dict[str, TableMaintenanceReport] = field(default_factory=dict)
    required_audit_report: dict[str, Any] | None = None
    lerobot_checkpoint_retention: dict[str, Any] | None = None
    aligned_tick_lifecycle: dict[str, Any] | None = None
    scalar_index_jobs: dict[str, Any] | None = None
    curation_membership_chunks: dict[str, Any] | None = None
    curation_row_plan_chunks: dict[str, Any] | None = None
    curation_replay_retention: dict[str, Any] | None = None
    lerobot_view_catalog: dict[str, Any] | None = None
    lerobot_view_retention: dict[str, Any] | None = None
    lerobot_view_readiness: dict[str, Any] | None = None


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _metric_dict(obj: Any, names: tuple[str, ...]) -> dict[str, int]:
    return {name: int(getattr(obj, name, 0) or 0) for name in names}


def _table_versions_payload(lake: Lake, tables: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"table": table, "version": int(lake.table(table).version), "tag": ""} for table in tables]


def _tag_pinned_versions(
    lake: Lake, table: str, versions: set[int]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Tag pinned versions that still exist; return (created tags, skipped versions)."""
    dataset = lake.table(table).to_lance()
    tags = dataset.tags
    created: list[str] = []
    skipped: list[int] = []
    existing = tags.list()
    # A pinned version can already have been pruned/compacted away -- e.g. lineage
    # records a historical table version (backlog 0098 inline emission, or a
    # refresh run between writes) that a later compaction superseded, or a snapshot
    # pins a version that was reclaimed. A tag cannot protect a version that no
    # longer exists, so tag only versions still present rather than crashing.
    try:
        available = {
            int(entry["version"] if isinstance(entry, dict) else entry.version)
            for entry in dataset.versions()
        }
    except Exception:  # noqa: BLE001 - fall back to attempting every pinned version
        available = None
    for tag in sorted(existing):
        managed_version = _managed_pin_version(tag)
        if managed_version is None or managed_version in versions:
            continue
        try:
            tags.delete(tag)
        except Exception as exc:  # noqa: BLE001 - expose the table/version context
            raise MaintenanceError(
                f"cannot remove expired managed pin tag {tag!r} from {table!r}: {exc}"
            ) from exc
    for version in sorted(versions):
        if available is not None and version not in available:
            skipped.append(version)
            continue
        tag = f"{_PIN_TAG_PREFIX}{version}"
        try:
            if tag not in existing:
                tags.create(tag, version)
            else:
                tags.update(tag, version)
        except Exception as exc:  # noqa: BLE001 - expose the table/version context
            raise MaintenanceError(
                f"cannot tag snapshot-pinned version {version} of {table!r}: {exc}"
            ) from exc
        created.append(tag)
    return tuple(created), tuple(skipped)


def _managed_pin_version(tag: str) -> int | None:
    if not tag.startswith(_PIN_TAG_PREFIX):
        return None
    try:
        return int(tag.removeprefix(_PIN_TAG_PREFIX))
    except ValueError:
        return None


def _table_retention_rows(
    table: str,
    pin_details: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(row for row in retention_pin_rows({table: pin_details}) if row["table"] == table)


def _versions_with_category(
    pin_details: dict[int, dict[str, Any]],
    category: str,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            version
            for version, detail in pin_details.items()
            if category in detail.get("categories", set())
        )
    )


def _cleanup_candidate_versions(lake: Lake, table: str, pinned_versions: set[int]) -> tuple[int, ...]:
    try:
        dataset = lake.table(table).to_lance()
        current = int(dataset.version)
        versions = dataset.versions()
    except Exception:
        return ()
    return tuple(
        sorted(
            int(item["version"])
            for item in versions
            if int(item.get("version") or 0) != current
            and int(item.get("version") or 0) not in pinned_versions
        )
    )


def _fragment_count(lake: Lake, table: str) -> int:
    return len(lake.table(table).to_lance().get_fragments())


def _refresh_indexes(lake: Lake, table: str) -> tuple[dict[str, Any], ...]:
    handle = lake.table(table)
    refreshed: list[dict[str, Any]] = []

    for column in sorted(fts_index_columns(handle)):
        refreshed.append(build_fts_index(lake, table=table, column=column).to_params())

    for column in sorted(scalar_index_columns(handle)):
        refreshed.append(
            build_scalar_index(
                lake,
                table=table,
                column=column,
                replace=True,
            ).to_params()
        )

    # Create any managed grain/lineage predicate indexes this table is missing
    # (backlog 0181 / BUG-15), so `lake maintain` upgrades a pre-existing lake whose
    # observations/lineage tables were never indexed. Runs after compaction above;
    # build-if-absent (replace=False), and the refresh loop above maintains them on
    # subsequent runs. Tables without a managed set return no rows.
    for result in build_predicate_indexes_for_table(lake, table, replace=False):
        if result.status != "already_present":
            refreshed.append(result.to_params())

    vector_columns = sorted(vector_index_columns(handle))
    if vector_columns:
        try:
            metrics = handle.to_lance().optimize.optimize_indices()
        except Exception as exc:  # noqa: BLE001 - wrap the engine error with context
            raise MaintenanceError(f"cannot optimize vector indexes for {table!r}: {exc}") from exc
        refreshed.append(
            {
                "table": table,
                "columns": vector_columns,
                "index_type": "VECTOR",
                "status": "optimized",
                "metrics": {
                    name: getattr(metrics, name)
                    for name in dir(metrics)
                    if not name.startswith("_")
                    and isinstance(getattr(metrics, name), int | float | str | bool | type(None))
                },
            }
        )

    return tuple(refreshed)


def maintain_lake(
    lake: Lake,
    *,
    tables: tuple[str, ...] | list[str] | None = None,
    compact: bool = True,
    refresh_indexes: bool = True,
    protect_lineage: bool = True,
    refresh_lineage: bool = True,
    cleanup_older_than: timedelta | None = timedelta(days=7),
    retain_versions: int | None = None,
    delete_unverified: bool = False,
    require_recent_audit: bool = False,
    audit_max_age: timedelta | None = timedelta(hours=24),
    lerobot_checkpoint_retention: bool = True,
    lerobot_checkpoint_retention_older_than: timedelta | None = timedelta(days=30),
    lerobot_checkpoint_retention_source_id: str | None = None,
    lerobot_checkpoint_retention_statuses: tuple[str, ...] | list[str] | None = None,
    lerobot_checkpoint_retain_completed_per_source: int = 10,
    lerobot_checkpoint_retain_failed_per_source: int = 10,
    aligned_tick_lifecycle_diagnostics: bool = True,
    reconcile_index_jobs: bool = True,
    request_predicate_index_jobs: bool = False,
    apply_predicate_recommendations: bool = False,
    predicate_recommendation_limit: int = 200,
    curation_chunk_maintenance: bool = True,
    compact_curation_chunks: bool = True,
    curation_replay_retention: bool = True,
    lerobot_view_catalog_compaction: bool = True,
    protect_lerobot_views: bool = True,
    lerobot_view_pin_max_views: int | None = None,
    lerobot_view_retention: bool = True,
    lerobot_view_retention_apply: bool = False,
    lerobot_view_retention_older_than: timedelta | None = None,
    lerobot_view_retain_latest_per_repo: int | None = None,
    lerobot_view_readiness: bool = True,
    created_by: str = "lancedb-robotics",
) -> MaintenanceReport:
    """Compact tables, refresh existing indexes, and prune old unpinned versions.

    Snapshot safety is explicit: before cleanup, every version referenced by a
    live ``dataset_snapshots.table_versions`` entry is tagged in its table. Lance
    cleanup is then run with ``error_if_tagged_old_versions=False``, which prunes
    eligible untagged versions while retaining tagged snapshot pins.
    """
    selected = tuple(tables or CANONICAL_TABLES)
    unknown = [table for table in selected if table not in CANONICAL_TABLES]
    if unknown:
        raise LakeError(f"unknown canonical table(s) for maintenance: {unknown}")

    # Maintenance drops to the underlying LanceDataset (get_fragments, compact_files,
    # cleanup_old_versions, version tags). Fail fast before any lineage refresh or
    # compaction when the backend cannot support it (e.g. a db:// remote DB).
    require_lake_capability(lake, MAINTENANCE, operation="lake maintenance")

    # Compaction/version-cleanup create new table versions. On a namespace-backed
    # lake whose namespace manages versions, refuse rather than fork the managed
    # version history behind the server (0129). No-op for local/object-store/db://.
    for table in selected:
        require_namespace_write_supported(lake.connection_spec, table)

    if protect_lineage and refresh_lineage:
        try:
            lake.lineage.refresh_graph()
        except Exception as exc:  # noqa: BLE001 - maintenance must explain the lake context
            raise MaintenanceError(f"cannot refresh lineage graph before maintenance: {exc}") from exc

    required_audit_report: dict[str, Any] | None = None
    cleanup_requested = cleanup_older_than is not None or retain_versions is not None
    if require_recent_audit and cleanup_requested:
        try:
            from lancedb_robotics.lineage_audit_catalog import (
                require_recent_passed_audit_report,
            )

            required_audit_report = require_recent_passed_audit_report(
                lake,
                max_age=audit_max_age,
            ).as_dict()
        except LineageError as exc:
            raise MaintenanceError(str(exc)) from exc

    started = datetime.now(UTC)
    input_versions = _table_versions_payload(lake, selected)
    lerobot_retention_report: dict[str, Any] | None = None
    if lerobot_checkpoint_retention and "lerobot_ingest_checkpoints" in selected:
        from lancedb_robotics.ingest import apply_lerobot_checkpoint_retention

        lerobot_retention_report = apply_lerobot_checkpoint_retention(
            lake,
            older_than=lerobot_checkpoint_retention_older_than,
            statuses=lerobot_checkpoint_retention_statuses,
            source_id=lerobot_checkpoint_retention_source_id,
            retain_completed_per_source=lerobot_checkpoint_retain_completed_per_source,
            retain_failed_per_source=lerobot_checkpoint_retain_failed_per_source,
            dry_run=False,
            compact=False,
            cleanup_older_than=None,
            retain_versions=None,
            now=started,
        ).to_params()
    snapshot_pin_details = snapshot_retention_pin_details(lake)
    lineage_pin_details = lineage_retention_pin_details(lake) if protect_lineage else {}
    # Versions pinned by non-retired published LeRobot views (backlog 0508) join
    # the same tag-before-cleanup pin map as snapshots and lineage: pruning one
    # would turn every open of the referencing view into StaleViewVersionError.
    # NOT best-effort -- silently skipping these pins and then pruning is the
    # exact data-loss shape the pin map exists to prevent, so a failure here
    # fails maintenance loudly before any cleanup runs.
    lerobot_view_pin_details: dict[str, dict[int, dict[str, Any]]] = {}
    if protect_lerobot_views:
        from lancedb_robotics.lerobot_facade.view_retention import (
            view_retention_pin_details,
        )

        try:
            if lerobot_view_pin_max_views is not None:
                lerobot_view_pin_details = view_retention_pin_details(
                    lake, max_views=lerobot_view_pin_max_views
                )
            else:
                lerobot_view_pin_details = view_retention_pin_details(lake)
        except Exception as exc:  # noqa: BLE001 - refuse to prune with unknown pins.
            raise MaintenanceError(
                f"cannot resolve published-view version pins: {exc}"
            ) from exc
    retention_pin_details = merge_retention_pin_details(
        snapshot_pin_details,
        lineage_pin_details,
        lerobot_view_pin_details,
    )
    table_reports: dict[str, TableMaintenanceReport] = {}

    for table in selected:
        handle = lake.table(table)
        version_before = int(handle.version)
        fragments_before = _fragment_count(lake, table)
        lineage_versions = lineage_pin_details.get(table, {})
        table_pin_details = retention_pin_details.get(table, {})
        pinned_versions = set(table_pin_details)
        pinned_tags, skipped_pins = _tag_pinned_versions(lake, table, pinned_versions)
        lineage_pin_reasons = tuple(
            {
                "version": version,
                "reasons": sorted(detail["reasons"]),
                "categories": sorted(detail["categories"]),
            }
            for version, detail in sorted(lineage_versions.items())
        )
        retention_reasons = _table_retention_rows(table, table_pin_details)
        retention_hold_versions = _versions_with_category(table_pin_details, "retention-hold")
        cleanup_candidate_versions = _cleanup_candidate_versions(lake, table, pinned_versions)
        warnings: tuple[str, ...] = ()
        if lineage_versions:
            warnings += (
                f"retained lineage-referenced {table} version(s): "
                + ", ".join(str(version) for version in sorted(lineage_versions)),
            )
        if skipped_pins:
            warnings += (
                f"skipped tagging pruned pinned {table} version(s) no longer on disk: "
                + ", ".join(str(version) for version in skipped_pins),
            )

        compaction = {
            "fragments_removed": 0,
            "fragments_added": 0,
            "files_removed": 0,
            "files_added": 0,
        }
        if compact:
            try:
                compaction = _metric_dict(
                    handle.to_lance().optimize.compact_files(),
                    ("fragments_removed", "fragments_added", "files_removed", "files_added"),
                )
            except Exception as exc:  # noqa: BLE001 - wrap the engine error with context
                raise MaintenanceError(f"cannot compact table {table!r}: {exc}") from exc

        index_results = _refresh_indexes(lake, table) if refresh_indexes else ()

        cleanup = None
        if cleanup_older_than is not None or retain_versions is not None:
            try:
                stats = lake.table(table).to_lance().cleanup_old_versions(
                    older_than=cleanup_older_than,
                    retain_versions=retain_versions,
                    delete_unverified=delete_unverified,
                    error_if_tagged_old_versions=False,
                )
            except Exception as exc:  # noqa: BLE001 - wrap the engine error with context
                raise MaintenanceError(f"cannot clean old versions for {table!r}: {exc}") from exc
            cleanup = _metric_dict(
                stats,
                (
                    "bytes_removed",
                    "old_versions",
                    "data_files_removed",
                    "transaction_files_removed",
                    "index_files_removed",
                    "deletion_files_removed",
                ),
            )

        version_after = int(lake.table(table).version)
        fragments_after = _fragment_count(lake, table)
        table_reports[table] = TableMaintenanceReport(
            table=table,
            version_before=version_before,
            version_after=version_after,
            fragments_before=fragments_before,
            fragments_after=fragments_after,
            fragments_removed=compaction["fragments_removed"],
            fragments_added=compaction["fragments_added"],
            files_removed=compaction["files_removed"],
            files_added=compaction["files_added"],
            indexes_refreshed=index_results,
            pinned_versions=tuple(sorted(pinned_versions)),
            pinned_tags=pinned_tags,
            lineage_pinned_versions=tuple(sorted(lineage_versions)),
            lineage_pin_reasons=lineage_pin_reasons,
            retention_reasons=retention_reasons,
            retention_hold_versions=retention_hold_versions,
            cleanup_candidate_versions=cleanup_candidate_versions,
            warnings=warnings,
            cleanup=cleanup,
        )

    # Surface aligned-tick retention posture (backlog 0135) as read-only
    # diagnostics when the hot training table is in scope. Never fails
    # maintenance: a diagnostic error is recorded rather than raised, and the
    # cleanup itself stays an explicit, dry-run-first operation elsewhere.
    aligned_tick_lifecycle: dict[str, Any] | None = None
    if aligned_tick_lifecycle_diagnostics and "aligned_ticks" in selected:
        try:
            from lancedb_robotics.aligned_tick_lifecycle import diagnose_aligned_ticks

            aligned_tick_lifecycle = diagnose_aligned_ticks(lake)
        except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort.
            aligned_tick_lifecycle = {"status": "failed", "reason": str(exc)}

    # Reconcile the durable scalar predicate index job lifecycle (backlog 0136)
    # without disturbing vector/FTS indexes: poll pending remote builds, expire
    # stale ones, and (opt-in) request the recommended aligned hot-predicate index
    # jobs so default researchers get indexes built for them. Best-effort: never
    # fails maintenance, and reconcile is a no-op returning [] until a job table
    # exists (the read path never creates it).
    scalar_index_job_report: dict[str, Any] | None = None
    if reconcile_index_jobs or request_predicate_index_jobs or apply_predicate_recommendations:
        report: dict[str, Any] = {}
        try:
            if request_predicate_index_jobs and (
                "aligned_ticks" in selected or "aligned_frames" in selected
            ):
                report["requested"] = [
                    result.to_dict()
                    for result in _request_aligned_predicate_index_jobs(
                        lake,
                        include_frames="aligned_frames" in selected,
                    )
                ]
            if apply_predicate_recommendations:
                # 0137: learn hot/selective predicates from persisted training-report
                # telemetry and request index jobs for the safe, high-confidence
                # typed-column ones -- so non-default hot predicates get indexed
                # automatically. Opt-in and best-effort; JSONB promotions stay advisory.
                from lancedb_robotics.predicate_index_telemetry import (
                    apply_index_recommendations,
                    recent_aligned_reports,
                    recommend_from_reports,
                )

                recommendations = recommend_from_reports(
                    recent_aligned_reports(lake, limit=predicate_recommendation_limit)
                )
                report["recommendations_applied"] = apply_index_recommendations(
                    lake, recommendations, min_confidence="high"
                )
            if reconcile_index_jobs:
                report["reconciled"] = _reconcile_scalar_index_jobs(lake)
            scalar_index_job_report = report or None
        except Exception as exc:  # noqa: BLE001 - job lifecycle is best-effort.
            scalar_index_job_report = {"status": "failed", "reason": str(exc)}

    # Validate chunked curation-view membership integrity and reclaim orphaned
    # chunk rows (backlog 0140). Best-effort: never fails maintenance. Migration
    # of legacy inline views is a deliberate layout change and is NOT run here --
    # it stays an explicit `curate migrate-views` operation. Superseded-revision
    # GC is likewise opt-in and only reported here as candidates.
    curation_membership_chunks: dict[str, Any] | None = None
    if curation_chunk_maintenance and "curation_view_membership_chunks" in selected:
        try:
            from lancedb_robotics.curate import (
                compact_view_membership,
                validate_view_membership,
            )

            report = {
                "validation": validate_view_membership(lake).to_params(),
                "compaction": compact_view_membership(
                    lake,
                    include_superseded=False,
                    dry_run=not compact_curation_chunks,
                    created_by=created_by,
                ).to_params(),
            }
            curation_membership_chunks = report
        except Exception as exc:  # noqa: BLE001 - chunk maintenance is best-effort.
            curation_membership_chunks = {"status": "failed", "reason": str(exc)}

    # Validate chunked row-plan storage and reclaim orphaned chunk rows (backlog
    # 0146). The row-plan write path *deliberately* produces orphan chunk rows as
    # its crash-safety mechanism (chunks first, header last), so something has to
    # collect them; leaving it to an operator remembering `curate compact-row-plans`
    # means they accumulate forever. Best-effort: never fails maintenance. The
    # in-flight grace window is honoured, so a compile still streaming chunks is
    # never reclaimed out from under itself.
    curation_row_plan_chunks: dict[str, Any] | None = None
    if curation_chunk_maintenance and "curation_row_plan_chunks" in selected:
        try:
            from lancedb_robotics.curation_row_plans import (
                compact_row_plan_chunks,
                validate_row_plan_storage,
            )

            # Compaction derives the orphan set itself -- and more precisely, since
            # it also separates in-flight writes -- so validation skips its own
            # orphan sweep (a whole-chunk-table scan plus a whole-header scan).
            compaction = compact_row_plan_chunks(
                lake,
                dry_run=not compact_curation_chunks,
            )
            curation_row_plan_chunks = {
                "validation": validate_row_plan_storage(
                    lake, include_orphans=False
                ).to_dict(),
                "compaction": compaction,
            }
        except Exception as exc:  # noqa: BLE001 - chunk maintenance is best-effort.
            curation_row_plan_chunks = {"status": "failed", "reason": str(exc)}

    # Verify curation as-of replay readiness (backlog 0143). The per-table loop
    # above already tagged every snapshot-pinned version (including the three
    # curation replay tables), so this read-only check confirms those pins are
    # protected + readable and surfaces any pruned/unreadable/unprotected replay
    # pin. Best-effort: never fails maintenance.
    curation_replay_retention_report: dict[str, Any] | None = None
    _curation_replay_tables = (
        "curation_views",
        "curation_view_membership_chunks",
        "curation_memberships",
    )
    if curation_replay_retention and any(
        table in selected for table in _curation_replay_tables
    ):
        try:
            from lancedb_robotics.curation_replay_retention import (
                curation_replay_readiness,
            )

            curation_replay_retention_report = curation_replay_readiness(lake).to_dict()
        except Exception as exc:  # noqa: BLE001 - readiness is best-effort.
            curation_replay_retention_report = {"status": "failed", "reason": str(exc)}

    # Collapse physical duplicate rows in the published-view catalog and repair
    # stale latest-view pointers (backlog 0507). Concurrent identical publishes
    # land benign same-key duplicates by design (reads dedupe); something has to
    # reclaim them, and leaving it to an operator remembering `train view
    # compact-catalog` means they accumulate forever. Best-effort for ordinary
    # failures -- but the compaction's data-loss postcondition
    # (ViewCatalogCompactionError: a key lost its last surviving row) must stay
    # loud, never a report line inside a successful run (BUG-04 rule). Note this
    # runs after the per-table compact -> index -> prune loop, so tier-2
    # re-added fragments stay unindexed and delete debris persists until the
    # NEXT maintenance cycle (duplicate-key volume only).
    lerobot_view_catalog_report: dict[str, Any] | None = None
    if lerobot_view_catalog_compaction and "lerobot_views" in selected:
        from lancedb_robotics.lerobot_facade.view_lifecycle import (
            ViewCatalogCompactionError,
            compact_view_catalog,
        )

        try:
            lerobot_view_catalog_report = compact_view_catalog(lake).to_params()
        except ViewCatalogCompactionError as exc:
            raise MaintenanceError(
                f"published-view catalog compaction postcondition failed: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - other failures are best-effort.
            lerobot_view_catalog_report = {"status": "failed", "reason": str(exc)}

    # Published-view retention (backlog 0508). Report-only by default: the
    # policy plan (age + retain-N-newest-per-repo candidates) is computed and
    # reported every run, but deleting candidates requires the explicit
    # `lerobot_view_retention_apply` opt-in -- published views are
    # reproducibility contracts (0111 activation posture). Orphan file-row
    # reconciliation (headerless rows a crashed publish/retire left) DOES apply:
    # the rows are invisible to every reader, and the grace window keeps
    # in-flight publishes safe. Best-effort: never fails maintenance.
    lerobot_view_retention_report: dict[str, Any] | None = None
    if lerobot_view_retention and "lerobot_views" in selected:
        try:
            from lancedb_robotics.lerobot_facade.view_retention import (
                DEFAULT_RETAIN_LATEST_PER_REPO,
                DEFAULT_VIEW_RETENTION_AGE,
                apply_view_retention,
                reconcile_orphan_view_files,
            )

            lerobot_view_retention_report = {
                "policy": apply_view_retention(
                    lake,
                    older_than=(
                        lerobot_view_retention_older_than
                        if lerobot_view_retention_older_than is not None
                        else DEFAULT_VIEW_RETENTION_AGE
                    ),
                    retain_latest_per_repo=(
                        lerobot_view_retain_latest_per_repo
                        if lerobot_view_retain_latest_per_repo is not None
                        else DEFAULT_RETAIN_LATEST_PER_REPO
                    ),
                    dry_run=not lerobot_view_retention_apply,
                ).to_params(),
                "applied": bool(lerobot_view_retention_apply),
                "orphan_file_rows": reconcile_orphan_view_files(lake).to_params(),
            }
        except Exception as exc:  # noqa: BLE001 - retention reporting is best-effort.
            lerobot_view_retention_report = {"status": "failed", "reason": str(exc)}

    # Published-view readiness (backlog 0508, 0143 shape). Runs AFTER the
    # per-table loop above tagged every merged pin, so freshly-protected view
    # pins report `protected` in the same run. Read-only and best-effort.
    lerobot_view_readiness_report: dict[str, Any] | None = None
    if lerobot_view_readiness and "lerobot_views" in selected:
        try:
            from lancedb_robotics.lerobot_facade.view_retention import view_readiness

            lerobot_view_readiness_report = view_readiness(lake).to_dict()
        except Exception as exc:  # noqa: BLE001 - readiness is best-effort.
            lerobot_view_readiness_report = {"status": "failed", "reason": str(exc)}

    finished = datetime.now(UTC)
    transform_id = "tfm-maintenance-" + _digest(
        {
            "started_at": started.isoformat(),
            "tables": selected,
            "versions": input_versions,
        }
    )
    params = {
        "tables": {
            table: report.to_params() for table, report in sorted(table_reports.items())
        },
        "compact": compact,
        "refresh_indexes": refresh_indexes,
        "protect_lineage": protect_lineage,
        "refresh_lineage": refresh_lineage,
        "cleanup_older_than_seconds": (
            cleanup_older_than.total_seconds() if cleanup_older_than is not None else None
        ),
        "retain_versions": retain_versions,
        "delete_unverified": delete_unverified,
        "require_recent_audit": require_recent_audit,
        "audit_max_age_seconds": (
            audit_max_age.total_seconds() if audit_max_age is not None else None
        ),
        "required_audit_report": required_audit_report,
        "lerobot_checkpoint_retention": lerobot_retention_report,
        "aligned_tick_lifecycle": aligned_tick_lifecycle,
        "scalar_index_jobs": scalar_index_job_report,
        "curation_membership_chunks": curation_membership_chunks,
        "curation_row_plan_chunks": curation_row_plan_chunks,
        "curation_replay_retention": curation_replay_retention_report,
        "lerobot_view_catalog": lerobot_view_catalog_report,
        "lerobot_view_retention": lerobot_view_retention_report,
        "lerobot_view_readiness": lerobot_view_readiness_report,
    }
    transform_row = {
        "transform_id": transform_id,
        "kind": MAINTENANCE_TRANSFORM_KIND,
        "input_uris": [],
        "input_table_versions": input_versions,
        "output_tables": list(selected),
        "params": json.dumps(params, sort_keys=True),
        "status": "completed",
        "started_at": started,
        "finished_at": finished,
        "created_by": created_by,
        "created_at": finished,
    }
    lake.table("transform_runs").add(
        pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA)
    )
    # Emit lineage inline (backlog 0098): record the maintenance run's lineage
    # slice without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)
    return MaintenanceReport(
        lake_uri=lake.uri,
        transform_id=transform_id,
        tables=table_reports,
        required_audit_report=required_audit_report,
        lerobot_checkpoint_retention=lerobot_retention_report,
        aligned_tick_lifecycle=aligned_tick_lifecycle,
        scalar_index_jobs=scalar_index_job_report,
        curation_membership_chunks=curation_membership_chunks,
        curation_row_plan_chunks=curation_row_plan_chunks,
        curation_replay_retention=curation_replay_retention_report,
        lerobot_view_catalog=lerobot_view_catalog_report,
        lerobot_view_retention=lerobot_view_retention_report,
        lerobot_view_readiness=lerobot_view_readiness_report,
    )
