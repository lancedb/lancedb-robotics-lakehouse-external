"""External experiment-tracker manifest sync (backlog 0101).

Backlog 0062 keeps LanceDB Robotics training/evaluation manifests canonical while
allowing external MLflow/W&B IDs as references. This module lets a team that
already operates an experiment tracker import historical runs into the canonical
tables, export Lance-native manifests back out for round-trip audit, and detect
drift before overwriting, all without making any tracker a hard dependency.

Design (mirrors the 0064 external-lineage split ``lineage`` vs
``lineage_integrations``):

- The interchange format is a plain-JSON **manifest bundle**: three lists of
  dicts (``training_runs`` / ``model_artifacts`` / ``evaluation_runs``) each
  keyed by an external idempotency id. The generic-JSON adapter is the format
  itself and needs no optional dependency; MLflow/W&B are optional live-fetch
  adapters that degrade with an actionable install hint (reusing
  :func:`lineage_integrations.require_integration_adapter`).
- **Idempotency** comes from a deterministic canonical id derived from
  ``(source, external_id)`` plus a delete-by-id upsert, so re-importing the same
  external run never duplicates a canonical row.
- **Drift** is detected by a content digest over the semantic fields, stored on
  the row's ``manifest_digest`` column. A re-import whose digest differs from the
  stored one is a conflict; the ``conflict`` policy decides the resolution
  (external-wins / lake-wins / append-superseding). A dry run and the dedicated
  :func:`drift_report` report conflicts and write nothing.
- Every actual write records one ``transform_runs`` row (kind ``tracker-import`` /
  ``tracker-export``) with status/error and emits its lineage slice inline (0098),
  so the sync itself is provenance-tracked. Imported manifest rows land in the
  canonical tables, so the run-manifest lineage graph (trained-on / produced-model
  / evaluated-model edges) reconciles on the next ``refresh_graph()`` exactly as
  for natively recorded manifests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.lineage_integrations import (
    LineageIntegrationError,
    require_integration_adapter,
)
from lancedb_robotics.materialization import normalize_table_versions
from lancedb_robotics.run_manifests import (
    EvaluationRunManifest,
    ModelArtifactManifest,
    TrainingRunManifest,
    sync_evaluation_run_metrics,
)
from lancedb_robotics.schemas import (
    EVALUATION_RUNS_SCHEMA,
    MODEL_ARTIFACTS_SCHEMA,
    TRAINING_RUNS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
)

# Sources handled without any optional dependency: the bundle IS the format.
_LOCAL_SOURCES = {"generic", "json", "lance", "lancedb", "lancedb-robotics", "file"}

# Conflict policies for a drifted external run.
CONFLICT_EXTERNAL_WINS = "external-wins"
CONFLICT_LAKE_WINS = "lake-wins"
CONFLICT_APPEND_SUPERSEDING = "append-superseding"
_CONFLICT_POLICIES = (
    CONFLICT_EXTERNAL_WINS,
    CONFLICT_LAKE_WINS,
    CONFLICT_APPEND_SUPERSEDING,
)

# Sync outcomes for one manifest row.
ACTION_CREATED = "created"
ACTION_UPDATED = "updated"
ACTION_UNCHANGED = "unchanged"
ACTION_SKIPPED = "skipped"
ACTION_SUPERSEDED = "superseded"

# Reserved external_refs keys this module writes onto imported rows.
_REF_SOURCE = "tracker_source"
_REF_EXTERNAL_ID = "tracker_external_id"
_REF_DIGEST = "tracker_manifest_digest"
_REF_SUPERSEDES = "tracker_supersedes"

_BUNDLE_SECTIONS = ("training_runs", "model_artifacts", "evaluation_runs")


class TrackerSyncError(Exception):
    """Raised when an experiment-tracker manifest sync cannot be completed."""


# ---------------------------------------------------------------------------
# Adapter availability (optional MLflow/W&B, generic JSON always available).
# ---------------------------------------------------------------------------


def require_tracker_adapter(source: str) -> dict[str, Any]:
    """Validate the adapter for ``source`` is usable; raise an install hint if not.

    Generic/JSON/Lance sources need no optional dependency. MLflow/W&B (and any
    other tracker) route through :func:`require_integration_adapter`, so an
    uninstalled client raises with the extra name to install.
    """

    normalized = str(source or "").strip().lower()
    if not normalized:
        raise TrackerSyncError("tracker source is required")
    if normalized in _LOCAL_SOURCES:
        return {"source": normalized, "module": None, "optional_extra": None, "native": True}
    try:
        info = require_integration_adapter(normalized)
    except LineageIntegrationError as exc:
        extra = normalized
        raise TrackerSyncError(
            f"{exc}; install it to sync {source!r} runs "
            f"(e.g. `pip install 'lancedb-robotics[{extra}]'`) or export a JSON "
            "bundle from the tracker and import it with source='generic'"
        ) from exc
    return {
        "source": normalized,
        "module": info["module"],
        "optional_extra": info["optional_extra"],
        "native": False,
    }


def load_tracker_bundle(source: str, **_options: Any) -> dict[str, Any]:
    """Fetch a manifest bundle directly from a live tracker (requires its client).

    The optional client is required first (an uninstalled tracker degrades with
    an install hint). Live fetch itself is not yet implemented in this build; the
    supported path today is to export a JSON bundle from the tracker and import
    it with ``source='generic'`` / ``--from`` (follow-on: FENG live-fetch task).
    """

    info = require_tracker_adapter(source)
    if info["native"]:
        raise TrackerSyncError(
            f"source {source!r} has no live tracker to fetch from; provide a JSON "
            "bundle (dict or --from PATH) instead"
        )
    raise TrackerSyncError(
        f"live fetch from {source!r} is not implemented in this build; export a "
        f"JSON bundle from {source!r} and import it with --from PATH (the "
        f"{info['optional_extra']!r} client is installed and the interface is ready)"
    )


# ---------------------------------------------------------------------------
# Public report shapes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestSyncEntry:
    """One manifest row's planned or applied sync outcome."""

    kind: str
    manifest_id: str
    external_id: str
    action: str
    drift: bool
    external_digest: str
    previous_digest: str | None = None
    superseded_by: str | None = None
    changed_fields: tuple[str, ...] = ()

    @property
    def written(self) -> bool:
        return self.action in (ACTION_CREATED, ACTION_UPDATED, ACTION_SUPERSEDED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "manifest_id": self.manifest_id,
            "external_id": self.external_id,
            "action": self.action,
            "drift": self.drift,
            "external_digest": self.external_digest,
            "previous_digest": self.previous_digest,
            "superseded_by": self.superseded_by,
            "changed_fields": list(self.changed_fields),
        }


@dataclass(frozen=True)
class TrackerImportReport:
    """Result of importing an external manifest bundle into canonical tables."""

    lake_uri: str
    source: str
    conflict: str
    dry_run: bool
    entries: tuple[ManifestSyncEntry, ...]
    transform_id: str | None
    tables_written: tuple[str, ...]

    def _count(self, action: str) -> int:
        return sum(1 for entry in self.entries if entry.action == action)

    @property
    def created(self) -> int:
        return self._count(ACTION_CREATED)

    @property
    def updated(self) -> int:
        return self._count(ACTION_UPDATED)

    @property
    def unchanged(self) -> int:
        return self._count(ACTION_UNCHANGED)

    @property
    def skipped(self) -> int:
        return self._count(ACTION_SKIPPED)

    @property
    def superseded(self) -> int:
        return self._count(ACTION_SUPERSEDED)

    @property
    def conflicts(self) -> tuple[ManifestSyncEntry, ...]:
        return tuple(entry for entry in self.entries if entry.drift)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "source": self.source,
            "conflict": self.conflict,
            "dry_run": self.dry_run,
            "transform_id": self.transform_id,
            "tables_written": list(self.tables_written),
            "counts": {
                "created": self.created,
                "updated": self.updated,
                "unchanged": self.unchanged,
                "skipped": self.skipped,
                "superseded": self.superseded,
                "conflicts": len(self.conflicts),
                "total": len(self.entries),
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class TrackerDriftReport:
    """Read-only report of which external runs conflict with canonical rows."""

    lake_uri: str
    source: str
    entries: tuple[ManifestSyncEntry, ...]

    @property
    def conflicts(self) -> tuple[ManifestSyncEntry, ...]:
        return tuple(entry for entry in self.entries if entry.drift)

    @property
    def new(self) -> tuple[ManifestSyncEntry, ...]:
        return tuple(entry for entry in self.entries if entry.action == ACTION_CREATED)

    @property
    def has_drift(self) -> bool:
        return bool(self.conflicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "source": self.source,
            "has_drift": self.has_drift,
            "counts": {
                "conflicts": len(self.conflicts),
                "new": len(self.new),
                "total": len(self.entries),
            },
            "conflicts": [entry.to_dict() for entry in self.conflicts],
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class TrackerExportReport:
    """Result of exporting Lance-native manifests into a portable bundle."""

    lake_uri: str
    source: str
    bundle: dict[str, Any]
    out_path: str | None
    transform_id: str | None

    def _len(self, section: str) -> int:
        return len(self.bundle.get(section) or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "source": self.source,
            "out_path": self.out_path,
            "transform_id": self.transform_id,
            "counts": {
                "training_runs": self._len("training_runs"),
                "model_artifacts": self._len("model_artifacts"),
                "evaluation_runs": self._len("evaluation_runs"),
            },
            "bundle": self.bundle,
        }


# ---------------------------------------------------------------------------
# Internal plan record.
# ---------------------------------------------------------------------------


@dataclass
class _PlannedWrite:
    entry: ManifestSyncEntry
    table: str
    id_column: str
    row: dict[str, Any] | None  # populated when a write will happen
    training_run_id: str | None = None  # for building parent maps


# ---------------------------------------------------------------------------
# Import.
# ---------------------------------------------------------------------------


def import_manifest_bundle(
    lake: Lake,
    bundle: Mapping[str, Any] | None = None,
    *,
    source: str = "generic",
    conflict: str = CONFLICT_EXTERNAL_WINS,
    dry_run: bool = False,
    require_adapter: bool = True,
    created_by: str = "lancedb-robotics",
    **fetch_options: Any,
) -> TrackerImportReport:
    """Import an external manifest bundle into the canonical manifest tables.

    ``bundle`` is a dict with any of ``training_runs`` / ``model_artifacts`` /
    ``evaluation_runs`` lists (see module docstring). When ``bundle`` is ``None``
    the bundle is fetched from a live tracker (requires the optional client). Import
    is idempotent per external id and applies ``conflict`` to drifted rows. A
    ``dry_run`` computes and reports the plan without writing.
    """

    if conflict not in _CONFLICT_POLICIES:
        raise TrackerSyncError(
            f"unknown conflict policy {conflict!r}; expected one of {list(_CONFLICT_POLICIES)}"
        )
    normalized_source = _normalize_source(source)
    if bundle is None:
        bundle = load_tracker_bundle(normalized_source, **fetch_options)
    elif require_adapter and normalized_source not in _LOCAL_SOURCES:
        # A caller labelling a provided bundle as an external tracker still needs
        # that tracker's client available for round-trip fidelity guarantees.
        require_tracker_adapter(normalized_source)

    planned = _plan_bundle(lake, bundle, source=normalized_source, conflict=conflict)
    entries = tuple(item.entry for item in planned)
    writes = [item for item in planned if item.row is not None]
    tables_written = tuple(sorted({item.table for item in writes}))

    if dry_run or not writes:
        return TrackerImportReport(
            lake_uri=lake.uri,
            source=normalized_source,
            conflict=conflict,
            dry_run=dry_run,
            entries=entries,
            transform_id=None,
            tables_written=() if dry_run else tables_written,
        )

    now = datetime.now(UTC)
    transform_id = _import_transform_id(normalized_source, conflict, writes)
    for item in writes:
        item.row["transform_id"] = transform_id

    try:
        _apply_writes(lake, writes)
        _record_transform(
            lake,
            transform_id=transform_id,
            kind="tracker-import",
            source=normalized_source,
            input_uri=f"tracker://{normalized_source}",
            output_tables=tables_written,
            params={
                "source": normalized_source,
                "conflict": conflict,
                "created": [e.manifest_id for e in entries if e.action == ACTION_CREATED],
                "updated": [e.manifest_id for e in entries if e.action == ACTION_UPDATED],
                "superseded": [e.manifest_id for e in entries if e.action == ACTION_SUPERSEDED],
                "row_ids": [item.entry.manifest_id for item in writes],
                "entries": [e.to_dict() for e in entries],
            },
            status="completed",
            error=None,
            created_by=created_by,
            now=now,
        )
    except Exception as exc:  # noqa: BLE001 - record the failed sync, then surface
        _record_transform(
            lake,
            transform_id=transform_id,
            kind="tracker-import",
            source=normalized_source,
            input_uri=f"tracker://{normalized_source}",
            output_tables=tables_written,
            params={"source": normalized_source, "conflict": conflict},
            status="failed",
            error=str(exc),
            created_by=created_by,
            now=datetime.now(UTC),
        )
        raise TrackerSyncError(f"tracker import failed: {exc}") from exc

    if "evaluation_runs" in tables_written:
        # Keep the materialized eval-metric surface (backlog 0100) consistent with
        # the imported evals; indexes are rebuilt lazily by ``train sync-metrics``.
        sync_evaluation_run_metrics(lake, build_indexes=False, created_by=created_by)

    return TrackerImportReport(
        lake_uri=lake.uri,
        source=normalized_source,
        conflict=conflict,
        dry_run=False,
        entries=entries,
        transform_id=transform_id,
        tables_written=tables_written,
    )


def drift_report(
    lake: Lake,
    bundle: Mapping[str, Any],
    *,
    source: str = "generic",
) -> TrackerDriftReport:
    """Report which external runs in ``bundle`` conflict with canonical rows.

    Read-only: it plans the import with the external-wins policy (which flags every
    changed row) and returns the outcome without writing anything.
    """

    normalized_source = _normalize_source(source)
    planned = _plan_bundle(
        lake, bundle, source=normalized_source, conflict=CONFLICT_EXTERNAL_WINS
    )
    return TrackerDriftReport(
        lake_uri=lake.uri,
        source=normalized_source,
        entries=tuple(item.entry for item in planned),
    )


# ---------------------------------------------------------------------------
# Export.
# ---------------------------------------------------------------------------


def export_manifest_bundle(
    lake: Lake,
    *,
    source: str = "generic",
    training_run_ids: Sequence[str] | None = None,
    include_checkpoints: bool = True,
    include_evaluations: bool = True,
    out_path: str | Path | None = None,
    created_by: str = "lancedb-robotics",
) -> TrackerExportReport:
    """Export canonical manifests into a portable bundle for an external tracker.

    Each exported row carries its full reproducibility contract -- dataset snapshot
    id, pinned table versions, code ref, params, environment, and (for checkpoints)
    the model artifact checksum plus the manifest digest -- so the bundle is a
    round-trip audit artifact. ``training_run_ids`` scopes the export to specific
    runs (and their checkpoints/evals). Writing ``out_path`` records a
    ``tracker-export`` transform.
    """

    normalized_source = _normalize_source(source)
    wanted_runs = {str(r) for r in training_run_ids} if training_run_ids is not None else None

    training_rows = _scan(lake, "training_runs")
    model_rows = _scan(lake, "model_artifacts") if include_checkpoints else []
    eval_rows = _scan(lake, "evaluation_runs") if include_evaluations else []

    if wanted_runs is not None:
        training_rows = [r for r in training_rows if str(r.get("training_run_id")) in wanted_runs]
        model_rows = [r for r in model_rows if str(r.get("training_run_id")) in wanted_runs]
        kept_models = {str(r.get("model_artifact_id")) for r in model_rows}
        eval_rows = [
            r
            for r in eval_rows
            if str(r.get("training_run_id")) in wanted_runs
            or str(r.get("model_artifact_id")) in kept_models
        ]

    bundle: dict[str, Any] = {
        "source": normalized_source,
        "lake_uri": lake.uri,
        "training_runs": [_export_training_run(row) for row in _sorted(training_rows, "training_run_id")],
        "model_artifacts": [_export_model_artifact(row) for row in _sorted(model_rows, "model_artifact_id")],
        "evaluation_runs": [_export_evaluation_run(row) for row in _sorted(eval_rows, "eval_run_id")],
    }

    transform_id: str | None = None
    resolved_out: str | None = None
    if out_path is not None:
        resolved_out = str(out_path)
        Path(resolved_out).parent.mkdir(parents=True, exist_ok=True)
        Path(resolved_out).write_text(json.dumps(bundle, indent=2, sort_keys=True, default=str))
        now = datetime.now(UTC)
        transform_id = _export_transform_id(normalized_source, bundle, resolved_out)
        _record_transform(
            lake,
            transform_id=transform_id,
            kind="tracker-export",
            source=normalized_source,
            input_uri=f"{lake.uri}#training_runs",
            output_tables=(),
            params={
                "source": normalized_source,
                "out_path": resolved_out,
                "training_run_ids": sorted(wanted_runs) if wanted_runs is not None else None,
                "counts": {section: len(bundle.get(section) or []) for section in _BUNDLE_SECTIONS},
            },
            status="completed",
            error=None,
            created_by=created_by,
            now=now,
        )

    return TrackerExportReport(
        lake_uri=lake.uri,
        source=normalized_source,
        bundle=bundle,
        out_path=resolved_out,
        transform_id=transform_id,
    )


def load_bundle_file(path: str | Path) -> dict[str, Any]:
    """Load a manifest bundle from a JSON file, raising an actionable error."""

    try:
        raw = Path(path).read_text()
    except OSError as exc:
        raise TrackerSyncError(f"cannot read manifest bundle {path!r}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrackerSyncError(f"manifest bundle {path!r} is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise TrackerSyncError(f"manifest bundle {path!r} must be a JSON object")
    return dict(parsed)


# ---------------------------------------------------------------------------
# Planning (pure): build rows + classify actions without writing.
# ---------------------------------------------------------------------------


def _plan_bundle(
    lake: Lake,
    bundle: Mapping[str, Any],
    *,
    source: str,
    conflict: str,
) -> list[_PlannedWrite]:
    if not isinstance(bundle, Mapping):
        raise TrackerSyncError("manifest bundle must be a mapping/object")

    existing = {
        "training_runs": _digest_index(lake, "training_runs", "training_run_id"),
        "model_artifacts": _digest_index(lake, "model_artifacts", "model_artifact_id"),
        "evaluation_runs": _digest_index(lake, "evaluation_runs", "eval_run_id"),
    }
    # Resolve eval training-run parents from planned + existing artifacts.
    artifact_to_training: dict[str, str] = {
        str(mid): str(tid)
        for mid, tid in _scan_pairs(lake, "model_artifacts", "model_artifact_id", "training_run_id")
    }

    planned: list[_PlannedWrite] = []

    for item in _section(bundle, "training_runs"):
        planned.append(
            _plan_training_run(item, source=source, conflict=conflict, existing=existing["training_runs"])
        )

    for item in _section(bundle, "model_artifacts"):
        write = _plan_model_artifact(
            item, source=source, conflict=conflict, existing=existing["model_artifacts"]
        )
        if write.training_run_id and write.row is not None:
            artifact_to_training[write.entry.manifest_id] = write.training_run_id
        planned.append(write)

    for item in _section(bundle, "evaluation_runs"):
        planned.append(
            _plan_evaluation_run(
                item,
                source=source,
                conflict=conflict,
                existing=existing["evaluation_runs"],
                artifact_to_training=artifact_to_training,
            )
        )
    return planned


def _plan_training_run(
    item: Mapping[str, Any],
    *,
    source: str,
    conflict: str,
    existing: Mapping[str, str],
) -> _PlannedWrite:
    external_id = _external_id(item, ("external_run_id", "external_id", "run_id", "training_run_id"))
    base_id = _canonical_id(item.get("training_run_id"), "train", source, external_id)
    table_versions = tuple(normalize_table_versions(item.get("table_versions") or ()))
    payload = {
        "dataset_id": _text(item.get("dataset_id")),
        "snapshot_name": _text(item.get("snapshot_name")),
        "snapshot_tag": _text(item.get("snapshot_tag")),
        "table_versions": [dict(v) for v in table_versions],
        "row_plan_id": item.get("row_plan_id"),
        "epoch_plan_id": item.get("epoch_plan_id"),
        "projection_manifest_ids": _string_list(item.get("projection_manifest_ids")),
        "code_ref": item.get("code_ref"),
        "package_versions": _obj(item.get("package_versions")),
        "environment": _obj(item.get("environment")),
        "hardware": _obj(item.get("hardware")),
        "runtime": _obj(item.get("runtime")),
        "hyperparameters": _obj(item.get("hyperparameters")),
        "random_seeds": _obj(item.get("random_seeds")),
        "split_policy": _obj(item.get("split_policy")),
        "status": _text(item.get("status")) or "completed",
    }
    external_digest = _digest(payload)
    manifest_id, action, drift, previous, superseded_by, changed = _classify(
        base_id, external_digest, existing, conflict
    )
    entry = ManifestSyncEntry(
        kind="training-run",
        manifest_id=manifest_id,
        external_id=external_id,
        action=action,
        drift=drift,
        external_digest=external_digest,
        previous_digest=previous,
        superseded_by=superseded_by,
        changed_fields=changed,
    )
    row = None
    if action in (ACTION_CREATED, ACTION_UPDATED, ACTION_SUPERSEDED):
        refs = _external_refs(item, source, external_id, external_digest, superseded_by)
        manifest = TrainingRunManifest(
            lake_uri="",
            training_run_id=manifest_id,
            dataset_id=payload["dataset_id"],
            snapshot_name=payload["snapshot_name"],
            snapshot_tag=payload["snapshot_tag"],
            table_versions=table_versions,
            row_plan_id=payload["row_plan_id"],
            epoch_plan_id=payload["epoch_plan_id"],
            projection_manifest_ids=tuple(payload["projection_manifest_ids"]),
            code_ref=payload["code_ref"],
            package_versions=payload["package_versions"],
            environment=payload["environment"],
            hardware=payload["hardware"],
            runtime=payload["runtime"],
            hyperparameters=payload["hyperparameters"],
            random_seeds=payload["random_seeds"],
            split_policy=payload["split_policy"],
            external_refs=refs,
            status=payload["status"],
            manifest_digest=external_digest,
            transform_id="",
            created_by="lancedb-robotics",
        )
        row = manifest.as_row(datetime.now(UTC))
    return _PlannedWrite(entry=entry, table="training_runs", id_column="training_run_id", row=row)


def _plan_model_artifact(
    item: Mapping[str, Any],
    *,
    source: str,
    conflict: str,
    existing: Mapping[str, str],
) -> _PlannedWrite:
    external_id = _external_id(
        item, ("external_artifact_id", "external_id", "model_artifact_id", "external_run_id")
    )
    base_id = _canonical_id(item.get("model_artifact_id"), "model", source, external_id)
    training_run_id = _text(item.get("training_run_id"))
    if not training_run_id:
        parent = _text(item.get("external_run_id"))
        if parent:
            training_run_id = _canonical_id(None, "train", source, parent)
    payload = {
        "training_run_id": training_run_id,
        "artifact_uri": item.get("artifact_uri"),
        "checksum": item.get("checksum"),
        "aliases": _string_list(item.get("aliases")),
        "framework": item.get("framework"),
        "epoch": _int_or_none(item.get("epoch")),
        "step": _int_or_none(item.get("step")),
        "metrics": _obj(item.get("metrics")),
        "metadata": _obj(item.get("metadata")),
    }
    external_digest = _digest(payload)
    manifest_id, action, drift, previous, superseded_by, changed = _classify(
        base_id, external_digest, existing, conflict
    )
    entry = ManifestSyncEntry(
        kind="model-artifact",
        manifest_id=manifest_id,
        external_id=external_id,
        action=action,
        drift=drift,
        external_digest=external_digest,
        previous_digest=previous,
        superseded_by=superseded_by,
        changed_fields=changed,
    )
    row = None
    if action in (ACTION_CREATED, ACTION_UPDATED, ACTION_SUPERSEDED):
        refs = _external_refs(item, source, external_id, external_digest, superseded_by)
        manifest = ModelArtifactManifest(
            lake_uri="",
            model_artifact_id=manifest_id,
            training_run_id=training_run_id,
            artifact_uri=payload["artifact_uri"],
            checksum=payload["checksum"],
            aliases=tuple(payload["aliases"]),
            framework=payload["framework"],
            epoch=payload["epoch"],
            step=payload["step"],
            metrics=payload["metrics"],
            metadata=payload["metadata"],
            external_refs=refs,
            manifest_digest=external_digest,
            transform_id="",
            created_by="lancedb-robotics",
        )
        row = manifest.as_row(datetime.now(UTC))
    return _PlannedWrite(
        entry=entry,
        table="model_artifacts",
        id_column="model_artifact_id",
        row=row,
        training_run_id=training_run_id,
    )


def _plan_evaluation_run(
    item: Mapping[str, Any],
    *,
    source: str,
    conflict: str,
    existing: Mapping[str, str],
    artifact_to_training: Mapping[str, str],
) -> _PlannedWrite:
    external_id = _external_id(
        item, ("external_eval_id", "external_id", "eval_run_id", "external_run_id")
    )
    base_id = _canonical_id(item.get("eval_run_id"), "eval", source, external_id)
    model_artifact_id = _text(item.get("model_artifact_id"))
    if not model_artifact_id:
        parent = _text(item.get("external_artifact_id"))
        if parent:
            model_artifact_id = _canonical_id(None, "model", source, parent)
    training_run_id = _text(item.get("training_run_id")) or artifact_to_training.get(
        model_artifact_id, ""
    )
    if not training_run_id:
        parent_run = _text(item.get("external_run_id"))
        if parent_run:
            training_run_id = _canonical_id(None, "train", source, parent_run)
    table_versions = tuple(normalize_table_versions(item.get("table_versions") or ()))
    payload = {
        "model_artifact_id": model_artifact_id,
        "training_run_id": training_run_id,
        "dataset_id": _text(item.get("dataset_id")),
        "snapshot_name": _text(item.get("snapshot_name")),
        "snapshot_tag": _text(item.get("snapshot_tag")),
        "table_versions": [dict(v) for v in table_versions],
        "metrics": _obj(item.get("metrics")),
        "slice_metrics": _obj(item.get("slice_metrics")),
        "failure_outputs": _obj(item.get("failure_outputs")),
        "code_ref": item.get("code_ref"),
        "package_versions": _obj(item.get("package_versions")),
        "environment": _obj(item.get("environment")),
        "hardware": _obj(item.get("hardware")),
        "runtime": _obj(item.get("runtime")),
        "status": _text(item.get("status")) or "completed",
    }
    external_digest = _digest(payload)
    manifest_id, action, drift, previous, superseded_by, changed = _classify(
        base_id, external_digest, existing, conflict
    )
    entry = ManifestSyncEntry(
        kind="evaluation-run",
        manifest_id=manifest_id,
        external_id=external_id,
        action=action,
        drift=drift,
        external_digest=external_digest,
        previous_digest=previous,
        superseded_by=superseded_by,
        changed_fields=changed,
    )
    row = None
    if action in (ACTION_CREATED, ACTION_UPDATED, ACTION_SUPERSEDED):
        refs = _external_refs(item, source, external_id, external_digest, superseded_by)
        manifest = EvaluationRunManifest(
            lake_uri="",
            eval_run_id=manifest_id,
            model_artifact_id=model_artifact_id,
            training_run_id=training_run_id,
            dataset_id=payload["dataset_id"],
            snapshot_name=payload["snapshot_name"],
            snapshot_tag=payload["snapshot_tag"],
            table_versions=table_versions,
            metrics=payload["metrics"],
            slice_metrics=payload["slice_metrics"],
            failure_outputs=payload["failure_outputs"],
            code_ref=payload["code_ref"],
            package_versions=payload["package_versions"],
            environment=payload["environment"],
            hardware=payload["hardware"],
            runtime=payload["runtime"],
            external_refs=refs,
            status=payload["status"],
            manifest_digest=external_digest,
            transform_id="",
            created_by="lancedb-robotics",
        )
        row = manifest.as_row(datetime.now(UTC))
    return _PlannedWrite(entry=entry, table="evaluation_runs", id_column="eval_run_id", row=row)


def _classify(
    base_id: str,
    external_digest: str,
    existing: Mapping[str, str],
    conflict: str,
) -> tuple[str, str, bool, str | None, str | None, tuple[str, ...]]:
    """Return (manifest_id, action, drift, previous_digest, superseded_by, changed).

    ``changed`` is a coarse ("manifest_digest",) marker when the content differs;
    field-level diffing is a follow-on. The superseding id is derived from the new
    digest so re-importing the same superseding content stays idempotent.
    """

    previous = existing.get(base_id)
    if previous is None:
        return base_id, ACTION_CREATED, False, None, None, ()
    if previous == external_digest:
        return base_id, ACTION_UNCHANGED, False, previous, None, ()
    # Drift: the external content changed relative to the canonical row.
    if conflict == CONFLICT_LAKE_WINS:
        return base_id, ACTION_SKIPPED, True, previous, None, ("manifest_digest",)
    if conflict == CONFLICT_APPEND_SUPERSEDING:
        superseding_id = f"{base_id}-{external_digest[:8]}"
        # Idempotent: if the superseding row already exists unchanged, no-op.
        if existing.get(superseding_id) == external_digest:
            return superseding_id, ACTION_UNCHANGED, True, external_digest, base_id, ()
        return superseding_id, ACTION_SUPERSEDED, True, previous, base_id, ("manifest_digest",)
    return base_id, ACTION_UPDATED, True, previous, None, ("manifest_digest",)


# ---------------------------------------------------------------------------
# Write mechanics + transform recording.
# ---------------------------------------------------------------------------

_SCHEMA_BY_TABLE = {
    "training_runs": TRAINING_RUNS_SCHEMA,
    "model_artifacts": MODEL_ARTIFACTS_SCHEMA,
    "evaluation_runs": EVALUATION_RUNS_SCHEMA,
}
_ID_COLUMN_BY_TABLE = {
    "training_runs": "training_run_id",
    "model_artifacts": "model_artifact_id",
    "evaluation_runs": "eval_run_id",
}


def _apply_writes(lake: Lake, writes: Sequence[_PlannedWrite]) -> None:
    by_table: dict[str, list[dict[str, Any]]] = {}
    for item in writes:
        if item.row is not None:
            by_table.setdefault(item.table, []).append(item.row)
    for table, rows in by_table.items():
        _upsert_rows(lake, table, _ID_COLUMN_BY_TABLE[table], rows, _SCHEMA_BY_TABLE[table])


def _upsert_rows(
    lake: Lake,
    table_name: str,
    id_column: str,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> None:
    ids = [str(row[id_column]) for row in rows]
    lake.table(table_name).delete(
        f"{id_column} IN ({', '.join(_sql_literal(value) for value in ids)})"
    )
    lake.table(table_name).add(pa.Table.from_pylist(rows, schema=schema))


def _record_transform(
    lake: Lake,
    *,
    transform_id: str,
    kind: str,
    source: str,
    input_uri: str,
    output_tables: Sequence[str],
    params: Mapping[str, Any],
    status: str,
    error: str | None,
    created_by: str,
    now: datetime,
) -> None:
    transform_row = {
        "transform_id": transform_id,
        "kind": kind,
        "source_id": source,
        "input_uris": [input_uri],
        "input_table_versions": [],
        "output_tables": list(output_tables),
        "params": json.dumps(_jsonable(params), sort_keys=True, separators=(",", ":"), default=str),
        "status": status,
        "error": error,
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    _upsert_rows(lake, "transform_runs", "transform_id", [transform_row], TRANSFORM_RUNS_SCHEMA)
    # Inline lineage emission (backlog 0098); refresh_graph() remains the safety net.
    emit_transform_lineage(lake, transform_row)


# ---------------------------------------------------------------------------
# Export projections.
# ---------------------------------------------------------------------------


def _export_training_run(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "external_run_id": row.get("training_run_id"),
        "training_run_id": row.get("training_run_id"),
        "dataset_id": row.get("dataset_id"),
        "snapshot_name": row.get("snapshot_name"),
        "snapshot_tag": row.get("snapshot_tag"),
        "table_versions": [_version(v) for v in row.get("table_versions") or []],
        "row_plan_id": row.get("row_plan_id"),
        "epoch_plan_id": row.get("epoch_plan_id"),
        "projection_manifest_ids": list(row.get("projection_manifest_ids") or []),
        "code_ref": row.get("code_ref"),
        "package_versions": _json_load(row.get("package_versions_json")),
        "environment": _json_load(row.get("environment_json")),
        "hardware": _json_load(row.get("hardware_json")),
        "runtime": _json_load(row.get("runtime_json")),
        "hyperparameters": _json_load(row.get("hyperparameters_json")),
        "random_seeds": _json_load(row.get("random_seeds_json")),
        "split_policy": _json_load(row.get("split_policy_json")),
        "status": row.get("status"),
        "manifest_digest": row.get("manifest_digest"),
        "external_refs": _kv_map(row),
    }


def _export_model_artifact(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "external_artifact_id": row.get("model_artifact_id"),
        "model_artifact_id": row.get("model_artifact_id"),
        "external_run_id": row.get("training_run_id"),
        "training_run_id": row.get("training_run_id"),
        "artifact_uri": row.get("artifact_uri"),
        "checksum": row.get("checksum"),
        "aliases": list(row.get("aliases") or []),
        "framework": row.get("framework"),
        "epoch": row.get("epoch"),
        "step": row.get("step"),
        "metrics": _json_load(row.get("metrics_json")),
        "metadata": _kv_map(row, column="metadata"),
        "manifest_digest": row.get("manifest_digest"),
        "external_refs": _kv_map(row),
    }


def _export_evaluation_run(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "external_eval_id": row.get("eval_run_id"),
        "eval_run_id": row.get("eval_run_id"),
        "external_artifact_id": row.get("model_artifact_id"),
        "model_artifact_id": row.get("model_artifact_id"),
        "training_run_id": row.get("training_run_id"),
        "dataset_id": row.get("dataset_id"),
        "snapshot_name": row.get("snapshot_name"),
        "snapshot_tag": row.get("snapshot_tag"),
        "table_versions": [_version(v) for v in row.get("table_versions") or []],
        "metrics": _json_load(row.get("metrics_json")),
        "slice_metrics": _json_load(row.get("slice_metrics_json")),
        "failure_outputs": _json_load(row.get("failure_outputs_json")),
        "code_ref": row.get("code_ref"),
        "package_versions": _json_load(row.get("package_versions_json")),
        "environment": _json_load(row.get("environment_json")),
        "hardware": _json_load(row.get("hardware_json")),
        "runtime": _json_load(row.get("runtime_json")),
        "status": row.get("status"),
        "manifest_digest": row.get("manifest_digest"),
        "external_refs": _kv_map(row),
    }


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _normalize_source(source: str) -> str:
    normalized = str(source or "").strip().lower()
    if not normalized:
        raise TrackerSyncError("tracker source is required")
    return normalized


def _section(bundle: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    value = bundle.get(name) or []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TrackerSyncError(f"bundle section {name!r} must be a list of objects")
    rows: list[Mapping[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise TrackerSyncError(f"bundle section {name!r} entries must be objects")
        rows.append(entry)
    return rows


def _external_id(item: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value):
            return str(value)
    raise TrackerSyncError(
        f"bundle entry is missing an external id; provide one of {list(keys)}: "
        f"{sorted(item)[:8]}"
    )


def _canonical_id(explicit: Any, prefix: str, source: str, external_id: str) -> str:
    if explicit is not None and str(explicit):
        return str(explicit)
    return f"{prefix}-{source}-{_digest([source, prefix, external_id])[:16]}"


def _external_refs(
    item: Mapping[str, Any],
    source: str,
    external_id: str,
    external_digest: str,
    superseded_by: str | None,
) -> dict[str, Any]:
    refs = {str(k): _ref_value(v) for k, v in _obj(item.get("external_refs")).items() if v is not None}
    refs[f"{source}_run_id"] = external_id
    refs[_REF_SOURCE] = source
    refs[_REF_EXTERNAL_ID] = external_id
    refs[_REF_DIGEST] = external_digest
    if superseded_by:
        refs[_REF_SUPERSEDES] = superseded_by
    return refs


def _ref_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)


def _digest_index(lake: Lake, table: str, id_column: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for row in _scan(lake, table):
        manifest_id = row.get(id_column)
        if manifest_id:
            index[str(manifest_id)] = str(row.get("manifest_digest") or "")
    return index


def _scan(lake: Lake, table: str) -> list[dict[str, Any]]:
    if table not in lake.table_names():
        return []
    return lake.table(table).to_arrow().to_pylist()


def _scan_pairs(lake: Lake, table: str, a: str, b: str) -> list[tuple[Any, Any]]:
    return [(row.get(a), row.get(b)) for row in _scan(lake, table)]


def _sorted(rows: Sequence[dict[str, Any]], id_column: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: str(row.get(id_column) or ""))


def _kv_map(row: Mapping[str, Any], *, column: str = "external_refs") -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in row.get(column) or []:
        if not isinstance(entry, Mapping) or entry.get("key") is None:
            continue
        result[str(entry["key"])] = "" if entry.get("value") is None else str(entry["value"])
    return result


def _version(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "table": value.get("table"),
        "version": value.get("version"),
        "tag": value.get("tag") or "",
    }


def _obj(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TrackerSyncError(f"expected an object or JSON object, got {value!r}: {exc.msg}") from exc
        if isinstance(parsed, Mapping):
            return {str(k): v for k, v in parsed.items()}
    raise TrackerSyncError(f"expected an object, got {type(value).__name__}")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise TrackerSyncError("expected a list of strings, got a string")
    return [str(v) for v in value if v is not None and str(v)]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TrackerSyncError(f"expected an integer, got {value!r}") from exc


def _json_load(value: Any) -> Any:
    if value in (None, ""):
        return {}
    if isinstance(value, (Mapping, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, set)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _digest(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode()).hexdigest()[:20]


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _import_transform_id(source: str, conflict: str, writes: Sequence[_PlannedWrite]) -> str:
    digest = _digest(
        {
            "source": source,
            "conflict": conflict,
            "writes": sorted(
                f"{item.entry.manifest_id}:{item.entry.external_digest}" for item in writes
            ),
        }
    )
    return f"tfm-tracker-import-{digest[:16]}"


def _export_transform_id(source: str, bundle: Mapping[str, Any], out_path: str) -> str:
    digest = _digest(
        {
            "source": source,
            "out_path": out_path,
            "ids": {
                section: sorted(str(e.get("external_run_id") or e.get("external_eval_id") or e.get("external_artifact_id") or "") for e in bundle.get(section) or [])
                for section in _BUNDLE_SECTIONS
            },
        }
    )
    return f"tfm-tracker-export-{digest[:16]}"


class LakeTrackerSync:
    """Convenience namespace exposed as ``lake.tracker``."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def import_bundle(self, bundle: Mapping[str, Any] | None = None, **kwargs: Any) -> TrackerImportReport:
        """Import an external manifest bundle into the canonical manifest tables."""

        return import_manifest_bundle(self._lake, bundle, **kwargs)

    def export_bundle(self, **kwargs: Any) -> TrackerExportReport:
        """Export canonical manifests into a portable, round-trippable bundle."""

        return export_manifest_bundle(self._lake, **kwargs)

    def drift(self, bundle: Mapping[str, Any], **kwargs: Any) -> TrackerDriftReport:
        """Report which external runs conflict with canonical rows (read-only)."""

        return drift_report(self._lake, bundle, **kwargs)

    def require_adapter(self, source: str) -> dict[str, Any]:
        """Validate a tracker adapter is installed (raises an install hint if not)."""

        return require_tracker_adapter(source)
