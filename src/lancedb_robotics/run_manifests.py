"""Training and evaluation run manifests.

These APIs record Lance-native experiment metadata without requiring MLflow,
W&B, or another external tracker. External run IDs stay as optional references;
the canonical source of record is the lake.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.lineage_hooks import (
    apply_lineage_context_to_manifest_fields,
    attach_lineage_context_to_params,
    begin_lineage_execution,
)
from lancedb_robotics.materialization import normalize_table_versions
from lancedb_robotics.schemas import (
    EVALUATION_RUN_METRICS_SCHEMA,
    EVALUATION_RUNS_SCHEMA,
    MODEL_ARTIFACTS_SCHEMA,
    TRAINING_REPORTS_SCHEMA,
    TRAINING_RUNS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
)

EVALUATION_RUN_METRICS_TABLE = "evaluation_run_metrics"
TRAINING_REPORTS_TABLE = "training_reports"
_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 1000
_SCAN_BATCH_SIZE = 2048


class RunManifestError(Exception):
    """Raised when a training/evaluation manifest cannot be recorded."""


@dataclass(frozen=True)
class ExternalReferenceUpdate:
    """Result of attaching external metadata-system refs to a manifest row."""

    lake_uri: str
    table_name: str
    id_column: str
    manifest_id: str
    external_refs: dict[str, str]
    replace: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "table_name": self.table_name,
            "id_column": self.id_column,
            "manifest_id": self.manifest_id,
            "external_refs": dict(self.external_refs),
            "replace": self.replace,
        }


@dataclass(frozen=True)
class TrainingRunManifest:
    """Recorded training run and its exact input snapshot contract."""

    lake_uri: str
    training_run_id: str
    dataset_id: str
    snapshot_name: str
    snapshot_tag: str
    table_versions: tuple[dict[str, Any], ...]
    row_plan_id: str | None
    epoch_plan_id: str | None
    projection_manifest_ids: tuple[str, ...]
    code_ref: str | None
    package_versions: dict[str, Any]
    environment: dict[str, Any]
    hardware: dict[str, Any]
    runtime: dict[str, Any]
    hyperparameters: dict[str, Any]
    random_seeds: dict[str, Any]
    split_policy: dict[str, Any]
    external_refs: dict[str, Any]
    status: str
    manifest_digest: str
    transform_id: str
    created_by: str
    lineage_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "lake_uri": self.lake_uri,
            "training_run_id": self.training_run_id,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "snapshot_tag": self.snapshot_tag,
            "table_versions": [dict(item) for item in self.table_versions],
            "row_plan_id": self.row_plan_id,
            "epoch_plan_id": self.epoch_plan_id,
            "projection_manifest_ids": list(self.projection_manifest_ids),
            "code_ref": self.code_ref,
            "package_versions": dict(self.package_versions),
            "environment": dict(self.environment),
            "hardware": dict(self.hardware),
            "runtime": dict(self.runtime),
            "hyperparameters": dict(self.hyperparameters),
            "random_seeds": dict(self.random_seeds),
            "split_policy": dict(self.split_policy),
            "external_refs": dict(self.external_refs),
            "status": self.status,
            "manifest_digest": self.manifest_digest,
            "transform_id": self.transform_id,
            "created_by": self.created_by,
        }
        if self.lineage_context:
            payload["lineage_context"] = dict(self.lineage_context)
        return payload

    def as_row(self, created_at: datetime) -> dict[str, Any]:
        return {
            "training_run_id": self.training_run_id,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "snapshot_tag": self.snapshot_tag,
            "table_versions": [dict(item) for item in self.table_versions],
            "row_plan_id": self.row_plan_id,
            "epoch_plan_id": self.epoch_plan_id,
            "projection_manifest_ids": list(self.projection_manifest_ids),
            "code_ref": self.code_ref,
            "package_versions_json": _json_dumps(self.package_versions),
            "environment_json": _json_dumps(self.environment),
            "hardware_json": _json_dumps(self.hardware),
            "runtime_json": _json_dumps(self.runtime),
            "hyperparameters_json": _json_dumps(self.hyperparameters),
            "random_seeds_json": _json_dumps(self.random_seeds),
            "split_policy_json": _json_dumps(self.split_policy),
            "external_refs": _kv_items(self.external_refs),
            "status": self.status,
            "manifest_digest": self.manifest_digest,
            "transform_id": self.transform_id,
            "created_by": self.created_by,
            "created_at": created_at,
        }


@dataclass(frozen=True)
class ModelArtifactManifest:
    """Recorded checkpoint/model artifact produced by a training run."""

    lake_uri: str
    model_artifact_id: str
    training_run_id: str
    artifact_uri: str | None
    checksum: str | None
    aliases: tuple[str, ...]
    framework: str | None
    epoch: int | None
    step: int | None
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    external_refs: dict[str, Any]
    manifest_digest: str
    transform_id: str
    created_by: str
    lineage_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "lake_uri": self.lake_uri,
            "model_artifact_id": self.model_artifact_id,
            "training_run_id": self.training_run_id,
            "artifact_uri": self.artifact_uri,
            "checksum": self.checksum,
            "aliases": list(self.aliases),
            "framework": self.framework,
            "epoch": self.epoch,
            "step": self.step,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "external_refs": dict(self.external_refs),
            "manifest_digest": self.manifest_digest,
            "transform_id": self.transform_id,
            "created_by": self.created_by,
        }
        if self.lineage_context:
            payload["lineage_context"] = dict(self.lineage_context)
        return payload

    def as_row(self, created_at: datetime) -> dict[str, Any]:
        return {
            "model_artifact_id": self.model_artifact_id,
            "training_run_id": self.training_run_id,
            "artifact_uri": self.artifact_uri,
            "checksum": self.checksum,
            "aliases": list(self.aliases),
            "framework": self.framework,
            "epoch": self.epoch,
            "step": self.step,
            "metrics_json": _json_dumps(self.metrics),
            "metadata": _kv_items(self.metadata),
            "external_refs": _kv_items(self.external_refs),
            "manifest_digest": self.manifest_digest,
            "transform_id": self.transform_id,
            "created_by": self.created_by,
            "created_at": created_at,
        }


@dataclass(frozen=True)
class EvaluationRunManifest:
    """Recorded evaluation manifest linking a model artifact to eval data."""

    lake_uri: str
    eval_run_id: str
    model_artifact_id: str
    training_run_id: str
    dataset_id: str
    snapshot_name: str
    snapshot_tag: str
    table_versions: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    slice_metrics: dict[str, Any]
    failure_outputs: dict[str, Any]
    code_ref: str | None
    package_versions: dict[str, Any]
    environment: dict[str, Any]
    hardware: dict[str, Any]
    runtime: dict[str, Any]
    external_refs: dict[str, Any]
    status: str
    manifest_digest: str
    transform_id: str
    created_by: str
    lineage_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "lake_uri": self.lake_uri,
            "eval_run_id": self.eval_run_id,
            "model_artifact_id": self.model_artifact_id,
            "training_run_id": self.training_run_id,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "snapshot_tag": self.snapshot_tag,
            "table_versions": [dict(item) for item in self.table_versions],
            "metrics": dict(self.metrics),
            "slice_metrics": dict(self.slice_metrics),
            "failure_outputs": dict(self.failure_outputs),
            "code_ref": self.code_ref,
            "package_versions": dict(self.package_versions),
            "environment": dict(self.environment),
            "hardware": dict(self.hardware),
            "runtime": dict(self.runtime),
            "external_refs": dict(self.external_refs),
            "status": self.status,
            "manifest_digest": self.manifest_digest,
            "transform_id": self.transform_id,
            "created_by": self.created_by,
        }
        if self.lineage_context:
            payload["lineage_context"] = dict(self.lineage_context)
        return payload

    def as_row(self, created_at: datetime) -> dict[str, Any]:
        return {
            "eval_run_id": self.eval_run_id,
            "model_artifact_id": self.model_artifact_id,
            "training_run_id": self.training_run_id,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "snapshot_tag": self.snapshot_tag,
            "table_versions": [dict(item) for item in self.table_versions],
            "metrics_json": _json_dumps(self.metrics),
            "slice_metrics_json": _json_dumps(self.slice_metrics),
            "failure_outputs_json": _json_dumps(self.failure_outputs),
            "code_ref": self.code_ref,
            "package_versions_json": _json_dumps(self.package_versions),
            "environment_json": _json_dumps(self.environment),
            "hardware_json": _json_dumps(self.hardware),
            "runtime_json": _json_dumps(self.runtime),
            "external_refs": _kv_items(self.external_refs),
            "status": self.status,
            "manifest_digest": self.manifest_digest,
            "transform_id": self.transform_id,
            "created_by": self.created_by,
            "created_at": created_at,
        }


class LakeEval:
    """Convenience namespace exposed as ``lake.eval`` and ``lake.evaluation``."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def record_run(
        self,
        snapshot: str | None = None,
        **kwargs: Any,
    ) -> EvaluationRunManifest:
        """Record an evaluation manifest for a model artifact and snapshot."""

        return record_evaluation_run(self._lake, snapshot=snapshot, **kwargs)

    def runs(
        self,
        *,
        model_artifact_id: str | None = None,
        dataset_id: str | None = None,
        snapshot: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return evaluation rows filtered by model artifact and/or snapshot."""

        return evaluation_runs(
            self._lake,
            model_artifact_id=model_artifact_id,
            dataset_id=dataset_id,
            snapshot=snapshot,
        )

    def query(self, **kwargs: Any) -> ManifestQueryResult:
        """Bounded/indexed evaluation query with an execution plan (backlog 0100)."""

        return query_evaluation_runs(self._lake, **kwargs)

    def list(self, **kwargs: Any) -> ManifestPage:
        """Deterministic paged evaluation listing with continuation tokens."""

        return list_evaluation_runs(self._lake, **kwargs)

    def metrics(self, **kwargs: Any) -> EvalMetricLookupResult:
        """Metric-key lookup over the materialized eval-metric surface (or scan)."""

        return query_eval_metrics(self._lake, **kwargs)

    def sync_metrics(self, *, build_indexes: bool = True) -> EvalMetricSyncReport:
        """Rebuild the materialized ``evaluation_run_metrics`` surface + indexes."""

        return sync_evaluation_run_metrics(self._lake, build_indexes=build_indexes)

    def retention_plan(
        self,
        *,
        older_than: timedelta | datetime | None = None,
    ) -> ManifestRetentionPlan:
        """Report which evaluation runs are protected vs safe to expire."""

        return plan_manifest_retention(
            self._lake, kinds=("evaluation-run",), older_than=older_than
        )

    def expire(self, eval_run_id: str, *, force: bool = False) -> dict[str, Any]:
        """Delete an evaluation run (refused if protected unless ``force``)."""

        return delete_manifest(
            self._lake, kind="evaluation-run", manifest_id=eval_run_id, force=force
        )

    def attach_external_refs(
        self,
        eval_run_id: str,
        refs: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> ExternalReferenceUpdate:
        """Attach MLflow/W&B/etc. references to an existing evaluation manifest."""

        return attach_evaluation_external_refs(
            self._lake,
            eval_run_id,
            refs,
            replace=replace,
        )


def record_training_run(
    lake: Lake,
    snapshot: str | None = None,
    *,
    dataset_id: str | None = None,
    dataset: Any | None = None,
    training_run_id: str | None = None,
    row_plan_id: str | None = None,
    epoch_plan_id: str | None = None,
    projection_manifest_ids: Sequence[str] | None = None,
    code_ref: str | None = None,
    package_versions: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    random_seeds: Mapping[str, Any] | None = None,
    split_policy: Mapping[str, Any] | None = None,
    external_refs: Mapping[str, Any] | None = None,
    status: str = "completed",
    created_by: str = "lancedb-robotics",
    lineage_context: Any | None = None,
) -> TrainingRunManifest:
    """Record a training run manifest against a dataset snapshot.

    ``dataset`` may be a ``LanceTrainingDataset``; when present, its manifest
    supplies the snapshot, row-plan, and epoch-plan identities unless explicit
    values override them.
    """

    inferred = _infer_training_dataset(dataset)
    dataset_id = dataset_id or inferred.get("dataset_id")
    snapshot = snapshot or inferred.get("snapshot_name")
    row_plan_id = row_plan_id or inferred.get("row_plan_id")
    epoch_plan_id = epoch_plan_id or inferred.get("epoch_plan_id")

    handle = begin_lineage_execution(
        lineage_context,
        operation="training-run",
        params={"snapshot": snapshot, "dataset_id": dataset_id, "status": status},
    )
    context = handle.finish(status=status)
    snapshot_row = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)
    table_versions = _snapshot_table_versions(snapshot_row)
    split = dict(split_policy) if split_policy is not None else _json_object(snapshot_row.get("split"))
    projections = _string_tuple(projection_manifest_ids, "projection_manifest_ids")
    fields = apply_lineage_context_to_manifest_fields(
        code_ref=code_ref,
        environment=environment,
        runtime=runtime,
        external_refs=external_refs,
        lineage_context=context,
    )

    payload = {
        "dataset_id": snapshot_row["dataset_id"],
        "snapshot_name": snapshot_row["name"],
        "snapshot_tag": snapshot_row.get("tag") or "",
        "table_versions": table_versions,
        "row_plan_id": row_plan_id,
        "epoch_plan_id": epoch_plan_id,
        "projection_manifest_ids": projections,
        "code_ref": fields["code_ref"],
        "package_versions": _mapping(package_versions, "package_versions"),
        "environment": _mapping(fields["environment"], "environment"),
        "hardware": _mapping(hardware, "hardware"),
        "runtime": _mapping(fields["runtime"], "runtime"),
        "hyperparameters": _mapping(hyperparameters, "hyperparameters"),
        "random_seeds": _mapping(random_seeds, "random_seeds"),
        "split_policy": split,
        "external_refs": _mapping(fields["external_refs"], "external_refs"),
        "status": status,
    }
    manifest_digest = _digest(payload)
    training_run_id = training_run_id or f"train-{manifest_digest}"
    transform_id = f"tfm-training-run-{manifest_digest}"
    manifest = TrainingRunManifest(
        lake_uri=lake.uri,
        training_run_id=training_run_id,
        dataset_id=str(snapshot_row["dataset_id"]),
        snapshot_name=str(snapshot_row["name"]),
        snapshot_tag=str(snapshot_row.get("tag") or ""),
        table_versions=table_versions,
        row_plan_id=row_plan_id,
        epoch_plan_id=epoch_plan_id,
        projection_manifest_ids=projections,
        code_ref=payload["code_ref"],
        package_versions=payload["package_versions"],
        environment=payload["environment"],
        hardware=payload["hardware"],
        runtime=payload["runtime"],
        hyperparameters=payload["hyperparameters"],
        random_seeds=payload["random_seeds"],
        split_policy=payload["split_policy"],
        external_refs=payload["external_refs"],
        status=status,
        manifest_digest=manifest_digest,
        transform_id=transform_id,
        created_by=created_by,
        lineage_context=context.to_dict() if context else {},
    )
    now = datetime.now(UTC)
    _replace_rows(
        lake,
        "training_runs",
        "training_run_id",
        [manifest.as_row(now)],
        TRAINING_RUNS_SCHEMA,
    )
    _write_transform(
        lake,
        transform_id=transform_id,
        kind="training-run",
        source_id=manifest.dataset_id,
        input_uris=[f"{lake.uri}#dataset_snapshots/{manifest.dataset_id}"],
        input_table_versions=manifest.table_versions,
        output_tables=["training_runs"],
        params={**manifest.to_dict(), "row_ids": [manifest.training_run_id]},
        status=status,
        created_by=created_by,
        now=now,
        lineage_context=context,
    )
    return manifest


def record_checkpoint(
    lake: Lake,
    *,
    training_run_id: str,
    model_artifact_id: str | None = None,
    artifact_uri: str | None = None,
    checksum: str | None = None,
    aliases: Sequence[str] | None = None,
    framework: str | None = None,
    epoch: int | None = None,
    step: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    external_refs: Mapping[str, Any] | None = None,
    created_by: str = "lancedb-robotics",
    lineage_context: Any | None = None,
) -> ModelArtifactManifest:
    """Record a checkpoint/model artifact produced by a training run."""

    handle = begin_lineage_execution(
        lineage_context,
        operation="model-artifact",
        params={"training_run_id": training_run_id, "model_artifact_id": model_artifact_id},
    )
    context = handle.finish(status="completed")
    training_row = _training_run(lake, training_run_id)
    normalized_aliases = _string_tuple(aliases, "aliases")
    if not (model_artifact_id or artifact_uri or checksum or normalized_aliases):
        raise RunManifestError(
            "checkpoint requires model_artifact_id, artifact_uri, checksum, or alias"
        )
    context_refs = context.external_reference_map() if context else {}
    merged_metadata = _mapping(metadata, "metadata")
    if context:
        merged_metadata.setdefault("lineage_context", context.to_dict())
    payload = {
        "training_run_id": training_run_id,
        "artifact_uri": artifact_uri,
        "checksum": checksum,
        "aliases": normalized_aliases,
        "framework": framework,
        "epoch": epoch,
        "step": step,
        "metrics": _mapping(metrics, "metrics"),
        "metadata": merged_metadata,
        "external_refs": _mapping(
            {**context_refs, **dict(external_refs or {})},
            "external_refs",
        ),
    }
    manifest_digest = _digest(payload)
    model_artifact_id = model_artifact_id or f"model-{manifest_digest}"
    transform_id = f"tfm-model-artifact-{manifest_digest}"
    manifest = ModelArtifactManifest(
        lake_uri=lake.uri,
        model_artifact_id=model_artifact_id,
        training_run_id=training_run_id,
        artifact_uri=artifact_uri,
        checksum=checksum,
        aliases=normalized_aliases,
        framework=framework,
        epoch=epoch,
        step=step,
        metrics=payload["metrics"],
        metadata=payload["metadata"],
        external_refs=payload["external_refs"],
        manifest_digest=manifest_digest,
        transform_id=transform_id,
        created_by=created_by,
        lineage_context=context.to_dict() if context else {},
    )
    now = datetime.now(UTC)
    _replace_rows(
        lake,
        "model_artifacts",
        "model_artifact_id",
        [manifest.as_row(now)],
        MODEL_ARTIFACTS_SCHEMA,
    )
    _write_transform(
        lake,
        transform_id=transform_id,
        kind="model-artifact",
        source_id=training_run_id,
        input_uris=[f"{lake.uri}#training_runs/{training_run_id}"],
        input_table_versions=_current_table_versions(lake, ("training_runs",)),
        output_tables=["model_artifacts"],
        params={
            **manifest.to_dict(),
            "dataset_id": training_row["dataset_id"],
            "row_ids": [manifest.model_artifact_id],
        },
        status="completed",
        created_by=created_by,
        now=now,
        lineage_context=context,
    )
    return manifest


def record_evaluation_run(
    lake: Lake,
    snapshot: str | None = None,
    *,
    model_artifact_id: str,
    dataset_id: str | None = None,
    eval_run_id: str | None = None,
    metrics: Mapping[str, Any],
    slice_metrics: Mapping[str, Any] | None = None,
    failure_outputs: Mapping[str, Any] | None = None,
    code_ref: str | None = None,
    package_versions: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    external_refs: Mapping[str, Any] | None = None,
    status: str = "completed",
    created_by: str = "lancedb-robotics",
    lineage_context: Any | None = None,
) -> EvaluationRunManifest:
    """Record evaluation metrics for a model artifact on a pinned snapshot."""

    handle = begin_lineage_execution(
        lineage_context,
        operation="evaluation-run",
        params={"snapshot": snapshot, "dataset_id": dataset_id, "status": status},
    )
    context = handle.finish(status=status)
    artifact_row = _model_artifact(lake, model_artifact_id)
    snapshot_row = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)
    table_versions = _snapshot_table_versions(snapshot_row)
    fields = apply_lineage_context_to_manifest_fields(
        code_ref=code_ref,
        environment=environment,
        runtime=runtime,
        external_refs=external_refs,
        lineage_context=context,
    )
    payload = {
        "model_artifact_id": model_artifact_id,
        "training_run_id": artifact_row["training_run_id"],
        "dataset_id": snapshot_row["dataset_id"],
        "snapshot_name": snapshot_row["name"],
        "snapshot_tag": snapshot_row.get("tag") or "",
        "table_versions": table_versions,
        "metrics": _mapping(metrics, "metrics"),
        "slice_metrics": _mapping(slice_metrics, "slice_metrics"),
        "failure_outputs": _mapping(failure_outputs, "failure_outputs"),
        "code_ref": fields["code_ref"],
        "package_versions": _mapping(package_versions, "package_versions"),
        "environment": _mapping(fields["environment"], "environment"),
        "hardware": _mapping(hardware, "hardware"),
        "runtime": _mapping(fields["runtime"], "runtime"),
        "external_refs": _mapping(fields["external_refs"], "external_refs"),
        "status": status,
    }
    manifest_digest = _digest(payload)
    eval_run_id = eval_run_id or f"eval-{manifest_digest}"
    transform_id = f"tfm-evaluation-run-{manifest_digest}"
    manifest = EvaluationRunManifest(
        lake_uri=lake.uri,
        eval_run_id=eval_run_id,
        model_artifact_id=model_artifact_id,
        training_run_id=str(artifact_row["training_run_id"]),
        dataset_id=str(snapshot_row["dataset_id"]),
        snapshot_name=str(snapshot_row["name"]),
        snapshot_tag=str(snapshot_row.get("tag") or ""),
        table_versions=table_versions,
        metrics=payload["metrics"],
        slice_metrics=payload["slice_metrics"],
        failure_outputs=payload["failure_outputs"],
        code_ref=payload["code_ref"],
        package_versions=payload["package_versions"],
        environment=payload["environment"],
        hardware=payload["hardware"],
        runtime=payload["runtime"],
        external_refs=payload["external_refs"],
        status=status,
        manifest_digest=manifest_digest,
        transform_id=transform_id,
        created_by=created_by,
        lineage_context=context.to_dict() if context else {},
    )
    now = datetime.now(UTC)
    _replace_rows(
        lake,
        "evaluation_runs",
        "eval_run_id",
        [manifest.as_row(now)],
        EVALUATION_RUNS_SCHEMA,
    )
    _write_transform(
        lake,
        transform_id=transform_id,
        kind="evaluation-run",
        source_id=model_artifact_id,
        input_uris=[
            f"{lake.uri}#model_artifacts/{model_artifact_id}",
            f"{lake.uri}#dataset_snapshots/{manifest.dataset_id}",
        ],
        input_table_versions=(
            *manifest.table_versions,
            *_current_table_versions(lake, ("model_artifacts",)),
        ),
        output_tables=["evaluation_runs"],
        params={**manifest.to_dict(), "row_ids": [manifest.eval_run_id]},
        status=status,
        created_by=created_by,
        now=now,
        lineage_context=context,
    )
    # Materialize this eval's aggregate/slice metrics inline (backlog 0100), the
    # same zero-divergence emission model as inline lineage (0098): a metric-key
    # lookup can push down to an indexed predicate without parsing manifest JSON.
    # Best-effort on lakes predating the table (they backfill via
    # ``sync_evaluation_run_metrics`` / ``train sync-metrics``).
    _emit_evaluation_metric_rows(lake, manifest, now=now, created_by=created_by)
    return manifest


def attach_training_external_refs(
    lake: Lake,
    training_run_id: str,
    refs: Mapping[str, Any],
    *,
    replace: bool = False,
) -> ExternalReferenceUpdate:
    """Attach external tracker references to an existing training manifest."""

    return _attach_external_refs(
        lake,
        table_name="training_runs",
        id_column="training_run_id",
        manifest_id=training_run_id,
        refs=refs,
        schema=TRAINING_RUNS_SCHEMA,
        replace=replace,
    )


def attach_model_external_refs(
    lake: Lake,
    model_artifact_id: str,
    refs: Mapping[str, Any],
    *,
    replace: bool = False,
) -> ExternalReferenceUpdate:
    """Attach external artifact/run references to an existing model artifact."""

    return _attach_external_refs(
        lake,
        table_name="model_artifacts",
        id_column="model_artifact_id",
        manifest_id=model_artifact_id,
        refs=refs,
        schema=MODEL_ARTIFACTS_SCHEMA,
        replace=replace,
    )


def attach_evaluation_external_refs(
    lake: Lake,
    eval_run_id: str,
    refs: Mapping[str, Any],
    *,
    replace: bool = False,
) -> ExternalReferenceUpdate:
    """Attach external tracker references to an existing evaluation manifest."""

    return _attach_external_refs(
        lake,
        table_name="evaluation_runs",
        id_column="eval_run_id",
        manifest_id=eval_run_id,
        refs=refs,
        schema=EVALUATION_RUNS_SCHEMA,
        replace=replace,
    )


def evaluation_runs(
    lake: Lake,
    *,
    model_artifact_id: str | None = None,
    dataset_id: str | None = None,
    snapshot: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Retrieve evaluation rows by model artifact and/or snapshot.

    Routes through the bounded/indexed query path (backlog 0100): the filters
    push down to scalar-indexed columns instead of materializing the whole
    ``evaluation_runs`` table and filtering in Python.
    """

    return query_evaluation_runs(
        lake,
        model_artifact_id=model_artifact_id,
        dataset_id=dataset_id,
        snapshot=snapshot,
    ).rows


def _attach_external_refs(
    lake: Lake,
    *,
    table_name: str,
    id_column: str,
    manifest_id: str,
    refs: Mapping[str, Any],
    schema: pa.Schema,
    replace: bool,
) -> ExternalReferenceUpdate:
    normalized_refs = {
        str(key): _metadata_value(value)
        for key, value in _mapping(refs, "external_refs").items()
        if value is not None
    }
    if not normalized_refs:
        raise RunManifestError("external_refs must contain at least one non-null reference")
    rows = lake.table(table_name).to_arrow().to_pylist()
    matches = [row for row in rows if row.get(id_column) == manifest_id]
    if not matches:
        raise RunManifestError(f"unknown {id_column} {manifest_id!r}")
    row = dict(matches[0])
    existing = {} if replace else _metadata_map(row, column="external_refs")
    merged = {**existing, **normalized_refs}
    row["external_refs"] = _kv_items(merged)
    _replace_rows(lake, table_name, id_column, [row], schema)
    return ExternalReferenceUpdate(
        lake_uri=lake.uri,
        table_name=table_name,
        id_column=id_column,
        manifest_id=manifest_id,
        external_refs=merged,
        replace=replace,
    )


def _infer_training_dataset(dataset: Any | None) -> dict[str, str | None]:
    if dataset is None:
        return {}
    manifest = getattr(dataset, "manifest", None)
    if manifest is None:
        raise RunManifestError("dataset must expose a training manifest")
    return {
        "dataset_id": getattr(manifest, "dataset_id", None),
        "snapshot_name": getattr(manifest, "snapshot_name", None),
        "row_plan_id": getattr(manifest, "row_plan_id", None),
        "epoch_plan_id": getattr(manifest, "epoch_plan_id", None),
    }


def _resolve_snapshot(
    lake: Lake,
    *,
    snapshot: str | None,
    dataset_id: str | None,
) -> dict[str, Any]:
    rows = lake.table("dataset_snapshots").to_arrow().to_pylist()
    if dataset_id is not None:
        matches = [row for row in rows if row["dataset_id"] == dataset_id]
        if not matches:
            raise RunManifestError(f"unknown dataset_id {dataset_id!r}")
        row = matches[0]
        if snapshot and snapshot not in {row["dataset_id"], row["name"], row.get("tag")}:
            raise RunManifestError(
                f"snapshot {snapshot!r} does not match dataset_id {dataset_id!r}"
            )
        return row
    if snapshot is None:
        raise RunManifestError("snapshot or dataset_id is required")
    matches = [
        row
        for row in rows
        if snapshot in {row["dataset_id"], row["name"], row.get("tag")}
    ]
    if not matches:
        raise RunManifestError(f"unknown dataset snapshot {snapshot!r}")
    return max(matches, key=lambda row: (_created_at_key(row), row["dataset_id"]))


def _training_run(lake: Lake, training_run_id: str) -> dict[str, Any]:
    for row in lake.table("training_runs").to_arrow().to_pylist():
        if row["training_run_id"] == training_run_id:
            return row
    raise RunManifestError(f"unknown training_run_id {training_run_id!r}")


def _model_artifact(lake: Lake, model_artifact_id: str) -> dict[str, Any]:
    for row in lake.table("model_artifacts").to_arrow().to_pylist():
        if row["model_artifact_id"] == model_artifact_id:
            return row
    raise RunManifestError(f"unknown model_artifact_id {model_artifact_id!r}")


def _snapshot_table_versions(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(normalize_table_versions(snapshot.get("table_versions") or ()))


def _created_at_key(row: Mapping[str, Any]) -> datetime:
    value = row.get("created_at")
    if isinstance(value, datetime):
        return value
    return datetime.min.replace(tzinfo=UTC)


def _json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise RunManifestError(f"invalid JSON object: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise RunManifestError("expected a JSON object")
    return parsed


def _mapping(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RunManifestError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _string_tuple(values: Sequence[str] | None, label: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise RunManifestError(f"{label} must be a sequence of strings")
    return tuple(str(value) for value in values if value is not None and str(value))


def _kv_items(metadata: Mapping[str, Any] | None) -> list[dict[str, str]]:
    if not metadata:
        return []
    return [
        {"key": str(key), "value": _metadata_value(value)}
        for key, value in sorted(metadata.items())
        if value is not None
    ]


def _metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def _metadata_map(row: Mapping[str, Any], *, column: str = "metadata") -> dict[str, str]:
    result: dict[str, str] = {}
    for item in row.get(column) or []:
        if not isinstance(item, Mapping) or item.get("key") is None:
            continue
        result[str(item["key"])] = "" if item.get("value") is None else str(item["value"])
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, set)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)


def _digest(payload: Any) -> str:
    return hashlib.sha1(_json_dumps(payload).encode()).hexdigest()[:20]


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _replace_rows(
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


def _current_table_versions(lake: Lake, tables: Sequence[str]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"table": table, "version": int(lake.table(table).version), "tag": ""}
        for table in tables
    )


def _write_transform(
    lake: Lake,
    *,
    transform_id: str,
    kind: str,
    source_id: str,
    input_uris: Sequence[str],
    input_table_versions: Sequence[dict[str, Any]],
    output_tables: Sequence[str],
    params: Mapping[str, Any],
    status: str,
    created_by: str,
    now: datetime,
    lineage_context: Any | None = None,
) -> None:
    params = attach_lineage_context_to_params(params, lineage_context)
    transform_row = {
        "transform_id": transform_id,
        "kind": kind,
        "source_id": source_id,
        "input_uris": list(input_uris),
        "input_table_versions": list(input_table_versions),
        "output_tables": list(output_tables),
        "params": _json_dumps(params),
        "status": status,
        "error": None,
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    _replace_rows(lake, "transform_runs", "transform_id", [transform_row], TRANSFORM_RUNS_SCHEMA)
    # Emit lineage inline (backlog 0098): record the training/eval import's lineage
    # slice without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)


# ---------------------------------------------------------------------------
# Backlog 0100: indexed/bounded manifest query, paged listing, materialized
# metric-key lookup, and lineage/feedback/snapshot retention planning over the
# training/evaluation run-manifest tables. The v1 write path (above) stores rich
# manifests as JSON strings; these read surfaces push down to scalar-indexed
# scalar columns (see indexing.PREDICATE_INDEX_COLUMNS_BY_TABLE) instead of
# scanning + parsing every manifest in Python.
# ---------------------------------------------------------------------------

_MANIFEST_ID_COLUMN = {
    "training_runs": "training_run_id",
    "model_artifacts": "model_artifact_id",
    "evaluation_runs": "eval_run_id",
    EVALUATION_RUN_METRICS_TABLE: "metric_row_id",
    TRAINING_REPORTS_TABLE: "report_id",
}
_MANIFEST_KIND_TABLE = {
    "training-run": "training_runs",
    "model-artifact": "model_artifacts",
    "evaluation-run": "evaluation_runs",
    "training-report": TRAINING_REPORTS_TABLE,
}


@dataclass(frozen=True)
class _Predicate:
    """A pushdownable scalar predicate (``column op value``)."""

    column: str
    op: str
    value: Any


def _sql_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return _sql_literal(str(value))


def _predicates_sql(predicates: Sequence[_Predicate]) -> str | None:
    clauses = [f"{p.column} {p.op} {_sql_value(p.value)}" for p in predicates]
    return " AND ".join(clauses) if clauses else None


def _predicate_matches(pred: _Predicate, row: Mapping[str, Any]) -> bool:
    actual = row.get(pred.column)
    if pred.op == "=":
        return actual == pred.value
    if actual is None:
        return False
    try:
        if pred.op == "<":
            return actual < pred.value
        if pred.op == "<=":
            return actual <= pred.value
        if pred.op == ">":
            return actual > pred.value
        if pred.op == ">=":
            return actual >= pred.value
    except TypeError:
        return False
    return False


@dataclass(frozen=True)
class ManifestQueryPlan:
    """How a manifest query executed: server-side pushdown vs full-scan fallback.

    ``bounded`` is True when the ``.search().where().select().limit()`` path ran
    (the engine narrowed the scan); False when the backend could not build the
    query and we fell back to materializing the table and filtering in Python.
    ``scanned_rows`` is the engine-returned count on the bounded path (already
    narrowed) and the whole-table count on the fallback -- honest either way.
    """

    table: str
    filter_sql: str | None
    columns: tuple[str, ...] | None
    limit: int | None
    bounded: bool
    scanned_rows: int
    matched_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "filter_sql": self.filter_sql,
            "columns": list(self.columns) if self.columns is not None else None,
            "limit": self.limit,
            "bounded": self.bounded,
            "scanned_rows": self.scanned_rows,
            "matched_rows": self.matched_rows,
        }


def _bounded_rows(
    lake: Lake,
    table: str,
    *,
    predicates: Sequence[_Predicate] = (),
    py_predicates: Sequence[Callable[[Mapping[str, Any]], bool]] = (),
    columns: Sequence[str] | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], ManifestQueryPlan]:
    """Read rows with server-side predicate/column/limit pushdown when available.

    ``predicates`` push down as a SQL ``WHERE`` and are re-applied in Python on
    the full-scan fallback so the fallback stays correct. ``py_predicates`` are
    Python-only post-filters -- ``external_refs``/``aliases`` live in list
    columns that do not push down -- and run on both paths without ever setting
    ``bounded``. When a Python post-filter is active the engine ``limit`` is
    dropped (it would truncate before the post-filter) and applied after.
    """
    handle = lake.table(table)
    available = set(handle.schema.names)
    where_sql = _predicates_sql(predicates)
    projected = [c for c in (columns or ()) if c in available] or None
    hard_limit = limit if not py_predicates else None
    rows: list[dict[str, Any]] | None = None
    bounded = False
    try:
        query = handle.search()
        if projected:
            query = query.select(projected)
        if where_sql:
            query = query.where(where_sql)
        if hard_limit is not None:
            query = query.limit(hard_limit)
        collected: list[dict[str, Any]] = []
        for batch in query.to_batches(batch_size=_SCAN_BATCH_SIZE):
            collected.extend(batch.to_pylist())
            if hard_limit is not None and len(collected) >= hard_limit:
                collected = collected[:hard_limit]
                break
        rows = collected
        bounded = True
    except Exception:  # noqa: BLE001 - backends without server-side query: fall back
        rows = None
        bounded = False
    if rows is None:
        materialized = handle.to_arrow().to_pylist()
        scanned = len(materialized)
        rows = [row for row in materialized if all(_predicate_matches(p, row) for p in predicates)]
        if projected:
            wanted = set(projected)
            rows = [{k: v for k, v in row.items() if k in wanted} for row in rows]
    else:
        scanned = len(rows)
    if py_predicates:
        rows = [row for row in rows if all(pred(row) for pred in py_predicates)]
    if limit is not None:
        rows = rows[:limit]
    plan = ManifestQueryPlan(
        table=table,
        filter_sql=where_sql,
        columns=tuple(projected) if projected else None,
        limit=limit,
        bounded=bounded,
        scanned_rows=scanned,
        matched_rows=len(rows),
    )
    return rows, plan


def _split_ref(external_ref: Any) -> tuple[str, str | None]:
    if isinstance(external_ref, (tuple, list)) and len(external_ref) == 2:
        key, value = external_ref
        return str(key), (None if value is None else str(value))
    text = str(external_ref)
    if "=" in text:
        key, _, value = text.partition("=")
        return key, value
    return text, None


def _external_ref_predicate(external_ref: Any) -> Callable[[Mapping[str, Any]], bool]:
    key, value = _split_ref(external_ref)

    def matches(row: Mapping[str, Any]) -> bool:
        refs = _metadata_map(row, column="external_refs")
        if key not in refs:
            return False
        return value is None or refs[key] == value

    return matches


def _alias_predicate(alias: str) -> Callable[[Mapping[str, Any]], bool]:
    target = str(alias)

    def matches(row: Mapping[str, Any]) -> bool:
        return target in (row.get("aliases") or ())

    return matches


def _sorted_by_created(rows: Sequence[dict[str, Any]], id_column: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (_created_at_key(row), str(row.get(id_column) or "")))


@dataclass(frozen=True)
class ManifestQueryResult:
    """Rows plus the execution plan of one bounded manifest query."""

    table: str
    rows: tuple[dict[str, Any], ...]
    plan: ManifestQueryPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "rows": [dict(row) for row in self.rows],
            "count": len(self.rows),
            "plan": self.plan.to_dict(),
        }


def query_training_runs(
    lake: Lake,
    *,
    dataset_id: str | None = None,
    snapshot: str | None = None,
    code_ref: str | None = None,
    status: str | None = None,
    external_ref: Any | None = None,
    limit: int | None = None,
) -> ManifestQueryResult:
    """Query ``training_runs`` by snapshot/dataset, code ref, status, tracker ref."""
    if snapshot is not None:
        dataset_id = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)["dataset_id"]
    predicates: list[_Predicate] = []
    if dataset_id is not None:
        predicates.append(_Predicate("dataset_id", "=", str(dataset_id)))
    if code_ref is not None:
        predicates.append(_Predicate("code_ref", "=", str(code_ref)))
    if status is not None:
        predicates.append(_Predicate("status", "=", str(status)))
    py = [_external_ref_predicate(external_ref)] if external_ref is not None else []
    rows, plan = _bounded_rows(lake, "training_runs", predicates=predicates, py_predicates=py, limit=limit)
    return ManifestQueryResult("training_runs", tuple(_sorted_by_created(rows, "training_run_id")), plan)


def query_model_artifacts(
    lake: Lake,
    *,
    training_run_id: str | None = None,
    artifact_uri: str | None = None,
    checksum: str | None = None,
    framework: str | None = None,
    alias: str | None = None,
    external_ref: Any | None = None,
    limit: int | None = None,
) -> ManifestQueryResult:
    """Query ``model_artifacts`` by training run, uri, checksum, framework, alias."""
    predicates: list[_Predicate] = []
    if training_run_id is not None:
        predicates.append(_Predicate("training_run_id", "=", str(training_run_id)))
    if artifact_uri is not None:
        predicates.append(_Predicate("artifact_uri", "=", str(artifact_uri)))
    if checksum is not None:
        predicates.append(_Predicate("checksum", "=", str(checksum)))
    if framework is not None:
        predicates.append(_Predicate("framework", "=", str(framework)))
    py: list[Callable[[Mapping[str, Any]], bool]] = []
    if alias is not None:
        py.append(_alias_predicate(alias))
    if external_ref is not None:
        py.append(_external_ref_predicate(external_ref))
    rows, plan = _bounded_rows(lake, "model_artifacts", predicates=predicates, py_predicates=py, limit=limit)
    return ManifestQueryResult("model_artifacts", tuple(_sorted_by_created(rows, "model_artifact_id")), plan)


def query_evaluation_runs(
    lake: Lake,
    *,
    model_artifact_id: str | None = None,
    training_run_id: str | None = None,
    dataset_id: str | None = None,
    snapshot: str | None = None,
    status: str | None = None,
    external_ref: Any | None = None,
    limit: int | None = None,
) -> ManifestQueryResult:
    """Query ``evaluation_runs`` by model artifact, training run, snapshot, status."""
    if snapshot is not None:
        dataset_id = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)["dataset_id"]
    predicates: list[_Predicate] = []
    if model_artifact_id is not None:
        predicates.append(_Predicate("model_artifact_id", "=", str(model_artifact_id)))
    if training_run_id is not None:
        predicates.append(_Predicate("training_run_id", "=", str(training_run_id)))
    if dataset_id is not None:
        predicates.append(_Predicate("dataset_id", "=", str(dataset_id)))
    if status is not None:
        predicates.append(_Predicate("status", "=", str(status)))
    py = [_external_ref_predicate(external_ref)] if external_ref is not None else []
    rows, plan = _bounded_rows(lake, "evaluation_runs", predicates=predicates, py_predicates=py, limit=limit)
    return ManifestQueryResult("evaluation_runs", tuple(_sorted_by_created(rows, "eval_run_id")), plan)


# ---- Materialized eval-metric surface ------------------------------------

def _flatten_eval_metrics(
    *,
    eval_run_id: str,
    model_artifact_id: str,
    training_run_id: str,
    dataset_id: str,
    snapshot_name: str,
    snapshot_tag: str,
    status: str,
    eval_created_at: datetime | None,
    metrics: Mapping[str, Any] | None,
    slice_metrics: Mapping[str, Any] | None,
    transform_id: str,
    created_by: str,
    now: datetime,
) -> list[dict[str, Any]]:
    """Explode one eval manifest's aggregate + slice metrics into indexable rows."""
    rows: list[dict[str, Any]] = []

    def add(scope: str, slice_label: str, metric: str, metric_key: str, value: Any) -> None:
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        rows.append(
            {
                "metric_row_id": f"evm-{_digest([eval_run_id, scope, metric_key])}",
                "eval_run_id": eval_run_id,
                "model_artifact_id": model_artifact_id,
                "training_run_id": training_run_id,
                "dataset_id": dataset_id,
                "snapshot_name": snapshot_name,
                "snapshot_tag": snapshot_tag,
                "scope": scope,
                "slice_label": slice_label,
                "metric": metric,
                "metric_key": metric_key,
                "score": float(value) if numeric else None,
                "value_json": _json_dumps(value),
                "status": status,
                "eval_created_at": eval_created_at,
                "transform_id": transform_id,
                "created_by": created_by,
                "created_at": now,
            }
        )

    for metric, value in sorted((metrics or {}).items()):
        add("aggregate", "", str(metric), str(metric), value)
    for slice_label, slice_value in sorted((slice_metrics or {}).items()):
        label = str(slice_label)
        if isinstance(slice_value, Mapping):
            for metric, value in sorted(slice_value.items()):
                add("slice", label, str(metric), f"{label}.{metric}", value)
        else:
            add("slice", label, label, label, slice_value)
    return rows


def _emit_evaluation_metric_rows(
    lake: Lake,
    manifest: EvaluationRunManifest,
    *,
    now: datetime,
    created_by: str,
) -> int:
    """Materialize one eval manifest's metrics into ``evaluation_run_metrics``.

    Best-effort: a lake predating the table (created before 0100) skips this and
    backfills via :func:`sync_evaluation_run_metrics`.
    """
    if EVALUATION_RUN_METRICS_TABLE not in lake.table_names():
        return 0
    rows = _flatten_eval_metrics(
        eval_run_id=manifest.eval_run_id,
        model_artifact_id=manifest.model_artifact_id,
        training_run_id=manifest.training_run_id,
        dataset_id=manifest.dataset_id,
        snapshot_name=manifest.snapshot_name,
        snapshot_tag=manifest.snapshot_tag,
        status=manifest.status,
        eval_created_at=now,
        metrics=manifest.metrics,
        slice_metrics=manifest.slice_metrics,
        transform_id=manifest.transform_id,
        created_by=created_by,
        now=now,
    )
    table = lake.table(EVALUATION_RUN_METRICS_TABLE)
    table.delete(f"eval_run_id = {_sql_literal(manifest.eval_run_id)}")
    if rows:
        table.add(pa.Table.from_pylist(rows, schema=EVALUATION_RUN_METRICS_SCHEMA))
    return len(rows)


def _clear_table(lake: Lake, table_name: str) -> None:
    handle = lake.table(table_name)
    try:
        handle.delete("true")
        return
    except Exception:  # noqa: BLE001 - backend without a literal-true predicate
        pass
    id_column = _MANIFEST_ID_COLUMN.get(table_name, "metric_row_id")
    ids = [str(row[id_column]) for row in handle.to_arrow().to_pylist()]
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        handle.delete(f"{id_column} IN ({', '.join(_sql_literal(value) for value in chunk)})")


@dataclass(frozen=True)
class EvalMetricSyncReport:
    """Result of rebuilding the materialized ``evaluation_run_metrics`` surface."""

    lake_uri: str
    eval_runs: int
    metric_rows: int
    index_results: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "eval_runs": self.eval_runs,
            "metric_rows": self.metric_rows,
            "index_results": [dict(item) for item in self.index_results],
        }


def sync_evaluation_run_metrics(
    lake: Lake,
    *,
    build_indexes: bool = True,
    created_by: str = "lancedb-robotics",
) -> EvalMetricSyncReport:
    """Rebuild ``evaluation_run_metrics`` from every ``evaluation_runs`` manifest.

    Deterministic and idempotent: the table is cleared and re-materialized from
    the source JSON columns, then its scalar predicate indexes are (re)built.
    Backfills lakes created before 0100 and repairs any drift.
    """
    eval_rows = lake.table("evaluation_runs").to_arrow().to_pylist()
    now = datetime.now(UTC)
    metric_rows: list[dict[str, Any]] = []
    for row in eval_rows:
        metric_rows.extend(
            _flatten_eval_metrics(
                eval_run_id=str(row.get("eval_run_id") or ""),
                model_artifact_id=str(row.get("model_artifact_id") or ""),
                training_run_id=str(row.get("training_run_id") or ""),
                dataset_id=str(row.get("dataset_id") or ""),
                snapshot_name=str(row.get("snapshot_name") or ""),
                snapshot_tag=str(row.get("snapshot_tag") or ""),
                status=str(row.get("status") or ""),
                eval_created_at=row.get("created_at"),
                metrics=_json_object(row.get("metrics_json")),
                slice_metrics=_json_object(row.get("slice_metrics_json")),
                transform_id=str(row.get("transform_id") or ""),
                created_by=created_by,
                now=now,
            )
        )
    _clear_table(lake, EVALUATION_RUN_METRICS_TABLE)
    if metric_rows:
        lake.table(EVALUATION_RUN_METRICS_TABLE).add(
            pa.Table.from_pylist(metric_rows, schema=EVALUATION_RUN_METRICS_SCHEMA)
        )
    index_results: tuple[dict[str, Any], ...] = ()
    if build_indexes and metric_rows:
        from lancedb_robotics.indexing import build_run_manifest_predicate_indexes

        index_results = tuple(
            result.to_params()
            for result in build_run_manifest_predicate_indexes(lake, include_metrics=True)
        )
    return EvalMetricSyncReport(
        lake_uri=lake.uri,
        eval_runs=len(eval_rows),
        metric_rows=len(metric_rows),
        index_results=index_results,
    )


def _metric_table_has_rows(lake: Lake) -> bool:
    if EVALUATION_RUN_METRICS_TABLE not in lake.table_names():
        return False
    try:
        return lake.table(EVALUATION_RUN_METRICS_TABLE).count_rows() > 0
    except Exception:  # noqa: BLE001 - treat an unreadable surface as unavailable
        return False


@dataclass(frozen=True)
class EvalMetricLookupResult:
    """Metric-key lookup rows plus whether the materialized index path was used.

    ``materialized`` is True when the lookup pushed down to the indexed
    ``evaluation_run_metrics`` surface, False when it fell back to parsing the
    ``evaluation_runs`` JSON columns in Python (no materialized surface yet).
    """

    rows: tuple[dict[str, Any], ...]
    materialized: bool
    plan: ManifestQueryPlan | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [dict(row) for row in self.rows],
            "count": len(self.rows),
            "materialized": self.materialized,
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }


def query_eval_metrics(
    lake: Lake,
    *,
    metric: str | None = None,
    metric_key: str | None = None,
    slice_label: str | None = None,
    scope: str | None = None,
    model_artifact_id: str | None = None,
    dataset_id: str | None = None,
    snapshot: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    limit: int | None = None,
) -> EvalMetricLookupResult:
    """Look up eval metrics/slice-metrics by key without parsing every manifest.

    Uses the materialized ``evaluation_run_metrics`` surface (indexed predicate
    pushdown) when it has rows; otherwise falls back to flattening the
    ``evaluation_runs`` JSON in Python and reports ``materialized=False`` so the
    caller knows an index is not yet available.
    """
    if snapshot is not None:
        dataset_id = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)["dataset_id"]
    predicates: list[_Predicate] = []
    if metric is not None:
        predicates.append(_Predicate("metric", "=", str(metric)))
    if metric_key is not None:
        predicates.append(_Predicate("metric_key", "=", str(metric_key)))
    if slice_label is not None:
        predicates.append(_Predicate("slice_label", "=", str(slice_label)))
    if scope is not None:
        predicates.append(_Predicate("scope", "=", str(scope)))
    if model_artifact_id is not None:
        predicates.append(_Predicate("model_artifact_id", "=", str(model_artifact_id)))
    if dataset_id is not None:
        predicates.append(_Predicate("dataset_id", "=", str(dataset_id)))
    if min_score is not None:
        predicates.append(_Predicate("score", ">=", float(min_score)))
    if max_score is not None:
        predicates.append(_Predicate("score", "<", float(max_score)))

    if _metric_table_has_rows(lake):
        rows, plan = _bounded_rows(
            lake, EVALUATION_RUN_METRICS_TABLE, predicates=predicates, limit=limit
        )
        ordered = sorted(
            rows, key=lambda row: (str(row.get("metric_key") or ""), str(row.get("eval_run_id") or ""))
        )
        return EvalMetricLookupResult(tuple(ordered), True, plan)

    now = datetime.now(UTC)
    flattened: list[dict[str, Any]] = []
    for row in lake.table("evaluation_runs").to_arrow().to_pylist():
        flattened.extend(
            _flatten_eval_metrics(
                eval_run_id=str(row.get("eval_run_id") or ""),
                model_artifact_id=str(row.get("model_artifact_id") or ""),
                training_run_id=str(row.get("training_run_id") or ""),
                dataset_id=str(row.get("dataset_id") or ""),
                snapshot_name=str(row.get("snapshot_name") or ""),
                snapshot_tag=str(row.get("snapshot_tag") or ""),
                status=str(row.get("status") or ""),
                eval_created_at=row.get("created_at"),
                metrics=_json_object(row.get("metrics_json")),
                slice_metrics=_json_object(row.get("slice_metrics_json")),
                transform_id=str(row.get("transform_id") or ""),
                created_by="lancedb-robotics",
                now=now,
            )
        )
    matched = [row for row in flattened if all(_predicate_matches(p, row) for p in predicates)]
    matched.sort(key=lambda row: (str(row.get("metric_key") or ""), str(row.get("eval_run_id") or "")))
    if limit is not None:
        matched = matched[:limit]
    return EvalMetricLookupResult(tuple(matched), False, None)


# ---- Paged listing --------------------------------------------------------

@dataclass(frozen=True)
class ManifestPage:
    """One deterministic page of manifest rows ordered by (created_at, id)."""

    table: str
    rows: tuple[dict[str, Any], ...]
    page_size: int
    total_count: int
    next_page_token: str | None
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "rows": [dict(row) for row in self.rows],
            "page": {
                "page_size": self.page_size,
                "returned": len(self.rows),
                "total_count": self.total_count,
                "next_page_token": self.next_page_token,
                "truncated": self.truncated,
            },
        }


def _page_query_digest(table: str, filters: Mapping[str, Any]) -> str:
    payload = {"table": table, "filters": {k: v for k, v in sorted(filters.items()) if v is not None}}
    return _digest(payload)[:16]


def _encode_page_token(table: str, digest: str, after_ts: str, after_id: str) -> str:
    raw = json.dumps(
        {"t": table, "q": digest, "after_ts": after_ts, "after_id": after_id},
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_page_token(token: str, table: str, digest: str) -> tuple[str, str]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(str(token).encode("ascii")).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a malformed token is a caller error
        raise RunManifestError("invalid manifest page token") from exc
    if payload.get("t") != table or payload.get("q") != digest:
        raise RunManifestError(
            "manifest page token does not match this query (filters changed?)"
        )
    return str(payload.get("after_ts") or ""), str(payload.get("after_id") or "")


def _cursor_start(ordered: Sequence[dict[str, Any]], id_column: str, cursor: tuple[str, str]) -> int:
    for index, row in enumerate(ordered):
        key = (_created_at_key(row).isoformat(), str(row.get(id_column) or ""))
        if key > cursor:
            return index
    return len(ordered)


def _paginate(
    lake: Lake,
    table: str,
    id_column: str,
    *,
    predicates: Sequence[_Predicate],
    filters: Mapping[str, Any],
    page_size: int,
    page_token: str | None,
    columns: Sequence[str] | None = None,
) -> ManifestPage:
    size = max(1, min(int(page_size), _MAX_PAGE_SIZE))
    # ``columns`` narrows the projection (e.g. summary listings that must not
    # pull large JSON payload columns into memory). The sort/cursor keys
    # (``created_at`` + ``id_column``) are always projected so paging stays
    # deterministic even when the caller asks for a subset.
    if columns is not None:
        columns = list(dict.fromkeys([*columns, "created_at", id_column]))
    rows, _plan = _bounded_rows(lake, table, predicates=predicates, columns=columns)
    ordered = _sorted_by_created(rows, id_column)
    digest = _page_query_digest(table, filters)
    start = 0
    if page_token:
        cursor = _decode_page_token(page_token, table, digest)
        start = _cursor_start(ordered, id_column, cursor)
    page = ordered[start : start + size]
    truncated = start + size < len(ordered)
    next_token = None
    if truncated and page:
        last = page[-1]
        next_token = _encode_page_token(
            table, digest, _created_at_key(last).isoformat(), str(last.get(id_column) or "")
        )
    return ManifestPage(
        table=table,
        rows=tuple(page),
        page_size=size,
        total_count=len(ordered),
        next_page_token=next_token,
        truncated=truncated,
    )


def list_training_runs(
    lake: Lake,
    *,
    page_size: int = _DEFAULT_PAGE_SIZE,
    page_token: str | None = None,
    dataset_id: str | None = None,
    snapshot: str | None = None,
    code_ref: str | None = None,
    status: str | None = None,
) -> ManifestPage:
    """Deterministic paged listing of ``training_runs`` (order: created_at, id)."""
    if snapshot is not None:
        dataset_id = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)["dataset_id"]
    predicates: list[_Predicate] = []
    if dataset_id is not None:
        predicates.append(_Predicate("dataset_id", "=", str(dataset_id)))
    if code_ref is not None:
        predicates.append(_Predicate("code_ref", "=", str(code_ref)))
    if status is not None:
        predicates.append(_Predicate("status", "=", str(status)))
    filters = {"dataset_id": dataset_id, "code_ref": code_ref, "status": status}
    return _paginate(
        lake,
        "training_runs",
        "training_run_id",
        predicates=predicates,
        filters=filters,
        page_size=page_size,
        page_token=page_token,
    )


def list_model_artifacts(
    lake: Lake,
    *,
    page_size: int = _DEFAULT_PAGE_SIZE,
    page_token: str | None = None,
    training_run_id: str | None = None,
    framework: str | None = None,
) -> ManifestPage:
    """Deterministic paged listing of ``model_artifacts``."""
    predicates: list[_Predicate] = []
    if training_run_id is not None:
        predicates.append(_Predicate("training_run_id", "=", str(training_run_id)))
    if framework is not None:
        predicates.append(_Predicate("framework", "=", str(framework)))
    filters = {"training_run_id": training_run_id, "framework": framework}
    return _paginate(
        lake,
        "model_artifacts",
        "model_artifact_id",
        predicates=predicates,
        filters=filters,
        page_size=page_size,
        page_token=page_token,
    )


def list_evaluation_runs(
    lake: Lake,
    *,
    page_size: int = _DEFAULT_PAGE_SIZE,
    page_token: str | None = None,
    model_artifact_id: str | None = None,
    training_run_id: str | None = None,
    dataset_id: str | None = None,
    snapshot: str | None = None,
    status: str | None = None,
) -> ManifestPage:
    """Deterministic paged listing of ``evaluation_runs``."""
    if snapshot is not None:
        dataset_id = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)["dataset_id"]
    predicates: list[_Predicate] = []
    if model_artifact_id is not None:
        predicates.append(_Predicate("model_artifact_id", "=", str(model_artifact_id)))
    if training_run_id is not None:
        predicates.append(_Predicate("training_run_id", "=", str(training_run_id)))
    if dataset_id is not None:
        predicates.append(_Predicate("dataset_id", "=", str(dataset_id)))
    if status is not None:
        predicates.append(_Predicate("status", "=", str(status)))
    filters = {
        "model_artifact_id": model_artifact_id,
        "training_run_id": training_run_id,
        "dataset_id": dataset_id,
        "status": status,
    }
    return _paginate(
        lake,
        "evaluation_runs",
        "eval_run_id",
        predicates=predicates,
        filters=filters,
        page_size=page_size,
        page_token=page_token,
    )


# ---- Retention planning ---------------------------------------------------

@dataclass(frozen=True)
class ManifestProtection:
    """Why (or whether) one manifest row is retained against expiry."""

    kind: str
    manifest_id: str
    protected: bool
    reasons: tuple[str, ...]
    categories: tuple[str, ...]
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "manifest_id": self.manifest_id,
            "protected": self.protected,
            "reasons": list(self.reasons),
            "categories": list(self.categories),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass(frozen=True)
class ManifestRetentionPlan:
    """Which manifest rows are protected vs safe to expire, and why (0100)."""

    lake_uri: str
    older_than: str | None
    protected: tuple[ManifestProtection, ...]
    deletable: tuple[ManifestProtection, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "older_than": self.older_than,
            "protected": [item.to_dict() for item in self.protected],
            "deletable": [item.to_dict() for item in self.deletable],
            "protected_count": len(self.protected),
            "deletable_count": len(self.deletable),
        }


def _project_rows(lake: Lake, table: str, columns: Sequence[str]) -> list[dict[str, Any]]:
    if table not in lake.table_names():
        return []
    rows, _plan = _bounded_rows(lake, table, columns=columns)
    return rows


def _lineage_referenced_artifact_ids(lake: Lake) -> set[str]:
    referenced: set[str] = set()
    for row in _project_rows(lake, "lineage_edges", ("from_artifact_id", "to_artifact_id")):
        for key in ("from_artifact_id", "to_artifact_id"):
            value = row.get(key)
            if value:
                referenced.add(str(value))
    return referenced


def _protection_index(lake: Lake) -> dict[tuple[str, str], tuple[set[str], set[str]]]:
    """Map (kind, manifest_id) -> (reasons, categories) from downstream refs."""
    from lancedb_robotics.lineage import (
        evaluation_run_artifact_id,
        model_artifact_lineage_id,
        training_run_artifact_id,
    )

    index: dict[tuple[str, str], tuple[set[str], set[str]]] = {}

    def note(kind: str, manifest_id: str, reason: str, category: str) -> None:
        if not manifest_id:
            return
        reasons, categories = index.setdefault((kind, str(manifest_id)), (set(), set()))
        reasons.add(reason)
        categories.add(category)

    # Downstream: evals pin their model artifact + training run.
    for row in _project_rows(
        lake, "evaluation_runs", ("eval_run_id", "model_artifact_id", "training_run_id")
    ):
        eval_id = str(row.get("eval_run_id") or "")
        note("model-artifact", str(row.get("model_artifact_id") or ""), f"evaluation-run:{eval_id}", "downstream")
        note("training-run", str(row.get("training_run_id") or ""), f"evaluation-run:{eval_id}", "downstream")
    # Downstream: model artifacts pin their training run.
    for row in _project_rows(lake, "model_artifacts", ("model_artifact_id", "training_run_id")):
        model_id = str(row.get("model_artifact_id") or "")
        note("training-run", str(row.get("training_run_id") or ""), f"model-artifact:{model_id}", "downstream")
    # Feedback loop: eval metric catalog references (backlog 0095).
    for row in _project_rows(
        lake, "eval_metric_catalog", ("evaluation_run_id", "training_run_id")
    ):
        note("evaluation-run", str(row.get("evaluation_run_id") or ""), "eval-metric-catalog", "feedback")
        note("training-run", str(row.get("training_run_id") or ""), "eval-metric-catalog", "feedback")
    # Snapshot retention: a run whose dataset_id is still a live snapshot.
    live_datasets = {
        str(row.get("dataset_id") or "")
        for row in _project_rows(lake, "dataset_snapshots", ("dataset_id",))
        if row.get("dataset_id")
    }
    for table, kind in (("training_runs", "training-run"), ("evaluation_runs", "evaluation-run")):
        id_column = _MANIFEST_ID_COLUMN[table]
        for row in _project_rows(lake, table, (id_column, "dataset_id")):
            if str(row.get("dataset_id") or "") in live_datasets:
                note(kind, str(row.get(id_column) or ""), f"snapshot:{row.get('dataset_id')}", "snapshot")
    # Lineage: a manifest whose lineage artifact is referenced by any edge.
    referenced = _lineage_referenced_artifact_ids(lake)
    if referenced:
        artifact_id_for = {
            "training-run": training_run_artifact_id,
            "model-artifact": model_artifact_lineage_id,
            "evaluation-run": evaluation_run_artifact_id,
        }
        for table, kind in (
            ("training_runs", "training-run"),
            ("model_artifacts", "model-artifact"),
            ("evaluation_runs", "evaluation-run"),
        ):
            id_column = _MANIFEST_ID_COLUMN[table]
            for row in _project_rows(lake, table, (id_column,)):
                manifest_id = str(row.get(id_column) or "")
                if manifest_id and artifact_id_for[kind](manifest_id) in referenced:
                    note(kind, manifest_id, "lineage-edge", "lineage")
    # Training reports (backlog 0115): a report is protected while the training
    # run it describes still exists -- do not GC the observability record of a
    # live run. Reports of expired/unknown runs are freely deletable.
    live_runs = {
        str(row.get("training_run_id") or "")
        for row in _project_rows(lake, "training_runs", ("training_run_id",))
        if row.get("training_run_id")
    }
    for row in _project_rows(
        lake, TRAINING_REPORTS_TABLE, ("report_id", "training_run_id")
    ):
        run_id = str(row.get("training_run_id") or "")
        if run_id and run_id in live_runs:
            note("training-report", str(row.get("report_id") or ""), f"training-run:{run_id}", "run")
    return index


def _manifest_row(lake: Lake, table: str, manifest_id: str) -> dict[str, Any] | None:
    id_column = _MANIFEST_ID_COLUMN[table]
    rows, _plan = _bounded_rows(
        lake, table, predicates=[_Predicate(id_column, "=", str(manifest_id))], limit=1
    )
    return rows[0] if rows else None


def manifest_protection(lake: Lake, *, kind: str, manifest_id: str) -> ManifestProtection:
    """Report whether a single manifest row is protected against expiry, and why."""
    if kind not in _MANIFEST_KIND_TABLE:
        raise RunManifestError(
            f"unknown manifest kind {kind!r}; expected one of {sorted(_MANIFEST_KIND_TABLE)}"
        )
    table = _MANIFEST_KIND_TABLE[kind]
    row = _manifest_row(lake, table, manifest_id)
    if row is None:
        raise RunManifestError(f"unknown {_MANIFEST_ID_COLUMN[table]} {manifest_id!r}")
    reasons, categories = _protection_index(lake).get((kind, str(manifest_id)), (set(), set()))
    return ManifestProtection(
        kind=kind,
        manifest_id=str(manifest_id),
        protected=bool(reasons),
        reasons=tuple(sorted(reasons)),
        categories=tuple(sorted(categories)),
        created_at=row.get("created_at") if isinstance(row.get("created_at"), datetime) else None,
    )


def _retention_cutoff(older_than: timedelta | datetime | None) -> datetime | None:
    if older_than is None:
        return None
    if isinstance(older_than, timedelta):
        return datetime.now(UTC) - older_than
    return older_than


def plan_manifest_retention(
    lake: Lake,
    *,
    kinds: Sequence[str] | None = None,
    older_than: timedelta | datetime | None = None,
) -> ManifestRetentionPlan:
    """Report which manifest rows are protected (lineage/feedback/snapshot/downstream).

    A row is protected when anything still references it; otherwise it is
    reported as deletable (subject to ``older_than``, when set). Protection wins
    over age -- a referenced row is never listed deletable. Reporting only; it
    performs no deletes (see :func:`delete_manifest`).
    """
    selected = tuple(kinds) if kinds is not None else tuple(_MANIFEST_KIND_TABLE)
    unknown = [kind for kind in selected if kind not in _MANIFEST_KIND_TABLE]
    if unknown:
        raise RunManifestError(f"unknown manifest kind(s): {unknown}")
    index = _protection_index(lake)
    cutoff = _retention_cutoff(older_than)
    protected: list[ManifestProtection] = []
    deletable: list[ManifestProtection] = []
    for kind in selected:
        table = _MANIFEST_KIND_TABLE[kind]
        id_column = _MANIFEST_ID_COLUMN[table]
        for row in _project_rows(lake, table, (id_column, "created_at")):
            manifest_id = str(row.get(id_column) or "")
            if not manifest_id:
                continue
            created = row.get("created_at") if isinstance(row.get("created_at"), datetime) else None
            reasons, categories = index.get((kind, manifest_id), (set(), set()))
            entry = ManifestProtection(
                kind=kind,
                manifest_id=manifest_id,
                protected=bool(reasons),
                reasons=tuple(sorted(reasons)),
                categories=tuple(sorted(categories)),
                created_at=created,
            )
            if reasons:
                protected.append(entry)
            elif cutoff is None or (created is not None and created < cutoff):
                deletable.append(entry)
    key = lambda item: (item.kind, item.manifest_id)  # noqa: E731 - local sort key
    return ManifestRetentionPlan(
        lake_uri=lake.uri,
        older_than=cutoff.isoformat() if cutoff else None,
        protected=tuple(sorted(protected, key=key)),
        deletable=tuple(sorted(deletable, key=key)),
    )


def delete_manifest(
    lake: Lake,
    *,
    kind: str,
    manifest_id: str,
    force: bool = False,
) -> dict[str, Any]:
    """Delete a manifest row, refusing protected rows unless ``force`` is set.

    A checkpoint referenced by an evaluation row or a lineage edge (or a run
    still pinned by a live snapshot) is protected; deleting it requires
    ``force=True``. Deleting an evaluation run also drops its materialized
    ``evaluation_run_metrics`` rows.
    """
    protection = manifest_protection(lake, kind=kind, manifest_id=manifest_id)
    if protection.protected and not force:
        raise RunManifestError(
            f"{kind} {manifest_id!r} is protected by {', '.join(protection.reasons)}; "
            "pass force=True to delete"
        )
    table = _MANIFEST_KIND_TABLE[kind]
    id_column = _MANIFEST_ID_COLUMN[table]
    lake.table(table).delete(f"{id_column} = {_sql_literal(str(manifest_id))}")
    if kind == "evaluation-run" and EVALUATION_RUN_METRICS_TABLE in lake.table_names():
        lake.table(EVALUATION_RUN_METRICS_TABLE).delete(
            f"eval_run_id = {_sql_literal(str(manifest_id))}"
        )
    return {
        "kind": kind,
        "manifest_id": str(manifest_id),
        "deleted": True,
        "forced": bool(force),
        "protection": protection.to_dict(),
    }


# ---------------------------------------------------------------------------
# Training loader/backend report catalog (backlog 0115).
#
# The 0069/0073 Enterprise training path already produces an in-memory,
# secret-free ``TrainingLoaderReport`` (and its lower-level ``backend`` report)
# per loader run. Those are inspectable for one run but vanish when the process
# exits. This catalog persists them as durable, queryable run history so
# platform teams can compare cold/warm epochs, audit fallbacks, and reload the
# exact report for a run/epoch/worker without re-running the loader.
#
# Design: one ``training_reports`` row per report, idempotent by content digest
# (``report_id == "trpt-<digest>"``); scalar columns carry the queryable
# dimensions; ``report_json``/``backend_json`` carry the full payloads for point
# reload; aggregatable counters are materialized so cross-worker/epoch sums are
# a bounded projection, not a payload scan. This is a *derived* projection of
# reports emitted elsewhere -- never a second source of truth for the
# in-memory ``dataset.manifest.backend`` shape, which stays readable as-is.
# ---------------------------------------------------------------------------

TRAINING_REPORT_CATALOG_SCHEMA_VERSION = "lancedb-robotics/training-report-catalog/v1"
_TRAINING_LOADER_REPORT_KIND = "lancedb-robotics/training-loader-report/v1"

# Summary listing projection: everything except the large JSON payload columns,
# so ``reports()`` never pulls full report bodies into memory.
_REPORT_SUMMARY_COLUMNS = (
    "report_id",
    "report_digest",
    "loader_kind",
    "training_run_id",
    "model_run_id",
    "model_id",
    "dataset_id",
    "snapshot_name",
    "alignment_id",
    "alignment_name",
    "row_plan_id",
    "tick_plan_id",
    "epoch_plan_id",
    "epoch",
    "worker_id",
    "num_workers",
    "requested_backend",
    "resolved_backend",
    "connection_kind",
    "execution_mode",
    "cache_policy",
    "prewarm_requested",
    "prewarm_status",
    "fallback",
    "fallback_reason",
    "fallback_from_backend",
    "fallback_to_backend",
    "cache_hits",
    "cache_misses",
    "bytes_read",
    "rows_hydrated",
    "pe_fanout",
    "status",
    "created_by",
    "created_at",
)

_REPORT_METRIC_COLUMNS = ("cache_hits", "cache_misses", "bytes_read", "rows_hydrated", "pe_fanout")


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    coerced = _int_or_none(value)
    return coerced if coerced is not None else 0


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


@dataclass(frozen=True)
class TrainingReportRecord:
    """One durable training-report catalog entry, with full payloads reloaded."""

    report_id: str
    report_digest: str
    lake_uri: str
    loader_kind: str | None
    training_run_id: str | None
    model_run_id: str | None
    model_id: str | None
    dataset_id: str | None
    snapshot_name: str | None
    alignment_id: str | None
    alignment_name: str | None
    row_plan_id: str | None
    tick_plan_id: str | None
    epoch_plan_id: str | None
    epoch: int | None
    worker_id: int | None
    num_workers: int | None
    requested_backend: str | None
    resolved_backend: str | None
    connection_kind: str | None
    execution_mode: str | None
    cache_policy: str | None
    prewarm_requested: bool
    prewarm_status: str | None
    fallback: bool
    fallback_reason: str | None
    fallback_from_backend: str | None
    fallback_to_backend: str | None
    cache_hits: int
    cache_misses: int
    bytes_read: int
    rows_hydrated: int
    pe_fanout: int
    status: str
    created_by: str
    created_at: datetime | None
    report: dict[str, Any]
    backend: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Scalar identity/metric fields only -- no full payloads."""
        return {
            "report_id": self.report_id,
            "report_digest": self.report_digest,
            "lake_uri": self.lake_uri,
            "loader_kind": self.loader_kind,
            "training_run_id": self.training_run_id,
            "model_run_id": self.model_run_id,
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "snapshot_name": self.snapshot_name,
            "alignment_id": self.alignment_id,
            "alignment_name": self.alignment_name,
            "row_plan_id": self.row_plan_id,
            "tick_plan_id": self.tick_plan_id,
            "epoch_plan_id": self.epoch_plan_id,
            "epoch": self.epoch,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "connection_kind": self.connection_kind,
            "execution_mode": self.execution_mode,
            "cache_policy": self.cache_policy,
            "prewarm_requested": self.prewarm_requested,
            "prewarm_status": self.prewarm_status,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "fallback_from_backend": self.fallback_from_backend,
            "fallback_to_backend": self.fallback_to_backend,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "bytes_read": self.bytes_read,
            "rows_hydrated": self.rows_hydrated,
            "pe_fanout": self.pe_fanout,
            "status": self.status,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.summary()
        payload["report"] = dict(self.report)
        payload["backend"] = dict(self.backend)
        return payload


def _redact_backend_report(backend: Mapping[str, Any]) -> dict[str, Any]:
    """Defensively strip any secret-shaped keys from the raw backend report."""
    from lancedb_robotics.training import _redact_report  # lazy: avoid import cycle

    redacted = _redact_report(_jsonable(dict(backend)))
    return redacted if isinstance(redacted, dict) else {}


def _coerce_training_report(
    report: Any | None,
    *,
    dataset: Any | None,
    backend: Mapping[str, Any] | None,
    training_run_id: str | None,
    model_run_id: str | None,
    model_id: str | None,
    extra: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the (redacted report payload, backend dict) to persist.

    Accepts a training dataset (pulls ``loader_report`` + ``manifest.backend``),
    a ``TrainingLoaderReport`` (or any object exposing ``to_dict``), or a plain
    payload mapping. Run hooks (training/model run ids) are overlaid onto the
    report's ``run`` block so they are captured in identity and digest.
    """
    backend_dict: dict[str, Any] = {}
    if dataset is not None:
        loader = dataset.loader_report(
            training_run_id=training_run_id,
            model_run_id=model_run_id,
            model_id=model_id,
            extra=extra,
        )
        payload = loader.to_dict() if hasattr(loader, "to_dict") else dict(loader)
        manifest = getattr(dataset, "manifest", None)
        backend_dict = _json_object(getattr(manifest, "backend", None))
    elif report is not None:
        if hasattr(report, "to_dict"):
            payload = report.to_dict()
        elif isinstance(report, Mapping):
            payload = dict(report)
        else:
            raise RunManifestError(
                "report must be a TrainingLoaderReport, a mapping, or provide dataset="
            )
        backend_dict = _json_object(backend) if backend is not None else {}
    else:
        raise RunManifestError("record_training_report requires report= or dataset=")

    if not isinstance(payload, Mapping):
        raise RunManifestError("training report payload must be a JSON object")
    payload = dict(payload)

    # Overlay run hooks so a directly-passed report still records identity.
    run = _json_object(payload.get("run"))
    if training_run_id is not None:
        run["training_run_id"] = str(training_run_id)
    if model_run_id is not None:
        run["model_run_id"] = str(model_run_id)
    if model_id is not None:
        run["model_id"] = str(model_id)
    if extra:
        for key, value in extra.items():
            run.setdefault(str(key), value)
    if run:
        payload["run"] = run

    if not backend_dict:
        # Fall back to the report's own remote-execution projection so a full
        # backend reload still has something meaningful when the raw backend
        # report was not supplied.
        backend_dict = _json_object(payload.get("remote_execution"))
    backend_dict = _redact_backend_report(backend_dict) if backend_dict else {}
    return payload, backend_dict


def _report_scalars(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the queryable/aggregatable scalar columns from a report payload."""
    loader = _json_object(payload.get("loader"))
    snapshot = _json_object(payload.get("snapshot"))
    alignment = _json_object(payload.get("alignment"))
    plans = _json_object(payload.get("plans"))
    worker = _json_object(plans.get("worker"))
    remote = _json_object(payload.get("remote_execution"))
    cache_cfg = _json_object(remote.get("cache"))
    metrics = _json_object(payload.get("metrics"))
    summary = _json_object(metrics.get("summary"))
    ops_by_type = _json_object(metrics.get("operations_by_type"))
    cache_metrics = _json_object(metrics.get("cache"))
    events = [e for e in (payload.get("fallback_events") or []) if isinstance(e, Mapping)]
    first = events[0] if events else {}
    prewarm_status = _str_or_none(summary.get("prewarm_status"))
    prewarm_requested = bool(
        _int_or_zero(ops_by_type.get("prewarm"))
        or (prewarm_status is not None and prewarm_status != "not-requested")
        or summary.get("prewarm_policy") is not None
    )
    return {
        "loader_kind": _str_or_none(loader.get("kind")),
        "dataset_id": _str_or_none(snapshot.get("id")),
        "snapshot_name": _str_or_none(snapshot.get("name")),
        "alignment_id": _str_or_none(alignment.get("id")),
        "alignment_name": _str_or_none(alignment.get("name")),
        "row_plan_id": _str_or_none(plans.get("row_plan_id")),
        "tick_plan_id": _str_or_none(plans.get("tick_plan_id")),
        "epoch_plan_id": _str_or_none(plans.get("epoch_plan_id")),
        "epoch": _int_or_none(plans.get("epoch")),
        "worker_id": _int_or_none(worker.get("id")),
        "num_workers": _int_or_none(worker.get("num_workers")),
        "requested_backend": _str_or_none(remote.get("requested_backend")),
        "resolved_backend": _str_or_none(remote.get("resolved_backend")),
        "connection_kind": _str_or_none(remote.get("connection_kind")),
        "execution_mode": _str_or_none(remote.get("execution_mode")),
        "cache_policy": _str_or_none(cache_cfg.get("policy")),
        "prewarm_requested": prewarm_requested,
        "prewarm_status": prewarm_status,
        "fallback": bool(events),
        "fallback_reason": _str_or_none(first.get("reason")),
        "fallback_from_backend": _str_or_none(first.get("from")),
        "fallback_to_backend": _str_or_none(first.get("to")),
        "cache_hits": _int_or_zero(cache_metrics.get("hits")),
        "cache_misses": _int_or_zero(cache_metrics.get("misses")),
        "bytes_read": _int_or_zero(summary.get("bytes_read")),
        "rows_hydrated": _int_or_zero(summary.get("rows_returned")),
        "pe_fanout": _int_or_zero(summary.get("pe_fanout")),
        "table_versions": tuple(
            normalize_table_versions(payload.get("table_versions") or ())
        ),
    }


def record_training_report(
    lake: Lake,
    report: Any | None = None,
    *,
    dataset: Any | None = None,
    backend: Mapping[str, Any] | None = None,
    training_run_id: str | None = None,
    model_run_id: str | None = None,
    model_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
    status: str = "recorded",
    metadata: Mapping[str, Any] | None = None,
    created_by: str = "lancedb-robotics",
) -> TrainingReportRecord:
    """Persist one Enterprise training loader/backend report, idempotent by content.

    Pass ``dataset=`` (a ``LanceTrainingDataset``/``AlignedFrameTrainingDataset``;
    its ``loader_report`` and ``manifest.backend`` are read) or a ``report=``
    (a ``TrainingLoaderReport`` or payload mapping, optionally with ``backend=``).
    Recording the same report twice is a no-op replace; two runs/epochs/workers
    with different content produce distinct rows with stable digests.
    """
    payload, backend_dict = _coerce_training_report(
        report,
        dataset=dataset,
        backend=backend,
        training_run_id=training_run_id,
        model_run_id=model_run_id,
        model_id=model_id,
        extra=extra,
    )
    scalars = _report_scalars(payload)
    run = _json_object(payload.get("run"))
    report_digest = _digest({"report": payload, "backend": backend_dict})
    report_id = f"trpt-{report_digest}"
    now = datetime.now(UTC)
    row = {
        "report_id": report_id,
        "report_digest": report_digest,
        "catalog_schema_version": TRAINING_REPORT_CATALOG_SCHEMA_VERSION,
        "report_schema_version": str(payload.get("kind") or _TRAINING_LOADER_REPORT_KIND),
        "lake_uri": lake.uri,
        "loader_kind": scalars["loader_kind"],
        "training_run_id": _str_or_none(run.get("training_run_id")),
        "model_run_id": _str_or_none(run.get("model_run_id")),
        "model_id": _str_or_none(run.get("model_id")),
        "dataset_id": scalars["dataset_id"],
        "snapshot_name": scalars["snapshot_name"],
        "alignment_id": scalars["alignment_id"],
        "alignment_name": scalars["alignment_name"],
        "table_versions": [dict(item) for item in scalars["table_versions"]],
        "row_plan_id": scalars["row_plan_id"],
        "tick_plan_id": scalars["tick_plan_id"],
        "epoch_plan_id": scalars["epoch_plan_id"],
        "epoch": scalars["epoch"],
        "worker_id": scalars["worker_id"],
        "num_workers": scalars["num_workers"],
        "requested_backend": scalars["requested_backend"],
        "resolved_backend": scalars["resolved_backend"],
        "connection_kind": scalars["connection_kind"],
        "execution_mode": scalars["execution_mode"],
        "cache_policy": scalars["cache_policy"],
        "prewarm_requested": scalars["prewarm_requested"],
        "prewarm_status": scalars["prewarm_status"],
        "fallback": scalars["fallback"],
        "fallback_reason": scalars["fallback_reason"],
        "fallback_from_backend": scalars["fallback_from_backend"],
        "fallback_to_backend": scalars["fallback_to_backend"],
        "cache_hits": scalars["cache_hits"],
        "cache_misses": scalars["cache_misses"],
        "bytes_read": scalars["bytes_read"],
        "rows_hydrated": scalars["rows_hydrated"],
        "pe_fanout": scalars["pe_fanout"],
        "report_json": _json_dumps(payload),
        "backend_json": _json_dumps(backend_dict),
        "status": str(status),
        "metadata": _kv_items(metadata),
        "created_by": created_by,
        "created_at": now,
    }
    _replace_rows(lake, TRAINING_REPORTS_TABLE, "report_id", [row], TRAINING_REPORTS_SCHEMA)
    return _report_record_from_row(row, report=payload, backend=backend_dict)


def _report_record_from_row(
    row: Mapping[str, Any],
    *,
    report: Mapping[str, Any] | None = None,
    backend: Mapping[str, Any] | None = None,
) -> TrainingReportRecord:
    if report is None:
        report = _json_object(row.get("report_json"))
    if backend is None:
        backend = _json_object(row.get("backend_json"))
    created_at = row.get("created_at")
    return TrainingReportRecord(
        report_id=str(row.get("report_id") or ""),
        report_digest=str(row.get("report_digest") or ""),
        lake_uri=str(row.get("lake_uri") or ""),
        loader_kind=_str_or_none(row.get("loader_kind")),
        training_run_id=_str_or_none(row.get("training_run_id")),
        model_run_id=_str_or_none(row.get("model_run_id")),
        model_id=_str_or_none(row.get("model_id")),
        dataset_id=_str_or_none(row.get("dataset_id")),
        snapshot_name=_str_or_none(row.get("snapshot_name")),
        alignment_id=_str_or_none(row.get("alignment_id")),
        alignment_name=_str_or_none(row.get("alignment_name")),
        row_plan_id=_str_or_none(row.get("row_plan_id")),
        tick_plan_id=_str_or_none(row.get("tick_plan_id")),
        epoch_plan_id=_str_or_none(row.get("epoch_plan_id")),
        epoch=_int_or_none(row.get("epoch")),
        worker_id=_int_or_none(row.get("worker_id")),
        num_workers=_int_or_none(row.get("num_workers")),
        requested_backend=_str_or_none(row.get("requested_backend")),
        resolved_backend=_str_or_none(row.get("resolved_backend")),
        connection_kind=_str_or_none(row.get("connection_kind")),
        execution_mode=_str_or_none(row.get("execution_mode")),
        cache_policy=_str_or_none(row.get("cache_policy")),
        prewarm_requested=bool(row.get("prewarm_requested")),
        prewarm_status=_str_or_none(row.get("prewarm_status")),
        fallback=bool(row.get("fallback")),
        fallback_reason=_str_or_none(row.get("fallback_reason")),
        fallback_from_backend=_str_or_none(row.get("fallback_from_backend")),
        fallback_to_backend=_str_or_none(row.get("fallback_to_backend")),
        cache_hits=_int_or_zero(row.get("cache_hits")),
        cache_misses=_int_or_zero(row.get("cache_misses")),
        bytes_read=_int_or_zero(row.get("bytes_read")),
        rows_hydrated=_int_or_zero(row.get("rows_hydrated")),
        pe_fanout=_int_or_zero(row.get("pe_fanout")),
        status=str(row.get("status") or ""),
        created_by=str(row.get("created_by") or ""),
        created_at=created_at if isinstance(created_at, datetime) else None,
        report=dict(report),
        backend=dict(backend),
    )


def _report_predicates(
    *,
    training_run_id: str | None,
    model_run_id: str | None,
    dataset_id: str | None,
    resolved_backend: str | None,
    requested_backend: str | None,
    connection_kind: str | None,
    cache_policy: str | None,
    fallback: bool | None,
    fallback_reason: str | None,
    fallback_to_backend: str | None,
    loader_kind: str | None,
    epoch: int | None,
    worker_id: int | None,
) -> list[_Predicate]:
    predicates: list[_Predicate] = []
    scalar = {
        "training_run_id": training_run_id,
        "model_run_id": model_run_id,
        "dataset_id": dataset_id,
        "resolved_backend": resolved_backend,
        "requested_backend": requested_backend,
        "connection_kind": connection_kind,
        "cache_policy": cache_policy,
        "fallback_reason": fallback_reason,
        "fallback_to_backend": fallback_to_backend,
        "loader_kind": loader_kind,
    }
    for column, value in scalar.items():
        if value is not None:
            predicates.append(_Predicate(column, "=", str(value)))
    if fallback is not None:
        predicates.append(_Predicate("fallback", "=", bool(fallback)))
    if epoch is not None:
        predicates.append(_Predicate("epoch", "=", int(epoch)))
    if worker_id is not None:
        predicates.append(_Predicate("worker_id", "=", int(worker_id)))
    return predicates


def query_training_reports(
    lake: Lake,
    *,
    training_run_id: str | None = None,
    model_run_id: str | None = None,
    dataset_id: str | None = None,
    snapshot: str | None = None,
    resolved_backend: str | None = None,
    requested_backend: str | None = None,
    connection_kind: str | None = None,
    cache_policy: str | None = None,
    fallback: bool | None = None,
    fallback_reason: str | None = None,
    fallback_to_backend: str | None = None,
    loader_kind: str | None = None,
    epoch: int | None = None,
    worker_id: int | None = None,
    limit: int | None = None,
) -> ManifestQueryResult:
    """Bounded query over ``training_reports`` by run/snapshot/backend/fallback.

    Scalar predicates push down to a SQL ``WHERE``; the summary projection keeps
    the full JSON payload columns out of the scan. Ordered by (created_at, id).
    """
    if snapshot is not None:
        dataset_id = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)["dataset_id"]
    predicates = _report_predicates(
        training_run_id=training_run_id,
        model_run_id=model_run_id,
        dataset_id=dataset_id,
        resolved_backend=resolved_backend,
        requested_backend=requested_backend,
        connection_kind=connection_kind,
        cache_policy=cache_policy,
        fallback=fallback,
        fallback_reason=fallback_reason,
        fallback_to_backend=fallback_to_backend,
        loader_kind=loader_kind,
        epoch=epoch,
        worker_id=worker_id,
    )
    rows, plan = _bounded_rows(
        lake,
        TRAINING_REPORTS_TABLE,
        predicates=predicates,
        columns=list(_REPORT_SUMMARY_COLUMNS),
        limit=limit,
    )
    return ManifestQueryResult(
        TRAINING_REPORTS_TABLE,
        tuple(_sorted_by_created(rows, "report_id")),
        plan,
    )


def list_training_reports(
    lake: Lake,
    *,
    page_size: int = _DEFAULT_PAGE_SIZE,
    page_token: str | None = None,
    training_run_id: str | None = None,
    dataset_id: str | None = None,
    snapshot: str | None = None,
    resolved_backend: str | None = None,
    cache_policy: str | None = None,
    fallback: bool | None = None,
    fallback_reason: str | None = None,
    loader_kind: str | None = None,
) -> ManifestPage:
    """Deterministic paged summary listing of ``training_reports`` (report history).

    Projects summary columns only -- the full report/backend JSON bodies are
    never materialized here (reload one with :func:`get_training_report`).
    """
    if snapshot is not None:
        dataset_id = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)["dataset_id"]
    predicates = _report_predicates(
        training_run_id=training_run_id,
        model_run_id=None,
        dataset_id=dataset_id,
        resolved_backend=resolved_backend,
        requested_backend=None,
        connection_kind=None,
        cache_policy=cache_policy,
        fallback=fallback,
        fallback_reason=fallback_reason,
        fallback_to_backend=None,
        loader_kind=loader_kind,
        epoch=None,
        worker_id=None,
    )
    filters = {
        "training_run_id": training_run_id,
        "dataset_id": dataset_id,
        "resolved_backend": resolved_backend,
        "cache_policy": cache_policy,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "loader_kind": loader_kind,
    }
    return _paginate(
        lake,
        TRAINING_REPORTS_TABLE,
        "report_id",
        predicates=predicates,
        filters=filters,
        page_size=page_size,
        page_token=page_token,
        columns=list(_REPORT_SUMMARY_COLUMNS),
    )


def get_training_report(
    lake: Lake,
    *,
    report_id: str | None = None,
    training_run_id: str | None = None,
    epoch: int | None = None,
    worker_id: int | None = None,
) -> TrainingReportRecord:
    """Reload one full report (payload + backend) by id or by run/epoch/worker.

    A point ``report_id`` lookup wins. Otherwise the (training_run_id, epoch,
    worker_id) handle selects the most recent matching report; ``epoch`` and
    ``worker_id`` narrow the match when several reports share a run.
    """
    if report_id is not None:
        predicates = [_Predicate("report_id", "=", str(report_id))]
    elif training_run_id is not None:
        predicates = [_Predicate("training_run_id", "=", str(training_run_id))]
        if epoch is not None:
            predicates.append(_Predicate("epoch", "=", int(epoch)))
        if worker_id is not None:
            predicates.append(_Predicate("worker_id", "=", int(worker_id)))
    else:
        raise RunManifestError("get_training_report requires report_id= or training_run_id=")
    rows, _plan = _bounded_rows(lake, TRAINING_REPORTS_TABLE, predicates=predicates)
    if not rows:
        handle = report_id or training_run_id
        raise RunManifestError(f"no training report found for {handle!r}")
    # Most recent first when a handle matches several report rows.
    latest = _sorted_by_created(rows, "report_id")[-1]
    return _report_record_from_row(latest)


@dataclass(frozen=True)
class TrainingReportMetricsAggregate:
    """Summed loader telemetry across matched report rows (workers x epochs)."""

    lake_uri: str
    report_count: int
    totals: dict[str, int]
    by_worker: dict[str, dict[str, int]]
    by_epoch: dict[str, dict[str, int]]
    plan: ManifestQueryPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "report_count": self.report_count,
            "totals": dict(self.totals),
            "by_worker": {k: dict(v) for k, v in self.by_worker.items()},
            "by_epoch": {k: dict(v) for k, v in self.by_epoch.items()},
            "plan": self.plan.to_dict(),
        }


def aggregate_training_report_metrics(
    lake: Lake,
    *,
    training_run_id: str | None = None,
    dataset_id: str | None = None,
    snapshot: str | None = None,
    resolved_backend: str | None = None,
) -> TrainingReportMetricsAggregate:
    """Sum cache hits/misses, bytes read, and PE fanout across matched reports.

    Reads only the metric + grouping columns (a bounded projection, never the
    JSON payloads) and reduces in Python; also breaks totals down per worker and
    per epoch for cold/warm comparison.
    """
    if snapshot is not None:
        dataset_id = _resolve_snapshot(lake, snapshot=snapshot, dataset_id=dataset_id)["dataset_id"]
    predicates: list[_Predicate] = []
    if training_run_id is not None:
        predicates.append(_Predicate("training_run_id", "=", str(training_run_id)))
    if dataset_id is not None:
        predicates.append(_Predicate("dataset_id", "=", str(dataset_id)))
    if resolved_backend is not None:
        predicates.append(_Predicate("resolved_backend", "=", str(resolved_backend)))
    columns = [*_REPORT_METRIC_COLUMNS, "worker_id", "num_workers", "epoch", "report_id", "created_at"]
    rows, plan = _bounded_rows(lake, TRAINING_REPORTS_TABLE, predicates=predicates, columns=columns)

    totals = {metric: 0 for metric in _REPORT_METRIC_COLUMNS}
    by_worker: dict[str, dict[str, int]] = {}
    by_epoch: dict[str, dict[str, int]] = {}

    def _accumulate(bucket: dict[str, int], row: Mapping[str, Any]) -> None:
        for metric in _REPORT_METRIC_COLUMNS:
            bucket[metric] = bucket.get(metric, 0) + _int_or_zero(row.get(metric))

    for row in rows:
        for metric in _REPORT_METRIC_COLUMNS:
            totals[metric] += _int_or_zero(row.get(metric))
        worker_id = row.get("worker_id")
        num_workers = row.get("num_workers")
        if worker_id is not None:
            key = f"{worker_id}/{num_workers}" if num_workers is not None else str(worker_id)
            _accumulate(by_worker.setdefault(key, {}), row)
        if row.get("epoch") is not None:
            _accumulate(by_epoch.setdefault(str(row.get("epoch")), {}), row)

    return TrainingReportMetricsAggregate(
        lake_uri=lake.uri,
        report_count=len(rows),
        totals=totals,
        by_worker=by_worker,
        by_epoch=by_epoch,
        plan=plan,
    )
