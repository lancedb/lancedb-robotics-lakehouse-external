"""`lancedb-robotics train` subcommands."""

import json
from pathlib import Path

import typer

train_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
preview_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
remote_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
tracker_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
report_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
conformance_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
plan_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
permutation_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
prewarm_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
warm_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
train_app.add_typer(preview_app, name="preview", help="Preview a snapshot as a training dataset.")
train_app.add_typer(remote_app, name="remote", help="Inspect Enterprise remote training setup.")
train_app.add_typer(
    plan_app,
    name="plan",
    help="Build and page Enterprise server-side row-plan artifacts.",
)
train_app.add_typer(
    tracker_app,
    name="tracker",
    help="Sync manifests with external experiment trackers (import/export/drift).",
)
train_app.add_typer(
    report_app,
    name="report",
    help="Query persisted Enterprise training loader/backend report history.",
)
train_app.add_typer(
    conformance_app,
    name="conformance",
    help="Enterprise training conformance matrix and fault-injection run.",
)
train_app.add_typer(
    permutation_app,
    name="permutation",
    help="Back a snapshot plan with LanceDB's native Permutation reader (projection + tensor formats).",
)
train_app.add_typer(
    prewarm_app,
    name="prewarm",
    help="Inspect, retry, and cancel durable Enterprise cache-prewarm JobRuns.",
)
train_app.add_typer(
    warm_app,
    name="warm",
    help="Query-driven cache warming: warm exactly what training reads via queries.",
)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_SNAPSHOT_OPTION = typer.Option(..., "--snapshot", help="Snapshot name to preview.")
_COLUMNS_OPTION = typer.Option(None, "--columns", help="Comma-separated sample fields to project.")
_BATCH_SIZE_OPTION = typer.Option(4, "--batch-size", help="Number of samples to preview.")
_REMOTE_AUTH_REF_OPTION = typer.Option(
    None,
    "--remote-auth-ref",
    help="LanceDB Enterprise API credential reference; resolves at runtime only.",
)
_REGION_OPTION = typer.Option(None, "--region", help="LanceDB Enterprise region.")
_HOST_OVERRIDE_OPTION = typer.Option(
    None,
    "--host-override",
    help="LanceDB Enterprise private endpoint override.",
)
_BACKEND_OPTION = typer.Option(
    "enterprise",
    "--backend",
    help="Training backend to inspect: auto, local, or enterprise.",
)
_CACHE_POLICY_OPTION = typer.Option(
    "none",
    "--cache-policy",
    help="Plan-executor cache policy to record: none, lazy, epoch, or snapshot.",
)
_PREWARM_OPTION = typer.Option(
    False,
    "--prewarm/--no-prewarm",
    help="Compatibility flag for cache prewarm intent; epoch/snapshot policies request prewarm.",
)
_PREWARM_INCLUDE_HEAVY_OPTION = typer.Option(
    False,
    "--prewarm-include-heavy/--no-prewarm-include-heavy",
    help="Allow payload/blob columns in Enterprise cache prewarm requests.",
)
_PREWARM_MAX_ROWS_OPTION = typer.Option(
    None,
    "--prewarm-max-rows",
    help="Maximum rows allowed in a prewarm request.",
)
_PREWARM_TIMEOUT_OPTION = typer.Option(
    None,
    "--prewarm-timeout-s",
    help="Timeout seconds recorded for prewarm status polling.",
)
_ALLOW_FALLBACK_OPTION = typer.Option(
    False,
    "--allow-fallback",
    help="Allow a non-Enterprise lake to run locally with an explicit fallback report.",
)
_FALLBACK_OPTION = typer.Option(
    None,
    "--fallback",
    help="Enterprise fallback policy: fail, warn, direct, or local.",
)
_FORMAT_OPTION = typer.Option(
    "text",
    "--format",
    help="Output format: text or json.",
)
_REPORT_OUT_OPTION = typer.Option(
    None,
    "--report-out",
    help="Write the structured training loader report JSON to this path.",
)
_PLAN_OUT_OPTION = typer.Option(
    None,
    "--out",
    help="Write the plan handle JSON to this path.",
)
_PLAN_HANDLE_OPTION = typer.Option(
    ...,
    "--handle",
    help="Path to a plan handle JSON written by `train plan build`.",
)


def _format_value(value) -> str:
    if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
        head = ", ".join(f"{x:.3f}" for x in value[:4])
        suffix = f", ... ({len(value)} dims)" if len(value) > 4 else ""
        return f"[{head}{suffix}]"
    return repr(value)


@preview_app.command("torch")
def torch_preview(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    columns: str = _COLUMNS_OPTION,
    batch_size: int = _BATCH_SIZE_OPTION,
) -> None:
    """Preview a dataset snapshot as deterministic (optionally torch) samples."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.training import (
        TrainingError,
        load_snapshot_preview,
        to_torch_dataset,
        torch_available,
    )

    try:
        opened = Lake.open(lake)
        selected = [c.strip() for c in columns.split(",")] if columns else None
        preview = load_snapshot_preview(opened, snapshot, columns=selected, batch_size=batch_size)
    except (LakeError, TrainingError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"lake: {preview.lake_uri}")
    typer.echo(f"snapshot: {preview.name} ({preview.dataset_id})")
    typer.echo(f"tag: {preview.tag}")
    typer.echo(f"split by: {preview.split_by}")
    typer.echo(f"scenarios: {preview.total_scenarios}")
    typer.echo(f"columns: {', '.join(preview.columns)}")

    if torch_available():
        to_torch_dataset(preview)  # validate the tensor path is constructible
        typer.echo("framework: torch ready (tensor batches available)")
    else:
        typer.echo(
            "framework: torch not installed (showing dict preview); "
            "install lancedb-robotics[torch] for tensor batches"
        )

    for index, sample in enumerate(preview.samples, start=1):
        fields = " ".join(f"{key}={_format_value(value)}" for key, value in sample.items())
        typer.echo(f"sample {index}: {fields}")


@remote_app.command("report")
def remote_report(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    columns: str = typer.Option(
        "observation_id",
        "--columns",
        help="Comma-separated native training columns to plan.",
    ),
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
    backend: str = _BACKEND_OPTION,
    cache_policy: str = _CACHE_POLICY_OPTION,
    prewarm: bool = _PREWARM_OPTION,
    prewarm_include_heavy: bool = _PREWARM_INCLUDE_HEAVY_OPTION,
    prewarm_max_rows: int | None = _PREWARM_MAX_ROWS_OPTION,
    prewarm_timeout_s: float | None = _PREWARM_TIMEOUT_OPTION,
    allow_fallback: bool = _ALLOW_FALLBACK_OPTION,
    fallback: str | None = _FALLBACK_OPTION,
    report_out: Path | None = _REPORT_OUT_OPTION,
    output_format: str = _FORMAT_OPTION,
) -> None:
    """Report the resolved backend for a version-pinned training snapshot."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.training import TrainingError

    try:
        selected = [c.strip() for c in columns.split(",") if c.strip()]
        prewarm_options = {"include_heavy": prewarm_include_heavy}
        if prewarm_max_rows is not None:
            prewarm_options["max_rows"] = prewarm_max_rows
        if prewarm_timeout_s is not None:
            prewarm_options["timeout_s"] = prewarm_timeout_s
        opened = Lake.open(
            lake,
            remote_auth_ref=remote_auth_ref,
            region=region,
            host_override=host_override,
        )
        dataset = opened.training.dataset(
            snapshot,
            columns=selected,
            backend=backend,
            cache_policy=cache_policy,
            prewarm=prewarm,
            prewarm_options=prewarm_options,
            allow_fallback=allow_fallback,
            fallback=fallback,
        )
    except (LakeError, TrainingError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    report = dataset.manifest.backend
    loader_report = dataset.loader_report()
    loader_report_payload = loader_report.to_dict()
    if report_out is not None:
        loader_report.write_json(report_out)
    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    "lake": opened.uri,
                    "snapshot": snapshot,
                    "manifest": dataset.manifest.to_dict(),
                    "backend": report,
                    "loader_report": loader_report_payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if output_format != "text":
        typer.echo("error: --format must be text or json", err=True)
        raise typer.Exit(code=2)

    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"snapshot: {snapshot} ({dataset.manifest.dataset_id})")
    typer.echo(f"backend: {report['resolved_backend']} (requested {report['requested_backend']})")
    typer.echo(f"execution: {report['execution_mode']}")
    typer.echo(f"connection: {report['connection_kind']} {report['display_uri']}")
    endpoint = report.get("request_routing", {}).get("http_endpoint")
    if endpoint:
        typer.echo(f"http endpoint: {endpoint}")
    typer.echo(f"row plan: {dataset.row_plan.plan_id}")
    typer.echo(f"rows planned: {report['metrics']['rows_planned']}")
    typer.echo(f"cache policy: {report['cache']['policy']}")
    typer.echo(f"prewarm: {report['cache']['prewarm_status']}")
    cache = loader_report_payload["metrics"]["cache"]
    typer.echo(f"cache hits/misses: {cache['hits']}/{cache['misses']}")
    typer.echo(f"bytes read: {loader_report_payload['metrics']['summary'].get('bytes_read')}")
    if report["cache"].get("prewarm_id"):
        typer.echo(f"prewarm id: {report['cache']['prewarm_id']}")
    if report_out is not None:
        typer.echo(f"loader report: {report_out}")
    if report.get("fallback"):
        typer.echo(
            "fallback: "
            f"{report['fallback']['from']} -> {report['fallback']['to']} "
            f"({report['fallback']['reason']})"
        )
    for warning in report.get("warnings", []):
        typer.echo(f"warning: {warning}")


@plan_app.command("build")
def plan_build(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    columns: str = typer.Option(
        "observation_id",
        "--columns",
        help="Comma-separated native training columns to plan.",
    ),
    shuffle: bool = typer.Option(
        False, "--shuffle/--no-shuffle", help="Seeded global epoch shuffle."
    ),
    shuffle_seed: int = typer.Option(0, "--shuffle-seed", help="Deterministic shuffle seed."),
    epoch: int = typer.Option(0, "--epoch", help="Global epoch offset for rotation."),
    page_size: int = typer.Option(
        1024, "--page-size", help="Rows per plan page for worker handoff."
    ),
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
    backend: str = _BACKEND_OPTION,
    allow_fallback: bool = _ALLOW_FALLBACK_OPTION,
    fallback: str | None = _FALLBACK_OPTION,
    out: Path | None = _PLAN_OUT_OPTION,
    output_format: str = _FORMAT_OPTION,
) -> None:
    """Build a version-pinned server-side row-plan artifact and print its handle."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.training import ServerSidePlanError, TrainingError

    try:
        selected = [c.strip() for c in columns.split(",") if c.strip()]
        opened = Lake.open(
            lake,
            remote_auth_ref=remote_auth_ref,
            region=region,
            host_override=host_override,
        )
        handle = opened.training.server_side_row_plan(
            snapshot,
            columns=selected,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            backend=backend,
            allow_fallback=allow_fallback,
            fallback=fallback,
            page_size=page_size,
        )
    except (LakeError, TrainingError, ServerSidePlanError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if out is not None:
        out.write_text(json.dumps(handle, indent=2, sort_keys=True) + "\n")
    if output_format == "json":
        typer.echo(json.dumps(handle, indent=2, sort_keys=True))
        return
    if output_format != "text":
        typer.echo("error: --format must be text or json", err=True)
        raise typer.Exit(code=2)

    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"snapshot: {snapshot} ({handle['snapshot_id']})")
    typer.echo(f"plan handle: {handle['plan_handle_id']}")
    typer.echo(f"row plan: {handle['row_plan_id']}")
    typer.echo(f"ordering: {handle['ordering_policy']}")
    typer.echo(f"rows: {handle['total_rows']}")
    typer.echo(f"pages: {handle['num_pages']} x {handle['page_size']}")
    typer.echo(f"store: {handle['store_kind']} {handle['store_ref']}")
    typer.echo(f"display uri: {handle['display_uri']}")
    if out is not None:
        typer.echo(f"handle: {out}")


@plan_app.command("page")
def plan_page(
    lake: str = _LAKE_OPTION,
    handle_path: Path = _PLAN_HANDLE_OPTION,
    worker_id: int = typer.Option(0, "--worker", help="Worker id (0-based)."),
    num_workers: int = typer.Option(1, "--num-workers", help="Total workers claiming pages."),
    resume_from: int = typer.Option(0, "--resume-from", help="Global epoch sample offset."),
    page_token: str | None = typer.Option(
        None, "--page-token", help="Fetch a specific page by its token."
    ),
    all_pages: bool = typer.Option(
        False, "--all", help="Return every page this worker claims, not just the first."
    ),
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
    output_format: str = _FORMAT_OPTION,
) -> None:
    """Fetch bounded pages of a server-side row-plan handle for a worker."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.training import ServerSidePlanError, TrainingError

    try:
        handle = json.loads(handle_path.read_text())
        opened = Lake.open(
            lake,
            remote_auth_ref=remote_auth_ref,
            region=region,
            host_override=host_override,
        )
        if all_pages:
            result: object = opened.training.row_plan_pages(
                handle,
                worker_id=worker_id,
                num_workers=num_workers,
                resume_from=resume_from,
            )
        else:
            result = opened.training.row_plan_page(
                handle,
                worker_id=worker_id,
                num_workers=num_workers,
                resume_from=resume_from,
                page_token=page_token,
            )
    except (LakeError, TrainingError, ServerSidePlanError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if output_format == "json":
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
        return
    if output_format != "text":
        typer.echo("error: --format must be text or json", err=True)
        raise typer.Exit(code=2)

    pages = result if isinstance(result, list) else [result]
    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"worker: {worker_id}/{num_workers} resume_from={resume_from}")
    for page in pages:
        token = page.get("page_token")
        typer.echo(
            f"page {page.get('page_index')}: {page.get('size')} rows "
            f"(offset {page.get('start_offset')}, token {token}, "
            f"next {page.get('next_page_token')})"
        )


# --- Backlog 0100: indexed manifest query, paged listing, metric lookup,
# --- and retention planning over training/evaluation run manifests. ---------

_MANIFEST_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_PAGE_SIZE_OPTION = typer.Option(50, "--page-size", help="Rows per page (1-1000).")
_PAGE_TOKEN_OPTION = typer.Option(None, "--page-token", help="Continuation token from a prior page.")
_MANIFEST_FORMAT_OPTION = typer.Option("text", "--format", help="Output format: text or json.")


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _open_lake_or_exit(lake: str):
    from lancedb_robotics.lake import Lake, LakeError

    try:
        return Lake.open(lake)
    except LakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _echo_manifest_page(page, columns) -> None:
    typer.echo(f"table: {page.table}")
    typer.echo(f"rows: {len(page.rows)}/{page.total_count}")
    for row in page.rows:
        typer.echo("  " + "  ".join(f"{col}={row.get(col)}" for col in columns))
    if page.next_page_token:
        typer.echo(f"next page token: {page.next_page_token}")


@train_app.command("runs")
def list_runs(
    lake: str = _MANIFEST_LAKE_OPTION,
    dataset: str = typer.Option(None, "--dataset", help="Filter by dataset id."),
    snapshot: str = typer.Option(None, "--snapshot", help="Filter by snapshot name/tag/id."),
    code_ref: str = typer.Option(None, "--code-ref", help="Filter by code ref."),
    status: str = typer.Option(None, "--status", help="Filter by run status."),
    page_size: int = _PAGE_SIZE_OPTION,
    page_token: str = _PAGE_TOKEN_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """List training-run manifests (deterministic paging by created_at, id)."""
    from lancedb_robotics.run_manifests import RunManifestError

    opened = _open_lake_or_exit(lake)
    try:
        page = opened.training.list_runs(
            page_size=page_size,
            page_token=page_token,
            dataset_id=dataset,
            snapshot=snapshot,
            code_ref=code_ref,
            status=status,
        )
    except RunManifestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(page.to_dict())
        return
    _echo_manifest_page(page, ("training_run_id", "dataset_id", "snapshot_name", "status"))


@train_app.command("checkpoints")
def list_checkpoints(
    lake: str = _MANIFEST_LAKE_OPTION,
    training_run: str = typer.Option(None, "--training-run", help="Filter by training run id."),
    framework: str = typer.Option(None, "--framework", help="Filter by framework."),
    page_size: int = _PAGE_SIZE_OPTION,
    page_token: str = _PAGE_TOKEN_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """List model-artifact/checkpoint manifests (deterministic paging)."""
    opened = _open_lake_or_exit(lake)
    page = opened.training.list_checkpoints(
        page_size=page_size,
        page_token=page_token,
        training_run_id=training_run,
        framework=framework,
    )
    if output_format == "json":
        _emit_json(page.to_dict())
        return
    _echo_manifest_page(page, ("model_artifact_id", "training_run_id", "artifact_uri", "framework"))


@train_app.command("evals")
def list_evals(
    lake: str = _MANIFEST_LAKE_OPTION,
    model_artifact: str = typer.Option(None, "--model-artifact", help="Filter by model artifact id."),
    snapshot: str = typer.Option(None, "--snapshot", help="Filter by snapshot name/tag/id."),
    dataset: str = typer.Option(None, "--dataset", help="Filter by dataset id."),
    status: str = typer.Option(None, "--status", help="Filter by eval status."),
    page_size: int = _PAGE_SIZE_OPTION,
    page_token: str = _PAGE_TOKEN_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """List evaluation-run manifests (deterministic paging)."""
    from lancedb_robotics.run_manifests import RunManifestError

    opened = _open_lake_or_exit(lake)
    try:
        page = opened.eval.list(
            page_size=page_size,
            page_token=page_token,
            model_artifact_id=model_artifact,
            snapshot=snapshot,
            dataset_id=dataset,
            status=status,
        )
    except RunManifestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(page.to_dict())
        return
    _echo_manifest_page(page, ("eval_run_id", "model_artifact_id", "snapshot_name", "status"))


@train_app.command("metrics")
def eval_metrics(
    lake: str = _MANIFEST_LAKE_OPTION,
    metric: str = typer.Option(None, "--metric", help="Metric name, e.g. success_rate."),
    metric_key: str = typer.Option(None, "--metric-key", help="Full key, e.g. night/rain.success_rate."),
    slice_label: str = typer.Option(None, "--slice", help="Slice label, e.g. night/rain."),
    snapshot: str = typer.Option(None, "--snapshot", help="Filter by snapshot name/tag/id."),
    model_artifact: str = typer.Option(None, "--model-artifact", help="Filter by model artifact id."),
    max_score: float = typer.Option(None, "--max-score", help="Keep rows with score < this value."),
    min_score: float = typer.Option(None, "--min-score", help="Keep rows with score >= this value."),
    limit: int = typer.Option(None, "--limit", help="Cap the number of rows returned."),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Look up eval aggregate/slice metrics by key (uses the materialized index)."""
    from lancedb_robotics.run_manifests import RunManifestError

    opened = _open_lake_or_exit(lake)
    try:
        result = opened.eval.metrics(
            metric=metric,
            metric_key=metric_key,
            slice_label=slice_label,
            snapshot=snapshot,
            model_artifact_id=model_artifact,
            max_score=max_score,
            min_score=min_score,
            limit=limit,
        )
    except RunManifestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(result.to_dict())
        return
    typer.echo(f"materialized index: {result.materialized}")
    typer.echo(f"rows: {len(result.rows)}")
    for row in result.rows:
        typer.echo(
            f"  {row.get('metric_key')}={row.get('score')} "
            f"eval={row.get('eval_run_id')} model={row.get('model_artifact_id')}"
        )


@train_app.command("sync-metrics")
def sync_metrics(
    lake: str = _MANIFEST_LAKE_OPTION,
    build_indexes: bool = typer.Option(
        True, "--build-indexes/--no-build-indexes", help="(Re)build scalar predicate indexes."
    ),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Rebuild the materialized evaluation_run_metrics surface from evaluation_runs."""
    opened = _open_lake_or_exit(lake)
    report = opened.eval.sync_metrics(build_indexes=build_indexes)
    if output_format == "json":
        _emit_json(report.to_dict())
        return
    typer.echo(f"eval runs: {report.eval_runs}")
    typer.echo(f"metric rows: {report.metric_rows}")
    built = [r for r in report.index_results if r.get("status") == "built"]
    typer.echo(f"indexes built: {len(built)}")


@train_app.command("retention")
def retention(
    lake: str = _MANIFEST_LAKE_OPTION,
    kind: str = typer.Option(
        None, "--kind", help="Limit to training-run, model-artifact, evaluation-run, or training-report."
    ),
    older_than_days: float = typer.Option(
        None, "--older-than-days", help="Only list unprotected rows older than N days as deletable."
    ),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Report which manifests are protected (lineage/feedback/snapshot) vs deletable."""
    from datetime import timedelta

    from lancedb_robotics.run_manifests import RunManifestError

    opened = _open_lake_or_exit(lake)
    kinds = [kind] if kind else None
    older_than = timedelta(days=older_than_days) if older_than_days is not None else None
    try:
        plan = opened.training.retention_plan(kinds=kinds, older_than=older_than)
    except RunManifestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(plan.to_dict())
        return
    typer.echo(f"protected: {len(plan.protected)}")
    for item in plan.protected:
        typer.echo(f"  {item.kind} {item.manifest_id} <- {', '.join(item.reasons)}")
    typer.echo(f"deletable: {len(plan.deletable)}")
    for item in plan.deletable:
        typer.echo(f"  {item.kind} {item.manifest_id}")


@train_app.command("expire")
def expire(
    lake: str = _MANIFEST_LAKE_OPTION,
    kind: str = typer.Option(
        ..., "--kind", help="training-run, model-artifact, evaluation-run, or training-report."
    ),
    manifest_id: str = typer.Option(..., "--id", help="Manifest id to delete."),
    force: bool = typer.Option(False, "--force", help="Delete even if the row is protected."),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Delete a manifest row; refuses protected rows unless --force is passed."""
    from lancedb_robotics.run_manifests import RunManifestError, delete_manifest

    opened = _open_lake_or_exit(lake)
    try:
        result = delete_manifest(opened, kind=kind, manifest_id=manifest_id, force=force)
    except RunManifestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(result)
        return
    typer.echo(f"deleted {result['kind']} {result['manifest_id']} (forced={result['forced']})")


# --- Backlog 0115: Enterprise training report catalog and run history --------

_REPORT_SUMMARY_DISPLAY = (
    "report_id",
    "training_run_id",
    "snapshot_name",
    "resolved_backend",
    "cache_policy",
    "epoch",
    "worker_id",
    "fallback",
    "cache_hits",
    "cache_misses",
)


@report_app.command("list")
def report_list(
    lake: str = _MANIFEST_LAKE_OPTION,
    training_run: str = typer.Option(None, "--training-run", help="Filter by training run id."),
    snapshot: str = typer.Option(None, "--snapshot", help="Filter by snapshot name/tag/id."),
    dataset: str = typer.Option(None, "--dataset", help="Filter by dataset id."),
    backend: str = typer.Option(None, "--backend", help="Filter by resolved backend kind."),
    cache_policy: str = typer.Option(None, "--cache-policy", help="Filter by cache policy."),
    fallback: bool = typer.Option(None, "--fallback/--no-fallback", help="Only reports that did/didn't fall back."),
    fallback_reason: str = typer.Option(None, "--fallback-reason", help="Filter by fallback reason."),
    loader_kind: str = typer.Option(None, "--loader-kind", help="native-training or aligned-training."),
    page_size: int = _PAGE_SIZE_OPTION,
    page_token: str = _PAGE_TOKEN_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """List persisted training-report history (summary rows, deterministic paging)."""
    from lancedb_robotics.run_manifests import RunManifestError

    opened = _open_lake_or_exit(lake)
    try:
        page = opened.training.reports(
            page_size=page_size,
            page_token=page_token,
            training_run_id=training_run,
            snapshot=snapshot,
            dataset_id=dataset,
            resolved_backend=backend,
            cache_policy=cache_policy,
            fallback=fallback,
            fallback_reason=fallback_reason,
            loader_kind=loader_kind,
        )
    except RunManifestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(page.to_dict())
        return
    _echo_manifest_page(page, _REPORT_SUMMARY_DISPLAY)


@report_app.command("get")
def report_get(
    lake: str = _MANIFEST_LAKE_OPTION,
    report_id: str = typer.Option(None, "--id", help="Report id (trpt-...)."),
    training_run: str = typer.Option(None, "--training-run", help="Reload latest report for a run."),
    epoch: int = typer.Option(None, "--epoch", help="Narrow by epoch."),
    worker_id: int = typer.Option(None, "--worker", help="Narrow by worker id."),
    report_out: str = typer.Option(None, "--report-out", help="Write the full report JSON to this path."),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Reload one full backend report by id or by training-run/epoch/worker."""
    from lancedb_robotics.run_manifests import RunManifestError

    opened = _open_lake_or_exit(lake)
    try:
        record = opened.training.get_report(
            report_id=report_id,
            training_run_id=training_run,
            epoch=epoch,
            worker_id=worker_id,
        )
    except RunManifestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if report_out:
        Path(report_out).write_text(json.dumps(record.report, indent=2, sort_keys=True) + "\n")
    if output_format == "json":
        _emit_json(record.to_dict())
        return
    typer.echo(f"report: {record.report_id}")
    typer.echo(f"training run: {record.training_run_id}")
    typer.echo(f"snapshot: {record.snapshot_name} ({record.dataset_id})")
    typer.echo(f"backend: {record.resolved_backend} (requested {record.requested_backend})")
    typer.echo(f"cache policy: {record.cache_policy}  prewarm: {record.prewarm_status}")
    typer.echo(f"epoch/worker: {record.epoch}/{record.worker_id} of {record.num_workers}")
    typer.echo(f"cache hits/misses: {record.cache_hits}/{record.cache_misses}")
    typer.echo(f"bytes read: {record.bytes_read}")
    if record.fallback:
        typer.echo(
            f"fallback: {record.fallback_from_backend}->{record.fallback_to_backend} "
            f"({record.fallback_reason})"
        )
    if report_out:
        typer.echo(f"report written: {report_out}")


@report_app.command("metrics")
def report_metrics(
    lake: str = _MANIFEST_LAKE_OPTION,
    training_run: str = typer.Option(None, "--training-run", help="Aggregate over one training run."),
    snapshot: str = typer.Option(None, "--snapshot", help="Aggregate over a snapshot."),
    dataset: str = typer.Option(None, "--dataset", help="Aggregate over a dataset id."),
    backend: str = typer.Option(None, "--backend", help="Filter by resolved backend kind."),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Sum cache hits/misses, bytes read, and PE fanout across matched reports."""
    from lancedb_robotics.run_manifests import RunManifestError

    opened = _open_lake_or_exit(lake)
    try:
        agg = opened.training.report_metrics(
            training_run_id=training_run,
            snapshot=snapshot,
            dataset_id=dataset,
            resolved_backend=backend,
        )
    except RunManifestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(agg.to_dict())
        return
    typer.echo(f"reports: {agg.report_count}")
    totals = agg.totals
    typer.echo(
        f"cache hits/misses: {totals['cache_hits']}/{totals['cache_misses']}  "
        f"bytes read: {totals['bytes_read']}  pe fanout: {totals['pe_fanout']}"
    )
    for worker, values in sorted(agg.by_worker.items()):
        typer.echo(f"  worker {worker}: hits={values.get('cache_hits')} misses={values.get('cache_misses')}")
    for epoch, values in sorted(agg.by_epoch.items()):
        typer.echo(f"  epoch {epoch}: hits={values.get('cache_hits')} misses={values.get('cache_misses')}")


_REPORT_MERGE_INPUT_OPTION = typer.Option(
    ...,
    "--report",
    "-r",
    help="Path to a per-worker report JSON file (repeatable).",
)


@report_app.command("merge")
def report_merge(
    report: list[str] = _REPORT_MERGE_INPUT_OPTION,
    job_id: str = typer.Option(
        None, "--job-id", help="Override the deterministic job id for the aggregate."
    ),
    out: str = typer.Option(
        None, "--out", help="Write the aggregated report JSON to this path."
    ),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Merge per-worker training loader report JSON files into one job-level report.

    Combines cache metrics, bytes read, PE fanout, and operation counts across
    workers; deduplicates shared prewarm events by prewarm_id; and flags mixed
    backends or fallbacks. Lake-free: reads and writes JSON files only.
    """
    from lancedb_robotics.training_report_aggregation import (
        TrainingReportAggregationError,
        aggregate_training_loader_reports,
        load_report_payloads,
    )

    try:
        payloads = load_report_payloads(report)
        aggregate = aggregate_training_loader_reports(payloads, job_id=job_id)
    except TrainingReportAggregationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    result = aggregate.to_dict()
    if out:
        aggregate.write_json(out)
    if output_format == "json":
        _emit_json(result)
        return
    job = result["job"]
    totals = result["totals"]
    typer.echo(f"job: {job['job_id']}")
    typer.echo(f"reports: {job['report_count']}  workers: {job['worker_count']}")
    typer.echo(
        f"backends: requested={','.join(job['backends']['requested']) or '-'} "
        f"resolved={','.join(job['backends']['resolved']) or '-'}"
    )
    typer.echo(
        f"bytes read: {totals['bytes_read']}  rows hydrated: {totals['rows_hydrated']}  "
        f"row ids coalesced: {totals['row_ids_coalesced']}"
    )
    ops = totals["operations"]
    typer.echo(
        f"requests: scan={ops['remote_scan']} take={ops['remote_take']} "
        f"filtered_read={ops['remote_filtered_read']} prewarm={ops['prewarm']}"
    )
    prewarm = result["prewarm"]
    statuses = ",".join(f"{k}={v}" for k, v in sorted(prewarm["statuses"].items())) or "-"
    typer.echo(
        f"prewarm: {prewarm['unique_prewarm_ids']} unique id(s), "
        f"submitted={prewarm['requests_submitted']} statuses={statuses}"
    )
    for worker, values in sorted(result["by_worker"].items()):
        typer.echo(
            f"  worker {worker}: bytes={values['bytes_read']} rows={values['rows_hydrated']}"
            + ("  [fell back]" if values.get("fell_back") else "")
        )
    for line in result["warnings"]:
        typer.echo(f"warning: {line}")
    if out:
        typer.echo(f"report written: {out}")


# --- Backlog 0124: loader-report schema + redaction conformance --------------

_REPORT_VALIDATE_INPUT_ARGUMENT = typer.Argument(
    ..., help="One or more training loader report JSON files to validate."
)


@report_app.command("validate")
def report_validate(
    report: list[str] = _REPORT_VALIDATE_INPUT_ARGUMENT,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Validate loader report JSON against the versioned schema + redaction contract.

    Checks each file against ``lancedb-robotics/training-loader-report/v1``: the
    report must match the committed schema *and* carry no unredacted credential
    (API key, bearer/authorization header, object-store access key, scoped/STS
    token, or namespace credential-vending value). Safe fields — ``display_uri``,
    endpoint/backend labels, and ``*_auth_ref`` names — are preserved. Exits
    non-zero if any report is malformed, off-schema, or credential-bearing, so a
    benchmark or catalog step can gate on it.
    """
    from lancedb_robotics.training_report_schema import (
        TRAINING_LOADER_REPORT_SCHEMA_ID,
        ReportValidationError,
        load_report_json,
        validate_training_loader_report,
    )

    results: list[dict] = []
    exit_code = 0
    for path in report:
        try:
            payload = load_report_json(path)
        except ReportValidationError as exc:
            exit_code = 1
            entry = {
                "path": path,
                "schema_id": exc.validation.schema_id,
                "ok": False,
                "schema_errors": exc.validation.schema_errors,
                "secret_findings": [],
            }
            results.append(entry)
            if output_format != "json":
                typer.echo(f"{path}: FAILED", err=True)
                for line in entry["schema_errors"]:
                    typer.echo(f"  schema: {line}", err=True)
            continue
        except OSError as exc:
            exit_code = 1
            typer.echo(f"{path}: error: {exc}", err=True)
            results.append({"path": path, "ok": False, "schema_errors": [str(exc)], "secret_findings": []})
            continue

        validation = validate_training_loader_report(payload)
        entry = {"path": path, **validation.to_dict()}
        results.append(entry)
        if not validation.ok:
            exit_code = 1
        if output_format != "json":
            if validation.ok:
                typer.echo(f"{path}: OK ({validation.schema_id})")
            else:
                typer.echo(f"{path}: FAILED ({validation.schema_id})", err=True)
                for line in validation.schema_errors:
                    typer.echo(f"  schema: {line}", err=True)
                for line in validation.secret_findings:
                    typer.echo(f"  redaction: {line}", err=True)

    if output_format == "json":
        _emit_json(
            {"schema_id": TRAINING_LOADER_REPORT_SCHEMA_ID, "ok": exit_code == 0, "reports": results}
        )
    if exit_code:
        raise typer.Exit(code=exit_code)


# --- Backlog 0101: external experiment-tracker manifest sync -----------------

_TRACKER_SOURCE_OPTION = typer.Option(
    "generic",
    "--source",
    help="Tracker source: generic (JSON bundle), mlflow, wandb, ...",
)
_TRACKER_FROM_OPTION = typer.Option(
    None,
    "--from",
    help="Path to a JSON manifest bundle to import (omit to live-fetch from the tracker).",
)
_TRACKER_CONFLICT_OPTION = typer.Option(
    "external-wins",
    "--conflict",
    help="Drift policy: external-wins, lake-wins, or append-superseding.",
)
_TRACKER_DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run/--apply",
    help="Plan and report the sync without writing (default applies).",
)
_TRACKER_OUT_OPTION = typer.Option(
    None, "--out", help="Write the JSON manifest bundle to this path."
)
_TRACKER_TRAINING_RUN_OPTION = typer.Option(
    None, "--training-run", help="Limit export to these training run ids (repeatable)."
)
_TRACKER_DRIFT_FROM_OPTION = typer.Option(
    ..., "--from", help="Path to the JSON manifest bundle to check."
)


def _echo_sync_entries(entries) -> None:
    for entry in entries:
        suffix = f" <- drift (was {entry.previous_digest})" if entry.drift else ""
        superseded = f" superseded_by={entry.superseded_by}" if entry.superseded_by else ""
        typer.echo(f"  {entry.action} {entry.kind} {entry.manifest_id}{suffix}{superseded}")


@tracker_app.command("import")
def tracker_import(
    lake: str = _MANIFEST_LAKE_OPTION,
    source: str = _TRACKER_SOURCE_OPTION,
    from_path: Path = _TRACKER_FROM_OPTION,
    conflict: str = _TRACKER_CONFLICT_OPTION,
    dry_run: bool = _TRACKER_DRY_RUN_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Import external tracker runs/checkpoints/evals into canonical manifests."""
    from lancedb_robotics.tracker_sync import (
        TrackerSyncError,
        import_manifest_bundle,
        load_bundle_file,
    )

    opened = _open_lake_or_exit(lake)
    try:
        bundle = load_bundle_file(from_path) if from_path is not None else None
        report = import_manifest_bundle(
            opened,
            bundle,
            source=source,
            conflict=conflict,
            dry_run=dry_run,
        )
    except TrackerSyncError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(report.to_dict())
        return
    typer.echo(f"source: {report.source} (conflict={report.conflict}, dry_run={report.dry_run})")
    typer.echo(
        f"created={report.created} updated={report.updated} unchanged={report.unchanged} "
        f"skipped={report.skipped} superseded={report.superseded} conflicts={len(report.conflicts)}"
    )
    _echo_sync_entries(report.entries)
    if report.transform_id:
        typer.echo(f"transform: {report.transform_id}")


@tracker_app.command("export")
def tracker_export(
    lake: str = _MANIFEST_LAKE_OPTION,
    source: str = _TRACKER_SOURCE_OPTION,
    out: Path = _TRACKER_OUT_OPTION,
    training_run: list[str] = _TRACKER_TRAINING_RUN_OPTION,
    no_checkpoints: bool = typer.Option(False, "--no-checkpoints", help="Exclude model artifacts."),
    no_evals: bool = typer.Option(False, "--no-evals", help="Exclude evaluation runs."),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Export canonical manifests into a portable, round-trippable JSON bundle."""
    from lancedb_robotics.tracker_sync import TrackerSyncError, export_manifest_bundle

    opened = _open_lake_or_exit(lake)
    try:
        report = export_manifest_bundle(
            opened,
            source=source,
            training_run_ids=training_run or None,
            include_checkpoints=not no_checkpoints,
            include_evaluations=not no_evals,
            out_path=out,
        )
    except TrackerSyncError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(report.to_dict())
        return
    counts = report.to_dict()["counts"]
    typer.echo(f"source: {report.source}")
    typer.echo(
        f"training_runs={counts['training_runs']} model_artifacts={counts['model_artifacts']} "
        f"evaluation_runs={counts['evaluation_runs']}"
    )
    if report.out_path:
        typer.echo(f"bundle: {report.out_path}")


@tracker_app.command("drift")
def tracker_drift(
    lake: str = _MANIFEST_LAKE_OPTION,
    source: str = _TRACKER_SOURCE_OPTION,
    from_path: Path = _TRACKER_DRIFT_FROM_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Report which external runs differ from canonical rows (read-only)."""
    from lancedb_robotics.tracker_sync import (
        TrackerSyncError,
        drift_report,
        load_bundle_file,
    )

    opened = _open_lake_or_exit(lake)
    try:
        report = drift_report(opened, load_bundle_file(from_path), source=source)
    except TrackerSyncError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(report.to_dict())
        return
    typer.echo(f"source: {report.source} (has_drift={report.has_drift})")
    typer.echo(f"conflicts={len(report.conflicts)} new={len(report.new)} total={len(report.entries)}")
    _echo_sync_entries(report.conflicts)


_CONFORMANCE_FORMAT_OPTION = typer.Option(
    "table", "--format", help="Output format: table, markdown, or json."
)


@conformance_app.command("matrix")
def conformance_matrix(
    output_format: str = _CONFORMANCE_FORMAT_OPTION,
    include_local_endpoint: bool = typer.Option(
        False,
        "--include-local-endpoint",
        help="Include the gated local Enterprise CLI endpoint row.",
    ),
) -> None:
    """Print the Enterprise training compatibility matrix (no lake required)."""
    from lancedb_robotics.enterprise_conformance import compatibility_matrix

    matrix = compatibility_matrix(include_local_endpoint=include_local_endpoint)
    if output_format == "json":
        _emit_json(matrix.to_dict())
        return
    if output_format == "markdown":
        typer.echo(matrix.to_markdown())
        return
    summary = matrix.to_dict()["summary"]
    typer.echo(
        "compatibility matrix: "
        + "  ".join(f"{key}={value}" for key, value in sorted(summary.items()))
    )
    for row in matrix.rows:
        detail = row.error_type or (
            f"fallback->{row.fallback_to}" if row.fallback_to else row.resolved_backend or "-"
        )
        typer.echo(
            f"  {row.name:34s} {row.category:12s} {row.connection_kind:22s} {detail}"
        )


@conformance_app.command("run")
def conformance_run(
    lake: str = _MANIFEST_LAKE_OPTION,
    snapshot: str = typer.Option(..., "--snapshot", help="Snapshot to replay through each case."),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if any conformance invariant is violated."
    ),
    include_local_endpoint: bool = typer.Option(
        False,
        "--include-local-endpoint",
        help="Also run the gated local Enterprise CLI endpoint case.",
    ),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Replay a snapshot through every backend case and injected fault."""
    from lancedb_robotics.enterprise_conformance import ConformanceError, run_conformance

    opened = _open_lake_or_exit(lake)
    try:
        report = run_conformance(
            opened,
            snapshot,
            include_local_endpoint=include_local_endpoint,
            strict=strict,
        )
    except ConformanceError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(report.to_dict())
    else:
        summary = report.summary()
        typer.echo(
            f"conformance: {summary['passed']}/{summary['total']} passed "
            f"(failed={summary['failed']})"
        )
        for outcome in report.outcomes:
            marker = "ok " if outcome.status == "pass" else "FAIL"
            detail = outcome.error_type or (
                f"fallback->{outcome.fallback_to}" if outcome.fallback_to else outcome.resolved_backend or "-"
            )
            typer.echo(f"  [{marker}] {outcome.name:34s} {outcome.category:12s} {detail}")
            for failure in outcome.failures:
                typer.echo(f"        - {failure}")
    if strict and not report.ok():
        raise typer.Exit(code=1)


@conformance_app.command("query-node")
def conformance_query_node(
    lake: str = _MANIFEST_LAKE_OPTION,
    snapshot: str = typer.Option(
        ..., "--snapshot", help="Snapshot to hydrate through the live query-node read client."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if any query-node invariant is violated."
    ),
    include_local_endpoint: bool = typer.Option(
        False,
        "--include-local-endpoint",
        help="Also record the gated local Sophon query-node endpoint.",
    ),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Prove the live query-node read-client contract for a pinned snapshot (0119)."""
    from lancedb_robotics.enterprise_conformance import (
        ConformanceError,
        run_query_node_conformance,
    )

    opened = _open_lake_or_exit(lake)
    try:
        report = run_query_node_conformance(
            opened,
            snapshot,
            include_local_endpoint=include_local_endpoint,
            strict=strict,
        )
    except ConformanceError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(report.to_dict())
    else:
        summary = report.summary()
        typer.echo(
            f"query-node conformance: {summary['passed']}/{summary['total']} passed "
            f"(failed={summary['failed']})"
        )
        metrics = report.metrics
        typer.echo(
            "  cache hits/misses: "
            f"{metrics.get('cache_hits')}/{metrics.get('cache_misses')}  "
            f"pe_fanout={metrics.get('pe_fanout')} (server telemetry)  "
            f"live_requests={metrics.get('live_hydration_requests')}"
        )
        for name, entry in report.checks.items():
            marker = "ok " if entry.get("status") == "pass" else "FAIL"
            typer.echo(f"  [{marker}] {name}")
        if report.local_endpoint:
            typer.echo(f"  local endpoint: {report.local_endpoint}")
    if strict and not report.ok():
        raise typer.Exit(code=1)


# Backlog 0345 deprecation alias: `train conformance plan-executor` still works
# (hidden) and delegates to the query-node command.
@conformance_app.command("plan-executor", hidden=True)
def conformance_plan_executor(
    lake: str = _MANIFEST_LAKE_OPTION,
    snapshot: str = typer.Option(..., "--snapshot"),
    strict: bool = typer.Option(False, "--strict"),
    include_local_endpoint: bool = typer.Option(False, "--include-local-endpoint"),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Deprecated alias for `train conformance query-node`."""
    conformance_query_node(
        lake=lake,
        snapshot=snapshot,
        strict=strict,
        include_local_endpoint=include_local_endpoint,
        output_format=output_format,
    )


@permutation_app.command("capability")
def permutation_capability(
    lake: str = _MANIFEST_LAKE_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Report whether this lake can back a plan with the native Permutation reader."""
    opened = _open_lake_or_exit(lake)
    cap = opened.training.permutation_capability()
    if output_format == "json":
        _emit_json(cap)
        return
    typer.echo(f"native permutation supported: {cap['supported']}")
    typer.echo(f"execution mode: {cap['execution_mode']}")
    typer.echo(f"reason: {cap['reason']}")
    typer.echo(f"torch available: {cap['torch_available']}")
    typer.echo(f"torch_col scalar-only: {cap['torch_col_scalar_only']}")
    typer.echo(f"output formats: {', '.join(cap['output_formats'])}")


@permutation_app.command("build")
def permutation_build(
    lake: str = _MANIFEST_LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    columns: str = typer.Option(
        "state_vector,action_vector",
        "--columns",
        help="Comma-separated native training columns to project.",
    ),
    read_format: str = typer.Option(
        "arrow",
        "--read-format",
        help="Native batch format: arrow, numpy, pandas, python, python_col, torch, torch_col, polars.",
    ),
    shuffle: bool = typer.Option(
        True, "--shuffle/--no-shuffle", help="Seeded global epoch shuffle."
    ),
    shuffle_seed: int = typer.Option(0, "--shuffle-seed", help="Deterministic shuffle seed."),
    epoch: int = typer.Option(0, "--epoch", help="Global epoch offset for rotation."),
    worker_id: int = typer.Option(0, "--worker", help="Worker id (0-based)."),
    num_workers: int = typer.Option(1, "--num-workers", help="Total workers partitioning the epoch."),
    resume_from: int = typer.Option(0, "--resume-from", help="Global epoch sample offset."),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Skip the bounded read-back equivalence check."
    ),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Build a native-Permutation-backed plan handle for a snapshot and print it."""
    from lancedb_robotics.training import TrainingError

    opened = _open_lake_or_exit(lake)
    selected = [c.strip() for c in columns.split(",") if c.strip()]
    try:
        plan = opened.training.permutation_plan(
            snapshot,
            columns=selected or None,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            worker_id=worker_id,
            num_workers=num_workers,
            resume_from=resume_from,
            output_format=read_format,
            verify=not no_verify,
        )
    except TrainingError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    handle = plan.to_dict()
    if output_format == "json":
        _emit_json(handle)
        return
    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"snapshot: {handle['snapshot_name']} ({handle['dataset_id']})")
    typer.echo(f"row plan: {handle['row_plan_id']}")
    typer.echo(f"epoch plan: {handle['epoch_plan_id']}")
    typer.echo(f"base table: {handle['base_table']} v{handle['base_table_version']}")
    typer.echo(f"permutation: {handle['permutation_ref']} ({handle['permutation_source']})")
    typer.echo(f"projection: {', '.join(handle['projection'])}")
    typer.echo(f"read format: {handle['output_format']}")
    typer.echo(f"rows: {handle['num_rows']}")
    equivalence = handle["equivalence"]
    typer.echo(
        "equivalence: "
        f"matches={equivalence['matches']} "
        f"({equivalence['checked_rows']}/{equivalence['total_rows']} checked)"
    )
    torch_col = handle["torch_col"]
    typer.echo(f"torch_col supported: {torch_col['supported']} ({torch_col['reason']})")
    for warning in handle["warnings"]:
        typer.echo(f"warning: {warning}")


_PREWARM_ID_OPTION = typer.Option(..., "--id", help="Prewarm JobRun id (the opaque prewarm-* id).")


def _prewarm_job_line(job: dict) -> str:
    return (
        f"  {job['prewarm_id']}  {job['status']:9s} policy={job['policy']:8s} "
        f"attach={job['attach_count']} retry={job['retry_count']} "
        f"workers={len(job['workers'])}"
    )


@prewarm_app.command("list")
def prewarm_list(
    lake: str = _MANIFEST_LAKE_OPTION,
    status: str = typer.Option(None, "--status", help="Filter by lifecycle status."),
    policy: str = typer.Option(None, "--policy", help="Filter by cache policy (epoch/snapshot)."),
    limit: int = typer.Option(None, "--limit", help="Max JobRuns to return."),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """List durable Enterprise cache-prewarm JobRuns for a lake (backlog 0121)."""
    opened = _open_lake_or_exit(lake)
    jobs = opened.training.prewarm_jobs(status=status, policy=policy, limit=limit)
    if output_format == "json":
        _emit_json(jobs)
        return
    if not jobs:
        typer.echo("no prewarm JobRuns recorded")
        return
    typer.echo(f"prewarm JobRuns ({len(jobs)}):")
    for job in jobs:
        typer.echo(_prewarm_job_line(job))


@prewarm_app.command("status")
def prewarm_status(
    lake: str = _MANIFEST_LAKE_OPTION,
    prewarm_id: str = _PREWARM_ID_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Show a single durable prewarm JobRun by id (full status history)."""
    opened = _open_lake_or_exit(lake)
    job = opened.training.prewarm_job(prewarm_id)
    if job is None:
        typer.echo(f"error: no prewarm JobRun for id {prewarm_id!r}", err=True)
        raise typer.Exit(code=1)
    if output_format == "json":
        _emit_json(job)
        return
    typer.echo(f"prewarm JobRun: {job['prewarm_id']}")
    typer.echo(f"status: {job['status']} (reason={job['terminal_reason']})")
    typer.echo(f"policy/scope: {job['policy']}/{job['scope']}")
    typer.echo(f"attach_count: {job['attach_count']}  retry_count: {job['retry_count']}")
    typer.echo(f"workers: {', '.join(job['workers']) or '-'}")
    typer.echo(f"submitted/completed: {job['submitted_at']} -> {job['completed_at']}")
    typer.echo(
        "executors: "
        f"pe_fanout={job['pe_fanout']} completed={job['completed_executors']} "
        f"failed={job['failed_executors']}"
    )
    for entry in job["status_history"]:
        typer.echo(f"  - {entry['at']}  {entry['status']}  {entry.get('reason') or ''}")


@prewarm_app.command("retry")
def prewarm_retry(
    lake: str = _MANIFEST_LAKE_OPTION,
    prewarm_id: str = _PREWARM_ID_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Re-submit a failed/canceled/expired prewarm JobRun (increments retry count)."""
    from lancedb_robotics.training_prewarm_jobs import PrewarmJobRunError

    opened = _open_lake_or_exit(lake)
    try:
        status = opened.training.retry_prewarm_job(prewarm_id)
    except PrewarmJobRunError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(status)
        return
    typer.echo(f"retried {prewarm_id}: status={status['status']} retry_count={status['retry_count']}")


@prewarm_app.command("cancel")
def prewarm_cancel(
    lake: str = _MANIFEST_LAKE_OPTION,
    prewarm_id: str = _PREWARM_ID_OPTION,
    reason: str = typer.Option(None, "--reason", help="Optional cancellation reason."),
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Cancel a no-longer-needed prewarm JobRun (complete JobRuns are left as-is)."""
    from lancedb_robotics.training_prewarm_jobs import PrewarmJobRunError

    opened = _open_lake_or_exit(lake)
    try:
        status = opened.training.cancel_prewarm_job(prewarm_id, reason=reason)
    except PrewarmJobRunError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output_format == "json":
        _emit_json(status)
        return
    typer.echo(f"canceled {prewarm_id}: status={status['status']}")


@prewarm_app.command("expire")
def prewarm_expire(
    lake: str = _MANIFEST_LAKE_OPTION,
    output_format: str = _MANIFEST_FORMAT_OPTION,
) -> None:
    """Sweep and mark TTL-elapsed prewarm JobRuns as expired (maintenance)."""
    opened = _open_lake_or_exit(lake)
    expired = opened.training.expire_prewarm_jobs()
    if output_format == "json":
        _emit_json(expired)
        return
    typer.echo(f"expired {len(expired)} prewarm JobRun(s)")
    for job in expired:
        typer.echo(_prewarm_job_line(job))


@prewarm_app.command("plan")
def prewarm_plan(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    columns: str = typer.Option(
        None, "--columns", help="Comma-separated native training columns to prewarm."
    ),
    cache_policy: str = typer.Option(
        "epoch", "--cache-policy", help="Prewarm scope policy: epoch or snapshot."
    ),
    include_heavy: bool = typer.Option(
        False, "--include-heavy/--no-include-heavy", help="Include payload/video columns."
    ),
    concurrency: int = typer.Option(
        None, "--concurrency", help="Concurrent fragment scans hint for the query node."
    ),
    estimate: bool = typer.Option(
        True, "--estimate/--no-estimate", help="Compute advisory cost estimates."
    ),
    backend: str = _BACKEND_OPTION,
    allow_fallback: bool = _ALLOW_FALLBACK_OPTION,
    fallback: str | None = _FALLBACK_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
    output_format: str = _FORMAT_OPTION,
) -> None:
    """Build the query-node page-cache prewarm plan for a snapshot (backlog 0122).

    Emits a valid ``PageCacheBeginPrewarmRequest`` per table plus advisory cost
    estimates (over-warm ratio, per-column bytes, local fragment estimate). The plan
    targets the query node only -- never a plan executor.
    """
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.training import TrainingError

    try:
        selected = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        opened = Lake.open(
            lake,
            remote_auth_ref=remote_auth_ref,
            region=region,
            host_override=host_override,
        )
        plan = opened.training.page_cache_prewarm_plan(
            snapshot,
            columns=selected,
            cache_policy=cache_policy,
            include_heavy=include_heavy,
            backend=backend,
            allow_fallback=allow_fallback,
            fallback=fallback,
            concurrency=concurrency,
            estimate=estimate,
        )
    except (LakeError, TrainingError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if output_format == "json":
        typer.echo(json.dumps(plan, indent=2, sort_keys=True))
        return
    if output_format != "text":
        typer.echo("error: --format must be text or json", err=True)
        raise typer.Exit(code=2)

    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"prewarm id: {plan['prewarm_id']}")
    typer.echo(f"database: {plan['database']}  applicable: {plan['applicable']}")
    if not plan["applicable"]:
        typer.echo(f"reason: {plan['reason']}")
    metrics = plan["metrics"]
    typer.echo(
        "estimate: "
        f"selected_rows={metrics['selected_rows']} "
        f"over_warm_ratio={metrics['over_warm_ratio']} "
        f"est_bytes={metrics['estimated_bytes']}"
    )
    for request in plan["wire_requests"]:
        typer.echo(
            f"  prewarm {request['db']}/{request['table']}"
            f"@v{request.get('table_version')} cols={request['columns']}"
        )


def _emit_query_warm_plan_text(opened, plan: dict) -> None:
    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"snapshot: {plan['snapshot_name']}  scope: {plan['scope']}")
    metrics = plan["metrics"]
    typer.echo(
        "warm: "
        f"queries={metrics['queries']} ids={metrics['total_ids']} "
        f"chunk_size={metrics['chunk_size']} unindexed_tables={metrics['unindexed_tables']}"
    )
    for warning in plan["warnings"]:
        typer.echo(f"  warning: {warning}")
    for table in plan["tables"]:
        pre = table.get("precondition") or {}
        typer.echo(
            f"  {table['table']}@v{table['version']} on {table['id_column']} "
            f"(indexed={pre.get('indexed')}): {len(table['queries'])} warm queries "
            f"over {table['total_ids']} ids, select={table['columns']}"
        )


@warm_app.command("plan")
def warm_plan(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    columns: str = typer.Option(
        None, "--columns", help="Comma-separated native training columns to warm."
    ),
    chunk_size: int = typer.Option(
        512, "--chunk-size", help="Ids per bounded WHERE ... IN (...) warm query."
    ),
    include_heavy: bool = typer.Option(
        False, "--include-heavy/--no-include-heavy", help="Include payload/video columns."
    ),
    backend: str = _BACKEND_OPTION,
    allow_fallback: bool = _ALLOW_FALLBACK_OPTION,
    fallback: str | None = _FALLBACK_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
    output_format: str = _FORMAT_OPTION,
) -> None:
    """Plan query-driven cache warming for a snapshot (backlog 0348).

    Emits bounded ``WHERE <stable id> IN (...)`` warm queries over the epoch subset so
    the query node warms exactly what training reads. Never uses ``_rowid`` and never
    contacts a plan executor. Warns when the id column is not scalar-indexed.
    """
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.training import TrainingError

    try:
        selected = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        opened = Lake.open(
            lake,
            remote_auth_ref=remote_auth_ref,
            region=region,
            host_override=host_override,
        )
        plan = opened.training.query_warm_plan(
            snapshot,
            columns=selected,
            chunk_size=chunk_size,
            include_heavy=include_heavy,
            backend=backend,
            allow_fallback=allow_fallback,
            fallback=fallback,
        )
    except (LakeError, TrainingError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if output_format == "json":
        typer.echo(json.dumps(plan, indent=2, sort_keys=True))
        return
    if output_format != "text":
        typer.echo("error: --format must be text or json", err=True)
        raise typer.Exit(code=2)
    _emit_query_warm_plan_text(opened, plan)
