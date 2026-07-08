"""`lancedb-robotics dataset` subcommands."""

from pathlib import Path

import typer

from lancedb_robotics.cli.lineage_context import (
    LINEAGE_CONTEXT_OPTION,
    echo_emitted_lineage,
    lineage_context_error,
    load_lineage_context,
)

dataset_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
snapshot_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
dataset_app.add_typer(snapshot_app, name="snapshot", help="Create and manage dataset snapshots.")

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_NAME_OPTION = typer.Option(..., "--name", help="Human-facing snapshot name.")
_FROM_SEARCH_OPTION = typer.Option(
    None, "--from-search", help="Select scenarios from a recorded search: 'last'."
)
_SCENARIO_ID_OPTION = typer.Option(
    None, "--scenario-id", help="Select an explicit scenario id; repeat for several."
)
_SPLIT_BY_OPTION = typer.Option(
    "run", "--split-by", help="Assign the train/val/test split by 'run' or 'scenario'."
)
_TAG_OPTION = typer.Option(None, "--tag", help="Snapshot tag (defaults to the name).")
_SNAPSHOT_OPTION = typer.Option(..., "--snapshot", help="Snapshot name to export.")
_OUT_OPTION = typer.Option(..., "--out", help="Output directory for the materialized dataset.")
_OPTIONAL_OUT_OPTION = typer.Option(
    None, "--out", help="Output directory for materialized export mode."
)
_FORMAT_OPTION = typer.Option(
    "lerobot", "--format", help="Dataset export format: lerobot, rlds, or webdataset."
)
_SUMMARY_FORMAT_OPTION = typer.Option(
    None,
    "--format",
    help="Projection format to compare; repeat for several. Defaults to all formats.",
)
_REQUIRE_NATIVE_OPTION = typer.Option(
    False,
    "--require-native-loader",
    help="Fail if the matching native loader extra is not installed.",
)
_SHARD_SIZE_OPTION = typer.Option(
    1000,
    "--shard-size",
    help="WebDataset samples per tar shard.",
)
_COMPRESSION_OPTION = typer.Option(
    "none",
    "--compression",
    help="WebDataset tar compression: none or gzip.",
)


@snapshot_app.command("create")
def create(
    lake: str = _LAKE_OPTION,
    name: str = _NAME_OPTION,
    from_search: str = _FROM_SEARCH_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    split_by: str = _SPLIT_BY_OPTION,
    tag: str = _TAG_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Freeze a scenario selection into a reproducible dataset snapshot."""
    from lancedb_robotics.dataset import DatasetError, create_snapshot
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.lineage_hooks import LineageHookError
    from lancedb_robotics.search import last_search

    try:
        context = load_lineage_context(lineage_context)
        opened = Lake.open(lake)
        scenario_ids, source = _resolve_selection(opened, from_search, scenario_id, last_search)
        manifest = create_snapshot(
            opened,
            name=name,
            scenario_ids=scenario_ids,
            source=source,
            split_by=split_by,
            tag=tag,
            lineage_context=context,
        )
    except (LakeError, DatasetError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except LineageHookError as exc:
        raise lineage_context_error(exc) from exc

    typer.echo(f"lake: {manifest.lake_uri}")
    typer.echo(f"dataset: {manifest.name} ({manifest.dataset_id})")
    typer.echo(f"tag: {manifest.tag}")
    typer.echo(f"source: {_source_label(manifest.source)}")
    typer.echo(f"scenarios: {len(manifest.scenario_ids)}")
    counts = " ".join(f"{split}={count}" for split, count in manifest.split_counts.items())
    typer.echo(f"split by {manifest.split_by}: {counts}")
    versions = " ".join(f"{table}@{version}" for table, version in manifest.table_versions)
    typer.echo(f"table versions: {versions}")
    typer.echo(f"transform: {manifest.transform_id}")
    echo_emitted_lineage(opened, manifest.transform_id)


@dataset_app.command("export")
def export_dataset(
    format_arg: str | None = typer.Argument(
        None, help="Dataset export format: lerobot, rlds, or webdataset."
    ),
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    out: Path = _OUT_OPTION,
    fmt: str = _FORMAT_OPTION,
    require_native_loader: bool = _REQUIRE_NATIVE_OPTION,
    shard_size: int = _SHARD_SIZE_OPTION,
    compression: str = _COMPRESSION_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Export a dataset snapshot as a materialized boundary projection."""
    from lancedb_robotics.dataset_export import DATASET_EXPORT_MANIFEST_FILENAME
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.lineage_hooks import LineageHookError
    from lancedb_robotics.projections import (
        PROJECTION_MANIFEST_FILENAME,
        ProjectionError,
        export_projection,
    )

    selected_format = _resolve_format_argument(format_arg, fmt)

    try:
        context = load_lineage_context(lineage_context)
        opened = Lake.open(lake)
        manifest = export_projection(
            opened,
            snapshot,
            out_dir=out,
            fmt=selected_format,
            require_native=require_native_loader,
            shard_size=shard_size,
            compression=compression,
            lineage_context=context,
        )
    except (LakeError, ProjectionError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except LineageHookError as exc:
        raise lineage_context_error(exc) from exc

    _echo_projection_manifest(manifest)
    typer.echo(f"out: {out}")
    typer.echo(f"content hash: {manifest.content_hashes.get('dataset', '')}")
    typer.echo(f"manifest: {Path(out) / DATASET_EXPORT_MANIFEST_FILENAME}")
    typer.echo(f"projection manifest: {Path(out) / PROJECTION_MANIFEST_FILENAME}")
    _echo_native_loader(manifest)


@dataset_app.command("project")
def project_dataset(
    format_name: str = typer.Argument(..., help="Projection format: lerobot, rlds, or webdataset."),
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    mode: str = typer.Option("live", "--mode", help="Projection mode: live, plan, or export."),
    out: Path | None = _OPTIONAL_OUT_OPTION,
    require_native_loader: bool = _REQUIRE_NATIVE_OPTION,
    shard_size: int = _SHARD_SIZE_OPTION,
    compression: str = _COMPRESSION_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Open, plan, or materialize an external-format dataset projection."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.lineage_hooks import LineageHookError
    from lancedb_robotics.projections import ProjectionError, ProjectionMode, export_projection

    try:
        context = load_lineage_context(lineage_context)
        opened = Lake.open(lake)
        selected_mode = ProjectionMode(mode)
        if selected_mode == ProjectionMode.EXPORT:
            if out is None:
                raise ProjectionError("export projection mode requires --out")
            manifest = export_projection(
                opened,
                snapshot,
                out_dir=out,
                fmt=format_name,
                require_native=require_native_loader,
                shard_size=shard_size,
                compression=compression,
                lineage_context=context,
            )
            sample_count = None
        elif selected_mode == ProjectionMode.PLAN:
            manifest = opened.projections.plan(
                format_name,
                snapshot,
                require_native=require_native_loader,
                shard_size=shard_size,
                compression=compression,
                lineage_context=context,
            )
            sample_count = None
        else:
            adapter = opened.projections.dataset(
                format_name,
                snapshot,
                mode=selected_mode,
                require_native=require_native_loader,
                lineage_context=context,
            )
            manifest = adapter.manifest
            sample_count = len(adapter)
    except (LakeError, ProjectionError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except LineageHookError as exc:
        raise lineage_context_error(exc) from exc

    _echo_projection_manifest(manifest)
    if sample_count is not None:
        typer.echo(f"samples: {sample_count}")
    if out is not None and manifest.mode == ProjectionMode.EXPORT:
        from lancedb_robotics.projections import PROJECTION_MANIFEST_FILENAME

        typer.echo(f"projection manifest: {Path(out) / PROJECTION_MANIFEST_FILENAME}")
    _echo_native_loader(manifest)


@dataset_app.command("materialization-summary")
def materialization_summary(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    format_name: list[str] = _SUMMARY_FORMAT_OPTION,
    require_native_loader: bool = _REQUIRE_NATIVE_OPTION,
    shard_size: int = _SHARD_SIZE_OPTION,
    compression: str = _COMPRESSION_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Compare planned projection copy costs before materializing an export."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.lineage_hooks import LineageHookError
    from lancedb_robotics.projections import ProjectionError

    try:
        context = load_lineage_context(lineage_context)
        summary = Lake.open(lake).projections.materialization_summary(
            snapshot,
            formats=format_name or None,
            require_native=require_native_loader,
            shard_size=shard_size,
            compression=compression,
            lineage_context=context,
        )
    except (LakeError, ProjectionError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except LineageHookError as exc:
        raise lineage_context_error(exc) from exc

    typer.echo(f"lake: {lake}")
    typer.echo(f"snapshot: {summary['snapshot_name']}")
    for row in summary["formats"]:
        accounting = row["accounting"]
        typer.echo(f"format: {row['format']} ({row['format_version']})")
        typer.echo(f"  copied_payload_bytes: {accounting.get('payload_bytes_copied')}")
        typer.echo(f"  logical_reference_bytes: {accounting.get('logical_reference_bytes')}")
        typer.echo(f"  planned_copy_bytes: {accounting.get('payload_bytes_planned')}")
        typer.echo(f"  metadata_bytes_written: {accounting.get('metadata_bytes_written')}")
        typer.echo(f"  copy_ratio: {float(accounting.get('copy_ratio') or 0.0):.6f}")
        typer.echo(f"  transform: {row['transform_id']}")


def _echo_projection_manifest(manifest) -> None:
    typer.echo(f"lake: {manifest.lake_uri}")
    typer.echo(f"snapshot: {manifest.snapshot_name} ({manifest.source_snapshot_id})")
    typer.echo(f"mode: {manifest.mode.value}")
    typer.echo(f"format: {manifest.format} ({manifest.format_version})")
    typer.echo(f"features: {', '.join(sorted(manifest.feature_schema.get('features', {})))}")
    accounting = manifest.accounting or {}
    if accounting:
        typer.echo(f"logical rows: {accounting.get('logical_row_count')}")
        typer.echo(
            "payload bytes: "
            f"referenced={accounting.get('payload_bytes_referenced')} "
            f"copied={accounting.get('payload_bytes_copied')} "
            f"logical={accounting.get('logical_reference_bytes')} "
            f"planned_copy={accounting.get('payload_bytes_planned')} "
            f"metadata={accounting.get('metadata_bytes_written')} "
            f"copy_ratio={float(accounting.get('copy_ratio') or 0.0):.6f}"
        )
        if accounting.get("dry_run"):
            typer.echo("materialization: dry-run estimate")
        else:
            typer.echo(
                f"materialization: {accounting.get('payload_copy_policy')}"
            )
    dry_run = manifest.feature_schema.get("dry_run") or {}
    if dry_run:
        typer.echo(f"planned shards: {dry_run.get('shard_count')}")
        typer.echo(f"estimated bytes: {dry_run.get('estimated_bytes')}")
        sample_schema = dry_run.get("sample_schema") or {}
        typer.echo(f"sample schema: {', '.join(sorted(sample_schema))}")
    typer.echo(f"lossiness: {len(manifest.lossiness)}")
    typer.echo(f"transform: {manifest.transform_id}")


def _echo_native_loader(manifest) -> None:
    native_loader = manifest.media_policy.get("native_loader") or {}
    if not native_loader:
        return
    native = "available" if native_loader.get("available") else "unavailable"
    typer.echo(f"native loader: {native}")
    if not native_loader.get("available"):
        missing = ", ".join(native_loader.get("missing") or [])
        typer.echo(f"install: {native_loader.get('install')} (missing {missing})")


def _resolve_format_argument(format_arg: str | None, fmt: str) -> str:
    if format_arg is not None and fmt != "lerobot" and format_arg != fmt:
        from lancedb_robotics.projections import ProjectionError

        raise ProjectionError(
            f"conflicting dataset export formats: argument {format_arg!r} and --format {fmt!r}"
        )
    return format_arg or fmt


def _resolve_selection(lake, from_search, scenario_id, last_search):
    from lancedb_robotics.dataset import DatasetError

    if scenario_id:
        return list(scenario_id), {"kind": "explicit"}
    if from_search:
        if from_search != "last":
            raise DatasetError("only --from-search last is supported")
        recorded = last_search(lake)
        if recorded is None:
            raise DatasetError("no recorded search to snapshot; run a search first")
        return list(recorded.get("scenario_ids", [])), {"kind": "search-last", **recorded}
    raise DatasetError("select scenarios with --from-search last or --scenario-id")


def _source_label(source: dict) -> str:
    if source.get("kind") == "search-last":
        return f'search ({source.get("mode")} "{source.get("query")}")'
    return "explicit"
