"""`lancedb-robotics export` subcommands."""

from pathlib import Path

import typer

from lancedb_robotics.cli.lineage_context import (
    LINEAGE_CONTEXT_OPTION,
    lineage_context_error,
    load_lineage_context,
)

export_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_SNAPSHOT_OPTION = typer.Option(..., "--snapshot", help="Snapshot name to export.")
_OUT_OPTION = typer.Option(..., "--out", help="Output directory for clips and the manifest.")
_PLAN_ONLY_OPTION = typer.Option(
    False, "--plan-only", help="Write the manifest only; do not emit clip files."
)


@export_app.command("mcap")
def mcap(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    out: Path = _OUT_OPTION,
    plan_only: bool = _PLAN_ONLY_OPTION,
    lineage_context: str | None = LINEAGE_CONTEXT_OPTION,
) -> None:
    """Export a snapshot's scenario windows as MCAP clips plus a manifest."""
    from lancedb_robotics.export import MANIFEST_FILENAME, ExportError, export_snapshot
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.lineage_hooks import LineageHookError

    try:
        context = load_lineage_context(lineage_context)
        opened = Lake.open(lake)
        manifest = export_snapshot(
            opened,
            snapshot,
            out_dir=out,
            plan_only=plan_only,
            lineage_context=context,
        )
    except (LakeError, ExportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except LineageHookError as exc:
        raise lineage_context_error(exc) from exc

    typer.echo(f"lake: {manifest.lake_uri}")
    typer.echo(f"snapshot: {manifest.snapshot_name} ({manifest.dataset_id})")
    typer.echo(f"format: {manifest.format}")
    typer.echo(f"out: {manifest.out_dir}")
    typer.echo(
        f"clips: {len(manifest.clips)} "
        f"(exported {manifest.exported}, skipped {manifest.skipped}, planned {manifest.planned})"
    )
    typer.echo(f"manifest: {Path(manifest.out_dir) / MANIFEST_FILENAME}")
    for index, clip in enumerate(manifest.clips, start=1):
        target = clip.out_path or "(not written)"
        detail = clip.lossiness if clip.status == "exported" else (clip.reason or clip.status)
        typer.echo(
            f"{index}. {clip.scenario_id}  [{clip.start_time_ns}..{clip.end_time_ns} ns]  "
            f"-> {target}  ({clip.status}: {detail})"
        )
        typer.echo(f"     source: {clip.source_uri or 'unknown'}")
