"""`lancedb-robotics align` subcommands."""

import typer

align_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_RATE_OPTION = typer.Option(..., "--rate-hz", help="Target query clock rate in Hz.")
_STREAM_OPTION = typer.Option(
    None,
    "--stream",
    help="Observation stream to align; repeat for several.",
)
_CLOCK_OPTION = typer.Option(
    "timestamp_ns",
    "--clock",
    help="Clock basis: timestamp_ns, robot_time_ns, header_time_ns, receive_time_ns.",
)
_TOLERANCE_OPTION = typer.Option(
    None,
    "--tolerance-ms",
    help="Maximum source distance from a query tick in milliseconds.",
)
_INTERPOLATION_OPTION = typer.Option(
    None,
    "--interpolation",
    help="Per-stream interpolation as stream=nearest|previous|linear; repeat for several.",
)
_LATENCY_OPTION = typer.Option(
    None,
    "--latency",
    help="Per-stream transport latency correction as stream=5ms; repeat for several.",
)
_RUN_OPTION = typer.Option(None, "--run-id", help="Restrict alignment to one run.")
_START_OPTION = typer.Option(None, "--start-ns", help="Start timestamp for the query clock.")
_END_OPTION = typer.Option(None, "--end-ns", help="End timestamp for the query clock.")


@align_app.command("create")
def create(
    name: str = typer.Argument(..., help="Stable aligned view name."),
    lake: str = _LAKE_OPTION,
    rate_hz: float = _RATE_OPTION,
    stream: list[str] | None = _STREAM_OPTION,
    clock: str = _CLOCK_OPTION,
    tolerance_ms: float | None = _TOLERANCE_OPTION,
    interpolation: list[str] | None = _INTERPOLATION_OPTION,
    latency: list[str] | None = _LATENCY_OPTION,
    run_id: str | None = _RUN_OPTION,
    start_ns: int | None = _START_OPTION,
    end_ns: int | None = _END_OPTION,
) -> None:
    """Create a deterministic aligned observation view and record lineage."""

    from lancedb_robotics.align import AlignmentError
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.scenarios import ScenarioError, parse_duration_ns

    try:
        opened = Lake.open(lake)
        view = opened.align.create_view(
            name,
            rate_hz=rate_hz,
            streams=stream or [],
            clock=clock,
            tolerance_ms=tolerance_ms,
            interpolation=_parse_interpolation(interpolation or []),
            latency_ns=_parse_latency(latency or [], parse_duration_ns),
            run_id=run_id,
            start_time_ns=start_ns,
            end_time_ns=end_ns,
        )
    except (AlignmentError, LakeError, ScenarioError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_view(view)


_MIGRATE_ALIGNMENT_OPTION = typer.Option(
    None,
    "--alignment",
    help="Alignment id or name to migrate; repeat for several. Default: all recorded jobs.",
)
_MIGRATE_DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    help="Plan jobs and tick counts without writing rows or lineage.",
)
_MIGRATE_VERIFY_OPTION = typer.Option(
    True,
    "--verify/--no-verify",
    help="Validate stored tick rows (metadata signatures, JSONB round-trip, summaries).",
)
_MIGRATE_REPLACE_OPTION = typer.Option(
    False,
    "--replace",
    help="Rewrite existing tick rows; the remediation for stale or failed-validation rows.",
)
_MIGRATE_CREATE_TABLE_OPTION = typer.Option(
    False,
    "--create-table",
    help="Explicitly create a missing aligned_ticks table (capability-gated on remote lakes).",
)
_MIGRATE_TICK_WINDOW_OPTION = typer.Option(
    None,
    "--tick-window",
    help="Ticks per bounded scan/write window (default 1024).",
)
_MIGRATE_BATCH_SIZE_OPTION = typer.Option(
    None,
    "--batch-size",
    help="Rows per aligned_ticks append batch (default 512).",
)
_MIGRATE_JSON_OPTION = typer.Option(
    False,
    "--json",
    help="Emit the full aligned-tick-migration/1 report as JSON.",
)


@align_app.command("migrate-ticks")
def migrate_ticks(
    lake: str = _LAKE_OPTION,
    alignment: list[str] | None = _MIGRATE_ALIGNMENT_OPTION,
    dry_run: bool = _MIGRATE_DRY_RUN_OPTION,
    verify: bool = _MIGRATE_VERIFY_OPTION,
    replace: bool = _MIGRATE_REPLACE_OPTION,
    create_table: bool = _MIGRATE_CREATE_TABLE_OPTION,
    tick_window: int = _MIGRATE_TICK_WINDOW_OPTION,
    batch_size: int = _MIGRATE_BATCH_SIZE_OPTION,
    json_output: bool = _MIGRATE_JSON_OPTION,
) -> None:
    """Batch-migrate recorded alignments into aligned_ticks with a validation report."""

    import json as json_mod

    from lancedb_robotics.aligned_tick_migration import AlignedTickMigrationError
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.training import TrainingError

    try:
        opened = Lake.open(lake)
        report = opened.training.migrate_aligned_ticks(
            alignments=alignment or None,
            dry_run=dry_run,
            verify=verify,
            replace=replace,
            create_missing_table=create_table,
            tick_window=tick_window,
            batch_size=batch_size,
        )
    except (AlignedTickMigrationError, TrainingError, LakeError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json_mod.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _echo_migration_report(report)
    if report["jobs_failed"]:
        raise typer.Exit(code=1)


_LIFECYCLE_ALIGNMENT_OPTION = typer.Option(
    None,
    "--alignment",
    help="Alignment id or name to act on; repeat for several. Default: all.",
)
_LIFECYCLE_INCLUDE_FRAMES_OPTION = typer.Option(
    True,
    "--include-frames/--no-include-frames",
    help="Also diagnose/clean the compatibility aligned_frames rows.",
)
_LIFECYCLE_TICK_WINDOW_OPTION = typer.Option(
    None,
    "--tick-window",
    help="Ticks per bounded scan/dedup window (default 1024).",
)
_LIFECYCLE_JSON_OPTION = typer.Option(
    False,
    "--json",
    help="Emit the full aligned-tick-lifecycle/1 report as JSON.",
)
_CLEANUP_APPLY_OPTION = typer.Option(
    False,
    "--apply",
    help="Apply the plan. Without this, cleanup is a dry-run (the default).",
)
_CLEANUP_NO_DUPLICATES_OPTION = typer.Option(
    True,
    "--duplicates/--no-duplicates",
    help="Collapse duplicate aligned_tick_id/aligned_frame_id rows.",
)
_CLEANUP_ORPHANS_OPTION = typer.Option(
    False,
    "--remove-orphans",
    help="Also remove rows for alignments with no recorded alignment_jobs row.",
)
_CLEANUP_BATCH_SIZE_OPTION = typer.Option(
    None,
    "--batch-size",
    help="Rows per re-add append batch during dedup (default 512).",
)


@align_app.command("diagnose-ticks")
def diagnose_ticks(
    lake: str = _LAKE_OPTION,
    alignment: list[str] | None = _LIFECYCLE_ALIGNMENT_OPTION,
    include_frames: bool = _LIFECYCLE_INCLUDE_FRAMES_OPTION,
    tick_window: int = _LIFECYCLE_TICK_WINDOW_OPTION,
    json_output: bool = _LIFECYCLE_JSON_OPTION,
) -> None:
    """Report aligned_ticks/aligned_frames size, duplicates, and stale rows."""

    import json as json_mod

    from lancedb_robotics.aligned_tick_lifecycle import AlignedTickLifecycleError
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        report = opened.training.diagnose_aligned_ticks(
            alignments=alignment or None,
            include_frames=include_frames,
            tick_window=tick_window,
        )
    except (AlignedTickLifecycleError, LakeError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json_mod.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _echo_lifecycle_report(report)


@align_app.command("cleanup-ticks")
def cleanup_ticks(
    lake: str = _LAKE_OPTION,
    alignment: list[str] | None = _LIFECYCLE_ALIGNMENT_OPTION,
    apply: bool = _CLEANUP_APPLY_OPTION,
    duplicates: bool = _CLEANUP_NO_DUPLICATES_OPTION,
    remove_orphans: bool = _CLEANUP_ORPHANS_OPTION,
    include_frames: bool = _LIFECYCLE_INCLUDE_FRAMES_OPTION,
    tick_window: int = _LIFECYCLE_TICK_WINDOW_OPTION,
    batch_size: int = _CLEANUP_BATCH_SIZE_OPTION,
    json_output: bool = _LIFECYCLE_JSON_OPTION,
) -> None:
    """Compact and retention-clean aligned_ticks (dry-run unless --apply)."""

    import json as json_mod

    from lancedb_robotics.aligned_tick_lifecycle import AlignedTickLifecycleError
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        report = opened.training.cleanup_aligned_ticks(
            alignments=alignment or None,
            dry_run=not apply,
            remove_duplicates=duplicates,
            remove_orphans=remove_orphans,
            include_frames=include_frames,
            tick_window=tick_window,
            batch_size=batch_size,
        )
    except (AlignedTickLifecycleError, LakeError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json_mod.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _echo_lifecycle_report(report)


def _echo_lifecycle_report(report: dict) -> None:
    totals = report["totals"]
    mode = ""
    if "dry_run" in report:
        mode = " (dry-run)" if report["dry_run"] else " (applied)"
    typer.echo(f"aligned-tick lifecycle{mode}")
    tick_table = report["tables"]["aligned_ticks"]
    if tick_table.get("exists"):
        typer.echo(
            f"aligned_ticks: {tick_table['rows']} rows, "
            f"{tick_table['fragments']} fragments, "
            f"{len(tick_table.get('pinned_versions') or [])} pinned version(s)"
        )
    else:
        typer.echo("aligned_ticks: absent")
    typer.echo(
        "ticks: "
        f"{totals['duplicate_tick_rows']} duplicate, "
        f"{totals['stale_tick_rows']} stale, "
        f"{totals['ticks_needing_rematerialize']} need re-materialize"
    )
    frame_table = report["tables"].get("aligned_frames", {})
    if frame_table.get("exists"):
        typer.echo(
            f"aligned_frames: {frame_table['rows']} rows, "
            f"{totals['duplicate_frame_rows']} duplicate"
        )
    typer.echo(
        f"orphans: {totals['orphan_alignments']} alignment(s), "
        f"{totals['orphan_tick_rows']} tick rows"
    )
    plan = report.get("plan")
    if plan is not None:
        typer.echo(
            "plan: "
            f"{plan['duplicate_tick_rows_removable']} duplicate tick rows, "
            f"{plan['duplicate_frame_rows_removable']} duplicate frame rows, "
            f"{plan['orphan_tick_rows_removable']} orphan tick rows removable"
        )
    applied = report.get("applied")
    if applied is not None:
        typer.echo(
            "applied: "
            f"{applied['duplicate_tick_rows_removed']} duplicate tick rows, "
            f"{applied['duplicate_frame_rows_removed']} duplicate frame rows, "
            f"{applied['orphan_tick_rows_removed']} orphan tick rows removed"
        )
        typer.echo(f"transform: {applied['transform_id']}")


def _echo_migration_report(report: dict) -> None:
    typer.echo(f"migration: {report['migration_id']}{' (dry-run)' if report['dry_run'] else ''}")
    typer.echo(
        "jobs: "
        f"{report['jobs_scanned']} scanned, "
        f"{report['jobs_migrated']} migrated, "
        f"{report['jobs_already_migrated']} already-migrated, "
        f"{report['jobs_planned']} planned, "
        f"{report['jobs_skipped']} skipped, "
        f"{report['jobs_failed']} failed"
    )
    typer.echo(
        f"rows: {report['aligned_ticks_written']} aligned_ticks written "
        f"from {report['source_aligned_frame_rows']} aligned_frames rows"
    )
    validation = report["validation"]
    typer.echo(
        "validation: "
        f"{validation['metadata_mismatches']} metadata mismatches, "
        f"{validation['jsonb_failures']} JSONB failures, "
        f"{validation['summary_mismatches']} summary mismatches"
    )
    for job in report["jobs"]:
        line = f"  {job['alignment_id']} ({job['alignment_name']}): {job['status']}"
        if job.get("ticks_written") is not None:
            line += f", ticks_written={job['ticks_written']}"
        if job.get("reason"):
            line += f" - {job['reason']}"
        typer.echo(line)


def _parse_interpolation(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        stream, method = _split_assignment(value, "--interpolation")
        parsed[stream] = method
    return parsed


def _parse_latency(values: list[str], parse_duration_ns) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        stream, duration = _split_assignment(value, "--latency")
        parsed[stream] = parse_duration_ns(duration)
    return parsed


def _split_assignment(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} must be stream=value")
    key, raw = value.split("=", 1)
    if not key or not raw:
        raise ValueError(f"{option} must be stream=value")
    return key, raw


def _echo_view(view) -> None:
    typer.echo(f"lake: {view.lake_uri}")
    typer.echo(f"view: {view.name}")
    typer.echo(f"alignment: {view.alignment_id}")
    typer.echo(f"transform: {view.transform_id}")
    typer.echo(f"rows: {len(view.rows)}")
    typer.echo(f"streams: {', '.join(view.streams)}")
    typer.echo(f"confidence: {view.quality_summary['confidence']:.6f}")
    flags = ", ".join(view.quality_flags) if view.quality_flags else "none"
    typer.echo(f"quality flags: {flags}")
