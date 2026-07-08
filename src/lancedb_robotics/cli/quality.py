"""`lancedb-robotics quality` subcommands."""

import typer

quality_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_PROFILE_OPTION = typer.Option(
    "demo", "--profile", help="Built-in profile name or path to a profile JSON file."
)
_RUN_OPTION = typer.Option(None, "--run", help="Validate a single run id instead of every run.")
_WRITE_OPTION = typer.Option(
    True,
    "--write/--dry-run",
    help="Write flags and quarantine back to the lake, or report only.",
)
_QUARANTINE_OPTION = typer.Option(
    True, "--quarantine/--no-quarantine", help="Mark failed runs as quarantined."
)


@quality_app.command("validate")
def quality_validate_command(
    lake: str = _LAKE_OPTION,
    profile: str = _PROFILE_OPTION,
    run: str = _RUN_OPTION,
    write: bool = _WRITE_OPTION,
    quarantine: bool = _QUARANTINE_OPTION,
) -> None:
    """Validate ingested runs against a quality profile and quarantine failures.

    Exit codes: 0 = every validated run passed, 1 = operational error (missing
    lake, unknown profile, unknown run), 2 = at least one run failed validation
    (results are still written unless --dry-run is given).
    """
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.quality import (
        ProfileError,
        QualityError,
        apply_quality_results,
        resolve_profile,
        validate_lake,
    )

    try:
        opened = Lake.open(lake)
        resolved_profile = resolve_profile(profile)
        reports = validate_lake(opened, resolved_profile, run_id=run)
    except (LakeError, ProfileError, QualityError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if write:
        apply_quality_results(opened, reports, resolved_profile, quarantine=quarantine)

    typer.echo(f"lake: {opened.uri}")
    typer.echo(f"profile: {resolved_profile.name}" + ("" if write else " (dry-run)"))
    failed = [r for r in reports if not r.passed]
    for report in reports:
        if report.passed:
            typer.echo(f"run {report.run_id}: passed")
            continue
        quarantined = quarantine and write
        typer.echo(f"run {report.run_id}: FAILED" + (" (quarantined)" if quarantined else ""))
        for rule in report.rules:
            typer.echo(f"  {rule.rule}: {rule.status}")
            if rule.status == "failed":
                for detail in rule.details:
                    typer.echo(f"    {detail}")
    quarantined_count = len(failed) if quarantine and write else 0
    typer.echo(
        f"runs: {len(reports)} validated, {len(reports) - len(failed)} passed, "
        f"{len(failed)} failed, {quarantined_count} quarantined"
    )
    if failed:
        raise typer.Exit(code=2)
