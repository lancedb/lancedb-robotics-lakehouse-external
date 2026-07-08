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
