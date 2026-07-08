"""`lancedb-robotics episodes` subcommands."""

import json
from pathlib import Path
from typing import Any

import typer

episodes_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
derivations_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
episodes_app.add_typer(
    derivations_app,
    name="derivations",
    help="List, dry-run, rebuild, supersede, and clear episode derivations.",
)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_START_EVENT_OPTION = typer.Option(
    None,
    "--start-event",
    help="Marker event_type that starts an episode; repeat for several.",
)
_STOP_EVENT_OPTION = typer.Option(
    None,
    "--stop-event",
    help="Marker event_type that stops an episode; repeat for several.",
)
_OUTCOME_OPTION = typer.Option(
    None,
    "--outcome",
    help="Map an end event to an outcome as event_type=outcome; repeat for several.",
)
_MARKER_OUTCOME_OPTION = typer.Option(
    None,
    "--marker-outcome",
    help="Map an end marker event to an outcome as event_type=outcome; repeat for several.",
)
_EVENT_TYPE_OPTION = typer.Option(..., "--event-type", help="Event type that fires the query.")
_OPTIONAL_EVENT_TYPE_OPTION = typer.Option(
    None,
    "--event-type",
    help="Event type that fires the query recipe.",
)
_BEFORE_OPTION = typer.Option("0s", "--before", help="Window before the event, e.g. 250ms.")
_AFTER_OPTION = typer.Option("0s", "--after", help="Window after the event, e.g. 2s.")
_WHERE_OPTION = typer.Option(
    None,
    "--where",
    help="LanceDB SQL predicate over observations, e.g. task_id = 'pick'.",
)
_SOURCE_TABLE_OPTION = typer.Option(
    None,
    "--source-table",
    help="Auxiliary source table to mine: events, labels, or model_outputs.",
)
_SOURCE_WHERE_OPTION = typer.Option(
    None,
    "--source-where",
    help="LanceDB SQL predicate over the auxiliary source table.",
)
_MERGE_GAP_OPTION = typer.Option(
    None,
    "--merge-gap",
    help="Merge planned intervals separated by at most this duration, e.g. 500ms.",
)
_MIN_DURATION_OPTION = typer.Option(
    None,
    "--min-duration",
    help="Drop planned intervals shorter than this duration.",
)
_MAX_DURATION_OPTION = typer.Option(
    None,
    "--max-duration",
    help="Drop planned intervals longer than this duration.",
)
_TOPIC_OPTION = typer.Option(
    None,
    "--topic",
    help="Only tag observations from this topic; repeat for several.",
)
_MODALITY_OPTION = typer.Option(
    None,
    "--modality",
    help="Only tag observations with this modality; repeat for several.",
)
_DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    help="Report planned intervals without writing episodes, videos, or frame annotations.",
)
_SCENARIO_ID_OPTION = typer.Option(
    None,
    "--scenario-id",
    help="Scenario id to promote; repeat for several.",
)
_SNAPSHOT_OPTION = typer.Option(
    None,
    "--snapshot",
    help="Dataset snapshot name or id whose selected scenarios should be promoted.",
)
_STATIC_OUTCOME_OPTION = typer.Option(
    None,
    "--outcome",
    help="Static outcome to assign to promoted scenario episodes.",
)
_STATIC_TASK_ID_OPTION = typer.Option(
    None,
    "--task-id",
    help="Static task_id to assign to promoted scenario episodes.",
)
_OUTCOME_TAG_OPTION = typer.Option(
    None,
    "--outcome-tag",
    help="Map a scenario coverage tag to an outcome as tag=outcome; repeat for several.",
)
_OUTCOME_SUMMARY_OPTION = typer.Option(
    None,
    "--outcome-summary",
    help="Map a scenario summary to an outcome as summary=outcome; repeat for several.",
)
_TASK_TAG_OPTION = typer.Option(
    None,
    "--task-tag",
    help="Map a scenario coverage tag to a task_id as tag=task_id; repeat for several.",
)
_TASK_SUMMARY_OPTION = typer.Option(
    None,
    "--task-summary",
    help="Map a scenario summary to a task_id as summary=task_id; repeat for several.",
)
_INTERVAL_FILE_OPTION = typer.Option(
    ...,
    "--file",
    help="JSONL or CSV interval manifest to import.",
)
_OPTIONAL_INTERVAL_FILE_OPTION = typer.Option(
    None,
    "--file",
    help="JSONL or CSV interval manifest for interval recipes.",
)
_INTERVAL_FORMAT_OPTION = typer.Option(
    None,
    "--format",
    help="Interval manifest format: jsonl or csv. Defaults to the file extension.",
)
_TIME_UNIT_OPTION = typer.Option(
    "ns",
    "--time-unit",
    help="Unit for generic numeric interval fields: ns, us, ms, or s.",
)
_ALLOW_CLIPPED_OPTION = typer.Option(
    False,
    "--allow-clipped",
    help="Clip intervals to run bounds instead of failing when they cross run bounds.",
)
_ALLOW_EMPTY_OPTION = typer.Option(
    False,
    "--allow-empty",
    help="Allow imported intervals that contain no observations.",
)
_OVERLAP_OPTION = typer.Option(
    "error",
    "--overlap",
    help="Overlap policy: error, replace, supersede, or preserve.",
)
_RECIPE_ARGUMENT = typer.Argument(
    ...,
    help="Recipe kind (markers, query, scenarios, intervals, predicate) or transform id.",
)
_TRANSFORM_ARGUMENT = typer.Argument(..., help="Episode derivation transform id.")
_WITH_RECIPE_OPTION = typer.Option(
    ...,
    "--with",
    help="Replacement recipe kind or transform id.",
)


@episodes_app.command("from-markers")
def from_markers(
    lake: str = _LAKE_OPTION,
    start_event: list[str] | None = _START_EVENT_OPTION,
    stop_event: list[str] | None = _STOP_EVENT_OPTION,
    outcome: list[str] | None = _OUTCOME_OPTION,
    overlap: str = _OVERLAP_OPTION,
) -> None:
    """Derive episodes from explicit start/stop marker events."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        report = opened.episodes.from_markers(
            start_event_types=start_event,
            stop_event_types=stop_event,
            outcome_by_event_type=_parse_outcomes(outcome or []),
            overlap_policy=overlap,
        )
    except (LakeError, EpisodeError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_report(report)


@episodes_app.command("from-query")
def from_query(
    lake: str = _LAKE_OPTION,
    event_type: str = _EVENT_TYPE_OPTION,
    before: str = _BEFORE_OPTION,
    after: str = _AFTER_OPTION,
    overlap: str = _OVERLAP_OPTION,
) -> None:
    """Mine episodes around events matching a firing condition."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.scenarios import ScenarioError, parse_duration_ns

    try:
        opened = Lake.open(lake)
        report = opened.episodes.from_query(
            event_type=event_type,
            before_ns=parse_duration_ns(before),
            after_ns=parse_duration_ns(after),
            overlap_policy=overlap,
        )
    except (LakeError, EpisodeError, ScenarioError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_report(report)


@episodes_app.command("mine")
def mine(
    lake: str = _LAKE_OPTION,
    where: str | None = _WHERE_OPTION,
    source_table: str | None = _SOURCE_TABLE_OPTION,
    source_where: str | None = _SOURCE_WHERE_OPTION,
    before: str = _BEFORE_OPTION,
    after: str = _AFTER_OPTION,
    merge_gap: str | None = _MERGE_GAP_OPTION,
    min_duration: str | None = _MIN_DURATION_OPTION,
    max_duration: str | None = _MAX_DURATION_OPTION,
    topic: list[str] | None = _TOPIC_OPTION,
    modality: list[str] | None = _MODALITY_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
    overlap: str = _OVERLAP_OPTION,
) -> None:
    """Mine episodes from frame, event, label, or model-output predicates."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.scenarios import ScenarioError, parse_duration_ns

    try:
        opened = Lake.open(lake)
        report = opened.episodes.from_predicate(
            where=where,
            source_table=source_table,
            source_where=source_where,
            before_ns=parse_duration_ns(before),
            after_ns=parse_duration_ns(after),
            merge_gap_ns=parse_duration_ns(merge_gap) if merge_gap else None,
            min_duration_ns=parse_duration_ns(min_duration) if min_duration else None,
            max_duration_ns=parse_duration_ns(max_duration) if max_duration else None,
            topics=topic,
            modalities=modality,
            dry_run=dry_run,
            overlap_policy=overlap,
        )
    except (LakeError, EpisodeError, ScenarioError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_report(report)


@episodes_app.command("from-scenarios")
def from_scenarios(
    lake: str = _LAKE_OPTION,
    scenario_id: list[str] | None = _SCENARIO_ID_OPTION,
    snapshot: str | None = _SNAPSHOT_OPTION,
    outcome: str | None = _STATIC_OUTCOME_OPTION,
    task_id: str | None = _STATIC_TASK_ID_OPTION,
    outcome_tag: list[str] | None = _OUTCOME_TAG_OPTION,
    outcome_summary: list[str] | None = _OUTCOME_SUMMARY_OPTION,
    task_tag: list[str] | None = _TASK_TAG_OPTION,
    task_summary: list[str] | None = _TASK_SUMMARY_OPTION,
    overlap: str = _OVERLAP_OPTION,
) -> None:
    """Promote scenario windows or a snapshot selection into physical episodes."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        report = opened.episodes.from_scenarios(
            scenario_ids=scenario_id,
            snapshot_name=snapshot,
            outcome=outcome,
            task_id=task_id,
            outcome_by_coverage_tag=_parse_assignments(outcome_tag or [], "--outcome-tag"),
            outcome_by_summary=_parse_assignments(outcome_summary or [], "--outcome-summary"),
            task_id_by_coverage_tag=_parse_assignments(task_tag or [], "--task-tag"),
            task_id_by_summary=_parse_assignments(task_summary or [], "--task-summary"),
            overlap_policy=overlap,
        )
    except (LakeError, EpisodeError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_report(report)


@episodes_app.command("import-intervals")
def import_intervals(
    lake: str = _LAKE_OPTION,
    file: Path = _INTERVAL_FILE_OPTION,
    manifest_format: str | None = _INTERVAL_FORMAT_OPTION,
    time_unit: str = _TIME_UNIT_OPTION,
    allow_clipped: bool = _ALLOW_CLIPPED_OPTION,
    allow_empty: bool = _ALLOW_EMPTY_OPTION,
    overlap: str = _OVERLAP_OPTION,
) -> None:
    """Import externally authored episode intervals from JSONL or CSV."""
    from lancedb_robotics.episodes import EpisodeError, load_interval_manifest
    from lancedb_robotics.lake import Lake, LakeError

    try:
        manifest = load_interval_manifest(file, format=manifest_format)
        opened = Lake.open(lake)
        report = opened.episodes.from_intervals(
            manifest.records,
            time_unit=time_unit,
            allow_clipped=allow_clipped,
            allow_empty=allow_empty,
            source_uri=manifest.source_uri,
            source_sha256=manifest.sha256,
            overlap_policy=overlap,
        )
    except (OSError, LakeError, EpisodeError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_report(report)


@derivations_app.command("list")
def derivations_list(lake: str = _LAKE_OPTION) -> None:
    """List recorded episode derivations."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        rows = opened.episodes.list_derivations()
    except (LakeError, EpisodeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for row in rows:
        _echo_json(row.to_dict())


@derivations_app.command("show")
def derivations_show(
    transform_id: str = _TRANSFORM_ARGUMENT,
    lake: str = _LAKE_OPTION,
) -> None:
    """Show one episode derivation."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        detail = opened.episodes.show_derivation(transform_id)
    except (LakeError, EpisodeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_json(detail.to_dict())


@derivations_app.command("dry-run")
def derivations_dry_run(
    recipe: str = _RECIPE_ARGUMENT,
    lake: str = _LAKE_OPTION,
    event_type: str | None = _OPTIONAL_EVENT_TYPE_OPTION,
    before: str = _BEFORE_OPTION,
    after: str = _AFTER_OPTION,
    where: str | None = _WHERE_OPTION,
    source_table: str | None = _SOURCE_TABLE_OPTION,
    source_where: str | None = _SOURCE_WHERE_OPTION,
    merge_gap: str | None = _MERGE_GAP_OPTION,
    min_duration: str | None = _MIN_DURATION_OPTION,
    max_duration: str | None = _MAX_DURATION_OPTION,
    topic: list[str] | None = _TOPIC_OPTION,
    modality: list[str] | None = _MODALITY_OPTION,
    start_event: list[str] | None = _START_EVENT_OPTION,
    stop_event: list[str] | None = _STOP_EVENT_OPTION,
    outcome: list[str] | None = _MARKER_OUTCOME_OPTION,
    scenario_id: list[str] | None = _SCENARIO_ID_OPTION,
    snapshot: str | None = _SNAPSHOT_OPTION,
    static_outcome: str | None = _STATIC_OUTCOME_OPTION,
    task_id: str | None = _STATIC_TASK_ID_OPTION,
    interval_file: Path | None = _OPTIONAL_INTERVAL_FILE_OPTION,
    manifest_format: str | None = _INTERVAL_FORMAT_OPTION,
    time_unit: str = _TIME_UNIT_OPTION,
    allow_clipped: bool = _ALLOW_CLIPPED_OPTION,
    allow_empty: bool = _ALLOW_EMPTY_OPTION,
    overlap: str = _OVERLAP_OPTION,
) -> None:
    """Dry-run a derivation recipe."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.scenarios import ScenarioError

    try:
        opened = Lake.open(lake)
        report = opened.episodes.dry_run(
            _recipe_from_cli(
                recipe,
                event_type=event_type,
                before=before,
                after=after,
                where=where,
                source_table=source_table,
                source_where=source_where,
                merge_gap=merge_gap,
                min_duration=min_duration,
                max_duration=max_duration,
                topic=topic,
                modality=modality,
                start_event=start_event,
                stop_event=stop_event,
                outcome=outcome,
                scenario_id=scenario_id,
                snapshot=snapshot,
                static_outcome=static_outcome,
                task_id=task_id,
                interval_file=interval_file,
                manifest_format=manifest_format,
                time_unit=time_unit,
                allow_clipped=allow_clipped,
                allow_empty=allow_empty,
            ),
            overlap_policy=overlap,
        )
    except (OSError, LakeError, EpisodeError, ScenarioError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_dry_run(report)


@derivations_app.command("rebuild")
def derivations_rebuild(
    transform_id: str = _TRANSFORM_ARGUMENT,
    lake: str = _LAKE_OPTION,
    overlap: str = typer.Option("replace", "--overlap"),
) -> None:
    """Rebuild a recorded derivation."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        report = opened.episodes.rebuild(transform_id, overlap_policy=overlap)
    except (LakeError, EpisodeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_report(report)


@derivations_app.command("supersede")
def derivations_supersede(
    old_transform_id: str = _TRANSFORM_ARGUMENT,
    lake: str = _LAKE_OPTION,
    with_recipe: str = _WITH_RECIPE_OPTION,
    event_type: str | None = _OPTIONAL_EVENT_TYPE_OPTION,
    before: str = _BEFORE_OPTION,
    after: str = _AFTER_OPTION,
    where: str | None = _WHERE_OPTION,
    source_table: str | None = _SOURCE_TABLE_OPTION,
    source_where: str | None = _SOURCE_WHERE_OPTION,
    merge_gap: str | None = _MERGE_GAP_OPTION,
    min_duration: str | None = _MIN_DURATION_OPTION,
    max_duration: str | None = _MAX_DURATION_OPTION,
    topic: list[str] | None = _TOPIC_OPTION,
    modality: list[str] | None = _MODALITY_OPTION,
    start_event: list[str] | None = _START_EVENT_OPTION,
    stop_event: list[str] | None = _STOP_EVENT_OPTION,
    outcome: list[str] | None = _MARKER_OUTCOME_OPTION,
    scenario_id: list[str] | None = _SCENARIO_ID_OPTION,
    snapshot: str | None = _SNAPSHOT_OPTION,
    static_outcome: str | None = _STATIC_OUTCOME_OPTION,
    task_id: str | None = _STATIC_TASK_ID_OPTION,
    interval_file: Path | None = _OPTIONAL_INTERVAL_FILE_OPTION,
    manifest_format: str | None = _INTERVAL_FORMAT_OPTION,
    time_unit: str = _TIME_UNIT_OPTION,
    allow_clipped: bool = _ALLOW_CLIPPED_OPTION,
    allow_empty: bool = _ALLOW_EMPTY_OPTION,
) -> None:
    """Supersede a derivation with a replacement recipe."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.scenarios import ScenarioError

    try:
        opened = Lake.open(lake)
        report = opened.episodes.supersede(
            old_transform_id,
            _recipe_from_cli(
                with_recipe,
                event_type=event_type,
                before=before,
                after=after,
                where=where,
                source_table=source_table,
                source_where=source_where,
                merge_gap=merge_gap,
                min_duration=min_duration,
                max_duration=max_duration,
                topic=topic,
                modality=modality,
                start_event=start_event,
                stop_event=stop_event,
                outcome=outcome,
                scenario_id=scenario_id,
                snapshot=snapshot,
                static_outcome=static_outcome,
                task_id=task_id,
                interval_file=interval_file,
                manifest_format=manifest_format,
                time_unit=time_unit,
                allow_clipped=allow_clipped,
                allow_empty=allow_empty,
            ),
        )
    except (OSError, LakeError, EpisodeError, ScenarioError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_report(report)


@derivations_app.command("clear")
def derivations_clear(
    transform_id: str = _TRANSFORM_ARGUMENT,
    lake: str = _LAKE_OPTION,
) -> None:
    """Clear a derivation's output rows and safe frame annotations."""
    from lancedb_robotics.episodes import EpisodeError
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        report = opened.episodes.clear(transform_id)
    except (LakeError, EpisodeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _echo_json(report.to_dict())


def _recipe_from_cli(
    recipe: str,
    *,
    event_type: str | None,
    before: str,
    after: str,
    where: str | None,
    source_table: str | None,
    source_where: str | None,
    merge_gap: str | None,
    min_duration: str | None,
    max_duration: str | None,
    topic: list[str] | None,
    modality: list[str] | None,
    start_event: list[str] | None,
    stop_event: list[str] | None,
    outcome: list[str] | None,
    scenario_id: list[str] | None,
    snapshot: str | None,
    static_outcome: str | None,
    task_id: str | None,
    interval_file: Path | None,
    manifest_format: str | None,
    time_unit: str,
    allow_clipped: bool,
    allow_empty: bool,
) -> dict[str, Any] | str:
    from lancedb_robotics.episodes import EpisodeError, load_interval_manifest
    from lancedb_robotics.scenarios import parse_duration_ns

    if recipe.startswith("tfm-"):
        return recipe
    kind = recipe.strip().lower().removeprefix("episode-")
    if kind in {"marker", "markers"}:
        return {
            "kind": "markers",
            "start_event_types": start_event,
            "stop_event_types": stop_event,
            "outcome_by_event_type": _parse_outcomes(outcome or []),
        }
    if kind == "query":
        if not event_type:
            raise EpisodeError("--event-type is required for query recipes")
        return {
            "kind": "query",
            "event_type": event_type,
            "before_ns": parse_duration_ns(before),
            "after_ns": parse_duration_ns(after),
        }
    if kind == "predicate":
        return {
            "kind": "predicate",
            "where": where,
            "source_table": source_table,
            "source_where": source_where,
            "before_ns": parse_duration_ns(before),
            "after_ns": parse_duration_ns(after),
            "merge_gap_ns": parse_duration_ns(merge_gap) if merge_gap else None,
            "min_duration_ns": parse_duration_ns(min_duration) if min_duration else None,
            "max_duration_ns": parse_duration_ns(max_duration) if max_duration else None,
            "topics": topic,
            "modalities": modality,
        }
    if kind in {"scenario", "scenarios"}:
        return {
            "kind": "scenarios",
            "scenario_ids": scenario_id,
            "snapshot_name": snapshot,
            "outcome": static_outcome,
            "task_id": task_id,
        }
    if kind in {"interval", "intervals"}:
        if interval_file is None:
            raise EpisodeError("--file is required for interval recipes")
        manifest = load_interval_manifest(interval_file, format=manifest_format)
        return {
            "kind": "intervals",
            "records": list(manifest.records),
            "time_unit": time_unit,
            "allow_clipped": allow_clipped,
            "allow_empty": allow_empty,
            "source_uri": manifest.source_uri,
            "source_sha256": manifest.sha256,
        }
    raise EpisodeError(
        "recipe must be one of markers, query, scenarios, intervals, predicate, or a transform id"
    )


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, sort_keys=True, default=str))


def _echo_dry_run(report: Any) -> None:
    if hasattr(report, "to_dict"):
        payload = report.to_dict()
        typer.echo(f"lake: {payload['lake_uri']}")
        typer.echo(f"boundary source: {payload['boundary_source']}")
        typer.echo("dry-run: true")
        typer.echo(f"planned episodes: {payload['episodes_planned']}")
        typer.echo(f"planned videos: {payload['videos_planned']}")
        typer.echo(f"planned frames: {payload['frames_planned']}")
        typer.echo(f"overlap conflicts: {len(payload['overlap_conflicts'])}")
        typer.echo(f"transform: {payload['transform_id']}")
        _echo_json(payload)
        return
    _echo_report(report)


def _parse_outcomes(values: list[str]) -> dict[str, str]:
    return _parse_assignments(values, "--outcome")


def _parse_assignments(values: list[str], option_name: str) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} must be key=value")
        key, mapped = value.split("=", 1)
        if not key or not mapped:
            raise ValueError(f"{option_name} must be key=value")
        outcomes[key] = mapped
    return outcomes


def _echo_report(report) -> None:
    typer.echo(f"lake: {report.lake_uri}")
    typer.echo(f"boundary source: {report.boundary_source}")
    if getattr(report, "dry_run", False):
        typer.echo("dry-run: true")
        typer.echo(f"planned intervals: {report.intervals_planned}")
        typer.echo(f"planned frames: {report.frames_planned}")
        overlap_conflicts = getattr(report, "overlap_conflicts", ())
        typer.echo(f"overlap conflicts: {len(overlap_conflicts)}")
    typer.echo(f"episodes: {report.episodes_written}")
    typer.echo(f"videos: {report.videos_written}")
    typer.echo(f"frames tagged: {report.frames_tagged}")
    typer.echo(f"transform: {report.transform_id}")
    if getattr(report, "dry_run", False):
        for conflict in getattr(report, "overlap_conflicts", ()):
            typer.echo(json.dumps(conflict.to_dict(), sort_keys=True))
        for interval in report.intervals:
            typer.echo(json.dumps(interval.to_dict(), sort_keys=True))
