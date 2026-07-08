"""`lancedb-robotics distributions` subcommands."""

import json
from datetime import timedelta
from pathlib import Path

import typer

distributions_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_DIMENSION_OPTION = typer.Option(..., "--dimension", help="Distribution dimension; repeat.")
_MIN_COUNT_OPTION = typer.Option(0, "--min-count-per-slice", help="Minimum desired rows per slice.")
_SNAPSHOT_OPTION = typer.Option(None, "--snapshot", help="Dataset snapshot source name.")
_VIEW_OPTION = typer.Option(None, "--view", help="Saved curation view source name.")
_SCENARIO_ID_OPTION = typer.Option(
    None, "--scenario-id", help="Explicit scenario source id; repeat for several."
)
_SPEC_NAME_OPTION = typer.Option("coverage", "--name", help="Distribution spec/report name.")
_FORMAT_OPTION = typer.Option("text", "--format", help="text, json, or markdown.")
_BATCH_SIZE_OPTION = typer.Option(4096, "--batch-size", help="Rows per Lance scan batch.")
_MAX_SLICE_COUNT_OPTION = typer.Option(
    None,
    "--max-slice-count",
    help="Fail or bucket when measured slice count exceeds this value.",
)
_OVERFLOW_OPTION = typer.Option("error", "--overflow", help="High-cardinality action: error or bucket.")
_RARE_SLICE_MIN_COUNT_OPTION = typer.Option(
    0,
    "--rare-slice-min-count",
    help="Bucket slices with fewer than this many rows.",
)
_TOP_K_OVERFLOW_OPTION = typer.Option(
    10,
    "--top-k-overflow",
    help="Number of overflow slice examples to record.",
)
_MAX_SCENARIO_IDS_OPTION = typer.Option(
    None,
    "--max-scenario-ids-per-slice",
    help="Maximum scenario ids persisted per output slice.",
)
_BIN_OPTION = typer.Option(
    None,
    "--bin",
    help="Explicit slice bin mapping as raw_label=bucket_label; repeatable.",
)
_OBSERVED_SNAPSHOT_OPTION = typer.Option(..., "--observed-snapshot", help="Observed snapshot name.")
_TARGET_MANIFEST_OPTION = typer.Option(..., "--target-manifest", help="External target JSON file.")
_SPEC_ID_OPTION = typer.Option(None, "--spec-id", help="Distribution spec id filter.")
_REPORT_ID_OPTION = typer.Option(..., "--report-id", help="Distribution report id.")
_SOURCE_KIND_OPTION = typer.Option(None, "--source-kind", help="Catalog source kind filter.")
_SOURCE_ID_OPTION = typer.Option(None, "--source-id", help="Catalog source id filter.")
_CREATED_BY_OPTION = typer.Option(None, "--created-by", help="Catalog creator filter.")
_LIMIT_OPTION = typer.Option(20, "--limit", help="Maximum catalog rows to print.")
_OLDER_THAN_DAYS_OPTION = typer.Option(
    30,
    "--older-than-days",
    help="Compact bodies created before this many days ago.",
)
_RETAIN_LATEST_OPTION = typer.Option(
    1,
    "--retain-latest-per-name",
    help="Keep this many newest bodies per report/comparison name and source.",
)


@distributions_app.command("measure")
def measure(
    lake: str = _LAKE_OPTION,
    dimension: list[str] = _DIMENSION_OPTION,
    min_count_per_slice: int = _MIN_COUNT_OPTION,
    snapshot: str | None = _SNAPSHOT_OPTION,
    view: str | None = _VIEW_OPTION,
    scenario_id: list[str] = _SCENARIO_ID_OPTION,
    name: str = _SPEC_NAME_OPTION,
    output_format: str = _FORMAT_OPTION,
    batch_size: int = _BATCH_SIZE_OPTION,
    max_slice_count: int | None = _MAX_SLICE_COUNT_OPTION,
    overflow: str = _OVERFLOW_OPTION,
    rare_slice_min_count: int = _RARE_SLICE_MIN_COUNT_OPTION,
    top_k_overflow: int = _TOP_K_OVERFLOW_OPTION,
    max_scenario_ids_per_slice: int | None = _MAX_SCENARIO_IDS_OPTION,
    bin: list[str] = _BIN_OPTION,
) -> None:
    """Measure a deterministic distribution report over a lake source."""
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        spec = opened.distributions.define(
            name=name,
            dimensions=dimension,
            min_count_per_slice=min_count_per_slice,
        )
        report = opened.distributions.measure(
            spec,
            source=_source(snapshot, view, scenario_id),
            batch_size=batch_size,
            max_slice_count=max_slice_count,
            overflow=overflow,
            rare_slice_min_count=rare_slice_min_count,
            top_k_overflow=top_k_overflow,
            max_scenario_ids_per_slice=max_scenario_ids_per_slice,
            slice_bins=_parse_bins(bin),
        )
    except Exception as exc:
        _exit(exc)
    _echo_report(report, output_format)


@distributions_app.command("compare")
def compare(
    lake: str = _LAKE_OPTION,
    observed_snapshot: str = _OBSERVED_SNAPSHOT_OPTION,
    target_manifest: str = _TARGET_MANIFEST_OPTION,
    dimension: list[str] = _DIMENSION_OPTION,
    min_count_per_slice: int = _MIN_COUNT_OPTION,
    name: str = _SPEC_NAME_OPTION,
    output_format: str = _FORMAT_OPTION,
) -> None:
    """Compare an observed snapshot against an external distribution manifest."""
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        target = json.loads(Path(target_manifest).read_text())
        target.setdefault("kind", "external-manifest")
        spec = opened.distributions.define(
            name=name,
            dimensions=dimension,
            min_count_per_slice=min_count_per_slice,
        )
        observed_report = opened.distributions.measure(
            spec,
            source={"kind": "snapshot", "name": observed_snapshot},
        )
        target_report = opened.distributions.measure(spec, source=target)
        comparison = opened.distributions.compare(observed=observed_report, target=target_report)
    except Exception as exc:
        _exit(exc)
    if output_format == "json":
        typer.echo(json.dumps(comparison.to_dict(), sort_keys=True, indent=2))
        return
    typer.echo(f"comparison: {comparison.comparison_id}")
    typer.echo(f"observed_report: {comparison.observed.report_id}")
    typer.echo(f"target_report: {comparison.target.report_id}")
    typer.echo(f"findings: {len(comparison.gap_findings)}")
    for finding in comparison.gap_findings:
        typer.echo(
            f"{finding.kind}: {finding.label} "
            f"observed={finding.observed_count} target={finding.target_count} "
            f"needed={finding.needed_count}"
        )
    typer.echo(f"transform: {comparison.transform_id}")


@distributions_app.command("list")
def list_reports(
    lake: str = _LAKE_OPTION,
    name: str | None = typer.Option(None, "--name", help="Distribution report name filter."),
    spec_id: str | None = _SPEC_ID_OPTION,
    source_kind: str | None = _SOURCE_KIND_OPTION,
    source_id: str | None = _SOURCE_ID_OPTION,
    created_by: str | None = _CREATED_BY_OPTION,
    limit: int = _LIMIT_OPTION,
    output_format: str = _FORMAT_OPTION,
) -> None:
    """List persisted distribution reports from the catalog."""
    try:
        from lancedb_robotics.lake import Lake

        opened = Lake.open(lake)
        entries = opened.distributions.list_reports(
            name=name,
            spec_id=spec_id,
            source_kind=source_kind,
            source_id=source_id,
            created_by=created_by,
            limit=limit,
        )
    except Exception as exc:
        _exit(exc)
    _echo_catalog(entries, output_format)


@distributions_app.command("get")
def get_report(
    lake: str = _LAKE_OPTION,
    report_id: str = _REPORT_ID_OPTION,
    output_format: str = _FORMAT_OPTION,
) -> None:
    """Fetch a persisted distribution report body by id."""
    try:
        from lancedb_robotics.lake import Lake

        report = Lake.open(lake).distributions.get_report(report_id)
    except Exception as exc:
        _exit(exc)
    _echo_report(report, output_format)


@distributions_app.command("latest")
def latest_report(
    lake: str = _LAKE_OPTION,
    name: str | None = typer.Option(None, "--name", help="Distribution report name filter."),
    spec_id: str | None = _SPEC_ID_OPTION,
    source_kind: str | None = _SOURCE_KIND_OPTION,
    source_id: str | None = _SOURCE_ID_OPTION,
    created_by: str | None = _CREATED_BY_OPTION,
    output_format: str = _FORMAT_OPTION,
) -> None:
    """Fetch the newest non-compacted distribution report matching filters."""
    try:
        from lancedb_robotics.lake import Lake

        report = Lake.open(lake).distributions.latest_report(
            name=name,
            spec_id=spec_id,
            source_kind=source_kind,
            source_id=source_id,
            created_by=created_by,
        )
    except Exception as exc:
        _exit(exc)
    _echo_report(report, output_format)


@distributions_app.command("compact")
def compact_reports(
    lake: str = _LAKE_OPTION,
    older_than_days: int = _OLDER_THAN_DAYS_OPTION,
    retain_latest_per_name: int = _RETAIN_LATEST_OPTION,
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview compaction without writing."),
    output_format: str = _FORMAT_OPTION,
) -> None:
    """Compact old report/comparison bodies while retaining audit metadata."""
    try:
        from lancedb_robotics.lake import Lake

        report = Lake.open(lake).distributions.compact_reports(
            older_than=timedelta(days=older_than_days),
            retain_latest_per_name=retain_latest_per_name,
            dry_run=dry_run,
        )
    except Exception as exc:
        _exit(exc)
    if output_format == "json":
        typer.echo(json.dumps(report.to_dict(), sort_keys=True, indent=2))
        return
    if output_format != "text":
        raise ValueError("--format must be text or json")
    typer.echo(f"compacted: {report.compacted_count}")
    typer.echo(f"body_bytes_before: {report.body_bytes_before}")
    typer.echo(f"body_bytes_after: {report.body_bytes_after}")
    if report.transform_id:
        typer.echo(f"transform: {report.transform_id}")


def _source(
    snapshot: str | None,
    view: str | None,
    scenario_id: list[str] | None,
):
    selected = sum(value is not None for value in (snapshot, view, scenario_id))
    if selected > 1:
        raise ValueError("choose only one of --snapshot, --view, or --scenario-id")
    if snapshot:
        return {"kind": "snapshot", "name": snapshot}
    if view:
        return {"kind": "view", "name": view}
    if scenario_id:
        return {"kind": "scenarios", "scenario_ids": scenario_id}
    return None


def _parse_bins(raw: list[str] | None) -> dict[str, str]:
    bins: dict[str, str] = {}
    for item in raw or ():
        if "=" not in item:
            raise ValueError("--bin must use raw_label=bucket_label")
        source, bucket = item.rsplit("=", 1)
        source = source.strip()
        bucket = bucket.strip()
        if not source or not bucket:
            raise ValueError("--bin must include non-empty raw and bucket labels")
        bins[source] = bucket
    return bins


def _echo_report(report, output_format: str) -> None:
    normalized = output_format.strip().lower()
    if normalized == "json":
        typer.echo(json.dumps(report.to_dict(), sort_keys=True, indent=2))
        return
    if normalized == "markdown":
        typer.echo(report.to_markdown())
        return
    if normalized != "text":
        raise ValueError("--format must be text, json, or markdown")
    typer.echo(f"lake: {report.lake.uri}")
    typer.echo(f"report: {report.report_id}")
    typer.echo(f"source: {report.source.get('kind', 'unknown')}")
    typer.echo(f"total: {report.total_count}")
    for label, count in report.slice_counts.items():
        typer.echo(f"slice: {label} count={count}")
    cardinality = report.execution.get("cardinality") or {}
    if cardinality.get("reason"):
        typer.echo(f"cardinality: {cardinality['reason']}")
    typer.echo(f"transform: {report.transform_id}")


def _echo_catalog(entries, output_format: str) -> None:
    normalized = output_format.strip().lower()
    if normalized == "json":
        typer.echo(json.dumps([entry.to_dict() for entry in entries], sort_keys=True, indent=2))
        return
    if normalized != "text":
        raise ValueError("--format must be text or json")
    for entry in entries:
        total = entry.summary.get("total_count", "")
        total_text = f" total={total}" if total != "" else ""
        typer.echo(
            f"report: {entry.report_id} name={entry.name} spec={entry.spec_id} "
            f"source={entry.source_kind}:{entry.source_id or entry.source_name}"
            f"{total_text} compacted={entry.body_compacted}"
        )


def _exit(exc: Exception) -> None:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1) from exc
