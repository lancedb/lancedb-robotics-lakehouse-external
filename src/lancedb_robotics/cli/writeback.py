"""`lancedb-robotics writeback` subcommands."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import typer

writeback_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_INPUT_ARGUMENT = typer.Argument(..., help="JSON file containing a row object, row list, or envelope.")
_SOURCE_OPTION = typer.Option(None, "--source", help="External tool/source name for lineage.")
_CREATED_BY_OPTION = typer.Option("lancedb-robotics", "--created-by", help="User or service name.")
_AUTH_REF_OPTION = typer.Option(
    None,
    "--auth-ref",
    help="Credential reference name; secrets resolve from matching environment/config, never lake rows.",
)
_STORAGE_OPTION = typer.Option(
    None,
    "--storage-option",
    help="Storage client option as key=value; repeat for endpoint_url, region, etc.",
)


def _open_lake(lake: str, auth_ref: str | None, storage_option: list[str] | None):
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.storage import parse_storage_option_pairs

    try:
        storage_options = parse_storage_option_pairs(storage_option)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        return Lake.open(lake, auth_ref=auth_ref, storage_options=storage_options)
    except LakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _print_report(report) -> None:
    typer.echo(f"kind: {report.kind}")
    typer.echo(f"rows: {report.rows_written}")
    typer.echo(f"tables: {', '.join(report.output_tables)}")
    if report.target_tables:
        typer.echo(f"targets: {', '.join(report.target_tables)}")
    typer.echo(f"transform: {report.transform_id}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


@writeback_app.command("labels")
def labels(
    input_file: Path = _INPUT_ARGUMENT,
    lake: str = _LAKE_OPTION,
    source: str | None = _SOURCE_OPTION,
    created_by: str = _CREATED_BY_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Import labels from a labeling-tool/export JSON file."""
    from lancedb_robotics.writeback import WritebackError, import_labels, load_writeback_rows

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        report = import_labels(
            opened,
            load_writeback_rows(input_file, key="labels"),
            source=source or "label-import",
            created_by=created_by,
        )
    except WritebackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_report(report)


@writeback_app.command("model-outputs")
def model_outputs(
    input_file: Path = _INPUT_ARGUMENT,
    lake: str = _LAKE_OPTION,
    source: str | None = _SOURCE_OPTION,
    created_by: str = _CREATED_BY_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Ingest model predictions/inferences from a JSON file."""
    from lancedb_robotics.writeback import (
        WritebackError,
        ingest_model_outputs,
        load_writeback_rows,
    )

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        report = ingest_model_outputs(
            opened,
            load_writeback_rows(input_file, key="model_outputs"),
            source=source or "model-output-import",
            created_by=created_by,
        )
    except WritebackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_report(report)


@writeback_app.command("feedback")
def feedback(
    input_file: Path = _INPUT_ARGUMENT,
    lake: str = _LAKE_OPTION,
    source: str | None = _SOURCE_OPTION,
    created_by: str = _CREATED_BY_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Record fleet, simulation, incident, or review feedback signals."""
    from lancedb_robotics.writeback import WritebackError, load_writeback_rows, record_feedback

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        report = record_feedback(
            opened,
            load_writeback_rows(input_file, key="feedback"),
            source=source or "feedback-import",
            created_by=created_by,
        )
    except WritebackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _print_report(report)


@writeback_app.command("trace-model-output")
def trace_model_output(
    model_output_id: str = typer.Argument(..., help="Model output id to trace."),
    lake: str = _LAKE_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Trace one model output back to its input, snapshot, source run, and transforms."""
    from lancedb_robotics.writeback import WritebackError
    from lancedb_robotics.writeback import trace_model_output as trace

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        payload = trace(opened, model_output_id)
    except WritebackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


@writeback_app.command("downstream-run")
def downstream_run(
    run_id: str = typer.Argument(..., help="Run id to inspect."),
    lake: str = _LAKE_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """List downstream labels, model outputs, and feedback for one run."""
    from lancedb_robotics.writeback import WritebackError, downstream_for_run

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        payload = downstream_for_run(opened, run_id)
    except WritebackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
