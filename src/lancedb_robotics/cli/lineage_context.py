"""Shared CLI parsing for external lineage context options."""

from __future__ import annotations

import typer

from lancedb_robotics.lineage_hooks import (
    LineageContext,
    LineageHookError,
    lineage_context_from_env,
    normalize_lineage_context,
)

LINEAGE_CONTEXT_OPTION = typer.Option(
    None,
    "--lineage-context",
    help=(
        "External lineage context as inline JSON, a JSON file path, or adapter name. "
        "Falls back to LANCEDB_ROBOTICS_LINEAGE_CONTEXT, "
        "LANCEDB_ROBOTICS_LINEAGE_CONTEXT_FILE, or LANCEDB_ROBOTICS_LINEAGE_ADAPTER."
    ),
)


def load_lineage_context(value: str | None) -> LineageContext | None:
    """Load explicit CLI context or the default environment context."""

    if value:
        return normalize_lineage_context(value)
    return lineage_context_from_env()


def lineage_context_error(exc: LineageHookError) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(code=1)


def echo_emitted_lineage(opened: object, transform_id: str | None) -> None:
    """Print the inline-emitted lineage summary for a write command (backlog 0098).

    Every canonical write path emits its lineage graph slice as part of the same
    operation, so a write command can show the emitted execution id + artifact
    count without asking the user to run ``lineage refresh`` first. Best-effort: a
    lineage read must never fail the write command's own success reporting.
    """

    if not transform_id:
        return
    try:
        summary = opened.lineage.emitted_transform_summary(transform_id)
    except Exception:  # noqa: BLE001 - lineage read must not break the command
        return
    if summary is None:
        return
    typer.echo(
        f"lineage: execution={summary.execution_id} "
        f"status={summary.status} artifacts={summary.artifact_count}"
    )
