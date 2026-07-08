"""`lancedb-robotics lineage` subcommands."""

import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import typer

lineage_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_WHERE_OPTION = typer.Option(
    None,
    "--where",
    help="Optional slice predicate, e.g. topic = '/imu' or task_id = 'cup_grasp'.",
)
_LIMIT_OPTION = typer.Option(None, "--limit", help="Maximum offending rows to include.")
_MAX_DEPTH_OPTION = typer.Option(None, "--max-depth", help="Maximum graph traversal depth.")
_EDGE_TYPE_OPTION = typer.Option(
    None,
    "--edge-type",
    help="Limit graph traversal to an edge type; repeat for multiple types.",
)
_TARGET_KIND_OPTION = typer.Option(
    None,
    "--target-kind",
    help="Return only paths to this artifact kind; repeat for multiple kinds.",
)
_KIND_OPTION = typer.Option(
    None,
    "--kind",
    help="Disambiguate the artifact handle, e.g. source, run, snapshot, model, transform.",
)
_FORMAT_OPTION = typer.Option(
    "tree",
    "--format",
    help="Output format: tree, json, ndjson, or summary.",
)
_PLAN_FORMAT_OPTION = typer.Option(
    "json",
    "--format",
    help="Output format: json for automation, tree for operators.",
)
_CREATED_AFTER_OPTION = typer.Option(
    None,
    "--created-after",
    help="Only traverse graph rows created at or after this ISO-8601 timestamp.",
)
_CREATED_BEFORE_OPTION = typer.Option(
    None,
    "--created-before",
    help="Only traverse graph rows created at or before this ISO-8601 timestamp.",
)
_TABLE_VERSION_OPTION = typer.Option(
    None,
    "--table-version",
    help="Constrain table-version artifacts as table=version; repeat for multiple tables.",
)
_PAGE_SIZE_OPTION = typer.Option(
    None,
    "--page-size",
    help="Return a bounded page of this many artifacts with a continuation handle (JSON).",
)
_PAGE_TOKEN_OPTION = typer.Option(
    None,
    "--page-token",
    help="Continuation handle from a prior page's next_page_token.",
)
_MAX_ARTIFACTS_OPTION = typer.Option(
    None,
    "--max-artifacts",
    help="Maximum artifact rows to emit in the report payload.",
)
_MAX_EDGES_OPTION = typer.Option(
    None,
    "--max-edges",
    help="Maximum edge rows to emit in the report payload.",
)
_MAX_EXECUTIONS_OPTION = typer.Option(
    None,
    "--max-executions",
    help="Maximum execution rows to emit in the report payload.",
)
_OUTPUT_OPTION = typer.Option(
    None,
    "--output",
    "-o",
    help="Write report output to this file instead of stdout.",
)
_REF_OPTION = typer.Option(
    None,
    "--ref",
    help="External reference as key=value; repeat for multiple refs.",
)
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
_LINEAGE_HEADER_OPTION = typer.Option(
    None,
    "--lineage-header",
    help="External target HTTP header as name=value; repeatable and not persisted.",
)
_VIEWER_FORMAT_OPTION = typer.Option(
    None,
    "--viewer-format",
    help="Viewer to reference in replay-bundle external links; repeat (default foxglove, rerun).",
)
_EXPORT_FORMAT_OPTION = typer.Option(
    "json",
    "--format",
    help="Output format: json (paged report) or ndjson (one record per line).",
)
_ARTIFACT_KIND_OPTION = typer.Option(
    None,
    "--artifact-kind",
    help="Only export artifacts of this kind, e.g. dataset-snapshot, model, run.",
)
_EXECUTION_KIND_OPTION = typer.Option(
    None,
    "--execution-kind",
    help="Only export executions of this kind, e.g. ingest, training-run.",
)
_BACKEND_OPTION = typer.Option(
    "openlineage",
    "--backend",
    help="URN backend flavor: openlineage or datahub.",
)
_SUMMARY_OPTION = typer.Option(
    False,
    "--summary",
    help="Append a trailing NDJSON summary record with the total record count.",
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _parse_output_format(value: str, *, allowed: set[str] | None = None) -> str:
    normalized = value.strip().lower()
    allowed_formats = allowed or {"tree", "json"}
    if normalized not in allowed_formats:
        choices = ", ".join(sorted(allowed_formats))
        typer.echo(f"error: --format must be one of: {choices}", err=True)
        raise typer.Exit(code=2)
    return normalized


def _parse_table_versions(values: list[str] | None) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            typer.echo("error: --table-version must look like table=version", err=True)
            raise typer.Exit(code=2)
        table, version = value.split("=", 1)
        try:
            parsed[table.strip()] = int(version)
        except ValueError as exc:
            typer.echo("error: --table-version version must be an integer", err=True)
            raise typer.Exit(code=2) from exc
    return parsed


def _parse_ref_pairs(
    values: list[str] | None,
    *,
    mlflow_run_id: str | None = None,
    wandb_run_id: str | None = None,
    wandb_artifact_id: str | None = None,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            typer.echo("error: --ref must look like key=value", err=True)
            raise typer.Exit(code=2)
        key, ref_value = value.split("=", 1)
        key = key.strip()
        if not key:
            typer.echo("error: --ref key cannot be empty", err=True)
            raise typer.Exit(code=2)
        refs[key] = ref_value.strip()
    if mlflow_run_id:
        refs["mlflow_run_id"] = mlflow_run_id
    if wandb_run_id:
        refs["wandb_run_id"] = wandb_run_id
    if wandb_artifact_id:
        refs["wandb_artifact_id"] = wandb_artifact_id
    if not refs:
        typer.echo("error: provide at least one --ref, --mlflow-run-id, or --wandb-* id", err=True)
        raise typer.Exit(code=2)
    return refs


def _parse_header_pairs(values: list[str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            typer.echo("error: --lineage-header must look like name=value", err=True)
            raise typer.Exit(code=2)
        key, header_value = value.split("=", 1)
        key = key.strip()
        if not key:
            typer.echo("error: --lineage-header name cannot be empty", err=True)
            raise typer.Exit(code=2)
        headers[key] = header_value.strip()
    return headers


def _exit_for_delivery_status(status: str) -> None:
    if status in {"failed", "partial"}:
        raise typer.Exit(code=1)


def _write_output(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text if text.endswith("\n") else f"{text}\n")
        return
    typer.echo(text)


def _emit_ndjson(records, output: str | None) -> None:
    """Stream one JSON object per line, bounded -- never buffers the full export."""
    if output:
        with Path(output).open("w") as handle:
            for record in records:
                handle.write(json.dumps(_json_ready(record), sort_keys=True) + "\n")
        return
    for record in records:
        typer.echo(json.dumps(_json_ready(record), sort_keys=True))


def _emit_graph(
    graph,
    *,
    output_format: str,
    evidence: bool,
    max_artifacts: int | None = None,
    max_edges: int | None = None,
    max_executions: int | None = None,
    output: str | None = None,
) -> None:
    parsed_format = _parse_output_format(
        output_format,
        allowed={"tree", "json", "ndjson", "summary"},
    )
    if parsed_format == "json":
        payload = graph.as_dict(
            include_evidence=evidence,
            max_artifacts=max_artifacts,
            max_edges=max_edges,
            max_executions=max_executions,
        )
        _write_output(json.dumps(_json_ready(payload), indent=2, sort_keys=True), output)
        return
    if parsed_format == "ndjson":
        records = graph.iter_ndjson_records(
            include_evidence=evidence,
            max_artifacts=max_artifacts,
            max_edges=max_edges,
            max_executions=max_executions,
        )
        _write_output(
            "\n".join(json.dumps(_json_ready(record), sort_keys=True) for record in records),
            output,
        )
        return
    if parsed_format == "summary":
        payload = graph.as_dict(
            include_evidence=evidence,
            max_artifacts=max_artifacts,
            max_edges=max_edges,
            max_executions=max_executions,
        )
        _write_output(_format_graph_summary(payload), output)
        return
    _write_output(
        _format_graph_tree(
            graph,
            evidence=evidence,
            max_artifacts=max_artifacts,
            max_edges=max_edges,
            max_executions=max_executions,
        ),
        output,
    )


def _emit_rebuild_plan(plan, *, output_format: str) -> None:
    if _parse_output_format(output_format) == "json":
        typer.echo(json.dumps(_json_ready(plan.as_dict()), indent=2, sort_keys=True))
        return
    total = plan.total_actions if plan.total_actions is not None else len(plan.actions)
    header = f"rebuild plan: {total} action(s)"
    if plan.affected_artifact_count is not None:
        header += f", {plan.affected_artifact_count} affected artifact(s)"
    if plan.policy_name:
        header += f" [policy: {plan.policy_name}]"
    lines = [header, f"roots: {', '.join(plan.root_artifact_ids)}"]
    if plan.actions_by_type:
        by_type = ", ".join(f"{k}={v}" for k, v in sorted(plan.actions_by_type.items()))
        lines.append(f"actions by type: {by_type}")
    if plan.invalidation:
        lines.append(f"invalidation: {plan.invalidation.invalidation_id}")
    if plan.summary_only:
        lines.append("(summary only; re-run with --page-size to list actions)")
    for action in plan.actions:
        label = action.name or action.artifact_id
        lines.append(f"{action.step}. {action.action} {action.kind} {label}")
    if plan.next_page_token:
        lines.append(f"next page token: {plan.next_page_token}")
    elif plan.truncated and not plan.summary_only:
        lines.append("(more actions available)")
    typer.echo("\n".join(lines))


def _load_action_policy(policy_file: str | None):
    """Build a MappingActionPolicy from a JSON file, or None for the default.

    The JSON may set any of ``kind_actions``, ``table_actions``, ``edge_actions``,
    ``severity_actions`` (each a ``{key: action}`` map) and a scalar ``fallback``.
    """

    if not policy_file:
        return None
    from lancedb_robotics.lineage import MappingActionPolicy

    with open(policy_file, encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise typer.BadParameter("action policy file must be a JSON object")
    return MappingActionPolicy(
        kind_actions=spec.get("kind_actions", {}) or {},
        table_actions=spec.get("table_actions", {}) or {},
        edge_actions=spec.get("edge_actions", {}) or {},
        severity_actions=spec.get("severity_actions", {}) or {},
        fallback=spec.get("fallback"),
        name=str(spec.get("name") or "custom"),
    )


def _emit_audit(report, *, output_format: str) -> None:
    if _parse_output_format(output_format) == "json":
        typer.echo(json.dumps(_json_ready(report.as_dict()), indent=2, sort_keys=True))
        return
    payload = report.as_dict()
    lines = [
        f"lineage audit: {payload['artifact_count']} artifact(s), {payload['edge_count']} edge(s)",
        f"unresolved references: {len(payload['unresolved_references'])}",
        f"missing sources: {len(payload['missing_sources'])}",
        f"missing table versions: {len(payload['missing_table_versions'])}",
        f"stale external links: {len(payload['stale_external_links'])}",
        f"retained versions: {len(payload['retained_versions'])}",
        f"cleanup candidates: {len(payload['cleanup_candidates'])}",
    ]
    typer.echo("\n".join(lines))


def _format_graph_tree(
    graph,
    *,
    evidence: bool,
    max_artifacts: int | None = None,
    max_edges: int | None = None,
    max_executions: int | None = None,
) -> str:
    payload = graph.as_dict(
        include_evidence=evidence,
        max_artifacts=max_artifacts,
        max_edges=max_edges,
        max_executions=max_executions,
    )
    artifacts = {row["artifact_id"]: row for row in payload["artifacts"]}
    edges = sorted(
        payload["edges"],
        key=lambda row: (row["edge_type"], row["from_artifact_id"], row["to_artifact_id"], row["edge_id"]),
    )
    direction = payload["direction"]
    # Index edges by their parent-side endpoint so each tree node visits only its
    # own incident edges. Scanning the full edge list per node is O(V*E) and hangs
    # on large graphs (a full-corpus trace can carry hundreds of thousands of
    # edges); `edges` is pre-sorted so each bucket keeps deterministic order.
    parent_key = "to_artifact_id" if direction == "upstream" else "from_artifact_id"
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        adjacency[edge[parent_key]].append(edge)
    roots = [root for root in payload["root_artifact_ids"] if root in artifacts]
    lines: list[str] = []
    if len(payload["root_artifact_ids"]) > 1:
        lines.append(
            f"{direction} {payload['resolved_handle']} ({len(payload['root_artifact_ids'])} roots)"
        )
    for root_id in roots:
        _append_tree_node(root_id, artifacts, adjacency, direction, lines, evidence=evidence)
    for warning in payload["warnings"]:
        lines.append(f"warning: {warning['message']}")
    return "\n".join(lines)


def _format_graph_summary(payload: dict[str, Any]) -> str:
    page = payload.get("page") or {}
    total_artifacts = page.get("total_artifacts", len(payload["artifacts"]))
    total_edges = page.get("total_edges", len(payload["edges"]))
    lines = [
        f"{payload['direction']} lineage report",
        f"handle: {payload['resolved_handle']}",
        f"roots: {len(payload['root_artifact_ids'])}",
        f"artifacts: {len(payload['artifacts'])}/{total_artifacts}",
        f"edges: {len(payload['edges'])}/{total_edges}",
        f"executions: {len(payload['executions'])}",
    ]
    if page.get("next_page_token"):
        lines.append(f"next_page_token: {page['next_page_token']}")
    if payload["warnings"]:
        lines.append("warnings:")
        lines.extend(f"- {warning['message']}" for warning in payload["warnings"])
    return "\n".join(lines)


def _append_tree_node(
    artifact_id: str,
    artifacts: dict[str, dict[str, Any]],
    adjacency: dict[str, list[dict[str, Any]]],
    direction: str,
    lines: list[str],
    *,
    evidence: bool,
    depth: int = 0,
    seen: set[str] | None = None,
) -> None:
    seen = set(seen or ())
    indent = "  " * depth
    prefix = "- " if depth == 0 else "  - "
    lines.append(f"{indent}{prefix}{_artifact_label(artifacts[artifact_id], evidence=evidence)}")
    if artifact_id in seen:
        return
    seen.add(artifact_id)
    for edge in adjacency.get(artifact_id, ()):
        child_id = edge["from_artifact_id"] if direction == "upstream" else edge["to_artifact_id"]
        if child_id not in artifacts:
            continue
        connector = "<-" if direction == "upstream" else "->"
        child_indent = "  " * (depth + 1)
        lines.append(f"{child_indent}{edge['edge_type']} {connector}")
        if child_id in seen:
            lines.append(
                f"{child_indent}  - {_artifact_label(artifacts[child_id], evidence=evidence)} (cycle)"
            )
            continue
        _append_tree_node(
            child_id,
            artifacts,
            adjacency,
            direction,
            lines,
            evidence=evidence,
            depth=depth + 1,
            seen=seen,
        )


def _artifact_label(row: dict[str, Any], *, evidence: bool) -> str:
    name = row.get("name") or row.get("artifact_id")
    label = f"{row.get('kind')} {name} [{row.get('artifact_id')}]"
    details: list[str] = []
    if row.get("table_name"):
        version = f"@{row['table_version']}" if row.get("table_version") is not None else ""
        details.append(f"{row['table_name']}{version}")
    if evidence and row.get("source_uri"):
        details.append(str(row["source_uri"]))
    if evidence and row.get("digest"):
        details.append(str(row["digest"]))
    if evidence:
        metadata = {item["key"]: item["value"] for item in row.get("metadata") or []}
        for key in ("channel", "offset", "log_time_ns"):
            if metadata.get(key) not in {None, ""}:
                details.append(f"{key}={metadata[key]}")
    return f"{label} ({', '.join(details)})" if details else label


@lineage_app.command("trace")
def trace(
    artifact: str = typer.Argument(..., help="Artifact id or domain handle to trace upstream."),
    lake: str = _LAKE_OPTION,
    where: str | None = _WHERE_OPTION,
    limit: int | None = _LIMIT_OPTION,
    kind: str | None = _KIND_OPTION,
    max_depth: int | None = _MAX_DEPTH_OPTION,
    edge_type: list[str] | None = _EDGE_TYPE_OPTION,
    target_kind: list[str] | None = _TARGET_KIND_OPTION,
    created_after: str | None = _CREATED_AFTER_OPTION,
    created_before: str | None = _CREATED_BEFORE_OPTION,
    table_version: list[str] | None = _TABLE_VERSION_OPTION,
    page_size: int | None = _PAGE_SIZE_OPTION,
    page_token: str | None = _PAGE_TOKEN_OPTION,
    output_format: str = _FORMAT_OPTION,
    evidence: bool = typer.Option(False, "--evidence", help="Include source/table evidence in report output."),
    max_artifacts: int | None = _MAX_ARTIFACTS_OPTION,
    max_edges: int | None = _MAX_EDGES_OPTION,
    max_executions: int | None = _MAX_EXECUTIONS_OPTION,
    output: str | None = _OUTPUT_OPTION,
    checkpoint: bool = typer.Option(
        False,
        "--checkpoint",
        help="Treat the argument as a model/checkpoint run and emit the legacy regression report.",
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Trace upstream provenance for an artifact handle."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    if checkpoint or where is not None or limit is not None:
        try:
            report = opened.lineage.trace_checkpoint(artifact, where=where, limit=limit)
        except LineageError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(_json_ready(report.as_dict()), indent=2, sort_keys=True))
        return

    try:
        graph = opened.lineage.trace(
            artifact,
            kind=kind,
            max_depth=max_depth,
            edge_types=edge_type or (),
            target_kinds=target_kind or (),
            created_after=created_after,
            created_before=created_before,
            table_versions=_parse_table_versions(table_version),
            page_size=page_size,
            page_token=page_token,
        )
    except LineageError as exc:
        try:
            report = opened.lineage.trace_checkpoint(artifact)
        except LineageError:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(json.dumps(_json_ready(report.as_dict()), indent=2, sort_keys=True))
        return
    _emit_graph(
        graph,
        output_format=output_format,
        evidence=evidence,
        max_artifacts=max_artifacts,
        max_edges=max_edges,
        max_executions=max_executions,
        output=output,
    )


def _emit_resolution(resolution, *, output_format: str) -> None:
    if _parse_output_format(output_format) == "json":
        typer.echo(json.dumps(_json_ready(resolution.as_dict()), indent=2, sort_keys=True))
        return
    payload = resolution.as_dict()
    lines = [
        f"{payload['status']}: {payload['handle']} "
        f"({payload['root_count']} root(s)"
        f"{', multi-root' if payload['multi_root'] else ''})",
        f"graph fresh: {payload['graph_fresh']}"
        + (f" (stale: {', '.join(payload['stale_tables'])})" if payload["stale_tables"] else ""),
    ]
    for candidate in payload["candidates"]:
        marker = "" if candidate["in_graph"] else " [pending refresh]"
        lines.append(
            f"  - {candidate['kind']} {candidate.get('name') or candidate['artifact_id']} "
            f"[{candidate['artifact_id']}]{marker} (matched: {', '.join(candidate['matched_on'])})"
        )
    for hint in payload["disambiguation_hints"]:
        lines.append(f"  hint: {hint['flag']} {hint['values']}")
    for command in payload["suggested_commands"]:
        lines.append(f"  try: {command}")
    if payload["message"]:
        lines.append(payload["message"])
    typer.echo("\n".join(lines))


@lineage_app.command("resolve")
def resolve(
    handle: str = typer.Argument(..., help="Artifact id or domain handle to resolve."),
    lake: str = _LAKE_OPTION,
    kind: str | None = _KIND_OPTION,
    table_version: int | None = typer.Option(
        None,
        "--table-version",
        help="Constrain table-version artifacts to this version.",
    ),
    output_format: str = _PLAN_FORMAT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Resolve a handle to lineage roots with ambiguity/staleness diagnostics.

    Read-only: reports exact/ambiguous/stale/unknown status, candidate evidence,
    graph freshness, and ready-to-run disambiguation or refresh commands without
    mutating lake state (backlog 0102).
    """
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        resolution = opened.lineage.resolve(handle, kind=kind, table_version=table_version)
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_resolution(resolution, output_format=output_format)


@lineage_app.command("checkpoint")
def checkpoint_trace(
    model_run_id: str = typer.Argument(..., help="Model/checkpoint run id to trace."),
    lake: str = _LAKE_OPTION,
    where: str | None = _WHERE_OPTION,
    limit: int | None = _LIMIT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Trace a model checkpoint/run to its snapshot, slice rows, and source logs."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        report = opened.lineage.trace_checkpoint(model_run_id, where=where, limit=limit)
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(report.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("export-evidence")
def export_evidence(
    artifact: str = typer.Argument(..., help="Checkpoint/model-output/feedback/artifact handle."),
    lake: str = _LAKE_OPTION,
    kind: str | None = _KIND_OPTION,
    checkpoint: bool = typer.Option(
        False,
        "--checkpoint",
        help="Treat the handle as a model/checkpoint run id.",
    ),
    where: str | None = _WHERE_OPTION,
    limit: int | None = _LIMIT_OPTION,
    max_depth: int | None = _MAX_DEPTH_OPTION,
    edge_type: list[str] | None = _EDGE_TYPE_OPTION,
    target_kind: list[str] | None = _TARGET_KIND_OPTION,
    refresh_graph: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Refresh canonical graph tables before graph-based evidence export.",
    ),
    materialize: bool = typer.Option(
        False,
        "--materialize",
        help="Write a materialized evidence directory instead of a plan-only manifest.",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for a materialized evidence pack.",
    ),
    include_payloads: bool = typer.Option(
        False,
        "--include-payloads",
        help="Materialize selected observation payload blobs.",
    ),
    include_attachments: bool = typer.Option(
        False,
        "--include-attachments",
        help="Materialize selected run attachment blobs.",
    ),
    include_video: bool = typer.Option(
        False,
        "--include-video",
        help="Materialize selected codec-aware video encoding blobs.",
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Export a deterministic source evidence pack manifest."""
    from lancedb_robotics.evidence import EvidencePackError
    from lancedb_robotics.lineage import LineageError

    if materialize and output_dir is None:
        typer.echo("error: --output-dir is required with --materialize", err=True)
        raise typer.Exit(code=2)

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        report = opened.lineage.evidence_pack(
            artifact,
            kind=kind,
            checkpoint=checkpoint,
            where=where,
            limit=limit,
            max_depth=max_depth,
            edge_types=edge_type or (),
            target_kinds=target_kind or (),
            refresh=refresh_graph,
            output_dir=output_dir,
            materialize=materialize,
            include_payloads=include_payloads,
            include_attachments=include_attachments,
            include_video=include_video,
        )
    except (EvidencePackError, LineageError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(report.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("export-replay")
def export_replay(
    artifact: str = typer.Argument(..., help="Checkpoint/model-output/feedback/artifact handle."),
    lake: str = _LAKE_OPTION,
    output_dir: str = typer.Option(..., "--output-dir", help="Directory for the replay bundle."),
    kind: str | None = _KIND_OPTION,
    checkpoint: bool = typer.Option(
        False,
        "--checkpoint",
        help="Treat the handle as a model/checkpoint run id.",
    ),
    where: str | None = _WHERE_OPTION,
    limit: int | None = _LIMIT_OPTION,
    max_depth: int | None = _MAX_DEPTH_OPTION,
    edge_type: list[str] | None = _EDGE_TYPE_OPTION,
    target_kind: list[str] | None = _TARGET_KIND_OPTION,
    refresh_graph: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Refresh canonical graph tables before graph-based evidence export.",
    ),
    include_mcap: bool = typer.Option(
        True,
        "--mcap/--no-mcap",
        help="Reconstruct one deterministic MCAP slice per source URI.",
    ),
    include_video: bool = typer.Option(
        False,
        "--video",
        help="Extract codec-aware video clip/GOP bytes from video encoding refs.",
    ),
    include_gops: bool = typer.Option(
        True,
        "--gops/--no-gops",
        help="Emit per-GOP byte slices alongside each video clip.",
    ),
    viewer_format: list[str] | None = _VIEWER_FORMAT_OPTION,
    max_bytes: int | None = typer.Option(
        None,
        "--max-bytes",
        help="Fail before writing if the bundle would exceed this many bytes.",
    ),
    max_files: int | None = typer.Option(
        None,
        "--max-files",
        help="Fail before writing if the bundle would emit more than this many files.",
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Export a deterministic, viewer-openable replay bundle from an evidence pack."""
    from lancedb_robotics.evidence import EvidencePackError
    from lancedb_robotics.lineage import LineageError
    from lancedb_robotics.storage import parse_storage_option_pairs

    if not include_mcap and not include_video:
        typer.echo("error: enable at least one of --mcap / --video", err=True)
        raise typer.Exit(code=2)

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        storage_options = parse_storage_option_pairs(storage_option)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        report = opened.lineage.replay_bundle(
            artifact,
            output_dir=output_dir,
            kind=kind,
            checkpoint=checkpoint,
            where=where,
            limit=limit,
            max_depth=max_depth,
            edge_types=edge_type or (),
            target_kinds=target_kind or (),
            refresh=refresh_graph,
            include_mcap=include_mcap,
            include_video=include_video,
            include_gops=include_gops,
            viewer_formats=viewer_format or ("foxglove", "rerun"),
            max_bytes=max_bytes,
            max_files=max_files,
            storage_options=storage_options or None,
            auth_ref=auth_ref,
        )
    except (EvidencePackError, LineageError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(report.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("audit")
def audit(
    artifact: str | None = typer.Argument(
        None,
        help="Optional artifact id or domain handle to audit; omit to audit the whole graph.",
    ),
    lake: str = _LAKE_OPTION,
    kind: str | None = _KIND_OPTION,
    refresh_graph: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Refresh canonical graph tables before auditing.",
    ),
    check_sources: bool = typer.Option(
        True,
        "--check-sources/--no-check-sources",
        help="Check local file:// or path source URIs for existence.",
    ),
    check_remote_sources: bool = typer.Option(
        False,
        "--check-remote-sources/--no-check-remote-sources",
        help="Opt in to object-store source existence checks using storage credentials.",
    ),
    record: bool = typer.Option(
        False,
        "--record/--no-record",
        help="Persist the full audit report in lineage_audit_reports.",
    ),
    created_by: str = typer.Option(
        "lancedb-robotics",
        "--created-by",
        help="Actor recorded on persisted audit reports.",
    ),
    page_size: int | None = _PAGE_SIZE_OPTION,
    page_token: str | None = _PAGE_TOKEN_OPTION,
    output_format: str = _PLAN_FORMAT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Audit lineage references, retained versions, and cleanup candidates."""
    from lancedb_robotics.lineage import LineageError
    from lancedb_robotics.storage import parse_storage_option_pairs

    if record and (page_size is not None or page_token is not None):
        typer.echo("error: --record cannot be combined with --page-size or --page-token", err=True)
        raise typer.Exit(code=2)
    try:
        storage_options = parse_storage_option_pairs(storage_option)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        report = opened.lineage.audit(
            artifact,
            kind=kind,
            refresh=refresh_graph,
            check_sources=check_sources,
            check_remote_sources=check_remote_sources,
            source_auth_ref=auth_ref,
            source_storage_options=storage_options,
            page_size=page_size,
            page_token=page_token,
        )
        entry = (
            opened.lineage.record_audit_report(report, created_by=created_by)
            if record
            else None
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if entry is None:
        _emit_audit(report, output_format=output_format)
        return
    if _parse_output_format(output_format) == "json":
        payload = report.as_dict()
        payload["catalog_entry"] = entry.as_dict()
        typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        return
    _emit_audit(report, output_format=output_format)
    typer.echo(f"recorded audit report: {entry.report_id}")


@lineage_app.command("audit-reports")
def audit_reports(
    lake: str = _LAKE_OPTION,
    status: str | None = typer.Option(
        None,
        "--status",
        help="Filter by audit status: passed, failed, or partial.",
    ),
    subject: str | None = typer.Option(
        None,
        "--subject",
        help="Filter by audited subject handle/artifact id.",
    ),
    created_by: str | None = typer.Option(
        None,
        "--created-by",
        help="Filter by actor recorded on the audit report.",
    ),
    page_size: int = typer.Option(50, "--page-size", help="Catalog page size."),
    cursor: str | None = typer.Option(None, "--cursor", help="Continuation cursor."),
    output_format: str = _PLAN_FORMAT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """List persisted lineage audit reports newest-first."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        page = opened.lineage.audit_reports(
            status=status,
            subject=subject,
            created_by=created_by,
            page_size=page_size,
            cursor=cursor,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if _parse_output_format(output_format) == "json":
        typer.echo(json.dumps(_json_ready(page.to_dict()), indent=2, sort_keys=True))
        return
    lines = [f"lineage audit reports: {len(page.reports)}"]
    for entry in page.reports:
        subject_text = entry.subject or "(whole lake)"
        lines.append(
            f"{entry.report_id} {entry.status} subject={subject_text} "
            f"findings={entry.finding_count} created_at={entry.created_at.isoformat()}"
        )
    if page.next_cursor:
        lines.append(f"next cursor: {page.next_cursor}")
    typer.echo("\n".join(lines))


@lineage_app.command("get-audit-report")
def get_audit_report(
    report_id: str = typer.Argument(..., help="Audit report id or digest."),
    lake: str = _LAKE_OPTION,
    entry_only: bool = typer.Option(
        False,
        "--entry-only",
        help="Return only catalog metadata, omitting the stored report payload.",
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Reload a persisted lineage audit report by id or digest."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        entry, report = opened.lineage.get_audit_report(report_id)
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = {"entry": entry.as_dict()} if entry_only else {"entry": entry.as_dict(), "report": report}
    typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


@lineage_app.command("export-audit-findings")
def export_audit_findings(
    report_id: str = typer.Argument(..., help="Audit report id or digest."),
    lake: str = _LAKE_OPTION,
    finding_type: str | None = typer.Option(
        None,
        "--finding-type",
        help="Filter findings, e.g. missing_sources, cleanup_candidates, unresolved.",
    ),
    output_format: str = _EXPORT_FORMAT_OPTION,
    page_size: int | None = _PAGE_SIZE_OPTION,
    page_token: str | None = _PAGE_TOKEN_OPTION,
    summary: bool = _SUMMARY_OPTION,
    output: str | None = _OUTPUT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Export findings from a persisted lineage audit report as JSON or NDJSON."""
    from lancedb_robotics.lineage import LineageError

    parsed_format = _parse_output_format(output_format, allowed={"json", "ndjson"})
    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        if parsed_format == "ndjson":
            lines = opened.lineage.iter_audit_findings_ndjson(
                report_id,
                finding_type=finding_type,
                page_size=page_size or 512,
                include_summary=summary,
            )
            if output:
                with Path(output).open("w") as handle:
                    for line in lines:
                        handle.write(line + "\n")
            else:
                for line in lines:
                    typer.echo(line)
            return
        page = opened.lineage.audit_findings(
            report_id,
            finding_type=finding_type,
            page_size=page_size or 100,
            cursor=page_token,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _write_output(json.dumps(_json_ready(page.to_dict()), indent=2, sort_keys=True), output)


@lineage_app.command("refresh")
def refresh(
    lake: str = _LAKE_OPTION,
    dry_run: bool = typer.Option(
        False,
        "--plan/--apply",
        "--dry-run/--no-dry-run",
        help="Print the refresh plan (source versions, changed tables, counts) without writing.",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--no-incremental",
        help="Skip the projection when no source table changed since the last refresh.",
    ),
    force_full: bool = typer.Option(
        False,
        "--force-full",
        help="Force a full re-projection even when the watermark shows no change.",
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Project canonical lake rows into lineage graph tables (incremental by default)."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        report = opened.lineage.refresh_graph(
            incremental=incremental,
            force_full=force_full,
            dry_run=dry_run,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(report.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("index-plan")
def index_plan(
    lake: str = _LAKE_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Report required/present/missing lineage predicate indexes as an actionable plan."""
    opened = _open_lake(lake, auth_ref, storage_option)
    plan = opened.lineage.index_plan()
    typer.echo(json.dumps(_json_ready(plan), indent=2, sort_keys=True))


_CYPHER_PARAM_OPTION = typer.Option(
    None,
    "--param",
    help="Bind a Cypher parameter as name=value ($name in the query); repeat "
    "for multiple parameters.",
)
_CYPHER_STRATEGY_OPTION = typer.Option(
    None,
    "--strategy",
    help="Execution strategy: datafusion (default) or native.",
)


@lineage_app.command("cypher")
def cypher(
    query: str = typer.Argument(
        ...,
        help="Cypher query over the lineage property graph (nodes: Artifact, "
        "Execution; relationship: DEPENDS_ON from upstream -> downstream).",
    ),
    lake: str = _LAKE_OPTION,
    param: list[str] | None = _CYPHER_PARAM_OPTION,
    strategy: str | None = _CYPHER_STRATEGY_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Run a Cypher query over the lineage graph (optional 'graph' extra, backlog 0099).

    Expert-only surface: canonical lineage stays in the Lance tables and
    trace/impact/audit remain the stable API. Requires
    ``pip install 'lancedb-robotics[graph]'``; without it this command exits with
    an actionable missing-extra message.
    """
    from lancedb_robotics.lineage import LineageError

    parameters: dict[str, str] = {}
    for value in param or []:
        if "=" not in value:
            typer.echo("error: --param must look like name=value", err=True)
            raise typer.Exit(code=2)
        key, bound = value.split("=", 1)
        key = key.strip()
        if not key:
            typer.echo("error: --param name cannot be empty", err=True)
            raise typer.Exit(code=2)
        parameters[key] = bound

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        rows = opened.lineage.cypher(query, parameters=parameters, strategy=strategy)
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(rows), indent=2, sort_keys=True))


@lineage_app.command("export-openlineage")
def export_openlineage(
    lake: str = _LAKE_OPTION,
    refresh_graph: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Refresh canonical graph tables before building the export payload.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--emit-ready",
        help="Mark the payload as dry-run validation or ready for an external emitter.",
    ),
    producer: str | None = typer.Option(
        None,
        "--producer",
        help="OpenLineage producer URI to place on events and facets.",
    ),
    emit: bool = typer.Option(
        False,
        "--emit",
        help="POST events to the configured OpenLineage target and record delivery state.",
    ),
    retry: bool = typer.Option(
        False,
        "--retry",
        help="Retry events not already delivered for this target.",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        help="Stable external target name used for idempotency; defaults to a runtime endpoint digest.",
    ),
    endpoint_url: str | None = typer.Option(
        None,
        "--endpoint-url",
        help="Runtime OpenLineage/Marquez HTTP endpoint; not persisted in lake rows.",
    ),
    lineage_auth_ref: str | None = typer.Option(
        None,
        "--lineage-auth-ref",
        help="Runtime credential reference for the external lineage target.",
    ),
    lineage_header: list[str] | None = _LINEAGE_HEADER_OPTION,
    output_format: str = _EXPORT_FORMAT_OPTION,
    page_size: int | None = _PAGE_SIZE_OPTION,
    page_token: str | None = _PAGE_TOKEN_OPTION,
    execution_kind: str | None = _EXECUTION_KIND_OPTION,
    created_after: str | None = _CREATED_AFTER_OPTION,
    created_before: str | None = _CREATED_BEFORE_OPTION,
    summary: bool = _SUMMARY_OPTION,
    output: str | None = _OUTPUT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Emit OpenLineage RunEvent JSON for canonical lineage executions.

    Add ``--page-size`` for a bounded page + continuation token, ``--format
    ndjson`` to stream one event per line, or ``--execution-kind`` /
    ``--created-after`` / ``--created-before`` to narrow the export (backlog 0105).
    """
    from lancedb_robotics.lineage_integrations import LineageIntegrationError

    if emit and retry:
        typer.echo("error: choose either --emit or --retry", err=True)
        raise typer.Exit(code=2)

    parsed_format = _parse_output_format(output_format, allowed={"json", "ndjson"})
    paged = page_size is not None or page_token is not None
    filtered = bool(execution_kind or created_after or created_before)
    if (emit or retry) and (parsed_format == "ndjson" or paged or summary or filtered):
        typer.echo(
            "error: --emit/--retry cannot be combined with --format ndjson, "
            "--page-size, --page-token, --summary, or export filters",
            err=True,
        )
        raise typer.Exit(code=2)

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        if emit or retry:
            report = opened.lineage.emit_openlineage(
                refresh=refresh_graph,
                producer=producer,
                target=target,
                endpoint_url=endpoint_url,
                auth_ref=lineage_auth_ref,
                headers=_parse_header_pairs(lineage_header),
                retry=retry,
            )
            typer.echo(json.dumps(_json_ready(report.to_dict()), indent=2, sort_keys=True))
            _exit_for_delivery_status(report.status)
            return
        if parsed_format == "ndjson":
            _emit_ndjson(
                opened.lineage.iter_openlineage_ndjson(
                    page_size=page_size or 512,
                    execution_kind=execution_kind,
                    created_after=created_after,
                    created_before=created_before,
                    refresh=refresh_graph,
                    dry_run=dry_run,
                    producer=producer,
                    include_summary=summary,
                ),
                output,
            )
            return
        if paged or filtered:
            page = opened.lineage.export_openlineage_page(
                page_size=page_size,
                page_token=page_token,
                execution_kind=execution_kind,
                created_after=created_after,
                created_before=created_before,
                refresh=refresh_graph,
                dry_run=dry_run,
                producer=producer,
            )
            _write_output(
                json.dumps(_json_ready(page.to_dict()), indent=2, sort_keys=True), output
            )
            return
        report = opened.lineage.export_openlineage(
            refresh=refresh_graph,
            dry_run=dry_run,
            producer=producer,
        )
    except LineageIntegrationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _write_output(json.dumps(_json_ready(report.to_dict()), indent=2, sort_keys=True), output)


@lineage_app.command("export-datahub")
def export_datahub(
    lake: str = _LAKE_OPTION,
    refresh_graph: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Refresh canonical graph tables before building the export payload.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--emit-ready",
        help="Mark the payload as dry-run validation or ready for an external emitter.",
    ),
    emit: bool = typer.Option(
        False,
        "--emit",
        help="POST edges to the configured DataHub target and record delivery state.",
    ),
    retry: bool = typer.Option(
        False,
        "--retry",
        help="Retry edges not already delivered for this target.",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        help="Stable external target name used for idempotency; defaults to a runtime endpoint digest.",
    ),
    endpoint_url: str | None = typer.Option(
        None,
        "--endpoint-url",
        help="Runtime DataHub HTTP endpoint; not persisted in lake rows.",
    ),
    lineage_auth_ref: str | None = typer.Option(
        None,
        "--lineage-auth-ref",
        help="Runtime credential reference for the external lineage target.",
    ),
    lineage_header: list[str] | None = _LINEAGE_HEADER_OPTION,
    output_format: str = _EXPORT_FORMAT_OPTION,
    page_size: int | None = _PAGE_SIZE_OPTION,
    page_token: str | None = _PAGE_TOKEN_OPTION,
    artifact_kind: str | None = _ARTIFACT_KIND_OPTION,
    created_after: str | None = _CREATED_AFTER_OPTION,
    created_before: str | None = _CREATED_BEFORE_OPTION,
    table_version: list[str] | None = _TABLE_VERSION_OPTION,
    summary: bool = _SUMMARY_OPTION,
    output: str | None = _OUTPUT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Emit DataHub-style upstream/downstream lineage JSON from the graph.

    Add ``--page-size`` for a bounded page + continuation token, ``--format
    ndjson`` to stream one edge per line, or ``--artifact-kind`` /
    ``--table-version`` / ``--created-*`` to narrow the export (backlog 0105).
    Endpoint filters keep an edge only when both endpoints pass.
    """
    from lancedb_robotics.lineage_integrations import LineageIntegrationError

    if emit and retry:
        typer.echo("error: choose either --emit or --retry", err=True)
        raise typer.Exit(code=2)

    parsed_format = _parse_output_format(output_format, allowed={"json", "ndjson"})
    paged = page_size is not None or page_token is not None
    filtered = bool(artifact_kind or created_after or created_before or table_version)
    if (emit or retry) and (parsed_format == "ndjson" or paged or summary or filtered):
        typer.echo(
            "error: --emit/--retry cannot be combined with --format ndjson, "
            "--page-size, --page-token, --summary, or export filters",
            err=True,
        )
        raise typer.Exit(code=2)

    table_versions = _parse_table_versions(table_version)
    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        if emit or retry:
            report = opened.lineage.emit_datahub(
                refresh=refresh_graph,
                target=target,
                endpoint_url=endpoint_url,
                auth_ref=lineage_auth_ref,
                headers=_parse_header_pairs(lineage_header),
                retry=retry,
            )
            typer.echo(json.dumps(_json_ready(report.to_dict()), indent=2, sort_keys=True))
            _exit_for_delivery_status(report.status)
            return
        if parsed_format == "ndjson":
            _emit_ndjson(
                opened.lineage.iter_datahub_ndjson(
                    page_size=page_size or 512,
                    artifact_kind=artifact_kind,
                    created_after=created_after,
                    created_before=created_before,
                    table_versions=table_versions,
                    refresh=refresh_graph,
                    dry_run=dry_run,
                    include_summary=summary,
                ),
                output,
            )
            return
        if paged or filtered:
            page = opened.lineage.export_datahub_page(
                page_size=page_size,
                page_token=page_token,
                artifact_kind=artifact_kind,
                created_after=created_after,
                created_before=created_before,
                table_versions=table_versions,
                refresh=refresh_graph,
                dry_run=dry_run,
            )
            _write_output(
                json.dumps(_json_ready(page.to_dict()), indent=2, sort_keys=True), output
            )
            return
        report = opened.lineage.export_datahub(refresh=refresh_graph, dry_run=dry_run)
    except LineageIntegrationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _write_output(json.dumps(_json_ready(report.to_dict()), indent=2, sort_keys=True), output)


@lineage_app.command("export-urns")
def export_urns(
    lake: str = _LAKE_OPTION,
    backend: str = _BACKEND_OPTION,
    refresh_graph: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Refresh canonical graph tables before building the catalog.",
    ),
    output_format: str = _EXPORT_FORMAT_OPTION,
    page_size: int | None = _PAGE_SIZE_OPTION,
    page_token: str | None = _PAGE_TOKEN_OPTION,
    artifact_kind: str | None = _ARTIFACT_KIND_OPTION,
    created_after: str | None = _CREATED_AFTER_OPTION,
    created_before: str | None = _CREATED_BEFORE_OPTION,
    table_version: list[str] | None = _TABLE_VERSION_OPTION,
    summary: bool = _SUMMARY_OPTION,
    output: str | None = _OUTPUT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Export the bulk artifact-URN catalog: id, backend URN, kind, table, digest.

    Bounded and resumable -- pass ``--page-size`` for a page + continuation
    token, or ``--format ndjson`` to stream the whole catalog one record per
    line (backlog 0105).
    """
    from lancedb_robotics.lineage_integrations import LineageIntegrationError

    parsed_format = _parse_output_format(output_format, allowed={"json", "ndjson"})
    table_versions = _parse_table_versions(table_version)
    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        if parsed_format == "ndjson":
            _emit_ndjson(
                opened.lineage.iter_artifact_urn_ndjson(
                    backend=backend,
                    page_size=page_size or 512,
                    artifact_kind=artifact_kind,
                    created_after=created_after,
                    created_before=created_before,
                    table_versions=table_versions,
                    refresh=refresh_graph,
                    include_summary=summary,
                ),
                output,
            )
            return
        page = opened.lineage.export_artifact_urns(
            backend=backend,
            page_size=page_size,
            page_token=page_token,
            artifact_kind=artifact_kind,
            created_after=created_after,
            created_before=created_before,
            table_versions=table_versions,
            refresh=refresh_graph,
        )
    except LineageIntegrationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _write_output(json.dumps(_json_ready(page.to_dict()), indent=2, sort_keys=True), output)


@lineage_app.command("resolve-urn")
def resolve_urn(
    urn: str = typer.Argument(..., help="External artifact URN emitted by an export command."),
    lake: str = _LAKE_OPTION,
    refresh_graph: bool = typer.Option(
        False,
        "--refresh",
        help="Refresh canonical graph tables before resolving the URN.",
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Resolve an exported external URN back to a canonical lineage artifact id."""
    from lancedb_robotics.lineage_integrations import LineageIntegrationError

    opened = _open_lake(lake, auth_ref, storage_option)
    if refresh_graph:
        opened.lineage.refresh_graph()
    try:
        artifact_id = opened.lineage.resolve_external_urn(urn)
    except LineageIntegrationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"artifact_id": artifact_id, "urn": urn}, indent=2, sort_keys=True))


@lineage_app.command("attach-refs")
def attach_refs(
    kind: str = typer.Argument(..., help="Manifest kind: training-run, model, or evaluation-run."),
    manifest_id: str = typer.Argument(..., help="Manifest row id to update."),
    lake: str = _LAKE_OPTION,
    ref: list[str] | None = _REF_OPTION,
    mlflow_run_id: str | None = typer.Option(None, "--mlflow-run-id", help="MLflow run id."),
    wandb_run_id: str | None = typer.Option(None, "--wandb-run-id", help="W&B run id."),
    wandb_artifact_id: str | None = typer.Option(None, "--wandb-artifact-id", help="W&B artifact id."),
    replace: bool = typer.Option(False, "--replace", help="Replace existing refs instead of merging."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Attach external metadata-system references to run manifest rows."""
    from lancedb_robotics.run_manifests import RunManifestError

    opened = _open_lake(lake, auth_ref, storage_option)
    refs = _parse_ref_pairs(
        ref,
        mlflow_run_id=mlflow_run_id,
        wandb_run_id=wandb_run_id,
        wandb_artifact_id=wandb_artifact_id,
    )
    normalized = kind.strip().lower().replace("_", "-")
    try:
        if normalized in {"training-run", "training"}:
            update = opened.training.attach_external_refs(manifest_id, refs, replace=replace)
        elif normalized in {"model", "model-artifact", "checkpoint"}:
            update = opened.training.attach_model_external_refs(manifest_id, refs, replace=replace)
        elif normalized in {"evaluation-run", "eval-run", "evaluation", "eval"}:
            update = opened.eval.attach_external_refs(manifest_id, refs, replace=replace)
        else:
            typer.echo(
                "error: kind must be training-run, model, or evaluation-run",
                err=True,
            )
            raise typer.Exit(code=2)
    except RunManifestError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(update.to_dict()), indent=2, sort_keys=True))


@lineage_app.command("check-adapter")
def check_adapter(
    adapter: str = typer.Argument(..., help="Optional adapter name, e.g. openlineage-client."),
) -> None:
    """Check whether an optional metadata-system adapter package is installed."""
    from lancedb_robotics.lineage_integrations import (
        LineageIntegrationError,
        require_integration_adapter,
    )

    try:
        payload = require_integration_adapter(adapter)
    except LineageIntegrationError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@lineage_app.command("plugins")
def plugins(
    family: str | None = typer.Option(
        None,
        "--family",
        help="Filter by adapter family: lineage-emitter, reference-importer, or manifest-sync.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the full descriptor as JSON."),
) -> None:
    """List registered metadata-integration plugins, capabilities, and availability.

    The registry never imports an optional dependency: `available` is a metadata
    probe (find_spec), and each row carries the install hint to enable it.
    """
    from lancedb_robotics.metadata_plugins import list_metadata_plugins

    descriptors = list_metadata_plugins(family=family)
    if as_json:
        typer.echo(json.dumps(list(descriptors), indent=2, sort_keys=True))
        return
    if not descriptors:
        typer.echo("no metadata plugins registered" + (f" for family {family!r}" if family else ""))
        return
    for descriptor in descriptors:
        caps = ",".join(sorted(k for k, v in descriptor["capabilities"].items() if v)) or "none"
        available = "available" if descriptor["available"] else "missing"
        typer.echo(f"{descriptor['name']:<14} {descriptor['family']:<18} [{available}] caps={caps}")
        if not descriptor["available"]:
            typer.echo(f"    install: {descriptor['dependency']['install_hint']}")


@lineage_app.command("conformance")
def conformance(
    plugin: str | None = typer.Option(
        None,
        "--plugin",
        help="Run conformance for one plugin by name; omit to run every registered plugin.",
    ),
    lake: str | None = typer.Option(
        None,
        "--lake",
        help="Optional lake path/URI; enables the emit-path checks against real tables.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the full conformance report as JSON."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Run the metadata-plugin conformance suite; exit non-zero on any failure."""
    from lancedb_robotics.metadata_plugins import (
        MetadataPluginError,
        default_registry,
        run_conformance,
        run_registry_conformance,
    )

    opened = _open_lake(lake, auth_ref, storage_option) if lake else None
    try:
        if plugin is not None:
            reports = [run_conformance(default_registry().get(plugin), lake=opened)]
        else:
            reports = list(run_registry_conformance(default_registry(), lake=opened))
    except MetadataPluginError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    payloads = [report.to_dict() for report in reports]
    if as_json:
        typer.echo(json.dumps(payloads, indent=2, sort_keys=True))
    else:
        for report in reports:
            counts = report.to_dict()["counts"]
            status = "PASS" if report.passed else "FAIL"
            typer.echo(
                f"[{status}] {report.adapter:<14} {report.family:<18} "
                f"passed={counts['passed']} failed={counts['failed']} skipped={counts['skipped']}"
            )
            for failure in report.failures:
                typer.echo(f"    x {failure.name}: {failure.detail}")
    if any(not report.passed for report in reports):
        raise typer.Exit(code=1)


@lineage_app.command("upstream")
def upstream(
    artifact: str = typer.Argument(..., help="Artifact id or domain handle to trace upstream."),
    lake: str = _LAKE_OPTION,
    kind: str | None = _KIND_OPTION,
    max_depth: int | None = _MAX_DEPTH_OPTION,
    edge_type: list[str] | None = _EDGE_TYPE_OPTION,
    target_kind: list[str] | None = _TARGET_KIND_OPTION,
    created_after: str | None = _CREATED_AFTER_OPTION,
    created_before: str | None = _CREATED_BEFORE_OPTION,
    table_version: list[str] | None = _TABLE_VERSION_OPTION,
    page_size: int | None = _PAGE_SIZE_OPTION,
    page_token: str | None = _PAGE_TOKEN_OPTION,
    output_format: str = _FORMAT_OPTION,
    evidence: bool = typer.Option(False, "--evidence", help="Include source/table evidence in report output."),
    max_artifacts: int | None = _MAX_ARTIFACTS_OPTION,
    max_edges: int | None = _MAX_EDGES_OPTION,
    max_executions: int | None = _MAX_EXECUTIONS_OPTION,
    output: str | None = _OUTPUT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Alias for `lineage trace`."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        graph = opened.lineage.trace(
            artifact,
            kind=kind,
            max_depth=max_depth,
            edge_types=edge_type or (),
            target_kinds=target_kind or (),
            created_after=created_after,
            created_before=created_before,
            table_versions=_parse_table_versions(table_version),
            page_size=page_size,
            page_token=page_token,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_graph(
        graph,
        output_format=output_format,
        evidence=evidence,
        max_artifacts=max_artifacts,
        max_edges=max_edges,
        max_executions=max_executions,
        output=output,
    )


@lineage_app.command("impact")
def impact(
    artifact: str = typer.Argument(..., help="Artifact id or domain handle to traverse downstream."),
    lake: str = _LAKE_OPTION,
    kind: str | None = _KIND_OPTION,
    max_depth: int | None = _MAX_DEPTH_OPTION,
    edge_type: list[str] | None = _EDGE_TYPE_OPTION,
    target_kind: list[str] | None = _TARGET_KIND_OPTION,
    created_after: str | None = _CREATED_AFTER_OPTION,
    created_before: str | None = _CREATED_BEFORE_OPTION,
    table_version: list[str] | None = _TABLE_VERSION_OPTION,
    page_size: int | None = _PAGE_SIZE_OPTION,
    page_token: str | None = _PAGE_TOKEN_OPTION,
    output_format: str = _FORMAT_OPTION,
    evidence: bool = typer.Option(False, "--evidence", help="Include source/table evidence in report output."),
    max_artifacts: int | None = _MAX_ARTIFACTS_OPTION,
    max_edges: int | None = _MAX_EDGES_OPTION,
    max_executions: int | None = _MAX_EXECUTIONS_OPTION,
    output: str | None = _OUTPUT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Traverse downstream dependents for an artifact handle."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        graph = opened.lineage.impact(
            artifact,
            kind=kind,
            max_depth=max_depth,
            edge_types=edge_type or (),
            target_kinds=target_kind or (),
            created_after=created_after,
            created_before=created_before,
            table_versions=_parse_table_versions(table_version),
            page_size=page_size,
            page_token=page_token,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_graph(
        graph,
        output_format=output_format,
        evidence=evidence,
        max_artifacts=max_artifacts,
        max_edges=max_edges,
        max_executions=max_executions,
        output=output,
    )


@lineage_app.command("invalidate")
def invalidate(
    artifact: str = typer.Argument(..., help="Artifact id or domain handle to invalidate."),
    lake: str = _LAKE_OPTION,
    kind: str | None = _KIND_OPTION,
    reason: str = typer.Option(..., "--reason", help="Why the artifact/execution is invalid."),
    severity: str = typer.Option("high", "--severity", help="Severity label, e.g. low/medium/high."),
    discovered_by: str | None = typer.Option(None, "--discovered-by", help="System or person that found it."),
    actor: str | None = typer.Option(None, "--actor", help="User or service recording the invalidation."),
    replacement: str | None = typer.Option(None, "--replacement", help="Optional superseding artifact handle."),
    refresh_graph: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Refresh canonical graph tables before resolving the handle.",
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Record an invalidation marker on a lineage artifact."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        marker = opened.lineage.invalidate(
            artifact,
            kind=kind,
            reason=reason,
            severity=severity,
            discovered_by=discovered_by,
            actor=actor,
            replacement=replacement,
            refresh=refresh_graph,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(marker.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("rebuild-plan")
def rebuild_plan(
    artifact: str | None = typer.Argument(
        None,
        help="Artifact id or domain handle to plan from; omit when using --provider.",
    ),
    lake: str = _LAKE_OPTION,
    kind: str | None = _KIND_OPTION,
    provider: str | None = typer.Option(None, "--provider", help="Provider name to invalidate."),
    provider_version: str | None = typer.Option(
        None,
        "--provider-version",
        help="Provider/model version to match.",
    ),
    embedding_column: str | None = typer.Option(
        None,
        "--embedding-column",
        help="Embedding column affected by the provider/version.",
    ),
    reason: str | None = typer.Option(None, "--reason", help="Reason copied into plan actions."),
    severity: str = typer.Option("high", "--severity", help="Severity label for recorded invalidations."),
    discovered_by: str | None = typer.Option(None, "--discovered-by", help="System or person that found it."),
    actor: str | None = typer.Option(None, "--actor", help="User or service recording the plan."),
    replacement: str | None = typer.Option(None, "--replacement", help="Optional superseding artifact handle."),
    record_invalidation: bool = typer.Option(
        False,
        "--record-invalidation",
        help="Persist an invalidation marker in the lineage graph.",
    ),
    refresh_graph: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Refresh canonical graph tables before planning.",
    ),
    max_depth: int | None = _MAX_DEPTH_OPTION,
    action_policy_file: str | None = typer.Option(
        None,
        "--action-policy",
        help="Path to a JSON action-policy file (kind/table/edge/severity action maps).",
    ),
    max_affected_artifacts: int | None = typer.Option(
        None,
        "--max-affected",
        help="Guardrail: fail if more than N artifacts are affected.",
    ),
    max_actions: int | None = typer.Option(
        None,
        "--max-actions",
        help="Guardrail: fail if the plan has more than N actions.",
    ),
    require_indexes: bool = typer.Option(
        False,
        "--require-indexes",
        help="Fail fast if lineage traversal indexes are missing (avoid full scans).",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Summary-first mode: emit aggregate counts only, no per-action rows.",
    ),
    page_size: int | None = typer.Option(
        None,
        "--page-size",
        help="Return a bounded page of actions plus a continuation token.",
    ),
    page_token: str | None = typer.Option(
        None,
        "--page-token",
        help="Continuation handle from a prior page (requires --page-size).",
    ),
    output_format: str = _PLAN_FORMAT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Emit an ordered rebuild/invalidations plan as JSON or tree text."""
    from lancedb_robotics.lineage import LineageError

    action_policy = _load_action_policy(action_policy_file)
    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        plan = opened.lineage.rebuild_plan(
            artifact,
            kind=kind,
            provider=provider,
            provider_version=provider_version,
            embedding_column=embedding_column,
            reason=reason,
            severity=severity,
            discovered_by=discovered_by,
            actor=actor,
            replacement=replacement,
            record_invalidation=record_invalidation,
            refresh=refresh_graph,
            max_depth=max_depth,
            action_policy=action_policy,
            max_affected_artifacts=max_affected_artifacts,
            max_actions=max_actions,
            require_indexes=require_indexes,
            page_size=page_size,
            page_token=page_token,
            summary=summary,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_rebuild_plan(plan, output_format=output_format)


# --- Evidence-pack catalog, retention, and materialization (backlog 0108) ---


def _parse_iso(value: str | None, *, flag: str) -> datetime | None:
    if value is None:
        return None
    from datetime import UTC

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        typer.echo(f"error: {flag} must be ISO-8601, got {value!r}", err=True)
        raise typer.Exit(code=2) from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


_SENSITIVE_SOURCE_OPTION = typer.Option(
    None,
    "--sensitive-source",
    help="Glob/substring for a sensitive source URI; repeat. Flags (or denies) matching packs.",
)
_ON_SENSITIVE_OPTION = typer.Option(
    "flag",
    "--on-sensitive",
    help="What to do on a sensitive-source match: flag (record redacted) or deny (refuse).",
)
_REDACT_OPTION = typer.Option(
    False,
    "--redact",
    help="Strip denied/secret context and environment keys from the pack before it is written.",
)
_REDACT_ALLOW_KEY_OPTION = typer.Option(
    None,
    "--redact-allow-key",
    help="When redacting, keep only these top-level context keys; repeat.",
)
_REDACT_DENY_FRAGMENT_OPTION = typer.Option(
    None,
    "--redact-deny-fragment",
    help="Override the default sensitive key fragments (secret, token, ...); repeat.",
)
_REDACT_DETECT_SECRETS_OPTION = typer.Option(
    False,
    "--redact-detect-secrets",
    help="Also mask values that match built-in credential patterns (AWS/GitHub/JWT/PEM).",
)
_EXTERNAL_CONTEXT_SOURCE_OPTION = typer.Option(
    None,
    "--source",
    help="Backfill source table; repeat (default lineage_executions, transform_runs).",
)


def _build_redaction_policy(
    *,
    redact: bool,
    name: str,
    allow_keys: list[str] | None,
    deny_fragments: list[str] | None,
    detect_secrets: bool,
):
    """Return a ContextRedactionPolicy when any redaction is requested, else None."""
    if not (redact or allow_keys or deny_fragments or detect_secrets):
        return None
    from lancedb_robotics.redaction import (
        DEFAULT_DENY_KEY_FRAGMENTS,
        DEFAULT_SECRET_VALUE_PATTERNS,
        ContextRedactionPolicy,
    )

    return ContextRedactionPolicy(
        name=name or "cli-redact",
        allow_keys=tuple(allow_keys or ()),
        deny_key_fragments=tuple(deny_fragments) if deny_fragments else DEFAULT_DENY_KEY_FRAGMENTS,
        secret_value_patterns=DEFAULT_SECRET_VALUE_PATTERNS if detect_secrets else (),
    )


@lineage_app.command("record-evidence")
def record_evidence(
    artifact: str = typer.Argument(..., help="Checkpoint/model-output/feedback/artifact handle."),
    lake: str = _LAKE_OPTION,
    kind: str | None = _KIND_OPTION,
    checkpoint: bool = typer.Option(False, "--checkpoint", help="Treat the handle as a model/checkpoint run id."),
    where: str | None = _WHERE_OPTION,
    limit: int | None = _LIMIT_OPTION,
    max_depth: int | None = _MAX_DEPTH_OPTION,
    edge_type: list[str] | None = _EDGE_TYPE_OPTION,
    target_kind: list[str] | None = _TARGET_KIND_OPTION,
    refresh_graph: bool = typer.Option(
        True, "--refresh/--no-refresh", help="Refresh canonical graph tables before graph-based export."
    ),
    retention_policy: str = typer.Option("default", "--retention-policy", help="Retention policy label."),
    protected: bool = typer.Option(False, "--protected", help="Mark the pack protected against expiry."),
    expires_at: str | None = typer.Option(None, "--expires-at", help="ISO-8601 expiry time for the pack."),
    redaction_policy: str = typer.Option("", "--redaction-policy", help="Redaction policy label stored on the row."),
    redact: bool = _REDACT_OPTION,
    redact_allow_key: list[str] | None = _REDACT_ALLOW_KEY_OPTION,
    redact_deny_fragment: list[str] | None = _REDACT_DENY_FRAGMENT_OPTION,
    redact_detect_secrets: bool = _REDACT_DETECT_SECRETS_OPTION,
    sensitive_source: list[str] | None = _SENSITIVE_SOURCE_OPTION,
    on_sensitive: str = _ON_SENSITIVE_OPTION,
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the catalog row/event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Build a plan-only evidence pack and record it in the durable catalog."""
    from lancedb_robotics.evidence import EvidencePackError
    from lancedb_robotics.lineage import LineageError

    policy = _build_redaction_policy(
        redact=redact,
        name=redaction_policy or "cli-redact",
        allow_keys=redact_allow_key,
        deny_fragments=redact_deny_fragment,
        detect_secrets=redact_detect_secrets,
    )
    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        pack = opened.lineage.evidence_pack(
            artifact,
            kind=kind,
            checkpoint=checkpoint,
            where=where,
            limit=limit,
            max_depth=max_depth,
            edge_types=edge_type or (),
            target_kinds=target_kind or (),
            refresh=refresh_graph,
            redaction_policy=policy,
        )
        entry = opened.lineage.record_evidence_pack(
            pack,
            retention_policy=retention_policy,
            protected=protected,
            expires_at=_parse_iso(expires_at, flag="--expires-at"),
            redaction_policy=redaction_policy or (policy.name if policy else ""),
            sensitive_source_patterns=sensitive_source or (),
            on_sensitive=on_sensitive,
            created_by=created_by,
        )
    except (EvidencePackError, LineageError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(entry.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("list-evidence")
def list_evidence(
    lake: str = _LAKE_OPTION,
    subject_kind: str | None = typer.Option(None, "--subject-kind", help="Filter by subject kind."),
    subject_handle: str | None = typer.Option(None, "--subject-handle", help="Filter by subject handle."),
    materialization_status: str | None = typer.Option(
        None, "--status", help="Filter by materialization status: planned, partial, materialized."
    ),
    protected: bool | None = typer.Option(None, "--protected/--unprotected", help="Filter by protection flag."),
    page_size: int = typer.Option(50, "--page-size", help="Entries per page."),
    cursor: str | None = typer.Option(None, "--cursor", help="Continuation cursor from a prior page."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """List catalog entries newest-first, filtered and paginated."""
    from lancedb_robotics.evidence import EvidencePackError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        page = opened.lineage.list_evidence_packs(
            subject_kind=subject_kind,
            subject_handle=subject_handle,
            materialization_status=materialization_status,
            protected=protected,
            page_size=page_size,
            cursor=cursor,
        )
    except EvidencePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(page.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("show-evidence")
def show_evidence(
    lake: str = _LAKE_OPTION,
    digest: str | None = typer.Option(None, "--digest", help="Load the pack by manifest digest / pack id."),
    subject: str | None = typer.Option(None, "--subject", help="Load the latest pack for this subject handle."),
    include_manifest: bool = typer.Option(
        False, "--include-manifest", help="Include the full reloaded manifest in the output."
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Reload a catalog entry (and optionally its manifest) by digest or subject."""
    from lancedb_robotics.evidence import EvidencePackError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        entry, manifest = opened.lineage.load_evidence_pack(digest=digest, subject=subject)
    except EvidencePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = entry.as_dict()
    if include_manifest:
        payload["manifest"] = manifest
    typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


@lineage_app.command("materialize-evidence")
def materialize_evidence(
    lake: str = _LAKE_OPTION,
    digest: str = typer.Option(..., "--digest", help="Materialize the pack recorded under this digest."),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Local directory destination."),
    output_uri: str | None = typer.Option(None, "--output-uri", help="Object-store destination URI."),
    include_payloads: bool = typer.Option(False, "--include-payloads", help="Copy observation payload blobs."),
    include_attachments: bool = typer.Option(False, "--include-attachments", help="Copy run attachment blobs."),
    include_video: bool = typer.Option(False, "--include-video", help="Copy video-encoding blobs."),
    max_bytes: int | None = typer.Option(None, "--max-bytes", help="Fail before copying if total exceeds this."),
    max_files: int | None = typer.Option(None, "--max-files", help="Fail before copying if file count exceeds this."),
    chunk_size: int = typer.Option(64, "--chunk-size", help="Blobs copied per bounded chunk."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip objects already present (idempotent)."),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the materialize event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Materialize a recorded pack's blobs to a local/object-store destination."""
    from lancedb_robotics.evidence import EvidencePackError
    from lancedb_robotics.storage import parse_storage_option_pairs

    if (output_dir is None) == (output_uri is None):
        typer.echo("error: pass exactly one of --output-dir / --output-uri", err=True)
        raise typer.Exit(code=2)

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        storage_options = parse_storage_option_pairs(storage_option)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    try:
        _entry, manifest = opened.lineage.load_evidence_pack(digest=digest)
        report = opened.lineage.materialize_evidence_pack(
            manifest,
            output_dir=output_dir,
            output_uri=output_uri,
            include_payloads=include_payloads,
            include_attachments=include_attachments,
            include_video=include_video,
            max_bytes=max_bytes,
            max_files=max_files,
            chunk_size=chunk_size,
            resume=resume,
            storage_options=storage_options or None,
            auth_ref=auth_ref,
            created_by=created_by,
        )
    except EvidencePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(report.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("evidence-retention")
def evidence_retention(
    lake: str = _LAKE_OPTION,
    older_than_days: float | None = typer.Option(
        None, "--older-than-days", help="Treat packs created before now-this-many-days as expirable."
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Report which recorded packs are protected vs safe to expire."""
    from datetime import timedelta

    from lancedb_robotics.evidence import EvidencePackError

    opened = _open_lake(lake, auth_ref, storage_option)
    older_than = timedelta(days=older_than_days) if older_than_days is not None else None
    try:
        plan = opened.lineage.evidence_retention_plan(older_than=older_than)
    except EvidencePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(plan), indent=2, sort_keys=True))


@lineage_app.command("set-evidence-retention")
def set_evidence_retention_cmd(
    lake: str = _LAKE_OPTION,
    digest: str = typer.Option(..., "--digest", help="Pack digest to update."),
    protected: bool | None = typer.Option(None, "--protected/--unprotected", help="Set/clear the protection flag."),
    retention_policy: str | None = typer.Option(None, "--retention-policy", help="New retention policy label."),
    expires_at: str | None = typer.Option(None, "--expires-at", help="New ISO-8601 expiry time."),
    clear_expires_at: bool = typer.Option(False, "--clear-expires-at", help="Clear any expiry time."),
    redaction_policy: str | None = typer.Option(None, "--redaction-policy", help="New redaction policy label."),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Update retention/protection metadata for a recorded pack."""
    from lancedb_robotics.evidence import EvidencePackError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        entry = opened.lineage.set_evidence_retention(
            digest,
            protected=protected,
            retention_policy=retention_policy,
            expires_at=_parse_iso(expires_at, flag="--expires-at"),
            clear_expires_at=clear_expires_at,
            redaction_policy=redaction_policy,
            created_by=created_by,
        )
    except EvidencePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(entry.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("expire-evidence")
def expire_evidence(
    lake: str = _LAKE_OPTION,
    digest: str = typer.Option(..., "--digest", help="Pack digest to expire (delete the catalog row)."),
    force: bool = typer.Option(False, "--force", help="Expire even if the pack is protected."),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Delete a catalog entry (refused if protected unless --force)."""
    from lancedb_robotics.evidence import EvidencePackError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        result = opened.lineage.expire_evidence_pack(digest, force=force, created_by=created_by)
    except EvidencePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(result), indent=2, sort_keys=True))


@lineage_app.command("evidence-events")
def evidence_events(
    lake: str = _LAKE_OPTION,
    pack_id: str | None = typer.Option(None, "--pack-id", help="Filter events by pack id / digest."),
    event_type: str | None = typer.Option(None, "--event-type", help="Filter by event type."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """List evidence-pack audit events (creation, materialization, retention, expiry)."""
    from lancedb_robotics.evidence import EvidencePackError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        events = opened.lineage.evidence_pack_events(pack_id=pack_id, event_type=event_type)
    except EvidencePackError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(events), indent=2, sort_keys=True))


# --- Durable rebuild-plan catalog, approvals, and dispatch (backlog 0109) ---


@lineage_app.command("save-rebuild-plan")
def save_rebuild_plan(
    artifact: str | None = typer.Argument(
        None,
        help="Artifact id or domain handle to plan from; omit when using --provider.",
    ),
    lake: str = _LAKE_OPTION,
    kind: str | None = _KIND_OPTION,
    provider: str | None = typer.Option(None, "--provider", help="Provider name to invalidate."),
    provider_version: str | None = typer.Option(None, "--provider-version", help="Provider/model version to match."),
    embedding_column: str | None = typer.Option(
        None, "--embedding-column", help="Embedding column affected by the provider/version."
    ),
    reason: str | None = typer.Option(None, "--reason", help="Reason copied into plan actions."),
    severity: str = typer.Option("high", "--severity", help="Severity label for the plan."),
    discovered_by: str | None = typer.Option(None, "--discovered-by", help="System or person that found it."),
    actor: str | None = typer.Option(None, "--actor", help="User or service recording the plan."),
    approver: str | None = typer.Option(None, "--approver", help="Approver, required when --status approved."),
    replacement: str | None = typer.Option(None, "--replacement", help="Optional superseding artifact handle."),
    record_invalidation: bool = typer.Option(
        False, "--record-invalidation", help="Persist an invalidation marker in the lineage graph."
    ),
    status: str = typer.Option("draft", "--status", help="Initial lifecycle status (default draft)."),
    note: str | None = typer.Option(None, "--note", help="Free-form note stored on the catalog row."),
    refresh_graph: bool = typer.Option(
        True, "--refresh/--no-refresh", help="Refresh canonical graph tables before planning."
    ),
    max_depth: int | None = _MAX_DEPTH_OPTION,
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the catalog row/event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Build a rebuild plan and record it in the durable catalog (idempotent by digest)."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        plan = opened.lineage.rebuild_plan(
            artifact,
            kind=kind,
            provider=provider,
            provider_version=provider_version,
            embedding_column=embedding_column,
            reason=reason,
            severity=severity,
            discovered_by=discovered_by,
            actor=actor,
            replacement=replacement,
            record_invalidation=record_invalidation,
            refresh=refresh_graph,
            max_depth=max_depth,
        )
        entry = opened.lineage.record_rebuild_plan(
            plan,
            status=status,
            actor=actor,
            approver=approver,
            note=note,
            created_by=created_by,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(entry.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("list-rebuild-plans")
def list_rebuild_plans(
    lake: str = _LAKE_OPTION,
    status: str | None = typer.Option(None, "--status", help="Filter by lifecycle status."),
    invalidation_id: str | None = typer.Option(None, "--invalidation-id", help="Filter by invalidation id."),
    root_artifact_id: str | None = typer.Option(None, "--root", help="Filter by a root artifact id."),
    page_size: int = typer.Option(50, "--page-size", help="Entries per page."),
    cursor: str | None = typer.Option(None, "--cursor", help="Continuation cursor from a prior page."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """List recorded rebuild plans newest-first, filtered and paginated."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        page = opened.lineage.rebuild_plans(
            status=status,
            invalidation_id=invalidation_id,
            root_artifact_id=root_artifact_id,
            page_size=page_size,
            cursor=cursor,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(page.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("show-rebuild-plan")
def show_rebuild_plan(
    plan_id: str = typer.Argument(..., help="Plan id or digest to reload."),
    lake: str = _LAKE_OPTION,
    include_plan: bool = typer.Option(
        False, "--include-plan", help="Include the full stored plan payload (roots + ordered actions)."
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Reload a catalog entry (and optionally its stored plan) by id or digest."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        entry, plan = opened.lineage.get_rebuild_plan(plan_id)
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = entry.as_dict()
    if include_plan:
        payload["plan"] = plan
    typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


@lineage_app.command("set-rebuild-status")
def set_rebuild_status(
    plan_id: str = typer.Argument(..., help="Plan id or digest to update."),
    status: str = typer.Argument(..., help="New lifecycle status."),
    lake: str = _LAKE_OPTION,
    expected_status: str | None = typer.Option(
        None, "--expected-status", help="Guard: current status must equal this or the update is rejected."
    ),
    expected_revision: int | None = typer.Option(
        None, "--expected-revision", help="Guard: current revision must equal this or the update is rejected."
    ),
    actor: str | None = typer.Option(None, "--actor", help="User or service recording the update."),
    approver: str | None = typer.Option(None, "--approver", help="Approver, required when moving to approved."),
    note: str | None = typer.Option(None, "--note", help="Free-form note stored on the row and event."),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Move a recorded rebuild plan to a new lifecycle status (optimistically)."""
    from lancedb_robotics.lineage import LineageError
    from lancedb_robotics.rebuild_catalog import RebuildPlanConflict

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        entry = opened.lineage.update_rebuild_plan_status(
            plan_id,
            status,
            expected_status=expected_status,
            expected_revision=expected_revision,
            actor=actor,
            approver=approver,
            note=note,
            created_by=created_by,
        )
    except RebuildPlanConflict as exc:
        typer.echo(f"conflict: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(entry.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("export-rebuild-plan")
def export_rebuild_plan(
    plan_id: str = typer.Argument(..., help="Plan id or digest to export for an orchestrator."),
    lake: str = _LAKE_OPTION,
    orchestrator: str | None = typer.Option(
        None, "--orchestrator", help="Orchestrator label recorded in the payload (default generic)."
    ),
    run_ref_prefix: str = typer.Option(
        "rebuild", "--run-ref-prefix", help="Prefix for deterministic external run references."
    ),
    dispatch: bool = typer.Option(
        False, "--dispatch", help="Transition an approved plan to dispatched (default is dry-run preview)."
    ),
    ndjson: bool = typer.Option(False, "--ndjson", help="Emit one JSON object per action instead of a payload."),
    actor: str | None = typer.Option(None, "--actor", help="Actor recorded on the dispatch event."),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the dispatch event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Emit a deterministic orchestrator handoff payload; --dispatch to record it."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        payload = opened.lineage.export_rebuild_plan_dispatch(
            plan_id,
            orchestrator=orchestrator,
            run_ref_prefix=run_ref_prefix,
            dry_run=not dispatch,
            actor=actor,
            created_by=created_by,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if ndjson:
        typer.echo(payload.as_ndjson())
        return
    typer.echo(json.dumps(_json_ready(payload.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("rebuild-plan-events")
def rebuild_plan_events(
    lake: str = _LAKE_OPTION,
    plan_id: str | None = typer.Option(None, "--plan-id", help="Filter events by plan id / digest."),
    event_type: str | None = typer.Option(None, "--event-type", help="Filter by event type."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """List rebuild-plan audit events (recording, status transitions, dispatch)."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        events = opened.lineage.rebuild_plan_events(plan_id=plan_id, event_type=event_type)
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(events), indent=2, sort_keys=True))


# --- Retention policy catalog + governance hooks (backlog 0111) --------------


def _load_policy_definition(policy_json: str | None) -> dict[str, Any] | None:
    """Load a policy definition from a file path (or '-' for stdin), else None."""
    if not policy_json:
        return None
    import sys

    try:
        raw = sys.stdin.read() if policy_json == "-" else Path(policy_json).read_text()
        loaded = json.loads(raw)
    except (OSError, ValueError) as exc:
        typer.echo(f"error: could not read --policy-json {policy_json!r}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(loaded, dict):
        typer.echo("error: --policy-json must contain a JSON object", err=True)
        raise typer.Exit(code=2)
    return loaded


_RETENTION_POLICY_NAME_OPTION = typer.Option(
    None,
    "--name",
    help="Policy name (required unless --policy-json supplies it).",
)
_RETENTION_POLICY_VERSION_OPTION = typer.Option("1", "--version", help="Policy version label.")
_RETENTION_SCOPE_KIND_OPTION = typer.Option(
    None,
    "--kind",
    help="Scope: artifact kind (repeatable).",
)
_RETENTION_SCOPE_TABLE_OPTION = typer.Option(
    None,
    "--table",
    help="Scope: canonical table name (repeatable).",
)
_RETENTION_SCOPE_OWNER_SELECTOR_OPTION = typer.Option(
    None,
    "--owner-selector",
    help="Scope: match artifacts whose metadata owner is one of these (repeatable).",
)
_RETENTION_SCOPE_SOURCE_OPTION = typer.Option(
    None,
    "--source",
    help="Scope: source URI (exact or prefix, repeatable).",
)
_RETENTION_SCOPE_DATASET_OPTION = typer.Option(
    None,
    "--dataset",
    help="Scope: dataset id/name (repeatable).",
)
_RETENTION_SCOPE_MODEL_OPTION = typer.Option(
    None,
    "--model",
    help="Scope: model id/name (repeatable).",
)
_RETENTION_SCOPE_DEPLOYMENT_OPTION = typer.Option(
    None,
    "--deployment",
    help="Scope: deployment id (repeatable).",
)
_RETENTION_SCOPE_PROJECT_OPTION = typer.Option(
    None,
    "--project",
    help="Scope: project id (repeatable).",
)
_RETENTION_SCOPE_NAME_PREFIX_OPTION = typer.Option(
    None,
    "--name-prefix",
    help="Scope: artifact name prefix (repeatable).",
)
_RETENTION_SCOPE_ARTIFACT_ID_OPTION = typer.Option(
    None,
    "--artifact-id",
    help="Scope: explicit artifact id (repeatable).",
)
_RETENTION_RETAIN_UNTIL_OPTION = typer.Option(
    None,
    "--retain-until",
    help="Rule: absolute retain-until (ISO-8601).",
)
_RETENTION_RETAIN_DAYS_OPTION = typer.Option(
    None,
    "--retain-days",
    help="Rule: retain N days from each artifact's creation.",
)
_RETENTION_LEGAL_HOLD_OPTION = typer.Option(
    False,
    "--legal-hold",
    help="Rule: indefinite legal hold.",
)
_RETENTION_AUDIT_HOLD_OPTION = typer.Option(
    False,
    "--audit-hold",
    help="Rule: indefinite audit hold.",
)
_RETENTION_PROMOTION_HOLD_OPTION = typer.Option(
    False,
    "--promotion-hold",
    help="Rule: indefinite promotion/deployment hold.",
)
_RETENTION_OWNER_OPTION = typer.Option(
    None,
    "--owner",
    help="Owning team/user stamped onto applied holds.",
)
_RETENTION_REASON_OPTION = typer.Option(
    None,
    "--reason",
    help="Reason template stamped onto applied holds.",
)
_RETENTION_POLICY_JSON_OPTION = typer.Option(
    None,
    "--policy-json",
    help="Path (or '-' for stdin) to a full policy definition JSON, overriding flags.",
)
_RETENTION_POLICY_STATUS_OPTION = typer.Option(
    "draft",
    "--status",
    help="Initial lifecycle status (default draft).",
)
_RETENTION_POLICY_ACTOR_OPTION = typer.Option(
    None,
    "--actor",
    help="User or service recording the policy.",
)
_RETENTION_POLICY_APPROVER_OPTION = typer.Option(
    None,
    "--approver",
    help="Approver, required when --status active.",
)
_RETENTION_POLICY_NOTE_OPTION = typer.Option(
    None,
    "--note",
    help="Free-form note stored on the catalog row.",
)
_RETENTION_POLICY_CREATED_BY_OPTION = typer.Option(
    None,
    "--created-by",
    help="Actor recorded on the catalog row/event.",
)


@lineage_app.command("save-retention-policy")
def save_retention_policy(
    lake: str = _LAKE_OPTION,
    name: str | None = _RETENTION_POLICY_NAME_OPTION,
    version: str = _RETENTION_POLICY_VERSION_OPTION,
    kind: list[str] | None = _RETENTION_SCOPE_KIND_OPTION,
    table: list[str] | None = _RETENTION_SCOPE_TABLE_OPTION,
    owner_selector: list[str] | None = _RETENTION_SCOPE_OWNER_SELECTOR_OPTION,
    source: list[str] | None = _RETENTION_SCOPE_SOURCE_OPTION,
    dataset: list[str] | None = _RETENTION_SCOPE_DATASET_OPTION,
    model: list[str] | None = _RETENTION_SCOPE_MODEL_OPTION,
    deployment: list[str] | None = _RETENTION_SCOPE_DEPLOYMENT_OPTION,
    project: list[str] | None = _RETENTION_SCOPE_PROJECT_OPTION,
    name_prefix: list[str] | None = _RETENTION_SCOPE_NAME_PREFIX_OPTION,
    artifact_id: list[str] | None = _RETENTION_SCOPE_ARTIFACT_ID_OPTION,
    retain_until: str | None = _RETENTION_RETAIN_UNTIL_OPTION,
    retain_days: int | None = _RETENTION_RETAIN_DAYS_OPTION,
    legal_hold: bool = _RETENTION_LEGAL_HOLD_OPTION,
    audit_hold: bool = _RETENTION_AUDIT_HOLD_OPTION,
    promotion_hold: bool = _RETENTION_PROMOTION_HOLD_OPTION,
    owner: str | None = _RETENTION_OWNER_OPTION,
    reason_template: str | None = _RETENTION_REASON_OPTION,
    policy_json: str | None = _RETENTION_POLICY_JSON_OPTION,
    status: str = _RETENTION_POLICY_STATUS_OPTION,
    actor: str | None = _RETENTION_POLICY_ACTOR_OPTION,
    approver: str | None = _RETENTION_POLICY_APPROVER_OPTION,
    note: str | None = _RETENTION_POLICY_NOTE_OPTION,
    created_by: str | None = _RETENTION_POLICY_CREATED_BY_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Record a retention policy in the durable catalog (idempotent by digest)."""
    from lancedb_robotics.lineage import LineageError
    from lancedb_robotics.retention_catalog import build_retention_policy

    definition = _load_policy_definition(policy_json)
    if definition is None:
        if not name:
            typer.echo("error: --name is required unless --policy-json is given", err=True)
            raise typer.Exit(code=2)
        definition = build_retention_policy(
            name=name,
            version=version,
            kinds=kind or (),
            tables=table or (),
            owners=owner_selector or (),
            sources=source or (),
            datasets=dataset or (),
            models=model or (),
            deployments=deployment or (),
            projects=project or (),
            name_prefixes=name_prefix or (),
            artifact_ids=artifact_id or (),
            retain_until=retain_until,
            retain_for_days=retain_days,
            legal_hold=legal_hold,
            audit_hold=audit_hold,
            promotion_hold=promotion_hold,
            owner=owner,
            reason_template=reason_template,
        )

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        entry = opened.lineage.record_retention_policy(
            definition,
            status=status,
            actor=actor,
            approver=approver,
            note=note,
            created_by=created_by,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(entry.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("list-retention-policies")
def list_retention_policies(
    lake: str = _LAKE_OPTION,
    status: str | None = typer.Option(None, "--status", help="Filter by lifecycle status."),
    name: str | None = typer.Option(None, "--name", help="Filter by policy name."),
    owner: str | None = typer.Option(None, "--owner", help="Filter by owning team/user."),
    page_size: int = typer.Option(50, "--page-size", help="Entries per page."),
    cursor: str | None = typer.Option(None, "--cursor", help="Continuation cursor from a prior page."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """List recorded retention policies newest-first, filtered and paginated."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        page = opened.lineage.retention_policies(
            status=status, name=name, owner=owner, page_size=page_size, cursor=cursor
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(page.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("show-retention-policy")
def show_retention_policy(
    policy_id: str = typer.Argument(..., help="Policy id or digest to reload."),
    lake: str = _LAKE_OPTION,
    include_definition: bool = typer.Option(
        False, "--include-definition", help="Include the full stored policy definition (scope + rules)."
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Reload a catalog entry (and optionally its stored definition) by id or digest."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        entry, definition = opened.lineage.get_retention_policy(policy_id)
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = entry.as_dict()
    if include_definition:
        payload["definition"] = definition
    typer.echo(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


@lineage_app.command("set-retention-status")
def set_retention_status(
    policy_id: str = typer.Argument(..., help="Policy id or digest to update."),
    status: str = typer.Argument(..., help="New lifecycle status (draft|active|suspended|archived)."),
    lake: str = _LAKE_OPTION,
    expected_status: str | None = typer.Option(
        None, "--expected-status", help="Guard: current status must equal this or the update is rejected."
    ),
    expected_revision: int | None = typer.Option(
        None, "--expected-revision", help="Guard: current revision must equal this or the update is rejected."
    ),
    actor: str | None = typer.Option(None, "--actor", help="User or service recording the update."),
    approver: str | None = typer.Option(None, "--approver", help="Approver, required when moving to active."),
    note: str | None = typer.Option(None, "--note", help="Free-form note stored on the row and event."),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Move a recorded retention policy to a new lifecycle status (optimistically)."""
    from lancedb_robotics.lineage import LineageError
    from lancedb_robotics.retention_catalog import RetentionPolicyConflict

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        entry = opened.lineage.update_retention_policy_status(
            policy_id,
            status,
            expected_status=expected_status,
            expected_revision=expected_revision,
            actor=actor,
            approver=approver,
            note=note,
            created_by=created_by,
        )
    except RetentionPolicyConflict as exc:
        typer.echo(f"conflict: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(entry.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("apply-retention-policy")
def apply_retention_policy(
    policy_id: str = typer.Argument(..., help="Policy id or digest to apply."),
    lake: str = _LAKE_OPTION,
    apply: bool = typer.Option(
        False, "--apply", help="Write holds to matching artifacts (default is a dry-run preview)."
    ),
    max_artifacts: int | None = typer.Option(
        None, "--max-artifacts", help="Guardrail: refuse if the scope matches more than this many artifacts."
    ),
    actor: str | None = typer.Option(None, "--actor", help="Actor recorded on the apply event."),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the apply event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Expand an active policy into explicit artifact holds; --apply to write."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        result = opened.lineage.apply_retention_policy(
            policy_id,
            dry_run=not apply,
            max_artifacts=max_artifacts,
            actor=actor,
            created_by=created_by,
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(result.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("release-retention-policy")
def release_retention_policy(
    policy_id: str = typer.Argument(..., help="Policy id or digest to release."),
    lake: str = _LAKE_OPTION,
    release: bool = typer.Option(
        False, "--release", help="Clear this policy's applied holds (default is a dry-run preview)."
    ),
    actor: str | None = typer.Option(None, "--actor", help="Actor recorded on the release event."),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the release event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Clear the holds a policy applied, leaving manual/other-policy holds; --release to write."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        result = opened.lineage.release_retention_policy(
            policy_id, dry_run=not release, actor=actor, created_by=created_by
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(result.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("resolve-retention-holds")
def resolve_retention_holds(
    lake: str = _LAKE_OPTION,
    no_snapshot: bool = typer.Option(
        False, "--no-snapshot", help="Exclude dataset-snapshot pins (lineage holds only)."
    ),
    no_conflicts: bool = typer.Option(
        False, "--no-conflicts", help="Skip active-policy shadowing conflict detection."
    ),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Merge policy-applied and artifact-local holds into maintenance's pin shape."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        resolution = opened.lineage.resolve_retention_holds(
            include_snapshot=not no_snapshot, detect_conflicts=not no_conflicts
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(resolution.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("retention-expiration-notices")
def retention_expiration_notices(
    lake: str = _LAKE_OPTION,
    within_days: int | None = typer.Option(
        None, "--within-days", help="Also report holds expiring within this many days (not just already expired)."
    ),
    notify: bool = typer.Option(
        False, "--notify", help="Emit an append-safe expiration-notified audit event per owning policy."
    ),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on notify events."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Report time-based holds that have expired or are expiring soon."""
    from datetime import timedelta

    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    within = timedelta(days=within_days) if within_days is not None else None
    try:
        notices = opened.lineage.retention_expiration_notices(
            within=within, notify=notify, created_by=created_by
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(notices), indent=2, sort_keys=True))


@lineage_app.command("export-retention-state")
def export_retention_state(
    lake: str = _LAKE_OPTION,
    policy_id: str | None = typer.Option(None, "--policy-id", help="Project a single policy by id/digest."),
    status: str | None = typer.Option(None, "--status", help="Project only policies in this status."),
    no_holds: bool = typer.Option(False, "--no-holds", help="Project policy rows only, without resolved holds."),
    ndjson: bool = typer.Option(False, "--ndjson", help="Emit one JSON object per policy/hold record."),
    output: str | None = _OUTPUT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Project policy + resolved-hold state out for external governance systems."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        projection = opened.lineage.export_retention_policy_state(
            policy_id=policy_id, status=status, include_holds=not no_holds
        )
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if ndjson:
        _write_output(projection.as_ndjson(), output)
        return
    _write_output(json.dumps(_json_ready(projection.as_dict()), indent=2, sort_keys=True), output)


@lineage_app.command("retention-policy-events")
def retention_policy_events(
    lake: str = _LAKE_OPTION,
    policy_id: str | None = typer.Option(None, "--policy-id", help="Filter events by policy id / digest."),
    event_type: str | None = typer.Option(None, "--event-type", help="Filter by event type."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """List retention-policy audit events (recording, status, apply/release, expiry)."""
    from lancedb_robotics.lineage import LineageError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        events = opened.lineage.retention_policy_events(policy_id=policy_id, event_type=event_type)
    except LineageError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(events), indent=2, sort_keys=True))


# --- Queryable external-context catalog (backlog 0114) ----------------------


@lineage_app.command("backfill-external-context")
def backfill_external_context(
    lake: str = _LAKE_OPTION,
    source: list[str] | None = _EXTERNAL_CONTEXT_SOURCE_OPTION,
    batch_size: int = typer.Option(512, "--batch-size", help="Rows scanned per batch (bounded memory)."),
    redact: bool = _REDACT_OPTION,
    redact_allow_key: list[str] | None = _REDACT_ALLOW_KEY_OPTION,
    redact_deny_fragment: list[str] | None = _REDACT_DENY_FRAGMENT_OPTION,
    redact_detect_secrets: bool = _REDACT_DETECT_SECRETS_OPTION,
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on catalog rows/events."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Index external run/job context from canonical rows into the catalog (idempotent)."""
    from lancedb_robotics.external_context_catalog import ExternalContextError

    policy = _build_redaction_policy(
        redact=redact,
        name="backfill-redact",
        allow_keys=redact_allow_key,
        deny_fragments=redact_deny_fragment,
        detect_secrets=redact_detect_secrets,
    )
    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        kwargs: dict[str, Any] = {"batch_size": batch_size, "redaction_policy": policy, "created_by": created_by}
        if source:
            kwargs["sources"] = tuple(source)
        report = opened.lineage.backfill_external_contexts(**kwargs)
    except ExternalContextError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(report.as_dict()), indent=2, sort_keys=True))


@lineage_app.command("find-external-context")
def find_external_context(
    lake: str = _LAKE_OPTION,
    provider: str | None = typer.Option(None, "--provider", help="Filter by external provider (mlflow, wandb, ...)."),
    external_run_id: str | None = typer.Option(None, "--run-id", help="Filter by external run id."),
    external_job_id: str | None = typer.Option(None, "--job-id", help="Filter by external job/experiment id."),
    external_parent_run_id: str | None = typer.Option(None, "--parent-run-id", help="Filter by external parent run id."),
    code_ref: str | None = typer.Option(None, "--code-ref", help="Filter by code reference (git SHA)."),
    environment_digest: str | None = typer.Option(None, "--environment-digest", help="Filter by environment digest."),
    artifact_uri: str | None = typer.Option(None, "--artifact-uri", help="Filter by external artifact URI/URN."),
    source_table: str | None = typer.Option(None, "--source-table", help="Filter by backfill source table."),
    page_size: int = typer.Option(50, "--page-size", help="Entries per page."),
    cursor: str | None = typer.Option(None, "--cursor", help="Continuation cursor from a prior page."),
    output: str | None = _OUTPUT_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Resolve external run/job handles to canonical executions/artifacts (paged JSON)."""
    from lancedb_robotics.external_context_catalog import ExternalContextError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        page = opened.lineage.find_external_context(
            provider=provider,
            external_run_id=external_run_id,
            external_job_id=external_job_id,
            external_parent_run_id=external_parent_run_id,
            code_ref=code_ref,
            environment_digest=environment_digest,
            artifact_uri=artifact_uri,
            source_table=source_table,
            page_size=page_size,
            cursor=cursor,
        )
    except ExternalContextError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _write_output(json.dumps(_json_ready(page.as_dict()), indent=2, sort_keys=True), output)


@lineage_app.command("expire-external-context")
def expire_external_context(
    context_id: str = typer.Argument(..., help="Catalog context id to expire."),
    lake: str = _LAKE_OPTION,
    force: bool = typer.Option(False, "--force", help="Expire even if the row is held (protected/legal/audit)."),
    created_by: str | None = typer.Option(None, "--created-by", help="Actor recorded on the event."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """Delete a recorded external-context row (refused if held unless --force)."""
    from lancedb_robotics.external_context_catalog import ExternalContextError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        result = opened.lineage.expire_external_context(context_id, force=force, created_by=created_by)
    except ExternalContextError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(result), indent=2, sort_keys=True))


@lineage_app.command("external-context-events")
def external_context_events(
    lake: str = _LAKE_OPTION,
    context_id: str | None = typer.Option(None, "--context-id", help="Filter events by context id."),
    event_type: str | None = typer.Option(None, "--event-type", help="Filter by event type."),
    auth_ref: str | None = _AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
) -> None:
    """List external-context audit events (backfill, retention, redaction, expiry)."""
    from lancedb_robotics.external_context_catalog import ExternalContextError

    opened = _open_lake(lake, auth_ref, storage_option)
    try:
        events = opened.lineage.external_context_events(context_id=context_id, event_type=event_type)
    except ExternalContextError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(_json_ready(events), indent=2, sort_keys=True))
