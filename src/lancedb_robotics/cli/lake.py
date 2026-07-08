"""`lancedb-robotics lake` subcommands."""

import json
from datetime import timedelta

import typer

lake_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(
    None,
    "--lake",
    help="Path, object-store URI, or db:// Enterprise URI to the lake; omit for namespace-only connections.",
)
_AUTH_REF_OPTION = typer.Option(
    None,
    "--auth-ref",
    help="Backward-compatible credential reference alias; prefer the plane-specific auth-ref options.",
)
_REMOTE_AUTH_REF_OPTION = typer.Option(
    None,
    "--remote-auth-ref",
    help="LanceDB Enterprise API credential reference; resolves at runtime only.",
)
_NAMESPACE_AUTH_REF_OPTION = typer.Option(
    None,
    "--namespace-auth-ref",
    help="REST namespace auth/context reference; resolves at runtime only.",
)
_STORAGE_AUTH_REF_OPTION = typer.Option(
    None,
    "--storage-auth-ref",
    help="Lake object-store credential reference; resolves at runtime only.",
)
_STORAGE_OPTION = typer.Option(
    None,
    "--storage-option",
    help="Storage client option as key=value; repeat for endpoint_url, region, etc.",
)
_REGION_OPTION = typer.Option(None, "--region", help="LanceDB Enterprise region.")
_HOST_OVERRIDE_OPTION = typer.Option(
    None,
    "--host-override",
    help="LanceDB Enterprise private endpoint override.",
)
_CLIENT_CONFIG_OPTION = typer.Option(
    None,
    "--client-config-json",
    help="JSON object forwarded to LanceDB client_config for Enterprise connections.",
)
_NAMESPACE_IMPL_OPTION = typer.Option(
    None,
    "--namespace-impl",
    help="Lance Namespace implementation, usually `rest` for query-node access.",
)
_NAMESPACE_URI_OPTION = typer.Option(
    None,
    "--namespace-uri",
    help="REST namespace endpoint URI.",
)
_NAMESPACE_DATABASE_OPTION = typer.Option(
    None,
    "--namespace-database",
    help="LanceDB database routing header x-lancedb-database.",
)
_NAMESPACE_PREFIX_OPTION = typer.Option(
    None,
    "--namespace-prefix",
    help="LanceDB database prefix routing header x-lancedb-database-prefix.",
)
_NAMESPACE_DELIMITER_OPTION = typer.Option(
    None,
    "--namespace-delimiter",
    help="Namespace path delimiter.",
)
_NAMESPACE_PROPERTY_OPTION = typer.Option(
    None,
    "--namespace-property",
    help="Raw namespace client property as key=value; repeat for header/context properties.",
)
_NAMESPACE_PUSHDOWN_OPTION = typer.Option(
    None,
    "--namespace-pushdown",
    help="Namespace operation to push down; repeat for QueryTable and CreateTable.",
)
_TABLE_OPTION = typer.Option(
    None,
    "--table",
    help="Canonical table to maintain; repeat to limit the run. Defaults to all tables.",
)
_COMPACT_OPTION = typer.Option(
    True,
    "--compact/--no-compact",
    help="Compact table fragments.",
)
_REFRESH_INDEXES_OPTION = typer.Option(
    True,
    "--refresh-indexes/--no-refresh-indexes",
    help="Refresh existing persistent FTS/vector indexes.",
)
_RETENTION_OPTION = typer.Option(
    True,
    "--retention/--no-retention",
    help="Clean old unpinned table versions after tagging snapshot-pinned versions.",
)
_CLEANUP_OLDER_THAN_DAYS_OPTION = typer.Option(
    7.0,
    "--cleanup-older-than-days",
    help="Retention age threshold in days; use 0 with --delete-unverified for tests/local exclusive runs.",
)
_RETAIN_VERSIONS_OPTION = typer.Option(
    None,
    "--retain-versions",
    help="Also keep at least this many recent versions per table during retention.",
)
_DELETE_UNVERIFIED_OPTION = typer.Option(
    False,
    "--delete-unverified",
    help="Allow deleting recent unverified files; only use when no other process is writing the lake.",
)
_REQUIRE_RECENT_AUDIT_OPTION = typer.Option(
    False,
    "--require-recent-audit/--no-require-recent-audit",
    help="Require a recent passed persisted lineage audit report before cleanup.",
)
_AUDIT_MAX_AGE_HOURS_OPTION = typer.Option(
    24.0,
    "--audit-max-age-hours",
    help="Maximum age for --require-recent-audit.",
)
_LEROBOT_CHECKPOINT_RETENTION_OPTION = typer.Option(
    True,
    "--lerobot-checkpoint-retention/--no-lerobot-checkpoint-retention",
    help="Summarize old LeRobot ingest checkpoint histories before maintaining the checkpoint table.",
)
_LEROBOT_CHECKPOINT_RETENTION_DAYS_OPTION = typer.Option(
    30.0,
    "--lerobot-checkpoint-retention-days",
    help="Only summarize terminal LeRobot jobs older than this many days; use -1 to ignore age.",
)
_LEROBOT_CHECKPOINT_SOURCE_ID_OPTION = typer.Option(
    None,
    "--lerobot-checkpoint-source-id",
    help="Limit LeRobot checkpoint retention to one source id.",
)
_LEROBOT_CHECKPOINT_STATUS_OPTION = typer.Option(
    None,
    "--lerobot-checkpoint-status",
    help="Terminal LeRobot job status eligible for checkpoint retention; repeat for completed/failed/skipped.",
)
_LEROBOT_RETAIN_COMPLETED_OPTION = typer.Option(
    10,
    "--lerobot-retain-completed-per-source",
    help="Keep this many completed LeRobot job histories fully expanded per source.",
)
_LEROBOT_RETAIN_FAILED_OPTION = typer.Option(
    10,
    "--lerobot-retain-failed-per-source",
    help="Keep this many failed LeRobot job histories fully expanded per source.",
)


@lake_app.command("init")
def init(
    lake: str | None = _LAKE_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    namespace_auth_ref: str | None = _NAMESPACE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
    client_config_json: str | None = _CLIENT_CONFIG_OPTION,
    namespace_impl: str | None = _NAMESPACE_IMPL_OPTION,
    namespace_uri: str | None = _NAMESPACE_URI_OPTION,
    namespace_database: str | None = _NAMESPACE_DATABASE_OPTION,
    namespace_prefix: str | None = _NAMESPACE_PREFIX_OPTION,
    namespace_delimiter: str | None = _NAMESPACE_DELIMITER_OPTION,
    namespace_property: list[str] | None = _NAMESPACE_PROPERTY_OPTION,
    namespace_pushdown: list[str] | None = _NAMESPACE_PUSHDOWN_OPTION,
) -> None:
    """Create the canonical lake tables (idempotent, ingests nothing)."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.schemas import SCHEMA_VERSIONS
    from lancedb_robotics.storage import parse_storage_option_pairs

    try:
        storage_options = parse_storage_option_pairs(storage_option)
        namespace_properties = _parse_namespace_properties(
            namespace_uri=namespace_uri,
            namespace_database=namespace_database,
            namespace_prefix=namespace_prefix,
            namespace_delimiter=namespace_delimiter,
            namespace_property=namespace_property,
        )
        client_config = _parse_client_config(client_config_json)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        opened = Lake.init(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            namespace_auth_ref=namespace_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_options=storage_options,
            region=region,
            host_override=host_override,
            client_config=client_config,
            namespace_client_impl=namespace_impl,
            namespace_client_properties=namespace_properties,
            namespace_client_pushdown_operations=namespace_pushdown,
        )
    except LakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"lake: {opened.uri}")
    for name in opened.table_names():
        typer.echo(f"  {name} (v{SCHEMA_VERSIONS[name]})")


@lake_app.command("tables")
def tables(
    lake: str | None = _LAKE_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    namespace_auth_ref: str | None = _NAMESPACE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
    client_config_json: str | None = _CLIENT_CONFIG_OPTION,
    namespace_impl: str | None = _NAMESPACE_IMPL_OPTION,
    namespace_uri: str | None = _NAMESPACE_URI_OPTION,
    namespace_database: str | None = _NAMESPACE_DATABASE_OPTION,
    namespace_prefix: str | None = _NAMESPACE_PREFIX_OPTION,
    namespace_delimiter: str | None = _NAMESPACE_DELIMITER_OPTION,
    namespace_property: list[str] | None = _NAMESPACE_PROPERTY_OPTION,
    namespace_pushdown: list[str] | None = _NAMESPACE_PUSHDOWN_OPTION,
) -> None:
    """List canonical tables with row counts and schema versions."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.storage import parse_storage_option_pairs

    try:
        storage_options = parse_storage_option_pairs(storage_option)
        namespace_properties = _parse_namespace_properties(
            namespace_uri=namespace_uri,
            namespace_database=namespace_database,
            namespace_prefix=namespace_prefix,
            namespace_delimiter=namespace_delimiter,
            namespace_property=namespace_property,
        )
        client_config = _parse_client_config(client_config_json)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        opened = Lake.open(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            namespace_auth_ref=namespace_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_options=storage_options,
            region=region,
            host_override=host_override,
            client_config=client_config,
            namespace_client_impl=namespace_impl,
            namespace_client_properties=namespace_properties,
            namespace_client_pushdown_operations=namespace_pushdown,
        )
    except LakeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    versions = opened.schema_versions()
    for name in opened.table_names():
        rows = opened.table(name).count_rows()
        typer.echo(f"{name}\tv{versions[name]}\t{rows} rows")


@lake_app.command("maintain")
def maintain(
    lake: str | None = _LAKE_OPTION,
    tables: list[str] | None = _TABLE_OPTION,
    compact: bool = _COMPACT_OPTION,
    refresh_indexes: bool = _REFRESH_INDEXES_OPTION,
    retention: bool = _RETENTION_OPTION,
    cleanup_older_than_days: float = _CLEANUP_OLDER_THAN_DAYS_OPTION,
    retain_versions: int | None = _RETAIN_VERSIONS_OPTION,
    delete_unverified: bool = _DELETE_UNVERIFIED_OPTION,
    require_recent_audit: bool = _REQUIRE_RECENT_AUDIT_OPTION,
    audit_max_age_hours: float = _AUDIT_MAX_AGE_HOURS_OPTION,
    lerobot_checkpoint_retention: bool = _LEROBOT_CHECKPOINT_RETENTION_OPTION,
    lerobot_checkpoint_retention_days: float = _LEROBOT_CHECKPOINT_RETENTION_DAYS_OPTION,
    lerobot_checkpoint_source_id: str | None = _LEROBOT_CHECKPOINT_SOURCE_ID_OPTION,
    lerobot_checkpoint_status: list[str] | None = _LEROBOT_CHECKPOINT_STATUS_OPTION,
    lerobot_retain_completed_per_source: int = _LEROBOT_RETAIN_COMPLETED_OPTION,
    lerobot_retain_failed_per_source: int = _LEROBOT_RETAIN_FAILED_OPTION,
    auth_ref: str | None = _AUTH_REF_OPTION,
    remote_auth_ref: str | None = _REMOTE_AUTH_REF_OPTION,
    namespace_auth_ref: str | None = _NAMESPACE_AUTH_REF_OPTION,
    storage_auth_ref: str | None = _STORAGE_AUTH_REF_OPTION,
    storage_option: list[str] | None = _STORAGE_OPTION,
    region: str | None = _REGION_OPTION,
    host_override: str | None = _HOST_OVERRIDE_OPTION,
    client_config_json: str | None = _CLIENT_CONFIG_OPTION,
    namespace_impl: str | None = _NAMESPACE_IMPL_OPTION,
    namespace_uri: str | None = _NAMESPACE_URI_OPTION,
    namespace_database: str | None = _NAMESPACE_DATABASE_OPTION,
    namespace_prefix: str | None = _NAMESPACE_PREFIX_OPTION,
    namespace_delimiter: str | None = _NAMESPACE_DELIMITER_OPTION,
    namespace_property: list[str] | None = _NAMESPACE_PROPERTY_OPTION,
    namespace_pushdown: list[str] | None = _NAMESPACE_PUSHDOWN_OPTION,
) -> None:
    """Compact fragments, refresh indexes, and prune unpinned old versions."""
    from lancedb_robotics.lake import Lake, LakeError
    from lancedb_robotics.maintenance import MaintenanceError, maintain_lake
    from lancedb_robotics.storage import parse_storage_option_pairs

    try:
        storage_options = parse_storage_option_pairs(storage_option)
        namespace_properties = _parse_namespace_properties(
            namespace_uri=namespace_uri,
            namespace_database=namespace_database,
            namespace_prefix=namespace_prefix,
            namespace_delimiter=namespace_delimiter,
            namespace_property=namespace_property,
        )
        client_config = _parse_client_config(client_config_json)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        opened = Lake.open(
            lake,
            auth_ref=auth_ref,
            remote_auth_ref=remote_auth_ref,
            namespace_auth_ref=namespace_auth_ref,
            storage_auth_ref=storage_auth_ref,
            storage_options=storage_options,
            region=region,
            host_override=host_override,
            client_config=client_config,
            namespace_client_impl=namespace_impl,
            namespace_client_properties=namespace_properties,
            namespace_client_pushdown_operations=namespace_pushdown,
        )
        cleanup_older_than = (
            timedelta(days=cleanup_older_than_days) if retention else None
        )
        report = maintain_lake(
            opened,
            tables=tuple(tables or ()),
            compact=compact,
            refresh_indexes=refresh_indexes,
            cleanup_older_than=cleanup_older_than,
            retain_versions=retain_versions,
            delete_unverified=delete_unverified,
            require_recent_audit=require_recent_audit,
            audit_max_age=timedelta(hours=audit_max_age_hours),
            lerobot_checkpoint_retention=lerobot_checkpoint_retention,
            lerobot_checkpoint_retention_older_than=(
                None
                if lerobot_checkpoint_retention_days < 0
                else timedelta(days=lerobot_checkpoint_retention_days)
            ),
            lerobot_checkpoint_retention_source_id=lerobot_checkpoint_source_id,
            lerobot_checkpoint_retention_statuses=(
                tuple(lerobot_checkpoint_status) if lerobot_checkpoint_status else None
            ),
            lerobot_checkpoint_retain_completed_per_source=lerobot_retain_completed_per_source,
            lerobot_checkpoint_retain_failed_per_source=lerobot_retain_failed_per_source,
        )
    except (LakeError, MaintenanceError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"lake: {report.lake_uri}")
    typer.echo(f"transform: {report.transform_id}")
    if report.required_audit_report:
        typer.echo(f"required audit report: {report.required_audit_report['report_id']}")
    if report.lerobot_checkpoint_retention:
        retention_report = report.lerobot_checkpoint_retention
        typer.echo(
            "lerobot checkpoints: "
            f"rows {retention_report['rows_before']}->{retention_report['rows_after']} "
            f"(deleted {retention_report['rows_deleted']}), "
            f"versions v{retention_report['version_before']}->v{retention_report['version_after']}, "
            f"jobs compacted {retention_report['jobs_compacted']}, "
            f"protected {retention_report['jobs_protected']}"
        )
    for name, result in report.tables.items():
        typer.echo(
            f"{name}: fragments {result.fragments_before}->{result.fragments_after} "
            f"(removed {result.fragments_removed}, added {result.fragments_added})"
        )
        if result.indexes_refreshed:
            typer.echo(f"  indexes refreshed: {len(result.indexes_refreshed)}")
        if result.pinned_versions:
            pins = ", ".join(str(version) for version in result.pinned_versions)
            typer.echo(f"  protected versions: {pins}")
        if result.retention_reasons:
            categories = sorted(
                {
                    category
                    for row in result.retention_reasons
                    for category in row.get("categories", [])
                }
            )
            typer.echo(f"  retained for: {', '.join(categories)}")
        if result.cleanup_candidate_versions:
            candidates = ", ".join(str(version) for version in result.cleanup_candidate_versions)
            typer.echo(f"  cleanup candidates: {candidates}")
        if result.cleanup:
            typer.echo(f"  old versions removed: {result.cleanup['old_versions']}")


def _parse_namespace_properties(
    *,
    namespace_uri: str | None,
    namespace_database: str | None,
    namespace_prefix: str | None,
    namespace_delimiter: str | None,
    namespace_property: list[str] | None,
) -> dict[str, str] | None:
    from lancedb_robotics.connections import namespace_properties_from_options
    from lancedb_robotics.storage import parse_storage_option_pairs

    raw = parse_storage_option_pairs(namespace_property)
    properties = namespace_properties_from_options(
        uri=namespace_uri,
        database=namespace_database,
        database_prefix=namespace_prefix,
        delimiter=namespace_delimiter,
        properties=raw,
    )
    return properties or None


def _parse_client_config(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--client-config-json must be a JSON object: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("--client-config-json must be a JSON object")
    return decoded
