"""Lake open/create API: the canonical-table substrate over a LanceDB database."""

from pathlib import Path
from typing import Any

import lancedb

from lancedb_robotics.connections import (
    ConnectionResolverError,
    LakeConnectionSpec,
    resolve_lake_connection,
)
from lancedb_robotics.schemas import (
    CANONICAL_TABLES,
    SCHEMA_METADATA_VERSION_KEY,
    TABLE_SCHEMAS,
)
from lancedb_robotics.storage import StorageConfigError


class LakeError(Exception):
    """Raised when a lake cannot be opened or is not a lancedb-robotics lake."""


def _list_tables(db: lancedb.DBConnection) -> list[str]:
    # lancedb 0.33 returns a ListTablesResponse(tables=[...]) here.
    return list(db.list_tables().tables or [])


def _connect(spec: LakeConnectionSpec) -> lancedb.DBConnection:
    kwargs = dict(spec.lancedb_connect_kwargs)
    try:
        if spec.uri is None:
            return lancedb.connect(**kwargs)
        return lancedb.connect(spec.uri, **kwargs)
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        StorageConfigError,
    ) as exc:
        raise LakeError(
            f"cannot connect to lake at {spec.display_uri}: {_actionable_storage_error(exc, spec)}"
        ) from exc


def _actionable_storage_error(exc: Exception, spec: LakeConnectionSpec | None = None) -> str:
    message = str(exc) or type(exc).__name__
    if spec and spec.kind == "lancedb_remote_db" and isinstance(
        exc, (ImportError, ModuleNotFoundError)
    ):
        return (
            f"{message}; install LanceDB with Enterprise remote DB support and configure "
            "remote_auth_ref/LANCEDB_API_KEY"
        )
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return (
            f"{message}; install the matching object-store dependency "
            "(for example lancedb-robotics[object-store]) or configure credentials"
        )
    return message


class Lake:
    """A LanceDB robotics lake: one database holding the canonical tables.

    Use :meth:`Lake.init` to create or update a lake in place (idempotent),
    and :meth:`Lake.open` to attach to an existing one without creating
    anything.
    """

    def __init__(
        self,
        db: lancedb.DBConnection,
        uri: str,
        *,
        connection_spec: LakeConnectionSpec | None = None,
    ) -> None:
        self._db = db
        self.uri = uri
        self.connection_spec = connection_spec
        self.capabilities = connection_spec.capabilities if connection_spec else None

    @classmethod
    def init(
        cls,
        uri: str | Path | None = None,
        *,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
        remote_auth_ref: str | None = None,
        namespace_auth_ref: str | None = None,
        storage_auth_ref: str | None = None,
        region: str | None = None,
        host_override: str | None = None,
        client_config: dict[str, Any] | None = None,
        namespace_client_impl: str | None = None,
        namespace_client_properties: dict[str, Any] | None = None,
        namespace_client_pushdown_operations: list[str] | tuple[str, ...] | None = None,
    ) -> "Lake":
        """Create the canonical tables at ``uri``, leaving existing ones intact.

        Idempotent: missing tables are created empty, existing tables (and
        their rows) are left untouched. Raises ``ValueError`` if an existing
        table's schema conflicts with the canonical schema.
        """
        try:
            spec = resolve_lake_connection(
                uri,
                storage_options=storage_options,
                auth_ref=auth_ref,
                remote_auth_ref=remote_auth_ref,
                namespace_auth_ref=namespace_auth_ref,
                storage_auth_ref=storage_auth_ref,
                region=region,
                host_override=host_override,
                client_config=client_config,
                namespace_client_impl=namespace_client_impl,
                namespace_client_properties=namespace_client_properties,
                namespace_client_pushdown_operations=namespace_client_pushdown_operations,
            )
        except (ConnectionResolverError, StorageConfigError) as exc:
            raise LakeError(str(exc)) from exc
        db = _connect(spec)
        lake = cls(db, spec.display_uri, connection_spec=spec)
        try:
            existing = set(_list_tables(db))
            for name, schema in TABLE_SCHEMAS.items():
                if name == "aligned_ticks" and name in existing:
                    # LanceDB 0.33 can report Arrow JSON extension schemas with
                    # storage metadata that fail create_table(..., exist_ok=True)
                    # validation even though the persisted table is usable.
                    continue
                db.create_table(name, schema=schema, exist_ok=True)
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise LakeError(
                f"cannot initialize lake at {uri}: {_actionable_storage_error(exc)}"
            ) from exc
        return lake

    @classmethod
    def open(
        cls,
        uri: str | Path | None = None,
        *,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
        remote_auth_ref: str | None = None,
        namespace_auth_ref: str | None = None,
        storage_auth_ref: str | None = None,
        region: str | None = None,
        host_override: str | None = None,
        client_config: dict[str, Any] | None = None,
        namespace_client_impl: str | None = None,
        namespace_client_properties: dict[str, Any] | None = None,
        namespace_client_pushdown_operations: list[str] | tuple[str, ...] | None = None,
    ) -> "Lake":
        """Attach to an existing lake. Raises :class:`LakeError` if ``uri`` does
        not contain the canonical tables; never creates or modifies anything."""
        try:
            spec = resolve_lake_connection(
                uri,
                storage_options=storage_options,
                auth_ref=auth_ref,
                remote_auth_ref=remote_auth_ref,
                namespace_auth_ref=namespace_auth_ref,
                storage_auth_ref=storage_auth_ref,
                region=region,
                host_override=host_override,
                client_config=client_config,
                namespace_client_impl=namespace_client_impl,
                namespace_client_properties=namespace_client_properties,
                namespace_client_pushdown_operations=namespace_client_pushdown_operations,
            )
        except (ConnectionResolverError, StorageConfigError) as exc:
            raise LakeError(str(exc)) from exc
        if spec.local_path_required and spec.uri is not None and not Path(spec.uri).exists():
            raise LakeError(f"no lake at {uri}; run `lancedb-robotics lake init` first")
        db = _connect(spec)
        try:
            existing = set(_list_tables(db))
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise LakeError(f"cannot inspect lake at {uri}: {_actionable_storage_error(exc)}") from exc
        missing = [name for name in CANONICAL_TABLES if name not in existing]
        if missing:
            raise LakeError(
                f"{uri} is missing canonical tables {missing}; "
                "run `lancedb-robotics lake init` to create them"
            )
        return cls(db, spec.display_uri, connection_spec=spec)

    def table_names(self) -> list[str]:
        """Canonical tables present, in canonical order."""
        existing = set(_list_tables(self._db))
        return [name for name in CANONICAL_TABLES if name in existing]

    def table(self, name: str) -> lancedb.table.Table:
        if name not in CANONICAL_TABLES:
            raise LakeError(f"unknown canonical table {name!r}; expected one of {CANONICAL_TABLES}")
        return self._db.open_table(name)

    @property
    def training(self):
        """Lance-native training views over version-pinned dataset snapshots."""
        from lancedb_robotics.training import LakeTraining

        return LakeTraining(self)

    @property
    def projections(self):
        """External-format live/export/plan projections over dataset snapshots."""
        from lancedb_robotics.projections import LakeProjections

        return LakeProjections(self)

    @property
    def episodes(self):
        """Episode derivation and frame/window accessors."""
        from lancedb_robotics.episodes import LakeEpisodes

        return LakeEpisodes(self)

    @property
    def video(self):
        """Codec-aware video encoding and GOP/frame accessors."""
        from lancedb_robotics.video import LakeVideo

        return LakeVideo(self)

    @property
    def align(self):
        """Multi-rate temporal alignment over canonical observations."""
        from lancedb_robotics.align import LakeAlign

        return LakeAlign(self)

    @property
    def curate(self):
        """Curation and mining workbench over scenario snapshots."""
        from lancedb_robotics.curate import LakeCurate

        return LakeCurate(self)

    @property
    def embeddings(self):
        """Pluggable embedding-creation pipelines: providers, decoders, embed(spec)."""
        from lancedb_robotics.embeddings import LakeEmbeddings

        return LakeEmbeddings(self)

    @property
    def distributions(self):
        """Distribution specs, balance reports, comparisons, and gap findings."""
        from lancedb_robotics.distributions import LakeDistributions

        return LakeDistributions(self)

    @property
    def lineage(self):
        """Checkpoint, snapshot, and source-log lineage queries."""
        from lancedb_robotics.lineage import LakeLineage

        return LakeLineage(self)

    @property
    def eval(self):
        """Evaluation run manifests over pinned snapshots and model artifacts."""
        from lancedb_robotics.run_manifests import LakeEval

        return LakeEval(self)

    @property
    def evaluation(self):
        """Alias for ``lake.eval`` for callers that avoid built-in names."""
        return self.eval

    @property
    def tracker(self):
        """External experiment-tracker manifest import/export/drift sync (0101)."""
        from lancedb_robotics.tracker_sync import LakeTrackerSync

        return LakeTrackerSync(self)

    def scope(self, **filters: Any):
        """Build a reusable curation scope from scalar filters."""
        from lancedb_robotics.curate import CurationScope

        return CurationScope.from_filters(**filters)

    def schema_versions(self) -> dict[str, str]:
        """Schema version per table, read back from persisted schema metadata."""
        versions: dict[str, str] = {}
        for name in self.table_names():
            metadata = self.table(name).schema.metadata or {}
            raw = metadata.get(SCHEMA_METADATA_VERSION_KEY.encode())
            versions[name] = raw.decode() if raw else "unknown"
        return versions
