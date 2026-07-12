"""Typed connection resolution for local, remote, namespace, and raw paths.

Backlog 0036 keeps the LanceDB table/query path and the pylance data-plane path
explicit. This module is the narrow seam where robotics SDK inputs become
LanceDB ``connect`` kwargs, Lance REST Namespace clients, or raw object handles.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lancedb_robotics.storage import (
    StorageConfigError,
    is_object_store_uri,
    lancedb_storage_options,
    resolve_storage_options,
    uri_scheme,
)

SUPPORTED_NAMESPACE_PUSHDOWNS = frozenset({"QueryTable", "CreateTable"})
UNSUPPORTED_LANCEDB_SCHEMES = frozenset({"lancedb", "phalanx"})
WORKER_NAMESPACE_PROPERTY_PREFIX = "_lancedb_worker_."


class ConnectionResolverError(Exception):
    """Base class for connection resolver failures."""


class UnsupportedLakeSchemeError(ConnectionResolverError):
    """Raised when a URI looks like an unsupported LanceDB/Lance API."""


class MissingEnterpriseCredentialsError(ConnectionResolverError):
    """Raised when a ``db://`` path cannot resolve a runtime API key."""


class NamespaceConfigError(ConnectionResolverError):
    """Raised when namespace configuration is incomplete or inconsistent."""


class NamespaceResolutionError(ConnectionResolverError):
    """Raised when a namespace client cannot be created or queried."""


class ScopedCredentialsExpired(ConnectionResolverError):
    """Raised when namespace-vended storage credentials are expired."""


class ManagedVersioningMismatch(ConnectionResolverError):
    """Raised when namespace managed-versioning state cannot be honored."""


@dataclass(frozen=True)
class LakeCapabilities:
    """Capabilities implied by the resolved backend path.

    ``server_side_query``/``direct_object_io``/``namespace_*``/``blob_fetch_remote``
    describe the data-plane the backend exposes. ``index_management``,
    ``table_versioning``, and ``schema_evolution`` are the control-plane
    capabilities the backlog-0128 capability gates consult before running the
    ``index``/``versioning``/``schema`` operation families classified by the 0076
    invocation audit. They default to ``False`` (the honest "not advertised"
    state) so a backend must positively claim them before those paths run.
    """

    server_side_query: bool = False
    direct_object_io: bool = False
    namespace_resolution: bool = False
    namespace_managed_versioning: bool = False
    geneva_worker_specs: bool = False
    blob_fetch_remote: bool = False
    index_management: bool = False
    table_versioning: bool = False
    schema_evolution: bool = False


@dataclass(frozen=True)
class PylanceAccessSpec:
    """Logical pylance namespace access that is safe to serialize."""

    namespace_client_impl: str
    namespace_client_properties: dict[str, str]
    namespace_auth_ref: str | None = None
    storage_auth_ref: str | None = None

    def worker_properties(self) -> dict[str, str]:
        """Properties after applying worker-specific namespace overrides."""
        return apply_worker_namespace_overrides(self.namespace_client_properties)


#: Data-plane labels (0129): how a read/write reaches storage. Provenance, not
#: secrets -- recorded in reports/manifests so a reader can tell whether IO went
#: through a local path, object store, the namespace-direct pylance data-plane,
#: or a ``db://`` remote query node.
DATA_PLANE_LOCAL = "local"
DATA_PLANE_OBJECT_STORE = "object_store"
DATA_PLANE_NAMESPACE_DIRECT = "namespace_direct"
DATA_PLANE_REMOTE_DB = "remote_db"
DATA_PLANE_UNCLASSIFIED = "unclassified"

_DATA_PLANE_BY_KIND = {
    "local_path": DATA_PLANE_LOCAL,
    "object_store_lancedb_oss": DATA_PLANE_OBJECT_STORE,
    "lancedb_remote_db": DATA_PLANE_REMOTE_DB,
}


@dataclass(frozen=True)
class LakeConnectionSpec:
    """Resolved LanceDB/Lance connection intent."""

    kind: str
    uri: str | None
    display_uri: str
    lancedb_connect_kwargs: dict[str, Any] = field(default_factory=dict)
    namespace_client_impl: str | None = None
    namespace_client_properties: dict[str, str] = field(default_factory=dict)
    namespace_client_pushdown_operations: tuple[str, ...] = ()
    pylance_access: PylanceAccessSpec | None = None
    auth_refs: dict[str, str | None] = field(default_factory=dict)
    direct_object_io_allowed: bool = False
    managed_versioning: bool = False
    capabilities: LakeCapabilities = field(default_factory=LakeCapabilities)
    local_path_required: bool = False

    @property
    def data_plane(self) -> str:
        """The data-plane label for how this backend reaches storage (0129).

        A namespace backend (``pylance_access`` set) is ``namespace_direct``
        regardless of ``kind``, because it routes through the direct-pylance
        adapter; otherwise the label follows ``kind``.
        """
        if self.pylance_access is not None:
            return DATA_PLANE_NAMESPACE_DIRECT
        return _DATA_PLANE_BY_KIND.get(self.kind, DATA_PLANE_UNCLASSIFIED)

    def safe_summary(self) -> dict[str, Any]:
        """Secret-free diagnostic summary suitable for logs or task records."""
        return {
            "kind": self.kind,
            "data_plane": self.data_plane,
            "uri": self.display_uri,
            "namespace_client_impl": self.namespace_client_impl,
            "namespace_client_properties": safe_namespace_properties(
                self.namespace_client_properties
            ),
            "namespace_client_pushdown_operations": list(
                self.namespace_client_pushdown_operations
            ),
            "auth_refs": {key: value for key, value in self.auth_refs.items() if value},
            "direct_object_io_allowed": self.direct_object_io_allowed,
            "managed_versioning": self.managed_versioning,
            "capabilities": self.capabilities.__dict__,
        }


@dataclass(frozen=True)
class NamespaceTableDescription:
    """Namespace ``describe_table`` response reduced to SDK-relevant fields."""

    table_id: tuple[str, ...]
    location: str | None
    storage_options: dict[str, str]
    expires_at_millis: int | None = None
    managed_versioning: bool = False
    raw_response: Any = None

    def credentials_expired(
        self,
        *,
        now_millis: int | None = None,
        refresh_margin_millis: int = 0,
    ) -> bool:
        if self.expires_at_millis is None:
            return False
        now = _now_millis() if now_millis is None else now_millis
        return self.expires_at_millis <= now + refresh_margin_millis


@dataclass
class PylanceNamespaceAccess:
    """Direct pylance access through a Lance Namespace client."""

    namespace_client: Any
    table_id: tuple[str, ...]
    namespace_client_impl: str | None = None
    namespace_client_properties: dict[str, str] = field(default_factory=dict)
    namespace_auth_ref: str | None = None
    storage_auth_ref: str | None = None
    _last_description: NamespaceTableDescription | None = field(default=None, init=False)

    def describe(
        self,
        *,
        version: int | str | None = None,
        vend_credentials: bool = True,
        expected_managed_versioning: bool | None = None,
    ) -> NamespaceTableDescription:
        """Call namespace ``describe_table`` and parse direct IO metadata."""
        request = make_describe_table_request(
            self.table_id, version=version, vend_credentials=vend_credentials
        )
        try:
            response = self.namespace_client.describe_table(request)
        except Exception as exc:  # pragma: no cover - concrete clients vary.
            raise NamespaceResolutionError(
                f"namespace describe_table failed for {list(self.table_id)}: {exc}"
            ) from exc
        description = namespace_table_description(self.table_id, response)
        if expected_managed_versioning is not None and (
            description.managed_versioning != expected_managed_versioning
        ):
            raise ManagedVersioningMismatch(
                "namespace managed_versioning="
                f"{description.managed_versioning} for {list(self.table_id)}, "
                f"expected {expected_managed_versioning}"
            )
        self._last_description = description
        return description

    def refresh_if_needed(
        self,
        description: NamespaceTableDescription | None = None,
        *,
        now_millis: int | None = None,
        refresh_margin_millis: int = 60_000,
    ) -> NamespaceTableDescription:
        """Refresh namespace-vended credentials when they are near expiration."""
        current = description or self._last_description or self.describe()
        if current.credentials_expired(
            now_millis=now_millis, refresh_margin_millis=refresh_margin_millis
        ):
            current = self.describe()
            if current.credentials_expired(now_millis=now_millis):
                raise ScopedCredentialsExpired(
                    f"namespace credentials for {list(self.table_id)} are expired"
                )
        return current

    def open_dataset(
        self,
        *,
        dataset_factory: Any | None = None,
        expected_managed_versioning: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open ``lance.dataset(namespace_client=..., table_id=...)``."""
        description = self.describe(
            vend_credentials=True,
            expected_managed_versioning=expected_managed_versioning,
        )
        self.refresh_if_needed(description)
        factory = dataset_factory
        if factory is None:
            try:
                import lance
            except ImportError as exc:  # pragma: no cover - pylance is a base dep.
                raise NamespaceResolutionError(
                    "direct pylance namespace access requires pylance; install pylance>=7"
                ) from exc
            factory = lance.dataset
        return factory(
            namespace_client=self.namespace_client,
            table_id=list(self.table_id),
            **kwargs,
        )

    def worker_spec(self, *, logical_inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Build a secret-free worker spec for Geneva or batch workers."""
        return namespace_worker_spec(
            namespace_client_impl=self.namespace_client_impl or "rest",
            namespace_client_properties=self.namespace_client_properties,
            table_ids=[self.table_id],
            namespace_auth_ref=self.namespace_auth_ref,
            storage_auth_ref=self.storage_auth_ref,
            logical_inputs=logical_inputs,
        )


class RuntimeNamespaceAuthProvider:
    """Dynamic Lance Namespace context provider backed by runtime auth refs."""

    def __init__(self, auth_ref: str | None = None) -> None:
        self.auth_ref = auth_ref

    def provide_context(self, info: Mapping[str, str]) -> dict[str, str]:
        return namespace_auth_context(self.auth_ref)


def resolve_lake_connection(
    uri: str | Path | None = None,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
    remote_auth_ref: str | None = None,
    namespace_auth_ref: str | None = None,
    storage_auth_ref: str | None = None,
    region: str | None = None,
    host_override: str | None = None,
    client_config: Mapping[str, Any] | None = None,
    namespace_client_impl: str | None = None,
    namespace_client_properties: Mapping[str, Any] | None = None,
    namespace_client_pushdown_operations: Sequence[str] | None = None,
    remote_capabilities: Mapping[str, bool] | None = None,
) -> LakeConnectionSpec:
    """Resolve a user lake target into a typed backend connection spec.

    ``remote_capabilities`` advertises control-plane support a specific LanceDB
    Enterprise ``db://`` deployment is known to expose (``index_management``,
    ``table_versioning``, ``schema_evolution``). It is ignored for local,
    object-store, and namespace backends, which already own these operations.
    """
    value = str(uri) if uri is not None else None
    pushdowns = normalize_namespace_pushdowns(namespace_client_pushdown_operations)
    control_plane = _direct_control_plane_capabilities()

    if namespace_client_impl:
        namespace_ref = namespace_auth_ref or auth_ref
        properties = normalize_namespace_properties(
            namespace_client_properties, namespace_auth_ref=namespace_ref
        )
        options = _lancedb_storage_options_if_present(
            value or "namespace://",
            storage_options=storage_options,
            auth_ref=storage_auth_ref,
        )
        kwargs: dict[str, Any] = {
            "namespace_client_impl": namespace_client_impl,
            "namespace_client_properties": properties,
        }
        if pushdowns:
            kwargs["namespace_client_pushdown_operations"] = list(pushdowns)
        if options:
            kwargs["storage_options"] = options
        capabilities = LakeCapabilities(
            server_side_query="QueryTable" in pushdowns,
            direct_object_io=True,
            namespace_resolution=True,
            geneva_worker_specs=True,
            blob_fetch_remote=True,
            **control_plane,
        )
        return LakeConnectionSpec(
            kind="rest_namespace_lancedb"
            if namespace_client_impl == "rest"
            else "namespace_lancedb",
            uri=None,
            display_uri=namespace_display_uri(namespace_client_impl, properties),
            lancedb_connect_kwargs=kwargs,
            namespace_client_impl=namespace_client_impl,
            namespace_client_properties=properties,
            namespace_client_pushdown_operations=pushdowns,
            pylance_access=PylanceAccessSpec(
                namespace_client_impl=namespace_client_impl,
                namespace_client_properties=properties,
                namespace_auth_ref=namespace_ref,
                storage_auth_ref=storage_auth_ref,
            ),
            auth_refs={
                "remote": None,
                "namespace": namespace_ref,
                "storage": storage_auth_ref,
                "source": None,
            },
            direct_object_io_allowed=True,
            capabilities=capabilities,
        )

    if value is None:
        raise NamespaceConfigError(
            "--lake is required unless --namespace-impl/namespace_client_impl is provided"
        )

    scheme = uri_scheme(value)
    _reject_unsupported_scheme(scheme)

    if scheme == "db":
        remote_ref = remote_auth_ref or auth_ref
        kwargs = enterprise_remote_kwargs(
            remote_auth_ref=remote_ref,
            region=region,
            host_override=host_override,
            client_config=client_config,
        )
        capabilities = LakeCapabilities(
            server_side_query=True,
            direct_object_io=False,
            namespace_resolution=False,
            geneva_worker_specs=False,
            blob_fetch_remote=True,
            **_advertised_remote_control_plane(remote_capabilities),
        )
        return LakeConnectionSpec(
            kind="lancedb_remote_db",
            uri=value,
            display_uri=value,
            lancedb_connect_kwargs=kwargs,
            auth_refs={
                "remote": remote_ref,
                "namespace": None,
                "storage": storage_auth_ref,
                "source": None,
            },
            direct_object_io_allowed=False,
            capabilities=capabilities,
        )

    if is_object_store_uri(value):
        storage_ref = storage_auth_ref or auth_ref
        options = lancedb_storage_options(
            value, storage_options=storage_options, auth_ref=storage_ref
        )
        kwargs = {"storage_options": options} if options else {}
        return LakeConnectionSpec(
            kind="object_store_lancedb_oss",
            uri=value,
            display_uri=value,
            lancedb_connect_kwargs=kwargs,
            auth_refs={
                "remote": None,
                "namespace": None,
                "storage": storage_ref,
                "source": None,
            },
            direct_object_io_allowed=True,
            capabilities=LakeCapabilities(
                direct_object_io=True, blob_fetch_remote=True, **control_plane
            ),
        )

    if scheme:
        raise UnsupportedLakeSchemeError(
            f"unsupported lake URI scheme '{scheme}://'; use a local path, object-store URI, "
            "`db://...` for LanceDB Enterprise, or namespace_client_impl='rest'"
        )

    return LakeConnectionSpec(
        kind="local_path",
        uri=value,
        display_uri=value,
        lancedb_connect_kwargs={},
        auth_refs={"remote": None, "namespace": None, "storage": None, "source": None},
        direct_object_io_allowed=True,
        capabilities=LakeCapabilities(direct_object_io=True, **control_plane),
        local_path_required=True,
    )


def resolve_raw_source_connection(
    uri: str | Path,
    *,
    source_auth_ref: str | None = None,
    storage_options: Mapping[str, Any] | None = None,
) -> LakeConnectionSpec:
    """Classify a raw source path without resolving secrets into durable state."""
    value = str(uri)
    kind = "raw_object" if is_object_store_uri(value) else "local_path"
    if kind == "raw_object":
        # Validate env/config shape early; open_binary_uri does the actual IO.
        resolve_storage_options(value, storage_options=storage_options, auth_ref=source_auth_ref)
    return LakeConnectionSpec(
        kind=kind,
        uri=value,
        display_uri=value,
        auth_refs={"remote": None, "namespace": None, "storage": None, "source": source_auth_ref},
        direct_object_io_allowed=True,
        capabilities=LakeCapabilities(
            direct_object_io=True, **_direct_control_plane_capabilities()
        ),
        local_path_required=kind == "local_path",
    )


#: Control-plane capabilities a direct-object-IO backend (local path, object-store
#: OSS, or namespace-backed Lance) fully owns: index management, table version
#: checkout, and schema evolution all run against the underlying dataset.
_DIRECT_CONTROL_PLANE_CAPABILITIES = ("index_management", "table_versioning", "schema_evolution")


def _direct_control_plane_capabilities() -> dict[str, bool]:
    return {name: True for name in _DIRECT_CONTROL_PLANE_CAPABILITIES}


def _advertised_remote_control_plane(
    remote_capabilities: Mapping[str, bool] | None,
) -> dict[str, bool]:
    """Resolve db:// control-plane flags from an operator's advertised support.

    Unadvertised flags stay ``False`` so the backlog-0128 capability gates report
    guidance rather than assume a remote deployment exposes index/version/schema
    operations through the plain ``db://`` Table surface.
    """
    advertised = {name: False for name in _DIRECT_CONTROL_PLANE_CAPABILITIES}
    for key, value in dict(remote_capabilities or {}).items():
        name = str(key)
        if name not in advertised:
            supported = ", ".join(_DIRECT_CONTROL_PLANE_CAPABILITIES)
            raise NamespaceConfigError(
                f"unknown remote capability {name!r}; advertise one of: {supported}"
            )
        advertised[name] = bool(value)
    return advertised


def enterprise_remote_kwargs(
    *,
    remote_auth_ref: str | None = None,
    region: str | None = None,
    host_override: str | None = None,
    client_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve runtime-only LanceDB Enterprise kwargs."""
    env_options: dict[str, Any] = {}
    normalized = _normalize_auth_ref(remote_auth_ref) if remote_auth_ref else None
    if normalized:
        env_options.update(_load_json_env(f"LANCEDB_ROBOTICS_AUTH_{normalized}_REMOTE_JSON"))
        env_options.update(
            _load_json_env(f"LANCEDB_ROBOTICS_AUTH_{normalized}_REMOTE_OPTIONS_JSON")
        )

    api_key = (
        env_options.get("api_key")
        or _load_env_if_present(normalized, "REMOTE_API_KEY")
        or os.environ.get("LANCEDB_API_KEY")
    )
    if not api_key:
        hint = (
            f"LANCEDB_ROBOTICS_AUTH_{normalized}_REMOTE_API_KEY"
            if normalized
            else "LANCEDB_API_KEY"
        )
        raise MissingEnterpriseCredentialsError(
            f"cannot connect to LanceDB Enterprise db:// URI: set {hint} or pass a "
            "remote_auth_ref that resolves an api_key"
        )

    resolved: dict[str, Any] = {
        "api_key": api_key,
        "region": region or env_options.get("region") or "us-east-1",
    }
    selected_host = host_override or env_options.get("host_override")
    if selected_host:
        resolved["host_override"] = selected_host
    selected_client_config = client_config or env_options.get("client_config")
    if selected_client_config:
        resolved["client_config"] = dict(selected_client_config)
    return resolved


def normalize_namespace_pushdowns(
    operations: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate namespace pushdown operation names."""
    normalized: list[str] = []
    for operation in operations or ():
        value = operation.strip()
        if not value:
            continue
        if value not in SUPPORTED_NAMESPACE_PUSHDOWNS:
            supported = ", ".join(sorted(SUPPORTED_NAMESPACE_PUSHDOWNS))
            raise NamespaceConfigError(
                f"unsupported namespace pushdown {value!r}; supported: {supported}"
            )
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def normalize_namespace_properties(
    properties: Mapping[str, Any] | None,
    *,
    namespace_auth_ref: str | None = None,
) -> dict[str, str]:
    """Normalize namespace properties and add runtime auth-provider refs."""
    normalized = {str(key): str(value) for key, value in (properties or {}).items()}
    if namespace_auth_ref:
        impl_key = "dynamic_context_provider.impl"
        if impl_key in normalized and normalized[impl_key] != _provider_class_path():
            raise NamespaceConfigError(
                "namespace_auth_ref cannot be combined with a custom "
                "dynamic_context_provider.impl"
            )
        normalized[impl_key] = _provider_class_path()
        normalized["dynamic_context_provider.auth_ref"] = namespace_auth_ref
    return normalized


def namespace_properties_from_options(
    *,
    uri: str | None = None,
    database: str | None = None,
    database_prefix: str | None = None,
    delimiter: str | None = None,
    properties: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build REST namespace properties from SDK/CLI-friendly option names."""
    result = {str(key): str(value) for key, value in (properties or {}).items()}
    if uri:
        result["uri"] = uri
    if delimiter:
        result["delimiter"] = delimiter
    if database:
        result["header.x-lancedb-database"] = database
    if database_prefix:
        result["header.x-lancedb-database-prefix"] = database_prefix
    return result


def namespace_display_uri(impl: str, properties: Mapping[str, str]) -> str:
    endpoint = properties.get("uri") or properties.get("root") or "configured"
    return f"namespace://{impl}/{endpoint}"


def namespace_auth_context(auth_ref: str | None) -> dict[str, str]:
    """Resolve dynamic namespace headers/context for one operation."""
    if not auth_ref:
        return {}
    normalized = _normalize_auth_ref(auth_ref)
    context = {
        str(key): str(value)
        for key, value in _load_json_env(
            f"LANCEDB_ROBOTICS_AUTH_{normalized}_NAMESPACE_CONTEXT_JSON"
        ).items()
    }
    headers = _load_json_env(f"LANCEDB_ROBOTICS_AUTH_{normalized}_NAMESPACE_HEADERS_JSON")
    for key, value in headers.items():
        name = str(key)
        context[name if name.startswith("headers.") else f"headers.{name}"] = str(value)
    bearer = os.environ.get(f"LANCEDB_ROBOTICS_AUTH_{normalized}_NAMESPACE_BEARER_TOKEN")
    if bearer:
        context["headers.Authorization"] = f"Bearer {bearer}"
    api_key = os.environ.get(f"LANCEDB_ROBOTICS_AUTH_{normalized}_NAMESPACE_API_KEY")
    if api_key and "headers.Authorization" not in context:
        context["headers.Authorization"] = f"Bearer {api_key}"
    return context


def create_namespace_client(
    *,
    impl: str = "rest",
    properties: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
    factory: Any | None = None,
) -> Any:
    """Create a Lance Namespace client from normalized robotics config."""
    normalized = normalize_namespace_properties(properties, namespace_auth_ref=auth_ref)
    if factory is not None:
        return factory(impl, normalized)
    try:
        if impl == "rest":
            from lance.namespace import RestNamespace

            return RestNamespace(**normalized)
        if impl in {"dir", "directory"}:
            from lance.namespace import DirectoryNamespace

            return DirectoryNamespace(**normalized)
        import lance_namespace

        return lance_namespace.connect(impl, normalized)
    except Exception as exc:
        raise NamespaceResolutionError(
            f"cannot create Lance namespace client {impl!r}: {exc}"
        ) from exc


def pylance_namespace_access(
    *,
    namespace_client: Any | None = None,
    table_id: Sequence[str],
    namespace_client_impl: str = "rest",
    namespace_client_properties: Mapping[str, Any] | None = None,
    namespace_auth_ref: str | None = None,
    storage_auth_ref: str | None = None,
) -> PylanceNamespaceAccess:
    """Build direct pylance namespace access for one table id."""
    properties = normalize_namespace_properties(
        namespace_client_properties, namespace_auth_ref=namespace_auth_ref
    )
    client = namespace_client or create_namespace_client(
        impl=namespace_client_impl, properties=properties
    )
    return PylanceNamespaceAccess(
        namespace_client=client,
        table_id=tuple(str(part) for part in table_id),
        namespace_client_impl=namespace_client_impl,
        namespace_client_properties=properties,
        namespace_auth_ref=namespace_auth_ref,
        storage_auth_ref=storage_auth_ref,
    )


def lance_dataset(
    *,
    namespace_client: Any,
    table_id: Sequence[str],
    dataset_factory: Any | None = None,
    expected_managed_versioning: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Open a pylance dataset through namespace direct data-plane access."""
    access = PylanceNamespaceAccess(
        namespace_client=namespace_client,
        table_id=tuple(str(part) for part in table_id),
    )
    return access.open_dataset(
        dataset_factory=dataset_factory,
        expected_managed_versioning=expected_managed_versioning,
        **kwargs,
    )


def make_describe_table_request(
    table_id: Sequence[str],
    *,
    version: int | str | None = None,
    vend_credentials: bool = True,
) -> Any:
    """Create a version-tolerant Lance Namespace ``DescribeTableRequest``."""
    values = {
        "id": list(table_id),
        "version": version,
        "vend_credentials": vend_credentials,
    }
    for import_path in (
        ("lance.namespace", "DescribeTableRequest"),
        ("lance_namespace_urllib3_client.models.describe_table_request", "DescribeTableRequest"),
    ):
        try:
            module = __import__(import_path[0], fromlist=[import_path[1]])
            cls = getattr(module, import_path[1])
        except (ImportError, AttributeError):
            continue
        for kwargs in (
            values,
            {"id": list(table_id), "version": version, "vendCredentials": vend_credentials},
            {"id": list(table_id), "version": version},
            {"id": list(table_id)},
        ):
            try:
                return cls(**kwargs)
            except (TypeError, ValueError):
                continue
    return SimpleNamespace(**values)


def namespace_table_description(
    table_id: Sequence[str],
    response: Any,
) -> NamespaceTableDescription:
    """Parse common Python/JSON forms of ``DescribeTableResponse``."""
    storage_options = _mapping_value(response, "storage_options", "storageOptions")
    storage_options = {str(key): str(value) for key, value in storage_options.items()}
    expires = _response_value(response, "expires_at_millis", "expiresAtMillis")
    if expires is None:
        expires = storage_options.get("expires_at_millis") or storage_options.get(
            "expiresAtMillis"
        )
    expires_at_millis = int(expires) if expires is not None else None
    managed = _response_value(response, "managed_versioning", "managedVersioning")
    return NamespaceTableDescription(
        table_id=tuple(str(part) for part in table_id),
        location=_response_value(response, "location", "table_uri", "tableUri"),
        storage_options=storage_options,
        expires_at_millis=expires_at_millis,
        managed_versioning=bool(managed),
        raw_response=response,
    )


def plan_namespace_write(
    access: PylanceNamespaceAccess,
    *,
    supports_managed_versioning: bool,
) -> NamespaceTableDescription:
    """Validate that a write path can honor namespace managed versioning."""
    description = access.describe(vend_credentials=True)
    if description.managed_versioning and not supports_managed_versioning:
        raise ManagedVersioningMismatch(
            "namespace reports managed_versioning=true; writes must use namespace "
            "table-version APIs or an implementation that explicitly supports them"
        )
    return description


def apply_worker_namespace_overrides(properties: Mapping[str, Any]) -> dict[str, str]:
    """Apply ``_lancedb_worker_.`` namespace property overrides."""
    base: dict[str, str] = {}
    overrides: dict[str, str] = {}
    for key, value in properties.items():
        name = str(key)
        if name.startswith(WORKER_NAMESPACE_PROPERTY_PREFIX):
            overrides[name[len(WORKER_NAMESPACE_PROPERTY_PREFIX) :]] = str(value)
        else:
            base[name] = str(value)
    base.update(overrides)
    return base


def namespace_worker_spec(
    *,
    namespace_client_impl: str,
    namespace_client_properties: Mapping[str, Any],
    table_ids: Sequence[Sequence[str]],
    namespace_auth_ref: str | None = None,
    storage_auth_ref: str | None = None,
    source_auth_ref: str | None = None,
    logical_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a secret-free direct data-plane worker spec."""
    worker_properties = apply_worker_namespace_overrides(namespace_client_properties)
    return {
        "kind": "pylance_direct_namespace",
        "namespace": {
            "impl": namespace_client_impl,
            "properties": safe_namespace_properties(worker_properties),
        },
        "table_ids": [[str(part) for part in table_id] for table_id in table_ids],
        "auth_refs": {
            key: value
            for key, value in {
                "namespace": namespace_auth_ref,
                "storage": storage_auth_ref,
                "source": source_auth_ref,
            }.items()
            if value
        },
        "logical_inputs": dict(logical_inputs or {}),
        "capabilities": {
            "direct_object_io": True,
            "namespace_resolution": True,
            "geneva_worker_specs": True,
        },
    }


def safe_namespace_properties(properties: Mapping[str, Any]) -> dict[str, str]:
    """Drop obvious secret-bearing namespace properties for diagnostics/specs."""
    safe: dict[str, str] = {}
    for key, value in properties.items():
        name = str(key)
        lowered = name.lower()
        if lowered.startswith("header.authorization"):
            continue
        if any(token in lowered for token in ("password", "secret", "token", "api_key")):
            continue
        if lowered.startswith("credential_vendor."):
            continue
        safe[name] = str(value)
    return safe


def _lancedb_storage_options_if_present(
    uri: str,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, str]:
    if storage_options or auth_ref:
        return lancedb_storage_options(uri, storage_options=storage_options, auth_ref=auth_ref)
    return {}


def _reject_unsupported_scheme(scheme: str) -> None:
    if scheme in UNSUPPORTED_LANCEDB_SCHEMES:
        if scheme == "lancedb":
            raise UnsupportedLakeSchemeError(
                "unsupported lake URI scheme 'lancedb://'; use LanceDB Enterprise "
                "`db://...` or namespace_client_impl='rest'"
            )
        raise UnsupportedLakeSchemeError(
            "unsupported lake URI scheme 'phalanx://'; model the query node as a "
            "Lance REST Namespace using namespace_client_impl='rest'"
        )


def _load_json_env(name: str) -> dict[str, Any]:
    raw = os.environ.get(name)
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StorageConfigError(f"{name} must contain a JSON object: {exc}") from exc
    if not isinstance(decoded, dict):
        raise StorageConfigError(f"{name} must contain a JSON object")
    return {str(key): value for key, value in decoded.items()}


def _load_env_if_present(normalized_ref: str | None, suffix: str) -> str | None:
    if not normalized_ref:
        return None
    return os.environ.get(f"LANCEDB_ROBOTICS_AUTH_{normalized_ref}_{suffix}")


def _normalize_auth_ref(auth_ref: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", auth_ref.strip()).strip("_")
    return normalized.upper()


def _provider_class_path() -> str:
    return "lancedb_robotics.connections.RuntimeNamespaceAuthProvider"


def _mapping_value(response: Any, *names: str) -> dict[str, Any]:
    value = _response_value(response, *names)
    return dict(value or {})


def _response_value(response: Any, *names: str) -> Any:
    if isinstance(response, Mapping):
        for name in names:
            if name in response:
                return response[name]
    for name in names:
        if hasattr(response, name):
            return getattr(response, name)
    if hasattr(response, "model_dump"):
        dumped = response.model_dump(by_alias=False)
        for name in names:
            if name in dumped:
                return dumped[name]
        dumped = response.model_dump(by_alias=True)
        for name in names:
            if name in dumped:
                return dumped[name]
    if hasattr(response, "to_dict"):
        dumped = response.to_dict()
        for name in names:
            if name in dumped:
                return dumped[name]
    return None


def _now_millis() -> int:
    return int(time.time() * 1000)
