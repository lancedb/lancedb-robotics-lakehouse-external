"""Shared direct-pylance execution adapter (backlog 0129).

Follow-up to 0036 (the connection resolver, which produces
:class:`~lancedb_robotics.connections.PylanceAccessSpec`) and 0128 (the
capability gates over the 0076 invocation audit). Several workflow modules drop
to the underlying ``lance.LanceDataset`` through ``Table.to_lance()`` to read
blob bytes, hydrate media, or run direct scans. On a plain ``db://`` remote that
drop now fails fast (0128). On a *namespace-backed* lake there is a supported
path -- ``lance.dataset(namespace_client=..., table_id=...)`` with credentials
vended by ``describe_table`` -- but nothing tied it to a logical lakehouse table
name or refreshed the vended credentials before IO.

This module is that missing seam. Given a resolved
:class:`~lancedb_robotics.connections.LakeConnectionSpec` that carries a
``pylance_access`` (a namespace backend) and a logical table name, it:

* derives the namespace ``table_id`` (see :func:`namespace_table_id`),
* builds a live namespace client from the *serializable*
  :class:`~lancedb_robotics.connections.PylanceAccessSpec` (secrets stay in
  auth-refs, resolved at runtime), and
* opens the dataset through
  :meth:`~lancedb_robotics.connections.PylanceNamespaceAccess.open_dataset`,
  which calls ``describe_table`` and refreshes near-expiry vended credentials
  *before* the first byte of direct IO.

For write paths (additive column mutation, maintenance that creates new table
versions) it exposes :func:`require_namespace_write_supported`, which refuses --
loudly, via :class:`~lancedb_robotics.connections.ManagedVersioningMismatch` --
to write directly against a namespace that reports managed versioning, rather
than silently forking the version history (SKILLS.md: converge or fail loudly,
never silently corrupt).

Local and object-store lakes have ``pylance_access is None`` and are untouched:
every entry point here is a no-op / pass-through for them, so their behavior is
byte-identical to before this module existed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from lancedb_robotics.connections import (
    DATA_PLANE_LOCAL as LOCAL,
)
from lancedb_robotics.connections import (
    DATA_PLANE_NAMESPACE_DIRECT as NAMESPACE_DIRECT,
)
from lancedb_robotics.connections import (
    DATA_PLANE_OBJECT_STORE as OBJECT_STORE,
)
from lancedb_robotics.connections import (
    DATA_PLANE_REMOTE_DB as REMOTE_DB,
)
from lancedb_robotics.connections import (
    DATA_PLANE_UNCLASSIFIED as UNCLASSIFIED,
)
from lancedb_robotics.connections import (
    PylanceNamespaceAccess,
    pylance_namespace_access,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cost at runtime.
    from lancedb_robotics.connections import LakeConnectionSpec

#: Data-plane labels recorded in reports/manifests so a reader can tell *how* a
#: read/write reached storage. Re-exported from :mod:`connections` (single source
#: of truth on :attr:`LakeConnectionSpec.data_plane`); provenance, never secrets.
__all__ = [
    "LOCAL",
    "OBJECT_STORE",
    "NAMESPACE_DIRECT",
    "REMOTE_DB",
    "UNCLASSIFIED",
    "NamespaceAccessFactory",
    "has_pylance_access",
    "data_plane",
    "namespace_table_id",
    "namespace_access",
    "open_direct_dataset",
    "require_namespace_write_supported",
    "data_plane_provenance",
]

#: A factory that returns a fully-built :class:`PylanceNamespaceAccess`. Tests
#: inject one to avoid contacting a real namespace server; production leaves it
#: ``None`` and the access is built from the spec.
NamespaceAccessFactory = Callable[["LakeConnectionSpec", str], PylanceNamespaceAccess]

_DEFAULT_DELIMITER = "."
_DATABASE_PREFIX_PROPERTY = "header.x-lancedb-database-prefix"
_DELIMITER_PROPERTY = "delimiter"


def has_pylance_access(connection_spec: LakeConnectionSpec | None) -> bool:
    """Whether ``connection_spec`` resolves to a namespace direct data-plane.

    ``True`` only for a namespace-backed lake (0036 ``pylance_access`` present).
    ``None`` (an unclassified in-process lake) and local/object-store/``db://``
    backends all return ``False`` -- they do not use the namespace-direct route.
    """
    return connection_spec is not None and connection_spec.pylance_access is not None


def data_plane(connection_spec: LakeConnectionSpec | None) -> str:
    """Classify the data-plane a resolved backend uses.

    Returns one of :data:`LOCAL`, :data:`OBJECT_STORE`, :data:`NAMESPACE_DIRECT`,
    :data:`REMOTE_DB`, or :data:`UNCLASSIFIED` (a ``None`` spec / bare in-process
    lake). Delegates to :attr:`LakeConnectionSpec.data_plane` so the labelling is
    single-sourced with ``safe_summary()``.
    """
    if connection_spec is None:
        return UNCLASSIFIED
    return connection_spec.data_plane


def namespace_table_id(
    connection_spec: LakeConnectionSpec,
    table_name: str,
) -> tuple[str, ...]:
    """Derive the namespace ``table_id`` for a logical lakehouse table.

    The default convention prepends the namespace database prefix (the
    ``header.x-lancedb-database-prefix`` property, split on the configured
    ``delimiter``) to ``table_name``. A namespace without a prefix maps a table
    to a bare ``(table_name,)``. This is a *representative* mapping matched to
    the flat canonical-table layout this lakehouse writes today; a deployment
    with a multi-level catalog can pass an explicit ``table_id`` to the adapter
    entry points instead (tracked as a follow-up).
    """
    properties = _pylance_properties(connection_spec)
    delimiter = properties.get(_DELIMITER_PROPERTY) or _DEFAULT_DELIMITER
    prefix = properties.get(_DATABASE_PREFIX_PROPERTY, "")
    parts = [part for part in prefix.split(delimiter) if part] if prefix else []
    parts.append(str(table_name))
    return tuple(parts)


def namespace_access(
    connection_spec: LakeConnectionSpec,
    table_name: str,
    *,
    table_id: tuple[str, ...] | None = None,
    namespace_client: Any | None = None,
    namespace_access_factory: NamespaceAccessFactory | None = None,
) -> PylanceNamespaceAccess:
    """Build :class:`PylanceNamespaceAccess` for one logical table.

    Requires a namespace-backed ``connection_spec`` (``pylance_access`` set);
    call :func:`has_pylance_access` first for backends that may be local. A
    ``namespace_access_factory`` (tests) short-circuits construction; otherwise
    a live client is built from the serializable ``pylance_access`` -- an
    injected ``namespace_client`` is reused, else one is created from the impl +
    auth-ref-bearing properties (secrets resolved at runtime, never persisted).
    """
    if namespace_access_factory is not None:
        return namespace_access_factory(connection_spec, table_name)
    access_spec = connection_spec.pylance_access
    if access_spec is None:
        raise ValueError(
            "namespace_access requires a namespace-backed connection_spec "
            "(pylance_access is None); guard with has_pylance_access() first"
        )
    resolved_id = table_id or namespace_table_id(connection_spec, table_name)
    return pylance_namespace_access(
        namespace_client=namespace_client,
        table_id=resolved_id,
        namespace_client_impl=access_spec.namespace_client_impl,
        namespace_client_properties=access_spec.namespace_client_properties,
        namespace_auth_ref=access_spec.namespace_auth_ref,
        storage_auth_ref=access_spec.storage_auth_ref,
    )


def open_direct_dataset(
    connection_spec: LakeConnectionSpec,
    table_name: str,
    *,
    table_id: tuple[str, ...] | None = None,
    namespace_client: Any | None = None,
    namespace_access_factory: NamespaceAccessFactory | None = None,
    dataset_factory: Any | None = None,
    expected_managed_versioning: bool | None = None,
    **dataset_kwargs: Any,
) -> Any:
    """Open a namespace-direct ``lance`` dataset for ``table_name``.

    Delegates to
    :meth:`~lancedb_robotics.connections.PylanceNamespaceAccess.open_dataset`,
    which calls ``describe_table`` and refreshes near-expiry vended credentials
    *before* opening ``lance.dataset(namespace_client=..., table_id=...)``
    (acceptance: "credential expiry triggers a refresh before direct IO"). This
    is the read seam blob/media hydration and direct scans route through when the
    lake is namespace-backed, so they never depend on ``Table.to_lance()`` from a
    remote table.
    """
    access = namespace_access(
        connection_spec,
        table_name,
        table_id=table_id,
        namespace_client=namespace_client,
        namespace_access_factory=namespace_access_factory,
    )
    return access.open_dataset(
        dataset_factory=dataset_factory,
        expected_managed_versioning=expected_managed_versioning,
        **dataset_kwargs,
    )


def require_namespace_write_supported(
    connection_spec: LakeConnectionSpec | None,
    table_name: str,
    *,
    supports_managed_versioning: bool = False,
    table_id: tuple[str, ...] | None = None,
    namespace_client: Any | None = None,
    namespace_access_factory: NamespaceAccessFactory | None = None,
) -> None:
    """Validate a direct write against namespace-managed versioning.

    A no-op for local/object-store/``db://``/unclassified backends
    (``pylance_access is None``) so their write paths are unchanged. For a
    namespace-backed lake it calls ``describe_table`` and raises
    :class:`~lancedb_robotics.connections.ManagedVersioningMismatch` when the
    namespace reports managed versioning and this write path does not support it
    -- refusing loudly instead of forking the managed version history behind the
    server's back. Callers that create new table versions (additive column
    mutation, maintenance/compaction) call this *before* the write.
    """
    if not has_pylance_access(connection_spec):
        return
    assert connection_spec is not None  # narrowed by has_pylance_access
    access = namespace_access(
        connection_spec,
        table_name,
        table_id=table_id,
        namespace_client=namespace_client,
        namespace_access_factory=namespace_access_factory,
    )
    from lancedb_robotics.connections import plan_namespace_write

    plan_namespace_write(access, supports_managed_versioning=supports_managed_versioning)


def data_plane_provenance(connection_spec: LakeConnectionSpec | None) -> dict[str, Any]:
    """Secret-free provenance describing how a read/write reached storage.

    Suitable to embed in reports, manifests, and task records: it names the
    data-plane, the backend kind, the namespace impl (``rest``/``dir`` -- not a
    secret), and boolean facts about credential vending and managed versioning.
    Resolved credentials, tokens, and vended session secrets are never included
    (SKILLS.md: secrets never live in the lake / in a persisted row).
    """
    plane = data_plane(connection_spec)
    provenance: dict[str, Any] = {
        "data_plane": plane,
        "backend_kind": connection_spec.kind if connection_spec is not None else None,
        "credential_vending": plane == NAMESPACE_DIRECT,
    }
    if connection_spec is not None:
        provenance["namespace_impl"] = connection_spec.namespace_client_impl
        provenance["managed_versioning"] = connection_spec.managed_versioning
        capabilities = connection_spec.capabilities
        provenance["direct_object_io"] = bool(getattr(capabilities, "direct_object_io", False))
    return provenance


def _pylance_properties(connection_spec: LakeConnectionSpec) -> dict[str, str]:
    access_spec = connection_spec.pylance_access
    if access_spec is not None:
        return dict(access_spec.namespace_client_properties)
    return dict(connection_spec.namespace_client_properties)
