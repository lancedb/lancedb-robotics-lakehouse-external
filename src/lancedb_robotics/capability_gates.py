"""Backend capability gates for direct-Lance and control-plane invocations (0128).

Follow-up to the 0076 invocation audit
(``docs/product/lancedb-invocation-remote-compatibility-audit.md``). Several
operation families the SDK relies on are not part of the plain LanceDB Enterprise
``db://`` Table/query surface:

- ``direct_lance`` / ``blob`` / ``maintenance`` drop to the underlying
  ``lance.LanceDataset`` (``Table.to_lance()``, ``take_blobs``, ``get_fragments``,
  ``compact_files``, ``cleanup_old_versions``), which needs *direct object IO*;
- ``index`` / ``versioning`` / ``schema`` mutate control-plane state
  (``create_*_index``, ``checkout`` / current-version reads, ``add_columns`` /
  ``drop_columns``) that a remote deployment may or may not expose through the
  ``db://`` Table API.

Reaching those paths after opening a ``db://`` lake used to raise a late, opaque
backend error. This module gates them *before* the underlying call and raises a
:class:`BackendCapabilityError` naming the operation family, the current backend,
the required capability, and the recommended fallback
(``object_store_lancedb_oss`` or ``pylance_direct_namespace``). Local paths,
object-store OSS lakes, and namespace-backed Lance advertise the matching
capability, so their behavior is unchanged.

The gate is capability-driven, not kind-driven: a backend that positively
advertises the capability (see ``resolve_lake_connection(..., remote_capabilities=...)``)
passes, and any future backend that lacks it is guarded automatically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cost at runtime.
    from lancedb_robotics.connections import LakeConnectionSpec

#: Operation families from the 0076 audit that are gated on remote backends.
DIRECT_LANCE = "direct_lance"
BLOB = "blob"
MAINTENANCE = "maintenance"
INDEX = "index"
VERSIONING = "versioning"
SCHEMA = "schema"

#: Fallback recommendations shared by the direct-object-IO families.
_DIRECT_IO_FALLBACKS = ("object_store_lancedb_oss", "pylance_direct_namespace")


@dataclass(frozen=True)
class OperationFamilyGate:
    """How one operation family maps to a required backend capability."""

    family: str
    capability: str  # attribute name on :class:`LakeCapabilities`
    label: str  # human-facing operation description
    fallbacks: tuple[str, ...]


#: Registry: audit operation family -> the capability that must be advertised.
OPERATION_GATES: dict[str, OperationFamilyGate] = {
    DIRECT_LANCE: OperationFamilyGate(
        family=DIRECT_LANCE,
        capability="direct_object_io",
        label="direct LanceDataset access (Table.to_lance / lance.dataset)",
        fallbacks=_DIRECT_IO_FALLBACKS,
    ),
    BLOB: OperationFamilyGate(
        family=BLOB,
        capability="direct_object_io",
        label="blob hydration (take_blobs via Table.to_lance)",
        fallbacks=_DIRECT_IO_FALLBACKS,
    ),
    MAINTENANCE: OperationFamilyGate(
        family=MAINTENANCE,
        capability="direct_object_io",
        label="lake maintenance (compaction, version cleanup, index refresh)",
        fallbacks=_DIRECT_IO_FALLBACKS,
    ),
    INDEX: OperationFamilyGate(
        family=INDEX,
        capability="index_management",
        label="index creation/refresh (vector, scalar, or full-text)",
        fallbacks=_DIRECT_IO_FALLBACKS,
    ),
    VERSIONING: OperationFamilyGate(
        family=VERSIONING,
        capability="table_versioning",
        label="table version checkout / current-version read",
        fallbacks=_DIRECT_IO_FALLBACKS,
    ),
    SCHEMA: OperationFamilyGate(
        family=SCHEMA,
        capability="schema_evolution",
        label="schema mutation (add_columns / drop_columns / alter_columns)",
        fallbacks=_DIRECT_IO_FALLBACKS,
    ),
}


class BackendCapabilityError(RuntimeError):
    """Raised when a backend cannot support a gated operation family.

    Carries the structured pieces (operation, family, backend, required
    capability, fallbacks) so callers can re-render guidance; ``str(error)`` is
    the actionable one-line message.
    """

    def __init__(
        self,
        *,
        operation: str,
        family: str,
        backend_kind: str,
        required_capability: str,
        fallbacks: Sequence[str],
        detail: str | None = None,
    ) -> None:
        self.operation = operation
        self.family = family
        self.backend_kind = backend_kind
        self.required_capability = required_capability
        self.fallbacks = tuple(fallbacks)
        self.detail = detail
        fallback_text = " or ".join(self.fallbacks) if self.fallbacks else "a supporting backend"
        message = (
            f"{operation} (operation family {family!r}) is not supported by backend "
            f"{backend_kind!r}: it requires the {required_capability!r} capability, which "
            f"this backend does not advertise. Use {fallback_text}, or advertise the "
            f"capability if the deployment supports it "
            f"(resolve_lake_connection(..., remote_capabilities={{{required_capability!r}: True}}))."
        )
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


def require_backend_capability(
    connection_spec: LakeConnectionSpec | None,
    family: str,
    *,
    operation: str | None = None,
    detail: str | None = None,
) -> None:
    """Raise :class:`BackendCapabilityError` if ``family`` is unsupported here.

    ``connection_spec`` is ``lake.connection_spec``. A ``None`` spec (a lake
    constructed without connection resolution, e.g. a bare ``Lake(db, uri)``) is
    treated as unclassified and passes through unchanged, preserving legacy and
    in-process behavior. Backends that advertise the capability pass silently.
    """
    if connection_spec is None:
        return
    gate = OPERATION_GATES.get(family)
    if gate is None:
        raise KeyError(f"unknown gated operation family {family!r}")
    capabilities = connection_spec.capabilities
    if capabilities is not None and bool(getattr(capabilities, gate.capability, False)):
        return
    raise BackendCapabilityError(
        operation=operation or gate.label,
        family=gate.family,
        backend_kind=connection_spec.kind,
        required_capability=gate.capability,
        fallbacks=gate.fallbacks,
        detail=detail,
    )


def require_lake_capability(lake: Any, family: str, *, operation: str | None = None) -> None:
    """Gate ``family`` against ``lake.connection_spec`` (lake-object convenience).

    A ``lake`` without a ``connection_spec`` attribute (e.g. a lightweight test
    double or a ``Lake`` built without connection resolution) is treated as
    unclassified and passes through, so the gate never breaks in-process callers
    that never resolved a backend.
    """
    require_backend_capability(
        getattr(lake, "connection_spec", None), family, operation=operation
    )


def backend_supports(
    connection_spec: LakeConnectionSpec | None,
    family: str,
) -> bool:
    """Return whether ``family`` runs on this backend (unclassified => True)."""
    if connection_spec is None:
        return True
    gate = OPERATION_GATES[family]
    capabilities = connection_spec.capabilities
    return capabilities is not None and bool(getattr(capabilities, gate.capability, False))


def lake_capability_reason(lake: Any, family: str) -> str | None:
    """Guidance string when ``family`` is unsupported here, else ``None``.

    For operations that degrade to a no-op rather than fail (an index build is a
    pure optimization: queries still run a full scan without it), a caller uses
    this to record a ``skipped`` outcome carrying the capability guidance instead
    of raising. Returns ``None`` when the backend advertises the capability or is
    unclassified, so local/object-store/namespace lakes behave exactly as before.
    """
    spec = getattr(lake, "connection_spec", None)
    if backend_supports(spec, family):
        return None
    gate = OPERATION_GATES[family]
    fallback_text = " or ".join(gate.fallbacks)
    return (
        f"backend {spec.kind!r} does not advertise the {gate.capability!r} capability "
        f"required for {gate.label}; use {fallback_text}, or advertise it via "
        f"remote_capabilities={{{gate.capability!r}: True}}"
    )


def gated_to_lance(
    handle: Any,
    connection_spec: LakeConnectionSpec | None,
    *,
    family: str = DIRECT_LANCE,
    operation: str | None = None,
) -> Any:
    """Gate a direct-Lance drop, then return ``handle.to_lance()``.

    ``handle`` is a lancedb ``Table`` (or an already-opened ``lance.LanceDataset``).
    The capability check happens *before* ``to_lance()`` so a ``db://`` lake fails
    with actionable guidance instead of an opaque late backend error. A bare
    dataset (no ``to_lance``) is returned as-is after the gate.
    """
    require_backend_capability(connection_spec, family, operation=operation)
    return handle.to_lance() if hasattr(handle, "to_lance") else handle
