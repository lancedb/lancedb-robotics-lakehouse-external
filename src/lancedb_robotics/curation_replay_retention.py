"""Curation replay retention protection and version-read conformance (backlog 0143).

Backlog 0082 makes as-of curation replay depend on the ``table_versions`` a
``dataset_snapshots`` row pins for ``curation_views``,
``curation_view_membership_chunks``, and ``curation_memberships``. That gives
correct audit semantics, but a production lake also needs the operational
guarantee that those pinned versions stay *readable* after compaction, version
cleanup, remote-backend migration, and namespace handoff -- and a way for a
researcher or safety reviewer to see a precise replay-readiness status before
they trust a snapshot for training or an investigation.

This module is that readiness layer on top of the existing machinery:

- Protection itself is already generic: ``maintain_lake`` tags every version a
  live ``dataset_snapshots.table_versions`` entry pins (via
  ``snapshot_retention_pin_details``), and the three curation tables are
  recorded in every curated snapshot (``curate._SOURCE_TABLES``). This module
  does not re-implement that; it *verifies* it and reports drift.
- :func:`curation_replay_conformance` classifies whether as-of curation replay
  is ``supported`` / ``capability-gated`` / ``unavailable`` on the resolved
  backend, reusing the 0128 capability gates (``VERSIONING`` -> the
  ``table_versioning`` capability) plus the 0129 ``namespace_managed_versioning``
  signal.
- :func:`curation_replay_readiness` enumerates each snapshot-pinned curation
  version and reports whether it is ``protected`` (tagged / current), present
  but ``unprotected``, ``pruned`` (gone from disk), or ``unreadable`` (present
  but a checkout fails), with an actionable suggested action.

The read is bounded: ``dataset_snapshots`` is streamed with a light projection
(no blob columns), and each pinned table's version/tag metadata is inspected
once per table (bounded by version count, not row count) rather than per pin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lancedb_robotics.capability_gates import (
    DIRECT_LANCE,
    VERSIONING,
    backend_supports,
)

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from lancedb_robotics.lake import Lake

#: Curation tables whose snapshot-pinned versions as-of replay depends on (0082).
CURATION_REPLAY_TABLES: tuple[str, ...] = (
    "curation_views",
    "curation_view_membership_chunks",
    "curation_memberships",
)

READINESS_SCHEMA_VERSION = "lancedb-robotics/curation-replay-readiness/v1"

#: Backend conformance classes for as-of curation replay.
REPLAY_SUPPORTED = "supported"
REPLAY_CAPABILITY_GATED = "capability-gated"
REPLAY_UNAVAILABLE = "unavailable"

#: Per-pin protection/readability states.
PIN_PROTECTED = "protected"
PIN_UNPROTECTED = "unprotected"
PIN_PRUNED = "pruned"
PIN_UNREADABLE = "unreadable"
PIN_BACKEND_GATED = "backend-gated"

#: Overall readiness verdicts.
READY = "ready"
AT_RISK = "at-risk"
BACKEND_GATED = "backend-gated"

_CAPABILITY = "table_versioning"
#: `created_at` is required: the scoped path picks the latest row per name by
#: `(created_at, dataset_id)` to match `curate._latest_snapshot_row`; if it is not
#: projected, the streamed path sees `created_at=None` and silently degrades to
#: max-by-dataset_id (a content hash, not time-ordered).
_SNAPSHOT_COLUMNS = ("name", "dataset_id", "table_versions", "created_at")
_SNAPSHOT_SCAN_BATCH = 4096
#: Mirrors ``maintenance._PIN_TAG_PREFIX`` (the managed snapshot-pin tag prefix);
#: kept as a local constant to avoid importing the maintenance module here.
#: ``test_pin_tag_prefix_matches_maintenance`` fails if the two ever drift.
_PIN_TAG_PREFIX = "lbr-snapshot-pin-v"


@dataclass(frozen=True)
class ReplayBackendConformance:
    """Whether as-of curation replay works on the resolved backend."""

    backend_kind: str
    data_plane: str
    status: str
    capability: str
    advertised: bool
    namespace_managed_versioning: bool
    fallbacks: tuple[str, ...]
    reason: str | None
    suggested_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "data_plane": self.data_plane,
            "status": self.status,
            "capability": self.capability,
            "advertised": self.advertised,
            "namespace_managed_versioning": self.namespace_managed_versioning,
            "fallbacks": list(self.fallbacks),
            "reason": self.reason,
            "suggested_action": self.suggested_action,
        }


@dataclass(frozen=True)
class CurationReplayPinStatus:
    """Protection/readability of one snapshot-pinned curation table version."""

    table: str
    version: int
    status: str
    on_disk: bool
    tagged: bool
    readable: bool | None
    current_version: int | None
    snapshots: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "version": self.version,
            "status": self.status,
            "on_disk": self.on_disk,
            "tagged": self.tagged,
            "readable": self.readable,
            "current_version": self.current_version,
            "snapshots": list(self.snapshots),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CurationReplayReadinessReport:
    """Replay-readiness of the curation tables the active snapshots pin."""

    lake_uri: str
    schema_version: str
    backend: ReplayBackendConformance
    pins: tuple[CurationReplayPinStatus, ...]
    snapshots_checked: int
    status: str
    ready: bool
    suggested_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lake_uri": self.lake_uri,
            "backend": self.backend.to_dict(),
            "pins": [pin.to_dict() for pin in self.pins],
            "snapshots_checked": self.snapshots_checked,
            "status": self.status,
            "ready": self.ready,
            "suggested_actions": list(self.suggested_actions),
            "at_risk_pins": [
                pin.to_dict()
                for pin in self.pins
                if pin.status in (PIN_PRUNED, PIN_UNREADABLE, PIN_UNPROTECTED)
            ],
        }


def curation_replay_conformance(lake: Lake) -> ReplayBackendConformance:
    """Classify as-of curation replay support for ``lake``'s resolved backend.

    ``supported``   -- the backend advertises ``table_versioning`` and does not
                       delegate version lifecycle to a namespace, so a client
                       ``checkout(pinned_version)`` works and ``lake maintain``
                       can tag/protect the pinned versions.
    ``capability-gated`` -- the backend does not advertise ``table_versioning``
                       (e.g. a ``db://`` remote DB by default) but a direct-IO
                       fallback plane (object store / direct-pylance namespace)
                       restores as-of reads.
    ``unavailable`` -- version lifecycle is namespace-managed
                       (``namespace_managed_versioning``): the SDK cannot
                       guarantee or protect a pinned version via its own tag
                       mechanism, so replay depends on the namespace retaining
                       and serving that version.
    """
    spec = getattr(lake, "connection_spec", None)
    if spec is None:
        # Unclassified / in-process lake: legacy behaviour, replay just works.
        return ReplayBackendConformance(
            backend_kind="unclassified",
            data_plane="unclassified",
            status=REPLAY_SUPPORTED,
            capability=_CAPABILITY,
            advertised=True,
            namespace_managed_versioning=False,
            fallbacks=(),
            reason=None,
            suggested_action="none; replay reads run in-process against the dataset",
        )

    capabilities = spec.capabilities
    managed = bool(getattr(capabilities, "namespace_managed_versioning", False))
    advertised = backend_supports(spec, VERSIONING)
    fallbacks = ("object_store_lancedb_oss", "pylance_direct_namespace")

    if managed:
        return ReplayBackendConformance(
            backend_kind=spec.kind,
            data_plane=spec.data_plane,
            status=REPLAY_UNAVAILABLE,
            capability=_CAPABILITY,
            advertised=advertised,
            namespace_managed_versioning=True,
            fallbacks=(),
            reason=(
                "namespace manages table versioning; the SDK cannot pin or "
                "protect a historical curation version, so as-of replay depends "
                "on the namespace retaining and serving that version"
            ),
            suggested_action=(
                "ask the namespace to retain the snapshot-pinned curation "
                "versions, or replay from an object-store copy that owns its "
                "own version history"
            ),
        )
    if advertised:
        return ReplayBackendConformance(
            backend_kind=spec.kind,
            data_plane=spec.data_plane,
            status=REPLAY_SUPPORTED,
            capability=_CAPABILITY,
            advertised=True,
            namespace_managed_versioning=False,
            fallbacks=(),
            reason=None,
            suggested_action="none; run `lake maintain` to keep pins protected",
        )
    return ReplayBackendConformance(
        backend_kind=spec.kind,
        data_plane=spec.data_plane,
        status=REPLAY_CAPABILITY_GATED,
        capability=_CAPABILITY,
        advertised=False,
        namespace_managed_versioning=False,
        fallbacks=fallbacks,
        reason=(
            f"backend {spec.kind!r} does not advertise the {_CAPABILITY!r} "
            "capability required for as-of table-version checkout"
        ),
        suggested_action=(
            "point the data plane at " + " or ".join(fallbacks) + " for replay, "
            "or advertise the capability via "
            f"remote_capabilities={{{_CAPABILITY!r}: True}} if the deployment supports it"
        ),
    )


@dataclass
class _TableVersionMeta:
    available: set[int] | None
    current: int | None
    tags: set[str]


def _table_version_meta(lake: Lake, table: str) -> _TableVersionMeta:
    """Version set, current version, and managed pin tags for ``table``.

    ``available is None`` means the version list could not be enumerated (the
    caller then treats every pin as still on disk rather than falsely reporting
    it pruned -- mirrors ``maintenance._tag_pinned_versions``).
    """
    try:
        dataset = lake.table(table).to_lance()
    except Exception:  # noqa: BLE001 - backend cannot drop to LanceDataset here.
        return _TableVersionMeta(available=None, current=None, tags=set())
    try:
        current = int(dataset.version)
    except Exception:  # noqa: BLE001
        current = None
    try:
        available = {
            int(entry["version"] if isinstance(entry, dict) else entry.version)
            for entry in dataset.versions()
        }
    except Exception:  # noqa: BLE001
        available = None
    tags: set[str] = set()
    try:
        tags = set(dataset.tags.list())
    except Exception:  # noqa: BLE001 - tags optional / backend may not expose them.
        tags = set()
    return _TableVersionMeta(available=available, current=current, tags=tags)


def _checkout_readable(lake: Lake, table: str, version: int) -> bool:
    """Whether ``table`` can be checked out at ``version`` (restores latest)."""
    handle = lake.table(table)
    checked_out = False
    try:
        handle.checkout(int(version))
        checked_out = True
    except Exception:  # noqa: BLE001 - unreadable pin is the signal, not an error.
        return False
    finally:
        if checked_out:
            try:
                handle.checkout_latest()
            except Exception:  # noqa: BLE001 - best-effort restore.
                pass
    return True


def _iter_snapshot_curation_pins(
    lake: Lake,
    *,
    snapshot_name: str | None,
) -> tuple[dict[tuple[str, int], set[str]], int]:
    """Collect ``{(table, version): {snapshot names}}`` for curation pins.

    ``dataset_snapshots`` is streamed with a light projection (never a blob
    column). When ``snapshot_name`` is given only that snapshot's latest row is
    scoped (matching replay's ``_latest_snapshot_row``); otherwise every active
    snapshot row is checked -- the same set ``snapshot_retention_pin_details``
    protects, so verification and protection stay consistent.
    """
    handle = lake.table("dataset_snapshots")
    available = set(handle.schema.names)
    projected = [column for column in _SNAPSHOT_COLUMNS if column in available]
    scoped = str(snapshot_name).strip() if snapshot_name else ""

    def _rows() -> Any:
        try:
            query = handle.search()
            if projected:
                query = query.select(projected)
            for batch in query.to_batches(batch_size=_SNAPSHOT_SCAN_BATCH):
                yield from batch.to_pylist()
            return
        except Exception:  # noqa: BLE001 - streamed scan unavailable on this backend.
            # Fallback materializes the whole snapshot catalog -- NOT bounded, but
            # no worse than the incumbent `_latest_snapshot_row` /
            # `snapshot_retention_pin_details` readers, and only reached when the
            # streamed+projected primary path above cannot run. A paged readiness
            # surface over huge snapshot catalogs is follow-up 0478.
            pass
        yield from handle.to_arrow().to_pylist()

    pins: dict[tuple[str, int], set[str]] = {}
    latest_for_name: dict[str, tuple[Any, list[Any]]] = {}
    checked = 0
    for row in _rows():
        name = str(row.get("name") or "")
        if scoped and name != scoped:
            continue
        table_versions = list(row.get("table_versions") or ())
        if scoped:
            # Defer: keep only the latest row for the scoped name.
            key = (row.get("created_at"), str(row.get("dataset_id") or ""))
            existing = latest_for_name.get(name)
            if existing is None or key > existing[0]:
                latest_for_name[name] = (key, table_versions)
            continue
        checked += 1
        for entry in table_versions:
            table = str(entry.get("table") or "")
            if table not in CURATION_REPLAY_TABLES or entry.get("version") is None:
                continue
            pins.setdefault((table, int(entry["version"])), set()).add(name)

    for name, (_key, table_versions) in latest_for_name.items():
        checked += 1
        for entry in table_versions:
            table = str(entry.get("table") or "")
            if table not in CURATION_REPLAY_TABLES or entry.get("version") is None:
                continue
            pins.setdefault((table, int(entry["version"])), set()).add(name)
    return pins, checked


def curation_replay_readiness(
    lake: Lake,
    *,
    snapshot_name: str | None = None,
    check_readability: bool = True,
) -> CurationReplayReadinessReport:
    """Report replay-readiness of curation versions the active snapshots pin.

    For a ``supported`` backend each pinned curation version is classified
    ``protected`` (managed pin tag, or the current version), ``unprotected``
    (on disk and readable but nothing keeps a later cleanup from pruning it),
    ``pruned`` (gone from disk), or ``unreadable`` (present but a checkout
    fails). For a ``capability-gated`` / ``unavailable`` backend the on-disk
    state cannot be inspected here (direct LanceDataset access is gated), so
    pins are reported ``backend-gated`` and the suggested action points at the
    fallback plane.
    """
    conformance = curation_replay_conformance(lake)
    pins_map, snapshots_checked = _iter_snapshot_curation_pins(
        lake, snapshot_name=snapshot_name
    )

    spec = getattr(lake, "connection_spec", None)
    can_inspect = backend_supports(spec, DIRECT_LANCE)
    can_read = (
        check_readability
        and can_inspect
        and conformance.status == REPLAY_SUPPORTED
    )

    statuses: list[CurationReplayPinStatus] = []
    tables = sorted({table for (table, _version) in pins_map})
    per_table = {table: _table_version_meta(lake, table) for table in tables} if can_inspect else {}

    for (table, version), names in sorted(pins_map.items()):
        snapshots = tuple(sorted(names))
        if not can_inspect:
            statuses.append(
                CurationReplayPinStatus(
                    table=table,
                    version=version,
                    status=PIN_BACKEND_GATED,
                    on_disk=False,
                    tagged=False,
                    readable=None,
                    current_version=None,
                    snapshots=snapshots,
                    detail=(
                        "backend cannot inspect on-disk versions; "
                        f"{conformance.suggested_action}"
                    ),
                )
            )
            continue
        meta = per_table[table]
        on_disk = meta.available is None or version in meta.available
        tagged = f"{_PIN_TAG_PREFIX}{version}" in meta.tags
        current = meta.current
        readable: bool | None = None
        if not on_disk:
            status = PIN_PRUNED
            detail = (
                f"snapshot-pinned {table}@{version} is no longer on disk; "
                "run `lake maintain` before cleanup to tag replay pins"
            )
        else:
            protected = tagged or (current is not None and version == current)
            if can_read:
                readable = _checkout_readable(lake, table, version)
            if readable is False:
                status = PIN_UNREADABLE
                detail = (
                    f"{table}@{version} is present but cannot be checked out on "
                    "this backend"
                )
            elif protected:
                status = PIN_PROTECTED
                detail = (
                    f"{table}@{version} is protected by a managed pin tag"
                    if tagged
                    else f"{table}@{version} is the current version"
                )
            else:
                status = PIN_UNPROTECTED
                detail = (
                    f"{table}@{version} is present but untagged; a version cleanup "
                    "could prune it -- run `lake maintain` to tag snapshot pins"
                )
        statuses.append(
            CurationReplayPinStatus(
                table=table,
                version=version,
                status=status,
                on_disk=on_disk,
                tagged=tagged,
                readable=readable,
                current_version=current,
                snapshots=snapshots,
                detail=detail,
            )
        )

    at_risk = [s for s in statuses if s.status in (PIN_PRUNED, PIN_UNREADABLE, PIN_UNPROTECTED)]
    suggested: list[str] = []
    # A backend can advertise `table_versioning` (so conformance is "supported")
    # yet not expose direct object IO (`db://` with remote_capabilities
    # table_versioning=True): then `can_inspect` is False and every pin came back
    # `backend-gated`. Those pins are never verified on disk, so the verdict must
    # NOT fall through to `ready` -- reporting readiness we never checked is
    # exactly the silent-degrade failure SKILLS.md forbids.
    if conformance.status != REPLAY_SUPPORTED or not can_inspect:
        overall = BACKEND_GATED
        ready = False
        if conformance.status != REPLAY_SUPPORTED:
            suggested.append(conformance.suggested_action)
        else:
            suggested.append(
                f"backend {conformance.backend_kind!r} advertises versioning but not "
                "direct object IO, so replay pins cannot be verified here; verify from "
                "object_store_lancedb_oss or pylance_direct_namespace"
            )
    elif at_risk:
        overall = AT_RISK
        ready = False
        if any(s.status == PIN_PRUNED for s in at_risk):
            suggested.append(
                "a snapshot-pinned curation version has been pruned; restore it "
                "or treat the affected snapshot as non-replayable"
            )
        if any(s.status == PIN_UNPROTECTED for s in at_risk):
            suggested.append(
                "run `lake maintain` to tag snapshot-pinned curation versions "
                "before the next version cleanup"
            )
        if any(s.status == PIN_UNREADABLE for s in at_risk):
            suggested.append(
                "a pinned curation version is unreadable on this backend; replay "
                "from a backend that owns the dataset version history"
            )
    else:
        overall = READY
        ready = True

    return CurationReplayReadinessReport(
        lake_uri=lake.uri,
        schema_version=READINESS_SCHEMA_VERSION,
        backend=conformance,
        pins=tuple(statuses),
        snapshots_checked=snapshots_checked,
        status=overall,
        ready=ready,
        suggested_actions=tuple(suggested),
    )
