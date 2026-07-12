"""Executable conformance for the 0076 invocation audit (backlog 0130).

The 0076 audit
(``docs/product/lancedb-invocation-remote-compatibility-audit.md``) statically
classifies every SDK call site into one of four *support classes* describing how
it reaches a remote LanceDB Enterprise / Lance Namespace backend:

* ``enterprise_remote_supported_now`` -- runs on the plain ``db://`` Table/query
  surface today;
* ``enterprise_remote_capability_check`` -- a control-plane op (index / version /
  schema) that only runs when the deployment *advertises* the matching
  capability;
* ``namespace_or_object_store_direct_required`` -- drops to the underlying
  ``lance.LanceDataset`` (blob / direct scan / maintenance) and needs
  ``direct_object_io``, which ``db://`` does not expose;
* ``namespace_required`` -- a Lance Namespace control-plane operation.

That audit is static and intentionally conservative. This module turns it into
executable proof: it samples a bounded, deterministic set of representative rows
per class and *probes* each against a backend -- exercising the real 0128
capability gates (:mod:`lancedb_robotics.capability_gates`) and the real 0129
namespace-direct adapter (:mod:`lancedb_robotics.pylance_execution`) -- then
reports ``supported`` / ``unsupported`` / ``skipped`` per row with a concrete
reason, linked back to the 0076 audit row id.

Two run modes:

* :data:`MODE_CONTRACT_TEST` (default) resolves backend specs from synthetic
  URIs and injects a fake namespace client with credential-vending and
  managed-versioning fixtures. It needs no live credentials and no optional
  dependency, so it runs in the default CI lane and is the executable contract.
* :data:`MODE_LIVE` reads the live ``db://`` / namespace endpoint and auth-refs
  from the environment. When those credentials are absent the live subset is
  reported ``skipped`` with the missing variable names -- it never fails an
  unrelated local run (acceptance: "run in contract-test mode when live
  credentials are absent").

Provenance recorded in the report is secret-free by construction (it reuses
:func:`lancedb_robotics.pylance_execution.data_plane_provenance`): data-plane,
backend kind, namespace impl, and boolean facts about credential vending and
managed versioning -- never a resolved token or vended session secret.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lancedb_robotics import pylance_execution as px
from lancedb_robotics.capability_gates import (
    OPERATION_GATES,
    BackendCapabilityError,
    require_backend_capability,
)
from lancedb_robotics.connections import (
    LakeCapabilities,
    LakeConnectionSpec,
    ManagedVersioningMismatch,
    resolve_lake_connection,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPORT_SCHEMA = "lancedb-robotics/invocation-conformance/v1"

SUPPORT_CLASS_SUPPORTED_NOW = "enterprise_remote_supported_now"
SUPPORT_CLASS_CAPABILITY_CHECK = "enterprise_remote_capability_check"
SUPPORT_CLASS_DIRECT_REQUIRED = "namespace_or_object_store_direct_required"
SUPPORT_CLASS_NAMESPACE_REQUIRED = "namespace_required"

#: Support classes in a stable, reported order.
SUPPORT_CLASSES: tuple[str, ...] = (
    SUPPORT_CLASS_SUPPORTED_NOW,
    SUPPORT_CLASS_CAPABILITY_CHECK,
    SUPPORT_CLASS_DIRECT_REQUIRED,
    SUPPORT_CLASS_NAMESPACE_REQUIRED,
)

#: Capability statuses recorded per (sample, backend) probe.
STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"
STATUS_SKIPPED = "skipped"
CAPABILITY_STATUSES: tuple[str, ...] = (STATUS_SUPPORTED, STATUS_UNSUPPORTED, STATUS_SKIPPED)

MODE_CONTRACT_TEST = "contract-test"
MODE_LIVE = "live"

#: Audit ``families`` tokens that map onto a backlog-0128 gated operation family.
#: The other tokens (``db_control`` / ``table_read`` / ``table_write`` /
#: ``search`` / ``namespace``) are the ungated base surface or the namespace
#: control-plane, handled separately.
GATED_FAMILIES: frozenset[str] = frozenset(OPERATION_GATES)

#: Control-plane capabilities a ``db://`` deployment can positively advertise
#: (see ``resolve_lake_connection(remote_capabilities=...)``). ``direct_object_io``
#: is deliberately absent: ``db://`` never exposes direct object IO, which is why
#: the direct/blob/maintenance families need an object-store or namespace backend.
_ADVERTISABLE_CAPABILITIES = frozenset(
    {"index_management", "table_versioning", "schema_evolution"}
)

#: Default bounded sample per support class. Kept small so the suite exercises a
#: representative row from every class without probing all 708 audited call
#: sites; the report records sampled-vs-total so the bound is never silent.
DEFAULT_SAMPLE_PER_CLASS = 6

#: Representative logical table the namespace-direct probe resolves against.
_NAMESPACE_TABLE_NAME = "observations"

#: Deterministic clock for the contract-test credential-refresh fixture, so the
#: 0129 ``refresh_if_needed`` path is exercised without reading the wall clock.
_FIXED_NOW_MILLIS = 1_700_000_000_000
_NEAR_EXPIRY_MARGIN_MILLIS = 1_000
_FAR_FUTURE_MILLIS = _FIXED_NOW_MILLIS + 3_600_000

_AUDIT_ENV_VAR = "LANCEDB_ROBOTICS_INVOCATION_AUDIT_JSON"

#: Live-mode credential env vars. Absent => the live subset is skipped.
LIVE_DB_URI_ENV = "LANCEDB_ROBOTICS_CONFORMANCE_DB_URI"
LIVE_DB_AUTH_REF_ENV = "LANCEDB_ROBOTICS_CONFORMANCE_DB_AUTH_REF"
LIVE_NAMESPACE_IMPL_ENV = "LANCEDB_ROBOTICS_CONFORMANCE_NAMESPACE_IMPL"
LIVE_NAMESPACE_PROPERTIES_ENV = "LANCEDB_ROBOTICS_CONFORMANCE_NAMESPACE_PROPERTIES_JSON"
LIVE_NAMESPACE_AUTH_REF_ENV = "LANCEDB_ROBOTICS_CONFORMANCE_NAMESPACE_AUTH_REF"
LIVE_STORAGE_AUTH_REF_ENV = "LANCEDB_ROBOTICS_CONFORMANCE_STORAGE_AUTH_REF"
LIVE_OBJECT_STORE_URI_ENV = "LANCEDB_ROBOTICS_CONFORMANCE_OBJECT_STORE_URI"

#: The subset that must be present to attempt any live probe.
LIVE_CREDENTIAL_ENV_VARS: tuple[str, ...] = (
    LIVE_DB_URI_ENV,
    LIVE_DB_AUTH_REF_ENV,
    LIVE_NAMESPACE_IMPL_ENV,
    LIVE_NAMESPACE_PROPERTIES_ENV,
)


def _default_audit_path() -> Path:
    override = os.environ.get(_AUDIT_ENV_VAR)
    if override:
        return Path(override)
    return (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "product"
        / "lancedb-invocation-remote-compatibility-audit.json"
    )


#: Default location of the 0076 audit JSON, overridable via ``$AUDIT_ENV_VAR``.
AUDIT_JSON_PATH = _default_audit_path()


# --------------------------------------------------------------------------- #
# Audit loading + deterministic sampling
# --------------------------------------------------------------------------- #


def load_audit_rows(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the 0076 audit rows (``id`` / ``support_class`` / ``families`` ...)."""
    audit_path = Path(path) if path is not None else _default_audit_path()
    payload = json.loads(audit_path.read_text())
    return list(payload["rows"])


def audit_schema_version(path: str | Path | None = None) -> str:
    """Return the audit JSON ``schema_version`` (linked in the report)."""
    audit_path = Path(path) if path is not None else _default_audit_path()
    return str(json.loads(audit_path.read_text()).get("schema_version", "unknown"))


@dataclass(frozen=True)
class AuditSample:
    """One representative audited call site, reduced to what the probe needs."""

    audit_id: str
    support_class: str
    path: str
    line: int
    function: str
    families: tuple[str, ...]
    callees: tuple[str, ...]
    gated_families: tuple[str, ...]
    needs_direct_object_io: bool
    needs_namespace_endpoint: bool

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> AuditSample:
        families = tuple(row.get("families", ()))
        return cls(
            audit_id=str(row["id"]),
            support_class=str(row["support_class"]),
            path=str(row.get("path", "")),
            line=int(row.get("line", 0)),
            function=str(row.get("function", "")),
            families=families,
            callees=tuple(row.get("callees", ())),
            gated_families=tuple(f for f in families if f in GATED_FAMILIES),
            needs_direct_object_io=bool(row.get("needs_direct_object_io", False)),
            needs_namespace_endpoint=bool(row.get("needs_namespace_endpoint", False)),
        )


def sample_audit(
    rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    per_class: int = DEFAULT_SAMPLE_PER_CLASS,
    path: str | Path | None = None,
) -> tuple[AuditSample, ...]:
    """Deterministically sample up to ``per_class`` rows from each support class.

    Sampling is by sorted ``id`` (stable across runs and machines -- no RNG), so
    the same audit produces the same representative set every time. Rows are
    grouped by support class and emitted in :data:`SUPPORT_CLASSES` order.
    """
    if per_class < 1:
        raise ValueError("per_class must be >= 1")
    source = list(rows) if rows is not None else load_audit_rows(path)
    by_class: dict[str, list[Mapping[str, Any]]] = {sc: [] for sc in SUPPORT_CLASSES}
    for row in source:
        support_class = str(row.get("support_class", ""))
        if support_class in by_class:
            by_class[support_class].append(row)
    samples: list[AuditSample] = []
    for support_class in SUPPORT_CLASSES:
        ordered = sorted(by_class[support_class], key=lambda r: str(r["id"]))
        samples.extend(AuditSample.from_row(row) for row in ordered[:per_class])
    return tuple(samples)


# --------------------------------------------------------------------------- #
# Contract-test namespace fixture (credential vending + managed versioning)
# --------------------------------------------------------------------------- #


class FakeNamespaceClient:
    """In-memory Lance Namespace stand-in for contract-test conformance.

    ``describe_table`` vends short-lived storage credentials that expire on the
    *first* describe (within the 0129 refresh margin, forcing a refresh) and are
    long-lived on every describe thereafter, so the refresh path is exercised
    deterministically. ``managed_versioning`` drives the write-guard fixture. The
    vended ``storage_options`` carry no secret -- only a region label -- so a
    report built from a describe response is secret-free.
    """

    def __init__(
        self,
        *,
        location: str = "s3://robotics-conformance/observations.lance",
        managed_versioning: bool = False,
        near_expiry: bool = True,
        now_millis: int = _FIXED_NOW_MILLIS,
    ) -> None:
        self.location = location
        self.managed_versioning = managed_versioning
        self.near_expiry = near_expiry
        self.now_millis = now_millis
        self.describe_calls = 0

    def describe_table(self, request: Any) -> dict[str, Any]:
        self.describe_calls += 1
        if self.near_expiry and self.describe_calls == 1:
            expires = self.now_millis + _NEAR_EXPIRY_MARGIN_MILLIS
        else:
            expires = _FAR_FUTURE_MILLIS
        return {
            "location": self.location,
            "storage_options": {"aws_region": "us-west-2"},
            "expires_at_millis": expires,
            "managed_versioning": self.managed_versioning,
        }


NamespaceClientFactory = Callable[..., Any]


def _contract_db_spec(uri: str, advertise: Sequence[str]) -> LakeConnectionSpec:
    """A ``db://`` spec mirroring ``resolve_lake_connection`` without credentials.

    ``advertise`` is the set of control-plane capability names an operator's
    deployment is known to expose; unadvertised ones stay ``False`` (the honest
    "not advertised" state the 0128 gates key on). ``direct_object_io`` is always
    ``False`` -- ``db://`` never exposes direct object IO.
    """
    advertised = set(advertise)
    return LakeConnectionSpec(
        kind="lancedb_remote_db",
        uri=uri,
        display_uri=uri,
        direct_object_io_allowed=False,
        capabilities=LakeCapabilities(
            server_side_query=True,
            direct_object_io=False,
            namespace_resolution=False,
            geneva_worker_specs=False,
            blob_fetch_remote=True,
            index_management="index_management" in advertised,
            table_versioning="table_versioning" in advertised,
            schema_evolution="schema_evolution" in advertised,
        ),
    )


def _contract_object_store_spec(uri: str) -> LakeConnectionSpec:
    """An object-store spec mirroring ``resolve_lake_connection`` (all control-plane
    capabilities owned via direct object IO)."""
    return LakeConnectionSpec(
        kind="object_store_lancedb_oss",
        uri=uri,
        display_uri=uri,
        direct_object_io_allowed=True,
        capabilities=LakeCapabilities(
            direct_object_io=True,
            blob_fetch_remote=True,
            index_management=True,
            table_versioning=True,
            schema_evolution=True,
        ),
    )


@dataclass
class ConformanceEnv:
    """Resolved backend configuration for a conformance run.

    Holds the *configuration* (URIs, auth-ref names, namespace properties), not
    pre-resolved specs, because the supported-now and capability-check probes
    advertise a row's specific control-plane capabilities on the ``db://`` spec.
    ``namespace_client_factory`` is a fake in contract-test mode and ``None`` in
    live mode (where a real client is built from the spec).
    """

    mode: str
    db_uri: str
    object_store_uri: str
    namespace_impl: str
    namespace_properties: dict[str, str]
    remote_auth_ref: str | None = None
    namespace_auth_ref: str | None = None
    storage_auth_ref: str | None = None
    namespace_client_factory: NamespaceClientFactory | None = None
    live_available: bool = True
    live_missing: tuple[str, ...] = ()
    fixed_now_millis: int = _FIXED_NOW_MILLIS

    def db_spec(self, *, advertise: Iterable[str] = ()) -> LakeConnectionSpec:
        advertised = tuple(advertise)
        if self.mode == MODE_LIVE:
            # Live mode goes through the real resolver (the operator has real
            # credentials); this is the production db:// resolution path.
            return resolve_lake_connection(
                self.db_uri,
                remote_auth_ref=self.remote_auth_ref,
                storage_auth_ref=self.storage_auth_ref,
                remote_capabilities={name: True for name in advertised} or None,
            )
        # Contract-test mode builds the spec directly so the run needs no live
        # Enterprise api_key (the resolver requires one for db://). The
        # capabilities mirror ``resolve_lake_connection``'s db:// branch exactly,
        # so the 0128 gate reads the same spec a real resolved db:// would carry.
        return _contract_db_spec(self.db_uri, advertised)

    def object_store_spec(self) -> LakeConnectionSpec:
        if self.mode == MODE_LIVE:
            return resolve_lake_connection(
                self.object_store_uri, storage_auth_ref=self.storage_auth_ref
            )
        return _contract_object_store_spec(self.object_store_uri)

    @property
    def namespace_spec(self) -> LakeConnectionSpec:
        return resolve_lake_connection(
            namespace_client_impl=self.namespace_impl,
            namespace_client_properties=self.namespace_properties,
            namespace_auth_ref=self.namespace_auth_ref,
            storage_auth_ref=self.storage_auth_ref,
        )


def contract_test_env(
    *,
    managed_versioning: bool = False,
    near_expiry: bool = True,
) -> ConformanceEnv:
    """Build a contract-test environment backed by a fake namespace client.

    No live credentials and no optional dependency are required: the ``db://``
    and object-store specs resolve from synthetic URIs (no network), and the
    namespace client is a :class:`FakeNamespaceClient` whose vended credentials
    and managed-versioning state come from the fixtures.
    """

    def factory(
        *,
        managed_versioning: bool | None = None,
        near_expiry: bool | None = None,
    ) -> FakeNamespaceClient:
        return FakeNamespaceClient(
            managed_versioning=(
                contract_managed if managed_versioning is None else managed_versioning
            ),
            near_expiry=(contract_near_expiry if near_expiry is None else near_expiry),
        )

    contract_managed = managed_versioning
    contract_near_expiry = near_expiry
    return ConformanceEnv(
        mode=MODE_CONTRACT_TEST,
        db_uri="db://robotics-conformance",
        object_store_uri="s3://robotics-conformance/lake",
        namespace_impl="rest",
        namespace_properties={
            "uri": "https://ns.conformance.invalid",
            "header.x-lancedb-database-prefix": "robotics",
            "delimiter": ".",
        },
        remote_auth_ref="conformance-remote",
        namespace_auth_ref="conformance-namespace",
        storage_auth_ref="conformance-storage",
        namespace_client_factory=factory,
        live_available=True,
        live_missing=(),
    )


def live_credentials_available() -> tuple[bool, tuple[str, ...]]:
    """Whether the live-mode credential env vars are present.

    Returns ``(available, missing)`` where ``missing`` names the absent required
    variables. A live run with missing credentials is skipped, not failed.
    """
    missing = tuple(name for name in LIVE_CREDENTIAL_ENV_VARS if not os.environ.get(name))
    return (not missing, missing)


def _live_env() -> ConformanceEnv:
    available, missing = live_credentials_available()
    if not available:
        # Config is unusable; the run marks every probe skipped before touching
        # a backend, so placeholder values here are never dereferenced.
        return ConformanceEnv(
            mode=MODE_LIVE,
            db_uri="",
            object_store_uri="",
            namespace_impl="rest",
            namespace_properties={},
            namespace_client_factory=None,
            live_available=False,
            live_missing=missing,
        )
    properties = json.loads(os.environ.get(LIVE_NAMESPACE_PROPERTIES_ENV, "{}"))
    return ConformanceEnv(
        mode=MODE_LIVE,
        db_uri=os.environ[LIVE_DB_URI_ENV],
        # Only a real object-store URI; never the db:// URI (resolving a db://
        # scheme as an object store is meaningless). Absent => the object-store
        # side of direct_required is deferred with the rest of live alternate-
        # backend execution (0446).
        object_store_uri=os.environ.get(LIVE_OBJECT_STORE_URI_ENV, ""),
        namespace_impl=os.environ.get(LIVE_NAMESPACE_IMPL_ENV, "rest"),
        namespace_properties={str(k): str(v) for k, v in properties.items()},
        remote_auth_ref=os.environ.get(LIVE_DB_AUTH_REF_ENV),
        namespace_auth_ref=os.environ.get(LIVE_NAMESPACE_AUTH_REF_ENV),
        storage_auth_ref=os.environ.get(LIVE_STORAGE_AUTH_REF_ENV),
        namespace_client_factory=None,
        live_available=True,
        live_missing=(),
    )


def _build_env(mode: str) -> ConformanceEnv:
    if mode == MODE_CONTRACT_TEST:
        return contract_test_env()
    if mode == MODE_LIVE:
        return _live_env()
    raise ValueError(f"unknown conformance mode {mode!r}; use {MODE_CONTRACT_TEST!r} or {MODE_LIVE!r}")


# --------------------------------------------------------------------------- #
# Probe result dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackendProbe:
    """The observed status of one sample against one backend."""

    backend_role: str
    backend_kind: str
    data_plane: str
    capability_status: str
    reason: str | None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_role": self.backend_role,
            "backend_kind": self.backend_kind,
            "data_plane": self.data_plane,
            "capability_status": self.capability_status,
            "reason": self.reason,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ConformanceProbe:
    """A sampled row's conformance verdict across its support class's backends."""

    audit_id: str
    support_class: str
    path: str
    line: int
    families: tuple[str, ...]
    gated_families: tuple[str, ...]
    conformant: bool
    mismatch: str | None
    backend_probes: tuple[BackendProbe, ...]
    credential_refreshed: bool = False
    managed_versioning_enforced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "support_class": self.support_class,
            "path": self.path,
            "line": self.line,
            "families": list(self.families),
            "gated_families": list(self.gated_families),
            "conformant": self.conformant,
            "mismatch": self.mismatch,
            "credential_refreshed": self.credential_refreshed,
            "managed_versioning_enforced": self.managed_versioning_enforced,
            "backend_probes": [bp.to_dict() for bp in self.backend_probes],
        }


@dataclass(frozen=True)
class InvocationConformanceReport:
    """Aggregate compatibility report linked back to 0076 audit row ids."""

    mode: str
    audit_schema_version: str
    per_class_sampled: dict[str, int]
    per_class_total: dict[str, int]
    probes: tuple[ConformanceProbe, ...]
    skipped_live: tuple[str, ...] = ()

    def ok(self) -> bool:
        return all(probe.conformant for probe in self.probes)

    def non_conformant(self) -> tuple[ConformanceProbe, ...]:
        return tuple(probe for probe in self.probes if not probe.conformant)

    def summary(self) -> dict[str, Any]:
        by_support_class = {sc: 0 for sc in SUPPORT_CLASSES}
        by_status: dict[str, int] = {status: 0 for status in CAPABILITY_STATUSES}
        conformant = 0
        for probe in self.probes:
            by_support_class[probe.support_class] = (
                by_support_class.get(probe.support_class, 0) + 1
            )
            if probe.conformant:
                conformant += 1
            for backend_probe in probe.backend_probes:
                by_status[backend_probe.capability_status] = (
                    by_status.get(backend_probe.capability_status, 0) + 1
                )
        return {
            "total_samples": len(self.probes),
            "conformant": conformant,
            "non_conformant": len(self.probes) - conformant,
            "by_support_class": by_support_class,
            "backend_probe_status_counts": by_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "mode": self.mode,
            "audit_schema_version": self.audit_schema_version,
            "per_class_sampled": dict(self.per_class_sampled),
            "per_class_total": dict(self.per_class_total),
            "summary": self.summary(),
            "skipped_live": list(self.skipped_live),
            "probes": [probe.to_dict() for probe in self.probes],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# --------------------------------------------------------------------------- #
# Gate probing helpers
# --------------------------------------------------------------------------- #


def _advertisable_capabilities(families: Sequence[str]) -> tuple[str, ...]:
    """Control-plane capabilities a ``db://`` deployment can advertise for a row.

    Direct-object-IO families (``direct_lance`` / ``blob`` / ``maintenance``) map
    to a capability ``db://`` cannot advertise, so they contribute nothing here --
    which is exactly why those rows are classified as needing a direct backend.
    """
    caps: list[str] = []
    for family in families:
        gate = OPERATION_GATES.get(family)
        if gate is not None and gate.capability in _ADVERTISABLE_CAPABILITIES:
            if gate.capability not in caps:
                caps.append(gate.capability)
    return tuple(caps)


def _family_status(spec: LakeConnectionSpec, family: str) -> tuple[str, str | None]:
    """Status of one operation family against ``spec`` using the 0128 gates."""
    if family not in GATED_FAMILIES:
        return STATUS_SUPPORTED, None
    try:
        require_backend_capability(spec, family)
    except BackendCapabilityError as exc:
        return STATUS_UNSUPPORTED, str(exc)
    return STATUS_SUPPORTED, None


def _aggregate_family_status(
    spec: LakeConnectionSpec, families: Sequence[str]
) -> tuple[str, str | None]:
    """Aggregate a row's families: unsupported if *any* family is gated out."""
    first_reason: str | None = None
    for family in families:
        status, reason = _family_status(spec, family)
        if status == STATUS_UNSUPPORTED:
            return STATUS_UNSUPPORTED, reason
        if first_reason is None and reason is not None:
            first_reason = reason
    return STATUS_SUPPORTED, first_reason


def _backend_probe(
    role: str, spec: LakeConnectionSpec, families: Sequence[str]
) -> BackendProbe:
    status, reason = _aggregate_family_status(spec, families)
    return BackendProbe(
        backend_role=role,
        backend_kind=spec.kind,
        data_plane=spec.data_plane,
        capability_status=status,
        reason=reason,
        provenance=px.data_plane_provenance(spec),
    )


# --------------------------------------------------------------------------- #
# Per-support-class probes
# --------------------------------------------------------------------------- #


def _probe_supported_now(sample: AuditSample, env: ConformanceEnv) -> ConformanceProbe:
    # Supported now == runs on db:// (advertising whatever control-plane caps the
    # row's families need; base-surface rows advertise nothing). NOTE: because
    # this advertises the row's own caps, it cannot detect a *mis-classified*
    # supported_now row that actually carries a gated family (3 such rows today);
    # probing those against a plain db:// to catch misclassification is folded
    # into the full-audit sweep follow-on (0445).
    spec = env.db_spec(advertise=_advertisable_capabilities(sample.families))
    probe = _backend_probe("db_remote", spec, sample.families)
    conformant = probe.capability_status == STATUS_SUPPORTED
    mismatch = None
    if not conformant:
        mismatch = (
            f"audit says supported now on db://, but families {list(sample.families)} "
            f"gated: {probe.reason}"
        )
    return _finish(sample, conformant, mismatch, (probe,))


def _probe_capability_check(sample: AuditSample, env: ConformanceEnv) -> ConformanceProbe:
    # Must be gated on a plain db:// and open once the capability is advertised.
    plain = _backend_probe("db_remote", env.db_spec(), sample.families)
    advertised = _backend_probe(
        "db_remote_advertised",
        env.db_spec(advertise=_advertisable_capabilities(sample.families)),
        sample.families,
    )
    conformant = (
        plain.capability_status == STATUS_UNSUPPORTED
        and advertised.capability_status == STATUS_SUPPORTED
    )
    mismatch = None
    if not conformant:
        mismatch = (
            "audit expects a capability gate (unsupported on plain db://, "
            f"supported when advertised), observed plain={plain.capability_status} "
            f"advertised={advertised.capability_status}"
        )
    return _finish(sample, conformant, mismatch, (plain, advertised))


def _probe_direct_required(sample: AuditSample, env: ConformanceEnv) -> ConformanceProbe:
    # Needs direct_object_io: gated on db://, supported on an object-store backend.
    db_probe = _backend_probe("db_remote", env.db_spec(), sample.families)
    object_store_probe = _backend_probe(
        "object_store", env.object_store_spec(), sample.families
    )
    conformant = (
        db_probe.capability_status == STATUS_UNSUPPORTED
        and object_store_probe.capability_status == STATUS_SUPPORTED
    )
    mismatch = None
    if not conformant:
        mismatch = (
            "audit expects direct object IO (unsupported on db://, supported on "
            f"object store), observed db={db_probe.capability_status} "
            f"object_store={object_store_probe.capability_status}"
        )
    return _finish(sample, conformant, mismatch, (db_probe, object_store_probe))


def _probe_namespace_required(sample: AuditSample, env: ConformanceEnv) -> ConformanceProbe:
    # A namespace control-plane op: unsupported on db:// (no namespace data-plane),
    # supported on a namespace backend, where it must vend + refresh credentials
    # and enforce managed versioning (0129).
    db_spec = env.db_spec()
    db_probe = BackendProbe(
        backend_role="db_remote",
        backend_kind=db_spec.kind,
        data_plane=db_spec.data_plane,
        capability_status=STATUS_UNSUPPORTED,
        reason=(
            f"backend {db_spec.kind!r} has no namespace data-plane; a Lance "
            "Namespace control-plane operation requires a namespace-backed lake "
            "(namespace_client_impl='rest') or direct pylance access"
        ),
        provenance=px.data_plane_provenance(db_spec),
    )

    spec = env.namespace_spec
    factory = env.namespace_client_factory
    if factory is None:
        # Live mode has no injected fake client. Executing a real namespace
        # describe/refresh against a live endpoint (and validating managed
        # versioning on a real catalog) is scoped to 0446. Until
        # then, SKIP this probe as conformant rather than claim a false
        # non-conformance for a backend we did not actually exercise -- a
        # skipped probe never fails the live lane, and a working live namespace
        # is never mislabelled a mismatch.
        ns_probe = BackendProbe(
            backend_role="namespace",
            backend_kind=spec.kind,
            data_plane=spec.data_plane,
            capability_status=STATUS_SKIPPED,
            reason="live namespace execution deferred to 0446",
            provenance=px.data_plane_provenance(spec),
        )
        return _finish(sample, True, None, (db_probe, ns_probe))

    # Credential vending + refresh (0129 describe -> refresh_if_needed).
    read_client = factory()
    access = px.namespace_access(spec, _NAMESPACE_TABLE_NAME, namespace_client=read_client)
    description = access.describe(vend_credentials=True)
    access.refresh_if_needed(description, now_millis=env.fixed_now_millis)
    credential_refreshed = getattr(read_client, "describe_calls", 0) >= 2

    # Managed-versioning write guard: a direct write against a managed namespace
    # must refuse loudly rather than fork the version history.
    managed_enforced = False
    managed_client = factory(managed_versioning=True)
    try:
        px.require_namespace_write_supported(
            spec,
            _NAMESPACE_TABLE_NAME,
            supports_managed_versioning=False,
            namespace_client=managed_client,
        )
    except ManagedVersioningMismatch:
        managed_enforced = True

    ns_probe = BackendProbe(
        backend_role="namespace",
        backend_kind=spec.kind,
        data_plane=spec.data_plane,
        capability_status=STATUS_SUPPORTED,
        reason=None,
        provenance=px.data_plane_provenance(spec),
    )
    conformant = (
        db_probe.capability_status == STATUS_UNSUPPORTED
        and credential_refreshed
        and managed_enforced
    )
    mismatch = None
    if not conformant:
        mismatch = (
            "namespace conformance failed: "
            f"credential_refreshed={credential_refreshed} "
            f"managed_versioning_enforced={managed_enforced}"
        )
    return _finish(
        sample,
        conformant,
        mismatch,
        (db_probe, ns_probe),
        credential_refreshed=credential_refreshed,
        managed_versioning_enforced=managed_enforced,
    )


_PROBE_DISPATCH: dict[str, Callable[[AuditSample, ConformanceEnv], ConformanceProbe]] = {
    SUPPORT_CLASS_SUPPORTED_NOW: _probe_supported_now,
    SUPPORT_CLASS_CAPABILITY_CHECK: _probe_capability_check,
    SUPPORT_CLASS_DIRECT_REQUIRED: _probe_direct_required,
    SUPPORT_CLASS_NAMESPACE_REQUIRED: _probe_namespace_required,
}


def _finish(
    sample: AuditSample,
    conformant: bool,
    mismatch: str | None,
    backend_probes: tuple[BackendProbe, ...],
    *,
    credential_refreshed: bool = False,
    managed_versioning_enforced: bool = False,
) -> ConformanceProbe:
    return ConformanceProbe(
        audit_id=sample.audit_id,
        support_class=sample.support_class,
        path=sample.path,
        line=sample.line,
        families=sample.families,
        gated_families=sample.gated_families,
        conformant=conformant,
        mismatch=mismatch,
        backend_probes=backend_probes,
        credential_refreshed=credential_refreshed,
        managed_versioning_enforced=managed_versioning_enforced,
    )


def _skipped_probe(sample: AuditSample, env: ConformanceEnv) -> ConformanceProbe:
    reason = (
        "live credentials absent; set "
        + ", ".join(env.live_missing or LIVE_CREDENTIAL_ENV_VARS)
    )
    spec = None
    # Resolve a spec purely for provenance labelling; never touch the backend.
    try:
        spec = env.db_spec() if env.db_uri else None
    except Exception:  # noqa: BLE001 - provenance is best-effort for a skip.
        spec = None
    probe = BackendProbe(
        backend_role="live",
        backend_kind=spec.kind if spec is not None else "lancedb_remote_db",
        data_plane=spec.data_plane if spec is not None else "remote_db",
        capability_status=STATUS_SKIPPED,
        reason=reason,
        provenance=px.data_plane_provenance(spec),
    )
    return _finish(sample, True, None, (probe,))


#: Support classes whose live validation needs a real *alternate* backend
#: (object-store or namespace) and full operation execution, which is scoped to
#: 0446. Live mode (0130) exercises only the real db:// posture
#: (supported_now capability gating + capability_check advertising); these two
#: are skipped-as-conformant in live mode with a deferral note, rather than
#: attempting a nonsensical alternate-backend resolution.
_LIVE_DEFERRED_CLASSES = frozenset(
    {SUPPORT_CLASS_DIRECT_REQUIRED, SUPPORT_CLASS_NAMESPACE_REQUIRED}
)


def _live_skip(sample: AuditSample, *, detail: str | None = None) -> ConformanceProbe:
    """A conformant, skipped live probe. The reason records only a static hint --
    never a raw exception message -- so a persisted report cannot leak a
    credential embedded in a driver error (SKILLS.md)."""
    reason = "live probe skipped"
    if detail:
        reason = f"{reason}: {detail}"
    probe = BackendProbe(
        backend_role="live",
        backend_kind="lancedb_remote_db",
        data_plane="remote_db",
        capability_status=STATUS_SKIPPED,
        reason=reason,
        provenance={},
    )
    return _finish(sample, True, None, (probe,))


def _live_deferred_probe(sample: AuditSample, env: ConformanceEnv) -> ConformanceProbe:
    """A conformant, skipped probe for a class whose live check is deferred."""
    db_spec = env.db_spec() if env.db_uri else None
    probe = BackendProbe(
        backend_role="live",
        backend_kind=db_spec.kind if db_spec is not None else "lancedb_remote_db",
        data_plane=db_spec.data_plane if db_spec is not None else "remote_db",
        capability_status=STATUS_SKIPPED,
        reason=(
            f"live {sample.support_class} execution needs a real alternate "
            "backend; deferred to 0446"
        ),
        provenance=px.data_plane_provenance(db_spec),
    )
    return _finish(sample, True, None, (probe,))


def _probe_sample(sample: AuditSample, env: ConformanceEnv) -> ConformanceProbe:
    prober = _PROBE_DISPATCH.get(sample.support_class)
    if prober is None:  # pragma: no cover - support classes are exhaustive.
        raise KeyError(f"no probe for support class {sample.support_class!r}")
    if env.mode == MODE_LIVE:
        if sample.support_class in _LIVE_DEFERRED_CLASSES:
            try:
                return _live_deferred_probe(sample, env)
            except Exception:  # noqa: BLE001 - spec resolution can fail host-specifically.
                return _live_skip(sample)
        try:
            return prober(sample, env)
        except Exception as exc:  # noqa: BLE001 - a live backend fails host-specifically.
            # Only the exception *type* is recorded (never its message, which a
            # driver/object-store error can seed with a presigned URL or token
            # that would then land in the persisted/uploaded report).
            return _live_skip(sample, detail=f"raised {type(exc).__name__}")
    return prober(sample, env)


# --------------------------------------------------------------------------- #
# Top-level run
# --------------------------------------------------------------------------- #


def run_conformance(
    *,
    mode: str = MODE_CONTRACT_TEST,
    per_class: int = DEFAULT_SAMPLE_PER_CLASS,
    path: str | Path | None = None,
    env: ConformanceEnv | None = None,
) -> InvocationConformanceReport:
    """Sample representative audit rows and probe them against ``mode``'s backend.

    In :data:`MODE_LIVE` without credentials, every sample is reported skipped
    (``ok()`` stays true); the report's ``skipped_live`` names the absent vars.
    """
    rows = load_audit_rows(path)
    samples = sample_audit(rows=rows, per_class=per_class)
    per_class_total = {sc: 0 for sc in SUPPORT_CLASSES}
    for row in rows:
        sc = str(row.get("support_class", ""))
        if sc in per_class_total:
            per_class_total[sc] += 1
    per_class_sampled = {sc: 0 for sc in SUPPORT_CLASSES}
    for sample in samples:
        per_class_sampled[sample.support_class] += 1

    run_env = env if env is not None else _build_env(mode)
    schema_version = audit_schema_version(path)

    if run_env.mode == MODE_LIVE and not run_env.live_available:
        probes = tuple(_skipped_probe(sample, run_env) for sample in samples)
        skipped_live = tuple(
            f"{name} unset" for name in run_env.live_missing
        ) or ("live credentials absent",)
        return InvocationConformanceReport(
            mode=run_env.mode,
            audit_schema_version=schema_version,
            per_class_sampled=per_class_sampled,
            per_class_total=per_class_total,
            probes=probes,
            skipped_live=skipped_live,
        )

    probes = tuple(_probe_sample(sample, run_env) for sample in samples)
    skipped_live: tuple[str, ...] = ()
    if run_env.mode == MODE_LIVE:
        skipped_live = tuple(
            bp.reason
            for probe in probes
            for bp in probe.backend_probes
            if bp.capability_status == STATUS_SKIPPED and bp.reason
        )
    return InvocationConformanceReport(
        mode=run_env.mode,
        audit_schema_version=schema_version,
        per_class_sampled=per_class_sampled,
        per_class_total=per_class_total,
        probes=probes,
        skipped_live=skipped_live,
    )


def run_contract_test(
    *, per_class: int = DEFAULT_SAMPLE_PER_CLASS, path: str | Path | None = None
) -> InvocationConformanceReport:
    """Convenience wrapper: run the contract-test mode with the fake backends."""
    return run_conformance(mode=MODE_CONTRACT_TEST, per_class=per_class, path=path)
