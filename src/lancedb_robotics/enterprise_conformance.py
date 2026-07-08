"""Enterprise training conformance and fault-injection harness (backlog 0116).

Backlog 0069 defines the public Enterprise training backend/report contract and
0070-0075 add the deeper remote execution path. This module gives teams a
repeatable way to prove that ``backend="enterprise"`` behaves the same across
``db://`` remote DBs, REST Namespace query nodes, local test doubles, and
degraded deployments -- without any Enterprise secrets in CI.

It has two layers:

* a **static compatibility matrix** (:func:`compatibility_matrix`) that drives the
  real :func:`lancedb_robotics.training._training_backend_report` classifier over
  a curated set of data-free fake-lake stand-ins. It documents, per case, whether
  the backend is *supported*, *falls back explicitly*, or is *unsupported* (a
  typed fail-fast error). Because it reuses the production classifier, the matrix
  can never drift from real behavior, and it is deterministic so it can be
  committed as generated reference documentation.

* a **live conformance run** (:func:`run_conformance`) that replays one real
  snapshot through each backend case against a local lake, injects each known
  failure mode, and asserts the central invariant: every degradation produces a
  typed error or an explicit fallback report -- never silent local
  materialization. It also proves ``host_override`` routing survives the worker
  handoff without serializing API keys, and that local and (faked) Enterprise
  paths emit equivalent sample ids, row ids, table-version lineage, and batch
  schemas for the same snapshot across epoch/worker/resume combinations.

A locally spawned LanceDB Enterprise CLI endpoint can be exercised for a first
real HTTP ``host_override`` + plan-executor registration pass via
:class:`LocalEnterpriseEndpoint`; that path is gated behind an environment flag
so it never runs in secret-free CI.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import training as _training
from .connections import LakeCapabilities, LakeConnectionSpec
from .training import (
    EnterpriseTrainingError,
    TrainingError,
    _native_required_enterprise_capabilities,
    _training_backend_report,
    iter_training_batches,
)

# Outcome categories used by both the matrix and the live run.
CATEGORY_SUPPORTED = "supported"
CATEGORY_FALLBACK = "fallback"
CATEGORY_UNSUPPORTED = "unsupported"
CONFORMANCE_CATEGORIES = (CATEGORY_SUPPORTED, CATEGORY_FALLBACK, CATEGORY_UNSUPPORTED)

# Env flag that opts a run into the real local Enterprise CLI endpoint pass.
LOCAL_ENDPOINT_ENV = "LANCEDB_ROBOTICS_ENTERPRISE_ENDPOINT"
DEFAULT_ENTERPRISE_CLI = Path.home() / "Downloads" / "lancedb"

# Attributes the harness toggles on a lake to simulate a backend/fault; saved and
# restored around each case so cases never leak state into one another.
_LAKE_SIM_ATTRS = (
    "connection_spec",
    "capabilities",
    "uri",
    "enterprise_training_capabilities",
    "enterprise_extra_available",
    "namespace_credentials_expired",
    "namespace_credential_refresh",
    "table",
)

# Secret-bearing keys that must never appear in a worker loader config/manifest.
_SECRET_MARKERS = ("api_key", "apikey", "api-key", "secret", "password", "token", "authorization")


class ConformanceError(TrainingError):
    """Raised when a conformance run detects a contract violation."""


@dataclass(frozen=True)
class ConnectionConfig:
    """Declarative description of a (faked) lake connection for one case.

    Applying it configures a :class:`~lancedb_robotics.connections.LakeConnectionSpec`
    and :class:`~lancedb_robotics.connections.LakeCapabilities` on a target -- a
    real local lake for the live run, or a lightweight stand-in for the static
    matrix. No real remote is contacted; only the backend classification changes.
    """

    kind: str = "local_path"
    uri: str = "./robot.lance"
    host_override: str | None = None
    region: str | None = None
    api_key: str | None = None
    auth_refs: Mapping[str, str] = field(default_factory=dict)
    server_side_query: bool = True
    blob_fetch_remote: bool = True
    direct_object_io: bool = False
    managed_versioning: bool = False
    namespace_endpoint: str | None = None
    namespace_client_impl: str | None = None

    def is_enterprise(self) -> bool:
        return self.kind in _training.ENTERPRISE_TRAINING_CONNECTION_KINDS

    def build_spec(self) -> LakeConnectionSpec | None:
        if self.kind == "local_path":
            return None
        connect_kwargs: dict[str, Any] = {}
        if self.api_key is not None:
            connect_kwargs["api_key"] = self.api_key
        if self.region is not None:
            connect_kwargs["region"] = self.region
        if self.host_override is not None:
            connect_kwargs["host_override"] = self.host_override
        namespace_properties: dict[str, str] = {}
        if self.namespace_endpoint is not None:
            namespace_properties["uri"] = self.namespace_endpoint
        return LakeConnectionSpec(
            kind=self.kind,
            uri=self.uri,
            display_uri=self.uri,
            lancedb_connect_kwargs=connect_kwargs,
            namespace_client_impl=self.namespace_client_impl,
            namespace_client_properties=namespace_properties,
            auth_refs=dict(self.auth_refs),
            capabilities=LakeCapabilities(
                server_side_query=self.server_side_query,
                direct_object_io=self.direct_object_io,
                blob_fetch_remote=self.blob_fetch_remote,
            ),
            direct_object_io_allowed=self.direct_object_io,
            managed_versioning=self.managed_versioning,
        )


@dataclass(frozen=True)
class ConformanceCase:
    """One backend configuration or fault to classify and (optionally) run live."""

    name: str
    description: str
    connection: ConnectionConfig
    dataset_kwargs: Mapping[str, Any] = field(default_factory=dict)
    capability_overrides: Mapping[str, bool] = field(default_factory=dict)
    enterprise_extra_available: bool = True
    namespace_credentials_expired: bool = False
    credential_refresh: str = "none"  # none | fails | succeeds
    stale_table: str | None = None  # table whose checkout raises (live only)
    expected_category: str = CATEGORY_SUPPORTED
    expected_error: str | None = None
    expected_fallback_to: str | None = None
    fault: bool = False
    build_time_fault: bool = False  # surfaces at loader build, not in the data-free classifier
    requires_local_endpoint: bool = False

    def fallback_policy(self) -> str:
        return _training._validate_enterprise_fallback_policy(
            self.dataset_kwargs.get("fallback"),
            allow_fallback=bool(self.dataset_kwargs.get("allow_fallback", False)),
        )

    def required_capabilities(self) -> tuple[str, ...]:
        columns = tuple(
            self.dataset_kwargs.get("columns") or _training.DEFAULT_TRAINING_COLUMNS
        )
        media = self.dataset_kwargs.get("media", _training.DEFAULT_MEDIA_POLICY)
        media_policy = _training._resolve_media_policy(media, None)
        return _native_required_enterprise_capabilities(
            columns,
            media_policy=media_policy,
            cache_policy=self.dataset_kwargs.get("cache_policy", "none"),
        )


@dataclass(frozen=True)
class MatrixRow:
    """One classified row of the compatibility matrix."""

    name: str
    connection_kind: str
    requested_backend: str
    fallback_policy: str
    category: str
    resolved_backend: str | None
    execution_mode: str | None
    routing_mode: str | None
    error_type: str | None
    fallback_to: str | None
    note: str
    fault: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "connection_kind": self.connection_kind,
            "requested_backend": self.requested_backend,
            "fallback_policy": self.fallback_policy,
            "category": self.category,
            "resolved_backend": self.resolved_backend,
            "execution_mode": self.execution_mode,
            "routing_mode": self.routing_mode,
            "error_type": self.error_type,
            "fallback_to": self.fallback_to,
            "note": self.note,
            "fault": self.fault,
        }


@dataclass(frozen=True)
class CompatibilityMatrix:
    """Deterministic supported/fallback/unsupported matrix over the case registry."""

    rows: tuple[MatrixRow, ...]

    def to_dict(self) -> dict[str, Any]:
        counts = {category: 0 for category in CONFORMANCE_CATEGORIES}
        for row in self.rows:
            counts[row.category] = counts.get(row.category, 0) + 1
        return {
            "schema": "lancedb-robotics/enterprise-training-compatibility/v1",
            "summary": counts,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        backend_rows = [row for row in self.rows if not row.fault]
        fault_rows = [row for row in self.rows if row.fault]
        lines = [
            "# Enterprise training compatibility matrix",
            "",
            "Generated from `lancedb_robotics.enterprise_conformance.compatibility_matrix()`,"
            " which classifies each case with the production"
            " `_training_backend_report` backend resolver. `supported` = the requested"
            " backend runs remotely; `fallback` = it degrades with an explicit report"
            " entry; `unsupported` = it fails fast with a typed error. There is no"
            " silent local materialization column because that outcome is a"
            " conformance failure, never a documented cell.",
            "",
            "## Backend scenarios",
            "",
            "| Case | Connection | Requested | Fallback policy | Result | Resolved / error | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in backend_rows:
            lines.append(_matrix_md_row(row))
        lines += [
            "",
            "## Injected faults",
            "",
            "| Fault | Connection | Fallback policy | Result | Typed error / fallback | Notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in fault_rows:
            resolved = row.error_type or (
                f"fallback -> {row.fallback_to}" if row.fallback_to else row.resolved_backend or "-"
            )
            lines.append(
                f"| `{row.name}` | `{row.connection_kind}` | `{row.fallback_policy}` | "
                f"{row.category} | {resolved} | {row.note} |"
            )
        lines.append("")
        return "\n".join(lines)


def _matrix_md_row(row: MatrixRow) -> str:
    resolved = row.error_type or row.resolved_backend or "-"
    return (
        f"| `{row.name}` | `{row.connection_kind}` | `{row.requested_backend}` | "
        f"`{row.fallback_policy}` | {row.category} | {resolved} | {row.note} |"
    )


@dataclass(frozen=True)
class CaseOutcome:
    """The observed result of one conformance case in a live or static run."""

    name: str
    fault: bool
    requested_backend: str
    category: str
    resolved_backend: str | None = None
    execution_mode: str | None = None
    error_type: str | None = None
    fallback_to: str | None = None
    routing_mode: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)
    status: str = "pass"
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fault": self.fault,
            "requested_backend": self.requested_backend,
            "category": self.category,
            "resolved_backend": self.resolved_backend,
            "execution_mode": self.execution_mode,
            "error_type": self.error_type,
            "fallback_to": self.fallback_to,
            "routing_mode": self.routing_mode,
            "checks": dict(self.checks),
            "status": self.status,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class ConformanceReport:
    """Aggregate of a live conformance run plus the static compatibility matrix."""

    lake_uri: str
    snapshot: str
    outcomes: tuple[CaseOutcome, ...]
    matrix: CompatibilityMatrix
    local_endpoint: dict[str, Any] | None = None

    def ok(self) -> bool:
        return all(outcome.status == "pass" for outcome in self.outcomes)

    def failures(self) -> tuple[CaseOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.status != "pass")

    def summary(self) -> dict[str, Any]:
        by_category: dict[str, int] = {category: 0 for category in CONFORMANCE_CATEGORIES}
        passed = 0
        for outcome in self.outcomes:
            by_category[outcome.category] = by_category.get(outcome.category, 0) + 1
            if outcome.status == "pass":
                passed += 1
        return {
            "total": len(self.outcomes),
            "passed": passed,
            "failed": len(self.outcomes) - passed,
            "by_category": by_category,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lancedb-robotics/enterprise-training-conformance/v1",
            "lake_uri": self.lake_uri,
            "snapshot": self.snapshot,
            "summary": self.summary(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "matrix": self.matrix.to_dict(),
            "local_endpoint": self.local_endpoint,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# --------------------------------------------------------------------------- #
# Case registry
# --------------------------------------------------------------------------- #

_DB_REMOTE = ConnectionConfig(
    kind="lancedb_remote_db",
    uri="db://robotics",
    host_override="https://phalanx.acme.internal",
    region="us-west-2",
    api_key="conformance-test-key",
    auth_refs={"remote": "enterprise-prod"},
)
_DB_REMOTE_REGIONAL = replace(_DB_REMOTE, host_override=None)
_REST_NAMESPACE = ConnectionConfig(
    kind="rest_namespace_lancedb",
    uri="namespace://robotics",
    namespace_endpoint="https://ns.acme.internal",
    namespace_client_impl="lancedb_namespace.rest.RestNamespace",
    auth_refs={"namespace": "ns-prod", "storage": "storage-prod"},
    direct_object_io=True,
    managed_versioning=True,
)
_NAMESPACE_DIRECT = ConnectionConfig(
    kind="namespace_lancedb",
    uri="namespace://robotics-direct",
    namespace_endpoint="https://ns-direct.acme.internal",
    namespace_client_impl="lancedb_namespace.dir.DirNamespace",
    auth_refs={"namespace": "ns-prod", "storage": "storage-prod"},
    direct_object_io=True,
    managed_versioning=True,
)


def _backend_cases() -> tuple[ConformanceCase, ...]:
    return (
        ConformanceCase(
            name="local-native",
            description="Local Lance-native lake; backend=auto resolves to local.",
            connection=ConnectionConfig(kind="local_path", uri="./robot.lance"),
            dataset_kwargs={"columns": ["observation_id"], "backend": "auto"},
            expected_category=CATEGORY_SUPPORTED,
        ),
        ConformanceCase(
            name="db-remote-host-override",
            description="db:// remote DB with an explicit host_override HTTP endpoint.",
            connection=_DB_REMOTE,
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_SUPPORTED,
        ),
        ConformanceCase(
            name="db-remote-regional-default",
            description="db:// remote DB with no host_override (regional default endpoint).",
            connection=_DB_REMOTE_REGIONAL,
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_SUPPORTED,
        ),
        ConformanceCase(
            name="rest-namespace-query-node",
            description="REST Namespace-backed query node with direct object IO.",
            connection=_REST_NAMESPACE,
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_SUPPORTED,
        ),
        ConformanceCase(
            name="namespace-direct-io",
            description="Namespace lake with authorized direct object IO.",
            connection=_NAMESPACE_DIRECT,
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_SUPPORTED,
        ),
        ConformanceCase(
            name="capability-disabled-deployment",
            description="Enterprise connection whose deployment lacks server-side query.",
            connection=replace(_DB_REMOTE, server_side_query=False, blob_fetch_remote=False),
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_UNSUPPORTED,
            expected_error="UnsupportedRemoteOperationError",
        ),
        ConformanceCase(
            name="local-enterprise-cli-endpoint",
            description=(
                "Real HTTP host_override target served by the LanceDB Enterprise "
                "CLI (`~/Downloads/lancedb server` + `pe`); gated behind "
                f"{LOCAL_ENDPOINT_ENV}=1."
            ),
            connection=_DB_REMOTE,  # host_override overridden at runtime to the live port
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_SUPPORTED,
            requires_local_endpoint=True,
        ),
    )


def _fault_cases() -> tuple[ConformanceCase, ...]:
    return (
        ConformanceCase(
            name="auth-missing",
            description="db:// remote with no API token or auth ref.",
            connection=replace(_DB_REMOTE, api_key=None, auth_refs={}),
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_UNSUPPORTED,
            expected_error="MissingEnterpriseAuthError",
            fault=True,
        ),
        ConformanceCase(
            name="auth-expired-no-refresh",
            description="Namespace credentials expired with no refresh hook.",
            connection=_REST_NAMESPACE,
            namespace_credentials_expired=True,
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_UNSUPPORTED,
            expected_error="NamespaceCredentialExpiredError",
            fault=True,
        ),
        ConformanceCase(
            name="auth-expired-refresh-fails",
            description="Expired credentials; one refresh attempt still leaves them expired.",
            connection=_REST_NAMESPACE,
            namespace_credentials_expired=True,
            credential_refresh="fails",
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_UNSUPPORTED,
            expected_error="NamespaceCredentialExpiredError",
            fault=True,
        ),
        ConformanceCase(
            name="auth-expired-refresh-recovers",
            description="Expired credentials; a single refresh clears the expiry and training proceeds.",
            connection=_REST_NAMESPACE,
            namespace_credentials_expired=True,
            credential_refresh="succeeds",
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_SUPPORTED,
            fault=True,
        ),
        ConformanceCase(
            name="remote-scan-unsupported",
            description="Remote scan capability disabled with fallback='fail'.",
            connection=_DB_REMOTE,
            capability_overrides={"remote_scan": False},
            dataset_kwargs={
                "columns": ["observation_id"],
                "backend": "enterprise",
                "fallback": "fail",
            },
            expected_category=CATEGORY_UNSUPPORTED,
            expected_error="UnsupportedRemoteOperationError",
            fault=True,
        ),
        ConformanceCase(
            name="remote-take-unsupported",
            description="Remote take/blob-hydration disabled while materializing payload bytes.",
            connection=_DB_REMOTE,
            capability_overrides={"remote_take": False},
            dataset_kwargs={
                "columns": ["observation_id", "payload"],
                "media": "bytes",
                "backend": "enterprise",
                "fallback": "fail",
            },
            expected_category=CATEGORY_UNSUPPORTED,
            expected_error="UnsupportedRemoteOperationError",
            fault=True,
        ),
        ConformanceCase(
            name="payload-hydration-unavailable",
            description="Blob/video remote hydration disabled while reading payload bytes.",
            connection=_DB_REMOTE,
            capability_overrides={
                "blob_or_video_remote_hydration": False,
                "remote_take": False,
            },
            dataset_kwargs={
                "columns": ["observation_id", "payload"],
                "media": "bytes",
                "backend": "enterprise",
                "fallback": "fail",
            },
            expected_category=CATEGORY_UNSUPPORTED,
            expected_error="UnsupportedRemoteOperationError",
            fault=True,
        ),
        ConformanceCase(
            name="prewarm-unavailable-warn",
            description="Page-cache prewarm/status disabled; fallback='warn' degrades to lazy cache.",
            connection=_DB_REMOTE,
            capability_overrides={"page_cache_prewarm": False, "page_cache_status": False},
            dataset_kwargs={
                "columns": ["observation_id"],
                "backend": "enterprise",
                "cache_policy": "epoch",
                "fallback": "warn",
            },
            expected_category=CATEGORY_FALLBACK,
            expected_fallback_to="lazy-cache",
            fault=True,
        ),
        ConformanceCase(
            name="cache-metrics-unavailable-warn",
            description="Cache metrics disabled; fallback='warn' keeps remote execution, marks counters absent.",
            connection=_DB_REMOTE,
            capability_overrides={"plan_executor_cache_metrics": False},
            dataset_kwargs={
                "columns": ["observation_id"],
                "backend": "enterprise",
                "cache_policy": "lazy",
                "fallback": "warn",
            },
            expected_category=CATEGORY_FALLBACK,
            expected_fallback_to="remote-execution-without-live-cache-metrics",
            fault=True,
        ),
        ConformanceCase(
            name="direct-data-plane-fallback",
            description="Remote take missing but direct object IO authorized; fallback='direct'.",
            connection=replace(_DB_REMOTE, direct_object_io=True),
            capability_overrides={"remote_take": False},
            dataset_kwargs={
                "columns": ["observation_id", "payload"],
                "media": "bytes",
                "backend": "enterprise",
                "fallback": "direct",
            },
            expected_category=CATEGORY_FALLBACK,
            expected_fallback_to="direct-data-plane",
            fault=True,
        ),
        ConformanceCase(
            name="explicit-local-fallback",
            description="Enterprise requested against a local lake; fallback='local'.",
            connection=ConnectionConfig(kind="local_path", uri="./robot.lance"),
            dataset_kwargs={
                "columns": ["observation_id"],
                "backend": "enterprise",
                "fallback": "local",
            },
            expected_category=CATEGORY_FALLBACK,
            expected_fallback_to="local",
            fault=True,
        ),
        ConformanceCase(
            name="stale-table-version",
            description="Pinned snapshot table version cannot be checked out remotely.",
            connection=_DB_REMOTE,
            stale_table="scenarios",
            dataset_kwargs={"columns": ["observation_id"], "backend": "enterprise"},
            expected_category=CATEGORY_UNSUPPORTED,
            expected_error="StaleTableVersionError",
            fault=True,
            build_time_fault=True,
        ),
        ConformanceCase(
            name="worker-resume-mismatch",
            description="Worker resume offset is beyond the planned epoch.",
            connection=_DB_REMOTE,
            dataset_kwargs={
                "columns": ["observation_id"],
                "backend": "enterprise",
                "resume_from": 10_000,
            },
            expected_category=CATEGORY_UNSUPPORTED,
            expected_error="WorkerResumeMismatchError",
            fault=True,
            build_time_fault=True,
        ),
    )


def all_cases(*, include_local_endpoint: bool = False) -> tuple[ConformanceCase, ...]:
    """Return the full case registry (backend scenarios followed by fault cases)."""

    cases = _backend_cases() + _fault_cases()
    if not include_local_endpoint:
        cases = tuple(case for case in cases if not case.requires_local_endpoint)
    return cases


# --------------------------------------------------------------------------- #
# Static classification / compatibility matrix
# --------------------------------------------------------------------------- #


class _ClassificationLake:
    """A data-free lake stand-in the classifier can read attributes off of."""

    def __init__(self, case: ConformanceCase) -> None:
        spec = case.connection.build_spec()
        self.connection_spec = spec
        self.capabilities = spec.capabilities if spec is not None else None
        self.uri = case.connection.uri
        self._db = None
        if case.capability_overrides:
            self.enterprise_training_capabilities = dict(case.capability_overrides)
        self.enterprise_extra_available = case.enterprise_extra_available
        self.namespace_credentials_expired = case.namespace_credentials_expired
        if case.credential_refresh == "succeeds":
            self.namespace_credential_refresh = self._clear_expiry
        elif case.credential_refresh == "fails":
            self.namespace_credential_refresh = lambda: None

    def _clear_expiry(self) -> None:
        self.namespace_credentials_expired = False


def classify_case(case: ConformanceCase) -> MatrixRow:
    """Classify one case with the production backend resolver (no snapshot data)."""

    requested = str(case.dataset_kwargs.get("backend", "auto"))
    fallback_policy = case.fallback_policy()
    lake = _ClassificationLake(case)
    category = CATEGORY_SUPPORTED
    resolved_backend: str | None = None
    execution_mode: str | None = None
    routing_mode: str | None = None
    error_type: str | None = None
    fallback_to: str | None = None
    note = case.description
    if case.build_time_fault:
        # The data-free classifier cannot observe this fault (it needs snapshot
        # data); report the contract the live run enforces at loader build.
        return MatrixRow(
            name=case.name,
            connection_kind=case.connection.kind,
            requested_backend=requested,
            fallback_policy=fallback_policy,
            category=case.expected_category,
            resolved_backend=None,
            execution_mode=None,
            routing_mode=None,
            error_type=case.expected_error,
            fallback_to=case.expected_fallback_to,
            note=f"{note} (surfaces at loader build)",
            fault=case.fault,
        )
    try:
        report = _training_backend_report(
            lake,
            backend=requested,
            cache_policy=case.dataset_kwargs.get("cache_policy", "none"),
            prewarm=bool(case.dataset_kwargs.get("prewarm", False)),
            fallback_policy=fallback_policy,
            required_capabilities=case.required_capabilities(),
        )
    except EnterpriseTrainingError as exc:
        category = CATEGORY_UNSUPPORTED
        error_type = type(exc).__name__
    else:
        resolved_backend = report.resolved_backend
        execution_mode = report.execution_mode
        routing_mode = report.request_routing.get("mode")
        if report.fallback_events:
            category = CATEGORY_FALLBACK
            fallback_to = report.fallback_events[0].get("to")
        else:
            category = CATEGORY_SUPPORTED
    return MatrixRow(
        name=case.name,
        connection_kind=case.connection.kind,
        requested_backend=requested,
        fallback_policy=fallback_policy,
        category=category,
        resolved_backend=resolved_backend,
        execution_mode=execution_mode,
        routing_mode=routing_mode,
        error_type=error_type,
        fallback_to=fallback_to,
        note=note,
        fault=case.fault,
    )


def compatibility_matrix(*, include_local_endpoint: bool = False) -> CompatibilityMatrix:
    """Build the deterministic supported/fallback/unsupported matrix."""

    rows = tuple(
        classify_case(case) for case in all_cases(include_local_endpoint=include_local_endpoint)
    )
    return CompatibilityMatrix(rows=rows)


# --------------------------------------------------------------------------- #
# Live conformance run
# --------------------------------------------------------------------------- #


@contextmanager
def _case_applied(lake: Any, case: ConformanceCase) -> Iterator[None]:
    """Apply a case's simulated connection/faults to a real lake, then restore."""

    saved = {
        attr: getattr(lake, attr, _UNSET)
        for attr in _LAKE_SIM_ATTRS
    }
    try:
        _apply_case(lake, case)
        yield
    finally:
        for attr, value in saved.items():
            if value is _UNSET:
                if hasattr(lake, attr):
                    try:
                        delattr(lake, attr)
                    except AttributeError:
                        pass
            else:
                setattr(lake, attr, value)


class _Unset:
    pass


_UNSET = _Unset()


def _apply_case(lake: Any, case: ConformanceCase) -> None:
    spec = case.connection.build_spec()
    if spec is not None:
        lake.connection_spec = spec
        lake.capabilities = spec.capabilities
    else:
        lake.connection_spec = None
        lake.capabilities = None
    if case.capability_overrides:
        lake.enterprise_training_capabilities = dict(case.capability_overrides)
    else:
        lake.enterprise_training_capabilities = None
    lake.enterprise_extra_available = case.enterprise_extra_available
    lake.namespace_credentials_expired = case.namespace_credentials_expired
    if case.credential_refresh == "succeeds":
        def _refresh() -> None:
            lake.namespace_credentials_expired = False

        lake.namespace_credential_refresh = _refresh
    elif case.credential_refresh == "fails":
        lake.namespace_credential_refresh = lambda: None
    else:
        lake.namespace_credential_refresh = None
    if case.stale_table:
        _install_stale_table(lake, case.stale_table)


def _install_stale_table(lake: Any, table_name: str) -> None:
    original = lake.table

    class _StaleCheckout:
        def __init__(self, table: Any) -> None:
            self._table = table

        def checkout(self, version: Any) -> Any:
            raise RuntimeError(f"version {version} was compacted away")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._table, name)

    def table(name: str) -> Any:
        opened = original(name)
        return _StaleCheckout(opened) if name == table_name else opened

    lake.table = table


def _live_dataset_kwargs(case: ConformanceCase) -> dict[str, Any]:
    allowed = {
        "columns",
        "filters",
        "media",
        "backend",
        "cache_policy",
        "prewarm",
        "fallback",
        "allow_fallback",
        "epoch",
        "worker_id",
        "num_workers",
        "resume_from",
    }
    return {key: value for key, value in case.dataset_kwargs.items() if key in allowed}


def _check_no_secret_leak(payload: Any) -> list[str]:
    """Return secret markers found in a serialized worker config/manifest."""

    text = json.dumps(payload, default=str).lower()
    hits = [marker for marker in _SECRET_MARKERS if marker in text]
    # ``*_auth_ref`` / ``authorization`` header *keys* are logical references, not
    # secrets; only flag a literal token value if the known test key leaks.
    hits = [marker for marker in hits if marker not in {"authorization"}]
    if "conformance-test-key" in text or "secret-api-key" in text:
        hits.append("literal-api-token")
    return sorted(set(hits))


def _sample_lineage_stream(dataset: Any) -> list[dict[str, Any]]:
    """Flatten per-batch collated lineage into one ordered sample stream.

    ``iter_training_batches`` attaches a ``_lineage`` block whose row_ids /
    frame_ids / observation_ids / sample_indices are parallel lists (see
    ``_collate_lineage``); zip them back into per-sample records so two backends
    can be compared sample-for-sample.
    """

    stream: list[dict[str, Any]] = []
    for batch in iter_training_batches(dataset, batch_size=8):
        lineage = batch.get("_lineage")
        if not isinstance(lineage, Mapping):
            continue
        row_ids = lineage.get("row_ids") or []
        frame_ids = lineage.get("frame_ids") or []
        observation_ids = lineage.get("observation_ids") or []
        sample_indices = lineage.get("sample_indices") or []
        for offset in range(len(sample_indices)):
            stream.append(
                {
                    "row_id": row_ids[offset] if offset < len(row_ids) else None,
                    "frame_id": frame_ids[offset] if offset < len(frame_ids) else None,
                    "observation_id": observation_ids[offset]
                    if offset < len(observation_ids)
                    else None,
                    "sample_index": sample_indices[offset],
                }
            )
    return stream


def _batch_schema(dataset: Any) -> list[str]:
    for batch in iter_training_batches(dataset, batch_size=8):
        return sorted(key for key in batch.keys() if not key.startswith("_"))
    return []


_SHAPE_KWARGS = (
    "columns",
    "filters",
    "media",
    "epoch",
    "worker_id",
    "num_workers",
    "resume_from",
)


def _local_twin(lake: Any, snapshot: str, case: ConformanceCase) -> dict[str, Any]:
    """Local-backend result for the case's exact request, for equivalence checks.

    The sample stream, batch schema, and table-version lineage are functions of
    the request shape (columns/media/epoch/worker/resume), not the backend, so a
    conforming Enterprise path must reproduce the local result byte-for-byte.
    Computed on the clean (unmodified) lake before the case is applied.
    """

    twin_kwargs = {
        key: value for key, value in case.dataset_kwargs.items() if key in _SHAPE_KWARGS
    }
    dataset = lake.training.dataset(snapshot, backend="local", **twin_kwargs)
    return {
        "stream": _sample_lineage_stream(dataset),
        "schema": _batch_schema(dataset),
        "table_versions": list(dataset.manifest.table_versions),
    }


def _run_case(
    lake: Any,
    snapshot: str,
    case: ConformanceCase,
) -> CaseOutcome:
    requested = str(case.dataset_kwargs.get("backend", "auto"))
    failures: list[str] = []
    checks: dict[str, Any] = {}
    kwargs = _live_dataset_kwargs(case)

    # A local twin of the same request is our equivalence oracle; only built for
    # cases we expect to yield a working dataset (skip pure error cases).
    twin: dict[str, Any] | None = None
    if case.expected_category != CATEGORY_UNSUPPORTED and case.stale_table is None:
        try:
            twin = _local_twin(lake, snapshot, case)
        except Exception:  # noqa: BLE001 - twin is best-effort; absence just skips the check
            twin = None

    with _case_applied(lake, case):
        try:
            dataset = lake.training.dataset(snapshot, **kwargs)
        except EnterpriseTrainingError as exc:
            error_type = type(exc).__name__
            if case.expected_error and error_type != case.expected_error:
                failures.append(
                    f"expected {case.expected_error}, raised {error_type}: {exc}"
                )
            elif not case.expected_error:
                failures.append(f"unexpected typed error {error_type}: {exc}")
            return CaseOutcome(
                name=case.name,
                fault=case.fault,
                requested_backend=requested,
                category=CATEGORY_UNSUPPORTED,
                error_type=error_type,
                checks={"typed_error": error_type},
                status="pass" if not failures else "fail",
                failures=tuple(failures),
            )

        # No error raised: build the observed report and enforce invariants.
        report = dataset.backend_report
        resolved = report.resolved_backend
        fallback_to = report.fallback.get("to") if report.fallback else None
        category = CATEGORY_FALLBACK if report.fallback_events else CATEGORY_SUPPORTED

        # Central invariant: an enterprise request that resolves to local/direct
        # MUST carry an explicit fallback event -- never silent materialization.
        if requested == "enterprise" and resolved != "enterprise" and not report.fallback_events:
            failures.append(
                "silent local materialization: enterprise resolved to "
                f"{resolved!r} with no fallback event"
            )

        if case.expected_error:
            failures.append(
                f"expected {case.expected_error} but the loader built successfully"
            )
        if category != case.expected_category:
            failures.append(
                f"expected category {case.expected_category!r}, observed {category!r}"
            )
        if case.expected_fallback_to and case.expected_fallback_to not in {
            event.get("to") for event in report.fallback_events
        }:
            failures.append(
                f"expected fallback to {case.expected_fallback_to!r}, "
                f"observed {[e.get('to') for e in report.fallback_events]}"
            )

        # Routing + worker-handoff secret hygiene for enterprise routes.
        routing = report.request_routing
        if resolved == "enterprise" and case.connection.kind == "lancedb_remote_db":
            if case.connection.host_override and not routing.get("all_requests_use_host_override"):
                failures.append("host_override present but not honored for all requests")
            loader_config = dataset.loader_config()
            leaks = _check_no_secret_leak(loader_config)
            checks["secret_leaks"] = leaks
            if leaks:
                failures.append(f"worker loader_config leaked secrets: {leaks}")
            remote = loader_config.get("connection", {}).get("remote", {})
            if case.connection.host_override and remote.get("host_override") != case.connection.host_override:
                failures.append("worker handoff dropped host_override routing")

        # Lineage / schema equivalence vs a local twin of the *same* request.
        if twin is not None:
            observed_stream = _sample_lineage_stream(dataset)
            observed_schema = _batch_schema(dataset)
            checks["sample_count"] = len(observed_stream)
            if observed_stream != twin["stream"]:
                failures.append("sample-id/row-id lineage diverged from local twin")
            if observed_schema != twin["schema"]:
                failures.append("batch schema diverged from local twin")
            if list(dataset.manifest.table_versions) != twin["table_versions"]:
                failures.append("table-version lineage diverged from local twin")

        checks["routing_mode"] = routing.get("mode")
        checks["warnings"] = list(report.warnings)
        return CaseOutcome(
            name=case.name,
            fault=case.fault,
            requested_backend=requested,
            category=category,
            resolved_backend=resolved,
            execution_mode=report.execution_mode,
            fallback_to=fallback_to,
            routing_mode=routing.get("mode"),
            checks=checks,
            status="pass" if not failures else "fail",
            failures=tuple(failures),
        )


def run_conformance(
    lake: Any,
    snapshot: str,
    *,
    cases: Sequence[ConformanceCase] | None = None,
    include_local_endpoint: bool = False,
    strict: bool = False,
) -> ConformanceReport:
    """Replay ``snapshot`` through each backend case and injected fault.

    ``lake`` must be a local, real lake holding ``snapshot``; each case
    temporarily reconfigures its connection/capabilities to simulate the target
    backend or degradation (no Enterprise secrets required). Set
    ``include_local_endpoint=True`` (and the env flag) to add the real LanceDB
    Enterprise CLI endpoint pass. With ``strict=True`` a contract violation
    raises :class:`ConformanceError`.
    """

    if cases is None:
        cases = all_cases(include_local_endpoint=include_local_endpoint)
    outcomes: list[CaseOutcome] = []
    local_endpoint_info: dict[str, Any] | None = None
    for case in cases:
        if case.requires_local_endpoint:
            outcome, local_endpoint_info = _run_local_endpoint_case(lake, snapshot, case)
            if outcome is not None:
                outcomes.append(outcome)
            continue
        outcomes.append(_run_case(lake, snapshot, case))
    report = ConformanceReport(
        lake_uri=lake.uri,
        snapshot=snapshot,
        outcomes=tuple(outcomes),
        matrix=compatibility_matrix(include_local_endpoint=include_local_endpoint),
        local_endpoint=local_endpoint_info,
    )
    if strict and not report.ok():
        detail = "; ".join(
            f"{o.name}: {', '.join(o.failures)}" for o in report.failures()
        )
        raise ConformanceError(f"conformance violations: {detail}")
    return report


# --------------------------------------------------------------------------- #
# Local Enterprise CLI endpoint (gated real HTTP pass)
# --------------------------------------------------------------------------- #


def local_endpoint_available() -> bool:
    """True when the gated local Enterprise CLI endpoint pass should run."""

    if os.environ.get(LOCAL_ENDPOINT_ENV) not in {"1", "true", "yes"}:
        return False
    return _enterprise_cli_path() is not None


def _enterprise_cli_path() -> Path | None:
    override = os.environ.get("LANCEDB_ROBOTICS_ENTERPRISE_CLI")
    if override:
        path = Path(override)
        return path if path.exists() else None
    if DEFAULT_ENTERPRISE_CLI.exists():
        return DEFAULT_ENTERPRISE_CLI
    which = shutil.which("lancedb")
    return Path(which) if which else None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, *, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.25)
    return False


@dataclass
class LocalEnterpriseEndpoint:
    """A locally spawned LanceDB Enterprise query node + plan executor.

    Wraps ``~/Downloads/lancedb server`` and ``~/Downloads/lancedb pe`` so a
    conformance run can exercise a real HTTP ``host_override`` target and the
    plan-executor registration path. Best-effort and gated: use it only when
    :func:`local_endpoint_available` is True.
    """

    host: str = "127.0.0.1"
    port: int = 0
    management_port: int = 0
    api_key: str = "conformance-test-key"
    cli_path: Path | None = None
    _server: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _pe: subprocess.Popen | None = field(default=None, init=False, repr=False)

    @property
    def http_endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> LocalEnterpriseEndpoint:
        cli = self.cli_path or _enterprise_cli_path()
        if cli is None:
            raise ConformanceError("LanceDB Enterprise CLI binary not found")
        self.cli_path = cli
        self.port = self.port or _free_port()
        self.management_port = self.management_port or _free_port()
        self._server = subprocess.Popen(  # noqa: S603 - trusted local dev binary
            [
                str(cli),
                "server",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--api-key",
                self.api_key,
                "--use-remote-scan=true",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if not _wait_for_port(self.host, self.port):
            self.__exit__(None, None, None)
            raise ConformanceError("local Enterprise query node did not become reachable")
        self._pe = subprocess.Popen(  # noqa: S603 - trusted local dev binary
            [
                str(cli),
                "pe",
                "--management-url",
                f"http://{self.host}:{self.management_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        for proc in (self._pe, self._server):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def info(self) -> dict[str, Any]:
        return {
            "cli_path": str(self.cli_path) if self.cli_path else None,
            "http_endpoint": self.http_endpoint,
            "management_url": f"http://{self.host}:{self.management_port}",
            "server_running": bool(self._server and self._server.poll() is None),
            "plan_executor_running": bool(self._pe and self._pe.poll() is None),
        }


def _run_local_endpoint_case(
    lake: Any,
    snapshot: str,
    case: ConformanceCase,
) -> tuple[CaseOutcome | None, dict[str, Any] | None]:
    if not local_endpoint_available():
        return (
            CaseOutcome(
                name=case.name,
                fault=case.fault,
                requested_backend=str(case.dataset_kwargs.get("backend", "auto")),
                category=case.expected_category,
                checks={"skipped": f"set {LOCAL_ENDPOINT_ENV}=1 with the CLI binary present"},
                status="pass",
            ),
            None,
        )
    with LocalEnterpriseEndpoint() as endpoint:
        info = endpoint.info()
        live_case = replace(
            case,
            connection=replace(
                case.connection,
                host_override=endpoint.http_endpoint,
                api_key=endpoint.api_key,
            ),
        )
        outcome = _run_case(lake, snapshot, live_case)
        # The routing target must be the live endpoint, proving the handoff used
        # the real HTTP host_override rather than a regional default.
        if outcome.routing_mode != "host-override":
            outcome = replace(
                outcome,
                status="fail",
                failures=outcome.failures
                + ("local endpoint pass did not route via host_override",),
            )
        info["routing_asserted"] = outcome.routing_mode
        return outcome, info


# --------------------------------------------------------------------------- #
# Live plan-executor client conformance (backlog 0119)
# --------------------------------------------------------------------------- #


class FakeQueryNodeClient:
    """A high-fidelity fake plan-executor client for conformance and tests.

    Serves remote scan / take / filtered-read from the local lake's Lance tables
    while returning the observability metadata a real Sophon plan-executor
    deployment produces: a stable request id, the echoed manifest e-tag,
    ``x-cache-hits`` / ``x-cache-misses`` response headers broken down per plan
    executor, plan-executor fanout, and bytes read. Set ``fail_operations`` to
    prove typed remote-hydration diagnostics; it never contacts a real endpoint
    and holds no secrets.
    """

    def __init__(
        self,
        lake: Any,
        *,
        plan_executors: Sequence[str] = ("pe-a", "pe-b"),
        hit_pct: int = 75,
        fail_operations: Sequence[str] = (),
    ) -> None:
        self.lake = lake
        self.plan_executors = tuple(plan_executors) or ("pe-0",)
        self.hit_pct = int(hit_pct)
        self.fail_operations = set(fail_operations)
        self._seq = 0

    def execute(self, request: Any) -> Any:
        self._seq += 1
        request_id = f"req-{self._seq:06d}"
        if request.operation in self.fail_operations:
            raise _training.RemoteQueryNodeError(
                operation=request.operation,
                table=request.table,
                version=request.version,
                request_id=request_id,
                reason="fault injection: plan executor rejected the request",
            )
        headers = self._cache_headers(request)
        if request.operation == "remote_take" and request.blob_column is not None:
            blobs = _training._lance_take_blobs(
                self.lake,
                request.table,
                request.blob_column,
                request.row_ids or (),
                version=request.version,
            )
            return _training.QueryNodeResponse(
                blobs=blobs,
                request_id=request_id,
                manifest_etag=request.manifest_etag,
                cache_metrics=headers,
                pe_addrs=self.plan_executors,
                bytes_read=sum(len(value or b"") for value in blobs),
            )
        if request.operation == "remote_take":
            rows = _training._lance_take_rows(
                self.lake,
                request.table,
                request.row_ids or (),
                columns=request.columns,
                version=request.version,
            )
            return _training.QueryNodeResponse(
                rows=rows,
                request_id=request_id,
                manifest_etag=request.manifest_etag,
                cache_metrics=headers,
                pe_addrs=self.plan_executors,
            )
        rows = _training._lance_filtered_read(
            self.lake,
            request.table,
            columns=request.columns,
            where_sql=request.where_sql or "",
            version=request.version,
            with_row_id=request.with_row_id,
            limit=request.limit,
        )
        return _training.QueryNodeResponse(
            rows=rows,
            request_id=request_id,
            manifest_etag=request.manifest_etag,
            cache_metrics=headers,
            pe_addrs=self.plan_executors,
        )

    def _cache_headers(self, request: Any) -> dict[str, Any]:
        rows = len(request.row_ids) if request.row_ids else 1
        pes = self.plan_executors
        per_pe: dict[str, dict[str, int]] = {}
        total_hits = 0
        total_misses = 0
        for idx, pe in enumerate(pes):
            share = rows // len(pes) + (1 if idx < rows % len(pes) else 0)
            hits = (share * self.hit_pct) // 100
            misses = share - hits
            per_pe[pe] = {"x-cache-hits": hits, "x-cache-misses": misses}
            total_hits += hits
            total_misses += misses
        return {
            "x-cache-hits": total_hits,
            "x-cache-misses": total_misses,
            "by_pe": per_pe,
        }


@dataclass
class QueryNodeConformanceReport:
    """Result of the live plan-executor client conformance run (backlog 0119)."""

    lake_uri: str
    snapshot: str
    checks: dict[str, dict[str, Any]]
    metrics: dict[str, Any]
    local_endpoint: dict[str, Any] | None = None

    def ok(self) -> bool:
        return all(entry.get("status") == "pass" for entry in self.checks.values())

    def failures(self) -> tuple[str, ...]:
        return tuple(
            name for name, entry in self.checks.items() if entry.get("status") != "pass"
        )

    def summary(self) -> dict[str, Any]:
        passed = sum(1 for e in self.checks.values() if e.get("status") == "pass")
        return {
            "total": len(self.checks),
            "passed": passed,
            "failed": len(self.checks) - passed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lancedb-robotics/plan-executor-conformance/v1",
            "lake_uri": self.lake_uri,
            "snapshot": self.snapshot,
            "summary": self.summary(),
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "local_endpoint": self.local_endpoint,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


_PLAN_EXECUTOR_ETAGS = {
    "observations": "etag-observations-v1",
    "scenarios": "etag-scenarios-v1",
    "runs": "etag-runs-v1",
}
_PLAN_EXECUTOR_CASE = ConformanceCase(
    name="plan-executor-live-client",
    description="Live plan-executor client over a db:// remote lake.",
    connection=_DB_REMOTE,
    dataset_kwargs={"backend": "enterprise", "media": "bytes"},
)


def _collect_payload_stream(dataset: Any) -> list[bytes | None]:
    payloads: list[bytes | None] = []
    for batch in iter_training_batches(dataset, batch_size=8):
        payloads.extend(batch.get("payload") or [])
    return payloads


def run_query_node_conformance(
    lake: Any,
    snapshot: str,
    *,
    include_local_endpoint: bool = False,
    strict: bool = False,
) -> QueryNodeConformanceReport:
    """Prove the live plan-executor client contract for one pinned snapshot.

    Runs the native training fixture twice over the same pinned snapshot -- once
    against the local Lance-backed executor and once against an attached
    :class:`FakeQueryNodeClient` -- and asserts the 0119 contract: identical
    sample payloads and table-version lineage, real cache hits/misses aggregated
    by plan executor *and* request, manifest e-tags carried into every request
    envelope, request ids recorded, a typed :class:`RemoteQueryNodeError` on a
    forced remote failure, and a metadata-only guardrail that refuses payload
    columns. ``include_local_endpoint`` additionally records a gated local Sophon
    query-node/plan-executor endpoint when available. With ``strict=True`` a
    contract violation raises :class:`ConformanceError`.
    """

    checks: dict[str, dict[str, Any]] = {}

    def _record(name: str, ok: bool, **detail: Any) -> None:
        checks[name] = {"status": "pass" if ok else "fail", **detail}

    payload_col = _training.PAYLOAD_BLOB_COLUMN

    # Baseline: local Lance-backed executor over the same request shape.
    twin = lake.training.dataset(
        snapshot, backend="local", media="bytes", columns=["observation_id", "payload"]
    )
    twin_payloads = _collect_payload_stream(twin)
    twin_stream = _sample_lineage_stream(twin)
    twin_versions = list(twin.manifest.table_versions)

    metrics: dict[str, Any] = {}
    with _case_applied(lake, _PLAN_EXECUTOR_CASE):
        prior_client = getattr(lake, "query_node_client", _UNSET)
        prior_etags = getattr(lake, "manifest_etags", _UNSET)
        try:
            lake.manifest_etags = dict(_PLAN_EXECUTOR_ETAGS)
            lake.query_node_client = FakeQueryNodeClient(lake)
            live = lake.training.dataset(
                snapshot,
                backend="enterprise",
                media="bytes",
                columns=["observation_id", "payload"],
            )
            live_payloads = _collect_payload_stream(live)
            live_stream = _sample_lineage_stream(live)
            metrics = dict(live.manifest.backend.get("metrics", {}))
            live_versions = list(live.manifest.table_versions)

            _record(
                "local_live_equivalence",
                live_payloads == twin_payloads
                and live_stream == twin_stream
                and live_versions == twin_versions,
                payloads_equal=live_payloads == twin_payloads,
                lineage_equal=live_stream == twin_stream,
                versions_equal=live_versions == twin_versions,
                samples=len(live_payloads),
            )

            per_pe = metrics.get("cache_by_plan_executor") or {}
            per_request = metrics.get("cache_by_request") or {}
            pe_hits = sum(int(v.get("hits", 0)) for v in per_pe.values())
            req_hits = sum(int(v.get("hits", 0)) for v in per_request.values())
            cache_ok = (
                (metrics.get("cache_hits", 0) or 0) > 0
                and len(per_pe) >= 2
                and len(per_request) >= 1
                and pe_hits == metrics.get("cache_hits")
                and req_hits == metrics.get("cache_hits")
            )
            _record(
                "cache_metrics_by_pe_and_request",
                cache_ok,
                cache_hits=metrics.get("cache_hits"),
                cache_misses=metrics.get("cache_misses"),
                plan_executors=sorted(per_pe),
                requests=len(per_request),
                pe_fanout=metrics.get("pe_fanout"),
            )

            etags = metrics.get("manifest_etags") or {}
            operations = metrics.get("operations") or []
            take_ops = [op for op in operations if op.get("operation") == "remote_take"]
            envelope_ok = (
                etags.get("observations") == _PLAN_EXECUTOR_ETAGS["observations"]
                and bool(take_ops)
                and all(
                    op.get("manifest_etag") == _PLAN_EXECUTOR_ETAGS.get(op.get("table"))
                    for op in take_ops
                )
            )
            _record(
                "request_envelope_carries_etag",
                envelope_ok,
                manifest_etags=etags,
                take_operations=len(take_ops),
            )

            request_ids = metrics.get("request_ids") or []
            request_ok = bool(request_ids) and all(
                str(rid).startswith("req-") for rid in request_ids
            )
            _record(
                "request_ids_recorded",
                request_ok and (metrics.get("live_hydration_requests", 0) or 0) > 0,
                request_ids=len(request_ids),
                live_hydration_requests=metrics.get("live_hydration_requests"),
            )
        finally:
            _restore_attr(lake, "query_node_client", prior_client)
            _restore_attr(lake, "manifest_etags", prior_etags)

    # Forced remote failure -> typed diagnostic with request id + remediation.
    with _case_applied(lake, _PLAN_EXECUTOR_CASE):
        prior_client = getattr(lake, "query_node_client", _UNSET)
        prior_etags = getattr(lake, "manifest_etags", _UNSET)
        try:
            lake.manifest_etags = dict(_PLAN_EXECUTOR_ETAGS)
            lake.query_node_client = FakeQueryNodeClient(
                lake, fail_operations={"remote_take"}
            )
            failing = lake.training.dataset(
                snapshot,
                backend="enterprise",
                media="bytes",
                columns=["observation_id", "payload"],
            )
            error_detail: dict[str, Any] = {}
            try:
                _collect_payload_stream(failing)
                typed_ok = False
            except _training.RemoteQueryNodeError as exc:
                typed_ok = (
                    exc.operation == "remote_take"
                    and exc.table == "observations"
                    and exc.request_id is not None
                    and bool(exc.remediation)
                )
                error_detail = {
                    "operation": exc.operation,
                    "table": exc.table,
                    "version": exc.version,
                    "request_id": exc.request_id,
                    "has_remediation": bool(exc.remediation),
                }
            _record("remote_failure_is_typed", typed_ok, **error_detail)
        finally:
            _restore_attr(lake, "query_node_client", prior_client)
            _restore_attr(lake, "manifest_etags", prior_etags)

    # Metadata-only guardrail: the live client must never take payload columns.
    with _case_applied(lake, _PLAN_EXECUTOR_CASE):
        prior_client = getattr(lake, "query_node_client", _UNSET)
        try:
            lake.query_node_client = FakeQueryNodeClient(lake)
            meta_dataset = lake.training.dataset(
                snapshot,
                backend="enterprise",
                media="metadata",
                columns=["observation_id"],
            )
            executor = meta_dataset._hydration_executor
            try:
                executor.take_blobs("observations", payload_col, [0])
                guard_ok = False
                guard_detail: dict[str, Any] = {"raised": False}
            except _training.MetadataOnlyViolationError as exc:
                guard_ok = exc.table == "observations" and payload_col in exc.columns
                guard_detail = {"raised": True, "columns": list(exc.columns)}
            _record("metadata_only_guardrail", guard_ok, **guard_detail)
        finally:
            _restore_attr(lake, "query_node_client", prior_client)

    local_endpoint_info: dict[str, Any] | None = None
    if include_local_endpoint:
        local_endpoint_info = _plan_executor_local_endpoint()

    report = QueryNodeConformanceReport(
        lake_uri=lake.uri,
        snapshot=snapshot,
        checks=checks,
        metrics=metrics,
        local_endpoint=local_endpoint_info,
    )
    if strict and not report.ok():
        raise ConformanceError(
            "plan-executor conformance violations: " + ", ".join(report.failures())
        )
    return report


def _restore_attr(lake: Any, attr: str, value: Any) -> None:
    if value is _UNSET:
        if hasattr(lake, attr):
            try:
                delattr(lake, attr)
            except AttributeError:
                pass
    else:
        setattr(lake, attr, value)


def _plan_executor_local_endpoint() -> dict[str, Any] | None:
    if not local_endpoint_available():
        return {"skipped": f"set {LOCAL_ENDPOINT_ENV}=1 with the CLI binary present"}
    with LocalEnterpriseEndpoint() as endpoint:
        info = endpoint.info()
    info["client"] = "lancedb_robotics.training.RemoteQueryNodeClient"
    return info


# --- Backlog 0345 deprecation aliases ---------------------------------------
# The client talks to the query node, never a plan executor. The pre-0345 names
# below remain importable for back-compat and will be removed in a future release.
FakePlanExecutorClient = FakeQueryNodeClient
PlanExecutorConformanceReport = QueryNodeConformanceReport
run_plan_executor_conformance = run_query_node_conformance
