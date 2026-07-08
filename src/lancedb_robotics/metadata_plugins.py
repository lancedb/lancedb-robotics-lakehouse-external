"""Metadata-integration plugin contract and conformance suite (backlog 0106).

Backlog 0064 introduced optional external-metadata adapters (OpenLineage,
DataHub, MLflow, W&B, ...) and 0101/0104 shipped the first concrete ones
(tracker manifest sync, OpenLineage/DataHub emitter delivery). Real deployments
add in-house emitters, catalog sync jobs, and tracker importers. Those adapters
need a *stable contract* they can evolve against outside the base package,
without breaking lineage identity, auth handling, dry-run behavior, or failure
recording.

This module is that contract. It defines:

- **Adapter families** -- ``lineage-emitter``, ``reference-importer``, and
  ``manifest-sync`` -- expressed as small ABC mixins a plugin subclasses.
- **Standard value types** -- capabilities, dependency probes, auth
  requirements, and structured emit results -- so every adapter reports the same
  shapes to the registry, the CLI, and audit.
- **A plugin registry** that never imports an optional dependency during base
  package import. Adapters register a *factory* and probe availability with
  ``importlib.util.find_spec`` (a metadata lookup, not an import).
- **A conformance suite** that a toy plugin can pass with zero optional
  dependencies, and that rejects adapters which mutate canonical artifact ids,
  drop reversible URNs, leak auth into persisted rows, or fail to record
  structured failure state.

The built-in adapters here are thin wrappers over the existing
``lineage_integrations`` (0064/0104/0105) and ``tracker_sync`` (0101) surfaces;
this module adds the *contract*, not a second implementation. Auth is always
runtime configuration (:class:`AdapterAuth`) and is never written into canonical
lineage or manifest rows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Families + errors.
# ---------------------------------------------------------------------------

FAMILY_LINEAGE_EMITTER = "lineage-emitter"
FAMILY_REFERENCE_IMPORTER = "reference-importer"
FAMILY_MANIFEST_SYNC = "manifest-sync"

KNOWN_FAMILIES = (
    FAMILY_LINEAGE_EMITTER,
    FAMILY_REFERENCE_IMPORTER,
    FAMILY_MANIFEST_SYNC,
)


class MetadataPluginError(Exception):
    """Raised when a metadata-integration plugin violates the adapter contract."""


# ---------------------------------------------------------------------------
# Standard value types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterCapabilities:
    """What an adapter can do. The conformance suite runs only the checks that a
    declared capability implies, and rejects a declaration whose interface method
    is missing."""

    dry_run: bool = False
    emit: bool = False
    retry: bool = False
    reversible_urns: bool = False
    import_bundle: bool = False
    live_fetch: bool = False
    paged: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "dry_run": self.dry_run,
            "emit": self.emit,
            "retry": self.retry,
            "reversible_urns": self.reversible_urns,
            "import_bundle": self.import_bundle,
            "live_fetch": self.live_fetch,
            "paged": self.paged,
        }


@dataclass(frozen=True)
class DependencyStatus:
    """Result of probing whether an adapter's optional dependency is importable.

    Carries every field the acceptance criteria require: adapter name, the
    module/package it needs, the optional extra or plugin that provides it, and a
    human-actionable install hint. ``available`` reflects a metadata-only probe
    (:func:`importlib.util.find_spec`) -- never an import.
    """

    adapter: str
    available: bool
    module: str | None
    optional_extra: str | None
    plugin: str | None
    native: bool
    install_hint: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "available": self.available,
            "module": self.module,
            "optional_extra": self.optional_extra,
            "plugin": self.plugin,
            "native": self.native,
            "install_hint": self.install_hint,
            "message": self.message,
        }


@dataclass(frozen=True)
class AdapterDependency:
    """Declares an adapter's optional runtime dependency (or that it is native).

    ``native`` adapters (e.g. the generic JSON bundle) need nothing installed.
    Everything else names an importable ``module`` plus how to get it: an
    ``optional_extra`` of this package, and/or a third-party ``plugin``
    distribution for in-house adapters.
    """

    adapter: str
    module: str | None = None
    optional_extra: str | None = None
    plugin: str | None = None
    native: bool = False

    def is_available(self) -> bool:
        """True when the module can be imported. Uses ``find_spec`` so probing a
        missing optional dependency never imports it (and never raises)."""

        if self.native:
            return True
        if not self.module:
            return False
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def install_hint(self) -> str:
        if self.native:
            return "no optional dependency required"
        options: list[str] = []
        if self.optional_extra:
            options.append(f"pip install 'lancedb-robotics[{self.optional_extra}]'")
        if self.plugin:
            options.append(f"pip install {self.plugin}")
        if not options and self.module:
            options.append(f"install a distribution providing module {self.module!r}")
        return " or ".join(options) if options else "install the adapter's provider"

    def probe(self) -> DependencyStatus:
        available = self.is_available()
        if self.native:
            message = f"adapter {self.adapter!r} is native (no optional dependency)"
        elif available:
            message = f"adapter {self.adapter!r} dependency module {self.module!r} is importable"
        else:
            message = (
                f"optional integration adapter {self.adapter!r} is not installed; "
                f"{self.install_hint()}"
            )
        return DependencyStatus(
            adapter=self.adapter,
            available=available,
            module=self.module,
            optional_extra=self.optional_extra,
            plugin=self.plugin,
            native=self.native,
            install_hint=self.install_hint(),
            message=message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "module": self.module,
            "optional_extra": self.optional_extra,
            "plugin": self.plugin,
            "native": self.native,
            "install_hint": self.install_hint(),
        }


@dataclass(frozen=True)
class AuthRequirement:
    """One runtime auth input an adapter accepts. Documentation only -- the value
    is always supplied at call time via :class:`AdapterAuth` and never persisted."""

    key: str
    description: str
    env_var: str | None = None
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "env_var": self.env_var,
            "required": self.required,
        }


@dataclass(frozen=True)
class AdapterAuth:
    """Runtime-only auth/config for an emit call.

    Everything here is ephemeral: endpoint, headers, a resolved ``auth_ref``, or a
    fully-constructed ``client``. None of it is ever written into canonical
    lineage or manifest rows -- the conformance suite asserts exactly that.
    """

    endpoint_url: str | None = None
    auth_ref: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    client: Any | None = None
    target: str | None = None

    def redacted(self) -> dict[str, Any]:
        """A log/telemetry-safe view: which auth inputs are present, never values."""

        return {
            "endpoint_url": self.endpoint_url,
            "auth_ref": self.auth_ref,
            "header_keys": sorted(self.headers.keys()),
            "has_client": self.client is not None,
            "target": self.target,
        }

    def secret_markers(self) -> set[str]:
        """String values that must never appear in persisted rows or payloads.

        Used by the conformance suite to prove auth non-persistence: every header
        value plus the raw ``auth_ref`` token.
        """

        markers: set[str] = set()
        for value in self.headers.values():
            text = str(value).strip()
            if text:
                markers.add(text)
                if text.lower().startswith("bearer "):
                    markers.add(text[len("bearer ") :].strip())
        if self.auth_ref:
            markers.add(str(self.auth_ref).strip())
        return {marker for marker in markers if marker}


@dataclass(frozen=True)
class AttemptRecord:
    """One payload's delivery outcome -- the audit/retry unit."""

    payload_digest: str
    status: str  # delivered | already-delivered | failed
    error: str | None = None
    remote_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_digest": self.payload_digest,
            "status": self.status,
            "error": self.error,
            "remote_ids": list(self.remote_ids),
        }


@dataclass(frozen=True)
class EmitResult:
    """Structured result of an emit/retry call, suitable for audit and retry."""

    adapter: str
    target: str
    mode: str  # emit | retry
    status: str
    attempts: tuple[AttemptRecord, ...]

    @property
    def delivered(self) -> int:
        return sum(1 for a in self.attempts if a.status == "delivered")

    @property
    def already_delivered(self) -> int:
        return sum(1 for a in self.attempts if a.status == "already-delivered")

    @property
    def failed(self) -> int:
        return sum(1 for a in self.attempts if a.status == "failed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "target": self.target,
            "mode": self.mode,
            "status": self.status,
            "delivered": self.delivered,
            "already_delivered": self.already_delivered,
            "failed": self.failed,
            "attempts": [a.to_dict() for a in self.attempts],
        }

    @classmethod
    def from_delivery_report(cls, report: Any, *, adapter: str) -> EmitResult:
        """Adapt a :class:`lineage_integrations.LineageDeliveryReport`."""

        attempts = tuple(
            AttemptRecord(
                payload_digest=a.payload_digest,
                status=a.status,
                error=a.error,
                remote_ids=tuple(a.remote_response_ids),
            )
            for a in report.attempts
        )
        return cls(
            adapter=adapter,
            target=report.target,
            mode=report.mode,
            status=report.status,
            attempts=attempts,
        )


# ---------------------------------------------------------------------------
# Adapter base + capability mixins.
# ---------------------------------------------------------------------------


class MetadataAdapter(ABC):
    """Base class for every metadata-integration adapter.

    Subclasses set ``name`` and ``family`` and implement :meth:`capabilities`
    and :meth:`dependency`. Capability mixins add the family-specific methods.
    """

    name: str = ""
    family: str = ""

    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...

    @abstractmethod
    def dependency(self) -> AdapterDependency: ...

    def auth_requirements(self) -> tuple[AuthRequirement, ...]:
        return ()

    def probe(self) -> DependencyStatus:
        return self.dependency().probe()

    def describe(self) -> dict[str, Any]:
        status = self.probe()
        return {
            "name": self.name,
            "family": self.family,
            "capabilities": self.capabilities().to_dict(),
            "dependency": self.dependency().to_dict(),
            "available": status.available,
            "auth_requirements": [req.to_dict() for req in self.auth_requirements()],
        }


class LineageEmitterMixin(ABC):
    """Emits lineage payloads to an external system.

    The single :meth:`build_payloads` is the shared contract for dry-run and real
    execution: :meth:`emit` MUST deliver exactly what ``build_payloads`` produced,
    so an inspected dry-run and a real emit never diverge.
    """

    @abstractmethod
    def build_payloads(self, lake: Any, *, refresh: bool = False) -> tuple[dict[str, Any], ...]: ...

    @abstractmethod
    def emit(
        self,
        lake: Any,
        *,
        auth: AdapterAuth,
        refresh: bool = False,
        retry: bool = False,
    ) -> EmitResult: ...


class ReferenceImporterMixin(ABC):
    """Maps canonical artifact ids to/from reversible external URNs."""

    @abstractmethod
    def to_external_urn(self, artifact_id: str) -> str: ...

    @abstractmethod
    def from_external_urn(self, urn: str) -> str: ...


class ManifestSyncMixin(ABC):
    """Provides an external manifest bundle for import into canonical tables."""

    @abstractmethod
    def load_bundle(self, **options: Any) -> Mapping[str, Any]: ...


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------

AdapterFactory = Callable[[], MetadataAdapter]


class MetadataPluginRegistry:
    """A registry of metadata-integration adapters keyed by name.

    Adapters register a zero-argument factory, so the registry never imports an
    optional dependency at registration or base-package import time -- a factory
    is only invoked when an adapter is actually requested, and even then the
    built-in adapters defer their heavy imports until a method runs.
    """

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, name: str, factory: AdapterFactory, *, replace: bool = False) -> None:
        key = _normalize_name(name)
        if not key:
            raise MetadataPluginError("plugin name is required")
        if key in self._factories and not replace:
            raise MetadataPluginError(
                f"metadata plugin {name!r} is already registered; pass replace=True to override"
            )
        self._factories[key] = factory

    def register_adapter(self, adapter: MetadataAdapter, *, replace: bool = False) -> None:
        self.register(adapter.name, lambda: adapter, replace=replace)

    def unregister(self, name: str) -> None:
        self._factories.pop(_normalize_name(name), None)

    def has(self, name: str) -> bool:
        return _normalize_name(name) in self._factories

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def get(self, name: str) -> MetadataAdapter:
        key = _normalize_name(name)
        factory = self._factories.get(key)
        if factory is None:
            raise MetadataPluginError(
                f"unknown metadata plugin {name!r}; registered: {list(self.names())}"
            )
        adapter = factory()
        if not isinstance(adapter, MetadataAdapter):
            raise MetadataPluginError(
                f"metadata plugin {name!r} factory returned {type(adapter).__name__}, "
                "expected a MetadataAdapter"
            )
        return adapter

    def adapters(self, *, family: str | None = None) -> tuple[MetadataAdapter, ...]:
        result = [self.get(name) for name in self.names()]
        if family is not None:
            result = [a for a in result if a.family == family]
        return tuple(result)

    def describe(self, *, family: str | None = None) -> tuple[dict[str, Any], ...]:
        return tuple(adapter.describe() for adapter in self.adapters(family=family))


_DEFAULT_REGISTRY: MetadataPluginRegistry | None = None


def default_registry() -> MetadataPluginRegistry:
    """Return the process-wide registry, registering built-ins on first use."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = MetadataPluginRegistry()
        _register_builtins(registry)
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY


def register_metadata_plugin(
    name: str,
    factory: AdapterFactory,
    *,
    replace: bool = False,
) -> None:
    """Register an in-house adapter factory in the default registry."""

    default_registry().register(name, factory, replace=replace)


def get_metadata_plugin(name: str) -> MetadataAdapter:
    return default_registry().get(name)


def list_metadata_plugins(*, family: str | None = None) -> tuple[dict[str, Any], ...]:
    return default_registry().describe(family=family)


def _normalize_name(name: str) -> str:
    return str(name or "").strip().lower()


# ---------------------------------------------------------------------------
# Built-in adapters -- thin wrappers over lineage_integrations / tracker_sync.
# Heavy imports stay inside methods so importing this module pulls no optional
# dependency and no lake machinery.
# ---------------------------------------------------------------------------


class _OpenLineageAdapter(MetadataAdapter, LineageEmitterMixin, ReferenceImporterMixin):
    name = "openlineage"
    family = FAMILY_LINEAGE_EMITTER

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            dry_run=True,
            emit=True,
            retry=True,
            reversible_urns=True,
            paged=True,
        )

    def dependency(self) -> AdapterDependency:
        return AdapterDependency(
            adapter=self.name,
            module="openlineage.client",
            optional_extra="openlineage",
            plugin=None,
            native=False,
        )

    def auth_requirements(self) -> tuple[AuthRequirement, ...]:
        return (
            AuthRequirement("endpoint_url", "OpenLineage/Marquez ingestion URL", "OPENLINEAGE_URL"),
            AuthRequirement("auth_ref", "env-scoped bearer/API key reference", None),
        )

    def build_payloads(self, lake: Any, *, refresh: bool = False) -> tuple[dict[str, Any], ...]:
        from lancedb_robotics.lineage_integrations import export_openlineage

        return tuple(export_openlineage(lake, refresh=refresh, dry_run=True).events)

    def emit(
        self,
        lake: Any,
        *,
        auth: AdapterAuth,
        refresh: bool = False,
        retry: bool = False,
    ) -> EmitResult:
        from lancedb_robotics.lineage_integrations import emit_openlineage

        report = emit_openlineage(
            lake,
            refresh=refresh,
            target=auth.target,
            endpoint_url=auth.endpoint_url,
            auth_ref=auth.auth_ref,
            headers=auth.headers,
            client=auth.client,
            adapter=self.name,
            retry=retry,
        )
        return EmitResult.from_delivery_report(report, adapter=self.name)

    def to_external_urn(self, artifact_id: str) -> str:
        from lancedb_robotics.lineage_integrations import external_artifact_urn

        return external_artifact_urn(artifact_id, backend="openlineage")

    def from_external_urn(self, urn: str) -> str:
        from lancedb_robotics.lineage_integrations import artifact_id_from_external_urn

        return artifact_id_from_external_urn(urn)


class _DataHubAdapter(MetadataAdapter, LineageEmitterMixin, ReferenceImporterMixin):
    name = "datahub"
    family = FAMILY_LINEAGE_EMITTER

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            dry_run=True,
            emit=True,
            retry=True,
            reversible_urns=True,
            paged=True,
        )

    def dependency(self) -> AdapterDependency:
        return AdapterDependency(
            adapter=self.name,
            module="datahub.emitter.rest_emitter",
            optional_extra="datahub",
            plugin=None,
            native=False,
        )

    def auth_requirements(self) -> tuple[AuthRequirement, ...]:
        return (
            AuthRequirement("endpoint_url", "DataHub GMS server URL", "DATAHUB_GMS_URL"),
            AuthRequirement("auth_ref", "env-scoped bearer token reference", "DATAHUB_TOKEN"),
        )

    def build_payloads(self, lake: Any, *, refresh: bool = False) -> tuple[dict[str, Any], ...]:
        from lancedb_robotics.lineage_integrations import export_datahub

        return tuple(export_datahub(lake, refresh=refresh, dry_run=True).edges)

    def emit(
        self,
        lake: Any,
        *,
        auth: AdapterAuth,
        refresh: bool = False,
        retry: bool = False,
    ) -> EmitResult:
        from lancedb_robotics.lineage_integrations import emit_datahub

        report = emit_datahub(
            lake,
            refresh=refresh,
            target=auth.target,
            endpoint_url=auth.endpoint_url,
            auth_ref=auth.auth_ref,
            headers=auth.headers,
            client=auth.client,
            adapter=self.name,
            retry=retry,
        )
        return EmitResult.from_delivery_report(report, adapter=self.name)

    def to_external_urn(self, artifact_id: str) -> str:
        from lancedb_robotics.lineage_integrations import external_artifact_urn

        return external_artifact_urn(artifact_id, backend="datahub")

    def from_external_urn(self, urn: str) -> str:
        from lancedb_robotics.lineage_integrations import artifact_id_from_external_urn

        return artifact_id_from_external_urn(urn)


class _GenericManifestAdapter(MetadataAdapter, ManifestSyncMixin):
    name = "generic"
    family = FAMILY_MANIFEST_SYNC

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(import_bundle=True, dry_run=True)

    def dependency(self) -> AdapterDependency:
        return AdapterDependency(adapter=self.name, native=True)

    def load_bundle(self, **options: Any) -> Mapping[str, Any]:
        """Return a manifest bundle. The generic adapter *is* the interchange
        format: it validates a provided ``bundle`` (or ``path``) and returns it."""

        from lancedb_robotics.tracker_sync import load_bundle_file

        if "bundle" in options and options["bundle"] is not None:
            bundle = options["bundle"]
            if not isinstance(bundle, Mapping):
                raise MetadataPluginError("bundle must be a mapping/object")
            return dict(bundle)
        if options.get("path"):
            return load_bundle_file(options["path"])
        return {"training_runs": [], "model_artifacts": [], "evaluation_runs": []}


class _TrackerSyncAdapter(MetadataAdapter, ManifestSyncMixin):
    """An optional experiment-tracker/catalog sync provider (MLflow, W&B, DVC,
    lakeFS, Kubeflow/MLMD). Live fetch requires the tracker's client; the always
    available path is to import a JSON bundle exported from the tracker."""

    family = FAMILY_MANIFEST_SYNC

    def __init__(self, name: str, module: str, optional_extra: str) -> None:
        self.name = name
        self._module = module
        self._optional_extra = optional_extra

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(import_bundle=True, live_fetch=True, dry_run=True)

    def dependency(self) -> AdapterDependency:
        return AdapterDependency(
            adapter=self.name,
            module=self._module,
            optional_extra=self._optional_extra,
            plugin=None,
            native=False,
        )

    def load_bundle(self, **options: Any) -> Mapping[str, Any]:
        from lancedb_robotics.tracker_sync import load_tracker_bundle

        return load_tracker_bundle(self.name, **options)


# The optional tracker/catalog ecosystems 0106 declares conformance fixtures for.
# (module, optional_extra) mirror lineage_integrations._KNOWN_ADAPTERS so the
# probe/hint shape stays consistent across the two entry points.
_TRACKER_ECOSYSTEMS: tuple[tuple[str, str, str], ...] = (
    ("mlflow", "mlflow", "mlflow"),
    ("wandb", "wandb", "wandb"),
    ("dvc", "dvc", "dvc"),
    ("lakefs", "lakefs", "lakefs"),
    ("kubeflow", "ml_metadata", "kubeflow-mlmd"),
)


def _register_builtins(registry: MetadataPluginRegistry) -> None:
    registry.register("openlineage", _OpenLineageAdapter)
    registry.register("datahub", _DataHubAdapter)
    registry.register("generic", _GenericManifestAdapter)
    for name, module, extra in _TRACKER_ECOSYSTEMS:
        registry.register(
            name,
            lambda name=name, module=module, extra=extra: _TrackerSyncAdapter(name, module, extra),
        )


# ---------------------------------------------------------------------------
# Conformance suite.
# ---------------------------------------------------------------------------


# Deliberately awkward canonical artifact ids: URN-reserved characters, unicode,
# and the two shapes this package actually mints. A conforming reversible-URN
# adapter must round-trip every one of these unchanged.
_CONFORMANCE_ARTIFACT_IDS: tuple[str, ...] = (
    "lancedb-robotics:snapshot:abc123",
    "lancedb-robotics:training-run:run/with slashes",
    "lancedb-robotics:row:table@v3#42",
    "lancedb-robotics:model-output:policy?q=1&r=2",
    "artifact:with:colons:and-üñíçödé",
)

_AUTH_SECRET_MARKER = "CONFORMANCE-SECRET-DO-NOT-PERSIST-0106"


@dataclass(frozen=True)
class ConformanceCheck:
    name: str
    status: str  # passed | failed | skipped
    detail: str

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class ConformanceReport:
    adapter: str
    family: str
    checks: tuple[ConformanceCheck, ...]

    @property
    def passed(self) -> bool:
        """True when no check failed (skipped checks do not fail conformance)."""

        return all(check.status != "failed" for check in self.checks)

    @property
    def failures(self) -> tuple[ConformanceCheck, ...]:
        return tuple(check for check in self.checks if check.status == "failed")

    @property
    def ran(self) -> tuple[ConformanceCheck, ...]:
        return tuple(check for check in self.checks if check.status != "skipped")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "family": self.family,
            "passed": self.passed,
            "counts": {
                "passed": sum(1 for c in self.checks if c.status == "passed"),
                "failed": len(self.failures),
                "skipped": sum(1 for c in self.checks if c.status == "skipped"),
                "total": len(self.checks),
            },
            "checks": [c.to_dict() for c in self.checks],
        }


class _RecordingSink:
    """An in-memory emitter sink used only by the conformance suite.

    When ``fail`` is set it raises, exercising the adapter's structured
    failure-recording path. It never echoes auth back, so a passing
    auth-non-persistence check reflects the adapter, not the sink.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def emit(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if self.fail:
            raise RuntimeError("conformance-induced sink failure")
        return {"remote_id": f"conformance-{len(self.calls)}", "status": "ok"}


def run_conformance(
    adapter: MetadataAdapter,
    *,
    lake: Any | None = None,
    artifact_ids: Sequence[str] | None = None,
) -> ConformanceReport:
    """Validate that ``adapter`` obeys the metadata-integration contract.

    Only the checks a declared capability implies are run; capability-irrelevant
    checks are marked ``skipped``. Emit-path checks that need canonical tables are
    ``skipped`` unless a ``lake`` is provided (a self-contained adapter that
    ignores the lake still runs them). A skipped check never fails conformance.
    """

    checks: list[ConformanceCheck] = []
    caps = adapter.capabilities()

    checks.append(_check_naming(adapter))
    checks.append(_check_capabilities(adapter, caps))
    checks.append(_check_dependency_shape(adapter))

    checks.append(_check_reversible_urns(adapter, caps, artifact_ids))
    checks.append(_check_dry_run_parity(adapter, caps, lake))
    checks.append(_check_auth_non_persistence(adapter, caps, lake))
    checks.append(_check_failure_recording(adapter, caps, lake))
    checks.append(_check_sync_bundle(adapter, caps))

    return ConformanceReport(adapter=adapter.name, family=adapter.family, checks=tuple(checks))


def run_registry_conformance(
    registry: MetadataPluginRegistry | None = None,
    *,
    lake: Any | None = None,
    artifact_ids: Sequence[str] | None = None,
) -> tuple[ConformanceReport, ...]:
    """Run the conformance suite over every adapter in a registry."""

    registry = registry or default_registry()
    return tuple(
        run_conformance(adapter, lake=lake, artifact_ids=artifact_ids)
        for adapter in registry.adapters()
    )


def _ok(name: str, detail: str) -> ConformanceCheck:
    return ConformanceCheck(name=name, status="passed", detail=detail)


def _fail(name: str, detail: str) -> ConformanceCheck:
    return ConformanceCheck(name=name, status="failed", detail=detail)


def _skip(name: str, detail: str) -> ConformanceCheck:
    return ConformanceCheck(name=name, status="skipped", detail=detail)


def _check_naming(adapter: MetadataAdapter) -> ConformanceCheck:
    name = "contract:naming"
    if not isinstance(adapter.name, str) or not adapter.name.strip():
        return _fail(name, "adapter.name must be a non-empty string")
    if adapter.family not in KNOWN_FAMILIES:
        return _fail(name, f"adapter.family {adapter.family!r} not in {list(KNOWN_FAMILIES)}")
    return _ok(name, f"name={adapter.name!r} family={adapter.family!r}")


def _check_capabilities(adapter: MetadataAdapter, caps: AdapterCapabilities) -> ConformanceCheck:
    name = "contract:capabilities"
    if not isinstance(caps, AdapterCapabilities):
        return _fail(name, "capabilities() must return AdapterCapabilities")
    missing: list[str] = []
    if caps.emit:
        for method in ("build_payloads", "emit"):
            if not callable(getattr(adapter, method, None)):
                missing.append(method)
    if caps.reversible_urns:
        for method in ("to_external_urn", "from_external_urn"):
            if not callable(getattr(adapter, method, None)):
                missing.append(method)
    if (caps.import_bundle or caps.live_fetch) and not callable(getattr(adapter, "load_bundle", None)):
        missing.append("load_bundle")
    if missing:
        return _fail(name, f"declared capabilities require missing methods: {sorted(set(missing))}")
    return _ok(name, f"capabilities consistent: {caps.to_dict()}")


def _check_dependency_shape(adapter: MetadataAdapter) -> ConformanceCheck:
    name = "dependency:probe-shape"
    try:
        dependency = adapter.dependency()
        status = adapter.probe()
    except Exception as exc:  # noqa: BLE001 - a probe must never raise
        return _fail(name, f"dependency()/probe() raised: {exc!r}")
    if not isinstance(dependency, AdapterDependency):
        return _fail(name, "dependency() must return AdapterDependency")
    if not isinstance(status, DependencyStatus):
        return _fail(name, "probe() must return DependencyStatus")
    if status.adapter != adapter.name:
        return _fail(name, "probe().adapter must match adapter.name")
    if not status.message.strip():
        return _fail(name, "probe().message must be non-empty")
    if not dependency.native:
        if not dependency.module:
            return _fail(name, "non-native adapter must name a dependency module")
        if not (dependency.optional_extra or dependency.plugin):
            return _fail(name, "non-native adapter must name an optional_extra or plugin")
        if not status.install_hint.strip():
            return _fail(name, "non-native adapter must supply a non-empty install_hint")
    payload = status.to_dict()
    required = {"adapter", "available", "module", "optional_extra", "plugin", "native", "install_hint"}
    if not required <= set(payload):
        return _fail(name, f"probe().to_dict() missing keys: {sorted(required - set(payload))}")
    return _ok(name, f"probe={payload}")


def _check_reversible_urns(
    adapter: MetadataAdapter,
    caps: AdapterCapabilities,
    artifact_ids: Sequence[str] | None,
) -> ConformanceCheck:
    name = "urn:reversible"
    if not caps.reversible_urns:
        return _skip(name, "adapter does not declare reversible_urns")
    ids = tuple(artifact_ids) if artifact_ids else _CONFORMANCE_ARTIFACT_IDS
    for artifact_id in ids:
        try:
            urn = adapter.to_external_urn(artifact_id)
        except Exception as exc:  # noqa: BLE001
            return _fail(name, f"to_external_urn({artifact_id!r}) raised: {exc!r}")
        if not isinstance(urn, str) or not urn:
            return _fail(name, f"to_external_urn({artifact_id!r}) returned an empty/non-str URN")
        try:
            recovered = adapter.from_external_urn(urn)
        except Exception as exc:  # noqa: BLE001
            return _fail(name, f"from_external_urn({urn!r}) raised: {exc!r}")
        if recovered != artifact_id:
            return _fail(
                name,
                f"URN not reversible: {artifact_id!r} -> {urn!r} -> {recovered!r} "
                "(adapter mutated or dropped the canonical artifact id)",
            )
    return _ok(name, f"round-tripped {len(ids)} canonical artifact ids losslessly")


def _check_dry_run_parity(
    adapter: MetadataAdapter,
    caps: AdapterCapabilities,
    lake: Any | None,
) -> ConformanceCheck:
    name = "payload:dry-run-parity"
    if not caps.emit:
        return _skip(name, "adapter does not declare emit")
    if lake is None:
        return _skip(name, "no lake provided; emit-path checks need one (or a self-contained adapter)")
    try:
        first = adapter.build_payloads(lake)
        second = adapter.build_payloads(lake)
    except Exception as exc:  # noqa: BLE001
        return _fail(name, f"build_payloads raised: {exc!r}")
    if _payloads_digest(first) != _payloads_digest(second):
        return _fail(name, "build_payloads is not deterministic across calls")
    return _ok(name, f"build_payloads deterministic across calls ({len(first)} payloads)")


def _check_auth_non_persistence(
    adapter: MetadataAdapter,
    caps: AdapterCapabilities,
    lake: Any | None,
) -> ConformanceCheck:
    name = "auth:non-persistence"
    if not caps.emit:
        return _skip(name, "adapter does not declare emit")
    if lake is None:
        return _skip(name, "no lake provided; emit-path checks need one (or a self-contained adapter)")
    auth = AdapterAuth(
        auth_ref=_AUTH_SECRET_MARKER,
        headers={"Authorization": f"Bearer {_AUTH_SECRET_MARKER}"},
        client=_RecordingSink(),
        target="conformance-auth-0106",
    )
    markers = auth.secret_markers()
    try:
        payloads = adapter.build_payloads(lake)
    except Exception as exc:  # noqa: BLE001
        return _fail(name, f"build_payloads raised: {exc!r}")
    if _contains_marker(payloads, markers):
        return _fail(name, "built payloads contain an auth secret marker")
    try:
        result = adapter.emit(lake, auth=auth)
    except Exception as exc:  # noqa: BLE001
        return _fail(name, f"emit raised instead of recording: {exc!r}")
    if _contains_marker(result.to_dict(), markers):
        return _fail(name, "EmitResult contains an auth secret marker")
    leaked = _scan_lake_for_markers(lake, markers)
    if leaked:
        return _fail(name, f"auth secret marker persisted into canonical rows: {sorted(leaked)}")
    return _ok(name, "auth ref/headers absent from payloads, result, and persisted rows")


def _check_failure_recording(
    adapter: MetadataAdapter,
    caps: AdapterCapabilities,
    lake: Any | None,
) -> ConformanceCheck:
    name = "emit:failure-recording"
    if not caps.emit:
        return _skip(name, "adapter does not declare emit")
    if lake is None:
        return _skip(name, "no lake provided; emit-path checks need one (or a self-contained adapter)")
    auth = AdapterAuth(client=_RecordingSink(fail=True), target="conformance-fail-0106")
    try:
        payloads = adapter.build_payloads(lake)
    except Exception as exc:  # noqa: BLE001
        return _fail(name, f"build_payloads raised: {exc!r}")
    if not payloads:
        return _skip(name, "adapter built zero payloads on this lake; nothing to fail")
    try:
        result = adapter.emit(lake, auth=auth)
    except Exception as exc:  # noqa: BLE001
        return _fail(name, f"emit raised instead of recording a failed attempt: {exc!r}")
    if result.failed < 1:
        return _fail(name, "a failing sink must produce >= 1 failed attempt")
    for attempt in result.attempts:
        if attempt.status == "failed" and not (attempt.error or "").strip():
            return _fail(name, "failed attempts must carry a non-empty error string")
        if attempt.status == "failed" and not attempt.payload_digest:
            return _fail(name, "failed attempts must carry a payload digest for retry")
    if result.status not in ("failed", "partial"):
        return _fail(name, f"result.status for all-failed emit must be failed/partial, got {result.status!r}")
    return _ok(name, f"recorded {result.failed} structured failed attempt(s)")


def _check_sync_bundle(adapter: MetadataAdapter, caps: AdapterCapabilities) -> ConformanceCheck:
    name = "sync:bundle-contract"
    if not caps.import_bundle:
        return _skip(name, "adapter does not declare import_bundle")
    if caps.live_fetch and not _is_native(adapter):
        # Optional live-fetch trackers require their client; the always-available
        # path is a JSON bundle. We only assert the probe/hint contract for them.
        return _skip(name, "live-fetch tracker; bundle import validated via JSON path, not live fetch")
    try:
        bundle = adapter.load_bundle(bundle={"training_runs": [], "model_artifacts": [], "evaluation_runs": []})
    except Exception as exc:  # noqa: BLE001
        return _fail(name, f"load_bundle raised on a valid empty bundle: {exc!r}")
    if not isinstance(bundle, Mapping):
        return _fail(name, "load_bundle must return a mapping")
    return _ok(name, "load_bundle honored the manifest-bundle interchange contract")


def _is_native(adapter: MetadataAdapter) -> bool:
    try:
        return adapter.dependency().native
    except Exception:  # noqa: BLE001
        return False


# --- conformance helpers ----------------------------------------------------


def _payloads_digest(payloads: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps(list(payloads), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _contains_marker(value: Any, markers: set[str]) -> bool:
    if not markers:
        return False
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return any(marker in encoded for marker in markers)


class LakeMetadataPlugins:
    """Lake-bound facade over the metadata-plugin registry (``lake.lineage.plugins``).

    Uses the process-wide default registry so in-house adapters registered with
    :func:`register_metadata_plugin` are visible here, and passes the bound lake
    into emit/conformance calls that need canonical tables.
    """

    def __init__(self, lake: Any) -> None:
        self._lake = lake

    @property
    def registry(self) -> MetadataPluginRegistry:
        return default_registry()

    def names(self) -> tuple[str, ...]:
        return self.registry.names()

    def get(self, name: str) -> MetadataAdapter:
        return self.registry.get(name)

    def describe(self, *, family: str | None = None) -> tuple[dict[str, Any], ...]:
        return self.registry.describe(family=family)

    def register(self, name: str, factory: AdapterFactory, *, replace: bool = False) -> None:
        self.registry.register(name, factory, replace=replace)

    def probe(self, name: str) -> DependencyStatus:
        return self.registry.get(name).probe()

    def conformance(
        self,
        name: str | None = None,
        *,
        artifact_ids: Sequence[str] | None = None,
    ) -> ConformanceReport | tuple[ConformanceReport, ...]:
        """Run the conformance suite for one adapter, or all when ``name`` is None.

        The bound lake enables the emit-path checks (dry-run parity, auth
        non-persistence, failure recording) against real canonical tables.
        """

        if name is not None:
            return run_conformance(self.registry.get(name), lake=self._lake, artifact_ids=artifact_ids)
        return run_registry_conformance(self.registry, lake=self._lake, artifact_ids=artifact_ids)


def _scan_lake_for_markers(lake: Any, markers: set[str]) -> set[str]:
    """Scan the canonical lineage + delivery tables for any auth secret marker."""

    if not markers:
        return set()
    found: set[str] = set()
    tables = (
        "lineage_artifacts",
        "lineage_executions",
        "lineage_edges",
        "lineage_delivery_attempts",
    )
    try:
        available = set(lake.table_names())
    except Exception:  # noqa: BLE001
        available = set()
    for table in tables:
        if table not in available:
            continue
        try:
            rows = lake.table(table).to_arrow().to_pylist()
        except Exception:  # noqa: BLE001
            continue
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
        for marker in markers:
            if marker in encoded:
                found.add(marker)
    return found
