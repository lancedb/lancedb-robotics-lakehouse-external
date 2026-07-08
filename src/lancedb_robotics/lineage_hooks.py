"""External execution context hooks for lineage-bearing operations.

The hook layer is deliberately dependency-light. It normalizes context from
orchestrators and experiment trackers into JSON payloads that can be recorded in
the lake without making those external systems canonical or requiring network
calls.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

LINEAGE_CONTEXT_ENV = "LANCEDB_ROBOTICS_LINEAGE_CONTEXT"
LINEAGE_CONTEXT_FILE_ENV = "LANCEDB_ROBOTICS_LINEAGE_CONTEXT_FILE"
LINEAGE_ADAPTER_ENV = "LANCEDB_ROBOTICS_LINEAGE_ADAPTER"

_ADAPTERS = {
    "airflow": ("airflow", "apache-airflow"),
    "apache-airflow": ("airflow", "apache-airflow"),
    "dagster": ("dagster", "dagster"),
    "ray": ("ray", "ray"),
    "slurm": ("pyslurm", "slurm"),
    "kubeflow": ("ml_metadata", "kubeflow-mlmd"),
    "mlmd": ("ml_metadata", "kubeflow-mlmd"),
    "mlflow": ("mlflow", "mlflow"),
    "wandb": ("wandb", "wandb"),
    "weights-and-biases": ("wandb", "wandb"),
    "openlineage": ("openlineage.client", "openlineage"),
    "openlineage-client": ("openlineage.client", "openlineage"),
}

DEFAULT_WORKER_CONTEXT_KEYS = (
    "provider",
    "external_run_id",
    "external_job_id",
    "external_parent_run_id",
    "external_url",
    "code_ref",
    "environment_digest",
    "environment",
    "external_refs",
    "artifact_refs",
    "facets",
)
DEFAULT_WORKER_REDACTED_KEY_FRAGMENTS = (
    "secret",
    "token",
    "password",
    "credential",
    "api_key",
    "access_key",
    "private_key",
    "auth",
)
_CURRENT_LINEAGE_CONTEXT: ContextVar[LineageContext | None] = ContextVar(
    "lancedb_robotics_lineage_context",
    default=None,
)


class LineageHookError(Exception):
    """Raised when lineage context cannot be parsed or an adapter is unavailable."""


class LineageHookConformanceError(LineageHookError):
    """Raised when a lineage hook fails the plugin conformance contract."""


class LineageHook(Protocol):
    """Provider hook for before/after execution lineage context."""

    def before_execution(
        self,
        operation: str,
        params: Mapping[str, Any] | None = None,
    ) -> LineageContext | Mapping[str, Any] | None:
        """Return context that should be attached before an operation is recorded."""

    def after_execution(
        self,
        operation: str,
        context: LineageContext | None,
        *,
        status: str,
        error: str | None = None,
    ) -> LineageContext | Mapping[str, Any] | None:
        """Return final context after an operation succeeds or fails."""


@dataclass(frozen=True)
class LineageHookAdapterSpec:
    """Optional hook adapter discovery metadata and install guidance."""

    name: str
    module_name: str
    extra: str
    network_calls_by_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module_name": self.module_name,
            "extra": self.extra,
            "install": f"lancedb-robotics[{self.extra}] or a plugin exposing {self.module_name}",
            "network_calls_by_default": self.network_calls_by_default,
        }


@dataclass(frozen=True)
class LineageHookConformanceIssue:
    """One hook conformance failure."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class LineageHookConformanceReport:
    """Result of running the dependency-free lineage hook conformance harness."""

    hook_name: str
    passed: bool
    issues: tuple[LineageHookConformanceIssue, ...]
    before_context: dict[str, Any] = field(default_factory=dict)
    after_context: dict[str, Any] = field(default_factory=dict)
    checks: tuple[str, ...] = (
        "callbacks-present",
        "before-normalizes",
        "normalization-idempotent",
        "after-normalizes",
        "after-does-not-mutate-input-context",
        "json-ready",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_name": self.hook_name,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "before_context": dict(self.before_context),
            "after_context": dict(self.after_context),
            "checks": list(self.checks),
        }


@dataclass(frozen=True)
class LineageContext:
    """Normalized external run/job/artifact context for a lake operation."""

    provider: str | None = None
    external_run_id: str | None = None
    external_job_id: str | None = None
    external_parent_run_id: str | None = None
    external_url: str | None = None
    code_ref: str | None = None
    environment_digest: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    external_refs: dict[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[dict[str, Any], ...] = ()
    facets: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    error: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.to_dict()

    def with_status(self, status: str, error: str | None = None) -> LineageContext:
        return replace(self, status=status or self.status, error=error)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "provider": self.provider,
            "external_run_id": self.external_run_id,
            "external_job_id": self.external_job_id,
            "external_parent_run_id": self.external_parent_run_id,
            "external_url": self.external_url,
            "code_ref": self.code_ref,
            "environment_digest": self.environment_digest,
            "environment": self.environment,
            "external_refs": self.external_refs,
            "artifact_refs": [dict(item) for item in self.artifact_refs],
            "facets": self.facets,
            "status": self.status,
            "error": self.error,
        }
        return {
            key: _jsonable(value)
            for key, value in payload.items()
            if _has_value(value)
        }

    def external_reference_map(self) -> dict[str, Any]:
        """Return string-key references suitable for manifest ``external_refs``."""

        refs = dict(self.external_refs)
        _set_if(refs, "external_provider", self.provider)
        _set_if(refs, "external_run_id", self.external_run_id)
        _set_if(refs, "external_job_id", self.external_job_id)
        _set_if(refs, "external_parent_run_id", self.external_parent_run_id)
        _set_if(refs, "external_url", self.external_url)
        _set_if(refs, "external_code_ref", self.code_ref)
        _set_if(refs, "external_environment_digest", self.environment_digest)
        if self.provider:
            prefix = _ref_key(self.provider)
            _set_if(refs, f"{prefix}_run_id", self.external_run_id)
            _set_if(refs, f"{prefix}_job_id", self.external_job_id)
            _set_if(refs, f"{prefix}_url", self.external_url)
        return {str(key): value for key, value in refs.items() if value is not None}

    def transform_params(self) -> dict[str, Any]:
        """Return params that expose context to lineage graph refresh."""

        payload = self.to_dict()
        if not payload:
            return {}
        params: dict[str, Any] = {
            "lineage_context": payload,
            "external_refs": self.external_reference_map(),
        }
        if self.provider:
            params["provider"] = self.provider
        if self.code_ref:
            params["code_ref"] = self.code_ref
        environment: dict[str, Any] = {}
        if self.environment:
            environment.update(self.environment)
        if self.environment_digest:
            environment.setdefault("digest", self.environment_digest)
        if environment:
            params["environment"] = environment
        if self.artifact_refs:
            params["external_artifacts"] = [dict(item) for item in self.artifact_refs]
        if self.facets:
            params["external_facets"] = dict(self.facets)
        if self.status:
            params["external_status"] = self.status
        if self.error:
            params["external_error"] = self.error
        return params


@dataclass
class LineageExecutionHandle:
    """Small lifecycle wrapper that invokes hook before/after callbacks."""

    operation: str
    hook: LineageHook | None = None
    context: LineageContext | None = None

    def finish(self, *, status: str, error: Exception | str | None = None) -> LineageContext | None:
        message = str(error) if error else None
        context = self.context.with_status(status, message) if self.context else None
        if self.hook is not None:
            updated = self.hook.after_execution(
                self.operation,
                context,
                status=status,
                error=message,
            )
            context = merge_lineage_contexts(context, normalize_lineage_context(updated))
        if context is None or context.is_empty:
            return None
        return context


@dataclass(frozen=True)
class StaticLineageHook:
    """Hook that always returns the same context payload."""

    context: LineageContext | Mapping[str, Any]

    def before_execution(
        self,
        operation: str,
        params: Mapping[str, Any] | None = None,
    ) -> LineageContext:
        return normalize_lineage_context(self.context) or LineageContext()

    def after_execution(
        self,
        operation: str,
        context: LineageContext | None,
        *,
        status: str,
        error: str | None = None,
    ) -> LineageContext | None:
        return context


class EnvironmentLineageHook:
    """Hook that reads context from process environment variables."""

    def before_execution(
        self,
        operation: str,
        params: Mapping[str, Any] | None = None,
    ) -> LineageContext | None:
        return lineage_context_from_env()

    def after_execution(
        self,
        operation: str,
        context: LineageContext | None,
        *,
        status: str,
        error: str | None = None,
    ) -> LineageContext | None:
        return context


class MLflowLineageHook:
    """Adapter hook for the current MLflow run, when MLflow is installed."""

    def before_execution(
        self,
        operation: str,
        params: Mapping[str, Any] | None = None,
    ) -> LineageContext:
        return lineage_context_from_adapter("mlflow")

    def after_execution(
        self,
        operation: str,
        context: LineageContext | None,
        *,
        status: str,
        error: str | None = None,
    ) -> LineageContext | None:
        return context


class WandbLineageHook:
    """Adapter hook for the current W&B run, when W&B is installed."""

    def before_execution(
        self,
        operation: str,
        params: Mapping[str, Any] | None = None,
    ) -> LineageContext:
        return lineage_context_from_adapter("wandb")

    def after_execution(
        self,
        operation: str,
        context: LineageContext | None,
        *,
        status: str,
        error: str | None = None,
    ) -> LineageContext | None:
        return context


class OpenLineageLineageHook:
    """Adapter hook for OpenLineage client context without emitting network calls."""

    def before_execution(
        self,
        operation: str,
        params: Mapping[str, Any] | None = None,
    ) -> LineageContext:
        return lineage_context_from_adapter("openlineage")

    def after_execution(
        self,
        operation: str,
        context: LineageContext | None,
        *,
        status: str,
        error: str | None = None,
    ) -> LineageContext | None:
        return context


class LineageContextScope:
    """Context manager that makes lineage context available to nested operations."""

    def __init__(
        self,
        lineage_context: Any | None,
        *,
        inherit: bool = True,
    ) -> None:
        self._lineage_context = lineage_context
        self._inherit = inherit
        self._token: Token[LineageContext | None] | None = None
        self._context: LineageContext | None = None

    def __enter__(self) -> LineageContext:
        parent = current_lineage_context() if self._inherit else None
        override = normalize_lineage_context(self._lineage_context)
        self._context = merge_lineage_contexts(parent, override)
        self._token = _CURRENT_LINEAGE_CONTEXT.set(self._context)
        return self._context or LineageContext()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._token is not None:
            _CURRENT_LINEAGE_CONTEXT.reset(self._token)
        self._token = None


def begin_lineage_execution(
    lineage_context: Any | None,
    *,
    operation: str,
    params: Mapping[str, Any] | None = None,
) -> LineageExecutionHandle:
    """Normalize a context/hook value and call any before-execution callback."""

    scoped = current_lineage_context()
    if _is_hook(lineage_context):
        hook = lineage_context
        context = merge_lineage_contexts(
            scoped,
            normalize_lineage_context(hook.before_execution(operation, params)),
        )
        return LineageExecutionHandle(operation=operation, hook=hook, context=context)
    context = merge_lineage_contexts(scoped, normalize_lineage_context(lineage_context))
    return LineageExecutionHandle(operation=operation, context=context)


def lineage_context_scope(
    lineage_context: Any | None,
    *,
    inherit: bool = True,
) -> LineageContextScope:
    """Return a scoped context manager for nested SDK operations."""

    return LineageContextScope(lineage_context, inherit=inherit)


def current_lineage_context() -> LineageContext | None:
    """Return the context active in the current contextvars scope."""

    return _CURRENT_LINEAGE_CONTEXT.get()


def lineage_context_for_worker(
    lineage_context: Any | None = None,
    *,
    include_keys: Sequence[str] | None = None,
    redact_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return an explicit, redacted context payload for worker/process launchers."""

    scoped = current_lineage_context()
    override = normalize_lineage_context(lineage_context)
    context = merge_lineage_contexts(scoped, override)
    if context is None or context.is_empty:
        return {}
    allowed = set(include_keys or DEFAULT_WORKER_CONTEXT_KEYS)
    payload = {
        key: value
        for key, value in context.to_dict().items()
        if key in allowed
    }
    redacted = tuple(redact_keys or DEFAULT_WORKER_REDACTED_KEY_FRAGMENTS)
    return _redact_context_payload(payload, redacted)


def lineage_context_env_for_worker(
    lineage_context: Any | None = None,
    *,
    include_keys: Sequence[str] | None = None,
    redact_keys: Sequence[str] | None = None,
) -> dict[str, str]:
    """Return environment variables that explicitly propagate worker context."""

    payload = lineage_context_for_worker(
        lineage_context,
        include_keys=include_keys,
        redact_keys=redact_keys,
    )
    if not payload:
        return {}
    return {LINEAGE_CONTEXT_ENV: json.dumps(payload, sort_keys=True)}


def normalize_lineage_context(value: Any | None) -> LineageContext | None:
    """Coerce a mapping, JSON string, file path, adapter name, or hook context."""

    if value is None:
        return None
    if isinstance(value, LineageContext):
        return None if value.is_empty else value
    if _is_hook(value):
        return begin_lineage_execution(value, operation="lineage-context").finish(
            status="completed"
        )
    if isinstance(value, Mapping):
        context = _context_from_mapping(value)
        return None if context.is_empty else context
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lower() in _ADAPTERS:
            return lineage_context_from_adapter(text)
        return _context_from_mapping(_load_json_context(text))
    raise LineageHookError(
        "lineage_context must be a mapping, JSON string, JSON file path, adapter "
        "name, LineageContext, or LineageHook"
    )


def merge_lineage_contexts(
    base: LineageContext | None,
    override: LineageContext | None,
) -> LineageContext | None:
    """Merge two contexts, with non-empty override fields winning."""

    if base is None:
        return override
    if override is None:
        return base
    merged = LineageContext(
        provider=override.provider or base.provider,
        external_run_id=override.external_run_id or base.external_run_id,
        external_job_id=override.external_job_id or base.external_job_id,
        external_parent_run_id=override.external_parent_run_id or base.external_parent_run_id,
        external_url=override.external_url or base.external_url,
        code_ref=override.code_ref or base.code_ref,
        environment_digest=override.environment_digest or base.environment_digest,
        environment={**base.environment, **override.environment},
        external_refs={**base.external_refs, **override.external_refs},
        artifact_refs=(*base.artifact_refs, *override.artifact_refs),
        facets={**base.facets, **override.facets},
        status=override.status or base.status,
        error=override.error or base.error,
    )
    return None if merged.is_empty else merged


def apply_lineage_context_to_manifest_fields(
    *,
    code_ref: str | None = None,
    environment: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    external_refs: Mapping[str, Any] | None = None,
    lineage_context: LineageContext | None = None,
) -> dict[str, Any]:
    """Merge context into common manifest fields without changing table schemas."""

    context = lineage_context
    refs = dict(context.external_reference_map()) if context else {}
    refs.update(dict(external_refs or {}))

    merged_environment = dict(context.environment) if context else {}
    if context and context.environment_digest:
        merged_environment.setdefault("digest", context.environment_digest)
    merged_environment.update(dict(environment or {}))

    merged_runtime = dict(runtime or {})
    if context:
        merged_runtime.setdefault("lineage_context", context.to_dict())

    return {
        "code_ref": code_ref or (context.code_ref if context else None),
        "environment": merged_environment,
        "runtime": merged_runtime,
        "external_refs": refs,
    }


def attach_lineage_context_to_params(
    params: Mapping[str, Any],
    lineage_context: Any | None,
) -> dict[str, Any]:
    """Return transform params with normalized external context folded in."""

    merged = dict(params)
    lineage_context = normalize_lineage_context(lineage_context)
    if lineage_context is None or lineage_context.is_empty:
        return merged
    for key, value in lineage_context.transform_params().items():
        if key == "external_refs" and isinstance(merged.get(key), Mapping):
            merged[key] = {**dict(value), **dict(merged[key])}
        elif key == "environment" and isinstance(merged.get(key), Mapping):
            merged[key] = {**dict(value), **dict(merged[key])}
        else:
            merged.setdefault(key, value)
    return merged


def lineage_context_digest(lineage_context: LineageContext | None) -> str:
    """Short stable digest for context-bearing execution IDs."""

    if lineage_context is None or lineage_context.is_empty:
        return ""
    encoded = json.dumps(lineage_context.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode()).hexdigest()[:12]


def lineage_context_from_env() -> LineageContext | None:
    """Read a generic JSON/file/adaptor context from environment variables."""

    raw = os.getenv(LINEAGE_CONTEXT_ENV)
    if raw:
        return normalize_lineage_context(raw)
    file_path = os.getenv(LINEAGE_CONTEXT_FILE_ENV)
    if file_path:
        return normalize_lineage_context(str(Path(file_path)))
    adapter = os.getenv(LINEAGE_ADAPTER_ENV)
    if adapter:
        return lineage_context_from_adapter(adapter)
    return None


def lineage_context_from_adapter(adapter: str) -> LineageContext:
    """Capture a current optional adapter context without making network calls."""

    normalized = str(adapter or "").strip().lower()
    module_name, extra = _adapter_metadata(normalized)
    _require_adapter(normalized, module_name, extra)
    if normalized == "mlflow":
        return _mlflow_context()
    if normalized in {"wandb", "weights-and-biases"}:
        return _wandb_context()
    if normalized in {"openlineage", "openlineage-client"}:
        return _openlineage_context()
    return LineageContext(provider=normalized)


def lineage_hook_adapter_specs() -> tuple[LineageHookAdapterSpec, ...]:
    """Return known dependency-light hook adapter specs for plugin discovery."""

    return tuple(
        LineageHookAdapterSpec(name=name, module_name=module, extra=extra)
        for name, (module, extra) in sorted(_ADAPTERS.items())
    )


def require_lineage_hook_adapter(adapter: str) -> dict[str, str]:
    """Validate that a hook adapter module is importable and return guidance metadata."""

    normalized = str(adapter or "").strip().lower()
    module_name, extra = _adapter_metadata(normalized)
    _require_adapter(normalized, module_name, extra)
    return {
        "adapter": normalized,
        "module_name": module_name,
        "extra": extra,
        "install": f"lancedb-robotics[{extra}] or a plugin exposing {module_name}",
    }


def check_lineage_hook_conformance(
    hook: Any,
    *,
    operation: str = "lineage-hook-conformance",
    params: Mapping[str, Any] | None = None,
) -> LineageHookConformanceReport:
    """Run the plugin hook conformance harness without optional dependencies."""

    hook_name = type(hook).__name__
    issues: list[LineageHookConformanceIssue] = []
    before_context: LineageContext | None = None
    after_context: LineageContext | None = None
    before_payload: dict[str, Any] = {}
    after_payload: dict[str, Any] = {}

    if not callable(getattr(hook, "before_execution", None)):
        issues.append(
            LineageHookConformanceIssue(
                "missing-before-execution",
                "hook must define callable before_execution(operation, params)",
            )
        )
    if not callable(getattr(hook, "after_execution", None)):
        issues.append(
            LineageHookConformanceIssue(
                "missing-after-execution",
                "hook must define callable after_execution(operation, context, status=..., error=...)",
            )
        )
    if issues:
        return LineageHookConformanceReport(
            hook_name=hook_name,
            passed=False,
            issues=tuple(issues),
        )

    try:
        raw_before = hook.before_execution(operation, dict(params or {}))
        before_context = normalize_lineage_context(raw_before)
        before_payload = before_context.to_dict() if before_context else {}
        _assert_json_ready(before_payload, "before_execution context")
    except Exception as exc:  # noqa: BLE001 - report contract failures together
        issues.append(
            LineageHookConformanceIssue(
                "before-execution-invalid",
                f"before_execution must return a JSON-ready lineage context: {exc}",
            )
        )

    if before_context is not None:
        try:
            renormalized = normalize_lineage_context(before_context.to_dict())
            if (renormalized.to_dict() if renormalized else {}) != before_payload:
                issues.append(
                    LineageHookConformanceIssue(
                        "normalization-not-idempotent",
                        "normalizing before_execution output twice changed the payload",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - normalize failures become issues
            issues.append(
                LineageHookConformanceIssue(
                    "normalization-not-idempotent",
                    f"normalizing before_execution output twice failed: {exc}",
                )
            )

    before_digest = _payload_digest(before_payload)
    try:
        raw_after = hook.after_execution(
            operation,
            before_context,
            status="completed",
            error=None,
        )
        mutated_payload = before_context.to_dict() if before_context else {}
        if _payload_digest(mutated_payload) != before_digest:
            issues.append(
                LineageHookConformanceIssue(
                    "context-mutated",
                    "after_execution must not mutate the context object it receives; return an override instead",
                )
            )
        after_context = normalize_lineage_context(raw_after)
        after_payload = after_context.to_dict() if after_context else {}
        _assert_json_ready(after_payload, "after_execution context")
    except Exception as exc:  # noqa: BLE001 - report contract failures together
        issues.append(
            LineageHookConformanceIssue(
                "after-execution-invalid",
                f"after_execution must return a JSON-ready lineage context: {exc}",
            )
        )

    return LineageHookConformanceReport(
        hook_name=hook_name,
        passed=not issues,
        issues=tuple(issues),
        before_context=before_payload,
        after_context=after_payload,
    )


def assert_lineage_hook_conformance(
    hook: Any,
    *,
    operation: str = "lineage-hook-conformance",
    params: Mapping[str, Any] | None = None,
) -> LineageHookConformanceReport:
    """Return a conformance report or raise with actionable issue messages."""

    report = check_lineage_hook_conformance(hook, operation=operation, params=params)
    if report.passed:
        return report
    details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
    raise LineageHookConformanceError(f"lineage hook {report.hook_name} failed conformance: {details}")


def capture_code_environment(
    *,
    provider: str = "local",
    env_keys: Sequence[str] | None = None,
) -> LineageContext:
    """Capture lightweight code/runtime metadata for callers without a tracker."""

    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    selected_env = {
        key: os.environ[key]
        for key in (env_keys or ())
        if key in os.environ
    }
    if selected_env:
        environment["env"] = selected_env
    digest = hashlib.sha256(
        json.dumps(_jsonable(environment), sort_keys=True).encode()
    ).hexdigest()
    return LineageContext(
        provider=provider,
        code_ref=_first_env(
            "GITHUB_SHA",
            "GIT_COMMIT",
            "CI_COMMIT_SHA",
            "BUILD_VCS_NUMBER",
        ),
        environment=environment,
        environment_digest=f"sha256:{digest}",
    )


def _adapter_metadata(adapter: str) -> tuple[str, str]:
    return _ADAPTERS.get(adapter, (adapter.replace("-", "_"), adapter))


def _context_from_mapping(raw: Mapping[str, Any]) -> LineageContext:
    payload = dict(raw.get("lineage_context") or raw)
    provider = _string_or_none(
        _first_present(payload, "provider", "tracker", "orchestrator", "system")
    )
    external_refs = _mapping_or_empty(
        _first_present(payload, "external_refs", "refs", "reference")
    )
    external_run_id = _string_or_none(
        _first_present(
            payload,
            "external_run_id",
            "run_id",
            "mlflow_run_id",
            "wandb_run_id",
            "openlineage_run_id",
        )
    )
    if external_run_id is None:
        for key in ("mlflow_run_id", "wandb_run_id", "openlineage_run_id"):
            external_run_id = _string_or_none(external_refs.get(key))
            if external_run_id:
                break
    if provider is None:
        for key in ("mlflow_run_id", "wandb_run_id", "openlineage_run_id"):
            if key in payload or key in external_refs:
                provider = key.removesuffix("_run_id")
                break
    context = LineageContext(
        provider=provider,
        external_run_id=external_run_id,
        external_job_id=_string_or_none(
            _first_present(payload, "external_job_id", "job_id", "task_id")
        ),
        external_parent_run_id=_string_or_none(
            _first_present(payload, "external_parent_run_id", "parent_run_id")
        ),
        external_url=_string_or_none(
            _first_present(payload, "external_url", "run_url", "url")
        ),
        code_ref=_string_or_none(
            _first_present(
                payload,
                "code_ref",
                "code_version",
                "git_sha",
                "git_commit",
                "commit_sha",
            )
        ),
        environment_digest=_string_or_none(
            _first_present(
                payload,
                "environment_digest",
                "env_digest",
                "container_digest",
                "image_digest",
            )
        ),
        environment=_mapping_or_empty(_first_present(payload, "environment", "env")),
        external_refs=external_refs,
        artifact_refs=_artifact_refs(
            _first_present(payload, "artifact_refs", "artifacts", "external_artifacts")
        ),
        facets=_mapping_or_empty(_first_present(payload, "facets", "external_facets")),
        status=_string_or_none(_first_present(payload, "status", "external_status")),
        error=_string_or_none(_first_present(payload, "error", "external_error")),
    )
    return context


def _load_json_context(value: str) -> Mapping[str, Any]:
    path = Path(value)
    if path.exists():
        value = path.read_text()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LineageHookError(
            f"invalid lineage context JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise LineageHookError("lineage context JSON must be an object")
    return parsed


def _require_adapter(adapter: str, module_name: str, extra: str) -> None:
    try:
        available = importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    if not available:
        raise LineageHookError(
            f"optional lineage hook adapter {adapter!r} is not installed; install "
            f"optional extra/plugin '{extra}' or provide a plugin exposing module {module_name!r}"
        )


def _mlflow_context() -> LineageContext:
    mlflow = importlib.import_module("mlflow")
    active_run = getattr(mlflow, "active_run", lambda: None)()
    if active_run is None:
        return LineageContext(provider="mlflow")
    info = getattr(active_run, "info", active_run)
    data = getattr(active_run, "data", None)
    tags = getattr(data, "tags", {}) if data is not None else {}
    artifact_uri = _string_or_none(getattr(info, "artifact_uri", None))
    artifact_refs = ({"provider": "mlflow", "artifact_uri": artifact_uri},) if artifact_uri else ()
    return LineageContext(
        provider="mlflow",
        external_run_id=_string_or_none(getattr(info, "run_id", None)),
        external_job_id=_string_or_none(getattr(info, "experiment_id", None)),
        external_url=_string_or_none(tags.get("mlflow.runName") if isinstance(tags, Mapping) else None),
        artifact_refs=artifact_refs,
        external_refs={
            "mlflow_experiment_id": _string_or_none(getattr(info, "experiment_id", None)),
            "mlflow_status": _string_or_none(getattr(info, "status", None)),
        },
    )


def _wandb_context() -> LineageContext:
    wandb = importlib.import_module("wandb")
    run = getattr(wandb, "run", None)
    if run is None:
        return LineageContext(provider="wandb")
    artifact_refs = []
    for attr in ("path", "group", "job_type"):
        value = _string_or_none(getattr(run, attr, None))
        if value:
            artifact_refs.append({"provider": "wandb", attr: value})
    return LineageContext(
        provider="wandb",
        external_run_id=_string_or_none(getattr(run, "id", None)),
        external_job_id=_string_or_none(getattr(run, "project", None)),
        external_url=_string_or_none(getattr(run, "url", None)),
        artifact_refs=tuple(artifact_refs),
        external_refs={
            "wandb_name": _string_or_none(getattr(run, "name", None)),
            "wandb_entity": _string_or_none(getattr(run, "entity", None)),
        },
    )


def _openlineage_context() -> LineageContext:
    namespace = os.getenv("OPENLINEAGE_NAMESPACE") or "default"
    job_name = os.getenv("OPENLINEAGE_JOB_NAME")
    run_id = os.getenv("OPENLINEAGE_RUN_ID")
    return LineageContext(
        provider="openlineage",
        external_run_id=run_id,
        external_job_id=job_name,
        facets={"openlineage": {"namespace": namespace, "job_name": job_name or ""}},
    )


def _artifact_refs(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (dict(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        refs = []
        for item in value:
            if isinstance(item, Mapping):
                refs.append(dict(item))
            else:
                refs.append({"value": item})
        return tuple(refs)
    return ({"value": value},)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LineageHookError("lineage context mapping fields must be JSON objects")
    return {str(key): item for key, item in value.items() if item is not None}


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ref_key(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def _set_if(refs: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and key not in refs:
        refs[key] = value


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, (Mapping, list, tuple, set)) and not value:
        return False
    return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _assert_json_ready(payload: Mapping[str, Any], label: str) -> None:
    try:
        json.dumps(_jsonable(payload), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise LineageHookError(f"{label} is not JSON-ready: {exc}") from exc


def _redact_context_payload(value: Any, fragments: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            text = str(key).lower()
            if any(fragment and fragment in text for fragment in fragments):
                continue
            redacted[str(key)] = _redact_context_payload(item, fragments)
        return redacted
    if isinstance(value, list):
        return [_redact_context_payload(item, fragments) for item in value]
    return value


def _is_hook(value: Any) -> bool:
    return hasattr(value, "before_execution") and hasattr(value, "after_execution")
