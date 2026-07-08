"""External lineage and metadata-system projections.

The canonical lineage graph stays in Lance tables. This module projects that
graph into common metadata-system shapes and keeps the external IDs
algorithmically reversible to the canonical ``lineage_artifacts.artifact_id``.
"""

from __future__ import annotations

import base64
import hashlib
import heapq
import importlib.util
import json
import os
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, unquote
from urllib.request import Request, urlopen

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import LINEAGE_DELIVERY_ATTEMPTS_SCHEMA

DEFAULT_PRODUCER = "https://github.com/lancedb/lancedb-robotics"
OPENLINEAGE_SCHEMA_URL = "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent"
LANCE_ARTIFACT_FACET_SCHEMA_URL = (
    "https://lancedb.github.io/lancedb-robotics/facets/lineage-artifact.json"
)
LANCE_EXECUTION_FACET_SCHEMA_URL = (
    "https://lancedb.github.io/lancedb-robotics/facets/lineage-execution.json"
)
LANCE_URN_PREFIX = "urn:lancedb-robotics:artifact:"
DATAHUB_DATASET_URN_PREFIX = (
    "urn:li:dataset:(urn:li:dataPlatform:lancedb-robotics,"
)

_KNOWN_ADAPTERS = {
    "openlineage": ("openlineage.client", "openlineage"),
    "openlineage-client": ("openlineage.client", "openlineage"),
    "marquez": ("openlineage.client", "openlineage"),
    "datahub": ("datahub.emitter.rest_emitter", "datahub"),
    "datahub-rest": ("datahub.emitter.rest_emitter", "datahub"),
    "mlflow": ("mlflow", "mlflow"),
    "wandb": ("wandb", "wandb"),
    "dvc": ("dvc", "dvc"),
    "lakefs": ("lakefs", "lakefs"),
    "kubeflow": ("ml_metadata", "kubeflow-mlmd"),
    "mlmd": ("ml_metadata", "kubeflow-mlmd"),
}


class LineageIntegrationError(Exception):
    """Raised when an external lineage projection cannot be produced."""


@dataclass(frozen=True)
class OpenLineageExportReport:
    """JSON-ready OpenLineage export payload."""

    lake_uri: str
    producer: str
    dry_run: bool
    events: tuple[dict[str, Any], ...]
    artifact_urns: tuple[dict[str, str], ...]

    @property
    def event_count(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "producer": self.producer,
            "dry_run": self.dry_run,
            "event_count": self.event_count,
            "artifact_urns": list(self.artifact_urns),
            "events": list(self.events),
        }


@dataclass(frozen=True)
class DataHubLineageReport:
    """JSON-ready DataHub-style upstream/downstream edge payload."""

    lake_uri: str
    dry_run: bool
    edges: tuple[dict[str, Any], ...]
    artifact_urns: tuple[dict[str, str], ...]

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "dry_run": self.dry_run,
            "edge_count": self.edge_count,
            "artifact_urns": list(self.artifact_urns),
            "edges": list(self.edges),
        }


@dataclass(frozen=True)
class LineageDeliveryAttempt:
    """One persisted or skipped delivery attempt for an external lineage item."""

    attempt_id: str
    backend: str
    target: str
    payload_kind: str
    payload_digest: str
    payload_count: int
    mode: str
    status: str
    remote_response_ids: tuple[str, ...] = ()
    error: str | None = None
    metadata: dict[str, str] | None = None
    created_by: str | None = None
    created_at: datetime | None = None
    persisted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "backend": self.backend,
            "target": self.target,
            "payload_kind": self.payload_kind,
            "payload_digest": self.payload_digest,
            "payload_count": self.payload_count,
            "mode": self.mode,
            "status": self.status,
            "remote_response_ids": list(self.remote_response_ids),
            "error": self.error,
            "metadata": dict(self.metadata or {}),
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "persisted": self.persisted,
        }


@dataclass(frozen=True)
class LineageDeliveryReport:
    """Summary of an emit or retry operation against an external lineage target."""

    lake_uri: str
    backend: str
    target: str
    mode: str
    status: str
    attempts: tuple[LineageDeliveryAttempt, ...]

    @property
    def payload_count(self) -> int:
        return sum(attempt.payload_count for attempt in self.attempts)

    @property
    def delivered_count(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.status == "delivered")

    @property
    def already_delivered_count(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.status == "already-delivered")

    @property
    def failed_count(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.status == "failed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "backend": self.backend,
            "target": self.target,
            "mode": self.mode,
            "status": self.status,
            "payload_count": self.payload_count,
            "attempt_count": len(self.attempts),
            "delivered_count": self.delivered_count,
            "already_delivered_count": self.already_delivered_count,
            "failed_count": self.failed_count,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def export_openlineage(
    lake: Lake,
    *,
    refresh: bool = True,
    dry_run: bool = True,
    producer: str = DEFAULT_PRODUCER,
) -> OpenLineageExportReport:
    """Project canonical graph executions to OpenLineage RunEvent dictionaries."""

    if refresh:
        lake.lineage.refresh_graph()
    artifacts = _artifact_rows(lake)
    executions = _execution_rows(lake)
    events = tuple(
        _openlineage_event(execution, artifacts, lake_uri=lake.uri, producer=producer)
        for execution in sorted(executions, key=lambda row: row["execution_id"])
    )
    return OpenLineageExportReport(
        lake_uri=lake.uri,
        producer=producer,
        dry_run=dry_run,
        events=events,
        artifact_urns=_artifact_urn_rows(artifacts, backend="openlineage"),
    )


def emit_openlineage(
    lake: Lake,
    *,
    refresh: bool = True,
    producer: str = DEFAULT_PRODUCER,
    target: str | None = None,
    endpoint_url: str | None = None,
    auth_ref: str | None = None,
    headers: Mapping[str, str] | None = None,
    client: Any | None = None,
    adapter: str = "openlineage",
    retry: bool = False,
    created_by: str | None = None,
) -> LineageDeliveryReport:
    """POST OpenLineage RunEvent payloads and persist idempotent delivery state."""

    report = export_openlineage(
        lake,
        refresh=refresh,
        dry_run=False,
        producer=producer,
    )
    return _deliver_payloads(
        lake,
        backend="openlineage",
        payload_kind="openlineage-run-event",
        payloads=report.events,
        target=target,
        endpoint_url=endpoint_url,
        auth_ref=auth_ref,
        headers=headers,
        client=client,
        adapter=adapter,
        retry=retry,
        created_by=created_by,
    )


def export_datahub(
    lake: Lake,
    *,
    refresh: bool = True,
    dry_run: bool = True,
) -> DataHubLineageReport:
    """Project canonical graph edges to DataHub-style upstream lineage payloads."""

    if refresh:
        lake.lineage.refresh_graph()
    artifacts = _artifact_rows(lake)
    edges = []
    for edge in sorted(
        lake.table("lineage_edges").to_arrow().to_pylist(),
        key=lambda row: (row["edge_type"], row["from_artifact_id"], row["to_artifact_id"], row["edge_id"]),
    ):
        upstream = artifacts.get(edge["from_artifact_id"])
        downstream = artifacts.get(edge["to_artifact_id"])
        if upstream is None or downstream is None:
            continue
        edges.append(_datahub_edge(edge, upstream, downstream))
    return DataHubLineageReport(
        lake_uri=lake.uri,
        dry_run=dry_run,
        edges=tuple(edges),
        artifact_urns=_artifact_urn_rows(artifacts, backend="datahub"),
    )


def emit_datahub(
    lake: Lake,
    *,
    refresh: bool = True,
    target: str | None = None,
    endpoint_url: str | None = None,
    auth_ref: str | None = None,
    headers: Mapping[str, str] | None = None,
    client: Any | None = None,
    adapter: str = "datahub",
    retry: bool = False,
    created_by: str | None = None,
) -> LineageDeliveryReport:
    """POST DataHub-style lineage edge payloads and persist delivery state."""

    report = export_datahub(lake, refresh=refresh, dry_run=False)
    return _deliver_payloads(
        lake,
        backend="datahub",
        payload_kind="datahub-upstream-lineage-edge",
        payloads=report.edges,
        target=target,
        endpoint_url=endpoint_url,
        auth_ref=auth_ref,
        headers=headers,
        client=client,
        adapter=adapter,
        retry=retry,
        created_by=created_by,
    )


def retry_lineage_delivery(
    lake: Lake,
    backend: str,
    *,
    refresh: bool = True,
    target: str | None = None,
    endpoint_url: str | None = None,
    auth_ref: str | None = None,
    headers: Mapping[str, str] | None = None,
    client: Any | None = None,
    adapter: str | None = None,
    producer: str = DEFAULT_PRODUCER,
    created_by: str | None = None,
) -> LineageDeliveryReport:
    """Retry delivery for the selected backend, skipping already delivered digests."""

    normalized = _normalize_backend(backend)
    if normalized == "openlineage":
        return emit_openlineage(
            lake,
            refresh=refresh,
            producer=producer,
            target=target,
            endpoint_url=endpoint_url,
            auth_ref=auth_ref,
            headers=headers,
            client=client,
            adapter=adapter or "openlineage",
            retry=True,
            created_by=created_by,
        )
    if normalized == "datahub":
        return emit_datahub(
            lake,
            refresh=refresh,
            target=target,
            endpoint_url=endpoint_url,
            auth_ref=auth_ref,
            headers=headers,
            client=client,
            adapter=adapter or "datahub",
            retry=True,
            created_by=created_by,
        )
    raise LineageIntegrationError(
        f"unknown lineage delivery backend {backend!r}; expected openlineage or datahub"
    )


def lineage_delivery_attempts(
    lake: Lake,
    *,
    backend: str | None = None,
    target: str | None = None,
    status: str | None = None,
) -> tuple[LineageDeliveryAttempt, ...]:
    """Return persisted lineage delivery attempts, optionally filtered."""

    rows = lake.table("lineage_delivery_attempts").to_arrow().to_pylist()
    backend_value = _normalize_backend(backend) if backend else None
    target_value = str(target).strip() if target else None
    status_value = str(status).strip().lower() if status else None
    attempts = []
    for row in rows:
        attempt = _attempt_from_row(row)
        if backend_value and attempt.backend != backend_value:
            continue
        if target_value and attempt.target != target_value:
            continue
        if status_value and attempt.status != status_value:
            continue
        attempts.append(attempt)
    return tuple(
        sorted(
            attempts,
            key=lambda attempt: (
                attempt.created_at.isoformat() if attempt.created_at else "",
                attempt.attempt_id,
            ),
        )
    )


def external_artifact_urn(artifact_id: str, *, backend: str = "openlineage") -> str:
    """Return a stable external URN for a canonical lineage artifact id."""

    artifact_id = str(artifact_id or "").strip()
    if not artifact_id:
        raise LineageIntegrationError("artifact_id is required")
    encoded = quote(artifact_id, safe="")
    normalized = str(backend or "openlineage").strip().lower()
    if normalized in {"openlineage", "lancedb", "lancedb-robotics", "default"}:
        return f"{LANCE_URN_PREFIX}{encoded}"
    if normalized == "datahub":
        return f"{DATAHUB_DATASET_URN_PREFIX}{encoded},PROD)"
    raise LineageIntegrationError(
        f"unknown external lineage backend {backend!r}; expected openlineage or datahub"
    )


def artifact_id_from_external_urn(urn: str) -> str:
    """Recover the canonical artifact id encoded in an exported external URN."""

    value = str(urn or "").strip()
    if value.startswith(LANCE_URN_PREFIX):
        return _decode_artifact_id(value.removeprefix(LANCE_URN_PREFIX), urn)
    if value.startswith(DATAHUB_DATASET_URN_PREFIX) and value.endswith(")"):
        body = value.removeprefix(DATAHUB_DATASET_URN_PREFIX)[:-1]
        encoded, _env = body.rsplit(",", 1)
        return _decode_artifact_id(encoded, urn)
    raise LineageIntegrationError(
        f"unsupported external artifact URN {urn!r}; expected a lancedb-robotics "
        "or DataHub dataset URN emitted by this package"
    )


def resolve_external_urn(lake: Lake, urn: str) -> str:
    """Resolve an exported external URN to an artifact currently present in the graph."""

    artifact_id = artifact_id_from_external_urn(urn)
    artifacts = _artifact_rows(lake)
    if artifact_id not in artifacts:
        raise LineageIntegrationError(
            f"external URN resolves to unknown artifact {artifact_id!r}; run "
            "lake.lineage.refresh_graph() or import the referenced artifact first"
        )
    return artifact_id


def require_integration_adapter(adapter: str) -> dict[str, str]:
    """Validate that an optional external-system adapter package is importable."""

    normalized = str(adapter or "").strip().lower()
    if not normalized:
        raise LineageIntegrationError("adapter name is required")
    module, extra = _KNOWN_ADAPTERS.get(
        normalized,
        (normalized.replace("-", "_"), normalized),
    )
    try:
        available = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    if not available:
        raise LineageIntegrationError(
            f"optional integration adapter {adapter!r} is not installed; install "
            f"optional extra/plugin '{extra}' or provide a plugin exposing module {module!r}"
        )
    return {"adapter": normalized, "module": module, "optional_extra": extra}


def _deliver_payloads(
    lake: Lake,
    *,
    backend: str,
    payload_kind: str,
    payloads: tuple[dict[str, Any], ...],
    target: str | None,
    endpoint_url: str | None,
    auth_ref: str | None,
    headers: Mapping[str, str] | None,
    client: Any | None,
    adapter: str,
    retry: bool,
    created_by: str | None,
) -> LineageDeliveryReport:
    backend = _normalize_backend(backend)
    mode = "retry" if retry else "emit"
    runtime_endpoint, runtime_headers = _runtime_emitter_config(
        backend,
        endpoint_url=endpoint_url,
        auth_ref=auth_ref,
        headers=headers,
    )
    target_value = _delivery_target(
        backend,
        target=target,
        endpoint_url=runtime_endpoint,
    )
    delivered_digests = _successful_delivery_digests(
        lake,
        backend=backend,
        target=target_value,
        payload_kind=payload_kind,
    )
    emitter_client = client
    attempts: list[LineageDeliveryAttempt] = []
    for ordinal, payload in enumerate(payloads):
        payload_digest = _payload_digest(payload)
        item_metadata = {"ordinal": str(ordinal)}
        if payload_digest in delivered_digests:
            attempts.append(
                _memory_attempt(
                    backend=backend,
                    target=target_value,
                    payload_kind=payload_kind,
                    payload_digest=payload_digest,
                    mode=mode,
                    status="already-delivered",
                    metadata=item_metadata,
                    created_by=created_by,
                )
            )
            continue

        if emitter_client is None:
            emitter_client = _default_emitter_client(
                backend,
                adapter=adapter,
                endpoint_url=runtime_endpoint,
                headers=runtime_headers,
            )

        try:
            response = _invoke_emitter(
                emitter_client,
                payload,
                backend=backend,
                payload_kind=payload_kind,
            )
        except Exception as exc:  # noqa: BLE001 - persisted as delivery error state
            attempt = _persist_attempt(
                lake,
                backend=backend,
                target=target_value,
                payload_kind=payload_kind,
                payload_digest=payload_digest,
                mode=mode,
                status="failed",
                remote_response_ids=(),
                error=str(exc) or type(exc).__name__,
                metadata=item_metadata,
                created_by=created_by,
            )
            attempts.append(attempt)
            continue

        attempt = _persist_attempt(
            lake,
            backend=backend,
            target=target_value,
            payload_kind=payload_kind,
            payload_digest=payload_digest,
            mode=mode,
            status="delivered",
            remote_response_ids=_remote_response_ids(response),
            error=None,
            metadata={
                **item_metadata,
                "response_digest": _payload_digest(_jsonable_response(response)),
            },
            created_by=created_by,
        )
        delivered_digests.add(payload_digest)
        attempts.append(attempt)

    return LineageDeliveryReport(
        lake_uri=lake.uri,
        backend=backend,
        target=target_value,
        mode=mode,
        status=_delivery_report_status(attempts),
        attempts=tuple(attempts),
    )


def _default_emitter_client(
    backend: str,
    *,
    adapter: str,
    endpoint_url: str | None,
    headers: Mapping[str, str],
) -> Any:
    if endpoint_url:
        return _HttpJsonEmitter(endpoint_url, headers=headers)
    info = require_integration_adapter(adapter)
    try:
        return _package_emitter_client(backend, endpoint_url=endpoint_url, headers=headers)
    except Exception as exc:  # noqa: BLE001 - normalize optional-client construction
        raise LineageIntegrationError(
            f"optional integration adapter {adapter!r} is installed via module "
            f"{info['module']!r}, but no usable emitter client could be constructed; "
            "provide endpoint_url, auth_ref, or an injected client"
        ) from exc


class _HttpJsonEmitter:
    def __init__(self, endpoint_url: str, *, headers: Mapping[str, str]) -> None:
        self.endpoint_url = endpoint_url
        self.headers = dict(headers)

    def emit(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = _canonical_json(payload).encode()
        request = Request(
            self.endpoint_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                **self.headers,
            },
            method="POST",
        )
        with urlopen(request, timeout=30.0) as response:
            raw = response.read().decode()
            body: Any = raw
            if raw:
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = raw
            return {
                "status": getattr(response, "status", None),
                "reason": getattr(response, "reason", None),
                "body": body,
            }


def _package_emitter_client(
    backend: str,
    *,
    endpoint_url: str | None,
    headers: Mapping[str, str],
) -> Any:
    if backend == "openlineage":
        return _openlineage_package_client(endpoint_url=endpoint_url, headers=headers)
    if backend == "datahub":
        return _datahub_package_client(endpoint_url=endpoint_url, headers=headers)
    raise LineageIntegrationError(
        f"unknown lineage delivery backend {backend!r}; expected openlineage or datahub"
    )


def _openlineage_package_client(
    *,
    endpoint_url: str | None,
    headers: Mapping[str, str],
) -> Any:
    module = __import__("openlineage.client", fromlist=["OpenLineageClient"])
    client_cls = getattr(module, "OpenLineageClient", None)
    if client_cls is None:
        client_module = __import__("openlineage.client.client", fromlist=["OpenLineageClient"])
        client_cls = client_module.OpenLineageClient
    if endpoint_url:
        try:
            return client_cls(url=endpoint_url, headers=dict(headers))
        except TypeError:
            return client_cls(endpoint_url)
    if hasattr(client_cls, "from_environment"):
        return client_cls.from_environment()
    return client_cls()


def _datahub_package_client(
    *,
    endpoint_url: str | None,
    headers: Mapping[str, str],
) -> Any:
    module = __import__("datahub.emitter.rest_emitter", fromlist=["DatahubRestEmitter"])
    client_cls = module.DatahubRestEmitter
    token = _bearer_token(headers)
    if endpoint_url and token:
        return client_cls(gms_server=endpoint_url, token=token)
    if endpoint_url:
        return client_cls(gms_server=endpoint_url)
    return client_cls()


def _invoke_emitter(
    client: Any,
    payload: dict[str, Any],
    *,
    backend: str,
    payload_kind: str,
) -> Any:
    for method_name in ("emit", "send", "post", "emit_event", "emit_lineage"):
        method = getattr(client, method_name, None)
        if callable(method):
            return method(payload)
    if callable(client):
        return client(payload)
    raise LineageIntegrationError(
        f"{backend} emitter client cannot handle {payload_kind}; expected an "
        "emit/send/post/emit_event/emit_lineage method or a callable"
    )


def _runtime_emitter_config(
    backend: str,
    *,
    endpoint_url: str | None,
    auth_ref: str | None,
    headers: Mapping[str, str] | None,
) -> tuple[str | None, dict[str, str]]:
    resolved_headers = {str(key): str(value) for key, value in (headers or {}).items()}
    env_endpoint, env_headers = _runtime_auth_ref_config(backend, auth_ref)
    for key, value in env_headers.items():
        resolved_headers.setdefault(key, value)
    return endpoint_url or env_endpoint or _default_endpoint_env(backend), resolved_headers


def _runtime_auth_ref_config(
    backend: str,
    auth_ref: str | None,
) -> tuple[str | None, dict[str, str]]:
    headers: dict[str, str] = {}
    if not auth_ref:
        return None, headers
    normalized = _normalize_auth_ref(auth_ref)
    prefix = f"LANCEDB_ROBOTICS_AUTH_{normalized}_LINEAGE"
    backend_prefix = f"{prefix}_{backend.upper()}"
    endpoint = os.getenv(f"{backend_prefix}_ENDPOINT") or os.getenv(f"{prefix}_ENDPOINT")
    headers.update(_json_env(f"{prefix}_HEADERS_JSON"))
    headers.update(_json_env(f"{backend_prefix}_HEADERS_JSON"))
    bearer = os.getenv(f"{backend_prefix}_BEARER_TOKEN") or os.getenv(f"{prefix}_BEARER_TOKEN")
    api_key = os.getenv(f"{backend_prefix}_API_KEY") or os.getenv(f"{prefix}_API_KEY")
    if bearer:
        headers.setdefault("Authorization", f"Bearer {bearer}")
    elif api_key:
        headers.setdefault("Authorization", f"Bearer {api_key}")
    return endpoint, headers


def _default_endpoint_env(backend: str) -> str | None:
    if backend == "openlineage":
        return os.getenv("OPENLINEAGE_URL")
    if backend == "datahub":
        return os.getenv("DATAHUB_GMS_URL")
    return None


def _normalize_auth_ref(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() else "_"
        for char in str(value).strip().upper()
    ).strip("_")
    if not normalized:
        raise LineageIntegrationError("auth_ref cannot be empty")
    return normalized


def _json_env(name: str) -> dict[str, str]:
    raw = os.getenv(name)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LineageIntegrationError(f"{name} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise LineageIntegrationError(f"{name} must be a JSON object")
    return {str(key): str(value) for key, value in parsed.items()}


def _bearer_token(headers: Mapping[str, str]) -> str | None:
    value = headers.get("Authorization") or headers.get("authorization")
    if not value:
        return os.getenv("DATAHUB_TOKEN")
    prefix = "Bearer "
    return value[len(prefix) :] if value.startswith(prefix) else value


def _delivery_target(
    backend: str,
    *,
    target: str | None,
    endpoint_url: str | None,
) -> str:
    explicit = str(target or "").strip()
    if explicit:
        return explicit
    if endpoint_url:
        digest = hashlib.sha256(endpoint_url.encode()).hexdigest()[:16]
        return f"{backend}:endpoint:{digest}"
    return f"{backend}:default"


def _normalize_backend(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized in {"openlineage", "openlineage-client", "marquez"}:
        return "openlineage"
    if normalized in {"datahub", "datahub-rest"}:
        return "datahub"
    return normalized


def _successful_delivery_digests(
    lake: Lake,
    *,
    backend: str,
    target: str,
    payload_kind: str,
) -> set[str]:
    result: set[str] = set()
    for row in lake.table("lineage_delivery_attempts").to_arrow().to_pylist():
        if (
            row.get("backend") == backend
            and row.get("target") == target
            and row.get("payload_kind") == payload_kind
            and row.get("status") == "delivered"
            and row.get("payload_digest")
        ):
            result.add(str(row["payload_digest"]))
    return result


def _persist_attempt(
    lake: Lake,
    *,
    backend: str,
    target: str,
    payload_kind: str,
    payload_digest: str,
    mode: str,
    status: str,
    remote_response_ids: tuple[str, ...],
    error: str | None,
    metadata: Mapping[str, Any],
    created_by: str | None,
) -> LineageDeliveryAttempt:
    now = datetime.now(UTC)
    attempt = LineageDeliveryAttempt(
        attempt_id=_delivery_attempt_id(
            backend=backend,
            target=target,
            payload_kind=payload_kind,
            payload_digest=payload_digest,
            mode=mode,
            status=status,
            created_at=now,
        ),
        backend=backend,
        target=target,
        payload_kind=payload_kind,
        payload_digest=payload_digest,
        payload_count=1,
        mode=mode,
        status=status,
        remote_response_ids=tuple(remote_response_ids),
        error=error,
        metadata={str(key): _metadata_value(value) for key, value in metadata.items()},
        created_by=created_by,
        created_at=now,
    )
    row = {
        "attempt_id": attempt.attempt_id,
        "backend": attempt.backend,
        "target": attempt.target,
        "payload_kind": attempt.payload_kind,
        "payload_digest": attempt.payload_digest,
        "payload_count": attempt.payload_count,
        "mode": attempt.mode,
        "status": attempt.status,
        "remote_response_ids": list(attempt.remote_response_ids),
        "error": attempt.error,
        "metadata": _kv_items(attempt.metadata),
        "created_by": attempt.created_by,
        "created_at": attempt.created_at,
    }
    table = pa.Table.from_pylist([row], schema=LINEAGE_DELIVERY_ATTEMPTS_SCHEMA)
    lake.table("lineage_delivery_attempts").add(table)
    return attempt


def _memory_attempt(
    *,
    backend: str,
    target: str,
    payload_kind: str,
    payload_digest: str,
    mode: str,
    status: str,
    metadata: Mapping[str, Any],
    created_by: str | None,
) -> LineageDeliveryAttempt:
    now = datetime.now(UTC)
    return LineageDeliveryAttempt(
        attempt_id=_delivery_attempt_id(
            backend=backend,
            target=target,
            payload_kind=payload_kind,
            payload_digest=payload_digest,
            mode=mode,
            status=status,
            created_at=now,
        ),
        backend=backend,
        target=target,
        payload_kind=payload_kind,
        payload_digest=payload_digest,
        payload_count=1,
        mode=mode,
        status=status,
        metadata={str(key): _metadata_value(value) for key, value in metadata.items()},
        created_by=created_by,
        created_at=now,
        persisted=False,
    )


def _delivery_attempt_id(
    *,
    backend: str,
    target: str,
    payload_kind: str,
    payload_digest: str,
    mode: str,
    status: str,
    created_at: datetime,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "backend": backend,
                "target": target,
                "payload_kind": payload_kind,
                "payload_digest": payload_digest,
                "mode": mode,
                "status": status,
                "created_at": created_at.isoformat(),
                "nonce": uuid.uuid4().hex,
            }
        ).encode()
    ).hexdigest()
    return f"lancedb-robotics:lineage-delivery:{digest[:32]}"


def _attempt_from_row(row: Mapping[str, Any]) -> LineageDeliveryAttempt:
    return LineageDeliveryAttempt(
        attempt_id=str(row.get("attempt_id") or ""),
        backend=str(row.get("backend") or ""),
        target=str(row.get("target") or ""),
        payload_kind=str(row.get("payload_kind") or ""),
        payload_digest=str(row.get("payload_digest") or ""),
        payload_count=int(row.get("payload_count") or 0),
        mode=str(row.get("mode") or ""),
        status=str(row.get("status") or ""),
        remote_response_ids=tuple(str(value) for value in row.get("remote_response_ids") or ()),
        error=row.get("error"),
        metadata=_metadata_map(row),
        created_by=row.get("created_by"),
        created_at=row.get("created_at"),
    )


def _delivery_report_status(attempts: list[LineageDeliveryAttempt]) -> str:
    if not attempts:
        return "no-op"
    failed = sum(1 for attempt in attempts if attempt.status == "failed")
    delivered = sum(1 for attempt in attempts if attempt.status == "delivered")
    already = sum(1 for attempt in attempts if attempt.status == "already-delivered")
    if failed and (delivered or already):
        return "partial"
    if failed:
        return "failed"
    if delivered:
        return "delivered"
    if already:
        return "already-delivered"
    return "no-op"


def _payload_digest(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _jsonable_response(response: Any) -> Any:
    if response is None or isinstance(response, (str, int, float, bool)):
        return response
    if isinstance(response, Mapping):
        return {str(key): _jsonable_response(value) for key, value in response.items()}
    if isinstance(response, (list, tuple)):
        return [_jsonable_response(value) for value in response]
    return str(response)


def _remote_response_ids(response: Any) -> tuple[str, ...]:
    found: list[str] = []

    def collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value and len(value) <= 512:
                found.append(value)
            return
        if isinstance(value, Mapping):
            for key in (
                "id",
                "remote_id",
                "response_id",
                "urn",
                "runId",
                "guid",
                "status",
            ):
                item = value.get(key)
                if item is not None:
                    collect(item)
            for key in ("body", "data", "result", "results"):
                if key in value:
                    collect(value[key])
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(response)
    deduped: list[str] = []
    for value in found:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _kv_items(metadata: Mapping[str, Any] | None) -> list[dict[str, str]]:
    if not metadata:
        return []
    return [
        {"key": str(key), "value": _metadata_value(value)}
        for key, value in sorted(metadata.items())
        if value is not None
    ]


def _metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _canonical_json(value)


def _openlineage_event(
    execution: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    *,
    lake_uri: str,
    producer: str,
) -> dict[str, Any]:
    inputs = [
        _openlineage_dataset(artifacts[artifact_id], lake_uri=lake_uri, producer=producer)
        for artifact_id in execution.get("input_artifact_ids") or []
        if artifact_id in artifacts
    ]
    outputs = [
        _openlineage_dataset(artifacts[artifact_id], lake_uri=lake_uri, producer=producer)
        for artifact_id in execution.get("output_artifact_ids") or []
        if artifact_id in artifacts
    ]
    execution_id = str(execution["execution_id"])
    return {
        "eventType": _openlineage_event_type(execution.get("status")),
        "eventTime": _iso_datetime(
            execution.get("finished_at")
            or execution.get("started_at")
            or execution.get("created_at")
            or datetime.now(UTC)
        ),
        "producer": producer,
        "schemaURL": OPENLINEAGE_SCHEMA_URL,
        "run": {
            "runId": _stable_uuid(execution_id),
            "facets": {
                "lancedb_robotics_execution": _execution_facet(execution, producer=producer)
            },
        },
        "job": {
            "namespace": "lancedb-robotics",
            "name": _job_name(execution),
            "facets": {
                "jobType": {
                    "_producer": producer,
                    "_schemaURL": "https://openlineage.io/spec/facets/2-0-2/JobTypeJobFacet.json",
                    "processingType": "BATCH",
                    "integration": "lancedb-robotics",
                    "jobType": str(execution.get("kind") or "lineage"),
                }
            },
        },
        "inputs": inputs,
        "outputs": outputs,
    }


def _openlineage_dataset(
    artifact: dict[str, Any],
    *,
    lake_uri: str,
    producer: str,
) -> dict[str, Any]:
    artifact_urn = external_artifact_urn(artifact["artifact_id"])
    return {
        "namespace": lake_uri,
        "name": artifact_urn,
        "facets": {
            "dataSource": {
                "_producer": producer,
                "_schemaURL": "https://openlineage.io/spec/facets/2-0-2/DatasourceDatasetFacet.json",
                "name": "lancedb-robotics",
                "uri": lake_uri,
            },
            "lancedb_robotics_artifact": _artifact_facet(
                artifact,
                artifact_urn=artifact_urn,
                producer=producer,
            ),
        },
    }


def _datahub_edge(
    edge: dict[str, Any],
    upstream: dict[str, Any],
    downstream: dict[str, Any],
) -> dict[str, Any]:
    upstream_urn = external_artifact_urn(upstream["artifact_id"], backend="datahub")
    downstream_urn = external_artifact_urn(downstream["artifact_id"], backend="datahub")
    return {
        "type": "DataHubUpstreamLineageEdge",
        "edge_type": edge["edge_type"],
        "upstreamUrn": upstream_urn,
        "downstreamUrn": downstream_urn,
        "upstream": _artifact_ref(upstream, artifact_urn=upstream_urn),
        "downstream": _artifact_ref(downstream, artifact_urn=downstream_urn),
        "lineage": {
            "edge_id": edge["edge_id"],
            "execution_id": edge.get("execution_id"),
            "metadata": _metadata_map(edge),
        },
        "auditStamp": {
            "actor": "urn:li:corpuser:lancedb-robotics",
            "time": _epoch_ms(edge.get("created_at")),
        },
    }


def _artifact_facet(
    artifact: dict[str, Any],
    *,
    artifact_urn: str,
    producer: str,
) -> dict[str, Any]:
    return {
        "_producer": producer,
        "_schemaURL": LANCE_ARTIFACT_FACET_SCHEMA_URL,
        **_artifact_ref(artifact, artifact_urn=artifact_urn),
    }


def _artifact_ref(artifact: dict[str, Any], *, artifact_urn: str) -> dict[str, Any]:
    return {
        "artifact_id": artifact["artifact_id"],
        "artifact_urn": artifact_urn,
        "kind": artifact.get("kind"),
        "name": artifact.get("name"),
        "table_name": artifact.get("table_name"),
        "table_version": artifact.get("table_version"),
        "table_tag": artifact.get("table_tag"),
        "row_grain": artifact.get("row_grain"),
        "row_ids": list(artifact.get("row_ids") or []),
        "source_uri": artifact.get("source_uri"),
        "source_id": artifact.get("source_id"),
        "digest": artifact.get("digest"),
        "producer_execution_id": artifact.get("producer_execution_id"),
        "metadata": _metadata_map(artifact),
    }


def _execution_facet(execution: dict[str, Any], *, producer: str) -> dict[str, Any]:
    return {
        "_producer": producer,
        "_schemaURL": LANCE_EXECUTION_FACET_SCHEMA_URL,
        "execution_id": execution["execution_id"],
        "kind": execution.get("kind"),
        "name": execution.get("name"),
        "transform_id": execution.get("transform_id"),
        "status": execution.get("status"),
        "code_ref": execution.get("code_ref"),
        "provider": execution.get("provider"),
        "params": _json_dict(execution.get("params_json")),
        "environment": _json_dict(execution.get("environment_json")),
        "input_table_versions": list(execution.get("input_table_versions") or []),
        "output_table_versions": list(execution.get("output_table_versions") or []),
        "metadata": _metadata_map(execution),
    }


def _artifact_rows(lake: Lake) -> dict[str, dict[str, Any]]:
    return {
        row["artifact_id"]: row
        for row in lake.table("lineage_artifacts").to_arrow().to_pylist()
    }


def _execution_rows(lake: Lake) -> list[dict[str, Any]]:
    return lake.table("lineage_executions").to_arrow().to_pylist()


def _artifact_urn_rows(
    artifacts: dict[str, dict[str, Any]],
    *,
    backend: str,
) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "artifact_id": artifact_id,
            "artifact_urn": external_artifact_urn(artifact_id, backend=backend),
        }
        for artifact_id in sorted(artifacts)
    )


def _openlineage_event_type(status: str | None) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"failed", "failure", "error"}:
        return "FAIL"
    if normalized in {"aborted", "cancelled", "canceled"}:
        return "ABORT"
    if normalized in {"running", "started", "start"}:
        return "START"
    return "COMPLETE"


def _job_name(execution: dict[str, Any]) -> str:
    name = execution.get("name") or execution.get("kind") or execution["execution_id"]
    return str(name)


def _metadata_map(row: dict[str, Any], *, column: str = "metadata") -> dict[str, str]:
    result: dict[str, str] = {}
    for item in row.get(column) or []:
        if not isinstance(item, dict) or item.get("key") is None:
            continue
        result[str(item["key"])] = "" if item.get("value") is None else str(item["value"])
    return result


def _json_dict(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        parsed = json.loads(str(payload))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stable_uuid(value: str) -> str:
    digest = hashlib.sha1(value.encode()).hexdigest()[:32]
    return str(uuid.UUID(hex=digest))


def _iso_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _epoch_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def _decode_artifact_id(encoded: str, original_urn: str) -> str:
    artifact_id = unquote(encoded)
    if not artifact_id:
        raise LineageIntegrationError(f"external artifact URN has no artifact id: {original_urn!r}")
    return artifact_id


# ---------------------------------------------------------------------------
# Backlog 0105: bounded, resumable external lineage export + bulk URN catalog.
#
# The full-materialization exporters above (``export_openlineage`` /
# ``export_datahub``) are fine for local validation and small pipelines but
# hold every event/edge/artifact in memory. The surfaces below add bounded
# pages, stable continuation tokens, pushed-down filters, and NDJSON streaming
# so fleet/enterprise lakes with millions of artifacts export without unbounded
# memory. Pages are consecutive slices of the canonical ids (``artifact_id`` /
# ``execution_id`` / ``edge_id``) sorted ascending, selected with a bounded
# top-k heap over a batched scan: the union of pages equals the full result
# with no duplicates, and peak rows in memory stay ~ (page_size + one batch)
# regardless of table size.
# ---------------------------------------------------------------------------

_EXPORT_SCAN_BATCH_SIZE = 1024
_OPENLINEAGE_PAYLOAD_KIND = "openlineage-run-event"
_DATAHUB_PAYLOAD_KIND = "datahub-upstream-lineage-edge"
_URN_CATALOG_PAYLOAD_KIND = "artifact-urn-catalog"
_EXPORT_SUMMARY_TYPE = "lineage-export-summary"


@dataclass(frozen=True)
class LineageExportFilters:
    """Normalized filters shared by every paged external export.

    Not every filter applies to every payload kind. The URN catalog honors
    ``artifact_kind``, ``created_*``, and ``table_versions``; OpenLineage events
    honor ``execution_kind`` and ``created_*`` (against the execution); DataHub
    edges honor ``created_*`` (against the edge) plus ``artifact_kind`` /
    ``table_versions`` applied to *both* endpoint artifacts.
    """

    artifact_kind: str | None = None
    execution_kind: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    table_versions: tuple[tuple[str, int], ...] = ()

    @classmethod
    def build(
        cls,
        *,
        artifact_kind: str | None = None,
        execution_kind: str | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
        table_versions: Mapping[str, int] | Iterable[str | tuple[str, int]] | None = None,
    ) -> LineageExportFilters:
        table_version_map = _coerce_table_versions(table_versions)
        return cls(
            artifact_kind=_clean_kind(artifact_kind),
            execution_kind=_clean_kind(execution_kind),
            created_after=_coerce_export_datetime(created_after, "created_after"),
            created_before=_coerce_export_datetime(created_before, "created_before"),
            table_versions=tuple(sorted(table_version_map.items())),
        )

    def table_version_map(self) -> dict[str, int]:
        return dict(self.table_versions)

    def digest_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "execution_kind": self.execution_kind,
            "created_after": self.created_after.isoformat() if self.created_after else None,
            "created_before": self.created_before.isoformat() if self.created_before else None,
            "table_versions": [[table, version] for table, version in self.table_versions],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.digest_payload()


@dataclass(frozen=True)
class LineageExportPage:
    """One bounded page of an external lineage export plus its scan accounting."""

    lake_uri: str
    payload_kind: str
    backend: str
    dry_run: bool
    records: tuple[dict[str, Any], ...]
    page_size: int | None
    matched_total: int
    scanned_rows: int
    peak_retained_rows: int
    next_page_token: str | None
    truncated: bool
    filters: LineageExportFilters
    producer: str | None = None

    @property
    def record_count(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lake_uri": self.lake_uri,
            "payload_kind": self.payload_kind,
            "backend": self.backend,
            "dry_run": self.dry_run,
            "record_count": self.record_count,
            "page_size": self.page_size,
            "matched_total": self.matched_total,
            "scanned_rows": self.scanned_rows,
            "peak_retained_rows": self.peak_retained_rows,
            "next_page_token": self.next_page_token,
            "truncated": self.truncated,
            "filters": self.filters.to_dict(),
            "producer": self.producer,
            "records": list(self.records),
        }


def export_artifact_urn_catalog(
    lake: Lake,
    *,
    backend: str = "openlineage",
    page_size: int | None = None,
    page_token: str | None = None,
    filters: LineageExportFilters | None = None,
    refresh: bool = False,
) -> LineageExportPage:
    """Return a bounded page of the bulk artifact-URN catalog.

    Each record carries the canonical artifact id, its reversible backend URN,
    kind, table name/version, tag, and digest -- joinable without reparsing any
    OpenLineage or DataHub payload.
    """

    filters = filters or LineageExportFilters()
    backend_norm = _normalize_backend(backend) or "openlineage"
    if backend_norm not in {"openlineage", "datahub"}:
        raise LineageIntegrationError(
            f"unknown external lineage backend {backend!r}; expected openlineage or datahub"
        )
    if refresh and page_token is None:
        lake.lineage.refresh_graph()
    digest = _export_query_digest(_URN_CATALOG_PAYLOAD_KIND, backend_norm, filters)
    after_id = _decode_export_page_token(page_token, digest) if page_token else None
    projection = (
        "artifact_id",
        "kind",
        "name",
        "table_name",
        "table_version",
        "table_tag",
        "digest",
        "created_at",
    )
    rows, matched, scanned, peak, has_next = _page_or_collect(
        lake,
        "lineage_artifacts",
        id_column="artifact_id",
        after_id=after_id,
        page_size=page_size,
        projection=projection,
        row_matcher=_artifact_row_matches,
        filters=filters,
        sql_filter=_kind_sql_filter(filters.artifact_kind),
    )
    records = tuple(_urn_catalog_row(row, backend=backend_norm) for row in rows)
    next_token = (
        _encode_export_page_token(str(rows[-1]["artifact_id"]), digest)
        if has_next and rows
        else None
    )
    return LineageExportPage(
        lake_uri=lake.uri,
        payload_kind=_URN_CATALOG_PAYLOAD_KIND,
        backend=backend_norm,
        dry_run=True,
        records=records,
        page_size=page_size,
        matched_total=matched,
        scanned_rows=scanned,
        peak_retained_rows=peak,
        next_page_token=next_token,
        truncated=next_token is not None,
        filters=filters,
    )


def export_openlineage_page(
    lake: Lake,
    *,
    page_size: int | None = None,
    page_token: str | None = None,
    filters: LineageExportFilters | None = None,
    refresh: bool = False,
    dry_run: bool = True,
    producer: str = DEFAULT_PRODUCER,
) -> LineageExportPage:
    """Return a bounded page of OpenLineage RunEvents sorted by execution id."""

    filters = filters or LineageExportFilters()
    if refresh and page_token is None:
        lake.lineage.refresh_graph()
    digest = _export_query_digest(_OPENLINEAGE_PAYLOAD_KIND, "openlineage", filters)
    after_id = _decode_export_page_token(page_token, digest) if page_token else None
    rows, matched, scanned, peak, has_next = _page_or_collect(
        lake,
        "lineage_executions",
        id_column="execution_id",
        after_id=after_id,
        page_size=page_size,
        projection=None,
        row_matcher=_execution_row_matches,
        filters=filters,
        sql_filter=_kind_sql_filter(filters.execution_kind),
    )
    referenced: set[str] = set()
    for execution in rows:
        referenced.update(execution.get("input_artifact_ids") or [])
        referenced.update(execution.get("output_artifact_ids") or [])
    artifacts = _fetch_artifacts_by_id(lake, referenced)
    records = tuple(
        _openlineage_event(execution, artifacts, lake_uri=lake.uri, producer=producer)
        for execution in rows
    )
    next_token = (
        _encode_export_page_token(str(rows[-1]["execution_id"]), digest)
        if has_next and rows
        else None
    )
    return LineageExportPage(
        lake_uri=lake.uri,
        payload_kind=_OPENLINEAGE_PAYLOAD_KIND,
        backend="openlineage",
        dry_run=dry_run,
        records=records,
        page_size=page_size,
        matched_total=matched,
        scanned_rows=scanned,
        peak_retained_rows=peak,
        next_page_token=next_token,
        truncated=next_token is not None,
        filters=filters,
        producer=producer,
    )


def export_datahub_page(
    lake: Lake,
    *,
    page_size: int | None = None,
    page_token: str | None = None,
    filters: LineageExportFilters | None = None,
    refresh: bool = False,
    dry_run: bool = True,
) -> LineageExportPage:
    """Return a bounded page of DataHub-style edges sorted by edge id.

    When ``artifact_kind`` or ``table_versions`` filters are set, an edge is
    included only when *both* endpoint artifacts pass -- so a page may emit
    fewer than ``page_size`` records, but the cursor stays on ``edge_id`` so
    pagination remains stable and duplicate-free.
    """

    filters = filters or LineageExportFilters()
    if refresh and page_token is None:
        lake.lineage.refresh_graph()
    digest = _export_query_digest(_DATAHUB_PAYLOAD_KIND, "datahub", filters)
    after_id = _decode_export_page_token(page_token, digest) if page_token else None
    projection = (
        "edge_id",
        "edge_type",
        "from_artifact_id",
        "to_artifact_id",
        "execution_id",
        "metadata",
        "created_at",
    )
    rows, matched, scanned, peak, has_next = _page_or_collect(
        lake,
        "lineage_edges",
        id_column="edge_id",
        after_id=after_id,
        page_size=page_size,
        projection=projection,
        row_matcher=_edge_row_matches,
        filters=filters,
        sql_filter=None,
    )
    endpoints: set[str] = set()
    for edge in rows:
        endpoints.add(edge["from_artifact_id"])
        endpoints.add(edge["to_artifact_id"])
    artifacts = _fetch_artifacts_by_id(lake, endpoints)
    apply_endpoint_filter = bool(filters.artifact_kind or filters.table_version_map())
    records: list[dict[str, Any]] = []
    for edge in rows:
        upstream = artifacts.get(edge["from_artifact_id"])
        downstream = artifacts.get(edge["to_artifact_id"])
        if upstream is None or downstream is None:
            continue
        if apply_endpoint_filter and not (
            _artifact_endpoint_matches(upstream, filters)
            and _artifact_endpoint_matches(downstream, filters)
        ):
            continue
        records.append(_datahub_edge(edge, upstream, downstream))
    next_token = (
        _encode_export_page_token(str(rows[-1]["edge_id"]), digest)
        if has_next and rows
        else None
    )
    return LineageExportPage(
        lake_uri=lake.uri,
        payload_kind=_DATAHUB_PAYLOAD_KIND,
        backend="datahub",
        dry_run=dry_run,
        records=tuple(records),
        page_size=page_size,
        matched_total=matched,
        scanned_rows=scanned,
        peak_retained_rows=peak,
        next_page_token=next_token,
        truncated=next_token is not None,
        filters=filters,
    )


def iter_openlineage_ndjson(
    lake: Lake,
    *,
    page_size: int = 512,
    filters: LineageExportFilters | None = None,
    refresh: bool = False,
    dry_run: bool = True,
    producer: str = DEFAULT_PRODUCER,
    include_summary: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield OpenLineage RunEvents one record at a time, paging under the hood."""

    yield from _iter_export_ndjson(
        lake,
        _OPENLINEAGE_PAYLOAD_KIND,
        backend="openlineage",
        page_size=page_size,
        filters=filters,
        refresh=refresh,
        dry_run=dry_run,
        producer=producer,
        include_summary=include_summary,
    )


def iter_datahub_ndjson(
    lake: Lake,
    *,
    page_size: int = 512,
    filters: LineageExportFilters | None = None,
    refresh: bool = False,
    dry_run: bool = True,
    include_summary: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield DataHub-style edges one record at a time, paging under the hood."""

    yield from _iter_export_ndjson(
        lake,
        _DATAHUB_PAYLOAD_KIND,
        backend="datahub",
        page_size=page_size,
        filters=filters,
        refresh=refresh,
        dry_run=dry_run,
        include_summary=include_summary,
    )


def iter_artifact_urn_ndjson(
    lake: Lake,
    *,
    backend: str = "openlineage",
    page_size: int = 512,
    filters: LineageExportFilters | None = None,
    refresh: bool = False,
    include_summary: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield bulk URN-catalog records one at a time, paging under the hood."""

    yield from _iter_export_ndjson(
        lake,
        _URN_CATALOG_PAYLOAD_KIND,
        backend=backend,
        page_size=page_size,
        filters=filters,
        refresh=refresh,
        include_summary=include_summary,
    )


def _iter_export_ndjson(
    lake: Lake,
    payload_kind: str,
    *,
    backend: str,
    page_size: int,
    filters: LineageExportFilters | None,
    refresh: bool,
    dry_run: bool = True,
    producer: str = DEFAULT_PRODUCER,
    include_summary: bool = False,
) -> Iterator[dict[str, Any]]:
    filters = filters or LineageExportFilters()
    if page_size is None or page_size < 1:
        raise LineageIntegrationError("ndjson export requires page_size >= 1")
    if refresh:
        lake.lineage.refresh_graph()
    token: str | None = None
    total = 0
    resolved_backend = backend
    while True:
        page = _export_page_dispatch(
            lake,
            payload_kind,
            backend=backend,
            page_size=page_size,
            page_token=token,
            filters=filters,
            refresh=False,
            dry_run=dry_run,
            producer=producer,
        )
        resolved_backend = page.backend
        for record in page.records:
            total += 1
            yield record
        if page.next_page_token is None:
            break
        token = page.next_page_token
    if include_summary:
        yield {
            "type": _EXPORT_SUMMARY_TYPE,
            "payload_kind": payload_kind,
            "backend": resolved_backend,
            "record_count": total,
            "filters": filters.to_dict(),
        }


def _export_page_dispatch(
    lake: Lake,
    payload_kind: str,
    *,
    backend: str,
    page_size: int | None,
    page_token: str | None,
    filters: LineageExportFilters,
    refresh: bool,
    dry_run: bool,
    producer: str,
) -> LineageExportPage:
    if payload_kind == _OPENLINEAGE_PAYLOAD_KIND:
        return export_openlineage_page(
            lake,
            page_size=page_size,
            page_token=page_token,
            filters=filters,
            refresh=refresh,
            dry_run=dry_run,
            producer=producer,
        )
    if payload_kind == _DATAHUB_PAYLOAD_KIND:
        return export_datahub_page(
            lake,
            page_size=page_size,
            page_token=page_token,
            filters=filters,
            refresh=refresh,
            dry_run=dry_run,
        )
    if payload_kind == _URN_CATALOG_PAYLOAD_KIND:
        return export_artifact_urn_catalog(
            lake,
            backend=backend,
            page_size=page_size,
            page_token=page_token,
            filters=filters,
            refresh=refresh,
        )
    raise LineageIntegrationError(f"unknown lineage export payload kind {payload_kind!r}")


# --- paged/bounded scan primitives -----------------------------------------


class _DescendingKey:
    """String wrapper with reversed ordering, to run ``heapq`` as a max-heap."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __lt__(self, other: _DescendingKey) -> bool:
        return self.value > other.value


def _page_or_collect(
    lake: Lake,
    table: str,
    *,
    id_column: str,
    after_id: str | None,
    page_size: int | None,
    projection: Sequence[str] | None,
    row_matcher: Callable[[Mapping[str, Any], LineageExportFilters], bool],
    filters: LineageExportFilters,
    sql_filter: str | None,
) -> tuple[list[dict[str, Any]], int, int, int, bool]:
    """Return ``(rows, matched_total, scanned_rows, peak_retained_rows, has_next)``.

    ``matched_total`` is the full count of rows passing the filter (stable
    across pages); ``rows`` is the page slice with ids greater than ``after_id``.
    When ``page_size`` is ``None`` every matched row is collected (unbounded).
    """

    handle = lake.table(table)
    available = set(handle.schema.names)
    proj = [column for column in (projection or ()) if column in available] or None

    if page_size is None:
        collected: list[dict[str, Any]] = []
        scanned = 0
        for batch_rows in _stream_batches(handle, proj, sql_filter):
            scanned += len(batch_rows)
            for row in batch_rows:
                if not row_matcher(row, filters):
                    continue
                rid = str(row.get(id_column) or "")
                if after_id is not None and rid <= after_id:
                    continue
                collected.append(row)
        collected.sort(key=lambda row: str(row.get(id_column) or ""))
        return collected, len(collected), scanned, len(collected), False

    if page_size < 1:
        raise LineageIntegrationError("page_size must be >= 1")

    heap: list[tuple[_DescendingKey, int, dict[str, Any]]] = []
    cap = page_size + 1
    seq = 0
    scanned = 0
    matched = 0
    peak = 0
    for batch_rows in _stream_batches(handle, proj, sql_filter):
        scanned += len(batch_rows)
        for row in batch_rows:
            if not row_matcher(row, filters):
                continue
            matched += 1
            rid = str(row.get(id_column) or "")
            if after_id is not None and rid <= after_id:
                continue
            if len(heap) < cap:
                heapq.heappush(heap, (_DescendingKey(rid), seq, row))
                seq += 1
            elif rid < heap[0][0].value:
                heapq.heapreplace(heap, (_DescendingKey(rid), seq, row))
                seq += 1
        peak = max(peak, len(batch_rows) + len(heap))
    ordered = [entry[2] for entry in sorted(heap, key=lambda entry: entry[0].value)]
    has_next = len(ordered) > page_size
    return ordered[:page_size], matched, scanned, peak, has_next


def _stream_batches(
    handle: Any,
    projection: Sequence[str] | None,
    sql_filter: str | None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield row batches, streaming server-side when the backend supports it.

    Mirrors ``run_manifests._bounded_rows``: the ``.search().select().where()``
    scan streams via ``to_batches`` (bounded memory); backends that cannot build
    a server-side query fall back to a full materialization (correctness only,
    not memory-bounded). Reads the module-level batch size at call time so tests
    can shrink it via monkeypatch.
    """

    batch_size = _EXPORT_SCAN_BATCH_SIZE
    try:
        query = handle.search()
        if projection:
            query = query.select(list(projection))
        if sql_filter:
            query = query.where(sql_filter)
        for batch in query.to_batches(batch_size=batch_size):
            yield batch.to_pylist()
        return
    except Exception:  # noqa: BLE001 - backends without server-side scan fall back
        pass
    materialized = handle.to_arrow().to_pylist()
    for start in range(0, len(materialized), batch_size):
        yield materialized[start : start + batch_size]


def _fetch_artifacts_by_id(lake: Lake, artifact_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    ids = [str(value) for value in artifact_ids if value]
    if not ids:
        return {}
    from lancedb_robotics.lineage import _fetch_rows_by_id_in

    return _fetch_rows_by_id_in(lake, "lineage_artifacts", "artifact_id", ids)


def _urn_catalog_row(artifact: Mapping[str, Any], *, backend: str) -> dict[str, Any]:
    artifact_id = artifact["artifact_id"]
    return {
        "artifact_id": artifact_id,
        "artifact_urn": external_artifact_urn(artifact_id, backend=backend),
        "kind": artifact.get("kind"),
        "name": artifact.get("name"),
        "table_name": artifact.get("table_name"),
        "table_version": artifact.get("table_version"),
        "table_tag": artifact.get("table_tag"),
        "digest": artifact.get("digest"),
    }


# --- filter predicates ------------------------------------------------------


def _artifact_row_matches(row: Mapping[str, Any], filters: LineageExportFilters) -> bool:
    if filters.artifact_kind and str(row.get("kind") or "") != filters.artifact_kind:
        return False
    if not _created_in_export_range(row.get("created_at"), filters):
        return False
    return _table_version_matches(row, filters)


def _execution_row_matches(row: Mapping[str, Any], filters: LineageExportFilters) -> bool:
    if filters.execution_kind and str(row.get("kind") or "") != filters.execution_kind:
        return False
    return _created_in_export_range(row.get("created_at"), filters)


def _edge_row_matches(row: Mapping[str, Any], filters: LineageExportFilters) -> bool:
    return _created_in_export_range(row.get("created_at"), filters)


def _artifact_endpoint_matches(row: Mapping[str, Any], filters: LineageExportFilters) -> bool:
    if filters.artifact_kind and str(row.get("kind") or "") != filters.artifact_kind:
        return False
    return _table_version_matches(row, filters)


def _table_version_matches(row: Mapping[str, Any], filters: LineageExportFilters) -> bool:
    table_versions = filters.table_version_map()
    if not table_versions:
        return True
    table_name = row.get("table_name")
    table_version = row.get("table_version")
    if table_name is None or table_version is None:
        return False
    return table_versions.get(str(table_name)) == int(table_version)


def _created_in_export_range(value: Any, filters: LineageExportFilters) -> bool:
    if value is None:
        return True
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    if filters.created_after and value < filters.created_after:
        return False
    if filters.created_before and value > filters.created_before:
        return False
    return True


# --- token + filter normalization ------------------------------------------


def _export_query_digest(
    payload_kind: str,
    backend: str,
    filters: LineageExportFilters,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "payload_kind": payload_kind,
                "backend": backend,
                "filters": filters.digest_payload(),
            }
        ).encode()
    ).hexdigest()[:16]


def _encode_export_page_token(after_id: str, digest: str) -> str:
    raw = _canonical_json({"after": after_id, "q": digest}).encode()
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_export_page_token(token: str, digest: str) -> str:
    try:
        payload = json.loads(base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8"))
        after_id = str(payload["after"])
        token_digest = str(payload["q"])
    except (ValueError, KeyError, TypeError) as exc:
        raise LineageIntegrationError(f"invalid lineage export page token: {token!r}") from exc
    if token_digest != digest:
        raise LineageIntegrationError(
            "lineage export page token does not match this query; a continuation "
            "handle is only valid for the payload kind, backend, and filters that "
            "produced it"
        )
    return after_id


def _kind_sql_filter(kind: str | None) -> str | None:
    if not kind:
        return None
    return "kind = '" + str(kind).replace("'", "''") + "'"


def _clean_kind(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _coerce_export_datetime(value: datetime | str | None, label: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise LineageIntegrationError(f"{label} must be an ISO-8601 datetime") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _coerce_table_versions(
    value: Mapping[str, int] | Iterable[str | tuple[str, int]] | None,
) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(table): int(version) for table, version in value.items()}
    result: dict[str, int] = {}
    for item in value:
        if isinstance(item, str):
            if "=" in item:
                table, version = item.split("=", 1)
            elif ":" in item:
                table, version = item.split(":", 1)
            else:
                raise LineageIntegrationError(
                    "table version filters must look like table=version"
                )
            result[table.strip()] = _int_version(version)
            continue
        table, version = item
        result[str(table)] = int(version)
    return result


def _int_version(value: str) -> int:
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise LineageIntegrationError("table version must be an integer") from exc
