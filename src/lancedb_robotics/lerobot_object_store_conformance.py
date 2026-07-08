"""LeRobot object-store provider conformance probes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from lancedb_robotics.adapters import AdapterError, get_adapter
from lancedb_robotics.storage import (
    OBJECT_STORE_SCHEMES,
    StorageConfigError,
    credential_resolution_order,
    fsspec_storage_options,
    is_object_store_uri,
    join_uri,
    list_uri,
    open_binary_uri,
    read_text_uri,
    resolve_storage_options,
    uri_info,
    uri_scheme,
)

LEROBOT_OBJECT_STORE_CONFORMANCE_SCHEMA = (
    "lancedb-robotics/lerobot-object-store-provider-conformance/v1"
)

_PROVIDER_SCHEMES = ("s3", "gs", "gcs", "az", "abfs", "abfss")
_SCHEME_EXTRA = {
    "s3": "s3fs",
    "gs": "gcsfs",
    "gcs": "gcsfs",
    "az": "adlfs",
    "abfs": "adlfs",
    "abfss": "adlfs",
}
_AUTH_MODES = ("explicit-options", "auth-ref-env", "provider-default")
_SENSITIVE_KEY_PARTS = ("secret", "token", "password", "credential", "access_key", "key")


@dataclass(frozen=True)
class LeRobotObjectStoreProbeOperation:
    """One storage or LeRobot adapter operation in a provider/auth case."""

    name: str
    status: str
    message: str
    uri: str | None = None
    error_class: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotObjectStoreConformanceCase:
    """Conformance result for one provider scheme and credential mode."""

    scheme: str
    provider: str
    auth_mode: str
    backend_extra: str
    root_uri: str | None
    auth_ref: str | None
    storage_option_keys: tuple[str, ...]
    fsspec_option_keys: tuple[str, ...]
    operations: tuple[LeRobotObjectStoreProbeOperation, ...]
    metadata_fields: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        statuses = [operation.status for operation in self.operations]
        if any(status == "failed" for status in statuses):
            return "failed"
        if any(status == "passed" for status in statuses):
            return "passed"
        return "skipped"

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status
        payload["passed"] = self.passed
        payload["metadata_fingerprints"] = {
            "etag": "etag" in self.metadata_fields,
            "version_id": "version_id" in self.metadata_fields,
            "generation": "generation" in self.metadata_fields,
            "size": "size" in self.metadata_fields,
            "last_modified": "last_modified" in self.metadata_fields,
        }
        payload["operations"] = [operation.to_params() for operation in self.operations]
        return payload


@dataclass(frozen=True)
class LeRobotObjectStoreConformanceReport:
    """Read-only LeRobot object-store provider conformance matrix."""

    schema_version: str
    roots: tuple[str, ...]
    schemes: tuple[str, ...]
    inspect_videos: bool
    credential_resolution_order: tuple[str, ...]
    cases: tuple[LeRobotObjectStoreConformanceCase, ...]
    credential_guidance: Mapping[str, str]

    @property
    def status_counts(self) -> dict[str, int]:
        counts = {"passed": 0, "failed": 0, "skipped": 0}
        for case in self.cases:
            counts[case.status] = counts.get(case.status, 0) + 1
        return counts

    @property
    def failed_count(self) -> int:
        return self.status_counts.get("failed", 0)

    @property
    def passed_count(self) -> int:
        return self.status_counts.get("passed", 0)

    @property
    def skipped_count(self) -> int:
        return self.status_counts.get("skipped", 0)

    @property
    def overall_status(self) -> str:
        if self.failed_count:
            return "failed"
        if self.passed_count:
            return "passed"
        return "skipped"

    @property
    def passed(self) -> bool:
        return self.failed_count == 0 and self.passed_count > 0

    def to_params(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "overall_status": self.overall_status,
            "passed": self.passed,
            "summary": {
                "passed": self.passed_count,
                "failed": self.failed_count,
                "skipped": self.skipped_count,
                "total": len(self.cases),
            },
            "roots": list(self.roots),
            "schemes": list(self.schemes),
            "inspect_videos": self.inspect_videos,
            "credential_resolution_order": list(self.credential_resolution_order),
            "credential_guidance": dict(self.credential_guidance),
            "cases": [case.to_params() for case in self.cases],
        }


def lerobot_object_store_conformance(
    *,
    roots: Sequence[str] | None = None,
    schemes: Sequence[str] | None = None,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
    include_provider_default: bool = True,
    inspect_videos: bool = False,
) -> LeRobotObjectStoreConformanceReport:
    """Probe LeRobot object-store roots without writing lake rows.

    The matrix exercises the storage primitives used by LeRobot source ingest
    (`uri_info`, `list_uri`, `read_text_uri`, `open_binary_uri`) and the
    LeRobot adapter's inspect/streaming preflight paths. It records credential
    option names but never resolved values.
    """

    normalized_roots = tuple(str(root).rstrip("/") for root in roots or ())
    for root in normalized_roots:
        if not is_object_store_uri(root):
            raise ValueError(f"root {root!r} is not a supported object-store URI")
    root_schemes = tuple(dict.fromkeys(uri_scheme(root) for root in normalized_roots))
    normalized_schemes = _normalize_schemes(schemes or root_schemes or _PROVIDER_SCHEMES)

    roots_by_scheme: dict[str, list[str]] = {scheme: [] for scheme in normalized_schemes}
    for root in normalized_roots:
        scheme = uri_scheme(root)
        roots_by_scheme.setdefault(scheme, []).append(root)

    cases: list[LeRobotObjectStoreConformanceCase] = []
    explicit_options = dict(storage_options) if storage_options is not None else None
    for scheme in normalized_schemes:
        scheme_roots = roots_by_scheme.get(scheme) or [None]
        for root in scheme_roots:
            for auth_mode in _AUTH_MODES:
                if auth_mode == "provider-default" and not include_provider_default:
                    continue
                cases.append(
                    _probe_case(
                        scheme=scheme,
                        root=root,
                        auth_mode=auth_mode,
                        storage_options=explicit_options,
                        auth_ref=auth_ref,
                        inspect_videos=inspect_videos,
                    )
                )

    return LeRobotObjectStoreConformanceReport(
        schema_version=LEROBOT_OBJECT_STORE_CONFORMANCE_SCHEMA,
        roots=normalized_roots,
        schemes=normalized_schemes,
        inspect_videos=inspect_videos,
        credential_resolution_order=credential_resolution_order(),
        cases=tuple(cases),
        credential_guidance=_credential_guidance(),
    )


def _probe_case(
    *,
    scheme: str,
    root: str | None,
    auth_mode: str,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
    inspect_videos: bool,
) -> LeRobotObjectStoreConformanceCase:
    backend_extra = _SCHEME_EXTRA[scheme]
    provider = _provider_name(scheme)
    active_storage_options: dict[str, Any] | None = None
    active_auth_ref: str | None = None
    skip_reason: str | None = None
    if root is None:
        skip_reason = f"no --root supplied for {scheme}://"
    elif auth_mode == "explicit-options":
        if storage_options is None:
            skip_reason = "no explicit storage_options supplied"
        else:
            active_storage_options = dict(storage_options)
    elif auth_mode == "auth-ref-env":
        if not auth_ref:
            skip_reason = "no auth_ref supplied"
        else:
            active_auth_ref = auth_ref
    elif auth_mode == "provider-default":
        active_storage_options = None
        active_auth_ref = None
    else:  # pragma: no cover - guarded by _AUTH_MODES.
        skip_reason = f"unknown auth mode {auth_mode!r}"

    if root is None or skip_reason is not None:
        return LeRobotObjectStoreConformanceCase(
            scheme=scheme,
            provider=provider,
            auth_mode=auth_mode,
            backend_extra=backend_extra,
            root_uri=root,
            auth_ref=active_auth_ref,
            storage_option_keys=tuple(sorted((storage_options or {}).keys()))
            if auth_mode == "explicit-options"
            else (),
            fsspec_option_keys=(),
            operations=(
                LeRobotObjectStoreProbeOperation(
                    name="case_setup",
                    status="skipped",
                    message=skip_reason or "case skipped",
                    uri=root,
                ),
            ),
            recommendations=(_install_recommendation(scheme),),
        )

    assert root is not None
    resolved_storage_options: dict[str, Any] = {}
    fsspec_options: dict[str, Any] = {}
    sensitive_values: tuple[str, ...] = ()
    operations: list[LeRobotObjectStoreProbeOperation] = []
    try:
        resolved_storage_options = resolve_storage_options(
            root,
            storage_options=active_storage_options,
            auth_ref=active_auth_ref,
        )
        fsspec_options = fsspec_storage_options(
            root,
            storage_options=active_storage_options,
            auth_ref=active_auth_ref,
        )
        sensitive_values = _sensitive_values(resolved_storage_options)
        operations.append(
            LeRobotObjectStoreProbeOperation(
                name="resolve_storage_options",
                status="passed",
                message="resolved storage option keys without exposing values",
                uri=root,
                details={
                    "storage_option_keys": tuple(sorted(resolved_storage_options.keys())),
                    "fsspec_option_keys": tuple(sorted(fsspec_options.keys())),
                    "auth_ref": active_auth_ref,
                },
            )
        )
    except StorageConfigError as exc:
        operations.append(_failed_operation("resolve_storage_options", root, exc, ()))

    info_uri = join_uri(root, "meta/info.json")
    operations.extend(
        (
            _run_operation(
                "uri_info_meta",
                info_uri,
                lambda: _uri_info_details(
                    info_uri,
                    storage_options=active_storage_options,
                    auth_ref=active_auth_ref,
                ),
                sensitive_values=sensitive_values,
            ),
            _run_operation(
                "list_meta",
                root,
                lambda: _list_details(
                    root,
                    storage_options=active_storage_options,
                    auth_ref=active_auth_ref,
                ),
                sensitive_values=sensitive_values,
            ),
            _run_operation(
                "read_info_json",
                info_uri,
                lambda: _read_info_details(
                    info_uri,
                    storage_options=active_storage_options,
                    auth_ref=active_auth_ref,
                ),
                sensitive_values=sensitive_values,
            ),
            _run_operation(
                "open_info_binary",
                info_uri,
                lambda: _open_binary_details(
                    info_uri,
                    storage_options=active_storage_options,
                    auth_ref=active_auth_ref,
                ),
                sensitive_values=sensitive_values,
            ),
            _run_operation(
                "lerobot_inspect",
                root,
                lambda: _inspect_details(
                    root,
                    storage_options=active_storage_options,
                    auth_ref=active_auth_ref,
                    inspect_videos=inspect_videos,
                ),
                sensitive_values=sensitive_values,
            ),
            _run_operation(
                "lerobot_ingest_stream_preflight",
                root,
                lambda: _ingest_preflight_details(
                    root,
                    storage_options=active_storage_options,
                    auth_ref=active_auth_ref,
                ),
                sensitive_values=sensitive_values,
            ),
        )
    )
    metadata_fields = _metadata_fields(operations)
    recommendations = _case_recommendations(scheme, operations, metadata_fields)
    return LeRobotObjectStoreConformanceCase(
        scheme=scheme,
        provider=provider,
        auth_mode=auth_mode,
        backend_extra=backend_extra,
        root_uri=root,
        auth_ref=active_auth_ref,
        storage_option_keys=tuple(sorted((active_storage_options or {}).keys())),
        fsspec_option_keys=tuple(sorted(fsspec_options.keys())),
        operations=tuple(operations),
        metadata_fields=metadata_fields,
        recommendations=recommendations,
    )


def _run_operation(
    name: str,
    uri: str,
    operation,
    *,
    sensitive_values: Sequence[str],
) -> LeRobotObjectStoreProbeOperation:
    try:
        details = operation()
    except (AdapterError, StorageConfigError) as exc:
        return _failed_operation(name, uri, exc, sensitive_values)
    except Exception as exc:  # noqa: BLE001 - matrix should report unexpected backend failures.
        return _failed_operation(name, uri, exc, sensitive_values)
    return LeRobotObjectStoreProbeOperation(
        name=name,
        status="passed",
        message="ok",
        uri=uri,
        details=details,
    )


def _failed_operation(
    name: str,
    uri: str | None,
    exc: BaseException,
    sensitive_values: Sequence[str],
) -> LeRobotObjectStoreProbeOperation:
    return LeRobotObjectStoreProbeOperation(
        name=name,
        status="failed",
        message=_redact(str(exc), sensitive_values),
        uri=uri,
        error_class=exc.__class__.__name__,
    )


def _uri_info_details(
    uri: str,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, Any]:
    info = uri_info(uri, storage_options=storage_options, auth_ref=auth_ref)
    return {
        "type": info.get("type"),
        "size": _metadata_size(info),
        "metadata_keys": _normalized_metadata_keys(info),
        "raw_metadata_keys": tuple(sorted(str(key) for key in info.keys())),
    }


def _list_details(
    root: str,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, Any]:
    matches = list_uri(root, pattern="meta/*.json", storage_options=storage_options, auth_ref=auth_ref)
    return {"match_count": len(matches), "sample": tuple(matches[:3])}


def _read_info_details(
    uri: str,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, Any]:
    text = read_text_uri(uri, storage_options=storage_options, auth_ref=auth_ref)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise StorageConfigError(f"{uri} did not contain a JSON object")
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    return {
        "codebase_version": payload.get("codebase_version"),
        "fps": payload.get("fps"),
        "feature_keys": tuple(sorted(str(key) for key in features.keys())),
    }


def _open_binary_details(
    uri: str,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, Any]:
    with open_binary_uri(uri, storage_options=storage_options, auth_ref=auth_ref) as stream:
        prefix = stream.read(64)
    return {"bytes_read": len(prefix)}


def _inspect_details(
    root: str,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
    inspect_videos: bool,
) -> dict[str, Any]:
    report = get_adapter("lerobot").inspect(
        root,
        inspect_videos=inspect_videos,
        storage_options=dict(storage_options or {}),
        auth_ref=auth_ref,
    )
    video_metadata_fields = sorted(
        {
            key
            for video in report.get("video_files") or ()
            for key in (video.get("object_metadata") or {}).keys()
        }
    )
    return {
        "source_identity_kind": (report.get("source_identity") or {}).get("kind"),
        "frame_count": report.get("frame_count"),
        "episode_count": report.get("episode_count"),
        "data_file_count": len(report.get("data_files") or ()),
        "video_count": len(report.get("video_files") or ()),
        "video_metadata_fields": tuple(video_metadata_fields),
    }


def _ingest_preflight_details(
    root: str,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, Any]:
    adapter = get_adapter("lerobot")
    batch = next(
        adapter.iter_frame_batches(
            root,
            batch_size=1,
            storage_options=dict(storage_options or {}),
            auth_ref=auth_ref,
        )
    )
    return {
        "data_file": batch.data_file,
        "row_group": batch.row_group,
        "row_count": batch.row_count,
        "bytes_scanned": batch.bytes_scanned,
    }


def _metadata_fields(operations: Sequence[LeRobotObjectStoreProbeOperation]) -> tuple[str, ...]:
    fields: set[str] = set()
    for operation in operations:
        if operation.status != "passed":
            continue
        details = dict(operation.details or {})
        fields.update(str(key) for key in details.get("metadata_keys") or ())
        fields.update(str(key) for key in details.get("video_metadata_fields") or ())
    return tuple(sorted(fields))


def _case_recommendations(
    scheme: str,
    operations: Sequence[LeRobotObjectStoreProbeOperation],
    metadata_fields: Sequence[str],
) -> tuple[str, ...]:
    recommendations: list[str] = []
    failed = [operation for operation in operations if operation.status == "failed"]
    if failed:
        recommendations.append(_install_recommendation(scheme))
    if any("access denied" in operation.message.lower() for operation in failed):
        recommendations.append(
            "Check that the raw-source auth_ref or storage options can list, stat, and read "
            "the LeRobot prefix."
        )
    if not {"etag", "version_id", "generation"}.intersection(metadata_fields):
        recommendations.append(
            "Provider metadata did not expose etag/version/generation; media fingerprints may "
            "fall back to size/mtime where available."
        )
    return tuple(dict.fromkeys(recommendations))


def _normalize_schemes(schemes: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for scheme in schemes:
        value = str(scheme).lower().strip().rstrip(":/")
        if value not in OBJECT_STORE_SCHEMES:
            raise ValueError(f"unsupported object-store scheme {scheme!r}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _provider_name(scheme: str) -> str:
    if scheme == "s3":
        return "S3-compatible"
    if scheme in {"gs", "gcs"}:
        return "Google Cloud Storage"
    return "Azure Blob/Data Lake"


def _credential_guidance() -> dict[str, str]:
    return {
        "s3": "Install s3fs. Use AWS env/config/IAM, auth_ref env JSON, or explicit "
        "storage options such as region, endpoint_url, profile, key, secret, or token.",
        "gs": "Install gcsfs. Use Application Default Credentials, workload identity, "
        "auth_ref env JSON, or explicit options such as token and project.",
        "gcs": "Same as gs://; install gcsfs and use ADC/workload identity or explicit "
        "token/project options.",
        "az": "Install adlfs. Use Azure env/managed identity, auth_ref env JSON, or "
        "explicit account_name, credential, connection_string, tenant_id/client_id options.",
        "abfs": "Same Azure backend as az://; install adlfs and provide account/credential "
        "options or managed identity.",
        "abfss": "Same Azure backend as abfs:// with TLS scheme semantics; install adlfs "
        "and provide account/credential options or managed identity.",
    }


def _install_recommendation(scheme: str) -> str:
    return (
        f"Install {_SCHEME_EXTRA[scheme]} or lancedb-robotics[object-store] and verify "
        f"{scheme}:// credentials."
    )


def _normalized_metadata_keys(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    normalized = set()
    aliases = {
        "ETag": "etag",
        "VersionId": "version_id",
        "Generation": "generation",
        "Size": "size",
        "ContentLength": "size",
        "content_length": "size",
        "LastModified": "last_modified",
        "updated": "last_modified",
        "mtime_ns": "last_modified",
        "mtime": "last_modified",
    }
    for key, value in metadata.items():
        if value is None:
            continue
        normalized.add(aliases.get(str(key), str(key)))
    return tuple(sorted(normalized))


def _metadata_size(metadata: Mapping[str, Any]) -> int | None:
    for key in ("size", "Size", "ContentLength", "content_length"):
        if metadata.get(key) is not None:
            return int(metadata[key])
    return None


def _sensitive_values(options: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key, value in _flatten_options(options):
        if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
            text = str(value)
            if text:
                values.append(text)
    return tuple(sorted(set(values), key=len, reverse=True))


def _flatten_options(options: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in options.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            rows.extend(_flatten_options(value, name))
        else:
            rows.append((name, value))
    return rows


def _redact(message: str, sensitive_values: Sequence[str]) -> str:
    redacted = message
    for value in sensitive_values:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted
