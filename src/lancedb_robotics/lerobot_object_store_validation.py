"""LeRobot object-store source validation policies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lancedb_robotics.storage import open_binary_uri

LEROBOT_OBJECT_STORE_VALIDATION_SCHEMA = (
    "lancedb-robotics/lerobot-object-store-validation/v1"
)
LEROBOT_OBJECT_STORE_VALIDATION_POLICIES = (
    "metadata-only",
    "sampled-validation",
    "strict-content-hash",
)
DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY = "metadata-only"
DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT = 8
DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES = 4096
DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES = 64 * 1024 * 1024

_STRONG_METADATA_KEYS = (
    "etag",
    "ETag",
    "version_id",
    "VersionId",
    "generation",
    "Generation",
)


@dataclass(frozen=True)
class LeRobotObjectStoreValidationObject:
    """One object considered by LeRobot object-store source validation."""

    uri: str
    relative_path: str
    info: Mapping[str, Any]
    metadata_fingerprint: str


@dataclass(frozen=True)
class LeRobotObjectStoreValidationConfig:
    """User-facing validation policy knobs for object-store LeRobot roots."""

    policy: str = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY
    sample_count: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT
    sample_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES
    strict_max_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES


def resolve_lerobot_object_store_validation_config(
    policy: str | None = None,
    *,
    sample_count: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
    sample_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
    strict_max_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
) -> LeRobotObjectStoreValidationConfig:
    """Normalize user-facing object-store validation options."""

    normalized = (policy or DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY).strip().lower()
    if normalized not in LEROBOT_OBJECT_STORE_VALIDATION_POLICIES:
        expected = ", ".join(LEROBOT_OBJECT_STORE_VALIDATION_POLICIES)
        raise ValueError(
            f"unknown LeRobot object-store validation policy {policy!r}; expected {expected}"
        )
    if sample_count < 0:
        raise ValueError(f"object_store_validation_sample_count must be >= 0, got {sample_count}")
    if sample_bytes < 1:
        raise ValueError(f"object_store_validation_sample_bytes must be >= 1, got {sample_bytes}")
    if strict_max_bytes < 0:
        raise ValueError(
            f"object_store_validation_strict_max_bytes must be >= 0, got {strict_max_bytes}"
        )
    return LeRobotObjectStoreValidationConfig(
        policy=normalized,
        sample_count=int(sample_count),
        sample_bytes=int(sample_bytes),
        strict_max_bytes=int(strict_max_bytes),
    )


def object_metadata_fingerprint(info: Mapping[str, Any]) -> str:
    """Return the same kind of stable, secret-free metadata digest used for manifests."""

    keys = (
        "etag",
        "ETag",
        "version_id",
        "VersionId",
        "generation",
        "Generation",
        "size",
        "Size",
        "ContentLength",
        "content_length",
        "last_modified",
        "LastModified",
        "mtime",
        "mtime_ns",
    )
    payload = {key: _json_ready(info[key]) for key in keys if info.get(key) is not None}
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_lerobot_object_store_source(
    root_uri: str,
    *,
    objects: Sequence[LeRobotObjectStoreValidationObject],
    config: LeRobotObjectStoreValidationConfig,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> dict[str, Any]:
    """Return source-validation evidence for an object-store LeRobot root."""

    ordered = tuple(sorted(objects, key=lambda item: item.relative_path))
    metadata = _metadata_quality(ordered)
    samples: list[dict[str, Any]] = []
    hashed_bytes = 0
    if config.policy == "sampled-validation":
        selected = _select_sample_objects(root_uri, ordered, limit=config.sample_count)
        for item in selected:
            sample = _read_prefix_sample(
                item,
                byte_count=config.sample_bytes,
                storage_options=storage_options,
                auth_ref=auth_ref,
            )
            samples.append(sample)
            hashed_bytes += int(sample["length"])
    elif config.policy == "strict-content-hash":
        total_size = _known_total_size(ordered)
        if total_size is not None and total_size > config.strict_max_bytes:
            raise ValueError(
                "strict-content-hash would read "
                f"{total_size} bytes from {root_uri}, exceeding "
                f"object_store_validation_strict_max_bytes={config.strict_max_bytes}"
            )
        for item in ordered:
            sample = _read_full_object_hash(
                item,
                strict_max_bytes=config.strict_max_bytes,
                bytes_seen=hashed_bytes,
                storage_options=storage_options,
                auth_ref=auth_ref,
            )
            samples.append(sample)
            hashed_bytes += int(sample["length"])

    evidence_digest = _evidence_digest(config.policy, samples)
    warnings = _validation_warnings(
        metadata,
        policy=config.policy,
        object_count=len(ordered),
        sampled_count=len(samples),
    )
    return {
        "schema_version": LEROBOT_OBJECT_STORE_VALIDATION_SCHEMA,
        "root_uri": root_uri.rstrip("/"),
        "policy": config.policy,
        "status": "passed",
        "assurance": _assurance(config.policy),
        "object_count": len(ordered),
        "sample_count": len(samples),
        "sample_count_configured": config.sample_count,
        "sample_bytes": config.sample_bytes,
        "strict_max_bytes": config.strict_max_bytes,
        "hashed_object_count": len(samples),
        "hashed_bytes": hashed_bytes,
        "metadata": metadata,
        "samples": samples,
        "evidence_digest": evidence_digest,
        "warnings": warnings,
    }


def validation_checksum_material(report: Mapping[str, Any] | None) -> str | None:
    """Return checksum material for policies that add content evidence."""

    if not report:
        return None
    policy = str(report.get("policy") or "")
    if policy == "metadata-only":
        return None
    payload = {
        "schema_version": report.get("schema_version"),
        "policy": policy,
        "evidence_digest": report.get("evidence_digest"),
        "sample_count": report.get("sample_count"),
        "hashed_bytes": report.get("hashed_bytes"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _select_sample_objects(
    root_uri: str,
    objects: Sequence[LeRobotObjectStoreValidationObject],
    *,
    limit: int,
) -> tuple[LeRobotObjectStoreValidationObject, ...]:
    if limit < 1:
        return ()
    ranked = sorted(
        objects,
        key=lambda item: (
            hashlib.sha256(f"{root_uri.rstrip('/')}\0{item.relative_path}".encode()).hexdigest(),
            item.relative_path,
        ),
    )
    return tuple(ranked[: min(limit, len(ranked))])


def _read_prefix_sample(
    item: LeRobotObjectStoreValidationObject,
    *,
    byte_count: int,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, Any]:
    with open_binary_uri(item.uri, storage_options=storage_options, auth_ref=auth_ref) as stream:
        payload = stream.read(byte_count)
    return {
        "relative_path": item.relative_path,
        "uri": item.uri,
        "mode": "prefix",
        "offset": 0,
        "length": len(payload),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "metadata_fingerprint": item.metadata_fingerprint,
    }


def _read_full_object_hash(
    item: LeRobotObjectStoreValidationObject,
    *,
    strict_max_bytes: int,
    bytes_seen: int,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    with open_binary_uri(item.uri, storage_options=storage_options, auth_ref=auth_ref) as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if bytes_seen + total > strict_max_bytes:
                raise ValueError(
                    "strict-content-hash exceeded "
                    f"object_store_validation_strict_max_bytes={strict_max_bytes}"
                )
            digest.update(chunk)
    return {
        "relative_path": item.relative_path,
        "uri": item.uri,
        "mode": "full-object",
        "offset": 0,
        "length": total,
        "sha256": "sha256:" + digest.hexdigest(),
        "metadata_fingerprint": item.metadata_fingerprint,
    }


def _metadata_quality(
    objects: Sequence[LeRobotObjectStoreValidationObject],
) -> dict[str, Any]:
    strong_keys: set[str] = set()
    weak_examples: list[str] = []
    strong_count = 0
    for item in objects:
        present = {key for key in _STRONG_METADATA_KEYS if item.info.get(key) is not None}
        strong_keys.update(_normalize_metadata_key(key) for key in present)
        if present:
            strong_count += 1
        elif len(weak_examples) < 5:
            weak_examples.append(item.relative_path)
    weak_count = max(0, len(objects) - strong_count)
    return {
        "strong_metadata_keys": sorted(strong_keys),
        "objects_with_strong_metadata": strong_count,
        "objects_without_strong_metadata": weak_count,
        "weak_object_examples": weak_examples,
    }


def _validation_warnings(
    metadata: Mapping[str, Any],
    *,
    policy: str,
    object_count: int,
    sampled_count: int,
) -> list[dict[str, Any]]:
    weak_count = int(metadata.get("objects_without_strong_metadata") or 0)
    if weak_count == 0:
        return []
    examples = list(metadata.get("weak_object_examples") or ())
    if policy == "strict-content-hash":
        return []
    if policy == "sampled-validation" and sampled_count >= object_count:
        return []
    if policy == "sampled-validation":
        message = (
            "provider metadata lacks etag/version/generation for some unsampled objects; "
            "strict-content-hash is required for full content assurance"
        )
    else:
        message = (
            "provider metadata lacks etag/version/generation for some objects; "
            "metadata-only cannot provide content-level assurance"
        )
    return [
        {
            "code": "weak-provider-metadata",
            "severity": "warning",
            "message": message,
            "object_count": weak_count,
            "examples": examples,
        }
    ]


def _known_total_size(objects: Sequence[LeRobotObjectStoreValidationObject]) -> int | None:
    total = 0
    for item in objects:
        size = _metadata_size(item.info)
        if size is None:
            return None
        total += size
    return total


def _metadata_size(info: Mapping[str, Any]) -> int | None:
    for key in ("size", "Size", "ContentLength", "content_length"):
        if info.get(key) is not None:
            return int(info[key])
    return None


def _evidence_digest(policy: str, samples: Sequence[Mapping[str, Any]]) -> str | None:
    if policy == "metadata-only":
        return None
    payload = [
        {
            "relative_path": sample.get("relative_path"),
            "mode": sample.get("mode"),
            "offset": sample.get("offset"),
            "length": sample.get("length"),
            "sha256": sample.get("sha256"),
            "metadata_fingerprint": sample.get("metadata_fingerprint"),
        }
        for sample in samples
    ]
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _assurance(policy: str) -> str:
    return {
        "metadata-only": "provider-metadata",
        "sampled-validation": "sampled-content",
        "strict-content-hash": "full-content",
    }[policy]


def _normalize_metadata_key(key: str) -> str:
    return {
        "ETag": "etag",
        "VersionId": "version_id",
        "Generation": "generation",
    }.get(key, key)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return [_json_ready(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
