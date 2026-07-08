"""LeRobot object-store manifest listing and cache helpers."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from lancedb_robotics.storage import (
    StorageConfigError,
    fsspec_storage_options,
    is_object_store_uri,
    join_uri,
    resolve_storage_options,
    uri_info,
    uri_scheme,
)

LEROBOT_OBJECT_STORE_MANIFEST_SCHEMA = "lancedb-robotics/lerobot-object-store-manifest/v1"


@dataclass(frozen=True)
class LeRobotObjectStoreManifestEntry:
    """One object in a LeRobot object-store source manifest."""

    uri: str
    relative_path: str
    info: Mapping[str, Any]
    fingerprint: str

    def to_params(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "relative_path": self.relative_path,
            "info": _json_ready(dict(self.info)),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class LeRobotObjectStoreManifest:
    """A listed LeRobot object-store root plus planner/cache metrics."""

    schema_version: str
    root_uri: str
    cache_key: str
    auth_ref: str | None
    storage_option_keys: tuple[str, ...]
    storage_options_digest: str
    entries: tuple[LeRobotObjectStoreManifestEntry, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def object_count(self) -> int:
        return len(self.entries)

    def matching_paths(self, pattern: str) -> list[str]:
        return [
            entry.uri
            for entry in self.entries
            if _matches_pattern(entry.relative_path, pattern)
        ]

    def info_for(self, uri_or_path: str) -> dict[str, Any] | None:
        relative = _relative_path(uri_or_path, self.root_uri, self.root_uri.split("://", 1)[-1])
        for entry in self.entries:
            if entry.uri == uri_or_path or entry.relative_path == relative:
                return dict(entry.info)
        return None

    def with_metrics(self, metrics: Mapping[str, Any]) -> LeRobotObjectStoreManifest:
        return replace(self, metrics=dict(metrics))

    def to_params(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "root_uri": self.root_uri,
            "cache_key": self.cache_key,
            "auth_ref": self.auth_ref,
            "storage_option_keys": list(self.storage_option_keys),
            "storage_options_digest": self.storage_options_digest,
            "entries": [entry.to_params() for entry in self.entries],
            "metrics": _json_ready(dict(self.metrics)),
            "object_count": self.object_count,
        }

    @classmethod
    def from_params(cls, payload: Mapping[str, Any]) -> LeRobotObjectStoreManifest:
        return cls(
            schema_version=str(
                payload.get("schema_version") or LEROBOT_OBJECT_STORE_MANIFEST_SCHEMA
            ),
            root_uri=str(payload["root_uri"]),
            cache_key=str(payload["cache_key"]),
            auth_ref=payload.get("auth_ref"),
            storage_option_keys=tuple(str(key) for key in payload.get("storage_option_keys") or ()),
            storage_options_digest=str(payload.get("storage_options_digest") or ""),
            entries=tuple(
                LeRobotObjectStoreManifestEntry(
                    uri=str(entry["uri"]),
                    relative_path=str(entry["relative_path"]),
                    info=dict(entry.get("info") or {}),
                    fingerprint=str(entry["fingerprint"]),
                )
                for entry in payload.get("entries") or ()
            ),
            metrics=dict(payload.get("metrics") or {}),
        )


class LeRobotObjectStoreManifestCache:
    """Optional in-memory or JSON-backed LeRobot object-store manifest cache."""

    def __init__(self, path: str | Path | None = None, *, validate: bool = True) -> None:
        self.path = Path(path) if path is not None else None
        self.validate = validate
        self._manifests: dict[str, LeRobotObjectStoreManifest] = {}
        self._last: dict[str, LeRobotObjectStoreManifest] = {}
        self._fresh_keys: set[str] = set()
        if self.path is not None and self.path.exists():
            self._load()

    def manifest(
        self,
        root_uri: str,
        *,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> LeRobotObjectStoreManifest:
        key, option_keys, options_digest = manifest_cache_identity(
            root_uri,
            storage_options=storage_options,
            auth_ref=auth_ref,
        )
        cached = self._manifests.get(key)
        if cached is not None:
            valid, validation = _validate_manifest(
                cached,
                storage_options=storage_options,
                auth_ref=auth_ref,
            ) if self.validate else (True, {"validation_info_calls": 0})
            if valid:
                if key in self._fresh_keys:
                    metrics = {**cached.metrics, **validation}
                else:
                    metrics = {
                        **cached.metrics,
                        **validation,
                        "cache_status": "hit",
                        "cache_hit": True,
                        "cache_invalidated": False,
                    }
                report = cached.with_metrics(metrics)
                self._last[key] = report
                return report

        rebuilt = build_lerobot_object_store_manifest(
            root_uri,
            storage_options=storage_options,
            auth_ref=auth_ref,
            cache_key=key,
            storage_option_keys=option_keys,
            storage_options_digest=options_digest,
            cache_status="invalidated" if cached is not None else "miss",
        )
        self._manifests[key] = rebuilt
        self._fresh_keys.add(key)
        self._last[key] = rebuilt
        self._save()
        return rebuilt

    def list_paths(
        self,
        root_uri: str,
        *,
        pattern: str,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> list[str]:
        return self.manifest(
            root_uri,
            storage_options=storage_options,
            auth_ref=auth_ref,
        ).matching_paths(pattern)

    def cached_info(
        self,
        root_uri: str,
        uri_or_path: str,
        *,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> dict[str, Any] | None:
        key, _, _ = manifest_cache_identity(
            root_uri,
            storage_options=storage_options,
            auth_ref=auth_ref,
        )
        manifest = self._last.get(key) or self._manifests.get(key)
        if manifest is None:
            return None
        return manifest.info_for(uri_or_path)

    def last_report(
        self,
        root_uri: str,
        *,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> dict[str, Any] | None:
        key, _, _ = manifest_cache_identity(
            root_uri,
            storage_options=storage_options,
            auth_ref=auth_ref,
        )
        manifest = self._last.get(key) or self._manifests.get(key)
        if manifest is None:
            return None
        return {
            "schema_version": manifest.schema_version,
            "root_uri": manifest.root_uri,
            "cache_key": manifest.cache_key,
            "auth_ref": manifest.auth_ref,
            "storage_option_keys": list(manifest.storage_option_keys),
            "storage_options_digest": manifest.storage_options_digest,
            "object_count": manifest.object_count,
            "metrics": dict(manifest.metrics),
        }

    def _load(self) -> None:
        if self.path is None:
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != LEROBOT_OBJECT_STORE_MANIFEST_SCHEMA:
            return
        for item in payload.get("manifests") or ():
            manifest = LeRobotObjectStoreManifest.from_params(item)
            self._manifests[manifest.cache_key] = manifest

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": LEROBOT_OBJECT_STORE_MANIFEST_SCHEMA,
            "manifests": [manifest.to_params() for manifest in self._manifests.values()],
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_lerobot_object_store_manifest_cache(
    cache: LeRobotObjectStoreManifestCache | str | Path | None,
) -> LeRobotObjectStoreManifestCache | None:
    """Normalize user-facing cache inputs to a manifest cache object."""
    if cache is None:
        return None
    if isinstance(cache, LeRobotObjectStoreManifestCache):
        return cache
    return LeRobotObjectStoreManifestCache(cache)


def manifest_cache_identity(
    root_uri: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> tuple[str, tuple[str, ...], str]:
    """Return a stable, secret-free identity for one root/auth/options tuple."""
    resolved = resolve_storage_options(
        root_uri,
        storage_options=storage_options,
        auth_ref=auth_ref,
    )
    canonical = json.dumps(_json_ready(resolved), sort_keys=True, separators=(",", ":"))
    options_digest = hashlib.sha256(canonical.encode()).hexdigest()
    key_payload = {
        "root_uri": root_uri.rstrip("/"),
        "scheme": uri_scheme(root_uri),
        "auth_ref": auth_ref,
        "storage_options_digest": options_digest,
    }
    cache_key = "sha256:" + hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return cache_key, tuple(sorted(str(key) for key in resolved.keys())), options_digest


def build_lerobot_object_store_manifest(
    root_uri: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
    cache_key: str | None = None,
    storage_option_keys: Sequence[str] | None = None,
    storage_options_digest: str | None = None,
    cache_status: str = "miss",
) -> LeRobotObjectStoreManifest:
    """List one object-store LeRobot root into a reusable manifest."""
    if not is_object_store_uri(root_uri):
        raise StorageConfigError(f"LeRobot manifest cache requires an object-store URI: {root_uri}")
    key, option_keys, options_digest = (
        manifest_cache_identity(root_uri, storage_options=storage_options, auth_ref=auth_ref)
        if cache_key is None
        else (
            cache_key,
            tuple(storage_option_keys or ()),
            storage_options_digest or "",
        )
    )
    start = time.perf_counter()
    options = fsspec_storage_options(root_uri, storage_options=storage_options, auth_ref=auth_ref)
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - base dependency normally supplies this.
        raise StorageConfigError(
            "object-store access requires fsspec; install the object-store extra and retry"
        ) from exc
    try:
        fs, base_path = fsspec.core.url_to_fs(root_uri.rstrip("/"), **options)
        raw_entries, list_metrics = _listed_entries(fs, base_path)
    except (ImportError, ModuleNotFoundError) as exc:
        raise StorageConfigError(
            f"cannot list {root_uri}: missing object-store dependency for "
            f"'{uri_scheme(root_uri)}://'; install the matching object-store package "
            f"or lancedb-robotics[object-store] and retry ({exc})"
        ) from exc
    except (PermissionError, OSError, ValueError) as exc:
        raise StorageConfigError(f"cannot list {root_uri}: {exc}") from exc

    entries = tuple(
        sorted(
            (
                _manifest_entry(root_uri, base_path, path, info)
                for path, info in raw_entries
                if str((info or {}).get("type") or "file").lower() != "directory"
            ),
            key=lambda entry: entry.relative_path,
        )
    )
    metrics = {
        **list_metrics,
        "cache_status": cache_status,
        "cache_hit": False,
        "cache_invalidated": cache_status == "invalidated",
        "listed_object_count": len(entries),
        "metadata_bytes": sum(len(_canonical_info(entry.info)) for entry in entries),
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
    }
    return LeRobotObjectStoreManifest(
        schema_version=LEROBOT_OBJECT_STORE_MANIFEST_SCHEMA,
        root_uri=root_uri.rstrip("/"),
        cache_key=key,
        auth_ref=auth_ref,
        storage_option_keys=tuple(option_keys),
        storage_options_digest=options_digest,
        entries=entries,
        metrics=metrics,
    )


def _listed_entries(fs, base_path: str) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    if hasattr(fs, "find_pages"):
        entries: list[tuple[str, dict[str, Any]]] = []
        page_count = 0
        for page in fs.find_pages(base_path, detail=True):
            page_count += 1
            entries.extend(_page_entries(page, fs))
        return entries, {"list_strategy": "find_pages", "list_calls": page_count}
    if hasattr(fs, "find"):
        before = int(getattr(fs, "list_call_count", 0) or 0)
        found = fs.find(base_path, detail=True)
        after = int(getattr(fs, "list_call_count", before) or before)
        list_calls = max(1, after - before)
        return _page_entries(found, fs), {"list_strategy": "find", "list_calls": list_calls}
    pattern = "/".join(part.strip("/") for part in (base_path, "**/*") if part.strip("/"))
    matches = sorted(str(path) for path in fs.glob(pattern))
    return (
        [(path, dict(fs.info(path))) for path in matches],
        {"list_strategy": "glob", "list_calls": 1},
    )


def _page_entries(page: Any, fs) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(page, Mapping):
        return [(str(path), dict(info or {})) for path, info in page.items()]
    entries: list[tuple[str, dict[str, Any]]] = []
    for item in page or ():
        if isinstance(item, Mapping):
            path = str(item.get("name") or item.get("path") or "")
            entries.append((path, dict(item)))
        else:
            path = str(item)
            entries.append((path, dict(fs.info(path))))
    return entries


def _manifest_entry(
    root_uri: str,
    base_path: str,
    path: str,
    info: Mapping[str, Any],
) -> LeRobotObjectStoreManifestEntry:
    relative = _relative_path(path, root_uri, base_path)
    uri = path if is_object_store_uri(path) else join_uri(root_uri, relative)
    normalized_info = dict(info)
    normalized_info["name"] = uri
    return LeRobotObjectStoreManifestEntry(
        uri=uri,
        relative_path=relative,
        info=normalized_info,
        fingerprint=_metadata_fingerprint(normalized_info),
    )


def _validate_manifest(
    manifest: LeRobotObjectStoreManifest,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> tuple[bool, dict[str, Any]]:
    info_calls = 0
    for entry in manifest.entries:
        info_calls += 1
        try:
            current = uri_info(entry.uri, storage_options=storage_options, auth_ref=auth_ref)
        except StorageConfigError:
            return False, {"validation_info_calls": info_calls}
        if _metadata_fingerprint(current) != entry.fingerprint:
            return False, {"validation_info_calls": info_calls}
    return True, {"validation_info_calls": info_calls}


def _relative_path(path: str, root_uri: str, base_path: str) -> str:
    text = str(path)
    root = root_uri.rstrip("/")
    if is_object_store_uri(text):
        prefix = root + "/"
        if text == root:
            return ""
        if text.startswith(prefix):
            return text[len(prefix) :].lstrip("/")
        text = text.split("://", 1)[1]
    base = base_path.rstrip("/")
    if text == base:
        return ""
    if text.startswith(base + "/"):
        return text[len(base) + 1 :].lstrip("/")
    return text.lstrip("/")


def _metadata_fingerprint(info: Mapping[str, Any]) -> str:
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


def _matches_pattern(relative_path: str, pattern: str) -> bool:
    if fnmatch.fnmatchcase(relative_path, pattern):
        return True
    if "**/" in pattern:
        return fnmatch.fnmatchcase(relative_path, pattern.replace("**/", ""))
    return False


def _canonical_info(info: Mapping[str, Any]) -> str:
    return json.dumps(_json_ready(dict(info)), sort_keys=True, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return [_json_ready(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
