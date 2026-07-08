"""URI and object-store helpers shared by lakes and raw-source readers."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

OBJECT_STORE_SCHEMES = frozenset({"s3", "gs", "gcs", "az", "abfs", "abfss"})

_SCHEME_EXTRA = {
    "s3": "s3fs",
    "gs": "gcsfs",
    "gcs": "gcsfs",
    "az": "adlfs",
    "abfs": "adlfs",
    "abfss": "adlfs",
}

_GLOBAL_OPTIONS_ENV = "LANCEDB_ROBOTICS_STORAGE_OPTIONS_JSON"
_AUTH_OPTIONS_PREFIX = "LANCEDB_ROBOTICS_AUTH_"


class StorageConfigError(Exception):
    """Raised when URI storage options or object-store dependencies are unusable."""


def uri_scheme(uri: str | Path) -> str:
    """Return the lower-case URI scheme, or ``""`` for ordinary local paths."""
    return urlparse(str(uri)).scheme.lower()


def is_object_store_uri(uri: str | Path) -> bool:
    """True for object-store URIs supported by the robotics substrate."""
    return uri_scheme(uri) in OBJECT_STORE_SCHEMES


def source_uri(path: str | Path) -> str:
    """Canonical provenance URI for a raw source.

    Local files are stored as absolute paths for stable provenance. Object-store
    URIs are already absolute names in their storage namespace and are preserved
    byte-for-byte so ``raw_uri`` points back to the original object.
    """
    value = str(path)
    return value if is_object_store_uri(value) else str(Path(path).resolve())


def display_name(uri: str | Path) -> str:
    """Human display name for a local path or object-store URI."""
    value = str(uri)
    parsed = urlparse(value)
    name = Path(parsed.path.rstrip("/")).name if parsed.path else ""
    return name or parsed.netloc or value


def credential_resolution_order() -> tuple[str, ...]:
    """Document the credential lookup order for object-store access."""
    return (
        "explicit storage_options passed by the Python API or --storage-option",
        "auth_ref-scoped env JSON: LANCEDB_ROBOTICS_AUTH_<AUTH_REF>_STORAGE_OPTIONS_JSON",
        "scheme/global env JSON: LANCEDB_ROBOTICS_<SCHEME>_STORAGE_OPTIONS_JSON then "
        "LANCEDB_ROBOTICS_STORAGE_OPTIONS_JSON",
        "provider standard environment variables, config files, IAM/workload identity, "
        "or managed credential chain",
    )


def parse_storage_option_pairs(pairs: Sequence[str] | None) -> dict[str, str]:
    """Parse CLI ``key=value`` storage options."""
    options: dict[str, str] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"storage option {pair!r} must be key=value")
        key, value = pair.split("=", 1)
        if not key:
            raise ValueError(f"storage option {pair!r} has an empty key")
        options[key] = value
    return options


def resolve_storage_options(
    uri: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> dict[str, Any]:
    """Resolve non-persisted options for a lake or raw source URI.

    Secrets may appear in environment-backed options or explicit in-memory
    ``storage_options``. Callers must pass the returned mapping only to storage
    clients and must not persist it in lake tables.
    """
    scheme = uri_scheme(uri)
    options: dict[str, Any] = {}
    options.update(_load_options_env(_GLOBAL_OPTIONS_ENV))
    if scheme:
        options.update(_load_options_env(f"LANCEDB_ROBOTICS_{scheme.upper()}_STORAGE_OPTIONS_JSON"))
    if auth_ref:
        normalized = _normalize_auth_ref(auth_ref)
        options.update(_load_options_env(f"{_AUTH_OPTIONS_PREFIX}{normalized}_STORAGE_OPTIONS_JSON"))
    if scheme == "s3":
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            options.setdefault("region", region)
    if storage_options:
        options.update(dict(storage_options))
    return options


def lancedb_storage_options(
    uri: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> dict[str, str]:
    """Storage options normalized for ``lancedb.connect``."""
    resolved = resolve_storage_options(uri, storage_options=storage_options, auth_ref=auth_ref)
    return {str(key): _stringify_option(value) for key, value in resolved.items()}


def fsspec_storage_options(
    uri: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> dict[str, Any]:
    """Storage options normalized for fsspec-backed raw-source reads."""
    resolved = resolve_storage_options(uri, storage_options=storage_options, auth_ref=auth_ref)
    scheme = uri_scheme(uri)
    if scheme == "s3" and "region" in resolved:
        options = dict(resolved)
        region = str(options.pop("region"))
        client_kwargs = options.get("client_kwargs")
        if not isinstance(client_kwargs, dict):
            client_kwargs = {}
        client_kwargs.setdefault("region_name", region)
        options["client_kwargs"] = client_kwargs
        return options
    return resolved


@contextmanager
def open_binary_uri(
    uri: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> Iterator[BinaryIO]:
    """Open a local or object-store URI for binary reading.

    Object-store reads use fsspec so backends such as S3FS can perform ranged
    reads against seekable MCAP files instead of staging a full local copy.
    """
    value = str(uri)
    if not is_object_store_uri(value):
        path = Path(value)
        if not path.is_file():
            raise StorageConfigError(f"no such file: {path}")
        with path.open("rb") as stream:
            yield stream
        return

    options = fsspec_storage_options(value, storage_options=storage_options, auth_ref=auth_ref)
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - base dependency normally supplies this.
        raise StorageConfigError(
            "object-store reads require fsspec; install the object-store extra and retry"
        ) from exc

    try:
        with fsspec.open(value, mode="rb", **options) as stream:
            yield stream
    except ImportError as exc:
        raise _missing_dependency_error(value, exc) from exc
    except ModuleNotFoundError as exc:
        raise _missing_dependency_error(value, exc) from exc
    except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
        raise StorageConfigError(f"cannot read {value}: {exc}") from exc


def join_uri(base: str | Path, *parts: str) -> str:
    """Join path segments onto a local or object-store base URI.

    Local paths keep OS semantics; object-store URIs are joined with ``/`` so a
    single output-destination string works for both materialization targets.
    """
    cleaned = [str(part).strip("/") for part in parts if str(part).strip("/")]
    base_str = str(base)
    if is_object_store_uri(base_str):
        if not cleaned:
            return base_str
        return base_str.rstrip("/") + "/" + "/".join(cleaned)
    path = Path(base_str)
    for part in cleaned:
        path = path / part
    return str(path)


def uri_exists(
    uri: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> bool:
    """True if a local or object-store object exists at ``uri``."""
    value = str(uri)
    if not is_object_store_uri(value):
        return Path(value).exists()
    options = fsspec_storage_options(value, storage_options=storage_options, auth_ref=auth_ref)
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - base dependency normally supplies this.
        raise StorageConfigError(
            "object-store access requires fsspec; install the object-store extra and retry"
        ) from exc
    try:
        fs, path = fsspec.core.url_to_fs(value, **options)
        return bool(fs.exists(path))
    except (ImportError, ModuleNotFoundError) as exc:
        raise _missing_dependency_error(value, exc) from exc
    except (PermissionError, OSError, ValueError) as exc:
        raise StorageConfigError(f"cannot stat {value}: {exc}") from exc


def uri_info(
    uri: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> dict[str, Any]:
    """Return local or object-store metadata for one URI."""
    value = str(uri)
    if not is_object_store_uri(value):
        path = Path(value)
        try:
            stat = path.stat()
        except OSError as exc:
            raise StorageConfigError(f"cannot stat {path}: {exc}") from exc
        return {
            "name": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "type": "directory" if path.is_dir() else "file",
        }
    options = fsspec_storage_options(value, storage_options=storage_options, auth_ref=auth_ref)
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - base dependency normally supplies this.
        raise StorageConfigError(
            "object-store access requires fsspec; install the object-store extra and retry"
        ) from exc
    try:
        fs, path = fsspec.core.url_to_fs(value, **options)
        return dict(fs.info(path))
    except (ImportError, ModuleNotFoundError) as exc:
        raise _missing_dependency_error(value, exc) from exc
    except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
        raise StorageConfigError(f"cannot stat {value}: {exc}") from exc


def list_uri(
    uri: str | Path,
    *,
    pattern: str = "**/*",
    files_only: bool = True,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> list[str]:
    """List child URIs below a local or object-store root."""
    value = str(uri)
    if not is_object_store_uri(value):
        root = Path(value)
        matches = sorted(root.glob(pattern))
        if files_only:
            matches = [path for path in matches if path.is_file()]
        return [str(path) for path in matches]

    options = fsspec_storage_options(value, storage_options=storage_options, auth_ref=auth_ref)
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - base dependency normally supplies this.
        raise StorageConfigError(
            "object-store access requires fsspec; install the object-store extra and retry"
        ) from exc
    try:
        fs, base_path = fsspec.core.url_to_fs(value.rstrip("/"), **options)
        glob_path = "/".join(part.strip("/") for part in (base_path, pattern) if part.strip("/"))
        matches = sorted(str(path) for path in fs.glob(glob_path))
        uris: list[str] = []
        for path in matches:
            info = dict(fs.info(path))
            if files_only and str(info.get("type") or "").lower() == "directory":
                continue
            if is_object_store_uri(path):
                uris.append(path)
                continue
            rel = path[len(base_path) :].lstrip("/") if path.startswith(base_path) else path
            uris.append(join_uri(value, rel))
        return uris
    except (ImportError, ModuleNotFoundError) as exc:
        raise _missing_dependency_error(value, exc) from exc
    except (PermissionError, OSError, ValueError) as exc:
        raise StorageConfigError(f"cannot list {value}: {exc}") from exc


def read_text_uri(
    uri: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
    encoding: str = "utf-8",
) -> str:
    """Read a local or object-store object as text."""
    with open_binary_uri(uri, storage_options=storage_options, auth_ref=auth_ref) as stream:
        return stream.read().decode(encoding)


def write_binary_uri(
    uri: str | Path,
    data: bytes,
    *,
    storage_options: Mapping[str, Any] | None = None,
    auth_ref: str | None = None,
) -> None:
    """Write ``data`` to a local or object-store URI, creating parents as needed."""
    value = str(uri)
    if not is_object_store_uri(value):
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    options = fsspec_storage_options(value, storage_options=storage_options, auth_ref=auth_ref)
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - base dependency normally supplies this.
        raise StorageConfigError(
            "object-store writes require fsspec; install the object-store extra and retry"
        ) from exc
    try:
        with fsspec.open(value, mode="wb", auto_mkdir=True, **options) as stream:
            stream.write(data)
    except (ImportError, ModuleNotFoundError) as exc:
        raise _missing_dependency_error(value, exc) from exc
    except (PermissionError, OSError, ValueError) as exc:
        raise StorageConfigError(f"cannot write {value}: {exc}") from exc


def _load_options_env(name: str) -> dict[str, Any]:
    raw = os.environ.get(name)
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StorageConfigError(f"{name} must contain a JSON object: {exc}") from exc
    if not isinstance(decoded, dict):
        raise StorageConfigError(f"{name} must contain a JSON object")
    return {str(key): value for key, value in decoded.items()}


def _normalize_auth_ref(auth_ref: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", auth_ref.strip()).strip("_")
    return normalized.upper()


def _stringify_option(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _missing_dependency_error(uri: str, exc: ImportError) -> StorageConfigError:
    scheme = uri_scheme(uri)
    package = _SCHEME_EXTRA.get(scheme, "the matching object-store package")
    return StorageConfigError(
        f"cannot read {uri}: missing object-store dependency for '{scheme}://'; "
        f"install {package} or lancedb-robotics[object-store] and retry ({exc})"
    )
