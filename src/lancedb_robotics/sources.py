"""Source registration: idempotent provenance anchors for raw robot logs.

Every ingested row traces back to an ``integration_sources`` row. The
registration key is the file *content* (its checksum), not the path: the same
bytes always register under the same ``source_id`` no matter which directory or
machine they live on, so ids — and everything derived from them — are portable
and reproducible. Re-registering the same bytes is a no-op that returns the
existing source; changed content is a new source. The absolute path is kept on
the row as ``uri`` provenance, just out of the id key.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import INTEGRATION_SOURCES_SCHEMA
from lancedb_robotics.storage import (
    StorageConfigError,
    display_name,
    open_binary_uri,
    source_uri,
)

if TYPE_CHECKING:
    from lancedb_robotics.recordings import Recording

_CHECKSUM_CHUNK_BYTES = 1 << 20


def file_checksum(
    path: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
    auth_ref: str | None = None,
) -> str:
    """SHA-256 hex digest of the file contents, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    try:
        with open_binary_uri(path, storage_options=storage_options, auth_ref=auth_ref) as stream:
            while chunk := stream.read(_CHECKSUM_CHUNK_BYTES):
                digest.update(chunk)
    except StorageConfigError as exc:
        from lancedb_robotics.adapters import AdapterError

        raise AdapterError(str(exc)) from exc
    return digest.hexdigest()


def content_key_digest(checksum: str, *, logical_name: str | None = None) -> str:
    """Path-independent digest derived from file content.

    Keyed on the content ``checksum`` alone, so the same bytes yield the same
    digest regardless of where they are read from. An optional stable
    ``logical_name`` can namespace otherwise-identical bytes into distinct ids;
    it must itself be path-independent (e.g. a dataset name), never an absolute
    path.
    """
    material = checksum if logical_name is None else f"{logical_name}\n{checksum}"
    return hashlib.sha256(material.encode()).hexdigest()


def recording_content_key(checksums: Sequence[str]) -> str:
    """Path-independent digest over an *ordered* list of shard checksums.

    A split recording is identified by the content of its shards in canonical
    order (backlog 0019), so the same shards from any directory or machine — and
    in any directory-listing order — curate to the same ``run_id``/``source_id``.
    For a single shard this equals :func:`content_key_digest`, so a one-shard
    recording and the same lone file share their ids.
    """
    return hashlib.sha256("\n".join(checksums).encode()).hexdigest()


@dataclass(frozen=True)
class SourceRegistration:
    """Outcome of :func:`register_source`; ``created`` is False on a no-op."""

    source_id: str
    uri: str
    checksum: str
    created: bool
    auth_ref: str | None = None


def register_source(
    lake: Lake,
    path: str | Path,
    *,
    adapter: str,
    inspect_report: dict | None = None,
    auth_ref: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> SourceRegistration:
    """Register ``path`` as a raw source, idempotently keyed by (URI, checksum).

    Records the adapter and per-topic schema fingerprints in metadata so a
    source row explains how it can be decoded. ``inspect_report`` (the
    adapter's inspect payload) supplies the fingerprints; when omitted, the
    adapter is asked to inspect the file.
    """
    uri = source_uri(path)
    checksum = file_checksum(uri, storage_options=storage_options, auth_ref=auth_ref)
    source_id = f"src-{content_key_digest(checksum)[:16]}"

    table = lake.table("integration_sources")
    if table.count_rows(f"source_id = '{source_id}'") > 0:
        return SourceRegistration(
            source_id=source_id,
            uri=uri,
            checksum=checksum,
            created=False,
            auth_ref=auth_ref,
        )

    if inspect_report is None:
        from lancedb_robotics.adapters import get_adapter

        inspect_report = get_adapter(adapter).inspect(
            uri, storage_options=storage_options, auth_ref=auth_ref
        )

    metadata = [
        {"key": "checksum", "value": checksum},
        {"key": "adapter", "value": adapter},
    ]
    for topic in inspect_report["topics"]:
        metadata.append(
            {
                "key": f"schema:{topic['topic']}",
                "value": f"{topic['schema_name']}:{topic['schema_encoding']}",
            }
        )

    row = {
        "source_id": source_id,
        "kind": "file",
        "display_name": display_name(uri),
        "uri": uri,
        "auth_ref": auth_ref,
        "metadata": metadata,
        "created_at": datetime.now(UTC),
    }
    table.add(pa.Table.from_pylist([row], schema=INTEGRATION_SOURCES_SCHEMA))
    return SourceRegistration(
        source_id=source_id,
        uri=uri,
        checksum=checksum,
        created=True,
        auth_ref=auth_ref,
    )


def register_recording_source(
    lake: Lake,
    recording: "Recording",
    *,
    adapter: str,
    inspect_report: dict,
    auth_ref: str | None = None,
) -> SourceRegistration:
    """Register a split recording as one source, keyed by its ordered shards.

    The ``source_id`` is content-addressed from the ordered shard checksums (so it
    matches the recording's ``run_id`` digest and is reorder/relocation stable).
    ``uri`` is the recording directory; ``metadata`` records the merged schema
    fingerprints plus a ``shard:N`` entry per shard (name + checksum) so the row
    explains exactly which bytes the run was assembled from. Re-registering the
    same shards is a no-op.
    """
    combined = recording_content_key(recording.checksums)
    source_id = f"src-{combined[:16]}"
    uri = recording.uri

    table = lake.table("integration_sources")
    if table.count_rows(f"source_id = '{source_id}'") > 0:
        return SourceRegistration(
            source_id=source_id,
            uri=uri,
            checksum=combined,
            created=False,
            auth_ref=auth_ref,
        )

    metadata = [
        {"key": "checksum", "value": combined},
        {"key": "adapter", "value": adapter},
        {"key": "kind", "value": "recording"},
        {"key": "shard_count", "value": str(len(recording.shards))},
    ]
    for index, shard in enumerate(recording.shards):
        metadata.append({"key": f"shard:{index}", "value": f"{shard.path.name}:{shard.checksum}"})
    for topic in inspect_report["topics"]:
        metadata.append(
            {
                "key": f"schema:{topic['topic']}",
                "value": f"{topic['schema_name']}:{topic['schema_encoding']}",
            }
        )

    row = {
        "source_id": source_id,
        "kind": "recording",
        "display_name": recording.root.name,
        "uri": uri,
        "auth_ref": auth_ref,
        "metadata": metadata,
        "created_at": datetime.now(UTC),
    }
    table.add(pa.Table.from_pylist([row], schema=INTEGRATION_SOURCES_SCHEMA))
    return SourceRegistration(
        source_id=source_id,
        uri=uri,
        checksum=combined,
        created=True,
        auth_ref=auth_ref,
    )
