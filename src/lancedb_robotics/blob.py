"""Lazy reads of Lance blob-encoded columns (decision 0024 / backlog 0035).

The heavy payload columns (``observations.payload_blob``, ``attachments.data``)
are Lance blob-encoded: a scan that does not project them reads no blob bytes,
and a row's bytes are materialized lazily by id through ``take_blobs`` /
``BlobFile``. lancedb's ``Table`` does not expose ``take_blobs``, so this module
is the single seam that drops to the underlying ``lance.LanceDataset`` and lets
callers fetch bytes by the *logical* row id (``observation_id`` /
``attachment_id``) rather than a raw offset.

This is the fast-access half of "Lance is the index *and* the fast-access layer":
curation/training scans stay metadata-only and cheap, then pull the exact bytes
they need by id at the moment of use.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

#: Blob columns of the canonical schema, by their owning table.
PAYLOAD_BLOB_COLUMN = "payload_blob"
ATTACHMENT_DATA_COLUMN = "data"

_ROW_ID = "_rowid"


def _to_dataset(handle: Any) -> Any:
    """Return a ``lance.LanceDataset`` from a lancedb ``Table`` or a bare dataset."""
    return handle.to_lance() if hasattr(handle, "to_lance") else handle


def fetch_blobs(
    handle: Any,
    blob_column: str,
    ids: Iterable[str],
    *,
    id_column: str,
) -> dict[str, bytes]:
    """Lazily read blob bytes by logical id; return ``{id: bytes}``.

    ``handle`` is a lancedb ``Table`` or a ``lance.LanceDataset``. ``ids`` are
    values of ``id_column`` (e.g. ``observation_id``). A metadata-only scan
    resolves each id to its row id, then ``take_blobs`` materializes only those
    rows -- no blob bytes are read for unrequested rows. Ids not present in the
    table are omitted from the result; a row whose blob is NULL maps to ``b""``.
    Duplicate ids are collapsed.
    """
    wanted = list(dict.fromkeys(str(i) for i in ids))  # de-dup, preserve order
    if not wanted:
        return {}

    dataset = _to_dataset(handle)
    index = dataset.to_table(columns=[id_column], with_row_id=True)
    rowid_by_id = dict(zip(index[id_column].to_pylist(), index[_ROW_ID].to_pylist(), strict=True))

    present = [i for i in wanted if i in rowid_by_id]
    if not present:
        return {}
    blob_files = dataset.take_blobs(blob_column, ids=[rowid_by_id[i] for i in present])
    return {i: bf.read() for i, bf in zip(present, blob_files, strict=True)}


def fetch_blobs_by_row_id(
    handle: Any,
    blob_column: str,
    row_ids: Iterable[int],
) -> dict[int, bytes]:
    """Lazily read blob bytes by Lance row id.

    Row ids are version-relative. Callers that persist row ids must checkout the
    table version that produced them before calling this helper.
    """
    wanted = list(dict.fromkeys(int(row_id) for row_id in row_ids))
    if not wanted:
        return {}

    dataset = _to_dataset(handle)
    blob_files = dataset.take_blobs(blob_column, ids=wanted)
    return {row_id: bf.read() for row_id, bf in zip(wanted, blob_files, strict=True)}


def fetch_blob(
    handle: Any,
    blob_column: str,
    row_id: str,
    *,
    id_column: str,
) -> bytes | None:
    """Lazily read one row's blob bytes by logical id, or ``None`` if absent."""
    return fetch_blobs(handle, blob_column, [row_id], id_column=id_column).get(row_id)
