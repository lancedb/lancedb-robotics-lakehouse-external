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
from typing import TYPE_CHECKING, Any

from lancedb_robotics import pylance_execution
from lancedb_robotics.capability_gates import BLOB, gated_to_lance

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from lancedb_robotics.connections import LakeConnectionSpec
    from lancedb_robotics.pylance_execution import NamespaceAccessFactory

#: Blob columns of the canonical schema, by their owning table.
PAYLOAD_BLOB_COLUMN = "payload_blob"
ATTACHMENT_DATA_COLUMN = "data"

_ROW_ID = "_rowid"


def _to_dataset(
    handle: Any,
    connection_spec: LakeConnectionSpec | None = None,
    *,
    table_name: str | None = None,
    namespace_access_factory: NamespaceAccessFactory | None = None,
) -> Any:
    """Return a ``lance.LanceDataset`` from a lancedb ``Table`` or a bare dataset.

    Routing by resolved backend (``connection_spec`` is ``lake.connection_spec``):

    * **Namespace-backed lake** (``pylance_access`` present, 0036/0129): open the
      dataset through the direct-pylance adapter --
      ``lance.dataset(namespace_client=..., table_id=...)`` with vended
      credentials refreshed before IO -- instead of ``Table.to_lance()`` on a
      remote table. The logical table name comes from ``table_name`` or the
      lancedb ``Table``'s ``name``.
    * **Otherwise** (local / object-store / ``db://`` / unclassified): the 0128
      ``blob`` capability gate runs *before* ``to_lance()`` so a ``db://`` backend
      without direct object IO fails with actionable guidance rather than a late,
      opaque error inside ``take_blobs``.
    """
    if pylance_execution.has_pylance_access(connection_spec):
        assert connection_spec is not None  # narrowed by has_pylance_access
        name = table_name or getattr(handle, "name", None)
        if name is not None:
            return pylance_execution.open_direct_dataset(
                connection_spec,
                name,
                namespace_access_factory=namespace_access_factory,
            )
        if hasattr(handle, "to_lance"):
            # A namespace-backed lancedb Table we cannot resolve a table_id for:
            # fail loudly rather than silently drop to an un-routed to_lance()
            # that would skip credential vending/refresh (SKILLS.md: a typed
            # error, never a silent degrade to a costlier/unsafe path).
            raise ValueError(
                "namespace-direct blob hydration needs a table name to resolve the "
                "namespace table_id, but the handle has no `name`; pass table_name= "
                "to the fetch_* call"
            )
        # A bare, already-opened lance dataset (no `to_lance`): it has already
        # reached storage through another path, so use it as-is.
        return handle
    return gated_to_lance(handle, connection_spec, family=BLOB, operation="blob hydration")


def fetch_blobs(
    handle: Any,
    blob_column: str,
    ids: Iterable[str],
    *,
    id_column: str,
    connection_spec: LakeConnectionSpec | None = None,
    table_name: str | None = None,
    namespace_access_factory: NamespaceAccessFactory | None = None,
) -> dict[str, bytes]:
    """Lazily read blob bytes by logical id; return ``{id: bytes}``.

    ``handle`` is a lancedb ``Table`` or a ``lance.LanceDataset``. ``ids`` are
    values of ``id_column`` (e.g. ``observation_id``). A metadata-only scan
    resolves each id to its row id, then ``take_blobs`` materializes only those
    rows -- no blob bytes are read for unrequested rows. Ids not present in the
    table are omitted from the result; a row whose blob is NULL maps to ``b""``.
    Duplicate ids are collapsed.

    ``connection_spec`` (``lake.connection_spec``) selects the data-plane: a
    namespace-backed lake opens the dataset through the direct-pylance adapter
    (credentials refreshed before IO); other backends gate the direct-Lance drop
    (``db://`` without direct object IO fails with actionable guidance). For the
    namespace route the logical table name is ``table_name`` or the lancedb
    ``Table``'s ``name``; ``namespace_access_factory`` is a test seam.
    """
    wanted = list(dict.fromkeys(str(i) for i in ids))  # de-dup, preserve order
    if not wanted:
        return {}

    dataset = _to_dataset(
        handle,
        connection_spec,
        table_name=table_name,
        namespace_access_factory=namespace_access_factory,
    )
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
    *,
    connection_spec: LakeConnectionSpec | None = None,
    table_name: str | None = None,
    namespace_access_factory: NamespaceAccessFactory | None = None,
) -> dict[int, bytes]:
    """Lazily read blob bytes by Lance row id.

    Row ids are version-relative. Callers that persist row ids must checkout the
    table version that produced them before calling this helper. See
    :func:`fetch_blobs` for the ``connection_spec`` data-plane routing.
    """
    wanted = list(dict.fromkeys(int(row_id) for row_id in row_ids))
    if not wanted:
        return {}

    dataset = _to_dataset(
        handle,
        connection_spec,
        table_name=table_name,
        namespace_access_factory=namespace_access_factory,
    )
    blob_files = dataset.take_blobs(blob_column, ids=wanted)
    return {row_id: bf.read() for row_id, bf in zip(wanted, blob_files, strict=True)}


def fetch_blob(
    handle: Any,
    blob_column: str,
    row_id: str,
    *,
    id_column: str,
    connection_spec: LakeConnectionSpec | None = None,
    table_name: str | None = None,
    namespace_access_factory: NamespaceAccessFactory | None = None,
) -> bytes | None:
    """Lazily read one row's blob bytes by logical id, or ``None`` if absent."""
    return fetch_blobs(
        handle,
        blob_column,
        [row_id],
        id_column=id_column,
        connection_spec=connection_spec,
        table_name=table_name,
        namespace_access_factory=namespace_access_factory,
    ).get(row_id)
