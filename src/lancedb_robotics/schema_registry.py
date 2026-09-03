"""Content-addressed registry for MCAP/ROS message schema definitions.

``adapters/decoders.py``'s ``PayloadDecoder.decode(message_encoding, schema,
data)`` needs a real schema object exposing ``.name``/``.encoding``/``.data``
to re-decode ros1/cdr/protobuf/flatbuffer payloads -- ``observations.
schema_encoding`` alone is not enough (see ``schemas/__init__.py``'s
``OBSERVATIONS_SCHEMA`` v5 comment; both ``adapters/mcap_adapter.py`` and
``adapters/rosbag_adapter.py`` already have this real schema object in hand
at ingest time, but only its ``.encoding`` survives onto the observation
row). This module persists the schema-definition bytes once per distinct
``(name, encoding, data)`` triple -- mirrors ``keyframe_map_artifacts``'
content addressing (``keyframe_maps.py``) -- so a live reader can re-decode
``payload_blob`` without reopening the original source file the way
``dataset_export.py``'s exporter does via ``raw_uri``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import pyarrow as pa
from mcap.records import Schema

from lancedb_robotics.adapters.decoders import DecodeResult, PayloadDecoder
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import SCHEMA_REGISTRY_SCHEMA


def schema_digest(*, name: str, encoding: str, data: bytes) -> str:
    """Content digest for a schema definition -- the registry's key.

    Length-prefixed join (mirrors ``video.py``'s ``_sha256_join``) so
    ``name``/``encoding``/``data`` can never collide across a naive
    concatenation boundary.
    """
    digest = hashlib.sha256()
    for chunk in (name.encode(), encoding.encode(), data):
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest()


def schema_registry_row(
    *, name: str, encoding: str, data: bytes, created_at: datetime
) -> dict[str, Any]:
    """Build one ``schema_registry`` row. Caller dedupes and writes it."""
    return {
        "schema_digest": schema_digest(name=name, encoding=encoding, data=data),
        "schema_name": name,
        "schema_encoding": encoding,
        "data": data,
        "created_at": created_at,
    }


def write_schema_registry_rows(lake: Lake, rows: list[dict[str, Any]]) -> None:
    """Bulk-write newly-seen schema rows.

    Content-addressed: a duplicate digest written by a separate ingest run is
    harmless (same convention as ``keyframe_map_artifacts`` -- readers match
    on digest and take the first hit, never enforcing uniqueness), so this
    never needs a pre-write existence check against the table.
    """
    if not rows:
        return
    lake.table("schema_registry").add(pa.Table.from_pylist(rows, schema=SCHEMA_REGISTRY_SCHEMA))


def fetch_registered_schema(
    lake: Lake,
    digest: str,
    *,
    cache: dict[str, Schema | None] | None = None,
) -> Schema | None:
    """Look up one schema_registry row and return it as a real ``mcap`` ``Schema``.

    ``Schema.id`` is derived deterministically from the digest (truncated to
    an int) rather than a placeholder constant -- ``PayloadDecoder``'s
    flatbuffer path caches parsed reflection per ``schema.id``
    (``adapters/decoders.py``'s ``_flatbuffer_decoder``); a constant id would
    silently reuse one schema's cached reflection for a different one.
    """
    if cache is not None and digest in cache:
        return cache[digest]
    rows = (
        lake.table("schema_registry")
        .search()
        .where(f"schema_digest = {_sql_literal(digest)}")
        .limit(1)
        .to_arrow()
        .to_pylist()
    )
    result = (
        Schema(
            id=int(digest[:16], 16),
            name=rows[0]["schema_name"],
            encoding=rows[0]["schema_encoding"],
            data=rows[0]["data"],
        )
        if rows
        else None
    )
    if cache is not None:
        cache[digest] = result
    return result


def redecode_payload_blob(
    lake: Lake,
    observation_row: Any,
    *,
    decoder: PayloadDecoder | None = None,
    schema_cache: dict[str, Schema | None] | None = None,
) -> DecodeResult:
    """Re-decode one observation's ``payload_blob`` using its registered schema.

    Never touches ``raw_uri`` -- unlike ``dataset_export.py``'s exporter
    workaround, this is safe for a live, random-access reader. Returns
    ``status="raw"`` (never raises) when the row predates ``schema_digest``
    or its registry entry is missing; callers should treat that as "not
    decodable from this row alone," not as an error to propagate.
    """
    digest = observation_row.get("schema_digest")
    message_encoding = observation_row.get("message_encoding")
    payload_blob = observation_row.get("payload_blob")
    if not digest or not message_encoding or payload_blob is None:
        return DecodeResult(
            status="raw",
            error="missing schema_digest/message_encoding/payload_blob to re-decode",
        )
    schema = fetch_registered_schema(lake, digest, cache=schema_cache)
    if schema is None:
        return DecodeResult(
            status="raw", error=f"schema_digest {digest!r} not found in schema_registry"
        )
    return (decoder or PayloadDecoder()).decode(message_encoding, schema, payload_blob)


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"
