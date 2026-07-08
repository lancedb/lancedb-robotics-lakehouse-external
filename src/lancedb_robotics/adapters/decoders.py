"""Decoder dispatch for MCAP message payloads (backlog 0014, completed in 0020).

Ingest is no longer envelope-only: each message's payload bytes are decoded to a
canonical JSON form. Dispatch is keyed by ``(message_encoding, schema_encoding)``
and **enumerates all seven registry message encodings**, all of which are now
decoded when their (optional) decoder is installed:

- stdlib: ``json``.
- official upstream decoder packages: ``ros1``, ``cdr``/ros2, ``protobuf`` (decision
  ``20260613T035038Z-decode-via-official-mcap-decoder-packages-not-mc``).
- ``flatbuffer`` (backlog 0020): decoded generically from the channel schema's
  embedded ``.bfbs`` reflection via the ``flatbuffers`` runtime, so a flatbuffer
  ``foxglove.*`` message lands the same fields/vectors as its protobuf twin (see
  :mod:`lancedb_robotics.adapters.flatbuffer`).
- ``cbor`` / ``msgpack`` (backlog 0020): self-describing binary, decoded
  schema-free to a generic structure (``cbor2`` / ``msgpack`` extras). No schema
  means no typed-extraction mapping unless the decoded shape matches a structural
  matcher (0015).

Every decoder is imported lazily, so the module loads with no extra installed and
a missing extra degrades to ``raw`` (never a crash). The remaining ``raw``-by-
design cases are: a missing decoder extra, a schemaless channel for an encoding
that needs a schema (``flatbuffer``/ros/proto), and the IDL **schema**-encoding
tail (``ros2idl``/``omgidl``), which the upstream ros2 factory does not support.

A message resolves to one of three outcomes, captured in :class:`DecodeResult`:

- ``decoded`` — ``payload_json`` holds the decoded message as canonical JSON.
  Large binary fields (image/pointcloud/compressed ``data``) are elided from the
  JSON and the original message bytes are hoisted to ``payload_blob`` so the
  message can be re-decoded without the source file.
- ``raw`` — no decoder is available (decoder extra missing, or
  schemaless/unsupported schema encoding). ``payload_json`` is NULL; the original
  bytes stay recoverable through the row's pointer provenance.
- ``failed`` — a decoder was available but raised. ``payload_json`` is NULL and
  ``decode_error`` records the exception; the row is kept, never dropped.

Dispatch tolerates ``schema=None`` (schemaless ``schema_id=0`` channels) and
unsupported schema encodings (``ros2idl``, ``omgidl``): those route to ``raw``
with the encoding recorded.
"""

from __future__ import annotations

import base64
import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Every MCAP registry message encoding, so dispatch is exhaustive by name.
# Value is the import that proves a decoder is available (None => stdlib only).
DECODER_REQUIREMENTS: dict[str, str | None] = {
    "json": None,
    "ros1": "mcap_ros1",
    "cdr": "mcap_ros2",
    "protobuf": "mcap_protobuf",
    "flatbuffer": "flatbuffers",  # generic .bfbs reflection decode (backlog 0020)
    "cbor": "cbor2",  # self-describing binary (backlog 0020)
    "msgpack": "msgpack",  # self-describing binary (backlog 0020)
}

# Every registry encoding is now decode-attempted (the schema-encoding tail
# ros2idl/omgidl still routes to raw via the upstream factory, see decoder_for).
_DECODE_NOW: frozenset[str] = frozenset(DECODER_REQUIREMENTS)

# Schema-free self-describing binary encodings: decoded without a schema, so they
# resolve even on a schemaless channel (unlike flatbuffer/ros/proto).
_SCHEMALESS_BINARY: frozenset[str] = frozenset({"cbor", "msgpack"})

# Bytes fields at or above this size are hoisted to ``payload_blob`` instead of
# being base64-inlined into ``payload_json`` (keeps metadata scans cheap).
DEFAULT_BLOB_THRESHOLD = 2048

DecoderFn = Callable[[bytes], Any]
"""payload bytes -> decoded Python object."""


def can_decode(message_encoding: str) -> bool:
    """True if this package can decode ``message_encoding`` in the current env.

    ``json`` needs only the stdlib; every other registry encoding needs its
    optional decoder extra to be importable (``flatbuffers`` for ``flatbuffer``,
    ``cbor2`` for ``cbor``, ``msgpack`` for ``msgpack``, the ``mcap_*`` packages
    for ros/proto). An unknown encoding returns False. This is a message-encoding
    capability probe only: a ``cdr`` channel can report True yet still land
    ``raw`` if its *schema* encoding is an unsupported IDL variant.
    """
    if message_encoding not in _DECODE_NOW:
        return False
    requirement = DECODER_REQUIREMENTS.get(message_encoding)
    return requirement is None or importlib.util.find_spec(requirement) is not None


@dataclass(frozen=True)
class DecodeResult:
    """Outcome of decoding one message payload.

    ``status`` is one of ``decoded`` | ``raw`` | ``failed``. ``payload_json`` is
    populated only on ``decoded``; ``payload_blob`` holds hoisted large-binary
    bytes (NULL for scalar messages); ``error`` is set only on ``failed``/``raw``
    to explain the outcome.
    """

    status: str
    payload_json: str | None = None
    payload_blob: bytes | None = None
    error: str | None = None


class PayloadDecoder:
    """Resolves and caches per-encoding decoders, then decodes message bytes.

    Upstream ``DecoderFactory`` instances are created lazily on first use so the
    module imports without any decoder extra installed.
    """

    def __init__(self, blob_threshold: int = DEFAULT_BLOB_THRESHOLD) -> None:
        self.blob_threshold = blob_threshold
        self._factories: dict[str, Any] = {}
        # Parsed flatbuffer reflection schemas, cached per MCAP schema id (parsing
        # a .bfbs is the one-time cost amortized across a channel's messages).
        self._flatbuffer_schemas: dict[int, Any] = {}

    def decoder_for(self, message_encoding: str, schema: Any | None) -> DecoderFn | None:
        """Return a ``bytes -> object`` decoder, or None if none applies.

        None means "route to raw": unknown encoding, missing decoder extra, a
        schemaless channel for an encoding that needs a schema, or an unsupported
        schema encoding (ros2idl/omgidl) for a ros stream.
        """
        if message_encoding not in _DECODE_NOW:
            return None  # unknown encoding -> raw

        if message_encoding == "json":
            # jsonschema or schemaless: JSON is self-describing either way.
            return _json_decoder

        if message_encoding in _SCHEMALESS_BINARY:
            # cbor/msgpack are self-describing: no schema needed, but the optional
            # lib must be present (else degrade to raw, never crash).
            if not can_decode(message_encoding):
                return None
            return _cbor_decoder if message_encoding == "cbor" else _msgpack_decoder

        if not can_decode(message_encoding):
            return None  # decoder extra not installed -> raw

        if schema is None:
            return None  # flatbuffer/ros/proto need a schema to decode -> raw

        if message_encoding == "flatbuffer":
            # Decode generically from the channel schema's embedded .bfbs
            # reflection; the parsed schema is cached per schema id.
            return self._flatbuffer_decoder(schema)

        factory = self._factory_for(message_encoding)
        if factory is None:
            return None
        # Upstream factories key on (message_encoding, schema) and return None
        # for a schema encoding they do not handle (e.g. ros2idl/omgidl).
        return factory.decoder_for(message_encoding, schema)

    def _flatbuffer_decoder(self, schema: Any) -> DecoderFn:
        """A ``bytes -> dict`` decoder for ``schema``'s embedded .bfbs reflection.

        Caches the parsed reflection per schema id. Raises if the ``.bfbs`` is
        unparseable; :meth:`decode` turns that into a ``failed`` result rather
        than letting it escape (consistent with the ros/proto factories).
        """
        schema_id = getattr(schema, "id", None)
        cached = self._flatbuffer_schemas.get(schema_id) if schema_id is not None else None
        if cached is None:
            from lancedb_robotics.adapters.flatbuffer import FlatbufferReflection

            cached = FlatbufferReflection(schema.data)
            if schema_id is not None:
                self._flatbuffer_schemas[schema_id] = cached
        return cached.decode

    def decode(self, message_encoding: str, schema: Any | None, data: bytes) -> DecodeResult:
        """Decode one payload into a :class:`DecodeResult` (never raises)."""
        try:
            # Resolving the decoder can itself raise: upstream factories parse the
            # schema descriptor eagerly, so a malformed/corrupt schema blows up
            # here rather than in the decode call. That is still a decode failure,
            # not an ingest crash.
            decoder = self.decoder_for(message_encoding, schema)
        except Exception as exc:  # noqa: BLE001 - decoder-init errors are first-class data
            return DecodeResult(status="failed", error=f"decoder-init {type(exc).__name__}: {exc}")
        if decoder is None:
            schema_encoding = getattr(schema, "encoding", None)
            return DecodeResult(status="raw", error=_raw_reason(message_encoding, schema_encoding))
        try:
            obj = decoder(data)
        except Exception as exc:  # noqa: BLE001 - decode errors are first-class data
            return DecodeResult(status="failed", error=f"{type(exc).__name__}: {exc}")
        blobs: list[bytes] = []
        jsonable = _to_jsonable(obj, blobs, self.blob_threshold)
        payload_blob = data if blobs else None  # re-decodable raw bytes when large-binary
        try:
            payload_json = json.dumps(jsonable, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as exc:
            return DecodeResult(status="failed", error=f"json-serialize: {exc}")
        return DecodeResult(status="decoded", payload_json=payload_json, payload_blob=payload_blob)

    def _factory_for(self, message_encoding: str) -> Any | None:
        if message_encoding in self._factories:
            return self._factories[message_encoding]
        factory: Any | None = None
        if message_encoding == "ros1":
            from mcap_ros1.decoder import DecoderFactory

            factory = DecoderFactory()
        elif message_encoding == "cdr":
            from mcap_ros2.decoder import DecoderFactory

            factory = DecoderFactory()
        elif message_encoding == "protobuf":
            from mcap_protobuf.decoder import DecoderFactory

            factory = DecoderFactory()
        self._factories[message_encoding] = factory
        return factory


def _raw_reason(message_encoding: str, schema_encoding: str | None) -> str:
    if message_encoding not in DECODER_REQUIREMENTS:
        return f"unknown message encoding '{message_encoding}'"
    if not can_decode(message_encoding):
        requirement = DECODER_REQUIREMENTS[message_encoding]
        return f"decoder for '{message_encoding}' not installed (needs {requirement})"
    if schema_encoding is None:
        return f"no schema for '{message_encoding}' channel"
    # The decoder is installed and a schema is present, but the upstream factory
    # declined it: an unsupported schema encoding such as ros2idl/omgidl.
    return f"no decoder for schema encoding '{schema_encoding}'"


def _json_decoder(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def _cbor_decoder(data: bytes) -> Any:
    import cbor2

    return cbor2.loads(data)


def _msgpack_decoder(data: bytes) -> Any:
    import msgpack

    # raw=False decodes msgpack str family to ``str`` (bin family stays ``bytes``,
    # which the blob-hoist logic then handles like any other binary field).
    return msgpack.unpackb(data, raw=False)


def _to_jsonable(obj: Any, blobs: list[bytes], threshold: int) -> Any:
    """Recursively convert a decoded message into JSON-serializable values.

    Large ``bytes`` fields are recorded in ``blobs`` and replaced by an elision
    marker (so ``payload_json`` stays scannable); small ``bytes`` are base64-
    inlined so the JSON is self-contained. Handles plain containers, ros1/ros2
    ``__slots__`` messages, protobuf messages, namedtuples, and numpy values.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray, memoryview)):
        raw = bytes(obj)
        if len(raw) >= threshold:
            blobs.append(raw)
            return {"__elided_bytes__": len(raw)}
        return {"__b64__": base64.b64encode(raw).decode("ascii")}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v, blobs, threshold) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, blobs, threshold) for v in obj]

    # protobuf message -> dict (bytes base64'd by the converter).
    if hasattr(obj, "DESCRIPTOR") and hasattr(type(obj), "ListFields"):
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(obj, preserving_proto_field_name=True)

    # numpy scalar / array, without importing numpy.
    if hasattr(obj, "item") and hasattr(obj, "dtype") and not hasattr(obj, "__len__"):
        return obj.item()
    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):
        return _to_jsonable(obj.tolist(), blobs, threshold)

    # ros1/ros2 dynamic messages carry their fields in __slots__.
    slots = getattr(obj, "__slots__", None)
    if slots:
        return {
            str(name): _to_jsonable(getattr(obj, name), blobs, threshold)
            for name in slots
            if hasattr(obj, name)
        }

    # namedtuple
    if hasattr(obj, "_asdict"):
        return _to_jsonable(obj._asdict(), blobs, threshold)

    if hasattr(obj, "__dict__"):
        return {
            str(k): _to_jsonable(v, blobs, threshold)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    return str(obj)  # last resort: keep something queryable rather than crash


__all__ = [
    "DECODER_REQUIREMENTS",
    "DEFAULT_BLOB_THRESHOLD",
    "DecodeResult",
    "DecoderFn",
    "PayloadDecoder",
    "can_decode",
]
