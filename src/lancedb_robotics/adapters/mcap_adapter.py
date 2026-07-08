"""MCAP adapter: inspect, ingest, and slice a robot log.

Inspection answers "what is inside this file" with deterministic metadata:
topics, encodings, schemas, counts, time ranges, chunk offsets, and whether
this package can decode each stream. Downstream ingest (backlog 0004) consumes
this payload to plan routing.

Real corpora are compressed (didi/demo zstd, nuScenes lz4) and occasionally
damaged, so reads classify their failures (backlog 0017):

- **codec gap** (:class:`CodecUnavailableError`) — a chunk needs a codec that is
  not installed. A hard error naming the package to install; never quarantined.
- **corruption** (:class:`CorruptMcapError`) — a CRC mismatch or a truncated
  byte stream. Ingest yields the readable prefix first, then raises, so the run
  is quarantined with a reason instead of lost.
- everything else (bad magic, missing file) stays a plain :class:`AdapterError`.
"""

import hashlib
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mcap.exceptions import (
    EndOfFile,
    McapError,
    RecordLengthLimitExceeded,
    UnsupportedCompressionError,
)
from mcap.reader import make_reader
from mcap.records import Attachment, Channel, Message, Metadata, Schema
from mcap.stream_reader import CRCValidationError, StreamReader
from mcap.writer import CompressionType, Writer

from lancedb_robotics.adapters import (
    AdapterError,
    AdapterInfo,
    CodecUnavailableError,
    CorruptMcapError,
)
from lancedb_robotics.adapters.decoders import PayloadDecoder, can_decode
from lancedb_robotics.storage import StorageConfigError, open_binary_uri

# Decode-capability probe lives in the decoder dispatch module (backlog 0014),
# which enumerates every registry encoding. Kept under the old private name so
# inspect's call site reads unchanged.
_can_decode = can_decode

# Chunk-compression names accepted by ``export`` (backlog 0017), mapped to the
# writer's enum. ``None`` keeps the writer default (zstd). Slicing a window into
# a chosen codec is how the deterministic lz4/zstd CI fixtures are minted.
_COMPRESSION_BY_NAME: dict[str, CompressionType] = {
    "none": CompressionType.NONE,
    "lz4": CompressionType.LZ4,
    "zstd": CompressionType.ZSTD,
}

# The codec name the `mcap` reader reports == the PyPI package name for it.
_CODEC_PACKAGES: dict[str, str] = {"zstandard": "zstandard", "lz4": "lz4"}

# Read failures that mean "damaged or not MCAP", caught wholesale and classified.
# CRCValidationError is a ValueError and struct.error is stdlib -- neither is an
# McapError -- so all three must be named explicitly.
_READ_ERRORS = (CRCValidationError, McapError, struct.error)


def _codec_error(exc: UnsupportedCompressionError, path: str | Path) -> CodecUnavailableError:
    # mcap raises UnsupportedCompressionError(f"unsupported compression type {codec}"),
    # so the bare codec name is the final token of the message.
    message = str(exc)
    codec = message.split()[-1] if message else "unknown"
    pkg = _CODEC_PACKAGES.get(codec, codec)
    return CodecUnavailableError(
        f"MCAP chunk uses '{codec}' compression but the '{codec}' codec is not "
        f"installed; run `pip install {pkg}` (or `uv add {pkg}`) and retry: {path}",
        codec=codec,
    )


def _reason(exc: Exception) -> str:
    """A non-empty human reason for an exception (some, e.g. EndOfFile, stringify empty)."""
    return str(exc) or type(exc).__name__


def _classify_read_error(exc: Exception, path: str | Path) -> AdapterError:
    """Map a low-level read failure onto the adapter's error taxonomy."""
    if isinstance(exc, UnsupportedCompressionError):
        return _codec_error(exc, path)
    reason = _reason(exc)
    if isinstance(exc, CRCValidationError):
        return CorruptMcapError(
            f"MCAP CRC validation failed: {path} ({reason})",
            status="crc-mismatch",
            reason=reason,
        )
    if isinstance(exc, (RecordLengthLimitExceeded, EndOfFile, struct.error)):
        return CorruptMcapError(
            f"MCAP appears truncated or corrupt: {path} ({reason})",
            status="truncated",
            reason=reason,
        )
    return AdapterError(f"not a valid MCAP file: {path} ({reason})")


class McapAdapter:
    info = AdapterInfo(name="mcap", format="mcap", capabilities=("inspect", "ingest", "export"))

    def inspect(
        self,
        path: str | Path,
        *,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> dict:
        """Return deterministic metadata about an MCAP file without ingesting it.

        A summary-less / unindexed file (live, append-only, or never finalized,
        backlog 0018) is a legitimate form, not an error: its metadata is
        recovered from one linear record pass and the report is marked
        ``indexed: False`` so downstream knows the stats are scan-derived rather
        than summary-derived.

        Raises :class:`CodecUnavailableError` if a chunk's codec is missing,
        :class:`CorruptMcapError` if the file is truncated/CRC-damaged, or a plain
        :class:`AdapterError` if it is missing or not MCAP. Inspect does not
        validate CRCs (it is the cheap planning pass); CRC mismatches surface
        during ingest, where the run can be quarantined.
        """
        try:
            with open_binary_uri(path, storage_options=storage_options, auth_ref=auth_ref) as stream:
                return self._inspect_stream(stream, path)
        except StorageConfigError as exc:
            raise AdapterError(str(exc)) from exc
        except _READ_ERRORS as exc:
            raise _classify_read_error(exc, path) from exc

    def ingest(
        self,
        path: str | Path,
        *,
        validate_crcs: bool = True,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> Iterator[dict]:
        """Yield one canonical message record per MCAP message, in log order.

        Each record carries envelope provenance (``topic``/``log_time_ns``/
        ``sequence``/``message_encoding``/``schema_name``) plus the decoded
        payload (backlog 0014): ``schema_encoding`` and ``decode_status``
        (``decoded`` | ``raw`` | ``failed``), ``payload_json`` (canonical JSON of
        the decoded message, NULL when undecodable), ``payload_blob`` (large
        binary bytes hoisted out, NULL for scalar messages), and ``decode_error``
        for non-decoded outcomes. Decode is dispatched per channel by
        ``(message_encoding, schema_encoding)``, so a single mixed-encoding file
        (json + protobuf + ros1, as in nuScenes) is handled correctly. A decode
        failure keeps the row; no message is dropped.

        ``sequence`` is the per-topic message index (0-based, log-time order),
        not MCAP's optional sequence field, which many writers leave at zero.
        Together with ``topic`` and ``log_time_ns`` it is enough provenance to
        re-locate the original bytes in the raw file.

        Integrity (backlog 0017): chunk/data CRCs are validated as messages are
        read when ``validate_crcs`` is set (the default; pass ``False`` to skip on
        the hot path for trusted data). On a CRC mismatch or a truncated file the
        readable prefix is yielded first and then :class:`CorruptMcapError` is
        raised, so the caller keeps the recovered rows and quarantines the run. A
        missing codec raises :class:`CodecUnavailableError` (a hard error).
        """
        per_topic_index: dict[str, int] = {}
        decoder = PayloadDecoder()
        yielded = 0
        try:
            with open_binary_uri(path, storage_options=storage_options, auth_ref=auth_ref) as stream:
                reader = make_reader(stream, validate_crcs=validate_crcs)
                for schema, channel, message in reader.iter_messages(log_time_order=True):
                    yield self._message_row(channel, schema, message, per_topic_index, decoder)
                    yielded += 1
            return  # healthy, complete read
        except StorageConfigError as exc:
            raise AdapterError(str(exc)) from exc
        except UnsupportedCompressionError as exc:
            raise _codec_error(exc, path) from exc
        except _READ_ERRORS as exc:
            if yielded:
                # The seeking reader already streamed a log-time-ordered prefix;
                # keep it and flag corruption (re-reading would duplicate rows).
                status = "crc-mismatch" if isinstance(exc, CRCValidationError) else "truncated"
                raise CorruptMcapError(
                    f"MCAP read stopped after {yielded} message(s): {path} ({_reason(exc)})",
                    status=status,
                    reason=_reason(exc),
                    recovered=yielded,
                ) from exc
            # Nothing yielded (e.g. a truncated footer breaks the index up-front):
            # fall through to a forward streaming pass to recover the prefix.
        yield from self._recover_prefix(
            path,
            per_topic_index,
            decoder,
            validate_crcs,
            storage_options=storage_options,
            auth_ref=auth_ref,
        )

    def _recover_prefix(
        self,
        path: str | Path,
        per_topic_index: dict[str, int],
        decoder: PayloadDecoder,
        validate_crcs: bool,
        *,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> Iterator[dict]:
        """Stream a damaged file forward, yielding every message before the damage.

        The seeking reader needs an intact summary/index; when that is gone, this
        forward pass over raw records recovers whatever decodes cleanly and then
        raises :class:`CorruptMcapError` (or :class:`AdapterError` if the file was
        never MCAP). Recovered rows are in file order, not strict log-time order;
        the run is quarantined, so downstream features exclude it regardless.
        """
        schemas: dict[int, Schema] = {}
        channels: dict[int, Channel] = {}
        recovered = 0
        try:
            with open_binary_uri(path, storage_options=storage_options, auth_ref=auth_ref) as stream:
                records = StreamReader(stream, validate_crcs=validate_crcs).records
                while True:
                    try:
                        record = next(records)
                    except StopIteration:
                        return  # streamed cleanly to the end -- nothing corrupt to report
                    except UnsupportedCompressionError as exc:
                        raise _codec_error(exc, path) from exc
                    except _READ_ERRORS as exc:
                        raise self._recovery_error(exc, path, recovered) from exc
                    if isinstance(record, Schema):
                        schemas[record.id] = record
                    elif isinstance(record, Channel):
                        channels[record.id] = record
                    elif isinstance(record, Message):
                        channel = channels.get(record.channel_id)
                        if channel is None:
                            continue
                        schema = schemas.get(channel.schema_id) if channel.schema_id else None
                        yield self._message_row(channel, schema, record, per_topic_index, decoder)
                        recovered += 1
        except StorageConfigError as exc:
            raise AdapterError(str(exc)) from exc

    @staticmethod
    def _recovery_error(exc: Exception, path: str | Path, recovered: int) -> AdapterError:
        classified = _classify_read_error(exc, path)
        if isinstance(classified, CorruptMcapError):
            classified.recovered = recovered
            return classified
        if recovered == 0:
            return classified  # never looked like MCAP (e.g. bad magic)
        # Recovered a real prefix, then hit a non-corruption read error: the
        # stream ended abnormally -- treat as truncation so the run is kept.
        return CorruptMcapError(
            f"MCAP stream ended abnormally after {recovered} message(s): {path} ({_reason(exc)})",
            status="truncated",
            reason=_reason(exc),
            recovered=recovered,
        )

    @staticmethod
    def _message_row(
        channel: Channel,
        schema: Schema | None,
        message: Message,
        per_topic_index: dict[str, int],
        decoder: PayloadDecoder,
    ) -> dict:
        topic = channel.topic
        sequence = per_topic_index.get(topic, 0)
        per_topic_index[topic] = sequence + 1
        result = decoder.decode(channel.message_encoding, schema, message.data)
        return {
            "topic": topic,
            "log_time_ns": message.log_time,
            "publish_time_ns": message.publish_time,
            "sequence": sequence,
            "message_encoding": channel.message_encoding,
            "schema_name": schema.name if schema else None,
            "schema_encoding": schema.encoding if schema else None,
            "decode_status": result.status,
            "decode_error": result.error,
            "payload_json": result.payload_json,
            "payload_blob": result.payload_blob,
        }

    def attachments(
        self,
        path: str | Path,
        *,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> Iterator[dict]:
        """Yield one record per MCAP attachment, in (log_time, name) order.

        Attachments are embedded files travelling with the log — camera
        calibration, intrinsics, mission/config blobs, thumbnails (backlog
        0016). Each record carries ``name``/``media_type``/``size``/``sha256``
        (the manifest) plus ``log_time_ns``/``create_time_ns`` and the raw
        ``data`` bytes, so the attachment is recoverable from the lake without
        reopening the source. Files with no attachments yield nothing.

        Raises the adapter's read-error taxonomy (codec / corruption / invalid).
        """
        try:
            with open_binary_uri(path, storage_options=storage_options, auth_ref=auth_ref) as stream:
                reader = make_reader(stream)
                records = [
                    {
                        "name": att.name,
                        "media_type": att.media_type,
                        "size": len(att.data),
                        "sha256": hashlib.sha256(att.data).hexdigest(),
                        "log_time_ns": att.log_time,
                        "create_time_ns": att.create_time,
                        "data": att.data,
                    }
                    for att in reader.iter_attachments()
                ]
        except StorageConfigError as exc:
            raise AdapterError(str(exc)) from exc
        except _READ_ERRORS as exc:
            raise _classify_read_error(exc, path) from exc
        records.sort(key=lambda a: (a["log_time_ns"], a["name"]))
        yield from records

    def metadata_records(
        self,
        path: str | Path,
        *,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> Iterator[dict]:
        """Yield one record per MCAP metadata record, in name order.

        Metadata records are log-level key/value blocks distinct from per-channel
        metadata (run conditions, vehicle/site identifiers, tooling versions) —
        e.g. nuScenes' single ``scene-info`` record (backlog 0016). Each record
        carries ``name`` and a ``metadata`` dict. Files with no metadata records
        yield nothing.

        Raises the adapter's read-error taxonomy (codec / corruption / invalid).
        """
        try:
            with open_binary_uri(path, storage_options=storage_options, auth_ref=auth_ref) as stream:
                reader = make_reader(stream)
                records = [
                    {"name": md.name, "metadata": dict(md.metadata)}
                    for md in reader.iter_metadata()
                ]
        except StorageConfigError as exc:
            raise AdapterError(str(exc)) from exc
        except _READ_ERRORS as exc:
            raise _classify_read_error(exc, path) from exc
        records.sort(key=lambda m: m["name"])
        yield from records

    def export(
        self,
        path: str | Path,
        *,
        start_time_ns: int,
        end_time_ns: int,
        out_path: str | Path,
        topics: tuple[str, ...] | list[str] | None = None,
        compression: str | None = None,
        storage_options: dict[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> dict:
        """Write a new MCAP containing the messages in ``[start, end]`` (inclusive).

        Reversible, lossless slicing: schemas, channels, and message bytes are
        copied verbatim, so the clip opens in any MCAP-aware tool. Optional
        ``topics`` restricts the slice to those channels. ``compression`` picks
        the output chunk codec (``"zstd"`` | ``"lz4"`` | ``"none"``); the default
        keeps the writer's zstd. Choosing the codec is how the deterministic
        lz4/zstd CI fixtures are sliced from real corpora (backlog 0017).

        Raises the adapter's read-error taxonomy if the source cannot be read.
        """
        writer_kwargs: dict = {}
        if compression is not None:
            try:
                writer_kwargs["compression"] = _COMPRESSION_BY_NAME[compression]
            except KeyError:
                raise AdapterError(
                    f"unknown compression {compression!r}; "
                    f"expected one of {sorted(_COMPRESSION_BY_NAME)}"
                ) from None
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        topic_filter = set(topics) if topics else None
        written_topics: set[str] = set()
        count = 0
        try:
            with (
                open_binary_uri(path, storage_options=storage_options, auth_ref=auth_ref) as source,
                out_path.open("wb") as sink,
            ):
                reader = make_reader(source)
                writer = Writer(sink, **writer_kwargs)
                writer.start()
                schema_ids: dict[int, int] = {}
                channel_ids: dict[int, int] = {}
                for schema, channel, message in reader.iter_messages(log_time_order=True):
                    if not (start_time_ns <= message.log_time <= end_time_ns):
                        continue
                    if topic_filter is not None and channel.topic not in topic_filter:
                        continue
                    if schema is not None and schema.id not in schema_ids:
                        schema_ids[schema.id] = writer.register_schema(
                            name=schema.name, encoding=schema.encoding, data=schema.data
                        )
                    if channel.id not in channel_ids:
                        channel_ids[channel.id] = writer.register_channel(
                            topic=channel.topic,
                            message_encoding=channel.message_encoding,
                            schema_id=schema_ids.get(schema.id, 0) if schema is not None else 0,
                            metadata=dict(channel.metadata or {}),
                        )
                    writer.add_message(
                        channel_id=channel_ids[channel.id],
                        log_time=message.log_time,
                        data=message.data,
                        publish_time=message.publish_time,
                        sequence=message.sequence,
                    )
                    written_topics.add(channel.topic)
                    count += 1
                writer.finish()
        except StorageConfigError as exc:
            raise AdapterError(str(exc)) from exc
        except UnsupportedCompressionError as exc:
            raise _codec_error(exc, path) from exc
        except _READ_ERRORS as exc:
            raise _classify_read_error(exc, path) from exc
        return {
            "out_path": str(out_path),
            "message_count": count,
            "topics": tuple(sorted(written_topics)),
            "start_time_ns": start_time_ns,
            "end_time_ns": end_time_ns,
        }

    def _inspect_stream(self, stream, path: str | Path) -> dict:
        reader = make_reader(stream)
        header = reader.get_header()
        summary = reader.get_summary()
        if summary is not None and summary.statistics is not None:
            return self._inspect_from_summary(reader, header, summary, path)
        # No summary section (footer.summary_start == 0) or no statistics: a
        # legitimate unindexed form -- a live/streaming or append-only recording,
        # or one whose writer never flushed the summary (the summary is written
        # last). The bytes are valid, just unindexed, so recover the same metadata
        # from one linear pass. This is distinct from truncation/corruption
        # (backlog 0017), which still surfaces via the read-error taxonomy.
        return self._inspect_from_scan(stream, header, path)

    def _inspect_from_summary(self, reader, header, summary, path: str | Path) -> dict:
        """Inspect via the summary section: counts/ranges are index-derived."""
        stats = summary.statistics

        # Per-topic time ranges are not in the summary; one message pass
        # recovers them. Counts come from the pass too, so they stay
        # consistent with the ranges even if the summary is stale.
        per_topic: dict[str, dict] = {}
        for _schema, channel, message in reader.iter_messages():
            self._tally(per_topic, channel.topic, message.log_time)

        topics = self._aggregate_topics(summary.channels.values(), summary.schemas, per_topic)

        chunks = [
            {
                "offset": chunk.chunk_start_offset,
                "length": chunk.chunk_length,
                "message_start_time_ns": chunk.message_start_time,
                "message_end_time_ns": chunk.message_end_time,
                "compression": chunk.compression,
            }
            for chunk in sorted(summary.chunk_indexes, key=lambda c: c.chunk_start_offset)
        ]

        # Attachment + metadata records (backlog 0016): MCAP's other two
        # first-class record types, which message-only readers ignore. Attachment
        # summaries come from the summary index, so their bytes are never read
        # during inspect; metadata records are small and recovered by an indexed
        # seek to report each record's name and key set.
        attachments = [
            {
                "name": idx.name,
                "media_type": idx.media_type,
                "size": idx.data_size,
                "log_time_ns": idx.log_time,
                "offset": idx.offset,
            }
            for idx in sorted(
                summary.attachment_indexes, key=lambda a: (a.log_time, a.offset, a.name)
            )
        ]
        metadata = [
            {"name": md.name, "keys": sorted(md.metadata)}
            for md in sorted(reader.iter_metadata(), key=lambda m: m.name)
        ]

        return self._report(
            header=header,
            path=path,
            indexed=True,
            message_count=stats.message_count,
            schema_count=stats.schema_count,
            channel_count=stats.channel_count,
            chunk_count=stats.chunk_count,
            start_time_ns=stats.message_start_time,
            end_time_ns=stats.message_end_time,
            topics=topics,
            chunks=chunks,
            attachments=attachments,
            metadata=metadata,
        )

    def _inspect_from_scan(self, stream, header, path: str | Path) -> dict:
        """Inspect a summary-less / unindexed file via one forward record pass.

        Recovers message count, per-topic counts and time ranges, the
        schema/channel inventory, and any attachment/metadata records by reading
        records linearly (chunks are broken up transparently). Chunk offsets need
        the summary index, so ``chunks`` is empty and ``chunk_count`` is 0; the
        ``indexed: False`` flag marks every stat as scan-derived, not
        summary-derived.
        """
        stream.seek(0)
        schemas: dict[int, Schema] = {}
        channels: dict[int, Channel] = {}
        per_topic: dict[str, dict] = {}
        attachments: list[dict] = []
        metadata: list[dict] = []
        message_count = 0
        start_time_ns: int | None = None
        end_time_ns: int | None = None
        for record in StreamReader(stream).records:
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                channels[record.id] = record
                per_topic.setdefault(record.topic, {"count": 0, "start": None, "end": None})
            elif isinstance(record, Message):
                channel = channels.get(record.channel_id)
                if channel is None:
                    continue  # message before its channel was declared; skip defensively
                self._tally(per_topic, channel.topic, record.log_time)
                message_count += 1
                t = record.log_time
                start_time_ns = t if start_time_ns is None else min(start_time_ns, t)
                end_time_ns = t if end_time_ns is None else max(end_time_ns, t)
            elif isinstance(record, Attachment):
                attachments.append(
                    {
                        "name": record.name,
                        "media_type": record.media_type,
                        "size": len(record.data),
                        "log_time_ns": record.log_time,
                        "offset": None,  # no summary index to source a byte offset
                    }
                )
            elif isinstance(record, Metadata):
                metadata.append({"name": record.name, "keys": sorted(record.metadata)})

        topics = self._aggregate_topics(channels.values(), schemas, per_topic)
        attachments.sort(key=lambda a: (a["log_time_ns"], a["name"]))
        metadata.sort(key=lambda m: m["name"])

        return self._report(
            header=header,
            path=path,
            indexed=False,
            message_count=message_count,
            schema_count=len(schemas),
            channel_count=len(channels),
            chunk_count=0,
            start_time_ns=start_time_ns if start_time_ns is not None else 0,
            end_time_ns=end_time_ns if end_time_ns is not None else 0,
            topics=topics,
            chunks=[],
            attachments=attachments,
            metadata=metadata,
        )

    @staticmethod
    def _tally(per_topic: dict[str, dict], topic: str, log_time: int) -> None:
        """Fold one message's log_time into a topic's running count/time range."""
        entry = per_topic.setdefault(topic, {"count": 0, "start": None, "end": None})
        entry["count"] += 1
        entry["start"] = log_time if entry["start"] is None else min(entry["start"], log_time)
        entry["end"] = log_time if entry["end"] is None else max(entry["end"], log_time)

    @staticmethod
    def _topic_entry(channel, schemas, per_topic: dict[str, dict]) -> dict:
        """Build one inspect ``topics`` entry from a channel + its (optional) schema."""
        schema = schemas.get(channel.schema_id) if channel.schema_id else None
        seen = per_topic.get(channel.topic, {"count": 0, "start": None, "end": None})
        return {
            "topic": channel.topic,
            "message_encoding": channel.message_encoding,
            "schema_name": schema.name if schema else None,
            "schema_encoding": schema.encoding if schema else None,
            "message_count": seen["count"],
            "start_time_ns": seen["start"],
            "end_time_ns": seen["end"],
            "can_decode": _can_decode(channel.message_encoding),
        }

    @classmethod
    def _aggregate_topics(cls, channels, schemas, per_topic: dict[str, dict]) -> list[dict]:
        """Build the ``topics`` list with exactly one entry per topic, name-sorted.

        Real ROS logs routinely advertise one topic under several channels (every
        node publishes ``/rosout``; ``/diagnostics`` has many publishers), so a
        plain per-channel list would emit duplicate rows whose ``message_count``s
        each repeat the topic's full total — making any consumer that sums the
        list over-count. ``per_topic`` already tallies counts/ranges by topic, so
        the merge keeps one row per topic and folds the channels' decode/schema
        facts together (``can_decode`` if *any* channel decodes).
        """
        by_topic: dict[str, dict] = {}
        for channel in channels:
            entry = cls._topic_entry(channel, schemas, per_topic)
            existing = by_topic.get(entry["topic"])
            if existing is None:
                by_topic[entry["topic"]] = entry
                continue
            for key in ("schema_name", "schema_encoding", "message_encoding"):
                if existing.get(key) is None and entry.get(key) is not None:
                    existing[key] = entry[key]
            existing["can_decode"] = existing["can_decode"] or entry["can_decode"]
        return sorted(by_topic.values(), key=lambda t: t["topic"])

    def _report(
        self,
        *,
        header,
        path: str | Path,
        indexed: bool,
        message_count: int,
        schema_count: int,
        channel_count: int,
        chunk_count: int,
        start_time_ns: int,
        end_time_ns: int,
        topics: list[dict],
        chunks: list[dict],
        attachments: list[dict],
        metadata: list[dict],
    ) -> dict:
        """Assemble the deterministic inspect payload shared by both read paths."""
        return {
            "adapter": self.info.name,
            "path": str(path),
            "profile": header.profile,
            "library": header.library,
            "message_count": message_count,
            "schema_count": schema_count,
            "channel_count": channel_count,
            "chunk_count": chunk_count,
            "start_time_ns": start_time_ns,
            "end_time_ns": end_time_ns,
            "duration_ns": end_time_ns - start_time_ns,
            # Whether counts/ranges are summary-index-derived (True) or recovered
            # from a linear scan of an unindexed file (False, backlog 0018).
            "indexed": indexed,
            "topics": topics,
            "chunks": chunks,
            "attachments": attachments,
            "metadata": metadata,
        }
