"""Split / multi-file (directory) recordings: one logical run from N shards.

A long robot recording is often *not one file*. ``rosbag2`` and other recorders
split a session into a *directory* of shards by size or duration::

    my_recording/
      metadata.yaml
      my_recording_0.mcap
      my_recording_1.mcap
      my_recording_2.mcap

``metadata.yaml`` records the storage identifier, the ordered shard list, and
the recording's overall time range. The shards together are **one logical run**
— a single continuous timeline split only for file-size management (backlog
0019).

This module resolves a path (single file, directory, or a ``metadata.yaml``)
into an ordered shard plan, merges per-shard inspection into one aggregate
report plus a shard inventory, and slices an export window that may span shard
boundaries into a single clip. The per-MCAP read/write stays in the adapter;
this layer only knows the rosbag2 directory *layout*.

Shard order is canonical — the ``metadata.yaml`` ``relative_file_paths`` list
when present, else a lexicographic sort of the ``*.mcap`` filenames — so the
operating system's directory-listing order never changes the resolved order,
and therefore never changes the content-addressed ``run_id`` derived from it.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from lancedb_robotics.adapters import AdapterError, CorruptMcapError, get_adapter
from lancedb_robotics.sources import file_checksum
from lancedb_robotics.storage import is_object_store_uri

METADATA_FILENAME = "metadata.yaml"
_YAML_SUFFIXES = (".yaml", ".yml")
_MCAP_SUFFIX = ".mcap"

# rosbag2 metadata.yaml nests everything under this top-level key.
_ROSBAG2_ROOT_KEY = "rosbag2_bagfile_information"


@dataclass(frozen=True)
class Shard:
    """One shard of a split recording: its path and content checksum."""

    path: Path
    checksum: str

    @property
    def uri(self) -> str:
        return str(self.path)


@dataclass(frozen=True)
class ShardPlan:
    """An ordered resolution of an ingest/inspect source into shard paths.

    Cheap to build (no checksum IO): it lists the ordered shard ``paths`` and any
    declared time range / storage identifier parsed from ``metadata.yaml``.
    ``is_split`` is False only for a bare single-file source, whose behavior is
    unchanged from before backlog 0019.
    """

    is_split: bool
    root: Path
    paths: tuple[Path, ...]
    metadata_path: Path | None = None
    storage_identifier: str | None = None
    declared_start_ns: int | None = None
    declared_end_ns: int | None = None
    declared_message_count: int | None = None

    @property
    def uri(self) -> str:
        return str(self.root)


@dataclass(frozen=True)
class Recording:
    """A resolved recording with per-shard checksums (the ingest view)."""

    plan: ShardPlan
    shards: tuple[Shard, ...]

    @property
    def is_split(self) -> bool:
        return self.plan.is_split

    @property
    def root(self) -> Path:
        return self.plan.root

    @property
    def uri(self) -> str:
        return self.plan.uri

    @property
    def checksums(self) -> list[str]:
        """Ordered shard checksums — the material for the content-addressed id."""
        return [shard.checksum for shard in self.shards]


def resolve_shards(path: str | Path) -> ShardPlan:
    """Resolve ``path`` into an ordered shard plan without reading shard bytes.

    Accepts a single ``*.mcap`` file (``is_split=False``), a ``metadata.yaml``
    file, or a directory. A directory with a ``metadata.yaml`` uses its declared
    shard order and time range; a bare directory falls back to a lexicographic
    sort of its ``*.mcap`` files plus a scan (no declared range).

    Raises :class:`AdapterError` for a missing path, an empty directory, a
    non-mcap storage identifier, or a declared shard that is missing on disk.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise AdapterError(f"no such path: {resolved}")

    if resolved.is_file():
        if resolved.suffix.lower() in _YAML_SUFFIXES or resolved.name == METADATA_FILENAME:
            return _plan_from_metadata(resolved)
        # A bare single MCAP file: unchanged single-file behavior.
        return ShardPlan(is_split=False, root=resolved, paths=(resolved,))

    if resolved.is_dir():
        metadata_path = resolved / METADATA_FILENAME
        if metadata_path.is_file():
            return _plan_from_metadata(metadata_path)
        return _plan_from_scan(resolved)

    raise AdapterError(f"unsupported source (not a file or directory): {resolved}")


def _plan_from_metadata(metadata_path: Path) -> ShardPlan:
    """Build a shard plan from a rosbag2-style ``metadata.yaml``."""
    root = metadata_path.parent
    try:
        doc = yaml.safe_load(metadata_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise AdapterError(f"invalid recording metadata {metadata_path}: {exc}") from exc
    info = doc.get(_ROSBAG2_ROOT_KEY, doc) if isinstance(doc, dict) else {}
    if not isinstance(info, dict):
        raise AdapterError(f"unexpected recording metadata structure in {metadata_path}")

    storage = info.get("storage_identifier")
    if storage is not None and str(storage).lower() != "mcap":
        raise AdapterError(
            f"unsupported storage identifier {storage!r} in {metadata_path}; "
            "only mcap recordings are supported"
        )

    relative = _declared_shard_names(info)
    if not relative:
        # A metadata.yaml without a usable file list: fall back to scanning its
        # directory rather than failing, so a hand-written manifest still works.
        return _plan_from_scan(root, metadata_path=metadata_path)

    paths: list[Path] = []
    for name in relative:
        shard = (root / name).resolve()
        if not shard.is_file():
            raise AdapterError(f"recording metadata {metadata_path} lists missing shard: {shard}")
        if shard.suffix.lower() != _MCAP_SUFFIX:
            raise AdapterError(f"recording shard is not an mcap file: {shard}")
        paths.append(shard)

    start_ns, end_ns = _declared_time_range(info)
    return ShardPlan(
        is_split=True,
        root=root,
        paths=tuple(paths),
        metadata_path=metadata_path,
        storage_identifier=str(storage) if storage is not None else None,
        declared_start_ns=start_ns,
        declared_end_ns=end_ns,
        declared_message_count=_as_int(info.get("message_count")),
    )


def _plan_from_scan(directory: Path, *, metadata_path: Path | None = None) -> ShardPlan:
    """Build a shard plan from a bare directory of ``*.mcap`` files (name order)."""
    paths = sorted(
        (p.resolve() for p in directory.iterdir() if p.suffix.lower() == _MCAP_SUFFIX),
        key=lambda p: p.name,
    )
    if not paths:
        raise AdapterError(f"no mcap shards found in directory: {directory}")
    return ShardPlan(
        is_split=True,
        root=directory.resolve(),
        paths=tuple(paths),
        metadata_path=metadata_path,
    )


def _declared_shard_names(info: dict) -> list[str]:
    """Ordered shard file names from a rosbag2 metadata dict, if any."""
    relative = info.get("relative_file_paths")
    if isinstance(relative, list) and relative:
        return [str(name) for name in relative]
    # Some writers carry per-file records instead of a flat list.
    files = info.get("files")
    if isinstance(files, list) and files:
        names = [f.get("path") for f in files if isinstance(f, dict) and f.get("path")]
        if names:
            return [str(name) for name in names]
    return []


def _declared_time_range(info: dict) -> tuple[int | None, int | None]:
    """Declared (start_ns, end_ns) from a rosbag2 metadata dict, if present."""
    starting = info.get("starting_time")
    start_ns = (
        _as_int(starting.get("nanoseconds_since_epoch")) if isinstance(starting, dict) else None
    )
    duration = info.get("duration")
    duration_ns = _as_int(duration.get("nanoseconds")) if isinstance(duration, dict) else None
    end_ns = start_ns + duration_ns if start_ns is not None and duration_ns is not None else None
    return start_ns, end_ns


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def resolve_recording(path: str | Path) -> Recording:
    """Resolve ``path`` into a :class:`Recording`, checksumming each shard.

    The shard checksums (in canonical order) are the material for the
    content-addressed ``run_id``/``source_id``; see
    :func:`lancedb_robotics.sources.recording_content_key`.
    """
    plan = resolve_shards(path)
    shards = tuple(Shard(path=p, checksum=file_checksum(p)) for p in plan.paths)
    return Recording(plan=plan, shards=shards)


def iter_shard_messages(recording: Recording, *, validate_crcs: bool = True) -> Iterator[dict]:
    """Yield one message record per shard message, in shard then log-time order.

    Each record is the adapter's per-shard message dict augmented with
    ``shard_index`` and ``shard_uri`` (the specific shard the bytes live in) and
    a recording-global per-topic ``sequence`` that is continuous across shard
    boundaries. A :class:`CorruptMcapError` from any shard propagates *after* its
    readable prefix has been yielded, carrying the recording-wide recovered count
    so the caller can quarantine the run.
    """
    adapter = get_adapter("mcap")
    seq_base: dict[str, int] = {}
    recovered = 0
    for index, shard in enumerate(recording.shards):
        shard_topic_count: dict[str, int] = {}
        try:
            for message in adapter.ingest(shard.path, validate_crcs=validate_crcs):
                topic = message["topic"]
                message["sequence"] = seq_base.get(topic, 0) + message["sequence"]
                message["shard_index"] = index
                message["shard_uri"] = shard.uri
                shard_topic_count[topic] = shard_topic_count.get(topic, 0) + 1
                recovered += 1
                yield message
        except CorruptMcapError as exc:
            # Roll the consumed prefix into the base so any later shard would keep
            # numbering forward, then re-raise with the recording-wide tally.
            for topic, count in shard_topic_count.items():
                seq_base[topic] = seq_base.get(topic, 0) + count
            raise CorruptMcapError(
                f"recording shard {index} ({shard.uri}) damaged: {exc}",
                status=exc.status,
                reason=exc.reason,
                recovered=recovered,
            ) from exc
        for topic, count in shard_topic_count.items():
            seq_base[topic] = seq_base.get(topic, 0) + count


def inspect_recording(path: str | Path) -> dict:
    """Inspect a split recording: merged aggregate report plus a shard inventory.

    Aggregates per-shard inspection into a single report whose ``message_count``,
    per-topic counts/ranges, and full time range match a single-file inspect of
    the concatenated shards. Adds ``is_split``/``shard_count``, a ``shards``
    inventory, and a ``gaps`` list flagging any non-contiguous or overlapping
    shard boundaries (the timeline should be continuous; gaps/overlaps are
    surfaced, not errors). Per-shard ``chunk_count`` sums, but chunk *offsets* are
    shard-local, so the merged report carries none.

    A summary-less shard is inspected by scan like any single file; a
    :class:`CorruptMcapError` shard is listed with ``readable: False`` and
    excluded from the aggregates rather than aborting the whole recording.
    """
    plan = resolve_shards(path)
    adapter = get_adapter("mcap")

    shard_entries: list[dict] = []
    per_topic: dict[str, dict] = {}
    message_count = 0
    chunk_count = 0
    start_ns: int | None = None
    end_ns: int | None = None
    profile = ""
    library = ""
    attachments: list[dict] = []
    metadata: list[dict] = []
    all_indexed = True

    for index, shard_path in enumerate(plan.paths):
        entry: dict = {
            "index": index,
            "path": str(shard_path),
            "name": shard_path.name,
            "size_bytes": shard_path.stat().st_size,
        }
        try:
            report = adapter.inspect(shard_path)
        except CorruptMcapError as exc:
            entry.update(readable=False, error=str(exc))
            shard_entries.append(entry)
            continue
        entry.update(
            readable=True,
            message_count=report["message_count"],
            start_time_ns=report["start_time_ns"],
            end_time_ns=report["end_time_ns"],
            indexed=report["indexed"],
        )
        shard_entries.append(entry)

        if index == 0:
            profile, library = report["profile"], report["library"]
        all_indexed = all_indexed and report["indexed"]
        message_count += report["message_count"]
        chunk_count += report["chunk_count"]
        if report["message_count"]:
            start_ns = report["start_time_ns"] if start_ns is None else min(
                start_ns, report["start_time_ns"]
            )
            end_ns = report["end_time_ns"] if end_ns is None else max(
                end_ns, report["end_time_ns"]
            )
        for topic in report["topics"]:
            _merge_topic(per_topic, topic)
        attachments.extend(report.get("attachments", []))
        metadata.extend(report.get("metadata", []))

    topics = sorted(per_topic.values(), key=lambda t: t["topic"])
    schema_names = {t["schema_name"] for t in topics if t["schema_name"] is not None}

    # Declared range (metadata.yaml) wins for the headline range when present; it
    # is the recording's own statement of its span. Counts/topics stay scan-derived.
    if plan.declared_start_ns is not None and plan.declared_end_ns is not None:
        start_ns, end_ns = plan.declared_start_ns, plan.declared_end_ns
    start_ns = start_ns or 0
    end_ns = end_ns or 0

    return {
        "adapter": adapter.info.name,
        "path": str(plan.root),
        "is_split": True,
        "shard_count": len(plan.paths),
        "profile": profile,
        "library": library,
        "message_count": message_count,
        "schema_count": len(schema_names),
        "channel_count": len(topics),
        "chunk_count": chunk_count,
        "start_time_ns": start_ns,
        "end_time_ns": end_ns,
        "duration_ns": end_ns - start_ns,
        "indexed": all_indexed,
        "topics": topics,
        # Chunk offsets are shard-local; the merged report cannot carry a single
        # offset space, so it reports none (each shard keeps its own under inspect).
        "chunks": [],
        "attachments": _sorted_attachments(attachments),
        "metadata": _sorted_metadata(metadata),
        "shards": shard_entries,
        "gaps": _detect_gaps(shard_entries),
        "storage_identifier": plan.storage_identifier,
        "metadata_path": str(plan.metadata_path) if plan.metadata_path else None,
    }


def _merge_topic(per_topic: dict[str, dict], topic: dict) -> None:
    """Fold one shard's topic entry into the recording-wide merge."""
    name = topic["topic"]
    existing = per_topic.get(name)
    if existing is None:
        per_topic[name] = dict(topic)
        return
    existing["message_count"] += topic["message_count"]
    for key in ("start_time_ns", "end_time_ns"):
        new = topic[key]
        if new is None:
            continue
        cur = existing[key]
        existing[key] = new if cur is None else (min(cur, new) if "start" in key else max(cur, new))
    # A later shard may be the first to carry the schema/decode info for a topic.
    for key in ("schema_name", "schema_encoding", "message_encoding"):
        if existing.get(key) is None and topic.get(key) is not None:
            existing[key] = topic[key]
    existing["can_decode"] = existing.get("can_decode") or topic.get("can_decode", False)


def _sorted_attachments(attachments: list[dict]) -> list[dict]:
    return sorted(attachments, key=lambda a: (a.get("log_time_ns") or 0, a.get("name") or ""))


def _sorted_metadata(metadata: list[dict]) -> list[dict]:
    return sorted(metadata, key=lambda m: m.get("name") or "")


def _detect_gaps(shard_entries: list[dict]) -> list[dict]:
    """Flag time gaps/overlaps between consecutive readable shards.

    Shards of one recording should tile a continuous timeline: each shard begins
    at or after the previous shard's end. A ``gap`` (next start strictly after the
    previous end) or an ``overlap`` (next start at or before the previous end) is
    reported for downstream awareness; neither is an error.
    """
    readable = [e for e in shard_entries if e.get("readable") and e.get("message_count")]
    gaps: list[dict] = []
    for prev, nxt in zip(readable, readable[1:], strict=False):
        prev_end, next_start = prev["end_time_ns"], nxt["start_time_ns"]
        if next_start > prev_end:
            gaps.append(
                {
                    "after_shard": prev["index"],
                    "before_shard": nxt["index"],
                    "kind": "gap",
                    "delta_ns": next_start - prev_end,
                }
            )
        elif next_start <= prev_end:
            gaps.append(
                {
                    "after_shard": prev["index"],
                    "before_shard": nxt["index"],
                    "kind": "overlap",
                    "delta_ns": prev_end - next_start,
                }
            )
    return gaps


def inspect_source(path: str | Path) -> dict:
    """Inspect any ingest source: single file (adapter) or split recording (merge).

    The CLI's single entry point: a directory or ``metadata.yaml`` yields the
    merged split report; anything else is a plain single-file inspect.
    """
    if is_object_store_uri(path):
        return get_adapter("mcap").inspect(path)
    plan = resolve_shards(path)
    if plan.is_split:
        return inspect_recording(path)
    return get_adapter("mcap").inspect(plan.paths[0])


def export_window(
    paths: Sequence[str | Path],
    *,
    start_time_ns: int,
    end_time_ns: int,
    out_path: str | Path,
    topics: tuple[str, ...] | list[str] | None = None,
    compression: str | None = None,
) -> dict:
    """Write one MCAP clip of ``[start, end]`` (inclusive) merged across ``paths``.

    Reads each shard in order and copies every in-window message into a single
    output writer, so a time window that spans a shard boundary becomes one
    correct clip. Schemas and channels are de-duplicated across shards by
    ``(name, encoding, data)`` and ``(topic, message_encoding)`` — shards of one
    recording share these, so the merged clip carries each once. A single-path
    ``paths`` is just the single-file export.

    Raises the adapter's read-error taxonomy if any shard cannot be read.
    """
    from mcap.exceptions import UnsupportedCompressionError
    from mcap.reader import make_reader
    from mcap.writer import Writer

    from lancedb_robotics.adapters.mcap_adapter import (
        _COMPRESSION_BY_NAME,
        _READ_ERRORS,
        _classify_read_error,
        _codec_error,
    )

    writer_kwargs: dict = {}
    if compression is not None:
        try:
            writer_kwargs["compression"] = _COMPRESSION_BY_NAME[compression]
        except KeyError:
            raise AdapterError(
                f"unknown compression {compression!r}; "
                f"expected one of {sorted(_COMPRESSION_BY_NAME)}"
            ) from None

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    topic_filter = set(topics) if topics else None
    written_topics: set[str] = set()
    count = 0

    shard_paths = [Path(p) for p in paths]
    for shard_path in shard_paths:
        if not shard_path.is_file():
            raise AdapterError(f"no such file: {shard_path}")

    schema_keys: dict[tuple[str, str, bytes], int] = {}
    channel_keys: dict[tuple[str, str, int], int] = {}
    try:
        with out.open("wb") as sink:
            writer = Writer(sink, **writer_kwargs)
            writer.start()
            for shard_path in shard_paths:
                with shard_path.open("rb") as source:
                    reader = make_reader(source)
                    for schema, channel, message in reader.iter_messages(log_time_order=True):
                        if not (start_time_ns <= message.log_time <= end_time_ns):
                            continue
                        if topic_filter is not None and channel.topic not in topic_filter:
                            continue
                        out_schema_id = _ensure_schema(writer, schema, schema_keys)
                        out_channel_id = _ensure_channel(
                            writer, channel, out_schema_id, channel_keys
                        )
                        writer.add_message(
                            channel_id=out_channel_id,
                            log_time=message.log_time,
                            data=message.data,
                            publish_time=message.publish_time,
                            sequence=message.sequence,
                        )
                        written_topics.add(channel.topic)
                        count += 1
            writer.finish()
    except UnsupportedCompressionError as exc:
        raise _codec_error(exc, shard_paths[0]) from exc
    except _READ_ERRORS as exc:
        raise _classify_read_error(exc, shard_paths[0]) from exc

    return {
        "out_path": str(out),
        "message_count": count,
        "topics": tuple(sorted(written_topics)),
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
    }


def _ensure_schema(writer, schema, schema_keys: dict) -> int:
    """Register ``schema`` in ``writer`` once, keyed by content (cross-shard safe)."""
    if schema is None:
        return 0
    key = (schema.name, schema.encoding, schema.data)
    out_id = schema_keys.get(key)
    if out_id is None:
        out_id = writer.register_schema(name=schema.name, encoding=schema.encoding, data=schema.data)
        schema_keys[key] = out_id
    return out_id


def _ensure_channel(writer, channel, schema_id: int, channel_keys: dict) -> int:
    """Register ``channel`` in ``writer`` once, keyed by (topic, encoding, schema)."""
    key = (channel.topic, channel.message_encoding, schema_id)
    out_id = channel_keys.get(key)
    if out_id is None:
        out_id = writer.register_channel(
            topic=channel.topic,
            message_encoding=channel.message_encoding,
            schema_id=schema_id,
            metadata=dict(channel.metadata or {}),
        )
        channel_keys[key] = out_id
    return out_id
