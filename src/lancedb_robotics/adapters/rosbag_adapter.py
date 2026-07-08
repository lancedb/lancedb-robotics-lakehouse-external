"""ROS bag adapter: inspect and ingest ROS1 `.bag` / ROS2 sqlite `.db3` logs.

ROS bags are a different container, not a different canonical data model. This
adapter uses `rosbags` only for container access, then routes raw `ros1` and
`cdr` payload bytes through the same `PayloadDecoder` registry as MCAP ingest.
The dependency is optional and imported lazily so base installs can discover the
format and report an actionable "install the rosbag extra" message.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from mcap.records import Schema

from lancedb_robotics.adapters import AdapterError, AdapterInfo
from lancedb_robotics.adapters.decoders import PayloadDecoder, can_decode
from lancedb_robotics.sources import content_key_digest, file_checksum, recording_content_key
from lancedb_robotics.storage import is_object_store_uri, source_uri

_BAG_SUFFIX = ".bag"
_DB3_SUFFIX = ".db3"
_YAML_SUFFIXES = (".yaml", ".yml")
_ROSBAG2_ROOT_KEY = "rosbag2_bagfile_information"


@dataclass(frozen=True)
class RosBagSource:
    """Resolved local ROS bag source and stable content identity."""

    uri: str
    reader_paths: tuple[Path, ...]
    content_paths: tuple[Path, ...]
    checksum: str
    digest: str
    kind: str
    storage_identifier: str

    @property
    def input_uris(self) -> list[str]:
        return [str(path) for path in self.content_paths]


class RosBagAdapter:
    info = AdapterInfo(name="rosbag", format="rosbag", capabilities=("inspect", "ingest"))

    def inspect(self, path: str | Path, **_: Any) -> dict:
        """Return deterministic metadata about a ROS1 or ROS2 bag source."""
        AnyReader = _require_any_reader()
        source = self.source(path)
        try:
            with AnyReader(list(source.reader_paths)) as reader:
                return self._inspect_reader(reader, source)
        except AdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 - third-party read failures become adapter data.
            raise AdapterError(f"cannot read ROS bag {source.uri}: {exc}") from exc

    def ingest(self, path: str | Path, **_: Any) -> Iterator[dict]:
        """Yield one canonical message record per ROS bag message."""
        AnyReader = _require_any_reader()
        source = self.source(path)
        decoder = PayloadDecoder()
        per_topic_index: dict[str, int] = {}
        try:
            with AnyReader(list(source.reader_paths)) as reader:
                schema_by_connection = {
                    id(connection): self._schema(reader, connection)
                    for connection in reader.connections
                }
                for connection, timestamp, rawdata in reader.messages():
                    topic = connection.topic
                    sequence = per_topic_index.get(topic, 0)
                    per_topic_index[topic] = sequence + 1
                    message_encoding = self._message_encoding(reader, connection)
                    schema = schema_by_connection.get(id(connection))
                    result = decoder.decode(message_encoding, schema, bytes(rawdata))
                    yield {
                        "topic": topic,
                        "log_time_ns": timestamp,
                        "publish_time_ns": timestamp,
                        "sequence": sequence,
                        "message_encoding": message_encoding,
                        "schema_name": schema.name
                        if schema
                        else self._schema_name(reader, connection),
                        "schema_encoding": schema.encoding if schema else None,
                        "decode_status": result.status,
                        "decode_error": result.error,
                        "payload_json": result.payload_json,
                        "payload_blob": result.payload_blob,
                    }
        except AdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 - third-party read failures become adapter data.
            raise AdapterError(f"cannot read ROS bag {source.uri}: {exc}") from exc

    def source(self, path: str | Path) -> RosBagSource:
        """Resolve a ROS bag path and compute its path-independent content id."""
        if is_object_store_uri(path):
            raise AdapterError(
                "ROS bag adapter currently supports local filesystem paths only; "
                "stage object-store bags locally or convert to MCAP for ranged remote reads"
            )
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise AdapterError(f"no such ROS bag path: {resolved}")

        reader_paths, content_paths, storage = _resolve_paths(resolved)
        checksums = [file_checksum(path) for path in content_paths]
        if len(content_paths) == 1 and content_paths[0] == resolved:
            checksum = checksums[0]
            digest = content_key_digest(checksum)[:16]
            kind = "file"
        else:
            checksum = recording_content_key(checksums)
            digest = checksum[:16]
            kind = "recording"

        return RosBagSource(
            uri=source_uri(resolved),
            reader_paths=reader_paths,
            content_paths=content_paths,
            checksum=checksum,
            digest=digest,
            kind=kind,
            storage_identifier=storage,
        )

    def _inspect_reader(self, reader, source: RosBagSource) -> dict:
        per_topic = self._topic_skeleton(reader)
        message_count = 0
        start_time_ns: int | None = None
        end_time_ns: int | None = None
        for connection, timestamp, _rawdata in reader.messages():
            entry = per_topic.setdefault(connection.topic, self._topic_entry(reader, connection))
            entry["message_count"] += 1
            entry["start_time_ns"] = (
                timestamp
                if entry["start_time_ns"] is None
                else min(entry["start_time_ns"], timestamp)
            )
            entry["end_time_ns"] = (
                timestamp if entry["end_time_ns"] is None else max(entry["end_time_ns"], timestamp)
            )
            message_count += 1
            start_time_ns = timestamp if start_time_ns is None else min(start_time_ns, timestamp)
            end_time_ns = timestamp if end_time_ns is None else max(end_time_ns, timestamp)

        topics = sorted(per_topic.values(), key=lambda topic: topic["topic"])
        schema_keys = {
            (topic["schema_name"], topic["schema_encoding"])
            for topic in topics
            if topic["schema_name"] is not None
        }
        start = start_time_ns if start_time_ns is not None else 0
        end = end_time_ns if end_time_ns is not None else 0
        profile = "ros2" if reader.is2 else "ros1"
        return {
            "adapter": self.info.name,
            "path": source.uri,
            "profile": profile,
            "library": f"rosbags/{_rosbags_version()}",
            "message_count": message_count,
            "schema_count": len(schema_keys),
            "channel_count": len(reader.connections),
            "chunk_count": 0,
            "start_time_ns": start,
            "end_time_ns": end,
            "duration_ns": max(0, end - start),
            "indexed": True,
            "topics": topics,
            "chunks": [],
            "attachments": [],
            "metadata": [],
            "storage_identifier": source.storage_identifier,
            "source_files": source.input_uris,
        }

    def _topic_skeleton(self, reader) -> dict[str, dict]:
        per_topic: dict[str, dict] = {}
        for connection in reader.connections:
            entry = self._topic_entry(reader, connection)
            existing = per_topic.get(entry["topic"])
            if existing is None:
                per_topic[entry["topic"]] = entry
                continue
            for key in ("schema_name", "schema_encoding", "message_encoding"):
                if existing.get(key) is None and entry.get(key) is not None:
                    existing[key] = entry[key]
            existing["can_decode"] = existing["can_decode"] or entry["can_decode"]
        return per_topic

    def _topic_entry(self, reader, connection) -> dict:
        message_encoding = self._message_encoding(reader, connection)
        schema = self._schema(reader, connection)
        return {
            "topic": connection.topic,
            "message_encoding": message_encoding,
            "schema_name": schema.name if schema else self._schema_name(reader, connection),
            "schema_encoding": schema.encoding if schema else None,
            "message_count": 0,
            "start_time_ns": None,
            "end_time_ns": None,
            "can_decode": can_decode(message_encoding),
        }

    def _schema(self, reader, connection) -> Schema | None:
        schema_name = self._schema_name(reader, connection)
        schema_encoding = self._schema_encoding(reader, connection)
        data = _message_definition_bytes(connection)
        if schema_name is None or schema_encoding is None or data is None:
            return None
        return Schema(
            id=int(getattr(connection, "id", 0) or 0),
            name=schema_name,
            encoding=schema_encoding,
            data=data,
        )

    @staticmethod
    def _message_encoding(reader, connection) -> str:
        if not reader.is2:
            return "ros1"
        return str(getattr(connection.ext, "serialization_format", None) or "cdr")

    @staticmethod
    def _schema_encoding(reader, connection) -> str | None:
        if not reader.is2:
            return "ros1msg"
        # rosbag2 sqlite stores normally carry CDR bytes with `.msg` text.
        return "ros2msg" if getattr(connection, "msgdef", None) is not None else None

    @staticmethod
    def _schema_name(reader, connection) -> str | None:
        msgtype = getattr(connection, "msgtype", None)
        if msgtype is None:
            return None
        value = str(msgtype)
        return value.replace("/msg/", "/", 1) if not reader.is2 else value


def _require_any_reader():
    if importlib.util.find_spec("rosbags.highlevel") is None:
        raise AdapterError(
            "ROS bag support requires the optional 'rosbag' extra; install with "
            "`pip install 'lancedb-robotics[rosbag]'` or `uv sync --extra rosbag` "
            "and retry"
        )
    from rosbags.highlevel import AnyReader

    return AnyReader


def _rosbags_version() -> str:
    try:
        return importlib.metadata.version("rosbags")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _resolve_paths(path: Path) -> tuple[tuple[Path, ...], tuple[Path, ...], str]:
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == _BAG_SUFFIX:
            return (path,), (path,), "rosbag1"
        if suffix == _DB3_SUFFIX:
            return (path,), (path,), "sqlite3"
        if suffix in _YAML_SUFFIXES:
            files = _db3_paths_from_metadata(path)
            return (path.parent,), files, "sqlite3"
        raise AdapterError(f"unsupported ROS bag file extension: {path}")

    if path.is_dir():
        metadata = path / "metadata.yaml"
        if metadata.is_file():
            files = _db3_paths_from_metadata(metadata)
            return (path,), files, "sqlite3"
        db3s = tuple(sorted(p.resolve() for p in path.iterdir() if p.suffix.lower() == _DB3_SUFFIX))
        if db3s:
            return (path,), db3s, "sqlite3"
        bags = tuple(sorted(p.resolve() for p in path.iterdir() if p.suffix.lower() == _BAG_SUFFIX))
        if bags:
            return bags, bags, "rosbag1"
        raise AdapterError(f"no ROS bag files found in directory: {path}")

    raise AdapterError(f"unsupported ROS bag source (not a file or directory): {path}")


def _db3_paths_from_metadata(metadata_path: Path) -> tuple[Path, ...]:
    try:
        doc = yaml.safe_load(metadata_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise AdapterError(f"invalid rosbag2 metadata {metadata_path}: {exc}") from exc
    info = doc.get(_ROSBAG2_ROOT_KEY, doc) if isinstance(doc, dict) else {}
    if not isinstance(info, dict):
        raise AdapterError(f"unexpected rosbag2 metadata structure in {metadata_path}")

    storage = info.get("storage_identifier")
    if storage is not None and str(storage).lower() != "sqlite3":
        raise AdapterError(
            f"unsupported rosbag2 storage identifier {storage!r} in {metadata_path}; "
            "only sqlite3 `.db3` stores are supported by the ROS bag adapter"
        )

    names = _declared_file_names(info)
    root = metadata_path.parent
    if not names:
        names = [path.name for path in sorted(root.glob(f"*{_DB3_SUFFIX}"))]
    if not names:
        raise AdapterError(f"rosbag2 metadata lists no sqlite `.db3` files: {metadata_path}")

    paths: list[Path] = []
    for name in names:
        db3 = (root / name).resolve()
        if db3.suffix.lower() != _DB3_SUFFIX:
            raise AdapterError(f"rosbag2 metadata file is not sqlite `.db3`: {db3}")
        if not db3.is_file():
            raise AdapterError(f"rosbag2 metadata {metadata_path} lists missing file: {db3}")
        paths.append(db3)
    return tuple(paths)


def _declared_file_names(info: dict) -> list[str]:
    relative = info.get("relative_file_paths")
    if isinstance(relative, list) and relative:
        return [str(name) for name in relative]
    files = info.get("files")
    if isinstance(files, list) and files:
        names = [item.get("path") for item in files if isinstance(item, dict) and item.get("path")]
        if names:
            return [str(name) for name in names]
    return []


def _message_definition_bytes(connection) -> bytes | None:
    msgdef = getattr(connection, "msgdef", None)
    data = getattr(msgdef, "data", None)
    if data is None:
        return None
    return data if isinstance(data, bytes) else str(data).encode("utf-8")


__all__ = ["RosBagAdapter", "RosBagSource"]
