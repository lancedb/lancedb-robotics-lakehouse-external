"""RLDS / TensorFlow Datasets source adapter.

The adapter reads an already-prepared TFDS builder directory (local or
``gs://``), keeps TensorFlow and TFDS behind the ``tfds`` optional extra, and
normalizes one RLDS episode/step at a time.  It deliberately does not use
``tfds.as_numpy`` over a complete dataset: nested RLDS ``steps`` datasets are
consumed incrementally and the read configuration disables prefetch while
interleaving one shard at a time.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import struct
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lancedb_robotics.adapters import AdapterError, AdapterInfo
from lancedb_robotics.rlds_contract import (
    RLDS_ACTION_KEY,
    RLDS_DISCOUNT_KEY,
    RLDS_IS_FIRST_KEY,
    RLDS_IS_LAST_KEY,
    RLDS_IS_TERMINAL_KEY,
    RLDS_OBSERVATION_KEY,
    RLDS_REWARD_KEY,
    RLDS_STEPS_KEY,
)
from lancedb_robotics.storage import (
    StorageConfigError,
    is_object_store_uri,
    join_uri,
    list_uri,
    open_binary_uri,
    read_text_uri,
    source_uri,
)

_INSTALL_HINT = "lancedb-robotics[tfds] (or lancedb-robotics[rlds] for native conformance)"
_DATASET_INFO = "dataset_info.json"
_BLOB_MAGIC = b"LRRLDSP1"
_BLOB_FORMAT = "rlds-observation-bundle-v1"
_BLOB_THRESHOLD_BYTES = 4096
_IMAGE_HINTS = ("image", "rgb", "depth", "camera", "pixels")
_STATE_KEYS = ("state", "robot_state", "proprio", "joint_state")


@dataclass(frozen=True)
class RldsFieldMapping:
    """Format-specific source-key mapping used by the RLDS adapter.

    Keys use dotted lookup.  ``state_key`` is first resolved below the RLDS
    ``observation`` mapping, then from the complete step.  ``language_key`` is
    resolved from the step first and then from ``observation``.  A missing
    timestamp is derived from ``fps`` when supplied, otherwise from the stable
    step ordinal (ordering semantics only; no physical rate is invented).
    """

    state_key: str | None = None
    action_key: str = RLDS_ACTION_KEY
    language_key: str = "observation.natural_language_instruction"
    timestamp_key: str | None = None
    fps: float | None = None

    def __post_init__(self) -> None:
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be > 0 when provided, got {self.fps}")
        for name in ("action_key", "language_key"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class RldsSource:
    """Resolved prepared-TFDS directory and portable content identity."""

    uri: str
    checksum: str
    digest: str
    input_uris: tuple[str, ...]
    artifact_uris: tuple[str, ...]
    relative_files: tuple[str, ...]
    dataset_info: dict[str, Any]
    kind: str = "dataset"
    storage_identifier: str = "rlds-tfds"
    identity_kind: str = "content-sha256"


@dataclass
class RldsEpisode:
    """One RLDS episode whose nested steps remain a lazy iterator."""

    split: str
    source_episode_index: int
    shard_index: int
    shard_episode_index: int
    shard_uri: str
    tfds_id: str | None
    metadata: dict[str, Any]
    steps: Iterator[dict[str, Any]]


class RldsAdapter:
    """Read prepared TFDS/RLDS datasets as bounded episode/step streams."""

    info = AdapterInfo(name="rlds", format="rlds", capabilities=("inspect", "ingest"))

    def availability(self) -> dict[str, Any]:
        missing = [
            module
            for module in ("tensorflow", "tensorflow_datasets")
            if not _module_available(module)
        ]
        return {
            "available": not missing,
            "modules": ["tensorflow", "tensorflow_datasets"],
            "missing": missing,
            "install": _INSTALL_HINT,
        }

    def source(
        self,
        source: str | Path | RldsSource,
        *,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> RldsSource:
        """Resolve and content-hash a prepared TFDS directory.

        Every artifact is hashed through a fixed-size buffer.  Identity is
        therefore independent of the local/object-store path and peak memory is
        independent of shard size.
        """
        if isinstance(source, RldsSource):
            return source
        value, dataset_info = _prepared_source_metadata(
            source,
            storage_options=storage_options,
            auth_ref=auth_ref,
        )
        info_uri = join_uri(value, _DATASET_INFO)

        try:
            files = tuple(
                uri
                for uri in list_uri(
                    value,
                    storage_options=storage_options,
                    auth_ref=auth_ref,
                )
                if _include_dataset_artifact(value, uri)
            )
        except StorageConfigError as exc:
            raise AdapterError(f"cannot list RLDS/TFDS dataset {value}: {exc}") from exc
        if info_uri not in files:
            files = tuple(sorted((*files, info_uri)))
        if not any("tfrecord" in _relative_file(value, uri).lower() for uri in files):
            raise AdapterError(
                f"prepared TFDS dataset {value} contains no TFRecord shards; "
                "pass the generated builder directory, not the download/cache root"
            )

        aggregate = hashlib.sha256()
        relative_files: list[str] = []
        for uri in sorted(files, key=lambda item: _relative_file(value, item)):
            relative = _relative_file(value, uri)
            relative_files.append(relative)
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            try:
                with open_binary_uri(
                    uri,
                    storage_options=storage_options,
                    auth_ref=auth_ref,
                ) as stream:
                    file_hash = hashlib.sha256()
                    while True:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        file_hash.update(chunk)
            except StorageConfigError as exc:
                raise AdapterError(f"cannot hash RLDS/TFDS artifact {uri}: {exc}") from exc
            aggregate.update(file_hash.digest())
            aggregate.update(b"\0")

        hexdigest = aggregate.hexdigest()
        return RldsSource(
            uri=value,
            checksum=f"sha256:{hexdigest}",
            # 128 bits keeps generated identifiers compact while retaining a
            # collision margin appropriate for corpus-scale source catalogs.
            digest=hexdigest[:32],
            # Keep canonical source/transform rows bounded even when a TFDS
            # dataset has thousands of shards.  Exact artifacts remain on the
            # runtime source manifest and each observation records its shard.
            input_uris=(value,),
            artifact_uris=tuple(sorted(files, key=lambda item: _relative_file(value, item))),
            relative_files=tuple(relative_files),
            dataset_info=dataset_info,
        )

    def inspect(
        self,
        source: str | Path | RldsSource,
        *,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Describe splits/features without decoding episode payloads."""
        tfds = _load_tfds()
        _validate_tfds_source_request(source, storage_options)
        if isinstance(source, RldsSource):
            source_uri_value = source.uri
            dataset_info = source.dataset_info
            identity = {
                "kind": source.identity_kind,
                "checksum": source.checksum,
                "artifact_count": len(source.artifact_uris),
            }
        else:
            source_uri_value, dataset_info = _prepared_source_metadata(
                source,
                storage_options=storage_options,
                auth_ref=auth_ref,
            )
            identity = {
                "kind": "metadata-only",
                "checksum": None,
                "artifact_count": None,
            }
        builder = _builder_from_directory(tfds, _tfds_uri(source_uri_value))
        split_rows = _split_summaries(builder)
        episode_count = sum(int(row["episode_count"]) for row in split_rows)
        shard_count = sum(int(row["shard_count"]) for row in split_rows)
        features = _feature_summary(getattr(getattr(builder, "info", None), "features", None))
        return {
            "adapter": self.info.name,
            "path": source_uri_value,
            "profile": "rlds-tfds",
            "library": f"tensorflow-datasets/{_package_version(tfds)}",
            "dataset_name": _builder_name(builder, dataset_info),
            "dataset_version": _builder_version(builder, dataset_info),
            "message_count": 0,
            "episode_count": episode_count,
            "shard_count": shard_count,
            "chunk_count": shard_count,
            "channel_count": 1,
            "start_time_ns": 0,
            "end_time_ns": 0,
            "duration_ns": 0,
            "indexed": True,
            "splits": split_rows,
            "features": features,
            "topics": [
                {
                    "topic": "rlds.steps",
                    "message_encoding": "tfrecord",
                    "schema_name": "rlds-step",
                    "schema_encoding": "tfds-feature",
                    "message_count": 0,
                    "start_time_ns": None,
                    "end_time_ns": None,
                    "can_decode": True,
                }
            ],
            "source_identity": identity,
            "attachments": [],
            "metadata": [],
        }

    def ingest(
        self,
        source: str | Path | RldsSource,
        *,
        splits: Sequence[str] | None = None,
        mapping: RldsFieldMapping | None = None,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
        **_: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield normalized step dictionaries for adapter conformance callers."""
        for episode in self.iter_episodes(
            source,
            splits=splits,
            mapping=mapping,
            storage_options=storage_options,
            auth_ref=auth_ref,
        ):
            yield from episode.steps

    def iter_episodes(
        self,
        source: str | Path | RldsSource,
        *,
        splits: Sequence[str] | None = None,
        mapping: RldsFieldMapping | None = None,
        storage_options: Mapping[str, Any] | None = None,
        auth_ref: str | None = None,
    ) -> Iterator[RldsEpisode]:
        """Yield episodes while keeping both shard and nested-step reads lazy."""
        tfds = _load_tfds()
        _validate_tfds_source_request(source, storage_options)
        resolved = self.source(source, storage_options=storage_options, auth_ref=auth_ref)
        builder = _builder_from_directory(tfds, _tfds_uri(resolved.uri))
        selected_splits = _select_splits(builder, splits)
        field_mapping = mapping or RldsFieldMapping()
        read_config = _bounded_read_config(tfds)
        source_episode_index = 0
        for split in selected_splits:
            split_info = getattr(builder.info, "splits", {})[split]
            shard_count = int(getattr(split_info, "num_shards", 0) or 0)
            if shard_count < 1:
                shard_count = len(getattr(split_info, "shard_lengths", ()) or ())
            if shard_count < 1:
                raise AdapterError(f"TFDS split {split!r} declares no physical shards")
            for shard_index in range(shard_count):
                instruction = _shard_instruction(tfds, split, shard_index)
                shard_uri = _shard_uri(resolved, split_info, shard_index)
                dataset = _builder_dataset(builder, instruction, read_config, split_name=split)
                for shard_episode_index, raw_episode in enumerate(dataset):
                    episode = _episode_mapping(raw_episode, split=split)
                    raw_steps = episode.pop(RLDS_STEPS_KEY)
                    tfds_id = _optional_text(episode.pop("tfds_id", None))
                    metadata = _json_safe(episode)
                    if bool(metadata.get("invalid", False)):
                        continue
                    yield RldsEpisode(
                        split=split,
                        source_episode_index=source_episode_index,
                        shard_index=shard_index,
                        shard_episode_index=shard_episode_index,
                        shard_uri=shard_uri,
                        tfds_id=tfds_id,
                        metadata=metadata,
                        steps=_normalized_steps(raw_steps, mapping=field_mapping),
                    )
                    source_episode_index += 1


def decode_rlds_observation_bundle(payload: bytes) -> dict[str, Any]:
    """Decode the bounded binary bundle used for large RLDS observation fields."""
    if len(payload) < len(_BLOB_MAGIC) + 8 or not payload.startswith(_BLOB_MAGIC):
        raise ValueError("payload is not an RLDS observation bundle")
    header_size = struct.unpack(">Q", payload[len(_BLOB_MAGIC) : len(_BLOB_MAGIC) + 8])[0]
    header_start = len(_BLOB_MAGIC) + 8
    header_end = header_start + header_size
    if header_end > len(payload):
        raise ValueError("RLDS observation bundle header is truncated")
    try:
        header = json.loads(payload[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid RLDS observation bundle header: {exc}") from exc
    body = payload[header_end:]
    fields: dict[str, bytes] = {}
    for entry in header.get("fields") or []:
        offset = int(entry["offset"])
        length = int(entry["length"])
        if offset < 0 or length < 0 or offset + length > len(body):
            raise ValueError("RLDS observation bundle field range is invalid")
        fields[str(entry["path"])] = body[offset : offset + length]
    return {"format": header.get("format"), "fields": fields, "metadata": header["fields"]}


def _prepared_source_metadata(
    source: str | Path,
    *,
    storage_options: Mapping[str, Any] | None,
    auth_ref: str | None,
) -> tuple[str, dict[str, Any]]:
    """Resolve a prepared directory and read only its TFDS metadata file."""
    value = str(source).rstrip("/")
    if not value:
        raise AdapterError("RLDS/TFDS source path must not be empty")
    if not is_object_store_uri(value):
        root = Path(value).expanduser()
        if not root.exists():
            raise AdapterError(f"no such RLDS/TFDS dataset directory: {root}")
        if not root.is_dir():
            raise AdapterError(f"RLDS/TFDS source must be a directory, got file: {root}")
        value = source_uri(root.resolve())

    info_uri = join_uri(value, _DATASET_INFO)
    try:
        raw_info = read_text_uri(
            info_uri,
            storage_options=storage_options,
            auth_ref=auth_ref,
        )
    except StorageConfigError as exc:
        raise AdapterError(
            f"not a prepared TFDS dataset: {value} is missing/read-failed "
            f"{_DATASET_INFO} ({exc})"
        ) from exc
    try:
        dataset_info = json.loads(raw_info)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"invalid {_DATASET_INFO} in {value}: {exc}") from exc
    if not isinstance(dataset_info, dict):
        raise AdapterError(f"invalid {_DATASET_INFO} in {value}: expected a JSON object")
    if dataset_info.get("format_version") == "rlds-tfds-style-v0":
        raise AdapterError(
            "this is the repository's Parquet `rlds-tfds-style-v0` export, not a "
            "prepared TensorFlow Datasets version directory; ingest expects TFDS "
            "dataset_info/features metadata plus TFRecord shards"
        )
    return value, dataset_info


def _load_tfds():
    try:
        return importlib.import_module("tensorflow_datasets")
    except Exception as exc:  # noqa: BLE001 - native optional imports fail by host.
        raise AdapterError(
            "RLDS/TFDS ingest requires TensorFlow Datasets and TensorFlow; "
            f"install `{_INSTALL_HINT}` on a supported host and retry "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return name in sys.modules


def _builder_from_directory(tfds, uri: str):
    try:
        return tfds.builder_from_directory(builder_dir=uri)
    except TypeError:
        try:
            return tfds.builder_from_directory(uri)
        except Exception as exc:  # noqa: BLE001 - normalize provider/library failures.
            raise AdapterError(f"cannot open prepared TFDS dataset {uri}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - normalize provider/library failures.
        raise AdapterError(f"cannot open prepared TFDS dataset {uri}: {exc}") from exc


def _bounded_read_config(tfds):
    read_config_type = getattr(tfds, "ReadConfig", None)
    if read_config_type is None:
        raise AdapterError("installed tensorflow-datasets has no public ReadConfig API")
    candidates = {
        "add_tfds_id": True,
        "skip_prefetch": True,
        "try_autocache": False,
        "interleave_cycle_length": 1,
        "interleave_block_length": 1,
        "num_parallel_calls_for_decode": 1,
        "num_parallel_calls_for_interleave_files": 1,
    }
    try:
        parameters = inspect.signature(read_config_type).parameters
    except (TypeError, ValueError):
        parameters = candidates
    kwargs = {key: value for key, value in candidates.items() if key in parameters}
    try:
        return read_config_type(**kwargs)
    except TypeError as exc:
        raise AdapterError(
            "installed tensorflow-datasets cannot configure bounded RLDS reads "
            f"with ReadConfig ({exc})"
        ) from exc


def _builder_dataset(builder, split: Any, read_config, *, split_name: str):
    try:
        return builder.as_dataset(split=split, shuffle_files=False, read_config=read_config)
    except Exception as exc:  # noqa: BLE001 - normalize TensorFlow/provider failures.
        raise AdapterError(f"cannot stream TFDS split {split_name!r}: {exc}") from exc


def _shard_instruction(tfds, split: str, shard_index: int):
    core = getattr(tfds, "core", None)
    instruction_type = getattr(core, "ReadInstruction", None)
    if instruction_type is None:
        raise AdapterError("installed tensorflow-datasets has no public ReadInstruction API")
    try:
        return instruction_type(
            split,
            from_=shard_index,
            to=shard_index + 1,
            unit="shard",
        )
    except Exception as exc:  # noqa: BLE001 - normalize TFDS version drift.
        raise AdapterError(f"cannot select TFDS shard {split}[{shard_index}]: {exc}") from exc


def _shard_uri(source: RldsSource, split_info: Any, shard_index: int) -> str:
    filepaths = tuple(getattr(split_info, "filepaths", ()) or ())
    if shard_index < len(filepaths):
        value = str(filepaths[shard_index])
        if is_object_store_uri(value) or Path(value).is_absolute():
            return value
        return join_uri(source.uri, value)
    instructions = tuple(getattr(split_info, "file_instructions", ()) or ())
    if shard_index < len(instructions):
        value = str(getattr(instructions[shard_index], "filename", "") or "")
        if value:
            if is_object_store_uri(value) or Path(value).is_absolute():
                return value
            return join_uri(source.uri, value)
    matches = [uri for uri in source.artifact_uris if "tfrecord" in uri.lower()]
    return matches[shard_index] if shard_index < len(matches) else source.uri


def _select_splits(builder, requested: Sequence[str] | None) -> tuple[str, ...]:
    splits = getattr(getattr(builder, "info", None), "splits", None)
    available = tuple(sorted(str(name) for name in (splits or {})))
    if not available:
        raise AdapterError("prepared TFDS dataset declares no readable splits")
    if requested is None:
        return available
    selected = tuple(dict.fromkeys(str(name) for name in requested if str(name)))
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise AdapterError(
            f"unknown TFDS split(s) {unknown}; available splits: {', '.join(available)}"
        )
    return selected


def _episode_mapping(value: Any, *, split: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        try:
            value = dict(value)
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"TFDS split {split!r} yielded a non-mapping episode") from exc
    episode = {str(key): item for key, item in value.items()}
    if RLDS_STEPS_KEY not in episode:
        raise AdapterError(
            f"TFDS split {split!r} episode has no RLDS `steps` dataset; "
            "pass an RLDS-compatible builder directory"
        )
    return episode


def _normalized_steps(
    raw_steps: Iterable[Any], *, mapping: RldsFieldMapping
) -> Iterator[dict[str, Any]]:
    for step_index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            try:
                raw_step = dict(raw_step)
            except (TypeError, ValueError) as exc:
                raise AdapterError(f"RLDS step {step_index} is not a mapping") from exc
        step = {str(key): value for key, value in raw_step.items()}
        observation = step.get(RLDS_OBSERVATION_KEY)
        if not isinstance(observation, Mapping):
            raise AdapterError(f"RLDS step {step_index} has no mapping-valued `observation`")
        observation = {str(key): value for key, value in observation.items()}
        state_value = _mapped_state(step, observation, mapping.state_key)
        action_value = _lookup(step, mapping.action_key)
        language = _lookup(step, mapping.language_key)
        if language is None:
            language = _lookup(observation, mapping.language_key)
        if language is None:
            language = _lookup(observation, "natural_language_instruction")
        if language is None:
            language = _lookup(step, "language_instruction")
        timestamp_ns, timestamp_source = _timestamp_ns(step, observation, step_index, mapping)
        sanitized_observation, payload_blob, blob_fields = _observation_payload(observation)
        yield {
            "step_index": step_index,
            "timestamp_ns": timestamp_ns,
            "timestamp_source": timestamp_source,
            RLDS_OBSERVATION_KEY: sanitized_observation,
            "payload_blob": payload_blob,
            "payload_blob_fields": blob_fields,
            "state_vector": _numeric_vector(state_value),
            "action_vector": _numeric_vector(action_value),
            RLDS_REWARD_KEY: _optional_float(_lookup(step, RLDS_REWARD_KEY)),
            RLDS_DISCOUNT_KEY: _optional_float(_lookup(step, RLDS_DISCOUNT_KEY)),
            RLDS_IS_FIRST_KEY: _optional_bool(_lookup(step, RLDS_IS_FIRST_KEY)),
            RLDS_IS_LAST_KEY: _optional_bool(_lookup(step, RLDS_IS_LAST_KEY)),
            RLDS_IS_TERMINAL_KEY: _optional_bool(_lookup(step, RLDS_IS_TERMINAL_KEY)),
            "language_instruction": _optional_text(language),
        }


def _mapped_state(
    step: Mapping[str, Any], observation: Mapping[str, Any], configured: str | None
) -> Any:
    if configured:
        value = _lookup(observation, configured)
        return value if value is not None else _lookup(step, configured)
    for key in _STATE_KEYS:
        value = _lookup(observation, key)
        if value is not None:
            return value
    return None


def _timestamp_ns(
    step: Mapping[str, Any],
    observation: Mapping[str, Any],
    step_index: int,
    mapping: RldsFieldMapping,
) -> tuple[int, str]:
    candidates = (
        (mapping.timestamp_key, "configured"),
        ("timestamp_ns", "timestamp_ns"),
        ("timestamp", "timestamp"),
    )
    for key, source in candidates:
        if not key:
            continue
        value = _lookup(step, key)
        if value is None:
            value = _lookup(observation, key)
        scalar = _scalar(value)
        if scalar is None:
            continue
        numeric = float(scalar)
        if str(key).endswith("_ns") or abs(numeric) >= 1_000_000_000_000:
            return int(numeric), source
        return int(round(numeric * 1_000_000_000)), source
    if mapping.fps is not None:
        return int(round(step_index * 1_000_000_000 / mapping.fps)), "derived-fps"
    return step_index, "step-ordinal"


def _observation_payload(
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes | None, list[dict[str, Any]]]:
    fields: list[tuple[str, str, list[int], bytes]] = []
    sanitized = _json_safe(observation, path="observation", blobs=fields)
    if not fields:
        return sanitized, None, []
    offset = 0
    metadata: list[dict[str, Any]] = []
    bodies: list[bytes] = []
    for path, dtype, shape, data in fields:
        metadata.append(
            {
                "path": path,
                "dtype": dtype,
                "shape": shape,
                "offset": offset,
                "length": len(data),
            }
        )
        bodies.append(data)
        offset += len(data)
    header = json.dumps(
        {"format": _BLOB_FORMAT, "fields": metadata},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = _BLOB_MAGIC + struct.pack(">Q", len(header)) + header + b"".join(bodies)
    return sanitized, payload, metadata


def _json_safe(
    value: Any,
    *,
    path: str = "",
    blobs: list[tuple[str, str, list[int], bytes]] | None = None,
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(
                item,
                path=f"{path}.{key}" if path else str(key),
                blobs=blobs,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path=f"{path}[{index}]", blobs=blobs)
            for index, item in enumerate(value)
        ]
    tensor = _tensor_numpy(value)
    if tensor is not value:
        return _json_safe(tensor, path=path, blobs=blobs)
    if isinstance(value, bytes):
        if not _payload_like(path) and len(value) < _BLOB_THRESHOLD_BYTES:
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                pass
        if blobs is not None:
            blobs.append((path, "bytes", [len(value)], value))
            return {"$blob": {"format": _BLOB_FORMAT, "path": path, "length": len(value)}}
        return {"$bytes": len(value)}
    if _array_like(value):
        shape = [int(item) for item in getattr(value, "shape", ())]
        dtype = str(getattr(value, "dtype", "unknown"))
        size = int(getattr(value, "size", 0) or 0)
        tobytes = getattr(value, "tobytes", None)
        if blobs is not None and callable(tobytes) and (
            len(shape) >= 2 or _payload_like(path) or size >= _BLOB_THRESHOLD_BYTES
        ):
            data = bytes(tobytes())
            blobs.append((path, dtype, shape, data))
            return {
                "$blob": {
                    "format": _BLOB_FORMAT,
                    "path": path,
                    "dtype": dtype,
                    "shape": shape,
                    "length": len(data),
                }
            }
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return _json_safe(tolist(), path=path, blobs=blobs)
    scalar = _scalar(value)
    if scalar is not value:
        return _json_safe(scalar, path=path, blobs=blobs)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _numeric_vector(value: Any, *, required: bool = False) -> list[float] | None:
    if value is None:
        if required:
            raise AdapterError("configured RLDS action value is null")
        return None
    value = _tensor_numpy(value)
    values: list[float] = []

    def visit(item: Any) -> None:
        item = _tensor_numpy(item)
        if isinstance(item, Mapping):
            for key in sorted(item):
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        if _array_like(item):
            tolist = getattr(item, "tolist", None)
            if callable(tolist):
                visit(tolist())
                return
        scalar = _scalar(item)
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise AdapterError(f"RLDS vector contains non-numeric value {scalar!r}")
        values.append(float(scalar))

    visit(value)
    return values


def _lookup(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in str(dotted).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _tensor_numpy(value: Any) -> Any:
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        try:
            return numpy()
        except Exception as exc:  # noqa: BLE001 - normalize TensorFlow tensor failures.
            raise AdapterError(f"cannot materialize one RLDS tensor value: {exc}") from exc
    return value


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            return value
    return value


def _array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _payload_like(path: str) -> bool:
    lowered = path.lower()
    return any(hint in lowered for hint in _IMAGE_HINTS)


def _optional_text(value: Any) -> str | None:
    value = _scalar(_tensor_numpy(value))
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    value = _scalar(_tensor_numpy(value))
    return None if value is None else bool(value)


def _optional_float(value: Any) -> float | None:
    value = _scalar(_tensor_numpy(value))
    return None if value is None else float(value)


def _split_summaries(builder) -> list[dict[str, Any]]:
    splits = getattr(getattr(builder, "info", None), "splits", None) or {}
    rows: list[dict[str, Any]] = []
    for name in sorted(splits):
        info = splits[name]
        instructions = getattr(info, "file_instructions", ()) or ()
        shard_count = len(instructions)
        if not shard_count:
            shard_lengths = getattr(info, "shard_lengths", ()) or ()
            shard_count = len(shard_lengths)
        rows.append(
            {
                "name": str(name),
                "episode_count": int(getattr(info, "num_examples", 0) or 0),
                "shard_count": int(shard_count),
                "num_bytes": int(getattr(info, "num_bytes", 0) or 0),
            }
        )
    return rows


def _validate_tfds_source_request(
    source: str | Path | RldsSource,
    storage_options: Mapping[str, Any] | None,
) -> None:
    uri = source.uri if isinstance(source, RldsSource) else str(source)
    if not is_object_store_uri(uri):
        return
    if not uri.startswith(("gs://", "gcs://")):
        raise AdapterError(
            "prepared TFDS object-store directories are currently supported through "
            "TensorFlow's `gs://` filesystem; mirror the dataset to GCS or use a local "
            "prepared directory"
        )
    if storage_options:
        raise AdapterError(
            "TensorFlow Datasets reads `gs://` through TensorFlow's GCS filesystem and "
            "Application Default Credentials; fsspec-style storage_options cannot be "
            "forwarded to builder_from_directory. Configure ADC (or omit explicit "
            "source storage options) and retry."
        )


def _tfds_uri(uri: str) -> str:
    """Normalize the fsspec GCS alias to TensorFlow's canonical scheme."""
    return f"gs://{uri[6:]}" if uri.startswith("gcs://") else uri


def _feature_summary(features: Any) -> dict[str, str]:
    if isinstance(features, Mapping):
        return {str(key): type(value).__name__ for key, value in features.items()}
    return {"schema": type(features).__name__ if features is not None else "unknown"}


def _builder_name(builder, dataset_info: Mapping[str, Any]) -> str:
    info = getattr(builder, "info", None)
    return str(getattr(info, "name", None) or dataset_info.get("name") or "unknown")


def _builder_version(builder, dataset_info: Mapping[str, Any]) -> str:
    info = getattr(builder, "info", None)
    return str(getattr(info, "version", None) or dataset_info.get("version") or "unknown")


def _package_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _include_dataset_artifact(root: str, uri: str) -> bool:
    relative = _relative_file(root, uri)
    parts = Path(relative).parts
    return bool(parts) and not any(part.startswith(".") for part in parts) and not relative.endswith(
        (".lock", ".tmp")
    )


def _relative_file(root: str, uri: str) -> str:
    if is_object_store_uri(root):
        prefix = root.rstrip("/") + "/"
        return str(uri)[len(prefix) :] if str(uri).startswith(prefix) else str(uri)
    try:
        return Path(uri).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(uri).name
