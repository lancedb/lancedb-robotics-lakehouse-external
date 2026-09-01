"""Canonical write path for prepared RLDS / TensorFlow Datasets sources."""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa

from lancedb_robotics.adapters import AdapterError, get_adapter
from lancedb_robotics.adapters.rlds_adapter import RldsFieldMapping
from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.rlds_contract import RLDS_CANONICAL_CONTRACT_VERSION
from lancedb_robotics.schemas import (
    EPISODES_SCHEMA,
    EVENTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RUNS_SCHEMA,
    SCENARIOS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
)

_LOCKS_GUARD = threading.Lock()
_INGEST_LOCKS: dict[str, threading.Lock] = {}


def ingest_rlds_impl(
    lake: Lake,
    source: str | Path,
    *,
    splits: tuple[str, ...] | list[str] | None = None,
    mapping: RldsFieldMapping | None = None,
    created_by: str = "lancedb-robotics",
    batch_size: int = 1024,
    compact: bool = True,
    prune_versions: bool = True,
    retain_versions: int = 2,
    index_predicates: bool = True,
    auth_ref: str | None = None,
    storage_options: dict[str, Any] | None = None,
):
    """Ingest a prepared RLDS/TFDS builder directory into canonical rows.

    Reads are explicitly sliced one physical TFDS shard at a time by the
    adapter.  Observation, episode, and scenario writes are batch-flushed; only
    one episode's state and one write batch are retained in Python.  Stable IDs
    derive from a path-independent content digest, so relocating a byte-identical
    dataset produces an audited no-op.
    """
    from lancedb_robotics.ingest import (
        IngestReport,
        _already_ingested_report,
        _finalize_ingest,
        _register_resolved_source,
    )

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    field_mapping = mapping or RldsFieldMapping()
    if not isinstance(field_mapping, RldsFieldMapping):
        raise TypeError("mapping must be an RldsFieldMapping")

    adapter = get_adapter("rlds")
    availability = adapter.availability()
    if not availability["available"]:
        missing = ", ".join(availability["missing"])
        raise AdapterError(
            "RLDS/TFDS ingest dependencies are unavailable; install "
            f"`{availability['install']}` on a supported host and retry "
            f"(missing: {missing})"
        )
    resolved = adapter.source(source, storage_options=storage_options, auth_ref=auth_ref)
    inspect_started = datetime.now(UTC)
    inspect_report = adapter.inspect(
        resolved,
        storage_options=storage_options,
        auth_ref=auth_ref,
    )
    inspect_finished = datetime.now(UTC)
    effective_splits = _effective_splits(inspect_report, splits)
    run_digest = _run_digest(resolved.checksum, effective_splits, field_mapping)
    run_id = f"run-rlds-{run_digest}"
    source_id = f"src-{resolved.digest}"
    inspect_transform_id = f"tfm-rlds-{run_digest}-inspect"
    ingest_transform_id = f"tfm-rlds-{run_digest}-ingest"
    lock = _ingest_lock(lake.uri, run_digest)

    with lock, _rlds_ingest_claim(lake, run_id=run_id, claimed_by=created_by):
        if lake.table("runs").count_rows(_equals("run_id", run_id)) > 0:
            return _already_ingested_report(
                lake,
                source_id,
                resolved.uri,
                resolved.checksum,
                run_id,
                inspect_report,
                auth_ref=auth_ref,
                created_by=created_by,
                adapter_name="rlds",
            )

        # A prior process may have failed after a bounded batch commit but
        # before the run completion row.  Deterministic IDs plus scoped cleanup
        # make an explicit retry converge instead of appending duplicates.
        _cleanup_partial_run(lake, run_id, inspect_transform_id, ingest_transform_id)
        registration = _register_resolved_source(
            lake,
            resolved,
            adapter="rlds",
            inspect_report=inspect_report,
            auth_ref=auth_ref,
        )
        now = datetime.now(UTC)
        observation_batch: list[dict[str, Any]] = []
        episode_batch: list[dict[str, Any]] = []
        scenario_batch: list[dict[str, Any]] = []
        total_observations = 0
        total_episodes = 0
        run_start: int | None = None
        run_end: int | None = None
        selected_splits: set[str] = set()
        shard_coordinates: set[tuple[str, int]] = set()

        try:
            for episode in adapter.iter_episodes(
                resolved,
                splits=effective_splits,
                mapping=field_mapping,
                storage_options=storage_options,
                auth_ref=auth_ref,
            ):
                canonical_episode_index = total_episodes
                episode_id = _episode_id(run_id, canonical_episode_index)
                scenario_id = _scenario_id(run_digest, canonical_episode_index)
                selected_splits.add(episode.split)
                shard_coordinates.add((episode.split, episode.shard_index))
                step_iterator = iter(episode.steps)
                try:
                    current = next(step_iterator)
                except StopIteration as exc:
                    raise AdapterError(
                        f"RLDS episode {episode.source_episode_index} in split "
                        f"{episode.split!r} has no steps"
                    ) from exc
                if current["is_first"] is not True:
                    raise AdapterError(
                        f"RLDS episode {episode.source_episode_index} must begin with "
                        "is_first=True"
                    )

                observation_ids: list[str] = []
                episode_start: int | None = None
                episode_end: int | None = None
                episode_task: str | None = None
                final_terminal: bool | None = None
                step_count = 0
                while True:
                    try:
                        following = next(step_iterator)
                    except StopIteration:
                        following = None
                    is_final = following is None
                    if step_count > 0 and current["is_first"] is not False:
                        raise AdapterError(
                            f"RLDS episode {episode.source_episode_index} has "
                            f"is_first={current['is_first']!r} at interior step {step_count}"
                        )
                    if current["is_last"] is not is_final:
                        expected = "True on the final step" if is_final else "False before the final step"
                        raise AdapterError(
                            f"RLDS episode {episode.source_episode_index} requires "
                            f"is_last={expected}; got {current['is_last']!r} at step {step_count}"
                        )

                    observation_id = _observation_id(
                        run_id, canonical_episode_index, step_count
                    )
                    language = current.get("language_instruction")
                    if episode_task is None and language:
                        episode_task = str(language)
                    observation_ids.append(observation_id)
                    timestamp_ns = int(current["timestamp_ns"])
                    episode_start = (
                        timestamp_ns if episode_start is None else min(episode_start, timestamp_ns)
                    )
                    episode_end = (
                        timestamp_ns if episode_end is None else max(episode_end, timestamp_ns)
                    )
                    run_start = timestamp_ns if run_start is None else min(run_start, timestamp_ns)
                    run_end = timestamp_ns if run_end is None else max(run_end, timestamp_ns)
                    final_terminal = current.get("is_terminal") if is_final else final_terminal
                    observation_batch.append(
                        _observation_row(
                            current,
                            run_id=run_id,
                            episode_id=episode_id,
                            episode_index=canonical_episode_index,
                            frame_index=step_count,
                            observation_id=observation_id,
                            task_id=str(language) if language else episode_task,
                            shard_uri=episode.shard_uri,
                            split=episode.split,
                            shard_index=episode.shard_index,
                            shard_episode_index=episode.shard_episode_index,
                            source_episode_index=episode.source_episode_index,
                            tfds_id=episode.tfds_id,
                            transform_id=ingest_transform_id,
                            created_at=now,
                        )
                    )
                    total_observations += 1
                    step_count += 1
                    _flush_rows(
                        lake.table("observations"),
                        observation_batch,
                        OBSERVATIONS_SCHEMA,
                        batch_size,
                    )
                    if following is None:
                        break
                    current = following

                outcome = (
                    "terminal"
                    if final_terminal is True
                    else "truncated"
                    if final_terminal is False
                    else None
                )
                provenance = {
                    "adapter": "rlds",
                    "source_uri": resolved.uri,
                    "source_checksum": resolved.checksum,
                    "split": episode.split,
                    "shard_index": episode.shard_index,
                    "shard_uri": episode.shard_uri,
                    "shard_episode_index": episode.shard_episode_index,
                    "source_episode_index": episode.source_episode_index,
                    "tfds_id": episode.tfds_id,
                    "source_episode_metadata": episode.metadata,
                    "field_mapping": asdict(field_mapping),
                }
                episode_batch.append(
                    {
                        "episode_id": episode_id,
                        "run_id": run_id,
                        "episode_index": canonical_episode_index,
                        "from_timestamp_ns": int(episode_start or 0),
                        "to_timestamp_ns": int(episode_end or 0),
                        "boundary_source": "rlds-flags",
                        "outcome": outcome,
                        "frame_count": step_count,
                        "camera_blobs": [],
                        "task_id": episode_task,
                        "embedding": None,
                        "provenance": json.dumps(
                            provenance, sort_keys=True, separators=(",", ":")
                        ),
                        "transform_id": ingest_transform_id,
                        "created_at": now,
                    }
                )
                scenario_batch.append(
                    {
                        "scenario_id": scenario_id,
                        "run_id": run_id,
                        "start_time_ns": int(episode_start or 0),
                        "end_time_ns": int(episode_end or 0),
                        "window_ns": max(0, int(episode_end or 0) - int(episode_start or 0)),
                        "is_partial": outcome == "truncated",
                        "topics": ["rlds.steps"],
                        "observation_ids": observation_ids,
                        "observation_count": len(observation_ids),
                        "scenario_type": "episode",
                        "trigger_event_id": None,
                        "source": "rlds-authored",
                        "parent_scenario_id": None,
                        "coverage_tags": ["rlds", "episode", episode.split],
                        "summary": episode_task,
                        "transform_id": ingest_transform_id,
                        "created_at": now,
                    }
                )
                total_episodes += 1
                _flush_rows(
                    lake.table("episodes"), episode_batch, EPISODES_SCHEMA, batch_size
                )
                _flush_rows(
                    lake.table("scenarios"), scenario_batch, SCENARIOS_SCHEMA, batch_size
                )

            if total_episodes == 0 or total_observations == 0:
                raise AdapterError("selected RLDS/TFDS splits contain no valid episode steps")
            _flush_all(lake.table("observations"), observation_batch, OBSERVATIONS_SCHEMA)
            _flush_all(lake.table("episodes"), episode_batch, EPISODES_SCHEMA)
            _flush_all(lake.table("scenarios"), scenario_batch, SCENARIOS_SCHEMA)
        except Exception:
            _cleanup_partial_run(lake, run_id, inspect_transform_id, ingest_transform_id)
            raise

        start_time_ns = int(run_start or 0)
        end_time_ns = int(run_end or start_time_ns)
        run_metadata = [
            {"key": "adapter", "value": "rlds"},
            {"key": "dataset_name", "value": str(inspect_report.get("dataset_name") or "")},
            {
                "key": "dataset_version",
                "value": str(inspect_report.get("dataset_version") or ""),
            },
            {"key": "splits", "value": ",".join(sorted(selected_splits))},
            {"key": "shard_count", "value": str(len(shard_coordinates))},
            {"key": "episode_count", "value": str(total_episodes)},
            {"key": "source_identity.kind", "value": resolved.identity_kind},
            {"key": "integrity.status", "value": "complete"},
        ]
        run_row = {
            "run_id": run_id,
            "run_kind": "dataset",
            "source": "rlds",
            "source_id": registration.source_id,
            "raw_uri": registration.uri,
            "start_time_ns": start_time_ns,
            "end_time_ns": end_time_ns,
            "duration_ns": max(0, end_time_ns - start_time_ns),
            "metadata": run_metadata,
            "quality_flags": None,
            "transform_id": ingest_transform_id,
            "created_at": now,
        }
        event_rows = [
            {
                "event_id": f"{run_id}:{event_type}",
                "run_id": run_id,
                "timestamp_ns": timestamp_ns,
                "event_type": event_type,
                "severity": "info",
                "source": "rlds-boundary",
                "transform_id": ingest_transform_id,
                "created_at": now,
            }
            for event_type, timestamp_ns in (
                ("run_start", start_time_ns),
                ("run_end", end_time_ns),
            )
        ]
        ingest_finished = datetime.now(UTC)
        params = {
            "adapter": "rlds",
            "canonical_contract_version": RLDS_CANONICAL_CONTRACT_VERSION,
            "run_id": run_id,
            "batch_size": batch_size,
            "splits": sorted(selected_splits),
            "shard_count": len(shard_coordinates),
            "episode_count": total_episodes,
            "observation_count": total_observations,
            "field_mapping": asdict(field_mapping),
            "source_identity": {
                "kind": resolved.identity_kind,
                "checksum": resolved.checksum,
                "artifact_count": len(resolved.relative_files),
            },
        }
        transform_rows = [
            {
                "transform_id": inspect_transform_id,
                "kind": "inspect",
                "source_id": registration.source_id,
                "input_uris": [registration.uri],
                "output_tables": [],
                "params": json.dumps({"adapter": "rlds"}, sort_keys=True),
                "status": "completed",
                "started_at": inspect_started,
                "finished_at": inspect_finished,
                "created_by": created_by,
                "created_at": now,
            },
            {
                "transform_id": ingest_transform_id,
                "kind": "ingest",
                "source_id": registration.source_id,
                "input_uris": [registration.uri],
                "output_tables": [
                    "runs",
                    "episodes",
                    "observations",
                    "scenarios",
                    "events",
                ],
                "params": json.dumps(params, sort_keys=True, separators=(",", ":")),
                "status": "completed",
                "started_at": inspect_started,
                "finished_at": ingest_finished,
                "created_by": created_by,
                "created_at": now,
            },
        ]
        try:
            lake.table("runs").add(pa.Table.from_pylist([run_row], schema=RUNS_SCHEMA))
            lake.table("events").add(pa.Table.from_pylist(event_rows, schema=EVENTS_SCHEMA))
            lake.table("transform_runs").add(
                pa.Table.from_pylist(transform_rows, schema=TRANSFORM_RUNS_SCHEMA)
            )
            compaction = _finalize_ingest(
                lake,
                compact=compact,
                prune_versions=prune_versions,
                retain_versions=retain_versions,
                created_by=created_by,
                index_predicates=index_predicates,
            )
            for transform_row in transform_rows:
                emit_transform_lineage(lake, transform_row)
        except Exception:
            _cleanup_partial_run(lake, run_id, inspect_transform_id, ingest_transform_id)
            raise

        return IngestReport(
            lake_uri=lake.uri,
            source=registration,
            run_id=run_id,
            already_ingested=False,
            transform_id=ingest_transform_id,
            compaction=compaction,
            rows_added={
                "integration_sources": 1 if registration.created else 0,
                "runs": 1,
                "episodes": total_episodes,
                "observations": total_observations,
                "scenarios": total_episodes,
                "events": len(event_rows),
                "transform_runs": len(transform_rows),
            },
            observations_by_topic={"rlds.steps": total_observations},
            message_count=total_observations,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            duration_ns=max(0, end_time_ns - start_time_ns),
            decode_by_status={"decoded": total_observations},
            decode_by_encoding={"tfrecord": total_observations},
            integrity_status="complete",
        )


def _observation_row(
    step: dict[str, Any],
    *,
    run_id: str,
    episode_id: str,
    episode_index: int,
    frame_index: int,
    observation_id: str,
    task_id: str | None,
    shard_uri: str,
    split: str,
    shard_index: int,
    shard_episode_index: int,
    source_episode_index: int,
    tfds_id: str | None,
    transform_id: str,
    created_at: datetime,
) -> dict[str, Any]:
    payload = {
        "format": "rlds-tfds",
        "split": split,
        "shard_index": shard_index,
        "shard_episode_index": shard_episode_index,
        "source_episode_index": source_episode_index,
        "step_index": frame_index,
        "tfds_id": tfds_id,
        "timestamp_source": step["timestamp_source"],
        "reward": step.get("reward"),
        "discount": step.get("discount"),
        "is_first": step.get("is_first"),
        "is_last": step.get("is_last"),
        "is_terminal": step.get("is_terminal"),
        "observation": step["observation"],
        "payload_blob_fields": step.get("payload_blob_fields") or [],
    }
    return {
        "observation_id": observation_id,
        "run_id": run_id,
        "episode_id": episode_id,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp_ns": int(step["timestamp_ns"]),
        "sensor_id": "rlds-step",
        "topic": "rlds.steps",
        "modality": "multimodal" if step.get("payload_blob") else "state_action",
        "robot_id": None,
        "site_id": None,
        "task_id": task_id,
        "software_version": None,
        "outcome": None,
        "raw_uri": shard_uri,
        "raw_channel": f"{split}/episode/{shard_episode_index}/steps",
        "raw_log_time_ns": int(step["timestamp_ns"]),
        "raw_sequence": frame_index,
        "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        "payload_blob": step.get("payload_blob"),
        "message_encoding": "tfrecord",
        "schema_encoding": "rlds-step-v1",
        "decode_status": "decoded",
        "decode_error": None,
        "state_vector": step.get("state_vector"),
        "action_vector": step.get("action_vector"),
        "caption": task_id,
        "quality_flags": None,
        "transform_id": transform_id,
        "created_at": created_at,
    }


def _flush_rows(table, rows: list[dict[str, Any]], schema, batch_size: int) -> None:
    if len(rows) >= batch_size:
        _flush_all(table, rows, schema)


def _flush_all(table, rows: list[dict[str, Any]], schema) -> None:
    if rows:
        table.add(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def _cleanup_partial_run(
    lake: Lake, run_id: str, inspect_transform_id: str, ingest_transform_id: str
) -> None:
    for table_name in ("observations", "episodes", "scenarios", "events", "runs"):
        table = lake.table(table_name)
        if table.count_rows(_equals("run_id", run_id)):
            table.delete(_equals("run_id", run_id))
    transform_filter = (
        f"transform_id = '{_escape(inspect_transform_id)}' OR "
        f"transform_id = '{_escape(ingest_transform_id)}'"
    )
    transforms = lake.table("transform_runs")
    if transforms.count_rows(transform_filter):
        transforms.delete(transform_filter)


def _ingest_lock(lake_uri: str, digest: str) -> threading.Lock:
    key = f"{lake_uri}\0{digest}"
    with _LOCKS_GUARD:
        return _INGEST_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _rlds_ingest_claim(lake: Lake, *, run_id: str, claimed_by: str):
    """Serialize RLDS multi-table writes across processes through a CAS gate.

    Lance does not provide a transaction spanning the canonical grain tables,
    so cleanup + bounded appends + finalization must not overlap another RLDS
    ingest.  The pre-seeded gate makes ``Table.update`` re-evaluate a NULL-owner
    predicate against the latest committed state; a loser updates zero rows and
    fails before deleting or writing canonical data.
    """
    table = lake.table("rlds_ingest_claims")
    total_rows = int(table.count_rows())
    global_rows = int(table.count_rows("claim_key = 'global'"))
    if total_rows != 1 or global_rows != 1:
        raise AdapterError(
            "RLDS ingest coordination is corrupt: rlds_ingest_claims must contain "
            "exactly one `global` gate row "
            f"(found total={total_rows}, global={global_rows}). No writes were "
            "started; restore the gate from a known-good lake or migrate into a "
            "freshly initialized lake"
        )
    token = uuid4().hex
    now = datetime.now(UTC)
    where = "claim_key = 'global' AND owner_token IS NULL"
    result = table.update(
        where=where,
        values={
            "owner_token": token,
            "run_id": run_id,
            "claimed_by": claimed_by,
            "claimed_at": now,
            "updated_at": now,
        },
    )
    if int(result.rows_updated) != 1:
        if int(result.rows_updated) > 1:
            # Defensive rollback for externally corrupted tables. Product code
            # never appends claim rows, and validation above normally catches
            # this before the CAS attempt.
            table.update(
                where=f"claim_key = 'global' AND owner_token = '{_escape(token)}'",
                values={
                    "owner_token": None,
                    "run_id": None,
                    "claimed_by": None,
                    "claimed_at": None,
                    "updated_at": datetime.now(UTC),
                },
            )
            raise AdapterError(
                "RLDS ingest coordination is corrupt: the write claim matched "
                "multiple rows; no canonical writes were started"
            )
        rows = (
            table.search()
            .where("claim_key = 'global'")
            .select(["owner_token", "run_id", "claimed_by", "claimed_at"])
            .limit(1)
            .to_arrow()
            .to_pylist()
        )
        if not rows:
            raise AdapterError(
                "RLDS ingest coordination gate disappeared during acquisition; "
                "no canonical writes were started. Restore or migrate the lake "
                "before retrying"
            )
        owner = rows[0]
        raise AdapterError(
            "another RLDS ingest already holds the lake-wide write claim "
            f"(run_id={owner.get('run_id')!r}, claimed_by={owner.get('claimed_by')!r}, "
            f"claimed_at={owner.get('claimed_at')!r}); wait for it to finish. "
            "Automatic stale-claim recovery is tracked separately so a hard crash "
            "fails closed instead of risking concurrent cleanup corruption"
        )

    body_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        try:
            released = table.update(
                where=f"claim_key = 'global' AND owner_token = '{_escape(token)}'",
                values={
                    "owner_token": None,
                    "run_id": None,
                    "claimed_by": None,
                    "claimed_at": None,
                    "updated_at": datetime.now(UTC),
                },
            )
        except Exception:
            if body_error is None:
                raise
        else:
            if int(released.rows_updated) != 1 and body_error is None:
                raise AdapterError(
                    "RLDS ingest completed but its lake-wide write claim could not be "
                    "released; inspect rlds_ingest_claims before starting another "
                    "RLDS ingest"
                )


def _effective_splits(
    inspect_report: dict[str, Any],
    requested: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    available = tuple(
        sorted(str(row["name"]) for row in inspect_report.get("splits") or ())
    )
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


def _episode_id(run_id: str, episode_index: int) -> str:
    return f"{run_id}:episode:{episode_index:06d}"


def _observation_id(run_id: str, episode_index: int, frame_index: int) -> str:
    return f"{run_id}:frame:{episode_index:06d}:{frame_index:06d}"


def _scenario_id(run_digest: str, episode_index: int) -> str:
    return f"scn-rlds-{run_digest}-episode-{episode_index:06d}"


def _run_digest(
    source_checksum: str,
    splits: tuple[str, ...],
    mapping: RldsFieldMapping,
) -> str:
    identity = {
        "canonical_contract_version": RLDS_CANONICAL_CONTRACT_VERSION,
        "source_checksum": source_checksum,
        "splits": list(splits),
        "field_mapping": asdict(mapping),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def _equals(column: str, value: str) -> str:
    return f"{column} = '{_escape(value)}'"


def _escape(value: str) -> str:
    return str(value).replace("'", "''")
