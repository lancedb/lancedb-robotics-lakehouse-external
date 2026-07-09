"""MCAP ingest: turn a raw robot log into canonical lakehouse rows.

One ingest writes one ``runs`` row, topic/time-indexed ``observations``,
one ``attachments`` row per embedded file, run-boundary ``events``, and two
``transform_runs`` lineage rows (inspect + ingest). Rows carry pointer
provenance (``raw_uri``/``raw_channel``/``raw_sequence``) back to the source
file. Beyond message streams, MCAP's other first-class records travel into the
lake too (backlog 0016): attachment bytes land in the ``attachments`` table, and
log-level metadata records merge into ``runs.metadata`` namespaced by record
name.

Run identity is content-addressed from the file bytes (not the source path), so
the same log ingested from anywhere yields the same ids — and re-ingesting an
unchanged file is an audited no-op instead of a duplicate run.

Scale and robustness (backlog 0017): observations stream to the lake in batches
of ``batch_size`` rather than materializing the whole table in memory, so a
multi-GB log ingests with bounded memory (drop ``batch_size`` toward ~100 for
image/lidar-heavy logs whose rows carry large ``payload_blob`` bytes). Chunk CRCs
are validated as messages are read; a CRC mismatch or a truncated file does not
abort the run — the readable prefix is kept and the run is quarantined with an
``integrity`` verdict that the quality gate (FS4) also honors. A missing
compression codec is the one hard error (it is fixable by installing the codec).
"""

import hashlib
import importlib.metadata
import importlib.util
import inspect as inspect_module
import json
import math
import shlex
import shutil
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa

from lancedb_robotics.adapters import AdapterError, CorruptMcapError, get_adapter
from lancedb_robotics.extract import LAYOUT_VERSION, extract
from lancedb_robotics.keyframe_maps import (
    KeyframeMapError,
    keyframe_map_artifact_referrer_row,
    keyframe_map_artifact_row,
    keyframe_map_entries_from_json,
    keyframe_map_ref,
    keyframe_map_shape,
    load_keyframe_map_json,
    should_inline_keyframe_map,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.lerobot_object_store_manifest import (
    LeRobotObjectStoreManifestCache,
    resolve_lerobot_object_store_manifest_cache,
)
from lancedb_robotics.lerobot_object_store_validation import (
    DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
    DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
)
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.recordings import (
    Recording,
    inspect_recording,
    iter_shard_messages,
    resolve_recording,
    resolve_shards,
)
from lancedb_robotics.schemas import (
    ATTACHMENTS_SCHEMA,
    EPISODES_SCHEMA,
    EVENTS_SCHEMA,
    INTEGRATION_SOURCES_SCHEMA,
    KEYFRAME_MAP_ARTIFACT_REFERRERS_SCHEMA,
    KEYFRAME_MAP_ARTIFACTS_SCHEMA,
    LEROBOT_CHECKPOINT_HOLDS_SCHEMA,
    LEROBOT_INGEST_CHECKPOINTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RUNS_SCHEMA,
    SCENARIOS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
    VIDEO_ENCODINGS_SCHEMA,
    VIDEOS_SCHEMA,
)
from lancedb_robotics.sources import (
    SourceRegistration,
    content_key_digest,
    file_checksum,
    recording_content_key,
    register_recording_source,
    register_source,
)
from lancedb_robotics.storage import display_name, is_object_store_uri, source_uri

_LEROBOT_EXISTING_ID_SET_LIMIT = 100_000
_LEROBOT_CHECKPOINT_TABLE = "lerobot_ingest_checkpoints"
_LEROBOT_CHECKPOINT_HOLDS_TABLE = "lerobot_checkpoint_holds"
_LEROBOT_TERMINAL_STATUSES = frozenset({"abandoned", "completed", "failed", "skipped"})
DEFAULT_LEROBOT_CLAIM_LEASE = timedelta(hours=6)
DEFAULT_LEROBOT_CLAIM_HEARTBEAT = timedelta(minutes=5)
DEFAULT_LEROBOT_CHECKPOINT_RETENTION_AGE = timedelta(days=30)
DEFAULT_LEROBOT_CHECKPOINT_RETAIN_COMPLETED_PER_SOURCE = 10
DEFAULT_LEROBOT_CHECKPOINT_RETAIN_FAILED_PER_SOURCE = 10
DEFAULT_LEROBOT_KEYFRAME_MAP_INLINE_THRESHOLD_BYTES = 64 * 1024
DEFAULT_LEROBOT_KEYFRAME_MAP_INLINE_THRESHOLD_FRAMES = 4096

# Default streaming batch size: observation rows are flushed to the table every
# this many messages. 1024 mirrors the proven default; drop toward ~100 for
# image/lidar-heavy logs (large payload_blob bytes per row) to bound memory.
DEFAULT_BATCH_SIZE = 1024

# Post-ingest compaction + retention (BUG-14 / backlog 0180). Each streaming
# `_flush` below is one `table.add` = one Lance fragment AND one new version, so a
# freshly-streamed `observations` grain is born at ~`batch_size` rows/fragment --
# 3-4 orders of magnitude below Lance's healthy ~1M-row `max_rows_per_file`
# target. That imposes a measured ~79x per-row scan tax (per-fragment footer/open
# overhead) on every later read, and one version per flush is unbounded
# manifest/metadata growth. So at the end of ingest we compact the grain table up
# to Lance's healthy fragment size and snapshot-safely prune the per-flush version
# churn, reusing `maintenance.maintain_lake` rather than reimplementing its
# pinning/cleanup safety. Compaction only rewrites fragments below target and is a
# no-op once they are large, so it stays incremental as more runs are appended
# (evidence: lance optimize.rs target_rows_per_fragment=1<<20, "no compaction
# needed -> no new version"). Retaining a couple of recent versions keeps a little
# time-travel while bounding the churn; versions pinned by a live snapshot/lineage
# are tagged and never pruned.
DEFAULT_INGEST_RETAIN_VERSIONS = 2

# Modality guesses from topic/schema naming conventions; "unknown" is an
# honest answer, downstream enrichment can overwrite it.
_MODALITY_HINTS: tuple[tuple[str, str], ...] = (
    ("image", "image"),
    ("camera", "image"),
    ("imu", "imu"),
    ("lidar", "pointcloud"),
    ("pointcloud", "pointcloud"),
    ("point_cloud", "pointcloud"),
    ("gps", "gps"),
    ("gnss", "gps"),
)


class LeRobotClaimPreconditionError(AdapterError):
    """Raised when a LeRobot claim CAS precondition no longer matches latest state."""

    def __init__(
        self,
        job_id: str,
        *,
        operation: str,
        lake_uri: str | None,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> None:
        self.error = "lerobot_claim_precondition_failed"
        self.operation = operation
        self.lake_uri = lake_uri
        self.job_id = job_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"LeRobot claim precondition failed for job {job_id}: latest claim no longer "
            f"matches expected state before {operation}; expected "
            f"checkpoint_id={expected.get('checkpoint_id')!r} "
            f"claim_token={expected.get('claim_token')!r}; actual "
            f"checkpoint_id={actual.get('checkpoint_id')!r} status={actual.get('status')!r} "
            f"phase={actual.get('phase')!r} claim_owner={actual.get('claim_owner')!r} "
            f"claim_token={actual.get('claim_token')!r}. Refresh "
            f"`lancedb-robotics ingest lerobot-job {job_id}` or rerun "
            "`lancedb-robotics ingest lerobot-claim-watchdog`."
        )

    def to_params(self) -> dict[str, Any]:
        payload = {
            "error": self.error,
            "operation": self.operation,
            "lake_uri": self.lake_uri,
            "job_id": self.job_id,
            "message": str(self),
        }
        payload.update({f"expected_latest_{key}": value for key, value in self.expected.items()})
        payload.update({f"actual_latest_{key}": value for key, value in self.actual.items()})
        return payload


def list_lerobot_ingest_jobs(
    lake: Lake,
    *,
    status: str | None = None,
    source_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List latest durable LeRobot ingest job checkpoints."""
    rows = _lerobot_checkpoint_rows(lake)
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = latest.get(str(row["job_id"]))
        if current is None or _lerobot_checkpoint_sort_key(row) > _lerobot_checkpoint_sort_key(
            current
        ):
            latest[str(row["job_id"])] = row
    filtered = [
        row
        for row in latest.values()
        if (source_id is None or row.get("source_id") == source_id)
        and (status is None or row.get("status") == status)
    ]
    ordered = sorted(filtered, key=_lerobot_checkpoint_sort_key, reverse=True)
    return [_hydrate_lerobot_checkpoint_row(row) for row in ordered[: max(0, int(limit))]]


def get_lerobot_ingest_job(lake: Lake, job_id: str) -> dict[str, Any]:
    """Return latest LeRobot ingest job state plus checkpoint history."""
    rows = [row for row in _lerobot_checkpoint_rows(lake) if row.get("job_id") == job_id]
    if not rows:
        raise KeyError(f"no LeRobot ingest job {job_id!r}")
    ordered = sorted(rows, key=_lerobot_checkpoint_sort_key)
    latest = _hydrate_lerobot_checkpoint_row(ordered[-1])
    latest["history"] = [_hydrate_lerobot_checkpoint_row(row) for row in ordered]
    return latest


def recommend_lerobot_media_inspection_timeouts(
    lake: Lake | None = None,
    *,
    checkpoint_rows: Sequence[Mapping[str, Any]] | None = None,
    transform_rows: Sequence[Mapping[str, Any]] | None = None,
    job_id: str | None = None,
    source_id: str | None = None,
    source_uri: str | None = None,
    storage_tier: str | None = None,
    provider: str | None = None,
    min_timeout_seconds: float = 1.0,
    max_timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Summarize LeRobot media-inspection timing and recommend timeout/retry settings."""
    if min_timeout_seconds <= 0:
        raise ValueError("min_timeout_seconds must be positive")
    if max_timeout_seconds < min_timeout_seconds:
        raise ValueError("max_timeout_seconds must be >= min_timeout_seconds")
    if lake is None and checkpoint_rows is None and transform_rows is None:
        raise ValueError("lake, checkpoint_rows, or transform_rows is required")
    samples = _lerobot_media_inspection_samples(
        lake,
        checkpoint_rows=checkpoint_rows,
        transform_rows=transform_rows,
        job_id=job_id,
        source_id=source_id,
        source_uri=source_uri,
        storage_tier=storage_tier,
        provider=provider,
    )
    return _lerobot_media_inspection_timeout_report(
        lake.uri if lake is not None else None,
        samples,
        filters={
            "job_id": job_id,
            "source_id": source_id,
            "source_uri": source_uri,
            "storage_tier": storage_tier,
            "provider": provider,
        },
        min_timeout_seconds=float(min_timeout_seconds),
        max_timeout_seconds=float(max_timeout_seconds),
    )


@dataclass(frozen=True)
class LeRobotClaimRecoveryReport:
    """Summary of an explicit stale LeRobot ingest claim recovery."""

    lake_uri: str
    job_id: str
    action: str
    status: str
    phase: str
    stale: bool
    force: bool
    previous_checkpoint_id: str
    previous_owner: str | None
    previous_token: str | None
    previous_updated_at: datetime | None
    previous_expires_at: datetime | None
    new_owner: str
    new_token: str
    recovery_checkpoint_id: str
    recovered_at: datetime

    def to_params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotClaimWatchdogFinding:
    """Read-only stale-claim watchdog finding for one latest LeRobot job row."""

    job_id: str
    source_id: str
    status: str
    phase: str
    stale: bool
    stale_reason: str
    lease_state: str
    checkpoint_id: str
    checkpoint_index: int
    claim_owner: str | None
    claim_token: str | None
    claim_generation: int | None
    last_heartbeat_at: datetime | None
    updated_at: datetime | None
    claim_expires_at: datetime | None
    expiration_source: str
    stale_seconds: float
    seconds_until_stale: float
    rows_seen: int
    observations_written: int
    episodes_written: int
    scenarios_written: int
    videos_written: int
    video_encodings_written: int
    rows_skipped_existing: int
    bytes_scanned: int
    suggested_recovery_command: str | None = None

    def to_params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotClaimWatchdogReport:
    """Dry-run watchdog report for running and inactive LeRobot ingest claims."""

    lake_uri: str
    generated_at: datetime
    source_id: str | None
    stale_after_seconds: float
    recovery_action: str
    new_owner: str | None
    stale_claims: tuple[LeRobotClaimWatchdogFinding, ...]
    live_claims: tuple[LeRobotClaimWatchdogFinding, ...]
    inactive_jobs: tuple[LeRobotClaimWatchdogFinding, ...]

    @property
    def stale_count(self) -> int:
        return len(self.stale_claims)

    @property
    def live_count(self) -> int:
        return len(self.live_claims)

    @property
    def inactive_count(self) -> int:
        return len(self.inactive_jobs)

    @property
    def total_jobs(self) -> int:
        return self.stale_count + self.live_count + self.inactive_count

    @property
    def has_stale(self) -> bool:
        return bool(self.stale_claims)

    def to_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stale_count"] = self.stale_count
        payload["live_count"] = self.live_count
        payload["inactive_count"] = self.inactive_count
        payload["total_jobs"] = self.total_jobs
        payload["has_stale"] = self.has_stale
        return payload


def watch_lerobot_ingest_claims(
    lake: Lake,
    *,
    source_id: str | None = None,
    stale_after: timedelta | int | float | None = DEFAULT_LEROBOT_CLAIM_LEASE,
    recovery_action: str = "abandon",
    new_owner: str | None = None,
    created_by: str = "lancedb-robotics",
    now: datetime | None = None,
) -> LeRobotClaimWatchdogReport:
    """Scan latest LeRobot ingest claims and produce a non-mutating recovery plan."""

    normalized_action = recovery_action.strip().lower()
    if normalized_action not in {"abandon", "steal"}:
        raise ValueError("recovery_action must be one of: abandon, steal")
    stale_delta = _normalize_lerobot_duration(
        stale_after,
        name="stale_after",
        default=DEFAULT_LEROBOT_CLAIM_LEASE,
    )
    reference = _coerce_lerobot_utc(now) or datetime.now(UTC)

    latest_by_job: dict[str, dict[str, Any]] = {}
    for row in _lerobot_checkpoint_rows(lake):
        job_id = str(row.get("job_id") or "")
        if not job_id:
            continue
        if source_id is not None and row.get("source_id") != source_id:
            continue
        current = latest_by_job.get(job_id)
        if current is None or _lerobot_checkpoint_sort_key(row) > _lerobot_checkpoint_sort_key(
            current
        ):
            latest_by_job[job_id] = row

    stale_claims: list[LeRobotClaimWatchdogFinding] = []
    live_claims: list[LeRobotClaimWatchdogFinding] = []
    inactive_jobs: list[LeRobotClaimWatchdogFinding] = []
    for row in sorted(latest_by_job.values(), key=_lerobot_checkpoint_sort_key, reverse=True):
        finding = _lerobot_claim_watchdog_finding(
            lake_uri=lake.uri,
            row=row,
            stale_after=stale_delta,
            recovery_action=normalized_action,
            new_owner=new_owner,
            created_by=created_by,
            now=reference,
        )
        if finding.status == "running" and finding.stale:
            stale_claims.append(finding)
        elif finding.status == "running":
            live_claims.append(finding)
        else:
            inactive_jobs.append(finding)

    return LeRobotClaimWatchdogReport(
        lake_uri=lake.uri,
        generated_at=reference,
        source_id=source_id,
        stale_after_seconds=stale_delta.total_seconds(),
        recovery_action=normalized_action,
        new_owner=new_owner,
        stale_claims=tuple(stale_claims),
        live_claims=tuple(live_claims),
        inactive_jobs=tuple(inactive_jobs),
    )


def recover_lerobot_ingest_claim(
    lake: Lake,
    job_id: str,
    *,
    action: str = "abandon",
    new_owner: str | None = None,
    stale_after: timedelta | int | float | None = DEFAULT_LEROBOT_CLAIM_LEASE,
    force: bool = False,
    created_by: str = "lancedb-robotics",
    expected_latest_checkpoint_id: str | None = None,
    expected_latest_claim_token: str | None = None,
    expected_checkpoint_index: int | None = None,
    now: datetime | None = None,
) -> LeRobotClaimRecoveryReport:
    """Append an auditable recovery checkpoint for a stale running LeRobot claim.

    Normal ingest never steals a running claim automatically. Operators must call
    this function, or the matching CLI command, once they have decided a claim is
    stale enough to abandon or hand to a new worker.
    """
    normalized_action = action.strip().lower()
    if normalized_action not in {"abandon", "steal"}:
        raise ValueError("action must be one of: abandon, steal")
    recovery_owner = new_owner or created_by
    if not str(recovery_owner).strip():
        raise ValueError("new_owner/created_by must not be empty")
    reference = _coerce_lerobot_utc(now) or datetime.now(UTC)
    stale_delta = _normalize_lerobot_duration(
        stale_after,
        name="stale_after",
        default=DEFAULT_LEROBOT_CLAIM_LEASE,
    )

    latest = _latest_lerobot_ingest_job(lake, job_id)
    if latest is None:
        raise KeyError(f"no LeRobot ingest job {job_id!r}")
    _assert_lerobot_claim_precondition(
        job_id,
        latest,
        operation="recover",
        lake_uri=lake.uri,
        expected_latest_checkpoint_id=expected_latest_checkpoint_id,
        expected_latest_claim_token=expected_latest_claim_token,
        expected_checkpoint_index=expected_checkpoint_index,
    )
    if latest.get("status") != "running":
        raise AdapterError(
            f"LeRobot ingest job {job_id} is not running; latest status is {latest.get('status')!r}"
        )

    progress = _lerobot_progress_from_checkpoint_row(latest)
    previous_claim = _lerobot_claim_from_progress(progress)
    previous_owner = str(latest.get("claim_owner") or previous_claim.get("owner") or "") or None
    previous_token = str(latest.get("claim_token") or previous_claim.get("token") or "") or None
    previous_updated_at = _coerce_lerobot_utc(latest.get("updated_at"))
    previous_expires_at = _lerobot_claim_expires_at(latest, stale_after=stale_delta)
    stale = previous_expires_at is None or previous_expires_at <= reference
    if not stale and not force:
        raise AdapterError(
            f"LeRobot ingest job {job_id} claim is still live until "
            f"{previous_expires_at.isoformat()}; pass force=True only after "
            "operator confirmation"
        )

    checkpoint_index = _next_lerobot_checkpoint_index(lake, job_id)
    recovery_token = _lerobot_claim_token(job_id, str(recovery_owner), reference, suffix="recovery")
    recovery_phase = "claim-abandoned" if normalized_action == "abandon" else "claim-stolen"
    recovery_checkpoint_id = f"{job_id}:{int(checkpoint_index):08d}"
    progress["status"] = "abandoned"
    progress["claim_recovery"] = {
        "action": normalized_action,
        "force": bool(force),
        "stale": bool(stale),
        "stale_after_seconds": stale_delta.total_seconds(),
        "previous_checkpoint_id": latest.get("checkpoint_id"),
        "previous_owner": previous_owner,
        "previous_token": previous_token,
        "previous_updated_at": previous_updated_at.isoformat() if previous_updated_at else None,
        "previous_claim_expires_at": previous_expires_at.isoformat()
        if previous_expires_at
        else None,
        "new_owner": str(recovery_owner),
        "new_token": recovery_token,
        "recovered_at": reference.isoformat(),
    }
    progress["claim"] = {
        **previous_claim,
        "owner": str(recovery_owner),
        "token": recovery_token,
        "generation": int(previous_claim.get("generation") or 0) + 1,
        "active": False,
        "claim_expires_at": None,
        "last_heartbeat_at": reference.isoformat(),
        "recovered_at": reference.isoformat(),
        "previous_owner": previous_owner,
        "previous_token": previous_token,
    }
    _lerobot_claim_cas_supersede(
        lake,
        job_id=job_id,
        prior_checkpoint_id=str(latest.get("checkpoint_id") or ""),
        prior_claim_token=previous_token,
        new_checkpoint_id=recovery_checkpoint_id,
        operation="recover",
    )
    _append_lerobot_recovery_checkpoint(
        lake,
        latest,
        progress=progress,
        checkpoint_index=checkpoint_index,
        checkpoint_id=recovery_checkpoint_id,
        phase=recovery_phase,
        claim_owner=str(recovery_owner),
        claim_token=recovery_token,
        created_by=created_by,
        now=reference,
    )

    return LeRobotClaimRecoveryReport(
        lake_uri=lake.uri,
        job_id=job_id,
        action=normalized_action,
        status="abandoned",
        phase=recovery_phase,
        stale=stale,
        force=bool(force),
        previous_checkpoint_id=str(latest.get("checkpoint_id") or ""),
        previous_owner=previous_owner,
        previous_token=previous_token,
        previous_updated_at=previous_updated_at,
        previous_expires_at=previous_expires_at,
        new_owner=str(recovery_owner),
        new_token=recovery_token,
        recovery_checkpoint_id=recovery_checkpoint_id,
        recovered_at=reference,
    )


@dataclass(frozen=True)
class LeRobotCheckpointHoldReport:
    """Summary of a first-class LeRobot checkpoint hold catalog operation."""

    lake_uri: str
    hold_id: str
    action: str
    active: bool
    selector: dict[str, Any]
    checkpoint_ids: tuple[str, ...]
    retain_until: datetime | None = None
    legal_hold: bool = False
    audit_hold: bool = False
    promotion_hold: bool = False
    owner: str | None = None
    reason: str | None = None
    created_at: datetime | None = None
    released_at: datetime | None = None
    released_by: str | None = None

    def to_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["table"] = _LEROBOT_CHECKPOINT_HOLDS_TABLE
        return payload


def hold_lerobot_checkpoints(
    lake: Lake,
    *,
    checkpoint_id: str | None = None,
    job_id: str | None = None,
    source_id: str | None = None,
    hf_repo_id: str | None = None,
    requested_revision: str | None = None,
    resolved_revision: str | None = None,
    status: str | tuple[str, ...] | list[str] | None = None,
    updated_after: datetime | str | None = None,
    updated_before: datetime | str | None = None,
    retain_until: datetime | str | None = None,
    legal_hold: bool = False,
    audit_hold: bool = True,
    promotion_hold: bool = False,
    owner: str | None = None,
    reason: str | None = None,
    created_by: str = "lancedb-robotics",
    now: datetime | None = None,
) -> LeRobotCheckpointHoldReport:
    """Create a catalog-backed hold over matching LeRobot checkpoint rows.

    The hold stores both the selector and the checkpoint ids matched at creation.
    Retention resolves active selectors again at plan time, so a job/source/repo
    hold can continue to protect all currently matching checkpoint rows without
    requiring operators to create lineage artifacts manually.
    """

    reference = _coerce_lerobot_utc(now) or datetime.now(UTC)
    retain_until_dt = _coerce_lerobot_utc(retain_until)
    if retain_until is not None and retain_until_dt is None:
        raise ValueError(f"invalid retain_until timestamp: {retain_until!r}")
    if retain_until_dt is not None and retain_until_dt <= reference:
        raise ValueError("retain_until must be in the future")
    if not (retain_until_dt or legal_hold or audit_hold or promotion_hold):
        raise ValueError("a checkpoint hold requires retain_until or legal/audit/promotion hold")

    selector = _normalize_lerobot_checkpoint_hold_selector(
        checkpoint_id=checkpoint_id,
        job_id=job_id,
        source_id=source_id,
        hf_repo_id=hf_repo_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        status=status,
        updated_after=updated_after,
        updated_before=updated_before,
    )
    matched_rows = _select_lerobot_checkpoint_rows(lake, selector)
    if not matched_rows:
        raise ValueError("LeRobot checkpoint hold selector matched no checkpoint rows")
    checkpoint_ids = tuple(
        str(row.get("checkpoint_id") or "")
        for row in sorted(matched_rows, key=_lerobot_checkpoint_sort_key)
        if row.get("checkpoint_id")
    )
    hold_payload = {
        "selector": selector,
        "checkpoint_ids": checkpoint_ids,
        "retain_until": retain_until_dt.isoformat() if retain_until_dt else None,
        "legal_hold": bool(legal_hold),
        "audit_hold": bool(audit_hold),
        "promotion_hold": bool(promotion_hold),
        "owner": owner or "",
        "reason": reason or "",
        "created_at": reference.isoformat(),
        "created_by": created_by,
    }
    hold_id = (
        "lerobot-hold-"
        + hashlib.sha1(
            json.dumps(hold_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    )
    row = {
        "hold_id": hold_id,
        "selector_json": json.dumps(selector, sort_keys=True, separators=(",", ":")),
        "checkpoint_ids": list(checkpoint_ids),
        "job_id": selector.get("job_id"),
        "source_id": selector.get("source_id"),
        "hf_repo_id": selector.get("hf_repo_id"),
        "requested_revision": selector.get("requested_revision"),
        "resolved_revision": selector.get("resolved_revision"),
        "statuses": list(selector.get("statuses") or ()),
        "updated_after": _coerce_lerobot_utc(selector.get("updated_after")),
        "updated_before": _coerce_lerobot_utc(selector.get("updated_before")),
        "retain_until": retain_until_dt,
        "legal_hold": bool(legal_hold),
        "audit_hold": bool(audit_hold),
        "promotion_hold": bool(promotion_hold),
        "owner": owner,
        "reason": reason,
        "active": True,
        "released_at": None,
        "released_by": None,
        "created_by": created_by,
        "created_at": reference,
    }
    _upsert_lerobot_rows(
        lake,
        _LEROBOT_CHECKPOINT_HOLDS_TABLE,
        "hold_id",
        [row],
        LEROBOT_CHECKPOINT_HOLDS_SCHEMA,
    )
    return LeRobotCheckpointHoldReport(
        lake_uri=lake.uri,
        hold_id=hold_id,
        action="created",
        active=True,
        selector=selector,
        checkpoint_ids=checkpoint_ids,
        retain_until=retain_until_dt,
        legal_hold=bool(legal_hold),
        audit_hold=bool(audit_hold),
        promotion_hold=bool(promotion_hold),
        owner=owner,
        reason=reason,
        created_at=reference,
    )


def release_lerobot_checkpoint_hold(
    lake: Lake,
    hold_id: str,
    *,
    released_by: str = "lancedb-robotics",
    now: datetime | None = None,
) -> LeRobotCheckpointHoldReport:
    """Release a catalog-backed LeRobot checkpoint hold without deleting audit history."""

    normalized = str(hold_id).strip()
    if not normalized:
        raise ValueError("hold_id must not be empty")
    reference = _coerce_lerobot_utc(now) or datetime.now(UTC)
    rows = _lerobot_checkpoint_hold_rows(lake)
    row = next((item for item in rows if item.get("hold_id") == normalized), None)
    if row is None:
        raise KeyError(f"no LeRobot checkpoint hold {normalized!r}")

    selector = _lerobot_checkpoint_hold_selector(row)
    checkpoint_ids = tuple(str(value) for value in row.get("checkpoint_ids") or () if value)
    row = dict(row)
    row["active"] = False
    row["released_at"] = reference
    row["released_by"] = released_by
    _upsert_lerobot_rows(
        lake,
        _LEROBOT_CHECKPOINT_HOLDS_TABLE,
        "hold_id",
        [row],
        LEROBOT_CHECKPOINT_HOLDS_SCHEMA,
    )
    return LeRobotCheckpointHoldReport(
        lake_uri=lake.uri,
        hold_id=normalized,
        action="released",
        active=False,
        selector=selector,
        checkpoint_ids=checkpoint_ids,
        retain_until=_coerce_lerobot_utc(row.get("retain_until")),
        legal_hold=bool(row.get("legal_hold")),
        audit_hold=bool(row.get("audit_hold")),
        promotion_hold=bool(row.get("promotion_hold")),
        owner=row.get("owner"),
        reason=row.get("reason"),
        created_at=_coerce_lerobot_utc(row.get("created_at")),
        released_at=reference,
        released_by=released_by,
    )


@dataclass(frozen=True)
class LeRobotCheckpointRetentionJobReport:
    """Retention decision for one durable LeRobot ingest job."""

    job_id: str
    source_id: str
    status: str
    phase: str
    reason: str
    rows_before: int
    rows_after: int
    rows_deleted: int
    terminal_checkpoint_id: str | None = None
    retained_checkpoint_ids: tuple[str, ...] = ()
    deleted_checkpoint_ids: tuple[str, ...] = ()
    hold_ids: tuple[str, ...] = ()
    hold_reasons: tuple[str, ...] = ()

    def to_params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotCheckpointRetentionReport:
    """Summary of one LeRobot checkpoint retention pass."""

    lake_uri: str
    dry_run: bool
    source_id: str | None
    statuses: tuple[str, ...]
    older_than_seconds: float | None
    retain_completed_per_source: int
    retain_failed_per_source: int
    rows_before: int
    rows_after: int
    rows_deleted: int
    version_before: int
    version_after: int
    fragments_before: int
    fragments_after: int
    jobs_seen: int
    jobs_compacted: int
    jobs_protected: int
    protected_checkpoint_ids: tuple[str, ...] = ()
    jobs: tuple[LeRobotCheckpointRetentionJobReport, ...] = ()
    compaction: dict[str, int] | None = None
    cleanup: dict[str, int] | None = None
    warnings: tuple[str, ...] = ()

    def to_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["table"] = _LEROBOT_CHECKPOINT_TABLE
        return payload


@dataclass(frozen=True)
class LeRobotCheckpointRetentionScheduleReport:
    """Machine-readable report for an automation-triggered checkpoint retention run."""

    lake_uri: str
    schedule_id: str
    dry_run: bool
    interval_seconds: float | None
    started_at: datetime
    finished_at: datetime
    next_run_after: datetime | None
    thresholds: dict[str, int | None]
    telemetry: dict[str, Any]
    alerts: tuple[dict[str, Any], ...]
    retention: LeRobotCheckpointRetentionReport

    def to_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retention"] = self.retention.to_params()
        payload["table"] = _LEROBOT_CHECKPOINT_TABLE
        return payload


@dataclass(frozen=True)
class LeRobotCheckpointRetentionPolicyProjection:
    """Projected outcome for one LeRobot checkpoint retention policy."""

    name: str
    description: str
    recommended_for: tuple[str, ...]
    retention_enabled: bool
    older_than_seconds: float | None
    retain_completed_per_source: int
    retain_failed_per_source: int
    source_id: str | None
    statuses: tuple[str, ...]
    rows_before: int
    rows_after: int
    rows_deleted: int
    version_before: int
    estimated_version_after: int
    estimated_version_delta: int
    fragments_before: int
    estimated_fragments_after: int
    jobs_seen: int
    jobs_compacted: int
    jobs_protected: int
    protected_jobs_by_reason: dict[str, int]
    hold_protected_jobs: int = 0
    warnings: tuple[str, ...] = ()

    def to_params(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeRobotCheckpointRetentionScalePlanReport:
    """Non-mutating LeRobot checkpoint retention scale plan and recommendations."""

    lake_uri: str | None
    mode: str
    scenario: str
    recommended_policy: str
    generated_at: datetime
    synthetic: dict[str, Any] | None
    policies: tuple[LeRobotCheckpointRetentionPolicyProjection, ...]

    def to_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["table"] = _LEROBOT_CHECKPOINT_TABLE
        return payload


@dataclass(frozen=True)
class LeRobotClaimRecoveryChaosCrashReport:
    """Deterministic recovery simulation for one LeRobot ingest crash point."""

    crash_point: str
    description: str
    recovery_required: bool
    checkpoint_rows_before: int
    recovery_checkpoint_rows: int
    retry_checkpoint_rows: int
    checkpoint_rows_after: int
    recovery_latency_seconds: float
    observations_before_retry: int
    observations_replayed: int
    observations_written_after_retry: int
    rows_skipped_existing: int
    accepted_recoveries: int
    cas_conflicts: int
    checkpoint_duplicate_rows: int
    duplicate_rows: dict[str, int]
    protections: dict[str, str]

    @property
    def passed(self) -> bool:
        """True only if BOTH axes are clean.

        ``duplicate_rows`` (observations/episodes/videos/events/runs/
        transform_runs) describes structural idempotency protections
        (deterministic ids, merge_insert) that are exercised by real
        end-to-end tests elsewhere (e.g.
        ``test_ingest_lerobot_failed_checkpoint_can_resume_without_duplicates``),
        not by this simulation call -- it is a claim about existing
        production code, not something re-verified per invocation.
        ``checkpoint_duplicate_rows`` is different: it is the REAL measured
        outcome of racing ``retry_owner_count`` concurrent recovery attempts
        against a real scratch lake (see
        ``_lerobot_claim_chaos_real_rehearsal``), not a model.
        """
        return self.checkpoint_duplicate_rows == 0 and all(
            int(count) == 0 for count in self.duplicate_rows.values()
        )

    def to_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True)
class LeRobotClaimRecoveryChaosReport:
    """Non-mutating LeRobot claim recovery chaos/scale simulation report."""

    lake_uri: str | None
    mode: str
    scenario: str
    generated_at: datetime
    seed: int
    source_id: str | None
    workload: dict[str, Any]
    watchdog: dict[str, Any]
    recovery: dict[str, Any]
    recommendations: dict[str, Any]
    duplicate_protection: dict[str, Any]
    crash_points: tuple[LeRobotClaimRecoveryChaosCrashReport, ...]
    retention_plan: LeRobotCheckpointRetentionScalePlanReport
    warnings: tuple[dict[str, Any], ...] = ()

    @property
    def passed(self) -> bool:
        return all(crash.passed for crash in self.crash_points)

    def to_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["crash_points"] = [crash.to_params() for crash in self.crash_points]
        payload["retention_plan"] = self.retention_plan.to_params()
        payload["passed"] = self.passed
        payload["table"] = _LEROBOT_CHECKPOINT_TABLE
        return payload


def apply_lerobot_checkpoint_retention(
    lake: Lake,
    *,
    older_than: timedelta | None = DEFAULT_LEROBOT_CHECKPOINT_RETENTION_AGE,
    statuses: tuple[str, ...] | list[str] | None = None,
    source_id: str | None = None,
    retain_completed_per_source: int = DEFAULT_LEROBOT_CHECKPOINT_RETAIN_COMPLETED_PER_SOURCE,
    retain_failed_per_source: int = DEFAULT_LEROBOT_CHECKPOINT_RETAIN_FAILED_PER_SOURCE,
    dry_run: bool = False,
    compact: bool = True,
    cleanup_older_than: timedelta | None = timedelta(days=7),
    retain_versions: int | None = DEFAULT_INGEST_RETAIN_VERSIONS,
    now: datetime | None = None,
) -> LeRobotCheckpointRetentionReport:
    """Compact old LeRobot ingest checkpoint histories into terminal summaries.

    The policy never removes the final evidence row for a terminal job. Eligible
    old completed/failed/skipped jobs keep their latest terminal checkpoint and
    drop only the high-frequency claim/media/batch/metadata progress rows. Running
    jobs, recently updated terminal jobs, the newest terminal histories per
    source/status, and checkpoints under active lineage retention holds stay fully
    expanded.
    """

    selected_statuses = _normalize_lerobot_retention_statuses(statuses)
    if older_than is not None and older_than < timedelta(0):
        raise ValueError("older_than must be non-negative or None")
    if retain_completed_per_source < 0:
        raise ValueError("retain_completed_per_source must be >= 0")
    if retain_failed_per_source < 0:
        raise ValueError("retain_failed_per_source must be >= 0")

    reference = _coerce_lerobot_utc(now) or datetime.now(UTC)
    rows = sorted(_lerobot_checkpoint_rows(lake), key=_lerobot_checkpoint_sort_key)
    rows_before = len(rows)
    table = lake.table(_LEROBOT_CHECKPOINT_TABLE)
    version_before = int(table.version)
    fragments_before = _lerobot_checkpoint_fragment_count(lake)
    hold_details_by_checkpoint = _active_lerobot_checkpoint_hold_details(lake, now=reference)
    protected_checkpoint_ids = set(hold_details_by_checkpoint)

    rows_by_job: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_job.setdefault(str(row.get("job_id") or ""), []).append(row)

    minimum_history_jobs = _minimum_lerobot_checkpoint_history_jobs(
        rows_by_job,
        statuses=selected_statuses,
        source_id=source_id,
        retain_completed_per_source=retain_completed_per_source,
        retain_failed_per_source=retain_failed_per_source,
    )
    job_reports: list[LeRobotCheckpointRetentionJobReport] = []
    delete_ids: list[str] = []

    for job_id, job_rows in sorted(rows_by_job.items()):
        ordered = sorted(job_rows, key=_lerobot_checkpoint_sort_key)
        latest = ordered[-1]
        latest_status = str(latest.get("status") or "")
        latest_source_id = str(latest.get("source_id") or "")
        checkpoint_ids = tuple(str(row.get("checkpoint_id") or "") for row in ordered)
        terminal = _terminal_lerobot_checkpoint_row(ordered, selected_statuses)
        terminal_checkpoint_id = str((terminal or latest).get("checkpoint_id") or "") or None
        retained_ids = checkpoint_ids
        deleted_for_job: tuple[str, ...] = ()
        reason = "protected"
        matched_hold_details = tuple(
            detail
            for checkpoint_id in checkpoint_ids
            for detail in hold_details_by_checkpoint.get(checkpoint_id, ())
        )
        hold_ids = tuple(
            sorted(
                {
                    str(detail.get("hold_id") or "")
                    for detail in matched_hold_details
                    if detail.get("hold_id")
                }
            )
        )
        hold_reasons = tuple(
            sorted(
                {
                    str(detail.get("reason") or "")
                    for detail in matched_hold_details
                    if detail.get("reason")
                }
            )
        )

        if source_id is not None and latest_source_id != source_id:
            reason = "source-filter"
        elif latest_status not in _LEROBOT_TERMINAL_STATUSES:
            reason = "active"
        elif latest_status not in selected_statuses:
            reason = "status-filter"
        elif matched_hold_details:
            reason = "retention-hold"
        elif job_id in minimum_history_jobs:
            reason = "minimum-history"
        elif older_than is not None and reference - _lerobot_checkpoint_time(latest) < older_than:
            reason = "recent"
        else:
            summary = terminal or latest
            summary_id = str(summary.get("checkpoint_id") or "")
            retained_ids = (summary_id,) if summary_id else ()
            deleted_for_job = tuple(
                checkpoint_id
                for checkpoint_id in checkpoint_ids
                if checkpoint_id and checkpoint_id != summary_id
            )
            reason = "terminal-summary" if deleted_for_job else "already-summary"
            delete_ids.extend(deleted_for_job)

        job_reports.append(
            LeRobotCheckpointRetentionJobReport(
                job_id=job_id,
                source_id=latest_source_id,
                status=latest_status,
                phase=str(latest.get("phase") or ""),
                reason=reason,
                rows_before=len(ordered),
                rows_after=len(retained_ids),
                rows_deleted=len(deleted_for_job),
                terminal_checkpoint_id=terminal_checkpoint_id,
                retained_checkpoint_ids=retained_ids,
                deleted_checkpoint_ids=deleted_for_job,
                hold_ids=hold_ids,
                hold_reasons=hold_reasons,
            )
        )

    compaction = None
    cleanup = None
    if dry_run:
        rows_after = rows_before - len(delete_ids)
        version_after = version_before
        fragments_after = fragments_before
    else:
        _delete_lerobot_checkpoint_ids(lake, delete_ids)
        if compact:
            checkpoint_table = lake.table(_LEROBOT_CHECKPOINT_TABLE)
            compaction = _lerobot_metric_dict(
                checkpoint_table.to_lance().optimize.compact_files(),
                ("fragments_removed", "fragments_added", "files_removed", "files_added"),
            )
        if cleanup_older_than is not None or retain_versions is not None:
            cleanup = _lerobot_metric_dict(
                lake.table(_LEROBOT_CHECKPOINT_TABLE)
                .to_lance()
                .cleanup_old_versions(
                    older_than=cleanup_older_than,
                    retain_versions=retain_versions,
                    error_if_tagged_old_versions=False,
                ),
                (
                    "bytes_removed",
                    "old_versions",
                    "data_files_removed",
                    "transaction_files_removed",
                    "index_files_removed",
                    "deletion_files_removed",
                ),
            )
        rows_after = int(lake.table(_LEROBOT_CHECKPOINT_TABLE).count_rows())
        version_after = int(lake.table(_LEROBOT_CHECKPOINT_TABLE).version)
        fragments_after = _lerobot_checkpoint_fragment_count(lake)

    return LeRobotCheckpointRetentionReport(
        lake_uri=lake.uri,
        dry_run=dry_run,
        source_id=source_id,
        statuses=selected_statuses,
        older_than_seconds=older_than.total_seconds() if older_than is not None else None,
        retain_completed_per_source=retain_completed_per_source,
        retain_failed_per_source=retain_failed_per_source,
        rows_before=rows_before,
        rows_after=rows_after,
        rows_deleted=rows_before - rows_after if not dry_run else len(delete_ids),
        version_before=version_before,
        version_after=version_after,
        fragments_before=fragments_before,
        fragments_after=fragments_after,
        jobs_seen=len(rows_by_job),
        jobs_compacted=sum(1 for item in job_reports if item.rows_deleted),
        jobs_protected=sum(1 for item in job_reports if not item.rows_deleted),
        protected_checkpoint_ids=tuple(sorted(protected_checkpoint_ids)),
        jobs=tuple(job_reports),
        compaction=compaction,
        cleanup=cleanup,
    )


def run_lerobot_checkpoint_retention_schedule(
    lake: Lake,
    *,
    schedule_id: str = "lerobot-checkpoint-retention",
    interval: timedelta | None = timedelta(days=1),
    older_than: timedelta | None = DEFAULT_LEROBOT_CHECKPOINT_RETENTION_AGE,
    statuses: tuple[str, ...] | list[str] | None = None,
    source_id: str | None = None,
    retain_completed_per_source: int = DEFAULT_LEROBOT_CHECKPOINT_RETAIN_COMPLETED_PER_SOURCE,
    retain_failed_per_source: int = DEFAULT_LEROBOT_CHECKPOINT_RETAIN_FAILED_PER_SOURCE,
    dry_run: bool = True,
    compact: bool = True,
    cleanup_older_than: timedelta | None = timedelta(days=7),
    retain_versions: int | None = DEFAULT_INGEST_RETAIN_VERSIONS,
    max_rows: int | None = None,
    max_rows_per_source: int | None = None,
    max_version_delta: int | None = None,
    now: datetime | None = None,
) -> LeRobotCheckpointRetentionScheduleReport:
    """Run one scheduled LeRobot checkpoint retention pass and emit telemetry.

    This is a scheduler-friendly hook rather than a long-running scheduler. Cron,
    systemd timers, Kubernetes CronJobs, or benchmark automation can call it once
    per cadence and route the structured report into dashboards or alerting.
    """

    normalized_schedule_id = str(schedule_id or "").strip()
    if not normalized_schedule_id:
        raise ValueError("schedule_id must not be empty")
    if interval is not None and interval < timedelta(0):
        raise ValueError("interval must be non-negative or None")
    thresholds = {
        "max_rows": _optional_non_negative_int("max_rows", max_rows),
        "max_rows_per_source": _optional_non_negative_int(
            "max_rows_per_source",
            max_rows_per_source,
        ),
        "max_version_delta": _optional_non_negative_int(
            "max_version_delta",
            max_version_delta,
        ),
    }

    started_at = _coerce_lerobot_utc(now) or datetime.now(UTC)
    retention = apply_lerobot_checkpoint_retention(
        lake,
        older_than=older_than,
        statuses=statuses,
        source_id=source_id,
        retain_completed_per_source=retain_completed_per_source,
        retain_failed_per_source=retain_failed_per_source,
        dry_run=dry_run,
        compact=compact,
        cleanup_older_than=cleanup_older_than,
        retain_versions=retain_versions,
        now=started_at,
    )
    finished_at = datetime.now(UTC)
    telemetry = _lerobot_checkpoint_retention_schedule_telemetry(retention)
    alerts = _lerobot_checkpoint_retention_schedule_alerts(
        telemetry,
        thresholds,
    )
    next_run_after = started_at + interval if interval is not None else None
    return LeRobotCheckpointRetentionScheduleReport(
        lake_uri=lake.uri,
        schedule_id=normalized_schedule_id,
        dry_run=dry_run,
        interval_seconds=interval.total_seconds() if interval is not None else None,
        started_at=started_at,
        finished_at=finished_at,
        next_run_after=next_run_after,
        thresholds=thresholds,
        telemetry=telemetry,
        alerts=alerts,
        retention=retention,
    )


def _optional_non_negative_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be >= 0")
    return normalized


def _lerobot_checkpoint_retention_schedule_telemetry(
    report: LeRobotCheckpointRetentionReport,
) -> dict[str, Any]:
    per_source: dict[str, dict[str, Any]] = {}
    reason_counts: dict[str, int] = {}
    for job in report.jobs:
        source_id = job.source_id or ""
        source = per_source.setdefault(
            source_id,
            {
                "jobs_seen": 0,
                "jobs_compacted": 0,
                "jobs_protected": 0,
                "rows_before": 0,
                "rows_after": 0,
                "rows_deleted": 0,
                "reason_counts": {},
            },
        )
        source["jobs_seen"] += 1
        source["rows_before"] += job.rows_before
        source["rows_after"] += job.rows_after
        source["rows_deleted"] += job.rows_deleted
        if job.rows_deleted:
            source["jobs_compacted"] += 1
        else:
            source["jobs_protected"] += 1
        source_reasons = source["reason_counts"]
        source_reasons[job.reason] = int(source_reasons.get(job.reason, 0)) + 1
        reason_counts[job.reason] = int(reason_counts.get(job.reason, 0)) + 1

    return {
        "rows_before": report.rows_before,
        "rows_after": report.rows_after,
        "rows_deleted": report.rows_deleted,
        "jobs_seen": report.jobs_seen,
        "jobs_compacted": report.jobs_compacted,
        "jobs_protected": report.jobs_protected,
        "reason_counts": dict(sorted(reason_counts.items())),
        "hold_protected_jobs": sum(1 for job in report.jobs if job.hold_ids),
        "version_before": report.version_before,
        "version_after": report.version_after,
        "version_delta": report.version_after - report.version_before,
        "fragments_before": report.fragments_before,
        "fragments_after": report.fragments_after,
        "fragments_delta": report.fragments_after - report.fragments_before,
        "per_source": {
            key: {
                **value,
                "reason_counts": dict(sorted(value["reason_counts"].items())),
            }
            for key, value in sorted(per_source.items())
        },
    }


def _lerobot_checkpoint_retention_schedule_alerts(
    telemetry: dict[str, Any],
    thresholds: dict[str, int | None],
) -> tuple[dict[str, Any], ...]:
    alerts: list[dict[str, Any]] = []
    max_rows = thresholds.get("max_rows")
    if max_rows is not None and int(telemetry["rows_after"]) > max_rows:
        alerts.append(
            {
                "level": "warning",
                "metric": "rows_after",
                "threshold": max_rows,
                "actual": int(telemetry["rows_after"]),
            }
        )
    max_rows_per_source = thresholds.get("max_rows_per_source")
    if max_rows_per_source is not None:
        for source_id, source in sorted(telemetry["per_source"].items()):
            if int(source["rows_after"]) > max_rows_per_source:
                alerts.append(
                    {
                        "level": "warning",
                        "metric": "rows_after_per_source",
                        "source_id": source_id,
                        "threshold": max_rows_per_source,
                        "actual": int(source["rows_after"]),
                    }
                )
    max_version_delta = thresholds.get("max_version_delta")
    if max_version_delta is not None and int(telemetry["version_delta"]) > max_version_delta:
        alerts.append(
            {
                "level": "warning",
                "metric": "version_delta",
                "threshold": max_version_delta,
                "actual": int(telemetry["version_delta"]),
            }
        )
    return tuple(alerts)


def plan_lerobot_checkpoint_retention_scale(
    lake: Lake | None = None,
    *,
    checkpoint_rows: Sequence[Mapping[str, Any]] | None = None,
    hold_details_by_checkpoint: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    scenario: str = "auto",
    source_id: str | None = None,
    statuses: tuple[str, ...] | list[str] | None = None,
    synthetic_sources: int | None = None,
    synthetic_completed_jobs_per_source: int = 0,
    synthetic_failed_jobs_per_source: int = 0,
    synthetic_running_jobs_per_source: int = 0,
    synthetic_checkpoints_per_job: int = 4,
    synthetic_terminal_age_days: float = 90.0,
    now: datetime | None = None,
) -> LeRobotCheckpointRetentionScalePlanReport:
    """Plan LeRobot checkpoint retention policies without mutating the lake.

    Pass ``lake`` to evaluate observed checkpoint rows through the same dry-run
    retention engine used by CLI/API retention. Pass ``checkpoint_rows`` when a
    caller already has observed rows and wants to plan without opening a lake.
    Omit both and provide ``synthetic_sources`` plus synthetic job counts to
    estimate growth before a large HF backfill exists.
    """

    input_count = sum(
        1
        for provided in (
            lake is not None,
            checkpoint_rows is not None,
            synthetic_sources is not None,
        )
        if provided
    )
    if input_count != 1:
        raise ValueError("provide exactly one of lake, checkpoint_rows, or synthetic_sources")
    if hold_details_by_checkpoint is not None and checkpoint_rows is None:
        raise ValueError("hold_details_by_checkpoint requires checkpoint_rows")
    generated_at = _coerce_lerobot_utc(now) or datetime.now(UTC)
    selected_statuses = _normalize_lerobot_retention_statuses(statuses)
    scenario_name = _normalize_lerobot_retention_plan_scenario(scenario)
    policies = _lerobot_retention_plan_policy_templates()
    if lake is not None:
        projections = tuple(
            _observed_lerobot_retention_policy_projection(
                lake,
                policy,
                statuses=selected_statuses,
                source_id=source_id,
                now=generated_at,
            )
            for policy in policies
        )
        mode = "observed"
        synthetic_payload = None
    elif checkpoint_rows is not None:
        rows = [dict(row) for row in checkpoint_rows]
        hold_details = {
            str(checkpoint_id): tuple(dict(detail) for detail in details)
            for checkpoint_id, details in (hold_details_by_checkpoint or {}).items()
        }
        projections = tuple(
            _row_backed_lerobot_retention_policy_projection(
                policy,
                rows=rows,
                hold_details_by_checkpoint=hold_details,
                statuses=selected_statuses,
                source_id=source_id,
                now=generated_at,
            )
            for policy in policies
        )
        mode = "checkpoint-rows"
        synthetic_payload = None
    else:
        synthetic_payload = _normalize_lerobot_retention_synthetic_payload(
            synthetic_sources=synthetic_sources,
            synthetic_completed_jobs_per_source=synthetic_completed_jobs_per_source,
            synthetic_failed_jobs_per_source=synthetic_failed_jobs_per_source,
            synthetic_running_jobs_per_source=synthetic_running_jobs_per_source,
            synthetic_checkpoints_per_job=synthetic_checkpoints_per_job,
            synthetic_terminal_age_days=synthetic_terminal_age_days,
        )
        projections = tuple(
            _synthetic_lerobot_retention_policy_projection(
                policy,
                statuses=selected_statuses,
                source_id=source_id,
                synthetic=synthetic_payload,
            )
            for policy in policies
        )
        mode = "synthetic"

    recommended_policy = _recommended_lerobot_retention_policy(
        scenario_name,
        projections,
    )
    return LeRobotCheckpointRetentionScalePlanReport(
        lake_uri=lake.uri if lake is not None else None,
        mode=mode,
        scenario=scenario_name,
        recommended_policy=recommended_policy,
        generated_at=generated_at,
        synthetic=synthetic_payload,
        policies=projections,
    )


def _lerobot_retention_plan_policy_templates() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "audit-window",
            "description": "Disable row summarization; use during legal/audit freezes.",
            "recommended_for": ("audit-window",),
            "retention_enabled": False,
            "older_than": None,
            "retain_completed_per_source": 10,
            "retain_failed_per_source": 10,
        },
        {
            "name": "local-smoke",
            "description": "Aggressive local smoke default with one expanded terminal history per source.",
            "recommended_for": ("local-smoke",),
            "retention_enabled": True,
            "older_than": timedelta(0),
            "retain_completed_per_source": 1,
            "retain_failed_per_source": 1,
        },
        {
            "name": "ci-disposable",
            "description": "Disposable CI policy; summarize all eligible terminal histories.",
            "recommended_for": ("ci",),
            "retention_enabled": True,
            "older_than": timedelta(0),
            "retain_completed_per_source": 0,
            "retain_failed_per_source": 0,
        },
        {
            "name": "mid-corpus",
            "description": "Mid-corpus backfill default with a short age gate and modest history floor.",
            "recommended_for": ("mid-corpus",),
            "retention_enabled": True,
            "older_than": timedelta(days=7),
            "retain_completed_per_source": 5,
            "retain_failed_per_source": 5,
        },
        {
            "name": "full-public-corpus",
            "description": "Conservative full-corpus default matching the shipped retention policy.",
            "recommended_for": ("full-public-corpus",),
            "retention_enabled": True,
            "older_than": DEFAULT_LEROBOT_CHECKPOINT_RETENTION_AGE,
            "retain_completed_per_source": DEFAULT_LEROBOT_CHECKPOINT_RETAIN_COMPLETED_PER_SOURCE,
            "retain_failed_per_source": DEFAULT_LEROBOT_CHECKPOINT_RETAIN_FAILED_PER_SOURCE,
        },
    )


def _normalize_lerobot_retention_plan_scenario(value: str) -> str:
    scenario = str(value or "auto").strip().lower().replace("_", "-")
    allowed = {
        "auto",
        "audit-window",
        "local-smoke",
        "ci",
        "mid-corpus",
        "full-public-corpus",
    }
    if scenario not in allowed:
        raise ValueError("scenario must be one of: " + ", ".join(sorted(allowed)))
    return scenario


def _observed_lerobot_retention_policy_projection(
    lake: Lake,
    policy: dict[str, Any],
    *,
    statuses: tuple[str, ...],
    source_id: str | None,
    now: datetime,
) -> LeRobotCheckpointRetentionPolicyProjection:
    if not policy["retention_enabled"]:
        rows = _lerobot_checkpoint_rows(lake)
        rows_by_job: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            rows_by_job.setdefault(str(row.get("job_id") or ""), []).append(row)
        version_before = int(lake.table(_LEROBOT_CHECKPOINT_TABLE).version)
        fragments_before = _lerobot_checkpoint_fragment_count(lake)
        return _disabled_lerobot_retention_policy_projection(
            policy,
            statuses=statuses,
            source_id=source_id,
            rows_before=len(rows),
            version_before=version_before,
            fragments_before=fragments_before,
            jobs_seen=len(rows_by_job),
        )

    report = apply_lerobot_checkpoint_retention(
        lake,
        older_than=policy["older_than"],
        statuses=statuses,
        source_id=source_id,
        retain_completed_per_source=policy["retain_completed_per_source"],
        retain_failed_per_source=policy["retain_failed_per_source"],
        dry_run=True,
        compact=False,
        cleanup_older_than=None,
        retain_versions=None,
        now=now,
    )
    return _projection_from_lerobot_retention_report(policy, report)


def _row_backed_lerobot_retention_policy_projection(
    policy: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    hold_details_by_checkpoint: Mapping[str, Sequence[Mapping[str, Any]]],
    statuses: tuple[str, ...],
    source_id: str | None,
    now: datetime,
) -> LeRobotCheckpointRetentionPolicyProjection:
    rows = sorted(rows, key=_lerobot_checkpoint_sort_key)
    rows_by_job: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_job.setdefault(str(row.get("job_id") or ""), []).append(row)

    version_before = 1
    fragments_before = _estimate_lerobot_fragments(len(rows))
    if not policy["retention_enabled"]:
        return _disabled_lerobot_retention_policy_projection(
            policy,
            statuses=statuses,
            source_id=source_id,
            rows_before=len(rows),
            version_before=version_before,
            fragments_before=fragments_before,
            jobs_seen=len(rows_by_job),
        )

    minimum_history_jobs = _minimum_lerobot_checkpoint_history_jobs(
        rows_by_job,
        statuses=statuses,
        source_id=source_id,
        retain_completed_per_source=int(policy["retain_completed_per_source"]),
        retain_failed_per_source=int(policy["retain_failed_per_source"]),
    )
    protected_checkpoint_ids = {
        str(checkpoint_id)
        for checkpoint_id, details in hold_details_by_checkpoint.items()
        if details
    }
    job_reports: list[LeRobotCheckpointRetentionJobReport] = []
    delete_ids: list[str] = []
    older_than = policy["older_than"]

    for job_id, job_rows in sorted(rows_by_job.items()):
        ordered = sorted(job_rows, key=_lerobot_checkpoint_sort_key)
        latest = ordered[-1]
        latest_status = str(latest.get("status") or "")
        latest_source_id = str(latest.get("source_id") or "")
        checkpoint_ids = tuple(str(row.get("checkpoint_id") or "") for row in ordered)
        terminal = _terminal_lerobot_checkpoint_row(ordered, statuses)
        terminal_checkpoint_id = str((terminal or latest).get("checkpoint_id") or "") or None
        retained_ids = checkpoint_ids
        deleted_for_job: tuple[str, ...] = ()
        matched_hold_details = tuple(
            detail
            for checkpoint_id in checkpoint_ids
            for detail in hold_details_by_checkpoint.get(checkpoint_id, ())
        )
        hold_ids = tuple(
            sorted(
                {
                    str(detail.get("hold_id") or "")
                    for detail in matched_hold_details
                    if detail.get("hold_id")
                }
            )
        )
        hold_reasons = tuple(
            sorted(
                {
                    str(detail.get("reason") or "")
                    for detail in matched_hold_details
                    if detail.get("reason")
                }
            )
        )

        if source_id is not None and latest_source_id != source_id:
            reason = "source-filter"
        elif latest_status not in _LEROBOT_TERMINAL_STATUSES:
            reason = "active"
        elif latest_status not in statuses:
            reason = "status-filter"
        elif matched_hold_details:
            reason = "retention-hold"
        elif job_id in minimum_history_jobs:
            reason = "minimum-history"
        elif older_than is not None and now - _lerobot_checkpoint_time(latest) < older_than:
            reason = "recent"
        else:
            summary = terminal or latest
            summary_id = str(summary.get("checkpoint_id") or "")
            retained_ids = (summary_id,) if summary_id else ()
            deleted_for_job = tuple(
                checkpoint_id
                for checkpoint_id in checkpoint_ids
                if checkpoint_id and checkpoint_id != summary_id
            )
            reason = "terminal-summary" if deleted_for_job else "already-summary"
            delete_ids.extend(deleted_for_job)

        job_reports.append(
            LeRobotCheckpointRetentionJobReport(
                job_id=job_id,
                source_id=latest_source_id,
                status=latest_status,
                phase=str(latest.get("phase") or ""),
                reason=reason,
                rows_before=len(ordered),
                rows_after=len(retained_ids),
                rows_deleted=len(deleted_for_job),
                terminal_checkpoint_id=terminal_checkpoint_id,
                retained_checkpoint_ids=retained_ids,
                deleted_checkpoint_ids=deleted_for_job,
                hold_ids=hold_ids,
                hold_reasons=hold_reasons,
            )
        )

    rows_after = len(rows) - len(delete_ids)
    report = LeRobotCheckpointRetentionReport(
        lake_uri="",
        dry_run=True,
        source_id=source_id,
        statuses=statuses,
        older_than_seconds=older_than.total_seconds() if older_than is not None else None,
        retain_completed_per_source=int(policy["retain_completed_per_source"]),
        retain_failed_per_source=int(policy["retain_failed_per_source"]),
        rows_before=len(rows),
        rows_after=rows_after,
        rows_deleted=len(delete_ids),
        version_before=version_before,
        version_after=version_before,
        fragments_before=fragments_before,
        fragments_after=_estimate_lerobot_fragments(rows_after),
        jobs_seen=len(rows_by_job),
        jobs_compacted=sum(1 for item in job_reports if item.rows_deleted),
        jobs_protected=sum(1 for item in job_reports if not item.rows_deleted),
        protected_checkpoint_ids=tuple(sorted(protected_checkpoint_ids)),
        jobs=tuple(job_reports),
    )
    return _projection_from_lerobot_retention_report(policy, report)


def _projection_from_lerobot_retention_report(
    policy: dict[str, Any],
    report: LeRobotCheckpointRetentionReport,
) -> LeRobotCheckpointRetentionPolicyProjection:
    reason_counts = _lerobot_retention_reason_counts(report.jobs)
    estimated_delta = 1 if report.rows_deleted else 0
    return LeRobotCheckpointRetentionPolicyProjection(
        name=policy["name"],
        description=policy["description"],
        recommended_for=tuple(policy["recommended_for"]),
        retention_enabled=bool(policy["retention_enabled"]),
        older_than_seconds=(
            policy["older_than"].total_seconds() if policy["older_than"] is not None else None
        ),
        retain_completed_per_source=int(policy["retain_completed_per_source"]),
        retain_failed_per_source=int(policy["retain_failed_per_source"]),
        source_id=report.source_id,
        statuses=report.statuses,
        rows_before=report.rows_before,
        rows_after=report.rows_after,
        rows_deleted=report.rows_deleted,
        version_before=report.version_before,
        estimated_version_after=report.version_before + estimated_delta,
        estimated_version_delta=estimated_delta,
        fragments_before=report.fragments_before,
        estimated_fragments_after=report.fragments_after,
        jobs_seen=report.jobs_seen,
        jobs_compacted=report.jobs_compacted,
        jobs_protected=report.jobs_protected,
        protected_jobs_by_reason=reason_counts,
        hold_protected_jobs=sum(1 for job in report.jobs if job.hold_ids),
        warnings=report.warnings,
    )


def _disabled_lerobot_retention_policy_projection(
    policy: dict[str, Any],
    *,
    statuses: tuple[str, ...],
    source_id: str | None,
    rows_before: int,
    version_before: int,
    fragments_before: int,
    jobs_seen: int,
) -> LeRobotCheckpointRetentionPolicyProjection:
    return LeRobotCheckpointRetentionPolicyProjection(
        name=policy["name"],
        description=policy["description"],
        recommended_for=tuple(policy["recommended_for"]),
        retention_enabled=False,
        older_than_seconds=None,
        retain_completed_per_source=int(policy["retain_completed_per_source"]),
        retain_failed_per_source=int(policy["retain_failed_per_source"]),
        source_id=source_id,
        statuses=statuses,
        rows_before=rows_before,
        rows_after=rows_before,
        rows_deleted=0,
        version_before=version_before,
        estimated_version_after=version_before,
        estimated_version_delta=0,
        fragments_before=fragments_before,
        estimated_fragments_after=fragments_before,
        jobs_seen=jobs_seen,
        jobs_compacted=0,
        jobs_protected=jobs_seen,
        protected_jobs_by_reason={"retention-disabled": jobs_seen} if jobs_seen else {},
    )


def _normalize_lerobot_retention_synthetic_payload(
    *,
    synthetic_sources: int | None,
    synthetic_completed_jobs_per_source: int,
    synthetic_failed_jobs_per_source: int,
    synthetic_running_jobs_per_source: int,
    synthetic_checkpoints_per_job: int,
    synthetic_terminal_age_days: float,
) -> dict[str, Any]:
    payload = {
        "sources": _optional_non_negative_int("synthetic_sources", synthetic_sources),
        "completed_jobs_per_source": _optional_non_negative_int(
            "synthetic_completed_jobs_per_source",
            synthetic_completed_jobs_per_source,
        ),
        "failed_jobs_per_source": _optional_non_negative_int(
            "synthetic_failed_jobs_per_source",
            synthetic_failed_jobs_per_source,
        ),
        "running_jobs_per_source": _optional_non_negative_int(
            "synthetic_running_jobs_per_source",
            synthetic_running_jobs_per_source,
        ),
        "checkpoints_per_job": _optional_non_negative_int(
            "synthetic_checkpoints_per_job",
            synthetic_checkpoints_per_job,
        ),
        "terminal_age_days": float(synthetic_terminal_age_days),
    }
    if payload["sources"] is None or payload["sources"] <= 0:
        raise ValueError("synthetic_sources must be > 0")
    if payload["checkpoints_per_job"] <= 0:
        raise ValueError("synthetic_checkpoints_per_job must be > 0")
    if payload["terminal_age_days"] < 0:
        raise ValueError("synthetic_terminal_age_days must be >= 0")
    return payload


def _synthetic_lerobot_retention_policy_projection(
    policy: dict[str, Any],
    *,
    statuses: tuple[str, ...],
    source_id: str | None,
    synthetic: dict[str, Any],
) -> LeRobotCheckpointRetentionPolicyProjection:
    sources = int(synthetic["sources"])
    completed = int(synthetic["completed_jobs_per_source"])
    failed = int(synthetic["failed_jobs_per_source"])
    running = int(synthetic["running_jobs_per_source"])
    checkpoints = int(synthetic["checkpoints_per_job"])
    terminal_age = timedelta(days=float(synthetic["terminal_age_days"]))
    total_jobs = sources * (completed + failed + running)
    rows_before = total_jobs * checkpoints
    fragments_before = _estimate_lerobot_fragments(rows_before)
    selected_sources = 1 if source_id is not None else sources
    source_filtered_sources = sources - selected_sources

    reason_counts: dict[str, int] = {}
    compacted_jobs = 0
    if not policy["retention_enabled"]:
        reason_counts["retention-disabled"] = total_jobs
    else:
        source_filtered_jobs = source_filtered_sources * (completed + failed + running)
        if source_filtered_jobs:
            reason_counts["source-filter"] = source_filtered_jobs
        eligible_by_age = policy["older_than"] is None or terminal_age >= policy["older_than"]
        if not eligible_by_age:
            reason_counts["recent"] = selected_sources * (completed + failed)
        else:
            if "completed" in statuses:
                compacted_completed = selected_sources * max(
                    0,
                    completed - int(policy["retain_completed_per_source"]),
                )
                kept_completed = selected_sources * min(
                    completed,
                    int(policy["retain_completed_per_source"]),
                )
                compacted_jobs += compacted_completed
                if compacted_completed:
                    reason_counts["terminal-summary"] = (
                        reason_counts.get("terminal-summary", 0) + compacted_completed
                    )
                if kept_completed:
                    reason_counts["minimum-history"] = (
                        reason_counts.get("minimum-history", 0) + kept_completed
                    )
            elif completed:
                reason_counts["status-filter"] = (
                    reason_counts.get("status-filter", 0) + selected_sources * completed
                )

            if "failed" in statuses:
                compacted_failed = selected_sources * max(
                    0,
                    failed - int(policy["retain_failed_per_source"]),
                )
                kept_failed = selected_sources * min(
                    failed,
                    int(policy["retain_failed_per_source"]),
                )
                compacted_jobs += compacted_failed
                if compacted_failed:
                    reason_counts["terminal-summary"] = (
                        reason_counts.get("terminal-summary", 0) + compacted_failed
                    )
                if kept_failed:
                    reason_counts["minimum-history"] = (
                        reason_counts.get("minimum-history", 0) + kept_failed
                    )
            elif failed:
                reason_counts["status-filter"] = (
                    reason_counts.get("status-filter", 0) + selected_sources * failed
                )

        if running:
            reason_counts["active"] = selected_sources * running

    rows_deleted = compacted_jobs * max(0, checkpoints - 1)
    rows_after = rows_before - rows_deleted
    estimated_delta = 1 if rows_deleted else 0
    return LeRobotCheckpointRetentionPolicyProjection(
        name=policy["name"],
        description=policy["description"],
        recommended_for=tuple(policy["recommended_for"]),
        retention_enabled=bool(policy["retention_enabled"]),
        older_than_seconds=(
            policy["older_than"].total_seconds() if policy["older_than"] is not None else None
        ),
        retain_completed_per_source=int(policy["retain_completed_per_source"]),
        retain_failed_per_source=int(policy["retain_failed_per_source"]),
        source_id=source_id,
        statuses=statuses,
        rows_before=rows_before,
        rows_after=rows_after,
        rows_deleted=rows_deleted,
        version_before=1,
        estimated_version_after=1 + estimated_delta,
        estimated_version_delta=estimated_delta,
        fragments_before=fragments_before,
        estimated_fragments_after=_estimate_lerobot_fragments(rows_after),
        jobs_seen=total_jobs,
        jobs_compacted=compacted_jobs,
        jobs_protected=total_jobs - compacted_jobs,
        protected_jobs_by_reason=dict(sorted(reason_counts.items())),
    )


def _estimate_lerobot_fragments(rows: int) -> int:
    if rows <= 0:
        return 0
    return max(1, (rows + 1023) // 1024)


def _lerobot_retention_reason_counts(
    jobs: tuple[LeRobotCheckpointRetentionJobReport, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.reason] = counts.get(job.reason, 0) + 1
    return dict(sorted(counts.items()))


def _recommended_lerobot_retention_policy(
    scenario: str,
    projections: tuple[LeRobotCheckpointRetentionPolicyProjection, ...],
) -> str:
    if scenario == "auto":
        rows_before = projections[0].rows_before if projections else 0
        if rows_before <= 1_000:
            scenario = "local-smoke"
        elif rows_before <= 100_000:
            scenario = "mid-corpus"
        else:
            scenario = "full-public-corpus"
    for projection in projections:
        if scenario in projection.recommended_for:
            return projection.name
    return projections[0].name if projections else ""


def simulate_lerobot_claim_recovery_chaos(
    lake: Lake | None = None,
    *,
    checkpoint_rows: Sequence[Mapping[str, Any]] | None = None,
    scenario: str = "auto",
    source_id: str | None = None,
    recovery_action: str = "abandon",
    new_owner: str | None = None,
    stale_after: timedelta | int | float | None = DEFAULT_LEROBOT_CLAIM_LEASE,
    claim_heartbeat_interval: timedelta | int | float | None = DEFAULT_LEROBOT_CLAIM_HEARTBEAT,
    source_size_frames: int = 4096,
    episode_count: int | None = None,
    camera_count: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retry_owner_count: int = 2,
    remote_latency_ms: float = 0.0,
    synthetic_sources: int | None = None,
    synthetic_completed_jobs_per_source: int = 0,
    synthetic_failed_jobs_per_source: int = 0,
    synthetic_running_jobs_per_source: int = 1,
    synthetic_checkpoints_per_job: int = 4,
    synthetic_terminal_age_days: float = 90.0,
    synthetic_stale_running_fraction: float = 0.25,
    synthetic_missing_lease_fraction: float = 0.0,
    seed: int = 0,
    now: datetime | None = None,
) -> LeRobotClaimRecoveryChaosReport:
    """Simulate LeRobot stale-claim recovery behavior without mutating a lake.

    The workload/watchdog/retention modeling is deterministic. It models the
    durable checkpoint timeline and the idempotent retry contracts that real
    LeRobot ingest uses: observations are skipped by deterministic
    ``observation_id`` and metadata tables are upserted by canonical ids. Pass
    a lake for observed checkpoint rows, pass ``checkpoint_rows`` for an
    offline export, or pass ``synthetic_sources`` plus synthetic counts before
    a large backfill exists.

    The CAS-race outcome is different: rather than modeling it, each call
    actually races ``retry_owner_count`` concurrent recovery attempts against
    a disposable scratch lake (never your ``lake``/``checkpoint_rows``/
    ``synthetic_sources`` input -- see ``_lerobot_claim_chaos_real_rehearsal``)
    and reports the real measured outcome. Exactly one winner and zero
    checkpoint duplicates has been verified reliably across many trials, but
    it is a property of the real CAS guard under real execution, not a
    mathematical guarantee restated here -- a report can, in principle,
    surface a duplicate if it ever happens.
    """

    input_count = sum(
        1
        for provided in (
            lake is not None,
            checkpoint_rows is not None,
            synthetic_sources is not None,
        )
        if provided
    )
    if input_count != 1:
        raise ValueError("provide exactly one of lake, checkpoint_rows, or synthetic_sources")
    action = str(recovery_action or "").strip().lower()
    if action not in {"abandon", "steal"}:
        raise ValueError("recovery_action must be one of: abandon, steal")
    scenario_name = _normalize_lerobot_retention_plan_scenario(scenario)
    generated_at = _coerce_lerobot_utc(now) or datetime.now(UTC)
    stale_delta = _normalize_lerobot_duration(
        stale_after,
        name="stale_after",
        default=DEFAULT_LEROBOT_CLAIM_LEASE,
    )
    heartbeat_delta = _normalize_lerobot_duration(
        claim_heartbeat_interval,
        name="claim_heartbeat_interval",
        default=DEFAULT_LEROBOT_CLAIM_HEARTBEAT,
    )
    frame_count = _positive_lerobot_int("source_size_frames", source_size_frames)
    normalized_episode_count = (
        min(frame_count, max(1, frame_count))
        if episode_count is None
        else _positive_lerobot_int("episode_count", episode_count)
    )
    normalized_camera_count = _positive_lerobot_int("camera_count", camera_count)
    normalized_batch_size = _positive_lerobot_int("batch_size", batch_size)
    normalized_retry_owner_count = _positive_lerobot_int("retry_owner_count", retry_owner_count)
    normalized_latency_ms = float(remote_latency_ms)
    if normalized_latency_ms < 0:
        raise ValueError("remote_latency_ms must be >= 0")

    if lake is not None:
        rows = _lerobot_checkpoint_rows(lake)
        mode = "observed"
        lake_uri = lake.uri
        retention_plan = plan_lerobot_checkpoint_retention_scale(
            lake,
            scenario=scenario_name,
            source_id=source_id,
            now=generated_at,
        )
        workload, watchdog = _lerobot_claim_chaos_from_rows(
            rows,
            lake_uri=lake.uri,
            source_id=source_id,
            stale_after=stale_delta,
            recovery_action=action,
            new_owner=new_owner,
            now=generated_at,
        )
    elif checkpoint_rows is not None:
        rows = [dict(row) for row in checkpoint_rows]
        mode = "checkpoint-rows"
        lake_uri = None
        retention_plan = plan_lerobot_checkpoint_retention_scale(
            checkpoint_rows=rows,
            scenario=scenario_name,
            source_id=source_id,
            now=generated_at,
        )
        workload, watchdog = _lerobot_claim_chaos_from_rows(
            rows,
            lake_uri="<checkpoint-rows>",
            source_id=source_id,
            stale_after=stale_delta,
            recovery_action=action,
            new_owner=new_owner,
            now=generated_at,
        )
    else:
        synthetic = _normalize_lerobot_claim_chaos_synthetic_payload(
            synthetic_sources=synthetic_sources,
            synthetic_completed_jobs_per_source=synthetic_completed_jobs_per_source,
            synthetic_failed_jobs_per_source=synthetic_failed_jobs_per_source,
            synthetic_running_jobs_per_source=synthetic_running_jobs_per_source,
            synthetic_checkpoints_per_job=synthetic_checkpoints_per_job,
            synthetic_terminal_age_days=synthetic_terminal_age_days,
            synthetic_stale_running_fraction=synthetic_stale_running_fraction,
            synthetic_missing_lease_fraction=synthetic_missing_lease_fraction,
        )
        mode = "synthetic"
        lake_uri = None
        retention_plan = plan_lerobot_checkpoint_retention_scale(
            scenario=scenario_name,
            source_id=source_id,
            synthetic_sources=synthetic["sources"],
            synthetic_completed_jobs_per_source=synthetic["completed_jobs_per_source"],
            synthetic_failed_jobs_per_source=synthetic["failed_jobs_per_source"],
            synthetic_running_jobs_per_source=synthetic["running_jobs_per_source"],
            synthetic_checkpoints_per_job=synthetic["checkpoints_per_job"],
            synthetic_terminal_age_days=synthetic["terminal_age_days"],
            now=generated_at,
        )
        workload, watchdog = _lerobot_claim_chaos_from_synthetic(
            synthetic,
            source_id=source_id,
            recovery_action=action,
            new_owner=new_owner,
            stale_after=stale_delta,
            now=generated_at,
            seed=int(seed),
        )

    crash_points = _lerobot_claim_chaos_crash_points(
        frame_count=frame_count,
        episode_count=normalized_episode_count,
        camera_count=normalized_camera_count,
        batch_size=normalized_batch_size,
        retry_owner_count=normalized_retry_owner_count,
        stale_after=stale_delta,
        heartbeat=heartbeat_delta,
        remote_latency_ms=normalized_latency_ms,
        seed=int(seed),
    )
    recovery = _lerobot_claim_chaos_recovery_summary(
        watchdog,
        retry_owner_count=normalized_retry_owner_count,
        stale_after=stale_delta,
        heartbeat=heartbeat_delta,
        batch_size=normalized_batch_size,
        frame_count=frame_count,
        remote_latency_ms=normalized_latency_ms,
    )
    recommendations = _lerobot_claim_chaos_recommendations(
        scenario_name,
        workload,
        configured_lease=stale_delta,
        configured_heartbeat=heartbeat_delta,
        remote_latency_ms=normalized_latency_ms,
    )
    duplicate_protection = _lerobot_claim_chaos_duplicate_protection(
        frame_count=frame_count,
        episode_count=normalized_episode_count,
        camera_count=normalized_camera_count,
    )
    warnings = _lerobot_claim_chaos_warnings(
        workload,
        watchdog,
        recovery,
        recommendations,
        crash_points,
        configured_lease=stale_delta,
        configured_heartbeat=heartbeat_delta,
    )
    return LeRobotClaimRecoveryChaosReport(
        lake_uri=lake_uri,
        mode=mode,
        scenario=scenario_name,
        generated_at=generated_at,
        seed=int(seed),
        source_id=source_id,
        workload=workload,
        watchdog=watchdog,
        recovery=recovery,
        recommendations=recommendations,
        duplicate_protection=duplicate_protection,
        crash_points=crash_points,
        retention_plan=retention_plan,
        warnings=warnings,
    )


def _positive_lerobot_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be > 0")
    return normalized


def _normalize_lerobot_fraction(name: str, value: float) -> float:
    normalized = float(value)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return normalized


def _normalize_lerobot_claim_chaos_synthetic_payload(
    *,
    synthetic_sources: int | None,
    synthetic_completed_jobs_per_source: int,
    synthetic_failed_jobs_per_source: int,
    synthetic_running_jobs_per_source: int,
    synthetic_checkpoints_per_job: int,
    synthetic_terminal_age_days: float,
    synthetic_stale_running_fraction: float,
    synthetic_missing_lease_fraction: float,
) -> dict[str, Any]:
    payload = _normalize_lerobot_retention_synthetic_payload(
        synthetic_sources=synthetic_sources,
        synthetic_completed_jobs_per_source=synthetic_completed_jobs_per_source,
        synthetic_failed_jobs_per_source=synthetic_failed_jobs_per_source,
        synthetic_running_jobs_per_source=synthetic_running_jobs_per_source,
        synthetic_checkpoints_per_job=synthetic_checkpoints_per_job,
        synthetic_terminal_age_days=synthetic_terminal_age_days,
    )
    payload["stale_running_fraction"] = _normalize_lerobot_fraction(
        "synthetic_stale_running_fraction",
        synthetic_stale_running_fraction,
    )
    payload["missing_lease_fraction"] = _normalize_lerobot_fraction(
        "synthetic_missing_lease_fraction",
        synthetic_missing_lease_fraction,
    )
    return payload


def _lerobot_fraction_count(total: int, fraction: float) -> int:
    if total <= 0 or fraction <= 0:
        return 0
    return min(total, int(round(total * fraction)))


def _lerobot_claim_chaos_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    lake_uri: str,
    source_id: str | None,
    stale_after: timedelta,
    recovery_action: str,
    new_owner: str | None,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows_by_job: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        job_id = str(row.get("job_id") or "")
        if not job_id:
            continue
        rows_by_job.setdefault(job_id, []).append(row)
    latest_rows = [
        max(job_rows, key=_lerobot_checkpoint_sort_key) for job_rows in rows_by_job.values()
    ]
    selected_latest = [
        row
        for row in latest_rows
        if source_id is None or str(row.get("source_id") or "") == source_id
    ]
    selected_job_ids = {str(row.get("job_id") or "") for row in selected_latest}
    selected_rows = [
        row
        for job_id, job_rows in rows_by_job.items()
        if job_id in selected_job_ids
        for row in job_rows
    ]

    status_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    source_ids = set()
    for row in selected_latest:
        status = str(row.get("status") or "unknown")
        phase = str(row.get("phase") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if row.get("source_id"):
            source_ids.add(str(row.get("source_id")))

    findings = [
        _lerobot_claim_watchdog_finding(
            lake_uri=lake_uri,
            row=dict(row),
            stale_after=stale_after,
            recovery_action=recovery_action,
            new_owner=new_owner,
            created_by="lancedb-robotics",
            now=now,
        )
        for row in selected_latest
    ]
    stale_findings = [
        finding for finding in findings if finding.status == "running" and finding.stale
    ]
    live_findings = [
        finding for finding in findings if finding.status == "running" and not finding.stale
    ]
    inactive_findings = [finding for finding in findings if finding.status != "running"]
    stale_reasons: dict[str, int] = {}
    for finding in stale_findings:
        stale_reasons[finding.stale_reason] = stale_reasons.get(finding.stale_reason, 0) + 1

    checkpoint_rows = len(selected_rows)
    jobs_seen = len(selected_latest)
    workload = {
        "mode": "checkpoint-rows",
        "sources": len(source_ids),
        "source_id": source_id,
        "jobs_seen": jobs_seen,
        "source_filtered_jobs": len(latest_rows) - jobs_seen,
        "checkpoint_rows": checkpoint_rows,
        "checkpoint_rows_per_job": (checkpoint_rows / jobs_seen if jobs_seen else 0.0),
        "status_counts": dict(sorted(status_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
    }
    watchdog = {
        "stale_count": len(stale_findings),
        "live_count": len(live_findings),
        "inactive_count": len(inactive_findings),
        "missing_lease_count": stale_reasons.get("missing-lease", 0)
        + stale_reasons.get("missing-expiration", 0),
        "stale_reasons": dict(sorted(stale_reasons.items())),
        "recovery_action": recovery_action,
        "sample_recovery_commands": tuple(
            finding.suggested_recovery_command
            for finding in stale_findings[:5]
            if finding.suggested_recovery_command
        ),
        "sample_jobs": tuple(
            {
                "job_id": finding.job_id,
                "source_id": finding.source_id,
                "status": finding.status,
                "phase": finding.phase,
                "stale_reason": finding.stale_reason,
                "checkpoint_id": finding.checkpoint_id,
                "checkpoint_index": finding.checkpoint_index,
            }
            for finding in findings[:5]
        ),
    }
    return workload, watchdog


def _lerobot_claim_chaos_from_synthetic(
    synthetic: dict[str, Any],
    *,
    source_id: str | None,
    recovery_action: str,
    new_owner: str | None,
    stale_after: timedelta,
    now: datetime,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = int(synthetic["sources"])
    completed = int(synthetic["completed_jobs_per_source"])
    failed = int(synthetic["failed_jobs_per_source"])
    running = int(synthetic["running_jobs_per_source"])
    checkpoints = int(synthetic["checkpoints_per_job"])
    selected_sources = 1 if source_id is not None else sources
    source_filtered_sources = max(0, sources - selected_sources)
    running_jobs = selected_sources * running
    stale_count = _lerobot_fraction_count(running_jobs, float(synthetic["stale_running_fraction"]))
    missing_lease_count = min(
        stale_count,
        _lerobot_fraction_count(running_jobs, float(synthetic["missing_lease_fraction"])),
    )
    expired_count = stale_count - missing_lease_count
    live_count = running_jobs - stale_count
    inactive_count = selected_sources * (completed + failed)
    selected_jobs = running_jobs + inactive_count
    selected_rows = selected_jobs * checkpoints
    source_filtered_jobs = source_filtered_sources * (completed + failed + running)
    source_label = source_id or "src-synthetic-0000"
    sample_jobs: list[dict[str, Any]] = []
    sample_commands: list[str] = []
    for index in range(min(stale_count, 5)):
        job_offset = (int(seed) + index) % max(1, running_jobs)
        job_id = f"job-synthetic-stale-{job_offset:04d}"
        checkpoint_index = checkpoints - 1
        checkpoint_id = f"{job_id}:{checkpoint_index:08d}"
        token = f"token-synthetic-{job_offset:04d}"
        reason = "missing-lease" if index < missing_lease_count else "expired-claim"
        sample_jobs.append(
            {
                "job_id": job_id,
                "source_id": source_label,
                "status": "running",
                "phase": "frame-batch",
                "stale_reason": reason,
                "checkpoint_id": checkpoint_id,
                "checkpoint_index": checkpoint_index,
            }
        )
        sample_commands.append(
            _lerobot_claim_recovery_command(
                lake_uri="<synthetic>",
                job_id=job_id,
                recovery_action=recovery_action,
                new_owner=new_owner,
                created_by="lancedb-robotics",
                stale_after=stale_after,
                expected_latest_checkpoint_id=checkpoint_id,
                expected_latest_claim_token=token,
                expected_checkpoint_index=checkpoint_index,
            )
        )

    stale_reasons = {}
    if expired_count:
        stale_reasons["expired-claim"] = expired_count
    if missing_lease_count:
        stale_reasons["missing-lease"] = missing_lease_count
    workload = {
        "mode": "synthetic",
        "sources": selected_sources,
        "total_sources": sources,
        "source_id": source_id,
        "jobs_seen": selected_jobs,
        "source_filtered_jobs": source_filtered_jobs,
        "checkpoint_rows": selected_rows,
        "checkpoint_rows_per_job": (selected_rows / selected_jobs if selected_jobs else 0.0),
        "status_counts": {
            "completed": selected_sources * completed,
            "failed": selected_sources * failed,
            "running": running_jobs,
        },
        "synthetic": dict(synthetic),
        "generated_at": now,
    }
    watchdog = {
        "stale_count": stale_count,
        "live_count": live_count,
        "inactive_count": inactive_count,
        "missing_lease_count": missing_lease_count,
        "stale_reasons": dict(sorted(stale_reasons.items())),
        "recovery_action": recovery_action,
        "sample_recovery_commands": tuple(sample_commands),
        "sample_jobs": tuple(sample_jobs),
    }
    return workload, watchdog


def _lerobot_claim_chaos_real_rehearsal(
    *, retry_owner_count: int, seed: int
) -> dict[str, int]:
    """Race real concurrent recovery attempts against a scratch lake.

    Verifies the backlog-0379 CAS guard by actually running it rather than
    modeling it: creates a throwaway lake (never the caller's real
    lake/checkpoint_rows/synthetic inputs -- this function's only inputs are
    ``retry_owner_count`` and ``seed``), seeds one running claim, and starts
    ``retry_owner_count`` threads racing ``recover_lerobot_ingest_claim``
    against it behind a barrier that forces them to hit the underlying CAS
    ``update(...)`` at the same instant. A plain append, or an insert-only
    ``merge_insert``, can silently let more than one racer "win" under a
    genuinely simultaneous race -- verified directly, see the 0379 decision
    record -- so this reports the REAL observed outcome instead of assuming
    exactly one winner.
    """
    scratch_root = Path(tempfile.mkdtemp(prefix="lerobot-claim-chaos-rehearsal-"))
    try:
        lake = Lake.init(scratch_root / "lake")
        job_id = f"rehearsal-{seed}"
        seeded_at = datetime(2026, 1, 1, tzinfo=UTC)
        seed_row = {
            "checkpoint_id": f"{job_id}:00000000",
            "job_id": job_id,
            "claim_token": "rehearsal-token",
            "checkpoint_index": 0,
            "status": "running",
            "phase": "claimed",
            "claim_owner": "rehearsal-seed",
            "created_by": "rehearsal-seed",
            "started_at": seeded_at,
            "updated_at": seeded_at - timedelta(hours=1),
            "created_at": seeded_at,
        }
        lake.table(_LEROBOT_CHECKPOINT_TABLE).add(
            pa.Table.from_pylist([seed_row], schema=LEROBOT_INGEST_CHECKPOINTS_SCHEMA)
        )

        barrier = threading.Barrier(retry_owner_count)
        outcomes: list[str] = []
        outcomes_lock = threading.Lock()

        def racer(owner: str) -> None:
            barrier.wait()
            try:
                recover_lerobot_ingest_claim(
                    lake,
                    job_id,
                    action="steal",
                    new_owner=owner,
                    stale_after=timedelta(seconds=1),
                    now=seeded_at + timedelta(hours=2),
                )
                outcome = "accepted"
            except LeRobotClaimPreconditionError:
                outcome = "conflict"
            with outcomes_lock:
                outcomes.append(outcome)

        threads = [
            threading.Thread(target=racer, args=(f"rehearsal-owner-{i}",))
            for i in range(retry_owner_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        recovery_rows = [
            candidate
            for candidate in _lerobot_checkpoint_rows(lake)
            if candidate.get("job_id") == job_id
            and candidate.get("phase") in ("claim-abandoned", "claim-stolen")
        ]
        return {
            "accepted_recoveries": outcomes.count("accepted"),
            "cas_conflicts": outcomes.count("conflict"),
            "checkpoint_duplicate_rows": max(0, len(recovery_rows) - 1),
        }
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def _lerobot_claim_chaos_crash_points(
    *,
    frame_count: int,
    episode_count: int,
    camera_count: int,
    batch_size: int,
    retry_owner_count: int,
    stale_after: timedelta,
    heartbeat: timedelta,
    remote_latency_ms: float,
    seed: int,
) -> tuple[LeRobotClaimRecoveryChaosCrashReport, ...]:
    # One real rehearsal covers every recovery-required crash point below: the
    # race itself (retry_owner_count concurrent recoverers) doesn't depend on
    # frame/episode/camera/batch counts, only on retry_owner_count, which is
    # fixed for this call.
    rehearsal = _lerobot_claim_chaos_real_rehearsal(
        retry_owner_count=retry_owner_count, seed=seed
    )
    frame_batches = max(1, (frame_count + batch_size - 1) // batch_size)
    partial_batches = max(1, frame_batches // 2)
    partial_frames = min(frame_count, partial_batches * batch_size)
    plans = (
        (
            "before-claim",
            "worker crashes before a durable claim checkpoint is appended",
            0,
            0,
        ),
        (
            "after-claim",
            "worker crashes after the initial claimed checkpoint",
            1,
            0,
        ),
        (
            "media-inspection",
            "worker crashes after media inspection progress is checkpointed",
            2,
            0,
        ),
        (
            "frame-batch",
            "worker crashes after one or more frame batches are flushed",
            2 + partial_batches,
            partial_frames,
        ),
        (
            "metadata-ready",
            "worker crashes after all observations and metadata-ready checkpoint",
            2 + frame_batches + 1,
            frame_count,
        ),
    )
    duplicate_rows = {
        "observations": 0,
        "episodes": 0,
        "videos": 0,
        "events": 0,
        "runs": 0,
        "transform_runs": 0,
    }
    protections = {
        "observations": "deterministic observation_id skip-existing resume",
        "episodes": "merge_insert episode_id",
        "videos": "merge_insert video_id",
        "events": "merge_insert event_id",
        "runs": "merge_insert run_id",
        "transform_runs": "merge_insert transform_id",
    }
    retry_checkpoint_rows = frame_batches + 4
    latency_seconds = _lerobot_claim_chaos_latency_seconds(
        stale_after=stale_after,
        heartbeat=heartbeat,
        retry_owner_count=retry_owner_count,
        remote_latency_ms=remote_latency_ms,
    )
    reports = []
    for crash_point, description, checkpoint_rows_before, observations_before_retry in plans:
        recovery_required = checkpoint_rows_before > 0
        recovery_rows = 1 if recovery_required else 0
        reports.append(
            LeRobotClaimRecoveryChaosCrashReport(
                crash_point=crash_point,
                description=description,
                recovery_required=recovery_required,
                checkpoint_rows_before=checkpoint_rows_before,
                recovery_checkpoint_rows=recovery_rows,
                retry_checkpoint_rows=retry_checkpoint_rows,
                checkpoint_rows_after=checkpoint_rows_before
                + recovery_rows
                + retry_checkpoint_rows,
                recovery_latency_seconds=latency_seconds if recovery_required else 0.0,
                observations_before_retry=observations_before_retry,
                observations_replayed=frame_count,
                observations_written_after_retry=frame_count - observations_before_retry,
                rows_skipped_existing=observations_before_retry,
                accepted_recoveries=rehearsal["accepted_recoveries"] if recovery_required else 0,
                cas_conflicts=rehearsal["cas_conflicts"] if recovery_required else 0,
                checkpoint_duplicate_rows=(
                    rehearsal["checkpoint_duplicate_rows"] if recovery_required else 0
                ),
                duplicate_rows=dict(duplicate_rows),
                protections=dict(protections),
            )
        )
    return tuple(reports)


def _lerobot_claim_chaos_latency_seconds(
    *,
    stale_after: timedelta,
    heartbeat: timedelta,
    retry_owner_count: int,
    remote_latency_ms: float,
) -> float:
    remote_round_trips = 4 + max(0, retry_owner_count - 1)
    contention_seconds = max(0, retry_owner_count - 1) * min(
        heartbeat.total_seconds(),
        60.0,
    )
    return round(
        stale_after.total_seconds()
        + contention_seconds
        + (remote_latency_ms / 1000.0) * remote_round_trips,
        3,
    )


def _lerobot_claim_chaos_recovery_summary(
    watchdog: dict[str, Any],
    *,
    retry_owner_count: int,
    stale_after: timedelta,
    heartbeat: timedelta,
    batch_size: int,
    frame_count: int,
    remote_latency_ms: float,
) -> dict[str, Any]:
    stale_count = int(watchdog.get("stale_count") or 0)
    frame_batches = max(1, (frame_count + batch_size - 1) // batch_size)
    retry_checkpoint_rows = frame_batches + 4
    cas_conflicts = stale_count * max(0, retry_owner_count - 1)
    return {
        "accepted_recoveries": stale_count,
        "cas_conflicts": cas_conflicts,
        "retry_owner_count": retry_owner_count,
        "forced_recoveries": 0,
        "post_recovery_retries": stale_count,
        "checkpoint_rows_added_if_all_recovered": stale_count * (1 + retry_checkpoint_rows),
        "retry_checkpoint_rows_per_job": retry_checkpoint_rows,
        "estimated_recovery_latency_seconds": (
            _lerobot_claim_chaos_latency_seconds(
                stale_after=stale_after,
                heartbeat=heartbeat,
                retry_owner_count=retry_owner_count,
                remote_latency_ms=remote_latency_ms,
            )
            if stale_count
            else 0.0
        ),
        "stale_after_seconds": stale_after.total_seconds(),
        "heartbeat_interval_seconds": heartbeat.total_seconds(),
        "remote_latency_ms": remote_latency_ms,
    }


def _lerobot_claim_chaos_recommendations(
    scenario: str,
    workload: dict[str, Any],
    *,
    configured_lease: timedelta,
    configured_heartbeat: timedelta,
    remote_latency_ms: float,
) -> dict[str, Any]:
    profile = _resolve_lerobot_claim_chaos_profile(scenario, workload)
    defaults = {
        "local-smoke": (300.0, 30.0, 1, 64),
        "ci": (900.0, 60.0, 1, 256),
        "mid-corpus": (3600.0, 300.0, 4, 10_000),
        "full-public-corpus": (21600.0, 300.0, 12, 100_000),
        "audit-window": (43200.0, 600.0, 1, 1_000_000),
    }
    lease_seconds, heartbeat_seconds, max_stale, max_rows_per_job = defaults[profile]
    if remote_latency_ms >= 5000:
        heartbeat_seconds = max(
            heartbeat_seconds, min(lease_seconds / 4.0, (remote_latency_ms / 1000.0) * 4.0)
        )
    return {
        "profile": profile,
        "lease_timeout_seconds": lease_seconds,
        "heartbeat_interval_seconds": heartbeat_seconds,
        "watchdog_stale_after_seconds": lease_seconds,
        "configured_lease_timeout_seconds": configured_lease.total_seconds(),
        "configured_heartbeat_interval_seconds": configured_heartbeat.total_seconds(),
        "warning_thresholds": {
            "stale_claims": max_stale,
            "checkpoint_rows_per_job": max_rows_per_job,
            "cas_conflict_rate": 0.0,
        },
        "rationale": (
            "short local/CI leases surface crashed workers quickly",
            "larger public-corpus leases tolerate remote media and object-store latency",
            "watchdog and recovery should use the same stale-after window as the worker lease",
        ),
    }


def _resolve_lerobot_claim_chaos_profile(scenario: str, workload: dict[str, Any]) -> str:
    if scenario == "auto":
        rows = int(workload.get("checkpoint_rows") or 0)
        jobs = int(workload.get("jobs_seen") or 0)
        if rows <= 1_000 and jobs <= 100:
            return "local-smoke"
        if rows <= 10_000:
            return "ci"
        if rows <= 100_000:
            return "mid-corpus"
        return "full-public-corpus"
    if scenario == "ci":
        return "ci"
    return scenario


def _lerobot_claim_chaos_duplicate_protection(
    *,
    frame_count: int,
    episode_count: int,
    camera_count: int,
) -> dict[str, Any]:
    table_keys = {
        "observations": "observation_id",
        "episodes": "episode_id",
        "videos": "video_id",
        "events": "event_id",
        "runs": "run_id",
        "transform_runs": "transform_id",
    }
    expected_rows = {
        "observations": frame_count,
        "episodes": episode_count,
        "videos": episode_count * camera_count,
        "events": 2,
        "runs": 1,
        "transform_runs": 2,
    }
    return {
        "status": "passed",
        "table_keys": table_keys,
        "expected_final_rows": expected_rows,
        "duplicate_rows_after_retry": {name: 0 for name in table_keys},
        "resume_strategy": "skip-existing-observation-id plus metadata merge_insert upserts",
    }


def _lerobot_claim_chaos_warnings(
    workload: dict[str, Any],
    watchdog: dict[str, Any],
    recovery: dict[str, Any],
    recommendations: dict[str, Any],
    crash_points: tuple[LeRobotClaimRecoveryChaosCrashReport, ...],
    *,
    configured_lease: timedelta,
    configured_heartbeat: timedelta,
) -> tuple[dict[str, Any], ...]:
    warnings: list[dict[str, Any]] = []
    thresholds = recommendations.get("warning_thresholds") or {}
    stale_threshold = int(thresholds.get("stale_claims") or 0)
    stale_count = int(watchdog.get("stale_count") or 0)
    if stale_count > stale_threshold:
        warnings.append(
            {
                "level": "warning",
                "metric": "stale_claims",
                "actual": stale_count,
                "threshold": stale_threshold,
            }
        )
    missing_lease = int(watchdog.get("missing_lease_count") or 0)
    if missing_lease:
        warnings.append(
            {
                "level": "warning",
                "metric": "missing_lease",
                "actual": missing_lease,
                "threshold": 0,
            }
        )
    cas_conflicts = int(recovery.get("cas_conflicts") or 0)
    if cas_conflicts:
        warnings.append(
            {
                "level": "warning",
                "metric": "cas_conflicts",
                "actual": cas_conflicts,
                "threshold": 0,
            }
        )
    max_rows_per_job = int(thresholds.get("checkpoint_rows_per_job") or 0)
    rows_per_job = float(workload.get("checkpoint_rows_per_job") or 0.0)
    if max_rows_per_job and rows_per_job > max_rows_per_job:
        warnings.append(
            {
                "level": "warning",
                "metric": "checkpoint_rows_per_job",
                "actual": rows_per_job,
                "threshold": max_rows_per_job,
            }
        )
    if configured_heartbeat >= configured_lease:
        warnings.append(
            {
                "level": "warning",
                "metric": "heartbeat_vs_lease",
                "actual": configured_heartbeat.total_seconds(),
                "threshold": configured_lease.total_seconds(),
            }
        )
    if not all(crash.passed for crash in crash_points):
        warnings.append(
            {
                "level": "error",
                "metric": "duplicate_protection",
                "actual": "failed",
                "threshold": "passed",
            }
        )
    return tuple(warnings)


def _modality(topic: str, schema_name: str | None) -> str:
    haystack = f"{topic} {schema_name or ''}".lower()
    for hint, modality in _MODALITY_HINTS:
        if hint in haystack:
            return modality
    return "unknown"


@dataclass(frozen=True)
class IngestReport:
    """What one ingest call did: identity, row deltas, time coverage, integrity."""

    lake_uri: str
    source: SourceRegistration
    run_id: str
    already_ingested: bool
    rows_added: dict[str, int] = field(default_factory=dict)
    observations_by_topic: dict[str, int] = field(default_factory=dict)
    message_count: int = 0
    start_time_ns: int = 0
    end_time_ns: int = 0
    duration_ns: int = 0
    # Decode coverage (backlog 0014/0017/0020): a per-run summary of how messages
    # resolved, which encodings appeared, and -- now that every registry encoding
    # is decode-attempted -- which encodings still landed ``raw`` (missing extra
    # or unsupported schema encoding), so the remaining gap is visible per run.
    decode_by_status: dict[str, int] = field(default_factory=dict)
    decode_by_encoding: dict[str, int] = field(default_factory=dict)
    decode_raw_by_encoding: dict[str, int] = field(default_factory=dict)
    # Integrity (backlog 0017): "complete" on a clean read; "crc-mismatch" or
    # "truncated" when the file was damaged, in which case the run is quarantined
    # and ``recovered_count`` records how many messages were salvaged.
    integrity_status: str = "complete"
    integrity_reason: str | None = None
    recovered_count: int = 0
    # The ``ingest`` transform_runs id for this run, so callers/CLI can surface the
    # inline-emitted lineage execution (backlog 0098). ``None`` on an
    # already-ingested no-op that recorded no ingest transform.
    transform_id: str | None = None
    ingest_job_id: str | None = None
    # Post-ingest compaction + version-retention summary (backlog 0180). ``None``
    # when ingest skipped finalize (``compact=False`` and ``prune_versions=False``,
    # or an already-ingested no-op). Otherwise carries the ``observations`` grain's
    # fragment/version counts before and after, plus any version-cleanup metrics --
    # so a "well-formed lake by default" is observable on the report.
    compaction: dict[str, Any] | None = None

    @property
    def quarantined(self) -> bool:
        return self.integrity_status != "complete"


def _observation_row(
    message: dict, *, run_id: str, raw_uri: str, transform_id: str, created_at: datetime
) -> dict:
    """Build one ``observations`` row from a decoded message + typed extraction."""
    payload = json.loads(message["payload_json"]) if message["payload_json"] else None
    typed = extract(message["schema_name"], payload)
    modality = typed.modality or _modality(message["topic"], message["schema_name"])
    return {
        "observation_id": f"{run_id}:{message['topic']}:{message['sequence']:06d}",
        "run_id": run_id,
        "episode_id": None,
        "episode_index": None,
        "frame_index": None,
        "timestamp_ns": message["log_time_ns"],
        "sensor_id": message["topic"].strip("/").replace("/", "_"),
        "topic": message["topic"],
        "modality": modality,
        "robot_id": None,
        "site_id": None,
        "task_id": None,
        "software_version": None,
        "outcome": None,
        "raw_uri": raw_uri,
        "raw_channel": message["topic"],
        "raw_log_time_ns": message["log_time_ns"],
        "raw_sequence": message["sequence"],
        "payload_json": message["payload_json"],
        "payload_blob": message["payload_blob"],
        "message_encoding": message["message_encoding"],
        "schema_encoding": message["schema_encoding"],
        "decode_status": message["decode_status"],
        "decode_error": message["decode_error"],
        "state_vector": typed.state_vector,
        "action_vector": typed.action_vector,
        "transform_id": transform_id,
        "created_at": created_at,
    }


def ingest_mcap(
    lake: Lake,
    path: str | Path,
    *,
    created_by: str = "lancedb-robotics",
    batch_size: int = DEFAULT_BATCH_SIZE,
    validate_crcs: bool = True,
    compact: bool = True,
    prune_versions: bool = True,
    retain_versions: int = DEFAULT_INGEST_RETAIN_VERSIONS,
    index_predicates: bool = True,
    auth_ref: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> IngestReport:
    """Ingest one MCAP recording into ``lake``; idempotent by content (checksum).

    ``path`` may be a single ``*.mcap`` file or a **split recording** — a
    directory of shards (or its ``metadata.yaml``) that ``rosbag2`` and friends
    produce when one session is split by size/duration (backlog 0019). A split
    recording ingests as exactly one ``run`` whose ``run_id`` is content-addressed
    from the *ordered shard checksums*, with per-topic ``sequence`` continuous
    across shard boundaries and each observation's ``raw_uri`` pointing at the
    specific shard it came from. See :func:`_ingest_split_recording`.

    ``run_id``/``source_id`` are content-addressed from the file bytes, so the
    same log ingested from any path or machine yields the same ids (and the same
    downstream scenario/dataset ids and split). The absolute path is kept as
    ``raw_uri`` provenance only. See decision 0023.

    Observations stream to the lake in batches of ``batch_size`` (backlog 0017)
    so memory stays bounded on multi-GB logs. Each streamed batch is one Lance
    fragment + one version, so when the run finishes ingest compacts the
    ``observations`` grain up to Lance's healthy ~1M-row fragment size and
    snapshot-safely prunes the per-flush version churn (``compact`` /
    ``prune_versions``, both default on; backlog 0180 / BUG-14) — a freshly
    ingested lake is well-formed instead of carrying a ~79x per-row scan tax.
    ``validate_crcs`` (default on)
    validates chunk CRCs while reading; pass ``False`` to skip on the hot path
    for trusted data. A CRC mismatch or a truncated file keeps the readable
    prefix and quarantines the run instead of aborting.

    Raises :class:`lancedb_robotics.adapters.CodecUnavailableError` if a chunk's
    compression codec is not installed, or a plain
    :class:`lancedb_robotics.adapters.AdapterError` if the file is missing or not
    a valid MCAP. A damaged file with a readable prefix is recovered and
    quarantined, not raised; a file with no recoverable messages (zero-byte,
    truncated magic) raises rather than writing an empty run (backlog 0018).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    # A directory or metadata.yaml is a split recording: N shards become one run
    # (backlog 0019). A bare file keeps the unchanged single-file path below. The
    # cheap shard-plan probe avoids re-checksumming a single file twice.
    if not is_object_store_uri(path) and resolve_shards(path).is_split:
        return _ingest_split_recording(
            lake,
            resolve_recording(path),
            created_by=created_by,
            batch_size=batch_size,
            validate_crcs=validate_crcs,
            compact=compact,
            prune_versions=prune_versions,
            retain_versions=retain_versions,
            index_predicates=index_predicates,
            auth_ref=auth_ref,
        )

    adapter = get_adapter("mcap")
    uri = source_uri(path)

    inspect_started = datetime.now(UTC)
    inspect_report = _safe_inspect(
        adapter,
        uri,
        storage_options=storage_options,
        auth_ref=auth_ref,
    )  # None if truncated/CRC-damaged
    inspect_finished = datetime.now(UTC)

    checksum = file_checksum(uri, storage_options=storage_options, auth_ref=auth_ref)
    digest = content_key_digest(checksum)[:16]  # content-addressed: path-independent
    run_id = f"run-{digest}"
    source_id = f"src-{digest}"

    if lake.table("runs").count_rows(f"run_id = '{run_id}'") > 0:
        return _already_ingested_report(
            lake,
            source_id,
            uri,
            checksum,
            run_id,
            inspect_report,
            auth_ref=auth_ref,
            created_by=created_by,
        )

    now = datetime.now(UTC)
    inspect_transform_id = f"tfm-{digest}-inspect"
    ingest_transform_id = f"tfm-{digest}-ingest"

    # Stream messages -> observation batches, flushing as we go. A damaged file
    # raises CorruptMcapError after the readable prefix has been yielded; keep
    # what we got and record the integrity verdict.
    observations_table = lake.table("observations")
    batch: list[dict] = []
    by_topic: dict[str, int] = {}
    topic_schema: dict[str, tuple[str | None, str | None]] = {}
    decode_by_status: dict[str, int] = {}
    decode_by_encoding: dict[str, int] = {}
    decode_raw_by_encoding: dict[str, int] = {}
    extracted_by_modality: dict[str, int] = {}
    stream_start: int | None = None
    stream_end: int | None = None
    total = 0
    integrity_status = "complete"
    integrity_reason: str | None = None
    recovered_count = 0

    try:
        for message in adapter.ingest(
            uri,
            validate_crcs=validate_crcs,
            storage_options=storage_options,
            auth_ref=auth_ref,
        ):
            topic = message["topic"]
            by_topic[topic] = by_topic.get(topic, 0) + 1
            topic_schema.setdefault(topic, (message["schema_name"], message["schema_encoding"]))
            status = message["decode_status"]
            decode_by_status[status] = decode_by_status.get(status, 0) + 1
            encoding = message["message_encoding"] or "unknown"
            decode_by_encoding[encoding] = decode_by_encoding.get(encoding, 0) + 1
            if status == "raw":
                decode_raw_by_encoding[encoding] = decode_raw_by_encoding.get(encoding, 0) + 1
            row = _observation_row(
                message,
                run_id=run_id,
                raw_uri=uri,
                transform_id=ingest_transform_id,
                created_at=now,
            )
            if row["modality"] != "unknown":
                extracted_by_modality[row["modality"]] = (
                    extracted_by_modality.get(row["modality"], 0) + 1
                )
            ts = message["log_time_ns"]
            stream_start = ts if stream_start is None else min(stream_start, ts)
            stream_end = ts if stream_end is None else max(stream_end, ts)
            batch.append(row)
            total += 1
            if len(batch) >= batch_size:
                _flush(observations_table, batch)
    except CorruptMcapError as exc:
        if total == 0:
            # Nothing was recoverable (e.g. a zero-byte or truncated-magic file):
            # there is no readable prefix to quarantine, so re-raise rather than
            # fabricate an empty run. This keeps the genuine-error path honest --
            # a malformed file never produces a silent empty ingest (backlog 0018).
            raise
        integrity_status = exc.status
        integrity_reason = exc.reason
        recovered_count = exc.recovered or total
    _flush(observations_table, batch)  # remainder, kept even when corruption stopped us

    # Source registration: use the inspect report's topic fingerprints when we
    # have them, else synthesize from what we streamed (a damaged file has no
    # summary to inspect). Each observation row already used ``uri`` for raw_uri.
    registration = register_source(
        lake,
        uri,
        adapter="mcap",
        inspect_report=inspect_report or _synthetic_report(topic_schema),
        auth_ref=auth_ref,
        storage_options=storage_options,
    )

    # Attachment + metadata records (backlog 0016). A damaged file cannot serve
    # these (no index), so recover nothing rather than crash.
    attachment_records = _safe_records(
        adapter.attachments,
        uri,
        storage_options=storage_options,
        auth_ref=auth_ref,
    )
    metadata_records = _safe_records(
        adapter.metadata_records,
        uri,
        storage_options=storage_options,
        auth_ref=auth_ref,
    )
    attachment_rows = [
        {
            "attachment_id": f"{run_id}:att:{index:04d}",
            "run_id": run_id,
            "name": att["name"],
            "media_type": att["media_type"],
            "size": att["size"],
            "sha256": att["sha256"],
            "log_time_ns": att["log_time_ns"],
            "create_time_ns": att["create_time_ns"],
            "data": att["data"],
            "transform_id": ingest_transform_id,
            "created_at": now,
        }
        for index, att in enumerate(attachment_records)
    ]
    metadata_kv = [
        {"key": f"{record['name']}.{key}", "value": value}
        for record in metadata_records
        for key, value in sorted(record["metadata"].items())
    ]

    # Time coverage: the inspect summary when available, else the streamed range.
    start_time_ns = inspect_report["start_time_ns"] if inspect_report else (stream_start or 0)
    end_time_ns = inspect_report["end_time_ns"] if inspect_report else (stream_end or 0)
    duration_ns = (
        inspect_report["duration_ns"] if inspect_report else max(0, end_time_ns - start_time_ns)
    )

    run_metadata = []
    if inspect_report:
        run_metadata += [
            {"key": "profile", "value": inspect_report["profile"]},
            {"key": "library", "value": inspect_report["library"]},
        ]
    run_metadata += metadata_kv
    # Integrity verdict travels on run.metadata so it survives the quality gate's
    # quality_flags overwrite; the gate reads it back as an integrity rule (FS4).
    run_metadata.append({"key": "integrity.status", "value": integrity_status})
    if integrity_status != "complete":
        run_metadata.append({"key": "integrity.recovered", "value": str(recovered_count)})
        if integrity_reason:
            run_metadata.append({"key": "integrity.reason", "value": integrity_reason})

    # A damaged read is quarantined immediately at ingest (do not silently pass);
    # a clean read leaves quality_flags NULL until the quality gate runs.
    quality_flags = None
    if integrity_status != "complete":
        quality_flags = ["quarantined", f"integrity:{integrity_status}"]

    run_row = {
        "run_id": run_id,
        "run_kind": "log",
        "source": "mcap",
        "source_id": registration.source_id,
        "raw_uri": registration.uri,
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
        "duration_ns": duration_ns,
        "metadata": run_metadata,
        "quality_flags": quality_flags,
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
            "source": "message-boundary",
            "transform_id": ingest_transform_id,
            "created_at": now,
        }
        for event_type, timestamp_ns in (
            ("run_start", start_time_ns),
            ("run_end", end_time_ns),
        )
    ]

    # Only declare attachments as an output table when this log actually wrote
    # one; a zero-attachment log (didi/demo) leaves the table untouched.
    ingest_output_tables = ["runs", "observations"]
    if attachment_rows:
        ingest_output_tables.append("attachments")
    ingest_output_tables.append("events")

    ingest_finished = datetime.now(UTC)
    transform_rows = [
        {
            "transform_id": inspect_transform_id,
            "kind": "inspect",
            "source_id": registration.source_id,
            "input_uris": [registration.uri],
            "output_tables": [],
            "params": json.dumps({"adapter": "mcap"}),
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
            "output_tables": ingest_output_tables,
            "params": json.dumps(
                {
                    "adapter": "mcap",
                    "run_id": run_id,
                    "batch_size": batch_size,
                    # Decode coverage (backlog 0014): how many messages decoded vs
                    # landed raw/failed, and the per-encoding mix. Consumed by the
                    # quality gate / 0017.
                    "decode_by_status": dict(sorted(decode_by_status.items())),
                    "decode_by_encoding": dict(sorted(decode_by_encoding.items())),
                    # Which encodings still landed raw (missing extra / unsupported
                    # schema encoding); empty when everything decoded (backlog 0020).
                    "decode_raw_by_encoding": dict(sorted(decode_raw_by_encoding.items())),
                    # Typed-extraction coverage (backlog 0015): how many
                    # observations were typed into a modality (and which), and
                    # the layout version the vectors were written against.
                    "extracted_by_modality": dict(sorted(extracted_by_modality.items())),
                    "extract_layout_version": LAYOUT_VERSION,
                    # Attachment + metadata record coverage (backlog 0016).
                    "attachment_count": len(attachment_rows),
                    "metadata_record_count": len(metadata_records),
                    # Integrity verdict (backlog 0017).
                    "integrity": {
                        "status": integrity_status,
                        "recovered": recovered_count,
                        "reason": integrity_reason,
                    },
                }
            ),
            "status": "completed" if integrity_status == "complete" else "recovered",
            "started_at": inspect_started,
            "finished_at": ingest_finished,
            "created_by": created_by,
            "created_at": now,
        },
    ]

    lake.table("runs").add(pa.Table.from_pylist([run_row], schema=RUNS_SCHEMA))
    # Observations were streamed in batches above; nothing left to bulk-write.
    if attachment_rows:
        lake.table("attachments").add(
            pa.Table.from_pylist(attachment_rows, schema=ATTACHMENTS_SCHEMA)
        )
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

    # Emit lineage for the inspect + ingest transforms inline (backlog 0098),
    # after finalize so the emitted table versions match the compacted lake.
    for _transform_row in transform_rows:
        emit_transform_lineage(lake, _transform_row)

    return IngestReport(
        lake_uri=lake.uri,
        compaction=compaction,
        source=registration,
        run_id=run_id,
        already_ingested=False,
        transform_id=ingest_transform_id,
        rows_added={
            "integration_sources": 1 if registration.created else 0,
            "runs": 1,
            "observations": total,
            "attachments": len(attachment_rows),
            "events": len(event_rows),
            "transform_runs": len(transform_rows),
        },
        observations_by_topic=dict(sorted(by_topic.items())),
        message_count=total,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        duration_ns=duration_ns,
        decode_by_status=dict(sorted(decode_by_status.items())),
        decode_by_encoding=dict(sorted(decode_by_encoding.items())),
        decode_raw_by_encoding=dict(sorted(decode_raw_by_encoding.items())),
        integrity_status=integrity_status,
        integrity_reason=integrity_reason,
        recovered_count=recovered_count,
    )


def ingest_rosbag(
    lake: Lake,
    path: str | Path,
    *,
    created_by: str = "lancedb-robotics",
    batch_size: int = DEFAULT_BATCH_SIZE,
    compact: bool = True,
    prune_versions: bool = True,
    retain_versions: int = DEFAULT_INGEST_RETAIN_VERSIONS,
    index_predicates: bool = True,
    auth_ref: str | None = None,
) -> IngestReport:
    """Ingest a ROS1 `.bag` or ROS2 sqlite `.db3` recording into ``lake``.

    ROS bag support is a container adapter over the same canonical ingest path as
    MCAP: the adapter yields decoded message dictionaries, observations keep
    pointer provenance through ``raw_uri``/``raw_channel``/``raw_sequence``, and
    run/source ids are content-addressed from the bag bytes. Missing ROS message
    decoder extras degrade per-message to ``decode_status='raw'``; a missing
    `rosbags` reader extra is reported by the adapter as an actionable
    :class:`AdapterError`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    adapter = get_adapter("rosbag")
    source = adapter.source(path)
    run_id = f"run-{source.digest}"
    source_id = f"src-{source.digest}"

    inspect_started = datetime.now(UTC)
    inspect_report = adapter.inspect(path)
    inspect_finished = datetime.now(UTC)

    if lake.table("runs").count_rows(f"run_id = '{run_id}'") > 0:
        return _already_ingested_report(
            lake,
            source_id,
            source.uri,
            source.checksum,
            run_id,
            inspect_report,
            auth_ref=auth_ref,
            created_by=created_by,
            adapter_name="rosbag",
        )

    now = datetime.now(UTC)
    inspect_transform_id = f"tfm-{source.digest}-inspect"
    ingest_transform_id = f"tfm-{source.digest}-ingest"

    observations_table = lake.table("observations")
    batch: list[dict] = []
    by_topic: dict[str, int] = {}
    decode_by_status: dict[str, int] = {}
    decode_by_encoding: dict[str, int] = {}
    decode_raw_by_encoding: dict[str, int] = {}
    extracted_by_modality: dict[str, int] = {}
    stream_start: int | None = None
    stream_end: int | None = None
    total = 0

    for message in adapter.ingest(path):
        topic = message["topic"]
        by_topic[topic] = by_topic.get(topic, 0) + 1
        status = message["decode_status"]
        decode_by_status[status] = decode_by_status.get(status, 0) + 1
        encoding = message["message_encoding"] or "unknown"
        decode_by_encoding[encoding] = decode_by_encoding.get(encoding, 0) + 1
        if status == "raw":
            decode_raw_by_encoding[encoding] = decode_raw_by_encoding.get(encoding, 0) + 1
        row = _observation_row(
            message,
            run_id=run_id,
            raw_uri=source.uri,
            transform_id=ingest_transform_id,
            created_at=now,
        )
        if row["modality"] != "unknown":
            extracted_by_modality[row["modality"]] = (
                extracted_by_modality.get(row["modality"], 0) + 1
            )
        ts = message["log_time_ns"]
        stream_start = ts if stream_start is None else min(stream_start, ts)
        stream_end = ts if stream_end is None else max(stream_end, ts)
        batch.append(row)
        total += 1
        if len(batch) >= batch_size:
            _flush(observations_table, batch)
    _flush(observations_table, batch)

    registration = _register_resolved_source(
        lake,
        source,
        adapter="rosbag",
        inspect_report=inspect_report,
        auth_ref=auth_ref,
    )

    start_time_ns = inspect_report["start_time_ns"] if inspect_report else (stream_start or 0)
    end_time_ns = inspect_report["end_time_ns"] if inspect_report else (stream_end or 0)
    duration_ns = (
        inspect_report["duration_ns"] if inspect_report else max(0, end_time_ns - start_time_ns)
    )

    run_metadata = [
        {"key": "profile", "value": inspect_report["profile"]},
        {"key": "library", "value": inspect_report["library"]},
        {"key": "storage_identifier", "value": inspect_report.get("storage_identifier", "")},
        {"key": "integrity.status", "value": "complete"},
    ]

    run_row = {
        "run_id": run_id,
        "run_kind": "log",
        "source": "rosbag",
        "source_id": registration.source_id,
        "raw_uri": registration.uri,
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
        "duration_ns": duration_ns,
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
            "source": "message-boundary",
            "transform_id": ingest_transform_id,
            "created_at": now,
        }
        for event_type, timestamp_ns in (
            ("run_start", start_time_ns),
            ("run_end", end_time_ns),
        )
    ]

    ingest_output_tables = ["runs", "observations", "events"]
    ingest_finished = datetime.now(UTC)
    transform_rows = [
        {
            "transform_id": inspect_transform_id,
            "kind": "inspect",
            "source_id": registration.source_id,
            "input_uris": source.input_uris,
            "output_tables": [],
            "params": json.dumps(
                {
                    "adapter": "rosbag",
                    "storage_identifier": source.storage_identifier,
                    "source_kind": source.kind,
                },
                sort_keys=True,
            ),
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
            "input_uris": source.input_uris,
            "output_tables": ingest_output_tables,
            "params": json.dumps(
                {
                    "adapter": "rosbag",
                    "run_id": run_id,
                    "batch_size": batch_size,
                    "decode_by_status": dict(sorted(decode_by_status.items())),
                    "decode_by_encoding": dict(sorted(decode_by_encoding.items())),
                    "decode_raw_by_encoding": dict(sorted(decode_raw_by_encoding.items())),
                    "extracted_by_modality": dict(sorted(extracted_by_modality.items())),
                    "extract_layout_version": LAYOUT_VERSION,
                    "integrity": {"status": "complete", "recovered": 0, "reason": None},
                    "source": {
                        "kind": source.kind,
                        "storage_identifier": source.storage_identifier,
                        "files": source.input_uris,
                    },
                },
                sort_keys=True,
            ),
            "status": "completed",
            "started_at": inspect_started,
            "finished_at": ingest_finished,
            "created_by": created_by,
            "created_at": now,
        },
    ]

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

    # Emit lineage for the inspect + ingest transforms inline (backlog 0098),
    # after finalize so the emitted table versions match the compacted lake.
    for _transform_row in transform_rows:
        emit_transform_lineage(lake, _transform_row)

    return IngestReport(
        lake_uri=lake.uri,
        compaction=compaction,
        source=registration,
        run_id=run_id,
        already_ingested=False,
        transform_id=ingest_transform_id,
        rows_added={
            "integration_sources": 1 if registration.created else 0,
            "runs": 1,
            "observations": total,
            "attachments": 0,
            "events": len(event_rows),
            "transform_runs": len(transform_rows),
        },
        observations_by_topic=dict(sorted(by_topic.items())),
        message_count=total,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        duration_ns=duration_ns,
        decode_by_status=dict(sorted(decode_by_status.items())),
        decode_by_encoding=dict(sorted(decode_by_encoding.items())),
        decode_raw_by_encoding=dict(sorted(decode_raw_by_encoding.items())),
    )


def ingest_lerobot(
    lake: Lake,
    source: str | Path,
    *,
    created_by: str = "lancedb-robotics",
    batch_size: int = DEFAULT_BATCH_SIZE,
    compact: bool = True,
    prune_versions: bool = True,
    retain_versions: int = DEFAULT_INGEST_RETAIN_VERSIONS,
    index_predicates: bool = True,
    auth_ref: str | None = None,
    storage_options: dict[str, Any] | None = None,
    source_manifest_cache: LeRobotObjectStoreManifestCache | str | Path | None = None,
    object_store_validation_policy: str = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
    object_store_validation_sample_count: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
    object_store_validation_sample_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
    object_store_validation_strict_max_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
    media_inspection_timeout_seconds: float | None = None,
    media_inspection_workers: int | None = None,
    media_inspection_retries: int = 0,
    media_inspection_retry_backoff_seconds: float = 0.0,
    media_inspection_retry_policy: str = "fixed",
    media_inspection_execution_mode: str = "thread",
    keyframe_map_inline_threshold_bytes: int
    | None = DEFAULT_LEROBOT_KEYFRAME_MAP_INLINE_THRESHOLD_BYTES,
    keyframe_map_inline_threshold_frames: int
    | None = DEFAULT_LEROBOT_KEYFRAME_MAP_INLINE_THRESHOLD_FRAMES,
    ingest_job_id: str | None = None,
    claim_owner: str | None = None,
    claim_lease_timeout: timedelta | int | float | None = DEFAULT_LEROBOT_CLAIM_LEASE,
    claim_heartbeat_interval: timedelta | int | float | None = DEFAULT_LEROBOT_CLAIM_HEARTBEAT,
    expected_latest_checkpoint_id: str | None = None,
    expected_latest_claim_token: str | None = None,
    expected_checkpoint_index: int | None = None,
    decoded_frame_conformance: dict[str, Any] | None = None,
) -> IngestReport:
    """Ingest a LeRobot dataset directory or HF Hub repo id into canonical rows.

    LeRobot is already structured as episodes and frames, so this path maps
    Parquet feature rows into first-class ``episodes`` and frame-grain
    ``observations`` without the MCAP/ROS message decoder registry. Per-camera
    MP4 streams are recorded as ``videos`` / ``video_encodings`` references
    pointing back to the source files; the ingest does not re-encode or copy
    those bytes into a new video.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    claim_lease = _normalize_lerobot_duration(
        claim_lease_timeout,
        name="claim_lease_timeout",
        default=DEFAULT_LEROBOT_CLAIM_LEASE,
    )
    claim_heartbeat = _normalize_lerobot_duration(
        claim_heartbeat_interval,
        name="claim_heartbeat_interval",
        default=DEFAULT_LEROBOT_CLAIM_HEARTBEAT,
    )
    keyframe_map_threshold_bytes, keyframe_map_threshold_frames = (
        _normalize_lerobot_keyframe_map_thresholds(
            keyframe_map_inline_threshold_bytes,
            keyframe_map_inline_threshold_frames,
        )
    )

    adapter = get_adapter("lerobot")
    manifest_cache = resolve_lerobot_object_store_manifest_cache(source_manifest_cache)
    if (
        manifest_cache is None
        and object_store_validation_policy != DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY
        and is_object_store_uri(source)
    ):
        manifest_cache = LeRobotObjectStoreManifestCache()
    inspect_started = datetime.now(UTC)
    inspect_report = adapter.inspect(
        source,
        inspect_videos=False,
        storage_options=storage_options,
        auth_ref=auth_ref,
        source_manifest_cache=manifest_cache,
        object_store_validation_policy=object_store_validation_policy,
        object_store_validation_sample_count=object_store_validation_sample_count,
        object_store_validation_sample_bytes=object_store_validation_sample_bytes,
        object_store_validation_strict_max_bytes=object_store_validation_strict_max_bytes,
    )
    inspect_finished = datetime.now(UTC)
    dataset = _lerobot_dataset_for_ingest(
        adapter,
        source,
        storage_options=storage_options,
        auth_ref=auth_ref,
        source_manifest_cache=manifest_cache,
        object_store_validation_policy=object_store_validation_policy,
        object_store_validation_sample_count=object_store_validation_sample_count,
        object_store_validation_sample_bytes=object_store_validation_sample_bytes,
        object_store_validation_strict_max_bytes=object_store_validation_strict_max_bytes,
    )
    run_id = f"run-{dataset.source.digest}"
    source_id = f"src-{dataset.source.digest}"
    inspect_transform_id = f"tfm-{dataset.source.digest}-inspect"
    ingest_transform_id = f"tfm-{dataset.source.digest}-ingest"
    job_id = ingest_job_id or _lerobot_ingest_job_id(source_id)
    owner = claim_owner or created_by
    claim_token = _lerobot_claim_token(job_id, owner, inspect_started)
    media_inspection_cache = _lerobot_media_inspection_cache(lake, job_id)
    _assert_lerobot_claim_precondition(
        job_id,
        _latest_lerobot_ingest_job(lake, job_id),
        operation="claim",
        lake_uri=lake.uri,
        expected_latest_checkpoint_id=expected_latest_checkpoint_id,
        expected_latest_claim_token=expected_latest_claim_token,
        expected_checkpoint_index=expected_checkpoint_index,
    )

    if _lerobot_completed_ingest_transform_exists(lake, ingest_transform_id):
        dataset = _lerobot_inspect_media(
            adapter,
            dataset,
            media_inspection_cache=media_inspection_cache,
            workers=media_inspection_workers,
            timeout_seconds=media_inspection_timeout_seconds,
            retry_count=media_inspection_retries,
            retry_backoff_seconds=media_inspection_retry_backoff_seconds,
            retry_policy=media_inspection_retry_policy,
            execution_mode=media_inspection_execution_mode,
        )
        inspect_report = _lerobot_inspect_report_with_media(inspect_report, dataset)
        inspect_finished = datetime.now(UTC)
        keyframe_map_plan = _lerobot_keyframe_map_offload_plan(
            dataset,
            run_id=run_id,
            transform_id=ingest_transform_id,
            created_at=inspect_finished,
            threshold_bytes=keyframe_map_threshold_bytes,
            threshold_frames=keyframe_map_threshold_frames,
        )
        if keyframe_map_plan["rows"]:
            _upsert_lerobot_rows(
                lake,
                "keyframe_map_artifacts",
                "artifact_id",
                list(keyframe_map_plan["rows"]),
                KEYFRAME_MAP_ARTIFACTS_SCHEMA,
            )
        if keyframe_map_plan["referrers"]:
            _upsert_lerobot_rows(
                lake,
                "keyframe_map_artifact_referrers",
                "referrer_id",
                _lerobot_keyframe_map_referrers_for_write(lake, keyframe_map_plan),
                KEYFRAME_MAP_ARTIFACT_REFERRERS_SCHEMA,
            )
        _record_lerobot_completed_checkpoint_if_needed(
            lake,
            dataset=dataset,
            inspect_report=inspect_report,
            job_id=job_id,
            source_id=source_id,
            run_id=run_id,
            transform_id=ingest_transform_id,
            started_at=inspect_started,
            created_by=created_by,
            claim_owner=owner,
            source_arg=source,
            claim_token=claim_token,
            keyframe_map_offloads=keyframe_map_plan["by_video"],
            keyframe_map_artifacts=_lerobot_keyframe_map_artifact_progress(keyframe_map_plan),
        )
        return _already_ingested_report(
            lake,
            source_id,
            dataset.source.uri,
            dataset.source.checksum,
            run_id,
            inspect_report,
            auth_ref=auth_ref,
            created_by=created_by,
            adapter_name="lerobot",
            ingest_job_id=job_id,
        )

    _assert_lerobot_ingest_claimable(lake, job_id)

    now = datetime.now(UTC)
    registration = _register_resolved_source(
        lake,
        dataset.source,
        adapter="lerobot",
        inspect_report=inspect_report,
        auth_ref=auth_ref,
    )
    expected_observations = int(
        inspect_report.get("frame_count") or inspect_report.get("message_count") or 0
    )
    existing_observation_ids = _lerobot_existing_observation_ids(lake, run_id)
    existing_observation_count = (
        len(existing_observation_ids)
        if existing_observation_ids is not None
        else int(lake.table("observations").count_rows(f"run_id = '{run_id}'"))
    )
    observations_table = lake.table("observations")
    batch: list[dict] = []
    episode_state: dict[int, dict[str, Any]] = {}
    progress = _lerobot_progress_template(
        dataset,
        expected_observations=expected_observations,
        existing_observations=existing_observation_count,
        batch_size=batch_size,
        source_identity=inspect_report.get("source_identity") or {},
    )
    progress["claim"] = _lerobot_new_claim_payload(
        owner=owner,
        token=claim_token,
        generation=_next_lerobot_claim_generation(lake, job_id),
        lease=claim_lease,
        heartbeat=claim_heartbeat,
    )
    checkpoint_index = _next_lerobot_checkpoint_index(lake, job_id)
    keyframe_map_plan = _empty_lerobot_keyframe_map_offload_plan(
        threshold_bytes=keyframe_map_threshold_bytes,
        threshold_frames=keyframe_map_threshold_frames,
    )
    progress["keyframe_map_artifacts"] = _lerobot_keyframe_map_artifact_progress(keyframe_map_plan)
    latest_before_claim = _latest_lerobot_ingest_job(lake, job_id)
    _assert_lerobot_claim_precondition(
        job_id,
        latest_before_claim,
        operation="claim",
        lake_uri=lake.uri,
        expected_latest_checkpoint_id=expected_latest_checkpoint_id,
        expected_latest_claim_token=expected_latest_claim_token,
        expected_checkpoint_index=expected_checkpoint_index,
    )
    claim_checkpoint_id = f"{job_id}:{int(checkpoint_index):08d}"
    if latest_before_claim is not None:
        # An existing row to CAS against (retrying/re-claiming a job that has
        # been ingested before -- e.g. after a prior abandon/steal). The very
        # first claim on a brand-new job_id has no prior row to gate on; see
        # the bootstrap path in ``_record_lerobot_ingest_checkpoint`` below.
        _lerobot_claim_cas_supersede(
            lake,
            job_id=job_id,
            prior_checkpoint_id=str(latest_before_claim.get("checkpoint_id") or ""),
            prior_claim_token=latest_before_claim.get("claim_token"),
            new_checkpoint_id=claim_checkpoint_id,
            operation="claim",
        )
    _record_lerobot_ingest_checkpoint(
        lake,
        dataset=dataset,
        job_id=job_id,
        source_id=source_id,
        run_id=run_id,
        transform_id=ingest_transform_id,
        progress=progress,
        status="running",
        phase="claimed",
        checkpoint_index=checkpoint_index,
        started_at=inspect_started,
        created_by=created_by,
        claim_owner=owner,
        source_arg=source,
        claim_token=claim_token,
        enforce_bootstrap_cas=latest_before_claim is None,
    )
    checkpoint_index += 1

    try:
        dataset = _lerobot_inspect_media(
            adapter,
            dataset,
            media_inspection_cache=media_inspection_cache,
            workers=media_inspection_workers,
            timeout_seconds=media_inspection_timeout_seconds,
            retry_count=media_inspection_retries,
            retry_backoff_seconds=media_inspection_retry_backoff_seconds,
            retry_policy=media_inspection_retry_policy,
            execution_mode=media_inspection_execution_mode,
        )
        inspect_report = _lerobot_inspect_report_with_media(inspect_report, dataset)
        inspect_finished = datetime.now(UTC)
        keyframe_map_plan = _lerobot_keyframe_map_offload_plan(
            dataset,
            run_id=run_id,
            transform_id=ingest_transform_id,
            created_at=inspect_finished,
            threshold_bytes=keyframe_map_threshold_bytes,
            threshold_frames=keyframe_map_threshold_frames,
        )
        progress["keyframe_map_artifacts"] = _lerobot_keyframe_map_artifact_progress(
            keyframe_map_plan
        )
        progress["media_inspection"] = _lerobot_sanitized_media_inspection(
            dataset.media_inspection,
            keyframe_map_plan["by_video"],
        )
        progress["video_diagnostics"] = _lerobot_video_diagnostics(dataset.video_files)
        _record_lerobot_ingest_checkpoint(
            lake,
            dataset=dataset,
            job_id=job_id,
            source_id=source_id,
            run_id=run_id,
            transform_id=ingest_transform_id,
            progress=progress,
            status="running",
            phase="media-inspection",
            checkpoint_index=checkpoint_index,
            started_at=inspect_started,
            created_by=created_by,
            claim_owner=owner,
            source_arg=source,
            claim_token=claim_token,
        )
        checkpoint_index += 1
    except Exception as exc:
        _record_lerobot_failed_checkpoint(
            lake,
            dataset=dataset,
            job_id=job_id,
            source_id=source_id,
            run_id=run_id,
            transform_id=ingest_transform_id,
            progress=progress,
            phase="media-inspection",
            checkpoint_index=checkpoint_index,
            started_at=inspect_started,
            created_by=created_by,
            claim_owner=owner,
            source_arg=source,
            claim_token=claim_token,
            error=exc,
        )
        raise

    try:
        iter_kwargs: dict[str, Any] = {"batch_size": batch_size}
        if _callable_accepts_keyword(adapter.iter_frame_batches, "storage_options"):
            iter_kwargs["storage_options"] = storage_options
        if _callable_accepts_keyword(adapter.iter_frame_batches, "auth_ref"):
            iter_kwargs["auth_ref"] = auth_ref
        for frame_batch in adapter.iter_frame_batches(dataset, **iter_kwargs):
            batch_seen = 0
            batch_written = 0
            progress["bytes_scanned"] = int(progress["bytes_scanned"]) + int(
                frame_batch.bytes_scanned
            )
            for frame in frame_batch.rows:
                row = _lerobot_observation_row(
                    dataset,
                    frame,
                    run_id=run_id,
                    transform_id=ingest_transform_id,
                    created_at=now,
                    episode_state=episode_state,
                )
                batch_seen += 1
                progress["rows_seen"] = int(progress["rows_seen"]) + 1
                if _lerobot_observation_already_present(
                    observations_table,
                    row["observation_id"],
                    known_ids=existing_observation_ids,
                ):
                    progress["rows_skipped_existing"] = int(progress["rows_skipped_existing"]) + 1
                    continue
                if existing_observation_ids is not None:
                    existing_observation_ids.add(row["observation_id"])
                batch.append(row)
                progress["rows_written"]["observations"] = (
                    int(progress["rows_written"]["observations"]) + 1
                )
                batch_written += 1
                progress["last_observation_id"] = row["observation_id"]
            _flush(observations_table, batch)
            progress["last_checkpoint"] = _lerobot_record_batch_progress(
                progress,
                dataset,
                frame_batch,
                rows_seen=batch_seen,
                observations_written=batch_written,
            )
            _record_lerobot_ingest_checkpoint(
                lake,
                dataset=dataset,
                job_id=job_id,
                source_id=source_id,
                run_id=run_id,
                transform_id=ingest_transform_id,
                progress=progress,
                status="running",
                phase="frame-batch",
                checkpoint_index=checkpoint_index,
                started_at=inspect_started,
                created_by=created_by,
                claim_owner=owner,
                source_arg=source,
                claim_token=claim_token,
            )
            checkpoint_index += 1
    except Exception as exc:
        _record_lerobot_failed_checkpoint(
            lake,
            dataset=dataset,
            job_id=job_id,
            source_id=source_id,
            run_id=run_id,
            transform_id=ingest_transform_id,
            progress=progress,
            phase="frame-batch",
            checkpoint_index=checkpoint_index,
            started_at=inspect_started,
            created_by=created_by,
            claim_owner=owner,
            source_arg=source,
            claim_token=claim_token,
            error=exc,
        )
        raise
    _flush(observations_table, batch)

    video_rows, encoding_rows, video_ids_by_episode = _lerobot_video_rows(
        dataset,
        run_id=run_id,
        episode_state=episode_state,
        transform_id=ingest_transform_id,
        created_at=now,
        keyframe_map_offloads=keyframe_map_plan["by_video"],
    )
    episode_rows = _lerobot_episode_rows(
        dataset,
        run_id=run_id,
        episode_state=episode_state,
        video_ids_by_episode=video_ids_by_episode,
        transform_id=ingest_transform_id,
        created_at=now,
    )
    scenario_rows = _lerobot_scenario_rows(
        dataset,
        run_id=run_id,
        episode_state=episode_state,
        transform_id=ingest_transform_id,
        created_at=now,
    )
    progress["rows_written"]["episodes"] = len(episode_rows)
    progress["rows_written"]["scenarios"] = len(scenario_rows)
    progress["rows_written"]["videos"] = len(video_rows)
    progress["rows_written"]["video_encodings"] = len(encoding_rows)
    progress["rows_written"]["keyframe_map_artifacts"] = len(keyframe_map_plan["rows"])
    progress["rows_written"]["keyframe_map_artifact_referrers"] = len(
        keyframe_map_plan["referrers"]
    )
    progress["video_diagnostics"] = _lerobot_video_diagnostics(dataset.video_files)
    progress["keyframe_map_artifacts"] = _lerobot_keyframe_map_artifact_progress(keyframe_map_plan)
    progress["media_inspection"] = _lerobot_sanitized_media_inspection(
        dataset.media_inspection,
        keyframe_map_plan["by_video"],
    )
    decoded_conformance = _lerobot_decoded_frame_conformance_report(
        dataset,
        encoding_rows,
        decoded_frame_conformance,
        keyframe_map_artifacts=keyframe_map_plan["rows"],
    )
    if decoded_conformance is not None:
        progress["decoded_frame_conformance"] = dict(decoded_conformance)
    _record_lerobot_ingest_checkpoint(
        lake,
        dataset=dataset,
        job_id=job_id,
        source_id=source_id,
        run_id=run_id,
        transform_id=ingest_transform_id,
        progress=progress,
        status="running",
        phase="metadata-ready",
        checkpoint_index=checkpoint_index,
        started_at=inspect_started,
        created_by=created_by,
        claim_owner=owner,
        source_arg=source,
        claim_token=claim_token,
    )
    checkpoint_index += 1

    start_time_ns = inspect_report["start_time_ns"]
    end_time_ns = inspect_report["end_time_ns"]
    duration_ns = inspect_report["duration_ns"]
    source_validation = dict(
        (progress["source_identity"].get("object_store_validation") or {})
        if isinstance(progress.get("source_identity"), dict)
        else {}
    )
    run_metadata = [
        {"key": "profile", "value": "lerobot"},
        {"key": "library", "value": inspect_report["library"]},
        {"key": "codebase_version", "value": dataset.codebase_version},
        {"key": "fps", "value": str(dataset.info.get("fps") or "")},
        {"key": "robot_type", "value": str(dataset.info.get("robot_type") or "")},
        {"key": "total_episodes", "value": str(len(dataset.episodes))},
        {"key": "total_frames", "value": str(expected_observations)},
        {"key": "camera_keys", "value": ",".join(dataset.camera_keys)},
        {"key": "integrity.status", "value": "complete"},
        {
            "key": "source_identity.kind",
            "value": str(progress["source_identity"].get("kind") or ""),
        },
    ]
    if source_validation:
        run_metadata.extend(
            [
                {
                    "key": "source_identity.validation_policy",
                    "value": str(source_validation.get("policy") or ""),
                },
                {
                    "key": "source_identity.assurance",
                    "value": str(source_validation.get("assurance") or ""),
                },
                {
                    "key": "source_identity.warning_count",
                    "value": str(len(source_validation.get("warnings") or ())),
                },
            ]
        )
    if dataset.source.repo_id:
        run_metadata.append({"key": "hf_repo_id", "value": dataset.source.repo_id})
    if dataset.source.revision:
        run_metadata.append({"key": "hf_revision", "value": dataset.source.revision})

    run_row = {
        "run_id": run_id,
        "run_kind": "dataset",
        "source": "lerobot",
        "source_id": registration.source_id,
        "raw_uri": registration.uri,
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
        "duration_ns": duration_ns,
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
            "source": "lerobot-boundary",
            "transform_id": ingest_transform_id,
            "created_at": now,
        }
        for event_type, timestamp_ns in (
            ("run_start", start_time_ns),
            ("run_end", end_time_ns),
        )
    ]

    output_tables = [
        "runs",
        "episodes",
        "observations",
        "scenarios",
        "events",
        _LEROBOT_CHECKPOINT_TABLE,
    ]
    if video_rows:
        output_tables.append("videos")
    if encoding_rows:
        output_tables.append("video_encodings")
    if keyframe_map_plan["rows"]:
        output_tables.append("keyframe_map_artifacts")
    if keyframe_map_plan["referrers"]:
        output_tables.append("keyframe_map_artifact_referrers")

    ingest_finished = datetime.now(UTC)
    unmapped_feature_keys = sorted(
        {
            key
            for state in episode_state.values()
            for key in (state.get("unmapped_feature_keys") or set())
        }
    )
    progress["status"] = "completed"
    ingest_params = {
        "adapter": "lerobot",
        "run_id": run_id,
        "codebase_version": dataset.codebase_version,
        "batch_size": batch_size,
        "episode_count": len(dataset.episodes),
        "frame_count": expected_observations,
        "camera_keys": list(dataset.camera_keys),
        "video_file_count": len(dataset.video_files),
        "video_files": _lerobot_video_file_summaries(
            dataset.video_files,
            keyframe_map_offloads=keyframe_map_plan["by_video"],
        ),
        "video_diagnostics": progress["video_diagnostics"],
        "media_inspection": progress["media_inspection"],
        "keyframe_map_artifacts": progress["keyframe_map_artifacts"],
        "data_files": list(dataset.data_files),
        "unmapped_feature_keys": unmapped_feature_keys,
        "integrity": {"status": "complete", "recovered": 0, "reason": None},
        "source_identity": progress["source_identity"],
        "progress": progress,
    }
    if decoded_conformance is not None:
        ingest_params["decoded_frame_conformance"] = decoded_conformance
    transform_rows = [
        {
            "transform_id": inspect_transform_id,
            "kind": "inspect",
            "source_id": registration.source_id,
            "input_uris": list(dataset.source.input_uris),
            "output_tables": [],
            "params": json.dumps(
                {
                    "adapter": "lerobot",
                    "codebase_version": dataset.codebase_version,
                    "native_loader": dataset.native_loader,
                    "media_inspection": progress["media_inspection"],
                },
                sort_keys=True,
            ),
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
            "input_uris": list(dataset.source.input_uris),
            "output_tables": output_tables,
            "params": json.dumps(
                ingest_params,
                sort_keys=True,
            ),
            "status": "completed",
            "started_at": inspect_started,
            "finished_at": ingest_finished,
            "created_by": created_by,
            "created_at": now,
        },
    ]

    try:
        _upsert_lerobot_rows(lake, "runs", "run_id", [run_row], RUNS_SCHEMA)
        _upsert_lerobot_rows(lake, "episodes", "episode_id", episode_rows, EPISODES_SCHEMA)
        _upsert_lerobot_rows(lake, "scenarios", "scenario_id", scenario_rows, SCENARIOS_SCHEMA)
        _upsert_lerobot_rows(lake, "videos", "video_id", video_rows, VIDEOS_SCHEMA)
        _upsert_lerobot_rows(
            lake,
            "video_encodings",
            "encoding_id",
            encoding_rows,
            VIDEO_ENCODINGS_SCHEMA,
        )
        _upsert_lerobot_rows(
            lake,
            "keyframe_map_artifacts",
            "artifact_id",
            list(keyframe_map_plan["rows"]),
            KEYFRAME_MAP_ARTIFACTS_SCHEMA,
        )
        _upsert_lerobot_rows(
            lake,
            "keyframe_map_artifact_referrers",
            "referrer_id",
            _lerobot_keyframe_map_referrers_for_write(lake, keyframe_map_plan),
            KEYFRAME_MAP_ARTIFACT_REFERRERS_SCHEMA,
        )
        _upsert_lerobot_rows(lake, "events", "event_id", event_rows, EVENTS_SCHEMA)
        _upsert_lerobot_rows(
            lake,
            "transform_runs",
            "transform_id",
            transform_rows,
            TRANSFORM_RUNS_SCHEMA,
        )

        compaction = _finalize_ingest(
            lake,
            compact=compact,
            prune_versions=prune_versions,
            retain_versions=retain_versions,
            created_by=created_by,
            index_predicates=index_predicates,
        )
        for _transform_row in transform_rows:
            emit_transform_lineage(lake, _transform_row)

        _record_lerobot_ingest_checkpoint(
            lake,
            dataset=dataset,
            job_id=job_id,
            source_id=source_id,
            run_id=run_id,
            transform_id=ingest_transform_id,
            progress=progress,
            status="completed",
            phase="finalized",
            checkpoint_index=checkpoint_index,
            started_at=inspect_started,
            created_by=created_by,
            claim_owner=owner,
            source_arg=source,
            claim_token=claim_token,
        )
    except Exception as exc:
        _record_lerobot_failed_checkpoint(
            lake,
            dataset=dataset,
            job_id=job_id,
            source_id=source_id,
            run_id=run_id,
            transform_id=ingest_transform_id,
            progress=progress,
            phase="finalize",
            checkpoint_index=checkpoint_index,
            started_at=inspect_started,
            created_by=created_by,
            claim_owner=owner,
            source_arg=source,
            claim_token=claim_token,
            error=exc,
        )
        raise

    return IngestReport(
        lake_uri=lake.uri,
        compaction=compaction,
        source=registration,
        run_id=run_id,
        already_ingested=False,
        transform_id=ingest_transform_id,
        ingest_job_id=job_id,
        rows_added={
            "integration_sources": 1 if registration.created else 0,
            "runs": 1,
            "episodes": len(episode_rows),
            "observations": int(progress["rows_written"]["observations"]),
            "scenarios": len(scenario_rows),
            "videos": len(video_rows),
            "video_encodings": len(encoding_rows),
            "keyframe_map_artifacts": len(keyframe_map_plan["rows"]),
            "keyframe_map_artifact_referrers": len(keyframe_map_plan["referrers"]),
            "events": len(event_rows),
            "transform_runs": len(transform_rows),
        },
        observations_by_topic={"lerobot.frames": int(progress["rows_seen"])},
        message_count=int(progress["rows_seen"]),
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        duration_ns=duration_ns,
        decode_by_status={"decoded": int(progress["rows_seen"])},
        decode_by_encoding={"parquet": int(progress["rows_seen"])},
    )


def _lerobot_observation_rows(
    dataset,
    *,
    run_id: str,
    transform_id: str,
    created_at: datetime,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    episode_state: dict[int, dict[str, Any]] = {}
    first_camera = dataset.camera_keys[0] if dataset.camera_keys else None
    for frame in dataset.frames:
        rows.append(
            _lerobot_observation_row(
                dataset,
                frame,
                run_id=run_id,
                transform_id=transform_id,
                created_at=created_at,
                episode_state=episode_state,
                first_camera=first_camera,
            )
        )
    return rows, episode_state


def _lerobot_observation_row(
    dataset,
    frame: dict[str, Any],
    *,
    run_id: str,
    transform_id: str,
    created_at: datetime,
    episode_state: dict[int, dict[str, Any]],
    first_camera: str | None = None,
) -> dict[str, Any]:
    if first_camera is None and dataset.camera_keys:
        first_camera = dataset.camera_keys[0]
    topic = f"observation.images.{first_camera}" if first_camera else "lerobot.frames"
    sensor_id = first_camera or "state_action"
    episode_index = int(frame["_episode_index"])
    frame_index = int(frame["_frame_index"])
    observation_id = f"{run_id}:frame:{episode_index:06d}:{frame_index:06d}"
    state = episode_state.setdefault(
        episode_index,
        {
            "episode_index": episode_index,
            "observation_ids": [],
            "start_time_ns": int(frame["_timestamp_ns"]),
            "end_time_ns": int(frame["_timestamp_ns"]),
            "frame_count": 0,
            "task": str(frame["_task"]),
            "task_index": int(frame["_task_index"]),
            "split": "train",
            "unmapped_feature_keys": set(),
        },
    )
    if observation_id not in state["observation_ids"]:
        state["observation_ids"].append(observation_id)
        state["frame_count"] = int(state["frame_count"]) + 1
    state["start_time_ns"] = min(int(state["start_time_ns"]), int(frame["_timestamp_ns"]))
    state["end_time_ns"] = max(int(state["end_time_ns"]), int(frame["_timestamp_ns"]))
    state.setdefault("unmapped_feature_keys", set()).update((frame.get("_unmapped") or {}).keys())
    payload = {
        "format": "lerobot",
        "codebase_version": dataset.codebase_version,
        "source_parquet": frame.get("_source_parquet"),
        "source_index": frame.get("index"),
        "source_observation_id": frame.get("observation_id"),
        "episode_index": episode_index,
        "frame_index": frame_index,
        "task_index": int(frame["_task_index"]),
        "task": frame["_task"],
        "images": frame.get("_images") or {},
        "unmapped": frame.get("_unmapped") or {},
    }
    return {
        "observation_id": observation_id,
        "run_id": run_id,
        "episode_id": _lerobot_episode_id(run_id, episode_index),
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp_ns": int(frame["_timestamp_ns"]),
        "sensor_id": sensor_id,
        "topic": topic,
        "modality": "video" if first_camera else "state_action",
        "task_id": str(frame["_task"]),
        "raw_uri": f"{dataset.source.uri}/{frame.get('_source_parquet')}",
        "raw_channel": "lerobot.frame",
        "raw_log_time_ns": int(frame["_timestamp_ns"]),
        "raw_sequence": int(frame["_global_index"]),
        "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        "payload_blob": None,
        "message_encoding": "parquet",
        "schema_encoding": "lerobot-feature",
        "decode_status": "decoded",
        "decode_error": None,
        "state_vector": list(frame.get("_state_vector") or []),
        "action_vector": list(frame.get("_action_vector") or []),
        "caption": str(frame.get("_caption") or frame["_task"]),
        "transform_id": transform_id,
        "created_at": created_at,
    }


def _lerobot_episode_rows(
    dataset,
    *,
    run_id: str,
    episode_state: dict[int, dict[str, Any]],
    video_ids_by_episode: dict[int, list[str]],
    transform_id: str,
    created_at: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in dataset.episodes:
        episode_index = int(episode["episode_index"])
        state = episode_state.get(episode_index)
        if state is None:
            continue
        provenance = {
            "adapter": "lerobot",
            "source_uri": dataset.source.uri,
            "codebase_version": dataset.codebase_version,
            "source_episode_id": episode.get("episode_id"),
            "source_scenario_id": episode.get("scenario_id"),
            "split": episode.get("split") or "train",
            "data_files": list(dataset.data_files),
        }
        rows.append(
            {
                "episode_id": _lerobot_episode_id(run_id, episode_index),
                "run_id": run_id,
                "episode_index": episode_index,
                "from_timestamp_ns": int(state["start_time_ns"]),
                "to_timestamp_ns": int(state["end_time_ns"]),
                "boundary_source": "lerobot-authored",
                "outcome": None,
                "frame_count": int(state["frame_count"]),
                "camera_blobs": list(video_ids_by_episode.get(episode_index, [])),
                "task_id": str(state["task"]),
                "embedding": None,
                "provenance": json.dumps(provenance, sort_keys=True, separators=(",", ":")),
                "transform_id": transform_id,
                "created_at": created_at,
            }
        )
    return rows


def _lerobot_scenario_rows(
    dataset,
    *,
    run_id: str,
    episode_state: dict[int, dict[str, Any]],
    transform_id: str,
    created_at: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in dataset.episodes:
        episode_index = int(episode["episode_index"])
        state = episode_state.get(episode_index)
        if state is None:
            continue
        scenario_id = _lerobot_scenario_id(run_id, episode_index)
        rows.append(
            {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "start_time_ns": int(state["start_time_ns"]),
                "end_time_ns": int(state["end_time_ns"]),
                "window_ns": max(0, int(state["end_time_ns"]) - int(state["start_time_ns"])),
                "is_partial": False,
                "topics": ["lerobot.frames"],
                "observation_ids": list(state["observation_ids"]),
                "observation_count": len(state["observation_ids"]),
                "scenario_type": "episode",
                "trigger_event_id": None,
                "source": "lerobot-authored",
                "parent_scenario_id": None,
                "coverage_tags": ["lerobot", "episode"],
                "summary": str(state["task"]),
                "transform_id": transform_id,
                "created_at": created_at,
            }
        )
    return rows


def _lerobot_video_rows(
    dataset,
    *,
    run_id: str,
    episode_state: dict[int, dict[str, Any]],
    transform_id: str,
    created_at: datetime,
    keyframe_map_offloads: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[str]]]:
    videos: list[dict[str, Any]] = []
    encodings: list[dict[str, Any]] = []
    ids_by_episode: dict[int, list[str]] = {}
    keyframe_map_offloads = keyframe_map_offloads or {}
    for video in dataset.video_files:
        episode_index = int(video["episode_index"])
        state = episode_state.get(episode_index)
        if state is None:
            continue
        camera_key = str(video["camera_key"])
        video_id = _lerobot_video_id(run_id, episode_index, camera_key)
        encoding_id = _lerobot_encoding_id(run_id, episode_index, camera_key)
        video_frame_count = _lerobot_video_frame_count(video, state)
        keyframe_json = _lerobot_keyframe_map_json(video)
        offload = keyframe_map_offloads.get((episode_index, camera_key))
        ids_by_episode.setdefault(episode_index, []).append(video_id)
        videos.append(
            {
                "video_id": video_id,
                "run_id": run_id,
                "episode_id": _lerobot_episode_id(run_id, episode_index),
                "episode_index": episode_index,
                "camera_key": camera_key,
                "sensor_id": camera_key,
                "topic": f"observation.images.{camera_key}",
                "from_timestamp_ns": int(state["start_time_ns"]),
                "to_timestamp_ns": int(state["end_time_ns"]),
                "frame_count": video_frame_count,
                "observation_ids": list(state["observation_ids"]),
                "raw_uri": str(video["uri"]),
                "codec": str(video.get("codec") or "unknown"),
                "uri": str(video["uri"]),
                "transform_id": transform_id,
                "created_at": created_at,
            }
        )
        encodings.append(
            {
                "encoding_id": encoding_id,
                "video_id": video_id,
                "run_id": run_id,
                "episode_id": _lerobot_episode_id(run_id, episode_index),
                "episode_index": episode_index,
                "camera_key": camera_key,
                "codec": str(video.get("codec") or "unknown"),
                "gop_size": _optional_int(video.get("gop_size")),
                "resolution": video.get("resolution"),
                "fps": _lerobot_video_fps(video, dataset),
                "frame_count": video_frame_count,
                "keyframe_map_ref": (
                    str(offload["keyframe_map_ref"])
                    if offload is not None
                    else _lerobot_keyframe_map_ref(video, keyframe_json)
                ),
                "keyframe_map_json": None if offload is not None else keyframe_json,
                "nvdec_compatible": _lerobot_video_nvdec_compatible(video),
                "source_size_bytes": int(video.get("size") or 0),
                "encoded_size_bytes": int(video.get("size") or 0),
                "data": None,
                "transform_id": transform_id,
                "created_at": created_at,
            }
        )
    return videos, encodings, ids_by_episode


def _lerobot_video_id(run_id: str, episode_index: int, camera_key: str) -> str:
    return f"vid-{run_id.removeprefix('run-')}-{int(episode_index):06d}-{_safe_id(camera_key)}"


def _lerobot_encoding_id(run_id: str, episode_index: int, camera_key: str) -> str:
    return f"enc-{run_id.removeprefix('run-')}-{int(episode_index):06d}-{_safe_id(camera_key)}"


def _lerobot_video_frame_count(video: dict[str, Any], state: dict[str, Any]) -> int:
    return int(video.get("frame_count") or state["frame_count"])


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _lerobot_video_fps(video: dict[str, Any], dataset) -> float:
    return float(video.get("fps") or dataset.info.get("fps") or 0.0)


def _lerobot_keyframe_map_json(video: dict[str, Any]) -> str:
    keyframe_map = video.get("keyframe_map") or []
    return json.dumps(keyframe_map, sort_keys=True, separators=(",", ":"))


def _lerobot_keyframe_map_ref(video: dict[str, Any], keyframe_json: str) -> str:
    if keyframe_json != "[]":
        return keyframe_map_ref(keyframe_json)
    return f"diagnostic:{video.get('path') or video.get('uri')}"


def _lerobot_video_nvdec_compatible(video: dict[str, Any]) -> bool:
    codec = str(video.get("codec") or "").lower()
    return codec in {"h264", "h265", "av1"}


def _lerobot_decoded_frame_conformance_report(
    dataset,
    encoding_rows: list[dict[str, Any]],
    options: dict[str, Any] | None,
    *,
    keyframe_map_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not options or not bool(options.get("enabled")):
        return None
    requested = str(options.get("backend") or "auto")
    samples = [dict(sample) for sample in options.get("samples") or []]
    backend = _resolve_decoded_frame_decoder_backend(requested)
    if backend is None:
        status = _decoded_frame_decoder_backend_status(requested)
        reason = str(status.get("reason") or status.get("install") or "decoder backend unavailable")
        return {
            "version": 1,
            "enabled": True,
            "mode": "decoded-frame-sample",
            "backend": {
                "requested": requested,
                "available": False,
                "status": status.get("status") or "unsupported",
                "reason": reason,
                "install": status.get("install") or reason,
            },
            "status": str(status.get("status") or "unsupported"),
            "reason": reason,
            "frames_checked": 0,
            "checks": [],
            "failures": [],
            "codec_coverage": {},
        }

    backend_name = str(getattr(backend, "name", requested))
    backend_version = str(getattr(backend, "version", "unknown"))
    supported_codecs = {str(codec).lower() for codec in getattr(backend, "supported_codecs", ())}
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    codec_coverage: dict[str, dict[str, Any]] = {}
    encodings_by_episode_camera = {
        (int(row["episode_index"]), str(row["camera_key"])): row for row in encoding_rows
    }
    keyframe_json_by_ref = {
        str(row.get("keyframe_map_ref")): str(row.get("keyframe_map_json") or "[]")
        for row in (keyframe_map_artifacts or [])
        if row.get("keyframe_map_ref")
    }
    videos_by_episode_camera = {
        (int(row["episode_index"]), str(row["camera_key"])): row for row in dataset.video_files
    }
    for sample in samples:
        camera_key = str(sample["camera_key"])
        episode_index = int(sample["episode_index"])
        frame_index = int(sample["frame_index"])
        key = (episode_index, camera_key)
        encoding = encodings_by_episode_camera.get(key)
        video = videos_by_episode_camera.get(key)
        if encoding is None or video is None:
            failure = {
                "camera_key": camera_key,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "codec": "unknown",
                "backend": backend_name,
                "reason": "missing-video-encoding",
            }
            failures.append(failure)
            checks.append({**failure, "status": "failed"})
            continue
        codec = str(encoding.get("codec") or video.get("codec") or "unknown").lower()
        coverage = codec_coverage.setdefault(
            codec,
            {
                "supported": (not supported_codecs or codec in supported_codecs),
                "frames_checked": 0,
                "failures": 0,
            },
        )
        expected_sha = str(
            sample.get("expected_pixel_sha256") or sample.get("expected_sha256") or ""
        )
        keyframe_json = encoding.get("keyframe_map_json") or keyframe_json_by_ref.get(
            str(encoding.get("keyframe_map_ref") or "")
        )
        keyframe_map = json.loads(keyframe_json or "[]")
        gop_entry = _lerobot_keyframe_entry_for_frame(keyframe_map, frame_index)
        result = _decode_lerobot_conformance_frame(
            backend,
            uri=str(video.get("uri") or video.get("path") or ""),
            camera_key=camera_key,
            episode_index=episode_index,
            frame_index=frame_index,
            codec=codec,
            keyframe_map=keyframe_map,
            gop_entry=gop_entry,
            video=video,
            encoding=encoding,
        )
        pixel_sha = str(result.get("pixel_sha256") or result.get("decoded_pixel_sha256") or "")
        pixels = result.get("pixels") or result.get("frame")
        if not pixel_sha and isinstance(pixels, bytes):
            pixel_sha = hashlib.sha256(pixels).hexdigest()
        passed = bool(expected_sha) and pixel_sha == expected_sha
        coverage["frames_checked"] = int(coverage["frames_checked"]) + 1
        check = {
            "camera_key": camera_key,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "path": video.get("path"),
            "uri": video.get("uri"),
            "codec": codec,
            "backend": backend_name,
            "gop_index": _optional_int(gop_entry.get("gop_index")) if gop_entry else None,
            "seek_frame_index": (
                _optional_int(result.get("seek_frame_index"))
                if result.get("seek_frame_index") is not None
                else (_optional_int(gop_entry.get("keyframe_frame_index")) if gop_entry else None)
            ),
            "seek_strategy": result.get("seek_strategy"),
            "fallback_reason": result.get("fallback_reason"),
            "decoded_frame_count": _optional_int(result.get("decoded_frame_count"))
            or _optional_int(result.get("frames_decoded"))
            or 1,
            "pixel_sha256": pixel_sha,
            "expected_pixel_sha256": expected_sha,
            "dtype": result.get("dtype"),
            "shape": list(result.get("shape") or []),
            "status": "passed" if passed else "failed",
        }
        checks.append(check)
        if not passed:
            coverage["failures"] = int(coverage["failures"]) + 1
            failure = {
                "camera_key": camera_key,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "codec": codec,
                "backend": backend_name,
                "reason": "pixel-sha256-mismatch",
                "expected_pixel_sha256": expected_sha,
                "pixel_sha256": pixel_sha,
            }
            failures.append(failure)
    return {
        "version": 1,
        "enabled": True,
        "mode": "decoded-frame-sample",
        "backend": {
            "name": backend_name,
            "version": backend_version,
            "requested": requested,
            "available": True,
        },
        "status": "failed" if failures else "passed",
        "reason": None if not failures else "decoded frame conformance failures",
        "frames_checked": sum(
            int(coverage["frames_checked"]) for coverage in codec_coverage.values()
        ),
        "checks": checks,
        "failures": failures,
        "codec_coverage": dict(sorted(codec_coverage.items())),
    }


def _decode_lerobot_conformance_frame(backend, **kwargs: Any) -> dict[str, Any]:
    for method_name in ("decode_lerobot_frame", "decode_frame", "frame_at"):
        method = getattr(backend, method_name, None)
        if callable(method):
            result = method(**kwargs)
            return dict(result or {})
    raise AdapterError(f"decoded-frame backend {backend!r} has no frame decode method")


def _resolve_decoded_frame_decoder_backend(name: str = "auto"):
    return _resolve_lerobot_decoded_frame_decoder(name)


def _resolve_lerobot_decoded_frame_decoder(name: str = "auto"):
    requested = str(name or "auto")
    if requested not in {"auto", "pyav"}:
        return None
    if importlib.util.find_spec("av") is None:
        return None

    class _PyAvDecodedFrameBackend:
        name = "pyav"
        version = _package_version("av")
        supported_codecs = ("h264", "h265", "av1", "mp4v", "mjpeg")

        def decode_frame(
            self,
            *,
            uri: str,
            frame_index: int,
            gop_entry: dict[str, Any] | None = None,
            **_: Any,
        ) -> dict[str, Any]:
            from lancedb_robotics.video import _decode_source_mp4_frame_pyav

            decoded = _decode_source_mp4_frame_pyav(
                uri,
                int(frame_index),
                entry=gop_entry,
                frame_entry=_lerobot_keyframe_frame_entry(gop_entry, int(frame_index)),
            )
            return {
                "pixels": decoded.frame,
                "pixel_sha256": hashlib.sha256(decoded.frame).hexdigest(),
                "dtype": decoded.dtype or "uint8",
                "shape": list(decoded.shape),
                "decoded_frame_count": decoded.frames_decoded,
                "seek_strategy": decoded.seek_strategy,
                "seek_frame_index": decoded.seek_frame_index,
                "fallback_reason": decoded.fallback_reason,
            }

    return _PyAvDecodedFrameBackend()


def _decoded_frame_decoder_backend_status(name: str = "auto") -> dict[str, Any]:
    requested = str(name or "auto")
    missing = []
    if requested in {"auto", "pyav"} and importlib.util.find_spec("av") is None:
        missing.append("av")
    available = not missing
    return {
        "requested": requested,
        "available": available,
        "status": "available" if available else "unsupported",
        "modules": ["av"],
        "missing": missing,
        "install": "lancedb-robotics[video-decode]",
        "reason": (
            "decoder backend available"
            if available
            else "decoder backend missing; install lancedb-robotics[video-decode]"
        ),
    }


def _lerobot_keyframe_entry_for_frame(
    keyframe_map: list[dict[str, Any]],
    frame_index: int,
) -> dict[str, Any] | None:
    for entry in keyframe_map:
        first = int(entry.get("first_frame_index", entry.get("keyframe_frame_index", 0)))
        last = int(entry.get("last_frame_index", first))
        if first <= int(frame_index) <= last:
            return dict(entry)
    return None


def _lerobot_keyframe_frame_entry(
    gop_entry: dict[str, Any] | None,
    frame_index: int,
) -> dict[str, Any] | None:
    if not gop_entry:
        return None
    for frame in gop_entry.get("frames") or []:
        if int(frame.get("frame_index", -1)) == int(frame_index):
            return dict(frame)
    return None


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _lerobot_video_diagnostics(video_files: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for video in video_files:
        diagnostics.extend(dict(item) for item in video.get("diagnostics") or [])
    return diagnostics


def _lerobot_inspect_report_with_media(inspect_report: dict[str, Any], dataset) -> dict[str, Any]:
    report = dict(inspect_report)
    report["video_files"] = [dict(row) for row in dataset.video_files]
    report["diagnostics"] = _lerobot_video_diagnostics(dataset.video_files)
    report["media_inspection"] = dict(dataset.media_inspection)
    return report


def _lerobot_dataset_for_ingest(
    adapter,
    source: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
    auth_ref: str | None = None,
    source_manifest_cache: LeRobotObjectStoreManifestCache | str | Path | None = None,
    object_store_validation_policy: str = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_POLICY,
    object_store_validation_sample_count: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_COUNT,
    object_store_validation_sample_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_VALIDATION_SAMPLE_BYTES,
    object_store_validation_strict_max_bytes: int = DEFAULT_LEROBOT_OBJECT_STORE_STRICT_MAX_BYTES,
):
    kwargs: dict[str, Any] = {"include_frames": False}
    if _callable_accepts_keyword(adapter.dataset, "inspect_videos"):
        kwargs["inspect_videos"] = False
    if _callable_accepts_keyword(adapter.dataset, "storage_options"):
        kwargs["storage_options"] = storage_options
    if _callable_accepts_keyword(adapter.dataset, "auth_ref"):
        kwargs["auth_ref"] = auth_ref
    if _callable_accepts_keyword(adapter.dataset, "source_manifest_cache"):
        kwargs["source_manifest_cache"] = source_manifest_cache
    if _callable_accepts_keyword(adapter.dataset, "object_store_validation_policy"):
        kwargs["object_store_validation_policy"] = object_store_validation_policy
    if _callable_accepts_keyword(adapter.dataset, "object_store_validation_sample_count"):
        kwargs["object_store_validation_sample_count"] = object_store_validation_sample_count
    if _callable_accepts_keyword(adapter.dataset, "object_store_validation_sample_bytes"):
        kwargs["object_store_validation_sample_bytes"] = object_store_validation_sample_bytes
    if _callable_accepts_keyword(adapter.dataset, "object_store_validation_strict_max_bytes"):
        kwargs["object_store_validation_strict_max_bytes"] = (
            object_store_validation_strict_max_bytes
        )
    return adapter.dataset(source, **kwargs)


def _lerobot_inspect_media(
    adapter,
    dataset,
    *,
    media_inspection_cache: tuple[dict[str, Any], ...],
    workers: int | None = None,
    timeout_seconds: float | None = None,
    retry_count: int = 0,
    retry_backoff_seconds: float = 0.0,
    retry_policy: str = "fixed",
    execution_mode: str = "thread",
):
    inspect_media = getattr(adapter, "inspect_media", None)
    if callable(inspect_media):
        kwargs: dict[str, Any] = {"media_inspection_cache": media_inspection_cache}
        if _callable_accepts_keyword(inspect_media, "media_inspection_workers"):
            kwargs["media_inspection_workers"] = workers
        if _callable_accepts_keyword(inspect_media, "media_inspection_timeout_seconds"):
            kwargs["media_inspection_timeout_seconds"] = timeout_seconds
        if _callable_accepts_keyword(inspect_media, "media_inspection_retries"):
            kwargs["media_inspection_retries"] = retry_count
        if _callable_accepts_keyword(inspect_media, "media_inspection_retry_backoff_seconds"):
            kwargs["media_inspection_retry_backoff_seconds"] = retry_backoff_seconds
        if _callable_accepts_keyword(inspect_media, "media_inspection_retry_policy"):
            kwargs["media_inspection_retry_policy"] = retry_policy
        if _callable_accepts_keyword(inspect_media, "media_inspection_execution_mode"):
            kwargs["media_inspection_execution_mode"] = execution_mode
        return inspect_media(dataset, **kwargs)

    from lancedb_robotics.adapters.lerobot_adapter import LeRobotAdapter

    return LeRobotAdapter().inspect_media(
        dataset,
        media_inspection_cache=media_inspection_cache,
        media_inspection_workers=workers,
        media_inspection_timeout_seconds=timeout_seconds,
        media_inspection_retries=retry_count,
        media_inspection_retry_backoff_seconds=retry_backoff_seconds,
        media_inspection_retry_policy=retry_policy,
        media_inspection_execution_mode=execution_mode,
    )


def _callable_accepts_keyword(function, name: str) -> bool:
    try:
        parameters = inspect_module.signature(function).parameters
    except (TypeError, ValueError):
        return True
    return name in parameters or any(
        parameter.kind == inspect_module.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _empty_lerobot_keyframe_map_offload_plan(
    *,
    threshold_bytes: int | None,
    threshold_frames: int | None,
) -> dict[str, Any]:
    return {
        "rows": [],
        "referrers": [],
        "by_video": {},
        "threshold_bytes": threshold_bytes,
        "threshold_frames": threshold_frames,
    }


def _empty_lerobot_keyframe_map_progress() -> dict[str, Any]:
    return {
        "inline_threshold_bytes": None,
        "inline_threshold_frames": None,
        "offloaded_video_count": 0,
        "artifact_count": 0,
        "referrer_count": 0,
        "artifacts": [],
    }


def _lerobot_keyframe_map_offload_plan(
    dataset,
    *,
    run_id: str,
    transform_id: str,
    created_at: datetime,
    threshold_bytes: int | None,
    threshold_frames: int | None,
) -> dict[str, Any]:
    rows_by_artifact: dict[str, dict[str, Any]] = {}
    referrers_by_id: dict[str, dict[str, Any]] = {}
    by_video: dict[tuple[int, str], dict[str, Any]] = {}
    for video in dataset.video_files:
        keyframe_json = _lerobot_keyframe_map_json(video)
        if should_inline_keyframe_map(
            keyframe_json,
            threshold_bytes=threshold_bytes,
            threshold_frames=threshold_frames,
        ):
            continue
        episode_index = int(video["episode_index"])
        camera_key = str(video["camera_key"])
        video_id = _lerobot_video_id(run_id, episode_index, camera_key)
        encoding_id = _lerobot_encoding_id(run_id, episode_index, camera_key)
        artifact = keyframe_map_artifact_row(
            keyframe_json,
            source_video_fingerprint=video.get("inspection_fingerprint"),
            inspection_id=video.get("inspection_id"),
            source_uri=video.get("uri"),
            source_path=video.get("path"),
            encoding_id=encoding_id,
            video_id=video_id,
            run_id=run_id,
            episode_id=_lerobot_episode_id(run_id, episode_index),
            episode_index=episode_index,
            camera_key=camera_key,
            transform_id=transform_id,
            created_at=created_at,
        )
        rows_by_artifact.setdefault(str(artifact["artifact_id"]), artifact)
        referrer = keyframe_map_artifact_referrer_row(artifact)
        referrers_by_id[str(referrer["referrer_id"])] = referrer
        by_video[(episode_index, camera_key)] = _lerobot_keyframe_map_artifact_summary(
            artifact,
            referrer=referrer,
        )
    return {
        "rows": list(rows_by_artifact.values()),
        "referrers": list(referrers_by_id.values()),
        "by_video": by_video,
        "threshold_bytes": threshold_bytes,
        "threshold_frames": threshold_frames,
    }


def _lerobot_keyframe_map_artifact_summary(
    row: dict[str, Any],
    *,
    referrer: dict[str, Any] | None = None,
    referrer_count: int | None = None,
) -> dict[str, Any]:
    summary = {
        "keyframe_map_inline": False,
        "keyframe_map_ref": row["keyframe_map_ref"],
        "keyframe_map_artifact_id": row["artifact_id"],
        "keyframe_map_json_size_bytes": int(row["json_size_bytes"]),
        "keyframe_map_frame_count": int(row["frame_count"]),
        "keyframe_map_gop_count": int(row["gop_count"]),
        "source_video_fingerprint": row.get("source_video_fingerprint"),
        "inspection_id": row.get("inspection_id"),
        "encoding_id": row.get("encoding_id"),
        "video_id": row.get("video_id"),
        "episode_index": row.get("episode_index"),
        "camera_key": row.get("camera_key"),
    }
    if referrer is not None:
        summary["keyframe_map_referrer_id"] = referrer.get("referrer_id")
    if referrer_count is not None:
        summary["keyframe_map_referrer_count"] = referrer_count
    return summary


def _lerobot_keyframe_map_artifact_progress(plan: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in plan.get("rows") or []]
    referrers = [dict(row) for row in plan.get("referrers") or []]
    referrer_counts: dict[str, int] = {}
    for referrer in referrers:
        artifact_id = str(referrer.get("artifact_id") or "")
        if artifact_id:
            referrer_counts[artifact_id] = referrer_counts.get(artifact_id, 0) + 1
    return {
        "inline_threshold_bytes": plan.get("threshold_bytes"),
        "inline_threshold_frames": plan.get("threshold_frames"),
        "offloaded_video_count": len(plan.get("by_video") or {}),
        "artifact_count": len(rows),
        "referrer_count": len(referrers),
        "artifacts": [
            _lerobot_keyframe_map_artifact_summary(
                row,
                referrer_count=referrer_counts.get(str(row.get("artifact_id") or ""), 0),
            )
            for row in sorted(rows, key=lambda item: str(item.get("artifact_id") or ""))
        ],
    }


def _lerobot_keyframe_map_referrers_for_write(
    lake: Lake,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    referrers = [dict(row) for row in plan.get("referrers") or []]
    if not referrers:
        return []
    table_version = int(lake.table("video_encodings").version)
    for row in referrers:
        row["referrer_table_version"] = table_version
    return referrers


def _lerobot_sanitized_media_inspection(
    media_inspection: dict[str, Any],
    keyframe_map_offloads: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    sanitized = dict(media_inspection or {})
    videos = []
    for row in sanitized.get("videos") or []:
        if isinstance(row, dict):
            videos.append(_lerobot_sanitized_video_summary(row, keyframe_map_offloads))
    sanitized["videos"] = videos
    sanitized["keyframe_map_offloaded_video_count"] = len(keyframe_map_offloads)
    return sanitized


def _lerobot_sanitized_video_summary(
    row: dict[str, Any],
    keyframe_map_offloads: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    sanitized = dict(row)
    key = (_optional_int(row.get("episode_index")), str(row.get("camera_key") or ""))
    offload = keyframe_map_offloads.get(key)
    if offload is not None:
        sanitized.pop("keyframe_map", None)
        sanitized.update(offload)
        return sanitized
    keyframe_map = sanitized.get("keyframe_map")
    if keyframe_map:
        keyframe_json = json.dumps(keyframe_map, sort_keys=True, separators=(",", ":"))
        frame_count, gop_count = keyframe_map_shape(keyframe_map_entries_from_json(keyframe_json))
        sanitized["keyframe_map_inline"] = True
        sanitized["keyframe_map_ref"] = keyframe_map_ref(keyframe_json)
        sanitized["keyframe_map_json_size_bytes"] = len(keyframe_json.encode())
        sanitized["keyframe_map_frame_count"] = frame_count
        sanitized["keyframe_map_gop_count"] = gop_count
    return sanitized


def _lerobot_video_file_summaries(
    video_files: tuple[dict[str, Any], ...],
    *,
    keyframe_map_offloads: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    keys = (
        "camera_key",
        "episode_index",
        "path",
        "uri",
        "codec",
        "codec_tag",
        "codec_profile",
        "resolution",
        "fps",
        "frame_count",
        "expected_frame_count",
        "gop_size",
        "size",
        "inspection_id",
        "inspection_status",
        "inspection_fingerprint",
        "inspection_bytes_read",
        "inspection_duration_ms",
        "inspection_attempts",
        "inspection_retries",
        "inspection_timeouts",
        "inspection_error_class",
        "inspection_retry_class",
        "inspection_retryable",
        "inspection_retry_policy",
        "inspection_attempt_errors",
        "inspection_reused",
        "inspection_reused_from",
        "inspection_error",
        "diagnostics",
    )
    summaries: list[dict[str, Any]] = []
    keyframe_map_offloads = keyframe_map_offloads or {}
    for video in video_files:
        summary = {key: video.get(key) for key in keys if key in video}
        offload = keyframe_map_offloads.get((int(video["episode_index"]), str(video["camera_key"])))
        if offload is not None:
            summary.update(offload)
        summaries.append(summary)
    return summaries


def _lerobot_episode_id(run_id: str, episode_index: int) -> str:
    return f"{run_id}:episode:{int(episode_index):06d}"


def _lerobot_scenario_id(run_id: str, episode_index: int) -> str:
    return f"scn-{run_id.removeprefix('run-')}-episode-{int(episode_index):06d}"


def _lerobot_existing_observation_ids(lake: Lake, run_id: str) -> set[str] | None:
    """Return known LeRobot observation ids for small partial retries.

    Fresh ingests take the zero-row fast path. Very large partial retries use
    per-row existence checks instead of materializing a huge id set.
    """
    table = lake.table("observations")
    existing = int(table.count_rows(f"run_id = '{run_id}'"))
    if existing == 0:
        return set()
    if existing > _LEROBOT_EXISTING_ID_SET_LIMIT:
        return None
    rows = (
        table.search()
        .where(f"run_id = '{run_id}'")
        .select(["observation_id"])
        .limit(existing)
        .to_arrow()
        .to_pylist()
    )
    return {str(row["observation_id"]) for row in rows}


def _lerobot_observation_already_present(
    table,
    observation_id: str,
    *,
    known_ids: set[str] | None,
) -> bool:
    if known_ids is not None:
        return observation_id in known_ids
    escaped = observation_id.replace("'", "''")
    return int(table.count_rows(f"observation_id = '{escaped}'")) > 0


def _lerobot_progress_template(
    dataset,
    *,
    expected_observations: int,
    existing_observations: int,
    batch_size: int,
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "running",
        "resume_strategy": "skip-existing-observation-id",
        "checkpoint_grain": "data-file-row-group-batch",
        "batch_size": batch_size,
        "expected_rows": expected_observations,
        "existing_rows_before": existing_observations,
        "rows_seen": 0,
        "rows_written": {
            "observations": 0,
            "episodes": 0,
            "scenarios": 0,
            "videos": 0,
            "video_encodings": 0,
            "keyframe_map_artifacts": 0,
            "keyframe_map_artifact_referrers": 0,
        },
        "rows_skipped_existing": 0,
        "bytes_scanned": 0,
        "data_files": [],
        "source_identity": dict(source_identity),
        "media_inspection": dict(getattr(dataset, "media_inspection", {}) or {}),
        "keyframe_map_artifacts": _empty_lerobot_keyframe_map_progress(),
        "video_diagnostics": _lerobot_video_diagnostics(dataset.video_files),
        "last_checkpoint": None,
        "last_observation_id": None,
    }


def _lerobot_record_batch_progress(
    progress: dict[str, Any],
    dataset,
    frame_batch,
    *,
    rows_seen: int,
    observations_written: int,
) -> dict[str, Any]:
    file_entry = _lerobot_progress_file_entry(progress, dataset, frame_batch.data_file)
    group_entry = _lerobot_progress_row_group_entry(file_entry, frame_batch.row_group)
    batch_entry = {
        "data_file": frame_batch.data_file,
        "row_group": frame_batch.row_group,
        "batch_index": frame_batch.batch_index,
        "rows_seen": rows_seen,
        "rows_written": {"observations": observations_written},
        "bytes_scanned": frame_batch.bytes_scanned,
    }
    group_entry["batches"].append(batch_entry)
    return {
        "data_file": frame_batch.data_file,
        "row_group": frame_batch.row_group,
        "batch_index": frame_batch.batch_index,
        "rows_seen": int(progress["rows_seen"]),
        "rows_written": dict(progress["rows_written"]),
        "bytes_scanned": int(progress["bytes_scanned"]),
    }


def _lerobot_progress_file_entry(
    progress: dict[str, Any],
    dataset,
    data_file: str,
) -> dict[str, Any]:
    for entry in progress["data_files"]:
        if entry["path"] == data_file:
            return entry
    fingerprint = {
        "kind": progress["source_identity"].get("kind") or "source-digest-path",
        "value": f"{dataset.source.digest}:{data_file}",
    }
    entry = {"path": data_file, "fingerprint": fingerprint, "row_groups": []}
    progress["data_files"].append(entry)
    return entry


def _lerobot_progress_row_group_entry(file_entry: dict[str, Any], row_group: int) -> dict[str, Any]:
    for entry in file_entry["row_groups"]:
        if entry["row_group"] == row_group:
            return entry
    entry = {"row_group": row_group, "batches": []}
    file_entry["row_groups"].append(entry)
    return entry


def _lerobot_checkpoint_rows(lake: Lake) -> list[dict[str, Any]]:
    return lake.table(_LEROBOT_CHECKPOINT_TABLE).to_arrow().to_pylist()


def _lerobot_checkpoint_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("updated_at") or row.get("created_at"),
        int(row.get("checkpoint_index") or 0),
        str(row.get("checkpoint_id") or ""),
    )


def _hydrate_lerobot_checkpoint_row(row: dict[str, Any]) -> dict[str, Any]:
    hydrated = dict(row)
    for key in ("hf_download_json", "source_identity_json", "progress_json"):
        value = hydrated.get(key)
        output_key = key.removesuffix("_json")
        if not value:
            hydrated[output_key] = {}
            continue
        try:
            hydrated[output_key] = json.loads(value)
        except json.JSONDecodeError:
            hydrated[output_key] = {"raw": value}
    progress = hydrated.get("progress") or {}
    if isinstance(progress, dict):
        rows_written = progress.get("rows_written")
        if isinstance(rows_written, dict):
            hydrated["rows_written"] = dict(rows_written)
    hydrated["adapter"] = "lerobot"
    return hydrated


def _lerobot_media_inspection_samples(
    lake: Lake | None,
    *,
    checkpoint_rows: Sequence[Mapping[str, Any]] | None,
    transform_rows: Sequence[Mapping[str, Any]] | None,
    job_id: str | None,
    source_id: str | None,
    source_uri: str | None,
    storage_tier: str | None,
    provider: str | None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    if checkpoint_rows is None and lake is not None:
        raw_checkpoint_rows = _lerobot_checkpoint_rows(lake)
    else:
        raw_checkpoint_rows = [dict(row) for row in checkpoint_rows or []]
    for row in raw_checkpoint_rows:
        sample = _lerobot_media_inspection_sample_from_checkpoint(
            row,
            storage_tier_override=None,
            provider_override=None,
        )
        if sample is None or not _lerobot_media_sample_matches(
            sample,
            job_id=job_id,
            source_id=source_id,
            source_uri=source_uri,
            storage_tier=storage_tier,
            provider=provider,
        ):
            continue
        key = _lerobot_media_sample_dedupe_key(sample)
        if key in seen:
            continue
        seen.add(key)
        samples.append(sample)

    if transform_rows is None and lake is not None:
        try:
            raw_transform_rows = lake.table("transform_runs").to_arrow().to_pylist()
        except Exception:  # noqa: BLE001 - older partial lakes may not have transform telemetry
            raw_transform_rows = []
    else:
        raw_transform_rows = [dict(row) for row in transform_rows or []]
    for row in raw_transform_rows:
        sample = _lerobot_media_inspection_sample_from_transform(
            row,
            storage_tier_override=None,
            provider_override=None,
        )
        if sample is None or not _lerobot_media_sample_matches(
            sample,
            job_id=job_id,
            source_id=source_id,
            source_uri=source_uri,
            storage_tier=storage_tier,
            provider=provider,
        ):
            continue
        key = _lerobot_media_sample_dedupe_key(sample)
        if key in seen:
            continue
        seen.add(key)
        samples.append(sample)
    return sorted(samples, key=_lerobot_media_sample_sort_key)


def _lerobot_media_inspection_sample_from_checkpoint(
    row: dict[str, Any],
    *,
    storage_tier_override: str | None,
    provider_override: str | None,
) -> dict[str, Any] | None:
    progress = _lerobot_progress_from_checkpoint_row(row)
    media = progress.get("media_inspection")
    if not isinstance(media, dict) or not media:
        return None
    sample_source_uri = str(row.get("source_uri") or row.get("source_ref") or "")
    return _lerobot_media_inspection_sample(
        source_kind="checkpoint",
        media_inspection=media,
        source_uri=sample_source_uri,
        storage_tier_override=storage_tier_override,
        provider_override=provider_override,
        checkpoint_id=str(row.get("checkpoint_id") or ""),
        checkpoint_index=_optional_lerobot_int(row.get("checkpoint_index")),
        job_id=str(row.get("job_id") or ""),
        source_id=str(row.get("source_id") or ""),
        run_id=str(row.get("run_id") or ""),
        transform_id=str(row.get("transform_id") or ""),
        status=str(row.get("status") or ""),
        phase=str(row.get("phase") or ""),
        updated_at=_coerce_lerobot_utc(row.get("updated_at") or row.get("created_at")),
    )


def _lerobot_media_inspection_sample_from_transform(
    row: dict[str, Any],
    *,
    storage_tier_override: str | None,
    provider_override: str | None,
) -> dict[str, Any] | None:
    if str(row.get("status") or "") != "completed":
        return None
    params = _lerobot_transform_params(row)
    if params.get("adapter") != "lerobot":
        return None
    media = params.get("media_inspection")
    if not isinstance(media, dict) or not media:
        return None
    input_uris = row.get("input_uris") or []
    sample_source_uri = str(input_uris[0] if input_uris else "")
    return _lerobot_media_inspection_sample(
        source_kind="transform",
        media_inspection=media,
        source_uri=sample_source_uri,
        storage_tier_override=storage_tier_override,
        provider_override=provider_override,
        checkpoint_id=None,
        checkpoint_index=None,
        job_id=None,
        source_id=str(row.get("source_id") or ""),
        run_id=str(params.get("run_id") or ""),
        transform_id=str(row.get("transform_id") or ""),
        status=str(row.get("status") or ""),
        phase=str(row.get("kind") or ""),
        updated_at=_coerce_lerobot_utc(row.get("finished_at") or row.get("created_at")),
    )


def _lerobot_transform_params(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("params")
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _lerobot_media_inspection_sample(
    *,
    source_kind: str,
    media_inspection: dict[str, Any],
    source_uri: str,
    storage_tier_override: str | None,
    provider_override: str | None,
    checkpoint_id: str | None,
    checkpoint_index: int | None,
    job_id: str | None,
    source_id: str,
    run_id: str,
    transform_id: str,
    status: str,
    phase: str,
    updated_at: datetime | None,
) -> dict[str, Any]:
    provider = provider_override or _lerobot_source_provider(source_uri)
    storage_tier = storage_tier_override or _lerobot_storage_tier(source_uri, provider=provider)
    video_count = _lerobot_media_report_video_count(media_inspection)
    return {
        "source_kind": source_kind,
        "checkpoint_id": checkpoint_id,
        "checkpoint_index": checkpoint_index,
        "job_id": job_id,
        "source_id": source_id,
        "run_id": run_id,
        "transform_id": transform_id,
        "source_uri": source_uri,
        "storage_tier": storage_tier,
        "provider": provider,
        "corpus_size_tier": _lerobot_corpus_size_tier(video_count),
        "video_count": video_count,
        "status": status,
        "phase": phase,
        "updated_at": updated_at,
        "media_inspection": media_inspection,
    }


def _lerobot_media_sample_matches(
    sample: dict[str, Any],
    *,
    job_id: str | None,
    source_id: str | None,
    source_uri: str | None,
    storage_tier: str | None,
    provider: str | None,
) -> bool:
    if job_id is not None and sample.get("job_id") != job_id:
        return False
    if source_id is not None and sample.get("source_id") != source_id:
        return False
    if source_uri is not None and sample.get("source_uri") != source_uri:
        return False
    if storage_tier is not None and sample.get("storage_tier") != storage_tier:
        return False
    if provider is not None and sample.get("provider") != provider:
        return False
    return True


def _lerobot_media_sample_sort_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    updated_at = sample.get("updated_at") or datetime.min.replace(tzinfo=UTC)
    return (
        updated_at,
        str(sample.get("source_kind") or ""),
        str(sample.get("checkpoint_id") or ""),
        str(sample.get("transform_id") or ""),
    )


def _lerobot_media_sample_dedupe_key(sample: dict[str, Any]) -> tuple[str, str, str]:
    media = sample.get("media_inspection") or {}
    payload = json.dumps(media, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return (
        str(sample.get("source_id") or sample.get("source_uri") or ""),
        str(sample.get("run_id") or sample.get("job_id") or ""),
        digest,
    )


def _lerobot_source_provider(uri: str | None) -> str:
    value = str(uri or "").strip().lower()
    if value.startswith("hf://"):
        return "huggingface"
    if "://" in value:
        scheme = value.split("://", maxsplit=1)[0]
        return {
            "s3": "s3",
            "gs": "gcs",
            "gcs": "gcs",
            "az": "azure",
            "abfs": "azure",
            "abfss": "azure",
            "file": "local",
        }.get(scheme, scheme)
    return "local"


def _lerobot_storage_tier(uri: str | None, *, provider: str) -> str:
    value = str(uri or "")
    if provider == "huggingface":
        return "huggingface"
    if is_object_store_uri(value):
        return "object-store"
    return "local"


def _lerobot_corpus_size_tier(video_count: int) -> str:
    if video_count <= 0:
        return "unknown"
    if video_count <= 2:
        return "local"
    if video_count <= 25:
        return "ci"
    if video_count <= 10_000:
        return "mid-corpus"
    return "full-corpus"


def _lerobot_media_report_video_count(media_inspection: dict[str, Any]) -> int:
    video_count = _optional_lerobot_int(media_inspection.get("video_count"))
    if video_count is not None:
        return max(0, video_count)
    videos = media_inspection.get("videos")
    return len(videos) if isinstance(videos, list) else 0


def _lerobot_media_inspection_timeout_report(
    lake_uri: str | None,
    samples: list[dict[str, Any]],
    *,
    filters: dict[str, Any],
    min_timeout_seconds: float,
    max_timeout_seconds: float,
) -> dict[str, Any]:
    telemetry = _lerobot_media_timeout_metrics(samples)
    recommendation = _lerobot_media_timeout_recommendation(
        telemetry,
        min_timeout_seconds=min_timeout_seconds,
        max_timeout_seconds=max_timeout_seconds,
    )
    groups = []
    for selector, group_samples in _lerobot_media_timeout_groups(samples).items():
        metrics = _lerobot_media_timeout_metrics(group_samples)
        groups.append(
            {
                "selector": {
                    "storage_tier": selector[0],
                    "provider": selector[1],
                    "corpus_size_tier": selector[2],
                },
                "sample_count": len(group_samples),
                "telemetry": metrics,
                "recommendation": _lerobot_media_timeout_recommendation(
                    metrics,
                    min_timeout_seconds=min_timeout_seconds,
                    max_timeout_seconds=max_timeout_seconds,
                ),
            }
        )
    return {
        "lake_uri": lake_uri,
        "adapter": "lerobot",
        "scope": "media-inspection-timeout-recommendations",
        "filters": {key: value for key, value in filters.items() if value is not None},
        "source_counts": {
            "reports": len(samples),
            "checkpoints": sum(
                1 for sample in samples if sample.get("source_kind") == "checkpoint"
            ),
            "completed_transforms": sum(
                1 for sample in samples if sample.get("source_kind") == "transform"
            ),
        },
        "telemetry": telemetry,
        "recommendation": recommendation,
        "groups": sorted(
            groups,
            key=lambda item: (
                str(item["selector"].get("storage_tier") or ""),
                str(item["selector"].get("provider") or ""),
                str(item["selector"].get("corpus_size_tier") or ""),
            ),
        ),
        "evidence": [_lerobot_media_timeout_evidence(sample) for sample in samples],
    }


def _lerobot_media_timeout_groups(
    samples: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        key = (
            str(sample.get("storage_tier") or "unknown"),
            str(sample.get("provider") or "unknown"),
            str(sample.get("corpus_size_tier") or "unknown"),
        )
        groups.setdefault(key, []).append(sample)
    return groups


def _lerobot_media_timeout_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    durations: list[float] = []
    duration_keys: set[tuple[str, ...]] = set()
    observed_timeouts: set[float] = set()
    observed_retries: set[int] = set()
    observed_backoffs: set[float] = set()
    status_counts: dict[str, int] = {}
    diagnostic_counts: dict[str, int] = {}
    execution_modes: dict[str, int] = {}
    totals = {
        "reported_video_count": 0,
        "video_sample_count": 0,
        "total_attempts": 0,
        "total_retries": 0,
        "total_timeouts": 0,
        "killed_worker_count": 0,
        "completed_video_count": 0,
        "failed_video_count": 0,
        "timeout_video_count": 0,
        "reused_video_count": 0,
        "duration_reused_excluded_count": 0,
    }
    for sample in samples:
        media = sample.get("media_inspection") or {}
        if not isinstance(media, dict):
            continue
        totals["reported_video_count"] += _lerobot_media_report_video_count(media)
        reported_attempts = _optional_lerobot_int(media.get("total_attempts"))
        reported_retries = _optional_lerobot_int(media.get("total_retries"))
        reported_timeouts = _optional_lerobot_int(media.get("total_timeouts"))
        totals["killed_worker_count"] += (
            _optional_lerobot_int(media.get("killed_worker_count")) or 0
        )
        timeout_seconds = _optional_lerobot_float(media.get("timeout_seconds"))
        if timeout_seconds is not None and timeout_seconds > 0:
            observed_timeouts.add(timeout_seconds)
        retry_count = _optional_lerobot_int(media.get("retry_count"))
        if retry_count is not None:
            observed_retries.add(max(0, retry_count))
        backoff = _optional_lerobot_float(media.get("retry_backoff_seconds"))
        if backoff is not None and backoff >= 0:
            observed_backoffs.add(backoff)
        execution_mode = str(media.get("execution_mode") or "").strip()
        if execution_mode:
            _increment_lerobot_count(execution_modes, execution_mode)
        has_report_status_counts = isinstance(media.get("status_counts"), dict)
        if has_report_status_counts:
            _merge_lerobot_counts(status_counts, media.get("status_counts"))
        _merge_lerobot_counts(diagnostic_counts, media.get("diagnostic_counts"))
        video_attempts = 0
        video_retries = 0
        video_timeouts = 0
        video_status_counts: dict[str, int] = {}
        for video in media.get("videos") or []:
            if not isinstance(video, dict):
                continue
            totals["video_sample_count"] += 1
            duration = _optional_lerobot_float(video.get("inspection_duration_ms"))
            if video.get("inspection_reused"):
                totals["reused_video_count"] += 1
                totals["duration_reused_excluded_count"] += 1
            elif duration is not None and duration >= 0:
                duration_key = _lerobot_media_video_key(sample, video)
                if duration_key not in duration_keys:
                    duration_keys.add(duration_key)
                    durations.append(duration)
            status = str(video.get("inspection_status") or "").strip().lower()
            if status:
                _increment_lerobot_count(video_status_counts, status)
            if status == "completed":
                totals["completed_video_count"] += 1
            elif status == "timeout":
                totals["timeout_video_count"] += 1
            elif status in {"failed", "error"}:
                totals["failed_video_count"] += 1
            video_retries += _optional_lerobot_int(video.get("inspection_retries")) or 0
            video_timeouts += _optional_lerobot_int(video.get("inspection_timeouts")) or 0
            video_attempts += _optional_lerobot_int(video.get("inspection_attempts")) or 0
            for diagnostic in video.get("diagnostics") or []:
                if isinstance(diagnostic, str):
                    _increment_lerobot_count(diagnostic_counts, diagnostic)
        if not has_report_status_counts:
            _merge_lerobot_counts(status_counts, video_status_counts)
        totals["total_attempts"] += (
            reported_attempts if reported_attempts is not None else video_attempts
        )
        totals["total_retries"] += (
            reported_retries if reported_retries is not None else video_retries
        )
        totals["total_timeouts"] += (
            reported_timeouts if reported_timeouts is not None else video_timeouts
        )
    duration_stats = _lerobot_media_duration_stats(durations)
    return {
        "reports_count": len(samples),
        **totals,
        "status_counts": dict(sorted(status_counts.items())),
        "diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "execution_modes": dict(sorted(execution_modes.items())),
        "duration_ms": duration_stats,
        "observed_timeout_seconds": sorted(observed_timeouts),
        "observed_retry_count": sorted(observed_retries),
        "observed_retry_backoff_seconds": sorted(observed_backoffs),
    }


def _lerobot_media_video_key(sample: dict[str, Any], video: dict[str, Any]) -> tuple[str, ...]:
    fingerprint = str(video.get("inspection_fingerprint") or "")
    if fingerprint:
        return ("fingerprint", fingerprint)
    return (
        "video",
        str(sample.get("source_id") or sample.get("source_uri") or ""),
        str(sample.get("run_id") or sample.get("job_id") or ""),
        str(video.get("path") or video.get("uri") or ""),
        str(video.get("episode_index") or ""),
        str(video.get("camera_key") or ""),
    )


def _lerobot_media_duration_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(_lerobot_percentile(ordered, 0.50), 3),
        "p95": round(_lerobot_percentile(ordered, 0.95), 3),
        "p99": round(_lerobot_percentile(ordered, 0.99), 3),
        "max": round(ordered[-1], 3),
    }


def _lerobot_percentile(ordered_values: list[float], percentile: float) -> float:
    index = min(
        len(ordered_values) - 1,
        max(0, math.ceil(percentile * len(ordered_values)) - 1),
    )
    return ordered_values[index]


def _lerobot_media_timeout_recommendation(
    telemetry: dict[str, Any],
    *,
    min_timeout_seconds: float,
    max_timeout_seconds: float,
) -> dict[str, Any]:
    if int(telemetry.get("reports_count") or 0) == 0:
        return {
            "status": "insufficient-data",
            "reason": "no LeRobot media-inspection telemetry matched the requested filters",
            "timeout_seconds": None,
            "retry_count": None,
            "retry_backoff_seconds": None,
            "flags": [],
            "apply_args": [],
        }
    durations = telemetry.get("duration_ms") or {}
    observed_timeout_seconds = [
        float(value) for value in telemetry.get("observed_timeout_seconds") or []
    ]
    current_timeout = max(observed_timeout_seconds) if observed_timeout_seconds else None
    current_retry = max([0, *[int(value) for value in telemetry.get("observed_retry_count") or []]])
    current_backoff = max(
        [0.0, *[float(value) for value in telemetry.get("observed_retry_backoff_seconds") or []]]
    )
    p99_seconds = None
    if durations.get("p99") is not None:
        p99_seconds = float(durations["p99"]) / 1000.0
    total_timeouts = int(telemetry.get("total_timeouts") or 0)
    timeout_video_count = int(telemetry.get("timeout_video_count") or 0)
    timeout_signal = total_timeouts > 0 or timeout_video_count > 0
    total_failures = int(telemetry.get("failed_video_count") or 0)
    killed_workers = int(telemetry.get("killed_worker_count") or 0)
    flags: list[str] = []
    reasons: list[str] = []
    candidate_timeout: float | None = None
    if p99_seconds is not None:
        headroom = 2.0 if timeout_signal or killed_workers else 1.5
        candidate_timeout = p99_seconds * headroom
        reasons.append(f"p99 media inspection duration {round(p99_seconds, 3)}s")
    if current_timeout is not None and (timeout_signal or killed_workers):
        candidate_timeout = max(candidate_timeout or 0.0, current_timeout * 2.0)
        flags.append("timeout-policy-too-aggressive")
        reasons.append(
            f"observed {total_timeouts} timeout attempt(s), "
            f"{timeout_video_count} timed-out video(s), and {killed_workers} killed worker(s)"
        )
    if candidate_timeout is None and current_timeout is not None:
        candidate_timeout = current_timeout
        reasons.append("no per-video duration samples; preserving observed timeout")
    recommended_timeout = None
    if candidate_timeout is not None:
        clamped = min(max_timeout_seconds, max(min_timeout_seconds, candidate_timeout))
        if clamped != candidate_timeout:
            flags.append("timeout-recommendation-clamped")
        recommended_timeout = _ceil_lerobot_timeout(clamped)
    if (
        p99_seconds is not None
        and current_timeout is not None
        and not timeout_signal
        and killed_workers == 0
        and recommended_timeout is not None
        and current_timeout > max(recommended_timeout * 2.0, p99_seconds * 4.0)
    ):
        flags.append("timeout-policy-too-loose")
    if killed_workers:
        flags.append("process-media-inspection-kills-observed")
    if total_failures:
        flags.append("media-inspection-failures-observed")
    if timeout_signal or total_failures or killed_workers:
        recommended_retry = min(3, max(1, current_retry + 1))
    else:
        recommended_retry = current_retry
    recommended_backoff = current_backoff
    apply_args = []
    if recommended_timeout is not None:
        apply_args.extend(
            ["--media-inspection-timeout-seconds", _format_lerobot_number(recommended_timeout)]
        )
    apply_args.extend(["--media-inspection-retries", str(recommended_retry)])
    if recommended_backoff > 0:
        apply_args.extend(
            [
                "--media-inspection-retry-backoff-seconds",
                _format_lerobot_number(recommended_backoff),
            ]
        )
    status = "adjust" if flags else "keep"
    return {
        "status": status,
        "timeout_seconds": recommended_timeout,
        "retry_count": recommended_retry,
        "retry_backoff_seconds": recommended_backoff,
        "basis": {
            "p99_seconds": round(p99_seconds, 3) if p99_seconds is not None else None,
            "observed_timeout_seconds": observed_timeout_seconds,
            "observed_retry_count": telemetry.get("observed_retry_count") or [],
            "total_timeouts": total_timeouts,
            "timeout_video_count": timeout_video_count,
            "total_failures": total_failures,
            "killed_worker_count": killed_workers,
        },
        "flags": sorted(set(flags)),
        "reason": "; ".join(reasons) if reasons else "existing settings match observed telemetry",
        "apply_args": apply_args,
    }


def _ceil_lerobot_timeout(value: float) -> float:
    return math.ceil(value * 10.0) / 10.0


def _format_lerobot_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _lerobot_media_timeout_evidence(sample: dict[str, Any]) -> dict[str, Any]:
    media = sample.get("media_inspection") or {}
    return {
        "source_kind": sample.get("source_kind"),
        "checkpoint_id": sample.get("checkpoint_id"),
        "checkpoint_index": sample.get("checkpoint_index"),
        "job_id": sample.get("job_id"),
        "source_id": sample.get("source_id"),
        "run_id": sample.get("run_id"),
        "transform_id": sample.get("transform_id"),
        "status": sample.get("status"),
        "phase": sample.get("phase"),
        "source_uri": sample.get("source_uri"),
        "storage_tier": sample.get("storage_tier"),
        "provider": sample.get("provider"),
        "corpus_size_tier": sample.get("corpus_size_tier"),
        "video_count": sample.get("video_count"),
        "timeout_seconds": _optional_lerobot_float(media.get("timeout_seconds")),
        "retry_count": _optional_lerobot_int(media.get("retry_count")),
        "total_timeouts": _optional_lerobot_int(media.get("total_timeouts")) or 0,
        "total_retries": _optional_lerobot_int(media.get("total_retries")) or 0,
        "killed_worker_count": _optional_lerobot_int(media.get("killed_worker_count")) or 0,
        "updated_at": sample.get("updated_at"),
    }


def _merge_lerobot_counts(target: dict[str, int], value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key, count in value.items():
        _increment_lerobot_count(target, str(key), _optional_lerobot_int(count) or 0)


def _increment_lerobot_count(target: dict[str, int], key: str, amount: int = 1) -> None:
    if not key:
        return
    target[key] = int(target.get(key) or 0) + int(amount)


def _normalize_lerobot_retention_statuses(
    statuses: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if statuses is None:
        return tuple(sorted(_LEROBOT_TERMINAL_STATUSES))
    normalized = tuple(
        dict.fromkeys(str(status).strip().lower() for status in statuses if str(status).strip())
    )
    if not normalized:
        return tuple(sorted(_LEROBOT_TERMINAL_STATUSES))
    unknown = sorted(set(normalized) - set(_LEROBOT_TERMINAL_STATUSES))
    if unknown:
        allowed = ", ".join(sorted(_LEROBOT_TERMINAL_STATUSES))
        raise ValueError(
            "LeRobot checkpoint retention can only compact terminal statuses; "
            f"got {unknown}, expected one of: {allowed}"
        )
    return normalized


def _minimum_lerobot_checkpoint_history_jobs(
    rows_by_job: dict[str, list[dict[str, Any]]],
    *,
    statuses: tuple[str, ...],
    source_id: str | None,
    retain_completed_per_source: int,
    retain_failed_per_source: int,
) -> set[str]:
    by_source_status: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for job_id, job_rows in rows_by_job.items():
        if not job_rows:
            continue
        latest = max(job_rows, key=_lerobot_checkpoint_sort_key)
        status = str(latest.get("status") or "")
        if status not in statuses:
            continue
        latest_source_id = str(latest.get("source_id") or "")
        if source_id is not None and latest_source_id != source_id:
            continue
        by_source_status.setdefault((latest_source_id, status), []).append((job_id, latest))

    keep: set[str] = set()
    for (_source_id, status), items in by_source_status.items():
        floor = retain_failed_per_source if status == "failed" else retain_completed_per_source
        if floor <= 0:
            continue
        ordered = sorted(
            items, key=lambda item: _lerobot_checkpoint_sort_key(item[1]), reverse=True
        )
        keep.update(job_id for job_id, _latest in ordered[:floor])
    return keep


def _terminal_lerobot_checkpoint_row(
    rows: list[dict[str, Any]],
    statuses: tuple[str, ...],
) -> dict[str, Any] | None:
    terminal_rows = [
        row
        for row in rows
        if str(row.get("status") or "") in statuses
        and str(row.get("status") or "") in _LEROBOT_TERMINAL_STATUSES
    ]
    if not terminal_rows:
        return None
    return max(terminal_rows, key=_lerobot_checkpoint_sort_key)


def _lerobot_checkpoint_time(row: dict[str, Any]) -> datetime:
    value = row.get("finished_at") or row.get("updated_at") or row.get("created_at")
    return _coerce_lerobot_utc(value) or datetime.min.replace(tzinfo=UTC)


def _coerce_lerobot_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


def _normalize_lerobot_duration(
    value: timedelta | int | float | None,
    *,
    name: str,
    default: timedelta,
) -> timedelta:
    if value is None:
        return default
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
    else:
        seconds = float(value)
    if seconds <= 0:
        raise ValueError(f"{name} must be greater than 0 seconds")
    return timedelta(seconds=seconds)


def _normalize_lerobot_keyframe_map_thresholds(
    threshold_bytes: int | None,
    threshold_frames: int | None,
) -> tuple[int | None, int | None]:
    if threshold_bytes is not None and int(threshold_bytes) < 0:
        raise ValueError("keyframe_map_inline_threshold_bytes must be >= 0 or None")
    if threshold_frames is not None and int(threshold_frames) < 0:
        raise ValueError("keyframe_map_inline_threshold_frames must be >= 0 or None")
    return (
        None if threshold_bytes is None else int(threshold_bytes),
        None if threshold_frames is None else int(threshold_frames),
    )


def _lerobot_claim_token(
    job_id: str,
    owner: str,
    timestamp: datetime,
    *,
    suffix: str | None = None,
) -> str:
    stamp = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    parts = [job_id, owner, stamp]
    if suffix:
        parts.append(suffix)
    return ":".join(parts)


def _lerobot_new_claim_payload(
    *,
    owner: str,
    token: str,
    generation: int,
    lease: timedelta,
    heartbeat: timedelta,
) -> dict[str, Any]:
    return {
        "owner": owner,
        "token": token,
        "generation": int(generation),
        "active": True,
        "lease_seconds": lease.total_seconds(),
        "heartbeat_interval_seconds": heartbeat.total_seconds(),
        "heartbeat_count": 0,
        "claim_expires_at": None,
        "last_heartbeat_at": None,
    }


def _refresh_lerobot_claim_lease(
    progress: dict[str, Any],
    *,
    status: str,
    claim_owner: str,
    claim_token: str,
    now: datetime,
) -> None:
    raw_claim = progress.get("claim")
    claim = dict(raw_claim) if isinstance(raw_claim, dict) else {}
    claim.setdefault("owner", claim_owner)
    claim.setdefault("token", claim_token)
    claim["active"] = status == "running"
    claim["last_heartbeat_at"] = now.isoformat()
    if status == "running":
        lease_seconds = float(
            claim.get("lease_seconds") or DEFAULT_LEROBOT_CLAIM_LEASE.total_seconds()
        )
        claim["lease_seconds"] = lease_seconds
        claim["claim_expires_at"] = (now + timedelta(seconds=lease_seconds)).isoformat()
        claim["heartbeat_count"] = int(claim.get("heartbeat_count") or 0) + 1
    else:
        claim["claim_expires_at"] = None
    progress["claim"] = claim


def _lerobot_progress_from_checkpoint_row(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("progress_json")
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _lerobot_claim_from_progress(progress: dict[str, Any]) -> dict[str, Any]:
    claim = progress.get("claim")
    return dict(claim) if isinstance(claim, dict) else {}


def _lerobot_claim_expires_at(
    row: dict[str, Any],
    *,
    stale_after: timedelta,
) -> datetime | None:
    expires_at, _source = _lerobot_claim_expiration_detail(row, stale_after=stale_after)
    return expires_at


def _lerobot_claim_expiration_detail(
    row: dict[str, Any],
    *,
    stale_after: timedelta,
) -> tuple[datetime | None, str]:
    progress = _lerobot_progress_from_checkpoint_row(row)
    claim = _lerobot_claim_from_progress(progress)
    expires_at = _coerce_lerobot_utc(claim.get("claim_expires_at") or claim.get("expires_at"))
    if expires_at is not None:
        return expires_at, "claim_expires_at"
    claim_stale_after = stale_after
    source = "missing-lease"
    try:
        if claim.get("lease_seconds") is not None:
            claim_stale_after = timedelta(seconds=float(claim["lease_seconds"]))
            source = "lease_seconds"
    except (TypeError, ValueError):
        claim_stale_after = stale_after
    updated_at = _coerce_lerobot_utc(row.get("updated_at") or row.get("created_at"))
    if updated_at is None:
        return None, "missing-timestamp"
    return updated_at + claim_stale_after, source


def _lerobot_claim_watchdog_finding(
    *,
    lake_uri: str,
    row: dict[str, Any],
    stale_after: timedelta,
    recovery_action: str,
    new_owner: str | None,
    created_by: str,
    now: datetime,
) -> LeRobotClaimWatchdogFinding:
    progress = _lerobot_progress_from_checkpoint_row(row)
    claim = _lerobot_claim_from_progress(progress)
    status = str(row.get("status") or "")
    expires_at, expiration_source = _lerobot_claim_expiration_detail(row, stale_after=stale_after)
    updated_at = _coerce_lerobot_utc(row.get("updated_at") or row.get("created_at"))
    last_heartbeat_at = _coerce_lerobot_utc(claim.get("last_heartbeat_at")) or updated_at
    stale = False
    stale_reason = f"latest-status-{status or 'unknown'}"
    stale_seconds = 0.0
    seconds_until_stale = 0.0
    suggested_recovery_command = None
    lease_state = "inactive"
    if status == "running":
        if expires_at is None:
            stale = True
            stale_reason = "missing-expiration"
            lease_state = "missing-lease"
        elif expires_at <= now:
            stale = True
            stale_reason = {
                "claim_expires_at": "expired-claim",
                "lease_seconds": "lease-timeout",
                "missing-lease": "missing-lease",
            }.get(expiration_source, "expired-claim")
            lease_state = "missing-lease" if expiration_source == "missing-lease" else "stale"
            stale_seconds = max(0.0, (now - expires_at).total_seconds())
        else:
            stale_reason = "missing-lease" if expiration_source == "missing-lease" else "live"
            lease_state = "missing-lease" if expiration_source == "missing-lease" else "live"
            seconds_until_stale = max(0.0, (expires_at - now).total_seconds())
        if stale:
            suggested_recovery_command = _lerobot_claim_recovery_command(
                lake_uri=lake_uri,
                job_id=str(row.get("job_id") or ""),
                recovery_action=recovery_action,
                new_owner=new_owner,
                created_by=created_by,
                stale_after=stale_after,
                expected_latest_checkpoint_id=str(row.get("checkpoint_id") or ""),
                expected_latest_claim_token=str(row.get("claim_token") or claim.get("token") or "")
                or None,
                expected_checkpoint_index=int(row.get("checkpoint_index") or 0),
            )

    return LeRobotClaimWatchdogFinding(
        job_id=str(row.get("job_id") or ""),
        source_id=str(row.get("source_id") or ""),
        status=status,
        phase=str(row.get("phase") or ""),
        stale=stale,
        stale_reason=stale_reason,
        lease_state=lease_state,
        checkpoint_id=str(row.get("checkpoint_id") or ""),
        checkpoint_index=int(row.get("checkpoint_index") or 0),
        claim_owner=str(row.get("claim_owner") or claim.get("owner") or "") or None,
        claim_token=str(row.get("claim_token") or claim.get("token") or "") or None,
        claim_generation=_optional_lerobot_int(claim.get("generation")),
        last_heartbeat_at=last_heartbeat_at,
        updated_at=updated_at,
        claim_expires_at=expires_at,
        expiration_source=expiration_source,
        stale_seconds=stale_seconds,
        seconds_until_stale=seconds_until_stale,
        rows_seen=int(row.get("rows_seen") or 0),
        observations_written=int(row.get("observations_written") or 0),
        episodes_written=int(row.get("episodes_written") or 0),
        scenarios_written=int(row.get("scenarios_written") or 0),
        videos_written=int(row.get("videos_written") or 0),
        video_encodings_written=int(row.get("video_encodings_written") or 0),
        rows_skipped_existing=int(row.get("rows_skipped_existing") or 0),
        bytes_scanned=int(row.get("bytes_scanned") or 0),
        suggested_recovery_command=suggested_recovery_command,
    )


def _optional_lerobot_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_lerobot_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _lerobot_claim_recovery_command(
    *,
    lake_uri: str,
    job_id: str,
    recovery_action: str,
    new_owner: str | None,
    created_by: str,
    stale_after: timedelta,
    expected_latest_checkpoint_id: str | None = None,
    expected_latest_claim_token: str | None = None,
    expected_checkpoint_index: int | None = None,
) -> str:
    parts = [
        "lancedb-robotics",
        "ingest",
        "lerobot-claim-recover",
        job_id,
        "--lake",
        lake_uri,
        "--action",
        recovery_action,
        "--stale-after-seconds",
        _format_lerobot_seconds(stale_after.total_seconds()),
        "--format",
        "json",
    ]
    if new_owner:
        parts.extend(["--new-owner", new_owner])
    if created_by != "lancedb-robotics":
        parts.extend(["--created-by", created_by])
    if expected_latest_checkpoint_id:
        parts.extend(["--expected-latest-checkpoint-id", expected_latest_checkpoint_id])
    if expected_latest_claim_token:
        parts.extend(["--expected-latest-claim-token", expected_latest_claim_token])
    if expected_checkpoint_index is not None:
        parts.extend(["--expected-checkpoint-index", str(int(expected_checkpoint_index))])
    return shlex.join(parts)


def _format_lerobot_seconds(seconds: float) -> str:
    return str(int(seconds)) if float(seconds).is_integer() else str(seconds)


def _next_lerobot_claim_generation(lake: Lake, job_id: str) -> int:
    generation = 0
    for row in _lerobot_checkpoint_rows(lake):
        if row.get("job_id") != job_id:
            continue
        progress = _lerobot_progress_from_checkpoint_row(row)
        claim = _lerobot_claim_from_progress(progress)
        try:
            generation = max(generation, int(claim.get("generation") or 0))
        except (TypeError, ValueError):
            continue
    return generation + 1


def _lerobot_checkpoint_hold_rows(lake: Lake) -> list[dict[str, Any]]:
    try:
        return lake.table(_LEROBOT_CHECKPOINT_HOLDS_TABLE).to_arrow().to_pylist()
    except Exception:  # noqa: BLE001 - older lakes may not have the optional catalog yet
        return []


def _normalize_lerobot_checkpoint_hold_selector(
    *,
    checkpoint_id: str | None,
    job_id: str | None,
    source_id: str | None,
    hf_repo_id: str | None,
    requested_revision: str | None,
    resolved_revision: str | None,
    status: str | tuple[str, ...] | list[str] | None,
    updated_after: datetime | str | None,
    updated_before: datetime | str | None,
) -> dict[str, Any]:
    selector: dict[str, Any] = {}
    for key, value in (
        ("checkpoint_id", checkpoint_id),
        ("job_id", job_id),
        ("source_id", source_id),
        ("hf_repo_id", hf_repo_id),
        ("requested_revision", requested_revision),
        ("resolved_revision", resolved_revision),
    ):
        if value is not None and str(value).strip():
            selector[key] = str(value).strip()

    statuses = _normalize_lerobot_hold_statuses(status)
    if statuses:
        selector["statuses"] = list(statuses)

    after_dt = _coerce_lerobot_utc(updated_after)
    before_dt = _coerce_lerobot_utc(updated_before)
    if updated_after is not None and after_dt is None:
        raise ValueError(f"invalid updated_after timestamp: {updated_after!r}")
    if updated_before is not None and before_dt is None:
        raise ValueError(f"invalid updated_before timestamp: {updated_before!r}")
    if after_dt is not None and before_dt is not None and after_dt > before_dt:
        raise ValueError("updated_after must be <= updated_before")
    if after_dt is not None:
        selector["updated_after"] = after_dt.isoformat()
    if before_dt is not None:
        selector["updated_before"] = before_dt.isoformat()

    if not selector:
        raise ValueError("a LeRobot checkpoint hold requires at least one selector")
    return selector


def _normalize_lerobot_hold_statuses(
    status: str | tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if status is None:
        return ()
    raw = (status,) if isinstance(status, str) else tuple(status)
    return tuple(dict.fromkeys(str(item).strip().lower() for item in raw if str(item).strip()))


def _select_lerobot_checkpoint_rows(
    lake: Lake,
    selector: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        row
        for row in _lerobot_checkpoint_rows(lake)
        if _lerobot_checkpoint_row_matches_selector(row, selector)
    ]


def _lerobot_checkpoint_row_matches_selector(
    row: dict[str, Any],
    selector: dict[str, Any],
) -> bool:
    for key in (
        "checkpoint_id",
        "job_id",
        "source_id",
        "hf_repo_id",
        "requested_revision",
        "resolved_revision",
    ):
        expected = selector.get(key)
        if expected is not None and str(row.get(key) or "") != str(expected):
            return False
    statuses = tuple(selector.get("statuses") or ())
    if statuses and str(row.get("status") or "").lower() not in statuses:
        return False
    checkpoint_time = _lerobot_checkpoint_time(row)
    updated_after = _coerce_lerobot_utc(selector.get("updated_after"))
    updated_before = _coerce_lerobot_utc(selector.get("updated_before"))
    if updated_after is not None and checkpoint_time < updated_after:
        return False
    if updated_before is not None and checkpoint_time > updated_before:
        return False
    return True


def _active_lerobot_checkpoint_hold_details(
    lake: Lake,
    *,
    now: datetime,
) -> dict[str, tuple[dict[str, str], ...]]:
    held: dict[str, list[dict[str, str]]] = {}

    def add(checkpoint_id: str, detail: dict[str, str]) -> None:
        if checkpoint_id:
            held.setdefault(str(checkpoint_id), []).append(detail)

    try:
        artifact_rows = lake.table("lineage_artifacts").to_arrow().to_pylist()
    except Exception:  # noqa: BLE001 - governance metadata should not block retention planning
        artifact_rows = []
    for row in artifact_rows:
        if str(row.get("table_name") or "") != _LEROBOT_CHECKPOINT_TABLE:
            continue
        metadata = _lerobot_metadata_map(row.get("metadata"))
        if not _lerobot_hold_active(metadata, now=now):
            continue
        detail = {
            "hold_id": str(row.get("artifact_id") or ""),
            "source": "lineage_artifact",
            "reason": metadata.get("reason") or "lineage retention hold",
            "owner": metadata.get("owner") or "",
        }
        for row_id in row.get("row_ids") or ():
            add(str(row_id), detail)
        for key in ("checkpoint_id", "lerobot_checkpoint_id"):
            add(metadata.get(key) or "", detail)

    checkpoint_rows = _lerobot_checkpoint_rows(lake)
    for row in _lerobot_checkpoint_hold_rows(lake):
        if not _lerobot_catalog_hold_active(row, now=now):
            continue
        selector = _lerobot_checkpoint_hold_selector(row)
        detail = {
            "hold_id": str(row.get("hold_id") or ""),
            "source": "lerobot_checkpoint_holds",
            "reason": str(row.get("reason") or "LeRobot checkpoint hold"),
            "owner": str(row.get("owner") or ""),
        }
        for checkpoint in checkpoint_rows:
            if _lerobot_checkpoint_row_matches_selector(checkpoint, selector):
                add(str(checkpoint.get("checkpoint_id") or ""), detail)

    return {checkpoint_id: tuple(details) for checkpoint_id, details in held.items()}


def _active_lerobot_checkpoint_hold_ids(lake: Lake, *, now: datetime) -> set[str]:
    return set(_active_lerobot_checkpoint_hold_details(lake, now=now))


def _lerobot_checkpoint_hold_selector(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("selector_json")
    if raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        if isinstance(parsed, dict):
            return parsed
    selector: dict[str, Any] = {}
    for key in (
        "job_id",
        "source_id",
        "hf_repo_id",
        "requested_revision",
        "resolved_revision",
    ):
        if row.get(key):
            selector[key] = str(row[key])
    if row.get("statuses"):
        selector["statuses"] = [str(status) for status in row["statuses"] if status]
    for key in ("updated_after", "updated_before"):
        value = _coerce_lerobot_utc(row.get(key))
        if value is not None:
            selector[key] = value.isoformat()
    return selector


def _lerobot_catalog_hold_active(row: dict[str, Any], *, now: datetime) -> bool:
    if not row.get("active"):
        return False
    if _coerce_lerobot_utc(row.get("released_at")) is not None:
        return False
    if (
        bool(row.get("legal_hold"))
        or bool(row.get("audit_hold"))
        or bool(row.get("promotion_hold"))
    ):
        return True
    retain_until = _coerce_lerobot_utc(row.get("retain_until"))
    return retain_until is not None and retain_until > now


def _lerobot_metadata_map(items: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or ():
        if isinstance(item, dict) and item.get("key") is not None:
            result[str(item["key"])] = "" if item.get("value") is None else str(item.get("value"))
    return result


def _lerobot_hold_active(metadata: dict[str, str], *, now: datetime) -> bool:
    if not metadata:
        return False
    if any(
        _lerobot_bool_metadata(metadata.get(key))
        for key in ("legal_hold", "audit_hold", "promotion_hold")
    ):
        return True
    retain_until = _coerce_lerobot_utc(metadata.get("retain_until"))
    return retain_until is not None and retain_until > now


def _lerobot_bool_metadata(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _delete_lerobot_checkpoint_ids(lake: Lake, checkpoint_ids: list[str]) -> None:
    unique = [checkpoint_id for checkpoint_id in dict.fromkeys(checkpoint_ids) if checkpoint_id]
    if not unique:
        return
    for chunk in _lerobot_chunks(unique, 250):
        predicate = (
            "checkpoint_id IN (" + ", ".join(_lerobot_sql_literal(value) for value in chunk) + ")"
        )
        lake.table(_LEROBOT_CHECKPOINT_TABLE).delete(predicate)


def _lerobot_chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _lerobot_sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _lerobot_metric_dict(obj: Any, names: tuple[str, ...]) -> dict[str, int]:
    return {name: int(getattr(obj, name, 0) or 0) for name in names}


def _lerobot_checkpoint_fragment_count(lake: Lake) -> int:
    try:
        return len(lake.table(_LEROBOT_CHECKPOINT_TABLE).to_lance().get_fragments())
    except Exception:  # noqa: BLE001 - report best-effort fragment metrics for older LanceDB backends
        return 0


def _latest_lerobot_ingest_job(lake: Lake, job_id: str) -> dict[str, Any] | None:
    rows = [row for row in _lerobot_checkpoint_rows(lake) if row.get("job_id") == job_id]
    if not rows:
        return None
    return max(rows, key=_lerobot_checkpoint_sort_key)


def _lerobot_media_inspection_cache(lake: Lake, job_id: str) -> tuple[dict[str, Any], ...]:
    rows = [
        row
        for row in sorted(
            _lerobot_checkpoint_rows(lake),
            key=_lerobot_checkpoint_sort_key,
            reverse=True,
        )
        if row.get("job_id") == job_id
    ]
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for row in rows:
        progress_json = row.get("progress_json")
        if not progress_json:
            continue
        try:
            progress = json.loads(progress_json)
        except json.JSONDecodeError:
            continue
        media_inspection = progress.get("media_inspection")
        if not isinstance(media_inspection, dict):
            continue
        for video in media_inspection.get("videos") or []:
            if not isinstance(video, dict):
                continue
            if video.get("inspection_status") != "completed":
                continue
            if not video.get("keyframe_map") and video.get("keyframe_map_ref"):
                try:
                    video["keyframe_map"] = keyframe_map_entries_from_json(
                        load_keyframe_map_json(lake, str(video["keyframe_map_ref"]))
                    )
                except KeyframeMapError:
                    pass
            fingerprint = str(video.get("inspection_fingerprint") or "")
            if fingerprint and fingerprint not in by_fingerprint:
                by_fingerprint[fingerprint] = dict(video)
    return tuple(by_fingerprint.values())


def _lerobot_ingest_job_id(source_id: str) -> str:
    return f"lerobot-ingest-{source_id.removeprefix('src-')}"


def _lerobot_completed_ingest_transform_exists(lake: Lake, transform_id: str) -> bool:
    return (
        int(
            lake.table("transform_runs").count_rows(
                f"transform_id = '{transform_id}' AND status = 'completed'"
            )
        )
        > 0
    )


def _next_lerobot_checkpoint_index(lake: Lake, job_id: str) -> int:
    latest = _latest_lerobot_ingest_job(lake, job_id)
    if latest is None:
        return 0
    return int(latest.get("checkpoint_index") or 0) + 1


def _lerobot_claim_cas_supersede(
    lake: Lake,
    *,
    job_id: str,
    prior_checkpoint_id: str,
    prior_claim_token: str | None,
    new_checkpoint_id: str,
    operation: str,
) -> None:
    """Atomically win the right to supersede ``prior_checkpoint_id``.

    A plain append -- or an insert-only ``merge_insert`` -- never conflicts
    under a genuinely concurrent race: Lance treats both as commutative
    inserts and happily commits both, producing two rows with the same
    intended-unique ``checkpoint_id`` (verified directly against this
    lancedb/lance version; see the 0379 decision record). ``update(where=...)``
    is different: Lance re-evaluates the ``where`` clause against the
    freshest committed state on every internal retry, so gating on
    ``superseded_by_checkpoint_id IS NULL`` (plus the caller's observed
    ``checkpoint_id``/``claim_token``) means a concurrent loser's retried
    filter matches zero rows once the winner has already flipped it --
    ``rows_updated`` tells the loser it lost, deterministically, even under a
    true simultaneous race. Only after this call succeeds is it safe to
    append the new checkpoint row.

    Raises ``LeRobotClaimPreconditionError`` if this caller lost the race.
    """
    table = lake.table(_LEROBOT_CHECKPOINT_TABLE)
    where = (
        f"checkpoint_id = {_lerobot_sql_literal(prior_checkpoint_id)} "
        f"AND claim_token = {_lerobot_sql_literal(prior_claim_token or '')} "
        "AND superseded_by_checkpoint_id IS NULL"
    )
    result = table.update(where=where, values={"superseded_by_checkpoint_id": new_checkpoint_id})
    if int(result.rows_updated) == 1:
        return
    _raise_lerobot_claim_lost_race(
        lake, job_id, operation=operation, checkpoint_id=new_checkpoint_id
    )


def _raise_lerobot_claim_lost_race(
    lake: Lake, job_id: str, *, operation: str, checkpoint_id: str
) -> None:
    """Raise a clear diagnostic naming the checkpoint that actually won a CAS race."""
    committed = _latest_lerobot_ingest_job(lake, job_id)
    raise LeRobotClaimPreconditionError(
        job_id,
        operation=operation,
        lake_uri=lake.uri,
        expected={"checkpoint_id": checkpoint_id},
        actual=_lerobot_claim_precondition_actual(committed),
    )


def _assert_lerobot_claim_precondition(
    job_id: str,
    latest: dict[str, Any] | None,
    *,
    operation: str,
    lake_uri: str | None,
    expected_latest_checkpoint_id: str | None,
    expected_latest_claim_token: str | None,
    expected_checkpoint_index: int | None,
) -> None:
    expected: dict[str, Any] = {}
    if expected_latest_checkpoint_id is not None:
        expected["checkpoint_id"] = str(expected_latest_checkpoint_id)
    if expected_latest_claim_token is not None:
        expected["claim_token"] = str(expected_latest_claim_token)
    if expected_checkpoint_index is not None:
        expected["checkpoint_index"] = int(expected_checkpoint_index)
    if not expected:
        return

    actual = _lerobot_claim_precondition_actual(latest)
    mismatched = {
        key: expected_value
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatched:
        raise LeRobotClaimPreconditionError(
            job_id,
            operation=operation,
            lake_uri=lake_uri,
            expected=expected,
            actual=actual,
        )


def _lerobot_claim_precondition_actual(latest: dict[str, Any] | None) -> dict[str, Any]:
    if latest is None:
        return {
            "checkpoint_id": None,
            "claim_token": None,
            "checkpoint_index": None,
            "status": None,
            "phase": None,
            "claim_owner": None,
            "claim_generation": None,
            "updated_at": None,
            "claim_expires_at": None,
            "stale": None,
            "stale_reason": "missing-latest",
        }
    progress = _lerobot_progress_from_checkpoint_row(latest)
    claim = _lerobot_claim_from_progress(progress)
    expires_at, expiration_source = _lerobot_claim_expiration_detail(
        latest,
        stale_after=DEFAULT_LEROBOT_CLAIM_LEASE,
    )
    now = datetime.now(UTC)
    status = str(latest.get("status") or "")
    if status != "running":
        stale = False
        stale_reason = f"latest-status-{status or 'unknown'}"
    elif expires_at is None:
        stale = True
        stale_reason = "missing-expiration"
    elif expires_at <= now:
        stale = True
        stale_reason = {
            "claim_expires_at": "expired-claim",
            "lease_seconds": "lease-timeout",
            "missing-lease": "missing-lease",
        }.get(expiration_source, "expired-claim")
    else:
        stale = False
        stale_reason = "missing-lease" if expiration_source == "missing-lease" else "live"
    updated_at = _coerce_lerobot_utc(latest.get("updated_at") or latest.get("created_at"))
    return {
        "checkpoint_id": str(latest.get("checkpoint_id") or ""),
        "claim_token": str(latest.get("claim_token") or claim.get("token") or "") or None,
        "checkpoint_index": int(latest.get("checkpoint_index") or 0),
        "status": status,
        "phase": str(latest.get("phase") or ""),
        "claim_owner": str(latest.get("claim_owner") or claim.get("owner") or "") or None,
        "claim_generation": _optional_lerobot_int(claim.get("generation")),
        "updated_at": updated_at,
        "claim_expires_at": expires_at,
        "stale": stale,
        "stale_reason": stale_reason,
    }


def _assert_lerobot_ingest_claimable(lake: Lake, job_id: str) -> None:
    latest = _latest_lerobot_ingest_job(lake, job_id)
    if latest is not None and latest.get("status") == "running":
        progress = _lerobot_progress_from_checkpoint_row(latest)
        claim = _lerobot_claim_from_progress(progress)
        owner = latest.get("claim_owner") or claim.get("owner")
        token = latest.get("claim_token") or claim.get("token")
        expires_at = _lerobot_claim_expires_at(latest, stale_after=DEFAULT_LEROBOT_CLAIM_LEASE)
        details = []
        if owner:
            details.append(f"owner={owner}")
        if token:
            details.append(f"token={token}")
        if expires_at is not None and expires_at > datetime.now(UTC):
            details.append(f"lease expires at {expires_at.isoformat()}")
        elif expires_at is not None:
            details.append(
                f"lease appears stale since {expires_at.isoformat()}; run "
                f"`lancedb-robotics ingest lerobot-claim-recover {job_id}` "
                "before retrying"
            )
        suffix = "; " + ", ".join(details) if details else ""
        raise AdapterError(
            f"LeRobot ingest job {job_id} is already running{suffix}; inspect "
            "`lancedb-robotics ingest lerobot-job` for progress or wait for the "
            "current claim to finish"
        )


def _record_lerobot_completed_checkpoint_if_needed(
    lake: Lake,
    *,
    dataset,
    inspect_report: dict[str, Any],
    job_id: str,
    source_id: str,
    run_id: str,
    transform_id: str,
    started_at: datetime,
    created_by: str,
    claim_owner: str,
    source_arg: str | Path,
    claim_token: str,
    keyframe_map_offloads: dict[tuple[int, str], dict[str, Any]] | None = None,
    keyframe_map_artifacts: dict[str, Any] | None = None,
) -> None:
    latest = _latest_lerobot_ingest_job(lake, job_id)
    phase = "reused-completed" if latest is not None else "already-completed"
    expected_observations = int(
        inspect_report.get("frame_count") or inspect_report.get("message_count") or 0
    )
    observation_count = int(lake.table("observations").count_rows(f"run_id = '{run_id}'"))
    progress = _lerobot_progress_template(
        dataset,
        expected_observations=expected_observations,
        existing_observations=observation_count,
        batch_size=0,
        source_identity=inspect_report.get("source_identity") or {},
    )
    progress["status"] = "completed"
    progress["rows_seen"] = max(expected_observations, observation_count)
    progress["rows_written"]["observations"] = observation_count
    progress["rows_written"]["episodes"] = int(
        lake.table("episodes").count_rows(f"run_id = '{run_id}'")
    )
    progress["rows_written"]["scenarios"] = int(
        lake.table("scenarios").count_rows(f"run_id = '{run_id}'")
    )
    progress["rows_written"]["videos"] = int(
        lake.table("videos").count_rows(f"run_id = '{run_id}'")
    )
    progress["rows_written"]["video_encodings"] = int(
        lake.table("video_encodings").count_rows(f"run_id = '{run_id}'")
    )
    progress["rows_written"]["keyframe_map_artifacts"] = int(
        lake.table("keyframe_map_artifacts").count_rows(f"run_id = '{run_id}'")
    )
    progress["rows_written"]["keyframe_map_artifact_referrers"] = int(
        lake.table("keyframe_map_artifact_referrers").count_rows(f"run_id = '{run_id}'")
    )
    progress["video_diagnostics"] = _lerobot_video_diagnostics(dataset.video_files)
    progress["media_inspection"] = _lerobot_sanitized_media_inspection(
        dataset.media_inspection,
        keyframe_map_offloads or {},
    )
    progress["keyframe_map_artifacts"] = (
        keyframe_map_artifacts or _empty_lerobot_keyframe_map_progress()
    )
    _record_lerobot_ingest_checkpoint(
        lake,
        dataset=dataset,
        job_id=job_id,
        source_id=source_id,
        run_id=run_id,
        transform_id=transform_id,
        progress=progress,
        status="completed",
        phase=phase,
        checkpoint_index=_next_lerobot_checkpoint_index(lake, job_id),
        started_at=started_at,
        created_by=created_by,
        claim_owner=claim_owner,
        source_arg=source_arg,
        claim_token=claim_token,
    )


def _upsert_lerobot_rows(
    lake: Lake,
    table_name: str,
    key_column: str,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=schema)
    (
        lake.table(table_name)
        .merge_insert(key_column)
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(table)
    )


def _record_lerobot_failed_checkpoint(
    lake: Lake,
    *,
    dataset,
    job_id: str,
    source_id: str,
    run_id: str,
    transform_id: str,
    progress: dict[str, Any],
    phase: str,
    checkpoint_index: int,
    started_at: datetime,
    created_by: str,
    claim_owner: str,
    source_arg: str | Path,
    claim_token: str,
    error: BaseException,
) -> None:
    progress["status"] = "failed"
    _record_lerobot_ingest_checkpoint(
        lake,
        dataset=dataset,
        job_id=job_id,
        source_id=source_id,
        run_id=run_id,
        transform_id=transform_id,
        progress=progress,
        status="failed",
        phase=phase,
        checkpoint_index=checkpoint_index,
        started_at=started_at,
        created_by=created_by,
        claim_owner=claim_owner,
        source_arg=source_arg,
        claim_token=claim_token,
        error=str(error),
    )


def _record_lerobot_ingest_checkpoint(
    lake: Lake,
    *,
    dataset,
    job_id: str,
    source_id: str,
    run_id: str,
    transform_id: str,
    progress: dict[str, Any],
    status: str,
    phase: str,
    checkpoint_index: int,
    started_at: datetime,
    created_by: str,
    claim_owner: str,
    source_arg: str | Path,
    claim_token: str,
    error: str | None = None,
    enforce_bootstrap_cas: bool = False,
) -> None:
    now = datetime.now(UTC)
    progress["status"] = status
    _refresh_lerobot_claim_lease(
        progress,
        status=status,
        claim_owner=claim_owner,
        claim_token=claim_token,
        now=now,
    )
    rows_written = progress.get("rows_written") or {}
    last = progress.get("last_checkpoint") or {}
    hf_download = _lerobot_hf_download_ledger(source_arg, dataset, progress)
    finished_at = now if status in _LEROBOT_TERMINAL_STATUSES else None
    row = {
        "checkpoint_id": f"{job_id}:{int(checkpoint_index):08d}",
        "job_id": job_id,
        "source_id": source_id,
        "run_id": run_id,
        "transform_id": transform_id,
        "source_uri": dataset.source.uri,
        "source_ref": hf_download.get("source_ref") or dataset.source.uri,
        "hf_repo_id": dataset.source.repo_id,
        "requested_revision": hf_download.get("requested_revision"),
        "resolved_revision": hf_download.get("resolved_revision"),
        "hf_cache_path": hf_download.get("cache_path"),
        "hf_download_json": json.dumps(hf_download, sort_keys=True, separators=(",", ":")),
        "source_identity_json": json.dumps(
            progress.get("source_identity") or {},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "status": status,
        "phase": phase,
        "claim_owner": claim_owner,
        "claim_token": claim_token,
        "checkpoint_index": int(checkpoint_index),
        "data_file": last.get("data_file"),
        "row_group": _optional_int(last.get("row_group")),
        "batch_index": _optional_int(last.get("batch_index")),
        "rows_seen": int(progress.get("rows_seen") or 0),
        "observations_written": int(rows_written.get("observations") or 0),
        "episodes_written": int(rows_written.get("episodes") or 0),
        "scenarios_written": int(rows_written.get("scenarios") or 0),
        "videos_written": int(rows_written.get("videos") or 0),
        "video_encodings_written": int(rows_written.get("video_encodings") or 0),
        "rows_skipped_existing": int(progress.get("rows_skipped_existing") or 0),
        "bytes_scanned": int(progress.get("bytes_scanned") or 0),
        "last_observation_id": progress.get("last_observation_id"),
        "progress_json": json.dumps(progress, sort_keys=True, separators=(",", ":")),
        "error": error,
        "started_at": started_at,
        "updated_at": now,
        "finished_at": finished_at,
        "created_by": created_by,
        "created_at": now,
    }
    data = pa.Table.from_pylist([row], schema=LEROBOT_INGEST_CHECKPOINTS_SCHEMA)
    if not enforce_bootstrap_cas:
        lake.table(_LEROBOT_CHECKPOINT_TABLE).add(data)
        return
    # Bootstrap: the very first checkpoint ever written for this job_id has no
    # prior row to CAS against (see _lerobot_claim_cas_supersede). Best-effort
    # dedup via merge_insert, then a read-back check: it closes the realistic
    # staggered-race window (one caller commits, the other's later, freshly
    # read merge correctly no-ops), but -- verified directly -- cannot
    # guarantee rejection of a truly simultaneous commit the way the
    # supersede-gated CAS above does. Detect that rare case after the fact
    # rather than silently accepting a duplicate first claim.
    table = lake.table(_LEROBOT_CHECKPOINT_TABLE)
    checkpoint_id = str(row["checkpoint_id"])
    table.merge_insert("checkpoint_id").when_not_matched_insert_all().execute(data)
    winners = [
        candidate
        for candidate in _lerobot_checkpoint_rows(lake)
        if candidate.get("checkpoint_id") == checkpoint_id
    ]
    if len(winners) != 1 or str(winners[0].get("claim_token") or "") != claim_token:
        _raise_lerobot_claim_lost_race(
            lake, job_id, operation="claim", checkpoint_id=checkpoint_id
        )


def _append_lerobot_recovery_checkpoint(
    lake: Lake,
    latest: dict[str, Any],
    *,
    progress: dict[str, Any],
    checkpoint_index: int,
    checkpoint_id: str,
    phase: str,
    claim_owner: str,
    claim_token: str,
    created_by: str,
    now: datetime,
) -> None:
    rows_written = progress.get("rows_written") or {}
    last = progress.get("last_checkpoint") or {}
    row = {
        **latest,
        "checkpoint_id": checkpoint_id,
        "status": "abandoned",
        "phase": phase,
        "claim_owner": claim_owner,
        "claim_token": claim_token,
        "checkpoint_index": int(checkpoint_index),
        "data_file": last.get("data_file"),
        "row_group": _optional_int(last.get("row_group")),
        "batch_index": _optional_int(last.get("batch_index")),
        "rows_seen": int(progress.get("rows_seen") or 0),
        "observations_written": int(rows_written.get("observations") or 0),
        "episodes_written": int(rows_written.get("episodes") or 0),
        "scenarios_written": int(rows_written.get("scenarios") or 0),
        "videos_written": int(rows_written.get("videos") or 0),
        "video_encodings_written": int(rows_written.get("video_encodings") or 0),
        "rows_skipped_existing": int(progress.get("rows_skipped_existing") or 0),
        "bytes_scanned": int(progress.get("bytes_scanned") or 0),
        "last_observation_id": progress.get("last_observation_id"),
        "progress_json": json.dumps(progress, sort_keys=True, separators=(",", ":")),
        "source_identity_json": json.dumps(
            progress.get("source_identity") or {},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "error": None,
        "updated_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    lake.table(_LEROBOT_CHECKPOINT_TABLE).add(
        pa.Table.from_pylist([row], schema=LEROBOT_INGEST_CHECKPOINTS_SCHEMA)
    )


def _lerobot_hf_download_ledger(
    source_arg: str | Path,
    dataset,
    progress: dict[str, Any],
) -> dict[str, Any]:
    requested_revision: str | None = None
    value = str(source_arg)
    if dataset.source.repo_id and "@" in value:
        requested_revision = value.rsplit("@", maxsplit=1)[1] or None
    resolved_revision = dataset.source.revision
    source_ref = dataset.source.uri
    if dataset.source.repo_id:
        source_ref = f"hf://{dataset.source.repo_id}" + (
            f"@{resolved_revision}" if resolved_revision else ""
        )
    return {
        "repo_id": dataset.source.repo_id,
        "repo_type": "dataset" if dataset.source.repo_id else None,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "cache_path": str(dataset.source.root) if dataset.source.repo_id else None,
        "source_ref": source_ref,
        "input_uris": list(dataset.source.input_uris),
        "source_identity": dict(progress.get("source_identity") or {}),
        "manifest_fingerprints": [
            {
                "path": item.get("path"),
                "fingerprint": item.get("fingerprint"),
            }
            for item in progress.get("data_files") or []
        ],
    }


def _safe_id(value: str) -> str:
    key = "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")
    return key or "camera"


def _ingest_split_recording(
    lake: Lake,
    recording: Recording,
    *,
    created_by: str,
    batch_size: int,
    validate_crcs: bool,
    compact: bool = True,
    prune_versions: bool = True,
    retain_versions: int = DEFAULT_INGEST_RETAIN_VERSIONS,
    index_predicates: bool = True,
    auth_ref: str | None = None,
) -> IngestReport:
    """Ingest a split recording's ordered shards as exactly one ``run`` (backlog 0019).

    The ``run_id``/``source_id`` are content-addressed from the *ordered shard
    checksums* (so they are reorder- and relocation-stable, and a re-ingest of the
    same directory is a no-op). Messages stream shard-by-shard in canonical order
    with a recording-global per-topic ``sequence`` that is continuous across shard
    boundaries, and each observation's ``raw_uri`` points at the specific shard it
    came from. The ``run`` row's ``raw_uri`` is the recording root (directory),
    which ``export`` resolves back into shards.

    Aggregate time range / profile come from the merged shard inspection (the
    declared ``metadata.yaml`` range when present). A damaged shard quarantines the
    whole run, keeping every message recovered up to the damage; a missing codec is
    still the one hard error. Single-file ingest is untouched — it never reaches
    here.
    """
    digest = recording_content_key(recording.checksums)[:16]
    run_id = f"run-{digest}"
    source_id = f"src-{digest}"
    uri = recording.uri

    inspect_started = datetime.now(UTC)
    merged_report = inspect_recording(recording.root)  # codec/invalid shard -> hard error
    inspect_finished = datetime.now(UTC)

    if lake.table("runs").count_rows(f"run_id = '{run_id}'") > 0:
        report = _already_ingested_report(
            lake,
            source_id,
            uri,
            recording_content_key(recording.checksums),
            run_id,
            merged_report,
            auth_ref=auth_ref,
            created_by=created_by,
        )
        return report

    now = datetime.now(UTC)
    inspect_transform_id = f"tfm-{digest}-inspect"
    ingest_transform_id = f"tfm-{digest}-ingest"

    observations_table = lake.table("observations")
    batch: list[dict] = []
    by_topic: dict[str, int] = {}
    topic_schema: dict[str, tuple[str | None, str | None]] = {}
    decode_by_status: dict[str, int] = {}
    decode_by_encoding: dict[str, int] = {}
    decode_raw_by_encoding: dict[str, int] = {}
    extracted_by_modality: dict[str, int] = {}
    stream_start: int | None = None
    stream_end: int | None = None
    total = 0
    integrity_status = "complete"
    integrity_reason: str | None = None
    recovered_count = 0

    try:
        for message in iter_shard_messages(recording, validate_crcs=validate_crcs):
            topic = message["topic"]
            by_topic[topic] = by_topic.get(topic, 0) + 1
            topic_schema.setdefault(topic, (message["schema_name"], message["schema_encoding"]))
            status = message["decode_status"]
            decode_by_status[status] = decode_by_status.get(status, 0) + 1
            encoding = message["message_encoding"] or "unknown"
            decode_by_encoding[encoding] = decode_by_encoding.get(encoding, 0) + 1
            if status == "raw":
                decode_raw_by_encoding[encoding] = decode_raw_by_encoding.get(encoding, 0) + 1
            # raw_uri is the *shard* the bytes live in, not the recording root.
            row = _observation_row(
                message,
                run_id=run_id,
                raw_uri=message["shard_uri"],
                transform_id=ingest_transform_id,
                created_at=now,
            )
            if row["modality"] != "unknown":
                extracted_by_modality[row["modality"]] = (
                    extracted_by_modality.get(row["modality"], 0) + 1
                )
            ts = message["log_time_ns"]
            stream_start = ts if stream_start is None else min(stream_start, ts)
            stream_end = ts if stream_end is None else max(stream_end, ts)
            batch.append(row)
            total += 1
            if len(batch) >= batch_size:
                _flush(observations_table, batch)
    except CorruptMcapError as exc:
        if total == 0:
            raise
        integrity_status = exc.status
        integrity_reason = exc.reason
        recovered_count = exc.recovered or total
    _flush(observations_table, batch)

    source_report = merged_report if merged_report["topics"] else _synthetic_report(topic_schema)
    registration = register_recording_source(
        lake, recording, adapter="mcap", inspect_report=source_report, auth_ref=auth_ref
    )

    # Attachment + metadata records aggregate across shards, in shard order.
    adapter = get_adapter("mcap")
    attachment_records: list[dict] = []
    metadata_records: list[dict] = []
    for shard in recording.shards:
        attachment_records.extend(_safe_records(adapter.attachments, shard.path))
        metadata_records.extend(_safe_records(adapter.metadata_records, shard.path))
    attachment_rows = [
        {
            "attachment_id": f"{run_id}:att:{index:04d}",
            "run_id": run_id,
            "name": att["name"],
            "media_type": att["media_type"],
            "size": att["size"],
            "sha256": att["sha256"],
            "log_time_ns": att["log_time_ns"],
            "create_time_ns": att["create_time_ns"],
            "data": att["data"],
            "transform_id": ingest_transform_id,
            "created_at": now,
        }
        for index, att in enumerate(attachment_records)
    ]
    metadata_kv = [
        {"key": f"{record['name']}.{key}", "value": value}
        for record in metadata_records
        for key, value in sorted(record["metadata"].items())
    ]

    # Time coverage: the union of the merged/declared range and what actually
    # streamed, so the run row always spans its observations even when a damaged
    # shard contributes recovered messages the (readable-only) merge excludes.
    starts: list[int] = [s for s in (stream_start,) if s is not None]
    ends: list[int] = [e for e in (stream_end,) if e is not None]
    if merged_report["message_count"]:
        starts.append(merged_report["start_time_ns"])
        ends.append(merged_report["end_time_ns"])
    start_time_ns = min(starts) if starts else 0
    end_time_ns = max(ends) if ends else 0
    duration_ns = max(0, end_time_ns - start_time_ns)

    run_metadata = [
        {"key": "profile", "value": merged_report["profile"]},
        {"key": "library", "value": merged_report["library"]},
        # Recording shape (backlog 0019): this run was assembled from N shards.
        {"key": "recording.shard_count", "value": str(len(recording.shards))},
    ]
    run_metadata += metadata_kv
    run_metadata.append({"key": "integrity.status", "value": integrity_status})
    if integrity_status != "complete":
        run_metadata.append({"key": "integrity.recovered", "value": str(recovered_count)})
        if integrity_reason:
            run_metadata.append({"key": "integrity.reason", "value": integrity_reason})

    quality_flags = None
    if integrity_status != "complete":
        quality_flags = ["quarantined", f"integrity:{integrity_status}"]

    run_row = {
        "run_id": run_id,
        "run_kind": "log",
        "source": "mcap",
        "source_id": registration.source_id,
        "raw_uri": registration.uri,
        "start_time_ns": start_time_ns,
        "end_time_ns": end_time_ns,
        "duration_ns": duration_ns,
        "metadata": run_metadata,
        "quality_flags": quality_flags,
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
            "source": "message-boundary",
            "transform_id": ingest_transform_id,
            "created_at": now,
        }
        for event_type, timestamp_ns in (
            ("run_start", start_time_ns),
            ("run_end", end_time_ns),
        )
    ]

    ingest_output_tables = ["runs", "observations"]
    if attachment_rows:
        ingest_output_tables.append("attachments")
    ingest_output_tables.append("events")

    shard_uris = [shard.uri for shard in recording.shards]
    ingest_finished = datetime.now(UTC)
    transform_rows = [
        {
            "transform_id": inspect_transform_id,
            "kind": "inspect",
            "source_id": registration.source_id,
            "input_uris": shard_uris,
            "output_tables": [],
            "params": json.dumps({"adapter": "mcap", "shard_count": len(recording.shards)}),
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
            "input_uris": shard_uris,
            "output_tables": ingest_output_tables,
            "params": json.dumps(
                {
                    "adapter": "mcap",
                    "run_id": run_id,
                    "batch_size": batch_size,
                    "decode_by_status": dict(sorted(decode_by_status.items())),
                    "decode_by_encoding": dict(sorted(decode_by_encoding.items())),
                    # Which encodings still landed raw (missing extra / unsupported
                    # schema encoding); empty when everything decoded (backlog 0020).
                    "decode_raw_by_encoding": dict(sorted(decode_raw_by_encoding.items())),
                    "extracted_by_modality": dict(sorted(extracted_by_modality.items())),
                    "extract_layout_version": LAYOUT_VERSION,
                    "attachment_count": len(attachment_rows),
                    "metadata_record_count": len(metadata_records),
                    "integrity": {
                        "status": integrity_status,
                        "recovered": recovered_count,
                        "reason": integrity_reason,
                    },
                    # Split-recording shape (backlog 0019): the ordered shard
                    # inventory and any flagged timeline gaps/overlaps.
                    "recording": {
                        "shard_count": len(recording.shards),
                        "shards": [
                            {"name": s.path.name, "checksum": s.checksum} for s in recording.shards
                        ],
                        "gaps": merged_report["gaps"],
                        "metadata_path": merged_report.get("metadata_path"),
                    },
                }
            ),
            "status": "completed" if integrity_status == "complete" else "recovered",
            "started_at": inspect_started,
            "finished_at": ingest_finished,
            "created_by": created_by,
            "created_at": now,
        },
    ]

    lake.table("runs").add(pa.Table.from_pylist([run_row], schema=RUNS_SCHEMA))
    if attachment_rows:
        lake.table("attachments").add(
            pa.Table.from_pylist(attachment_rows, schema=ATTACHMENTS_SCHEMA)
        )
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

    # Emit lineage for the inspect + ingest transforms inline (backlog 0098),
    # after finalize so the emitted table versions match the compacted lake.
    for _transform_row in transform_rows:
        emit_transform_lineage(lake, _transform_row)

    return IngestReport(
        lake_uri=lake.uri,
        compaction=compaction,
        source=registration,
        run_id=run_id,
        already_ingested=False,
        transform_id=ingest_transform_id,
        rows_added={
            "integration_sources": 1 if registration.created else 0,
            "runs": 1,
            "observations": total,
            "attachments": len(attachment_rows),
            "events": len(event_rows),
            "transform_runs": len(transform_rows),
        },
        observations_by_topic=dict(sorted(by_topic.items())),
        message_count=total,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        duration_ns=duration_ns,
        decode_by_status=dict(sorted(decode_by_status.items())),
        decode_by_encoding=dict(sorted(decode_by_encoding.items())),
        decode_raw_by_encoding=dict(sorted(decode_raw_by_encoding.items())),
        integrity_status=integrity_status,
        integrity_reason=integrity_reason,
        recovered_count=recovered_count,
    )


def _finalize_ingest(
    lake: Lake,
    *,
    compact: bool,
    prune_versions: bool,
    retain_versions: int,
    created_by: str,
    index_predicates: bool = True,
) -> dict[str, Any] | None:
    """Make the freshly-streamed ``observations`` grain a well-formed, indexed table.

    Three operations, all on by default, run once per ingest (not per flush):

    1. **Compaction** (BUG-14): each ``_flush`` above is one ``table.add`` = one
       Lance fragment + one version, so the grain is born at ~``batch_size``
       rows/fragment. Compaction merges the run's sub-target fragments up to
       Lance's healthy ~1M-row target; it only rewrites fragments *below* target,
       so it stays incremental as more runs are appended.
    2. **Version pruning** (BUG-14): snapshot-safe -- versions pinned by a live
       ``dataset_snapshots`` (or lineage) are tagged and never removed.
    3. **Predicate indexing** (BUG-15): build the ``observations`` scalar indexes
       (``run_id``/``observation_id``/``timestamp_ns`` BTREE, ``topic`` BITMAP) so
       ``run_id``-filtered reads push down to an indexed scan of one run.

    All three run through ``maintain_lake`` in the order compact -> build/refresh
    indexes -> prune. That order is load-bearing: indexes are built *after*
    compaction (which rewrites files and remaps indexes -- lance
    ``read_and_write.md``: "rewrite files before re-building indices"), and the
    version prune runs *last* so it also collects the compaction/index-build
    versions and the per-flush churn stays bounded. Pinning/cleanup safety lives in
    one place. ``index_predicates=False`` skips the index build (e.g. bulk
    backfill, then a single ``lake maintain``).
    """
    if not compact and not prune_versions and not index_predicates:
        return None
    # Local import: the maintenance -> lineage subgraph is heavier than ingest
    # needs at import time, and only this once-per-ingest finalize uses it.
    # maintain_lake runs compact -> refresh/build indexes -> cleanup in that order,
    # which is exactly what we want: predicate indexes are built *after* compaction
    # remaps files, and the final version prune also collects the index-build
    # versions so the churn stays bounded.
    from lancedb_robotics.maintenance import maintain_lake

    report = maintain_lake(
        lake,
        tables=("observations",),
        compact=compact,
        refresh_indexes=index_predicates,  # builds the observations predicate indexes (BUG-15)
        protect_lineage=True,  # tag-protect lineage-referenced versions before any prune
        refresh_lineage=False,  # do not rebuild the whole lineage graph on the ingest path
        cleanup_older_than=timedelta(0) if prune_versions else None,
        retain_versions=retain_versions if prune_versions else None,
        created_by=created_by,
    )
    table_report = report.tables.get("observations")
    if table_report is None:
        return None
    summary: dict[str, Any] = {
        "transform_id": report.transform_id,
        "fragments_before": table_report.fragments_before,
        "fragments_after": table_report.fragments_after,
        "version_before": table_report.version_before,
        "version_after": table_report.version_after,
        "pinned_versions": list(table_report.pinned_versions),
        "cleanup": table_report.cleanup,
    }
    if index_predicates:
        summary["indexes"] = sorted(
            entry["column"]
            for entry in table_report.indexes_refreshed
            if entry.get("column") and entry.get("status") in {"built", "already_present"}
        )
    return summary or None


def _flush(table, rows: list[dict]) -> None:
    """Write a batch of observation rows to the lake and clear it (streaming ingest)."""
    if rows:
        table.add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
        rows.clear()


def _safe_inspect(
    adapter,
    path: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
    auth_ref: str | None = None,
) -> dict | None:
    """Inspect ``path``; return None if it is damaged-but-readable.

    A :class:`CorruptMcapError` (truncation / CRC) means "no usable summary, but
    the message stream can still be recovered" — the caller streams the prefix
    instead. :class:`CodecUnavailableError` and other :class:`AdapterError`
    (missing file, not MCAP, summary-less) propagate as hard errors.
    """
    try:
        return adapter.inspect(path, storage_options=storage_options, auth_ref=auth_ref)
    except CorruptMcapError:
        return None


def _safe_records(
    reader,
    path: str | Path,
    *,
    storage_options: dict[str, Any] | None = None,
    auth_ref: str | None = None,
) -> list[dict]:
    """List attachment/metadata records, tolerating a damaged file (returns [])."""
    try:
        return list(reader(path, storage_options=storage_options, auth_ref=auth_ref))
    except CorruptMcapError:
        return []


def _register_resolved_source(
    lake: Lake,
    source,
    *,
    adapter: str,
    inspect_report: dict,
    auth_ref: str | None = None,
) -> SourceRegistration:
    """Register a source whose adapter already resolved content identity."""
    source_id = f"src-{source.digest}"
    table = lake.table("integration_sources")
    if table.count_rows(f"source_id = '{source_id}'") > 0:
        return SourceRegistration(
            source_id=source_id,
            uri=source.uri,
            checksum=source.checksum,
            created=False,
            auth_ref=auth_ref,
        )

    metadata = [
        {"key": "checksum", "value": source.checksum},
        {"key": "adapter", "value": adapter},
        {"key": "kind", "value": source.kind},
        {"key": "storage_identifier", "value": source.storage_identifier},
    ]
    if getattr(source, "identity_kind", None):
        metadata.append({"key": "source_identity.kind", "value": str(source.identity_kind)})
    object_store_validation = getattr(source, "object_store_validation", None)
    if object_store_validation:
        metadata.extend(
            [
                {
                    "key": "source_identity.validation_policy",
                    "value": str(object_store_validation.get("policy") or ""),
                },
                {
                    "key": "source_identity.assurance",
                    "value": str(object_store_validation.get("assurance") or ""),
                },
                {
                    "key": "source_identity.warning_count",
                    "value": str(len(object_store_validation.get("warnings") or ())),
                },
            ]
        )
    if getattr(source, "repo_id", None):
        metadata.append({"key": "hf_repo_id", "value": str(source.repo_id)})
    if getattr(source, "revision", None):
        metadata.append({"key": "hf_revision", "value": str(source.revision)})
    for index, uri in enumerate(source.input_uris):
        metadata.append({"key": f"file:{index}", "value": uri})
    for topic in inspect_report["topics"]:
        metadata.append(
            {
                "key": f"schema:{topic['topic']}",
                "value": f"{topic['schema_name']}:{topic['schema_encoding']}",
            }
        )

    row = {
        "source_id": source_id,
        "kind": source.kind,
        "display_name": display_name(source.uri),
        "uri": source.uri,
        "auth_ref": auth_ref,
        "metadata": metadata,
        "created_at": datetime.now(UTC),
    }
    table.add(pa.Table.from_pylist([row], schema=INTEGRATION_SOURCES_SCHEMA))
    return SourceRegistration(
        source_id=source_id,
        uri=source.uri,
        checksum=source.checksum,
        created=True,
        auth_ref=auth_ref,
    )


def _synthetic_report(topic_schema: dict[str, tuple[str | None, str | None]]) -> dict:
    """A minimal inspect-report stand-in built from the recovered stream.

    Only the ``topics`` fingerprints are needed by source registration; a damaged
    file has no real summary to inspect.
    """
    return {
        "topics": [
            {"topic": topic, "schema_name": name, "schema_encoding": encoding}
            for topic, (name, encoding) in sorted(topic_schema.items())
        ]
    }


def _already_ingested_report(
    lake: Lake,
    source_id: str,
    uri: str,
    checksum: str,
    run_id: str,
    inspect_report: dict | None,
    *,
    auth_ref: str | None = None,
    created_by: str = "lancedb-robotics",
    adapter_name: str = "mcap",
    ingest_job_id: str | None = None,
) -> IngestReport:
    now = datetime.now(UTC)
    rows_added = dict.fromkeys(
        (
            "integration_sources",
            "runs",
            "observations",
            "attachments",
            "events",
        ),
        0,
    )
    params = {
        "adapter": adapter_name,
        "run_id": run_id,
        "duplicate": True,
        "rows_added": rows_added,
    }
    if inspect_report:
        if inspect_report.get("media_inspection") is not None:
            params["media_inspection"] = inspect_report["media_inspection"]
        if inspect_report.get("diagnostics") is not None:
            params["diagnostics"] = inspect_report["diagnostics"]
    transform_id = (
        f"tfm-{run_id.removeprefix('run-')}-ingest-skip-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    transform_row = {
        "transform_id": transform_id,
        "kind": "ingest",
        "source_id": source_id,
        "input_uris": [uri],
        "output_tables": [],
        "params": json.dumps(params, sort_keys=True),
        "status": "skipped-duplicate",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    lake.table("transform_runs").add(
        pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA)
    )
    # Emit lineage for the skipped-duplicate ingest attempt inline (backlog 0098).
    emit_transform_lineage(lake, transform_row)
    zero_tables = (
        "integration_sources",
        "runs",
        "observations",
        "attachments",
        "events",
    )
    by_topic = (
        {t["topic"]: t["message_count"] for t in inspect_report["topics"]} if inspect_report else {}
    )
    return IngestReport(
        lake_uri=lake.uri,
        source=SourceRegistration(
            source_id=source_id,
            uri=uri,
            checksum=checksum,
            created=False,
            auth_ref=auth_ref,
        ),
        run_id=run_id,
        already_ingested=True,
        transform_id=transform_id,
        ingest_job_id=ingest_job_id,
        rows_added={**dict.fromkeys(zero_tables, 0), "transform_runs": 1},
        observations_by_topic=by_topic,
        message_count=inspect_report["message_count"] if inspect_report else 0,
        start_time_ns=inspect_report["start_time_ns"] if inspect_report else 0,
        end_time_ns=inspect_report["end_time_ns"] if inspect_report else 0,
        duration_ns=inspect_report["duration_ns"] if inspect_report else 0,
    )
