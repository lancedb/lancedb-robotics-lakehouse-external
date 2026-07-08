"""Shared projection/materialization accounting helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACCOUNTING_VERSION = "projection-accounting-v1"


@dataclass(frozen=True)
class ProjectionAccounting:
    """Logical-vs-materialized byte accounting for one snapshot boundary."""

    logical_row_count: int
    payload_bytes_referenced: int
    payload_bytes_copied: int
    metadata_bytes_written: int
    target_format: str
    target_path: str
    projection_transform_id: str
    source_snapshot_id: str
    source_snapshot_name: str
    source_table_versions: tuple[dict[str, Any], ...]
    selected_scenario_count: int = 0
    selected_observation_count: int = 0
    mode: str = ""
    payload_copy_policy: str = ""
    dry_run: bool = False
    payload_bytes_planned: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload_bytes_referenced = int(self.payload_bytes_referenced)
        payload_bytes_copied = int(self.payload_bytes_copied)
        payload_bytes_planned = int(self.payload_bytes_planned)
        logical_reference_bytes = max(0, payload_bytes_referenced - payload_bytes_copied)
        copy_ratio = (
            payload_bytes_copied / payload_bytes_referenced
            if payload_bytes_referenced
            else 0.0
        )
        return {
            "version": ACCOUNTING_VERSION,
            "logical_row_count": int(self.logical_row_count),
            "selected_scenario_count": int(self.selected_scenario_count),
            "selected_observation_count": int(self.selected_observation_count),
            "payload_bytes_referenced": payload_bytes_referenced,
            "payload_bytes_copied": payload_bytes_copied,
            "payload_bytes_planned": payload_bytes_planned,
            "planned_payload_bytes": payload_bytes_planned,
            "logical_reference_bytes": logical_reference_bytes,
            "metadata_bytes_written": int(self.metadata_bytes_written),
            "copy_ratio": copy_ratio,
            "target_format": self.target_format,
            "target_path": self.target_path,
            "projection_transform_id": self.projection_transform_id,
            "source_snapshot_id": self.source_snapshot_id,
            "source_snapshot_name": self.source_snapshot_name,
            "source_table_versions": [
                dict(version) for version in self.source_table_versions
            ],
            "mode": self.mode,
            "payload_copy_policy": self.payload_copy_policy,
            "dry_run": bool(self.dry_run),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ProjectionAccounting:
        data = dict(payload or {})
        copied = int(data.get("payload_bytes_copied") or 0)
        planned = data.get("payload_bytes_planned")
        if planned is None:
            planned = data.get("planned_payload_bytes")
        if planned is None:
            planned = data.get("payload_bytes_to_copy")
        if planned is None and bool(data.get("dry_run")):
            planned = copied
        return cls(
            logical_row_count=int(data.get("logical_row_count") or 0),
            selected_scenario_count=int(data.get("selected_scenario_count") or 0),
            selected_observation_count=int(data.get("selected_observation_count") or 0),
            payload_bytes_referenced=int(data.get("payload_bytes_referenced") or 0),
            payload_bytes_copied=copied,
            metadata_bytes_written=int(data.get("metadata_bytes_written") or 0),
            target_format=str(data.get("target_format") or ""),
            target_path=str(data.get("target_path") or ""),
            projection_transform_id=str(data.get("projection_transform_id") or ""),
            source_snapshot_id=str(data.get("source_snapshot_id") or ""),
            source_snapshot_name=str(data.get("source_snapshot_name") or ""),
            source_table_versions=tuple(
                normalize_table_versions(data.get("source_table_versions") or ())
            ),
            mode=str(data.get("mode") or ""),
            payload_copy_policy=str(data.get("payload_copy_policy") or ""),
            dry_run=bool(data.get("dry_run")),
            payload_bytes_planned=int(planned or 0),
        )


def normalize_table_versions(versions: Any) -> tuple[dict[str, Any], ...]:
    """Return table-version pins in the common manifest shape."""
    normalized: list[dict[str, Any]] = []
    for item in versions or ():
        if isinstance(item, dict):
            table = str(item.get("table") or "")
            version = item.get("version")
            tag = str(item.get("tag") or "")
        else:
            values = tuple(item)
            table = str(values[0]) if values else ""
            version = values[1] if len(values) > 1 else None
            tag = str(values[2]) if len(values) > 2 else ""
        if table and version is not None:
            normalized.append({"table": table, "version": int(version), "tag": tag})
    return tuple(normalized)


def json_metadata_bytes(payload: dict[str, Any]) -> int:
    """Return deterministic JSON byte size for manifest/accounting estimates."""
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def file_bytes(paths: Any) -> int:
    """Return total size of existing output files, ignoring missing paths."""
    total = 0
    for raw in paths or ():
        path = Path(raw)
        if path.is_file():
            total += path.stat().st_size
    return total


def metadata_bytes_written(paths: Any, *, payload_bytes_copied: int) -> int:
    """Approximate metadata/control bytes written beside copied payload bytes."""
    return max(0, file_bytes(paths) - int(payload_bytes_copied))


def payload_size(value: Any) -> int:
    """Best-effort size for inline bytes or lazy blob-reference metadata."""
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, dict):
        size = value.get("size")
        if size is not None:
            return int(size)
    return 0
