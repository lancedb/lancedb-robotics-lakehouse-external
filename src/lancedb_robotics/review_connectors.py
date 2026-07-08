"""Connector contracts for external review and labeling tools."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ReviewConnectorTask:
    """One review-queue item projected to an external review tool."""

    queue_id: str
    queue_name: str
    queue_item_id: str
    target_grain: str
    target_id: str
    scenario_id: str
    tool: str
    project_id: str
    idempotency_key: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "queue_name": self.queue_name,
            "queue_item_id": self.queue_item_id,
            "target_grain": self.target_grain,
            "target_id": self.target_id,
            "scenario_id": self.scenario_id,
            "tool": self.tool,
            "project_id": self.project_id,
            "idempotency_key": self.idempotency_key,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class ReviewConnectorResult:
    """Per-task connector export or status-sync result."""

    queue_item_id: str
    idempotency_key: str
    status: str
    external_task_id: str = ""
    external_url: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_item_id": self.queue_item_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "external_task_id": self.external_task_id,
            "external_url": self.external_url,
            "metadata": dict(self.metadata),
            "error": self.error,
        }


class ReviewToolConnector(Protocol):
    """Minimal external-tool connector surface used by review queues."""

    def upsert_tasks(
        self,
        tasks: Sequence[ReviewConnectorTask],
        *,
        project_id: str,
    ) -> Sequence[ReviewConnectorResult]:
        """Create or upsert external tasks for review queue items."""

    def sync_task_status(
        self,
        tasks: Sequence[ReviewConnectorTask],
        *,
        project_id: str,
    ) -> Sequence[ReviewConnectorResult]:
        """Fetch external task status for already-exported queue items."""

    def import_outcomes(
        self,
        tasks: Sequence[ReviewConnectorTask],
        *,
        project_id: str,
    ) -> Sequence[Mapping[str, Any]]:
        """Return reviewed outcome rows keyed by queue item, target, or scenario."""


class JsonFileReviewToolConnector:
    """Deterministic local connector backed by a JSON state file.

    This is intentionally simple: it exercises the connector contract without a
    network dependency and provides a concrete CLI target for dry runs, demos,
    and tests. Real Label Studio/FiftyOne connectors can implement the same
    protocol while preserving the lakehouse-side idempotency semantics.
    """

    def __init__(self, path: str | Path, *, tool: str = "json-file") -> None:
        self.path = Path(path)
        self.tool = tool

    def upsert_tasks(
        self,
        tasks: Sequence[ReviewConnectorTask],
        *,
        project_id: str,
    ) -> tuple[ReviewConnectorResult, ...]:
        state = self._load()
        project = self._project(state, project_id)
        results: list[ReviewConnectorResult] = []
        for task in tasks:
            stored = project["tasks"].get(task.idempotency_key)
            if stored is None:
                external_task_id = "json-task-" + _digest(
                    {"project_id": project_id, "idempotency_key": task.idempotency_key}
                )
                stored = {
                    "task": task.to_dict(),
                    "external_task_id": external_task_id,
                    "external_url": f"json://{project_id}/{external_task_id}",
                    "status": "exported",
                    "outcome": {},
                    "outcomes": [],
                }
                project["tasks"][task.idempotency_key] = stored
                status = "exported"
            else:
                status = "already-present"
                stored["task"] = task.to_dict()
            results.append(
                ReviewConnectorResult(
                    queue_item_id=task.queue_item_id,
                    idempotency_key=task.idempotency_key,
                    status=status,
                    external_task_id=str(stored.get("external_task_id") or ""),
                    external_url=str(stored.get("external_url") or ""),
                    metadata={"state_path": str(self.path)},
                )
            )
        self._save(state)
        return tuple(results)

    def sync_task_status(
        self,
        tasks: Sequence[ReviewConnectorTask],
        *,
        project_id: str,
    ) -> tuple[ReviewConnectorResult, ...]:
        state = self._load()
        project = self._project(state, project_id)
        results: list[ReviewConnectorResult] = []
        for task in tasks:
            stored = project["tasks"].get(task.idempotency_key)
            if stored is None:
                results.append(
                    ReviewConnectorResult(
                        queue_item_id=task.queue_item_id,
                        idempotency_key=task.idempotency_key,
                        status="failed",
                        error="external task is not present in connector state",
                        metadata={"state_path": str(self.path)},
                    )
                )
                continue
            results.append(
                ReviewConnectorResult(
                    queue_item_id=task.queue_item_id,
                    idempotency_key=task.idempotency_key,
                    status=str(stored.get("status") or "exported"),
                    external_task_id=str(stored.get("external_task_id") or ""),
                    external_url=str(stored.get("external_url") or ""),
                    metadata={"state_path": str(self.path)},
                )
            )
        return tuple(results)

    def import_outcomes(
        self,
        tasks: Sequence[ReviewConnectorTask],
        *,
        project_id: str,
    ) -> tuple[Mapping[str, Any], ...]:
        state = self._load()
        project = self._project(state, project_id)
        outcomes: list[Mapping[str, Any]] = []
        for task in tasks:
            stored = project["tasks"].get(task.idempotency_key) or {}
            for outcome in _stored_outcomes(stored):
                payload = dict(outcome)
                payload.setdefault("queue_item_id", task.queue_item_id)
                payload.setdefault("target_grain", task.target_grain)
                payload.setdefault("target_id", task.target_id)
                if task.scenario_id:
                    payload.setdefault("scenario_id", task.scenario_id)
                payload.setdefault("source", "human")
                metadata = dict(payload.get("metadata") or {})
                metadata.setdefault("external_task_id", stored.get("external_task_id") or "")
                metadata.setdefault("idempotency_key", task.idempotency_key)
                payload["metadata"] = metadata
                outcomes.append(payload)
        return tuple(outcomes)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"connector": self.tool, "projects": {}}
        with self.path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            return {"connector": self.tool, "projects": {}}
        loaded.setdefault("connector", self.tool)
        loaded.setdefault("projects", {})
        return loaded

    def _save(self, state: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _project(self, state: dict[str, Any], project_id: str) -> dict[str, Any]:
        projects = state.setdefault("projects", {})
        project = projects.setdefault(project_id, {"tasks": {}})
        project.setdefault("tasks", {})
        return project


def _stored_outcomes(stored: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    outcome = stored.get("outcome")
    if isinstance(outcome, Mapping) and outcome:
        rows.append(outcome)
    outcomes = stored.get("outcomes")
    if isinstance(outcomes, Sequence) and not isinstance(outcomes, (str, bytes)):
        rows.extend(row for row in outcomes if isinstance(row, Mapping))
    return rows


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]
