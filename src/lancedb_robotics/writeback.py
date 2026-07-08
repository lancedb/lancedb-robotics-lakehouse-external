"""Closed-loop writeback for labels, model outputs, and feedback.

The writeback tables are the source of record: they preserve full tool payloads,
reviewer/model/source metadata, and transform lineage. The hot curation/training
values are also denormalized onto the target grain (``observations`` and/or
``scenarios``) as additive Lance columns, so filters stay single-table
predicates while audit follows ids back to the provenance rows.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.lineage import emit_transform_lineage
from lancedb_robotics.schemas import (
    FEEDBACK_SCHEMA,
    LABELS_SCHEMA,
    MODEL_OUTPUTS_SCHEMA,
    TRANSFORM_RUNS_SCHEMA,
)

MAX_WRITEBACK_FILE_BYTES = 5 * 1024 * 1024
MAX_JSON_FIELD_BYTES = 1 * 1024 * 1024

_LABEL_DENORM_TYPES = {
    "label_id": pa.string(),
    "label_type": pa.string(),
    "label": pa.string(),
    "label_confidence": pa.float32(),
    "label_status": pa.string(),
}
_MODEL_OUTPUT_DENORM_TYPES = {
    "model_output_id": pa.string(),
    "model_version": pa.string(),
    "prediction": pa.string(),
    "prediction_score": pa.float32(),
}
_FEEDBACK_DENORM_TYPES = {
    "feedback_id": pa.string(),
    "feedback_type": pa.string(),
    "feedback_severity": pa.string(),
}


class WritebackError(Exception):
    """Raised when closed-loop writeback input is invalid or cannot be applied."""


@dataclass(frozen=True)
class WritebackReport:
    """Summary of one closed-loop writeback transform."""

    lake_uri: str
    kind: str
    transform_id: str
    rows_written: int
    row_ids: tuple[str, ...]
    output_tables: tuple[str, ...]
    target_tables: tuple[str, ...]


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _as_rows(rows: Iterable[dict[str, Any]] | dict[str, Any], noun: str) -> list[dict[str, Any]]:
    if isinstance(rows, dict):
        normalized = [rows]
    elif isinstance(rows, (str, bytes)):
        raise WritebackError(f"{noun} payload must be an object or list of objects")
    else:
        normalized = list(rows)
    if not normalized:
        raise WritebackError(f"no {noun} rows supplied")
    for index, row in enumerate(normalized):
        if not isinstance(row, dict):
            raise WritebackError(f"{noun} row {index} must be a JSON object")
    return normalized


def load_writeback_rows(path: str | Path, *, key: str) -> list[dict[str, Any]]:
    """Load a JSON writeback file.

    Accepts either a list of row objects, a single row object, or an envelope
    object with ``key`` holding the list. Oversized and malformed files fail
    before any lake writes happen.
    """
    file = Path(path)
    size = file.stat().st_size
    if size > MAX_WRITEBACK_FILE_BYTES:
        raise WritebackError(
            f"{file} is {size} bytes; maximum writeback file size is "
            f"{MAX_WRITEBACK_FILE_BYTES} bytes"
        )
    try:
        payload = json.loads(file.read_text())
    except json.JSONDecodeError as exc:
        raise WritebackError(f"{file} is not valid JSON: {exc.msg}") from exc

    if isinstance(payload, dict) and key in payload:
        payload = payload[key]
    return _as_rows(payload, key)


def _optional_str(row: dict[str, Any], key: str, *, default: str | None = None) -> str | None:
    value = row.get(key, default)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _required_str(row: dict[str, Any], key: str, noun: str) -> str:
    value = _optional_str(row, key)
    if value is None:
        raise WritebackError(f"{noun} row is missing required field {key!r}")
    return value


def _optional_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if row.get(key) is None:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError) as exc:
            raise WritebackError(f"{key} must be numeric") from exc
    return None


def _json_field(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key not in row or row[key] is None:
            continue
        value = row[key]
        payload = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        byte_size = len(payload.encode())
        if byte_size > MAX_JSON_FIELD_BYTES:
            raise WritebackError(
                f"{key} is {byte_size} bytes; maximum JSON field size is "
                f"{MAX_JSON_FIELD_BYTES} bytes"
            )
        return payload
    return None


def _metadata(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("metadata")
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [{"key": str(key), "value": _metadata_value(value)} for key, value in sorted(raw.items())]
    if isinstance(raw, list):
        items: list[dict[str, str]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict) or "key" not in item:
                raise WritebackError(f"metadata item {index} must contain key/value fields")
            items.append({"key": str(item["key"]), "value": _metadata_value(item.get("value", ""))})
        return items
    raise WritebackError("metadata must be an object or list of key/value objects")


def _metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _rows_by_id(lake: Lake, table: str, id_column: str) -> dict[str, dict[str, Any]]:
    return {row[id_column]: row for row in lake.table(table).to_arrow().to_pylist()}


def _lookup_maps(lake: Lake) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "runs": _rows_by_id(lake, "runs", "run_id"),
        "observations": _rows_by_id(lake, "observations", "observation_id"),
        "scenarios": _rows_by_id(lake, "scenarios", "scenario_id"),
        "events": _rows_by_id(lake, "events", "event_id"),
        "dataset_snapshots": _rows_by_id(lake, "dataset_snapshots", "dataset_id"),
        "labels": _rows_by_id(lake, "labels", "label_id"),
        "model_outputs": _rows_by_id(lake, "model_outputs", "model_output_id"),
    }


def _resolve_target(
    row: dict[str, Any],
    maps: dict[str, dict[str, dict[str, Any]]],
    *,
    noun: str,
    require_grain: bool,
    require_any: bool = True,
) -> dict[str, str | None]:
    run_id = _optional_str(row, "run_id")
    observation_id = _optional_str(row, "observation_id")
    scenario_id = _optional_str(row, "scenario_id")
    event_id = _optional_str(row, "event_id")
    dataset_id = _optional_str(row, "dataset_id")

    if observation_id is not None:
        observation = maps["observations"].get(observation_id)
        if observation is None:
            raise WritebackError(f"unknown observation_id {observation_id!r} in {noun} row")
        run_id = _merge_run_id(run_id, observation["run_id"], f"observation_id {observation_id!r}")

    if scenario_id is not None:
        scenario = maps["scenarios"].get(scenario_id)
        if scenario is None:
            raise WritebackError(f"unknown scenario_id {scenario_id!r} in {noun} row")
        run_id = _merge_run_id(run_id, scenario["run_id"], f"scenario_id {scenario_id!r}")

    if event_id is not None:
        event = maps["events"].get(event_id)
        if event is None:
            raise WritebackError(f"unknown event_id {event_id!r} in {noun} row")
        run_id = _merge_run_id(run_id, event["run_id"], f"event_id {event_id!r}")

    if run_id is not None and run_id not in maps["runs"]:
        raise WritebackError(f"unknown run_id {run_id!r} in {noun} row")
    if dataset_id is not None and dataset_id not in maps["dataset_snapshots"]:
        raise WritebackError(f"unknown dataset_id {dataset_id!r} in {noun} row")

    if require_grain and observation_id is None and scenario_id is None:
        raise WritebackError(f"{noun} row must target an observation_id or scenario_id")
    if require_any and all(value is None for value in (run_id, observation_id, scenario_id, event_id)):
        raise WritebackError(
            f"{noun} row must target at least one of run_id, observation_id, scenario_id, event_id"
        )

    return {
        "run_id": run_id,
        "observation_id": observation_id,
        "scenario_id": scenario_id,
        "event_id": event_id,
        "dataset_id": dataset_id,
    }


def _merge_run_id(existing: str | None, implied: str | None, source: str) -> str | None:
    if implied is None:
        return existing
    if existing is not None and existing != implied:
        raise WritebackError(f"run_id {existing!r} does not match {source}'s run_id {implied!r}")
    return implied


def _denorm_updates(
    rows: list[dict[str, Any]],
    fields: dict[str, pa.DataType],
    *,
    source_fields: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    source_fields = source_fields or {}
    updates = {"observations": {}, "scenarios": {}}
    for row in rows:
        values = {name: row.get(source_fields.get(name, name)) for name in fields}
        if row.get("observation_id"):
            updates["observations"][row["observation_id"]] = values
        if row.get("scenario_id"):
            updates["scenarios"][row["scenario_id"]] = values
    return updates


def _apply_denormalized_updates(
    lake: Lake,
    table_name: str,
    id_column: str,
    updates: dict[str, dict[str, Any]],
    field_types: dict[str, pa.DataType],
) -> bool:
    if not updates:
        return False

    table = lake.table(table_name)
    dataset = table.to_lance()
    schema_names = set(table.schema.names)
    projected = [id_column] + [name for name in field_types if name in schema_names]
    current = dataset.to_table(columns=projected)
    if current.num_rows == 0:
        return False

    ids = current[id_column].to_pylist()
    values_by_column: dict[str, list[Any]] = {}
    for name in field_types:
        values_by_column[name] = (
            current[name].to_pylist() if name in current.column_names else [None] * len(ids)
        )

    for index, row_id in enumerate(ids):
        update = updates.get(row_id)
        if update is None:
            continue
        for name, value in update.items():
            values_by_column[name][index] = value

    existing = [name for name in field_types if name in schema_names]
    if existing:
        dataset.drop_columns(existing)
    dataset.add_columns(
        pa.table(
            {
                name: pa.array(values, type=field_types[name])
                for name, values in values_by_column.items()
            }
        )
    )
    return True


def _delete_existing(lake: Lake, table_name: str, id_column: str, ids: list[str]) -> None:
    if not ids:
        return
    lake.table(table_name).delete(f"{id_column} IN ({', '.join(_sql_literal(i) for i in ids)})")


def _table_versions(lake: Lake, tables: Iterable[str]) -> list[dict[str, Any]]:
    versions = []
    for table in sorted(set(tables)):
        versions.append({"table": table, "version": int(lake.table(table).version), "tag": ""})
    return versions


def _source_transform_ids(
    rows: list[dict[str, Any]], maps: dict[str, dict[str, dict[str, Any]]]
) -> list[str]:
    ids: set[str] = set()
    for row in rows:
        for table_name, id_column in (
            ("runs", "run_id"),
            ("observations", "observation_id"),
            ("scenarios", "scenario_id"),
            ("events", "event_id"),
            ("dataset_snapshots", "dataset_id"),
        ):
            row_id = row.get(id_column)
            if not row_id:
                continue
            source = maps[table_name].get(row_id)
            if source and source.get("transform_id"):
                ids.add(source["transform_id"])
    return sorted(ids)


def _write_transform(
    lake: Lake,
    *,
    transform_id: str,
    kind: str,
    source: str,
    rows: list[dict[str, Any]],
    row_id_column: str,
    input_tables: Iterable[str],
    output_tables: Iterable[str],
    source_transform_ids: list[str],
    created_by: str,
    now: datetime,
) -> None:
    transform_row = {
        "transform_id": transform_id,
        "kind": kind,
        "source_id": source,
        "input_uris": [],
        "input_table_versions": _table_versions(lake, input_tables),
        "output_tables": sorted(set(output_tables)),
        "params": json.dumps(
            {
                "source": source,
                "row_ids": [row[row_id_column] for row in rows],
                "target_run_ids": sorted({row["run_id"] for row in rows if row.get("run_id")}),
                "target_observation_ids": sorted(
                    {row["observation_id"] for row in rows if row.get("observation_id")}
                ),
                "target_scenario_ids": sorted(
                    {row["scenario_id"] for row in rows if row.get("scenario_id")}
                ),
                "source_transform_ids": source_transform_ids,
            },
            sort_keys=True,
        ),
        "status": "completed",
        "started_at": now,
        "finished_at": now,
        "created_by": created_by,
        "created_at": now,
    }
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = {_sql_literal(transform_id)}")
    transforms.add(pa.Table.from_pylist([transform_row], schema=TRANSFORM_RUNS_SCHEMA))
    # Emit lineage inline (backlog 0098): labels/model-outputs/feedback writeback
    # records its execution + target row-set without a later refresh_graph().
    emit_transform_lineage(lake, transform_row)


def _finish_writeback(
    lake: Lake,
    *,
    table_name: str,
    schema: pa.Schema,
    id_column: str,
    kind: str,
    source: str,
    rows: list[dict[str, Any]],
    target_tables: set[str],
    input_tables: set[str],
    source_transform_ids: list[str],
    created_by: str,
    now: datetime,
) -> WritebackReport:
    row_ids = [row[id_column] for row in rows]
    transform_id = f"tfm-{kind}-{_digest({'source': source, 'rows': rows})}"
    for row in rows:
        row["transform_id"] = transform_id
        row["created_at"] = now

    _delete_existing(lake, table_name, id_column, row_ids)
    lake.table(table_name).add(pa.Table.from_pylist(rows, schema=schema))

    output_tables = {table_name, *target_tables}
    _write_transform(
        lake,
        transform_id=transform_id,
        kind=kind,
        source=source,
        rows=rows,
        row_id_column=id_column,
        input_tables=input_tables,
        output_tables=output_tables,
        source_transform_ids=source_transform_ids,
        created_by=created_by,
        now=now,
    )
    return WritebackReport(
        lake_uri=lake.uri,
        kind=kind,
        transform_id=transform_id,
        rows_written=len(rows),
        row_ids=tuple(row_ids),
        output_tables=tuple(sorted(output_tables)),
        target_tables=tuple(sorted(target_tables)),
    )


def import_labels(
    lake: Lake,
    labels: Iterable[dict[str, Any]] | dict[str, Any],
    *,
    source: str = "label-import",
    created_by: str = "lancedb-robotics",
) -> WritebackReport:
    """Import human or automatic labels and denormalize label scalars."""
    raw_rows = _as_rows(labels, "labels")
    maps = _lookup_maps(lake)
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        target = _resolve_target(raw, maps, noun="label", require_grain=True)
        label_type = _required_str(raw, "label_type", "label")
        label = _optional_str(raw, "label")
        if label is None and raw.get("value") is not None and not isinstance(raw["value"], (dict, list)):
            label = str(raw["value"])
        if label is None:
            raise WritebackError("label row is missing required field 'label'")
        label_value = _json_field(raw, "label_value", "value") or json.dumps(label)
        label_spec = _json_field(raw, "label_spec", "spec")
        target = {key: value for key, value in target.items() if key != "dataset_id"}
        row = {
            "label_id": _optional_str(raw, "label_id"),
            **target,
            "label_type": label_type,
            "label": label,
            "label_value": label_value,
            "label_spec": label_spec,
            "source": _optional_str(raw, "source", default=source),
            "reviewer": _optional_str(raw, "reviewer"),
            "confidence": _optional_float(raw, "confidence"),
            "status": _optional_str(raw, "status", default="active"),
            "metadata": _metadata(raw),
        }
        row["label_id"] = row["label_id"] or f"lbl-{_digest(row)}"
        rows.append(row)

    updates = _denorm_updates(
        rows,
        _LABEL_DENORM_TYPES,
        source_fields={"label_confidence": "confidence", "label_status": "status"},
    )
    target_tables: set[str] = set()
    if _apply_denormalized_updates(
        lake, "observations", "observation_id", updates["observations"], _LABEL_DENORM_TYPES
    ):
        target_tables.add("observations")
    if _apply_denormalized_updates(
        lake, "scenarios", "scenario_id", updates["scenarios"], _LABEL_DENORM_TYPES
    ):
        target_tables.add("scenarios")

    return _finish_writeback(
        lake,
        table_name="labels",
        schema=LABELS_SCHEMA,
        id_column="label_id",
        kind="label-writeback",
        source=source,
        rows=rows,
        target_tables=target_tables,
        input_tables={"runs", "observations", "scenarios", "events"},
        source_transform_ids=_source_transform_ids(rows, maps),
        created_by=created_by,
        now=now,
    )


def ingest_model_outputs(
    lake: Lake,
    outputs: Iterable[dict[str, Any]] | dict[str, Any],
    *,
    source: str = "model-output-import",
    created_by: str = "lancedb-robotics",
) -> WritebackReport:
    """Ingest model predictions/inferences and denormalize prediction scalars."""
    raw_rows = _as_rows(outputs, "model_outputs")
    maps = _lookup_maps(lake)
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        target = _resolve_target(raw, maps, noun="model output", require_grain=True)
        model_version = _required_str(raw, "model_version", "model output")
        prediction = _optional_str(raw, "prediction")
        if prediction is None and raw.get("output") is not None and not isinstance(raw["output"], (dict, list)):
            prediction = str(raw["output"])
        output_json = _json_field(raw, "output_json", "output")
        if prediction is None and output_json is None:
            raise WritebackError("model output row must include prediction or output")
        row = {
            "model_output_id": _optional_str(raw, "model_output_id"),
            **target,
            "model_version": model_version,
            "output_type": _optional_str(raw, "output_type", default="prediction"),
            "prediction": prediction,
            "output_json": output_json or json.dumps(prediction),
            "score": _optional_float(raw, "score", "confidence"),
            "producer_run_id": _optional_str(raw, "producer_run_id", default=_optional_str(raw, "model_run_id")),
            "source": _optional_str(raw, "source", default=source),
            "metadata": _metadata(raw),
        }
        row["model_output_id"] = row["model_output_id"] or f"out-{_digest(row)}"
        rows.append(row)

    updates = _denorm_updates(
        rows,
        _MODEL_OUTPUT_DENORM_TYPES,
        source_fields={"prediction_score": "score"},
    )
    target_tables: set[str] = set()
    if _apply_denormalized_updates(
        lake,
        "observations",
        "observation_id",
        updates["observations"],
        _MODEL_OUTPUT_DENORM_TYPES,
    ):
        target_tables.add("observations")
    if _apply_denormalized_updates(
        lake, "scenarios", "scenario_id", updates["scenarios"], _MODEL_OUTPUT_DENORM_TYPES
    ):
        target_tables.add("scenarios")

    return _finish_writeback(
        lake,
        table_name="model_outputs",
        schema=MODEL_OUTPUTS_SCHEMA,
        id_column="model_output_id",
        kind="model-output-writeback",
        source=source,
        rows=rows,
        target_tables=target_tables,
        input_tables={"runs", "observations", "scenarios", "dataset_snapshots"},
        source_transform_ids=_source_transform_ids(rows, maps),
        created_by=created_by,
        now=now,
    )


def record_feedback(
    lake: Lake,
    feedback: Iterable[dict[str, Any]] | dict[str, Any],
    *,
    source: str = "feedback-import",
    created_by: str = "lancedb-robotics",
) -> WritebackReport:
    """Record fleet/sim/incident feedback signals and denormalize severity."""
    raw_rows = _as_rows(feedback, "feedback")
    maps = _lookup_maps(lake)
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        raw = dict(raw)
        label_id = _optional_str(raw, "label_id")
        if label_id is not None:
            label = maps["labels"].get(label_id)
            if label is None:
                raise WritebackError(f"unknown label_id {label_id!r} in feedback row")
            _inherit_target(raw, label)

        model_output_id = _optional_str(raw, "model_output_id")
        if model_output_id is not None:
            output = maps["model_outputs"].get(model_output_id)
            if output is None:
                raise WritebackError(f"unknown model_output_id {model_output_id!r} in feedback row")
            _inherit_target(raw, output)

        target = _resolve_target(
            raw,
            maps,
            noun="feedback",
            require_grain=False,
            require_any=_optional_str(raw, "linked_incident_id") is None,
        )
        feedback_type = _required_str(raw, "feedback_type", "feedback")
        severity = _required_str(raw, "severity", "feedback")
        row = {
            "feedback_id": _optional_str(raw, "feedback_id"),
            **{key: value for key, value in target.items() if key != "dataset_id"},
            "label_id": label_id,
            "model_output_id": model_output_id,
            "feedback_type": feedback_type,
            "severity": severity,
            "linked_incident_id": _optional_str(raw, "linked_incident_id"),
            "notes": _optional_str(raw, "notes"),
            "source": _optional_str(raw, "source", default=source),
            "status": _optional_str(raw, "status", default="open"),
            "metadata": _metadata(raw),
        }
        row["feedback_id"] = row["feedback_id"] or f"fb-{_digest(row)}"
        rows.append(row)

    updates = _denorm_updates(
        rows,
        _FEEDBACK_DENORM_TYPES,
        source_fields={"feedback_severity": "severity"},
    )
    target_tables: set[str] = set()
    if _apply_denormalized_updates(
        lake, "observations", "observation_id", updates["observations"], _FEEDBACK_DENORM_TYPES
    ):
        target_tables.add("observations")
    if _apply_denormalized_updates(
        lake, "scenarios", "scenario_id", updates["scenarios"], _FEEDBACK_DENORM_TYPES
    ):
        target_tables.add("scenarios")

    return _finish_writeback(
        lake,
        table_name="feedback",
        schema=FEEDBACK_SCHEMA,
        id_column="feedback_id",
        kind="feedback-writeback",
        source=source,
        rows=rows,
        target_tables=target_tables,
        input_tables={"runs", "observations", "scenarios", "events", "labels", "model_outputs"},
        source_transform_ids=_source_transform_ids(rows, maps),
        created_by=created_by,
        now=now,
    )


def _inherit_target(raw: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("run_id", "observation_id", "scenario_id", "event_id"):
        if raw.get(key) is None and source.get(key) is not None:
            raw[key] = source[key]


def trace_model_output(lake: Lake, model_output_id: str) -> dict[str, Any]:
    """Resolve a model output to its input grain, snapshot, source run, and lineage.

    This is an audit/id traversal helper, not a curation hot path.
    """
    outputs = _rows_by_id(lake, "model_outputs", "model_output_id")
    output = outputs.get(model_output_id)
    if output is None:
        raise WritebackError(f"unknown model_output_id {model_output_id!r}")

    observation = _row_or_none(lake, "observations", "observation_id", output.get("observation_id"))
    scenario = _row_or_none(lake, "scenarios", "scenario_id", output.get("scenario_id"))
    if scenario is None and observation is not None:
        scenario = _scenario_containing_observation(lake, observation["observation_id"])
    dataset = _row_or_none(lake, "dataset_snapshots", "dataset_id", output.get("dataset_id"))
    if dataset is None and scenario is not None:
        dataset = _snapshot_containing_scenario(lake, scenario["scenario_id"])

    run_id = output.get("run_id")
    if run_id is None and observation is not None:
        run_id = observation.get("run_id")
    if run_id is None and scenario is not None:
        run_id = scenario.get("run_id")
    source_run = _row_or_none(lake, "runs", "run_id", run_id)

    transform_ids = {
        row.get("transform_id")
        for row in (output, observation, scenario, dataset, source_run)
        if row and row.get("transform_id")
    }
    transform_ids.update(_param_source_transform_ids(lake, output.get("transform_id")))

    return {
        "model_output": output,
        "observation": observation,
        "scenario": scenario,
        "dataset_snapshot": dataset,
        "source_run": source_run,
        "transform_runs": _transform_rows(lake, transform_ids),
    }


def downstream_for_run(lake: Lake, run_id: str) -> dict[str, Any]:
    """Find label, output, and feedback rows downstream of a run by id traversal."""
    run = _row_or_none(lake, "runs", "run_id", run_id)
    if run is None:
        raise WritebackError(f"unknown run_id {run_id!r}")

    observations = [
        row for row in lake.table("observations").to_arrow().to_pylist() if row["run_id"] == run_id
    ]
    scenarios = [
        row for row in lake.table("scenarios").to_arrow().to_pylist() if row["run_id"] == run_id
    ]
    events = [row for row in lake.table("events").to_arrow().to_pylist() if row["run_id"] == run_id]
    observation_ids = {row["observation_id"] for row in observations}
    scenario_ids = {row["scenario_id"] for row in scenarios}
    event_ids = {row["event_id"] for row in events}

    labels = [
        row
        for row in lake.table("labels").to_arrow().to_pylist()
        if _targets_run(row, run_id, observation_ids, scenario_ids, event_ids)
    ]
    model_outputs = [
        row
        for row in lake.table("model_outputs").to_arrow().to_pylist()
        if _targets_run(row, run_id, observation_ids, scenario_ids, event_ids)
    ]
    label_ids = {row["label_id"] for row in labels}
    output_ids = {row["model_output_id"] for row in model_outputs}
    feedback = [
        row
        for row in lake.table("feedback").to_arrow().to_pylist()
        if _targets_run(row, run_id, observation_ids, scenario_ids, event_ids)
        or row.get("label_id") in label_ids
        or row.get("model_output_id") in output_ids
    ]

    return {
        "run": run,
        "observation_ids": sorted(observation_ids),
        "scenario_ids": sorted(scenario_ids),
        "event_ids": sorted(event_ids),
        "labels": labels,
        "model_outputs": model_outputs,
        "feedback": feedback,
    }


def _targets_run(
    row: dict[str, Any],
    run_id: str,
    observation_ids: set[str],
    scenario_ids: set[str],
    event_ids: set[str],
) -> bool:
    return (
        row.get("run_id") == run_id
        or row.get("observation_id") in observation_ids
        or row.get("scenario_id") in scenario_ids
        or row.get("event_id") in event_ids
    )


def _row_or_none(lake: Lake, table: str, id_column: str, row_id: str | None) -> dict[str, Any] | None:
    if row_id is None:
        return None
    return _rows_by_id(lake, table, id_column).get(row_id)


def _scenario_containing_observation(lake: Lake, observation_id: str) -> dict[str, Any] | None:
    for row in lake.table("scenarios").to_arrow().to_pylist():
        if observation_id in (row.get("observation_ids") or []):
            return row
    return None


def _snapshot_containing_scenario(lake: Lake, scenario_id: str) -> dict[str, Any] | None:
    for row in lake.table("dataset_snapshots").to_arrow().to_pylist():
        try:
            spec = json.loads(row.get("query_spec") or "{}")
        except json.JSONDecodeError:
            continue
        if scenario_id in (spec.get("scenario_ids") or []):
            return row
    return None


def _param_source_transform_ids(lake: Lake, transform_id: str | None) -> set[str]:
    if transform_id is None:
        return set()
    for row in lake.table("transform_runs").to_arrow().to_pylist():
        if row["transform_id"] != transform_id:
            continue
        try:
            params = json.loads(row.get("params") or "{}")
        except json.JSONDecodeError:
            return set()
        return set(params.get("source_transform_ids") or [])
    return set()


def _transform_rows(lake: Lake, transform_ids: set[str | None]) -> list[dict[str, Any]]:
    wanted = {value for value in transform_ids if value}
    rows = [
        row for row in lake.table("transform_runs").to_arrow().to_pylist() if row["transform_id"] in wanted
    ]
    return sorted(rows, key=lambda row: row["transform_id"])
