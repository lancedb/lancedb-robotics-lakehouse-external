"""Versioned schema + redaction conformance for the training loader report (0124).

Backlog 0073 makes every native or aligned training loader emit a redacted,
JSON-serializable ``TrainingLoaderReport`` (see :mod:`lancedb_robotics.training`).
External benchmarks (0074), the report catalog (0115), experiment trackers, and
support tooling all consume that JSON. They need two guarantees that the emitter
alone cannot prove after the fact:

1. **Shape** — the JSON conforms to a *stable, versioned* contract, so a consumer
   can rely on the presence and type of the routing/plan/metric fields.
2. **Safety** — the JSON never carries a resolved credential (API key, bearer /
   authorization header, object-store access key, scoped/STS token, or a
   namespace credential-vending value), while still preserving the *safe* display
   and auth-*reference* fields that make a report useful.

This module is the committed contract for both. It is dependency-free (stdlib
only) and deterministic:

- :data:`TRAINING_LOADER_REPORT_SCHEMA` is the canonical JSON Schema (Draft
  2020-12) for ``lancedb-robotics/training-loader-report/v1``, mirrored to a
  committed file at :data:`SCHEMA_JSON_PATH` for external (jsonschema-based)
  tooling. A drift test keeps the two identical.
- :func:`validate_report_schema` is a small, self-contained validator for the
  subset of JSON Schema keywords the contract uses, so validation works without a
  third-party ``jsonschema`` install.
- :func:`scan_report_secrets` independently proves redaction: it flags any
  secret-shaped *key* whose value survived unredacted, and any secret-shaped
  *value* (bearer/basic headers, AWS/GitHub/Slack keys, JWTs, PEM blocks) that
  slipped through under an innocuous key. It never flags ``*_auth_ref`` names or
  ``display_uri``.
- :func:`validate_training_loader_report` runs both and returns a
  :class:`ReportValidation`.

Compatibility rules (see also ``docs/manual/concepts/training-loader-report.md``):

- The report ``kind`` string carries the version: ``.../v1``. A consumer must
  branch on it.
- v1 validation is **open**: unknown object properties are allowed and ignored.
  This is deliberate — a future minor revision may *add* fields (new metrics, new
  policy knobs) to a v1 report and existing validators must keep passing.
- A **breaking** change (renaming/removing a required field, changing a type, or
  changing the meaning of a field) requires a new ``kind`` suffix (``/v2``) and a
  new schema document; v1 consumers keep validating v1 reports unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lancedb_robotics.redaction import DEFAULT_SECRET_VALUE_PATTERNS

#: The report ``kind`` / schema ``$id``. The ``/v1`` suffix is the version marker.
TRAINING_LOADER_REPORT_SCHEMA_ID = "lancedb-robotics/training-loader-report/v1"

#: Marker that :func:`lancedb_robotics.training._redact_report` substitutes for a
#: redacted secret. A conformant report holds this string wherever a secret-shaped
#: key appears.
REPORT_REDACTION_MARKER = "<redacted>"

#: Committed JSON copy of :data:`TRAINING_LOADER_REPORT_SCHEMA` for external
#: (``jsonschema``-based) tooling. Kept identical by
#: ``tests/test_training_report_schema.py``.
SCHEMA_JSON_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "manual"
    / "reference"
    / "training_loader_report.v1.schema.json"
)


def _nullable(*types: str) -> dict[str, Any]:
    return {"type": [*types, "null"]}


_WORKER_DEF = {
    "type": "object",
    "required": ["id", "num_workers"],
    "properties": {
        "id": {"type": "integer"},
        "num_workers": {"type": "integer"},
        "resume_from": {"type": "integer"},
    },
}

_TABLE_VERSIONS_DEF = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["table", "version"],
        "properties": {
            "table": {"type": "string"},
            "version": _nullable("integer"),
            "tag": {"type": "string"},
        },
    },
}

_ENTITY_REF_DEF = {
    "type": "object",
    "required": ["id", "name"],
    "properties": {"id": _nullable("string"), "name": _nullable("string")},
}

#: Canonical JSON Schema for a v1 training loader report. Objects are **open**
#: (no ``additionalProperties: false``) so a future v1 minor revision can add
#: fields without breaking existing validators; see the module docstring.
TRAINING_LOADER_REPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": TRAINING_LOADER_REPORT_SCHEMA_ID,
    "title": "LanceDB Robotics training loader report (v1)",
    "type": "object",
    "required": [
        "kind",
        "loader",
        "lake",
        "table_versions",
        "plans",
        "policies",
        "remote_execution",
        "metrics",
        "fallback_events",
        "disabled_capabilities",
        "run",
    ],
    "properties": {
        "kind": {"const": TRAINING_LOADER_REPORT_SCHEMA_ID},
        "loader": {
            "type": "object",
            "required": ["kind", "access_pattern"],
            "properties": {
                "kind": {"enum": ["native-training", "aligned-training"]},
                "access_pattern": {"type": "string"},
            },
        },
        "lake": {
            "type": "object",
            "required": [
                "uri",
                "display_uri",
                "backend_kind",
                "connection_kind",
                "execution_mode",
                "request_routing",
            ],
            "properties": {
                "uri": _nullable("string"),
                "display_uri": _nullable("string"),
                "backend_kind": _nullable("string"),
                "connection_kind": _nullable("string"),
                # 0129: how IO reached storage (local / object_store /
                # namespace_direct / remote_db / unclassified). Optional/additive.
                "data_plane": _nullable("string"),
                "execution_mode": _nullable("string"),
                "request_routing": {"type": "object"},
            },
        },
        "table_versions": {"$ref": "#/$defs/table_versions"},
        "read_table_versions": {"$ref": "#/$defs/table_versions"},
        "snapshot": {"$ref": "#/$defs/entity_ref"},
        "alignment": {"$ref": "#/$defs/entity_ref"},
        "plans": {
            "type": "object",
            "required": [
                "epoch",
                "epoch_plan_id",
                "ordering_policy",
                "worker",
                "selected_rows",
                "total_rows",
            ],
            "properties": {
                "epoch": _nullable("integer"),
                "epoch_plan_id": _nullable("string"),
                "ordering_policy": _nullable("string"),
                "worker": {"$ref": "#/$defs/worker"},
                "selected_rows": _nullable("integer"),
                "total_rows": _nullable("integer"),
            },
        },
        "policies": {
            "type": "object",
            "required": ["columns", "enterprise_cache"],
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}},
                "enterprise_cache": {"type": "object"},
            },
        },
        "remote_execution": {
            "type": "object",
            "required": [
                "requested_backend",
                "resolved_backend",
                "execution_mode",
                "connection_kind",
                "display_uri",
                "capabilities",
                "plan_executor",
                "request_routing",
                "fallback_policy",
                "warnings",
            ],
            "properties": {
                "requested_backend": _nullable("string"),
                "resolved_backend": _nullable("string"),
                "execution_mode": _nullable("string"),
                "connection_kind": _nullable("string"),
                "data_plane": _nullable("string"),
                "display_uri": _nullable("string"),
                "capabilities": {"type": "object"},
                "plan_executor": {"type": "object"},
                "request_routing": {"type": "object"},
                "fallback_policy": _nullable("string"),
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
        },
        "metrics": {
            "type": "object",
            "required": ["summary", "operations_by_type", "cache", "operations"],
            "properties": {
                "summary": {"type": "object"},
                "operations_by_type": {"type": "object"},
                "cache": {"type": "object"},
                "operations": {"type": "array"},
            },
        },
        "fallback_events": {"type": "array", "items": {"type": "object"}},
        "disabled_capabilities": {"type": "array", "items": {"type": "string"}},
        "run": {"type": "object"},
    },
    "allOf": [
        {
            "if": {
                "properties": {
                    "loader": {"properties": {"kind": {"const": "native-training"}}}
                }
            },
            "then": {"required": ["snapshot"]},
        },
        {
            "if": {
                "properties": {
                    "loader": {"properties": {"kind": {"const": "aligned-training"}}}
                }
            },
            "then": {"required": ["alignment", "read_table_versions"]},
        },
    ],
    "$defs": {
        "worker": _WORKER_DEF,
        "table_versions": _TABLE_VERSIONS_DEF,
        "entity_ref": _ENTITY_REF_DEF,
    },
}


# --------------------------------------------------------------------------- #
# Minimal, dependency-free JSON Schema validator.
#
# Supports exactly the keywords TRAINING_LOADER_REPORT_SCHEMA uses: type,
# properties, required, items, const, enum, $ref (local #/$defs/...), allOf, and
# if/then. Objects are validated open (unknown properties ignored). This keeps a
# single source of truth (the schema dict) drivable without a third-party lib.
# --------------------------------------------------------------------------- #

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (Mapping,),
    "array": (list, tuple),
    "string": (str,),
    "boolean": (bool,),
    "null": (type(None),),
}


def _matches_type(value: Any, json_type: str) -> bool:
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    expected = _JSON_TYPES.get(json_type)
    if expected is None:
        return True
    if json_type == "object":
        # A mapping, but not a string (strings are Sequences, not Mappings).
        return isinstance(value, Mapping)
    return isinstance(value, expected)


def _resolve_ref(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref: {ref!r}")
    node: Any = root
    for token in ref[2:].split("/"):
        node = node[token]
    return node


def _validate(
    schema: Mapping[str, Any],
    value: Any,
    path: str,
    root: Mapping[str, Any],
    errors: list[str],
) -> None:
    if "$ref" in schema:
        _validate(_resolve_ref(schema["$ref"], root), value, path, root, errors)
        return

    loc = path or "(root)"
    declared = schema.get("type")
    if declared is not None:
        allowed = [declared] if isinstance(declared, str) else list(declared)
        if not any(_matches_type(value, t) for t in allowed):
            errors.append(f"{loc}: expected type {allowed}, got {_type_name(value)}")
            return  # further keyword checks assume the base type held

    if "const" in schema and value != schema["const"]:
        errors.append(f"{loc}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{loc}: {value!r} is not one of {schema['enum']}")

    if isinstance(value, Mapping):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{loc}: missing required property '{name}'")
        for name, subschema in schema.get("properties", {}).items():
            if name in value:
                child = f"{path}.{name}" if path else name
                _validate(subschema, value[name], child, root, errors)

    if isinstance(value, (list, tuple)) and "items" in schema:
        for index, item in enumerate(value):
            _validate(schema["items"], item, f"{path}[{index}]", root, errors)

    for subschema in schema.get("allOf", []):
        _validate(subschema, value, path, root, errors)

    if "if" in schema:
        if _conforms(schema["if"], value, root):
            if "then" in schema:
                _validate(schema["then"], value, path, root, errors)
        elif "else" in schema:
            _validate(schema["else"], value, path, root, errors)


def _conforms(schema: Mapping[str, Any], value: Any, root: Mapping[str, Any]) -> bool:
    probe: list[str] = []
    _validate(schema, value, "", root, probe)
    return not probe


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def validate_report_schema(payload: Any) -> list[str]:
    """Return a list of schema violations for ``payload`` (empty means valid)."""
    errors: list[str] = []
    _validate(TRAINING_LOADER_REPORT_SCHEMA, payload, "", TRAINING_LOADER_REPORT_SCHEMA, errors)
    return errors


# --------------------------------------------------------------------------- #
# Redaction conformance scan.
# --------------------------------------------------------------------------- #

#: Case-insensitive key fragments that must never carry a live value in a report.
#: Kept in sync with ``training._secret_report_key``; ``*auth_ref*`` is exempt.
SECRET_KEY_FRAGMENTS: tuple[str, ...] = (
    "authorization",
    "api_key",
    "apikey",
    "access_key",
    "secret_key",
    "session_token",
    "password",
    "secret",
    "credential",
    "bearer",
    "token",
)

_SECRET_VALUE_REGEXES = tuple(
    re.compile(pattern) for pattern in DEFAULT_SECRET_VALUE_PATTERNS
)


def is_secret_report_key(key: str) -> bool:
    """True if ``key`` names a secret-shaped field that must be redacted.

    ``*auth_ref*`` names are safe *references* to a credential (not the secret
    itself) and are always preserved.
    """
    lowered = str(key).lower().replace("-", "_")
    if "auth_ref" in lowered:
        return False
    return any(fragment in lowered for fragment in SECRET_KEY_FRAGMENTS)


def is_secret_report_value(value: str) -> bool:
    """True if a string value looks like a live credential header or key."""
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered.startswith(("bearer ", "basic ")):
        return True
    return any(regex.search(stripped) for regex in _SECRET_VALUE_REGEXES)


def scan_report_secrets(payload: Any) -> list[str]:
    """Return redaction violations in ``payload`` (empty means safe).

    Two independent checks, so a gap in either the key policy or the emitter is
    caught:

    - a secret-shaped **key** (:func:`is_secret_report_key`) whose value is not
      the redaction marker ``<redacted>``; and
    - any **value** that matches a credential pattern
      (:func:`is_secret_report_value`) under any key.
    """
    findings: list[str] = []
    _scan(payload, "", findings)
    return findings


def _scan(value: Any, path: str, findings: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if is_secret_report_key(str(key)) and item != REPORT_REDACTION_MARKER:
                findings.append(
                    f"{child}: secret-shaped key is not redacted "
                    f"(value {_preview(item)})"
                )
            _scan(item, child, findings)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan(item, f"{path}[{index}]", findings)
        return
    if isinstance(value, str) and value != REPORT_REDACTION_MARKER and is_secret_report_value(value):
        findings.append(f"{path}: value looks like a live credential {_preview(value)}")


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = str(text)
    return repr(text if len(text) <= 24 else text[:21] + "...")


# --------------------------------------------------------------------------- #
# Combined result.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReportValidation:
    """Outcome of validating a training loader report against the v1 contract."""

    schema_id: str = TRAINING_LOADER_REPORT_SCHEMA_ID
    schema_errors: list[str] = field(default_factory=list)
    secret_findings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.schema_errors and not self.secret_findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "ok": self.ok,
            "schema_errors": list(self.schema_errors),
            "secret_findings": list(self.secret_findings),
        }

    def raise_for_status(self) -> None:
        """Raise :class:`ReportValidationError` if the report is not conformant."""
        if not self.ok:
            raise ReportValidationError(self)


class ReportValidationError(ValueError):
    """A training loader report failed schema or redaction conformance."""

    def __init__(self, validation: ReportValidation) -> None:
        self.validation = validation
        problems = validation.schema_errors + validation.secret_findings
        summary = "; ".join(problems[:5])
        if len(problems) > 5:
            summary += f"; (+{len(problems) - 5} more)"
        super().__init__(
            f"training loader report is not conformant to "
            f"{validation.schema_id}: {summary}"
        )


def validate_training_loader_report(payload: Any) -> ReportValidation:
    """Validate a report payload against the v1 schema *and* redaction contract."""
    resolved = payload.to_dict() if hasattr(payload, "to_dict") else payload
    return ReportValidation(
        schema_errors=validate_report_schema(resolved),
        secret_findings=scan_report_secrets(resolved),
    )


def load_report_json(path: str | Path) -> Any:
    """Read and JSON-decode a report file, raising a clear error on bad JSON."""
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - message only
        raise ReportValidationError(
            ReportValidation(schema_errors=[f"invalid JSON: {exc}"])
        ) from exc
