"""Redaction policy for external lineage context and evidence-pack manifests (0114).

Backlog 0068 stores external run/job/code/environment context in manifest and
transform JSON fields, and 0113 added redacted worker propagation. This module
turns redaction into a *declarative, reusable policy* so the same rules apply
before an evidence pack or export is materialized and when an external-context
catalog row is stored.

A :class:`ContextRedactionPolicy` combines three controls:

- an **allowlist** of top-level context keys (when set, everything else is dropped);
- a **denylist** of case-insensitive key fragments (``secret``/``token``/... by
  default) -- matching keys are dropped from mappings and from
  ``list<struct<key,value>>`` metadata;
- **secret-value patterns** (opt-in regexes) that mask a value in place when it
  looks like a credential, even under an allowed key.

The policy is dependency-free and deterministic. It never mutates its input: it
returns redacted copies, so a caller can redact a manifest before hashing it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Default case-insensitive key fragments treated as sensitive and dropped.
#: Kept in sync with ``lineage_hooks.DEFAULT_WORKER_REDACTED_KEY_FRAGMENTS``.
DEFAULT_DENY_KEY_FRAGMENTS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "credential",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "auth",
)

#: Conservative, opt-in value patterns for common credential shapes. Off by
#: default so ordinary run IDs / URLs are never masked by accident.
DEFAULT_SECRET_VALUE_PATTERNS: tuple[str, ...] = (
    r"AKIA[0-9A-Z]{16}",  # AWS access key id
    r"ASIA[0-9A-Z]{16}",  # AWS temporary access key id
    r"gh[pousr]_[0-9A-Za-z]{20,}",  # GitHub tokens
    r"xox[baprs]-[0-9A-Za-z-]{10,}",  # Slack tokens
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",  # PEM private keys
    r"eyJ[0-9A-Za-z_-]{10,}\.[0-9A-Za-z_-]{10,}\.[0-9A-Za-z_-]{10,}",  # JWTs
)

DEFAULT_REDACTION_MARKER = "[redacted]"

#: JSON-encoded string columns whose values carry context and must be
#: parsed, redacted, and re-encoded inside an evidence-pack manifest row.
MANIFEST_JSON_FIELDS: tuple[str, ...] = (
    "params",
    "params_json",
    "environment_json",
    "runtime_json",
)

#: ``list<struct<key,value>>`` columns whose entries carry context.
MANIFEST_KV_FIELDS: tuple[str, ...] = ("external_refs", "metadata")

#: Plain nested dict fields that may appear in a manifest row.
MANIFEST_DICT_FIELDS: tuple[str, ...] = ("environment", "runtime", "lineage_context")

#: Manifest sections whose entries are per-table rows to walk.
_MANIFEST_ROW_SECTIONS: tuple[str, ...] = (
    "transform_runs",
    "lineage_executions",
    "model_artifacts",
    "model_outputs",
)


class RedactionError(ValueError):
    """Raised when a redaction policy specification is invalid."""


@dataclass(frozen=True)
class ContextRedactionPolicy:
    """Declarative redaction rules for lineage context and manifests."""

    name: str = "default"
    allow_keys: tuple[str, ...] = ()
    deny_key_fragments: tuple[str, ...] = DEFAULT_DENY_KEY_FRAGMENTS
    secret_value_patterns: tuple[str, ...] = ()
    redaction_marker: str = DEFAULT_REDACTION_MARKER
    _compiled: tuple[re.Pattern[str], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        try:
            compiled = tuple(re.compile(pattern) for pattern in self.secret_value_patterns)
        except re.error as exc:  # pragma: no cover - defensive
            raise RedactionError(f"invalid secret value pattern: {exc}") from exc
        object.__setattr__(self, "_compiled", compiled)

    # --- Construction --------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any] | None) -> ContextRedactionPolicy | None:
        """Build a policy from a mapping (CLI/JSON), or ``None`` for no redaction."""

        if not spec:
            return None
        allow = _as_str_tuple(spec.get("allow_keys"))
        deny = spec.get("deny_key_fragments")
        secrets = spec.get("secret_value_patterns")
        use_default_secrets = bool(spec.get("detect_secrets"))
        return cls(
            name=str(spec.get("name") or "custom"),
            allow_keys=allow,
            deny_key_fragments=(
                _as_str_tuple(deny) if deny is not None else DEFAULT_DENY_KEY_FRAGMENTS
            ),
            secret_value_patterns=(
                DEFAULT_SECRET_VALUE_PATTERNS
                if use_default_secrets
                else _as_str_tuple(secrets)
            ),
            redaction_marker=str(spec.get("redaction_marker") or DEFAULT_REDACTION_MARKER),
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-ready summary suitable for audit metadata."""

        return {
            "name": self.name,
            "allow_keys": list(self.allow_keys),
            "deny_key_fragments": list(self.deny_key_fragments),
            "secret_value_patterns": list(self.secret_value_patterns),
            "digest": self.digest(),
        }

    def digest(self) -> str:
        """Short stable digest identifying this policy's rules."""

        encoded = json.dumps(
            {
                "allow_keys": sorted(self.allow_keys),
                "deny_key_fragments": sorted(self.deny_key_fragments),
                "secret_value_patterns": sorted(self.secret_value_patterns),
                "redaction_marker": self.redaction_marker,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()[:12]

    # --- Core redaction ------------------------------------------------------

    def _is_denied(self, key: str) -> bool:
        lowered = str(key).lower()
        return any(fragment and fragment in lowered for fragment in self.deny_key_fragments)

    def _mask_if_secret(self, value: Any) -> Any:
        if not self._compiled or not isinstance(value, str):
            return value
        if any(pattern.search(value) for pattern in self._compiled):
            return self.redaction_marker
        return value

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                if self._is_denied(str(key)):
                    continue
                redacted[str(key)] = self._redact_value(item)
            return redacted
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._redact_value(item) for item in value]
        return self._mask_if_secret(value)

    def redact_context(self, context: Mapping[str, Any] | None) -> dict[str, Any]:
        """Redact a lineage-context payload, applying the allowlist at the top level."""

        if not context:
            return {}
        working: dict[str, Any] = {}
        for key, value in context.items():
            if self.allow_keys and str(key) not in self.allow_keys:
                continue
            if self._is_denied(str(key)):
                continue
            working[str(key)] = self._redact_value(value)
        return working

    def redacts(self, context: Mapping[str, Any] | None) -> bool:
        """Return True if redacting ``context`` would change it."""

        if not context:
            return False
        return self.redact_context(context) != {str(k): v for k, v in context.items()}

    # --- KV-list + JSON-string helpers --------------------------------------

    def _redact_kv_list(self, entries: Any) -> list[Any]:
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            return entries
        out: list[Any] = []
        for entry in entries:
            if isinstance(entry, Mapping) and "key" in entry:
                key = str(entry.get("key"))
                if self._is_denied(key):
                    continue
                new_entry = dict(entry)
                if "value" in new_entry:
                    new_entry["value"] = self._mask_if_secret(new_entry["value"])
                out.append(new_entry)
            else:
                out.append(self._redact_value(entry))
        return out

    def _redact_json_string(self, raw: Any) -> Any:
        if not isinstance(raw, str) or not raw.strip():
            return raw
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return self._mask_if_secret(raw)
        redacted = self._redact_value(parsed)
        return json.dumps(redacted, sort_keys=True, separators=(",", ":"))

    def _redact_row(self, row: Any) -> Any:
        if not isinstance(row, Mapping):
            return row
        out = dict(row)
        for key in list(out.keys()):
            if key in MANIFEST_JSON_FIELDS:
                out[key] = self._redact_json_string(out[key])
            elif key in MANIFEST_KV_FIELDS:
                out[key] = self._redact_kv_list(out[key])
            elif key in MANIFEST_DICT_FIELDS and isinstance(out[key], Mapping):
                out[key] = self._redact_value(out[key])
        return out

    def redact_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Return a deep copy of an evidence-pack manifest with context redacted.

        Walks the per-table ``rows`` map and the top-level ``transform_runs`` /
        ``lineage_executions`` / ``model_artifacts`` / ``model_outputs`` /
        ``training_run`` sections, redacting the JSON-string, KV-list, and nested
        dict fields that carry external context or environment metadata. Blob
        payload bytes are never touched -- only manifest metadata.
        """

        out = copy.deepcopy(dict(manifest))
        rows = out.get("rows")
        if isinstance(rows, Mapping):
            out["rows"] = {
                str(table): [self._redact_row(row) for row in (table_rows or [])]
                for table, table_rows in rows.items()
            }
        for section in _MANIFEST_ROW_SECTIONS:
            value = out.get(section)
            if isinstance(value, list):
                out[section] = [self._redact_row(row) for row in value]
        training_run = out.get("training_run")
        if isinstance(training_run, Mapping):
            out["training_run"] = self._redact_row(training_run)
        return out


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    raise RedactionError(f"expected a string or list of strings, got {type(value).__name__}")
