"""ContextRedactionPolicy unit tests (backlog 0114)."""

import json

from lancedb_robotics.redaction import (
    DEFAULT_SECRET_VALUE_PATTERNS,
    ContextRedactionPolicy,
)


def test_denylist_drops_sensitive_keys_recursively():
    policy = ContextRedactionPolicy()
    context = {
        "provider": "wandb",
        "api_key": "AKIA-not-real",
        "environment": {"python": "3.14", "auth_token": "abc", "digest": "sha256:x"},
    }
    redacted = policy.redact_context(context)
    assert redacted["provider"] == "wandb"
    assert "api_key" not in redacted
    assert "auth_token" not in redacted["environment"]
    assert redacted["environment"] == {"python": "3.14", "digest": "sha256:x"}


def test_allowlist_keeps_only_named_top_level_keys():
    policy = ContextRedactionPolicy(name="allow", allow_keys=("provider", "external_run_id"))
    context = {
        "provider": "mlflow",
        "external_run_id": "run-1",
        "external_url": "https://mlflow/run-1",
        "environment": {"python": "3.14"},
    }
    redacted = policy.redact_context(context)
    assert set(redacted) == {"provider", "external_run_id"}


def test_secret_value_patterns_mask_in_place_when_enabled():
    policy = ContextRedactionPolicy(
        name="secrets",
        deny_key_fragments=(),
        secret_value_patterns=DEFAULT_SECRET_VALUE_PATTERNS,
    )
    context = {"provider": "ci", "note": "deploy key AKIA1234567890ABCD99 leaked"}
    redacted = policy.redact_context(context)
    assert "AKIA" not in redacted["note"]
    assert redacted["note"] == "[redacted]"


def test_secret_patterns_off_by_default_leave_ids_untouched():
    policy = ContextRedactionPolicy()
    context = {"provider": "wandb", "external_run_id": "AKIA1234567890ABCD99"}
    redacted = policy.redact_context(context)
    # No secret patterns configured: ordinary-looking ids are preserved.
    assert redacted["external_run_id"] == "AKIA1234567890ABCD99"


def test_redact_manifest_walks_json_string_and_kv_fields():
    policy = ContextRedactionPolicy()
    manifest = {
        "schema_version": "lancedb-robotics/evidence-pack/v1",
        "rows": {
            "transform_runs": [
                {
                    "transform_id": "t1",
                    "params": json.dumps(
                        {"lineage_context": {"provider": "wandb", "wandb_api_key": "secret-xyz"}}
                    ),
                }
            ],
            "training_runs": [
                {
                    "training_run_id": "tr1",
                    "environment_json": json.dumps({"python": "3.14", "hf_token": "t0ken"}),
                    "external_refs": [
                        {"key": "external_run_id", "value": "run-1"},
                        {"key": "wandb_api_key", "value": "secret-xyz"},
                    ],
                }
            ],
        },
        "lineage_executions": [
            {"execution_id": "e1", "params_json": json.dumps({"password": "hunter2", "provider": "mlflow"})}
        ],
    }
    redacted = policy.redact_manifest(manifest)
    blob = json.dumps(redacted)
    assert "secret-xyz" not in blob
    assert "hunter2" not in blob
    assert "t0ken" not in blob
    # Non-sensitive values survive.
    assert "run-1" in blob
    assert "3.14" in blob
    # Original manifest is not mutated.
    assert "secret-xyz" in json.dumps(manifest)


def test_from_spec_and_digest_are_deterministic():
    spec = {"name": "p", "deny_key_fragments": ["secret", "token"], "detect_secrets": True}
    a = ContextRedactionPolicy.from_spec(spec)
    b = ContextRedactionPolicy.from_spec(spec)
    assert a is not None and b is not None
    assert a.digest() == b.digest()
    assert a.secret_value_patterns == DEFAULT_SECRET_VALUE_PATTERNS
    assert ContextRedactionPolicy.from_spec(None) is None
    assert ContextRedactionPolicy.from_spec({}) is None


def test_redacts_reports_change():
    policy = ContextRedactionPolicy()
    assert policy.redacts({"provider": "wandb", "api_key": "x"}) is True
    assert policy.redacts({"provider": "wandb"}) is False
