"""Schema + redaction conformance for the training loader report (backlog 0124).

Proves the v1 report contract: native and aligned reports validate against the
committed schema, the schema JSON never drifts from the module, secret-shaped
fields are removed recursively while safe display/auth-reference fields survive,
and the validator explains schema and redaction failures.
"""

import json

import pytest
import test_aligned_training_dataset as aligned_mod
import test_native_training_dataset as native_mod

from lancedb_robotics.training import TrainingLoaderReport
from lancedb_robotics.training_report_schema import (
    REPORT_REDACTION_MARKER,
    SCHEMA_JSON_PATH,
    TRAINING_LOADER_REPORT_SCHEMA,
    TRAINING_LOADER_REPORT_SCHEMA_ID,
    ReportValidation,
    ReportValidationError,
    is_secret_report_key,
    is_secret_report_value,
    scan_report_secrets,
    validate_report_schema,
    validate_training_loader_report,
)


def _native_report(tmp_path, **loader_report_kwargs):
    lake = native_mod._training_lake(tmp_path / "native.lance")
    dataset = lake.training.dataset("demo-v1", columns=["observation_id", "state_vector"])
    return dataset.loader_report(**loader_report_kwargs).to_dict()


def _aligned_report(tmp_path):
    lake, _view = aligned_mod._aligned_training_lake(tmp_path / "aligned.lance")
    dataset = lake.training.aligned_dataset(name="policy_bridge")
    return dataset.loader_report().to_dict()


# --- Schema conformance ------------------------------------------------------


def test_native_report_validates_against_schema(tmp_path):
    report = _native_report(tmp_path)
    assert report["kind"] == TRAINING_LOADER_REPORT_SCHEMA_ID
    assert validate_report_schema(report) == []
    validation = validate_training_loader_report(report)
    assert validation.ok, validation.to_dict()


def test_aligned_report_validates_against_schema(tmp_path):
    report = _aligned_report(tmp_path)
    assert report["loader"]["kind"] == "aligned-training"
    assert validate_report_schema(report) == []
    assert validate_training_loader_report(report).ok


def test_committed_schema_json_matches_module():
    # The committed JSON contract must never drift from the source-of-truth dict.
    assert SCHEMA_JSON_PATH.exists(), f"missing committed schema at {SCHEMA_JSON_PATH}"
    on_disk = SCHEMA_JSON_PATH.read_text()
    expected = json.dumps(TRAINING_LOADER_REPORT_SCHEMA, indent=2, sort_keys=True) + "\n"
    assert on_disk == expected, (
        "docs/manual/reference/training_loader_report.v1.schema.json is stale; "
        "regenerate it from TRAINING_LOADER_REPORT_SCHEMA."
    )
    assert json.loads(on_disk)["$id"] == TRAINING_LOADER_REPORT_SCHEMA_ID


def test_schema_rejects_wrong_kind(tmp_path):
    report = _native_report(tmp_path)
    report["kind"] = "lancedb-robotics/training-loader-report/v2"
    errors = validate_report_schema(report)
    assert any("const" in e and "kind" in e for e in errors), errors


def test_schema_rejects_missing_required_field(tmp_path):
    report = _native_report(tmp_path)
    del report["metrics"]
    errors = validate_report_schema(report)
    assert any("missing required property 'metrics'" in e for e in errors), errors


def test_schema_rejects_wrong_type(tmp_path):
    report = _native_report(tmp_path)
    report["plans"]["worker"]["id"] = "zero"  # must be integer
    errors = validate_report_schema(report)
    assert any("worker.id" in e and "integer" in e for e in errors), errors


def test_native_report_requires_snapshot(tmp_path):
    report = _native_report(tmp_path)
    del report["snapshot"]
    errors = validate_report_schema(report)
    assert any("missing required property 'snapshot'" in e for e in errors), errors


def test_aligned_report_requires_alignment_and_read_table_versions(tmp_path):
    report = _aligned_report(tmp_path)
    del report["alignment"]
    del report["read_table_versions"]
    errors = validate_report_schema(report)
    assert any("'alignment'" in e for e in errors), errors
    assert any("'read_table_versions'" in e for e in errors), errors


def test_unknown_fields_are_allowed_for_forward_compat(tmp_path):
    # v1 validation is open: a future minor revision may add fields.
    report = _native_report(tmp_path)
    report["metrics"]["summary"]["a_future_metric"] = 123
    report["a_future_top_level_section"] = {"anything": True}
    assert validate_report_schema(report) == []


# --- Redaction conformance ---------------------------------------------------


def test_emitter_redacts_nested_credentials_recursively(tmp_path):
    # The real emitter path: secret-shaped keys anywhere under `run.extra` are
    # replaced with the marker; the conformance scanner then finds nothing.
    report = _native_report(
        tmp_path,
        training_run_id="train-1",
        extra={
            "tracker": "wandb",
            "authorization": "Bearer sk-live-abc",
            "vault": {
                "api_key": "live-key",
                "session_token": "sts-scoped-token",
                "keep_me": "not-a-secret",
            },
            "object_store": {"aws_secret_access_key": "wJalrXUtnFEMI/EXAMPLEKEY"},
            "namespace_credential": "vended-token",
            "remote_auth_ref": "vault://team/lake",
        },
    )
    run = report["run"]
    assert run["tracker"] == "wandb"
    assert run["authorization"] == REPORT_REDACTION_MARKER
    assert run["vault"]["api_key"] == REPORT_REDACTION_MARKER
    assert run["vault"]["session_token"] == REPORT_REDACTION_MARKER
    assert run["vault"]["keep_me"] == "not-a-secret"
    assert run["object_store"]["aws_secret_access_key"] == REPORT_REDACTION_MARKER
    assert run["namespace_credential"] == REPORT_REDACTION_MARKER
    # Safe auth *reference* survives.
    assert run["remote_auth_ref"] == "vault://team/lake"

    encoded = json.dumps(report)
    for secret in ("sk-live-abc", "live-key", "sts-scoped-token", "wJalrXUtnFEMI", "vended-token"):
        assert secret not in encoded

    # A redacted report is conformant.
    validation = validate_training_loader_report(report)
    assert validation.ok, validation.secret_findings


def test_scanner_flags_secret_key_and_preserves_auth_ref():
    payload = {
        "run": {
            "authorization": "Bearer x",
            "nested": {"api_key": "live", "access_key": "AKIAIOSFODNN7EXAMPLE"},
            "items": [{"session_token": "sts"}, {"password": "hunter2"}],
            "remote_auth_ref": "vault://p",
            "namespace_auth_ref": "profile://n",
            "display_uri": "s3://bucket/lake",
        }
    }
    findings = scan_report_secrets(payload)
    blob = "\n".join(findings)
    for expected_key in (
        "run.authorization",
        "run.nested.api_key",
        "run.nested.access_key",
        "run.items[0].session_token",
        "run.items[1].password",
    ):
        assert expected_key in blob, f"{expected_key} not flagged: {findings}"
    # Safe references and display URIs must never be flagged.
    assert "auth_ref" not in blob
    assert "display_uri" not in blob


def test_scanner_detects_secret_value_under_innocuous_key():
    assert scan_report_secrets({"note": "AKIAIOSFODNN7EXAMPLE"})
    assert scan_report_secrets({"note": "Bearer sk-live-123"})
    assert scan_report_secrets({"blob": "-----BEGIN RSA PRIVATE KEY-----abc"})
    # An ordinary id/uri is not a secret.
    assert scan_report_secrets({"display_uri": "s3://bucket/x", "id": "run-123"}) == []


def test_scanner_treats_redaction_marker_as_safe():
    assert scan_report_secrets({"api_key": REPORT_REDACTION_MARKER}) == []


def test_secret_predicates_directly():
    assert is_secret_report_key("Authorization")
    assert is_secret_report_key("aws_secret_access_key")
    assert is_secret_report_key("namespace_credential")
    assert is_secret_report_key("x-api-key")
    assert not is_secret_report_key("remote_auth_ref")
    assert not is_secret_report_key("display_uri")
    assert is_secret_report_value("Bearer abc")
    assert is_secret_report_value("basic dXNlcjpwYXNz")
    assert not is_secret_report_value("s3://bucket/path")


# --- Combined validation surface --------------------------------------------


def test_validate_accepts_report_object_mapping_and_to_dict(tmp_path):
    report = _native_report(tmp_path)
    wrapped = TrainingLoaderReport(report)

    class _HasToDict:
        def to_dict(self):
            return report

    assert validate_training_loader_report(report).ok
    assert validate_training_loader_report(wrapped).ok
    assert validate_training_loader_report(_HasToDict()).ok


def test_raise_for_status_raises_on_bad_report(tmp_path):
    report = _native_report(tmp_path)
    report["run"]["authorization"] = "Bearer live"
    del report["metrics"]
    validation = validate_training_loader_report(report)
    assert not validation.ok
    assert validation.schema_errors and validation.secret_findings
    with pytest.raises(ReportValidationError) as excinfo:
        validation.raise_for_status()
    assert TRAINING_LOADER_REPORT_SCHEMA_ID in str(excinfo.value)


def test_validate_report_via_lake_namespace(tmp_path):
    lake = native_mod._training_lake(tmp_path / "native.lance")
    dataset = lake.training.dataset("demo-v1", columns=["observation_id"])
    validation = lake.training.validate_report(dataset.loader_report())
    assert isinstance(validation, ReportValidation)
    assert validation.ok
