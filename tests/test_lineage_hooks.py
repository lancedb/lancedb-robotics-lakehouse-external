"""Lineage hook plugin conformance and scoped worker context tests."""

import json

import pytest

from lancedb_robotics.lineage_hooks import (
    LINEAGE_CONTEXT_ENV,
    LineageContext,
    LineageHookConformanceError,
    LineageHookError,
    assert_lineage_hook_conformance,
    check_lineage_hook_conformance,
    current_lineage_context,
    lineage_context_env_for_worker,
    lineage_context_for_worker,
    lineage_context_scope,
    require_lineage_hook_adapter,
)


class _DagsterAssetHook:
    def before_execution(self, operation, params=None):
        return {
            "provider": "dagster",
            "external_run_id": "dagster-run-0113",
            "external_job_id": "asset-job",
            "external_refs": {
                "dagster_asset_key": "robotics/demo",
                "operation": operation,
            },
        }

    def after_execution(self, operation, context, *, status, error=None):
        return {"status": status, "external_refs": {"dagster_status": status}}


class _MutatingHook:
    def before_execution(self, operation, params=None):
        return LineageContext(
            provider="ray",
            external_run_id="ray-run-0113",
            external_refs={"ray_job_id": "job-0113"},
        )

    def after_execution(self, operation, context, *, status, error=None):
        context.external_refs["mutated"] = "after-return"
        return context


def test_custom_hook_conformance_runs_without_optional_dependencies():
    report = check_lineage_hook_conformance(_DagsterAssetHook())

    assert report.passed is True
    assert report.before_context["provider"] == "dagster"
    assert report.after_context["status"] == "completed"
    assert report.after_context["external_refs"]["dagster_status"] == "completed"


def test_hook_conformance_catches_context_mutation():
    report = check_lineage_hook_conformance(_MutatingHook())

    assert report.passed is False
    assert "context-mutated" in {issue.code for issue in report.issues}
    with pytest.raises(LineageHookConformanceError, match="context-mutated"):
        assert_lineage_hook_conformance(_MutatingHook())


def test_worker_context_propagation_is_explicit_and_redacted():
    assert current_lineage_context() is None

    with lineage_context_scope(
        {
            "provider": "airflow",
            "external_run_id": "dag-run-0113",
            "environment": {
                "cluster": "research-gpu",
                "AWS_SECRET_ACCESS_KEY": "do-not-propagate",
            },
            "external_refs": {
                "task_instance": "train[2026-07-04]",
                "secret_token": "do-not-propagate",
            },
        }
    ):
        payload = lineage_context_for_worker()
        env = lineage_context_env_for_worker()

    assert payload["provider"] == "airflow"
    assert payload["external_run_id"] == "dag-run-0113"
    assert payload["environment"] == {"cluster": "research-gpu"}
    assert payload["external_refs"] == {"task_instance": "train[2026-07-04]"}
    assert json.loads(env[LINEAGE_CONTEXT_ENV]) == payload
    assert current_lineage_context() is None


def test_missing_hook_adapter_names_extra_and_module():
    with pytest.raises(
        LineageHookError,
        match=r"optional extra/plugin 'definitely-missing-0113'.*module 'definitely_missing_0113'",
    ):
        require_lineage_hook_adapter("definitely-missing-0113")
