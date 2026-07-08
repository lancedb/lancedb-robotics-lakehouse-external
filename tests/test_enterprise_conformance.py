"""Enterprise training conformance and fault-injection tests (backlog 0116).

Exercises the harness in ``lancedb_robotics.enterprise_conformance``:

* the static compatibility matrix classifies every backend scenario and fault;
* a live run replays one snapshot through each case and enforces the central
  invariant -- every degradation is a typed error or an explicit fallback,
  never silent local materialization;
* worker handoff keeps ``host_override`` routing intact without leaking API keys;
* local and (faked) Enterprise paths emit equivalent sample/row/table-version
  lineage and cover the same epoch exactly once across worker partitions.
"""

from __future__ import annotations

import pytest

# Reuse the local training-lake builder + enterprise faker from the sibling suite.
from test_native_training_dataset import _mark_enterprise_lake, _training_lake

from lancedb_robotics import enterprise_conformance as ec
from lancedb_robotics.training import (
    NamespaceCredentialExpiredError,
    StaleTableVersionError,
    UnsupportedRemoteOperationError,
)


@pytest.fixture
def lake(tmp_path):
    return _training_lake(tmp_path / "robot.lance")


# --------------------------------------------------------------------------- #
# Static compatibility matrix
# --------------------------------------------------------------------------- #


def test_compatibility_matrix_classifies_every_case():
    matrix = ec.compatibility_matrix()
    by_name = {row.name: row for row in matrix.rows}

    # Backend scenarios.
    assert by_name["local-native"].category == ec.CATEGORY_SUPPORTED
    assert by_name["local-native"].resolved_backend == "local"
    assert by_name["db-remote-host-override"].category == ec.CATEGORY_SUPPORTED
    assert by_name["db-remote-host-override"].routing_mode == "host-override"
    assert by_name["db-remote-regional-default"].routing_mode == "regional-default"
    assert by_name["rest-namespace-query-node"].category == ec.CATEGORY_SUPPORTED
    assert by_name["capability-disabled-deployment"].category == ec.CATEGORY_UNSUPPORTED
    assert (
        by_name["capability-disabled-deployment"].error_type
        == "UnsupportedRemoteOperationError"
    )

    # Faults map to typed errors or explicit fallbacks.
    assert by_name["auth-missing"].error_type == "MissingEnterpriseAuthError"
    assert by_name["auth-expired-no-refresh"].error_type == "NamespaceCredentialExpiredError"
    assert by_name["auth-expired-refresh-fails"].error_type == "NamespaceCredentialExpiredError"
    assert by_name["auth-expired-refresh-recovers"].category == ec.CATEGORY_SUPPORTED
    assert by_name["remote-scan-unsupported"].error_type == "UnsupportedRemoteOperationError"
    assert by_name["prewarm-unavailable-warn"].category == ec.CATEGORY_FALLBACK
    assert by_name["prewarm-unavailable-warn"].fallback_to == "lazy-cache"
    assert by_name["direct-data-plane-fallback"].fallback_to == "direct-data-plane"
    assert by_name["explicit-local-fallback"].fallback_to == "local"

    # Build-time faults report their enforced contract with a note.
    assert by_name["stale-table-version"].error_type == "StaleTableVersionError"
    assert "surfaces at loader build" in by_name["stale-table-version"].note
    assert by_name["worker-resume-mismatch"].error_type == "WorkerResumeMismatchError"

    counts = matrix.to_dict()["summary"]
    assert counts["supported"] + counts["fallback"] + counts["unsupported"] == len(matrix.rows)
    # No silent-local column exists -- degradations are typed or explicit.
    assert set(counts) == {"supported", "fallback", "unsupported"}


def test_compatibility_matrix_markdown_is_deterministic():
    first = ec.compatibility_matrix().to_markdown()
    second = ec.compatibility_matrix().to_markdown()
    assert first == second
    assert "# Enterprise training compatibility matrix" in first
    assert "## Backend scenarios" in first
    assert "## Injected faults" in first
    assert "silent" in first  # documents that silent materialization is not a cell


def test_local_endpoint_row_is_opt_in():
    default_names = {row.name for row in ec.compatibility_matrix().rows}
    assert "local-enterprise-cli-endpoint" not in default_names
    with_endpoint = {
        row.name for row in ec.compatibility_matrix(include_local_endpoint=True).rows
    }
    assert "local-enterprise-cli-endpoint" in with_endpoint


# --------------------------------------------------------------------------- #
# Live conformance run
# --------------------------------------------------------------------------- #


def test_run_conformance_all_cases_pass(lake):
    report = lake.training.run_conformance("demo-v1")
    failures = report.failures()
    assert not failures, [(o.name, o.failures) for o in failures]
    assert report.ok()
    summary = report.summary()
    assert summary["failed"] == 0
    assert summary["passed"] == summary["total"]


def test_run_conformance_strict_passes_on_clean_lake(lake):
    report = lake.training.run_conformance("demo-v1", strict=True)
    assert report.ok()


def test_run_conformance_never_silently_materializes_local(lake):
    report = lake.training.run_conformance("demo-v1")
    for outcome in report.outcomes:
        if outcome.requested_backend != "enterprise":
            continue
        if outcome.resolved_backend == "local":
            # A local resolution for an enterprise request must be an explicit
            # fallback, not a silent downgrade.
            assert outcome.category == ec.CATEGORY_FALLBACK
            assert outcome.fallback_to is not None


# --------------------------------------------------------------------------- #
# Test-First Plan failing tests (now green)
# --------------------------------------------------------------------------- #


def test_fake_remote_scan_disabled_raises_typed_error_before_iteration(lake):
    _mark_enterprise_lake(lake)
    lake.enterprise_training_capabilities = {"remote_scan": False}
    with pytest.raises(UnsupportedRemoteOperationError, match="remote_scan"):
        lake.training.dataset("demo-v1", columns=["observation_id"], backend="enterprise")


def test_expired_scoped_credentials_refresh_once_then_fail(lake):
    _mark_enterprise_lake(lake, uri="namespace://robotics")
    # Present a namespace connection with expired creds and a refresh hook that
    # does not clear the expiry: exactly one refresh, then a targeted diagnostic.
    calls = []

    def refresh():
        calls.append(1)  # refresh attempted but credentials remain expired

    lake.namespace_credentials_expired = True
    lake.namespace_credential_refresh = refresh
    with pytest.raises(NamespaceCredentialExpiredError, match="single automatic credential refresh"):
        lake.training.dataset("demo-v1", columns=["observation_id"], backend="enterprise")
    assert calls == [1]


def test_expired_scoped_credentials_recover_after_single_refresh(lake):
    _mark_enterprise_lake(lake)

    def refresh():
        lake.namespace_credentials_expired = False

    lake.namespace_credentials_expired = True
    lake.namespace_credential_refresh = refresh
    dataset = lake.training.dataset("demo-v1", columns=["observation_id"], backend="enterprise")
    assert dataset.backend_report.resolved_backend == "enterprise"


def test_stale_table_version_reported_as_backend_error(lake, monkeypatch):
    _mark_enterprise_lake(lake)
    original = lake.table

    class _Stale:
        def __init__(self, table):
            self._table = table

        def checkout(self, version):
            raise RuntimeError(f"version {version} compacted away")

        def __getattr__(self, name):
            return getattr(self._table, name)

    def table(name):
        opened = original(name)
        return _Stale(opened) if name == "scenarios" else opened

    monkeypatch.setattr(lake, "table", table)
    with pytest.raises(StaleTableVersionError, match="recreate the training snapshot"):
        lake.training.dataset("demo-v1", columns=["observation_id"], backend="enterprise")


def test_local_and_fake_remote_partitions_cover_epoch_exactly_once(lake):
    num_workers = 2

    def partition(backend, worker_id):
        if backend == "enterprise":
            _mark_enterprise_lake(lake)
        else:
            lake.connection_spec = None
            lake.capabilities = None
        dataset = lake.training.dataset(
            "demo-v1",
            columns=["observation_id"],
            backend=backend,
            worker_id=worker_id,
            num_workers=num_workers,
        )
        return list(dataset.epoch_plan.sample_indices)

    local_union = sorted(sum((partition("local", w) for w in range(num_workers)), []))
    remote_union = sorted(sum((partition("enterprise", w) for w in range(num_workers)), []))

    total_frames = 3
    assert local_union == list(range(total_frames))  # covered exactly once
    assert remote_union == local_union  # backend does not perturb partitioning


# --------------------------------------------------------------------------- #
# Worker handoff secret hygiene + routing
# --------------------------------------------------------------------------- #


def test_worker_handoff_preserves_host_override_without_secrets(lake):
    report = lake.training.run_conformance("demo-v1")
    host_override_case = next(
        o for o in report.outcomes if o.name == "db-remote-host-override"
    )
    assert host_override_case.status == "pass"
    assert host_override_case.routing_mode == "host-override"
    assert host_override_case.checks.get("secret_leaks") == []


def test_check_no_secret_leak_flags_literal_token():
    leaks = ec._check_no_secret_leak({"connection": {"api_key": "conformance-test-key"}})
    assert "literal-api-token" in leaks
    clean = ec._check_no_secret_leak(
        {"connection": {"remote": {"remote_auth_ref": "enterprise-prod", "host_override": "http://x"}}}
    )
    assert clean == []


# --------------------------------------------------------------------------- #
# Local Enterprise CLI endpoint (gated)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not ec.local_endpoint_available(),
    reason=f"set {ec.LOCAL_ENDPOINT_ENV}=1 with the LanceDB Enterprise CLI present",
)
def test_local_enterprise_endpoint_routes_via_host_override(lake):
    report = lake.training.run_conformance("demo-v1", include_local_endpoint=True)
    endpoint_case = next(
        o for o in report.outcomes if o.name == "local-enterprise-cli-endpoint"
    )
    assert endpoint_case.status == "pass"
    assert endpoint_case.routing_mode == "host-override"
    assert report.local_endpoint is not None
    assert report.local_endpoint["server_running"] in {True, False}
