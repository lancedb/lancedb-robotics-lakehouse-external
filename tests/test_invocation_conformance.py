"""Tests for the 0076-audit invocation conformance suite (backlog 0130).

Follow-up to the static 0076 invocation audit and the 0128 capability gates /
0129 namespace-direct adapter. The suite turns the audit's four support classes
into executable proof: it samples representative rows per class and probes them
against a live or contract-test backend, then reports supported / unsupported /
skipped per row with a concrete reason linked back to the audit row id.

The default (fast) tests run the contract-test mode, which needs no live
credentials and no optional dependencies -- the namespace credential vending and
managed-versioning fixtures are fakes injected through the 0129 access factory.
The dedicated CI lane sets ``LANCEDB_ROBOTICS_REQUIRE_INVOCATION_CONFORMANCE=1``
so a contract-test regression fails rather than skips.
"""

from __future__ import annotations

import json

import pytest

from lancedb_robotics import invocation_conformance as ic

# The whole module is the invocation-conformance lane. Contract-test mode needs
# no optional dependency, so these also run in the default suite; the dedicated
# CI lane selects them with ``-m invocation_conformance``.
pytestmark = pytest.mark.invocation_conformance


# --------------------------------------------------------------------------- #
# Sampling: deterministic, covers all four support classes, links real ids
# --------------------------------------------------------------------------- #


def test_audit_rows_load_and_cover_four_support_classes() -> None:
    rows = ic.load_audit_rows()
    classes = {row["support_class"] for row in rows}
    assert classes == set(ic.SUPPORT_CLASSES)


def test_sampling_is_deterministic_and_bounded_per_class() -> None:
    first = ic.sample_audit(per_class=4)
    second = ic.sample_audit(per_class=4)
    # Deterministic: identical audit ids in identical order across calls.
    assert [s.audit_id for s in first] == [s.audit_id for s in second]
    # Bounded: at most per_class per support class, and every class represented.
    by_class: dict[str, int] = {}
    for sample in first:
        by_class[sample.support_class] = by_class.get(sample.support_class, 0) + 1
    for support_class in ic.SUPPORT_CLASSES:
        assert 1 <= by_class[support_class] <= 4


def test_samples_reference_real_audit_ids_and_gated_families() -> None:
    rows_by_id = {row["id"]: row for row in ic.load_audit_rows()}
    for sample in ic.sample_audit(per_class=3):
        assert sample.audit_id in rows_by_id
        # gated_families is exactly the row families that map to a 0128 gate.
        assert set(sample.gated_families) <= set(sample.families)
        assert set(sample.gated_families) <= ic.GATED_FAMILIES


# --------------------------------------------------------------------------- #
# Report schema (test-first plan #1): backend, audit ids, capability status,
# skip reason are all recorded.
# --------------------------------------------------------------------------- #


def test_report_schema_records_backend_ids_status_and_reasons() -> None:
    report = ic.run_conformance(mode=ic.MODE_CONTRACT_TEST, per_class=3)
    payload = json.loads(report.to_json())

    assert payload["schema"] == ic.REPORT_SCHEMA
    assert payload["mode"] == ic.MODE_CONTRACT_TEST
    # Sampled vs total per class is reported (no silent truncation).
    assert set(payload["per_class_sampled"]) == set(ic.SUPPORT_CLASSES)
    assert set(payload["per_class_total"]) == set(ic.SUPPORT_CLASSES)

    assert payload["probes"], "expected at least one probed sample"
    for probe in payload["probes"]:
        assert probe["audit_id"]
        assert probe["support_class"] in ic.SUPPORT_CLASSES
        assert isinstance(probe["conformant"], bool)
        assert probe["backend_probes"], "each sample probes at least one backend"
        for backend_probe in probe["backend_probes"]:
            assert backend_probe["backend_kind"]
            assert backend_probe["data_plane"]
            assert backend_probe["capability_status"] in ic.CAPABILITY_STATUSES
            # Every non-supported status carries a concrete reason.
            if backend_probe["capability_status"] != ic.STATUS_SUPPORTED:
                assert backend_probe["reason"]
            # Provenance is secret-free: booleans / impl labels only.
            provenance = backend_probe["provenance"]
            assert set(provenance) >= {"credential_vending", "managed_versioning"}
            assert json.dumps(provenance)  # serializable, no secret objects


# --------------------------------------------------------------------------- #
# Contract-test backend (test-first plan #2): one pass, one unsupported
# capability, one credential-refresh path.
# --------------------------------------------------------------------------- #


def test_contract_test_backend_has_pass_unsupported_and_refresh() -> None:
    report = ic.run_conformance(mode=ic.MODE_CONTRACT_TEST, per_class=4)

    statuses = {
        backend_probe.capability_status
        for probe in report.probes
        for backend_probe in probe.backend_probes
    }
    # At least one supported (pass) and one unsupported (a real capability gap).
    assert ic.STATUS_SUPPORTED in statuses
    assert ic.STATUS_UNSUPPORTED in statuses

    # The capability-check class must show BOTH: gated on a plain db:// backend
    # and supported once the deployment advertises the capability -- proof the
    # 0128 gate both gates and opens.
    cap_probes = [
        p for p in report.probes if p.support_class == ic.SUPPORT_CLASS_CAPABILITY_CHECK
    ]
    assert cap_probes
    for probe in cap_probes:
        # Two db:// probes (plain + advertised) share a backend_kind, so compare
        # the status *set*: the gate must both gate (plain) and open (advertised).
        statuses = {bp.capability_status for bp in probe.backend_probes}
        assert ic.STATUS_UNSUPPORTED in statuses
        assert ic.STATUS_SUPPORTED in statuses
        assert probe.conformant

    # At least one namespace_required sample exercises a credential refresh path.
    ns_probes = [
        p for p in report.probes if p.support_class == ic.SUPPORT_CLASS_NAMESPACE_REQUIRED
    ]
    assert ns_probes
    assert any(p.credential_refreshed for p in ns_probes)
    assert all(p.managed_versioning_enforced for p in ns_probes)


def test_contract_test_report_is_fully_conformant() -> None:
    # Every sampled row's observed gate/adapter behavior must match the audit's
    # support-class expectation. A mismatch here means the SDK's runtime gates
    # drifted from the static audit -- exactly what this suite exists to catch.
    report = ic.run_conformance(mode=ic.MODE_CONTRACT_TEST, per_class=6)
    non_conformant = [p for p in report.probes if not p.conformant]
    assert not non_conformant, [
        (p.audit_id, p.support_class, p.mismatch) for p in non_conformant
    ]
    assert report.ok()


def test_supported_now_probes_link_back_to_audit_row_id() -> None:
    report = ic.run_conformance(mode=ic.MODE_CONTRACT_TEST, per_class=3)
    ids = {p.audit_id for p in report.probes}
    rows_by_id = {row["id"]: row for row in ic.load_audit_rows()}
    # Acceptance: the report links every probe back to a 0076 audit row id.
    assert ids
    assert ids <= set(rows_by_id)


# --------------------------------------------------------------------------- #
# Managed-versioning enforcement is a real, loud refusal (0129 invariant).
# --------------------------------------------------------------------------- #


def test_namespace_managed_versioning_mismatch_is_enforced_loudly() -> None:
    from lancedb_robotics.connections import ManagedVersioningMismatch

    env = ic.contract_test_env(managed_versioning=True)
    spec = env.namespace_spec
    client = env.namespace_client_factory(managed_versioning=True)
    from lancedb_robotics import pylance_execution as px

    with pytest.raises(ManagedVersioningMismatch):
        px.require_namespace_write_supported(
            spec,
            "observations",
            supports_managed_versioning=False,
            namespace_client=client,
        )


def test_contract_test_backend_needs_no_live_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clearing every conformance env var must not affect the contract-test run.
    for name in ic.LIVE_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    report = ic.run_conformance(mode=ic.MODE_CONTRACT_TEST, per_class=2)
    assert report.ok()
    assert report.mode == ic.MODE_CONTRACT_TEST


# --------------------------------------------------------------------------- #
# Live mode without credentials (test-first plan #3): the live subset is marked
# skipped -- with a reason -- and does not fail unrelated local tests.
# --------------------------------------------------------------------------- #


def test_live_mode_without_credentials_skips_without_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ic.LIVE_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    available, missing = ic.live_credentials_available()
    assert not available
    assert missing  # names the absent env vars

    report = ic.run_conformance(mode=ic.MODE_LIVE, per_class=3)
    # A skipped live run is not a failure.
    assert report.ok()
    assert report.mode == ic.MODE_LIVE
    # Every backend probe is skipped and carries the missing-credential reason.
    backend_probes = [bp for p in report.probes for bp in p.backend_probes]
    assert backend_probes
    assert all(bp.capability_status == ic.STATUS_SKIPPED for bp in backend_probes)
    assert all(bp.reason for bp in backend_probes)
    assert report.skipped_live  # the report advertises the skip at the top level


def test_live_conformance_when_configured() -> None:
    # Skips unless the operator has wired the LANCEDB_ROBOTICS_CONFORMANCE_* live
    # endpoint/auth-ref vars (a dedicated live lane makes an absent endpoint a
    # hard failure via LANCEDB_ROBOTICS_REQUIRE_INVOCATION_CONFORMANCE=1). When
    # configured, no sampled row may be non-conformant against the live backend.
    from conftest import require_invocation_conformance_live

    require_invocation_conformance_live()
    report = ic.run_conformance(mode=ic.MODE_LIVE, per_class=3)
    assert report.mode == ic.MODE_LIVE
    non_conformant = [p for p in report.probes if not p.conformant]
    assert not non_conformant, [
        (p.audit_id, p.support_class, p.mismatch) for p in non_conformant
    ]


def test_live_credential_gate_skips_by_default_and_fails_only_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression guard for the CI lane that once set REQUIRE=1 unconditionally and
    # hard-failed by default. Default (no REQUIRE flag) + creds absent must be a
    # clean SKIP so the contract-test lane stays green; only an explicit
    # REQUIRE=1 (a lane that provisioned live creds) turns an absent endpoint into
    # a hard failure.
    from conftest import require_invocation_conformance_live

    for name in ic.LIVE_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LANCEDB_ROBOTICS_REQUIRE_INVOCATION_CONFORMANCE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_invocation_conformance_live()
    monkeypatch.setenv("LANCEDB_ROBOTICS_REQUIRE_INVOCATION_CONFORMANCE", "1")
    with pytest.raises(pytest.fail.Exception):
        require_invocation_conformance_live()


def test_require_flag_never_reddens_contract_test(monkeypatch: pytest.MonkeyPatch) -> None:
    # The require-flag governs only the *live* subset; a contract-test run must be
    # unaffected by it (so the flag alone can never turn the default lane red).
    monkeypatch.setenv("LANCEDB_ROBOTICS_REQUIRE_INVOCATION_CONFORMANCE", "1")
    for name in ic.LIVE_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    report = ic.run_conformance(mode=ic.MODE_CONTRACT_TEST, per_class=3)
    assert report.ok()


def test_live_mode_with_configured_endpoint_probes_db_and_defers_alternate_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Live mode with the endpoint configured but no injected fake client: the
    # db:// posture rows (supported_now, capability_check) resolve and probe
    # (pure capability reasoning, no network), while the alternate-backend rows
    # (direct_required, namespace_required) SKIP as conformant -- their real live
    # execution is 0446. Regression guard: the namespace probe used to
    # return conformant=False ("client unavailable"), which made a working live
    # backend look like a mismatch and the live lane un-passable, and the object-
    # store side used to resolve the db:// URI as an object store.
    monkeypatch.setenv("LANCEDB_API_KEY", "conformance-live-key")
    monkeypatch.setenv(ic.LIVE_DB_URI_ENV, "db://conformance-live")
    monkeypatch.setenv(ic.LIVE_DB_AUTH_REF_ENV, "live-remote")
    monkeypatch.setenv(ic.LIVE_NAMESPACE_IMPL_ENV, "rest")
    monkeypatch.setenv(ic.LIVE_NAMESPACE_PROPERTIES_ENV, '{"uri": "https://ns.live.invalid"}')
    monkeypatch.delenv(ic.LIVE_OBJECT_STORE_URI_ENV, raising=False)

    available, missing = ic.live_credentials_available()
    assert available and not missing

    report = ic.run_conformance(mode=ic.MODE_LIVE, per_class=3)
    assert report.ok()  # no false mismatch anywhere

    # db:// posture rows are really probed (supported / gated), not skipped.
    posture = [
        p
        for p in report.probes
        if p.support_class
        in (ic.SUPPORT_CLASS_SUPPORTED_NOW, ic.SUPPORT_CLASS_CAPABILITY_CHECK)
    ]
    assert posture
    assert all(p.conformant for p in posture)
    assert any(
        bp.capability_status != ic.STATUS_SKIPPED
        for p in posture
        for bp in p.backend_probes
    )

    # Alternate-backend rows are deferred: conformant, with a skipped probe.
    deferred = [
        p
        for p in report.probes
        if p.support_class
        in (ic.SUPPORT_CLASS_DIRECT_REQUIRED, ic.SUPPORT_CLASS_NAMESPACE_REQUIRED)
    ]
    assert deferred
    for probe in deferred:
        assert probe.conformant
        assert all(
            bp.capability_status == ic.STATUS_SKIPPED for bp in probe.backend_probes
        )


def test_summary_counts_reconcile_with_probes() -> None:
    report = ic.run_conformance(mode=ic.MODE_CONTRACT_TEST, per_class=3)
    summary = report.summary()
    assert summary["total_samples"] == len(report.probes)
    assert summary["conformant"] + summary["non_conformant"] == len(report.probes)
    # by_support_class sums back to the sample total.
    assert sum(summary["by_support_class"].values()) == len(report.probes)
