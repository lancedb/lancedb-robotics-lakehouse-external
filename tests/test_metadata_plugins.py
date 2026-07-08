"""Metadata-integration plugin contract + conformance suite tests (backlog 0106)."""

import hashlib
import json
import sys

import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.dataset import create_snapshot
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.metadata_plugins import (
    FAMILY_LINEAGE_EMITTER,
    AdapterAuth,
    AdapterCapabilities,
    AdapterDependency,
    AttemptRecord,
    DependencyStatus,
    EmitResult,
    LineageEmitterMixin,
    MetadataAdapter,
    MetadataPluginError,
    MetadataPluginRegistry,
    ReferenceImporterMixin,
    default_registry,
    list_metadata_plugins,
    run_conformance,
    run_registry_conformance,
)
from lancedb_robotics.scenarios import create_scenario_windows

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures + test doubles.
# ---------------------------------------------------------------------------


def _pipeline_lake(tmp_path, fixtures_dir):
    """A small lake with ingest -> snapshot -> training-run lineage (mirrors 0064)."""

    lake = Lake.init(tmp_path / "robot.lance")
    ingest_mcap(lake, fixtures_dir / "sample.mcap")
    create_scenario_windows(lake, window_ns=100_000_000)
    scenarios = sorted(
        lake.table("scenarios").to_arrow().to_pylist(),
        key=lambda row: (row["start_time_ns"], row["scenario_id"]),
    )
    create_snapshot(
        lake,
        name="plugin-demo",
        tag="plugin-demo",
        scenario_ids=[row["scenario_id"] for row in scenarios],
    )
    lake.training.record_run("plugin-demo", code_ref="git:trainer", hyperparameters={"lr": 0.001})
    lake.lineage.refresh_graph()
    return lake


class _FakeSink:
    """A capturing emitter sink; raises on the ``fail_ordinals`` calls."""

    def __init__(self, *, fail_ordinals=()):
        self.fail_ordinals = set(fail_ordinals)
        self.calls = []

    def emit(self, payload):
        ordinal = len(self.calls)
        self.calls.append(payload)
        if ordinal in self.fail_ordinals:
            raise RuntimeError(f"induced failure at ordinal {ordinal}")
        return {"remote_id": f"remote-{ordinal}", "status": "ok"}


def _digest(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class _ToyEmitterAdapter(MetadataAdapter, LineageEmitterMixin, ReferenceImporterMixin):
    """A fully self-contained emitter+importer plugin with no optional deps.

    It ignores the lake, mints reversible URNs itself, and delivers through an
    injected sink -- so it can pass the conformance suite without pulling any
    external metadata client.
    """

    name = "toy"
    family = FAMILY_LINEAGE_EMITTER
    _PREFIX = "urn:toy:artifact:"

    def capabilities(self):
        return AdapterCapabilities(dry_run=True, emit=True, retry=True, reversible_urns=True)

    def dependency(self):
        return AdapterDependency(adapter=self.name, native=True)

    def build_payloads(self, lake, *, refresh=False):
        return tuple(
            {"artifact_id": aid, "urn": self.to_external_urn(aid), "event": "toy"}
            for aid in (
                "lancedb-robotics:snapshot:abc123",
                "lancedb-robotics:training-run:run/with slashes",
            )
        )

    def emit(self, lake, *, auth, refresh=False, retry=False):
        client = auth.client
        attempts = []
        for payload in self.build_payloads(lake):
            digest = _digest(payload)
            try:
                response = client.emit(payload)
            except Exception as exc:  # noqa: BLE001 - recorded as structured failure
                attempts.append(
                    AttemptRecord(payload_digest=digest, status="failed", error=str(exc))
                )
                continue
            remote = tuple(
                str(response[key]) for key in ("remote_id",) if isinstance(response, dict) and response.get(key)
            )
            attempts.append(
                AttemptRecord(payload_digest=digest, status="delivered", remote_ids=remote)
            )
        failed = sum(1 for a in attempts if a.status == "failed")
        delivered = sum(1 for a in attempts if a.status == "delivered")
        status = "partial" if failed and delivered else "failed" if failed else "delivered"
        return EmitResult(
            adapter=self.name,
            target=auth.target or "toy",
            mode="retry" if retry else "emit",
            status=status,
            attempts=tuple(attempts),
        )

    def to_external_urn(self, artifact_id):
        from urllib.parse import quote

        return f"{self._PREFIX}{quote(artifact_id, safe='')}"

    def from_external_urn(self, urn):
        from urllib.parse import unquote

        if not urn.startswith(self._PREFIX):
            raise MetadataPluginError(f"not a toy urn: {urn!r}")
        return unquote(urn[len(self._PREFIX) :])


class _MutatingUrnAdapter(_ToyEmitterAdapter):
    """Violates the identity contract: its URN round-trip loses the original id."""

    name = "mutating"

    def from_external_urn(self, urn):
        return super().from_external_urn(urn) + "-MUTATED"


class _DroppingUrnAdapter(_ToyEmitterAdapter):
    """Violates the contract: from_external_urn cannot recover the id at all."""

    name = "dropping"

    def to_external_urn(self, artifact_id):
        return "urn:toy:opaque:constant"  # non-reversible

    def from_external_urn(self, urn):
        return "unknown"


class _CapturingOpenLineagePlugin(MetadataAdapter, LineageEmitterMixin):
    """A third-party-style plugin that consumes the 0064 OpenLineage payloads."""

    name = "capturing-openlineage"
    family = FAMILY_LINEAGE_EMITTER

    def capabilities(self):
        return AdapterCapabilities(dry_run=True, emit=True)

    def dependency(self):
        return AdapterDependency(adapter=self.name, native=True)

    def build_payloads(self, lake, *, refresh=False):
        from lancedb_robotics.lineage_integrations import export_openlineage

        return tuple(export_openlineage(lake, refresh=refresh, dry_run=True).events)

    def emit(self, lake, *, auth, refresh=False, retry=False):
        attempts = []
        for payload in self.build_payloads(lake):
            response = auth.client.emit(payload)
            attempts.append(
                AttemptRecord(
                    payload_digest=_digest(payload),
                    status="delivered",
                    remote_ids=(str(response.get("remote_id", "")),),
                )
            )
        return EmitResult(
            adapter=self.name, target=auth.target or "capture", mode="emit", status="delivered", attempts=tuple(attempts)
        )


# ---------------------------------------------------------------------------
# Acceptance criterion 1: a toy plugin passes conformance without optional deps.
# ---------------------------------------------------------------------------


def test_toy_plugin_passes_conformance_without_optional_dependencies(tmp_path):
    adapter = _ToyEmitterAdapter()
    lake = Lake.init(tmp_path / "empty.lance")  # no optional deps, no lineage rows

    report = run_conformance(adapter, lake=lake)

    assert report.passed, report.to_dict()
    ran = {check.name for check in report.ran}
    # The emit/urn/auth/failure checks actually ran (not skipped) and passed.
    assert {
        "urn:reversible",
        "payload:dry-run-parity",
        "auth:non-persistence",
        "emit:failure-recording",
    } <= ran


def test_custom_plugin_registers_and_receives_0064_openlineage_payloads(tmp_path, fixtures_dir):
    lake = _pipeline_lake(tmp_path, fixtures_dir)
    registry = MetadataPluginRegistry()
    registry.register("capturing-openlineage", _CapturingOpenLineagePlugin)

    adapter = registry.get("capturing-openlineage")
    sink = _FakeSink()
    result = adapter.emit(lake, auth=AdapterAuth(client=sink, target="capture"))

    expected = list(lake.lineage.export_openlineage(refresh=False).events)
    assert sink.calls == expected
    assert result.delivered == len(expected)
    assert result.status == "delivered"


# ---------------------------------------------------------------------------
# Acceptance criterion 2: dependency probe shape is actionable.
# ---------------------------------------------------------------------------


def test_missing_plugin_dependency_probe_returns_actionable_shape():
    from lancedb_robotics.metadata_plugins import _TrackerSyncAdapter

    adapter = _TrackerSyncAdapter(
        "definitely-missing-0106",
        "definitely_missing_0106_module",
        "definitely-missing-0106",
    )
    status = adapter.probe()

    assert isinstance(status, DependencyStatus)
    assert status.adapter == "definitely-missing-0106"
    assert status.available is False
    assert status.module == "definitely_missing_0106_module"
    assert status.optional_extra == "definitely-missing-0106"
    assert "pip install 'lancedb-robotics[definitely-missing-0106]'" in status.install_hint
    assert "not installed" in status.message
    payload = status.to_dict()
    assert {"adapter", "available", "module", "optional_extra", "plugin", "native", "install_hint"} <= set(payload)


def test_dependency_probe_never_imports_the_optional_dependency():
    # Probing every registered adapter must not *import* an optional client -- it
    # uses find_spec. Snapshot before/after so the assertion is robust to whatever
    # earlier tests in a full run may already have imported.
    watched = ("openlineage", "openlineage.client", "mlflow", "wandb", "datahub")
    before = {module for module in watched if module in sys.modules}
    list_metadata_plugins()  # describe() -> probe() -> find_spec() for every adapter
    newly_imported = {module for module in watched if module in sys.modules} - before
    assert not newly_imported, f"probing imported optional dependency: {sorted(newly_imported)}"


def test_native_adapter_probe_reports_no_dependency():
    adapter = default_registry().get("generic")
    status = adapter.probe()
    assert status.native is True
    assert status.available is True
    assert status.install_hint == "no optional dependency required"


# ---------------------------------------------------------------------------
# Acceptance criterion (test-first): conformance rejects identity violations.
# ---------------------------------------------------------------------------


def test_conformance_rejects_urn_mutating_adapter(tmp_path):
    lake = Lake.init(tmp_path / "m.lance")
    report = run_conformance(_MutatingUrnAdapter(), lake=lake)

    assert not report.passed
    urn_check = next(c for c in report.checks if c.name == "urn:reversible")
    assert urn_check.status == "failed"
    assert "not reversible" in urn_check.detail


def test_conformance_rejects_urn_dropping_adapter(tmp_path):
    lake = Lake.init(tmp_path / "d.lance")
    report = run_conformance(_DroppingUrnAdapter(), lake=lake)

    assert not report.passed
    urn_check = next(c for c in report.checks if c.name == "urn:reversible")
    assert urn_check.status == "failed"


def test_conformance_rejects_capability_without_interface(tmp_path):
    class _Liar(MetadataAdapter):
        name = "liar"
        family = FAMILY_LINEAGE_EMITTER

        def capabilities(self):
            return AdapterCapabilities(emit=True, reversible_urns=True)

        def dependency(self):
            return AdapterDependency(adapter=self.name, native=True)

    report = run_conformance(_Liar(), lake=Lake.init(tmp_path / "l.lance"))
    assert not report.passed
    caps_check = next(c for c in report.checks if c.name == "contract:capabilities")
    assert caps_check.status == "failed"
    assert "build_payloads" in caps_check.detail or "to_external_urn" in caps_check.detail


# ---------------------------------------------------------------------------
# Acceptance criterion 4: auth refs are runtime-only, never persisted.
# ---------------------------------------------------------------------------


def test_auth_refs_present_in_runtime_config_but_absent_from_persisted_rows(tmp_path, fixtures_dir):
    lake = _pipeline_lake(tmp_path, fixtures_dir)
    adapter = default_registry().get("openlineage")
    # auth_ref is a *reference name* (safe to log); the raw bearer token is the secret.
    ref_name = "prod-openlineage-cred"
    token = "SECRET-TOKEN-abc-0106"
    auth = AdapterAuth(
        auth_ref=ref_name,
        headers={"Authorization": f"Bearer {token}"},
        client=_FakeSink(),
        target="auth-test-0106",
    )

    result = adapter.emit(lake, auth=auth)
    assert result.delivered >= 1

    # Neither the reference name nor the raw token is ever written into canonical rows.
    for table in ("lineage_delivery_attempts", "lineage_artifacts", "lineage_executions", "lineage_edges"):
        rows = lake.table(table).to_arrow().to_pylist()
        blob = json.dumps(rows, default=str)
        assert token not in blob, f"auth token leaked into {table}"
        assert ref_name not in blob, f"auth ref name leaked into {table}"
    assert token not in json.dumps(result.to_dict())
    # The redacted view exposes only which inputs were present (the ref name), never the token.
    redacted = json.dumps(auth.redacted())
    assert token not in redacted
    assert ref_name in redacted


# ---------------------------------------------------------------------------
# Acceptance criterion 5: adapter failures record structured, retryable state.
# ---------------------------------------------------------------------------


def test_emit_failure_records_structured_status_and_retry_skips_delivered(tmp_path, fixtures_dir):
    lake = _pipeline_lake(tmp_path, fixtures_dir)
    adapter = default_registry().get("openlineage")

    failing = _FakeSink(fail_ordinals=[0])
    first = adapter.emit(lake, auth=AdapterAuth(client=failing, target="retry-test-0106"))
    assert first.failed == 1
    assert first.status == "partial"
    failed_attempts = [a for a in first.attempts if a.status == "failed"]
    assert failed_attempts and all(a.error for a in failed_attempts)
    assert all(a.payload_digest for a in failed_attempts)

    good = _FakeSink()
    retry = adapter.emit(lake, auth=AdapterAuth(client=good, target="retry-test-0106"), retry=True)
    # The already-delivered payloads are skipped; only the previously-failed one re-sends.
    assert retry.delivered == 1
    assert retry.already_delivered == first.delivered

    persisted = lake.lineage.lineage_delivery_attempts(backend="openlineage", target="retry-test-0106")
    assert any(a.status == "failed" and a.error for a in persisted)
    assert sum(1 for a in persisted if a.status == "delivered") == first.delivered + 1


# ---------------------------------------------------------------------------
# Built-in adapters + registry.
# ---------------------------------------------------------------------------


def test_builtin_registry_registers_all_ecosystems_without_optional_deps():
    names = set(default_registry().names())
    assert {"openlineage", "datahub", "generic", "mlflow", "wandb", "dvc", "lakefs", "kubeflow"} <= names


def test_all_builtin_adapters_pass_conformance(tmp_path, fixtures_dir):
    lake = _pipeline_lake(tmp_path, fixtures_dir)
    reports = run_registry_conformance(default_registry(), lake=lake)
    failing = {r.adapter: [f.to_dict() for f in r.failures] for r in reports if not r.passed}
    assert not failing, failing


def test_datahub_adapter_dry_run_matches_real_payload_contract(tmp_path, fixtures_dir):
    lake = _pipeline_lake(tmp_path, fixtures_dir)
    adapter = default_registry().get("datahub")
    payloads = adapter.build_payloads(lake)
    assert payloads
    sink = _FakeSink()
    adapter.emit(lake, auth=AdapterAuth(client=sink, target="datahub-contract-0106"))
    assert sink.calls == list(payloads)


def test_registry_rejects_duplicate_registration_and_unknown_lookup():
    registry = MetadataPluginRegistry()
    registry.register("dup", _ToyEmitterAdapter)
    with pytest.raises(MetadataPluginError, match="already registered"):
        registry.register("dup", _ToyEmitterAdapter)
    registry.register("dup", _ToyEmitterAdapter, replace=True)  # replace is allowed
    with pytest.raises(MetadataPluginError, match="unknown metadata plugin"):
        registry.get("nope")


def test_generic_manifest_adapter_load_bundle_round_trip():
    adapter = default_registry().get("generic")
    bundle = {"training_runs": [{"external_run_id": "r1"}], "model_artifacts": [], "evaluation_runs": []}
    assert adapter.load_bundle(bundle=bundle) == bundle
    assert adapter.load_bundle() == {"training_runs": [], "model_artifacts": [], "evaluation_runs": []}


# ---------------------------------------------------------------------------
# CLI surface.
# ---------------------------------------------------------------------------


def test_cli_plugins_lists_registered_adapters():
    result = runner.invoke(app, ["lineage", "plugins", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    names = {row["name"] for row in payload}
    assert {"openlineage", "datahub", "generic"} <= names
    openlineage = next(row for row in payload if row["name"] == "openlineage")
    assert openlineage["capabilities"]["reversible_urns"] is True


def test_cli_conformance_passes_for_all_plugins(tmp_path, fixtures_dir):
    lake = _pipeline_lake(tmp_path, fixtures_dir)
    result = runner.invoke(app, ["lineage", "conformance", "--lake", str(lake.uri), "--json"])
    assert result.exit_code == 0, result.output
    reports = json.loads(result.output)
    assert reports and all(report["passed"] for report in reports)


def test_cli_conformance_reports_failure_exit_code():
    # Register a broken adapter into the default registry the CLI reads. No --lake:
    # the urn-reversibility check fails on its own, so exit 1 reflects conformance,
    # not a lake-open error.
    default_registry().register("cli-broken-0106", _MutatingUrnAdapter, replace=True)
    try:
        result = runner.invoke(app, ["lineage", "conformance", "--plugin", "cli-broken-0106"])
        assert result.exit_code == 1, result.output
        assert "FAIL" in result.output
    finally:
        default_registry().unregister("cli-broken-0106")
