"""Backend capability gates for direct-Lance and control-plane invocations (0128).

Follow-up to the 0076 invocation audit. These tests exercise representative
guarded rows from that matrix: the ``direct_lance``/``blob`` drop, the
``maintenance`` path, ``index`` builds, ``versioning`` checkout, and ``schema``
mutation each fail fast with actionable guidance on a ``db://`` remote DB that
does not advertise the capability, while local and object-store lakes -- which
own the underlying dataset -- are unchanged.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

from lancedb_robotics import blob, curate
from lancedb_robotics.capability_gates import (
    BLOB,
    DIRECT_LANCE,
    INDEX,
    MAINTENANCE,
    OPERATION_GATES,
    SCHEMA,
    VERSIONING,
    BackendCapabilityError,
    backend_supports,
    gated_to_lance,
    require_backend_capability,
    require_lake_capability,
)
from lancedb_robotics.connections import (
    LakeCapabilities,
    LakeConnectionSpec,
    NamespaceConfigError,
    resolve_lake_connection,
)
from lancedb_robotics.enrich import EmbeddingProvider, enrich_scenarios
from lancedb_robotics.indexing import build_fts_index, build_scalar_index, build_vector_index
from lancedb_robotics.lake import Lake
from lancedb_robotics.maintenance import maintain_lake
from lancedb_robotics.schemas import SCENARIOS_SCHEMA

ALL_FAMILIES = (DIRECT_LANCE, BLOB, MAINTENANCE, INDEX, VERSIONING, SCHEMA)


def _remote_spec(**control_plane: bool) -> LakeConnectionSpec:
    """A ``db://`` spec whose data-plane matches the real resolver default."""
    return LakeConnectionSpec(
        kind="lancedb_remote_db",
        uri="db://robotics",
        display_uri="db://robotics",
        capabilities=LakeCapabilities(
            server_side_query=True, blob_fetch_remote=True, **control_plane
        ),
    )


# --------------------------------------------------------------------------- #
# Gate registry + resolver capability wiring
# --------------------------------------------------------------------------- #


def test_registry_covers_all_audited_families_with_real_capabilities() -> None:
    assert set(OPERATION_GATES) == set(ALL_FAMILIES)
    caps = set(LakeCapabilities().__dict__)
    for family, gate in OPERATION_GATES.items():
        assert gate.family == family
        assert gate.capability in caps, gate.capability
        assert gate.fallbacks  # every gate names a fallback


def test_local_backend_advertises_every_gated_family() -> None:
    spec = resolve_lake_connection("/tmp/lake.lance")
    assert spec.kind == "local_path"
    for family in ALL_FAMILIES:
        require_backend_capability(spec, family)  # no raise
        assert backend_supports(spec, family) is True


def test_object_store_backend_advertises_control_plane() -> None:
    spec = resolve_lake_connection("s3://bucket/lake.lance")
    assert spec.kind == "object_store_lancedb_oss"
    caps = spec.capabilities
    assert caps.index_management and caps.table_versioning and caps.schema_evolution
    for family in ALL_FAMILIES:
        require_backend_capability(spec, family)


def test_remote_db_gates_every_family_with_actionable_message() -> None:
    spec = _remote_spec()
    for family in ALL_FAMILIES:
        with pytest.raises(BackendCapabilityError) as excinfo:
            require_backend_capability(spec, family, operation=f"run {family}")
        error = excinfo.value
        assert error.family == family
        assert error.backend_kind == "lancedb_remote_db"
        message = str(error)
        assert "lancedb_remote_db" in message
        assert error.required_capability in message
        # Every direct/control-plane family points at the object-store fallback.
        assert "object_store_lancedb_oss" in message
        assert backend_supports(spec, family) is False


def test_none_spec_is_unclassified_and_passes() -> None:
    for family in ALL_FAMILIES:
        require_backend_capability(None, family)  # no raise
        assert backend_supports(None, family) is True


def test_require_lake_capability_tolerates_missing_connection_spec() -> None:
    class _BareLake:  # a test-double lake with no connection_spec attribute
        pass

    # Unclassified => passes (never AttributeErrors on the fake).
    require_lake_capability(_BareLake(), INDEX)

    class _RemoteLake:
        connection_spec = _remote_spec()

    with pytest.raises(BackendCapabilityError):
        require_lake_capability(_RemoteLake(), INDEX)


def test_unknown_family_is_a_programming_error() -> None:
    with pytest.raises(KeyError):
        require_backend_capability(_remote_spec(), "not_a_family")


# --------------------------------------------------------------------------- #
# Advertising remote capability (acceptance: "unless the backend advertises it")
# --------------------------------------------------------------------------- #


def test_remote_capabilities_advertise_selectively() -> None:
    spec = _remote_spec(table_versioning=True)
    require_backend_capability(spec, VERSIONING)  # advertised -> passes
    with pytest.raises(BackendCapabilityError):
        require_backend_capability(spec, INDEX)  # still not advertised


def test_resolver_advertise_path_flips_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANCEDB_API_KEY", "test-key")
    spec = resolve_lake_connection(
        "db://robotics",
        remote_capabilities={"index_management": True, "schema_evolution": True},
    )
    assert spec.kind == "lancedb_remote_db"
    assert spec.capabilities.index_management is True
    assert spec.capabilities.schema_evolution is True
    assert spec.capabilities.table_versioning is False
    require_backend_capability(spec, INDEX)
    require_backend_capability(spec, SCHEMA)
    with pytest.raises(BackendCapabilityError):
        require_backend_capability(spec, VERSIONING)


def test_resolver_rejects_unknown_advertised_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANCEDB_API_KEY", "test-key")
    with pytest.raises(NamespaceConfigError):
        resolve_lake_connection("db://robotics", remote_capabilities={"teleport": True})


# --------------------------------------------------------------------------- #
# Blob hydration fails *before* to_lance() (audit test-first plan)
# --------------------------------------------------------------------------- #


class _ToLanceTripwire:
    """A fake table handle whose to_lance() must never be reached when gated."""

    def __init__(self) -> None:
        self.to_lance_called = False

    def to_lance(self):  # pragma: no cover - the whole point is this stays uncalled
        self.to_lance_called = True
        raise AssertionError("to_lance() reached on a gated backend")


def test_blob_hydration_fails_before_to_lance_on_remote() -> None:
    handle = _ToLanceTripwire()
    with pytest.raises(BackendCapabilityError) as excinfo:
        blob.fetch_blobs(
            handle,
            blob.PAYLOAD_BLOB_COLUMN,
            ["obs-1", "obs-2"],
            id_column="observation_id",
            connection_spec=_remote_spec(),
        )
    assert handle.to_lance_called is False
    assert excinfo.value.family == BLOB
    assert "object_store_lancedb_oss" in str(excinfo.value)


def test_gated_to_lance_passes_through_on_local() -> None:
    class _Handle:
        def to_lance(self):
            return "DATASET"

    local = resolve_lake_connection("/tmp/lake.lance")
    assert gated_to_lance(_Handle(), local) == "DATASET"
    # bare dataset (no to_lance) returned as-is after the gate.
    sentinel = object()
    assert gated_to_lance(sentinel, local) is sentinel


# --------------------------------------------------------------------------- #
# Control-plane paths on a real local lake whose spec is flipped to remote
# --------------------------------------------------------------------------- #


@pytest.fixture
def local_lake(tmp_path) -> Lake:
    return Lake.init(tmp_path / "robot.lance")


def _as_remote(lake: Lake, **control_plane: bool) -> None:
    """Reclassify a real local lake as a db:// remote for gate assertions."""
    spec = _remote_spec(**control_plane)
    lake.connection_spec = spec
    lake.capabilities = spec.capabilities


def _seed_scenario(lake: Lake) -> None:
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-0001",
                    "run_id": "run-gate",
                    "start_time_ns": 10,
                    "end_time_ns": 15,
                    "window_ns": 5,
                    "is_partial": False,
                    "topics": ["/imu"],
                    "observation_ids": ["obs-1"],
                    "observation_count": 1,
                    "summary": "gate scenario",
                }
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )


def test_index_builds_skip_with_capability_reason_on_remote(local_lake: Lake) -> None:
    # Index builds are a pure optimization (queries still full-scan without one),
    # so an unsupported backend degrades to a `skipped` result carrying the
    # capability guidance rather than failing the caller (ingest/enrich/curate).
    _as_remote(local_lake)
    for builder, column in (
        (build_scalar_index, "scenario_id"),
        (build_vector_index, "embedding"),
        (build_fts_index, "summary"),
    ):
        result = builder(local_lake, table="scenarios", column=column)
        assert result.status == "skipped"
        assert "index_management" in (result.reason or "")
        assert "object_store_lancedb_oss" in (result.reason or "")


def test_index_builds_run_when_remote_advertises_index_management(local_lake: Lake) -> None:
    _as_remote(local_lake, index_management=True)
    # Advertised -> the builder proceeds past the gate (empty table -> skipped by
    # the existing below-floor/no-rows logic, not by the capability gate).
    result = build_scalar_index(local_lake, table="scenarios", column="scenario_id")
    assert "index_management" not in (result.reason or "")


def test_maintenance_gated_on_remote(local_lake: Lake) -> None:
    _as_remote(local_lake)
    with pytest.raises(BackendCapabilityError) as excinfo:
        maintain_lake(local_lake)
    assert excinfo.value.family == MAINTENANCE


def test_versioning_checkout_gated_on_remote(local_lake: Lake) -> None:
    _as_remote(local_lake)
    with pytest.raises(BackendCapabilityError) as excinfo:
        curate._rows_at_version(local_lake, "scenarios", 1)
    assert excinfo.value.family == VERSIONING


def test_versioning_current_version_read_allowed_without_checkout(local_lake: Lake) -> None:
    # version=None is a plain read (table_read), supported over db://; no gate.
    _as_remote(local_lake)
    assert curate._rows_at_version(local_lake, "scenarios", None) == []


def test_schema_mutation_gated_on_remote(local_lake: Lake) -> None:
    _seed_scenario(local_lake)

    class _Provider(EmbeddingProvider):
        name = "gate-test"
        dimension = 4

        def embed(self, ctx):  # pragma: no cover - gate fires before embedding
            return [0.0, 0.0, 0.0, 0.0]

        def embed_text(self, text):  # pragma: no cover
            return [0.0, 0.0, 0.0, 0.0]

    _as_remote(local_lake)
    with pytest.raises(BackendCapabilityError) as excinfo:
        enrich_scenarios(local_lake, embedding_provider=_Provider(), fts_index=False)
    assert excinfo.value.family == SCHEMA


def test_advertised_remote_versioning_checkout_runs(local_lake: Lake) -> None:
    # A deployment that advertises table_versioning gets the real checkout path.
    _as_remote(local_lake, table_versioning=True)
    assert curate._rows_at_version(local_lake, "scenarios", 1) == []


# --------------------------------------------------------------------------- #
# Local behavior is unchanged (no gate on a genuine local lake)
# --------------------------------------------------------------------------- #


def test_local_lake_control_plane_unchanged(local_lake: Lake) -> None:
    # Real local lake: these run their normal logic (no BackendCapabilityError).
    scalar = build_scalar_index(local_lake, table="scenarios", column="scenario_id")
    assert scalar.status in {"built", "already_present", "skipped"}
    fts = build_fts_index(local_lake, table="scenarios", column="summary")
    assert fts.status in {"built", "skipped", "already_present"}
    report = maintain_lake(local_lake, refresh_lineage=False, protect_lineage=False)
    assert report.lake_uri == local_lake.uri
    # blob hydration on a local lake resolves the dataset (empty -> {}).
    assert (
        blob.fetch_blobs(
            local_lake.table("observations"),
            blob.PAYLOAD_BLOB_COLUMN,
            ["obs-x"],
            id_column="observation_id",
            connection_spec=local_lake.connection_spec,
        )
        == {}
    )


def test_capabilities_frozen_dataclass_roundtrips() -> None:
    caps = LakeCapabilities(index_management=True)
    assert dataclasses.replace(caps, table_versioning=True).table_versioning is True
