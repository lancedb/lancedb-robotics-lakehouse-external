"""Shared direct-pylance execution adapter (backlog 0129).

Follow-up to 0036 (connection resolver) and 0128 (capability gates). These tests
exercise the adapter's contract from the test-first plan in the backlog:

* namespace-vended credentials near expiry are refreshed *before* a dataset is
  opened for direct IO;
* a managed-versioning namespace refuses a direct additive write that does not
  support managed versioning (loud failure, not silent version fork);
* blob hydration routed through the adapter on a namespace-backed lake returns
  byte-for-byte the same bytes as a local ``LanceDataset`` hydration, without
  ever calling ``Table.to_lance()`` on a remote table;

plus the data-plane classification / provenance surface and the guarantee that
local and object-store lakes are untouched (every entry point is a no-op /
pass-through for them).
"""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

from lancedb_robotics import blob, pylance_execution
from lancedb_robotics.connections import (
    LakeConnectionSpec,
    ManagedVersioningMismatch,
    namespace_properties_from_options,
    resolve_lake_connection,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.pylance_execution import (
    LOCAL,
    NAMESPACE_DIRECT,
    OBJECT_STORE,
    REMOTE_DB,
    UNCLASSIFIED,
    data_plane,
    data_plane_provenance,
    has_pylance_access,
    namespace_table_id,
    open_direct_dataset,
    require_namespace_write_supported,
)
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA

# --------------------------------------------------------------------------- #
# Fakes / helpers
# --------------------------------------------------------------------------- #


class _FakeNamespace:
    """A namespace client that returns canned ``describe_table`` responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def describe_table(self, request):
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[index]


def _request_attr(request, name):
    return request[name] if isinstance(request, dict) else getattr(request, name)


def _namespace_spec(
    *, database_prefix: str = "robotics", delimiter: str = "."
) -> LakeConnectionSpec:
    """A resolved namespace-backed spec (``pylance_access`` set, direct IO)."""
    properties = namespace_properties_from_options(
        uri="https://phalanx.acme.internal",
        database="acme",
        database_prefix=database_prefix,
        delimiter=delimiter,
    )
    return resolve_lake_connection(
        namespace_client_impl="rest",
        namespace_client_properties=properties,
        namespace_auth_ref="phalanx-prod",
    )


def _describe(*, session: str, expires_at_millis: str, managed_versioning: bool = False) -> dict:
    return {
        "location": "s3://robotics/observations.lance",
        "storage_options": {"session": session, "expires_at_millis": expires_at_millis},
        "managed_versioning": managed_versioning,
    }


# --------------------------------------------------------------------------- #
# Data-plane classification + provenance
# --------------------------------------------------------------------------- #


def test_data_plane_classifies_every_backend() -> None:
    assert data_plane(resolve_lake_connection("/tmp/lake.lance")) == LOCAL
    assert data_plane(resolve_lake_connection("s3://bucket/lake.lance")) == OBJECT_STORE
    assert data_plane(_namespace_spec()) == NAMESPACE_DIRECT
    assert data_plane(None) == UNCLASSIFIED


def test_remote_db_data_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANCEDB_API_KEY", "test-key")
    assert data_plane(resolve_lake_connection("db://robotics")) == REMOTE_DB


def test_has_pylance_access_only_for_namespace() -> None:
    assert has_pylance_access(_namespace_spec()) is True
    assert has_pylance_access(resolve_lake_connection("/tmp/lake.lance")) is False
    assert has_pylance_access(resolve_lake_connection("s3://bucket/lake.lance")) is False
    assert has_pylance_access(None) is False


def test_data_plane_provenance_is_secret_free() -> None:
    spec = _namespace_spec()
    provenance = data_plane_provenance(spec)
    assert provenance["data_plane"] == NAMESPACE_DIRECT
    assert provenance["credential_vending"] is True
    assert provenance["namespace_impl"] == "rest"
    assert provenance["direct_object_io"] is True
    # No resolved secret / vended session material leaks into provenance.
    blob_text = str(provenance)
    assert "Bearer" not in blob_text
    assert "phalanx-prod" not in blob_text  # auth-ref name is not surfaced here either


def test_data_plane_provenance_local_and_none() -> None:
    local = data_plane_provenance(resolve_lake_connection("/tmp/lake.lance"))
    assert local["data_plane"] == LOCAL
    assert local["credential_vending"] is False
    unclassified = data_plane_provenance(None)
    assert unclassified["data_plane"] == UNCLASSIFIED
    assert unclassified["backend_kind"] is None


def test_safe_summary_carries_data_plane() -> None:
    assert resolve_lake_connection("/tmp/lake.lance").safe_summary()["data_plane"] == LOCAL
    assert _namespace_spec().safe_summary()["data_plane"] == NAMESPACE_DIRECT


# --------------------------------------------------------------------------- #
# table_id derivation
# --------------------------------------------------------------------------- #


def test_namespace_table_id_prepends_database_prefix() -> None:
    spec = _namespace_spec(database_prefix="robotics")
    assert namespace_table_id(spec, "observations") == ("robotics", "observations")


def test_namespace_table_id_multi_level_prefix_and_delimiter() -> None:
    spec = _namespace_spec(database_prefix="team$robotics", delimiter="$")
    assert namespace_table_id(spec, "observations") == ("team", "robotics", "observations")


def test_namespace_table_id_without_prefix_is_bare_name() -> None:
    spec = _namespace_spec(database_prefix="")
    assert namespace_table_id(spec, "observations") == ("observations",)


# --------------------------------------------------------------------------- #
# Acceptance: credential refresh before opening a dataset
# --------------------------------------------------------------------------- #


def test_open_direct_dataset_refreshes_near_expiry_creds_before_open() -> None:
    namespace = _FakeNamespace(
        [
            _describe(session="old", expires_at_millis="1"),  # near expiry
            _describe(session="fresh", expires_at_millis="9999999999999"),
        ]
    )
    spec = _namespace_spec()
    opened: dict = {}

    def fake_dataset(**kwargs):
        opened.update(kwargs)
        return "DATASET"

    result = open_direct_dataset(
        spec, "observations", namespace_client=namespace, dataset_factory=fake_dataset
    )

    assert result == "DATASET"
    # Two describe calls: initial (near-expiry) then refresh before opening.
    assert len(namespace.requests) == 2
    # table_id derived from the namespace prefix + logical table name.
    assert _request_attr(namespace.requests[0], "id") == ["robotics", "observations"]
    assert opened["namespace_client"] is namespace
    assert opened["table_id"] == ["robotics", "observations"]


def test_open_direct_dataset_no_refresh_when_creds_fresh() -> None:
    namespace = _FakeNamespace([_describe(session="fresh", expires_at_millis="9999999999999")])
    spec = _namespace_spec()
    open_direct_dataset(
        spec, "observations", namespace_client=namespace, dataset_factory=lambda **k: "DATASET"
    )
    assert len(namespace.requests) == 1  # fresh creds -> no refresh describe


# --------------------------------------------------------------------------- #
# Acceptance: managed-versioning mismatch blocks a direct additive write
# --------------------------------------------------------------------------- #


def test_require_namespace_write_refuses_unsupported_managed_versioning() -> None:
    namespace = _FakeNamespace(
        [
            SimpleNamespace(
                location="s3://robotics/observations.lance",
                storage_options={},
                managed_versioning=True,
            )
        ]
    )
    spec = _namespace_spec()
    with pytest.raises(ManagedVersioningMismatch):
        require_namespace_write_supported(
            spec,
            "observations",
            supports_managed_versioning=False,
            namespace_client=namespace,
        )


def test_require_namespace_write_allows_when_support_declared() -> None:
    namespace = _FakeNamespace(
        [
            SimpleNamespace(
                location="s3://robotics/observations.lance",
                storage_options={},
                managed_versioning=True,
            )
        ]
    )
    spec = _namespace_spec()
    # A path that explicitly supports managed versioning passes the check.
    require_namespace_write_supported(
        spec, "observations", supports_managed_versioning=True, namespace_client=namespace
    )
    assert len(namespace.requests) == 1


def test_require_namespace_write_is_noop_for_non_namespace_backends() -> None:
    # Local / object-store / db:// / unclassified: no describe, no raise.
    require_namespace_write_supported(resolve_lake_connection("/tmp/lake.lance"), "observations")
    require_namespace_write_supported(
        resolve_lake_connection("s3://bucket/lake.lance"), "observations"
    )
    require_namespace_write_supported(None, "observations")


# --------------------------------------------------------------------------- #
# Acceptance: namespace-direct blob hydration == local, without Table.to_lance()
# --------------------------------------------------------------------------- #

_PAYLOADS = {
    "run-ns:/camera/front:000000": b"\x89PNG\r\n" + b"front;" * 4000,
    "run-ns:/lidar/top:000000": b"PCD" + b"\x01\x02\x03\x04" * 5000,
}


def _obs_row(observation_id: str, topic: str, blob_bytes: bytes | None, sequence: int) -> dict:
    return {
        "observation_id": observation_id,
        "run_id": "run-ns",
        "timestamp_ns": 1_700_000_000_000_000_000 + sequence,
        "topic": topic,
        "modality": "image" if "camera" in topic else "pointcloud",
        "raw_uri": "/archival/source.mcap",
        "raw_channel": topic,
        "raw_sequence": sequence,
        "payload_blob": blob_bytes,
        "decode_status": "decoded",
    }


@pytest.fixture
def blob_lake(tmp_path) -> Lake:
    lake = Lake.init(tmp_path / "ns-blobs.lance")
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": "run-ns", "run_kind": "drive", "raw_uri": "/archival/source.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    rows = [
        _obs_row(
            "run-ns:/camera/front:000000",
            "/camera/front",
            _PAYLOADS["run-ns:/camera/front:000000"],
            0,
        ),
        _obs_row(
            "run-ns:/lidar/top:000000", "/lidar/top", _PAYLOADS["run-ns:/lidar/top:000000"], 0
        ),
    ]
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    return lake


class _ToLanceTripwire:
    """Stands in for a remote lancedb Table; to_lance() must never be reached."""

    name = "observations"

    def __init__(self) -> None:
        self.to_lance_called = False

    def to_lance(self):  # pragma: no cover - the whole point is this stays uncalled
        self.to_lance_called = True
        raise AssertionError("to_lance() reached on a namespace-backed lake")


def test_namespace_direct_blob_hydration_matches_local_bytes(blob_lake: Lake) -> None:
    observations = blob_lake.table("observations")
    local_dataset = observations.to_lance()

    # Baseline: local hydration bytes.
    local_bytes = blob.fetch_blobs(
        observations,
        blob.PAYLOAD_BLOB_COLUMN,
        list(_PAYLOADS),
        id_column="observation_id",
        connection_spec=blob_lake.connection_spec,
    )
    assert local_bytes == _PAYLOADS

    # Namespace-direct: the adapter opens the *same underlying dataset* through a
    # fake access, so identical bytes prove the routing (not the local table).
    class _FakeAccess:
        def __init__(self) -> None:
            self.opened = False

        def open_dataset(self, **kwargs):
            self.opened = True
            return local_dataset

    fake_access = _FakeAccess()
    handle = _ToLanceTripwire()
    direct_bytes = blob.fetch_blobs(
        handle,
        blob.PAYLOAD_BLOB_COLUMN,
        list(_PAYLOADS),
        id_column="observation_id",
        connection_spec=_namespace_spec(),
        namespace_access_factory=lambda spec, table: fake_access,
    )

    assert direct_bytes == local_bytes == _PAYLOADS
    assert fake_access.opened is True
    # Critical: the namespace route never fell back to Table.to_lance().
    assert handle.to_lance_called is False


def test_local_blob_hydration_unchanged(blob_lake: Lake) -> None:
    # A genuine local lake never routes through the namespace adapter.
    observations = blob_lake.table("observations")
    assert blob.fetch_blobs(
        observations,
        blob.PAYLOAD_BLOB_COLUMN,
        ["run-ns:/lidar/top:000000"],
        id_column="observation_id",
        connection_spec=blob_lake.connection_spec,
    ) == {"run-ns:/lidar/top:000000": _PAYLOADS["run-ns:/lidar/top:000000"]}


def test_namespace_access_requires_pylance_access() -> None:
    with pytest.raises(ValueError, match="pylance_access is None"):
        pylance_execution.namespace_access(
            resolve_lake_connection("/tmp/lake.lance"), "observations"
        )


def test_namespace_blob_hydration_without_resolvable_name_fails_loudly() -> None:
    # A namespace-backed Table whose name cannot be resolved must fail loudly
    # rather than silently drop to an un-routed to_lance() (SKILLS.md: typed
    # error, never a silent degrade; BUG-11: pin the guardrail).
    class _NamelessTable:
        def to_lance(self):  # pragma: no cover - must never be reached
            raise AssertionError("to_lance() reached instead of failing loudly")

    with pytest.raises(ValueError, match="table name"):
        blob.fetch_blobs(
            _NamelessTable(),
            blob.PAYLOAD_BLOB_COLUMN,
            ["obs-1"],
            id_column="observation_id",
            connection_spec=_namespace_spec(),
        )


# --------------------------------------------------------------------------- #
# Wiring: real write paths run the managed-versioning guard before writing
# --------------------------------------------------------------------------- #


def _reclassify_as_managed_namespace(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> _FakeNamespace:
    """Point a real local lake's spec at a namespace that manages versions."""
    namespace = _FakeNamespace(
        [
            SimpleNamespace(
                location="s3://robotics/table.lance",
                storage_options={},
                managed_versioning=True,
            )
        ]
    )
    spec = _namespace_spec()
    lake.connection_spec = spec
    lake.capabilities = spec.capabilities
    # The guard builds a namespace client from the spec; return the fake instead
    # of contacting a real server.
    monkeypatch.setattr(
        "lancedb_robotics.connections.create_namespace_client",
        lambda **kwargs: namespace,
    )
    return namespace


def test_enrich_refuses_managed_versioning_namespace(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lancedb_robotics.enrich import EmbeddingProvider, enrich_scenarios
    from lancedb_robotics.schemas import SCENARIOS_SCHEMA

    lake = Lake.init(tmp_path / "enrich-ns.lance")
    lake.table("scenarios").add(
        pa.Table.from_pylist(
            [
                {
                    "scenario_id": "scn-0001",
                    "run_id": "run-ns",
                    "start_time_ns": 10,
                    "end_time_ns": 15,
                    "window_ns": 5,
                    "is_partial": False,
                    "topics": ["/imu"],
                    "observation_ids": ["obs-1"],
                    "observation_count": 1,
                    "summary": "ns scenario",
                }
            ],
            schema=SCENARIOS_SCHEMA,
        )
    )

    class _Provider(EmbeddingProvider):
        name = "ns-test"
        dimension = 4

        def embed(self, ctx):  # pragma: no cover - guard fires before embedding
            return [0.0, 0.0, 0.0, 0.0]

        def embed_text(self, text):  # pragma: no cover
            return [0.0, 0.0, 0.0, 0.0]

    _reclassify_as_managed_namespace(lake, monkeypatch)
    with pytest.raises(ManagedVersioningMismatch):
        enrich_scenarios(lake, embedding_provider=_Provider(), fts_index=False)


def test_maintenance_refuses_managed_versioning_namespace(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lancedb_robotics.maintenance import maintain_lake

    lake = Lake.init(tmp_path / "maint-ns.lance")
    _reclassify_as_managed_namespace(lake, monkeypatch)
    with pytest.raises(ManagedVersioningMismatch):
        maintain_lake(lake, refresh_lineage=False, protect_lineage=False)
