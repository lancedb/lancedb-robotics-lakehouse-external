"""Tests for the Enterprise page-cache prewarm request planner (backlog 0122).

The planner emits a VALID query-node ``PageCacheBeginPrewarmRequest``
(``{db, table, columns, table_version, concurrency}``) plus advisory cost
estimates. It must never emit plan-executor placement, ``pe_fanout``, or
row-id / fragment / row-group routing -- a client talks only to object storage
or the query node, never a plan executor.
"""

import pyarrow as pa

from lancedb_robotics.blob import PAYLOAD_BLOB_COLUMN
from lancedb_robotics.training_prewarm_planner import (
    PAGE_CACHE_PREWARM_SPEC_SCHEMA_VERSION,
    PrewarmPlannerOptions,
    TableMetadata,
    build_page_cache_prewarm_plan,
    resolve_prewarm_database,
)

# Wire keys the query node accepts (sophon PageCacheBeginPrewarmRequest + schema pin).
_ALLOWED_WIRE_KEYS = {
    "schema_version",
    "id",
    "db",
    "table",
    "columns",
    "table_version",
    "concurrency",
}
# Anything that would imply the client is routing to / naming a plan executor,
# or supplying row/fragment placement -- must never appear on the wire.
_FORBIDDEN_WIRE_KEYS = {
    "pe_fanout",
    "pe_addrs",
    "placement",
    "placement_slot",
    "fragments",
    "fragment_ranges",
    "row_groups",
    "row_group_ranges",
    "row_id_ranges",
    "row_ids",
}
# The plan must not reintroduce fragment-level reasoning anywhere -- fragment / PE
# fanout is the query node's job and appears only in the prewarm response.
_FORBIDDEN_ESTIMATE_KEYS = {
    "local_fragment_estimate",
    "fragment_count",
    "fragments",
    "row_groups",
    "pe_fanout",
}


def _request(*, tables=None, **overrides):
    base = {
        "kind": "lancedb-robotics/training-prewarm/v1",
        "requested": True,
        "prewarm_id": "prewarm-deadbeef",
        "policy": "epoch",
        "scope": "epoch",
        "excluded_columns": [],
        "tables": tables
        if tables is not None
        else [
            {
                "table": "observations",
                "version": 7,
                "projected_columns": ["state_vector", "caption", "payload_size"],
                "logical_columns": ["state_vector", "caption"],
                "row_count": 1000,
            }
        ],
    }
    base.update(overrides)
    return base


def _fake_metadata(*, total_rows=50_000):
    schema = pa.schema(
        [
            ("state_vector", pa.list_(pa.float32())),
            ("caption", pa.string()),
            ("payload_size", pa.int64()),
            (PAYLOAD_BLOB_COLUMN, pa.large_binary()),
        ]
    )

    def _fn(table, version):
        return TableMetadata(schema=schema, total_rows=total_rows)

    return _fn


# --------------------------------------------------------------------------- #
# Module-level planner behaviour
# --------------------------------------------------------------------------- #


def test_resolve_prewarm_database_from_db_uri():
    assert (
        resolve_prewarm_database(uri="db://robotics", connection_kind="lancedb_remote_db")
        == "robotics"
    )
    # Local / object-store lakes have no query node -> no database.
    assert resolve_prewarm_database(uri="/tmp/lake", connection_kind="local_path") is None
    assert (
        resolve_prewarm_database(uri="s3://bucket/lake", connection_kind="object_store_lancedb_oss")
        is None
    )


def test_plan_emits_valid_whole_table_wire_request():
    plan = build_page_cache_prewarm_plan(
        _request(),
        database="robotics",
        metadata_fn=_fake_metadata(),
        options=PrewarmPlannerOptions(concurrency=8),
    ).to_dict()

    assert plan["applicable"] is True
    assert plan["database"] == "robotics"
    assert len(plan["wire_requests"]) == 1
    wire = plan["wire_requests"][0]
    assert wire["db"] == "robotics"
    assert wire["table"] == "observations"
    assert wire["columns"] == ["state_vector", "caption", "payload_size"]
    assert wire["table_version"] == 7
    assert wire["concurrency"] == 8
    assert wire["schema_version"] == PAGE_CACHE_PREWARM_SPEC_SCHEMA_VERSION


def test_wire_request_never_carries_plan_executor_or_row_routing():
    """Architectural guardrail: no PE placement, no row/fragment ranges on the wire."""
    plan = build_page_cache_prewarm_plan(
        _request(),
        database="robotics",
        metadata_fn=_fake_metadata(),
        options=PrewarmPlannerOptions(),
    ).to_dict()
    wire = plan["wire_requests"][0]
    assert set(wire) <= _ALLOWED_WIRE_KEYS, set(wire) - _ALLOWED_WIRE_KEYS
    assert not (_FORBIDDEN_WIRE_KEYS & set(wire))


def test_advisory_estimate_reports_over_warm_ratio_and_bytes():
    plan = build_page_cache_prewarm_plan(
        _request(),
        database="robotics",
        metadata_fn=_fake_metadata(total_rows=50_000),
        options=PrewarmPlannerOptions(),
    ).to_dict()
    estimate = plan["tables"][0]["estimate"]
    # Whole-table prewarm warms 50k rows though training reads only 1k -> 50x.
    assert estimate["selected_rows"] == 1000
    assert estimate["total_rows"] == 50_000
    assert estimate["over_warm_ratio"] == 50.0
    assert estimate["source"] == "query-node"
    # No fragment-level reasoning anywhere in the estimate.
    assert not (_FORBIDDEN_ESTIMATE_KEYS & set(estimate))
    # state_vector (list) + caption (string) are variable-width; payload_size int64=8B.
    by_col = {c["column"]: c for c in estimate["columns"]}
    assert by_col["payload_size"]["basis"] == "schema-width"
    assert by_col["payload_size"]["bytes_per_row"] == 8.0
    assert by_col["caption"]["basis"] == "configured-variable"
    assert plan["metrics"]["over_warm_ratio"] == 50.0
    assert not (_FORBIDDEN_ESTIMATE_KEYS & set(plan["metrics"]))


def test_heavy_columns_estimated_separately_from_metadata():
    tables = [
        {
            "table": "observations",
            "version": 3,
            "projected_columns": ["caption", PAYLOAD_BLOB_COLUMN],
            "logical_columns": ["caption", "payload"],
            "row_count": 100,
        }
    ]
    plan = build_page_cache_prewarm_plan(
        _request(tables=tables),
        database="robotics",
        heavy_columns=(PAYLOAD_BLOB_COLUMN,),
        metadata_fn=_fake_metadata(total_rows=100),
        options=PrewarmPlannerOptions(heavy_bytes_per_row=4096),
    ).to_dict()
    estimate = plan["tables"][0]["estimate"]
    assert estimate["heavy_bytes"] == 4096 * 100
    assert estimate["metadata_bytes"] > 0
    assert estimate["metadata_bytes"] != estimate["heavy_bytes"]
    heavy = next(c for c in estimate["columns"] if c["column"] == PAYLOAD_BLOB_COLUMN)
    assert heavy["kind"] == "heavy"
    assert heavy["basis"] == "configured-heavy"


def test_heavy_bytes_unavailable_without_configured_rate():
    tables = [
        {
            "table": "observations",
            "version": 3,
            "projected_columns": [PAYLOAD_BLOB_COLUMN],
            "logical_columns": ["payload"],
            "row_count": 100,
        }
    ]
    plan = build_page_cache_prewarm_plan(
        _request(tables=tables),
        database="robotics",
        heavy_columns=(PAYLOAD_BLOB_COLUMN,),
        metadata_fn=_fake_metadata(total_rows=100),
        options=PrewarmPlannerOptions(),  # no heavy_bytes_per_row
    ).to_dict()
    heavy = plan["tables"][0]["estimate"]["columns"][0]
    assert heavy["basis"] == "unavailable"
    assert heavy["estimated_bytes"] == 0


def test_repeated_planning_is_deterministic():
    args = dict(
        database="robotics",
        metadata_fn=_fake_metadata(),
        options=PrewarmPlannerOptions(concurrency=4),
    )
    first = build_page_cache_prewarm_plan(_request(), **args).to_dict()
    second = build_page_cache_prewarm_plan(_request(), **args).to_dict()
    assert first["prewarm_id"] == second["prewarm_id"]
    assert first["wire_requests"] == second["wire_requests"]


def test_plan_not_applicable_without_query_node():
    plan = build_page_cache_prewarm_plan(
        _request(),
        database=None,  # local / object-store lake
        metadata_fn=_fake_metadata(),
        options=PrewarmPlannerOptions(),
    ).to_dict()
    assert plan["applicable"] is False
    assert "query-node" in plan["reason"]
    # Advisory estimates are still produced so a researcher can gauge cost.
    assert plan["tables"][0]["estimate"]["total_rows"] == 50_000


def test_estimate_degrades_when_metadata_unavailable():
    plan = build_page_cache_prewarm_plan(
        _request(),
        database="robotics",
        metadata_fn=lambda table, version: None,
        options=PrewarmPlannerOptions(),
    ).to_dict()
    estimate = plan["tables"][0]["estimate"]
    assert estimate["total_rows"] is None
    assert estimate["over_warm_ratio"] is None
    assert estimate["source"] == "unavailable"
    # Wire request is still valid even without any advisory metadata.
    assert plan["wire_requests"][0]["table"] == "observations"


# --------------------------------------------------------------------------- #
# End-to-end through a training dataset (Enterprise query-node backend)
# --------------------------------------------------------------------------- #


def _enterprise_lake(tmp_path):
    from test_native_training_dataset import _mark_enterprise_lake, _training_lake

    lake = _training_lake(tmp_path / "robot.lance")
    # Enterprise db:// query-node connection. The advisory estimator only ever reads
    # schema + row count (query-node-safe metadata) -- never fragments or a PE.
    _mark_enterprise_lake(lake)
    return lake


def test_dataset_page_cache_prewarm_plan_end_to_end(tmp_path):
    lake = _enterprise_lake(tmp_path)
    dataset = lake.training.dataset(
        "demo-v1",
        columns=["observation_id", "state_vector"],
        backend="enterprise",
        cache_policy="epoch",
    )
    plan = dataset.page_cache_prewarm_plan(concurrency=16)

    assert plan["applicable"] is True
    assert plan["database"] == "robotics"
    wire = plan["wire_requests"][0]
    assert wire["db"] == "robotics"
    assert wire["table"] == "observations"
    assert wire["concurrency"] == 16
    assert wire["table_version"] is not None
    # Guardrail again on the live path.
    assert set(wire) <= _ALLOWED_WIRE_KEYS
    assert not (_FORBIDDEN_WIRE_KEYS & set(wire))
    # Advisory estimate came from query-node metadata (schema + row count), no fragments.
    estimate = plan["tables"][0]["estimate"]
    assert estimate["total_rows"] is not None
    assert not (_FORBIDDEN_ESTIMATE_KEYS & set(estimate))


def test_lake_training_page_cache_prewarm_plan_helper(tmp_path):
    lake = _enterprise_lake(tmp_path)
    plan = lake.training.page_cache_prewarm_plan(
        "demo-v1",
        columns=["observation_id"],
        cache_policy="snapshot",
    )
    assert plan["applicable"] is True
    assert plan["scope"] == "snapshot"
    assert plan["wire_requests"][0]["db"] == "robotics"


def test_plan_reports_not_requested_for_local_backend(tmp_path):
    from test_native_training_dataset import _training_lake

    lake = _training_lake(tmp_path / "robot.lance")
    dataset = lake.training.dataset("demo-v1", columns=["observation_id"])
    plan = dataset.page_cache_prewarm_plan()
    # Local backend never requests prewarm -> plan is inapplicable, no wire requests.
    assert plan["applicable"] is False
    assert plan["wire_requests"] == []
