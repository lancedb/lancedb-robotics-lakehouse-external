"""Unit tests for predicate-index selectivity telemetry + recommendations (0137)."""

from __future__ import annotations

import pytest

from lancedb_robotics.connections import LakeCapabilities, LakeConnectionSpec
from lancedb_robotics.predicate_index_telemetry import (
    ACTION_CREATE,
    ACTION_NONE,
    ACTION_PROMOTE_JSONB,
    ACTION_REFRESH,
    INDEX_ABSENT_COLUMN,
    INDEX_FAILED,
    INDEX_PRESENT,
    INDEX_UNINDEXED,
    INDEX_UNSUPPORTED,
    OUTCOME_ADVISORY,
    OUTCOME_APPLIED,
    OUTCOME_SKIPPED,
    OUTCOME_SUPPRESSED,
    RecommendationPolicy,
    aggregate_predicate_observations,
    apply_index_recommendations,
    build_predicate_telemetry,
    extract_observations_from_reports,
    extract_predicate_observations,
    normalize_index_status,
    recommend_from_aggregates,
    recommend_from_reports,
    recommendation_from_dict,
)
from lancedb_robotics.scalar_index_jobs import InMemoryScalarIndexJobStore

# --------------------------------------------------------------------------- #
# Manifest / report fixtures                                                   #
# --------------------------------------------------------------------------- #

_UNINDEXED_REASON = (
    "no scalar index is present for this predicate; predicate pushdown remains available"
)
_UNSUPPORTED_REASON = (
    "backend does not expose create_scalar_index; predicate pushdown remains available"
)


def _predicate(
    column,
    *,
    status="skipped",
    reason=_UNINDEXED_REASON,
    used_in_filter=True,
    role="filter",
    num_rows=500_000,
    index_type="BTREE",
    job_status=None,
):
    payload = {
        "table": "aligned_ticks",
        "column": column,
        "status": status,
        "reason": reason,
        "used_in_filter": used_in_filter,
        "predicate_role": role,
        "num_rows": num_rows,
        "index_type": index_type,
    }
    if job_status is not None:
        payload["job_status"] = job_status
    return payload


def _manifest(
    *,
    selected=1_200,
    total=500_000,
    backend="local",
    predicates=None,
    statuses=("ok",),
):
    if predicates is None:
        predicates = [_predicate("alignment_id", num_rows=total)]
    return {
        "output_table": "aligned_ticks",
        "backend": {"resolved_backend": backend},
        "total_ticks": total,
        "selected_ticks": selected,
        "quality_policy": {"statuses": list(statuses)},
        "predicate_indexes": predicates,
    }


def _report(manifest):
    return {"predicate_telemetry": build_predicate_telemetry(manifest)}


def _reports(manifest, n=4):
    return [_report(manifest) for _ in range(n)]


# --------------------------------------------------------------------------- #
# normalize_index_status                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"status": "built"}, INDEX_PRESENT),
        ({"status": "already_present"}, INDEX_PRESENT),
        ({"status": "skipped", "reason": _UNINDEXED_REASON}, INDEX_UNINDEXED),
        ({"status": "skipped", "reason": _UNSUPPORTED_REASON}, INDEX_UNSUPPORTED),
        ({"status": "failed", "reason": "no 'foo' column to index in table 'aligned_ticks'"}, INDEX_ABSENT_COLUMN),
        ({"status": "failed", "reason": "build blew up"}, INDEX_FAILED),
        ({"status": ""}, INDEX_UNINDEXED),
    ],
)
def test_normalize_index_status(payload, expected):
    assert normalize_index_status(payload) == expected


def test_skipped_splits_unsupported_from_merely_unindexed():
    # The whole design hinges on this: describe_scalar_indexes returns "skipped"
    # for BOTH an unsupported backend and a buildable-but-unindexed column.
    assert normalize_index_status({"status": "skipped", "reason": _UNINDEXED_REASON}) != (
        normalize_index_status({"status": "skipped", "reason": _UNSUPPORTED_REASON})
    )


# --------------------------------------------------------------------------- #
# build_predicate_telemetry                                                    #
# --------------------------------------------------------------------------- #


def test_telemetry_section_shape_and_selectivity():
    section = build_predicate_telemetry(_manifest(selected=1_200, total=500_000))
    assert section["output_table"] == "aligned_ticks"
    assert section["backend_kind"] == "local"
    assert section["backend_supports_scalar_index"] is True
    assert section["total_rows"] == 500_000
    assert section["selected_rows"] == 1_200
    assert section["selectivity_fraction"] == pytest.approx(1_200 / 500_000)
    # alignment_id (typed) + stream_detail_json.status (jsonb) both used in filter.
    assert "alignment_id" in section["filter_columns"]
    assert "stream_detail_json.status" in section["filter_columns"]


def test_telemetry_derives_jsonb_path_predicate_from_quality_policy():
    section = build_predicate_telemetry(_manifest(statuses=("ok", "degraded")))
    jsonb = [p for p in section["predicates"] if p["column_kind"] == "jsonb_path"]
    assert len(jsonb) == 1
    assert jsonb[0]["column"] == "stream_detail_json.status"
    assert jsonb[0]["index_status"] == INDEX_UNINDEXED
    assert jsonb[0]["index_backed"] is False


def test_no_jsonb_predicate_when_no_stream_status_filter():
    section = build_predicate_telemetry(_manifest(statuses=()))
    assert all(p["column_kind"] == "typed" for p in section["predicates"])


def test_telemetry_never_leaks_literal_predicate_values():
    # A read filtering alignment_id='super-secret-id' must not record that value.
    manifest = _manifest()
    manifest["quality_policy"]["min_confidence"] = 0.97  # a literal filter parameter
    section = build_predicate_telemetry(manifest)
    blob = repr(section)
    assert "super-secret-id" not in blob
    assert "0.97" not in blob  # thresholds are filter *values*, never recorded


def test_unsupported_backend_flag_from_statuses():
    predicates = [_predicate("alignment_id", status="skipped", reason=_UNSUPPORTED_REASON)]
    section = build_predicate_telemetry(_manifest(predicates=predicates))
    assert section["backend_supports_scalar_index"] is False


# --------------------------------------------------------------------------- #
# extraction                                                                   #
# --------------------------------------------------------------------------- #


def test_extract_from_report_and_from_raw_manifest_agree():
    manifest = _manifest()
    from_report = extract_predicate_observations(_report(manifest))
    from_manifest = extract_predicate_observations(manifest)  # builds section on the fly
    assert {(o.column, o.column_kind) for o in from_report} == {
        (o.column, o.column_kind) for o in from_manifest
    }


def test_extract_from_loader_report_wrapper():
    manifest = _manifest()
    wrapper = {"loader_report": _report(manifest)}
    observations = extract_predicate_observations(wrapper)
    assert any(o.column == "alignment_id" for o in observations)


def test_extract_returns_empty_without_telemetry():
    assert extract_predicate_observations({"kind": "something-else"}) == []


def test_extract_from_reports_assigns_increasing_sequence():
    reports = _reports(_manifest(), n=3)
    observations = extract_observations_from_reports(reports)
    seqs = {o.column: [] for o in observations}
    for o in observations:
        seqs[o.column].append(o.sequence)
    assert seqs["alignment_id"] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# aggregation                                                                  #
# --------------------------------------------------------------------------- #


def test_aggregate_groups_and_counts_repeated_use():
    observations = extract_observations_from_reports(_reports(_manifest(), n=5))
    aggregates = {(a.table, a.column): a for a in aggregate_predicate_observations(observations)}
    agg = aggregates[("aligned_ticks", "alignment_id")]
    assert agg.observations == 5
    assert agg.filter_observations == 5
    assert agg.min_selectivity_fraction == pytest.approx(1_200 / 500_000)
    assert agg.max_total_rows == 500_000


def test_aggregate_latest_uses_sequence_not_input_order():
    # Newest observation (sequence=1) is index-backed; older (sequence=0) is not.
    unindexed = extract_predicate_observations(_report(_manifest()), sequence=1)
    indexed = extract_predicate_observations(
        _report(_manifest(predicates=[_predicate("alignment_id", status="already_present", reason=None)])),
        sequence=0,
    )
    # Feed newest first to prove ordering is by sequence, not list position.
    aggregates = aggregate_predicate_observations([*unindexed, *indexed])
    agg = next(a for a in aggregates if a.column == "alignment_id")
    assert agg.latest_index_backed is False  # sequence=1 wins
    assert agg.ever_index_backed is True


def test_aggregate_detects_stale_when_index_disappears():
    indexed = extract_predicate_observations(
        _report(_manifest(predicates=[_predicate("alignment_id", status="already_present", reason=None)])),
        sequence=0,
    )
    gone = extract_predicate_observations(_report(_manifest()), sequence=1)
    agg = next(
        a
        for a in aggregate_predicate_observations([*indexed, *gone])
        if a.column == "alignment_id"
    )
    assert agg.stale is True


def test_aggregate_estimated_unindexed_scan_rows():
    observations = extract_observations_from_reports(_reports(_manifest(total=500_000), n=3))
    agg = next(
        a for a in aggregate_predicate_observations(observations) if a.column == "alignment_id"
    )
    # 3 unindexed filtered reads x 500k rows each.
    assert agg.estimated_unindexed_scan_rows == 1_500_000


# --------------------------------------------------------------------------- #
# recommendations: the acceptance-criteria scenarios                          #
# --------------------------------------------------------------------------- #


def _by_column(recs):
    return {(r.table, r.column): r for r in recs}


def test_hot_selective_repeated_predicate_is_recommended():
    recs = _by_column(recommend_from_reports(_reports(_manifest(), n=4)))
    typed = recs[("aligned_ticks", "alignment_id")]
    assert typed.action == ACTION_CREATE
    assert typed.auto_applicable is True
    assert typed.index_type == "BTREE"


def test_jsonb_predicate_gets_promotion_advice_not_an_index():
    recs = _by_column(recommend_from_reports(_reports(_manifest(), n=4)))
    jsonb = recs[("aligned_ticks", "stream_detail_json.status")]
    assert jsonb.action == ACTION_PROMOTE_JSONB
    assert jsonb.column_kind == "jsonb_path"
    assert jsonb.auto_applicable is False  # never auto-index a JSONB sub-path
    assert jsonb.index_type is None


def test_small_table_is_not_recommended_by_default():
    recs = _by_column(recommend_from_reports(_reports(_manifest(selected=50, total=1_000), n=4)))
    typed = recs[("aligned_ticks", "alignment_id")]
    assert typed.action == ACTION_NONE
    assert "small-table" in typed.guardrails
    assert typed.forceable is True  # an index CAN be built; just not recommended


def test_low_selectivity_predicate_is_not_recommended_by_default():
    recs = _by_column(recommend_from_reports(_reports(_manifest(selected=450_000, total=500_000), n=4)))
    typed = recs[("aligned_ticks", "alignment_id")]
    assert typed.action == ACTION_NONE
    assert "low-selectivity" in typed.guardrails


def test_rarely_used_predicate_is_not_recommended_by_default():
    recs = _by_column(recommend_from_reports(_reports(_manifest(), n=1)))  # 1 observation
    typed = recs[("aligned_ticks", "alignment_id")]
    assert typed.action == ACTION_NONE
    assert "insufficient-observations" in typed.guardrails


def test_unsupported_backend_suppresses_recommendation():
    predicates = [
        _predicate("alignment_id", status="skipped", reason=_UNSUPPORTED_REASON),
    ]
    recs = _by_column(
        recommend_from_reports(_reports(_manifest(predicates=predicates, statuses=()), n=4))
    )
    typed = recs[("aligned_ticks", "alignment_id")]
    assert typed.action == ACTION_NONE
    assert "unsupported-backend" in typed.guardrails
    assert typed.forceable is False  # cannot force what the backend cannot build


def test_already_indexed_predicate_is_a_noop():
    predicates = [_predicate("alignment_id", status="already_present", reason=None)]
    recs = _by_column(recommend_from_reports(_reports(_manifest(predicates=predicates), n=4)))
    typed = recs[("aligned_ticks", "alignment_id")]
    assert typed.action == ACTION_NONE
    assert "already served" in typed.reason


def test_stale_index_gets_refresh_recommendation():
    indexed = extract_predicate_observations(
        _report(_manifest(predicates=[_predicate("alignment_id", status="already_present", reason=None)])),
        sequence=0,
    )
    gone = []
    for seq in range(1, 5):
        gone.extend(extract_predicate_observations(_report(_manifest()), sequence=seq))
    recs = _by_column(
        recommend_from_aggregates(aggregate_predicate_observations([*indexed, *gone]))
    )
    typed = recs[("aligned_ticks", "alignment_id")]
    assert typed.action == ACTION_REFRESH
    assert typed.auto_applicable is True


def test_absent_column_is_not_indexable():
    predicates = [
        _predicate(
            "ghost_col",
            status="failed",
            reason="no 'ghost_col' column to index in table 'aligned_ticks'",
        )
    ]
    recs = _by_column(recommend_from_reports(_reports(_manifest(predicates=predicates), n=4)))
    typed = recs[("aligned_ticks", "ghost_col")]
    assert typed.action == ACTION_NONE
    assert "absent-column" in typed.guardrails


def test_recommendations_ranked_most_actionable_first():
    recs = recommend_from_reports(_reports(_manifest(), n=8))
    actions = [r.action for r in recs]
    # create_scalar_index (actionable) ranks before promote_jsonb before none.
    assert actions.index(ACTION_CREATE) < actions.index(ACTION_PROMOTE_JSONB)


def test_policy_thresholds_can_be_relaxed():
    manifest = _manifest(selected=50, total=1_000)  # small table
    strict = _by_column(recommend_from_reports(_reports(manifest, n=4)))
    relaxed = _by_column(
        recommend_from_reports(
            _reports(manifest, n=4),
            policy=RecommendationPolicy(min_total_rows=100, min_observations=2),
        )
    )
    assert strict[("aligned_ticks", "alignment_id")].action == ACTION_NONE
    assert relaxed[("aligned_ticks", "alignment_id")].action == ACTION_CREATE


def test_high_confidence_needs_many_selective_reads():
    many = recommend_from_reports(_reports(_manifest(selected=100, total=500_000), n=8))
    typed = _by_column(many)[("aligned_ticks", "alignment_id")]
    assert typed.confidence == "high"


# --------------------------------------------------------------------------- #
# recommendation_from_dict round-trip                                         #
# --------------------------------------------------------------------------- #


def test_recommendation_round_trips_through_dict():
    original = recommend_from_reports(_reports(_manifest(), n=4))[0]
    restored = recommendation_from_dict(original.to_dict())
    assert restored == original


def test_recommendation_from_dict_requires_core_fields():
    with pytest.raises(ValueError):
        recommendation_from_dict({"column": "c", "action": "none"})  # missing table


# --------------------------------------------------------------------------- #
# apply: route safe recommendations into 0136 jobs                            #
# --------------------------------------------------------------------------- #


class _AsyncLake:
    """Async, buildable lake whose submit hook records a pending remote job."""

    def __init__(self, store):
        self.connection_spec = LakeConnectionSpec(
            kind="lancedb_remote_db",
            uri="db://robotics",
            display_uri="db://robotics",
            capabilities=LakeCapabilities(server_side_query=True, index_management=True),
        )
        self.scalar_index_job_store = store
        self.submitted = []

    def scalar_index_job_submit(self, request):
        self.submitted.append(request)
        return {"status": "active"}

    def table(self, _name):
        raise LookupError("no table in this fake")


def test_apply_requests_jobs_for_safe_typed_recommendations():
    store = InMemoryScalarIndexJobStore()
    lake = _AsyncLake(store)
    recs = recommend_from_reports(_reports(_manifest(selected=100, total=500_000), n=8))
    outcomes = {o["recommendation"]["column"]: o for o in apply_index_recommendations(lake, recs)}
    assert outcomes["alignment_id"]["outcome"] == OUTCOME_APPLIED
    assert outcomes["alignment_id"]["job"] is not None
    # A real job row was persisted.
    assert any(job.column == "alignment_id" for job in store.list())


def test_apply_treats_jsonb_promotion_as_advisory():
    store = InMemoryScalarIndexJobStore()
    lake = _AsyncLake(store)
    recs = recommend_from_reports(_reports(_manifest(selected=100, total=500_000), n=8))
    outcomes = {o["recommendation"]["column"]: o for o in apply_index_recommendations(lake, recs)}
    assert outcomes["stream_detail_json.status"]["outcome"] == OUTCOME_ADVISORY
    assert outcomes["stream_detail_json.status"]["job"] is None
    # No JSONB path was ever submitted for a build.
    assert all("stream_detail_json" not in r.get("column", "") for r in lake.submitted)


def test_apply_suppress_wins():
    store = InMemoryScalarIndexJobStore()
    lake = _AsyncLake(store)
    recs = recommend_from_reports(_reports(_manifest(selected=100, total=500_000), n=8))
    outcomes = {
        o["recommendation"]["column"]: o
        for o in apply_index_recommendations(lake, recs, suppress=["aligned_ticks.alignment_id"])
    }
    assert outcomes["alignment_id"]["outcome"] == OUTCOME_SUPPRESSED
    assert store.list() == []


def test_apply_skips_below_confidence_bar_unless_forced():
    store = InMemoryScalarIndexJobStore()
    lake = _AsyncLake(store)
    # 4 selective reads -> medium confidence (below the default "high" bar).
    recs = recommend_from_reports(_reports(_manifest(selected=1_200, total=500_000), n=4))
    default = {o["recommendation"]["column"]: o for o in apply_index_recommendations(lake, recs)}
    assert default["alignment_id"]["outcome"] == OUTCOME_SKIPPED

    forced = {
        o["recommendation"]["column"]: o
        for o in apply_index_recommendations(_AsyncLake(InMemoryScalarIndexJobStore()), recs, force=True)
    }
    assert forced["alignment_id"]["outcome"] == OUTCOME_APPLIED


def test_apply_force_applies_guardrail_suppressed_forceable_columns():
    store = InMemoryScalarIndexJobStore()
    lake = _AsyncLake(store)
    # Small table -> action none but forceable.
    recs = recommend_from_reports(_reports(_manifest(selected=50, total=1_000), n=4))
    forced = {
        o["recommendation"]["column"]: o
        for o in apply_index_recommendations(lake, recs, force=True)
    }
    assert forced["alignment_id"]["outcome"] == OUTCOME_APPLIED


def test_apply_never_forces_unsupported_backend_rec():
    store = InMemoryScalarIndexJobStore()
    lake = _AsyncLake(store)
    predicates = [_predicate("alignment_id", status="skipped", reason=_UNSUPPORTED_REASON)]
    recs = recommend_from_reports(_reports(_manifest(predicates=predicates, statuses=()), n=4))
    outcomes = {
        o["recommendation"]["column"]: o
        for o in apply_index_recommendations(lake, recs, force=True)
    }
    assert outcomes["alignment_id"]["outcome"] == OUTCOME_SKIPPED  # forceable is False


# --------------------------------------------------------------------------- #
# Integration: a real aligned loader report carries usable telemetry          #
# --------------------------------------------------------------------------- #


def test_real_aligned_report_emits_predicate_telemetry(tmp_path):
    import test_aligned_training_dataset as aligned_mod

    lake, _view = aligned_mod._aligned_training_lake(tmp_path / "aligned.lance")
    dataset = lake.training.aligned_dataset(name="policy_bridge")
    report = dataset.manifest.to_dict()["loader_report"]

    section = report["predicate_telemetry"]
    assert section["output_table"] == "aligned_ticks"
    columns = {p["column"] for p in section["predicates"]}
    assert "alignment_id" in columns  # the hot filter predicate

    # The recommender consumes the persisted-report shape end to end.
    observations = extract_predicate_observations(report)
    assert any(o.column == "alignment_id" and o.used_in_filter for o in observations)


def test_lake_facade_recommend_reads_persisted_reports(tmp_path):
    import test_aligned_training_dataset as aligned_mod

    lake, _view = aligned_mod._aligned_training_lake(tmp_path / "aligned.lance")
    dataset = lake.training.aligned_dataset(name="policy_bridge")
    # Persist several DISTINCT aligned reports so the recommender has real
    # repeated-use evidence (reports are idempotent by content digest, so the
    # bodies must differ to produce separate catalog rows).
    from lancedb_robotics.run_manifests import record_training_report

    base = dataset.manifest.to_dict()["loader_report"]
    for i in range(4):
        body = {**base, "run": {"training_run_id": f"run-{i}"}}
        record_training_report(lake, report=body)

    recs = lake.training.recommend_predicate_indexes(
        min_observations=1, min_total_rows=0, max_selectivity_fraction=1.0
    )
    # Facade returns JSON-able dicts (not dataclasses) and finds the hot predicate.
    assert recs and all(isinstance(rec, dict) for rec in recs)
    assert any(rec["column"] == "alignment_id" for rec in recs)


def _record_distinct_aligned_reports(lake, base, n):
    from lancedb_robotics.run_manifests import record_training_report

    ids = []
    for i in range(n):
        body = {**base, "run": {"training_run_id": f"run-{i:03d}"}}
        ids.append(record_training_report(lake, report=body).report_id)
    return ids


def test_recent_training_reports_is_bounded_and_oldest_first(tmp_path):
    import test_aligned_training_dataset as aligned_mod

    from lancedb_robotics.run_manifests import recent_training_reports

    lake, _view = aligned_mod._aligned_training_lake(tmp_path / "aligned.lance")
    dataset = lake.training.aligned_dataset(name="policy_bridge")
    base = dataset.manifest.to_dict()["loader_report"]
    recorded = set(_record_distinct_aligned_reports(lake, base, 6))

    window = recent_training_reports(lake, loader_kind="aligned-training", limit=4)
    # Bounded to `limit` (never the whole catalog), all valid, oldest-first.
    assert len(window) == 4
    assert all(record.report_id in recorded for record in window)
    created = [r.created_at for r in window if r.created_at is not None]
    assert created == sorted(created)
    # limit >= catalog size returns everything.
    assert len(recent_training_reports(lake, loader_kind="aligned-training", limit=99)) == 6


def test_recommend_does_not_fan_out_one_scan_per_report(tmp_path, monkeypatch):
    # Guards against reintroducing the O(limit x N) per-id fetch loop: if the
    # recommender still called get_training_report per id, this would raise.
    import test_aligned_training_dataset as aligned_mod

    import lancedb_robotics.run_manifests as rm

    lake, _view = aligned_mod._aligned_training_lake(tmp_path / "aligned.lance")
    dataset = lake.training.aligned_dataset(name="policy_bridge")
    base = dataset.manifest.to_dict()["loader_report"]
    _record_distinct_aligned_reports(lake, base, 4)

    def _boom(*args, **kwargs):
        raise AssertionError("recommender must not fetch reports one id at a time")

    monkeypatch.setattr(rm, "get_training_report", _boom)
    recs = lake.training.recommend_predicate_indexes(
        min_observations=1, min_total_rows=0, max_selectivity_fraction=1.0
    )
    assert any(rec["column"] == "alignment_id" for rec in recs)
