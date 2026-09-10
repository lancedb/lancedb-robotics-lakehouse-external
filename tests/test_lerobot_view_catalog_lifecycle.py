"""View-catalog keyset paging, latest-pointer resolve, and dedup compaction (0507).

Three surfaces over the 0490/0491 published-view catalog, each pinned here with
its degradation path (SKILLS.md: pin the guardrail and the mechanism, not just
the answer):

* ``list_view_pages``: stable descending ``(created_at, view_id)`` keyset
  cursor -- exact newest-first walk, cursor stability under concurrent
  publishes, the bounded-heap fallback when the backend cannot order, and
  typed errors for garbage cursors / out-of-bounds page sizes.
* ``get_view(repo_id=...)`` resolve chain: ``lerobot_view_latest`` pointer
  point-read first (newest-wins CAS verified against a real out-of-order
  update), then ordered top-1, then the loudly-warned guarded scan; stale
  pointers are ignored, and publish degrades loudly (never fails) when the
  pointer cannot be written.
* ``compact_view_catalog``: tier-1 strictly-older deletes, tier-2 re-add+marker
  for byte-identical ties, distinct keys preserved, idempotent re-runs, and the
  ``lake maintain`` report section.
"""

import warnings
from datetime import timedelta

import pyarrow as pa
import pytest
from test_lerobot_facade import _two_episode_lake

from lancedb_robotics.lerobot_facade import (
    CanonicalVectorMapping,
    ViewError,
    compact_view_catalog,
    get_view,
    list_view_pages,
    list_views,
    materialize_view,
    publish_view,
)
from lancedb_robotics.lerobot_facade import views as views_mod
from lancedb_robotics.lerobot_facade.views import (
    VIEW_FILES_TABLE,
    VIEW_LATEST_TABLE,
    VIEWS_TABLE,
    _latest_pointer_view_id,
    _update_latest_pointer,
)
from lancedb_robotics.schemas import (
    LEROBOT_VIEW_FILES_SCHEMA,
    LEROBOT_VIEW_LATEST_SCHEMA,
    LEROBOT_VIEWS_SCHEMA,
)

_MAPPING = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))


def _publish(lake, **overrides):
    kwargs = dict(
        repo_id="acme/pick-place-v1",
        fps=20,
        mapping=_MAPPING,
        name="facade_view",
        robot_type="test-arm",
        created_by="tests",
    )
    kwargs.update(overrides)
    return publish_view(lake, **kwargs)


def _publish_many(lake, *, repos=("acme/a", "acme/b"), fps_values=(20, 30)):
    """Publish one view per (repo, fps); distinct fps => distinct view ids."""
    published = []
    for repo in repos:
        for fps in fps_values:
            published.append(_publish(lake, repo_id=repo, fps=fps))
    return published


def _walk_pages(lake, *, repo_id=None, page_size=2):
    ids, cursor, pages = [], None, 0
    while True:
        page = list_view_pages(lake, repo_id=repo_id, page_size=page_size, cursor=cursor)
        assert len(page.rows) <= page_size
        ids.extend(row["view_id"] for row in page.rows)
        pages += 1
        assert pages < 50, "paging did not terminate"
        if not page.next_cursor:
            return ids
        cursor = page.next_cursor


def _table_rows(lake, table, columns):
    return lake.table(table).search().select(columns).to_arrow().to_pylist()


def _full_row(lake, table, where):
    rows = lake.table(table).search().where(where).limit(1).to_arrow().to_pylist()
    assert rows, where
    return dict(rows[0])


# ---------------------------------------------------------------------------
# Keyset paging
# ---------------------------------------------------------------------------


def test_list_view_pages_walks_catalog_exactly(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish_many(lake, repos=("acme/a", "acme/b", "acme/c"))

    expected = [row["view_id"] for row in list_views(lake, limit=100)]
    assert len(expected) == 6
    assert _walk_pages(lake, page_size=2) == expected
    assert _walk_pages(lake, page_size=4) == expected

    scoped = [row["view_id"] for row in list_views(lake, repo_id="acme/b", limit=100)]
    assert len(scoped) == 2
    assert _walk_pages(lake, repo_id="acme/b", page_size=1) == scoped


def test_list_view_pages_cursor_is_stable_across_new_publishes(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish_many(lake, repos=("acme/a",), fps_values=(10, 20, 30, 40))
    expected = [row["view_id"] for row in list_views(lake, limit=100)]

    first = list_view_pages(lake, page_size=2)
    assert [row["view_id"] for row in first.rows] == expected[:2]

    # Newer publishes land strictly before the cursor position and never shift
    # the remainder of an in-flight walk.
    _publish_many(lake, repos=("acme/z",), fps_values=(50, 60))
    second = list_view_pages(lake, page_size=2, cursor=first.next_cursor)
    assert [row["view_id"] for row in second.rows] == expected[2:4]
    assert not second.next_cursor


def test_list_view_pages_heap_fallback_warns_and_matches(tmp_path, monkeypatch):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish_many(lake, repos=("acme/a", "acme/b"))
    expected = _walk_pages(lake, page_size=2)

    monkeypatch.setattr(views_mod, "_list_view_page_ordered", lambda *a, **k: None)
    with pytest.warns(RuntimeWarning, match="bounded heap"):
        page = list_view_pages(lake, page_size=2)
    assert [row["view_id"] for row in page.rows] == expected[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert _walk_pages(lake, page_size=2) == expected
        assert _walk_pages(lake, page_size=3) == expected


def test_list_view_pages_typed_errors(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish(lake)
    with pytest.raises(ViewError, match="cursor"):
        list_view_pages(lake, cursor="not-a-cursor")
    with pytest.raises(ViewError, match="page_size"):
        list_view_pages(lake, page_size=0)
    with pytest.raises(ViewError, match="page_size"):
        list_view_pages(lake, page_size=views_mod._MAX_PAGE_SIZE + 1)


def test_list_views_guard_message_points_at_paging(tmp_path, monkeypatch):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish_many(lake, repos=("acme/a", "acme/b"))
    monkeypatch.setattr(views_mod, "_MAX_CATALOG_SCAN_ROWS", 2)
    with pytest.raises(ViewError, match="list_view_pages"):
        list_views(lake)
    # The paged surface keeps working past the unpaged guard.
    assert len(_walk_pages(lake, page_size=2)) == 4


# ---------------------------------------------------------------------------
# Latest-pointer resolve
# ---------------------------------------------------------------------------


def test_get_view_resolves_via_latest_pointer(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    first = _publish(lake, fps=20)
    second = _publish(lake, fps=30)
    assert first.latest_pointer_updated and second.latest_pointer_updated

    assert _latest_pointer_view_id(lake, "acme/pick-place-v1") == second.view_id
    assert get_view(lake, repo_id="acme/pick-place-v1")["view_id"] == second.view_id


def test_latest_pointer_newest_wins_against_out_of_order_update(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    first = _publish(lake, fps=20)
    second = _publish(lake, fps=30)

    # A late-arriving pointer write for the OLDER view must not regress the
    # pointer (the conditional newest-wins upsert, not last-writer-wins).
    assert _update_latest_pointer(
        lake,
        repo_id="acme/pick-place-v1",
        view_id=first.view_id,
        view_created_at=first.created_at,
    )
    assert _latest_pointer_view_id(lake, "acme/pick-place-v1") == second.view_id


def test_get_view_ignores_stale_pointer(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    second = _publish(lake, fps=30)

    # Corrupt the pointer to a header that does not exist.
    table = lake.table(VIEW_LATEST_TABLE)
    row = _full_row(lake, VIEW_LATEST_TABLE, "repo_id = 'acme/pick-place-v1'")
    row["view_id"] = "lrv-0000000000000000"
    table.merge_insert("repo_id").when_matched_update_all().when_not_matched_insert_all().execute(
        pa.Table.from_pylist([row], schema=LEROBOT_VIEW_LATEST_SCHEMA)
    )

    with pytest.warns(RuntimeWarning, match="stale pointer"):
        resolved = get_view(lake, repo_id="acme/pick-place-v1")
    assert resolved["view_id"] == second.view_id


def test_get_view_resolves_without_pointer_table(tmp_path, monkeypatch):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    first = _publish(lake, fps=20)
    second = _publish(lake, fps=30)
    lake._db.drop_table(VIEW_LATEST_TABLE)

    # Ordered top-1 fallback (no pointer row, no warning path asserted here).
    assert get_view(lake, repo_id="acme/pick-place-v1")["view_id"] == second.view_id

    # Guarded-scan last resort when the backend cannot order either.
    monkeypatch.setattr(views_mod, "_latest_pointer_view_id", lambda *a, **k: None)
    monkeypatch.setattr(views_mod, "_newest_view_id_ordered", lambda *a, **k: None)
    with pytest.warns(RuntimeWarning, match="guarded bounded scan"):
        resolved = get_view(lake, repo_id="acme/pick-place-v1")
    assert resolved["view_id"] == second.view_id
    assert first.view_id != second.view_id


def test_publish_degrades_loudly_when_pointer_unwritable(tmp_path, monkeypatch):
    lake = _two_episode_lake(tmp_path / "robot.lance")

    def _refuse(_lake):
        raise RuntimeError("pointer table refused")

    monkeypatch.setattr(views_mod, "_ensure_latest_table", _refuse)
    with pytest.warns(RuntimeWarning, match="could not update"):
        published = _publish(lake)
    assert published.latest_pointer_updated is False
    # The publish itself is complete and resolvable through the fallbacks.
    assert get_view(lake, repo_id="acme/pick-place-v1")["view_id"] == published.view_id


# ---------------------------------------------------------------------------
# Duplicate compaction
# ---------------------------------------------------------------------------


def _seed_duplicates(lake, published):
    """Physically duplicate one header (older copy), one file row (exact tie),
    and the pointer row (older view_created_at), mimicking the merge_insert
    append race."""
    header = _full_row(lake, VIEWS_TABLE, f"view_id = '{published.view_id}'")
    older_header = dict(header)
    older_header["created_at"] = header["created_at"] - timedelta(microseconds=1)
    lake.table(VIEWS_TABLE).add(
        pa.Table.from_pylist([older_header], schema=LEROBOT_VIEWS_SCHEMA)
    )

    file_row = _full_row(
        lake, VIEW_FILES_TABLE, f"file_id = '{published.view_id}/meta/info.json'"
    )
    lake.table(VIEW_FILES_TABLE).add(
        pa.Table.from_pylist([dict(file_row)], schema=LEROBOT_VIEW_FILES_SCHEMA)
    )

    pointer = _full_row(lake, VIEW_LATEST_TABLE, f"repo_id = '{published.repo_id}'")
    older_pointer = dict(pointer)
    older_pointer["view_created_at"] = pointer["view_created_at"] - timedelta(seconds=1)
    lake.table(VIEW_LATEST_TABLE).add(
        pa.Table.from_pylist([older_pointer], schema=LEROBOT_VIEW_LATEST_SCHEMA)
    )


def _distinct(rows, key):
    return {str(row[key]) for row in rows}


def test_compact_view_catalog_collapses_duplicates(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    _seed_duplicates(lake, published)

    headers_before = _table_rows(lake, VIEWS_TABLE, ["view_id"])
    files_before = _table_rows(lake, VIEW_FILES_TABLE, ["file_id"])
    assert len(headers_before) == 2  # duplicate seeded

    dry = compact_view_catalog(lake, dry_run=True)
    assert dry.dry_run and dry.rows_deleted >= 3
    assert len(_table_rows(lake, VIEWS_TABLE, ["view_id"])) == 2  # untouched

    report = compact_view_catalog(lake)
    by_table = {item.table: item for item in report.tables}
    assert by_table[VIEWS_TABLE].status == "compacted"
    assert by_table[VIEWS_TABLE].rows_deleted == 1
    assert by_table[VIEW_FILES_TABLE].canonical_rows_readded == 1  # exact tie
    assert by_table[VIEW_LATEST_TABLE].rows_deleted == 1
    assert report.converged

    headers_after = _table_rows(lake, VIEWS_TABLE, ["view_id", "created_at"])
    assert len(headers_after) == 1  # physical collapse
    assert _distinct(headers_after, "view_id") == _distinct(headers_before, "view_id")
    files_after = _table_rows(lake, VIEW_FILES_TABLE, ["file_id"])
    assert len(files_after) == len(set(r["file_id"] for r in files_before))
    assert _distinct(files_after, "file_id") == _distinct(files_before, "file_id")

    # The kept header is the newest copy; resolve, pointer, and materialization
    # all still work over the compacted catalog.
    assert get_view(lake, repo_id=published.repo_id)["view_id"] == published.view_id
    assert _latest_pointer_view_id(lake, published.repo_id) == published.view_id
    dest = materialize_view(
        lake, get_view(lake, view_id=published.view_id), tmp_path / "m", lake_uri=lake.uri
    )
    assert (dest / "meta" / "info.json").exists()

    # Idempotent: a re-run finds nothing.
    again = compact_view_catalog(lake)
    assert again.rows_deleted == 0 and again.converged


def test_compact_tier2_exact_tie_readds_marker_row(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    header = _full_row(lake, VIEWS_TABLE, f"view_id = '{published.view_id}'")
    lake.table(VIEWS_TABLE).add(
        pa.Table.from_pylist([dict(header)], schema=LEROBOT_VIEWS_SCHEMA)
    )

    report = compact_view_catalog(lake)
    by_table = {item.table: item for item in report.tables}
    assert by_table[VIEWS_TABLE].canonical_rows_readded == 1
    survivors = _table_rows(lake, VIEWS_TABLE, ["view_id", "created_at"])
    assert len(survivors) == 1
    # 0135 marker shape: the survivor is stamped strictly newer than the tie.
    assert survivors[0]["created_at"] == header["created_at"] + timedelta(microseconds=1)
    assert get_view(lake, view_id=published.view_id)["view_id"] == published.view_id


def test_compact_unordered_fallback_and_bound(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import view_lifecycle as lifecycle_mod

    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    _seed_duplicates(lake, published)

    # Force the dict fallback and let it run under the bound: same outcome.
    monkeypatch.setattr(
        lifecycle_mod, "_iter_duplicated_keys_ordered", lambda *a, **k: None
    )
    report = compact_view_catalog(lake)
    assert {item.table: item.rows_deleted for item in report.tables}[VIEWS_TABLE] == 1

    # Past the loud bound the table is reported skipped, never half-processed.
    _seed_duplicates(lake, _publish(lake, fps=44))
    monkeypatch.setattr(lifecycle_mod, "_MAX_UNORDERED_SCAN_ROWS", 1)
    skipped = compact_view_catalog(lake)
    assert all(
        item.status == "skipped-unordered-over-bound"
        for item in skipped.tables
        if item.status != "absent"
    )
    assert not skipped.converged


def test_maintain_lake_reports_view_catalog_section(tmp_path):
    from lancedb_robotics.maintenance import maintain_lake

    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    _seed_duplicates(lake, published)

    report = maintain_lake(lake, tables=("lerobot_views",), cleanup_older_than=None)
    section = report.lerobot_view_catalog
    assert section is not None and section.get("report_version")
    assert section["rows_deleted"] >= 1
    assert section["latest_pointer_reconciliation"]["status"] == "reconciled"
    assert len(_table_rows(lake, VIEWS_TABLE, ["view_id"])) == 1


# ---------------------------------------------------------------------------
# Scale-review hardening (H1/M1/M3): stale-pointer healing, data-loss raise,
# mechanism/bound pins
# ---------------------------------------------------------------------------


def test_publish_failure_removes_stale_pointer_and_falls_back(tmp_path, monkeypatch):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    first = _publish(lake, fps=20)
    assert _latest_pointer_view_id(lake, first.repo_id) == first.view_id

    # Make BOTH pointer write paths fail for the next publish; catalog writes
    # (no update condition) stay untouched.
    real_merge = views_mod._merge_insert_with_retry

    def _refuse_pointer_writes(table, key_column, data, *, update_condition=None):
        if update_condition is not None:
            raise RuntimeError("pointer write refused")
        return real_merge(table, key_column, data, update_condition=update_condition)

    monkeypatch.setattr(views_mod, "_merge_insert_with_retry", _refuse_pointer_writes)
    monkeypatch.setattr(
        views_mod,
        "_cas_update_latest_pointer",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cas refused")),
    )

    with pytest.warns(RuntimeWarning, match="stale pointer was removed"):
        second = _publish(lake, fps=30)
    assert second.latest_pointer_updated is False
    # The previous pointer must NOT keep serving `first` as latest (H1): it was
    # removed, and the resolve chain falls back to a real catalog read.
    assert _latest_pointer_view_id(lake, first.repo_id) is None
    assert get_view(lake, repo_id=first.repo_id)["view_id"] == second.view_id


def test_cas_fallback_updates_pointer_when_conditional_merge_rejected(
    tmp_path, monkeypatch
):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    first = _publish(lake, fps=20)

    real_merge = views_mod._merge_insert_with_retry

    def _reject_conditional(table, key_column, data, *, update_condition=None):
        if update_condition is not None:
            raise RuntimeError("conditional merge unsupported")
        return real_merge(table, key_column, data, update_condition=update_condition)

    monkeypatch.setattr(views_mod, "_merge_insert_with_retry", _reject_conditional)
    with pytest.warns(RuntimeWarning, match="predicate-gated CAS"):
        second = _publish(lake, fps=30)
    assert second.latest_pointer_updated is True
    assert _latest_pointer_view_id(lake, first.repo_id) == second.view_id

    # Out-of-order older update through the CAS path never regresses the pointer.
    with pytest.warns(RuntimeWarning, match="predicate-gated CAS"):
        assert _update_latest_pointer(
            lake,
            repo_id=first.repo_id,
            view_id=first.view_id,
            view_created_at=first.created_at,
        )
    assert _latest_pointer_view_id(lake, first.repo_id) == second.view_id


def test_compact_reconciles_stale_and_dangling_pointers(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    first = _publish(lake, fps=20)
    second = _publish(lake, fps=30)

    # Simulate the crash window: pointer left at the older view (valid header,
    # so the resolve chain serves it silently -- the H1 hazard).
    stale = pa.Table.from_pylist(
        [
            {
                "repo_id": first.repo_id,
                "view_id": first.view_id,
                "view_created_at": first.created_at,
                "created_at": first.created_at,
            }
        ],
        schema=LEROBOT_VIEW_LATEST_SCHEMA,
    )
    table = lake.table(VIEW_LATEST_TABLE)
    table.merge_insert(
        "repo_id"
    ).when_matched_update_all().when_not_matched_insert_all().execute(stale)
    assert get_view(lake, repo_id=first.repo_id)["view_id"] == first.view_id

    report = compact_view_catalog(lake)
    assert report.latest_pointer_reconciliation.pointers_repaired == 1
    assert get_view(lake, repo_id=first.repo_id)["view_id"] == second.view_id

    # A pointer claiming something newer than every header is dangling: removed.
    from datetime import timedelta as _td

    bogus = pa.Table.from_pylist(
        [
            {
                "repo_id": first.repo_id,
                "view_id": "lrv-ffffffffffffffff",
                "view_created_at": second.created_at + _td(days=1),
                "created_at": second.created_at,
            }
        ],
        schema=LEROBOT_VIEW_LATEST_SCHEMA,
    )
    table.merge_insert(
        "repo_id"
    ).when_matched_update_all().when_not_matched_insert_all().execute(bogus)
    report = compact_view_catalog(lake)
    assert report.latest_pointer_reconciliation.dangling_pointers_removed == 1
    assert _latest_pointer_view_id(lake, first.repo_id) is None
    assert get_view(lake, repo_id=first.repo_id)["view_id"] == second.view_id


def test_compact_raises_when_key_would_lose_all_rows(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import view_lifecycle as lifecycle_mod
    from lancedb_robotics.lerobot_facade.view_lifecycle import (
        ViewCatalogCompactionError,
    )

    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    _seed_duplicates(lake, published)

    # Sabotage the delete bound so every copy of a duplicated key matches: the
    # postcondition must catch the zero-survivor key and raise, never report
    # success (the data-loss detector itself, pinned).
    monkeypatch.setattr(
        lifecycle_mod,
        "_older_than_predicate",
        lambda key_column, key, order_columns, bound: (
            f"{key_column} = {views_mod._sql_literal(key)}"
        ),
    )
    with pytest.raises(ViewCatalogCompactionError, match="zero surviving rows"):
        compact_view_catalog(lake)


def test_maintain_lake_raises_on_compaction_data_loss(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import view_lifecycle as lifecycle_mod
    from lancedb_robotics.maintenance import MaintenanceError, maintain_lake

    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    _seed_duplicates(lake, published)
    monkeypatch.setattr(
        lifecycle_mod,
        "_older_than_predicate",
        lambda key_column, key, order_columns, bound: (
            f"{key_column} = {views_mod._sql_literal(key)}"
        ),
    )
    # The one failure maintenance must NOT swallow into a best-effort report.
    with pytest.raises(MaintenanceError, match="postcondition"):
        maintain_lake(lake, tables=("lerobot_views",), cleanup_older_than=None)


def test_ordered_page_read_early_breaks_at_page_bound(tmp_path):
    lake = _two_episode_lake(tmp_path / "robot.lance")
    _publish_many(lake, repos=("acme/a", "acme/b", "acme/c"))  # 6 views

    with warnings.catch_warnings():
        # Primary path must be taken: any fallback warning fails the test.
        warnings.simplefilter("error", RuntimeWarning)
        rows = views_mod._list_view_page_ordered(
            lake, where=None, cursor=None, page_size=2
        )
    # Early break: exactly page_size+1 rows collected, never the whole catalog.
    assert rows is not None and len(rows) == 3


def test_compact_tier2_crash_after_readd_converges(tmp_path):
    from lancedb_robotics.lerobot_facade import view_lifecycle as lifecycle_mod

    lake = _two_episode_lake(tmp_path / "robot.lance")
    published = _publish(lake)
    header = _full_row(lake, VIEWS_TABLE, f"view_id = '{published.view_id}'")
    lake.table(VIEWS_TABLE).add(
        pa.Table.from_pylist([dict(header)], schema=LEROBOT_VIEWS_SCHEMA)
    )
    # Simulate a crash after the tier-2 re-add but before the delete: the
    # marker row exists alongside both tied copies.
    lifecycle_mod._readd_canonical_row(
        lake.table(VIEWS_TABLE),
        LEROBOT_VIEWS_SCHEMA,
        "view_id",
        published.view_id,
        ("created_at",),
        (header["created_at"],),
    )
    assert len(_table_rows(lake, VIEWS_TABLE, ["view_id"])) == 3

    report = compact_view_catalog(lake)
    assert report.converged
    assert len(_table_rows(lake, VIEWS_TABLE, ["view_id"])) == 1
    assert get_view(lake, view_id=published.view_id)["view_id"] == published.view_id


def test_compact_per_run_cap_and_oversized_skip(tmp_path, monkeypatch):
    from lancedb_robotics.lerobot_facade import view_lifecycle as lifecycle_mod

    lake = _two_episode_lake(tmp_path / "robot.lance")
    for repo in ("acme/a", "acme/b"):
        published = _publish(lake, repo_id=repo)
        header = _full_row(lake, VIEWS_TABLE, f"view_id = '{published.view_id}'")
        older = dict(header)
        older["created_at"] = header["created_at"] - timedelta(microseconds=1)
        lake.table(VIEWS_TABLE).add(
            pa.Table.from_pylist([older], schema=LEROBOT_VIEWS_SCHEMA)
        )

    capped = compact_view_catalog(lake, max_keys_per_run=1)
    by_table = {item.table: item for item in capped.tables}
    assert by_table[VIEWS_TABLE].duplicate_keys_processed == 1
    assert by_table[VIEWS_TABLE].duplicate_keys_remaining == 1
    assert not capped.converged
    assert compact_view_catalog(lake).converged  # re-run converges

    # Oversized keys are skipped and reported, never half-processed.
    published = _publish(lake, repo_id="acme/c")
    header = _full_row(lake, VIEWS_TABLE, f"view_id = '{published.view_id}'")
    older = dict(header)
    older["created_at"] = header["created_at"] - timedelta(microseconds=1)
    lake.table(VIEWS_TABLE).add(
        pa.Table.from_pylist([older], schema=LEROBOT_VIEWS_SCHEMA)
    )
    monkeypatch.setattr(lifecycle_mod, "_MAX_COPIES_PER_KEY", 1)
    skipped = compact_view_catalog(lake)
    by_table = {item.table: item for item in skipped.tables}
    assert by_table[VIEWS_TABLE].keys_skipped_oversized >= 1
    assert not skipped.converged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_view_list_pages_and_compact_catalog(tmp_path):
    import json as json_mod

    from typer.testing import CliRunner

    from lancedb_robotics.cli import app

    lake_path = tmp_path / "robot.lance"
    lake = _two_episode_lake(lake_path)
    published = _publish(lake)
    _publish(lake, fps=30)
    _seed_duplicates(lake, published)
    runner = CliRunner()

    first = runner.invoke(
        app,
        ["train", "view", "list", "--lake", str(lake_path), "--page-size", "1",
         "--format", "json"],
    )
    assert first.exit_code == 0, first.output
    payload = json_mod.loads(first.stdout)
    assert len(payload["rows"]) == 1 and payload["next_cursor"]

    second = runner.invoke(
        app,
        ["train", "view", "list", "--lake", str(lake_path), "--page-size", "1",
         "--cursor", payload["next_cursor"], "--format", "json"],
    )
    assert second.exit_code == 0, second.output
    second_payload = json_mod.loads(second.stdout)
    assert len(second_payload["rows"]) == 1
    assert second_payload["rows"][0]["view_id"] != payload["rows"][0]["view_id"]

    compact = runner.invoke(
        app,
        ["train", "view", "compact-catalog", "--lake", str(lake_path), "--dry-run"],
    )
    assert compact.exit_code == 0, compact.output
    assert "dry-run" in compact.stdout

    applied = runner.invoke(
        app, ["train", "view", "compact-catalog", "--lake", str(lake_path)]
    )
    assert applied.exit_code == 0, applied.output
    assert "converged: True" in applied.stdout
