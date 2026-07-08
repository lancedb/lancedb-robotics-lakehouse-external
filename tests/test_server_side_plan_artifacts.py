"""Backlog 0117: Enterprise training server-side row-plan artifacts.

These tests drive the artifact contract (a version-pinned handle the query node
builds once, workers page through), plus the wiring into the Enterprise training
dataset, the ``lake.training`` API, and the ``train plan`` CLI.
"""

import json

import pytest

# Reuse the enterprise/lake fixtures and helpers from the native-training suite.
from test_native_training_dataset import (  # noqa: E402
    _mark_enterprise_lake,
    _training_lake,
)
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.training import (
    EnterpriseCapabilityError,
    ServerSidePlanError,
    ServerSidePlanUnavailableError,
    build_server_side_row_plan,
    open_server_side_row_plan,
)
from lancedb_robotics.training_plan_artifacts import (
    InMemoryServerSidePlanStore,
    LanceTablePlanPageStore,
)

runner = CliRunner()


def _synthetic_plan(total_rows, *, page_size=1024, store=None, shuffle=True):
    """A server-side artifact over a synthetic ordered id set (no lake needed)."""
    ordered_row_ids = list(range(total_rows))
    ordered_frame_ids = [f"obs-{index}" for index in range(total_rows)]
    return build_server_side_row_plan(
        row_plan_id="rowplan-synthetic",
        snapshot_id="ds-synthetic",
        snapshot_name="synthetic-v1",
        table_versions=[{"table": "observations", "version": 3, "tag": ""}],
        columns=["observation_id"],
        display_uri="db://robotics",
        connection_kind="lancedb_remote_db",
        ordered_row_ids=ordered_row_ids,
        ordered_frame_ids=ordered_frame_ids,
        ordering_policy="seeded-global-permutation" if shuffle else "snapshot-frame-order",
        shuffle=shuffle,
        shuffle_seed=23,
        epoch=2,
        pushed_filters={"modality": "modality"},
        logical_predicates=["modality = 'image'"],
        page_size=page_size,
        store=store,
        capabilities={"server_side_row_plan": True},
    )


# --------------------------------------------------------------------------- #
# Test-first plan: the four failing cases from the backlog record.            #
# --------------------------------------------------------------------------- #


def test_large_plan_four_workers_bounded_pages_no_duplicate_ids():
    total_rows = 100_000
    page_size = 1024
    artifact = _synthetic_plan(total_rows, page_size=page_size)
    assert artifact.total_rows == total_rows
    assert artifact.num_pages == (total_rows + page_size - 1) // page_size

    seen: list[int] = []
    owned_pages: dict[int, set[int]] = {}
    for worker_id in range(4):
        pages = list(artifact.iter_pages(worker_id=worker_id, num_workers=4))
        owned_pages[worker_id] = {page.page_index for page in pages}
        for page in pages:
            # bounded: no page exceeds the page size
            assert page.size <= page_size
            assert len(page.row_ids) == page.size
            seen.extend(page.row_ids)

    # Workers claim non-overlapping pages.
    for a in range(4):
        for b in range(a + 1, 4):
            assert owned_pages[a].isdisjoint(owned_pages[b])
    # Every page is claimed exactly once, covering the whole epoch with no dups.
    assert sum(len(pages) for pages in owned_pages.values()) == artifact.num_pages
    assert len(seen) == total_rows
    assert len(set(seen)) == total_rows


def test_resume_from_global_offset_returns_suffix_across_workers():
    total_rows = 100_000
    page_size = 1024
    resume_from = 5_003
    artifact = _synthetic_plan(total_rows, page_size=page_size)

    full_order = artifact.global_frame_ids()
    assert full_order == [f"obs-{index}" for index in artifact.worker_row_ids()]

    # Gather each worker's resumed pages, then reassemble in ascending page order.
    pages_by_index: dict[int, list[int]] = {}
    for worker_id in range(4):
        for page in artifact.iter_pages(
            worker_id=worker_id, num_workers=4, resume_from=resume_from
        ):
            pages_by_index[page.page_index] = list(page.row_ids)
    reassembled: list[int] = []
    for page_index in sorted(pages_by_index):
        reassembled.extend(pages_by_index[page_index])

    expected_suffix = artifact.worker_row_ids()[resume_from:]
    assert reassembled == expected_suffix
    assert len(reassembled) == total_rows - resume_from
    assert len(set(reassembled)) == len(reassembled)


def test_handle_serialize_reload_and_page_without_local_table():
    store = InMemoryServerSidePlanStore()
    artifact = _synthetic_plan(10_000, page_size=512, store=store)

    # The handle is small JSON metadata; the pages live in the store.
    handle = artifact.to_dict()
    handle_json = json.dumps(handle, sort_keys=True)
    reloaded_handle = json.loads(handle_json)
    store_payload = json.loads(json.dumps(store.dump()))

    # A worker in a fresh process: rebuild reader from serialized handle + store,
    # with no snapshot/Lance table object at all.
    reloaded_store = InMemoryServerSidePlanStore.load(store_payload)
    reopened = open_server_side_row_plan(reloaded_handle, store=reloaded_store)

    assert reopened.plan_handle_id == artifact.plan_handle_id
    assert reopened.worker_row_ids(worker_id=1, num_workers=3) == artifact.worker_row_ids(
        worker_id=1, num_workers=3
    )
    assert reopened.global_frame_ids() == artifact.global_frame_ids()


def test_disabled_capability_raises_typed_diagnostic(tmp_path):
    lake = _mark_enterprise_lake(_training_lake(tmp_path / "robot.lance"))
    lake.enterprise_training_capabilities = {"server_side_row_plan": False}

    with pytest.raises(ServerSidePlanUnavailableError) as excinfo:
        lake.training.server_side_row_plan("demo-v1", columns=["observation_id"])
    message = str(excinfo.value)
    assert "server_side_row_plan" in message
    assert "fallback='local'" in message
    assert excinfo.value.missing_capabilities == ("server_side_row_plan",)


# --------------------------------------------------------------------------- #
# Equivalence, secret hygiene, tokens.                                         #
# --------------------------------------------------------------------------- #


def test_local_and_server_side_handles_produce_equivalent_sample_ids(tmp_path):
    lake = _mark_enterprise_lake(
        _training_lake(tmp_path / "robot-large.lance", frame_count=53)
    )
    enterprise = lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
        backend="enterprise",
    )
    handle = enterprise.row_plan_handle
    assert handle is not None

    # Server-side global order equals the dataset's actual sample order.
    ordered_ids = [sample["observation_id"] for sample in enterprise]
    assert handle.global_frame_ids() == ordered_ids

    # And matches a plain local dataset built with the same knobs.
    local = _training_lake(tmp_path / "robot-local.lance", frame_count=53).training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
    )
    assert handle.global_frame_ids() == [sample["observation_id"] for sample in local]


def test_handle_is_secret_free_and_records_pinned_identity(tmp_path):
    lake = _mark_enterprise_lake(_training_lake(tmp_path / "robot.lance"))
    handle = lake.training.server_side_row_plan(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=7,
    )
    flat = json.dumps(handle).lower()
    assert "secret-api-key" not in flat
    assert "api_key" not in flat
    assert "authorization" not in flat

    assert handle["display_uri"] == "db://robotics"
    assert handle["snapshot_id"]
    assert handle["row_plan_id"].startswith("rowplan-")
    assert handle["plan_handle_id"].startswith("srvplan-")
    assert {tv["table"] for tv in handle["table_versions"]}
    assert handle["columns"] == ["observation_id"]
    assert handle["ordering_policy"] == "seeded-global-permutation"


def test_page_token_roundtrips_and_is_bound_to_the_handle():
    artifact = _synthetic_plan(4096, page_size=1000)
    other = _synthetic_plan(4096, page_size=500)

    token = artifact.page_token(2)
    assert artifact.parse_page_token(token) == 2
    page = artifact.page_from_token(token)
    assert page.page_index == 2
    assert page.start_offset == 2000

    with pytest.raises(ServerSidePlanError):
        other.parse_page_token(token)
    with pytest.raises(ServerSidePlanError):
        artifact.parse_page_token("not-a-real-token")


# --------------------------------------------------------------------------- #
# Dataset wiring + lake.training API + epoch backend capability.              #
# --------------------------------------------------------------------------- #


def test_enterprise_dataset_advertises_plan_summary_local_does_not(tmp_path):
    ent = _mark_enterprise_lake(_training_lake(tmp_path / "ent.lance", frame_count=9))
    dataset = ent.training.dataset(
        "demo-v1", columns=["observation_id"], shuffle=True, backend="enterprise"
    )
    summary = dataset.manifest.to_dict()["server_side_plan"]
    assert summary["available"] is True
    assert summary["total_rows"] == 9
    assert summary["num_pages"] == 1
    assert summary["kind"].startswith("lancedb-robotics/server-side-row-plan/")
    assert dataset.row_plan_handle is not None

    plain = _training_lake(tmp_path / "loc.lance", frame_count=9).training.dataset(
        "demo-v1", columns=["observation_id"]
    )
    assert plain.row_plan_handle is None
    assert "server_side_plan" not in plain.manifest.to_dict()


def test_server_side_row_plan_api_pages_via_durable_store(tmp_path):
    lake = _mark_enterprise_lake(
        _training_lake(tmp_path / "robot-large.lance", frame_count=41)
    )
    handle = lake.training.server_side_row_plan(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=1,
        page_size=8,
    )
    assert handle["store_kind"] == LanceTablePlanPageStore.kind
    assert handle["num_pages"] == (41 + 7) // 8

    # Page across 4 workers through the durable store; cover the epoch once.
    all_pages: dict[int, list[str]] = {}
    for worker_id in range(4):
        for page in lake.training.row_plan_pages(
            handle, worker_id=worker_id, num_workers=4
        ):
            all_pages[page["page_index"]] = list(page["frame_ids"])
    merged: list[str] = []
    for page_index in sorted(all_pages):
        merged.extend(all_pages[page_index])
    assert len(merged) == 41
    assert len(set(merged)) == 41

    # A single page fetch carries a next-page token for handoff.
    first = lake.training.row_plan_page(handle, worker_id=0, num_workers=4)
    assert first["size"] > 0
    assert first["next_page_token"] is not None
    follow = lake.training.row_plan_page(handle, page_token=first["next_page_token"])
    assert follow["page_index"] == 4


def test_epoch_backend_capability_reports_server_side_plan(tmp_path):
    ent = _mark_enterprise_lake(_training_lake(tmp_path / "ent.lance"))
    capability = ent.training.epoch_backend_capability()
    assert capability["server_side_plan"]["supported"] is True
    assert capability["server_side_plan"]["execution_mode"] == "server-side-plan-artifact"

    plain = _training_lake(tmp_path / "loc.lance")
    assert plain.training.epoch_backend_capability()["server_side_plan"]["supported"] is False


def test_local_backend_server_side_row_plan_is_rejected(tmp_path):
    plain = _training_lake(tmp_path / "loc.lance")
    with pytest.raises(EnterpriseCapabilityError):
        plain.training.server_side_row_plan("demo-v1", columns=["observation_id"])


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #


def test_cli_plan_page_reads_durable_handle(tmp_path):
    path = tmp_path / "robot.lance"
    lake = _mark_enterprise_lake(_training_lake(path, frame_count=20))
    handle = lake.training.server_side_row_plan(
        "demo-v1", columns=["observation_id"], shuffle=True, shuffle_seed=5, page_size=4
    )
    handle_path = tmp_path / "handle.json"
    handle_path.write_text(json.dumps(handle))

    result = runner.invoke(
        app,
        [
            "train",
            "plan",
            "page",
            "--lake",
            str(path),
            "--handle",
            str(handle_path),
            "--worker",
            "0",
            "--num-workers",
            "2",
            "--all",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    pages = json.loads(result.output)
    assert pages
    owned = {page["page_index"] for page in pages}
    assert all(index % 2 == 0 for index in owned)


def test_cli_plan_build_local_lake_reports_typed_error(tmp_path):
    path = tmp_path / "loc.lance"
    _training_lake(path)
    result = runner.invoke(
        app,
        [
            "train",
            "plan",
            "build",
            "--lake",
            str(path),
            "--snapshot",
            "demo-v1",
            "--backend",
            "enterprise",
        ],
    )
    assert result.exit_code == 1
    assert "error:" in result.output
