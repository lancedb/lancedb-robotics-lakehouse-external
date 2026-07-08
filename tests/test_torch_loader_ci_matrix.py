"""Torch-enabled Enterprise/native DataLoader CI matrix (backlog 0118).

This module hosts the dedicated torch CI lane. The lane installs the optional
``torch`` extra and runs with ``LANCEDB_ROBOTICS_REQUIRE_TORCH=1`` so a missing or
broken PyTorch install *fails* rather than silently skipping. It drives the public
DataLoader path with real PyTorch workers across the supported multiprocessing
start methods and proves the Enterprise remote-safe contract: worker partition
coverage, global ``resume_from``, backend display URI in batch lineage, and --
critically -- that no API key is ever serialized into a worker's picklable loader
config.

Platform / start-method support matrix (enforced via ``require_start_method``):

    start method | Linux | macOS | Windows | exercised for
    -------------+-------+-------+---------+---------------------------------
    spawn        |  yes  |  yes  |   yes   | pickling + ``__reduce__`` reopen
    fork         |  yes  |  yes  |   no    | in-process monkeypatched db://
    forkserver   |  yes  |  yes  |   no    | (not exercised here)

``fork`` inherits the parent address space, so a mocked in-process ``db://`` lake
survives into workers; ``spawn`` starts a fresh interpreter that must unpickle the
dataset and reopen the lake from its loader config alone. Windows has no ``fork``;
the mocked-Enterprise fork tests (in ``test_native_training_dataset``) document
that gap with an explicit skip instead of relying on an accidental one.
"""

import json
import pickle

import pytest
from conftest import require_start_method, require_torch_loader
from test_aligned_training_dataset import _aligned_training_lake
from test_native_training_dataset import _mark_enterprise_lake, _training_lake

import lancedb_robotics.training as training_mod
from lancedb_robotics.training import (
    to_torch_aligned_dataloader,
    to_torch_dataloader,
    to_torch_iterable_dataset,
)

# The mocked Enterprise API key that _mark_enterprise_lake plants in the
# connection spec. It must never cross the worker serialization boundary.
_MOCK_API_KEY = "secret-api-key"


@pytest.fixture
def local_lake(tmp_path):
    # A slightly larger epoch than the 3-frame default so resume_from has a
    # meaningful, unambiguous tail to verify.
    return _training_lake(tmp_path / "robot.lance", frame_count=8)


@pytest.mark.torch_loader
def test_torch_loader_lane_has_torch_installed():
    """CI smoke: torch must really import in the lane; elsewhere this skips cleanly.

    ``torch_available()`` only probes ``find_spec``; the explicit ``import torch``
    additionally proves the install is not merely discoverable but loadable, which
    is the failure mode a broken wheel would exhibit.
    """
    require_torch_loader()
    from lancedb_robotics.training import torch_available

    assert torch_available()
    import torch  # noqa: F401  -- prove the native extension actually loads


def test_require_torch_loader_converts_unexpected_skip_to_failure(monkeypatch):
    """The lane's core guarantee, always covered (this fakes torch away).

    Acceptance criterion "the job fails if torch loader tests are skipped
    unexpectedly": with the flag set, a torch-absent test must raise ``Failed``,
    never ``Skipped``.
    """
    monkeypatch.setattr(training_mod, "torch_available", lambda: False)

    monkeypatch.delenv("LANCEDB_ROBOTICS_REQUIRE_TORCH", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_torch_loader()

    monkeypatch.setenv("LANCEDB_ROBOTICS_REQUIRE_TORCH", "1")
    with pytest.raises(pytest.fail.Exception):
        require_torch_loader()


@pytest.mark.torch_loader
def test_native_spawn_workers_cover_epoch_once(local_lake):
    """A fresh spawn interpreter rebuilds the dataset from its config and the two
    workers cover the global epoch exactly once (no fork-inherited state)."""
    require_torch_loader()
    require_start_method("spawn")

    dataset = local_lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
    )
    loader = to_torch_dataloader(
        dataset,
        batch_size=1,
        num_workers=2,
        adapter="iterable",
        multiprocessing_context="spawn",
    )

    seen = []
    worker_ids = set()
    for batch in loader:
        seen.extend(batch["observation_id"])
        worker_ids.add(batch["_lineage"]["worker"]["id"])

    expected = [row["observation_id"] for row in dataset]
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == len(expected)
    assert worker_ids == {0, 1}


@pytest.mark.torch_loader
def test_native_spawn_worker_resume_from_yields_global_tail(local_lake):
    """Global ``resume_from`` semantics survive real spawn workers: the union of
    what the workers emit is exactly the global epoch order minus the first N."""
    require_torch_loader()
    require_start_method("spawn")

    base = local_lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
    )
    global_order = [row["observation_id"] for row in base]
    resume_from = 3

    resumed = local_lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
        resume_from=resume_from,
    )
    loader = to_torch_dataloader(
        resumed,
        batch_size=1,
        num_workers=2,
        adapter="iterable",
        multiprocessing_context="spawn",
    )

    seen = [obs_id for batch in loader for obs_id in batch["observation_id"]]

    tail = global_order[resume_from:]
    assert sorted(seen) == sorted(tail)
    assert len(seen) == len(set(seen)) == len(global_order) - resume_from


@pytest.mark.torch_loader
def test_aligned_spawn_workers_cover_ticks_once(tmp_path):
    """The aligned policy-tick iterable also covers its global tick epoch exactly
    once under real spawn workers (parity with the native path)."""
    require_torch_loader()
    require_start_method("spawn")

    lake, _ = _aligned_training_lake(tmp_path / "robot.lance")
    dataset = lake.training.aligned_dataset(
        name="policy_bridge",
        columns=["tick_index", "streams", "masks", "lineage"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
    )
    loader = to_torch_aligned_dataloader(
        dataset,
        batch_size=1,
        num_workers=2,
        adapter="iterable",
        multiprocessing_context="spawn",
    )

    seen = []
    worker_ids = set()
    for batch in loader:
        seen.extend(batch["tick_index"].tolist())
        worker_ids.add(batch["_lineage"]["worker"]["id"])

    expected = [sample["tick_index"] for sample in dataset]
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)) == len(expected)
    assert worker_ids == {0, 1}


@pytest.mark.torch_loader
def test_enterprise_iterable_dataset_pickles_without_api_key_and_reopens(
    local_lake, monkeypatch
):
    """A spawn worker reopens an Enterprise dataset from its picklable loader
    config without ever serializing the API key.

    Spawn transfers the dataset to workers via ``pickle``, so this asserts on the
    serialized bytes directly: the secret key is absent, while the reopen
    coordinates a legitimate worker needs (host override, auth *reference*, display
    URI) survive. It then confirms the revived config reopens with those
    coordinates and that the batch lineage a model reads is likewise secret-free.
    """
    require_torch_loader()

    _mark_enterprise_lake(local_lake)
    dataset = local_lake.training.dataset(
        "demo-v1",
        columns=["observation_id"],
        shuffle=True,
        shuffle_seed=23,
        epoch=2,
        backend="enterprise",
    )
    torch_dataset = to_torch_iterable_dataset(dataset)

    blob = pickle.dumps(torch_dataset)
    assert _MOCK_API_KEY.encode() not in blob
    # The reopen coordinates a spawn worker legitimately needs DO survive.
    assert b"https://phalanx.acme.internal" in blob  # host_override
    assert b"enterprise-prod" in blob  # remote auth *reference*, not the secret
    assert b"db://robotics" in blob  # display uri

    revived = pickle.loads(blob)
    connection = revived._config["connection"]
    assert connection["remote"]["host_override"] == "https://phalanx.acme.internal"
    assert connection["remote"]["remote_auth_ref"] == "enterprise-prod"
    assert _MOCK_API_KEY not in repr(revived._config)

    # In-process, workers reopen via the mocked endpoint; the batch lineage the
    # model consumes carries the display URI and no secret.
    monkeypatch.setattr(
        training_mod,
        "_open_lake_from_loader_config",
        lambda config: local_lake,
    )
    sample = next(iter(torch_dataset))
    lineage_json = json.dumps(sample["_lineage"])
    assert _MOCK_API_KEY not in lineage_json
    assert sample["_lineage"]["backend"]["display_uri"] == "db://robotics"
