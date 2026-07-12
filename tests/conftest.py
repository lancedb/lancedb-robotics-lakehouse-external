import importlib.util
import multiprocessing
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"

# Environment flag set by the torch-enabled DataLoader CI lane (backlog 0118).
# Mirrors LANCEDB_ROBOTICS_REQUIRE_RLDS_NATIVE for the RLDS native lane: in the
# default light dev/CI environment torch is absent and the loader tests skip, but
# the dedicated torch lane installs ``lancedb-robotics[torch]`` and sets this so a
# missing or broken torch install *fails* instead of silently skipping.
_REQUIRE_TORCH_ENV = "LANCEDB_ROBOTICS_REQUIRE_TORCH"
_REQUIRE_VIDEO_DECODE_ENV = "LANCEDB_ROBOTICS_REQUIRE_VIDEO_DECODE"
_REQUIRE_LEROBOT_NATIVE_BENCHMARK_ENV = (
    "LANCEDB_ROBOTICS_REQUIRE_LEROBOT_NATIVE_BENCHMARK"
)
_REQUIRE_INVOCATION_CONFORMANCE_ENV = "LANCEDB_ROBOTICS_REQUIRE_INVOCATION_CONFORMANCE"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def require_torch_loader() -> None:
    """Gate a torch DataLoader test on a working PyTorch install.

    Skips with an install hint when torch is unavailable in the default light
    environment. When the torch CI lane requested torch
    (``LANCEDB_ROBOTICS_REQUIRE_TORCH=1``) an absent/broken install *fails* the
    test instead, so an unexpected skip can never masquerade as a pass in the lane
    whose entire purpose is to exercise real PyTorch workers.
    """
    # Imported lazily and by attribute lookup so tests can monkeypatch
    # ``training.torch_available`` to simulate an absent install.
    from lancedb_robotics.training import torch_available

    if torch_available():
        return
    reason = "torch is not installed; install lancedb-robotics[torch]"
    if os.environ.get(_REQUIRE_TORCH_ENV) == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def require_video_decode() -> tuple[ModuleType, ModuleType]:
    """Gate a real decoded-source-video test on the video-decode extra.

    Default dev/CI environments can skip the expensive native media stack. The
    dedicated video-decode lane sets ``LANCEDB_ROBOTICS_REQUIRE_VIDEO_DECODE=1``
    so a missing or broken PyAV/NumPy install fails instead of producing an
    accidental all-skipped lane.
    """

    missing = [
        module
        for module in ("av", "numpy")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        reason = (
            "video-decode dependencies are not installed; install "
            "lancedb-robotics[video-decode] "
            f"(missing: {', '.join(missing)})"
        )
        if os.environ.get(_REQUIRE_VIDEO_DECODE_ENV) == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    try:
        import av
        import numpy as np
    except Exception as exc:  # noqa: BLE001 - native decoder imports fail in host-specific ways.
        reason = (
            "video-decode dependencies are installed but failed to import; install "
            f"lancedb-robotics[video-decode] ({type(exc).__name__}: {exc})"
        )
        if os.environ.get(_REQUIRE_VIDEO_DECODE_ENV) == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    return av, np


def require_lerobot_native_benchmark():
    """Gate the real official LeRobot benchmark lane on native dependencies.

    Default dev/CI runs keep the dependency-light skip behavior. The dedicated
    LeRobot-native benchmark lane sets
    ``LANCEDB_ROBOTICS_REQUIRE_LEROBOT_NATIVE_BENCHMARK=1`` so a missing
    LeRobot/Torch/decode stack or a moved official loader import fails the lane
    with the same dependency diagnostics that the benchmark report records.
    """

    from lancedb_robotics import benchmark

    status = benchmark._lerobot_native_dependency_status()
    if not status["available"]:
        missing = ", ".join(str(item) for item in status.get("missing") or [])
        reason = (
            "LeRobot-native benchmark dependencies are not installed; install "
            "lancedb-robotics[lerobot-native-bench]"
        )
        if missing:
            reason = f"{reason} (missing: {missing})"
        if os.environ.get(_REQUIRE_LEROBOT_NATIVE_BENCHMARK_ENV) == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    components, import_error = benchmark._load_lerobot_native_components()
    if components is None:
        reason = (
            "official LeRobot native benchmark loader could not be imported "
            f"({import_error})"
        )
        if os.environ.get(_REQUIRE_LEROBOT_NATIVE_BENCHMARK_ENV) == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    return components, status


def require_start_method(method: str) -> None:
    """Skip with an explicit, per-platform reason when a multiprocessing start
    method is unavailable, instead of relying on an accidental/implicit skip.

    Documented DataLoader worker start-method support (backlog 0118):

    * ``spawn``      -- all platforms (Linux, macOS, Windows). Re-imports the
      module and unpickles the dataset, so it exercises the picklability of the
      loader config and the ``__reduce__`` reopen contract.
    * ``fork``       -- Linux and macOS only (unavailable on Windows). Inherits the
      parent address space, so an in-process monkeypatched Enterprise lake survives
      into workers.
    * ``forkserver`` -- Linux and macOS.

    Skips here are the *documented* platform gaps the torch lane tolerates; the
    ``LANCEDB_ROBOTICS_REQUIRE_TORCH`` gate deliberately does not turn them into
    failures because they are expected on the platform, not accidental.
    """
    available = multiprocessing.get_all_start_methods()
    if method not in available:
        pytest.skip(
            f"multiprocessing start method {method!r} is unavailable on "
            f"{sys.platform} (available: {', '.join(available)})"
        )


def require_invocation_conformance_live() -> None:
    """Gate a live 0076-audit invocation conformance test on live credentials.

    The contract-test mode needs no credentials and always runs; a *live* probe
    needs the ``LANCEDB_ROBOTICS_CONFORMANCE_*`` db:///namespace endpoint and
    auth-ref vars. When they are absent the live subset skips cleanly, so the
    default lane stays green (backlog 0130 acceptance: "run in contract-test mode
    when live credentials are absent"). A dedicated live lane sets
    ``LANCEDB_ROBOTICS_REQUIRE_INVOCATION_CONFORMANCE=1`` so a *misconfigured*
    live run fails loudly instead of silently skipping the endpoint it exists to
    exercise.
    """
    from lancedb_robotics import invocation_conformance as ic

    available, missing = ic.live_credentials_available()
    if available:
        return
    reason = "live invocation-conformance credentials absent; set " + ", ".join(missing)
    if os.environ.get(_REQUIRE_INVOCATION_CONFORMANCE_ENV) == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def assert_matches_snapshot(name: str, actual: str) -> None:
    """Compare actual against tests/snapshots/<name>; set UPDATE_SNAPSHOTS=1 to regenerate."""
    path = SNAPSHOTS_DIR / name
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(actual)
    assert path.exists(), f"missing snapshot {path}; run with UPDATE_SNAPSHOTS=1 to create it"
    assert actual == path.read_text()
