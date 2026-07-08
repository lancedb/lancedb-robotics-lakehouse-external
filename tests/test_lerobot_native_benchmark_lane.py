"""Official LeRobotDataset benchmark fixture matrix (backlog 0372)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest
from conftest import require_lerobot_native_benchmark
from test_benchmark import benchmark_lake

from lancedb_robotics.benchmark import (
    BENCHMARK_METRICS,
    LEROBOT_NATIVE_FORMAT,
    run_benchmark_suite,
)

_REQUIRE_LEROBOT_NATIVE_BENCHMARK_ENV = (
    "LANCEDB_ROBOTICS_REQUIRE_LEROBOT_NATIVE_BENCHMARK"
)
_ARTIFACT_DIR_ENV = "LANCEDB_ROBOTICS_LEROBOT_NATIVE_ARTIFACT_DIR"
_PUBLIC_REPO_ENV = "LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_REPO_ID"
_PUBLIC_REVISION_ENV = "LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_REVISION"
_PUBLIC_SOURCE_URI_ENV = "LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_SOURCE_URI"
_LOCAL_REPO_ID = "lancedb-robotics/local-native-fixture"


def _artifact_dir(tmp_path: Path) -> Path:
    configured = os.environ.get(_ARTIFACT_DIR_ENV)
    return Path(configured) if configured else tmp_path / "lerobot-native-artifacts"


def _write_artifact(root: Path, name: str, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n")


def _dependency_status() -> dict[str, Any]:
    from lancedb_robotics import benchmark

    try:
        return benchmark._lerobot_native_dependency_status()
    except Exception as exc:  # noqa: BLE001 - optional native stacks fail in host-specific ways.
        return {"available": False, "status_error": f"{type(exc).__name__}: {exc}"}


def _official_local_lerobot_fixture(root: Path, dataset_cls: Any) -> Path:
    import numpy as np

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["arm_x", "arm_y"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["delta_x", "delta_y"],
        },
    }
    dataset = dataset_cls.create(
        _LOCAL_REPO_ID,
        fps=10,
        features=features,
        root=root,
        robot_type="aloha",
        use_videos=False,
    )
    try:
        for episode_index, task in enumerate(("pick cube", "place cube")):
            frame_count = 2 if episode_index == 0 else 1
            for frame_index in range(frame_count):
                value = float((episode_index * 2) + frame_index)
                dataset.add_frame(
                    {
                        "observation.state": np.array(
                            [value, value + 0.5],
                            dtype=np.float32,
                        ),
                        "action": np.array(
                            [value / 10.0, (value / 10.0) + 0.05],
                            dtype=np.float32,
                        ),
                        "task": task,
                    }
                )
            dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    return root


def test_require_lerobot_native_benchmark_converts_unexpected_skip_to_failure(monkeypatch):
    """The native benchmark CI lane must fail, not pass with all tests skipped."""

    import lancedb_robotics.benchmark as benchmark

    monkeypatch.setattr(
        benchmark,
        "_lerobot_native_dependency_status",
        lambda: {
            "available": False,
            "modules": ["lerobot", "torch"],
            "missing": ["lerobot", "torch"],
            "install": "install native stack",
            "versions": {},
            "decode_backend": {
                "selected": None,
                "available": False,
                "missing": ["torchcodec", "av"],
            },
        },
    )
    monkeypatch.delenv(_REQUIRE_LEROBOT_NATIVE_BENCHMARK_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_lerobot_native_benchmark()

    monkeypatch.setenv(_REQUIRE_LEROBOT_NATIVE_BENCHMARK_ENV, "1")
    with pytest.raises(pytest.fail.Exception):
        require_lerobot_native_benchmark()


def test_lerobot_native_loader_import_supports_current_package_path(monkeypatch):
    """LeRobot 0.4.x exposes the class below lerobot.datasets.lerobot_dataset."""

    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    import lancedb_robotics.benchmark as benchmark

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name in {"lerobot", "torch", "av"}:
            return object()
        if name == "torchcodec":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    status = benchmark._lerobot_native_dependency_status()
    components, import_error = benchmark._load_lerobot_native_components()

    assert status["available"] is True
    assert status["decode_backend"]["selected"] == "pyav"
    assert components is not None, import_error
    assert components.dataset_api in {
        "lerobot.datasets.LeRobotDataset",
        "lerobot.datasets.lerobot_dataset.LeRobotDataset",
    }


@pytest.mark.lerobot_native_benchmark
def test_lerobot_native_benchmark_runs_official_local_v3_fixture(tmp_path):
    artifact_root = _artifact_dir(tmp_path)
    try:
        components, dependency_status = require_lerobot_native_benchmark()
    except Exception as exc:
        _write_artifact(
            artifact_root,
            "lerobot-native-dependencies.json",
            {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "dependency_status": _dependency_status(),
            },
        )
        raise

    fixture_root = _official_local_lerobot_fixture(
        tmp_path / "official-local-v3",
        components.dataset_cls,
    )
    lake = benchmark_lake(tmp_path / "robot.lance")
    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LEROBOT_NATIVE_FORMAT],
        output_dir=artifact_root / "benchmark-artifacts",
        sample_limit=3,
        random_access_samples=3,
        random_frame_samples=2,
        frames_per_clip=2,
        seed=37,
        source_dataset_id=_LOCAL_REPO_ID,
        source_dataset_uri=str(fixture_root),
        source_dataset_size_tier="tiny-local-v3",
        source_dataset_size_bytes=0,
        source_dataset_size_note="generated by official LeRobotDataset.create",
    )
    _write_artifact(artifact_root, "lerobot-native-local-fixture-report.json", report)

    native = report["formats"][LEROBOT_NATIVE_FORMAT]
    assert native["status"] == "completed", native.get("skip_reason")
    assert set(native["metrics"]) == set(BENCHMARK_METRICS)
    for metric in BENCHMARK_METRICS:
        assert native["metrics"][metric]["status"] == "completed", metric
    assert native["metrics"]["dataloader_throughput"]["details"]["samples"] == 3
    assert native["metrics"]["random_access_latency"]["details"]["samples"] == 3
    assert native["metrics"]["random_frame_sampling"]["details"]["frames"] >= 2

    loader = native["native_loader"]
    assert loader["official_api"] == "lerobot.datasets.LeRobotDataset"
    assert loader["resolved_api"] == components.dataset_api
    assert loader["dependency_status"]["available"] is True
    assert loader["dependency_status"]["decode_backend"]["available"] is True
    assert loader["source"]["kind"] == "source-corpus"
    assert loader["source"]["repo_id"] == _LOCAL_REPO_ID
    assert loader["source"]["download_videos"] is False
    assert loader["dataset"]["class"] == "LeRobotDataset"
    assert loader["dataset"]["num_frames"] == 3
    assert loader["dataset"]["num_episodes"] == 2
    assert loader["dataset"]["camera_keys"] == []
    assert set(loader["dataset"]["feature_keys"]) >= {
        "action",
        "observation.state",
        "task_index",
    }
    assert dependency_status["versions"]["lerobot"]


@pytest.mark.lerobot_native_benchmark
def test_lerobot_native_benchmark_runs_pinned_public_fixture_when_configured(tmp_path):
    repo_id = os.environ.get(_PUBLIC_REPO_ENV)
    revision = os.environ.get(_PUBLIC_REVISION_ENV)
    if not repo_id or not revision:
        pytest.skip(
            "set LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_REPO_ID and "
            "LANCEDB_ROBOTICS_LEROBOT_NATIVE_PUBLIC_REVISION for the pinned "
            "public LeRobot smoke fixture"
        )

    artifact_root = _artifact_dir(tmp_path)
    require_lerobot_native_benchmark()
    lake = benchmark_lake(tmp_path / "robot.lance")
    source_uri = os.environ.get(_PUBLIC_SOURCE_URI_ENV) or f"hf://{repo_id}"
    report = run_benchmark_suite(
        lake,
        "bench-v1",
        formats=[LEROBOT_NATIVE_FORMAT],
        output_dir=artifact_root / "public-benchmark-artifacts",
        sample_limit=1,
        random_access_samples=1,
        random_frame_samples=1,
        frames_per_clip=1,
        seed=41,
        source_dataset_id=repo_id,
        source_dataset_revision=revision,
        source_dataset_uri=source_uri,
        source_dataset_size_tier="public-smoke",
        source_dataset_size_note="optional pinned public LeRobot native smoke fixture",
    )
    _write_artifact(artifact_root, "lerobot-native-public-fixture-report.json", report)

    native = report["formats"][LEROBOT_NATIVE_FORMAT]
    assert native["status"] == "completed", native.get("skip_reason")
    assert native["native_loader"]["source"]["kind"] == "source-corpus"
    assert native["native_loader"]["source"]["revision"] == revision
    for metric in BENCHMARK_METRICS:
        assert native["metrics"][metric]["status"] == "completed", metric
