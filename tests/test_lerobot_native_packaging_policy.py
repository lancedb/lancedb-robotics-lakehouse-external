import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE_BENCH_EXTRA = "lerobot-native-bench"


def _load_toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text())


def _requirement_name(requirement: str) -> str:
    return (
        requirement.split(";", maxsplit=1)[0]
        .split(">=", maxsplit=1)[0]
        .split("<", maxsplit=1)[0]
        .split("[", maxsplit=1)[0]
        .strip()
    )


def test_lerobot_native_bench_extra_is_single_supported_stack():
    pyproject = _load_toml("pyproject.toml")
    optional = pyproject["project"]["optional-dependencies"]

    requirements = {
        _requirement_name(requirement)
        for requirement in optional[NATIVE_BENCH_EXTRA]
    }
    assert requirements == {"lerobot", "torch", "av", "numpy"}
    assert "torchcodec" not in requirements

    conflicts = pyproject["tool"]["uv"]["conflicts"]
    assert [
        {"extra": NATIVE_BENCH_EXTRA},
        {"extra": "rlds"},
    ] in conflicts


def test_lerobot_native_benchmark_ci_uses_supported_extra():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "uv sync --extra dev --extra lerobot-native-bench" in workflow
    assert "uv sync --extra dev --extra lerobot --extra torch --extra video-decode" not in workflow


def test_uv_lock_records_lerobot_native_bench_extra():
    lock = _load_toml("uv.lock")
    package = next(
        package
        for package in lock["package"]
        if package["name"] == "lancedb-robotics"
    )

    assert NATIVE_BENCH_EXTRA in package["metadata"]["provides-extras"]
    locked_requirements = {
        item["name"]
        for item in package["optional-dependencies"][NATIVE_BENCH_EXTRA]
    }
    assert locked_requirements == {"lerobot", "torch", "av", "numpy"}

    requires_dist = package["metadata"]["requires-dist"]
    marked = {
        item["name"]
        for item in requires_dist
        if item.get("marker") == f"extra == '{NATIVE_BENCH_EXTRA}'"
    }
    assert {"lerobot", "torch", "av", "numpy"} <= marked


def test_lerobot_native_dependency_status_reports_install_policy(monkeypatch):
    import lancedb_robotics.benchmark as benchmark

    monkeypatch.setattr(benchmark, "_module_available", lambda module: False)

    status = benchmark._lerobot_native_dependency_status()

    assert status["install"] == "install `lancedb-robotics[lerobot-native-bench]`"
    assert status["install_policy"]["extra"] == NATIVE_BENCH_EXTRA
    assert status["install_policy"]["uv_sync"] == "uv sync --extra lerobot-native-bench"
    assert status["install_policy"]["declared_decode_backend"] == "pyav"
    assert status["install_policy"]["optional_probe_backends"] == ["torchcodec"]
    assert "PyAV" in status["decode_backend"]["policy"]

    versions = benchmark._format_versions()
    native_policy = versions["lerobot-native"]["install_policy"]
    assert native_policy["extra"] == NATIVE_BENCH_EXTRA
    assert native_policy["declared_decode_backend"] == "pyav"
