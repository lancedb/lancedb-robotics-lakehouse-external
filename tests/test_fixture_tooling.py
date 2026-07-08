"""Fixture-tooling: the make_* scripts reproduce their fixtures byte-for-byte (backlog 0017).

The committed compression/integrity fixtures are only trustworthy if their
generators are deterministic. The corpus-free generators (ros2/cdr, truncated,
crc-corrupt) are reproduced fully here; the corpus-sliced generators (zstd, lz4)
are reproduced when the source corpus is present and skipped otherwise (the
multi-GB corpora are not in the repo).
"""

import importlib.util
from pathlib import Path

import pytest

from lancedb_robotics.adapters import get_adapter

FIXTURES = Path(__file__).parent / "fixtures"


def _load(script_name: str):
    spec = importlib.util.spec_from_file_location(script_name, FIXTURES / f"{script_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- corpus-free generators: fully reproduced ----------------------------


def test_ros2_imu_fixture_is_byte_stable():
    pytest.importorskip("mcap_ros2")
    module = _load("make_ros2_imu_mcap")
    assert module.build_bytes() == module.OUT.read_bytes()


def test_truncated_fixture_is_byte_stable():
    module = _load("make_truncated_mcap")
    assert module.build_bytes() == module.OUT.read_bytes()


def test_crc_corrupt_fixture_is_byte_stable():
    module = _load("make_crc_corrupt_mcap")
    assert module.build_bytes() == module.OUT.read_bytes()


def test_flatbuffer_foxglove_fixture_is_byte_stable():
    # The flatbuffer fixture is generated from the Foxglove .bfbs schemas
    # (backlog 0020); it reproduces exactly so the committed file is trustworthy.
    pytest.importorskip("flatbuffers")
    pytest.importorskip("foxglove_schemas_flatbuffer")
    module = _load("make_flatbuffer_foxglove_mcap")
    assert module.build_bytes() == module.OUT.read_bytes()


def test_split_recording_fixture_is_byte_stable():
    # The split-recording shards and their metadata.yaml reproduce exactly
    # (backlog 0019), so the committed directory is trustworthy.
    module = _load("make_split_recording")
    for index, name in enumerate(module.SHARD_NAMES):
        assert module.build_shard_bytes(index) == (module.OUT_DIR / name).read_bytes()
    assert module.build_metadata_yaml() == (module.OUT_DIR / module.METADATA_NAME).read_text()


# --- corpus-sliced generators: reproduced when the corpus is present ------


def _assert_slice_reproducible(script_name: str, compression: str, tmp_path: Path):
    module = _load(script_name)
    source = module.DEFAULT_SOURCE
    if not source.is_file():
        pytest.skip(f"corpus not present: {source}")
    adapter = get_adapter("mcap")
    info = adapter.inspect(source)
    out = tmp_path / module.OUT.name
    adapter.export(
        source,
        start_time_ns=info["start_time_ns"],
        end_time_ns=info["start_time_ns"] + module.WINDOW_NS,
        out_path=out,
        topics=module.TOPICS,
        compression=compression,
    )
    assert out.read_bytes() == module.OUT.read_bytes()


def test_zstd_slice_is_byte_stable_from_corpus(tmp_path):
    _assert_slice_reproducible("make_zstd_slice_mcap", "zstd", tmp_path)


def test_lz4_slice_is_byte_stable_from_corpus(tmp_path):
    _assert_slice_reproducible("make_lz4_slice_mcap", "lz4", tmp_path)


# --- the committed fixtures stay small ------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "slice_ros1_zstd.mcap",
        "slice_mixed_lz4.mcap",
        "ros2_imu.mcap",
        "truncated.mcap",
        "crc_corrupt.mcap",
        "split_recording/recording_0.mcap",
        "split_recording/recording_1.mcap",
        "split_recording/recording_2.mcap",
    ],
)
def test_fixtures_are_small(name):
    # Genuine bytes, not gigabytes: every CI fixture stays well under 64 KB.
    assert (FIXTURES / name).stat().st_size < 64 * 1024
