"""Regenerate tests/fixtures/slice_mixed_lz4.mcap from the nuScenes corpus (backlog 0017).

Run: uv run python tests/fixtures/make_lz4_slice_mcap.py [SOURCE.mcap]

A tiny, deterministic slice of a real nuScenes scene, re-compressed with **lz4**
(nuScenes' native codec) by the lossless ``export`` (0011). One file, two
message encodings in the same lz4 chunks -- ``json`` (/imu, /odom) and
``protobuf`` (/gps, /pose) -- so the lz4 codec path and the mixed-encoding decode
dispatch are both tested on genuine bytes. (nuScenes also carries chatty ``ros1``
/diagnostics; it is excluded to keep the slice tiny, and ros1 decode is already
covered by the demo zstd slice.)

The slice keeps a 70 ms window (long enough to include the first /imu, which
starts ~53 ms in) and only small, scalar topics (no camera/lidar), so the fixture
stays a few KB while still covering json + protobuf decode and typed extraction
(imu + gps). Byte-stable: fixed window/topics + deterministic writer.
"""

import sys
from pathlib import Path

from lancedb_robotics.adapters import get_adapter


def _find_corpus(*parts: str) -> Path:
    """Locate ``data/<parts...>`` by walking up from this file.

    The corpus is gitignored and lives at the real repo root. Inside a nested
    git worktree that root is an ancestor of the worktree, not the worktree
    itself, so search upward rather than assuming a fixed depth.
    """
    rel = Path("data", *parts)
    for base in Path(__file__).resolve().parents:
        if (base / rel).is_file():
            return base / rel
    return Path(__file__).resolve().parents[2] / rel  # conventional fallback


DEFAULT_SOURCE = _find_corpus("nuScenes", "NuScenes-v1.0-mini-scene-0553.mcap")
OUT = Path(__file__).parent / "slice_mixed_lz4.mcap"

# json (/imu, /odom) + protobuf (/gps, /pose); all scalar (no image/pointcloud).
TOPICS = ("/imu", "/odom", "/gps", "/pose")
WINDOW_NS = 70_000_000  # 70 ms from the first message (first /imu is ~53 ms in)


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source.is_file():
        raise SystemExit(f"source corpus not found: {source}\n(pass a nuScenes .mcap path)")
    adapter = get_adapter("mcap")
    info = adapter.inspect(source)
    start = info["start_time_ns"]
    result = adapter.export(
        source,
        start_time_ns=start,
        end_time_ns=start + WINDOW_NS,
        out_path=OUT,
        topics=TOPICS,
        compression="lz4",
    )
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes): "
        f"{result['message_count']} messages, topics={result['topics']}"
    )


if __name__ == "__main__":
    main()
