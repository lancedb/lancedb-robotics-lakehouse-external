"""Regenerate tests/fixtures/slice_ros1_zstd.mcap from a real ros1msg corpus (backlog 0017).

Run: uv run python tests/fixtures/make_zstd_slice_mcap.py [SOURCE.mcap]

A tiny, deterministic slice of genuine ``ros1msg`` + **zstd** bytes, cut from a
real corpus file with the lossless ``export`` (0011) -- so the codec and ros1
decode paths are tested on real data without checking in gigabytes. The source
defaults to ``data/demo/demo.mcap`` (ros1msg, zstd, 59 MB); the didi corpus has
the identical encoding+compression shape and works the same way -- pass its path
to slice from didi instead.

The slice keeps a narrow time window and only the small, non-binary topics
(diagnostics + radar range/tracks), so the fixture stays a few KB while still
exercising multi-topic decode and typed extraction. Slicing is byte-stable: the
window/topic selection is fixed and the writer is deterministic, so re-running
reproduces the same bytes.
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


DEFAULT_SOURCE = _find_corpus("demo", "demo.mcap")
OUT = Path(__file__).parent / "slice_ros1_zstd.mcap"

# Small, non-binary ros1msg topics present in demo/didi. Excludes the
# image/pointcloud topics so the fixture stays tiny.
TOPICS = ("/diagnostics", "/radar/range", "/radar/tracks")
WINDOW_NS = 200_000_000  # 0.2 s from the first message


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source.is_file():
        raise SystemExit(f"source corpus not found: {source}\n(pass a path, e.g. data/didi/...mcap)")
    adapter = get_adapter("mcap")
    info = adapter.inspect(source)
    start = info["start_time_ns"]
    result = adapter.export(
        source,
        start_time_ns=start,
        end_time_ns=start + WINDOW_NS,
        out_path=OUT,
        topics=TOPICS,
        compression="zstd",
    )
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes): "
        f"{result['message_count']} messages, topics={result['topics']}"
    )


if __name__ == "__main__":
    main()
