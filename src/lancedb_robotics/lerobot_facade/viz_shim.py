"""Temporary ``lancedb-robotics-lerobot-viz`` console shim (backlog 0492).

``lerobot-dataset-viz`` constructs ``LeRobotDataset`` directly, so a storage
format is visible to it only if something in the process has already imported
the module that registers it. A library caller can do that; a console script
user cannot. Upstream entry-point discovery (``huggingface/lerobot`` PR
#4576, open) closes the gap by scanning the ``lerobot.dataset_readers``
group -- which this package declares in ``pyproject.toml`` -- on the first
registry lookup. Until a lerobot release ships that patch, Foxglove playback
against a published lakehouse view cannot work through lerobot's own CLI.

This shim is the missing seam. It does exactly two things before handing
``sys.argv`` to lerobot's own ``lerobot-dataset-viz`` parser:

1. imports the reader, registering ``"lancedb_robotics"``;
2. pre-localizes a URI ``--root``: the upstream CLI parses ``--root`` with
   ``type=Path``, and ``Path("file://lake")`` collapses to ``file:/lake``
   before the remote-root check (``"://" in root``) can see it, so a lake URI
   handed to the CLI falls through to a Hugging Face Hub lookup and 404s.
   The shim materializes ``meta/`` through lerobot's own
   ``localize_remote_root`` (which probes the reader registered in step 1)
   and delegates a plain local directory instead.

::

    lancedb-robotics-lerobot-viz \\
        --repo-id acme/pick-place-v1 \\
        --root file:///path/to/robot.lance \\
        --episode-index 0 \\
        --display-mode foxglove

**TEMPORARY** -- delete this module and its ``[project.scripts]`` entry once a
released ``lerobot`` both discovers the ``lerobot.dataset_readers``
entry-point group *and* accepts URI roots in ``lerobot-dataset-viz`` (backlog
0510 tracks the retirement and both preconditions). The explicit
import stays harmless in the meantime, even under a discovery-capable
lerobot: ``register_dataset_reader`` treats re-registering the same
name -> module pair as a no-op, so discovery finding the entry point after
this shim already registered it raises nothing and warns nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

_INSTALL_HINT = (
    "lancedb-robotics-lerobot-viz requires a `lerobot` with the storage-format "
    "registry (huggingface/lerobot PR #4363, merged upstream but not in any "
    "PyPI release yet). Install the dev-only, commit-pinned "
    "`lancedb-robotics[lerobot-main-dev]` extra into an isolated venv (see "
    "pyproject.toml), plus `foxglove-sdk` and `opencv-python` if you want "
    "`--display-mode foxglove`.\n"
    "Underlying import error: {error}"
)


def _delegated_argv(argv: list[str], localize: Callable[..., object]) -> list[str]:
    """Return ``argv`` with a URI ``--root`` replaced by its localized directory.

    A local-path (or absent) ``--root`` passes through untouched. A URI root is
    materialized via ``localize`` -- lerobot's ``localize_remote_root`` in real
    use -- because the upstream CLI's ``type=Path`` conversion mangles URIs
    before its own remote-root detection can run (see module docstring).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    known, rest = parser.parse_known_args(argv)
    if known.root is None or "://" not in known.root:
        return argv
    local_root = localize(known.repo_id, known.root)
    rewritten = [] if known.repo_id is None else ["--repo-id", known.repo_id]
    return [*rewritten, "--root", str(local_root), *rest]


def main() -> None:
    """Register the ``lancedb_robotics`` format, then run ``lerobot-dataset-viz``.

    Registering before delegating is the entire point of the shim: lerobot's
    own CLI never imports third-party reader modules (until PR #4576 ships),
    so without this import the dataset open fails with
    ``Unknown storage_format 'lancedb_robotics'``.
    """
    import sys

    try:
        from lerobot.datasets.storage import localize_remote_root
        from lerobot.scripts.lerobot_dataset_viz import main as viz_main

        import lancedb_robotics.lerobot_facade.dataset_reader  # noqa: F401
    except ImportError as error:
        # Covers both shapes: lerobot absent entirely, and a released lerobot
        # (<= 0.5.x) that predates the storage-format registry the reader
        # module imports at its own top level.
        raise SystemExit(_INSTALL_HINT.format(error=error)) from error
    sys.argv[1:] = _delegated_argv(sys.argv[1:], localize_remote_root)
    viz_main()


if __name__ == "__main__":  # pragma: no cover -- exercised via subprocess in tests
    main()
