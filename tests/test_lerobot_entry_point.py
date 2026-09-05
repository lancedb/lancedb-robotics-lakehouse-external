"""Entry-point registration for the lerobot adapter + interim viz shim (backlog 0492).

Two lanes, mirroring ``test_lerobot_dataset_reader.py``:

* Ungated: the declaration itself -- the ``lerobot.dataset_readers`` entry
  point's group, name, and target in ``pyproject.toml``, the target module's
  reader contract (checked by AST, never by import: the module hard-depends on
  lerobot), and the temporary ``lancedb-robotics-lerobot-viz`` shim staying
  import-light and failing actionably where lerobot's storage registry is
  missing. A package rename or module move breaks these tests before it can
  silently break discovery for installed clients.
* ``lerobot_main_dev``-marked: real end-to-end discovery against the pinned
  fork lerobot (``sarwarbhuiyan/lerobot@2a8c3cbc`` = upstream ``3f2c29ef`` +
  PR #4576's entry-point discovery). Installed dist metadata alone must make a
  subprocess that never imports this package resolve ``lancedb_robotics``
  through ``LeRobotDataset``, and the registry must read entry-point *names*
  without importing lancedb/pyarrow. These skip (or fail under
  ``LANCEDB_ROBOTICS_REQUIRE_LEROBOT_MAIN_DEV=1``) when the installed lerobot
  predates PR #4576 or when this package's dist metadata is not installed
  (e.g. a bare ``PYTHONPATH=src`` run: entry points live in ``*.dist-info``,
  so discovery is only observable from a venv with the package installed).
"""

import ast
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import require_lerobot_main_dev

from lancedb_robotics.lerobot_facade._reader_core import STORAGE_FORMAT

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REQUIRE_ENV = "LANCEDB_ROBOTICS_REQUIRE_LEROBOT_MAIN_DEV"

ENTRY_POINT_GROUP = "lerobot.dataset_readers"
READER_MODULE = "lancedb_robotics.lerobot_facade.dataset_reader"
VIZ_SHIM_SCRIPT = "lancedb-robotics-lerobot-viz"
VIZ_SHIM_TARGET = "lancedb_robotics.lerobot_facade.viz_shim:main"


def _pyproject() -> dict:
    return tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())


def _subprocess_env() -> dict[str, str]:
    """Child processes must resolve the same package tree as this process.

    Worktree runs execute against the main repo's venv with ``PYTHONPATH``
    pointing at the worktree's ``src`` (see the task-record conventions), so
    prepend the directory this process actually imported the package from.
    """
    import lancedb_robotics

    src_dir = str(Path(lancedb_robotics.__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ---------------------------------------------------------------------------
# Ungated: the declaration, checked against the source tree
# ---------------------------------------------------------------------------


def test_entry_point_declared_and_named_after_storage_format():
    """The group lerobot PR #4576 scans, the exact format name, our module."""
    entry_points = _pyproject()["project"]["entry-points"][ENTRY_POINT_GROUP]
    assert entry_points == {STORAGE_FORMAT: READER_MODULE}


def test_entry_point_target_module_satisfies_reader_contract():
    """The declared module binds ``DATASET_READER`` and ``localize_root``.

    Checked by AST, not by import: the module imports lerobot's registry at
    its own top level, which no released lerobot provides. Discovery
    (`storage.py`) imports the module and reads exactly these two names, so a
    rename or move that breaks either would strand every installed client.
    """
    module_path = _REPO_ROOT / "src" / Path(*READER_MODULE.split(".")).with_suffix(".py")
    assert module_path.is_file(), f"entry point targets a missing module: {module_path}"

    tree = ast.parse(module_path.read_text())
    top_level_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            top_level_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_level_names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            top_level_names.update(alias.asname or alias.name for alias in node.names)
    assert "DATASET_READER" in top_level_names
    assert "localize_root" in top_level_names


def test_viz_shim_console_script_declared():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts[VIZ_SHIM_SCRIPT] == VIZ_SHIM_TARGET


def test_viz_shim_module_imports_without_lerobot():
    """The shim module itself must not touch lerobot (or the training stack).

    The lerobot import is deferred into ``main()`` so the console script can be
    declared unconditionally and fail with an actionable message rather than
    at import. (lancedb/pyarrow do load here -- any in-package import pays for
    the package root's lake exports; the entry-point path's lazy-import
    guarantee is asserted by the gated registry test below, in a plain lerobot
    process that never imports this package.)
    """
    probe = (
        "import sys\n"
        "import lancedb_robotics.lerobot_facade.viz_shim as shim\n"
        "assert callable(shim.main)\n"
        "banned = ('lerobot', 'torch')\n"
        "loaded = [m for m in sys.modules for b in banned "
        "if m == b or m.startswith(b + '.')]\n"
        "assert not loaded, f'shim import dragged in {loaded}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_viz_shim_errors_actionably_without_storage_registry():
    """Absent/too-old lerobot -> SystemExit naming the extra, not a traceback."""
    try:
        registry_available = importlib.util.find_spec("lerobot.datasets.storage") is not None
    except ModuleNotFoundError:
        registry_available = False
    if registry_available:
        pytest.skip(
            "installed lerobot has the storage registry; the shim error path is "
            "unreachable here (its success path is covered by the gated lane)"
        )

    from lancedb_robotics.lerobot_facade import viz_shim

    with pytest.raises(SystemExit) as excinfo:
        viz_shim.main()
    message = str(excinfo.value)
    assert "lerobot-main-dev" in message
    assert "#4363" in message


def test_delegated_argv_passes_local_and_absent_roots_through():
    from lancedb_robotics.lerobot_facade import viz_shim

    def explode(*args):
        raise AssertionError("localize must not be called for a non-URI root")

    argv = ["--repo-id", "acme/x", "--root", "/data/x", "--episode-index", "0"]
    assert viz_shim._delegated_argv(argv, explode) == argv
    argv = ["--repo-id", "acme/x", "--episode-index", "0"]
    assert viz_shim._delegated_argv(argv, explode) == argv


def test_delegated_argv_localizes_uri_root():
    """A URI ``--root`` is materialized and replaced with a local directory.

    The upstream viz CLI parses ``--root`` with ``type=Path``, which collapses
    ``file://lake`` to ``file:/lake`` before its own remote-root check can
    see the scheme -- so the shim must localize first and delegate a plain
    directory, preserving every other argument.
    """
    from lancedb_robotics.lerobot_facade import viz_shim

    calls: list[tuple] = []

    def localize(repo_id, root):
        calls.append((repo_id, root))
        return "/cache/view-123"

    argv = [
        "--repo-id",
        "acme/x",
        "--root",
        "file:///lake.lance",
        "--episode-index",
        "3",
        "--display-mode",
        "foxglove",
    ]
    rewritten = viz_shim._delegated_argv(argv, localize)
    assert calls == [("acme/x", "file:///lake.lance")]
    assert rewritten == [
        "--repo-id",
        "acme/x",
        "--root",
        "/cache/view-123",
        "--episode-index",
        "3",
        "--display-mode",
        "foxglove",
    ]


# ---------------------------------------------------------------------------
# Real upstream discovery (gated -- see require_lerobot_main_dev)
# ---------------------------------------------------------------------------


def _require_entry_point_discovery() -> None:
    """Gate on the two extras the discovery tests need beyond the base lane.

    1. The installed lerobot must carry PR #4576 (the ``lerobot-main-dev``
       extra pins fork SHA ``2a8c3cbc`` for exactly this; an older pin has the
       registry but nothing that reads entry points).
    2. This package's dist metadata must be installed in the venv -- entry
       points live in ``*.dist-info``, so a bare ``PYTHONPATH=src`` run has
       nothing for lerobot to discover.
    """
    require_lerobot_main_dev()

    from lerobot.datasets import storage

    if not hasattr(storage, "DATASET_READER_ENTRY_POINT_GROUP"):
        reason = (
            "the installed lerobot predates PR #4576's entry-point discovery; "
            "reinstall the `lancedb-robotics[lerobot-main-dev]` extra (pinned to "
            "the fork SHA that carries it)"
        )
        if os.environ.get(_REQUIRE_ENV) == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    names = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP).names
    if STORAGE_FORMAT not in names:
        reason = (
            "lancedb-robotics dist metadata is not installed in this venv "
            "(entry points are only visible from an installed package, not from "
            "PYTHONPATH); `pip install -e .` into the gated venv"
        )
        if os.environ.get(_REQUIRE_ENV) == "1":
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.mark.lerobot_main_dev
def test_registry_reads_entry_point_names_without_importing_us():
    """Discovery must list ``lancedb_robotics`` without paying for it.

    The unknown-format error message is the one public surface that renders
    the registry's names without instantiating a reader: after it, nothing
    from this package's stack (lancedb_robotics, lancedb, pylance's ``lance``)
    may be imported. This is the acceptance criterion that installing the
    package does not change what unrelated lerobot commands load. pyarrow is
    deliberately not on the banned list: lerobot's own dataset stack imports
    it (parquet metadata) with or without this package installed.
    """
    _require_entry_point_discovery()

    probe = (
        "import sys\n"
        "from lerobot.datasets.storage import make_dataset_reader\n"
        "try:\n"
        "    make_dataset_reader('__no_such_format__')\n"
        "except ValueError as error:\n"
        "    message = str(error)\n"
        "else:\n"
        "    raise AssertionError('unknown format did not raise')\n"
        f"assert {STORAGE_FORMAT!r} in message, message\n"
        "banned = ('lancedb_robotics', 'lancedb', 'lance')\n"
        "loaded = [m for m in sys.modules for b in banned "
        "if m == b or m.startswith(b + '.')]\n"
        "assert not loaded, f'name scan imported {loaded}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.lerobot_main_dev
def test_published_view_opens_via_discovery_alone(tmp_path):
    """`pip install lancedb-robotics` + a lake URI is the whole client setup.

    The subprocess never imports ``lancedb_robotics``: entry-point discovery
    alone must resolve the format when ``LeRobotDataset`` opens a published
    view. Values are asserted against an in-process facade read of the same
    lake, so this also proves the discovered reader serves real frames.
    """
    _require_entry_point_discovery()

    from test_lerobot_facade import _two_episode_lake

    from lancedb_robotics.lerobot_facade import (
        CanonicalVectorMapping,
        LiveLeRobotFacade,
        publish_view,
    )

    lake_path = tmp_path / "robot.lance"
    lake = _two_episode_lake(lake_path)
    mapping = CanonicalVectorMapping(state_streams=("/gps", "/imu"), action_streams=("/action",))
    publish_view(lake, repo_id="acme/entry-point-v1", fps=20, mapping=mapping, name="facade_view")
    reference = LiveLeRobotFacade(lake, name="facade_view", mapping=mapping)

    probe = (
        "import json, sys\n"
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
        "assert not any(m == 'lancedb_robotics' or m.startswith('lancedb_robotics.') "
        "for m in sys.modules)\n"
        f"dataset = LeRobotDataset('acme/entry-point-v1', root='file://{lake_path}')\n"
        "item = dataset[0]\n"
        "print(json.dumps({\n"
        "    'len': len(dataset),\n"
        "    'num_episodes': dataset.num_episodes,\n"
        "    'state0': item['observation.state'].tolist(),\n"
        "    'task0': item['task'],\n"
        "}))\n"
    )
    env = _subprocess_env()
    env["LANCEDB_ROBOTICS_VIEW_CACHE"] = str(tmp_path / "view-cache")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    served = json.loads(result.stdout.splitlines()[-1])
    assert served["len"] == len(reference)
    assert served["num_episodes"] == 2
    assert served["state0"] == pytest.approx(reference[0]["observation.state"])
    assert served["task0"] == reference[0]["task"]


@pytest.mark.lerobot_main_dev
def test_viz_shim_delegates_to_lerobot_cli():
    """The shim reaches lerobot's own parser: `--help` renders its options."""
    require_lerobot_main_dev()

    result = subprocess.run(
        [sys.executable, "-m", "lancedb_robotics.lerobot_facade.viz_shim", "--help"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "--display-mode" in result.stdout
    assert "--episode-index" in result.stdout
