from conftest import assert_matches_snapshot
from typer.testing import CliRunner

from lancedb_robotics.cli import COMMAND_GROUPS, app

runner = CliRunner()

# The demo-spine command groups from the baseline showcase plan. This list is
# the contract: changing it is a deliberate product decision, not a refactor.
EXPECTED_GROUPS = [
    "lake",
    "inspect",
    "ingest",
    "quality",
    "align",
    "scenarios",
    "episodes",
    "video",
    "embed",
    "search",
    "curate",
    "gaps",
    "dataset",
    "train",
    "export",
    "lineage",
    "bench",
    "writeback",
]


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_version_flag():
    import lancedb_robotics

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == lancedb_robotics.__version__


def test_registered_groups_match_contract():
    assert list(COMMAND_GROUPS) == EXPECTED_GROUPS


def test_help_lists_every_command_group():
    result = runner.invoke(app, ["--help"])
    for group in EXPECTED_GROUPS:
        assert group in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage:" in result.output


def test_top_level_help_snapshot():
    result = runner.invoke(app, ["--help"])
    assert_matches_snapshot("cli_top_level_help.txt", result.output)
