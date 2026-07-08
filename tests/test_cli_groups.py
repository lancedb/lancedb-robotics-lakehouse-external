import pytest
from typer.testing import CliRunner

from lancedb_robotics.cli import COMMAND_GROUPS, app

runner = CliRunner()


@pytest.mark.parametrize("group", list(COMMAND_GROUPS))
def test_group_exposes_help(group):
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.parametrize("group", list(COMMAND_GROUPS))
def test_group_help_includes_description(group):
    result = runner.invoke(app, [group, "--help"])
    assert COMMAND_GROUPS[group].split()[0] in result.output
