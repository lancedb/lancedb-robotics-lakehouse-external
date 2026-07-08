"""Feature Set 0 integration check: the CLI help path works from a clean directory.

Narrative: the lakehouse shell exists; these commands are the demo spine.
"""

import subprocess
import sys


def test_cli_help_from_clean_temp_dir(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "lancedb_robotics", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    for group in ["lake", "ingest", "search", "dataset", "train", "export"]:
        assert group in result.stdout
