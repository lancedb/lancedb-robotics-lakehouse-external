"""CLI tests for `lancedb-robotics inspect mcap` (backlog 0003)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from lancedb_robotics.cli import app

runner = CliRunner()

SAMPLE = str(Path(__file__).parent / "fixtures" / "sample.mcap")
RECORDS = str(Path(__file__).parent / "fixtures" / "records.mcap")

# Keys downstream ingest and contract tests rely on; removing one is a
# product decision, not a refactor.
TOP_LEVEL_KEYS = {
    "adapter",
    "path",
    "profile",
    "library",
    "message_count",
    "schema_count",
    "channel_count",
    "chunk_count",
    "start_time_ns",
    "end_time_ns",
    "duration_ns",
    # Backlog 0018: whether stats are summary-index-derived or scan-derived.
    "indexed",
    "topics",
    "chunks",
    # Backlog 0016: MCAP's other first-class record types.
    "attachments",
    "metadata",
}

TOPIC_KEYS = {
    "topic",
    "message_encoding",
    "schema_name",
    "schema_encoding",
    "message_count",
    "start_time_ns",
    "end_time_ns",
    "can_decode",
}


def test_inspect_mcap_json_is_valid_and_complete():
    result = runner.invoke(app, ["inspect", "mcap", SAMPLE, "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload) == TOP_LEVEL_KEYS
    for topic in payload["topics"]:
        assert set(topic) == TOPIC_KEYS


def test_inspect_mcap_defaults_to_json():
    result = runner.invoke(app, ["inspect", "mcap", SAMPLE])
    assert result.exit_code == 0
    assert json.loads(result.output)["adapter"] == "mcap"


def test_inspect_mcap_json_is_stable_across_runs():
    first = runner.invoke(app, ["inspect", "mcap", SAMPLE, "--format", "json"])
    second = runner.invoke(app, ["inspect", "mcap", SAMPLE, "--format", "json"])
    assert first.output == second.output


def test_inspect_mcap_text_format_summarizes_topics():
    result = runner.invoke(app, ["inspect", "mcap", SAMPLE, "--format", "text"])
    assert result.exit_code == 0
    assert "/imu" in result.output
    assert "/camera/front" in result.output
    assert "5 messages" in result.output


def test_inspect_mcap_text_format_summarizes_records():
    # Backlog 0016: attachment + metadata records appear in the human summary.
    result = runner.invoke(app, ["inspect", "mcap", RECORDS, "--format", "text"])
    assert result.exit_code == 0
    assert "metadata: scene-info" in result.output
    assert "attachment: calibration.json" in result.output
    assert "application/octet-stream" in result.output


def test_inspect_mcap_text_format_omits_empty_records():
    # A plain log prints no attachment/metadata lines at all.
    result = runner.invoke(app, ["inspect", "mcap", SAMPLE, "--format", "text"])
    assert result.exit_code == 0
    assert "metadata:" not in result.output
    assert "attachment:" not in result.output


def test_inspect_mcap_missing_file_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["inspect", "mcap", str(tmp_path / "nope.mcap")])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_inspect_mcap_unknown_format_fails_cleanly():
    result = runner.invoke(app, ["inspect", "mcap", SAMPLE, "--format", "yaml"])
    assert result.exit_code == 1
    assert "error:" in result.output
