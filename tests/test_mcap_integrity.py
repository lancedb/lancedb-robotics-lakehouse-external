"""Byte-integrity coverage: CRC mismatch and truncation (backlog 0017).

``crc_corrupt.mcap`` is a structurally complete file with one flipped chunk byte;
``truncated.mcap`` is a multi-chunk file with its tail chopped off. Neither may
silently pass: the CRC file is quarantined while still surfacing the prefix it
could verify, and the truncated file recovers its readable prefix and quarantines
the rest -- never a hard crash on the whole file.
"""

import json

import pytest

from lancedb_robotics.adapters import CorruptMcapError, get_adapter
from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.quality import (
    DEMO_PROFILE,
    QUARANTINED_FLAG,
    apply_quality_results,
    validate_lake,
)


@pytest.fixture
def adapter():
    return get_adapter("mcap")


@pytest.fixture
def crc_corrupt(fixtures_dir):
    return fixtures_dir / "crc_corrupt.mcap"


@pytest.fixture
def truncated(fixtures_dir):
    return fixtures_dir / "truncated.mcap"


@pytest.fixture
def lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


def _run(lake, run_id):
    return next(r for r in lake.table("runs").to_arrow().to_pylist() if r["run_id"] == run_id)


def _run_metadata(run):
    return {e["key"]: e["value"] for e in run["metadata"]}


# --- CRC mismatch ---------------------------------------------------------


def test_crc_mismatch_quarantines_run(lake, crc_corrupt):
    report = ingest_mcap(lake, crc_corrupt)
    assert report.quarantined is True
    assert report.integrity_status == "crc-mismatch"
    run = _run(lake, report.run_id)
    assert QUARANTINED_FLAG in run["quality_flags"]
    assert "integrity:crc-mismatch" in run["quality_flags"]
    assert _run_metadata(run)["integrity.status"] == "crc-mismatch"


def test_crc_mismatch_keeps_verified_prefix_not_whole_file(lake, crc_corrupt):
    report = ingest_mcap(lake, crc_corrupt)
    written = lake.table("observations").count_rows(f"run_id = '{report.run_id}'")
    # The leading good chunks are kept; the run is not lost, and not the full file.
    assert 0 < written == report.recovered_count


def test_crc_validation_is_opt_out(adapter, crc_corrupt):
    # The corruption only surfaces when CRCs are checked; trusted-data callers can
    # skip the check and read every message (the existing pre-0017 behavior).
    silent = list(adapter.ingest(crc_corrupt, validate_crcs=False))
    assert len(silent) > 0
    with pytest.raises(CorruptMcapError):
        list(adapter.ingest(crc_corrupt, validate_crcs=True))


def test_crc_corrupt_inspect_still_succeeds(adapter, crc_corrupt):
    # Inspect does not validate CRCs (it is the cheap planning pass), so the run
    # is created and then quarantined during ingest -- not refused up front.
    info = adapter.inspect(crc_corrupt)
    assert info["message_count"] > 0


# --- truncation -----------------------------------------------------------


def test_truncated_recovers_prefix_and_quarantines(lake, truncated):
    report = ingest_mcap(lake, truncated)
    assert report.quarantined is True
    assert report.integrity_status == "truncated"
    written = lake.table("observations").count_rows(f"run_id = '{report.run_id}'")
    assert written == report.recovered_count > 0
    run = _run(lake, report.run_id)
    assert QUARANTINED_FLAG in run["quality_flags"]
    # Recovered rows are real decoded observations, not placeholders.
    rows = lake.table("observations").to_arrow().to_pylist()
    assert all(json.loads(r["payload_json"]) for r in rows if r["payload_json"])


def test_truncated_does_not_abort_the_run(lake, truncated):
    # The whole point: a truncated file yields a (quarantined) run, never an
    # exception that loses everything.
    report = ingest_mcap(lake, truncated)
    assert lake.table("runs").count_rows(f"run_id = '{report.run_id}'") == 1


def test_truncated_run_records_recovered_count_in_transform(lake, truncated):
    report = ingest_mcap(lake, truncated)
    ingest = next(
        t for t in lake.table("transform_runs").to_arrow().to_pylist() if t["kind"] == "ingest"
    )
    params = json.loads(ingest["params"])
    assert params["integrity"]["status"] == "truncated"
    assert params["integrity"]["recovered"] == report.recovered_count
    assert ingest["status"] == "recovered"


# --- quality gate honors integrity across re-validation -------------------


def test_quality_revalidation_keeps_damaged_run_quarantined(lake, crc_corrupt):
    report = ingest_mcap(lake, crc_corrupt)
    # The quality gate overwrites quality_flags ("latest validation wins"); the
    # byte-integrity rule must re-derive the quarantine from run metadata.
    reports = validate_lake(lake, DEMO_PROFILE, run_id=report.run_id)
    apply_quality_results(lake, reports, DEMO_PROFILE)
    run = _run(lake, report.run_id)
    assert QUARANTINED_FLAG in run["quality_flags"]
    assert any(r.rule == "byte-integrity" and r.status == "failed" for r in reports[0].rules)


def test_clean_run_passes_byte_integrity(lake, fixtures_dir):
    report = ingest_mcap(lake, fixtures_dir / "sample.mcap")
    assert report.quarantined is False
    reports = validate_lake(lake, DEMO_PROFILE, run_id=report.run_id)
    integrity = next(r for r in reports[0].rules if r.rule == "byte-integrity")
    assert integrity.status == "passed"
