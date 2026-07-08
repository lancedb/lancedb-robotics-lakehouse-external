"""Quality validation and quarantine tests (backlog 0005).

sample.mcap is the good fixture (3 `/imu` json + 2 `/camera/front` cbor,
strictly increasing timestamps). incomplete.mcap is the intentionally bad one
(2 `lcm` `/imu` messages sharing one timestamp, no `/camera/front`), so against
the `demo` profile it deterministically fails required-topics,
monotonic-timestamps, and decodable-streams while time-range-overlap passes.
(`lcm` is outside the MCAP registry, so it stays undecodable even though every
registry encoding is now decoded -- backlog 0020.)
"""

import json

import pytest

from lancedb_robotics.ingest import ingest_mcap
from lancedb_robotics.lake import Lake
from lancedb_robotics.quality import (
    ProfileError,
    QualityError,
    apply_quality_results,
    quarantined_run_ids,
    resolve_profile,
    validate_lake,
    validate_run,
)

BASE_NS = 1_700_000_000_000_000_000  # matches both fixture generators


@pytest.fixture
def lake(tmp_path):
    return Lake.init(tmp_path / "robot.lance")


@pytest.fixture
def good_run(lake, fixtures_dir):
    return ingest_mcap(lake, fixtures_dir / "sample.mcap").run_id


@pytest.fixture
def bad_run(lake, fixtures_dir):
    return ingest_mcap(lake, fixtures_dir / "incomplete.mcap").run_id


@pytest.fixture
def demo_profile():
    return resolve_profile("demo")


class TestProfiles:
    def test_demo_profile_pins_required_topics(self, demo_profile):
        assert demo_profile.name == "demo"
        assert {(t.topic, t.min_count) for t in demo_profile.required_topics} == {
            ("/imu", 3),
            ("/camera/front", 2),
        }
        assert "/imu" in demo_profile.decodable_topics

    def test_unknown_profile_raises(self):
        with pytest.raises(ProfileError, match="demo"):
            resolve_profile("nope")

    def test_profile_loads_from_json_file(self, tmp_path):
        spec = {
            "name": "custom",
            "required_topics": [{"topic": "/imu", "min_count": 1}],
            "decodable_topics": [],
        }
        path = tmp_path / "custom.json"
        path.write_text(json.dumps(spec))
        profile = resolve_profile(str(path))
        assert profile.name == "custom"
        assert [(t.topic, t.min_count) for t in profile.required_topics] == [("/imu", 1)]

    def test_invalid_profile_file_raises(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json")
        with pytest.raises(ProfileError):
            resolve_profile(str(path))


class TestRules:
    def test_good_fixture_passes_every_rule(self, lake, good_run, demo_profile):
        report = validate_run(lake, good_run, demo_profile)
        assert report.run_id == good_run
        assert report.passed is True
        assert {r.rule: r.status for r in report.rules} == {
            "required-topics": "passed",
            "monotonic-timestamps": "passed",
            "time-range-overlap": "passed",
            "decodable-streams": "passed",
            "byte-integrity": "passed",
        }

    def test_incomplete_fixture_fails_expected_rules(self, lake, bad_run, demo_profile):
        report = validate_run(lake, bad_run, demo_profile)
        assert report.passed is False
        assert {r.rule: r.status for r in report.rules} == {
            "required-topics": "failed",
            "monotonic-timestamps": "failed",
            "time-range-overlap": "passed",
            "decodable-streams": "failed",
            # incomplete.mcap is a structurally valid file (just bad data), so the
            # byte-integrity rule passes -- it fails only on CRC/truncation damage.
            "byte-integrity": "passed",
        }

    def test_failure_details_name_topics_and_reasons(self, lake, bad_run, demo_profile):
        report = validate_run(lake, bad_run, demo_profile)
        by_rule = {r.rule: r for r in report.rules}
        required = " ".join(by_rule["required-topics"].details)
        assert "/camera/front" in required and "missing" in required
        assert "/imu" in required and "3" in required
        assert "/imu" in " ".join(by_rule["monotonic-timestamps"].details)
        # incomplete.mcap's /imu uses 'lcm', an encoding outside the MCAP registry
        # with no decoder (backlog 0020), so decodable-streams names it.
        assert "lcm" in " ".join(by_rule["decodable-streams"].details)

    def test_report_is_deterministic(self, lake, bad_run, demo_profile):
        first = validate_run(lake, bad_run, demo_profile)
        second = validate_run(lake, bad_run, demo_profile)
        assert first.to_dict() == second.to_dict()

    def test_overlap_rule_fails_on_disjoint_topic_ranges(self, lake, tmp_path):
        _write_disjoint_mcap(tmp_path / "disjoint.mcap")
        run_id = ingest_mcap(lake, tmp_path / "disjoint.mcap").run_id
        profile = _profile_from_dict(
            {
                "name": "disjoint",
                "required_topics": [
                    {"topic": "/a", "min_count": 1},
                    {"topic": "/b", "min_count": 1},
                ],
            }
        )
        report = validate_run(lake, run_id, profile)
        by_rule = {r.rule: r for r in report.rules}
        assert by_rule["time-range-overlap"].status == "failed"
        assert report.passed is False

    def test_decode_rule_skips_when_raw_file_is_gone(self, lake, tmp_path, fixtures_dir):
        moved = tmp_path / "moved.mcap"
        moved.write_bytes((fixtures_dir / "sample.mcap").read_bytes())
        run_id = ingest_mcap(lake, moved).run_id
        moved.unlink()
        report = validate_run(lake, run_id, resolve_profile("demo"))
        by_rule = {r.rule: r for r in report.rules}
        assert by_rule["decodable-streams"].status == "skipped"
        # A skipped rule does not fail the run by itself.
        assert {r.status for r in report.rules} == {"passed", "skipped"}

    def test_validate_unknown_run_raises(self, lake, demo_profile):
        with pytest.raises(QualityError, match="run-nope"):
            validate_run(lake, "run-nope", demo_profile)


class TestValidateLake:
    def test_validates_every_run_sorted_by_run_id(self, lake, good_run, bad_run, demo_profile):
        reports = validate_lake(lake, demo_profile)
        assert [r.run_id for r in reports] == sorted([good_run, bad_run])
        assert {r.run_id: r.passed for r in reports} == {good_run: True, bad_run: False}

    def test_single_run_selection(self, lake, good_run, bad_run, demo_profile):
        reports = validate_lake(lake, demo_profile, run_id=bad_run)
        assert [r.run_id for r in reports] == [bad_run]

    def test_empty_lake_returns_no_reports(self, lake, demo_profile):
        assert validate_lake(lake, demo_profile) == []


class TestApplyResults:
    @pytest.fixture
    def applied(self, lake, good_run, bad_run, demo_profile):
        reports = validate_lake(lake, demo_profile)
        apply_quality_results(lake, reports, demo_profile)
        return reports

    def _run_flags(self, lake, run_id):
        rows = lake.table("runs").to_arrow().to_pylist()
        return next(r for r in rows if r["run_id"] == run_id)["quality_flags"]

    def test_passed_run_gets_passed_flag(self, applied, lake, good_run):
        assert self._run_flags(lake, good_run) == ["quality:demo:passed"]

    def test_failed_run_gets_failure_flags_and_quarantine(self, applied, lake, bad_run):
        flags = self._run_flags(lake, bad_run)
        assert "quality:demo:failed" in flags
        assert "quarantined" in flags
        assert "quality:failed:required-topics" in flags
        assert "quality:failed:monotonic-timestamps" in flags
        assert "quality:failed:decodable-streams" in flags

    def test_quarantined_run_ids_reads_back(self, applied, lake, good_run, bad_run):
        assert quarantined_run_ids(lake) == [bad_run]

    def test_failing_topic_observations_are_flagged(self, applied, lake, good_run, bad_run):
        rows = lake.table("observations").to_arrow().to_pylist()
        bad_imu = [r for r in rows if r["run_id"] == bad_run and r["topic"] == "/imu"]
        assert bad_imu
        for row in bad_imu:
            assert "quality:failed:monotonic-timestamps" in row["quality_flags"]
            assert "quality:failed:decodable-streams" in row["quality_flags"]
        for row in (r for r in rows if r["run_id"] == good_run):
            assert not row["quality_flags"]

    def test_quality_transform_lineage_carries_report(self, applied, lake, bad_run):
        transforms = [
            t for t in lake.table("transform_runs").to_arrow().to_pylist() if t["kind"] == "quality"
        ]
        assert len(transforms) == 2  # one per validated run
        by_run = {json.loads(t["params"])["run_id"]: t for t in transforms}
        params = json.loads(by_run[bad_run]["params"])
        assert params["profile"] == "demo"
        assert params["report"]["passed"] is False
        assert by_run[bad_run]["status"] == "completed"
        assert by_run[bad_run]["output_tables"] == ["runs", "observations"]

    def test_revalidation_is_latest_wins(self, applied, lake, bad_run, demo_profile):
        reports = validate_lake(lake, demo_profile)
        apply_quality_results(lake, reports, demo_profile)
        flags = self._run_flags(lake, bad_run)
        assert flags.count("quarantined") == 1  # no duplicate flags on re-apply
        quality_transforms = [
            t for t in lake.table("transform_runs").to_arrow().to_pylist() if t["kind"] == "quality"
        ]
        assert len(quality_transforms) == 2  # still one per (run, profile)

    def test_no_quarantine_keeps_failure_flags_only(self, lake, bad_run, demo_profile):
        reports = validate_lake(lake, demo_profile, run_id=bad_run)
        apply_quality_results(lake, reports, demo_profile, quarantine=False)
        flags = self._run_flags(lake, bad_run)
        assert "quality:demo:failed" in flags
        assert "quarantined" not in flags
        assert quarantined_run_ids(lake) == []


def _profile_from_dict(spec):
    from lancedb_robotics.quality import ValidationProfile

    return ValidationProfile.from_dict(spec)


def _write_disjoint_mcap(path):
    """Two json topics whose time ranges do not overlap at all."""
    import json as _json

    from mcap.writer import CompressionType, Writer

    with path.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start(profile="", library="lancedb-robotics-fixture")
        for topic, offset in (("/a", 0), ("/b", 10_000_000_000)):
            schema = writer.register_schema(
                name=f"sample.{topic.strip('/').upper()}",
                encoding="jsonschema",
                data=_json.dumps({"type": "object"}).encode(),
            )
            channel = writer.register_channel(
                topic=topic, message_encoding="json", schema_id=schema
            )
            for i in range(2):
                log_time = BASE_NS + offset + i * 100_000_000
                writer.add_message(
                    channel_id=channel,
                    log_time=log_time,
                    publish_time=log_time,
                    data=_json.dumps({"i": i}).encode(),
                )
        writer.finish()
