"""Quality validation and quarantine: the lake's first quality gate.

A :class:`ValidationProfile` declares what a healthy run looks like (required
topics with minimum counts, stream rules). :func:`validate_run` evaluates one
ingested run against a profile and returns a deterministic
:class:`RunQualityReport`; :func:`apply_quality_results` writes the verdict
back to the lake:

- ``runs.quality_flags`` is overwritten with the new verdict via in-place
  ``Table.update`` (latest validation wins; re-validating never stacks duplicate
  flags). ``runs`` has no blob column, so in-place update is fine there,
- ``observations.quality_flags`` is rewritten **additively** -- decision 0024 /
  backlog 0035. ``observations`` is blob-encoded (``payload_blob``), and in-place
  ``Table.update`` is not supported on a table with a blob column, so the verdict
  is recomputed for the whole column and written via ``drop_columns`` +
  ``add_columns`` on the underlying Lance dataset (a new, versioned table state
  that never reads or rewrites the blob bytes). Observations of runs not in this
  validation keep their existing flags,
- failed runs are flagged ``quarantined`` so later feature sets (scenarios,
  search, snapshots) can exclude them,
- one ``transform_runs`` row per (run, profile) records lineage with the full
  report JSON in ``params``.

Rules only read what ingest already wrote, with one exception: the
decodable-streams rule re-inspects the raw source file and is reported as
``skipped`` (not failed) when that file is no longer reachable.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA

QUARANTINED_FLAG = "quarantined"

RULE_REQUIRED_TOPICS = "required-topics"
RULE_MONOTONIC = "monotonic-timestamps"
RULE_OVERLAP = "time-range-overlap"
RULE_DECODABLE = "decodable-streams"
RULE_INTEGRITY = "byte-integrity"


class ProfileError(Exception):
    """Raised when a validation profile cannot be resolved or parsed."""


class QualityError(Exception):
    """Raised when validation cannot run (for example, an unknown run id)."""


@dataclass(frozen=True)
class RequiredTopic:
    """One topic the profile insists on, with a minimum message count."""

    topic: str
    min_count: int = 1


@dataclass(frozen=True)
class ValidationProfile:
    """What a healthy run looks like for one class of robot log."""

    name: str
    required_topics: tuple[RequiredTopic, ...]
    decodable_topics: tuple[str, ...] = ()
    require_monotonic: bool = True
    require_overlap: bool = True

    @classmethod
    def from_dict(cls, spec: dict) -> "ValidationProfile":
        try:
            return cls(
                name=spec["name"],
                required_topics=tuple(
                    RequiredTopic(topic=t["topic"], min_count=int(t.get("min_count", 1)))
                    for t in spec.get("required_topics", [])
                ),
                decodable_topics=tuple(spec.get("decodable_topics", [])),
                require_monotonic=bool(spec.get("require_monotonic", True)),
                require_overlap=bool(spec.get("require_overlap", True)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfileError(f"invalid profile spec: {exc}") from exc


# The built-in demo profile matches the showcase fixture: a healthy run has 3
# `/imu` messages we can decode natively and 2 `/camera/front` frames.
DEMO_PROFILE = ValidationProfile(
    name="demo",
    required_topics=(
        RequiredTopic("/imu", min_count=3),
        RequiredTopic("/camera/front", min_count=2),
    ),
    decodable_topics=("/imu",),
)

PROFILES: dict[str, ValidationProfile] = {"demo": DEMO_PROFILE}


def resolve_profile(name_or_path: str) -> ValidationProfile:
    """Look up a built-in profile by name, or load one from a JSON file path."""
    if name_or_path in PROFILES:
        return PROFILES[name_or_path]
    path = Path(name_or_path)
    if path.is_file():
        try:
            return ValidationProfile.from_dict(json.loads(path.read_text()))
        except json.JSONDecodeError as exc:
            raise ProfileError(f"profile file is not valid JSON: {path} ({exc})") from exc
    raise ProfileError(
        f"unknown profile {name_or_path!r}; "
        f"expected one of {sorted(PROFILES)} or a path to a profile JSON file"
    )


@dataclass(frozen=True)
class RuleResult:
    """Outcome of one rule: passed, failed (with details), or skipped."""

    rule: str
    status: str  # "passed" | "failed" | "skipped"
    details: tuple[str, ...] = ()
    failed_topics: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "status": self.status,
            "details": list(self.details),
            "failed_topics": list(self.failed_topics),
        }


@dataclass(frozen=True)
class RunQualityReport:
    """Deterministic verdict for one run against one profile."""

    run_id: str
    profile: str
    rules: tuple[RuleResult, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return all(rule.status != "failed" for rule in self.rules)

    @property
    def failed_rules(self) -> tuple[RuleResult, ...]:
        return tuple(rule for rule in self.rules if rule.status == "failed")

    def run_flags(self, *, quarantine: bool = True) -> list[str]:
        """The ``runs.quality_flags`` value this report implies."""
        if self.passed:
            return [f"quality:{self.profile}:passed"]
        flags = [f"quality:{self.profile}:failed"]
        flags += [f"quality:failed:{rule.rule}" for rule in self.failed_rules]
        if quarantine:
            flags.append(QUARANTINED_FLAG)
        return flags

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "profile": self.profile,
            "passed": self.passed,
            "rules": [rule.to_dict() for rule in self.rules],
        }


def _run_rows(lake: Lake) -> list[dict]:
    return sorted(lake.table("runs").to_arrow().to_pylist(), key=lambda r: r["run_id"])


def _observations_by_topic(lake: Lake, run_id: str) -> dict[str, list[dict]]:
    rows = [r for r in lake.table("observations").to_arrow().to_pylist() if r["run_id"] == run_id]
    by_topic: dict[str, list[dict]] = {}
    for row in rows:
        by_topic.setdefault(row["topic"], []).append(row)
    for topic_rows in by_topic.values():
        topic_rows.sort(key=lambda r: r["raw_sequence"])
    return by_topic


def _check_required_topics(
    profile: ValidationProfile, by_topic: dict[str, list[dict]]
) -> RuleResult:
    details: list[str] = []
    failed: list[str] = []
    for required in profile.required_topics:
        rows = by_topic.get(required.topic)
        if rows is None:
            details.append(f"{required.topic}: missing")
            failed.append(required.topic)
        elif len(rows) < required.min_count:
            details.append(
                f"{required.topic}: {len(rows)} messages < required {required.min_count}"
            )
            failed.append(required.topic)
    if failed:
        return RuleResult(RULE_REQUIRED_TOPICS, "failed", tuple(details), tuple(failed))
    return RuleResult(RULE_REQUIRED_TOPICS, "passed")


def _check_monotonic(by_topic: dict[str, list[dict]]) -> RuleResult:
    details: list[str] = []
    failed: list[str] = []
    for topic in sorted(by_topic):
        rows = by_topic[topic]
        for previous, current in zip(rows, rows[1:], strict=False):
            if current["timestamp_ns"] <= previous["timestamp_ns"]:
                details.append(
                    f"{topic}: timestamp not strictly increasing "
                    f"at sequence {current['raw_sequence']}"
                )
                failed.append(topic)
                break
    if failed:
        return RuleResult(RULE_MONOTONIC, "failed", tuple(details), tuple(failed))
    return RuleResult(RULE_MONOTONIC, "passed")


def _check_overlap(profile: ValidationProfile, by_topic: dict[str, list[dict]]) -> RuleResult:
    ranges = {
        required.topic: (
            min(r["timestamp_ns"] for r in by_topic[required.topic]),
            max(r["timestamp_ns"] for r in by_topic[required.topic]),
        )
        for required in profile.required_topics
        if by_topic.get(required.topic)
    }
    if len(ranges) < 2:
        return RuleResult(
            RULE_OVERLAP, "passed", ("fewer than 2 required topics present; nothing to compare",)
        )
    # Intervals pairwise-overlap iff the latest start is no later than the
    # earliest end (Helly's theorem in one dimension).
    latest_start_topic = max(ranges, key=lambda t: ranges[t][0])
    earliest_end_topic = min(ranges, key=lambda t: ranges[t][1])
    if ranges[latest_start_topic][0] > ranges[earliest_end_topic][1]:
        return RuleResult(
            RULE_OVERLAP,
            "failed",
            (
                f"{latest_start_topic} starts after {earliest_end_topic} ends; "
                "required streams do not share a time window",
            ),
            tuple(sorted({latest_start_topic, earliest_end_topic})),
        )
    return RuleResult(RULE_OVERLAP, "passed")


def _check_decodable(profile: ValidationProfile, raw_uri: str) -> RuleResult:
    if not profile.decodable_topics:
        return RuleResult(RULE_DECODABLE, "passed", ("profile requires no decodable topics",))
    path = Path(raw_uri)
    if not path.is_file():
        return RuleResult(RULE_DECODABLE, "skipped", (f"raw source not reachable: {raw_uri}",))
    from lancedb_robotics.adapters import AdapterError, get_adapter

    try:
        inspect_report = get_adapter("mcap").inspect(path)
    except AdapterError as exc:
        # A damaged source (codec gap, truncation, CRC) cannot be re-inspected;
        # the byte-integrity rule owns that verdict, so skip rather than crash.
        return RuleResult(RULE_DECODABLE, "skipped", (f"source not inspectable: {exc}",))
    topics = {t["topic"]: t for t in inspect_report["topics"]}
    details: list[str] = []
    failed: list[str] = []
    for topic in profile.decodable_topics:
        info = topics.get(topic)
        if info is None:
            details.append(f"{topic}: not present in source")
            failed.append(topic)
        elif not info["can_decode"]:
            details.append(f"{topic}: message encoding '{info['message_encoding']}' has no decoder")
            failed.append(topic)
    if failed:
        return RuleResult(RULE_DECODABLE, "failed", tuple(details), tuple(failed))
    return RuleResult(RULE_DECODABLE, "passed")


def _run_integrity_status(run: dict) -> str:
    """Read the ingest-stamped ``integrity.status`` from a run's metadata.

    Absent (runs written before backlog 0017) is treated as ``complete``.
    """
    for entry in run.get("metadata") or []:
        if entry["key"] == "integrity.status":
            return entry["value"]
    return "complete"


def _check_integrity(run: dict) -> RuleResult:
    """Re-assert the ingest-time byte-integrity verdict (backlog 0017).

    Ingest already quarantines a CRC-damaged or truncated read, but the quality
    gate overwrites ``quality_flags`` (latest validation wins). Reading the
    durable ``integrity.status`` back from run metadata keeps a damaged run
    quarantined across re-validation instead of silently clearing it.
    """
    status = _run_integrity_status(run)
    if status == "complete":
        return RuleResult(RULE_INTEGRITY, "passed")
    return RuleResult(
        RULE_INTEGRITY,
        "failed",
        (f"source read was not clean at ingest: integrity '{status}'",),
    )


def validate_run(lake: Lake, run_id: str, profile: ValidationProfile) -> RunQualityReport:
    """Evaluate one ingested run against ``profile`` without writing anything.

    Raises :class:`QualityError` if ``run_id`` is not in the lake.
    """
    run = next((r for r in _run_rows(lake) if r["run_id"] == run_id), None)
    if run is None:
        raise QualityError(f"no such run in {lake.uri}: {run_id}")
    by_topic = _observations_by_topic(lake, run_id)

    rules = [_check_required_topics(profile, by_topic)]
    if profile.require_monotonic:
        rules.append(_check_monotonic(by_topic))
    if profile.require_overlap:
        rules.append(_check_overlap(profile, by_topic))
    rules.append(_check_decodable(profile, run["raw_uri"]))
    rules.append(_check_integrity(run))
    return RunQualityReport(run_id=run_id, profile=profile.name, rules=tuple(rules))


def validate_lake(
    lake: Lake, profile: ValidationProfile, *, run_id: str | None = None
) -> list[RunQualityReport]:
    """Validate every run in the lake (or one run), sorted by run id."""
    if run_id is not None:
        return [validate_run(lake, run_id, profile)]
    return [validate_run(lake, run["run_id"], profile) for run in _run_rows(lake)]


def _observation_topic_failures(report: RunQualityReport) -> dict[str, list[str]]:
    """Map each failing topic to its ``quality:failed:<rule>`` flags for one run."""
    topic_failures: dict[str, list[str]] = {}
    for rule in report.failed_rules:
        for topic in rule.failed_topics:
            topic_failures.setdefault(topic, []).append(f"quality:failed:{rule.rule}")
    return topic_failures


def _apply_observation_flags(lake: Lake, reports: list[RunQualityReport]) -> None:
    """Recompute ``observations.quality_flags`` additively (decision 0024).

    ``observations`` carries a blob-encoded ``payload_blob``, so in-place
    ``Table.update`` is unavailable. Instead the whole ``quality_flags`` column is
    recomputed -- failing topics of validated runs get their rule flags, passing
    observations go to NULL, and observations of runs *not* in this validation
    keep their existing flags -- and written via ``drop_columns`` + ``add_columns``
    on the Lance dataset. That commits a new, versioned table state without ever
    reading or rewriting the blob bytes (latest validation wins).
    """
    if not reports:
        return
    failures_by_run = {r.run_id: _observation_topic_failures(r) for r in reports}

    dataset = lake.table("observations").to_lance()
    current = dataset.to_table(columns=["run_id", "topic", "quality_flags"])
    if current.num_rows == 0:
        return

    run_ids = current["run_id"].to_pylist()
    topics = current["topic"].to_pylist()
    existing = current["quality_flags"].to_pylist()
    recomputed: list[list[str] | None] = []
    for run_id, topic, existing_flags in zip(run_ids, topics, existing, strict=True):
        run_failures = failures_by_run.get(run_id)
        if run_failures is None:
            recomputed.append(existing_flags)  # run not validated: keep as-is
        else:
            recomputed.append(run_failures.get(topic) or None)  # latest wins

    new_column = pa.table({"quality_flags": pa.array(recomputed, type=pa.list_(pa.string()))})
    # Order is preserved: drop_columns is a metadata-only edit that does not move
    # rows, so add_columns aligns the recomputed column to the same scan order.
    dataset.drop_columns(["quality_flags"])
    dataset.add_columns(new_column)


def apply_quality_results(
    lake: Lake,
    reports: list[RunQualityReport],
    profile: ValidationProfile,
    *,
    quarantine: bool = True,
    created_by: str = "lancedb-robotics",
) -> None:
    """Write validation verdicts back to the lake; latest validation wins.

    Overwrites ``quality_flags`` on each validated run (in-place ``Table.update``;
    ``runs`` has no blob column), recomputes ``observations.quality_flags``
    additively for the validated runs (``drop_columns`` + ``add_columns``, since
    ``observations`` is blob-encoded -- see :func:`_apply_observation_flags`), and
    replaces the (run, profile) quality lineage row in ``transform_runs``.
    """
    now = datetime.now(UTC)
    runs_table = lake.table("runs")
    transforms_table = lake.table("transform_runs")
    transform_rows = []

    for report in reports:
        runs_table.update(
            where=f"run_id = '{report.run_id}'",
            values={"quality_flags": report.run_flags(quarantine=quarantine)},
        )

        transform_id = f"tfm-quality-{profile.name}-{report.run_id.removeprefix('run-')}"
        transforms_table.delete(f"transform_id = '{transform_id}'")
        transform_rows.append(
            {
                "transform_id": transform_id,
                "kind": "quality",
                "input_uris": [],
                "input_table_versions": [],
                "output_tables": ["runs", "observations"],
                "params": json.dumps(
                    {
                        "profile": profile.name,
                        "run_id": report.run_id,
                        "quarantine": quarantine,
                        "report": report.to_dict(),
                    },
                    sort_keys=True,
                ),
                "status": "completed",
                "started_at": now,
                "finished_at": now,
                "created_by": created_by,
                "created_at": now,
            }
        )

    _apply_observation_flags(lake, reports)

    if transform_rows:
        transforms_table.add(pa.Table.from_pylist(transform_rows, schema=TRANSFORM_RUNS_SCHEMA))


def quarantined_run_ids(lake: Lake) -> list[str]:
    """Run ids currently flagged as quarantined, sorted."""
    return sorted(
        run["run_id"]
        for run in lake.table("runs").to_arrow().to_pylist()
        if QUARANTINED_FLAG in (run["quality_flags"] or [])
    )
