"""Full baseline showcase as one integration scenario (backlog 0012).

Runs the entire demo spine — init → inspect → ingest → quality → window →
enrich → search → snapshot → train-preview → export — on the deterministic
sample fixture, asserts the key command summaries (golden), proves end-to-end
lineage, and checks that the narrative doc's command block stays in sync with
the commands this test runs.

Only path/machine-independent facts are asserted — and since backlog 0013 made
run ids content-addressed (keyed on file bytes, not the absolute path), that
now includes the *named* split. The single-run fixture lands all four windows
in ``train`` on every machine, so the bucket is pinned rather than just its
shape.
"""

import json
import re
from pathlib import Path

from conftest import assert_matches_snapshot
from mcap.reader import make_reader
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.lake import Lake

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[1]
NARRATIVE = REPO_ROOT / "docs" / "narratives" / "baseline-ingest-to-training-showcase.md"

# Placeholders substituted with real paths at run time. This list is the single
# source of truth for the runnable sequence; the narrative-sync test checks the
# doc's command block against it.
_LAKE = "<lake>"
_FIXTURE = "<fixture>"
_CLIPS = "<clips>"

PIPELINE: list[tuple[str, list[str]]] = [
    ("lake init", ["lake", "init", "--lake", _LAKE]),
    ("inspect mcap", ["inspect", "mcap", _FIXTURE, "--format", "text"]),
    ("ingest mcap", ["ingest", "mcap", _FIXTURE, "--lake", _LAKE]),
    ("quality validate", ["quality", "validate", "--lake", _LAKE, "--profile", "demo"]),
    ("scenarios create", ["scenarios", "create", "--lake", _LAKE, "--window", "50ms"]),
    ("scenarios enrich", ["scenarios", "enrich", "--lake", _LAKE]),
    ("search hybrid", ["search", "hybrid", "imu observations", "--lake", _LAKE]),
    (
        "dataset snapshot create",
        [
            "dataset",
            "snapshot",
            "create",
            "--lake",
            _LAKE,
            "--from-search",
            "last",
            "--name",
            "demo-v1",
        ],
    ),
    (
        "train preview torch",
        ["train", "preview", "torch", "--lake", _LAKE, "--snapshot", "demo-v1"],
    ),
    ("export mcap", ["export", "mcap", "--lake", _LAKE, "--snapshot", "demo-v1", "--out", _CLIPS]),
]

# Deterministic, machine-independent summary lines each step must print. These
# are the golden assertions; they exclude ids, absolute paths, and float scores
# (which vary by environment). The named split no longer varies and is pinned by
# a separate assertion below.
EXPECTED: dict[str, list[str]] = {
    "lake init": [
        "  integration_sources (v1)",
        "  runs (v1)",
        "  episodes (v1)",
        "  observations (v4)",
        "  videos (v1)",
        "  attachments (v2)",
        "  events (v1)",
        "  scenarios (v3)",
        "  dataset_snapshots (v1)",
        "  labels (v1)",
        "  model_outputs (v1)",
        "  feedback (v1)",
        "  transform_runs (v1)",
    ],
    "inspect mcap": [
        "  5 messages, 2 topics, 0.2s (1700000000000000000 .. 1700000000200000000)",
        "  /camera/front\tcbor\tsample.CompressedImage\t2 msgs\tdecodable",
        "  /imu\tjson\tsample.Imu\t3 msgs\tdecodable",
    ],
    "ingest mcap": [
        "rows added:",
        "  integration_sources +1",
        "  runs +1",
        "  observations +5",
        # sample.mcap carries no attachments (backlog 0016).
        "  attachments +0",
        "  events +2",
        "  transform_runs +2",
        "observations by topic:",
        "  /camera/front\t2",
        "  /imu\t3",
    ],
    "quality validate": [
        "profile: demo",
        "runs: 1 validated, 1 passed, 0 failed, 0 quarantined",
    ],
    "scenarios create": [
        "window: 50ms (50000000 ns)",
        "topics: all topics",
        "partial final window: included",
        "runs: 1",
        "scenarios: 4 created",
    ],
    "scenarios enrich": [
        "caption provider: demo-template-v1",
        "embedding provider: demo-hash-v1 (dim 16)",
        "scenarios enriched: 4",
        # transform_id derives from provider config only, so it is stable.
        "transform: tfm-enrich-b531e97abc4798d2",
        "fts index: built (FTS over scenarios.summary, 4 rows)",
    ],
    "search hybrid": [
        "mode: hybrid",
        'query: "imu observations"',
        "results: 4",
    ],
    "dataset snapshot create": [
        "tag: demo-v1",
        'source: search (hybrid "imu observations")',
        "scenarios: 4",
    ],
    "train preview torch": [
        "tag: demo-v1",
        "split by: run",
        "scenarios: 4",
        "columns: scenario_id, split, summary, topics, embedding",
    ],
    "export mcap": [
        "format: mcap",
        "clips: 4 (exported 4, skipped 0, planned 0)",
    ],
}

EXPECTED_TRANSFORM_KINDS = {
    "inspect",
    "ingest",
    "quality",
    "scenario-windowing",
    "enrichment",
    "search",
    "dataset-snapshot",
    "export",
}

EXPECTED_ROW_COUNTS = {
    "integration_sources": 1,
    "runs": 1,
    "observations": 5,
    "events": 2,
    "scenarios": 4,
    "dataset_snapshots": 1,
}


def _argv(template: list[str], *, lake: str, fixture: str, clips: str) -> list[str]:
    mapping = {_LAKE: lake, _FIXTURE: fixture, _CLIPS: clips}
    return [mapping.get(token, token) for token in template]


def test_full_showcase_pipeline(tmp_path, fixtures_dir):
    lake_path = str(tmp_path / "demo.robot.lance")
    fixture = str(fixtures_dir / "sample.mcap")
    clips = str(tmp_path / "demo-clips")

    transcript: list[str] = []
    outputs: dict[str, str] = {}
    for name, template in PIPELINE:
        result = runner.invoke(app, _argv(template, lake=lake_path, fixture=fixture, clips=clips))
        assert result.exit_code == 0, f"step {name!r} failed:\n{result.output}"
        outputs[name] = result.output

        transcript.append(f"## {name}")
        for line in EXPECTED[name]:
            assert line in result.output, f"step {name!r} missing summary line: {line!r}"
            transcript.append(line)
        if name == "train preview torch":
            torch_lines = (
                "framework: torch ready (tensor batches available)",
                "framework: torch not installed (showing dict preview); "
                "install lancedb-robotics[torch] for tensor batches",
            )
            assert any(line in result.output for line in torch_lines), (
                f"step {name!r} missing torch framework status line"
            )
            transcript.append("framework: torch optional path exercised")
        transcript.append("")

    # Golden summary across the whole pipeline (machine-independent lines only).
    assert_matches_snapshot("showcase_summary.txt", "\n".join(transcript).rstrip() + "\n")

    # Split is run-keyed and run ids are content-addressed (backlog 0013), so
    # the named bucket is reproducible: all four windows land in train.
    match = re.search(
        r"split by run: train=(\d+) val=(\d+) test=(\d+)", outputs["dataset snapshot create"]
    )
    assert match, outputs["dataset snapshot create"]
    assert tuple(int(n) for n in match.groups()) == (4, 0, 0)

    # End-to-end lineage: one transform_runs row per pipeline stage.
    lake = Lake.open(lake_path)
    kinds = {row["kind"] for row in lake.table("transform_runs").to_arrow().to_pylist()}
    assert EXPECTED_TRANSFORM_KINDS <= kinds, (
        f"missing transform kinds: {EXPECTED_TRANSFORM_KINDS - kinds}"
    )

    for table, expected in EXPECTED_ROW_COUNTS.items():
        assert lake.table(table).count_rows() == expected, f"{table} row count"

    # Exported clips are real, valid MCAP plus a manifest.
    clip_files = sorted(Path(clips).glob("*.mcap"))
    assert len(clip_files) == 4
    manifest = json.loads((Path(clips) / "export_manifest.json").read_text())
    assert manifest["exported"] == 4
    with clip_files[0].open("rb") as handle:
        clip_messages = list(make_reader(handle).iter_messages())
    assert clip_messages  # each clip carries the window's raw messages


# --- narrative ⟷ runnable script synchronization ---------------------------


def _doc_commands() -> list[str]:
    text = NARRATIVE.read_text()
    block = re.search(r"```bash\n(.*?)```", text, re.DOTALL)
    assert block, "narrative is missing a ```bash command block"
    return [
        line.strip()
        for line in block.group(1).splitlines()
        if line.strip().startswith("lancedb-robotics ")
    ]


def test_narrative_commands_match_runnable_pipeline():
    doc_commands = _doc_commands()

    # Same number of steps, same ordered subcommand paths as the test runs.
    assert len(doc_commands) == len(PIPELINE)
    for doc_command, (subcommand, _template) in zip(doc_commands, PIPELINE, strict=True):
        assert doc_command.startswith(f"lancedb-robotics {subcommand}"), (
            f"narrative step {doc_command!r} does not match pipeline step {subcommand!r}"
        )


def test_narrative_keeps_the_corrected_flags():
    text = NARRATIVE.read_text()
    # Corrections this task makes to the original showcase script must stick.
    assert "--format text" in text
    assert "--format table" not in text  # the old, unsupported format
    assert "scenarios enrich" in text  # required before search
    assert "--window 50ms" in text
    assert "--from-search last" in text
