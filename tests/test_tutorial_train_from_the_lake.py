"""Execute the "from a raw log to a training batch" tutorial (backlog 0519).

``tests/test_manual_docs_verification.py`` proves the tutorial's commands
*exist*. It cannot prove the sequence *works*, or that the numbers the prose
quotes are the numbers the code produces — a reordered step or a stale count
would ship green. This test closes that gap the way
``test_integration_showcase.py`` already does for the baseline vertical slice:
run the published command sequence start to finish on the bundled fixture,
assert the figures the doc states, and fail if doc and behaviour drift apart.

Three layers, by environment cost:

* **Always** — the five CLI commands that build the lake, and the doc-sync check
  that the tutorial's and README's command blocks still match ``PIPELINE``.
* **``require_torch_loader``** — Path A's dataset, deterministic shuffle, global
  ``resume_from``, and a real ``DataLoader``; plus Path B's publish and the
  ``delta_timestamps`` action chunk, which assemble ``torch`` tensors.
* **Gated lane only** — opening the published view through upstream lerobot's
  own ``LeRobotDataset``. That needs the dev-only, commit-pinned
  ``lerobot-main-dev`` extra, so it lives behind ``require_lerobot_main_dev``
  and the default suite never depends on a pinned fork.

The fixture's run id is asserted as a literal on purpose: ingest is
content-addressed on the log's *bytes*, so ``run-be4244bb87aeb236`` is stable
across machines and checkout paths. The tutorial hands readers that literal to
paste into ``align create --run-id``; if regenerating the fixture ever changed
it, the doc would silently stop working and this assertion is what catches it.
The published ``view_id`` is deliberately *not* pinned — it digests the pinned
versions of every canonical table, so adding an unrelated table would change it,
which is drift in the digest, not in the tutorial.
"""

import json
import re
from pathlib import Path

import pytest
from conftest import require_lerobot_main_dev, require_torch_loader
from typer.testing import CliRunner

from lancedb_robotics.cli import app
from lancedb_robotics.lake import Lake

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = REPO_ROOT / "docs" / "manual" / "tutorials" / "train-from-the-lake.md"
README = REPO_ROOT / "README.md"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "teleop.mcap"
PROFILE = REPO_ROOT / "tests" / "fixtures" / "teleop-profile.json"
INTERVALS = REPO_ROOT / "tests" / "fixtures" / "teleop-episodes.jsonl"

# Content-addressed on the fixture's bytes; see the module docstring.
RUN_ID = "run-be4244bb87aeb236"
ALIGNMENT = "teleop_20hz"
REPO_ID = "demo/teleop-v1"
STREAMS = ["/joint-positions", "/gripper", "/action"]
FPS = 20

# What the tutorial claims the fixture holds.
OBSERVATIONS = 240
TICKS = 80
EPISODES = 2
FRAMES_PER_EPISODE = 40
STATE_WIDTH = 7  # /joint-positions (6) + /gripper (1)
ACTION_WIDTH = 6

# Placeholders substituted with real paths at run time. This list is the single
# source of truth for the runnable sequence; the doc-sync tests below check the
# tutorial's and README's command blocks against it.
_LAKE = "<lake>"

PIPELINE: list[tuple[str, list[str]]] = [
    ("lake init", ["lake", "init", "--lake", _LAKE]),
    ("ingest mcap", ["ingest", "mcap", str(FIXTURE), "--lake", _LAKE]),
    (
        "quality validate",
        ["quality", "validate", "--lake", _LAKE, "--profile", str(PROFILE)],
    ),
    (
        "episodes import-intervals",
        ["episodes", "import-intervals", "--lake", _LAKE, "--file", str(INTERVALS)],
    ),
    (
        "align create",
        [
            "align", "create", ALIGNMENT,
            "--lake", _LAKE,
            "--rate-hz", str(FPS),
            "--run-id", RUN_ID,
            *[token for stream in STREAMS for token in ("--stream", stream)],
        ],
    ),
]


@pytest.fixture(scope="module")
def tutorial_lake(tmp_path_factory) -> Lake:
    """Run the tutorial's five build commands; return the resulting lake."""
    lake_path = tmp_path_factory.mktemp("tutorial") / "robot.lance"
    for label, template in PIPELINE:
        argv = [str(lake_path) if token == _LAKE else token for token in template]
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, f"`{label}` failed:\n{result.output}"
    return Lake.open(str(lake_path))


# --- the lake the tutorial builds -------------------------------------------


def test_ingest_writes_the_stated_rows_under_a_stable_run_id(tutorial_lake: Lake) -> None:
    runs = tutorial_lake.table("runs").to_arrow().to_pylist()
    assert [row["run_id"] for row in runs] == [RUN_ID]

    observations = tutorial_lake.table("observations").to_arrow().to_pylist()
    assert len(observations) == OBSERVATIONS

    # Typed extraction, per the tutorial's step 1: both RobotState topics land a
    # 6-float vector and the GripperState topic a 1-float one -- and the
    # *commanded* stream populates state_vector too, which is why observed and
    # commanded are separated later by stream name rather than by column.
    widths = {
        row["topic"]: (row["modality"], len(row["state_vector"]))
        for row in observations
        if row["state_vector"] is not None
    }
    assert widths == {
        "/joint-positions": ("joint_state", 6),
        "/gripper": ("gripper", 1),
        "/action": ("joint_state", 6),
    }
    assert {row["decode_status"] for row in observations} == {"decoded"}


def test_quality_profile_passes_the_run(tutorial_lake: Lake) -> None:
    run = tutorial_lake.table("runs").to_arrow().to_pylist()[0]
    assert not run.get("quarantined"), "the tutorial's profile must not quarantine its own run"


def test_imported_intervals_become_two_disjoint_episodes(tutorial_lake: Lake) -> None:
    episodes = sorted(
        tutorial_lake.table("episodes").to_arrow().to_pylist(),
        key=lambda row: row["from_timestamp_ns"],
    )
    assert len(episodes) == EPISODES
    assert [row["episode_index"] for row in episodes] == [0, 1]
    assert {row["task_id"] for row in episodes} == {"pick the cube"}
    # Disjoint, so no tick can be claimed by two episodes -- episodes that fall
    # back to touching scenario windows can double-count the boundary tick.
    assert episodes[0]["to_timestamp_ns"] < episodes[1]["from_timestamp_ns"]


def test_alignment_materializes_one_complete_tick_per_instant(tutorial_lake: Lake) -> None:
    ticks = tutorial_lake.table("aligned_ticks").to_arrow().to_pylist()
    assert len(ticks) == TICKS
    assert all(not tick["has_missing"] for tick in ticks)
    assert all(not tick["has_out_of_tolerance"] for tick in ticks)

    values = json.loads(ticks[0]["stream_values_json"])
    assert sorted(values) == sorted(STREAMS)
    # Ticks are 50 ms apart and values move every tick -- a constant-valued
    # fixture would make the windowing assertions below vacuous.
    ordered = sorted(ticks, key=lambda row: row["tick_index"])
    first = json.loads(ordered[0]["stream_values_json"])["/action"]
    second = json.loads(ordered[1]["stream_values_json"])["/action"]
    assert first != second


# --- Path A: the native torch loader ----------------------------------------


def _aligned(lake: Lake, **kwargs):
    return lake.training.aligned_dataset(
        ALIGNMENT, streams=STREAMS, require_streams=True, **kwargs
    )


def test_path_a_dataset_yields_every_tick_with_all_streams(tutorial_lake: Lake) -> None:
    dataset = _aligned(tutorial_lake, shuffle=True, shuffle_seed=17)
    assert len(dataset) == TICKS

    sample = dataset[0]
    assert sorted(sample["streams"]) == sorted(STREAMS)
    assert len(sample["streams"]["/joint-positions"]["value"]) == 6
    assert len(sample["streams"]["/gripper"]["value"]) == 1
    assert len(sample["streams"]["/action"]["value"]) == 6


def test_path_a_shuffle_is_seeded_and_resume_is_global(tutorial_lake: Lake) -> None:
    def order(**kwargs):
        return [s["tick_index"] for s in _aligned(tutorial_lake, shuffle=True, **kwargs)]

    baseline = order(shuffle_seed=17)
    assert sorted(baseline) == list(range(TICKS))
    assert order(shuffle_seed=17) == baseline, "same seed must reproduce the epoch"
    assert order(shuffle_seed=99) != baseline, "a different seed must reorder it"
    assert baseline != sorted(baseline), "shuffle must not be the identity"
    assert order(shuffle_seed=17, resume_from=40) == baseline[40:], (
        "resume_from is global: it drops the first N of the same order"
    )


def test_path_a_spawn_workers_cover_the_epoch_exactly_once(tutorial_lake: Lake) -> None:
    require_torch_loader()

    # Spawn, not fork: Lance readers are not fork-safe, which is why the
    # tutorial passes multiprocessing_context explicitly rather than relying on
    # the platform default.
    loader = _aligned(tutorial_lake, shuffle=True, shuffle_seed=17).torch_dataloader(
        batch_size=8,
        num_workers=2,
        adapter="iterable",
        multiprocessing_context="spawn",
    )

    seen: list[int] = []
    workers: set[int] = set()
    for batch in loader:
        seen.extend(int(tick) for tick in batch["tick_index"])
        workers.add(batch["_lineage"]["worker"]["id"])

    assert sorted(seen) == list(range(TICKS))
    assert workers == {0, 1}


# --- Path B: the published LeRobot view -------------------------------------


@pytest.fixture(scope="module")
def published(tutorial_lake: Lake):
    require_torch_loader()
    from lancedb_robotics.lerobot_facade import CanonicalVectorMapping, publish_view

    mapping = CanonicalVectorMapping(
        state_streams=("/joint-positions", "/gripper"),
        action_streams=("/action",),
    )
    view = publish_view(
        tutorial_lake,
        repo_id=REPO_ID,
        fps=FPS,
        name=ALIGNMENT,
        mapping=mapping,
        robot_type="so100",
    )
    return view, mapping


def test_path_b_publish_reports_the_stated_totals(tutorial_lake: Lake, published) -> None:
    view, _mapping = published
    assert view.repo_id == REPO_ID
    assert view.view_id.startswith("lrv-")
    assert view.total_frames == TICKS
    assert view.total_episodes == EPISODES


def test_path_b_publish_is_idempotent_over_an_unchanged_lake(
    tutorial_lake: Lake, published
) -> None:
    from lancedb_robotics.lerobot_facade import publish_view

    view, mapping = published
    again = publish_view(
        tutorial_lake,
        repo_id=REPO_ID,
        fps=FPS,
        name=ALIGNMENT,
        mapping=mapping,
        robot_type="so100",
    )
    assert again.view_id == view.view_id


def test_path_b_composes_the_declared_feature_widths(tutorial_lake: Lake, published) -> None:
    from lancedb_robotics.lerobot_facade import LiveLeRobotFacade

    _view, mapping = published
    facade = LiveLeRobotFacade(tutorial_lake, name=ALIGNMENT, mapping=mapping)

    assert len(facade) == TICKS
    item = facade[0]
    # --state-stream is ordered and concatenated: 6 joints + 1 gripper.
    assert len(item["observation.state"]) == STATE_WIDTH
    assert len(item["action"]) == ACTION_WIDTH
    assert (item["episode_index"], item["frame_index"]) == (0, 0)

    last = facade[len(facade) - 1]
    assert (last["episode_index"], last["frame_index"]) == (
        EPISODES - 1,
        FRAMES_PER_EPISODE - 1,
    )


def test_path_b_action_chunk_clamps_at_the_episode_boundary(
    tutorial_lake: Lake, published
) -> None:
    """The tutorial's central claim about windows: a chunk never crosses a cut."""
    require_torch_loader()
    import torch

    from lancedb_robotics.lerobot_facade import LiveLeRobotFacade
    from lancedb_robotics.lerobot_facade import _reader_core as core

    _view, mapping = published
    facade = LiveLeRobotFacade(tutorial_lake, name=ALIGNMENT, mapping=mapping)
    reference = [facade[i] for i in range(len(facade))]

    episode_from = [0, FRAMES_PER_EPISODE]
    episode_to = [FRAMES_PER_EPISODE, TICKS]
    chunk = list(range(8))  # 8 steps, matching delta_timestamps {i/20 for i in range(8)}

    # Frame 0 sits mid-episode: the whole chunk is real.
    # Frame 36 is four frames from episode 0's end: the tail clamps and pads.
    plans = core.plan_windows([0, 36], {"action": chunk}, episode_from, episode_to)
    items = core.hydrate_windowed_items(
        facade, plans, abs_to_facade=None, return_uint8=False, image_transforms=None
    )

    assert items[0]["action"].shape == (len(chunk), ACTION_WIDTH)
    assert items[0]["action_is_pad"].tolist() == [False] * len(chunk)
    torch.testing.assert_close(
        items[0]["action"],
        torch.tensor([reference[i]["action"] for i in chunk], dtype=torch.float32),
    )

    # Frames 36..39 are real; 40..43 would be episode 1, so they clamp to 39.
    assert items[1]["action_is_pad"].tolist() == [False] * 4 + [True] * 4
    expected_tail = [reference[i]["action"] for i in (36, 37, 38, 39)] + [
        reference[39]["action"]
    ] * 4
    torch.testing.assert_close(
        items[1]["action"], torch.tensor(expected_tail, dtype=torch.float32)
    )
    # The clamped members are episode 0's last frame, never episode 1's first.
    assert reference[40]["action"] != reference[39]["action"]


def test_path_b_opens_through_upstream_lerobot(tutorial_lake: Lake, published) -> None:
    """The gated leg: the same view through lerobot's own LeRobotDataset."""
    require_lerobot_main_dev()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import lancedb_robotics.lerobot_facade.dataset_reader  # noqa: F401  (registers)

    dataset = LeRobotDataset(REPO_ID, root=f"file://{tutorial_lake.uri}")
    assert len(dataset) == TICKS
    assert len(dataset.meta.stats["observation.state"]["mean"]) == STATE_WIDTH

    item = dataset[0]
    assert tuple(item["observation.state"].shape) == (STATE_WIDTH,)
    assert tuple(item["action"].shape) == (ACTION_WIDTH,)

    windowed = LeRobotDataset(
        REPO_ID,
        root=f"file://{tutorial_lake.uri}",
        delta_timestamps={"action": [i / FPS for i in range(8)]},
    )
    chunked = windowed[0]
    assert tuple(chunked["action"].shape) == (8, ACTION_WIDTH)
    assert chunked["action_is_pad"].tolist() == [False] * 8


# --- doc sync: the prose must run what this test runs -----------------------


def _doc_commands(path: Path) -> list[str]:
    """Every ``lancedb-robotics`` line from the doc's first ```bash block."""
    block = re.search(r"```bash\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)
    assert block, f"{path.name} is missing a ```bash command block"
    return [
        line.strip()
        for line in block.group(1).splitlines()
        if line.strip().startswith("lancedb-robotics ")
    ]


@pytest.mark.parametrize("doc", [TUTORIAL, README], ids=lambda p: p.name)
def test_doc_build_sequence_matches_the_runnable_pipeline(doc: Path) -> None:
    # The README's first bash block is its install snippet; the tutorial's is
    # the build sequence. Find the block that starts with `lake init`.
    text = doc.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    build = next(
        (b for b in blocks if b.strip().startswith("lancedb-robotics lake init")), None
    )
    assert build, f"{doc.name} has no `lancedb-robotics lake init ...` command block"

    commands = [
        line.strip() for line in build.splitlines() if line.strip().startswith("lancedb-robotics ")
    ]
    assert len(commands) == len(PIPELINE), (
        f"{doc.name} shows {len(commands)} build commands; the pipeline runs {len(PIPELINE)}"
    )
    for command, (subcommand, _template) in zip(commands, PIPELINE, strict=True):
        assert command.startswith(f"lancedb-robotics {subcommand}"), (
            f"{doc.name} step {command!r} does not match pipeline step {subcommand!r}"
        )


@pytest.mark.parametrize("doc", [TUTORIAL, README], ids=lambda p: p.name)
def test_docs_quote_the_values_this_test_asserts(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    # The literals a reader copies or trusts. If the fixture or the pipeline
    # changes these, the prose must change with it.
    for literal in (RUN_ID, ALIGNMENT, REPO_ID, str(OBSERVATIONS), str(TICKS)):
        assert literal in text, f"{doc.name} no longer mentions {literal!r}"


def test_tutorial_states_the_lerobot_extra_caveat() -> None:
    """The live-reader caveat is load-bearing honesty, not decoration."""
    text = TUTORIAL.read_text(encoding="utf-8")
    assert "lerobot-main-dev" in text
    assert "lerobot.dataset_readers" in text
    assert "lancedb_robotics.lerobot_facade.dataset_reader" in text
