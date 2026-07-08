"""CLI entrypoint for lancedb-robotics.

Top-level command groups form the demo spine of the baseline ingest showcase.
Each group starts empty; later feature sets register real subcommands inside
their group module.
"""

import typer

from lancedb_robotics import __version__
from lancedb_robotics.cli.align import align_app
from lancedb_robotics.cli.bench import bench_app
from lancedb_robotics.cli.curate import curate_app
from lancedb_robotics.cli.dataset import dataset_app
from lancedb_robotics.cli.distributions import distributions_app
from lancedb_robotics.cli.embed import embed_app
from lancedb_robotics.cli.episodes import episodes_app
from lancedb_robotics.cli.export import export_app
from lancedb_robotics.cli.ingest import ingest_app
from lancedb_robotics.cli.inspect import inspect_app
from lancedb_robotics.cli.lake import lake_app
from lancedb_robotics.cli.lineage import lineage_app
from lancedb_robotics.cli.quality import quality_app
from lancedb_robotics.cli.scenarios import scenarios_app
from lancedb_robotics.cli.search import search_app
from lancedb_robotics.cli.train import train_app
from lancedb_robotics.cli.video import video_app
from lancedb_robotics.cli.writeback import writeback_app

# Ordered contract of top-level command groups. Tests pin this list; changing
# it is a product decision, not a refactor.
COMMAND_GROUPS: dict[str, str] = {
    "lake": "Create and manage a LanceDB robotics lake.",
    "inspect": "Inspect robot log files (MCAP first) without ingesting.",
    "ingest": "Register sources and ingest robot logs into canonical lake rows.",
    "quality": "Validate required streams and manage quarantine results.",
    "align": "Create multi-rate temporally aligned observation views.",
    "scenarios": "Create searchable scenario/clip windows over ingested runs.",
    "episodes": "Derive and inspect first-class episode boundaries and frames.",
    "video": "Encode and inspect codec-aware camera video frames.",
    "embed": "Create observation-level embedding columns (image/pixel vectors).",
    "search": "Search scenarios and camera frames with scalar, text, vector, hybrid, and image queries.",
    "curate": "Curate, sample, mine, and snapshot scenario selections.",
    "gaps": "Analyze distribution gaps and balance reports.",
    "dataset": "Create reproducible dataset snapshots from search results.",
    "train": "Preview dataset snapshots as training datasets.",
    "export": "Export selected clips back to MCAP and replay-tool workflows.",
    "lineage": "Refresh and traverse canonical lineage graph provenance.",
    "bench": "Run reproducible benchmark harnesses for training data access.",
    "writeback": "Import labels, model outputs, and feedback into the closed loop.",
}

app = typer.Typer(
    name="lancedb-robotics",
    help="Multimodal data lakehouse substrate for Physical AI pipelines, built on LanceDB.",
    no_args_is_help=True,
    rich_markup_mode=None,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the package version and exit.",
    ),
) -> None:
    """Multimodal data lakehouse substrate for Physical AI pipelines, built on LanceDB."""


# Groups with real subcommands; the rest stay as empty placeholders until
# their feature set lands.
IMPLEMENTED_GROUPS: dict[str, typer.Typer] = {
    "lake": lake_app,
    "inspect": inspect_app,
    "ingest": ingest_app,
    "quality": quality_app,
    "align": align_app,
    "scenarios": scenarios_app,
    "episodes": episodes_app,
    "video": video_app,
    "embed": embed_app,
    "search": search_app,
    "curate": curate_app,
    "gaps": distributions_app,
    "dataset": dataset_app,
    "train": train_app,
    "export": export_app,
    "lineage": lineage_app,
    "bench": bench_app,
    "writeback": writeback_app,
}

for _name, _help in COMMAND_GROUPS.items():
    _sub = IMPLEMENTED_GROUPS.get(_name)
    if _sub is None:
        _sub = typer.Typer(no_args_is_help=True, rich_markup_mode=None)
    app.add_typer(_sub, name=_name, help=_help)
