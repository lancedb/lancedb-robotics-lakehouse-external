"""`lancedb-robotics bench` subcommands."""

from pathlib import Path

import typer

bench_app = typer.Typer(no_args_is_help=True, rich_markup_mode=None)

_LAKE_OPTION = typer.Option(..., "--lake", help="Path or object-store URI to the lake.")
_SNAPSHOT_OPTION = typer.Option(..., "--snapshot", help="Snapshot name to benchmark.")
_FORMATS_OPTION = typer.Option(
    "enterprise-lance,lance,lerobot-default,webdataset,deeplake",
    "--formats",
    help=(
        "Comma-separated formats: enterprise-lance, lance, lerobot-default, "
        "lerobot-native, webdataset, deeplake."
    ),
)
_OUT_OPTION = typer.Option(None, "--out", help="Write the structured JSON report here.")
_ARTIFACTS_OPTION = typer.Option(
    None,
    "--artifacts",
    help="Directory for materialized comparison artifacts.",
)
_PUBLIC_ARTIFACT_ROOT_OPTION = typer.Option(
    ...,
    "--artifact-root",
    "--artifacts",
    help="Stable directory where public benchmark runs, index, and dashboard are retained.",
)
_PUBLICATION_DESTINATION_OPTION = typer.Option(
    ...,
    "--destination",
    "--publish-to",
    help="Durable publication root, e.g. a static-site directory or object-store URI.",
)
_REPORT_ID_OPTION = typer.Option(
    None,
    "--report-id",
    help="Stable report id for CI reruns; defaults to source/revision/timestamp.",
)
_RETENTION_CLASS_OPTION = typer.Option(
    "public-benchmark-history",
    "--retention-class",
    help="Retention class recorded on published benchmark manifests.",
)
_RETAIN_LATEST_OPTION = typer.Option(
    None,
    "--retain-latest",
    min=1,
    help="Mark only the latest N unclaimed reports as protected; claim-linked reports stay protected.",
)
_CLAIMS_OPTION = typer.Option(
    None,
    "--claims",
    "--claim-manifest",
    help="JSON claim manifest to validate against retained public benchmark reports.",
)
_DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    help="Plan publication and retention metadata without writing destination files.",
)
_CAPACITY_DRY_RUN_OPTION = typer.Option(
    False,
    "--capacity-dry-run",
    help="Write capacity/dry-run artifacts without preparing or running the benchmark.",
)
_CAPACITY_MAX_SOURCE_BYTES_OPTION = typer.Option(
    None,
    "--capacity-max-source-bytes",
    help="Maximum estimated source bytes allowed for public LeRobot benchmark runs.",
)
_CAPACITY_MAX_ARTIFACT_BYTES_OPTION = typer.Option(
    None,
    "--capacity-max-artifact-bytes",
    help="Maximum estimated retained artifact bytes allowed for public LeRobot runs.",
)
_CAPACITY_TIME_BUDGET_SECONDS_OPTION = typer.Option(
    None,
    "--capacity-time-budget-seconds",
    help="Maximum estimated wall-clock seconds allowed for public LeRobot runs.",
)
_CAPACITY_REQUIRE_GPU_OPTION = typer.Option(
    None,
    "--capacity-require-gpu/--no-capacity-require-gpu",
    help="Require detectable GPU capacity before running this public benchmark.",
)
_CAPACITY_REQUIRE_OBJECT_STORE_OPTION = typer.Option(
    None,
    "--capacity-require-object-store/--no-capacity-require-object-store",
    help="Require an object-store publication destination before running.",
)
_CAPACITY_PUBLICATION_DESTINATION_OPTION = typer.Option(
    None,
    "--capacity-publication-destination",
    help="Object-store publication destination checked by the capacity gate.",
)
_SAMPLE_LIMIT_OPTION = typer.Option(
    256,
    "--sample-limit",
    help="Maximum samples used for throughput metrics.",
)
_RANDOM_ACCESS_SAMPLES_OPTION = typer.Option(
    64,
    "--random-access-samples",
    help="Number of seeded random-access probes.",
)
_RANDOM_FRAME_SAMPLES_OPTION = typer.Option(
    16,
    "--random-frame-samples",
    help="Number of seeded clips for the N-frames-per-clip workload.",
)
_FRAMES_PER_CLIP_OPTION = typer.Option(
    4,
    "--frames-per-clip",
    help="Frames sampled per random clip.",
)
_SEED_OPTION = typer.Option(0, "--seed", help="Deterministic random-access seed.")
_QUERY_LIMIT_OPTION = typer.Option(
    128,
    "--query-limit",
    help="Maximum scenarios selected by the benchmark curation query.",
)
_ENTERPRISE_FIXTURE_OPTION = typer.Option(
    None,
    "--enterprise-fixture-uri",
    help="Run enterprise-lance against a fake local db:// fixture URI.",
)
_ENTERPRISE_REGION_OPTION = typer.Option(
    "us-east-1",
    "--enterprise-region",
    help="Region recorded for the Enterprise benchmark endpoint.",
)
_ENTERPRISE_HOST_OVERRIDE_OPTION = typer.Option(
    None,
    "--enterprise-host-override",
    help="HTTP endpoint recorded for the Enterprise benchmark endpoint.",
)
_SOURCE_DATASET_ID_OPTION = typer.Option(
    None,
    "--source-dataset-id",
    help="Dataset id recorded in the benchmark descriptor, e.g. lerobot/droid_100.",
)
_SOURCE_DATASET_REVISION_OPTION = typer.Option(
    None,
    "--source-dataset-revision",
    help="Dataset revision or commit recorded in the benchmark descriptor.",
)
_SOURCE_DATASET_URI_OPTION = typer.Option(
    None,
    "--source-dataset-uri",
    help="Source URI recorded in the benchmark descriptor, e.g. hf://lerobot/droid_100.",
)
_SOURCE_SIZE_TIER_OPTION = typer.Option(
    None,
    "--source-size-tier",
    help="Dataset size tier recorded in the report, e.g. droid-100, mid, full.",
)
_SOURCE_SIZE_BYTES_OPTION = typer.Option(
    None,
    "--source-size-bytes",
    help="Measured source dataset bytes recorded in the report.",
)
_SOURCE_SIZE_NOTE_OPTION = typer.Option(
    None,
    "--source-size-note",
    help="Free-form size note recorded in the report.",
)
_STORAGE_TIER_OPTION = typer.Option(
    "local",
    "--storage-tier",
    help="Storage tier label recorded in the report, e.g. local, object-store, hf-cache.",
)
_LEROBOT_NATIVE_SOURCE_MODE_OPTION = typer.Option(
    "auto",
    "--lerobot-native-source-mode",
    help="Native LeRobot source policy: auto, source, or projection.",
)
_LEROBOT_NATIVE_CACHE_MODE_OPTION = typer.Option(
    "auto",
    "--lerobot-native-cache-mode",
    help="Native LeRobot cache/download policy: auto, cache-only, or download.",
)
_LEROBOT_NATIVE_EPISODE_LIMIT_OPTION = typer.Option(
    None,
    "--lerobot-native-episode-limit",
    help="Limit the official LeRobotDataset native arm to the first N episodes.",
)


@bench_app.command("run")
def run(
    lake: str = _LAKE_OPTION,
    snapshot: str = _SNAPSHOT_OPTION,
    formats: str = _FORMATS_OPTION,
    out: Path | None = _OUT_OPTION,
    artifacts: Path | None = _ARTIFACTS_OPTION,
    sample_limit: int = _SAMPLE_LIMIT_OPTION,
    random_access_samples: int = _RANDOM_ACCESS_SAMPLES_OPTION,
    random_frame_samples: int = _RANDOM_FRAME_SAMPLES_OPTION,
    frames_per_clip: int = _FRAMES_PER_CLIP_OPTION,
    seed: int = _SEED_OPTION,
    query_limit: int = _QUERY_LIMIT_OPTION,
    source_dataset_id: str | None = _SOURCE_DATASET_ID_OPTION,
    source_dataset_revision: str | None = _SOURCE_DATASET_REVISION_OPTION,
    source_dataset_uri: str | None = _SOURCE_DATASET_URI_OPTION,
    source_size_tier: str | None = _SOURCE_SIZE_TIER_OPTION,
    source_size_bytes: int | None = _SOURCE_SIZE_BYTES_OPTION,
    source_size_note: str | None = _SOURCE_SIZE_NOTE_OPTION,
    storage_tier: str = _STORAGE_TIER_OPTION,
    lerobot_native_source_mode: str = _LEROBOT_NATIVE_SOURCE_MODE_OPTION,
    lerobot_native_cache_mode: str = _LEROBOT_NATIVE_CACHE_MODE_OPTION,
    lerobot_native_episode_limit: int | None = _LEROBOT_NATIVE_EPISODE_LIMIT_OPTION,
    enterprise_fixture_uri: str | None = _ENTERPRISE_FIXTURE_OPTION,
    enterprise_region: str = _ENTERPRISE_REGION_OPTION,
    enterprise_host_override: str | None = _ENTERPRISE_HOST_OVERRIDE_OPTION,
) -> None:
    """Run the reproducible benchmark harness and emit a structured report."""
    from lancedb_robotics.benchmark import (
        ENTERPRISE_LANCE_FORMAT,
        BenchmarkError,
        run_benchmark_suite,
        write_benchmark_report,
    )
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        selected_formats = tuple(item.strip() for item in formats.split(",") if item.strip())
        artifact_dir = artifacts
        if artifact_dir is None and out is not None:
            artifact_dir = out.parent / f"{out.stem}-artifacts"
        report = run_benchmark_suite(
            opened,
            snapshot,
            formats=selected_formats,
            output_dir=artifact_dir,
            sample_limit=sample_limit,
            random_access_samples=random_access_samples,
            random_frame_samples=random_frame_samples,
            frames_per_clip=frames_per_clip,
            seed=seed,
            query_limit=query_limit,
            source_dataset_id=source_dataset_id,
            source_dataset_revision=source_dataset_revision,
            source_dataset_uri=source_dataset_uri,
            source_dataset_size_tier=source_size_tier,
            source_dataset_size_bytes=source_size_bytes,
            source_dataset_size_note=source_size_note,
            storage_tier=storage_tier,
            lerobot_native_source_mode=lerobot_native_source_mode,
            lerobot_native_cache_mode=lerobot_native_cache_mode,
            lerobot_native_episode_limit=lerobot_native_episode_limit,
            enterprise_fixture_uri=enterprise_fixture_uri,
            enterprise_region=enterprise_region,
            enterprise_host_override=enterprise_host_override,
        )
        if out is not None:
            write_benchmark_report(report, out)
    except (LakeError, BenchmarkError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    dataset = report["dataset"]
    typer.echo(f"lake: {dataset['lake_uri']}")
    typer.echo(f"snapshot: {dataset['snapshot_name']} ({dataset['dataset_id']})")
    typer.echo(f"scenarios: {dataset['scenario_count']}")
    source = dataset.get("source_corpus") or {}
    typer.echo(
        f"source dataset: {source.get('dataset_id')} "
        f"tier={source.get('size_tier')} storage={source.get('storage_tier')}"
    )
    for fmt, result in report["formats"].items():
        status = result["status"]
        typer.echo(f"format {fmt}: {status}")
        if status == "completed":
            metrics = result["metrics"]
            if fmt == ENTERPRISE_LANCE_FORMAT:
                cold = result["phases"]["cold_cache"]["cache"]
                warm = result["phases"]["warm_second_epoch"]["cache"]
                throughput = metrics["shuffled_epoch_throughput"]["value"]
                latency = metrics["random_access_latency"]["value"]
                filter_ms = metrics["subset_filter_change"]["value"]
                typer.echo(
                    f"  cold cache: hits={cold['hits']} misses={cold['misses']}"
                )
                typer.echo(
                    f"  warm second epoch: hits={warm['hits']} misses={warm['misses']}"
                )
                typer.echo(f"  throughput: {throughput:.3f} samples/s")
                typer.echo(f"  random access: {latency:.3f} ms/sample")
                typer.echo(f"  filter change: {filter_ms:.3f} ms")
            else:
                throughput = metrics["dataloader_throughput"]["value"]
                latency = metrics["random_access_latency"]["value"]
                random_frames = metrics["random_frame_sampling"]["value"]
                storage = metrics["storage_footprint"]["value"]
                typer.echo(f"  throughput: {throughput:.3f} samples/s")
                typer.echo(f"  random access: {latency:.3f} ms/sample")
                typer.echo(f"  random frames: {random_frames:.3f} frames/s")
                typer.echo(f"  storage: {storage} bytes")
        else:
            typer.echo(f"  reason: {result['skip_reason']}")
    if out is not None:
        typer.echo(f"report: {out}")
    if artifact_dir is not None:
        typer.echo(f"artifacts: {artifact_dir}")


@bench_app.command("run-public-lerobot")
def run_public_lerobot(
    lake: str = _LAKE_OPTION,
    artifact_root: Path = _PUBLIC_ARTIFACT_ROOT_OPTION,
    source: str | None = typer.Option(
        None,
        "--source",
        help="LeRobot local path or HF repo id to prepare; defaults to lerobot/droid_100.",
    ),
    revision: str | None = typer.Option(
        None,
        "--revision",
        "--source-dataset-revision",
        help="Dataset revision or commit recorded in retained artifacts.",
    ),
    report_id: str | None = _REPORT_ID_OPTION,
    snapshot: str = typer.Option(
        "lerobot-droid-100-benchmark",
        "--snapshot",
        help="Snapshot name to create and benchmark.",
    ),
    formats: str = _FORMATS_OPTION,
    sample_limit: int = _SAMPLE_LIMIT_OPTION,
    random_access_samples: int = _RANDOM_ACCESS_SAMPLES_OPTION,
    random_frame_samples: int = _RANDOM_FRAME_SAMPLES_OPTION,
    frames_per_clip: int = _FRAMES_PER_CLIP_OPTION,
    seed: int = _SEED_OPTION,
    query_limit: int = _QUERY_LIMIT_OPTION,
    size_tier: str = typer.Option(
        "droid-100",
        "--size-tier",
        help="Dataset size tier label recorded in retained artifacts.",
    ),
    storage_tier: str = _STORAGE_TIER_OPTION,
    source_size_bytes: int | None = _SOURCE_SIZE_BYTES_OPTION,
    source_size_note: str | None = _SOURCE_SIZE_NOTE_OPTION,
    lerobot_native_source_mode: str = _LEROBOT_NATIVE_SOURCE_MODE_OPTION,
    lerobot_native_cache_mode: str = _LEROBOT_NATIVE_CACHE_MODE_OPTION,
    lerobot_native_episode_limit: int | None = _LEROBOT_NATIVE_EPISODE_LIMIT_OPTION,
    enterprise_fixture_uri: str | None = _ENTERPRISE_FIXTURE_OPTION,
    enterprise_region: str = _ENTERPRISE_REGION_OPTION,
    enterprise_host_override: str | None = _ENTERPRISE_HOST_OVERRIDE_OPTION,
    capacity_dry_run: bool = _CAPACITY_DRY_RUN_OPTION,
    capacity_max_source_bytes: int | None = _CAPACITY_MAX_SOURCE_BYTES_OPTION,
    capacity_max_artifact_bytes: int | None = _CAPACITY_MAX_ARTIFACT_BYTES_OPTION,
    capacity_time_budget_seconds: int | None = _CAPACITY_TIME_BUDGET_SECONDS_OPTION,
    capacity_require_gpu: bool | None = _CAPACITY_REQUIRE_GPU_OPTION,
    capacity_require_object_store: bool | None = _CAPACITY_REQUIRE_OBJECT_STORE_OPTION,
    capacity_publication_destination: str | None = _CAPACITY_PUBLICATION_DESTINATION_OPTION,
) -> None:
    """Run and retain the public LeRobot benchmark artifact set."""
    from lancedb_robotics.benchmark import (
        DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
        BenchmarkError,
        run_public_lerobot_benchmark,
    )
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        selected_formats = tuple(item.strip() for item in formats.split(",") if item.strip())
        result = run_public_lerobot_benchmark(
            opened,
            artifact_root=artifact_root,
            source=source or DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
            revision=revision,
            report_id=report_id,
            snapshot_name=snapshot,
            formats=selected_formats,
            sample_limit=sample_limit,
            random_access_samples=random_access_samples,
            random_frame_samples=random_frame_samples,
            frames_per_clip=frames_per_clip,
            seed=seed,
            query_limit=query_limit,
            size_tier=size_tier,
            storage_tier=storage_tier,
            source_dataset_size_bytes=source_size_bytes,
            source_dataset_size_note=source_size_note,
            lerobot_native_source_mode=lerobot_native_source_mode,
            lerobot_native_cache_mode=lerobot_native_cache_mode,
            lerobot_native_episode_limit=lerobot_native_episode_limit,
            enterprise_fixture_uri=enterprise_fixture_uri,
            enterprise_region=enterprise_region,
            enterprise_host_override=enterprise_host_override,
            capacity_dry_run=capacity_dry_run,
            capacity_max_source_bytes=capacity_max_source_bytes,
            capacity_max_artifact_bytes=capacity_max_artifact_bytes,
            capacity_time_budget_seconds=capacity_time_budget_seconds,
            capacity_require_gpu=capacity_require_gpu,
            capacity_require_object_store=capacity_require_object_store,
            capacity_publication_destination=capacity_publication_destination,
        )
    except (LakeError, BenchmarkError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    manifest = result["manifest"]
    typer.echo(f"public benchmark: {result['report_id']}")
    typer.echo(f"status: {result['status']}")
    typer.echo(f"run: {result['run_dir']}")
    if "prepare_report" in result["paths"]:
        typer.echo(f"prepare: {result['paths']['prepare_report']}")
    if "benchmark_report" in result["paths"]:
        typer.echo(f"benchmark: {result['paths']['benchmark_report']}")
    typer.echo(f"capacity: {result['paths']['capacity_report']}")
    capacity = result.get("capacity") or {}
    typer.echo(f"capacity status: {capacity.get('status')}")
    for reason in capacity.get("skip_reasons") or []:
        typer.echo(f"  capacity reason: {reason}")
    for tier in capacity.get("tiers") or []:
        selected = " selected" if tier.get("selected") else ""
        typer.echo(f"capacity tier {tier.get('tier')}{selected}: {tier.get('status')}")
        reasons = tier.get("skip_reasons") or []
        if reasons:
            typer.echo(f"  tier reason: {reasons[0]}")
    typer.echo(f"manifest: {result['paths']['artifact_manifest']}")
    typer.echo(f"dashboard: {result['paths']['dashboard']}")
    for fmt, status in manifest["format_statuses"].items():
        typer.echo(f"format {fmt}: {status}")
        if status == "skipped":
            typer.echo(f"  reason: {manifest['skipped'].get(fmt)}")


@bench_app.command("publish-public-lerobot")
def publish_public_lerobot(
    artifact_root: Path = _PUBLIC_ARTIFACT_ROOT_OPTION,
    destination: str = _PUBLICATION_DESTINATION_OPTION,
    report_id: str | None = _REPORT_ID_OPTION,
    retention_class: str = _RETENTION_CLASS_OPTION,
    retain_latest: int | None = _RETAIN_LATEST_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
) -> None:
    """Publish retained public LeRobot benchmark artifacts."""
    from lancedb_robotics.benchmark import BenchmarkError, publish_public_lerobot_benchmark

    try:
        report = publish_public_lerobot_benchmark(
            artifact_root,
            destination=destination,
            report_id=report_id,
            retention_class=retention_class,
            retain_latest=retain_latest,
            dry_run=dry_run,
        )
    except BenchmarkError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    files = report["files"]
    typer.echo(f"publication: {report['status']}")
    typer.echo(f"destination: {report['destination']}")
    typer.echo(f"backend: {report['backend']}")
    typer.echo(f"reports: {', '.join(report['report_ids'])}")
    typer.echo(
        "files: "
        f"written={len(files['planned_or_written'])} "
        f"updated={len(files['updated'])} "
        f"unchanged={len(files['unchanged'])}"
    )
    typer.echo(f"retention: {retention_class}")
    if retain_latest is not None:
        typer.echo(f"retain latest: {retain_latest}")


@bench_app.command("validate-public-lerobot")
def validate_public_lerobot(
    artifact_root: Path = _PUBLIC_ARTIFACT_ROOT_OPTION,
    claims: Path | None = _CLAIMS_OPTION,
    out: Path | None = _OUT_OPTION,
) -> None:
    """Validate retained public LeRobot benchmark artifacts and claims."""
    from lancedb_robotics.benchmark import (
        validate_public_lerobot_benchmark_claims,
        write_benchmark_report,
    )

    report = validate_public_lerobot_benchmark_claims(
        artifact_root,
        claims_path=claims,
    )
    if out is not None:
        write_benchmark_report(report, out)

    typer.echo(f"validation: {report['status']}")
    typer.echo(f"reports: {report['manifest_count']}")
    typer.echo(f"claims: {report['claim_count']}")
    typer.echo(f"errors: {report['error_count']}")
    for diagnostic in report["diagnostics"]:
        prefix = diagnostic["level"]
        code = diagnostic["code"]
        message = diagnostic["message"]
        typer.echo(f"{prefix} {code}: {message}")
    if out is not None:
        typer.echo(f"report: {out}")
    if report["status"] != "passed":
        raise typer.Exit(code=1)


@bench_app.command("public-dashboard")
def public_dashboard(
    artifact_root: Path = _PUBLIC_ARTIFACT_ROOT_OPTION,
) -> None:
    """Rebuild the retained public LeRobot benchmark history dashboard."""
    from lancedb_robotics.benchmark import write_public_lerobot_benchmark_dashboard

    dashboard = write_public_lerobot_benchmark_dashboard(artifact_root)
    index = dashboard["index"]
    typer.echo(f"artifact root: {artifact_root}")
    typer.echo(f"runs: {index['run_count']}")
    typer.echo(f"latest: {index.get('latest_report_id')}")
    typer.echo(f"index: {dashboard['index_path']}")
    typer.echo(f"dashboard: {dashboard['dashboard_path']}")


@bench_app.command("prepare-lerobot")
def prepare_lerobot(
    lake: str = _LAKE_OPTION,
    source: str | None = typer.Option(
        None,
        "--source",
        help="LeRobot local path or HF repo id to prepare; defaults to lerobot/droid_100.",
    ),
    revision: str | None = typer.Option(
        None,
        "--revision",
        "--source-dataset-revision",
        help="Dataset revision or commit recorded in the prepare descriptor.",
    ),
    snapshot: str = typer.Option(
        "lerobot-droid-100-benchmark",
        "--snapshot",
        help="Snapshot name to create for the prepared benchmark corpus.",
    ),
    size_tier: str = typer.Option(
        "droid-100",
        "--size-tier",
        help="Dataset size tier label recorded in the prepare descriptor.",
    ),
    storage_tier: str = typer.Option(
        "hf-cache",
        "--storage-tier",
        help="Storage tier label recorded in the prepare descriptor.",
    ),
    out: Path | None = _OUT_OPTION,
) -> None:
    """Prepare a public LeRobot dataset for benchmark runs."""
    from lancedb_robotics.benchmark import (
        DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
        BenchmarkError,
        prepare_lerobot_benchmark_dataset,
        write_benchmark_report,
    )
    from lancedb_robotics.lake import Lake, LakeError

    try:
        opened = Lake.open(lake)
        report = prepare_lerobot_benchmark_dataset(
            opened,
            source or DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
            revision=revision,
            snapshot_name=snapshot,
            size_tier=size_tier,
            storage_tier=storage_tier,
        )
        if out is not None:
            write_benchmark_report(report, out)
    except (LakeError, BenchmarkError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"prepared: {report['dataset_id']}")
    typer.echo(f"snapshot: {report['snapshot_name']} ({report['snapshot_dataset_id']})")
    typer.echo(f"run: {report['run_id']}")
    typer.echo(f"scenarios: {report['scenario_count']}")
    typer.echo(f"source: {report['source_uri']}")
    if out is not None:
        typer.echo(f"report: {out}")
