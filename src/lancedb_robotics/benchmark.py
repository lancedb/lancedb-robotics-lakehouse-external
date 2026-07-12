"""Reproducible benchmark harness for Lance-native robotics datasets.

The harness measures the four PRD proof metrics on a version-pinned dataset
snapshot:

- dataloader throughput, with GPU utilization recorded when it is discoverable;
- random-access latency;
- query-to-dataset curation time;
- storage footprint at unchanged source-payload quality.

Comparison formats are optional and represented explicitly in the report. The
dependency-light LeRobot-default and WebDataset projections run through the
existing dataset export path; Deep Lake is skipped with install/adapter notes
until a native comparison adapter lands.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lancedb_robotics import __version__
from lancedb_robotics import training as _training
from lancedb_robotics.connections import LakeCapabilities, LakeConnectionSpec
from lancedb_robotics.dataset import DatasetError, create_snapshot
from lancedb_robotics.dataset_export import (
    DATASET_EXPORT_MANIFEST_FILENAME,
    LEROBOT_FORMAT_VERSION,
    WEBDATASET_FORMAT_VERSION,
    DatasetExportError,
    export_dataset_snapshot,
    native_loader_status,
)
from lancedb_robotics.ingest import ingest_lerobot
from lancedb_robotics.lake import Lake
from lancedb_robotics.storage import source_uri
from lancedb_robotics.training import TrainingError

BENCHMARK_REPORT_SCHEMA_VERSION = "lancedb-robotics-benchmark-v0"
PUBLIC_LEROBOT_BENCHMARK_SCHEMA_VERSION = "lancedb-robotics-public-lerobot-benchmark-v0"
PUBLIC_LEROBOT_PUBLICATION_SCHEMA_VERSION = (
    "lancedb-robotics-public-lerobot-benchmark-publication-v0"
)
PUBLIC_LEROBOT_CAPACITY_SCHEMA_VERSION = (
    "lancedb-robotics-public-lerobot-benchmark-capacity-v0"
)
PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION = (
    "lancedb-robotics-public-lerobot-benchmark-claims-v0"
)
PUBLIC_LEROBOT_CLAIM_VALIDATION_SCHEMA_VERSION = (
    "lancedb-robotics-public-lerobot-benchmark-claim-validation-v0"
)
DEFAULT_LEROBOT_BENCHMARK_DATASET_ID = "lerobot/droid_100"
DEFAULT_LEROBOT_BENCHMARK_SOURCE_URI = "hf://lerobot/droid_100"
DEFAULT_LEROBOT_BENCHMARK_SIZE_TIER = "droid-100"
PUBLIC_LEROBOT_CAPACITY_ENV = {
    "max_source_bytes": "LANCEDB_ROBOTICS_PUBLIC_LEROBOT_MAX_SOURCE_BYTES",
    "max_artifact_bytes": "LANCEDB_ROBOTICS_PUBLIC_LEROBOT_MAX_ARTIFACT_BYTES",
    "time_budget_seconds": "LANCEDB_ROBOTICS_PUBLIC_LEROBOT_TIME_BUDGET_SECONDS",
    "require_gpu": "LANCEDB_ROBOTICS_PUBLIC_LEROBOT_REQUIRE_GPU",
    "require_object_store": "LANCEDB_ROBOTICS_PUBLIC_LEROBOT_REQUIRE_OBJECT_STORE",
    "publication_destination": "LANCEDB_ROBOTICS_PUBLIC_LEROBOT_PUBLICATION_DESTINATION",
}
PUBLIC_LEROBOT_CAPACITY_ESTIMATES = {
    "droid-100": {
        "source_bytes": 50_000_000_000,
        "decoded_sample_count": 100_000,
        "output_artifact_bytes": 2_000_000_000,
        "time_budget_seconds": 3_600,
        "note": "Default public smoke tier for scheduled benchmark evidence.",
    },
    "mid": {
        "source_bytes": 500_000_000_000,
        "decoded_sample_count": 1_000_000,
        "output_artifact_bytes": 25_000_000_000,
        "time_budget_seconds": 14_400,
        "note": "Representative mid-size public LeRobot tier; requires configured budgets.",
    },
    "full": {
        "source_bytes": 5_000_000_000_000,
        "decoded_sample_count": 10_000_000,
        "output_artifact_bytes": 250_000_000_000,
        "time_budget_seconds": 86_400,
        "note": "Full public-corpus tier; requires explicit capacity and cost budgets.",
    },
}
PUBLIC_LEROBOT_BUDGET_REQUIRED_TIERS = frozenset({"mid", "full"})
PUBLIC_LEROBOT_OBJECT_STORE_SCHEMES = frozenset(
    {"s3", "gs", "gcs", "az", "abfs", "abfss"}
)

ENTERPRISE_LANCE_FORMAT = "enterprise-lance"
LANCE_FORMAT = "lance"
LEROBOT_DEFAULT_FORMAT = "lerobot-default"
LEROBOT_NATIVE_FORMAT = "lerobot-native"
LEROBOT_NATIVE_SOURCE_MODES = frozenset({"auto", "source", "projection"})
LEROBOT_NATIVE_CACHE_MODES = frozenset({"auto", "cache-only", "download"})
LEROBOT_NATIVE_BENCH_EXTRA = "lerobot-native-bench"
LEROBOT_NATIVE_BENCH_INSTALL = f"lancedb-robotics[{LEROBOT_NATIVE_BENCH_EXTRA}]"
LEROBOT_NATIVE_BENCH_UV_SYNC = f"uv sync --extra {LEROBOT_NATIVE_BENCH_EXTRA}"
LEROBOT_NATIVE_DECODE_POLICY = (
    "TorchCodec is probed first when available from the host or LeRobot stack; "
    "PyAV is the declared stable decode dependency for the supported "
    "lerobot-native-bench extra and CI lane."
)
WEBDATASET_FORMAT = "webdataset"
DEEPLAKE_FORMAT = "deeplake"

# Backlog 0126: analytics-lakehouse baselines. Parquet is dependency-light
# (pyarrow is already required); Iceberg is optional and gated on pyiceberg plus a
# reachable catalog. Both materialize the same logical observation set the Lance
# and WebDataset arms read, so the comparison is fair and explicit about what is
# metadata/table-scan cost versus Python/PyTorch payload hydration cost.
PARQUET_FORMAT = "parquet"
ICEBERG_FORMAT = "iceberg"
PARQUET_ANALYTICS_LAYOUT_VERSION = "parquet-analytics-table-v0"
ICEBERG_ANALYTICS_LAYOUT_VERSION = "iceberg-analytics-table-v0"
ICEBERG_MODULE = "pyiceberg"
ICEBERG_INSTALL_HINT = "pip install 'pyiceberg[sql-sqlite,pyarrow]'"
# A live catalog is configured either by the standard pyiceberg env config or by
# pointing these repo-specific variables at a catalog URI / warehouse directory.
ICEBERG_CATALOG_URI_ENV = "LANCEDB_ROBOTICS_ICEBERG_CATALOG_URI"
ICEBERG_WAREHOUSE_ENV = "LANCEDB_ROBOTICS_ICEBERG_WAREHOUSE"
ICEBERG_NAMESPACE = "lancedb_robotics_bench"
# Payload placement vocabulary recorded on every materialized analytics baseline.
PAYLOAD_PLACEMENT_INLINE = "inline"
PAYLOAD_PLACEMENT_REFERENCED = "referenced"
PAYLOAD_PLACEMENT_COPIED = "copied"
ENTERPRISE_BENCHMARK_CONNECTION_KINDS = frozenset(
    {"lancedb_remote_db", "rest_namespace_lancedb", "namespace_lancedb"}
)
ENTERPRISE_NAMESPACE_CONNECTION_KINDS = frozenset(
    {"rest_namespace_lancedb", "namespace_lancedb"}
)

# Backlog 0125: report profile/confidence labels for the enterprise-lance format.
# fake-local-db reuses local Lance tables behind a simulated Enterprise connection
# report; live-db/live-namespace talk to an already-opened Enterprise endpoint.
# Only fake-local-db numbers are "sdk-contract-only" -- live profiles are labeled
# production-calibrated so a report can never be mistaken for the other kind.
FAKE_LOCAL_DB_PROFILE = "fake-local-db"
LIVE_DB_PROFILE = "live-db"
LIVE_NAMESPACE_PROFILE = "live-namespace"
ENTERPRISE_BENCHMARK_CONFIDENCE = {
    FAKE_LOCAL_DB_PROFILE: "sdk-contract-only",
    LIVE_DB_PROFILE: "production-calibrated",
    LIVE_NAMESPACE_PROFILE: "production-calibrated",
}
ENTERPRISE_INSTANCE_CLASS_ENV = "LANCEDB_ROBOTICS_ENTERPRISE_INSTANCE_CLASS"

# Required vs optional capabilities preflighted before a live benchmark run.
# Missing the required one blocks the whole format (soft skip); missing an
# optional one degrades only the phases that depend on it.
_ENTERPRISE_REQUIRED_CAPABILITY = "remote_scan"
_ENTERPRISE_OPTIONAL_CAPABILITIES = (
    "remote_take",
    "blob_or_video_remote_hydration",
    "plan_executor_cache_metrics",
    "page_cache_prewarm",
    "page_cache_status",
)

DEFAULT_BENCHMARK_FORMATS = (
    ENTERPRISE_LANCE_FORMAT,
    LANCE_FORMAT,
    LEROBOT_DEFAULT_FORMAT,
    WEBDATASET_FORMAT,
    DEEPLAKE_FORMAT,
)
SUPPORTED_BENCHMARK_FORMATS = DEFAULT_BENCHMARK_FORMATS + (
    LEROBOT_NATIVE_FORMAT,
    PARQUET_FORMAT,
    ICEBERG_FORMAT,
    "lerobot",
    "lerobot-v3",
)
# Analytics baselines are opt-in (not in DEFAULT_BENCHMARK_FORMATS) so existing
# reports and the public LeRobot default stay stable; researchers request them
# explicitly with --formats parquet,iceberg.
ANALYTICS_BENCHMARK_FORMATS = (PARQUET_FORMAT, ICEBERG_FORMAT)

METRIC_DATALOADER_THROUGHPUT = "dataloader_throughput"
METRIC_RANDOM_ACCESS_LATENCY = "random_access_latency"
METRIC_RANDOM_FRAME_SAMPLING = "random_frame_sampling"
METRIC_QUERY_TO_DATASET_CURATION = "query_to_dataset_curation"
METRIC_STORAGE_FOOTPRINT = "storage_footprint"
# Analytics-baseline-specific metrics (backlog 0126). These sit alongside the
# five shared BENCHMARK_METRICS on parquet/iceberg entries and separate table
# metadata-scan cost from Python/PyTorch payload hydration cost. The shuffled and
# filter-change metric names match the enterprise-lance report keys so the two
# families read consistently.
METRIC_METADATA_SCAN_LATENCY = "metadata_scan_latency"
METRIC_ROW_HYDRATION_LATENCY = "row_hydration_latency"
METRIC_SHUFFLED_EPOCH_THROUGHPUT = "shuffled_epoch_throughput"
METRIC_SUBSET_FILTER_CHANGE = "subset_filter_change"
ANALYTICS_BASELINE_METRICS = (
    METRIC_METADATA_SCAN_LATENCY,
    METRIC_ROW_HYDRATION_LATENCY,
    METRIC_SHUFFLED_EPOCH_THROUGHPUT,
    METRIC_SUBSET_FILTER_CHANGE,
)
BENCHMARK_METRICS = (
    METRIC_DATALOADER_THROUGHPUT,
    METRIC_RANDOM_ACCESS_LATENCY,
    METRIC_RANDOM_FRAME_SAMPLING,
    METRIC_QUERY_TO_DATASET_CURATION,
    METRIC_STORAGE_FOOTPRINT,
)


class BenchmarkError(Exception):
    """Raised when a benchmark report cannot be produced."""


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Deterministic parameters recorded in every benchmark report."""

    formats: tuple[str, ...] = DEFAULT_BENCHMARK_FORMATS
    sample_limit: int = 256
    random_access_samples: int = 64
    random_frame_samples: int = 16
    frames_per_clip: int = 4
    seed: int = 0
    query_limit: int = 128
    fixed_quality: str = "source-payload-bytes"
    source_dataset_id: str | None = None
    source_dataset_revision: str | None = None
    source_dataset_uri: str | None = None
    source_dataset_size_tier: str | None = None
    source_dataset_size_bytes: int | None = None
    source_dataset_size_note: str | None = None
    storage_tier: str = "local"
    lerobot_native_source_mode: str = "auto"
    lerobot_native_cache_mode: str = "auto"
    lerobot_native_episode_limit: int | None = None
    enterprise_fixture_uri: str | None = None
    enterprise_region: str = "us-east-1"
    enterprise_host_override: str | None = None
    enterprise_live: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "formats": list(self.formats),
            "sample_limit": self.sample_limit,
            "random_access_samples": self.random_access_samples,
            "random_frame_samples": self.random_frame_samples,
            "frames_per_clip": self.frames_per_clip,
            "seed": self.seed,
            "query_limit": self.query_limit,
            "fixed_quality": self.fixed_quality,
            "source_dataset_id": self.source_dataset_id,
            "source_dataset_revision": self.source_dataset_revision,
            "source_dataset_uri": self.source_dataset_uri,
            "source_dataset_size_tier": self.source_dataset_size_tier,
            "source_dataset_size_bytes": self.source_dataset_size_bytes,
            "source_dataset_size_note": self.source_dataset_size_note,
            "storage_tier": self.storage_tier,
            "lerobot_native_source_mode": self.lerobot_native_source_mode,
            "lerobot_native_cache_mode": self.lerobot_native_cache_mode,
            "lerobot_native_episode_limit": self.lerobot_native_episode_limit,
            "enterprise_fixture_uri": self.enterprise_fixture_uri,
            "enterprise_region": self.enterprise_region,
            "enterprise_host_override": self.enterprise_host_override,
            "enterprise_live": self.enterprise_live,
        }


@dataclass(frozen=True)
class _LeRobotNativeComponents:
    dataset_cls: Any
    dataloader_cls: Any
    dataset_api: str = "lerobot.datasets.LeRobotDataset"


def run_benchmark_suite(
    lake: Lake,
    snapshot_name: str,
    *,
    formats: tuple[str, ...] | list[str] | None = None,
    output_dir: str | Path | None = None,
    sample_limit: int = 256,
    random_access_samples: int = 64,
    random_frame_samples: int = 16,
    frames_per_clip: int = 4,
    seed: int = 0,
    query_limit: int = 128,
    source_dataset_id: str | None = None,
    source_dataset_revision: str | None = None,
    source_dataset_uri: str | None = None,
    source_dataset_size_tier: str | None = None,
    source_dataset_size_bytes: int | None = None,
    source_dataset_size_note: str | None = None,
    storage_tier: str = "local",
    lerobot_native_source_mode: str = "auto",
    lerobot_native_cache_mode: str = "auto",
    lerobot_native_episode_limit: int | None = None,
    enterprise_fixture_uri: str | None = None,
    enterprise_region: str = "us-east-1",
    enterprise_host_override: str | None = None,
    enterprise_live: bool = False,
) -> dict[str, Any]:
    """Run the benchmark harness and return a structured JSON-safe report."""
    selected_formats = _normalize_formats(formats or DEFAULT_BENCHMARK_FORMATS)
    config = BenchmarkRunConfig(
        formats=selected_formats,
        sample_limit=_positive_int(sample_limit, "sample_limit"),
        random_access_samples=_positive_int(random_access_samples, "random_access_samples"),
        random_frame_samples=_positive_int(random_frame_samples, "random_frame_samples"),
        frames_per_clip=_positive_int(frames_per_clip, "frames_per_clip"),
        seed=int(seed),
        query_limit=_positive_int(query_limit, "query_limit"),
        source_dataset_id=source_dataset_id,
        source_dataset_revision=source_dataset_revision,
        source_dataset_uri=source_dataset_uri,
        source_dataset_size_tier=source_dataset_size_tier,
        source_dataset_size_bytes=source_dataset_size_bytes,
        source_dataset_size_note=source_dataset_size_note,
        storage_tier=str(storage_tier),
        lerobot_native_source_mode=_normalize_lerobot_native_source_mode(
            lerobot_native_source_mode
        ),
        lerobot_native_cache_mode=_normalize_lerobot_native_cache_mode(
            lerobot_native_cache_mode
        ),
        lerobot_native_episode_limit=_optional_positive_int(
            lerobot_native_episode_limit,
            "lerobot_native_episode_limit",
        ),
        enterprise_fixture_uri=enterprise_fixture_uri,
        enterprise_region=str(enterprise_region),
        enterprise_host_override=enterprise_host_override,
        enterprise_live=bool(enterprise_live),
    )
    artifact_root = Path(output_dir) if output_dir is not None else None
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)

    snapshot = _snapshot_metadata(lake, snapshot_name)
    hardware = _hardware_info()
    report = {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "dataset": _dataset_report(lake, snapshot, config),
        "params": config.to_dict(),
        "storage_tiers": _storage_tier_report(lake, config),
        "hardware": hardware,
        "format_versions": _format_versions(),
        "methodology": _methodology(),
        "formats": {},
    }

    for fmt in selected_formats:
        if fmt == ENTERPRISE_LANCE_FORMAT:
            report["formats"][fmt] = _run_enterprise_lance_format(
                lake,
                snapshot,
                config=config,
                hardware=hardware,
            )
        elif fmt == LANCE_FORMAT:
            report["formats"][fmt] = _run_lance_format(
                lake,
                snapshot,
                config=config,
                hardware=hardware,
            )
        elif fmt == LEROBOT_DEFAULT_FORMAT:
            report["formats"][fmt] = _run_lerobot_default_format(
                lake,
                snapshot,
                config=config,
                artifact_root=artifact_root,
            )
        elif fmt == LEROBOT_NATIVE_FORMAT:
            report["formats"][fmt] = _run_lerobot_native_format(
                lake,
                snapshot,
                config=config,
                artifact_root=artifact_root,
            )
        elif fmt == WEBDATASET_FORMAT:
            report["formats"][fmt] = _run_webdataset_format(
                lake,
                snapshot,
                config=config,
                artifact_root=artifact_root,
            )
        elif fmt == PARQUET_FORMAT:
            report["formats"][fmt] = _run_parquet_format(
                lake,
                snapshot,
                config=config,
                artifact_root=artifact_root,
            )
        elif fmt == ICEBERG_FORMAT:
            report["formats"][fmt] = _run_iceberg_format(
                lake,
                snapshot,
                config=config,
                artifact_root=artifact_root,
            )
        elif fmt == DEEPLAKE_FORMAT:
            report["formats"][fmt] = _skip_optional_format(fmt)
        else:  # Defensive guard; _normalize_formats should catch this first.
            raise BenchmarkError(f"unsupported benchmark format {fmt!r}")

    report["comparison_table"] = _comparison_table(report["formats"])
    return report


def write_benchmark_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write ``report`` as sorted, indented JSON and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return out


ENTERPRISE_CALIBRATION_SCHEMA_VERSION = "lancedb-robotics-enterprise-benchmark-calibration-v1"


def compare_enterprise_benchmark_results(
    fake_report: Mapping[str, Any],
    live_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Calibrate a fake-local ``enterprise-lance`` result against a live one.

    Each argument is either a full ``run_benchmark_suite`` report or the
    ``report["formats"]["enterprise-lance"]`` entry from one, taken from two
    separate runs over the same logical sample set: one with
    ``enterprise_fixture_uri`` (profile ``fake-local-db``) and one with
    ``enterprise_live=True`` against a real endpoint (profile ``live-db`` or
    ``live-namespace``). The result labels every compared metric with its
    source's confidence level so fake-local numbers can never be read as
    production performance (backlog 0125).
    """
    fake_entry = _enterprise_format_entry(fake_report)
    live_entry = _enterprise_format_entry(live_report)
    for label, entry in (("fake_report", fake_entry), ("live_report", live_entry)):
        if entry.get("status") != "completed":
            raise BenchmarkError(
                f"{label} enterprise-lance entry is not completed "
                f"(status={entry.get('status')!r})"
            )
    fake_profile = str(fake_entry["remote_endpoint"]["profile"])
    live_profile = str(live_entry["remote_endpoint"]["profile"])
    if fake_profile != FAKE_LOCAL_DB_PROFILE:
        raise BenchmarkError(
            f"fake_report must use the {FAKE_LOCAL_DB_PROFILE!r} profile, got {fake_profile!r}"
        )
    if live_profile not in (LIVE_DB_PROFILE, LIVE_NAMESPACE_PROFILE):
        raise BenchmarkError(
            f"live_report must use a live Enterprise profile, got {live_profile!r}"
        )

    metric_names = (
        "query_to_first_batch_latency",
        "shuffled_epoch_throughput",
        METRIC_RANDOM_ACCESS_LATENCY,
    )
    deltas: dict[str, Any] = {}
    for name in metric_names:
        fake_value = float(fake_entry["metrics"][name]["value"])
        live_value = float(live_entry["metrics"][name]["value"])
        deltas[name] = {
            "fake_local_value": fake_value,
            "live_value": live_value,
            "unit": fake_entry["metrics"][name].get("unit"),
            "absolute_delta": live_value - fake_value,
            "relative_delta_pct": (
                ((live_value - fake_value) / fake_value * 100.0) if fake_value else None
            ),
        }
    return {
        "schema_version": ENTERPRISE_CALIBRATION_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "fake_local": {
            "profile": fake_profile,
            "confidence": fake_entry["remote_endpoint"].get("confidence", "sdk-contract-only"),
        },
        "live": {
            "profile": live_profile,
            "confidence": live_entry["remote_endpoint"].get(
                "confidence", "production-calibrated"
            ),
            "degraded_capabilities": list(
                live_entry["remote_endpoint"].get("degraded_capabilities") or []
            ),
            "degraded_phases": list(live_entry.get("degraded_phases") or []),
        },
        "metrics": deltas,
        "notes": [
            "fake-local-db numbers are sdk-contract-only: they prove report shape and "
            "cache-phase wiring, not production performance.",
            f"live numbers are labeled {live_entry['remote_endpoint'].get('confidence')} and "
            "reflect the configured endpoint's real network, placement, and cache behavior.",
        ],
    }


def _enterprise_format_entry(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if "formats" in report:
        entry = report["formats"].get(ENTERPRISE_LANCE_FORMAT)
        if entry is None:
            raise BenchmarkError("report has no enterprise-lance format entry")
        return entry
    return report


def plan_public_lerobot_benchmark_capacity(
    *,
    size_tier: str = DEFAULT_LEROBOT_BENCHMARK_SIZE_TIER,
    storage_tier: str = "hf-cache",
    artifact_root: str | Path | None = None,
    source_dataset_size_bytes: int | None = None,
    sample_limit: int = 256,
    random_access_samples: int = 64,
    random_frame_samples: int = 16,
    frames_per_clip: int = 4,
    max_source_bytes: int | None = None,
    max_artifact_bytes: int | None = None,
    time_budget_seconds: int | None = None,
    require_gpu: bool | None = None,
    gpu_available: bool | None = None,
    require_object_store: bool | None = None,
    publication_destination: str | None = None,
) -> dict[str, Any]:
    """Plan whether a public LeRobot benchmark tier fits configured budgets."""

    tier_key = _normalize_capacity_tier(size_tier)
    budget = _public_lerobot_capacity_budget(
        artifact_root=artifact_root,
        max_source_bytes=max_source_bytes,
        max_artifact_bytes=max_artifact_bytes,
        time_budget_seconds=time_budget_seconds,
        require_gpu=require_gpu,
        gpu_available=gpu_available,
        require_object_store=require_object_store,
        publication_destination=publication_destination,
        storage_tier=storage_tier,
    )
    requested_samples = {
        "sample_limit": _positive_int(sample_limit, "sample_limit"),
        "random_access_samples": _positive_int(
            random_access_samples, "random_access_samples"
        ),
        "random_frame_samples": _positive_int(
            random_frame_samples, "random_frame_samples"
        ),
        "frames_per_clip": _positive_int(frames_per_clip, "frames_per_clip"),
        "decoded_frame_probes": int(random_frame_samples) * int(frames_per_clip),
    }
    ordered_tiers = list(PUBLIC_LEROBOT_CAPACITY_ESTIMATES)
    if tier_key not in ordered_tiers:
        ordered_tiers.append(tier_key)
    tiers = [
        _public_lerobot_capacity_tier_plan(
            tier,
            budget=budget,
            storage_tier=storage_tier,
            source_dataset_size_bytes=(
                source_dataset_size_bytes if tier == tier_key else None
            ),
            selected=tier == tier_key,
        )
        for tier in ordered_tiers
    ]
    selected = next(tier for tier in tiers if tier["selected"])
    return {
        "schema_version": PUBLIC_LEROBOT_CAPACITY_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "status": selected["status"],
        "selected_tier": tier_key,
        "storage_tier": str(storage_tier),
        "skip_reasons": list(selected["skip_reasons"]),
        "requested_samples": requested_samples,
        "budgets": budget,
        "selected": selected,
        "tiers": tiers,
    }


def validate_public_lerobot_benchmark_claims(
    artifact_root: str | Path,
    *,
    claims_path: str | Path | None = None,
    claims: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate retained public LeRobot benchmark evidence and claim references."""

    root = Path(artifact_root)
    diagnostics: list[dict[str, Any]] = []
    if not root.exists():
        _add_validation_error(
            diagnostics,
            "artifact-root-missing",
            f"public benchmark artifact root does not exist: {root}",
            path=str(root),
        )
    manifests = _public_benchmark_manifest_candidates(root) if root.exists() else []
    if not manifests:
        _add_validation_error(
            diagnostics,
            "manifest-missing",
            f"no public LeRobot benchmark manifests found under {root}",
            path=str(root / "runs"),
        )
    manifest_by_id = {str(manifest.get("report_id")): manifest for manifest in manifests}
    manifest_summaries = [
        _validate_public_benchmark_manifest(manifest, root=root, diagnostics=diagnostics)
        for manifest in manifests
    ]
    index = _load_public_benchmark_index(root, diagnostics)
    _validate_public_benchmark_index(
        index,
        manifests=manifest_by_id,
        root=root,
        diagnostics=diagnostics,
    )
    claim_payload, claim_entries = _load_public_benchmark_claims(
        claims_path=claims_path,
        claims=claims,
        diagnostics=diagnostics,
    )
    claim_summaries = [
        _validate_public_benchmark_claim(
            claim,
            claim_index=index,
            manifests=manifest_by_id,
            diagnostics=diagnostics,
        )
        for index, claim in enumerate(claim_entries)
    ]
    error_count = sum(1 for diagnostic in diagnostics if diagnostic["level"] == "error")
    warning_count = sum(1 for diagnostic in diagnostics if diagnostic["level"] == "warning")
    return {
        "schema_version": PUBLIC_LEROBOT_CLAIM_VALIDATION_SCHEMA_VERSION,
        "validated_at": _now_iso(),
        "status": "failed" if error_count else "passed",
        "artifact_root": str(root),
        "manifest_count": len(manifests),
        "claim_count": len(claim_entries),
        "error_count": error_count,
        "warning_count": warning_count,
        "diagnostics": diagnostics,
        "manifests": manifest_summaries,
        "claims": claim_summaries,
        "claim_manifest": claim_payload,
    }


def run_public_lerobot_benchmark(
    lake: Lake,
    *,
    artifact_root: str | Path,
    source: str | Path = DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
    revision: str | None = None,
    report_id: str | None = None,
    snapshot_name: str = "lerobot-droid-100-benchmark",
    formats: tuple[str, ...] | list[str] | None = None,
    sample_limit: int = 256,
    random_access_samples: int = 64,
    random_frame_samples: int = 16,
    frames_per_clip: int = 4,
    seed: int = 0,
    query_limit: int = 128,
    size_tier: str = DEFAULT_LEROBOT_BENCHMARK_SIZE_TIER,
    storage_tier: str = "hf-cache",
    source_dataset_size_bytes: int | None = None,
    source_dataset_size_note: str | None = None,
    enterprise_fixture_uri: str | None = None,
    enterprise_region: str = "us-east-1",
    enterprise_host_override: str | None = None,
    lerobot_native_source_mode: str = "auto",
    lerobot_native_cache_mode: str = "auto",
    lerobot_native_episode_limit: int | None = None,
    capacity_dry_run: bool = False,
    capacity_max_source_bytes: int | None = None,
    capacity_max_artifact_bytes: int | None = None,
    capacity_time_budget_seconds: int | None = None,
    capacity_require_gpu: bool | None = None,
    capacity_require_object_store: bool | None = None,
    capacity_publication_destination: str | None = None,
    created_by: str = "lancedb-robotics-bench",
    compact: bool = True,
    prune_versions: bool = True,
    index_predicates: bool = True,
) -> dict[str, Any]:
    """Run and retain the scheduled public LeRobot benchmark artifact set.

    This is intentionally a CI-friendly orchestration helper rather than a
    scheduler daemon: callers provide a destination and an optional report id,
    and the helper writes all retained artifacts plus an updated history index.
    """

    resolved_revision = _public_benchmark_revision(source, revision)
    _require_public_lerobot_revision(source, resolved_revision)
    created_at = _now_iso()
    commit = _git_commit()
    resolved_report_id = report_id or _public_benchmark_report_id(
        source=source,
        revision=resolved_revision,
        size_tier=size_tier,
        created_at=created_at,
        commit=commit,
    )
    root = Path(artifact_root)
    run_dir = root / "runs" / resolved_report_id
    report_dir = run_dir / "reports"
    benchmark_artifacts_dir = run_dir / "artifacts"
    log_dir = run_dir / "logs"
    prepare_path = report_dir / "prepare.json"
    benchmark_path = report_dir / "benchmark.json"
    manifest_path = run_dir / "artifact-manifest.json"
    log_path = log_dir / "run.log"
    capacity_path = report_dir / "capacity.json"
    selected_formats = _normalize_formats(formats or DEFAULT_BENCHMARK_FORMATS)
    capacity = plan_public_lerobot_benchmark_capacity(
        size_tier=size_tier,
        storage_tier=storage_tier,
        artifact_root=root,
        source_dataset_size_bytes=source_dataset_size_bytes,
        sample_limit=sample_limit,
        random_access_samples=random_access_samples,
        random_frame_samples=random_frame_samples,
        frames_per_clip=frames_per_clip,
        max_source_bytes=capacity_max_source_bytes,
        max_artifact_bytes=capacity_max_artifact_bytes,
        time_budget_seconds=capacity_time_budget_seconds,
        require_gpu=capacity_require_gpu,
        require_object_store=capacity_require_object_store,
        publication_destination=capacity_publication_destination,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    benchmark_artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_report(capacity, capacity_path)

    if capacity_dry_run or capacity["status"] == "skipped":
        return _write_public_lerobot_capacity_result(
            lake,
            artifact_root=root,
            run_dir=run_dir,
            report_dir=report_dir,
            manifest_path=manifest_path,
            log_path=log_path,
            capacity_path=capacity_path,
            report_id=resolved_report_id,
            created_at=created_at,
            commit=commit,
            source=source,
            revision=resolved_revision,
            snapshot_name=snapshot_name,
            size_tier=size_tier,
            storage_tier=storage_tier,
            formats=selected_formats,
            capacity=capacity,
            dry_run=capacity_dry_run,
        )

    prepare_report = prepare_lerobot_benchmark_dataset(
        lake,
        source,
        revision=resolved_revision,
        snapshot_name=snapshot_name,
        size_tier=size_tier,
        storage_tier=storage_tier,
        created_by=created_by,
        compact=compact,
        prune_versions=prune_versions,
        index_predicates=index_predicates,
    )
    write_benchmark_report(prepare_report, prepare_path)

    benchmark_report = run_benchmark_suite(
        lake,
        str(prepare_report["snapshot_name"]),
        formats=selected_formats,
        output_dir=benchmark_artifacts_dir,
        sample_limit=sample_limit,
        random_access_samples=random_access_samples,
        random_frame_samples=random_frame_samples,
        frames_per_clip=frames_per_clip,
        seed=seed,
        query_limit=query_limit,
        source_dataset_id=str(prepare_report["dataset_id"]),
        source_dataset_revision=prepare_report.get("revision") or resolved_revision,
        source_dataset_uri=str(prepare_report["source_uri"]),
        source_dataset_size_tier=str(prepare_report["size_tier"]),
        source_dataset_size_bytes=(
            source_dataset_size_bytes
            if source_dataset_size_bytes is not None
            else prepare_report.get("source_size_bytes")
        ),
        source_dataset_size_note=source_dataset_size_note or prepare_report.get("size_note"),
        storage_tier=str(prepare_report["storage_tier"]),
        enterprise_fixture_uri=enterprise_fixture_uri,
        enterprise_region=enterprise_region,
        enterprise_host_override=enterprise_host_override,
        lerobot_native_source_mode=lerobot_native_source_mode,
        lerobot_native_cache_mode=lerobot_native_cache_mode,
        lerobot_native_episode_limit=lerobot_native_episode_limit,
    )
    benchmark_report["capacity"] = capacity
    write_benchmark_report(benchmark_report, benchmark_path)

    _write_public_benchmark_log(
        log_path,
        report_id=resolved_report_id,
        created_at=created_at,
        commit=commit,
        source=source,
        revision=resolved_revision,
        formats=selected_formats,
        prepare_path=prepare_path,
        benchmark_path=benchmark_path,
        capacity_path=capacity_path,
    )
    manifest = _public_benchmark_manifest(
        report_id=resolved_report_id,
        created_at=created_at,
        finished_at=_now_iso(),
        commit=commit,
        artifact_root=root,
        run_dir=run_dir,
        prepare_path=prepare_path,
        benchmark_path=benchmark_path,
        manifest_path=manifest_path,
        log_path=log_path,
        capacity_path=capacity_path,
        benchmark_artifacts_dir=benchmark_artifacts_dir,
        prepare_report=prepare_report,
        benchmark_report=benchmark_report,
        capacity=capacity,
    )
    write_benchmark_report(manifest, manifest_path)
    manifest["artifact_files"] = _artifact_files(run_dir)
    write_benchmark_report(manifest, manifest_path)
    dashboard = write_public_lerobot_benchmark_dashboard(root)
    manifest["history"] = {
        "index_path": str(dashboard["index_path"]),
        "dashboard_path": str(dashboard["dashboard_path"]),
        "run_count": dashboard["index"]["run_count"],
        "latest_report_id": dashboard["index"].get("latest_report_id"),
    }
    write_benchmark_report(manifest, manifest_path)
    dashboard = write_public_lerobot_benchmark_dashboard(root)
    return {
        "status": "completed",
        "report_id": resolved_report_id,
        "artifact_root": str(root),
        "run_dir": str(run_dir),
        "prepare_report": prepare_report,
        "benchmark_report": benchmark_report,
        "manifest": manifest,
        "capacity": capacity,
        "paths": {
            "prepare_report": str(prepare_path),
            "benchmark_report": str(benchmark_path),
            "capacity_report": str(capacity_path),
            "artifact_manifest": str(manifest_path),
            "log": str(log_path),
            "benchmark_artifacts": str(benchmark_artifacts_dir),
            "index": str(dashboard["index_path"]),
            "dashboard": str(dashboard["dashboard_path"]),
        },
        "index": dashboard["index"],
    }


def _write_public_lerobot_capacity_result(
    lake: Lake,
    *,
    artifact_root: Path,
    run_dir: Path,
    report_dir: Path,
    manifest_path: Path,
    log_path: Path,
    capacity_path: Path,
    report_id: str,
    created_at: str,
    commit: str,
    source: str | Path,
    revision: str | None,
    snapshot_name: str,
    size_tier: str,
    storage_tier: str,
    formats: tuple[str, ...],
    capacity: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    run_status = "dry-run" if dry_run else "skipped"
    skip_reasons = list(capacity.get("skip_reasons") or [])
    if dry_run:
        skip_reasons = [
            "capacity dry-run requested; prepare and benchmark execution were not run"
        ] + skip_reasons
    reason = "; ".join(skip_reasons) or "capacity gate skipped benchmark execution"
    capacity_report = dict(capacity)
    capacity_report["run_status"] = run_status
    capacity_report["dry_run"] = bool(dry_run)
    write_benchmark_report(capacity_report, capacity_path)
    _write_public_benchmark_log(
        log_path,
        report_id=report_id,
        created_at=created_at,
        commit=commit,
        source=source,
        revision=revision,
        formats=formats,
        capacity_path=capacity_path,
        status=run_status,
        skip_reason=reason,
    )
    manifest = {
        "schema_version": PUBLIC_LEROBOT_BENCHMARK_SCHEMA_VERSION,
        "report_id": report_id,
        "status": run_status,
        "skip_reason": reason,
        "created_at": created_at,
        "finished_at": _now_iso(),
        "commit": commit,
        "dataset": {
            "dataset_id": _public_lerobot_dataset_id(source),
            "revision": revision,
            "source_uri": _source_uri_for_benchmark(source),
            "size_tier": str(size_tier),
            "storage_tier": str(storage_tier),
            "snapshot_name": snapshot_name,
            "snapshot_dataset_id": None,
            "scenario_count": None,
        },
        "paths": {
            "artifact_root": str(artifact_root),
            "run_dir": str(run_dir),
            "capacity_report": str(capacity_path),
            "artifact_manifest": str(manifest_path),
            "log": str(log_path),
            "benchmark_artifacts": str(run_dir / "artifacts"),
        },
        "capacity": capacity_report,
        "format_statuses": {fmt: "skipped" for fmt in formats},
        "skipped": {fmt: reason for fmt in formats},
        "comparison_table": [],
        "hardware": _hardware_info(),
        "format_versions": _format_versions(),
        "storage_tiers": {
            "active": str(storage_tier),
            "tiers": [
                {
                    "tier": str(storage_tier),
                    "status": run_status,
                    "skip_reason": reason,
                }
            ],
        },
        "artifact_files": [],
        "history": {},
    }
    write_benchmark_report(manifest, manifest_path)
    manifest["artifact_files"] = _artifact_files(run_dir)
    write_benchmark_report(manifest, manifest_path)
    dashboard = write_public_lerobot_benchmark_dashboard(artifact_root)
    manifest["history"] = {
        "index_path": str(dashboard["index_path"]),
        "dashboard_path": str(dashboard["dashboard_path"]),
        "run_count": dashboard["index"]["run_count"],
        "latest_report_id": dashboard["index"].get("latest_report_id"),
    }
    write_benchmark_report(manifest, manifest_path)
    dashboard = write_public_lerobot_benchmark_dashboard(artifact_root)
    return {
        "status": run_status,
        "report_id": report_id,
        "artifact_root": str(artifact_root),
        "run_dir": str(run_dir),
        "capacity": capacity_report,
        "manifest": manifest,
        "paths": {
            "capacity_report": str(capacity_path),
            "artifact_manifest": str(manifest_path),
            "log": str(log_path),
            "benchmark_artifacts": str(run_dir / "artifacts"),
            "index": str(dashboard["index_path"]),
            "dashboard": str(dashboard["dashboard_path"]),
        },
        "index": dashboard["index"],
    }


def write_public_lerobot_benchmark_dashboard(artifact_root: str | Path) -> dict[str, Any]:
    """Rebuild the retained public LeRobot benchmark history and markdown dashboard."""

    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    manifests = _public_benchmark_manifests(root)
    runs = [_public_benchmark_history_row(manifest, root=root) for manifest in manifests]
    runs.sort(key=lambda row: (str(row.get("created_at") or ""), str(row["report_id"])))
    latest = runs[-1] if runs else None
    index = {
        "schema_version": PUBLIC_LEROBOT_BENCHMARK_SCHEMA_VERSION,
        "updated_at": _now_iso(),
        "run_count": len(runs),
        "latest_report_id": latest["report_id"] if latest else None,
        "runs": runs,
    }
    index_path = root / "index.json"
    dashboard_path = root / "dashboard.md"
    write_benchmark_report(index, index_path)
    dashboard_path.write_text(_public_benchmark_dashboard_markdown(index) + "\n")
    return {
        "index_path": index_path,
        "dashboard_path": dashboard_path,
        "index": index,
    }


def publish_public_lerobot_benchmark(
    artifact_root: str | Path,
    *,
    destination: str | Path,
    report_id: str | None = None,
    retention_class: str = "public-benchmark-history",
    retain_latest: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Publish retained public LeRobot benchmark artifacts to a durable target."""

    root = Path(artifact_root)
    if not root.exists():
        raise BenchmarkError(f"public benchmark artifact root does not exist: {root}")
    if retain_latest is not None and int(retain_latest) <= 0:
        raise BenchmarkError("retain_latest must be positive when provided")
    manifests = _public_benchmark_manifests(root)
    if report_id is not None:
        manifests = [manifest for manifest in manifests if manifest.get("report_id") == report_id]
    if not manifests:
        target = f" report_id={report_id!r}" if report_id else ""
        raise BenchmarkError(f"no retained public LeRobot benchmark manifests found in {root}{target}")

    retention = _public_benchmark_retention_plan(
        manifests,
        retention_class=retention_class,
        retain_latest=retain_latest,
    )
    destination_value = str(destination)
    backend = _publication_backend(destination_value)
    published_at = _now_iso()

    refreshed: list[dict[str, Any]] = []
    for manifest in manifests:
        manifest_path = Path(str(manifest["_manifest_path"]))
        run_dir = manifest_path.parent
        updated = _refresh_public_benchmark_manifest(
            manifest,
            root=root,
            run_dir=run_dir,
            destination=destination_value,
            backend_name=backend["name"],
            published_at=published_at,
            dry_run=dry_run,
            retention=retention[str(manifest["report_id"])],
        )
        write_benchmark_report(updated, manifest_path)
        refreshed.append({**updated, "_manifest_path": str(manifest_path)})

    dashboard = write_public_lerobot_benchmark_dashboard(root)
    files = _public_benchmark_publication_files(root, refreshed)
    for rel in ("index.json", "dashboard.md"):
        path = root / rel
        if path.exists() and rel not in files:
            files.append(rel)
    files = sorted(files)

    written: list[str] = []
    updated_files: list[str] = []
    unchanged: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for rel in files:
        source_path = root / rel
        if not source_path.is_file():
            continue
        source_bytes = source_path.read_bytes()
        source_sha = _sha256_bytes(source_bytes)
        existing = _publication_read_bytes(backend, rel)
        immutable = rel.startswith("runs/")
        if existing is not None:
            existing_sha = _sha256_bytes(existing)
            if immutable and existing_sha != source_sha:
                conflicts.append(
                    {
                        "path": rel,
                        "source_sha256": source_sha,
                        "destination_sha256": existing_sha,
                    }
                )
                continue
            if existing_sha == source_sha:
                unchanged.append(rel)
                continue
            if dry_run:
                updated_files.append(rel)
                continue
            _publication_write_bytes(backend, rel, source_bytes)
            updated_files.append(rel)
            continue
        if dry_run:
            written.append(rel)
            continue
        _publication_write_bytes(backend, rel, source_bytes)
        written.append(rel)

    if conflicts:
        detail = ", ".join(item["path"] for item in conflicts[:5])
        raise BenchmarkError(
            "published public LeRobot benchmark run artifacts are immutable; "
            f"destination has different content for {detail}"
        )

    report_ids = sorted(str(manifest["report_id"]) for manifest in refreshed)
    status = "dry-run" if dry_run else "published"
    return {
        "schema_version": PUBLIC_LEROBOT_PUBLICATION_SCHEMA_VERSION,
        "status": status,
        "artifact_root": str(root),
        "destination": destination_value,
        "backend": backend["name"],
        "published_at": published_at,
        "dry_run": bool(dry_run),
        "report_ids": report_ids,
        "retention": {
            "class": retention_class,
            "retain_latest": retain_latest,
            "plan": {key: dict(value) for key, value in sorted(retention.items())},
        },
        "files": {
            "planned_or_written": written,
            "updated": updated_files,
            "unchanged": unchanged,
            "conflicts": conflicts,
            "count": len(files),
        },
        "index_path": str(dashboard["index_path"]),
        "dashboard_path": str(dashboard["dashboard_path"]),
    }


def prepare_lerobot_benchmark_dataset(
    lake: Lake,
    source: str | Path = DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
    *,
    revision: str | None = None,
    snapshot_name: str = "lerobot-droid-100-benchmark",
    size_tier: str = DEFAULT_LEROBOT_BENCHMARK_SIZE_TIER,
    storage_tier: str = "hf-cache",
    created_by: str = "lancedb-robotics-bench",
    compact: bool = True,
    prune_versions: bool = True,
    index_predicates: bool = True,
) -> dict[str, Any]:
    """Fetch/ingest a public LeRobot corpus and freeze a benchmark snapshot.

    The default source is the Hugging Face Hub `lerobot/droid_100` dataset. Tests
    can pass a local LeRobot fixture path, so this prepare step stays offline in
    CI while the same command works against the public corpus in real runs.
    """
    source_ref = _lerobot_source_with_revision(source, revision)
    started_at = _now_iso()
    ingest_report = ingest_lerobot(
        lake,
        source_ref,
        created_by=created_by,
        compact=compact,
        prune_versions=prune_versions,
        index_predicates=index_predicates,
    )
    scenario_ids = sorted(
        str(row["scenario_id"])
        for row in lake.table("scenarios").to_arrow().to_pylist()
        if row["run_id"] == ingest_report.run_id
    )
    if not scenario_ids:
        raise BenchmarkError(
            f"LeRobot benchmark source {source!r} ingested no episode scenarios"
        )
    manifest = create_snapshot(
        lake,
        name=snapshot_name,
        scenario_ids=scenario_ids,
        source={
            "kind": "benchmark-prepare",
            "format": "lerobot",
            "source": str(source),
            "source_ref": str(source_ref),
            "revision": revision,
            "size_tier": size_tier,
            "storage_tier": storage_tier,
            "default_public_dataset": DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
        },
        split_by="scenario",
        created_by=created_by,
    )
    source_size_bytes = _path_size(source) if not _looks_like_hf_source(str(source)) else None
    return {
        "status": "completed",
        "prepared_at": started_at,
        "lake_uri": lake.uri,
        "dataset_id": str(source) if _looks_like_hf_source(str(source)) else Path(source).name,
        "source_uri": _source_uri_for_benchmark(source),
        "source_ref": str(source_ref),
        "revision": revision,
        "size_tier": str(size_tier),
        "storage_tier": str(storage_tier),
        "source_size_bytes": source_size_bytes,
        "size_note": _benchmark_size_note(size_tier, source_size_bytes),
        "availability": _lerobot_source_availability(source),
        "run_id": ingest_report.run_id,
        "snapshot_name": manifest.name,
        "snapshot_dataset_id": manifest.dataset_id,
        "scenario_count": len(scenario_ids),
        "rows_added": dict(ingest_report.rows_added),
        "verification_note": (
            "Use bench run with the returned snapshot_name and pass the descriptor "
            "fields into the report params for reproducible comparisons."
        ),
    }


def _source_corpus_descriptor(
    lake: Lake,
    snapshot: dict[str, Any],
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    dataset_id = config.source_dataset_id or snapshot["snapshot_name"]
    source_uri_value = config.source_dataset_uri or lake.uri
    size_bytes = (
        int(config.source_dataset_size_bytes)
        if config.source_dataset_size_bytes is not None
        else _path_size(source_uri_value)
    )
    size_tier = config.source_dataset_size_tier or _size_tier_from_bytes(size_bytes)
    return {
        "dataset_id": dataset_id,
        "revision": config.source_dataset_revision,
        "source_uri": source_uri_value,
        "size_tier": size_tier,
        "storage_tier": config.storage_tier,
        "size_bytes": size_bytes,
        "size_note": config.source_dataset_size_note
        or _benchmark_size_note(size_tier, size_bytes),
        "public_default_dataset_id": DEFAULT_LEROBOT_BENCHMARK_DATASET_ID,
        "public_default_source_uri": DEFAULT_LEROBOT_BENCHMARK_SOURCE_URI,
        "logical_snapshot": snapshot["snapshot_name"],
    }


def _lerobot_source_with_revision(source: str | Path, revision: str | None) -> str | Path:
    if revision is None or not _looks_like_hf_source(str(source)):
        return source
    value = str(source)
    if "@" in value:
        return value
    return f"{value}@{revision}"


def _looks_like_hf_source(value: str) -> bool:
    if Path(value).expanduser().exists():
        return False
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:@.+)?$", value))


def _source_uri_for_benchmark(source: str | Path) -> str:
    value = str(source)
    if _looks_like_hf_source(value):
        return "hf://" + value
    return source_uri(source)


def _public_lerobot_dataset_id(source: str | Path) -> str:
    value = str(source)
    if _looks_like_hf_source(value):
        return value.split("@", 1)[0]
    return Path(value).name


def _size_tier_from_bytes(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unspecified"
    if size_bytes < 1_000_000_000:
        return "fixture"
    if size_bytes < 100_000_000_000:
        return "mid"
    return "large"


def _benchmark_size_note(size_tier: str, size_bytes: int | None) -> str:
    if size_bytes is not None:
        return f"{size_bytes} bytes measured from local source/artifacts"
    if size_tier == DEFAULT_LEROBOT_BENCHMARK_SIZE_TIER:
        return (
            "DROID-100 is the default public smoke tier; exact bytes are recorded "
            "when the prepare command downloads or points at a local cache."
        )
    return "size unavailable; record source_size_bytes for published benchmark runs"


def _normalize_capacity_tier(size_tier: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(size_tier).strip().lower()).strip("-") or (
        DEFAULT_LEROBOT_BENCHMARK_SIZE_TIER
    )


def _public_lerobot_capacity_budget(
    *,
    artifact_root: str | Path | None,
    max_source_bytes: int | None,
    max_artifact_bytes: int | None,
    time_budget_seconds: int | None,
    require_gpu: bool | None,
    gpu_available: bool | None,
    require_object_store: bool | None,
    publication_destination: str | None,
    storage_tier: str,
) -> dict[str, Any]:
    gpu_required = _env_bool(
        PUBLIC_LEROBOT_CAPACITY_ENV["require_gpu"],
        explicit=require_gpu,
        default=False,
    )
    object_store_required = _env_bool(
        PUBLIC_LEROBOT_CAPACITY_ENV["require_object_store"],
        explicit=require_object_store,
        default=False,
    ) or str(storage_tier).strip().lower() == "object-store"
    destination = publication_destination or os.environ.get(
        PUBLIC_LEROBOT_CAPACITY_ENV["publication_destination"]
    )
    observed_gpu = None
    if gpu_required:
        observed_gpu = bool(
            gpu_available if gpu_available is not None else _public_lerobot_gpu_available()
        )
    return {
        "max_source_bytes": _env_int(
            PUBLIC_LEROBOT_CAPACITY_ENV["max_source_bytes"],
            explicit=max_source_bytes,
        ),
        "max_artifact_bytes": _env_int(
            PUBLIC_LEROBOT_CAPACITY_ENV["max_artifact_bytes"],
            explicit=max_artifact_bytes,
        ),
        "time_budget_seconds": _env_int(
            PUBLIC_LEROBOT_CAPACITY_ENV["time_budget_seconds"],
            explicit=time_budget_seconds,
        ),
        "require_gpu": gpu_required,
        "gpu_available": observed_gpu,
        "require_object_store": object_store_required,
        "publication_destination": destination,
        "publication_destination_ready": (
            _object_store_destination_ready(destination)
            if object_store_required
            else None
        ),
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "artifact_root_free_bytes": (
            _available_bytes(Path(artifact_root)) if artifact_root is not None else None
        ),
        "environment": dict(PUBLIC_LEROBOT_CAPACITY_ENV),
    }


def _public_lerobot_capacity_tier_plan(
    size_tier: str,
    *,
    budget: Mapping[str, Any],
    storage_tier: str,
    source_dataset_size_bytes: int | None,
    selected: bool,
) -> dict[str, Any]:
    estimate = _public_lerobot_capacity_estimate(
        size_tier,
        source_dataset_size_bytes=source_dataset_size_bytes,
    )
    requires_budget = size_tier in PUBLIC_LEROBOT_BUDGET_REQUIRED_TIERS
    checks: list[dict[str, Any]] = [
        _capacity_upper_bound_check(
            name="source-bytes",
            observed=estimate.get("source_bytes"),
            limit=budget.get("max_source_bytes"),
            required=requires_budget,
            missing_reason=(
                f"{size_tier} requires --capacity-max-source-bytes or "
                f"{PUBLIC_LEROBOT_CAPACITY_ENV['max_source_bytes']}"
            ),
        ),
        _capacity_upper_bound_check(
            name="artifact-bytes",
            observed=estimate.get("output_artifact_bytes"),
            limit=budget.get("max_artifact_bytes"),
            required=requires_budget,
            missing_reason=(
                f"{size_tier} requires --capacity-max-artifact-bytes or "
                f"{PUBLIC_LEROBOT_CAPACITY_ENV['max_artifact_bytes']}"
            ),
        ),
        _capacity_upper_bound_check(
            name="time-budget-seconds",
            observed=estimate.get("time_budget_seconds"),
            limit=budget.get("time_budget_seconds"),
            required=requires_budget,
            missing_reason=(
                f"{size_tier} requires --capacity-time-budget-seconds or "
                f"{PUBLIC_LEROBOT_CAPACITY_ENV['time_budget_seconds']}"
            ),
        ),
        _capacity_upper_bound_check(
            name="artifact-root-free-bytes",
            observed=estimate.get("output_artifact_bytes"),
            limit=budget.get("artifact_root_free_bytes"),
            required=False,
            missing_reason="artifact root free space could not be measured",
        ),
    ]
    if budget.get("require_gpu"):
        gpu_available = budget.get("gpu_available")
        checks.append(
            {
                "name": "gpu-availability",
                "status": "passed" if gpu_available else "skipped",
                "required": True,
                "observed": gpu_available,
                "skip_reason": None
                if gpu_available
                else "GPU availability was required but no GPU was detected",
            }
        )
    else:
        checks.append(
            {
                "name": "gpu-availability",
                "status": "not-required",
                "required": False,
                "observed": None,
                "skip_reason": None,
            }
        )
    if budget.get("require_object_store"):
        ready = budget.get("publication_destination_ready") or {}
        checks.append(
            {
                "name": "object-store-destination",
                "status": "passed" if ready.get("ready") else "skipped",
                "required": True,
                "observed": budget.get("publication_destination"),
                "skip_reason": ready.get("reason"),
            }
        )
    else:
        checks.append(
            {
                "name": "object-store-destination",
                "status": "not-required",
                "required": False,
                "observed": budget.get("publication_destination"),
                "skip_reason": None,
            }
        )
    skip_reasons = [
        str(check["skip_reason"])
        for check in checks
        if check.get("status") == "skipped" and check.get("skip_reason")
    ]
    return {
        "tier": size_tier,
        "selected": selected,
        "status": "skipped" if skip_reasons else "allowed",
        "storage_tier": str(storage_tier),
        "requires_budget": requires_budget,
        "estimate": estimate,
        "checks": checks,
        "skip_reasons": skip_reasons,
    }


def _public_lerobot_capacity_estimate(
    size_tier: str,
    *,
    source_dataset_size_bytes: int | None,
) -> dict[str, Any]:
    estimate = dict(PUBLIC_LEROBOT_CAPACITY_ESTIMATES.get(size_tier) or {})
    if not estimate:
        estimate = {
            "source_bytes": source_dataset_size_bytes,
            "decoded_sample_count": None,
            "output_artifact_bytes": None,
            "time_budget_seconds": None,
            "note": (
                "No built-in public LeRobot estimate for this tier; pass measured "
                "source bytes and explicit budgets for scheduled runs."
            ),
        }
    elif source_dataset_size_bytes is not None:
        estimate["source_bytes"] = int(source_dataset_size_bytes)
        estimate["note"] = (
            f"{estimate['note']} Source bytes were overridden by the measured "
            "or configured run value."
        )
    estimate["source"] = (
        "configured-or-measured"
        if source_dataset_size_bytes is not None
        else "tier-estimate"
    )
    return estimate


def _capacity_upper_bound_check(
    *,
    name: str,
    observed: int | None,
    limit: int | None,
    required: bool,
    missing_reason: str,
) -> dict[str, Any]:
    if observed is None:
        return {
            "name": name,
            "status": "skipped" if required else "unknown",
            "required": required,
            "observed": None,
            "limit": limit,
            "skip_reason": missing_reason if required else None,
        }
    if limit is None:
        return {
            "name": name,
            "status": "skipped" if required else "not-configured",
            "required": required,
            "observed": observed,
            "limit": None,
            "skip_reason": missing_reason if required else None,
        }
    if int(observed) > int(limit):
        return {
            "name": name,
            "status": "skipped",
            "required": required,
            "observed": int(observed),
            "limit": int(limit),
            "skip_reason": f"{name} estimate {observed} exceeds configured limit {limit}",
        }
    return {
        "name": name,
        "status": "passed",
        "required": required,
        "observed": int(observed),
        "limit": int(limit),
        "skip_reason": None,
    }


def _env_int(name: str, *, explicit: int | None) -> int | None:
    raw = explicit if explicit is not None else os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{name} must be an integer byte/count budget") from exc
    if value < 0:
        raise BenchmarkError(f"{name} must be non-negative")
    return value


def _env_bool(name: str, *, explicit: bool | None, default: bool) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise BenchmarkError(f"{name} must be a boolean value")


def _available_bytes(path: Path) -> int | None:
    probe = path.expanduser()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:
        return None


def _public_lerobot_gpu_available() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() not in {"", "-1"}:
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _object_store_destination_ready(destination: str | None) -> dict[str, Any]:
    if not destination:
        return {
            "ready": False,
            "reason": (
                "object-store destination required but no publication destination "
                f"was configured with --capacity-publication-destination or "
                f"{PUBLIC_LEROBOT_CAPACITY_ENV['publication_destination']}"
            ),
        }
    scheme = _uri_scheme(destination)
    if scheme not in PUBLIC_LEROBOT_OBJECT_STORE_SCHEMES:
        return {
            "ready": False,
            "reason": (
                f"publication destination {destination!r} is not an object-store URI; "
                f"expected one of {sorted(PUBLIC_LEROBOT_OBJECT_STORE_SCHEMES)}"
            ),
        }
    return {"ready": True, "reason": None, "scheme": scheme}


def _uri_scheme(value: str) -> str | None:
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*)://", str(value))
    return match.group(1).lower() if match else None


def _lerobot_source_availability(source: str | Path) -> dict[str, Any]:
    value = str(source)
    path = Path(value).expanduser()
    if path.exists():
        return {
            "status": "available",
            "reason": None,
            "notes": [f"local source path exists: {path}"],
        }
    if _looks_like_hf_source(value):
        return {
            "status": "skipped",
            "reason": "HF Hub availability is validated by ingest_lerobot during prepare",
            "notes": [
                "prepare avoids a second network probe; record the resolved revision "
                "for published benchmark runs"
            ],
        }
    return {
        "status": "skipped",
        "reason": "source path was not accessible for a local size/availability probe",
        "notes": [
            "ingest_lerobot completed, but the source itself was not readable as a "
            "local path from this process"
        ],
    }


def _normalize_formats(formats: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in formats:
        fmt = raw.strip().lower()
        if fmt in {"enterprise", "remote-lance", "lancedb-enterprise"}:
            fmt = ENTERPRISE_LANCE_FORMAT
        if fmt == "lerobot":
            fmt = LEROBOT_DEFAULT_FORMAT
        if fmt in {"lerobot-v3", "official-lerobot"}:
            fmt = LEROBOT_NATIVE_FORMAT
        if fmt in {"apache-iceberg", "iceberg-table"}:
            fmt = ICEBERG_FORMAT
        if fmt in {"parquet-table", "analytics-parquet"}:
            fmt = PARQUET_FORMAT
        if fmt not in SUPPORTED_BENCHMARK_FORMATS:
            raise BenchmarkError(
                f"unknown benchmark format {raw!r}; choose from "
                f"{', '.join(DEFAULT_BENCHMARK_FORMATS + (LEROBOT_NATIVE_FORMAT,) + ANALYTICS_BENCHMARK_FORMATS)}"
            )
        if fmt not in normalized:
            normalized.append(fmt)
    if not normalized:
        raise BenchmarkError("at least one benchmark format is required")
    return tuple(normalized)


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise BenchmarkError(f"{name} must be positive")
    return parsed


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _normalize_lerobot_native_source_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in LEROBOT_NATIVE_SOURCE_MODES:
        raise BenchmarkError(
            "lerobot_native_source_mode must be one of "
            f"{', '.join(sorted(LEROBOT_NATIVE_SOURCE_MODES))}"
        )
    return mode


def _normalize_lerobot_native_cache_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in LEROBOT_NATIVE_CACHE_MODES:
        raise BenchmarkError(
            "lerobot_native_cache_mode must be one of "
            f"{', '.join(sorted(LEROBOT_NATIVE_CACHE_MODES))}"
        )
    return mode


def _snapshot_metadata(lake: Lake, snapshot_name: str) -> dict[str, Any]:
    rows = [
        row
        for row in lake.table("dataset_snapshots").to_arrow().to_pylist()
        if row["name"] == snapshot_name
    ]
    if not rows:
        raise BenchmarkError(f"no snapshot named {snapshot_name!r} in {lake.uri}")
    row = max(rows, key=lambda item: (item["created_at"], item["dataset_id"]))
    query_spec = json.loads(row["query_spec"] or "{}")
    split = json.loads(row["split"] or "{}")
    scenario_ids = tuple(sorted(str(sid) for sid in query_spec.get("scenario_ids", [])))
    return {
        "row": row,
        "dataset_id": row["dataset_id"],
        "snapshot_name": row["name"],
        "tag": row["tag"],
        "kind": row["kind"],
        "query_spec": query_spec,
        "scenario_ids": scenario_ids,
        "split": split,
        "table_versions": [
            {
                "table": tv["table"],
                "version": int(tv["version"]),
                "tag": tv.get("tag") or "",
            }
            for tv in row["table_versions"]
        ],
    }


def _dataset_report(
    lake: Lake,
    snapshot: dict[str, Any],
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    return {
        "lake_uri": lake.uri,
        "dataset_id": snapshot["dataset_id"],
        "snapshot_name": snapshot["snapshot_name"],
        "tag": snapshot["tag"],
        "kind": snapshot["kind"],
        "scenario_count": len(snapshot["scenario_ids"]),
        "split": snapshot["split"],
        "table_versions": snapshot["table_versions"],
        "source_corpus": _source_corpus_descriptor(lake, snapshot, config),
    }


def _storage_tier_report(lake: Lake, config: BenchmarkRunConfig) -> dict[str, Any]:
    active = str(config.storage_tier or "local")
    tiers: list[dict[str, Any]] = [
        {
            "tier": active,
            "status": "completed",
            "lake_uri": lake.uri,
            "note": "Benchmark metrics in this report were measured on this storage tier.",
        }
    ]
    for expected in ("local", "object-store"):
        if expected == active:
            continue
        tiers.append(
            {
                "tier": expected,
                "status": "skipped",
                "skip_reason": (
                    f"{expected} was not requested for this run; rerun bench run "
                    f"with --storage-tier {expected} and a matching lake URI."
                ),
            }
        )
    return {
        "active": active,
        "tiers": tiers,
    }


def _comparison_table(
    formats: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fmt, result in formats.items():
        metrics = result.get("metrics") or {}
        row: dict[str, Any] = {
            "format": fmt,
            "status": result.get("status"),
            "skip_reason": result.get("skip_reason"),
        }
        for metric in BENCHMARK_METRICS:
            measurement = metrics.get(metric) or {}
            row[metric] = measurement.get("value")
            row[f"{metric}_unit"] = measurement.get("unit")
        rows.append(row)
    return rows


def _run_lance_format(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    try:
        dataset = lake.training.dataset(snapshot["snapshot_name"])
    except TrainingError as exc:
        raise BenchmarkError(f"cannot build Lance training dataset: {exc}") from exc

    throughput = _measure_lance_throughput(dataset, config, hardware)
    random_access = _measure_lance_random_access(dataset, config)
    random_frame_sampling = _measure_lance_random_frame_sampling(dataset, config)
    curation = _measure_lance_curation(lake, snapshot, config)
    storage = _measure_lance_storage(lake, dataset, throughput)

    return {
        "status": "completed",
        "format": LANCE_FORMAT,
        "metrics": {
            METRIC_DATALOADER_THROUGHPUT: throughput,
            METRIC_RANDOM_ACCESS_LATENCY: random_access,
            METRIC_RANDOM_FRAME_SAMPLING: random_frame_sampling,
            METRIC_QUERY_TO_DATASET_CURATION: curation,
            METRIC_STORAGE_FOOTPRINT: storage,
        },
        "notes": [
            "Lance path uses lake.training.dataset over the pinned snapshot.",
            "Payload bytes are materialized only through requested training samples.",
        ],
        "dataset_manifest": dataset.manifest.to_dict(),
    }


def _run_enterprise_lance_format(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    enterprise_lake, endpoint, skip_reason = _enterprise_benchmark_lake(lake, config)
    if enterprise_lake is None:
        return _skip_enterprise_lance(skip_reason)

    is_fake = endpoint.get("profile") == FAKE_LOCAL_DB_PROFILE
    degraded_capabilities = frozenset(endpoint.get("degraded_capabilities") or ())
    cache_metrics_available = is_fake or "plan_executor_cache_metrics" not in degraded_capabilities
    prewarm_available = is_fake or "page_cache_prewarm" not in degraded_capabilities

    try:
        cold = _run_enterprise_phase(
            enterprise_lake,
            snapshot,
            config=config,
            hardware=hardware,
            phase="cold_cache",
            epoch=0,
            cache_policy="lazy",
            cache_metrics=(
                _enterprise_cache_metrics(hits=0, misses=config.sample_limit) if is_fake else None
            ),
            cache_metrics_available=cache_metrics_available,
        )
        prewarmed = _run_enterprise_phase(
            enterprise_lake,
            snapshot,
            config=config,
            hardware=hardware,
            phase="prewarmed",
            epoch=0,
            cache_policy="epoch",
            prewarm=is_fake,
            cache_metrics_available=cache_metrics_available,
            prewarm_available=prewarm_available,
        )
        warm = _run_enterprise_phase(
            enterprise_lake,
            snapshot,
            config=config,
            hardware=hardware,
            phase="warm_second_epoch",
            epoch=1,
            cache_policy="lazy",
            cache_metrics=(
                _enterprise_cache_metrics(hits=config.sample_limit, misses=0) if is_fake else None
            ),
            cache_metrics_available=cache_metrics_available,
        )
        filter_change = _measure_enterprise_filter_change(
            enterprise_lake,
            snapshot,
            base_row_plan_id=str(cold["loader_report"]["plans"]["row_plan_id"]),
            base_row_count=int(cold["loader_report"]["plans"].get("selected_rows") or 0),
            config=config,
        )
    except TrainingError as exc:
        return _skip_enterprise_lance(f"cannot run Enterprise benchmark path: {exc}")

    phases = {
        "cold_cache": cold,
        "prewarmed": prewarmed,
        "warm_second_epoch": warm,
    }
    degraded_phases = sorted(name for name, phase in phases.items() if phase["status"] == "degraded")
    return {
        "status": "completed",
        "format": ENTERPRISE_LANCE_FORMAT,
        "remote_endpoint": endpoint,
        "confidence": endpoint.get("confidence"),
        "degraded_phases": degraded_phases,
        "metrics": {
            "query_to_first_batch_latency": cold["metrics"][
                "query_to_first_batch_latency"
            ],
            "shuffled_epoch_throughput": warm["metrics"]["shuffled_epoch_throughput"],
            METRIC_RANDOM_ACCESS_LATENCY: warm["metrics"][METRIC_RANDOM_ACCESS_LATENCY],
            "subset_filter_change": filter_change,
            METRIC_STORAGE_FOOTPRINT: _measure_enterprise_storage(lake, phases),
        },
        "phases": phases,
        "notes": [
            "Enterprise path uses the native training loader with backend='enterprise'.",
            "Fake-local db:// fixtures reuse local LanceDB tables with an Enterprise connection report.",
            "No training copy is materialized for filter changes or epoch reruns.",
            (
                "fake-local-db cache/prewarm numbers are synthesized to prove report shape; "
                "live-db/live-namespace profiles never synthesize cache metrics -- missing "
                "endpoint telemetry is reported as a degraded phase, not assumed zero."
            ),
        ],
        "dataset_manifest": warm["dataset_manifest"],
    }


def _enterprise_live_profile(connection_kind: str) -> str:
    """Map a live connection kind to its report profile label."""
    if connection_kind in ENTERPRISE_NAMESPACE_CONNECTION_KINDS:
        return LIVE_NAMESPACE_PROFILE
    return LIVE_DB_PROFILE


def _validate_enterprise_capabilities(lake: Lake) -> tuple[list[dict[str, Any]], str | None]:
    """Preflight capability negotiation for a live Enterprise endpoint.

    Delegates to the same capability matrix the real training loader uses
    (:func:`training._enterprise_training_capabilities` over
    :func:`training._lake_capabilities_dict`) so this preflight can never drift
    from what a subsequent ``lake.training.dataset(..., backend="enterprise")``
    call will actually see -- including any ``lake.enterprise_training_capabilities``
    override the caller has installed.

    Returns ``(checks, blocking_reason)``. ``blocking_reason`` is set only when the
    endpoint lacks the one capability the benchmark cannot run without at all
    (remote scan); every other missing capability degrades only the specific
    phases that depend on it instead of failing the whole format.
    """
    spec = getattr(lake, "connection_spec", None)
    connection_kind = str(getattr(spec, "kind", "local_path") or "local_path")
    base_capabilities = _training._lake_capabilities_dict(lake)
    matrix = _training._enterprise_training_capabilities(
        lake,
        spec,
        base_capabilities,
        connection_kind=connection_kind,
        resolved_backend="enterprise",
        is_enterprise_connection=True,
    )

    checks: list[dict[str, Any]] = []
    required_available = bool(matrix.get(_ENTERPRISE_REQUIRED_CAPABILITY))
    checks.append(
        {
            "name": _ENTERPRISE_REQUIRED_CAPABILITY,
            "required": True,
            "available": required_available,
            "status": "passed" if required_available else "unsupported",
        }
    )
    blocking_reason = None
    if not required_available:
        blocking_reason = (
            "live Enterprise endpoint lacks required capability "
            f"{_ENTERPRISE_REQUIRED_CAPABILITY!r}; cannot run the enterprise-lance benchmark "
            "against this endpoint"
        )
    for name in _ENTERPRISE_OPTIONAL_CAPABILITIES:
        available = bool(matrix.get(name))
        checks.append(
            {
                "name": name,
                "required": False,
                "available": available,
                "status": "passed" if available else "degraded",
            }
        )
    return checks, blocking_reason


def _enterprise_benchmark_lake(
    lake: Lake,
    config: BenchmarkRunConfig,
) -> tuple[Lake | None, dict[str, Any], str]:
    spec = getattr(lake, "connection_spec", None)
    connection_kind = str(getattr(spec, "kind", "local_path") or "local_path")
    is_live_connection = connection_kind in ENTERPRISE_BENCHMARK_CONNECTION_KINDS

    if config.enterprise_fixture_uri:
        if config.enterprise_live:
            raise BenchmarkError(
                "enterprise_live=True cannot be combined with enterprise_fixture_uri; "
                "open a live Enterprise db:// or namespace lake instead of requesting the "
                "fake-local fixture"
            )
        fixture = _fake_enterprise_lake(lake, config)
        endpoint = _enterprise_endpoint_report(fixture, profile=FAKE_LOCAL_DB_PROFILE)
        return fixture, endpoint, ""

    if config.enterprise_live and not is_live_connection:
        raise BenchmarkError(
            "enterprise_live=True requires a lake opened with a live Enterprise db:// or "
            f"namespace connection; got connection_kind={connection_kind!r}. Open the lake "
            "with Lake.open('db://...') or a namespace URI, or omit enterprise_live to run "
            "the fake-local db:// fixture instead."
        )

    if not is_live_connection:
        return (
            None,
            {},
            "enterprise-lance requires an Enterprise db:// or namespace-backed lake; pass "
            "enterprise_fixture_uri to run the fake-local db:// benchmark fixture, or open a "
            "live Enterprise lake and pass enterprise_live=True",
        )

    profile = _enterprise_live_profile(connection_kind)
    checks, blocking_reason = _validate_enterprise_capabilities(lake)
    if blocking_reason:
        return None, {}, blocking_reason
    endpoint = _enterprise_endpoint_report(lake, profile=profile, capability_checks=checks)
    return lake, endpoint, ""


def _fake_enterprise_lake(lake: Lake, config: BenchmarkRunConfig) -> Lake:
    uri = str(config.enterprise_fixture_uri)
    host_override = config.enterprise_host_override or "https://enterprise-benchmark.local"
    spec = LakeConnectionSpec(
        kind="lancedb_remote_db",
        uri=uri,
        display_uri=uri,
        lancedb_connect_kwargs={
            "region": config.enterprise_region,
            "host_override": host_override,
        },
        auth_refs={
            "remote": "benchmark-fixture",
            "namespace": None,
            "storage": None,
            "source": None,
        },
        direct_object_io_allowed=False,
        capabilities=LakeCapabilities(
            server_side_query=True,
            direct_object_io=False,
            namespace_resolution=False,
            geneva_worker_specs=False,
            blob_fetch_remote=True,
        ),
    )
    return Lake(lake._db, uri, connection_spec=spec)  # noqa: SLF001 - benchmark fixture wraps the same DB.


def _enterprise_endpoint_report(
    lake: Lake,
    *,
    profile: str,
    capability_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec = getattr(lake, "connection_spec", None)
    kwargs = dict(getattr(spec, "lancedb_connect_kwargs", {}) or {})
    capabilities = getattr(spec, "capabilities", None)
    capability_dict = capabilities.__dict__ if capabilities is not None else {}
    checks = list(capability_checks or [])
    degraded_capabilities = sorted(
        str(check["name"]) for check in checks if check.get("status") == "degraded"
    )
    # Only region/host_override/cache_policy are read out of the connect kwargs --
    # never the raw mapping -- so an api_key or other secret in
    # ``lancedb_connect_kwargs`` can never reach the benchmark report.
    return {
        "uri": str(getattr(spec, "display_uri", lake.uri) or lake.uri),
        "profile": profile,
        "confidence": ENTERPRISE_BENCHMARK_CONFIDENCE.get(profile, "unknown"),
        "connection_kind": str(getattr(spec, "kind", "local_path") or "local_path"),
        "region": kwargs.get("region"),
        "host_override": kwargs.get("host_override"),
        "cache_policy": kwargs.get("cache_policy"),
        "capabilities": dict(capability_dict),
        "capability_checks": checks,
        "degraded_capabilities": degraded_capabilities,
        "software_versions": {
            "lancedb-robotics": __version__,
            "lancedb": _package_version("lancedb"),
            "pylance": _package_version("pylance"),
        },
        "hardware_class": os.environ.get(ENTERPRISE_INSTANCE_CLASS_ENV),
    }


def _run_enterprise_phase(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    hardware: dict[str, Any],
    phase: str,
    epoch: int,
    cache_policy: str,
    prewarm: bool = False,
    cache_metrics: Mapping[str, Any] | None = None,
    cache_metrics_available: bool = True,
    prewarm_available: bool = True,
) -> dict[str, Any]:
    if cache_metrics is not None:
        lake.plan_executor_cache_metrics = cache_metrics
    if prewarm:
        lake.plan_executor_prewarm = _enterprise_prewarm_hook(lake)

    start_ns = time.perf_counter_ns()
    dataset = lake.training.dataset(
        snapshot["snapshot_name"],
        columns=["observation_id", "payload", "payload_size"],
        media="bytes",
        backend="enterprise",
        cache_policy=cache_policy,
        shuffle=True,
        shuffle_seed=config.seed,
        epoch=epoch,
        prewarm_options={"wait": True},
        # A live endpoint that is missing an optional capability (cache metrics,
        # prewarm) should degrade with a reported fallback event, not hard-fail
        # the whole benchmark phase; the preflight capability_checks already
        # decided what "degraded" means for this run.
        fallback="warn",
    )
    setup_seconds = _elapsed_seconds(start_ns)
    first_batch = _measure_enterprise_first_batch(dataset, config, setup_seconds)
    throughput = _measure_lance_throughput(dataset, config, hardware)
    random_access = _measure_lance_random_access(dataset, config)
    report = dataset.loader_report(
        training_run_id=f"benchmark-{phase}",
        extra={"benchmark_phase": phase},
    ).to_dict()
    cache = report["metrics"]["cache"]
    prewarm_report = _enterprise_prewarm_report(dataset, report)

    degraded_reasons: list[str] = []
    if cache_metrics_available:
        cache_block = {
            "hits": int(cache.get("hits") or 0),
            "misses": int(cache.get("misses") or 0),
            "by_plan_executor": cache.get("by_plan_executor") or {},
            "by_operation": cache.get("by_operation") or {},
        }
    else:
        degraded_reasons.append(
            "plan_executor_cache_metrics capability unavailable on this endpoint; "
            "hits/misses are unavailable, not zero -- rely on latency deltas instead"
        )
        cache_block = {"hits": None, "misses": None, "by_plan_executor": {}, "by_operation": {}}
    if phase == "prewarmed" and not prewarm_available:
        degraded_reasons.append(
            "page_cache_prewarm capability unavailable on this endpoint; the prewarm "
            "request was rejected or degraded to lazy-cache"
        )

    return {
        "status": "degraded" if degraded_reasons else "completed",
        "degraded_reasons": degraded_reasons,
        "phase": phase,
        "metrics": {
            "query_to_first_batch_latency": first_batch,
            "shuffled_epoch_throughput": throughput,
            METRIC_RANDOM_ACCESS_LATENCY: random_access,
        },
        "cache": cache_block,
        "prewarm": prewarm_report,
        "loader_report": report,
        "dataset_manifest": dataset.manifest.to_dict(),
    }


def _measure_enterprise_first_batch(
    dataset: Any,
    config: BenchmarkRunConfig,
    setup_seconds: float,
) -> dict[str, Any]:
    count = min(len(dataset), config.sample_limit)
    indices = list(range(count))
    start_ns = time.perf_counter_ns()
    samples = dataset.__getitems__(indices) if indices else []
    first_batch_seconds = _elapsed_seconds(start_ns)
    payload_bytes = sum(int(sample.get("payload_size") or 0) for sample in samples)
    return {
        "status": "completed",
        "value": (setup_seconds + first_batch_seconds) * 1000.0,
        "unit": "ms",
        "details": {
            "dataset_setup_seconds": setup_seconds,
            "first_batch_seconds": first_batch_seconds,
            "samples": count,
            "payload_bytes_materialized": payload_bytes,
        },
    }


def _measure_enterprise_filter_change(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    base_row_plan_id: str,
    base_row_count: int,
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    filters = {"modality": "image"}
    start_ns = time.perf_counter_ns()
    filtered = lake.training.dataset(
        snapshot["snapshot_name"],
        columns=["observation_id", "modality"],
        filters=filters,
        media="metadata",
        backend="enterprise",
        cache_policy="none",
        shuffle=True,
        shuffle_seed=config.seed,
        fallback="warn",
    )
    duration = _elapsed_seconds(start_ns)
    return {
        "status": "completed",
        "value": duration * 1000.0,
        "unit": "ms",
        "details": {
            "operation": "change_subset_filter",
            "filters": filters,
            "base_snapshot": snapshot["snapshot_name"],
            "base_row_plan_id": base_row_plan_id,
            "filtered_row_plan_id": filtered.row_plan.plan_id,
            "base_rows": base_row_count,
            "filtered_rows": len(filtered),
            "materialized_bytes_written": 0,
        },
    }


def _measure_enterprise_storage(
    source_lake: Lake,
    phases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_stored_bytes = _path_size(source_lake.uri)
    payload_bytes_hydrated = 0
    rows_hydrated = 0
    cache_hits = 0
    cache_misses = 0
    cache_metrics_available = True
    for phase in phases.values():
        report = phase.get("loader_report") or {}
        metrics = report.get("metrics") or {}
        summary = metrics.get("summary") or {}
        payload_bytes_hydrated += int(summary.get("bytes_read") or 0)
        rows_hydrated += int(summary.get("rows_returned") or 0)
        # Read the phase's already-corrected cache block, not the raw loader
        # report, so a degraded phase contributes "unavailable" rather than a
        # misleading zero to the rollup.
        cache = phase.get("cache") or {}
        if cache.get("hits") is None and cache.get("misses") is None:
            cache_metrics_available = False
            continue
        cache_hits += int(cache.get("hits") or 0)
        cache_misses += int(cache.get("misses") or 0)
    return {
        "status": "completed",
        "value": int(source_stored_bytes or 0),
        "unit": "bytes",
        "details": {
            "source_stored_bytes": source_stored_bytes,
            "materialized_bytes_written": 0,
            "payload_bytes_hydrated": payload_bytes_hydrated,
            "rows_hydrated": rows_hydrated,
            "cache_hits": cache_hits if cache_metrics_available else None,
            "cache_misses": cache_misses if cache_metrics_available else None,
            "fixed_quality": "source payload bytes; no training copy or transcoding",
        },
    }


def _enterprise_prewarm_hook(lake: Lake):
    def prewarm(request: Mapping[str, Any]) -> dict[str, Any]:
        row_count = int(request.get("row_count") or 0)
        projected = list(request.get("projected_columns") or [])
        lake.plan_executor_cache_metrics = _enterprise_cache_metrics(
            hits=max(row_count, 1),
            misses=0,
        )
        return {
            "status": "complete",
            "prewarm_id": request.get("prewarm_id"),
            "completed_executors": 1,
            "failed_executors": 0,
            "pe_fanout": 1,
            "cache_hits": row_count,
            "cache_misses": 0,
            "warm_bytes": row_count * max(len(projected), 1) * 8,
            "cold_bytes": 0,
            "duration_ms": 1.0,
        }

    return prewarm


def _enterprise_prewarm_report(dataset: Any, loader_report: Mapping[str, Any]) -> dict[str, Any]:
    cache_state = dataset.manifest.backend.get("cache") or {}
    metrics = loader_report.get("metrics") or {}
    cache = metrics.get("cache") or {}
    prewarm = cache.get("prewarm") or {}
    return {
        "requested": bool(cache_state.get("prewarm_requested")),
        "executed": bool(cache_state.get("prewarm_executed")),
        "status": cache_state.get("prewarm_status"),
        "prewarm_id": cache_state.get("prewarm_id"),
        "hits": int(prewarm.get("hits") or 0),
        "misses": int(prewarm.get("misses") or 0),
    }


def _enterprise_cache_metrics(*, hits: int, misses: int) -> dict[str, Any]:
    return {
        "remote_take": {
            "per_addr": {
                "pe-benchmark": {
                    "hits": int(hits),
                    "misses": int(misses),
                }
            }
        }
    }


def _measure_lance_throughput(
    dataset: Any,
    config: BenchmarkRunConfig,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    count = min(len(dataset), config.sample_limit)
    start_ns = time.perf_counter_ns()
    payload_bytes = 0
    for index in range(count):
        sample = dataset[index]
        payload_bytes += int(sample.get("payload_size") or 0)
    duration = _elapsed_seconds(start_ns)
    return {
        "status": "completed",
        "value": _rate(count, duration),
        "unit": "samples/s",
        "details": {
            "samples": count,
            "duration_seconds": duration,
            "payload_bytes_materialized": payload_bytes,
            "gpu_utilization_pct": hardware["gpu"].get("utilization_pct"),
            "columns": list(dataset.columns),
        },
    }


def _measure_lance_random_access(dataset: Any, config: BenchmarkRunConfig) -> dict[str, Any]:
    indices = _random_indices(len(dataset), config.random_access_samples, config.seed)
    latencies_ms: list[float] = []
    for index in indices:
        start_ns = time.perf_counter_ns()
        dataset[index]
        latencies_ms.append(_elapsed_seconds(start_ns) * 1000.0)
    return {
        "status": "completed",
        "value": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "unit": "ms/sample",
        "details": {
            "samples": len(indices),
            "indices": indices[:16],
            "mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50_ms": _percentile(latencies_ms, 0.50),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "max_ms": max(latencies_ms) if latencies_ms else 0.0,
        },
    }


def _measure_lance_random_frame_sampling(dataset: Any, config: BenchmarkRunConfig) -> dict[str, Any]:
    plans = _random_frame_plans(
        len(dataset),
        clips=config.random_frame_samples,
        frames_per_clip=config.frames_per_clip,
        seed=config.seed,
    )
    latencies_ms: list[float] = []
    payload_bytes = 0
    for plan in plans:
        for index in plan:
            start_ns = time.perf_counter_ns()
            sample = dataset[index]
            latencies_ms.append(_elapsed_seconds(start_ns) * 1000.0)
            payload_bytes += int(sample.get("payload_size") or 0)
    return _random_frame_metric(
        plans,
        latencies_ms,
        payload_bytes=payload_bytes,
        access_path="lance-native-training-dataset",
    )


def _measure_lance_curation(
    lake: Lake,
    snapshot: dict[str, Any],
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    scenario_ids = list(snapshot["scenario_ids"][: config.query_limit])
    if not scenario_ids:
        raise BenchmarkError("benchmark snapshot contains no scenarios to curate")
    name = f"bench-{_slug(snapshot['snapshot_name'])}-{config.seed}"
    start_ns = time.perf_counter_ns()
    try:
        manifest = create_snapshot(
            lake,
            name=name,
            scenario_ids=scenario_ids,
            source={
                "kind": "benchmark-query",
                "base_snapshot": snapshot["snapshot_name"],
                "query_limit": config.query_limit,
                "seed": config.seed,
                "method": "deterministic-scenario-id-selection",
            },
            split_by=snapshot["split"].get("by", "run"),
            tag=f"{snapshot['tag']}-benchmark",
            created_by="lancedb-robotics-bench",
        )
    except DatasetError as exc:
        raise BenchmarkError(f"cannot create benchmark curation snapshot: {exc}") from exc
    duration = _elapsed_seconds(start_ns)
    return {
        "status": "completed",
        "value": duration * 1000.0,
        "unit": "ms",
        "details": {
            "base_snapshot": snapshot["snapshot_name"],
            "curated_snapshot_name": manifest.name,
            "curated_dataset_id": manifest.dataset_id,
            "transform_id": manifest.transform_id,
            "selected_scenarios": len(manifest.scenario_ids),
            "table_versions": [
                {"table": table, "version": int(version)}
                for table, version in manifest.table_versions
            ],
        },
    }


def _measure_lance_storage(
    lake: Lake,
    dataset: Any,
    throughput: dict[str, Any],
) -> dict[str, Any]:
    stored_bytes = _path_size(lake.uri)
    payload_bytes = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        payload_bytes += int(sample.get("payload_size") or 0)
    video_rows = lake.table("video_encodings").to_arrow().to_pylist()
    encoded_video_bytes = sum(int(row.get("encoded_size_bytes") or 0) for row in video_rows)
    value = stored_bytes if stored_bytes is not None else payload_bytes + encoded_video_bytes
    return {
        "status": "completed",
        "value": int(value),
        "unit": "bytes",
        "details": {
            "stored_bytes": stored_bytes,
            "materialized_bytes_written": 0,
            "selected_payload_bytes": payload_bytes,
            "payload_bytes_hydrated": throughput["details"][
                "payload_bytes_materialized"
            ],
            "encoded_video_bytes": encoded_video_bytes,
            "sample_payload_bytes_materialized": throughput["details"][
                "payload_bytes_materialized"
            ],
            "fixed_quality": "source payload bytes; no transcoding or quality reduction",
        },
    }


def _run_lerobot_default_format(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    artifact_root: Path | None,
) -> dict[str, Any]:
    if artifact_root is None:
        with tempfile.TemporaryDirectory(prefix="lancedb-robotics-bench-") as tmp:
            return _run_lerobot_default_in_dir(
                lake,
                snapshot,
                config=config,
                out_dir=Path(tmp) / LEROBOT_DEFAULT_FORMAT,
                ephemeral=True,
            )
    return _run_lerobot_default_in_dir(
        lake,
        snapshot,
        config=config,
        out_dir=artifact_root / LEROBOT_DEFAULT_FORMAT,
        ephemeral=False,
    )


def _run_lerobot_default_in_dir(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    out_dir: Path,
    ephemeral: bool,
) -> dict[str, Any]:
    try:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        start_export_ns = time.perf_counter_ns()
        manifest = export_dataset_snapshot(
            lake,
            snapshot["snapshot_name"],
            out_dir=out_dir,
            fmt="lerobot",
            created_by="lancedb-robotics-bench",
        )
        export_seconds = _elapsed_seconds(start_export_ns)
    except DatasetExportError as exc:
        raise BenchmarkError(f"cannot export LeRobot-default projection: {exc}") from exc

    rows = _read_lerobot_rows(out_dir)
    throughput = _measure_rows_throughput(rows, config)
    random_access = _measure_rows_random_access(rows, config)
    random_frame_sampling = _measure_rows_random_frame_sampling(rows, config)
    storage = _measure_projection_storage(out_dir, manifest.data_files)
    curation = {
        "status": "completed",
        "value": export_seconds * 1000.0,
        "unit": "ms",
        "details": {
            "operation": "materialize_lerobot_projection",
            "snapshot_name": manifest.snapshot_name,
            "dataset_id": manifest.dataset_id,
            "transform_id": manifest.transform_id,
            "episode_count": manifest.episode_count,
            "step_count": manifest.step_count,
        },
    }

    return {
        "status": "completed",
        "format": LEROBOT_DEFAULT_FORMAT,
        "metrics": {
            METRIC_DATALOADER_THROUGHPUT: throughput,
            METRIC_RANDOM_ACCESS_LATENCY: random_access,
            METRIC_RANDOM_FRAME_SAMPLING: random_frame_sampling,
            METRIC_QUERY_TO_DATASET_CURATION: curation,
            METRIC_STORAGE_FOOTPRINT: storage,
        },
        "notes": [
            "LeRobot-default comparison uses the repo's deterministic projection writer.",
            "The native LeRobot package is optional; parquet projection rows are read directly.",
        ],
        "projection_manifest": {
            **manifest.to_dict(),
            "manifest_path": None
            if ephemeral
            else str(out_dir / DATASET_EXPORT_MANIFEST_FILENAME),
        },
    }


def _run_lerobot_native_format(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    artifact_root: Path | None,
) -> dict[str, Any]:
    status = _lerobot_native_dependency_status()
    if not status["available"]:
        return _skip_lerobot_native(_lerobot_native_skip_reason(status), status=status)

    components, import_error = _load_lerobot_native_components()
    if components is None:
        import_status = {
            **status,
            "available": False,
            "import_error": import_error,
        }
        return _skip_lerobot_native(
            f"official LeRobot native loader could not be imported: {import_error}",
            status=import_status,
        )

    if artifact_root is None:
        with tempfile.TemporaryDirectory(prefix="lancedb-robotics-native-lerobot-") as tmp:
            return _run_lerobot_native_with_components(
                lake,
                snapshot,
                config=config,
                artifact_root=Path(tmp),
                components=components,
                dependency_status=status,
                ephemeral=True,
            )
    return _run_lerobot_native_with_components(
        lake,
        snapshot,
        config=config,
        artifact_root=artifact_root,
        components=components,
        dependency_status=status,
        ephemeral=False,
    )


def _run_lerobot_native_with_components(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    artifact_root: Path,
    components: _LeRobotNativeComponents,
    dependency_status: Mapping[str, Any],
    ephemeral: bool,
) -> dict[str, Any]:
    source, export_manifest, materialization_ms = _lerobot_native_source(
        lake,
        snapshot,
        config=config,
        artifact_root=artifact_root,
    )
    if source.get("status") == "skipped":
        return _skip_lerobot_native(
            str(source.get("skip_reason") or "LeRobot-native source could not be resolved"),
            status=dependency_status,
            source=source,
            projection_manifest=export_manifest,
        )

    start_open_ns = time.perf_counter_ns()
    try:
        dataset = _open_lerobot_native_dataset(
            components.dataset_cls,
            source,
            dependency_status=dependency_status,
        )
    except Exception as exc:  # pragma: no cover - exercised by optional stacks.
        return _skip_lerobot_native(
            f"official LeRobot native loader could not open the benchmark source: {exc}",
            status=dependency_status,
            source=source,
            projection_manifest=export_manifest,
        )
    open_ms = _elapsed_seconds(start_open_ns) * 1000.0

    try:
        throughput = _measure_lerobot_native_throughput(
            dataset,
            components.dataloader_cls,
            config,
        )
        random_access = _measure_lerobot_native_random_access(dataset, config)
        random_frame_sampling = _measure_lerobot_native_random_frame_sampling(dataset, config)
    except Exception as exc:  # pragma: no cover - exercised by optional stacks.
        return _skip_lerobot_native(
            f"official LeRobot native loader failed during sample access: {exc}",
            status=dependency_status,
            source=source,
            projection_manifest=export_manifest,
        )

    curation = {
        "status": "completed",
        "value": materialization_ms + open_ms,
        "unit": "ms",
        "details": {
            "operation": source["operation"],
            "source": source,
            "materialization_ms": materialization_ms,
            "loader_open_ms": open_ms,
            "projection_manifest": export_manifest,
        },
    }
    storage = _measure_lerobot_native_storage(
        dataset,
        source=source,
        projection_manifest=export_manifest,
        ephemeral=ephemeral,
        throughput=throughput,
    )
    dataset_report = _lerobot_native_dataset_report(dataset)
    return {
        "status": "completed",
        "format": LEROBOT_NATIVE_FORMAT,
        "metrics": {
            METRIC_DATALOADER_THROUGHPUT: throughput,
            METRIC_RANDOM_ACCESS_LATENCY: random_access,
            METRIC_RANDOM_FRAME_SAMPLING: random_frame_sampling,
            METRIC_QUERY_TO_DATASET_CURATION: curation,
            METRIC_STORAGE_FOOTPRINT: storage,
        },
        "native_loader": {
            "official_api": "lerobot.datasets.LeRobotDataset",
            "resolved_api": components.dataset_api,
            "dependency_status": dict(dependency_status),
            "source": source,
            "dataset": dataset_report,
        },
        "projection_manifest": export_manifest,
        "notes": [
            "LeRobot-native comparison opens the benchmark corpus through the official LeRobotDataset API.",
            "The base install stays dependency-light; this arm skips unless the optional LeRobot/Torch/decode stack is available.",
        ],
    }


def _lerobot_native_source(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, float]:
    source_mode = config.lerobot_native_source_mode
    source = None if source_mode == "projection" else _source_lerobot_native_dataset(config)
    if source is not None:
        source = _with_lerobot_native_resolver_report(source, config)
        if config.lerobot_native_cache_mode == "cache-only" and not source.get("root"):
            reason = (
                "lerobot-native cache-only mode requires source_dataset_uri to be an "
                "existing local LeRobot root or prepared cache path; refusing implicit "
                "HF Hub download"
            )
            return _unresolved_lerobot_native_source(reason, config, source=source), None, 0.0
        return source, None, 0.0

    if source_mode == "source":
        reason = (
            "lerobot-native source mode requires a local source_dataset_uri path, "
            "hf:// source URI, or HF-style source_dataset_id; projection fallback is disabled"
        )
        return _unresolved_lerobot_native_source(reason, config), None, 0.0

    out_dir = artifact_root / LEROBOT_NATIVE_FORMAT / "projection"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    start_export_ns = time.perf_counter_ns()
    try:
        manifest = export_dataset_snapshot(
            lake,
            snapshot["snapshot_name"],
            out_dir=out_dir,
            fmt="lerobot",
            created_by="lancedb-robotics-bench",
        )
    except DatasetExportError as exc:
        raise BenchmarkError(f"cannot export LeRobot-native projection: {exc}") from exc
    materialization_ms = _elapsed_seconds(start_export_ns) * 1000.0
    source = {
        "kind": "projection",
        "operation": "materialize_lerobot_projection_then_open_official_loader",
        "repo_id": manifest.dataset_id,
        "root": str(out_dir),
        "revision": None,
        "download_videos": False,
        "source_uri": str(out_dir),
    }
    return _with_lerobot_native_resolver_report(source, config), manifest.to_dict(), materialization_ms


def _source_lerobot_native_dataset(config: BenchmarkRunConfig) -> dict[str, Any] | None:
    repo_id = config.source_dataset_id
    revision = config.source_dataset_revision
    source_uri_value = config.source_dataset_uri
    root: str | None = None

    if source_uri_value:
        value = str(source_uri_value)
        if value.startswith("hf://"):
            repo_id = value.removeprefix("hf://")
        else:
            path = Path(value)
            if path.exists():
                root = str(path)
                repo_id = repo_id or path.name

    if root and repo_id:
        return {
            "kind": "source-corpus",
            "operation": "open_official_lerobot_loader",
            "repo_id": str(repo_id),
            "root": root,
            "revision": revision,
            "download_videos": False,
            "source_uri": source_uri_value,
        }

    if repo_id and _looks_like_hf_source(str(repo_id)):
        repo_value = str(repo_id)
        if "@" in repo_value:
            repo_value, embedded_revision = repo_value.rsplit("@", maxsplit=1)
            revision = revision or embedded_revision
        return {
            "kind": "source-corpus",
            "operation": "open_official_lerobot_loader",
            "repo_id": repo_value,
            "root": root,
            "revision": revision,
            "download_videos": True,
            "source_uri": source_uri_value or f"hf://{repo_value}",
        }
    return None


def _with_lerobot_native_resolver_report(
    source: Mapping[str, Any],
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    episode_filter = _lerobot_native_episode_filter(config)
    resolved = {
        **dict(source),
        "resolver": _lerobot_native_resolver_report(config, source=source),
        "preflight": _lerobot_native_preflight_report(config, source=source),
        "episode_filter": episode_filter,
    }
    episodes = episode_filter.get("episodes")
    if episodes is not None:
        resolved["episodes"] = list(episodes)
    return resolved


def _unresolved_lerobot_native_source(
    reason: str,
    config: BenchmarkRunConfig,
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_payload = dict(source or {})
    return {
        **source_payload,
        "status": "skipped",
        "kind": source_payload.get("kind") or "unresolved-source",
        "operation": "resolve_official_lerobot_loader_source",
        "skip_reason": reason,
        "repo_id": source_payload.get("repo_id") or config.source_dataset_id,
        "root": source_payload.get("root"),
        "revision": source_payload.get("revision") or config.source_dataset_revision,
        "download_videos": bool(source_payload.get("download_videos")),
        "source_uri": source_payload.get("source_uri") or config.source_dataset_uri,
        "resolver": _lerobot_native_resolver_report(config, source=source),
        "preflight": _lerobot_native_preflight_report(config, source=source),
        "episode_filter": _lerobot_native_episode_filter(config),
    }


def _lerobot_native_resolver_report(
    config: BenchmarkRunConfig,
    *,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "source_mode": config.lerobot_native_source_mode,
        "cache_mode": config.lerobot_native_cache_mode,
        "selected_kind": (source or {}).get("kind"),
        "source_dataset_id": config.source_dataset_id,
        "source_dataset_revision": config.source_dataset_revision,
        "source_dataset_uri": config.source_dataset_uri,
        "projection_fallback_enabled": config.lerobot_native_source_mode == "auto",
        "cache_only": config.lerobot_native_cache_mode == "cache-only",
    }


def _lerobot_native_preflight_report(
    config: BenchmarkRunConfig,
    *,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = source or {}
    return {
        "source_size_tier": config.source_dataset_size_tier,
        "source_size_bytes": config.source_dataset_size_bytes,
        "source_size_note": config.source_dataset_size_note,
        "download_videos": bool(source.get("download_videos")),
        "implicit_hub_download": bool(source.get("download_videos") and not source.get("root")),
        "sample_budgets": {
            "sample_limit": config.sample_limit,
            "random_access_samples": config.random_access_samples,
            "random_frame_samples": config.random_frame_samples,
            "frames_per_clip": config.frames_per_clip,
            "query_limit": config.query_limit,
        },
    }


def _lerobot_native_episode_filter(config: BenchmarkRunConfig) -> dict[str, Any]:
    limit = config.lerobot_native_episode_limit
    if limit is None:
        return {
            "mode": "all",
            "episode_limit": None,
            "episodes": None,
            "applied_to_loader": False,
        }
    return {
        "mode": "first-n",
        "episode_limit": limit,
        "episodes": list(range(limit)),
        "applied_to_loader": True,
    }


def _open_lerobot_native_dataset(
    dataset_cls: Any,
    source: Mapping[str, Any],
    *,
    dependency_status: Mapping[str, Any],
) -> Any:
    kwargs: dict[str, Any] = {
        "revision": source.get("revision"),
        "download_videos": bool(source.get("download_videos")),
    }
    if source.get("root"):
        kwargs["root"] = source.get("root")
    if source.get("episodes") is not None:
        kwargs["episodes"] = list(source.get("episodes") or [])
    decode_backend = (dependency_status.get("decode_backend") or {}).get("selected")
    if decode_backend:
        kwargs["video_backend"] = decode_backend
    try:
        return dataset_cls(str(source["repo_id"]), **kwargs)
    except TypeError:
        kwargs.pop("video_backend", None)
        return dataset_cls(str(source["repo_id"]), **kwargs)


def _measure_lerobot_native_throughput(
    dataset: Any,
    dataloader_cls: Any,
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    count = min(len(dataset), config.sample_limit)
    start_ns = time.perf_counter_ns()
    materialized = 0
    yielded = 0
    loader = dataloader_cls(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        materialized += _materialized_bytes(batch)
        yielded += 1
        if yielded >= count:
            break
    duration = _elapsed_seconds(start_ns)
    return {
        "status": "completed",
        "value": _rate(yielded, duration),
        "unit": "samples/s",
        "details": {
            "samples": yielded,
            "duration_seconds": duration,
            "payload_bytes_materialized": materialized,
            "access_path": "official-lerobot-dataloader",
            "dataloader": "torch.utils.data.DataLoader",
        },
    }


def _measure_lerobot_native_random_access(dataset: Any, config: BenchmarkRunConfig) -> dict[str, Any]:
    indices = _random_indices(len(dataset), config.random_access_samples, config.seed)
    latencies_ms: list[float] = []
    materialized = 0
    for index in indices:
        start_ns = time.perf_counter_ns()
        sample = dataset[index]
        latencies_ms.append(_elapsed_seconds(start_ns) * 1000.0)
        materialized += _materialized_bytes(sample)
    return {
        "status": "completed",
        "value": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "unit": "ms/sample",
        "details": {
            "samples": len(indices),
            "indices": indices[:16],
            "mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50_ms": _percentile(latencies_ms, 0.50),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "max_ms": max(latencies_ms) if latencies_ms else 0.0,
            "payload_bytes_materialized": materialized,
            "access_path": "official-lerobot-dataset",
        },
    }


def _measure_lerobot_native_random_frame_sampling(
    dataset: Any,
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    plans = _random_frame_plans(
        len(dataset),
        clips=config.random_frame_samples,
        frames_per_clip=config.frames_per_clip,
        seed=config.seed,
    )
    latencies_ms: list[float] = []
    materialized = 0
    for plan in plans:
        for index in plan:
            start_ns = time.perf_counter_ns()
            sample = dataset[index]
            latencies_ms.append(_elapsed_seconds(start_ns) * 1000.0)
            materialized += _materialized_bytes(sample)
    return _random_frame_metric(
        plans,
        latencies_ms,
        payload_bytes=materialized,
        access_path="official-lerobot-dataset",
    )


def _measure_lerobot_native_storage(
    dataset: Any,
    *,
    source: Mapping[str, Any],
    projection_manifest: Mapping[str, Any] | None,
    ephemeral: bool,
    throughput: Mapping[str, Any],
) -> dict[str, Any]:
    root = getattr(dataset, "root", None) or source.get("root")
    stored_bytes = _path_size(root) if root else None
    materialized = 0
    if projection_manifest:
        accounting = projection_manifest.get("accounting") or {}
        materialized = int(accounting.get("metadata_bytes_written") or 0)
        materialized += int(
            accounting.get("payload_bytes_copied") or 0
        )
    return {
        "status": "completed",
        "value": int(stored_bytes or materialized or 0),
        "unit": "bytes",
        "details": {
            "stored_bytes": stored_bytes,
            "materialized_bytes_written": (
                0 if source.get("kind") == "source-corpus" else materialized
            ),
            "source_kind": source.get("kind"),
            "root": str(root) if root else None,
            "ephemeral_projection": bool(ephemeral and projection_manifest),
            "payload_bytes_hydrated": int(
                (throughput.get("details") or {}).get("payload_bytes_materialized") or 0
            ),
            "fixed_quality": "official LeRobotDataset access; source payload quality is not reduced",
        },
    }


def _lerobot_native_dataset_report(dataset: Any) -> dict[str, Any]:
    meta = getattr(dataset, "meta", None)
    features = getattr(dataset, "features", None) or getattr(meta, "features", None) or {}
    camera_keys = getattr(meta, "camera_keys", None) or []
    return {
        "class": type(dataset).__name__,
        "repo_id": getattr(dataset, "repo_id", None),
        "root": str(getattr(dataset, "root", "")),
        "revision": getattr(dataset, "revision", None),
        "num_frames": _optional_int(
            getattr(dataset, "num_frames", None),
            default=len(dataset),
        ),
        "num_episodes": _optional_int(getattr(dataset, "num_episodes", None)),
        "fps": _optional_int(getattr(dataset, "fps", None) or getattr(meta, "fps", None)),
        "camera_keys": list(camera_keys),
        "feature_keys": sorted(str(key) for key in features),
    }


def _lerobot_native_dependency_status() -> dict[str, Any]:
    core_modules = ("lerobot", "torch")
    module_statuses = {
        module: _safe_module_status(module)
        for module in (*core_modules, "torchcodec", "av")
    }
    missing_core = [
        module
        for module in core_modules
        if not module_statuses[module]["available"]
    ]
    decode_backends = {
        "torchcodec": {
            "module": "torchcodec",
            "available": module_statuses["torchcodec"]["available"],
            "version": module_statuses["torchcodec"]["version"],
            "import_error": module_statuses["torchcodec"]["import_error"],
        },
        "pyav": {
            "module": "av",
            "available": module_statuses["av"]["available"],
            "version": module_statuses["av"]["version"],
            "import_error": module_statuses["av"]["import_error"],
        },
    }
    selected_decode_backend = None
    for backend in ("torchcodec", "pyav"):
        if decode_backends[backend]["available"]:
            selected_decode_backend = backend
            break
    missing_decode = [] if selected_decode_backend else ["torchcodec", "av"]
    missing = [*missing_core, *missing_decode]
    return {
        "available": not missing,
        "modules": list(core_modules),
        "missing": missing,
        "install": f"install `{LEROBOT_NATIVE_BENCH_INSTALL}`",
        "install_policy": {
            "extra": LEROBOT_NATIVE_BENCH_EXTRA,
            "uv_sync": LEROBOT_NATIVE_BENCH_UV_SYNC,
            "pip_install": f"pip install '{LEROBOT_NATIVE_BENCH_INSTALL}'",
            "decode_policy": LEROBOT_NATIVE_DECODE_POLICY,
            "declared_decode_backend": "pyav",
            "optional_probe_backends": ["torchcodec"],
        },
        "versions": {
            module: module_statuses[module]["version"]
            for module in ("lerobot", "torch", "torchcodec", "av")
        },
        "import_errors": {
            module: module_statuses[module]["import_error"]
            for module in ("lerobot", "torch", "torchcodec", "av")
            if module_statuses[module]["import_error"]
        },
        "decode_backends": decode_backends,
        "decode_backend": {
            "selected": selected_decode_backend,
            "available": selected_decode_backend is not None,
            "missing": missing_decode,
            "policy": LEROBOT_NATIVE_DECODE_POLICY,
        },
    }


def _safe_module_status(module: str) -> dict[str, Any]:
    if not _module_available(module):
        return {
            "available": False,
            "version": _package_version(module),
            "import_error": None,
        }
    try:
        __import__(module)
    except Exception as exc:  # pragma: no cover - depends on optional packages.
        return {
            "available": False,
            "version": _package_version(module),
            "import_error": str(exc),
        }
    return {
        "available": True,
        "version": _package_version(module),
        "import_error": None,
    }


def _load_lerobot_native_components() -> tuple[_LeRobotNativeComponents | None, str | None]:
    dataset_cls: Any | None = None
    dataloader_cls: Any | None = None
    dataset_api = "lerobot.datasets.LeRobotDataset"
    import_errors: list[str] = []
    try:
        from lerobot.datasets import LeRobotDataset
        dataset_cls = LeRobotDataset
    except Exception as exc:  # pragma: no cover - depends on optional packages.
        import_errors.append(f"lerobot.datasets.LeRobotDataset: {exc}")

    if dataset_cls is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset_cls = LeRobotDataset
            dataset_api = "lerobot.datasets.lerobot_dataset.LeRobotDataset"
        except Exception as exc:  # pragma: no cover - depends on optional packages.
            import_errors.append(f"lerobot.datasets.lerobot_dataset.LeRobotDataset: {exc}")

    try:
        from torch.utils.data import DataLoader
        dataloader_cls = DataLoader
    except Exception as exc:  # pragma: no cover - depends on optional packages.
        import_errors.append(f"torch.utils.data.DataLoader: {exc}")

    if dataset_cls is None or dataloader_cls is None:
        return None, "; ".join(import_errors)

    return _LeRobotNativeComponents(
        dataset_cls=dataset_cls,
        dataloader_cls=dataloader_cls,
        dataset_api=dataset_api,
    ), None


def _lerobot_native_skip_reason(status: Mapping[str, Any]) -> str:
    missing = ", ".join(str(item) for item in status.get("missing") or [])
    install = status.get("install") or "install the optional LeRobot native stack"
    if missing:
        return (
            "optional LeRobot native dependency stack is not available; "
            f"missing {missing}; {install}"
        )
    return "optional LeRobot native dependency stack is not available"


def _skip_lerobot_native(
    reason: str,
    *,
    status: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
    projection_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "format": LEROBOT_NATIVE_FORMAT,
        "metrics": {
            metric: _skipped_metric(reason)
            for metric in BENCHMARK_METRICS
        },
        "skip_reason": reason,
        "native_loader": {
            "official_api": "lerobot.datasets.LeRobotDataset",
            "dependency_status": dict(status),
            "source": dict(source or {}),
        },
        "projection_manifest": dict(projection_manifest or {}),
        "notes": [
            reason,
            "Skipped explicitly so dependency-light benchmark reports cannot be "
            "read as official LeRobot loader coverage.",
        ],
    }


def _skipped_metric(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "value": None,
        "unit": None,
        "skip_reason": reason,
        "details": {},
    }


def _read_lerobot_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((out_dir / "data").glob("**/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _run_webdataset_format(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    artifact_root: Path | None,
) -> dict[str, Any]:
    if artifact_root is None:
        with tempfile.TemporaryDirectory(prefix="lancedb-robotics-bench-") as tmp:
            return _run_webdataset_in_dir(
                lake,
                snapshot,
                config=config,
                out_dir=Path(tmp) / WEBDATASET_FORMAT,
                ephemeral=True,
            )
    return _run_webdataset_in_dir(
        lake,
        snapshot,
        config=config,
        out_dir=artifact_root / WEBDATASET_FORMAT,
        ephemeral=False,
    )


def _run_webdataset_in_dir(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    out_dir: Path,
    ephemeral: bool,
) -> dict[str, Any]:
    try:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        start_export_ns = time.perf_counter_ns()
        manifest = export_dataset_snapshot(
            lake,
            snapshot["snapshot_name"],
            out_dir=out_dir,
            fmt=WEBDATASET_FORMAT,
            created_by="lancedb-robotics-bench",
        )
        export_seconds = _elapsed_seconds(start_export_ns)
    except DatasetExportError as exc:
        raise BenchmarkError(f"cannot export WebDataset projection: {exc}") from exc

    rows = _read_webdataset_rows(out_dir, manifest.data_files)
    throughput = _measure_rows_throughput(rows, config)
    random_access = _measure_rows_random_access(rows, config)
    random_frame_sampling = _measure_rows_random_frame_sampling(rows, config)
    storage = _measure_projection_storage(out_dir, manifest.data_files)
    curation = {
        "status": "completed",
        "value": export_seconds * 1000.0,
        "unit": "ms",
        "details": {
            "operation": "materialize_webdataset_projection",
            "snapshot_name": manifest.snapshot_name,
            "dataset_id": manifest.dataset_id,
            "transform_id": manifest.transform_id,
            "episode_count": manifest.episode_count,
            "step_count": manifest.step_count,
        },
    }

    return {
        "status": "completed",
        "format": WEBDATASET_FORMAT,
        "metrics": {
            METRIC_DATALOADER_THROUGHPUT: throughput,
            METRIC_RANDOM_ACCESS_LATENCY: random_access,
            METRIC_RANDOM_FRAME_SAMPLING: random_frame_sampling,
            METRIC_QUERY_TO_DATASET_CURATION: curation,
            METRIC_STORAGE_FOOTPRINT: storage,
        },
        "notes": [
            "WebDataset comparison uses deterministic tar shards projected from Lance.",
            "The native webdataset package is optional and reported separately.",
        ],
        "projection_manifest": {
            **manifest.to_dict(),
            "manifest_path": None
            if ephemeral
            else str(out_dir / DATASET_EXPORT_MANIFEST_FILENAME),
        },
    }


def _read_webdataset_rows(out_dir: Path, data_files: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in sorted(data_files):
        if not (rel.endswith(".tar") or rel.endswith(".tar.gz")):
            continue
        mode = "r:gz" if rel.endswith(".tar.gz") else "r:"
        with tarfile.open(out_dir / rel, mode=mode) as tar:
            for member in sorted(tar.getmembers(), key=lambda item: item.name):
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                rows.append(json.loads(extracted.read().decode()))
    return rows


def _measure_rows_throughput(rows: list[dict[str, Any]], config: BenchmarkRunConfig) -> dict[str, Any]:
    count = min(len(rows), config.sample_limit)
    start_ns = time.perf_counter_ns()
    materialized = 0
    for row in rows[:count]:
        materialized += _row_materialized_bytes(row)
    duration = _elapsed_seconds(start_ns)
    return {
        "status": "completed",
        "value": _rate(count, duration),
        "unit": "samples/s",
        "details": {
            "samples": count,
            "duration_seconds": duration,
            "row_materialized_bytes": materialized,
            "gpu_utilization_pct": None,
        },
    }


def _measure_rows_random_access(rows: list[dict[str, Any]], config: BenchmarkRunConfig) -> dict[str, Any]:
    indices = _random_indices(len(rows), config.random_access_samples, config.seed)
    latencies_ms: list[float] = []
    for index in indices:
        start_ns = time.perf_counter_ns()
        rows[index]
        latencies_ms.append(_elapsed_seconds(start_ns) * 1000.0)
    return {
        "status": "completed",
        "value": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "unit": "ms/sample",
        "details": {
            "samples": len(indices),
            "indices": indices[:16],
            "mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50_ms": _percentile(latencies_ms, 0.50),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "max_ms": max(latencies_ms) if latencies_ms else 0.0,
        },
    }


def _measure_rows_random_frame_sampling(rows: list[dict[str, Any]], config: BenchmarkRunConfig) -> dict[str, Any]:
    plans = _random_frame_plans(
        len(rows),
        clips=config.random_frame_samples,
        frames_per_clip=config.frames_per_clip,
        seed=config.seed,
    )
    latencies_ms: list[float] = []
    materialized = 0
    for plan in plans:
        for index in plan:
            start_ns = time.perf_counter_ns()
            row = rows[index]
            latencies_ms.append(_elapsed_seconds(start_ns) * 1000.0)
            materialized += _row_materialized_bytes(row)
    return _random_frame_metric(
        plans,
        latencies_ms,
        payload_bytes=materialized,
        access_path="materialized-format-rows",
    )


def _measure_projection_storage(out_dir: Path, data_files: tuple[str, ...]) -> dict[str, Any]:
    stored_bytes = _path_size(out_dir) or 0
    return {
        "status": "completed",
        "value": stored_bytes,
        "unit": "bytes",
        "details": {
            "stored_bytes": stored_bytes,
            "materialized_bytes_written": stored_bytes,
            "data_file_count": len(data_files),
            "fixed_quality": "source payload bytes; no transcoding or quality reduction",
        },
    }


# ---------------------------------------------------------------------------
# Analytics-lakehouse baselines (backlog 0126): Parquet and optional Iceberg.
#
# Both materialize the snapshot's observation rows -- the same logical sample set
# the Lance/enterprise-lance and WebDataset arms read -- into a columnar analytics
# table with payloads stored INLINE at fixed source quality. The metrics separate
# table metadata-scan cost from Python/PyTorch payload hydration cost, and the
# subset-filter-change metric records the rewrite/repack an analytics table needs
# when the filter changes, in contrast to Lance's version-pinned row plan.
# ---------------------------------------------------------------------------


def _snapshot_analytics_table(lake: Lake, snapshot: dict[str, Any]) -> pa.Table:
    """Select the snapshot's observation rows as an analytics table.

    Payloads are kept inline (``payload_blob`` binary column) at fixed source
    quality; nothing is transcoded. Row order is the deterministic observations
    table order filtered to the observation ids reachable from the snapshot's
    scenarios, so re-materializing the same snapshot yields the same table.
    """
    scenario_ids = set(snapshot["scenario_ids"])
    observation_ids: list[str] = []
    seen: set[str] = set()
    for row in lake.table("scenarios").to_arrow().to_pylist():
        if str(row["scenario_id"]) not in scenario_ids:
            continue
        for oid in row.get("observation_ids") or []:
            key = str(oid)
            if key not in seen:
                seen.add(key)
                observation_ids.append(key)
    observations = lake.table("observations").to_arrow()
    if not observation_ids:
        return observations.slice(0, 0)
    mask = pc.is_in(
        observations["observation_id"],
        value_set=pa.array(observation_ids, type=pa.string()),
    )
    return observations.filter(mask)


def _analytics_row_group_size(num_rows: int) -> int:
    """Pick a row-group size that yields multiple row groups when possible.

    Multiple row groups make the metadata-scan vs payload-hydration split and the
    row-group-bounded random-access penalty observable even on small fixtures.
    """
    if num_rows <= 1:
        return 1
    return max(1, math.ceil(num_rows / 8))


def _write_analytics_parquet(table: pa.Table, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        row_group_size=_analytics_row_group_size(table.num_rows),
    )
    return table.num_rows


def _run_parquet_format(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    artifact_root: Path | None,
) -> dict[str, Any]:
    if artifact_root is None:
        with tempfile.TemporaryDirectory(prefix="lancedb-robotics-bench-") as tmp:
            return _run_parquet_in_dir(
                lake,
                snapshot,
                config=config,
                out_dir=Path(tmp) / PARQUET_FORMAT,
                ephemeral=True,
            )
    return _run_parquet_in_dir(
        lake,
        snapshot,
        config=config,
        out_dir=artifact_root / PARQUET_FORMAT,
        ephemeral=False,
    )


def _run_parquet_in_dir(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    out_dir: Path,
    ephemeral: bool,
) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "table.parquet"

    start_materialize_ns = time.perf_counter_ns()
    table = _snapshot_analytics_table(lake, snapshot)
    row_count = _write_analytics_parquet(table, table_path)
    materialize_seconds = _elapsed_seconds(start_materialize_ns)

    metadata_scan = _measure_parquet_metadata_scan(table_path)
    rows, hydration = _measure_parquet_hydration(table_path, config)
    throughput = _measure_rows_throughput(rows, config)
    random_access = _measure_parquet_random_access(table_path, config)
    random_frame_sampling = _measure_rows_random_frame_sampling(rows, config)
    shuffled = _measure_rows_shuffled_epoch(rows, config)
    # Storage footprint reflects only the primary materialized table; it is
    # measured before the filter-change rewrite so the transient repack artifact
    # does not inflate the stored-bytes metric.
    storage = _measure_projection_storage(out_dir, ("table.parquet",))
    storage["details"]["payload_placement"] = PAYLOAD_PLACEMENT_INLINE
    filter_change = _measure_parquet_filter_change(table, out_dir, config)

    curation = {
        "status": "completed",
        "value": materialize_seconds * 1000.0,
        "unit": "ms",
        "details": {
            "operation": "materialize_parquet_analytics_table",
            "snapshot_name": snapshot["snapshot_name"],
            "dataset_id": snapshot["dataset_id"],
            "rows": row_count,
            "row_group_size": _analytics_row_group_size(row_count),
            "payload_placement": PAYLOAD_PLACEMENT_INLINE,
        },
    }

    return {
        "status": "completed",
        "format": PARQUET_FORMAT,
        "payload_placement": PAYLOAD_PLACEMENT_INLINE,
        "metrics": {
            METRIC_DATALOADER_THROUGHPUT: throughput,
            METRIC_RANDOM_ACCESS_LATENCY: random_access,
            METRIC_RANDOM_FRAME_SAMPLING: random_frame_sampling,
            METRIC_QUERY_TO_DATASET_CURATION: curation,
            METRIC_STORAGE_FOOTPRINT: storage,
            METRIC_METADATA_SCAN_LATENCY: metadata_scan,
            METRIC_ROW_HYDRATION_LATENCY: hydration,
            METRIC_SHUFFLED_EPOCH_THROUGHPUT: shuffled,
            METRIC_SUBSET_FILTER_CHANGE: filter_change,
        },
        "notes": [
            "Parquet analytics baseline materializes the snapshot's observation "
            "rows into a single columnar table with payloads stored inline.",
            "Metadata scan reads only the Parquet footer/row-group statistics; "
            "hydration reads and materializes payload bytes for the sampled rows.",
            "A subset-filter change rewrites/repacks a new Parquet table, whereas "
            "Lance creates a new version-pinned row plan without rewriting payloads.",
            "Baseline is not penalized for missing robotics loader features it does "
            "not claim; it measures analytics table-scan plus Python/PyTorch hydration.",
        ],
        "table_manifest": {
            "layout": PARQUET_ANALYTICS_LAYOUT_VERSION,
            "table_path": None if ephemeral else str(table_path),
            "rows": row_count,
            "row_group_size": _analytics_row_group_size(row_count),
            "payload_placement": PAYLOAD_PLACEMENT_INLINE,
            "columns": list(table.schema.names),
        },
    }


def _measure_parquet_metadata_scan(table_path: Path) -> dict[str, Any]:
    start_ns = time.perf_counter_ns()
    metadata = pq.read_metadata(table_path)
    schema = pq.read_schema(table_path)
    duration_ms = _elapsed_seconds(start_ns) * 1000.0
    return {
        "status": "completed",
        "value": duration_ms,
        "unit": "ms",
        "details": {
            "operation": "read_parquet_footer_metadata",
            "num_rows": metadata.num_rows,
            "num_row_groups": metadata.num_row_groups,
            "num_columns": metadata.num_columns,
            "column_names": list(schema.names),
            "read_payload": False,
        },
    }


def _measure_parquet_hydration(
    table_path: Path,
    config: BenchmarkRunConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_ns = time.perf_counter_ns()
    table = pq.read_table(table_path)
    rows = table.to_pylist()
    duration_seconds = _elapsed_seconds(start_ns)
    count = min(len(rows), config.sample_limit)
    payload_bytes = sum(_row_materialized_bytes(row) for row in rows[:count])
    per_sample_ms = (duration_seconds * 1000.0 / count) if count else 0.0
    return rows, {
        "status": "completed",
        "value": per_sample_ms,
        "unit": "ms/sample",
        "details": {
            "operation": "read_table_and_hydrate_payload",
            "samples": count,
            "rows_loaded": len(rows),
            "duration_seconds": duration_seconds,
            "payload_bytes_materialized": int(payload_bytes),
            "access_path": "full-table-read-then-index",
        },
    }


def _measure_parquet_random_access(
    table_path: Path,
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(table_path)
    metadata = parquet_file.metadata
    num_rows = metadata.num_rows
    indices = _random_indices(num_rows, config.random_access_samples, config.seed)
    row_group_rows = [
        metadata.row_group(rg).num_rows for rg in range(metadata.num_row_groups)
    ]
    offsets: list[int] = []
    running = 0
    for count in row_group_rows:
        offsets.append(running)
        running += count
    latencies_ms: list[float] = []
    for index in indices:
        target_rg = 0
        for rg in range(len(offsets)):
            if index >= offsets[rg]:
                target_rg = rg
        start_ns = time.perf_counter_ns()
        # Analytics random access must read the whole row group holding the row.
        parquet_file.read_row_group(target_rg)
        latencies_ms.append(_elapsed_seconds(start_ns) * 1000.0)
    return {
        "status": "completed",
        "value": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "unit": "ms/sample",
        "details": {
            "samples": len(indices),
            "indices": indices[:16],
            "num_row_groups": metadata.num_row_groups,
            "mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50_ms": _percentile(latencies_ms, 0.50),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "max_ms": max(latencies_ms) if latencies_ms else 0.0,
            "access_path": "read-row-group-per-sample",
        },
    }


def _measure_rows_shuffled_epoch(
    rows: list[dict[str, Any]],
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    order = list(range(len(rows)))
    random.Random(config.seed + 101).shuffle(order)
    start_ns = time.perf_counter_ns()
    materialized = 0
    for index in order:
        materialized += _row_materialized_bytes(rows[index])
    duration = _elapsed_seconds(start_ns)
    return {
        "status": "completed",
        "value": _rate(len(order), duration),
        "unit": "samples/s",
        "details": {
            "samples": len(order),
            "duration_seconds": duration,
            "payload_bytes_materialized": int(materialized),
            "shuffle_seed": config.seed + 101,
        },
    }


def _measure_parquet_filter_change(
    table: pa.Table,
    out_dir: Path,
    config: BenchmarkRunConfig,
) -> dict[str, Any]:
    selected_rows = max(1, table.num_rows // 2) if table.num_rows else 0
    filtered = table.slice(0, selected_rows)
    repack_path = out_dir / "table-filtered.parquet"
    start_ns = time.perf_counter_ns()
    _write_analytics_parquet(filtered, repack_path)
    duration_ms = _elapsed_seconds(start_ns) * 1000.0
    rewritten_bytes = _path_size(repack_path) or 0
    return {
        "status": "completed",
        "value": duration_ms,
        "unit": "ms",
        "details": {
            "operation": "rewrite_parquet_table_for_filter_change",
            "selected_rows": selected_rows,
            "base_rows": table.num_rows,
            "rewritten_bytes": int(rewritten_bytes),
            "materialized_bytes_written": int(rewritten_bytes),
            "requires_full_rewrite": True,
            "contrast": (
                "Lance re-plans a new version-pinned row plan over the same "
                "payloads; the Parquet analytics table must rewrite/repack a new "
                "table artifact for the changed subset filter."
            ),
        },
    }


def _run_iceberg_format(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    artifact_root: Path | None,
) -> dict[str, Any]:
    if not _module_available(ICEBERG_MODULE):
        return _skip_iceberg(
            f"optional dependency {ICEBERG_MODULE!r} is not installed; "
            f"install it with `{ICEBERG_INSTALL_HINT}` and configure a catalog "
            f"({ICEBERG_CATALOG_URI_ENV} or standard pyiceberg config) to run the "
            "Iceberg analytics baseline",
        )
    catalog_config, catalog_reason = _iceberg_catalog_config(artifact_root)
    if catalog_config is None:
        return _skip_iceberg(catalog_reason, catalog=None)
    if artifact_root is None:
        with tempfile.TemporaryDirectory(prefix="lancedb-robotics-iceberg-") as tmp:
            return _run_iceberg_in_dir(
                lake,
                snapshot,
                config=config,
                out_dir=Path(tmp) / ICEBERG_FORMAT,
                catalog_config=catalog_config,
                ephemeral=True,
            )
    return _run_iceberg_in_dir(
        lake,
        snapshot,
        config=config,
        out_dir=artifact_root / ICEBERG_FORMAT,
        catalog_config=catalog_config,
        ephemeral=False,
    )


def _iceberg_catalog_config(
    artifact_root: Path | None,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve an Iceberg catalog configuration for the benchmark.

    Priority: an explicit catalog URI env var, otherwise a local SQLite catalog
    rooted in the artifact directory (only usable when artifacts are retained --
    an ephemeral run with no configured catalog is a structured skip so a report
    can never imply Iceberg coverage it did not actually produce).
    """
    catalog_uri = os.environ.get(ICEBERG_CATALOG_URI_ENV)
    warehouse = os.environ.get(ICEBERG_WAREHOUSE_ENV)
    if catalog_uri:
        return (
            {
                "kind": "configured",
                "uri": catalog_uri,
                "warehouse": warehouse,
                "source": ICEBERG_CATALOG_URI_ENV,
            },
            "",
        )
    if artifact_root is None:
        return (
            None,
            "no Iceberg catalog configured; set "
            f"{ICEBERG_CATALOG_URI_ENV} or run with --artifacts so a local SQLite "
            "catalog can be created for the benchmark",
        )
    return (
        {
            "kind": "local-sqlite",
            "warehouse": str(artifact_root / ICEBERG_FORMAT / "warehouse"),
            "source": "artifact-local-default",
        },
        "",
    )


def _run_iceberg_in_dir(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    out_dir: Path,
    catalog_config: dict[str, Any],
    ephemeral: bool,
) -> dict[str, Any]:
    try:
        return _run_iceberg_materialized(
            lake,
            snapshot,
            config=config,
            out_dir=out_dir,
            catalog_config=catalog_config,
            ephemeral=ephemeral,
        )
    except Exception as exc:  # pragma: no cover - depends on optional pyiceberg stack.
        return _skip_iceberg(
            f"Iceberg analytics baseline could not run end to end: {exc}",
            catalog=catalog_config,
        )


def _run_iceberg_materialized(
    lake: Lake,
    snapshot: dict[str, Any],
    *,
    config: BenchmarkRunConfig,
    out_dir: Path,
    catalog_config: dict[str, Any],
    ephemeral: bool,
) -> dict[str, Any]:  # pragma: no cover - exercised only with pyiceberg installed.
    from pyiceberg.catalog.sql import SqlCatalog

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warehouse = Path(catalog_config.get("warehouse") or (out_dir / "warehouse"))
    warehouse.mkdir(parents=True, exist_ok=True)
    catalog_uri = catalog_config.get("uri") or f"sqlite:///{warehouse / 'catalog.db'}"
    catalog = SqlCatalog(
        "lancedb_robotics_bench",
        uri=catalog_uri,
        warehouse=f"file://{warehouse}",
    )

    start_materialize_ns = time.perf_counter_ns()
    table = _snapshot_analytics_table(lake, snapshot)
    try:
        catalog.create_namespace(ICEBERG_NAMESPACE)
    except Exception:  # noqa: BLE001 - namespace may already exist.
        pass
    identifier = f"{ICEBERG_NAMESPACE}.observations"
    try:
        catalog.drop_table(identifier)
    except Exception:  # noqa: BLE001 - table may not exist yet.
        pass
    iceberg_table = catalog.create_table(identifier, schema=table.schema)
    iceberg_table.append(table)
    materialize_seconds = _elapsed_seconds(start_materialize_ns)

    start_meta_ns = time.perf_counter_ns()
    scan = iceberg_table.scan()
    planned_files = list(scan.plan_files())
    metadata_ms = _elapsed_seconds(start_meta_ns) * 1000.0

    start_hydration_ns = time.perf_counter_ns()
    scanned = scan.to_arrow()
    rows = scanned.to_pylist()
    hydration_seconds = _elapsed_seconds(start_hydration_ns)
    count = min(len(rows), config.sample_limit)
    payload_bytes = sum(_row_materialized_bytes(row) for row in rows[:count])

    metadata_scan = {
        "status": "completed",
        "value": metadata_ms,
        "unit": "ms",
        "details": {
            "operation": "iceberg_plan_files",
            "planned_data_files": len(planned_files),
            "read_payload": False,
        },
    }
    hydration = {
        "status": "completed",
        "value": (hydration_seconds * 1000.0 / count) if count else 0.0,
        "unit": "ms/sample",
        "details": {
            "operation": "iceberg_scan_to_arrow_and_hydrate",
            "samples": count,
            "rows_loaded": len(rows),
            "duration_seconds": hydration_seconds,
            "payload_bytes_materialized": int(payload_bytes),
        },
    }
    throughput = _measure_rows_throughput(rows, config)
    random_access = _measure_rows_random_access(rows, config)
    random_frame_sampling = _measure_rows_random_frame_sampling(rows, config)
    shuffled = _measure_rows_shuffled_epoch(rows, config)

    filtered = table.slice(0, max(1, table.num_rows // 2) if table.num_rows else 0)
    start_filter_ns = time.perf_counter_ns()
    iceberg_table.overwrite(filtered)
    filter_ms = _elapsed_seconds(start_filter_ns) * 1000.0
    storage = _measure_projection_storage(out_dir, ("warehouse",))
    storage["details"]["payload_placement"] = PAYLOAD_PLACEMENT_INLINE

    curation = {
        "status": "completed",
        "value": materialize_seconds * 1000.0,
        "unit": "ms",
        "details": {
            "operation": "materialize_iceberg_table",
            "rows": table.num_rows,
            "payload_placement": PAYLOAD_PLACEMENT_INLINE,
        },
    }
    filter_change = {
        "status": "completed",
        "value": filter_ms,
        "unit": "ms",
        "details": {
            "operation": "iceberg_overwrite_for_filter_change",
            "selected_rows": filtered.num_rows,
            "base_rows": table.num_rows,
            "requires_new_snapshot": True,
            "contrast": (
                "Lance re-plans a new version-pinned row plan; the Iceberg table "
                "writes a new snapshot with rewritten data files for the filter."
            ),
        },
    }
    return {
        "status": "completed",
        "format": ICEBERG_FORMAT,
        "payload_placement": PAYLOAD_PLACEMENT_INLINE,
        "metrics": {
            METRIC_DATALOADER_THROUGHPUT: throughput,
            METRIC_RANDOM_ACCESS_LATENCY: random_access,
            METRIC_RANDOM_FRAME_SAMPLING: random_frame_sampling,
            METRIC_QUERY_TO_DATASET_CURATION: curation,
            METRIC_STORAGE_FOOTPRINT: storage,
            METRIC_METADATA_SCAN_LATENCY: metadata_scan,
            METRIC_ROW_HYDRATION_LATENCY: hydration,
            METRIC_SHUFFLED_EPOCH_THROUGHPUT: shuffled,
            METRIC_SUBSET_FILTER_CHANGE: filter_change,
        },
        "notes": [
            "Iceberg analytics baseline materializes the snapshot's observation "
            "rows into an Iceberg table with payloads stored inline.",
            "A subset-filter change writes a new Iceberg snapshot with rewritten "
            "data files, whereas Lance creates a new version-pinned row plan.",
        ],
        "catalog": {
            "kind": catalog_config.get("kind"),
            "uri": catalog_uri,
            "warehouse": str(warehouse),
            "namespace": ICEBERG_NAMESPACE,
            "identifier": identifier,
        },
        "table_manifest": {
            "layout": ICEBERG_ANALYTICS_LAYOUT_VERSION,
            "rows": table.num_rows,
            "payload_placement": PAYLOAD_PLACEMENT_INLINE,
            "columns": list(table.schema.names),
        },
    }


def _skip_iceberg(
    reason: str,
    *,
    catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "format": ICEBERG_FORMAT,
        "metrics": {metric: _skipped_metric(reason) for metric in BENCHMARK_METRICS},
        "skip_reason": reason,
        "iceberg": {
            "module": ICEBERG_MODULE,
            "available": _module_available(ICEBERG_MODULE),
            "install": ICEBERG_INSTALL_HINT,
            "catalog_env": ICEBERG_CATALOG_URI_ENV,
            "warehouse_env": ICEBERG_WAREHOUSE_ENV,
            "catalog": dict(catalog or {}),
        },
        "payload_placement": PAYLOAD_PLACEMENT_INLINE,
        "notes": [
            reason,
            "Skipped explicitly so reports without an Iceberg catalog cannot be "
            "read as Iceberg coverage.",
        ],
    }


def _skip_optional_format(fmt: str) -> dict[str, Any]:
    if fmt == DEEPLAKE_FORMAT:
        module = "deeplake"
        install = "install the `deeplake` package and add the Deep Lake adapter"
        follow_up = "Deep Lake is tracked as an optional competitive comparison adapter"
    else:
        raise BenchmarkError(f"cannot skip unknown optional format {fmt!r}")

    if _module_available(module):
        reason = f"{module} is importable, but no {fmt} benchmark adapter is registered yet"
    else:
        reason = f"optional dependency {module!r} is not installed; {install}"
    return {
        "status": "skipped",
        "format": fmt,
        "metrics": {},
        "skip_reason": reason,
        "notes": [
            reason,
            f"Skipped explicitly so the report cannot be read as coverage; {follow_up}.",
        ],
    }


def _skip_enterprise_lance(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "format": ENTERPRISE_LANCE_FORMAT,
        "metrics": {},
        "skip_reason": reason,
        "notes": [
            reason,
            "Skipped explicitly so local-only benchmark reports cannot be read as Enterprise coverage.",
        ],
    }


def _public_benchmark_report_id(
    *,
    source: str | Path,
    revision: str | None,
    size_tier: str,
    created_at: str,
    commit: str,
) -> str:
    timestamp = created_at.replace("-", "").replace(":", "").replace("Z", "Z")
    revision_part = revision or commit or "unversioned"
    return _slug(f"{source}-{size_tier}-{revision_part}-{timestamp}")[:96]


def _public_benchmark_revision(source: str | Path, revision: str | None) -> str | None:
    if revision:
        return str(revision)
    value = str(source)
    if _looks_like_hf_source(value) and "@" in value:
        return value.rsplit("@", maxsplit=1)[-1] or None
    return None


def _require_public_lerobot_revision(source: str | Path, revision: str | None) -> None:
    value = str(source)
    if _looks_like_hf_source(value) and "@" not in value and not revision:
        raise BenchmarkError(
            "public LeRobot benchmark runs require a pinned source revision; "
            "pass --revision <hf-commit> or include @<revision> in the source"
        )


def _public_benchmark_manifest(
    *,
    report_id: str,
    created_at: str,
    finished_at: str,
    commit: str,
    artifact_root: Path,
    run_dir: Path,
    prepare_path: Path,
    benchmark_path: Path,
    manifest_path: Path,
    log_path: Path,
    capacity_path: Path,
    benchmark_artifacts_dir: Path,
    prepare_report: Mapping[str, Any],
    benchmark_report: Mapping[str, Any],
    capacity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = benchmark_report.get("dataset", {}).get("source_corpus") or {}
    return {
        "schema_version": PUBLIC_LEROBOT_BENCHMARK_SCHEMA_VERSION,
        "report_id": report_id,
        "status": "completed",
        "created_at": created_at,
        "finished_at": finished_at,
        "commit": commit,
        "dataset": {
            "dataset_id": source.get("dataset_id") or prepare_report.get("dataset_id"),
            "revision": source.get("revision") or prepare_report.get("revision"),
            "source_uri": source.get("source_uri") or prepare_report.get("source_uri"),
            "size_tier": source.get("size_tier") or prepare_report.get("size_tier"),
            "storage_tier": source.get("storage_tier") or prepare_report.get("storage_tier"),
            "snapshot_name": benchmark_report.get("dataset", {}).get("snapshot_name")
            or prepare_report.get("snapshot_name"),
            "snapshot_dataset_id": benchmark_report.get("dataset", {}).get("dataset_id")
            or prepare_report.get("snapshot_dataset_id"),
            "scenario_count": benchmark_report.get("dataset", {}).get("scenario_count")
            or prepare_report.get("scenario_count"),
        },
        "paths": {
            "artifact_root": str(artifact_root),
            "run_dir": str(run_dir),
            "prepare_report": str(prepare_path),
            "benchmark_report": str(benchmark_path),
            "capacity_report": str(capacity_path),
            "artifact_manifest": str(manifest_path),
            "log": str(log_path),
            "benchmark_artifacts": str(benchmark_artifacts_dir),
        },
        "capacity": dict(capacity or benchmark_report.get("capacity") or {}),
        "format_statuses": _format_statuses(benchmark_report),
        "skipped": _skipped_formats(benchmark_report),
        "comparison_table": list(benchmark_report.get("comparison_table") or []),
        "hardware": dict(benchmark_report.get("hardware") or {}),
        "format_versions": dict(benchmark_report.get("format_versions") or {}),
        "storage_tiers": dict(benchmark_report.get("storage_tiers") or {}),
        "artifact_files": [],
        "history": {},
    }


def _write_public_benchmark_log(
    path: Path,
    *,
    report_id: str,
    created_at: str,
    commit: str,
    source: str | Path,
    revision: str | None,
    formats: tuple[str, ...],
    prepare_path: Path | None = None,
    benchmark_path: Path | None = None,
    capacity_path: Path | None = None,
    status: str = "completed",
    skip_reason: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"report_id={report_id}",
        f"status={status}",
        f"created_at={created_at}",
        f"commit={commit}",
        f"source={source}",
        f"revision={revision or ''}",
        f"formats={','.join(formats)}",
    ]
    if skip_reason:
        lines.append(f"skip_reason={skip_reason}")
    if prepare_path is not None:
        lines.append(f"prepare_report={prepare_path}")
    if benchmark_path is not None:
        lines.append(f"benchmark_report={benchmark_path}")
    if capacity_path is not None:
        lines.append(f"capacity_report={capacity_path}")
    path.write_text("\n".join(lines) + "\n")


def _public_benchmark_manifests(root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted((root / "runs").glob("*/artifact-manifest.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == PUBLIC_LEROBOT_BENCHMARK_SCHEMA_VERSION:
            payload["_manifest_path"] = str(path)
            manifests.append(payload)
    return manifests


def _public_benchmark_manifest_candidates(root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted((root / "runs").glob("*/artifact-manifest.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            payload["_manifest_path"] = str(path)
            manifests.append(payload)
    return manifests


def _validate_public_benchmark_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    report_id = str(manifest.get("report_id") or "")
    manifest_path = str(manifest.get("_manifest_path") or "")
    status = str(manifest.get("status") or "")
    if manifest.get("schema_version") != PUBLIC_LEROBOT_BENCHMARK_SCHEMA_VERSION:
        _add_validation_error(
            diagnostics,
            "manifest-schema-version",
            (
                f"manifest {report_id or manifest_path} has schema "
                f"{manifest.get('schema_version')!r}; expected "
                f"{PUBLIC_LEROBOT_BENCHMARK_SCHEMA_VERSION!r}"
            ),
            report_id=report_id or None,
            path=manifest_path,
        )
    for field in ("report_id", "status", "created_at", "commit"):
        _require_manifest_field(
            manifest,
            field,
            diagnostics,
            report_id=report_id,
            path=manifest_path,
        )
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        _add_validation_error(
            diagnostics,
            "manifest-dataset-missing",
            f"manifest {report_id} is missing dataset metadata",
            report_id=report_id,
            path=manifest_path,
        )
        dataset = {}
    for field in ("dataset_id", "revision", "source_uri", "size_tier", "storage_tier"):
        if not dataset.get(field):
            _add_validation_error(
                diagnostics,
                f"manifest-dataset-{field}-missing",
                f"manifest {report_id} is missing dataset.{field}",
                report_id=report_id,
                path=manifest_path,
            )
    format_statuses = manifest.get("format_statuses")
    if not isinstance(format_statuses, Mapping) or not format_statuses:
        _add_validation_error(
            diagnostics,
            "manifest-format-statuses-missing",
            f"manifest {report_id} is missing format_statuses evidence",
            report_id=report_id,
            path=manifest_path,
        )
        format_statuses = {}
    hardware = manifest.get("hardware")
    if not isinstance(hardware, Mapping) or not hardware.get("cpu_count"):
        _add_validation_error(
            diagnostics,
            "manifest-hardware-missing",
            f"manifest {report_id} is missing required hardware evidence",
            report_id=report_id,
            path=manifest_path,
        )
    paths = manifest.get("paths")
    if not isinstance(paths, Mapping):
        _add_validation_error(
            diagnostics,
            "manifest-paths-missing",
            f"manifest {report_id} is missing paths",
            report_id=report_id,
            path=manifest_path,
        )
        paths = {}
    required_paths = ["artifact_manifest", "log"]
    if status == "completed":
        required_paths.extend(["prepare_report", "benchmark_report"])
        if not manifest.get("comparison_table"):
            _add_validation_error(
                diagnostics,
                "manifest-comparison-table-missing",
                f"completed manifest {report_id} is missing comparison_table",
                report_id=report_id,
                path=manifest_path,
            )
    elif status in {"skipped", "dry-run"}:
        required_paths.append("capacity_report")
        if not manifest.get("skip_reason"):
            _add_validation_error(
                diagnostics,
                "manifest-skip-reason-missing",
                f"{status} manifest {report_id} is missing skip_reason",
                report_id=report_id,
                path=manifest_path,
            )
        if not manifest.get("capacity"):
            _add_validation_error(
                diagnostics,
                "manifest-capacity-missing",
                f"{status} manifest {report_id} is missing capacity evidence",
                report_id=report_id,
                path=manifest_path,
            )
    else:
        _add_validation_error(
            diagnostics,
            "manifest-status-unknown",
            f"manifest {report_id} has unknown status {status!r}",
            report_id=report_id,
            path=manifest_path,
        )
    for field in required_paths:
        value = paths.get(field)
        if not value:
            _add_validation_error(
                diagnostics,
                f"manifest-path-{field}-missing",
                f"manifest {report_id} is missing paths.{field}",
                report_id=report_id,
                path=manifest_path,
            )
        elif not _public_benchmark_path_exists(value, root=root):
            _add_validation_error(
                diagnostics,
                f"manifest-path-{field}-not-found",
                f"manifest {report_id} points to missing {field}: {value}",
                report_id=report_id,
                path=str(value),
            )
    return {
        "report_id": report_id,
        "status": status,
        "path": manifest_path,
        "dataset_revision": dataset.get("revision"),
        "storage_tier": dataset.get("storage_tier"),
        "format_statuses": dict(format_statuses),
    }


def _load_public_benchmark_index(
    root: Path,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    index_path = root / "index.json"
    try:
        payload = json.loads(index_path.read_text())
    except FileNotFoundError:
        _add_validation_error(
            diagnostics,
            "index-missing",
            f"public LeRobot benchmark index is missing: {index_path}",
            path=str(index_path),
        )
        return {}
    except json.JSONDecodeError as exc:
        _add_validation_error(
            diagnostics,
            "index-json-invalid",
            f"public LeRobot benchmark index is not valid JSON: {exc}",
            path=str(index_path),
        )
        return {}
    return payload if isinstance(payload, dict) else {}


def _validate_public_benchmark_index(
    index: Mapping[str, Any],
    *,
    manifests: Mapping[str, Mapping[str, Any]],
    root: Path,
    diagnostics: list[dict[str, Any]],
) -> None:
    if not index:
        return
    index_path = str(root / "index.json")
    if index.get("schema_version") != PUBLIC_LEROBOT_BENCHMARK_SCHEMA_VERSION:
        _add_validation_error(
            diagnostics,
            "index-schema-version",
            (
                f"index has schema {index.get('schema_version')!r}; expected "
                f"{PUBLIC_LEROBOT_BENCHMARK_SCHEMA_VERSION!r}"
            ),
            path=index_path,
        )
    rows = index.get("runs")
    if not isinstance(rows, list):
        _add_validation_error(
            diagnostics,
            "index-runs-missing",
            "index is missing runs list",
            path=index_path,
        )
        return
    if index.get("run_count") != len(rows):
        _add_validation_error(
            diagnostics,
            "index-run-count-mismatch",
            f"index run_count {index.get('run_count')} does not match {len(rows)} rows",
            path=index_path,
        )
    for row in rows:
        if not isinstance(row, Mapping):
            _add_validation_error(
                diagnostics,
                "index-row-invalid",
                "index contains a non-object run row",
                path=index_path,
            )
            continue
        report_id = str(row.get("report_id") or "")
        manifest = manifests.get(report_id)
        if not manifest:
            _add_validation_error(
                diagnostics,
                "index-report-missing-manifest",
                f"index row {report_id!r} has no retained manifest",
                report_id=report_id or None,
                path=index_path,
            )
            continue
        dataset = manifest.get("dataset") or {}
        _require_index_match(
            row,
            manifest,
            "commit",
            diagnostics,
            report_id=report_id,
            path=index_path,
        )
        for field in ("revision", "storage_tier", "size_tier"):
            if row.get(field) != dataset.get(field):
                _add_validation_error(
                    diagnostics,
                    f"index-{field}-mismatch",
                    (
                        f"index row {report_id} has {field}={row.get(field)!r}; "
                        f"manifest has {dataset.get(field)!r}"
                    ),
                    report_id=report_id,
                    path=index_path,
                )
        if row.get("status") and row.get("status") != manifest.get("status"):
            _add_validation_error(
                diagnostics,
                "index-status-mismatch",
                (
                    f"index row {report_id} has status={row.get('status')!r}; "
                    f"manifest has {manifest.get('status')!r}"
                ),
                report_id=report_id,
                path=index_path,
            )
        if not row.get("format_statuses"):
            _add_validation_error(
                diagnostics,
                "index-format-statuses-missing",
                f"index row {report_id} is missing format_statuses",
                report_id=report_id,
                path=index_path,
            )


def _load_public_benchmark_claims(
    *,
    claims_path: str | Path | None,
    claims: Mapping[str, Any] | list[Mapping[str, Any]] | None,
    diagnostics: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if claims_path is not None and claims is not None:
        _add_validation_error(
            diagnostics,
            "claims-source-conflict",
            "pass either claims_path or claims, not both",
            path=str(claims_path),
        )
        return {}, []
    if claims_path is None and claims is None:
        return {"schema_version": PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION, "claims": []}, []
    payload: Any
    if claims_path is not None:
        path = Path(claims_path)
        try:
            payload = json.loads(path.read_text())
        except FileNotFoundError:
            _add_validation_error(
                diagnostics,
                "claims-file-missing",
                f"claim manifest does not exist: {path}",
                path=str(path),
            )
            return {}, []
        except json.JSONDecodeError as exc:
            _add_validation_error(
                diagnostics,
                "claims-json-invalid",
                f"claim manifest is not valid JSON: {exc}",
                path=str(path),
            )
            return {}, []
    else:
        payload = claims
    if isinstance(payload, list):
        entries = payload
        return {"schema_version": PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION, "claims": entries}, entries
    if not isinstance(payload, Mapping):
        _add_validation_error(
            diagnostics,
            "claims-invalid",
            "claim manifest must be an object with a claims list or a claims list",
            path=str(claims_path) if claims_path is not None else None,
        )
        return {}, []
    if payload.get("schema_version") != PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION:
        _add_validation_error(
            diagnostics,
            "claims-schema-version",
            (
                f"claim manifest has schema {payload.get('schema_version')!r}; "
                f"expected {PUBLIC_LEROBOT_CLAIMS_SCHEMA_VERSION!r}"
            ),
            path=str(claims_path) if claims_path is not None else None,
        )
    entries = payload.get("claims")
    if not isinstance(entries, list):
        _add_validation_error(
            diagnostics,
            "claims-list-missing",
            "claim manifest is missing a claims list",
            path=str(claims_path) if claims_path is not None else None,
        )
        return dict(payload), []
    return dict(payload), [entry for entry in entries if isinstance(entry, Mapping)]


def _validate_public_benchmark_claim(
    claim: Mapping[str, Any],
    *,
    claim_index: int,
    manifests: Mapping[str, Mapping[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_id = str(claim.get("id") or claim.get("claim_id") or f"claim-{claim_index + 1}")
    report_id = str(claim.get("report_id") or "")
    fmt = str(claim.get("format") or "")
    metric = str(claim.get("metric") or "")
    expected_status = str(claim.get("status") or "completed")
    for field in ("report_id", "format", "commit", "dataset_revision", "storage_tier"):
        if not claim.get(field):
            _add_validation_error(
                diagnostics,
                f"claim-{field}-missing",
                f"claim {claim_id} is missing {field}",
                claim_id=claim_id,
                report_id=report_id or None,
            )
    if expected_status == "completed" and not metric:
        _add_validation_error(
            diagnostics,
            "claim-metric-missing",
            f"completed claim {claim_id} is missing metric",
            claim_id=claim_id,
            report_id=report_id or None,
        )
    manifest = manifests.get(report_id)
    if not manifest:
        _add_validation_error(
            diagnostics,
            "claim-report-missing",
            f"claim {claim_id} references unknown report_id {report_id!r}",
            claim_id=claim_id,
            report_id=report_id or None,
        )
        return {
            "id": claim_id,
            "report_id": report_id,
            "format": fmt,
            "metric": metric or None,
            "status": "failed",
        }
    dataset = manifest.get("dataset") or {}
    _validate_claim_match(
        claim,
        "commit",
        manifest.get("commit"),
        diagnostics,
        claim_id=claim_id,
        report_id=report_id,
    )
    _validate_claim_match(
        claim,
        "dataset_revision",
        dataset.get("revision"),
        diagnostics,
        claim_id=claim_id,
        report_id=report_id,
    )
    _validate_claim_match(
        claim,
        "storage_tier",
        dataset.get("storage_tier"),
        diagnostics,
        claim_id=claim_id,
        report_id=report_id,
    )
    format_statuses = manifest.get("format_statuses") or {}
    actual_status = format_statuses.get(fmt)
    if actual_status is None:
        _add_validation_error(
            diagnostics,
            "claim-format-missing",
            f"claim {claim_id} references format {fmt!r} not present in report {report_id}",
            claim_id=claim_id,
            report_id=report_id,
        )
    elif expected_status != actual_status:
        _add_validation_error(
            diagnostics,
            "claim-format-status-mismatch",
            (
                f"claim {claim_id} expects {fmt} status {expected_status!r}; "
                f"report {report_id} recorded {actual_status!r}"
            ),
            claim_id=claim_id,
            report_id=report_id,
        )
    if expected_status == "completed" and actual_status != "completed":
        _add_validation_error(
            diagnostics,
            "claim-format-not-measured",
            (
                f"claim {claim_id} presents {fmt} as measured, but report "
                f"{report_id} recorded status {actual_status!r}"
            ),
            claim_id=claim_id,
            report_id=report_id,
        )
    if expected_status == "completed" and metric:
        _validate_claim_metric(
            claim,
            manifest,
            fmt=fmt,
            metric=metric,
            claim_id=claim_id,
            report_id=report_id,
            diagnostics=diagnostics,
        )
    return {
        "id": claim_id,
        "report_id": report_id,
        "format": fmt,
        "metric": metric or None,
        "status": "failed"
        if any(
            diagnostic.get("claim_id") == claim_id and diagnostic["level"] == "error"
            for diagnostic in diagnostics
        )
        else "passed",
    }


def _validate_claim_metric(
    claim: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    fmt: str,
    metric: str,
    claim_id: str,
    report_id: str,
    diagnostics: list[dict[str, Any]],
) -> None:
    rows = list(manifest.get("comparison_table") or [])
    row = next((item for item in rows if str(item.get("format")) == fmt), None)
    if not row:
        _add_validation_error(
            diagnostics,
            "claim-format-row-missing",
            f"claim {claim_id} cannot find comparison row for {fmt} in {report_id}",
            claim_id=claim_id,
            report_id=report_id,
        )
        return
    if metric not in row or row.get(metric) in (None, ""):
        _add_validation_error(
            diagnostics,
            "claim-metric-missing-in-report",
            f"claim {claim_id} references metric {metric!r} missing from {report_id}/{fmt}",
            claim_id=claim_id,
            report_id=report_id,
        )
        return
    if "value" in claim or "expected_value" in claim:
        expected = _optional_float(claim.get("value", claim.get("expected_value")))
        observed = _optional_float(row.get(metric))
        tolerance = _optional_float(claim.get("tolerance")) or 0.0
        if expected is None or observed is None:
            _add_validation_error(
                diagnostics,
                "claim-metric-value-invalid",
                f"claim {claim_id} has non-numeric expected or observed metric value",
                claim_id=claim_id,
                report_id=report_id,
            )
            return
        if abs(observed - expected) > tolerance:
            _add_validation_error(
                diagnostics,
                "claim-metric-value-mismatch",
                (
                    f"claim {claim_id} expected {metric}={expected}±{tolerance}; "
                    f"report {report_id}/{fmt} recorded {observed}"
                ),
                claim_id=claim_id,
                report_id=report_id,
            )


def _require_manifest_field(
    manifest: Mapping[str, Any],
    field: str,
    diagnostics: list[dict[str, Any]],
    *,
    report_id: str,
    path: str,
) -> None:
    if not manifest.get(field):
        _add_validation_error(
            diagnostics,
            f"manifest-{field}-missing",
            f"manifest {report_id or path} is missing {field}",
            report_id=report_id or None,
            path=path,
        )


def _require_index_match(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    field: str,
    diagnostics: list[dict[str, Any]],
    *,
    report_id: str,
    path: str,
) -> None:
    if row.get(field) != manifest.get(field):
        _add_validation_error(
            diagnostics,
            f"index-{field}-mismatch",
            (
                f"index row {report_id} has {field}={row.get(field)!r}; "
                f"manifest has {manifest.get(field)!r}"
            ),
            report_id=report_id,
            path=path,
        )


def _validate_claim_match(
    claim: Mapping[str, Any],
    field: str,
    expected: Any,
    diagnostics: list[dict[str, Any]],
    *,
    claim_id: str,
    report_id: str,
) -> None:
    if claim.get(field) and claim.get(field) != expected:
        _add_validation_error(
            diagnostics,
            f"claim-{field}-mismatch",
            (
                f"claim {claim_id} has {field}={claim.get(field)!r}; "
                f"report {report_id} has {expected!r}"
            ),
            claim_id=claim_id,
            report_id=report_id,
        )


def _public_benchmark_path_exists(value: Any, *, root: Path) -> bool:
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.exists()


def _add_validation_error(
    diagnostics: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    report_id: str | None = None,
    claim_id: str | None = None,
    path: str | None = None,
) -> None:
    diagnostic: dict[str, Any] = {
        "level": "error",
        "code": code,
        "message": message,
    }
    if report_id:
        diagnostic["report_id"] = report_id
    if claim_id:
        diagnostic["claim_id"] = claim_id
    if path:
        diagnostic["path"] = path
    diagnostics.append(diagnostic)


def _public_benchmark_history_row(manifest: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    dataset = manifest.get("dataset") or {}
    paths = manifest.get("paths") or {}
    comparison = list(manifest.get("comparison_table") or [])
    retention = manifest.get("retention") or {}
    publication = manifest.get("publication") or {}
    capacity = dict(manifest.get("capacity") or {})
    capacity_reasons = list(
        capacity.get("skip_reasons")
        or (capacity.get("selected") or {}).get("skip_reasons")
        or []
    )
    return {
        "report_id": manifest.get("report_id"),
        "status": manifest.get("status") or "completed",
        "skip_reason": manifest.get("skip_reason"),
        "created_at": manifest.get("created_at"),
        "commit": manifest.get("commit"),
        "dataset_id": dataset.get("dataset_id"),
        "revision": dataset.get("revision"),
        "size_tier": dataset.get("size_tier"),
        "storage_tier": dataset.get("storage_tier"),
        "snapshot_name": dataset.get("snapshot_name"),
        "scenario_count": dataset.get("scenario_count"),
        "format_statuses": dict(manifest.get("format_statuses") or {}),
        "skipped": dict(manifest.get("skipped") or {}),
        "metrics": _history_metrics(comparison),
        "capacity_status": capacity.get("status"),
        "capacity_skip_reasons": capacity_reasons,
        "capacity": capacity,
        "retention_class": retention.get("class"),
        "retention_protected": bool(retention.get("protected")),
        "publication": dict(publication),
        "artifact_manifest": _relative_to_root(paths.get("artifact_manifest"), root),
        "benchmark_report": _relative_to_root(paths.get("benchmark_report"), root),
        "prepare_report": _relative_to_root(paths.get("prepare_report"), root),
        "capacity_report": _relative_to_root(paths.get("capacity_report"), root),
        "log": _relative_to_root(paths.get("log"), root),
    }


def _public_benchmark_dashboard_markdown(index: Mapping[str, Any]) -> str:
    lines = [
        "# Public LeRobot Benchmark History",
        "",
        f"Updated: {index.get('updated_at')}",
        f"Runs: {index.get('run_count')}",
        "",
        "| Report | Status | Capacity | Commit | Dataset | Revision | Tier | Storage | Formats | Lance samples/s | LeRobot default samples/s | LeRobot native samples/s | Random frames/s | Artifacts |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in reversed(list(index.get("runs") or [])):
        metrics = row.get("metrics") or {}
        statuses = row.get("format_statuses") or {}
        formats = ", ".join(f"{fmt}:{status}" for fmt, status in sorted(statuses.items()))
        artifact = row.get("artifact_manifest") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("report_id") or ""),
                    str(row.get("status") or ""),
                    _capacity_summary(row),
                    str(row.get("commit") or "")[:12],
                    str(row.get("dataset_id") or ""),
                    str(row.get("revision") or ""),
                    str(row.get("size_tier") or ""),
                    str(row.get("storage_tier") or ""),
                    formats,
                    _display_metric(metrics.get("lance_throughput")),
                    _display_metric(metrics.get("lerobot_default_throughput")),
                    _display_metric(metrics.get("lerobot_native_throughput")),
                    _display_metric(metrics.get("lance_random_frame_sampling")),
                    artifact,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _format_statuses(report: Mapping[str, Any]) -> dict[str, str]:
    formats = report.get("formats") or {}
    return {
        str(fmt): str((result or {}).get("status") or "unknown")
        for fmt, result in sorted(formats.items())
    }


def _skipped_formats(report: Mapping[str, Any]) -> dict[str, str]:
    formats = report.get("formats") or {}
    skipped: dict[str, str] = {}
    for fmt, result in formats.items():
        if (result or {}).get("status") == "skipped":
            skipped[str(fmt)] = str((result or {}).get("skip_reason") or "")
    return dict(sorted(skipped.items()))


def _history_metrics(comparison_table: list[Mapping[str, Any]]) -> dict[str, float | None]:
    by_format = {str(row.get("format")): row for row in comparison_table}
    lance = by_format.get(LANCE_FORMAT) or {}
    lerobot = by_format.get(LEROBOT_DEFAULT_FORMAT) or {}
    lerobot_native = by_format.get(LEROBOT_NATIVE_FORMAT) or {}
    return {
        "lance_throughput": _optional_float(lance.get(METRIC_DATALOADER_THROUGHPUT)),
        "lerobot_default_throughput": _optional_float(
            lerobot.get(METRIC_DATALOADER_THROUGHPUT)
        ),
        "lerobot_native_throughput": _optional_float(
            lerobot_native.get(METRIC_DATALOADER_THROUGHPUT)
        ),
        "lance_random_access_ms": _optional_float(lance.get(METRIC_RANDOM_ACCESS_LATENCY)),
        "lance_random_frame_sampling": _optional_float(
            lance.get(METRIC_RANDOM_FRAME_SAMPLING)
        ),
        "lance_storage_bytes": _optional_float(lance.get(METRIC_STORAGE_FOOTPRINT)),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _display_metric(value: Any) -> str:
    parsed = _optional_float(value)
    return "" if parsed is None else f"{parsed:.3f}"


def _capacity_summary(row: Mapping[str, Any]) -> str:
    status = str(row.get("capacity_status") or "")
    reasons = [str(reason) for reason in row.get("capacity_skip_reasons") or []]
    if not reasons:
        return status
    joined = "; ".join(reasons)
    return f"{status}: {joined[:160]}"


def _artifact_files(run_dir: Path) -> list[str]:
    files = []
    for child in sorted(run_dir.rglob("*")):
        if child.is_file():
            files.append(str(child.relative_to(run_dir)))
    return files


def _public_benchmark_publication_files(root: Path, manifests: list[Mapping[str, Any]]) -> list[str]:
    files: set[str] = set()
    for manifest in manifests:
        manifest_path = Path(str(manifest["_manifest_path"]))
        run_dir = manifest_path.parent
        for rel in _artifact_files(run_dir):
            files.add(str(run_dir.relative_to(root) / rel))
    return sorted(files)


def _refresh_public_benchmark_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    run_dir: Path,
    destination: str,
    backend_name: str,
    published_at: str,
    dry_run: bool,
    retention: Mapping[str, Any],
) -> dict[str, Any]:
    updated = {key: value for key, value in manifest.items() if not str(key).startswith("_")}
    artifact_files = _artifact_files(run_dir)
    updated["artifact_files"] = artifact_files
    updated["artifact_checksums"] = _artifact_file_checksums(run_dir, artifact_files)
    updated["retention"] = dict(retention)
    existing_publication = dict(updated.get("publication") or {})
    published_targets = list(existing_publication.get("targets") or [])
    if not any(target.get("destination") == destination for target in published_targets):
        published_targets.append(
            {
                "destination": destination,
                "backend": backend_name,
                "published_at": published_at,
                "status": "planned" if dry_run else "published",
                "artifact_root": str(root),
                "file_count": len(artifact_files),
            }
        )
    updated["publication"] = {
        "latest_destination": destination,
        "latest_backend": backend_name,
        "target_count": len(published_targets),
        "targets": published_targets,
    }
    return updated


def _artifact_file_checksums(run_dir: Path, artifact_files: list[str]) -> dict[str, dict[str, Any]]:
    checksums: dict[str, dict[str, Any]] = {}
    for rel in artifact_files:
        if rel == "artifact-manifest.json":
            continue
        path = run_dir / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        checksums[rel] = {
            "sha256": _sha256_bytes(data),
            "size_bytes": len(data),
        }
    return dict(sorted(checksums.items()))


def _public_benchmark_retention_plan(
    manifests: list[Mapping[str, Any]],
    *,
    retention_class: str,
    retain_latest: int | None,
) -> dict[str, dict[str, Any]]:
    sorted_manifests = sorted(
        manifests,
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("report_id") or "")),
    )
    latest_ids = {
        str(item.get("report_id"))
        for item in (sorted_manifests[-retain_latest:] if retain_latest else sorted_manifests)
    }
    plan: dict[str, dict[str, Any]] = {}
    for manifest in sorted_manifests:
        report_id = str(manifest.get("report_id"))
        claim_linked = _public_benchmark_claim_linked(manifest)
        protected = claim_linked or report_id in latest_ids
        if claim_linked:
            reason = "claim-linked"
        elif retain_latest is None:
            reason = "unbounded-history"
        elif protected:
            reason = f"within-latest-{retain_latest}-window"
        else:
            reason = f"outside-latest-{retain_latest}-window"
        plan[report_id] = {
            "class": retention_class,
            "retain_latest": retain_latest,
            "protected": protected,
            "claim_linked": claim_linked,
            "reason": reason,
            "gc_action": "preserve" if protected else "eligible",
        }
    return plan


def _public_benchmark_claim_linked(manifest: Mapping[str, Any]) -> bool:
    for key in ("claim", "claims", "claim_refs", "claim_references"):
        value = manifest.get(key)
        if value:
            return True
    retention = manifest.get("retention") or {}
    return bool(retention.get("claim_linked"))


def _publication_backend(destination: str) -> dict[str, Any]:
    if destination.startswith("file://"):
        return {"name": "filesystem", "root": Path(destination.removeprefix("file://"))}
    if "://" not in destination:
        return {"name": "filesystem", "root": Path(destination)}
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - exercised when optional fsspec missing.
        raise BenchmarkError(
            "object-store publication destinations require optional dependency 'fsspec'"
        ) from exc
    fs, root = fsspec.core.url_to_fs(destination)
    return {"name": "fsspec", "fs": fs, "root": root.rstrip("/")}


def _publication_read_bytes(backend: Mapping[str, Any], rel: str) -> bytes | None:
    if backend["name"] == "filesystem":
        path = Path(backend["root"]) / rel
        return path.read_bytes() if path.exists() else None
    fs = backend["fs"]
    path = _publication_fsspec_path(backend, rel)
    if not fs.exists(path):
        return None
    with fs.open(path, "rb") as handle:
        return handle.read()


def _publication_write_bytes(backend: Mapping[str, Any], rel: str, data: bytes) -> None:
    if backend["name"] == "filesystem":
        path = Path(backend["root"]) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    fs = backend["fs"]
    path = _publication_fsspec_path(backend, rel)
    parent = path.rsplit("/", maxsplit=1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "wb") as handle:
        handle.write(data)


def _publication_fsspec_path(backend: Mapping[str, Any], rel: str) -> str:
    root = str(backend["root"]).rstrip("/")
    return f"{root}/{rel}" if root else rel


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative_to_root(value: Any, root: Path) -> str:
    if not value:
        return ""
    path = Path(str(value))
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _git_commit() -> str:
    for name in ("GITHUB_SHA", "BUILDKITE_COMMIT", "CI_COMMIT_SHA"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _format_versions() -> dict[str, Any]:
    return {
        ENTERPRISE_LANCE_FORMAT: {
            "lancedb-robotics": __version__,
            "lancedb": _package_version("lancedb"),
            "pylance": _package_version("pylance"),
            "pyarrow": _package_version("pyarrow"),
            "loader_report_schema": "lancedb-robotics/training-loader-report/v1",
        },
        LANCE_FORMAT: {
            "lancedb-robotics": __version__,
            "lancedb": _package_version("lancedb"),
            "pylance": _package_version("pylance"),
            "pyarrow": _package_version("pyarrow"),
        },
        LEROBOT_DEFAULT_FORMAT: {
            "layout": LEROBOT_FORMAT_VERSION,
            "native_loader": native_loader_status("lerobot"),
            "pyarrow": _package_version("pyarrow"),
        },
        LEROBOT_NATIVE_FORMAT: {
            "layout": LEROBOT_FORMAT_VERSION,
            "official_api": "lerobot.datasets.LeRobotDataset",
            "supported_import_paths": [
                "lerobot.datasets.LeRobotDataset",
                "lerobot.datasets.lerobot_dataset.LeRobotDataset",
            ],
            "lerobot": _package_version("lerobot"),
            "torch": _package_version("torch"),
            "torchcodec": _package_version("torchcodec"),
            "av": _package_version("av"),
            "install_policy": {
                "extra": LEROBOT_NATIVE_BENCH_EXTRA,
                "uv_sync": LEROBOT_NATIVE_BENCH_UV_SYNC,
                "pip_install": f"pip install '{LEROBOT_NATIVE_BENCH_INSTALL}'",
                "decode_policy": LEROBOT_NATIVE_DECODE_POLICY,
                "declared_decode_backend": "pyav",
                "optional_probe_backends": ["torchcodec"],
            },
            "decode_backends": {
                "torchcodec": _module_available("torchcodec"),
                "pyav": _module_available("av"),
            },
            "availability_note": (
                "The native arm performs guarded imports only when requested and "
                "records detailed dependency_status in formats.lerobot-native."
            ),
        },
        WEBDATASET_FORMAT: {
            "layout": WEBDATASET_FORMAT_VERSION,
            "native_loader": native_loader_status("webdataset"),
        },
        PARQUET_FORMAT: {
            "layout": PARQUET_ANALYTICS_LAYOUT_VERSION,
            "pyarrow": _package_version("pyarrow"),
            "payload_placement": PAYLOAD_PLACEMENT_INLINE,
            "note": "Dependency-light analytics baseline; pyarrow is already required.",
        },
        ICEBERG_FORMAT: {
            "layout": ICEBERG_ANALYTICS_LAYOUT_VERSION,
            "pyiceberg": _package_version(ICEBERG_MODULE),
            "available": _module_available(ICEBERG_MODULE),
            "payload_placement": PAYLOAD_PLACEMENT_INLINE,
            "install_policy": {
                "install": ICEBERG_INSTALL_HINT,
                "catalog_env": ICEBERG_CATALOG_URI_ENV,
                "warehouse_env": ICEBERG_WAREHOUSE_ENV,
            },
        },
        DEEPLAKE_FORMAT: {
            "deeplake": _package_version("deeplake"),
            "available": _module_available("deeplake"),
        },
    }


def _hardware_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu_info(),
    }


def _gpu_info() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "utilization_pct": None,
            "note": "nvidia-smi not found",
        }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "utilization_pct": None, "note": str(exc)}
    if result.returncode != 0:
        return {
            "available": False,
            "utilization_pct": None,
            "note": result.stderr.strip() or "nvidia-smi returned non-zero",
        }
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    utilizations: list[int] = []
    names: list[str] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        try:
            utilizations.append(int(parts[0]))
        except (IndexError, ValueError):
            continue
        if len(parts) > 1:
            names.append(parts[1])
    utilization = statistics.mean(utilizations) if utilizations else None
    return {
        "available": bool(utilizations),
        "utilization_pct": utilization,
        "names": names,
    }


def _methodology() -> dict[str, Any]:
    return {
        "deterministic": True,
        "sample_order": "Lance uses snapshot frame order; random access uses Python Random(seed).",
        "random_frame_sampling": (
            "A seeded N-frames-per-clip workload samples short frame windows to "
            "separate random frame access from sequential dataloader scans."
        ),
        "curation": (
            "A deterministic benchmark query re-freezes the first query_limit "
            "scenario ids from the base snapshot as a new version-pinned snapshot."
        ),
        "storage": "Footprint uses on-disk bytes when local, plus selected payload/video byte details.",
        "comparison_skips": "Unavailable comparison formats are emitted as skipped report entries.",
        "enterprise_remote": (
            "enterprise-lance measures backend='enterprise' native training over a "
            "db:// or namespace-backed lake; enterprise_fixture_uri enables a "
            "fake-local db:// fixture for reproducible CI reports."
        ),
        "analytics_baselines": (
            "parquet and iceberg are opt-in analytics-lakehouse baselines over the "
            "same logical observation set. They separate metadata/table-scan latency "
            "(metadata_scan_latency) from Python/PyTorch payload hydration latency "
            "(row_hydration_latency), and report a shuffled-epoch throughput and a "
            "subset-filter-change rewrite cost. They are not penalized for robotics "
            "loader features they do not claim."
        ),
        "analytics_filter_change": (
            "For analytics baselines a changed subset filter forces a table "
            "rewrite/repack (Parquet) or a new snapshot with rewritten data files "
            "(Iceberg), recorded with materialized_bytes_written; Lance instead "
            "creates a new version-pinned row plan without rewriting payload bytes."
        ),
        "payload_placement": (
            "Analytics baselines record whether payloads are inline, referenced, or "
            "copied; the parquet/iceberg baselines store payload bytes inline at "
            "fixed source quality."
        ),
    }


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _path_size(path: str | Path) -> int | None:
    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return root.stat().st_size
    total = 0
    for child in root.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _random_indices(count: int, sample_count: int, seed: int) -> list[int]:
    if count <= 0 or sample_count <= 0:
        return []
    rng = random.Random(seed)
    return [rng.randrange(count) for _ in range(sample_count)]


def _random_frame_plans(
    count: int,
    *,
    clips: int,
    frames_per_clip: int,
    seed: int,
) -> list[list[int]]:
    if count <= 0 or clips <= 0 or frames_per_clip <= 0:
        return []
    rng = random.Random(seed + 17)
    plans: list[list[int]] = []
    for _ in range(clips):
        first = rng.randrange(count)
        plans.append([min(count - 1, first + offset) for offset in range(frames_per_clip)])
    return plans


def _random_frame_metric(
    plans: list[list[int]],
    latencies_ms: list[float],
    *,
    payload_bytes: int,
    access_path: str,
) -> dict[str, Any]:
    frames = sum(len(plan) for plan in plans)
    total_ms = sum(latencies_ms)
    return {
        "status": "completed",
        "value": _rate(frames, total_ms / 1000.0 if total_ms else 0.0),
        "unit": "frames/s",
        "details": {
            "clips": len(plans),
            "frames_per_clip": len(plans[0]) if plans else 0,
            "frames": frames,
            "plans_preview": plans[:8],
            "mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50_ms": _percentile(latencies_ms, 0.50),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "max_ms": max(latencies_ms) if latencies_ms else 0.0,
            "payload_bytes_materialized": int(payload_bytes),
            "access_path": access_path,
        },
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _elapsed_seconds(start_ns: int) -> float:
    return max((time.perf_counter_ns() - start_ns) / 1_000_000_000.0, 1e-9)


def _rate(count: int, seconds: float) -> float:
    return float(count) / seconds if seconds > 0 else 0.0


def _row_materialized_bytes(row: dict[str, Any]) -> int:
    total = 0
    for value in row.values():
        total += _materialized_bytes(value)
    return total


def _materialized_bytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode())
    if isinstance(value, memoryview):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_materialized_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_materialized_bytes(item) for item in value)
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return 8

    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return int(nbytes)
        except (TypeError, ValueError):
            pass

    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel()) * int(element_size())
        except (TypeError, ValueError):
            return 0
    return 0


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "snapshot"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
