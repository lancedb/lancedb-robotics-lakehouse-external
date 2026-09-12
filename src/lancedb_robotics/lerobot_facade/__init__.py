"""Live LeRobot-shaped facade over a materialized alignment.

See ``reader.py`` for the design rationale: this reads directly from
``aligned_ticks`` via ``training.AlignedFrameTrainingDataset`` and, for
camera streams, ``videos``/``video_encodings`` via ``videos.VideoIndex``,
with no static export step and no changes to ``dataset_export.py``.

``views.publish_view`` (backlogs 0490/0491) is the front door for exposing a
facade configuration to LeRobot clients: it derives the ``meta/`` manifest —
including normalization statistics (``meta/stats.json``) — over a
version-pinned read of the lake and stores it in the canonical
``lerobot_views``/``lerobot_view_files`` tables, so a client pointed at
``root=<lake uri>`` materializes it through the same Lance connection.
``manifest.write_dataset_manifest`` (the original hand-run local writer) is
deprecated and no longer exported here.

``dataset_reader.py`` (a real ``lerobot.datasets.dataset_reader.
BaseDatasetReader`` implementation, so first-party lerobot tooling can open
this facade as a genuine ``LeRobotDataset``) is deliberately **not** imported
here: it hard-depends on upstream's storage-format registry, which does not
exist in any released ``lerobot`` version yet. Import it explicitly to opt
in: ``import lancedb_robotics.lerobot_facade.dataset_reader``.
"""

from .doctor import DoctorReport, LeRobotFacadeError, preflight
from .episodes import FacadeEpisode, build_episode_index
from .mapping import CanonicalVectorMapping, compose_vector, validate_mapping
from .reader import LiveLeRobotFacade, to_torch_dataset
from .stats import ViewStatsError, compute_view_stats, serialize_stats
from .videos import VideoIndex, camera_feature_key
from .view_lifecycle import (
    LatestPointerReconciliation,
    ViewCatalogCompactionError,
    ViewCatalogCompactionReport,
    compact_view_catalog,
)
from .view_retention import (
    OrphanViewFileReport,
    ProtectedViewError,
    ViewPinConformance,
    ViewReadinessReport,
    ViewRetentionReport,
    ViewRetirementError,
    ViewRetirementReport,
    apply_view_retention,
    plan_view_retention,
    reconcile_orphan_view_files,
    retire_view,
    view_pin_conformance,
    view_readiness,
    view_retention_pin_details,
)
from .views import (
    PinnedLake,
    PublishedView,
    StaleViewVersionError,
    ViewError,
    ViewNotFoundError,
    ViewPublishError,
    ViewsPage,
    get_view,
    list_view_pages,
    list_views,
    materialize_view,
    open_published_facade,
    publish_view,
    resolve_view_root,
)

__all__ = [
    "CanonicalVectorMapping",
    "DoctorReport",
    "FacadeEpisode",
    "LatestPointerReconciliation",
    "LeRobotFacadeError",
    "LiveLeRobotFacade",
    "OrphanViewFileReport",
    "PinnedLake",
    "ProtectedViewError",
    "PublishedView",
    "StaleViewVersionError",
    "VideoIndex",
    "ViewCatalogCompactionError",
    "ViewCatalogCompactionReport",
    "ViewError",
    "ViewNotFoundError",
    "ViewPinConformance",
    "ViewPublishError",
    "ViewReadinessReport",
    "ViewRetentionReport",
    "ViewRetirementError",
    "ViewRetirementReport",
    "ViewStatsError",
    "ViewsPage",
    "apply_view_retention",
    "build_episode_index",
    "camera_feature_key",
    "compact_view_catalog",
    "compose_vector",
    "compute_view_stats",
    "get_view",
    "list_view_pages",
    "list_views",
    "materialize_view",
    "open_published_facade",
    "plan_view_retention",
    "preflight",
    "publish_view",
    "reconcile_orphan_view_files",
    "resolve_view_root",
    "retire_view",
    "serialize_stats",
    "to_torch_dataset",
    "validate_mapping",
    "view_pin_conformance",
    "view_readiness",
    "view_retention_pin_details",
]
