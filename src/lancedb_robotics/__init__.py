"""LanceDB Robotics: multimodal data lakehouse substrate for Physical AI pipelines."""

from lancedb_robotics.connections import (
    create_namespace_client as namespace_client,
)
from lancedb_robotics.connections import (
    lance_dataset,
    namespace_worker_spec,
    pylance_namespace_access,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.review_connectors import (
    JsonFileReviewToolConnector,
    ReviewConnectorResult,
    ReviewConnectorTask,
    ReviewToolConnector,
)
from lancedb_robotics.scoring import (
    ActiveLearningScorer,
    Calibration,
    CompositeScorer,
    ConfidenceMarginScorer,
    DistributionGapBoostScorer,
    EnsembleDisagreementScorer,
    EntropyScorer,
    FailureSeverityScorer,
    HighLossScorer,
    LabelCostPenaltyScorer,
    MissingLabelsScorer,
    ScorerResult,
    ScoringCandidate,
    builtin_scorer_names,
    register_scorer,
    resolve_scorer,
)

__version__ = "0.0.1"


def connect(uri=None, **kwargs):
    """Open an existing robotics lake through the typed connection resolver."""
    return Lake.open(uri, **kwargs)


__all__ = [
    "Lake",
    "JsonFileReviewToolConnector",
    "ReviewConnectorResult",
    "ReviewConnectorTask",
    "ReviewToolConnector",
    "ActiveLearningScorer",
    "Calibration",
    "CompositeScorer",
    "ConfidenceMarginScorer",
    "DistributionGapBoostScorer",
    "EnsembleDisagreementScorer",
    "EntropyScorer",
    "FailureSeverityScorer",
    "HighLossScorer",
    "LabelCostPenaltyScorer",
    "MissingLabelsScorer",
    "ScorerResult",
    "ScoringCandidate",
    "builtin_scorer_names",
    "register_scorer",
    "resolve_scorer",
    "__version__",
    "connect",
    "lance_dataset",
    "namespace_client",
    "namespace_worker_spec",
    "pylance_namespace_access",
]
