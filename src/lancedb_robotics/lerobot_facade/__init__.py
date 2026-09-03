"""Live LeRobot-shaped facade over a materialized alignment.

See ``reader.py`` for the design rationale: this reads directly from
``aligned_ticks`` via ``training.AlignedFrameTrainingDataset`` and, for
camera streams, ``videos``/``video_encodings`` via ``videos.VideoIndex``,
with no static export step and no changes to ``dataset_export.py``.
"""

from .doctor import DoctorReport, LeRobotFacadeError, preflight
from .episodes import FacadeEpisode, build_episode_index
from .mapping import CanonicalVectorMapping, compose_vector, validate_mapping
from .reader import LiveLeRobotFacade, to_torch_dataset
from .videos import VideoIndex, camera_feature_key

__all__ = [
    "CanonicalVectorMapping",
    "DoctorReport",
    "FacadeEpisode",
    "LeRobotFacadeError",
    "LiveLeRobotFacade",
    "VideoIndex",
    "build_episode_index",
    "camera_feature_key",
    "compose_vector",
    "preflight",
    "to_torch_dataset",
    "validate_mapping",
]
