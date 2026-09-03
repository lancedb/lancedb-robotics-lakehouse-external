"""Preflight validation for the live LeRobot facade.

Runs a small set of checks before a facade starts serving frames -- borrowing
the "doctor as pre-flight, not afterthought" idea validated against real
corpora by the ``lerobot-lance-doctor`` tool this facade design was informed
by: cheap, read-only checks that catch a silently wrong/empty dataset before
a training loop ever sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .episodes import FacadeEpisode
from .mapping import CanonicalVectorMapping, validate_mapping


class LeRobotFacadeError(Exception):
    """Raised when the live LeRobot facade cannot serve the requested data."""


@dataclass(frozen=True)
class DoctorReport:
    alignment_id: str
    empty_episodes: tuple[str, ...] = field(default_factory=tuple)
    gappy_episodes: tuple[str, ...] = field(default_factory=tuple)

    def raise_if_unusable(self) -> None:
        """Raise on hard failures. Gaps are reported but not fatal."""
        if self.empty_episodes:
            raise LeRobotFacadeError(
                f"alignment {self.alignment_id!r} resolves zero ticks for episodes "
                f"{list(self.empty_episodes)!r} -- the alignment's time range does "
                "not cover these episodes, or every one of their ticks was filtered "
                "out by the quality policy (require_streams/min_confidence/statuses)"
            )


def preflight(
    job: dict[str, Any],
    mapping: CanonicalVectorMapping,
    episodes: tuple[FacadeEpisode, ...],
) -> DoctorReport:
    """Validate a resolved alignment job + mapping + episode index together.

    ``job`` must already come from a lookup that guarantees materialization
    (e.g. ``training._resolve_alignment_job``, which raises if the alignment
    was never written to ``aligned_ticks``/``aligned_frames``) -- this
    function does not re-check that.
    """
    validate_mapping(mapping, job.get("streams") or ())
    empty = tuple(episode.episode_id for episode in episodes if not episode.tick_indices)
    gappy = tuple(
        episode.episode_id
        for episode in episodes
        if episode.tick_indices
        and (episode.tick_indices[-1] - episode.tick_indices[0] + 1) > len(episode.tick_indices)
    )
    return DoctorReport(
        alignment_id=str(job["alignment_id"]), empty_episodes=empty, gappy_episodes=gappy
    )
