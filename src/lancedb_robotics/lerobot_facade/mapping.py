"""Declarative canonical state/action vector composition for the LeRobot facade.

MCAP/ROS ingest writes one ``observations`` row per raw message, and each
row's ``state_vector``/``action_vector`` length is fixed by message *type*
(``extract.py``'s per-topic layouts). There is no synchronized "one row per
timestep across all sensors" representation upstream. ``align.py`` already
solves that reconciliation problem -- ``lake.align.create_view(streams=[...])``
produces one row per tick in ``aligned_ticks``, with each requested stream's
own vector inside ``stream_values_json`` (surfaced by
:class:`~lancedb_robotics.training.AlignedFrameTrainingDataset` as a
``"streams"`` dict on each sample).

This module composes a fixed-width canonical ``observation.state``/``action``
vector by concatenating named streams' per-tick values in a declared order --
the sibling, at this aligned-tick read layer, of backlog 0254's
``state_key=(a, b)`` tuple-concatenation semantics (that backlog item targets
a different layer: new external-format ingest adapters at source-to-canonical
mapping time, not the alignment read path, and is not reused here).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalVectorMapping:
    """Declares which aligned streams compose a LeRobot frame's features.

    ``state_streams``/``action_streams`` are concatenated, in the given
    order, into one flat vector each (see :func:`compose_vector`).
    ``camera_streams`` declares which aligned streams are cameras -- each
    resolves to an ``observation.images.<key>`` feature via the video index
    (``lerobot_facade/videos.py``), not vector composition. At least one of
    the three must be non-empty.
    """

    state_streams: tuple[str, ...] = ()
    action_streams: tuple[str, ...] = ()
    camera_streams: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.state_streams or self.action_streams or self.camera_streams):
            raise ValueError(
                "CanonicalVectorMapping must declare at least one of "
                "state_streams/action_streams/camera_streams"
            )

    @property
    def streams(self) -> tuple[str, ...]:
        """All streams referenced by this mapping, de-duplicated, order-preserving."""
        seen: dict[str, None] = {}
        for stream in (*self.state_streams, *self.action_streams, *self.camera_streams):
            seen.setdefault(stream, None)
        return tuple(seen)


def validate_mapping(mapping: CanonicalVectorMapping, job_streams: Sequence[str]) -> None:
    """Raise ``ValueError`` naming any declared stream absent from the alignment.

    Fails fast, before any I/O -- an aligned tick can only ever carry a value
    for a stream the alignment was actually built with (the ``streams=[...]``
    passed to ``lake.align.create_view``).
    """
    available = set(job_streams)
    missing = [stream for stream in mapping.streams if stream not in available]
    if missing:
        raise ValueError(
            f"mapping declares streams not present in this alignment: {missing!r} "
            f"(alignment streams: {sorted(available)!r})"
        )


def compose_vector(
    stream_samples: Mapping[str, Mapping[str, object]],
    streams: tuple[str, ...],
) -> list[float] | None:
    """Concatenate ``streams``' values from one hydrated tick sample, in order.

    ``stream_samples`` is the ``"streams"`` dict on a sample returned by
    :class:`~lancedb_robotics.training.AlignedFrameTrainingDataset` --
    ``{stream: {"value": [...] | None, "status": ..., ...}}``. Returns
    ``None`` if ``streams`` is empty (no feature declared) or if any named
    component's value is missing (unaligned, out-of-tolerance, or filtered).

    Callers that want every yielded frame to have a complete vector should
    construct the underlying dataset with ``require_streams`` covering these
    streams -- that filters incomplete ticks out of the read plan entirely,
    upstream of this function. This is a defensive fallback, not the primary
    completeness gate.
    """
    if not streams:
        return None
    parts: list[float] = []
    for stream in streams:
        sample = stream_samples.get(stream)
        value = sample.get("value") if sample is not None else None
        if value is None:
            return None
        parts.extend(float(item) for item in value)
    return parts
