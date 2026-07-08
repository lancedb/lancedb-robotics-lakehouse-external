"""Real caption providers and degrade-safe resolution (backlog 0022).

The demo caption provider is intentionally structural and deterministic. This
module adds selectable content-derived providers behind the same
``CaptionProvider`` contract:

- ``image-stats`` is dependency-free and captions camera payload bytes by their
  visual signal statistics. It is not a VLM, but it proves the real data path:
  decoded image ``payload_blob`` -> frame captions -> scenario summary.
- ``vlm-api`` is a generic credentialed HTTP hook for production VLM captioning.
  Missing endpoint/key degrades to the demo provider with a clear notice.
"""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from lancedb_robotics.enrich import (
    CaptionProvider,
    DemoCaptionProvider,
    EnrichmentError,
    ScenarioContext,
)

DEFAULT_CAPTION_PROVIDER = "demo"
_API_KEY_ENV = "LANCEDB_ROBOTICS_CAPTION_API_KEY"
_API_ENDPOINT_ENV = "LANCEDB_ROBOTICS_CAPTION_ENDPOINT"


class CaptionProviderUnavailable(EnrichmentError):
    """Raised when a real caption provider is not configured locally."""


def _topic_label(ctx: ScenarioContext) -> str:
    topics = sorted({obs.topic for obs in ctx.camera_observations})
    if topics:
        return ", ".join(topics)
    return ", ".join(ctx.topics) if ctx.topics else "no camera topics"


def _dominant(counter: Counter[str], fallback: str) -> str:
    if not counter:
        return fallback
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


class ImageStatisticsCaptionProvider(CaptionProvider):
    """Caption camera windows from decoded image bytes without optional deps.

    The provider aggregates simple frame-level visual terms (brightness,
    contrast, texture) from ``payload_blob``. That keeps CI hermetic while still
    exercising the important substrate path: the caption is a function of camera
    content, not scenario metadata boilerplate.
    """

    name = "image-statistics-v1"
    version = "1"
    uses_camera_payloads = True

    def caption(self, ctx: ScenarioContext) -> str:
        frames = [obs for obs in ctx.camera_observations if obs.payload_blob]
        camera_count = len(ctx.camera_observations)
        topic_label = _topic_label(ctx)
        if camera_count == 0:
            return f"no camera observations in {ctx.run_id} window"
        if not frames:
            return f"{camera_count} camera observations on {topic_label} with no decoded image bytes"

        brightness: Counter[str] = Counter()
        contrast: Counter[str] = Counter()
        texture: Counter[str] = Counter()
        for obs in frames:
            b, c, t = self._frame_terms(obs.payload_blob or b"")
            brightness[b] += 1
            contrast[c] += 1
            texture[t] += 1

        terms = (
            _dominant(brightness, "unknown-light"),
            _dominant(contrast, "unknown-contrast"),
            _dominant(texture, "unknown-texture"),
        )
        return f"{len(frames)} camera frames on {topic_label}: {' '.join(terms)} scene"

    def _frame_terms(self, payload: bytes) -> tuple[str, str, str]:
        if not payload:
            return ("empty", "unknown-contrast", "unknown-texture")
        sample = payload[:8192]
        mean = sum(sample) / len(sample)
        variance = sum((value - mean) ** 2 for value in sample) / len(sample)
        stdev = math.sqrt(variance)
        if mean < 85:
            brightness = "dark"
        elif mean > 150:
            brightness = "bright"
        else:
            brightness = "mid-tone"

        if stdev > 70:
            contrast = "high-contrast"
        elif stdev < 20:
            contrast = "low-contrast"
        else:
            contrast = "moderate-contrast"

        if len(sample) < 2:
            texture = "smooth"
        else:
            avg_delta = sum(
                abs(right - left) for left, right in zip(sample, sample[1:], strict=False)
            ) / (len(sample) - 1)
            if avg_delta > 50:
                texture = "striped"
            elif avg_delta > 15:
                texture = "textured"
            else:
                texture = "smooth"
        return (brightness, contrast, texture)


class VlmApiCaptionProvider(CaptionProvider):
    """Generic HTTP VLM caption hook configured by environment variables."""

    name = "vlm-api"
    uses_camera_payloads = True
    # A VLM caption is scene-semantic free text, so it is folded into the dense
    # ``embed_text`` for vector search alongside the scene description (BUG-07).
    caption_is_semantic = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv(_API_KEY_ENV)
        self.endpoint = endpoint or os.getenv(_API_ENDPOINT_ENV)
        self.timeout_s = timeout_s
        if not self.api_key or not self.endpoint:
            raise CaptionProviderUnavailable(
                "caption provider 'vlm-api' needs "
                f"{_API_ENDPOINT_ENV} and {_API_KEY_ENV}"
            )
        self.version = self.endpoint

    def caption(self, ctx: ScenarioContext) -> str:
        frames = [
            {
                "observation_id": obs.observation_id,
                "topic": obs.topic,
                "image_base64": base64.b64encode((obs.payload_blob or b"")[:256_000]).decode(
                    "ascii"
                ),
            }
            for obs in ctx.camera_observations[:4]
            if obs.payload_blob
        ]
        if not frames:
            return ImageStatisticsCaptionProvider().caption(ctx)

        payload = {
            "scenario": {
                "scenario_id": ctx.scenario_id,
                "run_id": ctx.run_id,
                "start_time_ns": ctx.start_time_ns,
                "end_time_ns": ctx.end_time_ns,
                "topics": list(ctx.topics),
            },
            "frames": frames,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode())
        caption = str(body.get("caption", "")).strip()
        if not caption:
            raise EnrichmentError("caption provider 'vlm-api' response did not include caption")
        return caption


@dataclass(frozen=True)
class CaptionProviderInfo:
    """Registry entry for selectable scenario caption providers."""

    factory: Callable[[], CaptionProvider]
    requires_configuration: bool


CAPTION_PROVIDER_REGISTRY: dict[str, CaptionProviderInfo] = {
    "demo": CaptionProviderInfo(DemoCaptionProvider, requires_configuration=False),
    "image-stats": CaptionProviderInfo(
        ImageStatisticsCaptionProvider, requires_configuration=False
    ),
    "image-statistics": CaptionProviderInfo(
        ImageStatisticsCaptionProvider, requires_configuration=False
    ),
    "vlm-api": CaptionProviderInfo(VlmApiCaptionProvider, requires_configuration=True),
}


def resolve_caption_provider(
    name: str = DEFAULT_CAPTION_PROVIDER,
    *,
    fallback: str | None = DEFAULT_CAPTION_PROVIDER,
) -> tuple[CaptionProvider, str | None]:
    """Build ``name``; degrade to ``fallback`` when a real provider is unavailable.

    ``fallback=None`` is **strict mode** (backlog BUG-05/BUG-11): an unavailable
    provider re-raises :class:`CaptionProviderUnavailable` instead of silently
    degrading to the demo captioner. The CLI ``--strict`` flag maps to it.
    """
    info = CAPTION_PROVIDER_REGISTRY.get(name)
    if info is None:
        raise EnrichmentError(
            f"unknown caption provider {name!r}; choose from {sorted(CAPTION_PROVIDER_REGISTRY)}"
        )
    try:
        return info.factory(), None
    except CaptionProviderUnavailable as exc:
        if fallback is None:
            raise  # strict: fail loud rather than silently use the demo captioner
        provider = CAPTION_PROVIDER_REGISTRY[fallback].factory()
        notice = f"{exc} -- falling back to the {provider.name!r} provider"
        return provider, notice


__all__ = [
    "CAPTION_PROVIDER_REGISTRY",
    "DEFAULT_CAPTION_PROVIDER",
    "CaptionProviderInfo",
    "CaptionProviderUnavailable",
    "ImageStatisticsCaptionProvider",
    "VlmApiCaptionProvider",
    "resolve_caption_provider",
]
