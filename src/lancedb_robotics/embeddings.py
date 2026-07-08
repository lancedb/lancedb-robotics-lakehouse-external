"""Real, pluggable embedding providers + observation image embeddings (backlog 0021).

The baseline loop's only embedder was :class:`~lancedb_robotics.enrich.DemoEmbeddingProvider`,
which hashes a scenario's *structural descriptor* -- stable for fixtures, but
carrying no semantic or perceptual meaning, so vector/hybrid search ranked by
hash collisions. This module makes search real, behind the existing
:class:`~lancedb_robotics.enrich.EmbeddingProvider` contract:

- **Content-based, dependency-free providers** (the default upgrade path that
  runs everywhere, including CI): :class:`HashedTextEmbeddingProvider` embeds the
  *caption text* via signed feature hashing, so a query and a matching caption
  land near each other in one shared space; :class:`HashedImageEmbeddingProvider`
  embeds decoded image bytes so near-duplicate frames are near neighbors. These
  are deterministic stand-ins under the real contract -- semantic in content, not
  a structural hash, but with no model weights to download.
- **Model-backed providers behind the optional ``embeddings`` extra**
  (sentence-transformers, CLIP), lazily imported exactly like the decoder extras
  (backlog 0014/0017): absent extra -> a clear :class:`EmbeddingExtraMissing`,
  which :func:`resolve_embedding_provider` turns into a degrade-safe fallback to
  the demo provider with a notice, never a crash.

Multiple pluggable columns per table (decision 0025): re-embedding with a new
model is a *new column* with its own index, not a rewrite, so existing columns
and the snapshots pinned to them stay valid. :func:`embed_observations` writes
per-frame image vectors onto a *new* ``observations`` column additively -- the
table is blob-encoded (``payload_blob``), so in-place ``Table.update`` is
unavailable and the column is committed via ``drop_columns`` + ``add_columns`` on
the Lance dataset (the same additive/versioned write as quality, decision 0024),
never reading or rewriting the blob bytes for unrelated rows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata

import pyarrow as pa

from lancedb_robotics.blob import PAYLOAD_BLOB_COLUMN
from lancedb_robotics.enrich import (
    DEFAULT_EMBEDDING_DIMENSION,
    DemoEmbeddingProvider,
    EmbeddingProvider,
    EnrichmentError,
    ScenarioContext,
)
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import TRANSFORM_RUNS_SCHEMA

#: Default column for per-frame image embeddings on ``observations``. Per-column
#: naming keeps multiple models side by side (decision 0025), e.g. ``emb_img_clip``.
DEFAULT_IMAGE_EMBEDDING_COLUMN = "emb_image"

#: Observation modalities treated as camera frames (see ``extract._IMAGE``).
DEFAULT_IMAGE_MODALITIES: tuple[str, ...] = ("image",)

#: Dimension for the dependency-free content-based hashers (feature-hash width).
DEFAULT_CONTENT_DIMENSION = 64

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingExtraMissing(EnrichmentError):
    """Raised when a model-backed provider's optional ``embeddings`` extra is absent."""


def _import_extra(module: str, *, provider: str, hint: str = "") -> object:
    """Import a model dependency, mapping *both* failure modes to ``EmbeddingExtraMissing``.

    The optional ``embeddings`` extra is degrade-safe by contract (backlog 0021
    AC4): a model-backed provider must never crash the loop, it degrades to the
    demo provider with a notice. Two distinct failures must therefore funnel into
    the same :class:`EmbeddingExtraMissing` that :func:`resolve_embedding_provider`
    catches:

    1. the extra is absent (``find_spec`` is ``None``) -- the original case; and
    2. the package is *installed but fails to import* -- e.g. ``sentence-transformers``
       5.x eagerly importing ``torchcodec`` whose native dylib won't load on this
       host (BUG-09). A bare ``from X import Y`` would let that ``OSError`` /
       ``ImportError`` escape ``resolve_embedding_provider`` (which only catches
       ``EmbeddingExtraMissing``) and crash, defeating the degrade-safe contract.

    ``hint`` is appended to the broken-import message so the notice is actionable
    (which pin/extra to install). Returns the imported module.
    """
    if importlib.util.find_spec(module) is None:
        raise EmbeddingExtraMissing(
            f"embedding provider {provider!r} needs the optional dependency {module!r}; "
            "install it with `pip install 'lancedb-robotics[embeddings]'`"
        )
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        message = (
            f"embedding provider {provider!r} dependency {module!r} is installed but "
            f"failed to import ({type(exc).__name__}: {exc})"
        )
        if hint:
            message = f"{message}; {hint}"
        raise EmbeddingExtraMissing(message) from exc


#: Actionable pin for the BUG-09 torchcodec import failure, reused in the notice.
_SENTENCE_TRANSFORMERS_HINT = (
    "newer sentence-transformers (4.x/5.x) eagerly import torchcodec, whose native "
    "library may fail to load on this host; install a text-capable release with "
    "`pip install 'sentence-transformers>=3,<4'`"
)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec  # all-zero input (e.g. empty text/bytes): leave as the zero vector
    return [v / norm for v in vec]


# --- content-based, dependency-free providers (the everywhere default) ------


class HashedTextEmbeddingProvider(EmbeddingProvider):
    """Signed feature-hashing text embedder: content-based, deterministic, dep-free.

    Tokens of the text (the scenario caption for :meth:`embed`, the query for
    :meth:`embed_text`) are hashed into a fixed-width vector with signed buckets
    (the "hashing trick"), then L2-normalized. Text that shares tokens lands near
    in cosine space, and because :meth:`embed` and :meth:`embed_text` hash the
    *same way* they share one space -- so a query is a near neighbor of a caption
    about the same thing. Stable across processes (``hashlib``, not salted
    ``hash()``), so fixtures and snapshots are reproducible.
    """

    name = "hashed-text-v1"
    version = "1"

    def __init__(self, dimension: int = DEFAULT_CONTENT_DIMENSION) -> None:
        if dimension <= 0:
            raise EnrichmentError("embedding dimension must be positive")
        self.dimension = dimension

    def embed(self, ctx: ScenarioContext) -> list[float]:
        # Embed the dense, description-forward ``embed_text`` (BUG-07); fall back to
        # the summary, then topics, so the vector is never empty for a real window.
        text = ctx.embed_text or ctx.summary or " ".join(ctx.topics)
        return self._embed_text(text)

    def embed_text(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in _TOKEN_RE.findall((text or "").lower()):
            digest = hashlib.sha1(token.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 == 0 else -1.0
            vec[bucket] += sign
        return _l2_normalize(vec)


class HashedImageEmbeddingProvider(EmbeddingProvider):
    """Downsampled-byte image embedder: near-duplicate frames -> near neighbors.

    Decoded image bytes are pooled into ``dimension`` contiguous buckets (mean of
    each chunk, scaled to ``[0, 1]``) and L2-normalized. Two frames whose bytes
    are nearly identical pool to nearly identical vectors; an unrelated frame is
    farther -- enough to assert distance ordering deterministically without a
    model. A structural stand-in for a real CLIP image tower, not a perceptual
    embedder; it is image-only, so :meth:`embed` (scenario context) raises.
    """

    name = "hashed-image-v1"
    version = "1"

    def __init__(self, dimension: int = DEFAULT_CONTENT_DIMENSION) -> None:
        if dimension <= 0:
            raise EnrichmentError("embedding dimension must be positive")
        self.dimension = dimension

    def embed(self, ctx: ScenarioContext) -> list[float]:
        raise NotImplementedError(
            f"{self.name} is image-only; use embed_image on camera observations"
        )

    def embed_image(self, image: bytes) -> list[float]:
        size = len(image)
        if size == 0:
            return [0.0] * self.dimension
        vec: list[float] = []
        for i in range(self.dimension):
            lo = i * size // self.dimension
            hi = (i + 1) * size // self.dimension
            chunk = image[lo:hi]
            vec.append((sum(chunk) / len(chunk) / 255.0) if chunk else 0.0)
        return _l2_normalize(vec)


# --- model-backed providers behind the optional ``embeddings`` extra --------


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Real text embeddings via sentence-transformers (optional ``embeddings`` extra).

    ``embed`` and ``embed_text`` share the model's sentence space, so a free-text
    query genuinely ranks semantically-similar captions together. Lazily imported;
    absent extra raises :class:`EmbeddingExtraMissing` (caught by
    :func:`resolve_embedding_provider`).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        sentence_transformers = _import_extra(
            "sentence_transformers",
            provider="sentence-transformers",
            hint=_SENTENCE_TRANSFORMERS_HINT,
        )
        SentenceTransformer = sentence_transformers.SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers:{model_name}"
        self.version = model_name

    def embed(self, ctx: ScenarioContext) -> list[float]:
        # Embed the dense ``embed_text`` so the model sees scene semantics, not the
        # topic boilerplate that collapses every vector together (BUG-07).
        return self._encode(ctx.embed_text or ctx.summary or " ".join(ctx.topics))

    def embed_text(self, text: str) -> list[float]:
        return self._encode(text)

    def _encode(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vector]


class ClipEmbeddingProvider(EmbeddingProvider):
    """Real image+text CLIP embeddings (optional ``embeddings`` extra).

    A single shared CLIP space holds both towers: :meth:`embed_image` embeds a
    camera frame, :meth:`embed_text` / :meth:`embed` embed a query / caption, so
    "find scenes that look like X" works across the text query and the image
    column. Lazily imports ``open_clip``/``torch``/``PIL``; absent extra raises
    :class:`EmbeddingExtraMissing`.
    """

    def __init__(
        self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"
    ) -> None:
        open_clip = _import_extra("open_clip", provider="clip")
        torch = _import_extra("torch", provider="clip")
        _import_extra("PIL", provider="clip")  # imported per-frame in embed_image

        self._torch = torch
        # Use the GPU when present: CLIP image embedding over a large frame corpus is
        # the throughput-critical path, and a batched forward pass on CUDA is ~1-2
        # orders of magnitude faster than per-frame CPU calls.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model.eval()
        self._model.to(self._device)
        self._tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            probe = self._model.encode_text(self._tokenizer(["probe"]).to(self._device))
        self.dimension = int(probe.shape[1])
        self.name = f"clip:{model_name}/{pretrained}"
        self.version = f"{model_name}/{pretrained}"

    def embed(self, ctx: ScenarioContext) -> list[float]:
        # Embed the dense ``embed_text`` (scene description / VLM text) so the CLIP
        # text tower ranks by scene content, not the structural boilerplate (BUG-07).
        return self.embed_text(ctx.embed_text or ctx.summary or " ".join(ctx.topics))

    def embed_text(self, text: str) -> list[float]:
        with self._torch.no_grad():
            tokens = self._tokenizer([text]).to(self._device)
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(x) for x in features[0].cpu()]

    def embed_image(self, image: bytes) -> list[float]:
        # Single-frame convenience; the batched paths below are what scale uses.
        vec = self.embed_images([image])[0]
        return vec if vec is not None else [0.0] * self.dimension

    def embed_images(self, images: list[bytes | None]) -> list[list[float] | None]:
        """Batched CLIP embedding of *encoded* image bytes (JPEG/PNG)."""
        import io

        from PIL import Image

        pils: list = []
        for data in images:
            if not data:
                pils.append(None)
                continue
            try:
                pils.append(Image.open(io.BytesIO(data)).convert("RGB"))
            except Exception:  # noqa: BLE001 - undecodable frame -> None, never fatal
                pils.append(None)
        return self.embed_pil_images(pils)

    def embed_pil_images(self, images: list) -> list[list[float] | None]:
        """Batched CLIP embedding of already-decoded PIL images: one device forward pass.

        Returns one vector per input, ``None`` where the image is missing, so callers
        can align results back to ids. Preprocessing is per-frame on CPU; only the
        stacked ``encode_image`` runs on the (GPU) device.
        """
        tensors = []
        keep: list[int] = []
        for i, pil in enumerate(images):
            if pil is None:
                continue
            tensors.append(self._preprocess(pil))
            keep.append(i)
        results: list[list[float] | None] = [None] * len(images)
        if not tensors:
            return results
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        features = features.cpu()
        for slot, idx in enumerate(keep):
            results[idx] = [float(x) for x in features[slot]]
        return results


# --- pluggable registry + degrade-safe resolution ---------------------------


@dataclass(frozen=True)
class ProviderInfo:
    """Registry entry: how to build a provider, and what it can embed."""

    factory: Callable[..., EmbeddingProvider]
    requires_extra: bool
    modality: str  # "text" | "image" | "text+image"


def _demo_factory(dimension: int | None = None) -> EmbeddingProvider:
    return DemoEmbeddingProvider(dimension=dimension or DEFAULT_EMBEDDING_DIMENSION)


def _hashed_text_factory(dimension: int | None = None) -> EmbeddingProvider:
    return HashedTextEmbeddingProvider(dimension=dimension or DEFAULT_CONTENT_DIMENSION)


def _hashed_image_factory(dimension: int | None = None) -> EmbeddingProvider:
    return HashedImageEmbeddingProvider(dimension=dimension or DEFAULT_CONTENT_DIMENSION)


def _sentence_transformers_factory(dimension: int | None = None) -> EmbeddingProvider:
    return SentenceTransformerEmbeddingProvider()  # dimension comes from the model


def _clip_factory(dimension: int | None = None) -> EmbeddingProvider:
    return ClipEmbeddingProvider()  # dimension comes from the model


#: Name -> how to build it. Demo stays the deterministic default; the hashed
#: providers are the dependency-free real-contract upgrade; sentence-transformers
#: / clip are the model-backed providers behind the ``embeddings`` extra.
PROVIDER_REGISTRY: dict[str, ProviderInfo] = {
    "demo": ProviderInfo(_demo_factory, requires_extra=False, modality="text"),
    "hashed-text": ProviderInfo(_hashed_text_factory, requires_extra=False, modality="text"),
    "hashed-image": ProviderInfo(_hashed_image_factory, requires_extra=False, modality="image"),
    "sentence-transformers": ProviderInfo(
        _sentence_transformers_factory, requires_extra=True, modality="text"
    ),
    "clip": ProviderInfo(_clip_factory, requires_extra=True, modality="text+image"),
}

DEFAULT_PROVIDER = "demo"

#: Modules each model-backed provider imports in its ``__init__``. Used to probe
#: availability *without* constructing a provider (which would download/load a
#: model). Keep in sync with the provider ``__init__``s above.
_PROVIDER_EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "sentence-transformers": ("sentence_transformers",),
    "clip": ("open_clip", "torch", "PIL"),
}


def embedding_extra_available(name: str) -> bool:
    """Whether provider ``name``'s optional ``embeddings`` modules are importable.

    A spec-only probe (no model construction or download), so callers and tests can
    tell whether :func:`resolve_embedding_provider` would build the real provider or
    degrade to the demo fallback, without paying the model-load cost. A provider
    that requires no extra is always available; an unknown name is not. Note this
    can be ``True`` for a package that is installed but fails to import at runtime
    (the BUG-09 case) -- ``resolve_embedding_provider`` still degrades safely there.
    """
    info = PROVIDER_REGISTRY.get(name)
    if info is None:
        return False
    if not info.requires_extra:
        return True
    modules = _PROVIDER_EXTRA_MODULES.get(name, ())
    return bool(modules) and all(
        importlib.util.find_spec(module) is not None for module in modules
    )


#: Entry-point group installed packages use to contribute providers (backlog 0189
#: production path). Each entry point's name is the provider name and it loads to a
#: :class:`ProviderInfo`. Built-ins and explicit ``register()`` win over plugins.
EMBEDDING_PROVIDER_ENTRY_POINT_GROUP = "lancedb_robotics.embedding_providers"

_ENTRY_POINTS_LOADED = False


def load_provider_entry_points(*, force: bool = False) -> None:
    """Discover and register providers advertised via installed entry points.

    The production-path complement to the in-process :meth:`LakeEmbeddings.register`:
    a pip-installed package exposes a provider under
    :data:`EMBEDDING_PROVIDER_ENTRY_POINT_GROUP` and it becomes resolvable by name
    with no code change here. Degrade-safe: a missing group, an unreadable
    distribution, or a broken plugin is skipped, never raised, so a bad third-party
    plugin can't break resolution. Cached after the first sweep; ``force=True``
    re-scans. Built-ins and explicit ``register()`` entries are never overwritten.
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED and not force:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        eps = metadata.entry_points(group=EMBEDDING_PROVIDER_ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 - discovery is best-effort
        return
    for ep in eps:
        if ep.name in PROVIDER_REGISTRY:
            continue  # built-ins and explicit register() take precedence
        try:
            info = ep.load()
        except Exception:  # noqa: BLE001 - a broken plugin never breaks resolution
            continue
        if isinstance(info, ProviderInfo):
            PROVIDER_REGISTRY[ep.name] = info


def resolve_embedding_provider(
    name: str = DEFAULT_PROVIDER,
    *,
    dimension: int | None = None,
    fallback: str | None = DEFAULT_PROVIDER,
) -> tuple[EmbeddingProvider, str | None]:
    """Build the named provider; degrade safely to ``fallback`` if its extra is absent.

    Returns ``(provider, notice)``. ``notice`` is ``None`` on success, or a short
    human-readable message when a model-backed provider's ``embeddings`` extra was
    missing and we fell back to the demo provider (backlog 0021 AC4 -- never a
    crash, fixtures stay green). An unknown ``name`` is a usage error, not a
    fallback, so it raises.

    ``fallback=None`` is **strict mode** (backlog BUG-05/BUG-11): a missing model
    extra re-raises :class:`EmbeddingExtraMissing` instead of silently degrading to
    a demo/hash embedder, so a run never quietly produces fake vectors. The CLI
    ``--strict`` flag maps to ``fallback=None``.
    """
    load_provider_entry_points()  # register any installed third-party providers
    info = PROVIDER_REGISTRY.get(name)
    if info is None:
        raise EnrichmentError(
            f"unknown embedding provider {name!r}; choose from {sorted(PROVIDER_REGISTRY)}"
        )
    try:
        return info.factory(dimension=dimension), None
    except EmbeddingExtraMissing as exc:
        if fallback is None:
            raise  # strict: fail loud rather than silently use a demo embedder
        fallback_info = PROVIDER_REGISTRY[fallback]
        provider = fallback_info.factory(dimension=dimension)
        notice = f"{exc} -- falling back to the {provider.name!r} provider"
        return provider, notice


# --- observation image embeddings (a new column, written additively) --------


@dataclass(frozen=True)
class ObservationEmbeddingReport:
    """Summary of one observation image-embedding transform."""

    lake_uri: str
    transform_id: str
    column: str
    embedding_provider: str
    embedding_provider_version: str
    embedding_dimension: int
    modalities: tuple[str, ...]
    observations_embedded: int
    observations_skipped: int
    index: dict | None = None


def _stable_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def _decode_ros_image(raw: bytes | None, meta: dict | None):
    """Decode a raw ROS ``sensor_msgs/Image`` payload into a PIL RGB image.

    Camera topics differ across corpora: nuScenes ships JPEG (``CompressedImage``,
    which PIL opens directly), while didi ships raw ``bayer_grbg8`` buffers
    (``sensor_msgs/Image``) that PIL cannot open. This reconstructs the frame from
    the payload_json geometry (``height``/``width``/``encoding``/``step``) using
    numpy only -- no cv2. Bayer is demosaiced by 2x2-block averaging to a
    half-resolution RGB image, ample for CLIP's 224px input. ``rgb8``/``bgr8``/
    ``mono8`` are reshaped directly; encoded/unknown frames fall back to PIL.
    Returns ``None`` for absent/undecodable data (never raises).
    """
    if not raw:
        return None
    import io

    import numpy as np
    from PIL import Image

    meta = meta or {}
    enc = str(meta.get("encoding") or "").lower()
    h = int(meta.get("height") or 0)
    w = int(meta.get("width") or 0)
    step = int(meta.get("step") or 0)
    raw_pixel = enc.startswith("bayer") or enc in ("rgb8", "bgr8", "mono8", "8uc1")
    if not raw_pixel or not h or not w:
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")  # JPEG/PNG passthrough
        except Exception:  # noqa: BLE001 - not an encoded image either -> skip
            return None
    try:
        buf = np.frombuffer(raw, dtype=np.uint8)
        if enc.startswith("bayer"):
            row = step if step >= w else w
            if buf.size < row * h:
                return None
            grid = buf[: row * h].reshape(h, row)[:, :w]
            hh, ww = (h // 2) * 2, (w // 2) * 2
            g = grid[:hh, :ww].astype(np.uint16)
            cells = {
                "tl": g[0::2, 0::2],
                "tr": g[0::2, 1::2],
                "bl": g[1::2, 0::2],
                "br": g[1::2, 1::2],
            }
            chan: dict[str, list] = {"r": [], "g": [], "b": []}
            for ch, pos in zip(enc.replace("bayer_", "")[:4], ("tl", "tr", "bl", "br"), strict=False):
                if ch in chan:
                    chan[ch].append(cells[pos])

            def pick(ch):
                xs = chan[ch]
                return (sum(xs) // len(xs)) if xs else cells["tl"]

            rgb = np.stack([pick("r"), pick("g"), pick("b")], axis=-1).astype(np.uint8)
            return Image.fromarray(rgb, "RGB")
        if enc in ("rgb8", "bgr8"):
            row = step if step >= w * 3 else w * 3
            if buf.size < row * h:
                return None
            arr = buf[: row * h].reshape(h, row)[:, : w * 3].reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[:, :, ::-1]
            return Image.fromarray(np.ascontiguousarray(arr), "RGB")
        # mono8 / 8UC1
        row = step if step >= w else w
        if buf.size < row * h:
            return None
        return Image.fromarray(buf[: row * h].reshape(h, row)[:, :w], "L").convert("RGB")
    except Exception:  # noqa: BLE001 - any malformed frame -> skip, never fatal
        return None


# --- pluggable decoders + declarative pipeline spec (backlog 0189) -----------


def _geom(pj: str | None) -> dict | None:
    """Parse a ``payload_json`` string into the geometry the ROS image decoder needs."""
    if not pj:
        return None
    try:
        d = json.loads(pj)
    except Exception:  # noqa: BLE001 - no geometry -> PIL passthrough in the decoder
        return None
    return {k: d.get(k) for k in ("encoding", "height", "width", "step")}


def _decode_ros_image_row(raw: bytes | None, aux: dict) -> object | None:
    """Decoder-registry adapter: raw payload bytes + row ``aux`` -> PIL RGB image."""
    return _decode_ros_image(raw, _geom(aux.get("payload_json")))


def _identity_row(value: object, aux: dict) -> object:
    """Pass a column value through unchanged (bytes for image models, str for text)."""
    return value


@dataclass(frozen=True)
class DecoderInfo:
    """A registered decode step: turn a raw input column value into a model input.

    ``fn(value, aux) -> decoded`` where ``aux`` maps each name in ``aux_columns``
    to that row's value. ``output`` tells the engine which provider space to call:
    ``"pil"`` (batched ``embed_pil_images`` when the provider supports it),
    ``"image_bytes"`` (``embed_images``/``embed_image`` on the raw bytes), or
    ``"text"`` (``embed_text`` on the string value).
    """

    fn: Callable[..., object]
    output: str  # "pil" | "image_bytes" | "text"
    aux_columns: tuple[str, ...] = ()


#: Name -> decode step. ``ros_image`` reproduces the observation image path (raw
#: ROS/JPEG bytes -> PIL via ``payload_json`` geometry); ``image_bytes`` and
#: ``text`` are dependency-free passthroughs. New modalities register here.
DECODER_REGISTRY: dict[str, DecoderInfo] = {
    "ros_image": DecoderInfo(_decode_ros_image_row, "pil", ("payload_json",)),
    "image_bytes": DecoderInfo(_identity_row, "image_bytes"),
    "text": DecoderInfo(_identity_row, "text"),
}


@dataclass(frozen=True)
class Source:
    """Where an embedding pipeline reads its input rows from."""

    table: str = "observations"
    input: str = PAYLOAD_BLOB_COLUMN
    id: str = "observation_id"
    modality_column: str | None = "modality"
    modalities: tuple[str, ...] = DEFAULT_IMAGE_MODALITIES  # empty/None => all rows
    decoder: str = "ros_image"


@dataclass(frozen=True)
class EmbeddingSpec:
    """A declarative embedding-creation pipeline (backlog 0189).

    ``provider`` is either a registry name (resolved degrade-safely) or an already
    built :class:`EmbeddingProvider`. Running a spec writes ``target_column``
    additively and optionally builds its ANN index; multiple specs give multiple
    columns side by side (decision 0025).
    """

    provider: str | EmbeddingProvider
    target_column: str = DEFAULT_IMAGE_EMBEDDING_COLUMN
    source: Source = Source()
    index: object | None = None
    dimension: int | None = None
    image_batch_size: int = 256


def embed(
    lake: Lake,
    spec: EmbeddingSpec,
    *,
    created_by: str = "lancedb-robotics",
    checkpoint_file: str | None = None,
) -> ObservationEmbeddingReport:
    """Run a declarative :class:`EmbeddingSpec`: source -> decode -> provider -> column.

    Reads the source column (incl. blob-encoded payloads) lazily by id in
    ``image_batch_size`` batches, decodes via the registered decoder, embeds with
    the provider, and writes the vectors as a *new* fixed-size-list column
    additively (``drop_columns`` + ``add_columns``; a blob-encoded table cannot
    take an in-place ``Table.update``). Rows outside ``modalities`` or whose bytes
    are absent/undecodable map to NULL (never a crash). Streaming, resumable
    (``checkpoint_file``), bounded-memory. Pass ``spec.index`` to build the ANN
    index right after.
    """
    import os
    import tempfile

    import lance

    if isinstance(spec.provider, str):
        provider, _notice = resolve_embedding_provider(spec.provider, dimension=spec.dimension)
    else:
        provider = spec.provider
    src = spec.source
    column = spec.target_column
    blob_column = src.input
    modalities = tuple(src.modalities or ())
    modality_column = src.modality_column
    decoder = DECODER_REGISTRY.get(src.decoder)
    if decoder is None:
        raise EnrichmentError(
            f"unknown decoder {src.decoder!r}; choose from {sorted(DECODER_REGISTRY)}"
        )
    index = spec.index
    image_batch_size = spec.image_batch_size

    lance_table = lake.table(src.table)
    dataset = lance_table.to_lance()
    if dataset.count_rows() == 0:
        raise EnrichmentError(f"no {src.table} rows to embed in {lake.uri}; ingest a run first")
    if modality_column and modalities:
        meta = dataset.to_table(columns=[modality_column])
        target_count = sum(1 for m in meta[modality_column].to_pylist() if m in modalities)
    else:
        target_count = dataset.count_rows()

    dimension = provider.dimension
    vector_type = pa.list_(pa.float32(), dimension)
    use_pil = decoder.output == "pil" and hasattr(provider, "embed_pil_images")
    use_batched_bytes = decoder.output in ("pil", "image_bytes") and hasattr(
        provider, "embed_images"
    )
    use_text = decoder.output == "text"

    transform_key = {
        "kind": "embedding",
        "column": column,
        "embedding_provider": provider.name,
        "embedding_dimension": dimension,
        "modalities": list(modalities),
    }
    transform_id = f"tfm-embed-{_stable_digest(transform_key)}"
    if checkpoint_file is None:
        checkpoint_file = os.path.join(tempfile.gettempdir(), f"lr-{transform_id}.ckpt")

    # Streaming, resumable, bounded-memory materialization via a lance batch UDF (replaces the
    # former buffer-all: every vector held in RAM + one add_columns(pa.Table) commit, which OOMs
    # at scale). Lance reads ``read_columns`` (incl. the blob-encoded ``payload_blob``, which it
    # materializes as bytes) in ``image_batch_size`` batches, applies the UDF, writes new
    # per-fragment column files, and commits once at the end -- peak RSS scales with the batch,
    # not the corpus. ``checkpoint_file`` resumes a crashed run without recompute.
    # NOTE: a batch UDF cannot call back into Lance (nested runtime), so blobs must arrive via
    # ``read_columns`` (read-all) rather than a selective ``take_blobs``; for ``observations``
    # only ~4% of rows carry blobs, so the read-all cost is small. Selective/distributed variants
    # (a DataReplacement fragment loop, or Geneva ``backfill``) share this same UDF body.
    aux_cols = tuple(c for c in decoder.aux_columns if c in dataset.schema.names)

    @lance.batch_udf(
        output_schema=pa.schema([pa.field(column, vector_type)]),
        checkpoint_file=checkpoint_file,
    )
    def _embed_batch(batch: pa.RecordBatch):
        n = batch.num_rows
        cols = batch.schema.names
        mods = (
            batch[modality_column].to_pylist()
            if (modality_column and modality_column in cols)
            else [None] * n
        )
        values = batch[blob_column].to_pylist()
        # Only aux columns actually projected into ``read_columns`` reach the batch
        # (the PIL path adds them; other paths don't need them) -- guard like the
        # original ``"payload_json" in cols`` check so a non-PIL run never KeyErrors.
        aux = {c: batch[c].to_pylist() for c in aux_cols if c in cols}
        idxs = [i for i in range(n) if (not modalities) or mods[i] in modalities]
        out: list[list[float] | None] = [None] * n
        if idxs:
            # Prepare inputs for the modality the engine chose (decode PIL only for
            # providers that batch it), then hand the batch to the provider's one
            # ``embed_batch`` entry point -- the engine no longer calls
            # modality-specific methods, so a new provider overrides one method.
            if use_pil:
                prepared = [decoder.fn(values[i], {c: aux[c][i] for c in aux}) for i in idxs]
                kind = "pil"
            elif use_batched_bytes:
                prepared = [values[i] for i in idxs]
                kind = "image_bytes"
            elif use_text:
                prepared = [values[i] for i in idxs]
                kind = "text"
            else:
                prepared = [values[i] for i in idxs]
                kind = "image"
            vecs = provider.embed_batch(prepared, kind=kind)
            for i, v in zip(idxs, vecs, strict=True):
                out[i] = v
        return pa.record_batch([pa.array(out, type=vector_type)], [column])

    read_columns: list[str] = []
    if modality_column:
        read_columns.append(modality_column)
    read_columns.append(blob_column)
    if use_pil:
        read_columns.extend(aux_cols)
    if column in dataset.schema.names:
        dataset.drop_columns([column])  # a prior *completed* run -- recompute cleanly
    dataset.add_columns(_embed_batch, read_columns=read_columns, batch_size=image_batch_size)

    # Count from the committed column so the totals are correct even after a resumed run reused
    # checkpointed batches (the UDF is not re-invoked for those).
    emb_col = lake.table(src.table).to_lance().to_table(columns=[column])[column]
    embedded = len(emb_col) - emb_col.null_count
    skipped = max(0, target_count - embedded)

    index_params: dict | None = None
    if index is not None:
        from lancedb_robotics.indexing import IndexSpec, build_vector_index

        index_spec = index if isinstance(index, IndexSpec) else IndexSpec()
        index_params = build_vector_index(
            lake, table=src.table, column=column, spec=index_spec
        ).to_params()
    now = datetime.now(UTC)
    transforms = lake.table("transform_runs")
    transforms.delete(f"transform_id = '{transform_id}'")
    transforms.add(
        pa.Table.from_pylist(
            [
                {
                    "transform_id": transform_id,
                    "kind": "embedding",
                    "input_uris": [],
                    "input_table_versions": [],
                    "output_tables": [src.table],
                    "params": json.dumps(
                        {
                            **transform_key,
                            "embedding_provider_version": provider.version,
                            "observations_embedded": embedded,
                            "observations_skipped": skipped,
                            "index": index_params,
                        },
                        sort_keys=True,
                    ),
                    "status": "completed",
                    "started_at": now,
                    "finished_at": now,
                    "created_by": created_by,
                    "created_at": now,
                }
            ],
            schema=TRANSFORM_RUNS_SCHEMA,
        )
    )

    return ObservationEmbeddingReport(
        lake_uri=lake.uri,
        transform_id=transform_id,
        column=column,
        embedding_provider=provider.name,
        embedding_provider_version=provider.version,
        embedding_dimension=dimension,
        modalities=modalities,
        observations_embedded=embedded,
        observations_skipped=skipped,
        index=index_params,
    )


def embed_observations(
    lake: Lake,
    provider: EmbeddingProvider,
    *,
    column: str = DEFAULT_IMAGE_EMBEDDING_COLUMN,
    modalities: Sequence[str] = DEFAULT_IMAGE_MODALITIES,
    id_column: str = "observation_id",
    blob_column: str = PAYLOAD_BLOB_COLUMN,
    index: object | None = None,
    created_by: str = "lancedb-robotics",
    image_batch_size: int = 256,
    checkpoint_file: str | None = None,
) -> ObservationEmbeddingReport:
    """Embed camera observations' image payloads into a new vector ``column``.

    Back-compat shim over :func:`embed` (backlog 0189): builds the observation
    image :class:`EmbeddingSpec` and runs it, so existing callers, tests, and
    snapshots are unchanged. New pipelines should call ``lake.embeddings.embed``.
    """
    return embed(
        lake,
        EmbeddingSpec(
            provider=provider,
            target_column=column,
            source=Source(
                table="observations",
                input=blob_column,
                id=id_column,
                modality_column="modality",
                modalities=tuple(modalities),
                decoder="ros_image",
            ),
            index=index,
            image_batch_size=image_batch_size,
        ),
        created_by=created_by,
        checkpoint_file=checkpoint_file,
    )


class LakeEmbeddings:
    """SDK surface for pluggable embedding creation (``lake.embeddings``)."""

    def __init__(self, lake: Lake) -> None:
        self._lake = lake

    def providers(self) -> dict[str, dict]:
        """Registered providers with modality and (probe-only) availability."""
        load_provider_entry_points()  # include installed third-party providers
        return {
            name: {
                "modality": info.modality,
                "requires_extra": info.requires_extra,
                "available": embedding_extra_available(name),
            }
            for name, info in PROVIDER_REGISTRY.items()
        }

    def decoders(self) -> dict[str, str]:
        """Registered decode steps mapped to their output kind."""
        return {name: info.output for name, info in DECODER_REGISTRY.items()}

    def register(
        self,
        name: str,
        factory: Callable[..., EmbeddingProvider],
        *,
        modality: str,
        requires_extra: bool = False,
        extra_modules: Sequence[str] = (),
    ) -> None:
        """Register a provider at runtime (in-process -- the prototype path).

        The production path (backlog 0189) adds entry-point discovery for
        installed packages; this covers notebooks and in-repo experiments.
        """
        PROVIDER_REGISTRY[name] = ProviderInfo(factory, requires_extra, modality)
        if extra_modules:
            _PROVIDER_EXTRA_MODULES[name] = tuple(extra_modules)

    def register_decoder(
        self,
        name: str,
        fn: Callable[..., object],
        *,
        output: str,
        aux_columns: Sequence[str] = (),
    ) -> None:
        """Register a decode step (raw input value + row aux -> model input)."""
        DECODER_REGISTRY[name] = DecoderInfo(fn, output, tuple(aux_columns))

    def embed(
        self,
        spec: EmbeddingSpec,
        *,
        created_by: str = "lancedb-robotics",
        checkpoint_file: str | None = None,
    ) -> ObservationEmbeddingReport:
        """Run an :class:`EmbeddingSpec` against this lake."""
        return embed(self._lake, spec, created_by=created_by, checkpoint_file=checkpoint_file)


__all__ = [
    "DECODER_REGISTRY",
    "EMBEDDING_PROVIDER_ENTRY_POINT_GROUP",
    "DEFAULT_CONTENT_DIMENSION",
    "DEFAULT_IMAGE_EMBEDDING_COLUMN",
    "DEFAULT_IMAGE_MODALITIES",
    "DEFAULT_PROVIDER",
    "PROVIDER_REGISTRY",
    "ClipEmbeddingProvider",
    "DecoderInfo",
    "EmbeddingExtraMissing",
    "EmbeddingSpec",
    "HashedImageEmbeddingProvider",
    "HashedTextEmbeddingProvider",
    "LakeEmbeddings",
    "ObservationEmbeddingReport",
    "ProviderInfo",
    "SentenceTransformerEmbeddingProvider",
    "Source",
    "embed",
    "embed_observations",
    "embedding_extra_available",
    "load_provider_entry_points",
    "resolve_embedding_provider",
]
