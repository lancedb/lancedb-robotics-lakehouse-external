FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV LANCEDB_ROBOTICS_REQUIRE_RLDS_NATIVE=1 \
    TFDS_DISABLE_GCS=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tests ./tests

RUN uv sync --locked --extra dev --extra tfds
RUN uv run --no-sync python -c "import tensorflow as tf, tensorflow_datasets as tfds; assert tf.__version__.startswith('2.21.'), tf.__version__; assert tfds.__version__ == '4.9.10', tfds.__version__"

CMD ["uv", "run", "--no-sync", "pytest", "-q", "-m", "rlds_native", "tests/test_rlds_adapter.py"]
