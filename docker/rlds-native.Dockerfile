FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV LANCEDB_ROBOTICS_REQUIRE_RLDS_NATIVE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tests ./tests

RUN uv sync --locked --extra dev --extra rlds
RUN uv run --no-sync python -c "import importlib.util, platform, rlds, reverb; assert platform.system() == 'Linux', platform.platform(); assert platform.machine() == 'x86_64', platform.machine(); assert importlib.util.find_spec('tensorflow_datasets') is not None"

CMD ["uv", "run", "--no-sync", "pytest", "-q", "-m", "rlds_native", "tests/test_projections.py", "tests/test_dataset_export.py"]
