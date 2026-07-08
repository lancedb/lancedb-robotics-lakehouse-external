"""CLI tests for `embed observations` (backlog 0187 -- the F3 embedding half).

The command wraps the streaming/resumable ``embed_observations`` seam, so these
tests pin the CLI contract: the column is populated with per-frame vectors,
non-target and payload-less rows map to NULL (never a crash), a re-run with the
same checkpoint file resumes instead of recomputing, and the ANN index build
follows the scale-aware type default (0186 policy) unless --index-type is given.
"""

import pyarrow as pa
from typer.testing import CliRunner

from lancedb_robotics import embeddings as emb
from lancedb_robotics.cli import app
from lancedb_robotics.lake import Lake
from lancedb_robotics.schemas import OBSERVATIONS_SCHEMA, RUNS_SCHEMA

runner = CliRunner()

_RUN_ID = "run-embed-cli"


def _image_obs_lake(path, *, images: int = 3) -> Lake:
    """A lake with ``images`` decodable camera rows, one blob-less image row, and
    one non-image row -- the NULL-safety fixture from the 0187 test-first plan."""
    lake = Lake.init(path)
    lake.table("runs").add(
        pa.Table.from_pylist(
            [{"run_id": _RUN_ID, "run_kind": "drive", "raw_uri": "/logs/embed.mcap"}],
            schema=RUNS_SCHEMA,
        )
    )
    rows = [
        {
            "observation_id": f"{_RUN_ID}:/cam:{i:06d}",
            "run_id": _RUN_ID,
            "topic": "/cam",
            "modality": "image",
            "payload_blob": bytes((i * 7 + j) % 256 for j in range(256)),
            "decode_status": "decoded",
            "raw_sequence": i,
        }
        for i in range(images)
    ]
    rows.append(
        {  # an image row with no payload: must map to NULL, never crash
            "observation_id": f"{_RUN_ID}:/cam:9999",
            "run_id": _RUN_ID,
            "topic": "/cam",
            "modality": "image",
            "payload_blob": None,
            "decode_status": "failed",
            "raw_sequence": images,
        }
    )
    rows.append(
        {  # a non-image row: outside --modality, must hold NULL
            "observation_id": f"{_RUN_ID}:/imu:000000",
            "run_id": _RUN_ID,
            "topic": "/imu",
            "modality": "imu",
            "payload_blob": b"\x01\x02\x03",
            "decode_status": "decoded",
            "raw_sequence": images + 1,
        }
    )
    lake.table("observations").add(pa.Table.from_pylist(rows, schema=OBSERVATIONS_SCHEMA))
    return lake


def _column_by_id(lake: Lake, column: str) -> dict:
    table = (
        lake.table("observations")
        .to_lance()
        .to_table(columns=["observation_id", column])
    )
    return dict(
        zip(table["observation_id"].to_pylist(), table[column].to_pylist(), strict=True)
    )


def test_embed_observations_populates_column_with_null_safety(tmp_path):
    lake = _image_obs_lake(tmp_path / "robot.lance")

    result = runner.invoke(
        app,
        [
            "embed", "observations", "--lake", str(tmp_path / "robot.lance"),
            "--provider", "hashed-image",
            "--checkpoint-file", str(tmp_path / "embed.ckpt"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "provider: hashed-image-v1 (dim 64)" in result.output
    assert "column: emb_image" in result.output
    assert "observations embedded: 3" in result.output
    assert "observations skipped: 1" in result.output  # the blob-less image row
    assert f"checkpoint: {tmp_path / 'embed.ckpt'}" in result.output
    # The scale-aware default (resolved in build_vector_index, 0186) picks
    # IVF_FLAT at this size, and the build then skips below the 256-row training
    # floor -- the skip reason names the resolved type, never silent.
    assert "vector index: skipped" in result.output
    assert "IVF_FLAT" in result.output

    vectors = _column_by_id(lake, "emb_image")
    for i in range(3):
        assert vectors[f"{_RUN_ID}:/cam:{i:06d}"] is not None
    assert vectors[f"{_RUN_ID}:/cam:9999"] is None
    assert vectors[f"{_RUN_ID}:/imu:000000"] is None


def test_embed_observations_resumes_a_crashed_run_from_checkpoint(tmp_path):
    """An interrupted run re-invoked with the same --checkpoint-file resumes:
    already-embedded batches are served from the checkpoint, not recomputed.
    (A *completed* run re-executed recomputes cleanly by design.)"""
    state = {"batches": 0, "fail": True}

    class FlakyProvider(emb.HashedImageEmbeddingProvider):
        name = "flaky-image-v1"

        def embed_batch(self, inputs, *, kind):
            state["batches"] += 1
            if state["fail"] and state["batches"] >= 2:
                raise RuntimeError("simulated crash mid-run")
            return super().embed_batch(inputs, kind=kind)

    emb.PROVIDER_REGISTRY["flaky-image"] = emb.ProviderInfo(
        factory=lambda dimension=None: FlakyProvider(dimension=dimension or 16),
        requires_extra=False,
        modality="image",
    )
    try:
        _image_obs_lake(tmp_path / "robot.lance")
        args = [
            "embed", "observations", "--lake", str(tmp_path / "robot.lance"),
            "--provider", "flaky-image", "--no-index",
            "--image-batch-size", "1",  # 5 rows -> 5 batches, crash on the 2nd
            "--checkpoint-file", str(tmp_path / "resume.ckpt"),
        ]

        first = runner.invoke(app, args)
        assert first.exit_code != 0  # crashed mid-run, nothing committed
        first_batches = state["batches"]
        assert first_batches >= 2

        state["fail"] = False
        second = runner.invoke(app, args)
        assert second.exit_code == 0, second.output
        assert "observations embedded: 3" in second.output
        # Resume, not restart: the checkpointed batch(es) from the crashed run were
        # served from the file, so the second pass invoked fewer than all 5 batches.
        second_batches = state["batches"] - first_batches
        assert second_batches < 5, f"expected a resumed run, got {second_batches} batches"
    finally:
        emb.PROVIDER_REGISTRY.pop("flaky-image", None)


def test_embed_observations_no_index(tmp_path):
    _image_obs_lake(tmp_path / "robot.lance")

    result = runner.invoke(
        app,
        [
            "embed", "observations", "--lake", str(tmp_path / "robot.lance"),
            "--no-index",
            "--checkpoint-file", str(tmp_path / "noindex.ckpt"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "vector index: none (--no-index)" in result.output
    assert "scale-aware default" not in result.output


def test_embed_observations_explicit_index_type_wins_over_scale_default(tmp_path):
    _image_obs_lake(tmp_path / "robot.lance")

    result = runner.invoke(
        app,
        [
            "embed", "observations", "--lake", str(tmp_path / "robot.lance"),
            "--index-type", "ivf_pq",
            "--checkpoint-file", str(tmp_path / "explicit.ckpt"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "scale-aware default" not in result.output  # explicit choice, no downgrade
    # Below the 256-row training floor the build itself still skips, and the skip
    # reason names the *requested* type.
    assert "vector index: skipped" in result.output
    assert "IVF_PQ" in result.output


def test_embed_observations_unknown_provider_fails(tmp_path):
    _image_obs_lake(tmp_path / "robot.lance")

    result = runner.invoke(
        app,
        ["embed", "observations", "--lake", str(tmp_path / "robot.lance"),
         "--provider", "no-such-provider"],
    )

    assert result.exit_code == 1
    assert "unknown embedding provider" in result.output
