"""Regenerate tests/fixtures/teleop.mcap -- the tutorial's offline on-ramp (backlog 0519).

Run: uv run python tests/fixtures/make_teleop_mcap.py

Every other bundled MCAP fixture is a *decode-family probe*: ``sample.mcap``
carries one json ``/imu`` channel and one cbor ``/camera/front`` channel because
what it tests is the decoder registry. None of them is robot-shaped, so nothing
in the tree could carry a reader from ``ingest`` all the way to a training batch:
the alignment engine and the LeRobot facade are both built around a *state and
action stream pair* sampled on a common clock, and no fixture had one.

This is that fixture. It is the smallest thing that is still honestly the real
shape:

* **One run, three streams, one clock.** A 6-DoF arm's observed joint positions
  (``/joint-positions``), its gripper aperture (``/gripper``), and the commanded
  joint targets (``/action``) -- all at 20 Hz over ``TOTAL_SECONDS`` seconds.
* **Real typed extraction, no special case.** The channels use the schema names
  ``RobotState`` and ``GripperState``, which ``extract.py`` already routes (they
  are XDOF ABC-130k's own names). ``/joint-positions`` and ``/action`` therefore
  land 6-float ``state_vector``s and ``/gripper`` a 1-float one, exactly as the
  protobuf originals do -- ``extract()`` has no topic argument, so observed and
  commanded are separated downstream *by stream name*, which is precisely what a
  ``CanonicalVectorMapping`` does. A 6+1 state composition against a 6-wide
  action is also deliberately heterogeneous: mismatched per-topic vector widths
  are the bug the facade exists to prevent, so the tutorial's happy path
  exercises it.
* **json payloads in zstd chunks.** Both are handled by the base install (no
  optional extra to resolve before reaching a training batch), and zstd chunking
  is what the real corpora use -- so the fixture pays the same per-chunk
  decompression the README names as the reason raw logs are a poor training
  substrate, rather than quietly sidestepping it.
* **Values that actually move.** Every tick differs, and ``/action`` leads
  ``/joint-positions`` by one tick (it is a *command*, not an echo). A
  ``delta_timestamps`` action chunk over a constant-valued fixture looks
  identical whether or not the windowing works; backlog 0509 hit exactly that
  and had to grow a separate varying-value fixture to test against.
* **Fixed timestamps.** The epoch and the trajectory are closed-form, so
  regenerating reproduces the file byte for byte and inspection metadata stays
  stable.

Two ``--window 2s`` scenario windows fall out of ``TOTAL_SECONDS`` -- the facade
buckets aligned ticks into episodes by physical episode if one exists and by
scenario window otherwise, so the tutorial gets two 40-frame episodes without a
promotion step. Episode boundaries are what make an action chunk's padding
visible at the tail of an episode.
"""

import json
import math
from pathlib import Path

from mcap.writer import CompressionType, Writer

BASE_NS = 1_700_000_000_000_000_000  # fixed epoch so the fixture is deterministic
RATE_HZ = 20
TOTAL_SECONDS = 4.0
PERIOD_NS = int(1e9 / RATE_HZ)
SAMPLES = int(TOTAL_SECONDS * RATE_HZ)
JOINTS = 6

OUT = Path(__file__).parent / "teleop.mcap"


def _joint_positions(step: int) -> list[float]:
    """A smooth, closed-form 6-DoF trajectory -- deterministic, never constant."""
    phase = 2 * math.pi * (step / SAMPLES)
    return [round(0.1 * j + 0.5 * math.sin(phase + 0.3 * j), 6) for j in range(JOINTS)]


def _gripper_aperture(step: int) -> float:
    """Aperture ramps 0 -> 1 across each 2-second window, then repeats."""
    within = step % (SAMPLES // 2)
    return round(within / ((SAMPLES // 2) - 1), 6)


def main() -> None:
    with OUT.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.ZSTD)
        writer.start(profile="", library="lancedb-robotics-fixture")

        # `RobotState` and `GripperState` are the real schema names extract.py
        # routes (see `_BY_SCHEMA`); the json schema record is informational.
        robot_state_schema = writer.register_schema(
            name="RobotState",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "position": {"type": "array", "items": {"type": "number"}}
                    },
                }
            ).encode(),
        )
        gripper_schema = writer.register_schema(
            name="GripperState",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "position": {"type": "array", "items": {"type": "number"}}
                    },
                }
            ).encode(),
        )

        joints_channel = writer.register_channel(
            topic="/joint-positions",
            message_encoding="json",
            schema_id=robot_state_schema,
        )
        gripper_channel = writer.register_channel(
            topic="/gripper",
            message_encoding="json",
            schema_id=gripper_schema,
        )
        action_channel = writer.register_channel(
            topic="/action",
            message_encoding="json",
            schema_id=robot_state_schema,
        )

        for step in range(SAMPLES):
            time_ns = BASE_NS + step * PERIOD_NS
            for channel_id, payload in (
                (joints_channel, {"position": _joint_positions(step)}),
                (gripper_channel, {"position": [_gripper_aperture(step)]}),
                # The command leads the observation by one tick: an action is a
                # target, not an echo of where the arm already is.
                (action_channel, {"position": _joint_positions(step + 1)}),
            ):
                writer.add_message(
                    channel_id=channel_id,
                    log_time=time_ns,
                    publish_time=time_ns,
                    data=json.dumps(payload, separators=(",", ":")).encode(),
                )

        writer.finish()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
