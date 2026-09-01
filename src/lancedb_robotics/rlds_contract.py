"""Shared RLDS field names used by import and export projections."""

from __future__ import annotations

RLDS_CANONICAL_CONTRACT_VERSION = "rlds-canonical-v1"
RLDS_EPISODE_METADATA_KEY = "episode_metadata"
RLDS_STEPS_KEY = "steps"
RLDS_OBSERVATION_KEY = "observation"
RLDS_ACTION_KEY = "action"
RLDS_REWARD_KEY = "reward"
RLDS_DISCOUNT_KEY = "discount"
RLDS_IS_FIRST_KEY = "is_first"
RLDS_IS_LAST_KEY = "is_last"
RLDS_IS_TERMINAL_KEY = "is_terminal"
RLDS_STEP_METADATA_KEY = "metadata"

RLDS_REQUIRED_MARKER_KEYS = (RLDS_IS_FIRST_KEY, RLDS_IS_LAST_KEY)
RLDS_STANDARD_STEP_FIELDS = (
    RLDS_OBSERVATION_KEY,
    RLDS_ACTION_KEY,
    RLDS_REWARD_KEY,
    RLDS_DISCOUNT_KEY,
    RLDS_IS_FIRST_KEY,
    RLDS_IS_LAST_KEY,
    RLDS_IS_TERMINAL_KEY,
    RLDS_STEP_METADATA_KEY,
)
