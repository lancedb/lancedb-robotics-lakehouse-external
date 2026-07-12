"""Guard the cheapest-to-detect anti-patterns from SKILLS.md so they can't
regress silently (BUG-01/06/13/14/15).

These are deliberately narrow, high-precision, zero-LLM checks -- static scans
over tracked source plus one config-coverage assertion, not an attempt to
enforce every rule in SKILLS.md. Per SKILLS.md's own testing-skill rule ("pin
every guardrail with a regression test, or it will regress" -- BUG-11), this
file is the enforcement half of a few of its rules; the rest still relies on
the pre-merge checklist and review.

If a check here starts failing because of a genuine, reviewed exception (not a
regression), update the relevant baseline/allowlist in this file with a
comment explaining why -- do not delete the check.
"""

from __future__ import annotations

import re
from pathlib import Path

from lancedb_robotics import indexing

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "lancedb_robotics"

# --------------------------------------------------------------------------
# BUG-01 / BUG-06 / BUG-13 / BUG-14: observations (352K rows), lineage_edges
# (1.29M rows), and lineage_artifacts (704K rows) are the three tables
# SKILLS.md documents as too large to fully materialize. A full, unfiltered
# `.to_arrow().to_pylist()` over one of them is the exact shape those bugs
# shipped as. This baseline is today's reviewed count per file -- it does not
# need to hit zero, but it must not grow without a conscious update here (and,
# ideally, a scoping fix instead of a bigger baseline).
# --------------------------------------------------------------------------
_WATCHED_TABLES = ("observations", "lineage_edges", "lineage_artifacts")
_FULL_TABLE_MATERIALIZATION_BASELINE = {
    "curate.py": 2,
    "episodes.py": 3,
    "ingest.py": 1,
    "lineage.py": 10,
    "lineage_integrations.py": 2,
    "quality.py": 1,
    "scenarios.py": 1,
    "video.py": 1,
    "writeback.py": 1,
}
_FULL_TABLE_MATERIALIZATION_RE = re.compile(
    r'\.table\(\s*["\'](?:' + "|".join(_WATCHED_TABLES) + r')["\']\s*\)'
    r"\s*\.to_arrow\(\)\s*\.to_pylist\(\)"
)


def _count_full_table_materializations(path: Path) -> int:
    return len(_FULL_TABLE_MATERIALIZATION_RE.findall(path.read_text(encoding="utf-8")))


def test_no_new_full_table_materialization_of_watched_tables():
    current: dict[str, int] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        count = _count_full_table_materializations(path)
        if count:
            key = str(path.relative_to(SRC_ROOT))
            current[key] = current.get(key, 0) + count

    regressions = {
        name: (_FULL_TABLE_MATERIALIZATION_BASELINE.get(name, 0), count)
        for name, count in current.items()
        if count > _FULL_TABLE_MATERIALIZATION_BASELINE.get(name, 0)
    }
    assert not regressions, (
        "New unfiltered `.to_arrow().to_pylist()` call(s) on a full "
        f"{_WATCHED_TABLES} table (baseline -> found): {regressions}. This is "
        "the exact shape of BUG-01/06/13/14 (see SKILLS.md's 'Resource usage' "
        "and 'bounded graph and lineage traversal' sections). Scope the read "
        "to referenced rows or take by row id instead of materializing the "
        "whole table -- or, if this is a reviewed, deliberately bounded "
        "exception, update `_FULL_TABLE_MATERIALIZATION_BASELINE` above with a "
        "comment explaining why."
    )


# --------------------------------------------------------------------------
# BUG-06 (round 2): a hardcoded `observation_id IN (...)` predicate blew up the
# Lance/DataFusion query planner past 6 GB on a 178K-id snapshot, independent
# of any scalar index -- chunking would not have fixed it, only not building it
# did. Today's baseline is zero; this must never regress.
# --------------------------------------------------------------------------
_HARDCODED_OBSERVATION_ID_IN_RE = re.compile(r"""f["']observation_id IN \(""")


def test_no_hardcoded_observation_id_in_predicate():
    offenders = [
        str(path.relative_to(SRC_ROOT))
        for path in sorted(SRC_ROOT.rglob("*.py"))
        if _HARDCODED_OBSERVATION_ID_IN_RE.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"Found a hardcoded `f\"observation_id IN (...)\"` predicate in {offenders}. "
        "BUG-06 (round 2) removed exactly this pattern. Resolve to row ids and "
        "`take_row_ids` instead (see SKILLS.md's 'Resolve to row ids and take' rule)."
    )


# --------------------------------------------------------------------------
# BUG-15: observations and the lineage graph had zero scalar indexes, so every
# filter behind other fixes was a full scan. Guard that the three tables stay
# covered in the predicate-index config, regardless of future refactors to
# indexing.py.
# --------------------------------------------------------------------------
def test_bug15_watched_tables_still_have_predicate_indexes():
    for table in _WATCHED_TABLES:
        columns = indexing.PREDICATE_INDEX_COLUMNS_BY_TABLE.get(table)
        assert columns, (
            f"{table!r} has no entry in indexing.PREDICATE_INDEX_COLUMNS_BY_TABLE -- "
            "this is the exact regression BUG-15 fixed (a full scan behind every "
            "filter on this table). See SKILLS.md's 'New filter/join-key columns "
            "need a scalar index' rule."
        )
