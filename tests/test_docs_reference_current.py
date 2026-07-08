"""Fail if the manual's generated reference pages drift from the code (0288).

The reference half of ``docs/manual`` is produced by ``scripts/gen_docs_reference.py``
from the live Typer command tree, the ``lake.*`` namespaces, and ``TABLE_SCHEMAS``.
This test re-renders each page in-memory and compares it to the checked-in file, so
a change to a CLI command, a public ``lake.*`` method, or a canonical table fails
the suite until the author regenerates:

    python scripts/gen_docs_reference.py

That is the enforcement behind the ``doc-sync`` skill: an advisory recipe can be
skipped, a failing test cannot.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "gen_docs_reference.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_docs_reference", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GEN = _load_generator()


@pytest.mark.parametrize("filename", sorted(_GEN.GENERATED_FILES))
def test_generated_reference_page_is_current(filename):
    render = _GEN.GENERATED_FILES[filename]
    expected = render()
    path = _GEN.REFERENCE_DIR / filename
    assert path.exists(), (
        f"missing generated reference page {path}; run `python scripts/gen_docs_reference.py`"
    )
    actual = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"docs/manual/reference/{filename} is stale; "
        "run `python scripts/gen_docs_reference.py` to regenerate it"
    )


def test_reference_pages_are_regenerated_deterministically():
    # Rendering twice must be byte-identical, or the drift test would be flaky.
    for render in _GEN.GENERATED_FILES.values():
        assert render() == render()
