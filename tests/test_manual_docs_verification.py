"""Hold the manual's hand-written chapters to the code (backlog 0519).

``tests/doc_verify.py`` (backlog 0138) already knows how to resolve local links,
walk ``lancedb-robotics`` snippets against the live Typer tree, and find
placeholder artifacts in prose — but only ``README.md`` and the AI Research Lead
evidence pack were ever held to it. Everything under ``docs/manual`` that a human
wrote (Tutorials, Concepts, Journeys) was ungated: a renamed CLI command or a
moved file rotted it silently, and ``tests/test_docs_reference_current.py`` only
covers the *generated* reference half.

Adding a tutorial into that blind spot is how a tutorial decays, so this test
gates it — and, since the machinery is dependency-free and already written, its
neighbours too.

Two kinds of check:

* **Per page** — links resolve (same-page anchors against that page's own
  headings), every ``lancedb-robotics`` snippet names a real command, and no
  ``TODO``/``FIXME``-class artifact survives in prose.
* **Navigation, both directions** — every hand-written page is reachable from
  ``mkdocs.yml``'s ``nav:`` *and* from ``docs/manual/index.md``, and every nav
  entry points at a file that exists. An orphaned page is a page nobody reads.

Cross-*file* anchor fragments (``other.md#section``) are validated to the file
only, matching ``test_readme_verification.py``'s documented scope; deeper
cross-file anchor validation is tracked as a follow-up.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from doc_verify import (
    ROOT,
    find_placeholder_artifacts,
    heading_slugs,
    iter_cli_invocations,
    iter_local_links,
    resolve_command_path,
)

MANUAL = ROOT / "docs" / "manual"
MKDOCS = ROOT / "mkdocs.yml"
INDEX = MANUAL / "index.md"

# ``reference/*.generated.md`` is owned by test_docs_reference_current.py, which
# re-renders it from the code; hand-editing it is already an error there.
GENERATED_DIR = MANUAL / "reference"


def _handwritten_pages() -> list[Path]:
    return sorted(
        path
        for path in MANUAL.rglob("*.md")
        if GENERATED_DIR not in path.parents and path != INDEX
    )


PAGES = _handwritten_pages()
PAGE_IDS = [str(path.relative_to(MANUAL)) for path in PAGES]


def _nav_targets(node: object) -> list[str]:
    """Every document path referenced anywhere in a mkdocs ``nav:`` tree."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [target for item in node for target in _nav_targets(item)]
    if isinstance(node, dict):
        return [target for value in node.values() for target in _nav_targets(value)]
    return []


@pytest.fixture(scope="module")
def nav_targets() -> list[str]:
    # mkdocs.yml uses no custom YAML tags in this repo's nav, so a safe load is
    # enough; if a theme ever adds one, this fails loudly rather than silently
    # skipping the coverage checks below.
    config = yaml.safe_load(MKDOCS.read_text(encoding="utf-8"))
    targets = _nav_targets(config.get("nav"))
    assert targets, "mkdocs.yml has no nav entries"
    return targets


@pytest.fixture(scope="module")
def index_links() -> set[str]:
    return {path for path, _anchor in iter_local_links(INDEX.read_text(encoding="utf-8")) if path}


# --- per page ---------------------------------------------------------------


def test_pages_were_discovered() -> None:
    assert PAGES, "no hand-written manual pages found (glob regression?)"


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_local_links_resolve(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    slugs = heading_slugs(text)
    broken: list[str] = []
    for target, anchor in iter_local_links(text):
        if not target:  # same-page anchor
            if anchor and anchor not in slugs:
                broken.append(f"#{anchor} (no matching heading in {page.name})")
            continue
        if not (page.parent / target).resolve().exists():
            broken.append(target)
    assert not broken, f"unresolved local links in {page.name}: " + ", ".join(sorted(broken))


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_cli_snippets_reference_real_commands(page: Path) -> None:
    failures: list[str] = []
    for tokens in iter_cli_invocations(page.read_text(encoding="utf-8")):
        ok, detail = resolve_command_path(tokens)
        if not ok:
            failures.append(f"`lancedb-robotics {' '.join(tokens)}` -> {detail}")
    assert not failures, f"{page.name} references nonexistent CLI commands:\n" + "\n".join(failures)


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_no_placeholder_artifacts(page: Path) -> None:
    found = find_placeholder_artifacts(page.read_text(encoding="utf-8"))
    assert not found, f"placeholder artifacts left in {page.name} prose: {found}"


# --- navigation, both directions --------------------------------------------


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_page_is_reachable(page: Path, nav_targets: list[str], index_links: set[str]) -> None:
    relative = str(page.relative_to(MANUAL))
    assert relative in nav_targets, (
        f"{relative} is not in mkdocs.yml nav; add it or the page ships unreachable"
    )
    assert relative in index_links, (
        f"{relative} is not linked from docs/manual/index.md; add it there too"
    )


def test_every_nav_entry_exists(nav_targets: list[str]) -> None:
    missing = [target for target in nav_targets if not (MANUAL / target).exists()]
    assert not missing, f"mkdocs.yml nav points at missing files: {missing}"


def test_index_links_point_at_real_files() -> None:
    text = INDEX.read_text(encoding="utf-8")
    slugs = heading_slugs(text)
    broken: list[str] = []
    for target, anchor in iter_local_links(text):
        if not target:
            if anchor and anchor not in slugs:
                broken.append(f"#{anchor}")
            continue
        if not (INDEX.parent / target).resolve().exists():
            broken.append(target)
    assert not broken, "unresolved local links in docs/manual/index.md: " + ", ".join(sorted(broken))


def test_tutorials_precede_journeys_in_the_nav(nav_targets: list[str]) -> None:
    """Tutorials teach a newcomer; journeys are the reference treatment of a task.

    Only the relative order is asserted -- where Concepts sits between them is a
    presentation choice, but a reference-grade journey appearing before the
    tutorial would put the ~47K ingest chapter in front of the on-ramp, which is
    the exact problem backlog 0519 was filed to fix.
    """
    tutorials = [i for i, target in enumerate(nav_targets) if target.startswith("tutorials/")]
    journeys = [i for i, target in enumerate(nav_targets) if target.startswith("journeys/")]
    assert tutorials, "the nav has no tutorials/ page"
    assert journeys, "the nav has no journeys/ pages (glob regression?)"
    assert min(tutorials) < min(journeys), (
        "a journeys/ page precedes every tutorials/ page in mkdocs.yml nav"
    )
