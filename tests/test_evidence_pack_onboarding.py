"""Keep the AI Research Lead evidence pack honest (backlog 0139).

``docs/narratives/ai-research-lead-evidence-pack.md`` is the guided evaluation an
AI Research Lead runs to check the README's claims. That only works if it cannot
silently drift from the code: a broken link, a stale command snippet, or a missing
section would undermine the exact credibility the pack exists to build.

This was the "0138 guardrails where applicable" leg of 0139. It is deliberately
cheap — no optional heavy dependencies, no real corpus — so it runs on the default
dev suite. The parsing helpers it used (``iter_local_links``, ``iter_cli_invocations``,
``resolve_command_path``, ``heading_slugs``) were promoted into
``tests/doc_verify.py`` when backlog 0138 landed the shared README claim/link/
command registry; this test and ``test_readme_verification`` now share them.

It checks that, in the evidence pack:

* every local Markdown link resolves to an existing file, and every in-page anchor
  resolves to a real heading;
* every ``lancedb-robotics …`` snippet in a fenced code block parses to a command
  group + subcommand that exists in the live Typer CLI (including nested groups
  such as ``dataset snapshot create`` and ``train preview torch``);
* the required sections are present;
* no placeholder / transcript artifacts remain in prose.
"""

from __future__ import annotations

import pytest
from doc_verify import (
    ROOT,
    find_placeholder_artifacts,
    heading_slugs,
    iter_cli_invocations,
    iter_local_links,
    resolve_command_path,
)

DOC = ROOT / "docs" / "narratives" / "ai-research-lead-evidence-pack.md"

# Sections the pack promises (Scope / Acceptance Criteria of backlog 0139).
REQUIRED_HEADING_PREFIXES = [
    "## The claims",
    "## Guided mini-demo",
    "## Why not just build this on cloud services",
    "## Benchmark evidence",
    "## FAQ",
    "## Decision tree",
    "## What remains external",
]


@pytest.fixture(scope="module")
def doc_text() -> str:
    assert DOC.exists(), f"missing evidence pack {DOC}"
    return DOC.read_text(encoding="utf-8")


def test_required_sections_present(doc_text: str):
    heading_lines = [ln.strip() for ln in doc_text.splitlines() if ln.startswith("## ")]
    for prefix in REQUIRED_HEADING_PREFIXES:
        assert any(ln.startswith(prefix) for ln in heading_lines), (
            f"evidence pack is missing a required section starting with {prefix!r}"
        )


def test_local_links_resolve(doc_text: str):
    slugs = heading_slugs(doc_text)
    broken: list[str] = []
    for path, anchor in iter_local_links(doc_text):
        if not path:  # same-page anchor
            if anchor and anchor not in slugs:
                broken.append(f"#{anchor} (no matching heading)")
            continue
        target = (DOC.parent / path).resolve()
        if not target.exists():
            broken.append(path)
    assert not broken, "unresolved local links in evidence pack: " + ", ".join(sorted(broken))


def test_cli_snippets_reference_real_commands(doc_text: str):
    invocations = list(iter_cli_invocations(doc_text))
    assert invocations, "expected at least one lancedb-robotics command snippet"
    failures: list[str] = []
    saw_command = False
    for tokens in invocations:
        ok, detail = resolve_command_path(tokens)
        if not ok:
            failures.append(f"`lancedb-robotics {' '.join(tokens)}` -> {detail}")
        elif detail:
            saw_command = True
    assert not failures, "evidence pack references nonexistent CLI commands:\n" + "\n".join(failures)
    assert saw_command, "no command-group snippet was validated (parser regression?)"


def test_nested_command_paths_are_exercised(doc_text: str):
    """Guard the multi-level parser: the pack must show a 3-level command."""
    paths = {detail for tokens in iter_cli_invocations(doc_text)
             for ok, detail in [resolve_command_path(tokens)] if ok and detail}
    assert "dataset snapshot create" in paths, (
        "expected the pinned `dataset snapshot create` snippet; nested-command "
        "validation would not be exercised without it"
    )


def test_no_placeholder_artifacts(doc_text: str):
    found = find_placeholder_artifacts(doc_text)
    assert not found, f"placeholder/transcript artifacts left in prose: {found}"


def test_readme_links_to_evidence_pack():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "ai-research-lead-evidence-pack.md" in readme, (
        "README must link to the evidence pack (acceptance criterion)"
    )
