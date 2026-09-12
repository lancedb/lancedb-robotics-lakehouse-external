"""Keep the root README trustworthy for a cold AI Research Lead (backlog 0138).

``README.md`` is the executive technical entry point (backlog 0080): it names
implemented / evolving / planned capabilities, links local docs, and shows
runnable ``lancedb-robotics`` snippets. As the SDK and backlog keep moving, that
surface drifts unless something deterministic holds it to the code. A broken
local link, a stale command snippet, a vanished section, or a status claim with
no evidence would undermine the exact credibility backlog 0080 is building.

This is that guardrail. It is deliberately cheap — no optional heavy
dependencies, no real corpus — so it runs on the default ``uv run pytest`` dev
path. The parsing helpers live in ``tests/doc_verify.py`` (promoted from the
0139 evidence-pack test); the claim-evidence registry lives in
``docs/readme-claim-evidence.toml`` (see that file's header for how contributors
update it when roadmap work ships).

Scope note: cross-*file* anchor fragments (``other.md#section``) are validated
only to the file, not the anchor — matching the evidence-pack verifier. Deeper
cross-file anchor validation is tracked as its own follow-up.
"""

from __future__ import annotations

import pytest
from doc_verify import (
    ROOT,
    find_placeholder_artifacts,
    heading_slugs,
    iter_cli_invocations,
    iter_local_links,
    iter_status_claims,
    load_claim_registry,
    resolve_command_path,
    resolve_evidence,
    section_body,
)

README = ROOT / "README.md"
REGISTRY = ROOT / "docs" / "readme-claim-evidence.toml"

# The README's own headings (backlog 0080 layout). These are the real section
# titles — the backlog paraphrases them as "What You Get Today" / "What Remains
# External"; match the shipped wording by a distinctive prefix.
REQUIRED_SECTION_PREFIXES = [
    "Who this is for",
    "Quickstart",
    "Train from the lake",  # runnable ingest -> align -> publish -> batch path
    "What you get",         # feature-breadth status list
    "Adopt incrementally",
    "What stays external",  # a.k.a. "What Remains External"
    "Where this is going",
    "Learn more",
    "Development",
]

# The feature-breadth section heading (status claims live under it).
FEATURE_SECTION_PREFIX = "What you get"


@pytest.fixture(scope="module")
def readme_text() -> str:
    assert README.exists(), f"missing README {README}"
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def registry() -> list[dict]:
    assert REGISTRY.exists(), f"missing claim-evidence registry {REGISTRY}"
    return load_claim_registry(REGISTRY)


# --- links, sections, snippets, artifacts -----------------------------------


def test_required_sections_present(readme_text: str) -> None:
    heading_lines = [ln[3:].strip() for ln in readme_text.splitlines() if ln.startswith("## ")]
    missing = [
        prefix
        for prefix in REQUIRED_SECTION_PREFIXES
        if not any(ln.startswith(prefix) for ln in heading_lines)
    ]
    assert not missing, f"README is missing required section(s): {missing}"


def test_local_links_resolve(readme_text: str) -> None:
    slugs = heading_slugs(readme_text)
    broken: list[str] = []
    for path, anchor in iter_local_links(readme_text):
        if not path:  # same-page anchor
            if anchor and anchor not in slugs:
                broken.append(f"#{anchor} (no matching README heading)")
            continue
        target = (README.parent / path).resolve()
        if not target.exists():
            broken.append(path)
    assert not broken, "unresolved local links in README: " + ", ".join(sorted(broken))


def test_cli_snippets_reference_real_commands(readme_text: str) -> None:
    invocations = list(iter_cli_invocations(readme_text))
    assert invocations, "expected at least one lancedb-robotics command snippet in the README"
    failures: list[str] = []
    saw_command = False
    for tokens in invocations:
        ok, detail = resolve_command_path(tokens)
        if not ok:
            failures.append(f"`lancedb-robotics {' '.join(tokens)}` -> {detail}")
        elif detail:
            saw_command = True
    assert not failures, "README references nonexistent CLI commands:\n" + "\n".join(failures)
    assert saw_command, "no command-group snippet was validated (parser regression?)"


def test_nested_command_paths_are_exercised(readme_text: str) -> None:
    """The vertical-slice snippet must show a 3-level command, exercising the
    multi-level parser (a shallow parser would pass a stale nested command)."""
    paths = {
        detail
        for tokens in iter_cli_invocations(readme_text)
        for ok, detail in [resolve_command_path(tokens)]
        if ok and detail
    }
    assert "dataset snapshot create" in paths, (
        "expected the `dataset snapshot create` snippet; without a 3-level command "
        "the nested-command validation would not be exercised"
    )


def test_no_placeholder_artifacts(readme_text: str) -> None:
    found = find_placeholder_artifacts(readme_text)
    assert not found, f"placeholder/transcript artifacts left in README prose: {found}"


def test_status_legend_present(readme_text: str) -> None:
    body = section_body(readme_text, FEATURE_SECTION_PREFIX)
    assert body, f"README has no '## {FEATURE_SECTION_PREFIX}...' section"
    for symbol in ("✅", "🚧", "🔭"):
        assert symbol in body, f"status legend symbol {symbol} missing from feature-breadth section"


# --- claim-evidence registry ------------------------------------------------


def _shipped_or_evolving_claims(readme_text: str) -> dict[str, str]:
    body = section_body(readme_text, FEATURE_SECTION_PREFIX)
    return {
        title: status
        for status, title in iter_status_claims(body)
        if status in ("shipped", "evolving")
    }


def test_status_claims_have_registry_entries(readme_text: str, registry: list[dict]) -> None:
    claims = _shipped_or_evolving_claims(readme_text)
    assert claims, "no ✅/🚧 status claims parsed from the feature-breadth section (parser regression?)"
    registered = {c["key"] for c in registry}
    unregistered = sorted(set(claims) - registered)
    assert not unregistered, (
        "README shipped/evolving claims missing from docs/readme-claim-evidence.toml "
        f"(add a [[claim]] entry — see that file's header): {unregistered}"
    )


def test_registry_has_no_orphans(readme_text: str, registry: list[dict]) -> None:
    claims = _shipped_or_evolving_claims(readme_text)
    orphans = sorted(c["key"] for c in registry if c["key"] not in claims)
    assert not orphans, (
        "docs/readme-claim-evidence.toml has entries with no matching README "
        f"✅/🚧 claim (rename or delete them): {orphans}"
    )


def test_registry_status_matches_readme(readme_text: str, registry: list[dict]) -> None:
    claims = _shipped_or_evolving_claims(readme_text)
    mismatched = [
        f"{c['key']}: README={claims[c['key']]} registry={c['status']}"
        for c in registry
        if c["key"] in claims and c["status"] != claims[c["key"]]
    ]
    assert not mismatched, "registry status disagrees with the README bullet: " + "; ".join(mismatched)


def test_every_claim_has_resolvable_evidence(registry: list[dict]) -> None:
    problems: list[str] = []
    for claim in registry:
        evidence = claim.get("evidence", [])
        if not evidence:
            problems.append(f"{claim['key']}: no evidence listed")
            continue
        for pointer in evidence:
            ok, detail = resolve_evidence(pointer)
            if not ok:
                problems.append(f"{claim['key']}: {pointer} -> {detail}")
    assert not problems, "claim-evidence registry has unresolved evidence:\n" + "\n".join(problems)


def test_registry_keys_are_unique(registry: list[dict]) -> None:
    keys = [c["key"] for c in registry]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"duplicate claim keys in docs/readme-claim-evidence.toml: {dupes}"
