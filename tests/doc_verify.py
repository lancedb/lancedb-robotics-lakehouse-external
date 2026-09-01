"""Shared, dependency-free documentation guardrail helpers (backlog 0138).

These helpers keep prose docs (the README, the AI Research Lead evidence pack,
and any future doc that names commands or claims features) honest against the
code without needing optional heavy dependencies or real-corpus data, so they
run on the default dev test suite.

They started life inline in ``tests/test_evidence_pack_onboarding.py`` (the 0139
"guardrails where applicable" leg) and were promoted here when backlog 0138
landed the README claim/link/command registry. ``test_evidence_pack_onboarding``
and ``test_readme_verification`` both import from this module.

Two kinds of check live here:

* **Structural parsing** of Markdown — local links + anchors, fenced code
  blocks, ``lancedb-robotics`` CLI snippets resolved against the live Typer tree,
  heading slugs, and a prose view with code stripped out.
* **Claim-evidence registry** support — parse a status list (``✅``/``🚧``/``🔭``
  bullets) out of a doc section, load the ``docs/readme-claim-evidence.toml``
  registry, and resolve each evidence pointer to a real repo file or backlog id.

Nothing here imports an optional extra; the only third-party import is ``typer``
(a core dependency) to walk the real command tree, plus stdlib ``tomllib``.
"""

from __future__ import annotations

import re
import shlex
import tomllib
from collections.abc import Iterator
from pathlib import Path

import typer

from lancedb_robotics.cli import app

# Repo root (this file lives in ``<root>/tests``).
ROOT = Path(__file__).resolve().parents[1]

# Status legend shared by the README feature-breadth list and the registry.
STATUS_SYMBOLS: dict[str, str] = {"✅": "shipped", "🚧": "evolving", "🔭": "planned"}

# Evidence pointer kinds the registry may use. All except ``backlog`` name a
# repo-relative file path; ``backlog`` names a 4-digit backlog id.
EVIDENCE_KINDS = frozenset({"code", "test", "doc", "narrative", "decision", "backlog"})

# Bare-word artifacts that must never appear in *prose* (code spans are exempt,
# so a doc can describe the check itself).
PLACEHOLDER_WORDS = ["TODO", "FIXME", "TKTK", "XXX", "PLACEHOLDER"]
PLACEHOLDER_PHRASES = ["lorem ipsum"]


# --- Markdown structural parsing --------------------------------------------


def slugify(heading: str) -> str:
    """GitHub-flavored heading -> anchor slug (lowercase, punctuation dropped)."""
    text = heading.strip().lstrip("#").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def heading_slugs(text: str) -> set[str]:
    return {
        slugify(line)
        for line in text.splitlines()
        if re.match(r"#{1,6}\s+\S", line)
    }


def iter_local_links(text: str) -> Iterator[tuple[str, str]]:
    """Yield (target_path, anchor) for local Markdown links (skip web/mailto)."""
    for _label, target in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", text):
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        yield path, anchor


def iter_code_blocks(text: str) -> Iterator[tuple[str | None, str]]:
    """Yield (lang, body) for each fenced ``` code block."""
    in_block = False
    lang: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if in_block:
                yield lang, "\n".join(buf)
                in_block, lang, buf = False, None, []
            else:
                in_block = True
                lang = line.strip()[3:].strip() or None
            continue
        if in_block:
            buf.append(line)


def iter_cli_invocations(text: str) -> Iterator[list[str]]:
    """Yield token lists following ``lancedb-robotics`` inside fenced code blocks."""
    for _lang, block in iter_code_blocks(text):
        # Join backslash-continued lines into one logical command.
        logical: list[str] = []
        current = ""
        for raw in block.splitlines():
            stripped = raw.rstrip()
            if stripped.endswith("\\"):
                current += stripped[:-1] + " "
            else:
                current += stripped
                logical.append(current)
                current = ""
        if current:
            logical.append(current)

        for line in logical:
            if not line.strip() or line.strip().startswith("#"):
                continue
            for segment in re.split(r"&&|\|\||[;|]", line):
                if "lancedb-robotics" not in segment:
                    continue
                try:
                    # comments=True drops an inline trailing ``# ...`` shell
                    # comment (e.g. ``lake --help  # every group has --help``)
                    # so it is not mistaken for a positional/subcommand.
                    tokens = shlex.split(segment, comments=True)
                except ValueError:
                    tokens = segment.split("#", 1)[0].split()
                if "lancedb-robotics" not in tokens:
                    continue
                idx = tokens.index("lancedb-robotics")
                yield tokens[idx + 1 :]


def resolve_command_path(tokens: list[str]) -> tuple[bool, str]:
    """Walk the live Typer/Click tree following command-position tokens.

    A group node always expects a subcommand, so a non-flag token that is not a
    child of a group is an error. Once a leaf command is reached, the remaining
    non-flag tokens are positional args and parsing stops. Returns (ok, detail).
    """
    node = typer.main.get_command(app)
    path: list[str] = []
    for token in tokens:
        if token.startswith("-"):
            continue  # option flag (its value is consumed as a positional below)
        subcommands = getattr(node, "commands", None)
        if subcommands:
            if token in subcommands:
                node = subcommands[token]
                path.append(token)
                continue
            return False, f"'{token}' is not a subcommand of '{' '.join(path) or 'lancedb-robotics'}'"
        break  # leaf command reached; token is a positional argument
    return True, " ".join(path)


def strip_code(text: str) -> str:
    """Remove fenced blocks and inline code spans so prose can be scanned alone."""
    without_fences = re.sub(r"```.*?```", " ", text, flags=re.S)
    return re.sub(r"`[^`]*`", " ", without_fences)


def find_placeholder_artifacts(text: str) -> list[str]:
    """Return placeholder/transcript artifacts left in *prose* (code exempt)."""
    prose = strip_code(text)
    found: list[str] = []
    for word in PLACEHOLDER_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", prose):
            found.append(word)
    for phrase in PLACEHOLDER_PHRASES:
        if phrase.lower() in prose.lower():
            found.append(phrase)
    return found


# --- Status claim + evidence registry ---------------------------------------


def section_body(text: str, heading_prefix: str) -> str:
    """Return the body of the ``##``-level section whose heading starts with
    ``heading_prefix`` (exclusive of the next ``##`` heading). Empty if absent."""
    lines = text.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and line[3:].lstrip().startswith(heading_prefix):
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end])


_CLAIM_RE = re.compile(r"^- (✅|🚧|🔭)\s+\*\*(.+?)\*\*")


def iter_status_claims(text: str) -> Iterator[tuple[str, str]]:
    """Yield (status, title) for top-level ``- <emoji> **Title**`` bullets.

    ``status`` is one of ``shipped``/``evolving``/``planned``. Only bullets that
    start a line (no indent) and lead with a status emoji + a **bold** title are
    treated as claims; roadmap prose without a bold title is ignored.
    """
    for line in text.splitlines():
        m = _CLAIM_RE.match(line)
        if m:
            yield STATUS_SYMBOLS[m.group(1)], m.group(2).strip()


def load_claim_registry(path: Path) -> list[dict]:
    """Load the claim-evidence registry TOML into a list of claim dicts."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data.get("claim", []))


def parse_evidence_pointer(pointer: str) -> tuple[str, str]:
    """Split a ``kind:target`` evidence pointer. Raises ValueError on a bad kind."""
    kind, sep, target = pointer.partition(":")
    if not sep or kind not in EVIDENCE_KINDS:
        raise ValueError(
            f"evidence pointer {pointer!r} must be '<kind>:<target>' with kind in "
            f"{sorted(EVIDENCE_KINDS)}"
        )
    return kind, target.strip()


def resolve_evidence(pointer: str, root: Path = ROOT) -> tuple[bool, str]:
    """Resolve a ``kind:target`` evidence pointer to a real repo artifact.

    ``backlog:NNNN`` resolves against ``.miagent/backlog/NNNN-*.md``; every other
    kind names a repo-relative file that must exist. Returns (ok, detail).
    """
    try:
        kind, target = parse_evidence_pointer(pointer)
    except ValueError as exc:
        return False, str(exc)
    if kind == "backlog":
        if not re.fullmatch(r"\d{3,4}", target):
            return False, f"backlog id {target!r} is not 3-4 digits"
        matches = sorted((root / ".miagent" / "backlog").glob(f"{target}-*.md"))
        if not matches:
            return False, f"no .miagent/backlog/{target}-*.md"
        return True, str(matches[0].relative_to(root))
    resolved = (root / target).resolve()
    if not resolved.exists():
        return False, f"missing file {target}"
    return True, target
