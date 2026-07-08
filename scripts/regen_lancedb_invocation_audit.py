"""Regenerate the LanceDB invocation remote-compatibility audit (JSON + MD).

The audit manifest is line-keyed: every ``src/lancedb_robotics`` line shift or new
LanceDB/Lance call site makes ``tests/test_lancedb_invocation_audit.py`` stale.
This regenerator re-scans source with the test's own scanner, re-attaches the
human-curated compatibility fields per ``(path, function, callees)`` group in line
order (so per-line curation survives a shift), copies curated fields for genuinely
new call sites from a same-callees donor, and rewrites both the JSON source of
truth and the derived Markdown roll-ups + matrix. Run from the repo root:

    python scripts/regen_lancedb_invocation_audit.py
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = ROOT / "docs" / "product" / "lancedb-invocation-remote-compatibility-audit.json"
AUDIT_MD = ROOT / "docs" / "product" / "lancedb-invocation-remote-compatibility-audit.md"
TEST = ROOT / "tests" / "test_lancedb_invocation_audit.py"

_CURATED_KEYS = (
    "support_class",
    "backend_paths",
    "auth_by_backend",
    "auth_planes",
    "enterprise_db",
    "fallback_behavior",
    "needs_direct_object_io",
    "needs_namespace_endpoint",
)


def _load_scanner():
    spec = importlib.util.spec_from_file_location("audit_test", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _group_key(row: dict) -> tuple:
    return (row["path"], row["function"], tuple(sorted(row["callees"])))


def main() -> None:
    scanner = _load_scanner()
    scanned = scanner.scan_invocations()
    old_doc = json.loads(AUDIT_JSON.read_text())
    old_rows = old_doc["rows"]

    # Per (path, function, callees) group, in line order, so a line shift keeps the
    # curated fields on the same relative call site.
    old_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in sorted(old_rows, key=lambda r: r["line"]):
        old_groups[_group_key(row)].append(row)
    consumed: dict[tuple, int] = defaultdict(int)

    # Donor curated fields for genuinely new call sites, keyed by callee signature.
    donor_by_callees: dict[tuple, dict] = {}
    for row in old_rows:
        donor_by_callees.setdefault(tuple(sorted(row["callees"])), row)

    new_rows: list[dict] = []
    unmatched: list[dict] = []
    for row in scanned:
        key = _group_key(row)
        bucket = old_groups.get(key, [])
        idx = consumed[key]
        curated_src = None
        if idx < len(bucket):
            curated_src = bucket[idx]
            consumed[key] = idx + 1
        else:
            curated_src = donor_by_callees.get(tuple(sorted(row["callees"])))
        if curated_src is None:
            raise SystemExit(
                f"no curated donor for new call site {row['path']}:{row['line']} "
                f"{row['function']} callees={row['callees']}"
            )
        merged = dict(row)
        for curated_key in _CURATED_KEYS:
            merged[curated_key] = curated_src[curated_key]
        new_rows.append(merged)
        if idx >= len(bucket):
            unmatched.append(merged)

    new_doc = {
        "backlog": old_doc["backlog"],
        "schema_version": old_doc["schema_version"],
        "scanner": old_doc["scanner"],
        "rows": new_rows,
    }
    AUDIT_JSON.write_text(json.dumps(new_doc, indent=2) + "\n")

    _write_md(new_rows)

    print(f"rows: {len(old_rows)} -> {len(new_rows)}")
    print(f"new call sites (curated from donor): {len(unmatched)}")
    for row in unmatched:
        print(f"  {row['path']}:{row['line']} {row['function']} {row['callees']} -> {row['support_class']}")


def _write_md(rows: list[dict]) -> None:
    text = AUDIT_MD.read_text()
    header = text.split("## Support Roll-Up", 1)[0].rstrip() + "\n\n"

    # Preserve the human-authored "Meaning" text per support class.
    meanings: dict[str, str] = {}
    in_table = False
    for line in text.splitlines():
        if line.strip() == "## Support Roll-Up":
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0].strip(" `") in {"Support class", "---"}:
            continue
        meanings[cells[0].strip(" `")] = cells[2]

    support = Counter(r["support_class"] for r in rows)
    families: Counter = Counter()
    for row in rows:
        families.update(row["families"])
    files = Counter(r["path"] for r in rows)

    out = [header.rstrip(), ""]
    out.append("## Support Roll-Up")
    out.append("| Support class | Count | Meaning |")
    out.append("| --- | ---: | ---: |")
    for cls in sorted(support):
        out.append(f"| `{cls}` | {support[cls]} | {meanings.get(cls, '')} |")
    out.append("")
    out.append("## Family Roll-Up")
    out.append("| Operation family | Count |")
    out.append("| --- | ---: |")
    for family in sorted(families):
        out.append(f"| `{family}` | {families[family]} |")
    out.append("")
    out.append("## File Coverage")
    out.append("| File | Rows |")
    out.append("| --- | ---: |")
    for path in sorted(files):
        out.append(f"| `{path}` | {files[path]} |")
    out.append("")
    out.append("## Detailed Invocation Matrix")
    out.append("| ID | Call site | Families | Callees | Support | Backends | Auth refs (union) |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(rows, key=lambda r: (r["path"], r["line"])):
        call_site = f"`{row['path']}:{row['line']}` `{row['function']}`"
        families_cell = ", ".join(row["families"])
        callees_cell = ", ".join(row["callees"])
        backends_cell = ", ".join(row["backend_paths"])
        auth_cell = ", ".join(row["auth_planes"])
        out.append(
            f"| `{row['id']}` | {call_site} | `{families_cell}` | `{callees_cell}` | "
            f"`{row['support_class']}` | `{backends_cell}` | `{auth_cell}` |"
        )
    AUDIT_MD.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
