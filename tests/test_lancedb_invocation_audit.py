from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "lancedb_robotics"
AUDIT_JSON = ROOT / "docs" / "product" / "lancedb-invocation-remote-compatibility-audit.json"
AUDIT_MD = ROOT / "docs" / "product" / "lancedb-invocation-remote-compatibility-audit.md"

DB_METHODS = {"list_tables", "create_table", "open_table"}
TABLE_METHODS = {
    "add",
    "add_columns",
    "checkout",
    "count_rows",
    "create_fts_index",
    "create_index",
    "create_scalar_index",
    "delete",
    "list_indices",
    "merge_insert",
    "search",
    "take_row_ids",
    "to_arrow",
}
QUERY_METHODS = {
    "limit",
    "nearest_to",
    "rerank",
    "select",
    "to_arrow",
    "to_batches",
    "to_list",
    "where",
}
DATASET_METHODS = {
    "add_columns",
    "cleanup_old_versions",
    "drop_columns",
    "get_fragments",
    "take",
    "take_blobs",
    "to_table",
    "versions",
}
OPTIMIZE_METHODS = {"compact_files", "optimize_indices"}
TAG_METHODS = {"create", "delete", "list", "update"}
LANCEDB_NAME_CALLS = {"ColumnOrdering", "RRFReranker"}


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive around future syntax.
        return ""


def _norm(text: str, *, limit: int = 160) -> str:
    return re.sub(r"\s+", " ", text.strip())[:limit]


def _names(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", text))


def _returns_table(call: ast.AST) -> bool:
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return False
    attr = call.func.attr
    receiver = _unparse(call.func.value)
    if attr in {"open_table", "create_table"}:
        return True
    return (
        attr == "table"
        and receiver != "pa"
        and (
            receiver.endswith("lake")
            or receiver.endswith("opened")
            or receiver in {"lake", "opened", "self._lake", "self.lake", "selection.lake"}
        )
    )


def _returns_dataset(call: ast.AST) -> bool:
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"open_dataset", "to_lance"}
    )


def _uses_table(expr: ast.AST, table_vars: set[str]) -> bool:
    text = _unparse(expr)
    return (".table(" in text and "pa.table(" not in text) or bool(_names(text) & table_vars)


def _uses_dataset(expr: ast.AST, dataset_vars: set[str]) -> bool:
    text = _unparse(expr)
    return (
        ".to_lance()" in text
        or "lance.dataset" in text
        or bool(_names(text) & dataset_vars)
    )


class _InvocationScanner(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines = path.read_text().splitlines()
        self.scope = ["<module>"]
        self.rows: list[dict[str, Any]] = []
        self.table_vars: set[str] = set()
        self.dataset_vars: set[str] = set()
        self.db_vars: set[str] = {"db"}
        self.tag_vars: set[str] = set()

    def _function(self) -> str:
        names = [name for name in self.scope if name != "<module>"]
        return ".".join(names) if names else "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        previous = (
            self.table_vars.copy(),
            self.dataset_vars.copy(),
            self.db_vars.copy(),
            self.tag_vars.copy(),
        )
        self.scope.append(node.name)
        self.table_vars = set()
        self.dataset_vars = set()
        self.db_vars = {"db"}
        self.tag_vars = set()

        for item in ast.walk(node):
            if not isinstance(item, ast.Assign | ast.AnnAssign):
                continue
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            value = item.value
            if value is None:
                continue
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if _returns_table(value):
                    self.table_vars.add(target.id)
                if _returns_dataset(value):
                    self.dataset_vars.add(target.id)
                if isinstance(value, ast.Call) and _unparse(value.func) in {
                    "_connect",
                    "lancedb.connect",
                }:
                    self.db_vars.add(target.id)
                if isinstance(value, ast.Attribute) and value.attr == "tags":
                    if _uses_dataset(value.value, self.dataset_vars):
                        self.tag_vars.add(target.id)

        self.generic_visit(node)
        self.scope.pop()
        self.table_vars, self.dataset_vars, self.db_vars, self.tag_vars = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def _add(self, node: ast.AST, family: str, callee: str) -> None:
        line = self.lines[node.lineno - 1] if node.lineno - 1 < len(self.lines) else ""
        self.rows.append(
            {
                "path": self.path.relative_to(ROOT).as_posix(),
                "line": node.lineno,
                "function": self._function(),
                "family": family,
                "callee": callee,
                "text": _norm(line),
            }
        )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            receiver = node.func.value
            receiver_text = _unparse(receiver)
            callee = _unparse(node.func)
            if callee == "lancedb.connect":
                self._add(node, "db_control", "lancedb.connect")
            elif callee == "lance_namespace.connect":
                self._add(node, "namespace", "lance_namespace.connect")
            elif callee == "lance.dataset" or (attr == "dataset" and receiver_text == "lance"):
                self._add(node, "direct_lance", "lance.dataset")
            elif attr in DB_METHODS and (
                receiver_text in self.db_vars or receiver_text.endswith("._db")
            ):
                self._add(node, "db_control", f"DBConnection.{attr}")
            elif _returns_table(node):
                self._add(node, "db_control", "Lake.table")
            elif attr == "to_lance" and _uses_table(receiver, self.table_vars):
                self._add(node, "direct_lance", "Table.to_lance")
            elif attr in TABLE_METHODS and _uses_table(receiver, self.table_vars):
                family = {
                    "add": "table_write",
                    "add_columns": "table_write",
                    "checkout": "versioning",
                    "count_rows": "table_read",
                    "create_fts_index": "index",
                    "create_index": "index",
                    "create_scalar_index": "index",
                    "delete": "table_write",
                    "list_indices": "index",
                    "merge_insert": "table_write",
                    "search": "search",
                    "take_row_ids": "table_read",
                    "to_arrow": "table_read",
                }[attr]
                self._add(node, family, f"Table.{attr}")
            elif attr in QUERY_METHODS and (
                ".search(" in receiver_text or _uses_table(receiver, self.table_vars)
            ):
                self._add(
                    node,
                    "search" if ".search(" in receiver_text else "table_read",
                    f"Query.{attr}",
                )
            elif attr in DATASET_METHODS and _uses_dataset(receiver, self.dataset_vars):
                family = {
                    "cleanup_old_versions": "maintenance",
                    "get_fragments": "maintenance",
                    "take_blobs": "blob",
                    "versions": "versioning",
                }.get(attr, "direct_lance")
                self._add(node, family, f"LanceDataset.{attr}")
            elif attr in OPTIMIZE_METHODS and (
                ".to_lance().optimize" in receiver_text or "optimize" in receiver_text
            ):
                self._add(node, "maintenance", f"LanceDataset.optimize.{attr}")
            elif attr in TAG_METHODS and receiver_text in self.tag_vars:
                self._add(node, "maintenance", f"LanceTags.{attr}")
            elif attr in {"open_dataset", "describe", "refresh_if_needed"} and (
                "PylanceNamespaceAccess" in ".".join(self.scope)
            ):
                family = "direct_lance" if attr == "open_dataset" else "namespace"
                self._add(node, family, f"PylanceNamespaceAccess.{attr}")
        elif isinstance(node.func, ast.Name) and node.func.id in LANCEDB_NAME_CALLS:
            self._add(
                node,
                "search" if node.func.id == "RRFReranker" else "table_read",
                node.func.id,
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr == "version" and (
            _uses_table(node.value, self.table_vars) or _uses_dataset(node.value, self.dataset_vars)
        ):
            callee = (
                "LanceDataset.version"
                if _uses_dataset(node.value, self.dataset_vars)
                else "Table.version"
            )
            self._add(node, "versioning", callee)
        elif node.attr == "schema" and (
            _uses_table(node.value, self.table_vars) or _uses_dataset(node.value, self.dataset_vars)
        ):
            callee = (
                "LanceDataset.schema"
                if _uses_dataset(node.value, self.dataset_vars)
                else "Table.schema"
            )
            self._add(node, "table_read", callee)
        elif node.attr == "tags" and _uses_dataset(node.value, self.dataset_vars):
            self._add(node, "maintenance", "LanceDataset.tags")
        self.generic_visit(node)


def scan_invocations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        scanner = _InvocationScanner(path)
        scanner.visit(ast.parse(path.read_text(), filename=str(path)))
        rows.extend(scanner.rows)

    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["path"], row["line"], row["function"])
        current = grouped.setdefault(
            key,
            {
                "path": row["path"],
                "line": row["line"],
                "function": row["function"],
                "families": set(),
                "callees": set(),
                "text": row["text"],
            },
        )
        current["families"].add(row["family"])
        current["callees"].add(row["callee"])

    output: list[dict[str, Any]] = []
    for row in sorted(grouped.values(), key=lambda item: (item["path"], item["line"])):
        callees = sorted(row["callees"])
        identity = f"{row['path']}:{row['line']}:{row['function']}:{','.join(callees)}"
        output.append(
            {
                "id": hashlib.sha1(identity.encode()).hexdigest()[:10],
                "path": row["path"],
                "line": row["line"],
                "function": row["function"],
                "families": sorted(row["families"]),
                "callees": callees,
                "text": row["text"],
            }
        )
    return output


def _load_audit_rows() -> list[dict[str, Any]]:
    return json.loads(AUDIT_JSON.read_text())["rows"]


def test_lancedb_invocation_audit_manifest_is_current() -> None:
    actual = {row["id"]: row for row in scan_invocations()}
    audited = {row["id"]: row for row in _load_audit_rows()}
    assert set(actual) == set(audited)
    for row_id, row in actual.items():
        audited_row = audited[row_id]
        for key in ("path", "line", "function", "families", "callees", "text"):
            assert audited_row[key] == row[key]


def test_lancedb_invocation_audit_rollup_matches_inventory() -> None:
    expected = Counter(row["support_class"] for row in _load_audit_rows())
    observed: dict[str, int] = {}
    in_table = False
    for line in AUDIT_MD.read_text().splitlines():
        if line.strip() == "## Support Roll-Up":
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.startswith("|"):
            continue
        cells = [cell.strip(" `") for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"Support class", "---"}:
            continue
        observed[cells[0]] = int(cells[1])
    assert observed == dict(sorted(expected.items()))


def test_enterprise_remote_rows_do_not_require_storage_auth_only_for_db_path() -> None:
    for row in _load_audit_rows():
        if "lancedb_remote_db" not in row["backend_paths"]:
            continue
        if row["support_class"] in {
            "enterprise_remote_supported_now",
            "enterprise_remote_capability_check",
        }:
            assert "storage_auth_ref" not in row["auth_by_backend"]["lancedb_remote_db"], row["id"]
