"""Native LanceDB ``Permutation`` read backend over robotics training plans (backlog 0120).

Backlog 0077 persists a deterministic epoch order as a ``(row_id, split_id)`` LanceDB
table -- which happens to be the *exact* schema LanceDB's native
``lancedb.permutation.Permutation`` reader consumes -- but it still reads sample data
back through the robotics hydration executor. Backlog 0120 closes that gap without
duplicating the ordering logic: it adopts the native ``Permutation`` as an
execution/read adapter that consumes the *same* 0077 permutation table as its
permutation table and the pinned ``observations`` snapshot table as its base table.

That buys two native-execution wins for the single-table native observation path:

* lazy column projection via :meth:`Permutation.select_columns`, which reduces the
  columns physically read (never touching ``payload_json``/``payload_blob`` unless
  they are asked for), and
* efficient tensor output via :meth:`Permutation.with_format` (``"torch_col"`` and the
  other native formats).

Snapshot lineage is preserved on the plan handle (row-plan id, epoch-plan id, pinned
table versions, permutation ref) rather than smuggled into the tensor batches, and no
payload/blob materialization is forced.

Aligned multi-source tick plans cannot be expressed by a single-table permutation --
one tick groups rows from several source tables -- so they fall back explicitly via
:class:`AlignedPermutationUnsupportedError`; the aligned path stays executor-backed.

Relationship to 0077: 0077 owns the deterministic order (it writes the permutation
table); 0120 reads through it natively. 0120 is additive -- the Python/executor epoch
backend is unchanged and remains the compatibility path.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from lancedb_robotics.training import (
    _TRAINING_TO_OBSERVATION_COLUMN,
    EPOCH_BACKEND_LANCEDB_PERMUTATION,
    EPOCH_PERMUTATION_TABLE_PREFIX,
    AlignedTrainingTickPlan,
    EpochPlan,
    TrainingError,
    TrainingRowPlan,
    _create_or_reuse_epoch_permutation_table,
    _stable_digest,
)

NATIVE_PERMUTATION_PLAN_KIND = "lancedb-robotics/native-permutation-plan/v1"
NATIVE_PERMUTATION_EXECUTION_MODE = "lancedb-native-permutation-reader"
BASE_TABLE = "observations"
DEFAULT_NATIVE_PERMUTATION_FORMAT = "arrow"
DEFAULT_EQUIVALENCE_VERIFY_LIMIT = 512

# Native output formats accepted by ``Permutation.with_format`` (see LanceDB docs
# ``training/torch``). ``torch``/``torch_col`` additionally require the torch extra.
SUPPORTED_NATIVE_FORMATS = (
    "arrow",
    "python",
    "python_col",
    "numpy",
    "pandas",
    "torch",
    "torch_col",
    "polars",
)


class NativePermutationError(TrainingError):
    """Base error for the native LanceDB permutation read backend."""


class NativePermutationUnavailableError(NativePermutationError):
    """The native ``lancedb.permutation`` reader cannot back this lake/plan."""


class AlignedPermutationUnsupportedError(NativePermutationError):
    """A single-table permutation cannot express an aligned grouped-source tick plan."""


class TorchColUnsupportedError(NativePermutationError):
    """``torch_col`` was requested for columns it cannot zero-copy convert."""


@dataclass(frozen=True)
class NativePermutationPlan:
    """Serializable handle describing a native-``Permutation``-backed training plan.

    The live native reader is attached as :attr:`_permutation` (excluded from
    :meth:`to_dict`); iterate it via :meth:`iter_batches`/:meth:`take`/:meth:`reader`.
    """

    plan_kind: str
    row_plan_id: str
    epoch_plan_id: str | None
    snapshot_name: str
    dataset_id: str
    base_table: str
    base_table_version: int | None
    permutation_table: str
    permutation_ref: str
    permutation_source: str
    split: int
    output_format: str
    requested_columns: tuple[str, ...]
    projection: tuple[str, ...]
    num_rows: int
    shuffle: bool
    shuffle_seed: int | None
    epoch: int
    worker_id: int
    num_workers: int
    resume_from: int
    table_versions: tuple[dict[str, Any], ...]
    materialization_policies: dict[str, str]
    lineage: dict[str, Any]
    equivalence: dict[str, Any]
    torch_col: dict[str, Any]
    warnings: tuple[str, ...] = ()
    _permutation: Any = field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_kind": self.plan_kind,
            "row_plan_id": self.row_plan_id,
            "epoch_plan_id": self.epoch_plan_id,
            "snapshot_name": self.snapshot_name,
            "dataset_id": self.dataset_id,
            "base_table": self.base_table,
            "base_table_version": self.base_table_version,
            "permutation_table": self.permutation_table,
            "permutation_ref": self.permutation_ref,
            "permutation_source": self.permutation_source,
            "split": self.split,
            "output_format": self.output_format,
            "requested_columns": list(self.requested_columns),
            "projection": list(self.projection),
            "num_rows": self.num_rows,
            "shuffle": self.shuffle,
            "shuffle_seed": self.shuffle_seed,
            "epoch": self.epoch,
            "worker": {
                "id": self.worker_id,
                "num_workers": self.num_workers,
                "resume_from": self.resume_from,
            },
            "table_versions": [dict(item) for item in self.table_versions],
            "materialization_policies": dict(self.materialization_policies),
            "lineage": dict(self.lineage),
            "equivalence": dict(self.equivalence),
            "torch_col": dict(self.torch_col),
            "warnings": list(self.warnings),
        }

    def reader(self) -> Any:
        """Return the live projected + formatted native ``Permutation`` reader."""
        if self._permutation is None:
            raise NativePermutationError("no live permutation reader is attached")
        return self._permutation

    def iter_batches(self, batch_size: int, *, skip_last_batch: bool = False):
        """Iterate the native reader in ``batch_size`` batches (chosen output format)."""
        return self.reader().iter(batch_size, skip_last_batch=skip_last_batch)

    def take(self, offsets: Sequence[int]) -> Any:
        """Take rows by offset through the current projection + output format."""
        return self.reader().take_offsets(list(offsets))


def _native_permutation_module() -> Any | None:
    """Return the ``lancedb.permutation`` module iff the native reader is usable."""
    if importlib.util.find_spec("lancedb.permutation") is None:
        return None
    try:
        import lancedb.permutation as permutation
    except Exception:
        return None
    required = ("Permutation", "PermutationReader")
    if not all(hasattr(permutation, name) for name in required):
        return None
    return permutation


def native_permutation_capability(lake: Any | None) -> dict[str, Any]:
    """Report whether ``lake`` can back a training plan with the native reader."""
    module = _native_permutation_module()
    db = getattr(lake, "_db", None) if lake is not None else None
    torch_available = importlib.util.find_spec("torch") is not None
    if module is None:
        supported = False
        reason = "lancedb.permutation native reader is not importable"
    elif db is None:
        supported = False
        reason = "lake does not expose a LanceDB connection"
    elif not hasattr(db, "open_table"):
        supported = False
        reason = "LanceDB connection cannot open the base table"
    else:
        supported = True
        reason = "native lancedb.permutation reader and base-table access are available"
    return {
        "supported": supported,
        "execution_mode": NATIVE_PERMUTATION_EXECUTION_MODE,
        "reason": reason,
        "output_formats": list(SUPPORTED_NATIVE_FORMATS),
        "torch_available": torch_available,
        # torch_col zero-copies only scalar numeric columns; list/nested tensor
        # columns (state/action vectors) are rejected by DLPack in this stack.
        "torch_col_scalar_only": True,
    }


def native_permutation_supported_for_plan(plan: Any) -> tuple[bool, str]:
    """Report whether ``plan`` can be backed by a single-table native permutation."""
    if isinstance(plan, AlignedTrainingTickPlan) or getattr(plan, "source_row_ids", None) is not None:
        return (
            False,
            "aligned multi-source tick plans group rows from several source tables and "
            "cannot be expressed by a single-table permutation; use the executor-backed "
            "aligned path",
        )
    if not isinstance(plan, TrainingRowPlan):
        return (False, f"unsupported plan type: {type(plan).__name__}")
    return (True, "single-table observation row plan")


def _base_table_version(row_plan: TrainingRowPlan, table: str) -> int | None:
    for item in row_plan.table_versions:
        if item.get("table") == table:
            version = item.get("version")
            return int(version) if version is not None else None
    return None


def _ordered_row_ids(row_plan: TrainingRowPlan, sample_indices: Sequence[int]) -> tuple[int, ...]:
    row_ids = row_plan.row_ids
    ordered: list[int] = []
    for index in sample_indices:
        if index >= len(row_ids) or row_ids[index] is None:
            raise NativePermutationUnavailableError(
                "row plan has missing row ids and cannot be backed by a native "
                "permutation; rebuild the snapshot row plan with materialized row ids"
            )
        ordered.append(int(row_ids[index]))
    return tuple(ordered)


def _native_permutation_table_name(
    row_plan: TrainingRowPlan,
    ordered_row_ids: tuple[int, ...],
    *,
    shuffle_seed: int | None,
    epoch: int,
    worker_id: int,
    num_workers: int,
    resume_from: int,
) -> str:
    payload = {
        "row_plan_id": row_plan.plan_id,
        "dataset_id": row_plan.dataset_id,
        "table_versions": row_plan.table_versions,
        "shuffle_seed": shuffle_seed,
        "epoch": epoch,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "resume_from": resume_from,
        "ordered_row_ids": ordered_row_ids,
    }
    return EPOCH_PERMUTATION_TABLE_PREFIX + _stable_digest(payload)


def _torch_col_report(base_schema: pa.Schema, projection: Sequence[str]) -> dict[str, Any]:
    scalar: list[str] = []
    incompatible: list[str] = []
    for name in projection:
        field_type = base_schema.field(name).type
        if _is_torch_col_scalar(field_type):
            scalar.append(name)
        else:
            incompatible.append(name)
    supported = not incompatible
    if supported:
        reason = (
            "all projected columns are scalar numeric and DLPack-convertible; "
            "torch_col additionally requires the read rows to contain no nulls"
        )
    else:
        reason = (
            "torch_col zero-copies only scalar numeric columns; "
            f"{incompatible} are list/nested/non-numeric and are rejected by DLPack "
            "in this lance stack -- project scalar numeric columns, or use "
            "output_format='torch' (per-row) or 'arrow'/'numpy'"
        )
    return {
        "supported": supported,
        "scalar_columns": scalar,
        "incompatible_columns": incompatible,
        "reason": reason,
    }


def _is_torch_col_scalar(field_type: pa.DataType) -> bool:
    return (
        pa.types.is_integer(field_type)
        or pa.types.is_floating(field_type)
        or pa.types.is_boolean(field_type)
    )


def build_native_permutation_plan(
    lake: Any,
    row_plan: Any,
    epoch_plan: EpochPlan,
    *,
    columns: Sequence[str] | None = None,
    output_format: str = DEFAULT_NATIVE_PERMUTATION_FORMAT,
    verify: bool = True,
    verify_limit: int = DEFAULT_EQUIVALENCE_VERIFY_LIMIT,
) -> NativePermutationPlan:
    """Back ``row_plan``/``epoch_plan`` with LanceDB's native ``Permutation`` reader.

    ``columns`` are robotics training column names (defaulting to the row plan's
    columns); they are mapped to physical observation columns and pushed into
    :meth:`Permutation.select_columns`. ``output_format`` is any native format from
    :data:`SUPPORTED_NATIVE_FORMATS`. When ``verify`` is set, the plan reads back a
    bounded prefix of observation ids and records whether the native order matches the
    epoch plan.

    Raises :class:`AlignedPermutationUnsupportedError` for aligned grouped-source tick
    plans, :class:`NativePermutationUnavailableError` when the native reader or row ids
    are missing, and :class:`TorchColUnsupportedError` when ``torch_col`` is asked for
    columns it cannot convert.
    """
    supported, reason = native_permutation_supported_for_plan(row_plan)
    if not supported:
        raise AlignedPermutationUnsupportedError(reason)

    if output_format not in SUPPORTED_NATIVE_FORMATS:
        raise NativePermutationError(
            f"unsupported output format {output_format!r}; expected one of "
            f"{SUPPORTED_NATIVE_FORMATS}"
        )

    capability = native_permutation_capability(lake)
    if not capability["supported"]:
        raise NativePermutationUnavailableError(capability["reason"])
    module = _native_permutation_module()
    db = lake._db

    sample_indices = tuple(epoch_plan.sample_indices)
    if not sample_indices:
        raise NativePermutationUnavailableError(
            "epoch plan selected zero samples; nothing to back with a permutation"
        )
    ordered_row_ids = _ordered_row_ids(row_plan, sample_indices)

    warnings: list[str] = []

    # Reuse the 0077 permutation table verbatim when it already encodes this exact
    # order (single-worker global epoch); otherwise materialize a worker-scoped table
    # via the same 0077 helper so there is one code path for the ordering artifact.
    permutation_table_name, permutation_source, reuse_warning = _resolve_permutation_table(
        db,
        row_plan,
        epoch_plan,
        sample_indices,
        ordered_row_ids,
    )
    if reuse_warning:
        warnings.append(reuse_warning)
    permutation_table = db.open_table(permutation_table_name)

    base_version = _base_table_version(row_plan, BASE_TABLE)
    base_table = db.open_table(BASE_TABLE)
    if base_version is not None:
        try:
            base_table.checkout(base_version)
        except Exception as exc:  # pragma: no cover - defensive; version pin is best-effort
            warnings.append(
                f"could not pin {BASE_TABLE} to version {base_version} ({exc}); "
                "reading latest -- stable row ids keep the mapping valid"
            )

    permutation = module.Permutation.from_tables(base_table, permutation_table, 0)

    requested_columns = tuple(columns) if columns is not None else tuple(row_plan.columns)
    projection = _project_columns(requested_columns)
    base_schema = base_table.schema

    torch_col = _torch_col_report(base_schema, projection)
    if output_format == "torch_col" and not torch_col["supported"]:
        raise TorchColUnsupportedError(torch_col["reason"])

    reader = permutation.select_columns(list(projection)).with_format(output_format)

    equivalence = _verify_equivalence(
        permutation,
        row_plan,
        sample_indices,
        verify=verify,
        verify_limit=verify_limit,
        total_rows=len(ordered_row_ids),
    )

    lineage = {
        "snapshot_name": row_plan.snapshot_name,
        "dataset_id": row_plan.dataset_id,
        "row_plan_id": row_plan.plan_id,
        "epoch_plan_id": epoch_plan.plan_id,
        "base_table": BASE_TABLE,
        "base_table_version": base_version,
        "table_versions": [dict(item) for item in row_plan.table_versions],
        "permutation_table": permutation_table_name,
        "permutation_ref": f"lancedb://{permutation_table_name}",
        "materialization_policies": dict(row_plan.materialization_policies),
    }

    return NativePermutationPlan(
        plan_kind=NATIVE_PERMUTATION_PLAN_KIND,
        row_plan_id=row_plan.plan_id,
        epoch_plan_id=epoch_plan.plan_id,
        snapshot_name=row_plan.snapshot_name,
        dataset_id=row_plan.dataset_id,
        base_table=BASE_TABLE,
        base_table_version=base_version,
        permutation_table=permutation_table_name,
        permutation_ref=f"lancedb://{permutation_table_name}",
        permutation_source=permutation_source,
        split=0,
        output_format=output_format,
        requested_columns=requested_columns,
        projection=projection,
        num_rows=len(ordered_row_ids),
        shuffle=epoch_plan.shuffle,
        shuffle_seed=epoch_plan.shuffle_seed,
        epoch=epoch_plan.epoch,
        worker_id=epoch_plan.worker_id,
        num_workers=epoch_plan.num_workers,
        resume_from=epoch_plan.resume_from,
        table_versions=tuple(dict(item) for item in row_plan.table_versions),
        materialization_policies=dict(row_plan.materialization_policies),
        lineage=lineage,
        equivalence=equivalence,
        torch_col=torch_col,
        warnings=tuple(warnings),
        _permutation=reader,
    )


def _resolve_permutation_table(
    db: Any,
    row_plan: TrainingRowPlan,
    epoch_plan: EpochPlan,
    sample_indices: tuple[int, ...],
    ordered_row_ids: tuple[int, ...],
) -> tuple[str, str, str | None]:
    backend = epoch_plan.backend
    is_global_order = tuple(epoch_plan.global_order) == sample_indices
    if (
        backend.kind == EPOCH_BACKEND_LANCEDB_PERMUTATION
        and backend.permutation_table
        and is_global_order
    ):
        try:
            existing = db.open_table(backend.permutation_table)
            stored = tuple(int(row["row_id"]) for row in existing.to_arrow().to_pylist())
        except Exception:
            stored = None
        if stored == ordered_row_ids:
            return (backend.permutation_table, "reused-0077-table", None)
        warning = (
            f"0077 permutation table {backend.permutation_table} did not match the "
            "requested order; materialized a fresh ordering table"
        )
    else:
        warning = None
    name = _native_permutation_table_name(
        row_plan,
        ordered_row_ids,
        shuffle_seed=epoch_plan.shuffle_seed,
        epoch=epoch_plan.epoch,
        worker_id=epoch_plan.worker_id,
        num_workers=epoch_plan.num_workers,
        resume_from=epoch_plan.resume_from,
    )
    _create_or_reuse_epoch_permutation_table(db, name, ordered_row_ids)
    return (name, "materialized-order-table", warning)


def _project_columns(requested_columns: Sequence[str]) -> tuple[str, ...]:
    projection: list[str] = []
    seen: set[str] = set()
    for column in requested_columns:
        physical = _TRAINING_TO_OBSERVATION_COLUMN.get(column, column)
        if physical not in seen:
            seen.add(physical)
            projection.append(physical)
    if not projection:
        raise NativePermutationError("at least one column must be projected")
    return tuple(projection)


def _verify_equivalence(
    permutation: Any,
    row_plan: TrainingRowPlan,
    sample_indices: tuple[int, ...],
    *,
    verify: bool,
    verify_limit: int,
    total_rows: int,
) -> dict[str, Any]:
    if not verify:
        return {
            "checked": False,
            "matches": None,
            "checked_rows": 0,
            "total_rows": total_rows,
        }
    limit = min(verify_limit, total_rows)
    check_reader = (
        permutation.select_columns(["observation_id"]).with_format("arrow").with_take(limit)
    )
    read_ids: list[str] = []
    for batch in check_reader.iter(min(limit, 1024) or 1, skip_last_batch=False):
        read_ids.extend(batch.column("observation_id").to_pylist())
        if len(read_ids) >= limit:
            break
    read_ids = read_ids[:limit]
    expected = [row_plan.frame_ids[index] for index in sample_indices[:limit]]
    return {
        "checked": True,
        "matches": read_ids == expected,
        "checked_rows": len(read_ids),
        "total_rows": total_rows,
    }
