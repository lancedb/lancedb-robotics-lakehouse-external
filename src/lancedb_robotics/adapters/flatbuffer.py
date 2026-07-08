"""Generic FlatBuffers decode via embedded schema reflection (backlog 0020).

A FlatBuffers message is **not** self-describing: the bytes carry no field names
or types, only a vtable of offsets. To decode one generically we need its schema.
MCAP carries that schema inline -- a ``flatbuffer`` channel's :class:`Schema`
holds the binary reflection schema (a ``.bfbs``, itself a serialized
``reflection.Schema``) in ``schema.data``. The Foxglove SDK and recorders emit
``foxglove.*`` messages this way, as the flatbuffer twin of the protobuf
``foxglove.*`` types.

This module parses that ``.bfbs`` once per schema and then walks any message of
that schema into a plain ``dict`` whose field names and nesting match the
protobuf/ros decode of the same logical type -- so 0015 typed extraction routes a
flatbuffer ``foxglove.LocationFix`` to the same ``gps`` vector as its protobuf
twin (decision: encoding is an implementation detail; the same robot message
must land as the same lakehouse row).

Only the ``flatbuffers`` runtime is required (the ``flatbuffer`` extra); it is
imported lazily so the package loads without it. The reflection reader is
hand-rolled on top of ``flatbuffers.Table`` because the pip ``flatbuffers``
runtime no longer ships the generated ``reflection`` classes. Field vtable
offsets follow the documented ``reflection.fbs`` layout (``voffset = 4 + 2*i``).

Decode is **degrade-safe**: bytes/``[ubyte]`` fields are returned as ``bytes`` so
the caller's blob-hoisting elides large image/pointcloud payloads exactly as it
does for protobuf; a field whose base type we do not model becomes a visible
sentinel string rather than a crash; scalar fields left at their schema default
are absent (matching proto3 ``MessageToDict``), so downstream accessors fall back
to the same defaults.
"""

from __future__ import annotations

from typing import Any

# reflection.fbs BaseType enum (stable; the order is part of the format).
(
    _NONE, _UTYPE, _BOOL, _BYTE, _UBYTE, _SHORT, _USHORT, _INT, _UINT, _LONG,
    _ULONG, _FLOAT, _DOUBLE, _STRING, _VECTOR, _OBJ, _UNION, _ARRAY, _VECTOR64,
) = range(19)

# Inline byte sizes for the scalar/element base types, used to stride vectors.
_SCALAR_SIZE: dict[int, int] = {
    _BOOL: 1, _BYTE: 1, _UBYTE: 1, _UTYPE: 1, _SHORT: 2, _USHORT: 2,
    _INT: 4, _UINT: 4, _LONG: 8, _ULONG: 8, _FLOAT: 4, _DOUBLE: 8,
}


def _voffset(field_index: int) -> int:
    """Vtable offset of the ``field_index``-th field in a reflection table."""
    return 4 + 2 * field_index


class FlatbufferReflection:
    """A parsed ``.bfbs`` reflection schema that decodes its messages to dicts.

    Construct once per MCAP schema (parsing the ``.bfbs`` is the only up-front
    cost) and reuse for every message on that channel. Construction raises if the
    ``flatbuffers`` runtime is absent or the schema bytes are not a valid
    reflection schema; both surface to the caller as a decode failure rather than
    crashing ingest.
    """

    def __init__(self, bfbs: bytes) -> None:
        from flatbuffers import encode, packer
        from flatbuffers.table import Table

        self._encode = encode
        self._packer = packer
        self._Table = Table

        schema = self._root(bfbs)
        # Schema table: objects(0), enums(1), file_ident(2), file_ext(3),
        # root_table(4). Index objects by reflection-table position so a Type's
        # ``index`` (into the objects vector) resolves to the right sub-table.
        self._objects = [_RObject(self, o) for o in self._vector_of_tables(schema, 0)]
        root = self._subtable(schema, 4)
        if root is None:
            raise ValueError("reflection schema has no root_table")
        self._root_object = _RObject(self, root)

    @property
    def root_name(self) -> str | None:
        return self._root_object.name

    def decode(self, data: bytes) -> dict[str, Any]:
        """Decode one message buffer of this schema into a plain dict."""
        return self._decode_object(self._root_object, self._root(data))

    # --- low-level flatbuffers.Table helpers -------------------------------

    def _root(self, buf: bytes):
        pos = self._encode.Get(self._packer.uoffset, buf, 0)
        return self._Table(buf, pos)

    @staticmethod
    def _scalar(table, field_index, flags, default=0):
        off = table.Offset(_voffset(field_index))
        return table.Get(flags, off + table.Pos) if off != 0 else default

    @staticmethod
    def _string(table, field_index):
        off = table.Offset(_voffset(field_index))
        if off == 0:
            return None
        return table.String(off + table.Pos).decode("utf-8", "replace")

    def _subtable(self, table, field_index):
        off = table.Offset(_voffset(field_index))
        if off == 0:
            return None
        return self._Table(table.Bytes, table.Indirect(off + table.Pos))

    def _vector_of_tables(self, table, field_index):
        off = table.Offset(_voffset(field_index))
        if off == 0:
            return []
        start = table.Vector(off)
        out = []
        for j in range(table.VectorLen(off)):
            elem_off = start + j * 4
            pos = elem_off + self._encode.Get(self._packer.uoffset, table.Bytes, elem_off)
            out.append(self._Table(table.Bytes, pos))
        return out

    # --- message walk ------------------------------------------------------

    def _decode_object(self, obj: _RObject, table) -> dict[str, Any]:
        return self._decode_struct(obj, table) if obj.is_struct else self._decode_table(obj, table)

    def _decode_table(self, obj: _RObject, table) -> dict[str, Any]:
        from flatbuffers import number_types as nt

        out: dict[str, Any] = {}
        for field in obj.fields:
            try:
                value = self._read_table_field(field, table, nt)
            except Exception:  # noqa: BLE001 - one bad field must not sink the message
                continue
            if value is not _ABSENT:
                out[field.name] = value
        return out

    def _read_table_field(self, field: _RField, table, nt) -> Any:
        bt = field.base_type
        voff = _voffset_for(field)
        off = table.Offset(voff)
        if off == 0:
            # Absent => the field holds its schema default. A *zero* default is
            # omitted (matching proto3 MessageToDict, and what the caller's
            # ``.get(k, 0.0)`` accessors already assume), but a non-zero default
            # is emitted: foxglove ``Vector3`` defaults x/y/z to 1.0 and
            # ``Quaternion`` defaults w to 1.0, so omitting them would silently
            # diverge from the protobuf twin (proto3 has no non-zero scalar
            # defaults, so its 1.0s are written explicitly). Emitting them keeps
            # flatbuffer/protobuf vector parity exact.
            default = field.scalar_default if bt in _SCALAR_FLAGS else None
            return default if default else _ABSENT
        if bt in _SCALAR_FLAGS:
            return table.Get(_SCALAR_FLAGS[bt](nt), off + table.Pos)
        if bt == _STRING:
            return table.String(off + table.Pos).decode("utf-8", "replace")
        if bt == _OBJ:
            sub = self._objects[field.type_index]
            if sub.is_struct:
                return self._decode_struct(sub, self._Table(table.Bytes, off + table.Pos))
            return self._decode_table(sub, self._Table(table.Bytes, table.Indirect(off + table.Pos)))
        if bt == _VECTOR:
            return self._decode_vector(field, table, off, nt)
        return f"<unsupported flatbuffer base_type {bt}>"

    def _decode_struct(self, obj: _RObject, table) -> dict[str, Any]:
        # Structs are inline and vtable-free: each field sits at a fixed byte
        # offset from the struct's position (``field.offset`` is that byte offset,
        # not a vtable voffset as it is for tables).
        from flatbuffers import number_types as nt

        out: dict[str, Any] = {}
        for field in obj.fields:
            bt = field.base_type
            pos = table.Pos + field.offset
            try:
                if bt in _SCALAR_FLAGS:
                    out[field.name] = table.Get(_SCALAR_FLAGS[bt](nt), pos)
                elif bt == _OBJ:
                    sub = self._objects[field.type_index]
                    out[field.name] = self._decode_struct(sub, self._Table(table.Bytes, pos))
                else:
                    out[field.name] = f"<unsupported flatbuffer struct base_type {bt}>"
            except Exception:  # noqa: BLE001 - degrade-safe per field
                continue
        return out

    def _decode_vector(self, field: _RField, table, off, nt) -> Any:
        element = field.element
        start = table.Vector(off)
        length = table.VectorLen(off)
        # [ubyte]/[byte] is binary payload (image/pointcloud/compressed data):
        # return raw bytes so the caller hoists/inlines it like protobuf bytes,
        # never a multi-thousand-element int list in payload_json.
        if element in (_UBYTE, _BYTE):
            return bytes(table.Bytes[start : start + length])
        if element in _SCALAR_FLAGS:
            flags = _SCALAR_FLAGS[element](nt)
            size = _SCALAR_SIZE[element]
            return [table.Get(flags, start + j * size) for j in range(length)]
        if element == _STRING:
            out = []
            for j in range(length):
                eoff = start + j * 4
                pos = eoff + self._encode.Get(self._packer.uoffset, table.Bytes, eoff)
                out.append(self._Table(table.Bytes, pos).String(pos).decode("utf-8", "replace"))
            return out
        if element == _OBJ:
            sub = self._objects[field.type_index]
            out = []
            for j in range(length):
                eoff = start + j * 4
                pos = eoff + self._encode.Get(self._packer.uoffset, table.Bytes, eoff)
                out.append(self._decode_object(sub, self._Table(table.Bytes, pos)))
            return out
        return [f"<unsupported flatbuffer vector element {element}>"] * length


# Map a reflection scalar base type to the matching ``flatbuffers`` number flags.
# Values are lambdas over the ``number_types`` module so the module imports
# without ``flatbuffers`` present.
_SCALAR_FLAGS = {
    _BOOL: lambda nt: nt.BoolFlags,
    _BYTE: lambda nt: nt.Int8Flags,
    _UBYTE: lambda nt: nt.Uint8Flags,
    _UTYPE: lambda nt: nt.Uint8Flags,
    _SHORT: lambda nt: nt.Int16Flags,
    _USHORT: lambda nt: nt.Uint16Flags,
    _INT: lambda nt: nt.Int32Flags,
    _UINT: lambda nt: nt.Uint32Flags,
    _LONG: lambda nt: nt.Int64Flags,
    _ULONG: lambda nt: nt.Uint64Flags,
    _FLOAT: lambda nt: nt.Float32Flags,
    _DOUBLE: lambda nt: nt.Float64Flags,
}

_ABSENT = object()  # sentinel: field not present in the buffer (schema default)


def _voffset_for(field: _RField) -> int:
    """Vtable offset of a table field == its reflection ``offset`` value.

    The reflection ``Field.offset`` stores the field's voffset in the data
    table's vtable, which is exactly what ``Table.Offset`` expects.
    """
    return field.offset


class _RObject:
    """A reflection ``Object`` (a table or struct definition)."""

    __slots__ = ("name", "is_struct", "fields")

    def __init__(self, schema: FlatbufferReflection, table) -> None:
        from flatbuffers import number_types as nt

        # Object: name(0), fields(1), is_struct(2).
        self.name = schema._string(table, 0)
        self.is_struct = bool(schema._scalar(table, 2, nt.BoolFlags, 0))
        self.fields = [_RField(schema, f) for f in schema._vector_of_tables(table, 1)]


class _RField:
    """A reflection ``Field`` (name, type, vtable/struct offset, scalar default)."""

    __slots__ = ("name", "offset", "base_type", "element", "type_index", "_default_real",
                 "_default_integer")

    def __init__(self, schema: FlatbufferReflection, table) -> None:
        from flatbuffers import number_types as nt

        # Field: name(0), type(1), id(2), offset(3), default_integer(4),
        # default_real(5); Type: base_type(0), element(1), index(2).
        self.name = schema._string(table, 0)
        self.offset = schema._scalar(table, 3, nt.Uint16Flags, 0)
        self._default_integer = schema._scalar(table, 4, nt.Int64Flags, 0)
        self._default_real = schema._scalar(table, 5, nt.Float64Flags, 0.0)
        type_table = schema._subtable(table, 1)
        self.base_type = schema._scalar(type_table, 0, nt.Int8Flags, 0)
        self.element = schema._scalar(type_table, 1, nt.Int8Flags, 0)
        self.type_index = schema._scalar(type_table, 2, nt.Int32Flags, -1)

    @property
    def scalar_default(self) -> float | int:
        """The field's declared default, typed as the field's base type."""
        if self.base_type in (_FLOAT, _DOUBLE):
            return self._default_real
        return self._default_integer


__all__ = ["FlatbufferReflection"]
