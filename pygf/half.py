"""IEEE-754 binary16 storage for the ``half`` element types.

ctypes has no ``c_half``, so a half has to be stored as its 16 bit pattern in a
``c_uint16`` field and converted on every access. Python's ``struct`` module has
had a native binary16 format since 3.6, so the conversion needs no third-party
dependency -- which matters, because this is the only numpy-independent reason
the rest of the library could not be.

The component names (``x``, ``y``, ``z``, ``w``) are properties here and the ctypes
fields are private (``_hx`` and friends), because a ctypes Structure cannot carry
both a field and a property under one name. Two consequences follow, and both are
load-bearing:

  * ``ctypes.Structure.__setattr__`` dispatches to a property setter, so the
    existing ``genVec.__setattr__`` swizzle path keeps working unchanged.
  * Anything that read ``_fields_`` names directly would get the raw bit pattern
    instead of a float, so ``__iter__`` and ``__contains__`` go through
    ``self[i]`` rather than the field names.
"""

from __future__ import annotations

import ctypes
import struct
from typing import Any, List, Tuple

_PACK_HALF:Any = struct.Struct("e").pack
_UNPACK_HALF:Any = struct.Struct("e").unpack
# A uint16 field reads back as a plain int, so the conversion needs an int ->
# 2-byte step that is not the same as the float -> bytes one.
_TO_BITS:Any = struct.Struct("<H").pack
_FROM_BITS:Any = struct.Struct("<H").unpack

_POS_INF_BITS: int = 0x7C00
_NEG_INF_BITS: int = 0xFC00

COMPONENT_NAMES: str = "xyzw"


def to_bits(value:Any)->int:
    """The binary16 bit pattern for ``value``.

    ``struct`` raises OverflowError above 65504 rather than rounding to
    infinity, which is what IEEE-754 round-to-nearest does and what a GPU buffer
    would contain. Saturating to a signed infinity keeps a value that is merely
    too large from aborting a whole computation.
    """
    value = float(value)

    try:
        return _FROM_BITS(_PACK_HALF(value))[0]
    except OverflowError:
        return _NEG_INF_BITS if value < 0 else _POS_INF_BITS


def from_bits(bits:int)->float:
    """The Python float a binary16 bit pattern represents."""
    return _UNPACK_HALF(_TO_BITS(bits & 0xFFFF))[0]


def half_fields(count:int)->List[Tuple[str, Any]]:
    """The private ``_fields_`` for a half vector or quaternion of ``count`` components."""
    return [
        (f"_h{COMPONENT_NAMES[i]}", ctypes.c_uint16)
        for i in range(count)
    ]


def half_property(index:int)->property:
    """A converting accessor for one component of a half type.

    ``index`` is the position in COMPONENT_NAMES, so index 0 is ``x`` for a
    vector and ``w`` for a quaternion -- the two disagree deliberately, because
    genVec and genQuat number their components differently.
    """
    field_name = f"_h{COMPONENT_NAMES[index]}"

    def getter(self)->float:
        return from_bits(getattr(self, field_name))

    def setter(self, value:Any)->None:
        setattr(self, field_name, to_bits(value))

    return property(getter, setter)


def half_properties(component_names:str)->List[property]:
    """One converting accessor per name in ``component_names``."""
    return [
        half_property(COMPONENT_NAMES.index(name))
        for name in component_names
    ]
