from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from typing_extensions import TypeAlias

import numpy as np

from .genType import MathForm, genType
from .helper import (
    Number,
    from_import,
    generate_getter_swizzles,
    generate_setter_swizzles,
    in_annotations,
    is_number,
)

if TYPE_CHECKING:
    from .genMat import genMat


class _VectorDtype:
    """Non-data descriptor that answers ``vec_class.dtype`` and ``vec.dtype``.

    On the class it returns the structured numpy dtype for the whole vector
    (``np.dtype(cls._fields_)``) -- the value PyGPGPU's host-side ``float4.dtype``
    historically exposed -- so ``np.array(..., dtype=float4.dtype)`` keeps
    working. On an instance it returns the scalar ctypes type that pygf's type
    arithmetic (``vec_type(self.dtype, ...)``) needs, unchanged from the old
    property. A non-data descriptor (no ``__set__``) is required so that both the
    class- and instance-level reads are supported; a plain ``property`` could not
    return different values to the two.
    """
    def __get__(self, obj:Optional[genVec], objtype:Optional[type] = None)->Any:
        if obj is None:
            return np.dtype(cast(Any, objtype)._fields_)

        return cast(Any, objtype)._fields_[0][1]


class genVec(genType, ctypes.Structure):

    dtype = _VectorDtype()

    _attr_index_map:Dict[str, int] = {
        'x': 0,
        'y': 1,
        'z': 2,
        'w': 3,
        'r': 0,
        'g': 1,
        'b': 2,
        'a': 3,
        's': 0,
        't': 1,
        'p': 2,
        'q': 3
    }

    # OpenCL spells a vector's components ``s0``-style, and a 16-wide vector
    # names the slots above nine in hex, either case. The names are attribute
    # names, so every numeric swizzle carries the ``s`` prefix; the index of a
    # character is its hex value. On by default: the numeric spelling is part
    # of the host surface, so ``float2().s01`` and ``float16().sA`` read here
    # like every other swizzle. A subclass that wants a GLSL-only surface
    # (no ``s0``-style names) turns this off.
    _numeric_swizzles:bool = True
    _NUMERIC_LENGTHS:Tuple[int, ...] = (1, 2, 3, 4, 8, 16)

    _all_getter_swizzles:Set[str] = set()
    _all_setter_swizzles:Set[str] = set()
    __all_total_swizzles:Set[str] = set()
    __namespaces:List[str] = ['xyzw', 'rgba', 'stpq']

    _related_mat: Optional[genMat]
    _mat_start_index: int
    _on_changed: Optional[Callable[[], None]]

    def __init__(self, *args):
        genType.__init__(self)
        self._related_mat:Optional[genMat] = None
        self._mat_start_index:int = -1

        # A sequence argument carries its components: `float4([1, 2], 3, 4)` is
        # the positional call, which is what the ctypes-flavoured callers write.
        expanded:List[Any] = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                expanded.extend(arg)
            else:
                expanded.append(arg)
        args = tuple(expanded)

        i: int = 0
        n_data: int = len(self)
        n_args: int = len(args)

        if n_args == 0:
            return

        if n_args == 1:
            arg = args[0]
            if is_number(arg):
                for i in range(n_data):
                    self[i] = arg
                return

        for i_arg, arg in enumerate(args):
            if is_number(arg):
                self[i] = arg

                i += 1
                if i == n_data:
                    if n_args != 1 and i_arg != n_args - 1:
                        raise ValueError(f"invalid arguments for {self.__class__.__name__}()")

                    return

            elif isinstance(arg, genVec):
                sub_n_arg: int = len(arg)
                for sub_i_arg, value in enumerate(arg):
                    self[i] = value

                    i += 1
                    if i == n_data:
                        if n_args != 1 and (i_arg != n_args - 1 or sub_i_arg != sub_n_arg - 1):
                            raise ValueError(f"invalid arguments for {self.__class__.__name__}()")

                        return

            else:
                raise TypeError(f"invalid argument type(s) for {self.__class__.__name__}()")

        raise ValueError(f"invalid arguments for {self.__class__.__name__}()")

    @property
    def math_form(self)->MathForm:
        return MathForm.Vec

    def __len__(self)->int:
        return len(self._fields_)

    @property
    def shape(self)->Tuple[int]:
        return (len(self),)

    @classmethod
    def numpy_dtype(cls)->Any:
        """The structured numpy dtype for this vector type.

        Built from the ctypes ``_fields_`` (``('x', c_float)`` and friends), so
        ``np.dtype(cls.numpy_dtype())`` names each component x/y/z as a floating
        or integer scalar exactly as the type stores it. ``cls.dtype`` returns
        the same value on the class (see ``_VectorDtype``); this method exists
        so 2/4-width code can ask for it without relying on descriptor semantics.
        """
        return np.dtype(cls._fields_)

    @staticmethod
    def vec_type(dtype:type, size:int)->type:
        return genType.gen_type(MathForm.Vec, dtype, (size,))

    def _vec_result_type(self, size:int)->type:
        """The vector class a swizzle of ``size`` components answers.

        ``vec_type`` names pygf's own modules, so a dtype pygf carries no module
        for (a caller package's ``char2`` or ``float8``) raises there, and the
        result belongs to the operand's own family -- its class name minus the
        width digits, loaded beside that class's modules. The family load is
        also authoritative for ``ushort``: ``c_uint16`` is pygf's ``half``
        dtype, and a ``ushort2`` swizzle must answer a ``ushort2``. One
        component is the scalar itself, whatever family the class belongs to.
        """
        if size == 1:
            return self.vec_type(self.dtype, 1)

        cls:type = type(self)
        cls_name:str = cls.__name__
        family:str = cls_name.rstrip("0123456789")
        if len(family) == len(cls_name):
            # A name carrying no width digits is an *alias* (color3f, point3f):
            # it is named after a meaning, not a family, and pygf's own naming
            # resolves it to the base vector the alias subclasses.
            return self.vec_type(self.dtype, size)
        if genType._family_name_of(self.dtype) == family:
            try:
                return self.vec_type(self.dtype, size)
            except ValueError:
                pass  # pygf names the family but carries no module of this width

        package:Optional[str] = cls.__module__.rpartition(".")[0] or None
        name:str = f"{family}{size}"
        try:
            return from_import(f".{name}", name, package=package)
        except (ImportError, AttributeError):
            raise AttributeError(
                f"no {name} vector class beside {cls.__module__}: "
                f"'{cls.__name__}' has no {size}-component swizzle result") from None

    def _letter_index(self, ch:str)->int:
        """The component a GLSL spelling names.

        On a 16-wide vector the lowercase hex digits ``a``-``f`` name the slots
        above nine (OpenCL's ``sA``-style spelling, either case), which shadows
        rgba's ``a`` and ``b`` exactly as the OpenCL layout does.
        """
        if len(self) >= 16 and ch in "abcdef":
            return int(ch, 16)
        return self._attr_index_map[ch]

    def _component_attr(self, index:int)->str:
        """The attribute name a component is stored through.

        The class's own spelling wins: ``x`` on the GLSL-shaped classes and on
        a half (whose ``x`` is the converting property), ``s0`` on the
        OpenCL-shaped wide ones. Writing through the *field* name alone would
        bypass the half property and store a raw bit pattern; writing through
        the GLSL spelling on a wide class would name nothing.
        """
        field_name:str = self._fields_[index][0]
        spelling:str = 'xyzw'[index] if index < 4 else field_name
        return spelling if hasattr(type(self), spelling) else field_name

    def _field_value(self, index:int, value:Any)->Any:
        """The value a component slot stores, converted the way C would.

        A float into a raw integer field truncates toward zero -- the C
        float-to-integer conversion -- instead of reaching ctypes and raising.
        A converting accessor (half's ``x`` property) owns its own conversion
        and is left alone; a ``c_uint16`` raw field (``ushort``) is converted.
        """
        spelling:str = 'xyzw'[index] if index < 4 else self._fields_[index][0]
        if isinstance(getattr(type(self), spelling, None), property):
            return value

        field_type:type = self._fields_[index][1]
        if field_type in genType.INTEGER_FIELD_TYPES and isinstance(value, float):
            return int(value)

        return value

    def _numeric_charset(self)->str:
        """The numeric characters this width's components are named by."""
        n:int = len(self)
        if n >= 16:
            return "0123456789ABCDEFabcdef"
        if n >= 8:
            return "01234567"
        return "0123456789ABCDEF"[:n]

    def _numeric_indices(self, name:str)->Optional[List[int]]:
        """The component indices a numeric swizzle names, or None.

        ``s0`` reads one component, ``s01`` two, ``sAF`` names slots 10 and 15
        on a 16-wide vector. The ``s`` prefix is part of every numeric name
        (digits cannot start an attribute), the characters come from this
        width's set, and the length is one of the enumerable ones -- which is
        what keeps a read of ``s01230123`` on a four-wide vector (an ``s``-spelled
        eight-component result) legal while ``s01234`` is not.
        """
        if not self._numeric_swizzles:
            return None
        if len(name) < 2 or name[0] != 's' or len(name) - 1 not in self._NUMERIC_LENGTHS:
            return None

        charset:str = self._numeric_charset()
        indices:List[int] = []
        for char in name[1:]:
            if char not in charset:
                return None
            indices.append(int(char, 16))

        return indices

    def _update_data(self, indices:Optional[Iterable[int]] = None):
        genType._update_data(self, indices)

        if self._related_mat is not None:
            if indices is None:
                indices = range(len(self))

            for index in indices:
                self._related_mat[self._mat_start_index + index] = self[index]

            self._related_mat._call_on_changed()

    def _int_result_type(self)->type:
        """The integer vector class of this width, in this vector's own family.

        Comparisons answer an ``intN`` (per-component 0/1, the device's own
        convention). ``gen_type`` names pygf's modules, so a width pygf carries
        no ``int`` module for -- an 8- or 16-wide vector, whose ``int8`` lives
        in the package that added it -- loads the family class beside the
        operand's own modules instead.
        """
        try:
            return genType.gen_type(MathForm.Vec, ctypes.c_int, self.shape)
        except ValueError:
            package:Optional[str] = type(self).__module__.rpartition(".")[0] or None
            name:str = f"int{len(self)}"
            return from_import(f".{name}", name, package=package)

    def _compare_op(self, operator:str, other:Union[float, bool, int, genType])->genVec:
        """Element-wise comparison returning an ``intN`` vector, not a bool one.

        This matches both PyGPGPU's GPU behaviour (CUDA/OpenCL translate a vector
        comparison into per-component 0/1 in an integer vector) and its legacy
        ``genVec._compare_op``. ``==``/``!=`` therefore return an ``int2/3/4`` too,
        unlike ``genType`` where they answer a Python ``bool`` -- this override
        keeps vectors aligned with the device while matrices and quaternions keep
        the GLSL-style boolean result.
        """
        other_is_homo:bool = self._is_homo(other)
        if not other_is_homo and not is_number(other):
            return NotImplemented

        itype:type = self._int_result_type()
        result:genVec = itype()

        operator_func:Callable[[Any, Any], Any] = self._operator_funcs[operator]
        for i in range(len(self)):
            result[i] = int(operator_func(self[i], genType._at(other, i) if other_is_homo else other))

        return result

    def _compare_rop(self, operator:str, other:Union[float, bool, int, genType])->genVec:
        if not is_number(other):
            return NotImplemented

        itype:type = self._int_result_type()
        result:genVec = itype()

        operator_func:Callable[[Any, Any], Any] = self._operator_funcs[operator]
        for i in range(len(result)):
            result[i] = int(operator_func(other, self[i]))

        return result

    def __eq__(self:Any, other:object)->Any:
        if not self._is_homo(other) and not is_number(other):
            return False
        return self._compare_op("==", cast(Any, other))

    def __req__(self, other:object)->Any:
        return self._compare_rop("==", cast(Any, other))

    def __ne__(self:Any, other:object)->Any:
        if not self._is_homo(other) and not is_number(other):
            return True
        return self._compare_op("!=", cast(Any, other))

    def __rne__(self, other:object)->Any:
        return self._compare_rop("!=", cast(Any, other))

    def __getattr__(self, name:str)->Union[float,bool,int,genVec]:
        indices:Optional[List[int]] = self._numeric_indices(name)
        if indices is not None:
            vec_type:type = self._vec_result_type(len(indices))
            return vec_type(*(self[i] for i in indices))

        if name not in self.__getter_swizzles():
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        indices = [self._letter_index(ch) for ch in name]
        vec_type = self._vec_result_type(len(name))
        return vec_type(*(self[i] for i in indices))

    def __setattr__(self, name:str, value:Union[float,bool,int,genVec])->None:
        # A component *field* of this class stores through the conversion: a
        # float into a raw integer slot truncates the way C would. A converting
        # accessor (half's `x` property) is not a field name and falls through
        # to the normal attribute path below, which reaches its setter. An
        # abstract intermediate (genVec2, genQuat) has no fields at all.
        fields = getattr(type(self), "_fields_", None)
        if fields:
            for index, (field_name, _field_type) in enumerate(fields):
                if field_name == name:
                    ctypes.Structure.__setattr__(self, name, self._field_value(index, value))
                    return

        if hasattr(self.__class__, name) or in_annotations(name, self.__class__):
            super().__setattr__(name, value)
            return

        indices:Optional[List[int]] = self._numeric_indices(name)
        if indices is not None:
            if len(set(indices)) != len(indices):
                raise AttributeError(f"property '{name}' of '{self.__class__.__name__}' object has no setter")
        elif name in self.__setter_swizzles():
            indices = [self._letter_index(ch) for ch in name]
        elif name in self.__getter_swizzles():
            raise AttributeError(f"property '{name}' of '{self.__class__.__name__}' object has no setter")
        else:
            if name in self.__total_swizzles():
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

            super().__setattr__(name, value)
            return

        # A swizzle writes one component per named slot. The value may be a
        # matching vector, a sequence of that length, or a scalar broadcast.
        value_is_vec:bool = (isinstance(value, genVec) and len(value) == len(indices))
        value_is_sequence:bool = (
            not value_is_vec and isinstance(value, (list, tuple))
            and len(value) == len(indices))
        if not value_is_vec and not value_is_sequence and not is_number(value):
            raise TypeError(f"can not set '{value.__class__.__name__}' object to property '{name}' of '{self.__class__.__name__}' object")

        update_indices:List[int] = []
        for i, index in enumerate(indices):
            component:Any = value[i] if (value_is_vec or value_is_sequence) else value
            ctypes.Structure.__setattr__(self, self._component_attr(index),
                                         self._field_value(index, component))
            update_indices.append(index)
        self._update_data(update_indices)

    def __getitem__(self, index:Union[int, slice, str])->Union[float,int,bool,genVec]:
        # A component may also be named, which is the spelling a structure member uses:
        # `v["x"]` reads the same thing `v.x` and `v[0]` do. Device code that indexes a
        # structure (`s["v"]["x"]`) then runs unchanged in plain python.
        if isinstance(index, str):
            return getattr(self, index)
        if isinstance(index, slice):
            # a sliced GLSL spelling reads as that swizzle, as it always has
            return getattr(self, 'xyzw'[index])
        # Through the component's own accessor, never a fixed spelling: `x` on
        # the GLSL-shaped classes, `s0` on the OpenCL-shaped wide ones -- and
        # always the converting property for a half.
        return getattr(self, self._component_attr(index))

    def __setitem__(self, index:Union[int,slice,str], value:Union[float,int,bool,genVec])->None:
        if isinstance(index, str):
            setattr(self, index, value)
            return
        if isinstance(index, slice):
            setattr(self, 'xyzw'[index], value)
            return
        setattr(self, self._component_attr(index), self._field_value(index, value))

    def __getter_swizzles(self):
        if not self.__class__._all_getter_swizzles:
            n:int = len(self)
            self.__class__._all_getter_swizzles = set(
                generate_getter_swizzles(
                    namespace[:n] for namespace in genVec.__namespaces
                )
            )

        return self.__class__._all_getter_swizzles

    def __setter_swizzles(self):
        if not self.__class__._all_setter_swizzles:
            n:int = len(self)
            self.__class__._all_setter_swizzles = set(
                generate_setter_swizzles(
                    namespace[:n] for namespace in genVec.__namespaces
                )
            )

        return self.__class__._all_setter_swizzles

    @staticmethod
    def __total_swizzles():
        if not genVec.__all_total_swizzles:
            genVec.__all_total_swizzles = set(generate_getter_swizzles(genVec.__namespaces))

        return genVec.__all_total_swizzles

    def __iter__(self):
        # Through self[i], not the _fields_ names: a half field is a c_uint16
        # holding a bit pattern, and reading the field directly would yield that
        # integer instead of the float the component represents.
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, value:Any):
        return any(component == value for component in self)

VecType: TypeAlias = Union[genVec, Tuple[Number, ...]]
