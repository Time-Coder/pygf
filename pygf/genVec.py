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

    def _update_data(self, indices:Optional[Iterable[int]] = None):
        genType._update_data(self, indices)

        if self._related_mat is not None:
            if indices is None:
                indices = range(len(self))

            for index in indices:
                self._related_mat[self._mat_start_index + index] = self[index]

            self._related_mat._call_on_changed()

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

        itype:type = genType.gen_type(self.math_form, ctypes.c_int, self.shape)
        result:genVec = itype()

        operator_func:Callable[[Any, Any], Any] = self._operator_funcs[operator]
        for i in range(len(self)):
            result[i] = int(operator_func(self[i], genType._at(other, i) if other_is_homo else other))

        return result

    def _compare_rop(self, operator:str, other:Union[float, bool, int, genType])->genVec:
        if not is_number(other):
            return NotImplemented

        itype:type = genType.gen_type(self.math_form, ctypes.c_int, self.shape)
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
        if name not in self.__getter_swizzles():
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        indices:List[int] = [self._attr_index_map[ch] for ch in name]
        vec_type = self.vec_type(self.dtype, len(name))
        return vec_type(*(self[i] for i in indices))

    def __setattr__(self, name:str, value:Union[float,bool,int,genVec]):
        if hasattr(self.__class__, name) or in_annotations(name, self.__class__):
            super().__setattr__(name, value)
            return

        if name not in self.__setter_swizzles():
            if name in self.__getter_swizzles():
                raise AttributeError(f"property '{name}' of '{self.__class__.__name__}' object has no setter")

            if name in self.__total_swizzles():
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

            super().__setattr__(name, value)
            return

        update_indices:List[int] = []
        value_is_vec:bool = (isinstance(value, genVec) and len(value) == len(name))
        if not value_is_vec and not is_number(value):
            raise TypeError(f"can not set '{value.__class__.__name__}' object to property '{name}' of '{self.__class__.__name__}' object")

        for i, ch in enumerate(name):
            index:int = self._attr_index_map[ch]
            ctypes.Structure.__setattr__(self, ch, value[i] if value_is_vec else value)
            update_indices.append(index)
        self._update_data(update_indices)

    def __getitem__(self, index:Union[int, slice, str])->Union[float,int,bool,genVec]:
        # A component may also be named, which is the spelling a structure member uses:
        # `v["x"]` reads the same thing `v.x` and `v[0]` do. Device code that indexes a
        # structure (`s["v"]["x"]`) then runs unchanged in plain python.
        if isinstance(index, str):
            return getattr(self, index)
        attr_name:str = 'xyzw'[index]
        return getattr(self, attr_name)

    def __setitem__(self, index:Union[int,slice,str], value:Union[float,int,bool,genVec])->None:
        if isinstance(index, str):
            setattr(self, index, value)
            return
        attr_name:str = 'xyzw'[index]
        setattr(self, attr_name, value)

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
