from __future__ import annotations

import ctypes
import math
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from .helper import (
    Number as Number,  # re-exported: defined in helper so is_number can use it
)
from .helper import from_import, is_number


class MathForm(Enum):
    Scalar = 0
    Vec = 1
    Mat = 2
    Quat = 3


def _load_type(name:str, dtype:type, math_form:MathForm, shape:Tuple[int, ...])->type:
    """Resolve one generated type module, or say plainly what is missing.

    The module name is derived from the dtype and the shape, so a dtype in
    __dtype_name_map with no matching module used to surface as a bare
    ModuleNotFoundError naming an internal module. c_uint had exactly that
    problem: 'uint' was mapped, so gen_type asked for uint3 and it did not exist.
    """
    try:
        return from_import("." + name, name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"no {name} type: {dtype.__name__} is a known dtype but has no "
            f"{math_form.name} of shape {shape}"
        ) from e

def _has_negative(x:Any)->bool:
    """Whether any scalar leaf below ``x`` is negative, descending into rows."""
    if isinstance(x, genType):
        return any(_has_negative(x[i]) for i in range(x._slot_count()))

    return bool(x < 0)

def _equal_leaves(x:Any, y:Any)->bool:
    """Element-wise equality that descends into rows.

    ``range(len(self))`` is wrong for a genMat: len() is ctypes' flat element
    count (9 for a 3x3) while x[i] only has `rows` valid indices, so comparing
    two matrices raised IndexError instead of returning True.
    """
    if isinstance(x, genType):
        return all(_equal_leaves(x[i], y[i]) for i in range(x._slot_count()))

    return bool(x == y)

class genType:

    __type_order:List[type] = [
        bool, ctypes.c_bool,
        int, ctypes.c_int, ctypes.c_uint,
        # c_uint16 is the half storage type. It sits above uint so that the
        # unsigned wrap-around special case in _bin_op_dtype (which keys off
        # __uint_index) does not apply to it, and below float so half promotes
        # to float rather than the other way round.
        ctypes.c_uint16,
        float, ctypes.c_float, ctypes.c_double
    ]
    __uint_index:int = __type_order.index(ctypes.c_uint)
    __gen_type_map:Dict[Tuple[MathForm, type, Tuple[int, ...]], type] = {}
    __dtype_name_map:Dict[type, str] = {
        ctypes.c_bool: 'bool',
        ctypes.c_int: 'int',
        ctypes.c_uint: 'uint',
        ctypes.c_uint16: 'half',
        ctypes.c_float: 'float',
        ctypes.c_double: 'double',
        bool: 'bool',
        int: 'int',
        float: 'float'
    }
    __dtype_python_type_map:Dict[type, type] = {
        ctypes.c_bool: bool,
        ctypes.c_int: int,
        ctypes.c_uint: int,
        ctypes.c_uint16: float,
        ctypes.c_float: float,
        ctypes.c_double: float,
    }

    _operator_funcs:Dict[str, Callable[[Any,Any], Any]] = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
        "//": lambda x, y: x // y,
        "%": lambda x, y: x % y,
        "**": lambda x, y: x ** y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y,
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y
    }

    def __init__(self):
        self._on_changed:Optional[Callable[[], None]] = None

    # genType models a fixed-size container: math_form, dtype, shape and every
    # element-wise operator below are defined in terms of these. Each concrete
    # subclass (genVec, genMat, genQuat) implements them, so declaring them here
    # is what lets the element-wise helpers in funcs.py be typed against genType.
    def __len__(self)->int:
        raise NotImplementedError(f"{type(self).__name__} does not implement __len__")

    def __getitem__(self, index:Any)->Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement __getitem__")

    def __setitem__(self, index:Any, value:Any)->None:
        raise NotImplementedError(f"{type(self).__name__} does not implement __setitem__")

    def __iter__(self)->Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement __iter__")

    def __str__(self)->str:
        return f"{self.__class__.__name__}({', '.join([str(value) for value in self])})"

    def __repr__(self)->str:
        return f"{self.__class__.__name__}({', '.join([str(value) for value in self])})"

    @property
    def on_changed(self)->Optional[Callable[[], None]]:
        return self._on_changed

    @on_changed.setter
    def on_changed(self, on_changed:Optional[Callable[[], None]]):
        if on_changed is not None and not callable(on_changed):
            raise TypeError('on_changed should be a function')

        self._on_changed = on_changed

    @property
    def math_form(self)->MathForm:
        raise NotImplementedError(f"{type(self).__name__} does not define math_form")

    @property
    def dtype(self)->type:
        raise NotImplementedError(f"{type(self).__name__} does not define dtype")

    @property
    def shape(self)->Tuple[int, ...]:
        raise NotImplementedError(f"{type(self).__name__} does not define shape")

    @staticmethod
    def gen_type(math_form:MathForm, dtype:type, shape:Tuple[int, ...])->type:
        key:Tuple[MathForm, type, Tuple[int, ...]] = (math_form, dtype, shape)
        dtype_name:str = genType.__dtype_name_map[dtype]
        suffix:str = dtype_name[0]
        if key not in genType.__gen_type_map:
            if math.prod(shape) == 1 or math_form == MathForm.Scalar:
                genType.__gen_type_map[key] = genType.__dtype_python_type_map[dtype]
            elif math_form == MathForm.Vec:
                result_name:str = f"{dtype_name}{shape[0]}"
                genType.__gen_type_map[key] = _load_type(result_name, dtype, math_form, shape)
            elif math_form == MathForm.Mat:
                result_name:str = f"matrix{shape[0]}{suffix}"
                genType.__gen_type_map[key] = _load_type(result_name, dtype, math_form, shape)
            elif math_form == MathForm.Quat:
                result_name:str = f"quat{suffix}"
                genType.__gen_type_map[key] = _load_type(result_name, dtype, math_form, shape)

        return genType.__gen_type_map[key]

    def _slot_count(self) -> int:
        """How many top-level slots this container exposes to the operators below.

        A genVec and a genQuat are flat, so their length is the element count. A
        genMat is two dimensional and mixes the two conventions: __len__ is
        ctypes' flat element count (16 for a 4x4) while m[i] is the i-th row, and
        there are only ``rows`` of those. Driving the element-wise loops with
        ``len()`` therefore runs off the end of the array on the first row past the
        row count, which is why matrix addition and subtraction raised IndexError
        while matrix multiplication -- the one operator genMat._op handles itself --
        worked.
        """
        if self.math_form == MathForm.Mat:
            return cast(Any, self).rows

        return len(self)

    def _call_on_changed(self):
        if self._on_changed is None:
            return

        self._on_changed()

    def _update_data(self, indices:Optional[Iterable[int]] = None):
        self._call_on_changed()

    @staticmethod
    def _bin_op_dtype(operator:str, type1:type, type2:type, type2_has_negative:bool=False)->type:
        type1_order = genType.__type_order.index(type1)
        type2_order = genType.__type_order.index(type2)
        if type1_order <= genType.__uint_index and type2_order <= genType.__uint_index and (
            operator == "/" or (operator == "**" and type2_has_negative)
        ):
            return ctypes.c_float
        else:
            return (type1 if type1_order > type2_order else type2)

    @staticmethod
    def _bin_op_type(operator:str, value1:Union[float, bool, int, genType], value2:Union[float, bool, int, genType])->type:
        value1_dtype:type = type(value1) if is_number(value1) else cast(genType, value1).dtype
        value2_dtype:type = type(value2) if is_number(value2) else cast(genType, value2).dtype

        if isinstance(value1, genType) and isinstance(value2, genType) and not value1._is_homo(value2):
            raise TypeError(f"unsupported operand type(s) for {operator}: '{value1.__class__.__name__}' and '{value2.__class__.__name__}'")

        second_has_negative:bool = False
        if operator == "**":
            if isinstance(value2, genType):
                # _slot_count, not len(): a matrix slot is a row, and `row < 0`
                # would answer the question about the wrong thing.
                for i in range(value2._slot_count()):
                    if _has_negative(value2[i]):
                        second_has_negative = True
                        break
            else:
                second_has_negative = (value2 < 0)

        math_form:Optional[MathForm] = None
        shape:Optional[Tuple[int, ...]] = (1,)
        if isinstance(value1, genType):
            math_form = value1.math_form
            shape = value1.shape
        elif isinstance(value2, genType):
            math_form = value2.math_form
            shape = value2.shape

        result_dtype:type = genType._bin_op_dtype(operator, value1_dtype, value2_dtype, second_has_negative)
        result_type:type = genType.gen_type(cast(MathForm, math_form), result_dtype, shape)
        return result_type

    def __neg__(self)->genType:
        result_type = self.__class__
        if self.dtype == ctypes.c_uint:
            result_type = self.gen_type(self.math_form, ctypes.c_int, self.shape)

        result:genType = result_type()
        for i in range(result._slot_count()):
            result[i] = ((not self[i]) if self.dtype == ctypes.c_bool else -self[i])

        return result

    def _is_homo(self, other:Any)->bool:
        return (
            isinstance(other, genType) and
            self.math_form == other.math_form and
            self.shape == other.shape
        )

    @staticmethod
    def _at(other:Any, index:Any)->Any:
        """Element ``index`` of a shape-matched operand.

        Only call this when _is_homo(other) holds, which means the other side is a
        genType of the same shape; the cast records what the runtime guarantees.
        """
        return cast(genType, other)[index]

    def _op(self, operator:str, other:Union[float, bool, int, genType])->genType:
        # An operand that is neither a number nor a shape-matched genType has no
        # element-wise meaning here, and returning NotImplemented rather than
        # raising is what lets Python try the other operand's reflected method.
        # Without it, `double3(...) + attribute` treats the whole attribute as one
        # scalar, c_double.__add__ hands it to the attribute's __radd__, and every
        # element ends up assigned a vector.
        other_is_homo:bool = self._is_homo(other)
        if not other_is_homo and not is_number(other):
            return NotImplemented

        result_type = self._bin_op_type(operator, self, other)
        result:genType = result_type()
        operator_func:Callable[[Any,Any], Any] = self._operator_funcs[operator]
        for i in range(result._slot_count()):
            result[i] = operator_func(self[i], genType._at(other, i) if other_is_homo else other)

        return result

    def _rop(self, operator:str, other:Union[float, bool, int, genType])->genType:
        # The reflected form only ever has a scalar on the left, so anything else
        # has no meaning to offer and Python should hear so via NotImplemented.
        if not is_number(other):
            return NotImplemented

        result_type = self._bin_op_type(operator, other, self)
        result:genType = result_type()
        operator_func:Callable[[Any,Any], Any] = self._operator_funcs[operator]
        for i in range(result._slot_count()):
            result[i] = operator_func(other, self[i])

        return result

    def _iop(self, operator:str, other:Union[float, bool, int, genType])->genType:
        other_is_homo:bool = self._is_homo(other)
        if not other_is_homo and not is_number(other):
            raise TypeError(f"unsupported operand type(s) for {operator}=: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

        operator_func:Callable[[Any,Any], Any] = self._operator_funcs[operator]
        for i in range(self._slot_count()):
            self[i] = operator_func(self[i], genType._at(other, i) if other_is_homo else other)

        self._update_data()

        return self

    def _compare_op(self, operator:str, other:Union[float, bool, int, genType])->genType:
        btype = self.gen_type(self.math_form, ctypes.c_bool, self.shape)
        result:genType = btype()

        other_is_homo:bool = self._is_homo(other)
        if not other_is_homo and not is_number(other):
            # NotImplemented, not TypeError, so that `gf_value > attribute` can
            # reach the attribute's own comparison rather than stopping here.
            return NotImplemented

        operator_func:Callable[[Any,Any], Any] = self._operator_funcs[operator]
        for i in range(self._slot_count()):
            result[i] = operator_func(self[i], genType._at(other, i) if other_is_homo else other)

        return result

    def _compare_rop(self, operator:str, other:Union[float, bool, int, genType])->genType:
        if not is_number(other):
            return NotImplemented

        btype = self.gen_type(self.math_form, ctypes.c_bool, self.shape)
        result:genType = btype()

        operator_func:Callable[[Any,Any], Any] = self._operator_funcs[operator]
        for i in range(result._slot_count()):
            result[i] = operator_func(other, self[i])

        return result

    # The `object` in these three unions is not sloppiness. Each operator has to
    # accept an operand of *any* type so that it can return NotImplemented for one
    # it does not understand, which is what lets Python fall through to the other
    # side's reflected method. Narrowing them to `float | genType` is what made
    # `ty` report the deliberate `vec + "x"` checks as errors, and would hide a
    # genuine NotImplemented path from anyone maintaining them.
    def __add__(self, other:object)->genType:
        return self._op("+", cast(Any, other))

    def __radd__(self, other:object)->genType:
        return self._rop("+", cast(Any, other))

    def __iadd__(self, other:object)->genType:
        return self._iop("+", cast(Any, other))

    def __sub__(self, other:object)->genType:
        return self._op("-", cast(Any, other))

    def __rsub__(self, other:object)->genType:
        return self._rop("-", cast(Any, other))

    def __isub__(self, other:object)->genType:
        return self._iop("-", cast(Any, other))

    def __mul__(self, other:object)->genType:
        return self._op("*", cast(Any, other))

    def __rmul__(self, other:object)->genType:
        return self._rop("*", cast(Any, other))

    def __imul__(self, other:object)->genType:
        return self._iop("*", cast(Any, other))

    def __truediv__(self, other:object)->genType:
        return self._op("/", cast(Any, other))

    def __rtruediv__(self, other:object)->genType:
        return self._rop("/", cast(Any, other))

    def __itruediv__(self, other:Union[float, bool, int, genType]):
        return self._iop("/", cast(Any, other))

    def __floordiv__(self, other:object)->genType:
        return self._op("//", cast(Any, other))

    def __rfloordiv__(self, other:object)->genType:
        return self._rop("//", cast(Any, other))

    def __ifloordiv__(self, other:Union[float, bool, int, genType]):
        return self._iop("//", cast(Any, other))

    def __mod__(self, other:object)->genType:
        return self._op("%", cast(Any, other))

    def __rmod__(self, other:object)->genType:
        return self._rop("%", cast(Any, other))

    def __imod__(self, other:Union[float, bool, int, genType]):
        return self._iop("%", cast(Any, other))

    def __pow__(self, other:object)->genType:
        return self._op("**", cast(Any, other))

    def __rpow__(self, other:object)->genType:
        return self._rop("**", cast(Any, other))

    def __ipow__(self, other:Union[float, bool, int, genType]):
        return self._iop("**", cast(Any, other))

    def __eq__(self, other:object)->bool:
        if not isinstance(other, self.__class__):
            return False

        return _equal_leaves(self, other)

    def __req__(self, other:object)->bool:
        return (self == other)

    def __ne__(self, other:object)->bool:
        if not isinstance(other, self.__class__):
            return True

        return not _equal_leaves(self, other)

    def __rne__(self, other:object)->bool:
        return (self != other)

    def __gt__(self, other:object)->genType:
        return self._compare_op(">", cast(Any, other))

    def __rgt__(self, other:object)->genType:
        return self._compare_rop(">", cast(Any, other))

    def __lt__(self, other:object)->genType:
        return self._compare_op("<", cast(Any, other))

    def __rlt__(self, other:object)->genType:
        return self._compare_rop("<", cast(Any, other))

    def __ge__(self, other:object)->genType:
        return self._compare_op(">=", cast(Any, other))

    def __rge__(self, other:object)->genType:
        return self._compare_rop(">=", cast(Any, other))

    def __le__(self, other:object)->genType:
        return self._compare_op("<=", cast(Any, other))

    def __rle__(self, other:object)->genType:
        return self._compare_rop("<=", cast(Any, other))
