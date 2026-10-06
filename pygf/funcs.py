import builtins
import ctypes
import math
from collections.abc import Callable
from typing import Any, List, Union, cast

from .genMat import genMat
from .genQuat import genQuat
from .genType import genType
from .genVec import genVec
from .genVec3 import genVec3
from .helper import Number, is_number


def _add(x:Any, y:Any)->Any:
    return x + y

def _sub(x:Any, y:Any)->Any:
    return x - y

def _mul(x:Any, y:Any)->Any:
    return x * y

def _div(x:Any, y:Any)->Any:
    return x / y


def _single_op(x:Any, op:Callable[[Any], Any], op_name:str)->Any:
    """Apply `op` element-wise, descending through rows.

    Accepts a bare number as well as a container: `_single_op` is the shared body
    of every unary helper, and GLSL defines all of them for scalars too.
    """
    if is_number(x):
        return op(x)
    elif isinstance(x, genType):
        result:genType = x.__class__()
        for i in range(x._slot_count()):
            _assign_slot(result, i, _single_op(x[i], op, op_name))

        return result
    else:
        raise TypeError(f"{op_name} not supported for type {x.__class__.__name__}")

def abs(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, builtins.abs, "abs")

def sign(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, lambda x: math.copysign(1, x), "sign")

def floor(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.floor, "floor")

def ceil(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.ceil, "ceil")

def trunc(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.trunc, "trunc")

def round(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, lambda x: math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5), "round")

def _round_even(x:float)->float:
    """Banker's rounding: halfway cases go to the nearest even integer.

    Python's built-in round() already does this, so the fix is to reach it --
    `round` is this module's own genType wrapper, and passing that here made
    roundEven a byte-for-byte copy of round (half away from zero).
    """
    return builtins.round(x)

def roundEven(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, _round_even, "roundEven")

def fract(x:Union[genType, Number])->Union[genType, float]:
    # GLSL: fract(x) = x - floor(x). Using trunc made fract(-1.5) return -0.5,
    # which is not a fractional part of anything.
    return _single_op(x, lambda x: x - math.floor(x), "fract")

def mod(x:genType, y:genType)->genType:
    return x % y

def _bin_op(x:Any, y:Any, op:Callable[[Any,Any], Any], op_name:str)->Any:
    """Apply `op` element-wise to two operands, descending through rows.

    Accepts bare numbers as well as containers: GLSL defines all of these for
    scalars, and every one of them already worked at runtime -- the annotations
    were simply narrower than the behaviour.
    """
    if is_number(x) and is_number(y):
        return op(x, y)

    if isinstance(x, genType) and isinstance(y, genType) and (x.math_form != y.math_form or x.shape != y.shape):
        raise TypeError(f"not defined {op_name} between '{x.__class__.__name__}' and '{y.__class__.__name__}'")

    # The result type is derived once, from the two containers. Deriving it per
    # level meant recursing on bare floats, and _bin_op_type cannot classify a
    # Python float -- it raised KeyError looking up a dtype name for `float`.
    result:genType = genType._bin_op_type(op_name, x, y)()
    _fill_bin(result, x, y, op)

    return result

def _fill_bin(result:genType, x:Any, y:Any, op:Callable[[Any,Any], Any])->None:
    """Fill `result` slot by slot from x and y, descending while either is a container.

    A matrix slot is a row, so a second level is reached whenever x[i] or y[i]
    is itself a genType.
    """
    x_slots:int = x._slot_count() if isinstance(x, genType) else 0
    y_slots:int = y._slot_count() if isinstance(y, genType) else 0

    if x_slots and y_slots and x_slots != y_slots:
        raise TypeError(f"not defined between '{x.__class__.__name__}' and '{y.__class__.__name__}'")

    for i in range(result._slot_count()):
        x_slot = x[i] if x_slots else x
        y_slot = y[i] if y_slots else y

        if isinstance(x_slot, genType) or isinstance(y_slot, genType):
            inner = (x_slot if isinstance(x_slot, genType) else y_slot).__class__()
            _fill_bin(inner, x_slot, y_slot, op)
            _assign_slot(result, i, inner)
        else:
            result[i] = op(x_slot, y_slot)

def _assign_slot(target:genType, index:int, value:Any)->None:
    """Store one computed slot, whether the target slot is a scalar or a row.

    genVec.__setitem__ takes a scalar for an int index; genMat.__setitem__ takes
    a row. Assigning through target[index] covers both without the caller having
    to know which one it has.
    """
    target[index] = value

def min(x:Union[genType, Number], y:Union[genType, Number])->Union[genType, float]:
    return _bin_op(x, y, builtins.min, "min")

def max(x:Union[genType, Number], y:Union[genType, Number])->Union[genType, float]:
    return _bin_op(x, y, builtins.max, "max")

def clamp(x:Union[genType, Number],
          min_value:Union[genType, Number],
          max_value:Union[genType, Number])->Union[genType, float]:
    return min(max(x, min_value), max_value)

def mix(x:genType, y:genType, a:Union[genType, Number])->genType:
    # Written through the operator helpers rather than with `*` and `-` directly:
    # `a` may be a bare scalar here, and the operators are what accept one.
    one_minus_a = _bin_op(a, 1.0, lambda a, one: one - a, "-")
    return _bin_op(_bin_op(x, one_minus_a, _mul, "*"), _bin_op(y, a, _mul, "*"), _add, "+")

def step(edge:Union[genType, Number], x:Union[genType, Number])->Union[genType, float]:
    return _bin_op(edge, x, lambda edge, x: float(x >= edge), "step")

def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if x <= edge0:
        return 0.0
    if x >= edge1:
        return 1.0
    t = (x - edge0) / (edge1 - edge0)
    return t * t * (3.0 - 2.0 * t)

def smoothstep(edge0: genType, edge1: genType, x: genType)->genType:
    if (not (is_number(edge0) and is_number(edge1))) and not edge0._is_homo(edge1):
        raise ValueError('edge0 and edge1 must be same type')

    return _bin_op(edge1, x, lambda edge1, x: _smoothstep(cast(float, edge0), cast(float, edge1), cast(float, x)), "smoothstep")

def sqrt(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.sqrt, "sqrt")

def inversesqrt(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, lambda x: 1 / math.sqrt(x), "inversesqrt")

def pow(x: genType, y: genType)->genType:
    return x ** y

def exp(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.exp, "exp")

def exp2(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, lambda x: 2 ** x, "exp2")

def exp10(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, lambda x: 10 ** x, "exp10")

def log(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.log, "log")

def log2(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, lambda x: math.log(x) / math.log(2), "log2")

def log10(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, lambda x: math.log(x) / math.log(10), "log10")

def sin(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.sin, "sin")

def cos(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.cos, "cos")

def tan(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.tan, "tan")

def asin(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.asin, "asin")

def acos(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.acos, "acos")

def atan(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.atan, "atan")

def sinh(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.sinh, "sinh")

def cosh(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.cosh, "cosh")

def tanh(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.tanh, "tanh")

def asinh(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.asinh, "asinh")

def acosh(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.acosh, "acosh")

def atanh(x:Union[genType, Number])->Union[genType, float]:
    return _single_op(x, math.atanh, "atanh")

def _reject_mat(x:genType, op_name:str)->None:
    """Refuse a genMat for the operations that only make sense on a flat container.

    length and dot sum over slots. Recursing into a matrix would return the norm
    of its flattened element list, which is a number nobody wants: GfLength and
    GfDot are vector and quaternion operations and neither has a matrix form.
    Raising here also takes normalize(matrix) down with them, correctly.
    """
    if isinstance(x, genMat):
        raise TypeError(f"{op_name} not defined for '{x.__class__.__name__}'")

def length(x:Union[genType, Number])->float:
    if is_number(x):
        # builtins.abs, not the module-level genType one.
        return builtins.abs(cast(float, x))

    if not isinstance(x, genType):
        raise TypeError(f"length not supported for type {x.__class__.__name__}")

    _reject_mat(x, "length")

    sum: float = 0
    for i in range(x._slot_count()):
        sum += x[i] ** 2

    return math.sqrt(sum)

def normalize(x:Union[genType, Number])->Union[genType, float]:
    # Through _bin_op rather than `x / length(x)`: length() returns a plain float
    # for a scalar input, and `/` between a genType and a float is what the
    # operator path handles.
    return _bin_op(x, length(x), _div, "/")

def distance(x: genType, y: genType)->float:
    return length(x - y)

def dot(x: genType, y: genType)->float:
    if not isinstance(x, genType) or not isinstance(y, genType) or not x._is_homo(y):
        raise TypeError(f"not defined dot between '{x.__class__.__name__}' and '{y.__class__.__name__}'")

    _reject_mat(x, "dot")
    _reject_mat(y, "dot")

    sum: float = 0
    for i in range(x._slot_count()):
        sum += x[i] * y[i]

    return sum

def cross(x: genVec3, y: genVec3)->genVec3:
    if not isinstance(x, genVec3) or not isinstance(y, genVec3):
        raise TypeError(f"not defined cross between '{x.__class__.__name__}' and '{y.__class__.__name__}'")

    result_dtype:type = ctypes.c_double if (x.dtype == ctypes.c_double or y.dtype == ctypes.c_double) else ctypes.c_float
    result_type:type = genVec.vec_type(result_dtype, 3)
    return result_type(x.y * y.z - x.z * y.y, x.z * y.x - x.x * y.z, x.x * y.y - x.y * y.x)

def reflect(I:genVec, N:genVec)->genType:  # noqa: E741 - GLSL spells the incident vector `I`
    return I - 2 * dot(I, N) * N

def refract(I:genVec, N:genVec, eta:float)->genType:  # noqa: E741 - GLSL spells the incident vector `I`
    return I - (eta * dot(I, N) + math.sqrt(1 - eta * eta * (1 - dot(I, N) * dot(I, N)))) * N

def faceforward(N:genVec, I:genVec, Nref:genVec)->genType:  # noqa: E741 - GLSL spells the incident vector `I`
    return (N if dot(Nref, I) < 0 else -N)

def determinant(m:genMat)->float:
    if not isinstance(m, genMat) or m.rows != m.cols:
        raise TypeError(f'not defined determinant for {m.__class__.__name__}')

    if m.rows == 2:
        return m.at(0, 0) * m.at(1, 1) - m.at(0, 1) * m.at(1, 0)

    if m.rows == 3:
        return (m.at(0, 0) * m.at(1, 1) * m.at(2, 2) +
                m.at(0, 1) * m.at(1, 2) * m.at(2, 0) +
                m.at(0, 2) * m.at(1, 0) * m.at(2, 1) -
                m.at(0, 2) * m.at(1, 1) * m.at(2, 0) -
                m.at(0, 0) * m.at(1, 2) * m.at(2, 1) -
                m.at(0, 1) * m.at(1, 0) * m.at(2, 2))

    if m.rows == 4:
        det = 0.0
        for row in range(4):
            submatrix_data = []
            for i in range(4):
                if i == row:
                    continue
                for j in range(1, 4):
                    submatrix_data.append(m.at(i, j))

            submatrix_type = genMat.mat_type(ctypes.c_double, (3, 3))
            submatrix = submatrix_type()
            for i in range(3):
                for j in range(3):
                    submatrix[i, j] = submatrix_data[i * 3 + j]

            cofactor = ((-1) ** row) * m.at(row, 0) * determinant(submatrix)
            det += cofactor

        return det

    raise TypeError(f'not defined determinant for a {m.rows}x{m.cols} matrix')

def transpose(m:genMat)->genMat:
    if not isinstance(m, genMat):
        raise TypeError(f'not defined transpose for {m.__class__.__name__}')

    result_type:type = genMat.mat_type(m.dtype, m.shape[::-1])
    result:genMat = result_type()
    for i in range(result.rows):
        for j in range(result.cols):
            result.put(i, j, m.at(j, i))

    return result

def trace(m:genMat)->float:
    if not isinstance(m, genMat) or m.rows != m.cols:
        raise TypeError(f'not defined trace for {m.__class__.__name__}')

    trace:float = 0.0
    for i in range(m.rows):
        trace += m.at(i, i)

    return trace

def conjugate(m:genQuat)->genQuat:
    if not isinstance(m, genQuat):
        raise TypeError(f'not defined conjugate for {m.__class__.__name__}')

    return m.__class__(m.w, -m.x, -m.y, -m.z)

def inverse(m:Union[genMat, genQuat])->Union[genMat, genQuat]:
    if isinstance(m, genQuat):
        return cast(genQuat, conjugate(m) / length(m))

    if not isinstance(m, genMat) or m.rows != m.cols:
        raise TypeError(f'not defined inverse for {m.__class__.__name__}')

    result_dtype:type = (ctypes.c_double if m.dtype == ctypes.c_double else ctypes.c_float)
    result_type:type = genMat.mat_type(result_dtype, m.shape)

    det = determinant(m)
    if det == 0:
        raise ValueError("Matrix is not invertible (determinant is zero)")

    result = result_type()

    if m.rows == 2:
        result.put(0, 0, m.at(1, 1) / det)
        result.put(1, 1, m.at(0, 0) / det)
        result.put(0, 1, -m.at(0, 1) / det)
        result.put(1, 0, -m.at(1, 0) / det)
        return result

    if m.rows == 3:
        result.put(0, 0, (m.at(1, 1) * m.at(2, 2) - m.at(1, 2) * m.at(2, 1)) / det)
        result.put(1, 0, -(m.at(1, 0) * m.at(2, 2) - m.at(1, 2) * m.at(2, 0)) / det)
        result.put(2, 0, (m.at(1, 0) * m.at(2, 1) - m.at(1, 1) * m.at(2, 0)) / det)

        result.put(0, 1, -(m.at(0, 1) * m.at(2, 2) - m.at(0, 2) * m.at(2, 1)) / det)
        result.put(1, 1, (m.at(0, 0) * m.at(2, 2) - m.at(0, 2) * m.at(2, 0)) / det)
        result.put(2, 1, -(m.at(0, 0) * m.at(2, 1) - m.at(0, 1) * m.at(2, 0)) / det)

        result.put(0, 2, (m.at(0, 1) * m.at(1, 2) - m.at(0, 2) * m.at(1, 1)) / det)
        result.put(1, 2, -(m.at(0, 0) * m.at(1, 2) - m.at(0, 2) * m.at(1, 0)) / det)
        result.put(2, 2, (m.at(0, 0) * m.at(1, 1) - m.at(0, 1) * m.at(1, 0)) / det)

        return result

    if m.rows == 4:
        for i in range(4):
            for j in range(4):
                submatrix_data = []
                for row in range(4):
                    if row == i:
                        continue
                    for col in range(4):
                        if col == j:
                            continue
                        submatrix_data.append(m.at(row, col))

                submatrix_type = genMat.mat_type(ctypes.c_double, (3, 3))
                submatrix = submatrix_type()
                for row in range(3):
                    for col in range(3):
                        submatrix[row, col] = submatrix_data[row * 3 + col]

                cofactor = ((-1) ** (i + j)) * determinant(submatrix)
                result[j, i] = cofactor / det

        return result

    raise TypeError(f'not defined inverse for a {m.rows}x{m.cols} matrix')

def matrixCompMult(x:genMat, y:genMat)->genMat:
    if not isinstance(x, genMat) or not isinstance(y, genMat) or x.shape != y.shape:
        raise TypeError(f"not defined matrixCompMult between '{x.__class__.__name__}' and '{y.__class__.__name__}'")

    result_type:type = genType._bin_op_type('*', x, y)
    result:genMat = result_type()

    for i in range(x.rows):
        for j in range(y.cols):
            result.put(i, j, x.at(i, j) * y.at(i, j))

    return result

def outerProduct(x:genVec, y:genVec)->genMat:
    if not isinstance(x, genVec) or not isinstance(y, genVec):
        raise TypeError(f"not defined outerProduct between '{x.__class__.__name__}' and '{y.__class__.__name__}'")

    # Matrices here are square-only, so x and y have to agree on length: a
    # mismatched pair used to build a len(y)xlen(y) matrix and then write len(x)
    # rows into it, which raised IndexError from deep inside ctypes.
    if len(x) != len(y):
        raise TypeError(f"not defined outerProduct between '{x.__class__.__name__}' and '{y.__class__.__name__}': lengths differ ({len(x)} != {len(y)})")

    result_dtype:type = genType._bin_op_dtype('*', x.dtype, y.dtype)
    result_type:type = genMat.mat_type(result_dtype, (len(y), len(x)))
    result:genMat = result_type()

    for i in range(len(x)):
        for j in range(len(y)):
            result.put(i, j, cast(float, x[i]) * cast(float, y[j]))

    return result

def lessThan(x:genType, y:genType)->genType:
    return x < y

def lessThanEqual(x:genType, y:genType)->genType:
    return x <= y

def greaterThan(x:genType, y:genType)->genType:
    return x > y

def greaterThanEqual(x:genType, y:genType)->genType:
    return x >= y

def equal(x:genType, y:genType)->genType:
    return x._compare_op("==", y)

def notEqual(x:genType, y:genType)->genType:
    return x._compare_op("!=", y)

def _slot_truth(x:genType)->List[bool]:
    """Truth value of every scalar leaf, flattened.

    A matrix slot is a row, and a row is truthy for as long as it has a
    non-zero length -- so `any` over matrix slots reported True for an all-zero
    matrix. Descending to the leaves is what makes the answer mean what it says.
    """
    values:List[bool] = []
    for i in range(x._slot_count()):
        slot = x[i]
        if isinstance(slot, genType):
            values.extend(_slot_truth(slot))
        else:
            values.append(bool(slot))

    return values

def any(x:genType)->bool:
    if not isinstance(x, genType):
        return bool(x)

    # `any`/`all` below shadow the builtins, so the comprehension needs builtins.any
    return builtins.any(_slot_truth(x))

def all(x:genType)->bool:
    if not isinstance(x, genType):
        return bool(x)

    return builtins.all(_slot_truth(x))

def not_(x:Union[genType, Number])->Union[genType, float]:
    if not isinstance(x, genType):
        return (not x)

    # gen_type hands back the class, so instantiate before assigning elements.
    btype = genType.gen_type(x.math_form, ctypes.c_bool, x.shape)
    result:genType = btype()
    for i in range(x._slot_count()):
        slot = x[i]
        result[i] = _single_op(slot, lambda v: not v, "not_") if isinstance(slot, genType) else not slot

    return result

def sizeof(x:genType)->int:
    return ctypes.sizeof(cast(Any, x))
