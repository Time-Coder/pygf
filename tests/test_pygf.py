"""Regression tests for pygf, the ctypes-backed graphics math library.

Every check here corresponds to a defect that type checking or review surfaced
but that the other smoke scripts never exercised, so it would otherwise stay
broken.

Original gf-layer defects (inherited from pyusd/gf):

  * funcs.abs / min / max went through __builtins__, which is a dict inside an
    imported module, so all three raised AttributeError.
  * funcs.outerProduct called genMat.gen_type, which genMat does not define; it
    inherits genType.gen_type and needs a math_form argument.
  * genType.gen_type builds "bool3" / "matrix2b" / "quatb" module names for a
    c_bool dtype, but only the numeric vector/matrix/quat modules existed, so
    every ordering comparison and not_() raised ModuleNotFoundError.
  * funcs.not_ assigned to the class gen_type returns instead of an instance.
  * genMatIterator implemented __next__ but not __iter__.
  * genMat._iop wrote the product back with self[:] = product[:], and a slice
    index matched neither branch of __setitem__, so `matrix *= matrix` silently
    left the matrix unchanged.
  * The genQuat w/x/y/z getters read `super().w`, but the ctypes field
    descriptor lives on the concrete subclass ahead of genQuat in the MRO, so
    super() never sees it and the body raised AttributeError whenever called.
  * genType._op and genType._compare_op took any operand without checking it. A
    duck-typed object that is neither a number nor a shape-matched genType was
    treated as one scalar, so `double3(...) + attribute` computed a result type
    from the attribute's delegated dtype and then assigned a whole vector into
    every element. They return NotImplemented instead, which is what lets Python
    reach the other operand's reflected method.
  * The element-wise loops in _op, _rop, _iop, _compare_op and _compare_rop all
    counted slots with len(). That is right for a flat genVec or genQuat and wrong
    for a genMat, whose __len__ is ctypes' flat element count while m[i] is a row,
    so matrix addition, subtraction and scalar multiply raised IndexError. genMat
    only rescues `*` itself, which is why multiplication worked.
  * genQuat.__init__ passed [1, 0, 0, 0] to ctypes.Structure.__init__ for the
    no-argument case. ctypes wants one positional value per field, so it raised
    "must be real number, not list" -- and since every quaternion operator
    default-constructs its result, all of quaternion arithmetic was dead.

Defects found while extracting gf out into pygf (they had to be fixed *before*
the package could be published, because a shared dependency freezes its API):

  * funcs.py never received the _slot_count treatment the operators got, so
    abs / sqrt / min / max / clamp / length / dot / any / all / not_ were broken
    for matrices. Seven of twelve representative calls raised.
  * The fix is recursion, not just the slot count: a matrix slot is a row, so
    `builtins.abs(row)` still fails after the count is corrected.
  * funcs.any on a matrix walked rows, and a row is truthy for as long as it has
    a length, so any(zero_matrix) silently answered True.
  * genType.__eq__ / __ne__ / __neg__ and genMat.__contains__ counted
    len(self) elements while self[i] yields a row, so matrix == matrix raised
    IndexError.
  * genMat.rows and genMat.cols were shape[1] and shape[0], i.e. transposed.
    Square-only matrices hid it; __getitem__ mixed the two conventions against
    ctypes' flat storage.
  * genMat.mat_type silently truncated a non-square request to shape[0]x
    shape[0], so outerProduct(float3, float2) built a 2x2 and wrote 3 rows.
  * funcs.roundEven passed this module's own `round` instead of builtins.round,
    so it was byte-for-byte half-away-from-zero rather than banker's rounding.
  * funcs.fract used trunc instead of floor, so fract(-1.5) was -0.5.
  * patch_nparray replaced numpy.array globally and rewrote matrix rows into
    0-dimensional structured arrays, so np.array(matrix3d()) raised IndexError
    where unpatched numpy returns a flat (9,) array.
  * gen_type could not build a uint vector: 'uint' was in the dtype name map
    but no uint2/3/4 modules existed.
  * half2/3/4 were float2/3/4 subclasses with c_float fields, so there was no
    16-bit half anywhere despite half being a supported element type.
  * 9 modules shipped both a .py and a .pyi, which makes a type checker see two
    distinct nominal types for one class.

Run with:
    python tests/test_pygf.py
"""

import ctypes
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pygf import funcs
from pygf.double3 import double3
from pygf.float2 import float2
from pygf.float3 import float3
from pygf.genMat import genMat
from pygf.genQuat import genQuat
from pygf.genVec2 import genVec2
from pygf.genVec3 import genVec3
from pygf.genVec4 import genVec4
from pygf.half3 import half3
from pygf.half4 import half4
from pygf.int3 import int3
from pygf.matrix3d import matrix3d
from pygf.matrix4d import matrix4d
from pygf.quatd import quatd
from pygf.uint3 import uint3

FAILURES = []


def check(label, actual, expected):
    # A numpy array does not compare with `!=` -- it raises "truth value of an
    # array is ambiguous" -- so the result is reduced element-wise first. `Any`
    # because a genVec __getitem__ is typed to yield either a scalar or a genVec,
    # and `tuple(...)` of that union is not provably an Iterable.
    same:Any = actual == expected
    if isinstance(same, np.ndarray):
        same = bool(same.all())

    if not same:
        FAILURES.append(f"{label}: got {actual!r}, expected {expected!r}")
        print(f"FAIL {label}: got {actual!r}, expected {expected!r}")
    else:
        print(f"PASS {label}")


def check_raises(label, exc, fn):
    try:
        result = fn()
    except exc:
        print(f"PASS {label}")
        return
    except Exception as e:  # noqa: BLE001 - report the wrong exception type
        FAILURES.append(f"{label}: raised {type(e).__name__}, expected {exc.__name__}")
        print(f"FAIL {label}: raised {type(e).__name__}, expected {exc.__name__}")
        return

    FAILURES.append(f"{label}: returned {result!r}, expected {exc.__name__}")
    print(f"FAIL {label}: returned {result!r}, expected {exc.__name__}")


def vec_type(cls, dtype, size):
    return cls.vec_type(dtype, size)


def as_matrix(value:Any)->Any:
    """A value the checker should treat as a matrix.

    An in-place operator returns whatever `_iop` returns, which is annotated
    against `genType` -- the shared base, which has no `at`. The value really is
    the same matrix instance (that is what `_iop` guarantees for `*=`), so this
    narrows the static type without changing what runs.
    """
    return value


def components(container:Any)->Tuple[Any, ...]:
    """The scalars of a vector, as a tuple.

    `tuple(vec)` is what a caller would write, but a genVec __getitem__ is typed
    to yield either a scalar or another genVec, so the union is not provably
    Iterable and `ty` rejects it. Iterating is what genVec.__iter__ does, so this
    is the same walk with the type made explicit.
    """
    return tuple(iter(container))


V2 = vec_type(genVec2, ctypes.c_float, 2)
V3 = vec_type(genVec3, ctypes.c_float, 3)
V4 = vec_type(genVec4, ctypes.c_float, 4)
M2 = genMat.mat_type(ctypes.c_double, (2, 2))
M3 = genMat.mat_type(ctypes.c_double, (3, 3))
QUAT = genQuat.quat_type(ctypes.c_double)


def as_rows(matrix):
    return [[matrix.at(i, j) for j in range(matrix.cols)] for i in range(matrix.rows)]


def row0(matrix):
    return [matrix.at(0, j) for j in range(matrix.cols)]


# --- builtins reached through the module, not through __builtins__ ------------
print("--- scalar builtins ---")
check("abs of a negative scalar", funcs.abs(-2.5), 2.5)
check("length of a negative scalar", funcs.length(-2.5), 2.5)
check("normalize of a scalar", funcs.normalize(2.0), 1.0)
check("min of two vectors", components(funcs.min(V3(3, 4, 0), V3(1, 1, 1))), (1.0, 1.0, 0.0))
check("max of two vectors", components(funcs.max(V3(3, 4, 0), V3(1, 1, 1))), (3.0, 4.0, 1.0))

# --- outerProduct needs the matrix factory, not gen_type ---------------------
print()
print("--- outerProduct ---")
outer = funcs.outerProduct(V3(1, 2, 3), V3(4, 5, 6))
check("outerProduct shape", outer.shape, (3, 3))
check("outerProduct rows", [components(outer[i]) for i in range(3)],
      [(4.0, 5.0, 6.0), (8.0, 10.0, 12.0), (12.0, 15.0, 18.0)])
# Matrices are square-only, so vectors of different lengths have to be refused.
# This built a len(y)xlen(y) matrix and then wrote len(x) rows into it, raising
# IndexError out of ctypes.
check_raises("outerProduct refuses unequal lengths", TypeError,
             lambda: funcs.outerProduct(float3(1, 2, 3), float2(1, 2)))
check_raises("mat_type refuses a non-square shape", ValueError,
             lambda: genMat.mat_type(ctypes.c_double, (3, 2)))

# --- bool result types have to exist ----------------------------------------
print()
print("--- vector comparisons produce bool vectors ---")
check("float2 less-than", components(V2(1, 2) < V2(2, 1)), (True, False))
check("float3 less-than", components(V3(1, 2, 3) < V3(3, 2, 1)), (True, False, False))
check("float3 greater-equal", components(V3(1, 2, 3) >= V3(3, 2, 1)), (False, True, True))
check("float4 less-equal", components(V4(1, 2, 3, 4) <= V4(4, 3, 2, 1)), (True, True, False, False))
check("scalar against vector", components(V3(1, 2, 3) > 2.0), (False, False, True))
check("bool3 result type", type(V3(1, 2, 3) < V3(3, 2, 1)).__name__, "bool3")
check("bool2 result type", type(V2(1, 2) < V2(2, 1)).__name__, "bool2")
check("bool4 result type", type(V4(1, 2, 3, 4) <= V4(4, 3, 2, 1)).__name__, "bool4")

# --- not_ instantiates its result -------------------------------------------
print()
print("--- not_ ---")
check("not_ of a float2", components(funcs.not_(V2(1, 0))), (False, True))
check("not_ of a float3", components(funcs.not_(V3(1, 0, 1))), (False, True, False))
check("not_ of a float4", components(funcs.not_(V4(0, 1, 0, 1))), (True, False, True, False))
check("not_ result type", type(funcs.not_(V3(1, 0, 1))).__name__, "bool3")
check("not_ of a matrix descends into rows",
      as_rows(funcs.not_(matrix3d(1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0))),
      [[False, True, True], [True, False, True], [True, True, False]])
check("not_ matrix result type", type(funcs.not_(matrix3d())).__name__, "matrix3b")

# --- matrix comparisons stay element-wise ------------------------------------
print()
print("--- matrix comparisons ---")
mat_a, mat_b = M2(), M2()
mat_a[0, 0], mat_a[1, 1] = 1.0, 5.0
mat_b[0, 0], mat_b[1, 1] = 2.0, 3.0

check("matrix2d less-than", as_rows(mat_a < mat_b), [[True, False], [False, False]])
check("matrix2d greater-than", as_rows(mat_a > mat_b), [[False, False], [False, True]])
check("matrix2d greater-equal", as_rows(mat_a >= mat_b), [[False, True], [True, True]])
check("scalar against matrix", as_rows(mat_a > 2.0), [[False, False], [False, True]])
check("matrix3d less-equal type", type(M3() <= M3()).__name__, "matrix3b")

# --- determinant / inverse are numeric, and *= actually writes back -----------
print()
print("--- matrix arithmetic ---")
for size, MatrixT in ((2, M2), (3, M3)):
    mat = MatrixT()
    other = MatrixT()
    for i in range(size):
        for j in range(size):
            mat.put(i, j, float((i * 3 + j * 7) % 5) + 0.5)
            other.put(i, j, float((i + j) % 4) + 1.0)

    product = mat * other
    check(f"matrix{size}d product rows",
          [product.at(i, j) for i in range(size) for j in range(size)],
          [sum(mat.at(i, k) * other.at(k, j) for k in range(size)) for i in range(size) for j in range(size)])

    expected_det = funcs.determinant(product)
    check(f"matrix{size}d determinant is a float", isinstance(expected_det, float), True)

    inverse = funcs.inverse(product)
    identity = product * inverse
    check(f"matrix{size}d M * inverse(M) is the identity",
          all(abs(identity.at(i, j) - (1.0 if i == j else 0.0)) < 1e-9
              for i in range(size) for j in range(size)),
          True)

# in-place multiply used to be a no-op because a slice index matched no branch
# of __setitem__.
diag = M2()
diag.put(0, 0, 1.0)
diag.put(1, 1, 2.0)
factor = M2()
factor.put(0, 0, 3.0)
factor.put(1, 1, 4.0)
diag *= factor
check("matrix2d *= writes back", [diag.at(i, i) for i in range(2)], [3.0, 8.0])

# --- a matrix iterator is itself iterable -----------------------------------
print()
print("--- matrix iteration ---")
matrix = M2()
iterator = iter(matrix)
check("iter of an iterator", type(iter(iterator)).__name__, "genMatIterator")
check("iter of a matrix", type(iterator).__name__, "genMatIterator")
check("rows from a matrix", len(list(matrix)), 2)

# --- quaternion fields come from the generated _fields_ ----------------------
print()
print("--- quaternion ---")
quat = QUAT(1.0, 0.0, 0.0, 0.0)
check("quat w", quat.w, 1.0)
check("quat xyz", components(quat.xyz), (0.0, 0.0, 0.0))
# The ctypes field descriptor sits on the concrete subclass, ahead of genQuat in
# the MRO, so instance reads never reach these properties. Calling them directly
# used to raise AttributeError because super() cannot see the descriptor.
rich = QUAT(1.0, 2.0, 3.0, 4.0)
check("quat w getter called directly", genQuat.w.fget(rich), 1.0)
check("quat x getter called directly", genQuat.x.fget(rich), 2.0)
check("quat y getter called directly", genQuat.y.fget(rich), 3.0)
check("quat z getter called directly", genQuat.z.fget(rich), 4.0)

# An operand that is neither a number nor a shape-matched genType has to come back
# as NotImplemented, so Python falls through to the other side's reflected method
# instead of this side inventing an element-wise result out of the whole object.
check("vec op rejects a foreign operand", double3(1, 2, 3).__add__("x"), NotImplemented)
check("vec reflected op rejects a foreign operand", double3(1, 2, 3).__radd__("x"), NotImplemented)
check(
    "vec comparison rejects a foreign operand",
    double3(1, 2, 3).__gt__("x"),
    NotImplemented,
)
check_raises("mixing a vec with a str raises", TypeError, lambda: double3(1, 2, 3) + "x")

# A genMat indexes a row at a time while __len__ is the flat element count, so the
# element-wise loops count rows. All four of these raised IndexError before.
print()
print("--- matrix slots count rows ---")
check("matrix + matrix", as_rows(matrix3d() + matrix3d()),
      [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
check("matrix - matrix", as_rows(matrix3d() - matrix3d()),
      [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
check("matrix * scalar", as_rows(matrix3d() * 2),
      [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
check("4x4 matrix + 4x4 matrix", row0(matrix4d() + matrix4d()), [2.0, 0.0, 0.0, 0.0])
scaled = matrix3d()
scaled *= 3
check("matrix *= scalar", as_matrix(scaled).at(1, 1), 3.0)
prod = matrix4d()
prod[0, 0] = 2.0
doubler = matrix4d()
doubler[0, 0] = 3.0
prod *= doubler
check("matrix *= matrix writes the product back", as_matrix(prod).at(0, 0), 6.0)
check("matrix *= matrix leaves the rest", as_matrix(prod).at(2, 2), 1.0)
untouched = matrix4d()
untouched *= matrix4d()
check("matrix *= identity is a no-op", as_matrix(untouched).at(0, 0), 1.0)

# genQuat has to default-construct, because every quaternion operator builds its
# result that way. This raised "must be real number, not list".
check("quat default construct", str(quatd()), "quatd(1.0, 0.0, 0.0, 0.0)")
identity = quatd(1.0, 0.0, 0.0, 0.0)
check("quat + quat", str(identity + identity), "quatd(2.0, 0.0, 0.0, 0.0)")
check("quat * scalar", str(identity * 2), "quatd(2.0, 0.0, 0.0, 0.0)")
check("quat * quat is the Hamilton product", str(identity * identity), "quatd(1.0, 0.0, 0.0, 0.0)")
# A half turn about X maps (x, y, z) to (x, -y, -z).
check("quat * vec rotates", str(quatd(0.0, 1.0, 0.0, 0.0) * double3(2.0, 3.0, 4.0)), "double3(2.0, -3.0, -4.0)")

# --- element-wise funcs on matrices -----------------------------------------
# funcs.py never got the _slot_count treatment the operators got. The fix needs
# recursion as well as the count: a matrix slot is a row, so passing the row
# whole to builtins.abs still fails once the count is right.
print()
print("--- element-wise funcs on matrices ---")
three = matrix3d(3.0)
check("abs of a 4x4", row0(funcs.abs(matrix4d())), [1.0, 0.0, 0.0, 0.0])
# A matrix takes exactly rows*cols scalars, so this spells out all nine.
check("abs of a matrix of -3", as_rows(funcs.abs(matrix3d(0.0, -3.0, 0.0,
                                                         0.0, -6.0, 0.0,
                                                         0.0, 0.0, -9.0))),
      [[0.0, 3.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 9.0]])
check("sqrt of the identity", as_matrix(funcs.sqrt(matrix4d())).at(0, 0), 1.0)
check("floor of a matrix", row0(funcs.floor(matrix3d(1.5, -1.5, 2.5,
                                                    0.0, 0.0, 0.0,
                                                    0.0, 0.0, 0.0))), [1.0, -2.0, 2.0])
check("min of two matrices", row0(funcs.min(matrix3d(), three)), [1.0, 0.0, 0.0])
check("max of two matrices", row0(funcs.max(matrix3d(), three)), [3.0, 0.0, 0.0])
check("clamp of a matrix", as_rows(funcs.clamp(three, 2.0, 2.0)),
      [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
check("min of a matrix and a scalar", row0(funcs.min(three, 1.0)), [1.0, 0.0, 0.0])
check("matrixCompMult", row0(funcs.matrixCompMult(matrix3d(), three)), [3.0, 0.0, 0.0])

# --- any / all descend to the leaves ----------------------------------------
# A row is truthy for as long as it has a length, so any() over matrix slots
# answered True for an all-zero matrix.
print()
print("--- any / all on matrices ---")
zero = matrix3d(0.0)
check("any of a zero matrix", funcs.any(zero), False)
check("any of the identity", funcs.any(matrix3d()), True)
check("all of the identity", funcs.all(matrix3d()), False)
check("any of an all-9 diagonal", funcs.any(matrix3d(9.0)), True)
# `all` means "every element is non-zero", the same as it does for a vector.
check("all of a vector of ones", funcs.all(double3(1, 1, 1)), True)
check("all of a vector with a zero", funcs.all(double3(1, 0, 1)), False)

# --- matrix equality, negation and containment ------------------------------
# These counted len(self) elements while self[i] yields a row, so `m == m` raised
# IndexError rather than returning True.
print()
print("--- matrix equality / negation / containment ---")
check("matrix equals itself", matrix3d() == matrix3d(), True)
check("zero matrix equals a zero matrix", zero == matrix3d(0.0), True)
check("identity differs from 3x identity", matrix3d() == three, False)
check("identity is not 3x identity", matrix3d() != three, True)
check("negated identity", as_rows(-matrix3d()),
      [[-1.0, -0.0, -0.0], [-0.0, -1.0, -0.0], [-0.0, -0.0, -1.0]])
check("a zero is in the zero matrix", 0.0 in zero, True)
check("a nine is not in the zero matrix", 9.0 in zero, False)
check("a three is on the diagonal", 3.0 in three, True)

# --- length and dot are vector and quaternion operations --------------------
# Recursing into a matrix would return the norm of its flattened element list,
# which is a number nobody wants; GfLength and GfDot have no matrix form.
print()
print("--- length / dot reject matrices ---")
check_raises("length of a matrix", TypeError, lambda: funcs.length(matrix3d()))
check_raises("dot of two matrices", TypeError, lambda: funcs.dot(matrix3d(), matrix3d()))
check_raises("normalize of a matrix", TypeError, lambda: funcs.normalize(matrix3d()))
check("length of a vector still works", funcs.length(double3(3, 4, 0)), 5.0)
check("length of a quaternion still works", funcs.length(quatd(1.0, 0.0, 0.0, 0.0)), 1.0)

# --- roundEven is banker's rounding -----------------------------------------
# funcs.py defines its own round, which shadowed the builtin inside the module,
# so passing `round` here made roundEven identical to round.
print()
print("--- rounding ---")
check("round is half away from zero", funcs.round(double3(1.5, 2.5, -2.5)), double3(2.0, 3.0, -3.0))
check("roundEven is half to even", funcs.roundEven(double3(1.5, 2.5, -2.5)), double3(2.0, 2.0, -2.0))
check("roundEven of -2.5", funcs.roundEven(double3(-2.5)), double3(-2.0))
check("roundEven of a matrix", row0(funcs.roundEven(matrix3d(1.5, 2.5, 0.0,
                                                          0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0))), [2.0, 2.0, 0.0])
# GLSL defines fract(x) = x - floor(x). trunc made fract(-1.5) return -0.5.
check("fract of a negative value", funcs.fract(double3(-1.5)), double3(0.5))
check("fract of a positive value", funcs.fract(double3(2.25)), double3(0.25))

# --- rows and cols are not transposed ---------------------------------------
print()
print("--- shape metadata ---")
check("matrix rows", matrix3d().rows, 3)
check("matrix cols", matrix3d().cols, 3)
check("rows of a row vector", double3(1, 2, 3)[0], 1.0)
check("row 0 of a matrix has one value per column", components(matrix3d()[0]), (1.0, 0.0, 0.0))

# --- uint vectors exist ------------------------------------------------------
# 'uint' was in the dtype name map, so gen_type built the module name "uint3" --
# for a module that did not exist.
print()
print("--- unsigned integer vectors ---")
check("uint3 default construct", list(uint3()), [0, 0, 0])
check("uint3 values", list(uint3(1, 2, 3)), [1, 2, 3])
check("uint3 dtype", uint3().dtype, ctypes.c_uint)
check("int3 still works", list(int3(1, 2, 3)), [1, 2, 3])

# --- half is a real 16-bit float --------------------------------------------
# half3 used to subclass float3 with c_float fields, so there was no 16-bit
# half anywhere even though half is a supported element type.
print()
print("--- 16-bit half ---")
check("half3 stores 2 bytes per element", ctypes.sizeof(half3), 6)
check("half4 stores 2 bytes per element", ctypes.sizeof(half4), 8)
check("half3 rounds to binary16", list(half3(1.0, 0.333251953125, -2.5)), [1.0, 0.333251953125, -2.5])
# 1.0001 is not representable, so it rounds to the nearest binary16, which is 1.0
check("half3 loses precision", float(half3(1.0001)[0]), 1.0)
check("half3 dtype", half3().dtype, ctypes.c_uint16)
check("half max magnitude", float(half3(65504.0)[0]), 65504.0)
check("half3 is not a float3", isinstance(half3(), float3), False)
check("half3 values read back as floats",
      [type(v).__name__ for v in half3(1.0, 2.0, 3.0)], ["float", "float", "float"])

# --- numpy interop -----------------------------------------------------------
print()
print("--- numpy interop ---")
check("np.array of a vector is its components", np.array(double3(1, 2, 3)), [1.0, 2.0, 3.0])
check("np.array of a vector shape", np.array(double3(1, 2, 3)).shape, (3,))
check("np.array of a matrix keeps its shape", np.array(matrix3d(0.0)).shape, (3, 3))
check("np.array of a matrix4d keeps its shape", np.array(matrix4d(0.0)).shape, (4, 4))
check("np.array of a list of vectors", np.array([double3(1, 2, 3)]).shape, (1, 3))
# A half field is a uint16 bit pattern, so the patch has to decode it rather
# than hand numpy the raw integer.
check("np.array of a half vector", np.array(half3(1.0, 2.0, 3.0)), [1.0, 2.0, 3.0])

print()
if FAILURES:
    print(f"{len(FAILURES)} failure(s):")
    for failure in FAILURES:
        print(f"  {failure}")
    raise SystemExit(1)

print("all pygf checks passed")
