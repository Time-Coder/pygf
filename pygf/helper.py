import ctypes
import importlib
import itertools
from collections.abc import Iterable
from decimal import Decimal
from types import ModuleType
from typing import Any, Dict, List, TypeAlias, Union

import numpy as np

# The scalar side of the gf type split, as opposed to the genType containers. It
# lives here rather than in genType so is_number can use it without a cycle.
Number: TypeAlias = Union[float, int, bool, Decimal]

_module_map:Dict[str, ModuleType] = {}

def in_annotations(name:str, cls:type)->bool:
    for klass in cls.__mro__:
        if hasattr(klass, '__annotations__') and name in klass.__annotations__:
            return True

    return False

def from_import(module_name:str, attr_name:str)->type:
    if module_name not in _module_map:
        module = None
        if module_name.startswith("."):
            module = importlib.import_module(module_name, package=__package__)
        else:
            module = importlib.import_module(module_name)
        _module_map[module_name] = module

    return getattr(_module_map[module_name], attr_name)

def is_number(value:Any)->bool:
    """Whether ``value`` is a scalar rather than one of the gf containers.

    Deliberately not a TypeGuard: the accepted types include ctypes scalars and
    the hand-written dtypes wrappers, so a guard would claim more than it can
    prove and pushes errors downstream instead of removing them.
    """
    return isinstance(value, (
        float, bool, int, Decimal,
        ctypes.c_bool, ctypes.c_int8, ctypes.c_uint8,
        ctypes.c_int16, ctypes.c_uint16,
        ctypes.c_int32, ctypes.c_uint32,
        ctypes.c_int64, ctypes.c_uint64,
        ctypes.c_float, ctypes.c_double, ctypes.c_longdouble
    )) or value.__class__.__name__ in (
        'int8', 'int16', 'int32', 'int64',
        'uint8', 'uint16', 'uint32', 'uint64',
        'float16', 'float32', 'float64', 'float128',
        'bool'
    )

def generate_getter_swizzles(char_sets:Iterable[str])->List[str]:
    result:List[str] = []

    for char_set in char_sets:
        for length in range(1, 4 + 1):
            for combo in itertools.product(char_set, repeat=length):
                swizzle = ''.join(combo)
                result.append(swizzle)

    return result

def generate_setter_swizzles(char_sets:Iterable[str])->List[str]:
    result:List[str] = []

    for char_set in char_sets:
        for length in range(1, len(char_set) + 1):
            for combo in itertools.permutations(char_set, length):
                swizzle = ''.join(combo)
                result.append(swizzle)

    return result

def generate_swizzle_defines(type_name:str, dtype_name:str, char_sets:List[str])->str:
    result:str = ""
    vec_basename = type_name[:-1]
    getter_swizzles:List[str] = generate_getter_swizzles(char_sets)
    setter_swizzles:List[str] = generate_setter_swizzles(char_sets)
    for swizzle in getter_swizzles:
        return_type_name:str = dtype_name
        input_type_name:str = "Union[bool, int, float]"

        n_swizzle = len(swizzle)
        if n_swizzle > 1:
            return_type_name:str = vec_basename + str(n_swizzle)
            input_type_name:str = f"Union[bool, int, float, genVec{n_swizzle}]"

        result += f"""
    @property
    def {swizzle}(self)->{return_type_name}: ...
"""

        if swizzle in setter_swizzles:
            result += f"""
    @{swizzle}.setter
    def {swizzle}(self, value:{input_type_name})->None: ...
"""

    return result


def _component_names(obj:Any)->List[str]:
    """The public component names of a pygf container, in order.

    Reading ``_fields_`` names is wrong for a half type: its fields are private
    ``_hx`` style uint16 slots, and handing numpy those hands it bit patterns --
    np.array(half3(1,2,3)) came back as [15360, 16384, 16896].
    """
    if hasattr(obj, "_fields_"):
        names = [name for name, *_ in obj._fields_]
        # A half type's fields are private `_hx` slots; the component a caller
        # means is the public name.
        if names and names[0].startswith("_h"):
            return [name[2:] for name in names]

        return names

    return []

def _raw_components(obj:Any)->List[Any]:
    """Component values of a ctypes Structure or Array, read through the public
    component accessors so a half slot decodes to a float."""
    if hasattr(obj, "_fields_"):
        return [getattr(obj, name) for name in _component_names(obj)]

    return [ctypes.Array.__getitem__(obj, i) for i in range(ctypes.Array.__len__(obj))]

def _is_container(obj:Any)->bool:
    """Whether obj is any pygf container, so numpy.array can be given components.

    A genMat counts even though it has no ``_fields_``: it is a ctypes Array of
    rows, and ctypes' flat storage makes np.array fall back to a (9,) array for a
    3x3. Reading it row by row keeps the nesting that gives the (3, 3) shape a
    caller expects from a matrix.
    """
    if hasattr(obj, "_fields_"):
        return bool(_component_names(obj))

    return hasattr(obj, "rows") and hasattr(obj, "cols")

def _flatten(obj:Any)->Any:
    """A plain nested list of floats for a pygf container.

        numpy has no idea what a genVec or a genMat is: it reads a genVec as a
    0-dimensional structured array, and a genMat as its flat ctypes storage.

    A genMat's __getitem__ yields rows, and its __len__ is the flat element
    count, so an earlier version indexed past the end of it and raised IndexError
    -- np.array(matrix3d()) has to give a (3, 3) array, not blow up.

    Note that (3, 3) is a *different* answer from the one unpatched numpy gives for
    a matrix, which is the flat (9,). Nested rows are what a caller means by a
    matrix, but anything that wanted the flat layout has to ask for it afterwards
    with `.reshape(-1)`.
    """
    if hasattr(obj, "_fields_"):
        return [float(v) for v in _raw_components(obj)]

    if isinstance(obj, (list, tuple)):
        return [_flatten(ele) for ele in obj]

    # A genMat: rows are genVecs, so recursing through the row types keeps the
    # nesting that gives numpy the right shape.
    return [_flatten(row) for row in obj]

def patch_nparray():
    """Teach numpy.array to read a pygf container as its components.

    This replaces ``numpy.array`` process-wide, which is a real hazard for any
    third party in the same interpreter. It is kept -- rather than dropped in
    favour of an opt-in ``to_numpy()`` -- because callers across PyUSD, and any
    renderer built on this library, already pass these types to np.array
    directly. The replacement only intercepts arguments it recognises (a pygf
    container, or a sequence containing one) and delegates everything else
    straight to the real numpy.array.

    Two things this must get right, both of which it got wrong before:

      * A genMat's ``__len__`` is ctypes' flat element count while ``obj[i]`` is a
        row. Rewriting each row into the structured array that a bare genVec
        produces made numpy index past the end, so np.array(matrix3d()) raised
        IndexError. Rows become plain nested lists instead, which yields (3, 3).
      * A half type stores uint16 bit patterns in private ``_hx`` fields, so the
        component names have to be mapped back to the public ones before reading,
        or numpy is handed the bit patterns.
    """
    if np.array.__module__ != "numpy":
        return

    np_array = np.array

    def array(*args, **kwargs):
        obj = args[0] if args else None

        if _is_container(obj):
            return np_array(_flatten(obj), *args[1:], **kwargs)

        if isinstance(obj, (list, tuple)) and any(_is_container(ele) for ele in obj):
            return np_array([_flatten(ele) for ele in obj], *args[1:], **kwargs)

        return np_array(*args, **kwargs)

    # numpy types array as an attribute but the stub describes it as a set of
    # overloads, so a checker cannot see that swapping in a replacement function
    # is exactly what this patch does.
    np.array = array  # ty: ignore
