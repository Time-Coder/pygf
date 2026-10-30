"""A numpy view whose elements are vectors, and the constructors that produce it.

A vector type is a structured dtype (``np.dtype(cls._fields_)``), so an array of vectors is
an ordinary structured array. Its elements are ``np.void`` records, though, and that is not
what a caller wants: ``arr[0].x`` does not exist on a record, and ``arr[0] = float3(...)``
is rejected because numpy cannot read a ctypes structure as a sequence. Both are host
representation details, not something a caller should have to work around.

``VectorArray`` closes that gap, and ``install_vector_array_patches`` makes the numpy
constructors return it, so the array a caller builds behaves the way the types suggest:

    points = np.zeros(4, dtype=float3.dtype)     # a VectorArray
    points[0] = float3(1.0, 2.0, 3.0)            # accepts a vector
    points[0].x                                  # gives a float back

The dtype registry is built from the ``genVec`` hierarchy. Types defined outside this package
(PyGPGPU carries 8- and 16-wide ones) join it through ``register_vector_type``, which is why
the lookup is by dtype rather than by class.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple, Type

import numpy as np

__all__ = [
    "VectorArray",
    "vector_class_for_dtype",
    "register_vector_type",
    "vector_dtype_of_sequence",
    "wrap_vector_array",
    "install_vector_array_patches",
]

# numpy dtype of a vector type -> the class that describes it. Built once, lazily: importing
# this module must not depend on which type modules have been imported already.
_VECTOR_TYPES: Dict[Any, Type[Any]] = {}
_BUILT: bool = False
_PATCHED: bool = False


def register_vector_type(cls: Type[Any]) -> None:
    """Record a vector type so its dtype is recognized by the view.

    Idempotent, and it keeps the first class registered for a dtype: an alias such as
    ``color3f`` shares ``float3``'s layout, and rebuilding an element as the base type is the
    honest answer when the dtype cannot tell the two apart.

    The package's own hierarchy is registered on first lookup regardless of when this is
    called, so a sibling package registering its types first cannot hide them.
    """
    _ensure_registry()
    fields = getattr(cls, "_fields_", None)
    if not fields:
        return
    try:
        # Built exactly like the vector dtypes are, so it is *not* an aligned record: that
        # alignment flag is what tells a caller's `ctypes.Structure` apart from a vector
        # whose layout matches it.
        dtype = np.dtype(fields)
    except TypeError:
        return
    _VECTOR_TYPES.setdefault(dtype, cls)


def _subclasses(cls: type) -> Iterator[type]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _subclasses(subclass)


def _build_registry() -> None:
    """Every concrete vector type in this package, keyed by its dtype."""
    from .genVec import genVec

    for cls in _subclasses(genVec):
        fields = getattr(cls, "_fields_", None)
        if not fields:
            continue
        try:
            dtype = np.dtype(fields)
        except TypeError:
            continue
        _VECTOR_TYPES.setdefault(dtype, cls)


def _ensure_registry() -> None:
    global _BUILT
    if _BUILT:
        return
    _BUILT = True
    _build_registry()


def vector_class_for_dtype(dtype: Any) -> Optional[Type[Any]]:
    """The vector class an array dtype denotes, or None.

    An *aligned* record is not one of ours: `np.dtype` over a `ctypes.Structure` produces an
    aligned layout, so a caller's `{float x; float y;}` compares equal to `float2` while
    being a structure they declared. The vector dtypes this package ships are built with
    ``np.dtype(cls._fields_)`` and are not aligned, which is the only thing that separates
    the two -- they compare and hash equal.
    """
    _ensure_registry()
    if not isinstance(dtype, np.dtype) or dtype.fields is None:
        return None
    if dtype.isalignedstruct:
        return None
    return _VECTOR_TYPES.get(dtype)


class VectorArray(np.ndarray):
    """Array view whose vector elements read back as vectors instead of records."""

    def __getitem__(self, item: Any) -> Any:
        value: Any = super().__getitem__(item)
        if isinstance(value, np.void):
            cls: Optional[Type[Any]] = vector_class_for_dtype(value.dtype)
            if cls is not None:
                return cls.from_buffer_copy(value.tobytes())
        return value

    def __setitem__(self, item: Any, value: Any) -> None:
        if isinstance(value, np.ndarray) and not value.shape:
            value = value[()]
        if hasattr(value, "_fields_") and hasattr(type(value), "_fields_"):
            # A ctypes structure is not a sequence numpy can read, but its components are
            # exactly the record numpy expects.
            value = tuple(getattr(value, name) for name, _ in type(value)._fields_)
        super().__setitem__(item, value)


def wrap_vector_array(value: Any) -> Any:
    """``value`` as a `VectorArray` when it holds vectors; otherwise unchanged.

    An array that is already a `VectorArray` is returned as-is, so the identity of a
    caller's object survives a round trip through a function that hands it back.
    """
    if not isinstance(value, np.ndarray):
        return value
    if isinstance(value, VectorArray):
        return value
    if vector_class_for_dtype(value.dtype) is None:
        return value
    return value.view(VectorArray)


# ---------------------------------------------------------------------------
# Building one from a sequence of vectors
# ---------------------------------------------------------------------------


def _leaves(value: Any) -> Iterator[Any]:
    """Every non-sequence element of a (possibly nested) list/tuple."""
    if isinstance(value, (list, tuple)):
        for element in value:
            yield from _leaves(element)
        return
    yield value


def _sequence_shape(value: Any) -> Optional[Tuple[int, ...]]:
    """The rectangular shape of a nested list/tuple, or None when it is ragged."""
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    inner: Optional[Tuple[int, ...]] = None
    for element in value:
        shape = _sequence_shape(element)
        if shape is None:
            return None
        if inner is None:
            inner = shape
        elif inner != shape:
            return None
    return (len(value),) + (inner or ())


def vector_dtype_of_sequence(value: Any) -> Optional[Any]:
    """The vector dtype a sequence of vector instances denotes, or None.

    Every element has to be a vector of one type: a mixed list stays numpy's business, so it
    keeps raising numpy's own error instead of being read as records.
    """
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        first: Any = next(_leaves(value), None)
    except TypeError:
        return None
    if first is None:
        return None
    cls: Optional[Type[Any]] = _class_of_instance(first)
    if cls is None:
        return None
    for leaf in _leaves(value):
        if _class_of_instance(leaf) is not cls:
            return None
    return cls.numpy_dtype()


def _class_of_instance(value: Any) -> Optional[Type[Any]]:
    """The vector class a value is an instance of, or None.

    Duck-typed rather than restricted to ``genVec``: a sibling package that defines wider
    vectors (PyGPGPU carries 8- and 16-wide ones) implements the same two attributes, and its
    arrays deserve the same view.
    """
    cls = type(value)
    if getattr(cls, "_fields_", None) and callable(getattr(cls, "numpy_dtype", None)):
        return cls
    return None


def _fill_from_sequence(target: np.ndarray, value: Any, index: Tuple[int, ...]) -> None:
    """Write a nested list of vectors into an already-allocated array."""
    if isinstance(value, (list, tuple)):
        for position, element in enumerate(value):
            _fill_from_sequence(target, element, index + (position,))
        return
    target[index] = value


def _from_vector_sequence(value: Any, dtype: Any) -> Optional[Any]:
    """Build an array of vectors from a nested list, or None when the shape is ragged.

    numpy (and the ``patch_nparray`` rewrite) reads each vector as a sequence of its
    components, so ``np.array([float3(...), float3(...)])`` would become a wider array of
    numbers. An array of vectors has to be filled record by record instead.
    """
    shape: Optional[Tuple[int, ...]] = _sequence_shape(value)
    if shape is None:
        return None
    target: Any = np.zeros(shape, dtype=dtype)
    _fill_from_sequence(target, value, ())
    return wrap_vector_array(target)


# ---------------------------------------------------------------------------
# The numpy patches
# ---------------------------------------------------------------------------

_CONSTRUCTORS: Tuple[str, ...] = ("zeros", "empty", "ones", "full", "zeros_like", "empty_like")


def install_vector_array_patches() -> None:
    """Make the numpy constructors return a `VectorArray` for a vector dtype.

    Only the constructors that take a dtype (or copy one from a ``_like`` argument) are
    touched, and only the result is re-viewed: the allocation, the shape and the dtype are
    numpy's own, so nothing about array semantics changes beyond element access.
    ``np.asarray`` is included because a caller who has the dtype in hand should get the same
    view as one who does not.

    ``np.array`` is handled by ``helper.patch_nparray``, which already owns that function
    process-wide; this only adds the constructors beside it. Installing twice is a no-op.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    for name in _CONSTRUCTORS:
        original: Any = getattr(np, name, None)
        if original is None:
            continue
        setattr(np, name, _patched_constructor(name, original))

    original_asarray: Any = np.asarray
    setattr(np, "asarray", _patched_asarray(original_asarray))


def _patched_constructor(name: str, original: Any) -> Any:
    """A constructor that returns its result through ``wrap_vector_array``."""

    def patched(*args: Any, **kwargs: Any) -> Any:
        return wrap_vector_array(original(*args, **kwargs))

    patched.__name__ = name
    patched.__doc__ = original.__doc__
    return patched


def _patched_asarray(original: Any) -> Any:
    """``np.asarray``: infer a vector dtype, then hand back a `VectorArray`."""

    def patched(obj: Any, dtype: Any = None, *args: Any, **kwargs: Any) -> Any:
        inferred: Optional[Any] = vector_dtype_of_sequence(obj)
        if inferred is not None and (dtype is None or np.dtype(dtype) == inferred):
            built: Any = _from_vector_sequence(obj, inferred)
            if built is not None:
                return built
        if dtype is None and inferred is not None:
            kwargs.setdefault("dtype", inferred)
        return wrap_vector_array(original(obj, *args, **kwargs))

    patched.__name__ = "asarray"
    patched.__doc__ = original.__doc__
    return patched


def _sequence_dtype_from_array(obj: Any, dtype: Any) -> Tuple[Any, Optional[Any]]:
    """(dtype to hand numpy, an array built here or None) for ``np.array``.

    Called by ``patch_nparray``, which owns the ``np.array`` name.
    """
    inferred: Optional[Any] = vector_dtype_of_sequence(obj)
    if inferred is None:
        return dtype, None
    if dtype is None or np.dtype(dtype) == inferred:
        built: Any = _from_vector_sequence(obj, inferred)
        if built is not None:
            return dtype, built
    return dtype, None
