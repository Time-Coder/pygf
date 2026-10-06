# pygf

Graphics math types for Python: vectors, matrices and quaternions, built on
`ctypes` with no compiled extension. A shared foundation for scene-description,
shading-language and rendering projects that all need the same numeric types and
none of them should have to depend on each other for them.

```python
from pygf import double3, matrix4d, quatd, funcs

v = double3(1.0, 0.0, 0.0)

m = matrix4d() * 2.0                    # element-wise, `*` not matrix multiply
m = matrix4d() * matrix4d()             # matrix multiply
n = funcs.normalize(double3(3.0, 4.0, 0.0))
w = funcs.dot(v, n)

r = quatd(0.0, 1.0, 0.0, 0.0) * v       # rotate by a half turn about X
```

Note that `*` is matrix multiply for `matrix * matrix`, matching what this layer
has always done, and that there is no `@` operator.

## What is here

| Group | Types |
|---|---|
| Vectors | `float2/3/4`, `double2/3/4`, `int2/3/4`, `uint2/3/4`, `half2/3/4`, `bool2/3/4` |
| Matrices | `matrix2/3/4` in `f`, `d`, `b` |
| Quaternions | `quatf`, `quatd`, `quath`, `quatb` |
| USD aliases | `point3f`, `normal3f`, `vector3f`, `color3f`, `texCoord2f`, `frame4d`, ... |
| Functions | the GLSL-shaped set: `abs`, `mix`, `clamp`, `cross`, `dot`, `normalize`, `determinant`, `inverse`, ... |

## Design notes

**Element types, not operators.** A `genVec` counts its slots with `_slot_count`,
which is `rows` for a matrix and the element count for a vector or quaternion.
`len()` is ctypes' *flat* element count on a matrix -- 16 for a 4x4 -- while
`m[i]` yields the i-th row, so driving the element-wise loops with `len()`
overruns. The element-wise helpers in `funcs.py` also *recurse*, because a matrix
slot is a row and handing that row to `builtins.abs` still fails once the count is
right.

**Matrices are square only.** `genMat.mat_type` refuses a non-square shape rather
than silently truncating it to `shape[0] x shape[0]`.

**`half` is a real IEEE-754 binary16.** Stored as `c_uint16` bit patterns in
private fields, converted on access through `struct`'s native `e` format. The
component names are properties, so `ctypes.Structure.__setattr__` still reaches
them. Overflow saturates to infinity rather than raising, which is what a GPU
buffer would hold.

**Swizzles are runtime-dynamic.** `genVec.__getattr__` builds every combination
of `xyzw` / `rgba` / `stpq` on demand, so the class carries no swizzle
declarations at runtime. A generated `if TYPE_CHECKING:` block declares all of
them for type checkers and IDEs; see `tools/sync_swizzle_blocks.py`.

**numpy interop.** Importing `pygf` replaces `numpy.array` so that it reads a
container as its components: `np.array(matrix3d())` gives a `(3, 3)` array
rather than ctypes' flat storage. Arguments it does not recognise are passed
straight through.

## Layout

```
pygf/          the package
tests/         test_pygf.py -- stdlib plus pygf, nothing else
tools/         sync_swizzle_blocks.py -- the only writer of the swizzle blocks
static_check.py
publish.py
```

## Development

```
python -m pip install -e ".[dev]"
python static_check.py
python tests/test_pygf.py
```

`static_check.py` runs `compileall`, `ruff check --fix`, and `ty check`. Every
check in the test file corresponds to a defect that type checking or review
surfaced; the module docstring in the test lists them.

After changing `genVec`'s swizzle generation, regenerate the declaration blocks:

```
python tools/sync_swizzle_blocks.py
```

It is idempotent, and it is the only thing that writes those blocks.

## Publishing

```
python -m pip install -e ".[publish]"
python publish.py build
python publish.py
```

## License

MIT