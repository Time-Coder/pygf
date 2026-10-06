# Repository Guidelines

`pygf` is a standalone graphics-math library: vectors, matrices and quaternions on
`ctypes`, with no compiled extension. It exists so that PyUSD, PyMaterialX and
PyRHI can all depend on the same numeric types without depending on each other.

## Layout

- `pygf/` is the package: the type hierarchy (`genType`, `genVec`, `genMat`,
  `genQuat` and their width-specific subclasses), `funcs.py` for the
  GLSL-shaped function set, `half.py` for binary16 conversion, and `helper.py` for
  the pieces the rest of the package shares.
- `tests/test_pygf.py` is the test suite. It imports stdlib and `pygf` and nothing
  else -- no test fixtures, no other project -- so it stays runnable here
  independently of whatever depends on the package.
- `tools/sync_swizzle_blocks.py` regenerates the `if TYPE_CHECKING:` swizzle blocks.
- `static_check.py`, `publish.py`.

## Commands

```powershell
python -m pip install -e ".[dev]"
python static_check.py          # compileall, ruff check --fix, ty check
python tests/test_pygf.py
python tools/sync_swizzle_blocks.py
```

`python ../pygf/static_check.py` and `python ../pygf/tests/test_pygf.py` are how
PyUSD reaches these checks; see `../PyUSD/AGENTS.md` for what it relies on.

## Conventions

Four spaces, double-quoted strings, the 88-column target from `ruff.toml`
(inherited from the sibling PyUSD project until this repo grows its own).
`snake_case` for modules and functions, `PascalCase` for classes, and the USD
schema spellings for the element types.

## Architecture Notes

These are the constraints that are easy to get wrong and that a type checker will
not always catch.

- **A module must not ship both a `.py` and a `.pyi`.** When it does, `ty` treats
  the two declarations of the same class as distinct nominal types, so passing an
  instance from the implementation into a parameter annotated with the stub is an
  `invalid-argument-type` error even though the code is correct. Nine modules did
  (300-3600 lines of swizzle properties each), which is why the declarations live in
  an `if TYPE_CHECKING:` block inside the class instead. At runtime the block does
  not execute, so `genVec.__getattr__` stays the single path for swizzles and a
  checker still sees every one of them.
- **`genType` declares `__len__`/`__getitem__`/`__setitem__`/`__iter__` that raise
  `NotImplementedError`**, because every element-wise helper in `funcs.py` is
  annotated against `genType` while only ever receiving `genVec`/`genMat`/`genQuat`.
  `genMat` therefore forwards `__len__` to `ctypes.Array.__len__` explicitly, since
  `genType` comes first in its MRO and would otherwise shadow it.
- **A matrix slot is a row, and `len()` is not the slot count.** `genMat.__len__`
  is ctypes' flat element count -- 16 for a 4x4 -- while `m[i]` is the i-th row, so
  a loop driven by `len()` overruns. `genType._slot_count` returns `rows` for a
  matrix and the element count otherwise, and the loops use it. That is not
  sufficient on its own: the element-wise helpers also have to *recurse*, because
  handing a row to `builtins.abs` still fails once the count is right.
  `_equal_leaves` and `_has_negative` in `genType.py` exist for the same reason, and
  `funcs.any`/`all` descend because a row is truthy for as long as it has a length --
  walking rows directly made `any(zero_matrix)` answer `True`.
- **Element access on a matrix goes through `genMat.at(row, col)` and
  `genMat.put(row, col, value)`**, never `m[i, j]`, so the code does not depend on
  ctypes' flat storage layout. `__getitem__` has to advertise `genVec` as a return
  type for the same reason: an int index yields a row.
- **The quaternions' w/x/y/z properties are shadowed at instance level** by the
  ctypes field descriptor on the concrete subclass, which sits ahead of `genQuat`
  in the MRO. Their bodies therefore reach the field through
  `ctypes.Structure.__getattribute__`, and the test file calls them via
  `genQuat.w.fget(quat)`, which is the only way to reach them.
- **`genQuat.__init__` assigns through `self[i]`, not
  `ctypes.Structure.__init__`.** The latter writes the raw bit pattern of a half
  field, so the identity quaternion `(1, 0, 0, 0)` would read back as `w == 1.4e-45`.
  The same reasoning moved `genVec.__iter__` and `__contains__` onto `self[i]`
  rather than `_fields_` names.
- **`half` is real IEEE-754 binary16.** ctypes has no `c_half`, so a half stores its
  bit pattern in a `c_uint16` and converts on access through `struct`'s native `e`
  format -- stdlib since 3.6, and the reason the package's arithmetic needs no
  numpy. The components are properties and the fields are private (`_hx`), because
  a ctypes Structure cannot carry both a field and a converting property under one
  name. `ctypes.Structure.__setattr__` does dispatch to a property setter, which is
  what keeps the `genVec.__setattr__` swizzle path working. Overflow saturates to
  infinity rather than raising, which is what a GPU buffer would hold.
- **`genVec3`, `genMat3` and `genQuat` have no `dtype`, and that is by design.** They
  are abstract intermediates, and the `dtype`/`math_form`/`shape` they would have to
  guess belong to the concrete subclass. Only instantiate the concrete types
  (`double3`, `matrix4d`, `quatd`), and reach for `gen_type` when the concrete type
  is not known statically.
- **`gen_type` resolves module names relative to its own package**, via
  `helper.from_import`, which passes `package=__package__`. That is why every
  concrete type module has to sit beside `helper.py` in the same directory, and
  why the package can be renamed or moved wholesale without touching its insides.
  A dtype in `__dtype_name_map` with no matching module reports a `ValueError`
  naming the width that is missing, rather than a bare `ModuleNotFoundError`.
- **`funcs.py` shadows builtins deliberately.** It defines `abs`, `min`, `max`,
  `round`, `pow` and `all`, so the bodies that want the builtin say
  `builtins.abs`. Getting this wrong is how `roundEven` came to be a byte-for-byte
  copy of `round`, and how `funcs.abs` once raised `AttributeError` by going through
  `__builtins__` -- a dict inside an imported module.
- **`patch_nparray` replaces `numpy.array` process-wide.** That is a real hazard
  for any third party in the same interpreter. It is kept because callers pass these
  types to `np.array` directly, and it now only intercepts arguments it recognises.
  Two things it has to get right: a `genMat` yields rows while `__len__` is ctypes'
  flat storage, so rows must become plain nested lists -- that is what turns
  `np.array(matrix3d())` from an `IndexError` into a `(3, 3)` array; and a half
  stores bit patterns in private fields, so the component names must be mapped back
  to the public ones or numpy is handed the bit patterns.
- **The swizzle block is generated, and the generator is the only writer.**
  `tools/sync_swizzle_blocks.py` rewrites the whole fenced region every run, so
  hand-editing inside the markers is lost. Four things about it are load-bearing:
  the sibling imports sit in a **module-level** `if TYPE_CHECKING:` block (at module
  level they are a circular import, since `double2`'s block names `double4` and vice
  versa; inside the class body they become class attributes, which is not what an
  annotation in the same body resolves against); a self-reference is **quoted**
  (`-> "uint4"`), because a class body is not in the scope of its own name; the
  import set is derived from the emitted text rather than tabulated, because a width
  the block does not mention gets reported as unused and removed; and it loads
  `helper.py` **by path**, since `from pygf.helper import ...` would execute the
  package `__init__` and therefore every module the tool is about to rewrite.
- **`[tool.setuptools.packages.find]` resolves this package** because
  `pyproject.toml` sits beside `pygf/` at the root. While the two shared one
  directory, `find` searched for a `pygf/` subdirectory, found nothing, and an
  editable install produced a resolver with an empty mapping -- `import pygf` worked
  from the repository root and failed from anywhere else.

## Testing Guidelines

`python tests/test_pygf.py` is a plain script with a hand-rolled `check()`, not a
pytest module; it exits non-zero on the first summary of failures. Every check
corresponds to a defect that type checking or review surfaced, and the module
docstring lists them -- add to that list when adding a check. Run
`static_check.py` as well; `ty` has found real bugs here that the runtime tests
missed, which is most of the list.

## Guidelines from the shared skill

- Reuse and verification: search the codebase for an existing implementation before
  adding one, and verify a fix against the real behaviour rather than the expectation.
- For every fix, attempt to break it: add a case to `tests/test_pygf.py` that fails
  without the change.
- Match the project's style in the file you edit rather than your own.
- Solve the problem that was asked about. Avoid defensive code beyond what the task
  requires, and do not refactor code the task does not touch.