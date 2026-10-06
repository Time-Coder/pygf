"""Fold the generated ``.pyi`` swizzle declarations into their ``.py`` modules.

Nine modules in pygf shipped both a ``.py`` and a ``.pyi``::

    double2/3/4  float2/3/4  int2/3/4

That is the one configuration ``AGENTS.md`` rules out: when a module ships both,
a type checker reads the two declarations of the same class as two distinct
nominal types, so passing an instance from the implementation into a parameter
annotated with the stub is an ``invalid-argument-type`` error even when the code
is correct. It cost five findings in ``prim.py`` for as long as ``prim.pyi``
existed, and ``ty`` has no setting that merges the two views.

The stubs are huge -- 300 to 3600 lines of swizzle properties per module, since
``genVec`` generates every combination of ``xyzw`` / ``rgba`` / ``stpq`` -- so they
cannot simply be deleted, or the swizzles become invisible to a checker and a
stub-free module falls through ``genVec.__getattr__``, which is ``Any``.

So the declarations move into an ``if TYPE_CHECKING:`` block inside the class
body. At runtime the block does not execute, so the class keeps exactly the
attributes it has today and ``genVec.__getattr__`` stays the single runtime path
for swizzles; a checker and an IDE still see all of them. This is the same
arrangement ``prim.py`` already uses for its 41 generated API accessors.

Each swizzle is emitted as a ``@property``, not a bare annotation, because the
getter and the setter have different types and only the property form says so.
A bare ``xyz: float3`` would type the read and silently accept any assignment.

Two regions are written, and the split is what makes the package importable:

  * a **module-level** ``if TYPE_CHECKING:`` block with the imports the
    annotations name -- the sibling widths and the ``genVecN`` base types;
  * a **class-level** ``if TYPE_CHECKING:`` block with the swizzle properties.

Both are guarded, so neither executes at runtime. That matters: ``double2``'s
block annotates 4-component swizzles as ``double4``, and ``double4``'s as
``double2``, so hoisting those imports to real module level would be a circular
import. Keeping them inside ``if TYPE_CHECKING:`` leaves the cycle unreachable.

They must still be at module level rather than inside the class body, because a
class-body import is a class attribute -- which is not what an annotation in the
same body resolves against, and it made ``ruff`` report ``F821`` on every one of
the 6000-odd swizzle return types.

Run with::

    python tools/sync_swizzle_blocks.py

Both regions are rewritten from scratch every run, so re-running is safe and
produces byte-identical output.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, List, Tuple

# The package directory, i.e. the sibling `pygf/` next to this `tools/` -- not the
# repository root. While the package and this script shared one directory the two
# were indistinguishable here, and the flattening is exactly what made the path
# wrong rather than obviously wrong.
GF_DIR = Path(__file__).resolve().parent.parent / "pygf"

# Imported by module path rather than as `from pygf.helper import ...`, which
# would execute pygf/__init__.py and therefore every module this tool is about to
# rewrite. A generator that cannot run against a half-rewritten package cannot
# finish the job.
def _load_helper() -> Any:
    """Import `helper.py` by path, without executing pygf/__init__.py.

    `from pygf.helper import ...` would import the package, which imports every
    module this tool is about to rewrite -- so a file caught mid-edit would make
    the tool unrunnable, and a generator that cannot run cannot finish the job.

    The loader is built from the file rather than from a name, and both results
    are asserted rather than ignored: a None here means the path does not resolve,
    which is a real error worth hearing about rather than a type-checking gap.
    """
    spec = importlib.util.spec_from_file_location("_pygf_helper", GF_DIR / "helper.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {GF_DIR / 'helper.py'}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


generate_swizzle_defines = _load_helper().generate_swizzle_defines

IMPORTS_BEGIN = "# --- BEGIN generated swizzle imports ---"
IMPORTS_END = "# --- END generated swizzle imports ---"
SWIZZLES_BEGIN = "    # --- BEGIN generated swizzles ---"
SWIZZLES_END = "    # --- END generated swizzles ---"

# (module name, component dtype name, swizzle char sets). The char sets are the
# namespaces genVec builds swizzles from: a 2-vector exposes xy/rg/st, a 3-vector
# xyz/rgb/stp, a 4-vector xyzw/rgba/stpq. Anything longer would swizzle across
# components that do not exist.
VEC_INFOS: List[Tuple[str, str, List[str]]] = [
    ("bool2", "bool", ["xy", "rg", "st"]),
    ("bool3", "bool", ["xyz", "rgb", "stp"]),
    ("bool4", "bool", ["xyzw", "rgba", "stpq"]),
    ("int2", "int", ["xy", "rg", "st"]),
    ("int3", "int", ["xyz", "rgb", "stp"]),
    ("int4", "int", ["xyzw", "rgba", "stpq"]),
    ("uint2", "int", ["xy", "rg", "st"]),
    ("uint3", "int", ["xyz", "rgb", "stp"]),
    ("uint4", "int", ["xyzw", "rgba", "stpq"]),
    ("half2", "float", ["xy", "rg", "st"]),
    ("half3", "float", ["xyz", "rgb", "stp"]),
    ("half4", "float", ["xyzw", "rgba", "stpq"]),
    ("float2", "float", ["xy", "rg", "st"]),
    ("float3", "float", ["xyz", "rgb", "stp"]),
    ("float4", "float", ["xyzw", "rgba", "stpq"]),
    ("double2", "float", ["xy", "rg", "st"]),
    ("double3", "float", ["xyz", "rgb", "stp"]),
    ("double4", "float", ["xyzw", "rgba", "stpq"]),
]

# A swizzle of n components is annotated ``{basename}{n}``, so a module's block
# names every width from 2 up to 4 -- its siblings *and* itself. The self-reference
# is required: a 4-vector's 4-component swizzles are annotated ``matrix4d``-style,
# i.e. its own class, and omitting it left `ruff` reporting F821 on every one of
# them. It is safe because the import sits inside `if TYPE_CHECKING:`, so the
# self-reference never executes.
# The widths a module's block can mention are decided by its char sets, not by its
# own width: a 2-vector swizzles xy/rg/st, so its longest swizzle has three
# components and it never names the 4-vector -- while a 4-vector names all three
# smaller widths. Importing a width the block does not use is what made `ruff`
# report F401 and then remove the line, which broke the next run. So the set is
# derived from the emitted text in `_annotation_names` rather than tabulated.
def _annotation_names(basename: str, swizzles: str) -> List[str]:
    """Every pygf class name the emitted swizzle text refers to."""
    names = set(re.findall(rf"\b{re.escape(basename)}[234]\b", swizzles))
    names.update(re.findall(r"\bgenVec[234]\b", swizzles))

    return sorted(names)


def quote_self_reference(swizzles: str, own_class: str) -> str:
    """Turn annotations naming the module's own class into forward references.

    A class body is not in the scope of its own name, so `-> uint4` inside
    uint4.py is unresolvable -- `ty` reported it on every 4-component swizzle,
    5186 diagnostics in total. Importing the class from its own module does not
    help either: that resolves to a module which has not finished executing, so
    `ty` calls it "no member uint4".

    A quoted annotation is the way out: it is never evaluated (the block is under
    `if TYPE_CHECKING:` anyway) and a type checker resolves it against the
    enclosing scope once the class exists.
    """
    return re.sub(rf"(?<![\"\w]){re.escape(own_class)}\b", f'"{own_class}"', swizzles)



SWIZZLE_COMMENT = [
    "# Every swizzle genVec can build, as declarations only. This block never",
    "# executes, so the runtime path stays genVec.__getattr__.",
    "# Regenerate with tools/sync_swizzle_blocks.py.",
]


def _region(lines: List[str], begin: str, end: str) -> Tuple[int, int]:
    """Index of a marker pair, or (-1, -1) when the file does not carry it.

    Matching is on stripped text: the class-level region is indented by 4 and the
    module-level one by none, and the same constant has to serve both.
    """
    first = last = -1

    for i, line in enumerate(lines):
        if line.strip() == begin.strip():
            first = i
        elif line.strip() == end.strip():
            last = i

    if first < 0 and last < 0:
        return -1, -1

    if first < 0 or last < first:
        raise RuntimeError(f"found only {'BEGIN' if first >= 0 else 'END'} marker for {begin}; repair it by hand")

    return first, last


def strip_region(lines: List[str], begin: str, end: str, guard: str | None) -> List[str]:
    """Remove one generated region together with the guard that wraps it.

    The guard sits outside the fenced markers, so removing only the markers would
    leave an ``if TYPE_CHECKING:`` with no body under it -- an IndentationError.
    Walking up to the nearest guard line is what makes a re-run safe.

    Markers are matched on stripped text, so a region is findable at whatever
    indentation it happens to sit at.
    """
    first, last = _region(lines, begin, end)
    if first < 0:
        return lines

    start = first
    if guard is not None:
        while start > 0 and lines[start - 1].strip() != guard:
            start -= 1

        if start > 0:
            start -= 1

    after = last + 1

    # Absorb the blank lines that separated the region from what follows. Each
    # rebuild re-inserts its own separator, so leaving these behind would add one
    # more per run.
    while after < len(lines) and not lines[after].strip():
        after += 1

    return lines[:start] + lines[after:]


def imports_block(basename: str, swizzles: str, own_base: str, own_class: str) -> List[str]:
    """The module-level guarded imports the swizzle annotations name.

    Two names are excluded because the module already binds them:

      * the base class, which is a real runtime import;
      * the module's own class, which is being defined on this line.

    The second exclusion is not just tidiness. `from .double3 import double3`
    inside double3.py resolves to a module that has not finished executing, so a
    type checker reports "module has no member double3" -- 18 errors, one per
    vector module. The self-reference is unnecessary because the name is already
    bound in the enclosing class body scope.
    """
    names = [name for name in _annotation_names(basename, swizzles) if name not in (own_base, own_class)]

    lines = ["if TYPE_CHECKING:", ""]
    lines.extend(f"    from .{name} import {name}" for name in names)
    lines.append("")
    lines.append(IMPORTS_BEGIN)
    lines.append(IMPORTS_END)

    return lines


def swizzles_block(swizzles: str) -> List[str]:
    """The class-level guarded swizzle properties, at class-body indentation."""

    lines = ["    if TYPE_CHECKING:", ""]
    lines.extend(f"        {comment}" for comment in SWIZZLE_COMMENT)
    lines.append(f"    {SWIZZLES_BEGIN}")
    # The generator emits the properties at class-body indentation, which is one
    # level too shallow once they sit inside the guard. Adding a single level
    # here -- rather than hand-editing the file -- is what keeps the emitted text
    # and the file in agreement.
    lines.extend(f"    {line}" if line.strip() else "" for line in swizzles.rstrip("\n").split("\n"))
    lines.append(f"    {SWIZZLES_END}")

    return lines


def ensure_typing_import(source: str) -> str:
    """Add TYPE_CHECKING and Union to the typing import."""
    match = re.search(r"^from typing import (.+)$", source, re.MULTILINE)
    if match:
        present = match.group(1).split(", ")
        missing = [name for name in ("TYPE_CHECKING", "Union") if name not in present]
        if not missing:
            return source

        names = sorted({*present, *missing})
        return source[:match.start()] + f"from typing import {', '.join(names)}" + source[match.end():]

    anchor = re.search(r"^from \.", source, re.MULTILINE)
    if anchor:
        return source[:anchor.start()] + "from typing import TYPE_CHECKING, Union\n" + source[anchor.start():]

    raise RuntimeError("could not find a place to import TYPE_CHECKING")


def rebuild(source: str, basename: str, size: int, dtype_name: str, char_sets: List[str]) -> str:
    """Return the module with both regions freshly written.

    The class-level block goes last, because the class body runs to end-of-file.
    The module-level block goes just after the imports.
    """
    lines = source.splitlines()

    lines = strip_region(lines, IMPORTS_BEGIN, IMPORTS_END, "if TYPE_CHECKING:")
    lines = strip_region(lines, SWIZZLES_BEGIN, SWIZZLES_END, "if TYPE_CHECKING:")

    while lines and not lines[-1].strip():
        lines.pop()

    # Generated once and used for both regions, so the imports can be derived from
    # the annotations that actually name them rather than a tabulated guess.
    own_class = f"{basename}{size}"
    swizzles = quote_self_reference(
        generate_swizzle_defines(own_class, dtype_name, char_sets),
        own_class,
    )

    # The base class this module already imports at runtime is read back out of
    # the file rather than recomputed, so the two cannot disagree.
    own_base = next(
        (line.split()[1] for line in lines
         if line.startswith("from .") and line.split()[1] == f"genVec{size}"
         and line.split()[3] == f"genVec{size}"),
        f"genVec{size}",
    )

    # Module-level region, after the import block.
    imports = imports_block(basename, swizzles, own_base, own_class)
    anchor = max(i for i, line in enumerate(lines) if line.startswith(("import ", "from ")))
    lines[anchor + 1:anchor + 1] = ["", *imports]

    # Class-level region, at the end of the class body.
    lines.extend(["", *swizzles_block(swizzles)])

    return ensure_typing_import("\n".join(lines) + "\n")


def main() -> int:
    changed: List[str] = []
    stubs_removed: List[str] = []

    for type_name, dtype_name, char_sets in VEC_INFOS:
        size = int(type_name[-1])
        basename = type_name[:-1]
        py_path = GF_DIR / f"{type_name}.py"
        pyi_path = GF_DIR / f"{type_name}.pyi"

        if not py_path.exists():
            print(f"skip {type_name}: no .py")
            continue

        source = py_path.read_text(encoding="utf-8")
        updated = rebuild(source, basename, size, dtype_name, char_sets)

        if updated != source:
            py_path.write_text(updated, encoding="utf-8")
            changed.append(type_name)

        # The stub has been folded in, so it must not ship beside it.
        if pyi_path.exists():
            pyi_path.unlink()
            stubs_removed.append(pyi_path.name)

    if stubs_removed:
        print(f"removed {len(stubs_removed)} stub(s): {', '.join(stubs_removed)}")

    print(f"updated {len(changed)} module(s): {', '.join(changed) if changed else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
