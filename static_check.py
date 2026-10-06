#!/usr/bin/env python3
"""Run the repository static checks: compileall, ruff, ty.

    python static_check.py

`ruff check --fix` runs before `ty`, so a formatting pass can resolve findings
rather than leaving them for a human. Both packages are compiled: a failure in
either is as real as one in the other.
"""

import subprocess
import sys

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

root = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0] or "."

for package in ("pygf",):
    subprocess.check_call([sys.executable, "-m", "compileall", "-q", package], cwd=root)

subprocess.check_call([sys.executable, "-m", "ruff", "check", "--fix", "--unsafe-fixes", "."], cwd=root)
subprocess.check_call([sys.executable, "-m", "ty", "check", "."], cwd=root)