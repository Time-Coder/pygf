#!/usr/bin/env python3
"""Build the sdist and wheel, then upload both to PyPI.

Run `python publish.py build` to rebuild first, or just `python publish.py` if
dist/ is already current.

Only the artifacts of one version are uploaded. dist/ accumulates: a wheel from a
previous version sitting there would be sent too, and PyPI refuses a duplicate
filename -- so the run would fail on a file nobody meant to republish. Raise the
version in pyproject.toml first; re-uploading a version that already exists needs
`-u`, which this does not pass.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"


def _version() -> str:
    """The version pyproject.toml declares, which is the one dist/ should hold."""
    import re

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.M)
    if match is None:
        raise SystemExit("no version in pyproject.toml")
    return match.group(1)


def main() -> int:
    # The artifacts are built from the working tree, so a tree that differs from the commit
    # means the upload would carry code the history does not have -- 0.2.1 went out that way.
    # `build_wheel.py` refuses to build a dirty tree for the same reason; this refuses to
    # *upload* one, which also catches an edit made between the build and the upload.
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    modified = [line for line in status.splitlines() if not line.startswith("??")]
    if modified:
        print("refusing to publish artifacts built from a working tree that differs from"
              " the commit:")
        for line in modified:
            print(f"  {line}")
        return 1

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        subprocess.check_call([sys.executable, "-m", "build", str(ROOT)])

    if not DIST.is_dir() or not any(DIST.glob("pygf-*")):
        print(f"nothing to upload in {DIST}; run: python {Path(__file__).name} build")
        return 1

    # Narrowed to the version being published, rather than everything in dist/.
    version = _version()
    artifacts = [
        str(found)
        for pattern in (f"pygf-{version}.tar.gz", f"pygf-{version}-*.whl")
        for found in DIST.glob(pattern)
    ]
    stale = sorted(
        p.name for p in DIST.glob("pygf-*")
        if str(p) not in artifacts
    )
    if stale:
        print(f"not uploading: {', '.join(stale)} (not version {version})")

    # `-u` replaces an existing release, which is what a fix to a published
    # version needs; PyPI rejects a second upload of the same filename otherwise.
    #
    # twine draws its progress bar with `rich`, which writes to the Windows console with
    # the *console's* code page: on a GBK console the bullet in its progress line raised
    # UnicodeEncodeError and the upload died at 0%, with nothing registered -- the release
    # went out only after setting PYTHONIOENCODING by hand. The child gets UTF-8 stdio of
    # its own, so publishing never depends on the caller's code page.
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.check_call([sys.executable, "-m", "twine", "upload", *artifacts, "--verbose"],
                          env=environment)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())