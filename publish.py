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
    subprocess.check_call([sys.executable, "-m", "twine", "upload", *artifacts, "--verbose"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())