#!/usr/bin/env python3
"""Build the sdist and wheel, then upload both to PyPI.

Run `build_publish.py build` first (or just run this) so `dist/` is current.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        subprocess.check_call([sys.executable, "-m", "build", str(ROOT)])

    if not DIST.is_dir() or not any(DIST.glob("pygf-*")):
        print(f"nothing to upload in {DIST}; run: python {Path(__file__).name} build")
        return 1

    artifacts = [str(path) for pattern in ("pygf-*.tar.gz", "pygf-*.whl") for path in DIST.glob(pattern)]

    # `-u` replaces an existing release, which is what a fix to a published
    # version needs; PyPI rejects a second upload of the same filename otherwise.
    subprocess.check_call([sys.executable, "-m", "twine", "upload", *artifacts, "--verbose"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())