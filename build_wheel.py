import os
import shutil
import subprocess
import sys

self_folder = os.path.dirname(os.path.abspath(__file__))
build_folder = os.path.join(self_folder, "build")

# What is built is the *working tree*, not the commit -- so a modified tracked file is
# shipped while the history says nothing about it, and the release stops being
# reproducible from the repository. 0.2.1 went out that way: its wheel carried an
# uncommitted complex implementation. A dirty tree is refused unless `--allow-dirty` says
# the caller knows (untracked files are only listed: they are not shipped unless the
# packaging finds them, which is a different decision).
status = subprocess.run(["git", "status", "--porcelain"], cwd=self_folder,
                        capture_output=True, text=True, check=True).stdout
modified = [line for line in status.splitlines() if not line.startswith("??")]
untracked = [line for line in status.splitlines() if line.startswith("??")]
for line in untracked:
    print(f"untracked, not part of the build unless packaging picks it up: {line[3:]}")
if modified and "--allow-dirty" not in sys.argv:
    print("refusing to build a working tree that differs from the commit:")
    for line in modified:
        print(f"  {line}")
    raise SystemExit("commit or stash those changes first, or pass --allow-dirty")

if os.path.isdir(build_folder):
    shutil.rmtree(build_folder)

subprocess.check_call([
    sys.executable,
    "-m",
    "build",
    "--no-isolation",
    "--wheel",
    "--sdist",
    "--outdir",
    os.path.join(self_folder, "dist"),
], cwd=self_folder)
