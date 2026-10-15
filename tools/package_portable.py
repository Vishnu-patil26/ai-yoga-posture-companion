"""Build the small portable package for another laptop.

The repository is large (about 215 MB) because it carries the training photographs
and landmark tables used to *fit* the poses.  Running the app needs none of that, so
this makes a zip of only what the app and its offline tests need - code, assets,
fitted references, demo data, docs, tests - plus the two pose-model files, and
START_HERE.bat at the top.  About 25 MB; the libraries are installed on the
other laptop by START_HERE.bat.

    python tools/package_portable.py                 -> dist/AI-Yoga-Companion-portable.zip

Left out on purpose (needed only to refit or re-evaluate the poses; clone the
repository for those): data/datasets, data/landmarks.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "dist")
NAME = "AI-Yoga-Companion"
SKIP_PREFIXES = ("data/datasets/", "data/landmarks/", "dist/")

README = """AI Yoga Posture & Wellness Companion
====================================

1. Unzip this folder anywhere (Desktop is fine).
2. Double-click START_HERE.bat.
   First time: about 5-10 minutes and an internet connection (it installs Python 3.11
   if the laptop has none, then the libraries). After that it opens in seconds.
3. In the app: load or create a profile, or press "Skip - practise as a guest",
   then "Camera: choose a pose".

Needs Windows 10/11 and a webcam. Stand about 2 m from the camera so your whole body
is in the picture. If the camera is blocked: Settings > Privacy & security > Camera >
"Let desktop apps access your camera".

No camera? The Pose library (demo videos, hierarchy) works without one, and
START_HERE.bat setup-only checks the install without opening the app.

Source, datasets and full history: https://github.com/Vishnu-patil26/ai-yoga-posture-companion
"""


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=HERE, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and not f.startswith(SKIP_PREFIXES)]


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{NAME}-portable.zip")
    files = tracked_files()
    models = [os.path.join("models", f) for f in sorted(os.listdir(os.path.join(HERE, "models")))
              if f.endswith(".task")]
    missing = [m for m in ("models/pose_landmarker_full.task",) if not os.path.isfile(os.path.join(HERE, m))]
    if missing:
        print("missing", missing, "- run python bootstrap.py first")
        return 1
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files + [m.replace(os.sep, "/") for m in models]:
            full = os.path.join(HERE, f)
            if os.path.isfile(full):
                z.write(full, f"{NAME}/{f}")
        z.writestr(f"{NAME}/READ_ME_FIRST.txt", README.replace("\n", "\r\n"))
    size = os.path.getsize(path) / 1e6
    print(f"wrote {os.path.relpath(path, HERE)}  ({size:.1f} MB, {len(files) + len(models)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
