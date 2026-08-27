"""Download the public yoga-pose dataset used to fit and validate the reference.

TensorFlow / Laurence Moroney 5-class yoga poses (chair, cobra, dog, tree,
warrior), Apache-2.0, ~100 MB.  See docs/REFERENCES.md for why this dataset and
what was derived from it.

    python tools/get_dataset.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

URL = ("https://storage.googleapis.com/download.tensorflow.org/data/"
       "pose_classification/yoga_poses.zip")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "datasets", "yoga_poses")
MARKER = os.path.join(DEST, "train", "tree")
MIN_BYTES = 50_000_000


def _progress(done: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, done * block * 100 // total)
    sys.stdout.write(f"\r  downloading... {pct:3d}%")
    sys.stdout.flush()


def main() -> int:
    if os.path.isdir(MARKER) and os.listdir(MARKER):
        print(f"dataset already present at {os.path.relpath(DEST, ROOT)}")
        return 0

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), "yoga_poses.zip")
    try:
        urllib.request.urlretrieve(URL, tmp, _progress)
        print()
        size = os.path.getsize(tmp)
        if size < MIN_BYTES:
            raise OSError(f"download looks truncated ({size} bytes)")
        # Extract to a scratch directory first so a failure part-way through
        # cannot leave a half-populated dataset that later looks complete.
        staging = DEST + ".part"
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(staging)
        shutil.rmtree(DEST, ignore_errors=True)
        os.replace(staging, DEST)
    except Exception as exc:
        print(f"\ncould not fetch the dataset: {exc}")
        print(f"download it by hand from:\n  {URL}\nand unzip it into {DEST}")
        return 1
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)

    counts = {
        cls: len(os.listdir(os.path.join(DEST, split, cls)))
        for split in ("train", "test")
        if os.path.isdir(os.path.join(DEST, split))
        for cls in sorted(os.listdir(os.path.join(DEST, split)))
    }
    print(f"dataset ready at {os.path.relpath(DEST, ROOT)}")
    print("  " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("\nnow you can run:")
    print("  python tools/fit_reference.py data/datasets/yoga_poses/train/tree")
    print("  python tools/validate_dataset.py data/datasets/yoga_poses/test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
