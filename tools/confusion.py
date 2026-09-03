"""Held-out confusion matrix across the whole asana library.

`tools/validate_dataset.py` answers a one-pose question: does Vrikshasana score
tree images above everything else?  With a library of poses the real question
is different - score every test image against *every* asana and see whether the
best-scoring one is the right one.  That is the number that says whether the
library as a whole works, and it is the one that shows which pairs get confused.

Fitted on train/, scored here on test/, which fitting never saw.

    python tools/confusion.py data/datasets/yoga_poses/test
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoga import asanas as asana_lib                             # noqa: E402
from yoga.evaluator import compute_features, evaluate            # noqa: E402
from yoga.landmarks import PoseTracker                           # noqa: E402

#: dataset class -> library key it should be recognised as
CLASS_TO_ASANA = {
    "tree": "vrikshasana",
    "warrior": "virabhadrasana",
    "chair": "utkatasana",
    "dog": "adho_mukha",
    "cobra": "bhujangasana",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Library-wide confusion matrix")
    ap.add_argument("root")
    ap.add_argument("--model", default="full", choices=["full", "lite", "heavy"])
    args = ap.parse_args(argv)

    library = {k: a for k, a in asana_lib.LIBRARY.items()}
    keys = sorted(library)
    classes = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d))
                     and d in CLASS_TO_ASANA)

    tracker = PoseTracker(model=args.model, running_mode="image", smooth=False)
    matrix = {c: {k: 0 for k in keys} for c in classes}
    skipped = {c: 0 for c in classes}

    for cls in classes:
        files: list[str] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(glob.glob(os.path.join(args.root, cls, pattern)))
        for path in sorted(files):
            img = cv2.imread(path)
            if img is None:
                continue
            pose = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if pose is None:
                skipped[cls] += 1
                continue
            feats = compute_features(pose)
            if feats["group_visibility"]["torso"] < 0.6:
                skipped[cls] += 1
                continue
            best, best_score = None, -1.0
            for key, asana in library.items():
                ev = evaluate(asana, feats)
                if ev.score > best_score:
                    best, best_score = key, ev.score
            matrix[cls][best] += 1
    tracker.close()

    short = {k: library[k].sanskrit[:11] for k in keys}
    print("")
    print("rows = true class, columns = asana the library scored highest")
    print("")
    header = f"{'true':10s}" + "".join(f"{short[k]:>13s}" for k in keys) + f"{'n':>7s}{'acc':>8s}"
    print(header)
    print("-" * len(header))
    total_right = total_n = 0
    for cls in classes:
        row = matrix[cls]
        n = sum(row.values())
        want = CLASS_TO_ASANA[cls]
        right = row.get(want, 0)
        total_right += right
        total_n += n
        cells = "".join(f"{row[k]:>13d}" for k in keys)
        acc = right / n if n else 0.0
        print(f"{cls:10s}{cells}{n:>7d}{acc:>7.0%}")
    print("-" * len(header))
    print(f"{'overall':10s}{'':{13 * len(keys)}s}{total_n:>7d}"
          f"{(total_right / total_n if total_n else 0):>7.0%}")
    print("")
    print("skipped (torso not visible enough to score): "
          + ", ".join(f"{c} {skipped[c]}" for c in classes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
