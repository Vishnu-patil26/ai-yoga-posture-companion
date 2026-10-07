"""Held-out evaluation of the WHOLE pose library, the number the viva will ask for.

Every held-out image is scored against every asana in the library and the
best-scoring asana is taken as the prediction - so this measures recognition
across all ten poses at once, not one pose against a background.

Held-out sets
  * the five TensorFlow/Moroney classes: their own `test/` split, which the
    fitting never saw (tools/confusion.py does the same for just those five);
  * every class fitted with `tools/fit_asana.py --holdout F`: the files set aside
    at fitting time and recorded in data/asana_fits.json.

Output: a confusion matrix, per-pose recall (with n), the overall figure, and
data/library_eval.json for the report.

    python tools/library_eval.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from yoga import asanas as asana_lib                              # noqa: E402
from yoga.evaluator import compute_features, evaluate             # noqa: E402
from yoga.landmarks import PoseTracker                            # noqa: E402

TF_TEST = os.path.join(HERE, "data", "datasets", "yoga_poses", "test")
#: "warrior" is left out on purpose: that dataset class mixes Warrior I, II and III
#: and the library's Virabhadrasana is strictly Warrior II.
TF_CLASS = {"tree": "vrikshasana", "chair": "utkatasana", "dog": "adho_mukha",
            "cobra": "bhujangasana"}
FITS = os.path.join(HERE, "data", "asana_fits.json")
OUT = os.path.join(HERE, "data", "library_eval.json")


def held_out_sets() -> dict[str, list[str]]:
    """true asana key -> list of image paths the library has never been fitted on."""
    sets: dict[str, list[str]] = {}
    for cls, key in TF_CLASS.items():
        files = []
        for pat in ("*.jpg", "*.jpeg", "*.png"):
            files += glob.glob(os.path.join(TF_TEST, cls, pat))
        if files:
            sets[key] = sorted(files)
    try:
        with open(FITS, encoding="utf-8") as fh:
            fits = json.load(fh)
    except (OSError, ValueError):
        fits = {}
    for key, spec in fits.items():
        if spec.get("holdout"):
            sets.setdefault(key, []).extend(
                os.path.join(os.path.dirname(FITS), f) for f in spec["holdout"])
    return sets


def main() -> int:
    sets = held_out_sets()
    library = dict(asana_lib.LIBRARY)
    # the hand-tuned Vrikshasana is the library's tree; its fitted twin is not registered
    keys = sorted(library)
    tracker = PoseTracker(model="full", running_mode="image", smooth=False)
    matrix = {t: {k: 0 for k in keys} for t in sets}
    skipped = {t: 0 for t in sets}
    for true, files in sets.items():
        for path in files:
            img = cv2.imread(path)
            pose = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img is not None else None
            if pose is None:
                skipped[true] += 1
                continue
            feats = compute_features(pose)
            if feats["group_visibility"]["torso"] < 0.6:
                skipped[true] += 1
                continue
            best = max(library, key=lambda k: evaluate(library[k], feats).score)
            matrix[true][best] += 1
    tracker.close()

    short = {k: library[k].sanskrit[:10] for k in keys}
    head = f"{'true':14s}" + "".join(f"{short[k]:>11s}" for k in keys) + f"{'n':>6s}{'recall':>8s}"
    print("\nrows = true pose, columns = asana the library scored highest\n")
    print(head)
    print("-" * len(head))
    tot_ok = tot_n = 0
    report = {}
    for true in sorted(sets):
        row = matrix[true]
        n = sum(row.values())
        ok = row.get(true, 0)
        tot_ok += ok
        tot_n += n
        print(f"{true:14s}" + "".join(f"{row[k]:>11d}" for k in keys) + f"{n:>6d}{(ok / n if n else 0):>8.0%}")
        report[true] = {"n": n, "correct": ok, "recall": round(ok / n, 3) if n else None,
                        "skipped": skipped[true], "confusions": {k: v for k, v in row.items() if v and k != true}}
    print("-" * len(head))
    overall = tot_ok / tot_n if tot_n else 0.0
    print(f"{'overall':14s}{'':{11 * len(keys)}s}{tot_n:>6d}{overall:>8.0%}")
    print("skipped (torso not visible): " + ", ".join(f"{k} {v}" for k, v in skipped.items()))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"overall_recall": round(overall, 3), "n": tot_n, "poses": report}, fh, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
