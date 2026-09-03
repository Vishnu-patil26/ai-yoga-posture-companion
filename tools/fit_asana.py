"""Fit a reference for EVERY pose class in a dataset, not just Vrikshasana.

The hand-built Vrikshasana definition uses features that only mean something
for a one-legged standing balance (which leg is standing, how far up it the
other foot sits).  Those cannot describe Warrior or Chair, so this fits each
class against a **side-agnostic** vocabulary instead - bent/straight knee,
bent/straight elbow, high/low arm, spine tilt, stance width - which is
mirror-invariant and therefore valid for any asana and either side of it.

Method is the same as tools/fit_reference.py: target = median of correctly
performed examples, tolerance = k x the MAD-based robust sigma, bounded.  The
output is data/asana_fits.json, which yoga/asanas.py loads at import so the
fitted poses appear in the library alongside the hand-tuned one.

    python tools/fit_asana.py data/datasets/yoga_poses/train
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoga.evaluator import compute_features                      # noqa: E402
from yoga.landmarks import PoseTracker                           # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "asana_fits.json")

#: (feature, label, weight, unit, lower bound on tolerance, upper bound)
GENERIC_FEATURES = [
    ("knee_bent",      "Bent knee",      1.2, "deg",   8.0, 30.0),
    ("knee_straight",  "Straight knee",  1.0, "deg",   6.0, 25.0),
    ("elbow_bent",     "Bent elbow",     0.6, "deg",  12.0, 45.0),
    ("elbow_straight", "Straight elbow", 0.6, "deg",  10.0, 40.0),
    ("arm_raise_high", "Higher arm",     0.9, "deg",  12.0, 40.0),
    ("arm_raise_low",  "Lower arm",      0.9, "deg",  12.0, 40.0),
    ("spine_tilt",     "Spine angle",    1.3, "deg",   8.0, 30.0),
    ("stance_width",   "Stance width",   1.1, "ratio", 0.18, 0.90),
]

TOL_K = 1.5

#: Human names for the classes the public dataset ships with.
NAMES = {
    "tree":    ("vrikshasana_fit", "Vrikshasana", "Tree Pose", 2),
    "warrior": ("virabhadrasana", "Virabhadrasana", "Warrior Pose", 2),
    "chair":   ("utkatasana", "Utkatasana", "Chair Pose", 2),
    "dog":     ("adho_mukha", "Adho Mukha Svanasana", "Downward Dog", 1),
    "cobra":   ("bhujangasana", "Bhujangasana", "Cobra Pose", 1),
}


def robust_sigma(values: list[float]) -> float:
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    return 1.4826 * mad


def fit_class(tracker, folder: str) -> tuple[dict, int, int]:
    files: list[str] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(os.path.join(folder, pattern)))
    files.sort()

    samples: dict[str, list[float]] = {f[0]: [] for f in GENERIC_FEATURES}
    used = skipped = 0
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        pose = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if pose is None:
            skipped += 1
            continue
        feats = compute_features(pose)
        # Only the torso is required here.  Demanding the full Vrikshasana
        # visibility gate would throw away most of the floor poses, whose legs
        # are genuinely harder to see - and their torso geometry is exactly
        # what distinguishes them.
        if feats["group_visibility"]["torso"] < 0.6:
            skipped += 1
            continue
        used += 1
        for key, *_ in GENERIC_FEATURES:
            v = feats.get(key)
            if isinstance(v, float) and not math.isnan(v):
                samples[key].append(v)

    checks = {}
    for key, label, weight, unit, lo, hi in GENERIC_FEATURES:
        vals = samples[key]
        if len(vals) < 25:
            continue
        vals.sort()
        med = statistics.median(vals)
        tol = min(hi, max(lo, TOL_K * robust_sigma(vals)))
        checks[key] = {
            "label": label, "target": round(med, 3), "tol": round(tol, 3),
            "zero_at": round(tol * 3.0, 3), "weight": weight, "unit": unit,
            "robust_sd": round(robust_sigma(vals), 3),
            "p10": round(vals[len(vals) // 10], 3),
            "p90": round(vals[9 * len(vals) // 10], 3),
            "n": len(vals),
        }
    return checks, used, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit every pose class in a dataset")
    ap.add_argument("root", help="folder containing one sub-folder per class")
    ap.add_argument("--model", default="full", choices=["full", "lite", "heavy"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    classes = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d)))
    tracker = PoseTracker(model=args.model, running_mode="image", smooth=False)
    fits = {}
    print("")
    print(f"{'class':10s}{'used':>7s}{'skipped':>9s}{'checks':>8s}   asana")
    print("-" * 62)
    for cls in classes:
        checks, used, skipped = fit_class(tracker, os.path.join(args.root, cls))
        if not checks:
            print(f"{cls:10s}{used:7d}{skipped:9d}{0:8d}   (too few usable images)")
            continue
        key, sanskrit, name, level = NAMES.get(
            cls, (cls, cls.title(), cls.title(), 2))
        fits[key] = {
            "key": key, "sanskrit": sanskrit, "name": name, "level": level,
            "source_class": cls, "n_images": used, "checks": checks,
        }
        print(f"{cls:10s}{used:7d}{skipped:9d}{len(checks):8d}   {sanskrit}")
    tracker.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(fits, fh, indent=2)
    print("")
    print(f"{len(fits)} asana(s) fitted -> {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
