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
    ("neck_dev",       "Neck alignment", 0.8, "deg",   8.0, 25.0),
    # pose-structure features (see evaluator.compute_features)
    ("hip_angle_min",  "Hip fold",       1.2, "deg",  10.0, 35.0),
    ("hip_angle_max",  "Open hip",       0.8, "deg",  10.0, 35.0),
    ("arm_torso_min",  "Arm to trunk",   0.9, "deg",  12.0, 40.0),
    ("arm_torso_max",  "Other arm to trunk", 0.7, "deg", 12.0, 40.0),
    ("hip_rise",       "Hip height",     1.1, "ratio", 0.12, 0.60),
    ("head_drop",      "Head position",  0.8, "ratio", 0.10, 0.50),
    ("thigh_level",    "Thigh angle",    1.0, "deg",  10.0, 35.0),
    # visible-side versions for side-view poses (see evaluator.compute_features)
    ("hip_angle_vis",   "Hip fold",      1.2, "deg",  12.0, 40.0),
    ("knee_angle_vis",  "Knee bend",     1.2, "deg",  10.0, 35.0),
    ("elbow_angle_vis", "Elbow bend",    0.9, "deg",  14.0, 45.0),
    ("arm_torso_vis",   "Arm to trunk",  0.9, "deg",  14.0, 45.0),
]

TOL_K = 1.5

#: Classes whose correct form comes in more than one accepted shape, as groups of
#: step folders (the "Yoga for all" dataset numbers the build-up steps).  Each
#: group is fitted on its own and becomes a Variant - the same mechanism that
#: lets Vrikshasana accept hands overhead or at the heart - so the practitioner
#: is scored against whichever form they are actually attempting.
VARIANT_GROUPS = {
    "tadasana": {"arms down": ("step_1", "step_5"),
                 "arms up": ("step_2", "step_3", "step_4")},
}

#: A class with fewer usable images than this is flagged `small_sample` in the
#: fit and shown as "limited data" in the app.
ROBUST_N = 25

#: Human names for the classes the public dataset ships with.
NAMES = {
    "tree":    ("vrikshasana_fit", "Vrikshasana", "Tree Pose", 2),
    "warrior": ("virabhadrasana", "Virabhadrasana", "Warrior Pose", 2),
    "chair":   ("utkatasana", "Utkatasana", "Chair Pose", 2),
    "dog":     ("adho_mukha", "Adho Mukha Svanasana", "Downward Dog", 1),
    "cobra":   ("bhujangasana", "Bhujangasana", "Cobra Pose", 1),
    # Poses added from the "Yoga for all" dataset (Zenodo 7818789) and
    # Wikimedia Commons - see tools/get_yoga_for_all.py / get_commons_poses.py.
    "tadasana":     ("tadasana", "Tadasana", "Mountain Pose", 1),
    "marjaryasana": ("marjaryasana", "Marjaryasana-Bitilasana", "Cat-Cow", 1),
    "trikonasana":  ("trikonasana", "Trikonasana", "Triangle Pose", 2),
    "balasana":     ("balasana", "Balasana", "Child's Pose", 1),
    "sukhasana":    ("sukhasana", "Sukhasana", "Easy Pose", 1),
}


def robust_sigma(values: list[float]) -> float:
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    return 1.4826 * mad


def list_images(folder: str) -> list[str]:
    """Every image under `folder`, recursively and in a stable order."""
    files: list[str] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    # "wrong/" folders (labelled faulty poses) are for tools/deviation_study.py,
    # never for fitting what a correct pose looks like.
    return sorted(f for f in files if f"{os.sep}wrong{os.sep}" not in f)


def split_holdout(files: list[str], fraction: float) -> tuple[list[str], list[str]]:
    """Deterministic hold-out: every k-th file is set aside and never fitted on."""
    if fraction <= 0 or len(files) < 8:
        return files, []
    k = max(2, round(1.0 / fraction))
    held = files[k - 1::k]
    return [f for i, f in enumerate(files) if (i + 1) % k], held


def fit_class(tracker, folder: str, files: list[str] | None = None,
              min_samples: int = ROBUST_N,
              weights: dict[str, float] | None = None) -> tuple[dict, int, int]:
    """Fit one class.  `weights` (feature -> weight) restricts the fit to the
    measurements that define the pose and sets their weights; None = all."""
    if files is None:
        files = list_images(folder)
    vocab = [(k, l, (weights[k] if weights else w), u, lo, hi)
             for k, l, w, u, lo, hi in GENERIC_FEATURES if not weights or k in weights]

    samples: dict[str, list[float]] = {f[0]: [] for f in vocab}
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
        for key, *_ in vocab:
            v = feats.get(key)
            if isinstance(v, float) and not math.isnan(v):
                samples[key].append(v)

    checks = {}
    for key, label, weight, unit, lo, hi in vocab:
        vals = samples[key]
        if len(vals) < min_samples:
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
    ap.add_argument("--only", default="",
                    help="comma-separated class names to fit (others are left alone)")
    ap.add_argument("--min-samples", type=int, default=ROBUST_N,
                    help="usable images a feature needs before it is fitted; classes fitted "
                         "with fewer than 25 are flagged small_sample")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite the fits file instead of merging into it")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="fraction of each class set aside, never fitted on, and "
                         "recorded in the fit so tools/library_eval.py can score it")
    args = ap.parse_args(argv)

    classes = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d)))
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        classes = [c for c in classes if c in wanted]
    tracker = PoseTracker(model=args.model, running_mode="image", smooth=False)
    fits = {}
    print("")
    print(f"{'class':10s}{'used':>7s}{'skipped':>9s}{'checks':>8s}   asana")
    print("-" * 62)
    for cls in classes:
        folder = os.path.join(args.root, cls)
        train_files, held = split_holdout(list_images(folder), args.holdout)
        checks, used, skipped = fit_class(tracker, folder, train_files, args.min_samples)
        if not checks:
            print(f"{cls:10s}{used:7d}{skipped:9d}{0:8d}   (too few usable images)")
            continue
        key, sanskrit, name, level = NAMES.get(
            cls, (cls, cls.title(), cls.title(), 2))
        fits[key] = {
            "key": key, "sanskrit": sanskrit, "name": name, "level": level,
            "source_class": cls, "n_images": used, "checks": checks,
            "source_root": os.path.relpath(args.root, os.path.dirname(OUT)),
        }
        if used < ROBUST_N:
            fits[key]["small_sample"] = True
        groups = VARIANT_GROUPS.get(cls)
        if groups:
            fits[key]["variants"] = {}
            for vname, steps in groups.items():
                vfiles = [f for f in train_files
                          if any(f"{os.sep}{s}{os.sep}" in f for s in steps)]
                vchecks, vused, _ = fit_class(tracker, folder, vfiles, args.min_samples)
                if vchecks:
                    fits[key]["variants"][vname] = {"checks": vchecks, "n_images": vused}
                    print(f"{'':10s}{vused:7d}{'':9s}{len(vchecks):8d}     variant '{vname}'")
        if held:
            fits[key]["holdout"] = [os.path.relpath(f, os.path.dirname(OUT)) for f in held]
        print(f"{cls:10s}{used:7d}{skipped:9d}{len(checks):8d}   {sanskrit}")
    tracker.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not args.replace and os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as fh:
            merged = json.load(fh)
        merged.update(fits)                      # new/refitted classes win, others kept
        fits = merged
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(fits, fh, indent=2)
    print("")
    print(f"{len(fits)} asana(s) fitted -> {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
