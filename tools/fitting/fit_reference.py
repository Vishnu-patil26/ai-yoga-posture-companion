"""Derive the reference angles for an asana **from a dataset**, not by hand.

The literature does not publish a standard table of ideal joint angles per
asana.  What the papers actually do (Thoutam et al. 2022; Anilkumar et al.
2021) is take the reference from data: measure a set of correctly performed
examples and use their central value.  This tool does that explicitly and
writes the numbers down so they can be cited and re-derived.

Method
------
1. Run BlazePose over every image in a folder of correctly-performed examples.
2. Drop frames the visibility gate rejects - the network could not see the body.
3. For each feature, report the **median** and the **MAD-based robust spread**
   (1.4826 x median absolute deviation, the consistent estimator of sigma for
   a normal).  Median and MAD are used rather than mean and standard deviation
   because a web-scraped pose dataset always contains some mislabelled images,
   side views and partial bodies, and those would drag a mean badly.
4. Tolerance = k x robust sigma, clamped to a sane floor and ceiling, so the
   band is set by how much correctly-performed examples actually vary.

    python tools/fitting/fit_reference.py data/datasets/yoga_poses/train/tree
    python tools/fitting/fit_reference.py <folder> --out yoga/reference_vrikshasana.json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from yoga import asanas as asana_lib                            # noqa: E402
from yoga.evaluator import compute_features, evaluate           # noqa: E402
from yoga.landmarks import PoseTracker                          # noqa: E402

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

#: Tolerance = TOL_K x robust sigma of the correct examples.  1.5 sigma keeps
#: roughly the middle 87% of correctly-performed examples inside the band.
TOL_K = 1.5

#: Floors and ceilings so a freak-tight or freak-wide sample cannot produce a
#: meaningless band.  (feature key -> (min tol, max tol))
TOL_BOUNDS = {
    "spine_tilt": (5.0, 14.0),
    "hip_level": (4.0, 12.0),
    "shoulder_level": (4.0, 12.0),
    "standing_knee": (6.0, 18.0),
    "folded_knee": (12.0, 45.0),
    "folded_thigh_open": (10.0, 35.0),
    "foot_height_ratio": (0.10, 0.30),
    "arm_raise_left": (12.0, 45.0),
    "arm_raise_right": (12.0, 45.0),
    "elbow_left": (15.0, 45.0),
    "elbow_right": (15.0, 45.0),
}


def robust_sigma(values: list[float]) -> float:
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    return 1.4826 * mad


def collect(folder: str, model: str, asana) -> tuple[list[dict], int, int]:
    files: list[str] = []
    for pattern in IMAGE_GLOBS:
        files.extend(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    files.sort()
    if not files:
        raise SystemExit("No images found under " + folder)

    tracker = PoseTracker(model=model, running_mode="image", smooth=False)
    feats: list[dict] = []
    no_body = 0
    rejected = 0
    for i, path in enumerate(files):
        img = cv2.imread(path)
        if img is None:
            continue
        pose = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if pose is None:
            no_body += 1
            continue
        f = compute_features(pose)
        if not evaluate(asana, f).usable:
            rejected += 1
            continue
        f["_path"] = path
        feats.append(f)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(files)} images, {len(feats)} usable")
    tracker.close()
    return feats, no_body, rejected


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit reference angles from a dataset")
    ap.add_argument("folder", help="folder of correctly-performed examples")
    ap.add_argument("--asana", default="vrikshasana")
    ap.add_argument("--model", default="full", choices=["full", "lite", "heavy"])
    ap.add_argument("--out", default=None, help="write the fitted reference JSON here")
    ap.add_argument("--tol-k", type=float, default=TOL_K)
    args = ap.parse_args(argv)

    asana = asana_lib.get(args.asana)
    print("fitting " + asana.sanskrit + " reference from " + args.folder)
    feats, no_body, rejected = collect(args.folder, args.model, asana)
    total = len(feats) + no_body + rejected
    print("")
    print(f"images                 : {total}")
    print(f"  no body detected     : {no_body}")
    print(f"  body too incomplete  : {rejected}")
    print(f"  usable for fitting   : {len(feats)}")
    if len(feats) < 20:
        raise SystemExit("Too few usable examples to fit a reference.")

    sides = [f["standing_side"] for f in feats]
    print(f"  standing leg         : {sides.count('left')} left / "
          f"{sides.count('right')} right")

    fitted = {
        "asana": asana.key,
        "source": os.path.relpath(args.folder).replace("\\", "/"),
        "n_images": total,
        "n_usable": len(feats),
        "method": f"median +/- {args.tol_k} x (1.4826 x MAD), bounded",
        "checks": {},
    }

    print("")
    header = ("check".ljust(22) + "median".rjust(9) + "robust sd".rjust(11)
              + "p10".rjust(9) + "p90".rjust(9)
              + "fitted tol".rjust(12) + "hand-set".rjust(11))
    print(header)
    print("-" * len(header))

    for check in asana.checks:
        vals = [f[check.key] for f in feats
                if isinstance(f.get(check.key), float) and not math.isnan(f[check.key])]
        if len(vals) < 20:
            continue
        vals.sort()
        med = statistics.median(vals)
        sd = robust_sigma(vals)
        p10 = vals[int(0.10 * (len(vals) - 1))]
        p90 = vals[int(0.90 * (len(vals) - 1))]
        lo, hi = TOL_BOUNDS.get(check.key, (check.tol * 0.5, check.tol * 3.0))
        tol = min(hi, max(lo, args.tol_k * sd))
        fitted["checks"][check.key] = {
            "target": round(med, 4), "tol": round(tol, 4),
            "robust_sd": round(sd, 4), "p10": round(p10, 4), "p90": round(p90, 4),
            "n": len(vals),
            "hand_set_target": check.target, "hand_set_tol": check.tol,
        }
        fmt = "{:9.2f}{:11.2f}{:9.2f}{:9.2f}{:12.2f}{:11.2f}"
        print(check.key.ljust(22) + fmt.format(med, sd, p10, p90, tol, check.tol)
              + f"   (target was {check.target:g})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(fitted, fh, indent=2)
        print("\nfitted reference -> " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
