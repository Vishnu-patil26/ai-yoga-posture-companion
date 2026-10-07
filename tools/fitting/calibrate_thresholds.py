"""Set each pose's hold threshold from data instead of one number for everyone.

A pose with wide fitted tolerances (Cobra, Chair) hands out high scores to bodies
that are doing something else; a tight pose (Tadasana) does not.  One shared
"start the hold at 70" therefore means different things per pose.  This picks, for
each fitted pose, the lowest start score at which at most ``--max-false`` of the
held-out photographs of OTHER poses would start its hold, clamped to 60-90, and
records it (and the release score, 15 below) in data/asana_fits.json, which
yoga/asanas.py reads at load time.

Caveat, stated plainly: the held-out photographs are used here to choose a
threshold, so the recognition figures in tools/evaluation/library_eval.py are slightly
optimistic after this runs.  The external right/wrong check (tools/evaluation/external_check.py)
never sees any of it.

    python tools/fitting/calibrate_thresholds.py [--max-false 0.03]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from tools.evaluation import library_eval as le                   # noqa: E402
from yoga import asanas as asana_lib                               # noqa: E402
from yoga.evaluator import compute_features, evaluate             # noqa: E402
from yoga.landmarks import PoseTracker                             # noqa: E402

LO, HI, RELEASE_GAP = 60, 90, 15
MIN_OWN = 0.70                      #: the intended pose must be able to start its hold


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-false", type=float, default=0.03,
                    help="largest share of other poses' photos allowed to start this hold")
    args = ap.parse_args(argv)

    sets = le.held_out_sets()
    tracker = PoseTracker(model="full", running_mode="image", smooth=False)
    feats = {}                                               # (true pose, path) -> features
    for true, files in sets.items():
        for path in files:
            img = cv2.imread(path)
            pose = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img is not None else None
            if pose is None:
                continue
            f = compute_features(pose)
            if f["group_visibility"]["torso"] >= 0.6:
                feats[(true, path)] = f
    tracker.close()

    with open(le.FITS, encoding="utf-8") as fh:
        fits = json.load(fh)
    print(f"{'pose':15s}{'enter':>6s}{'own pass':>10s}{'others pass':>13s}")
    print("-" * 44)
    for key, asana in asana_lib.LIBRARY.items():
        if key == "vrikshasana" or key not in fits:
            continue                                         # hand-tuned: left at 78
        own = [evaluate(asana, f).score for (t, _), f in feats.items() if t == key]
        other = [evaluate(asana, f).score for (t, _), f in feats.items() if t != key]
        own_pass = lambda t: sum(s >= t for s in own) / max(1, len(own))        # noqa: E731
        false_pass = lambda t: sum(s >= t for s in other) / max(1, len(other))  # noqa: E731
        enter = next((t for t in range(LO, HI + 1) if false_pass(t) <= args.max_false), HI)
        permissive = own_pass(enter) < MIN_OWN
        if permissive:
            # The false-accept limit would make the hold unreachable for the person
            # actually doing the pose.  Usability wins; the pose is flagged instead.
            enter = max((t for t in range(LO, HI + 1) if own_pass(t) >= MIN_OWN), default=LO)
        fits[key]["enter_score"] = float(enter)
        fits[key]["exit_score"] = float(enter - RELEASE_GAP)
        fits[key]["permissive"] = bool(permissive)
        print(f"{key:15s}{enter:>6d}{own_pass(enter):>10.0%}{false_pass(enter):>13.1%}"
              + ("   permissive: cannot separate itself from other poses" if permissive else ""))
    with open(le.FITS, "w", encoding="utf-8") as fh:
        json.dump(fits, fh, indent=2)
    print(f"\nthresholds written to {os.path.relpath(le.FITS, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
