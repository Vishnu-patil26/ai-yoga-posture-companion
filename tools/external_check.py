"""Does the score separate CORRECT from WRONG form?  (external, never fitted on)

Recognition (tools/library_eval.py) says "this is Tadasana".  This asks the harder
question a trainer must answer: given photos *labelled* right or wrong by the
dataset's authors ("Yoga for all", Zenodo 7818789), does the library score the
right ones higher?  The library is fitted on other photographs entirely
(tools/build_library.py), so these are unseen people in unseen rooms.

Reports, per pose: mean score of right vs wrong photos, AUC (probability a random
right photo outscores a random wrong one; 0.5 = no skill) and the share of right /
wrong photos that would start a hold.  Writes data/external_check.json.

    python tools/external_check.py
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from yoga import asanas as asana_lib                              # noqa: E402
from yoga.evaluator import compute_features, evaluate             # noqa: E402
from yoga.landmarks import PoseTracker                            # noqa: E402

ROOT = os.path.join(HERE, "data", "datasets", "yoga_for_all")
OUT = os.path.join(HERE, "data", "external_check.json")


def score_folder(tracker, asana, folder: str) -> list[float]:
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True)):
        img = cv2.imread(path)
        pose = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img is not None else None
        if pose is None:
            continue
        feats = compute_features(pose)
        if feats["group_visibility"]["torso"] < 0.6:
            continue
        out.append(evaluate(asana, feats).score)
    return out


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    a, b = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def main() -> int:
    tracker = PoseTracker(model="full", running_mode="image", smooth=False)
    report = {}
    print(f"\n{'pose':15s}{'n right':>8s}{'n wrong':>9s}{'mean right':>12s}{'mean wrong':>12s}"
          f"{'AUC':>7s}{'right>=enter':>14s}{'wrong>=enter':>14s}")
    print("-" * 92)
    for pose_key in ("tadasana", "marjaryasana", "bhujangasana"):
        asana = asana_lib.LIBRARY[pose_key]
        right = score_folder(tracker, asana, os.path.join(ROOT, pose_key, "right"))
        wrong_by = {}
        for d in sorted(glob.glob(os.path.join(ROOT, pose_key, "wrong", "step_*"))):
            wrong_by[os.path.basename(d)] = score_folder(tracker, asana, d)
        wrong = [s for v in wrong_by.values() for s in v]
        e = asana.enter_score
        report[pose_key] = {
            "n_right": len(right), "n_wrong": len(wrong),
            "mean_right": round(statistics.mean(right), 1), "mean_wrong": round(statistics.mean(wrong), 1),
            "auc": round(auc(right, wrong), 3),
            "right_ge_enter": round(sum(s >= e for s in right) / len(right), 3),
            "wrong_ge_enter": round(sum(s >= e for s in wrong) / len(wrong), 3),
            "wrong_by_fault_mean": {k: round(statistics.mean(v), 1) for k, v in wrong_by.items() if v},
        }
        r = report[pose_key]
        print(f"{pose_key:15s}{r['n_right']:>8d}{r['n_wrong']:>9d}{r['mean_right']:>12.1f}"
              f"{r['mean_wrong']:>12.1f}{r['auc']:>7.2f}{r['right_ge_enter']:>13.0%}{r['wrong_ge_enter']:>14.0%}")
    tracker.close()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
