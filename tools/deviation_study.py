"""Angle deviation of wrong poses (meeting 25/9: "angle deviation of wrong poses
-> correction").

The "Yoga for all" dataset labels every photo as a correct step ("Right Step k")
or the wrong version of the same step with the fault named ("Wrong Step k -
Legs", "- Hand and Legs", "- Head" ...).  That pairing is what this study uses:
for each wrong folder it measures how far every joint angle sits from the
matching *right* step, in degrees and in robust standard deviations of the right
step, and ranks the features by how well they separate the two (AUC).

Output
    data/deviation_tables.json    machine-readable, read by yoga/corrections.py
    docs/ANGLE_DEVIATION.md       the tables for the report

    python tools/deviation_study.py [data/datasets/yoga_for_all]
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
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from yoga.evaluator import compute_features                       # noqa: E402
from yoga.landmarks import PoseTracker                            # noqa: E402

FEATURES = [
    ("knee_bent", "bent knee"), ("knee_straight", "straighter knee"),
    ("elbow_bent", "bent elbow"), ("elbow_straight", "straighter elbow"),
    ("arm_raise_high", "higher arm"), ("arm_raise_low", "lower arm"),
    ("spine_tilt", "spine angle"), ("neck_dev", "neck alignment"),
    ("stance_width", "stance width"), ("shoulder_level", "shoulder level"),
    ("hip_level", "hip level"),
]
UNITS = {"stance_width": "x torso"}
OUT_JSON = os.path.join(HERE, "data", "deviation_tables.json")
OUT_MD = os.path.join(HERE, "docs", "ANGLE_DEVIATION.md")


def auc(a: list[float], b: list[float]) -> float:
    """P(a random wrong value > a random right value); 0.5 = no separation."""
    if not a or not b:
        return float("nan")
    x = np.asarray(a)[:, None]
    y = np.asarray(b)[None, :]
    return float(((x > y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def robust_sd(vals: list[float]) -> float:
    med = statistics.median(vals)
    return 1.4826 * statistics.median([abs(v - med) for v in vals])


def measure(tracker, folder: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
        img = cv2.imread(path)
        if img is None:
            continue
        pose = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if pose is None:
            continue
        f = compute_features(pose)
        if f["group_visibility"]["torso"] < 0.6:
            continue
        out.append({k: f[k] for k, _ in FEATURES if isinstance(f.get(k), float)
                    and not math.isnan(f[k])})
    return out


def study_pose(tracker, pose_dir: str) -> dict:
    right: dict[str, list[dict]] = {}
    wrong: dict[str, list[dict]] = {}
    for d in sorted(glob.glob(os.path.join(pose_dir, "right", "step_*"))):
        right[os.path.basename(d)[5:]] = measure(tracker, d)
    for d in sorted(glob.glob(os.path.join(pose_dir, "wrong", "step_*"))):
        wrong[os.path.basename(d)[5:]] = measure(tracker, d)

    rows = []
    for folder, samples in wrong.items():
        step, _, fault = folder.partition("-_")
        fault = fault.replace("_", " ") or "unspecified"
        ref = right.get(step) or []
        if len(samples) < 4 or len(ref) < 8:
            continue
        feats = []
        for key, label in FEATURES:
            r = [s[key] for s in ref if key in s]
            w = [s[key] for s in samples if key in s]
            if len(r) < 8 or len(w) < 4:
                continue
            sd = max(robust_sd(r), 1e-6)
            dev = statistics.median(w) - statistics.median(r)
            a = auc(w, r)
            feats.append({
                "feature": key, "label": label, "right_median": round(statistics.median(r), 3),
                "wrong_median": round(statistics.median(w), 3), "deviation": round(dev, 3),
                "z": round(dev / sd, 2), "auc": round(a, 3),
                "separation": round(abs(a - 0.5) * 2, 3), "n_right": len(r), "n_wrong": len(w),
            })
        feats.sort(key=lambda f: f["separation"], reverse=True)
        rows.append({"step": step, "fault": fault, "n_wrong": len(samples),
                     "n_right": len(ref), "features": feats})
    return {"right_steps": {k: len(v) for k, v in right.items()}, "faults": rows}


def to_markdown(result: dict) -> str:
    lines = ["# Angle deviation of wrong poses", "",
             "Source: 'Yoga for all' (Zenodo 7818789, CC BY 4.0). For every labelled wrong",
             "folder, each joint angle is compared with the matching *right* step.",
             "`deviation` = median(wrong) - median(right), in degrees (stance in torso",
             "lengths). `z` = deviation in robust standard deviations of the right step.",
             "`AUC` = chance that a wrong image has the larger value (0.5 = no signal).",
             "Only the three strongest features per fault are shown.", "",
             "**Limits.** Each pose in this dataset has one or two volunteers, so these are",
             "within-person deviations; the tables show *which* angles betray *which*",
             "fault, not population-wide thresholds (see docs/COLLECTION_PROTOCOL.md).", ""]
    for pose, data in result.items():
        lines += [f"## {pose}", ""]
        lines += ["| Wrong step | Fault | n | Feature | Right | Wrong | Deviation | z | AUC |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in data["faults"]:
            for i, f in enumerate(row["features"][:3]):
                head = (f"{row['step']} | {row['fault']} | {row['n_wrong']}" if i == 0
                        else " | | ")
                lines.append(f"| {head} | {f['label']} | {f['right_median']} | "
                             f"{f['wrong_median']} | {f['deviation']:+.1f} | {f['z']:+.1f} | "
                             f"{f['auc']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", default=os.path.join(HERE, "data", "datasets", "yoga_for_all"))
    args = ap.parse_args(argv)

    tracker = PoseTracker(model="full", running_mode="image", smooth=False)
    result = {}
    for pose in sorted(d for d in os.listdir(args.root)
                       if os.path.isdir(os.path.join(args.root, d))):
        print(f"measuring {pose} ...", flush=True)
        result[pose] = study_pose(tracker, os.path.join(args.root, pose))
        print(f"  {len(result[pose]['faults'])} wrong folders analysed", flush=True)
    tracker.close()

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(result))
    print(f"wrote {os.path.relpath(OUT_JSON, HERE)} and {os.path.relpath(OUT_MD, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
