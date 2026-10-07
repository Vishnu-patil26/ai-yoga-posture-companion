"""Export a landmark + angle dataset for every asana, in one documented layout.

For every photograph of every pose in the project's datasets this runs the same
BlazePose tracker the app uses and writes

    data/landmarks/
        manifest.csv           one row per source image, detected or not (nothing is dropped silently)
        columns.csv            data dictionary: every column, its unit and meaning
        all_poses.csv          every detected image, all poses, one table
        <pose>/
            landmarks.csv      image_id + 33 landmarks x (x, y, z, vis)           (132 + 1 columns)
            angles.csv         image_id + named joint angles / body measurements
            combined.csv       image metadata + angles + landmarks, ready for ML
            skeletons/<image_id>.png   the detected skeleton on a black background

Differences from a bare "one script per pose" export, on purpose:
  * one registry of sources, one loop - no path edited by hand per pose;
  * images with no detected person stay in manifest.csv with a reason;
  * ids are stable (<pose>_<source>_<nnnn>) and the label is the pose, not a file path;
  * landmarks are image-normalised (x, y in 0-1); z is the *world* depth in metres,
    hip-centred; angles are interior, mirror-invariant where it matters (columns.csv);
  * skeletons are drawn on black, never over the photograph, so no photo is reproduced;
  * UTF-8, no BOM, no unnamed index column, fixed rounding.

    python tools/datasets/export_landmarks.py [--out data/landmarks] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from yoga import angles as A                                       # noqa: E402
from yoga.evaluator import compute_features                        # noqa: E402
from yoga.landmarks import L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, PoseTracker  # noqa: E402
from yoga.taxonomy import FEATURES as FEATURE_DOC, LANDMARK_NAMES  # noqa: E402

DATA = os.path.join(HERE, "data")
D = os.path.join(DATA, "datasets")
OUT_DEFAULT = os.path.join(DATA, "landmarks")

LANDMARKS = [LANDMARK_NAMES[i].upper().replace(" ", "_") for i in range(33)]
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12),
               (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16),
               (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
               (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31),
               (28, 32)]
LEFT = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}

POSES = ["tadasana", "trikonasana", "virabhadrasana", "utkatasana", "vrikshasana", "adho_mukha",
         "bhujangasana", "marjaryasana", "balasana", "sukhasana"]

# name in output, source in compute_features / local computation, unit, group
ANGLE_COLUMNS = [
    ("left_knee_angle", "knee_left", "deg", "side"), ("right_knee_angle", "knee_right", "deg", "side"),
    ("left_elbow_angle", "elbow_left", "deg", "side"), ("right_elbow_angle", "elbow_right", "deg", "side"),
    ("left_shoulder_angle", "_sh_l", "deg", "side"), ("right_shoulder_angle", "_sh_r", "deg", "side"),
    ("left_hip_angle", "_hip_l", "deg", "side"), ("right_hip_angle", "_hip_r", "deg", "side"),
    ("left_arm_raise", "arm_raise_left", "deg", "side"), ("right_arm_raise", "arm_raise_right", "deg", "side"),
    ("knee_bent", "knee_bent", "deg", "mirror-invariant"), ("knee_straight", "knee_straight", "deg", "mirror-invariant"),
    ("elbow_bent", "elbow_bent", "deg", "mirror-invariant"), ("elbow_straight", "elbow_straight", "deg", "mirror-invariant"),
    ("arm_raise_high", "arm_raise_high", "deg", "mirror-invariant"), ("arm_raise_low", "arm_raise_low", "deg", "mirror-invariant"),
    ("hip_angle_min", "hip_angle_min", "deg", "mirror-invariant"), ("hip_angle_max", "hip_angle_max", "deg", "mirror-invariant"),
    ("arm_torso_min", "arm_torso_min", "deg", "mirror-invariant"), ("arm_torso_max", "arm_torso_max", "deg", "mirror-invariant"),
    ("hip_angle_vis", "hip_angle_vis", "deg", "visible side"), ("knee_angle_vis", "knee_angle_vis", "deg", "visible side"),
    ("elbow_angle_vis", "elbow_angle_vis", "deg", "visible side"), ("arm_torso_vis", "arm_torso_vis", "deg", "visible side"),
    ("spine_tilt", "spine_tilt", "deg", "body"), ("spine_tilt_signed", "spine_tilt_signed", "deg", "body"),
    ("neck_dev", "neck_dev", "deg", "body"), ("shoulder_level", "shoulder_level", "deg", "body"),
    ("hip_level", "hip_level", "deg", "body"), ("thigh_level", "thigh_level", "deg", "body"),
    ("hip_rise", "hip_rise", "torso lengths", "body"), ("head_drop", "head_drop", "torso lengths", "body"),
    ("stance_width", "stance_width", "torso lengths", "body"), ("frontality", "frontality", "ratio", "body"),
]
META = ["image_id", "pose", "variant", "source", "split", "quality", "fault", "licence", "image",
        "image_in_repo", "width", "height", "usable", "torso_visibility", "min_visibility"]
MANIFEST = ["image_id", "pose", "variant", "source", "split", "quality", "fault", "licence", "image",
            "image_in_repo", "detected", "usable", "failure", "width", "height"]

LIC = {"TF": "Apache-2.0", "Y107": "not stated by the host (web photographs); included for research and education",
       "COMMONS": "Creative Commons, per file (data/datasets/commons/credits.json)",
       "ZEN": "CC BY 4.0"}
#: committed to GitHub?  (see data/datasets/README.md)
IN_REPO = {"TF": True, "Y107": True, "COMMONS": True, "ZEN": True}


def _images(folder: str) -> list[str]:
    out: list[str] = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        out += glob.glob(os.path.join(folder, "**", pat), recursive=True)
    return sorted(out)


def _fit_holdouts() -> set[str]:
    try:
        fits = json.load(open(os.path.join(DATA, "asana_fits.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {h.replace("\\", "/") for s in fits.values() for h in s.get("holdout", [])}


def records() -> list[dict]:
    """Every source photograph with the pose, split, quality and licence it carries."""
    hold = _fit_holdouts()
    rec: list[dict] = []

    def add(path, pose, source, variant="", split="train", quality="unlabelled", fault=""):
        rec.append(dict(path=path, pose=pose, source=source, variant=variant, split=split,
                        quality=quality, fault=fault))

    tf = {"chair": ("utkatasana", ""), "cobra": ("bhujangasana", ""), "dog": ("adho_mukha", ""),
          "tree": ("vrikshasana", ""), "warrior": ("virabhadrasana", "mixed Warrior I/II/III")}
    for split in ("train", "test"):
        for cls, (pose, var) in tf.items():
            for p in _images(os.path.join(D, "yoga_poses", split, cls)):
                add(p, pose, "TF", var, split)
    y = {"adho_mukha_svanasana": ("adho_mukha", ""), "balasana": ("balasana", ""),
         "bhujangasana": ("bhujangasana", ""), "bitilasana": ("marjaryasana", "cow"),
         "marjaryasana": ("marjaryasana", "cat"), "sukhasana": ("sukhasana", ""),
         "tadasana": ("tadasana", "arms down"), "urdhva_hastasana": ("tadasana", "arms up"),
         "utkatasana": ("utkatasana", ""), "utthita_trikonasana": ("trikonasana", ""),
         "virabhadrasana_ii": ("virabhadrasana", "Warrior II")}
    for slug, (pose, var) in y.items():
        for p in _images(os.path.join(D, "yoga107", slug)):
            rel = os.path.relpath(p, DATA).replace("\\", "/")
            add(p, pose, "Y107", var, "holdout" if rel in hold else "train")
    for p in _images(os.path.join(D, "commons", "trikonasana")):
        rel = os.path.relpath(p, DATA).replace("\\", "/")
        add(p, "trikonasana", "COMMONS", "", "holdout" if rel in hold else "train")
    zen = {"tadasana": "tadasana", "marjaryasana": "marjaryasana", "bhujangasana": "bhujangasana"}
    for folder, pose in zen.items():
        for kind in ("right", "wrong"):
            for sd in sorted(glob.glob(os.path.join(D, "yoga_for_all", folder, kind, "step_*"))):
                step = os.path.basename(sd)[5:]
                num, _, fault = step.partition("-_")
                for p in _images(sd):
                    add(p, pose, "ZEN", f"step {num}", "external", kind, fault.replace("_", " "))
    rec.sort(key=lambda r: (POSES.index(r["pose"]), r["source"], r["path"]))
    counter: dict = {}
    for r in rec:
        k = (r["pose"], r["source"])
        counter[k] = counter.get(k, 0) + 1
        r["image_id"] = f"{r['pose']}_{r['source'].lower()}_{counter[k]:04d}"
    return rec


def draw_skeleton(pose, max_side: int = 320) -> np.ndarray:
    """The detected skeleton on a black canvas (never over the photograph)."""
    s = max_side / max(pose.width, pose.height)
    w, h = max(1, int(pose.width * s)), max(1, int(pose.height * s))
    img = np.zeros((h, w, 3), np.uint8)
    pts = (pose.pts * s).astype(int)
    for a, b in CONNECTIONS:
        col = (255, 160, 60) if (a in LEFT and b in LEFT) else (60, 160, 255) if (a not in LEFT and b not in LEFT and a > 0 and b > 0) else (200, 200, 200)
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_AA)
    for i in range(33):
        cv2.circle(img, tuple(pts[i]), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def measure(pose) -> dict:
    f = compute_features(pose)
    p = pose.pts
    f["_sh_l"] = A.joint_angle(p[L_HIP], p[L_SHOULDER], p[L_ELBOW])
    f["_sh_r"] = A.joint_angle(p[R_HIP], p[R_SHOULDER], p[R_ELBOW])
    f["_hip_l"] = A.joint_angle(p[L_SHOULDER], p[L_HIP], p[L_KNEE])
    f["_hip_r"] = A.joint_angle(p[R_SHOULDER], p[R_HIP], p[R_KNEE])
    return f


def _num(v, nd):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), nd)


def column_dictionary() -> list[dict]:
    rows = [dict(column="image_id", group="meta", unit="", definition="stable id: <pose>_<source>_<nnnn>")]
    meta = {"pose": "class label (one of the ten poses)", "variant": "accepted form or dataset step, if any",
            "source": "TF, Y107, COMMONS or ZEN (see data/datasets/README.md)",
            "split": "train / test / holdout (never fitted on) / external (never fitted on, labelled right-wrong)",
            "quality": "right / wrong when the source labels form; otherwise unlabelled",
            "fault": "named fault for wrong photos (ZEN only)", "licence": "licence of the source photo",
            "image": "path of the source photo relative to the repository root",
            "image_in_repo": "True if the photo is committed to GitHub; False if it must be fetched",
            "width": "photo width in pixels", "height": "photo height in pixels",
            "usable": "torso visible enough to score (the gate the fits use)",
            "torso_visibility": "lowest visibility of shoulders and hips, 0-1",
            "min_visibility": "lowest visibility over the core landmarks, 0-1"}
    for k, v in meta.items():
        rows.append(dict(column=k, group="meta", unit="", definition=v))
    for n in LANDMARKS:
        rows += [dict(column=f"{n}_x", group="landmark", unit="0-1", definition=f"{n} horizontal position, image-normalised"),
                 dict(column=f"{n}_y", group="landmark", unit="0-1", definition=f"{n} vertical position, image-normalised (down = larger)"),
                 dict(column=f"{n}_z", group="landmark", unit="metres", definition=f"{n} WORLD depth, hip-centred (not the image-relative z of the raw BlazePose output)"),
                 dict(column=f"{n}_vis", group="landmark", unit="0-1", definition=f"{n} visibility reported by BlazePose")]
    extra = {"left_shoulder_angle": "hip-shoulder-elbow angle, left", "right_shoulder_angle": "hip-shoulder-elbow angle, right",
             "left_hip_angle": "shoulder-hip-knee angle, left (180 = standing tall)", "right_hip_angle": "shoulder-hip-knee angle, right",
             "left_knee_angle": "hip-knee-ankle interior angle, left (180 = straight)", "right_knee_angle": "hip-knee-ankle interior angle, right",
             "left_elbow_angle": "shoulder-elbow-wrist interior angle, left", "right_elbow_angle": "shoulder-elbow-wrist interior angle, right",
             "left_arm_raise": "shoulder-to-wrist angle from vertical, left (0 = straight up)",
             "right_arm_raise": "shoulder-to-wrist angle from vertical, right",
             "frontality": "shoulder width over torso length (about 0.5+ = facing the camera, near 0 = side-on)"}
    for name, src, unit, group in ANGLE_COLUMNS:
        doc = extra.get(name) or (FEATURE_DOC[src].meaning if src in FEATURE_DOC else "")
        rows.append(dict(column=name, group=f"angle ({group})", unit=unit, definition=doc))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    ap.add_argument("--no-skeletons", action="store_true")
    args = ap.parse_args(argv)

    recs = records()
    if args.limit:
        recs = recs[:args.limit]
    tracker = PoseTracker(model="full", running_mode="image", smooth=False)
    lm_cols = [f"{n}_{s}" for n in LANDMARKS for s in ("x", "y", "z", "vis")]
    ang_cols = [c[0] for c in ANGLE_COLUMNS]
    per_pose: dict = {p: {"lm": [], "ang": [], "comb": []} for p in POSES}
    manifest, all_rows = [], []
    for k, r in enumerate(recs, 1):
        rel = os.path.relpath(r["path"], HERE).replace("\\", "/")
        base = {m: r.get(m, "") for m in ("image_id", "pose", "variant", "source", "split", "quality", "fault")}
        base.update(licence=LIC[r["source"]], image=rel, image_in_repo=IN_REPO[r["source"]])
        img = cv2.imread(r["path"])
        pose, why = None, ""
        if img is None:
            why = "unreadable image"
        else:
            pose = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if pose is None:
                why = "no person detected"
        if pose is None:
            h, w = (img.shape[:2] if img is not None else ("", ""))
            manifest.append({**base, "detected": False, "usable": False, "failure": why, "width": w, "height": h})
            continue
        f = measure(pose)
        usable = bool(f["group_visibility"]["torso"] >= 0.6)
        nxy = pose.pts / np.array([pose.width, pose.height], dtype=float)
        lm = {}
        for i, n in enumerate(LANDMARKS):
            lm[f"{n}_x"], lm[f"{n}_y"] = _num(nxy[i, 0], 6), _num(nxy[i, 1], 6)
            lm[f"{n}_z"], lm[f"{n}_vis"] = _num(pose.world[i, 2], 6), _num(pose.vis[i], 4)
        ang = {name: _num(f.get(src), 3 if unit == "deg" else 4) for name, src, unit, _ in ANGLE_COLUMNS}
        meta = {**base, "width": pose.width, "height": pose.height, "usable": usable,
                "torso_visibility": _num(f["group_visibility"]["torso"], 4), "min_visibility": _num(f["min_visibility"], 4)}
        pp = per_pose[r["pose"]]
        pp["lm"].append({"image_id": r["image_id"], **lm})
        pp["ang"].append({"image_id": r["image_id"], **ang})
        pp["comb"].append({**{m: meta[m] for m in META}, **ang, **lm})
        manifest.append({**base, "detected": True, "usable": usable, "failure": "", "width": pose.width, "height": pose.height})
        if not args.no_skeletons:
            d = os.path.join(args.out, r["pose"], "skeletons")
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, r["image_id"] + ".png"), draw_skeleton(pose))
        if k % 300 == 0:
            print(f"  {k} / {len(recs)}", flush=True)
    tracker.close()

    def write(path, cols, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    for pose, d in per_pose.items():
        if not d["lm"]:
            continue
        write(os.path.join(args.out, pose, "landmarks.csv"), ["image_id"] + lm_cols, d["lm"])
        write(os.path.join(args.out, pose, "angles.csv"), ["image_id"] + ang_cols, d["ang"])
        write(os.path.join(args.out, pose, "combined.csv"), META + ang_cols + lm_cols, d["comb"])
        all_rows += d["comb"]
    write(os.path.join(args.out, "all_poses.csv"), META + ang_cols + lm_cols, all_rows)
    write(os.path.join(args.out, "manifest.csv"), MANIFEST, manifest)
    write(os.path.join(args.out, "columns.csv"), ["column", "group", "unit", "definition"], column_dictionary())

    det = sum(1 for m in manifest if m["detected"])
    print(f"\n{'pose':15s}{'images':>8s}{'detected':>10s}{'usable':>8s}")
    for pose in POSES:
        ms = [m for m in manifest if m["pose"] == pose]
        print(f"{pose:15s}{len(ms):>8d}{sum(m['detected'] for m in ms):>10d}{sum(m['usable'] for m in ms):>8d}")
    print(f"{'total':15s}{len(manifest):>8d}{det:>10d}{sum(m['usable'] for m in manifest):>8d}")
    print(f"\nwritten to {os.path.relpath(args.out, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
