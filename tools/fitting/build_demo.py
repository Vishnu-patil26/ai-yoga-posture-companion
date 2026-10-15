"""Pick the real, typical body shape of each pose for the demo animations.

The demo "videos" (yoga/demo.py) are drawn from data, not invented: for every pose
this takes the landmark table in data/landmarks/ (BlazePose on real photographs),
puts every usable photo in a common frame - hips at the origin, one torso length
as the unit, side-on poses all facing the same way, front-on poses also mirrored
so left and right count the same - and keeps the *medoid*: the one real photograph
whose shape is closest to the typical (median) distance of all the others.  Its 13
joints are written to data/demo_poses.json together with the id of the photograph
they came from.

Cat-Cow gets two shapes (cat and cow) because the exercise is the movement between
them.  Photographs from the labelled "Yoga for all" set (one volunteer, many camera
angles) are left out so a single person does not decide the shape.

    python tools/fitting/build_demo.py                -> data/demo_poses.json
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

LANDMARKS = os.path.join(HERE, "data", "landmarks")
OUT = os.path.join(HERE, "data", "demo_poses.json")

#: the joints the demo figure is drawn from, in a fixed order
JOINTS = ("NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
          "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
          "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX")
CORE = JOINTS[1:13]                     # shoulders .. ankles: must be in the picture
MIRROR = {j: j.replace("LEFT", "@").replace("RIGHT", "LEFT").replace("@", "RIGHT") for j in JOINTS}

#: which photographs stand for each pose: (pose, shape name) -> manifest variant ("" = any)
RECIPE = {
    "tadasana": {"shape": "arms down"},
    "trikonasana": {"shape": ""},
    "virabhadrasana": {"shape": "Warrior II"},
    "utkatasana": {"shape": ""},
    "vrikshasana": {"shape": ""},
    "adho_mukha": {"shape": ""},
    "bhujangasana": {"shape": ""},
    "marjaryasana": {"cat": "cat", "cow": "cow"},
    "balasana": {"shape": ""},
    "sukhasana": {"shape": ""},
}
SIDE_ON = {"utkatasana", "adho_mukha", "bhujangasana", "marjaryasana", "balasana"}
#: For these the typical photograph from the pooled datasets is a poor picture of the pose
#: (limbs hidden or mis-placed in many of them), so the shape comes from the clean textbook
#: photograph the app already shows in its gallery (assets/gallery, with credits in
#: assets/gallery/credits.json), read by the same BlazePose tracker.
FROM_GALLERY = {"trikonasana", "virabhadrasana", "utkatasana", "vrikshasana", "adho_mukha", "bhujangasana"}
LANDMARK_ID = {"NOSE": 0, "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12, "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
               "LEFT_WRIST": 15, "RIGHT_WRIST": 16, "LEFT_HIP": 23, "RIGHT_HIP": 24, "LEFT_KNEE": 25,
               "RIGHT_KNEE": 26, "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28, "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
               "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32}


#: how strict to be about "the whole body is in the picture": (min visibility, margin outside the frame).
#: Poses on the floor are often cut off or self-occluded, so the ladder is climbed until
#: enough photographs remain.
LADDER = ((0.35, 0.02), (0.2, 0.05), (0.1, 0.1), (0.0, 0.3))
MIN_POOL = 25


def _load(pose: str, min_vis: float, margin: float) -> list[dict]:
    """Usable, unlabelled-or-right photographs of one pose with their joints in pixels."""
    meta = {}
    with open(os.path.join(LANDMARKS, "manifest.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["pose"] == pose and r["usable"] == "True" and r["source"] != "ZEN" \
                    and r["quality"] != "wrong":
                meta[r["image_id"]] = r
    out = []
    with open(os.path.join(LANDMARKS, pose, "landmarks.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = meta.get(r["image_id"])
            if m is None:
                continue
            w, h = float(m["width"]), float(m["height"])
            pts, vis, inside = {}, [], True
            for j in JOINTS:
                x, y, v = float(r[j + "_x"]), float(r[j + "_y"]), float(r[j + "_vis"])
                pts[j] = (x * w, y * h)
                if j in CORE:
                    vis.append(v)
                    if not (-margin <= x <= 1 + margin and -margin <= y <= 1 + margin):
                        inside = False
            if inside and min(vis) >= min_vis:
                out.append({"id": r["image_id"], "variant": m["variant"], "pts": pts})
    return out


def _normalise(pts: dict, face_right: bool) -> dict | None:
    """Hips at the origin, torso length 1, y down; side-on poses face +x."""
    hip = (np.array(pts["LEFT_HIP"]) + np.array(pts["RIGHT_HIP"])) / 2
    sho = (np.array(pts["LEFT_SHOULDER"]) + np.array(pts["RIGHT_SHOULDER"])) / 2
    torso = float(np.linalg.norm(sho - hip))
    if torso < 20:
        return None
    out = {j: (np.array(p) - hip) / torso for j, p in pts.items()}
    if face_right:
        nose = out["NOSE"]
        s = (sho - hip) / torso
        lean = s[0] + 0.5 * (nose[0] - s[0])
        if lean < 0:
            out = {j: np.array([-p[0], p[1]]) for j, p in out.items()}
    return out


def _mirrored(shape: dict) -> dict:
    return {MIRROR[j]: np.array([-p[0], p[1]]) for j, p in shape.items()}


def medoid(shapes: list[dict], ids: list[str]) -> tuple[int, int]:
    """Index of the typical shape, and how many shapes were compared."""
    arr = np.array([[s[j] for j in CORE + ("NOSE",)] for s in shapes])        # (N, 13, 2)
    d = np.linalg.norm(arr[:, None] - arr[None, :], axis=-1).mean(axis=-1)    # (N, N)
    return int(np.argmin(np.median(d, axis=1))), len(shapes)


def _from_gallery(pose: str, tracker) -> dict:
    import cv2
    path = os.path.join(HERE, "assets", "gallery", pose + ".jpg")
    img = cv2.imread(path)
    found = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img is not None else None
    if found is None:
        raise SystemExit(f"{pose}: no person found in {path}")
    pts = {j: tuple(map(float, found.pts[i])) for j, i in LANDMARK_ID.items()}
    s = _normalise(pts, pose in SIDE_ON)
    if s is None:
        raise SystemExit(f"{pose}: body too small in {path}")
    print(f"  {pose:15s} shape  from assets/gallery/{pose}.jpg (textbook photograph)")
    return {"source": f"assets/gallery/{pose}.jpg", "compared": 1,
            "pts": {j: [round(float(s[j][0]), 4), round(float(s[j][1]), 4)] for j in JOINTS}}


def build() -> dict:
    result: dict = {}
    tracker = None
    for pose, shapes in RECIPE.items():
        if pose in FROM_GALLERY:
            if tracker is None:
                from yoga.landmarks import PoseTracker
                tracker = PoseTracker(model="full", running_mode="image", smooth=False)
            result[pose] = {"view": "side" if pose in SIDE_ON else "front",
                            "shapes": {"shape": _from_gallery(pose, tracker)}}
            continue
        entry: dict = {"view": "side" if pose in SIDE_ON else "front", "shapes": {}}
        for name, variant in shapes.items():
            for min_vis, margin in LADDER:
                photos = _load(pose, min_vis, margin)
                pool = photos if (name == "shape" and not variant) else                     [p for p in photos if p["variant"] == variant]
                if len(pool) >= MIN_POOL:
                    break
            norm, ids = [], []
            for p in pool:
                s = _normalise(p["pts"], pose in SIDE_ON)
                if s is None:
                    continue
                norm.append(s)
                ids.append(p["id"])
                if pose not in SIDE_ON:                    # front-on: the mirror image is as good a photo
                    norm.append(_mirrored(s))
                    ids.append(p["id"] + "+mirror")
            if len(norm) < 5:
                raise SystemExit(f"{pose}/{name}: only {len(norm)} usable photographs")
            k, n = medoid(norm, ids)
            entry["shapes"][name] = {
                "source": ids[k], "compared": n,
                "pts": {j: [round(float(norm[k][j][0]), 4), round(float(norm[k][j][1]), 4)]
                        for j in JOINTS}}
            print(f"  {pose:15s} {name:6s} from {ids[k]:32s} ({n} compared)")
        result[pose] = entry
    if tracker is not None:
        tracker.close()
    return result


def main() -> int:
    data = build()
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"note": "medoid shapes from data/landmarks, hips at origin, torso length 1, y down",
                   "joints": list(JOINTS), "poses": data}, fh, indent=1)
    print(f"wrote {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
