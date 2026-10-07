"""Offline checks for the landmark/angle dataset export (no camera).

    python tests/test_landmark_export.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from tools.datasets import export_landmarks as ex                  # noqa: E402
from tools.datasets import validate_landmarks as vl                # noqa: E402

FAILS: list[str] = []
N = 0


def check(cond: bool, label: str) -> None:
    global N
    N += 1
    if not cond:
        FAILS.append(label)
        print(f"  FAIL  {label}")


class _Pose:                                   # the few fields draw_skeleton reads
    width, height = 640, 480
    pts = np.tile(np.array([320.0, 240.0]), (33, 1)) + np.arange(33)[:, None] * 3.0


def main() -> int:
    check(len(ex.LANDMARKS) == 33 and len(set(ex.LANDMARKS)) == 33, "33 unique landmark names")
    check(all(0 <= a < 33 and 0 <= b < 33 for a, b in ex.CONNECTIONS), "skeleton connections index real landmarks")
    check(len(ex.CONNECTIONS) == len(set(ex.CONNECTIONS)), "no duplicate connections")
    names = [c[0] for c in ex.ANGLE_COLUMNS]
    check(len(names) == len(set(names)), "angle column names are unique")
    check(set(ex.POSES) == {"tadasana", "trikonasana", "virabhadrasana", "utkatasana", "vrikshasana",
                            "adho_mukha", "bhujangasana", "marjaryasana", "balasana", "sukhasana"},
          "the ten poses")
    doc = {d["column"] for d in ex.column_dictionary()}
    cols = ex.META + names + [f"{n}_{s}" for n in ex.LANDMARKS for s in ("x", "y", "z", "vis")]
    check(all(c in doc for c in cols), "the data dictionary covers every column")
    check(all(d["definition"] for d in ex.column_dictionary()), "every column has a definition")
    img = ex.draw_skeleton(_Pose())
    check(img.shape[2] == 3 and max(img.shape[:2]) == 320 and img.any(), "skeleton image is 320 px and not blank")
    recs = ex.records()
    ids = [r["image_id"] for r in recs]
    check(len(ids) == len(set(ids)), "every source image gets a unique stable id")
    check(all(r["pose"] in ex.POSES for r in recs), "every record belongs to one of the ten poses")
    check(all(r["split"] in vl.SPLITS and r["quality"] in vl.QUALITY for r in recs), "allowed splits and qualities")
    check(all(r["split"] == "external" for r in recs if r["source"] == "ZEN"), "right/wrong photos are never fitted on")
    root = ex.OUT_DEFAULT
    if os.path.isfile(os.path.join(root, "manifest.csv")):
        problems = vl.validate(root)
        for p in problems[:10]:
            print("   ", p)
        check(not problems, "data/landmarks is internally consistent")
    else:
        print("  skip  data/landmarks not generated (run tools/datasets/export_landmarks.py)")
    print(f"\n{N - len(FAILS)}/{N} landmark-export checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
