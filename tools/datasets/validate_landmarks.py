"""Check that data/landmarks/ is internally consistent - the opposite of a CSV that merely looks tidy.

Verifies, for every pose and for the dataset as a whole:
  * the same files exist in every pose folder, with the documented columns and no BOM or unnamed index;
  * landmarks.csv, angles.csv and combined.csv hold the same image ids in the same order;
  * every manifest row is accounted for: detected images appear exactly once in their pose tables, failures
    carry a reason, nothing is dropped silently;
  * ids are unique and carry the right pose; splits and qualities are from the allowed sets;
  * visibilities are 0-1, normalised coordinates sit near the image, no NaN/inf text, angles are in range;
  * every detected image has its skeleton PNG, and columns.csv documents every column.

    python tools/datasets/validate_landmarks.py [data/landmarks]
"""

from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from tools.datasets import export_landmarks as ex                  # noqa: E402

SPLITS = {"train", "test", "holdout", "external"}
QUALITY = {"right", "wrong", "unlabelled"}


def _read(path: str) -> tuple[list[str], list[dict]]:
    raw = open(path, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path}: has a UTF-8 BOM")
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames or []), list(rd)


def validate(root: str) -> list[str]:
    bad: list[str] = []
    lm_cols = [f"{n}_{s}" for n in ex.LANDMARKS for s in ("x", "y", "z", "vis")]
    ang_cols = [c[0] for c in ex.ANGLE_COLUMNS]
    want = {"landmarks.csv": ["image_id"] + lm_cols, "angles.csv": ["image_id"] + ang_cols,
            "combined.csv": ex.META + ang_cols + lm_cols}
    try:
        mf_cols, manifest = _read(os.path.join(root, "manifest.csv"))
        _, docs = _read(os.path.join(root, "columns.csv"))
        all_cols, all_rows = _read(os.path.join(root, "all_poses.csv"))
    except (OSError, ValueError) as exc:
        return [f"cannot read top-level files: {exc}"]
    if mf_cols != ex.MANIFEST:
        bad.append("manifest.csv columns differ from the documented layout")
    if all_cols != want["combined.csv"]:
        bad.append("all_poses.csv columns differ from combined.csv layout")
    documented = {d["column"] for d in docs}
    missing_doc = [c for c in want["combined.csv"] if c not in documented]
    if missing_doc:
        bad.append(f"columns.csv does not document: {missing_doc[:5]}")
    ids = [m["image_id"] for m in manifest]
    if len(ids) != len(set(ids)):
        bad.append("manifest has duplicate image ids")
    for m in manifest:
        if m["detected"] != "True" and not m["failure"]:
            bad.append(f"{m['image_id']}: not detected but no failure reason")
        if m["split"] not in SPLITS or m["quality"] not in QUALITY:
            bad.append(f"{m['image_id']}: unknown split/quality")
        if m["source"] == "ZEN" and (m["split"] != "external" or m["quality"] == "unlabelled"):
            bad.append(f"{m['image_id']}: labelled source must be external with right/wrong")
        if not m["image_id"].startswith(m["pose"] + "_"):
            bad.append(f"{m['image_id']}: id does not carry its pose")

    seen = 0
    for pose in ex.POSES:
        det = [m for m in manifest if m["pose"] == pose and m["detected"] == "True"]
        pdir = os.path.join(root, pose)
        if not det:
            continue
        tables = {}
        for name, cols in want.items():
            try:
                got_cols, rows = _read(os.path.join(pdir, name))
            except (OSError, ValueError) as exc:
                bad.append(f"{pose}/{name}: {exc}")
                continue
            if got_cols != cols:
                bad.append(f"{pose}/{name}: columns differ from the documented layout")
            tables[name] = rows
        if len(tables) != 3:
            continue
        order = [r["image_id"] for r in tables["combined.csv"]]
        for name in ("landmarks.csv", "angles.csv"):
            if [r["image_id"] for r in tables[name]] != order:
                bad.append(f"{pose}/{name}: ids differ from combined.csv")
        if sorted(order) != sorted(m["image_id"] for m in det):
            bad.append(f"{pose}: detected images in manifest != rows in tables")
        seen += len(order)
        for r in tables["combined.csv"]:
            for c in lm_cols:
                v = r[c]
                if v.lower() in ("nan", "inf", "-inf", ""):
                    bad.append(f"{r['image_id']}.{c}: '{v}'")
                    break
                x = float(v)
                if c.endswith("_vis") and not 0 <= x <= 1:
                    bad.append(f"{r['image_id']}.{c}: visibility {x}")
                if c.endswith(("_x", "_y")) and not -0.6 <= x <= 1.6:
                    bad.append(f"{r['image_id']}.{c}: coordinate {x} far outside the image")
            for c in ang_cols:
                v = r[c]
                if v in ("",):
                    continue
                if v.lower() in ("nan", "inf", "-inf"):
                    bad.append(f"{r['image_id']}.{c}: '{v}'")
                elif c.endswith("_angle") and not 0 <= float(v) <= 180.5:
                    bad.append(f"{r['image_id']}.{c}: angle {v}")
            if not os.path.isfile(os.path.join(pdir, "skeletons", r["image_id"] + ".png")):
                bad.append(f"{r['image_id']}: skeleton PNG missing")
    if seen != len(all_rows):
        bad.append(f"all_poses.csv has {len(all_rows)} rows, per-pose tables have {seen}")
    return bad


def main(argv=None) -> int:
    root = (argv or sys.argv[1:] or [ex.OUT_DEFAULT])[0]
    problems = validate(root)
    for p in problems[:40]:
        print("  PROBLEM", p)
    print(f"\n{len(problems)} problem(s) in {os.path.relpath(root, HERE)}" if problems
          else f"\n{os.path.relpath(root, HERE)} is consistent")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
