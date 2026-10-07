"""Held-out validation of the fitted reference (Methodology Stage 5).

The reference angles were fitted on the *train* split of one asana.  This
scores the *test* split - never seen during fitting - and also scores four
other asanas, which act as the negative class: a system that says "that is a
correct Tree Pose" must say it about tree images and not about dog, cobra,
chair or warrior images.

    python tools/evaluation/validate_dataset.py data/datasets/yoga_poses/test --positive tree
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from yoga import asanas as asana_lib                            # noqa: E402
from yoga.evaluator import compute_features, evaluate           # noqa: E402
from yoga.landmarks import PoseTracker                          # noqa: E402

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png")


def score_folder(tracker, asana, folder: str) -> tuple[list[float], int, dict]:
    files: list[str] = []
    for pattern in IMAGE_GLOBS:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    files.sort()
    scores: list[float] = []
    variants: dict[str, int] = {}
    skipped = 0
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        pose = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if pose is None:
            skipped += 1
            continue
        ev = evaluate(asana, compute_features(pose))
        if not ev.usable:
            skipped += 1
            continue
        scores.append(ev.score)
        variants[ev.variant_name] = variants.get(ev.variant_name, 0) + 1
    return scores, skipped, variants


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Held-out validation")
    ap.add_argument("root", help="folder containing one sub-folder per class")
    ap.add_argument("--positive", default="tree", help="the class the asana describes")
    ap.add_argument("--asana", default="vrikshasana")
    ap.add_argument("--model", default="full", choices=["full", "lite", "heavy"])
    args = ap.parse_args(argv)

    asana = asana_lib.get(args.asana)
    classes = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d)))
    if args.positive not in classes:
        raise SystemExit(f"'{args.positive}' is not one of {classes}")

    tracker = PoseTracker(model=args.model, running_mode="image", smooth=False)
    per_class: dict[str, list[float]] = {}

    print("")
    print(f"scoring every image against {asana.sanskrit}, threshold "
          f"{asana.enter_score:.0f}%")
    print("")
    head = ("class".ljust(12) + "n".rjust(5) + "skipped".rjust(9) + "mean".rjust(9)
            + "median".rjust(9) + "p10".rjust(8) + "p90".rjust(8) + "  >= threshold")
    print(head)
    print("-" * len(head))
    for cls in classes:
        scores, skipped, variants = score_folder(
            tracker, asana, os.path.join(args.root, cls))
        if not scores:
            continue
        per_class[cls] = scores
        s = sorted(scores)
        share = 100.0 * sum(1 for v in scores if v >= asana.enter_score) / len(scores)
        mark = "  <- positive class" if cls == args.positive else ""
        print(cls.ljust(12) + f"{len(scores):5d}{skipped:9d}"
              + f"{statistics.mean(scores):9.1f}{statistics.median(scores):9.1f}"
              + f"{s[len(s) // 10]:8.1f}{s[9 * len(s) // 10]:8.1f}"
              + f"{share:14.0f}%" + mark)
    tracker.close()

    pos = per_class.get(args.positive, [])
    neg = [v for c, vs in per_class.items() if c != args.positive for v in vs]
    if not pos or not neg:
        return 0

    print("")
    print("separation")
    thr = asana.enter_score
    tp = sum(1 for v in pos if v >= thr)
    fp = sum(1 for v in neg if v >= thr)
    fn = len(pos) - tp
    tn = len(neg) - fp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(f"  at the live threshold ({thr:.0f}%): "
          f"precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
    print(f"    accepted {tp}/{len(pos)} {args.positive} images, "
          f"and {fp}/{len(neg)} images of other asanas")

    # AUC by rank (probability a random tree image outscores a random non-tree)
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0        # 1-based average rank for ties
        rank_sum += sum(avg_rank for k in range(i, j) if merged[k][1] == 1)
        i = j
    auc = (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    print(f"  AUC (ranking, threshold-free): {auc:.3f}")

    best = max(((t, sum(1 for v in pos if v >= t) / len(pos)
                 - sum(1 for v in neg if v >= t) / len(neg))
                for t in range(30, 100)), key=lambda x: x[1])
    print(f"  best separating threshold would be {best[0]}% "
          f"(Youden J = {best[1]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
