"""Run the trainer over a recorded image or video, headless.

This is the harness Stage 5 of the methodology needs: point it at a clip, get a
per-frame CSV of every joint angle, its deviation and the alignment score.
Those rows are what get compared against a yoga instructor's frame labels to
produce accuracy / precision / recall and the joint-angle error table.

    python tools/session/analyse.py clip.mp4
    python tools/session/analyse.py clip.mp4 --csv out.csv --video annotated.mp4
    python tools/session/analyse.py photo.jpg --image annotated.png
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from yoga import asanas as asana_lib                            # noqa: E402
from yoga import calibration                                    # noqa: E402
from yoga.evaluator import compute_features, evaluate           # noqa: E402
from yoga.landmarks import PoseTracker                          # noqa: E402
from yoga.overlay import draw_hud, draw_skeleton                # noqa: E402
from yoga.state_machine import PoseStateMachine                 # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Headless analysis of a clip or photo")
    ap.add_argument("source", help="path to an image or video file")
    ap.add_argument("--asana", default="vrikshasana")
    ap.add_argument("--model", default="full", choices=["full", "lite", "heavy"])
    ap.add_argument("--user", default=None, help="apply this calibration profile")
    ap.add_argument("--csv", default=None, help="write per-frame measurements here")
    ap.add_argument("--video", default=None, help="write the annotated video here")
    ap.add_argument("--image", default=None, help="write the annotated still here")
    ap.add_argument("--stride", type=int, default=1, help="analyse every Nth frame")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.source):
        raise SystemExit("No such file: " + args.source)

    asana = asana_lib.get(args.asana)
    if args.user:
        asana = calibration.apply_profile(asana, calibration.load_profile(args.user))

    is_image = os.path.splitext(args.source)[1].lower() in IMAGE_EXT
    tracker = PoseTracker(model=args.model, running_mode="image" if is_image else "video")
    sm = PoseStateMachine(hold_target_s=asana.hold_target_s,
                          enter_score=asana.enter_score, exit_score=asana.exit_score)

    writer = None
    rows: list[dict] = []
    scores: list[float] = []
    missed = 0
    n = 0

    if is_image:
        frames = [(0.0, cv2.imread(args.source))]
        fps = 1.0
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise SystemExit("Could not open " + args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = _iter_frames(cap, fps, args.stride)

    t_start = time.perf_counter()
    for t, frame in frames:
        if frame is None:
            continue
        n += 1
        pose = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(t * 1000), fps)
        ev = None
        if pose is None:
            missed += 1
            sm.update(t, 0.0, False)
        else:
            ev = evaluate(asana, compute_features(pose))
            sm.update(t, ev.score, ev.usable)
            if ev.usable:
                scores.append(ev.score)
                row = {"t_s": round(t, 3), "score": round(ev.score, 2),
                       "state": sm.state.value,
                       "standing_side": ev.features.get("standing_side", "")}
                for r in ev.results:
                    if r.ok is None:
                        continue
                    row[r.check.key] = round(r.value, 3)
                    row[r.check.key + "_dev"] = round(r.deviation, 3)
                    row[r.check.key + "_ok"] = int(r.ok)
                rows.append(row)

        if args.video or args.image:
            if pose is not None:
                draw_skeleton(frame, pose, ev)
            banner = (ev.primary.cue() if (ev and ev.primary)
                      else ("Good. Hold it." if ev and ev.usable else "No body detected"))
            draw_hud(frame, ev, sm, banner, fps=fps, hint=asana.setup_hint)
            if args.image:
                cv2.imwrite(args.image, frame)
            elif args.video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.video, fourcc, fps / max(1, args.stride),
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)

    if writer is not None:
        writer.release()
    tracker.close()
    wall = time.perf_counter() - t_start

    if args.csv and rows:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print("")
    print("source            : " + args.source)
    print(f"frames analysed   : {n}   ({missed} with no body detected)")
    print(f"processing rate   : {n / max(1e-6, wall):.1f} fps "
          f"({1000 * wall / max(1, n):.1f} ms per frame)")
    if scores:
        print(f"alignment score   : mean {statistics.mean(scores):.1f}%  "
              f"median {statistics.median(scores):.1f}%  "
              f"best {max(scores):.1f}%  worst {min(scores):.1f}%")
    print(f"holds completed   : {sm.stats.completed}   "
          f"longest {sm.stats.longest_s:.1f}s   total in pose {sm.stats.total_hold_s:.1f}s")

    if rows:
        print("")
        print("per-joint summary (frames scored while the body was visible)")
        print("check".ljust(24) + "mean".rjust(9) + "sd".rjust(9)
              + "mean|dev|".rjust(11) + "in tol".rjust(9))
        for c in asana.checks:
            vals = [r[c.key] for r in rows if c.key in r]
            devs = [abs(r[c.key + "_dev"]) for r in rows if c.key + "_dev" in r]
            oks = [r[c.key + "_ok"] for r in rows if c.key + "_ok" in r]
            if not vals:
                continue
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            print(c.key.ljust(24)
                  + f"{statistics.mean(vals):9.2f}{sd:9.2f}"
                  + f"{statistics.mean(devs):11.2f}"
                  + f"{100.0 * sum(oks) / len(oks):8.0f}%")
    if args.csv and rows:
        print("\nper-frame measurements -> " + args.csv)
    if args.video:
        print("annotated video        -> " + args.video)
    if args.image:
        print("annotated still        -> " + args.image)
    return 0


def _iter_frames(cap, fps, stride):
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % max(1, stride) == 0:
            yield i / fps, frame
        i += 1
    cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
