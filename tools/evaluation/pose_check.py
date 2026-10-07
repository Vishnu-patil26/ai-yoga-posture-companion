"""End-to-end check of every pose, no camera needed.

Each pose's own reference photograph (assets/gallery/) is pushed through the same
chain the live trainer runs - landmark detection, feature measurement, scoring,
the step-by-step guide, the hold state machine, the cue engine, and the overlay
drawing - for 30 simulated seconds.  A textbook photo of a pose should be
recognised, scored well, walk the guide through its steps and complete a hold;
when it does not, that pose's tuning is the suspect.

Writes one annotated "camera screen" per pose to <out>/<pose>.png and prints a table.

    python tools/evaluation/pose_check.py [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from yoga import asanas as asana_lib                              # noqa: E402
from yoga import phrasing, posespecs, routines                    # noqa: E402
from yoga.coach import JOURNEY                                    # noqa: E402
from yoga.evaluator import compute_features, evaluate             # noqa: E402
from yoga.feedback import CueEngine                               # noqa: E402
from yoga.guide import StepGuide                                  # noqa: E402
from yoga.landmarks import PoseTracker                            # noqa: E402
from yoga.overlay import (draw_framing_panel, draw_hud, draw_pose_guide,   # noqa: E402
                          draw_skeleton, draw_step_panel)
from yoga.state_machine import PoseStateMachine, State           # noqa: E402

W, H, DT, SECONDS = 1280, 720, 0.1, 30.0


def canvas_from(photo: np.ndarray) -> np.ndarray:
    """Letterbox the photo into a webcam-shaped frame, as a person would appear."""
    s = min(W / photo.shape[1], H / photo.shape[0])
    im = cv2.resize(photo, (int(photo.shape[1] * s), int(photo.shape[0] * s)))
    frame = np.full((H, W, 3), (60, 60, 60), np.uint8)
    y, x = (H - im.shape[0]) // 2, (W - im.shape[1]) // 2
    frame[y:y + im.shape[0], x:x + im.shape[1]] = im
    return frame


def check_pose(tracker, key: str, out_dir: str) -> dict:
    photo = cv2.imread(os.path.join(HERE, "assets", "gallery", key + ".jpg"))
    frame0 = canvas_from(photo)
    pose = tracker.process(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
    row = {"pose": key, "detected": pose is not None}
    if pose is None:
        return row
    feats = compute_features(pose)
    asana = asana_lib.LIBRARY[key]
    sm = PoseStateMachine(hold_target_s=asana.hold_target_s, enter_score=asana.enter_score,
                          exit_score=asana.exit_score)
    cues = CueEngine(state_cues=True, phrase_fn=phrasing.spoken_phrase)
    spec = posespecs.STEPS.get(key)
    guide = StepGuide(asana, spec) if spec else None
    spoken, completed, guide_done_at, steps_at = [], 0, None, []
    ev = None
    t = 0.0
    while t < SECONDS:
        ev = evaluate(asana, feats)
        state = sm.update(t, ev.score if ev.usable else 0.0, ev.usable)
        if guide is not None and not guide.finished:
            before = guide.index
            said = guide.update(t, ev)
            if said and guide.index != before:
                steps_at.append(round(t, 1))
            if guide.finished:
                guide_done_at = round(t, 1)
        elif ev.usable:
            c = cues.update(t, ev, state, speaker_busy=False)
            if c is not None:
                spoken.append(c.text)
        if sm.just_completed:
            completed += 1
        t += DT
    # draw the camera screen exactly as the trainer would at the end of the run
    frame = frame0.copy()
    draw_skeleton(frame, pose, ev)
    draw_hud(frame, ev, sm, spoken[-1] if spoken else "", 28.0, hint=asana.setup_hint,
             right_margin=282)
    draw_framing_panel(frame, ev.features.get("framing"))
    draw_step_panel(frame, JOURNEY[-2:], 0)       # what the app shows for a non-Tree pose
    info = routines.POSES[key]
    draw_pose_guide(frame, canvas_from(photo)[:, :], info.name, info.sanskrit,
                    tuple(s.text for s in guide.steps) if guide else info.steps,
                    current=guide.index if guide else None, done=bool(guide and guide.finished))
    cv2.imwrite(os.path.join(out_dir, key + ".png"), frame)
    row.update(usable=ev.usable, score=round(ev.score, 1), variant=ev.variant_name or "-",
               reason=ev.reason, fails=[(r.check.key, r.check.cue_for(r.deviation))
                                        for r in ev.failures[:2]],
               guide=("-" if guide is None else f"{len(steps_at)}/{len(guide.steps)}"),
               guide_done_at=guide_done_at, hold=completed > 0, state=state.name,
               enter=asana.enter_score, cues=spoken[:2])
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results", "pose_check"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    tracker = PoseTracker(model="full", running_mode="image", smooth=False)
    rows = [check_pose(tracker, k, args.out) for k in routines.POSES]
    tracker.close()
    print(f"\n{'pose':15s}{'seen':>5s}{'usable':>7s}{'score':>7s}{'enter':>6s}{'variant':>10s}"
          f"{'guide':>7s}{'hold':>6s}  weakest checks / reason")
    print("-" * 110)
    for r in rows:
        if not r["detected"]:
            print(f"{r['pose']:15s}{'NO':>5s}")
            continue
        weak = "; ".join(f"{k}: {c}" for k, c in r["fails"]) or r["reason"] or "-"
        print(f"{r['pose']:15s}{'yes':>5s}{str(r['usable'])[0]:>7s}{r['score']:>7.1f}{r['enter']:>6.0f}"
              f"{r['variant']:>10s}{r['guide']:>7s}{('YES' if r['hold'] else 'no'):>6s}  {weak[:70]}")
    bad = [r["pose"] for r in rows if not r["detected"] or not r.get("hold")]
    print(f"\nreference photos that did NOT complete a hold: {bad or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
