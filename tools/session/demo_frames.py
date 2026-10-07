"""Render the live overlay to PNG files - no camera needed.

Uses the synthetic skeletons from the self-test, so the pictures show exactly
what the app draws on a webcam frame.  Handy for the report, the presentation
and for showing the guide the feedback behaviour without setting up a mat.

    python tools/session/demo_frames.py            -> data/demo/*.png
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from yoga.reference import ideal_pose as make_tree_pose         # noqa: E402
from yoga.asanas import VRIKSHASANA                             # noqa: E402
from yoga.evaluator import compute_features, evaluate           # noqa: E402
from yoga.feedback import CueEngine                             # noqa: E402
from yoga.overlay import draw_hud, draw_skeleton                # noqa: E402
from yoga.state_machine import PoseStateMachine, State          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "demo")

SCENES = [
    ("01_correct_hold", {}, State.HOLDING, 12.4),
    ("02_slouched_spine", dict(spine_tilt=22.0), State.SETUP, 0.0),
    ("03_bent_standing_leg", dict(standing_knee_bend=35.0), State.SETUP, 0.0),
    ("04_foot_too_low", dict(foot_ratio=0.22), State.SETUP, 0.0),
    ("05_knee_not_opened", dict(thigh_open=15.0), State.SETUP, 0.0),
    ("06_hands_at_heart", dict(hands_at_chest=True), State.HOLDING, 6.2),
    ("07_hold_complete", {}, State.COMPLETE, 20.0),
]


def studio_background(w: int, h: int) -> np.ndarray:
    """A plain gradient stands in for the webcam image."""
    frame = np.zeros((h, w, 3), np.uint8)
    top, bottom = np.array([64, 52, 44]), np.array([24, 20, 18])
    for y in range(h):
        frame[y, :] = bottom + (top - bottom) * (1.0 - y / h)
    cv2.line(frame, (0, int(h * 0.93)), (w, int(h * 0.93)), (70, 62, 55), 3)
    return frame


def render(name: str, kwargs: dict, state: State, elapsed: float) -> str:
    pose = make_tree_pose(**kwargs)
    ev = evaluate(VRIKSHASANA, compute_features(pose))

    sm = PoseStateMachine(hold_target_s=VRIKSHASANA.hold_target_s)
    sm.state = state
    sm.elapsed = elapsed
    sm.progress = min(1.0, elapsed / VRIKSHASANA.hold_target_s)
    sm.stats.best_score = max(ev.score, 92.0)
    if state is State.COMPLETE:
        from yoga.state_machine import Hold
        sm.stats.holds.append(Hold(0.0, 20.0, 91.2, 96.0, True))

    cue = CueEngine(persist_s=0.0, global_cooldown_s=0.0).update(1.0, ev, State.SETUP)
    banner = cue.text if cue else "Good. Hold it."
    if state is State.COMPLETE:
        banner = "Well held. Release the pose slowly."

    frame = studio_background(pose.width, pose.height)
    draw_skeleton(frame, pose, ev)
    draw_hud(frame, ev, sm, banner, fps=28.6, hint=VRIKSHASANA.setup_hint)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    cv2.imwrite(path, frame)
    return path


def main() -> int:
    print("rendering overlay demo frames")
    for name, kwargs, state, elapsed in SCENES:
        path = render(name, kwargs, state, elapsed)
        pose = make_tree_pose(**kwargs)
        ev = evaluate(VRIKSHASANA, compute_features(pose))
        fault = ev.primary.check.label if ev.primary else "none"
        print(f"  {name:<24} score {ev.score:5.1f}%  primary fault: {fault}")
        print(f"  {'':<24} -> {path}")
    print(f"\n{len(SCENES)} frames written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
