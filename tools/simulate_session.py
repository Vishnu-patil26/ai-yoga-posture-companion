"""Drive a whole practice session through the pipeline without a camera.

Feeds synthetic skeletons (the ones from the self-test) through the real
evaluator, state machine, cue engine and SQLite logger, following a scripted
90-second practice: a sloppy start, a correction, a full 20-second hold, a
failed second attempt, then a clean one.

It exists so the analytics half of the project can be demonstrated and marked
on a laptop with no webcam, and so the log/report code is exercised on data
whose right answer is known.

    python tools/simulate_session.py
    python tools/report.py
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoga.reference import ideal_pose as make_tree_pose         # noqa: E402
from yoga.asanas import VRIKSHASANA                             # noqa: E402
from yoga.evaluator import compute_features, evaluate           # noqa: E402
from yoga.feedback import CueEngine                             # noqa: E402
from yoga.phrasing import spoken_phrase                         # noqa: E402
from yoga.filters import RollingMean                            # noqa: E402
from yoga.state_machine import PoseStateMachine, State          # noqa: E402
from yoga.storage import SessionLog                             # noqa: E402

FPS = 30.0

# (until_s, label, pose kwargs or None for "no body in frame")
SCRIPT = [
    (5.0, "walking into frame", None),
    (13.0, "first attempt - slouched, foot too low", dict(spine_tilt=19.0, foot_ratio=0.30)),
    (19.0, "responding to the cues", dict(spine_tilt=9.0, foot_ratio=0.55, thigh_open=45.0)),
    (44.0, "settled - full hold", dict()),
    (49.0, "releasing the pose", dict(foot_ratio=0.05, arm_tilt=172.0, thigh_open=8.0)),
    (58.0, "second attempt - standing knee bent", dict(standing_knee_bend=28.0)),
    (85.0, "corrected - second hold", dict(spine_tilt=4.0)),
    (90.0, "finished", dict(foot_ratio=0.05, arm_tilt=172.0, thigh_open=8.0)),
]


def scripted(t: float):
    for until, label, kwargs in SCRIPT:
        if t < until:
            return label, kwargs
    return SCRIPT[-1][1], SCRIPT[-1][2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Simulated practice session")
    ap.add_argument("--no-db", action="store_true", help="do not write to the practice log")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args(argv)

    rng = random.Random(20260827)          # fixed seed: the run is reproducible
    asana = VRIKSHASANA
    sm = PoseStateMachine(hold_target_s=asana.hold_target_s,
                          enter_score=asana.enter_score, exit_score=asana.exit_score)
    cues = CueEngine(phrase_fn=spoken_phrase)
    smooth = RollingMean(window=5)
    log = None if args.no_db else SessionLog(asana.key, source="simulated")

    frames = int(90.0 * FPS)
    last_label = ""
    last_sample = -1.0

    print("simulated practice session - " + asana.sanskrit)
    print("")

    for i in range(frames):
        t = i / FPS
        label, kwargs = scripted(t)
        if label != last_label and not args.quiet:
            print(f"[{t:6.1f}s]  -- {label} --")
            last_label = label

        if kwargs is None:
            smooth.reset()
            state = sm.update(t, 0.0, False)
            continue

        # A held pose is never perfectly still: add a slow sway plus noise.
        sway = 2.2 * math.sin(t * 1.7) + rng.gauss(0.0, 0.9)
        pose = make_tree_pose(**{**kwargs,
                                 "spine_tilt": kwargs.get("spine_tilt", 0.0) + sway,
                                 "thigh_open": kwargs.get("thigh_open", 62.0) + sway * 0.8})
        ev = evaluate(asana, compute_features(pose))
        score = smooth(ev.score)
        ev.score = score
        state = sm.update(t, score, ev.usable)

        cue = cues.update(t, ev, state)
        if cue is not None:
            if not args.quiet:
                print(f"[{t:6.1f}s]  {cue.level:<6} {cue.text}")
            if log:
                log.log_cue(t, cue.key, cue.level, cue.text)
        if log and ev.usable and t - last_sample >= 0.5:
            last_sample = t
            log.log_joints(t, ev.results)
        if sm.just_completed and not args.quiet:
            h = sm.stats.holds[-1]
            print(f"[{t:6.1f}s]  HOLD COMPLETE - {h.duration_s:.1f}s at {h.avg_score:.1f}% average")

    st = sm.stats
    if log:
        log.finish(st, frames, FPS)

    print("")
    print("---------------- simulated session summary ----------------")
    print(f"holds attempted    : {len(st.holds)}")
    print(f"holds completed    : {st.completed}")
    print(f"longest hold       : {st.longest_s:.1f}s")
    print(f"total time in pose : {st.total_hold_s:.1f}s")
    print(f"best alignment     : {st.best_score:.1f}%")
    if log:
        print(f"logged to          : {log.path} (session id {log.session_id})")
        print("run 'python tools/report.py' to see the practice history")
    print("-----------------------------------------------------------")

    ok = st.completed >= 2 and st.longest_s >= asana.hold_target_s - 0.5
    print("SCRIPT CHECK: " + ("PASS" if ok else "FAIL")
          + f"  (expected 2 completed holds, got {st.completed})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
