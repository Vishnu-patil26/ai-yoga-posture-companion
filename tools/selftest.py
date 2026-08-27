"""Offline verification of the scoring engine - no camera, no video needed.

Builds synthetic skeletons with *known* joint geometry, pushes them through the
exact same feature/scoring/state-machine code the live app uses, and checks the
numbers that come back.  This is what makes the marking objective: the claim
"the app measures the spine angle to within a degree" is checked here rather
than asserted.

    python tools/selftest.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoga import landmarks as LM                                    # noqa: E402
from yoga.asanas import VRIKSHASANA                                 # noqa: E402
from yoga.evaluator import compute_features, evaluate               # noqa: E402
from yoga.feedback import CueEngine                                 # noqa: E402
from yoga.reference import ideal_pose as make_tree_pose             # noqa: E402
from yoga.state_machine import PoseStateMachine, State              # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f"   {detail}"
    print(line)
    if not condition:
        FAILURES.append(name)


def approx(name: str, got: float, want: float, tol: float) -> None:
    check(name, abs(got - want) <= tol, f"got {got:.2f}, want {want:.2f} +/- {tol}")


# ------------------------------------------------------------------- test cases
def test_geometry() -> None:
    print("\n1. geometry - do the measured angles match the skeleton we built?")
    f = compute_features(make_tree_pose())
    approx("spine measured upright", f["spine_tilt"], 0.0, 0.5)
    approx("hips measured level", f["hip_level"], 0.0, 0.5)
    approx("shoulders measured level", f["shoulder_level"], 0.0, 0.5)
    approx("standing leg measured straight", f["standing_knee"], 178.0, 1.0)
    approx("lifted thigh opening recovered", f["folded_thigh_open"], 62.0, 1.5)
    approx("foot height ratio recovered", f["foot_height_ratio"], 0.70, 0.02)
    check("standing leg identified", f["standing_side"] == "right",
          f"got {f['standing_side']}")

    f2 = compute_features(make_tree_pose(spine_tilt=14.0, hip_tilt=9.0, standing="left"))
    approx("spine lean of 14 deg recovered", f2["spine_tilt"], 14.0, 0.6)
    approx("hip tilt of 9 deg recovered", f2["hip_level"], 9.0, 0.6)
    check("standing leg identified (left)", f2["standing_side"] == "left",
          f"got {f2['standing_side']}")


def test_scoring() -> None:
    print("\n2. scoring - a good pose scores high, a specific fault is named")
    good = evaluate(VRIKSHASANA, compute_features(make_tree_pose()))
    check("correct pose is usable", good.usable)
    check("correct pose scores >= 95", good.score >= 95.0, f"score {good.score:.1f}%")
    check("correct pose has no failing joint", not good.failures,
          f"failing: {[r.check.key for r in good.failures]}")

    cases = [
        ("slouched spine", dict(spine_tilt=22.0), "spine_tilt"),
        ("bent standing leg", dict(standing_knee_bend=35.0), "standing_knee"),
        ("foot too low", dict(foot_ratio=0.22), "foot_height_ratio"),
        ("knee not opened out", dict(thigh_open=15.0), "folded_thigh_open"),
        ("arms out sideways", dict(arm_tilt=88.0), None),
        ("hips dropped", dict(hip_tilt=24.0), "hip_level"),
    ]
    for name, kwargs, expect in cases:
        ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(**kwargs)))
        primary = ev.primary.check.key if ev.primary else "-"
        if expect is None:
            check(f"{name} -> flagged (arm check)",
                  primary.startswith("arm_raise") or primary.startswith("wrist_to_chest"),
                  f"score {ev.score:.1f}%, primary '{primary}'")
        else:
            check(f"{name} -> primary fault is '{expect}'", primary == expect,
                  f"score {ev.score:.1f}%, primary '{primary}'")
        check(f"{name} -> score dropped below the good pose", ev.score < good.score - 3,
              f"{ev.score:.1f}% vs {good.score:.1f}%")

    ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(spine_tilt=9.0)))
    check("a 9 deg lean is flagged but still scores well",
          ev.primary is not None and ev.primary.check.key == "spine_tilt" and ev.score > 85,
          f"score {ev.score:.1f}%")
    ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(spine_tilt=5.0)))
    check("a 5 deg lean is inside tolerance", not ev.failures, f"score {ev.score:.1f}%")

    # The weakest-link blend: one badly wrong joint must be enough to deny the
    # hold, even when every other joint is perfect.
    for name, kwargs in (("slouched spine", dict(spine_tilt=22.0)),
                         ("foot at the ankle", dict(foot_ratio=0.22)),
                         ("standing leg bent", dict(standing_knee_bend=40.0))):
        ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(**kwargs)))
        check(f"one bad joint ({name}) denies the hold",
              ev.score < VRIKSHASANA.enter_score,
              f"score {ev.score:.1f}% vs enter {VRIKSHASANA.enter_score:.0f}%")

    # Both taught arm forms must be accepted.  Clustering the dataset showed
    # the arms split cleanly in two and every subject appeared in both
    # clusters, so scoring against a single arm target would mark one of the
    # two standard forms wrong.
    for name, kwargs, want in (("arms overhead", dict(arm_tilt=8.0), "arms overhead"),
                               ("hands at the heart", dict(hands_at_chest=True),
                                "hands at the heart")):
        ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(**kwargs)))
        check(f"{name} is accepted", ev.score >= 95.0 and not ev.failures,
              f"score {ev.score:.1f}%, variant '{ev.variant_name}'")
        check(f"{name} is recognised as that form", ev.variant_name == want,
              f"got '{ev.variant_name}'")

    # Anything that is neither taught form must be refused.  Arms hanging by
    # the sides is the important case: the arm *angle* alone cannot tell it
    # from hands-at-the-heart (both point down from the shoulder), so without
    # the wrist-to-chest check somebody just standing there scored as a
    # correct Vrikshasana.
    for name, kwargs in (("arms out sideways", dict(arm_tilt=88.0)),
                         ("arms hanging by the sides", dict(arm_tilt=172.0))):
        ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(**kwargs)))
        check(f"{name} is flagged", bool(ev.failures),
              f"score {ev.score:.1f}%, primary "
              f"'{ev.primary.check.key if ev.primary else None}'")
        check(f"{name} does not count as a hold", ev.score < VRIKSHASANA.enter_score,
              f"score {ev.score:.1f}%")

    # ...and the wrist-to-chest measurement is what separates them.
    heart = compute_features(make_tree_pose(hands_at_chest=True))
    sides = compute_features(make_tree_pose(arm_tilt=172.0))
    check("wrist-to-chest separates hands-at-heart from arms-down",
          heart["wrist_to_chest_left"] < 0.4 < sides["wrist_to_chest_left"],
          f"heart {heart['wrist_to_chest_left']:.2f} vs sides "
          f"{sides['wrist_to_chest_left']:.2f}")

    # The pelvis genuinely hikes in this asana - the fitted median is 10 deg
    # off level - so a level-hipped pose must never be corrected.
    ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(hip_tilt=0.0)))
    hips = [r for r in ev.results if r.check.key == "hip_level"]
    check("perfectly level hips are not called a fault",
          hips and hips[0].ok is not False, f"hip_level ok={hips[0].ok if hips else '?'}")


def test_state_machine() -> None:
    print("\n3. hold timer - hysteresis, grace window and completion")
    sm = PoseStateMachine(hold_target_s=10.0, enter_score=78, exit_score=62,
                          enter_stable_s=0.6, grace_s=1.2)
    dt, t = 1 / 30.0, 0.0

    for _ in range(30):                       # 1s of a poor pose
        t += dt
        sm.update(t, 50.0, True)
    check("poor pose does not start the timer", sm.state is State.SETUP)

    t += dt
    sm.update(t, 90.0, True)
    check("one good frame is not enough to start", sm.state is State.SETUP)
    for _ in range(25):                       # ~0.8s more of a good pose
        t += dt
        sm.update(t, 90.0, True)
    check("sustained good pose starts the timer", sm.state is State.HOLDING)

    for _ in range(90):                       # 3s of a clean hold
        t += dt
        sm.update(t, 88.0, True)

    start_elapsed = sm.elapsed
    for _ in range(20):                       # 0.67s wobble, inside the grace window
        t += dt
        sm.update(t, 55.0, True)
    check("a short wobble does not end the hold", sm.state is State.HOLDING)
    check("timer kept running through the wobble", sm.elapsed > start_elapsed,
          f"{sm.elapsed:.2f}s")

    for _ in range(60):                       # 2s below the exit score
        t += dt
        sm.update(t, 55.0, True)
    check("a real loss of the pose ends the hold", sm.state is State.SETUP)
    check("the abandoned attempt was logged", len(sm.stats.holds) == 1
          and not sm.stats.holds[0].completed)

    for _ in range(int(12.0 / dt)):           # 12s of a clean hold
        t += dt
        sm.update(t, 88.0, True)
    check("a full hold completes", sm.state is State.COMPLETE)
    check("completed hold recorded", sm.stats.completed == 1,
          f"holds={len(sm.stats.holds)} completed={sm.stats.completed}")
    approx("completed hold duration", sm.stats.holds[-1].duration_s, 10.0, 0.2)

    for _ in range(60):
        t += dt
        sm.update(t, 30.0, True)
    check("re-arms after the practitioner releases", sm.state is State.SETUP)


def test_cue_engine() -> None:
    print("\n4. cue engine - persistence, cooldown and escalation")
    ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(spine_tilt=24.0)))
    cues = CueEngine(persist_s=1.0, cue_cooldown_s=7.0, global_cooldown_s=3.0, firm_after=2)

    said = []
    t = 0.0
    for i in range(int(40 / 0.1)):
        t = i * 0.1
        c = cues.update(t, ev, State.SETUP)
        if c is not None:
            said.append((round(t, 1), c.level, c.text))

    check("nothing is spoken in the first second", all(s[0] >= 1.0 for s in said),
          f"first cue at {said[0][0] if said else 'never'}s")
    check("the fault was eventually spoken", len(said) >= 2, f"{len(said)} cues in 40s")
    check("cues are not repeated faster than the cooldown",
          all(said[i + 1][0] - said[i][0] >= 6.9 for i in range(len(said) - 1)),
          str([s[0] for s in said]))
    check("repeating the same fault escalates to a firm cue",
          any(s[1] == "firm" for s in said),
          str([s[1] for s in said]))
    check("the cue names the right correction",
          "chest" in said[0][2].lower() or "spine" in said[0][2].lower(),
          said[0][2] if said else "")

    good = evaluate(VRIKSHASANA, compute_features(make_tree_pose()))
    quiet = CueEngine()
    spoken = [quiet.update(i * 0.1, good, State.SETUP) for i in range(200)]
    check("a correct pose is not nagged", not any(s for s in spoken))


def test_visibility_gate() -> None:
    print("\n5. visibility - grade what we can see, refuse to invent the rest")

    # The single most characteristic joint of Vrikshasana is half-occluded by
    # definition: the lifted foot is pressed into the standing thigh.  Over 120
    # tree-pose photographs its measured median visibility is 0.50.  Losing
    # confidence in it must NOT stop the pose being graded.
    pose = make_tree_pose()
    feats = compute_features(pose)
    folded = feats["folded_side"]
    pose.vis[LM.L_ANKLE if folded == "left" else LM.R_ANKLE] = 0.30
    ev = evaluate(VRIKSHASANA, compute_features(pose))
    check("a half-occluded lifted foot still gets graded", ev.usable, ev.reason)
    check("and still scores well", ev.score >= 95.0, f"score {ev.score:.1f}%")

    # Below the floor the affected checks drop out, but the rest stand.
    pose = make_tree_pose()
    pose.vis[LM.L_ANKLE if folded == "left" else LM.R_ANKLE] = 0.05
    ev = evaluate(VRIKSHASANA, compute_features(pose))
    ungraded = [r.check.key for r in ev.results if r.ok is None]
    check("an invisible lifted leg ungrades only its own checks",
          set(ungraded) == {"folded_knee", "folded_thigh_open", "foot_height_ratio"},
          str(ungraded))
    check("the rest of the body is still graded", ev.usable, ev.reason)

    # Too much of the body gone - refuse, and say why.
    pose = make_tree_pose()
    for i in (LM.L_ANKLE, LM.R_ANKLE, LM.L_KNEE, LM.R_KNEE):
        pose.vis[i] = 0.05
    ev = evaluate(VRIKSHASANA, compute_features(pose))
    check("both legs invisible - frame is refused", not ev.usable, ev.reason)
    check("refusal explains itself", "see enough" in ev.reason.lower(), ev.reason)

    # Torso gone - the practitioner is not properly in frame at all.
    pose = make_tree_pose()
    for i in (LM.L_SHOULDER, LM.R_SHOULDER, LM.L_HIP, LM.R_HIP):
        pose.vis[i] = 0.2
    ev = evaluate(VRIKSHASANA, compute_features(pose))
    check("torso not visible - told to step back", not ev.usable, ev.reason)
    check("step-back message names the frame", "frame" in ev.reason.lower(), ev.reason)

    # A landmark pushed outside the image is worthless whatever the network says.
    pose = make_tree_pose()
    pose.pts[LM.L_ANKLE][1] = pose.height + 200
    pose.pts[LM.R_ANKLE][1] = pose.height + 200
    ev = evaluate(VRIKSHASANA, compute_features(pose))
    off = [r.check.key for r in ev.results if r.ok is None]
    check("landmarks outside the frame are not graded", len(off) >= 3, str(off))


def main() -> int:
    print("=" * 68)
    print("AI Yoga Companion - offline engine self-test (Vrikshasana)")
    print("=" * 68)
    test_geometry()
    test_scoring()
    test_state_machine()
    test_cue_engine()
    test_visibility_gate()
    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
