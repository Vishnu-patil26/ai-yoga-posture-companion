"""Offline verification of the scoring engine - no camera, no video needed.

Builds synthetic skeletons with *known* joint geometry, pushes them through the
exact same feature/scoring/state-machine code the live app uses, and checks the
numbers that come back.  This is what makes the marking objective: the claim
"the app measures the spine angle to within a degree" is checked here rather
than asserted.

    python tests/selftest.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoga import landmarks as LM                                    # noqa: E402
from yoga.asanas import VRIKSHASANA                                 # noqa: E402
from yoga.evaluator import compute_features, evaluate               # noqa: E402
from yoga.feedback import CueEngine, Speaker, estimate_speech_seconds  # noqa: E402
from yoga import phrasing                                           # noqa: E402
from yoga import overlay                                            # noqa: E402
from yoga.coach import JOURNEY, SEQUENCE, Coach                     # noqa: E402
from yoga.reference import STEP_FIGURES, step_figure                # noqa: E402
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

    # 60 s, not 40: repeats now back off (7 s, then 14, then 28), so the third
    # telling - the one that escalates to a firm cue - lands around 43 s.
    said = []
    t = 0.0
    for i in range(int(60 / 0.1)):
        t = i * 0.1
        c = cues.update(t, ev, State.SETUP)
        if c is not None:
            said.append((round(t, 1), c.level, c.text))

    check("nothing is spoken in the first second", all(s[0] >= 1.0 for s in said),
          f"first cue at {said[0][0] if said else 'never'}s")
    check("the fault was eventually spoken", len(said) >= 2, f"{len(said)} cues in 60s")
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


def test_phrasing() -> None:
    print("\n6. body-relative phrasing - words, never a raw number")
    import re
    number_pattern = re.compile(r"-?\d+\.?\d*\s*(deg|degrees|%)|\(\+|\(-")

    cases = [
        ("slouched spine", dict(spine_tilt=22.0)),
        ("foot at the ankle", dict(foot_ratio=0.22)),
        ("standing leg bent", dict(standing_knee_bend=30.0)),
        ("hips hiked", dict(hip_tilt=24.0)),
        ("arms by the sides", dict(arm_tilt=172.0)),
    ]
    for name, kwargs in cases:
        ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(**kwargs)))
        for res in ev.failures:
            table_cell = phrasing.short_words(res, ev.features)
            spoken = phrasing.spoken_phrase(res, "soft", ev.features)
            check(f"{name}: '{res.check.key}' table cell has no raw number",
                  not number_pattern.search(table_cell), repr(table_cell))
            check(f"{name}: '{res.check.key}' spoken line has no raw number",
                  not number_pattern.search(spoken), repr(spoken))
            check(f"{name}: '{res.check.key}' table cell fits the column (<=18 chars)",
                  len(table_cell) <= 18, f"{len(table_cell)} chars: {table_cell!r}")

    # A correct pose must never be told it is wrong, in either wording.
    good = evaluate(VRIKSHASANA, compute_features(make_tree_pose()))
    for res in good.results:
        if res.ok:
            check(f"correct '{res.check.key}' reads as correct",
                  phrasing.short_words(res, good.features) == "correct",
                  phrasing.short_words(res, good.features))

    # An ungraded check must say so, not silently claim "correct".
    pose = make_tree_pose()
    folded = compute_features(pose)["folded_side"]
    idx = LM.L_ANKLE if folded == "left" else LM.R_ANKLE
    pose.vis[idx] = 0.05
    ev = evaluate(VRIKSHASANA, compute_features(pose))
    ungraded = [r for r in ev.results if r.ok is None]
    check("an ungraded joint is flagged 'can't see', not scored as correct",
          ungraded and all(phrasing.short_words(r, ev.features) == "can't see" for r in ungraded),
          str([(r.check.key, phrasing.short_words(r, ev.features)) for r in ungraded]))

    # The hand-width analogy must at least point the right direction: a
    # bigger deviation must never report FEWER hand-widths than a smaller one.
    small = evaluate(VRIKSHASANA, compute_features(make_tree_pose(foot_ratio=0.55)))
    big = evaluate(VRIKSHASANA, compute_features(make_tree_pose(foot_ratio=0.22)))
    r_small = next(r for r in small.results if r.check.key == "foot_height_ratio")
    r_big = next(r for r in big.results if r.check.key == "foot_height_ratio")
    hw_small = phrasing.handwidths(r_small, small.features)
    hw_big = phrasing.handwidths(r_big, big.features)
    check("a bigger miss reports more hand-widths than a smaller one",
          hw_small is not None and hw_big is not None and hw_big > hw_small,
          f"small={hw_small}, big={hw_big}")


def test_skeleton_colouring() -> None:
    print("\n7. skeleton colouring - every graded bone, worst verdict wins")

    good = evaluate(VRIKSHASANA, compute_features(make_tree_pose()))
    status = overlay._segment_status(good)
    check("a fully correct pose colours every graded bone green",
          bool(status) and all(s == "ok" for s in status.values()), str(status))
    check("every check with a segment mapping actually gets drawn",
          set(overlay._CHECK_SEGMENTS) <= {c.key for c in VRIKSHASANA.checks} | {"wrist_to_chest_left", "wrist_to_chest_right"},
          str(set(overlay._CHECK_SEGMENTS)))

    bad = evaluate(VRIKSHASANA, compute_features(make_tree_pose(standing_knee_bend=35.0)))
    status = overlay._segment_status(bad)
    check("a bent standing leg turns its own bones red",
          status.get("standing_thigh") == "fail" and status.get("standing_shin") == "fail",
          str(status))
    check("an unrelated correct bone stays green",
          status.get("shoulder_line") == "ok", str(status))

    # folded_knee and foot_height_ratio both touch the folded shin - if either
    # fails, the shared bone must show the failure, not silently average it away.
    one_bad = evaluate(VRIKSHASANA, compute_features(make_tree_pose(foot_ratio=0.20)))
    status = overlay._segment_status(one_bad)
    check("a shared bone shows red when EITHER of its checks fails",
          status.get("folded_shin") == "fail", str(status))

    pose = make_tree_pose()
    folded = compute_features(pose)["folded_side"]
    for idx in ((LM.L_HIP, LM.L_KNEE, LM.L_ANKLE) if folded == "left"
               else (LM.R_HIP, LM.R_KNEE, LM.R_ANKLE)):
        pose.vis[idx] = 0.05
    ev = evaluate(VRIKSHASANA, compute_features(pose))
    status = overlay._segment_status(ev)
    check("an invisible leg is coloured 'ungraded', not silently green",
          status.get("folded_thigh") == "ungraded" and status.get("folded_shin") == "ungraded",
          str(status))
    check("fail still outranks ungraded when both apply to the same bone",
          overlay._STATUS_RANK["fail"] > overlay._STATUS_RANK["ungraded"] >
          overlay._STATUS_RANK["ok"])


def test_rest_timer() -> None:
    print("\n8. rest timer - a visible, countdown gap between hold attempts")

    sm = PoseStateMachine(hold_target_s=3.0, enter_score=78, exit_score=62,
                          enter_stable_s=0.3, grace_s=0.6)
    coach = Coach(hold_target_s=3.0, rest_target_s=2.0)
    good = evaluate(VRIKSHASANA, compute_features(make_tree_pose()))
    dt, t = 1 / 30.0, 0.0

    coach.update(t, None, State.WAITING)          # the opening greeting

    # Completing the guided practice is what opens the rest window - the
    # sequence owns the session while it runs, so a good alignment score no
    # longer short-circuits it.
    t += dt
    coach._on_complete(t)
    check("finishing the practice starts the rest window",
          coach.resting and coach.rest_remaining > 0.0,
          f"resting={coach.resting}, remaining={coach.rest_remaining:.2f}")
    check("the rest window is shown, not just spoken",
          coach.step_key == "rest", coach.step_key)

    # Tick through the rest window on a body that has stepped out of frame -
    # exactly what actually happens after lowering the foot - and confirm the
    # countdown only ever counts down, and a "1" mark gets spoken before it ends.
    seen_one = False
    prev_remaining = coach.rest_remaining
    while coach.resting:
        t += dt
        sm.update(t, 0.0, False)
        said = coach.update(t, None, sm.state)
        check("rest countdown never goes negative", coach.rest_remaining >= 0.0,
              coach.rest_remaining)
        check("rest countdown only counts down", coach.rest_remaining <= prev_remaining + 1e-6,
              f"{coach.rest_remaining} after {prev_remaining}")
        prev_remaining = coach.rest_remaining
        if said is not None and said.text == "1":
            seen_one = True
    check("the last second of rest is spoken", seen_one)
    check("resting ends within the configured window", t < 100.0)  # sanity, not a real bound

    # Once rest ends, guiding resumes rather than staying silent forever.
    said = None
    for _ in range(400):                          # up to ~13s of frames
        t += dt
        sm.update(t, 0.0, False)
        said = coach.update(t, None, sm.state)
        if said is not None:
            break
    check("guiding resumes once the rest window ends", said is not None,
          "coach never spoke again after the rest window")

    # Releasing and jumping straight back into a good hold - all inside the
    # rest window - cancels the wait early.  A high score alone does not do
    # this: COMPLETE only re-arms once the practitioner actually releases
    # (see the "re-arms after the practitioner releases" check above), so a
    # real early return has to go COMPLETE -> released -> HOLDING again.
    coach2 = Coach(hold_target_s=3.0, rest_target_s=5.0)
    coach2.update(0.0, None, State.WAITING)
    coach2._on_complete(0.1)
    check("rest window opened after finishing the practice", coach2.resting)

    t2 = 0.1
    for _ in range(30):                            # a second of resting
        t2 += dt
        coach2.update(t2, good, State.SETUP)
    check("still resting a moment later", coach2.resting)

    # Going straight back into a good hold cancels the wait early.
    coach2.update(t2 + dt, good, State.HOLDING, elapsed=0.5)
    check("re-entering a hold during rest ends the wait early",
          not coach2.resting, f"resting={coach2.resting}")


def test_speech_pacing() -> None:
    print("\n9. speech pacing - a new line never starts over one still playing")

    short = estimate_speech_seconds("Hold it.")
    long_sentence = estimate_speech_seconds(
        "Lift your chest and stack your spine over your hips - about half a hand's width.")
    check("a longer sentence is estimated to take longer to say",
          long_sentence > short, f"short={short:.2f}s long={long_sentence:.2f}s")
    check("even one word takes a non-zero, sane amount of time",
          0.2 < estimate_speech_seconds("Hold.") < 2.0,
          f"{estimate_speech_seconds('Hold.'):.2f}s")
    check("speaking rate changes the estimate the right way",
          estimate_speech_seconds("test words here", rate_wpm=80)
          > estimate_speech_seconds("test words here", rate_wpm=300))

    # Speaker(enabled=False) never starts a thread or touches a real TTS
    # engine - this must stay true, or the offline suite (and the Docker
    # build, which runs this file as its own gate) would depend on audio
    # hardware being present.
    sp = Speaker(enabled=False)
    check("a disabled speaker starts no worker thread", sp._thread is None)

    sp.say("This should be silently skipped.", 10.0)
    check("a disabled speaker never claims to be busy",
          not sp.busy(10.0), f"free_at={sp.free_at}")

    # The pacing contract `say(text, t)` / `busy(t)` is pure arithmetic and is
    # exercised directly here without a real engine behind it.
    text = "Lift your chest and stack your spine over your hips - about half a hand's width."
    expected_duration = sp.estimate_seconds(text)
    sp.free_at = 5.0 + expected_duration            # what say(text, 5.0) would set
    check("busy() is true for the whole estimated duration of the line",
          sp.busy(5.0) and sp.busy(5.0 + expected_duration - 0.05),
          f"duration={expected_duration:.2f}s")
    check("busy() clears once the estimated duration has elapsed",
          not sp.busy(5.0 + expected_duration + 0.05))

    short_text = "Hold it."
    check("a short line frees the speaker up sooner than a long one",
          sp.estimate_seconds(short_text) < sp.estimate_seconds(text))

    # A line offered while another is playing must WAIT, not vanish.  With a
    # depth-1 queue the app logged and displayed cues it never actually spoke -
    # and the ones landing on a busy speaker were the important ones ("hold
    # it", the first correction), so the practitioner heard the introduction
    # and then silence through the part that mattered.
    import queue as _q2

    class _QueueOnly(Speaker):
        def __init__(self):
            super().__init__(enabled=False)
            self.enabled = True                  # queue, but never start a thread
            self._q = _q2.Queue(maxsize=2)

    qs = _QueueOnly()
    qs.say("first line, currently playing", 0.0)
    busy_until = qs.free_at
    qs.say("second line, offered mid-sentence", 0.5)
    check("a line offered while the speaker is busy is queued, not dropped",
          qs._q.qsize() == 2, f"queue holds {qs._q.qsize()}")
    check("the queued line is timed to start after the one before it",
          qs.free_at > busy_until,
          f"free_at {qs.free_at:.2f} vs previous end {busy_until:.2f}")

    # And a correction must not even be raised when it cannot be delivered -
    # otherwise its cooldown starts for a cue nobody heard.
    ev_bad = evaluate(VRIKSHASANA, compute_features(make_tree_pose(spine_tilt=24.0)))
    busy_engine = CueEngine(phrase_fn=phrasing.spoken_phrase, state_cues=False)
    for i in range(60):
        busy_engine.update(i / 10.0, ev_bad, State.HOLDING, speaker_busy=True)
    check("no correction is raised while the voice is busy",
          not busy_engine.history, f"{len(busy_engine.history)} raised")
    free_engine = CueEngine(phrase_fn=phrasing.spoken_phrase, state_cues=False)
    raised = [free_engine.update(i / 10.0, ev_bad, State.HOLDING, speaker_busy=False)
              for i in range(60)]
    check("the same fault IS raised once the voice is free",
          any(r is not None for r in raised))

    # A clock restart must clear the stamp, or cues queued on the new
    # timeline read as "still speaking" and are silently swallowed - which is
    # exactly what happened to the first cues after the walkthrough.
    sp.free_at = 99.0
    sp.reset_timeline()
    check("reset_timeline clears a stale busy-until stamp",
          not sp.busy(0.0), f"free_at={sp.free_at}")

    # Every queued line must reach the speaking backend exactly once.  A
    # single long-lived pyttsx3 engine speaks only the FIRST line on
    # Windows/SAPI5 and silently no-ops afterwards, so the worker builds a
    # fresh engine per line; this guards the drain loop that does it,
    # without needing audio hardware in the suite.
    import queue as _queue

    spoken: list[str] = []

    class _CountingSpeaker(Speaker):
        def __init__(self):
            super().__init__(enabled=False)      # no real thread, no audio
            self._q = _queue.Queue(maxsize=10)

        def _speak_once(self, _pyttsx3, text, rate, volume):
            spoken.append(text)

    cs = _CountingSpeaker()
    for line in ("first line", "second line", "third line"):
        cs._q.put(line)
    cs._q.put(None)
    cs._run(rate=165, volume=1.0)                # drains the queue, no audio
    check("every queued line reaches the speaking backend, not just the first",
          spoken == ["first line", "second line", "third line"], str(spoken))


def test_journey_steps() -> None:
    print("\n10. the seven-step journey - one source of truth for UI and demo")

    check("the journey is the guided steps plus hold and rest",
          len(JOURNEY) == len(SEQUENCE) + 2, f"{len(JOURNEY)} vs {len(SEQUENCE)}+2")
    check("the journey starts with the guided entry steps, in order",
          [j.key for j in JOURNEY[:len(SEQUENCE)]] == [s.key for s in SEQUENCE],
          str([j.key for j in JOURNEY]))
    check("it ends with hold then rest",
          JOURNEY[-2].key == "hold" and JOURNEY[-1].key == "rest",
          f"{JOURNEY[-2].key}, {JOURNEY[-1].key}")
    check("every stage has a title and something to say",
          all(j.title and j.say for j in JOURNEY))

    # The demo walkthrough draws a figure for every stage; a missing one would
    # silently fall back to the hold figure and show the wrong shape.
    for j in JOURNEY:
        check(f"'{j.key}' has its own reference figure",
              j.key in STEP_FIGURES, f"missing from STEP_FIGURES: {j.key}")
        check(f"'{j.key}' figure actually builds", step_figure(j.key) is not None)

    # journey_index must agree with what the practitioner is actually doing.
    coach = Coach(hold_target_s=3.0, rest_target_s=2.0)
    check("a fresh session reports step 1", coach.journey_index(State.WAITING) == 0,
          coach.journey_index(State.WAITING))

    coach.index = 2
    check("mid-guidance reports the matching entry step",
          coach.journey_index(State.SETUP) == 2, coach.journey_index(State.SETUP))

    check("holding reports the hold stage, whatever the guided index says",
          coach.journey_index(State.HOLDING) == len(JOURNEY) - 2,
          coach.journey_index(State.HOLDING))
    check("COMPLETE still reads as the hold stage, not off the end",
          coach.journey_index(State.COMPLETE) == len(JOURNEY) - 2,
          coach.journey_index(State.COMPLETE))

    coach.resting = True
    check("resting reports the rest stage even though state is still COMPLETE",
          coach.journey_index(State.COMPLETE) == len(JOURNEY) - 1,
          coach.journey_index(State.COMPLETE))

    # The index is what indexes JOURNEY for display - it must never go out of
    # range, whatever the guided index has been left at.
    coach.resting = False
    coach.index = 99
    idx = coach.journey_index(State.SETUP)
    check("a runaway guided index cannot index past the journey",
          0 <= idx < len(JOURNEY), f"index={idx}, len={len(JOURNEY)}")


def test_speech_restraint() -> None:
    print("\n11. speech restraint - coaching, not nagging")

    ev = evaluate(VRIKSHASANA, compute_features(make_tree_pose(spine_tilt=24.0)))
    res = ev.failures[0]

    full = phrasing.spoken_phrase(res, "soft", ev.features, terse=False)
    terse = phrasing.spoken_phrase(res, "soft", ev.features, terse=True)
    check("the first telling still says how far off you are",
          any(w in full for w in ("hand", "touch", "little", "bit", "way")), full)
    check("a repeat is shorter than the first telling",
          len(terse) < len(full), f"{terse!r} vs {full!r}")
    check("a spoken correction is short enough to say in one breath",
          estimate_speech_seconds(full) < 4.5,
          f"{estimate_speech_seconds(full):.1f}s: {full!r}")
    check("a repeat is very short", estimate_speech_seconds(terse) < 2.0,
          f"{estimate_speech_seconds(terse):.1f}s: {terse!r}")
    check("neither wording contains a raw number",
          not any(ch.isdigit() for ch in full + terse), f"{full!r} {terse!r}")

    # Repeating the same fault must back off, not keep the same cadence.
    cues = CueEngine(phrase_fn=phrasing.spoken_phrase)
    said_at = []
    t = 0.0
    while t < 120.0:
        t += 1 / 30.0
        cue = cues.update(t, ev, State.HOLDING)
        if cue is not None and not cue.key.startswith("state."):
            said_at.append(round(t, 1))
    gaps = [said_at[i + 1] - said_at[i] for i in range(len(said_at) - 1)]
    check("the same fault is not repeated at a fixed cadence",
          len(gaps) >= 2 and gaps[-1] > gaps[0] + 1.0,
          f"spoken at {said_at}, gaps {[round(g,1) for g in gaps]}")
    check("a persistent fault is not said endlessly over two minutes",
          len(said_at) <= 6, f"{len(said_at)} corrections in 120s: {said_at}")

    # The talk budget governs corrections only.  state_cues=False mirrors the
    # live app, where the coach owns the "hold it" announcements - otherwise
    # the first call returns that state cue rather than a correction.
    quiet = CueEngine(phrase_fn=phrasing.spoken_phrase, state_cues=False)
    quiet.note_spoken(0.0, 20.0)             # 20s of speech already banked
    for _ in range(40):                      # let the fault persist past the gate
        quiet.update(10.0, ev, State.HOLDING)
    check("a saturated talk budget suppresses further corrections",
          quiet.update(10.0, ev, State.HOLDING) is None,
          f"ratio={quiet.talking_ratio(10.0):.2f}")
    check("the budget frees up again as the window slides",
          quiet.talking_ratio(200.0) < quiet.talk_budget,
          f"ratio at 200s = {quiet.talking_ratio(200.0):.2f}")

    # Countdowns must stay sparse - they interrupt the quietest part of the
    # practice, and the numbers are already on screen the whole time.
    from yoga.coach import COUNTDOWN_AT, REST_SPEAK_AT
    check("the hold countdown speaks at most 3 times",
          len(COUNTDOWN_AT) <= 3, str(COUNTDOWN_AT))
    check("the rest countdown speaks at most 2 times",
          len(REST_SPEAK_AT) <= 2, str(REST_SPEAK_AT))


def test_step_stability() -> None:
    print("\n12. step stability - advance on the timer, not on a flicker")
    import yoga.coach as coach_mod

    good = evaluate(VRIKSHASANA, compute_features(make_tree_pose()))
    noisy_pose = make_tree_pose()
    noisy_pose.pts[LM.L_ANKLE][1] = noisy_pose.height + 300   # feet out of shot
    noisy_pose.pts[LM.R_ANKLE][1] = noisy_pose.height + 300
    dropout = evaluate(VRIKSHASANA, compute_features(noisy_pose))
    check("the dropout frame really is ungradable", not dropout.usable, dropout.reason)

    def run(every: int, frames: int = 200):
        c = Coach()
        c.update(0.0, good, State.SETUP)
        dt, t = 1 / 30.0, 0.0
        collapses, prev, advanced = 0, 0.0, None
        for i in range(frames):
            t += dt
            ev = dropout if (every and i % every == every - 1) else good
            c.update(t, ev, State.SETUP)
            if c.confirm_fraction < prev - 0.01:
                collapses += 1
            prev = c.confirm_fraction
            if advanced is None and c.index == 1:
                advanced = t
        return collapses, advanced

    # Clean signal: the step advances exactly when the dwell completes.
    _, advanced = run(every=0)
    dwell = SEQUENCE[0].dwell
    check("with a clean signal the step advances when the timer completes",
          advanced is not None and abs(advanced - dwell) < 0.15,
          f"advanced at {advanced}, dwell {dwell}")

    # One-in-six frames ungradable - a normal amount of landmark noise.
    collapses, advanced = run(every=6)
    check("a brief dropout does not cancel the confirmation",
          collapses <= 2, f"{collapses} collapses of the confirm bar")
    check("the step still advances through the noise",
          advanced is not None,
          f"advanced at {advanced}" if advanced else "never advanced")

    # A sustained loss *should* restart the guidance - the practitioner really
    # has left, and pretending otherwise would be worse.  Run long enough to
    # actually be past step one first, or this proves nothing.
    c = Coach()
    c.update(0.0, good, State.SETUP)
    for i in range(1, 70):
        c.update(i / 30.0, good, State.SETUP)
    started_at = c.index
    check("the coach really did advance before the dropout test",
          started_at > 0, f"index {started_at}")
    t = 70 / 30.0
    for _ in range(120):                      # 4s of genuine absence
        t += 1 / 30.0
        c.update(t, dropout, State.SETUP)
    check("a sustained loss does restart the guidance",
          c.index == 0 and not c.confirming,
          f"index {started_at} -> {c.index}, confirming={c.confirming}")

    # The confirmation bar belongs to the guided steps only.
    c = Coach()
    c.update(0.0, good, State.SETUP)
    c.finished = True                     # practice done; the hold branch applies
    c.index = len(SEQUENCE)
    c.confirming, c.confirm_fraction = True, 0.7
    c.update(1.0, good, State.HOLDING, elapsed=1.0)
    check("the confirm bar is cleared once the hold begins",
          not c.confirming and c.confirm_fraction == 0.0,
          f"confirming={c.confirming}, fraction={c.confirm_fraction}")

    # Multi-line instructions must not be spoken over each other.
    c = Coach()
    c.update(0.0, good, State.SETUP)
    emitted = sum(1 for i in range(1, 60)
                  if c.update(i / 30.0, good, State.SETUP, speaker_busy=True) is not None)
    check("no instruction line is emitted while the voice is still speaking",
          emitted == 0, f"{emitted} lines emitted while busy")


def test_asana_library() -> None:
    print("\n13. asana library - more than one pose, every target fitted")
    from yoga import asanas as lib

    check("the library holds more than one asana", len(lib.LIBRARY) > 1,
          f"{len(lib.LIBRARY)}: {sorted(lib.LIBRARY)}")
    check("the hand-tuned Vrikshasana is still the one registered for its key",
          lib.LIBRARY["vrikshasana"] is lib.VRIKSHASANA)
    check("its automatically fitted twin is not also registered",
          "vrikshasana_fit" not in lib.LIBRARY, sorted(lib.LIBRARY))

    for key, asana in lib.LIBRARY.items():
        check(f"'{key}' has checks to score against", len(asana.checks) > 0,
              f"{len(asana.checks)} checks")
        check(f"'{key}' has a sane tolerance on every check",
              all(c.tol > 0 and c.zero_at > c.tol for c in asana.checks),
              str([(c.key, c.tol, c.zero_at) for c in asana.checks
                   if not (c.tol > 0 and c.zero_at > c.tol)]))
        check(f"'{key}' declares what it needs to see",
              all(c.needs for c in asana.checks if c.weight > 0))

    # The side-agnostic features must be genuinely mirror-invariant, or a pose
    # done on the other side would not match its own fitted reference.
    left = compute_features(make_tree_pose(standing="left"))
    right = compute_features(make_tree_pose(standing="right"))
    for key in ("knee_bent", "knee_straight", "stance_width"):
        check(f"'{key}' is the same whichever side leads",
              abs(left[key] - right[key]) < 1.0,
              f"left {left[key]:.2f} vs right {right[key]:.2f}")

    # Every fitted asana must actually score a body without raising.
    feats = compute_features(make_tree_pose())
    for key, asana in lib.LIBRARY.items():
        ev = evaluate(asana, feats)
        check(f"'{key}' scores a real body without error",
              0.0 <= ev.score <= 100.0, f"score {ev.score:.1f}")

    # ...and the tree pose must score highest against Vrikshasana, not against
    # one of the poses fitted from a different class.
    scores = {k: evaluate(a, feats).score for k, a in lib.LIBRARY.items()}
    best = max(scores, key=scores.get)
    check("a tree pose scores highest against Vrikshasana",
          best == "vrikshasana",
          ", ".join(f"{k}={v:.0f}" for k, v in sorted(scores.items(),
                                                      key=lambda kv: -kv[1])))


def test_full_practice() -> None:
    print("\n14. the whole practice - settle, up, eyes, count, down, up, down, rest")
    from yoga.coach import _arms_down, _arms_overhead

    def ev(**kw):
        return evaluate(VRIKSHASANA, compute_features(make_tree_pose(**kw)))

    stand = ev(foot_ratio=0.02, thigh_open=3.0, arm_tilt=168.0, standing_knee_bend=1.0)
    tree_heart = ev(hands_at_chest=True)
    tree_up = ev(arm_tilt=8.0)

    # "Arms down" had no detector at all - the old condition only ever asked
    # "are they up OR at the heart", so once they were up nothing could see
    # them come down again, and the sequence stuck there for ever.
    check("arms overhead is recognised", _arms_overhead(tree_up.features))
    check("arms overhead is NOT counted as down",
          not _arms_down(tree_up.features))
    check("hands at the heart is recognised as down",
          _arms_down(tree_heart.features))
    check("arms lowered to the sides is also down",
          _arms_down(compute_features(make_tree_pose(arm_tilt=172.0))))

    def body(step_key):
        if step_key in ("frame", "stand"):
            return stand
        if step_key in ("weight", "foot", "steady", "down1", "down2"):
            return tree_heart
        return tree_up

    coach = Coach(rest_target_s=3.0)
    coach.update(0.0, stand, State.SETUP)
    t, dt = 0.0, 1 / 30.0
    visited, said = [], []
    for _ in range(4000):
        t += dt
        key_before = coach.step_key
        line = coach.update(t, body(key_before), State.SETUP)
        if not visited or visited[-1] != coach.step_key:
            visited.append(coach.step_key)
        if line is not None:
            said.append(line.text)
        if coach.resting:
            break

    for key in ("steady", "hands", "eyes", "count", "down1", "up2", "down2"):
        check(f"the practice reaches '{key}'", key in visited, str(visited))
    check("every stage is reached in order, none skipped",
          [k for k in visited if k in {s.key for s in SEQUENCE}]
          == [s.key for s in SEQUENCE], str(visited))
    check("the practice ends in the rest window", coach.resting)
    check("the arms are asked down, then up, then down again",
          sum(1 for s in said if "down" in s.lower()) >= 2
          and any("up again" in s.lower() for s in said),
          str([s for s in said if "down" in s.lower() or "up" in s.lower()]))
    check("a good alignment score cannot cut the practice short",
          len(visited) > 5, f"only reached {visited}")

    # Each stage must wait for its dwell, so the whole thing cannot race past.
    total_dwell = sum(s.dwell for s in SEQUENCE)
    check("the practice takes at least the sum of its dwell times",
          t >= total_dwell * 0.9, f"{t:.1f}s vs {total_dwell:.1f}s of dwell")


def main() -> int:
    print("=" * 68)
    print("AI Yoga Companion - offline engine self-test (Vrikshasana)")
    print("=" * 68)
    test_geometry()
    test_scoring()
    test_state_machine()
    test_cue_engine()
    test_visibility_gate()
    test_phrasing()
    test_skeleton_colouring()
    test_rest_timer()
    test_speech_pacing()
    test_journey_steps()
    test_speech_restraint()
    test_step_stability()
    test_asana_library()
    test_full_practice()
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
