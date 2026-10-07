"""Offline checks for the routine / profile layer and the launcher screens.

    python tools/test_flow.py        (no camera needed; opens and closes a window)
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoga import asanas as asana_lib                              # noqa: E402
from yoga import profile as prof                                  # noqa: E402
from yoga import routines as rt                                   # noqa: E402

FAILS: list[str] = []
N = 0


def check(cond: bool, label: str) -> None:
    global N
    N += 1
    if not cond:
        FAILS.append(label)
        print(f"  FAIL  {label}")


def base(**kw) -> prof.Profile:
    d = dict(name="Asha", age=30, weight_kg=60, height_cm=165, waist_cm=75, hip_cm=95,
             bp="None", job="sitting", diet="Vegetarian", location="Thane")
    d.update(kw)
    return prof.Profile(**d)


def test_catalogue() -> None:
    check(len(rt.POSES) == 10, "ten poses in the catalogue")
    for r in rt.all_routines().values():
        check(all(k in rt.POSES for k in r.poses), f"{r.key}: every pose exists")
    check(len(rt.LIFESTYLE) == 5 and len(rt.GOALS) == 3, "5 lifestyle + 3 goal routines")
    check({r.key for r in rt.by_position()} == {"pos_standing", "pos_sitting", "pos_floor"},
          "three body-position groups")
    # every scored pose really resolves to a fitted asana, and nothing else claims to
    scored = {k for k, p in rt.POSES.items() if p.is_scored}
    check(scored == set(rt.POSES), f"all ten poses are camera-scored, got {sorted(scored)}")
    check(all(k in asana_lib.LIBRARY for k in scored), "scored poses load from the library")
    check(rt.recommend("sitting") == "desk_sitting"
          and rt.recommend("standing") == "standing_workers"
          and rt.recommend("mixed") == "lumbar_flex", "job -> recommended routine")


def test_profile() -> None:
    check(prof.validate(base()) == [], "a normal profile validates")
    check(len(prof.validate(base(age=0))) == 1, "age is required")
    check(prof.validate(base(waist_cm=None, hip_cm=None, bust_cm=None)) == [],
          "body measurements are optional")
    check(len(prof.validate(base(weight_kg=900))) == 1, "absurd weight rejected")
    check(len(prof.validate(base(name="../x"))) == 1, "unsafe name rejected")
    p = base(weight_kg=60, height_cm=150)
    check(abs(p.bmi - 26.67) < 0.01 and p.bmi_band == "overweight", "BMI maths and band")
    check(abs(base().whr - 75 / 95) < 1e-9, "waist-to-hip ratio")
    with tempfile.TemporaryDirectory() as d:
        old = prof.USERS_DIR
        prof.USERS_DIR = d
        try:
            prof.save(base())
            back = prof.load("Asha")
            check(back == base(), "profile round-trips through disk")
            check(prof.saved_names() == ["Asha"], "saved profiles are listed")
            prof.append_history(base(), [{"pose": "x"}])
            check(prof.saved_names() == ["Asha"], "history file is not listed as a profile")
        finally:
            prof.USERS_DIR = old


def test_personalise() -> None:
    routine = rt.all_routines()["standing_workers"]          # includes Downward Dog
    healthy = prof.personalise(base(), routine)
    check(all(not i.skipped for i in healthy.items), "healthy adult: nothing skipped")
    check(all(i.hold_s == 20 for i in healthy.items), "healthy adult: full holds")
    high = prof.personalise(base(bp="High"), routine)
    dog = next(i for i in high.items if i.pose_key == "adho_mukha")
    check(dog.skipped, "high BP skips the inversion")
    check(all(i.hold_s <= 20 for i in high.active), "high BP never lengthens a hold")
    old = prof.personalise(base(age=65), routine)
    check(next(i for i in old.items if i.pose_key == "adho_mukha").skipped, "60+ skips inversion")
    check(all(i.hold_s < 20 for i in old.active), "60+ shortens holds")
    diab = prof.personalise(base(diabetic=True), routine)
    check(any("glucose" in n for n in diab.notes), "diabetic gets a glucose caution")
    check(any("Diabetic" in d for d in diab.diet), "diabetic diet notes")
    stacked = prof.personalise(base(age=70, bp="High", diabetic=True, weight_kg=110), routine)
    check(all(i.hold_s >= 10 for i in stacked.active), "holds never drop below 10 s")
    check(any("not medical advice" in d for d in healthy.diet), "disclaimer present")
    # gentler only: no profile can ever lengthen a hold beyond the pose default
    for r in rt.all_routines().values():
        pl = prof.personalise(base(age=70, bp="High", diabetic=True), r)
        check(all(i.hold_s <= rt.POSES[i.pose_key].hold_s for i in pl.active),
              f"{r.key}: holds only shrink")


def test_posespecs_and_guide() -> None:
    import ast
    from yoga import posespecs as ps
    from yoga.guide import StepGuide
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "yoga", "evaluator.py"), encoding="utf-8").read()
    real = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant)
            and isinstance(n.value, str)}
    fitted = [k for k in rt.POSES if k != "vrikshasana"]
    check(set(ps.FEATURES) == set(ps.CUES) == set(ps.STEPS) == set(fitted),
          "posespecs covers exactly the nine fitted poses")
    for k in fitted:
        a = asana_lib.LIBRARY[k]
        have = {c.key for c in a.checks}
        check(set(ps.FEATURES[k]) == have, f"{k}: fitted checks == the pose's defining measurements")
        check(set(ps.CUES[k]) <= have, f"{k}: every cue belongs to a measured feature")
        check(all(f in real for f in ps.FEATURES[k]), f"{k}: every feature exists in the evaluator")
        check(3 <= len(ps.STEPS[k]) <= 3, f"{k}: three guided steps")
        check(all(any(key in have for key in keys) for _, keys in ps.STEPS[k]),
              f"{k}: every step checks at least one measured feature")
        # a pose-specific cue, not the generic fallback, for its most heavily weighted feature
        top = max(ps.FEATURES[k], key=ps.FEATURES[k].get)
        c = a.check(top)
        check(bool(c.cue_under or c.cue_over), f"{k}: top feature '{top}' has a directional cue")
    check(asana_lib.LIBRARY["utkatasana"].check("knee_bent").cue_over.startswith("Sit lower"),
          "Chair uses its own wording, not the generic one")

    class _R:                                   # minimal stand-ins for CheckResult / Evaluation
        def __init__(self, key, ok):
            self.check = type("C", (), {"key": key})()
            self.ok = ok

    class _E:
        def __init__(self, ok, usable=True):
            self.results = [_R("stance_width", ok), _R("knee_bent", True)]
            self.usable = usable

    g = StepGuide(asana_lib.LIBRARY["utkatasana"], ps.STEPS["utkatasana"])
    check(g.update(0.0, _E(False)) == ps.STEPS["utkatasana"][0][0], "guide speaks step 1 first")
    check(g.update(2.0, _E(True)) is None and g.index == 0, "needs the min dwell before advancing")
    check(g.update(3.1, _E(True)) is None and g.index == 0, "needs the pose held for the settle time")
    check(g.update(3.4, _E(True)) is not None and g.index == 1, "advances once held and settled")
    g2 = StepGuide(asana_lib.LIBRARY["utkatasana"], ps.STEPS["utkatasana"])
    g2.update(0.0, _E(False))
    check(g2.update(10.0, _E(False)) is None and g2.index == 0, "a wrong pose does not advance early")
    check(g2.update(26.0, _E(False)) is not None and g2.index == 1, "timeout moves on so nobody is trapped")
    g3 = StepGuide(asana_lib.LIBRARY["utkatasana"], ps.STEPS["utkatasana"])
    g3.update(0.0, None)
    for t in (4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 26.0, 52.0, 78.0):
        g3.update(t, None)
    check(g3.finished, "guide finishes after the last step")


def test_screens() -> None:
    try:
        import launcher
        app = launcher.App()
    except Exception as exc:                                  # no display (CI / Docker)
        print(f"  skip  launcher screens ({type(exc).__name__}: {exc})")
        return
    try:
        app.update()
        app.v["name"].set("Asha")
        app.v["age"].set("30")
        app.v["weight"].set("60")
        app.v["height"].set("165")
        app.v["job"].set("standing")
        old = prof.USERS_DIR
        prof.USERS_DIR = tempfile.mkdtemp()
        try:
            app._submit_profile()
            app.update()
            check(app.profile is not None and app.profile.name == "Asha", "profile screen -> routines")
            check(app.choice.get() == "standing_workers", "standing job pre-selects standing routine")
            app._pick_routine()
            app.update()
            check(app.plan is not None and len(app.plan.items) == 3, "routine screen -> plan")
            app.voice.set(False)
            app.start_practice()
            app.update()
            check(app.pos == 0 and len(app.queue) == 3, "practice starts at pose 1")
            # drive a guided pose end to end with the clock short-circuited
            # the timer fallback still exists for a pose whose fit is missing
            from yoga import asanas as _al
            saved = _al.LIBRARY.pop("balasana")
            try:
                guided = prof.PlanItem("balasana", 20)
                check(not rt.POSES["balasana"].is_scored, "a pose without a fit falls back to the timer")
                app.queue = app.queue + [guided]
                app.pos = app.queue.index(guided)
                app.next_pose()
                app._run_guided(rt.POSES[guided.pose_key], guided)
                app.update()
                check(app.big.cget("text") == "5", "guided pose opens with a 5 s get-ready count")
                app._done(rt.POSES[guided.pose_key], "ok", completed=True)
                check(app.results and app.results[-1]["completed"], "guided pose result recorded")
            finally:
                _al.LIBRARY["balasana"] = saved
            app.pos = len(app.queue)
            app.next_pose()
            app.update()
            check(os.path.isfile(os.path.join(prof.USERS_DIR, "Asha_history.json")),
                  "summary writes the history file")
        finally:
            prof.USERS_DIR = old
    finally:
        app.destroy()


def main() -> int:
    for fn in (test_catalogue, test_profile, test_personalise, test_posespecs_and_guide,
               test_screens):
        print(fn.__name__)
        fn()
    print(f"\n{N - len(FAILS)}/{N} flow checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
