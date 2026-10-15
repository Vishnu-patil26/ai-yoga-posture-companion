"""Offline checks for the routine / profile layer and the launcher screens.

    python tests/test_flow.py        (no camera needed; opens and closes a window)
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


def test_theme_and_draw() -> None:
    import numpy as np
    from yoga import draw, theme
    check(theme.bgr("#ff8000") == (0, 128, 255), "bgr() swaps channels for OpenCV")
    check(theme.glow_at(0) == theme.GLOW[0] and theme.glow_at(1) == theme.GLOW[2],
          "the title rule runs blue to pink")
    check(all(theme.TOKENS[k].startswith("#") and len(theme.TOKENS[k]) == 7 for k in theme.TOKENS),
          "every theme token is a #rrggbb colour")
    frame = np.zeros((100, 300, 3), np.uint8)
    with draw.Pen(frame, 0, 0, 150, 100) as pen:
        w = pen.text((10, 50), "Hello", 20, (255, 255, 255))
        pen.text((290, 50), "clipped away", 20, (255, 255, 255), anchor="r")
    check(frame[:, :150].any() and not frame[:, 150:].any(),
          "text is drawn only inside the rectangle given to the pen")
    check(abs(w - draw.Pen.width("Hello", 20)) < 1e-6, "text() returns the width it drew")
    again = np.zeros_like(frame)
    with draw.Pen(again, 0, 0, 150, 100) as pen:
        pen.text((10, 50), "Hello", 20, (255, 255, 255))
    check(np.array_equal(again[:, :150], frame[:, :150]),
          "a cached string draws identically the second time")
    check(draw.Pen.clip("a very long label indeed", 60, 14).endswith("\u2026"),
          "a long string is clipped with an ellipsis")
    check(draw.Pen.clip("ok", 600, 14) == "ok", "a short string is left alone")
    lines = draw.Pen.wrap("stand tall and lengthen your spine so your shoulders stack over your hips",
                          140, 13)
    check(len(lines) > 1 and all(draw.Pen.width(ln, 13) <= 140 for ln in lines),
          "wrap keeps every line inside the width")
    coloured = np.zeros((40, 100, 3), np.uint8)
    draw.Pen(coloured).text((4, 30), "I", 24, (10, 20, 250))
    ys, xs = np.nonzero(coloured.sum(axis=2))
    px = coloured[ys[0], xs[0]]
    check(px[2] >= px[0] and px[2] >= px[1], "colour tuples are (b, g, r), like OpenCV")


def test_hud_layout() -> None:
    import numpy as np
    from yoga import overlay
    from yoga.coach import JOURNEY
    from yoga.state_machine import PoseStateMachine
    frame = np.full((720, 1280, 3), 90, np.uint8)
    sm = PoseStateMachine(hold_target_s=20)
    overlay.draw_hud(frame, None, sm, "No body detected - step into frame", 28.0, hint="x",
                     right_margin=282)
    check(int(frame[300, 10].mean()) < 20, "the left panel is near-black")
    check(int(frame[300, 700].mean()) > 60, "the camera picture is untouched in the middle")
    overlay.draw_step_panel(frame, JOURNEY[:-1], 6)
    check(44 + overlay.FRAMING_H + 10 + overlay.MAX_STEP_ROWS * 20 + 36 <= 720 - overlay.CARD_H - 16,
          "the step list ends above the reference card even for Tree's twelve stages")
    check(all(callable(getattr(overlay, f, None)) for f in
              ("draw_hud", "draw_framing_panel", "draw_step_panel", "draw_reference_card",
               "draw_pose_guide", "draw_rest_panel", "draw_intro_frame", "draw_calibration")),
          "every overlay screen exists")


def test_window_scaling() -> None:
    import numpy as np
    import app as trainer
    # a window of a different size gets the picture scaled to fit, never a corner of grey
    calls: list[tuple] = []
    real_rect, real_show = trainer.cv2.getWindowImageRect, trainer.cv2.imshow
    trainer.cv2.getWindowImageRect = lambda name: (0, 0, 1920, 1080)
    trainer.cv2.imshow = lambda name, img: calls.append(img.shape)
    try:
        trainer.show("w", np.zeros((720, 1280, 3), np.uint8))
        check(calls[-1] == (1080, 1920, 3), "a maximised window shows the picture scaled to fill it")
        trainer.cv2.getWindowImageRect = lambda name: (0, 0, 1280, 720)
        trainer.show("w", np.zeros((720, 1280, 3), np.uint8))
        check(calls[-1] == (720, 1280, 3), "a window of the picture's own size is not rescaled")
        trainer.cv2.getWindowImageRect = lambda name: (0, 0, 0, 0)
        trainer.show("w", np.zeros((720, 1280, 3), np.uint8))
        check(calls[-1] == (720, 1280, 3), "an unreadable window size falls back to the plain picture")
    finally:
        trainer.cv2.getWindowImageRect, trainer.cv2.imshow = real_rect, real_show


def test_fit_frame() -> None:
    import numpy as np
    import app as trainer
    from yoga import theme
    small = np.full((480, 640, 3), 255, np.uint8)
    out = trainer.fit_frame(small, 1280, 720)
    check(out.shape == (720, 1280, 3), "a 4:3 camera frame is fitted to the 1280x720 layout")
    check(int(out[360, 640].mean()) == 255 and tuple(out[360, 10]) == theme.bgr(theme.TOKENS["bg"]),
          "the picture is centred with black bars either side")
    same = np.zeros((720, 1280, 3), np.uint8)
    check(trainer.fit_frame(same, 1280, 720) is same, "a frame that already fits is not copied")
    big = trainer.fit_frame(np.zeros((1080, 1920, 3), np.uint8), 1280, 720)
    check(big.shape == (720, 1280, 3), "a 1080p frame is scaled down to fit")


def test_launcher_ui() -> None:
    try:
        import launcher
        app = launcher.App()
    except Exception as exc:                                  # no display (CI / Docker)
        print(f"  skip  launcher ui ({type(exc).__name__}: {exc})")
        return
    from yoga import ui
    old = prof.USERS_DIR
    prof.USERS_DIR = tempfile.mkdtemp()
    try:
        app.update()
        check(app.cget("bg") == "#050505", "the window is on the black page colour")
        # a bad profile is explained on the screen, not in a pop-up
        app.v["name"].set("Asha")
        app.v["age"].set("abc")
        app.v["weight"].set("60")
        app.v["height"].set("165")
        app._submit_profile()
        app.update()
        check(app.profile is None and len(app.flash_box.winfo_children()) == 1,
              "a bad profile shows an inline message and stays on the screen")
        check(app.ent["age"].cget("highlightbackground") == "#ff5c5c",
              "the offending field is outlined in red")
        app.v["age"].set("30")
        app.v["medication"].set("metformin")
        app._submit_profile()
        app.update()
        check(app.profile is not None and app.profile.medication == "metformin"
              and bool(app.profile.user_id), "medication is saved and the profile gets an id")
        check(len(app.flash_box.winfo_children()) == 0 and app.flash_box.winfo_reqheight() <= 2,
              "a dismissed message leaves no gap on the next screen")
        # ask the coach, in words
        app.ask_var.set("my lower back hurts and I only have 10 minutes")
        app._ask()
        app.update()
        check(app.choice.get() == "back_pain", "the coach picks the back-pain routine from a sentence")
        check(len(app.ask_out.winfo_children()) > 0, "the coach explains its choice")
        app.ask_var.set("")
        app._ask()
        check(len(app.ask_out.winfo_children()) == 1, "an empty question gets a prompt, not a crash")
        # the pose library: tick order, badges
        app.show_gallery()
        app.update()
        check(set(app.pick_vars) == set(rt.POSES), "the library shows every pose")
        for k in ("tadasana", "vrikshasana"):
            app.pick_vars[k].set(True)
            app.picked.append(k)
        app.show_gallery()
        app.update()
        check(app.picked == ["tadasana", "vrikshasana"], "ticks keep the order they were made in")
        check(app.badges["vrikshasana"].cget("text") == "#2"
              and app.badges["tadasana"].cget("text") == "#1", "the order badge numbers the ticked poses")
        app._clear_picks()
        check(app.picked == [] and not app.badges["tadasana"].winfo_ismapped(),
              "Clear removes ticks and badges")
        app._build_custom()
        app.update()
        check(len(app.flash_box.winfo_children()) == 1 and app.plan is None,
              "building a routine with nothing ticked explains why")
        # a toggle behaves as a radio group / a check box
        v = ui.tk.StringVar(value="a")
        r1 = ui.Toggle(app.page, "one", v, "a", "radio")
        r2 = ui.Toggle(app.page, "two", v, "b", "radio")
        r2._click()
        check(v.get() == "b" and r2._on() and not r1._on(), "radio toggles select one value")
        c = ui.tk.BooleanVar(value=False)
        t = ui.Toggle(app.page, "tick", c)
        t._click()
        t._click()
        check(c.get() is False, "a check toggle flips on every click")
        # the wheel handler must never raise once a scroll area has been rebuilt
        app.show_profile()
        app.update()
        errors: list = []
        app.report_callback_exception = lambda *a: errors.append(a)
        app.event_generate("<MouseWheel>", delta=-120)
        app.update()
        check(not errors, "scrolling after changing screens raises nothing")
        app.quit_app()
        check(True, "quit_app returns cleanly")
    finally:
        prof.USERS_DIR = old
        try:
            app.destroy()
        except Exception:
            pass


def test_launcher_practice() -> None:
    """The scored-pose path, with the camera trainer replaced by a stand-in."""
    try:
        import launcher
        app = launcher.App()
    except Exception as exc:                                  # no display (CI / Docker)
        print(f"  skip  launcher practice ({type(exc).__name__}: {exc})")
        return
    import app as trainer
    real = trainer.run_live
    old = prof.USERS_DIR
    prof.USERS_DIR = tempfile.mkdtemp()
    try:
        app.update()
        app.profile = base()
        prof.save(app.profile)
        app.voice.set(False)
        routine = rt.Routine("t", "Test routine", "custom", "x", ("tadasana", "utkatasana", "vrikshasana"))
        app.plan = prof.personalise(app.profile, routine)
        seen: list[list[str]] = []

        def finished(args) -> int:
            seen.append(list(vars(args).keys()))
            args.result = {"completed": 1, "attempted": 1, "longest_s": 20.0, "best": 91.2}
            return 0

        trainer.run_live = finished
        app.start_practice()
        app.update()
        item = app.queue[0]
        app._begin(rt.POSES[item.pose_key], item)
        app.update()
        check(app.winfo_viewable(), "the launcher comes back after the camera window closes")
        check(app.pos == 1 and app.results[-1]["completed"] and "91.2%" in app.results[-1]["result"],
              "a completed hold is recorded with its numbers and the routine moves on")

        def no_camera(args) -> int:
            raise SystemExit("Camera 0 could not be used.")

        trainer.run_live = no_camera
        item = app.queue[1]
        n_results = len(app.results)
        app._begin(rt.POSES[item.pose_key], item)
        app.update()
        check(app.pos == 1 and len(app.results) == n_results,
              "a missing camera does not skip the pose or lose the routine")
        check(len(app.flash_box.winfo_children()) == 1, "the camera problem is shown on the pose screen")
        check(app.winfo_viewable(), "the launcher is shown again after a camera failure")

        def quit_early(args) -> int:
            args.result = {"completed": 0, "attempted": 0, "longest_s": 0.0, "best": 0.0}
            return 0

        trainer.run_live = quit_early
        app._begin(rt.POSES[item.pose_key], item)
        app.update()
        check(app.pos == 2 and app.results[-1]["result"] == "not completed",
              "quitting a pose early is recorded as not completed")
        app.show_summary()
        app.show_summary()
        path = os.path.join(prof.USERS_DIR, "Asha_history.json")
        import json
        with open(path, encoding="utf-8") as fh:
            entries = json.load(fh)
        check(len(entries) == 1, "showing the summary twice saves the session once")
    finally:
        trainer.run_live = real
        prof.USERS_DIR = old
        try:
            app.destroy()
        except Exception:
            pass


def test_demo_clips() -> None:
    import numpy as np
    from yoga import demo
    check(demo.available(), "the demo data file exists (data/demo_poses.json)")
    for key in rt.POSES:
        c = demo.clip(key)
        check(c is not None, f"{key}: has a demo clip")
        if c is None:
            continue
        times = np.linspace(0.0, c.duration, 24)
        finite = all(np.isfinite(p).all() for t in times for p in c.points(t).values())
        check(finite, f"{key}: every joint of every frame is a real number")
        low = min(max(p[1] for j, p in c.points(t).items() if j != "NOSE") for t in times)
        check(abs(low) < 1e-6, f"{key}: the figure stands on the ground line")
        check({c.step(t) for t in times} <= {0, 1, 2}, f"{key}: the clip only ever shows steps 1-3")
        if key != "marjaryasana":                    # cat-cow blends two real shapes; the rest are kinematic
            arm = [float(np.linalg.norm(c.points(t)["LEFT_ELBOW"] - c.points(t)["LEFT_SHOULDER"])) for t in times]
            leg = [float(np.linalg.norm(c.points(t)["LEFT_KNEE"] - c.points(t)["LEFT_HIP"])) for t in times]
            check(max(arm) - min(arm) < 1e-6 and max(leg) - min(leg) < 1e-6,
                  f"{key}: bones keep their length through the movement")
        img = c.render(220, 180, 3.0, caption="a step", title=key.upper())
        check(img.shape == (180, 220, 3) and int(img.max()) > 200, f"{key}: renders a picture")
    tree = demo.clip("vrikshasana")
    check(tree.progress(0.1) == 0.0 and tree.progress(4.0) == 1.0, "a demo starts at rest and holds the pose")
    cc = demo.clip("marjaryasana")
    check(0.0 <= min(cc.progress(t) for t in np.linspace(0, 6, 30)) and max(cc.progress(t) for t in np.linspace(0, 6, 30)) <= 1.0,
          "cat-cow moves smoothly between its two shapes")


def test_camera_controls() -> None:
    import numpy as np
    import cv2
    import app as trainer
    from yoga import overlay
    frame = np.zeros((720, 1280, 3), np.uint8)
    btns = overlay.draw_toolbar(frame, "Mountain Pose", True, True, False)
    ids = {b[0] for b in btns}
    check({"prev", "menu", "next", "voice", "demo", "again"} <= ids, "the toolbar has pose, voice, demo and repeat buttons")
    v = next(b for b in btns if b[0] == "voice")
    check(overlay.hit(btns, (v[1] + v[3]) / 2, (v[2] + v[4]) / 2) == "voice", "a click inside a button finds it")
    check(overlay.hit(btns, 5, 700) is None, "a click on empty picture finds nothing")
    check(all(0 <= b[1] < b[3] <= 1280 and 0 <= b[2] < b[4] <= 720 for b in btns), "every button is inside the picture")
    locked = {b[0] for b in overlay.draw_toolbar(frame, "Tree", False, False, False)}
    check("prev" not in locked and "next" not in locked, "a locked session has no previous/next buttons")
    entries = trainer.pose_menu_entries()
    check(len(entries) == 10 and entries[4][0] == "vrikshasana" and entries[4][3] == "full coach",
          "the pose list holds all ten poses and marks the full coach")
    menu = overlay.draw_pose_menu(frame, entries, "tadasana")
    check(len([b for b in menu if b[0].startswith("pose:")]) == 10, "the pose menu has a button per pose")
    check(overlay.hit(menu, (menu[1][1] + menu[1][3]) / 2, (menu[1][2] + menu[1][4]) / 2) == menu[1][0],
          "a click on a pose in the menu selects that pose")
    # a click in a maximised window maps back to the 1280x720 picture
    trainer._VIEW["w"] = (1.5, 100.0, 30.0)
    check(trainer.window_to_frame("w", 400.0, 180.0) == (200.0, 100.0), "window clicks are mapped back to the picture")
    ctl = trainer._Controls("w")
    ctl.buttons = [("voice", 190, 90, 210, 110)]
    ctl.on_mouse(cv2.EVENT_LBUTTONDOWN, 400, 180, 0, None)
    check(ctl.click == "voice", "a mouse click is recorded against the button under it")
    ctl.on_mouse(cv2.EVENT_MOUSEMOVE, 0, 0, 0, None)
    check(ctl.hover is None, "moving off a button clears the hover highlight")
    trainer._VIEW.pop("w", None)
    # the pose card can carry the animated demo in place of the photo
    card = np.zeros((720, 1280, 3), np.uint8)
    small = np.full((overlay.DEMO_BOX_H, overlay.CARD_W - 28, 3), 200, np.uint8)
    overlay.draw_pose_guide(card, None, "Chair Pose", "Utkatasana", ("a", "b", "c"), current=0, demo_img=small)
    x = 1280 - overlay.CARD_W - 16 + 14 + 40
    check(int(card[720 - overlay.CARD_H - 16 + 34 + 20, x].mean()) == 200, "the demo picture is shown on the pose card")


def test_launcher_choose_and_audio() -> None:
    try:
        import launcher
        app = launcher.App()
    except Exception as exc:                                  # no display (CI / Docker)
        print(f"  skip  launcher choose/audio ({type(exc).__name__}: {exc})")
        return
    import app as trainer
    real = trainer.run_live
    old = prof.USERS_DIR
    prof.USERS_DIR = tempfile.mkdtemp()
    said: list[str] = []
    try:
        app.update()
        app.say = lambda text: said.append(text)              # no real voice in a test
        # guest: any pose without the form
        app._make_guest_and_go()
        app.update()
        check(app.guest and app.profile.name == "Guest" and not prof.saved_names(),
              "'practise as a guest' needs no form and saves nothing")
        # one pose straight from the library
        launched: list[tuple] = []

        def fake_live(args) -> int:
            launched.append((args.asana, args.stop_after_hold, args.no_voice))
            args.results = [{"key": args.asana, "name": "x", "completed": 1, "attempted": 1,
                             "longest_s": 20.0, "best": 90.0},
                            {"key": "utkatasana", "name": "Chair Pose", "completed": 0, "attempted": 1,
                             "longest_s": 0.0, "best": 55.0}]
            args.result = {"completed": 1, "attempted": 1, "longest_s": 20.0, "best": 90.0}
            return 0

        trainer.run_live = fake_live
        app.voice.set(False)
        app.practise_now("tadasana")
        app.update()
        check(app.plan.routine.title == "Practise Mountain Pose" and len(app.queue) == 1,
              "'practise this pose now' builds a one-pose session")
        app._begin(rt.POSES["tadasana"], app.queue[0])
        app.update()
        check(launched[-1][0] == "tadasana" and launched[-1][1] is True and launched[-1][2] is True,
              "the camera opens on the chosen pose, silent because the voice is off")
        rows = [r["pose"] for r in app.results]
        check(any("Chair Pose (chosen in the camera window)" in r for r in rows),
              "a pose switched to inside the camera window is counted in the session")
        # free practice: the pose is chosen in the camera window
        app.practise_now("vrikshasana", free=True)
        app.update()
        check(launched[-1][0] == "vrikshasana" and launched[-1][1] is False,
              "free practice opens the camera without stopping after the first hold")
        check(app.plan.routine.title == "Free practice" and len(app.results) == 2,
              "free practice ends on a summary of every pose practised")
        check(len(prof.saved_names()) == 0, "a guest session writes no history")
        # a pose that the profile rules out is explained, not started
        app.profile = base(bp="High")
        app.guest = False
        app.practise_now("adho_mukha")
        app.update()
        check(len(app.flash_box.winfo_children()) == 1 and "skipped" in app.flash_box.winfo_children()[0].winfo_children()[0].winfo_children()[0].cget("text"),
              "a head-below-heart pose is refused for high blood pressure, with the reason")
        # choosing another pose on the get-ready screen
        app.profile = base()
        app.plan = prof.personalise(app.profile, rt.Routine("t", "T", "custom", "x", ("tadasana", "utkatasana")))
        app.start_practice()
        app.update()
        names = [i.pose_key for i in app.queue]
        app._change_pose("vrikshasana", "tadasana", __import__("tkinter").StringVar())
        app.update()
        check(app.queue[0].pose_key == "vrikshasana" and len(app.queue) == len(names),
              "the pose can be changed on the get-ready screen")
        # audio: play and stop on demand, and the voice switch silences at once
        said.clear()
        app.voice.set(False)
        app._play_audio(rt.POSES["vrikshasana"])
        check(app.voice.get() is True and said and said[-1].startswith("Tree Pose."),
              "'Play audio' turns the voice on and reads the steps")
        stopped: list[int] = []

        class _Mute:
            def cancel(self):
                stopped.append(1)

            def close(self):
                pass

        app.speaker = _Mute()
        app.hush()
        app.toggle_voice()                                    # on -> off
        check(len(stopped) == 2 and app.voice.get() is False, "'Stop audio' and the voice switch both cut the speech")
        check(app.voice_btn.cget("text") == "Voice: OFF", "the sidebar button shows the voice state")
        app.speaker = None
    finally:
        trainer.run_live = real
        prof.USERS_DIR = old
        try:
            app.destroy()
        except Exception:
            pass


def main() -> int:
    for fn in (test_catalogue, test_profile, test_personalise, test_posespecs_and_guide,
               test_screens, test_theme_and_draw, test_hud_layout, test_window_scaling, test_fit_frame,
               test_launcher_ui, test_launcher_practice, test_demo_clips, test_camera_controls,
               test_launcher_choose_and_audio):
        print(fn.__name__)
        fn()
    print(f"\n{N - len(FAILS)}/{N} flow checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
