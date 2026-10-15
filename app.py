"""AI Yoga Posture & Wellness Companion - live trainer.

Base implementation covering one asana end to end:
webcam -> BlazePose (33 landmarks) -> joint angles -> per-joint comparison with
the reference asana -> alignment score, hold timer, spoken correction, session
log.  No video is recorded; frames are used and discarded.

    python app.py                          # live webcam, Vrikshasana
    python app.py --calibrate              # learn this practitioner's tolerance
    python app.py --source clip.mp4        # run against a recorded clip
    python app.py --no-voice               # silent, for a quiet room / viva

Quit with q or Esc (video window focused), or by closing the window.
Other keys:  n / b next / previous pose   l pose list (or 1-9, 0)   d demo video   v voice on/off
             a say the last instruction again   g guide panels   m metric numbers   r reset   s snapshot
Everything above is also a button across the top of the camera window.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import cv2
import numpy as np

from yoga import asanas as asana_lib
from yoga import calibration
from yoga.coach import JOURNEY, QUIET_BEFORE_END_S, SEQUENCE, Coach
from yoga.evaluator import compute_features, evaluate
from yoga.feedback import CueEngine, Speaker, play_alert
from yoga.filters import RollingMean
from yoga import demo, phrasing, theme
from yoga.landmarks import PoseTracker
from yoga.overlay import (
    CARD_W, COL_OK, DEMO_BOX_H, draw_calibration, draw_demo_panel, draw_framing_panel, draw_hud,
    draw_intro_frame, draw_pose_guide, draw_pose_menu, draw_reference_card, draw_rest_panel,
    draw_skeleton, draw_step_panel, draw_toolbar,
)
from yoga.overlay import hit as overlay_hit
from yoga.reference import ARM_FORMS, step_figure
from yoga.state_machine import PoseStateMachine, State
from yoga.storage import SessionLog

HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW = "AI Yoga Companion"
WINDOW_CALIBRATE = "AI Yoga Companion - calibration"


#: Windows we have actually seen mapped on screen at least once.
_window_seen: set[str] = set()


def window_closed(name: str) -> bool:
    """True once the user has closed a window that was previously open.

    OpenCV's HighGUI has no close callback: clicking the X destroys the window
    but does not stop the program, and the next `imshow` simply builds a new
    one - so the app appears unclosable and keeps holding the camera.  Polling
    the window property is the only way to notice.

    The "previously open" part matters.  A window takes a moment to be mapped
    after the first `imshow`, and during that gap the visibility property
    reads 0 - indistinguishable from having been closed.  Treating that as a
    close made the app quit during the first frame of the walkthrough.  So a
    window only counts as closed once we have seen it open.
    """
    try:
        visible = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return name in _window_seen        # destroyed after having been open
    if visible:
        _window_seen.add(name)
        return False
    return name in _window_seen

#: How long a spoken cue stays on screen before the banner falls back to
#: describing the current state.
BANNER_HOLD_S = 4.0


def _idle_banner(ev, state, asana) -> str:
    """What the banner says when no cue is currently being shown."""
    if ev is None:
        return "No body detected - step into frame"
    if not ev.usable:
        return ev.reason or "Cannot see enough of your body to score"
    if state is State.HOLDING:
        return "Good. Hold it."
    if state is State.COMPLETE:
        return "Well held. Release the pose slowly."
    if ev.primary is not None:
        return ev.primary.cue()
    return asana.setup_hint


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AI Yoga Posture & Wellness Companion")
    ap.add_argument("--asana", default="vrikshasana", help="asana key from the library")
    ap.add_argument("--source", default="0", help="webcam index or path to a video file")
    ap.add_argument("--model", default="full", choices=["full", "lite", "heavy"])
    ap.add_argument("--user", default="default", help="calibration profile name")
    ap.add_argument("--calibrate", action="store_true", help="record a personal tolerance profile")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-db", action="store_true", help="do not write the session log")
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the webcam image")
    ap.add_argument("--show-raw", action="store_true", help="draw the unsmoothed landmarks too")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hold", type=float, default=None, help="override hold target, seconds")
    ap.add_argument("--record", default=None, help="write the annotated view to this .mp4")
    ap.add_argument("--no-guide", action="store_true",
                    help="hide the reference card and framing panel")
    ap.add_argument("--no-coach", action="store_true",
                    help="skip the spoken step-by-step entry into the pose")
    ap.add_argument("--metric", action="store_true",
                    help="show raw degrees/ratio numbers on the joint table "
                         "instead of body-relative words")
    ap.add_argument("--no-intro", action="store_true",
                    help="skip the walkthrough and go straight to the camera")
    ap.add_argument("--no-steps", action="store_true",
                    help="skip the step-by-step walk-in for fitted poses")
    ap.add_argument("--lock-pose", action="store_true",
                    help="do not let the camera window switch to another pose")
    ap.add_argument("--stop-after-hold", action="store_true",
                    help="end the session a few seconds after the first completed hold "
                         "(used by the routine launcher)")
    ap.add_argument("--demo", action="store_true",
                    help="play the walkthrough only, then exit - for showing "
                         "the whole flow without a camera")
    return ap.parse_args(argv)


def _camera_backends():
    """Capture backends to try, in order.

    `isOpened()` is not enough on Windows: DirectShow will report a camera as
    open and then block forever on the first read, which is exactly how this
    app appeared to hang on a machine whose webcam was perfectly fine.  Each
    backend is therefore proved by actually pulling a frame out of it.
    """
    if sys.platform == "win32":
        return [(cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_ANY, "default")]
    return [(cv2.CAP_ANY, "default")]


def _open_camera(index: int, width: int, height: int):
    problems = []
    for backend, name in _camera_backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            problems.append(f"{name}: would not open")
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"camera {index} opened via {name} "
                  f"({frame.shape[1]}x{frame.shape[0]})")
            return cap
        problems.append(f"{name}: opened but delivered no frames")
        cap.release()
    raise SystemExit(
        f"Camera {index} could not be used.\n  "
        + "\n  ".join(problems)
        + "\n\nMost often this is a Windows privacy setting: Settings > Privacy &\n"
        "security > Camera > 'Let desktop apps access your camera'. Otherwise\n"
        "close any other app using the camera, or try --source 1.\n"
        "No camera at all? Run against a clip:  python app.py --source clip.mp4"
    )


def _screen_size() -> tuple[int, int]:
    """Usable screen size in pixels (a sensible default where it cannot be read)."""
    if sys.platform == "win32":
        try:
            import ctypes
            u = ctypes.windll.user32
            return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
        except Exception:
            pass
    return 1920, 1080


_OPEN: set[str] = set()                #: windows this module has created and not yet destroyed


def open_window(name: str, width: int, height: int) -> None:
    """Create the camera window once, resizable, no bigger than the screen.

    A fixed-size OpenCV window that is maximised shows the picture in one corner
    and grey everywhere else.  A normal window can be resized, and `show()` below
    scales the picture to whatever size it has.  Calling this again for a window
    that is already open does nothing (so it never undoes the size the user chose).
    """
    if name in _OPEN:
        return
    sw, sh = _screen_size()
    s = min(1.0, (sw * 0.94) / width, (sh * 0.88) / height)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, int(width * s), int(height * s))
    _OPEN.add(name)


def close_windows() -> None:
    """Destroy every window and forget them, so the next session builds a fresh one."""
    _OPEN.clear()
    _VIEW.clear()
    cv2.destroyAllWindows()


#: window name -> (scale, x offset, y offset) of the picture inside the window, for clicks
_VIEW: dict[str, tuple[float, float, float]] = {}


def show(name: str, frame: np.ndarray) -> None:
    """imshow, but scaled (with black bars) to fill the window at its current size."""
    try:
        _x, _y, ww, wh = cv2.getWindowImageRect(name)
    except cv2.error:
        ww = wh = 0
    h, w = frame.shape[:2]
    if ww > 64 and wh > 64 and (ww, wh) != (w, h):
        scale = min(ww / w, wh / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        _VIEW[name] = (nw / w, (ww - nw) // 2, (wh - nh) // 2)
        frame = fit_frame(frame, ww, wh)
    else:
        _VIEW[name] = (1.0, 0.0, 0.0)
    cv2.imshow(name, frame)


def window_to_frame(name: str, x: float, y: float) -> tuple[float, float]:
    """A point in the window -> the same point in the 1280x720 picture."""
    scale, ox, oy = _VIEW.get(name, (1.0, 0.0, 0.0))
    return (x - ox) / scale, (y - oy) / scale


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox a camera frame into ``width`` x ``height``.

    The panels are laid out for a 1280x720 picture.  A webcam that answers with
    640x480 (or 1920x1080) would otherwise put the left panel and the right-hand
    cards over almost the whole image.  Scaling to fit, with black bars, keeps
    the layout the same whatever the camera or video file delivers.
    """
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    s = min(width / w, height / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.empty((height, width, 3), np.uint8)
    out[:] = theme.bgr(theme.TOKENS["bg"])
    y, x = (height - nh) // 2, (width - nw) // 2
    out[y:y + nh, x:x + nw] = im
    return out


def open_source(source: str, width: int, height: int):
    if source.isdigit():
        return _open_camera(int(source), width, height), True
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Could not open video file '" + source + "'.")
    return cap, False


# ------------------------------------------------------------------------ intro
#: Minimum time a walkthrough slide stays up.  This is a preview someone
#: watches before starting, not the lesson - it should read as a quick flick
#: through the shapes, so it is paced to a glance rather than to a full
#: reading.  The whole seven-step run lands around ten seconds.
INTRO_MIN_SLIDE_S = 0.95


def run_intro(args, asana, speaker, record_writer=None, journey=None) -> bool:
    """Play the whole practice through once before the camera starts.

    Returns False if the practitioner asked to quit outright (q/Esc/window
    closed); True if it finished or they skipped ahead with any other key.

    Each slide waits for its own spoken line to finish rather than using a
    fixed delay, so the walkthrough stays in step with the voice however long
    or short the instruction happens to be.
    """
    width, height = args.width, args.height
    open_window(WINDOW, width, height)
    canvas = np.zeros((height, width, 3), np.uint8)
    # One clock for the whole walkthrough, not one per slide, so the speaker's
    # "when am I next free" stamps stay on a single monotonic timeline.
    started = time.perf_counter()

    journey = journey or JOURNEY
    # The preview shows each distinct shape once.  The second arms-up and
    # arms-down are the same picture as the first, so previewing them only
    # makes the walkthrough longer without showing anything new; the practice
    # itself still runs every stage.
    journey = [j for j in journey if j.key not in ("up2", "down2")]
    for index, step in enumerate(journey):
        speak_s = speaker.estimate_seconds(step.say) if not args.no_voice else 0.0
        total_s = max(INTRO_MIN_SLIDE_S, speak_s + 0.25)
        pose = step_figure(step.key)
        spoken = False
        t0 = time.perf_counter()

        while True:
            now = time.perf_counter()
            elapsed = now - t0
            if not spoken:
                spoken = True
                speaker.say(step.say, now - started)
                print(f"  [walkthrough {index + 1}/{len(journey)}] {step.title}")
            draw_intro_frame(canvas, asana, journey, index, pose, elapsed, total_s)
            show(WINDOW, canvas)
            if record_writer is not None:
                record_writer.write(canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return False
            if window_closed(WINDOW):
                return False
            if key != 255:                 # any other key skips the walkthrough
                return True
            if elapsed >= total_s:
                break
    return True


# --------------------------------------------------------------------- calibrate
def run_calibration(args) -> int:
    asana = asana_lib.get(args.asana)
    cap, _ = open_source(args.source, args.width, args.height)
    tracker = PoseTracker(model=args.model)
    speaker = Speaker(enabled=not args.no_voice)
    open_window(WINDOW_CALIBRATE, args.width, args.height)
    samples: list[dict] = []
    countdown, capture_s = 6.0, 8.0
    t0 = time.perf_counter()
    spoken: set[int] = set()

    print("")
    print("Calibration - " + asana.sanskrit)
    print("  " + asana.setup_hint)
    print("  Get into your best version of the pose. Recording for "
          f"{capture_s:.0f}s after a {countdown:.0f}s countdown.")
    print("")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = fit_frame(frame, args.width, args.height)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            t = time.perf_counter() - t0
            pose = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(t * 1000))
            if pose is not None:
                draw_skeleton(frame, pose, None)

            if t < countdown:
                left = int(countdown - t) + 1
                if left not in spoken:
                    spoken.add(left)
                    if left <= 3:
                        speaker.say(str(left))
                draw_calibration(frame, f"Get into the pose...  {left}", asana.setup_hint)
            else:
                if pose is not None:
                    feats = compute_features(pose)
                    if evaluate(asana, feats).usable:
                        samples.append(feats)
                remaining = max(0.0, countdown + capture_s - t)
                draw_calibration(frame, f"HOLD   {remaining:.1f}s left",
                                 f"{len(samples)} usable samples so far", COL_OK)
            show(WINDOW_CALIBRATE, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if window_closed(WINDOW_CALIBRATE):
                break
            if t >= countdown + capture_s:
                break
    except KeyboardInterrupt:
        print("\ncalibration cancelled")
    finally:
        cap.release()
        tracker.close()
        close_windows()

    if len(samples) < 15:
        print(f"Only {len(samples)} usable frames - not enough to calibrate. "
              "Make sure your whole body is in frame and try again.")
        speaker.close()
        return 1

    prof = calibration.build_profile(asana, samples, user=args.user)
    path = calibration.save_profile(prof, user=args.user)
    print("")
    print(f"Calibrated from {len(samples)} frames -> {path}")
    print("")
    header = "check".ljust(22) + "target".rjust(10) + "tol".rjust(10)
    header += "your median".rjust(14) + "your sd".rjust(10)
    print(header)
    for key, p in prof["checks"].items():
        print(key.ljust(22)
              + f"{p['target']:10.2f}{p['tol']:10.2f}"
              + f"{p['observed_median']:14.2f}{p['observed_sd']:10.2f}")
    speaker.say("Calibration saved")
    speaker.close()
    return 0


# -------------------------------------------------------------------- live loop
class PoseSession:
    """Everything that belongs to one pose being practised.

    Switching pose in the camera window builds a new one: its own hold timer,
    cue engine, coach or step guide and practice log - and the camera, the
    tracker and the voice carry straight on.
    """

    def __init__(self, key: str, args, hold: float | None = None, source_label: str = "webcam") -> None:
        self.key = key
        asana = asana_lib.get(key)
        profile = calibration.load_profile(args.user)
        if profile:
            asana = calibration.apply_profile(asana, profile)
            print(f"Using calibration profile '{args.user}' ({profile.get('n_samples')} frames)")
        else:
            print("No calibration profile found - using default tolerances "
                  "(run with --calibrate to personalise them)")
        if hold:
            asana = replace(asana, hold_target_s=hold)
        self.asana = asana

        # The guided entry sequence is written for Vrikshasana specifically - it
        # talks about shifting onto one foot and pressing the other into the inner
        # thigh.  Reading that out over a Warrior or a Chair would be actively
        # wrong, so for any other asana the app scores and corrects without
        # pretending to know how to walk you in.  The hold timer, the rest window
        # and the corrections all still apply.
        self.guided = key == "vrikshasana"
        self.no_coach = bool(args.no_coach) or not self.guided
        if not self.guided and not args.no_coach:
            from yoga import posespecs as _ps
            how = ("a step-by-step walk-in and scoring" if _ps.STEPS.get(key) and not args.no_steps
                   else "scoring mode")
            print(f"note: {asana.sanskrit} runs with {how} (hold timer and corrections active); "
                  f"the longer 12-stage coach is written for Vrikshasana")

        self.sm = PoseStateMachine(hold_target_s=asana.hold_target_s,
                                   enter_score=asana.enter_score, exit_score=asana.exit_score)
        self.cues = CueEngine(state_cues=self.no_coach, phrase_fn=phrasing.spoken_phrase)
        self.coach = Coach(hold_target_s=asana.hold_target_s)
        self.smooth = RollingMean(window=5)
        self.journey = (JOURNEY[:-2] + JOURNEY[-1:]) if self.guided else JOURNEY[-2:]
        self.log = None if args.no_db else SessionLog(key, source=source_label)
        self.banner = asana.setup_hint
        self.banner_until = 0.0
        self.last_line = asana.setup_hint          # what "say it again" repeats
        self.done_at: float | None = None
        self.frames = 0
        self.started = time.perf_counter()
        # What to show the person for a pose that has no drawn figure: its photo and
        # the steps for getting into it (yoga/routines.py is the single source).
        from yoga import routines as _routines
        info = _routines.POSES.get(key)
        self.guide_name = info.name if info else asana.name
        self.guide_steps = info.steps if info else ()
        self.guide_photo = cv2.imread(os.path.join(HERE, "assets", "gallery", key + ".jpg"))
        self.step_guide = None
        if not self.guided and not args.no_steps:
            from yoga import posespecs
            from yoga.guide import StepGuide
            spec = posespecs.STEPS.get(key)
            if spec:
                self.step_guide = StepGuide(asana, spec)
                self.guide_steps = tuple(s.text for s in self.step_guide.steps)

    def announce(self) -> None:
        print("")
        print(f"{self.asana.sanskrit} ({self.asana.name}) - hold target {self.asana.hold_target_s:.0f}s, "
              f"enter at {self.asana.enter_score:.0f}%")
        print(f"{len(self.journey)} steps: " + " -> ".join(s.title for s in self.journey))
        print(self.asana.setup_hint)

    def reset(self) -> None:
        self.sm = PoseStateMachine(hold_target_s=self.asana.hold_target_s,
                                   enter_score=self.asana.enter_score, exit_score=self.asana.exit_score)
        self.cues = CueEngine(state_cues=self.no_coach, phrase_fn=phrasing.spoken_phrase)
        self.coach = Coach(hold_target_s=self.asana.hold_target_s)
        self.smooth.reset()

    def result(self) -> dict:
        st = self.sm.stats
        return {"key": self.key, "name": self.asana.name, "completed": st.completed,
                "attempted": len(st.holds), "longest_s": round(st.longest_s, 1),
                "best": round(st.best_score, 1)}

    def finish(self) -> None:
        elapsed = max(1e-3, time.perf_counter() - self.started)
        if self.log:
            self.log.finish(self.sm.stats, self.frames, self.frames / elapsed)


class _Controls:
    """Mouse state for the camera window: what is under the pointer, what was clicked."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.buttons: list = []
        self.hover = None
        self.click = None

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        fx, fy = window_to_frame(self.name, x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = overlay_hit(self.buttons, fx, fy)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.click = overlay_hit(self.buttons, fx, fy)


_NUMBER_KEYS = {ord(c): i for i, c in enumerate("1234567890")}


def pose_menu_entries() -> list[tuple[str, str, str, str]]:
    """``(key, name, sanskrit, tag)`` for every pose the trainer can score."""
    from yoga import routines as _routines
    out = []
    for key, p in _routines.POSES.items():
        if key in asana_lib.LIBRARY:
            out.append((key, p.name, p.sanskrit, "full coach" if key == "vrikshasana" else "camera-scored"))
    return out


def run_live(args) -> int:
    cap, live = open_source(args.source, args.width, args.height)
    open_window(WINDOW, args.width, args.height)
    ctl = _Controls(WINDOW)
    cv2.setMouseCallback(WINDOW, ctl.on_mouse)
    tracker = PoseTracker(model=args.model)
    speaker = Speaker(enabled=not args.no_voice)
    label = "webcam" if live else args.source
    ps = PoseSession(args.asana, args, hold=args.hold, source_label=label)
    entries = pose_menu_entries()
    keys = [e[0] for e in entries]
    can_switch = not getattr(args, "lock_pose", False)
    results: list[dict] = []

    writer = None
    fps = 30.0
    last_sample = -1.0
    t0 = time.perf_counter()
    prev = t0
    elapsed = 0.0
    menu_open = demo_open = False
    demo_t0 = 0.0

    ps.announce()
    print("To finish: press q or Esc with the video window focused, "
          "or close the window.   Keys: n/b next/previous pose, l pose list, d demo, "
          "v voice on/off, a say it again")
    print("")

    # Show the whole practice once before asking anyone to do it.  Quitting
    # during the walkthrough must still release the camera, so this sits
    # inside the same try/finally as the live loop.
    try:
        if not args.no_intro:
            if not run_intro(args, ps.asana, speaker, journey=ps.journey):
                cap.release()
                tracker.close()
                close_windows()
                speaker.close()
                ps.finish()
                print("quit during the walkthrough")
                args.results, args.result = [], {"completed": 0, "attempted": 0, "longest_s": 0.0, "best": 0.0}
                return 0
    except Exception as exc:                  # a broken intro must never block practice
        print(f"[intro] skipped ({exc})")

    # Start the session clock *after* the walkthrough.  Everything downstream
    # is measured from t0 - the hold timer, the cue cooldowns, the reported
    # frame rate and the session duration - so leaving it started before the
    # walkthrough would bill ~20s of slides to the practice and report a frame
    # rate for work the loop never did.
    t0 = time.perf_counter()
    prev = t0
    # The walkthrough stamped `free_at` on its own per-slide clock; carried
    # into the session it would read as "still speaking" and swallow the
    # opening cues.
    speaker.reset_timeline()
    ps.started = time.perf_counter()

    def switch_to(new_key: str, now_t: float) -> None:
        nonlocal ps, menu_open, demo_open
        if new_key == ps.key or new_key not in keys:
            return
        results.append(ps.result())
        ps.finish()
        speaker.cancel()
        ps = PoseSession(new_key, args, hold=None, source_label=label)
        ps.started = time.perf_counter()
        speaker.reset_timeline()
        menu_open = False
        line = f"{ps.guide_name}. {ps.asana.setup_hint}"
        ps.banner, ps.banner_until, ps.last_line = line, now_t + BANNER_HOLD_S, line
        if speaker.enabled:
            speaker.say(line, now_t)
        ps.announce()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = fit_frame(frame, args.width, args.height)
            if live and not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 1e-4:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            t = now - t0
            ps.frames += 1
            asana, sm, cues, coach = ps.asana, ps.sm, ps.cues, ps.coach

            pose = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   int(t * 1000), fps)

            ev = None
            if pose is not None:
                ev = evaluate(asana, compute_features(pose))
                score = ps.smooth(ev.score) if ev.usable else 0.0
                if ev.usable:
                    ev.score = score
                state = sm.update(t, score, ev.usable)
            else:
                ps.smooth.reset()
                state = sm.update(t, 0.0, False)

            # The coach walks a beginner into the pose; once they are in it the
            # correction engine takes over.  Coach lines pre-empt corrections
            # so the two never talk over each other.
            said = None
            if not ps.no_coach:
                said = coach.update(t, ev, state, sm.elapsed,
                                    speaker_busy=speaker.busy(t))
                if said is not None:
                    ps.banner = ps.last_line = said.text
                    ps.banner_until = t + max(BANNER_HOLD_S, speaker.estimate_seconds(said.text))
                    # Never start a new line over one that is still being
                    # spoken - pyttsx3 has no notion of priority, so a second
                    # say() mid-sentence either garbles both or is silently
                    # dropped by the queue.  The banner and the log always
                    # reflect the guidance either way; only the voice waits.
                    if not speaker.busy(t):
                        speaker.say(said.text, t)
                    print(f"[{t:6.1f}s] coach  {said.text}")
                    if ps.log:
                        ps.log.log_cue(t, "coach." + said.step_key, "guide", said.text)

            step_guide = ps.step_guide
            guide_active = step_guide is not None and not step_guide.finished
            if guide_active and ev is not None:
                spoken = step_guide.update(t, ev)
                if spoken:
                    ps.banner = ps.last_line = spoken
                    ps.banner_until = t + max(BANNER_HOLD_S, speaker.estimate_seconds(spoken))
                    speaker.say(spoken, t)
                    print(f"[{t:6.1f}s] step   {spoken}")
            guide_active = step_guide is not None and not step_guide.finished

            if ev is not None:
                # CueEngine hands off to the coach whenever the coach still has
                # something to say (mid-guidance, or the rest window between
                # attempts) - `coach.resting` must BLOCK corrections here, not
                # enable them, or a joint correction talks over the rest timer.
                # The closing seconds of a hold belong to the countdown and to
                # finishing the pose; a new fault raised there cannot be acted
                # on in time and only competes with it for the speaker.
                finishing = (state is State.HOLDING
                             and sm.hold_target_s - sm.elapsed <= QUIET_BEFORE_END_S)
                coach_quiet = (not coach.resting and not finishing and not guide_active and (
                    ps.no_coach or (said is None and coach.finished)))
                cue = (cues.update(t, ev, state, speaker_busy=speaker.busy(t))
                       if coach_quiet else None)
                if cue is not None:
                    ps.banner = ps.last_line = cue.text
                    ps.banner_until = t + max(BANNER_HOLD_S, speaker.estimate_seconds(cue.text))
                    if not speaker.busy(t):
                        speaker.say(cue.text, t)
                        cues.note_spoken(t, speaker.estimate_seconds(cue.text))
                    print(f"[{t:6.1f}s] {cue.level:<6} {cue.text}")
                    _chk = asana.check(cue.key)
                    if _chk is not None and _chk.alert and speaker.enabled:
                        play_alert()
                    if ps.log:
                        ps.log.log_cue(t, cue.key, cue.level, cue.text)
                if ps.log and ev.usable and t - last_sample >= 0.5:
                    last_sample = t
                    ps.log.log_joints(t, ev.results)

            # The banner is cue-driven, but a cue is deliberately rare - it has
            # to persist a second and clear a cooldown before it is spoken.
            # Without a fallback the last message just sits there, which is how
            # "no body detected" ended up printed over a body being scored.
            if t >= ps.banner_until:
                ps.banner = _idle_banner(ev, state, asana)

            if sm.just_completed:
                last = sm.stats.holds[-1]
                print(f"[{t:6.1f}s] HOLD COMPLETE - {last.duration_s:.1f}s "
                      f"at {last.avg_score:.1f}% average")
                if ps.done_at is None:
                    ps.done_at = t

            if pose is not None:
                draw_skeleton(frame, pose, ev, show_raw=args.show_raw)
            draw_hud(frame, ev, sm, ps.banner, fps, hint=asana.setup_hint,
                     right_margin=(0 if args.no_guide else 282), metric=args.metric)

            clip = demo.clip(ps.key) if demo.available() else None
            if not args.no_guide:
                draw_framing_panel(frame, ev.features.get("framing") if ev else None)
                if ps.guided:
                    # The practice stages map straight onto the panel; rest is
                    # the last entry.
                    step_index = (len(ps.journey) - 1 if coach.resting
                                  else min(coach.index, len(SEQUENCE) - 1))
                else:
                    # Scoring mode has only the two stages it actually runs.
                    step_index = 1 if coach.resting else 0
                draw_step_panel(frame, ps.journey, step_index)
                if coach.resting:
                    draw_rest_panel(frame, coach.rest_remaining, coach.rest_target_s)
                elif not ps.guided:
                    small = (clip.render(CARD_W - 28, DEMO_BOX_H, t, compact=True) if clip else None)
                    draw_pose_guide(frame, ps.guide_photo, ps.guide_name, asana.sanskrit, ps.guide_steps,
                                    current=(step_guide.index if step_guide else None),
                                    done=bool(step_guide and step_guide.finished), demo_img=small)
                else:
                    # While holding, cycle the card between the two accepted
                    # arm forms so the practitioner can see both are allowed.
                    key = coach.step_key
                    if key == "hold":
                        key = ARM_FORMS[int(t / 4.0) % len(ARM_FORMS)]
                    step_no = min(coach.index + 1, len(SEQUENCE))
                    draw_reference_card(frame, step_figure(key), step_no, len(SEQUENCE),
                                        coach.title, coach.instruction or asana.setup_hint,
                                        done=coach.finished,
                                        confirming=coach.confirming,
                                        confirm_fraction=coach.confirm_fraction,
                                        step_seconds=coach.step_seconds)

            # ---- the controls: pose chooser, voice, demo
            buttons: list = []
            if demo_open and clip is not None:
                dt_demo = t - demo_t0
                buttons += draw_demo_panel(frame, clip, dt_demo, ps.guide_steps[clip.step(dt_demo)]
                                           if ps.guide_steps else "", f"DEMO  {asana.sanskrit.upper()}",
                                           ctl.hover)
            buttons += draw_toolbar(frame, f"{ps.guide_name}  ({asana.sanskrit})", can_switch,
                                    speaker.enabled, demo_open, ctl.hover)
            if menu_open:
                buttons = draw_pose_menu(frame, entries, ps.key, ctl.hover)
            ctl.buttons = buttons

            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, 25.0,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)

            show(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if window_closed(WINDOW):
                break

            # ---- what was asked for, by key or by click
            action = ctl.click
            ctl.click = None
            if key in (ord("q"), 27):
                if key == 27 and (menu_open or demo_open):
                    menu_open = demo_open = False             # Esc first closes what is open
                else:
                    break
            elif key == ord("v"):
                action = "voice"
            elif key == ord("a"):
                action = "again"
            elif key == ord("d"):
                action = "close_demo" if demo_open else "demo"
            elif key == ord("l"):
                action = "close_menu" if menu_open else "menu"
            elif key == ord("n"):
                action = "next"
            elif key == ord("b"):
                action = "prev"
            elif key in _NUMBER_KEYS and menu_open and _NUMBER_KEYS[key] < len(keys):
                action = "pose:" + keys[_NUMBER_KEYS[key]]
            elif key in _NUMBER_KEYS and can_switch and _NUMBER_KEYS[key] < len(keys):
                action = "pose:" + keys[_NUMBER_KEYS[key]]

            if action == "voice":
                on = speaker.set_enabled(not speaker.enabled)
                print("voice " + ("on" if on else "off"))
                if on:
                    speaker.say("Voice on. " + ps.last_line, t)
            elif action == "again":
                if speaker.enabled and ps.last_line:
                    speaker.cancel()
                    speaker.reset_timeline()
                    speaker.say(ps.last_line, t)
                elif not speaker.enabled:
                    print("voice is off - press v to turn it on")
            elif action == "demo" and clip is not None:
                demo_open, demo_t0, menu_open = True, t, False
            elif action == "close_demo":
                demo_open = False
            elif action == "menu" and can_switch:
                menu_open, demo_open = True, False
            elif action == "close_menu":
                menu_open = False
            elif action in ("next", "prev") and can_switch:
                step = 1 if action == "next" else -1
                switch_to(keys[(keys.index(ps.key) + step) % len(keys)], t)
            elif action and action.startswith("pose:") and can_switch:
                switch_to(action[5:], t)
            elif action == "pose_locked":
                print("this session is locked to one pose (started from a routine)")

            if (getattr(args, "stop_after_hold", False) and ps.done_at is not None
                    and t - ps.done_at > 3.0):                # let "well held" register
                break
            if key == ord("g"):
                args.no_guide = not args.no_guide
                print("guide " + ("off" if args.no_guide else "on"))
            if key == ord("m"):
                args.metric = not args.metric
                print("joint table: " + ("raw numbers" if args.metric else "words"))
            if key == ord("r"):
                ps.reset()
                print("session reset")
            if key == ord("s"):
                snapdir = os.path.join(HERE, "data", "snapshots")
                os.makedirs(snapdir, exist_ok=True)
                name = os.path.join(snapdir, f"snap_{int(time.time())}.png")
                cv2.imwrite(name, frame)
                print("snapshot -> " + name)
    except KeyboardInterrupt:
        # Ctrl+C in the terminal is a legitimate way out; fall through to the
        # finally block so the camera is released and the session is still saved.
        print("\nstopping")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        tracker.close()
        close_windows()
        elapsed = max(1e-3, time.perf_counter() - t0)
        ps.finish()
        speaker.close()

    results.append(ps.result())
    for r in results:
        print("")
        print("---------------- session summary ----------------")
        print(f"asana              : {r['name']}")
        print(f"holds attempted    : {r['attempted']}")
        print(f"holds completed    : {r['completed']}")
        print(f"longest hold       : {r['longest_s']:.1f}s")
        print(f"best alignment     : {r['best']:.1f}%")
    print(f"duration           : {elapsed:.1f}s over {sum(1 for _ in results)} pose(s)")
    if ps.log:
        print(f"session logged to  : {ps.log.path} (session id {ps.log.session_id})")
    print("-------------------------------------------------")
    # Read by the routine launcher, which runs several poses in one process:
    # `result` is for the pose it asked for, `results` for every pose practised.
    asked = next((r for r in results if r["key"] == args.asana), None)
    pick = asked or {"completed": 0, "attempted": 0, "longest_s": 0.0, "best": 0.0}
    args.result = {k: pick[k] for k in ("completed", "attempted", "longest_s", "best")}
    args.results = results
    return 0


def run_demo(args) -> int:
    """The walkthrough on its own - no camera touched at all."""
    asana = asana_lib.get(args.asana)
    speaker = Speaker(enabled=not args.no_voice)
    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 25.0, (args.width, args.height))
    print("")
    preview = [j for j in JOURNEY if j.key not in ("up2", "down2")]
    print(f"{asana.sanskrit} ({asana.name}) - walkthrough, "
          f"{len(preview)} slides covering {len(JOURNEY)} stages")
    print("")
    try:
        run_intro(args, asana, speaker, record_writer=writer)
    finally:
        if writer is not None:
            writer.release()
        close_windows()
        speaker.close()
    print("\nwalkthrough finished")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.demo:
        return run_demo(args)
    return run_calibration(args) if args.calibrate else run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
