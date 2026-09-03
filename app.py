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
Other keys:  v voice on/off   g guide panels   m metric numbers   r reset   s snapshot
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
from yoga.feedback import CueEngine, Speaker
from yoga.filters import RollingMean
from yoga import phrasing
from yoga.landmarks import PoseTracker
from yoga.overlay import (
    draw_framing_panel, draw_hud, draw_intro_frame, draw_reference_card,
    draw_rest_panel, draw_skeleton, draw_step_panel,
)
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
            cv2.imshow(WINDOW, canvas)
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
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            t = time.perf_counter() - t0
            pose = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(t * 1000))

            if t < countdown:
                left = int(countdown - t) + 1
                if left not in spoken:
                    spoken.add(left)
                    if left <= 3:
                        speaker.say(str(left))
                cv2.putText(frame, f"Get into the pose... {left}", (40, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 190, 250), 3, cv2.LINE_AA)
            else:
                if pose is not None:
                    feats = compute_features(pose)
                    if evaluate(asana, feats).usable:
                        samples.append(feats)
                remaining = max(0.0, countdown + capture_s - t)
                cv2.putText(frame, f"HOLD  {len(samples)} samples  {remaining:.1f}s left",
                            (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 220, 120), 3, cv2.LINE_AA)

            if pose is not None:
                draw_skeleton(frame, pose, None)
            cv2.imshow(WINDOW_CALIBRATE, frame)
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
        cv2.destroyAllWindows()

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
def run_live(args) -> int:
    asana = asana_lib.get(args.asana)
    profile = calibration.load_profile(args.user)
    if profile:
        asana = calibration.apply_profile(asana, profile)
        print(f"Using calibration profile '{args.user}' "
              f"({profile.get('n_samples')} frames)")
    else:
        print("No calibration profile found - using default tolerances "
              "(run with --calibrate to personalise them)")
    if args.hold:
        asana = replace(asana, hold_target_s=args.hold)

    # The guided entry sequence is written for Vrikshasana specifically - it
    # talks about shifting onto one foot and pressing the other into the inner
    # thigh.  Reading that out over a Warrior or a Chair would be actively
    # wrong, so for any other asana the app scores and corrects without
    # pretending to know how to walk you in.  The hold timer, the rest window
    # and the corrections all still apply.
    guided = asana.key == "vrikshasana"
    if not guided and not args.no_coach:
        args.no_coach = True
        print(f"note: step-by-step guidance is written for Vrikshasana only, so "
              f"{asana.sanskrit} runs in scoring mode (hold timer and "
              f"corrections still active)")

    cap, live = open_source(args.source, args.width, args.height)
    tracker = PoseTracker(model=args.model)
    sm = PoseStateMachine(hold_target_s=asana.hold_target_s,
                          enter_score=asana.enter_score, exit_score=asana.exit_score)
    cues = CueEngine(state_cues=args.no_coach, phrase_fn=phrasing.spoken_phrase)
    coach = Coach(hold_target_s=asana.hold_target_s)
    speaker = Speaker(enabled=not args.no_voice)
    smooth_score = RollingMean(window=5)
    log = None if args.no_db else SessionLog(
        asana.key, source=("webcam" if live else args.source))

    writer = None
    frames = 0
    fps = 30.0
    last_sample = -1.0
    banner = asana.setup_hint
    banner_until = 0.0
    t0 = time.perf_counter()
    prev = t0
    elapsed = 0.0

    print("")
    print(f"{asana.sanskrit} ({asana.name}) - hold target {asana.hold_target_s:.0f}s, "
          f"enter at {asana.enter_score:.0f}%")
    journey = (JOURNEY[:-2] + JOURNEY[-1:]) if guided else JOURNEY[-2:]
    print(f"{len(journey)} steps: " + " -> ".join(s.title for s in journey))
    print(asana.setup_hint)
    print("To finish: press q or Esc with the video window focused, "
          "or close the window.")
    print("")

    # Show the whole practice once before asking anyone to do it.  Quitting
    # during the walkthrough must still release the camera, so this sits
    # inside the same try/finally as the live loop.
    try:
        if not args.no_intro:
            if not run_intro(args, asana, speaker, journey=journey):
                cap.release()
                tracker.close()
                cv2.destroyAllWindows()
                speaker.close()
                if log:
                    log.finish(sm.stats, 0, 0.0)
                print("quit during the walkthrough")
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

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if live and not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 1e-4:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            t = now - t0
            frames += 1

            pose = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   int(t * 1000), fps)

            ev = None
            if pose is not None:
                ev = evaluate(asana, compute_features(pose))
                score = smooth_score(ev.score) if ev.usable else 0.0
                if ev.usable:
                    ev.score = score
                state = sm.update(t, score, ev.usable)
            else:
                smooth_score.reset()
                state = sm.update(t, 0.0, False)

            # The coach walks a beginner into the pose; once they are in it the
            # correction engine takes over.  Coach lines pre-empt corrections
            # so the two never talk over each other.
            said = None
            if not args.no_coach:
                said = coach.update(t, ev, state, sm.elapsed,
                                    speaker_busy=speaker.busy(t))
                if said is not None:
                    banner = said.text
                    banner_until = t + max(BANNER_HOLD_S, speaker.estimate_seconds(said.text))
                    # Never start a new line over one that is still being
                    # spoken - pyttsx3 has no notion of priority, so a second
                    # say() mid-sentence either garbles both or is silently
                    # dropped by the queue.  The banner and the log always
                    # reflect the guidance either way; only the voice waits.
                    if not speaker.busy(t):
                        speaker.say(said.text, t)
                    print(f"[{t:6.1f}s] coach  {said.text}")
                    if log:
                        log.log_cue(t, "coach." + said.step_key, "guide", said.text)

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
                coach_quiet = not coach.resting and not finishing and (
                    args.no_coach or (said is None and coach.finished))
                cue = (cues.update(t, ev, state, speaker_busy=speaker.busy(t))
                       if coach_quiet else None)
                if cue is not None:
                    banner = cue.text
                    banner_until = t + max(BANNER_HOLD_S, speaker.estimate_seconds(cue.text))
                    if not speaker.busy(t):
                        speaker.say(cue.text, t)
                        cues.note_spoken(t, speaker.estimate_seconds(cue.text))
                    print(f"[{t:6.1f}s] {cue.level:<6} {cue.text}")
                    if log:
                        log.log_cue(t, cue.key, cue.level, cue.text)
                if log and ev.usable and t - last_sample >= 0.5:
                    last_sample = t
                    log.log_joints(t, ev.results)

            # The banner is cue-driven, but a cue is deliberately rare - it has
            # to persist a second and clear a cooldown before it is spoken.
            # Without a fallback the last message just sits there, which is how
            # "no body detected" ended up printed over a body being scored.
            if t >= banner_until:
                banner = _idle_banner(ev, state, asana)

            if sm.just_completed:
                last = sm.stats.holds[-1]
                print(f"[{t:6.1f}s] HOLD COMPLETE - {last.duration_s:.1f}s "
                      f"at {last.avg_score:.1f}% average")

            if pose is not None:
                draw_skeleton(frame, pose, ev, show_raw=args.show_raw)
            draw_hud(frame, ev, sm, banner, fps, hint=asana.setup_hint,
                     right_margin=(0 if args.no_guide else 282), metric=args.metric)

            if not args.no_guide:
                draw_framing_panel(frame, ev.features.get("framing") if ev else None)
                if guided:
                    # The practice stages map straight onto the panel; rest is
                    # the last entry.
                    step_index = (len(journey) - 1 if coach.resting
                                  else min(coach.index, len(SEQUENCE) - 1))
                else:
                    # Scoring mode has only the two stages it actually runs.
                    step_index = 1 if coach.resting else 0
                draw_step_panel(frame, journey, step_index)
                if coach.resting:
                    draw_rest_panel(frame, coach.rest_remaining, coach.rest_target_s)
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

            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, 25.0,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):                 # q or Esc
                break
            if window_closed(WINDOW):
                break
            if key == ord("v"):
                speaker.enabled = not speaker.enabled
                print("voice " + ("on" if speaker.enabled else "off"))
            if key == ord("g"):
                args.no_guide = not args.no_guide
                print("guide " + ("off" if args.no_guide else "on"))
            if key == ord("m"):
                args.metric = not args.metric
                print("joint table: " + ("raw numbers" if args.metric else "words"))
            if key == ord("r"):
                sm = PoseStateMachine(hold_target_s=asana.hold_target_s,
                                      enter_score=asana.enter_score,
                                      exit_score=asana.exit_score)
                cues = CueEngine(state_cues=args.no_coach, phrase_fn=phrasing.spoken_phrase)
                coach = Coach(hold_target_s=asana.hold_target_s)
                smooth_score.reset()
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
        cv2.destroyAllWindows()
        elapsed = max(1e-3, time.perf_counter() - t0)
        if log:
            log.finish(sm.stats, frames, frames / elapsed)
        speaker.close()

    st = sm.stats
    print("")
    print("---------------- session summary ----------------")
    print(f"asana              : {asana.sanskrit} ({asana.name})")
    print(f"duration           : {elapsed:.1f}s over {frames} frames "
          f"({frames / elapsed:.1f} fps)")
    print(f"holds attempted    : {len(st.holds)}")
    print(f"holds completed    : {st.completed}")
    print(f"longest hold       : {st.longest_s:.1f}s")
    print(f"total time in pose : {st.total_hold_s:.1f}s")
    print(f"best alignment     : {st.best_score:.1f}%")
    if log:
        print(f"session logged to  : {log.path} (session id {log.session_id})")
    print("-------------------------------------------------")
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
        cv2.destroyAllWindows()
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
