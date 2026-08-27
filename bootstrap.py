#!/usr/bin/env python3
"""Self-healing setup for the AI Yoga Posture & Wellness Companion.

Run this on a clean clone, on any laptop, with the system Python:

    python bootstrap.py

It does not assume anything works.  Every requirement is a *check* paired with
a *repair*, and the whole set is run in a loop: check everything, fix what is
broken, check again.  A repair that fixes two problems at once is fine; a
repair that fails is reported rather than left to surface later as a confusing
traceback.  The loop stops when a pass makes no further progress, so it cannot
spin forever.

It is safe to run repeatedly - every repair is idempotent.

    python bootstrap.py            # set up, repair, verify
    python bootstrap.py --check    # diagnose only, change nothing
    python bootstrap.py --run      # set up, then start the live trainer
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import venv
from dataclasses import dataclass
from typing import Callable

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
MODELS = os.path.join(ROOT, "models")

#: MediaPipe publishes wheels for these; 3.13+ has none at the time of writing.
SUPPORTED = ((3, 9), (3, 10), (3, 11), (3, 12))
PREFERRED = ("3.12", "3.11", "3.10")

MODEL_URLS = {
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}
#: A truncated download still leaves a file on disk, so size is checked too.
MODEL_MIN_BYTES = 1_000_000

MAX_PASSES = 4

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    # Piped to a file, a log or CI: escape codes would just be noise.
    GREEN = RED = YELLOW = DIM = RESET = ""
elif platform.system() == "Windows":
    try:                                    # enable ANSI on legacy consoles
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        GREEN = RED = YELLOW = DIM = RESET = ""


def say(msg: str, colour: str = "") -> None:
    print(f"{colour}{msg}{RESET}", flush=True)


# --------------------------------------------------------------------- helpers
def venv_python() -> str:
    exe = "python.exe" if os.name == "nt" else "python"
    sub = "Scripts" if os.name == "nt" else "bin"
    return os.path.join(VENV, sub, exe)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def find_supported_python() -> str | None:
    """An interpreter MediaPipe actually has wheels for."""
    if sys.version_info[:2] in SUPPORTED:
        return sys.executable
    # Windows ships the `py` launcher; elsewhere look for python3.N on PATH.
    if os.name == "nt" and shutil.which("py"):
        for tag in PREFERRED:
            probe = run(["py", f"-{tag}", "-c", "import sys; print(sys.executable)"])
            if probe.returncode == 0 and probe.stdout.strip():
                return probe.stdout.strip()
    for tag in PREFERRED:
        exe = shutil.which(f"python{tag}")
        if exe:
            return exe
    return None


# ----------------------------------------------------------------------- steps
@dataclass
class Step:
    name: str
    check: Callable[[], bool]
    repair: Callable[[], str]      #: returns "" on success, else the reason
    fatal: bool = True             #: a failure here stops the run
    detail: str = ""


def check_interpreter() -> bool:
    return find_supported_python() is not None


def repair_interpreter() -> str:
    return (f"No supported Python found (this one is "
            f"{sys.version_info.major}.{sys.version_info.minor}).\n"
            "    MediaPipe publishes no wheels for 3.13+, so install Python 3.12\n"
            "    from https://www.python.org/downloads/ (tick 'Add to PATH'),\n"
            "    then run this script again.\n"
            "    Or skip the install entirely and use Docker:  docker compose run --rm verify")


def check_venv() -> bool:
    return os.path.isfile(venv_python())


def repair_venv() -> str:
    base = find_supported_python()
    if base is None:
        return "no supported interpreter to build the environment from"
    if os.path.isdir(VENV):                 # half-made environment
        shutil.rmtree(VENV, ignore_errors=True)
    if base == sys.executable:
        venv.EnvBuilder(with_pip=True, clear=True).create(VENV)
    else:
        r = run([base, "-m", "venv", "--clear", VENV])
        if r.returncode != 0:
            return (r.stderr or r.stdout).strip()[:400]
    return "" if os.path.isfile(venv_python()) else "environment was not created"


#: Importing is not proof that a package works.  Two OpenCV distributions
#: share the `cv2` directory, so removing one leaves a `cv2` that still
#: imports and then has no attributes at all - a break that otherwise
#: surfaces much later as a baffling AttributeError.  So the check *uses*
#: each package rather than merely importing it.
SMOKE_TEST = """
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
assert cv2.__version__
cv2.cvtColor(np.zeros((4, 4, 3), np.uint8), cv2.COLOR_BGR2RGB)
cv2.putText(np.zeros((8, 8, 3), np.uint8), "x", (1, 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((4, 4, 3), np.uint8))
vision.PoseLandmarkerOptions, vision.RunningMode, BaseOptions
"""


def check_packages() -> bool:
    if not os.path.isfile(venv_python()):
        return False
    return run([venv_python(), "-c", SMOKE_TEST]).returncode == 0


def repair_packages() -> str:
    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    req = os.path.join(ROOT, "requirements.txt")

    # A half-removed OpenCV still imports and then has no attributes, because
    # the two distributions share the `cv2` directory.  Clear them out before
    # reinstalling so exactly one build owns it.  A cv2 that is simply absent
    # needs no such surgery - only say so when it is genuinely broken.
    importable = run([py, "-c", "import cv2"]).returncode == 0
    usable = importable and run([py, "-c", "import cv2; cv2.__version__"]).returncode == 0
    if importable and not usable:
        say("      cv2 imports but is broken - reinstalling OpenCV cleanly", DIM)
        run([py, "-m", "pip", "uninstall", "-y", "-q", "opencv-python",
             "opencv-contrib-python", "opencv-python-headless"])

    r = run([py, "-m", "pip", "install", "-r", req], timeout=1800)
    if r.returncode == 0 and check_packages():
        return ""

    # Second attempt: force fresh wheels rather than anything cached or partial.
    say("      retrying with --force-reinstall --no-cache-dir", DIM)
    r = run([py, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir",
             "-r", req], timeout=1800)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
        return "pip install failed:\n      " + "\n      ".join(tail)
    return ""


def check_models() -> bool:
    return all(
        os.path.isfile(os.path.join(MODELS, n))
        and os.path.getsize(os.path.join(MODELS, n)) > MODEL_MIN_BYTES
        for n in MODEL_URLS)


def repair_models() -> str:
    os.makedirs(MODELS, exist_ok=True)
    problems = []
    for name, url in MODEL_URLS.items():
        path = os.path.join(MODELS, name)
        if os.path.isfile(path) and os.path.getsize(path) > MODEL_MIN_BYTES:
            continue
        if os.path.isfile(path):            # truncated from a previous attempt
            os.remove(path)
        tmp = path + ".part"
        for attempt in (1, 2, 3):
            try:
                say(f"      downloading {name} (attempt {attempt})", DIM)
                urllib.request.urlretrieve(url, tmp)
                if os.path.getsize(tmp) <= MODEL_MIN_BYTES:
                    raise OSError(f"only {os.path.getsize(tmp)} bytes")
                os.replace(tmp, path)
                break
            except (urllib.error.URLError, OSError) as exc:
                if os.path.isfile(tmp):
                    os.remove(tmp)
                if attempt == 3:
                    problems.append(f"{name}: {exc}")
    return "; ".join(problems)


def check_engine() -> bool:
    if not os.path.isfile(venv_python()):
        return False
    r = run([venv_python(), os.path.join("tools", "selftest.py")], timeout=600)
    return r.returncode == 0


def repair_engine() -> str:
    r = run([venv_python(), os.path.join("tools", "selftest.py")], timeout=600)
    failing = [ln.strip() for ln in (r.stdout or "").splitlines() if "[FAIL]" in ln]
    if failing:
        return "self-test failures (this is a code problem, not a setup one):\n      " \
            + "\n      ".join(failing[:6])
    return ((r.stderr or r.stdout).strip().splitlines() or ["self-test did not run"])[-1]


STEPS = [
    Step("Python interpreter", check_interpreter, repair_interpreter,
         detail="MediaPipe needs CPython 3.9-3.12"),
    Step("Virtual environment", check_venv, repair_venv, detail=".venv"),
    Step("Dependencies", check_packages, repair_packages,
         detail="mediapipe, opencv, numpy, pyttsx3"),
    Step("Pose model", check_models, repair_models,
         detail="BlazePose .task files (~15 MB)"),
    Step("Engine self-test", check_engine, repair_engine,
         detail="66 checks, no camera needed"),
]


# ------------------------------------------------------------------- the loop
def doctor(fix: bool = True) -> int:
    say("")
    say("AI Yoga Posture & Wellness Companion - environment check")
    say(f"  {platform.system()} {platform.release()}   "
        f"python {platform.python_version()}   {ROOT}", DIM)
    say("")

    remaining = list(STEPS)
    for pass_no in range(1, MAX_PASSES + 1):
        if not remaining:
            break
        if pass_no > 1:
            say(f"  -- re-checking after repairs (pass {pass_no}) --", DIM)

        still_broken: list[tuple[Step, str]] = []
        progress = False
        for step in remaining:
            if step.check():
                say(f"  [ OK ] {step.name}", GREEN)
                progress = True
                continue
            if not fix:
                say(f"  [FAIL] {step.name}  {DIM}{step.detail}{RESET}", RED)
                still_broken.append((step, "not checked - diagnosis only"))
                continue

            say(f"  [ .. ] {step.name} - repairing  {DIM}{step.detail}{RESET}", YELLOW)
            try:
                reason = step.repair()
            except Exception as exc:                    # never crash the doctor
                reason = f"{type(exc).__name__}: {exc}"
            if not reason and step.check():
                say(f"  [ OK ] {step.name}  {DIM}(repaired){RESET}", GREEN)
                progress = True
            else:
                say(f"  [FAIL] {step.name}", RED)
                for line in (reason or "still failing after repair").splitlines():
                    say(f"         {line}", RED)
                still_broken.append((step, reason))

        remaining = [s for s, _ in still_broken]
        if not remaining or not progress:
            break

    say("")
    if not remaining:
        say("  Everything is ready.", GREEN)
        say("")
        say("  Next:")
        say(f"    {os.path.relpath(venv_python(), ROOT)} app.py            # live trainer (webcam)")
        say(f"    {os.path.relpath(venv_python(), ROOT)} tools/selftest.py # verify, no camera")
        say("")
        return 0

    fatal = [s for s in remaining if s.fatal]
    say(f"  {len(remaining)} problem(s) could not be fixed automatically:", RED)
    for step in remaining:
        say(f"    - {step.name}", RED)
    say("")
    say("  Nothing was left half-done; re-run this script after resolving the above.", DIM)
    say("  A container avoids the whole question:  docker compose run --rm verify", DIM)
    say("")
    return 1 if fatal else 0


#: Tasks reachable once the environment is healthy.  Routing them through here
#: means every platform uses the same command and none of them has to know
#: where the virtual environment put its interpreter.
TASKS = {
    "run": ["app.py"],
    "calibrate": ["app.py", "--calibrate"],
    "test": [os.path.join("tools", "selftest.py")],
    "demo": [os.path.join("tools", "simulate_session.py")],
    "report": [os.path.join("tools", "report.py")],
    "frames": [os.path.join("tools", "demo_frames.py")],
    "dataset": [os.path.join("tools", "get_dataset.py")],
    "fit": [os.path.join("tools", "fit_reference.py"),
            os.path.join("data", "datasets", "yoga_poses", "train", "tree")],
    "validate": [os.path.join("tools", "validate_dataset.py"),
                 os.path.join("data", "datasets", "yoga_poses", "test")],
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Self-healing setup, doctor and task runner",
        epilog="tasks: " + ", ".join(f"--{t}" for t in TASKS))
    ap.add_argument("--check", action="store_true",
                    help="diagnose only, change nothing")
    for task in TASKS:
        ap.add_argument(f"--{task}", action="store_true",
                        help=f"set up if needed, then run: {' '.join(TASKS[task])}")
    args, extra = ap.parse_known_args(argv)

    wanted = [t for t in TASKS if getattr(args, t)]
    if len(wanted) > 1:
        say(f"  pick one task, not {len(wanted)}", RED)
        return 2

    code = doctor(fix=not args.check)
    if code != 0 or not wanted:
        return code

    task = wanted[0]
    say(f"  starting: {task}", DIM)
    cmd = [venv_python()] + [os.path.join(ROOT, p) for p in TASKS[task][:1]] \
        + TASKS[task][1:] + extra
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
