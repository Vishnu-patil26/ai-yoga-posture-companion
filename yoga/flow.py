"""One whole practice of a pose, as a single sequence, for the nine fitted poses.

    1  Stand straight     whole body in the picture, arms by your sides
    2  Turn               sideways (or face the camera, for poses the camera reads from the front)
    3  Get into the pose  slowly, one step at a time; each step waits for your body (yoga/guide.py)
    4  Hold               for the hold time (10 seconds); the corrections speak now, and only now
    5  Angle check        arms, legs, spine and neck, said out loud and shown on screen
    6  Release and breathe  come out, stand straight, three slow breaths, then it is over

When it ends the camera window offers the next pose (n) or the same one again (r).

This module is pure logic: it is given the evaluation, the hold state and the clock, and says what
should be spoken.  It touches no camera and no voice, so the order and the timing can be tested
without either.  Tree has its own, longer coach (yoga/coach.py) and does not use this.

Nothing here changes how a pose is scored.  The hold timer is simply not allowed to start until the
steps are done (``hold_open``), and the angle check is built from the per-joint results the scorer
already produces while the pose was held.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import posespecs
from .feedback import estimate_speech_seconds
from .guide import StepGuide
from .state_machine import State

HOLD_S = 10                      #: seconds to hold each pose in this sequence

#: how far the body must have turned.  frontality = shoulder width / torso length.
SIDE_MAX = 0.35
FRONT_MIN = 0.45

SETTLE_S = 1.2                   #: a check must hold this long to count
TIMEOUT_S = 15.0                 #: nobody is trapped on a step the camera cannot confirm
HOLD_PATIENCE_S = 120.0          #: how long to wait for the pose to be held before moving on
REPORT_EXTRA_S = 4.0             #: time to read the angle check after it has been said

PHASES = ("stand", "turn", "steps", "hold", "check", "release")

#: which part of the body each measurement belongs to, for the angle check
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Arms", ("arm_raise_high", "arm_raise_low", "elbow_straight", "elbow_bent", "elbow_angle_vis",
              "arm_torso_vis", "arm_torso_min", "arm_torso_max")),
    ("Legs", ("knee_bent", "knee_straight", "knee_angle_vis", "thigh_level", "stance_width")),
    ("Spine and hips", ("spine_tilt", "hip_angle_min", "hip_angle_max", "hip_angle_vis", "hip_rise")),
    ("Neck and head", ("neck_dev", "head_drop")),
)

KIND = {"tadasana": "stand", "trikonasana": "stand", "virabhadrasana": "stand", "utkatasana": "stand",
        "adho_mukha": "floor", "bhujangasana": "floor", "marjaryasana": "floor", "balasana": "floor",
        "sukhasana": "seat"}

_STAND = {
    "stand": "First, stand up straight. Feet together, arms relaxed by your sides, and your whole body in the picture, head to feet.",
    "floor": "First, stand up straight at the end of your mat, with your whole body in the picture, head to feet.",
    "seat": "First, stand up straight in front of where you will sit, with your whole body in the picture, head to feet.",
}
_TURN_SIDE = "Now turn and stand sideways, so the camera sees your side. Your head, shoulder, hip, knee and ankle make one line."
_TURN_FRONT = "Now face the camera, with your shoulders square to it, so it can see both arms and both legs."
_BEGIN = "Good. Now get into the pose slowly, one step at a time."
_RELEASE = "Slowly come out of the pose. Stand up straight with your arms by your sides."

#: the breathing after the pose: (line, seconds to wait after it has been said)
_BREATHS: tuple[tuple[str, float], ...] = (
    ("Now breathe in slowly through your nose.", 3.0),
    ("And breathe out slowly.", 3.0),
    ("Breathe in.", 3.0),
    ("And out.", 3.0),
    ("One more time. Breathe in.", 3.0),
    ("And out. Let everything relax.", 2.0),
)
_END = "Well done. This pose is finished. Press n for the next pose, or r to repeat this one."


@dataclass(frozen=True)
class Phase:
    title: str                   #: what the step panel shows


class PoseFlow:
    def __init__(self, key: str, asana, steps: list[tuple], hold_s: float = HOLD_S) -> None:
        self.key = key
        self.hold_s = max(1, int(round(hold_s)))
        self.view = posespecs.VIEW.get(key, "front")
        self.kind = KIND.get(key, "stand")
        self.guide = StepGuide(asana, steps, settle_s=SETTLE_S, timeout_s=TIMEOUT_S)
        turn_title = "Turn sideways" if self.view == "side" else "Face the camera"
        self.journey = [Phase("Stand straight"), Phase(turn_title), Phase("Get into the pose"),
                        Phase(f"Hold {self.hold_s} seconds"), Phase("Angle check"),
                        Phase("Release and breathe")]
        self.index = 0
        self.finished = False
        self.finished_at: float | None = None
        self.report: list[dict] | None = None
        self.avg = 0.0
        self.last_line = ""
        self._started = False
        self._quiet_until = 0.0              #: a sentence is still being said until then
        self._t0: float | None = None
        self._ok_since: float | None = None
        self._stats: dict[str, dict] = {}
        self._sched: list[tuple[float, str]] = []
        self._next_at = 0.0

    # ------------------------------------------------------------------ what the screen needs
    @property
    def phase(self) -> str:
        return PHASES[min(self.index, len(PHASES) - 1)]

    @property
    def hold_open(self) -> bool:
        """The hold timer may count only now: after the steps, before the angle check."""
        return not self.finished and self.phase == "hold"

    @property
    def cues_allowed(self) -> bool:
        """Spoken corrections belong to the hold; every other phase has its own script."""
        return self.hold_open

    @property
    def show_report(self) -> bool:
        return self.phase in ("check", "release") and bool(self.report)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(s.label for s in self.guide.steps)

    @property
    def step_current(self) -> int | None:
        """Which pose step to highlight on the card; None before the steps begin."""
        if self.index < PHASES.index("steps"):
            return None
        return min(self.guide.index, len(self.guide.steps) - 1)

    @property
    def steps_done(self) -> bool:
        return self.index > PHASES.index("steps")

    # ------------------------------------------------------------------ the sequence
    def _say(self, t: float, line: str) -> str:
        self.last_line = line
        self._quiet_until = t + estimate_speech_seconds(line)
        return line

    def _next_phase(self) -> None:
        self.index += 1
        self._t0 = None
        self._ok_since = None

    def _gate(self, t: float, ok: bool) -> bool:
        """True once ``ok`` has held for the settle time, or the step has run out of time."""
        if self._t0 is None:
            self._t0 = t
            self._ok_since = None
        if ok:
            if self._ok_since is None:
                self._ok_since = t
        else:
            self._ok_since = None
        return ((self._ok_since is not None and t - self._ok_since >= SETTLE_S)
                or t - self._t0 >= TIMEOUT_S)

    def update(self, t: float, ev, state: State | None = None, completed: bool = False,
               hold_avg: float = 0.0) -> str | None:
        """Advance if the moment has come.  Returns the line to say, or None."""
        if self.finished:
            return None
        self._observe(ev, state)
        if not self._started:
            self._started = True
            return self._say(t, _STAND[self.kind])
        if t < self._quiet_until:
            return None                                  # the last sentence is still being said
        ph = self.phase

        if ph == "stand":
            framing = ev.features.get("framing") if ev is not None else None
            if self._gate(t, bool(framing and framing.get("ok"))):
                self._next_phase()
                return self._say(t, _TURN_SIDE if self.view == "side" else _TURN_FRONT)

        elif ph == "turn":
            fr = float(ev.features.get("frontality", 0.0)) if ev is not None else None
            ok = fr is not None and (fr <= SIDE_MAX if self.view == "side" else fr >= FRONT_MIN)
            if self._gate(t, ok):
                self._next_phase()
                return self._say(t, _BEGIN)

        elif ph == "steps":
            spoken = self.guide.update(t, ev)
            if self.guide.finished:
                self._next_phase()
                return self._say(t, f"Now hold the pose for {self.hold_s} seconds. Keep breathing slowly.")
            if spoken:
                return self._say(t, spoken)

        elif ph == "hold":
            if self._t0 is None:
                self._t0 = t
            if completed:
                self.avg = hold_avg
                self.report = self._build_report()
                self._next_phase()
                return self._say(t, self._report_line())
            if t - self._t0 >= HOLD_PATIENCE_S:          # never held: do not wait for ever
                self.report = self._build_report()
                self._next_phase()
                if self.report:
                    return self._say(t, self._report_line())
                self._next_phase()
                self._begin_release(t)
                return self._say(t, "That was not held long enough to check. Come out of the pose slowly. "
                                    "Press r to try it again, or n for the next pose.")

        elif ph == "check":
            if self._t0 is None:
                self._t0 = t
            if t - self._t0 >= REPORT_EXTRA_S:
                self._next_phase()
                self._begin_release(t)
                return self._say(t, _RELEASE)

        elif ph == "release":
            if t >= self._next_at and self._sched:
                wait, line = self._sched.pop(0)
                self._next_at = t + estimate_speech_seconds(line) + wait
                return self._say(t, line)
            if not self._sched and t >= self._next_at:
                self.finished, self.finished_at = True, t
                return self._say(t, _END)
        return None

    def _begin_release(self, t: float) -> None:
        self._sched = [(w, ln) for ln, w in _BREATHS]
        self._next_at = t + estimate_speech_seconds(_RELEASE) + 3.0

    # ------------------------------------------------------------------ the angle check
    def _observe(self, ev, state) -> None:
        """Collect how each joint did while the pose was actually being held."""
        if self.phase != "hold" or state is not State.HOLDING or ev is None or not ev.usable:
            return
        for r in ev.results:
            if r.ok is None:
                continue
            s = self._stats.setdefault(r.check.key, {"n": 0, "ok": 0, "cues": {}})
            s["n"] += 1
            if r.ok:
                s["ok"] += 1
            else:
                cue = r.cue()
                s["cues"][cue] = s["cues"].get(cue, 0) + 1

    def _build_report(self) -> list[dict]:
        rows: list[dict] = []
        for name, keys in GROUPS:
            have = [self._stats[k] for k in keys if k in self._stats and self._stats[k]["n"] >= 3]
            if not have:
                continue
            n = sum(s["n"] for s in have)
            frac = sum(s["ok"] for s in have) / n
            cue = ""
            if frac < 0.85:
                worst = min(have, key=lambda s: s["ok"] / s["n"])
                cues = worst["cues"]
                cue = max(cues, key=cues.get) if cues else ""
            verdict = "good" if frac >= 0.85 else ("nearly" if frac >= 0.6 else "fix")
            rows.append({"group": name, "verdict": verdict, "pct": round(100 * frac), "cue": cue})
        return rows

    def _report_line(self) -> str:
        parts = ["Angle check."]
        for r in self.report or ():
            if r["verdict"] == "good":
                parts.append(f"{r['group']}: good.")
            else:
                parts.append(f"{r['group']}: {r['cue'].rstrip('.')}." if r["cue"] else f"{r['group']}: a little off.")
        if self.avg:
            parts.append(f"Overall, {self.avg:.0f} percent.")
        return " ".join(parts)
