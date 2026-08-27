"""Guided entry into the asana - the spoken instructor.

The scorer can tell you *what is wrong*, but it cannot get a beginner into the
pose in the first place: telling somebody who is standing still to "open the
lifted knee out to the side" is meaningless when there is no lifted knee yet.

So before the correction engine takes over, a short scripted sequence walks the
practitioner in, one instruction at a time, and each step **waits for the body
to actually do it** before moving on - the condition is measured from the same
features the scorer uses, not from a timer.  Nobody is left behind, and nobody
who is already in the pose has to sit through the whole script: a step whose
condition is already satisfied is passed straight through.

Every step also names a figure for the on-screen reference card, so the
practitioner can see the shape as well as hear it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .state_machine import State


@dataclass(frozen=True)
class Step:
    key: str                     #: also selects the reference-card figure
    title: str                   #: short label on the card
    say: str                     #: spoken instruction
    done: Callable[[dict], bool]  #: measured condition to advance
    dwell: float = 0.5           #: seconds the condition must hold
    repeat_s: float = 9.0        #: re-speak if the practitioner is stuck


def _framing_ok(f: dict) -> bool:
    return bool(f.get("framing", {}).get("ok"))


def _standing_tall(f: dict) -> bool:
    kl, kr = f.get("knee_left", 0.0), f.get("knee_right", 0.0)
    return _framing_ok(f) and min(kl, kr) > 160.0 and f.get("spine_tilt", 99) < 14.0


def _weight_shifted(f: dict) -> bool:
    kl, kr = f.get("knee_left", 0.0), f.get("knee_right", 0.0)
    return abs(kl - kr) > 25.0 or f.get("foot_height_ratio", 0.0) > 0.15


def _foot_placed(f: dict) -> bool:
    return f.get("foot_height_ratio", 0.0) > 0.55 and f.get("folded_thigh_open", 0.0) > 30.0


#: Wrist-to-chest distance, in torso lengths, below which the palms count as
#: drawn in to the heart.  Fitted: 0.26 median at the heart (p90 0.45) against
#: 1.06 for the overhead form (p10 0.65).
HEART_MAX = 0.55


def _hands_placed(f: dict) -> bool:
    """Either taught arm form counts - overhead, or pressed at the heart.

    Arms simply hanging by the sides must count as neither, which is why the
    heart test is on wrist-to-chest distance rather than the arm angle: hands
    at the sternum and hands at the hips both point downwards.
    """
    left = f.get("arm_raise_left", 90.0)
    right = f.get("arm_raise_right", 90.0)
    overhead = max(left, right) < 45.0
    wl = f.get("wrist_to_chest_left", 9.0)
    wr = f.get("wrist_to_chest_right", 9.0)
    heart = max(wl, wr) < HEART_MAX
    return overhead or heart


SEQUENCE: tuple[Step, ...] = (
    Step("frame", "Get in frame",
         "Step back until your whole body, head to feet, is in the frame.",
         _framing_ok, dwell=0.8),
    Step("stand", "Stand tall",
         "Stand tall with your feet together, facing the camera.",
         _standing_tall, dwell=0.8),
    Step("weight", "Shift your weight",
         "Now shift all your weight onto one foot.",
         _weight_shifted, dwell=0.4),
    Step("foot", "Foot to the thigh",
         "Bend the other knee, open it out to the side, "
         "and press that foot into your inner thigh.",
         _foot_placed, dwell=0.4),
    Step("hands", "Hands",
         "Bring your palms together at your heart, or reach both arms overhead.",
         _hands_placed, dwell=0.4),
)

#: Spoken at these many seconds remaining in a hold.
COUNTDOWN_AT = (10, 5, 3, 2, 1)


@dataclass
class Say:
    """Something for the coach to speak and show."""
    text: str
    screen: str = ""
    step_key: str = "hold"
    priority: bool = True        #: coach lines pre-empt correction cues


@dataclass
class Coach:
    """Walks the practitioner into the asana, then hands over to the scorer."""

    hold_target_s: float = 20.0
    index: int = 0
    finished: bool = False       #: sequence done - corrections take over
    step_key: str = "frame"
    title: str = "Get in frame"
    instruction: str = ""
    holds_done: int = 0
    #: Which arm form to suggest next time, so the practitioner is nudged to
    #: try both rather than repeating the one they find easy.
    suggest_form: int = 0

    _since: float | None = None
    _last_said: float = -1e9
    _spoken_countdown: set = field(default_factory=set)
    _last_state: State | None = None
    _greeted: bool = False

    # ------------------------------------------------------------------ update
    def update(self, t: float, ev, state: State, elapsed: float = 0.0) -> Say | None:
        """Returns a line to speak, or None.  Safe to call every frame."""
        if not self._greeted:
            self._greeted = True
            self._last_said = t
            self.instruction = SEQUENCE[0].say
            return Say("Let's begin. " + SEQUENCE[0].say, SEQUENCE[0].title, "frame")

        entering_hold = state is State.HOLDING and self._last_state is not State.HOLDING
        completing = state is State.COMPLETE and self._last_state is not State.COMPLETE
        self._last_state = state

        if completing:
            return self._on_complete(t)

        if state is State.HOLDING:
            self.finished = True
            self.step_key, self.title = "hold", "Hold"
            if entering_hold:
                self._spoken_countdown.clear()
                self.instruction = "Hold steady and breathe."
                return Say("Good. Hold it, and breathe.", "Hold", "hold")
            return self.countdown_line(elapsed)

        if state is State.COMPLETE:
            return None

        # Back out of the pose (or never in it) - resume guiding.
        if ev is None or not ev.usable:
            self.finished = False
            self.index = 0
            self._since = None
            self.step_key, self.title = "frame", "Get in frame"
            reason = (ev.reason if ev is not None else "Step into view")
            self.instruction = reason
            if t - self._last_said > 6.0:
                self._last_said = t
                return Say(reason, self.title, "frame")
            return None

        return self._advance(t, ev.features)

    # ---------------------------------------------------------------- internals
    def _advance(self, t: float, feats: dict) -> Say | None:
        # Skip any leading steps the practitioner has already satisfied, so
        # somebody who simply stands up in Tree Pose is not talked through it.
        while self.index < len(SEQUENCE) and SEQUENCE[self.index].done(feats):
            if self._since is None:
                self._since = t
            if t - self._since < SEQUENCE[self.index].dwell:
                break
            self.index += 1
            self._since = None
            if self.index >= len(SEQUENCE):
                self.finished = True
                self.step_key, self.title = "hold", "Hold"
                self.instruction = "Find your balance and hold."
                self._last_said = t
                return Say("That's it. Find your balance and hold.", "Hold", "hold")
            step = SEQUENCE[self.index]
            self.step_key, self.title, self.instruction = step.key, step.title, step.say
            self._last_said = t
            return Say(step.say, step.title, step.key)

        if self.index >= len(SEQUENCE):
            self.finished = True
            return None

        step = SEQUENCE[self.index]
        if not step.done(feats):
            self._since = None
        self.finished = False
        self.step_key, self.title, self.instruction = step.key, step.title, step.say

        # Stuck on this step - say it again, with the specific framing advice
        # if that is what is holding them up.
        if t - self._last_said >= step.repeat_s:
            self._last_said = t
            advice = feats.get("framing", {}).get("advice", "")
            text = advice if (step.key == "frame" and advice) else step.say
            return Say(text, step.title, step.key)
        return None

    def _on_complete(self, t: float) -> Say:
        self.holds_done += 1
        self.finished = True
        self.step_key, self.title = "done", "Well held"
        self.index = 0
        self._since = None
        self._last_said = t
        self.suggest_form = 1 - self.suggest_form
        nudge = ("This time, try it with your hands at your heart."
                 if self.suggest_form else
                 "This time, reach both arms overhead.")
        self.instruction = "Lower your foot. " + nudge
        return Say("Well held. Lower your foot slowly, and stand on the other leg. "
                   + nudge, "Well held", "done")

    # ------------------------------------------------------------------ display
    def countdown_line(self, elapsed: float) -> Say | None:
        """Spoken countdown over the last seconds of a hold."""
        remaining = self.hold_target_s - elapsed
        for mark in COUNTDOWN_AT:
            if mark not in self._spoken_countdown and remaining <= mark + 0.05:
                self._spoken_countdown.add(mark)
                word = f"{mark} seconds" if mark >= 5 else str(mark)
                return Say(word, "Hold", "hold")
        return None

    def reset(self) -> None:
        self.index = 0
        self.finished = False
        self._since = None
        self._last_said = -1e9
        self._greeted = False
        self._spoken_countdown.clear()
        self.step_key, self.title = "frame", "Get in frame"
