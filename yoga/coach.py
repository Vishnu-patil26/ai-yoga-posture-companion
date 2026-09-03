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
    say: str                     #: the whole instruction, for the screen
    done: Callable[[dict], bool]  #: measured condition to advance
    #: The instruction broken into separate spoken lines.  A three-clause
    #: sentence said in one breath is hard to follow while you are trying to
    #: balance; delivered a line at a time, each one waiting for the previous
    #: to finish, it becomes something you can actually act on.  Defaults to
    #: the whole sentence as a single line.
    lines: tuple[str, ...] = ()
    #: Seconds the condition must hold before the step is counted as done.
    #: Long enough to be a real confirmation the position is settled, not a
    #: momentary flicker - and it is shown on screen as it fills.
    dwell: float = 1.5
    repeat_s: float = 9.0        #: re-speak if the practitioner is stuck
    #: Short label shown while the position is being confirmed.
    brief: str = ""

    def spoken_lines(self) -> tuple[str, ...]:
        return self.lines or (self.say,)


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


def _arms_overhead(f: dict) -> bool:
    """Both arms reaching up."""
    return max(f.get("arm_raise_left", 90.0), f.get("arm_raise_right", 90.0)) < 50.0


def _arms_down(f: dict) -> bool:
    """Arms brought back down - to the heart, or lowered to the sides.

    This is the position the practice keeps returning to, and it had no
    detector at all: `_hands_placed` only ever asked "are the arms up OR at
    the heart", so once they were up there was nothing that could recognise
    them coming down again.  Both resting positions count, but drifting
    halfway does not - the wrists have to actually arrive somewhere.
    """
    wl = f.get("wrist_to_chest_left", 9.0)
    wr = f.get("wrist_to_chest_right", 9.0)
    at_heart = max(wl, wr) < HEART_MAX
    by_sides = min(f.get("arm_raise_left", 0.0),
                   f.get("arm_raise_right", 0.0)) > 120.0
    return at_heart or by_sides


def _hands_placed(f: dict) -> bool:
    """Either taught arm form counts - overhead, or pressed at the heart.

    Arms simply hanging by the sides must count as neither, which is why the
    heart test is on wrist-to-chest distance rather than the arm angle: hands
    at the sternum and hands at the hips both point downwards.
    """
    wl = f.get("wrist_to_chest_left", 9.0)
    wr = f.get("wrist_to_chest_right", 9.0)
    return _arms_overhead(f) or max(wl, wr) < HEART_MAX


def _balanced(f: dict) -> bool:
    """Still standing in the pose - the shape is held, whatever the arms do."""
    return _foot_placed(f) and f.get("standing_knee", 0.0) > 150.0


# The practice, as it is actually taught: settle into the shape, hold it
# steady, take the arms up, close the eyes, count, then bring the arms down
# and up and down again before resting.  Each stage waits for the body to
# reach it - `dwell` is how long the position must be held, so a stage that
# is really "stay here and count to ten" is simply a long dwell rather than a
# special case.
SEQUENCE: tuple[Step, ...] = (
    Step("frame", "Get in frame",
         "Step back until your whole body, head to feet, is in the frame.",
         _framing_ok,
         lines=("Step back from the camera.",
                "I need to see you from head to feet."),
         dwell=1.5),
    Step("stand", "Stand tall",
         "Stand tall with your feet together, facing the camera.",
         _standing_tall,
         lines=("Stand tall, feet together.", "Face the camera."),
         dwell=1.5),
    Step("weight", "Shift your weight",
         "Now shift all your weight onto one foot.",
         _weight_shifted,
         lines=("Now shift your weight.", "All of it onto one foot."),
         dwell=1.2),
    Step("foot", "Foot to the thigh",
         "Bend the other knee, open it out to the side, "
         "and press that foot into your inner thigh.",
         _foot_placed,
         lines=("Bend your other knee.",
                "Open that knee out to the side.",
                "Press the foot into your inner thigh."),
         dwell=1.5),
    Step("steady", "Find your balance",
         "Stay there and find your balance.",
         _balanced,
         lines=("Good. Stay there.", "Find your balance and breathe."),
         dwell=6.0, repeat_s=12.0),
    Step("hands", "Hands up",
         "Reach both arms up overhead.",
         _arms_overhead,
         lines=("Now take your hands up.", "Reach both arms overhead."),
         dwell=1.5),
    Step("eyes", "Close your eyes",
         "Close your eyes and keep your balance.",
         _balanced,
         lines=("Now close your eyes.", "Keep your balance."),
         dwell=3.0, repeat_s=12.0),
    Step("count", "Count to ten",
         "Hold it, and count to ten.",
         _balanced,
         lines=("Hold it there.", "Count slowly to ten."),
         dwell=10.0, repeat_s=15.0),
    Step("down1", "Hands down",
         "Bring your hands back down.",
         _arms_down,
         lines=("And bring your hands down.",),
         dwell=1.2),
    Step("up2", "Hands up again",
         "Reach both arms up overhead again.",
         _arms_overhead,
         lines=("Hands up again.",),
         dwell=1.2),
    Step("down2", "Hands down again",
         "And bring your hands down again.",
         _arms_down,
         lines=("And down again.",),
         dwell=1.2),
)


@dataclass(frozen=True)
class JourneyStep:
    """One stage of the whole practice, as a person experiences it."""
    key: str
    title: str
    say: str
    kind: str          #: "entry" (guided in), "hold", or "rest"


#: The complete practice, start to finish - the five guided entry steps plus
#: the hold and the rest that follow them.  `SEQUENCE` above is only the part
#: the coach actively talks you through; holding and resting are states, not
#: instructions, so they live outside it.  Both the on-screen step panel and
#: the demo walkthrough read this one list, so "step 4 of 7" always means the
#: same thing in both places and cannot drift out of sync.
JOURNEY: tuple[JourneyStep, ...] = tuple(
    # The walkthrough is a preview, not the lesson - it uses each step's first
    # spoken line rather than the whole instruction, so it moves at a glance
    # per slide instead of reading a paragraph at somebody who has not started
    # yet.  The full wording arrives in the session itself, when it is useful.
    JourneyStep(s.key, s.title, s.spoken_lines()[0], "entry") for s in SEQUENCE
) + (
    JourneyStep("hold", "Hold the pose", "Hold steady and breathe.", "hold"),
    JourneyStep("rest", "Rest", "Lower your foot and rest.", "rest"),
)

#: Spoken at these many seconds remaining in a hold.  Deliberately sparse:
#: counting 10-5-3-2-1 is five interruptions during the part of the practice
#: that most needs quiet concentration.  A single mid-point marker and the
#: last two beats is enough to know where you are.
COUNTDOWN_AT = (10, 2, 1)

#: Spoken at these many seconds remaining in the rest between attempts.  The
#: rest panel shows a large number on screen the whole time, so the voice only
#: needs to mark the end of it.
REST_SPEAK_AT = (2, 1)

#: A step's condition may flicker false for this long without cancelling the
#: confirmation already in progress - landmark noise, not the practitioner
#: actually leaving the position.
CONFIRM_GRACE_S = 0.4

#: How long the body may be ungradable before the guided run restarts from the
#: first step.  Generous on purpose: a momentary dropout mid-pose should never
#: throw the practitioner back to "step back from the camera".
LOST_GRACE_S = 1.2

#: Corrections stop this many seconds before a hold completes - the last beats
#: belong to the countdown and to finishing the pose, not to a new fault.
QUIET_BEFORE_END_S = 4.0


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

    #: How long to rest between one completed hold and the next attempt.
    #: Long enough to actually recover balance, short enough that a practice
    #: session of several attempts does not mostly consist of waiting.
    rest_target_s: float = 8.0
    resting: bool = False
    rest_remaining: float = 0.0

    #: How far through confirming the current position we are, 0..1.  Drawn as
    #: a filling bar so the practitioner can see the app agreeing with them
    #: rather than wondering whether it noticed.
    confirm_fraction: float = 0.0
    confirming: bool = False
    #: Seconds spent on the current step, for the on-screen step timer.
    step_seconds: float = 0.0

    _since: float | None = None
    _last_said: float = -1e9
    _spoken_countdown: set = field(default_factory=set)
    _rest_until: float | None = None
    _rest_spoken: set = field(default_factory=set)
    _last_state: State | None = None
    _greeted: bool = False
    #: Lines of the current step still to be said, one at a time.
    _pending: list = field(default_factory=list)
    _step_started: float | None = None
    _speaker_busy: bool = False
    _lost_since: float | None = None
    _unusable_since: float | None = None

    # ------------------------------------------------------------------ update
    def update(self, t: float, ev, state: State, elapsed: float = 0.0,
               speaker_busy: bool = False) -> Say | None:
        """Returns a line to speak, or None.  Safe to call every frame.

        `speaker_busy` lets a multi-line instruction be delivered one line at
        a time: the next line is held back until the previous has finished
        being said, instead of both being fired at once and one of them
        dropped.
        """
        self._speaker_busy = speaker_busy
        if not self._greeted:
            self._greeted = True
            self._last_said = t
            self._enter_step(0, t)
            return Say("Let's begin.", SEQUENCE[0].title, "frame")

        entering_hold = state is State.HOLDING and self._last_state is not State.HOLDING
        completing = state is State.COMPLETE and self._last_state is not State.COMPLETE
        self._last_state = state

        # The guided sequence IS the practice, so nothing may cut it short.
        # This used to sit below the HOLDING branch, which meant that the
        # moment the alignment score was good enough the coach abandoned the
        # choreography and announced "hold it" - the practitioner never got
        # past finding their balance, and the arm work never happened at all.
        # While the sequence is running it owns the session; the hold timer
        # and its countdown only take over once the sequence is done.
        sequence_running = not self.finished and self.index < len(SEQUENCE)
        if sequence_running and not self.resting and ev is not None and ev.usable:
            self._unusable_since = None
            return self._advance(t, ev.features)

        if completing:
            return self._on_complete(t)

        if state in (State.HOLDING, State.COMPLETE) or self.resting:
            # The confirmation bar belongs to the guided entry steps only.
            # Left set, it kept drawing "holding position..." over the Hold
            # card, where there is nothing being confirmed.
            self.confirming = False
            self.confirm_fraction = 0.0

        if self.resting:
            if state is State.HOLDING:
                # Already back in a good pose - the rest was only ever a
                # suggestion, never a hard gate, so let them through early
                # rather than making them stand and wait pointlessly.
                self.resting = False
                self._spoken_countdown.clear()
                self.finished = True
                self.step_key, self.title = "hold", "Hold"
                self.instruction = "Hold steady and breathe."
                return Say("Good. Hold it, and breathe.", "Hold", "hold")
            return self._rest_tick(t)

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
            # A frame or two where the body cannot be graded is normal: a leg
            # goes briefly unreadable and `usable` flickers false.  Throwing
            # the practitioner back to step one for that is what made the
            # guidance lurch about mid-pose.  Only a sustained loss counts.
            if self._unusable_since is None:
                self._unusable_since = t
            if t - self._unusable_since < LOST_GRACE_S:
                return None                  # hold position, say nothing
            self.finished = False
            if self.index != 0:
                self._enter_step(0, t)
            self._since = None
            self.confirming = False
            self.confirm_fraction = 0.0
            self.step_key, self.title = "frame", "Get in frame"
            reason = (ev.reason if ev is not None else "Step into view")
            self.instruction = reason
            if t - self._last_said > 6.0:
                self._last_said = t
                return Say(reason, self.title, "frame")
            return None

        self._unusable_since = None
        return self._advance(t, ev.features)

    # ---------------------------------------------------------------- internals
    def _enter_step(self, index: int, t: float) -> None:
        """Move to a step and queue its instruction, a line at a time."""
        self.index = index
        self._since = None
        self._lost_since = None
        self._step_started = t
        self.confirm_fraction = 0.0
        self.confirming = False
        if index < len(SEQUENCE):
            step = SEQUENCE[index]
            self.step_key, self.title = step.key, step.title
            self.instruction = step.say
            self._pending = list(step.spoken_lines())
        else:
            self._pending = []

    def _advance(self, t: float, feats: dict) -> Say | None:
        if self._step_started is None:
            self._step_started = t
        self.step_seconds = t - self._step_started

        # Skip any leading steps the practitioner has already satisfied, so
        # somebody who simply stands up in Tree Pose is not talked through it.
        while self.index < len(SEQUENCE) and SEQUENCE[self.index].done(feats):
            if self._since is None:
                self._since = t
            dwell = SEQUENCE[self.index].dwell
            self.confirming = True
            self.confirm_fraction = min(1.0, (t - self._since) / max(1e-6, dwell))
            if t - self._since < dwell:
                break
            # Position confirmed - only now does the next step begin.
            nxt = self.index + 1
            if nxt >= len(SEQUENCE):
                # The whole practice is done - the arms have come down for the
                # last time.  Go to the rest, rather than immediately asking
                # for another twenty-second hold on a leg that has just worked.
                self._enter_step(nxt, t)
                self.finished = True
                return self._on_complete(t)
            # Enter the next step but do NOT speak here: emitting a line the
            # caller is about to drop (because the voice is mid-sentence)
            # loses it outright.  The pending-drain below is gated on the
            # speaker being free, so the first line comes out the moment it
            # can actually be heard.
            self._enter_step(nxt, t)
            break

        if self.index >= len(SEQUENCE):
            self.finished = True
            return None

        step = SEQUENCE[self.index]
        if step.done(feats):
            self._lost_since = None
        else:
            # Landmark noise makes a borderline condition flicker false for a
            # frame or two.  Resetting the confirmation on the first such frame
            # made the bar fill and collapse repeatedly and the step appear to
            # jump about.  Only a sustained loss actually cancels it.
            if self._lost_since is None:
                self._lost_since = t
            if t - self._lost_since >= CONFIRM_GRACE_S:
                self._since = None
                self.confirming = False
                self.confirm_fraction = 0.0
        self.finished = False
        self.step_key, self.title, self.instruction = step.key, step.title, step.say

        # Deliver the rest of this step's instruction one line at a time, each
        # waiting for the previous to finish being spoken.
        if self._pending and not self._speaker_busy and t - self._last_said >= 0.35:
            self._last_said = t
            return Say(self._pending.pop(0), step.title, step.key)

        # Stuck on this step - say it again, with the specific framing advice
        # if that is what is holding them up.
        if not self._pending and t - self._last_said >= step.repeat_s:
            self._last_said = t
            advice = feats.get("framing", {}).get("advice", "")
            text = advice if (step.key == "frame" and advice) else step.spoken_lines()[0]
            return Say(text, step.title, step.key)
        return None

    def _on_complete(self, t: float) -> Say:
        self.holds_done += 1
        self.index = 0
        self._since = None
        self._last_said = t
        self.suggest_form = 1 - self.suggest_form
        nudge = ("This time, try it with your hands at your heart."
                 if self.suggest_form else
                 "This time, reach both arms overhead.")
        self.instruction = "Lower your foot. " + nudge

        # A timed rest before the next attempt, rather than dropping straight
        # back into "step back until your whole body is in frame" - the
        # practitioner just held a balance pose for the full target and
        # deserves a beat, and a visible countdown means they always know
        # exactly how long that beat is.
        self.resting = True
        self.finished = True
        self.step_key, self.title = "rest", "Rest"
        self._rest_until = t + self.rest_target_s
        self.rest_remaining = self.rest_target_s
        self._rest_spoken.clear()
        # Kept short on purpose: the full version took 8.3 s to say, which ran
        # straight over the start of the rest countdown.  The arm-form nudge
        # stays on screen in `self.instruction` instead of being spoken.
        return Say("Well held. Lower your foot and rest.", "Well held", "done")

    def _rest_tick(self, t: float) -> Say | None:
        remaining = (self._rest_until or t) - t
        self.rest_remaining = max(0.0, remaining)
        self.instruction = f"Resting - next attempt in {int(remaining) + 1}s"
        if remaining <= 0.0:
            # Rest over - begin the whole practice again, not a bare hold.
            # Each round is the same choreography: settle, arms up, eyes
            # closed, count, and the arms down-up-down that finish it.
            self.resting = False
            self._rest_spoken.clear()
            self.finished = False
            self._enter_step(0, t)
            return None                  # the next update() call resumes guiding
        for mark in REST_SPEAK_AT:
            if mark not in self._rest_spoken and remaining <= mark + 0.05:
                self._rest_spoken.add(mark)
                return Say(str(mark), "Rest", "rest")
        return None

    # ------------------------------------------------------------------ display
    def journey_index(self, state: State) -> int:
        """Which of the seven `JOURNEY` stages the practitioner is in now.

        Resting is checked before holding: `COMPLETE` still reads as a hold
        state, but once the rest window has opened the honest answer is
        "resting", which is what the panel and the voice are both doing.
        """
        if self.resting:
            return len(JOURNEY) - 1
        if state in (State.HOLDING, State.COMPLETE):
            return len(JOURNEY) - 2
        return min(self.index, len(SEQUENCE) - 1)

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
        self._pending = []
        self._step_started = None
        self._lost_since = None
        self._unusable_since = None
        self.confirm_fraction = 0.0
        self.confirming = False
        self.step_seconds = 0.0
        self.resting = False
        self.rest_remaining = 0.0
        self._rest_until = None
        self._rest_spoken.clear()
        self._since = None
        self._last_said = -1e9
        self._greeted = False
        self._spoken_countdown.clear()
        self.step_key, self.title = "frame", "Get in frame"
