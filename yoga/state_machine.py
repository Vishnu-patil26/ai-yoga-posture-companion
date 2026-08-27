"""Pose state machine and hold timer (Methodology Stage 4, 'Core Engine').

Wobble is normal in Vrikshasana, so the timer must not reset the instant the
score dips.  The machine below uses hysteresis (a high score to start, a lower
one to stop) plus a grace window, so only a genuine loss of the pose ends the
hold.  Time is passed in rather than read from the clock, which lets the whole
machine be tested offline against a recorded video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    WAITING = "WAITING"     #: no usable body in frame
    SETUP = "SETUP"         #: body seen, pose not good enough yet
    HOLDING = "HOLDING"     #: alignment above threshold, timer running
    COMPLETE = "COMPLETE"   #: full hold achieved, waiting for the user to release


@dataclass
class Hold:
    started_at: float
    duration_s: float
    avg_score: float
    best_score: float
    completed: bool


@dataclass
class SessionStats:
    holds: list[Hold] = field(default_factory=list)
    best_score: float = 0.0

    @property
    def completed(self) -> int:
        return sum(1 for h in self.holds if h.completed)

    @property
    def total_hold_s(self) -> float:
        return sum(h.duration_s for h in self.holds)

    @property
    def longest_s(self) -> float:
        return max((h.duration_s for h in self.holds), default=0.0)


class PoseStateMachine:
    def __init__(self, hold_target_s: float = 20.0, enter_score: float = 78.0,
                 exit_score: float = 62.0, enter_stable_s: float = 0.6,
                 grace_s: float = 1.2, min_logged_hold_s: float = 2.0) -> None:
        self.hold_target_s = hold_target_s
        self.enter_score = enter_score
        self.exit_score = exit_score
        self.enter_stable_s = enter_stable_s
        self.grace_s = grace_s
        self.min_logged_hold_s = min_logged_hold_s

        self.state = State.WAITING
        self.stats = SessionStats()
        self.elapsed = 0.0            #: seconds held in the current attempt
        self.progress = 0.0           #: 0..1 towards hold_target_s

        self._above_since: float | None = None
        self._below_since: float | None = None
        self._hold_start: float | None = None
        self._score_sum = 0.0
        self._score_n = 0
        self._hold_best = 0.0
        self.just_completed = False   #: True for the single tick a hold finishes

    # ------------------------------------------------------------------ update
    def update(self, t: float, score: float, usable: bool) -> State:
        self.just_completed = False

        if not usable:
            if self.state is State.HOLDING:
                self._end_hold(t, completed=False)
            self.state = State.WAITING
            self.elapsed = 0.0
            self.progress = 0.0
            self._above_since = None
            return self.state

        self.stats.best_score = max(self.stats.best_score, score)

        if self.state in (State.WAITING, State.SETUP):
            self.state = State.SETUP
            if score >= self.enter_score:
                self._above_since = t if self._above_since is None else self._above_since
                if t - self._above_since >= self.enter_stable_s:
                    self._begin_hold(t, score)
            else:
                self._above_since = None

        elif self.state is State.HOLDING:
            self._score_sum += score
            self._score_n += 1
            self._hold_best = max(self._hold_best, score)
            self.elapsed = t - (self._hold_start or t)
            self.progress = min(1.0, self.elapsed / self.hold_target_s)

            if score < self.exit_score:
                self._below_since = t if self._below_since is None else self._below_since
                if t - self._below_since >= self.grace_s:
                    self._end_hold(t, completed=False)
                    self.state = State.SETUP
                    self._above_since = None
            else:
                self._below_since = None

            if self.state is State.HOLDING and self.elapsed >= self.hold_target_s:
                self._end_hold(t, completed=True)
                self.state = State.COMPLETE
                self.just_completed = True

        elif self.state is State.COMPLETE:
            # Wait until the practitioner comes out of the pose before re-arming.
            if score < self.exit_score:
                self.state = State.SETUP
                self._above_since = None
                self.elapsed = 0.0
                self.progress = 0.0

        return self.state

    # ------------------------------------------------------------------ private
    def _begin_hold(self, t: float, score: float) -> None:
        self.state = State.HOLDING
        self._hold_start = t
        self._score_sum = score
        self._score_n = 1
        self._hold_best = score
        self._below_since = None
        self.elapsed = 0.0
        self.progress = 0.0

    def _end_hold(self, t: float, completed: bool) -> None:
        if self._hold_start is None:
            return
        duration = t - self._hold_start
        if duration >= self.min_logged_hold_s or completed:
            avg = self._score_sum / max(1, self._score_n)
            self.stats.holds.append(Hold(started_at=self._hold_start, duration_s=duration,
                                         avg_score=avg, best_score=self._hold_best,
                                         completed=completed))
        self._hold_start = None
        self._score_sum = 0.0
        self._score_n = 0
        self._hold_best = 0.0
        if not completed:
            self.elapsed = 0.0
            self.progress = 0.0
