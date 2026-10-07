"""Step-by-step walk-in for every pose except Vrikshasana (which has its own Coach).

Each step is an instruction plus the measurements that must be right before it
counts as done (yoga/posespecs.py STEPS).  A step completes when every one of
its measurements has been inside tolerance for ``settle_s`` seconds in a row -
so the person is never told the next thing until their body has actually
reached the last one - or after ``timeout_s`` so that a measurement the camera
cannot see well (or a fit that is too strict for this person) never traps them.

The guide only reads the same ``Evaluation`` the scorer already produces, so it
adds no extra vision work and cannot disagree with the score.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    text: str
    keys: tuple[str, ...]          #: measurements that must be inside tolerance


class StepGuide:
    def __init__(self, asana, steps: list[tuple[str, tuple[str, ...]]],
                 settle_s: float = 1.2, min_dwell_s: float = 3.0,
                 timeout_s: float = 25.0) -> None:
        present = {c.key for c in asana.checks}
        for v in asana.variants:
            present |= {c.key for c in v.overrides}
        # Keys this pose does not measure are dropped, never required.
        self.steps = [Step(text, tuple(k for k in keys if k in present)) for text, keys in steps]
        self.settle_s, self.min_dwell_s, self.timeout_s = settle_s, min_dwell_s, timeout_s
        self.index = 0
        self.finished = not self.steps
        self._started: float | None = None
        self._ok_since: float | None = None

    @property
    def current(self) -> Step | None:
        return None if self.finished else self.steps[self.index]

    def update(self, t: float, ev) -> str | None:
        """Advance if the current step is satisfied.  Returns text to say when a
        new instruction (or the closing line) begins, otherwise None."""
        if self.finished:
            return None
        if self._started is None:                      # very first call: speak step 1
            self._started = t
            return self.steps[0].text
        step = self.steps[self.index]
        results = {r.check.key: r.ok for r in ev.results} if ev is not None and ev.usable else None
        ok = results is not None and all(results.get(k) is not False for k in step.keys)
        if ok:
            self._ok_since = self._ok_since if self._ok_since is not None else t
        else:
            self._ok_since = None
        elapsed = t - self._started
        done = (ok and self._ok_since is not None and t - self._ok_since >= self.settle_s
                and elapsed >= self.min_dwell_s)
        if not done and elapsed < self.timeout_s:
            return None
        self.index += 1
        self._started, self._ok_since = t, None
        if self.index >= len(self.steps):
            self.finished = True
            return "Good. Now hold it."
        return self.steps[self.index].text
