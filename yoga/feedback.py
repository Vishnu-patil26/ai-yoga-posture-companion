"""Corrective feedback: what to say, when to say it, and saying it aloud.

Methodology Stage 3 'Correction Logic' + Stage 4 'Intelligent Feedback'.

Two rules keep the coaching usable rather than nagging:
  * a deviation must *persist* before it is spoken, so a wobble is not flagged;
  * each cue has its own cooldown, and repeating the same fault escalates the
    wording from a soft reminder to a firm one.

`phrase_fn` is the hook where the LLM voice-guru of Stage 4 plugs in: give it a
callable that turns the measured deviation into a sentence and the engine will
use that instead of the canned string.  The base implementation runs fully
offline with the canned strings, so the demo never depends on a network call.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from .evaluator import CheckResult, Evaluation
from .state_machine import State


@dataclass
class Cue:
    text: str
    key: str
    level: str          #: "soft" | "firm" | "info" | "praise"
    spoken_at: float


class CueEngine:
    def __init__(self, persist_s: float = 1.0, cue_cooldown_s: float = 7.0,
                 global_cooldown_s: float = 3.0, firm_after: int = 2,
                 phrase_fn=None, state_cues: bool = True) -> None:
        #: When the guided coach is running it owns the "hold it" / "well held"
        #: announcements; this engine then emits corrections only, so the two
        #: never say the same thing one frame apart.
        self.state_cues = state_cues
        self.persist_s = persist_s
        self.cue_cooldown_s = cue_cooldown_s
        self.global_cooldown_s = global_cooldown_s
        self.firm_after = firm_after
        self.phrase_fn = phrase_fn

        self._offending_since: dict[str, float] = {}
        self._last_said: dict[str, float] = {}
        self._times_said: dict[str, int] = {}
        self._last_any = -1e9
        self._last_state: State | None = None
        self.history: list[Cue] = []
        self.current: Cue | None = None

    def reset_pose(self) -> None:
        self._offending_since.clear()

    # ------------------------------------------------------------------ update
    def update(self, t: float, ev: Evaluation, state: State) -> Cue | None:
        cue = self._state_cue(t, state, ev)
        if cue is None and state in (State.SETUP, State.HOLDING) and ev.usable:
            cue = self._correction_cue(t, ev)
        if cue is not None:
            self._last_any = t
            self._last_said[cue.key] = t
            self._times_said[cue.key] = self._times_said.get(cue.key, 0) + 1
            self.history.append(cue)
            self.current = cue
        return cue

    # ---------------------------------------------------------------- internal
    def _state_cue(self, t: float, state: State, ev: Evaluation) -> Cue | None:
        prev, self._last_state = self._last_state, state
        if state == prev:
            return None
        if state is State.HOLDING:
            self.reset_pose()          # a fresh hold starts with a clean slate
        if not self.state_cues:
            return None
        if state is State.HOLDING:
            return Cue("Good. Hold it.", "state.holding", "info", t)
        if state is State.COMPLETE:
            return Cue("Well held. Release the pose slowly.", "state.complete", "praise", t)
        if state is State.WAITING and prev is not None:
            return Cue(ev.reason or "Step into frame", "state.waiting", "info", t)
        return None

    def _correction_cue(self, t: float, ev: Evaluation) -> Cue | None:
        if t - self._last_any < self.global_cooldown_s:
            return None

        live = {r.check.key for r in ev.failures}
        for key in list(self._offending_since):
            if key not in live:
                self._offending_since.pop(key)

        for res in ev.failures:
            key = res.check.key
            since = self._offending_since.setdefault(key, t)
            if t - since < self.persist_s:
                continue
            if t - self._last_said.get(key, -1e9) < self.cue_cooldown_s:
                continue
            return self._build(t, res)
        return None

    def _build(self, t: float, res: CheckResult) -> Cue:
        times = self._times_said.get(res.check.key, 0)
        level = "firm" if times >= self.firm_after else "soft"
        text = res.cue()
        if self.phrase_fn is not None:
            try:
                text = self.phrase_fn(res, level) or text
            except Exception:
                pass                      # never let the LLM break the session
        elif level == "firm":
            text = f"Again - {text[0].lower()}{text[1:]}"
        return Cue(text, res.check.key, level, t)


class Speaker:
    """Text-to-speech on a worker thread.

    pyttsx3's run loop blocks, so it lives in its own thread with the engine
    created inside that thread (a Windows SAPI5 requirement).  If the queue is
    still busy the new cue is dropped rather than stacked - stale coaching is
    worse than none.
    """

    def __init__(self, enabled: bool = True, rate: int = 165, volume: float = 1.0) -> None:
        self.enabled = enabled
        self.available = False
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._run, args=(rate, volume), daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        if not self.enabled or not text:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass

    def close(self) -> None:
        if self._thread is None:
            return
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self, rate: int, volume: float) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            self.available = True
        except Exception as exc:                       # no voice on this machine
            print(f"[voice] text-to-speech unavailable ({exc}); cues will print only")
            self.enabled = False
            return
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
        try:
            engine.stop()
        except Exception:
            pass
