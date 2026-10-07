"""Corrective feedback: what to say, when to say it, and saying it aloud.

Methodology Stage 3 'Correction Logic' + Stage 4 'Intelligent Feedback'.

Two rules keep the coaching usable rather than nagging:
  * a deviation must *persist* before it is spoken, so a wobble is not flagged;
  * each cue has its own cooldown, and repeating the same fault escalates the
    wording from a soft reminder to a firm one.

`phrase_fn` is the hook where the correction is turned into a sentence.  Give
it a callable `(CheckResult, level, features) -> str` and the engine speaks
that instead of the canned string - `yoga/phrasing.py` fills it by default,
converting each deviation into body-relative words ("about a hand's width")
rather than a number.  The signature also accepts a future LLM voice-guru
without any change here; the base implementation runs fully offline, so the
demo never depends on a network call.
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

        #: Rolling record of (when, how long) for CORRECTIONS spoken, used to
        #: stop them piling up.  Guided steps and the hold countdown are not
        #: counted: they are the essential spine of the session and are already
        #: rate-limited by their own logic, so charging them here just starved
        #: the corrections entirely - measured, zero corrections in a 20s hold
        #: that had a real fault the whole way through.
        self._spoken: list[tuple[float, float]] = []
        self.talk_budget = 0.35
        self.budget_window_s = 25.0

    def note_spoken(self, t: float, seconds: float) -> None:
        """Record that something was actually voiced, for the talk budget."""
        self._spoken.append((t, seconds))

    def talking_ratio(self, t: float) -> float:
        """Share of the recent window spent speaking, 0..1."""
        window_start = t - self.budget_window_s
        self._spoken = [(s, d) for s, d in self._spoken if s >= window_start]
        if t <= 0:
            return 0.0
        spent = sum(d for _, d in self._spoken)
        return spent / min(self.budget_window_s, max(1e-6, t))

    def reset_pose(self) -> None:
        self._offending_since.clear()

    # ------------------------------------------------------------------ update
    def update(self, t: float, ev: Evaluation, state: State,
               speaker_busy: bool = False) -> Cue | None:
        cue = self._state_cue(t, state, ev)
        # Do not *raise* a correction that cannot be delivered.  Generating one
        # while the voice is mid-sentence used to log it and show it, then
        # throw the audio away - so the transcript claimed a correction the
        # practitioner never heard, and its cooldown started anyway.
        if (cue is None and not speaker_busy
                and state in (State.SETUP, State.HOLDING) and ev.usable):
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
        # Corrections are the optional half of the coaching - the guided steps
        # and the hold timer are not.  When the recent window is already mostly
        # talking, drop the correction rather than adding to the pile.
        if self.talking_ratio(t) >= self.talk_budget:
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
            # Each time the same fault is raised its cooldown doubles.  Saying
            # "chest up" every seven seconds for a whole hold is nagging, not
            # teaching - if it has not landed after two or three tellings, a
            # fourth at the same cadence will not help either.
            said_before = self._times_said.get(key, 0)
            cooldown = self.cue_cooldown_s * (2 ** min(said_before, 3))
            if t - self._last_said.get(key, -1e9) < cooldown:
                continue
            return self._build(t, res, ev.features)
        return None

    def _build(self, t: float, res: CheckResult, features: dict) -> Cue:
        times = self._times_said.get(res.check.key, 0)
        level = "firm" if times >= self.firm_after else "soft"
        # Say it in full once; after that the practitioner knows the fault and
        # only needs reminding, so repeats use the short form.  Repeating a
        # six-second sentence verbatim is what made the coaching feel like
        # spam rather than teaching.
        text = res.cue()
        if self.phrase_fn is not None:
            try:
                text = self.phrase_fn(res, level, features, times >= 1) or text
            except TypeError:             # a phrase_fn that predates `terse`
                try:
                    text = self.phrase_fn(res, level, features) or text
                except Exception:
                    pass
            except Exception:
                pass                      # never let a bad phrase break the session
        elif level == "firm":
            text = f"Again - {text[0].lower()}{text[1:]}"
        return Cue(text, res.check.key, level, t)


#: Fallback words-per-minute if the engine's own rate cannot be read back.
#: pyttsx3's "rate" property is already in words per minute, so this is only
#: a backstop.
DEFAULT_WPM = 165

#: A short breath/punctuation allowance added to every duration estimate -
#: real speech is never quite as fast as a flat word-rate implies.
SPEECH_PAD_S = 0.35


def estimate_speech_seconds(text: str, rate_wpm: int = DEFAULT_WPM) -> float:
    """How long `text` will take to speak, roughly.

    This is what lets the app avoid two failure modes that only show up with
    real audio, never in the offline self-test: starting a new sentence
    before the last one has finished (pyttsx3 then either garbles both or
    silently drops the new one), and a banner that vanishes from the screen
    before the matching voice line is even done.
    """
    words = max(1, len(text.split()))
    return words / (max(1, rate_wpm) / 60.0) + SPEECH_PAD_S


class Speaker:
    """Text-to-speech on a worker thread.

    pyttsx3's run loop blocks, so it lives in its own thread with the engine
    created inside that thread (a Windows SAPI5 requirement).

    `free_at` is a timeline the caller supplies (the same `t` used everywhere
    else in the app) rather than a raw wall-clock read, so pacing decisions
    stay testable and consistent with the rest of the session.  `say(text, t)`
    stamps how long that line is expected to take and `busy(t)` reports
    whether it is still "speaking" as far as that timeline is concerned - the
    caller is expected to check this *before* offering the next line, so a
    new correction is never fired mid-sentence over an old one.  The queue
    itself still drops on overflow as a backstop, not the primary mechanism.
    """

    def __init__(self, enabled: bool = True, rate: int = 165, volume: float = 1.0) -> None:
        self.enabled = enabled
        self.available = False
        self.rate = rate
        self.free_at = 0.0
        self._warned = False
        #: Depth 2: one line playing, one waiting.  Depth 1 meant anything
        #: offered while a line was still being said was thrown away - and the
        #: lines that landed on top of another were exactly the important ones
        #: ("hold it", the first correction), so the practitioner heard the
        #: introduction, then silence through the part that mattered.  Deeper
        #: than 2 would just build a backlog of stale coaching.
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._run, args=(rate, volume), daemon=True)
            self._thread.start()

    def estimate_seconds(self, text: str) -> float:
        return estimate_speech_seconds(text, self.rate)

    def busy(self, t: float) -> bool:
        """True if a line said at or before `t` is still expected to be
        speaking, on the timeline the caller passes to `say()`."""
        return t < self.free_at

    def reset_timeline(self) -> None:
        """Forget when the speaker is next free.

        Call this whenever the caller's clock restarts - otherwise a
        `free_at` stamped on the old timeline reads as "still speaking" against
        the new one, and the first cues of the session are silently swallowed.
        """
        self.free_at = 0.0

    def say(self, text: str, t: float | None = None) -> None:
        """Queue a line to be spoken.

        `free_at` is advanced from whichever is later - now, or the end of
        what is already queued - so a line waiting its turn is accounted for
        rather than being treated as if it starts immediately.
        """
        if not self.enabled or not text:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            return                        # genuinely saturated; do not lie about timing
        if t is not None:
            start = max(t, self.free_at)
            self.free_at = start + self.estimate_seconds(text)

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
        except Exception as exc:                       # no voice on this machine
            print(f"[voice] text-to-speech unavailable ({exc}); cues will print only")
            self.enabled = False
            return
        self.available = True
        while True:
            text = self._q.get()
            if text is None:
                break
            self._speak_once(pyttsx3, text, rate, volume)

    def _speak_once(self, pyttsx3, text: str, rate: int, volume: float) -> None:
        """Speak one line on a throwaway engine.

        A single long-lived pyttsx3 engine only speaks the FIRST line on
        Windows/SAPI5: every later `runAndWait()` returns immediately without
        producing audio, so the app appears to go silent after the greeting.
        Measured here - line 1 took 5.43 s, lines 2 and 3 returned in 0.37 s
        and 0.27 s having said nothing.

        Building a fresh engine per line side-steps it completely and costs
        0.02-0.12 s to construct, which is nothing next to the seconds the
        line itself takes to say - and it happens on this worker thread, off
        the render loop entirely.
        """
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            # Report once rather than silently swallowing every failure - a
            # blanket `except: pass` here is what hid the bug above.
            if not self._warned:
                self._warned = True
                print(f"[voice] could not speak ({type(exc).__name__}: {exc}); "
                      "cues will continue on screen")
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass


def play_alert() -> None:
    """A short two-tone chime for faults the person should hear immediately.

    Runs on its own thread because winsound.Beep blocks for its duration, and
    never raises - a missing sound device must not stop a practice session.
    """
    import threading

    def _beep() -> None:
        try:
            import winsound
            winsound.Beep(880, 110)
            winsound.Beep(660, 140)
        except Exception:
            try:
                print("", end="", flush=True)
            except Exception:
                pass

    threading.Thread(target=_beep, daemon=True).start()
