"""Optional language-model wording for the coach.

The app never needs a language model.  Poses are scored from joint angles, the
routine is chosen by ``yoga.nlp`` and every sentence has a template version.
This module only offers a nicer rewording of two things, and only when a Groq
key is available (free tier, model ``openai/gpt-oss-20b``).  The key is read
from ``GROQ_API_KEY``, in the environment or in a git-ignored ``.env`` file in
the project root.  Without a key, or when Groq is slow, rate-limited or
unreachable, the caller keeps the template text it already has.  There is no
local model: a teammate's laptop only needs the key.

What the model may do.  It is handed facts the deterministic code already
decided (the routine that was picked, the cautions, the numbers of a session)
and asked to say them in two or three friendly sentences.  It chooses nothing.
Its reply then goes through ``check_reply``, which rejects anything that names
a pose outside the routine, gives a dose, tells the user to stop a medicine,
promises a cure, or mentions a head-below-heart or belly-down pose when the
user's flags say to avoid it.  A rejected, empty, slow or failed reply returns
``None`` and the caller shows the template text instead.

What leaves the computer.  Short text only: the sentence the user typed to the
coach (e-mail addresses, phone numbers and links removed) or the numbers of a
finished session.  Never video, frames, landmarks, the user's name, the user ID
or the medication field.  ``YOGA_LLM=off`` keeps everything on the machine.

Standard library only (``urllib``), so there is nothing new to install.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"
USER_AGENT = "AI-Yoga-Companion/1.0"

GROQ_TIMEOUT_S = 15.0
MAX_REPLY_CHARS = 700

SYSTEM = (
    "You are the voice of a beginner yoga trainer app. Write calm, plain English in "
    "two or three short sentences, no lists, no markdown, no emojis. Use only the facts "
    "you are given. Never diagnose, never mention doses, never tell anyone to stop or "
    "change a medicine, never promise a cure, and never say a routine treats, eases or "
    "improves a medical condition: say only what the poses do for the body. Do not name "
    "any pose that is not in the facts. Do not repeat the cautions; they are shown "
    "separately.")


# ---------------------------------------------------------------- settings
def _load_env(path: str | None = None) -> None:
    """Read ``KEY=value`` lines from ``.env`` into ``os.environ`` (never overrides)."""
    path = path or os.path.join(ROOT, ".env")
    try:
        with open(path, encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _groq_key() -> str:
    _load_env()
    return os.environ.get("GROQ_API_KEY", "").strip()


def enabled() -> bool:
    """True when a key is present and ``YOGA_LLM`` is not ``off``."""
    _load_env()
    return os.environ.get("YOGA_LLM", "on").strip().lower() != "off" and bool(_groq_key())


def groq_model() -> str:
    return os.environ.get("YOGA_GROQ_MODEL", GROQ_MODEL)


def describe() -> str:
    """One line for the interface: where the wording comes from."""
    if enabled():
        return (f"Wording by an online model ({groq_model()} on Groq). Your sentence is sent "
                "without your name; set YOGA_LLM=off to stay offline.")
    return "Template wording only (no Groq key set, or YOGA_LLM=off)."


# ------------------------------------------------------------- the calls
@dataclass(frozen=True)
class Reply:
    text: str
    provider: str          #: always "groq"
    model: str
    seconds: float


_cooldown: dict[str, float] = {}       # provider -> time.monotonic() before which we skip it
_lock = threading.Lock()


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _hold_off(provider: str, seconds: float) -> None:
    with _lock:
        _cooldown[provider] = time.monotonic() + seconds


def _cooling(provider: str) -> bool:
    with _lock:
        return time.monotonic() < _cooldown.get(provider, 0.0)


def _call_groq(system: str, user: str, max_tokens: int) -> str:
    key = _groq_key()
    if not key:
        raise RuntimeError("no GROQ_API_KEY")
    data = _post_json(GROQ_URL, {
        "model": groq_model(),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.4,
        # gpt-oss spends part of the budget on hidden reasoning; keep it short and leave room
        "reasoning_effort": "low",
        "max_completion_tokens": max_tokens + 400,
    }, {"Authorization": f"Bearer {key}"}, GROQ_TIMEOUT_S)
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("reply cut off")
    return choice["message"].get("content") or ""


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S)
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"')
            .replace("”", '"').replace("‑", "-").replace("–", "-")
            .replace("—", ", ").replace(" ", " "))      # plain ASCII, one spelling per pose
    text = re.sub(r"[*_`#>]+", "", text)                       # markdown the app cannot show
    text = re.sub(r"\s*\n+\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def complete(user: str, *, system: str = SYSTEM, max_tokens: int = 200) -> Reply | None:
    """Ask Groq.  ``None`` on any failure (no key, offline, rate limit); never raises."""
    if not enabled() or _cooling("groq"):
        return None
    t0 = time.monotonic()
    try:
        text = _clean(_call_groq(system, user, max_tokens))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:                                       # rate limited: back off
            try:
                wait = float(exc.headers.get("retry-after", "30"))
            except (TypeError, ValueError):
                wait = 30.0
            _hold_off("groq", min(max(wait, 5.0), 300.0))
        elif exc.code in (401, 403):                              # bad key: do not keep trying
            _hold_off("groq", 3600.0)
        else:
            _hold_off("groq", 20.0)
        return None
    except Exception:                                             # offline, timeout, bad JSON
        _hold_off("groq", 20.0)
        return None
    return Reply(text, "groq", groq_model(), time.monotonic() - t0) if text else None


# ----------------------------------------------------------------- privacy
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_LONG_NUMBER = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){7,}(?!\d)")


def scrub(text: str, names: tuple[str, ...] = ()) -> str:
    """Remove e-mail addresses, phone numbers, links and the user's own name."""
    out = _URL.sub("[link]", _EMAIL.sub("[email]", text or ""))
    out = _LONG_NUMBER.sub("[number]", out)
    for n in names:
        n = (n or "").strip()
        if len(n) >= 2:
            out = re.sub(re.escape(n), "[name]", out, flags=re.I)
    return out.strip()[:400]


# ------------------------------------------------------------ safety check
_BANNED = (
    (r"\bstop (?:taking|using|your)\b[^.]{0,40}\b(?:medic|tablet|pill|insulin|inhaler|drug)", "tells user to stop medicine"),
    (r"\b(?:skip|miss|change|reduce|increase|adjust)\b[^.]{0,30}\b(?:medic|dose|dosage|tablet|insulin)", "changes medicine"),
    (r"\b\d+\s?(?:mg|mcg|ml|units?)\b|\bdos(?:e|es|age)\b", "mentions a dose"),
    (r"\b(?:cure|cures|cured|curing|heal(?:s|ed)? (?:your|the) )", "promises a cure"),
    (r"\bdiagnos", "diagnoses"),
    (r"\bguarantee", "guarantees an outcome"),
    (r"\b(?:treat|relieve|ease|lower|reduce|improve|manage|control|fix|help(?:s|ing)? with)\b"
     r"[^.]{0,40}\b(?:blood pressure|hypertension|diabet|asthma|heart|arthritis|migraine|"
     r"depress|osteoporosis|glaucoma|sugar)", "claims to treat a medical condition"),
    (r"\b(?:comes?|come) with (?:high|low) blood pressure", "links the routine to a condition"),
    (r"\b(?:no need|don'?t need|do not need)\b[^.]{0,25}\bdoctor", "discourages seeing a doctor"),
    (r"\binstead of\b[^.]{0,25}\b(?:doctor|medic|treatment)", "replaces medical care"),
)

#: avoid-flag -> words naming the poses that flag rules out
_HEAD_DOWN = r"downward|adho|headstand|shoulder ?stand|sirsasana|sarvangasana|forward fold|uttanasana|handstand|plough|plow|halasana|legs up the wall"
_BELLY_DOWN = r"cobra|bhujangasana|on your belly|belly[- ]down|prone|locust|bow pose"
_FLAG_BANS = {
    "avoid_inversion": _HEAD_DOWN,
    "high_bp": _HEAD_DOWN, "heart_condition": _HEAD_DOWN, "glaucoma": _HEAD_DOWN,
    "recent_surgery": _HEAD_DOWN, "migraine": _HEAD_DOWN, "vertigo": _HEAD_DOWN,
    "pregnant": _BELLY_DOWN,
}


def check_reply(text: str, *, flags=(), allowed_poses=()) -> tuple[bool, str]:
    """``(ok, reason)``.  ``allowed_poses`` are the pose names the routine contains;
    any other library pose named in the reply is rejected."""
    t = (text or "").strip()
    if not t:
        return False, "empty"
    if len(t) > MAX_REPLY_CHARS:
        return False, "too long"
    low = t.lower()
    for rx, why in _BANNED:
        if re.search(rx, low):
            return False, why
    for flag in flags or ():
        rx = _FLAG_BANS.get(flag)
        if rx and re.search(rx, low):
            return False, f"names a pose ruled out by '{flag}'"
    if allowed_poses:
        from yoga import routines as rt
        given = {a.lower() for a in allowed_poses}
        allowed = set(given)
        for p in rt.POSES.values():                       # a pose is allowed under both its names
            if p.name.lower() in given or p.sanskrit.lower() in given:
                allowed |= {p.name.lower(), p.sanskrit.lower()}
        for p in rt.POSES.values():
            for label in (p.name, p.sanskrit):
                lab = label.lower()
                if lab in low and lab not in allowed:
                    return False, f"names '{label}', which is not in the routine"
    return True, "ok"


_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers_ok(text: str, facts: str, extra: int = 0) -> bool:
    """Every number in ``text`` must appear in ``facts`` (or be a small count up to
    ``extra``), so the model cannot invent a score or a duration."""
    seen = set(_NUM.findall(facts or "")) | {str(i) for i in range(extra + 1)}
    return all(n in seen for n in _NUM.findall(text or ""))


# -------------------------------------------------------- the two features
def explain_recommendation(question: str, routine, minutes=None, cautions=(), flags=(),
                           names: tuple[str, ...] = ()) -> Reply | None:
    """A friendly 2-3 sentence reason why ``routine`` fits what the user typed.

    ``routine`` is a ``yoga.routines.Routine``.  Returns ``None`` when no model is
    available or the reply fails ``check_reply``; keep the template text then.
    """
    from yoga import routines as rt
    pose_names = [rt.POSES[k].name for k in routine.poses if k in rt.POSES]
    facts = {
        "user_said": scrub(question, names),
        "chosen_routine": routine.title,
        "routine_is_for": routine.target,
        "poses_in_order": pose_names,
        "minutes_available": minutes,
    }
    prompt = ("Facts (already decided by the app):\n" + json.dumps(facts, ensure_ascii=False)
              + "\n\nIn two or three sentences, tell the user why this routine suits what they "
              "said. Mention one or two of the poses by name.")
    return _checked(prompt, flags=flags, allowed=pose_names, facts=json.dumps(facts), counts=len(pose_names))


def session_note(results: list[dict], names: tuple[str, ...] = (), flags=()) -> Reply | None:
    """A short coach's note about a finished session, from the launcher's result rows."""
    rows = []
    for r in results or []:
        rows.append({"pose": str(r.get("pose", "")),
                     "completed": bool(r.get("completed")),
                     "result": scrub(str(r.get("result", "")), names)})
    if not rows:
        return None
    prompt = ("Facts about one finished yoga session (numbers come from the camera scoring):\n"
              + json.dumps(rows, ensure_ascii=False)
              + "\n\nWrite two or three encouraging sentences about this session. Quote only "
              "numbers that appear in the facts. If a pose was skipped or not completed, say "
              "that kindly and do not blame the user.")
    return _checked(prompt, flags=flags, allowed=[r["pose"] for r in rows],
                    facts=json.dumps(rows), counts=len(rows))


def _checked(prompt: str, *, flags, allowed, facts: str, counts: int) -> Reply | None:
    rep = complete(prompt)
    if rep is None:
        return None
    ok, _why = check_reply(rep.text, flags=flags, allowed_poses=allowed)
    if ok and not numbers_ok(rep.text, facts, counts):
        ok = False
    return rep if ok else None
