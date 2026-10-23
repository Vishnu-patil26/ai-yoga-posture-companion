"""Offline checks for the optional language-model layer (``yoga/llm.py``).

    python tests/test_llm.py          (no camera, no network, no API key needed)
    python tests/test_llm.py --live   (also makes two real calls; needs GROQ_API_KEY)

The model itself is replaced by a fake, so these checks cover everything around it:
the ``.env`` reader, what is stripped before text leaves the machine, the safety
checker, the back-off after a rate limit, and the guarantee that a failed or
rejected reply returns ``None`` so the caller keeps its template text.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoga import llm                                              # noqa: E402
from yoga import nlp                                              # noqa: E402
from yoga import routines as rt                                   # noqa: E402

FAILS: list[str] = []
N = 0


def check(cond: bool, label: str) -> None:
    global N
    N += 1
    if not cond:
        FAILS.append(label)
        print(f"  FAIL  {label}")


REAL_LOAD_ENV = llm._load_env
llm._load_env = lambda path=None: None            # tests never read the developer's real .env
os.environ.pop("YOGA_LLM", None)
os.environ.pop("GROQ_API_KEY", None)


def with_key(on: bool = True) -> None:
    if on:
        os.environ["GROQ_API_KEY"] = "gsk_" + "x" * 52
    else:
        os.environ.pop("GROQ_API_KEY", None)
    llm._cooldown.clear()


# ----------------------------------------------------------------- .env reader
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, ".env")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# comment\nTEST_LLM_A=one\nTEST_LLM_B = \"two\"\n\nbroken line\nTEST_LLM_C='three'\n")
    os.environ["TEST_LLM_C"] = "already set"
    REAL_LOAD_ENV(p)
    check(os.environ.get("TEST_LLM_A") == "one", "env: plain value read")
    check(os.environ.get("TEST_LLM_B") == "two", "env: quotes and spaces stripped")
    check(os.environ.get("TEST_LLM_C") == "already set", "env: an existing variable is never overridden")
    REAL_LOAD_ENV(os.path.join(d, "missing.env"))
    check(True, "env: a missing file is not an error")
    for k in ("TEST_LLM_A", "TEST_LLM_B", "TEST_LLM_C"):
        os.environ.pop(k, None)

# ------------------------------------------------------------------ switches
with_key(False)
check(not llm.enabled(), "no key -> disabled")
check("Template wording only" in llm.describe(), "no key -> describe says templates")
with_key(True)
check(llm.enabled(), "key -> enabled")
check("Groq" in llm.describe() and "without your name" in llm.describe(), "describe discloses Groq and the name rule")
os.environ["YOGA_LLM"] = "off"
check(not llm.enabled(), "YOGA_LLM=off disables even with a key")
check(llm.complete("hello") is None, "YOGA_LLM=off -> complete() is None without any call")
os.environ.pop("YOGA_LLM")

# ---------------------------------------------------------------- scrubbing
s = llm.scrub("mail me at asha.k@example.com or call +91 98765 43210, see https://x.io/a, I am Asha",
              ("Asha",))
check("@" not in s and "example" not in s, "scrub: e-mail removed")
check("98765" not in s and "43210" not in s, "scrub: phone number removed")
check("x.io" not in s, "scrub: link removed")
check("Asha" not in s and "[name]" in s, "scrub: the user's name removed")
check(llm.scrub("my back hurts, 10 minutes, I am 34") == "my back hurts, 10 minutes, I am 34",
      "scrub: ordinary numbers (minutes, age) are kept")
check(len(llm.scrub("a" * 2000)) <= 400, "scrub: capped at 400 characters")

# ------------------------------------------------------------ safety checker
ROUTINE = ("Mountain Pose", "Tree Pose")


def ok(text, flags=(), poses=ROUTINE):
    return llm.check_reply(text, flags=flags, allowed_poses=poses)[0]


check(ok("This routine starts with Mountain Pose and then Tree Pose, which steady your balance."),
      "check: a plain grounded reply passes")
check(ok("Tadasana is a calm start and Tree Pose follows."), "check: Sanskrit name of an allowed pose passes")
check(not ok(""), "check: empty rejected")
check(not ok("x" * 800), "check: too long rejected")
check(not ok("You could stop taking your tablets and just practise."), "check: stop medicine rejected")
check(not ok("Take 500 mg before class."), "check: a dose rejected")
check(not ok("Yoga will cure your back pain."), "check: cure rejected")
check(not ok("I diagnose a tight psoas."), "check: diagnosis rejected")
check(not ok("No need to see a doctor about this."), "check: discouraging a doctor rejected")
check(not ok("This routine helps with high blood pressure and calms you."),
      "check: treating a condition rejected")
check(not ok("It can ease the tension that comes with high blood pressure."),
      "check: linking routine to a condition rejected")
check(not ok("Finish with Warrior II for strength."), "check: pose outside the routine rejected")
check(not ok("Finish with Virabhadrasana II."), "check: Sanskrit pose outside the routine rejected")
check(ok("Mountain Pose then Warrior II.", poses=("Mountain Pose", "Warrior II")),
      "check: a pose that IS in the routine passes")
check(ok("Mountain Pose then Virabhadrasana II.", poses=("Mountain Pose", "Warrior II")),
      "check: a pose passes under its other name")
check(not ok("Try a gentle downward dog.", flags=("avoid_inversion",), poses=()),
      "check: head-below-heart pose rejected under avoid_inversion")
check(not ok("Add a headstand.", flags=("high_bp",), poses=()), "check: headstand rejected under high_bp")
check(not ok("Lie on your belly for cobra.", flags=("pregnant",), poses=()),
      "check: belly-down pose rejected when pregnant")
check(ok("Try a gentle downward dog.", flags=(), poses=()), "check: same text passes with no flags")
check(llm.numbers_ok("held for 20 s with 91 % alignment", '[{"result": "held 20 s, best alignment 91%"}]', 3),
      "numbers: figures that are in the facts pass")
check(not llm.numbers_ok("you held it for 45 s", '[{"result": "held 20 s"}]', 3),
      "numbers: an invented figure is rejected")
check(llm.numbers_ok("you completed 2 of 3 poses", '[{"result": "held 20 s"}]', 3),
      "numbers: a small count up to the row count passes")

# -------------------------------------------- complete(): back-off and failures
calls = {"n": 0}


def fake_ok(system, user, max_tokens):
    calls["n"] += 1
    return "  **Two** calm\n sentences.  "


with_key(True)
llm._call_groq = fake_ok
r = llm.complete("hi")
check(r is not None and r.text == "Two calm sentences.", "complete: markdown and whitespace cleaned")
check(r is not None and r.provider == "groq" and r.model == llm.groq_model(), "complete: provider and model reported")


def http_error(code, headers=None):
    def raiser(system, user, max_tokens):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", code, "x", headers or {}, io.BytesIO(b""))
    return raiser


with_key(True)
calls["n"] = 0
llm._call_groq = http_error(429, {"retry-after": "60"})
check(llm.complete("hi") is None, "complete: HTTP 429 -> None")
check(llm.complete("hi") is None and calls["n"] == 1, "complete: after a 429 the next call is skipped, not retried")
with_key(True)
calls["n"] = 0
llm._call_groq = http_error(401)
check(llm.complete("hi") is None and llm.complete("hi") is None and calls["n"] == 1,
      "complete: a rejected key is not retried")
with_key(True)


def offline(system, user, max_tokens):
    raise OSError("network down")


llm._call_groq = offline
check(llm.complete("hi") is None, "complete: network failure -> None, no exception")
with_key(True)
llm._call_groq = lambda s, u, m: "   "
check(llm.complete("hi") is None, "complete: an empty reply -> None")
with_key(False)
llm._call_groq = fake_ok
calls["n"] = 0
check(llm.complete("hi") is None and calls["n"] == 0, "complete: no key -> None without any call")

# ------------------------------------------------- the two features, fake model
with_key(True)
u = nlp.understand("my lower back hurts after sitting all day and I only have 10 minutes")
routine = rt.all_routines()[u.best]
poses = [rt.POSES[k].name for k in routine.poses]
good = f"This routine fits your back. It moves through {poses[0]} and {poses[1]}, and 10 minutes is enough."


def say(text):
    llm._call_groq = lambda s, u_, m: text
    llm._cooldown.clear()


say(good)
rep = llm.explain_recommendation("my lower back hurts", routine, u.minutes, u.cautions, u.flags, ("Asha",))
check(rep is not None and rep.text == good, "explain: a good reply is returned")
say("Skip your tablets and do " + poses[0] + ".")
check(llm.explain_recommendation("x", routine, 10, [], set()) is None, "explain: an unsafe reply -> None (template stays)")
say("This also includes Warrior II and a headstand." if "Warrior II" not in poses else "A headstand helps.")
check(llm.explain_recommendation("x", routine, 10, [], {"avoid_inversion"}) is None,
      "explain: a ruled-out pose -> None")
say("You will feel better in 45 minutes.")
check(llm.explain_recommendation("x", routine, 10, [], set()) is None, "explain: an invented number -> None")

rows = [{"pose": "Mountain Pose", "result": "held 20 s, best alignment 91%", "completed": True, "scored": True},
        {"pose": "Tree Pose", "result": "skipped", "completed": False, "scored": True}]
say("Mountain Pose went well: 20 s at 91 % alignment. Tree Pose was skipped, which is fine.")
rep = llm.session_note(rows, ("Asha",), ())
check(rep is not None and "91" in rep.text, "session: a grounded note is returned")
say("Mountain Pose held 30 s, great.")
check(llm.session_note(rows, ("Asha",), ()) is None, "session: an invented number -> None")
say("Mountain Pose was great.")
check(llm.session_note([], (), ()) is None, "session: no rows -> None, no call")

seen = {}


def spy(system, user, max_tokens):
    seen["user"] = user
    return good


llm._call_groq = spy
llm._cooldown.clear()
llm.session_note([{"pose": "Mountain Pose", "result": "held 20 s for Asha, asha@x.com", "completed": True,
                   "scored": True}], ("Asha",), ())
check("Asha" not in seen["user"].replace("[name]", "") and "@" not in seen["user"],
      "session: the name and e-mail never reach the provider")

# ------------------------------------------------- settings written from the app
saved_env = {k: os.environ.get(k) for k in ("GROQ_API_KEY", "YOGA_LLM")}
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("YOGA_LLM", None)
GOOD = "gsk_" + "k" * 40
check(llm.key_problem("") != "" and llm.key_problem("   ") != "", "settings: an empty key is refused")
check(llm.key_problem("gsk_abc def" + "x" * 20) != "", "settings: a key with a space is refused")
check(llm.key_problem("gsk_short") != "", "settings: a too short key is refused")
check(llm.key_problem(GOOD) == "", "settings: a normal key is accepted")
check(llm.clean_key('  "' + GOOD + "'  ") == GOOD, "settings: quotes and spaces around a pasted key are removed")
check(llm.mask(GOOD) == "gsk_...kkkk" and GOOD not in llm.mask(GOOD), "settings: the key is masked to its ends")
check(llm.mask("abc") == "***", "settings: a very short key is hidden completely")
check(llm.status() == "nokey" and not llm.has_key(), "settings: no key -> status nokey")

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, ".env")
    llm.save_settings(key=GOOD, online=True, path=p)
    text = open(p, encoding="utf-8").read()
    check(text.splitlines()[0].startswith("#") and f"GROQ_API_KEY={GOOD}" in text and "YOGA_LLM=on" in text,
          "settings: a new .env gets a header, the key and the switch")
    check(os.environ.get("GROQ_API_KEY") == GOOD and llm.status() == "on" and llm.enabled(),
          "settings: saving takes effect in this process at once")
    check(llm.key_source(p) == "file", "settings: key_source says the file when the file holds the key")
    os.environ["GROQ_API_KEY"] = "gsk_" + "z" * 40
    check(llm.key_source(p) == "environment", "settings: key_source says environment when it differs from the file")
    os.environ["GROQ_API_KEY"] = GOOD

    llm.save_settings(online=False, path=p)
    check("YOGA_LLM=off" in open(p, encoding="utf-8").read() and llm.status() == "off" and not llm.enabled(),
          "settings: switching off keeps the key but disables the wording")
    check(llm.complete("hello") is None, "settings: switched off -> complete() makes no call")

    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# mine\nOTHER=1\nGROQ_API_KEY=old\nYOGA_LLM=off\nGROQ_API_KEY=dup\n")
    llm.save_settings(key=GOOD, online=True, path=p)
    lines = open(p, encoding="utf-8").read().splitlines()
    check(lines.count("# mine") == 1 and "OTHER=1" in lines, "settings: comments and other settings are kept")
    check(sum(1 for x in lines if x.startswith("GROQ_API_KEY=")) == 1 and f"GROQ_API_KEY={GOOD}" in lines,
          "settings: duplicate key lines collapse to one with the new value")
    check(sum(1 for x in lines if x.startswith("YOGA_LLM=")) == 1 and "YOGA_LLM=on" in lines,
          "settings: the switch line is replaced, not repeated")

    llm._hold_off("groq", 3600.0)
    llm.save_settings(key=GOOD, path=p)
    check(not llm._cooling("groq"), "settings: saving a key ends the one-hour wait after a refused key")

    llm.save_settings(key="", path=p)
    after = open(p, encoding="utf-8").read()
    check("GROQ_API_KEY" not in after and "OTHER=1" in after, "settings: removing the key deletes only its line")
    check("GROQ_API_KEY" not in os.environ and llm.status() == "nokey", "settings: removing the key clears it here too")

    bad = os.path.join(d, "missing_folder", ".env")
    os.environ["GROQ_API_KEY"] = GOOD
    try:
        llm.save_settings(key="gsk_" + "n" * 40, path=bad)
        raised = False
    except OSError:
        raised = True
    check(raised and os.environ["GROQ_API_KEY"] == GOOD, "settings: an unwritable file raises and changes nothing")
    os.environ.pop("GROQ_API_KEY", None)

# the connection test, with the network call replaced
real_post = llm._post_json
sent = {}


def fake_ok(url, payload, headers, timeout):
    sent.update(url=url, headers=headers, payload=payload)
    return {"choices": [{"message": {"content": "ok"}}]}


def raiser(exc):
    def f(url, payload, headers, timeout):
        raise exc
    return f


def http(code):
    return urllib.error.HTTPError("https://x", code, "x", {}, io.BytesIO(b""))


try:
    llm._post_json = fake_ok
    ok, msg = llm.check_connection(GOOD)
    check(ok and msg.startswith("Connected") and sent["headers"].get("Authorization") == f"Bearer {GOOD}",
          "connection test: a good answer says connected and sends the typed key")
    check(sent["payload"].get("max_completion_tokens", 0) <= 400 and "api.groq.com" in sent["url"],
          "connection test: one small call to Groq")
    check(GOOD not in msg, "connection test: the message never repeats the key")
    ok, msg = llm.check_connection(None)
    check(not ok and "no key" in msg.lower(), "connection test: nothing to test without a key")
    os.environ["GROQ_API_KEY"] = GOOD
    check(llm.check_connection()[0], "connection test: with no argument it tests the saved key")
    for exc, want_ok, word in ((http(401), False, "did not accept"), (http(403), False, "did not accept"),
                               (http(429), True, "valid"), (http(404), False, "no model"),
                               (http(500), False, "error"), (urllib.error.URLError("down"), False, "reach"),
                               (TimeoutError("slow"), False, "failed")):
        llm._post_json = raiser(exc)
        ok, msg = llm.check_connection(GOOD)
        check(ok is want_ok and word in msg.lower(), f"connection test: {type(exc).__name__} {getattr(exc, 'code', '')} -> {word}")
    llm._cooldown.clear()
    check(not llm._cooling("groq"), "connection test: a failed test does not put the wording on hold")
finally:
    llm._post_json = real_post
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

# ------------------------------------------------------------------- live (opt in)
if "--live" in sys.argv:
    os.environ.pop("GROQ_API_KEY", None)
    llm._load_env = REAL_LOAD_ENV
    import importlib
    importlib.reload(llm)
    llm._cooldown.clear()
    if not llm.enabled():
        print("  --live: no GROQ_API_KEY found; skipped")
    else:
        rep = llm.explain_recommendation("my lower back hurts after sitting all day", routine, 10, [], set())
        print("  live explain :", None if rep is None else f"[{rep.seconds:.1f}s] {rep.text}")
        rep2 = llm.session_note(rows, (), ())
        print("  live session :", None if rep2 is None else f"[{rep2.seconds:.1f}s] {rep2.text}")
        check(rep is not None, "live: explain returned a checked reply")
        check(rep2 is not None, "live: session note returned a checked reply")

print(f"\n{N - len(FAILS)}/{N} checks passed" + ("" if not FAILS else "  FAILED: " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
