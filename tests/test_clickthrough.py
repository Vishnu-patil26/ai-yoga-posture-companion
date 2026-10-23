"""Click-through test of the launcher: presses the real buttons, tabs, switches and cards.

    python tests/test_clickthrough.py        (no camera needed; opens and closes a window)

Two parts.  The first walks the whole flow the way a person does, pressing the actual
buttons, and checks which screen each press lands on (Home, Back, Skip, Stop...).  The
second goes through every screen and presses every button, switch, tab and drop-down on
it, one at a time from a fresh copy of the screen, and fails if anything raises, even
inside a Tk callback (Tk normally swallows those).  The camera trainer is replaced by a
stand-in and the voice is silent, so nothing real is opened.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["YOGA_LLM"] = "off"

from yoga import profile as prof                                  # noqa: E402
from yoga import routines as rt                                   # noqa: E402
from yoga import ui                                               # noqa: E402

FAILS: list[str] = []
N = 0
ERRORS: list[str] = []
SAID: list[str] = []


def check(cond: bool, label: str) -> None:
    global N
    N += 1
    if not cond:
        FAILS.append(label)
        print(f"  FAIL  {label}".encode("ascii", "replace").decode())


def widgets(root):
    yield root
    for child in root.winfo_children():
        yield from widgets(child)


def norm(text) -> str:
    return " ".join(str(text).split())


def title(win) -> str:
    """The big heading of the screen that is showing."""
    for w in widgets(win.head):
        if isinstance(w, tk.Label):
            return norm(w.cget("text"))
    return ""


def buttons(root) -> list:
    return [w for w in widgets(root) if isinstance(w, ttk.Button)]


def region(win, where: str):
    return {"head": win.head, "foot": win.foot, "page": win.page, "side": win.side,
            "all": win}[where]


def button_texts(win, where: str) -> list[str]:
    return [norm(b.cget("text")) for b in buttons(region(win, where))]


def click(win, text: str, where: str = "foot", nth: int = 0) -> bool:
    """Press the button whose text starts with ``text`` (arrows and spacing ignored)."""
    hits = [b for b in buttons(region(win, where)) if norm(b.cget("text")).startswith(norm(text))]
    if len(hits) <= nth:
        check(False, f"a '{text}' button exists in the {where} (found {len(hits)})")
        return False
    hits[nth].invoke()
    try:
        win.update()
    except tk.TclError:                                   # the press closed the window (Quit)
        pass
    return True


def settle(win) -> None:
    win.update()
    win.update_idletasks()


def no_errors(label: str) -> None:
    check(not ERRORS, f"{label}: no exception ({ERRORS[-1].splitlines()[-1] if ERRORS else ''})")
    ERRORS.clear()


def base(**kw) -> prof.Profile:
    d = dict(name="Asha", age=30, weight_kg=60, height_cm=165, waist_cm=75, hip_cm=95,
             bp="None", job="sitting", diet="Vegetarian", location="Thane")
    d.update(kw)
    return prof.Profile(**d)


# ------------------------------------------------------------ the camera stand-in
def fake_camera(trainer, done: bool = True):
    """Replace the camera window with something that returns at once."""
    calls: list[list[str]] = []

    def run_live(args) -> int:
        calls.append(list(getattr(args, "asana", "").split()))
        args.result = {"completed": 1 if done else 0, "attempted": 1, "longest_s": 20.0, "best": 91.2}
        args.results = [{"key": args.asana, "attempted": 1, "completed": 1 if done else 0,
                         "longest_s": 20.0, "best": 91.2}]
        return 0

    trainer.run_live = run_live
    return calls


# ------------------------------------------------------------ states of the app
def reset(win) -> None:
    win.profile = None
    win.plan = None
    win.guest = False
    win.results = []
    win.queue = []
    win.pos = 0
    win._saved_path = ""
    win._choice_val = ""
    win.picked.clear()
    win._lib_back = win._plan_from = win._practice_back = None
    win.voice.set(False)


def at_profile(win) -> None:
    reset(win)
    win.show_profile()
    settle(win)


def at_routines(win) -> None:
    reset(win)
    win.profile = base()
    prof.save(win.profile)
    win.show_routines()
    settle(win)


def at_library(win) -> None:
    at_routines(win)
    win.open_library()
    settle(win)


def at_plan(win) -> None:
    at_routines(win)
    win._pick_routine()
    settle(win)


def at_pose1(win) -> None:
    at_plan(win)
    win.start_practice()
    settle(win)


def at_pose2(win) -> None:
    at_pose1(win)
    item = win.queue[0]
    win._done(rt.POSES[item.pose_key], "held 20 s, best alignment 90%", completed=True)
    settle(win)


def at_summary(win) -> None:
    at_pose2(win)
    win.show_summary()
    settle(win)


def at_guided(win) -> None:
    """A pose with no fitted reference, which runs as a guided timer."""
    at_plan(win)
    win.start_practice()
    item = win.queue[0]
    from yoga import asanas as _al
    win._saved_fit = _al.LIBRARY.pop(item.pose_key, None)
    win._saved_key = item.pose_key
    win._run_guided(rt.POSES[item.pose_key], item)
    settle(win)


def restore_guided(win) -> None:
    from yoga import asanas as _al
    if getattr(win, "_saved_fit", None) is not None:
        _al.LIBRARY[win._saved_key] = win._saved_fit
        win._saved_fit = None


def wait_for(win, cond, seconds: float = 6.0) -> bool:
    """Pump the Tk event loop until ``cond()`` is true (a background thread is working)."""
    import time
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        win.update()
        if cond():
            return True
        time.sleep(0.03)
    win.update()
    return bool(cond())


def at_settings(win) -> None:
    """The Settings screen with a key already saved (in a temporary .env)."""
    reset(win)
    from yoga import llm
    llm.save_settings(key="gsk_" + "c" * 40, online=True)
    win.show_settings()
    settle(win)


SCREENS = {"settings": at_settings, "profile": at_profile, "routines": at_routines, "library": at_library, "plan": at_plan,
           "pose 1": at_pose1, "pose 2": at_pose2, "summary": at_summary, "guided pose": at_guided}


# ======================================================================= part 1
def part_one(win, trainer, calls) -> None:
    print("  part 1: the whole flow, button by button")
    at_profile(win)
    check(title(win) == "Tell us about you", "the app opens on the profile screen")
    check("Home" not in button_texts(win, "head"), "no Home button on the start screen itself")
    check("Home" in button_texts(win, "side"), "a Home button sits in the sidebar")

    # Home on the start screen keeps what was typed
    win.v["name"].set("Asha")
    click(win, "Home", "side")
    check(win.v["name"].get() == "Asha" and title(win) == "Tell us about you",
          "Home on the start screen does not wipe the half-filled form")
    # the step list: only reachable steps look clickable
    check(str(win.nav_rows[3][0].cget("cursor")) == "arrow", "the Practice step is not a clickable link")
    check(str(win.nav_rows[1][0].cget("cursor")) == "arrow", "Routine is not clickable before a profile exists")

    # fill in and continue
    for k, val in (("age", "30"), ("weight", "60"), ("height", "165")):
        win.v[k].set(val)
    win.v["job"].set("standing")
    click(win, "Continue")
    check(title(win).startswith("Hello Asha"), "Continue -> routines")
    check(str(win.nav_rows[1][0].cget("cursor")) == "hand2", "Routine becomes clickable once there is a profile")
    check("Home" in button_texts(win, "head"), "Home is at the top of the routines screen")
    check("← Profile" in button_texts(win, "foot"), "Back on routines says where it goes: Profile")

    click(win, "Home", "head")
    check(title(win) == "Tell us about you" and win.v["name"].get() == "Asha",
          "Home -> the start, with the saved profile filled in")
    click(win, "Continue")
    click(win, "← Profile")
    check(title(win) == "Tell us about you", "Back on routines -> profile")
    click(win, "Continue")

    # the routine tabs and cards
    for key in ("lifestyle", "goal", "position"):
        win.rt_tabs.select(key)
        win.update()
        shown = [k for k, f in win.rt_lists.items() if f.winfo_manager() == "pack"]
        check(shown == [key], f"routine tab '{key}' shows only its own list")
    win.rt_tabs.select("goal")
    for tg in [w for w in widgets(win.page) if isinstance(w, ui.Toggle) and w.kind == "radio"]:
        if tg.winfo_ismapped():
            tg._click()
            check(win.choice.get() == tg.value, f"routine card '{tg.value}' can be chosen")
    win.ask_var.set("my lower back hurts and I only have 10 minutes")
    click(win, "Ask", "page")
    check(win.choice.get() == "back_pain" and win.ask_out.winfo_children(), "Ask -> the coach picks a routine")
    no_errors("routines screen")

    # library from routines, and back
    click(win, "Pick my own poses", "page")
    check(title(win) == "Pose library", "the library link opens the pose library")
    check("← Routines" in button_texts(win, "foot"), "Back in the library says Routines")
    click(win, "← Routines")
    check(title(win).startswith("Hello Asha"), "Back in the library -> routines")

    # library -> custom routine -> plan -> Back returns to the library, picks kept
    click(win, "Pick my own poses", "page")
    click(win, "Build my routine")
    check(title(win) == "Pose library" and win.flash_box.winfo_children(),
          "Build my routine with nothing ticked explains why and stays")
    ticks = [w for w in widgets(win.page) if isinstance(w, ui.Toggle) and w.kind == "check"]
    ticks[0]._click()
    ticks[1]._click()
    check(len(win.picked) == 2, "two poses ticked")
    click(win, "Clear")
    check(win.picked == [], "Clear empties the selection")
    ticks[0]._click()
    ticks[2]._click()
    click(win, "Build my routine")
    check(title(win) == "My own routine", "Build my routine -> the plan of the ticked poses")
    check("← Pose library" in button_texts(win, "foot"), "Back on that plan says Pose library")
    click(win, "← Pose library")
    check(title(win) == "Pose library" and len(win.picked) == 2, "Back -> the library, with the ticks kept")
    click(win, "Home", "head")
    check(title(win) == "Tell us about you", "Home from the library -> the start")
    click(win, "Continue")

    # plan, its controls, and the first pose
    click(win, "See my plan")
    check(win.plan is not None and title(win) == win.plan.routine.title, "See my plan -> the plan")
    check("← Routines" in button_texts(win, "foot"), "Back on the plan says Routines")
    before = win.voice.get()
    for tg in [w for w in widgets(win.foot) if isinstance(w, ui.Toggle)]:
        tg._click()
    check(win.voice.get() != before, "the Voice switch on the plan flips the voice")
    win.voice.set(False)
    for cb in [w for w in widgets(win.foot) if isinstance(w, ttk.Combobox)]:
        cb.set("1")
        cb.event_generate("<<ComboboxSelected>>")
    check(win.camera.get() == "1", "the camera drop-down on the plan changes the camera")
    win.camera.set("0")
    click(win, "← Routines")
    check(title(win).startswith("Hello Asha"), "Back on the plan -> routines")
    click(win, "See my plan")
    click(win, "Start practice")
    check(title(win).startswith("Pose 1 of"), "Start practice -> pose 1")
    check("← Plan" in button_texts(win, "foot") and "Stop routine" not in button_texts(win, "foot"),
          "before anything is done the pose screen has Back to the plan, not Stop")
    click(win, "← Plan")
    check(win.plan is not None and title(win) == win.plan.routine.title, "Back from pose 1 -> the plan")
    click(win, "Start practice")

    # on the pose screen: audio, demo/photo tabs, switching pose
    click(win, "Play audio", "page")
    click(win, "Stop audio", "page")
    check(bool(SAID), "Play audio speaks the instructions")
    win.voice.set(False)
    for tb in [w for w in widgets(win.page) if isinstance(w, ui.Tabs)]:
        for key in tb.items:
            tb.select(key)
            check(tb.active == key, f"demo tab '{key}' can be selected")
    first = title(win)
    box = [w for w in widgets(win.page) if isinstance(w, ttk.Combobox)][0]
    other = [n for n in box.cget("values") if n not in first][0]
    box.set(other)
    box.event_generate("<<ComboboxSelected>>")
    win.update()
    check(other in title(win) or "skipped for you" in " ".join(
        str(w.cget("text")) for w in widgets(win.flash_box) if isinstance(w, tk.Label)),
        "choosing another pose swaps it (or says why it is skipped)")
    win.update()

    # the camera runs and returns
    click(win, "I'm ready")
    check(win.pos >= 1 and win.results and win.results[-1].get("completed"), "I'm ready -> camera -> result recorded")
    check(any(b.startswith("Stop routine") for b in button_texts(win, "foot")) or title(win).startswith("Session"),
          "after a pose, Stop routine is offered")
    n = win.pos
    click(win, "Skip this pose")
    check(win.pos == n + 1 or title(win) == "Session summary", "Skip this pose -> the next pose")
    no_errors("pose screens")

    # Home half-way through keeps the results
    at_pose2(win)
    click(win, "Home", "head")
    check(title(win) == "Session summary" and win.flash_box.winfo_children(),
          "Home half-way through a routine shows the summary first, with a message")
    check(any("20 s" in str(r.get("result")) for r in win.results), "the finished pose is still in the results")
    click(win, "Home", "head")
    check(title(win) == "Tell us about you", "Home again -> the start")

    # sidebar buttons half-way through do the same
    for label in ("Pose library", "Camera: choose a pose"):
        at_pose2(win)
        click(win, label, "side")
        check(title(win) == "Session summary", f"sidebar '{label}' half-way keeps the session")
    at_pose2(win)
    click(win, "Stop routine")
    check(title(win) == "Session summary", "Stop routine -> summary")
    check(title(win) == "Session summary" and "Home" in button_texts(win, "head"), "the summary has Home")
    click(win, "New routine")
    check(title(win).startswith("Hello Asha"), "New routine -> routines")

    # the first pose without progress: Home goes straight to the start
    at_pose1(win)
    click(win, "Home", "head")
    check(title(win) == "Tell us about you", "Home on pose 1 with nothing done -> the start")

    # guided (timer) pose
    at_guided(win)
    check(title(win) == rt.POSES[win.queue[0].pose_key].name and "Finish early" in button_texts(win, "foot"),
          "a pose without a fit runs as a guided timer with Finish early")
    click(win, "Skip")
    check(win.results and str(win.results[-1]["result"]).startswith("skipped"), "Skip on the timer is recorded")
    restore_guided(win)
    at_guided(win)
    click(win, "Home", "head")
    check(title(win) == "Tell us about you" and win._after is None,
          "Home during the get-ready count stops the timer and goes to the start")
    restore_guided(win)
    no_errors("guided screen")

    # sidebar: voice, camera, library, collect, steps
    at_profile(win)
    t0 = win.voice_btn.cget("text")
    click(win, "Voice", "side")
    check(win.voice_btn.cget("text") != t0, "the sidebar Voice button flips its label")
    win.voice.set(False)
    click(win, "Pose library", "side")
    check(title(win) == "Pose library" and "← Profile" in button_texts(win, "foot"),
          "the sidebar library with no profile: Back says Profile")
    click(win, "← Profile")
    check(title(win) == "Tell us about you", "Back -> the profile screen")
    at_plan(win)
    click(win, "Pose library", "side")
    check("← Plan" in button_texts(win, "foot"), "the sidebar library from the plan: Back says Plan")
    click(win, "← Plan")
    check(win.plan is not None and title(win) == win.plan.routine.title, "Back -> the plan")
    at_profile(win)
    click(win, "Camera: choose a pose", "side")
    check(win.guest and title(win) == "Session summary" and calls, "Camera: choose a pose -> trainer -> summary")
    click(win, "Home", "head")
    check(title(win) == "Tell us about you", "Home from the free-practice summary -> the start")

    started: list = []
    import launcher
    real_popen = launcher.subprocess.Popen
    launcher.subprocess.Popen = lambda *a, **k: started.append(a)
    try:
        click(win, "Contribute samples", "side")
    finally:
        launcher.subprocess.Popen = real_popen
    check(started and "collect.py" in str(started[0]), "Contribute samples starts the collector")

    at_plan(win)
    for i, expect in ((0, "Tell us about you"), (1, "Hello Asha"), (2, win.plan.routine.title)):
        win.nav_rows[i][2].event_generate("<Button-1>")
        win.update()
        check(title(win).startswith(expect), f"clicking sidebar step {i + 1} goes there")
        at_plan(win)
    win.nav_rows[3][2].event_generate("<Button-1>")
    win.update()
    check(title(win) == win.plan.routine.title, "clicking the Practice step does nothing")
    brand = [w for w in widgets(win.side) if isinstance(w, tk.Label) and "AI Yoga" in str(w.cget("text"))][0]
    brand.event_generate("<Button-1>")
    win.update()
    check(title(win) == "Tell us about you", "clicking the app name goes Home")
    no_errors("sidebar")


def part_tuned(win) -> None:
    """Only in the copy that has the 'Tuned poses' switch: every screen with the tuned poses on."""
    if not hasattr(win, "toggle_tuned"):
        return
    print("  part 3: tuned poses switched on")
    at_routines(win)
    if win._tuned_on:
        click(win, "Tuned poses", "side")                 # start from OFF whatever the sweep left
    click(win, "Tuned poses", "side")
    check(win._tuned_on, "the Tuned poses switch turns ON")
    win.plan = prof.personalise(win.profile, rt.Routine("all", "All poses", "custom", "x", tuple(rt.POSES)))
    win._plan_from = None
    win.show_plan()
    settle(win)
    texts = [norm(w.cget("text")) for w in widgets(win.page) if isinstance(w, tk.Label)]
    check(any("on trial" in t for t in texts), "a plan with the tuned poses on says which are on trial")
    no_errors("the plan with tuned poses on")
    click(win, "Start practice")
    check(title(win).startswith("Pose 1 of"), "Start practice works with the tuned poses on")
    for k in list(rt.POSES):                              # each pose's own screen builds
        win.pos = 0
        win.queue = [prof.PlanItem(k, 20)]
        win.next_pose()
        settle(win)
        check(title(win).startswith("Pose 1 of"), f"the pose screen builds for {k} with tuned poses on")
    no_errors("pose screens with tuned poses on")
    click(win, "Tuned poses", "side")
    check(not win._tuned_on, "the Tuned poses switch turns OFF again")


def part_settings(win, opened) -> None:
    """The Settings screen: paste, save, test, switch, remove, and the ways in and out."""
    from yoga import llm
    print("  part 4: the settings screen")

    def sidebar_words() -> str:
        return " ".join(norm(w.cget("text")) for w in widgets(win.status) if isinstance(w, tk.Label))

    def status_words() -> str:
        # the pill prints its word in capitals, so compare in lower case
        return " ".join(norm(w.cget("text")) for w in widgets(win.set_status) if isinstance(w, tk.Label)).lower()

    def flash_words() -> str:
        return " ".join(norm(w.cget("text")) for w in widgets(win.flash_box) if isinstance(w, tk.Label))

    def env_text() -> str:
        try:
            return open(llm.ENV_PATH, encoding="utf-8").read()
        except OSError:
            return ""

    llm.save_settings(key="")
    os.environ.pop("YOGA_LLM", None)
    try:
        os.remove(llm.ENV_PATH)
    except OSError:
        pass
    at_profile(win)
    check("Settings" in button_texts(win, "side"), "a Settings button sits in the sidebar")
    check("online wording: no key" in sidebar_words(), "the sidebar says there is no key yet")
    click(win, "Settings", "side")
    check(title(win) == "Settings", "the sidebar Settings button opens the settings")
    check("\u2190 Profile" in button_texts(win, "foot"), "Back on Settings names where it goes")
    check("no key yet" in status_words(), "the settings say no key yet")
    for want in ("Save and connect", "Test connection", "Get a free key", "Remove key", "Paste"):
        check(want in button_texts(win, "page"), f"settings has a '{want}' button")

    click(win, "Save and connect", "page")
    check("Paste your Groq key" in flash_words() and not env_text(), "saving an empty key explains and writes nothing")
    click(win, "Test connection", "page")
    check("Paste your Groq key" in flash_words(), "testing with no key explains")
    tog = [w for w in widgets(win.page) if isinstance(w, ui.Toggle) and w.lbl.cget("text") == "Use online wording"][0]
    tog._click()
    check(not win.online_var.get() and "Save a Groq key first" in flash_words(), "switching on without a key is refused")
    win.key_var.set("gsk_has a space" + "x" * 20)
    click(win, "Save and connect", "page")
    check("no spaces" in flash_words() and not env_text(), "a key with a space is refused and not written")

    # the show-the-key switch
    shown = [w for w in widgets(win.page) if isinstance(w, ui.Toggle) and w.lbl.cget("text") == "Show the key"][0]
    check(str(win.key_entry.cget("show")) != "", "the key is hidden by default")
    shown._click()
    check(str(win.key_entry.cget("show")) == "", "Show the key reveals it")
    shown._click()
    check(str(win.key_entry.cget("show")) != "", "and hides it again")

    # paste from the clipboard (stand-in, so the real clipboard is left alone)
    good = "gsk_" + "p" * 40
    win.clipboard_get = lambda: "  " + good + "\n"
    click(win, "Paste", "page")
    check(win.key_var.get() == good, "Paste puts the clipboard text in the box, cleaned")
    win.clipboard_get = lambda: (_ for _ in ()).throw(tk.TclError("empty"))
    win.key_var.set("")
    click(win, "Paste", "page")
    check("clipboard is empty" in flash_words(), "Paste with an empty clipboard says so")

    # save and connect
    win.key_var.set(good)
    click(win, "Save and connect", "page")
    check(f"GROQ_API_KEY={good}" in env_text() and "YOGA_LLM=on" in env_text(), "Save and connect writes the key to .env")
    check(llm.status() == "on" and win.key_var.get() == "", "the key is live at once and the box is cleared")
    check("on" in status_words() and "gsk_...pppp" in status_words() and good not in status_words(),
          "the status shows the masked key, never the whole key")
    check("online wording: on" in sidebar_words(), "the sidebar line follows")
    check(wait_for(win, lambda: "Connected" in flash_words()), "the connection test result appears by itself")
    check(win.online_var.get(), "the switch is on")

    # test with a typed key that is wrong
    win.key_var.set("gsk_bad" + "b" * 40)
    click(win, "Test connection", "page")
    check(wait_for(win, lambda: "did not accept" in flash_words()), "a wrong key is reported by Test connection")
    check(f"GROQ_API_KEY={good}" in env_text(), "testing a typed key does not save it")
    win.key_var.set("")

    # switch off and on
    tog._click()
    check(llm.status() == "off" and "YOGA_LLM=off" in env_text(), "switching the wording off is saved")
    check("switched off" in status_words() and "online wording: off" in sidebar_words(), "the screen and sidebar show it")
    tog._click()
    check(llm.status() == "on" and "YOGA_LLM=on" in env_text(), "switching it on again is saved")

    # the key page
    click(win, "Get a free key", "page")
    check(opened and "console.groq.com/keys" in opened[-1], "Get a free key opens Groq's key page")

    # remove
    click(win, "Remove key", "page")
    check(llm.status() == "nokey" and "GROQ_API_KEY" not in env_text(), "Remove key deletes it from the file and from memory")
    check("no key yet" in status_words() and "online wording: no key" in sidebar_words(), "and the screen says so")
    click(win, "Remove key", "page")
    check("No key is saved" in flash_words(), "removing again explains")

    # in and out
    click(win, "\u2190 Profile")
    check(title(win) == "Tell us about you", "Back from Settings -> where it was opened")
    at_routines(win)
    check("Set up online wording (optional) \u2192" in button_texts(win, "page"), "the routine screen offers to set the wording up")
    click(win, "Set up online wording", "page")
    check(title(win) == "Settings" and "\u2190 Routines" in button_texts(win, "foot"), "that button opens Settings, Back names Routines")
    click(win, "\u2190 Routines")
    check(title(win).startswith("Hello Asha"), "Back -> the routine screen")
    llm.save_settings(key="gsk_" + "q" * 40, online=True)
    at_routines(win)
    check("Set up online wording (optional) \u2192" not in button_texts(win, "page"), "with the wording on, that button is gone")
    llm.save_settings(key="")
    at_pose2(win)
    click(win, "Settings", "side")
    check(title(win) == "Session summary", "Settings half-way through a routine keeps the session first")
    click(win, "Home", "head")
    no_errors("the settings screen")


# ======================================================================= part 2
def sweep(win) -> None:
    print("  part 2: every control on every screen, once, from a fresh screen")
    pressed = 0
    for name, setup in SCREENS.items():
        setup(win)
        plan = []                                         # (kind, index) of everything on the screen
        for kind, cls in (("button", ttk.Button), ("toggle", ui.Toggle), ("tabs", ui.Tabs),
                          ("combo", ttk.Combobox)):
            for i, w in enumerate([w for w in widgets(win) if isinstance(w, cls)]):
                if kind == "combo":                       # each value is its own press
                    plan += [(kind, cls, i, j) for j in range(min(3, len(w.cget("values"))))]
                else:
                    plan.append((kind, cls, i, 0))
        for kind, cls, i, j in plan:
            setup(win)
            found = [w for w in widgets(win) if isinstance(w, cls)]
            if i >= len(found):
                continue
            w = found[i]
            label = f"{name}: {kind} #{i}" + (f" '{norm(w.cget('text'))}'" if kind == "button" else "")
            if kind == "button":
                if norm(w.cget("text")) in ("Quit",):
                    continue
                if str(w.cget("state")) == "disabled":
                    continue
                import launcher
                real = launcher.subprocess.Popen
                launcher.subprocess.Popen = lambda *a, **k: None
                try:
                    w.invoke()
                    win.update()
                finally:
                    launcher.subprocess.Popen = real
            elif kind == "toggle":
                before = w._on()
                w._click()
                win.update()
                if w.winfo_exists() and w.kind == "check":
                    check(w._on() != before, f"{label} flips")
            elif kind == "tabs":
                for key in list(w.items):
                    w.select(key)
                    win.update()
                    check(w.active == key, f"{label} selects '{key}'")
            elif kind == "combo":
                w.set(list(w.cget("values"))[j])
                w.event_generate("<<ComboboxSelected>>")
                win.update()
                label += f" value {j}"
            pressed += 1
            no_errors(label)
            check(bool(win.winfo_exists()), f"{label}: the window is still there")
            restore_guided(win)
    # leave the app in a clean state
    print(f"    pressed {pressed} controls")
    check(pressed > 100, f"the sweep reached a meaningful number of controls ({pressed})")


def main() -> int:
    try:
        import launcher
        win = launcher.App()
    except Exception as exc:                                  # no display (CI / Docker)
        print(f"  skip  click-through ({type(exc).__name__}: {exc})")
        return 0
    import app as trainer
    real_run = trainer.run_live
    old_dir = prof.USERS_DIR
    prof.USERS_DIR = tempfile.mkdtemp()
    win.report_callback_exception = lambda et, ev, tb: ERRORS.append(
        "".join(traceback.format_exception(et, ev, tb)))
    win.say = lambda text: SAID.append(text)                  # no real voice in a test
    calls = fake_camera(trainer)
    from yoga import llm
    import webbrowser
    saved_env = {k: os.environ.get(k) for k in ("GROQ_API_KEY", "YOGA_LLM")}
    real_llm = (llm.ENV_PATH, llm.check_connection, llm.complete, webbrowser.open)
    llm.ENV_PATH = os.path.join(tempfile.mkdtemp(), ".env")        # never the developer's real key file
    os.environ.pop("GROQ_API_KEY", None)
    opened: list[str] = []

    def fake_check(key=None):
        if key and "bad" in key:
            return False, "Groq did not accept this key. Check that it was copied completely."
        return True, "Connected. stand-in answered in 0.1 s."

    llm.check_connection = fake_check
    llm.complete = lambda *a, **k: None                              # no network call, ever
    webbrowser.open = lambda url, *a, **k: opened.append(url) or True
    try:
        win.update()
        part_one(win, trainer, calls)
        part_tuned(win)
        part_settings(win, opened)
        sweep(win)
        # Quit last: it destroys the window
        at_profile(win)
        click(win, "Home", "side")
        at_summary(win)
        click(win, "Quit")
        gone = False
        try:
            win.winfo_exists()
        except tk.TclError:
            gone = True
        check(gone, "Quit closes the window")
    finally:
        trainer.run_live = real_run
        prof.USERS_DIR = old_dir
        llm.ENV_PATH, llm.check_connection, llm.complete, webbrowser.open = real_llm
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            win.destroy()
        except tk.TclError:
            pass
    print()
    if FAILS:
        print(f"{len(FAILS)} of {N} click-through checks FAILED")
        return 1
    print(f"{N}/{N} click-through checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
