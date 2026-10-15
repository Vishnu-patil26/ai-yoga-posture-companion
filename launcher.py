"""AI Yoga Companion - the front door.

    profile  ->  choose a routine  ->  your personalised plan  ->  practice

Step 1 collects the health profile, step 2 offers the routines (by lifestyle,
by goal, by body position, or by asking the coach in your own words), step 3
shows the plan after the profile has been applied (skipped poses, shortened
holds, cautions, diet notes), and step 4 runs it pose by pose: scored poses open
the live camera trainer, guided poses show a spoken walk-in and a timer.

    python launcher.py            (or: python bootstrap.py --ui)

The look is the project's black theme (yoga/theme.py, yoga/ui.py): near-black
page, hairline borders, square corners, monospaced capitals, white as the one
accent.  Nothing here opens an operating-system dialog - problems are shown on
the screen they belong to.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yoga import profile as prof                                  # noqa: E402
from yoga import routines as rt                                   # noqa: E402
from yoga import demo as demo_lib                                 # noqa: E402
from yoga import ui                                               # noqa: E402
from yoga.theme import TOKENS as T                                # noqa: E402

GALLERY = os.path.join(HERE, "assets", "gallery")
STEPS = ("Profile", "Routine", "Plan", "Practice")
SUB_WRAP = 760


def pose_mode(pose: rt.Pose) -> tuple[str, str]:
    """The truthful label for how a pose is handled: (text, pill colour).

    Only Tree has the long spoken coach written for it; the other scored poses
    have a short three-step walk-in and scoring; a pose with no fitted reference
    is timed, never scored.
    """
    if pose.key == "vrikshasana":
        return "full coach", "good"
    if pose.is_scored:
        return "camera-scored", "good"
    return "timed only", "warn"


class App(tk.Tk):
    def __init__(self) -> None:
        ui.init_dpi()
        super().__init__()
        ui.setup(self)
        self.title("AI Yoga Posture & Wellness Companion")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(ui.px(1180), sw - 60), min(ui.px(800), sh - 90)
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.minsize(min(ui.px(980), w), min(ui.px(660), h))

        self.profile: prof.Profile | None = None
        self.plan: prof.Plan | None = None
        self.results: list[dict] = []
        self.voice = tk.BooleanVar(value=True)
        self.camera = tk.StringVar(value="0")
        self.speaker = None
        self._after = None
        self._thumbs: dict = {}
        self.picked: list[str] = []          #: gallery selection, in the order ticked
        self._choice_val = ""                #: routine the user last settled on
        self._saved_path = ""                #: history file of the finished session
        self.queue: list = []
        self.pos = 0
        self.guest = False                   #: practising without a saved profile
        self._anim = None                    #: the demo-video animation timer
        try:
            with open(os.path.join(GALLERY, "credits.json"), encoding="utf-8") as fh:
                self.credits = json.load(fh)
        except (OSError, ValueError):
            self.credits = {}

        self._build_shell()
        self.voice.trace_add("write", lambda *_: self._voice_changed())
        self._voice_changed()
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.show_profile()

    # ------------------------------------------------------------- the shell
    def _build_shell(self) -> None:
        self.side = tk.Frame(self, bg=T["card"], width=ui.px(264))
        self.side.pack(side="left", fill="y")
        self.side.pack_propagate(False)
        tk.Frame(self, bg=T["border"], width=1).pack(side="left", fill="y")
        self.main = tk.Frame(self, bg=T["bg"])
        self.main.pack(side="left", fill="both", expand=True)
        self._build_sidebar()

        self.head = tk.Frame(self.main, bg=T["bg"])
        self.head.pack(side="top", fill="x", padx=ui.px(36), pady=(ui.px(30), 0))
        self.flash_box = tk.Frame(self.main, bg=T["bg"])
        self.flash_box.pack(side="top", fill="x", padx=ui.px(36))
        self.foot = tk.Frame(self.main, bg=T["bg"])
        self.foot.pack(side="bottom", fill="x")
        ui.rule(self.main).pack(side="bottom", fill="x")
        self.scroll = ui.ScrollArea(self.main)
        self.scroll.pack(side="top", fill="both", expand=True)
        self.page = tk.Frame(self.scroll.body, bg=T["bg"])
        self.page.pack(fill="both", expand=True, padx=ui.px(36), pady=(ui.px(12), ui.px(24)))
        self.foot_bar = self.foot

    def _build_sidebar(self) -> None:
        s = self.side
        inner = tk.Frame(s, bg=T["card"])
        inner.pack(fill="both", expand=True, padx=ui.px(24), pady=ui.px(30))
        ui.label(inner, "AI Yoga\nPosture Companion", "brand", bg="card").pack(anchor="w")
        ui.Hairline(inner, 40, 2).pack(anchor="w", pady=(ui.px(10), ui.px(8)))
        ui.label(inner, "Pose correction from one webcam. Runs on this computer.", "mono",
                 bg="card", fit=True).pack(fill="x")

        nav = tk.Frame(inner, bg=T["card"])
        nav.pack(fill="x", pady=(ui.px(26), 0))
        self.nav_rows = []
        for i, name in enumerate(STEPS):
            row = tk.Frame(nav, bg=T["card"], cursor="hand2")
            row.pack(fill="x", pady=ui.px(5))
            dot = tk.Canvas(row, width=ui.px(16), height=ui.px(16), bg=T["card"],
                            highlightthickness=0, bd=0)
            dot.pack(side="left", padx=(0, ui.px(10)))
            txt = tk.Label(row, text=name, font=ui.F["body"], fg=T["muted"], bg=T["card"], bd=0)
            txt.pack(side="left")
            for w in (row, dot, txt):
                w.bind("<Button-1>", lambda _e, k=i: self._nav_go(k))
            self.nav_rows.append((row, dot, txt))

        tk.Frame(inner, bg=T["border"], height=1).pack(fill="x", pady=ui.px(22))
        self.status = tk.Frame(inner, bg=T["card"])
        self.status.pack(fill="x")

        tk.Frame(inner, bg=T["card"]).pack(fill="both", expand=True)       # spacer
        self.voice_btn = ui.button(inner, "Voice: ON", self.toggle_voice, "side")
        self.voice_btn.pack(fill="x", pady=(0, ui.px(6)))
        ui.button(inner, "Camera: choose a pose", self.free_practice, "side").pack(fill="x")
        ui.button(inner, "Pose library", self.show_gallery, "side").pack(fill="x", pady=(ui.px(6), 0))
        ui.button(inner, "Contribute samples", self.open_collect, "side").pack(
            fill="x", pady=(ui.px(6), ui.px(14)))
        ui.label(inner, "Your data never leaves this computer.\nGeneral wellness guidance,\nnot medical advice.",
                 "small", bg="card", justify="left").pack(anchor="w")

    def _nav(self, step: int) -> None:
        for i, (_row, dot, txt) in enumerate(self.nav_rows):
            dot.delete("all")
            r = ui.px(7)
            c = ui.px(8)
            if i == step:
                dot.create_oval(c - r, c - r, c + r, c + r, outline=T["accent"], fill=T["accent"])
            elif i < step:
                dot.create_oval(c - r + 2, c - r + 2, c + r - 2, c + r - 2, outline=T["good"],
                                fill=T["good"])
            else:
                dot.create_oval(c - r, c - r, c + r, c + r, outline=T["border2"], fill=T["card"])
            txt.configure(fg=T["text"] if i == step else (T["muted"] if i > step else T["text"]),
                          font=ui.F["bold"] if i == step else ui.F["body"])
        ui.clear(self.status)
        p = self.profile
        if p is None:
            ui.pill(self.status, "no profile yet", "warn", bg="card").pack(anchor="w")
            ui.label(self.status, "Step 1 creates one.", "small", bg="card").pack(anchor="w", pady=(6, 0))
        elif self.guest:
            ui.pill(self.status, "guest session", "warn", bg="card").pack(anchor="w")
            ui.label(self.status, "Nothing is saved.", "small", bg="card").pack(anchor="w", pady=(6, 0))
        else:
            ui.pill(self.status, "profile saved", "good", bg="card").pack(anchor="w")
            ui.label(self.status, p.name, "bold", bg="card").pack(anchor="w", pady=(8, 0))
            bits = [f"ID {p.user_id}" if p.user_id else "", f"BMI {p.bmi:.1f} ({p.bmi_band})"]
            ui.label(self.status, "\n".join(b for b in bits if b), "mono", bg="card",
                     justify="left").pack(anchor="w", pady=(2, 0))

    def _nav_go(self, i: int) -> None:
        if i == 0:
            self.show_profile()
        elif i == 1 and self.profile:
            self.show_routines()
        elif i == 2 and self.plan:
            self.show_plan()

    # ----------------------------------------------------------- screen plumbing
    def _screen(self, step: int, title: str, sub: str = "") -> tk.Frame:
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        if self._anim:
            self.after_cancel(self._anim)
            self._anim = None
        self.unbind("<Return>")
        self.hush()
        for box in (self.head, self.flash_box, self.page, self.foot):
            ui.clear(box)
        self.foot_bar = tk.Frame(self.foot, bg=T["bg"])
        self.foot_bar.pack(fill="x", padx=ui.px(36), pady=ui.px(14))
        ui.label(self.head, title, "h1", fit=True).pack(fill="x")
        ui.Hairline(self.head).pack(anchor="w", pady=(ui.px(10), ui.px(8)))
        if sub:
            ui.label(self.head, sub, "mono", fit=True).pack(fill="x")
        self._nav(step)
        self.scroll.to_top()
        return self.page

    def flash(self, text: str, kind: str = "warn") -> None:
        """Show a message on the current screen (replaces the old pop-up dialogs)."""
        ui.clear(self.flash_box)
        ui.alert(self.flash_box, text, kind).pack(fill="x", pady=(ui.px(12), 0))

    def back_btn(self, command, text: str = "←  Back") -> ui.ttk.Button:
        b = ui.button(self.foot_bar, text, command)
        b.pack(side="left")
        return b

    def next_btn(self, text: str, command) -> ui.ttk.Button:
        b = ui.button(self.foot_bar, text, command, "primary")
        b.pack(side="right")
        return b

    def thumb(self, key: str, size: int, bg: str = "card"):
        """A pose photo as a Tk image, or None when the photo or PIL is missing."""
        n = ui.px(size)
        if (key, n, bg) not in self._thumbs:
            img = None
            try:
                from PIL import Image, ImageOps, ImageTk
                im = Image.open(os.path.join(GALLERY, key + ".jpg")).convert("RGB")
                im = ImageOps.contain(im, (n, n), Image.LANCZOS)      # never crop a head off
                canvas = Image.new("RGB", (n, n), T[bg])
                canvas.paste(im, ((n - im.width) // 2, (n - im.height) // 2))
                img = ImageTk.PhotoImage(canvas)
            except Exception:
                pass
            self._thumbs[(key, n, bg)] = img
        return self._thumbs[(key, n, bg)]

    def say(self, text: str) -> None:
        try:
            if self.speaker is None and self.voice.get():
                from yoga.feedback import Speaker
                self.speaker = Speaker(enabled=True)
            if self.speaker is not None and self.voice.get():
                self.speaker.say(text)
        except Exception:                      # no speech engine: the screen still works
            self.speaker = None

    def hush(self) -> None:
        """Stop speaking right now (leaving a screen, skipping, stopping)."""
        if self.speaker is not None:
            try:
                self.speaker.cancel()
            except Exception:
                pass

    def toggle_voice(self) -> None:
        self.voice.set(not self.voice.get())

    def _voice_changed(self) -> None:
        """Voice on/off is one switch for the whole app: off silences it at once."""
        on = bool(self.voice.get())
        if not on:
            self.hush()
        try:
            self.voice_btn.configure(text="Voice: ON" if on else "Voice: OFF")
        except (AttributeError, tk.TclError):
            pass

    def quit_app(self) -> None:
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        if self._anim:
            self.after_cancel(self._anim)
            self._anim = None
        if self.speaker is not None:
            try:
                self.speaker.close()
            except Exception:
                pass
        self.destroy()

    def open_collect(self) -> None:
        exe = sys.executable
        w = exe.replace("python.exe", "pythonw.exe")
        if os.path.isfile(w):
            exe = w
        try:
            subprocess.Popen([exe, os.path.join(HERE, "collect.py")], cwd=HERE)
        except OSError as exc:
            self.flash(f"Could not open the sample collector: {exc}", "bad")

    # ------------------------------------------------------ 1. the profile
    def show_profile(self) -> None:
        b = self._screen(0, "Tell us about you",
                         "Used only on this computer. It decides which poses are "
                         "right for you and how long to hold them.")
        self.v = {k: tk.StringVar() for k in
                  ("name", "age", "weight", "height", "bust", "waist", "hip", "location",
                   "medication")}
        self.v["diet"] = tk.StringVar(value=prof.DIETS[0])
        self.v["bp"] = tk.StringVar(value="None")
        self.v["job"] = tk.StringVar(value="sitting")
        self.v["diabetic"] = tk.BooleanVar(value=False)
        self.ent: dict[str, tk.Entry] = {}
        if self.profile is not None:                         # coming back to edit
            self._fill(self.profile)

        saved = prof.saved_names()
        if saved:
            row = ui.frame(b)
            row.pack(fill="x", pady=(ui.px(8), ui.px(18)))
            ui.label(row, "returning? load a saved profile", "cap").pack(anchor="w", pady=(0, 5))
            cur = tk.StringVar(value="")
            cb = ui.combo(row, cur, saved, 28)
            cb.pack(anchor="w")
            cb.bind("<<ComboboxSelected>>", lambda _e: self._load_saved(cur.get()), add="+")

        form = ui.frame(b)
        form.pack(fill="x", pady=(ui.px(6), 0))
        form.columnconfigure(0, weight=1, uniform="f")
        form.columnconfigure(1, weight=1, uniform="f")
        fields = [("Name", "name"), ("Age", "age"), ("Weight (kg)", "weight"),
                  ("Height (cm)", "height"), ("Bust / shoulder (cm)", "bust"),
                  ("Waist (cm)", "waist"), ("Hip (cm)", "hip"), ("Location (city)", "location")]
        for i, (text, key) in enumerate(fields):
            r, c = divmod(i, 2)
            cell = ui.frame(form)
            cell.grid(row=r, column=c, sticky="ew", padx=(0, ui.px(24)) if c == 0 else 0,
                      pady=(0, ui.px(14)))
            ui.label(cell, text, "cap").pack(anchor="w", pady=(0, ui.px(5)))
            e = ui.entry(cell, self.v[key], 24)
            e.pack(fill="x", ipady=ui.px(6))
            self.ent[key] = e
        r = len(fields) // 2
        for c, (text, key, vals) in enumerate((("Diet plan", "diet", prof.DIETS),
                                               ("Blood pressure", "bp", prof.BP))):
            cell = ui.frame(form)
            cell.grid(row=r, column=c, sticky="ew", padx=(0, ui.px(24)) if c == 0 else 0,
                      pady=(0, ui.px(14)))
            ui.label(cell, text, "cap").pack(anchor="w", pady=(0, ui.px(5)))
            ui.combo(cell, self.v[key], vals, 24).pack(fill="x")
        cell = ui.frame(form)
        cell.grid(row=r + 1, column=0, columnspan=2, sticky="ew", pady=(0, ui.px(14)))
        ui.label(cell, "medication or conditions (optional)", "cap").pack(anchor="w", pady=(0, ui.px(5)))
        e = ui.entry(cell, self.v["medication"], 60)
        e.pack(fill="x", ipady=ui.px(6))
        self.ent["medication"] = e
        ui.label(cell, "For example: metformin, amlodipine. It only makes your routine gentler - "
                       "never a dose, never advice to start or stop anything.", "small",
                 fit=True).pack(fill="x", pady=(ui.px(4), 0))

        opts = ui.frame(b)
        opts.pack(fill="x", pady=(ui.px(4), ui.px(10)))
        ui.Toggle(opts, "I am diabetic", self.v["diabetic"]).pack(side="left", padx=(0, ui.px(30)))
        ui.label(opts, "my day job is", "cap").pack(side="left", padx=(0, ui.px(10)))
        for j in prof.JOBS:
            ui.Toggle(opts, j.capitalize(), self.v["job"], j, "radio").pack(side="left", padx=(0, ui.px(14)))

        ui.label(b, "Age, weight and height are required. Bust, waist and hip are optional - "
                    "they give your waist-to-hip ratio.", "small", fit=True).pack(fill="x")
        self.next_btn("Continue  →", self._submit_profile)
        ui.button(self.foot_bar, "Skip - practise as a guest", self._make_guest_and_go, "ghost").pack(side="left")
        self.bind("<Return>", lambda _e: self._submit_profile())
        self.ent["name"].focus_set()

    def _make_guest(self) -> None:
        """A throw-away profile so any pose can be tried without filling the form in.
        Nothing about it is saved, and the routine rules treat it as an ordinary adult."""
        self.profile = prof.Profile(name="Guest", age=30, weight_kg=60.0, height_cm=165.0)
        self.guest = True

    def _make_guest_and_go(self) -> None:
        self._make_guest()
        self.show_routines()

    def _fill(self, p: prof.Profile) -> None:
        s = lambda x: "" if x in (None, 0, 0.0) else str(x)       # noqa: E731
        for k, val in (("name", p.name), ("age", s(p.age)), ("weight", s(p.weight_kg)),
                       ("height", s(p.height_cm)), ("bust", s(p.bust_cm)),
                       ("waist", s(p.waist_cm)), ("hip", s(p.hip_cm)),
                       ("location", p.location), ("diet", p.diet), ("bp", p.bp),
                       ("job", p.job), ("medication", p.medication)):
            self.v[k].set(val)
        self.v["diabetic"].set(p.diabetic)

    def _load_saved(self, name: str) -> None:
        p = prof.load(name)
        if not p:
            self.flash(f"Could not read the saved profile '{name}'.", "bad")
            return
        self._fill(p)
        self.flash(f"Loaded {p.name}" + (f"  ({p.user_id})" if p.user_id else "")
                   + ". Check the details and continue.", "good")

    def _num(self, key: str, cast=float):
        t = self.v[key].get().strip()
        if not t:
            return None
        try:
            return cast(t)
        except ValueError:
            return float("nan")

    def _submit_profile(self) -> None:
        p = prof.Profile(
            name=self.v["name"].get().strip(),
            age=self._num("age", int) or 0, weight_kg=self._num("weight") or 0.0,
            height_cm=self._num("height") or 0.0, bust_cm=self._num("bust"),
            waist_cm=self._num("waist"), hip_cm=self._num("hip"),
            diabetic=bool(self.v["diabetic"].get()), bp=self.v["bp"].get(),
            job=self.v["job"].get(), diet=self.v["diet"].get(),
            location=self.v["location"].get().strip(),
            medication=self.v["medication"].get().strip())
        errs = prof.validate(p)
        nums = {"age": p.age, "weight": p.weight_kg, "height": p.height_cm}
        if any(v != v for v in nums.values()):                  # NaN = not a number
            errs = [e for e in errs if not e.startswith(("Age", "Weight", "Height"))]
            errs.append("Age, weight and height must be numbers.")
        for k in ("bust", "waist", "hip"):
            v = getattr(p, {"bust": "bust_cm", "waist": "waist_cm", "hip": "hip_cm"}[k])
            if v is not None and v != v:
                errs.append({"bust": "Bust/shoulder", "waist": "Waist", "hip": "Hip"}[k]
                            + " (cm): must be a number.")
        starts = {"name": "Name", "age": "Age", "weight": "Weight", "height": "Height",
                  "bust": "Bust", "waist": "Waist", "hip": "Hip", "medication": "Medication"}
        for key, ent in self.ent.items():
            lead = starts.get(key)
            ui.mark(ent, bool(lead) and (any(e.startswith(lead) for e in errs)
                                         or (key in nums and nums[key] != nums[key])))
        if errs:
            self.flash("Please fix:\n" + "\n".join("  " + e for e in errs), "bad")
            return
        try:
            prof.save(p)
            self.guest = False
        except OSError as exc:
            self.flash(f"Could not save the profile: {exc}", "bad")
            return
        self.profile = p
        self.show_routines()

    # ----------------------------------------------------- 2. the routine
    def show_routines(self) -> None:
        p = self.profile
        rec = rt.recommend(p.job)
        b = self._screen(1, f"Hello {p.name} — what do you need today?",
                         f"BMI {p.bmi:.1f} ({p.bmi_band})"
                         + (f"  |  waist-to-hip {p.whr:.2f}" if p.whr else "")
                         + (f"  |  {p.location}" if p.location else ""))
        self.choice = tk.StringVar(value=self._choice_val or rec)
        self.choice.trace_add("write", lambda *_: setattr(self, "_choice_val", self.choice.get()))
        routines = rt.all_routines()

        # ---- ask the coach, in your own words
        box = ui.Card(b, "card")
        box.pack(fill="x", pady=(ui.px(8), ui.px(20)))
        ui.label(box.body, "ask the coach", "cap", bg="card").pack(anchor="w")
        row = ui.frame(box.body, "card")
        row.pack(fill="x", pady=(ui.px(8), ui.px(4)))
        self.ask_var = tk.StringVar()
        self.ask_entry = ui.entry(row, self.ask_var, 50)
        self.ask_entry.pack(side="left", fill="x", expand=True, ipady=ui.px(7))
        ui.button(row, "Ask", self._ask, "primary").pack(side="left", padx=(ui.px(10), 0))
        ui.label(box.body, 'Try:  "my lower back hurts and I only have 10 minutes"', "small",
                 bg="card").pack(anchor="w")
        self.ask_out = ui.frame(box.body, "card")
        self.ask_out.pack(fill="x")
        self.ask_entry.bind("<Return>", lambda _e: (self._ask(), "break")[1])

        # ---- the routines, three groups
        groups = {"lifestyle": rt.LIFESTYLE, "goal": rt.GOALS, "position": rt.by_position()}
        self.rt_tabs = ui.Tabs(b, [("lifestyle", "By lifestyle"), ("goal", "By goal"),
                                   ("position", "By body position")], self._rt_tab)
        self.rt_tabs.pack(anchor="w")
        ui.rule(b).pack(fill="x", pady=(0, ui.px(14)))
        self.rt_host = ui.frame(b)
        self.rt_host.pack(fill="x")
        self.rt_lists: dict[str, tk.Frame] = {}
        for key, items in groups.items():
            lst = ui.frame(self.rt_host)
            self.rt_lists[key] = lst
            for r in items:
                self._routine_card(lst, r, rec)
        ui.button(b, "Pick my own poses from the library  →", self.show_gallery, "ghost").pack(
            anchor="w", pady=(ui.px(6), 0))
        start = next((k for k, items in groups.items()
                      if any(r.key == self.choice.get() for r in items)), "lifestyle")
        self.rt_tabs.select(start)

        self.back_btn(self.show_profile)
        self.next_btn("See my plan  →", self._pick_routine)
        self.bind("<Return>", lambda _e: self._pick_routine())

    def _rt_tab(self, key: str) -> None:
        for k, lst in self.rt_lists.items():
            lst.pack_forget()
        self.rt_lists[key].pack(fill="x")

    def _routine_card(self, parent: tk.Misc, r: rt.Routine, rec: str) -> None:
        card = ui.Card(parent, "bg", pad=12)
        card.pack(fill="x", pady=(0, ui.px(10)))
        top = ui.frame(card.body)
        top.pack(fill="x")
        ui.Toggle(top, r.title, self.choice, r.key, "radio", font="h3").pack(side="left")
        if r.key == rec:
            ui.pill(top, "recommended for you", "good").pack(side="right")
        ui.label(card.body, r.target, "muted", fit=True).pack(
            fill="x", padx=(ui.px(25), 0), pady=(ui.px(2), 0))
        ui.label(card.body, "  >  ".join(rt.POSES[k].name for k in r.poses), "mono",
                 fit=True).pack(fill="x", padx=(ui.px(25), 0), pady=(ui.px(4), 0))
        strip = ui.frame(card.body)
        strip.pack(anchor="w", padx=(ui.px(25), 0), pady=(ui.px(8), 0))
        for k in r.poses:
            im = self.thumb(k, 44, "bg")
            if im is not None:
                tk.Label(strip, image=im, bg=T["bg"], bd=0, highlightthickness=1,
                         highlightbackground=T["border"]).pack(side="left", padx=(0, ui.px(6)))

        def refresh(*_a) -> None:
            if card.winfo_exists():
                card.outline("accent" if self.choice.get() == r.key else "border")
        self.choice.trace_add("write", refresh)
        refresh()
        ui.make_clickable(card, lambda: self.choice.set(r.key))

    def _ask(self) -> None:
        text = self.ask_var.get().strip()
        ui.clear(self.ask_out)
        if not text:
            ui.label(self.ask_out, "Type a sentence about how you feel or what you need.", "warn",
                     bg="card").pack(anchor="w", pady=(ui.px(8), 0))
            return
        try:
            from yoga import nlp
            u = nlp.understand(text)
        except Exception as exc:               # never let the language helper break the screen
            ui.label(self.ask_out, f"The coach could not read that ({type(exc).__name__}).", "bad",
                     bg="card").pack(anchor="w", pady=(ui.px(8), 0))
            return
        routines = rt.all_routines()
        if u.best in routines:
            self.choice.set(u.best)
            group = routines[u.best].group
            self.rt_tabs.select(group if group in self.rt_lists else "lifestyle")
        ui.rule(self.ask_out, "border").pack(fill="x", pady=(ui.px(10), ui.px(8)))
        ui.label(self.ask_out, u.explanation, "muted", bg="card", fit=True).pack(fill="x")
        for c in u.cautions:
            ui.label(self.ask_out, c, "warn", bg="card", fit=True).pack(
                fill="x", pady=(ui.px(4), 0))

    def _pick_routine(self) -> None:
        routine = rt.all_routines()[self.choice.get()]
        self.plan = prof.personalise(self.profile, routine)
        self.show_plan()

    # --------------------------------------------------- 2b. the pose library
    def show_gallery(self) -> None:
        b = self._screen(1, "Pose library",
                         "Every pose, with a photo. Tick the poses you want - they run in "
                         "the order you tick them.")
        try:
            from yoga import taxonomy

            def cat_of(k: str) -> str:
                try:
                    return taxonomy.category_of(k).label
                except Exception:                  # a missing label must never break the screen
                    return ""
        except Exception:
            taxonomy = None
            cat_of = lambda k: ""                                                   # noqa: E731

        tabs = ui.Tabs(b, [("gallery", "Poses"), ("tree", "Hierarchy")], self._lib_tab)
        tabs.pack(anchor="w")
        ui.rule(b).pack(fill="x", pady=(0, ui.px(14)))
        self.lib = {"gallery": ui.frame(b), "tree": ui.frame(b)}

        self.pick_vars: dict[str, tk.BooleanVar] = {}
        self.badges: dict[str, tk.Label] = {}
        self.count_lbl = ui.label(self.foot_bar, "", "mono")

        def refresh() -> None:
            n = len(self.picked)
            self.count_lbl.config(text=(f"{n} selected" if n else "nothing selected yet"))
            for k, badge in self.badges.items():
                if k in self.picked:
                    badge.config(text=f"#{self.picked.index(k) + 1}")
                    badge.pack(side="right")
                else:
                    badge.pack_forget()

        def toggle(key: str) -> None:
            if self.pick_vars[key].get():
                if key not in self.picked:
                    self.picked.append(key)
            elif key in self.picked:
                self.picked.remove(key)
            refresh()

        g = self.lib["gallery"]
        for pos, text in rt.POSITIONS.items():
            keys = [k for k, p in rt.POSES.items() if p.position == pos]
            if not keys:
                continue
            ui.label(g, text, "cap_text").pack(anchor="w", pady=(ui.px(6), ui.px(6)))
            ui.rule(g).pack(fill="x", pady=(0, ui.px(10)))
            grid = ui.frame(g)
            grid.pack(fill="x")
            for c in range(3):
                grid.columnconfigure(c, weight=1, uniform="cards")
            for i, k in enumerate(keys):
                pose = rt.POSES[k]
                card = ui.Card(grid, "card", pad=10)
                card.grid(row=i // 3, column=i % 3, padx=ui.px(5), pady=ui.px(5), sticky="nsew")
                body = card.body
                im = self.thumb(k, 176, "card")
                if im is not None:
                    tk.Label(body, image=im, bg=T["card"], bd=0).pack()
                head = ui.frame(body, "card")
                head.pack(fill="x", pady=(ui.px(8), 0))
                self.pick_vars[k] = tk.BooleanVar(value=k in self.picked)
                ui.Toggle(head, pose.name, self.pick_vars[k], command=lambda kk=k: toggle(kk),
                          font="bold").pack(side="left")
                badge = ui.pill(head, "#1", "accent", bg="card")
                self.badges[k] = badge
                ui.label(body, pose.sanskrit, "mono", bg="card").pack(anchor="w")
                tags = ui.frame(body, "card")
                tags.pack(anchor="w", pady=(ui.px(6), ui.px(6)))
                mode, kind = pose_mode(pose)
                ui.pill(tags, mode, kind, bg="card").pack(side="left")
                cat = cat_of(k)
                if cat:
                    ui.label(body, cat, "cap", bg="card", fit=True).pack(fill="x", pady=(0, ui.px(4)))
                ui.label(body, pose.benefit, "muted", bg="card", fit=True).pack(fill="x")
                c = self.credits.get(k)
                if c:
                    ui.label(body, f"Photo: {c['artist'][:30]} - {c['license']}", "small", bg="card",
                             fit=True).pack(fill="x", pady=(ui.px(6), 0))
                ui.make_clickable(card, lambda kk=k: (self.pick_vars[kk].set(not self.pick_vars[kk].get()),
                                                      toggle(kk)))
                ui.button(body, "Practise this pose now  →", lambda kk=k: self.practise_now(kk),
                          "ghost").pack(fill="x", pady=(ui.px(10), 0))

        # ---- the hierarchy: position > category > pose > body points
        t = self.lib["tree"]
        ui.label(t, "Body position > functional category > pose > the body points and joint angles "
                    "the app measures.", "muted", fit=True).pack(fill="x", pady=(0, ui.px(10)))
        holder = ui.Card(t, "bg", pad=2)
        holder.pack(fill="both", expand=True)
        tv = ui.ttk.Treeview(holder.body, show="tree", height=18, selectmode="browse")
        sb = ui.ttk.Scrollbar(holder.body, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        tv.column("#0", width=ui.px(640), stretch=True)

        def add(parent: str, node: dict, depth: int) -> None:
            for text, kids in node.items():
                iid = tv.insert(parent, "end", text="  " + text, open=depth < 2)
                add(iid, kids, depth + 1)
        if taxonomy is not None:
            try:
                add("", taxonomy.tree(details=True), 0)
            except Exception as exc:
                tv.insert("", "end", text=f"  hierarchy unavailable ({type(exc).__name__})")

        self.back_btn(self.show_routines if self.profile else self.show_profile)
        ui.button(self.foot_bar, "Clear", self._clear_picks, "ghost").pack(side="left", padx=ui.px(10))
        self.count_lbl.pack(side="left", padx=ui.px(10))
        self.next_btn("Build my routine  →", self._build_custom)
        tabs.select("gallery")
        refresh()

    def _lib_tab(self, key: str) -> None:
        for k, f in self.lib.items():
            f.pack_forget()
        self.lib[key].pack(fill="both", expand=True)

    def _clear_picks(self) -> None:
        self.picked.clear()
        for v in self.pick_vars.values():
            v.set(False)
        for badge in self.badges.values():
            badge.pack_forget()
        self.count_lbl.config(text="nothing selected yet")

    def _build_custom(self) -> None:
        if not self.picked:
            self.flash("Tick at least one pose first.", "warn")
            return
        if self.profile is None:
            self.show_profile()
            self.flash("Make your profile first - it decides which of those poses are right for you.",
                       "warn")
            return
        routine = rt.Routine("custom", "My own routine", "custom",
                             "The poses you picked, in the order you picked them.",
                             tuple(self.picked))
        self.plan = prof.personalise(self.profile, routine)
        self.show_plan()

    # --------------------------------------------------------- 3. the plan
    def show_plan(self) -> None:
        plan = self.plan
        b = self._screen(2, plan.routine.title, plan.routine.target)

        head = ("#", "pose", "position", "how", "hold")
        table = ui.frame(b)
        table.pack(fill="x", pady=(ui.px(8), 0))
        weights = (0, 4, 2, 5, 1)
        for c, w in enumerate(weights):
            table.columnconfigure(c, weight=w, minsize=ui.px(40) if c == 0 else 0)
        for c, text in enumerate(head):
            ui.label(table, text, "cap").grid(row=0, column=c, sticky="w", padx=(0, ui.px(12)),
                                              pady=(0, ui.px(6)))
        ui.rule(table).grid(row=1, column=0, columnspan=5, sticky="ew")
        for i, it in enumerate(plan.items, 1):
            pose = rt.POSES[it.pose_key]
            r = i * 2 + 0
            mode, kind = pose_mode(pose)
            dim = it.skipped
            fgc = "dim" if dim else "text"
            ui.label(table, f"{i:02d}", "mono", fg=fgc).grid(row=r, column=0, sticky="w",
                                                            pady=ui.px(9), padx=(0, ui.px(12)))
            name = ui.frame(table)
            name.grid(row=r, column=1, sticky="w", padx=(0, ui.px(12)))
            ui.label(name, pose.name, "bold", fg=fgc).pack(anchor="w")
            ui.label(name, pose.sanskrit, "mono", fg="dim" if dim else None).pack(anchor="w")
            ui.label(table, rt.POSITIONS[pose.position], "muted", fg="dim" if dim else None).grid(
                row=r, column=2, sticky="w", padx=(0, ui.px(12)))
            how = ui.frame(table)
            how.grid(row=r, column=3, sticky="w", padx=(0, ui.px(12)))
            if dim:
                ui.pill(how, "skipped", "muted").pack(anchor="w")
                ui.label(how, it.reason, "small", wrap=300).pack(anchor="w", pady=(ui.px(3), 0))
            else:
                ui.pill(how, mode, kind).pack(anchor="w")
                text = {"full coach": "12-stage spoken coach, camera scores your alignment",
                        "camera-scored": "3-step walk-in, camera scores your alignment",
                        "timed only": "voice-guided timer, no scoring"}[mode]
                ui.label(how, text, "small", wrap=300).pack(anchor="w", pady=(ui.px(3), 0))
            ui.label(table, "-" if dim else f"{it.hold_s} s", "mono_text" if not dim else "mono").grid(
                row=r, column=4, sticky="w")
            ui.rule(table).grid(row=r + 1, column=0, columnspan=5, sticky="ew")
        if any(not rt.POSES[i.pose_key].is_scored for i in plan.active):
            ui.label(b, "Poses without a dataset-fitted reference are timed, not scored - the app "
                        "never makes up angles for them.", "small", fit=True).pack(
                fill="x", pady=(ui.px(8), 0))
        tk.Frame(b, bg=T["bg"], height=ui.px(16)).pack()

        cols = ui.frame(b)
        cols.pack(fill="x")
        cols.columnconfigure(0, weight=1, uniform="n")
        cols.columnconfigure(1, weight=1, uniform="n")
        for c, (title, lines) in enumerate((
                ("for you", plan.notes or ["No special precautions from your profile."]),
                ("diet plan", plan.diet))):
            card = ui.Card(cols, "card")
            card.grid(row=0, column=c, sticky="nsew", padx=(0, ui.px(8)) if c == 0 else (ui.px(8), 0))
            ui.label(card.body, title, "cap", bg="card").pack(anchor="w", pady=(0, ui.px(8)))
            for ln in lines:
                ui.label(card.body, "-  " + ln, "muted", bg="card", fit=True).pack(
                    fill="x", pady=(0, ui.px(5)))

        self.back_btn(self.show_routines)
        ui.Toggle(self.foot_bar, "Voice", self.voice).pack(side="left", padx=(ui.px(24), ui.px(18)))
        ui.label(self.foot_bar, "camera", "cap").pack(side="left", padx=(0, ui.px(8)))
        ui.combo(self.foot_bar, self.camera, ("0", "1", "2", "3", "4", "5"), 3).pack(side="left")
        self.next_btn("Start practice  →", self.start_practice)
        self.bind("<Return>", lambda _e: self.start_practice())

    # ------------------------------------------------------- 4. practice
    def start_practice(self) -> None:
        self.queue = list(self.plan.active)
        self.results = [{"pose": rt.POSES[i.pose_key].name, "result": "skipped for your profile"}
                        for i in self.plan.items if i.skipped]
        self._saved_path = ""
        self.pos = 0
        self.next_pose()

    def next_pose(self, error: str = "") -> None:
        if self.pos >= len(self.queue):
            return self.show_summary()
        item = self.queue[self.pos]
        pose = rt.POSES[item.pose_key]
        n = len(self.queue)
        b = self._screen(3, f"Pose {self.pos + 1} of {n}: {pose.name}",
                         f"{pose.sanskrit}  |  {pose.benefit}")
        if error:
            self.flash(error, "bad")
        mode, kind = pose_mode(pose)

        cols = ui.frame(b)
        cols.pack(fill="x", pady=(ui.px(8), 0))
        self._demo_card(cols, pose.key, 300).pack(side="left", anchor="n")
        right = ui.frame(cols)
        right.pack(side="left", fill="both", expand=True, padx=(ui.px(28), 0), anchor="n")

        # choose a different pose without going back to the library
        pick = ui.frame(right)
        pick.pack(fill="x")
        ui.label(pick, "pose", "cap").pack(side="left", padx=(0, ui.px(10)))
        names = {p.name: k for k, p in rt.POSES.items()}
        chosen = tk.StringVar(value=pose.name)
        box = ui.combo(pick, chosen, list(names), 22)
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda _e: self._change_pose(names[chosen.get()], pose.key, chosen),
                 add="+")

        ui.label(right, "get ready", "cap").pack(anchor="w", pady=(ui.px(18), 0))
        for i, st in enumerate(pose.steps, 1):
            row = ui.frame(right)
            row.pack(fill="x", pady=(ui.px(10), 0))
            ui.label(row, f"{i}", "mono_text").pack(side="left", anchor="n", padx=(0, ui.px(12)))
            ui.label(row, st, "body", fit=True).pack(side="left", fill="x", expand=True)
        info = ui.frame(right)
        info.pack(fill="x", pady=(ui.px(18), 0))
        ui.pill(info, mode, kind).pack(side="left")
        ui.label(info, "The camera scores your alignment live and coaches you." if pose.is_scored
                 else "Voice-guided timer. This pose has no fitted reference yet, so it is timed, "
                      "not scored.", "muted", fit=True).pack(side="left", fill="x", expand=True,
                                                             padx=(ui.px(10), 0))
        # audio, on demand: play the instructions, or stop them
        audio = ui.frame(right)
        audio.pack(fill="x", pady=(ui.px(16), 0))
        ui.button(audio, "Play audio", lambda: self._play_audio(pose)).pack(side="left")
        ui.button(audio, "Stop audio", self.hush, "ghost").pack(side="left", padx=(ui.px(8), 0))

        ui.stat_row(b, (("hold", f"{item.hold_s} s", ""), ("pose", f"{self.pos + 1} / {n}", ""),
                        ("camera", self.camera.get() if pose.is_scored else "-", ""))).pack(
            fill="x", pady=(ui.px(22), 0))

        ui.button(self.foot_bar, "Stop routine", self.show_summary).pack(side="left")
        ui.button(self.foot_bar, "Skip this pose", lambda: self._done(pose, "skipped"),
                  "ghost").pack(side="left", padx=ui.px(10))
        self._go = self.next_btn("I'm ready — start  →", lambda: self._begin(pose, item))
        self.bind("<Return>", lambda _e: self._begin(pose, item))
        self.say(f"Next: {pose.name}. " + " ".join(pose.steps))

    # ------------------------------------------------------ the demo video + audio
    def _demo_card(self, parent: tk.Misc, key: str, size: int) -> ui.Card:
        """A card that shows the looping demo of the pose, or its photograph (two tabs)."""
        card = ui.Card(parent, "card", pad=10)
        clip = demo_lib.clip(key) if demo_lib.available() else None
        view = tk.Label(card.body, bg=T["card"], bd=0)
        photo = self.thumb(key, size, "card")
        state = {"mode": "demo" if clip else "photo", "t0": time.perf_counter()}

        def show_mode(m: str) -> None:
            state["mode"], state["t0"] = m, time.perf_counter()
            if m == "photo" and photo is not None:
                view.configure(image=photo)
                view.image = photo

        if clip is not None:
            tabs = ui.Tabs(card.body, [("demo", "Demo video"), ("photo", "Photo")], show_mode)
            tabs.pack(anchor="w", pady=(0, ui.px(6)))
            tabs.select("demo", fire=False)
        view.pack()
        if clip is None:
            show_mode("photo")
        else:
            self._animate(view, clip, ui.px(size), state)
        return card

    def _animate(self, label: tk.Label, clip, size: int, state: dict) -> None:
        """One frame of the demo, then schedule the next (about 15 a second)."""
        if not label.winfo_exists():
            return
        if state["mode"] == "demo":
            try:
                from PIL import Image, ImageTk
                t = time.perf_counter() - state["t0"]
                steps = rt.POSES[clip.key].steps
                frame = clip.render(size, size, t, caption=steps[min(len(steps) - 1, clip.step(t))],
                                    title=rt.POSES[clip.key].sanskrit.upper())
                im = ImageTk.PhotoImage(Image.fromarray(frame[:, :, ::-1]))
                label.configure(image=im)
                label.image = im                     # Tk does not keep its own reference
            except Exception:
                state["mode"] = "photo"
        self._anim = self.after(66, self._animate, label, clip, size, state)

    def _play_audio(self, pose: rt.Pose) -> None:
        if not self.voice.get():
            self.voice.set(True)                     # asking to hear it turns the voice on
        self.hush()
        self.say(f"{pose.name}. " + " ".join(pose.steps))

    def _change_pose(self, new_key: str, old_key: str, var: tk.StringVar) -> None:
        """Swap the pose on the get-ready screen for another one, keeping the profile's rules."""
        if new_key == old_key:
            return
        one = rt.Routine("one", "One pose", "custom", "", (new_key,))
        item = prof.personalise(self.profile, one).items[0]
        if item.skipped:
            self.flash(f"{rt.POSES[new_key].name} is skipped for you: {item.reason}.", "warn")
            var.set(rt.POSES[old_key].name)
            return
        self.queue[self.pos] = item
        self.next_pose()

    def _begin(self, pose: rt.Pose, item: prof.PlanItem) -> None:
        self.unbind("<Return>")
        if pose.is_scored:
            self._run_scored(pose, item)
        else:
            self._run_guided(pose, item)

    def _launch(self, key: str, hold: int, routine: bool):
        """Open the camera trainer and wait for it.  Returns its arguments (with the
        results filled in) or None after showing why it could not start."""
        try:
            import app as trainer
            argv = ["--asana", key, "--hold", str(hold), "--source", self.camera.get(),
                    "--no-intro", "--user", self.profile.name]
            if routine:
                argv.append("--stop-after-hold")
            if not self.voice.get():
                argv.append("--no-voice")
            args = trainer.parse_args(argv)
        except (SystemExit, Exception) as exc:
            self._launch_error = f"The trainer could not be loaded: {exc}"
            return None
        if self.speaker is not None:                  # free the audio device for the trainer
            try:
                self.speaker.close()
            except Exception:
                pass
            self.speaker = None
        self.withdraw()
        try:
            trainer.run_live(args)
        except (SystemExit, Exception) as exc:        # camera missing etc: don't lose the routine
            self.deiconify()
            self.lift()
            self._launch_error = str(exc) or "The trainer could not start."
            return None
        self.deiconify()
        self.lift()
        self.focus_force()
        return args

    def _run_scored(self, pose: rt.Pose, item: prof.PlanItem) -> None:
        try:
            self._go.configure(text="Starting the camera…", state="disabled")
            self.update_idletasks()
        except tk.TclError:
            pass
        args = self._launch(pose.key, item.hold_s, routine=True)
        if args is None:
            return self.next_pose(error=self._launch_error)
        r = getattr(args, "result", None) or {}
        done = r.get("completed", 0) > 0
        # poses the person switched to in the camera window are part of the session too
        for other in getattr(args, "results", []):
            if other["key"] != pose.key and other["attempted"] > 0 and other["key"] in rt.POSES:
                self.results.append({"pose": rt.POSES[other["key"]].name + " (chosen in the camera window)",
                                     "result": (f"held {other['longest_s']} s, best alignment {other['best']}%"
                                                if other["completed"] else "not completed"),
                                     "completed": other["completed"] > 0, "scored": True})
        self._done(pose, (f"held {r['longest_s']} s, best alignment {r['best']}%"
                          if done else "not completed"), completed=done)

    def free_practice(self) -> None:
        """Open the camera straight away; the pose is chosen in the camera window."""
        self.practise_now("vrikshasana", free=True)

    def practise_now(self, key: str, free: bool = False) -> None:
        """One pose, straight from the library (or the sidebar), without a routine."""
        if self.profile is None:
            self._make_guest()
        one = rt.Routine("one", f"Practise {rt.POSES[key].name}", "custom",
                         "One pose, chosen from the library.", (key,))
        plan = prof.personalise(self.profile, one)
        if not plan.active:
            self.flash(f"{rt.POSES[key].name} is skipped for you: {plan.items[0].reason}.", "warn")
            return
        if not free:
            self.plan = plan
            return self.start_practice()
        self._nav(3)
        args = self._launch(key, plan.active[0].hold_s, routine=False)
        if args is None:
            self.flash(self._launch_error, "bad")
            return
        self.plan = prof.Plan(routine=rt.Routine("free", "Free practice", "custom",
                                                 "The poses you chose in the camera window.", ()))
        self.queue, self.pos, self._saved_path = [], 0, ""
        self.results = [{"pose": rt.POSES[r["key"]].name,
                         "result": (f"held {r['longest_s']} s, best alignment {r['best']}%"
                                    if r["completed"] else "not completed"),
                         "completed": r["completed"] > 0, "scored": True}
                        for r in getattr(args, "results", []) if r["attempted"] > 0 and r["key"] in rt.POSES]
        self.show_summary()

    def _run_guided(self, pose: rt.Pose, item: prof.PlanItem) -> None:
        b = self._screen(3, pose.name, f"{pose.sanskrit}  |  guided, not scored")
        self.remaining = item.hold_s
        cols = ui.frame(b)
        cols.pack(fill="x", pady=(ui.px(8), 0))
        left = ui.Card(cols, "card", pad=10)
        left.pack(side="left", anchor="n")
        im = self.thumb(pose.key, 210, "card")
        if im is not None:
            tk.Label(left.body, image=im, bg=T["card"], bd=0).pack()
        right = ui.Card(cols, "bg", pad=22)
        right.pack(side="left", fill="both", expand=True, padx=(ui.px(24), 0))
        self.msg = ui.label(right.body, "", "cap_text")
        self.msg.pack(anchor="w")
        self.big = ui.label(right.body, "", "big")
        self.big.pack(anchor="w", pady=(ui.px(10), ui.px(10)))
        self.bar = ui.ttk.Progressbar(right.body, style="Thin.Horizontal.TProgressbar",
                                      mode="determinate", maximum=max(1, item.hold_s), value=0)
        self.bar.pack(fill="x")
        ui.button(self.foot_bar, "Finish early",
                  lambda: self._done(pose, "finished early", completed=True)).pack(side="left")
        ui.button(self.foot_bar, "Skip", lambda: self._done(pose, "skipped"),
                  "ghost").pack(side="left", padx=ui.px(10))
        self.say("Get into position. Starting in five seconds.")
        self._tick(pose, 5, prep=True)

    def _tick(self, pose: rt.Pose, n: int, prep: bool) -> None:
        if not self.big.winfo_exists():                # the screen was left mid-countdown
            return
        if prep:
            self.big.config(text=str(n))
            self.msg.config(text="GET INTO POSITION")
            self.bar.configure(value=0)
            if n > 0:
                self._after = self.after(1000, self._tick, pose, n - 1, True)
                return
            self.say("Hold.")
            return self._tick(pose, self.remaining, False)
        self.big.config(text=str(n))
        self.msg.config(text="HOLD - BREATHE SLOWLY")
        self.bar.configure(value=self.remaining - n)
        if n == self.remaining // 2 and n > 5:
            self.say("Halfway. Keep breathing.")
        if n <= 0:
            self.say("Well done. Release slowly.")
            return self._done(pose, f"held {self.remaining} s (timed)", completed=True)
        self._after = self.after(1000, self._tick, pose, n - 1, False)

    def _done(self, pose: rt.Pose, result: str, completed: bool = False) -> None:
        self.results.append({"pose": pose.name, "result": result, "completed": completed,
                             "scored": pose.is_scored})
        self.pos += 1
        self.next_pose()

    # --------------------------------------------------------- summary
    def show_summary(self) -> None:
        b = self._screen(3, "Session summary", f"{self.plan.routine.title}")
        if not self._saved_path and not self.guest:    # one history entry per session, however it ended
            try:
                self._saved_path = prof.append_history(self.profile, self.results)
            except OSError as exc:
                self._saved_path = ""
                self.flash(f"Could not save this session: {exc}", "bad")
        done = sum(1 for r in self.results if r.get("completed"))
        skipped = sum(1 for r in self.results if str(r.get("result", "")).startswith("skipped"))
        scored = sum(1 for r in self.results if r.get("completed") and r.get("scored"))
        total = len(self.plan.items) or len(self.results)
        ui.stat_row(b, (("poses completed", f"{done} / {total}", ""),
                        ("camera-scored", str(scored), "completed with the camera"),
                        ("skipped", str(skipped), ""))).pack(fill="x", pady=(ui.px(8), ui.px(20)))

        try:
            from yoga import nlp
            note = nlp.session_summary(self.results, self.profile)
        except Exception:
            note = ""
        if note:
            card = ui.Card(b, "card")
            card.pack(fill="x", pady=(0, ui.px(20)))
            ui.label(card.body, "coach's note", "cap", bg="card").pack(anchor="w", pady=(0, ui.px(6)))
            ui.label(card.body, note, "body", bg="card", fit=True).pack(fill="x")

        table = ui.frame(b)
        table.pack(fill="x")
        table.columnconfigure(0, weight=3)
        table.columnconfigure(1, weight=1)
        table.columnconfigure(2, weight=4)
        for c, text in enumerate(("pose", "status", "result")):
            ui.label(table, text, "cap").grid(row=0, column=c, sticky="w", padx=(0, ui.px(12)),
                                              pady=(0, ui.px(6)))
        ui.rule(table).grid(row=1, column=0, columnspan=3, sticky="ew")
        for i, r in enumerate(self.results):
            rr = 2 + i * 2
            res = str(r.get("result", ""))
            if r.get("completed"):
                status, kind = "done", "good"
            elif res.startswith("skipped"):
                status, kind = "skipped", "muted"
            else:
                status, kind = "not completed", "warn"
            ui.label(table, r["pose"], "bold").grid(row=rr, column=0, sticky="w", pady=ui.px(9),
                                                    padx=(0, ui.px(12)))
            ui.pill(table, status, kind).grid(row=rr, column=1, sticky="w", padx=(0, ui.px(12)))
            ui.label(table, res, "muted", wrap=340).grid(row=rr, column=2, sticky="w")
            ui.rule(table).grid(row=rr + 1, column=0, columnspan=3, sticky="ew")
        if self.guest:
            ui.label(b, "Guest session - nothing was saved.", "small", fit=True).pack(
                fill="x", pady=(ui.px(14), 0))
        elif self._saved_path:
            ui.label(b, f"Saved to {self._saved_path}", "small", fit=True).pack(
                fill="x", pady=(ui.px(14), 0))

        self.next_btn("New routine  →", self.show_routines)
        ui.button(self.foot_bar, "Quit", self.quit_app, "ghost").pack(side="left")


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
