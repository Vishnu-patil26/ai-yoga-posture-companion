"""AI Yoga Companion - the front door.

    profile  ->  choose a routine  ->  your personalised plan  ->  practice

Step 1 collects the health profile, step 2 offers the routines (by lifestyle,
by goal, or by body position), step 3 shows the plan after the profile has
been applied (skipped poses, shortened holds, cautions, diet notes), and step 4
runs it pose by pose: scored poses open the live camera trainer, guided poses
show a spoken walk-in and a timer.

    python launcher.py            (or: python bootstrap.py --ui)
"""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yoga import profile as prof                                  # noqa: E402
from yoga import routines as rt                                   # noqa: E402

GALLERY = os.path.join(HERE, "assets", "gallery")
BG, FG, ACCENT, MUTED = "#f7f5ef", "#23302a", "#2f7d5b", "#6b7a72"
STEPS = ("1  Profile", "2  Routine", "3  Plan", "4  Practice")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Yoga Posture & Wellness Companion")
        self.geometry("980x700")
        self.minsize(900, 640)
        self.configure(bg=BG)
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        st.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        st.configure("Sub.TLabel", foreground=MUTED)
        st.configure("Big.TLabel", font=("Segoe UI", 54, "bold"), foreground=ACCENT)
        st.configure("Accent.TButton", background=ACCENT, foreground="white",
                     font=("Segoe UI", 11, "bold"), padding=8)
        st.map("Accent.TButton", background=[("active", "#256a4c")])
        st.configure("Card.TLabelframe", background=BG)
        st.configure("TNotebook", background=BG)
        st.configure("Treeview", rowheight=26)

        self.profile: prof.Profile | None = None
        self.plan: prof.Plan | None = None
        self.results: list[dict] = []
        self.voice = tk.BooleanVar(value=True)
        self.camera = tk.StringVar(value="0")
        self.speaker = None
        self._after = None
        self._thumbs: dict = {}
        self.picked: list[str] = []          #: gallery selection, in the order ticked
        try:
            with open(os.path.join(GALLERY, "credits.json"), encoding="utf-8") as fh:
                self.credits = json.load(fh)
        except (OSError, ValueError):
            self.credits = {}

        self.header = ttk.Frame(self)
        self.header.pack(fill="x", padx=24, pady=(14, 0))
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True, padx=24, pady=10)
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.show_profile()

    # ------------------------------------------------------------ helpers
    def _screen(self, step: int, title: str, sub: str = "") -> ttk.Frame:
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        for w in (*self.header.winfo_children(), *self.body.winfo_children()):
            w.destroy()
        bar = ttk.Frame(self.header)
        bar.pack(fill="x")
        for i, s in enumerate(STEPS):
            ttk.Label(bar, text=s, foreground=ACCENT if i == step else MUTED,
                      font=("Segoe UI", 10, "bold" if i == step else "normal")
                      ).pack(side="left", padx=(0, 22))
        ttk.Label(self.header, text=title, style="Title.TLabel").pack(anchor="w", pady=(8, 0))
        if sub:
            ttk.Label(self.header, text=sub, style="Sub.TLabel").pack(anchor="w")
        return self.body

    def thumb(self, key: str, size: int):
        """A pose photo as a Tk image, or None when the photo or PIL is missing."""
        if (key, size) not in self._thumbs:
            img = None
            try:
                from PIL import Image, ImageOps, ImageTk
                im = Image.open(os.path.join(GALLERY, key + ".jpg")).convert("RGB")
                im = ImageOps.contain(im, (size, size), Image.LANCZOS)      # never crop a head off
                canvas = Image.new("RGB", (size, size), BG)
                canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
                img = ImageTk.PhotoImage(canvas)
            except Exception:
                pass
            self._thumbs[(key, size)] = img
        return self._thumbs[(key, size)]

    def say(self, text: str) -> None:
        if self.speaker is None and self.voice.get():
            from yoga.feedback import Speaker
            self.speaker = Speaker(enabled=True)
        if self.speaker is not None and self.voice.get():
            self.speaker.say(text)

    def quit_app(self) -> None:
        if self.speaker is not None:
            self.speaker.close()
        self.destroy()

    # ------------------------------------------------------ 1. the profile
    def show_profile(self) -> None:
        b = self._screen(0, "Tell us about you",
                         "Used only on this computer. It decides which poses are "
                         "right for you and how long to hold them.")
        self.v = {k: tk.StringVar() for k in
                  ("name", "age", "weight", "height", "bust", "waist", "hip", "location")}
        self.v["diet"] = tk.StringVar(value=prof.DIETS[0])
        self.v["bp"] = tk.StringVar(value="None")
        self.v["job"] = tk.StringVar(value="sitting")
        self.v["diabetic"] = tk.BooleanVar(value=False)

        saved = prof.saved_names()
        if saved:
            row = ttk.Frame(b)
            row.pack(fill="x", pady=(0, 8))
            ttk.Label(row, text="Returning? Load a saved profile:").pack(side="left")
            cb = ttk.Combobox(row, values=saved, state="readonly", width=22)
            cb.pack(side="left", padx=8)
            cb.bind("<<ComboboxSelected>>", lambda _e: self._load_saved(cb.get()))

        form = ttk.Frame(b)
        form.pack(fill="x")
        fields = [("Name", "name"), ("Age", "age"), ("Weight (kg)", "weight"),
                  ("Height (cm)", "height"), ("Bust / shoulder (cm)", "bust"),
                  ("Waist (cm)", "waist"), ("Hip (cm)", "hip"), ("Location (city)", "location")]
        for i, (label, key) in enumerate(fields):
            r, c = divmod(i, 2)
            ttk.Label(form, text=label).grid(row=r, column=c * 2, sticky="w", pady=5, padx=(0, 10))
            ttk.Entry(form, textvariable=self.v[key], width=26).grid(
                row=r, column=c * 2 + 1, sticky="w", padx=(0, 30))
        r = len(fields) // 2
        ttk.Label(form, text="Diet plan").grid(row=r, column=0, sticky="w", pady=5)
        ttk.Combobox(form, textvariable=self.v["diet"], values=prof.DIETS,
                     state="readonly", width=23).grid(row=r, column=1, sticky="w")
        ttk.Label(form, text="Blood pressure").grid(row=r, column=2, sticky="w", padx=(0, 10))
        ttk.Combobox(form, textvariable=self.v["bp"], values=prof.BP,
                     state="readonly", width=23).grid(row=r, column=3, sticky="w")

        opts = ttk.Frame(b)
        opts.pack(fill="x", pady=14)
        ttk.Checkbutton(opts, text="I am diabetic", variable=self.v["diabetic"]).pack(
            side="left", padx=(0, 30))
        ttk.Label(opts, text="My day job is:").pack(side="left")
        for j in prof.JOBS:
            ttk.Radiobutton(opts, text=j.capitalize(), value=j,
                            variable=self.v["job"]).pack(side="left", padx=8)

        ttk.Label(b, text="Age, weight and height are required. Bust, waist and hip are "
                          "optional - they give your waist-to-hip ratio.",
                  style="Sub.TLabel").pack(anchor="w")
        ttk.Button(b, text="Continue  >", style="Accent.TButton",
                   command=self._submit_profile).pack(anchor="e", pady=16)

    def _load_saved(self, name: str) -> None:
        p = prof.load(name)
        if not p:
            return
        s = lambda x: "" if x in (None, 0, 0.0) else str(x)       # noqa: E731
        for k, val in (("name", p.name), ("age", s(p.age)), ("weight", s(p.weight_kg)),
                       ("height", s(p.height_cm)), ("bust", s(p.bust_cm)),
                       ("waist", s(p.waist_cm)), ("hip", s(p.hip_cm)),
                       ("location", p.location), ("diet", p.diet), ("bp", p.bp),
                       ("job", p.job)):
            self.v[k].set(val)
        self.v["diabetic"].set(p.diabetic)

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
            location=self.v["location"].get().strip())
        errs = prof.validate(p)
        if any(v != v for v in (p.age, p.weight_kg, p.height_cm)):     # NaN = not a number
            errs.append("Age, weight and height must be numbers.")
        if errs:
            messagebox.showerror("Please fix", "\n".join(errs))
            return
        prof.save(p)
        self.profile = p
        self.show_routines()

    # ----------------------------------------------------- 2. the routine
    def show_routines(self) -> None:
        p = self.profile
        rec = rt.recommend(p.job)
        b = self._screen(1, f"Hello {p.name} - what do you need today?",
                         f"BMI {p.bmi:.1f} ({p.bmi_band})"
                         + (f"  |  waist-to-hip {p.whr:.2f}" if p.whr else "")
                         + (f"  |  {p.location}" if p.location else ""))
        self.choice = tk.StringVar(value=rec)
        nb = ttk.Notebook(b)
        nb.pack(fill="both", expand=True)
        tabs = (("By lifestyle", rt.LIFESTYLE), ("By goal", rt.GOALS),
                ("By body position", rt.by_position()))
        for title, routines in tabs:
            f = ttk.Frame(nb, padding=12)
            nb.add(f, text=title)
            for r in routines:
                star = "   * recommended for you" if r.key == rec else ""
                ttk.Radiobutton(f, text=r.title + star, value=r.key,
                                variable=self.choice).pack(anchor="w", pady=(8, 0))
                names = "  >  ".join(rt.POSES[k].name for k in r.poses)
                ttk.Label(f, text=f"{r.target}\n{names}", style="Sub.TLabel",
                          wraplength=820, justify="left").pack(anchor="w", padx=24)
                strip = ttk.Frame(f)
                strip.pack(anchor="w", padx=24, pady=(2, 0))
                for k in r.poses:
                    im = self.thumb(k, 46)
                    if im is not None:
                        tk.Label(strip, image=im, bg=BG, bd=0).pack(side="left", padx=(0, 4))
        lib = ttk.Frame(nb, padding=12)
        nb.add(lib, text="Pick my own poses")
        ttk.Label(lib, text="Not sure which routine? Browse every pose in the project with "
                            "its photo and tick the ones you want.", wraplength=820).pack(anchor="w")
        ttk.Button(lib, text="Open the pose library  >", style="Accent.TButton",
                   command=self.show_gallery).pack(anchor="w", pady=14)
        nb.select(0)
        bar = ttk.Frame(b)
        bar.pack(fill="x", pady=12)
        ttk.Button(bar, text="<  Back", command=self.show_profile).pack(side="left")
        ttk.Button(bar, text="See my plan  >", style="Accent.TButton",
                   command=self._pick_routine).pack(side="right")

    def _pick_routine(self) -> None:
        routine = rt.all_routines()[self.choice.get()]
        self.plan = prof.personalise(self.profile, routine)
        self.show_plan()

    # ------------------------------------------------ 2b. the pose library
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
            cat_of = lambda k: ""                                                   # noqa: E731

        outer = ttk.Frame(b)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self.pick_vars: dict[str, tk.BooleanVar] = {}
        self.count_lbl = ttk.Label(b, text="Nothing selected yet.", style="Sub.TLabel")

        def refresh() -> None:
            self.count_lbl.config(text=(f"{len(self.picked)} selected:  "
                                        + "  >  ".join(rt.POSES[k].name for k in self.picked))
                                  if self.picked else "Nothing selected yet.")

        def toggle(key: str) -> None:
            if self.pick_vars[key].get():
                if key not in self.picked:
                    self.picked.append(key)
            elif key in self.picked:
                self.picked.remove(key)
            refresh()

        for pos, label in rt.POSITIONS.items():
            keys = [k for k, p in rt.POSES.items() if p.position == pos]
            if not keys:
                continue
            ttk.Label(inner, text=label.upper(), foreground=ACCENT,
                      font=("Segoe UI", 11, "bold")).grid(columnspan=3, sticky="w", pady=(14, 4))
            row = ttk.Frame(inner)
            row.grid(sticky="w")
            for i, k in enumerate(keys):
                pose = rt.POSES[k]
                card = ttk.LabelFrame(row, style="Card.TLabelframe", padding=8)
                card.grid(row=i // 3, column=i % 3, padx=6, pady=6, sticky="n")
                im = self.thumb(k, 190)
                if im is not None:
                    tk.Label(card, image=im, bg=BG, bd=0).pack()
                self.pick_vars[k] = tk.BooleanVar(value=k in self.picked)
                ttk.Checkbutton(card, text=pose.name, variable=self.pick_vars[k],
                                command=lambda kk=k: toggle(kk)).pack(anchor="w", pady=(6, 0))
                ttk.Label(card, text=pose.sanskrit, style="Sub.TLabel").pack(anchor="w")
                cat = cat_of(k)
                badge = "camera-scored" if pose.is_scored else "voice-guided"
                ttk.Label(card, text=f"{cat + '  |  ' if cat else ''}{badge}",
                          foreground=ACCENT, font=("Segoe UI", 8, "bold")).pack(anchor="w")
                ttk.Label(card, text=pose.benefit, wraplength=190, justify="left").pack(anchor="w")
                c = self.credits.get(k)
                if c:
                    ttk.Label(card, text=f"Photo: {c['artist'][:28]} - {c['license']}",
                              style="Sub.TLabel", font=("Segoe UI", 7)).pack(anchor="w")
        self.count_lbl.pack(anchor="w", pady=(6, 0))
        bar = ttk.Frame(b)
        bar.pack(fill="x", pady=8)
        ttk.Button(bar, text="<  Back", command=self.show_routines).pack(side="left")
        ttk.Button(bar, text="Clear", command=self._clear_picks).pack(side="left", padx=10)
        ttk.Button(bar, text="Build my routine  >", style="Accent.TButton",
                   command=self._build_custom).pack(side="right")
        refresh()

    def _clear_picks(self) -> None:
        self.picked.clear()
        for v in self.pick_vars.values():
            v.set(False)
        self.count_lbl.config(text="Nothing selected yet.")

    def _build_custom(self) -> None:
        if not self.picked:
            messagebox.showinfo("Pick some poses", "Tick at least one pose first.")
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
        cols = ("#", "Pose", "Position", "How", "Hold")
        tv = ttk.Treeview(b, columns=cols, show="headings", height=len(plan.items))
        for c, w in zip(cols, (40, 260, 120, 300, 80)):
            tv.heading(c, text=c)
            tv.column(c, width=w, anchor="w")
        for i, it in enumerate(plan.items, 1):
            pose = rt.POSES[it.pose_key]
            how = ("camera scores your alignment" if pose.is_scored
                   else "voice-guided timer (no scoring yet)")
            if it.skipped:
                how, hold = it.reason, "-"
            else:
                hold = f"{it.hold_s} s"
            tv.insert("", "end", values=(i, f"{pose.name} ({pose.sanskrit})",
                                         rt.POSITIONS[pose.position], how, hold))
        tv.pack(fill="x")
        ttk.Label(b, text="Poses without a dataset-fitted reference are guided, not "
                          "scored - the app never makes up angles for them.",
                  style="Sub.TLabel").pack(anchor="w", pady=(4, 8))

        info = tk.Text(b, height=11, wrap="word", bg="white", relief="flat",
                       font=("Segoe UI", 10), padx=10, pady=8)
        info.pack(fill="both", expand=True)
        info.insert("end", "FOR YOU\n", "h")
        for n in plan.notes or ["No special precautions from your profile."]:
            info.insert("end", f"  - {n}\n")
        info.insert("end", "\nDIET PLAN\n", "h")
        for d in plan.diet:
            info.insert("end", f"  - {d}\n")
        info.tag_configure("h", font=("Segoe UI", 10, "bold"), foreground=ACCENT)
        info.configure(state="disabled")

        bar = ttk.Frame(b)
        bar.pack(fill="x", pady=10)
        ttk.Button(bar, text="<  Back", command=self.show_routines).pack(side="left")
        ttk.Checkbutton(bar, text="Voice", variable=self.voice).pack(side="left", padx=20)
        ttk.Label(bar, text="Camera").pack(side="left")
        ttk.Spinbox(bar, from_=0, to=5, width=3, textvariable=self.camera).pack(side="left", padx=6)
        ttk.Button(bar, text="Start practice  >", style="Accent.TButton",
                   command=self.start_practice).pack(side="right")

    # ------------------------------------------------------- 4. practice
    def start_practice(self) -> None:
        self.queue = list(self.plan.active)
        self.results = [{"pose": rt.POSES[i.pose_key].name, "result": "skipped for your profile"}
                        for i in self.plan.items if i.skipped]
        self.pos = 0
        self.next_pose()

    def next_pose(self) -> None:
        if self.pos >= len(self.queue):
            return self.show_summary()
        item = self.queue[self.pos]
        pose = rt.POSES[item.pose_key]
        n = len(self.queue)
        b = self._screen(3, f"Pose {self.pos + 1} of {n}: {pose.name}",
                         f"{pose.sanskrit} - {pose.benefit}")
        im = self.thumb(pose.key, 200)
        if im is not None:
            tk.Label(b, image=im, bg=BG, bd=0).pack(anchor="w", pady=(6, 0))
        ttk.Label(b, text="Get ready", font=("Segoe UI", 12, "bold"),
                  foreground=ACCENT).pack(anchor="w", pady=(8, 2))
        for s in pose.steps:
            ttk.Label(b, text="  -  " + s, wraplength=860, justify="left").pack(anchor="w", pady=2)
        mode = ("The camera will score your alignment live and coach you."
                if pose.is_scored else
                "Voice-guided timer. This pose has no fitted reference yet, so it "
                "is timed, not scored.")
        ttk.Label(b, text=mode, style="Sub.TLabel", wraplength=860).pack(anchor="w", pady=14)
        ttk.Label(b, text=f"Hold for {item.hold_s} seconds.").pack(anchor="w")
        bar = ttk.Frame(b)
        bar.pack(fill="x", pady=24)
        ttk.Button(bar, text="Stop routine", command=self.show_summary).pack(side="left")
        ttk.Button(bar, text="Skip this pose",
                   command=lambda: self._done(pose, "skipped")).pack(side="left", padx=10)
        ttk.Button(bar, text="I'm ready - start", style="Accent.TButton",
                   command=lambda: self._begin(pose, item)).pack(side="right")
        self.say(f"Next: {pose.name}. " + " ".join(pose.steps))

    def _begin(self, pose: rt.Pose, item: prof.PlanItem) -> None:
        if pose.is_scored:
            self._run_scored(pose, item)
        else:
            self._run_guided(pose, item)

    def _run_scored(self, pose: rt.Pose, item: prof.PlanItem) -> None:
        import app as trainer
        argv = ["--asana", pose.key, "--hold", str(item.hold_s), "--source", self.camera.get(),
                "--no-intro", "--stop-after-hold", "--user", self.profile.name]
        if pose.key != "vrikshasana":
            argv.append("--no-coach")
        if not self.voice.get():
            argv.append("--no-voice")
        args = trainer.parse_args(argv)
        if self.speaker is not None:                  # free the audio device for the trainer
            self.speaker.close()
            self.speaker = None
        self.withdraw()
        try:
            trainer.run_live(args)
        except (SystemExit, Exception) as exc:        # camera missing etc: don't lose the routine
            self.deiconify()
            messagebox.showerror("Camera problem", str(exc) or "The trainer could not start.")
            return self.next_pose()
        self.deiconify()
        r = getattr(args, "result", None) or {}
        done = r.get("completed", 0) > 0
        self._done(pose, (f"held {r['longest_s']} s, best alignment {r['best']}%"
                          if done else "not completed"), completed=done)

    def _run_guided(self, pose: rt.Pose, item: prof.PlanItem) -> None:
        b = self._screen(3, pose.name, f"{pose.sanskrit} - guided, not scored")
        self.remaining = item.hold_s
        self.big = ttk.Label(b, text="", style="Big.TLabel")
        self.big.pack(pady=30)
        self.msg = ttk.Label(b, text="", font=("Segoe UI", 13))
        self.msg.pack()
        bar = ttk.Frame(b)
        bar.pack(pady=30)
        ttk.Button(bar, text="Finish early",
                   command=lambda: self._done(pose, "finished early", completed=True)).pack(
            side="left", padx=8)
        ttk.Button(bar, text="Skip",
                   command=lambda: self._done(pose, "skipped")).pack(side="left", padx=8)
        self.say("Get into position. Starting in five seconds.")
        self._tick(pose, 5, prep=True)

    def _tick(self, pose: rt.Pose, n: int, prep: bool) -> None:
        if prep:
            self.big.config(text=str(n))
            self.msg.config(text="Get into position")
            if n > 0:
                self._after = self.after(1000, self._tick, pose, n - 1, True)
                return
            self.say("Hold.")
            return self._tick(pose, self.remaining, False)
        self.big.config(text=str(n))
        self.msg.config(text="Hold - breathe slowly")
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
        path = prof.append_history(self.profile, self.results)
        done = sum(1 for r in self.results if r.get("completed"))
        ttk.Label(b, text=f"{done} of {len(self.plan.items)} poses completed.",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=8)
        for r in self.results:
            ttk.Label(b, text=f"  {r['pose']}  -  {r['result']}").pack(anchor="w", pady=2)
        ttk.Label(b, text=f"Saved to {path}", style="Sub.TLabel").pack(anchor="w", pady=14)
        bar = ttk.Frame(b)
        bar.pack(fill="x")
        ttk.Button(bar, text="New routine", style="Accent.TButton",
                   command=self.show_routines).pack(side="left")
        ttk.Button(bar, text="Quit", command=self.quit_app).pack(side="right")


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
