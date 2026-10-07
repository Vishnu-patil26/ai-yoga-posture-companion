"""AI Yoga Companion - volunteer photo collection.

    consent  ->  details (get an ID)  ->  capture, pose by pose  ->  summary

The team's own sample of diverse, healthy volunteers, so the angle references can
later be refitted on more than the one or two people per pose that the public
datasets show (see docs/policy/COLLECTION_PROTOCOL.md).  Every volunteer first reads and
accepts the consent / copyright text, confirms they are healthy, chooses what
happens to their face (keep, blur, or store points only), and is given a random
ID - no name is stored.  Then, for each pose, a 5-second countdown is followed by
5 photos one second apart, each labelled right or wrong.

    python collect.py

All the rules (what is stored where, blurring, diversity counts, withdrawal) live
in yoga/collection.py and are tested without a camera; this file is the window
around them.  The camera is opened the same way the trainer opens it, so a video
file can be used instead of a webcam: type its path in the "Camera" box.
"""

from __future__ import annotations

import math
import os
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yoga import collection as col                                # noqa: E402
from yoga.landmarks import CONNECTIONS, CORE_LANDMARKS, Pose, PoseTracker  # noqa: E402
from yoga.routines import POSES                                   # noqa: E402

BG, FG, ACCENT, MUTED = "#f7f5ef", "#23302a", "#2f7d5b", "#6b7a72"
BAD = "#b3402f"
STEPS = ("1  Consent", "2  Details", "3  Capture", "4  Summary")

WINDOW = "Collect"
CAPTURE_W, CAPTURE_H = 1280, 720
BURST_N = 5                  #: photos per attempt
COUNTDOWN_S = 5.0            #: time to get into the pose
BURST_INTERVAL_S = 1.0       #: gap between photos
#: The live skeleton is drawn from a detection every this-many preview frames;
#: detecting on every frame would only make the countdown stutter.
PREVIEW_DETECT_EVERY = 3
MAX_READ_FAILURES = 100

FACE_LABELS = {
    "keep": "Keep my face in the photos",
    "blur": "Blur my face automatically before saving (recommended)",
    "landmarks_only": "Store body points only - no photo of me is saved",
}
VIEW_HINT = {
    "standing": "Face the camera, full body in frame (head to feet), about 2 m away, "
                "camera at about 1 m height.",
    "sitting": "Sit facing the camera, full body in frame, about 2 m away, camera at "
               "about 1 m height.",
    "floor": "Place the mat side-on to the camera so the whole body is seen from the "
             "side, about 2 m away, camera at about 1 m height.",
}


# ---------------------------------------------------------------- the capture
@dataclass
class Shot:
    frame: np.ndarray          #: the un-mirrored camera frame, BGR
    pose: Pose | None          #: None when no body was found


FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT_BGR = (91, 125, 47)


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
          colour: tuple[int, int, int], thick: int = 2) -> None:
    """Text with a dark outline so it reads on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thick, cv2.LINE_AA)


def joints_outside(pose: Pose) -> list[int]:
    """Core joints whose position lies outside the picture.

    BlazePose happily extrapolates a leg past the bottom edge, so a "detected"
    body can still be cut off.  Position, not BlazePose's visibility score, is
    the test: visibility is low for perfectly good photos whenever a limb is
    self-occluded (median 0.1-0.3 over the public Dog / Cobra / Tree photos),
    which would make the warning fire on nearly every attempt.
    """
    margin = 0.02 * max(pose.width, pose.height)
    return [i for i in CORE_LANDMARKS
            if not (-margin <= pose.pts[i][0] <= pose.width + margin
                    and -margin <= pose.pts[i][1] <= pose.height + margin)]


def coverage_note(pose: Pose | None) -> tuple[str, bool]:
    """One line for the volunteer: was a body found, and is all of it in the picture?"""
    if pose is None:
        return "No body detected - step into frame", False
    out = joints_outside(pose)
    if not out:
        return "Body detected - whole body in the picture", True
    return f"Body detected - {len(out)} joint(s) outside the picture; step back", False


def _draw_skeleton(img: np.ndarray, pose: Pose, mirror: bool) -> None:
    w = img.shape[1]
    pts = np.asarray(pose.pts, dtype=float).copy()
    if mirror:
        pts[:, 0] = w - 1 - pts[:, 0]
    for a, b in CONNECTIONS:
        if pose.vis[a] < 0.4 or pose.vis[b] < 0.4:
            continue
        if not (np.all(np.isfinite(pts[a])) and np.all(np.isfinite(pts[b]))):
            continue
        cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])),
                 ACCENT_BGR, 3, cv2.LINE_AA)


def render_preview(frame: np.ndarray, *, caption: str, big: str = "", pose: Pose | None = None,
                   mirror: bool = True, flash: bool = False) -> np.ndarray:
    """The picture shown in the 'Collect' window (never what gets saved)."""
    disp = cv2.flip(frame, 1) if mirror else frame.copy()
    h, w = disp.shape[:2]
    if pose is not None:
        _draw_skeleton(disp, pose, mirror)
    cv2.rectangle(disp, (0, 0), (w, 46), (35, 35, 35), -1)
    _text(disp, caption, (14, 32), 0.8, (255, 255, 255))
    if big:                                   # in the corner, so it never hides the body
        (tw, th), _ = cv2.getTextSize(big, FONT, 4.0, 9)
        _text(disp, big, (w - tw - 30, 60 + th), 4.0, (255, 255, 255), 9)
    note, ok = coverage_note(pose)
    _text(disp, note, (14, h - 20), 0.7, (80, 200, 80) if ok else (60, 60, 235))
    _text(disp, "Esc = cancel", (max(14, w - 170), h - 20), 0.6, (230, 230, 230), 1)
    if flash:
        cv2.addWeighted(np.full_like(disp, 255), 0.5, disp, 0.5, 0, disp)
    return disp


def grab_burst(cap, tracker, *, n: int = BURST_N, countdown_s: float = COUNTDOWN_S,
               interval_s: float = BURST_INTERVAL_S, mirror: bool = True, show=None,
               caption: str = "", clock=time.perf_counter) -> list[Shot]:
    """Count down, then take `n` photos `interval_s` apart.

    `cap` needs only ``.read() -> (ok, frame)`` and `tracker` only
    ``.process(rgb) -> Pose | None``, so this runs against fakes in the tests.
    The camera is read continuously from the first moment - the preview and the
    driver's frame buffer both stay fresh - and a photo is simply the newest
    frame once its time is due.  `show(image) -> bool` draws the preview and
    returns False to cancel (Esc / window closed); pass None for no preview.
    The photos are the raw camera frames: only the preview is mirrored.
    Returns the photos taken so far if cancelled.
    """
    shots: list[Shot] = []
    live: Pose | None = None
    flash_until = -1.0
    failures = frame_no = 0
    t0 = clock()
    while len(shots) < n:
        ok, frame = cap.read()
        if not ok or frame is None:
            failures += 1
            if failures > MAX_READ_FAILURES:
                raise RuntimeError("The camera stopped delivering frames.")
            time.sleep(0.02)
            continue
        failures = 0
        t = clock() - t0
        if t >= countdown_s + len(shots) * interval_s:
            live = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            shots.append(Shot(frame.copy(), live))
            flash_until = t + 0.15
        elif show is not None and frame_no % PREVIEW_DETECT_EVERY == 0:
            live = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_no += 1
        if show is not None:
            if t < countdown_s:
                big, info = str(max(1, math.ceil(countdown_s - t))), "Get into position"
            else:
                big, info = "", f"Hold still - photo {min(len(shots) + 1, n)} of {n}"
            disp = render_preview(frame, caption=f"{caption}   {info}".strip(), big=big,
                                  pose=live, mirror=mirror, flash=t < flash_until)
            if not show(disp):
                break
    return shots


# ------------------------------------------------------------------ the window
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Yoga Companion - volunteer photo collection")
        self.geometry("980x720")
        self.minsize(900, 660)
        self.configure(bg=BG)
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        st.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        st.configure("Sub.TLabel", foreground=MUTED)
        st.configure("Big.TLabel", font=("Segoe UI", 54, "bold"), foreground=ACCENT)
        st.configure("Accent.TButton", background=ACCENT, foreground="white",
                     font=("Segoe UI", 11, "bold"), padding=8)
        st.map("Accent.TButton", background=[("active", "#256a4c"), ("disabled", "#b9c4be")])
        st.configure("Card.TLabelframe", background=BG)
        st.configure("Treeview", rowheight=24)

        self.volunteer: col.Volunteer | None = None
        self.screen = ""
        self.consent_time = ""
        self.last_paths: list[str] = []
        self.last_result: dict = {}
        self._status = ("", None)            #: (text, ok) - survives screen rebuilds
        self._tracker: PoseTracker | None = None
        self._fault_entry: ttk.Entry | None = None
        self.burst_n, self.countdown_s, self.interval_s = BURST_N, COUNTDOWN_S, BURST_INTERVAL_S

        self.consent_ok = tk.BooleanVar(value=False)
        self.healthy_ok = tk.BooleanVar(value=False)
        self.face_policy = tk.StringVar(value="blur")
        self.v = {k: tk.StringVar() for k in
                  ("age_group", "sex", "height", "weight", "body_type", "experience")}
        self.camera = tk.StringVar(value="0")
        self.pose_var = tk.StringVar()
        self.quality = tk.StringVar(value="right")
        self.fault = tk.StringVar()
        self.quality.trace_add("write", lambda *_: self._sync_fault())

        self._pose_labels = {k: f"{i}. {p.name} ({p.sanskrit})"
                             for i, (k, p) in enumerate(POSES.items(), 1)}

        self.header = ttk.Frame(self)
        self.header.pack(fill="x", padx=24, pady=(14, 0))
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True, padx=24, pady=10)
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.show_consent()

    # ------------------------------------------------------------ helpers
    def _screen(self, name: str, step: int, title: str, sub: str = "") -> ttk.Frame:
        self.screen = name
        self._fault_entry = None
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
            ttk.Label(self.header, text=sub, style="Sub.TLabel", wraplength=900,
                      justify="left").pack(anchor="w")
        return self.body

    # Dialogs go through these so a test can replace them (a modal box would
    # otherwise block the test forever).
    def _warn(self, title: str, text: str) -> None:
        messagebox.showerror(title, text)

    def _info(self, title: str, text: str) -> None:
        messagebox.showinfo(title, text)

    def _confirm(self, title: str, text: str) -> bool:
        return bool(messagebox.askyesno(title, text, default=messagebox.NO, icon="warning"))

    def _set_status(self, text: str, ok: bool | None = None) -> None:
        self._status = (text, ok)
        lbl = getattr(self, "status_lbl", None)
        try:
            if lbl is not None and lbl.winfo_exists():
                lbl.configure(text=text, foreground=MUTED if ok is None else ACCENT if ok else BAD)
        except tk.TclError:
            pass

    def quit_app(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
        cv2.destroyAllWindows()
        self.destroy()

    def new_volunteer(self) -> None:
        """Back to a blank consent screen - nothing carries over between people."""
        self.volunteer = None
        self.consent_time = ""
        self.last_paths = []
        self.last_result = {}
        self._status = ("", None)
        self.consent_ok.set(False)
        self.healthy_ok.set(False)
        self.face_policy.set("blur")
        for var in self.v.values():
            var.set("")
        self.quality.set("right")
        self.fault.set("")
        self.show_consent()

    # ---------------------------------------------------------- 1. consent
    def show_consent(self) -> None:
        b = self._screen("consent", 0, "Volunteer consent",
                         "Please read this, then tick both boxes. You can stop at any time.")
        # Packed bottom-up first, so a short window shrinks the text box (which
        # scrolls) instead of pushing the tick boxes and buttons out of sight.
        bar = ttk.Frame(b)
        bar.pack(side="bottom", fill="x", pady=(4, 0))
        ttk.Button(bar, text="Data summary", command=self.show_summary).pack(side="left")
        ttk.Button(bar, text="Withdraw a volunteer", command=self.ask_withdraw).pack(
            side="left", padx=8)
        self.consent_btn = ttk.Button(bar, text="I agree - continue  >", style="Accent.TButton",
                                      command=self._accept_consent)
        self.consent_btn.pack(side="right")
        fp = ttk.Labelframe(b, text="My face", style="Card.TLabelframe", padding=(10, 4))
        fp.pack(side="bottom", fill="x", pady=8)
        for key in col.FACE_POLICIES:
            ttk.Radiobutton(fp, text=FACE_LABELS[key], value=key,
                            variable=self.face_policy).pack(anchor="w")
        ttk.Checkbutton(b, variable=self.healthy_ok, command=self._sync_consent,
                        text="I confirm I am healthy: I have no injury or condition that makes "
                             "these yoga poses unsafe for me.").pack(side="bottom", anchor="w", pady=2)
        ttk.Checkbutton(b, variable=self.consent_ok, command=self._sync_consent,
                        text="I have read the text above, I am 18 or older, and I agree to take "
                             "part and to the copyright licence.").pack(side="bottom", anchor="w",
                                                                         pady=(10, 2))
        box = ttk.Frame(b)
        box.pack(fill="both", expand=True)
        txt = tk.Text(box, wrap="word", bg="white", relief="flat", font=("Segoe UI", 10),
                      padx=12, pady=8, height=6)
        sb = ttk.Scrollbar(box, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.tag_configure("h", font=("Segoe UI", 10, "bold"), foreground=ACCENT)
        for line in col.CONSENT_TEXT.splitlines():
            heading = bool(line) and line[0].isalpha() and line.split("(")[0].strip().isupper()
            txt.insert("end", line + "\n", "h" if heading else ())
        txt.configure(state="disabled")                   # read-only
        self.consent_view = txt
        self._sync_consent()

    def _sync_consent(self) -> None:
        both = bool(self.consent_ok.get() and self.healthy_ok.get())
        self.consent_btn.state(["!disabled"] if both else ["disabled"])

    def _accept_consent(self) -> None:
        if not (self.consent_ok.get() and self.healthy_ok.get()):
            self._warn("Both boxes are needed",
                       "Please tick that you agree and that you are healthy to continue.")
            return
        self.consent_time = col.now_iso()
        self.show_details()

    # ----------------------------------------------------------- 2. details
    def show_details(self) -> None:
        b = self._screen("details", 1, "About you",
                         "No name is asked for. Height and weight are optional. "
                         "You will get a random ID.")
        form = ttk.Frame(b)
        form.pack(fill="x", pady=6)
        rows = (("Age group", "age_group", col.AGE_GROUPS), ("Sex", "sex", col.SEXES),
                ("Body type", "body_type", col.BODY_TYPES),
                ("Yoga experience", "experience", col.EXPERIENCE))
        for i, (label, key, choices) in enumerate(rows):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=6, padx=(0, 14))
            ttk.Combobox(form, textvariable=self.v[key], values=choices, state="readonly",
                         width=24).grid(row=i, column=1, sticky="w")
        ttk.Label(form, text="Height (cm, optional)").grid(row=0, column=2, sticky="w", padx=(40, 14))
        ttk.Entry(form, textvariable=self.v["height"], width=12).grid(row=0, column=3, sticky="w")
        ttk.Label(form, text="Weight (kg, optional)").grid(row=1, column=2, sticky="w", padx=(40, 14))
        ttk.Entry(form, textvariable=self.v["weight"], width=12).grid(row=1, column=3, sticky="w")
        ttk.Label(b, text=f"Face: {FACE_LABELS[self.face_policy.get()]}", style="Sub.TLabel"
                  ).pack(anchor="w", pady=(10, 0))
        bar = ttk.Frame(b)
        bar.pack(fill="x", pady=20)
        ttk.Button(bar, text="<  Back", command=self.show_consent).pack(side="left")
        ttk.Button(bar, text="Get my ID  >", style="Accent.TButton",
                   command=self._submit_details).pack(side="right")

    def _num(self, key: str) -> float | None:
        t = self.v[key].get().strip().replace(",", ".")
        if not t:
            return None
        try:
            return float(t)
        except ValueError:
            return float("nan")

    def _submit_details(self) -> None:
        vol = col.Volunteer(
            age_group=self.v["age_group"].get(), sex=self.v["sex"].get(),
            height_cm=self._num("height"), weight_kg=self._num("weight"),
            body_type=self.v["body_type"].get(), experience=self.v["experience"].get(),
            healthy_confirmed=bool(self.healthy_ok.get()), consent=bool(self.consent_ok.get()),
            consent_version=col.CONSENT_VERSION, consent_time=self.consent_time,
            face_policy=self.face_policy.get())
        errs = col.validate_volunteer(vol)
        if errs:
            self._warn("Please fix", "\n".join(errs))
            return
        try:
            col.register(vol)
        except (col.CollectionError, OSError) as exc:
            self._warn("Could not register", str(exc))
            return
        self.volunteer = vol
        self.show_id()

    def show_id(self) -> None:
        vol = self.volunteer
        b = self._screen("id", 1, "Your volunteer ID",
                         "This ID, and nothing else, links you to your data.")
        ttk.Label(b, text=vol.volunteer_id, style="Big.TLabel").pack(pady=(30, 6))
        ttk.Label(b, text="Write this ID down or take a photo of this screen.\n"
                          "To withdraw later, give this ID to the project supervisor: "
                          "all your photos, points and details are then deleted.\n"
                          "Without the ID we cannot find your data.",
                  font=("Segoe UI", 12), justify="center", wraplength=800).pack(pady=10)
        bar = ttk.Frame(b)
        bar.pack(pady=24)
        ttk.Button(bar, text="Copy ID", command=lambda: self._copy(vol.volunteer_id)).pack(
            side="left", padx=8)
        ttk.Button(bar, text="I have noted it - continue  >", style="Accent.TButton",
                   command=self.show_capture).pack(side="left", padx=8)

    def _copy(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)

    # ----------------------------------------------------------- 3. capture
    def show_capture(self) -> None:
        vol = self.volunteer
        b = self._screen("capture", 2, f"Photos - volunteer {vol.volunteer_id}",
                         "Note this ID: it is the only way to withdraw your data later.")
        # The button bar is packed first so the two columns above do not squeeze it out.
        bar = ttk.Frame(b)
        bar.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(bar, text="Summary", command=self.show_summary).pack(side="left")
        ttk.Button(bar, text="Finish - next volunteer", command=self.new_volunteer).pack(
            side="left", padx=8)
        ttk.Button(bar, text="Next pose  >", command=self.next_pose).pack(side="right")
        ttk.Button(bar, text="Retake last", command=self.retake_last).pack(side="right", padx=8)
        ttk.Button(bar, text=f"Start ({self.countdown_s:g} s countdown, {self.burst_n} photos)",
                   style="Accent.TButton", command=self.start_capture).pack(side="right")

        right = ttk.Frame(b)                  # packed first so it is never squeezed out
        right.pack(side="right", fill="y", padx=(20, 0))
        left = ttk.Frame(b)
        left.pack(side="left", fill="both", expand=True)

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="Pose").pack(side="left")
        self.pose_cb = ttk.Combobox(row, textvariable=self.pose_var, state="readonly", width=44,
                                    values=list(self._pose_labels.values()))
        self.pose_cb.pack(side="left", padx=8)
        self.pose_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_guidance())
        if self.pose_var.get() not in self._pose_labels.values():
            self.pose_var.set(next(iter(self._pose_labels.values())))
        self.guidance = ttk.Label(left, text="", wraplength=520, justify="left")
        self.guidance.pack(anchor="w", pady=4)

        qf = ttk.Labelframe(left, text="This attempt is", style="Card.TLabelframe", padding=(10, 4))
        qf.pack(fill="x", pady=8)
        ttk.Radiobutton(qf, text="Right - a correct attempt", value="right",
                        variable=self.quality).pack(anchor="w")
        ttk.Radiobutton(qf, text="Wrong - a visible or deliberate fault (decided with the "
                                 "supervisor)", value="wrong", variable=self.quality
                        ).pack(anchor="w")
        fr = ttk.Frame(qf)
        fr.pack(fill="x", pady=(4, 2))
        ttk.Label(fr, text="What is wrong (optional, e.g. knees bent)").pack(side="left")
        self._fault_entry = ttk.Entry(fr, textvariable=self.fault, width=28)
        self._fault_entry.pack(side="left", padx=8)

        cam = ttk.Frame(left)
        cam.pack(fill="x", pady=4)
        ttk.Label(cam, text="Camera (number, or a video file path)").pack(side="left")
        ttk.Entry(cam, textvariable=self.camera, width=16).pack(side="left", padx=8)

        self.status_lbl = ttk.Label(left, text="", wraplength=520, justify="left",
                                    font=("Segoe UI", 11, "bold"))
        self.status_lbl.pack(anchor="w", pady=10)

        ttk.Label(right, text="Your photos so far", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.counts = ttk.Treeview(right, columns=("pose", "right", "wrong"), show="headings",
                                   height=len(POSES))
        for c, w, anchor in (("pose", 150, "w"), ("right", 58, "center"), ("wrong", 58, "center")):
            self.counts.heading(c, text=c.capitalize())
            self.counts.column(c, width=w, anchor=anchor)
        self.counts.pack()

        self._refresh_guidance()
        self._refresh_counts()
        self._sync_fault()
        self._set_status(*self._status)

    def _pose_key(self) -> str:
        for key, label in self._pose_labels.items():
            if label == self.pose_var.get():
                return key
        return next(iter(POSES))

    def select_pose(self, key: str) -> None:
        self.pose_var.set(self._pose_labels[key])
        if self.screen == "capture":
            self._refresh_guidance()

    def _refresh_guidance(self) -> None:
        pose = POSES[self._pose_key()]
        text = ("\n".join(f"  -  {s}" for s in pose.steps)
                + f"\n\n{VIEW_HINT[pose.position]}")
        self.guidance.configure(text=text)

    def _refresh_counts(self) -> None:
        self.counts.delete(*self.counts.get_children())
        counts = col.volunteer_counts(self.volunteer.volunteer_id)
        for key, pose in POSES.items():
            c = counts[key]
            self.counts.insert("", "end", values=(pose.name, c["right"], c["wrong"]))

    def _sync_fault(self) -> None:
        entry = self._fault_entry
        try:
            if entry is not None and entry.winfo_exists():
                entry.state(["!disabled"] if self.quality.get() == "wrong" else ["disabled"])
        except tk.TclError:
            pass

    def next_pose(self) -> None:
        keys = list(POSES)
        nxt = keys[(keys.index(self._pose_key()) + 1) % len(keys)]
        self.quality.set("right")
        self.fault.set("")
        self.last_paths = []
        self.select_pose(nxt)
        self._set_status(f"Next: {POSES[nxt].name}. Press Start when ready.", None)

    def retake_last(self) -> None:
        if not self.last_paths:
            self._set_status("Nothing to retake yet.", None)
            return
        n = 0
        for path in self.last_paths:
            try:
                n += bool(col.delete_sample(path))
            except (ValueError, OSError):
                pass
        self.last_paths = []
        self._refresh_counts()
        self._set_status(f"Deleted the last {n} photo(s). Press Start to retake.", None)

    # ---- the camera side: each of these three can be replaced in a test -----
    def _open_camera(self):
        """(capture, is_live) via the trainer's own helper, so it behaves identically."""
        import app as trainer
        return trainer.open_source(self.camera.get().strip() or "0", CAPTURE_W, CAPTURE_H)

    def _get_tracker(self):
        if self._tracker is None:
            # Image mode, no smoothing: each photo is judged alone, the same way
            # tools/fitting/fit_asana.py will later re-read the saved images.
            self._tracker = PoseTracker(model="full", running_mode="image", smooth=False)
        return self._tracker

    def _preview_fn(self):
        import app as trainer

        def show(image: np.ndarray) -> bool:
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(1) & 0xFF
            return key not in (27, ord("q")) and not trainer.window_closed(WINDOW)
        return show

    def _capture(self, caption: str) -> list[Shot]:
        cap, live = self._open_camera()
        try:
            return grab_burst(cap, self._get_tracker(), n=self.burst_n,
                              countdown_s=self.countdown_s, interval_s=self.interval_s,
                              mirror=bool(live), show=self._preview_fn(), caption=caption)
        finally:
            cap.release()
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass

    def start_capture(self) -> None:
        if self.volunteer is None:
            return
        key, quality = self._pose_key(), self.quality.get()
        fault = self.fault.get() if quality == "wrong" else ""
        self._set_status("Opening the camera...", None)
        self.update_idletasks()
        self.withdraw()                      # hide this window, like the launcher does
        try:
            shots = self._capture(f"{POSES[key].name} - {quality}")
        except (SystemExit, Exception) as exc:     # camera missing etc: keep the session
            self.deiconify()
            self._set_status("The camera could not be used.", False)
            self._warn("Camera problem", str(exc) or "The camera could not be opened.")
            return
        self.deiconify()
        if not shots:
            self._set_status("Cancelled - nothing was saved.", None)
            return
        self._store_shots(shots, key, quality, fault)

    def _store_shots(self, shots: list[Shot], key: str, quality: str, fault: str) -> dict:
        """Save the photos with a body in them and tell the volunteer what happened."""
        vid = self.volunteer.volunteer_id
        paths: list[str] = []
        errors: list[str] = []
        for s in shots:
            if s.pose is None:               # nothing to measure, nothing to keep
                continue
            try:
                paths.append(col.save_sample(vid, key, s.frame, s.pose, quality, fault))
            except (col.CollectionError, OSError) as exc:
                errors.append(str(exc))
        found = [s for s in shots if s.pose is not None]
        cut = sum(1 for s in found if joints_outside(s.pose))
        self.last_paths = paths
        self.last_result = {"shots": len(shots), "detected": len(found), "saved": len(paths),
                            "cut_off": cut, "errors": errors}
        msg = f"Body detected in {len(found)} of {len(shots)} photos - {len(paths)} saved."
        if not found:
            msg += " Step into frame and press Start to retake."
        elif cut:
            msg += (f" Part of the body was outside the picture in {cut} of them - "
                    "step back or move the camera, then Retake last.")
        if errors:
            msg += " " + errors[0]
        self._set_status(msg, bool(paths) and not errors and not cut)
        self._refresh_counts()
        return self.last_result

    # ----------------------------------------------------------- 4. summary
    def show_summary(self) -> None:
        b = self._screen("summary", 3, "Summary and diversity",
                         "Counts per pose and volunteer; the list at the bottom says "
                         "where the sample is still thin.")
        box = ttk.Frame(b)
        box.pack(fill="both", expand=True)
        txt = tk.Text(box, wrap="none", bg="white", relief="flat", font=("Consolas", 10),
                      padx=12, pady=8)
        sb = ttk.Scrollbar(box, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.insert("1.0", col.summary().text())
        txt.configure(state="disabled")
        self.summary_view = txt

        bar = ttk.Frame(b)
        bar.pack(fill="x", pady=(10, 0))
        ttk.Button(bar, text="Withdraw a volunteer", command=self.ask_withdraw).pack(side="left")
        ttk.Button(bar, text="Export right-labelled photos for fitting",
                   command=self.export_photos).pack(side="left", padx=8)
        if self.volunteer is not None:
            ttk.Button(bar, text="Back to photos", command=self.show_capture).pack(side="right")
        ttk.Button(bar, text="New volunteer", style="Accent.TButton",
                   command=self.new_volunteer).pack(side="right", padx=8)

    def export_photos(self) -> None:
        try:
            res = col.export_for_fitting(col.EXPORT_DIR)
        except (ValueError, OSError) as exc:
            self._warn("Export failed", str(exc))
            return
        rel = os.path.relpath(col.EXPORT_DIR, HERE)
        lines = [f"  {k}: {n}" for k, n in res["per_pose"].items() if n] or ["  (nothing yet)"]
        text = (f"Copied {res['total']} right-labelled photos from {res['volunteers']} "
                f"volunteers to\n{col.EXPORT_DIR}\n\n" + "\n".join(lines)
                + (f"\n\n{res['skipped_no_image']} right-labelled sample(s) have no photo "
                   "(points-only volunteers)." if res["skipped_no_image"] else "")
                + "\n\nTo refit, write to a separate file so the current references are kept:"
                  f"\n  python tools/fitting/fit_asana.py {rel} --out data/asana_fits_collected.json")
        self._info("Export for fitting", text)

    # ------------------------------------------------------------ withdraw
    def ask_withdraw(self) -> None:
        raw = simpledialog.askstring("Withdraw a volunteer",
                                     "Type the volunteer ID (for example V-3F9A2C):", parent=self)
        if not raw:
            return
        vid = raw.strip().upper()
        if not col.ID_RE.fullmatch(vid):
            self._warn("Not an ID", "A volunteer ID looks like V-3F9A2C (V, a dash, six "
                                    "letters or digits from 0-9 and A-F).")
            return
        if not self._confirm("Withdraw a volunteer",
                             f"Delete ALL photos, body points and details for {vid}?\n"
                             "This cannot be undone."):
            return
        _ok, message = self.do_withdraw(vid)
        self._info("Withdraw a volunteer", message)
        if self.volunteer is not None and self.volunteer.volunteer_id == vid:
            self.new_volunteer()
        elif self.screen == "summary":
            self.show_summary()

    def do_withdraw(self, volunteer_id: str) -> tuple[bool, str]:
        """Delete a volunteer (and their export copies); (done, message to show)."""
        try:
            done = col.withdraw(volunteer_id, also_in=col.EXPORT_DIR)
        except (ValueError, OSError) as exc:
            return False, f"Could not withdraw {volunteer_id}: {exc}"
        if done:
            return True, (f"All data for {volunteer_id} has been deleted, including any "
                          "copies made for fitting.")
        return False, f"No volunteer with ID {volunteer_id} was found."


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
