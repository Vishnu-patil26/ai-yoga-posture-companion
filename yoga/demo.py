"""Looping demonstration of each pose, drawn from the real pose data.

Each pose's final shape is a real photograph's joints: the typical (medoid)
shape of that pose in data/landmarks, chosen by tools/fitting/build_demo.py and
stored in data/demo_poses.json.  The animation moves a stick figure from a plain
starting position into that shape, holds it breathing, and comes back:

    standing  ->  Triangle, Warrior II, Chair, Tree
    all fours ->  Downward Dog
    face down ->  Cobra
    kneeling  ->  Child's pose
    Mountain  ->  arms down, sweep them overhead, hold, lower them
    (Easy pose starts where it ends, so it just breathes; Cat-Cow moves between its
     two real shapes, cat and cow)

The figure keeps every bone at its real length: it is built from joint *angles*
(forward kinematics), not by sliding points, and the angles are the ones measured
on the photograph.  The starting positions are generic illustration, not part of
any scoring.  Nothing here touches the camera or the scorer.

    from yoga import demo
    clip = demo.Demo("utkatasana")
    frame = clip.render(480, 360, t)          # a BGR picture of the figure at time t
"""

from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

from . import theme
from .draw import Pen

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "demo_poses.json")

#: seconds: hold the start, move in, hold the pose (breathing), move back
T_START, T_IN, T_HOLD, T_OUT = 1.0, 2.2, 4.0, 1.8
LOOP_S = T_START + T_IN + T_HOLD + T_OUT
CAT_COW_S = 6.0

# absolute segment angles in degrees, screen coordinates: 0 = right, 90 = down, -90 = up
STAND = dict(T=-90, ua_L=90, fa_L=90, ua_R=90, fa_R=90, th_L=90, sh_L=90, th_R=90, sh_R=90)
ALL_FOURS = dict(T=-3, ua_L=90, fa_L=90, ua_R=90, fa_R=90, th_L=90, sh_L=180, th_R=90, sh_R=180)
PRONE = dict(T=0, ua_L=135, fa_L=40, ua_R=135, fa_R=40, th_L=180, sh_L=180, th_R=180, sh_R=180)
KNEEL = dict(T=-90, ua_L=90, fa_L=90, ua_R=90, fa_R=90, th_L=90, sh_L=180, th_R=90, sh_R=180)

START = {"trikonasana": STAND, "virabhadrasana": STAND, "utkatasana": STAND, "vrikshasana": STAND,
         "adho_mukha": ALL_FOURS, "bhujangasana": PRONE, "balasana": KNEEL}

_END_JOINT = {"ua": "ELBOW", "fa": "WRIST", "th": "KNEE", "sh": "ANKLE"}
KEYS = ("T", "ua_L", "fa_L", "ua_R", "fa_R", "th_L", "sh_L", "th_R", "sh_R")
BONES = (("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_HIP", "RIGHT_HIP"),
         ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
         ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
         ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
         ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
         ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
         ("LEFT_ANKLE", "LEFT_HEEL"), ("LEFT_HEEL", "LEFT_FOOT_INDEX"), ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
         ("RIGHT_ANKLE", "RIGHT_HEEL"), ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"), ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"))
JOINTS_DRAWN = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
                "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE")

_DB: dict | None = None


def available() -> bool:
    return os.path.isfile(DATA)


def _db() -> dict:
    global _DB
    if _DB is None:
        with open(DATA, encoding="utf-8") as fh:
            _DB = json.load(fh)["poses"]
    return _DB


def _ang(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _smooth(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


class _Shape:
    """One real pose turned into lengths, local offsets and absolute segment angles."""

    def __init__(self, pts: dict[str, list[float]]) -> None:
        p = {k: np.array(v, dtype=float) for k, v in pts.items()}
        self.H = (p["LEFT_HIP"] + p["RIGHT_HIP"]) / 2
        self.S = (p["LEFT_SHOULDER"] + p["RIGHT_SHOULDER"]) / 2
        torso = self.S - self.H
        self.lt = float(np.linalg.norm(torso))
        self.params = {"T": _ang(torso)}
        u, v = self._basis(self.params["T"])
        self.hip_off = {s: self._local(p[f"{s}_HIP"] - self.H, u, v) for s in ("LEFT", "RIGHT")}
        self.sho_off = {s: self._local(p[f"{s}_SHOULDER"] - self.S, u, v) for s in ("LEFT", "RIGHT")}
        self.nose_off = self._local(p["NOSE"] - self.S, u, v)
        self.len, self.foot = {}, {}
        for side, c in (("LEFT", "L"), ("RIGHT", "R")):
            for name, a, b in (("ua", "SHOULDER", "ELBOW"), ("fa", "ELBOW", "WRIST"),
                               ("th", "HIP", "KNEE"), ("sh", "KNEE", "ANKLE")):
                d = p[f"{side}_{b}"] - p[f"{side}_{a}"]
                self.params[f"{name}_{c}"] = _ang(d)
                self.len[f"{name}_{c}"] = float(np.linalg.norm(d))
            u2, v2 = self._basis(self.params[f"sh_{c}"])
            self.foot[side] = {k: self._local(p[f"{side}_{k}"] - p[f"{side}_ANKLE"], u2, v2)
                               for k in ("HEEL", "FOOT_INDEX")}

    @staticmethod
    def _basis(angle: float):
        a = math.radians(angle)
        u = np.array([math.cos(a), math.sin(a)])
        return u, np.array([-u[1], u[0]])

    @staticmethod
    def _local(d, u, v):
        return np.array([float(d @ u), float(d @ v)])

    def pose(self, prm: dict[str, float]) -> dict[str, np.ndarray]:
        """Joint positions for these segment angles (hips at the origin, y down)."""
        out: dict[str, np.ndarray] = {}
        u, v = self._basis(prm["T"])
        H = np.zeros(2)
        S = H + self.lt * u
        for side, c in (("LEFT", "L"), ("RIGHT", "R")):
            hip = H + self.hip_off[side][0] * u + self.hip_off[side][1] * v
            sho = S + self.sho_off[side][0] * u + self.sho_off[side][1] * v
            out[f"{side}_HIP"], out[f"{side}_SHOULDER"] = hip, sho
            for root, bones in ((sho, ("ua", "fa")), (hip, ("th", "sh"))):
                cur = root
                for name in bones:
                    a = math.radians(prm[f"{name}_{c}"])
                    cur = cur + self.len[f"{name}_{c}"] * np.array([math.cos(a), math.sin(a)])
                    out[f"{side}_{_END_JOINT[name]}"] = cur
            fu, fv = self._basis(prm[f"sh_{c}"])
            for k, off in self.foot[side].items():
                out[f"{side}_{k}"] = out[f"{side}_ANKLE"] + off[0] * fu + off[1] * fv
        out["NOSE"] = S + self.nose_off[0] * u + self.nose_off[1] * v
        return out


class Demo:
    """The looping clip for one pose."""

    def __init__(self, key: str) -> None:
        entry = _db()[key]
        self.key, self.view = key, entry["view"]
        shapes = entry["shapes"]
        self.cat_cow = key == "marjaryasana"
        if self.cat_cow:
            self.a, self.b = _Shape(shapes["cat"]["pts"]), _Shape(shapes["cow"]["pts"])
            self.start = dict(self.a.params)
            self.end = dict(self.b.params)
            self.shape = self.a
        else:
            self.shape = _Shape(shapes["shape"]["pts"])
            self.end = dict(self.shape.params)
            self.start = dict(START.get(key, self.end))
            if key == "tadasana":
                # Mountain: start with the arms down (the real shape), sweep them overhead and back.
                # An arm up is the same bone pointing the other way up, on the same side of the body.
                self.start = dict(self.end)
                for k in ("ua_L", "fa_L", "ua_R", "fa_R"):
                    self.end[k] = -self.end[k]
        self.moves = self.cat_cow or any(abs(_wrap(self.end[k] - self.start[k])) > 8 for k in KEYS)
        self.duration = CAT_COW_S if self.cat_cow else LOOP_S
        self._extent = self._measure()

    # ------------------------------------------------------------------ timeline
    def progress(self, t: float) -> float:
        """0 = start position, 1 = the pose."""
        t = t % self.duration
        if self.cat_cow:
            return 0.5 - 0.5 * math.cos(2 * math.pi * t / CAT_COW_S)
        if not self.moves:
            return 1.0
        if t < T_START:
            return 0.0
        if t < T_START + T_IN:
            return _smooth((t - T_START) / T_IN)
        if t < T_START + T_IN + T_HOLD:
            return 1.0
        return 1.0 - _smooth((t - T_START - T_IN - T_HOLD) / T_OUT)

    def step(self, t: float) -> int:
        """Which of the four captions this moment illustrates: 0 the start position, 1 moving
        into the pose, 2 holding it, 3 coming out (yoga/instructions.py has the words).  Cat-Cow
        uses 1 for cow and 2 for cat."""
        u = self.progress(t)
        if self.cat_cow:
            return 1 if u > 0.5 else 2
        t = t % self.duration
        if not self.moves:
            return int(min(3, 4 * t / self.duration))
        if t < T_START + 0.4:
            return 0
        if t >= T_START + T_IN + T_HOLD:
            return 3
        return 1 if u < 0.97 else 2

    def params(self, t: float) -> dict[str, float]:
        u = self.progress(t)
        prm = {k: self.start[k] + _wrap(self.end[k] - self.start[k]) * u for k in KEYS}
        breathe = math.sin(2 * math.pi * (t % 4.0) / 4.0) * (1.0 if u > 0.95 else 0.0)
        prm["T"] += 0.8 * breathe
        for k in ("ua_L", "ua_R"):
            prm[k] += 1.2 * breathe
        return prm

    def points(self, t: float) -> dict[str, np.ndarray]:
        if self.cat_cow:
            u = self.progress(t)
            pa, pb = self.a.pose(self.a.params), self.b.pose(self.b.params)
            pts = {j: pa[j] + (pb[j] - pa[j]) * u for j in pa}
        else:
            pts = self.shape.pose(self.params(t))
        ys = [p[1] for j, p in pts.items() if j != "NOSE"]
        xs = [p[0] for p in pts.values()]
        shift = np.array([-(max(xs) + min(xs)) / 2, -max(ys)])      # feet / hands on the ground at y = 0
        return {j: p + shift for j, p in pts.items()}

    def head(self, pts: dict) -> tuple[np.ndarray, float]:
        """Centre and radius of the drawn head, from the nose and shoulders."""
        sho = (pts["LEFT_SHOULDER"] + pts["RIGHT_SHOULDER"]) / 2
        return pts["NOSE"] + (pts["NOSE"] - sho) * 0.25, self.shape.lt * 0.2

    def _measure(self):
        lo, hi = np.array([1e9, 1e9]), np.array([-1e9, -1e9])
        for i in range(48):
            pts = self.points(self.duration * i / 48)
            for p in pts.values():
                lo, hi = np.minimum(lo, p), np.maximum(hi, p)
            c, r = self.head(pts)
            lo, hi = np.minimum(lo, c - r), np.maximum(hi, c + r)
        return lo, hi

    # ------------------------------------------------------------------ drawing
    def render(self, w: int, h: int, t: float, *, caption: str = "", step: int | None = None,
               title: str = "", compact: bool = False) -> np.ndarray:
        """A picture of the figure at time ``t`` on the project's black background.

        ``compact`` is for a small card: no ground label, thinner lines.  ``caption``
        is drawn under the figure (and ``step`` numbers it)."""
        img = np.empty((h, w, 3), np.uint8)
        img[:] = theme.bgr(theme.TOKENS["bg"])
        top = 30 if (title and not compact) else 10
        bottom = (62 if caption else 14) if not compact else 12
        lo, hi = self._extent
        span = np.maximum(hi - lo, 1e-3)
        s = min((w - 24) / span[0], (h - top - bottom - 4) / span[1])
        ox = w / 2 - (lo[0] + hi[0]) / 2 * s
        oy = h - bottom - 2

        def px(p):
            return int(round(ox + p[0] * s)), int(round(oy + p[1] * s))

        pts = self.points(t)
        ground = theme.bgr(theme.TOKENS["border2"])
        cv2.line(img, (12, oy), (w - 12, oy), ground, 1)
        bone = theme.bgr("#e6e6e6")
        accent = theme.bgr(theme.TOKENS["accent"])
        thick = 2 if compact else 3
        for a, b in BONES:
            cv2.line(img, px(pts[a]), px(pts[b]), bone, thick, cv2.LINE_AA)
        r = max(2, int(s * 0.035)) if compact else max(3, int(s * 0.04))
        for j in JOINTS_DRAWN:
            cv2.circle(img, px(pts[j]), r, accent, -1, cv2.LINE_AA)
        hc, hr = self.head(pts)
        sc = (pts["LEFT_SHOULDER"] + pts["RIGHT_SHOULDER"]) / 2
        d = hc - sc
        if np.linalg.norm(d) > hr * 1.05:                          # a neck from the shoulders to the head
            cv2.line(img, px(sc), px(hc - d / np.linalg.norm(d) * hr), bone, thick, cv2.LINE_AA)
        cv2.circle(img, px(hc), max(5, int(hr * s)), accent, thick - 1 if compact else 2, cv2.LINE_AA)
        if not compact:
            with Pen(img, 0, 0, w, h) as pen:
                if title:
                    pen.text((14, 20), title, 10, theme.bgr(theme.TOKENS["muted"]), "mono", True, track=1.6)
                    pen.text((w - 14, 20), "SIDE VIEW" if self.view == "side" else "FRONT VIEW", 10,
                             theme.bgr(theme.TOKENS["muted"]), "mono", False, "r", 1.4)
                if self.cat_cow:
                    pen.text((w / 2, 44), "COW  (breathe in)" if self.progress(t) > 0.5 else "CAT  (breathe out)",
                             11, accent, "mono", True, "m", 1.4)
                if caption:
                    n = (step if step is not None else self.step(t)) + 1
                    lines = Pen.wrap(caption, w - 52, 13)[:2]
                    pen.text((14, h - 40), f"{n}", 12, accent, "mono", True)
                    for i, line in enumerate(lines):
                        pen.text((32, h - 40 + i * 17), line, 13, theme.bgr(theme.TOKENS["text"]))
        return img


_CACHE: dict[str, Demo] = {}


def clip(key: str) -> Demo | None:
    """The demo for a pose, or None when the data file or the pose is missing."""
    if key not in _CACHE:
        try:
            _CACHE[key] = Demo(key)
        except (OSError, KeyError, ValueError):
            return None
    return _CACHE[key]
