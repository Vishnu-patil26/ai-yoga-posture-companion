"""On-screen feedback (Methodology Stage 2 'live skeleton overlay').

Draws the skeleton, colours the limb that is actually wrong, and shows the
joint-by-joint table so a panel member can see *why* the score is what it is
instead of trusting a single number.

The look is the project's black theme (yoga/theme.py): near-black panels with
hairline borders, square corners, monospaced capitals for labels, sentences in
a clean sans, white as the one accent, and three verdict colours that mean the
same thing everywhere - mint = correct, red = fix this, amber = cannot see.
Text is drawn by yoga/draw.py with real TrueType fonts.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import phrasing, theme
from .draw import Pen, bar, panel, rule
from .evaluator import FRAME_FILL_MAX, FRAME_FILL_MIN, Evaluation
from .landmarks import (
    CONNECTIONS, L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, Pose,
)
from .state_machine import PoseStateMachine, State

FONT = cv2.FONT_HERSHEY_SIMPLEX          # only the calibration prompt still needs it

# BGR
_T = theme.TOKENS
COL_OK = theme.bgr(_T["good"])
COL_WARN = theme.bgr(_T["warn"])
COL_BAD = theme.bgr(_T["bad"])
COL_BONE = theme.bgr("#e6e6e6")
COL_RAW = (110, 110, 110)
COL_TEXT = theme.bgr(_T["text"])
COL_DIM = theme.bgr(_T["muted"])
COL_FAINT = theme.bgr(_T["dim"])
COL_ACCENT = theme.bgr(_T["accent"])
COL_CYAN = theme.bgr(_T["accent2"])
COL_RULE = theme.bgr(_T["border"])
COL_RULE2 = theme.bgr(_T["border2"])
COL_PANEL = theme.bgr(_T["bg"])
COL_RAISED = theme.bgr(_T["card2"])

PANEL_W = 330
CARD_W, CARD_H = 250, 300
GUIDE_CARD_H = CARD_H            #: the pose guide card: the step list above it needs the rest of the column
FRAMING_H = 182                   #: height of the "can the camera see you" panel
MAX_STEP_ROWS = 6                 #: the step list must end above the card that sits below it
COL_CARD_BONE = COL_BONE

MONO = "mono"


def _score_colour(score: float) -> tuple[int, int, int]:
    if score >= 78:
        return COL_OK
    if score >= 60:
        return COL_WARN
    return COL_BAD


#: Every anatomical segment the app grades, keyed by a stable id rather than
#: by landmark index - a check like "foot_height_ratio" and "folded_knee" both
#: touch the folded shin, and the id is what lets their two verdicts be
#: combined into one colour for that one bone.
_CHECK_SEGMENTS = {
    "spine_tilt": ("spine",),
    "hip_level": ("hip_line",),
    "shoulder_level": ("shoulder_line",),
    "standing_knee": ("standing_thigh", "standing_shin"),
    "folded_knee": ("folded_thigh", "folded_shin"),
    "folded_thigh_open": ("folded_thigh",),
    "foot_height_ratio": ("folded_shin",),
    "arm_raise_left": ("left_upper_arm", "left_forearm"),
    "arm_raise_right": ("right_upper_arm", "right_forearm"),
    "wrist_to_chest_left": ("left_forearm",),
    "wrist_to_chest_right": ("right_forearm",),
}

_STATUS_RANK = {"ok": 0, "ungraded": 1, "fail": 2}


def _segment_points(pose: Pose, feats: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pixel endpoints of every segment named in `_CHECK_SEGMENTS`."""
    p = pose.pts
    standing = feats.get("standing_side", "left")
    folded = feats.get("folded_side", "right")
    s_hip, s_knee, s_ankle = ((L_HIP, L_KNEE, L_ANKLE) if standing == "left"
                              else (R_HIP, R_KNEE, R_ANKLE))
    d_hip, d_knee, d_ankle = ((L_HIP, L_KNEE, L_ANKLE) if folded == "left"
                              else (R_HIP, R_KNEE, R_ANKLE))
    mid_sh = (p[L_SHOULDER] + p[R_SHOULDER]) * 0.5
    mid_hip = (p[L_HIP] + p[R_HIP]) * 0.5
    return {
        "spine": (mid_hip, mid_sh),
        "hip_line": (p[L_HIP], p[R_HIP]),
        "shoulder_line": (p[L_SHOULDER], p[R_SHOULDER]),
        "standing_thigh": (p[s_hip], p[s_knee]),
        "standing_shin": (p[s_knee], p[s_ankle]),
        "folded_thigh": (p[d_hip], p[d_knee]),
        "folded_shin": (p[d_knee], p[d_ankle]),
        "left_upper_arm": (p[L_SHOULDER], p[L_ELBOW]),
        "left_forearm": (p[L_ELBOW], p[L_WRIST]),
        "right_upper_arm": (p[R_SHOULDER], p[R_ELBOW]),
        "right_forearm": (p[R_ELBOW], p[R_WRIST]),
    }


def _segment_status(ev: Evaluation) -> dict[str, str]:
    """Worst verdict touching each segment: "ok", "fail", or "ungraded".

    A segment can be named by more than one check (the folded shin is graded
    by both `folded_knee` and `foot_height_ratio`); when their verdicts
    disagree, the segment is drawn as whichever is worse - a bone is only
    shown correct when *everything* measured on it is correct.
    """
    status: dict[str, str] = {}
    for res in ev.results:
        if res.check.weight <= 0.0:            # a dormant variant placeholder
            continue
        state = "fail" if res.ok is False else ("ungraded" if res.ok is None else "ok")
        for seg in _CHECK_SEGMENTS.get(res.check.key, ()):
            if seg not in status or _STATUS_RANK[state] > _STATUS_RANK[status[seg]]:
                status[seg] = state
    return status


def _dashed_line(frame: np.ndarray, a, b, colour, thickness: int = 5,
                 dash: float = 11.0, gap: float = 8.0) -> None:
    """A dashed bone reads as 'unknown' at a glance, not just a duller colour -
    useful for a practitioner who has not registered what the solid colours
    mean yet, and for anyone colour-blind to red/green."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    if length < 1e-3:
        return
    direction = (b - a) / length
    pos, draw = 0.0, True
    while pos < length:
        end = min(pos + (dash if draw else gap), length)
        if draw:
            p1 = tuple((a + direction * pos).astype(int))
            p2 = tuple((a + direction * end).astype(int))
            cv2.line(frame, p1, p2, colour, thickness, cv2.LINE_AA)
        pos, draw = end, not draw


def draw_skeleton(frame: np.ndarray, pose: Pose, ev: Evaluation | None,
                  show_raw: bool = False) -> None:
    p = pose.pts.astype(int)

    if show_raw:
        for a, b in CONNECTIONS:
            cv2.line(frame, tuple(pose.raw[a].astype(int)), tuple(pose.raw[b].astype(int)),
                     COL_RAW, 1, cv2.LINE_AA)

    for a, b in CONNECTIONS:
        if pose.vis[a] < 0.4 or pose.vis[b] < 0.4:
            continue
        cv2.line(frame, tuple(p[a]), tuple(p[b]), COL_BONE, 3, cv2.LINE_AA)
    for i in range(len(p)):
        if pose.vis[i] < 0.4:
            continue
        cv2.circle(frame, tuple(p[i]), 4, COL_ACCENT, -1, cv2.LINE_AA)

    if ev is None:
        return

    # Every graded bone is coloured by its own verdict: mint when correct,
    # red when wrong, dashed amber when it could not be judged at all.  This
    # is the whole graded structure at a glance, not just the worst fault.
    points = _segment_points(pose, ev.features)
    status = _segment_status(ev)
    for seg_id, state in status.items():
        if seg_id not in points:
            continue
        a, b = points[seg_id]
        if state == "ok":
            cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), COL_OK, 5, cv2.LINE_AA)
        elif state == "fail":
            cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), COL_BAD, 6, cv2.LINE_AA)
        else:                                        # ungraded - "incomplete"
            _dashed_line(frame, a, b, COL_WARN, thickness=5)


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.88,
           border: str | None = "border2") -> None:
    panel(frame, x, y, w, h, alpha, border)


def _bar(frame, x, y, w, h, frac, colour) -> None:
    bar(frame, x, y, w, h, frac, colour)


def _label(pen: Pen, xy, s: str, colour=COL_DIM, size: int = 10, anchor: str = "l") -> None:
    """The small monospaced capital that names a block (ALIGNMENT, HOLD ...)."""
    pen.text(xy, s.upper(), size, colour, MONO, False, anchor, track=1.6)


def draw_hud(frame: np.ndarray, ev: Evaluation | None, sm: PoseStateMachine,
             cue_text: str, fps: float, hint: str = "",
             right_margin: int = 0, metric: bool = False) -> None:
    """`right_margin` keeps the bottom banner clear of the reference card.

    `metric` shows the old raw degrees/ratio numbers instead of body-relative
    words - kept for anyone checking the engine's numbers directly, but the
    practitioner-facing default is words, never a measurement scale.
    """
    h, w = frame.shape[:2]
    asana = ev.asana if ev else None
    px_w = PANEL_W

    _panel(frame, 0, 0, px_w + 1, h, alpha=0.94, border=None)
    cv2.line(frame, (px_w, 0), (px_w, h), COL_RULE2, 1)
    rule(frame, 16, 44, 40, 2)

    score = ev.score if (ev and ev.usable) else 0.0
    colour = _score_colour(score)
    state_txt = sm.state.value if ev else "WAITING"

    # ---- shapes first (bars, hairlines, tag boxes), text last
    y_score_bar = 186
    _bar(frame, 16, y_score_bar, px_w - 32, 4, score / 100.0, colour)
    y_hold_bar = 232
    _bar(frame, 16, y_hold_bar, px_w - 32, 4, sm.progress,
         COL_OK if sm.progress >= 1 else COL_ACCENT)
    cv2.line(frame, (16, 108), (px_w - 16, 108), COL_RULE, 1)

    tag_w = int(Pen.width(state_txt, 11, MONO, True, 1.2)) + 18
    tag_x1 = px_w - 16
    cv2.rectangle(frame, (tag_x1 - tag_w, 128), (tag_x1, 150), colour, 1)

    rows = []
    if ev:
        rows = [r for r in ev.results if r.check.weight > 0.0]
    y0 = 270
    cv2.line(frame, (16, y0 + 8), (px_w - 16, y0 + 8), COL_RULE, 1)
    row_y = y0 + 32
    stats_y = h - 176
    rh = int(min(22, max(16, (stats_y - 62 - row_y) / max(1, len(rows)))))
    for i, res in enumerate(rows):
        col = COL_WARN if res.ok is None else (COL_OK if res.ok else COL_BAD)
        cy = row_y + i * rh - 5
        cv2.rectangle(frame, (px_w - 24, cy - 3), (px_w - 18, cy + 3), col, -1)

    with Pen(frame, 0, 0, px_w, h) as pen:
        pen.text((16, 34), "AI YOGA COMPANION", 12, COL_TEXT, MONO, True, track=2.0)
        if asana:
            pen.text((16, 76), asana.sanskrit, 19, COL_TEXT, "sans", True)
            pen.text((16, 98), f"{asana.name}  |  level {asana.level}  |  hold {asana.hold_target_s:.0f}s",
                     11, COL_DIM, MONO)
        else:
            pen.text((16, 76), "Waiting for a pose", 17, COL_DIM, "sans", True)

        _label(pen, (16, 134), "alignment")
        pen.text((16, 178), f"{score:.1f}%", 38, colour, "sans", True)
        pen.text((tag_x1 - tag_w // 2, 144), state_txt, 11, colour, MONO, True, "m", 1.2)

        _label(pen, (16, 222), "hold")
        pen.text((px_w - 16, 222), f"{sm.elapsed:.1f}s / {sm.hold_target_s:.0f}s", 12,
                 COL_TEXT, MONO, False, "r")

        if ev:
            _label(pen, (16, y0), "joint")
            _label(pen, (px_w - 36, y0), "metric" if metric else "how far off", anchor="r")
            for i, res in enumerate(rows):
                col = COL_WARN if res.ok is None else (COL_OK if res.ok else COL_BAD)
                words = res.display() if metric else phrasing.short_words(res, ev.features)
                yy = row_y + i * rh
                pen.text((16, yy), Pen.clip(res.label, 160, 13), 13, COL_TEXT if res.ok else col)
                pen.text((px_w - 36, yy), Pen.clip(words, 130, 11, MONO), 11, col, MONO, False, "r")
            yy = row_y + len(rows) * rh + 10
            if ev.asana.key == "vrikshasana":                 # only Tree has a standing leg
                pen.text((16, yy), f"standing leg: {ev.features.get('standing_side', '?')}",
                         11, COL_DIM, MONO)
                yy += 16
            if ev.variant_name:
                pen.text((16, yy), f"form: {ev.variant_name}", 11, COL_DIM, MONO)
                yy += 16
            if (ev.asana.key == "vrikshasana"
                    and not ev.features.get("group_visibility", {}).get("frontal", 1.0)):
                pen.text((16, yy), "turn to face the camera", 11, COL_WARN, MONO)
                yy += 16

        st = sm.stats
        pen.text((16, stats_y), f"holds completed  {st.completed}", 11, COL_DIM, MONO)
        pen.text((16, stats_y + 16), f"longest hold     {st.longest_s:.1f}s", 11, COL_DIM, MONO)
        pen.text((16, stats_y + 32), f"best alignment   {st.best_score:.1f}%", 11, COL_DIM, MONO)

        if ev:
            for j, (txt, c) in enumerate((("correct", COL_OK), ("fix this", COL_BAD),
                                          ("dashed = can't see", COL_WARN))):
                pen.text((32, h - 108 + j * 15), txt, 10, c, MONO)
        pen.text((16, h - 52), f"{fps:4.1f} fps", 10, COL_DIM, MONO)
        pen.text((16, h - 38), "q quit   v voice   a repeat", 10, COL_FAINT, MONO)
        pen.text((16, h - 24), "n next  b back  l poses  d demo", 10, COL_FAINT, MONO)
        pen.text((16, h - 10), "g guide  m metric  r reset  s snap", 10, COL_FAINT, MONO)
    if ev:
        for j, c in enumerate((COL_OK, COL_BAD, COL_WARN)):
            cv2.rectangle(frame, (16, h - 117 + j * 15), (23, h - 110 + j * 15), c, -1)

    banner = cue_text or hint
    if banner:
        bw = max(120, w - px_w - right_margin)
        # Wrap rather than truncate - a cue cut off halfway is worse than none.
        lines = Pen.wrap(banner, bw - 44, 21, "sans", True)[:2]
        bh = 24 + 32 * len(lines)
        _panel(frame, px_w, h - bh, bw, bh, alpha=0.86, border=None)
        cv2.line(frame, (px_w, h - bh), (px_w + bw, h - bh), COL_RULE2, 1)
        col = COL_OK if sm.state is State.COMPLETE else COL_TEXT
        with Pen(frame, px_w, h - bh, px_w + bw, h) as pen:
            ly = h - bh + 36
            for line in lines:
                pen.text((px_w + 22, ly), line, 21, col, "sans", True)
                ly += 32

    if sm.state is State.COMPLETE:
        label = "HOLD COMPLETE"
        tw = int(Pen.width(label, 14, MONO, True, 2.0)) + 28
        cv2.rectangle(frame, (px_w + 24, 56), (px_w + 24 + tw, 90), COL_OK, 1)
        with Pen(frame, px_w, 50, px_w + tw + 60, 100) as pen:
            pen.text((px_w + 38, 78), label, 14, COL_OK, MONO, True, track=2.0)


# =============================================================================
# Guidance panels: what to do next, and whether the camera can see you.
# =============================================================================

def _draw_figure(frame: np.ndarray, pose: Pose, x: int, y: int, w: int, h: int,
                 colour=COL_CARD_BONE) -> None:
    """Draw a reference skeleton scaled to fit the given box."""
    p = pose.pts
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    scale = min(w / span[0], h / span[1])
    off = np.array([x + (w - span[0] * scale) / 2.0, y + (h - span[1] * scale) / 2.0])
    q = ((p - lo) * scale + off).astype(int)

    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(q[a]), tuple(q[b]), colour, 2, cv2.LINE_AA)
    for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE,
              L_ANKLE, R_ANKLE, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST):
        cv2.circle(frame, tuple(q[i]), 3, COL_ACCENT, -1, cv2.LINE_AA)


def draw_reference_card(frame: np.ndarray, pose: Pose, step_no: int, step_total: int,
                        title: str, instruction: str, done: bool = False,
                        confirming: bool = False, confirm_fraction: float = 0.0,
                        step_seconds: float = 0.0) -> None:
    """Bottom-right card: the shape to make, and the instruction, right now.

    A practitioner cannot read a joint-angle table while balancing on one leg
    with their eyes half closed.  They can glance at a figure.  The figure is
    built by the same code the scorer grades against, so what is shown is by
    construction what is accepted.
    """
    h, w = frame.shape[:2]
    x, y = w - CARD_W - 16, h - CARD_H - 16
    _panel(frame, x, y, CARD_W, CARD_H, alpha=0.88)
    cv2.line(frame, (x + 14, y + 34), (x + CARD_W - 14, y + 34), COL_RULE, 1)

    _draw_figure(frame, pose, x + 20, y + 44, CARD_W - 40, 150,
                 COL_OK if done else COL_CARD_BONE)

    # Confirmation timer: while the position is right, a bar fills and only
    # when it is full does the next step begin.  Without it the practitioner
    # cannot tell whether the app has noticed them get it right, and the step
    # appears to change at random.
    by = y + CARD_H - 16
    if confirming:
        _bar(frame, x + 14, by, CARD_W - 28, 4, confirm_fraction, COL_OK)
    elif step_seconds > 0.0:
        _bar(frame, x + 14, by, CARD_W - 28, 4, 0.0, COL_DIM)

    with Pen(frame, x, y, x + CARD_W, y + CARD_H) as pen:
        _label(pen, (x + 14, y + 23), "do this")
        if step_total:
            tag = "done" if done else f"step {step_no}/{step_total}"
            pen.text((x + CARD_W - 14, y + 23), tag, 11, COL_OK if done else COL_DIM, MONO, False, "r")
        pen.text((x + 14, y + 220), Pen.clip(title, CARD_W - 28, 16, "sans", True), 16,
                 COL_OK if done else COL_TEXT, "sans", True)
        ty = y + 240
        for line in Pen.wrap(instruction, CARD_W - 28, 12)[:2]:
            pen.text((x + 14, ty), line, 12, COL_DIM)
            ty += 16
        if confirming:
            pen.text((x + 14, by - 8), "holding position...", 11, COL_OK, MONO)
        elif step_seconds > 0.0:
            pen.text((x + 14, by - 8), f"on this step: {step_seconds:4.1f}s", 11, COL_DIM, MONO)


def draw_pose_guide(frame: np.ndarray, photo: np.ndarray | None, name: str,
                    sanskrit: str, steps: tuple[str, ...], current: int | None = None,
                    done: bool = False, demo_img: np.ndarray | None = None) -> None:
    """Bottom-right card for poses without a hand-built figure: photo + steps.

    Only Vrikshasana has the drawn reference figure and the walk-in script.  The
    other poses used to show Tree's figure, which told people to make the wrong
    shape; this shows the actual pose and how to get into it.
    """
    h, w = frame.shape[:2]
    ch = GUIDE_CARD_H
    x, y = w - CARD_W - 16, h - ch - 16
    _panel(frame, x, y, CARD_W, ch, alpha=0.90)
    top = y + 34
    if demo_img is not None and demo_img.size:
        dh, dw = demo_img.shape[:2]
        x0 = x + (CARD_W - dw) // 2
        frame[top:top + dh, x0:x0 + dw] = demo_img
        cv2.rectangle(frame, (x0 - 1, top - 1), (x0 + dw, top + dh), COL_RULE2, 1)
        top += dh + 10
    elif photo is not None and photo.size:
        ph = 112
        scale = ph / photo.shape[0]
        pw = max(1, int(photo.shape[1] * scale))
        if pw > CARD_W - 28:
            scale = (CARD_W - 28) / photo.shape[1]
            pw, ph = CARD_W - 28, max(1, int(photo.shape[0] * scale))
        thumb = cv2.resize(photo, (pw, ph), interpolation=cv2.INTER_AREA)
        px0 = x + (CARD_W - pw) // 2
        frame[top:top + ph, px0:px0 + pw] = thumb
        cv2.rectangle(frame, (px0 - 1, top - 1), (px0 + pw, top + ph), COL_RULE2, 1)
        top += ph + 10
    cv2.line(frame, (x + 14, y + 30), (x + CARD_W - 14, y + 30), COL_RULE, 1)

    with Pen(frame, x, y, x + CARD_W, y + CARD_H) as pen:
        _label(pen, (x + 14, y + 21), "make this shape")
        if current is not None:
            tag = "all steps done" if done else f"step {min(current + 1, len(steps))}/{len(steps)}"
            pen.text((x + CARD_W - 14, y + 21), tag, 11, COL_OK if done else COL_WARN, MONO, False, "r")
        pen.text((x + 14, top + 12), Pen.clip(name, CARD_W - 28, 14, "sans", True), 14, COL_TEXT, "sans", True)
        ly = top + 32
        for i, step in enumerate(steps[:6], 1):
            # finished steps are mint, the one to do now is bright, later ones dim
            if current is None:
                colour = COL_DIM
            elif done or i - 1 < current:
                colour = COL_OK
            elif i - 1 == current:
                colour = COL_TEXT
            else:
                colour = COL_DIM
            for j, line in enumerate(Pen.wrap(step, CARD_W - 52, 12)[:2]):
                if ly > y + ch - 6:
                    return
                if j == 0:
                    pen.text((x + 14, ly), f"{i}", 11, colour, MONO, True)
                pen.text((x + 30, ly), line, 12, colour)
                ly += 15
            ly += 4


def draw_step_panel(frame: np.ndarray, journey, current: int) -> None:
    """Right-hand strip: where you are in the practice.

    A single highlighted card tells you what to do *now* but never how much is
    left.  The list answers "which step, and how many" without the practitioner
    having to ask.  Tree has twelve stages, which would run into the card below,
    so at most six rows are shown: a window that slides to keep the current stage
    in view, with the full count in the header.
    """
    h, w = frame.shape[:2]
    total = len(journey)
    shown = min(total, MAX_STEP_ROWS)
    first = max(0, min(current - 2, total - shown))
    row_h = 20
    ph = 36 + row_h * shown
    x, y = w - CARD_W - 16, 44 + FRAMING_H + 10
    _panel(frame, x, y, CARD_W, ph, alpha=0.88)
    cv2.line(frame, (x + 14, y + 28), (x + CARD_W - 14, y + 28), COL_RULE, 1)

    ry = y + 48
    for i in range(first, first + shown):
        if i == current:
            cv2.rectangle(frame, (x + 1, ry - 14), (x + CARD_W - 2, ry + 5), COL_RAISED, -1)
            cv2.rectangle(frame, (x + 1, ry - 14), (x + 3, ry + 5), COL_ACCENT, -1)
        if i < current:
            cv2.rectangle(frame, (x + 14, ry - 7), (x + 20, ry - 1), COL_OK, -1)
        elif i == current:
            cv2.rectangle(frame, (x + 14, ry - 7), (x + 20, ry - 1), COL_ACCENT, -1)
        else:
            cv2.rectangle(frame, (x + 14, ry - 7), (x + 20, ry - 1), COL_FAINT, 1)
        ry += row_h

    with Pen(frame, x, y, x + CARD_W, y + ph) as pen:
        _label(pen, (x + 14, y + 20), f"step {min(current + 1, total)} of {total}", COL_TEXT)
        ry = y + 48
        for i in range(first, first + shown):
            step = journey[i]
            col = COL_OK if i < current else (COL_TEXT if i == current else COL_DIM)
            pen.text((x + 32, ry), Pen.clip(step.title, CARD_W - 48, 12, "sans", i == current), 12,
                     col, "sans", i == current)
            ry += row_h


def draw_rest_panel(frame: np.ndarray, remaining: float, target: float) -> None:
    """Bottom-right, replacing the reference card during the timed rest
    between one completed hold and the next attempt.

    A spoken countdown alone is easy to miss over a breath or a shuffled
    step; a large, continuously-visible number is the timer a practitioner
    can actually check without breaking their attention.
    """
    h, w = frame.shape[:2]
    x, y = w - CARD_W - 16, h - CARD_H - 16
    _panel(frame, x, y, CARD_W, CARD_H, alpha=0.90, border=None)
    cv2.rectangle(frame, (x, y), (x + CARD_W - 1, y + CARD_H - 1), COL_WARN, 1)
    cv2.line(frame, (x + 14, y + 32), (x + CARD_W - 14, y + 32), COL_RULE, 1)

    frac = 1.0 - max(0.0, min(1.0, remaining / max(1e-6, target)))
    _bar(frame, x + 16, y + 210, CARD_W - 32, 4, frac, COL_WARN)

    secs = max(0, int(remaining + 0.999))
    with Pen(frame, x, y, x + CARD_W, y + CARD_H) as pen:
        _label(pen, (x + 14, y + 22), "rest")
        pen.text((x + CARD_W - 14, y + 22), "next attempt in", 11, COL_DIM, MONO, False, "r")
        pen.text((x + CARD_W // 2, y + 160), str(secs), 96, COL_WARN, "sans", True, "m")
        pen.text((x + 16, y + 248), "breathe, and shake out the leg", 12, COL_DIM)
        pen.text((x + 16, y + 268), "jump back in early any time", 11, COL_FAINT, MONO)


def draw_angle_report(frame: np.ndarray, rows: list[dict], avg: float, line: str = "") -> None:
    """Bottom-right, replacing the pose card once the hold is over: how each part of the body did.

    ``rows`` are the per-part verdicts the sequence built from the joints it measured while the pose
    was held (yoga/flow.py): good, nearly there, or fix this, with the one correction that mattered.
    """
    h, w = frame.shape[:2]
    x, y = w - CARD_W - 16, h - CARD_H - 16
    _panel(frame, x, y, CARD_W, CARD_H, alpha=0.90)
    cv2.line(frame, (x + 14, y + 30), (x + CARD_W - 14, y + 30), COL_RULE, 1)
    colour = {"good": COL_OK, "nearly": COL_WARN, "fix": COL_BAD}
    word = {"good": "good", "nearly": "nearly there", "fix": "fix this"}
    with Pen(frame, x, y, x + CARD_W, y + CARD_H) as pen:
        _label(pen, (x + 14, y + 21), "angle check")
        if avg:
            pen.text((x + CARD_W - 14, y + 21), f"{avg:.0f}% overall", 11, COL_OK if avg >= 70 else COL_WARN,
                     MONO, False, "r")
        ry = y + 54
        for r in rows[:4]:
            pen.text((x + 14, ry), r["group"], 14, COL_TEXT, "sans", True)
            pen.text((x + CARD_W - 14, ry), word[r["verdict"]], 12, colour[r["verdict"]], MONO, True, "r")
            ry += 19
            if r["cue"]:
                for ln in Pen.wrap(r["cue"], CARD_W - 28, 12)[:2]:
                    pen.text((x + 14, ry), ln, 12, COL_DIM)
                    ry += 15
            ry += 10
        if line:
            cv2.line(frame, (x + 14, y + CARD_H - 74), (x + CARD_W - 14, y + CARD_H - 74), COL_RULE, 1)
            for i, ln in enumerate(Pen.wrap(line, CARD_W - 28, 12)[:3]):
                pen.text((x + 14, y + CARD_H - 56 + i * 16), ln, 12, COL_TEXT)


FRAMING_ROWS = (("head", "Head in shot"), ("torso", "Body in shot"),
                ("feet", "Feet in shot"), ("facing", "Facing camera"))


def draw_framing_panel(frame: np.ndarray, framing: dict | None) -> None:
    """Top-right panel: can the camera actually see you?

    Every downstream number depends on this, so it is shown plainly rather than
    left for the practitioner to infer from a score that will not rise.  The
    header carries the one thing to fix next, because a person balancing on one
    leg will read a heading and nothing else.
    """
    h, w = frame.shape[:2]
    pw, ph = CARD_W, FRAMING_H
    x, y = w - pw - 16, 44

    advice = (framing or {}).get("advice", "")
    ready = bool(framing and framing.get("ok")) and not advice
    colour = COL_OK if ready else COL_WARN
    _panel(frame, x, y, pw, ph, alpha=0.88, border=None)
    cv2.rectangle(frame, (x, y), (x + pw - 1, y + ph - 1), colour, 1)

    ry = y + 68
    for i, (key, _label_text) in enumerate(FRAMING_ROWS):
        good = bool(framing and framing.get(key))
        cv2.rectangle(frame, (x + pw - 22, ry + i * 20 - 8), (x + pw - 16, ry + i * 20 - 2),
                      COL_OK if good else COL_BAD, -1)
    # Distance: how much of the frame height the body fills, with the
    # acceptable band marked on the bar.
    fill = float(framing.get("fill", 0.0)) if framing else 0.0
    dist_ok = bool(framing and framing.get("distance_ok"))
    bx, by, bw = x + 14, y + ph - 22, pw - 28
    _bar(frame, bx, by, bw, 4, min(1.0, fill), COL_OK if dist_ok else COL_WARN)
    for frac in (FRAME_FILL_MIN, FRAME_FILL_MAX):
        mx = bx + int(bw * frac)
        cv2.line(frame, (mx, by - 4), (mx, by + 8), COL_ACCENT, 1)

    with Pen(frame, x, y, x + pw, y + ph) as pen:
        if ready or not advice:
            pen.text((x + 14, y + 26), "IN FRAME" if ready else "NOT READY", 13, colour, MONO, True,
                     track=1.2)
        else:                                  # a sentence: sentence case, up to two lines
            for j, line in enumerate(Pen.wrap(advice, pw - 28, 13, "sans", True)[:2]):
                pen.text((x + 14, y + 24 + j * 17), line, 13, colour, "sans", True)
        for i, (key, label) in enumerate(FRAMING_ROWS):
            good = bool(framing and framing.get(key))
            pen.text((x + 14, ry + i * 20), label, 12, COL_TEXT if good else COL_BAD)
        pen.text((x + 14, by - 10), "distance", 11, COL_DIM, MONO)
        pen.text((x + pw - 14, by - 10),
                 "too near" if fill > FRAME_FILL_MAX else ("too far" if fill < FRAME_FILL_MIN else "good"),
                 11, COL_OK if dist_ok else COL_WARN, MONO, False, "r")


def draw_calibration(frame: np.ndarray, title: str, sub: str, colour=COL_WARN) -> None:
    """The one-line prompt on the calibration view (countdown, then recording)."""
    h, w = frame.shape[:2]
    tw = Pen.width(title, 26, "sans", True)
    bh = 76 if sub else 52
    _panel(frame, 24, 24, int(tw) + 48, bh, alpha=0.88, border=None)
    cv2.rectangle(frame, (24, 24), (24 + int(tw) + 47, 24 + bh - 1), colour, 1)
    with Pen(frame, 0, 0, w, 120) as pen:
        pen.text((48, 60), title, 26, colour, "sans", True)
        if sub:
            pen.text((48, 86), sub, 12, COL_DIM, MONO)


# =============================================================================
# Demo walkthrough: what the whole practice looks like, before you start.
# =============================================================================

def draw_intro_frame(frame: np.ndarray, asana, journey, index: int,
                     pose: Pose | None, elapsed: float, total_s: float) -> None:
    """One frame of the pre-session walkthrough.

    Someone standing in front of a camera for the first time has no idea what
    is about to be asked of them.  Flicking through the whole sequence once,
    with the shape for each stage, means the live session starts with the
    practitioner already knowing where it is going - and it doubles as the
    thing to show an audience who will never install this.
    """
    h, w = frame.shape[:2]
    frame[:] = COL_PANEL
    step = journey[index]
    total = len(journey)

    cv2.line(frame, (44, 100), (w - 44, 100), COL_RULE, 1)
    rule(frame, 44, 62, 56, 2)

    # The whole practice as a row of numbered chips, so the shape of the
    # session is visible from the very first slide.
    chip_w, chip_h, gap = min(160, (w - 88 - (total - 1) * 8) // max(1, total)), 34, 8
    total_w = total * chip_w + (total - 1) * gap
    cx = (w - total_w) // 2
    for i in range(total):
        x0 = cx + i * (chip_w + gap)
        if i < index:
            border = COL_OK
        elif i == index:
            border = COL_ACCENT
            cv2.rectangle(frame, (x0, 118), (x0 + chip_w, 118 + chip_h), COL_RAISED, -1)
        else:
            border = COL_RULE
        cv2.rectangle(frame, (x0, 118), (x0 + chip_w, 118 + chip_h), border, 1)

    # Figure on the left, instruction on the right.
    if pose is not None:
        _draw_figure(frame, pose, 96, 190, 300, h - 320, COL_CARD_BONE)

    overall = (index + min(1.0, elapsed / max(1e-6, total_s))) / total
    _bar(frame, 44, h - 78, w - 88, 3, overall, COL_ACCENT)

    with Pen(frame) as pen:
        pen.text((44, 46), "AI YOGA COMPANION", 14, COL_TEXT, MONO, True, track=2.4)
        pen.text((44, 86), f"{asana.sanskrit}  |  {asana.name}", 13, COL_DIM, MONO)
        pen.text((w - 44, 46), "WALKTHROUGH", 12, COL_WARN, MONO, True, "r", 2.0)
        for i in range(total):
            x0 = cx + i * (chip_w + gap)
            tcol = COL_OK if i < index else (COL_TEXT if i == index else COL_FAINT)
            pen.text((x0 + chip_w // 2, 141), f"{i + 1}", 13, tcol, MONO, True, "m")
        tx = 470
        _label(pen, (tx, 222), f"step {index + 1} of {total}", COL_WARN, 12)
        pen.text((tx, 272), Pen.clip(step.title, w - tx - 70, 36, "sans", True), 36, COL_TEXT, "sans", True)
        ty = 322
        for line in Pen.wrap(step.say, w - tx - 70, 19)[:3]:
            pen.text((tx, ty), line, 19, COL_DIM)
            ty += 30
        if index + 1 < total:
            pen.text((tx, ty + 24), f"next  {journey[index + 1].title}", 13, COL_FAINT, MONO)
        pen.text((44, h - 44), "press any key to skip and start", 12, COL_DIM, MONO)


# =============================================================================
# Controls on the camera view: toolbar, pose menu, demo panel.
# Each returns the clickable rectangles it drew as (id, x0, y0, x1, y1), in the
# same 1280x720 coordinates as everything else, so the caller can hit-test a click.
# =============================================================================

DEMO_BOX_H = 112                  #: height of the little animated demo on the pose card


def hit(buttons, x: float, y: float):
    """The id of the button under (x, y), or None."""
    for bid, x0, y0, x1, y1 in reversed(buttons):
        if x0 <= x <= x1 and y0 <= y <= y1:
            return bid
    return None


def _btn_border(bid, hover, on: bool = False, kind: str = "plain"):
    if bid == hover:
        return COL_ACCENT
    if on:
        return COL_OK if kind == "voice" else COL_ACCENT
    return COL_RULE2


def draw_toolbar(frame: np.ndarray, pose_label: str, can_switch: bool, voice_on: bool,
                 demo_on: bool, hover=None) -> list:
    """The strip across the top of the picture: pick the pose, voice on/off, demo."""
    h, w = frame.shape[:2]
    y0, y1 = 10, 42
    x = PANEL_W + 14
    right = w - CARD_W - 30
    gap = 8
    items = []
    if can_switch:
        items += [("prev", "<", 34, "icon"), ("menu", pose_label, 230, "pose"), ("next", ">", 34, "icon")]
    else:
        items += [("pose_locked", pose_label, 230, "pose")]
    items += [("voice", "VOICE ON" if voice_on else "VOICE OFF", 106, "voice"),
              ("demo", "DEMO", 70, "demo"), ("again", "SAY AGAIN", 100, "plain")]
    total = sum(it[2] for it in items) + gap * (len(items) - 1)
    if x + total > right:                                  # a narrow picture: shrink the pose name
        items = [(a, b, c - (x + total - right) if k == "pose" else c, k) for a, b, c, k in items]
    buttons, drawn = [], []
    for bid, label, bw, kind in items:
        x1 = x + bw
        buttons.append((bid, x, y0, x1, y1))
        drawn.append((bid, label, kind, x, x1))
        x = x1 + gap
    for bid, label, kind, bx0, bx1 in drawn:
        _panel(frame, bx0, y0, bx1 - bx0, y1 - y0, alpha=0.84, border=None)
        on = (bid == "voice" and voice_on) or (bid == "demo" and demo_on)
        cv2.rectangle(frame, (bx0, y0), (bx1 - 1, y1 - 1), _btn_border(bid, hover, on, kind), 1)
        if bid == "menu":                                  # a small down-triangle: this opens a list
            cx, cy = bx1 - 16, (y0 + y1) // 2
            cv2.fillPoly(frame, [np.array([(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 4)], np.int32)], COL_ACCENT)
    with Pen(frame, 0, 0, w, y1 + 4) as pen:
        for bid, label, kind, bx0, bx1 in drawn:
            mid = (bx0 + bx1) // 2
            if kind == "pose":
                pen.text((bx0 + 12, 31), Pen.clip(label, bx1 - bx0 - 40, 14, "sans", True), 14, COL_TEXT,
                         "sans", True)
            elif kind == "icon":
                pen.text((mid, 31), label, 16, COL_TEXT, MONO, True, "m")
            else:
                colour = COL_OK if (bid == "voice" and voice_on) else (
                    COL_DIM if (bid == "voice") else COL_TEXT)
                pen.text((mid, 30), label, 11, colour, MONO, True, "m", 1.2)
    return buttons


def draw_pose_menu(frame: np.ndarray, poses: list, current: str, hover=None) -> list:
    """Full-picture list of every pose; click one (or press its number) to switch.

    ``poses`` is ``[(key, name, sanskrit, tag), ...]``."""
    h, w = frame.shape[:2]
    _panel(frame, PANEL_W, 0, w - PANEL_W, h, alpha=0.96, border=None)
    left, top = PANEL_W + 40, 108
    cols = 2
    cw = (w - PANEL_W - 80 - 20) // cols
    ch, gapy = 58, 12
    buttons = [("close_menu", w - 150, 20, w - 30, 52)]
    cells = []
    for i, (key, name, sans, tag) in enumerate(poses):
        r, c = i % ((len(poses) + 1) // 2), i // ((len(poses) + 1) // 2)
        x0 = left + c * (cw + 20)
        y0 = top + r * (ch + gapy)
        buttons.append((f"pose:{key}", x0, y0, x0 + cw, y0 + ch))
        cells.append((key, name, sans, tag, x0, y0, i))
    for key, name, sans, tag, x0, y0, i in cells:
        bid = f"pose:{key}"
        border = COL_OK if key == current else (COL_ACCENT if bid == hover else COL_RULE2)
        if bid == hover:
            cv2.rectangle(frame, (x0 + 1, y0 + 1), (x0 + cw - 2, y0 + ch - 2), COL_RAISED, -1)
        cv2.rectangle(frame, (x0, y0), (x0 + cw - 1, y0 + ch - 1), border, 1)
    cv2.rectangle(frame, (w - 150, 20), (w - 31, 51), COL_ACCENT if hover == "close_menu" else COL_RULE2, 1)
    with Pen(frame, 0, 0, w, h) as pen:
        pen.text((left, 62), "CHOOSE A POSE", 14, COL_TEXT, MONO, True, track=2.4)
        pen.text((left, 84), "Click one, or press its number.  The timer and the guide start afresh.",
                 12, COL_DIM)
        pen.text((w - 90, 41), "ESC  CLOSE", 11, COL_TEXT, MONO, True, "m", 1.2)
        for key, name, sans, tag, x0, y0, i in cells:
            n = (i + 1) % 10
            pen.text((x0 + 16, y0 + 36), f"{n}", 15, COL_DIM, MONO, True)
            pen.text((x0 + 44, y0 + 26), name, 16, COL_OK if key == current else COL_TEXT, "sans", True)
            pen.text((x0 + 44, y0 + 45), sans, 11, COL_DIM, MONO)
            pen.text((x0 + cw - 16, y0 + 36), tag.upper(), 10, COL_OK if tag == "full coach" else COL_DIM,
                     MONO, True, "r", 1.2)
    return buttons


def draw_demo_panel(frame: np.ndarray, clip, t: float, caption: str, title: str, hover=None) -> list:
    """A large looping demo of the pose over the picture, between the side panels."""
    h, w = frame.shape[:2]
    x0, x1 = PANEL_W + 16, w - CARD_W - 32
    y0, y1 = 56, h - 120
    bw, bh = x1 - x0, y1 - y0
    img = clip.render(bw - 2, bh - 2, t, caption=caption, title=title)
    frame[y0 + 1:y1 - 1, x0 + 1:x1 - 1] = img
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), COL_RULE2, 1)
    bx0, bx1, by0, by1 = x1 - 118, x1 - 10, y0 + 34, y0 + 62
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), COL_ACCENT if hover == "close_demo" else COL_RULE2, 1)
    with Pen(frame, 0, 0, w, h) as pen:
        pen.text(((bx0 + bx1) // 2, by0 + 20), "D  CLOSE", 11, COL_TEXT, MONO, True, "m", 1.2)
    return [("close_demo", bx0, by0, bx1, by1)]
