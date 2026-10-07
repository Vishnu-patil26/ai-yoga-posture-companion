"""On-screen feedback (Methodology Stage 2 'live skeleton overlay').

Draws the skeleton, colours the limb that is actually wrong, and shows the
joint-by-joint table so a panel member can see *why* the score is what it is
instead of trusting a single number.  Text is deliberately ASCII-only because
OpenCV's Hershey fonts cannot render anything else.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import phrasing
from .evaluator import FRAME_FILL_MAX, FRAME_FILL_MIN, Evaluation
from .landmarks import (
    CONNECTIONS, L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, Pose,
)
from .state_machine import PoseStateMachine, State

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
COL_OK = (120, 220, 120)
COL_WARN = (60, 190, 250)
COL_BAD = (70, 70, 245)
COL_BONE = (215, 180, 110)
COL_RAW = (110, 110, 110)
COL_TEXT = (240, 240, 240)
COL_DIM = (170, 170, 170)
COL_PANEL = (28, 24, 20)

PANEL_W = 330


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
        cv2.circle(frame, tuple(p[i]), 4, (255, 255, 255), -1, cv2.LINE_AA)

    if ev is None:
        return

    # Every graded bone is coloured by its own verdict: green when correct,
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


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.72) -> None:
    sub = frame[y:y + h, x:x + w]
    if sub.size == 0:
        return
    box = np.full(sub.shape, COL_PANEL, dtype=np.uint8)
    cv2.addWeighted(box, alpha, sub, 1 - alpha, 0, sub)


def _bar(frame, x, y, w, h, frac, colour, bg=(60, 60, 60)) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(frame, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), colour, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (110, 110, 110), 1)


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if len(trial) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


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

    _panel(frame, 0, 0, PANEL_W, h)
    y = 30
    cv2.putText(frame, "AI YOGA COMPANION", (16, y), FONT, 0.62, COL_TEXT, 2, cv2.LINE_AA)
    y += 26
    if asana:
        cv2.putText(frame, f"{asana.sanskrit} - {asana.name}", (16, y), FONT, 0.5, COL_DIM, 1, cv2.LINE_AA)
        y += 18
        cv2.putText(frame, f"level {asana.level}   hold {asana.hold_target_s:.0f}s",
                    (16, y), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
    y += 24
    cv2.line(frame, (12, y), (PANEL_W - 12, y), (80, 80, 80), 1)
    y += 30

    score = ev.score if (ev and ev.usable) else 0.0
    colour = _score_colour(score)
    cv2.putText(frame, f"{score:5.1f}%", (16, y + 8), FONT, 1.1, colour, 2, cv2.LINE_AA)
    cv2.putText(frame, "alignment", (168, y - 2), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
    if ev:
        cv2.putText(frame, f"{sm.state.value}", (168, y + 16), FONT, 0.46, colour, 1, cv2.LINE_AA)
    y += 22
    _bar(frame, 16, y, PANEL_W - 32, 9, score / 100.0, colour)
    y += 34

    cv2.putText(frame, f"HOLD  {sm.elapsed:5.1f}s / {sm.hold_target_s:.0f}s",
                (16, y), FONT, 0.5, COL_TEXT, 1, cv2.LINE_AA)
    y += 10
    _bar(frame, 16, y, PANEL_W - 32, 9, sm.progress, COL_OK if sm.progress >= 1 else (200, 170, 90))
    y += 30

    if ev:
        cv2.putText(frame, "JOINT", (16, y), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, "MEAS (dev)" if metric else "HOW FAR OFF",
                    (188, y), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
        y += 8
        cv2.line(frame, (12, y), (PANEL_W - 12, y), (70, 70, 70), 1)
        y += 18
        for res in ev.results:
            if res.check.weight <= 0.0:
                continue
            if res.ok is None:
                mark, col = "?", COL_WARN            # matches the dashed-amber bone
            elif res.ok:
                mark, col = "OK", COL_OK
            else:
                mark, col = "X", COL_BAD
            words = res.display() if metric else phrasing.short_words(res, ev.features)
            cv2.putText(frame, res.label[:22], (16, y), FONT, 0.42, col, 1, cv2.LINE_AA)
            cv2.putText(frame, words[:20], (188, y), FONT, 0.42, col, 1, cv2.LINE_AA)
            cv2.putText(frame, mark, (PANEL_W - 34, y), FONT, 0.42, col, 1, cv2.LINE_AA)
            y += 19
        y += 6
        if ev.asana.key == "vrikshasana":                  # only Tree has a standing leg
            cv2.putText(frame, f"standing leg: {ev.features.get('standing_side','?')}",
                        (16, y), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
            y += 16
        if ev.variant_name:
            cv2.putText(frame, f"form: {ev.variant_name}", (16, y),
                        FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
            y += 16
        if (ev.asana.key == "vrikshasana"
                and not ev.features.get("group_visibility", {}).get("frontal", 1.0)):
            cv2.putText(frame, "turn to face the camera", (16, y),
                        FONT, 0.42, COL_WARN, 1, cv2.LINE_AA)
            y += 16
        y += 4

    st = sm.stats
    cv2.putText(frame, f"holds completed: {st.completed}", (16, y), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
    y += 17
    cv2.putText(frame, f"longest hold:    {st.longest_s:.1f}s", (16, y), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
    y += 17
    cv2.putText(frame, f"best alignment:  {st.best_score:.1f}%", (16, y), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)

    if ev:
        cv2.putText(frame, "green = correct", (16, h - 68), FONT, 0.38, COL_OK, 1, cv2.LINE_AA)
        cv2.putText(frame, "red = fix this", (16, h - 52), FONT, 0.38, COL_BAD, 1, cv2.LINE_AA)
        cv2.putText(frame, "dashed = can't see", (16, h - 36), FONT, 0.38, COL_WARN, 1, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:4.1f} fps", (16, h - 14), FONT, 0.46, COL_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "q/Esc to quit   v voice  g guide  m metric  r reset  s snapshot",
                (110, h - 14), FONT, 0.40, COL_DIM, 1, cv2.LINE_AA)

    banner = cue_text or hint
    if banner:
        bw = max(120, w - PANEL_W - right_margin)
        # Wrap rather than truncate - a cue cut off halfway is worse than none.
        lines = _wrap(banner, max(12, int(bw / 13)))[:2]
        bh = 26 + 30 * len(lines)
        _panel(frame, PANEL_W, h - bh, bw, bh, alpha=0.82)
        col = COL_OK if sm.state is State.COMPLETE else COL_TEXT
        ly = h - bh + 32
        for line in lines:
            cv2.putText(frame, line, (PANEL_W + 20, ly), FONT, 0.60, col, 2, cv2.LINE_AA)
            ly += 30

    if sm.state is State.COMPLETE:
        cv2.putText(frame, "HOLD COMPLETE", (PANEL_W + 30, 34), FONT, 0.9, COL_OK, 2, cv2.LINE_AA)


# =============================================================================
# Guidance panels: what to do next, and whether the camera can see you.
# =============================================================================

CARD_W, CARD_H = 250, 300
COL_CARD_BONE = (200, 165, 105)


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
        cv2.circle(frame, tuple(q[i]), 3, (255, 255, 255), -1, cv2.LINE_AA)


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
    _panel(frame, x, y, CARD_W, CARD_H, alpha=0.80)
    cv2.rectangle(frame, (x, y), (x + CARD_W, y + CARD_H), (90, 84, 78), 1)

    cv2.putText(frame, "DO THIS", (x + 14, y + 24), FONT, 0.46, COL_DIM, 1, cv2.LINE_AA)
    if step_total:
        tag = "done" if done else f"step {step_no}/{step_total}"
        cv2.putText(frame, tag, (x + CARD_W - 78, y + 24), FONT, 0.42,
                    COL_OK if done else COL_DIM, 1, cv2.LINE_AA)
    cv2.line(frame, (x + 12, y + 32), (x + CARD_W - 12, y + 32), (80, 76, 70), 1)

    _draw_figure(frame, pose, x + 20, y + 40, CARD_W - 40, 170,
                 COL_OK if done else COL_CARD_BONE)

    cv2.putText(frame, title[:24], (x + 14, y + 226), FONT, 0.56,
                COL_OK if done else COL_TEXT, 2, cv2.LINE_AA)
    ty = y + 248
    for line in _wrap(instruction, 30)[:2]:
        cv2.putText(frame, line, (x + 14, ty), FONT, 0.40, COL_DIM, 1, cv2.LINE_AA)
        ty += 15

    # Confirmation timer: while the position is right, a bar fills and only
    # when it is full does the next step begin.  Without it the practitioner
    # cannot tell whether the app has noticed them get it right, and the step
    # appears to change at random.
    by = y + CARD_H - 30
    if confirming:
        cv2.putText(frame, "holding position...", (x + 14, by - 8), FONT, 0.40,
                    COL_OK, 1, cv2.LINE_AA)
        _bar(frame, x + 14, by, CARD_W - 28, 9, confirm_fraction, COL_OK)
    elif step_seconds > 0.0:
        cv2.putText(frame, f"on this step: {step_seconds:4.1f}s", (x + 14, by - 8),
                    FONT, 0.40, COL_DIM, 1, cv2.LINE_AA)
        _bar(frame, x + 14, by, CARD_W - 28, 9, 0.0, COL_DIM)


def draw_pose_guide(frame: np.ndarray, photo: np.ndarray | None, name: str,
                    sanskrit: str, steps: tuple[str, ...], current: int | None = None,
                    done: bool = False) -> None:
    """Bottom-right card for poses without a hand-built figure: photo + steps.

    Only Vrikshasana has the drawn reference figure and the walk-in script.  The
    other poses used to show Tree's figure, which told people to make the wrong
    shape; this shows the actual pose and how to get into it.
    """
    h, w = frame.shape[:2]
    x, y = w - CARD_W - 16, h - CARD_H - 16
    _panel(frame, x, y, CARD_W, CARD_H, alpha=0.84)
    cv2.rectangle(frame, (x, y), (x + CARD_W, y + CARD_H), (90, 84, 78), 1)
    cv2.putText(frame, "MAKE THIS SHAPE", (x + 14, y + 22), FONT, 0.46, COL_DIM, 1, cv2.LINE_AA)
    if current is not None:
        tag = "all steps done" if done else f"step {min(current + 1, len(steps))}/{len(steps)}"
        cv2.putText(frame, tag, (x + CARD_W - 112, y + 22), FONT, 0.42,
                    COL_OK if done else COL_WARN, 1, cv2.LINE_AA)
    top = y + 32
    if photo is not None and photo.size:
        ph = 118
        scale = ph / photo.shape[0]
        pw = max(1, int(photo.shape[1] * scale))
        if pw > CARD_W - 28:
            scale = (CARD_W - 28) / photo.shape[1]
            pw, ph = CARD_W - 28, max(1, int(photo.shape[0] * scale))
        thumb = cv2.resize(photo, (pw, ph), interpolation=cv2.INTER_AREA)
        frame[top:top + ph, x + (CARD_W - pw) // 2:x + (CARD_W - pw) // 2 + pw] = thumb
        top += ph + 8
    cv2.putText(frame, f"{name} ({sanskrit})"[:34], (x + 14, top + 12), FONT, 0.46, COL_TEXT, 1, cv2.LINE_AA)
    ly = top + 32
    for i, step in enumerate(steps[:3], 1):
        # finished steps are green, the one to do now is bright, later ones dim
        if current is None:
            colour = COL_DIM
        elif done or i - 1 < current:
            colour = COL_OK
        elif i - 1 == current:
            colour = COL_TEXT
        else:
            colour = COL_DIM
        for j, line in enumerate(_wrap(step, 33)[:3]):
            if ly > y + CARD_H - 6:
                return
            cv2.putText(frame, (f"{i}. " if j == 0 else "    ") + line, (x + 14, ly),
                        FONT, 0.38, colour, 1, cv2.LINE_AA)
            ly += 15
        ly += 3


def draw_step_panel(frame: np.ndarray, journey, current: int) -> None:
    """Right-hand strip: every stage of the practice, and which one you are in.

    A single highlighted card tells you what to do *now* but never how much is
    left.  Showing the whole list with a position in it answers "which step,
    and how many" without the practitioner having to ask.
    """
    h, w = frame.shape[:2]
    total = len(journey)
    row_h = 19
    ph = 30 + row_h * total
    x, y = w - CARD_W - 16, 44 + 168 + 10
    _panel(frame, x, y, CARD_W, ph, alpha=0.80)
    cv2.rectangle(frame, (x, y), (x + CARD_W, y + ph), (90, 84, 78), 1)

    cv2.putText(frame, f"STEP {min(current + 1, total)} OF {total}", (x + 14, y + 20),
                FONT, 0.44, COL_TEXT, 1, cv2.LINE_AA)
    cv2.line(frame, (x + 12, y + 27), (x + CARD_W - 12, y + 27), (80, 76, 70), 1)

    ry = y + 44
    for i, step in enumerate(journey):
        if i < current:
            mark, col = "OK", COL_OK              # already done this attempt
        elif i == current:
            mark, col = ">", COL_WARN             # where you are now
        else:
            mark, col = "", COL_DIM               # still to come
        if i == current:
            cv2.rectangle(frame, (x + 8, ry - 12), (x + CARD_W - 8, ry + 5),
                          (58, 52, 46), -1)
        cv2.putText(frame, mark, (x + 14, ry), FONT, 0.38, col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{i + 1}. {step.title}"[:24], (x + 38, ry),
                    FONT, 0.38, col, 1, cv2.LINE_AA)
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
    _panel(frame, x, y, CARD_W, CARD_H, alpha=0.82)
    cv2.rectangle(frame, (x, y), (x + CARD_W, y + CARD_H), COL_WARN, 1)

    cv2.putText(frame, "REST", (x + 14, y + 24), FONT, 0.46, COL_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "next attempt in", (x + CARD_W - 120, y + 24), FONT, 0.38,
                COL_DIM, 1, cv2.LINE_AA)
    cv2.line(frame, (x + 12, y + 32), (x + CARD_W - 12, y + 32), (80, 76, 70), 1)

    secs = max(0, int(remaining + 0.999))
    text = str(secs)
    scale = 2.7
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 5)
    cv2.putText(frame, text, (x + (CARD_W - tw) // 2, y + 60 + th), FONT, scale,
                COL_WARN, 5, cv2.LINE_AA)

    frac = 1.0 - max(0.0, min(1.0, remaining / max(1e-6, target)))
    _bar(frame, x + 16, y + 210, CARD_W - 32, 10, frac, COL_WARN)

    cv2.putText(frame, "breathe, and shake out the leg", (x + 16, y + 246),
                FONT, 0.40, COL_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "jump back in early any time", (x + 16, y + 266),
                FONT, 0.36, COL_DIM, 1, cv2.LINE_AA)


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
    pw, ph = CARD_W, 168
    x, y = w - pw - 16, 44
    _panel(frame, x, y, pw, ph, alpha=0.82)

    advice = (framing or {}).get("advice", "")
    ready = bool(framing and framing.get("ok")) and not advice
    colour = COL_OK if ready else COL_WARN
    cv2.rectangle(frame, (x, y), (x + pw, y + ph), colour, 1)

    heading = "IN FRAME" if ready else (advice.upper() if advice else "NOT READY")
    scale = 0.56 if len(heading) <= 12 else 0.42
    cv2.putText(frame, heading[:30], (x + 14, y + 26), FONT, scale, colour, 2, cv2.LINE_AA)

    ry = y + 52
    for key, label in FRAMING_ROWS:
        good = bool(framing and framing.get(key))
        col = COL_OK if good else COL_BAD
        cv2.putText(frame, label, (x + 14, ry), FONT, 0.42, col, 1, cv2.LINE_AA)
        cv2.putText(frame, "OK" if good else "X", (x + pw - 34, ry),
                    FONT, 0.42, col, 1, cv2.LINE_AA)
        ry += 19

    # Distance: how much of the frame height the body fills, with the
    # acceptable band marked on the bar.
    fill = float(framing.get("fill", 0.0)) if framing else 0.0
    cv2.putText(frame, "distance", (x + 14, ry + 14), FONT, 0.40, COL_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "too near" if fill > FRAME_FILL_MAX else
                ("too far" if fill < FRAME_FILL_MIN else "good"),
                (x + pw - 74, ry + 14), FONT, 0.40,
                COL_OK if (framing and framing.get("distance_ok")) else COL_WARN,
                1, cv2.LINE_AA)
    bx, by, bw = x + 14, ry + 22, pw - 28
    _bar(frame, bx, by, bw, 9, min(1.0, fill),
         COL_OK if (framing and framing.get("distance_ok")) else COL_WARN)
    for frac in (FRAME_FILL_MIN, FRAME_FILL_MAX):
        mx = bx + int(bw * frac)
        cv2.line(frame, (mx, by - 3), (mx, by + 12), (235, 235, 235), 1)


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
    frame[:] = (30, 26, 22)
    step = journey[index]
    total = len(journey)

    # Header
    cv2.putText(frame, "AI YOGA COMPANION", (44, 52), FONT, 0.70, COL_TEXT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{asana.sanskrit} - {asana.name}", (44, 78), FONT, 0.5,
                COL_DIM, 1, cv2.LINE_AA)
    tag = "WALKTHROUGH"
    (tw, _), _ = cv2.getTextSize(tag, FONT, 0.46, 1)
    cv2.putText(frame, tag, (w - 44 - tw, 52), FONT, 0.46, COL_WARN, 1, cv2.LINE_AA)
    cv2.line(frame, (44, 96), (w - 44, 96), (72, 68, 62), 1)

    # The whole practice as a row of numbered chips, so the shape of the
    # session is visible from the very first slide.
    chip_w, chip_h, gap = 160, 34, 8
    total_w = total * chip_w + (total - 1) * gap
    cx = (w - total_w) // 2
    for i, other in enumerate(journey):
        x0 = cx + i * (chip_w + gap)
        if i < index:
            border, text_col = COL_OK, COL_OK
        elif i == index:
            border, text_col = COL_WARN, COL_TEXT
            cv2.rectangle(frame, (x0, 112), (x0 + chip_w, 112 + chip_h), (54, 48, 42), -1)
        else:
            border, text_col = (78, 74, 68), (128, 122, 116)
        cv2.rectangle(frame, (x0, 112), (x0 + chip_w, 112 + chip_h), border, 1)
        cv2.putText(frame, f"{i + 1}", (x0 + 9, 134), FONT, 0.44, border, 1, cv2.LINE_AA)
        cv2.putText(frame, other.title[:18], (x0 + 26, 134), FONT, 0.36,
                    text_col, 1, cv2.LINE_AA)

    # Figure on the left, instruction on the right.
    if pose is not None:
        _draw_figure(frame, pose, 96, 176, 300, h - 300, COL_CARD_BONE)

    tx = 470
    cv2.putText(frame, f"STEP {index + 1} OF {total}", (tx, 214), FONT, 0.5,
                COL_WARN, 1, cv2.LINE_AA)
    cv2.putText(frame, step.title[:26], (tx, 262), FONT, 1.05, COL_TEXT, 2, cv2.LINE_AA)
    ty = 312
    for line in _wrap(step.say, 40)[:3]:
        cv2.putText(frame, line, (tx, ty), FONT, 0.54, COL_DIM, 1, cv2.LINE_AA)
        ty += 32

    if index + 1 < total:
        cv2.putText(frame, f"next:  {journey[index + 1].title}", (tx, ty + 26),
                    FONT, 0.44, (128, 122, 116), 1, cv2.LINE_AA)

    # Progress through the whole walkthrough, not just this slide.
    overall = (index + min(1.0, elapsed / max(1e-6, total_s))) / total
    _bar(frame, 44, h - 84, w - 88, 6, overall, COL_WARN)
    cv2.putText(frame, "press any key to skip and start", (44, h - 50),
                FONT, 0.46, COL_DIM, 1, cv2.LINE_AA)
