"""On-screen feedback (Methodology Stage 2 'live skeleton overlay').

Draws the skeleton, colours the limb that is actually wrong, and shows the
joint-by-joint table so a panel member can see *why* the score is what it is
instead of trusting a single number.  Text is deliberately ASCII-only because
OpenCV's Hershey fonts cannot render anything else.
"""

from __future__ import annotations

import cv2
import numpy as np

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


def _segments_for(key: str, pose: Pose, feats: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Which bones a failing check refers to, so the right limb turns red."""
    p = pose.pts
    standing = feats.get("standing_side", "left")
    folded = feats.get("folded_side", "right")
    s_hip, s_knee, s_ankle = ((L_HIP, L_KNEE, L_ANKLE) if standing == "left"
                              else (R_HIP, R_KNEE, R_ANKLE))
    d_hip, d_knee, d_ankle = ((L_HIP, L_KNEE, L_ANKLE) if folded == "left"
                              else (R_HIP, R_KNEE, R_ANKLE))
    mid_sh = (p[L_SHOULDER] + p[R_SHOULDER]) * 0.5
    mid_hip = (p[L_HIP] + p[R_HIP]) * 0.5

    table = {
        "spine_tilt": [(mid_hip, mid_sh)],
        "hip_level": [(p[L_HIP], p[R_HIP])],
        "shoulder_level": [(p[L_SHOULDER], p[R_SHOULDER])],
        "standing_knee": [(p[s_hip], p[s_knee]), (p[s_knee], p[s_ankle])],
        "folded_knee": [(p[d_hip], p[d_knee]), (p[d_knee], p[d_ankle])],
        "folded_thigh_open": [(p[d_hip], p[d_knee])],
        "foot_height_ratio": [(p[d_knee], p[d_ankle])],
        "arm_raise_left": [(p[L_SHOULDER], p[L_ELBOW]), (p[L_ELBOW], p[L_WRIST])],
        "arm_raise_right": [(p[R_SHOULDER], p[R_ELBOW]), (p[R_ELBOW], p[R_WRIST])],
        "elbow_left": [(p[L_SHOULDER], p[L_ELBOW]), (p[L_ELBOW], p[L_WRIST])],
        "elbow_right": [(p[R_SHOULDER], p[R_ELBOW]), (p[R_ELBOW], p[R_WRIST])],
    }
    return table.get(key, [])


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

    # The single worst fault is drawn red; the rest of the faults amber.
    for rank, res in enumerate(ev.failures[:3]):
        colour = COL_BAD if rank == 0 else COL_WARN
        for a, b in _segments_for(res.check.key, pose, ev.features):
            cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), colour, 5, cv2.LINE_AA)


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
             right_margin: int = 0) -> None:
    """`right_margin` keeps the bottom banner clear of the reference card."""
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
        cv2.putText(frame, "MEAS (dev)", (188, y), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
        y += 8
        cv2.line(frame, (12, y), (PANEL_W - 12, y), (70, 70, 70), 1)
        y += 18
        for res in ev.results:
            if res.check.weight <= 0.0:
                continue
            if res.ok is None:
                mark, col = "-", COL_DIM
            elif res.ok:
                mark, col = "OK", COL_OK
            else:
                mark, col = "X", COL_BAD
            cv2.putText(frame, res.label[:22], (16, y), FONT, 0.42, col, 1, cv2.LINE_AA)
            cv2.putText(frame, res.display(), (188, y), FONT, 0.42, col, 1, cv2.LINE_AA)
            cv2.putText(frame, mark, (PANEL_W - 34, y), FONT, 0.42, col, 1, cv2.LINE_AA)
            y += 19
        y += 6
        cv2.putText(frame, f"standing leg: {ev.features.get('standing_side','?')}",
                    (16, y), FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
        y += 16
        if ev.variant_name:
            cv2.putText(frame, f"form: {ev.variant_name}", (16, y),
                        FONT, 0.42, COL_DIM, 1, cv2.LINE_AA)
            y += 16
        if not ev.features.get("group_visibility", {}).get("frontal", 1.0):
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

    cv2.putText(frame, f"{fps:4.1f} fps", (16, h - 14), FONT, 0.46, COL_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "q quit  v voice  r reset  s snapshot  g guide",
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
                        title: str, instruction: str, done: bool = False) -> None:
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

    cv2.putText(frame, title[:24], (x + 14, y + 232), FONT, 0.56,
                COL_OK if done else COL_TEXT, 2, cv2.LINE_AA)
    ty = y + 254
    for line in _wrap(instruction, 30)[:3]:
        cv2.putText(frame, line, (x + 14, ty), FONT, 0.40, COL_DIM, 1, cv2.LINE_AA)
        ty += 16


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
