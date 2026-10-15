"""Proper typography on OpenCV frames.

``cv2.putText`` only has the Hershey stroke fonts: one weight, no letter
spacing, ASCII only.  The black theme is monospaced capitals for labels and a
clean sans for sentences, so text on the live view is drawn with the TrueType
files the theme names.

Rendering text with Pillow on every frame costs about 0.2 ms a string, and a
screen has sixty of them.  Almost all of them repeat from frame to frame (the
labels, the joint names, the words), so each distinct string is rendered once
into a small greyscale mask and kept; drawing it afterwards is a few array
operations on a patch of the frame.  Only strings that really change (the score,
the clock) are rendered again.

Shapes (bars, hairlines, boxes) are drawn with OpenCV as before.  A ``Pen`` is a
context manager only to keep the call sites tidy; it can also clip text to a
rectangle.

If no TrueType font exists at all the pen falls back to ``cv2.putText`` so the
trainer still runs; it just looks plainer.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import theme

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:                                   # Pillow missing: Hershey fallback
    Image = ImageDraw = ImageFont = None

_FONTS: dict = {}
_WIDTHS: dict = {}
_MASKS: dict = {}
_MASK_LIMIT = 4000


def _font(kind: str, bold: bool, size: int):
    key = (kind, bold, size)
    if key not in _FONTS:
        f = None
        if ImageFont is not None:
            path = theme.font_file(kind, bold)
            try:
                f = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
            except Exception:
                f = None
        _FONTS[key] = f
    return _FONTS[key]


def _text_width(s: str, size: int, kind: str, bold: bool, track: float) -> float:
    key = (s, size, kind, bold, track)
    w = _WIDTHS.get(key)
    if w is None:
        f = _font(kind, bold, size)
        w = len(s) * size * 0.55 + track * len(s) if f is None else float(f.getlength(s)) + track * len(s)
        if len(_WIDTHS) > 20000:
            _WIDTHS.clear()
        _WIDTHS[key] = w
    return w


def _mask(s: str, size: int, kind: str, bold: bool, track: float):
    """The text as an 8-bit coverage mask plus where its top-left sits relative to
    the baseline origin.  Cached."""
    key = (s, size, kind, bold, track)
    hit = _MASKS.get(key)
    if hit is not None:
        return hit
    f = _font(kind, bold, size)
    ascent, descent = f.getmetrics()
    tw = _text_width(s, size, kind, bold, track)
    pad = 2
    w, h = int(math.ceil(tw)) + 2 * pad, ascent + descent + 2 * pad
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    ox, oy = float(pad), float(ascent + pad)
    if track:
        for ch in s:
            d.text((ox, oy), ch, font=f, fill=255, anchor="ls")
            ox += float(f.getlength(ch)) + track
    else:
        d.text((ox, oy), s, font=f, fill=255, anchor="ls")
    # coverage in 0..64 (6 bits) so the blend below fits in int16
    cov = ((np.asarray(img, dtype=np.uint16) * 64 + 127) // 255).astype(np.int16)
    out = (cov[..., None], -pad, -(ascent + pad))
    if len(_MASKS) >= _MASK_LIMIT:
        _MASKS.clear()
    _MASKS[key] = out
    return out


class Pen:
    """Draw text on a BGR frame, optionally clipped to a rectangle.  Use as a context manager."""

    def __init__(self, frame: np.ndarray, x0: int = 0, y0: int = 0,
                 x1: int | None = None, y1: int | None = None) -> None:
        h, w = frame.shape[:2]
        self.frame = frame
        self.x0, self.y0 = max(0, int(x0)), max(0, int(y0))
        self.x1 = min(w, int(x1) if x1 is not None else w)
        self.y1 = min(h, int(y1) if y1 is not None else h)
        self.ok = Image is not None and self.x1 > self.x0 and self.y1 > self.y0

    def __enter__(self) -> "Pen":
        return self

    def __exit__(self, *exc) -> None:
        return None

    # ---------------------------------------------------------------- measuring
    @staticmethod
    def width(s: str, size: int = 14, kind: str = "sans", bold: bool = False,
              track: float = 0.0) -> float:
        return _text_width(s, size, kind, bold, track)

    @staticmethod
    def wrap(text: str, max_px: float, size: int = 14, kind: str = "sans",
             bold: bool = False) -> list[str]:
        """Break ``text`` into lines no wider than ``max_px`` (never cut mid-word)."""
        lines, line = [], ""
        for word in text.split():
            trial = (line + " " + word).strip()
            if line and _text_width(trial, size, kind, bold, 0.0) > max_px:
                lines.append(line)
                line = word
            else:
                line = trial
        if line:
            lines.append(line)
        return lines

    @staticmethod
    def clip(s: str, max_px: float, size: int = 14, kind: str = "sans", bold: bool = False,
             track: float = 0.0) -> str:
        """``s`` shortened with an ellipsis so it is no wider than ``max_px``."""
        if _text_width(s, size, kind, bold, track) <= max_px:
            return s
        while s and _text_width(s + "…", size, kind, bold, track) > max_px:
            s = s[:-1]
        return s.rstrip() + "…"

    # ------------------------------------------------------------------ drawing
    def text(self, xy, s: str, size: int = 14, color=(242, 242, 242), kind: str = "sans",
             bold: bool = False, anchor: str = "l", track: float = 0.0) -> float:
        """Draw ``s`` with its baseline at ``xy``.  ``anchor`` is "l", "m" or "r"
        (horizontal) - the vertical anchor is always the baseline.  ``track`` adds
        letter spacing in pixels.  Returns the width drawn."""
        if not s:
            return 0.0
        x, y = xy
        tw = _text_width(s, size, kind, bold, track)
        if anchor == "m":
            x -= tw / 2
        elif anchor == "r":
            x -= tw
        if not self.ok or _font(kind, bold, size) is None:
            cv2.putText(self.frame, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size / 28.0,
                        color, 2 if bold else 1, cv2.LINE_AA)
            return tw
        mask, dx, dy = _mask(s, size, kind, bold, track)
        left, top = int(round(x)) + dx, int(round(y)) + dy
        mh, mw = mask.shape[:2]
        # clip to the pen's rectangle
        fx0, fy0 = max(left, self.x0), max(top, self.y0)
        fx1, fy1 = min(left + mw, self.x1), min(top + mh, self.y1)
        if fx1 <= fx0 or fy1 <= fy0:
            return tw
        m = mask[fy0 - top:fy1 - top, fx0 - left:fx1 - left]
        roi = self.frame[fy0:fy1, fx0:fx1]
        r = roi.astype(np.int16)
        r += ((np.asarray(color, dtype=np.int16) - r) * m) >> 6
        roi[:] = r
        return tw


def rule(frame: np.ndarray, x: int, y: int, w: int, h: int = 2) -> None:
    """The blue-orange-pink hairline that sits under every title."""
    H, W = frame.shape[:2]
    for i in range(w):
        xx = x + i
        if 0 <= xx < W:
            frame[max(0, y):min(H, y + h), xx] = theme.bgr(theme.glow_at(i / max(1, w - 1)))


def panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.88,
          border: str | None = "border2", fill: str = "bg") -> None:
    """A black translucent box with a one-pixel border and square corners."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame[y0:y1, x0:x1]
    c = theme.bgr(theme.TOKENS[fill])
    if c[0] == c[1] == c[2]:                       # a grey fill: one in-place pass, no temporary
        cv2.convertScaleAbs(sub, dst=sub, alpha=1.0 - alpha, beta=c[0] * alpha)
    else:
        cv2.addWeighted(np.full(sub.shape, c, dtype=np.uint8), alpha, sub, 1 - alpha, 0, sub)
    if border:
        col = theme.bgr(theme.TOKENS.get(border, border))
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), col, 1)


def bar(frame: np.ndarray, x: int, y: int, w: int, h: int, frac: float, colour,
        track: str = "border") -> None:
    """A thin square progress bar: dark track, solid fill."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), theme.bgr(theme.TOKENS[track]), -1)
    fill = int(w * max(0.0, min(1.0, frac)))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), colour, -1)
