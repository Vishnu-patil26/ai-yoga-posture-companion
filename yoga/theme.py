"""The look of the whole project: one black theme, used by every screen.

The values are the design tokens of the Smart Customer Segmentation app's dark
theme (near-black page, hairline borders, square corners, white as the one
accent, monospaced capitals for labels, a thin blue-orange-pink gradient rule
under titles).  They live here, with no Tk and no OpenCV import, so the
launcher, the volunteer-collection window and the live camera HUD all draw from
the same list and cannot drift apart.

Colours are written as ``#rrggbb`` strings.  Tk takes them as they are;
OpenCV wants BGR tuples, so ``bgr()`` converts.
"""

from __future__ import annotations

import os
import sys

TOKENS = {
    "bg": "#050505",       # the page
    "card": "#0d0d0d",     # sidebar and cards
    "card2": "#141414",    # inputs and raised rows
    "border": "#262626",   # every hairline
    "border2": "#3a3a3a",  # a hairline that has to read on top of video
    "text": "#f2f2f2",
    "muted": "#8c8c8c",
    "dim": "#5c5c5c",
    "accent": "#ffffff",   # primary button, active marker
    "accent2": "#3fd8ff",  # the one highlight colour
    "good": "#2fe0b0",
    "warn": "#ffc247",
    "bad": "#ff5c5c",
}

#: the rule under every title
GLOW = ("#4f8cff", "#ff8a3d", "#ff4fa3")

#: categorical colours, same order as the segmentation app's charts
PALETTE = ("#4f8cff", "#ff8a3d", "#ff4fa3", "#3fd8ff", "#d4f25a",
           "#9b7bff", "#2fe0b0", "#ff5c5c", "#ffc247", "#e6e6e6")

# Typefaces, first one installed wins.  Tk and Pillow find them differently, so
# there is a family list (Tk) and a file list (Pillow); both end in something
# that exists everywhere so a missing font degrades the look, never the app.
SANS_FAMILIES = ("Segoe UI", "SF Pro Text", "Helvetica Neue", "DejaVu Sans", "Arial")
MONO_FAMILIES = ("Cascadia Mono", "Consolas", "SF Mono", "Menlo", "DejaVu Sans Mono", "Courier New")

_FONT_DIRS = (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu",
              "/usr/share/fonts/dejavu", "/Library/Fonts", "/System/Library/Fonts",
              "/System/Library/Fonts/Supplemental")
_FONT_FILES = {
    ("sans", False): ("segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"),
    ("sans", True): ("segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"),
    ("mono", False): ("CascadiaMono.ttf", "consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"),
    ("mono", True): ("consolab.ttf", "DejaVuSansMono-Bold.ttf", "Menlo.ttc"),
}


def font_file(kind: str = "sans", bold: bool = False) -> str | None:
    """Path of an installed TrueType file for Pillow, or None when there is none."""
    for name in _FONT_FILES[(kind, bold)]:
        for d in _FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def bgr(h: str) -> tuple[int, int, int]:
    """``#rrggbb`` -> the (b, g, r) tuple OpenCV draws with."""
    r, g, b = hex_rgb(h)
    return b, g, r


def mix(a: str, b: str, t: float) -> str:
    """The colour a fraction ``t`` of the way from ``a`` to ``b``."""
    t = max(0.0, min(1.0, t))
    ra, rb = hex_rgb(a), hex_rgb(b)
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ra, rb))


def glow_at(t: float) -> str:
    """A point along the title rule: 0 = blue, 0.5 = orange, 1 = pink."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return mix(GLOW[0], GLOW[1], t * 2)
    return mix(GLOW[1], GLOW[2], (t - 0.5) * 2)


def is_windows() -> bool:
    return sys.platform == "win32"
