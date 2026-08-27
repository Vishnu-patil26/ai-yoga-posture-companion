"""Joint geometry (Methodology Stage 3, 'Asana Comparison').

Everything here works in image pixel space with the origin at the top-left,
so `y` grows *downwards*.  The helpers below hide that so the asana
definitions can be written the way a yoga teacher would say them:
"is the spine upright", "is the standing knee straight".
"""

from __future__ import annotations

import math

import numpy as np

# Unit vectors in image space.
UP = np.array([0.0, -1.0])
DOWN = np.array([0.0, 1.0])
RIGHT = np.array([1.0, 0.0])


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Unsigned angle between two vectors, in degrees (0..180)."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = float(np.dot(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at `b` for the chain a-b-c, in degrees.

    180 deg means the limb is straight, small values mean deeply folded.
    """
    return angle_between(a - b, c - b)


def tilt_from_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """How far the segment a->b leans away from straight up, in degrees."""
    return angle_between(b - a, UP)


def tilt_from_down(a: np.ndarray, b: np.ndarray) -> float:
    """How far the segment a->b leans away from straight down, in degrees."""
    return angle_between(b - a, DOWN)


def tilt_from_horizontal(a: np.ndarray, b: np.ndarray) -> float:
    """How far the line a-b is off level, in degrees (0 = perfectly level)."""
    t = angle_between(b - a, RIGHT)
    return min(t, 180.0 - t)


def signed_tilt_from_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """Lean of a->b from vertical; positive means leaning to the image right."""
    v = b - a
    return math.degrees(math.atan2(float(v[0]), float(-v[1])))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
