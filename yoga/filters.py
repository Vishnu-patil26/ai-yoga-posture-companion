"""Signal conditioning for landmark jitter (Methodology Stage 3).

MediaPipe landmarks wobble by a few pixels even when the practitioner is
perfectly still.  Feeding that straight into an angle calculation makes the
joint angles flicker by 3-5 degrees, which in turn makes the voice cues
chatter.  We damp it with a One-Euro filter: heavy smoothing when the body is
still, almost no smoothing when it is moving fast, so a real correction is
never delayed.
"""

from __future__ import annotations

import math

import numpy as np


class _LowPass:
    def __init__(self) -> None:
        self.y: float | None = None

    def __call__(self, x: float, alpha: float) -> float:
        self.y = x if self.y is None else alpha * x + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    """Scalar One-Euro filter (Casiez, Roussel & Vogel, 2012)."""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, freq: float | None = None) -> float:
        if freq and freq > 1e-6:
            self.freq = freq
        prev = self._prev if self._prev is not None else x
        dx = (x - prev) * self.freq
        edx = self._dx(dx, self._alpha(self.d_cutoff, self.freq))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        out = self._x(x, self._alpha(cutoff, self.freq))
        self._prev = x
        return out


class LandmarkSmoother:
    """One-Euro filter applied per landmark, per axis."""

    def __init__(self, n_points: int, n_axes: int = 3, freq: float = 30.0,
                 min_cutoff: float = 1.2, beta: float = 0.02) -> None:
        self._f = [[OneEuroFilter(freq, min_cutoff, beta) for _ in range(n_axes)]
                   for _ in range(n_points)]
        self.n_axes = n_axes

    def __call__(self, pts: np.ndarray, freq: float | None = None) -> np.ndarray:
        out = np.empty_like(pts, dtype=float)
        for i in range(pts.shape[0]):
            for a in range(min(self.n_axes, pts.shape[1])):
                out[i, a] = self._f[i][a](float(pts[i, a]), freq)
        return out


class RollingMean:
    """Small moving average, used to steady the reported alignment score."""

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._buf: list[float] = []

    def __call__(self, x: float) -> float:
        self._buf.append(x)
        if len(self._buf) > self.window:
            self._buf.pop(0)
        return sum(self._buf) / len(self._buf)

    def reset(self) -> None:
        self._buf.clear()
