"""MediaPipe BlazePose wrapper (Methodology Stage 3, 'Pose Estimation').

Wraps the MediaPipe Tasks PoseLandmarker so the rest of the code never has to
know about mediapipe types.  Every frame in gives one `Pose` out, holding the
33 body landmarks in pixel coordinates, their visibility scores, and the
One-Euro-smoothed copy that the geometry actually uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .filters import LandmarkSmoother

# ---------------------------------------------------------------- landmark ids
NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

N_LANDMARKS = 33

#: The landmarks an asana check can depend on.  If any of these is below the
#: visibility floor we refuse to score the frame rather than guess.
CORE_LANDMARKS = (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)

CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@dataclass
class Pose:
    """One frame of body geometry."""

    #: (33, 2) smoothed landmark positions in pixels, origin top-left.
    pts: np.ndarray
    #: (33, 2) raw, unsmoothed pixel positions - drawn faintly for comparison.
    raw: np.ndarray
    #: (33,) visibility in 0..1 as reported by BlazePose.
    vis: np.ndarray
    #: (33, 3) world landmarks in metres, hip-centred (unused by the base
    #: implementation, kept because Stage 3 depth work will need it).
    world: np.ndarray
    width: int = 0
    height: int = 0
    meta: dict = field(default_factory=dict)

    def p(self, idx: int) -> np.ndarray:
        return self.pts[idx]

    def mid(self, a: int, b: int) -> np.ndarray:
        return (self.pts[a] + self.pts[b]) * 0.5

    def min_visibility(self, ids=CORE_LANDMARKS) -> float:
        return float(np.min(self.vis[list(ids)]))


class PoseTracker:
    """Stateful, per-video-stream pose estimator."""

    def __init__(self, model: str = "full", min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, smooth: bool = True,
                 running_mode: str = "video") -> None:
        path = model if os.path.isfile(model) else os.path.join(_MODEL_DIR, f"pose_landmarker_{model}.task")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Pose model not found: {path}\n"
                "Run setup.bat, or download it with:\n"
                "  curl -L -o models/pose_landmarker_full.task https://storage.googleapis.com/"
                "mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/"
                "pose_landmarker_full.task"
            )
        mode = (vision.RunningMode.VIDEO if running_mode == "video"
                else vision.RunningMode.IMAGE)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mode = mode
        self._smoother = LandmarkSmoother(N_LANDMARKS, n_axes=2) if smooth else None
        self.model_path = path

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int = 0,
                fps: float | None = None) -> Pose | None:
        """`frame_rgb` is an HxWx3 uint8 RGB image.  Returns None if no body."""
        h, w = frame_rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        if self._mode == vision.RunningMode.VIDEO:
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        else:
            result = self._landmarker.detect(image)

        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]
        raw = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=float)
        vis = np.array([_score(lm) for lm in lms], dtype=float)
        world = (np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]], dtype=float)
                 if result.pose_world_landmarks else np.zeros((N_LANDMARKS, 3)))

        pts = self._smoother(raw, fps) if self._smoother is not None else raw.copy()
        return Pose(pts=pts, raw=raw, vis=vis, world=world, width=w, height=h)


def _score(lm) -> float:
    v = getattr(lm, "visibility", None)
    p = getattr(lm, "presence", None)
    vals = [x for x in (v, p) if x is not None]
    return float(min(vals)) if vals else 1.0
