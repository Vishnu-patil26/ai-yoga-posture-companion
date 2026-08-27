"""The reference figure: what the asana should look like.

Builds a skeleton with exactly the geometry asked for, in the same landmark
format the camera produces.  Two things use it:

  * the on-screen reference card, which shows the practitioner what to do next
    without needing a video of an instructor;
  * the offline self-test, which needs poses whose right answer is known.

Because the card and the tests are drawn from the same builder, the figure the
practitioner is shown is by construction the figure the scorer accepts.
"""

from __future__ import annotations

import math

import numpy as np

from . import landmarks as LM
from .landmarks import Pose

#: Canvas the reference figure is built on; it is rescaled wherever it is drawn.
W, H = 1280, 720


def _dir_from_down(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.sin(r), math.cos(r)])


def _dir_from_up(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.sin(r), -math.cos(r)])


def ideal_pose(spine_tilt: float = 0.0, hip_tilt: float = 0.0,
                   shoulder_tilt: float = 0.0, thigh_open: float = 62.0,
                   foot_ratio: float = 0.70, arm_tilt: float = 8.0,
                   elbow_bend: float = 12.0, standing_knee_bend: float = 2.0,
                   standing: str = "right", hands_at_chest: bool = False) -> Pose:
    """Build a Vrikshasana skeleton with the geometry we asked for.

    Angles are in degrees.  `standing` names the anatomical leg on the floor.
    """
    pts = np.zeros((LM.N_LANDMARKS, 2), dtype=float)

    hip_c = np.array([W * 0.60, H * 0.55])
    torso, hip_w, sh_w = 150.0, 66.0, 94.0
    thigh, shank, upper_arm, fore_arm = 135.0, 130.0, 95.0, 92.0

    sh_c = hip_c + _dir_from_up(spine_tilt) * torso

    def spread(centre, width, tilt_deg):
        r = math.radians(tilt_deg)
        off = np.array([math.cos(r), math.sin(r)]) * (width / 2.0)
        return centre - off, centre + off

    pts[LM.L_HIP], pts[LM.R_HIP] = spread(hip_c, hip_w, hip_tilt)
    pts[LM.L_SHOULDER], pts[LM.R_SHOULDER] = spread(sh_c, sh_w, shoulder_tilt)

    s_hip_i = LM.L_HIP if standing == "left" else LM.R_HIP
    s_knee_i = LM.L_KNEE if standing == "left" else LM.R_KNEE
    s_ankle_i = LM.L_ANKLE if standing == "left" else LM.R_ANKLE
    d_hip_i = LM.R_HIP if standing == "left" else LM.L_HIP
    d_knee_i = LM.R_KNEE if standing == "left" else LM.L_KNEE
    d_ankle_i = LM.R_ANKLE if standing == "left" else LM.L_ANKLE

    # Standing leg: hangs straight down, optionally with a small bend.
    s_hip = pts[s_hip_i]
    pts[s_knee_i] = s_hip + _dir_from_down(0.0) * thigh
    pts[s_ankle_i] = pts[s_knee_i] + _dir_from_down(standing_knee_bend) * shank

    # Lifted leg: thigh swung out to the side, foot placed up the standing leg.
    # The side sign puts the lifted knee away from the body midline.
    side = 1.0 if pts[d_hip_i][0] >= hip_c[0] else -1.0
    d_hip = pts[d_hip_i]
    pts[d_knee_i] = d_hip + _dir_from_down(thigh_open * side) * thigh

    span = float(pts[s_ankle_i][1] - s_hip[1])
    foot_y = pts[s_ankle_i][1] - foot_ratio * span
    if foot_ratio < 0.12:
        # Foot on the floor: the leg hangs straight below its own knee, which
        # is what standing actually looks like.  Placing it under the *standing*
        # knee would leave a permanently bent leg and no figure could ever
        # satisfy a "stand tall" check.
        pts[d_ankle_i] = np.array([pts[d_knee_i][0], foot_y])
    else:
        pts[d_ankle_i] = np.array([pts[s_knee_i][0] + 12.0 * side, foot_y])

    if hands_at_chest:
        # Palms pressed together in front of the sternum.  This cannot be built
        # by rotating a straight arm - the forearm has to fold back in - so it
        # is placed directly: wrists at the chest, elbows dropped out wide.
        chest = sh_c + (hip_c - sh_c) * 0.35
        for sh_i, el_i, wr_i, sgn in ((LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST, -1.0),
                                      (LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST, 1.0)):
            pts[wr_i] = chest + np.array([sgn * 6.0, 0.0])
            pts[el_i] = pts[sh_i] + _dir_from_down(58.0 * sgn) * upper_arm * 0.8
    else:
        # Arms reaching, both sides symmetric.
        for sh_i, el_i, wr_i, sgn in ((LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST, -1.0),
                                      (LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST, 1.0)):
            sh = pts[sh_i]
            pts[el_i] = sh + _dir_from_up(arm_tilt * sgn) * upper_arm
            pts[wr_i] = pts[el_i] + _dir_from_up((arm_tilt + elbow_bend) * sgn) * fore_arm

    # Head and feet, only so the drawing/visibility code has something sane.
    head = sh_c + _dir_from_up(spine_tilt) * 58.0
    pts[LM.NOSE] = head
    for idx, dx, dy in ((1, -7, -7), (2, -12, -7), (3, -17, -7),
                        (4, 7, -7), (5, 12, -7), (6, 17, -7),
                        (7, -24, -2), (8, 24, -2), (9, -7, 11), (10, 7, 11)):
        pts[idx] = head + np.array([float(dx), float(dy)])
    for ankle_i, heel_i, foot_i in ((LM.L_ANKLE, LM.L_HEEL, LM.L_FOOT),
                                    (LM.R_ANKLE, LM.R_HEEL, LM.R_FOOT)):
        pts[heel_i] = pts[ankle_i] + np.array([-8.0, 14.0])
        pts[foot_i] = pts[ankle_i] + np.array([16.0, 16.0])

    # Hands (17-22) - never used by a check, but they are drawn, so they must
    # not be left sitting at the origin.
    for wr_i, ids in ((LM.L_WRIST, (17, 19, 21)), (LM.R_WRIST, (18, 20, 22))):
        for n, idx in enumerate(ids):
            pts[idx] = pts[wr_i] + np.array([(n - 1) * 7.0, -14.0 - n * 3.0])

    vis = np.ones(LM.N_LANDMARKS)
    return Pose(pts=pts, raw=pts.copy(), vis=vis,
                world=np.zeros((LM.N_LANDMARKS, 3)), width=W, height=H)


#: Poses for the reference card, one per step of the guided sequence.
STEP_FIGURES = {
    "frame":  dict(foot_ratio=0.02, thigh_open=3.0, arm_tilt=168.0, standing_knee_bend=1.0),
    "stand":  dict(foot_ratio=0.02, thigh_open=3.0, arm_tilt=168.0, standing_knee_bend=1.0),
    "weight": dict(foot_ratio=0.16, thigh_open=18.0, arm_tilt=168.0),
    "foot":   dict(foot_ratio=0.80, thigh_open=56.0, arm_tilt=168.0),
    "hands":  dict(foot_ratio=0.80, thigh_open=56.0, arm_tilt=12.0),
    "hold":   dict(foot_ratio=0.80, thigh_open=56.0, arm_tilt=12.0),
    "heart":  dict(foot_ratio=0.80, thigh_open=56.0, hands_at_chest=True),
    "done":   dict(foot_ratio=0.02, thigh_open=3.0, arm_tilt=168.0),
}

#: The two accepted arm forms, cycled on the card so the practitioner can see
#: that either is allowed.
ARM_FORMS = ("hold", "heart")


def step_figure(step_key: str) -> Pose:
    """The figure to draw on the reference card for a step of the sequence."""
    return ideal_pose(**STEP_FIGURES.get(step_key, STEP_FIGURES["hold"]))
