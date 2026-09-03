"""Feature measurement and joint-by-joint scoring (Methodology Stage 3).

`compute_features` turns 33 landmarks into the handful of numbers a yoga
teacher actually looks at.  `evaluate` compares those numbers against the
selected asana and reports the deviation *per joint*, not as a single pass or
fail - which is the point the methodology makes about existing apps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import angles as A
from .asanas import Asana, Check, Variant
from .landmarks import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
    Pose,
)

#: A check is graded only when the body parts it needs are at least this
#: visible.  The floor is deliberately low.  BlazePose reports `visibility` as
#: "in frame and not occluded", and in Vrikshasana the lifted ankle is pressed
#: against the standing thigh, so it is *always* partly occluded - measured
#: median visibility 0.50 over 120 tree-pose photographs, tenth percentile
#: 0.28.  A strict floor would therefore refuse to grade the single most
#: characteristic joint of the pose.  What actually makes a landmark unusable
#: is being outside the frame, which is handled separately below.
LANDMARK_FLOOR = 0.25

#: The torso is never occluded in a standing asana, so if we cannot see it
#: clearly the practitioner is not properly in frame and nothing is graded.
TORSO_FLOOR = 0.60

#: A frame is scored only if checks worth this share of the total weight could
#: be measured.
MEASURABLE_WEIGHT_FLOOR = 0.60

#: Shoulder-width / torso-length below which the practitioner is too turned
#: away for image-plane "is it level" checks to mean anything.  Measured over
#: 195 tree-pose photographs: median 0.48, so this keeps roughly the frontal
#: half.  A practitioner following the setup hint sits well above it.
FRONTAL_MIN = 0.45

#: Named groups a check can declare in `Check.needs`.  "standing_leg" and
#: "folded_leg" are resolved per frame to whichever leg the practitioner is on;
#: "frontal" is 1.0 only when the body is square enough to the camera.
VISIBILITY_GROUPS = ("torso", "shoulders", "hips", "standing_leg", "folded_leg",
                     "left_arm", "right_arm", "frontal")


def compute_features(pose: Pose) -> dict:
    """Measure the body.  Values are degrees unless the name says otherwise."""
    p = pose.pts
    ls, rs = p[L_SHOULDER], p[R_SHOULDER]
    lh, rh = p[L_HIP], p[R_HIP]
    mid_sh = (ls + rs) * 0.5
    mid_hip = (lh + rh) * 0.5

    f: dict = {
        "spine_tilt": A.tilt_from_vertical(mid_hip, mid_sh),
        "spine_tilt_signed": A.signed_tilt_from_vertical(mid_hip, mid_sh),
        "shoulder_level": A.tilt_from_horizontal(ls, rs),
        "hip_level": A.tilt_from_horizontal(lh, rh),
        "knee_left": A.joint_angle(lh, p[L_KNEE], p[L_ANKLE]),
        "knee_right": A.joint_angle(rh, p[R_KNEE], p[R_ANKLE]),
        "elbow_left": A.joint_angle(ls, p[L_ELBOW], p[L_WRIST]),
        "elbow_right": A.joint_angle(rs, p[R_ELBOW], p[R_WRIST]),
        "arm_raise_left": A.tilt_from_vertical(ls, p[L_WRIST]),
        "arm_raise_right": A.tilt_from_vertical(rs, p[R_WRIST]),
        "torso_px": A.distance(mid_sh, mid_hip),
    }

    # How far the wrists are from the middle of the chest, in torso lengths.
    # `arm_raise` alone cannot tell hands-pressed-at-the-heart from arms
    # hanging by the sides - both point downwards from the shoulder - but this
    # separates them cleanly: at the heart the wrists are drawn in to the
    # sternum, by the sides they are a whole torso away.
    chest = mid_sh + (mid_hip - mid_sh) * 0.35
    if f["torso_px"] > 1e-3:
        f["wrist_to_chest_left"] = A.distance(p[L_WRIST], chest) / f["torso_px"]
        f["wrist_to_chest_right"] = A.distance(p[R_WRIST], chest) / f["torso_px"]
    else:
        f["wrist_to_chest_left"] = f["wrist_to_chest_right"] = float("nan")

    # ---- which leg is the practitioner standing on? -------------------------
    # The straighter leg; if both are similar, the one whose foot is lower.
    kl, kr = f["knee_left"], f["knee_right"]
    if math.isnan(kl) or math.isnan(kr):
        standing = "left"
    elif abs(kl - kr) < 15.0:
        standing = "left" if p[L_ANKLE][1] > p[R_ANKLE][1] else "right"
    else:
        standing = "left" if kl > kr else "right"
    folded = "right" if standing == "left" else "left"
    f["standing_side"] = standing
    f["folded_side"] = folded

    s_hip, s_knee, s_ankle = (lh, p[L_KNEE], p[L_ANKLE]) if standing == "left" else (rh, p[R_KNEE], p[R_ANKLE])
    d_hip, d_knee, d_ankle = (lh, p[L_KNEE], p[L_ANKLE]) if folded == "left" else (rh, p[R_KNEE], p[R_ANKLE])

    f["standing_knee"] = f[f"knee_{standing}"]
    f["folded_knee"] = f[f"knee_{folded}"]
    # How far the lifted thigh is swung out to the side, measured from straight down.
    f["folded_thigh_open"] = A.tilt_from_down(d_hip, d_knee)

    # Where the lifted foot sits on the standing leg.
    # 0.0 = at the standing ankle, ~0.5 = at the knee, 1.0 = at the hip.
    leg_span = float(s_ankle[1] - s_hip[1])
    f["foot_height_ratio"] = (float(s_ankle[1] - d_ankle[1]) / leg_span
                              if leg_span > 1e-3 else float("nan"))

    # Pixel lengths of the body's own segments.  Nothing here is used for
    # scoring - phrasing.py uses them as "rulers" to turn a tilt or a ratio
    # into a body-relative distance (see docs there for why).
    f["standing_leg_px"] = abs(leg_span)
    f["hip_width_px"] = A.distance(lh, rh)
    f["shoulder_width_px"] = A.distance(ls, rs)

    # ---- side-agnostic geometry, for asanas other than this one -------------
    # "Left" and "right" are the wrong handles for a pose library: the same
    # asana done on the other side, or simply mirrored by the webcam, swaps
    # them and the reference no longer matches.  Sorting each pair into
    # bent/straight instead makes every measurement mirror-invariant, which is
    # what lets one set of checks describe a pose regardless of which side the
    # practitioner leads with.
    kl, kr = f["knee_left"], f["knee_right"]
    el, er = f["elbow_left"], f["elbow_right"]
    al, ar = f["arm_raise_left"], f["arm_raise_right"]
    f["knee_bent"], f["knee_straight"] = min(kl, kr), max(kl, kr)
    f["elbow_bent"], f["elbow_straight"] = min(el, er), max(el, er)
    f["arm_raise_high"], f["arm_raise_low"] = min(al, ar), max(al, ar)
    # How wide the feet are planted, in torso lengths - the thing that
    # separates a lunge from a stand without caring which foot is forward.
    f["stance_width"] = (A.distance(p[L_ANKLE], p[R_ANKLE]) / f["torso_px"]
                         if f["torso_px"] > 1e-3 else float("nan"))

    # How square-on the practitioner is: shoulder width over torso length.
    # Level-of-the-hips and level-of-the-shoulders are measured in the image
    # plane, so they are only meaningful when the body faces the camera - turn
    # 45 degrees and a perfectly level pelvis photographs as a tilted one.
    f["frontality"] = (A.distance(ls, rs) / f["torso_px"]) if f["torso_px"] > 1e-3 else 0.0

    f["min_visibility"] = pose.min_visibility()
    f["group_visibility"] = _group_visibility(pose, standing, folded)
    f["group_visibility"]["frontal"] = 1.0 if f["frontality"] >= FRONTAL_MIN else 0.0
    f["framing"] = _framing(pose, f)
    return f


#: The practitioner should fill this much of the frame height, head to heel.
#: Below it they are too far away for the ankles to be located reliably; above
#: it, the raised arms or the feet start leaving the frame.
FRAME_FILL_MIN, FRAME_FILL_MAX = 0.45, 0.95


def _framing(pose: Pose, f: dict) -> dict:
    """Is the practitioner properly in shot?  Drives the on-screen guide."""
    p, v = pose.pts, pose.vis
    h = float(pose.height or 1)
    margin = 0.01 * max(pose.width, pose.height)

    def inside(i: int) -> bool:
        x, y = p[i]
        return -margin <= x <= pose.width + margin and -margin <= y <= pose.height + margin

    head_ok = v[NOSE] > 0.5 and inside(NOSE)
    torso_ok = f["group_visibility"]["torso"] >= TORSO_FLOOR
    feet_ok = all(inside(i) for i in (L_ANKLE, R_ANKLE)) and \
        max(v[L_ANKLE], v[R_ANKLE]) > LANDMARK_FLOOR
    top = float(min(p[NOSE][1], p[L_SHOULDER][1], p[R_SHOULDER][1]))
    bottom = float(max(p[L_ANKLE][1], p[R_ANKLE][1]))
    fill = max(0.0, (bottom - top) / h)

    if not (head_ok or torso_ok):
        advice = "Step into view"
    elif not feet_ok or fill > FRAME_FILL_MAX:
        advice = "Step back - your feet must be in shot"
    elif fill < FRAME_FILL_MIN:
        advice = "Come closer to the camera"
    elif f["frontality"] < FRONTAL_MIN:
        advice = "Turn to face the camera"
    else:
        advice = ""

    return {
        "head": bool(head_ok),
        "torso": bool(torso_ok),
        "feet": bool(feet_ok),
        "facing": bool(f["frontality"] >= FRONTAL_MIN),
        "fill": fill,
        "distance_ok": bool(FRAME_FILL_MIN <= fill <= FRAME_FILL_MAX),
        "ok": bool(head_ok and torso_ok and feet_ok
                   and FRAME_FILL_MIN <= fill <= FRAME_FILL_MAX),
        "advice": advice,
    }


def _group_visibility(pose: Pose, standing: str, folded: str) -> dict:
    """Lowest usable confidence per body part.

    A landmark that has drifted outside the image is scored 0 no matter what
    the network claims: BlazePose will happily extrapolate a leg past the
    bottom edge of the frame, and grading that would be inventing data.
    """
    v = pose.vis.copy()
    margin = 0.02 * max(pose.width, pose.height)
    x, y = pose.pts[:, 0], pose.pts[:, 1]
    outside = (x < -margin) | (x > pose.width + margin) | (y < -margin) | (y > pose.height + margin)
    v[outside] = 0.0

    def lo(*ids: int) -> float:
        return float(min(v[i] for i in ids))

    s_hip, s_knee, s_ankle = ((L_HIP, L_KNEE, L_ANKLE) if standing == "left"
                              else (R_HIP, R_KNEE, R_ANKLE))
    d_hip, d_knee, d_ankle = ((L_HIP, L_KNEE, L_ANKLE) if folded == "left"
                              else (R_HIP, R_KNEE, R_ANKLE))
    return {
        "shoulders": lo(L_SHOULDER, R_SHOULDER),
        "hips": lo(L_HIP, R_HIP),
        "torso": lo(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP),
        "standing_leg": lo(s_hip, s_knee, s_ankle),
        "folded_leg": lo(d_hip, d_knee, d_ankle),
        "left_arm": lo(L_SHOULDER, L_ELBOW, L_WRIST),
        "right_arm": lo(R_SHOULDER, R_ELBOW, R_WRIST),
    }


@dataclass
class CheckResult:
    check: Check
    value: float
    deviation: float      #: value - target (signed)
    error: float          #: how far outside the tolerance band, 0 when inside
    score: float          #: 0..1
    ok: bool | None       #: None when the feature could not be measured

    @property
    def label(self) -> str:
        return self.check.label

    def cue(self) -> str:
        return self.check.cue_for(self.deviation)

    def display(self) -> str:
        if self.ok is None:
            return "--"
        if self.check.unit == "ratio":
            return f"{self.value:.2f} ({self.deviation:+.2f})"
        return f"{self.value:.0f} ({self.deviation:+.0f})"


@dataclass
class Evaluation:
    asana: Asana
    score: float                    #: 0..100, weighted over every measurable check
    results: list[CheckResult]
    features: dict
    usable: bool                    #: False when the body is not visible enough
    reason: str = ""
    #: Which accepted form of the asana the practitioner was scored against.
    variant: Variant | None = None

    @property
    def variant_name(self) -> str:
        return self.variant.name if self.variant else ""

    @property
    def failures(self) -> list[CheckResult]:
        """Checks outside tolerance, worst (most weight x error) first."""
        bad = [r for r in self.results if r.ok is False]
        bad.sort(key=lambda r: r.check.weight * (1.0 - r.score), reverse=True)
        return bad

    @property
    def primary(self) -> CheckResult | None:
        bad = self.failures
        return bad[0] if bad else None


def score_check(check: Check, value: float, group_vis: dict | None = None) -> CheckResult:
    """Grade one check, or report it unmeasurable if we cannot see the joint."""
    if group_vis is not None and check.needs:
        if min(group_vis.get(g, 1.0) for g in check.needs) < LANDMARK_FLOOR:
            return CheckResult(check, float("nan"), float("nan"), float("nan"), 0.0, None)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return CheckResult(check, float("nan"), float("nan"), float("nan"), 0.0, None)
    deviation = value - check.target
    if check.mode == "max":          # only too much is a fault
        error = max(0.0, deviation - check.tol)
    elif check.mode == "min":        # only too little is a fault
        error = max(0.0, -deviation - check.tol)
    else:
        error = max(0.0, abs(deviation) - check.tol)
    span = max(1e-6, check.zero_at - check.tol)
    score = max(0.0, min(1.0, 1.0 - error / span))
    return CheckResult(check, value, deviation, error, score, error <= 0.0)


#: How much of the alignment score comes from the single worst joint.  A plain
#: weighted average is far too forgiving - with eleven checks, one badly wrong
#: joint moves the average by a few points and the app would happily award a
#: "hold" to a pose the practitioner is doing wrong.  A yoga teacher grades the
#: other way round: the pose is only as good as its weakest link.
WORST_WEIGHT = 0.40


def _combine(measurable: list[CheckResult]) -> float:
    """Blend the weighted mean with the worst single joint.  Returns 0..1."""
    wsum = sum(r.check.weight for r in measurable)
    mean = sum(r.score * r.check.weight for r in measurable) / wsum
    # A low-weight check cannot drag the score all the way down; its shortfall
    # is scaled by how much that joint is declared to matter.
    worst = min(1.0 - (1.0 - r.score) * min(1.0, r.check.weight) for r in measurable)
    return (1.0 - WORST_WEIGHT) * mean + WORST_WEIGHT * worst


def evaluate(asana: Asana, features: dict) -> Evaluation:
    """Score the body against the asana, choosing the best-fitting variant.

    When an asana has more than one accepted form, every form is scored and the
    highest wins.  The practitioner is then corrected towards the form they are
    already attempting instead of being told to do the other one.
    """
    if asana.variants:
        best = max((_evaluate_one(asana, features, v) for v in asana.variants),
                   key=lambda e: (e.usable, e.score))
        return best
    return _evaluate_one(asana, features, None)


def _evaluate_one(asana: Asana, features: dict, variant) -> Evaluation:
    group_vis = features.get("group_visibility")
    checks = asana.checks_for(variant)
    results = [score_check(c, features.get(c.key), group_vis) for c in checks]
    measurable = [r for r in results if r.ok is not None]

    usable = True
    reason = ""
    total_weight = sum(c.weight for c in checks)
    measured_weight = sum(r.check.weight for r in measurable)
    torso = (group_vis or {}).get("torso", 1.0)
    if torso < TORSO_FLOOR:
        usable, reason = False, "Step back - your whole body must be in frame"
    elif measured_weight < total_weight * MEASURABLE_WEIGHT_FLOOR:
        usable, reason = False, "Cannot see enough of your body to score"

    if measurable:
        score = 100.0 * _combine(measurable)
    else:
        # Keep the more specific reason if one was already worked out above.
        score, usable = 0.0, False
        reason = reason or "No body detected"

    return Evaluation(asana=asana, score=score, results=results,
                      features=features, usable=usable, reason=reason,
                      variant=variant)
