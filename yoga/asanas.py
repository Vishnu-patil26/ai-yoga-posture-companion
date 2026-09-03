"""Asana library (Methodology Stage 4, 'asana library holding the reference angles').

An asana is nothing more than a list of `Check`s.  Each check names one
measurable feature of the body, the value it should have, and how far the
practitioner is allowed to stray before it is called a mistake.  Adding a
second asana later means adding another `Asana(...)` here - no engine change.

The base implementation ships one asana, Vrikshasana (Tree Pose), which is the
asana used in the methodology diagram.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Check:
    """One joint-level rule.  `target` +/- `tol` is 'correct'."""

    key: str                 #: name of the feature in the measured feature dict
    label: str               #: short label for the on-screen table
    target: float
    tol: float               #: inside this band the check scores a full 1.0
    zero_at: float           #: absolute deviation at which the check scores 0.0
    weight: float = 1.0
    unit: str = "deg"        #: "deg" or "ratio" - only affects display
    #: "band" penalises straying either way.  "max" penalises only values above
    #: target+tol, "min" only values below target-tol.  One-sided checks matter:
    #: the hips in Vrikshasana sit about 10 degrees off level even when the pose
    #: is right, so tilt beyond that is a fault while perfectly level hips are
    #: not.  A two-sided band would have called a good pose wrong.
    mode: str = "band"
    #: Body parts this check needs to see.  Named groups from
    #: `evaluator.VISIBILITY_GROUPS`, resolved per frame - "standing_leg" and
    #: "folded_leg" follow whichever leg the practitioner is actually on.
    needs: tuple[str, ...] = ()
    cue: str = ""            #: fallback spoken cue
    cue_under: str = ""      #: spoken when the measured value is below target
    cue_over: str = ""       #: spoken when the measured value is above target

    def cue_for(self, deviation: float) -> str:
        if deviation < 0 and self.cue_under:
            return self.cue_under
        if deviation > 0 and self.cue_over:
            return self.cue_over
        return self.cue or f"Adjust your {self.label.lower()}"


@dataclass(frozen=True)
class Variant:
    """An accepted way of performing the same asana.

    Vrikshasana is taught with the hands overhead *or* pressed together at the
    heart; both are correct. Scoring against a single arm target would mark
    one of the two standard forms wrong. Each variant replaces some of the
    base checks by key; the practitioner is scored against whichever variant
    fits best, so they are never corrected towards a form they did not choose.
    """

    key: str
    name: str
    overrides: tuple[Check, ...]


@dataclass(frozen=True)
class Asana:
    key: str
    name: str
    sanskrit: str
    level: int
    checks: tuple[Check, ...]
    variants: tuple[Variant, ...] = ()
    hold_target_s: float = 20.0
    #: score (0-100) needed to start the hold timer, and to keep it running.
    enter_score: float = 78.0
    exit_score: float = 62.0
    setup_hint: str = ""
    notes: str = ""

    def check(self, key: str) -> Check | None:
        for c in self.checks:
            if c.key == key:
                return c
        return None

    def checks_for(self, variant: Variant | None) -> tuple[Check, ...]:
        """The base checks with this variant's overrides applied."""
        if variant is None:
            return self.checks
        replaced = {c.key: c for c in variant.overrides}
        return tuple(replaced.get(c.key, c) for c in self.checks)


# ---------------------------------------------------------------------------
# Reference geometry for Vrikshasana.
#
# The numbers below are NOT hand-picked.  They were fitted from 195 usable
# tree-pose photographs (5 subjects) in the public TensorFlow/Moroney yoga-pose
# dataset with tools/fit_reference.py: target = median, tolerance = 1.5 x the
# MAD-based robust sigma, bounded.  See docs/REFERENCES.md for the method, the
# literature it follows and the limitations of the sample.
#
# Where the fitted value disagreed with the textbook ideal, the disagreement is
# noted in the comment and the decision explained - the data is not followed
# blindly for alignment faults that are wrong for everybody.
# ---------------------------------------------------------------------------

VRIKSHASANA = Asana(
    key="vrikshasana",
    name="Tree Pose",
    sanskrit="Vrikshasana",
    level=2,
    hold_target_s=20.0,
    enter_score=78.0,
    exit_score=62.0,
    setup_hint="Stand facing the camera, full body in frame, about 2 m away.",
    notes=(
        "Standing leg is detected automatically - either side is accepted. "
        "Both standard arm forms are accepted (overhead, or hands at the heart). "
        "Tolerances narrow to the practitioner's own body after calibration."
    ),
    checks=(
        Check(
            # Fitted median 3.5 deg on frontal views, p90 5.8.  The target is
            # held at the ideal 0 rather than the median: a leaning spine is a
            # fault for everybody, and the fitted median is close enough to 0
            # to confirm the ideal is the right reference.
            key="spine_tilt", label="Spine upright", target=0.0, tol=8.0, zero_at=26.0,
            weight=1.4, needs=("torso",), mode="max",
            cue_over="Lift your chest and stack your spine over your hips",
            cue="Lift your chest and stack your spine over your hips",
        ),
        Check(
            # Fitted median 10.1 deg even on square-on views.  The pelvis
            # genuinely hikes over the standing leg in this asana, so a target
            # of 0 - the obvious guess - would have nagged every correct
            # practitioner.  One-sided: more level than this is never a fault.
            key="hip_level", label="Hips level", target=10.0, tol=8.0, zero_at=32.0,
            weight=0.9, needs=("hips", "frontal"), mode="max",
            cue_over="Settle your hips - don't push the standing hip out to the side",
        ),
        Check(
            # Noisy in a single camera view (fitted p90 21-30 deg), so it is
            # kept at low weight and a wide band rather than dropped.
            key="shoulder_level", label="Shoulders level", target=6.0, tol=14.0,
            zero_at=38.0, weight=0.5, needs=("shoulders", "frontal"), mode="max",
            cue_over="Drop the raised shoulder and keep them even",
        ),
        Check(
            # Fitted median 176.1 deg, robust sigma 3.6, p10 168.3.
            key="standing_knee", label="Standing leg straight", target=176.0, tol=8.0,
            zero_at=40.0, weight=1.2, needs=("standing_leg",), mode="min",
            cue_under="Straighten your standing leg",
            cue="Straighten your standing leg",
        ),
        Check(
            # Fitted median 29.6 deg, robust sigma 11.0 - the heel is drawn in
            # much further than the 45 deg first guessed.
            key="folded_knee", label="Lifted knee folded", target=30.0, tol=17.0,
            zero_at=65.0, weight=0.9, needs=("folded_leg",),
            cue_over="Fold the lifted leg more and draw the heel in",
            cue_under="Ease the lifted heel away from the hip",
        ),
        Check(
            # Fitted median 55.9 deg, robust sigma 12.4.
            key="folded_thigh_open", label="Lifted knee opened out", target=56.0,
            tol=19.0, zero_at=62.0, weight=1.0, needs=("folded_leg",),
            cue_under="Open the lifted knee out to the side",
            cue_over="Bring the lifted knee slightly forward",
        ),
        Check(
            # The most consistent measurement in the whole asana: fitted median
            # 0.80 of the standing leg's length, robust sigma 0.04, p10 0.70.
            # One-sided - a foot placed higher still is not a fault.
            key="foot_height_ratio", label="Foot placement", target=0.80, tol=0.10,
            zero_at=0.45, weight=1.2, needs=("standing_leg", "folded_leg"),
            unit="ratio", mode="min",
            cue_under="Place the foot higher, onto the inner thigh",
        ),
        # Arm checks below are the "arms overhead" form; the hands-at-heart
        # variant overrides them.  See VARIANT_* under the asana.
        Check(
            key="arm_raise_left", label="Left arm overhead", target=15.0, tol=22.0,
            zero_at=80.0, weight=0.8, needs=("left_arm",), mode="max",
            cue_over="Reach the left arm straight up overhead",
        ),
        Check(
            key="arm_raise_right", label="Right arm overhead", target=15.0, tol=22.0,
            zero_at=80.0, weight=0.8, needs=("right_arm",), mode="max",
            cue_over="Reach the right arm straight up overhead",
        ),
        # Not scored in the overhead form (the wrists are nowhere near the
        # chest and should not be), but present so the hands-at-heart variant
        # has a slot to override.  Weight 0 keeps it out of the overhead score.
        Check(
            key="wrist_to_chest_left", label="Left palm", target=1.06, tol=2.0,
            zero_at=4.0, weight=0.0, needs=("left_arm",), unit="ratio",
        ),
        Check(
            key="wrist_to_chest_right", label="Right palm", target=1.06, tol=2.0,
            zero_at=4.0, weight=0.0, needs=("right_arm",), unit="ratio",
        ),
    ),
    # Clustering the fitted arm angles split cleanly in two, and every one of
    # the five subjects appears in both clusters - so this is two accepted
    # forms of the asana, not two kinds of practitioner.
    #   overhead      : arm raise 14-21 deg off vertical
    #   hands at heart: arm raise 125-143 deg
    # The elbow angle was measured and then dropped as a check: within each arm
    # form it is itself bimodal (straight arms vs palms pressed together), so it
    # does not discriminate a correct pose from an incorrect one.
    variants=(
        Variant(
            key="overhead", name="arms overhead",
            overrides=(),          # the base checks already describe this form
        ),
        Variant(
            key="heart", name="hands at the heart",
            overrides=(
                Check(
                    key="arm_raise_left", label="Left hand at heart", target=134.0,
                    tol=36.0, zero_at=60.0, weight=0.6, needs=("left_arm",),
                    cue="Bring the palms together in front of your chest",
                ),
                Check(
                    key="arm_raise_right", label="Right hand at heart", target=134.0,
                    tol=36.0, zero_at=60.0, weight=0.6, needs=("right_arm",),
                    cue="Bring the palms together in front of your chest",
                ),
                # `arm_raise` alone cannot separate hands-at-the-heart from
                # arms hanging by the sides - both point down from the
                # shoulder.  Wrist-to-chest distance does, cleanly: fitted
                # median 0.26 torso lengths at the heart (p90 0.45) against
                # 1.06 for the overhead form (p10 0.65).  Without this the
                # app would accept somebody simply standing with their arms
                # down as a correct hands-at-the-heart Vrikshasana.
                Check(
                    key="wrist_to_chest_left", label="Left palm in to chest",
                    target=0.26, tol=0.24, zero_at=0.85, weight=0.7,
                    needs=("left_arm",), unit="ratio", mode="max",
                    cue_over="Draw your palms in to the centre of your chest",
                ),
                Check(
                    key="wrist_to_chest_right", label="Right palm in to chest",
                    target=0.26, tol=0.24, zero_at=0.85, weight=0.7,
                    needs=("right_arm",), unit="ratio", mode="max",
                    cue_over="Draw your palms in to the centre of your chest",
                ),
            ),
        ),
    ),
)

LIBRARY: dict[str, Asana] = {VRIKSHASANA.key: VRIKSHASANA}


def get(key: str) -> Asana:
    try:
        return LIBRARY[key]
    except KeyError:
        raise SystemExit(f"Unknown asana '{key}'. Available: {', '.join(LIBRARY)}")


# ---------------------------------------------------------------------------
# Fitted asanas.
#
# VRIKSHASANA above is hand-tuned: its checks use features that only exist for
# a one-legged standing balance, and each target was argued for against the
# data one at a time.  That does not scale to a library.
#
# The poses below are fitted automatically by tools/fit_asana.py from a
# side-agnostic feature vocabulary (bent/straight knee, bent/straight elbow,
# high/low arm, spine angle, stance width), which is mirror-invariant and so
# describes an asana without caring which side it is performed on.  Every
# target is the median of the dataset class and every tolerance is derived
# from its spread - nothing here is guessed.
#
# Loading is best-effort: a missing or unreadable fits file leaves the library
# with the hand-tuned asana only, rather than preventing the app from starting.
# ---------------------------------------------------------------------------

import json as _json
import os as _os

_FITS_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "data", "asana_fits.json")


def _asana_from_fit(spec: dict) -> Asana:
    checks = []
    for key, c in spec["checks"].items():
        checks.append(Check(
            key=key, label=c["label"], target=float(c["target"]),
            tol=float(c["tol"]), zero_at=float(c["zero_at"]),
            weight=float(c["weight"]), unit=c.get("unit", "deg"),
            needs=("torso",),
            cue=f"Adjust your {c['label'].lower()}",
        ))
    return Asana(
        key=spec["key"], name=spec["name"], sanskrit=spec["sanskrit"],
        level=int(spec.get("level", 2)), checks=tuple(checks),
        setup_hint="Stand facing the camera, full body in frame, about 2 m away.",
        notes=(f"Fitted from {spec.get('n_images', '?')} images of the "
               f"'{spec.get('source_class')}' class with tools/fit_asana.py."),
    )


def load_fitted(path: str = _FITS_PATH) -> dict[str, Asana]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = _json.load(fh)
    except (OSError, ValueError):
        return {}
    out = {}
    for key, spec in data.items():
        try:
            out[key] = _asana_from_fit(spec)
        except (KeyError, TypeError, ValueError):
            continue                       # skip a malformed entry, keep the rest
    return out


#: Keys the fitted loader must not register - the hand-tuned Vrikshasana is a
#: better definition of the same pose (it uses features specific to a one-legged
#: balance, and accepts both arm forms), so its automatic twin would only be a
#: confusing second entry for the same asana.
_SUPERSEDED_BY_HAND_TUNED = {"vrikshasana_fit"}

for _key, _asana in load_fitted().items():
    if _key in _SUPERSEDED_BY_HAND_TUNED:
        continue
    LIBRARY.setdefault(_key, _asana)
