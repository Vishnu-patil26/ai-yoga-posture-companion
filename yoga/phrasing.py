"""Body-relative language: say a correction in words a person can act on
without knowing what a degree or a ratio is.

The conversion is anthropometric, not measured on the practitioner: an adult
hand (palm to fingertip) is on average close to 11% of standing height, and a
torso (shoulder to hip) is close to 29% of height - so a hand is roughly 0.38
of a torso length.  That single constant, `HAND_WIDTH_FRAC_OF_TORSO`, is what
lets every linear check - however it happens to be normalised for scoring -
be expressed in the same everyday unit.  It is deliberately an approximation:
nobody holding a balance pose benefits from a number precise to the
millimetre, they benefit from something they can act on immediately.

Two kinds of check need two kinds of phrase:

* **Linear** checks (foot placement, palms-to-chest, how level the hips and
  shoulders are, how far the spine leans) describe a *distance* between body
  parts, so they get the hand-width analogy the project was asked for.
* **Bend** checks (a knee angle, an arm raised from the shoulder) describe a
  *rotation*, not a distance.  Forcing a fake "hand-width" onto a joint bend
  would be precise-looking but not actually true, so those get a plain
  qualitative severity word instead ("a little", "quite a bit") - honest
  about being approximate rather than dressing up a guess as a measurement.
"""

from __future__ import annotations

import math

from .evaluator import CheckResult

#: See the module docstring for where this comes from.
HAND_WIDTH_FRAC_OF_TORSO = 0.38

#: Which stored feature is the right "ruler" for a linear-style check, and
#: whether its deviation is already a fraction of that ruler ("ratio") or a
#: tilt angle measured across it ("tilt", converted via the small-baseline
#: approximation linear_px = ruler_px * tan(angle)).
_RULER: dict[str, tuple[str, str]] = {
    "foot_height_ratio": ("standing_leg_px", "ratio"),
    "wrist_to_chest_left": ("torso_px", "ratio"),
    "wrist_to_chest_right": ("torso_px", "ratio"),
    "hip_level": ("hip_width_px", "tilt"),
    "shoulder_level": ("shoulder_width_px", "tilt"),
    "spine_tilt": ("torso_px", "tilt"),
}

#: Degrees beyond this are clamped before the tan() conversion - a joint
#: reading 85 degrees off is landmark noise, not a real hand-width figure.
_MAX_TILT_DEG = 75.0

#: Full phrasing for voice and the wrapping banner - no column-width limit.
_HAND_WIDTH_WORDS = (
    (0.25, "just a touch"),
    (0.75, "about half a hand's width"),
    (1.5, "about a hand's width"),
    (2.5, "about one and a half hand's widths"),
    (float("inf"), "more than two hand's widths"),
)

#: Compact phrasing for the joint table, which is a narrow fixed-width
#: column - the full sentences above would run off the edge or get truncated
#: mid-word, which is worse than a short phrase.
_HAND_WIDTH_WORDS_SHORT = (
    (0.25, "a touch off"),
    (0.75, "~half a hand"),
    (1.5, "~1 hand's width"),
    (2.5, "~1.5 hand widths"),
    (float("inf"), "2+ hand widths"),
)

_SEVERITY_WORDS = (
    (0.15, "just a touch"),
    (0.4, "a little"),
    (0.7, "quite a bit"),
    (float("inf"), "a long way"),
)


def _bucket(value: float, table: tuple[tuple[float, str], ...]) -> str:
    for limit, word in table:
        if value <= limit:
            return word
    return table[-1][1]


def handwidths(result: CheckResult, features: dict) -> float | None:
    """Roughly how many hand-widths off this check is.

    Returns None for a check with no natural linear analogy (a joint bend),
    for an ungraded check, or if the reference lengths could not be measured.
    """
    ruler = _RULER.get(result.check.key)
    if ruler is None or result.ok is None:
        return None
    feature_name, kind = ruler
    baseline = features.get(feature_name, 0.0)
    hand_px = features.get("torso_px", 0.0) * HAND_WIDTH_FRAC_OF_TORSO
    if baseline <= 1e-3 or hand_px <= 1e-3:
        return None
    if kind == "ratio":
        linear_px = abs(result.deviation) * baseline
    else:
        deg = max(-_MAX_TILT_DEG, min(_MAX_TILT_DEG, result.deviation))
        linear_px = baseline * abs(math.tan(math.radians(deg)))
    return linear_px / hand_px


def severity_word(result: CheckResult) -> str:
    """A qualitative sense of how far off a bend check is."""
    if result.ok is None:
        return "can't see it"
    if result.ok:
        return "correct"
    span = max(1e-6, result.check.zero_at - result.check.tol)
    return _bucket(result.error / span, _SEVERITY_WORDS)


def short_words(result: CheckResult, features: dict) -> str:
    """What the on-screen joint table shows in place of a raw number.

    Kept deliberately short - this renders in a fixed-width column, not a
    wrapped sentence, so the full hand-width phrasing lives in
    `spoken_phrase` instead.
    """
    if result.ok is None:
        return "can't see"
    if result.ok:
        return "correct"
    hw = handwidths(result, features)
    if hw is not None:
        return _bucket(hw, _HAND_WIDTH_WORDS_SHORT)
    return severity_word(result)


def spoken_phrase(result: CheckResult, level: str, features: dict,
                  terse: bool = False) -> str:
    """What the voice coach actually says.

    First time a fault comes up, the full instruction plus a body-relative
    sense of how far off it is.  On a repeat (`terse`), just the short form -
    the practitioner already knows what the problem is and does not need the
    whole sentence again.  Matches the `phrase_fn` hook in
    `feedback.CueEngine`; the on-screen text is unaffected either way.
    """
    if result.ok:
        return result.cue()
    if terse:
        return short_cue(result)
    # First time: the short instruction plus how far off it is.  Speaking the
    # full written sentence AND the qualifier takes 6.2 s, which is most of a
    # breath and crowds out the hold countdown; the complete wording stays on
    # screen in the banner, where reading it costs nothing.
    hw = handwidths(result, features)
    qualifier = _bucket(hw, _HAND_WIDTH_WORDS) if hw is not None else severity_word(result)
    return f"{short_cue(result).rstrip('.')} - {qualifier}."


# =============================================================================
# Spoken brevity.
#
# A written instruction and a spoken one are not the same thing.  "Lift your
# chest and stack your spine over your hips - about half a hand's width" reads
# well on screen and takes 6.2 seconds to say out loud; repeated every seven
# seconds it means the app is talking more than it is silent.  Measured on a
# real session, the full-sentence cues produced 39.9 s of speech inside a
# 24.9 s hold - a 160% talking ratio, so most of it was dropped mid-flow.
#
# So the screen keeps the full sentence and the voice gets a short form.  The
# FIRST time a fault is raised it is said in full, with the hand-width sense
# of how far off it is; every repeat after that is the terse version, the way
# a teacher says "chest up" the second time rather than repeating themselves
# word for word.
# =============================================================================

#: (check key, direction) -> what to actually say out loud on a repeat.
#: Direction is "over" when the measured value is above target, "under" below.
_SHORT_CUES: dict[tuple[str, str], str] = {
    ("spine_tilt", "over"): "Chest up.",
    ("spine_tilt", "under"): "Chest up.",
    ("hip_level", "over"): "Level your hips.",
    ("shoulder_level", "over"): "Shoulders level.",
    ("standing_knee", "under"): "Straighten that leg.",
    ("folded_knee", "over"): "Heel in closer.",
    ("folded_knee", "under"): "Ease the heel down.",
    ("folded_thigh_open", "under"): "Knee out wider.",
    ("folded_thigh_open", "over"): "Knee forward a little.",
    ("foot_height_ratio", "under"): "Foot higher.",
    ("arm_raise_left", "over"): "Left arm up.",
    ("arm_raise_right", "over"): "Right arm up.",
    ("wrist_to_chest_left", "over"): "Palms to your chest.",
    ("wrist_to_chest_right", "over"): "Palms to your chest.",
}


def short_cue(result: CheckResult) -> str:
    """The terse spoken form of a correction.

    Falls back to the opening clause of the full cue, so a newly added check
    still says something sensible before anyone writes a short form for it.
    """
    direction = "under" if result.deviation < 0 else "over"
    terse = _SHORT_CUES.get((result.check.key, direction))
    if terse:
        return terse
    full = result.cue()
    clause = full.split(" - ")[0].split(",")[0].strip()
    words = clause.split()
    if len(words) > 5:
        clause = " ".join(words[:5])
    return clause.rstrip(".") + "."
