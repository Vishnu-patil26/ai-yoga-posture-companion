"""The hierarchical model of yoga categories (project-guide meeting, 28 Oct viva).

A four-level tree, plain data plus a few lookup helpers:

    level 1  BODY POSITION        where the body is relative to the floor
    level 2  FUNCTIONAL CATEGORY  what the pose is *for* (balance, backbend, ...)
    level 3  POSE                 a key from ``yoga.routines.POSES``
    level 4  BODY POINTS          the BlazePose landmarks that matter for the
                                  category, and the joint angles / features the
                                  app measures from them

Why this shape.  A yoga teacher does not think "ten unrelated poses"; they
think "a balance, two strength poses, a side stretch ...".  Grouping by
position first (standing / seated / all-fours / prone) is the safest first cut
because the position decides what a single front-on camera can even see, and
grouping by function second is what lets the app say *why* a pose is in a
routine.  Each category belongs to exactly one position, so the model is a
strict tree - every pose has one path from root to leaf and the whole thing can
be shown in a Tk ``Treeview`` without cross-links.  The functional idea
"forward fold" in a different position (a standing forward fold, say) would be
its own node under that position rather than a second parent.

Level 4 is deliberately truthful.  ``Category.features`` lists names that
really exist in ``yoga.evaluator.compute_features`` (``validate`` reads that
file and checks), and ``measured_features`` reports the subset the app scores
*today* - which is empty for the guided poses, because no dataset reference
exists for them and the app never invents angles (see ``routines.py``).

This module has no dependency on mediapipe: the landmark ids below mirror
``yoga/landmarks.py`` and ``validate`` re-reads that file to prove they agree.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

from yoga import routines

# ------------------------------------------------------------ landmark ids
# Same names and values as yoga/landmarks.py (checked by validate()).
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

#: Constant name -> id, the subset of landmarks.py this module relies on.
LANDMARK_CONSTANTS = {
    "NOSE": NOSE, "L_EYE": L_EYE, "R_EYE": R_EYE, "L_EAR": L_EAR, "R_EAR": R_EAR,
    "L_SHOULDER": L_SHOULDER, "R_SHOULDER": R_SHOULDER, "L_ELBOW": L_ELBOW,
    "R_ELBOW": R_ELBOW, "L_WRIST": L_WRIST, "R_WRIST": R_WRIST, "L_HIP": L_HIP,
    "R_HIP": R_HIP, "L_KNEE": L_KNEE, "R_KNEE": R_KNEE, "L_ANKLE": L_ANKLE,
    "R_ANKLE": R_ANKLE, "L_HEEL": L_HEEL, "R_HEEL": R_HEEL, "L_FOOT": L_FOOT,
    "R_FOOT": R_FOOT,
}

#: The 33 BlazePose landmark names, by id.
LANDMARK_NAMES = {
    0: "nose", 1: "left eye inner", 2: "left eye", 3: "left eye outer",
    4: "right eye inner", 5: "right eye", 6: "right eye outer", 7: "left ear",
    8: "right ear", 9: "mouth left", 10: "mouth right", 11: "left shoulder",
    12: "right shoulder", 13: "left elbow", 14: "right elbow", 15: "left wrist",
    16: "right wrist", 17: "left pinky", 18: "right pinky", 19: "left index",
    20: "right index", 21: "left thumb", 22: "right thumb", 23: "left hip",
    24: "right hip", 25: "left knee", 26: "right knee", 27: "left ankle",
    28: "right ankle", 29: "left heel", 30: "right heel", 31: "left foot index",
    32: "right foot index",
}

_EARS = (L_EAR, R_EAR)
_SHOULDERS = (L_SHOULDER, R_SHOULDER)
_ELBOWS = (L_ELBOW, R_ELBOW)
_WRISTS = (L_WRIST, R_WRIST)
_HIPS = (L_HIP, R_HIP)
_KNEES = (L_KNEE, R_KNEE)
_ANKLES = (L_ANKLE, R_ANKLE)


# ---------------------------------------------------------------- features
@dataclass(frozen=True)
class Feature:
    """One number ``compute_features`` produces, and the landmarks it needs."""

    name: str
    kind: str                 #: "tilt" | "angle" | "ratio"
    points: tuple[int, ...]   #: landmark ids the measurement is built from
    meaning: str              #: plain-English description for the viva


_TORSO = _SHOULDERS + _HIPS

FEATURES: dict[str, Feature] = {f.name: f for f in (
    Feature("spine_tilt", "tilt", _TORSO,
            "lean of the hip-to-shoulder line from vertical (0 = upright)"),
    Feature("spine_tilt_signed", "tilt", _TORSO,
            "the same lean with a sign, so a left and a right side-bend differ"),
    Feature("neck_dev", "angle", _TORSO + _EARS,
            "angle between the spine and the shoulder-to-ear line (0 = head stacked)"),
    Feature("shoulder_level", "tilt", _SHOULDERS,
            "how far the shoulder line is from horizontal"),
    Feature("hip_level", "tilt", _HIPS,
            "how far the hip line is from horizontal"),
    Feature("knee_bent", "angle", _HIPS + _KNEES + _ANKLES,
            "hip-knee-ankle angle of the more bent leg (side-agnostic)"),
    Feature("knee_straight", "angle", _HIPS + _KNEES + _ANKLES,
            "hip-knee-ankle angle of the straighter leg (side-agnostic)"),
    Feature("elbow_bent", "angle", _SHOULDERS + _ELBOWS + _WRISTS,
            "shoulder-elbow-wrist angle of the more bent arm (side-agnostic)"),
    Feature("elbow_straight", "angle", _SHOULDERS + _ELBOWS + _WRISTS,
            "shoulder-elbow-wrist angle of the straighter arm (side-agnostic)"),
    Feature("arm_raise_high", "tilt", _SHOULDERS + _WRISTS,
            "shoulder-to-wrist angle from vertical for the higher arm (0 = straight up)"),
    Feature("arm_raise_low", "tilt", _SHOULDERS + _WRISTS,
            "shoulder-to-wrist angle from vertical for the lower arm"),
    Feature("arm_raise_left", "tilt", (L_SHOULDER, L_WRIST),
            "left shoulder-to-wrist angle from vertical"),
    Feature("arm_raise_right", "tilt", (R_SHOULDER, R_WRIST),
            "right shoulder-to-wrist angle from vertical"),
    Feature("hip_angle_min", "angle", _TORSO + _KNEES,
            "shoulder-hip-knee angle of the more folded side (180 = standing tall, small = folded)"),
    Feature("hip_angle_vis", "angle", _TORSO + _KNEES,
            "shoulder-hip-knee angle on the side the camera can see (for side-view poses)"),
    Feature("knee_angle_vis", "angle", _HIPS + _KNEES + _ANKLES,
            "hip-knee-ankle angle on the visible side (for side-view poses)"),
    Feature("elbow_angle_vis", "angle", _SHOULDERS + _ELBOWS + _WRISTS,
            "shoulder-elbow-wrist angle on the visible side (for side-view poses)"),
    Feature("arm_torso_vis", "angle", _TORSO + _WRISTS,
            "hip-shoulder-wrist angle on the visible side (for side-view poses)"),
    Feature("hip_angle_max", "angle", _TORSO + _KNEES,
            "shoulder-hip-knee angle of the more open side"),
    Feature("arm_torso_min", "angle", _TORSO + _WRISTS,
            "hip-shoulder-wrist angle of the arm closer to the trunk (how far the arm is opened)"),
    Feature("arm_torso_max", "angle", _TORSO + _WRISTS,
            "hip-shoulder-wrist angle of the arm further from the trunk"),
    Feature("hip_rise", "ratio", _TORSO,
            "how far the hips are above the shoulders, in torso lengths (inverted V is positive)"),
    Feature("head_drop", "ratio", (NOSE,) + _TORSO,
            "how far the nose is below the shoulder line, in torso lengths (cat vs cow)"),
    Feature("thigh_level", "tilt", _HIPS + _KNEES,
            "how far the thigh is from horizontal (0 = parallel to the floor)"),
    Feature("stance_width", "ratio", _ANKLES + _TORSO,
            "distance between the ankles in torso lengths (feet apart or together)"),
    Feature("standing_knee", "angle", _HIPS + _KNEES + _ANKLES,
            "knee angle of the leg the person is standing on"),
    Feature("folded_knee", "angle", _HIPS + _KNEES + _ANKLES,
            "knee angle of the lifted, folded leg"),
    Feature("folded_thigh_open", "tilt", _HIPS + _KNEES,
            "how far the lifted thigh is swung out to the side from straight down"),
    Feature("foot_height_ratio", "ratio", _HIPS + _ANKLES,
            "where the lifted foot rests on the standing leg (0 ankle, 0.5 knee, 1 hip)"),
    Feature("wrist_to_chest_left", "ratio", (L_WRIST,) + _TORSO,
            "left wrist to mid-chest distance in torso lengths (hands at the heart)"),
    Feature("wrist_to_chest_right", "ratio", (R_WRIST,) + _TORSO,
            "right wrist to mid-chest distance in torso lengths (hands at the heart)"),
)}


# ------------------------------------------------------------------- tree
#: Level 1.  Key -> label, in display order.
POSITIONS: dict[str, str] = {
    "standing": "Standing",
    "seated": "Seated",
    "all_fours_kneeling": "All-fours & kneeling",
    "prone": "Prone (face-down)",
}


@dataclass(frozen=True)
class Category:
    """Level 2 node, carrying the level 3 poses and the level 4 detail."""

    key: str
    label: str
    position: str                    #: key into POSITIONS - exactly one
    purpose: str                     #: one sentence: what this group is for
    poses: tuple[str, ...]           #: level 3, keys from routines.POSES
    points: tuple[int, ...]          #: level 4: BlazePose landmark ids that matter
    features: tuple[str, ...]        #: level 4: names from compute_features
    focus: str = ""                  #: why those points / angles for this group


CATEGORIES: dict[str, Category] = {c.key: c for c in (
    Category(
        "foundation", "Foundation & alignment", "standing",
        "Teach the neutral, stacked posture every other standing pose starts from.",
        ("tadasana",),
        _EARS + _SHOULDERS + _HIPS + _KNEES + _ANKLES,
        ("spine_tilt", "neck_dev", "shoulder_level", "hip_level", "knee_straight",
         "stance_width"),
        "Head over shoulders over hips over ankles: it is all about vertical "
        "stacking and level lines."),
    Category(
        "balance", "Balance", "standing",
        "Train single-leg stability and steady attention.",
        ("vrikshasana",),
        _EARS + _SHOULDERS + _WRISTS + _HIPS + _KNEES + _ANKLES,
        ("neck_dev", "spine_tilt", "hip_level", "shoulder_level", "standing_knee",
         "folded_knee", "folded_thigh_open", "foot_height_ratio", "arm_raise_left",
         "arm_raise_right", "wrist_to_chest_left", "wrist_to_chest_right"),
        "One straight standing leg carries the body, so the knee, the foot "
        "placement and the pelvis matter most."),
    Category(
        "strength", "Strength & stance", "standing",
        "Build leg and hip strength by holding a loaded knee bend with a long spine.",
        ("virabhadrasana", "utkatasana"),
        _EARS + _SHOULDERS + _ELBOWS + _WRISTS + _HIPS + _KNEES + _ANKLES,
        ("knee_bent", "knee_straight", "stance_width", "spine_tilt", "neck_dev",
         "elbow_bent", "elbow_straight", "arm_raise_high", "arm_raise_low"),
        "The bend of the working knee, the width of the stance and an upright "
        "spine separate a strong pose from a collapsed one."),
    Category(
        "lateral", "Lateral stretch", "standing",
        "Lengthen the side body and open hips and hamstrings by bending sideways "
        "over straight legs.",
        ("trikonasana",),
        _SHOULDERS + _ELBOWS + _WRISTS + _HIPS + _KNEES + _ANKLES,
        ("spine_tilt_signed", "spine_tilt", "knee_straight", "stance_width",
         "elbow_straight", "arm_raise_high", "arm_raise_low"),
        "A wide stance, straight legs and one arm reaching up while the other "
        "reaches down; the signed spine tilt tells which way the body bends."),
    Category(
        "stillness", "Stillness & breath", "seated",
        "Sit tall and steady so the breath and attention can settle.",
        ("sukhasana",),
        _EARS + _SHOULDERS + _HIPS + _KNEES,
        ("spine_tilt", "neck_dev", "shoulder_level", "hip_level"),
        "Nothing to perform: a tall spine, a level pelvis and a head that is not "
        "dropped forward."),
    Category(
        "mobility", "Spinal mobility", "all_fours_kneeling",
        "Move the spine through bending and arching with the breath to warm and "
        "free the back.",
        ("marjaryasana",),
        _EARS + _SHOULDERS + _ELBOWS + _WRISTS + _HIPS + _KNEES + _ANKLES,
        ("spine_tilt", "neck_dev", "elbow_straight", "knee_bent"),
        "Wrists under shoulders, knees under hips; what changes is the curve of "
        "the spine and the position of the head."),
    Category(
        "fold_rest", "Forward fold & rest", "all_fours_kneeling",
        "Fold forward over the thighs to release the lower back and calm the body.",
        ("balasana",),
        _EARS + _SHOULDERS + _ELBOWS + _WRISTS + _HIPS + _KNEES + _ANKLES,
        ("spine_tilt", "neck_dev", "knee_bent", "elbow_straight"),
        "The hips sit back towards the heels while the arms and spine lengthen "
        "forward."),
    Category(
        "inversion", "Inversion (head below heart)", "all_fours_kneeling",
        "Bring the hips above the head to stretch calves and hamstrings and "
        "lengthen the spine - skipped when high blood pressure, age 60+ or some "
        "medicines make that unwise.",
        ("adho_mukha",),
        _EARS + _SHOULDERS + _ELBOWS + _WRISTS + _HIPS + _KNEES + _ANKLES,
        ("knee_straight", "knee_bent", "elbow_straight", "elbow_bent", "spine_tilt",
         "arm_raise_high", "arm_raise_low", "stance_width", "neck_dev"),
        "Straight legs and arms form the inverted V; the spine line and the head "
        "position between the arms show whether it is long or collapsed."),
    Category(
        "backbend", "Backbend", "prone",
        "Extend the spine gently to undo a rounded desk posture while the hips "
        "stay on the floor.",
        ("bhujangasana",),
        _EARS + _SHOULDERS + _ELBOWS + _WRISTS + _HIPS + _KNEES + _ANKLES,
        ("spine_tilt", "neck_dev", "elbow_bent", "elbow_straight", "arm_raise_high",
         "arm_raise_low", "knee_straight", "knee_bent", "stance_width"),
        "How far the chest lifts (spine tilt), how straight the arms push and "
        "whether the neck stays long."),
)}

#: How routines.py's three coarse positions map onto the four taxonomy ones.
_COARSE_TO_FINE = {
    "standing": {"standing"},
    "sitting": {"seated"},
    "floor": {"all_fours_kneeling", "prone"},
}

#: routines.py pose tags that name the same idea as a category.
_TAG_TO_CATEGORY = {
    "inversion": "inversion", "backbend": "backbend", "forward_fold": "fold_rest",
    "balance": "balance", "side_bend": "lateral",
}


# ----------------------------------------------------------------- lookups
def category_of(pose_key: str) -> Category:
    """The one category a pose sits in.  Raises KeyError for an unknown pose."""
    for c in CATEGORIES.values():
        if pose_key in c.poses:
            return c
    raise KeyError(pose_key)


def position_of(pose_key: str) -> str:
    """The pose's body-position key (a key of ``POSITIONS``)."""
    return category_of(pose_key).position


def pose_path(pose_key: str) -> str:
    """Root-to-leaf path, e.g. ``Standing > Balance > Tree Pose``."""
    c = category_of(pose_key)
    return " > ".join((POSITIONS[c.position], c.label, routines.POSES[pose_key].name))


def measured_features(pose_key: str) -> list[str]:
    """Features the app actually scores for this pose *today*.

    Read from the asana library at run time, so it is empty for a guided pose
    (no fitted reference) and grows by itself when ``tools/fitting/fit_asana.py`` turns
    a guided pose into a scored one.
    """
    from yoga import asanas
    asana = asanas.LIBRARY.get(pose_key)
    if asana is None:
        return []
    keys: list[str] = []
    checks = list(asana.checks)
    for variant in asana.variants:
        checks.extend(variant.overrides)
    for c in checks:
        if c.weight > 0 and c.key not in keys:
            keys.append(c.key)
    return keys


def features_for(pose_key: str) -> list[str]:
    """Feature names for this pose: its category's, plus anything the app scores
    for it today that the category does not already list.

    The union keeps the model truthful without hand-editing: when a dataset fit
    makes a guided pose scored, whatever it is scored on shows up here.
    """
    base = list(category_of(pose_key).features)
    return base + [f for f in measured_features(pose_key) if f not in base]


def body_points(pose_key: str) -> list[tuple[int, str]]:
    """``(landmark_id, name)`` for every body point that matters for this pose:
    the category's points plus the landmarks every listed feature is built from."""
    ids = set(category_of(pose_key).points)
    for f in features_for(pose_key):
        feat = FEATURES.get(f)
        if feat is not None:
            ids.update(feat.points)
    return [(i, LANDMARK_NAMES[i]) for i in sorted(ids)]


def feature_meaning(name: str) -> str:
    feat = FEATURES.get(name)
    return feat.meaning if feat else "measured by the asana's fitted reference"


def _pose_label(pose_key: str) -> str:
    p = routines.POSES[pose_key]
    return f"{p.name} ({p.sanskrit})"


def tree(details: bool = True) -> dict:
    """The whole model as nested dicts, ready for a Tk ``Treeview``.

    Every node is ``{text: children}`` and a leaf has ``{}`` children, so one
    recursive loop fills the widget::

        def add(parent, node):
            for text, kids in node.items():
                add(tv.insert(parent, "end", text=text), kids)

    With ``details`` each pose carries its level-4 leaves (body points and
    features); without it the tree stops at the pose.
    """
    out: dict = {}
    for pos_key, pos_label in POSITIONS.items():
        cats: dict = {}
        for c in CATEGORIES.values():
            if c.position != pos_key:
                continue
            poses: dict = {}
            for pk in c.poses:
                node: dict = {}
                if details:
                    node["Body points"] = {f"{i}  {name}": {}
                                           for i, name in body_points(pk)}
                    node["Joint angles / features"] = {
                        f"{f} - {feature_meaning(f)}": {} for f in features_for(pk)}
                poses[_pose_label(pk)] = node
            cats[c.label] = poses
        out[pos_label] = cats
    return out


def describe_tree() -> str:
    """The model as printable indented text (for the viva and the README)."""
    lines = ["Yoga taxonomy: body position > functional category > pose > body points"]
    for pos_key, pos_label in POSITIONS.items():
        lines.append(f"{pos_label}")
        for c in CATEGORIES.values():
            if c.position != pos_key:
                continue
            lines.append(f"  {c.label} - {c.purpose}")
            for pk in c.poses:
                mode = "scored" if routines.POSES[pk].is_scored else "guided"
                lines.append(f"    {_pose_label(pk)} [{mode}]")
                pts = ", ".join(f"{i} {name}" for i, name in body_points(pk))
                lines.append(f"      body points: {pts}")
                lines.append(f"      features: {', '.join(features_for(pk))}")
                now = measured_features(pk)
                lines.append("      measured now: " + (", ".join(now) if now
                                                       else "nothing (guided, not scored)"))
    return "\n".join(lines)


# -------------------------------------------------------------- validation
_PKG = os.path.dirname(os.path.abspath(__file__))


def _compute_features_names() -> set[str] | None:
    """Every string literal inside ``evaluator.compute_features`` (None if unreadable).

    Parsed with ``ast`` instead of imported, so checking the taxonomy never
    needs mediapipe, and so a feature name only counts if the function really
    mentions it.
    """
    try:
        with open(os.path.join(_PKG, "evaluator.py"), encoding="utf-8") as fh:
            module = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_features":
            return {n.value for n in ast.walk(node)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return None


def _landmark_constants() -> dict[str, int] | None:
    """Module-level integer constants of yoga/landmarks.py, parsed not imported."""
    try:
        with open(os.path.join(_PKG, "landmarks.py"), encoding="utf-8") as fh:
            module = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return None
    out: dict[str, int] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant) \
                and isinstance(value.value, int):
            out[target.id] = value.value
        elif isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            for t, v in zip(target.elts, value.elts):
                if isinstance(t, ast.Name) and isinstance(v, ast.Constant) \
                        and isinstance(v.value, int):
                    out[t.id] = v.value
    return out


def validate(categories: dict[str, Category] | None = None) -> list[str]:
    """Problems with the model; an empty list means it is consistent.

    ``categories`` defaults to the real ``CATEGORIES`` and exists so the tests
    can feed in a deliberately broken copy and prove each check fires.
    """
    cats = CATEGORIES if categories is None else categories
    problems: list[str] = []

    # --- levels 1-3: every pose in exactly one category, one position
    seen: dict[str, list[str]] = {}
    for c in cats.values():
        if c.position not in POSITIONS:
            problems.append(f"{c.key}: unknown position '{c.position}'")
        if not c.purpose.strip():
            problems.append(f"{c.key}: no purpose sentence")
        for pk in c.poses:
            seen.setdefault(pk, []).append(c.key)
            if pk not in routines.POSES:
                problems.append(f"{c.key}: unknown pose key '{pk}'")
    for pk in routines.POSES:
        homes = seen.get(pk, [])
        if not homes:
            problems.append(f"pose '{pk}' is in no category")
        elif len(homes) > 1:
            problems.append(f"pose '{pk}' is in {len(homes)} categories: {', '.join(homes)}")
    for pos in POSITIONS:
        if not any(c.position == pos for c in cats.values()):
            problems.append(f"position '{pos}' has no category")

    # --- agree with routines.py, which the rest of the app already uses
    for pk, pose in routines.POSES.items():
        homes = seen.get(pk, [])
        if len(homes) != 1:
            continue
        c = cats[homes[0]]
        if c.position not in _COARSE_TO_FINE.get(pose.position, set()):
            problems.append(f"pose '{pk}' is '{pose.position}' in routines.py but "
                            f"'{c.position}' here")
        for tag in pose.tags:
            want = _TAG_TO_CATEGORY.get(tag)
            if want and want in cats and c.key != want:
                problems.append(f"pose '{pk}' is tagged '{tag}' but sits in "
                                f"'{c.key}', not '{want}'")

    # --- level 4: landmarks
    for c in cats.values():
        for i in c.points:
            if not isinstance(i, int) or not 0 <= i <= 32:
                problems.append(f"{c.key}: landmark id {i!r} outside 0..32")
            elif i not in LANDMARK_NAMES:
                problems.append(f"{c.key}: landmark id {i} has no name")
    real = _landmark_constants()
    if real is None:
        problems.append("cannot read yoga/landmarks.py to cross-check landmark ids")
    else:
        for name, val in LANDMARK_CONSTANTS.items():
            if real.get(name) != val:
                problems.append(f"landmark {name}: taxonomy says {val}, "
                                f"landmarks.py says {real.get(name)}")

    # --- level 4: features
    names = _compute_features_names()
    if names is None:
        problems.append("cannot read compute_features in yoga/evaluator.py")
    for c in cats.values():
        for f in c.features:
            if names is not None and f not in names:
                problems.append(f"{c.key}: feature '{f}' is not computed by compute_features")
            feat = FEATURES.get(f)
            if feat is None:
                problems.append(f"{c.key}: feature '{f}' has no entry in FEATURES")
                continue
            missing = [i for i in feat.points if i not in c.points]
            if missing:
                problems.append(
                    f"{c.key}: feature '{f}' needs landmarks "
                    f"{[LANDMARK_NAMES.get(i, i) for i in missing]} that are not listed")
    for f, feat in FEATURES.items():
        if names is not None and f not in names:
            problems.append(f"FEATURES['{f}'] is not computed by compute_features")

    # --- whatever the app scores today must be a feature compute_features produces
    # (an asana check on a name that does not exist would silently never score)
    if names is not None:
        for pk in routines.POSES:
            bad = [f for f in measured_features(pk) if f not in names]
            if bad:
                problems.append(f"pose '{pk}' is scored on {bad}, which compute_features "
                                "does not produce")
    return problems
