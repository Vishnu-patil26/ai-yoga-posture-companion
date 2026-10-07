"""What makes each pose *that pose*: its structure, its cues, its guided steps.

Vrikshasana is tuned by hand (yoga/asanas.py).  Every other pose is fitted from
photographs, but a fit alone only says "be like the typical photo".  This file
adds the part a yoga teacher supplies, per pose:

* ``FEATURES``  - which body measurements define the pose, with a weight
  multiplier for the ones that matter most (a Warrior is judged on its front
  knee and stance width, a Child's pose on how far the hips fold).  Targets and
  tolerances are NOT here: they are fitted from data by tools/build_library.py.
* ``CUES``      - what to say when a measurement is below / above its target,
  worded for that pose ("sit lower", not "adjust your knee").
* ``STEPS``     - the guided walk-in: each step is an instruction plus the
  measurements that must be right before it counts as done (yoga/guide.py).

Measurement conventions (see evaluator.compute_features): joint angles are
interior angles, so smaller = more bent; arm angles are measured from straight
up, so smaller = more raised; "min/max" sorts a left/right pair so either side
works.  Below target is the first cue, above target is the second.
"""

from __future__ import annotations

#: pose key -> {feature: weight multiplier}
FEATURES: dict[str, dict[str, float]] = {
    "tadasana": {"spine_tilt": 1.4, "neck_dev": 1.0, "knee_straight": 1.0, "knee_bent": 1.0,
                 "stance_width": 1.2, "arm_raise_high": 1.0, "arm_raise_low": 1.0,
                 "hip_angle_min": 0.8},
    "trikonasana": {"stance_width": 1.3, "knee_straight": 1.3, "knee_bent": 1.2,
                    "spine_tilt": 1.4, "arm_raise_high": 1.2, "arm_raise_low": 1.0,
                    "elbow_straight": 1.0, "hip_angle_min": 1.0, "neck_dev": 0.8},
    "virabhadrasana": {"knee_bent": 1.5, "knee_straight": 1.4, "stance_width": 1.3,
                       "arm_raise_high": 1.1, "arm_raise_low": 1.1, "elbow_straight": 0.8,
                       "spine_tilt": 1.2, "hip_angle_min": 0.8, "neck_dev": 0.8},
    "utkatasana": {"knee_bent": 1.3, "knee_straight": 1.3, "hip_angle_min": 1.2,
                   "hip_angle_max": 1.0, "thigh_level": 1.2, "spine_tilt": 1.0,
                   "arm_raise_high": 1.0, "arm_raise_low": 1.0, "stance_width": 0.9,
                   "neck_dev": 0.8},
    # side-view poses: the camera sees one arm and one leg, so they use the *_vis
    # (visible side) measurements, not the sorted left/right pairs
    "adho_mukha": {"hip_rise": 1.6, "knee_angle_vis": 1.3, "elbow_angle_vis": 1.2,
                   "arm_torso_vis": 1.2, "spine_tilt": 1.0, "hip_angle_vis": 1.3,
                   "head_drop": 0.8},
    "bhujangasana": {"spine_tilt": 1.4, "hip_rise": 1.5, "hip_angle_vis": 1.3, "elbow_angle_vis": 1.1,
                     "arm_torso_vis": 1.0, "knee_angle_vis": 0.8, "neck_dev": 0.7,
                     "head_drop": 0.6},
    "marjaryasana": {"knee_angle_vis": 1.2, "arm_torso_vis": 1.2, "elbow_angle_vis": 1.2,
                     "hip_angle_vis": 1.3, "head_drop": 1.3, "spine_tilt": 1.0,
                     "thigh_level": 1.0},
    "balasana": {"hip_angle_vis": 1.5, "knee_angle_vis": 1.3, "spine_tilt": 1.2, "head_drop": 1.2,
                 "arm_torso_vis": 0.8, "thigh_level": 0.8, "neck_dev": 0.5},
    "sukhasana": {"spine_tilt": 1.5, "neck_dev": 1.0, "hip_angle_vis": 1.2,
                  "knee_angle_vis": 1.2, "thigh_level": 0.8, "head_drop": 0.6},
}

#: pose key -> {feature: (cue when below target, cue when above target)}
CUES: dict[str, dict[str, tuple[str, str]]] = {
    "tadasana": {
        "spine_tilt": ("", "Stand tall - stack your shoulders over your hips"),
        "knee_straight": ("Straighten your knees - lift your kneecaps", ""),
        "knee_bent": ("Straighten your knees - lift your kneecaps", ""),
        "stance_width": ("", "Bring your feet together"),
    },
    "trikonasana": {
        "stance_width": ("Step your feet wider apart", "Bring your feet a little closer"),
        "knee_straight": ("Keep both legs straight", ""),
        "knee_bent": ("Keep both legs straight", ""),
        "spine_tilt": ("Reach further out and bend sideways over your front leg",
                       "Lift your torso a little - don't collapse forward"),
        "arm_raise_high": ("", "Reach your top arm straight up to the ceiling"),
        "elbow_straight": ("Straighten your arms", ""),
    },
    "virabhadrasana": {
        "knee_bent": ("Ease your front knee up - not past a right angle",
                      "Bend your front knee more - aim for a right angle"),
        "knee_straight": ("Straighten your back leg", ""),
        "stance_width": ("Step your feet wider apart", "Bring your feet a little closer"),
        "arm_raise_high": ("Lower your arms to shoulder height", "Lift your arms to shoulder height"),
        "arm_raise_low": ("Lower your arms to shoulder height", "Lift your arms to shoulder height"),
        "spine_tilt": ("", "Stay upright - don't lean over your front knee"),
        "elbow_straight": ("Straighten your arms out to the sides", ""),
    },
    "utkatasana": {
        "knee_bent": ("Rise a little - thighs about level with the floor",
                      "Sit lower - bend your knees as if into a chair"),
        "knee_straight": ("Rise a little - thighs about level with the floor",
                          "Sit lower - bend your knees as if into a chair"),
        "thigh_level": ("", "Sit back and lower your hips until your thighs are nearly level"),
        "hip_angle_min": ("Lift your chest - open your hips a little",
                          "Sit your hips back and fold slightly forward"),
        "spine_tilt": ("", "Lift your chest - keep your spine long"),
        "arm_raise_high": ("", "Reach your arms up beside your ears"),
        "arm_raise_low": ("", "Reach your arms up beside your ears"),
        "stance_width": ("", "Bring your feet together"),
    },
    "adho_mukha": {
        "hip_rise": ("Lift your hips higher toward the ceiling", ""),
        "knee_angle_vis": ("Straighten your legs as far as is comfortable", ""),
        "elbow_angle_vis": ("Straighten your arms - press the floor away", ""),
        "arm_torso_vis": ("Press your chest back toward your thighs", ""),
        "hip_angle_vis": ("", "Fold at the hips - send your hips up and back"),
        "head_drop": ("", "Relax your head down between your arms"),
    },
    "bhujangasana": {
        "spine_tilt": ("Lower your chest a little - keep your hips down",
                       "Lift your chest a little higher"),
        "hip_rise": ("Ease your chest down a little", "Press your hips down into the floor"),
        "hip_angle_vis": ("Press your hips and thighs into the floor", ""),
        "elbow_angle_vis": ("Press the floor away to lift your chest",
                            "Keep a soft bend in your elbows and tuck them in"),
        "knee_angle_vis": ("Keep your legs long and together", ""),
    },
    "marjaryasana": {
        "knee_angle_vis": ("Keep your knees under your hips", "Keep your knees under your hips"),
        "elbow_angle_vis": ("Keep your arms straight, wrists under your shoulders", ""),
        "hip_angle_vis": ("Stack your hips over your knees", "Stack your hips over your knees"),
        "head_drop": ("Drop your head a little more", "Lift your head a little"),
        "arm_torso_vis": ("Wrists under your shoulders", "Wrists under your shoulders"),
    },
    "balasana": {
        "hip_angle_vis": ("", "Sit your hips back toward your heels and fold forward"),
        "knee_angle_vis": ("", "Sit back onto your heels"),
        "head_drop": ("Rest your forehead down", ""),
        "spine_tilt": ("", "Fold forward over your thighs"),
        "arm_torso_vis": ("Let your arms rest long", ""),
    },
    "sukhasana": {
        "spine_tilt": ("", "Sit tall - lift the crown of your head"),
        "hip_angle_vis": ("", "Sit on a folded blanket if your hips are tight, and lean forward from the hips"),
        "knee_angle_vis": ("", "Let your knees open and your shins cross"),
        "thigh_level": ("", "Let your knees drop toward the floor"),
    },
}

_FRONT = "Stand facing the camera, full body in frame, about 2 m away."
_SIDE = "Place the camera low and to your side, full body in frame, about 2 m away."

#: where to put the camera: front-on for standing/seated poses, side-on for floor poses
SETUP_HINTS: dict[str, str] = {
    "tadasana": _FRONT, "trikonasana": _FRONT, "virabhadrasana": _FRONT,
    "utkatasana": "Stand side-on or at an angle to the camera, full body in frame, about 2 m away.",
    "sukhasana": "Sit facing the camera, full body in frame, about 2 m away.",
    "adho_mukha": _SIDE, "bhujangasana": _SIDE, "marjaryasana": _SIDE, "balasana": _SIDE,
}

#: pose key -> list of (spoken / shown instruction, measurements that must be right)
STEPS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "tadasana": [
        ("Stand with your feet together and your weight even.", ("stance_width", "knee_straight", "knee_bent")),
        ("Lengthen your spine - shoulders over hips, chin level.", ("spine_tilt", "neck_dev")),
        ("Arms relaxed at your sides, or reach them overhead.", ("arm_raise_high", "arm_raise_low")),
    ],
    "trikonasana": [
        ("Step your feet wide apart.", ("stance_width",)),
        ("Keep both legs straight.", ("knee_straight", "knee_bent")),
        ("Reach one hand toward your leg, the other up to the ceiling.",
         ("spine_tilt", "arm_raise_high", "elbow_straight")),
    ],
    "virabhadrasana": [
        ("Step your feet wide apart.", ("stance_width",)),
        ("Bend your front knee to a right angle, keep the back leg straight.", ("knee_bent", "knee_straight")),
        ("Arms out to shoulder height, torso upright.", ("arm_raise_high", "arm_raise_low", "spine_tilt")),
    ],
    "utkatasana": [
        ("Bring your feet together.", ("stance_width",)),
        ("Bend your knees and sit back as if into a chair.",
         ("knee_bent", "knee_straight", "thigh_level", "hip_angle_min")),
        ("Lift your arms overhead and keep your spine long.", ("arm_raise_high", "arm_raise_low", "spine_tilt")),
    ],
    "adho_mukha": [
        ("From hands and knees, lift your hips up and back.", ("hip_rise", "hip_angle_vis")),
        ("Straighten your legs and press the floor away with straight arms.",
         ("knee_angle_vis", "elbow_angle_vis")),
        ("Relax your head between your arms.", ("head_drop", "arm_torso_vis")),
    ],
    "bhujangasana": [
        ("Lie face down with your hands under your shoulders.", ("hip_angle_vis",)),
        ("Lift your chest, keep your hips on the floor.", ("spine_tilt", "hip_rise", "hip_angle_vis")),
        ("Keep a soft bend in your elbows and your legs long.", ("elbow_angle_vis", "knee_angle_vis")),
    ],
    "marjaryasana": [
        ("Come onto hands and knees, wrists under shoulders, knees under hips.",
         ("arm_torso_vis", "knee_angle_vis", "hip_angle_vis")),
        ("Breathe in, drop your belly and lift your head; breathe out, round your back.", ("head_drop",)),
        ("Keep your arms straight and flow slowly.", ("elbow_angle_vis",)),
    ],
    "balasana": [
        ("Kneel and sit back onto your heels.", ("knee_angle_vis", "thigh_level")),
        ("Fold forward over your thighs.", ("hip_angle_vis", "spine_tilt")),
        ("Rest your forehead down, arms long in front of you or alongside you.", ("head_drop",)),
    ],
    "sukhasana": [
        ("Sit cross-legged, on a folded blanket if your hips are tight.",
         ("knee_angle_vis", "hip_angle_vis", "thigh_level")),
        ("Sit tall and lift the crown of your head.", ("spine_tilt", "neck_dev")),
        ("Rest your hands on your knees, shoulders relaxed.", ("spine_tilt", "knee_angle_vis")),
    ],
}
