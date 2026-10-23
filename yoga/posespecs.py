"""What makes each pose *that pose*: its structure, its cues, its guided steps.

Vrikshasana is tuned by hand (yoga/asanas.py).  Every other pose is fitted from
photographs, but a fit alone only says "be like the typical photo".  This file
adds the part a yoga teacher supplies, per pose:

* ``FEATURES``  - which body measurements define the pose, with a weight
  multiplier for the ones that matter most (a Warrior is judged on its front
  knee and stance width, a Child's pose on how far the hips fold).  Targets and
  tolerances are NOT here: they are fitted from data by tools/fitting/build_library.py.
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
#: which way to stand to the camera for each pose: "side" shows the spine and knee angles best,
#: "front" is needed when both arms or both legs must be seen (a wide stance, a side bend)
VIEW: dict[str, str] = {
    "tadasana": "front", "trikonasana": "front", "virabhadrasana": "front", "sukhasana": "front",
    "utkatasana": "side", "adho_mukha": "side", "bhujangasana": "side", "marjaryasana": "side",
    "balasana": "side",
}

STEPS: dict[str, list[tuple[str, tuple[str, ...], str]]] = {
    # (what is said and shown at the bottom, the measurements that must hold, the short label on the card)
    "tadasana": [
        ("Stand with your feet together and your weight even on both feet.",
         ("stance_width", "knee_straight", "knee_bent"), "Feet together, weight even"),
        ("Straighten your legs, but do not lock your knees. Lift your kneecaps a little.",
         ("knee_straight", "knee_bent"), "Legs straight, knees soft"),
        ("Lengthen your spine. Stack your shoulders over your hips and keep your chin level.",
         ("spine_tilt", "neck_dev"), "Spine tall, chin level"),
        ("Drop your shoulders away from your ears. Let your arms hang, or reach them straight overhead.",
         ("arm_raise_high", "arm_raise_low"), "Shoulders down, arms ready"),
        ("Look straight ahead and breathe slowly through your nose.", (), "Look ahead, breathe slowly"),
    ],
    "trikonasana": [
        ("Step your feet wide apart, about one leg length. Turn your right foot out.",
         ("stance_width",), "Feet wide apart"),
        ("Keep both legs straight, but do not lock your knees.",
         ("knee_straight", "knee_bent"), "Both legs straight"),
        ("Reach out over your right leg and bend sideways from your hip. Keep your chest open.",
         ("spine_tilt",), "Bend sideways from the hip"),
        ("Stretch your top arm straight up to the ceiling, in line with your lower arm.",
         ("arm_raise_high", "elbow_straight"), "Top arm straight up"),
        ("Look up at your top hand if your neck is comfortable. Breathe slowly.", (), "Look up, breathe slowly"),
    ],
    "virabhadrasana": [
        ("Step your feet wide apart, about one leg length. Turn your front foot out.",
         ("stance_width",), "Feet wide apart"),
        ("Bend your front knee until it is above your ankle. Do not let it pass your toes.",
         ("knee_bent",), "Bend the front knee"),
        ("Keep your back leg straight and press its outer edge into the floor.",
         ("knee_straight",), "Back leg straight"),
        ("Stand tall through your torso, with your shoulders over your hips.",
         ("spine_tilt",), "Torso upright"),
        ("Stretch your arms out to the sides at shoulder height, palms down. Look over your front hand.",
         ("arm_raise_high", "arm_raise_low"), "Arms out, shoulder height"),
    ],
    "utkatasana": [
        ("Stand with your feet together and your arms by your sides.", ("stance_width",), "Feet together"),
        ("Bend your knees and send your hips back, as if sitting down onto a chair. Keep your weight in your heels.",
         ("knee_bent", "knee_straight"), "Bend knees, hips back"),
        ("Sink until your thighs are close to level with the floor. Keep your knees over your ankles.",
         ("thigh_level", "hip_angle_min"), "Thighs nearly level"),
        ("Lift your chest and keep your back long. Do not round your back.",
         ("spine_tilt",), "Chest up, back long"),
        ("Raise your arms overhead beside your ears, palms facing each other, shoulders down.",
         ("arm_raise_high", "arm_raise_low"), "Arms up by your ears"),
    ],
    "adho_mukha": [
        ("From your hands and knees, tuck your toes and lift your hips up and back.",
         ("hip_rise", "hip_angle_vis"), "Hips up and back"),
        ("Press the floor away and straighten your arms.", ("elbow_angle_vis",), "Arms straight"),
        ("Straighten your legs as far as is comfortable. Bend your knees a little if your back rounds.",
         ("knee_angle_vis",), "Legs straight"),
        ("Press your chest back toward your thighs and lengthen your spine.",
         ("arm_torso_vis",), "Chest toward thighs"),
        ("Let your head hang relaxed between your arms.", ("head_drop",), "Head relaxed"),
    ],
    "bhujangasana": [
        ("Lie face down with your legs together and your hands flat under your shoulders.",
         ("hip_angle_vis",), "Face down, hands by shoulders"),
        ("Breathe in and lift your head and chest, keeping your hips on the floor.",
         ("spine_tilt", "hip_rise", "hip_angle_vis"), "Lift head and chest"),
        ("Press the tops of your feet and your hips into the floor.",
         ("hip_rise", "hip_angle_vis"), "Hips stay on the floor"),
        ("Keep a soft bend in your elbows and your legs long and together.",
         ("elbow_angle_vis", "knee_angle_vis"), "Elbows soft, legs long"),
        ("Draw your shoulders away from your ears and look straight ahead.", (), "Shoulders down, look ahead"),
    ],
    "marjaryasana": [
        ("Come onto your hands and knees. Put your wrists under your shoulders and your knees under your hips.",
         ("arm_torso_vis", "knee_angle_vis", "hip_angle_vis"), "Hands and knees"),
        ("Press your hands into the floor and keep your arms straight, with soft elbows.",
         ("elbow_angle_vis",), "Arms straight"),
        ("Breathe in. Let your belly sink, lift your chest and look slightly forward.",
         ("head_drop",), "Cow: belly down, chest up"),
        ("Breathe out. Round your back up toward the ceiling and tuck your chin.",
         ("head_drop",), "Cat: round your back"),
        ("Keep moving slowly between cow and cat, one breath for each movement.", (), "Flow with your breath"),
    ],
    "balasana": [
        ("Kneel and sit back toward your heels. Put a cushion under your hips if they do not reach.",
         ("knee_angle_vis", "thigh_level"), "Kneel, sit back"),
        ("Breathe out and fold your body forward over your thighs.",
         ("hip_angle_vis", "spine_tilt"), "Fold forward"),
        ("Stretch your arms forward on the floor, or rest them alongside your body.", (), "Arms long"),
        ("Rest your forehead down and relax your neck and shoulders.", ("head_drop",), "Forehead down"),
        ("Breathe slowly into your lower back.", (), "Breathe into your back"),
    ],
    "sukhasana": [
        ("Sit on the floor and cross your shins. Sit on a folded blanket if your hips or knees feel tight.",
         ("knee_angle_vis", "hip_angle_vis", "thigh_level"), "Sit cross-legged"),
        ("Let your knees fall open and down. Do not force them.", ("thigh_level",), "Knees open and low"),
        ("Sit on your sit bones and lift the crown of your head so your spine is long.",
         ("spine_tilt",), "Sit tall"),
        ("Keep your chin level and your head stacked over your spine.", ("neck_dev",), "Head over spine"),
        ("Rest your hands on your knees, drop your shoulders and breathe slowly.", (), "Hands on knees, relax"),
    ],
}
