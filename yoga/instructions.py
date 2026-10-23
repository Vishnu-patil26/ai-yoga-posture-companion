"""Plain, numbered how-to instructions for every pose.

This is what a person reads (and can hear) before and while doing a pose: where to
put the camera, the steps in the order to do them, how to breathe, the mistakes
people usually make and how to come out.  It is teaching text only.  Nothing here
touches the scorer, and nothing is invented about the camera: the measured walk-in
that the camera follows lives in ``yoga/posespecs.py`` (STEPS) and says the same
things in shorter form.

``demo`` holds four captions that match what the stick-figure demo is doing at each
moment: the start position, moving into the pose, holding it, and coming out.

The guidance is general wellness guidance, not medical advice.  Poses that are not
suitable for everyone say so in ``careful``; the profile rules in ``yoga/profile.py``
also skip head-below-heart poses for people they apply to.
"""

from __future__ import annotations

from dataclasses import dataclass

_FRONT = "Stand facing the camera, whole body in the picture, about 2 metres away."
_SIDE = "Put the camera low and to your side so it sees your whole body, about 2 metres away."


@dataclass(frozen=True)
class HowTo:
    camera: str                      #: where to put yourself and the camera
    steps: tuple[str, ...]           #: the numbered steps, in order
    breath: str                      #: breathing and how long to stay
    mistakes: tuple[str, ...]        #: "what goes wrong: how to fix it"
    release: str                     #: how to come out
    demo: tuple[str, str, str, str]  #: captions: start, moving in, holding, coming out
    careful: str = ""                #: who should take care or skip


HOWTO: dict[str, HowTo] = {
    "tadasana": HowTo(
        camera=_FRONT,
        steps=(
            "Stand tall with your feet together, big toes touching. If that feels wobbly, put them hip-width apart.",
            "Spread your weight evenly over both feet: the ball of the big toe, the ball of the little toe and the heel.",
            "Straighten your legs without locking the knees. Lift your kneecaps a little.",
            "Lengthen your spine. Imagine a string pulling the crown of your head up. Keep your hips level.",
            "Let your shoulders drop away from your ears and open your chest. Keep your chin level.",
            "Let your arms hang by your sides with the palms facing forward. For the arms-up version, reach them straight overhead with the palms facing each other.",
            "Look straight ahead and breathe slowly through your nose.",
        ),
        breath="Breathe in for about 4 seconds and out for about 4 seconds. Stay for 10 seconds.",
        mistakes=(
            "Leaning back and arching the lower back: tuck your tailbone slightly and lengthen your spine.",
            "Locking the knees: keep a tiny soft bend.",
            "Weight on the heels or the toes only: feel it spread through the whole foot.",
            "Shoulders hunched or chin pushed up: relax the shoulders and keep the chin level.",
        ),
        release="Lower your arms slowly and shake out your hands.",
        demo=("Start standing: feet together, arms by your sides.",
              "Breathe in and sweep your arms out and up.",
              "Arms overhead, palms facing, spine long. Breathe slowly.",
              "Breathe out and lower your arms back down."),
    ),
    "utkatasana": HowTo(
        camera="Stand side-on or at an angle to the camera so it can see your knees and hips bend. Whole body in the picture, about 2 metres away.",
        steps=(
            "Stand with your feet together, or hip-width apart if you need more balance. Arms by your sides.",
            "Bend your knees and send your hips back, as if you were sitting down onto a chair behind you.",
            "Keep your weight in your heels. Your knees stay over your ankles and do not push past your toes.",
            "Sink until your thighs are close to level with the floor. A smaller squat is fine if that is too much.",
            "Lift your chest and keep your back long. A small lean forward is normal, but do not round your back.",
            "Raise your arms overhead beside your ears with the palms facing each other. Keep your shoulders down.",
            "Look forward and breathe slowly and steadily.",
        ),
        breath="Breathe slowly. Stay for 10 seconds. If your thighs shake a lot, rise a little and keep going.",
        mistakes=(
            "Knees falling inward: press them gently out so they line up over your second toe.",
            "Heels lifting off the floor: shift your weight back into the heels.",
            "Lower back arching hard: draw your belly in and tuck your tailbone slightly.",
            "Upper back rounding: lift your chest and reach your arms up and back.",
        ),
        release="Breathe in, press into your feet, straighten your legs and lower your arms.",
        demo=("Start standing: feet together, arms by your sides.",
              "Bend your knees and sit your hips back like sitting on a chair.",
              "Thighs near level, chest lifted, arms up by your ears. Breathe.",
              "Press into your feet and stand back up."),
        careful="Skip or keep the squat shallow if your knees hurt.",
    ),
    "virabhadrasana": HowTo(
        camera=_FRONT,
        steps=(
            "Stand tall, then step your feet wide apart, about one leg length.",
            "Turn your front foot out 90 degrees so the toes point to the side. Turn your back foot in a little.",
            "Bend your front knee until it is above your ankle. Do not let it go past your toes.",
            "Keep your back leg straight and press the outer edge of the back foot into the floor.",
            "Keep your torso upright, with your shoulders over your hips. Do not lean over the front leg.",
            "Stretch both arms out to the sides at shoulder height, palms facing down, one over each leg.",
            "Turn your head and look over your front hand. Breathe slowly.",
        ),
        breath="Breathe slowly. Stay for 10 seconds, then repeat with the other leg in front.",
        mistakes=(
            "Front knee falling inward: push it out over the little toe side of the foot.",
            "Leaning over the front thigh: stack your shoulders over your hips.",
            "Arms drooping: reach long through the fingertips at shoulder height.",
            "Stance too narrow: step wider so you can bend the knee to a right angle.",
        ),
        release="Straighten the front leg, lower your arms and bring your feet together.",
        demo=("Start standing: feet together, arms by your sides.",
              "Step your feet wide, turn the front foot out and bend that knee.",
              "Front knee bent, back leg straight, arms out at shoulder height.",
              "Straighten your front leg and bring your feet together."),
        careful="Keep the bend shallower if your knees or hips are tender.",
    ),
    "trikonasana": HowTo(
        camera=_FRONT,
        steps=(
            "Stand with your feet wide apart, about one leg length. Reach your arms out to the sides at shoulder height.",
            "Turn your right foot out 90 degrees and your left foot in a little. Keep both legs straight, knees not locked.",
            "Reach your right arm out over your right leg. Bend sideways from the hip, not the waist, with a long spine.",
            "Rest your right hand on your shin or ankle, or on a block. Do not press on the knee joint.",
            "Stretch your left arm straight up so that your two arms make one vertical line.",
            "Keep your chest open and turn your gaze up to your top hand, if your neck is comfortable.",
            "Breathe slowly. Then come up and repeat on the other side.",
        ),
        breath="Breathe slowly. Stay for 10 seconds on each side.",
        mistakes=(
            "Folding forward instead of sideways: imagine your body between two panes of glass.",
            "Locking the front knee: keep it straight but soft.",
            "Hand pressing on the knee: move it lower, or use a block.",
            "Chest collapsing toward the floor: turn it up and open it.",
        ),
        release="Press into your back foot and rise up with a long spine, then lower your arms.",
        demo=("Start standing: feet together, arms by your sides.",
              "Step your feet wide, reach one hand toward your leg and the other up.",
              "Legs straight, bend sideways, top arm straight up. Breathe.",
              "Come back up and bring your feet together."),
        careful="Keep the bend small and use a block if your back or hamstrings feel tight.",
    ),
    "sukhasana": HowTo(
        camera="Sit facing the camera, whole body in the picture, about 2 metres away.",
        steps=(
            "Sit on the floor. If your hips or knees feel tight, sit on a folded blanket so your hips are higher than your knees.",
            "Cross your shins and let your knees fall open and down. Do not force them.",
            "Rest your feet under the opposite knees, relaxed.",
            "Sit on your sit bones and lift the crown of your head so your spine is long.",
            "Rest your hands on your knees. Let your shoulders drop and your chest open.",
            "Keep your chin level, relax your face and jaw, and close your eyes if you like.",
            "Breathe slowly through your nose.",
        ),
        breath="Breathe in and out slowly through the nose. Stay for 10 seconds. Switch which shin is in front next time.",
        mistakes=(
            "Back rounding and slumping: sit higher on a blanket and lengthen your spine.",
            "Knees much higher than hips: add height under your hips.",
            "Forcing the knees down: let them hang loose, they will soften with time.",
            "Tense shoulders: let them drop and rest your hands lightly.",
        ),
        release="Uncross your legs slowly and stretch them out in front of you.",
        demo=("Sit cross-legged, on a folded blanket if your hips are tight.",
              "Let your knees fall open and down. Lift the crown of your head.",
              "Spine long, hands on your knees, shoulders relaxed. Breathe slowly.",
              "Relax your face and jaw. Close your eyes if you like."),
        careful="Use a cushion if your knees hurt, and skip it if crossing the legs is painful.",
    ),
    "marjaryasana": HowTo(
        camera=_SIDE,
        steps=(
            "Come onto your hands and knees. Put your wrists under your shoulders and your knees under your hips.",
            "Spread your fingers wide and press your whole hand into the floor. Keep your arms straight with soft elbows.",
            "Start with a flat back and a long neck, looking at the floor between your hands.",
            "Cow: breathe in, let your belly sink toward the floor, lift your chest and tailbone and look slightly forward.",
            "Cat: breathe out, press into your hands, round your back up toward the ceiling and tuck your chin and tailbone.",
            "Move slowly between cow and cat, one breath for each movement.",
            "Keep your wrists under your shoulders and your knees under your hips all the way through.",
        ),
        breath="Breathe in for cow and out for cat. Keep it slow and smooth for 10 seconds, then rest on a flat back.",
        mistakes=(
            "Moving too fast: match each movement to one full breath.",
            "Hands too far forward or knees too far back: stack wrists under shoulders and knees under hips.",
            "Elbows locking hard: keep them soft.",
            "Neck cranked back in cow: look slightly forward, keep the neck long.",
        ),
        release="Return to a flat back, sit back onto your heels and rest.",
        demo=("Start on hands and knees with a flat back.",
              "Breathe in: belly drops, chest and tailbone lift (cow).",
              "Breathe out: round your back up and tuck your chin (cat).",
              "Move slowly back and forth with your breath."),
        careful="Move gently if your wrists or knees are sore, and put padding under your knees.",
    ),
    "balasana": HowTo(
        camera=_SIDE,
        steps=(
            "Kneel on a mat with your big toes touching and your knees about hip-width apart, or wider if that is more comfortable.",
            "Sit back so your hips rest toward your heels. Put a cushion between your hips and heels if they do not reach.",
            "Breathe out and fold your body forward over your thighs.",
            "Stretch your arms forward on the floor with the palms down, or rest them alongside your body with the palms up.",
            "Rest your forehead on the floor or on a cushion. Relax your neck and shoulders.",
            "Let your back widen as you breathe. Send your breath into your lower back.",
        ),
        breath="Breathe slowly and deeply. Stay for 10 seconds, or longer if it feels good.",
        mistakes=(
            "Hips lifting off the heels: put a folded blanket or cushion between them.",
            "Forcing the forehead to the floor: rest it on a cushion or on your stacked hands.",
            "Shoulders tight near the ears: let your arms be heavy.",
            "Pain in the knees: open the knees wider and pad under them.",
        ),
        release="Walk your hands in, press into them and roll up slowly, with your head last.",
        demo=("Start kneeling and sitting upright on your heels.",
              "Breathe out and fold forward over your thighs.",
              "Forehead down, arms long, back soft. Breathe.",
              "Walk your hands in and roll up slowly."),
        careful="Skip it or use lots of padding if you have a knee injury. Be careful in late pregnancy.",
    ),
    "adho_mukha": HowTo(
        camera=_SIDE,
        steps=(
            "Start on your hands and knees. Put your wrists under your shoulders and your knees under your hips. Spread your fingers wide.",
            "Tuck your toes under. Breathe out and lift your knees off the floor.",
            "Send your hips up and back toward the ceiling so your body makes an upside-down V.",
            "Press the floor away with your hands and straighten your arms. Keep your elbows soft, not locked.",
            "Straighten your legs as far as is comfortable. Bend the knees a little if your back rounds. Your heels do not have to touch the floor.",
            "Lengthen your spine and press your chest back toward your thighs. Keep your head relaxed between your upper arms.",
            "Breathe slowly and keep pressing the floor away.",
        ),
        breath="Breathe slowly and evenly. Stay for 10 seconds.",
        mistakes=(
            "Lower back rounding: bend your knees and lift your tailbone toward the ceiling.",
            "Shoulders creeping up toward the ears: press the floor away and move them down your back.",
            "Weight dumped onto the hands or the heels: spread it evenly and keep the hips high.",
            "Hands too close to your feet: walk them forward so the stance is long.",
        ),
        release="Bend your knees and lower them to the floor, then rest in child's pose.",
        demo=("Start on hands and knees, wrists under shoulders.",
              "Tuck your toes and lift your hips up and back.",
              "Upside-down V: arms and legs straight, head relaxed. Breathe.",
              "Bend your knees and lower back down."),
        careful="This head-below-heart pose is skipped for high blood pressure, age 60 or more, glaucoma and some medicines. Skip it if your wrists or shoulders hurt.",
    ),
    "bhujangasana": HowTo(
        camera=_SIDE,
        steps=(
            "Lie face down on a mat with your legs straight and together. The tops of your feet rest on the floor.",
            "Place your hands flat on the floor under your shoulders, with your elbows close to your body.",
            "Press the tops of your feet, your thighs and your hips gently into the floor.",
            "Breathe in and lift your head and chest, using your back muscles first. Do not push up with your arms alone.",
            "Keep a soft bend in your elbows and tuck them in. Draw your shoulders away from your ears.",
            "Look forward with a long neck. Keep your hips and thighs on the floor.",
            "Breathe slowly and stay low enough that your lower back feels comfortable.",
        ),
        breath="Breathe slowly. Stay for 10 seconds, then lower down with a long breath out.",
        mistakes=(
            "Shoulders hunched up near the ears: slide them down your back.",
            "Lifting too high, with pinching in the lower back: lower down a little.",
            "Arms locked straight: keep a soft bend and lift with the back.",
            "Legs falling apart or rolling out: keep them long and together.",
        ),
        release="Breathe out and lower your chest, then turn your head to one side and rest.",
        demo=("Start lying face down, hands under your shoulders.",
              "Breathe in and lift your head and chest off the floor.",
              "Chest lifted, hips on the floor, elbows soft. Breathe.",
              "Breathe out and lower down slowly."),
        careful="Skip this pose in pregnancy or with a recent back or abdominal injury or surgery. Stay low and gentle if your lower back is sensitive.",
    ),
    "vrikshasana": HowTo(
        camera=_FRONT,
        steps=(
            "Stand tall with your feet together and your arms by your sides, facing the camera.",
            "Shift your weight onto one foot. Keep that leg straight but not locked.",
            "Bend the other knee and lift that foot.",
            "Place the sole of the lifted foot on the inside of your standing thigh, or on your calf. Never press it on the knee.",
            "Open the lifted knee out to the side. Keep your hips level and facing forward.",
            "Bring your palms together at your chest.",
            "When you feel steady, reach your arms overhead. Look at one still spot ahead of you and breathe slowly.",
        ),
        breath="Breathe slowly. Stay for 20 seconds, then switch to the other leg.",
        mistakes=(
            "Foot pressed on the knee: move it above or below the knee joint.",
            "Hip pushed out to the side: tuck the tailbone slightly and keep the hips level.",
            "Standing knee locked: keep a tiny soft bend.",
            "Looking down or around: pick one still spot at eye level.",
        ),
        release="Lower your arms, then your foot, and stand tall for a few breaths.",
        demo=("Start standing: feet together, arms by your sides.",
              "Shift your weight, lift one foot to your inner thigh.",
              "Foot on the thigh, hands at the heart or overhead. Balance and breathe.",
              "Lower your foot and arms and stand tall."),
        careful="Keep a wall or chair within reach if your balance is unsteady.",
    ),
}


def sequence(key: str) -> str:
    """One line for the get-ready screen: what the camera window will take you through."""
    from . import posespecs
    if key not in posespecs.VIEW:
        return ""
    turn = "turn sideways" if posespecs.VIEW[key] == "side" else "face the camera"
    return (f"In the camera window you are taken through it in order: stand straight, {turn}, get into the "
            "pose one step at a time, hold for 10 seconds, an angle check of your arms, legs, spine and neck, "
            "then release and breathe.")


def howto(key: str) -> HowTo | None:
    return HOWTO.get(key)


def demo_caption(key: str, phase: int) -> str:
    """The caption for one phase (0 start, 1 moving in, 2 holding, 3 coming out) of a demo."""
    h = HOWTO.get(key)
    if h is None:
        return ""
    return h.demo[max(0, min(len(h.demo) - 1, phase))]


def spoken(key: str) -> str:
    """Everything the 'Play audio' button reads: the camera set-up, each numbered step, the breath."""
    h = HOWTO.get(key)
    if h is None:
        return ""
    steps = " ".join(f"Step {i}. {s}" for i, s in enumerate(h.steps, 1))
    return f"{h.camera} {steps} {h.breath} To come out: {h.release}"
