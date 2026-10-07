"""The pose catalogue and the routines people choose from.

Two kinds of pose live here and the difference matters:

* **scored** - the asana has a reference fitted from a dataset (see
  data/asana_fits.json), so the live trainer measures the body and corrects it.
* **guided** - no dataset reference exists yet, so the launcher walks the person
  through it with spoken steps and a timer, and says plainly that it is not
  measuring them.  Nothing here invents joint angles for a pose we have no data
  for; dropping images into a class folder and running tools/fit_asana.py turns
  a guided pose into a scored one with no code change (``is_scored`` is read
  from the asana library at run time).

Routines come from the project meeting and the lifestyle-category sheet
(dataset.pdf): five lifestyle routines, three goal routines, and a "by body
position" picker built from the whole catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass

from yoga import asanas as asana_lib


@dataclass(frozen=True)
class Pose:
    key: str                  #: library key when scored, otherwise just an id
    sanskrit: str
    name: str
    position: str             #: "standing" | "sitting" | "floor"
    benefit: str
    steps: tuple[str, ...]    #: spoken / shown walk-in for guided mode
    hold_s: int = 20
    tags: frozenset = frozenset()

    @property
    def is_scored(self) -> bool:
        return self.key in asana_lib.LIBRARY


POSITIONS = {
    "standing": "Standing",
    "sitting": "Sitting",
    "floor": "Lying / floor",
}

_P = lambda *a, **k: Pose(*a, **k)                      # noqa: E731

POSES: dict[str, Pose] = {p.key: p for p in (
    _P("tadasana", "Tadasana", "Mountain Pose", "standing",
       "Resets posture and lengthens the spine.",
       ("Stand with feet hip-width apart, weight even on both feet.",
        "Lift the chest, let the shoulders drop away from the ears.",
        "Arms relaxed at your sides, chin level. Breathe slowly."), 20),
    _P("trikonasana", "Trikonasana", "Triangle Pose", "standing",
       "A long side stretch for the waist, hips and hamstrings.",
       ("Step your feet wide apart, turn the right foot out.",
        "Reach the right hand down the leg, left arm up to the ceiling.",
        "Keep both legs straight and the chest open. Then swap sides."), 20,
       frozenset({"side_bend"})),
    _P("virabhadrasana", "Virabhadrasana II", "Warrior II", "standing",
       "Builds hip and knee strength.",
       ("Step your feet wide, turn the front foot out.",
        "Bend the front knee over the ankle, arms out at shoulder height.",
        "Gaze over the front hand."), 20, frozenset({"deep_knee"})),
    _P("utkatasana", "Utkatasana", "Chair Pose", "standing",
       "Core and thigh strength; relieves a stiff lower back.",
       ("Feet together, bend the knees as if sitting back into a chair.",
        "Lift the arms overhead, keep the spine long.",
        "Weight in the heels."), 20, frozenset({"deep_knee"})),
    _P("vrikshasana", "Vrikshasana", "Tree Pose", "standing",
       "Ankle, knee and balance control.",
       ("Shift your weight onto one foot.",
        "Press the other foot into the inner thigh, never the knee.",
        "Hands at the heart or overhead."), 20, frozenset({"balance"})),
    _P("adho_mukha", "Adho Mukha Svanasana", "Downward Dog", "floor",
       "Deep stretch for calves and hamstrings.",
       ("Start on hands and knees, lift the hips up and back.",
        "Straighten the legs as far as is comfortable, press the floor away.",
        "Head relaxed between the arms."), 20, frozenset({"inversion"})),
    _P("bhujangasana", "Bhujangasana", "Cobra Pose", "floor",
       "Reverses a slouched, rounded spine.",
       ("Lie face down, hands under the shoulders.",
        "Lift the chest, keep the hips on the floor.",
        "Shoulders away from the ears."), 20, frozenset({"backbend"})),
    _P("marjaryasana", "Marjaryasana-Bitilasana", "Cat-Cow", "floor",
       "Mobilises the whole spine.",
       ("Come onto hands and knees, wrists under shoulders.",
        "Breathe in: drop the belly, lift the chest and tailbone (cow).",
        "Breathe out: round the back, tuck the chin (cat). Repeat slowly."), 30,
       frozenset({"kneeling"})),
    _P("balasana", "Balasana", "Child's Pose", "floor",
       "Releases the lower back; a resting pose.",
       ("Kneel, sit back on the heels.",
        "Fold forward, arms stretched out in front or alongside the body.",
        "Forehead down, breathe into the back."), 30,
       frozenset({"kneeling", "forward_fold"})),
    _P("sukhasana", "Sukhasana", "Easy Pose", "sitting",
       "Seated stillness for breathing and focus.",
       ("Sit cross-legged, sit tall on a folded blanket if the hips are tight.",
        "Hands on the knees, shoulders relaxed.",
        "Close the eyes and breathe slowly through the nose."), 30),
)}


@dataclass(frozen=True)
class Routine:
    key: str
    title: str
    group: str                #: "lifestyle" | "goal" | "position"
    target: str
    poses: tuple[str, ...]


LIFESTYLE = (
    Routine("desk_sitting", "Desk-bound / sitting workers", "lifestyle",
            "Relieve a stiff back and tight hips after hours in a chair.",
            ("marjaryasana", "bhujangasana", "balasana")),
    Routine("standing_workers", "Prolonged standing workers", "lifestyle",
            "Unload the knees and calves, realign the spine.",
            ("tadasana", "adho_mukha", "vrikshasana")),
    Routine("homemakers", "Homemakers / daily household routine", "lifestyle",
            "Counter forward bending and long hours on the feet.",
            ("marjaryasana", "bhujangasana", "vrikshasana", "balasana")),
    Routine("students", "Students / study focus", "lifestyle",
            "Undo text-neck and laptop slouch, wake up the mind.",
            ("tadasana", "adho_mukha", "trikonasana", "sukhasana")),
    Routine("lumbar_flex", "Lumbar care & general flexibility", "lifestyle",
            "Core strength and lower-body mobility.",
            ("virabhadrasana", "utkatasana", "balasana")),
)

GOALS = (
    Routine("daily_energy", "Daily energy flow", "goal",
            "Full-body activation and balance.",
            ("tadasana", "trikonasana", "virabhadrasana", "adho_mukha", "vrikshasana")),
    Routine("back_pain", "Back-pain relief", "goal",
            "Spine and lumbar care.",
            ("marjaryasana", "bhujangasana", "balasana", "utkatasana")),
    Routine("stress_relief", "Stress relief & focus", "goal",
            "Calmness and mindfulness.",
            ("balasana", "sukhasana", "adho_mukha", "tadasana")),
)


def by_position() -> tuple[Routine, ...]:
    out = []
    for pos, label in POSITIONS.items():
        keys = tuple(k for k, p in POSES.items() if p.position == pos)
        if keys:
            out.append(Routine(f"pos_{pos}", f"{label} poses", "position",
                               f"Every {label.lower()} pose in the library.", keys))
    return tuple(out)


def all_routines() -> dict[str, Routine]:
    return {r.key: r for r in (*LIFESTYLE, *GOALS, *by_position())}


def recommend(job: str) -> str:
    """The lifestyle routine that fits the person's working day."""
    return {"sitting": "desk_sitting", "standing": "standing_workers"}.get(
        job, "lumbar_flex")
