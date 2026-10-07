"""The user's health profile and what it changes about the routine.

Collected on the first screen: diet, age, weight, height, bust/shoulder, waist,
hip, diabetic, blood pressure, standing or sitting job, location, medication.
It is stored on this machine only (data/users/<name>.json).

What each field actually does - nothing here is decoration:

* age, weight, height  -> BMI band, hold lengths
* waist, hip           -> waist-to-hip ratio (shown, not scored)
* bust/shoulder        -> recorded for the body-size summary
* diabetic             -> shorter holds, pre-practice cautions, diet notes
* blood pressure       -> head-below-heart poses are skipped when high
* job                  -> which lifestyle routine is recommended
* diet                 -> the diet-plan notes
* location             -> shown on the summary and stored with each session
* medication           -> free text read by yoga.nlp: each medicine class it
                          recognises becomes a short caution on the plan (never
                          a dose, never advice to start or stop anything), and a
                          few classes - blood thinners, blood-pressure tablets,
                          beta blockers - also skip head-below-heart poses
* user_id              -> a short readable id ("YC-7F3A9C") made the first time
                          the profile is saved; it keys the stored data, so the
                          profile file and every history entry carry the same id

The rules are deliberately conservative and the app says it is not medical
advice.  They only ever make a routine gentler, never harder.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field

from yoga.routines import POSES, Routine

USERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "users")

DIETS = ("Vegetarian", "Vegan", "Eggetarian", "Non-vegetarian", "Jain")
BP = ("None", "High", "Low")
JOBS = ("sitting", "standing", "mixed")
MEDICATION_MAX = 300

DISCLAIMER = ("General wellness guidance, not medical advice. If you have a "
              "condition, check with your doctor before starting.")


@dataclass
class Profile:
    name: str = ""
    age: int = 0
    weight_kg: float = 0.0
    height_cm: float = 0.0
    bust_cm: float | None = None          #: bust / shoulder width (optional)
    waist_cm: float | None = None
    hip_cm: float | None = None
    diabetic: bool = False
    bp: str = "None"
    job: str = "sitting"
    diet: str = "Vegetarian"
    location: str = ""
    #: Short readable id, made on first save.  Excluded from ``==`` on purpose:
    #: it is bookkeeping about the *record*, so two profiles describing the same
    #: person's body and habits compare equal whether or not one has been saved.
    user_id: str = field(default="", compare=False)
    #: Free text ("metformin, amlodipine"); read by yoga.nlp for cautions.
    medication: str = ""

    @property
    def bmi(self) -> float:
        return self.weight_kg / ((self.height_cm / 100.0) ** 2)

    @property
    def bmi_band(self) -> str:
        b = self.bmi
        return ("underweight" if b < 18.5 else "healthy" if b < 25
                else "overweight" if b < 30 else "obese")

    @property
    def whr(self) -> float | None:
        if self.waist_cm and self.hip_cm:
            return self.waist_cm / self.hip_cm
        return None


_RANGES = {"age": (10, 100), "weight_kg": (20, 250), "height_cm": (100, 230),
           "bust_cm": (50, 200), "waist_cm": (40, 200), "hip_cm": (50, 200)}
_LABEL = {"age": "Age", "weight_kg": "Weight (kg)", "height_cm": "Height (cm)",
          "bust_cm": "Bust/shoulder (cm)", "waist_cm": "Waist (cm)",
          "hip_cm": "Hip (cm)"}


def validate(p: Profile) -> list[str]:
    """Human-readable problems; empty means the profile is usable."""
    errs = []
    if not re.fullmatch(r"[A-Za-z0-9 _-]{1,30}", p.name.strip()):
        errs.append("Name: use 1-30 letters, numbers, spaces or dashes.")
    for f, (lo, hi) in _RANGES.items():
        v = getattr(p, f)
        required = f in ("age", "weight_kg", "height_cm")
        if v in (None, 0, 0.0):
            if required:
                errs.append(f"{_LABEL[f]}: required.")
            continue
        if not lo <= v <= hi:
            errs.append(f"{_LABEL[f]}: expected {lo}-{hi}.")
    if p.bp not in BP:
        errs.append("Blood pressure: choose None, High or Low.")
    if p.job not in JOBS:
        errs.append("Job: choose sitting, standing or mixed.")
    if len(p.medication) > MEDICATION_MAX:
        errs.append(f"Medication: keep it under {MEDICATION_MAX} characters.")
    return errs


# ---------------------------------------------------------------- storage
def _path(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()) or "user"
    return os.path.join(USERS_DIR, f"{safe}.json")


def _stored_ids() -> set[str]:
    """Every user_id already on disk, so a new one can never collide with one."""
    ids: set[str] = set()
    try:
        names = os.listdir(USERS_DIR)
    except OSError:
        return ids
    for f in names:
        if not f.endswith(".json") or f.endswith("_history.json"):
            continue
        try:
            with open(os.path.join(USERS_DIR, f), encoding="utf-8") as fh:
                uid = json.load(fh).get("user_id")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(uid, str) and uid:
            ids.add(uid)
    return ids


def new_user_id() -> str:
    """A short unique readable id such as ``YC-7F3A9C`` (6 hex digits of a uuid4)."""
    taken = _stored_ids()
    while True:
        uid = "YC-" + uuid.uuid4().hex[:6].upper()
        if uid not in taken:
            return uid


def save(p: Profile) -> str:
    """Write the profile.  Makes ``p.user_id`` first if it is empty.

    Saving the same name again keeps the id already on disk instead of issuing
    a new one: the id identifies the person's record, and the launcher builds a
    fresh Profile object every time the form is submitted.
    """
    os.makedirs(USERS_DIR, exist_ok=True)
    path = _path(p.name)
    if not p.user_id:
        try:
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh).get("user_id", "")
        except (OSError, ValueError, AttributeError):
            old = ""
        p.user_id = old if isinstance(old, str) and old else new_user_id()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(p), fh, indent=2)
    return path


def load(name: str) -> Profile | None:
    try:
        with open(_path(name), encoding="utf-8") as fh:
            data = json.load(fh)
        return Profile(**{k: v for k, v in data.items() if k in Profile.__dataclass_fields__})
    except (OSError, ValueError, TypeError):
        return None


def saved_names() -> list[str]:
    try:
        return sorted(f[:-5] for f in os.listdir(USERS_DIR)
                      if f.endswith(".json") and not f.endswith("_history.json"))
    except OSError:
        return []


def append_history(p: Profile, rows: list[dict]) -> str:
    os.makedirs(USERS_DIR, exist_ok=True)
    path = _path(p.name)[:-5] + "_history.json"
    try:
        with open(path, encoding="utf-8") as fh:
            hist = json.load(fh)
    except (OSError, ValueError):
        hist = []
    hist.append({"user_id": p.user_id, "location": p.location, "poses": rows})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(hist, fh, indent=2)
    return path


# ------------------------------------------------------------ personalise
@dataclass
class PlanItem:
    pose_key: str
    hold_s: int
    skipped: bool = False
    reason: str = ""


@dataclass
class Plan:
    routine: Routine
    items: list[PlanItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    diet: list[str] = field(default_factory=list)

    @property
    def active(self) -> list[PlanItem]:
        return [i for i in self.items if not i.skipped]


def personalise(p: Profile, routine: Routine) -> Plan:
    """Apply the profile to a routine.  Only ever gentler, never harder."""
    factor, notes = 1.0, []
    skip_inversion = False

    if p.age >= 60:
        factor *= 0.75
        skip_inversion = True
        notes.append("Age 60+: holds shortened and head-below-heart poses skipped.")
    if p.bp == "High":
        factor *= 0.85
        skip_inversion = True
        notes.append("High blood pressure: head-below-heart poses skipped, breathe "
                     "steadily and never hold your breath.")
    if p.bp == "Low":
        notes.append("Low blood pressure: rise slowly after floor poses to avoid dizziness.")
    if p.diabetic:
        factor *= 0.85
        notes.append("Diabetic: practise 1-2 hours after a meal, keep a glucose source "
                     "within reach, and stop if shaky or light-headed.")
    if p.bmi_band == "obese":
        factor *= 0.85
        notes.append("BMI is in the obese range: use a folded mat under the knees and "
                     "keep holds comfortable.")
    elif p.bmi_band == "underweight":
        notes.append("BMI is underweight: keep sessions short and don't practise "
                     "on an empty stomach.")
    if p.job == "sitting":
        notes.append("Seated job: stand up and walk for two minutes every hour as well.")
    elif p.job == "standing":
        notes.append("Standing job: finish with your feet up for a few minutes.")
    med_skip = False
    if p.medication.strip():
        # imported here, not at the top: nlp is only needed when there is text to
        # read, and this keeps `import yoga.profile` free of any NLP start-up cost
        from yoga import nlp
        notes.extend(nlp.medication_flags(p.medication))
        if "avoid_inversion" in nlp.safety_flags(p.medication):
            med_skip = skip_inversion = True
            notes.append("Head-below-heart poses are skipped because of the medicine or "
                         "condition you listed. Check with your doctor.")
    factor = max(0.5, factor)

    plan = Plan(routine=routine, notes=notes, diet=diet_plan(p))
    for key in routine.poses:
        pose = POSES[key]
        if skip_inversion and "inversion" in pose.tags:
            plan.items.append(PlanItem(key, 0, True, _skip_reason(p, med_skip)))
            continue
        hold = max(10, int(round(pose.hold_s * factor / 5.0)) * 5)
        plan.items.append(PlanItem(key, hold))
    return plan


def _skip_reason(p: Profile, med_skip: bool) -> str:
    """Why head-below-heart poses were dropped, naming only what applied."""
    if not med_skip:
        return "skipped for your age / blood pressure"
    if p.age >= 60 or p.bp == "High":
        return "skipped for your age / blood pressure / medication"
    return "skipped for your medication"


def diet_plan(p: Profile) -> list[str]:
    protein = {
        "Vegetarian": "protein from dal, beans, paneer, curd and sprouts",
        "Vegan": "protein from lentils, chickpeas, tofu, soya and nuts",
        "Eggetarian": "protein from eggs, dal, curd and sprouts",
        "Non-vegetarian": "lean protein: fish, chicken, eggs, plus dal",
        "Jain": "protein from dal, beans, curd and nuts (no root vegetables)",
    }.get(p.diet, "protein at every meal")
    out = [f"Eating style: {p.diet}. Build each meal around {protein}."]
    if p.diabetic:
        out.append("Diabetic: choose low-glycaemic carbs (millets, oats, whole grains), "
                   "avoid sugary drinks, keep meal times regular.")
    else:
        out.append("Carbs: whole grains and fruit over refined flour and sugar.")
    if p.bp == "High":
        out.append("High BP: cut back on salt, pickles, papad and packaged snacks.")
    if p.bmi_band in ("overweight", "obese"):
        out.append("Weight: smaller portions, vegetables filling half the plate, "
                   "no late-night heavy meals.")
    elif p.bmi_band == "underweight":
        out.append("Weight: add calorie-dense foods - nuts, ghee, banana, full-fat curd.")
    out.append("Drink water through the day; eat 2-3 hours before practice.")
    out.append(DISCLAIMER)
    return out
