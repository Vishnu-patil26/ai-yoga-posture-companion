"""Volunteer photo collection - the data side (no camera, no GUI).

Why this exists.  The angle references in data/asana_fits.json were fitted from
public datasets that show one or two people per pose, so every target is really
"what that one person's body does".  The meeting notes ask for the team's own
sample of *diverse, healthy* volunteers, collected under consent and copyright
terms.  This module is everything about that which can be tested without a
webcam: who the volunteer is, what they agreed to, where each photo and its
landmarks go on disk, how to see whether the sample is diverse enough yet, how
to honour a withdrawal, and how to hand the right-labelled photos to
tools/fit_asana.py.  collect.py is only the capture window around it.

On disk (all under COLLECTED_DIR, which is gitignored by a file this module
writes into it - these are photographs of real people):

    data/collected/<volunteer_id>/volunteer.json   who, and what they agreed to
                                  consent.txt      the exact text they saw
                                  <pose_key>/<n>.jpg    the photo (see face_policy)
                                  <pose_key>/<n>.json   landmarks, features, label

Privacy rules the code enforces rather than merely documents:

* A volunteer without both ``consent`` and ``healthy_confirmed`` is never
  registered, so nothing can be saved for them.
* The volunteer's ``face_policy`` is a promise.  "blur" refuses to store an
  image when it cannot tell where the head is (an unblurred face would break the
  promise), and "landmarks_only" never writes a JPEG at all.
* Every volunteer id and pose key that becomes part of a path is checked against
  a strict pattern / the POSES catalogue first, so no input can walk out of
  COLLECTED_DIR.
* ``withdraw`` deletes the whole folder, and optionally the copies that
  ``export_for_fitting`` made, which are named after the volunteer id for
  exactly that reason.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

from yoga.evaluator import compute_features
from yoga.landmarks import (
    L_EAR, L_HIP, L_SHOULDER, N_LANDMARKS, NOSE, R_EAR, R_HIP, R_SHOULDER, Pose,
)
from yoga.routines import POSES

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Where volunteers live.  A module-level variable (like profile.USERS_DIR) so a
#: test can point it at a temporary folder.  Always read at call time.
COLLECTED_DIR = os.path.join(_PROJECT, "data", "collected")
#: Where `export_for_fitting` writes by default (used by the collect.py button).
EXPORT_DIR = os.path.join(_PROJECT, "data", "collected_export")

# ---------------------------------------------------------------- vocabulary
AGE_GROUPS = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
SEXES = ("female", "male", "other", "prefer not to say")
BODY_TYPES = ("slim", "average", "athletic", "broad-built", "larger-built",
              "prefer not to say")
EXPERIENCE = ("never", "beginner", "regular")
FACE_POLICIES = ("keep", "blur", "landmarks_only")
QUALITIES = ("right", "wrong")

#: The diversity report is about people we can ask for; these are the answers
#: that mean "I did not say", so a missing one is never flagged as a gap.
_NOT_SAID = "prefer not to say"

HEIGHT_RANGE = (100.0, 230.0)       # same sanity limits as yoga/profile.py
WEIGHT_RANGE = (20.0, 250.0)

#: Target used by `summary()`: a pose's reference should rest on at least this
#: many different people.  A target to aim for, not a result.
TARGET_VOLUNTEERS_PER_POSE = 10
#: One category holding more than this share of the volunteers is flagged ...
DOMINANT_SHARE = 0.60
#: ... but only once there are enough volunteers for a share to mean anything.
MIN_VOLUNTEERS_FOR_SHARE = 5

RETENTION_MONTHS = 12
CONSENT_VERSION = "v1-2026-10"

CONSENT_TEXT = f"""\
VOLUNTEER CONSENT AND COPYRIGHT LICENCE  (version {CONSENT_VERSION})

WHAT THIS IS
AI Yoga Companion is a college project: a program that watches a person do a yoga pose through a webcam and says which joints to adjust. To work for many different bodies, and not just a few, the team needs photos of healthy volunteers doing 10 yoga poses. Taking part is entirely your choice.

WHAT WE STORE
- A few photos of you for each pose you try (none at all if you choose "points only" below).
- For every photo: 33 body points found by software (they include rough positions of the nose, eyes, ears and mouth), the joint angles worked out from them, and a label saying whether the pose was done right or wrong.
- Your age group, sex, body type and yoga experience, and your height and weight only if you choose to give them.
- A random volunteer ID (like V-3F9A2C) instead of your name, the time you agreed, and a copy of this text.
We do NOT store your name, contact details, location or any sound.

YOUR FACE
You choose: keep the photos as they are, blur your face automatically before saving, or store only the body points and no photo at all. Automatic blurring relies on the software's guess of where your head is. It is good but not perfect, so choose "points only" if you want no image of you to exist.

HOW WE USE IT
Only to fit and test the angle references of this project, and to report results as averages and charts. Your photos will not be published, sold or shared outside the project team, their guide and the examiners, and will not be used to identify anyone or for advertising.

COPYRIGHT
You keep the copyright in photos of you. You give the project team a free, non-exclusive licence to store and use them only for the purposes above. The licence ends when you withdraw or your data is deleted, whichever comes first.

HEALTH
By ticking the health box you confirm that you are an adult in good health, with no injury or condition that makes these poses unsafe for you. Go only as far as is comfortable and stop at once if anything hurts. This is not medical advice.

YOUR RIGHTS
- You may skip any pose or stop at any time, without giving a reason.
- You may withdraw later. Give your volunteer ID to the project supervisor and your whole folder (photos, points and details) is deleted. Without the ID we cannot find your data, so please write it down.
- Data is kept only until the project has been assessed, and for at most {RETENTION_MONTHS} months, then deleted.
"""

ID_RE = re.compile(r"V-[0-9A-F]{6}")
_SAMPLE_RE = re.compile(r"(\d+)\.json")
_EXPORT_NAME_RE = re.compile(r"V-[0-9A-F]{6}_\d+\.jpg")

#: Longest fault note we keep; it is a short label, not a diary.
MAX_FAULT_CHARS = 120
JPEG_QUALITY = 92
SAMPLE_SCHEMA = 1


class CollectionError(ValueError):
    """The request is well formed but breaks a collection rule."""


# ----------------------------------------------------------------- volunteer
@dataclass
class Volunteer:
    """One person who agreed to be photographed.  No name is stored on purpose."""

    volunteer_id: str = ""            #: blank until `register` assigns one
    age_group: str = ""
    sex: str = ""
    height_cm: float | None = None    #: optional
    weight_kg: float | None = None    #: optional
    body_type: str = ""
    experience: str = ""
    #: The "healthy person" gate: no current injury or condition that makes the
    #: poses unsafe.  Self-declared - the supervisor is the second check.
    healthy_confirmed: bool = False
    consent: bool = False
    consent_version: str = ""         #: blank -> CONSENT_VERSION on register
    consent_time: str = ""            #: ISO 8601; blank -> now on register
    face_policy: str = "blur"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Volunteer":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def new_volunteer_id() -> str:
    """`V-` + six uppercase hex digits, never one that already has a folder.

    Six hex digits is 16.7 million ids - plenty for a class project, short
    enough to read out loud when someone asks to withdraw - but a random draw
    can still repeat, so the draw is checked against the folders that exist.
    """
    try:
        taken = set(os.listdir(COLLECTED_DIR))
    except OSError:
        taken = set()
    while True:
        vid = "V-" + uuid.uuid4().hex[:6].upper()
        if vid not in taken:
            return vid


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_volunteer(v: Volunteer) -> list[str]:
    """Human-readable problems; empty means the volunteer may be registered.

    A blank id / consent time / consent version is fine - `register` fills them
    in - but a filled one must be well formed.
    """
    errs: list[str] = []
    if v.consent is not True:
        errs.append("Consent: the volunteer must agree to the consent and licence text.")
    if v.healthy_confirmed is not True:
        errs.append("Health: the volunteer must confirm they have no injury or condition "
                    "that makes these poses unsafe.")
    for label, value, choices in (
            ("Age group", v.age_group, AGE_GROUPS), ("Sex", v.sex, SEXES),
            ("Body type", v.body_type, BODY_TYPES), ("Experience", v.experience, EXPERIENCE),
            ("Face policy", v.face_policy, FACE_POLICIES)):
        if value not in choices:
            errs.append(f"{label}: choose one of {', '.join(choices)}.")
    for label, value, (lo, hi) in (("Height (cm)", v.height_cm, HEIGHT_RANGE),
                                   ("Weight (kg)", v.weight_kg, WEIGHT_RANGE)):
        if value is None:
            continue                                  # optional
        if not _is_number(value) or not lo <= value <= hi:
            errs.append(f"{label}: leave blank or give a number from {lo:g} to {hi:g}.")
    if v.volunteer_id and not ID_RE.fullmatch(v.volunteer_id):
        errs.append("Volunteer ID: expected V- followed by six hex digits (0-9, A-F).")
    if v.consent_version and v.consent_version != CONSENT_VERSION:
        errs.append(f"Consent version: expected {CONSENT_VERSION}.")
    if v.consent_time:
        try:
            datetime.fromisoformat(v.consent_time)
        except (TypeError, ValueError):
            errs.append("Consent time: expected an ISO 8601 timestamp.")
    return errs


# --------------------------------------------------------------------- paths
def _within(path: str, root: str) -> bool:
    try:
        real, base = os.path.realpath(path), os.path.realpath(root)
        return os.path.commonpath([real, base]) == base
    except ValueError:                                # e.g. a different drive
        return False


def _check_id(volunteer_id) -> str:
    if not isinstance(volunteer_id, str) or not ID_RE.fullmatch(volunteer_id):
        raise ValueError(f"Not a volunteer id: {volunteer_id!r} (expected V-XXXXXX, hex).")
    return volunteer_id


def _check_pose(pose_key) -> str:
    if not isinstance(pose_key, str) or pose_key not in POSES:
        raise ValueError(f"Unknown pose {pose_key!r}. Choose one of: {', '.join(POSES)}.")
    return pose_key


def _vol_dir(volunteer_id) -> str:
    path = os.path.join(COLLECTED_DIR, _check_id(volunteer_id))
    if not _within(path, COLLECTED_DIR):              # belt and braces (symlinks)
        raise ValueError("Volunteer folder escapes the collection folder.")
    return path


def _pose_dir(volunteer_id, pose_key) -> str:
    return os.path.join(_vol_dir(volunteer_id), _check_pose(pose_key))


def _protect(folder: str) -> None:
    """Make git ignore a folder of photographs of real people.

    The project .gitignore does not know about these folders, and one careless
    `git add .` would publish volunteers' faces.  A `.gitignore` containing `*`
    inside the folder protects it (and itself) wherever it lives.
    """
    marker = os.path.join(folder, ".gitignore")
    if not os.path.exists(marker):
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("*\n")


def _write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _write_json(path: str, obj) -> None:
    _write_bytes(path, json.dumps(obj, indent=2, allow_nan=False).encode("utf-8"))


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _jsonable(x):
    """Plain JSON types only; NaN / inf become null (JSON has no NaN)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


# ------------------------------------------------------------- registration
def register(volunteer: Volunteer) -> str:
    """Validate, give the volunteer an id, write their folder.  Returns its path.

    The object passed in is updated in place (id, consent version and time are
    filled when blank) so the caller can read the id back.  Nothing is written
    unless the volunteer passes `validate_volunteer`, and an existing folder is
    never overwritten.
    """
    if not volunteer.volunteer_id:
        volunteer.volunteer_id = new_volunteer_id()
    if not volunteer.consent_version:
        volunteer.consent_version = CONSENT_VERSION
    if not volunteer.consent_time:
        volunteer.consent_time = now_iso()
    problems = validate_volunteer(volunteer)
    if problems:
        raise CollectionError("; ".join(problems))
    path = _vol_dir(volunteer.volunteer_id)
    os.makedirs(COLLECTED_DIR, exist_ok=True)
    _protect(COLLECTED_DIR)
    os.makedirs(path, exist_ok=False)                 # FileExistsError, never overwrite
    _write_json(os.path.join(path, "volunteer.json"), volunteer.to_dict())
    header = (f"Consent version: {volunteer.consent_version}\n"
              f"Volunteer ID: {volunteer.volunteer_id}\n"
              f"Consent given: {volunteer.consent_time}\n"
              f"Healthy-person confirmation: yes\n"
              f"Face policy chosen: {volunteer.face_policy}\n"
              f"{'-' * 60}\n")
    _write_bytes(os.path.join(path, "consent.txt"), (header + CONSENT_TEXT).encode("utf-8"))
    return path


def load_volunteer(volunteer_id: str) -> Volunteer | None:
    data = _read_json(os.path.join(_vol_dir(volunteer_id), "volunteer.json"))
    return Volunteer.from_dict(data) if isinstance(data, dict) else None


def list_volunteers() -> list[Volunteer]:
    """Every registered volunteer, oldest consent first."""
    out = []
    try:
        names = sorted(os.listdir(COLLECTED_DIR))
    except OSError:
        return []
    for name in names:
        if ID_RE.fullmatch(name) and os.path.isdir(os.path.join(COLLECTED_DIR, name)):
            v = load_volunteer(name)
            if v is not None:
                out.append(v)
    out.sort(key=lambda v: (v.consent_time, v.volunteer_id))
    return out


# --------------------------------------------------------------- face blurring
#: Smallest circle we blur, in pixels, whatever the landmarks say.
MIN_BLUR_RADIUS_PX = 16


def head_region(pose: Pose | None) -> tuple[int, int, int] | None:
    """Centre x, centre y and radius (pixels) of the circle that covers the head.

    No face detector is used.  BlazePose already places the nose and both ears;
    their centroid is the head centre (nudged up the neck towards the crown) and
    the radius is the largest of several body-relative measures, because any
    one of them collapses in some view:
    the ear gap vanishes in a side view, the shoulder width in a side view too,
    and the nose-to-ear spread when the head is turned away.  The torso length
    never collapses, so it is the floor.  Returns None when the head cannot be
    located (no pose, or non-finite coordinates).
    """
    if pose is None:
        return None
    p = np.asarray(pose.pts, dtype=float)
    head = p[[NOSE, L_EAR, R_EAR]]
    if not np.all(np.isfinite(head)):
        return None
    centre = head.mean(axis=0)
    spread = float(np.max(np.linalg.norm(head - centre, axis=1)))
    ear_gap = float(np.linalg.norm(p[L_EAR] - p[R_EAR]))
    shoulders = float(np.linalg.norm(p[L_SHOULDER] - p[R_SHOULDER]))
    torso = float(np.linalg.norm((p[L_SHOULDER] + p[R_SHOULDER]) * 0.5
                                 - (p[L_HIP] + p[R_HIP]) * 0.5))
    candidates = [1.8 * spread, 0.8 * ear_gap, 0.4 * shoulders, 0.30 * torso,
                  float(MIN_BLUR_RADIUS_PX)]
    radius = 1.15 * max(c for c in candidates if math.isfinite(c))
    # The nose and ears sit at face level, below the crown, so the centroid is
    # low and the hair would show above the disc (seen on real Cobra photos).
    # Slide the centre away from the shoulders, i.e. along the neck - which also
    # follows the head when it hangs down in Downward Dog.
    neck = centre - (p[L_SHOULDER] + p[R_SHOULDER]) * 0.5
    reach = float(np.linalg.norm(neck)) if np.all(np.isfinite(neck)) else 0.0
    if reach > 1e-6:
        centre = centre + neck / reach * (0.25 * radius)
    return int(round(centre[0])), int(round(centre[1])), int(round(radius))


def blur_faces(frame_bgr: np.ndarray, pose: Pose | None) -> np.ndarray:
    """A copy of the frame with a Gaussian-blurred disc over the head.

    Only pixels inside the disc change.  With ``pose=None`` (or a pose whose head
    cannot be located) there is nothing to aim at, so the copy comes back
    UNCHANGED - it is not anonymised.  Callers that promise blurring must check
    `head_region` first; `save_sample` does.
    """
    out = frame_bgr.copy()
    region = head_region(pose)
    if region is None:
        return out
    cx, cy, r = region
    h, w = out.shape[:2]
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x0 >= x1 or y0 >= y1:                          # head entirely outside the frame
        return out
    roi = out[y0:y1, x0:x1]
    # Strong enough that eyes and mouth are gone: sigma is over half the radius.
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=max(3.0, 0.6 * r))
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), r, 255, -1)
    roi[mask > 0] = blurred[mask > 0]                 # roi is a view into out
    return out


# ------------------------------------------------------------------- samples
def _next_sample_number(pose_dir: str) -> int:
    try:
        names = os.listdir(pose_dir)
    except OSError:
        return 1
    nums = [int(m.group(1)) for n in names if (m := re.fullmatch(r"(\d+)\.(?:json|jpg)", n))]
    return max(nums, default=0) + 1


def _clean_fault(text: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()[:MAX_FAULT_CHARS]


def _landmark_rows(pose: Pose) -> list[dict]:
    pts = np.asarray(pose.pts, dtype=float)
    if pts.shape != (N_LANDMARKS, 2):
        raise ValueError(f"Expected {N_LANDMARKS} landmarks, got shape {pts.shape}.")
    world = np.asarray(pose.world, dtype=float)
    z = world[:, 2] if world.shape == (N_LANDMARKS, 3) else np.zeros(N_LANDMARKS)
    return [{"x": round(float(pts[i, 0]), 2), "y": round(float(pts[i, 1]), 2),
             "z": round(float(z[i]), 4), "visibility": round(float(pose.vis[i]), 3)}
            for i in range(N_LANDMARKS)]


def save_sample(volunteer_id: str, pose_key: str, frame_bgr: np.ndarray, pose: Pose | None,
                quality: str, fault: str = "") -> str:
    """Store one photo (as the volunteer's face policy allows) and its JSON.

    Returns the path of the ``<n>.json`` file.  In the JSON, ``x`` and ``y`` are
    pixels in the (un-mirrored) camera frame, ``z`` is BlazePose's hip-centred
    world depth in metres, and ``features`` is exactly what the live trainer
    would measure, so a sample can be refitted without re-running the detector.
    ``fault`` (e.g. "knees bent") is kept only for ``quality == "wrong"``.
    """
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {QUALITIES}, got {quality!r}.")
    vol = load_volunteer(volunteer_id)
    pose_dir = _pose_dir(volunteer_id, pose_key)      # validates both names first
    if vol is None:
        raise CollectionError(f"Volunteer {volunteer_id} is not registered.")
    if not (isinstance(frame_bgr, np.ndarray) and frame_bgr.ndim == 3
            and frame_bgr.shape[2] == 3 and frame_bgr.dtype == np.uint8):
        raise ValueError("frame_bgr must be an HxWx3 uint8 BGR image.")

    policy = vol.face_policy
    if policy == "landmarks_only" and pose is None:
        raise CollectionError("This volunteer chose 'points only' and no body was found, "
                              "so there is nothing to store.")
    image = None
    if policy == "keep":
        image = frame_bgr
    elif policy == "blur":
        if head_region(pose) is None:
            raise CollectionError("Cannot blur the face because no head was found; "
                                  "storing the photo would break the volunteer's choice.")
        image = blur_faces(frame_bgr, pose)

    record = {
        "schema": SAMPLE_SCHEMA,
        "volunteer_id": vol.volunteer_id,
        "pose": pose_key,
        "captured_at": now_iso(),
        "quality": quality,
        "fault": _clean_fault(fault) if quality == "wrong" else "",
        "face_policy": policy,
        "image": None,
        "image_size": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],
        "detected": pose is not None,
        "landmarks": _landmark_rows(pose) if pose is not None else None,
        "features": _jsonable(compute_features(pose)) if pose is not None else None,
    }

    os.makedirs(pose_dir, exist_ok=True)
    n = _next_sample_number(pose_dir)
    stem = f"{n:03d}"
    jpg_path = os.path.join(pose_dir, stem + ".jpg")
    json_path = os.path.join(pose_dir, stem + ".json")
    if image is not None:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise CollectionError("The photo could not be encoded as JPEG.")
        _write_bytes(jpg_path, buf.tobytes())
        record["image"] = stem + ".jpg"
    try:
        _write_json(json_path, record)
    except Exception:
        if os.path.exists(jpg_path):                  # never leave a photo without its record
            os.remove(jpg_path)
        raise
    return json_path


def delete_sample(sample_json_path: str) -> bool:
    """Delete one sample (its JSON and photo), e.g. for "retake last"."""
    path = os.path.abspath(sample_json_path)
    pose_dir = os.path.dirname(path)
    vol_dir = os.path.dirname(pose_dir)
    same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))  # noqa: E731
    # It must be exactly <COLLECTED_DIR>/<volunteer id>/<pose key>/<n>.json -
    # this function deletes files, so it accepts nothing looser than that.
    if not (_within(path, COLLECTED_DIR) and _SAMPLE_RE.fullmatch(os.path.basename(path))
            and os.path.basename(pose_dir) in POSES
            and ID_RE.fullmatch(os.path.basename(vol_dir))
            and same(os.path.dirname(vol_dir), COLLECTED_DIR)):
        raise ValueError(f"Not a sample file inside the collection folder: {sample_json_path!r}")
    gone = False
    for ext in (".json", ".jpg"):
        victim = path[:-5] + ext
        if os.path.isfile(victim):
            os.remove(victim)
            gone = True
    return gone


def _iter_samples(volunteer_id: str):
    """(pose_key, number, record) for every readable sample of one volunteer."""
    base = _vol_dir(volunteer_id)
    for pose_key in POSES:
        folder = os.path.join(base, pose_key)
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in names:
            m = _SAMPLE_RE.fullmatch(name)
            if not m:
                continue
            rec = _read_json(os.path.join(folder, name))
            if isinstance(rec, dict):
                yield pose_key, int(m.group(1)), rec


def volunteer_counts(volunteer_id: str) -> dict[str, dict[str, int]]:
    """pose_key -> {"right": n, "wrong": n} for one volunteer (all poses listed)."""
    out = {k: {q: 0 for q in QUALITIES} for k in POSES}
    for pose_key, _n, rec in _iter_samples(volunteer_id):
        if rec.get("quality") in QUALITIES:
            out[pose_key][rec["quality"]] += 1
    return out


# ------------------------------------------------------------------- summary
@dataclass
class CollectionSummary:
    n_volunteers: int = 0
    n_samples: int = 0
    #: pose_key -> quality -> volunteer_id -> number of samples
    counts: dict = field(default_factory=dict)
    #: pose_key -> volunteers with at least one RIGHT-labelled sample (the ones
    #: tools/fit_asana.py will actually learn from)
    volunteers_per_pose: dict = field(default_factory=dict)
    #: attribute -> category -> volunteers (every category listed, zeros too)
    diversity: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)

    def text(self) -> str:
        lines = [f"Volunteers: {self.n_volunteers}    Samples: {self.n_samples}", "",
                 "Per pose  (people with a right-labelled sample / right / wrong samples)"]
        for key, pose in POSES.items():
            c = self.counts.get(key, {})
            right = sum(c.get("right", {}).values())
            wrong = sum(c.get("wrong", {}).values())
            lines.append(f"  {pose.name:<22s} {self.volunteers_per_pose.get(key, 0):>3d} / "
                         f"{right:>4d} / {wrong:>4d}   ({key})")
        lines += ["", "Diversity (volunteers)"]
        for attr, cats in self.diversity.items():
            lines.append(f"  {attr.replace('_', ' '):<11s} "
                         + ", ".join(f"{k} {v}" for k, v in cats.items()))
        lines += ["", "Needs attention" if self.flags else "No gaps flagged against the targets."]
        lines += [f"  - {f}" for f in self.flags]
        return "\n".join(lines)


def summary() -> CollectionSummary:
    """Counts per pose x quality x volunteer, and where the sample is thin."""
    vols = list_volunteers()
    s = CollectionSummary(n_volunteers=len(vols))
    s.counts = {k: {q: {} for q in QUALITIES} for k in POSES}
    for v in vols:
        for pose_key, _n, rec in _iter_samples(v.volunteer_id):
            q = rec.get("quality")
            if q not in QUALITIES:
                continue
            by_vol = s.counts[pose_key][q]
            by_vol[v.volunteer_id] = by_vol.get(v.volunteer_id, 0) + 1
            s.n_samples += 1
    s.volunteers_per_pose = {k: len(s.counts[k]["right"]) for k in POSES}

    for attr, choices in (("age_group", AGE_GROUPS), ("sex", SEXES),
                          ("body_type", BODY_TYPES), ("experience", EXPERIENCE)):
        s.diversity[attr] = {c: sum(1 for v in vols if getattr(v, attr) == c) for c in choices}

    plural = lambda n: f"{n} volunteer" + ("" if n == 1 else "s")          # noqa: E731
    if not vols:
        s.flags.append("No volunteers registered yet.")
        return s
    for key in POSES:
        n = s.volunteers_per_pose[key]
        if n == 0:
            s.flags.append(f"{key}: no right-labelled samples yet "
                           f"(target {TARGET_VOLUNTEERS_PER_POSE} volunteers) - collect more.")
        elif n < TARGET_VOLUNTEERS_PER_POSE:
            s.flags.append(f"{key}: only {plural(n)} with a right-labelled sample "
                           f"(target {TARGET_VOLUNTEERS_PER_POSE}) - collect more.")
        if n and not s.counts[key]["wrong"]:
            s.flags.append(f"{key}: no wrong-labelled samples yet - needed to check that "
                           "real faults are caught.")
    for attr, cats in s.diversity.items():
        if attr == "experience":
            continue
        for cat, n in cats.items():
            if n == 0 and cat != _NOT_SAID and cat != "other":
                s.flags.append(f"{attr.replace('_', ' ')}: no volunteers in '{cat}' yet.")
        top_cat, top_n = max(cats.items(), key=lambda kv: kv[1])
        if len(vols) >= MIN_VOLUNTEERS_FOR_SHARE and top_n / len(vols) > DOMINANT_SHARE:
            s.flags.append(f"{attr.replace('_', ' ')}: '{top_cat}' is {top_n} of "
                           f"{len(vols)} volunteers ({100 * top_n // len(vols)}%) - recruit "
                           "more variety.")
    return s


# ------------------------------------------------------------ withdraw/export
def _rm_readonly(func, path, _exc) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def withdraw(volunteer_id: str, also_in: str | None = None) -> bool:
    """Right to withdraw: delete the volunteer's whole folder.

    Returns False if no such volunteer exists.  ``also_in`` names an export
    folder (see `export_for_fitting`) whose ``<id>_<n>.jpg`` copies are removed
    too, so a withdrawal does not leave the photos behind in a derived copy.
    """
    path = _vol_dir(volunteer_id)
    existed = os.path.isdir(path)
    if existed:
        shutil.rmtree(path, onerror=_rm_readonly)
    if also_in and os.path.isdir(also_in):
        _purge_export(also_in, only_id=volunteer_id)
    return existed and not os.path.exists(path)


def _purge_export(dest: str, only_id: str | None = None) -> int:
    """Remove files this module exported (`V-XXXXXX_n.jpg`) and nothing else."""
    removed = 0
    for pose_key in POSES:
        folder = os.path.join(dest, pose_key)
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            if _EXPORT_NAME_RE.fullmatch(name) and (only_id is None
                                                    or name.startswith(only_id + "_")):
                os.remove(os.path.join(folder, name))
                removed += 1
    return removed


def export_for_fitting(dest: str) -> dict:
    """Copy every right-labelled photo into ``dest/<pose_key>/<id>_<n>.jpg``.

    That folder-per-class layout is what tools/fit_asana.py reads.  The export
    mirrors the collection: copies this function made earlier are removed first,
    so a volunteer who has since withdrawn does not linger.  Files it did not
    create are left alone.  ``dest`` may not sit inside the collection folder.

    Returns ``{"per_pose": {pose_key: n}, "total": n, "volunteers": n,
    "skipped_no_image": n}``; the last counts right-labelled samples whose
    volunteer chose "points only" (they have landmarks but no photo to copy).
    """
    dest = os.path.abspath(dest)
    if _within(dest, COLLECTED_DIR):
        raise ValueError("The export folder must be outside the collection folder.")
    created = not os.path.exists(dest)
    os.makedirs(dest, exist_ok=True)
    if created:
        _protect(dest)
    _purge_export(dest)

    per_pose = {k: 0 for k in POSES}
    people: set[str] = set()
    skipped = 0
    for v in list_volunteers():
        base = _vol_dir(v.volunteer_id)
        for pose_key, n, rec in _iter_samples(v.volunteer_id):
            if rec.get("quality") != "right":
                continue
            src = os.path.join(base, pose_key, f"{n:03d}.jpg")
            if not os.path.isfile(src):
                skipped += 1
                continue
            out_dir = os.path.join(dest, pose_key)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(src, os.path.join(out_dir, f"{v.volunteer_id}_{n}.jpg"))
            per_pose[pose_key] += 1
            people.add(v.volunteer_id)
    return {"per_pose": per_pose, "total": sum(per_pose.values()),
            "volunteers": len(people), "skipped_no_image": skipped}
