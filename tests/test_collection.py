"""Offline checks for volunteer photo collection (yoga/collection.py and collect.py).

    python tests/test_collection.py     (no camera needed; opens and closes a window)

Everything runs against a temporary folder - the module-level COLLECTED_DIR is
pointed at it for the duration - so the real data/collected is never touched.
The camera is replaced by a scripted fake; what this therefore cannot show is
that a real webcam delivers frames to the capture loop (see the protocol doc).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import types

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import collect                                                    # noqa: E402
from yoga import collection as col                                # noqa: E402
from yoga.landmarks import (                                      # noqa: E402
    L_ANKLE, L_EAR, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE,
    R_ANKLE, R_EAR, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, Pose,
)
from yoga.routines import POSES                                   # noqa: E402

FAILS: list[str] = []
N = 0


def check(cond: bool, label: str) -> None:
    global N
    N += 1
    if not cond:
        FAILS.append(label)
        print(f"  FAIL  {label}")


def raises(fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:                                  # the wrong kind of failure
        return False
    return False


# ------------------------------------------------------------------ fixtures
@contextlib.contextmanager
def sandbox():
    """Point the collection at a throwaway folder."""
    old = col.COLLECTED_DIR, col.EXPORT_DIR
    tmp = tempfile.mkdtemp()
    col.COLLECTED_DIR = os.path.join(tmp, "collected")
    col.EXPORT_DIR = os.path.join(tmp, "export")
    try:
        yield tmp
    finally:
        col.COLLECTED_DIR, col.EXPORT_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


def vol(**kw) -> col.Volunteer:
    d = dict(age_group="25-34", sex="female", height_cm=165, weight_kg=58, body_type="average",
             experience="beginner", healthy_confirmed=True, consent=True, face_policy="keep")
    d.update(kw)
    return col.Volunteer(**d)


def make_pose(w: int = 320, h: int = 240, head: tuple[int, int] = (160, 50),
              degenerate: bool = False) -> Pose:
    """A standing figure, or (degenerate) every joint on one point -> NaN angles."""
    pts = np.zeros((33, 2))
    if degenerate:
        pts[:] = (w / 2, h / 2)
    else:
        cx, cy = head
        pts[:] = (cx, cy + 90)                         # anything not listed sits mid-body
        pts[NOSE] = (cx, cy)
        pts[L_EAR], pts[R_EAR] = (cx + 12, cy + 2), (cx - 12, cy + 2)
        pts[L_SHOULDER], pts[R_SHOULDER] = (cx + 25, cy + 40), (cx - 25, cy + 40)
        pts[L_ELBOW], pts[R_ELBOW] = (cx + 30, cy + 68), (cx - 30, cy + 68)
        pts[L_WRIST], pts[R_WRIST] = (cx + 30, cy + 95), (cx - 30, cy + 95)
        pts[L_HIP], pts[R_HIP] = (cx + 18, cy + 100), (cx - 18, cy + 100)
        pts[L_KNEE], pts[R_KNEE] = (cx + 18, cy + 145), (cx - 18, cy + 145)
        pts[L_ANKLE], pts[R_ANKLE] = (cx + 18, cy + 185), (cx - 18, cy + 185)
    world = np.zeros((33, 3))
    world[:, 2] = np.linspace(-0.1, 0.1, 33)
    return Pose(pts=pts, raw=pts.copy(), vis=np.ones(33), world=world, width=w, height=h)


def noise_frame(seed: int = 0, w: int = 320, h: int = 240) -> np.ndarray:
    """High-frequency noise, so a blur visibly changes it."""
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def circle_mask(shape, region) -> np.ndarray:
    cx, cy, r = region
    m = np.zeros(shape[:2], np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_image(path: str):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


# --------------------------------------------------------------------- tests
def test_consent_gate() -> None:
    check(col.validate_volunteer(vol()) == [], "a complete volunteer validates")
    check(col.validate_volunteer(vol(height_cm=None, weight_kg=None)) == [],
          "height and weight are optional")
    check(any("Consent" in e for e in col.validate_volunteer(vol(consent=False))),
          "no consent -> rejected")
    check(any("Health" in e for e in col.validate_volunteer(vol(healthy_confirmed=False))),
          "no healthy confirmation -> rejected")
    check(len(col.validate_volunteer(vol(consent=False, healthy_confirmed=False))) == 2,
          "both gates report separately")
    check(col.validate_volunteer(vol(consent=1)) != [], "a truthy non-True consent does not pass")
    for field, bad in (("age_group", "17-24"), ("sex", "x"), ("body_type", ""),
                       ("experience", "expert"), ("face_policy", "show"),
                       ("height_cm", 40), ("weight_kg", 900), ("height_cm", float("nan")),
                       ("weight_kg", True), ("volunteer_id", "../x"),
                       ("consent_version", "v0"), ("consent_time", "yesterday")):
        check(len(col.validate_volunteer(vol(**{field: bad}))) == 1, f"{field}={bad!r} rejected")
    check(set(col.AGE_GROUPS) == {"18-24", "25-34", "35-44", "45-54", "55-64", "65+"},
          "the six age groups")
    with sandbox():
        for kw in ({"consent": False}, {"healthy_confirmed": False}, {"age_group": ""}):
            check(raises(lambda kw=kw: col.register(vol(**kw))), f"register refuses {kw}")
        check(col.list_volunteers() == [] and not os.path.exists(col.COLLECTED_DIR),
              "a refused registration writes nothing")


def test_ids_and_registration() -> None:
    check(bool(col.ID_RE.fullmatch(col.new_volunteer_id())), "new ids look like V-XXXXXX (hex)")
    with sandbox() as tmp:
        v = vol()
        path = col.register(v)
        check(bool(col.ID_RE.fullmatch(v.volunteer_id)), "register assigns a well-formed id")
        check(path == os.path.join(col.COLLECTED_DIR, v.volunteer_id) and os.path.isdir(path),
              "register returns the volunteer folder")
        check(v.consent_version == col.CONSENT_VERSION and bool(v.consent_time),
              "consent version and time are filled in")
        check(col.load_volunteer(v.volunteer_id) == v, "volunteer.json round-trips")
        txt = open(os.path.join(path, "consent.txt"), encoding="utf-8").read()
        check(col.CONSENT_TEXT in txt, "consent.txt holds the exact text shown")
        check(col.CONSENT_VERSION in txt and v.consent_time in txt and v.volunteer_id in txt,
              "consent.txt records version, time and id")
        check(all(w in col.CONSENT_TEXT for w in ("copyright", "licence", "withdraw", "blur",
                                                  "choice", "not be published, sold or shared")),
              "consent text covers copyright, licence, withdrawal, blur, voluntariness, no sale")
        check("name" in col.CONSENT_TEXT and "NOT store your name" in col.CONSENT_TEXT,
              "consent text says no name is stored")
        check(os.path.isfile(os.path.join(col.COLLECTED_DIR, ".gitignore")),
              "the collection folder protects itself from git")

        # ids are unique: a repeat draw is skipped, and an existing folder is never overwritten
        class _U:
            def __init__(self, h): self.hex = h
        draws = iter(["aaaaaa" + "0" * 26, "aaaaaa" + "0" * 26, "bbbbbb" + "0" * 26])
        real = col.uuid
        col.uuid = types.SimpleNamespace(uuid4=lambda: _U(next(draws)))
        try:
            a, b = vol(), vol()
            col.register(a)
            col.register(b)
        finally:
            col.uuid = real
        check(a.volunteer_id == "V-AAAAAA" and b.volunteer_id == "V-BBBBBB",
              "a drawn id that already exists is skipped")
        check(raises(lambda: col.register(vol(volunteer_id="V-AAAAAA")), FileExistsError),
              "an existing volunteer is never overwritten")
        ids = {col.register(x) for x in (vol() for _ in range(30))}
        check(len(ids) == 30, "30 registrations -> 30 distinct folders")
        check(len(col.list_volunteers()) == 33, "list_volunteers sees everyone")
        check(os.listdir(tmp) == ["collected"], "nothing is written outside the collection folder")


def test_samples_and_summary() -> None:
    frame, pose = noise_frame(1), make_pose()
    with sandbox():
        A, B, C = vol(age_group="18-24", sex="female", body_type="slim", face_policy="keep"), \
            vol(age_group="25-34", sex="male", body_type="athletic", face_policy="blur"), \
            vol(age_group="65+", sex="other", body_type="larger-built",
                face_policy="landmarks_only")
        for v in (A, B, C):
            col.register(v)
        pa = col.save_sample(A.volunteer_id, "tadasana", frame, pose, "right")
        pa2 = col.save_sample(A.volunteer_id, "tadasana", frame, pose, "right")
        pw = col.save_sample(A.volunteer_id, "tadasana", frame, pose, "wrong", "knees bent")
        col.save_sample(A.volunteer_id, "vrikshasana", frame, pose, "right")
        col.save_sample(B.volunteer_id, "tadasana", frame, pose, "right")
        col.save_sample(B.volunteer_id, "vrikshasana", frame, pose, "right")
        pc = col.save_sample(C.volunteer_id, "tadasana", frame, pose, "right")

        adir = os.path.join(col.COLLECTED_DIR, A.volunteer_id, "tadasana")
        check(pa.endswith("001.json") and pa2.endswith("002.json") and pw.endswith("003.json"),
              "samples are numbered 001, 002, 003 per pose")
        check(os.path.isfile(os.path.join(adir, "001.jpg")), "keep policy stores the jpg")
        check(read_image(os.path.join(adir, "001.jpg")).shape == frame.shape,
              "the stored jpg is a readable image of the same size")
        rec = read_json(pa)
        check(len(rec["landmarks"]) == 33 and set(rec["landmarks"][0]) == {"x", "y", "z", "visibility"},
              "33 landmarks with x, y, z, visibility")
        check(abs(rec["landmarks"][0]["z"] + 0.1) < 1e-3 and rec["landmarks"][NOSE]["x"] == 160.0,
              "landmark values are the pose's")
        check(isinstance(rec["features"], dict) and "knee_left" in rec["features"]
              and isinstance(rec["features"]["group_visibility"], dict),
              "features are the evaluator's (nested dicts kept)")
        check(rec["quality"] == "right" and rec["fault"] == "" and rec["detected"] is True
              and rec["image"] == "001.jpg" and rec["image_size"] == [320, 240]
              and rec["pose"] == "tadasana" and rec["volunteer_id"] == A.volunteer_id
              and rec["face_policy"] == "keep", "the record carries label, image and context")
        wrec = read_json(pw)
        check(wrec["quality"] == "wrong" and wrec["fault"] == "knees bent", "wrong + fault stored")
        pf = col.save_sample(A.volunteer_id, "vrikshasana", frame, pose, "right", "ignored")
        check(read_json(pf)["fault"] == "", "a fault note is dropped on a right sample")
        pf2 = col.save_sample(A.volunteer_id, "balasana", frame, pose, "wrong",
                              "  hips\x00 high\n" + "x" * 300)
        fault = read_json(pf2)["fault"]
        check(fault.startswith("hips high x") and len(fault) == col.MAX_FAULT_CHARS,
              "fault text is cleaned and capped")

        # NaN features must not leak into the JSON (it has no NaN)
        bad = make_pose(degenerate=True)
        pn = col.save_sample(A.volunteer_id, "tadasana", frame, bad, "right")
        raw = open(pn, encoding="utf-8").read()
        check("NaN" not in raw and "Infinity" not in raw and read_json(pn)["features"]["knee_left"] is None,
              "NaN features are written as null")

        # face policy
        bdir = os.path.join(col.COLLECTED_DIR, B.volunteer_id, "tadasana")
        check(os.path.isfile(os.path.join(bdir, "001.jpg")), "blur policy stores a jpg")
        region = col.head_region(pose)
        saved = read_image(os.path.join(bdir, "001.jpg"))
        inside, outside = circle_mask(frame.shape, region) > 0, circle_mask(frame.shape, region) == 0
        # JPEG itself smooths noise, so compare the outside with an unblurred JPEG, not the raw frame
        ref = cv2.imdecode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, col.JPEG_QUALITY])[1], 1)
        check(saved[inside].std() < 0.25 * ref[inside].std()
              and abs(saved[outside].std() - ref[outside].std()) < 0.05 * ref[outside].std(),
              "the stored 'blur' photo is blurred over the head only")
        cdir = os.path.join(col.COLLECTED_DIR, C.volunteer_id, "tadasana")
        check(sorted(os.listdir(cdir)) == ["001.json"], "landmarks_only stores no jpg")
        check(read_json(pc)["image"] is None and len(read_json(pc)["landmarks"]) == 33,
              "landmarks_only still stores the landmarks")
        check(raises(lambda: col.save_sample(C.volunteer_id, "tadasana", frame, None, "right"),
                     col.CollectionError), "landmarks_only with no body refuses to store")
        check(raises(lambda: col.save_sample(B.volunteer_id, "tadasana", frame, None, "right"),
                     col.CollectionError), "blur with no head located refuses to store")
        check(sorted(os.listdir(bdir)) == ["001.jpg", "001.json"], "refused samples leave nothing")
        pnone = col.save_sample(A.volunteer_id, "sukhasana", frame, None, "right")
        nrec = read_json(pnone)
        check(nrec["detected"] is False and nrec["landmarks"] is None and nrec["features"] is None
              and nrec["image"] == "001.jpg", "keep policy can store a frame with no body found")

        # input checks
        check(raises(lambda: col.save_sample(A.volunteer_id, "tadasana", frame, pose, "good")),
              "unknown quality rejected")
        check(raises(lambda: col.save_sample("V-FFFFFF", "tadasana", frame, pose, "right"),
                     col.CollectionError), "unknown volunteer rejected")
        check(raises(lambda: col.save_sample(A.volunteer_id, "tadasana", frame[:, :, 0], pose, "right")),
              "a non-BGR frame is rejected")
        check(raises(lambda: col.save_sample(A.volunteer_id, "tadasana", frame.astype(float), pose, "right")),
              "a non-uint8 frame is rejected")

        # delete (used by 'retake last')
        check(col.delete_sample(pw) and not os.path.exists(pw)
              and not os.path.exists(pw[:-5] + ".jpg"), "delete_sample removes the json and jpg")
        check(raises(lambda: col.delete_sample(os.path.join(os.path.dirname(col.COLLECTED_DIR),
                                                            "outside.json"))),
              "delete_sample refuses a path outside the collection")
        check(raises(lambda: col.delete_sample(os.path.join(col.COLLECTED_DIR, A.volunteer_id,
                                                            "volunteer.json"))),
              "delete_sample refuses volunteer.json")
        check(os.path.isfile(os.path.join(col.COLLECTED_DIR, A.volunteer_id, "volunteer.json")),
              "volunteer.json survives")

        # counts and diversity
        cnt = col.volunteer_counts(A.volunteer_id)
        check(cnt["tadasana"] == {"right": 3, "wrong": 0} and cnt["balasana"]["wrong"] == 1
              and set(cnt) == set(POSES), "volunteer_counts per pose and quality")
        s = col.summary()
        check(s.n_volunteers == 3, "summary counts volunteers")
        t = s.counts["tadasana"]["right"]
        check(t == {A.volunteer_id: 3, B.volunteer_id: 1, C.volunteer_id: 1},
              "counts per pose x quality x volunteer")
        check(s.volunteers_per_pose["tadasana"] == 3 and s.volunteers_per_pose["vrikshasana"] == 2
              and s.volunteers_per_pose["balasana"] == 0, "volunteers per pose")
        check(s.diversity["age_group"]["18-24"] == 1 and s.diversity["age_group"]["35-44"] == 0
              and s.diversity["sex"] == {"female": 1, "male": 1, "other": 1, "prefer not to say": 0}
              and s.diversity["body_type"]["slim"] == 1, "diversity counts per attribute")
        flags = "\n".join(s.flags)
        check("tadasana: only 3 volunteers" in flags and "vrikshasana: only 2 volunteers" in flags,
              "thin poses are flagged with their volunteer count")
        check("balasana: no right-labelled samples yet" in flags, "empty poses are flagged")
        check("age group: no volunteers in '35-44'" in flags, "missing age groups are flagged")
        check("sex: no volunteers in 'other'" not in flags and "prefer not to say" not in flags,
              "'other' / 'prefer not to say' are never flagged as missing")
        check("Volunteers: 3" in s.text() and "Needs attention" in s.text(), "summary text renders")
        check(col.summary().n_samples == sum(sum(sum(q.values()) for q in p.values())
                                             for p in s.counts.values()), "n_samples matches counts")

    with sandbox():
        check(col.summary().flags == ["No volunteers registered yet."], "empty collection summary")
        for _ in range(6):
            col.register(vol(sex="female", age_group="25-34"))
        flags = "\n".join(col.summary().flags)
        check("sex: 'female' is 6 of 6 volunteers (100%)" in flags, "a dominant category is flagged")


def test_blur() -> None:
    frame, pose = noise_frame(2), make_pose()
    before = frame.copy()
    region = col.head_region(pose)
    check(region is not None and abs(region[0] - 160) <= 2 and 35 <= region[1] <= 51
          and col.MIN_BLUR_RADIUS_PX <= region[2] <= 60,
          "head region sits on the head, nudged towards the crown (up), not towards the body")
    check(region[1] + region[2] >= 60 > 51, "the disc reaches below the chin")
    out = col.blur_faces(frame, pose)
    check(np.array_equal(frame, before), "blur_faces does not modify its input")
    check(out is not frame and out.shape == frame.shape and out.dtype == frame.dtype,
          "blur_faces returns a new image of the same shape")
    mask = circle_mask(frame.shape, region)
    diff = np.abs(out.astype(int) - frame.astype(int)).sum(axis=2)
    check(not diff[mask == 0].any(), "no pixel outside the head circle changes")
    check(diff[mask > 0].mean() > 30 and (diff[mask > 0] > 0).mean() > 0.9,
          "pixels inside the head circle are blurred")
    check(out[mask > 0].std() < 0.5 * frame[mask > 0].std(), "the head region lost its detail")
    same = col.blur_faces(frame, None)
    check(np.array_equal(same, frame) and same is not frame, "no pose -> an unchanged copy")
    check(col.head_region(None) is None, "no pose -> no head region")
    nan_pose = make_pose()
    nan_pose.pts[NOSE] = np.nan
    check(col.head_region(nan_pose) is None, "a non-finite head position is not guessed")
    # head partly above the frame, and another wholly outside: must not crash or touch the rest
    edge = make_pose(head=(160, 6))
    out2 = col.blur_faces(frame, edge)
    r2 = col.head_region(edge)
    d2 = np.abs(out2.astype(int) - frame.astype(int)).sum(axis=2)
    check(d2[circle_mask(frame.shape, r2) == 0].sum() == 0 and d2.any(),
          "a head at the frame edge is blurred and clipped safely")
    gone = make_pose(head=(160, -400))
    check(np.array_equal(col.blur_faces(frame, gone), frame), "a head outside the frame is a no-op")
    # side view: both ears on top of each other, the radius must not collapse
    side = make_pose()
    side.pts[L_EAR] = side.pts[R_EAR] = side.pts[NOSE] + np.array([-6.0, 2.0])
    check(col.head_region(side)[2] >= col.MIN_BLUR_RADIUS_PX, "side view keeps a usable radius")


def test_paths() -> None:
    frame, pose = noise_frame(3), make_pose()
    bad_ids = ["../x", "..\\x", "V-ABCDEF/../..", "v-abcdef", "V-12345", "V-1234567", "V-ABCDEG",
               "", "V-ABCDEF\n", None, "V-ABCDEF/", "V-ABCDEF\\..", "C:\\evil", "/etc/passwd"]
    bad_poses = ["../x", "tadasana/..", "", "TADASANA", "tadasana\x00", None, "..\\..\\evil",
                 "tadasana/", "x"]
    with sandbox() as tmp:
        v = vol()
        col.register(v)
        for bad in bad_ids:
            check(raises(lambda b=bad: col.withdraw(b)), f"withdraw rejects id {bad!r}")
            check(raises(lambda b=bad: col.save_sample(b, "tadasana", frame, pose, "right")),
                  f"save_sample rejects id {bad!r}")
            check(raises(lambda b=bad: col.load_volunteer(b)), f"load_volunteer rejects id {bad!r}")
            if bad:                                    # blank means "assign one", not "bad"
                check(raises(lambda b=bad: col.register(vol(volunteer_id=b))),
                      f"register rejects id {bad!r}")
        for bad in bad_poses:
            check(raises(lambda b=bad: col.save_sample(v.volunteer_id, b, frame, pose, "right")),
                  f"save_sample rejects pose {bad!r}")
        check(os.listdir(tmp) == ["collected"] and sorted(os.listdir(col.COLLECTED_DIR)) ==
              sorted([".gitignore", v.volunteer_id]), "no rejected input created anything")
        check(sorted(os.listdir(os.path.join(col.COLLECTED_DIR, v.volunteer_id))) ==
              ["consent.txt", "volunteer.json"], "no stray pose folder was created")
        check(raises(lambda: col.export_for_fitting(os.path.join(col.COLLECTED_DIR, v.volunteer_id))),
              "export refuses a destination inside the collection")
        check(raises(lambda: col.export_for_fitting(col.COLLECTED_DIR)),
              "export refuses the collection folder itself")
        # a hostile sample file name in the tree is never followed or counted
        os.makedirs(os.path.join(col.COLLECTED_DIR, "..evil"), exist_ok=True)
        check([x.volunteer_id for x in col.list_volunteers()] == [v.volunteer_id],
              "folders that are not volunteer ids are ignored")


def test_withdraw() -> None:
    frame, pose = noise_frame(4), make_pose()
    with sandbox():
        A, B = vol(), vol(age_group="45-54")
        col.register(A)
        col.register(B)
        for v in (A, B):
            for _ in range(2):
                col.save_sample(v.volunteer_id, "tadasana", frame, pose, "right")
        col.export_for_fitting(col.EXPORT_DIR)
        exp = os.path.join(col.EXPORT_DIR, "tadasana")
        check(len(os.listdir(exp)) == 4, "export starts with both volunteers' photos")
        check(col.withdraw(A.volunteer_id, also_in=col.EXPORT_DIR) is True, "withdraw returns True")
        check(not os.path.exists(os.path.join(col.COLLECTED_DIR, A.volunteer_id)),
              "the volunteer's whole folder is gone")
        check(sorted(os.listdir(exp)) == [f"{B.volunteer_id}_1.jpg", f"{B.volunteer_id}_2.jpg"],
              "their exported copies are gone too")
        check(os.path.isfile(os.path.join(col.COLLECTED_DIR, B.volunteer_id, "tadasana", "001.jpg")),
              "other volunteers are untouched")
        check([v.volunteer_id for v in col.list_volunteers()] == [B.volunteer_id]
              and col.summary().n_volunteers == 1, "the volunteer disappears from the summary")
        check(col.withdraw(A.volunteer_id) is False, "withdrawing twice returns False")
        check(col.withdraw("V-000000") is False, "an unknown (well-formed) id returns False")
        # a read-only file (Windows) must not block deletion
        p = col.save_sample(B.volunteer_id, "tadasana", frame, pose, "right")
        os.chmod(p, 0o444)
        check(col.withdraw(B.volunteer_id) is True and col.list_volunteers() == [],
              "a read-only file does not block a withdrawal")


def test_export() -> None:
    frame, pose = noise_frame(5), make_pose()
    with sandbox():
        A = vol(face_policy="keep")
        B = vol(face_policy="blur", age_group="55-64")
        C = vol(face_policy="landmarks_only", age_group="65+")
        for v in (A, B, C):
            col.register(v)
        save = col.save_sample
        save(A.volunteer_id, "tadasana", frame, pose, "right")
        save(A.volunteer_id, "tadasana", frame, pose, "right")
        save(A.volunteer_id, "tadasana", frame, pose, "wrong", "arms down")
        save(A.volunteer_id, "vrikshasana", frame, pose, "right")
        save(B.volunteer_id, "tadasana", frame, pose, "right")
        save(C.volunteer_id, "tadasana", frame, pose, "right")
        dest = os.path.join(os.path.dirname(col.COLLECTED_DIR), "fit")
        res = col.export_for_fitting(dest)
        check(res["per_pose"]["tadasana"] == 3 and res["per_pose"]["vrikshasana"] == 1
              and res["total"] == 4 and res["volunteers"] == 2 and res["skipped_no_image"] == 1
              and res["per_pose"]["balasana"] == 0, "export counts")
        check(sorted(os.listdir(os.path.join(dest, "tadasana"))) ==
              sorted([f"{A.volunteer_id}_1.jpg", f"{A.volunteer_id}_2.jpg", f"{B.volunteer_id}_1.jpg"]),
              "folder-per-pose, files named <volunteer_id>_<n>.jpg, right-labelled only")
        check(sorted(d for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d))) ==
              ["tadasana", "vrikshasana"], "only poses with photos get a folder")
        check(read_image(os.path.join(dest, "tadasana", f"{B.volunteer_id}_1.jpg")) is not None,
              "exported files are readable images")
        check(os.path.isfile(os.path.join(dest, ".gitignore")),
              "a newly created export folder protects itself from git")
        # mirrors the collection and leaves foreign files alone
        with open(os.path.join(dest, "tadasana", "notes.txt"), "w") as fh:
            fh.write("mine")
        shutil.copyfile(os.path.join(dest, "tadasana", f"{A.volunteer_id}_1.jpg"),
                        os.path.join(dest, "tadasana", "mine.jpg"))
        col.withdraw(B.volunteer_id)
        res2 = col.export_for_fitting(dest)
        check(sorted(os.listdir(os.path.join(dest, "tadasana"))) ==
              sorted(["mine.jpg", "notes.txt", f"{A.volunteer_id}_1.jpg", f"{A.volunteer_id}_2.jpg"])
              and res2["per_pose"]["tadasana"] == 2, "re-export mirrors the collection, foreign files stay")

    # the layout is the one tools/fitting/fit_asana.py reads: run its own fit on the export
    with sandbox():
        v = vol()
        col.register(v)
        for i in range(30):
            col.save_sample(v.volunteer_id, "tadasana", noise_frame(i), pose, "right")
        dest = os.path.join(os.path.dirname(col.COLLECTED_DIR), "fit")
        col.export_for_fitting(dest)
        fit_asana = importlib.import_module("tools.fitting.fit_asana")

        class _Tracker:
            def process(self, rgb, *a, **k):
                return make_pose(w=rgb.shape[1], h=rgb.shape[0])

        checks, used, skipped = fit_asana.fit_class(_Tracker(), os.path.join(dest, "tadasana"))
        check(used == 30 and skipped == 0 and "spine_tilt" in checks,
              "fit_asana.fit_class reads the exported folder")


class FakeCap:
    """Delivers the same frame forever, like a camera on a still scene."""

    def __init__(self, frame, fail: bool = False):
        self.frame, self.fail, self.released = frame, fail, False
        self.reads = 0

    def read(self):
        time.sleep(0.004)
        self.reads += 1
        return (False, None) if self.fail else (True, self.frame)

    def release(self):
        self.released = True


class FakeTracker:
    """Returns a scripted pose (or None) per call and records what it was given."""

    def __init__(self, script):
        self.script, self.calls, self.seen = list(script), 0, []

    def process(self, rgb, *a, **k):
        self.seen.append(rgb)
        self.calls += 1
        return self.script[(self.calls - 1) % len(self.script)]


def test_burst() -> None:
    frame = noise_frame(6)
    frame[:, 0] = (0, 0, 255)                             # a red column on the left edge
    pose = make_pose()
    cap, trk = FakeCap(frame), FakeTracker([pose, None, pose])
    t0 = time.perf_counter()
    shots = collect.grab_burst(cap, trk, n=3, countdown_s=0.1, interval_s=0.05)
    took = time.perf_counter() - t0
    check(len(shots) == 3 and 0.19 <= took < 1.5, "a burst takes n photos after the countdown")
    check([s.pose is not None for s in shots] == [True, False, True], "each photo carries its own pose")
    check(trk.calls == 3 and np.array_equal(trk.seen[0][..., 0], frame[..., 2]),
          "the tracker is given RGB, once per photo when there is no preview")
    check(shots[0].frame is not frame and np.array_equal(shots[0].frame, frame),
          "photos are copies of the raw, un-mirrored frames")
    check(tuple(shots[0].frame[100, 0]) == (0, 0, 255), "the saved frame is not mirrored")

    shown = []

    def show(img):
        shown.append(img)
        return True
    collect.grab_burst(FakeCap(frame), FakeTracker([None]), n=2, countdown_s=0.08,
                       interval_s=0.03, show=show, caption="Tree")
    check(len(shown) >= 3 and shown[0].shape == frame.shape, "the preview is drawn every frame")
    check(tuple(shown[0][100, -1]) == (0, 0, 255) and tuple(shown[0][100, 0]) != (0, 0, 255),
          "the preview is mirrored")
    check(not np.array_equal(shown[0], frame[:, ::-1]), "the preview carries an overlay")
    check(collect.grab_burst(FakeCap(frame), FakeTracker([None]), n=3, countdown_s=0.05,
                             interval_s=0.02, show=lambda img: False) == [],
          "Esc during the countdown cancels with no photos")
    count = []
    got = collect.grab_burst(FakeCap(frame), FakeTracker([None]), n=3, countdown_s=0.0,
                             interval_s=0.03, show=lambda img: count.append(1) or len(count) < 3)
    check(len(got) < 3, "cancelling mid-burst returns the photos taken so far")
    old = collect.MAX_READ_FAILURES
    collect.MAX_READ_FAILURES = 3
    try:
        check(raises(lambda: collect.grab_burst(FakeCap(frame, fail=True), FakeTracker([None]),
                                                n=1, countdown_s=0.0), RuntimeError),
              "a camera that stops delivering frames raises")
    finally:
        collect.MAX_READ_FAILURES = old
    note, ok = collect.coverage_note(None)
    check(not ok and "No body" in note, "coverage note: no body")
    check(collect.coverage_note(pose) == ("Body detected - whole body in the picture", True),
          "coverage note: whole body inside the picture")
    cut = make_pose()
    cut.pts[L_ANKLE] = (160, 400)                         # ankle below the 240 px frame
    check(collect.joints_outside(cut) == [L_ANKLE] and collect.coverage_note(cut)[1] is False
          and "outside the picture" in collect.coverage_note(cut)[0],
          "coverage note: a joint outside the picture")
    low = make_pose()
    low.vis[L_KNEE] = 0.1                                 # self-occluded, still in frame
    check(collect.coverage_note(low)[1] is True, "low BlazePose visibility alone is not a warning")


def test_app() -> None:
    trainer = importlib.import_module("app")
    check(callable(getattr(trainer, "open_source", None)) and callable(getattr(trainer, "window_closed", None)),
          "app.open_source / app.window_closed exist for collect.py to reuse")
    try:
        app = collect.App()
    except Exception as exc:                              # no display (CI / Docker)
        print(f"  skip  collect.App screens ({type(exc).__name__}: {exc})")
        return
    with sandbox():
        try:
            app.update()
            warned: list[tuple[str, str]] = []
            infos: list[str] = []
            app._warn = lambda t, m: warned.append((t, m))
            app._info = lambda t, m: infos.append(m)
            answers = {"confirm": False}
            app._confirm = lambda t, m: answers["confirm"]

            # ---- consent screen
            check(app.screen == "consent", "the app opens on the consent screen")
            shown_text = app.consent_view.get("1.0", "end")
            check(col.CONSENT_TEXT.strip() in shown_text.strip(), "the whole consent text is shown")
            check(str(app.consent_view.cget("state")) == "disabled", "the consent box is read-only")
            check(app.consent_btn.instate(["disabled"]), "continue is disabled before any box is ticked")
            app.consent_ok.set(True)
            app._sync_consent()
            check(app.consent_btn.instate(["disabled"]), "one box is not enough")
            app._accept_consent()
            check(app.screen == "consent" and len(warned) == 1, "cannot continue without both boxes")
            app.healthy_ok.set(True)
            app._sync_consent()
            check(app.consent_btn.instate(["!disabled"]), "continue enables when both are ticked")
            app.face_policy.set("landmarks_only")
            app._accept_consent()
            check(app.screen == "details" and bool(app.consent_time), "consent -> details screen")

            # ---- details screen
            app._submit_details()
            check(app.screen == "details" and len(warned) == 2 and col.list_volunteers() == [],
                  "blank details are refused and nothing is registered")
            for k, val in (("age_group", "25-34"), ("sex", "female"), ("body_type", "average"),
                           ("experience", "beginner"), ("height", "abc")):
                app.v[k].set(val)
            app._submit_details()
            check(app.screen == "details" and len(warned) == 3, "a non-numeric height is refused")
            app.v["height"].set("165")
            app.v["weight"].set("58,5")
            app._submit_details()
            check(app.screen == "id" and app.volunteer is not None, "details -> volunteer id screen")
            vid = app.volunteer.volunteer_id
            stored = col.load_volunteer(vid)
            check(stored is not None and stored.face_policy == "landmarks_only"
                  and stored.weight_kg == 58.5 and stored.consent and stored.healthy_confirmed,
                  "the chosen face policy and details were registered")
            labels = [w.cget("text") for w in app.body.winfo_children() if isinstance(w, collect.ttk.Label)]
            check(vid in labels, "the volunteer's id is shown prominently")

            # ---- capture screen with a fake camera
            app.show_capture()
            app.update()
            check(app.screen == "capture" and vid in [w.cget("text") for w in
                  app.header.winfo_children() if isinstance(w, collect.ttk.Label)][-2],
                  "the id stays in the capture header")
            check(len(app.counts.get_children()) == len(POSES), "the counts table lists all poses")
            check(app._fault_entry.instate(["disabled"]), "the fault box is off for a right attempt")
            app.quality.set("wrong")
            check(app._fault_entry.instate(["!disabled"]), "the fault box opens for a wrong attempt")
            app.quality.set("right")
            app.burst_n, app.countdown_s, app.interval_s = 3, 0.03, 0.02
            frame, pose = noise_frame(7), make_pose()
            cap = FakeCap(frame)
            app._open_camera = lambda: (cap, True)
            app._get_tracker = lambda: FakeTracker([pose, pose, None])
            app._preview_fn = lambda: None
            app.select_pose("vrikshasana")
            app.start_capture()
            app.update()
            res = app.last_result
            check(res["shots"] == 3 and res["detected"] == 2 and res["saved"] == 2 and cap.released,
                  "a capture saves the photos with a body and releases the camera")
            check("2 of 3" in app._status[0] and app.state() == "normal",
                  "the volunteer is told how many were detected; the window is back")
            vdir = os.path.join(col.COLLECTED_DIR, vid, "vrikshasana")
            check(sorted(os.listdir(vdir)) == ["001.json", "002.json"],
                  "points-only volunteer: json saved, no jpg")
            check(col.volunteer_counts(vid)["vrikshasana"]["right"] == 2
                  and app.counts.item(app.counts.get_children()[4])["values"][1] in (2, "2"),
                  "the counts table updates")
            app.retake_last()
            check(os.listdir(vdir) == [] and col.volunteer_counts(vid)["vrikshasana"]["right"] == 0,
                  "retake last deletes the previous attempt")
            app.quality.set("wrong")
            app.fault.set("knee bent")
            app.start_capture()
            wrong = [read_json(os.path.join(vdir, f)) for f in sorted(os.listdir(vdir))]
            check(len(wrong) == 2 and all(r["quality"] == "wrong" and r["fault"] == "knee bent"
                                          for r in wrong), "a wrong attempt keeps its label and fault")
            app.next_pose()
            check(app._pose_key() == "adho_mukha" and app.quality.get() == "right"
                  and app.fault.get() == "", "next pose advances and resets the label")
            app._open_camera = lambda: (_ for _ in ()).throw(SystemExit("Camera 0 could not be used."))
            nwarn = len(warned)
            app.start_capture()
            check(len(warned) == nwarn + 1 and "Camera 0" in warned[-1][1] and app.state() == "normal",
                  "a missing camera is reported and the session survives")
            app.pose_var.set("")
            check(app._pose_key() == "tadasana", "an empty pose box falls back to the first pose")

            # ---- summary, export, withdraw
            app.show_summary()
            app.update()
            check(app.screen == "summary" and "Volunteers: 1" in app.summary_view.get("1.0", "end"),
                  "the summary screen shows the diversity report")
            app.export_photos()
            check(len(infos) == 1 and "Copied 0 right-labelled" in infos[0],
                  "export reports (points-only volunteers have no photos)")
            collect.simpledialog.askstring = lambda *a, **k: vid.lower() + " "
            app.ask_withdraw()
            check(col.load_volunteer(vid) is not None, "withdrawal needs confirmation")
            answers["confirm"] = True
            app.ask_withdraw()
            check(col.list_volunteers() == [] and not os.path.exists(os.path.join(col.COLLECTED_DIR, vid)),
                  "withdrawing by id (any case) deletes the volunteer")
            check(app.screen == "consent" and not app.consent_ok.get() and app.volunteer is None,
                  "after withdrawing the current volunteer the app returns to a blank consent screen")
            ok, msg = app.do_withdraw("V-ABCDEF")
            check(ok is False and "No volunteer" in msg, "do_withdraw: unknown id")
            ok, msg = app.do_withdraw("../x")
            check(ok is False and "Could not" in msg, "do_withdraw: malformed id")
            nwarn = len(warned)
            collect.simpledialog.askstring = lambda *a, **k: "not an id"
            app.ask_withdraw()
            check(len(warned) == nwarn + 1, "ask_withdraw rejects a malformed id")

            # ---- a second volunteer who accepts blurring, through the same screens
            app.consent_ok.set(True)
            app.healthy_ok.set(True)
            app.face_policy.set("blur")
            app._accept_consent()
            for k, val in (("age_group", "55-64"), ("sex", "male"), ("body_type", "broad-built"),
                           ("experience", "regular")):
                app.v[k].set(val)
            app._submit_details()
            vid2 = app.volunteer.volunteer_id
            check(vid2 != vid, "a new volunteer gets a new id")
            app.show_capture()
            app._open_camera = lambda: (FakeCap(frame), True)
            app.select_pose("balasana")
            app.start_capture()
            bdir = os.path.join(col.COLLECTED_DIR, vid2, "balasana")
            check(sorted(os.listdir(bdir)) == ["001.jpg", "001.json", "002.jpg", "002.json"],
                  "a blur volunteer gets blurred jpgs through the app")
            app.export_photos()
            check("Copied 2 right-labelled photos from 1 volunteers" in infos[-1]
                  and "fit_asana.py" in infos[-1] and "--out" in infos[-1],
                  "export reports counts and the refit command (to a separate file)")
            app.new_volunteer()
            check(app.screen == "consent" and not app.healthy_ok.get() and app.v["sex"].get() == "",
                  "nothing carries over to the next volunteer")
        finally:
            app.destroy()


def main() -> int:
    real_root = col.COLLECTED_DIR
    existed = os.path.exists(real_root)
    for fn in (test_consent_gate, test_ids_and_registration, test_samples_and_summary, test_blur,
               test_paths, test_withdraw, test_export, test_burst, test_app):
        print(fn.__name__)
        fn()
    check(col.COLLECTED_DIR == real_root and os.path.exists(real_root) == existed,
          "the tests never touched the real data/collected folder")
    print(f"\n{N - len(FAILS)}/{N} collection checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
