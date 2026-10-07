"""Offline checks for the taxonomy and the 'Ask the coach' NLP layer.

    python tools/test_nlp.py        (no camera, no network, no model downloads)

Honesty note on the accuracy numbers it prints.  ``DEV`` is a development set
the module's author wrote *while* building the lexicons, so it is optimistic -
it measures "does the code do what the author meant", not generalisation.
``BLIND`` was written after the lexicons were frozen and was scored once before
any tuning; one rule (spine conditions -> back-pain routine) and two bugs
(stemmer ordering, a double-counted mood word) were fixed after seeing its
failures, so it is no longer fully blind either.  Both sets are small, both are
written by the same person, and neither is a substitute for utterances from
real users.
"""

from __future__ import annotations

import dataclasses
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yoga import nlp                                              # noqa: E402
from yoga import profile as prof                                  # noqa: E402
from yoga import routines as rt                                   # noqa: E402
from yoga import taxonomy as tx                                   # noqa: E402

FAILS: list[str] = []
N = 0


def check(cond: bool, label: str) -> None:
    global N
    N += 1
    if not cond:
        FAILS.append(label)
        print(f"  FAIL  {label}")


def base(**kw) -> prof.Profile:
    d = dict(name="Asha", age=30, weight_kg=60, height_cm=165, waist_cm=75, hip_cm=95,
             bp="None", job="sitting", diet="Vegetarian", location="Thane")
    d.update(kw)
    return prof.Profile(**d)


# ------------------------------------------------------------------ data
#: (utterance, the routine a sensible teacher would pick).  Development set.
DEV = [
    ("my lower back hurts after sitting all day, I only have 10 minutes", "back_pain"),
    ("I work at a desk for 9 hours and my hips feel tight", "desk_sitting"),
    ("I'm a teacher and my legs ache by evening", "standing_workers"),
    ("my knees hurt from standing at the shop all day", "standing_workers"),
    ("I'm studying for exams and my neck is stiff", "students"),
    ("too much laptop and phone, I have text neck and rounded shoulders", "students"),
    ("I feel stressed and anxious about work", "stress_relief"),
    ("I can't sleep at night, my mind keeps racing", "stress_relief"),
    ("I want to relax and unwind", "stress_relief"),
    ("I feel tired and have no energy in the morning", "daily_energy"),
    ("I want a full body workout to start my day", "daily_energy"),
    ("feeling lazy and sluggish today", "daily_energy"),
    ("I am a homemaker, cooking and cleaning all day, and my back aches", "homemakers"),
    ("housewife with a lot of household chores, tired and sore", "homemakers"),
    ("I want to improve my flexibility and core strength", "lumbar_flex"),
    ("my hamstrings are tight and I can't touch my toes", "lumbar_flex"),
    ("slipped disc history, I need gentle lumbar care", "back_pain"),
    ("I only want standing poses", "pos_standing"),
    ("something I can do sitting in a chair", "pos_sitting"),
    ("I want floor poses on the mat", "pos_floor"),
    ("backache", "back_pain"),
    ("software developer, long hours on the computer, my back is stiff", "desk_sitting"),
    ("I stand all day at work and my feet are swollen", "standing_workers"),
    ("kamar mein dard hai", "back_pain"),
    ("gardan mein dard hai", "students"),
    ("bahut thakan hai", "daily_energy"),
    ("tanav hai aur neend nahi aati", "stress_relief"),
    ("ghutne mein dard", "standing_workers"),
    ("I'm exhausted after the night shift", "daily_energy"),
    ("I feel really sad and unmotivated", "daily_energy"),
    ("my shoulders and neck are tense from typing at my desk", "desk_sitting"),
    ("college student with long lectures, sleepy and can't focus", "students"),
    ("I get panic attacks and feel overwhelmed", "stress_relief"),
    ("my calves are sore from standing at the factory", "standing_workers"),
    ("need a quick 5 minute stretch for my stiff back", "back_pain"),
    ("I want to try child's pose and easy pose to calm down", "stress_relief"),
    ("I feel strong and full of energy, give me a challenge", "daily_energy"),
    ("my lower back is tight from lifting kids and doing housework", "homemakers"),
    ("I have high blood pressure and a desk job", "desk_sitting"),
    ("work from home, back and hip pain", "desk_sitting"),
    ("I'm a security guard standing for 10 hours with lower back pain", "standing_workers"),
    ("teacher with knee pain and arthritis", "standing_workers"),
    ("mujhe bahut tanav hai", "stress_relief"),
    ("just something general to stay fit", "daily_energy"),
]

#: Written after the lexicons were frozen; see the module docstring for its limits.
BLIND = [
    ("sitting at my computer 8 hours a day is killing my spine", "desk_sitting"),
    ("I feel anxious before my presentations", "stress_relief"),
    ("need more energy in the afternoons", "daily_energy"),
    ("I work as a nurse on long shifts and my ankles swell", "standing_workers"),
    ("my neck is killing me from studying", "students"),
    ("sore lower back", "back_pain"),
    ("I cook for the whole family every day and my feet hurt", "homemakers"),
    ("want to get more supple and be able to bend forward", "lumbar_flex"),
    ("I'm overwhelmed with deadlines", "stress_relief"),
    ("can you give me poses I can do lying down", "pos_floor"),
    ("I'm worn out and drained", "daily_energy"),
    ("sciatica pain down my leg", "back_pain"),
    ("typing all day has made my wrists and shoulders ache", "desk_sitting"),
    ("I have to stand behind the counter all day at my shop", "standing_workers"),
    ("feeling low and unmotivated, need a boost", "daily_energy"),
    ("my mind won't switch off at night", "stress_relief"),
    ("I am revising for my board exams and feel stiff all over", "students"),
    ("tight hips from sitting", "desk_sitting"),
    ("back pain", "back_pain"),
    ("do something calming for me please", "stress_relief"),
    ("my upper back and neck are stiff from the phone", "students"),
    ("knee pain when I climb stairs", "standing_workers"),
    ("I've been feeling really down lately", "daily_energy"),
    ("poses I can do standing up, no mat", "pos_standing"),
    ("something quiet and seated where I just breathe", "pos_sitting"),
    ("I sleep badly and wake up tired", "daily_energy"),
    ("my husband says I slouch at my desk", "desk_sitting"),
    ("stiff lower back every morning", "back_pain"),
    ("I'm a driver and sit for long hours, my back hurts", "desk_sitting"),
]


def accuracy(cases: list[tuple[str, str]]) -> tuple[float, float, list[str]]:
    top1 = top2 = 0
    misses = []
    for text, want in cases:
        keys = [k for k, _ in nlp.understand(text).ranking]
        top1 += keys[0] == want
        top2 += want in keys[:2]
        if want not in keys[:2]:
            misses.append(f"{text!r} wanted {want}, got {keys[:3]}")
    return top1 / len(cases), top2 / len(cases), misses


# ------------------------------------------------------------------ tests
def test_taxonomy() -> None:
    check(tx.validate() == [], f"taxonomy validates: {tx.validate()}")
    check(list(tx.POSITIONS.values()) == ["Standing", "Seated", "All-fours & kneeling",
                                          "Prone (face-down)"], "four body positions")
    check(len(tx.CATEGORIES) >= 8, "at least eight functional categories")
    for key, pose in rt.POSES.items():
        homes = [c.key for c in tx.CATEGORIES.values() if key in c.poses]
        check(len(homes) == 1, f"{key}: in exactly one category (got {homes})")
        check(tx.position_of(key) in tx.POSITIONS, f"{key}: has a body position")
        check(tx.category_of(key).position == tx.position_of(key), f"{key}: position agrees")
    check(tx.position_of("vrikshasana") == "standing"
          and tx.category_of("vrikshasana").key == "balance", "Tree Pose = standing/balance")
    check(tx.category_of("adho_mukha").key == "inversion"
          and tx.position_of("adho_mukha") == "all_fours_kneeling",
          "Downward Dog = all-fours/inversion")
    check(tx.position_of("bhujangasana") == "prone"
          and tx.category_of("bhujangasana").key == "backbend", "Cobra = prone/backbend")
    check(tx.position_of("sukhasana") == "seated", "Easy Pose is seated")
    check(all(c.purpose.strip().endswith(".") for c in tx.CATEGORIES.values()),
          "every category has a one-sentence purpose")
    try:
        tx.category_of("not_a_pose")
        check(False, "category_of rejects an unknown pose")
    except KeyError:
        check(True, "category_of rejects an unknown pose")

    for key in rt.POSES:
        pts = tx.body_points(key)
        ids = [i for i, _ in pts]
        check(pts and all(0 <= i <= 32 for i in ids), f"{key}: landmark ids in 0..32")
        check(11 in ids and 23 in ids, f"{key}: shoulders and hips are always body points")
        check(all(isinstance(n, str) and n for _, n in pts), f"{key}: points are named")
        feats = tx.features_for(key)
        check(feats and all(f in tx.FEATURES for f in feats), f"{key}: features described")
        check(set(tx.measured_features(key)) <= set(feats),
              f"{key}: everything scored today is listed")
    check((11, "left shoulder") in tx.body_points("tadasana"), "landmark 11 is the left shoulder")
    check({"standing_knee", "foot_height_ratio"} <= set(tx.features_for("vrikshasana")),
          "Tree Pose lists its one-legged-balance features")
    check("foot_height_ratio" not in tx.features_for("balasana"),
          "features are not shared across unrelated categories")

    # the model must be a tree a Treeview can show
    t = tx.tree()
    check(list(t) == list(tx.POSITIONS.values()), "tree() roots are the four positions")
    labels = []

    def walk(node: dict, depth: int = 0) -> None:
        for text, kids in node.items():
            check(isinstance(text, str) and isinstance(kids, dict), "tree node is {str: dict}")
            if depth == 2:
                labels.append(text)
            walk(kids, depth + 1)

    walk(t)
    check(len(labels) == len(rt.POSES), f"every pose is a level-3 node ({len(labels)})")
    check(all(any(p.sanskrit in lab for lab in labels) for p in rt.POSES.values()),
          "every pose appears by name in the tree")
    pose_node = t["Standing"]["Balance"]["Tree Pose (Vrikshasana)"]
    check(set(pose_node) == {"Body points", "Joint angles / features"}, "level 4 under a pose")
    check(any("left shoulder" in k for k in pose_node["Body points"]), "body points listed")
    check(all(v == {} for v in tx.tree(details=False)["Standing"]["Balance"].values()),
          "tree(details=False) stops at the pose")
    text = tx.describe_tree()
    check(all(p.name in text for p in rt.POSES.values()) and "Prone" in text,
          "describe_tree names every pose and position")
    check(tx.pose_path("balasana") == "All-fours & kneeling > Forward fold & rest > Child's Pose",
          "pose_path reads root to leaf")

    # validate() must actually fire - feed it deliberately broken models
    cats = dict(tx.CATEGORIES)
    bal = cats["balance"]
    bad = {**cats, "balance": dataclasses.replace(bal, poses=("vrikshasana", "no_such_pose"))}
    check(any("unknown pose key" in p for p in tx.validate(bad)), "validate: unknown pose key")
    bad = {**cats, "balance": dataclasses.replace(bal, poses=())}
    check(any("vrikshasana" in p and "no category" in p for p in tx.validate(bad)),
          "validate: pose in zero categories")
    bad = {**cats, "balance": dataclasses.replace(bal, poses=("vrikshasana", "tadasana"))}
    check(any("tadasana" in p and "2 categories" in p for p in tx.validate(bad)),
          "validate: pose in two categories")
    bad = {**cats, "balance": dataclasses.replace(bal, features=bal.features + ("not_a_feature",))}
    check(any("not_a_feature" in p for p in tx.validate(bad)), "validate: invented feature")
    bad = {**cats, "balance": dataclasses.replace(bal, points=bal.points + (99,))}
    check(any("99" in p for p in tx.validate(bad)), "validate: landmark id out of range")
    bad = {**cats, "balance": dataclasses.replace(bal, points=(11, 12))}
    check(any("needs landmarks" in p for p in tx.validate(bad)),
          "validate: a feature whose landmarks are not listed")
    bad = {**cats, "balance": dataclasses.replace(bal, position="upside_down")}
    check(any("unknown position" in p for p in tx.validate(bad)), "validate: unknown position")


def test_tokenise() -> None:
    check(nlp.tokenize("My lower back is hurting!") == ["lower", "back", "hurt"],
          f"tokenise + stem + stop words: {nlp.tokenize('My lower back is hurting!')}")
    check(nlp.tokenize("") == [] and nlp.tokenize("   ...  ") == [], "empty text -> no tokens")
    pairs = [("sitting", "sits"), ("stretches", "stretching"), ("tired", "tire"),
             ("aching", "aches"), ("anxieties", "anxiety"), ("studies", "studying"),
             ("tiredness", "tired"), ("worried", "worries")]
    for a, b in pairs:
        check(nlp.stem(a) == nlp.stem(b), f"stem({a}) == stem({b}) -> {nlp.stem(a)}/{nlp.stem(b)}")
    check(nlp.stem("sitting") == "sit" and nlp.stem("stress") == "stress"
          and nlp.stem("pain") == "pain", "stemmer leaves short roots alone")
    check(nlp.stem("knees") == nlp.stem("knee"), "plural knee")
    check("not" not in nlp.tokenize("I do not have time") and nlp.tokenize("the and of") == [],
          "stop words are dropped")
    check(nlp.tokenize("Can't sleep") == ["sleep"], "contractions are expanded")
    check(nlp.tokenize("BACK PAIN") == nlp.tokenize("back pain"), "case-insensitive")
    # Hinglish lexicon, one word at a time
    check(nlp.tokenize("kamar") == ["lower", "back"], "kamar -> lower back")
    check(nlp.tokenize("dard") == ["pain"], "dard -> pain")
    check(nlp.tokenize("gardan") == ["neck"], "gardan -> neck")
    check(nlp.tokenize("ghutna") == ["knee"] and nlp.tokenize("ghutne") == ["knee"],
          "ghutna/ghutne -> knee")
    check(nlp.tokenize("thakan") == [nlp.stem("tired")], "thakan -> tired")
    check(nlp.tokenize("tanav") == ["stress"], "tanav -> stress")
    check(nlp.tokenize("neend") == ["sleep"], "neend -> sleep")
    check(nlp.tokenize("mujhe kamar mein dard hai") == ["lower", "back", "pain"],
          "Hinglish function words are dropped")


def test_entities() -> None:
    e = nlp.extract_entities("my lower back hurts after sitting all day, I only have 10 minutes")
    check(e["body_part"] == ["lower back"], f"lower back is one body part, not 'back': {e['body_part']}")
    check(e["symptom"] == ["pain"] and e["job"] == ["sitting"], "pain + sitting job")
    check(e["duration"] == ["10 minutes"], "duration text")
    e = nlp.extract_entities("pain in my neck, shoulder and right knee")
    check(e["body_part"] == ["neck", "shoulder", "knee"], f"several body parts: {e['body_part']}")
    check(nlp.extract_entities("my back and upper back hurt")["body_part"] == ["back", "upper back"],
          "upper back is its own part")
    check(nlp.extract_entities("I am coming back to work soon")["body_part"] == [],
          "'back' as an adverb is not a body part")
    check(nlp.extract_entities("hamstrings and calves are tight")["body_part"] == ["leg"],
          "hamstring/calf map to leg")
    check("stiffness" in nlp.extract_entities("my hips feel stiff")["symptom"], "stiffness symptom")
    check(nlp.extract_entities("no pain at all")["symptom"] == [], "negated symptom is ignored")

    for text, minutes, dur in [
        ("I only have 10 minutes", 10, "10 minutes"),
        ("I have half an hour", 30, "half an hour"),
        ("just 2 min", 2, "2 min"),
        ("ten minutes please", 10, "ten minutes"),
        ("a quarter of an hour", 15, "a quarter of an hour"),
        ("I have an hour", 60, "an hour"),
        ("5-10 minutes", 5, "5-10 minutes"),
        ("I have a few minutes", 5, "a few minutes"),
        ("quick session please", nlp.QUICK_MINUTES, "quick"),
        ("give me a short one", nlp.QUICK_MINUTES, "short"),
        ("15mins after work", 15, "15mins"),
    ]:
        check(nlp.parse_minutes(text) == minutes, f"minutes({text!r}) = {nlp.parse_minutes(text)}")
        check(dur in nlp.extract_entities(text)["duration"], f"duration entity for {text!r}")
    # how long you sit is not how long you have
    check(nlp.parse_minutes("I sit at a desk for 9 hours") is None, "9 hours of sitting is not time available")
    check(nlp.extract_entities("I sit at a desk for 9 hours")["exposure"] == ["9 hours"],
          "...it is recorded as exposure")
    check(nlp.parse_minutes("after sitting for 2 hours I only have 15 mins") == 15,
          "time available beats time sat")
    check(nlp.parse_minutes("I stand 6 hours a day") is None, "6 hours a day is exposure")
    check(nlp.parse_minutes("I feel short of breath") is None, "'short of breath' is not a short session")
    check(nlp.parse_minutes("hello") is None, "no duration -> None")

    for text, cond in [
        ("I have diabetes", "diabetes"), ("I am diabetic", "diabetes"),
        ("I have high blood pressure", "high_bp"), ("my BP is high", "high_bp"),
        ("I have hypertension", "high_bp"), ("my blood pressure is low", "low_bp"),
        ("I am pregnant", "pregnancy"), ("I'm in my second trimester", "pregnancy"),
        ("I have arthritis", "arthritis"), ("I have asthma", "asthma"),
        ("history of slip disc", "slip_disc"), ("I had a slipped disc", "slip_disc"),
        ("I get migraines", "migraine"),
    ]:
        check(cond in nlp.extract_entities(text)["condition"],
              f"condition {cond!r} in {text!r} -> {nlp.extract_entities(text)['condition']}")
    check(nlp.extract_entities("I don't have diabetes")["condition"] == [], "negated condition ignored")
    check(nlp.extract_entities("my bp is normal")["condition"] == [], "normal BP is not a condition")
    check(nlp.extract_entities("I take sugar in my tea")["condition"] == [], "'sugar' alone is not diabetes")
    check(nlp.extract_entities("I had surgery 5 years ago")["condition"] == [], "old surgery ignored")

    jobs = {"I sit at a desk in an office": "sitting", "I use my laptop all day": "sitting",
            "I'm a teacher": "standing", "I run a shop": "standing", "I work in a kitchen": "homemaker",
            "I'm a housewife": "homemaker", "I am a student": "student", "exam season": "student"}
    for text, job in jobs.items():
        check(job in nlp.extract_entities(text)["job"], f"job {job!r} in {text!r}")
    check(nlp.extract_entities("I only want standing poses")["job"] == [],
          "'standing poses' is a position request, not a job")
    check(nlp.extract_entities("I only want standing poses")["position_request"] == ["standing"],
          "position request recognised")
    check(nlp.extract_entities("I want to be more flexible and relax")["goal"] == ["flexibility", "relax"],
          "goals recognised")


def test_mood() -> None:
    m = lambda t: nlp.analyse_mood(t)[0]                                # noqa: E731
    check(m("I am so stressed and anxious") == "stressed", "stressed")
    check(m("I feel really tired") == "tired" and m("I'm exhausted") == "tired", "tired")
    check(m("I feel sad and lonely") == "low", "low")
    check(m("I feel full of energy today") == "energetic", "energetic")
    check(m("I feel calm and relaxed") == "calm", "calm")
    check(m("the weather is nice") == "neutral" and m("") == "neutral", "neutral")
    # negation
    check(m("I am not stressed") != "stressed", "'I am not stressed' is not stressed")
    check(m("I am not stressed") == "neutral", "...it is neutral")
    check(m("I'm not tired at all") != "tired", "not tired")
    check(m("I don't feel sad") != "low", "don't feel sad")
    check(m("there is no stress in my life") != "stressed", "no stress")
    check(m("not stressed but tired") == "tired", "negation stops at 'but'")
    check(m("I am not energetic") != "energetic", "not energetic is not energetic")
    check(nlp.extract_entities("I am not stressed")["negated"] == ["stressed"], "negated word recorded")
    check(m("I want to relax") == "neutral" and "relax" in nlp.extract_entities("I want to relax")["goal"],
          "'I want to relax' is a goal, not a calm state")
    check(m("I can't sleep") == "stressed", "can't sleep reads as stressed")
    check(m("I did not sleep well") == "tired", "poor sleep reads as tired")
    # intensity and polarity
    a = nlp.analyse_mood("my back hurts")[1]
    b = nlp.analyse_mood("my back hurts a lot, it is terrible")[1]
    check(a < 0 and b < a, f"pain is negative and more pain is more negative ({a}, {b})")
    check(nlp.analyse_mood("I feel great and happy")[1] > 0.5, "positive sentiment")
    check(nlp.analyse_mood("not good")[1] < 0 < nlp.analyse_mood("not bad")[1], "negation flips polarity")
    check(-1 <= nlp.analyse_mood("terrible awful horrible worst pain")[1] <= 1, "sentiment is bounded")
    check(m("I feel very tired and stressed") == "tired", "an intensifier outweighs an equal plain mood")
    u = nlp.understand("I feel very tired and stressed")
    check(u.mood_scores["stressed"] == 1.0, "stress/stressed/stressful count a word once")
    # Hinglish moods
    check(m("tanav hai") == "stressed", "tanav -> stressed")
    check(m("bahut thakan hai") == "tired", "thakan -> tired")
    check(m("tanav nahi hai") != "stressed", "Hinglish negation: tanav nahi hai")
    check(m("neend nahi aati") == "stressed", "neend nahi aati reads as a sleep problem")


def test_recommend() -> None:
    # the TF-IDF space itself
    ix = nlp._index()
    norms = (ix.matrix ** 2).sum(axis=1) ** 0.5
    check(abs(norms - 1.0).max() < 1e-9, "every routine vector is unit length")
    check(set(ix.keys) == set(rt.all_routines()), "an index row for every routine")
    for key, r in rt.all_routines().items():
        sc = dict(nlp.tfidf_scores(nlp._doc_text(r)))
        check(max(sc, key=sc.get) == key and sc[key] > 0.99, f"{key}: its own description retrieves it")
    check(all(abs(s) < 1e-12 for _, s in nlp.tfidf_scores("zzz qqq")), "unknown words have zero cosine")
    check(all(0.0 <= s <= 1.0 + 1e-9 for _, s in nlp.tfidf_scores("my lower back hurts")),
          "cosine is in 0..1")

    u = nlp.understand("my lower back hurts after sitting all day, I only have 10 minutes")
    keys = [k for k, _ in u.ranking]
    check(keys[0] in ("back_pain", "desk_sitting") and set(keys[:2]) == {"back_pain", "desk_sitting"},
          f"the worked example: {u.ranking[:3]}")
    check(u.minutes == 10 and u.entities["body_part"] == ["lower back"], "worked example slots")
    check(len(u.ranking) == len(rt.all_routines()), "ranking covers every routine")
    check(all(a[1] >= b[1] for a, b in zip(u.ranking, u.ranking[1:])), "ranking is sorted by score")
    check("lower back" in u.explanation and "Back-pain relief" in u.explanation
          and "10 min" in u.explanation, "explanation names the slots and the time")
    check(all(isinstance(t, str) for t in u.tokens), "tokens are strings")
    key, why = nlp.recommend("my lower back hurts after sitting all day")
    check(key in rt.all_routines() and why.startswith("Best match:"), "recommend() returns key + why")
    key, why = nlp.recommend("hello there")
    check(key == "daily_energy" and "could not find" in why, "no signal -> balanced default, said plainly")
    key, why = nlp.recommend("")
    check(key in rt.all_routines(), "empty text still returns a routine")
    # negation must not leak into the recommendation through word overlap
    u = nlp.understand("I am not stressed")
    check(dict(u.ranking)["stress_relief"] < 0.05, f"'not stressed' does not pull in stress relief: {u.ranking[:2]}")
    u = nlp.understand("I'm not stressed, just tired and low on energy")
    check(u.ranking[0][0] == "daily_energy" and dict(u.ranking)["stress_relief"] < 0.05
          and "did not count" in u.explanation, "tired-but-not-stressed goes to daily energy")
    check("tired" in u.explanation and " tir," not in u.explanation,
          "the explanation quotes the person's words, not stems")
    check("flexibility" in nlp.extract_entities("I can't touch my toes")["goal"],
          "negating an ability ('can't touch my toes') is still a flexibility goal")
    # constraints: a 2-minute window and a head-below-heart restriction show up in the reasons
    u = nlp.understand("I have 1 minute, my neck hurts")
    check(any("longer than" in why for why in [u.explanation]), "time that does not fit is said so")
    u = nlp.understand("I have high blood pressure and my neck is stiff from studying")
    check("avoid_inversion" in u.flags and "skipped for you" in u.explanation,
          "an inversion-heavy routine is marked down when inversions are unsafe")

    d1, d2, miss = accuracy(DEV)
    print(f"  development set ({len(DEV)} utterances, written alongside the code): "
          f"top-1 {d1:.1%}, top-2 {d2:.1%}")
    for m_ in miss:
        print("    miss:", m_)
    check(len(DEV) >= 30, "at least 30 development utterances")
    check(d2 >= 0.90, f"development top-2 accuracy >= 90% (got {d2:.1%})")
    b1, b2, miss = accuracy(BLIND)
    print(f"  second set ({len(BLIND)} utterances, written after freezing; see docstring): "
          f"top-1 {b1:.1%}, top-2 {b2:.1%}")
    for m_ in miss:
        print("    miss:", m_)
    check(b2 >= 0.85, f"second-set top-2 accuracy >= 85% (got {b2:.1%})")
    check(d1 >= 0.75 and b1 >= 0.75, "top-1 accuracy stays above 75% on both sets")


def test_safety() -> None:
    f = nlp.safety_flags
    check({"diabetic"} <= f("I have diabetes"), "diabetes -> diabetic")
    check({"high_bp", "avoid_inversion"} <= f("I have high BP"), "high BP -> high_bp + avoid_inversion")
    check({"low_bp"} <= f("my blood pressure is low") and "avoid_inversion" not in f("low bp"),
          "low BP -> low_bp only")
    check({"pregnant", "avoid_inversion"} <= f("I am pregnant"), "pregnant")
    check("joint_pain" in f("I have arthritis") and "joint_pain" in f("my knees hurt"),
          "joint_pain from a condition or from joint pain itself")
    check("avoid_inversion" in f("I get dizzy sometimes") and "vertigo" in f("I get dizzy sometimes"),
          "dizziness -> avoid inversion")
    check("red_flag" in f("I have chest pain") and "red_flag" in f("I fainted yesterday"), "red flags")
    check("spine_injury" in f("slip disc") and "asthma" in f("I have asthma"), "spine + asthma flags")
    check("heart_condition" in f("heart problem") and "glaucoma" in f("glaucoma"), "heart + eye flags")
    check(f("I don't have diabetes or asthma") == set() or f("I don't have diabetes") == set(),
          "negated conditions raise nothing")
    check(f("") == set() and f("I feel fine") == set(), "no condition, no flags")
    check({"bp_unspecified", "avoid_inversion"} <= f("I have a bp problem"),
          "an unspecified BP problem is treated conservatively")
    cs = nlp.flag_cautions({"high_bp", "diabetic", "avoid_inversion"})
    check(len(cs) == 2 and all(c.endswith(nlp.DOCTOR) for c in cs),
          "flag cautions end with the doctor note; derived flags add none")

    # medication cautions
    for text, word, flag in [
        ("insulin", "hypoglycaemia", "diabetic"), ("metformin 500mg twice a day", "blood sugar", "diabetic"),
        ("amlodipine", "rise slowly", "avoid_inversion"), ("telmisartan", "rise slowly", "on_bp_medication"),
        ("atenolol", "heart rate", "avoid_inversion"), ("beta blocker", "heart rate", "avoid_inversion"),
        ("warfarin", "bleeding", "on_anticoagulant"), ("blood thinner", "bleeding", "avoid_inversion"),
        ("ibuprofen", "hide pain", "on_painkiller"), ("atorvastatin", "muscle", "on_statin"),
        ("salbutamol inhaler", "inhaler", "asthma"), ("prednisolone", "bones", "on_steroid"),
    ]:
        cs = nlp.medication_flags(text)
        check(len(cs) >= 1 and word in " ".join(cs).lower(), f"{text!r} -> caution mentioning {word!r}")
        check(flag in nlp.safety_flags(text), f"{text!r} raises {flag}")
    check(len(nlp.medication_flags("metformin and warfarin")) == 2, "two medicines, two cautions")
    check(nlp.medication_flags("Metformin, Amlodipine") == nlp.medication_flags("metformin and amlodipine"),
          "case and punctuation do not matter")
    check(nlp.medication_flags("") == [] and nlp.medication_flags("none") == []
          and nlp.medication_flags("I take no medicines") == [] and nlp.medication_flags("N/A") == [],
          "no medication -> no cautions")
    check(nlp.medication_flags("not on insulin") == [], "a negated medicine raises nothing")
    check(nlp.medication_flags("zorbitrol") == [nlp.GENERIC_MEDICATION_CAUTION],
          "an unrecognised medicine gets the generic caution, not silence")
    # the safety stance: cautions only, never doses, always a doctor note
    for row in nlp._MED_TABLE:
        c = nlp.medication_flags(row[1][0])[0]
        check(c.endswith("Check with your doctor."), f"{row[0]}: ends with the doctor note")
        check(not re.search(r"\d|\bmg\b|\bdose|\bdosage|\bstop taking|\bincrease\b", c, re.I),
              f"{row[0]}: no dosing or stop/start advice")
    check(nlp.GENERIC_MEDICATION_CAUTION.endswith("Check with your doctor."), "generic caution has the note")
    check(all(c.endswith("Check with your doctor.") for c in nlp.flag_cautions(nlp._FLAG_CAUTIONS)),
          "every flag caution ends with the doctor note")
    u = nlp.understand("I'm on metformin and my blood pressure is high")
    check("diabetic" in u.flags and "high_bp" in u.flags and len(u.cautions) >= 3
          and "Care:" in u.explanation, "understand() carries cautions into the explanation")


def test_summary() -> None:
    rows = [
        {"pose": "Tree Pose", "result": "held 20 s, best alignment 87%", "completed": True, "scored": True},
        {"pose": "Child's Pose", "result": "held 30 s (timed)", "completed": True, "scored": False},
        {"pose": "Cobra Pose", "result": "skipped", "completed": False, "scored": True},
    ]
    p = base(name="Asha Rao", bp="Low")
    s = nlp.session_summary(rows, p)
    sentences = re.findall(r"[^.!?]+[.!?]", s)
    check(bool(s.strip()), "summary is not empty")
    check(2 <= len(sentences) <= 4, f"two to four sentences, got {len(sentences)}: {s}")
    check("2 of 3" in s, "mentions the completed count")
    check("Tree Pose" in s and "Child's Pose" in s, "names the completed poses")
    check("30 s" in s and "87%" in s, "reports the longest hold and the best alignment")
    check("Asha" in s and "Rao" not in s, "uses the first name only")
    check(s[0].isupper() and s.rstrip()[-1] in ".!?" and ".." not in s and "  " not in s,
          "capitalised, terminated, no doubled punctuation or spaces")
    check(nlp.session_summary(rows, p) == s, "same session, same summary")

    full = [dict(r, completed=True) for r in rows]
    s = nlp.session_summary(full, None)
    check("all 3 poses" in s, f"all completed says so: {s}")
    s = nlp.session_summary(rows[:1], base(job="standing"))
    check("1 of 1 pose" in s and "1 of 1 poses" not in s, f"singular grammar: {s}")
    s = nlp.session_summary([dict(r, completed=False) for r in rows], base())
    check("No poses were completed" in s or "did not finish" in s, f"zero completed: {s}")
    s = nlp.session_summary([], None)
    check(bool(s.strip()) and len(re.findall(r"[^.!?]+[.!?]", s)) >= 2, f"empty session: {s}")
    many = [{"pose": f"Pose {i}", "result": "held 20 s (timed)", "completed": True, "scored": False}
            for i in range(6)]
    s = nlp.session_summary(many, None)
    check("and 3 more" in s and "all 6 poses" in s, f"long pose lists are truncated: {s}")
    skipped = rows + [{"pose": "Downward Dog", "result": "skipped for your profile"}]
    # mixed job, no BP, no medication: the left-out note is the only tip that applies
    check("left out for your profile" in nlp.session_summary(skipped, base(job="mixed")),
          "poses skipped for the profile are explained in the tip")
    check("left out for your profile" not in nlp.session_summary(rows, base(job="mixed")),
          "...and only when some were")
    check(nlp.session_summary(skipped, base(medication="warfarin"), seed=3),
          "rows without 'completed' are tolerated")
    variants = {nlp.session_summary(rows, p, seed=i) for i in range(12)}
    check(len(variants) >= 3, f"wording varies with the seed ({len(variants)} variants)")
    check(all(2 <= len(re.findall(r"[^.!?]+[.!?]", v)) <= 4 for v in variants), "every variant is 2-4 sentences")


def test_determinism() -> None:
    texts = [t for t, _ in DEV[:12]] + ["", "hello", "I have diabetes and I am not stressed"]
    for t in texts:
        a, b = nlp.understand(t), nlp.understand(t)
        check(a == b, f"same input, same Understanding: {t!r}")
    check(nlp.recommend(DEV[0][0]) == nlp.recommend(DEV[0][0]), "recommend() is deterministic")
    u = nlp.understand("I have diabetes")
    check(isinstance(u.flags, set) and isinstance(u.ranking, list) and isinstance(u.explanation, str),
          "Understanding field types")


def test_profile_integration() -> None:
    check(re.fullmatch(r"YC-[0-9A-F]{6}", prof.new_user_id()) is not None, "id looks like YC-7F3A9C")
    check(len({prof.new_user_id() for _ in range(50)}) == 50, "ids are unique")
    check(base().user_id == "" and base().medication == "", "new fields default to empty")
    check(base(user_id="YC-AAAAAA") == base(), "user_id does not affect profile equality")
    check(len(prof.validate(base(medication="x" * 400))) == 1, "an absurdly long medication field is rejected")
    check(prof.validate(base(medication="metformin, amlodipine")) == [], "a normal medication field validates")
    with tempfile.TemporaryDirectory() as d:
        old = prof.USERS_DIR
        prof.USERS_DIR = d
        try:
            p = base(medication="metformin")
            path = prof.save(p)
            check(re.fullmatch(r"YC-[0-9A-F]{6}", p.user_id) is not None, "save() makes an id")
            import json
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            check(raw["user_id"] == p.user_id and raw["medication"] == "metformin", "both fields are persisted")
            back = prof.load("Asha")
            check(back.user_id == p.user_id and back.medication == "metformin", "both fields load back")
            first = p.user_id
            prof.save(base())                                      # a fresh object, same name
            check(prof.load("Asha").user_id == first, "saving the same name again keeps its id")
            prof.save(base(name="Ravi"))
            check(prof.load("Ravi").user_id != first, "a different person gets a different id")
            hist = prof.append_history(back, [{"pose": "x"}])
            with open(hist, encoding="utf-8") as fh:
                check(json.load(fh)[-1]["user_id"] == first, "history entries carry the id")
            # an older file written before these fields existed still loads
            legacy = {"name": "Old", "age": 40, "weight_kg": 70.0, "height_cm": 170.0, "bp": "Low",
                      "job": "standing", "diet": "Vegan", "location": "Pune"}
            with open(os.path.join(d, "Old.json"), "w", encoding="utf-8") as fh:
                json.dump(legacy, fh)
            old_p = prof.load("Old")
            check(old_p is not None and old_p.user_id == "" and old_p.medication == "" and old_p.bp == "Low",
                  "a legacy profile file loads with empty new fields")
            prof.save(old_p)
            check(prof.load("Old").user_id.startswith("YC-"), "...and gets an id the next time it is saved")
            ids = [prof.load(n).user_id for n in prof.saved_names()]
            check(len(ids) == len(set(ids)) == 3, "every stored id is distinct")
        finally:
            prof.USERS_DIR = old

    routine = rt.all_routines()["standing_workers"]                # includes Downward Dog
    plain = prof.personalise(base(), routine)
    check(all(not i.skipped for i in plain.items), "no medication: nothing skipped")
    check(plain.notes == prof.personalise(base(medication=""), routine).notes
          and plain.notes == prof.personalise(base(medication="  "), routine).notes,
          "empty medication changes nothing")
    for med, word in [("warfarin", "bleeding"), ("amlodipine", "rise slowly"), ("atenolol", "heart rate")]:
        pl = prof.personalise(base(medication=med), routine)
        dog = next(i for i in pl.items if i.pose_key == "adho_mukha")
        check(dog.skipped and "medication" in dog.reason, f"{med}: inversion skipped, reason names medication")
        check(any(word in n for n in pl.notes), f"{med}: caution added to the plan notes")
        check(all(i.hold_s <= rt.POSES[i.pose_key].hold_s for i in pl.active), f"{med}: holds only shrink")
    pl = prof.personalise(base(medication="metformin"), routine)
    check(not any(i.skipped for i in pl.items) and any("blood sugar" in n for n in pl.notes),
          "metformin: cautions but no skipped poses")
    pl = prof.personalise(base(medication="insulin", diabetic=True), routine)
    check(sum("glucose" in n for n in pl.notes) >= 1 and any("hypoglycaemia" in n for n in pl.notes),
          "the existing diabetic note and the insulin caution both appear")
    pl = prof.personalise(base(medication="zorbitrol"), routine)
    check(nlp.GENERIC_MEDICATION_CAUTION in pl.notes, "an unrecognised medicine still gets a caution")
    pl = prof.personalise(base(bp="High", medication="telmisartan"), routine)
    check(next(i for i in pl.items if i.pose_key == "adho_mukha").reason
          == "skipped for your age / blood pressure / medication", "reason lists every cause")
    pl = prof.personalise(base(bp="High"), routine)
    check(next(i for i in pl.items if i.pose_key == "adho_mukha").reason
          == "skipped for your age / blood pressure", "the original reason text is unchanged")
    for r in rt.all_routines().values():
        pl = prof.personalise(base(age=70, bp="High", diabetic=True, medication="warfarin, metformin"), r)
        check(all(i.hold_s <= rt.POSES[i.pose_key].hold_s for i in pl.active), f"{r.key}: still only gentler")


def main() -> int:
    for fn in (test_taxonomy, test_tokenise, test_entities, test_mood, test_recommend,
               test_safety, test_summary, test_determinism, test_profile_integration):
        print(fn.__name__)
        fn()
    print(f"\n{N - len(FAILS)}/{N} nlp checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
