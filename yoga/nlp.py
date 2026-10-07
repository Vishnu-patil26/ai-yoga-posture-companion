"""'Ask the coach' - classical, offline NLP over a sentence the user types.

    "my lower back hurts after sitting all day, I only have 10 minutes"

goes through the same small pipeline every time, and every stage is a named,
inspectable function (the project guide wants the NLP *explained*, so nothing
here is a black box and nothing needs a network or a model download):

  a. tokenisation + light normalisation   ``tokenize``, ``stem``
  b. entity / slot extraction             body part, symptom, condition, job,
                                          duration, goal, medication
  c. sentiment and mood                   lexicon polarity + a mood label,
                                          with negation ("not tired")
  d. intent -> routine recommendation     hand-written TF-IDF + cosine over the
                                          routine descriptions, plus rule boosts
                                          from the slots, with a plain-English
                                          explanation of why
  e. safety flags and medication cautions what the personaliser may act on
  f. natural-language generation          ``session_summary``

Why classical NLP and not a neural model: the vocabulary is tiny and closed
(ten poses, eight routines, a few dozen body words), the app must run offline
on a student laptop, and a viva examiner can follow a lexicon and a cosine
similarity line by line.  The cost is honest and stated: it understands what
the lexicons and rules below cover, nothing more, and it is evaluated only on a
hand-written development set (see tools/test_nlp.py).

Safety stance: this module never gives dosing or diagnosis.  It turns what the
person says into *cautions* (each ending "check with your doctor") and into
flags such as ``avoid_inversion`` that can only ever make a routine gentler.

Standard library + numpy only.  No randomness except a seeded ``random.Random``.
"""

from __future__ import annotations

import math
import random
import re
import zlib
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from yoga import routines

BOUNDARY = "|"          #: clause boundary marker kept in the token stream

DOCTOR = "Check with your doctor."


# =========================================================================
# a. TOKENISATION + NORMALISATION
# =========================================================================
_CONTRACTIONS = (
    (re.compile(r"\bcan['’]?t\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\bwon['’]?t\b"), "will not"),
    (re.compile(r"\b(do|does|did|is|are|was|were|has|have|had|could|would|should)"
                r"nt\b"), r"\1 not"),
    (re.compile(r"n['’]t\b"), " not"),
    (re.compile(r"['’](?:m|ve|re|s|ll|d)\b"), ""),
)

# --- Hinglish lexicon ---------------------------------------------------
# Words that appear in this project's meeting notes (Hindi written in Latin
# letters).  Deliberately small and explicit: each is rewritten to the English
# word(s) the rest of the pipeline already understands.  Not a translator.
HINGLISH = {
    "dard": ("pain",),                  # pain
    "kamar": ("lower", "back"),         # lower back
    "gardan": ("neck",),                # neck
    "ghutna": ("knee",), "ghutne": ("knee",), "ghutnon": ("knee",),   # knee
    "thakan": ("tired",), "thaka": ("tired",), "thaki": ("tired",),   # tiredness
    "thakaan": ("tired",),
    "tanav": ("stress",),               # stress
    "neend": ("sleep",),                # sleep
    "chinta": ("worry",),               # worry
    "kandha": ("shoulder",), "kandhe": ("shoulder",),                  # shoulder
    "pith": ("back",), "peeth": ("back",),                             # back
    "padhai": ("study",),               # studying
    "accha": ("good",), "achha": ("good",), "acha": ("good",),         # good
    "theek": ("fine",),                 # fine / okay
    "nahin": ("nahi",), "nahee": ("nahi",),    # normalise "no" to one spelling
}
#: Hinglish function words that carry no meaning for us.
HINGLISH_STOP = {
    "mujhe", "mere", "meri", "mera", "mein", "hai", "hain", "ho", "hota", "hoti",
    "hote", "raha", "rahi", "rahe", "rha", "rhi", "ka", "ki", "ke", "ko", "se",
    "aur", "bhi", "aati", "aata", "aate", "lag", "lagta", "lagti", "toh", "kya",
    "hoon", "hu", "hun", "par", "pe", "aaj", "kal", "bohot", "bahut", "zyada",
}

# Compound words and run-together negations rewritten to their parts.
_EXPAND = {
    **{k: v for k, v in HINGLISH.items()},
    "backache": ("back", "ache"), "headache": ("head", "ache"),
    "lowerback": ("lower", "back"), "wfh": ("work", "from", "home"),
    "dont": ("do", "not"), "doesnt": ("does", "not"), "didnt": ("did", "not"),
    "isnt": ("is", "not"), "wasnt": ("was", "not"), "wont": ("will", "not"),
}

_STOP = frozenset("""
a an the and or but if so of to in on at by for with from as is are was were be
been being am i me my mine myself we our you your he she it its they them their
this that these those there here do does did doing have has had having will
would can could should may might must shall very really quite just also too
still even only some any all each every more most much many what which who
whom whose when where why how then than get got getting go going goes went come
came please pls kindly about around into onto over under up down out off again
yoga pose poses asana asanas routine session today want wanna need like help
something anything everything nothing not no never nor without day days time
times hour hours minute minutes min mins feel feeling feels felt am after before
during while because since until though although however yet nor
""".split()) | frozenset(HINGLISH_STOP)

_IRREGULAR = {
    "felt": "feel", "slept": "sleep", "sat": "sit", "stood": "stand",
    "feet": "foot", "achy": "ache", "children": "child", "women": "woman",
}


def _undouble(base: str) -> str:
    """sitt -> sit, stopp -> stop; but keep the doubles of pull, stress, buzz."""
    if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "lszaeiou":
        return base[:-1]
    return base


def stem(word: str) -> str:
    """A small suffix stemmer, written for this module.

    It is *consistent* rather than linguistically perfect: every inflection of
    a word lands on the same string (``sitting``/``sits`` -> ``sit``,
    ``stretches``/``stretching`` -> ``stretch``, ``tired``/``tire`` -> ``tir``),
    which is all a lexicon lookup and a TF-IDF index need.  Every lexicon in
    this file is passed through this same function, so a mismatch between how a
    word is written there and how a user types it cannot happen.
    """
    w = _IRREGULAR.get(word, word)
    if len(w) <= 3 or not w.isalpha():
        return w
    # plurals
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith(("ches", "shes", "xes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    # a few derivational endings first, so tiredness -> tired -> tir like tired
    for _ in range(2):
        for suf, minbase in (("ness", 3), ("ful", 3), ("ly", 4), ("ation", 4)):
            if w.endswith(suf) and len(w) - len(suf) >= minbase:
                w = w[: -len(suf)]
                break
    # verb endings
    if w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ing") and len(w) - 3 >= 3 and re.search(r"[aeiouy]", w[:-3]):
        w = _undouble(w[:-3])
    elif w.endswith("ed") and len(w) - 2 >= 3 and re.search(r"[aeiouy]", w[:-2]):
        w = _undouble(w[:-2])
    # a trailing silent e, so tire/tired/tiring and ache/aching all meet
    if w.endswith("e") and not w.endswith("ee") and len(w) >= 4:
        w = w[:-1]
    return w


def _clean(text: str) -> str:
    t = text.lower()
    for rx, rep in _CONTRACTIONS:
        t = rx.sub(rep, t)
    return t


_TOKEN_RX = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[.,;:!?\n()]")
_PUNCT = frozenset(".,;:!?\n()")


def _raw_tokens(clean: str) -> list[str]:
    """Words, numbers and clause boundaries; Hinglish and compounds expanded."""
    t = clean.replace("&", " and ").replace("-", " ").replace("/", " ")
    out: list[str] = []
    for m in _TOKEN_RX.finditer(t):
        tok = m.group()
        if tok in _PUNCT:
            if out and out[-1] != BOUNDARY:
                out.append(BOUNDARY)
        else:
            out.extend(_EXPAND.get(tok, (tok,)))
    return out


@dataclass
class _Analysis:
    clean: str
    raws: list[str]          #: lower-case tokens incl. BOUNDARY markers
    stems: list[str]         #: stem of each raw token (BOUNDARY kept)
    content: list[str]       #: stems that carry meaning (no stop words/numbers)
    content_idx: list[int]   #: position in ``raws`` of each content token

    def surface(self, stem_: str) -> str:
        """The word the person actually typed for a stem (for showing back to them)."""
        for r, s in zip(self.raws, self.stems):
            if s == stem_ and r != BOUNDARY:
                return r
        return stem_


def _analyse(text: str) -> _Analysis:
    clean = _clean(text or "")
    raws = _raw_tokens(clean)
    stems = [t if t == BOUNDARY else stem(t) for t in raws]
    keep = [i for i, r in enumerate(raws)
            if r != BOUNDARY and r not in _STOP and not r[0].isdigit()]
    return _Analysis(clean, raws, stems, [stems[i] for i in keep], keep)


def tokenize(text: str) -> list[str]:
    """Normalised content tokens: lower-cased, Hinglish mapped, stop words
    dropped, stemmed.  This is what the TF-IDF index sees."""
    return _analyse(text).content


# ---- lexicon machinery ---------------------------------------------------
def _phrase_stems(phrase: str) -> tuple[str, ...]:
    return tuple(stem(r) for r in _raw_tokens(_clean(phrase)) if r != BOUNDARY)


def _compile(src: dict[str, list[str]]) -> dict[tuple[str, ...], str]:
    lex: dict[tuple[str, ...], str] = {}
    for canon, phrases in src.items():
        for ph in phrases:
            key = _phrase_stems(ph)
            if key:
                lex.setdefault(key, canon)
    return lex


def _find(stems: list[str], lex: dict, maxlen: int | None = None) -> list[tuple[int, int, object]]:
    """Greedy longest-match of lexicon phrases over a stem sequence.

    Longest first so that ``lower back`` is one body part and not ``back``
    plus a stray ``lower``.  Matches never cross a clause boundary.
    """
    if maxlen is None:
        maxlen = max((len(k) for k in lex), default=1)
    out, i, n = [], 0, len(stems)
    while i < n:
        hit = None
        for length in range(min(maxlen, n - i), 0, -1):
            seg = tuple(stems[i:i + length])
            if BOUNDARY in seg:
                continue
            canon = lex.get(seg)
            if canon is not None:
                hit = (i, i + length, canon)
                break
        if hit:
            out.append(hit)
            i = hit[1]
        else:
            i += 1
    return out


# ---- negation and intensity ---------------------------------------------
_NEG_PRE = frozenset({"not", "no", "never", "without", "neither", "nor", "hardly",
                      "barely"})
_NEG_POST = frozenset({"nahi"})                   # Hinglish: "stress nahi hai"
_CONTRAST = frozenset({"but", "however", "although", "though", "yet", "except"})
_INTENS = frozenset({"very", "so", "really", "extremely", "super", "too", "bahut",
                     "bohot", "zyada", "completely", "totally", "absolutely",
                     "incredibly", "terribly", "severely", "deeply", "badly"})
_DIMIN = frozenset({"slightly", "little", "bit", "somewhat", "mildly", "mild",
                    "kinda", "abit"})
_GOAL_CUES = frozenset({"want", "wanna", "need", "wish", "like", "hope", "help",
                        "how", "try", "trying", "looking", "something", "anything",
                        "learn", "teach", "give", "suggest"})


def _negated(raws: list[str], start: int, end: int) -> bool:
    """Is the span ``raws[start:end]`` negated?

    A negator up to three words before it, inside the same clause and not
    separated by 'but'/'however', or a Hinglish 'nahi' right after it.  Cheap
    and wrong on subtle cases ("not sure whether I'm tired") - which is the
    honest trade for something a viva can read in ten lines.
    """
    k, steps = start - 1, 0
    while k >= 0 and steps < 3:
        t = raws[k]
        if t == BOUNDARY or t in _CONTRAST:
            break
        if t in _NEG_PRE:
            return True
        k -= 1
        steps += 1
    for t in raws[end:end + 2]:
        if t == BOUNDARY:
            break
        if t in _NEG_POST:
            return True
    return False


def _modifier(raws: list[str], start: int) -> float:
    for t in raws[max(0, start - 2):start][::-1]:
        if t == BOUNDARY:
            break
        if t in _INTENS:
            return 1.5
        if t in _DIMIN:
            return 0.6
    return 1.0


def _goal_cue(raws: list[str], start: int) -> bool:
    """A want/need/help-me cue earlier in the same clause: 'I want to be
    flexible and *relax*' asks for calm, 'I feel *relaxed*' reports it."""
    for t in raws[max(0, start - 6):start][::-1]:
        if t == BOUNDARY or t in _CONTRAST:
            return False
        if t in _GOAL_CUES:
            return True
    return False


# =========================================================================
# b. ENTITY / SLOT EXTRACTION  (lexicons + regex)
# =========================================================================
_BODY_SRC = {
    "lower back": ["lower back", "low back", "lumbar", "small of the back", "tailbone",
                   "lower spine"],
    "upper back": ["upper back", "mid back", "middle back", "shoulder blade",
                   "shoulder blades", "thoracic"],
    "neck": ["neck", "cervical", "text neck"],
    "shoulder": ["shoulder", "rotator cuff"],
    "back": ["back", "backbone"],
    "spine": ["spine", "spinal", "vertebra", "vertebrae"],
    "hip": ["hip", "pelvis", "pelvic", "glute"],
    "knee": ["knee", "kneecap"],
    "ankle": ["ankle"],
    "foot": ["foot", "heel", "sole of the foot", "plantar"],
    "leg": ["leg", "thigh", "calf", "calves", "hamstring", "quad", "quadriceps", "shin"],
    "wrist": ["wrist", "carpal"],
    "elbow": ["elbow"],
    "hand": ["hand", "finger"],
    "arm": ["arm", "biceps", "triceps"],
    "head": ["head", "temple"],
    "chest": ["chest"],
    "abdomen": ["belly", "abdomen", "stomach", "tummy"],
}
_JOINT_PARTS = {"knee", "ankle", "wrist", "elbow", "hip", "shoulder"}

_SYMPTOM_SRC = {
    "pain": ["pain", "hurt", "ache", "sore", "cramp", "spasm", "throb", "sciatica",
             "strain", "pulled muscle", "injury", "injured", "burning"],
    "stiffness": ["stiff", "tight", "rigid", "knot", "locked", "stuck"],
    "swelling": ["swollen", "swelling", "puffy"],
    "numbness": ["numb", "tingling", "pins and needles"],
    "dizziness": ["dizzy", "dizziness", "giddy", "vertigo", "light headed",
                  "lightheaded"],
    "fainting": ["faint", "fainting", "fainted", "blackout", "passed out"],
}

_COND_SRC = {
    "diabetes": ["diabetes", "diabetic", "blood sugar", "sugar patient",
                 "sugar problem", "high sugar", "sugar level", "have sugar"],
    "high_bp": ["hypertension", "hypertensive"],
    "low_bp": ["hypotension"],
    "pregnancy": ["pregnant", "pregnancy", "trimester", "expecting a baby",
                  "expecting mother", "expecting mom", "expecting a child"],
    "arthritis": ["arthritis", "arthritic", "joint pain", "joint problem",
                  "rheumatoid", "osteoarthritis", "gout"],
    "asthma": ["asthma", "asthmatic", "wheeze", "wheezing", "breathing problem",
               "shortness of breath", "short of breath", "breathless"],
    "slip_disc": ["slip disc", "slipped disc", "herniated disc", "disc bulge",
                  "bulging disc", "prolapsed disc", "disc problem", "disc prolapse"],
    "sciatica": ["sciatica", "sciatic"],
    "scoliosis": ["scoliosis"],
    "heart": ["heart problem", "heart disease", "heart patient", "heart condition",
              "cardiac", "angina", "pacemaker", "heart attack"],
    "glaucoma": ["glaucoma", "eye pressure", "detached retina"],
    "osteoporosis": ["osteoporosis", "osteopenia", "weak bones", "brittle bones"],
    "hernia": ["hernia"],
    "surgery": ["surgery", "operated", "fracture", "fractured", "broken bone"],
    "migraine": ["migraine"],
}
#: a blood-pressure mention; whether it means high or low is read from its neighbours
_BP_SRC = {"bp": ["bp", "blood pressure"]}
_BP_HIGH = frozenset({"high", "elevated", "raised", "hypertension", "up"})
_BP_LOW = frozenset({"low", "dropping", "drop", "hypotension"})
_BP_FINE = frozenset({"normal", "fine", "ok", "okay", "good", "healthy", "stable"})

_JOB_SRC = {
    "sitting": ["sit", "seated", "desk", "office", "laptop", "computer",
                "work from home", "sedentary", "driver", "driving", "programmer",
                "developer", "coder", "accountant", "call centre", "call center",
                "typing", "cubicle"],
    "standing": ["stand", "shop", "shopkeeper", "retail", "teacher", "nurse",
                 "factory", "guard", "waiter", "cashier", "barber", "hairdresser",
                 "on my feet", "surgeon", "chef", "labourer", "laborer"],
    "homemaker": ["homemaker", "housewife", "household", "housework", "house work",
                  "chores", "cooking", "cook", "kitchen", "stay at home",
                  "cleaning", "sweeping", "laundry", "kids", "kid", "toddler"],
    "student": ["student", "study", "exam", "homework", "lecture", "university",
                "revision", "tuition", "text neck"],
}

_GOAL_SRC = {
    "relax": ["unwind", "destress", "de stress", "wind down", "chill out"],
    "energy": ["energise", "energize", "boost", "wake up", "refresh", "stamina"],
    "flexibility": ["flexible", "flexibility", "supple", "mobility", "touch my toes",
                    "touch toes", "stretch"],
    "strength": ["strength", "strengthen", "core", "tone", "toning", "muscle",
                 "stronger"],
    "posture": ["posture", "slouch", "hunch", "hunched", "rounded shoulders"],
    "focus": ["focus", "concentrate", "concentration", "attention", "mindful",
              "mindfulness", "meditation", "meditate"],
    "sleep": ["sleep better", "better sleep", "sleep well"],
    "weight": ["weight loss", "lose weight", "burn calories", "slim"],
    "fitness": ["fit", "fitness", "workout", "work out", "exercise", "stay active",
                "full body", "challenge"],
    "balance": ["balance", "balancing"],
}
_GOAL_TEXT = {
    "relax": "relax or unwind", "energy": "boost your energy",
    "flexibility": "improve flexibility", "strength": "build strength",
    "posture": "improve your posture", "focus": "focus", "sleep": "sleep better",
    "weight": "lose weight", "fitness": "get a general workout",
    "balance": "work on balance",
}

_BODY_LEX = _compile(_BODY_SRC)
_SYMPTOM_LEX = _compile(_SYMPTOM_SRC)
_COND_LEX = _compile(_COND_SRC)
_BP_LEX = _compile(_BP_SRC)
_JOB_LEX = _compile(_JOB_SRC)
_GOAL_LEX = _compile(_GOAL_SRC)

#: "back" in these contexts is an adverb, not a body part ("come back", "back to work")
_BACK_PREV = frozenset({"come", "came", "get", "got", "go", "going", "went", "give",
                        "bring", "call", "take", "put", "step", "fall", "be", "am",
                        "are", "was", "were", "will", "right"})
_BACK_NEXT = frozenset({"to", "from", "home", "soon", "again", "later", "then", "in"})

# ---- position requests: "only standing poses", "something I can do in a chair"
_POS_RX = {
    "seated": [
        r"\b(?:sitting|seated|chair)\s+(?:poses?|asanas?|postures?|yoga|ones|exercises?|stretches)\b",
        r"\b(?:poses?|asanas?|yoga|exercises?|stretches)\b[^.,]{0,15}\b(?:while\s+sitting|in\s+(?:a\s+|my\s+)?chair|on\s+(?:a\s+|my\s+)?chair|seated|sitting\s+down)\b",
        r"\b(?:do|doing|practi[sc]e)\s+(?:it\s+|them\s+|these\s+|something\s+|yoga\s+|exercises?\s+|poses?\s+)?(?:while\s+|in\s+(?:a\s+|my\s+)?|on\s+(?:a\s+|my\s+)?)?(?:sitting|seated|chair)\b",
        r"\bchair\s+yoga\b",
        r"\bcan not stand for\b",
    ],
    "standing": [
        r"\b(?:standing|upright)\s+(?:poses?|asanas?|postures?|ones|exercises?)\b",
        r"\bno\s+floor\b",
        r"\b(?:poses?|asanas?|exercises?)\b[^.,]{0,15}\b(?:while\s+standing|on\s+my\s+feet|standing\s+up)\b",
        r"\b(?:do|doing|practi[sc]e)\s+(?:it\s+|them\s+|these\s+|something\s+|yoga\s+)?(?:while\s+)?standing\b",
        r"\bcan not (?:get|go)\s+down\b",
        r"\b(?:without|no)\s+(?:a\s+)?mat\b",
    ],
    "floor": [
        r"\b(?:floor|mat|lying|kneeling|prone|ground)\s+(?:poses?|asanas?|postures?|work|ones|exercises?|yoga)\b",
        r"\bposes?\b[^.,]{0,15}\bon\s+(?:the\s+|a\s+|my\s+)?(?:floor|mat|ground)\b",
        r"\b(?:lie|lying)\s+down\b",
        r"\bon\s+(?:the\s+)?(?:floor|mat)\b",
    ],
}
_POS_RX = {k: [re.compile(p) for p in v] for k, v in _POS_RX.items()}
_POS_ROUTINE = {"seated": "pos_sitting", "standing": "pos_standing", "floor": "pos_floor"}


def _position_requests(clean: str) -> list[str]:
    return [k for k, rxs in _POS_RX.items() if any(rx.search(clean) for rx in rxs)]


# ---- durations -------------------------------------------------------------
_NUMW = ("twenty five|forty five|one|two|three|four|five|six|seven|eight|nine|ten|"
         "eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|ninety")
_W2N = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "fifteen": 15, "twenty": 20, "twenty five": 25, "thirty": 30,
        "forty": 40, "forty five": 45, "fifty": 50, "sixty": 60, "ninety": 90}
_HALF_RX = re.compile(r"\bhalf\s+(?:an?\s+)?hour\b")
_QUARTER_RX = re.compile(r"\b(?:a\s+)?quarter\s+(?:of\s+)?(?:an?\s+)?hour\b")
_HOURHALF_RX = re.compile(r"\bhour\s+and\s+(?:a\s+)?half\b")
_FEW_RX = re.compile(r"\b(?:a\s+)?(few|couple(?:\s+of)?)\s+(?:minutes?|mins?)\b")
_NUM_RX = re.compile(
    r"\b(?:(\d+(?:\.\d+)?)|(" + _NUMW + r"|an?))\s*(?:(?:-|to)\s*\d+(?:\.\d+)?\s*)?"
    r"(hours?|hrs?|minutes?|mins?)\b")
_QUICK_RX = re.compile(r"\b(quick|quickly|short(?!\s+of\b)|brief|briefly|in a hurry|"
                       r"hurry|little time|no time|not much time)\b")
_AVAIL_RX = re.compile(r"\b(?:have|has|got|only|spare|free|available|just|within|"
                       r"takes?|need|want|about|around|roughly)\s*"
                       r"(?:(?:about|around|roughly|only|just|at least)\s+)*$")
_EXPOSURE_VERB_RX = re.compile(r"\b(?:sit|sits|sitting|sat|stand|stands|standing|stood|"
                               r"work|works|working|desk|shift|chair|drive|driving|"
                               r"commute|travel|walk|walking|study|studying|typing|"
                               r"screen|laptop|computer|phone|lying|sleep|slept)\b")
_EXPOSURE_AFTER_RX = re.compile(r"^\s*(?:a|per|each|every)\s+(?:day|shift|night)\b|"
                                r"^\s*(?:straight|daily|continuously|at a stretch|everyday)\b")
#: "quick" has no number in it; this is the one assumption the module makes.
QUICK_MINUTES = 10


def _durations(clean: str) -> tuple[list[tuple[str, float]], list[str], int | None]:
    """``([(text, minutes), ...] available, [exposure texts], minutes)``.

    A number of minutes can mean *time I have* ("I only have 10 minutes") or
    *how long I do something* ("I sit for 9 hours").  Only the first is a
    constraint on the session, so the context before the number decides; hours
    of 2 or more are never a session length.
    """
    spans: list[tuple[int, int, float]] = []

    def add(m: re.Match, minutes: float) -> None:
        if not any(s < m.end() and m.start() < e for s, e, _ in spans):
            spans.append((m.start(), m.end(), minutes))

    for m in _HALF_RX.finditer(clean):
        add(m, 30.0)
    for m in _QUARTER_RX.finditer(clean):
        add(m, 15.0)
    for m in _HOURHALF_RX.finditer(clean):
        add(m, 90.0)
    for m in _FEW_RX.finditer(clean):
        add(m, 2.0 if m.group(1).startswith("couple") else 5.0)
    for m in _NUM_RX.finditer(clean):
        value = float(m.group(1)) if m.group(1) else float(_W2N[m.group(2)])
        add(m, value * (60.0 if m.group(3).startswith(("h",)) else 1.0))

    available: list[tuple[str, float]] = []
    exposure: list[str] = []
    for start, end, minutes in sorted(spans):
        text = clean[start:end].strip()
        before = re.split(r"[.,;!?\n]", clean[max(0, start - 40):start])[-1]
        after = clean[end:end + 20]
        if _AVAIL_RX.search(before):
            is_exposure = False
        elif _EXPOSURE_VERB_RX.search(before) or _EXPOSURE_AFTER_RX.search(after):
            is_exposure = True
        else:
            is_exposure = minutes >= 120
        if is_exposure:
            exposure.append(text)
        else:
            available.append((text, minutes))
    if available:
        return available, exposure, int(round(available[0][1]))
    quick = _QUICK_RX.search(clean)
    if quick:
        return [(quick.group(1), float(QUICK_MINUTES))], exposure, QUICK_MINUTES
    return [], exposure, None


# =========================================================================
# c. SENTIMENT / MOOD
# =========================================================================
MOODS = ("stressed", "tired", "low", "energetic", "calm")     # tie-break order

# (mood, weight, phrases).  A phrase may appear in several rows (burn-out is
# both stressed and tired); weights add.
_MOOD_SRC = (
    ("stressed", 1.0, ["stress", "stressed", "stressful", "anxious", "anxiety",
                       "overwhelmed", "panic", "panicking", "panicky", "burnout",
                       "burned out", "burnt out"]),
    ("stressed", 0.8, ["worried", "worry", "nervous", "overthinking", "racing thoughts",
                       "mind racing", "mind keeps racing", "can not relax", "irritable",
                       "frustrated", "restless", "on edge", "edgy", "agitated"]),
    ("stressed", 0.6, ["tense", "tension", "deadline", "angry", "annoyed"]),
    ("stressed", 0.9, ["insomnia", "sleepless", "can not sleep", "could not sleep",
                       "trouble sleeping", "difficulty sleeping", "unable to sleep",
                       "sleep nahi"]),
    ("tired", 0.4, ["insomnia", "sleepless", "can not sleep", "could not sleep",
                    "trouble sleeping", "difficulty sleeping", "unable to sleep",
                    "sleep nahi"]),
    ("tired", 1.0, ["tired", "exhausted", "exhaustion", "fatigue", "fatigued",
                    "drained", "worn out", "no energy", "low energy", "lack of energy",
                    "out of energy", "run down", "lethargic", "weary"]),
    ("tired", 0.8, ["sleepy", "drowsy", "sluggish", "lazy", "tiredness", "burned out",
                    "burnt out", "night shift"]),
    ("tired", 0.9, ["did not sleep", "not sleeping well", "slept badly", "slept poorly",
                    "bad sleep", "poor sleep", "not slept"]),
    ("stressed", 0.3, ["did not sleep", "not sleeping well", "slept badly",
                       "slept poorly", "bad sleep", "poor sleep", "not slept"]),
    ("low", 1.0, ["sad", "depressed", "depression", "unhappy", "hopeless", "miserable",
                  "lonely", "feel low", "feeling low", "feel down", "feeling down",
                  "demotivated", "unmotivated", "no motivation", "lost interest",
                  "gloomy", "moody"]),
    ("energetic", 1.0, ["energetic", "energised", "energized", "pumped", "full of energy",
                        "lots of energy", "high energy", "fired up", "ready to go",
                        "vibrant", "lively", "refreshed"]),
    ("energetic", 0.7, ["motivated", "feel strong", "feeling strong"]),
    ("calm", 1.0, ["calm", "relaxed", "relax", "peaceful", "serene", "tranquil",
                   "composed", "at peace", "at ease", "chilled", "content"]),
    ("calm", 0.5, ["happy", "good mood", "feel good", "feeling good", "feel great",
                   "feeling great", "feel fine", "feeling fine", "feel better",
                   "feeling better"]),
)
_MOOD_LEX: dict[tuple[str, ...], list[tuple[str, float]]] = {}
for _mood, _w, _phrases in _MOOD_SRC:
    for _ph in _phrases:
        _key = _phrase_stems(_ph)
        if not _key:
            continue
        _entries = _MOOD_LEX.setdefault(_key, [])
        # stress / stressed / stressful share one stem: keep the largest weight
        # per mood rather than counting the same word three times
        _old = next((e for e in _entries if e[0] == _mood), None)
        if _old is None:
            _entries.append((_mood, _w))
        elif _w > _old[1]:
            _entries[_entries.index(_old)] = (_mood, _w)
#: Polarity of each mood word, for the sentiment score.
_MOOD_POLARITY = {"stressed": -1.0, "tired": -0.8, "low": -1.3, "energetic": 1.0,
                  "calm": 0.8}
#: Calm-type words that are really a request when a want/need cue precedes them.
_CALM_GOAL_WORDS = frozenset(stem(w) for w in ("relax", "calm", "peace"))
#: Mood words that, when negated, point at their opposite.
_MOOD_OPPOSITE = {"energetic": "tired", "calm": "stressed"}

# Single-word polarity for everything that is not a mood word.
_SENT_SRC = {
    -1.0: ["pain", "hurt", "ache", "cramp", "injured", "injury", "bad", "worse",
           "suffer", "hate", "severe", "sore"],
    -0.8: ["stiff", "uncomfortable", "discomfort", "struggle", "annoyed", "swollen"],
    -1.5: ["terrible", "awful", "horrible", "worst", "unbearable"],
    -0.6: ["weak", "problem", "trouble", "difficult", "numb", "heavy", "tight"],
    1.0: ["good", "better", "enjoy", "relieved", "improve"],
    1.3: ["great", "best", "love", "wonderful", "excellent", "glad", "happy"],
    0.6: ["fine", "nice", "comfortable", "healthy", "fresh", "helpful", "thank",
          "strong", "easy", "well"],
}
_SENT_LEX: dict[tuple[str, ...], float] = {}
for _w, _words in _SENT_SRC.items():
    for _ph in _words:
        _SENT_LEX.setdefault(_phrase_stems(_ph), _w)
for _k in list(_SENT_LEX):                         # mood words are scored via _MOOD_POLARITY
    if _k in _MOOD_LEX:
        del _SENT_LEX[_k]


@dataclass
class _MoodResult:
    mood: str
    sentiment: float
    scores: dict[str, float]
    evidence: list[str]
    negated: list[str]
    goals: list[str]
    negated_spans: list[tuple[int, int]] = field(default_factory=list)


def _mood_and_sentiment(a: _Analysis) -> _MoodResult:
    scores = dict.fromkeys(MOODS, 0.0)
    evidence: list[str] = []
    negated: list[str] = []
    spans: list[tuple[int, int]] = []
    goals: list[str] = []
    total = 0.0

    # mood phrases
    for s, e, entries in _find(a.stems, _MOOD_LEX):
        surface = " ".join(a.raws[s:e])
        mod = _modifier(a.raws, s)
        # entries is the stored list [(mood, weight), ...] for this phrase
        is_calm_word = e - s == 1 and a.stems[s] in _CALM_GOAL_WORDS
        if is_calm_word and _goal_cue(a.raws, s):          # "I want to relax" is a goal
            goals.append("relax")
            scores["stressed"] += 0.4            # a request, not a state: stays below the 0.5 label threshold
            total -= 0.3
            continue
        if _negated(a.raws, s, e):
            negated.append(surface)
            spans.append((s, e))
            for mood, w in entries:
                opp = _MOOD_OPPOSITE.get(mood)
                if opp:
                    scores[opp] += 0.5 * w
                total -= 0.5 * _MOOD_POLARITY[mood] * w      # "not tired" is mildly positive
            continue
        evidence.append(surface)
        for mood, w in entries:
            scores[mood] += w * mod
            total += _MOOD_POLARITY[mood] * w * mod

    # everything else with a polarity
    for s, e, w in _find(a.stems, _SENT_LEX):
        mod = _modifier(a.raws, s)
        total += (-0.5 * w if _negated(a.raws, s, e) else w) * mod

    best = max(MOODS, key=lambda m: (round(scores[m], 6), -MOODS.index(m)))
    mood = best if scores[best] >= 0.5 else "neutral"
    sentiment = round(total / math.sqrt(total * total + 3.0), 3)
    return _MoodResult(mood, sentiment, {m: round(v, 3) for m, v in scores.items()},
                       evidence, negated, goals, spans)


def analyse_mood(text: str) -> tuple[str, float]:
    """``(mood, sentiment)``: a label from stressed/tired/low/energetic/calm/neutral
    and a polarity in -1..1.  Negation is handled ('not tired' is not tired)."""
    r = _mood_and_sentiment(_analyse(text))
    return r.mood, r.sentiment


# =========================================================================
# e. SAFETY FLAGS + MEDICATION CAUTIONS
# =========================================================================
# Cautions only - never dosing, never diagnosis, always a pointer to the doctor.
_FLAG_CAUTIONS = {
    "high_bp": "High blood pressure: skip head-below-heart poses, breathe steadily "
               "and never hold your breath.",
    "low_bp": "Low blood pressure: rise slowly after floor poses to avoid dizziness.",
    "bp_unspecified": "You mentioned a blood-pressure problem: head-below-heart poses "
                      "are skipped to be safe, and rise slowly from the floor.",
    "diabetic": "Diabetes: practise 1-2 hours after a meal, keep a glucose source "
                "within reach and stop if shaky or light-headed.",
    "pregnant": "Pregnancy: avoid lying on your belly, deep twists and holding your "
                "breath, and keep cool - get a doctor's OK before practising.",
    "joint_pain": "Joint pain: stay inside a comfortable range, pad the knees and "
                  "wrists, and stop if a joint hurts sharply.",
    "asthma": "Breathing condition: keep your inhaler within reach, breathe steadily "
              "and stop if you wheeze.",
    "spine_injury": "Disc or spine problem: avoid deep forward folds and strong "
                    "backbends, stay gentle, and stop if pain shoots into a leg.",
    "heart_condition": "Heart condition: keep effort light, avoid head-below-heart "
                       "poses and stop at chest discomfort or breathlessness.",
    "glaucoma": "Eye-pressure condition: avoid head-below-heart poses.",
    "osteoporosis": "Weak bones: avoid forcing deep folds, twists or backbends.",
    "hernia": "Hernia: avoid straining or deep compression of the belly.",
    "recent_surgery": "Recent surgery or fracture: get clearance before practising and "
                      "avoid head-below-heart poses.",
    "migraine": "Migraine: avoid head-below-heart poses and stop if a headache starts.",
    "vertigo": "Dizziness: avoid head-below-heart poses and rise slowly.",
    "red_flag": "Chest pain, fainting or numbness are not things to stretch out: stop "
                "and get medical advice before practising.",
}
_FLAG_ORDER = tuple(_FLAG_CAUTIONS)

#: Conditions -> the flags they raise.  ``avoid_inversion`` is the one the
#: personaliser acts on; it only ever removes poses.
_COND_FLAGS = {
    "diabetes": {"diabetic"},
    "high_bp": {"high_bp", "avoid_inversion"},
    "low_bp": {"low_bp"},
    "bp": {"bp_unspecified", "avoid_inversion"},
    "pregnancy": {"pregnant", "avoid_inversion"},
    "arthritis": {"joint_pain"},
    "asthma": {"asthma"},
    "slip_disc": {"spine_injury"}, "sciatica": {"spine_injury"},
    "scoliosis": {"spine_injury"},
    "heart": {"heart_condition", "avoid_inversion"},
    "glaucoma": {"glaucoma", "avoid_inversion"},
    "osteoporosis": {"osteoporosis"},
    "hernia": {"hernia"},
    "surgery": {"recent_surgery", "avoid_inversion"},
    "migraine": {"migraine", "avoid_inversion"},
}

# class key -> (names/phrases, flags, caution).  The text is deliberately
# general: what a class of medicine can do to someone exercising, not what to
# take or how much.
_MED_TABLE = (
    ("insulin", ["insulin", "lantus", "novorapid", "mixtard", "huminsulin"],
     {"diabetic", "on_blood_sugar_medication"},
     "Insulin can lower blood sugar during activity (hypoglycaemia risk): keep a fast "
     "sugar source within reach, practise after a light meal, and stop if shaky, sweaty "
     "or dizzy."),
    ("diabetes_tablets", ["metformin", "glycomet", "glimepiride", "gliclazide",
                          "glipizide", "sitagliptin", "sugar tablet", "sugar tablets",
                          "diabetes tablet", "diabetes medicine", "diabetic medicine"],
     {"diabetic", "on_blood_sugar_medication"},
     "Diabetes tablets: some can lower blood sugar during activity, so keep a quick "
     "sugar source nearby, do not practise on an empty stomach, and stop if shaky or "
     "light-headed."),
    ("bp_tablets", ["amlodipine", "telmisartan", "losartan", "olmesartan", "enalapril",
                    "ramipril", "lisinopril", "norvasc", "telma", "bp tablet",
                    "bp tablets", "bp medicine", "bp medication", "bp pill", "bp pills"],
     {"on_bp_medication", "avoid_inversion"},
     "Blood-pressure tablets can make you dizzy when you stand up quickly: rise slowly "
     "after floor poses and avoid holding head-down poses."),
    ("beta_blocker", ["atenolol", "metoprolol", "propranolol", "bisoprolol", "carvedilol",
                      "beta blocker", "beta blockers"],
     {"on_bp_medication", "avoid_inversion"},
     "Beta blockers can slow your heart rate and blunt how effort feels: judge effort "
     "by how you feel rather than your pulse, move gently and rise slowly."),
    ("blood_thinner", ["warfarin", "acitrom", "blood thinner", "blood thinners", "heparin",
                       "apixaban", "eliquis", "rivaroxaban", "xarelto", "dabigatran"],
     {"on_anticoagulant", "avoid_inversion"},
     "Blood thinners raise bleeding and bruising risk: avoid deep inversions, sudden or "
     "forceful movements, and anything with a risk of falling."),
    ("antiplatelet", ["aspirin", "ecosprin", "clopidogrel", "disprin"],
     {"on_antiplatelet"},
     "Aspirin-type medicines make bruising easier: pad the knees and wrists and avoid "
     "forceful movements."),
    ("painkillers", ["ibuprofen", "diclofenac", "naproxen", "combiflam", "brufen",
                     "voveran", "painkiller", "painkillers", "pain killer", "pain killers"],
     {"on_painkiller"},
     "Painkillers can hide pain, so you may stretch further than is safe: stay well "
     "inside your comfortable range."),
    ("statin", ["atorvastatin", "rosuvastatin", "simvastatin", "statin", "statins",
                "lipitor"],
     {"on_statin"},
     "Statins can cause muscle aches: ease off and stop if you feel unusual muscle pain "
     "or weakness."),
    ("sedating", ["alprazolam", "clonazepam", "diazepam", "zolpidem", "sertraline",
                  "escitalopram", "amitriptyline", "antidepressant", "antidepressants",
                  "sleeping pill", "sleeping pills", "sleeping tablet", "anxiety tablet",
                  "anxiety tablets"],
     {"on_sedative"},
     "Some of these medicines cause drowsiness or dizziness: practise near a wall or "
     "chair, move slowly and rise slowly."),
    ("inhaler", ["salbutamol", "asthalin", "budesonide", "formoterol", "montelukast",
                 "inhaler", "inhalers"],
     {"asthma"},
     "Keep your inhaler within reach, breathe steadily and stop if you wheeze."),
    ("steroid", ["prednisolone", "prednisone", "wysolone", "dexamethasone", "steroid",
                 "steroids"],
     {"on_steroid"},
     "Long-term steroids can weaken bones and joints: avoid forcing stretches or deep "
     "folds."),
    ("diuretic", ["furosemide", "lasix", "spironolactone", "hydrochlorothiazide",
                  "water pill", "water pills", "water tablet", "water tablets"],
     {"on_diuretic"},
     "Water tablets can cause dehydration and dizziness: sip water and rise slowly."),
    ("thyroid", ["levothyroxine", "thyroxine", "thyronorm", "eltroxin", "thyroid tablet",
                 "thyroid tablets", "thyroid medicine"],
     {"on_thyroid"},
     "Thyroid medicine: if your heart races or you feel jittery, rest and slow your "
     "breathing."),
    ("prenatal", ["prenatal", "prenatal vitamin", "prenatal vitamins"],
     {"pregnant", "avoid_inversion"},
     "Prenatal supplements suggest pregnancy: avoid lying on your belly, deep twists "
     "and head-below-heart poses."),
)
_MED_LEX = _compile({row[0]: row[1] for row in _MED_TABLE})
_MED_BY_KEY = {row[0]: row for row in _MED_TABLE}
GENERIC_MEDICATION_CAUTION = ("Medicines can change how exercise affects you: practise "
                              "gently, rise slowly from the floor, and stop if you feel "
                              "unwell. " + DOCTOR)
_NONE_WORDS = frozenset({"none", "nil", "na", "n", "a", "nothing", "no", "nope", "nahi"})


def _med_hits(a: _Analysis) -> list[str]:
    """Medication class keys mentioned (and not negated), in table order."""
    found = {canon for s, e, canon in _find(a.stems, _MED_LEX)
             if not _negated(a.raws, s, e)}
    return [row[0] for row in _MED_TABLE if row[0] in found]


def _condition_hits(a: _Analysis) -> list[str]:
    out: list[str] = []
    for s, e, canon in _find(a.stems, _COND_LEX):
        if _negated(a.raws, s, e):
            continue
        if canon == "surgery" and re.search(r"\byears?\b", a.clean):   # "surgery 5 years ago"
            continue
        if canon not in out:
            out.append(canon)
    # blood pressure: read high/low from the neighbouring words
    for s, e, _ in _find(a.stems, _BP_LEX):
        if _negated(a.raws, s, e):
            continue
        near = {t for t in a.raws[max(0, s - 3):e + 3] if t != BOUNDARY}
        if near & _BP_FINE:
            continue
        kind = ("high_bp" if near & _BP_HIGH else "low_bp" if near & _BP_LOW else "bp")
        if kind not in out:
            out.append(kind)
    return out


def _flags_from(conditions: list[str], symptoms: list[str], parts: list[str],
                meds: list[str]) -> set[str]:
    flags: set[str] = set()
    for c in conditions:
        flags |= _COND_FLAGS.get(c, set())
    if "dizziness" in symptoms:
        flags |= {"vertigo", "avoid_inversion"}
    if "fainting" in symptoms or "numbness" in symptoms \
            or ("chest" in parts and "pain" in symptoms):
        flags.add("red_flag")
    if "pain" in symptoms and set(parts) & _JOINT_PARTS:
        flags.add("joint_pain")
    for m in meds:
        flags |= _MED_BY_KEY[m][2]
    return flags


def safety_flags(text: str) -> set[str]:
    """Flags raised by conditions *and* medicines in free text.

    ``avoid_inversion`` is the one the personaliser acts on.  Negation is
    respected: "I don't have diabetes" raises nothing.
    """
    a = _analyse(text)
    ent = _extract(a)
    return _flags_from(ent.entities["condition"], ent.entities["symptom"],
                       ent.entities["body_part"], _med_hits(a))


def flag_cautions(flags) -> list[str]:
    """One caution sentence per flag that has one, in a fixed order, each ending
    with 'Check with your doctor.'"""
    return [f"{_FLAG_CAUTIONS[f]} {DOCTOR}" for f in _FLAG_ORDER if f in flags]


def medication_flags(text: str) -> list[str]:
    """Cautions for the medicines named in free text.

    Maps names and classes (insulin, metformin, amlodipine, telmisartan,
    atenolol / beta blocker, warfarin / blood thinner, painkillers, ...) to a
    short caution.  Cautions only: no dose, no advice to start or stop anything,
    and each ends with 'Check with your doctor.'

    Text that names something we do not recognise gets one generic caution
    instead of silence, because "no caution" would be a false reassurance.
    Empty text, 'none', and negations ("I take no medicines") give ``[]``.
    """
    a = _analyse(text)
    hits = _med_hits(a)
    if hits:
        return [f"{_MED_BY_KEY[k][3]} {DOCTOR}" for k in hits]
    words = [r for r in a.raws if r != BOUNDARY and not r[0].isdigit()]
    if not words or all(w in _NONE_WORDS for w in words) or any(w in _NEG_PRE for w in words):
        return []
    return [GENERIC_MEDICATION_CAUTION]


# =========================================================================
# extraction: put the lexicons to work
# =========================================================================
ENTITY_KEYS = ("body_part", "symptom", "condition", "job", "duration", "exposure",
               "goal", "mood_word", "negated", "medication", "position_request")


@dataclass
class _Extraction:
    entities: dict[str, list[str]]
    minutes: int | None
    mood: _MoodResult
    #: content tokens minus anything the person negated ("not stressed" must not
    #: pull the stress routine in through its word overlap)
    query: list[str]


def _add(lst: list[str], v: str) -> None:
    if v not in lst:
        lst.append(v)


def _extract(a: _Analysis) -> _Extraction:
    ent: dict[str, list[str]] = {k: [] for k in ENTITY_KEYS}
    dropped: set[int] = set()            # raw-token positions the person negated

    def drop(s: int, e: int) -> None:
        dropped.update(range(s, e))

    for s, e, canon in _find(a.stems, _BODY_LEX):
        if canon == "back":
            prev = a.raws[s - 1] if s > 0 else ""
            nxt = a.raws[e] if e < len(a.raws) else ""
            if prev in _BACK_PREV or nxt in _BACK_NEXT:
                continue
        _add(ent["body_part"], canon)
    for s, e, canon in _find(a.stems, _SYMPTOM_LEX):
        if _negated(a.raws, s, e):
            drop(s, e)
        else:
            _add(ent["symptom"], canon)
    for c in _condition_hits(a):
        _add(ent["condition"], c)

    ent["position_request"] = _position_requests(a.clean)
    for s, e, canon in _find(a.stems, _JOB_LEX):
        # "standing poses" / "do it sitting" name a position, not a job
        if canon == "sitting" and a.stems[s:e] in (["sit"], ["seat"]) \
                and "seated" in ent["position_request"]:
            continue
        if canon == "standing" and a.stems[s:e] == ["stand"] \
                and "standing" in ent["position_request"]:
            continue
        if _negated(a.raws, s, e):
            drop(s, e)
        else:
            _add(ent["job"], canon)

    avail, exposure, minutes = _durations(a.clean)
    for text, _m in avail:
        _add(ent["duration"], text)
    for text in exposure:
        _add(ent["exposure"], text)

    mood = _mood_and_sentiment(a)
    for s, e in mood.negated_spans:
        drop(s, e)
    for w in mood.evidence:
        _add(ent["mood_word"], w)
    for w in mood.negated:
        _add(ent["negated"], w)
    # Negation applies to *states* (mood, symptom, condition, job), not to goals:
    # "I can't touch my toes" negates an ability and is exactly a flexibility need.
    for s, e, canon in _find(a.stems, _GOAL_LEX):
        _add(ent["goal"], canon)
    for g in mood.goals:
        _add(ent["goal"], g)
    for key in _med_hits(a):
        _add(ent["medication"], key)
    query = [t for i, t in zip(a.content_idx, a.content) if i not in dropped]
    return _Extraction(ent, minutes, mood, query)


def extract_entities(text: str) -> dict[str, list[str]]:
    """Slots found in the text: body_part, symptom, condition, job, duration,
    exposure (how long they sit/stand - *not* time available), goal, mood_word,
    negated, medication (class keys), position_request."""
    return _extract(_analyse(text)).entities


def parse_minutes(text: str) -> int | None:
    """Minutes the person says they have, or None.  'quick' counts as
    ``QUICK_MINUTES`` (an assumption, reported as such in the explanation)."""
    return _extract(_analyse(text)).minutes


# =========================================================================
# d. INTENT -> ROUTINE RECOMMENDATION  (TF-IDF + cosine + rule boosts)
# =========================================================================
# Curated symptom / synonym text per routine.  This is the "training data" of
# the recommender: plain words a person might use, read by the same tokeniser
# as the user's sentence.
_CURATED = {
    "desk_sitting": "sitting all day desk job office worker computer laptop chair long "
                    "hours sedentary stiff back lower back ache tight hips hip flexors "
                    "slouch posture rounded shoulders stiffness after sitting work from "
                    "home software developer programmer typing",
    "standing_workers": "standing all day on my feet shop shopkeeper teacher nurse "
                        "factory guard retail waiter swollen feet tired legs aching "
                        "calves knee pain ankle heel foot pain standing job sore legs "
                        "varicose",
    "homemakers": "homemaker housewife household chores housework cooking kitchen "
                  "cleaning sweeping laundry bending lifting kids family daily routine "
                  "sore back tired legs aching feet stay at home mother",
    "students": "student study studying exam college homework reading phone laptop "
                "screen text neck neck pain stiff neck shoulders slouch concentration "
                "focus mind long hours lecture notes sleepy",
    "lumbar_flex": "lumbar flexibility flexible stiff tight hamstrings core strength "
                   "stretch mobility general flexibility lower back care weak core hips "
                   "legs thigh strong supple stiffness bending touch toes",
    "daily_energy": "energy energetic tired low energy fatigue lazy sluggish morning "
                    "wake up refreshed full body workout activate strength balance "
                    "stamina motivation boost lethargic exhausted active start the day "
                    "fit fitness stay fit general overall sad low mood",
    "back_pain": "back pain backache lower back pain lumbar spine stiff back sore back "
                 "slip disc sciatica spine care hurts ache relief tightness",
    "stress_relief": "stress stressed anxious anxiety tension worried calm relax "
                     "relaxation peace mindfulness meditation breathing breath focus "
                     "sleep insomnia sleepless overwhelmed restless mind unwind "
                     "burnout nervous panic",
    "pos_standing": "standing poses only stay on my feet upright no floor work no mat "
                    "cannot get down on the floor",
    "pos_sitting": "sitting seated poses chair cross legged sit on the floor meditation "
                   "cannot stand breathing",
    "pos_floor": "floor poses lying down kneeling mat hands and knees on the ground prone",
}
W_COS = 1.5                  #: weight of TF-IDF cosine in the final score
NO_SIGNAL = 0.05             #: below this the recommender says it has nothing to go on
_DEFAULT_ORDER = ("daily_energy", "desk_sitting", "standing_workers", "homemakers",
                  "students", "lumbar_flex", "back_pain", "stress_relief",
                  "pos_standing", "pos_sitting", "pos_floor")
#: Per pose: seconds in the launcher's 5 s get-ready + hold + a few seconds to release.
_SECONDS_OVERHEAD = 10


def estimate_minutes(routine: routines.Routine) -> float:
    secs = sum(routines.POSES[k].hold_s + _SECONDS_OVERHEAD for k in routine.poses)
    return secs / 60.0


def _doc_text(r: routines.Routine) -> str:
    pose_text = " ".join(f"{routines.POSES[k].name} {routines.POSES[k].sanskrit} "
                         f"{routines.POSES[k].benefit}" for k in r.poses)
    return f"{r.title} {r.title} {r.target} {pose_text} {_CURATED.get(r.key, '')}"


@dataclass
class _Index:
    keys: list[str]
    vocab: dict[str, int]
    idf: np.ndarray
    matrix: np.ndarray          #: (n_routines, n_terms), rows L2-normalised


def _tfidf_vector(tokens: list[str], vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    v = np.zeros(len(vocab))
    for t in tokens:
        j = vocab.get(t)
        if j is not None:
            v[j] += 1.0
    nz = v > 0
    v[nz] = 1.0 + np.log(v[nz])            # sublinear term frequency
    v *= idf
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


@lru_cache(maxsize=1)
def _index() -> _Index:
    """Build the TF-IDF space once: smoothed idf = ln((1+N)/(1+df)) + 1."""
    rts = routines.all_routines()
    keys = list(rts)
    docs = [tokenize(_doc_text(rts[k])) for k in keys]
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    df = np.zeros(len(vocab))
    for d in docs:
        for t in set(d):
            df[vocab[t]] += 1.0
    n = len(keys)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    matrix = np.vstack([_tfidf_vector(d, vocab, idf) for d in docs])
    return _Index(keys, vocab, idf, matrix)


def tfidf_scores(text: str) -> list[tuple[str, float]]:
    """Cosine similarity of the text to every routine description (no rules)."""
    ix = _index()
    q = _tfidf_vector(tokenize(text), ix.vocab, ix.idf)
    sims = ix.matrix @ q
    return [(k, round(float(s), 4)) for k, s in zip(ix.keys, sims)]


def _inverse_vocab() -> dict[int, str]:
    return {i: t for t, i in _index().vocab.items()}


def _rules(ent: dict[str, list[str]], mood: _MoodResult) -> dict[str, list[tuple[float, str]]]:
    """Slot -> routine boosts.  Each carries the reason that is shown to the user."""
    R: dict[str, list[tuple[float, str]]] = {}

    def add(key: str, amount: float, why: str) -> None:
        R.setdefault(key, []).append((amount, why))

    parts = set(ent["body_part"])
    symptoms = set(ent["symptom"])
    pain = bool(symptoms & {"pain", "stiffness"})
    with_pain = " with pain or stiffness" if pain else ""

    if "lower back" in parts:
        add("back_pain", 0.45 + (0.15 if pain else 0.0),
            f"you mentioned your lower back{with_pain}")
        add("lumbar_flex", 0.15, "lower-back (lumbar) care")
    elif "back" in parts:
        add("back_pain", 0.35 + (0.15 if pain else 0.0),
            f"you mentioned your back{with_pain}")
    if "upper back" in parts:
        add("back_pain", 0.2, "you mentioned your upper back")
        add("desk_sitting", 0.1, "upper-back tightness is common after desk work")
    if "spine" in parts:
        add("back_pain", 0.15, "you mentioned your spine")
    if set(ent["condition"]) & {"slip_disc", "sciatica", "scoliosis"}:
        add("back_pain", 0.4, "you mentioned a disc, sciatica or spine condition")
    if "neck" in parts:
        add("students", 0.3, f"you mentioned your neck{with_pain} (text-neck care)")
        add("desk_sitting", 0.15, "neck strain is common after screen work")
    if "shoulder" in parts:
        add("students", 0.15, "you mentioned your shoulders")
        add("desk_sitting", 0.15, "you mentioned your shoulders")
    if "hip" in parts:
        add("desk_sitting", 0.2, "you mentioned your hips")
        add("lumbar_flex", 0.1, "hip mobility")
    if parts & {"knee", "ankle", "foot", "leg"}:
        named = ", ".join(sorted(parts & {"knee", "ankle", "foot", "leg"}))
        add("standing_workers", 0.3, f"you mentioned your {named} (strain from standing)")
        add("homemakers", 0.1, f"you mentioned your {named}")
    if parts & {"wrist", "hand", "elbow"}:
        add("desk_sitting", 0.2, "wrist and hand strain from typing")

    job_why = {
        "sitting": ("desk_sitting", 0.4, "you described desk or seated work"),
        "standing": ("standing_workers", 0.4, "you described standing work"),
        "homemaker": ("homemakers", 0.5, "you described a household routine"),
        "student": ("students", 0.5, "you described studying"),
    }
    for j in ent["job"]:
        key, amt, why = job_why[j]
        add(key, amt, why)

    for m in MOODS:
        score = min(1.0, mood.scores[m])
        if score <= 0:
            continue
        words = ", ".join(mood.evidence) or m
        if m == "stressed":
            add("stress_relief", round(0.55 * score, 3), f"you sound stressed ('{words}')")
        elif m == "tired":
            add("daily_energy", round(0.45 * score, 3), f"you sound tired ('{words}')")
        elif m == "low":
            add("daily_energy", round(0.3 * score, 3), f"you sound low ('{words}')")
            add("stress_relief", round(0.1 * score, 3), "a gentle calming practice can help low mood")
        elif m == "energetic":
            add("daily_energy", round(0.4 * score, 3), f"you sound energetic ('{words}')")

    goal_rules = {
        "relax": (("stress_relief", 0.35),), "energy": (("daily_energy", 0.35),),
        "flexibility": (("lumbar_flex", 0.4),),
        "strength": (("lumbar_flex", 0.3), ("daily_energy", 0.1)),
        "posture": (("desk_sitting", 0.15), ("students", 0.15)),
        "focus": (("students", 0.2), ("stress_relief", 0.15)),
        "sleep": (("stress_relief", 0.3),), "weight": (("daily_energy", 0.35),),
        "fitness": (("daily_energy", 0.35),),
        "balance": (("daily_energy", 0.15), ("standing_workers", 0.1)),
    }
    for g in ent["goal"]:
        for key, amt in goal_rules.get(g, ()):
            add(key, amt, f"you want to {_GOAL_TEXT[g]}")

    for pr in ent["position_request"]:
        add(_POS_ROUTINE[pr], 0.9, f"you asked for {pr} poses")
    return R


@dataclass
class _Candidate:
    key: str
    cosine: float
    reasons: list[tuple[float, str]]
    penalties: list[tuple[float, str]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return (W_COS * self.cosine + sum(a for a, _ in self.reasons)
                + sum(a for a, _ in self.penalties))


def _rank(ent: dict[str, list[str]], mood: _MoodResult, flags: set[str],
          minutes: int | None, tokens: list[str]) -> tuple[list[_Candidate], np.ndarray]:
    ix = _index()
    q = _tfidf_vector(tokens, ix.vocab, ix.idf)
    sims = ix.matrix @ q
    rules = _rules(ent, mood)
    rts = routines.all_routines()
    cands = []
    for i, key in enumerate(ix.keys):
        c = _Candidate(key, float(sims[i]), list(rules.get(key, [])))
        est = estimate_minutes(rts[key])
        if minutes is not None and est > minutes:
            c.penalties.append((-0.1, f"it takes about {est:.0f} min, longer than your "
                                      f"{minutes} min"))
        if "avoid_inversion" in flags and any(
                "inversion" in routines.POSES[p].tags for p in rts[key].poses):
            c.penalties.append((-0.1, "it contains a head-below-heart pose that would "
                                      "be skipped for you"))
        cands.append(c)
    order = {k: i for i, k in enumerate(_DEFAULT_ORDER)}
    cands.sort(key=lambda c: (-round(c.score, 6), order.get(c.key, len(order))))
    return cands, q


def _explain(cands: list[_Candidate], q: np.ndarray, ent: dict[str, list[str]],
             minutes: int | None, mood: _MoodResult, cautions: list[str],
             a: _Analysis) -> str:
    rts = routines.all_routines()
    top = cands[0]
    lines: list[str] = []
    no_signal = top.score < NO_SIGNAL
    if no_signal:
        lines.append("I could not find a clear need in that sentence, so I am suggesting "
                     f"the balanced '{rts[top.key].title}'. Try saying where it hurts, "
                     "what your day is like, or how you feel.")
    else:
        lines.append(f"Best match: {rts[top.key].title} (score {top.score:.2f}).")
        lines.append("Why:")
        for amt, why in top.reasons:
            lines.append(f"  - {why} (+{amt:.2f})")
        ix = _index()
        contrib = q * ix.matrix[ix.keys.index(top.key)]
        inv = _inverse_vocab()
        best = [a.surface(inv[j]) for j in np.argsort(-contrib)[:3] if contrib[j] > 1e-9]
        if best:
            lines.append(f"  - your wording is close to this routine's description - "
                         f"key words: {', '.join(best)} (+{W_COS * top.cosine:.2f})")
        if not top.reasons and not best:
            lines.append("  - it is the closest general fit")
        for amt, why in top.penalties:
            lines.append(f"  - but {why} ({amt:.2f})")
        if len(cands) > 1 and cands[1].score >= NO_SIGNAL:
            r2 = cands[1]
            why2 = r2.reasons[0][1] if r2.reasons else "similar wording"
            lines.append(f"Also considered: {rts[r2.key].title} ({r2.score:.2f}) because "
                         f"{why2}.")
    est = estimate_minutes(rts[top.key])
    if minutes is not None:
        fits = "fits" if est <= minutes else "is longer than"
        note = " ('quick' is read as about 10 minutes)" if ent["duration"] and \
            ent["duration"][0] in ("quick", "quickly", "short", "brief", "briefly") else ""
        lines.append(f"Time: about {est:.0f} min of poses, which {fits} your "
                     f"{minutes} min{note}.")
    else:
        lines.append(f"Time: about {est:.0f} min of poses.")
    if mood.negated:
        lines.append("You said you are not " + ", ".join(mood.negated)
                     + ", so I did not count that as a need.")
    if cautions:
        lines.append("Care: " + " ".join(cautions))
    return "\n".join(lines)


@dataclass
class Understanding:
    """Everything the pipeline found in one piece of text."""

    tokens: list[str]
    entities: dict[str, list[str]]
    mood: str
    sentiment: float
    flags: set[str]
    minutes: int | None
    ranking: list[tuple[str, float]]
    explanation: str
    cautions: list[str] = field(default_factory=list)
    mood_scores: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> str:
        return self.ranking[0][0]


def understand(text: str) -> Understanding:
    """Run the whole pipeline on ``text``.  Deterministic: same text, same answer."""
    a = _analyse(text)
    ex = _extract(a)
    ent = ex.entities
    flags = _flags_from(ent["condition"], ent["symptom"], ent["body_part"], _med_hits(a))
    cautions = flag_cautions(flags) + [
        f"{_MED_BY_KEY[k][3]} {DOCTOR}" for k in ent["medication"]]
    cands, q = _rank(ent, ex.mood, flags, ex.minutes, ex.query)
    explanation = _explain(cands, q, ent, ex.minutes, ex.mood, cautions, a)
    return Understanding(
        tokens=a.content, entities=ent, mood=ex.mood.mood, sentiment=ex.mood.sentiment,
        flags=flags, minutes=ex.minutes,
        ranking=[(c.key, round(c.score, 3)) for c in cands], explanation=explanation,
        cautions=cautions, mood_scores=ex.mood.scores)


def recommend(text: str) -> tuple[str, str]:
    """``(routine_key, explanation)`` for the best-matching routine."""
    u = understand(text)
    return u.best, u.explanation


# =========================================================================
# f. NATURAL-LANGUAGE GENERATION
# =========================================================================
_HELD_RX = re.compile(r"held\s+(\d+(?:\.\d+)?)\s*s\b")
_ALIGN_RX = re.compile(r"best alignment\s+(\d+(?:\.\d+)?)\s*%")


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def _join(names: list[str], cap: int = 3) -> str:
    shown = names[:cap]
    more = len(names) - len(shown)
    if more > 0:
        shown = shown + [f"{more} more"]
    if len(shown) <= 1:
        return "".join(shown)
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def _sentence(s: str) -> str:
    s = s.strip()
    s = s[0].upper() + s[1:] if s else s
    return s if s.endswith((".", "!", "?")) else s + "."


def session_summary(results: list[dict], profile=None, seed: int | None = None) -> str:
    """Two to four natural sentences about a finished session.

    ``results`` are the rows the launcher builds, one per pose:
    ``{"pose": name, "result": text, "completed": bool, "scored": bool}``.  The
    numbers come from the rows (``held 20 s, best alignment 87%``); the wording
    is chosen from templates with a ``random.Random`` seeded from the content,
    so the same session always reads the same way and different sessions vary.
    """
    rows = list(results or [])
    total = len(rows)
    done = [r for r in rows if r.get("completed")]
    c = len(done)
    first = ""
    name = (getattr(profile, "name", "") or "").strip()
    if name:
        first = name.split()[0]

    best_hold, best_align = None, None
    for r in done:
        txt = str(r.get("result", ""))
        m = _HELD_RX.search(txt)
        if m and (best_hold is None or float(m.group(1)) > best_hold[0]):
            best_hold = (float(m.group(1)), r.get("pose", "a pose"))
        m = _ALIGN_RX.search(txt)
        if m and (best_align is None or float(m.group(1)) > best_align[0]):
            best_align = (float(m.group(1)), r.get("pose", "a pose"))
    left_out = sum(1 for r in rows if str(r.get("result", "")).startswith(
        "skipped for your profile"))

    key = f"{name}|{c}|{total}|{best_hold}|{best_align}|{left_out}"
    rng = random.Random(seed if seed is not None else zlib.crc32(key.encode("utf-8")))
    pick = rng.choice

    who = f", {first}" if first else ""
    pose_word = "pose" if total == 1 else "poses"
    names = [str(r.get("pose", "a pose")) for r in done]
    sentences: list[str] = []

    # 1. what was done
    if total == 0:
        sentences.append(pick([
            "No poses were recorded this session",
            "This session has no completed poses on record"]))
    elif c == 0:
        sentences.append(pick([
            "No poses were completed this time, and that is okay",
            f"You did not finish any of the {total} {pose_word} this time, "
            "and that is okay"]))
    elif c == total and total > 1:
        sentences.append(pick([
            f"Nice session: you completed all {total} poses ({_join(names)})",
            f"You completed all {total} poses today ({_join(names)})",
            f"That was all {total} poses done: {_join(names)}"]))
    else:
        sentences.append(pick([
            f"Nice session: you completed {c} of {total} {pose_word} "
            f"({_join(names)})",
            f"You completed {c} of {total} {pose_word} today ({_join(names)})",
            f"That was {c} of {total} {pose_word} done: {_join(names)}"]))

    # 2. the numbers, only when there are any
    if best_hold and best_align:
        if best_hold[1] == best_align[1]:
            sentences.append(
                f"Your best effort was {best_hold[1]}, held for {_num(best_hold[0])} s "
                f"with {_num(best_align[0])}% alignment")
        else:
            sentences.append(pick([
                f"Your longest hold was {_num(best_hold[0])} s in {best_hold[1]}, and "
                f"your best alignment was {_num(best_align[0])}% in {best_align[1]}",
                f"You held {best_hold[1]} for {_num(best_hold[0])} s, and your cleanest "
                f"alignment was {_num(best_align[0])}% in {best_align[1]}"]))
    elif best_hold:
        sentences.append(pick([
            f"Your longest hold was {_num(best_hold[0])} s in {best_hold[1]}",
            f"You held {best_hold[1]} the longest, for {_num(best_hold[0])} s"]))

    # 3. encouragement - the only sentence that uses the first name, so it
    # appears exactly once when there is one
    if total == 0 or c == 0:
        sentences.append(pick([
            f"Showing up is the hardest part{who}, so try again tomorrow, even for a few "
            "minutes",
            f"Even two minutes of movement counts{who}, so come back when you are ready"]))
    elif c == total:
        sentences.append(pick([
            f"Finishing the whole routine is what builds the habit, so well done{who}",
            f"Great consistency{who} - finishing every pose is exactly how progress adds up"]))
    elif c / total >= 0.5:
        sentences.append(pick([
            f"That is a solid effort{who}, and the rest will come with practice",
            f"Most of the routine is done{who}, which is a good place to build from"]))
    else:
        sentences.append(pick([
            f"Even a short session counts{who}, so be glad you showed up",
            f"Every pose you finish is progress{who}, and it gets easier each time"]))

    # 4. one tip, chosen from what applies to this person
    tips: list[str] = []
    if left_out:
        tips.append("remember that poses left out for your profile are there to keep "
                    "you safe, so there is no need to make them up")
    bp = getattr(profile, "bp", "None")
    if bp == "Low":
        tips.append("rise slowly after floor poses to avoid dizziness")
    elif bp == "High":
        tips.append("keep your breathing steady and never hold your breath in a pose")
    if getattr(profile, "diabetic", False):
        tips.append("keep a glucose source within reach whenever you practise")
    job = getattr(profile, "job", "")
    if job == "sitting":
        tips.append("stand up and walk for two minutes every hour of your working day")
    elif job == "standing":
        tips.append("finish your day with your feet up for a few minutes")
    if str(getattr(profile, "medication", "") or "").strip():
        tips.append("keep the cautions for your medicines in mind and ask your doctor "
                    "if anything feels wrong")
    if best_align and best_align[0] < 75:
        tips.append("focus on one cue at a time, such as a tall spine, to lift your "
                    "alignment score")
    if not tips:
        tips.append("drink some water and take a few slow breaths before you move on")
    sentences.append(pick(["One tip: ", "Tip for next time: "]) + pick(tips))

    return " ".join(_sentence(s) for s in sentences)
