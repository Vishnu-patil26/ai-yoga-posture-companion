# Validation

Reproduce: `python tools/fitting/build_library.py` -> `tools/fitting/calibrate_thresholds.py` -> `tools/evaluation/library_eval.py` -> `tools/evaluation/external_check.py` -> `tools/evaluation/pose_check.py` -> `tools/evaluation/make_validation.py`.

## 1. Recognition - is it the right pose? (`tools/evaluation/library_eval.py`)

Each held-out photo is scored against all ten poses; the best score is the prediction. **Overall 86% on 547 photos.**

| Pose | n | Recall | Wrongly taken for |
|---|---|---|---|
| adho_mukha | 107 | 100% | - |
| balasana | 12 | 92% | utkatasana x1 |
| bhujangasana | 132 | 93% | vrikshasana x9 |
| marjaryasana | 32 | 84% | bhujangasana x5 |
| sukhasana | 11 | 64% | bhujangasana x1, vrikshasana x3 |
| tadasana | 20 | 80% | bhujangasana x2, utkatasana x1, vrikshasana x1 |
| trikonasana | 23 | 65% | bhujangasana x8 |
| utkatasana | 100 | 94% | bhujangasana x1, tadasana x5 |
| virabhadrasana | 14 | 93% | bhujangasana x1 |
| vrikshasana | 96 | 62% | bhujangasana x37 |

Per-pose n is small (11-132): one photo moves a percentage by several points. Hold-outs are every 4th photo of each Hugging Face / Wikimedia source plus the TensorFlow set's own test split. After `calibrate_thresholds.py` these figures are slightly optimistic (section 3).

## 2. Correctness - does the score separate right form from wrong? (`tools/evaluation/external_check.py`)

Labelled photos from *Yoga for all* (Zenodo 7818789), never used for fitting: 'right steps' vs 'wrong steps' with the fault named. AUC 0.5 = no skill, 1.0 = perfect.

| Pose | right (n) | wrong (n) | mean right | mean wrong | AUC |
|---|---|---|---|---|---|
| tadasana | 120 | 60 | 79.1 | 72.8 | 0.602 |
| marjaryasana | 96 | 45 | 52.0 | 38.4 | 0.631 |
| bhujangasana | 64 | 70 | 62.0 | 61.9 | 0.492 |

**Honest reading.** The library recognises poses far better than it judges form: AUC is about 0.6 for Tadasana and Cat-Cow and about 0.5 (no skill) for Cobra. Likely causes: this dataset is photographed from many angles and distances while the references come mostly from front-on web photos; the features are 2-D image-plane angles; each set shows one person. **Do not present the score as an instructor-grade assessment.** Hand-tuning (as done for Tree) plus labelled volunteer data (`collect.py`) is the route to fix it.

## 3. Hold thresholds (`tools/fitting/calibrate_thresholds.py`)

Each fitted pose starts its hold at the lowest score where at most 3% of other poses' held-out photos would also start it (floor 60). If that makes the hold unreachable for the real pose (< 70% pass) the threshold is lowered and the pose is flagged **permissive**. The held-out photos choose these numbers, so section 1 is mildly optimistic; section 2 is untouched by it.

| Pose | start / release | permissive |
|---|---|---|
| utkatasana | 60 / 45 | no |
| bhujangasana | 71 / 56 | **yes** |
| adho_mukha | 60 / 45 | no |
| virabhadrasana | 60 / 45 | no |
| marjaryasana | 60 / 45 | no |
| tadasana | 60 / 45 | no |
| trikonasana | 60 / 45 | no |
| balasana | 60 / 45 | no |
| sukhasana | 60 / 45 | no |
| vrikshasana (hand-tuned) | 78 / 62 | no |

**Permissive: bhujangasana.** Its fitted tolerances are so wide that other poses also pass it (about one in five of their photos), and a Tree demo video completed a Cobra hold at 100% in `app.py`. A hip-height measurement was added to tighten it and did not fix this; it needs hand-tuned, pose-specific checks like Tree's.

## 4. End to end per pose (`tools/evaluation/pose_check.py`)

Each pose's reference photograph goes through the live pipeline (detection, scoring, step guide, hold, cues, overlay) for 30 simulated seconds; annotated screens are written to `data/results/pose_check/`. Every pose is detected and completes a hold **except Child's pose**, whose close-up reference photo scores about 52: the folded body hides its own limbs and the landmark model misplaces the arms and torso. Child's pose is the weakest pose for a single camera.

## 5. Not validated

* Live webcam use end to end (one 2-minute Chair session, before the step guide existed).
* Whether corrections improve a person's form.
* Small photo sets: Sukhasana 37, Warrior II 41, Balasana 49.
