# Where the reference angles come from

**Question:** what are the *correct* joint angles for Vrikshasana, and who says so?

This document records the literature search, the dataset chosen, the method used
to derive the numbers in `yoga/asanas.py`, and the held-out result. Every number
in the asana definition can be traced back to this page and re-derived with
`tools/fit_reference.py`.

---

## 1. What the literature actually provides

The honest headline: **no standard table of ideal joint angles per asana exists
in the published literature.** Studies that do angle-based yoga correction each
derive their own reference from data. That is the finding, and it decided this
project's approach.

| Source | What it gives | What it does **not** give |
|---|---|---|
| Verma, Kumawat, Nakashima & Raman, *Yoga-82* (CVPRW 2020) — 28.4k images, 82 classes, 3-level hierarchy | The standard fine-grained **classification** benchmark for yoga poses | No joint-angle ground truth, no per-pose reference geometry. Distributed as image URLs behind a request form. |
| Thoutam et al., *Yoga Pose Estimation and Feedback Generation Using Deep Learning* (Comput. Intell. Neurosci. 2022) — the closest work to this project | A **12-angle feature set** (each limb segment measured against the x-axis), six asanas including Tree, feedback as the signed difference from a reference, 99.58% classification with an MLP | **No numeric table** of reference angles, and **no stated tolerance in degrees**. The reference is described only as the average over correct performances. |
| Biomechanics literature on Vrikshasana (e.g. *An Insight into the Biomechanics … of Vrikshāsanā*; *The Physical Demands of the Tree Pose … in Seniors*, PMC3437689) | Qualitative kinematics: in Tree Pose the lifted **hip is abducted, flexed and externally rotated**, the **knee flexed**, the **ankle dorsiflexed**; the spine stays neutral | No camera-frame angle targets usable by a single-webcam system |
| Bazarevsky et al., *BlazePose* (2020) | The 33-landmark topology and the on-device model this project runs | Nothing pose-specific |

**Consequence.** Following Thoutam et al.'s method but fixing its gap, this
project derives the reference from a dataset **and writes the numbers down**,
with tolerances, so they can be cited, criticised and reproduced.

---

## 2. Dataset

**TensorFlow / Laurence Moroney yoga-pose dataset** — 5 classes
(`chair`, `cobra`, `dog`, `tree`, `warrior`), 1,000 training and 442 test
images, direct download, no form or credentials:

```
https://storage.googleapis.com/download.tensorflow.org/data/pose_classification/yoga_poses.zip
```

Chosen over Yoga-82 because it (a) contains Tree Pose, (b) downloads directly
rather than through a request form and dead image URLs, and (c) ships four other
asanas that serve as a ready-made **negative class** for validation.

* Fitted on: `train/tree` — 200 images, **195 usable**.
* Validated on: `test/*` — 442 images, tree never seen during fitting.

**Limitation, stated plainly:** the tree class contains only **5 subjects**
(`girl1`, `girl2`, `girl3`, `guy1`, `guy2`), each photographed many times. So
these are 195 *images* but 5 *people*. The numbers are a sound starting
reference, not a population norm. Four of the five stand on the left leg, which
is why the fitted standing-leg split is 190/5 — a property of the dataset, not
of the detector (the two knee angles differ by a median of 145°, so the standing
leg is never ambiguous).

---

## 3. Method

`tools/fit_reference.py`:

1. Run BlazePose over every image of the correctly-performed asana.
2. Discard frames the visibility gate rejects.
3. Per feature: **target = median**, **spread = 1.4826 × MAD** (the consistent
   robust estimator of σ), **tolerance = 1.5 × spread**, bounded.
4. Median and MAD rather than mean and standard deviation because a
   web-collected pose dataset always contains side views, crops and the odd
   mislabelled image, and those would drag a mean badly.

Then the numbers were read rather than pasted in — which is where the three
findings below came from.

---

## 4. Three things the data changed

### 4.1 The visibility gate was rejecting 87% of correct poses

The first implementation refused to score a frame unless **every** core landmark
was at least 0.55 visible. Measured over 120 tree-pose photographs:

| landmark | median visibility | 10th pct |
|---|---|---|
| shoulders, hips | 1.00 | ≥0.99 |
| standing knee/ankle | 0.94–0.96 | 0.71–0.76 |
| **lifted ankle** | **0.50** | **0.28** |
| occluded elbow / wrist | 0.71–0.73 | 0.05–0.07 |

`presence` was ~1.00 everywhere — it is `visibility`, the occlusion score, that
drops. And of course it does: **in Vrikshasana the lifted foot is pressed into
the standing thigh, so the single most characteristic joint of the pose is
occluded by definition.** Only 3.3% of correct tree images cleared the old gate;
174 of 200 were thrown away.

Fixed by grading **per check**: each check declares the body parts it needs, a
check whose parts are too faint is reported *ungraded* rather than failing, and
the frame is scored if enough weight remains. Landmarks that have drifted
outside the image are scored zero regardless of what the network claims.

**Result: usable frames went from 26/200 to 195/200.** This was a real bug that
would have made the live app say "step back" at a correctly posed user.

### 4.2 The hips are *not* level in Tree Pose

Fitted hip-line tilt: **median 12.7°** over all views, and **still ~10°**
(p75 ≈ 12°, p90 ≈ 18°) when restricted to square-on views only. The pelvis
genuinely hikes over the standing leg — which is exactly the hip abduction the
biomechanics literature describes.

The obvious hand-set target of 0° ± 6° would therefore have **corrected every
correct practitioner**, several times a minute. The check now targets 10° and is
**one-sided**: more tilt than that is a fault, perfectly level hips never are.

This also forced a `frontality` gate (shoulder width ÷ torso length). "Is it
level" is measured in the image plane, so a practitioner turned 45° photographs
a level pelvis as a tilted one. Level checks are only graded when the body is
square enough to the camera; otherwise the HUD says *turn to face the camera*.

### 4.3 The arms are bimodal — there are two correct forms

1-D clustering of the fitted arm angles (n = 145 images where both arms were
measurable):

| feature | cluster A | cluster B |
|---|---|---|
| left arm raise | 20.6° (σ 17.9, n=66) | 125.0° (σ 28.9, n=79) |
| right arm raise | 13.7° (σ 12.1, n=71) | 142.9° (σ 23.2, n=74) |

**Every one of the five subjects appears in both clusters**, so this is two
accepted forms of the asana — hands overhead, and hands pressed together at the
heart — not two kinds of practitioner. A single arm target would have marked one
of the two standard forms wrong.

The asana therefore carries **variants**: both forms are scored and the better
match wins, so a practitioner is corrected towards the form they are actually
attempting.

The **elbow checks were dropped entirely.** Their clusters (≈48° and ≈158°) cut
*across* the arm forms rather than along them — palms can be pressed together
overhead as well as at the chest — so the elbow angle does not separate a
correct pose from an incorrect one. Better to measure nothing than to measure
something that does not discriminate.

---

## 5. The fitted reference

From 195 usable `train/tree` images:

| check | fitted median | robust σ | p10 | p90 | shipped target | shipped tol | mode |
|---|---|---|---|---|---|---|---|
| spine_tilt | 4.90° | 2.65 | 1.19 | 8.07 | **0°** (ideal kept) | 8° | max |
| hip_level | 12.73° | 8.37 | 5.36 | 33.35 | **10°** | 8° | max |
| shoulder_level | 5.16° | 4.40 | 1.15 | 26.09 | 6° | 14° | max |
| standing_knee | 176.06° | 3.64 | 168.27 | 179.31 | **176°** | 8° | min |
| folded_knee | 29.62° | 10.95 | 15.35 | 49.23 | **30°** | 17° | band |
| folded_thigh_open | 55.90° | 12.41 | 28.67 | 70.81 | **56°** | 19° | band |
| foot_height_ratio | 0.80 | 0.04 | 0.70 | 0.85 | **0.80** | 0.10 | min |
| arm_raise (overhead) | — | — | — | — | 15° | 22° | max |
| arm_raise (at heart) | — | — | — | — | 134° | 36° | band |

Two targets deliberately **do not** follow the median:

* `spine_tilt` is held at the ideal **0°** — a leaning spine is wrong for
  everybody, and the fitted median of 3.5° (frontal views) confirms the ideal is
  the right reference rather than contradicting it.
* `shoulder_level` keeps a low weight and a wide band because it is the one
  measurement that got *noisier* as views got more frontal (p90 rising 21° → 30°
  on a shrinking sample) — it is simply unreliable from one camera.

Everything else is the fitted value. For comparison, the first hand-picked guess
had `folded_knee` at 45° (data: 30°), `foot_height_ratio` at 0.70 (data: 0.80)
and `hip_level` at 0° (data: 10°).

---

## 6. Held-out validation

`tools/validate_dataset.py` on the **test** split, which was never used for
fitting. Every image of all five classes is scored against Vrikshasana; the four
non-tree classes are the negative set.

| class | n scored | skipped | mean | median | p10 | p90 | ≥ 78% threshold |
|---|---|---|---|---|---|---|---|
| chair | 84 | 0 | 29.8 | 26.1 | 23.8 | 40.0 | 0% |
| cobra | 84 | 32 | 32.8 | 30.3 | 26.4 | 41.5 | 1% |
| dog | 39 | 51 | 41.1 | 42.6 | 35.9 | 43.8 | 0% |
| **tree** | **95** | **1** | **90.1** | **90.5** | **81.3** | **97.1** | **94%** |
| warrior | 83 | 26 | 31.0 | 29.9 | 27.3 | 36.7 | 0% |

At the live hold threshold of 78%:

* **precision 0.989, recall 0.937, F1 0.962**
* accepted 89 of 95 tree images, and **1 of 290** images of other asanas
* **AUC 0.998** (threshold-free ranking)
* the best separating threshold would be 67% (Youden J = 0.993); 78% was chosen
  before this test and is left as is, since it errs towards not awarding a hold

**Read the `skipped` column honestly.** 32 cobra, 51 dog and 26 warrior images
could not be scored at all — they are lying-down and folded poses where the
torso or legs are not visible enough to grade. Those are reported as *"cannot
score"*, never as *"correct pose"*, which is the safe failure; but it does mean
the negative set effectively shrank for those two classes. Tree, by contrast,
was scorable in 95 of 96 images.

---

## 7. Limitations

1. **Five subjects.** The reference reflects 5 people, not a population.
2. **Studio photographs, not practice footage** — lighting and framing are
   kinder than a living room, and there is no motion, so nothing here validates
   the temporal behaviour (the hold timer and cue debouncing are verified
   separately, on synthetic sequences, in `tools/selftest.py`).
3. **No instructor labels.** The dataset's tree images are assumed correct
   because they are labelled `tree`; nobody graded their *quality*. The next
   step in the methodology — an instructor labelling frames correct/incorrect —
   is what would turn this into a true accuracy measurement rather than a
   pose-identity one. `tools/analyse.py --csv` already emits the per-frame rows
   that step needs.
4. **Single camera, 2-D.** Every angle is measured in the image plane. The
   frontality gate limits the damage but does not remove it; BlazePose's world
   landmarks (already carried on the `Pose` object) are the route to fixing it.
5. **Not a medical or certification instrument.** These are geometric agreement
   measures against a fitted reference, nothing more.

---

## 8. References

1. Verma, M., Kumawat, S., Nakashima, Y., Raman, S. (2020). *Yoga-82: A New Dataset for Fine-grained Classification of Human Poses.* CVPR Workshops. https://arxiv.org/abs/2004.10362 · https://sites.google.com/view/yoga-82/home
2. Thoutam, V. A. et al. (2022). *Yoga Pose Estimation and Feedback Generation Using Deep Learning.* Computational Intelligence and Neuroscience, 2022:4311350. https://pmc.ncbi.nlm.nih.gov/articles/PMC8970937/
3. Bazarevsky, V. et al. (2020). *BlazePose: On-device Real-time Body Pose Tracking.* arXiv:2006.10204. https://arxiv.org/abs/2006.10204
4. Casiez, G., Roussel, N., Vogel, D. (2012). *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.* CHI '12.
5. Moroney, L. (2021). *Yoga pose dataset.* https://laurencemoroney.com/2021/08/23/yogapose-dataset.html
6. *An Insight into the Biomechanics and Other Details of Vrikshāsanā, One of the Standing Yoga Āsanās.* https://www.researchgate.net/publication/366508144
7. *The Physical Demands of the Tree (Vriksasana) and One-Leg Balance Poses Performed by Seniors: A Biomechanical Examination.* https://pmc.ncbi.nlm.nih.gov/articles/PMC3437689/
8. MediaPipe Pose Landmarker (Tasks API) documentation. https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
