# Implementation note — base version, one asana

**Project:** AI Yoga Posture & Wellness Companion using Computer Vision & MediaPipe
**Deliverable:** working **real-time** base implementation of the full pipeline
for a single asana — **Vrikshasana (Tree Pose)**, with its reference geometry
fitted from a public dataset
**Prepared for:** project guide review

---

## 1. What this deliverable is, and what it is not

The methodology proposes three parallel tracks off one camera feed: an asana
trainer, background desk-posture monitoring, and contactless vitals (rPPG). This
deliverable builds **the asana-trainer track, completely**, for one asana.

That choice is deliberate. The trainer is the track that carries the project's
research gap — *"yoga apps show a pose but never correct you"* — so it is the
part that must be proved to work before the other two are worth building. The
desk-posture track reuses the same landmarks and the same comparison engine with
a different reference; the rPPG track is an independent signal path off the same
frames.

**Implemented and verified:** landmark extraction, jitter filtering, joint
geometry, per-joint comparison against the reference asana, alignment scoring,
hold timing, spoken corrective feedback with debouncing, personal calibration,
the live overlay, the practice log, and an offline evaluation harness.

**Not yet built:** the remaining asanas, desk mode, rPPG vitals, the LLM
phrasing layer (the hook is in place), and the packaged installer.

---

## 2. What happens to one frame

| # | Step | Code | Cost |
|---|---|---|---|
| 1 | Capture BGR frame, mirror it | `app.py` | camera-bound |
| 2 | BlazePose → 33 landmarks + visibility | `landmarks.PoseTracker.process` | 13.9 ms |
| 3 | One-Euro filter per landmark, per axis | `filters.LandmarkSmoother` | included above |
| 4 | Landmarks → 16 body measurements | `evaluator.compute_features` | 0.13 ms |
| 5 | Measurements → 9 per-joint verdicts + score, best-fitting variant | `evaluator.evaluate` | included above |
| 6 | Score + time → SETUP / HOLDING / COMPLETE | `state_machine.PoseStateMachine.update` | negligible |
| 7 | Worst fault → a cue, if the rules allow it | `feedback.CueEngine.update` | negligible |
| 8 | Cue → speech on a worker thread | `feedback.Speaker` | off the hot path |
| 9 | Skeleton + HUD drawn | `overlay` | ~2 ms |
| 10 | Derived numbers → SQLite (2 Hz) | `storage.SessionLog` | negligible |

**Measured**, 1000×667 frame, Intel Core i7-13xxx (16 logical cores), CPU only,
median of 50 frames after warm-up, excluding capture and drawing:

| model | landmark inference | geometry + scoring | total | ceiling |
|---|---|---|---|---|
| `pose_landmarker_lite` | 9.9 ms | 0.12 ms | 10.0 ms | 99.6 fps |
| `pose_landmarker_full` | 13.9 ms | 0.13 ms | 14.1 ms | 71.1 fps |

End to end — decode, filter, measure, score, draw the overlay and display the
window — `app.py` sustained **28.2 fps** over a 1000×667 clip on the same
machine, i.e. the loop keeps up with a 30 fps camera with headroom to spare.

The full model is the default: at 14 ms it is already ~2× faster than a 30 fps
webcam delivers frames, so the accuracy is free. **The scoring layer this
project actually contributes costs 0.13 ms** — the pipeline is bounded entirely
by the neural network, which matters when Stage 5 reports cue latency.

*(MediaPipe 1.0 removed the legacy `mp.solutions.pose` API that most tutorials
use. This implementation is built on the current Tasks API, `PoseLandmarker`,
which needs the `.task` model file that `setup.bat` downloads.)*

---

## 3. How Vrikshasana is defined

> The numbers in this section are **fitted from 195 tree-pose photographs**, not
> hand-picked. The literature search, the dataset, the fitting method, what the
> data overturned, and the held-out validation are in
> **[REFERENCES.md](REFERENCES.md)** — read that alongside this section.

An asana is a list of `Check`s (`yoga/asanas.py`). Each names one measurable
feature, its correct value, the band that still counts as correct, and the
deviation at which that check scores zero. Adding another asana means adding
another list — the engine does not change.

| check | measures | target | tolerance | one-sided? | weight |
|---|---|---|---|---|---|
| `spine_tilt` | mid-hip → mid-shoulder, off vertical | 0° | 8° | too much only | 1.4 |
| `hip_level` | hip line off horizontal | **10°** | 8° | too much only | 0.9 |
| `shoulder_level` | shoulder line off horizontal | 6° | 14° | too much only | 0.5 |
| `standing_knee` | hip–knee–ankle, standing leg | 176° | 8° | too bent only | 1.2 |
| `folded_knee` | hip–knee–ankle, lifted leg | **30°** | 17° | both | 0.9 |
| `folded_thigh_open` | lifted thigh, off straight-down | 56° | 19° | both | 1.0 |
| `foot_height_ratio` | where the foot sits on the standing leg | **0.80** | 0.10 | too low only | 1.2 |
| `arm_raise` — overhead form | shoulder → wrist, off vertical | 15° | 22° | too low only | 0.8 |
| `arm_raise` — hands-at-heart form | shoulder → wrist, off vertical | 134° | 36° | both | 0.6 |

Three of those targets are not what a first guess would give. `hip_level` is
10°, not 0 — the pelvis genuinely hikes over the standing leg, and a target of 0
would have corrected every correct practitioner. `folded_knee` is 30°, not the
45° first assumed. `foot_height_ratio` is 0.80, not 0.70. The elbow checks that
a first pass included were **deleted** because the data showed they do not
discriminate. All four decisions are evidenced in [REFERENCES.md](REFERENCES.md)
§4–5.

**Two forms are accepted.** Clustering the fitted arm angles split cleanly in
two — hands overhead, and hands pressed at the heart — with every subject
appearing in both clusters. Both are scored and the better match wins, so the
practitioner is corrected towards the form they are actually doing rather than
told to do the other one.

`foot_height_ratio` is scale-free by construction: it is the lifted ankle's
height expressed as a fraction of the standing leg's own length, so it means the
same for a tall practitioner and a short one, and it does not change when
someone stands closer to the camera.

**The standing leg is detected, not configured.** Whichever leg is straighter is
taken as the standing one, with the lower foot breaking a tie, so either side of
the pose is accepted without the user telling the app anything.

---

## 4. Five design decisions, and why

### 4.1 The score is weakest-link, not an average

A plain weighted mean over nine checks is far too forgiving: one badly wrong
joint moves the average a few points, and the app cheerfully awards a "hold" to
a pose the practitioner is doing wrong. That is exactly the failure the project
criticises in existing apps.

```
score = 0.60 × weighted_mean  +  0.40 × worst_single_joint
```

A low-weight check cannot drag the score all the way down — its shortfall is
scaled by its own weight — so a slightly bent elbow is a note, while a collapsed
spine is a refusal. Measured effect on the same synthetic pose set:

| pose | plain average | weakest-link | hold allowed? |
|---|---|---|---|
| correct | 100.0 % | 100.0 % | yes |
| 5° lean | 100.0 % | 100.0 % | yes |
| 9° lean | 99.1 % | 97.2 % | yes |
| spine slouched 22° | 87.5 % | **61.4 %** | **no** |
| foot at ankle height | 81.8 % | **49.1 %** | **no** |
| standing leg bent 40° | 87.9 % | **57.8 %** | **no** |
| lifted knee not opened out | 94.1 % | **76.0 %** | **no** |
| hips hiked 24° | 97.4 % | 89.4 % | yes |
| arms out sideways | 97.5 % | 94.4 % | yes |

Under the plain average **every** one of the first four wrong poses would have
cleared the 78 % threshold and been counted as a successful hold. The last two
rows are deliberate: hip hike and arm position are graded, cued and scored, but
they do not by themselves refuse a Vrikshasana — the balance and the standing
leg are what make the pose.

### 4.2 The hold timer has hysteresis and a grace window

Vrikshasana wobbles — that is the whole point of the pose. A timer that resets
on the first dip is useless. The machine needs 78 % sustained for 0.6 s to
*start*, but only drops below 62 % for a continuous 1.2 s to *stop*. A 0.7 s
wobble is ridden out with the clock still running; a genuine loss of balance
ends the attempt. Both behaviours are asserted in the self-test.

### 4.3 Landmarks are One-Euro filtered, not averaged

BlazePose landmarks jitter by a few pixels even on a motionless body, which
becomes 3–5° of angle flicker and makes the voice cues chatter. A plain moving
average would fix that but would also delay a real correction. The One-Euro
filter smooths heavily when the body is still and barely at all when it moves
fast, so stillness is quiet and a genuine correction is still immediate.

### 4.4 Cues are governed by persistence, cooldown and escalation

A fault must persist **1 s** before it is spoken, each cue has a **7 s**
cooldown, and there is a **3 s** floor between any two cues. Repeating the same
fault escalates the wording from soft to firm. The result on a scripted session:

```
[   6.0s]  soft   Place the foot higher, onto the inner thigh
[   9.0s]  soft   Lift your chest and stack your spine over your hips
[  13.7s]  info   Good. Hold it.
[  33.7s]  praise Well held. Release the pose slowly.
[  52.1s]  soft   Straighten your standing leg
[  86.1s]  firm   Again - place the foot higher, onto the inner thigh
```

Only the **single worst** fault is ever spoken, never a list — a practitioner
with their eyes closed can act on one instruction, not four.

### 4.5 Grade what is visible; refuse to invent the rest

The first version of this refused a frame unless *every* core landmark was at
least 0.55 visible — and it turned out to throw away **174 of 200 correct tree
poses**. In Vrikshasana the lifted foot is pressed into the standing thigh, so
the most characteristic joint in the asana is occluded by definition: its
measured median visibility across 120 photographs is 0.50. The gate was
refusing the pose for being the pose. (See [REFERENCES.md](REFERENCES.md) §4.1.)

It now grades **per check**. Each check declares the body parts it needs; a
check whose parts are too faint is reported *ungraded* rather than failed, and
the frame is still scored if enough weight remains. Landmarks that have drifted
outside the image are scored zero whatever the network claims, because BlazePose
will happily extrapolate a leg past the bottom edge of the frame. Only when the
torso itself is unclear, or too little of the body is gradable, is the frame
refused outright — with a message saying which. Usable frames went from
**26/200 to 195/200**.

Reporting a confident number computed from landmarks the network could not
actually see would be the worst failure mode for a system whose whole purpose is
correcting people — but so is refusing to help someone who is doing it right.

---

## 5. Personal calibration

`run.bat --calibrate` records ~8 s of the practitioner's own best attempt and
writes `profiles/<user>.json`.

* **Flexibility-dependent checks** (foot height, thigh opening, elbows) have
  their target moved half-way towards what the person can actually do.
* **Alignment checks** (spine upright, hips level) keep the ideal target —
  being crooked is wrong for everybody — and only their tolerance widens.
* Tolerance grows with the practitioner's own measured wobble, capped at 1.8×
  the default so calibration can never render a check meaningless.

---

## 6. Verification

`test.bat` runs everything below with **no camera**. All **63 checks pass**.

| group | what is asserted |
|---|---|
| Geometry | Skeletons built with a known spine lean / hip tilt / thigh opening / foot height are measured back to within **0.6°** and **0.02** of the value they were built with — on both sides of the body. |
| Standing-leg detection | The correct leg is identified for a left-standing and a right-standing pose. |
| Fault attribution | Six deliberately wrong poses each yield the **correct primary fault**, and each scores below the correct pose. |
| Score behaviour | A 5° lean is inside tolerance; a 9° lean is flagged but still scores 94.9 %; a badly wrong joint denies the hold. |
| Hold timer | A poor pose never starts the timer; one good frame is not enough; a 0.7 s wobble does not end a hold; a 2 s loss does; a full hold completes at 10.0 s against a 10 s target; the abandoned attempt is still logged. |
| Cue rules | Silent for the first second; cooldown respected (cues at 1.0, 8.0, 15.0, 22.0, 29.0, 36.0 s); escalates soft → firm; **a correct pose produces no cue at all** over 20 s. |
| Variants | Both taught arm forms score 100 % and are each identified by name; arms out sideways is flagged but does not deny the hold; perfectly level hips are never called a fault. |
| Visibility | A half-occluded lifted foot is still graded and still scores 100 %; an invisible lifted leg ungrades only its own three checks and the rest of the body is still scored; both legs gone refuses the frame and says why; a hidden torso says "step back"; landmarks pushed outside the image are not graded. |

**Beyond the unit level:**

* `tools/session/simulate_session.py` drives a scripted 90-second practice — sloppy
  start, correction, full hold, failed second attempt, clean second hold —
  through the real evaluator, state machine, cue engine and SQLite logger, and
  asserts that exactly two holds complete.
* `tools/session/report.py` then reads that log back and reports the recurring
  weaknesses and the most frequent corrections.
* **Held-out dataset validation** (`tools/evaluation/validate_dataset.py`): the reference
  was fitted on 195 `train/tree` images, then every image of the untouched
  `test` split of five asanas was scored against Vrikshasana. Tree averaged
  **90.1 %** with 94 % over the hold threshold; chair, cobra, dog and warrior
  averaged 30–41 % with 0–1 % over it. **Precision 0.989, recall 0.937,
  F1 0.962, AUC 0.998** — 89 of 95 tree images accepted, 1 of 290 others. Full
  table and caveats in [REFERENCES.md](REFERENCES.md) §6–7.
* `data/samples/example_not_vrikshasana.png` is the system run on a photograph
  of a real person in **Warrior II**. It scores **38.7 %** against Vrikshasana
  and flags foot placement and both arms — the correct answer, on real imagery
  it has never seen.

---

## 7. Privacy and ethics, as built

* **No frame is written to disk.** Frames are decoded, measured and discarded.
  The only images that exist are ones the user explicitly saves with `s`.
* The practice log holds derived numbers only — joint angles, deviations,
  scores, hold durations, cue text — in a local SQLite file. It never leaves
  the machine.
* Every stage runs locally; the application makes no network call at run time,
  and nothing but the one-time model download at setup.
* The camera window is visible for the whole session, so the camera is never on
  without the user seeing it.
* Wellness and education only. The alignment score is a geometric agreement
  measure against a reference asana, not a clinical assessment, and the app does
  not replace a teacher.

---

## 8. What comes next, in order

1. **More asanas** — Tadasana, Virabhadrasana II, Trikonasana. Each is one
   `Asana(...)` entry and one run of `tools/fitting/fit_reference.py` against that
   class of the same dataset; the engine is unchanged. This is the cheapest and
   most visible next step, and the dataset already on disk contains warrior,
   chair, dog and cobra.
2. **Instructor ground truth** — the validation in §6 proves the system tells
   Vrikshasana apart from other asanas; it does **not** yet prove it can tell a
   *good* Vrikshasana from a *bad* one, because nobody graded the dataset's
   quality. Record practitioners, have an instructor label every frame
   correct/incorrect, run `tools/session/analyse.py --csv`, and report accuracy,
   precision, recall and the per-joint angle error. The harness is written; the
   labelled footage is what is missing.
3. **Desk-posture mode** — same landmarks, a seated reference, running in the
   background.
4. **rPPG vitals** — facial-ROI RGB means, band-pass 0.7–4 Hz, POS/CHROM + FFT.
   Independent of everything above and can be developed in parallel.
5. **LLM phrasing** — pass the measured deviations to a model and speak its
   sentence instead of the canned one. `CueEngine(phrase_fn=...)` already takes
   this callable and falls back to the canned string if the call fails, so the
   demo can never be broken by a network problem.
6. **Packaging** — Streamlit or browser front end, installer, user manual.
