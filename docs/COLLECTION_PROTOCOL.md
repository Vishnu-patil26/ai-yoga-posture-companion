# Volunteer photo collection protocol

Version of the consent text this describes: `v1-2026-10` (`yoga/collection.py`).
This is a student-project protocol, not legal advice. Have the project guide
approve it, and the consent text, before the first volunteer is photographed.

## 1. Purpose

The angle references in `data/asana_fits.json` were fitted from public datasets that
show one or two people per pose, so each target really means "what that person's body
does". The team collects its own photos of **diverse, healthy volunteers** doing the 10
poses in `yoga/routines.py`, so the references can be refitted and checked on more
bodies. Meeting notes (viva 28 Oct 2026): *sample collection - ID create - data - collect photo
of diverse - copyright - model training for healthy person (QA).*

## 2. Who may take part

- Adults (18+) who are healthy: no current injury or condition that makes these poses
  unsafe. This is self-declared (second tick box) and the supervisor is the second check.
  Anyone unsure is thanked and not enrolled.
- Participation is voluntary. A volunteer may skip any pose, stop at any time, and
  withdraw later by ID. No payment, no pressure, and no marks or favours attached.

## 3. Recruitment target (a target, not a result)

Aim for, per pose:

- at least **10 different volunteers** with a right-labelled attempt;
- every age group (18-24 ... 65+) and every body type represented by at least 2
  volunteers, and neither female nor male above 60% of the volunteers;
- some **wrong** attempts per pose (decided with the supervisor), so the checker can be
  tested on real faults and not only on good form.

The Summary screen of `collect.py` (or `collection.summary().text()`) lists exactly where
the sample is still thin against these numbers. `tools/fit_asana.py` needs at least 25 usable images
per pose, and one attempt gives 5, so 10 volunteers give about 50.

## 4. Consent, copyright and privacy

1. The volunteer reads the consent screen (`CONSENT_TEXT`), ticks **both** boxes (agree;
   healthy) and picks a **face policy**: `keep`, `blur` (default; automatic Gaussian blur
   over the head) or `landmarks_only` (no photo is ever written).
2. They get a random ID `V-XXXXXX` and are told to note it. **No name is stored.** If the
   team wants a name-to-ID sheet for convenience, keep it on paper, apart from the data,
   and destroy it when the data is deleted.
3. Copyright: the volunteer keeps it and grants the team a free, non-exclusive licence to use
   the photos only to fit and test pose references for this project, and to report averages and
   charts. Photos are not published, sold or shared beyond the team, guide and examiners.
4. `data/collected/<ID>/consent.txt` keeps the exact text shown, its version, the time and
   the choices made. A volunteer who has not ticked both boxes cannot be registered, so
   nothing can be saved for them.
5. Retention: until the project is assessed and no longer than 12 months, then delete
   `data/collected/` and `data/collected_export/`.

## 5. Capture setup

- Camera at about **1 m** height, about **2 m** from the volunteer, full body in frame
  (head to feet, hands and feet included), plain background, even light, no strong backlight.
- Standing and sitting poses: face the camera. **Floor poses** (Downward Dog, Cobra, Cat-Cow,
  Child's Pose): mat side-on to the camera so the whole body is seen from the side.
- Plain, fitted clothing that shows the joints (no long coats or loose robes).
- In `collect.py`: choose the pose, mark the attempt **right** or **wrong** (add a short fault
  note such as "knees bent"), press Start. A 5-second countdown is followed by 5 photos
  1 second apart. The volunteer sees whether a body was found; photos with no body found are
  discarded, and **Retake last** deletes the previous attempt. Saved photos are the raw
  (un-mirrored) camera frames; only the preview is mirrored.

## 6. Labelling

"Right" or "wrong" is the supervisor's judgement together with the volunteer, using the pose's
steps shown on screen. A second person should confirm a sample of labels before a refit.
Record a wrong attempt only when the fault is visible or deliberate; free text in the fault
box stays with the sample.

## 7. Storage and withdrawal

- Layout: `data/collected/<ID>/volunteer.json`, `consent.txt`, and per pose
  `<pose_key>/<n>.jpg` (not for `landmarks_only`) plus `<n>.json` (33 landmarks, computed
  features, label). Nothing leaves the project computer.
- The folder `data/collected/` and the export folder each contain a `.gitignore` with `*`
  so they cannot be committed by accident. The project `.gitignore` should also list them.
- **Withdrawal:** in `collect.py` use *Withdraw a volunteer* (type the ID, confirm), or
  `collection.withdraw("V-XXXXXX", also_in=collection.EXPORT_DIR)`. The whole folder and the
  volunteer's exported copies are deleted. Without the ID the data cannot be found, so
  insist that it is noted. If a fit already used the volunteer's photos, export and refit
  again so the numbers no longer include them.

## 8. Refitting the references

1. `Export right-labelled photos for fitting` (Summary screen), or
   `collection.export_for_fitting("data/collected_export")`. This copies right-labelled
   photos to `<pose_key>/<ID>_<n>.jpg`, the folder-per-class layout of `tools/fit_asana.py`.
2. `python tools/fit_asana.py data/collected_export --out data/asana_fits_collected.json`.
   Always pass `--out`: the default output is `data/asana_fits.json`, and it would be replaced by a
   file holding only the classes found. Compare the new targets and tolerances with the
   current ones, and replace the old file only if the team agrees.
3. The display names in the output come from the folder names (e.g. `Adho_Mukha`) and can be
   edited. The hand-tuned Vrikshasana stays in the library either way.

## 9. Limitations

- Self-declared health, one camera and one room: the sample is diverse in people, not in
  setups. Five photos from one attempt are near-duplicates, so the effective sample size is
  the number of **volunteers**, not photos.
- Automatic face blur depends on BlazePose's head estimate. It is least reliable for
  head-down poses (Downward Dog, Child's Pose); spot-check blurred photos, or
  have the volunteer choose `landmarks_only`.
- `tools/fit_asana.py` re-detects the pose in the exported images, so blurred faces may shift
  head landmarks (the neck feature), and `landmarks_only` volunteers contribute nothing
  to an image-based refit (counted as "skipped" on export). Their stored landmarks and features
  could feed a landmark-based refit, which is not built yet.
- This code and its tests were checked with a scripted camera and public sample images, not a
  real webcam session with volunteers. Do a dry run with the team before recruiting.
