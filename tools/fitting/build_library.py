"""Fit the whole pose library from its recipe - one reproducible command.

For every pose except Vrikshasana (hand-tuned in yoga/asanas.py) this says WHERE
its photographs come from and which measurements define it (yoga/posespecs.py),
fits the targets and tolerances from the data (median, 1.5 x robust sigma), keeps
25 % of each new source aside as a hold-out that is never fitted on, and writes
data/asana_fits.json with the provenance of every pose.

    python tools/datasets/get_yoga107.py            # once: the 107-pose photos (~1.1 GB)
    python tools/fitting/build_library.py          # fit all nine fitted poses
    python tools/evaluation/library_eval.py           # score the hold-outs against the library

Sources
  TF   data/datasets/yoga_poses/train/<class>     (its own test/ split is the hold-out)
  Y    data/datasets/yoga107/<slug>               (Hugging Face rotemvahava/yoga-poses-107)
  C    data/datasets/commons/<pose>               (Wikimedia Commons, per-file CC licences)
The "Yoga for all" photos (data/datasets/yoga_for_all) are deliberately NOT fitted
on: their right/wrong labels make them the external check in tools/evaluation/library_eval.py.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from tools.fitting import fit_asana as fa                         # noqa: E402
from yoga import posespecs                                         # noqa: E402
from yoga.landmarks import PoseTracker                             # noqa: E402

D = os.path.join(HERE, "data", "datasets")
TF = lambda c: ("TF", os.path.join(D, "yoga_poses", "train", c))   # noqa: E731
Y = lambda s: ("Y", os.path.join(D, "yoga107", s))                 # noqa: E731
C = lambda p: ("C", os.path.join(D, "commons", p))                 # noqa: E731

#: key -> display names, level, and where the photographs come from.
#: "variants" are accepted alternative forms: the person is scored against
#: whichever fits best, exactly like Vrikshasana's two arm forms.
RECIPE = {
    "tadasana": dict(sanskrit="Tadasana", name="Mountain Pose", level=1,
                     variants={"arms down": [Y("tadasana")], "arms up": [Y("urdhva_hastasana")]}),
    "trikonasana": dict(sanskrit="Trikonasana", name="Triangle Pose", level=2,
                        sources=[Y("utthita_trikonasana"), C("trikonasana")]),
    "virabhadrasana": dict(sanskrit="Virabhadrasana II", name="Warrior II", level=2,
                           sources=[Y("virabhadrasana_ii")]),
    "utkatasana": dict(sanskrit="Utkatasana", name="Chair Pose", level=2,
                       sources=[TF("chair"), Y("utkatasana")]),
    "adho_mukha": dict(sanskrit="Adho Mukha Svanasana", name="Downward Dog", level=1,
                       sources=[TF("dog"), Y("adho_mukha_svanasana")]),
    "bhujangasana": dict(sanskrit="Bhujangasana", name="Cobra Pose", level=1,
                         sources=[TF("cobra"), Y("bhujangasana")]),
    "marjaryasana": dict(sanskrit="Marjaryasana-Bitilasana", name="Cat-Cow", level=1,
                         variants={"cat": [Y("marjaryasana")], "cow": [Y("bitilasana")]}),
    # Child's pose is taught with the arms stretched forward OR resting alongside the
    # body.  One median would sit between the two and match neither, so the photos
    # are split by their measured arm angle (hip-shoulder-wrist) and each form fitted.
    "balasana": dict(sanskrit="Balasana", name="Child's Pose", level=1,
                     sources=[Y("balasana")],
                     split=("arm_torso_vis", 100.0, ("arms alongside", "arms forward"))),
    "sukhasana": dict(sanskrit="Sukhasana", name="Easy Pose", level=1,
                      sources=[Y("sukhasana")]),
}
HOLDOUT = 0.25


def gather(sources: list) -> tuple[list[str], list[str], list[str]]:
    """(train files, held-out files, provenance strings) over several sources."""
    train, held, prov = [], [], []
    for kind, folder in sources:
        files = fa.list_images(folder)
        if kind == "TF":                          # TF has its own test/ split
            tr, ho = files, []
        else:
            tr, ho = fa.split_holdout(files, HOLDOUT)
        train += tr
        held += ho
        prov.append(f"{kind}:{os.path.basename(folder)} ({len(tr)} fit, {len(ho)} held out)")
    return train, held, prov


def main() -> int:
    tracker = PoseTracker(model="full", running_mode="image", smooth=False)
    fits: dict = {}
    print(f"\n{'pose':16s}{'fit':>5s}{'held':>6s}{'checks':>8s}  sources")
    print("-" * 78)
    for key, rec in RECIPE.items():
        weights = posespecs.FEATURES[key]
        groups = rec.get("variants") or {"": rec["sources"]}
        split = rec.get("split")
        all_train, all_held, prov = [], [], []
        per_group = {}
        for gname, srcs in groups.items():
            tr, ho, pv = gather(srcs)
            per_group[gname] = tr
            all_train += tr
            all_held += ho
            prov += pv
        if not all_train:
            print(f"{key:16s}  no images found - run tools/datasets/get_yoga107.py first")
            continue
        if split:
            groups = {n: [] for n in split[2]}
            per_group = {n: [] for n in split[2]}
            from yoga.evaluator import compute_features
            import cv2
            for f in all_train:
                img = cv2.imread(f)
                pz = tracker.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img is not None else None
                v = compute_features(pz).get(split[0]) if pz is not None else None
                if v is not None and v == v:
                    per_group[split[2][0] if v < split[1] else split[2][1]].append(f)
            rec = {**rec, "variants": groups}
        base, used, skipped = fa.fit_class(tracker, "", all_train, 25, weights)
        spec = {"key": key, "sanskrit": rec["sanskrit"], "name": rec["name"],
                "level": rec["level"], "source_class": key, "n_images": used, "checks": base,
                "sources": prov, "holdout": [os.path.relpath(f, os.path.join(HERE, "data"))
                                             for f in all_held]}
        if rec.get("variants"):
            spec["variants"] = {}
            for gname, files in per_group.items():
                vchecks, vused, _ = fa.fit_class(tracker, "", files, 20, weights)
                if vchecks:
                    spec["variants"][gname] = {"checks": vchecks, "n_images": vused}
        fits[key] = spec
        print(f"{key:16s}{used:>5d}{len(all_held):>6d}{len(base):>8d}  {'; '.join(prov)}")
    tracker.close()

    out = fa.OUT
    existing = json.load(open(out, encoding="utf-8")) if os.path.isfile(out) else {}
    existing.update(fits)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
    print(f"\n{len(fits)} poses fitted -> {os.path.relpath(out, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
