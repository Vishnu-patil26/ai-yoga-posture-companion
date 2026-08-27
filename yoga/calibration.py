"""Personal tolerance (Methodology Stage 3, 'Calibration').

A stiff beginner and a trained practitioner cannot be judged against the same
numbers - the beginner would be corrected constantly and would stop using the
app.  So the reference geometry is fixed, but *how far* the practitioner may
sit from it is learned once from their own best attempt.

Flexibility-dependent checks (how high the foot goes, how far the knee opens)
have their target moved part-way towards what the person can actually do.
Alignment checks (spine upright, hips level) keep the ideal target - being
crooked is wrong for everybody - and only their tolerance is widened.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import replace

from .asanas import Asana

PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles"
)

#: Checks whose target may be moved towards the practitioner's own range.
FLEXIBILITY_KEYS = {"foot_height_ratio", "folded_thigh_open", "folded_knee",
                    "arm_raise_left", "arm_raise_right", "elbow_left", "elbow_right"}

#: How far the target is allowed to move towards the observed value (0..1).
BLEND = 0.5
#: Tolerance may grow by at most this factor.
MAX_TOL_GROWTH = 1.8


def profile_path(user: str) -> str:
    return os.path.join(PROFILE_DIR, f"{user}.json")


def build_profile(asana: Asana, samples: list[dict], user: str = "default") -> dict:
    """`samples` is a list of feature dicts captured during the calibration hold."""
    prof = {"user": user, "asana": asana.key, "n_samples": len(samples), "checks": {}}
    for check in asana.checks:
        vals = [s[check.key] for s in samples
                if isinstance(s.get(check.key), float) and s[check.key] == s[check.key]]
        if len(vals) < 5:
            continue
        med = statistics.median(vals)
        spread = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        target = check.target
        if check.key in FLEXIBILITY_KEYS:
            target = check.target + BLEND * (med - check.target)
        tol = min(check.tol * MAX_TOL_GROWTH, max(check.tol, check.tol * 0.6 + 1.5 * spread))
        prof["checks"][check.key] = {
            "target": round(target, 4), "tol": round(tol, 4),
            "observed_median": round(med, 4), "observed_sd": round(spread, 4),
        }
    return prof


def save_profile(prof: dict, user: str = "default") -> str:
    os.makedirs(PROFILE_DIR, exist_ok=True)
    path = profile_path(user)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(prof, fh, indent=2)
    return path


def load_profile(user: str = "default") -> dict | None:
    path = profile_path(user)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def apply_profile(asana: Asana, prof: dict | None) -> Asana:
    """Return a copy of the asana with personalised targets and tolerances."""
    if not prof or prof.get("asana") != asana.key:
        return asana
    tuned = []
    for check in asana.checks:
        p = prof.get("checks", {}).get(check.key)
        if not p:
            tuned.append(check)
            continue
        tol = float(p["tol"])
        # keep the score->0 point a sensible distance beyond the new tolerance
        zero_at = max(check.zero_at, tol + (check.zero_at - check.tol))
        tuned.append(replace(check, target=float(p["target"]), tol=tol, zero_at=zero_at))
    return replace(asana, checks=tuple(tuned))
