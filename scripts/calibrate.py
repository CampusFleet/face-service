"""
Threshold calibration (Task 11): FNMR @ FMR on a consented calibration set.

Layout (private storage, never committed):
    calibration/<subject_id>/<session_id>/<image>.jpg      ≥ 2 sessions per subject

Genuine pairs  = same subject, different sessions (different days/lighting, like real boarding).
Impostor pairs = different subjects (all cross pairs).

Usage:
    python scripts/calibrate.py /secure/calibration --fmr 1e-4 --out report.json [--sql]

Prints the cosine-distance threshold achieving the target false-match rate and the resulting
false-non-match rate; --sql prints the statement that stores it in face_match_config.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.pipeline import MODEL_VERSION, FacePipeline  # noqa: E402


def threshold_at_fmr(impostor: np.ndarray, fmr: float) -> float:
    """Largest distance threshold t such that P(impostor distance <= t) <= fmr."""
    imp = np.sort(impostor)
    k = int(np.floor(fmr * len(imp)))  # number of impostor pairs allowed to match
    if k <= 0:
        return float(imp[0]) - 1e-6
    return float((imp[k - 1] + imp[k]) / 2) if k < len(imp) else float(imp[-1])


def rates(genuine: np.ndarray, impostor: np.ndarray, t: float) -> tuple[float, float]:
    fnmr = float(np.mean(genuine > t)) if len(genuine) else float("nan")
    fmr = float(np.mean(impostor <= t)) if len(impostor) else float("nan")
    return fnmr, fmr


def load_templates(root: str, pipe: FacePipeline):
    data: dict[str, dict[str, list[np.ndarray]]] = {}
    rejected = 0
    for subject in sorted(os.listdir(root)):
        sp = os.path.join(root, subject)
        if not os.path.isdir(sp):
            continue
        for session in sorted(os.listdir(sp)):
            sess = os.path.join(sp, session)
            if not os.path.isdir(sess):
                continue
            for f in sorted(os.listdir(sess)):
                img = cv2.imread(os.path.join(sess, f))
                if img is None:
                    continue
                r = pipe.template(img)
                if not r.accepted:
                    rejected += 1
                    continue
                data.setdefault(subject, {}).setdefault(session, []).append(np.asarray(r.template))
    return data, rejected


def pairs(data):
    genuine, impostor = [], []
    flat = [(s, sess, t) for s, ss in data.items() for sess, ts in ss.items() for t in ts]
    for (s1, se1, t1), (s2, se2, t2) in itertools.combinations(flat, 2):
        d = 1.0 - float(np.dot(t1, t2))
        if s1 == s2:
            if se1 != se2:
                genuine.append(d)
        else:
            impostor.append(d)
    return np.asarray(genuine), np.asarray(impostor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fmr", type=float, default=1e-4)
    ap.add_argument("--out", default="calibration-report.json")
    ap.add_argument("--sql", action="store_true")
    a = ap.parse_args()

    pipe = FacePipeline(os.path.join(os.path.dirname(__file__), "..", "models"))
    data, rejected = load_templates(a.root, pipe)
    genuine, impostor = pairs(data)
    if len(impostor) < 1 / a.fmr:
        print(f"WARNING: {len(impostor)} impostor pairs cannot resolve FMR={a.fmr}; need ≥ {int(1 / a.fmr)}. Add subjects.", file=sys.stderr)
    t = threshold_at_fmr(impostor, a.fmr)
    fnmr, fmr = rates(genuine, impostor, t)
    report = {
        "model_version": MODEL_VERSION,
        "subjects": len(data),
        "templates": sum(len(v) for s in data.values() for v in s.values()),
        "rejected_images": rejected,
        "genuine_pairs": int(len(genuine)),
        "impostor_pairs": int(len(impostor)),
        "fmr_target": a.fmr,
        "threshold": round(t, 4),
        "fmr_measured": fmr,
        "fnmr_measured": fnmr,
        "operating_points": {str(f): dict(zip(("threshold", "fnmr"), (round(threshold_at_fmr(impostor, f), 4), rates(genuine, impostor, threshold_at_fmr(impostor, f))[0]))) for f in (1e-2, 1e-3, 1e-4)},
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if a.sql:
        print(
            "\nINSERT INTO face_match_config (model_version, threshold, fmr_target, fnmr_measured, calibration_pairs, calibrated_at, notes, updated_at) "
            f"VALUES ('{MODEL_VERSION}', {report['threshold']}, {a.fmr}, {fnmr:.6f}, {len(genuine) + len(impostor)}, now(), "
            f"'{report['subjects']} subjects', now()) ON CONFLICT (model_version) DO UPDATE SET threshold = EXCLUDED.threshold, "
            "fmr_target = EXCLUDED.fmr_target, fnmr_measured = EXCLUDED.fnmr_measured, calibration_pairs = EXCLUDED.calibration_pairs, "
            "calibrated_at = EXCLUDED.calibrated_at, notes = EXCLUDED.notes, updated_at = now();"
        )


if __name__ == "__main__":
    main()
