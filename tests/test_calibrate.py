import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from calibrate import rates, threshold_at_fmr  # noqa: E402


def test_threshold_respects_target_fmr():
    rng = np.random.default_rng(0)
    impostor = rng.uniform(0.4, 1.2, 100_000)
    t = threshold_at_fmr(impostor, 1e-3)
    _, fmr = rates(np.array([0.1]), impostor, t)
    assert fmr <= 1e-3


def test_fnmr_reported_at_threshold():
    genuine = np.array([0.1, 0.2, 0.3, 0.7])
    impostor = np.linspace(0.5, 1.5, 10_000)
    t = threshold_at_fmr(impostor, 1e-4)
    fnmr, _ = rates(genuine, impostor, t)
    assert t < 0.5 + 1e-3 and fnmr == 0.25  # only the 0.7 genuine pair is rejected
