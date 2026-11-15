import os

import cv2
import pytest

from app.pipeline import TEMPLATE_DIM, FacePipeline, cosine_distance

HERE = os.path.dirname(__file__)
GOLDEN = os.path.join(HERE, "golden")
MODELS = os.path.join(HERE, "..", "models")

# Provisional decision threshold (cosine distance). Recalibrated in Task 11.
MATCH_THRESHOLD = 0.60


@pytest.fixture(scope="module")
def pipe():
    return FacePipeline(MODELS)


def img(name):
    im = cv2.imread(os.path.join(GOLDEN, name))
    assert im is not None, name
    return im


def test_template_shape_and_norm(pipe):
    r = pipe.template(img("obama_a.jpg"))
    assert r.accepted, r.rejection_reasons
    assert len(r.template) == TEMPLATE_DIM
    assert abs(sum(v * v for v in r.template) - 1.0) < 1e-3


def test_same_person_matches(pipe):
    a = pipe.template(img("obama_a.jpg")).template
    b = pipe.template(img("obama_b.jpg")).template
    assert cosine_distance(a, b) < MATCH_THRESHOLD


def test_different_person_does_not_match(pipe):
    a = pipe.template(img("obama_a.jpg")).template
    c = pipe.template(img("biden.jpg")).template
    assert cosine_distance(a, c) > MATCH_THRESHOLD


def test_blurred_face_rejected(pipe):
    blurred = cv2.GaussianBlur(img("obama_a.jpg"), (0, 0), 8)
    r = pipe.template(blurred)
    assert not r.accepted
    assert "TOO_BLURRY" in r.rejection_reasons


@pytest.mark.parametrize("name", ["silentface_image_F1.jpg", "silentface_image_F2.jpg"])
def test_photo_of_photo_spoof_rejected(pipe, name):
    r = pipe.template(img(name))
    assert not r.accepted
    assert "PRESENTATION_ATTACK_SUSPECTED" in r.rejection_reasons


def test_real_face_passes_pad(pipe):
    r = pipe.template(img("silentface_image_T1.jpg"))
    assert r.pad.passed and r.pad.real_score > 0.9


def test_no_face(pipe):
    import numpy as np
    r = pipe.template(np.full((480, 640, 3), 127, dtype=np.uint8))
    assert r.faces_detected == 0 and r.rejection_reasons == ["NO_FACE"] and r.template is None


def test_signed_yaw_flips_with_mirrored_image(pipe):
    """Mirroring the image swaps the turn direction; the unsigned ratio stays the same."""
    img = cv2.imread(os.path.join(GOLDEN, "obama_a.jpg"))
    a = pipe.template(img, run_pad=False).quality
    b = pipe.template(cv2.flip(img, 1), run_pad=False).quality
    assert abs(a.yaw_ratio - abs(a.yaw_signed)) < 1e-3
    if abs(a.yaw_signed) > 0.02:  # needs a visible turn to compare signs
        assert (a.yaw_signed > 0) != (b.yaw_signed > 0)
