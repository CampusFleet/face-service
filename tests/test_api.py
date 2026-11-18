import os
import time

import pytest

os.environ.setdefault("FACE_SERVICE_KEY", "test-face-service-key-0123456789abcdef")

from fastapi.testclient import TestClient  # noqa: E402

from app import auth  # noqa: E402
from app.main import app  # noqa: E402

KEY = os.environ["FACE_SERVICE_KEY"].encode()
GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
client = TestClient(app)


def signed_post(body: bytes, ts: int | None = None, key: bytes = KEY, path="/v1/template"):
    ts = str(ts if ts is not None else int(time.time()))
    return client.post(
        path,
        content=body,
        headers={"content-type": "image/jpeg", "x-cf-timestamp": ts, "x-cf-signature": auth.sign(key, ts, "POST", path, body)},
    )


def jpg(name):
    with open(os.path.join(GOLDEN, name), "rb") as f:
        return f.read()


def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"]


def test_template_ok():
    r = signed_post(jpg("obama_a.jpg"))
    assert r.status_code == 200
    j = r.json()
    assert j["accepted"] and len(j["template"]) == 512 and j["quality"]["passed"] and j["pad"]["passed"]


def test_spoof_reported():
    j = signed_post(jpg("silentface_image_F2.jpg")).json()
    assert not j["accepted"] and "PRESENTATION_ATTACK_SUSPECTED" in j["rejection_reasons"]


@pytest.mark.parametrize("case", ["unsigned", "wrong_key", "stale", "tampered"])
def test_auth_rejections(case):
    body = jpg("obama_a.jpg")
    if case == "unsigned":
        r = client.post("/v1/template", content=body)
    elif case == "wrong_key":
        r = signed_post(body, key=b"x" * 40)
    elif case == "stale":
        r = signed_post(body, ts=int(time.time()) - 600)
    else:
        ts = str(int(time.time()))
        sig = auth.sign(KEY, ts, "POST", "/v1/template", body)
        r = client.post("/v1/template", content=body + b"x", headers={"x-cf-timestamp": ts, "x-cf-signature": sig})
    assert r.status_code == 401


def test_bad_image():
    r = signed_post(b"not an image")
    assert r.status_code == 400
