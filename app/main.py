"""
Campus Fleet face service.

POST /v1/template   body: raw image bytes (image/jpeg or image/png), max 4 MB
                    query: pad=true|false (default true)
                    → { model_version, faces_detected, accepted, rejection_reasons,
                        template[128] | null, quality{...}, pad{...}, bbox }
GET  /healthz       → { ok, model_version }   (unauthenticated, no data)

The service is stateless: it never stores images or templates.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import auth
from .pipeline import MODEL_VERSION, FacePipeline, decode_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from verify_models import verify as verify_models  # noqa: E402

MODELS_DIR = os.getenv("FACE_MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
MAX_BYTES = 4 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("face-service")

_errors = verify_models(MODELS_DIR, required_only=True)
if _errors:
    raise RuntimeError(f"Model verification failed: {_errors}")

pipeline = FacePipeline(MODELS_DIR)
app = FastAPI(title="Campus Fleet Face Service", version=MODEL_VERSION, docs_url=None, redoc_url=None)


@app.get("/healthz")
def healthz():
    return {"ok": True, "model_version": MODEL_VERSION}


@app.post("/v1/template")
async def template(request: Request):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    body = await request.body()
    if len(body) > MAX_BYTES:
        raise HTTPException(413, "Image too large")

    err = auth.verify(
        auth.get_key(),
        request.headers.get("x-cf-timestamp"),
        request.headers.get("x-cf-signature"),
        request.method,
        request.url.path,
        body,
    )
    if err:
        log.warning("auth rejected request_id=%s reason=%s", request_id, err)
        raise HTTPException(401, "Unauthorized")

    try:
        img = decode_image(body)
    except ValueError as e:
        raise HTTPException(400, str(e))

    run_pad = request.query_params.get("pad", "true").lower() != "false"
    t0 = time.perf_counter()
    result = pipeline.template(img, run_pad=run_pad)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info(
        "template request_id=%s faces=%d accepted=%s reasons=%s ms=%s",
        request_id, result.faces_detected, result.accepted, result.rejection_reasons, ms,
    )

    q, p = result.quality, result.pad
    return JSONResponse(
        {
            "model_version": result.model_version,
            "faces_detected": result.faces_detected,
            "accepted": result.accepted,
            "rejection_reasons": result.rejection_reasons,
            "template": result.template,
            "quality": None if q is None else {
                "score": q.score, "face_px": q.face_px, "sharpness": q.sharpness,
                "yaw_ratio": q.yaw_ratio, "det_score": q.det_score, "passed": q.passed,
            },
            "pad": None if p is None else {"real_score": p.real_score, "passed": p.passed},
            "bbox": result.bbox,
            "elapsed_ms": ms,
        },
        headers={"x-request-id": request_id, "cache-control": "no-store"},
    )
