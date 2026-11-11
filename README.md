# CampusFleet face service

Stateless Python service that turns one face image into a 128-d template plus quality and
anti-spoofing scores. It never stores images or templates. Model sources, licences and checksums: [`MODELS.md`](MODELS.md).

Part of CampusFleet Face ID boarding:

| Repo | Role |
|---|---|
| [CampusFleet/platform](https://github.com/CampusFleet/platform) | Web app and gate APIs. Calls this service, stores templates, matches and decides |
| **CampusFleet/face-service** (this repo) | Face detection, quality checks, anti-spoofing and templates |
| [CampusFleet/face-gate-agent](https://github.com/CampusFleet/face-gate-agent) | Raspberry Pi agent on the bus. Crops faces at the doors and uploads them to `platform` |

## API

`POST /v1/template`: the body is the raw JPEG/PNG (≤ 4 MB). Requests are signed with the shared service key:

```
X-CF-Timestamp: <unix seconds>
X-CF-Signature: hex(HMAC_SHA256(FACE_SERVICE_KEY, "<ts>\nPOST\n/v1/template\n<sha256_hex(body)>"))
```

The client is `src/lib/face/client.ts` in `platform`. If you change this format, change both repos.

Response:

```json
{ "model_version": "yunet2023mar+sface2021dec+minifasnet-v2v1se/1",
  "faces_detected": 1, "accepted": true, "rejection_reasons": [],
  "template": [128 floats, L2-normalised],
  "quality": { "score": 0.93, "face_px": 222, "sharpness": 3226.3, "yaw_ratio": 0.005, "det_score": 0.95, "passed": true },
  "pad": { "real_score": 1.0, "passed": true }, "bbox": [x, y, w, h], "elapsed_ms": 61.2 }
```

`rejection_reasons`: `NO_FACE`, `FACE_TOO_SMALL`, `TOO_BLURRY`, `FACE_NOT_FRONTAL`,
`LOW_DETECTION_CONFIDENCE`, `PRESENTATION_ATTACK_SUSPECTED`.

`GET /healthz`: liveness only, returns no data.

## Layout

```
app/        FastAPI app (main.py), request signing (auth.py), face pipeline (pipeline.py)
models/     ONNX runtime models + MiniFASNet .pth sources, MANIFEST.sha256
scripts/    verify_models.py, sign_and_post.py (manual test), calibrate.py (threshold calibration)
convert/    one-off MiniFASNet .pth → ONNX conversion (needs torch; not used at runtime)
tests/      pytest suite + golden images
```

## Local development

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest httpx
python scripts/verify_models.py && python -m pytest -q
FACE_SERVICE_KEY=$(openssl rand -hex 32) uvicorn app.main:app --port 8080
FACE_SERVICE_KEY=... python scripts/sign_and_post.py tests/golden/obama_a.jpg
```

CI (`.github/workflows/ci.yml`) runs the model checksum check and the tests on every push and pull request.

## Threshold calibration

Collect consented volunteer images in private storage (never commit them) as
`<dir>/<subject_id>/<session_id>/<image>.jpg`, with at least 30 subjects and 2+ sessions each on different days. Then run:

```bash
python scripts/calibrate.py <dir> --fmr 1e-4 --out report.json --sql
```

This prints the cosine-distance threshold for the target false-match rate, the resulting false-non-match
rate, and the SQL that stores the threshold in `face_match_config` in the platform database.

## Deploy (Mumbai)

Any container host in an India region works, for example Fly.io (`bom`). The Dockerfile verifies model
checksums at build time.

1. Build from the repo root.
2. Set `FACE_SERVICE_KEY` (at least 32 chars, generate with `openssl rand -hex 32`).
3. Size: 2 vCPU / 2 GB RAM, health check `GET /healthz`, HTTPS only.
4. In the platform's Vercel project set `FACE_SERVICE_URL=https://<host>` and the same `FACE_SERVICE_KEY`.
