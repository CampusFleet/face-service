# Face ID model register

Status: **licences verified 2026-11-11** against the upstream sources below. Sign-off by the university's
designated reviewer is recorded at the bottom before the pilot starts.

| Purpose | Model | Version / file | Source | Licence | SHA-256 |
|---|---|---|---|---|---|
| Face detection + 5 landmarks | YuNet | `face_detection_yunet_2023mar.onnx` | [opencv_zoo/face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT ("All files in this directory are licensed under MIT License") | `8f2383e4…99bec0` |
| Face recognition (512-d) | AuraFace v1 (ResNet100, ArcFace loss) | `auraface_glintr100.onnx` (upstream `glintr100.onnx`) | [fal/AuraFace-v1](https://huggingface.co/fal/AuraFace-v1) | Apache-2.0 (trained on commercially licensed data) | `a7933ea5…5c60` |
| Passive anti-spoofing (PAD) | MiniFASNetV2, scale 2.7, 80×80 | `2.7_80x80_MiniFASNetV2.onnx` (converted from `.pth`) | [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | Apache-2.0 (repo `LICENSE`) | `1cb86b65…94419` |
| Passive anti-spoofing (PAD) | MiniFASNetV1SE, scale 4.0, 80×80 | `4_0_0_80x80_MiniFASNetV1SE.onnx` (converted from `.pth`) | same | Apache-2.0 | `0a4539e3…a978a9` |

Full checksums: [`models/MANIFEST.sha256`](models/MANIFEST.sha256).
Verify with `python scripts/verify_models.py` (runs in CI and during the Docker build).

## Conversion record (MiniFASNet)

- Upstream ships PyTorch weights only. `convert/convert_minifasnet.py` exports them to ONNX
  (opset 13) using the upstream Apache-2.0 model definition (`convert/MiniFASNet.py`, `convert/utility.py`,
  licence copied to `convert/LICENSE.minivision`).
- Parity check on 4 random 80×80 inputs: max |softmax(ONNX) − softmax(PyTorch)| = **6e-8** for both models.
- Pre-processing matches upstream: BGR, crop around face box × scale, resize to 80×80, **no** /255 or mean
  normalisation (upstream `to_tensor` returns raw float pixels). Class index 1 = real face.

## Replaced model (do not use)

| Model | Reason |
|---|---|
| InsightFace `w600k_mbf.onnx` (MobileFaceNet, WebFace600K) — previously in `public/models/` | InsightFace pretrained models are licensed for **non-commercial research only** ([model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)). Not permitted for university production use. Removed from use; the legacy `student_face_profiles` table in `platform` is no longer read. |

## Fallback backend

If calibration misses the accuracy target, the planned fallback is AWS Rekognition (`ap-south-1`).
It is **not implemented yet**. Using it would require a data-processing agreement approved by the
university before any student image is sent.

## Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Technical owner | Aditya Pandey | | |
| University reviewer (legal / IT) | | | |
