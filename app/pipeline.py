"""
Face pipeline: detect (YuNet) → quality gate → 5-point alignment + template (SFace) → passive PAD (MiniFASNet).

All models are licence-clean (MIT / Apache-2.0, see docs/biometrics/models.md) and run on CPU.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

MODEL_VERSION = "yunet2023mar+sface2021dec+minifasnet-v2v1se/1"
TEMPLATE_DIM = 128

# Quality thresholds (tuned conservatively; recalibrate in Task 11)
MIN_FACE_PX = int(os.getenv("FACE_MIN_SIZE_PX", "80"))          # shorter side of the face box
MIN_SHARPNESS = float(os.getenv("FACE_MIN_SHARPNESS", "40"))    # variance of Laplacian on the aligned crop
MAX_YAW_RATIO = float(os.getenv("FACE_MAX_YAW_RATIO", "0.35"))  # |nose offset| / eye distance
MIN_DET_SCORE = float(os.getenv("FACE_MIN_DET_SCORE", "0.80"))
PAD_REAL_THRESHOLD = float(os.getenv("FACE_PAD_REAL_THRESHOLD", "0.60"))

PAD_MODELS = [
    # (file, crop scale) — scale is encoded in the upstream file name
    ("2.7_80x80_MiniFASNetV2.onnx", 2.7),
    ("4_0_0_80x80_MiniFASNetV1SE.onnx", 4.0),
]


@dataclass
class Quality:
    face_px: int
    sharpness: float
    yaw_ratio: float
    det_score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    # Signed head turn, same measure as yaw_ratio. Positive = the subject turned to their own LEFT
    # (nose toward the right edge of the unmirrored image). Lets the platform check the
    # enrollment challenge side; yaw_ratio stays unsigned for compatibility.
    yaw_signed: float = 0.0

    @property
    def score(self) -> float:
        """0..1 summary used for ranking captures (not for pass/fail)."""
        size = min(1.0, self.face_px / (MIN_FACE_PX * 2))
        sharp = min(1.0, self.sharpness / (MIN_SHARPNESS * 4))
        pose = max(0.0, 1.0 - self.yaw_ratio / max(MAX_YAW_RATIO, 1e-6))
        return round(float(0.4 * size + 0.3 * sharp + 0.2 * pose + 0.1 * self.det_score), 4)


@dataclass
class PadResult:
    real_score: float
    passed: bool


@dataclass
class TemplateResult:
    faces_detected: int
    template: list[float] | None
    quality: Quality | None
    pad: PadResult | None
    bbox: list[int] | None
    model_version: str = MODEL_VERSION

    @property
    def accepted(self) -> bool:
        return bool(self.template and self.quality and self.quality.passed and self.pad and self.pad.passed)

    @property
    def rejection_reasons(self) -> list[str]:
        if self.faces_detected == 0:
            return ["NO_FACE"]
        reasons = list(self.quality.reasons) if self.quality else []
        if self.pad and not self.pad.passed:
            reasons.append("PRESENTATION_ATTACK_SUSPECTED")
        return reasons


class FacePipeline:
    def __init__(self, models_dir: str):
        self._lock = threading.Lock()  # cv2 DNN objects are not thread-safe
        self.detector = cv2.FaceDetectorYN.create(
            os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"), "", (320, 320),
            score_threshold=0.6, nms_threshold=0.3, top_k=50,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            os.path.join(models_dir, "face_recognition_sface_2021dec.onnx"), ""
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.pad_sessions = [
            (ort.InferenceSession(os.path.join(models_dir, f), opts, providers=["CPUExecutionProvider"]), scale)
            for f, scale in PAD_MODELS
        ]

    # ── detection ─────────────────────────────────────────────────────────
    def detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        with self._lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)
        return faces if faces is not None else np.zeros((0, 15), dtype=np.float32)

    # ── quality ───────────────────────────────────────────────────────────
    @staticmethod
    def _quality(face: np.ndarray, aligned: np.ndarray) -> Quality:
        x, y, fw, fh = face[:4]
        rx, ry, lx, ly, nx, ny = face[4:10]  # right eye, left eye, nose
        eye_dist = max(1e-6, float(np.hypot(lx - rx, ly - ry)))
        yaw_signed = (float(nx) - (float(rx) + float(lx)) / 2) / eye_dist
        yaw_ratio = abs(yaw_signed)
        sharp = float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        face_px = int(min(fw, fh))
        det = float(face[14])
        reasons = []
        if face_px < MIN_FACE_PX:
            reasons.append("FACE_TOO_SMALL")
        if sharp < MIN_SHARPNESS:
            reasons.append("TOO_BLURRY")
        if yaw_ratio > MAX_YAW_RATIO:
            reasons.append("FACE_NOT_FRONTAL")
        if det < MIN_DET_SCORE:
            reasons.append("LOW_DETECTION_CONFIDENCE")
        return Quality(face_px, round(sharp, 2), round(yaw_ratio, 4), round(det, 4), not reasons, reasons, round(yaw_signed, 4))

    # ── presentation attack detection ────────────────────────────────────
    @staticmethod
    def _pad_crop(img: np.ndarray, bbox: tuple[float, float, float, float], scale: float) -> np.ndarray:
        """Upstream CropImage._get_new_box: expand box by scale, keep inside the image, resize to 80x80."""
        src_h, src_w = img.shape[:2]
        x, y, bw, bh = bbox
        scale = min((src_h - 1) / bh, min((src_w - 1) / bw, scale))
        nw, nh = bw * scale, bh * scale
        cx, cy = bw / 2 + x, bh / 2 + y
        x1, y1, x2, y2 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
        if x1 < 0:
            x2 -= x1; x1 = 0
        if y1 < 0:
            y2 -= y1; y1 = 0
        if x2 > src_w - 1:
            x1 -= x2 - src_w + 1; x2 = src_w - 1
        if y2 > src_h - 1:
            y1 -= y2 - src_h + 1; y2 = src_h - 1
        crop = img[int(y1): int(y2) + 1, int(x1): int(x2) + 1]
        return cv2.resize(crop, (80, 80))

    def pad(self, img: np.ndarray, bbox: tuple[float, float, float, float]) -> PadResult:
        total = np.zeros(3, dtype=np.float64)
        for sess, scale in self.pad_sessions:
            crop = self._pad_crop(img, bbox, scale).astype(np.float32)  # raw BGR pixels, no /255 (as upstream)
            logits = sess.run(None, {"input": crop.transpose(2, 0, 1)[None]})[0][0]
            e = np.exp(logits - logits.max())
            total += e / e.sum()
        real = float(total[1] / len(self.pad_sessions))
        return PadResult(round(real, 4), real >= PAD_REAL_THRESHOLD)

    # ── full pipeline ─────────────────────────────────────────────────────
    def template(self, img: np.ndarray, run_pad: bool = True) -> TemplateResult:
        faces = self.detect(img)
        if len(faces) == 0:
            return TemplateResult(0, None, None, None, None)
        # Largest face = the person at the door / in front of the enrollment camera
        face = max(faces, key=lambda f: f[2] * f[3])
        with self._lock:
            aligned = self.recognizer.alignCrop(img, face)
            feat = self.recognizer.feature(aligned).flatten().astype(np.float64)
        norm = np.linalg.norm(feat)
        template = (feat / norm).round(6).tolist() if norm > 1e-9 else None
        quality = self._quality(face, aligned)
        bbox = tuple(float(v) for v in face[:4])
        pad = self.pad(img, bbox) if run_pad else PadResult(1.0, True)
        return TemplateResult(len(faces), template, quality, pad, [int(v) for v in bbox])


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image")
    # Cap work per request: longest side 1280 px
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        s = 1280 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def cosine_distance(a: list[float], b: list[float]) -> float:
    return float(1.0 - np.dot(np.asarray(a), np.asarray(b)))
