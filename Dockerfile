# Campus Fleet face service — CPU only, stateless.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime models only (ONNX); .pth sources are for conversion and not shipped.
COPY models/*.onnx models/MANIFEST.sha256 ./models/
COPY scripts/verify_models.py ./scripts/
COPY app ./app

RUN python scripts/verify_models.py models --runtime

RUN useradd --uid 10001 --no-create-home face && chown -R face /srv
USER face

EXPOSE 8080
# 2 workers ≈ 2 vCPU; each request is ~50–150 ms CPU.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--proxy-headers"]
