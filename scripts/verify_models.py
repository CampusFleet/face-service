"""Verifies every model file against models/MANIFEST.sha256. Exit code 1 on any mismatch.

Usage: python scripts/verify_models.py [models_dir]
Run in CI and at container start so a tampered or wrong model never serves traffic.
"""
import hashlib
import os
import sys


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(models_dir: str, required_only: bool = False) -> list[str]:
    errors = []
    with open(os.path.join(models_dir, "MANIFEST.sha256")) as f:
        entries = [line.split() for line in f if line.strip()]
    for expected, name in entries:
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            # .pth sources are only needed for conversion, not at runtime
            if required_only and name.endswith(".pth"):
                continue
            errors.append(f"missing: {name}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(f"checksum mismatch: {name} expected {expected[:12]}… got {actual[:12]}…")
    return errors


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "models")
    errs = verify(d, required_only="--runtime" in sys.argv)
    for e in errs:
        print("FAIL", e)
    print("OK: all model checksums verified" if not errs else f"{len(errs)} problem(s)")
    sys.exit(1 if errs else 0)
