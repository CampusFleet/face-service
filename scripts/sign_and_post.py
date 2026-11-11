"""Demo client: signs an image and posts it to the face service.
Usage: FACE_SERVICE_KEY=... python scripts/sign_and_post.py image.jpg [http://localhost:8080]
"""
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.request

path = "/v1/template"
img = open(sys.argv[1], "rb").read()
base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"
key = os.environ["FACE_SERVICE_KEY"].encode()
ts = str(int(time.time()))
sig = hmac.new(key, f"{ts}\nPOST\n{path}\n{hashlib.sha256(img).hexdigest()}".encode(), hashlib.sha256).hexdigest()
req = urllib.request.Request(base + path, data=img, method="POST",
                             headers={"content-type": "image/jpeg", "x-cf-timestamp": ts, "x-cf-signature": sig})
out = json.load(urllib.request.urlopen(req))
out["template"] = (out["template"][:4] + ["…"]) if out.get("template") else None
print(json.dumps(out, indent=2))
