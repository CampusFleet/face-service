"""Store AuraFace weights as float16 (half the file size) while the model still computes in float32.

Every float32 initializer becomes float16 plus a Cast(to=float) node; ONNX Runtime folds the casts at
load time, so inference is unchanged apart from weight rounding (checked against the original below).
Needed because the float32 model (260 MB) pushes the Vercel Python bundle over its 500 MB limit.
Usage: python convert/auraface_fp16_weights.py models/auraface_glintr100.onnx models/auraface_glintr100_fp16w.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
casts = []
for init in list(g.initializer):
    if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < 1024:
        continue  # keep small tensors (biases, BN stats) in float32
    arr = numpy_helper.to_array(init).astype(np.float16)
    half = numpy_helper.from_array(arr, init.name + "__fp16")
    g.initializer.remove(init)
    g.initializer.append(half)
    casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
for n in reversed(casts):
    g.node.insert(0, n)
onnx.save(m, dst)
print(f"converted {len(casts)} tensors")
