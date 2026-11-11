"""
Converts the Apache-2.0 MiniFASNet PyTorch weights (minivision-ai/Silent-Face-Anti-Spoofing)
to ONNX so the face service runs them with onnxruntime only (no PyTorch at runtime).

Run once (inside the converter image):  python convert_minifasnet.py ../models
"""
import os
import sys
from collections import OrderedDict

import torch

from MiniFASNet import MiniFASNetV1SE, MiniFASNetV2
from utility import get_kernel, parse_model_name

MAPPING = {"MiniFASNetV2": MiniFASNetV2, "MiniFASNetV1SE": MiniFASNetV1SE}


def convert(path: str, out_dir: str) -> str:
    name = os.path.basename(path)
    h, w, model_type, _scale = parse_model_name(name)
    model = MAPPING[model_type](conv6_kernel=get_kernel(h, w))
    state = torch.load(path, map_location="cpu")
    if next(iter(state)).startswith("module."):
        state = OrderedDict((k[7:], v) for k, v in state.items())
    model.load_state_dict(state)
    model.eval()
    out = os.path.join(out_dir, name.replace(".pth", ".onnx"))
    torch.onnx.export(
        model,
        torch.zeros(1, 3, h, w),
        out,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,
    )
    return out


if __name__ == "__main__":
    models_dir = sys.argv[1] if len(sys.argv) > 1 else "../models"
    for f in sorted(os.listdir(models_dir)):
        if f.endswith(".pth"):
            print("converted", convert(os.path.join(models_dir, f), models_dir))
