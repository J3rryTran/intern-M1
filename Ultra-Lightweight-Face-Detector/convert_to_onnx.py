"""Convert a trained RFB-640 landmark checkpoint (.pth state_dict) to ONNX.

The exported graph is the is_test forward, so post-processing that can be
traced is already baked in:
    scores    [1, 23500, 2]  - softmax probabilities (bg, face)
    boxes     [1, 23500, 4]  - corner-form (x1, y1, x2, y2), percent coords
    landmarks [1, 23500, 10] - (x0, y0, ..., x4, y4), percent coords
Only score-thresholding + NMS remain to be done by the consumer (see
vision/utils/box_utils_numpy.py, same as the original repo's ONNX demos).

Examples:
    python convert_to_onnx.py                                   # best.pth
    python convert_to_onnx.py --model_path models/train-landmark/RFB-landmark-Epoch-24-Loss-22.8504.pth
    python convert_to_onnx.py --device cuda:0
"""
import argparse
import os
import sys

import torch

from vision.ssd.config.fd_config import define_img_size

define_img_size()  # hardcoded 640x640
from vision.ssd.config import fd_config
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd

parser = argparse.ArgumentParser(description='Export RFB-640-landmark .pth to ONNX')
parser.add_argument('--model_path', default='models/train-landmark/best.pth',
                    help='state_dict checkpoint to export')
parser.add_argument('--output', default=None,
                    help='output .onnx path (default: models/onnx/<checkpoint name>.onnx)')
parser.add_argument('--device', default='cpu',
                    help='device used for tracing: cpu (default) or cuda:0 - '
                         'the resulting ONNX runs anywhere either way')
parser.add_argument('--opset', default=17, type=int, help='ONNX opset version')
parser.add_argument('--skip_check', action='store_true',
                    help='skip the onnxruntime parity check after export')
args = parser.parse_args()

if not os.path.isfile(args.model_path):
    sys.exit(f"Checkpoint not found: {args.model_path}\n"
             f"Pass --model_path <file.pth> (e.g. a RFB-landmark-Epoch-*.pth).")

W, H = fd_config.image_size  # 640, 640

net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=args.device)
net.load(args.model_path)
net.eval()
net.to(args.device)

output_path = args.output
if output_path is None:
    stem = os.path.splitext(os.path.basename(args.model_path))[0]
    output_path = os.path.join("models", "onnx", f"{stem}.onnx")
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

dummy_input = torch.randn(1, 3, H, W).to(args.device)  # (N, C, H, W)
torch.onnx.export(net, dummy_input, output_path,
                  verbose=False,
                  input_names=['input'],
                  output_names=['scores', 'boxes', 'landmarks'],
                  opset_version=args.opset,
                  dynamo=False)  # legacy tracer: priors get baked in as constants
size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"Exported {args.model_path}\n     ->  {output_path} ({size_mb:.2f} MB, opset {args.opset})")
print(f"input : input     [1, 3, {H}, {W}]  (RGB, normalized (x-127)/128)")
print(f"output: scores    [1, {fd_config.priors.size(0)}, 2]")
print(f"        boxes     [1, {fd_config.priors.size(0)}, 4]")
print(f"        landmarks [1, {fd_config.priors.size(0)}, 10]")

if not args.skip_check:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed - skipping parity check "
              "(pip install onnxruntime, or pass --skip_check).")
        sys.exit(0)
    x = torch.randn(1, 3, H, W).to(args.device)
    with torch.no_grad():
        torch_out = [t.cpu().numpy() for t in net(x)]
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.cpu().numpy()})
    for name, a, b in zip(("scores", "boxes", "landmarks"), torch_out, onnx_out):
        diff = float(np.abs(a - b).max())
        status = "OK" if diff < 1e-4 else "MISMATCH"
        print(f"parity {name:<9}: max |torch - onnx| = {diff:.2e}  {status}")
