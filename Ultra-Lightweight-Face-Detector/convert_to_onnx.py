"""Convert a trained RFB landmark checkpoint (.pth state_dict) to ONNX.

The exported graph is the is_test forward, so post-processing that can be
traced is already baked in (P priors: 320->5875, 480->13242, 640->23500):
    scores    [1, P, 2]  - softmax probabilities (bg, face)
    boxes     [1, P, 4]  - corner-form (x1, y1, x2, y2), percent coords
    landmarks [1, P, 10] - (x0, y0, ..., x4, y4), percent coords
Only score-thresholding + NMS remain to be done by the consumer (see
vision/utils/box_utils_numpy.py, same as the original repo's ONNX demos).

IMPORTANT: priors are baked in as constants -> --img_size must match the
resolution you will run inference at (use the size the model was trained on).

--int8 additionally writes <name>_<size>_int8.onnx: ONNX Runtime static PTQ
(QDQ, per-channel, same recipe as intern-M1/quantize.py) calibrated on real
images, then checked against the fp32 model on confident priors.

Examples:
    python convert_to_onnx.py --model_path models/rfb_xxx/best.pth              # 320
    python convert_to_onnx.py --model_path models/rfb_xxx/best.pth --img_size 640
    python convert_to_onnx.py --model_path models/rfb_xxx/best.pth \
        --int8 --calib_dir ../../data/exp/filtered2_recall/dataset_cus/images/val
"""
import argparse
import os
import sys

import torch

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(description='Export RFB-landmark .pth to ONNX')
parser.add_argument('--model_path', default='models/train-landmark/best.pth',
                    help='state_dict checkpoint to export')
parser.add_argument('--img_size', default=320, type=int,
                    help='square input size the graph is traced at (320/480/640); '
                         'must match the intended inference resolution')
parser.add_argument('--output', default=None,
                    help='output .onnx path (default: models/onnx/<name>_<size>.onnx)')
parser.add_argument('--device', default='cpu',
                    help='device used for tracing: cpu (default) or cuda:0 - '
                         'the resulting ONNX runs anywhere either way')
parser.add_argument('--opset', default=17, type=int, help='ONNX opset version')
parser.add_argument('--int8', action='store_true',
                    help='also write a static-quantized int8 ONNX (needs --calib_dir)')
parser.add_argument('--calib_dir', default=None,
                    help='folder of representative images (jpg/png) for int8 calibration')
parser.add_argument('--calib_n', default=200, type=int,
                    help='number of calibration images to use')
parser.add_argument('--skip_check', action='store_true',
                    help='skip the onnxruntime parity check after export')
args = parser.parse_args()

if args.int8 and not args.calib_dir:
    sys.exit("--int8 needs --calib_dir <folder of images> for calibration.")

define_img_size(args.img_size)
from vision.ssd.config import fd_config
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd

if not os.path.isfile(args.model_path):
    sys.exit(f"Checkpoint not found: {args.model_path}\n"
             f"Pass --model_path <file.pth> (e.g. models/<run>/best.pth).")

W, H = fd_config.image_size

net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=args.device)
net.load(args.model_path)
net.eval()
net.to(args.device)

output_path = args.output
if output_path is None:
    stem = os.path.splitext(os.path.basename(args.model_path))[0]
    output_path = os.path.join("models", "onnx", f"{stem}_{args.img_size}.onnx")
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

if args.int8:
    import glob

    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
        from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                              QuantType, quantize_static)
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as e:
        sys.exit(f"--int8 needs onnxruntime + opencv ({e.name} missing): "
                 f"pip install onnxruntime opencv-python")

    files = sorted(f for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
                   for f in glob.glob(os.path.join(args.calib_dir, ext)))[:args.calib_n]
    if not files:
        sys.exit(f"No calibration images found in {args.calib_dir}")

    def _load(f):
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H)).astype(np.float32)
        return ((img - 127.0) / 128.0).transpose(2, 0, 1)[None]  # NCHW, same as TestTransform

    prep_path = output_path.replace(".onnx", ".prep.onnx")
    try:
        quant_pre_process(output_path, prep_path)  # shape inference + cleanup
    except Exception as e:
        print(f"quant_pre_process failed ({e}); quantizing the raw model instead.")
        prep_path = output_path

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(_load(f) for f in files)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"input": x}

    int8_path = output_path.replace(".onnx", "_int8.onnx")
    quantize_static(prep_path, int8_path, _Reader(),
                    quant_format=QuantFormat.QDQ,
                    per_channel=True,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8)
    if prep_path != output_path and os.path.isfile(prep_path):
        os.remove(prep_path)
    print(f"int8  : {int8_path} ({os.path.getsize(int8_path)/1024/1024:.2f} MB, "
          f"calibrated on {len(files)} imgs)")

    # fp32 vs int8 on real images: deviation over confident priors (score > 0.5)
    s32 = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    d_sc = d_bx = d_lm = 0.0
    n_conf = 0
    for f in files[:8]:
        x = _load(f)
        a_sc, a_bx, a_lm = s32.run(None, {"input": x})
        b_sc, b_bx, b_lm = s8.run(None, {"input": x})
        m = a_sc[0, :, 1] > 0.5
        if m.any():
            n_conf += int(m.sum())
            d_sc = max(d_sc, float(np.abs(a_sc[0, m, 1] - b_sc[0, m, 1]).max()))
            d_bx = max(d_bx, float(np.abs(a_bx[0, m] - b_bx[0, m]).max()))
            d_lm = max(d_lm, float(np.abs(a_lm[0, m] - b_lm[0, m]).max()))
    print(f"int8 check (8 real imgs, {n_conf} confident priors): "
          f"max dScore {d_sc:.4f}, max dBox {d_bx:.4f}, max dLandm {d_lm:.4f} "
          f"(percent coords; <0.01 ~ under 1% of image size)")

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
