"""Convert any .pt / .pth checkpoint to TFLite - the format is auto-detected.

One entry point for every checkpoint flavour in this project:

    RFB state_dict (.pth/.pt)   -> convert_to_tflite.py  (ONNX -> onnx2tf -> tflite)
    Ultralytics YOLO (.pt)      -> YOLO(...).export(format="tflite")
    TorchScript / nn.Module     -> torch.onnx.export -> onnx2tf -> tflite

Install once (CPU is fine):
    pip install onnx onnxruntime onnx2tf tensorflow tf_keras psutil onnxsim
    pip install ultralytics          # only for YOLO .pt

Examples:
    python convert_pt_to_tflite.py models/rfb_landmark_xxx/best.pth
    python convert_pt_to_tflite.py models/rfb_landmark_xxx/best.pth --img_size 640 --int8 \
        --calib_dir ../../data/exp/filtered2_recall/dataset_cus/images/val
    python convert_pt_to_tflite.py ../ultralytics/yf26-s2.pt --half
    python convert_pt_to_tflite.py ../ultralytics/yf26-s2.pt --int8 \
        --data ../../data/exp/filtered2_recall/dataset_cus/dataset.yaml
    python convert_pt_to_tflite.py any_model.pt --detect_only    # just print the format

int8 calibration input differs per route: --calib_dir (folder of images) for
RFB, --data (dataset .yaml) + --fraction for Ultralytics.

For batch-converting several YOLO models at once, ../ultralytics/to_tflite.py
already does that (scan folder + skip existing); this script handles one file.
"""
import argparse
import os
import subprocess
import sys
import zipfile

import torch

parser = argparse.ArgumentParser(description='Convert any .pt/.pth checkpoint to TFLite')
parser.add_argument('model_path', help='checkpoint to convert (.pt or .pth)')
parser.add_argument('--img_size', default=320, type=int,
                    help='square input size; for RFB it must match the training/inference size')
parser.add_argument('--out_dir', default=None, help='output folder (default: next to the checkpoint)')
parser.add_argument('--half', action='store_true', help='fp16 weights (YOLO/generic route)')
parser.add_argument('--int8', action='store_true', help='also write an int8 tflite')
parser.add_argument('--calib_dir', default=None,
                    help='RFB route: folder of images used for int8 calibration')
parser.add_argument('--calib_n', default=100, type=int, help='RFB route: number of calibration images')
parser.add_argument('--data', default=None,
                    help='YOLO route: dataset .yaml used for int8 calibration')
parser.add_argument('--fraction', default=0.06, type=float,
                    help='YOLO route: fraction of the val split to calibrate on (0.06 ~ 460 imgs)')
parser.add_argument('--opset', default=17, type=int, help='ONNX opset for the intermediate graph')
parser.add_argument('--detect_only', action='store_true', help='print the detected format and exit')
args = parser.parse_args()

if not os.path.isfile(args.model_path):
    sys.exit(f"Checkpoint not found: {args.model_path}")

REPO = os.path.dirname(os.path.abspath(__file__))
stem = os.path.splitext(os.path.basename(args.model_path))[0]
out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.model_path)) or ".",
                                       f"{stem}_tflite")


def detect_kind(path):
    """rfb | state_dict | ultralytics | torchscript | module (never unpickles blindly)."""
    try:  # TorchScript archives carry a constants.pkl / code/ folder
        torch.jit.load(path, map_location="cpu")
        return "torchscript"
    except Exception:
        pass
    try:  # plain state_dict: safe to load without executing pickled classes
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and obj:
            keys = list(obj.keys())
            if any(k.startswith(("base_net.", "classification_headers.")) for k in keys):
                return "rfb"
            return "state_dict"
    except Exception:
        pass
    try:  # peek at the pickle bytes instead of unpickling to spot the framework
        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
            blob = z.read(name) if name else b""
        if b"ultralytics" in blob:
            return "ultralytics"
    except Exception:
        pass
    return "module"


kind = detect_kind(args.model_path)
print(f"Detected: {kind}  ({args.model_path})")
if args.detect_only:
    sys.exit(0)

# int8 calibration input differs per route: images folder (RFB/ORT) vs dataset yaml (ultralytics)
if args.int8:
    if kind == "ultralytics" and not args.data:
        sys.exit("--int8 on a YOLO checkpoint needs --data <dataset.yaml> (not a folder), e.g.\n"
                 "  --data ../../data/exp/filtered2_recall/dataset_cus/dataset.yaml")
    if kind != "ultralytics" and not args.calib_dir:
        sys.exit("--int8 needs --calib_dir <folder of images> for calibration.")


def run_rfb():
    """Delegate to convert_to_tflite.py so the RFB graph stays defined in one place."""
    cmd = [sys.executable, os.path.join(REPO, "convert_to_tflite.py"),
           "--model_path", args.model_path,
           "--img_size", str(args.img_size),
           "--out_dir", out_dir,
           "--opset", str(args.opset)]
    if args.int8:
        cmd += ["--int8", "--calib_dir", args.calib_dir, "--calib_n", str(args.calib_n)]
    print("->", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


def run_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("This is an Ultralytics checkpoint: pip install ultralytics")
    kw = dict(format="tflite", imgsz=args.img_size, half=args.half, device="cpu")
    if args.int8:  # ultralytics calibrates from the dataset yaml, not a plain folder
        kw.update(int8=True, half=False, data=args.data, fraction=args.fraction)
    produced = YOLO(args.model_path).export(**kw)
    print(f"TFLite -> {produced}")


def run_generic():
    """TorchScript / pickled nn.Module -> ONNX -> onnx2tf -> tflite."""
    try:
        import onnx2tf
    except ImportError:
        sys.exit("Missing onnx2tf: pip install onnx onnx2tf tensorflow tf_keras psutil onnxsim")
    if kind == "torchscript":
        net = torch.jit.load(args.model_path, map_location="cpu")
    else:
        # full nn.Module pickle: trusted-source load (weights_only=True already failed)
        net = torch.load(args.model_path, map_location="cpu", weights_only=False)
        if isinstance(net, dict):
            sys.exit("Checkpoint is a dict, not a model. If it is a state_dict for a custom "
                     "architecture, build the model yourself and torch.save the module.")
    net.eval()
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{stem}_{args.img_size}.onnx")
    torch.onnx.export(net, torch.randn(1, 3, args.img_size, args.img_size), onnx_path,
                      input_names=["input"], opset_version=args.opset, dynamo=False)
    print(f"ONNX   -> {onnx_path}")
    onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=out_dir,
                    copy_onnx_input_output_names_to_tflite=True, non_verbose=True)
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".tflite"):
            print(f"TFLite -> {os.path.join(out_dir, f)} "
                  f"({os.path.getsize(os.path.join(out_dir, f))/1024/1024:.2f} MB)")


if kind == "rfb":
    run_rfb()
elif kind == "ultralytics":
    run_ultralytics()
elif kind == "state_dict":
    sys.exit("Plain state_dict of an unknown architecture - a state_dict alone has no graph.\n"
             "Use convert_to_tflite.py if it is an RFB checkpoint, or save the built model with "
             "torch.save(model) / torch.jit.script(model) first.")
else:
    run_generic()
