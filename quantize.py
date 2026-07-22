#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Quantize a (optionally pruned + fine-tuned) YOLO model to INT8 for edge inference.

Methods (pick with --method) — run each and compare, as requested:

  fx     PyTorch FX-graph post-training static quantization -> INT8 model for CPU torch/ONNX.
         Robust only after `model.fuse()`; the Detect head sometimes resists FX symbolic tracing, in which
         case the script reports it and you should fall back to --method onnx. Saved as TorchScript.

  onnx   Export FP32 ONNX (via Ultralytics) then ONNX Runtime *static* INT8 quantization (QDQ, per-channel).
         This is the most portable edge path (ONNX Runtime on ARM/x86). Output: <model>.int8.onnx

  ncnn   Export to NCNN (via Ultralytics) and print the exact ncnn2table + ncnn2int8 calibration commands.
         Produces an INT8 NCNN model for ARM mobile.

All methods calibrate INT8 activation ranges on REAL images from --data (a folder of images, or the val
images of a dataset). Calibration data should resemble deployment data.

Examples
--------
    python edge/quantize.py --model runs/yolo26n_struct.pt --method onnx --data datasets/coco8/images/val --imgsz 640
    python edge/quantize.py --model runs/yolo26n_struct.pt --method fx   --data datasets/coco8/images/val
    python edge/quantize.py --model runs/yolo26n_struct.pt --method ncnn --data datasets/coco8/images/val

NOTE: not executed in the authoring environment (no torch). Run on a machine with torch + onnxruntime
(and the ncnn tools for --method ncnn) installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(data: str, limit: int) -> list[Path]:
    """Collect up to ``limit`` calibration image paths from a folder (recursively) or a single image."""
    p = Path(data)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"no calibration images found under {data!r}")
    return files[:limit]


def load_batch(path: Path, imgsz: int) -> np.ndarray:
    """Letterbox an image to (1, 3, imgsz, imgsz) float32 in [0, 1], CHW, RGB — matching Ultralytics preprocess."""
    import cv2  # lazy: only needed at run time

    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = round(h * r), round(w * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top : top + nh, left : left + nw] = im
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW
    return np.ascontiguousarray(x)


# --------------------------------------------------------------------------------------------------------- FX
def quant_fx(model_path: str, data: str, imgsz: int, n: int) -> None:
    """PyTorch FX-graph static PTQ to INT8 (CPU)."""
    import torch
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    from ultralytics import YOLO

    torch_model = YOLO(model_path).model.fuse().eval().float()
    example = torch.zeros(1, 3, imgsz, imgsz)
    qmap = get_default_qconfig_mapping("x86")
    try:
        prepared = prepare_fx(torch_model, qmap, (example,))
    except Exception as exc:  # FX symbolic trace can fail on the detection head
        raise SystemExit(
            f"FX tracing failed ({exc}).\nThe YOLO head is not always FX-traceable — use '--method onnx' instead, "
            "or quantize only the backbone."
        ) from exc

    with torch.no_grad():
        for path in list_images(data, n):  # calibrate activation ranges
            prepared(torch.from_numpy(load_batch(path, imgsz)))
    int8 = convert_fx(prepared)

    out = str(Path(model_path).with_suffix("")) + ".int8.torchscript"
    torch.jit.save(torch.jit.trace(int8, example), out)
    print(f"saved {out}  (PyTorch INT8, CPU)")


# ------------------------------------------------------------------------------------------------------- ONNX
def quant_onnx(model_path: str, data: str, imgsz: int, n: int) -> None:
    """Export FP32 ONNX then ONNX Runtime static INT8 (QDQ, per-channel)."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from ultralytics import YOLO

    fp32 = YOLO(model_path).export(format="onnx", imgsz=imgsz, opset=13, simplify=True, dynamic=False)
    prep = str(Path(fp32).with_suffix("")) + ".prep.onnx"
    quant_pre_process(fp32, prep)  # shape inference + cleanup, recommended before static quant

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter(load_batch(p, imgsz) for p in list_images(data, n))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"images": x}

    out = str(Path(model_path).with_suffix("")) + ".int8.onnx"
    quantize_static(
        prep,
        out,
        Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"saved {out}  (ONNX Runtime INT8)")


# ------------------------------------------------------------------------------------------------------- NCNN
def quant_ncnn(model_path: str, data: str, imgsz: int, n: int) -> None:
    """Export NCNN (FP32) and print the calibration + INT8 conversion commands for the ncnn toolchain."""
    from ultralytics import YOLO

    ncnn_dir = YOLO(model_path).export(format="ncnn", imgsz=imgsz)  # -> *_ncnn_model/{model.param,model.bin}
    stem = Path(model_path).stem
    print(f"exported FP32 NCNN model to: {ncnn_dir}")
    print("\nBuild the INT8 model with the ncnn tools (https://github.com/Tencent/ncnn):")
    print(f"  # 1) list calibration images (use {n} images from {data})")
    print(f"  find {data} -type f > imagelist.txt")
    print("  # 2) build the calibration table")
    print(
        f"  ncnn2table {ncnn_dir}/model.param {ncnn_dir}/model.bin imagelist.txt {stem}.table "
        f'mean=[0,0,0] norm=[0.00392,0.00392,0.00392] shape=[{imgsz},{imgsz},3] pixel=RGB thread=4 method=kl'
    )
    print("  # 3) quantize to INT8")
    print(
        f"  ncnn2int8 {ncnn_dir}/model.param {ncnn_dir}/model.bin {stem}-int8.param {stem}-int8.bin {stem}.table"
    )


def main() -> None:
    """Dispatch to the chosen INT8 quantization method."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="trained/pruned model .pt")
    ap.add_argument("--method", choices=["fx", "onnx", "ncnn"], default="onnx")
    ap.add_argument("--data", required=True, help="folder of calibration images (e.g. the val split)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=200, help="number of calibration images")
    args = ap.parse_args()

    {"fx": quant_fx, "onnx": quant_onnx, "ncnn": quant_ncnn}[args.method](
        args.model, args.data, args.imgsz, args.n
    )


if __name__ == "__main__":
    main()
