"""TFLite INT8 - duong vong: export SavedModel (FP32) roi tu quantize bang TFLiteConverter.

Vi sao can file nay:
    Duong INT8 mac dinh cua ultralytics dung onnx2tf, bi loi op `Tile` trong head Pose26
    (end2end NMS-free) o CHE DO INT8 - loi ca khi da ha opset. Nhung ban FP32 thi chay duoc.
    => Ta lay SavedModel FP32 (onnx2tf tao ok) roi goi thang tf.lite.TFLiteConverter voi
       representative_dataset -> tu lam PTQ, bo qua nhanh INT8 loi cua onnx2tf.

Usage:
    python to_tflite_int8_tf.py yf26-s2.pt
    python to_tflite_int8_tf.py yf26-s2.pt yf26-r2.pt yf26-r3.pt shufflenet-slim-250e.pt
"""

import shutil
import sys
from pathlib import Path

import numpy as np

DATA_IMAGES = Path("/mnt/d/DOC/Face_identity_detection_system/experiment/data/exp/"
                   "filtered2_recall/dataset_cus/images/val")
N_CALIB = 300      # so anh calibration (khuyen nghi >=300)
IMGSZ_FALLBACK = 320


def train_imgsz(pt: Path) -> int:
    try:
        import torch
        a = torch.load(pt, map_location="cpu", weights_only=False).get("train_args", {}) or {}
        return int(a.get("imgsz") or IMGSZ_FALLBACK)
    except Exception:
        return IMGSZ_FALLBACK


def make_saved_model(pt: Path, imgsz: int) -> Path:
    """Export SavedModel FP32 (nhanh nay chay duoc; fallback opset 17 cho RepViT)."""
    from ultralytics import YOLO
    err = None
    for opset in (None, 17):
        try:
            kw = {"opset": opset} if opset else {}
            out = YOLO(str(pt)).export(format="saved_model", imgsz=imgsz, device="cpu", **kw)
            return Path(out)
        except Exception as e:
            err = e
    raise RuntimeError(f"khong export duoc SavedModel: {err}")


def rep_dataset(imgsz: int):
    """Sinh du lieu calibration tu anh val that (NHWC, float32 0..1)."""
    import cv2
    files = sorted(DATA_IMAGES.glob("*.jpg"))[:N_CALIB]
    if not files:
        raise RuntimeError(f"Khong thay anh calibration trong {DATA_IMAGES}")

    def gen():
        for f in files:
            im = cv2.imread(str(f))
            if im is None:
                continue
            im = cv2.resize(im, (imgsz, imgsz))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            yield [im[None, ...]]
    return gen, len(files)


def convert(pt: Path) -> Path | None:
    import tensorflow as tf

    imgsz = train_imgsz(pt)
    out = Path(f"{pt.stem}_int8.tflite")
    print(f"\n>>> {pt}  ->  {out}   (imgsz={imgsz})")
    try:
        sm = make_saved_model(pt, imgsz)
        print(f"    SavedModel: {sm}")

        gen, n = rep_dataset(imgsz)
        print(f"    calibration: {n} anh tu {DATA_IMAGES.name}/")

        # Thu 2 bo op: builtin truoc; neu model co op la (vd Erf tu GELU cua RepViT)
        # thi them SELECT_TF_OPS (Flex delegate) - file to hon va runtime phai co flex.
        opsets = [
            ("builtin", [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]),
            ("builtin+flex", [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS,
                              tf.lite.OpsSet.SELECT_TF_OPS]),
        ]
        tflite_model, used, last = None, None, None
        for label, ops in opsets:
            try:
                conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                conv.representative_dataset = gen
                conv.target_spec.supported_ops = ops
                tflite_model, used = conv.convert(), label
                break
            except Exception as e:
                last = e
                if label == "builtin":
                    print("    ...builtin khong du op (vd Erf/GELU) -> thu them SELECT_TF_OPS")
        if tflite_model is None:
            raise last
        out.write_bytes(tflite_model)
        if used != "builtin":
            print(f"    LUU Y: dung {used} -> runtime phai bat Flex delegate")

        mb_in, mb_out = pt.stat().st_size / 1e6, out.stat().st_size / 1e6
        print(f"    OK  {mb_in:.2f} MB (.pt) -> {mb_out:.2f} MB (int8 tflite)  "
              f"giam {100*(1-mb_out/mb_in):.0f}%")
        return out
    except Exception as e:
        print(f"    LOI: {str(e).splitlines()[0][:200]}")
        return None


def main():
    args = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    models = [m for m in args if m.exists()] or [p for p in Path(".").glob("*.pt")]
    if not models:
        raise SystemExit("Khong tim thay .pt nao.")
    ok = [convert(m) for m in models]
    print(f"\nXong: {sum(x is not None for x in ok)}/{len(models)} thanh cong.")


if __name__ == "__main__":
    main()
