"""Chuyen .pt -> TFLite INT8 (PTQ static, calibration tu dataset).

Ban INT8 nho ~4x so FP32, chay nhanh tren CPU ARM / NPU mobile / Coral EdgeTPU.
Muon ban FP32/FP16 thi dung file rieng: to_tflite.py

Usage:
    python to_tflite_int8.py                 # quet & convert TAT CA model (bo qua cai da co)
    python to_tflite_int8.py yf26-s2.pt      # convert 1 file cu the
    python to_tflite_int8.py --list          # chi liet ke trang thai
    python to_tflite_int8.py yf26-s2.pt --force

Luu y:
  - Can `pip install tensorflow` (ultralytics tu AutoUpdate lan dau, ~600MB).
  - Calibration lay anh tu DATA ben duoi; ORT/TFLite khuyen nghi >300 anh.
  - Landmark (pose) nhay INT8 hon box -> nen do lai mAP bang: python infer.py val <file>
"""

import sys
from pathlib import Path

from ultralytics import YOLO

DATA = "/mnt/d/DOC/Face_identity_detection_system/experiment/data/exp/filtered2_recall/dataset_cus/dataset.yaml"
IMGSZ = 320            # fallback khi khong doc duoc imgsz tu checkpoint
CALIB_FRACTION = 0.06  # ~460 anh (>300 theo khuyen nghi)


def train_imgsz(pt: Path) -> int:
    """Doc imgsz luc train tu checkpoint -> calibration & export dung kich thuoc."""
    try:
        import torch
        a = torch.load(pt, map_location="cpu", weights_only=False).get("train_args", {}) or {}
        return int(a.get("imgsz") or IMGSZ)
    except Exception:
        return IMGSZ


def find_models() -> list[Path]:
    root = Path(".")
    return [p for p in root.glob("*.pt")] + sorted(root.glob("runs/**/weights/best.pt"))


def out_path_for(pt: Path) -> Path:
    stem = pt.parent.parent.name if pt.name == "best.pt" else pt.stem
    return Path(f"{stem}_int8.tflite")


def convert(pt: Path, imgsz: int | None = None) -> Path | None:
    out = out_path_for(pt)
    sz = imgsz or train_imgsz(pt)
    print(f"\n>>> {pt}  ->  {out}   (imgsz={sz})")
    # opset 17 la fallback: onnx2tf dich sai op Tile trong head Pose26 o opset moi (backbone RepViT)
    err = None
    for opset in (None, 17):
        try:
            # device='cpu' de tranh keo onnxruntime-gpu/CUDA khong khop trong buoc trung gian
            kw = {"opset": opset} if opset else {}
            produced = Path(YOLO(str(pt)).export(
                format="tflite", imgsz=sz, int8=True, data=DATA,
                fraction=CALIB_FRACTION, device="cpu", **kw,
            ))
            if produced.resolve() != out.resolve():
                produced.replace(out)
            mb_in, mb_out = pt.stat().st_size / 1e6, out.stat().st_size / 1e6
            note = "" if opset is None else f"  (opset {opset})"
            print(f"    OK  {mb_in:.2f} MB (.pt) -> {mb_out:.2f} MB (int8 tflite)  "
                  f"giam {100*(1-mb_out/mb_in):.0f}%{note}")
            return out
        except Exception as e:
            err = str(e).splitlines()[0]
            if opset is None:
                print(f"    ...opset mac dinh loi ({err[:70]}) -> thu opset 17")
    print(f"    LOI: {err}")
    return None


def main():
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("--")]
    force = "--force" in argv
    list_only = "--list" in argv
    imgsz = next((int(a.split("=")[1]) for a in argv if a.startswith("--imgsz=")), None)

    models = [Path(a) for a in args] if args else find_models()
    models = [m for m in models if m.exists()]
    if not models:
        raise SystemExit("Khong tim thay file .pt nao.")

    todo, done = [], []
    for m in models:
        (done if (out_path_for(m).exists() and not force) else todo).append(m)

    print("=" * 68)
    print(f"{'MODEL':<48} {'TFLITE INT8':>18}")
    print("=" * 68)
    for m in done:
        print(f"{str(m):<48} {'DA CO - bo qua':>18}")
    for m in todo:
        print(f"{str(m):<48} {'can convert':>18}")
    print("=" * 68)

    if list_only or not todo:
        if not todo:
            print("Tat ca model deu da co ban TFLite INT8.")
        return

    print(f"\nBat dau convert {len(todo)} model (imgsz={imgsz or 'tu doc tu checkpoint'}, calib={CALIB_FRACTION:.0%})...")
    ok = [convert(m, imgsz) for m in todo]
    print(f"\nXong: {sum(x is not None for x in ok)}/{len(todo)} thanh cong.")


if __name__ == "__main__":
    main()
