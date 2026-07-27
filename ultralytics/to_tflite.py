"""Chuyen .pt -> TFLite (FP32 mac dinh, hoac FP16 voi --half).

Dung cho deploy mobile/edge (Android, Raspberry Pi, Coral...).
Muon ban INT8 thi dung file rieng: to_tflite_int8.py

Usage:
    python to_tflite.py                    # quet & convert TAT CA model (bo qua cai da co)
    python to_tflite.py yf26-s2.pt         # convert 1 file cu the
    python to_tflite.py --half             # xuat ban FP16 (nho ~2x, gan nhu khong mat accuracy)
    python to_tflite.py --list             # chi liet ke trang thai
    python to_tflite.py yf26-s2.pt --force # ep convert lai du da co

Luu y: export TFLite di qua TensorFlow SavedModel nen can `pip install tensorflow`
(lan dau chay ultralytics se tu AutoUpdate, tai ~600MB).
"""

import sys
from pathlib import Path

from ultralytics import YOLO

IMGSZ = 320   # fallback khi khong doc duoc imgsz tu checkpoint


def train_imgsz(pt: Path) -> int:
    """Doc imgsz luc train tu checkpoint -> export dung kich thuoc (tranh lech 320/480/640)."""
    try:
        import torch
        a = torch.load(pt, map_location="cpu", weights_only=False).get("train_args", {}) or {}
        return int(a.get("imgsz") or IMGSZ)
    except Exception:
        return IMGSZ


def find_models() -> list[Path]:
    """Tim moi .pt: thu muc goc + runs/*/weights/best.pt."""
    root = Path(".")
    return [p for p in root.glob("*.pt")] + sorted(root.glob("runs/**/weights/best.pt"))


def out_path_for(pt: Path, half: bool) -> Path:
    """<ten>.tflite hoac <ten>_fp16.tflite (best.pt -> lay ten thu muc run)."""
    stem = pt.parent.parent.name if pt.name == "best.pt" else pt.stem
    return Path(f"{stem}_fp16.tflite" if half else f"{stem}.tflite")


def convert(pt: Path, half: bool, imgsz: int | None = None) -> Path | None:
    out = out_path_for(pt, half)
    sz = imgsz or train_imgsz(pt)
    print(f"\n>>> {pt}  ->  {out}   (imgsz={sz})")
    # opset 17 la fallback: onnx2tf dich sai op Tile trong head Pose26 o opset moi (backbone RepViT)
    err = None
    for opset in (None, 17):
        try:
            kw = {"opset": opset} if opset else {}
            produced = Path(YOLO(str(pt)).export(format="tflite", imgsz=sz, half=half, device="cpu", **kw))
            if produced.resolve() != out.resolve():
                produced.replace(out)
            mb_in, mb_out = pt.stat().st_size / 1e6, out.stat().st_size / 1e6
            note = "" if opset is None else f"  (opset {opset})"
            print(f"    OK  {mb_in:.2f} MB (.pt) -> {mb_out:.2f} MB (tflite){note}")
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
    half = "--half" in argv
    force = "--force" in argv
    list_only = "--list" in argv
    imgsz = next((int(a.split("=")[1]) for a in argv if a.startswith("--imgsz=")), None)

    models = [Path(a) for a in args] if args else find_models()
    models = [m for m in models if m.exists()]
    if not models:
        raise SystemExit("Khong tim thay file .pt nao.")

    todo, done = [], []
    for m in models:
        (done if (out_path_for(m, half).exists() and not force) else todo).append(m)

    kind = "TFLITE FP16" if half else "TFLITE FP32"
    print("=" * 68)
    print(f"{'MODEL':<48} {kind:>18}")
    print("=" * 68)
    for m in done:
        print(f"{str(m):<48} {'DA CO - bo qua':>18}")
    for m in todo:
        print(f"{str(m):<48} {'can convert':>18}")
    print("=" * 68)

    if list_only or not todo:
        if not todo:
            print(f"Tat ca model deu da co ban {kind}.")
        return

    print(f"\nBat dau convert {len(todo)} model (imgsz={imgsz or 'tu doc tu checkpoint'}, half={half})...")
    ok = [convert(m, half, imgsz) for m in todo]
    print(f"\nXong: {sum(x is not None for x in ok)}/{len(todo)} thanh cong.")


if __name__ == "__main__":
    main()
