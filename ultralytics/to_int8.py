"""Chuyen .pt -> ONNX INT8 (PTQ static, calibration tu dataset). KHONG val -> chay nhanh.

Dung khi chi can file INT8; muon do mAP truoc/sau thi dung export_int8.py.

Usage:
    python to_int8.py                      # quet & convert TAT CA model tim thay (bo qua cai da co)
    python to_int8.py yf26-s2.pt           # convert 1 file cu the
    python to_int8.py a.pt b.pt --force    # ep convert lai du da co
    python to_int8.py --list               # chi liet ke trang thai, khong convert
"""

import sys
from pathlib import Path

from ultralytics import YOLO

DATA = "/mnt/d/DOC/Face_identity_detection_system/experiment/data/exp/filtered2_recall/dataset_cus/dataset.yaml"
IMGSZ = 320           # khop imgsz luc train (cac run hien tai deu 320)
CALIB_FRACTION = 0.06  # ~460 anh (ORT khuyen nghi >300)


def is_int8(onnx_path: Path) -> bool:
    """ONNX da quantize chua (co node QuantizeLinear/QLinearConv)."""
    if not onnx_path.exists():
        return False
    raw = onnx_path.read_bytes()
    return b"QuantizeLinear" in raw or b"QLinearConv" in raw


def find_models() -> list[Path]:
    """Tim moi .pt: thu muc goc + runs/*/weights/best.pt."""
    root = Path(".")
    found = [p for p in root.glob("*.pt") if not p.name.endswith("_int8.pt")]
    found += sorted(root.glob("runs/**/weights/best.pt"))
    return found


def int8_path_for(pt: Path) -> Path:
    """File INT8 tuong ung: <ten>_int8.onnx (best.pt -> <ten_run>_int8.onnx o goc)."""
    if pt.name == "best.pt":
        return Path(f"{pt.parent.parent.name}_int8.onnx")
    return pt.with_name(f"{pt.stem}_int8.onnx")


def convert(pt: Path) -> Path | None:
    out = int8_path_for(pt)
    print(f"\n>>> {pt}  ->  {out}")
    try:
        # device='cpu': tranh onnxruntime-gpu (build CUDA 13) xung dot voi torch cu12.8
        produced = YOLO(str(pt)).export(
            format="onnx", imgsz=IMGSZ, int8=True, data=DATA,
            fraction=CALIB_FRACTION, device="cpu",
        )
        produced = Path(produced)
        if produced.resolve() != out.resolve():
            produced.replace(out)
        mb_in, mb_out = pt.stat().st_size / 1e6, out.stat().st_size / 1e6
        print(f"    OK  {mb_in:.2f} MB (.pt) -> {mb_out:.2f} MB (int8)  giam {100*(1-mb_out/mb_in):.0f}%")
        return out
    except Exception as e:
        print(f"    LOI: {e}")
        return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    list_only = "--list" in sys.argv

    models = [Path(a) for a in args] if args else find_models()
    models = [m for m in models if m.exists()]
    if not models:
        raise SystemExit("Khong tim thay file .pt nao.")

    todo, done = [], []
    for m in models:
        (done if (is_int8(int8_path_for(m)) and not force) else todo).append(m)

    print("=" * 66)
    print(f"{'MODEL':<46} {'INT8':>18}")
    print("=" * 66)
    for m in done:
        print(f"{str(m):<46} {'DA CO - bo qua':>18}")
    for m in todo:
        print(f"{str(m):<46} {'can convert':>18}")
    print("=" * 66)

    if list_only or not todo:
        if not todo:
            print("Tat ca model deu da co ban INT8.")
        return

    print(f"\nBat dau convert {len(todo)} model (imgsz={IMGSZ}, calib={CALIB_FRACTION:.0%})...")
    ok = [convert(m) for m in todo]
    print(f"\nXong: {sum(x is not None for x in ok)}/{len(todo)} thanh cong.")


if __name__ == "__main__":
    main()
