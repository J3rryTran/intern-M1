"""INT8 PTQ pipeline: val FP32 -> export ONNX INT8 (calib tu dataset) -> val INT8 -> verdict.

Usage:
    python export_int8.py <path/to/best.pt> [mAP50-drop-threshold, default 0.02]

Vi du:
    python export_int8.py runs/pose/runs_face/scratch_yolo26n-shufflenetv2-face-pose-slim_250e/weights/best.pt
"""

import sys
from pathlib import Path

from ultralytics import YOLO

DATA = "/mnt/d/DOC/Face_identity_detection_system/experiment/data/exp/filtered2_recall/dataset_cus/dataset.yaml"
IMGSZ = 640
CALIB_FRACTION = 0.03  # ~3% train (~900 anh) lam calibration - toi thieu nen ~300


def val_metrics(model, tag, device):
    m = model.val(data=DATA, imgsz=IMGSZ, device=device, verbose=False, plots=False)
    row = {
        "tag": tag,
        "box_mAP50": m.box.map50, "box_mAP5095": m.box.map,
        "pose_mAP50": m.pose.map50, "pose_mAP5095": m.pose.map,
    }
    print(f"[{tag}] box mAP50={row['box_mAP50']:.4f} mAP50-95={row['box_mAP5095']:.4f} | "
          f"pose mAP50={row['pose_mAP50']:.4f} mAP50-95={row['pose_mAP5095']:.4f}")
    return row


def main():
    pt = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/pose/runs_face/scratch_yolo26n-shufflenetv2-face-pose-slim_250e/weights/best.pt"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    pt = Path(pt)
    assert pt.exists(), f"Khong thay weights: {pt} (train xong chua?)"

    print(f"=== 1/4 Val FP32 (.pt, GPU) === {pt}")
    fp32 = val_metrics(YOLO(str(pt)), "FP32 .pt", device=0)

    print("=== 2/4 Export ONNX INT8 (ORT static quant, calib tu dataset) ===")
    # Luu y: nhanh int8 cua exporter tao best_int8.onnx roi XOA file onnx trung gian,
    # nen export int8 truoc, fp32 onnx sau (de doi chieu kich thuoc/toc do).
    int8_path = YOLO(str(pt)).export(
        format="onnx", imgsz=IMGSZ, int8=True, data=DATA, fraction=CALIB_FRACTION, device=0
    )
    fp32_onnx = YOLO(str(pt)).export(format="onnx", imgsz=IMGSZ, device=0)

    sz_pt = pt.stat().st_size / 1e6
    sz_32 = Path(fp32_onnx).stat().st_size / 1e6
    sz_8 = Path(int8_path).stat().st_size / 1e6
    print(f"Size: best.pt {sz_pt:.1f}MB | onnx fp32 {sz_32:.1f}MB | onnx int8 {sz_8:.1f}MB "
          f"(giam {100 * (1 - sz_8 / sz_32):.0f}%)")

    print("=== 3/4 Val ONNX INT8 (onnxruntime, cham hon - kien nhan) ===")
    int8 = val_metrics(YOLO(int8_path), "INT8 onnx", device="cpu")

    print("=== 4/4 Verdict ===")
    d_box = fp32["box_mAP50"] - int8["box_mAP50"]
    d_pose = fp32["pose_mAP50"] - int8["pose_mAP50"]
    print(f"Drop mAP50: box {d_box:+.4f} | pose {d_pose:+.4f} (nguong {threshold})")
    if max(d_box, d_pose) <= threshold:
        print(f"PASS -> dung duoc: {int8_path}")
    else:
        print("FAIL -> drop qua nguong. Thu: tang CALIB_FRACTION (0.03->0.1), "
              "hoac OpenVINO int8 (NNCF PTQ thuong tot hon ORT MinMax): "
              "model.export(format='openvino', int8=True, data=DATA), "
              "hoac chap nhan FP16: model.export(format='onnx', half=True).")


if __name__ == "__main__":
    main()
