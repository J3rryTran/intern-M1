import argparse
import time
from pathlib import Path

import torch

from ultralytics import YOLO

DATA = "/mnt/d/DOC/Face_identity_detection_system/experiment/data/exp/filtered2_recall/dataset_cus/dataset.yaml"
IMGSZ = 640


def cmd_val(args):
    """Đo mAP trên split val/test — tương đương bước validate cuối lúc train."""
    model = YOLO(args.weights)
    m = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,      # xuất PR curve, confusion matrix vào save_dir
        save_json=False,
        verbose=True,
    )

    print("\n" + "=" * 62)
    print(f"  WEIGHTS : {args.weights}")
    print(f"  SPLIT   : {args.split}   |  imgsz {args.imgsz}  |  conf {args.conf}  iou {args.iou}")
    print("=" * 62)
    print("  BOX (khuôn mặt)")
    print(f"    precision {m.box.mp:.4f}   recall {m.box.mr:.4f}")
    print(f"    mAP50     {m.box.map50:.4f}   mAP50-95 {m.box.map:.4f}")
    if getattr(m, "pose", None) is not None:
        print("  POSE (5 landmark)")
        print(f"    precision {m.pose.mp:.4f}   recall {m.pose.mr:.4f}")
        print(f"    mAP50     {m.pose.map50:.4f}   mAP50-95 {m.pose.map:.4f}")
    sp = m.speed  # ms/ảnh
    total = sum(sp.values())
    print(f"  SPEED   : {sp['preprocess']:.1f} pre + {sp['inference']:.1f} infer + "
          f"{sp['postprocess']:.1f} post = {total:.1f} ms  (~{1000 / total:.0f} FPS)")
    print(f"  SAVED   : {m.save_dir}")
    print("=" * 62)
    return m


def cmd_predict(args):
    """Chạy suy luận trên ảnh / thư mục / video, lưu ảnh đã vẽ box + landmark."""
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_txt,
        stream=False,
        verbose=False,
    )

    n_face = 0
    for r in results:
        n_face += len(r.boxes)
    print(f"\nĐã xử lý {len(results)} ảnh/frame, phát hiện {n_face} khuôn mặt.")
    if results:
        print(f"Kết quả lưu tại: {results[0].save_dir}")
        r = results[0]
        if r.keypoints is not None and len(r.boxes):
            print(f"Ảnh đầu: {len(r.boxes)} mặt, keypoints shape {tuple(r.keypoints.data.shape)} "
                  f"(n_face, 5 landmark, [x, y, conf])")
    return results


def cmd_bench(args):
    """Đo tốc độ suy luận thuần trên ảnh giả lập (không cần dataset)."""
    model = YOLO(args.weights)
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    model.predict(dummy, device=args.device, verbose=False)  # warm-up

    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(dummy, device=args.device, verbose=False)
    dt = (time.perf_counter() - t0) / n * 1000

    info = model.model.info(verbose=False)
    print(f"\n{Path(args.weights).name}: {dt:.2f} ms/ảnh  (~{1000 / dt:.0f} FPS) "
          f"@ imgsz {args.imgsz}, device {args.device}")
    if info:
        print(f"  layers {info[0]} | params {info[1]:,} | GFLOPs {info[3]:.1f}")


def main():
    p = argparse.ArgumentParser(description="Val / predict cho YOLO26 face-pose")
    p.add_argument("mode", choices=["val", "predict", "bench"])
    p.add_argument("weights", help="đường dẫn .pt (vd runs_face/.../weights/best.pt)")
    p.add_argument("source", nargs="?", default=None, help="ảnh/thư mục/video (chỉ cho predict)")
    p.add_argument("--data", default=DATA)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--conf", type=float, default=None, help="mặc định: 0.001 cho val, 0.25 cho predict")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default=0, help="0 cho GPU, 'cpu' cho CPU")
    p.add_argument("--save-txt", action="store_true", help="lưu nhãn .txt kèm kết quả predict")
    args = p.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Không thấy weights: {args.weights}")

    if args.mode == "val":
        args.conf = 0.001 if args.conf is None else args.conf  # conf thấp để mAP chuẩn
        cmd_val(args)
    elif args.mode == "predict":
        if not args.source:
            raise SystemExit("predict cần <source>: đường dẫn ảnh/thư mục/video")
        args.conf = 0.25 if args.conf is None else args.conf
        cmd_predict(args)
    else:
        args.conf = 0.25 if args.conf is None else args.conf
        cmd_bench(args)


if __name__ == "__main__":
    main()
