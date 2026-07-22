import os
import sys
import torch
from pathlib import Path

from ultralytics import YOLO, settings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# ---- Weights & Biases ----
# Bật/tắt logging W&B ngay tại đây (SETTINGS toàn cục, KHÔNG phải arg của model.train()).
# Yêu cầu: `pip install wandb` + `wandb login` (hoặc export WANDB_API_KEY=...) một lần.
USE_WANDB = True
if USE_WANDB:
    settings.update({"wandb": True})
    os.environ.setdefault("WANDB_PROJECT", "face-pose")   # đổi tên project tại đây
else:
    settings.update({"wandb": False})

# task="pose" (5 landmark khuôn mặt) -> phải dùng yaml head Pose26 (*-pose.yaml),
# yaml detect thường không có nhánh keypoint nên không học được landmark.
MODELS = {
    "base": "yolo26.yaml",                       # baseline detect (chưa có biến thể pose)
    "shufflenet-slim": "yolo26n-shufflenetv2-face-pose-slim.yaml",  # bỏ C2PSA + slim P5 (giữ đủ P3/P4/P5)
    "shufflenet-slim2": "yolo26n-shufflenetv2-face-pose-slim2.yaml",  # = slim, bớt thêm 1 block s1 ở P5
    "repvit-slim": "yolo26n-repvit-face-pose-slim.yaml",            # cùng phép cắt với shufflenet-slim
}


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "shufflenet-slim"
    cfg = MODELS.get(key, key)
    model = YOLO(cfg)
    EPOCHS = 2
    run_name = f"scratch_{Path(cfg).stem}_{EPOCHS}e"
    print(f"🏗️  Kiến trúc: {cfg}  ->  runs_face/{run_name}")

    # ====== FULL RUN 250 EPOCH (shufflenet, RTX 3060 12GB, batch 16 + AMP) ======
    results = model.train(
        data="/mnt/d/DOC/Face_identity_detection_system/experiment/data/exp/filtered2_recall/dataset_cus/dataset.yaml",
        task="pose",

        epochs=EPOCHS,
        imgsz=640,
        batch=16,            # 32->16: ảnh đông mặt spike TaskAlignedAssigner -> OOM 12GB (đã dính ở epoch 2)
        patience=60,
        device=0,
        workers=8,
        cache=False,
        amp=False,            # bật AMP: ~1/2 VRAM + nhanh hơn trên Ampere; nếu loss NaN vài epoch đầu -> trả về False
        save_period=25,      # run dài: snapshot mỗi 25 epoch (last.pt vẫn tự lưu mỗi epoch để resume)

        # ---- Optimizer / LR ----
        optimizer="AdamW",
        lr0=0.005,           # giữ: đã validate tốt nhất cho shufflenet
        lrf=0.01,            # 0.15->0.01: run dài cần decay sâu (LR cuối ~5e-5); 0.15 là logic sprint
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,   # 1->3: run dài trả warmup về chuẩn cho ổn định đầu train
        cos_lr=True,

        # ---- Loss gains ----
        box=8.5,             # tốt nhất cho shufflenet (9.0 không cải thiện)
        cls=0.5,
        dfl=1.5,

        single_cls=True,

        # ---- Augmentation (nhẹ cho sprint ngắn) ----
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,         # 5->0: mặt thẳng đứng
        translate=0.10,
        scale=0.50,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=20,     # 6->20: run dài tắt mosaic 20 epoch cuối -> box khít (chuẩn ultralytics)
        erasing=0.0,         # bỏ random-erasing
        mixup=0.0,
        copy_paste=0.0,

        project="runs_face",
        name=run_name,
        pretrained=False,
        save=True,
        plots=True,
    )
    return results


if __name__ == "__main__":
    main()
