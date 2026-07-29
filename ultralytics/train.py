import os
import sys
import torch
import wandb
from pathlib import Path

from ultralytics import YOLO, settings
from ultralytics.utils import SETTINGS

# ---- Weights & Biases ----
settings.update({"wandb": True})

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"CUDA device set to: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Please ensure a GPU runtime is selected.")

wandb.login(key="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")
os.environ["WANDB_ENTITY"] = "trantrung20023"

print("wandb version :", wandb.__version__)
print("api_key set   :", wandb.api.api_key is not None)
print("SETTINGS[wandb]:", SETTINGS["wandb"], type(SETTINGS["wandb"]))

# task="pose" (5 landmark khuôn mặt) -> phải dùng yaml head Pose26 (*-pose.yaml),
# yaml detect thường không có nhánh keypoint nên không học được landmark.
MODELS = {
    "base": "yolo26.yaml",                       # baseline detect (chưa có biến thể pose)
    "shufflenet-slim": "yolo26n-shufflenetv2-face-pose-slim.yaml",  # bỏ C2PSA + slim P5 (giữ đủ P3/P4/P5)
    "shufflenet-slim2": "yolo26n-shufflenetv2-face-pose-slim2.yaml",  # = slim, bớt thêm 1 block s1 ở P5
    "repvit-slim": "yolo26n-repvit-face-pose-slim.yaml",            # cùng phép cắt với shufflenet-slim
    "repvit-slim2": "yolo26n-repvit-face-pose-slim2.yaml",          # = repvit-slim, bỏ 1 block P5 + SimSPPF
    "repvit-slim3": "yolo26n-repvit-face-pose-slim3.yaml",          # = repvit-slim2, bỏ hẳn SimSPPF (test RF)
}


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "shufflenet-slim"
    cfg = MODELS.get(key, key)
    model = YOLO(cfg)
    EPOCHS = 150
    run_name = f"{Path(cfg).stem}_{EPOCHS}e"
    print(f"🏗️  Kiến trúc: {cfg}  ->  runs_face/{run_name}")

    # ====== FULL RUN 250 EPOCH (shufflenet, RTX 3060 12GB, batch 16 + AMP) ======
    model.train(
    data="/mnt/d/DOC/Face_identity_detection_system/experiment/data/exp/filtered2_recall/dataset_cus/dataset.yaml",
    task="pose",

    epochs=150, imgsz=320, batch=32, patience=20,
    device=0, workers=8, cache=False, amp=False, save_period=50,

    optimizer="AdamW",
    lr0=0.003,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    cos_lr=True,

    box=8.5, cls=0.5, dfl=1.5,
    single_cls=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=5.0, translate=0.10, scale=0.50, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=20, mixup=0.0, copy_paste=0.0,

    project="repvit-slim2",
    name=run_name,
    pretrained=False, save=True, plots=True,
)


if __name__ == "__main__":
    main()
