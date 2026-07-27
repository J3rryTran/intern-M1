#!/usr/bin/env bash
# Fine-tune RFB landmark.

# ========================== CONFIG ==========================
DATA=/mnt/d/DOC/Face_identity_detection_system/experiment/data/exp/filtered2_recall/dataset_cus
IMG_SIZE=320
EPOCHS=30
FREEZE=5
BATCH=16
WORKERS=4
SCHEDULER=cosine      # cosine | multi-step | plateau

# ------------------- MONITOR (wandb / console / CSV) -------------------
# Tu dong moi epoch khi bat wandb: train|val loss, acc, Face P/R/F1,
# landmark PCK@thr + per-point (eyeL/eyeR/nose/mouthL/mouthR), NME, lr.
# 2 muc duoi KHONG bat mac dinh -> phai set o day, khong se VANG tren wandb:
MAP_PERIOD=5          # mAP@[.5:.95] moi N epoch (+ epoch cuoi); 0 = tat. RAT cham tren CPU.
TEST_SPLIT=           # acc/loss/mAP test cuoi run; "" = tat (dataset hien chi co train/val)
PCK_THRESHOLD=0.1     # nguong PCK cho landmark accuracy

USE_WANDB=0           # 1 = bat wandb (login 1 lan truoc: `wandb login`)
WANDB_PROJECT=rfb640-landmark
WANDB_ENTITY=         # team/user tren wandb; "" = tai khoan mac dinh

RUN_NAME=rfb_landmark_$(date +%Y%m%d_%H%M%S)
OUT=models/${RUN_NAME}
# ======================================================================

cd "$(dirname "$0")"
mkdir -p "$OUT"

# --WANDB_API_KEY=xxx tren CLI: tach ra, export, tu bat wandb; con lai chuyen cho train.py
PASS_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --WANDB_API_KEY=*) export WANDB_API_KEY="${1#*=}"; USE_WANDB=1 ;;
        --WANDB_API_KEY)   export WANDB_API_KEY="$2"; USE_WANDB=1; shift ;;
        *) PASS_ARGS+=("$1") ;;
    esac
    shift
done

WANDB_ARGS=()
if [ "$USE_WANDB" = "1" ]; then
    WANDB_ARGS=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN_NAME")
    [ -n "$WANDB_ENTITY" ] && WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    # kiem tra login mot lan, truoc khi train, de khong chay ca run roi moi phat hien
    python - <<'PY' || exit 1
import os, sys
try:
    import wandb
except ImportError:
    sys.exit("wandb chua cai:  pip install wandb")
if not (os.environ.get("WANDB_API_KEY") or wandb.api.api_key):
    sys.exit("wandb chua co API key. Chon 1 cach (key: https://wandb.ai/authorize):\n"
             "   bash train.sh --WANDB_API_KEY=<key>   # nhap tay moi lan chay\n"
             "   wandb login                           # luu 1 lan / may (~/.netrc)\n"
             "   export WANDB_API_KEY=<key>            # bien moi truong / phien\n"
             "hoac dat USE_WANDB=0 de train khong dung wandb.")
PY
fi

echo "Run: $RUN_NAME  |  img_size: $IMG_SIZE  |  wandb: $USE_WANDB  |  map_period: $MAP_PERIOD  |  log: $OUT/train.log"

python train.py \
    --datasets "$DATA" \
    --train_split train --val_split val --test_split "$TEST_SPLIT" \
    --img_size "$IMG_SIZE" \
    --num_epochs "$EPOCHS" --freeze_epochs "$FREEZE" \
    --batch_size "$BATCH" --num_workers "$WORKERS" \
    --scheduler "$SCHEDULER" \
    --map_period "$MAP_PERIOD" --pck_threshold "$PCK_THRESHOLD" \
    --checkpoint_folder "$OUT" \
    "${WANDB_ARGS[@]}" \
    "${PASS_ARGS[@]}" 2>&1 | tee "$OUT/train.log"
