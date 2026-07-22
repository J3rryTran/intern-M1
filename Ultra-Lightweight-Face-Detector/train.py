"""CLI wrapper around vision.trainer.

All the actual logic lives in vision/trainer.py so the CLI and the Python API
(RFBLandmark().train(...)) stay in sync. For hyper-parameter sweeps prefer the
API:

    from vision.trainer import RFBLandmark
    RFBLandmark().train(data="../data/exp", epochs=50, batch=32, landm_weight=0.5)
"""
import argparse
import logging
import os
import sys

from vision.utils.misc import str2bool

parser = argparse.ArgumentParser(
    description='Fine-tune RFB-640 face detector with a 5-point landmark head')

parser.add_argument('--datasets', nargs='+', default=["../data/exp"],
                    help='Training dataset root(s) (each containing images/ and labels/)')
parser.add_argument('--validation_dataset', default=None,
                    help='Validation dataset root (default: the first --datasets root)')
parser.add_argument('--test_dataset', default=None,
                    help='Test dataset root (default: the first --datasets root)')
parser.add_argument('--train_split', default="train", type=str,
                    help='Subfolder under images/ and labels/ for training ("" for flat layout)')
parser.add_argument('--val_split', default="val", type=str,
                    help='Subfolder under images/ and labels/ for validation ("" for flat layout)')
parser.add_argument('--test_split', default="", type=str,
                    help='Subfolder for a final test evaluation on the best model ("" = no test)')

# Two-stage fine-tuning
parser.add_argument('--freeze_epochs', default=5, type=int,
                    help='Stage 1 length: epochs training only the new landmark heads')
parser.add_argument('--stage1_lr', default=1e-3, type=float,
                    help='LR for the new heads while the rest of the net is frozen')
parser.add_argument('--stage2_lr', default=1e-4, type=float,
                    help='LR for heads/extras after unfreezing')
parser.add_argument('--backbone_lr_factor', default=0.1, type=float,
                    help='Backbone LR = stage2_lr * this factor during stage 2')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')

# Loss
parser.add_argument('--landm_weight', default=1.0, type=float,
                    help='w_lm: weight of the landmark loss in the total loss')
parser.add_argument('--landm_loss', default="wing", type=str,
                    help='Landmark loss type: "wing" (default) or "smooth_l1"')
parser.add_argument('--pck_threshold', default=0.1, type=float,
                    help='landmark accuracy (PCK): point correct if error < thr * face size')

# Checkpoints
parser.add_argument('--pretrained_ssd', default='models/pretrained/version-RFB-640.pth',
                    help='Pretrained detection-only checkpoint, loaded with strict=False')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint (with the landmark head) to resume training from')

# Scheduler (stage 2 only; stage 1 runs at a constant stage1_lr)
parser.add_argument('--scheduler', default="cosine", type=str,
                    help="Stage-2 scheduler: cosine | multi-step | plateau (reduce LR when val loss stops improving)")
parser.add_argument('--plateau_factor', default=0.5, type=float,
                    help='plateau: multiply LR by this when stuck')
parser.add_argument('--plateau_patience', default=5, type=int,
                    help='plateau: epochs without val-loss improvement before reducing LR')
parser.add_argument('--plateau_min_lr', default=1e-6, type=float, help='plateau: LR floor')

# In-train adaptation
parser.add_argument('--warmup_epochs', default=1.0, type=float,
                    help='linear LR warmup at stage-2 start, in epochs (0 = off)')
parser.add_argument('--clip_grad_norm', default=10.0, type=float,
                    help='max gradient norm (0 = off)')
parser.add_argument('--landm_guard_off', action='store_true',
                    help='disable auto landm_weight cut when detection regresses')
parser.add_argument('--landm_guard_margin', default=1.1, type=float,
                    help='guard trigger: val reg/cls > baseline * margin')
parser.add_argument('--landm_guard_factor', default=0.7, type=float,
                    help='guard: multiply landm_weight by this when triggered')
parser.add_argument('--landm_guard_min', default=0.1, type=float,
                    help='guard: landm_weight floor')
parser.add_argument('--milestones', default="30,45", type=str,
                    help="milestones (counted from the start of stage 2) for MultiStepLR")
parser.add_argument('--t_max', default=-1, type=float,
                    help='T_max for CosineAnnealingLR; -1 = remaining epochs after stage 1')

# Train params
parser.add_argument('--batch_size', default=24, type=int, help='Batch size for training')
parser.add_argument('--num_epochs', default=50, type=int,
                    help='total number of epochs (stage 1 + stage 2)')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=1, type=int,
                    help='validate (and checkpoint) every this many epochs')
parser.add_argument('--use_cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--cuda_index', default="0", type=str, help='CUDA device index')
parser.add_argument('--checkpoint_folder', default='models/train-landmark/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--overlap_threshold', default=0.35, type=float,
                    help='min IoU for prior<->GT matching')
parser.add_argument('--optimizer_type', default="SGD", type=str, help='SGD or Adam')

# COCO detection mAP (expensive; off by default)
parser.add_argument('--map_period', default=0, type=int,
                    help='compute COCO mAP@[.5:.95] every N epochs (0 = never; also runs on last epoch)')
parser.add_argument('--map_conf', default=0.02, type=float, help='score threshold for mAP detections')
parser.add_argument('--map_nms_iou', default=0.5, type=float, help='NMS IoU threshold for mAP')
parser.add_argument('--map_max_det', default=200, type=int, help='max detections per image for mAP')

# Weights & Biases (optional; see vision/wandb_logger.py)
parser.add_argument('--wandb', action='store_true', help='log this run to Weights & Biases')
parser.add_argument('--wandb_project', default="rfb640-landmark", type=str, help='W&B project')
parser.add_argument('--wandb_entity', default=None, type=str, help='W&B team/user')
parser.add_argument('--wandb_run_name', default=None, type=str,
                    help='W&B run name (default: --checkpoint_folder name)')
parser.add_argument('--wandb_tags', nargs='*', default=None, help='W&B tags')
parser.add_argument('--wandb_notes', default="", type=str, help='W&B run notes')
parser.add_argument('--wandb_mode', default="online", type=str,
                    help='online | offline (sync later with `wandb sync`) | disabled')
parser.add_argument('--wandb_steps', action='store_true',
                    help='also log per-step curves to W&B (default: epoch-only monitoring)')
parser.add_argument('--wandb_no_model', action='store_true',
                    help='do not upload best.pth as a W&B artifact')
parser.add_argument('--require_gpu', action='store_true',
                    help='fail loudly if no CUDA GPU (never silently train on CPU); '
                         'auto-on when running on Colab')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    args = parser.parse_args()

    import torch

    from vision.trainer import RFBLandmark, TrainConfig, _on_colab

    device = f"cuda:{args.cuda_index.split(',')[0].strip()}" \
        if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    # require a GPU when explicitly asked, or on Colab with --use_cuda on
    require_cuda = args.use_cuda and (args.require_gpu or _on_colab())
    if device.startswith("cuda"):
        logging.info("Use Cuda.")

    checkpoint_folder = args.checkpoint_folder.rstrip("/\\")
    project, name = os.path.split(checkpoint_folder)

    cfg = TrainConfig(
        data=args.datasets,
        val_data=args.validation_dataset,
        test_data=args.test_dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        epochs=args.num_epochs,
        freeze_epochs=args.freeze_epochs,
        batch=args.batch_size,
        workers=args.num_workers,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        backbone_lr_factor=args.backbone_lr_factor,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer_type,
        scheduler=args.scheduler,
        milestones=args.milestones,
        t_max=args.t_max,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_min_lr=args.plateau_min_lr,
        warmup_epochs=args.warmup_epochs,
        clip_grad_norm=args.clip_grad_norm,
        landm_guard=not args.landm_guard_off,
        landm_guard_margin=args.landm_guard_margin,
        landm_guard_factor=args.landm_guard_factor,
        landm_guard_min=args.landm_guard_min,
        landm_weight=args.landm_weight,
        landm_loss=args.landm_loss,
        pck_threshold=args.pck_threshold,
        overlap_threshold=args.overlap_threshold,
        project=project or ".",
        name=name or "train-landmark",
        exist_ok=True,  # CLI keeps writing into the folder the user named
        val_period=args.validation_epochs,
        map_period=args.map_period,
        map_conf=args.map_conf,
        map_nms_iou=args.map_nms_iou,
        map_max_det=args.map_max_det,
        resume=args.resume,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        wandb_log_steps=args.wandb_steps,
        wandb_save_model=not args.wandb_no_model,
    )

    model = RFBLandmark(weights=args.resume or args.pretrained_ssd, device=device,
                        require_cuda=require_cuda)
    results = model.train(cfg)
    logging.info(f"Done. Best NME {results.best_nme:.4f} @ epoch {results.best_epoch} "
                 f"-> {results.best_checkpoint}")
