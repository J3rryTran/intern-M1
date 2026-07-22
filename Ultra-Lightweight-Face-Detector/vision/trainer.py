import csv
import itertools
import logging
import os
import time
from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader

from vision.ssd.config.fd_config import define_img_size

define_img_size()  # hardcoded 320x320; idempotent, must run before priors are read

from vision.datasets.yolo_pose_dataset import YoloPoseDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing import TestTransform, TrainAugmentation
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.ssd import MatchPrior
from vision.utils import box_utils
from vision.utils.misc import Timer, freeze_net_layers, print_model_summary, store_labels
from vision.wandb_logger import create_logger

DEFAULT_WEIGHTS = "models/pretrained/version-RFB-640.pth"
# key prefix of the new head (allowed to be missing in the pretrained file)
NEW_HEAD_PREFIXES = ("landmark_headers",)


@dataclass
class TrainConfig:
    """Every hyper-parameter of a fine-tune run, with the CLI defaults."""

    # data
    data: Union[str, List[str]] = None
    val_data: Optional[str] = None          # defaults to `data` (first root)
    test_data: Optional[str] = None         # defaults to `data` (first root)
    train_split: str = "train"              # "" for a flat images/ layout
    val_split: str = "val"
    test_split: str = ""                    # "" disables the final test evaluation

    # schedule
    epochs: int = 50
    freeze_epochs: int = 5                  # stage 1: only the landmark head trains
    batch: int = 24
    workers: int = 4

    # learning rates
    stage1_lr: float = 1e-3
    stage2_lr: float = 1e-4
    backbone_lr_factor: float = 0.1         # backbone LR = stage2_lr * this
    momentum: float = 0.9
    weight_decay: float = 5e-4
    optimizer: str = "SGD"                  # SGD | Adam
    scheduler: str = "cosine"               # cosine | multi-step | plateau (stage 2 only)
    milestones: str = "30,45"
    t_max: float = -1                       # -1 -> epochs remaining after stage 1
    plateau_factor: float = 0.5             # plateau: LR *= factor when stuck
    plateau_patience: int = 5               # plateau: epochs without val-loss improvement
    plateau_min_lr: float = 1e-6

    # loss
    loc_weight: float = 2.0
    landm_weight: float = 1.0
    landm_loss: str = "wing"                # wing | smooth_l1
    wing_w: float = 10.0
    wing_epsilon: float = 2.0
    neg_pos_ratio: int = 3
    overlap_threshold: float = 0.35         # IoU for prior <-> GT matching
    pck_threshold: float = 0.1              # landmark "correct" if error < thr * face size

    # in-train adaptation
    warmup_epochs: float = 1.0              # linear LR ramp at stage-2 start (0 = off)
    clip_grad_norm: float = 10.0            # max gradient norm (0 = off)
    landm_guard: bool = True                # detection regresses -> auto-cut landm_weight
    landm_guard_margin: float = 1.1         # trigger: val reg/cls > baseline * margin
    landm_guard_factor: float = 0.7
    landm_guard_min: float = 0.1

    # run bookkeeping
    project: str = "models"
    name: str = "train-landmark"
    exist_ok: bool = False                  # False -> auto-suffix name2, name3...
    val_period: int = 1
    verbose: bool = True

    # COCO-style detection mAP (expensive: decode + NMS + box matching per image)
    map_period: int = 0                     # 0 = never; N = compute mAP every N epochs (+ last)
    map_conf: float = 0.02                  # score threshold for candidate detections
    map_nms_iou: float = 0.5                # NMS IoU threshold
    map_max_det: int = 200                  # max detections kept per image

    # Weights & Biases (optional; see vision/wandb_logger.py)
    wandb: bool = False                     # master switch
    wandb_project: str = "rfb640-landmark"
    wandb_entity: Optional[str] = None      # team/user; None = your default
    wandb_run_name: Optional[str] = None    # None -> falls back to `name`
    wandb_tags: Optional[List[str]] = None
    wandb_notes: str = ""
    wandb_mode: str = "online"              # online | offline | disabled
    wandb_log_steps: bool = False           # per-step curves (default: epoch-only monitoring)
    wandb_save_model: bool = True           # upload best.pth as an artifact
    seed: Optional[int] = 0
    resume: Optional[str] = None            # checkpoint WITH the landmark head


@dataclass
class TrainResults:
    """What a run produced. `history` has one dict per validated epoch."""

    save_dir: str = ""
    baseline: dict = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)
    test: dict = field(default_factory=dict)   # final test-set metrics (if test_split set)
    best_nme: float = float("nan")
    best_epoch: int = -1
    best_checkpoint: str = ""
    last_checkpoint: str = ""
    config: dict = field(default_factory=dict)

    @property
    def final(self) -> dict:
        return self.history[-1] if self.history else {}

    steps_csv: str = ""
    epochs_csv: str = ""
    wandb_url: str = ""

    @property
    def detection_kept(self) -> bool:
        """True when the final detection losses did not regress >10% vs the
        pretrained baseline (the core constraint of this fine-tune)."""
        if not self.history or not self.baseline:
            return False
        last = self.history[-1]
        return (last["reg"] <= self.baseline["reg"] * 1.1
                and last["cls"] <= self.baseline["cls"] * 1.1)


class _CsvLogger:
    """Append rows to a CSV, header written once. Missing keys become ""."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.f = open(path, "w", newline="", encoding="utf-8")
        # restval="" fills missing columns; extrasaction="ignore" drops any extra
        # keys so callers can pass a superset without maintaining the header
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames, restval="",
                                extrasaction="ignore")
        self.w.writeheader()

    def write(self, row: dict):
        self.w.writerow(row)

    def flush(self):
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def _resolve_save_dir(project: str, name: str, exist_ok: bool) -> str:
    path = os.path.join(project, name)
    if not exist_ok:
        i = 2
        while os.path.exists(path):
            path = os.path.join(project, f"{name}{i}")
            i += 1
    os.makedirs(path, exist_ok=True)
    return path


POINT_NAMES = ("eyeL", "eyeR", "nose", "mouthL", "mouthR")  # fixed landmark order


def landmark_stats(pred_landms, gt_landms, gt_locations, labels, gt_landm_mask, priors,
                   center_variance, size_variance, pck_thr=0.1):
    """NME + per-point PCK on (positive prior) ∩ (labelled point).
    Error normalized by sqrt(box area); a point is "correct" if error < pck_thr.
    Returns (nme_sum, n_points, pck_correct[5], n_per_point[5])."""
    valid = (labels > 0).unsqueeze(-1) & (gt_landm_mask > 0)
    if int(valid.sum().item()) == 0:
        return 0.0, 0, [0] * 5, [0] * 5
    batch, num_priors = labels.size(0), labels.size(1)
    pred_pts = box_utils.decode_landm(pred_landms, priors, center_variance).reshape(batch, num_priors, 5, 2)
    gt_pts = box_utils.decode_landm(gt_landms, priors, center_variance).reshape(batch, num_priors, 5, 2)
    boxes = box_utils.convert_locations_to_boxes(gt_locations, priors, center_variance, size_variance)
    scale = torch.sqrt((boxes[..., 2] * boxes[..., 3]).clamp(min=1e-8)).unsqueeze(-1)
    nme = ((pred_pts - gt_pts) ** 2).sum(dim=-1).sqrt() / scale   # [B, P, 5]
    correct = ((nme < pck_thr) & valid).sum(dim=(0, 1))
    per_point = valid.sum(dim=(0, 1))
    return (float(nme[valid].sum().item()), int(valid.sum().item()),
            correct.tolist(), per_point.tolist())


def _map_collate(batch):
    """Collate for the mAP loader: stack images, keep raw GT boxes as a list
    (variable box count per image can't be stacked). Dataset without a
    target_transform yields (image, boxes, labels, landms, mask); we keep the
    image tensor and the percent-coord GT boxes."""
    images = torch.stack([b[0] for b in batch])
    gt_boxes = [torch.as_tensor(b[1], dtype=torch.float32) for b in batch]
    return images, gt_boxes


def prior_metrics(tp, fp, fn, tn):
    """Classification metrics from prior-level confusion counts (positive=face).

    Returns a dict: acc (overall prior accuracy), pos_precision/pos_recall/pos_f1
    (the face class, the meaningful ones) and neg_precision/neg_recall.
    """
    def _r(a, b):
        return a / b if b else float("nan")
    pos_p, pos_r = _r(tp, tp + fp), _r(tp, tp + fn)
    f1 = _r(2 * pos_p * pos_r, pos_p + pos_r) \
        if (pos_p == pos_p and pos_r == pos_r and (pos_p + pos_r) > 0) else float("nan")
    return {"acc": _r(tp + tn, tp + tn + fp + fn),
            "pos_precision": pos_p, "pos_recall": pos_r, "pos_f1": f1,
            "neg_precision": _r(tn, tn + fn), "neg_recall": _r(tn, tn + fp)}


def average_precision(preds, total_gt):
    """VOC-2010/COCO all-point AP for a single class.

    Args:
        preds: list of (score, is_tp) across all images, one entry per detection.
        total_gt: total number of ground-truth boxes (recall denominator).
    Returns:
        AP in [0, 1]; nan if there are no GT boxes.
    """
    if total_gt == 0:
        return float("nan")
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: x[0], reverse=True)
    tp = fp = 0
    rec, prec = [], []
    for _, is_tp in preds:
        if is_tp:
            tp += 1
        else:
            fp += 1
        rec.append(tp / total_gt)
        prec.append(tp / (tp + fp))
    # precision envelope (monotonic non-increasing from the right), then integrate
    mrec = [0.0] + rec + [rec[-1]]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def _on_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


class RFBLandmark:
    """RFB-640 face detector + 5-point landmark head, with a YOLO-like API."""

    def __init__(self, weights: Optional[str] = DEFAULT_WEIGHTS, device=None,
                 num_classes: int = 2, require_cuda=None):
        """
        Args:
            weights: detection-only checkpoint (loaded with strict=False, the
                landmark head keeps its random init), a checkpoint that already
                has the landmark head, or None to start from scratch.
            device: "cuda:0" / "cpu" / torch.device. Default: CUDA when available.
            num_classes: 2 (BACKGROUND + face).
            require_cuda: True -> raise if no GPU (never silently trains on CPU);
                False -> allow CPU; None (default) -> require a GPU when running
                on Colab, allow CPU elsewhere.
        """
        cuda_ok = torch.cuda.is_available()
        dev = torch.device(device) if device is not None else torch.device(
            "cuda:0" if cuda_ok else "cpu")
        if require_cuda is None:
            # auto: on Colab, require a GPU unless the caller explicitly asked for CPU
            asked_cpu = device is not None and torch.device(device).type == "cpu"
            require_cuda = _on_colab() and not asked_cpu
        if require_cuda and not cuda_ok:
            raise RuntimeError(
                f"GPU required (Colab or require_cuda=True) but no CUDA device is "
                f"available (torch {torch.__version__}). Enable a GPU runtime, or "
                f"pass require_cuda=False / device='cpu' to allow CPU.")
        if dev.type == "cuda" and not cuda_ok:
            raise RuntimeError(
                f"Requested a CUDA device but torch.cuda.is_available() is False "
                f"(torch {torch.__version__}). A '+cpu' build has no CUDA - reinstall "
                f"a +cu wheel, or pass device='cpu'.")
        self.device = dev
        if dev.type == "cuda":
            logging.info(f"Using GPU: {torch.cuda.get_device_name(dev)}")
        else:
            logging.warning(
                f"Training on CPU (torch {torch.__version__}). Expected a GPU? A '+cpu' "
                f"torch build has no CUDA; reinstall a +cu wheel.")
        self.num_classes = num_classes
        self.weights = weights
        self.net = create_Mb_Tiny_RFB_fd(num_classes, device=self.device)
        if weights:
            self.load(weights)
        self.net.to(self.device)

    def load(self, weights: str):
        """Load a checkpoint, verifying that only the new head is missing."""
        missing, unexpected = self.net.init_from_pretrained_detector(weights)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint {weights}: {unexpected}")
        not_new = [k for k in missing if not k.startswith(NEW_HEAD_PREFIXES)]
        if not_new:
            raise RuntimeError(f"Checkpoint {weights} is missing non-head keys: {not_new}")
        logging.info(f"Loaded {weights}: all old keys matched; "
                     f"{len(missing)} landmark head keys keep their random init.")
        return self

    def _detection_modules(self):
        """Modules that already existed in the pretrained detector."""
        n = self.net
        return [n.base_net, n.source_layer_add_ons, n.extras,
                n.classification_headers, n.regression_headers]

    def _eval_loaders(self, cfg, root, split, with_map, skip_missing=False):
        """(loss/acc loader, mAP loader) for one eval split. split=None -> flat
        layout. skip_missing: return (None, None) instead of raising (optional
        test set). The mAP loader needs raw GT boxes -> no target_transform."""
        img_dir = os.path.join(root, "images", split) if split else os.path.join(root, "images")
        if skip_missing and not os.path.isdir(img_dir):
            return None, None
        test_transform = TestTransform(fd_config.image_size, fd_config.image_mean_test, fd_config.image_std)
        target_transform = MatchPrior(fd_config.priors, fd_config.center_variance,
                                      fd_config.size_variance, cfg.overlap_threshold)
        ds = YoloPoseDataset(root, transform=test_transform,
                             target_transform=target_transform, split=split)
        loader = DataLoader(ds, cfg.batch, num_workers=cfg.workers, shuffle=False)
        map_loader = None
        if with_map:
            map_ds = YoloPoseDataset(root, transform=test_transform,
                                     target_transform=None, split=split)
            map_loader = DataLoader(map_ds, cfg.batch, num_workers=cfg.workers,
                                    shuffle=False, collate_fn=_map_collate)
        return loader, map_loader

    def _build_loaders(self, cfg: TrainConfig):
        train_transform = TrainAugmentation(fd_config.image_size, fd_config.image_mean, fd_config.image_std)

        roots = [cfg.data] if isinstance(cfg.data, str) else list(cfg.data)
        target_transform = MatchPrior(fd_config.priors, fd_config.center_variance,
                                      fd_config.size_variance, cfg.overlap_threshold)
        datasets = [YoloPoseDataset(r, transform=train_transform,
                                    target_transform=target_transform,
                                    split=cfg.train_split or None) for r in roots]
        train_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        train_loader = DataLoader(train_ds, cfg.batch, num_workers=cfg.workers, shuffle=True)

        val_loader, val_map = self._eval_loaders(
            cfg, cfg.val_data or roots[0], cfg.val_split or None, cfg.map_period > 0)
        test_loader, test_map = (None, None)
        if cfg.test_split:  # "" = no test evaluation
            test_loader, test_map = self._eval_loaders(
                cfg, cfg.test_data or roots[0], cfg.test_split,
                cfg.map_period > 0, skip_missing=True)
        return (train_loader, val_loader, val_map, test_loader, test_map,
                datasets[0].class_names)

    def _build_optimizer(self, params, lr, cfg: TrainConfig):
        if cfg.optimizer == "SGD":
            return torch.optim.SGD(params, lr=lr, momentum=cfg.momentum,
                                   weight_decay=cfg.weight_decay)
        if cfg.optimizer == "Adam":
            return torch.optim.Adam(params, lr=lr)
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    def _build_stage2(self, cfg: TrainConfig):
        """Unfrozen phase: heads at stage2_lr, backbone ~10x lower."""
        for p in self.net.parameters():
            p.requires_grad = True
        head_params = itertools.chain(
            self.net.source_layer_add_ons.parameters(), self.net.extras.parameters(),
            self.net.classification_headers.parameters(), self.net.regression_headers.parameters(),
            self.net.landmark_headers.parameters())
        params = [
            {'params': self.net.base_net.parameters(), 'lr': cfg.stage2_lr * cfg.backbone_lr_factor},
            {'params': head_params},
        ]
        optimizer = self._build_optimizer(params, cfg.stage2_lr, cfg)
        if cfg.scheduler == "multi-step":
            milestones = [int(v.strip()) for v in cfg.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        elif cfg.scheduler == "cosine":
            t_max = cfg.t_max if cfg.t_max > 0 else max(cfg.epochs - cfg.freeze_epochs, 1)
            scheduler = CosineAnnealingLR(optimizer, t_max)
        elif cfg.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=cfg.plateau_factor,
                                          patience=cfg.plateau_patience, min_lr=cfg.plateau_min_lr)
        else:
            raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")
        if cfg.verbose:
            logging.info(f"Stage 2: full network unfrozen. Head LR {cfg.stage2_lr}, "
                         f"backbone LR {cfg.stage2_lr * cfg.backbone_lr_factor}, "
                         f"{cfg.scheduler} scheduler.")
        return optimizer, scheduler

    def _train_epoch(self, loader, criterion, optimizer, cfg, epoch, frozen_modules,
                     step_logger=None, wb=None, steps_per_epoch=0, warmup=None):
        """One training epoch (no console output; per-step metrics go to the
        step CSV). warmup = (start_epoch, total_steps, base_lrs) or None.
        Returns (epoch-average train metrics, seconds)."""
        self.net.train(True)
        if frozen_modules:
            # stage 1: keep BatchNorm running stats of the frozen modules untouched,
            # otherwise detection would drift even with frozen weights
            for m in frozen_modules:
                m.eval()
        stage = 1 if frozen_modules else 2
        totals = {"loss": 0.0, "reg": 0.0, "cls": 0.0, "landm": 0.0}
        tp = fp = fn = tn = 0  # prior-level confusion, for train accuracy
        pck_c, pck_n = [0] * 5, [0] * 5
        priors = fd_config.priors.to(self.device)
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        steps = 0
        start = time.perf_counter()
        for i, (images, boxes, labels, landms, landm_mask) in enumerate(loader):
            images = images.to(self.device)
            boxes = boxes.to(self.device)
            labels = labels.to(self.device)
            landms = landms.to(self.device)
            landm_mask = landm_mask.to(self.device)

            if warmup is not None:
                # linear per-step LR ramp over the first warmup steps of stage 2
                ws, wt, base_lrs = warmup
                done = (epoch - ws) * steps_per_epoch + i
                if 0 <= done < wt:
                    frac = (done + 1) / wt
                    for g, b in zip(optimizer.param_groups, base_lrs):
                        g["lr"] = b * frac

            optimizer.zero_grad()
            confidence, locations, pred_landms = self.net(images)
            loss, reg, cls, landm = criterion(
                confidence, locations, labels, boxes,
                predicted_landmarks=pred_landms, gt_landmarks=landms,
                gt_landm_mask=landm_mask)
            loss.backward()
            if cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable, cfg.clip_grad_norm)
            optimizer.step()

            with torch.no_grad():
                pred_pos = confidence.argmax(dim=2) > 0
                gt_pos = labels > 0
                tp += int((pred_pos & gt_pos).sum().item())
                fp += int((pred_pos & ~gt_pos).sum().item())
                fn += int((~pred_pos & gt_pos).sum().item())
                tn += int((~pred_pos & ~gt_pos).sum().item())
                _, _, ck, nk = landmark_stats(pred_landms, landms, boxes, labels, landm_mask,
                                              priors, fd_config.center_variance,
                                              fd_config.size_variance, cfg.pck_threshold)
                for k in range(5):
                    pck_c[k] += ck[k]
                    pck_n[k] += nk[k]

            row = {"loss": loss.item(), "reg": reg.item(),
                   "cls": cls.item(), "landm": landm.item()}
            for k, v in row.items():
                totals[k] += v
            steps += 1
            lr = optimizer.param_groups[0]["lr"]
            if step_logger is not None:
                step_logger.write({"epoch": epoch, "step": i, "stage": stage, "lr": lr,
                                   **{k: round(v, 6) for k, v in row.items()}})
            if wb is not None and wb.enabled:
                wb.log_step({**row, "lr": lr}, global_step=epoch * steps_per_epoch + i)
        if step_logger is not None:
            step_logger.flush()  # one flush per epoch: crash loses at most 1 epoch of rows
        elapsed = time.perf_counter() - start
        metrics = {k: v / max(steps, 1) for k, v in totals.items()}
        metrics.update(prior_metrics(tp, fp, fn, tn))
        total_c, total_n = sum(pck_c), sum(pck_n)
        metrics["landm_acc"] = total_c / total_n if total_n else float("nan")
        return metrics, elapsed

    @torch.no_grad()
    def validate(self, loader, criterion, pck_thr=0.1) -> dict:
        """Loss components + face P/R (prior-level) + landmark NME + landmark
        accuracy (PCK@pck_thr, overall and per point)."""
        self.net.eval()
        run = {"loss": 0.0, "reg": 0.0, "cls": 0.0, "landm": 0.0}
        nme_sum, nme_count, num = 0.0, 0, 0
        tp = fp = fn = tn = 0
        pck_c, pck_n = [0] * 5, [0] * 5
        priors = fd_config.priors.to(self.device)
        for images, boxes, labels, landms, landm_mask in loader:
            images = images.to(self.device)
            boxes = boxes.to(self.device)
            labels = labels.to(self.device)
            landms = landms.to(self.device)
            landm_mask = landm_mask.to(self.device)
            num += 1

            confidence, locations, pred_landms = self.net(images)
            loss, reg, cls, landm = criterion(
                confidence, locations, labels, boxes,
                predicted_landmarks=pred_landms, gt_landmarks=landms,
                gt_landm_mask=landm_mask)
            s, c, ck, nk = landmark_stats(pred_landms, landms, boxes, labels, landm_mask,
                                          priors, fd_config.center_variance,
                                          fd_config.size_variance, pck_thr)

            pred_pos = confidence.argmax(dim=2) > 0
            gt_pos = labels > 0
            tp += int((pred_pos & gt_pos).sum().item())
            fp += int((pred_pos & ~gt_pos).sum().item())
            fn += int((~pred_pos & gt_pos).sum().item())
            tn += int((~pred_pos & ~gt_pos).sum().item())

            run["loss"] += loss.item()
            run["reg"] += reg.item()
            run["cls"] += cls.item()
            run["landm"] += landm.item()
            nme_sum += s
            nme_count += c
            for k in range(5):
                pck_c[k] += ck[k]
                pck_n[k] += nk[k]
        metrics = {k: v / max(num, 1) for k, v in run.items()}
        metrics["nme"] = nme_sum / nme_count if nme_count else float("nan")
        metrics.update(prior_metrics(tp, fp, fn, tn))
        total_c, total_n = sum(pck_c), sum(pck_n)
        metrics["landm_acc"] = total_c / total_n if total_n else float("nan")
        metrics["landm_acc_points"] = [c / n if n else float("nan")
                                       for c, n in zip(pck_c, pck_n)]
        return metrics

    @torch.no_grad()
    def compute_map(self, map_loader, conf_threshold=0.02, nms_iou=0.5, max_det=200):
        """COCO-style detection mAP over the val set (single class = face).

        The real detector metric: decode the box head, threshold by score, run
        NMS, then match kept boxes to GT boxes at IoU thresholds 0.50:0.05:0.95.
        Args:
            map_loader: loader yielding (images, list_of_gt_boxes) - GT boxes in
                percent corner form (from _map_collate).
        Returns:
            {"map50", "map75", "map"}: AP@0.5, AP@0.75, and the mean AP over the
            10 thresholds (the headline COCO number).
        """
        self.net.eval()
        priors = fd_config.priors.to(self.device)
        cv, sv = fd_config.center_variance, fd_config.size_variance
        thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95
        preds_per_t = {t: [] for t in thresholds}   # each: list of (score, is_tp)
        total_gt = 0

        for images, gt_list in map_loader:
            images = images.to(self.device)
            confidence, locations, _ = self.net(images)
            scores = torch.softmax(confidence, dim=2)[:, :, 1]  # face prob [B, P]
            boxes = box_utils.convert_locations_to_boxes(locations, priors, cv, sv)
            boxes = box_utils.center_form_to_corner_form(boxes)  # [B, P, 4] percent

            for i in range(images.size(0)):
                gt = gt_list[i].to(self.device)  # [K, 4] percent corner
                total_gt += gt.size(0)

                keep = scores[i] > conf_threshold
                if keep.sum() == 0:
                    continue
                bs = torch.cat([boxes[i][keep], scores[i][keep].unsqueeze(1)], dim=1)
                bs = box_utils.hard_nms(bs, nms_iou, top_k=max_det)  # [D, 5], score-sorted
                det_boxes, det_scores = bs[:, :4], bs[:, 4]

                if gt.size(0) == 0:  # all detections are false positives
                    for t in thresholds:
                        preds_per_t[t].extend((float(x), 0) for x in det_scores)
                    continue

                ious = box_utils.iou_of(gt.unsqueeze(0), det_boxes.unsqueeze(1))  # [D, K]
                # detections already score-sorted by hard_nms; greedy match per threshold
                for t in thresholds:
                    taken = set()
                    for d in range(det_boxes.size(0)):
                        best_iou, best_j = ious[d].max(0)
                        j = int(best_j.item())
                        if best_iou.item() >= t and j not in taken:
                            taken.add(j)
                            preds_per_t[t].append((det_scores[d].item(), 1))
                        else:
                            preds_per_t[t].append((det_scores[d].item(), 0))

        aps = {t: average_precision(preds_per_t[t], total_gt) for t in thresholds}
        valid = [v for v in aps.values() if v == v]  # drop nan (no GT)
        return {"map50": aps[0.5], "map75": aps[0.75],
                "map": sum(valid) / len(valid) if valid else float("nan")}

    def train(self, cfg: Optional[TrainConfig] = None, **overrides) -> TrainResults:
        """Run a two-stage fine-tune. Returns a TrainResults.

        Args:
            cfg: a TrainConfig; omit to build one from the defaults.
            **overrides: any TrainConfig field (data=, epochs=, batch=,
                landm_weight=, ...). Unknown names raise TypeError.
        """
        cfg = TrainConfig() if cfg is None else cfg
        valid = {f.name for f in fields(TrainConfig)}
        unknown = set(overrides) - valid
        if unknown:
            raise TypeError(f"Unknown train() argument(s): {sorted(unknown)}. "
                            f"Valid: {sorted(valid)}")
        for k, v in overrides.items():
            setattr(cfg, k, v)

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        save_dir = _resolve_save_dir(cfg.project, cfg.name, cfg.exist_ok)
        results = TrainResults(save_dir=save_dir, config=asdict(cfg))
        if cfg.verbose:
            logging.info(f"Run dir: {save_dir}  |  device: {self.device}")
            logging.info(cfg)

        if cfg.resume:
            self.load(cfg.resume)
        self.net.to(self.device)

        (train_loader, val_loader, map_loader,
         test_loader, test_map_loader, class_names) = self._build_loaders(cfg)
        store_labels(os.path.join(save_dir, "labels.txt"), class_names)
        if cfg.verbose:
            logging.info(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
            print_model_summary(self.net,
                                input_size=(3, fd_config.image_size[1], fd_config.image_size[0]),
                                device=self.device, model_name="RFB-640-landmark")

        criterion = MultiboxLoss(fd_config.priors, neg_pos_ratio=cfg.neg_pos_ratio,
                                 center_variance=fd_config.center_variance,
                                 size_variance=fd_config.size_variance, device=self.device,
                                 loc_weight=cfg.loc_weight, landm_weight=cfg.landm_weight,
                                 landm_loss_type=cfg.landm_loss,
                                 wing_w=cfg.wing_w, wing_epsilon=cfg.wing_epsilon)

        n_params = sum(p.numel() for p in self.net.parameters())
        wb = create_logger(cfg, save_dir, extra_config={
            "device": str(self.device), "weights": self.weights,
            "parameters": n_params, "priors": fd_config.priors.size(0),
            "image_size": list(fd_config.image_size),
            "train_batches": len(train_loader), "val_batches": len(val_loader)})
        results.wandb_url = wb.run_url

        # Baseline with the pretrained weights: detection must not fall below this.
        results.baseline = self.validate(val_loader, criterion, cfg.pck_threshold)
        if cfg.map_period > 0:
            results.baseline.update(self.compute_map(
                map_loader, cfg.map_conf, cfg.map_nms_iou, cfg.map_max_det))
        wb.log_baseline(results.baseline)
        if cfg.verbose:
            logging.info(f"Baseline (pretrained, random landmark head): "
                         f"Regression Loss {results.baseline['reg']:.4f}, "
                         f"Classification Loss {results.baseline['cls']:.4f}"
                         + (f", mAP@.5 {results.baseline['map50']:.4f}, "
                            f"mAP@.5:.95 {results.baseline['map']:.4f}"
                            if cfg.map_period > 0 else ""))

        # metrics on disk: one row per step / per epoch (two separate CSVs)
        step_logger = _CsvLogger(
            os.path.join(save_dir, "train_steps.csv"),
            ["epoch", "step", "stage", "lr", "loss", "reg", "cls", "landm"])
        epoch_logger = _CsvLogger(
            os.path.join(save_dir, "train_epochs.csv"),
            ["epoch", "stage", "lr", "landm_weight", "train_time_s",
             "train_loss", "train_reg", "train_cls", "train_landm",
             "train_acc", "train_landm_acc",
             "train_pos_precision", "train_pos_recall", "train_pos_f1",
             "val_time_s", "val_loss", "val_reg", "val_cls", "val_landm", "val_nme",
             "val_acc", "val_landm_acc",
             *[f"val_landm_acc_{n}" for n in POINT_NAMES],
             "val_pos_precision", "val_pos_recall", "val_pos_f1",
             "val_neg_precision", "val_neg_recall",
             "val_map50", "val_map75", "val_map"])
        results.steps_csv = step_logger.path
        results.epochs_csv = epoch_logger.path

        frozen_modules, scheduler, warmup = None, None, None

        def _make_warmup(start_epoch, opt):
            if cfg.warmup_epochs <= 0:
                return None
            return (start_epoch, max(1, int(cfg.warmup_epochs * len(train_loader))),
                    [g["lr"] for g in opt.param_groups])

        if cfg.freeze_epochs > 0:
            for m in self._detection_modules():
                freeze_net_layers(m)
            frozen_modules = self._detection_modules()
            optimizer = self._build_optimizer(self.net.landmark_headers.parameters(),
                                              cfg.stage1_lr, cfg)
            if cfg.verbose:
                logging.info(f"Stage 1: training ONLY the landmark heads for "
                             f"{cfg.freeze_epochs} epochs at LR {cfg.stage1_lr} "
                             f"(frozen modules kept in eval mode).")
        else:
            optimizer, scheduler = self._build_stage2(cfg)
            warmup = _make_warmup(0, optimizer)

        try:
            for epoch in range(cfg.epochs):
                if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs:
                    frozen_modules = None
                    optimizer, scheduler = self._build_stage2(cfg)
                    warmup = _make_warmup(epoch, optimizer)
                stage = 1 if frozen_modules else 2

                train_metrics, train_time = self._train_epoch(
                    train_loader, criterion, optimizer, cfg, epoch, frozen_modules,
                    step_logger=step_logger, wb=wb, steps_per_epoch=len(train_loader),
                    warmup=warmup)
                # plateau steps after validation (needs the val metric)
                if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step()
                if cfg.verbose:
                    logging.info(
                        f"Epoch: {epoch} (stage {stage}) - {train_time:.1f}s - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Regression {train_metrics['reg']:.4f}, "
                        f"Classification {train_metrics['cls']:.4f}, "
                        f"Landmark {train_metrics['landm']:.4f}, "
                        f"Train Acc {train_metrics['acc']:.4f}, "
                        f"Landmark Acc {train_metrics['landm_acc']:.4f}, "
                        f"Face Recall {train_metrics['pos_recall']:.4f} F1 {train_metrics['pos_f1']:.4f}, "
                        f"lr {optimizer.param_groups[0]['lr']:.2e}")
                epoch_row = {"epoch": epoch, "stage": stage,
                             "lr": optimizer.param_groups[0]["lr"],
                             "landm_weight": round(criterion.landm_weight, 4),
                             "train_time_s": round(train_time, 1),
                             **{f"train_{k}": round(v, 6) for k, v in train_metrics.items()}}

                if epoch % cfg.val_period == 0 or epoch == cfg.epochs - 1:
                    val_start = time.perf_counter()
                    m = self.validate(val_loader, criterion, cfg.pck_threshold)
                    m["epoch"] = epoch
                    m["lr"] = optimizer.param_groups[0]["lr"]
                    m["stage"] = stage
                    m["train_time"] = round(train_time, 1)
                    m["val_time"] = round(time.perf_counter() - val_start, 1)
                    for k, v in train_metrics.items():
                        m[f"train_{k}"] = v

                    if isinstance(scheduler, ReduceLROnPlateau):
                        prev_lr = optimizer.param_groups[0]["lr"]
                        scheduler.step(m["loss"])
                        new_lr = optimizer.param_groups[0]["lr"]
                        if new_lr < prev_lr and cfg.verbose:
                            logging.info(f"Plateau: val loss stuck -> LR reduced to {new_lr:.2e}")

                    # COCO mAP: expensive, only every map_period epochs (+ last)
                    do_map = cfg.map_period > 0 and (
                        epoch % cfg.map_period == 0 or epoch == cfg.epochs - 1)
                    if do_map:
                        map_start = time.perf_counter()
                        mp = self.compute_map(map_loader, cfg.map_conf,
                                              cfg.map_nms_iou, cfg.map_max_det)
                        m.update(mp)
                        m["map_time"] = round(time.perf_counter() - map_start, 1)

                    results.history.append(m)
                    epoch_row.update({"val_time_s": m["val_time"],
                                      "val_loss": round(m["loss"], 6),
                                      "val_reg": round(m["reg"], 6),
                                      "val_cls": round(m["cls"], 6),
                                      "val_landm": round(m["landm"], 6),
                                      "val_nme": round(m["nme"], 6),
                                      "val_acc": round(m["acc"], 6),
                                      "val_landm_acc": round(m["landm_acc"], 6),
                                      **{f"val_landm_acc_{n}": round(a, 6) for n, a
                                         in zip(POINT_NAMES, m["landm_acc_points"])},
                                      "val_pos_precision": round(m["pos_precision"], 6),
                                      "val_pos_recall": round(m["pos_recall"], 6),
                                      "val_pos_f1": round(m["pos_f1"], 6),
                                      "val_neg_precision": round(m["neg_precision"], 6),
                                      "val_neg_recall": round(m["neg_recall"], 6)})
                    if do_map:
                        epoch_row.update({"val_map50": round(m["map50"], 6),
                                          "val_map75": round(m["map75"], 6),
                                          "val_map": round(m["map"], 6)})
                    if cfg.verbose:
                        logging.info(
                            f"Epoch: {epoch}, Validation ({m['val_time']:.1f}s) Loss: {m['loss']:.4f}, "
                            f"Validation Regression Loss {m['reg']:.4f} (baseline {results.baseline['reg']:.4f}), "
                            f"Validation Classification Loss: {m['cls']:.4f} (baseline {results.baseline['cls']:.4f}), "
                            f"Validation Landmark Loss: {m['landm']:.4f}, "
                            f"Validation Landmark NME: {m['nme']:.4f}")
                        logging.info(
                            f"           Val Acc {m['acc']:.4f}  |  "
                            f"Face - Precision {m['pos_precision']:.4f}, "
                            f"Recall {m['pos_recall']:.4f}, F1 {m['pos_f1']:.4f}")
                        pts = "  ".join(f"{n} {a:.3f}" for n, a
                                        in zip(POINT_NAMES, m["landm_acc_points"]))
                        logging.info(
                            f"           Landmark Acc@{cfg.pck_threshold:g}: "
                            f"{m['landm_acc']:.4f}  |  {pts}")
                        if do_map:
                            base_map = results.baseline.get("map")
                            base_str = f" (baseline {base_map:.4f})" if base_map == base_map else ""
                            logging.info(
                                f"           Detection mAP@.5 {m['map50']:.4f}, "
                                f"mAP@.75 {m['map75']:.4f}, "
                                f"mAP@.5:.95 {m['map']:.4f}{base_str}  ({m['map_time']:.1f}s)")
                    if (m["reg"] > results.baseline["reg"] * cfg.landm_guard_margin
                            or m["cls"] > results.baseline["cls"] * cfg.landm_guard_margin):
                        # detection regressed vs baseline
                        if cfg.landm_guard and criterion.landm_weight > cfg.landm_guard_min:
                            criterion.landm_weight = max(
                                criterion.landm_weight * cfg.landm_guard_factor,
                                cfg.landm_guard_min)
                            logging.warning(f"Guard: detection regressed -> landm_weight "
                                            f"cut to {criterion.landm_weight:.3f}")
                        else:
                            logging.warning("Detection regressed vs baseline - consider "
                                            "lowering landm_weight or raising freeze_epochs.")

                    # YOLO-style checkpoints: last.pth is overwritten on every
                    # validation, best.pth only when val NME improves
                    path = os.path.join(save_dir, "last.pth")
                    self.net.save(path)
                    results.last_checkpoint = path
                    is_best = (m["nme"] == m["nme"]
                               and (results.best_epoch < 0 or m["nme"] < results.best_nme))
                    if is_best:
                        results.best_nme, results.best_epoch = m["nme"], epoch
                        best = os.path.join(save_dir, "best.pth")
                        self.net.save(best)
                        results.best_checkpoint = best
                    if cfg.verbose:
                        logging.info(f"Saved last.pth (epoch {epoch})"
                                     + (f" + best.pth (NME {m['nme']:.4f})" if is_best else ""))

                epoch_logger.write(epoch_row)
                epoch_logger.flush()
                wb.log_epoch(epoch_row, global_step=(epoch + 1) * len(train_loader) - 1)

            # Final test-set evaluation on the BEST checkpoint (if a test split
            # was given and exists). Test data is touched only once, at the end.
            if test_loader is not None:
                if results.best_checkpoint and os.path.isfile(results.best_checkpoint):
                    self.load(results.best_checkpoint)
                    self.net.to(self.device)
                results.test = self.validate(test_loader, criterion, cfg.pck_threshold)
                if cfg.map_period > 0 and test_map_loader is not None:
                    results.test.update(self.compute_map(
                        test_map_loader, cfg.map_conf, cfg.map_nms_iou, cfg.map_max_det))
                wb.log_test(results.test)
                if cfg.verbose:
                    t = results.test
                    map_str = (f", mAP@.5 {t['map50']:.4f}, mAP@.5:.95 {t['map']:.4f}"
                               if "map" in t else "")
                    pts = "  ".join(f"{n} {a:.3f}" for n, a
                                    in zip(POINT_NAMES, t["landm_acc_points"]))
                    logging.info(
                        f"TEST (best model): Loss {t['loss']:.4f}, "
                        f"Test Acc {t['acc']:.4f}, "
                        f"Face Precision {t['pos_precision']:.4f} Recall {t['pos_recall']:.4f} "
                        f"F1 {t['pos_f1']:.4f}, NME {t['nme']:.4f}, "
                        f"Landmark Acc {t['landm_acc']:.4f}{map_str}")
                    logging.info(f"TEST landmark per-point: {pts}")
        finally:
            step_logger.close()
            epoch_logger.close()
            wb.log_summary(results)
            wb.save_model(results.best_checkpoint)
            wb.finish()

        return results
