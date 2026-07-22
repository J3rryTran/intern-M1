"""Optional Weights & Biases logging for the RFB-640 landmark fine-tune.

wandb is a soft dependency: if it is not installed (or login fails) training
keeps running and only a warning is logged - a 3-hour run must never die
because of a metrics service.

Enable it from the Python API:

    from vision.trainer import RFBLandmark
    RFBLandmark().train(data="../data/exp", epochs=50, batch=32,
                        wandb=True, wandb_project="rfb640-landmark",
                        wandb_run_name="baseline")

or from the CLI:

    python train.py --wandb --wandb_project rfb640-landmark --wandb_run_name baseline

Authentication (once per machine):
    pip install wandb
    wandb login                      # or: export WANDB_API_KEY=...
On Colab, `import wandb; wandb.login()` prompts for the key. With no key at
all, pass wandb_mode="offline" to write runs to disk and `wandb sync` later.

What gets logged:
    config          - every TrainConfig field + device/param counts
    train/step_*    - per optimizer step (loss, reg, cls, landm, lr)
    train/*, val/*  - per epoch averages, plus val NME and timings
    baseline/*      - the pretrained detection losses this run must not regress
    summary         - best NME/epoch, whether detection was kept
    artifact        - best.pth (optional, wandb_save_model)
"""
import logging
import os
from typing import Optional


class NullWandbLogger:
    """No-op stand-in so the trainer never needs `if logger is not None`."""

    enabled = False
    run_url = ""

    def log_step(self, row, global_step):
        pass

    def log_epoch(self, row, global_step):
        pass

    def log_baseline(self, baseline):
        pass

    def log_test(self, test):
        pass

    def log_summary(self, results):
        pass

    def save_model(self, path, aliases=None):
        pass

    def finish(self):
        pass


class WandbLogger(NullWandbLogger):
    """Thin wrapper over wandb.run. Never raises: any wandb failure degrades
    to a warning and disables further logging."""

    enabled = True

    def __init__(self, cfg, save_dir, extra_config=None):
        import wandb  # imported lazily; caller already checked availability

        self._wandb = wandb
        from dataclasses import asdict

        config = asdict(cfg)
        config["save_dir"] = save_dir
        if extra_config:
            config.update(extra_config)

        self.run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name or cfg.name,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            notes=cfg.wandb_notes or None,
            mode=cfg.wandb_mode,
            dir=save_dir,
            config=config,
            reinit=True,   # allow several runs in one process (sweeps)
        )
        # Two separate x-axes so the globs don't collide:
        #   step/*  -> global_step  (per-step curves, one point per optimizer step)
        #   train/*, val/*, time/*  -> epoch  (one point per epoch)
        wandb.define_metric("global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("step/*", step_metric="global_step")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("time/*", step_metric="epoch")
        wandb.define_metric("val/nme", step_metric="epoch", summary="min")
        wandb.define_metric("val/loss", step_metric="epoch", summary="min")
        wandb.define_metric("val/pos_recall", step_metric="epoch", summary="max")
        wandb.define_metric("val/pos_f1", step_metric="epoch", summary="max")
        wandb.define_metric("val/map50", step_metric="epoch", summary="max")
        wandb.define_metric("val/map", step_metric="epoch", summary="max")
        wandb.define_metric("val/landm_acc", step_metric="epoch", summary="max")
        self.run_url = getattr(self.run, "url", "") or ""
        self._save_model = cfg.wandb_save_model
        self._log_steps = cfg.wandb_log_steps

    def _guard(self, fn, *a, **kw):
        """Run a wandb call; on failure warn once and go silent."""
        if not self.enabled:
            return
        try:
            fn(*a, **kw)
        except Exception as e:  # noqa: BLE001 - metrics must never kill training
            logging.warning(f"wandb logging failed ({e}); disabling wandb for this run.")
            self.enabled = False

    def log_step(self, row, global_step):
        if not self._log_steps:
            return
        # own namespace + own x-axis value so points spread across steps,
        # not collapse onto the current epoch
        payload = {f"step/{k}": v for k, v in row.items() if k != "lr"}
        payload["step/lr"] = row.get("lr")
        payload["global_step"] = global_step
        self._guard(self._wandb.log, payload, step=global_step)

    def log_epoch(self, row, global_step):
        payload = {"epoch": row["epoch"], "global_step": global_step,
                   "stage": row.get("stage"), "lr": row.get("lr")}
        if row.get("landm_weight") is not None:
            payload["landm_weight"] = row["landm_weight"]
        for k in ("train_loss", "train_reg", "train_cls", "train_landm",
                  "train_acc", "train_landm_acc",
                  "train_pos_precision", "train_pos_recall", "train_pos_f1"):
            if row.get(k) is not None:
                payload[f"train/{k[len('train_'):]}"] = row[k]
        for k in ("val_loss", "val_reg", "val_cls", "val_landm", "val_nme",
                  "val_acc", "val_landm_acc",
                  "val_landm_acc_eyeL", "val_landm_acc_eyeR", "val_landm_acc_nose",
                  "val_landm_acc_mouthL", "val_landm_acc_mouthR",
                  "val_pos_precision", "val_pos_recall", "val_pos_f1",
                  "val_neg_precision", "val_neg_recall",
                  "val_map50", "val_map75", "val_map"):
            if row.get(k) not in (None, ""):
                payload[f"val/{k[len('val_'):]}"] = row[k]
        if row.get("train_time_s") is not None:
            payload["time/train_s"] = row["train_time_s"]
        if row.get("val_time_s") not in (None, ""):
            payload["time/val_s"] = row["val_time_s"]
        self._guard(self._wandb.log, payload, step=global_step)

    def log_baseline(self, baseline):
        self._guard(self._wandb.run.summary.update,
                    {f"baseline/{k}": v for k, v in baseline.items()})

    def log_test(self, test):
        self._guard(self._wandb.run.summary.update,
                    {f"test/{k}": v for k, v in test.items()})

    def log_summary(self, results):
        self._guard(self._wandb.run.summary.update, {
            "best/nme": results.best_nme,
            "best/epoch": results.best_epoch,
            "best/checkpoint": results.best_checkpoint,
            "detection_kept": results.detection_kept,
        })

    def save_model(self, path, aliases=None):
        if not self._save_model or not path or not os.path.isfile(path):
            return

        def _upload():
            art = self._wandb.Artifact(
                name=f"{self.run.name}-weights".replace("/", "-"),
                type="model",
                metadata={"file": os.path.basename(path)})
            art.add_file(path)
            self.run.log_artifact(art, aliases=aliases or ["best"])

        self._guard(_upload)

    def finish(self):
        self._guard(self._wandb.finish)


def create_logger(cfg, save_dir, extra_config=None):
    """Build a WandbLogger when cfg.wandb is on and wandb is importable.

    Returns a NullWandbLogger otherwise, so callers can log unconditionally.
    """
    if not getattr(cfg, "wandb", False):
        return NullWandbLogger()
    try:
        import wandb  # noqa: F401
    except ImportError:
        logging.warning("wandb=True but the wandb package is not installed "
                        "(pip install wandb); continuing without it.")
        return NullWandbLogger()
    try:
        logger = WandbLogger(cfg, save_dir, extra_config)
    except Exception as e:  # noqa: BLE001 - e.g. not logged in, no network
        logging.warning(f"Could not start a wandb run ({e}); continuing without it. "
                        f"Tip: `wandb login`, or pass wandb_mode='offline'.")
        return NullWandbLogger()
    if logger.run_url:
        logging.info(f"wandb run: {logger.run_url}")
    return logger
