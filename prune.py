from __future__ import annotations

import argparse

import torch
import torch.nn.utils.prune as prune

from ultralytics import YOLO

# Head modules whose channels must stay intact (they encode nc / reg_max / kpt_shape).
_HEAD_NAMES = {"Detect", "Pose", "Pose26", "v10Detect"}


def count(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, nonzero_params)."""
    total = sum(p.numel() for p in model.parameters())
    nonzero = int(sum((p != 0).sum().item() for p in model.parameters()))
    return total, nonzero


def prune_unstructured(model: torch.nn.Module, ratio: float) -> None:
    """Global L1 unstructured pruning over all Conv2d/Linear weights, made permanent (weights set to 0)."""
    targets = [
        (m, "weight") for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=ratio)
    for module, name in targets:
        prune.remove(module, name)  # bake the mask into the weights


def prune_structured(yolo: YOLO, ratio: float, imgsz: int) -> None:
    """Dependency-aware structured channel pruning via torch-pruning (physically removes channels)."""
    try:
        import torch_pruning as tp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("structured pruning requires torch-pruning:  pip install torch-pruning") from exc

    model = yolo.model.eval()
    example = torch.zeros(1, 3, imgsz, imgsz)
    ignored = [m for m in model.modules() if m.__class__.__name__ in _HEAD_NAMES]
    pruner = tp.pruner.MagnitudePruner(
        model,
        example,
        importance=tp.importance.MagnitudeImportance(p=1),  # L1 channel importance
        pruning_ratio=ratio,
        ignored_layers=ignored,
    )
    pruner.step()  # mutates model in place


def main() -> None:
    """Parse args, prune, report the parameter reduction, and save the result."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model .pt or .yaml")
    ap.add_argument("--mode", choices=["structured", "unstructured"], default="structured")
    ap.add_argument("--ratio", type=float, default=0.3, help="fraction to prune in (0, 1)")
    ap.add_argument("--imgsz", type=int, default=640, help="example input size for the dependency graph")
    ap.add_argument("--out", default="pruned.pt")
    args = ap.parse_args()
    assert 0.0 < args.ratio < 1.0, "--ratio must be in (0, 1)"

    yolo = YOLO(args.model)
    before = count(yolo.model)
    if args.mode == "structured":
        prune_structured(yolo, args.ratio, args.imgsz)
    else:
        prune_unstructured(yolo.model, args.ratio)
    after = count(yolo.model)

    print(f"mode={args.mode} ratio={args.ratio}")
    print(f"params  : {before[0]:,} -> {after[0]:,}")
    print(f"nonzero : {before[1]:,} -> {after[1]:,}")
    yolo.save(args.out)
    print(f"saved   : {args.out}")
    print(f"next    : yolo detect train model={args.out} data=coco8.yaml epochs=100  # fine-tune to recover mAP")


if __name__ == "__main__":
    main()
