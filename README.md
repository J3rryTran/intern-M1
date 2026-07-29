# Face Detection + 5-Point Landmark for Edge Devices

> 🎓 **Intern Project — M1** — training and comparing ultra-lightweight face detection models with 5 facial landmarks (both eyes, nose, both mouth corners), targeting edge deployment (ARM/x86 CPU, mobile, MCU).

## Credits & License

This is a **student/research internship project** built on top of the open-source projects below. Sincere thanks to their authors.

| Component in this repo | Upstream source | Original license |
|---|---|---|
| `ultralytics/` (customized: added ShuffleV2Block/RepViTBlock, face-pose YAMLs, train/infer/export scripts) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) v8.4.62 | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| `prune.py`, `quantize.py` (written against the Ultralytics API, carrying the Ultralytics header) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| `Ultra-Lightweight-Face-Detector/` (customized: kept `version-RFB` only, added landmark head + trainer) | [Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | [MIT](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/LICENSE) |

Inherited code keeps the original license of its respective project. Note that because this repo contains code derived from Ultralytics, the work as a whole must comply with the terms of **AGPL-3.0** when distributed.

### References

- [WIDER FACE: A Face Detection Benchmark](http://shuoyang1213.me/WIDERFACE/)
- [RetinaFace (InsightFace)](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) — 5-landmark annotations
- [ShuffleNet V2](https://arxiv.org/abs/1807.11164) · [RepViT](https://arxiv.org/abs/2307.09283) · [RFBNet](https://github.com/ruinmessi/RFBNet)
- [torch-pruning](https://github.com/VainF/Torch-Pruning) · [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) · [NCNN](https://github.com/Tencent/ncnn)

## Overview

This project trains and compares several ultra-lightweight face detection + 5-landmark models, then optimizes them for edge deployment through **pruning** and **INT8 quantization**.

### Model summary

Every model uses a **320×320 input** so that the comparison is fair and the resolution matches edge-device constraints.

| Model | Directory / config | Backbone | Landmarks | Input | Notes |
|---|---|---|---|---|---|
| **RFB-320 + landmark** | [`Ultra-Lightweight-Face-Detector/`](Ultra-Lightweight-Face-Detector/) | RFB (SSD-style) | 5-point head added on top, two-stage fine-tuning (freeze → unfreeze), Wing loss, evaluated with PCK | 320 | Fork of Linzaer, `version-RFB` only; the exported ONNX has softmax + box/landmark decoding baked in, so only NMS is needed externally |
| **yolo26n base** (key `base`) | `yolo26.yaml` | Stock YOLO26n | ✗ (detection only) | 320 | Reference baseline |
| **shufflenet-slim** (key `shufflenet-slim`) | `yolo26n-shufflenetv2-face-pose-slim.yaml` | ShuffleNetV2 | ✓ Pose26 head | 320 | Two *slim* cuts applied: drop `C2PSA` (−63K params) and slim the neck's P5 branch (≈−69K params); all three scales P3/P4/P5 kept |
| **shufflenet-slim2** (key `shufflenet-slim2`) | `yolo26n-shufflenetv2-face-pose-slim2.yaml` | ShuffleNetV2 | ✓ Pose26 head | 320 | Same as `shufflenet-slim` plus one more stride-1 `ShuffleV2Block` removed from the P5 backbone (−9.2K params) |
| **repvit-slim** (key `repvit-slim`) | `yolo26n-repvit-face-pose-slim.yaml` | RepViT | ✓ Pose26 head | 320 | *Slim* cuts are **identical** to `shufflenet-slim`, deliberately, so the two backbones can be compared A/B fairly |

The YOLO26n face-pose variants share the same settings: `nc: 1` (a single *face* class), `kpt_shape: [5, 3]` — 5 landmarks per face as `(x, y, visible)` — `end2end: True`, and `reg_max: 1`. The `ShuffleV2Block` and `RepViTBlock` modules were added to `ultralytics/nn/modules/block.py`.

> **Environment:** Python **3.12**, PyTorch + CUDA, trained on an RTX 3060 12GB. Each subdirectory is a separate library with its own installation procedure — see the README inside it.

## Dataset

Source: **WIDER FACE** with RetinaFace-style 5-landmark annotations, filtered and converted.

- **RFB-320 + landmark:** `images/{train,val}/` + `labels/{train,val}/` directories, passed via `--datasets`.
- **YOLO26n face-pose:** YOLO pose format with a `dataset.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val

kpt_shape: [5, 3]
flip_idx: [1, 0, 2, 4, 3]

names:
  0: face
```

`flip_idx` swaps the eye pair (0↔1) and the mouth-corner pair (3↔4) while leaving the nose (2) in place. It has to be correct: otherwise horizontal-flip augmentation silently corrupts the landmark labels.

> ⚠️ The `data` paths in `ultralytics/train.py`, `infer.py` and `export_int8.py` are hard-coded to the original test machine — update them before running.

## RFB-320 + landmark

Run everything inside [`Ultra-Lightweight-Face-Detector/`](Ultra-Lightweight-Face-Detector/); see the [fork's README](Ultra-Lightweight-Face-Detector/README.md) for the full argument list.

| Task | Script |
|---|---|
| Fine-tune the landmark head (two stages) | `train.py` |
| Export to ONNX | `convert_to_onnx.py` |
| Image demo | `detect_imgs.py` / `detect_imgs_onnx.py` |
| Webcam / video demo | `run_video_face_detect.py` / `run_video_face_detect_onnx.py` |
| Utilities: FLOPs count, label check | `cal_flops.py`, `check_gt_box.py` |

## YOLO26n face-pose

Run from the **repo root** so that the customized `ultralytics/` package in this repo takes precedence.

| Task | Command |
|---|---|
| Train | `python -m ultralytics.train <key>` |
| Evaluate box + pose mAP | `python -m ultralytics.infer val <weights>` |
| Predict on image / folder / video | `python -m ultralytics.infer predict <weights> <source>` |
| Speed benchmark | `python -m ultralytics.infer bench <weights> --device cpu` |

`<key>` is one of `base` | `shufflenet-slim` | `shufflenet-slim2` | `repvit-slim`, or pass a YAML filename from `ultralytics/cfg/models/` directly.

## Edge optimization

Applies to trained checkpoints from either model.

| Script | Purpose |
|---|---|
| `prune.py` | `structured` pruning (torch-pruning — genuinely reduces FLOPs/latency/size, **requires fine-tuning afterwards**) or `unstructured` pruning (zeroes the smallest weights; only pays off if the runtime exploits sparsity) |
| `quantize.py` | INT8 quantization calibrated on real images, three export paths: `onnx` (ONNX Runtime QDQ — recommended), `fx` (PyTorch FX static PTQ), `ncnn` |
| `ultralytics/export_int8.py` | Validated INT8 pipeline: val FP32 → export INT8 ONNX → val INT8 → PASS/FAIL against an mAP50 drop threshold (0.02 by default) |

## Results

*(To be updated once all experiments are complete.)*

**Evaluation protocol.** All models are trained and evaluated on the same train/val split, at the same input resolution (**320×320**), and benchmarked on the same machine at batch size 1. Report the exact CPU/GPU model together with the numbers — latency is meaningless without it.

- Test machine: `<CPU model>` / `<GPU model>`
- Runtime: PyTorch `<version>`, ONNX Runtime `<version>`, CUDA `<version>`
- Python 3.12

### 1. Model comparison

| Model | Input | Params (M) | GFLOPs | Size (MB) | box mAP50 | box mAP50-95 | pose mAP50 | PCK@0.05 | NME | CPU latency (ms) | CPU FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RFB-320 + landmark | 320 | | | | | | | | | | |
| yolo26n **base** (detection only, reference) | 320 | | | | | | — | — | — | | |
| yolo26n-shufflenetv2-face-pose-**slim** | 320 | | | | | | | | | | |
| yolo26n-shufflenetv2-face-pose-**slim2** | 320 | | | | | | | | | | |
| yolo26n-repvit-face-pose-**slim** | 320 | | | | | | | | | | |

All numbers are FP32. `Params`, `GFLOPs` and `Size` measure model cost; `box mAP` measures detection quality; `pose mAP50`, `PCK@0.05` and `NME` measure landmark quality (NME is the only metric where lower is better). `base` has no landmark head, so its pose columns stay empty — it is there to show what the lightweight backbones give up relative to the stock YOLO26n detector.

### 2. FP32 vs INT8 (after pruning / quantization)

Target: keep the mAP50 drop within **0.02** — the PASS threshold used by `export_int8.py` — in exchange for a smaller, faster model.

| Model | Precision | Size (MB) | box mAP50 | Δ mAP50 | CPU latency (ms) |
|---|---|---|---|---|---|
| shufflenet-slim | FP32 | | | — | |
| shufflenet-slim | INT8 (ONNX) | | | | |
| shufflenet-slim2 | FP32 | | | — | |
| shufflenet-slim2 | INT8 (ONNX) | | | | |
| repvit-slim | FP32 | | | — | |
| repvit-slim | INT8 (ONNX) | | | | |
| RFB-320 + landmark | FP32 | | | — | |
| RFB-320 + landmark | INT8 (ONNX) | | | | |

### 3. Charts

Drop the image files into [`images/`](images/) and uncomment the corresponding line.

**3.1. Accuracy–efficiency trade-off** — *scatter plot*, GFLOPs (or CPU latency) on the x-axis, box mAP50 on the y-axis, one point per model, marker size proportional to parameter count. This is the single most important chart for an edge deployment: it shows which model buys the most accuracy per unit of compute, and which ones sit below the Pareto front.

<!-- ![Accuracy vs GFLOPs trade-off](images/tradeoff_map_vs_gflops.png) -->

**3.2. Per-metric comparison** — *grouped bar chart*, one group per model with box mAP50 and pose mAP50 side by side. Keep params and GFLOPs in a separate chart; mixing them with mAP on one axis makes both unreadable because the scales differ by orders of magnitude.

<!-- ![mAP comparison across models](images/bar_map_comparison.png) -->
<!-- ![Params and GFLOPs comparison](images/bar_params_gflops.png) -->

**3.3. FP32 vs INT8** — *paired bar chart*, two bars per model, plotted twice: once for latency and once for box mAP50. Side-by-side pairs make the speedup and the accuracy cost readable in a single glance.

<!-- ![FP32 vs INT8](images/bar_fp32_vs_int8.png) -->

**3.4. Qualitative results** — *side-by-side image grid*, the same test images run through every model with boxes and 5 landmarks drawn. Pick hard cases on purpose: small faces, crowds, profile views, low light, occlusion. Numbers hide failure modes that a single image makes obvious.

<!-- ![Qualitative comparison](images/qualitative_comparison.jpg) -->

**3.5. Training curves** *(optional)* — *line chart* of loss and mAP per epoch, taken from W&B or from the `results.csv` that Ultralytics writes per run, to compare how fast each backbone converges.

<!-- ![Training curves](images/training_curves.png) -->
