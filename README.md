# Face Detection + 5-Point Landmark cho thiết bị Edge

> 🎓 **Dự án thực tập (Intern Project — M1)** — thử nghiệm và so sánh các mô hình phát hiện khuôn mặt siêu nhẹ kèm 5 điểm landmark (2 mắt, mũi, 2 khoé miệng), hướng tới triển khai trên thiết bị edge (CPU ARM/x86, mobile, MCU).

Dự án được xây dựng dựa trên hai mã nguồn mở:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (bản 8.4.62) — giấy phép **AGPL-3.0**
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) của Linzaer — giấy phép **MIT**

Xem chi tiết ở mục [Ghi công & Giấy phép](#ghi-công--giấy-phép).

## Tổng quan

Dự án thử nghiệm **hai hướng tiếp cận** cho bài toán face detection + landmark, sau đó tối ưu mô hình cho edge bằng **pruning** và **INT8 quantization**:

| Hướng | Thư mục | Ý tưởng chính |
|---|---|---|
| 1. RFB-640 + landmark head | [`Ultra-Lightweight-Face-Detector/`](Ultra-Lightweight-Face-Detector/) | Fork của Linzaer, chỉ giữ **version-RFB @ 640×480**, gắn thêm nhánh dự đoán 5 landmark và fine-tune 2 giai đoạn từ checkpoint gốc |
| 2. YOLO26n + backbone nhẹ | [`ultralytics/`](ultralytics/) | Bản Ultralytics tùy biến: thay backbone bằng **ShuffleNetV2** / **RepViT**, head **Pose26** học 5 keypoint (`kpt_shape: [5, 3]`), có các biến thể *slim* cắt giảm tham số |

### Hướng 1 — RFB-640 + landmark (fork Ultra-Light-Fast-Generic-Face-Detector-1MB)

- Chỉ giữ kiến trúc `version-RFB`, input cố định **640×480**; đã bỏ `slim` và các biến thể 320.
- Thêm **landmark head 5 điểm**, fine-tune 2 giai đoạn: (1) đóng băng mạng, chỉ train head mới; (2) mở băng toàn mạng với learning rate thấp (backbone thấp hơn nữa).
- Loss landmark: **Wing loss** (mặc định) hoặc Smooth L1; đánh giá bằng **PCK**.
- Hỗ trợ export **ONNX** (đã bake sẵn softmax + decode box/landmark, chỉ cần NMS phía ngoài), demo ảnh/video bằng PyTorch và ONNX Runtime.
- Checkpoint kèm sẵn trong `Ultra-Lightweight-Face-Detector/models/`: `version-RFB-640.pth` (gốc, chỉ detect), `version-RFB-640.onnx`, và checkpoint đã fine-tune landmark trong `train-landmark/`.

Chi tiết xem [README của fork](Ultra-Lightweight-Face-Detector/README.md).

### Hướng 2 — YOLO26n với backbone ShuffleNetV2 / RepViT

Các block `ShuffleV2Block`, `RepViTBlock` (+`RepViTSE`) được thêm vào `ultralytics/nn/modules/block.py`, khai báo kiến trúc bằng YAML trong [`ultralytics/cfg/models/`](ultralytics/cfg/models/):

| Key (dùng với `train.py`) | File YAML | Ghi chú |
|---|---|---|
| `base` | `yolo26.yaml` | Baseline YOLO26n gốc (detect, không có nhánh pose) |
| — | `yolo26n-shufflenetv2-face.yaml` | Thay backbone ShuffleNetV2, giữ nguyên neck + head Detect (chưa có landmark) |
| — | `yolo26n-repvit-face.yaml` | Thay backbone RepViT, giữ nguyên neck + head Detect (chưa có landmark) |
| `shufflenet-slim` | `yolo26n-shufflenetv2-face-pose-slim.yaml` | ShuffleNetV2 + head Pose26, áp dụng 2 phép cắt *slim* bên dưới |
| `shufflenet-slim2` | `yolo26n-shufflenetv2-face-pose-slim2.yaml` | Như `shufflenet-slim`, cắt thêm 1 `ShuffleV2Block` stride-1 ở P5 backbone (P5 còn 1×s2 + 1×s1, −9.2K params); giữ nguyên 4 block P4 |
| `repvit-slim` | `yolo26n-repvit-face-pose-slim.yaml` | RepViT + head Pose26, áp dụng **đúng 2 phép cắt *slim*** như `shufflenet-slim` — cố ý cắt giống hệt nhau để so sánh A/B công bằng giữa hai backbone |

Các YAML không có key vẫn train được bằng cách truyền thẳng tên file: `python -m ultralytics.train yolo26n-shufflenetv2-face.yaml`.

**Phép cắt *slim*** (so với bản face-pose dùng neck/head YOLO26 nguyên gốc — có `C2PSA` và nhánh P5 `C3k2` 512 kênh kèm attention):

1. **Bỏ block attention `C2PSA`** ở cuối backbone (−63K params); giữ lại `SPPF` để bảo toàn receptive field.
2. **Slim nhánh P5 của neck:** `C3k2` cuối giảm 512→384 (đầu ra 96 kênh thay vì 128) và bỏ `attn=True` (~−69K params).
3. **Giữ đủ 3 scale P3/P4/P5** — dữ liệu train nhiều mặt nhỏ/đông (ưu tiên recall) nên không cắt P3.

Đặc điểm chung của các bản face-pose: `nc: 1` (1 lớp *face*), `kpt_shape: [5, 3]` — mỗi mặt 5 landmark `(x, y, visible)`, `end2end: True`, `reg_max: 1`.

## Cấu trúc thư mục

```
intern-M1/
├── Ultra-Lightweight-Face-Detector/   # Hướng 1: fork ULFD-1MB (MIT)
│   ├── train.py                       #   fine-tune RFB-640 + landmark head (2 giai đoạn)
│   ├── convert_to_onnx.py             #   export checkpoint landmark sang ONNX
│   ├── detect_imgs(_onnx).py          #   demo trên ảnh (PyTorch / ONNX Runtime)
│   ├── run_video_face_detect(_onnx).py#   demo webcam/video
│   ├── cal_flops.py, check_gt_box.py  #   tiện ích: đo FLOPs, kiểm tra nhãn
│   ├── models/                        #   checkpoint .pth / .onnx
│   └── vision/                        #   kiến trúc SSD/RFB, trainer, datasets, transforms
├── ultralytics/                       # Hướng 2: Ultralytics tùy biến (AGPL-3.0)
│   ├── cfg/models/                    #   YAML kiến trúc yolo26n-*-face-pose*
│   ├── nn/modules/block.py            #   + ShuffleV2Block, RepViTBlock, RepViTSE
│   ├── train.py                       #   train face-pose (chọn kiến trúc qua key)
│   ├── infer.py                       #   val / predict / bench
│   └── export_int8.py                 #   pipeline PTQ INT8 (ONNX) + kiểm định mAP
├── prune.py                           # Pruning: structured (torch-pruning) / unstructured
├── quantize.py                        # Quantization INT8: FX / ONNX Runtime / NCNN
├── images/                            # ảnh demo / kết quả
└── LICENSE
```

## Cài đặt

Yêu cầu Python ≥ 3.9 và PyTorch (khuyến nghị có GPU để train; các thử nghiệm gốc chạy trên RTX 3060 12GB).

```bash
# Dependencies chung
pip install torch torchvision opencv-python numpy

# Hướng 1 (ULFD) — tuỳ chọn thêm
pip install -r Ultra-Lightweight-Face-Detector/requirements.txt

# Hướng 2 (YOLO26) — cài ultralytics để kéo đủ dependencies;
# bản tùy biến trong repo sẽ được ưu tiên khi chạy `python -m ultralytics.*` từ thư mục gốc
pip install ultralytics

# Tối ưu edge — tuỳ chọn
pip install torch-pruning        # prune.py --mode structured
pip install onnx onnxruntime     # quantize.py --method onnx / fx
pip install wandb                # log thí nghiệm bằng Weights & Biases (tuỳ chọn)
```

## Dữ liệu

Cả hai hướng dùng dữ liệu khuôn mặt kèm 5 landmark, gốc từ **WIDER FACE** + nhãn landmark kiểu RetinaFace, đã lọc/chuyển đổi lại:

- **Hướng 1 (ULFD):** thư mục dạng `images/{train,val}/` + `labels/{train,val}/` (truyền qua `--datasets`); pipeline VOC gốc của repo Linzaer vẫn dùng được — xem [README của fork](Ultra-Lightweight-Face-Detector/README.md).
- **Hướng 2 (YOLO26):** định dạng **YOLO pose** với `dataset.yaml` (`kpt_shape: [5, 3]`).

> ⚠️ Đường dẫn `data` trong `ultralytics/train.py`, `ultralytics/infer.py`, `ultralytics/export_int8.py` đang hard-code theo máy thử nghiệm — sửa lại cho đúng vị trí dataset của bạn trước khi chạy.

## Sử dụng

Các lệnh dưới đây chạy từ **thư mục gốc repo** (riêng Hướng 1 chạy trong thư mục fork).

### Hướng 1 — fine-tune RFB-640 + landmark

```bash
cd Ultra-Lightweight-Face-Detector

# Fine-tune từ checkpoint detect gốc (2 giai đoạn: freeze -> unfreeze)
python train.py --datasets ../data/exp --freeze_epochs 5 --landm_weight 1.0 --landm_loss wing

# Export ONNX rồi demo
python convert_to_onnx.py
python detect_imgs_onnx.py
python run_video_face_detect_onnx.py
```

### Hướng 2 — train / đánh giá / suy luận YOLO26n face-pose

```bash
# Train (key kiến trúc: base | shufflenet-slim | shufflenet-slim2 | repvit-slim, hoặc truyền thẳng file YAML)
python -m ultralytics.train shufflenet-slim

# Đo mAP trên val/test (box + pose), kèm tốc độ
python -m ultralytics.infer val runs_face/<run_name>/weights/best.pt

# Suy luận ảnh / thư mục / video (vẽ box + 5 landmark)
python -m ultralytics.infer predict runs_face/<run_name>/weights/best.pt path/to/images

# Benchmark tốc độ thuần (không cần dataset)
python -m ultralytics.infer bench runs_face/<run_name>/weights/best.pt --device cpu
```

### Tối ưu cho edge

**1. Pruning (`prune.py`)** — hai chế độ, chạy cả hai để so sánh:

- `structured`: cắt hẳn các kênh conv ít quan trọng theo đồ thị phụ thuộc (torch-pruning) → giảm thật FLOPs/latency/kích thước. **Bắt buộc fine-tune lại** sau khi prune.
- `unstructured`: zero các trọng số nhỏ nhất (L1 toàn cục) → sparsity cao, chỉ lợi khi runtime khai thác được sparse.

```bash
python prune.py --model runs_face/<run>/weights/best.pt --mode structured --ratio 0.3 --out runs/pruned.pt
# rồi fine-tune để hồi mAP:
yolo detect train model=runs/pruned.pt data=<dataset.yaml> epochs=100 imgsz=640
```

**2. Quantization INT8 (`quantize.py`)** — calibrate trên ảnh thật, 3 đường xuất:

```bash
python quantize.py --model runs/pruned.pt --method onnx --data <folder_ảnh_val> --imgsz 640   # ONNX Runtime QDQ (khuyến nghị, dễ mang đi nhất)
python quantize.py --model runs/pruned.pt --method fx   --data <folder_ảnh_val>              # PyTorch FX static PTQ (CPU)
python quantize.py --model runs/pruned.pt --method ncnn --data <folder_ảnh_val>              # export NCNN + in lệnh ncnn2table/ncnn2int8
```

**3. Pipeline INT8 có kiểm định (`ultralytics/export_int8.py`)** — val FP32 → export ONNX INT8 (calibrate từ dataset) → val INT8 → PASS/FAIL theo ngưỡng drop mAP50 (mặc định 0.02):

```bash
python -m ultralytics.export_int8 runs_face/<run>/weights/best.pt 0.02
```

## Kết quả

*(sẽ cập nhật sau khi hoàn tất các thí nghiệm — so sánh params / GFLOPs / mAP50 box·pose / latency FP32 vs INT8 giữa các biến thể)*

| Model | Params | GFLOPs | mAP50 (box) | mAP50 (pose) | Latency |
|---|---|---|---|---|---|
| RFB-640 + landmark | | | | | |
| yolo26n-shufflenetv2-face-pose-slim | | | | | |
| yolo26n-shufflenetv2-face-pose-slim2 | | | | | |
| yolo26n-repvit-face-pose-slim | | | | | |

## Ghi công & Giấy phép

Đây là **dự án thực tập phục vụ học tập/nghiên cứu**, xây dựng trên mã nguồn của các dự án sau — xin chân thành cảm ơn các tác giả:

| Thành phần trong repo | Nguồn gốc | Giấy phép gốc |
|---|---|---|
| `ultralytics/` (đã tùy biến: thêm ShuffleV2Block/RepViTBlock, YAML face-pose, script train/infer/export) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) v8.4.62 | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| `prune.py`, `quantize.py` (viết theo API và mang header Ultralytics) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| `Ultra-Lightweight-Face-Detector/` (đã tùy biến: chỉ giữ RFB-640, thêm landmark head + trainer) | [Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | [MIT](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/LICENSE) |

Phần mã kế thừa giữ nguyên giấy phép gốc của từng dự án nêu trên. Lưu ý: do repo chứa mã dẫn xuất từ Ultralytics, toàn bộ tác phẩm khi phân phối phải tuân thủ các điều khoản của **AGPL-3.0**.

### Tài liệu tham khảo

- [WIDER FACE: A Face Detection Benchmark](http://shuoyang1213.me/WIDERFACE/)
- [RetinaFace (InsightFace)](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) — nhãn 5 landmark
- [ShuffleNet V2](https://arxiv.org/abs/1807.11164) · [RepViT](https://arxiv.org/abs/2307.09283) · [RFBNet](https://github.com/ruinmessi/RFBNet)
- [torch-pruning](https://github.com/VainF/Torch-Pruning) · [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) · [NCNN](https://github.com/Tencent/ncnn)
