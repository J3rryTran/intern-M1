import argparse
import gc
import os
import sys

import cv2
import torch

from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import resolve_device_dtype

parser = argparse.ArgumentParser(
    description='detect_imgs')

parser.add_argument('--threshold', default=0.6, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--device', default="cpu", type=str,
                    help='inference mặc định chạy CPU; truyền cuda:0 nếu muốn dùng GPU')
parser.add_argument('--precision', default="fp32", type=str,
                    help='compute precision: fp32 | fp16 | bf16 (fp16/bf16 chỉ có lợi trên GPU; CPU nên giữ fp32)')
parser.add_argument('--cpu_threads', default=0, type=int,
                    help='Giới hạn số luồng CPU cho PyTorch (0 = tự động, khuyến nghị: 1-8)')
parser.add_argument('--gpu_memory_fraction', default=0.0, type=float,
                    help='Giới hạn phần trăm VRAM GPU (0.0-1.0, 0.0 = không giới hạn, VD: 0.5 = dùng 50%% VRAM)')
parser.add_argument('--limit_ram', default=0.0, type=float,
                    help='Giới hạn RAM tối đa cho tiến trình (GB, 0.0 = không giới hạn, VD: 8 = 8GB)')
args = parser.parse_args()
_ram_limit_job_handle = None


def limit_ram_usage(gb):
    global _ram_limit_job_handle
    if gb <= 0:
        return
    num_bytes = int(gb * (1024 ** 3))

    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', wintypes.LARGE_INTEGER),
                ('PerJobUserTimeLimit', wintypes.LARGE_INTEGER),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_ulonglong),
                ('WriteOperationCount', ctypes.c_ulonglong),
                ('OtherOperationCount', ctypes.c_ulonglong),
                ('ReadTransferCount', ctypes.c_ulonglong),
                ('WriteTransferCount', ctypes.c_ulonglong),
                ('OtherTransferCount', ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JobObjectExtendedLimitInformation = 9

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            print(f"[Hardware] Không tạo được Job Object, bỏ qua giới hạn RAM "
                  f"(err={ctypes.get_last_error()})")
            return

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = num_bytes
        ok = kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            print(f"[Hardware] SetInformationJobObject thất bại "
                  f"(err={ctypes.get_last_error()})")
            return
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            print(f"[Hardware] AssignProcessToJobObject thất bại "
                  f"(err={ctypes.get_last_error()})")
            return
        _ram_limit_job_handle = job  # giữ handle để giới hạn còn hiệu lực
    else:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (num_bytes, num_bytes))

    print(f"[Hardware] Giới hạn RAM tiến trình: {gb:.1f} GB")


limit_ram_usage(args.limit_ram)
define_img_size()  # must be called before importing the FD builder (populates config priors)

# ====== Giới hạn tài nguyên phần cứng ======
if args.cpu_threads > 0:
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(args.cpu_threads)
    print(f"[Hardware] Giới hạn CPU threads: {args.cpu_threads}")
else:
    print(f"[Hardware] CPU threads: {torch.get_num_threads()} (tự động)")

if args.device != 'cpu' and torch.cuda.is_available():
    if args.gpu_memory_fraction > 0.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        print(f"[Hardware] Giới hạn GPU VRAM: {args.gpu_memory_fraction * 100:.0f}%")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[Hardware] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print(f"[Hardware] Chạy trên CPU")

result_path = "./detect_imgs_results"
label_path = "./models/voc-model-labels.txt"
# resolve device + dtype together (fp16 on CPU -> fp32, bf16 without support -> fp16)
device, dtype = resolve_device_dtype(args.precision, device=args.device)
print(f"[Hardware] Precision: {args.precision} -> device {device}, dtype {dtype}")

with open(label_path) as f:
    class_names = [name.strip() for name in f]
model_path = "models/pretrained/version-RFB-640.pth"
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
net.load(model_path)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size,
                                            device=device, dtype=dtype)

if not os.path.exists(result_path):
    os.makedirs(result_path)
listdir = os.listdir(args.path)
total_faces = 0
total_images = 0
all_confidences = []

for file_path in listdir:
    img_path = os.path.join(args.path, file_path)
    orig_image = cv2.imread(img_path)
    if orig_image is None:
        continue
    total_images += 1
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
    total_faces += boxes.size(0)

    face_details = []
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        confidence = probs[i].item() * 100
        all_confidences.append(probs[i].item())
        face_details.append(f"{confidence:.1f}%")
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        label = f"{confidence:.1f}%"
        cv2.putText(orig_image, label, (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(result_path, file_path), orig_image)

    if len(face_details) > 0:
        print(f"[{file_path}] Found {len(probs)} faces: {', '.join(face_details)}")
    else:
        print(f"[{file_path}] Found 0 faces.")

# ====== Thống kê tổng hợp ======
print("\n" + "=" * 60)
print("                   KẾT QUẢ ĐÁNH GIÁ MODEL")
print("=" * 60)
print(f"  Tổng số ảnh xử lý       : {total_images}")
print(f"  Tổng số khuôn mặt       : {total_faces}")
if total_images > 0:
    print(f"  Trung bình faces/ảnh     : {total_faces / total_images:.2f}")
if len(all_confidences) > 0:
    avg_conf = sum(all_confidences) / len(all_confidences) * 100
    min_conf = min(all_confidences) * 100
    max_conf = max(all_confidences) * 100
    print(f"  Confidence trung bình    : {avg_conf:.2f}%")
    print(f"  Confidence thấp nhất     : {min_conf:.2f}%")
    print(f"  Confidence cao nhất      : {max_conf:.2f}%")
else:
    print("  Không phát hiện khuôn mặt nào.")
print(f"  CPU threads đã dùng     : {torch.get_num_threads()}")
print("=" * 60)

del net, predictor
if args.device != 'cpu' and torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
print("Cleanup complete.")
