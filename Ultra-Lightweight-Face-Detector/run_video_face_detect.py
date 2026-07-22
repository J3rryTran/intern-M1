"""
This code uses the pytorch model to detect faces from live video or camera.
"""
import argparse
import sys
import cv2

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--threshold', default=0.7, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cpu", type=str,
                    help='inference mặc định chạy CPU; truyền cuda:0 nếu muốn dùng GPU')
parser.add_argument('--precision', default="fp32", type=str,
                    help='compute precision: fp32 | fp16 | bf16 (fp16/bf16 chỉ có lợi trên GPU; CPU nên giữ fp32)')
parser.add_argument('--video_path', default="/home/linzai/Videos/video/16_1.MP4", type=str,
                    help='path of video')
args = parser.parse_args()

define_img_size()  # must be called before importing the FD builder (populates config priors)

from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer, resolve_device_dtype

label_path = "./models/voc-model-labels.txt"

cap = cv2.VideoCapture(args.video_path)  # capture from video
# cap = cv2.VideoCapture(0)  # capture from camera

with open(label_path) as f:
    class_names = [name.strip() for name in f]
num_classes = len(class_names)
# resolve device + dtype together (fp16 on CPU -> fp32, bf16 without support -> fp16)
test_device, dtype = resolve_device_dtype(args.precision, device=args.test_device)
print(f"Precision: {args.precision} -> device {test_device}, dtype {dtype}")

candidate_size = args.candidate_size
threshold = args.threshold

model_path = "models/pretrained/version-RFB-640.pth"
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
net.load(model_path)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size,
                                            device=test_device, dtype=dtype)

timer = Timer()
sum = 0
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    interval = timer.end()
    print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f" {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)

        # cv2.putText(orig_image, label,
        #             (box[0], box[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,  # font scale
        #             (0, 0, 255),
        #             2)  # line type
    orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
    sum += boxes.size(0)
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("all face num:{}".format(sum))
