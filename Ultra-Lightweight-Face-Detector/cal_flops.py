"""Report model complexity (params / GFLOPs / inference time).

Uses the dependency-free helpers in vision.utils.misc (print_model_summary +
profile_gflops) instead of the abandoned torchstat / torchsummary / ptflops.
"""
import time

import torch

from vision.ssd.config.fd_config import define_img_size
from vision.ssd.config import fd_config
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.utils.misc import print_model_summary

define_img_size()  # populate priors for the configured input size
device = "cpu"
width, height = fd_config.image_size  # currently 320x320

fd = create_Mb_Tiny_RFB_fd(2, device=device)
fd.eval()
fd.to(device)

print_model_summary(fd, input_size=(3, height, width), device=device)

x = torch.randn(1, 3, height, width).to(device)
with torch.no_grad():
    for _ in range(5):
        t = time.time()
        fd(x)
        print(f"inference time: {time.time() - t:.4f} s")
