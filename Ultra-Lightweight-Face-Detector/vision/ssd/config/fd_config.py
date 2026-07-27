import math

import numpy as np

from vision.utils.box_utils import generate_priors

image_mean_test = image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2

min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
shrinkage_list = []
image_size = [320, 320]  # hardcoded input size 320*320 (square)
# feature map [w-row, h-row] per stride 8/16/32/64; square input -> both rows equal
feature_map_w_h_list = [[40, 20, 10, 5], [40, 20, 10, 5]]  # feature map size for 320*320
priors = []


def define_img_size(size=320):
    """Set the (square) input size and rebuild the priors.

    Feature maps are derived by repeated ceil-halving - exactly what the
    stride-2 convs produce - so any size works (multiples of 32 recommended):
    320 -> 5,875 priors, 640 -> 23,500. Fully convolutional, so the pretrained
    weights load unchanged at any size. Re-callable (idempotent per size).
    """
    global image_size, feature_map_w_h_list, priors
    image_size = [size, size]
    fm, s = [], size
    for _ in range(6):
        s = math.ceil(s / 2)
        fm.append(s)
    feature_map_w_h_list = [fm[2:6], fm[2:6]]  # strides 8/16/32/64

    shrinkage_list.clear()  # reset so re-calling stays idempotent
    for i in range(0, len(image_size)):
        item_list = []
        for k in range(0, len(feature_map_w_h_list[i])):
            item_list.append(image_size[i] / feature_map_w_h_list[i][k])
        shrinkage_list.append(item_list)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
