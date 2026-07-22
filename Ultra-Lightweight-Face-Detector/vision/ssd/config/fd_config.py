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


def define_img_size():
    """Hardcoded to input size 320x320 (square). Populates the global shrinkage_list and priors.

    Everything downstream is fully convolutional, so switching the spatial size
    only changes the number of priors (5,875 at 320x320) - the pretrained
    weights in version-RFB-640.pth still load unchanged.
    """
    global image_size, feature_map_w_h_list, priors
    image_size = [320, 320]
    feature_map_w_h_list = [[40, 20, 10, 5], [40, 20, 10, 5]]

    shrinkage_list.clear()  # reset so re-calling define_img_size() stays idempotent
    for i in range(0, len(image_size)):
        item_list = []
        for k in range(0, len(feature_map_w_h_list[i])):
            item_list.append(image_size[i] / feature_map_w_h_list[i][k])
        shrinkage_list.append(item_list)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
