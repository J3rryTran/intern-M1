# from https://github.com/amdegroot/ssd.pytorch
#
# Extended for 5-point facial landmarks:
# every transform now takes and returns (image, boxes, labels, landms, vis).
#   landms (N, 5, 2): landmark coords, same unit as boxes at that stage
#       (pixels before ToPercentCoords, percent [0, 1] after). Masked-out
#       points hold the dummy value (-1, -1).
#   vis (N, 5): per-point supervision mask: >0 = the point has a usable label
#       and lies inside the image, 0 = no label / cropped away. It is a
#       training mask only - the model does not predict it.
# Landmark point order is FIXED everywhere:
#   0=left eye, 1=right eye, 2=nose, 3=left mouth corner, 4=right mouth corner
# ("left/right" as seen in the image). Horizontal mirroring therefore has to
# swap the symmetric pairs: see RandomMirror.FLIP_IDX.
# landms/vis may be None (legacy detection-only pipelines) - they then pass
# through untouched.


import types

import cv2
import numpy as np
import torch
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def object_converage_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    return inter / area_a  # [A,B]


def mask_invalid_landmarks(landms, vis):
    """Reset coords of points with v == 0 to the dummy (-1, -1).

    Called after every geometric transform so stale/garbage coordinates never
    leak through the pipeline. Mutates landms in place and returns it.

    Args:
        landms (N, 5, 2) or None. vis (N, 5) or None.
    """
    if landms is not None and vis is not None and len(landms):
        landms[vis == 0] = -1.0
    return landms


class Compose:
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, landms=None, vis=None):
        for t in self.transforms:
            img, boxes, labels, landms, vis = t(img, boxes, labels, landms, vis)
        return img, boxes, labels, landms, vis


class Lambda:
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None, landms=None, vis=None):
        return self.lambd(img, boxes, labels, landms, vis)


class ConvertFromInts:
    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        return image.astype(np.float32), boxes, labels, landms, vis


class SubtractMeans:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels, landms, vis


class imgprocess:
    """Divide the image by std (pixel-only, picklable lambda replacement)."""

    def __init__(self, std):
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        image = image.astype(np.float32)
        image /= self.std
        return image.astype(np.float32), boxes, labels, landms, vis


class ToAbsoluteCoords:
    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        if landms is not None:
            landms[:, :, 0] *= width
            landms[:, :, 1] *= height
            mask_invalid_landmarks(landms, vis)

        return image, boxes, labels, landms, vis


class ToPercentCoords:
    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        if landms is not None:
            landms[:, :, 0] /= width
            landms[:, :, 1] /= height
            # keep the dummy value exactly (-1, -1) for v=0 points
            mask_invalid_landmarks(landms, vis)

        return image, boxes, labels, landms, vis


class Resize:
    def __init__(self, size=(300, 300)):
        self.size = size

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        # boxes/landms are expected in percent coords at this stage
        # (ToPercentCoords runs before Resize), so only the image changes.
        image = cv2.resize(image, (self.size[0],
                                   self.size[1]))
        return image, boxes, labels, landms, vis


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels, landms, vis


class RandomHue:
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, landms, vis


class RandomLightingNoise:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels, landms, vis


class ConvertColor:
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels, landms, vis


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels, landms, vis


class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels, landms, vis


class ToCV2Image:
    def __call__(self, tensor, boxes=None, labels=None, landms=None, vis=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels, landms, vis


class ToTensor:
    def __call__(self, cvimage, boxes=None, labels=None, landms=None, vis=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels, landms, vis


def _crop_landmarks(landms, vis, keep_mask, rect):
    """Shared landmark logic for the RandomSampleCrop variants.

    Keeps the same faces as the box mask, shifts points into the crop frame,
    and invalidates (v=0, coords=(-1, -1)) every point that falls outside the
    crop - a box can survive the crop while some of its points do not.

    Args:
        landms (N, 5, 2): pixel coords. vis (N, 5): per-point mask.
        keep_mask (N,) bool: faces kept by the crop (box center inside rect).
        rect (4,): crop rectangle (x1, y1, x2, y2) in pixels.
    Returns:
        (kept_landms, kept_vis) in the crop's coordinate frame.
    """
    current_landms = landms[keep_mask, :, :].copy()
    current_vis = vis[keep_mask, :].copy()
    current_landms[:, :, 0] -= rect[0]
    current_landms[:, :, 1] -= rect[1]
    crop_w = rect[2] - rect[0]
    crop_h = rect[3] - rect[1]
    inside = ((current_landms[:, :, 0] >= 0) & (current_landms[:, :, 0] < crop_w) &
              (current_landms[:, :, 1] >= 0) & (current_landms[:, :, 1] < crop_h))
    current_vis[~inside] = 0
    mask_invalid_landmarks(current_landms, current_vis)
    return current_landms, current_vis


class RandomSampleCrop:
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        landms (N, 5, 2) / vis (N, 5): landmarks + their mask, cropped along
            with their boxes; points outside the crop get mask 0 and (-1, -1).
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes, landms, vis)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        height, width, _ = image.shape
        # without GT boxes there is nothing to constrain the crop: keep the image
        if boxes is None or len(boxes) == 0:
            return image, boxes, labels, landms, vis
        while True:
            # randomly choose a mode (by index: np.random.choice rejects
            # ragged sequences like (None, (0.1, None), ...) on numpy >= 1.24)
            mode = self.sample_options[random.randint(len(self.sample_options))]
            if mode is None:
                return image, boxes, labels, landms, vis

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                current_landms, current_vis = landms, vis
                if landms is not None:
                    current_landms, current_vis = _crop_landmarks(landms, vis, mask, rect)

                return current_image, current_boxes, current_labels, current_landms, current_vis


class RandomSampleCrop_v2:
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        landms (N, 5, 2) / vis (N, 5): landmarks + their mask, cropped along
            with their boxes; points outside the crop get mask 0 and (-1, -1).
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes, landms, vis)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9

            # randomly sample a patch
            (1, None),
            (1, None),
            (1, None),
            (1, None),
        )

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        height, width, _ = image.shape
        # without GT boxes there is nothing to constrain the crop: keep the image
        if boxes is None or len(boxes) == 0:
            return image, boxes, labels, landms, vis
        while True:
            # randomly choose a mode (by index: np.random.choice rejects
            # ragged sequences like (None, (1, None), ...) on numpy >= 1.24)
            mode = self.sample_options[random.randint(len(self.sample_options))]
            if mode is None:
                return image, boxes, labels, landms, vis

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w != 1:
                    continue
                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = object_converage_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                current_landms, current_vis = landms, vis
                if landms is not None:
                    current_landms, current_vis = _crop_landmarks(landms, vis, mask, rect)

                return current_image, current_boxes, current_labels, current_landms, current_vis


class Expand:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        if random.randint(2):
            return image, boxes, labels, landms, vis

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        if landms is not None:
            # expansion only translates content; no point can leave the image
            landms = landms.copy()
            landms[:, :, 0] += int(left)
            landms[:, :, 1] += int(top)
            mask_invalid_landmarks(landms, vis)

        return image, boxes, labels, landms, vis


class RandomMirror:
    # After a horizontal flip the anatomical sides swap, so the FIXED index
    # convention (0=left eye, 1=right eye, 2=nose, 3=left mouth, 4=right mouth,
    # as seen in the image) requires re-ordering both coords AND their mask:
    FLIP_IDX = [1, 0, 2, 4, 3]

    def __call__(self, image, boxes=None, classes=None, landms=None, vis=None):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            if boxes is not None:
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]
            if landms is not None:
                landms = landms.copy()
                vis = vis.copy()
                # flip x, then swap symmetric pairs (coords and flags together)
                landms[:, :, 0] = width - landms[:, :, 0]
                landms = landms[:, self.FLIP_IDX, :]
                vis = vis[:, self.FLIP_IDX]
                # restore the dummy (-1, -1) for v=0 points mangled by the flip
                mask_invalid_landmarks(landms, vis)
        return image, boxes, classes, landms, vis


class SwapChannels:
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes=None, labels=None, landms=None, vis=None):
        im = image.copy()
        im, boxes, labels, landms, vis = self.rand_brightness(im, boxes, labels, landms, vis)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels, landms, vis = distort(im, boxes, labels, landms, vis)
        return self.rand_light_noise(im, boxes, labels, landms, vis)
