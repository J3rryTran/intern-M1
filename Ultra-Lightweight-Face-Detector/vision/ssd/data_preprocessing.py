from ..transforms.transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size (w, h): the size of the final image (RFB-640: (640, 640)).
            mean: mean pixel value per channel (subtracted).
            std: pixel std divisor.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            RandomSampleCrop_v2(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            imgprocess(std),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels, landms=None, vis=None):
        """
        Args:
            img: the output of cv.imread in RGB layout (HWC uint8).
            boxes (N, 4): bounding boxes in pixel corner form (x1, y1, x2, y2).
            labels (N,): labels of boxes.
            landms (N, 5, 2), optional: pixel landmark coords, (-1, -1) for
                masked-out points.
            vis (N, 5), optional: per-point supervision mask (>0 = usable label).
        Returns:
            (img, boxes, labels) when landms is None (legacy detection-only);
            (img, boxes, labels, landms, vis) otherwise. img is a normalized
            CHW float tensor; boxes/landms are in percent coords [0, 1].
        """
        img, boxes, labels, landms, vis = self.augment(img, boxes, labels, landms, vis)
        if landms is None:
            return img, boxes, labels
        return img, boxes, labels, landms, vis


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            imgprocess(std),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels, landms=None, vis=None):
        """Deterministic transform for validation. Same signature/returns as
        TrainAugmentation (3-tuple without landms, 5-tuple with)."""
        image, boxes, labels, landms, vis = self.transform(image, boxes, labels, landms, vis)
        if landms is None:
            return image, boxes, labels
        return image, boxes, labels, landms, vis


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            imgprocess(std),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _, _, _ = self.transform(image)
        return image
