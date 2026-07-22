import logging
import os

import cv2
import numpy as np

NUM_LANDMARKS = 5
TOKENS_PER_LINE = 20
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


class YoloPoseDataset:
    def __init__(self, root, transform=None, target_transform=None, split=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images_dir = os.path.join(root, "images", split) if split else os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels", split) if split else os.path.join(root, "labels")
        if not os.path.isdir(self.images_dir):
            raise ValueError(f"Image directory not found: {self.images_dir}")

        self.image_paths = sorted(
            os.path.join(self.images_dir, name)
            for name in os.listdir(self.images_dir)
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No image found under {self.images_dir}")
        self.ids = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]
        self.class_names = ('BACKGROUND', 'face')
        logging.info(f"YoloPoseDataset: {len(self.image_paths)} images from {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _label_path(self, image_path):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.labels_dir, stem + ".txt")

    def _parse_label_file(self, label_path, width, height):
        """Label line: cls cx cy w h + 5x(x, y, v), all normalized.

        The v column is read ONLY to build the landmark supervision mask
        (1.0 = point has a real label, 0.0 = missing -> dummy (-1, -1) coords).
        The model does not predict visibility.
        """
        boxes, labels, landms, mask = [], [], [], []
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                tokens = line.split()
                if len(tokens) != TOKENS_PER_LINE:
                    if tokens:
                        logging.debug(f"Skip malformed line ({len(tokens)} tokens) in {label_path}")
                    continue
                try:
                    values = [float(t) for t in tokens]
                except ValueError:
                    logging.debug(f"Skip non-numeric line in {label_path}")
                    continue
                if int(values[0]) != 0:
                    continue

                cx, cy, bw, bh = values[1:5]
                x1 = np.clip((cx - bw / 2.0) * width, 0, width)
                y1 = np.clip((cy - bh / 2.0) * height, 0, height)
                x2 = np.clip((cx + bw / 2.0) * width, 0, width)
                y2 = np.clip((cy + bh / 2.0) * height, 0, height)
                if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                    continue

                face_points = np.full((NUM_LANDMARKS, 2), -1.0, dtype=np.float32)
                face_mask = np.zeros(NUM_LANDMARKS, dtype=np.float32)
                for k in range(NUM_LANDMARKS):
                    lx, ly, v = values[5 + 3 * k: 8 + 3 * k]
                    if int(round(v)) <= 0:
                        continue  # no label for this point: keep dummy + mask 0
                    face_points[k, 0] = np.clip(lx * width, 0, width - 1)
                    face_points[k, 1] = np.clip(ly * height, 0, height - 1)
                    face_mask[k] = 1.0

                boxes.append([x1, y1, x2, y2])
                labels.append(1)
                landms.append(face_points)
                mask.append(face_mask)

        if not boxes:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0, NUM_LANDMARKS, 2), dtype=np.float32),
                    np.zeros((0, NUM_LANDMARKS), dtype=np.float32))
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.stack(landms).astype(np.float32),
                np.stack(mask).astype(np.float32))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes, labels, landms, mask = self._parse_label_file(self._label_path(image_path), width, height)

        if self.transform:
            image, boxes, labels, landms, mask = self.transform(image, boxes, labels, landms, mask)
        if self.target_transform:
            boxes, labels, landms, mask = self.target_transform(boxes, labels, landms, mask)
        return image, boxes, labels, landms, mask
