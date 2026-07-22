from collections import namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision.utils import box_utils

GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, landmark_headers: nn.ModuleList = None,
                 is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.

        landmark_headers is optional. When given, the model additionally
        predicts, per prior, 10 encoded landmark offsets (5 points x (x, y)),
        computed from the same feature maps as the detection heads.
        """
        super().__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.landmark_headers = landmark_headers
        self.has_landmark_heads = landmark_headers is not None
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run the SSD forward pass.

        Args:
            x (batch, 3, H, W): normalized input images (RFB-640: 3x640x640).
        Returns:
            Training mode (is_test=False):
                confidences (batch, num_priors, num_classes): raw class logits.
                locations (batch, num_priors, 4): encoded box offsets.
              and, only when landmark heads exist:
                landmarks (batch, num_priors, 10): encoded landmark offsets
                    (x0, y0, ..., x4, y4).
            Test mode (is_test=True):
                confidences after softmax, boxes decoded to corner-form percent
                coords, and (when landmark heads exist) landmarks decoded to
                percent coords.
        """
        confidences = []
        locations = []
        landmarks = []
        start_layer_index = 0
        header_index = 0
        end_layer_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location, landmark = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            landmarks.append(landmark)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location, landmark = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            landmarks.append(landmark)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        if self.has_landmark_heads:
            landmarks = torch.cat(landmarks, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            if not self.has_landmark_heads:
                return confidences, boxes
            landmarks = box_utils.decode_landm(landmarks, self.priors, self.config.center_variance)
            return confidences, boxes, landmarks
        else:
            if not self.has_landmark_heads:
                return confidences, locations
            return confidences, locations, landmarks

    def compute_header(self, i, x):
        """Apply the i-th prediction heads to feature map x.

        Args:
            i: index of the feature map / head (0..3 for RFB-640).
            x (batch, C_i, H_i, W_i): feature map.
        Returns:
            confidence (batch, H_i*W_i*priors_i, num_classes),
            location (batch, H_i*W_i*priors_i, 4),
            landmark (batch, H_i*W_i*priors_i, 10) or None.
        """
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        landmark = None
        if self.has_landmark_heads:
            landmark = self.landmark_headers[i](x)
            landmark = landmark.permute(0, 2, 3, 1).contiguous()
            landmark = landmark.view(landmark.size(0), -1, 10)

        return confidence, location, landmark

    def _init_landmark_heads(self):
        if self.has_landmark_heads:
            self.landmark_headers.apply(_xavier_init_)

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location="cpu", weights_only=True), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)
        self._init_landmark_heads()

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location="cpu", weights_only=True)
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith(("classification_headers", "regression_headers",
                                           "landmark_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)
        self._init_landmark_heads()

    def init_from_pretrained_detector(self, model):
        """Fine-tune entry point: load a detection-only checkpoint with
        strict=False, keeping every pretrained weight (backbone + detection
        heads) and leaving only the new landmark heads randomly
        (xavier) initialized.

        Returns:
            (missing_keys, unexpected_keys) as reported by load_state_dict:
            missing_keys must only contain landmark head keys.
        """
        state_dict = torch.load(model, map_location="cpu", weights_only=True)
        self._init_landmark_heads()
        incompatible = self.load_state_dict(state_dict, strict=False)
        return incompatible.missing_keys, incompatible.unexpected_keys

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)
        self._init_landmark_heads()

    def load(self, model):
        """Load a checkpoint. strict=False so that old detection-only
        checkpoints still load into the landmark-extended architecture (the
        new heads then keep their random init and only detection outputs are
        meaningful). Shape mismatches still raise."""
        self.load_state_dict(torch.load(model, map_location="cpu", weights_only=True), strict=False)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior:
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels, gt_landmarks=None, gt_landm_mask=None):
        """Match ground truth to priors and encode the regression targets.

        Args:
            gt_boxes (num_targets, 4): corner-form boxes, percent coords [0, 1].
            gt_labels (num_targets,): int labels (0 = background is reserved).
            gt_landmarks (num_targets, 5, 2), optional: percent landmark coords,
                (-1, -1) dummy where the point has no usable label.
            gt_landm_mask (num_targets, 5), optional: per-point supervision mask
                (>0 = point has a real label, 0 = missing -> excluded from the
                landmark loss). This is a training mask only; the model does not
                predict it.
        Returns:
            locations (num_priors, 4): encoded box targets.
            labels (num_priors,): per-prior class labels.
            and, only when gt_landmarks is given:
            encoded_landmarks (num_priors, 10): encode_landm() targets
                (garbage where the mask is 0 / background priors - masked in loss).
            landm_mask (num_priors, 5): the per-point mask gathered per prior.
        """
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        if gt_landmarks is None:
            boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                    self.corner_form_priors, self.iou_threshold)
            boxes = box_utils.corner_form_to_center_form(boxes)
            locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
            return locations, labels

        if type(gt_landmarks) is np.ndarray:
            gt_landmarks = torch.from_numpy(gt_landmarks)
        if type(gt_landm_mask) is np.ndarray:
            gt_landm_mask = torch.from_numpy(gt_landm_mask)
        boxes, labels, landmarks, landm_mask = box_utils.assign_priors(
            gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold,
            gt_landmarks, gt_landm_mask)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        encoded_landmarks = box_utils.encode_landm(
            landmarks.reshape(landmarks.size(0), 10), self.center_form_priors, self.center_variance)
        return locations, labels, encoded_landmarks, landm_mask


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
