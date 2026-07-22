import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import box_utils


def wing_loss(pred, target, w=10.0, epsilon=2.0):
    diff = (pred - target).abs()
    c = w - w * math.log(1.0 + w / epsilon)
    return torch.where(diff < w, w * torch.log(1.0 + diff / epsilon), diff - c).sum()


class MultiboxLoss(nn.Module):
    def __init__(self, priors, neg_pos_ratio,
                 center_variance, size_variance, device,
                 loc_weight=2.0, landm_weight=1.0,
                 landm_loss_type="wing", wing_w=10.0, wing_epsilon=2.0):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)
        self.loc_weight = loc_weight
        self.landm_weight = landm_weight
        if landm_loss_type not in ("wing", "smooth_l1"):
            raise ValueError(f"Unsupported landm_loss_type: {landm_loss_type}")
        self.landm_loss_type = landm_loss_type
        self.wing_w = wing_w
        self.wing_epsilon = wing_epsilon

    def forward(self, confidence, predicted_locations, labels, gt_locations,
                predicted_landmarks=None, gt_landmarks=None, gt_landm_mask=None):
        """
        loss = loc_weight * loc + cls + landm_weight * landm

        gt_landm_mask (batch, num_priors, 5) is a supervision mask only (>0 =
        the point has a real label): points without one carry dummy (-1, -1)
        coords and must stay out of the landmark loss.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], reduction='sum')
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')  # smooth_l1_loss
        # smooth_l1_loss = F.mse_loss(predicted_locations, gt_locations, reduction='sum')  #l2 loss
        # guard against batches without any positive prior (e.g. face-less images)
        num_pos = max(gt_locations.size(0), 1)

        if predicted_landmarks is None:
            return smooth_l1_loss / num_pos, classification_loss / num_pos

        # two-level mask: positive priors AND points that actually have a label
        point_mask = pos_mask.unsqueeze(-1) & (gt_landm_mask > 0)
        batch, num_priors = pos_mask.size(0), pos_mask.size(1)
        pred_points = predicted_landmarks.reshape(batch, num_priors, 5, 2)[point_mask]
        gt_points = gt_landmarks.reshape(batch, num_priors, 5, 2)[point_mask]
        if self.landm_loss_type == "wing":
            landmark_loss = wing_loss(pred_points, gt_points, self.wing_w, self.wing_epsilon)
        else:
            landmark_loss = F.smooth_l1_loss(pred_points, gt_points, reduction='sum')

        regression_loss = smooth_l1_loss / num_pos
        classification_loss = classification_loss / num_pos
        landmark_loss = landmark_loss / num_pos
        total_loss = (self.loc_weight * regression_loss
                      + classification_loss
                      + self.landm_weight * landmark_loss)
        return total_loss, regression_loss, classification_loss, landmark_loss
