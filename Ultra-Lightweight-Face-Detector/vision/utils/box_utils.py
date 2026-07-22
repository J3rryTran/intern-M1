import math

import torch


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True) -> torch.Tensor:
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print(f"priors nums:{len(priors)}")
    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    r"""Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def encode_landm(landmarks, center_form_priors, center_variance):
    """Encode 5-point landmarks as offsets w.r.t. their matched priors.

    For each point (lx, ly) and prior (pcx, pcy, pw, ph):
        enc_x = (lx - pcx) / (center_variance * pw)
        enc_y = (ly - pcy) / (center_variance * ph)

    Args:
        landmarks (..., num_priors, 10): (x0, y0, ..., x4, y4) in percent
            coordinates relative to image size (same unit as priors).
        center_form_priors (num_priors, 4): (cx, cy, w, h) percent coordinates.
        center_variance: center scale factor, same 0.1 as used for boxes.
    Returns:
        encoded (..., num_priors, 10): unitless offsets.
    """
    if center_form_priors.dim() + 1 == landmarks.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    lead_shape = landmarks.shape[:-1]
    points = landmarks.reshape(lead_shape + (5, 2))
    centers = center_form_priors[..., :2].unsqueeze(-2)  # (..., num_priors, 1, 2)
    sizes = center_form_priors[..., 2:].unsqueeze(-2)    # (..., num_priors, 1, 2)
    encoded = (points - centers) / (center_variance * sizes)
    return encoded.reshape(lead_shape + (10,))


def decode_landm(encoded_landmarks, center_form_priors, center_variance):
    """Inverse of encode_landm.

    Args:
        encoded_landmarks (..., num_priors, 10): unitless landmark offsets.
        center_form_priors (num_priors, 4): (cx, cy, w, h) percent coordinates.
        center_variance: center scale factor, same 0.1 as used for boxes.
    Returns:
        landmarks (..., num_priors, 10): (x0, y0, ..., x4, y4) percent coords.
    """
    if center_form_priors.dim() + 1 == encoded_landmarks.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    lead_shape = encoded_landmarks.shape[:-1]
    points = encoded_landmarks.reshape(lead_shape + (5, 2))
    centers = center_form_priors[..., :2].unsqueeze(-2)
    sizes = center_form_priors[..., 2:].unsqueeze(-2)
    decoded = points * (center_variance * sizes) + centers
    return decoded.reshape(lead_shape + (10,))


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold, gt_landmarks=None, gt_landm_mask=None):
    """Assign ground truth boxes and targets to priors.

    Landmarks and their mask (when given) are gathered with the exact same
    matched target index as the boxes: one IoU matching, every per-prior
    target comes from the same ground-truth face.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes (corner form, percent coords).
        gt_labels (num_targets): labels of targets (0 is reserved for background).
        corner_form_priors (num_priors, 4): corner form priors, percent coords.
        iou_threshold: minimum IoU for a prior to keep its matched label.
        gt_landmarks (num_targets, 5, 2), optional: percent landmark coords,
            (-1, -1) dummy where the point has no usable label.
        gt_landm_mask (num_targets, 5), optional: per-point supervision mask
            (>0 = real label, 0 = missing).
    Returns:
        boxes (num_priors, 4): matched boxes for priors.
        labels (num_priors): labels for priors (0 = background).
        landmarks (num_priors, 5, 2), landm_mask (num_priors, 5): only when
            gt_landmarks is given.
    """
    with_landmarks = gt_landmarks is not None
    num_priors = corner_form_priors.size(0)
    if gt_boxes.numel() == 0:
        # Image without any face: every prior is background. Use the priors
        # themselves as (masked-out) box targets so the encoded locations stay
        # finite; the landmark mask is all-zero so no landmark is supervised.
        boxes = corner_form_priors.clone()
        labels = torch.zeros(num_priors, dtype=torch.int64)
        if not with_landmarks:
            return boxes, labels
        landmarks = torch.zeros((num_priors, 5, 2), dtype=corner_form_priors.dtype)
        landm_mask = torch.zeros((num_priors, 5), dtype=corner_form_priors.dtype)
        return boxes, labels, landmarks, landm_mask

    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    if not with_landmarks:
        return boxes, labels
    landmarks = gt_landmarks[best_target_per_prior_index]
    landm_mask = gt_landm_mask[best_target_per_prior_index]
    return boxes, labels, landmarks, landm_mask


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])
