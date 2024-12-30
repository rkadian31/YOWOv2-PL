import numpy as np
import torch
import torchvision
from typing import Tuple, Optional


def nms(bboxes, scores, nms_thresh, scale_aware: bool = True):
    """Enhanced Pure Python NMS with scale-aware processing."""
    x1 = bboxes[:, 0]  # xmin
    y1 = bboxes[:, 1]  # ymin
    x2 = bboxes[:, 2]  # xmax
    y2 = bboxes[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)
    
    # Scale-aware NMS for high resolution
    if scale_aware:
        # Adjust threshold based on box area
        mean_area = areas.mean()
        area_weights = torch.sqrt(areas / mean_area)
        adjusted_scores = scores * area_weights
        order = adjusted_scores.argsort()[::-1]
    else:
        order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU with scale awareness
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        if scale_aware:
            # Scale-aware IoU
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            # Adjust threshold based on box size
            dynamic_thresh = nms_thresh * (1 + 0.1 * np.log(areas[i] / mean_area))
            inds = np.where(iou <= dynamic_thresh)[0]
        else:
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            inds = np.where(iou <= nms_thresh)[0]
            
        order = order[inds + 1]

    return keep


def soft_nms(bboxes, scores, nms_thresh, sigma=0.5, score_thresh=0.001):
    """Soft-NMS implementation for high resolution."""
    N = bboxes.shape[0]
    indexes = torch.arange(N)
    
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        tscore = scores[i]
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                indexes[i] = pos + maxpos
                indexes[maxpos + pos] = i
                scores[i], scores[maxpos + pos] = scores[maxpos + pos], scores[i]
                bboxes[i], bboxes[maxpos + pos] = bboxes[maxpos + pos].clone(), bboxes[i].clone()

        # IoU calculate
        xx1 = torch.maximum(bboxes[i, 0], bboxes[pos:, 0])
        yy1 = torch.maximum(bboxes[i, 1], bboxes[pos:, 1])
        xx2 = torch.minimum(bboxes[i, 2], bboxes[pos:, 2])
        yy2 = torch.minimum(bboxes[i, 3], bboxes[pos:, 3])
        w = torch.maximum(torch.tensor(0, dtype=torch.float32), xx2 - xx1)
        h = torch.maximum(torch.tensor(0, dtype=torch.float32), yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] *= weight

    keep = scores > score_thresh
    return keep


def multiclass_nms_class_agnostic(
    scores: torch.Tensor,
    labels: torch.Tensor,
    bboxes: torch.Tensor,
    nms_thresh: float,
    high_resolution: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enhanced class-agnostic NMS for high resolution."""
    if high_resolution:
        # Use soft-NMS for high resolution
        keep = soft_nms(bboxes, scores, nms_thresh)
    else:
        keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


def multiclass_nms_class_aware(
    scores: torch.Tensor,
    labels: torch.Tensor,
    bboxes: torch.Tensor,
    nms_thresh: float,
    num_classes: int,
    high_resolution: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enhanced class-aware NMS for high resolution."""
    keep = torch.zeros(bboxes.size(0), dtype=torch.int32)
    
    if high_resolution:
        # Calculate box areas for scale-aware processing
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        mean_area = areas.mean()
        
        for i in range(num_classes):
            inds = torch.where(labels == i)[0]
            if inds.size(0) == 0:
                continue
                
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_areas = areas[inds]
            
            # Adjust NMS threshold based on box size
            size_factor = torch.sqrt(c_areas / mean_area)
            dynamic_thresh = nms_thresh * (1 + 0.1 * torch.log(size_factor))
            
            c_keep = torchvision.ops.nms(
                c_bboxes,
                c_scores,
                dynamic_thresh.mean()
            )
            keep[inds[c_keep]] = 1
    else:
        for i in range(num_classes):
            inds = torch.where(labels == i)[0]
            if inds.size(0) == 0:
                continue
                
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = torchvision.ops.nms(c_bboxes, c_scores, nms_thresh)
            keep[inds[c_keep]] = 1

    keep = torch.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


def multiclass_nms_tensor(
    scores: torch.Tensor,
    labels: torch.Tensor,
    bboxes: torch.Tensor,
    nms_thresh: float,
    num_classes: int,
    class_agnostic: bool = False,
    high_resolution: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enhanced multiclass NMS with high resolution support."""
    if class_agnostic:
        return multiclass_nms_class_agnostic(
            scores, labels, bboxes, nms_thresh, high_resolution
        )
    else:
        return multiclass_nms_class_aware(
            scores, labels, bboxes, nms_thresh, num_classes, high_resolution
        )
