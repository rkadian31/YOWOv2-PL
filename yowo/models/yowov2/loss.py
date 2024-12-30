from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import SimOTA
from yowo.utils.box_ops import get_ious
from yowo.utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


class SigmoidFocalLoss(object):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, logits, targets):      
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                        target=targets, 
                                                        reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()

        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class Criterion(nn.Module):
    def __init__(
        self, 
        img_size,
        loss_conf_weight,
        loss_cls_weight,
        loss_reg_weight,
        focal_loss,
        center_sampling_radius,
        topk_candicate,
        num_classes=80, 
        multi_hot=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        # Adjusted weights for high resolution
        self.loss_conf_weight = loss_conf_weight * 1.5  # Increased weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight * 1.2   # Increased weight
        self.focal_loss = focal_loss
        self.multi_hot = multi_hot

        # Modified loss functions for high resolution
        if self.focal_loss:
            self.obj_lossf = SigmoidFocalLoss(
                alpha=0.25,
                gamma=2.0,
                reduction='none'
            )
            self.cls_lossf = SigmoidFocalLoss(
                alpha=0.25,
                gamma=2.0,
                reduction='none'
            )
        else:
            self.obj_lossf = nn.BCEWithLogitsLoss(reduction='none')
            self.cls_lossf = nn.BCEWithLogitsLoss(reduction='none')

        # Added scale-aware IoU loss
        self.scale_aware_iou = True
            
        # Modified matcher for high resolution
        self.matcher = SimOTA(
            num_classes=num_classes,
            center_sampling_radius=center_sampling_radius * 2,  # Increased radius
            topk_candidate=topk_candicate * 2  # Increased candidates
            )
        
        # Added adaptive weight module
        self.adaptive_weight = nn.Parameter(torch.ones(3))  # [conf, cls, reg]

    def get_scale_aware_weights(self, tgt_bboxes):
        """Calculate scale-aware weights based on target box sizes"""
        box_areas = (tgt_bboxes[:, 2] - tgt_bboxes[:, 0]) * (tgt_bboxes[:, 3] - tgt_bboxes[:, 1])
        weights = torch.sqrt(box_areas) / self.img_size
        return weights.clamp(min=0.1, max=2.0)

    def __call__(
        self, 
        outputs: Dict[str, List[torch.Tensor] | List[int]], 
        targets: List[Dict[str, torch.Tensor] | Any]
    ):        
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']

        # Check if input is high resolution
        is_high_res = self.img_size >= 1080

        # Concatenate predictions
        conf_preds = torch.cat(outputs['pred_conf'], dim=1)
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        cls_targets = []
        box_targets = []
        conf_targets = []
        fg_masks = []
        scale_weights = []

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # Denormalize tgt_bbox
            tgt_bboxes *= self.img_size

            # Calculate scale-aware weights
            if is_high_res and self.scale_aware_iou:
                weights = self.get_scale_aware_weights(tgt_bboxes)
                scale_weights.append(weights)

            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum([ab.shape[0] for ab in anchors])
                cls_target = conf_preds.new_zeros((0, self.num_classes))
                box_target = conf_preds.new_zeros((0, 4))
                conf_target = conf_preds.new_zeros((num_anchors, 1))
                fg_mask = conf_preds.new_zeros(num_anchors).bool()
            else:
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.matcher(
                    fpn_strides=fpn_strides,
                    anchors=anchors,
                    pred_conf=conf_preds[batch_idx],
                    pred_cls=cls_preds[batch_idx], 
                    pred_box=box_preds[batch_idx],
                    tgt_labels=tgt_labels,
                    tgt_bboxes=tgt_bboxes,
                    )

                conf_target = fg_mask.unsqueeze(-1)
                box_target = tgt_bboxes[matched_gt_inds]
                
                if self.multi_hot:
                    cls_target = gt_matched_classes.float()
                else:
                    cls_target = F.one_hot(gt_matched_classes.long(), self.num_classes)
                cls_target = cls_target * pred_ious_this_matching.unsqueeze(-1)

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            conf_targets.append(conf_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        conf_targets = torch.cat(conf_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        
        if scale_weights:
            scale_weights = torch.cat(scale_weights, 0)

        num_foregrounds = fg_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foregrounds)
        num_foregrounds = (num_foregrounds / get_world_size()).clamp(1.0)

        # Calculate losses with adaptive weights
        loss_conf = self.obj_lossf(conf_preds.view(-1, 1), conf_targets.float())
        loss_conf = loss_conf.sum() / num_foregrounds

        matched_cls_preds = cls_preds.view(-1, self.num_classes)[fg_masks]
        loss_cls = self.cls_lossf(matched_cls_preds, cls_targets)
        loss_cls = loss_cls.sum() / num_foregrounds

        matched_box_preds = box_preds.view(-1, 4)[fg_masks]
        ious = get_ious(matched_box_preds,
                       box_targets,
                       box_mode="xyxy",
                       iou_type='giou')
        
        if is_high_res and self.scale_aware_iou:
            ious = ious * scale_weights
            
        loss_box = (1.0 - ious).sum() / num_foregrounds

        # Apply adaptive weights
        weights = F.softmax(self.adaptive_weight, dim=0)
        losses = weights[0] * self.loss_conf_weight * loss_conf + \
                weights[1] * self.loss_cls_weight * loss_cls + \
                weights[2] * self.loss_reg_weight * loss_box

        loss_dict = dict(
                loss_conf = loss_conf,
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict


def build_criterion(
    img_size,
    loss_conf_weight,
    loss_cls_weight,
    loss_reg_weight,
    focal_loss,
    center_sampling_radius,
    topk_candicate,
    num_classes=80, 
    multi_hot=False
):
    return Criterion(
        img_size,
        loss_conf_weight,
        loss_cls_weight,
        loss_reg_weight,
        focal_loss,
        center_sampling_radius,
        topk_candicate,
        num_classes=num_classes, 
        multi_hot=multi_hot
    )
    
