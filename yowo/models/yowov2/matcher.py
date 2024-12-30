import torch
import torch.nn.functional as F
from yowo.utils.box_ops import *

class SimOTA(object):
    def __init__(self, num_classes, center_sampling_radius, topk_candidate):
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius
        self.topk_candidate = topk_candidate
        # Added adaptive parameters for high resolution
        self.min_box_size = 8  # Minimum box size in pixels
        self.iou_weight = 3.0  # Weight for IoU loss
        self.resolution_aware = True  # Enable resolution-aware matching

    @torch.no_grad()
    def __call__(
        self, 
        fpn_strides, 
        anchors,
        pred_conf,
        pred_cls,
        pred_box,
        tgt_labels,
        tgt_bboxes,
        ): 
        # Get input resolution
        img_size = max(tgt_bboxes.max().item(), 1080)  # Assume normalized coordinates
        is_high_res = img_size >= 1080

        # Adjust parameters for high resolution
        if is_high_res:
            self.center_sampling_radius *= 2
            self.topk_candidate *= 2
            self.iou_weight *= 1.5

        strides = torch.cat([torch.ones_like(anchor_i[:, 0]) * stride_i
                           for stride_i, anchor_i in zip(fpn_strides, anchors)], dim=-1)
        anchors = torch.cat(anchors, dim=0)
        num_anchor = anchors.shape[0]        
        num_gt = len(tgt_labels)

        # Get positive candidates with resolution awareness
        fg_mask, is_in_boxes_and_center = \
            self.get_in_boxes_info(
                tgt_bboxes,
                anchors,
                strides,
                num_anchor,
                num_gt,
                img_size
                )

        conf_preds_ = pred_conf[fg_mask]
        cls_preds_ = pred_cls[fg_mask]
        box_preds_ = pred_box[fg_mask]
        num_in_boxes_anchor = box_preds_.shape[0]

        # Calculate IoUs with resolution awareness
        pair_wise_ious, _ = self.get_box_iou(tgt_bboxes, box_preds_, img_size)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if len(tgt_labels.shape) == 1:
            gt_cls = F.one_hot(tgt_labels.long(), self.num_classes)
        elif len(tgt_labels.shape) == 2:
            gt_cls = tgt_labels

        gt_cls = gt_cls.float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)

        with torch.cuda.amp.autocast(enabled=False):
            score_preds_ = torch.sqrt(
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * conf_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                score_preds_, gt_cls, reduction="none"
            ).sum(-1)
        del score_preds_

        # Adjust cost calculation for high resolution
        if is_high_res:
            # Add scale-aware weighting
            box_areas = (tgt_bboxes[:, 2] - tgt_bboxes[:, 0]) * (tgt_bboxes[:, 3] - tgt_bboxes[:, 1])
            scale_weights = torch.sqrt(box_areas).unsqueeze(1) / img_size
            scale_weights = scale_weights.clamp(min=0.1, max=2.0)
            pair_wise_ious_loss *= scale_weights

        cost = (
            pair_wise_cls_loss
            + self.iou_weight * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(
            cost,
            pair_wise_ious,
            tgt_labels,
            num_gt,
            fg_mask,
            img_size
            )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes,
        anchors,
        strides,
        num_anchors,
        num_gt,
        img_size
        ):
        # Adjust for high resolution
        is_high_res = img_size >= 1080
        center_radius = self.center_sampling_radius * (2 if is_high_res else 1)

        x_centers = anchors[:, 0]
        y_centers = anchors[:, 1]

        x_centers = x_centers.unsqueeze(0).repeat(num_gt, 1)
        y_centers = y_centers.unsqueeze(0).repeat(num_gt, 1)

        gt_bboxes_l = gt_bboxes[:, 0].unsqueeze(1).repeat(1, num_anchors)
        gt_bboxes_t = gt_bboxes[:, 1].unsqueeze(1).repeat(1, num_anchors)
        gt_bboxes_r = gt_bboxes[:, 2].unsqueeze(1).repeat(1, num_anchors)
        gt_bboxes_b = gt_bboxes[:, 3].unsqueeze(1).repeat(1, num_anchors)

        b_l = x_centers - gt_bboxes_l
        b_r = gt_bboxes_r - x_centers
        b_t = y_centers - gt_bboxes_t
        b_b = gt_bboxes_b - y_centers
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        # Add minimum box size check for high resolution
        if is_high_res:
            box_sizes = torch.minimum(
                torch.minimum(b_l, b_r),
                torch.minimum(b_t, b_b)
            )
            valid_size_mask = box_sizes > self.min_box_size
            is_in_boxes = (bbox_deltas.min(dim=-1).values > 0.0) & valid_size_mask
        else:
            is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0

        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5
        center_radius_ = center_radius * strides.unsqueeze(0)

        gt_bboxes_l = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) - center_radius_
        gt_bboxes_t = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) - center_radius_
        gt_bboxes_r = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) + center_radius_
        gt_bboxes_b = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) + center_radius_

        c_l = x_centers - gt_bboxes_l
        c_r = gt_bboxes_r - x_centers
        c_t = y_centers - gt_bboxes_t
        c_b = gt_bboxes_b - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask, img_size):
        # Adjust matching for high resolution
        is_high_res = img_size >= 1080
        
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        ious_in_boxes_matrix = pair_wise_ious

        # Adjust topk candidates for high resolution
        n_candidate_k = min(
            self.topk_candidate * (2 if is_high_res else 1),
            ious_in_boxes_matrix.size(1)
        )

        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        
        # Adjust dynamic_ks for high resolution
        if is_high_res:
            dynamic_ks = [max(k, 2) for k in dynamic_ks.tolist()]
        else:
            dynamic_ks = dynamic_ks.tolist()

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1

        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
