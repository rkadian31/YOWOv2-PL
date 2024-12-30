import torch
import torch.nn as nn
import torch.nn.functional as F

from yowo.models.backbone2d import build_backbone_2d
from yowo.models.backbone3d import build_backbone_3d
from .encoder import build_channel_encoder
from .head import build_head

from yowo.utils.nms import multiclass_nms_tensor
from ..schemas import ModelConfig


def aggregate_features(feat_2ds: list[torch.Tensor], spatial_sizes: list[torch.Tensor]):
    spatial_size_3d = spatial_sizes[-1]
    kernel_sizes = [spatial_size //
                    spatial_size_3d for spatial_size in spatial_sizes]
    out_feat_2ds = []
    for i, feat_2d in enumerate(feat_2ds):
        kernel_size = kernel_sizes[i]
        if kernel_size > 1:
            out_feat_2d = F.unfold(
                feat_2d,
                kernel_size=(kernel_size, kernel_size),
                dilation=1,
                stride=(kernel_size, kernel_size)
            )
            out_feat_2d = out_feat_2d.view(feat_2d.size(0), feat_2d.size(
                1), -1, spatial_size_3d, spatial_size_3d)
            out_feat_2d = torch.mean(out_feat_2d, dim=2)
        else:
            out_feat_2d = feat_2d

        out_feat_2ds.append(out_feat_2d)

    return out_feat_2ds

# You Only Watch Once


class YOWO(nn.Module):
    def __init__(
        self,
        params: ModelConfig
    ):
        # Add memory optimization for high resolution
        self.use_checkpoint = params.img_size[0] >= 1080 or params.img_size[1] >= 1920
        
        super(YOWO, self).__init__()
        # Add high resolution support
        self.high_resolution = params.img_size[0] >= 1080 or params.img_size[1] >= 1920
        # Adjust stride for high resolution
        if self.high_resolution:
            self.stride = [s * 2 for s in params.stride]  # Increase stride for memory efficiency
        else:
            self.stride = params.stride
        self.feature_scales = [1.5 if self.high_resolution else 1.0] * len(self.stride)
        self.num_classes = params.num_classes
        self.conf_thresh = params.conf_thresh
        self.nms_thresh = params.nms_thresh
        self.multi_hot = params.multi_hot
        self.use_aggregate_feat = params.use_aggregate_feat

        # ------------------ Network ---------------------
        # 2D backbone
        self.backbone_2d, bk_dim_2d = build_backbone_2d(
            model_name=params.backbone_2d,
            pretrained=params.pretrained_2d,
            use_blurpool=params.use_blurpool,
            high_resolution=self.high_resolution
        )
        # 3D backbone
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            model_name=params.backbone_3d,
            pretrained=params.pretrained_3d

        # Move this block after backbone_3d initialization
        if self.high_resolution:
            bk_dim_2d = [int(dim * 1.5) for dim in bk_dim_2d]
            bk_dim_3d = int(bk_dim_3d * 1.5)
            params.head_dim = int(params.head_dim * 1.5)
        )

        # cls channel encoder
        self.cls_channel_encoders = nn.ModuleList(
            [build_channel_encoder(
                head_act=params.head_act,
                head_norm=params.head_norm,
                in_dim=bk_dim_2d[i] + bk_dim_3d,
                out_dim=params.head_dim
            ) for i in range(len(params.stride))
            ])

        # reg channel & spatial encoder
        self.reg_channel_encoders = nn.ModuleList(
            [build_channel_encoder(
                head_act=params.head_act,
                head_norm=params.head_norm,
                in_dim=bk_dim_2d[i] + bk_dim_3d,
                out_dim=params.head_dim
            ) for i in range(len(params.stride))
            ])

        # head
        self.heads = nn.ModuleList(
            [build_head(
                num_cls_heads=params.num_cls_heads,
                num_reg_heads=params.num_reg_heads,
                head_act=params.head_act,
                head_norm=params.head_norm,
                head_dim=params.head_dim,
                head_depthwise=params.head_depthwise
            ) for _ in range(len(params.stride))]
        )

        # pred
        head_dim = params.head_dim
        self.conf_preds = nn.ModuleList(
            modules=[
                nn.Conv2d(head_dim, 1, kernel_size=1)
                for _ in range(len(params.stride))
            ])
        self.cls_preds = nn.ModuleList(
            modules=[
                nn.Conv2d(head_dim, self.num_classes, kernel_size=1)
                for _ in range(len(params.stride))
            ])
        self.reg_preds = nn.ModuleList(
            modules=[
                nn.Conv2d(head_dim, 4, kernel_size=1)
                for _ in range(len(params.stride))
            ])

        # init yowo
        self.init_yowo()
        img_size = torch.tensor(params.img_size)
        topk = torch.tensor(params.topk)
        spatial_sizes = torch.tensor([
            img_size // self.stride[i] for i in range(len(params.stride))
        ])
        num_regions = torch.tensor([
            spatial_size**2 for spatial_size in spatial_sizes
        ])
        self.register_buffer('img_size', img_size, persistent=False)
        self.register_buffer('topk', topk, persistent=False)
        self.register_buffer('spatial_sizes', spatial_sizes, persistent=False)
        self.register_buffer('num_regions', num_regions, persistent=False)

    def init_yowo(self):
        # Init yolo
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        for conf_pred in self.conf_preds:
            b = conf_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            conf_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

   def generate_anchors(self, fmp_size, stride, device):
    fmp_h, fmp_w = fmp_size
    
    if self.high_resolution:
        # Use denser anchor grid for high resolution
        base_stride = stride * 0.75  # Reduce effective stride
        anchor_y, anchor_x = torch.meshgrid(
            [torch.arange(fmp_h, device=device), 
             torch.arange(fmp_w, device=device)], 
            indexing='ij'
        )
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float() + 0.5
        anchor_xy *= base_stride
        
        # Add additional scales for high resolution
        scales = torch.tensor([0.75, 1.0, 1.25], device=device)
        anchors = anchor_xy.unsqueeze(2) * scales.view(1, 1, -1, 1)
        anchors = anchors.view(-1, 2)
    else:
        # Original anchor generation
        anchor_y, anchor_x = torch.meshgrid(
            [torch.arange(fmp_h, device=device), 
             torch.arange(fmp_w, device=device)], 
            indexing='ij'
        )
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float() + 0.5
        anchor_xy *= stride
        anchors = anchor_xy.reshape(-1, 2)
    
    return anchors

    def decode_boxes(self, anchors, pred_reg, stride):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors + pred_reg[..., :2] * stride
        # size of bbox
        pred_box_wh = pred_reg[..., 2:].exp() * stride

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box

    def post_process_one_hot(
        self,
        conf_preds: list[torch.Tensor],
        cls_preds: list[torch.Tensor],
        reg_preds: list[torch.Tensor]
    ) -> tuple[torch.Tensor]:
        """
        Input:
            conf_preds: (List[Tensor]) [H x W, 1]
            cls_preds: (List[Tensor]) [H x W, C]
            reg_preds: (List[Tensor]) [H x W, 4]
        Output:
            scores: (Tensor) [M,]
            labels: (Tensor) [M,]
            bboxes: (Tensor) [M, 4]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []

        for level, (conf_pred_i, cls_pred_i, reg_pred_i) in enumerate(zip(conf_preds, cls_preds, reg_preds)):
            # (H x W x C,)
            scores_i = (torch.sqrt(conf_pred_i.sigmoid()
                        * cls_pred_i.sigmoid())).flatten()

            # Keep top k top scoring indices only.
            num_topk = torch.minimum(self.topk, self.num_regions[level])

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            reg_pred_i = reg_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(reg_pred_i)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # nms
        scores, labels, bboxes = multiclass_nms_tensor(
            scores=scores,
            labels=labels,
            bboxes=bboxes,
            nms_thresh=self.nms_thresh,
            num_classes=self.num_classes,
            class_agnostic=False
        )
        # print(scores.shape, labels.shape, bboxes.shape)
        out_boxes = torch.cat(
            [bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        return out_boxes

    def post_process_multi_hot(
        self,
        conf_preds: list[torch.Tensor],
        cls_preds: list[torch.Tensor],
        reg_preds: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Input:
            conf_preds: (List[Tensor]) [H x W, 1]
            cls_preds: (List[Tensor]) [H x W, C]
            reg_preds: (List[Tensor]) [H x W, 4]
        Output:
            out_boxes: (Tensor) [M, 5 + C]
        """
        all_conf_preds = []
        all_cls_preds = []
        all_box_preds = []
        for conf_pred_i, cls_pred_i, reg_pred_i in zip(conf_preds, cls_preds, reg_preds):
            # conf pred
            conf_pred_i = torch.sigmoid(conf_pred_i.squeeze(-1))   # [M,]

            # cls_pred
            cls_pred_i = torch.sigmoid(cls_pred_i)                 # [M, C]

            # topk
            topk_conf_pred_i, topk_inds = torch.topk(
                conf_pred_i, self.topk)
            topk_cls_pred_i = cls_pred_i[topk_inds]
            topk_box_pred_i = reg_pred_i[topk_inds]

            # threshold
            keep = topk_conf_pred_i.gt(self.conf_thresh)
            topk_conf_pred_i = topk_conf_pred_i[keep]
            topk_cls_pred_i = topk_cls_pred_i[keep]
            topk_box_pred_i = topk_box_pred_i[keep]

            all_conf_preds.append(topk_conf_pred_i)
            all_cls_preds.append(topk_cls_pred_i)
            all_box_preds.append(topk_box_pred_i)

        # concatenate
        conf_preds = torch.cat(all_conf_preds, dim=0)  # [M,]
        cls_preds = torch.cat(all_cls_preds, dim=0)    # [M, C]
        box_preds = torch.cat(all_box_preds, dim=0)    # [M, 4]

        scores, labels, bboxes = multiclass_nms_tensor(
            scores=conf_preds,
            labels=cls_preds,
            bboxes=box_preds,
            nms_thresh=self.nms_thresh,
            num_classes=self.num_classes,
            class_agnostic=True
        )

        # [M, 5 + C]
        out_boxes = torch.cat([bboxes, scores.unsqueeze(-1), labels], dim=-1)

        return out_boxes

    @torch.no_grad()
    def post_processing(
        self,
        outputs: dict[str, torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Input:
            outputs: (Dict) -> {
                'pred_conf': (Tensor) [B, M, 1]
                'pred_cls':  (Tensor) [B, M, C]
                'pred_reg':  (Tensor) [B, M, 4]
                'anchors':   (Tensor) [M, 2]
                'stride':    (Int)
            }
        return:
            out_boxes: (List[Tensor])   [B, M, 5 + C] # multihot
                                        [B, M, 6] # onehot
        """

        all_conf_preds = outputs['pred_conf']
        all_cls_preds = outputs['pred_cls']
        all_reg_preds = outputs['pred_box']

        num_batches = all_conf_preds[0].shape[0]

        # batch process
        batch_bboxes = []
        # batch_bboxes = []
        for batch_idx in range(num_batches):
            cur_conf_preds = []
            cur_cls_preds = []
            cur_reg_preds = []
            for conf_preds, cls_preds, reg_preds in zip(all_conf_preds, all_cls_preds, all_reg_preds):
                # [B, M, C] -> [M, C]
                cur_conf_preds.append(conf_preds[batch_idx])
                cur_cls_preds.append(cls_preds[batch_idx])
                cur_reg_preds.append(reg_preds[batch_idx])

            if self.multi_hot:
                # post-process
                out_boxes = self.post_process_multi_hot(
                    cur_conf_preds, cur_cls_preds, cur_reg_preds)
            else:
                out_boxes = self.post_process_one_hot(
                    cur_conf_preds, cur_cls_preds, cur_reg_preds)

            # normalize bbox
            out_boxes[..., :4] /= torch.maximum(
                self.img_size,
                self.img_size
            )
            out_boxes[..., :4] = out_boxes[..., :4].clamp(0., 1.)

            batch_bboxes.append(out_boxes)

        return batch_bboxes

def forward(self, video_clips: torch.Tensor) -> dict[str, torch.Tensor]:
    B, C, T, H, W = video_clips.shape
    # Add resolution check and processing
    if self.high_resolution:
        attention = self._create_spatial_attention(H, W)
        video_clips = video_clips * attention
        
    # key frame
    key_frame = video_clips[:, :, -1, :, :]
    # 3D backbone
    feat_3d = self.backbone_3d(video_clips)
    # 2D backbone
    # list of Tensor (3 scales)
    cls_feats, reg_feats = self.backbone_2d(key_frame)

    # Process features if high resolution - ADD HERE
    if self.high_resolution:
        cls_feats = [self._process_high_res_features(f) for f in cls_feats]
        reg_feats = [self._process_high_res_features(f) for f in reg_feats]
        # Also process 3D features
        feat_3d = self._process_high_res_features(feat_3d)

    # non-shared heads
    all_conf_preds = []
    all_cls_preds = []
    all_box_preds = []
    all_anchors = []

    if self.use_aggregate_feat:
        cls_feats = aggregate_features(
            feat_2ds=cls_feats,
            spatial_sizes=self.spatial_sizes
        )
        reg_feats = aggregate_features(
            feat_2ds=reg_feats,
            spatial_sizes=self.spatial_sizes
        )
        for level, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
            if self.use_aggregate_feat:
                cls_feat = self.cls_channel_encoders[level](
                    cls_feat, feat_3d)
                reg_feat = self.reg_channel_encoders[level](
                    cls_feat, feat_3d)

                cls_feat = F.interpolate(
                    cls_feat, scale_factor=2 ** (2 - level))
                reg_feat = F.interpolate(
                    reg_feat, scale_factor=2 ** (2 - level))
            else:
                # upsample
                feat_3d_up = F.interpolate(
                    feat_3d, scale_factor=2 ** (2 - level))

                # encoder
                cls_feat = self.cls_channel_encoders[level](
                    cls_feat, feat_3d_up)
                reg_feat = self.reg_channel_encoders[level](
                    reg_feat, feat_3d_up)

            # head
            cls_feat, reg_feat = self.heads[level](cls_feat, reg_feat)

            # pred
            conf_pred: torch.Tensor = self.conf_preds[level](reg_feat)
            cls_pred: torch.Tensor = self.cls_preds[level](cls_feat)
            reg_pred: torch.Tensor = self.reg_preds[level](reg_feat)

            # generate anchors
            fmp_size = conf_pred.shape[-2:]
            anchors = self.generate_anchors(
                fmp_size, self.stride[level], conf_pred.device)

            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            conf_pred = conf_pred.permute(
                0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # decode box: [M, 4]
            box_pred = self.decode_boxes(anchors, reg_pred, self.stride[level])

            all_conf_preds.append(conf_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        # output dict
        outputs = {
            "pred_conf": all_conf_preds,       # List(Tensor) [B, M, 1]
            "pred_cls": all_cls_preds,         # List(Tensor) [B, M, C]
            "pred_box": all_box_preds,         # List(Tensor) [B, M, 4]
            "anchors": all_anchors,            # List(Tensor) [B, M, 2]
            "strides": self.stride             # List(Int)
        }

        return outputs

 def _create_spatial_attention(self, H, W):
        """Create spatial attention for high resolution features"""
        attention = torch.ones((1, 1, 1, H, W), device=self.device)
        if H >= 1080 or W >= 1920:
            # Create grid-based attention
            grid_h, grid_w = torch.meshgrid(
                torch.linspace(0, 1, H, device=self.device),
                torch.linspace(0, 1, W, device=self.device)'
                indexing='ij'  # Add indexing parameter
            )
            attention = torch.sigmoid((grid_h + grid_w) / 2).view(1, 1, 1, H, W)
        return attention

    def _process_high_res_features(self, feat):
        """Process high resolution features"""
        if feat.size(-1) >= 240:  # Only for larger feature maps
            attention = F.adaptive_avg_pool2d(feat, 1)
            attention = torch.sigmoid(attention)
            feat = feat * attention
        return feat
