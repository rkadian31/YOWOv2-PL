import torch
import torch.nn as nn
import torch.nn.functional as F

from yowo.models.basic.utils import Conv as Conv2d
from yowo.models.basic.types import (
    ACTIVATION,
    NORM
)

class DecoupledHead(nn.Module):
    def __init__(
        self,
        num_cls_head: int,
        num_reg_head: int,
        head_act: ACTIVATION,
        head_norm: NORM,
        head_dim: int,
        head_depthwise: bool,
        high_resolution: bool = False
    ):
        super().__init__()
        print('==============================')
        print('Head: Decoupled Head')
        print(f'High Resolution Mode: {high_resolution}')

        self.num_cls_head = num_cls_head
        self.num_reg_head = num_reg_head
        self.act_type = head_act
        self.norm_type = head_norm
        self.head_dim = head_dim * (1.5 if high_resolution else 1)
        self.depthwise = head_depthwise
        self.high_resolution = high_resolution

        # Efficient feature processing for high resolution
        if high_resolution:
            self.downsample = nn.Sequential(
                nn.AdaptiveAvgPool2d((None, None)),
                Conv2d(head_dim, head_dim, k=1, 
                      act_type=head_act, norm_type=head_norm)
            )

        # Classification head with resolution-aware design
        self.cls_head = self._build_branch(
            num_cls_head, 
            is_classifier=True
        )

        # Regression head with resolution-aware design
        self.reg_head = self._build_branch(
            num_reg_head, 
            is_classifier=False
        )

        # Resolution-specific attention modules
        if high_resolution:
            self.cls_attention = self._build_attention_module(self.head_dim)
            self.reg_attention = self._build_attention_module(self.head_dim)

    def _build_branch(self, num_layers: int, is_classifier: bool):
        """Builds a resolution-aware head branch"""
        layers = []
        in_dim = self.head_dim
        
        for i in range(num_layers):
            # Add efficient convolution for high resolution
            conv = Conv2d(
                in_dim,
                self.head_dim,
                k=3 if not self.high_resolution else 1,
                p=1 if not self.high_resolution else 0,
                s=1,
                act_type=self.act_type,
                norm_type=self.norm_type,
                depthwise=self.depthwise and not self.high_resolution
            )
            
            layers.append(conv)
            
            # Add regularization for high resolution
            if self.high_resolution:
                if is_classifier:
                    layers.append(nn.Dropout2d(0.1))
                else:
                    layers.append(nn.GroupNorm(8, self.head_dim))
            
            in_dim = self.head_dim
        
        return nn.Sequential(*layers)

    def _build_attention_module(self, channels):
        """Builds an efficient attention module for high resolution"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

    def _process_high_res_features(self, x):
        """Efficiently processes high resolution features"""
        B, C, H, W = x.shape
        if H * W > 128 * 128:
            # Adaptive pooling for very large feature maps
            scale_factor = min(1.0, (128 * 128) / (H * W))
            new_h = int(H * scale_factor ** 0.5)
            new_w = int(W * scale_factor ** 0.5)
            x = F.adaptive_avg_pool2d(x, (new_h, new_w))
        return x

    def forward(self, cls_feat, reg_feat):
        """Forward pass with resolution-aware processing"""
        _, _, H, W = cls_feat.shape
        is_high_res = H >= 270 or W >= 480  # Quarter of 1080p

        if is_high_res and self.high_resolution:
            # Process high resolution features
            cls_feat = self._process_high_res_features(cls_feat)
            reg_feat = self._process_high_res_features(reg_feat)

            # Apply attention mechanisms
            cls_feat = cls_feat * self.cls_attention(cls_feat)
            reg_feat = reg_feat * self.reg_attention(reg_feat)

        # Process through head branches
        cls_feats = self.cls_head(cls_feat)
        reg_feats = self.reg_head(reg_feat)

        if is_high_res and self.high_resolution:
            # Final refinement for high resolution
            cls_feats = F.dropout2d(cls_feats, p=0.1, training=self.training)
            reg_feats = F.dropout2d(reg_feats, p=0.1, training=self.training)

        return cls_feats, reg_feats


def build_head(
    num_cls_heads: int,
    num_reg_heads: int,
    head_act: str,
    head_norm: str,
    head_dim: int,
    head_depthwise: bool,
    input_size: tuple = None  # Add input size parameter
):
    # Determine if high resolution mode should be enabled
    high_resolution = False
    if input_size is not None:
        h, w = input_size
        high_resolution = h >= 1080 or w >= 1920

    return DecoupledHead(
        num_cls_heads,
        num_reg_heads,
        head_act,
        head_norm,
        head_dim,
        head_depthwise,
        high_resolution=high_resolution
    )
