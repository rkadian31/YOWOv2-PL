import torch
import torch.nn as nn

from yowo.models.basic.utils import Conv as Conv2d


class DecoupledHead(nn.Module):
    def __init__(
        self,
        num_cls_heads: int,
        num_reg_heads: int,
        head_act: str,
        head_norm: str,
        head_dim: int,
        head_depthwise: bool
    ):
        super().__init__()
        print('==============================')
        print('Head: Decoupled Head (High Resolution)')
        
        # Increased dimensions for high resolution
        self.num_cls_heads = num_cls_heads
        self.num_reg_heads = num_reg_heads
        self.act_type = head_act
        self.norm_type = head_norm
        self.head_dim = head_dim * 2  # Doubled head dimension
        self.depthwise = head_depthwise

        # Added adaptive pooling for high resolution features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, None))
        
        # Classification head with increased capacity
        cls_layers = []
        in_dim = self.head_dim
        for i in range(self.num_cls_heads):
            cls_layers.extend([
                Conv2d(
                    in_dim, 
                    self.head_dim,
                    k=3, p=1, s=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    depthwise=self.depthwise
                ),
                nn.Dropout2d(0.1) if i < self.num_cls_heads - 1 else nn.Identity()
            ])
            in_dim = self.head_dim
        self.cls_head = nn.Sequential(*cls_layers)

        # Regression head with increased capacity
        reg_layers = []
        in_dim = self.head_dim
        for i in range(self.num_reg_heads):
            reg_layers.extend([
                Conv2d(
                    in_dim,
                    self.head_dim,
                    k=3, p=1, s=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    depthwise=self.depthwise
                ),
                nn.Dropout2d(0.1) if i < self.num_reg_heads - 1 else nn.Identity()
            ])
            in_dim = self.head_dim
        self.reg_head = nn.Sequential(*reg_layers)

        # Added spatial attention modules
        self.cls_spatial_attention = nn.Sequential(
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.reg_spatial_attention = nn.Sequential(
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Added channel attention modules
        self.cls_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_dim, self.head_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim // 16, self.head_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.reg_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.head_dim, self.head_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim // 16, self.head_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cls_feat, reg_feat):
        # Check resolution
        _, _, h, w = cls_feat.shape
        is_high_res = h >= 1080 and w >= 1920

        if is_high_res:
            # Classification branch with attention
            cls_feats = self.cls_head(cls_feat)
            cls_spatial_att = self.cls_spatial_attention(cls_feats)
            cls_channel_att = self.cls_channel_attention(cls_feats)
            cls_feats = cls_feats * cls_spatial_att * cls_channel_att

            # Regression branch with attention
            reg_feats = self.reg_head(reg_feat)
            reg_spatial_att = self.reg_spatial_attention(reg_feats)
            reg_channel_att = self.reg_channel_attention(reg_feats)
            reg_feats = reg_feats * reg_spatial_att * reg_channel_att
        else:
            # Original processing for lower resolutions
            cls_feats = self.cls_head(cls_feat)
            reg_feats = self.reg_head(reg_feat)

        return cls_feats, reg_feats


def build_head(
    num_cls_heads: int,
    num_reg_heads: int,
    head_act: str,
    head_norm: str,
    head_dim: int,
    head_depthwise: bool
):
    # Adjust head dimensions based on expected input resolution
    return DecoupledHead(
        num_cls_heads,
        num_reg_heads,
        head_act,
        head_norm,
        head_dim,
        head_depthwise
    )
