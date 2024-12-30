from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from yowo.utils.validate import validate_literal_types
from yowo.models.basic.utils import Conv
from yowo.models.basic.types import (
    NORM,
    ACTIVATION
)
from .types import FPN_SIZE


class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's head
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        fpn_size: FPN_SIZE = 'large', 
        depthwise: bool = False, 
        act_type: ACTIVATION = 'silu', 
        norm_type: NORM = 'BN'
    ):
        super(ELANBlock, self).__init__()
        
 # Adjust dimensions for high resolution
        if high_resolution:
            in_dim = int(in_dim * 1.5)
            out_dim = int(out_dim * 1.5)
            
        if fpn_size == 'tiny' or fpn_size == 'nano':
            e1, e2 = (0.25, 1.0) if not high_resolution else (0.375, 1.5)
            width = 2
            depth = 1
        elif fpn_size == 'large':
            e1, e2 = (0.5, 0.5) if not high_resolution else (0.75, 0.75)
            width = 4
            depth = 1
        elif fpn_size == 'huge':
            e1, e2 = (0.5, 0.5) if not high_resolution else (0.75, 0.75)
            width = 4
            depth = 2

        inter_dim = int(in_dim * e1)
        inter_dim2 = int(inter_dim * e2) 
        
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv3 = nn.ModuleList()
        
        for idx in range(width):
            if idx == 0:
                cvs = [Conv(inter_dim, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            else:
                cvs = [Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)]
            # deeper
            if depth > 1:
                for _ in range(1, depth):
                    cvs.append(Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise))
                self.cv3.append(nn.Sequential(*cvs))
            else:
                self.cv3.append(cvs[0])

        self.out = Conv(inter_dim*2+inter_dim2*len(self.cv3), out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        inter_outs = [x1, x2]
        for m in self.cv3:
            y1 = inter_outs[-1]
            y2 = m(y1)
            inter_outs.append(y2)

        # [B, C_in, H, W] -> [B, C_out, H, W]
        out = self.out(torch.cat(inter_outs, dim=1))

        return out


class DownSample(nn.Module):
    def __init__(self, in_dim: int, depthwise: bool = False, act_type: ACTIVATION = 'silu', norm_type: NORM = 'BN'):
        super().__init__()
        inter_dim = in_dim
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out
class DownSample(nn.Module):
    def __init__(
        self,
        in_dim: int,
        depthwise: bool = False,
        act_type: ACTIVATION = 'silu',
        norm_type: NORM = 'BN',
        high_resolution: bool = False
    ):
        super().__init__()
        # Adjust dimensions for high resolution
        if high_resolution:
            in_dim = int(in_dim * 1.5)
            
        inter_dim = in_dim
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        
        # Enhanced convolution for high resolution
        if high_resolution:
            self.cv2 = nn.Sequential(
                Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
                Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type,
                     norm_type=norm_type, depthwise=depthwise),
                Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type,
                     norm_type=norm_type, depthwise=depthwise)
            )
        else:
            self.cv2 = nn.Sequential(
                Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
                Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type,
                     norm_type=norm_type, depthwise=depthwise)
            )
        
        self.high_resolution = high_resolution

    def forward(self, x):
        """
        Enhanced forward pass with high resolution support
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            out: Output tensor of shape [B, 2C, H//2, W//2]
        """
        if self.high_resolution:
            # Enhanced processing for high resolution
            # First branch: maxpool -> conv
            x1 = self.cv1(self.mp(x))
            
            # Second branch: enhanced conv path
            x2 = self.cv2(x)
            
            # Optional: Add additional processing for high resolution features
            if x1.size(2) >= 270:  # Check if feature map is still large (1080/4)
                # Apply attention or additional processing
                x1 = self._process_high_res_features(x1)
                x2 = self._process_high_res_features(x2)
            
        else:
            # Original processing
            x1 = self.cv1(self.mp(x))
            x2 = self.cv2(x)

        # Concatenate along channel dimension
        out = torch.cat([x1, x2], dim=1)
        
        return out

    def _process_high_res_features(self, x):
        """
        Additional processing for high resolution features
        """
        B, C, H, W = x.shape
        
        # Apply spatial attention if feature map is large
        if H * W > 128 * 128:
            # Simple spatial attention
            spatial_att = F.adaptive_avg_pool2d(x, (1, 1))
            spatial_att = torch.sigmoid(spatial_att)
            x = x * spatial_att
            
        return x

# PaFPN-ELAN
class PaFPNELAN(nn.Module):
    def __init__(
        self, 
        in_dims: List[int] = [768, 1536, 1536], 
        out_dim: int = 384,
        fpn_size: FPN_SIZE = 'large',
        depthwise: bool = False,
        norm_type: NORM = 'BN',
        act_type: ACTIVATION = 'silu',
        high_resolution: bool = False
    ):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.high_resolution = high_resolution
        
        # Adjust dimensions for high resolution
        if high_resolution:
            c3, c4, c5 = [int(dim * 1.5) for dim in in_dims]
        else:
            c3, c4, c5 = in_dims

        # Determine network width based on resolution and fpn_size
        if high_resolution:
            if fpn_size == 'tiny':
                width = 0.75
            elif fpn_size == 'nano':
                width = 0.75
            elif fpn_size == 'large':
                width = 2.0
            elif fpn_size == 'huge':
                width = 2.5
        else:
            if fpn_size == 'tiny':
                width = 0.5
            elif fpn_size == 'nano':
                assert depthwise
                width = 0.5
            elif fpn_size == 'large':
                width = 1.5
            elif fpn_size == 'huge':
                width = 2.0

        # Top down
        ## P5 -> P4
        self.cv1 = Conv(c5, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(c4, int(256 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_1 = ELANBlock(
            in_dim=int(256 * width) + int(256 * width),
            out_dim=int(256 * width),
            fpn_size=fpn_size,
            depthwise=depthwise,
            norm_type=norm_type,
            act_type=act_type,
            high_resolution=high_resolution
        )

        ## P4 -> P3
        self.cv3 = Conv(int(256 * width), int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.cv4 = Conv(c3, int(128 * width), k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_2 = ELANBlock(
            in_dim=int(128 * width) + int(128 * width),
            out_dim=int(128 * width),
            fpn_size=fpn_size,
            depthwise=depthwise,
            norm_type=norm_type,
            act_type=act_type,
            high_resolution=high_resolution
        )

        # Bottom up
        # P3 -> P4
        if fpn_size == 'large' or fpn_size == 'huge':
            self.mp1 = DownSample(
                int(128 * width),
                act_type=act_type,
                norm_type=norm_type,
                depthwise=depthwise,
                high_resolution=high_resolution
            )
        elif fpn_size == 'tiny':
            self.mp1 = Conv(
                int(128 * width),
                int(256 * width),
                k=3, p=1, s=2,
                act_type=act_type,
                norm_type=norm_type,
                depthwise=depthwise
            )
        elif fpn_size == 'nano':
            self.mp1 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),
                Conv(int(128 * width), int(256 * width), k=1, act_type=act_type, norm_type=norm_type)
            )

        self.head_elan_3 = ELANBlock(
            in_dim=int(256 * width) + int(256 * width),
            out_dim=int(256 * width),
            fpn_size=fpn_size,
            depthwise=depthwise,
            norm_type=norm_type,
            act_type=act_type,
            high_resolution=high_resolution
        )

        # P4 -> P5
        if fpn_size == 'large' or fpn_size == 'huge':
            self.mp2 = DownSample(
                int(256 * width),
                act_type=act_type,
                norm_type=norm_type,
                depthwise=depthwise,
                high_resolution=high_resolution
            )
        elif fpn_size == 'tiny':
            self.mp2 = Conv(
                int(256 * width),
                int(512 * width),
                k=3, p=1, s=2,
                act_type=act_type,
                norm_type=norm_type,
                depthwise=depthwise
            )
        elif fpn_size == 'nano':
            self.mp2 = nn.Sequential(
                nn.MaxPool2d((2, 2), 2),
                Conv(int(256 * width), int(512 * width), k=1, act_type=act_type, norm_type=norm_type)
            )

        self.head_elan_4 = ELANBlock(
            in_dim=int(512 * width) + c5,
            out_dim=int(512 * width),
            fpn_size=fpn_size,
            depthwise=depthwise,
            norm_type=norm_type,
            act_type=act_type,
            high_resolution=high_resolution
        )

        # Head convolutions
        self.head_conv_1 = Conv(
            int(192 * width), int(384 * width), k=3, p=1,
            act_type=act_type, norm_type=norm_type, depthwise=depthwise
        )
        self.head_conv_2 = Conv(
            int(384 * width), int(768 * width), k=3, p=1,
            act_type=act_type, norm_type=norm_type, depthwise=depthwise
        )
        self.head_conv_3 = Conv(
            int(768 * width), int(1536 * width), k=3, p=1,
            act_type=act_type, norm_type=norm_type, depthwise=depthwise
        )

        # Output projection layers
        if self.out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(
                    in_dim, self.out_dim, k=1,
                    norm_type=norm_type, act_type=act_type
                )
                for in_dim in [int(256 * width), int(512 * width), int(1024 * width)]
            ])

        # High resolution specific layers
        if high_resolution:
            self.hr_attention = nn.ModuleList([
                self._make_hr_attention(dim) 
                for dim in [int(384 * width), int(768 * width), int(1536 * width)]
            ])

    def _make_hr_attention(self, dim):
        """Creates attention module for high resolution features"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(dim, dim // 16, k=1, act_type='relu'),
            Conv(dim // 16, dim, k=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Forward pass with high resolution support
        Args:
            features: List of features [c3, c4, c5]
        Returns:
            List of processed features
        """
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)

        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # P3 -> P4
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.head_elan_3(c15)

        # P4 -> P5
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.head_elan_4(c18)

        # Head convolutions
        c20 = self.head_conv_1(c13)
        c21 = self.head_conv_2(c16)
        c22 = self.head_conv_3(c19)

        out_feats = [c20, c21, c22]  # [P3, P4, P5]

        # Apply high resolution attention if enabled
        if self.high_resolution:
            out_feats = [
                feat * att(feat) 
                for feat, att in zip(out_feats, self.hr_attention)
            ]

        # Output projection layers
        if self.out_dim is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats
        
def build_fpn(
    fpn_size: FPN_SIZE,
    fpn_depthwise: bool,
    fpn_norm: NORM,
    fpn_act: ACTIVATION,
    in_dims: int, 
    out_dim: int,
    high_resolution: bool = False,
    input_size: Optional[Tuple[int, int]] = None
):
    print('==============================')
    print('FPN: pafpn_elan')
    print(f'High Resolution Mode: {high_resolution}')
    
    validate_literal_types(fpn_size, FPN_SIZE)
    
    fpn_net = PaFPNELAN(
        in_dims=in_dims,
        out_dim=out_dim,
        fpn_size=fpn_size,
        depthwise=fpn_depthwise,
        norm_type=fpn_norm,
        act_type=fpn_act,
        high_resolution=high_resolution,
        input_size=input_size
    )                      

    return fpn_net

if __name__ == '__main__':
    fpn_net = build_fpn(
        fpn_size="tiny",
        fpn_depthwise=True,
        fpn_norm="BN",
        fpn_act="lrelu",
        in_dims=1024,
        out_dim=256
    )
