import math
from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from yowo.utils.validate import validate_literal_types
from .types import (
    PADDING_MODE,
    ACTIVATION,
    NORM,
    NORM_3D
)


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_conv2d(
    c1: int,
    c2: int,
    k: Union[int, Tuple[int, int]],
    p: Union[int, Tuple[int, int]],
    s: Union[int, Tuple[int, int]],
    d: Union[int, Tuple[int, int]],
    g: int,
    padding_mode: PADDING_MODE = 'ZERO',
    bias: bool = False,
    high_resolution: bool = False
):
    validate_literal_types(padding_mode, PADDING_MODE)
    
    # Adjust channels for high resolution
    if high_resolution:
        c2 = int(c2 * math.sqrt(1920 * 1080 / (224 * 224)))
    
    if padding_mode == 'ZERO':
        conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
    elif padding_mode == 'SAME':
        conv = Conv2dSamePadding(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
    
    # Initialize weights considering high resolution
    if high_resolution:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    
    return conv


def get_activation(act_type: Optional[ACTIVATION] = None):
    if act_type is None:
        return nn.Identity()
    
    validate_literal_types(act_type, ACTIVATION)
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)


def get_norm(dim: int, norm_type: Optional[NORM] = None, high_resolution: bool = False):
    if norm_type is None:
        return nn.Identity()
    
    validate_literal_types(norm_type, NORM)
    if norm_type == 'BN':
        if high_resolution:
            return nn.SyncBatchNorm(dim)  # Better for high resolution
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        num_groups = 32 if not high_resolution else 64
        return nn.GroupNorm(num_groups=num_groups, num_channels=dim)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(dim, track_running_stats=True)


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_cache = {}
        
    def _compute_padding(self, input_size):
        height, width = input_size
        stride_h, stride_w = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        kernel_h, kernel_w = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        dilation_h, dilation_w = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)

        output_h = math.ceil(height / stride_h)
        output_w = math.ceil(width / stride_w)

        pad_h = max(0, (output_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - height)
        pad_w = max(0, (output_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - width)

        return (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    def forward(self, x):
        input_size = (x.size(-2), x.size(-1))
        if input_size not in self.pad_cache:
            self.pad_cache[input_size] = self._compute_padding(input_size)
        
        padding = self.pad_cache[input_size]
        return super().forward(F.pad(x, padding))


class Conv(nn.Module):
    def __init__(
        self, 
        c1: int,
        c2: int,
        k: int = 1,
        p: int = 0,
        s: int = 1,
        d: int = 1,
        act_type: Optional[ACTIVATION] = None,
        norm_type: Optional[NORM] = None,
        padding_mode: PADDING_MODE = 'SAME',
        depthwise: bool = False,
        high_resolution: bool = False
    ):
        super().__init__()
        self.high_resolution = high_resolution
        convs = []
        add_bias = False if norm_type else True

        # Resolution-aware channel scaling
        if high_resolution:
            scale_factor = math.sqrt(1920 * 1080 / (224 * 224))
            c2 = int(c2 * scale_factor)

        if depthwise:
            # Depthwise conv with resolution awareness
            convs.append(
                get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1,
                          padding_mode=padding_mode, bias=add_bias,
                          high_resolution=high_resolution)
            )
            if norm_type:
                convs.append(get_norm(c1, norm_type, high_resolution))
            if act_type:
                convs.append(get_activation(act_type))
            
            # Pointwise conv
            convs.append(
                get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1,
                          bias=add_bias, high_resolution=high_resolution)
            )
            if norm_type:
                convs.append(get_norm(c2, norm_type, high_resolution))
            if act_type:
                convs.append(get_activation(act_type))
        else:
            convs.append(
                get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1,
                          padding_mode=padding_mode, bias=add_bias,
                          high_resolution=high_resolution)
            )
            if norm_type:
                convs.append(get_norm(c2, norm_type, high_resolution))
            if act_type:
                convs.append(get_activation(act_type))

        self.convs = nn.Sequential(*convs)
        
        # Add spatial attention for high resolution
        if high_resolution:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, c2 // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2 // 16, c2, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        x = self.convs(x)
        if self.high_resolution and x.size(-1) >= 480:  # Only apply attention for larger feature maps
            x = x * self.attention(x)
        return x

class Conv3d(nn.Module):
    def __init__(
        self, 
        c1,                                             
        c2,                                             
        k=1,                                            
        p=0,                                            
        s=1,                                            
        d=1,                                            
        g=1,
        act_type: Optional[ACTIVATION] = None,          
        norm_type: Optional[NORM_3D] = None,           
        depthwise=False):
        super(Conv3d, self).__init__()
        convs = []
        add_bias = False if norm_type else True

        # Added resolution-aware channel scaling
        c2 = int(c2 * 1.5)  # Increase channels for higher resolution

        if depthwise:
            assert c1 == c2, "In depthwise conv, the in_dim (c1) should be equal to out_dim (c2)."
            # Modified kernel size for higher resolution
            k = k if isinstance(k, (list, tuple)) else (k, k+2, k+2)
            p = p if isinstance(p, (list, tuple)) else (p, p+1, p+1)
            
            convs.append(get_conv3d(c1, c2, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            if norm_type:
                convs.append(get_norm3d(c2, norm_type))
            if act_type:
                convs.append(get_activation(act_type))
            # Modified pointwise conv
            convs.append(get_conv3d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm3d(c2, norm_type))
            if act_type:
                convs.append(get_activation(act_type))
        else:
            # Modified kernel size for higher resolution
            k = k if isinstance(k, (list, tuple)) else (k, k+2, k+2)
            p = p if isinstance(p, (list, tuple)) else (p, p+1, p+1)
            
            convs.append(get_conv3d(c1, c2, k=k, p=p, s=s, d=d, g=g, bias=add_bias))
            if norm_type:
                convs.append(get_norm3d(c2, norm_type))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        # Add resolution check
        _, _, _, h, w = x.shape
        if h >= 1080 and w >= 1920:
            # Optional: Add channel attention for high-resolution inputs
            attention = F.adaptive_avg_pool3d(x, 1)
            attention = torch.sigmoid(attention)
            x = x * attention
        return self.convs(x)

