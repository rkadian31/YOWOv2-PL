#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
from typing import Optional, Union, Literal


def constant_init(module: nn.Module, val: float, bias: float = 0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(
    module: nn.Module,
    gain: float = 1,
    bias: float = 0,
    distribution: Literal['uniform', 'normal'] = 'normal',
    high_resolution: bool = False
):
    assert distribution in ['uniform', 'normal']
    
    # Adjust gain for high resolution
    if high_resolution:
        gain *= math.sqrt(2.0)
    
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
        
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(
    module: nn.Module,
    mean: float = 0,
    std: float = 1,
    bias: float = 0,
    high_resolution: bool = False
):
    # Adjust std for high resolution
    if high_resolution:
        std *= math.sqrt(2.0)
    
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(
    module: nn.Module,
    a: float = 0,
    b: float = 1,
    bias: float = 0,
    high_resolution: bool = False
):
    # Adjust range for high resolution
    if high_resolution:
        scale = math.sqrt(2.0)
        a *= scale
        b *= scale
    
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: Literal['fan_in', 'fan_out'] = 'fan_out',
    nonlinearity: str = 'relu',
    bias: float = 0,
    distribution: Literal['uniform', 'normal'] = 'normal',
    high_resolution: bool = False
):
    assert distribution in ['uniform', 'normal']
    
    # Adjust parameters for high resolution
    if high_resolution:
        if mode == 'fan_out':
            a *= math.sqrt(2.0)
    
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity
        )
        
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def scale_aware_init(
    module: nn.Module,
    input_size: tuple,
    std_scale: float = 0.01
) -> None:
    """Scale-aware initialization for high resolution."""
    if isinstance(module, nn.Conv2d):
        # Calculate scale based on input resolution
        resolution_scale = math.sqrt(input_size[0] * input_size[1] / (224 * 224))
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        std = math.sqrt(2.0 / fan_out) * resolution_scale
        nn.init.normal_(module.weight, mean=0.0, std=std)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def resolution_aware_init(
    module: nn.Module,
    input_size: tuple,
    base_size: int = 224
) -> None:
    """Resolution-aware initialization."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        scale = math.sqrt(input_size[0] * input_size[1] / (base_size * base_size))
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
        std = math.sqrt(2.0 / (fan_in + fan_out)) * scale
        nn.init.normal_(module.weight, mean=0.0, std=std)
        
        if module.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound)


def init_weights(
    m: nn.Module,
    zero_init_final_gamma: bool = False,
    high_resolution: bool = False,
    input_size: Optional[tuple] = None
) -> None:
    """Enhanced ResNet-style weight initialization with resolution awareness."""
    if isinstance(m, nn.Conv2d):
        if high_resolution and input_size is not None:
            scale_aware_init(m, input_size)
        else:
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = (
            hasattr(m, "final_bn") and m.final_bn and zero_init_final_gamma
        )
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
        
    elif isinstance(m, nn.Linear):
        if high_resolution and input_size is not None:
            resolution_aware_init(m, input_size)
        else:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()


def high_resolution_init(
    model: nn.Module,
    input_size: tuple,
    method: Literal['scale_aware', 'resolution_aware'] = 'scale_aware'
) -> None:
    """Initialize model weights considering high resolution input."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if method == 'scale_aware':
                scale_aware_init(m, input_size)
            else:
                resolution_aware_init(m, input_size)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
