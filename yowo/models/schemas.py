from dataclasses import dataclass
from typing import List, Literal, Tuple, Union
from lightning.pytorch.cli import LRSchedulerCallable


@dataclass
class LossConfig:
    topk_candicate: int = 10
    center_sampling_radius: float = 2.5
    loss_conf_weight: float | int = 1
    loss_cls_weight: float | int = 1
    loss_reg_weight: float | int = 5
    focal_loss: bool = False
    # Added parameters for high resolution
    scale_aware_weight: bool = True
    small_object_weight: float = 2.0
    large_object_weight: float = 1.0
    resolution_aware_loss: bool = True


@dataclass
class LRSChedulerConfig:
    scheduler: LRSchedulerCallable
    interval: Literal["step", "epoch"] = "epoch"
    frequency: int = 1
    # Added parameters for high resolution
    warmup_epochs: int = 5
    resolution_scale_factor: float = 1.5


@dataclass
class WarmupLRConfig:
    name: Literal["linear", "exp", "cosine"] = "linear"
    max_iter: int = 500
    factor: float = 0.00066667
    # Added parameters for high resolution
    high_res_factor: float = 0.001
    warmup_momentum: float = 0.9


@dataclass
class ModelConfig:
    backbone_2d: str
    backbone_3d: str
    pretrained_2d: bool
    pretrained_3d: bool
    head_dim: int
    head_norm: str
    head_act: str
    head_depthwise: bool
    num_classes: int
    stride: List[int]
    # Modified image size to handle different resolutions
    img_size: Union[int, Tuple[int, int]] = 224
    conf_thresh: float = 0.05
    nms_thresh: float = 0.6
    topk: int = 50
    multi_hot: bool = False
    num_cls_heads: int = 2
    num_reg_heads: int = 2
    use_aggregate_feat: bool = False
    use_blurpool: bool = False
    
    # Added parameters for high resolution
    high_resolution: bool = False
    resolution_aware_features: bool = True
    feature_scales: List[int] = None
    anchor_scales: List[float] = None
    additional_backbone_stages: int = 0
    use_fpn: bool = False
    fpn_channels: int = 256
    use_deformable: bool = False
    
    # Memory optimization parameters
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    batch_division_factor: int = 1
    
    # High resolution specific parameters
    min_object_size: int = 8
    max_object_size: int = 1024
    anchor_generator_config: dict = None
    
    def __post_init__(self):
        # Auto-configure for high resolution
        if isinstance(self.img_size, tuple):
            self.high_resolution = max(self.img_size) >= 1080
        elif isinstance(self.img_size, int):
            self.high_resolution = self.img_size >= 1080
            
        if self.high_resolution:
            # Adjust parameters for high resolution
            if self.feature_scales is None:
                self.feature_scales = [8, 16, 32, 64, 128]
            
            if self.anchor_scales is None:
                self.anchor_scales = [32, 64, 128, 256, 512]
                
            # Increase head dimensions for high resolution
            self.head_dim *= 2
            
            # Adjust strides for high resolution
            self.stride = [s * 2 for s in self.stride]
            
            # Enable memory optimization features
            self.gradient_checkpointing = True
            self.mixed_precision = True
            
            # Configure anchor generator for high resolution
            if self.anchor_generator_config is None:
                self.anchor_generator_config = {
                    'aspect_ratios': [0.5, 1.0, 2.0],
                    'scales_per_octave': 3,
                    'base_sizes': self.anchor_scales
                }
                
            # Enable FPN for high resolution
            self.use_fpn = True
            
            # Adjust batch processing
            self.batch_division_factor = 2
            
            # Enable deformable convolutions for high resolution
            self.use_deformable = True


@dataclass
class HighResolutionConfig:
    """New configuration class for high resolution specific parameters"""
    enable_tiling: bool = True
    tile_size: int = 640
    tile_overlap: float = 0.2
    adaptive_pooling: bool = True
    feature_refinement: bool = True
    multi_scale_inference: bool = True
    nms_threshold_small: float = 0.5
    nms_threshold_large: float = 0.7
    score_threshold_small: float = 0.05
    score_threshold_large: float = 0.1
    max_detections_per_tile: int = 100
    merge_strategy: Literal["nms", "soft-nms", "weighted"] = "weighted"
    
    # Multi-scale training parameters
    scale_range: Tuple[float, float] = (0.8, 1.2)
    scale_step: float = 0.2
    min_scale_area: int = 32 * 32
    max_scale_area: int = 1024 * 1024
