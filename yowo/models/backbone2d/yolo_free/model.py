import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import math

from yowo.utils.validate import validate_literal_types
from .backbone import build_backbone
from .neck import build_neck
from .fpn import build_fpn
from .head import build_head
from .types import YOLO_FREE_VERSION
from .config import (
    YOLO_FREE_CONFIG,
    MODEL_URLS
)


class FreeYOLO(nn.Module):
    def __init__(self, config):
        super(FreeYOLO, self).__init__()
        # Resolution config
        self.min_height = 1080
        self.max_width = 1920
        self.high_resolution = True
        
        # Basic Config
        self.cfg = config
        
        # Adjust model capacity for high resolution
        self._adjust_model_capacity()
        
        # Network Parameters
        self._build_network()
        
        # Initialize weights considering high resolution
        self._init_weights()

    def _adjust_model_capacity(self):
        """Adjust model capacity for high resolution"""
        if self.high_resolution:
            # Scale factors based on resolution
            scale = math.sqrt(1920 * 1080 / (224 * 224))
            
            # Adjust dimensions
            self.cfg['neck_dim'] = int(self.cfg['neck_dim'] * scale)
            self.cfg['head_dim'] = int(self.cfg['head_dim'] * scale)
            
            # Adjust FPN dimensions
            self.cfg['fpn_dim'] = [int(dim * scale) for dim in self.cfg['fpn_dim']]

    def _build_network(self):
        """Build network components"""
        # Backbone with resolution awareness
        self.backbone, bk_dim = build_backbone(
            self.cfg['backbone'],
            high_resolution=self.high_resolution
        )

        # Neck with enhanced capacity
        self.neck = build_neck(
            model_name=self.cfg['neck'],
            expand_ratio=self.cfg['expand_ratio'] * (1.5 if self.high_resolution else 1.0),
            pooling_size=self._adjust_pooling_size(),
            neck_act=self.cfg['neck_act'],
            neck_norm=self.cfg['neck_norm'],
            neck_depthwise=self.cfg['neck_depthwise'],
            in_dim=bk_dim[-1], 
            out_dim=self.cfg['neck_dim'],
            high_resolution=self.high_resolution
        )

        # FPN with resolution awareness
        self.fpn = build_fpn(
            fpn_size=self.cfg['fpn_size'],
            fpn_depthwise=self.cfg['fpn_depthwise'],
            fpn_norm=self.cfg['fpn_norm'],
            fpn_act=self.cfg['fpn_act'],
            in_dims=self.cfg['fpn_dim'], 
            out_dim=self.cfg['head_dim'],
            high_resolution=self.high_resolution
        )

        # Non-shared heads with resolution awareness
        self.non_shared_heads = nn.ModuleList([
            build_head(
                num_cls_head=self.cfg['num_cls_head'],
                num_reg_head=self.cfg['num_reg_head'],
                head_act=self.cfg['head_act'],
                head_norm=self.cfg['head_norm'],
                head_dim=self.cfg['head_dim'],
                head_depthwise=self.cfg['head_depthwise'],
                high_resolution=self.high_resolution
            ) 
            for _ in range(len(self.cfg['stride']))
        ])

    def _adjust_pooling_size(self):
        """Adjust pooling size based on resolution"""
        if self.high_resolution:
            return (self.cfg['pooling_size'][0] * 2, 
                   self.cfg['pooling_size'][1] * 2)
        return self.cfg['pooling_size']

    def _init_weights(self):
        """Initialize weights with resolution awareness"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _validate_input_size(self, x):
        """Validate input size and adjust if necessary"""
        _, _, h, w = x.shape
        if h < self.min_height or w > self.max_width:
            raise ValueError(
                f"Input resolution must be at least {self.min_height}x{self.max_width}, "
                f"got {h}x{w}"
            )
        
        # Optional: Add resolution-specific processing
        if h >= 1080 and w >= 1920:
            return self._process_high_res_input(x)
        return x

    def _process_high_res_input(self, x):
        """Process high resolution input"""
        # Optional: Add specific processing for high resolution
        return x

    def forward(self, x):
        # Validate and process input
        x = self._validate_input_size(x)
        
        # Backbone
        feats = self.backbone(x)

        # Neck with resolution awareness
        feats['layer4'] = self.neck(feats['layer4'])

        # FPN
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
        pyramid_feats = self.fpn(pyramid_feats)

        # Non-shared heads
        all_cls_feats = []
        all_reg_feats = []
        
        for feat, head in zip(pyramid_feats, self.non_shared_heads):
            cls_feat, reg_feat = head(feat)
            
            # Optional: Add resolution-specific processing
            if self.high_resolution:
                cls_feat = self._process_high_res_features(cls_feat)
                reg_feat = self._process_high_res_features(reg_feat)
                
            all_cls_feats.append(cls_feat)
            all_reg_feats.append(reg_feat)

        return all_cls_feats, all_reg_feats

    def _process_high_res_features(self, feat):
        """Process features for high resolution"""
        if feat.size(-1) >= 240:  # Only for larger feature maps
            attention = nn.functional.adaptive_avg_pool2d(feat, 1)
            attention = torch.sigmoid(attention)
            feat = feat * attention
        return feat


def build_yolo_free(
    model_name: YOLO_FREE_VERSION = 'yolo_free_large',
    pretrained: bool = False,
    high_resolution: bool = True
):
    validate_literal_types(model_name, YOLO_FREE_VERSION)
    
    # Model config
    cfg = YOLO_FREE_CONFIG[model_name]
    
    # Adjust config for high resolution if needed
    if high_resolution:
        cfg = _adjust_config_for_high_res(cfg)
    
    # Build model
    model = FreeYOLO(cfg)
    feat_dims = [model.cfg['head_dim']] * 3

    # Load pretrained weights
    if pretrained:
        _load_pretrained_weights(model, model_name)

    return model, feat_dims


def _adjust_config_for_high_res(cfg):
    """Adjust config for high resolution"""
    scale = math.sqrt(1920 * 1080 / (224 * 224))
    cfg['neck_dim'] = int(cfg['neck_dim'] * scale)
    cfg['head_dim'] = int(cfg['head_dim'] * scale)
    cfg['fpn_dim'] = [int(dim * scale) for dim in cfg['fpn_dim']]
    return cfg


def _load_pretrained_weights(model, model_name):
    """Load and adjust pretrained weights"""
    url = MODEL_URLS[model_name]
    if url is None:
        print('No pretrained weights available')
        return

    print(f'Loading pretrained weights: {model_name.upper()}')
    checkpoint = load_state_dict_from_url(url, map_location='cpu')
    checkpoint_state_dict = checkpoint.pop('model')
    
    # Adjust weights for high resolution if needed
    if model.high_resolution:
        checkpoint_state_dict = _adjust_weights_for_high_res(
            checkpoint_state_dict,
            model.state_dict()
        )
    
    model.load_state_dict(checkpoint_state_dict, strict=False)
