# SPP block with CSP module
class SPPBlockCSP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        expand_ratio: float = 0.5,
        pooling_size: List[int] | int = [7, 13, 19],  # Increased pool sizes for larger resolution
        act_type: ACTIVATION = 'lrelu',
        norm_type: NORM = 'BN',
        depthwise: bool = False
        ):
        super(SPPBlockCSP, self).__init__()
        # Increase intermediate dimensions for higher resolution
        inter_dim = int(in_dim * expand_ratio * 1.5)  # Increased by 1.5x
        
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        
        # Add an additional conv layer for better feature extraction
        self.m = nn.Sequential(
            Conv(
                inter_dim, inter_dim, k=3, p=1, 
                act_type=act_type, norm_type=norm_type, 
                depthwise=depthwise
            ),
            Conv(
                inter_dim, inter_dim, k=1,
                act_type=act_type, norm_type=norm_type
            ),
            SPP(
                inter_dim, 
                inter_dim, 
                expand_ratio=1.0, 
                pooling_size=pooling_size, 
                act_type=act_type, 
                norm_type=norm_type
            ),
            Conv(
                inter_dim, inter_dim, k=3, p=1, 
                act_type=act_type, norm_type=norm_type, 
                depthwise=depthwise
            )
        )
        
        # Add squeeze-and-excitation for better feature refinement
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(inter_dim * 2, inter_dim // 16, k=1),
            nn.ReLU(inplace=True),
            Conv(inter_dim // 16, inter_dim * 2, k=1),
            nn.Sigmoid()
        )
        
        self.cv3 = Conv(inter_dim * 2, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        
        # Concatenate and apply channel attention
        concat_feat = torch.cat([x1, x3], dim=1)
        se_weight = self.se(concat_feat)
        concat_feat = concat_feat * se_weight
        
        y = self.cv3(concat_feat)
        return y


# SPPF with modifications for higher resolution
class SPPF(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: List[int] | int = 7):  # Increased kernel size
        super().__init__()
        inter_dim = in_dim // 2  # hidden channels
        self.cv1 = Conv(in_dim, inter_dim, k=1)
        
        # Add an additional parallel branch for multi-scale feature extraction
        self.parallel_conv = Conv(inter_dim, inter_dim, k=3, p=1)
        
        self.cv2 = Conv(inter_dim * 5, out_dim, k=1)  # Adjusted for additional branch
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        parallel_feat = self.parallel_conv(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        return self.cv2(torch.cat((x, parallel_feat, y1, y2, y3), 1))


def build_neck(
    model_name: NECK,
    expand_ratio: float,
    pooling_size: List[int] | int,
    neck_act: ACTIVATION,
    neck_norm: NORM,
    neck_depthwise: bool,
    in_dim: int, 
    out_dim: int
):
    # Adjust input dimension for higher resolution
    in_dim = int(in_dim * 1.5)  # Increase input dimension
    out_dim = int(out_dim * 1.5)  # Increase output dimension
    
    validate_literal_types(model_name, NECK)
    
    if model_name == 'spp_block_csp':
        pooling_size = [7, 13, 19]  # Adjusted pool sizes for higher resolution
