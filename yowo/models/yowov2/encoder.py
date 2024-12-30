# Channel Self Attention Module
class CSAM(nn.Module):
    def __init__(self):
        super(CSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        # Added adaptive pooling for high resolution
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.up_sample = nn.Upsample(scale_factor=None, mode='bilinear', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Downsample for attention computation
        if H >= 1080 and W >= 1920:
            x_down = self.avg_pool(x)
            B, C, H_down, W_down = x_down.size()
            
            # Compute attention on downsampled feature
            query = x_down.view(B, C, -1)
            key = x_down.view(B, C, -1).permute(0, 2, 1)
            value = x_down.view(B, C, -1)

            energy = torch.bmm(query, key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
            attention = self.softmax(energy_new)

            out = torch.bmm(attention, value)
            out = out.view(B, C, H_down, W_down)
            
            # Upsample back to original size
            self.up_sample.size = (H, W)
            out = self.up_sample(out)
        else:
            # Original implementation for lower resolutions
            query = x.view(B, C, -1)
            key = x.view(B, C, -1).permute(0, 2, 1)
            value = x.view(B, C, -1)

            energy = torch.bmm(query, key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
            attention = self.softmax(energy_new)

            out = torch.bmm(attention, value)
            out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


# Spatial Self Attention Module
class SSAM(nn.Module):
    def __init__(self):
        super(SSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        # Added adaptive pooling for high resolution
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(64, 64))
        self.up_sample = nn.Upsample(scale_factor=None, mode='bilinear', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Downsample for attention computation
        if H >= 1080 and W >= 1920:
            x_down = self.avg_pool(x)
            B, C, H_down, W_down = x_down.size()
            
            # Compute attention on downsampled feature
            query = x_down.view(B, C, -1).permute(0, 2, 1)
            key = x_down.view(B, C, -1)
            value = x_down.view(B, C, -1).permute(0, 2, 1)

            energy = torch.bmm(query, key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
            attention = self.softmax(energy_new)

            out = torch.bmm(attention, value)
            out = out.permute(0, 2, 1).contiguous().view(B, C, H_down, W_down)
            
            # Upsample back to original size
            self.up_sample.size = (H, W)
            out = self.up_sample(out)
        else:
            # Original implementation for lower resolutions
            query = x.view(B, C, -1).permute(0, 2, 1)
            key = x.view(B, C, -1)
            value = x.view(B, C, -1).permute(0, 2, 1)

            energy = torch.bmm(query, key)
            energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
            attention = self.softmax(energy_new)

            out = torch.bmm(attention, value)
            out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        out = self.gamma * out + x
        return out


# Channel Encoder
class ChannelEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type: Optional[ACTIVATION] = None, norm_type: Optional[NORM] = None):
        super().__init__()
        # Increased intermediate dimensions for high resolution
        mid_dim = out_dim * 2
        
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, mid_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(mid_dim, mid_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            CSAM(),
            Conv2d(mid_dim, mid_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(mid_dim, out_dim, kernel_size=1)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.fuse_convs(x)


# Spatial Encoder
class SpatialEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type: Optional[ACTIVATION] = None, norm_type: Optional[NORM] = None):
        super().__init__()
        # Increased intermediate dimensions for high resolution
        mid_dim = out_dim * 2
        
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, mid_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(mid_dim, mid_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SSAM(),
            Conv2d(mid_dim, mid_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(mid_dim, out_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.fuse_convs(x)
