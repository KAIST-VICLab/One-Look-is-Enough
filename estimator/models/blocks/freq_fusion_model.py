import torch
import torch.nn as nn
import torch.nn.functional as F
from estimator.registry import MODELS
import pytorch_wavelets as pw

class DoubleConvWOBN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Upv1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None, align_corners=True):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if mid_channels is not None:
            self.conv = DoubleConvWOBN(in_channels, out_channels, mid_channels)
        else:
            self.conv = DoubleConvWOBN(in_channels, out_channels, in_channels)
        
        self.__algin_corners = align_corners

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        #?align
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=self.__algin_corners)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class WaveletFusionConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_nonlinearity=False):
        super(WaveletFusionConv, self).__init__()
        self.in_channels = in_channels
        # self.out_channels = out_channels
        
        # Convolution layers to process frequency components
        self.low_freq_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False) if not use_nonlinearity else nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))
        self.high_freq_conv = nn.Conv2d(in_channels * 6, in_channels * 3, kernel_size=3, padding=1, bias=False, groups=3) if not use_nonlinearity else nn.Sequential(nn.Conv2d(in_channels * 6, in_channels * 3, kernel_size=3, padding=1, bias=False, groups=3), nn.ReLU(inplace=True))

        # # Final fusion layer
        # self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False)

        self.wave_transofrm = HaarWaveletTransform(in_channels)
    
    @staticmethod
    def pad_to_even(x):
        _, _, H, W = x.shape
        pad_H = 1 if H % 2 != 0 else 0
        pad_W = 1 if W % 2 != 0 else 0
        return F.pad(x, (0, pad_W, 0, pad_H))  # (왼쪽, 오른쪽, 위, 아래) 패딩

    def forward(self, feat_c, feat_f):
        """
        Args:
            feat_c: Coarse feature (B, C, H, W)
            feat_f: Fine feature (B, C, H, W)
        Returns:
            Fused feature map (B, C, H, W)
        """
        # Ensure the input sizes match
        _, _, h_c, w_c = feat_c.shape
        _, _, h_f, w_f = feat_f.shape
        assert h_c == h_f and w_c == w_f, "Input feature sizes do not match"

        if h_c % 2 != 0 or w_c % 2 != 0:
            feat_c = self.pad_to_even(feat_c)
            feat_f = self.pad_to_even(feat_f)

        # Apply Haar Wavelet Transform
        ll_c, lh_c, hl_c, hh_c = self.wave_transofrm(feat_c)
        ll_f, lh_f, hl_f, hh_f = self.wave_transofrm(feat_f)

        # Low-frequency fusion (coarse-level features dominate)
        low_freq_fusion = torch.cat([ll_c, ll_f], dim=1)
        ll_fusion = self.low_freq_conv(low_freq_fusion)

        # High-frequency fusion (fine-level features dominate)
        high_freq_fusion = torch.cat([lh_c, lh_f, hl_c, hl_f, hh_c, hh_f], dim=1)
        high_freq_out = self.high_freq_conv(high_freq_fusion)
        lh_fusion, hl_fusion, hh_fusion = torch.chunk(high_freq_out, 3, dim=1)

        feat_fused = self.wave_transofrm.inverse(ll_fusion, lh_fusion, hl_fusion, hh_fusion)

        if h_c % 2 != 0 or w_c % 2 != 0:
            feat_fused = feat_fused[:, :, :h_c, :w_c]

        return feat_fused

class HaarWaveletTransform(nn.Module):
    def __init__(self, num_channels, device='cpu'):
        super().__init__()
        self.num_channels = num_channels
        self.device = device

        # Forward Haar Wavelet filters
        haar_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        haar_lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        haar_hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        haar_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        # Register buffers
        self.register_buffer("haar_ll", haar_ll.repeat(num_channels, 1, 1, 1))
        self.register_buffer("haar_lh", haar_lh.repeat(num_channels, 1, 1, 1))
        self.register_buffer("haar_hl", haar_hl.repeat(num_channels, 1, 1, 1))
        self.register_buffer("haar_hh", haar_hh.repeat(num_channels, 1, 1, 1))
        
    
    def forward(self, x):
        """Forward Haar Wavelet Transform."""
        ll = F.conv2d(x, self.haar_ll, stride=2, padding=0, groups=self.num_channels)
        lh = F.conv2d(x, self.haar_lh, stride=2, padding=0, groups=self.num_channels)
        hl = F.conv2d(x, self.haar_hl, stride=2, padding=0, groups=self.num_channels)
        hh = F.conv2d(x, self.haar_hh, stride=2, padding=0, groups=self.num_channels)
        return ll, lh, hl, hh

    def inverse(self, ll, lh, hl, hh):
        """Inverse Haar Wavelet Transform."""
        ll_recon = F.conv_transpose2d(ll, self.haar_ll, stride=2, padding=0, groups=self.num_channels)
        lh_recon = F.conv_transpose2d(lh, self.haar_lh, stride=2, padding=0, groups=self.num_channels)
        hl_recon = F.conv_transpose2d(hl, self.haar_hl, stride=2, padding=0, groups=self.num_channels)
        hh_recon = F.conv_transpose2d(hh, self.haar_hh, stride=2, padding=0, groups=self.num_channels)
        return (ll_recon + lh_recon + hl_recon + hh_recon)

        
@MODELS.register_module()
class Fusion_freq_selective(nn.Module):
    def __init__(
        self, 
        n_channels, 
        inter_resolution = False,
        in_channels=[32, 256, 256, 256, 256, 256],
        using_levels = [0,1,2,3,4,5],
        align_corners=True,
        residual=False,
        bn=True,
        wavelet='haar',
        use_nonlinearity=False
        ):
        
        super(Fusion_freq_selective, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, in_channels[0])
        
        self.down_conv_list = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            lay = Down(in_channels[idx], in_channels[idx+1])
            self.down_conv_list.append(lay)
        
        self.__align_corners = align_corners
            
        in_channels_inv = in_channels[::-1]
        
        self.inter_resolution = inter_resolution 
        self.residual = residual

        self.using_levels = using_levels
        self.fusion_convs = nn.ModuleDict()
        self.wavelet_fusion_convs = nn.ModuleDict()
        for idx in using_levels:
            self.wavelet_fusion_convs[f'{idx}'] = WaveletFusionConv(in_channels_inv[idx], use_nonlinearity=use_nonlinearity)
            # self.wavelet_fusion_convs[f'{idx}'] = DoubleConvWOBN(in_channels_inv[idx]*2, in_channels_inv[idx])
            self.fusion_convs[f'{idx}'] = DoubleConv(in_channels_inv[idx] *3, in_channels_inv[idx]) if bn else DoubleConvWOBN(in_channels_inv[idx] *3, in_channels_inv[idx])
                     
    def forward(self, 
                input_tensor, 
                coarse_feats_roi,
                fine_feats):

        feat_list = []
        
        x = self.inc(input_tensor)
        feat_list.append(x)
        
        for layer in self.down_conv_list:
            x = layer(x)
            feat_list.append(x)
        
        feat_inv_list = feat_list[::-1]
        coarse_feats = coarse_feats_roi
        output = coarse_feats_roi.copy()
        
        #^ feat_enc는 input_tensor로부터 얻은 feature map
        #^ feat_c는 depth-anything(coarse)의 roi feature map
        for idx in self.using_levels:
                
            feat_enc = feat_inv_list[idx]
            feat_c = coarse_feats[idx]
            feat_f = fine_feats[idx]
            
            _, _, h, w = feat_enc.shape
            if h != feat_c.shape[-2] or w != feat_c.shape[-1]:
                #?align
                feat_enc = F.interpolate(feat_enc, size=feat_c.shape[-2:], mode='bilinear', align_corners=self.__align_corners)
            feat_freq_fusion = self.wavelet_fusion_convs[f'{idx}'](feat_c, feat_f)
            # feat_freq_fusion = self.wavelet_fusion_convs[f'{idx}'](torch.cat([feat_c, feat_f], dim=1))
            
            conv_output = self.fusion_convs[f'{idx}'](torch.cat([feat_c, feat_enc, feat_freq_fusion], dim=1))
            # conv_output = self.fusion_convs[f'{idx}'](torch.cat([feat_enc, feat_freq_fusion], dim=1))
            
            feat_out = feat_c + conv_output if self.residual else conv_output
            
            output[idx] = feat_out
                
        return output