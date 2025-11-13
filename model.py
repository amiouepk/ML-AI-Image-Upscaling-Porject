import torch
import torch.nn as nn

# --- 1. The Helper Function (Standard 2D Convolution) ---
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias
    )

# --- 2. The Residual Block (Key EDSR Component) ---
class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True):
        super(ResBlock, self).__init__()
        
        # 1. First Conv: n_feats -> n_feats
        self.c1 = default_conv(n_feats, n_feats, kernel_size, bias=bias)
        
        # 2. ReLU Activation
        self.relu = nn.ReLU(inplace=True)
        
        # 3. Second Conv: n_feats -> n_feats
        self.c2 = default_conv(n_feats, n_feats, kernel_size, bias=bias)
        
        # NOTE: EDSR paper explicitly states to *remove* Batch Normalization (BN)
        # We also use a scaling factor of 0.1 at the end for stability, 
        # as suggested in the original paper, though it can be optional.

    def forward(self, x):
        residual = x
        out = self.c2(self.relu(self.c1(x)))
        
        # Add a residual scaling factor (often set to 0.1) for training stability
        out = out * 0.1 + residual 
        
        return out

# --- 3. The Upsampler (PixelShuffle) ---
class Upsampler(nn.Sequential):
    def __init__(self, scale_factor, n_feats, out_channels=3):
        m = []
        
        # Upsampling is done in 2x stages (e.g., 4x is two 2x steps)
        if (scale_factor & (scale_factor - 1)) == 0:    # Is scale_factor a power of 2? (2, 4, 8)
            for _ in range(int(math.log2(scale_factor))):
                # Output feature map size is n_feats * (2^2) = n_feats * 4
                m.append(default_conv(n_feats, 4 * n_feats, 3))
                # PixelShuffle converts 4 channels into 2x spatial size
                m.append(nn.PixelShuffle(2))
        elif scale_factor == 3: # 3x scale factor
            m.append(default_conv(n_feats, 9 * n_feats, 3))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        
        # Final Convolution to reduce channels to the output (e.g., RGB=3)
        m.append(default_conv(n_feats, out_channels, 3))

        super(Upsampler, self).__init__(*m)


# --- 4. The Full EDSR Model ---
class EDSR(nn.Module):
    def __init__(self, scale_factor=4, n_resblocks=16, n_feats=64, n_channels=3):
        super(EDSR, self).__init__()
        
        kernel_size = 3
        
        # Head: Initial Feature Extractor (LR -> Feature Map)
        self.head = default_conv(n_channels, n_feats, kernel_size)
        
        # Body: Stack of Residual Blocks
        m_body = [
            ResBlock(n_feats, kernel_size, bias=True) for _ in range(n_resblocks)
        ]
        
        # Global Skip Connection Conv
        m_body.append(default_conv(n_feats, n_feats, kernel_size))
        
        self.body = nn.Sequential(*m_body)
        
        # Tail: Upsampling and Final Output
        self.tail = Upsampler(scale_factor, n_feats, n_channels)

    def forward(self, x):
        # Initial Feature Extraction
        x = self.head(x)
        
        # Body with Global Skip Connection
        res = self.body(x)
        res += x # Add global skip connection
        
        # Upsample and Final Output
        x = self.tail(res)
        
        return x