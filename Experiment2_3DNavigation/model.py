import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block1D(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.Mish()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock1D(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Unet1D(nn.Module):
    def __init__(self, 
                 input_dim=3, 
                 base_channels=32, 
                 dim_mults=(1, 2, 4), 
                 num_classes=3  # 0,1,2 (Top, Side, Bottom)
                 ):
        super().__init__()
        
        dims = [base_channels, *map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # 1. Input
        self.init_conv = nn.Conv1d(input_dim, base_channels, 7, padding=3)
        
        # 2. Time Embedding
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.label_emb = nn.Embedding(num_classes + 1, time_dim)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Down
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResnetBlock1D(dim_in, dim_in, time_dim),
                ResnetBlock1D(dim_in, dim_in, time_dim),
                nn.Conv1d(dim_in, dim_out, 3, 2, 1)
            ]))

        # Mid
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock1D(mid_dim, mid_dim, time_dim)
        self.mid_block2 = ResnetBlock1D(mid_dim, mid_dim, time_dim)

        # Up
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose1d(dim_out, dim_in, 4, 2, 1),
                ResnetBlock1D(dim_in * 2, dim_in, time_dim),
                ResnetBlock1D(dim_in, dim_in, time_dim),
            ]))

        # Final
        self.final_res_block = ResnetBlock1D(base_channels * 2, base_channels, time_dim)
        self.final_conv = nn.Conv1d(base_channels, input_dim, 1)
        self.final_conv.weight.data.zero_()
        self.final_conv.bias.data.zero_()

    def forward(self, x, time, cond=None):
        """
        x: (B, 3, 32)
        time: (B,)
        cond: (B,)  (0,1,2 or 3)
        """

        t = self.time_mlp(time) # (B, time_dim)
        
        if cond is not None:
            c = self.label_emb(cond) # (B, time_dim)
            t = t + c 

        x = self.init_conv(x)
        r = x.clone()
        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for upsample, block1, block2 in self.ups:
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        
        return self.final_conv(x)
