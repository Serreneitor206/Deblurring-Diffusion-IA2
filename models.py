import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    """
    U-Net condicional que predice x0 = blur + delta.
    Encoder: 3 pools (÷8). Bottleneck: +1 pool (÷16).
    """
    def __init__(self, in_ch=3, cond_ch=3, base_ch=64, time_emb_dim=256,
                 predict_x0=True, residual=True):
        super().__init__()
        self.predict_x0 = predict_x0
        self.residual = residual
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # === Encoder ===
        self.conv_in = nn.Conv2d(in_ch + cond_ch, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch, base_ch, time_emb_dim)                # 128x128
        self.down2 = ResBlock(base_ch, base_ch * 2, time_emb_dim)            # 64x64
        self.down3 = ResBlock(base_ch * 2, base_ch * 4, time_emb_dim)        # 32x32

        # === Bottleneck (a 16x16) ===
        self.cond_bottle = nn.Conv2d(cond_ch, base_ch * 4, 1)
        self.mid1 = ResBlock(base_ch * 4 * 2, base_ch * 4, time_emb_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # === Decoder ===
        # up3: recupera a 32x32; entrada = u3 (256) + d3 (256) + cond (256) = 768
        self.cond_up3 = nn.Conv2d(cond_ch, base_ch * 4, 1)
        self.up3 = ResBlock(base_ch * 12, base_ch * 2, time_emb_dim)

        # up2: recupera a 64x64; entrada = u3 (128) + d2 (128) + cond (128) = 384
        self.cond_up2 = nn.Conv2d(cond_ch, base_ch * 2, 1)
        self.up2 = ResBlock(base_ch * 6, base_ch, time_emb_dim)

        # up1: recupera a 128x128; entrada = u2 (64) + d1 (64) + cond (64) = 192
        self.cond_up1 = nn.Conv2d(cond_ch, base_ch, 1)
        self.up1 = ResBlock(base_ch * 3, base_ch, time_emb_dim)

        self.conv_out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, cond, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.time_emb_dim))

        # === Encoder ===
        h = self.conv_in(torch.cat([x, cond], dim=1))    # 128x128
        d1 = self.down1(h, t_emb)                        # 128x128, base_ch
        d2 = self.down2(F.avg_pool2d(d1, 2), t_emb)      # 64x64,   base_ch*2
        d3 = self.down3(F.avg_pool2d(d2, 2), t_emb)      # 32x32,   base_ch*4

        # === Bottleneck (a 16x16) ===
        d3_pooled = F.avg_pool2d(d3, 2)                  # 16x16, base_ch*4
        cond_b = F.interpolate(cond, size=d3_pooled.shape[-2:],
                               mode='bilinear', align_corners=False)
        cond_b = self.cond_bottle(cond_b)                # 16x16, base_ch*4
        m = self.mid1(torch.cat([d3_pooled, cond_b], dim=1), t_emb)   # 512 -> 256
        m = self.mid2(m, t_emb)                          # 16x16, base_ch*4

        # === Decoder ===
        # up3: 16x16 -> 32x32
        u3 = F.interpolate(m, size=d3.shape[-2:], mode='nearest')    # 32x32, 256
        cond_u3 = F.interpolate(cond, size=u3.shape[-2:],
                                mode='bilinear', align_corners=False)
        cond_u3 = self.cond_up3(cond_u3)                             # 32x32, 256
        u3 = self.up3(torch.cat([u3, d3, cond_u3], dim=1), t_emb)    # 768 -> 128

        # up2: 32x32 -> 64x64
        u2 = F.interpolate(u3, size=d2.shape[-2:], mode='nearest')   # 64x64, 128
        cond_u2 = F.interpolate(cond, size=u2.shape[-2:],
                                mode='bilinear', align_corners=False)
        cond_u2 = self.cond_up2(cond_u2)                             # 64x64, 128
        u2 = self.up2(torch.cat([u2, d2, cond_u2], dim=1), t_emb)    # 384 -> 64

        # up1: 64x64 -> 128x128
        u1 = F.interpolate(u2, size=d1.shape[-2:], mode='nearest')   # 128x128, 64
        cond_u1 = self.cond_up1(cond)                                # 128x128, 64
        u1 = self.up1(torch.cat([u1, d1, cond_u1], dim=1), t_emb)    # 192 -> 64

        delta = self.conv_out(u1)

        # Residual
        if self.residual:
            return cond + delta
        return delta