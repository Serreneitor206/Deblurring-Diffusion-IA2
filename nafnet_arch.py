"""
NAFNet autocontenido (sin dependencia de BasicSR).

Adaptado del repositorio oficial:
    https://github.com/megvii-research/NAFNet  (MIT License, (c) 2022 megvii-model)
    Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.

Cambios respecto al original:
  * LayerNorm2d y el pooling local se calculan siempre en fp32, para que la
    inferencia con autocast (fp16) no pierda precision (cumsum en fp16 desborda).
  * El TLC (Test-time Local Converter, Chu et al. 2021) se implementa de forma
    explicita: cada pooling global de la atencion de canal se reemplaza por un
    pooling local cuyo tamano equivale a 1.5x el parche de entrenamiento (256),
    igual que la configuracion oficial de test `NAFNetLocal`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Bloques basicos
# --------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """LayerNorm sobre el eje de canales, calculado en fp32."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight.float().view(1, -1, 1, 1) * y + self.bias.float().view(1, -1, 1, 1)
        return y.to(dtype)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LocalAvgPool2d(nn.Module):
    """
    Pooling local del TLC. `kernel_size` esta en pixeles del mapa de features
    y se fija una sola vez con `NAFNet.enable_tlc()`.
    Si el kernel cubre todo el mapa, equivale al pooling global original.
    """

    def __init__(self):
        super().__init__()
        self.kernel_size = None  # (kh, kw), se asigna en enable_tlc

    def forward(self, x):
        if self.kernel_size is None:
            return F.adaptive_avg_pool2d(x, 1)
        n, c, h, w = x.shape
        k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
        if k1 >= h and k2 >= w:
            return F.adaptive_avg_pool2d(x, 1)

        dtype = x.dtype
        s = x.float().cumsum(dim=-1).cumsum_(dim=-2)
        s = F.pad(s, (1, 0, 1, 0))
        out = (s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2]
               - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2]) / (k1 * k2)
        _h, _w = out.shape[2:]
        pad = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
        out = F.pad(out, pad, mode='replicate')
        return out.to(dtype)


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        # Simplified Channel Attention (los nombres coinciden con los pesos oficiales)
        self.sca = nn.Sequential(LocalAvgPool2d(), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()

        ffn = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


# --------------------------------------------------------------------------
# Red completa
# --------------------------------------------------------------------------
class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1,
                 enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    # ---------------- TLC ----------------
    def _set_pool_kernels(self, module, kernel):
        for m in module.modules():
            if isinstance(m, LocalAvgPool2d):
                m.kernel_size = (kernel, kernel)

    def enable_tlc(self, train_size=256, ratio=1.5):
        """
        Replica `NAFNetLocal` oficial: base_size = 1.5 * train_size.
        En cada nivel (resolucion 1/2^i) el kernel es base_size / 2^i.
        """
        base = int(train_size * ratio)
        for i, enc in enumerate(self.encoders):
            self._set_pool_kernels(enc, base // (2 ** i))
        self._set_pool_kernels(self.middle_blks, base // (2 ** len(self.encoders)))
        n = len(self.decoders)
        for j, dec in enumerate(self.decoders):
            self._set_pool_kernels(dec, base // (2 ** (n - 1 - j)))
        return self

    # ---------------- forward ----------------
    def forward(self, inp):
        _, _, H, W = inp.shape
        inp = self._pad(inp)

        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        # Residual global en fp32 para no perder precision en la salida
        x = self.ending(x).float() + inp.float()
        return x[:, :, :H, :W]

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph))


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def enable_grad_checkpointing(net):
    """
    Envuelve cada NAFBlock con torch.utils.checkpoint: en entrenamiento no guarda
    sus activaciones y las recalcula en el backward (menos VRAM, ~30% mas lento).
    """
    from torch.utils.checkpoint import checkpoint

    for m in net.modules():
        if isinstance(m, NAFBlock) and not hasattr(m, "_orig_forward"):
            m._orig_forward = m.forward

            def fwd(x, _m=m):
                if _m.training and torch.is_grad_enabled():
                    return checkpoint(_m._orig_forward, x, use_reentrant=False)
                return _m._orig_forward(x)

            m.forward = fwd
    return net


NAFNET_CONFIGS = {
    # nombre : (width, enc_blk_nums, middle_blk_num, dec_blk_nums)
    "NAFNet-GoPro-width64": (64, (1, 1, 1, 28), 1, (1, 1, 1, 1)),
    "NAFNet-GoPro-width32": (32, (1, 1, 1, 28), 1, (1, 1, 1, 1)),
}


def build_nafnet(name="NAFNet-GoPro-width64", weights=None, tlc=True, device="cpu"):
    width, enc, mid, dec = NAFNET_CONFIGS[name]
    net = NAFNet(width=width, enc_blk_nums=enc, middle_blk_num=mid, dec_blk_nums=dec)
    if weights is not None:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt.get("params", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        net.load_state_dict(state, strict=True)
    if tlc:
        net.enable_tlc()
    return net.to(device).eval()
