import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from config import Config
from data_loader import GoProDataset
from models import ConditionalUNet
from diffusion import DiffusionScheduler
from utils import compute_psnr_ssim, save_image


def pad_to_multiple(x, mult=16):
    h, w = x.shape[-2:]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph == 0 and pw == 0:
        return x, (0, 0)
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), (ph, pw)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    model = ConditionalUNet(base_ch=Config.BASE_CHANNELS,
                            time_emb_dim=Config.TIME_EMB_DIM,
                            predict_x0=Config.PREDICT_X0,
                            residual=Config.RESIDUAL).to(device)
    ckpt = os.path.join(Config.CKPT_DIR, "model_final.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Cargado: {ckpt}")

    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS,
                                   beta_start=Config.BETA_START,
                                   beta_end=Config.BETA_END,
                                   schedule=Config.SCHEDULE, device=device)

    test_ds = GoProDataset(split=Config.SPLIT_TEST,
                           patch_size=Config.PATCH_SIZE, mode='native')
    if Config.MAX_TEST_IMAGES:
        test_ds.pairs = test_ds.pairs[:Config.MAX_TEST_IMAGES]
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    psnrs, ssims = [], []

    with torch.no_grad():
        for idx, (blur, sharp) in enumerate(loader):
            blur = blur.to(device)
            sharp = sharp.to(device)

            blur_pad, (ph, pw) = pad_to_multiple(blur, 16)
            recon_pad = scheduler.ddim_sample(model, cond=blur_pad, shape=blur_pad.shape, num_steps=100, eta=0.0)

            if ph > 0: recon_pad = recon_pad[:, :, :-ph, :]
            if pw > 0: recon_pad = recon_pad[:, :, :, :-pw]

            recon_n = (recon_pad.clamp(-1, 1) + 1) / 2
            sharp_n = (sharp.clamp(-1, 1) + 1) / 2
            blur_n = (blur.clamp(-1, 1) + 1) / 2

            recon_np = recon_n.squeeze(0).cpu().permute(1, 2, 0).numpy()
            sharp_np = sharp_n.squeeze(0).cpu().permute(1, 2, 0).numpy()
            blur_np = blur_n.squeeze(0).cpu().permute(1, 2, 0).numpy()

            psnr, ssim = compute_psnr_ssim(recon_np, sharp_np)
            psnrs.append(psnr); ssims.append(ssim)
            print(f"[{idx+1}/{len(loader)}] PSNR={psnr:.2f} | SSIM={ssim:.4f}")

            save_image(blur_np, os.path.join(Config.OUTPUT_DIR, f"blur_{idx:04d}.png"))
            save_image(recon_np, os.path.join(Config.OUTPUT_DIR, f"recon_{idx:04d}.png"))
            save_image(sharp_np, os.path.join(Config.OUTPUT_DIR, f"sharp_{idx:04d}.png"))

    if psnrs:
        print("=" * 50)
        print(f"PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
        print(f"SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()