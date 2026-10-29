import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from config import Config
from data_loader import GoProDataset
from models import ConditionalUNet
from diffusion import DiffusionScheduler
from utils import plot_training_curves


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print(f"Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 50)

    # === Dataset de TRAIN ===
    full_ds = GoProDataset(split=Config.SPLIT_TRAIN,
                           patch_size=Config.PATCH_SIZE, mode='crop')
    if Config.MAX_TRAIN_IMAGES:
        full_ds.pairs = full_ds.pairs[:Config.MAX_TRAIN_IMAGES]

    # === Split train / val (90% / 10%) ===
    random.seed(42)
    random.shuffle(full_ds.pairs)
    n_val = max(1, int(0.10 * len(full_ds.pairs)))
    val_pairs = full_ds.pairs[:n_val]
    train_pairs = full_ds.pairs[n_val:]

    # Dataset de validación con center_crop (reproducible)
    val_ds = GoProDataset(split=Config.SPLIT_TRAIN,
                          patch_size=Config.PATCH_SIZE, mode='center_crop')
    val_ds.pairs = val_pairs

    full_ds.pairs = train_pairs
    train_ds = full_ds

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS,
                              pin_memory=(device.type == 'cuda'), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS,
                            pin_memory=(device.type == 'cuda'), drop_last=False)

    # === Modelo ===
    model = ConditionalUNet(base_ch=Config.BASE_CHANNELS,
                            time_emb_dim=Config.TIME_EMB_DIM,
                            predict_x0=Config.PREDICT_X0,
                            residual=Config.RESIDUAL).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parámetros del modelo: {n_params:.2f} M")

    # === Scheduler de difusión ===
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS,
                                   beta_start=Config.BETA_START,
                                   beta_end=Config.BETA_END,
                                   schedule=Config.SCHEDULE, device=device)

    optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    os.makedirs(Config.CKPT_DIR, exist_ok=True)

    # === Historial de losses ===
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(Config.EPOCHS):
        # ====== TRAIN ======
        model.train()
        epoch_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [TRAIN]")

        for blur, sharp in pbar:
            blur = blur.to(device, non_blocking=True)
            sharp = sharp.to(device, non_blocking=True)

            # t en [1, T) para evitar inestabilidad en t=0
            t = torch.randint(1, Config.TIMESTEPS, (sharp.size(0),), device=device)
            x_t, _ = scheduler.q_sample(sharp, t)

            with torch.amp.autocast('cuda', enabled=use_amp):
                x0_pred = model(x_t, blur, t)
                loss = torch.nn.functional.mse_loss(x0_pred, sharp)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ====== VALIDATION ======
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for blur, sharp in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [VAL]",
                                    leave=False):
                blur = blur.to(device, non_blocking=True)
                sharp = sharp.to(device, non_blocking=True)

                t = torch.randint(1, Config.TIMESTEPS, (sharp.size(0),), device=device)
                x_t, _ = scheduler.q_sample(sharp, t)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    x0_pred = model(x_t, blur, t)
                    loss = torch.nn.functional.mse_loss(x0_pred, sharp)

                epoch_val_loss += loss.item()

        avg_val = epoch_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val)

        lr_scheduler.step()

        # Log
        print(f"Epoch {epoch+1}: train_loss = {avg_train:.5f} | val_loss = {avg_val:.5f} "
              f"| lr = {lr_scheduler.get_last_lr()[0]:.2e}")

        # Guardar mejor modelo
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(),
                       os.path.join(Config.CKPT_DIR, "model_best.pt"))
            print(f"  ⭐ Nuevo mejor modelo (val={avg_val:.5f})")

        # Checkpoints periódicos
        if (epoch + 1) % Config.SAVE_EVERY == 0 or epoch == Config.EPOCHS - 1:
            torch.save(model.state_dict(),
                       os.path.join(Config.CKPT_DIR, f"model_epoch{epoch+1}.pt"))

    # === Modelo final ===
    torch.save(model.state_dict(), os.path.join(Config.CKPT_DIR, "model_final.pt"))
    print("\n✅ Entrenamiento finalizado.")

    # === Gráfica ===
    plot_path = os.path.join(Config.OUTPUT_DIR, "training_curve.png")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    plot_training_curves(train_losses, val_losses, plot_path)

    # === Resumen final ===
    print("\n" + "=" * 50)
    print(f"RESUMEN:")
    print(f"  Train final:  {train_losses[-1]:.5f}")
    print(f"  Val final:    {val_losses[-1]:.5f}")
    best_epoch = int(np.argmin(val_losses)) + 1
    print(f"  Mejor val:    {best_val_loss:.5f} (epoch {best_epoch})")
    print("=" * 50)


if __name__ == "__main__":
    main()