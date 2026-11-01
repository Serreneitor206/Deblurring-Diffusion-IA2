"""
Paso 2 del pipeline: fine-tuning de NAFNet-GoPro sobre el GoPro de Kaggle.

Que hace:
  1. Detecta las secuencias (escenas) del train comparando frames consecutivos,
     porque el dataset de Kaggle renombro los archivos a 0.png, 1.png, ...
  2. Separa ~10% de las secuencias como validacion (escenas completas, sin fuga).
  3. Mide la validacion con los pesos originales (linea base, paso 0).
  4. Entrena con parches de 256, fp16, gradient checkpointing y PSNR loss
     (la misma loss con la que se entreno NAFNet).
  5. Guarda best.pth (formato compatible con precompute_nafnet.py) y last.pt
     para reanudar.

Uso:
    python finetune_nafnet.py                       # entrenamiento completo
    python finetune_nafnet.py --iters 200 --val_every 100   # prueba rapida
    python finetune_nafnet.py --resume              # continuar donde quedo

Si hay OOM: --batch 1 --accum 2
"""
import argparse
import csv
import json
import math
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nafnet_arch import build_nafnet, enable_grad_checkpointing
from precompute_nafnet import (DEFAULT_DATA, ensure_weights, find_split_dir,
                               list_pairs, predict, psnr_u8, read_rgb)


# --------------------------------------------------------------------------
# Deteccion de secuencias
# --------------------------------------------------------------------------
def thumb(path, size=(80, 45)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def detect_sequences(pairs, cache_path):
    """
    Devuelve una lista de secuencias (listas de indices). Un corte de escena es
    un salto grande entre miniaturas sharp consecutivas.
    """
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        if data.get("n_pairs") == len(pairs):
            return data["sequences"]

    print("Detectando secuencias (solo la primera vez)...")
    thumbs = [thumb(s) for _, s in tqdm(pairs, desc="miniaturas")]
    d = np.array([np.abs(thumbs[i + 1] - thumbs[i]).mean() for i in range(len(thumbs) - 1)])
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-6
    thr = max(med + 10 * mad, 3 * med, 0.03)
    cuts = [i + 1 for i in np.where(d > thr)[0]]

    bounds = [0] + cuts + [len(pairs)]
    seqs = [list(range(a, b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    # Si la deteccion no es fiable, bloques contiguos de 100 frames
    if len(seqs) < 5:
        blk = max(10, len(pairs) // 20)
        print(f"[!] No se detectaron cortes claros: uso bloques contiguos de {blk} frames")
        seqs = [list(range(i, min(i + blk, len(pairs)))) for i in range(0, len(pairs), blk)]

    with open(cache_path, "w") as f:
        json.dump({"n_pairs": len(pairs), "threshold": float(thr),
                   "sequences": seqs}, f)
    return seqs


def split_sequences(seqs, val_frac, seed=0):
    """Elige secuencias completas repartidas a lo largo del dataset para validacion."""
    total = sum(len(s) for s in seqs)
    order = list(range(len(seqs)))
    random.Random(seed).shuffle(order)
    val, n = [], 0
    for i in order[:-1]:  # siempre queda al menos una secuencia para train
        if n >= val_frac * total:
            break
        val.append(i)
        n += len(seqs[i])
    train_idx = [j for i, s in enumerate(seqs) if i not in val for j in s]
    val_idx = [j for i in sorted(val) for j in seqs[i]]
    return train_idx, val_idx, sorted(val)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, pairs, patch=256, patches_per_epoch=10000):
        self.pairs = pairs
        self.patch = patch
        self.length = patches_per_epoch

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        b_p, s_p = random.choice(self.pairs)
        blur, sharp = read_rgb(b_p), read_rgb(s_p)
        H, W, _ = blur.shape
        p = self.patch
        y, x = random.randint(0, H - p), random.randint(0, W - p)
        blur, sharp = blur[y:y + p, x:x + p], sharp[y:y + p, x:x + p]

        # Aumentos geometricos (los mismos para ambas imagenes)
        if random.random() < 0.5:
            blur, sharp = blur[:, ::-1], sharp[:, ::-1]
        if random.random() < 0.5:
            blur, sharp = blur[::-1], sharp[::-1]
        k = random.randint(0, 3)
        blur, sharp = np.rot90(blur, k), np.rot90(sharp, k)

        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return to_t(blur), to_t(sharp)


def worker_init(wid):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed + wid)
    np.random.seed((seed + wid) % 2 ** 32)


# --------------------------------------------------------------------------
# Loss y validacion
# --------------------------------------------------------------------------
def psnr_loss(pred, target):
    """PSNR loss de NAFNet: minimizar -PSNR (equivale a log-MSE por imagen)."""
    mse = ((pred.float() - target.float()) ** 2).mean(dim=(1, 2, 3))
    return (10.0 / math.log(10.0)) * torch.log(mse + 1e-8).mean()


@torch.no_grad()
def validate(net, val_pairs, device, amp):
    net.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vals, state = [], {}
    for b_p, s_p in val_pairs:
        blur, sharp = read_rgb(b_p), read_rgb(s_p)
        # Imagen completa; si no cabe en VRAM pasa solo a tiles
        y = predict(net, blur, device, amp, None, 64, state)
        vals.append(psnr_u8(y, sharp))
    net.train()
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA)
    ap.add_argument("--model", default="NAFNet-GoPro-width64")
    ap.add_argument("--weights", default="./pretrained/NAFNet-GoPro-width64.pth")
    ap.add_argument("--out_dir", default="./checkpoints/nafnet_ft")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1, help="Acumulacion de gradiente")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--val_images", type=int, default=40)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_ckpt", action="store_true", help="Sin gradient checkpointing")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if amp else ""))

    # ---------------- datos ----------------
    root = os.path.normpath(args.data_root)
    pairs = list_pairs(find_split_dir(root, "train"))
    seqs = detect_sequences(pairs, os.path.join(args.out_dir, "sequences.json"))
    tr_idx, va_idx, va_seqs = split_sequences(seqs, args.val_frac, args.seed)
    lens = sorted(len(s) for s in seqs)
    print(f"Secuencias detectadas: {len(seqs)} (tamanos min/mediana/max: "
          f"{lens[0]}/{lens[len(lens) // 2]}/{lens[-1]})")
    print(f"Train: {len(tr_idx)} imagenes | Val: {len(va_idx)} imagenes "
          f"(secuencias {va_seqs})")
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({"train": tr_idx, "val": va_idx, "val_sequences": va_seqs}, f)

    train_pairs = [pairs[i] for i in tr_idx]
    step = max(1, len(va_idx) // args.val_images)
    val_pairs = [pairs[i] for i in va_idx[::step][:args.val_images]]

    loader = DataLoader(
        PatchDataset(train_pairs, args.patch, patches_per_epoch=10 ** 9),
        batch_size=args.batch, num_workers=args.workers, pin_memory=amp,
        worker_init_fn=worker_init, persistent_workers=args.workers > 0, drop_last=True)
    it_loader = iter(loader)

    # ---------------- modelo ----------------
    ensure_weights(args.weights, args.model)
    net = build_nafnet(args.model, args.weights, tlc=True, device=device)
    if not args.no_ckpt:
        enable_grad_checkpointing(net)
    net.train()

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        t = (it - args.warmup) / max(1, args.iters - args.warmup)
        return 1e-7 + 0.5 * (args.lr - 1e-7) * (1 + math.cos(math.pi * t))

    start, best = 0, -1.0
    last_path = os.path.join(args.out_dir, "last.pt")
    log_path = os.path.join(args.out_dir, "log.csv")
    if args.resume and os.path.isfile(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["params"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["iter"], ck["best"]
        print(f"Reanudando desde la iteracion {start} (mejor val: {best:.2f} dB)")
    else:
        base = validate(net, val_pairs, device, amp)
        best = base
        print(f"Validacion con pesos originales: {base:.2f} dB")
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerows([["iter", "val_psnr", "lr"], [0, base, 0]])
        torch.save({"params": net.state_dict()}, os.path.join(args.out_dir, "best.pth"))

    # ---------------- entrenamiento ----------------
    pbar = tqdm(range(start, args.iters), initial=start, total=args.iters)
    run_loss, t0 = 0.0, time.time()
    for it in pbar:
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            blur, sharp = next(it_loader)
            blur, sharp = blur.to(device, non_blocking=True), sharp.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                pred = net(blur)
            loss = psnr_loss(pred, sharp) / args.accum
            scaler.scale(loss).backward()
            run_loss += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)  # igual que el entrenamiento oficial
        scaler.step(opt)
        scaler.update()

        if (it + 1) % 50 == 0:
            pbar.set_postfix(loss=f"{run_loss / 50:.3f}", lr=f"{lr_at(it):.1e}")
            run_loss = 0.0

        if (it + 1) % args.val_every == 0 or it + 1 == args.iters:
            v = validate(net, val_pairs, device, amp)
            tag = ""
            if v > best:
                best = v
                torch.save({"params": net.state_dict()}, os.path.join(args.out_dir, "best.pth"))
                tag = "  <- mejor"
            tqdm.write(f"[iter {it + 1}] val PSNR = {v:.2f} dB{tag}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([it + 1, v, lr_at(it)])
            torch.save({"params": net.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "iter": it + 1, "best": best}, last_path)

    print(f"\nListo en {(time.time() - t0) / 3600:.2f} h. Mejor val PSNR: {best:.2f} dB")
    print(f"Pesos: {os.path.join(args.out_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
