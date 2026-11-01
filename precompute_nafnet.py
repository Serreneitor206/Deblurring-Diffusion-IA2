"""
Paso 1 del pipeline: precalcula las predicciones de NAFNet (y_hat) para GoPro.

Para cada split guarda:
    <out_root>/<split>/nafnet/<nombre>.png    (PNG RGB de 16 bits)
    <out_root>/<split>/metrics.csv             (PSNR/SSIM por imagen)
    <out_root>/summary.json                    (medias + estadisticas del residuo)

Tambien hace la "Fase A": mide PSNR de blur y de NAFNet en train y test, y
calcula media/desviacion del residuo (sharp - y_hat), que el refinador de
difusion usara para normalizar.

Uso tipico (desde la raiz del repo):
    python precompute_nafnet.py
    python precompute_nafnet.py --splits test --limit 20        # prueba rapida
    python precompute_nafnet.py --tile 512                      # forzar tiles

Es reanudable: si se interrumpe, al relanzarlo reutiliza los PNG ya guardados.
"""
import argparse
import csv
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from tqdm import tqdm

from nafnet_arch import build_nafnet

try:
    from config import Config
    DEFAULT_DATA = getattr(Config, "DATA_PATH", None)
except Exception:
    DEFAULT_DATA = None

GDRIVE_IDS = {
    "NAFNet-GoPro-width64": "1S0PVRbyTakYY9a82kujgZLbMihfNBLfC",
}


# --------------------------------------------------------------------------
# Pesos
# --------------------------------------------------------------------------
def ensure_weights(path, model_name):
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_id = GDRIVE_IDS.get(model_name)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Descargando pesos de {model_name} ...")
    try:
        import gdown
        gdown.download(url, path, quiet=False)
    except Exception as e:
        print(f"[!] Descarga automatica fallida: {e}")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No se encontraron los pesos en '{path}'.\n"
            f"Descargalos manualmente desde:\n"
            f"  https://drive.google.com/file/d/{file_id}/view\n"
            f"y guardalos en esa ruta.")
    return path


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def find_split_dir(root, split):
    for cand in (split, split.capitalize(), split.upper()):
        p = os.path.join(root, cand)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"No encontre el split '{split}' en {root}")


def natural_key(path):
    """Orden natural: '2.png' < '10.png' (sorted() normal daria '10' < '2')."""
    import re
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_pairs(split_dir):
    """Estructura plana: split/blur/*.png y split/sharp/*.png emparejados por nombre."""
    exts = ("*.png", "*.jpg", "*.jpeg")
    blur = []
    for e in exts:
        blur += glob.glob(os.path.join(split_dir, "blur", e))
    sharp = {}
    for e in exts:
        for f in glob.glob(os.path.join(split_dir, "sharp", e)):
            sharp[os.path.basename(f)] = f
    pairs = [(b, sharp[os.path.basename(b)]) for b in sorted(blur, key=natural_key)
             if os.path.basename(b) in sharp]
    missing = len(blur) - len(pairs)
    if missing:
        print(f"[!] {missing} imagenes blur sin pareja sharp (se omiten)")
    return pairs


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"No se pudo leer {path}")
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / scale


def write_rgb16(path, img):
    img16 = np.round(np.clip(img, 0, 1) * 65535.0).astype(np.uint16)
    cv2.imwrite(path, cv2.cvtColor(img16, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 1])


# --------------------------------------------------------------------------
# Inferencia
# --------------------------------------------------------------------------
def _ramp(n, overlap):
    """Peso 1D para mezclar tiles: rampa lineal en los bordes."""
    w = np.ones(n, dtype=np.float32)
    if overlap > 0:
        r = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
        w[:overlap] = r
        w[-overlap:] = r[::-1]
    return w


@torch.no_grad()
def run_tiled(net, x, tile, overlap, amp):
    _, _, H, W = x.shape
    out = torch.zeros((1, 3, H, W), dtype=torch.float32, device=x.device)
    acc = torch.zeros((1, 1, H, W), dtype=torch.float32, device=x.device)
    stride = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    for y0 in ys:
        for x0 in xs:
            th, tw = min(tile, H), min(tile, W)
            patch = x[:, :, y0:y0 + th, x0:x0 + tw]
            with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
                pred = net(patch).float()
            wy = _ramp(th, overlap if th == tile else 0)
            wx = _ramp(tw, overlap if tw == tile else 0)
            # Los bordes de la imagen completa no se atenuan
            if y0 == 0: wy[:overlap] = 1
            if y0 + th >= H: wy[-overlap:] = 1
            if x0 == 0: wx[:overlap] = 1
            if x0 + tw >= W: wx[-overlap:] = 1
            wgt = torch.from_numpy(np.outer(wy, wx)).to(x.device)[None, None]
            out[:, :, y0:y0 + th, x0:x0 + tw] += pred * wgt
            acc[:, :, y0:y0 + th, x0:x0 + tw] += wgt
    return out / acc.clamp_min(1e-8)


@torch.no_grad()
def predict(net, img, device, amp, tile, overlap, state):
    x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
    if tile is None and not state.get("force_tile"):
        try:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                y = net(x).float()
            return y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            state["force_tile"] = True
            print("\n[!] OOM con imagen completa: cambio a modo tiles (512, solape 64)")
    t = tile or 512
    y = run_tiled(net, x, t, overlap, amp)
    return y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------
def psnr_u8(a, b):
    """PSNR estandar de GoPro: ambas imagenes cuantizadas a 8 bits."""
    a8 = np.round(np.clip(a, 0, 1) * 255).astype(np.float64)
    b8 = np.round(np.clip(b, 0, 1) * 255).astype(np.float64)
    mse = np.mean((a8 - b8) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim_u8(a, b):
    from skimage.metrics import structural_similarity
    a8 = np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)
    b8 = np.round(np.clip(b, 0, 1) * 255).astype(np.uint8)
    return float(structural_similarity(a8, b8, channel_axis=2, data_range=255))


class ResidualStats:
    """Media y desviacion por canal de (sharp - y_hat) en rango [-1, 1]."""

    def __init__(self):
        self.n = 0
        self.s = np.zeros(3)
        self.s2 = np.zeros(3)
        self.abs = np.zeros(3)

    def update(self, sharp, pred):
        r = (sharp.astype(np.float64) - pred.astype(np.float64)) * 2.0  # escala [-1,1]
        r = r.reshape(-1, 3)
        self.n += r.shape[0]
        self.s += r.sum(0)
        self.s2 += (r ** 2).sum(0)
        self.abs += np.abs(r).sum(0)

    def result(self):
        mean = self.s / self.n
        std = np.sqrt(np.maximum(self.s2 / self.n - mean ** 2, 0))
        return {"mean": mean.tolist(), "std": std.tolist(),
                "std_global": float(np.sqrt(self.s2.sum() / (3 * self.n) - mean.mean() ** 2)),
                "mean_abs": (self.abs / self.n).tolist()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def process_split(split, args, net, device, state):
    split_dir = find_split_dir(args.data_root, split)
    pairs = list_pairs(split_dir)
    if args.limit:
        step = max(1, len(pairs) // args.limit)  # muestreo repartido entre secuencias
        pairs = pairs[::step][:args.limit]
    print(f"\n=== {split}: {len(pairs)} pares  ({split_dir})")

    out_dir = os.path.join(args.out_root, split, "nafnet")
    os.makedirs(out_dir, exist_ok=True)

    do_ssim = args.ssim == "all" or (args.ssim == "test" and split.lower() == "test")
    rows, stats = [], ResidualStats()
    pool = ThreadPoolExecutor(max_workers=2)
    futures = []
    t0 = time.time()

    for blur_p, sharp_p in tqdm(pairs, desc=split):
        name = os.path.splitext(os.path.basename(blur_p))[0] + ".png"
        out_p = os.path.join(out_dir, name)
        blur, sharp = read_rgb(blur_p), read_rgb(sharp_p)

        if os.path.isfile(out_p) and not args.overwrite:
            pred = read_rgb(out_p)
        else:
            pred = predict(net, blur, device, args.amp, args.tile, args.overlap, state)
            futures.append(pool.submit(write_rgb16, out_p, pred))

        row = {"name": name,
               "psnr_blur": psnr_u8(blur, sharp),
               "psnr_nafnet": psnr_u8(pred, sharp)}
        if do_ssim:
            row["ssim_blur"] = ssim_u8(blur, sharp)
            row["ssim_nafnet"] = ssim_u8(pred, sharp)
        rows.append(row)
        stats.update(sharp, pred)

    for f in futures:
        f.result()
    pool.shutdown()

    with open(os.path.join(args.out_root, split, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"n": len(rows), "seconds": round(time.time() - t0, 1),
               "residual": stats.result()}
    for k in rows[0]:
        if k != "name":
            summary[k] = float(np.mean([r[k] for r in rows]))
    print(f"  PSNR blur   = {summary['psnr_blur']:.2f} dB")
    print(f"  PSNR NAFNet = {summary['psnr_nafnet']:.2f} dB")
    if do_ssim:
        print(f"  SSIM blur   = {summary['ssim_blur']:.4f} | SSIM NAFNet = {summary['ssim_nafnet']:.4f}")
    print(f"  Residuo (sharp - y_hat, rango [-1,1]): std = {summary['residual']['std_global']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA, help="Carpeta con train/ y test/")
    ap.add_argument("--out_root", default=None,
                    help="Destino (por defecto: <data_root>_nafnet)")
    ap.add_argument("--model", default="NAFNet-GoPro-width64")
    ap.add_argument("--weights", default="./pretrained/NAFNet-GoPro-width64.pth")
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--limit", type=int, default=0, help="Procesar solo N imagenes por split")
    ap.add_argument("--tile", type=int, default=None, help="Forzar tiles de este tamano")
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--fp32", action="store_true", help="Desactivar fp16 (mas lento, mas VRAM)")
    ap.add_argument("--no_tlc", action="store_true", help="Desactivar TLC (no recomendado)")
    ap.add_argument("--ssim", choices=["none", "test", "all"], default="test")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.data_root:
        ap.error("Indica --data_root (o define Config.DATA_PATH)")
    args.data_root = os.path.normpath(args.data_root)
    if args.out_root is None:
        args.out_root = args.data_root + "_nafnet"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = (device.type == "cuda") and not args.fp32
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"fp16: {args.amp} | TLC: {not args.no_tlc} | salida: {args.out_root}")

    ensure_weights(args.weights, args.model)
    net = build_nafnet(args.model, args.weights, tlc=not args.no_tlc, device=device)
    print(f"{args.model}: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parametros")

    os.makedirs(args.out_root, exist_ok=True)
    summary_path = os.path.join(args.out_root, "summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    state = {}
    for split in args.splits:
        summary[split] = process_split(split, args, net, device, state)
        summary[split]["limit"] = args.limit
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # ---------------- Fase A: diagnostico ----------------
    tr = next((summary[k] for k in summary if k.lower() == "train"), None)
    te = next((summary[k] for k in summary if k.lower() == "test"), None)
    print("\n" + "=" * 60)
    print("FASE A")
    if te:
        print(f"  Test : blur {te['psnr_blur']:.2f} dB -> NAFNet {te['psnr_nafnet']:.2f} dB "
              f"(referencia oficial NAFNet-64: 33.71 dB)")
    if tr:
        print(f"  Train: blur {tr['psnr_blur']:.2f} dB -> NAFNet {tr['psnr_nafnet']:.2f} dB")
    if tr and te:
        gap = tr["psnr_nafnet"] - te["psnr_nafnet"]
        print(f"  Brecha train-test de NAFNet: {gap:+.2f} dB")
        print(f"  std residuo  train {tr['residual']['std_global']:.4f} | "
              f"test {te['residual']['std_global']:.4f}")
        if gap < 1.0:
            print("  -> Brecha pequena: se puede entrenar el refinador con estos y_hat.")
        else:
            print("  -> Brecha grande: el residuo en train no representa al de test; "
                  "habra que aplicar mitigaciones.")
    print("=" * 60)


if __name__ == "__main__":
    main()
