"""
Diagnostico rapido: ¿el GoPro de Kaggle es igual al GoPro estandar?

Revisa una muestra de pares y reporta:
  * formato, resolucion y tamano de archivo
  * brillo medio de blur vs sharp (en niveles 0-255). En el GoPro estandar
    ('blur', promedio de frames en sRGB) la diferencia es ~0. En la variante
    'blur_gamma' (promedio en espacio lineal) el blur sale mas brillante.
  * indice de bloques 8x8 (>1.1 sugiere que las imagenes pasaron por JPEG)
  * resumen de metrics.csv de NAFNet, si existe

Uso:
    python diagnose_dataset.py
    python diagnose_dataset.py --n 100
"""
import argparse
import csv
import os

import numpy as np

from precompute_nafnet import DEFAULT_DATA, find_split_dir, list_pairs, read_rgb


def blockiness(gray):
    """Relacion entre diferencias en bordes de bloque 8x8 y el resto (columnas)."""
    d = np.abs(np.diff(gray, axis=1))
    cols = np.arange(d.shape[1])
    at_edge = d[:, cols % 8 == 7].mean()
    inside = d[:, cols % 8 != 7].mean()
    return at_edge / max(inside, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    root = os.path.normpath(args.data_root)

    for split in ("train", "test"):
        try:
            pairs = list_pairs(find_split_dir(root, split))
        except FileNotFoundError as e:
            print(e)
            continue
        step = max(1, len(pairs) // args.n)
        sample = pairs[::step][:args.n]

        diffs, hi_diffs, blocks_b, blocks_s, shapes, sizes = [], [], [], [], set(), []
        for b_p, s_p in sample:
            b, s = read_rgb(b_p), read_rgb(s_p)
            shapes.add(b.shape)
            sizes.append(os.path.getsize(b_p) / 1e6)
            diffs.append((b.mean() - s.mean()) * 255)
            # Zonas brillantes del sharp: alli el efecto gamma es mas fuerte
            mask = s.mean(2) > np.quantile(s.mean(2), 0.9)
            hi_diffs.append((b.mean(2)[mask].mean() - s.mean(2)[mask].mean()) * 255)
            blocks_b.append(blockiness(b.mean(2)))
            blocks_s.append(blockiness(s.mean(2)))

        print(f"\n=== {split}: {len(pairs)} pares, muestra {len(sample)}")
        print(f"  extension: {os.path.splitext(sample[0][0])[1]} | resoluciones: {shapes}")
        print(f"  tamano medio blur: {np.mean(sizes):.2f} MB")
        print(f"  brillo blur - sharp (niveles 0-255): media {np.mean(diffs):+.2f}, "
              f"positivo en {np.mean(np.array(diffs) > 0) * 100:.0f}% de las imagenes")
        print(f"  idem en zonas brillantes (top 10%): {np.mean(hi_diffs):+.2f}")
        print(f"  indice de bloques 8x8: blur {np.mean(blocks_b):.3f} | sharp {np.mean(blocks_s):.3f}")

    csv_path = os.path.join(root + "_nafnet", "test", "metrics.csv")
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        p = np.array([float(r["psnr_nafnet"]) for r in rows])
        g = p - np.array([float(r["psnr_blur"]) for r in rows])
        print(f"\n=== NAFNet en test ({len(rows)} imagenes)")
        print(f"  PSNR percentiles 10/50/90: {np.percentile(p, 10):.2f} / "
              f"{np.percentile(p, 50):.2f} / {np.percentile(p, 90):.2f}")
        print(f"  ganancia sobre blur percentiles 10/50/90: {np.percentile(g, 10):+.2f} / "
              f"{np.percentile(g, 50):+.2f} / {np.percentile(g, 90):+.2f}")
        print(f"  imagenes donde NAFNet empeora: {(g < 0).sum()}")
        names = [r["name"] for r in rows]
        print(f"  ejemplo de nombres: {names[0]}, {names[len(names) // 2]}")


if __name__ == "__main__":
    main()
