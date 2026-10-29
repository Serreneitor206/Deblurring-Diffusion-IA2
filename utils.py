import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

def compute_psnr_ssim(pred, gt, data_range=1.0):
    psnr = peak_signal_noise_ratio(gt, pred, data_range=data_range)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=data_range)
    return psnr, ssim

def save_image(img, path):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def plot_training_curves(train_losses, val_losses, save_path):
    """Genera y guarda la gráfica de loss de train y validación."""
    import matplotlib
    matplotlib.use('Agg')          # backend sin pantalla
    import matplotlib.pyplot as plt
    import csv

    epochs = list(range(1, len(train_losses) + 1))

    # === Gráfica ===
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='o', markersize=4)

    # Marcar mejor epoch (menor val_loss)
    best_idx = int(np.argmin(val_losses))
    best_epoch = epochs[best_idx]
    best_val = val_losses[best_idx]
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
                label=f'Best Val: epoch {best_epoch} ({best_val:.4f})')
    plt.scatter([best_epoch], [best_val], color='green', s=100, zorder=5,
                edgecolor='black', linewidth=1.5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Escala logarítmica si el rango es muy grande
    if max(train_losses) / max(min(train_losses), 1e-8) > 50:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfica guardada en: {save_path}")

    # === También guardar CSV ===
    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for i, (tr, vl) in enumerate(zip(train_losses, val_losses), 1):
            writer.writerow([i, tr, vl])
    print(f"📊 Datos guardados en: {csv_path}")