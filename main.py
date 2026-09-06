import torch
import os
import numpy as np
import time
from config import Config
from data_loader import GoProDataset
from deblur_sdi import DeblurSDI
from utils import compute_psnr_ssim, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(fast_mode=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Usando: {device}")
    
    # Definir parámetros según el modo
    T_val = Config.FAST_T if fast_mode else Config.T
    S_val = Config.FAST_S if fast_mode else Config.S
    print(f"Modo {'RÁPIDO (Interfaz)' if fast_mode else 'PAPER ORIGINAL'}: T={T_val}, S={S_val}")

    dataset = GoProDataset(split='test', img_size=Config.IMG_SIZE)
    dataset.pairs = dataset.pairs[:5]  # Solo 5 imágenes para probar
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    all_psnr = []
    all_ssim = []

    for idx, (blur, sharp) in enumerate(tqdm(dataloader, desc="Procesando")):
        blur, sharp = blur.to(device), sharp.to(device)
        
        # Instanciar el deblurring con los parámetros correspondientes
        deblur = DeblurSDI(blur, kernel_size=Config.KERNEL_SIZE, T=T_val, S=S_val)
        
        # Si estamos en modo rápido, intentar cargar pesos pre-calculados
        if fast_mode:
            deblur.load_weights()
            
        start_time = time.time()
        recon, kernel_est = deblur.run()
        end_time = time.time()
        
        # Si estamos en modo original lento, guardamos lo aprendido de esta imagen
        if not fast_mode and idx == 0: 
            deblur.save_weights()
            print("   [!] Pesos guardados para futuros usos rápidos.")

        recon_np = recon.squeeze(0).cpu().permute(1,2,0).numpy()
        sharp_np = sharp.squeeze(0).cpu().permute(1,2,0).numpy()
        
        psnr, ssim = compute_psnr_ssim(recon_np, sharp_np)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        
        save_image(recon_np, os.path.join(Config.OUTPUT_PATH, f"recon_{idx:04d}.png"))
        print(f"   Imagen {idx+1} lista en {end_time-start_time:.2f}s | PSNR={psnr:.2f}, SSIM={ssim:.4f}")

if __name__ == "__main__":
    # PASO 1: Ejecuta con fast_mode=False al menos una vez para replicar el paper y generar los pesos iniciales.
    # PASO 2: Para simular tu interfaz web, cambia a fast_mode=True.
    main(fast_mode=False)