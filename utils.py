import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

def compute_psnr_ssim(img_pred, img_gt, data_range=1.0):
    # img: [H, W, C] en rango [0,1]
    psnr = peak_signal_noise_ratio(img_gt, img_pred, data_range=data_range)
    ssim = structural_similarity(img_gt, img_pred, multichannel=True, channel_axis=2,
                                 data_range=data_range)
    return psnr, ssim

def save_image(img, path):
    # img: [H, W, C] en [0,1]
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)