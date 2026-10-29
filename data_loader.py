import os
import glob
import random
import cv2
import numpy as np                  # ← AÑADE ESTA LÍNEA
import torch
from torch.utils.data import Dataset
from config import Config

class GoProDataset(Dataset):
    """
    mode='crop' -> Recorta un parche aleatorio de PATCH_SIZE en resolución nativa.
                   Se usa en TRAIN.
    mode='full' -> Redimensiona la imagen completa a PATCH_SIZE.
                   Se usa en TEST.
    """
    def __init__(self, split='train', patch_size=128, mode='crop'):
        self.patch_size = patch_size
        self.mode = mode

        base_path = Config.DATA_PATH
        if base_path is None:
            import kagglehub
            base_path = kagglehub.dataset_download("adwythdarsanr/gopro-image-deblurring-dataset")
            base_path = os.path.join(base_path, 'gopro')

        # Detectar carpeta de split (train/train/gopro...)
        split_path = None
        for candidate in [split, split.capitalize(), split.upper()]:
            path = os.path.join(base_path, candidate)
            if os.path.isdir(path):
                split_path = path
                break
        if split_path is None:
            raise FileNotFoundError(f"No encontré '{split}' en {base_path}")

        print(f"Usando split_path: {split_path}")

        direct_blur = os.path.join(split_path, 'blur')
        direct_sharp = os.path.join(split_path, 'sharp')

        self.pairs = []
        # === Estructura PLANA: split/blur y split/sharp ===
        if os.path.isdir(direct_blur) and os.path.isdir(direct_sharp):
            blur_files = sorted(glob.glob(os.path.join(direct_blur, '*.png')))
            sharp_map = {os.path.basename(f): f
                         for f in sorted(glob.glob(os.path.join(direct_sharp, '*.png')))}
            for b in blur_files:
                name = os.path.basename(b)
                if name in sharp_map:
                    self.pairs.append((b, sharp_map[name]))
        # === Estructura CON SUBCARPETAS ===
        else:
            for folder in sorted(os.listdir(split_path)):
                bdir = os.path.join(split_path, folder, 'blur')
                sdir = os.path.join(split_path, folder, 'sharp')
                if not (os.path.isdir(bdir) and os.path.isdir(sdir)):
                    continue
                for b, s in zip(sorted(glob.glob(os.path.join(bdir, '*.png'))),
                                sorted(glob.glob(os.path.join(sdir, '*.png')))):
                    self.pairs.append((b, s))

        print(f"Cargados {len(self.pairs)} pares desde {split} (modo={mode}).")

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, blur, sharp, size):
        h, w = blur.shape[:2]
        if h < size or w < size:
            blur = cv2.resize(blur, (size, size))
            sharp = cv2.resize(sharp, (size, size))
            return blur, sharp

        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        blur = blur[y:y+size, x:x+size]
        sharp = sharp[y:y+size, x:x+size]

        # === Augmentation ===
        if random.random() < 0.5:
            blur = np.ascontiguousarray(blur[:, ::-1])
            sharp = np.ascontiguousarray(sharp[:, ::-1])
        if random.random() < 0.5:
            blur = np.ascontiguousarray(blur[::-1, :])
            sharp = np.ascontiguousarray(sharp[::-1, :])
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            blur = np.ascontiguousarray(np.rot90(blur, k))
            sharp = np.ascontiguousarray(np.rot90(sharp, k))

        return blur, sharp

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        blur = cv2.cvtColor(cv2.imread(blur_path), cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(cv2.imread(sharp_path), cv2.COLOR_BGR2RGB)

        if self.mode == 'crop':
            blur, sharp = self._random_crop(blur, sharp, self.patch_size)
        elif self.mode == 'center_crop':
            # Crop central FIJO (para validación reproducible)
            h, w = blur.shape[:2]
            cy, cx = h // 2, w // 2
            s = self.patch_size // 2
            y0, x0 = max(0, cy - s), max(0, cx - s)
            blur = blur[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
            sharp = sharp[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
            if blur.shape[0] != self.patch_size or blur.shape[1] != self.patch_size:
                blur = cv2.resize(blur, (self.patch_size, self.patch_size))
                sharp = cv2.resize(sharp, (self.patch_size, self.patch_size))
        elif self.mode == 'native':
            max_side = 640
            h, w = blur.shape[:2]
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                blur = cv2.resize(blur, (new_w, new_h))
                sharp = cv2.resize(sharp, (new_w, new_h))
        else:  # 'full'
            blur = cv2.resize(blur, (self.patch_size, self.patch_size))
            sharp = cv2.resize(sharp, (self.patch_size, self.patch_size))

        blur = torch.from_numpy(blur).permute(2, 0, 1).float() / 127.5 - 1.0
        sharp = torch.from_numpy(sharp).permute(2, 0, 1).float() / 127.5 - 1.0
        return blur, sharp