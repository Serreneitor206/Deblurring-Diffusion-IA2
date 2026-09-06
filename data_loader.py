import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from config import Config

class GoProDataset(Dataset):
    def __init__(self, split='train', img_size=256):
        self.img_size = img_size
        self.base_path = Config.DATA_PATH
        
        # Buscar la ruta del split (train o test)
        self.split_path = os.path.join(self.base_path, split)
        if not os.path.exists(self.split_path):
             self.split_path = os.path.join(self.base_path, 'gopro', split)
             
        print(f"Usando directorio: {self.split_path}")

        self.pairs = []
        blur_dir = os.path.join(self.split_path, 'blur')
        sharp_dir = os.path.join(self.split_path, 'sharp')
        
        if os.path.exists(blur_dir) and os.path.exists(sharp_dir):
            blur_files = sorted(glob.glob(os.path.join(blur_dir, '*.png')))
            sharp_files = sorted(glob.glob(os.path.join(sharp_dir, '*.png')))
            for b_path, s_path in zip(blur_files, sharp_files):
                self.pairs.append((b_path, s_path))

        print(f"Cargados {len(self.pairs)} pares de imágenes para {split}.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        
        blur = cv2.imread(blur_path)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.imread(sharp_path)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
        
        blur = cv2.resize(blur, (self.img_size, self.img_size))
        sharp = cv2.resize(sharp, (self.img_size, self.img_size))
        
        blur = torch.from_numpy(blur).permute(2,0,1).float() / 255.0
        sharp = torch.from_numpy(sharp).permute(2,0,1).float() / 255.0
        
        return blur, sharp