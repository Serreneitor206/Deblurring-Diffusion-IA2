import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config import Config
from data_loader import GoProDataset
from models import UNet

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en: {device}")

    # Cargar dataset de entrenamiento
    train_dataset = GoProDataset(split='train', img_size=Config.IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    # Inicializar modelo, pérdida y optimizador
    model = UNet().to(device)
    criterion = nn.MSELoss() # Error cuadrático medio para comparar pixeles
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

    print("Iniciando entrenamiento...")
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for blur, sharp in progress_bar:
            blur = blur.to(device)
            sharp = sharp.to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(blur)
            
            # Calcular error y ajustar pesos (Backward)
            loss = criterion(outputs, sharp)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} - Loss Promedio: {epoch_loss/len(train_loader):.6f}")

        # Guardar modelo cada 10 épocas y al final
        if (epoch + 1) % 10 == 0 or epoch == Config.EPOCHS - 1:
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Modelo guardado en {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()