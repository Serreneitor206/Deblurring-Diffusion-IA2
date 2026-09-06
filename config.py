import os

class Config:
    # Ruta local
    DATA_PATH = r"C:\Users\LENOVO\Documents\IA2Proyecto\archive\Gopro"  
    
    # Parámetros ORIGINALES del paper (Para replicación rigurosa)
    T = 30
    S = 200
    
    # Parámetros RÁPIDOS (Para la interfaz en segundos)
    FAST_T = 3
    FAST_S = 10
    
    BETA_START = 1e-4
    BETA_END = 2e-2
    LAMBDA_K = 2e-3
    MU = 1.0
    
    KERNEL_SIZE = 25  
    
    LR_IMAGE = 1e-3
    LR_KERNEL = 2.5e-4
    LR_DECAY = 0.95
    MIN_LR = 1e-5
    KERNEL_HIDDEN_DIM = 2000
    KERNEL_HIDDEN_LAYERS = 3
    
    IMG_SIZE = 256
    OUTPUT_PATH = "./results_gopro"
    
    # Rutas para guardar los pesos y no empezar desde cero en la interfaz
    UNET_SAVE_PATH = "./unet_warm_weights.pth"
    KERNEL_SAVE_PATH = "./kernel_warm_weights.pth"