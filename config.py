import os

class Config:
    DATA_PATH = r"C:\Users\LENOVO\Documents\IA2Proyecto\archive\Gopro"
    SPLIT_TRAIN = 'train'
    SPLIT_TEST = 'test'

    PATCH_SIZE = 128
    MAX_TRAIN_IMAGES = 2103         # ← subimos
    MAX_TEST_IMAGES = 10

    BATCH_SIZE = 2                 # baja a 1 si hay OOM
    NUM_WORKERS = 2
    EPOCHS = 60                    # ← subimos
    LR = 1e-6
    WEIGHT_DECAY = 1e-4
    SAVE_EVERY = 10
    CKPT_DIR = "./checkpoints"
    OUTPUT_DIR = "./results"

    # Difusión
    TIMESTEPS = 500
    BETA_START = 1e-4
    BETA_END = 0.02
    SCHEDULE = "cosine"            # ← cambio de linear a cosine

    # Modelo
    BASE_CHANNELS = 64             # ← 2x más grande
    TIME_EMB_DIM = 256             # ← 2x más grande
    PREDICT_X0 = True              # ← predicción de imagen limpia
    RESIDUAL = True                # ← x0 = blur + delta