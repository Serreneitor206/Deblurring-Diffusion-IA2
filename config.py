import os


class Config:
    # Carpeta del dataset GoPro (debe contener train/ y test/, cada una con blur/ y sharp/).
    # Se puede cambiar sin tocar el codigo con la variable de entorno GOPRO_PATH.
    DATA_PATH = os.environ.get(
        "GOPRO_PATH", r"C:\Users\LENOVO\Documents\IA2Proyecto\archive\Gopro")
